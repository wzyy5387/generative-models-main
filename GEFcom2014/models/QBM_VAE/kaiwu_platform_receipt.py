"""Manual, offline binding of a Kaiwu response to a platform task record.

The installed public SDK returns samples but does not document a task-ID
attribute. This module never queries the platform. It binds an already saved
raw response to an operator-exported platform record without editing the raw
response or the original per-task audit.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .kaiwu_adapter import matrix_sha256, validate_raw_hardware_spins


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(archive, key):
    value = np.asarray(archive[key]).reshape(-1)
    if value.size != 1:
        raise ValueError("%s must be scalar" % key)
    return value[0].item()


def _parse_aware_timestamp(value, field):
    """Parse an ISO timestamp with an explicit UTC offset."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty timezone-aware ISO timestamp" % field)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("%s is not a valid ISO timestamp" % field) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("%s must include a timezone offset" % field)
    return parsed.astimezone(timezone.utc)


def _same_instant(left, right, field):
    return _parse_aware_timestamp(left, field) == _parse_aware_timestamp(right, field)


def create_platform_verification(raw_response, matrix_file, platform_record,
                                 evidence_file, verified_by, output_path):
    """Validate an operator-reviewed task record and save a separate receipt.

    ``platform_record`` must be an exported/recorded platform task entry with
    an actual ``task_id`` or ``job_id`` field. The evidence itself is hashed;
    the verifier and time are recorded as an explicit human attestation.
    """
    raw_response = Path(raw_response).resolve()
    matrix_file = Path(matrix_file).resolve()
    platform_record = Path(platform_record).resolve()
    evidence_file = Path(evidence_file).resolve()
    output_path = Path(output_path)
    if not verified_by or not str(verified_by).strip():
        raise ValueError("verified_by must identify the human reviewer")
    if not evidence_file.is_file():
        raise FileNotFoundError("Platform evidence file is required")
    record = json.loads(platform_record.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Platform task record must be a JSON object")
    if (record.get("task_id") and record.get("job_id")
            and str(record["task_id"]).strip() != str(record["job_id"]).strip()):
        raise ValueError("Platform task record task_id and job_id disagree")
    platform_id_key = "task_id" if record.get("task_id") else "job_id"
    platform_id = str(record.get(platform_id_key, "")).strip()
    if not platform_id:
        raise ValueError("Platform task record must contain a real task_id or job_id")

    with np.load(raw_response, allow_pickle=False) as archive:
        required = ("samples", "instance_id", "task_name", "matrix_sha256",
                    "requested_reads", "returned_reads", "stage", "hardware_gain")
        if any(key not in archive for key in required):
            raise ValueError("Raw response lacks required task identity fields")
        samples = validate_raw_hardware_spins(
            archive["samples"], int(_scalar(archive, "requested_reads")), 49)
        raw_identity = {key: _scalar(archive, key) for key in required if key != "samples"}
    with np.load(matrix_file, allow_pickle=False) as archive:
        if "hardware_matrix" not in archive:
            raise ValueError("Matrix file lacks hardware_matrix")
        matrix = np.asarray(archive["hardware_matrix"])
    calculated_matrix_hash = matrix_sha256(matrix)
    if (matrix.shape != (49, 49)
            or calculated_matrix_hash != str(raw_identity["matrix_sha256"])):
        raise ValueError("Raw response and frozen hardware matrix identity differ")

    expected = {
        "task_name": str(raw_identity["task_name"]),
        "matrix_sha256": calculated_matrix_hash,
        "status": "completed",
        "requested_reads": int(raw_identity["requested_reads"]),
        "returned_reads": int(samples.shape[0]),
    }
    for key, value in expected.items():
        if key not in record or str(record[key]) != str(value):
            raise ValueError("Platform record %s does not match saved response" % key)
    client_called_at = str(_scalar_from_npz(raw_response, "submitted_at"))
    client_returned_at = str(_scalar_from_npz(raw_response, "completed_at"))
    platform_submitted_at = record.get("submitted_at")
    platform_completed_at = record.get("completed_at")
    client_called_utc = _parse_aware_timestamp(client_called_at, "client_called_at")
    client_returned_utc = _parse_aware_timestamp(client_returned_at, "client_returned_at")
    platform_submitted_utc = _parse_aware_timestamp(platform_submitted_at, "platform_submitted_at")
    platform_completed_utc = _parse_aware_timestamp(platform_completed_at, "platform_completed_at")
    if client_returned_utc < client_called_utc:
        raise ValueError("Client call return time precedes client call time")
    if platform_completed_utc < platform_submitted_utc:
        raise ValueError("Platform completion precedes platform submission")
    if not record.get("platform_backend"):
        raise ValueError("Platform record must include completion time and backend")
    receipt = {
        "schema_version": 1,
        "verification_status": "human_verified_platform_record",
        "instance_id": str(raw_identity["instance_id"]),
        "task_id": platform_id,
        "platform_identifier_field": platform_id_key,
        "task_name": str(raw_identity["task_name"]),
        "matrix_sha256": calculated_matrix_hash,
        "matrix_file_path": str(matrix_file),
        "matrix_file_sha256": sha256_file(matrix_file),
        "raw_response_path": str(raw_response),
        "raw_response_sha256": sha256_file(raw_response),
        "stage": str(raw_identity["stage"]),
        "gain": float(raw_identity["hardware_gain"]),
        "requested_reads": int(raw_identity["requested_reads"]),
        "returned_reads": int(samples.shape[0]),
        # Keep client-call and platform-event timestamps semantically separate.
        # submitted_at/completed_at remain legacy aliases for the client times.
        "submitted_at": client_called_at,
        "completed_at": client_returned_at,
        "client_called_at": client_called_at,
        "client_returned_at": client_returned_at,
        "platform_submitted_at": str(platform_submitted_at),
        "platform_completed_at": str(platform_completed_at),
        "platform_identifier_value": platform_id,
        "platform_time_semantics": "client timestamps are local SDK call/return events; platform timestamps are platform submit/complete events; cross-system event equality is not assumed pending platform confirmation",
        "platform_backend": str(record["platform_backend"]),
        "sdk_version": str(_scalar_from_npz(raw_response, "sdk_version")),
        "sdk_versions": {
            "importlib_metadata_version": str(_scalar_from_npz(raw_response, "kaiwu_distribution_version")),
            "kaiwu_module_version": str(_scalar_from_npz(raw_response, "kaiwu_module_version")),
        },
        "platform_record_path": str(platform_record),
        "platform_record_sha256": sha256_file(platform_record),
        "evidence_path": str(evidence_file),
        "evidence_sha256": sha256_file(evidence_file),
        "verified_by": str(verified_by).strip(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return output_path, receipt


def _scalar_from_npz(path, key):
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise ValueError("Raw response lacks %s" % key)
        return _scalar(archive, key)


def validate_platform_verification(raw_response, verification_path, expected=None):
    """Check a separate human-verified receipt against the immutable response."""
    raw_response = Path(raw_response).resolve()
    verification_path = Path(verification_path).resolve()
    receipt = json.loads(verification_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValueError("Unsupported platform verification receipt")
    if receipt.get("verification_status") != "human_verified_platform_record":
        raise ValueError("Platform verification receipt is not approved")
    if sha256_file(raw_response) != receipt.get("raw_response_sha256"):
        raise ValueError("Raw response changed after platform verification")
    with np.load(raw_response, allow_pickle=False) as archive:
        checks = {
            "instance_id": str(_scalar(archive, "instance_id")),
            "task_name": str(_scalar(archive, "task_name")),
            "matrix_sha256": str(_scalar(archive, "matrix_sha256")),
            "stage": str(_scalar(archive, "stage")),
            "gain": float(_scalar(archive, "hardware_gain")),
            "requested_reads": int(_scalar(archive, "requested_reads")),
            "returned_reads": int(_scalar(archive, "returned_reads")),
        "client_called_at": str(_scalar(archive, "submitted_at")),
        "client_returned_at": str(_scalar(archive, "completed_at")),
        }
    for key, value in checks.items():
        if key in ("client_called_at", "client_returned_at"):
            if not _same_instant(receipt.get(key), value, key):
                raise ValueError("Platform verification %s mismatch" % key)
        elif str(receipt.get(key)) != str(value):
            raise ValueError("Platform verification %s mismatch" % key)
    identifier_field = receipt.get("platform_identifier_field")
    identifier_value = receipt.get("platform_identifier_value")
    if (not receipt.get("task_id") or identifier_field not in ("task_id", "job_id")
            or not identifier_value):
        raise ValueError("Platform verification lacks a real platform identifier")
    if (not receipt.get("verified_by") or not receipt.get("verified_at")
            or not receipt.get("platform_record_sha256") or not receipt.get("evidence_sha256")):
        raise ValueError("Platform verification lacks reviewer/evidence provenance")
    for path_key, hash_key in (("platform_record_path", "platform_record_sha256"),
                               ("evidence_path", "evidence_sha256"),
                               ("matrix_file_path", "matrix_file_sha256")):
        path_value = receipt.get(path_key)
        if not path_value or not Path(path_value).is_file():
            raise ValueError("Platform verification source file is missing: %s" % path_key)
        if sha256_file(path_value) != receipt.get(hash_key):
            raise ValueError("Platform verification source hash changed: %s" % path_key)
    platform_record = json.loads(Path(receipt["platform_record_path"]).read_text(encoding="utf-8"))
    platform_id = platform_record.get(identifier_field)
    if str(platform_id) != str(identifier_value) or str(identifier_value) != str(receipt["task_id"]):
        raise ValueError("Platform verification identifier differs from the original task record")
    platform_fields = {
        "task_name": receipt["task_name"],
        "matrix_sha256": receipt["matrix_sha256"],
        "status": "completed",
        "requested_reads": receipt["requested_reads"],
        "returned_reads": receipt["returned_reads"],
        "platform_backend": receipt["platform_backend"],
    }
    for key, value in platform_fields.items():
        if str(platform_record.get(key)) != str(value):
            raise ValueError("Platform verification differs from source task record: %s" % key)
    for receipt_key, record_key in (("platform_submitted_at", "submitted_at"),
                                    ("platform_completed_at", "completed_at")):
        if not _same_instant(receipt.get(receipt_key), platform_record.get(record_key), receipt_key):
            raise ValueError("Platform verification differs from source task record: %s" % record_key)
    raw_call_time = str(_scalar_from_npz(raw_response, "submitted_at"))
    raw_return_time = str(_scalar_from_npz(raw_response, "completed_at"))
    if not _same_instant(receipt.get("client_called_at"), raw_call_time, "client_called_at"):
        raise ValueError("Platform verification differs from raw client call time")
    if not _same_instant(receipt.get("client_returned_at"), raw_return_time, "client_returned_at"):
        raise ValueError("Platform verification differs from raw client return time")
    with np.load(receipt["matrix_file_path"], allow_pickle=False) as archive:
        if "hardware_matrix" not in archive or matrix_sha256(archive["hardware_matrix"]) != receipt["matrix_sha256"]:
            raise ValueError("Platform verification hardware matrix identity mismatch")
    if expected:
        for key, value in expected.items():
            if key == "gain":
                if float(receipt.get(key, float("nan"))) != float(value):
                    raise ValueError("Platform verification %s mismatch" % key)
            elif str(receipt.get(key, "")) != str(value):
                raise ValueError("Platform verification %s mismatch" % key)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-response", type=Path, required=True)
    parser.add_argument("--matrix-file", type=Path, required=True)
    parser.add_argument("--platform-record", type=Path, required=True,
                        help="Operator-exported platform record; never SDK queried")
    parser.add_argument("--evidence-file", type=Path, required=True,
                        help="Saved platform evidence reviewed by the operator")
    parser.add_argument("--verified-by", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path, receipt = create_platform_verification(
        args.raw_response, args.matrix_file, args.platform_record,
        args.evidence_file, args.verified_by, args.output)
    print(json.dumps({"verification": str(path.resolve()),
                      "instance_id": receipt["instance_id"],
                      "task_id": receipt["task_id"],
                      "raw_response_sha256": receipt["raw_response_sha256"]}))


if __name__ == "__main__":
    main()
