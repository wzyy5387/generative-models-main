# -*- coding: utf-8 -*-

"""Convert vendor-neutral JSON/CSV/NPZ Ising responses to canonical NPZ."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .prepare_bosonic_submission import sha256_file
from .kaiwu_adapter import normalize_hardware_spins


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "bosonic_responses" / "imported"


class VSResponseValidationError(ValueError):
    """A validation failure with the frozen task and offending response."""

    def __init__(self, message, task, record):
        super().__init__("%s [gain=%s, task=%s]" %
                         (message, task.get("gain"), task.get("task_name")))
        self.task = task
        self.record = record


def strict_response_record_valid(record, expected=None, require_physical=True):
    """Validate both the audit record and the raw 49-spin NPZ contents."""
    if not isinstance(record, dict) or record.get("completed") is not True:
        return False
    required = ("instance_id", "stage", "task_id", "task_name", "matrix_sha256",
                "response_file_sha256", "requested_reads", "returned_reads",
                "response_path", "gain", "sdk_version", "platform_backend",
                "submitted_at", "completed_at", "hardware_metadata_complete")
    if any(not str(record.get(key, "")) for key in required):
        return False
    if require_physical:
        if record.get("hardware_claim") is not True or record.get("physical_platform_used") is not True:
            return False
    elif record.get("hardware_claim") is not False or record.get("physical_platform_used") is not False:
        return False
    if record.get("hardware_metadata_complete") is not True:
        return False
    try:
        requested = int(record["requested_reads"])
        returned = int(record["returned_reads"])
    except (TypeError, ValueError):
        return False
    if requested != returned or returned <= 0:
        return False
    response = Path(record["response_path"])
    if not response.is_file() or response.suffix.lower() != ".npz":
        return False
    try:
        if sha256_file(response) != str(record["response_file_sha256"]):
            return False
        verification = None
        verification_path = record.get("platform_verification_path")
        if verification_path:
            verification_path = Path(verification_path)
            from .kaiwu_platform_receipt import validate_platform_verification
            verified_raw_path = Path(record.get("platform_verification_source_response", response))
            if not verification_path.is_file() or sha256_file(verification_path) != record.get("platform_verification_sha256"):
                return False
            try:
                verification = validate_platform_verification(verified_raw_path, verification_path, {
                    "instance_id": record["instance_id"],
                    "task_id": record["task_id"],
                    "task_name": record["task_name"],
                    "matrix_sha256": record["matrix_sha256"],
                    "stage": record["stage"],
                    "gain": record["gain"],
                    "requested_reads": record["requested_reads"],
                    "returned_reads": record["returned_reads"],
                })
            except (OSError, KeyError, TypeError, ValueError):
                return False
            receipt_fields = (
                "platform_identifier_field", "platform_identifier_value",
            )
            if any(str(record.get(key, "")) != str(verification.get(key, ""))
                   for key in receipt_fields):
                return False
            from .kaiwu_platform_receipt import _same_instant
            for record_key, receipt_key in (
                ("submitted_at", "client_called_at"),
                ("completed_at", "client_returned_at"),
                ("client_called_at", "client_called_at"),
                ("client_returned_at", "client_returned_at"),
                ("platform_submitted_at", "platform_submitted_at"),
                ("platform_completed_at", "platform_completed_at"),
            ):
                if record.get(record_key):
                    try:
                        if not _same_instant(record[record_key], verification[receipt_key], record_key):
                            return False
                    except ValueError:
                        return False
        with np.load(response, allow_pickle=False) as archive:
            if (verification is not None
                    and response.resolve() != verified_raw_path.resolve()):
                for key in ("client_called_at", "client_returned_at", "platform_submitted_at",
                            "platform_completed_at", "platform_identifier_field",
                            "platform_identifier_value"):
                    if key not in archive:
                        return False
                    archived_value = np.asarray(archive[key]).reshape(-1)
                    if archived_value.size != 1:
                        return False
                    actual = str(archived_value[0].item())
                    if key in ("client_called_at", "client_returned_at", "platform_submitted_at", "platform_completed_at"):
                        try:
                            if not _same_instant(actual, verification[key], key):
                                return False
                        except ValueError:
                            return False
                    elif actual != str(verification[key]):
                        return False
            samples = np.asarray(archive["samples"])
            if samples.shape == (returned, 49):
                hardware_samples = samples
            elif ("hardware_samples" in archive
                  and samples.shape == (returned, int(archive["logical_n_bits"]))
                  and int(archive["hardware_n_bits"]) == 49):
                hardware_samples = np.asarray(archive["hardware_samples"])
                if hardware_samples.shape != (returned, 49):
                    return False
                expected_logical = normalize_hardware_spins(
                    hardware_samples,
                    logical_n_bits=int(archive["logical_n_bits"]),
                    auxiliary_spin_index=int(archive["auxiliary_spin_index"]),
                )
                if not np.array_equal(samples, expected_logical):
                    return False
            else:
                return False
            if not np.isin(hardware_samples, (-1, 1)).all() or not np.isin(samples, (-1, 1)).all():
                return False
            fields = {
                "instance_id": record["instance_id"],
                "task_name": record["task_name"],
                "matrix_sha256": record["matrix_sha256"],
                "stage": record["stage"],
                "hardware_gain": record["gain"],
                "requested_reads": requested,
                "returned_reads": returned,
            }
            for key, expected_value in fields.items():
                if key not in archive:
                    return False
                value = np.asarray(archive[key]).reshape(-1)
                if value.size != 1:
                    return False
                actual = value[0].item()
                if key == "hardware_gain":
                    if float(actual) != float(expected_value):
                        return False
                elif str(actual) != str(expected_value):
                    return False
            if "task_id" not in archive:
                return False
            archived_task_id = str(np.asarray(archive["task_id"]).reshape(-1)[0].item())
            if archived_task_id:
                if archived_task_id != str(record["task_id"]):
                    return False
            elif verification is None or str(verification["task_id"]) != str(record["task_id"]):
                return False
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return False
    if expected:
        for key, expected_value in expected.items():
            if key == "gain":
                try:
                    if float(record.get(key)) != float(expected_value):
                        return False
                except (TypeError, ValueError):
                    return False
            elif str(record.get(key, "")) != str(expected_value):
                return False
    if require_physical and record.get("platform_backend") != "kaiwu.cim.CIMOptimizer":
        return False
    versions = record.get("sdk_versions", {})
    if not isinstance(versions, dict) or not versions.get("importlib_metadata_version") or not versions.get("kaiwu_module_version"):
        return False
    return True


def expand_samples(samples, n_bits, zero_spin=-1, reverse=False):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[1] != n_bits:
        raise ValueError(
            "Samples must have shape (n_reads, %d); got %s"
            % (n_bits, samples.shape)
        )
    if (not np.issubdtype(samples.dtype, np.number)
            or np.issubdtype(samples.dtype, np.complexfloating)
            or not np.isfinite(samples).all()):
        raise ValueError("Samples must contain finite numeric values")
    raw_values = set(np.unique(samples).tolist())
    if not (raw_values.issubset({-1, 1}) or raw_values.issubset({0, 1})):
        raise ValueError("Samples must contain only {-1,+1} or {0,1}")
    if reverse:
        samples = samples[:, ::-1]
    samples = samples.astype(np.int8)
    values = set(np.unique(samples).tolist())
    if values.issubset({0, 1}):
        samples = np.where(samples == 0, zero_spin, -zero_spin).astype(np.int8)
    elif not values.issubset({-1, 1}):
        raise ValueError("Samples must contain only {-1,+1} or {0,1}")
    return samples


def bitstrings_to_samples(bitstrings, n_bits):
    rows = []
    for bitstring in bitstrings:
        bitstring = str(bitstring).strip().replace(" ", "")
        if len(bitstring) != n_bits or set(bitstring) - {"0", "1"}:
            raise ValueError("Invalid %d-bit response string: %s" % (n_bits, bitstring))
        rows.append([int(value) for value in bitstring])
    return np.asarray(rows, dtype=np.int8)


def apply_counts(samples, counts):
    if counts is None:
        return samples
    raw_counts = np.asarray(counts)
    if (not np.issubdtype(raw_counts.dtype, np.number)
            or np.issubdtype(raw_counts.dtype, np.complexfloating)
            or not np.isfinite(raw_counts).all()
            or not np.equal(raw_counts, np.floor(raw_counts)).all()):
        raise ValueError("counts must contain finite integer values")
    counts = raw_counts.astype(np.int64)
    if counts.ndim != 1 or counts.size != samples.shape[0] or np.any(counts <= 0):
        raise ValueError("counts must be positive integers matching sample rows")
    return np.repeat(samples, counts, axis=0)


def read_json_response(path, n_bits):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "samples" in data:
        samples = np.asarray(data["samples"])
    elif "bitstrings" in data:
        samples = bitstrings_to_samples(data["bitstrings"], n_bits)
    else:
        raise ValueError("%s has neither samples nor bitstrings" % path)
    samples = apply_counts(samples, data.get("counts"))
    metadata = {
        key: data[key]
        for key in (
            "instance_id",
            "task_id",
            "task_name",
            "matrix_sha256",
            "hardware_matrix_sha256",
            "bit_order",
            "job_id",
            "latency_s",
            "applied_scale",
            "backend",
            "platform_metadata",
            "sdk_version",
            "kaiwu_distribution_version",
            "kaiwu_module_version",
            "platform_backend",
            "submitted_at",
            "completed_at",
            "requested_reads",
            "returned_reads",
            "hardware_claim",
            "physical_platform_used",
            "checkpoint_dir",
        )
        if key in data
    }
    return samples, metadata


def read_csv_response(path, n_bits):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("%s is empty" % path)
    fields = list(rows[0])
    if "bitstring" in fields:
        samples = bitstrings_to_samples([row["bitstring"] for row in rows], n_bits)
    else:
        spin_fields = []
        for index in range(n_bits):
            candidates = ("s%d" % index, "spin_%d" % index, "q%d" % index)
            field = next((name for name in candidates if name in fields), None)
            if field is None:
                raise ValueError(
                    "%s requires bitstring or indexed spin columns" % path
                )
            spin_fields.append(field)
        samples = np.asarray(
            [[int(row[field]) for field in spin_fields] for row in rows]
        )
    counts = [int(row["count"]) for row in rows] if "count" in fields else None
    samples = apply_counts(samples, counts)
    metadata = {
        key: rows[0][key]
        for key in (
            "instance_id",
            "task_id",
            "task_name",
            "matrix_sha256",
            "hardware_matrix_sha256",
            "bit_order",
            "job_id",
            "backend",
            "sdk_version",
            "kaiwu_distribution_version",
            "kaiwu_module_version",
            "platform_backend",
            "submitted_at",
            "completed_at",
            "requested_reads",
            "returned_reads",
            "hardware_claim",
            "physical_platform_used",
            "checkpoint_dir",
        )
        if key in fields and rows[0][key] != ""
    }
    for key in ("latency_s", "applied_scale"):
        if key in fields and rows[0][key] != "":
            metadata[key] = float(rows[0][key])
    return samples, metadata


def read_npz_response(path, n_bits):
    with np.load(path, allow_pickle=False) as response:
        if "samples" not in response:
            raise ValueError("%s does not contain samples" % path)
        samples = np.asarray(response["samples"])
        metadata = {}
        for key in (
            "instance_id",
            "task_id",
            "task_name",
            "matrix_sha256",
            "hardware_matrix_sha256",
            "bit_order",
            "job_id",
            "latency_s",
            "applied_scale",
            "backend",
            "sdk_version",
            "kaiwu_distribution_version",
            "kaiwu_module_version",
            "platform_backend",
            "submitted_at",
            "completed_at",
            "requested_reads",
            "returned_reads",
            "hardware_claim",
            "physical_platform_used",
            "checkpoint_dir",
        ):
            if key in response:
                value = np.asarray(response[key]).reshape(-1)
                if value.size != 1:
                    raise ValueError("%s must be scalar in %s" % (key, path))
                metadata[key] = value[0].item()
    return samples, metadata


def locate_response(input_dir, instance_id):
    matches = [
        input_dir / (instance_id + suffix)
        for suffix in (".json", ".csv", ".npz")
        if (input_dir / (instance_id + suffix)).is_file()
    ]
    if len(matches) > 1:
        raise ValueError("Multiple response files found for %s" % instance_id)
    return matches[0] if matches else None


def verify_package(submission_manifest):
    submission_manifest = Path(submission_manifest).resolve()
    package_dir = submission_manifest.parent
    submission = json.loads(submission_manifest.read_text(encoding="utf-8"))
    for instance in submission["instances"]:
        payload_path = package_dir / instance["payload"]
        if sha256_file(payload_path) != instance["payload_sha256"]:
            raise ValueError("Payload hash mismatch for %s" % instance["instance_id"])
        matrix_path = instance.get("hardware_matrix")
        if matrix_path is not None:
            matrix_path = package_dir / matrix_path
            if sha256_file(matrix_path) != instance["hardware_matrix_sha256_file"]:
                raise ValueError(
                    "Hardware matrix file hash mismatch for %s"
                    % instance["instance_id"]
                )
    return submission


def import_responses(
    submission_manifest,
    input_dir,
    output_dir,
    zero_spin=-1,
    reverse=False,
    allow_missing=False,
):
    submission = verify_package(submission_manifest)
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "response_import_audit.json"
    if audit_path.exists():
        raise RuntimeError("Existing response import audit; refusing to overwrite: %s" % audit_path)
    existing_canonical = [
        output_dir / (str(item["instance_id"]) + ".npz")
        for item in submission["instances"]
        if (output_dir / (str(item["instance_id"]) + ".npz")).exists()
    ]
    if existing_canonical:
        raise RuntimeError("Existing canonical response; refusing to overwrite: %s" % existing_canonical[0])
    records = []
    missing = []
    for instance in submission["instances"]:
        instance_id = instance["instance_id"]
        hardware_n_bits = int(instance.get("hardware_n_bits", instance["n_bits"]))
        logical_n_bits = int(instance.get("logical_n_bits", hardware_n_bits))
        is_hardware_response = "hardware_matrix" in instance
        response_path = locate_response(input_dir, instance_id)
        if response_path is None:
            missing.append(instance_id)
            continue
        if response_path.suffix == ".json":
            samples, metadata = read_json_response(
                response_path, hardware_n_bits
            )
        elif response_path.suffix == ".csv":
            samples, metadata = read_csv_response(
                response_path, hardware_n_bits
            )
        else:
            samples, metadata = read_npz_response(
                response_path, hardware_n_bits
            )
        if metadata.get("instance_id", instance_id) != instance_id:
            raise ValueError("Response instance_id mismatch for %s" % instance_id)
        samples = expand_samples(
            samples,
            hardware_n_bits,
            zero_spin=zero_spin,
            reverse=reverse,
        )
        if is_hardware_response:
            task_id = str(metadata.get("task_id", metadata.get("job_id", "")))
            if not task_id:
                from .kaiwu_platform_receipt import validate_platform_verification
                verification_path = input_dir / (instance_id + ".platform_verification.json")
                if not verification_path.is_file():
                    raise ValueError(
                        "Kaiwu response %s is waiting for a separate verified platform task receipt; task_name is not a task ID"
                        % instance_id
                    )
                expected_name = json.loads(
                    (Path(submission_manifest).resolve().parent / instance["payload"]).read_text(encoding="utf-8")
                ).get("task_name")
                receipt = validate_platform_verification(response_path, verification_path, {
                    "instance_id": instance_id,
                    "task_name": expected_name,
                    "matrix_sha256": instance.get("hardware_matrix_sha256"),
                    "stage": submission["stage"],
                    "gain": instance.get("hardware_gain"),
                    "requested_reads": submission["requested_reads_per_instance"],
                    "returned_reads": samples.shape[0],
                })
                task_id = str(receipt["task_id"])
                metadata.update({
                    "task_id": task_id,
                    "task_name": receipt["task_name"],
                    "matrix_sha256": receipt["matrix_sha256"],
                    "sdk_version": receipt["sdk_version"],
                    "kaiwu_distribution_version": receipt["sdk_versions"]["importlib_metadata_version"],
                    "kaiwu_module_version": receipt["sdk_versions"]["kaiwu_module_version"],
                    "platform_backend": receipt["platform_backend"],
                    "submitted_at": receipt["submitted_at"],
                    "completed_at": receipt["completed_at"],
                    "client_called_at": receipt["client_called_at"],
                    "client_returned_at": receipt["client_returned_at"],
                    "platform_submitted_at": receipt["platform_submitted_at"],
                    "platform_completed_at": receipt["platform_completed_at"],
                    "platform_identifier_field": receipt["platform_identifier_field"],
                    "platform_identifier_value": receipt["platform_identifier_value"],
                    "requested_reads": receipt["requested_reads"],
                    "returned_reads": receipt["returned_reads"],
                    "platform_verification_path": str(verification_path.resolve()),
                    "platform_verification_sha256": sha256_file(verification_path),
                    "platform_verification_source_response": str(response_path.resolve()),
                })
            expected_task_name = None
            payload = json.loads(
                (Path(submission_manifest).resolve().parent / instance["payload"]).read_text(
                    encoding="utf-8"
                )
            )
            expected_task_name = payload.get("task_name")
            if metadata.get("task_name") and metadata["task_name"] != expected_task_name:
                raise ValueError("Response task_name mismatch for %s" % instance_id)
            response_matrix_hash = metadata.get(
                "matrix_sha256", metadata.get("hardware_matrix_sha256", "")
            )
            if response_matrix_hash != instance.get("hardware_matrix_sha256"):
                raise ValueError("Response matrix SHA-256 mismatch for %s" % instance_id)
            if metadata.get("bit_order") and metadata["bit_order"] != (
                "index_descending" if reverse else "index_ascending"
            ):
                raise ValueError("Response bit order mismatch for %s" % instance_id)
            logical_samples = normalize_hardware_spins(
                samples,
                logical_n_bits=logical_n_bits,
                auxiliary_spin_index=int(instance["auxiliary_spin_index"]),
            )
            matrix_hash = instance.get("hardware_matrix_sha256")
        else:
            logical_samples = samples
            matrix_hash = None
        output_path = output_dir / (instance_id + ".npz")
        response_arrays = {
            "samples": logical_samples,
            "instance_id": str(instance_id),
            "task_name": str(metadata.get("task_name", "")),
            "matrix_sha256": str(matrix_hash or ""),
            "stage": str(submission["stage"]),
            "hardware_gain": float(instance.get("hardware_gain", -1.0) or -1.0),
            "bit_order": "index_descending" if reverse else "index_ascending",
            "latency_s": float(metadata.get("latency_s", np.nan)),
            "applied_scale": float(metadata.get("applied_scale", np.nan)),
            "backend": str(metadata.get("backend", "bosonic_platform")),
            "job_id": str(metadata.get("job_id", metadata.get("task_id", ""))),
            "task_id": str(metadata.get("task_id", metadata.get("job_id", ""))),
            "raw_response_sha256": sha256_file(response_path),
            "sdk_version": str(metadata.get("sdk_version", "")),
            "platform_backend": str(metadata.get("platform_backend", "")),
            "submitted_at": str(metadata.get("submitted_at", "")),
            "completed_at": str(metadata.get("completed_at", "")),
            "client_called_at": str(metadata.get("client_called_at", metadata.get("submitted_at", ""))),
            "client_returned_at": str(metadata.get("client_returned_at", metadata.get("completed_at", ""))),
            "platform_submitted_at": str(metadata.get("platform_submitted_at", "")),
            "platform_completed_at": str(metadata.get("platform_completed_at", "")),
            "platform_identifier_field": str(metadata.get("platform_identifier_field", "")),
            "platform_identifier_value": str(metadata.get("platform_identifier_value", "")),
            "requested_reads": int(metadata.get(
                "requested_reads", submission["requested_reads_per_instance"]
            )),
            "returned_reads": int(samples.shape[0]),
            "checkpoint_dir": str(metadata.get("checkpoint_dir", "")),
        }
        platform_complete = bool(
            is_hardware_response and task_id and metadata.get("task_name") and
            metadata.get("sdk_version") and metadata.get("platform_backend") and
            str(metadata.get("platform_backend")) == "kaiwu.cim.CIMOptimizer" and
            metadata.get("kaiwu_distribution_version") and
            metadata.get("kaiwu_module_version") and
            metadata.get("submitted_at") and metadata.get("completed_at") and
            response_matrix_hash and sha256_file(response_path) and
            int(metadata.get("requested_reads", submission["requested_reads_per_instance"])) == submission["requested_reads_per_instance"] and
            int(metadata.get("returned_reads", samples.shape[0])) == samples.shape[0] and
            samples.shape[0] == submission["requested_reads_per_instance"] and
            True
        )
        response_arrays["hardware_claim"] = bool(platform_complete)
        response_arrays["physical_platform_used"] = bool(platform_complete)
        if is_hardware_response:
            response_arrays.update(
                {
                    "hardware_samples": samples,
                    "logical_n_bits": np.asarray(logical_n_bits),
                    "hardware_n_bits": np.asarray(hardware_n_bits),
                    "auxiliary_spin_index": np.asarray(instance["auxiliary_spin_index"]),
                    "hardware_matrix_sha256": np.asarray(matrix_hash),
                    "hardware_gain": np.asarray(instance["hardware_gain"]),
                }
            )
        np.savez_compressed(output_path, **response_arrays)
        canonical_hash = sha256_file(output_path)
        records.append(
            {
                "instance_id": instance_id,
                "stage": submission["stage"],
                "gain": instance.get("hardware_gain"),
                "n_reads": int(samples.shape[0]),
                "expected_reads": submission["requested_reads_per_instance"],
                "read_count_matches": bool(
                    samples.shape[0] == submission["requested_reads_per_instance"]
                ),
                "raw_response": str(response_path),
                "raw_response_sha256": sha256_file(response_path),
                "canonical_response": str(output_path),
                "hardware_n_bits": hardware_n_bits,
                "logical_n_bits": logical_n_bits,
                "task_id": str(
                    metadata.get(
                        "task_id",
                        metadata.get("job_id", ""),
                    )
                ),
                "task_name": str(metadata.get("task_name", "")),
                "matrix_sha256": matrix_hash,
                "hardware_matrix_sha256": matrix_hash,
                "sdk_version": str(metadata.get("sdk_version", "")),
                "sdk_versions": {
                    "importlib_metadata_version": str(metadata.get("kaiwu_distribution_version", "")),
                    "kaiwu_module_version": str(metadata.get("kaiwu_module_version", "")),
                },
                "platform_backend": str(metadata.get("platform_backend", "")),
                "client_called_at": str(metadata.get("client_called_at", metadata.get("submitted_at", ""))),
                "client_returned_at": str(metadata.get("client_returned_at", metadata.get("completed_at", ""))),
                "platform_submitted_at": str(metadata.get("platform_submitted_at", "")),
                "platform_completed_at": str(metadata.get("platform_completed_at", "")),
                "platform_identifier_field": str(metadata.get("platform_identifier_field", "")),
                "platform_identifier_value": str(metadata.get("platform_identifier_value", "")),
                "submitted_at": str(metadata.get("submitted_at", "")),
                "completed_at": str(metadata.get("completed_at", "")),
                "response_path": str(output_path.resolve()),
                "response_file_sha256": canonical_hash,
                "platform_verification_path": metadata.get("platform_verification_path", ""),
                "platform_verification_sha256": metadata.get("platform_verification_sha256", ""),
                "platform_verification_source_response": metadata.get("platform_verification_source_response", ""),
                "checkpoint_dir": str(metadata.get("checkpoint_dir", "")),
                "requested_reads": int(metadata.get(
                    "requested_reads", submission["requested_reads_per_instance"]
                )),
                "returned_reads": int(samples.shape[0]),
                "completed": bool(platform_complete),
                "hardware_claim": bool(platform_complete),
                "physical_platform_used": bool(platform_complete),
                "hardware_metadata_complete": bool(platform_complete),
            }
        )
    audit = {
        "method_name": submission.get("method_name", "FA-BM-VAE"),
        "model_role": submission.get(
            "model_role", "forecast_anchor_conditional_bm_vae"
        ),
        "training_backend": submission.get("training_backend", "classical"),
        "sampling_stage": submission.get(
            "sampling_stage", "post_training_conditional_ising_latent"
        ),
        "hardware_substitution": submission.get(
            "hardware_substitution",
            "platform replaces only frozen conditional Ising latent sampling",
        ),
        "hardware_claim": bool(records) and all(
            record.get("hardware_claim", False) for record in records
        ),
        "physical_platform_used": bool(records) and all(
            record.get("physical_platform_used", False) for record in records
        ),
        "submission_stage": submission["stage"],
        "submission_manifest": str(Path(submission_manifest).resolve()),
        "n_expected": len(submission["instances"]),
        "n_imported": len(records),
        "missing_instance_ids": missing,
        "zero_spin": int(zero_spin),
        "bit_order": "index_descending" if reverse else "index_ascending",
        "records": records,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    if missing and not allow_missing:
        raise FileNotFoundError(
            "Missing %d platform responses; see %s" % (len(missing), audit_path)
        )
    return audit_path, audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zero-spin", type=int, choices=[-1, 1], default=-1)
    parser.add_argument(
        "--bit-order",
        choices=["index-ascending", "index-descending"],
        default="index-ascending",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    audit_path, audit = import_responses(
        args.submission_manifest,
        args.input_dir,
        args.output_dir,
        zero_spin=args.zero_spin,
        reverse=args.bit_order == "index-descending",
        allow_missing=args.allow_missing,
    )
    print(
        "Imported %d/%d responses; audit: %s"
        % (audit["n_imported"], audit["n_expected"], audit_path)
    )


if __name__ == "__main__":
    main()
