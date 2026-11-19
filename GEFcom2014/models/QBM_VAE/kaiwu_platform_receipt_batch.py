"""Batch-create offline platform receipts for already saved frozen VS responses."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .kaiwu_adapter import ENERGY_CONVENTION, matrix_sha256, platform_task_name
from .kaiwu_13k_protocol import load_protocol
from .kaiwu_platform_receipt import create_platform_verification, validate_platform_verification
from .prepare_bosonic_submission import sha256_file
from .run_kaiwu_13k_sampling import (
    _preparation_identity,
    _validate_preparation_identity,
    planned_tasks,
)


def _check_saved_response(task, response_path):
    with np.load(response_path, allow_pickle=False) as archive:
        expected = {
            "instance_id": str(task["instance_id"]),
            "task_name": str(task["task_name"]),
            "matrix_sha256": str(task["matrix_sha256"]),
            "hardware_energy_convention": ENERGY_CONVENTION,
            "stage": "vs",
        }
        for key, value in expected.items():
            if key not in archive or str(np.asarray(archive[key]).reshape(-1)[0].item()) != value:
                raise ValueError("Saved response does not match frozen task field %s" % key)
        if float(np.asarray(archive["hardware_gain"]).reshape(-1)[0]) != float(task["gain"]):
            raise ValueError("Saved response gain does not match frozen task")
        samples = np.asarray(archive["samples"])
        requested = int(np.asarray(archive["requested_reads"]).reshape(-1)[0])
        returned = int(np.asarray(archive["returned_reads"]).reshape(-1)[0])
        if (samples.shape != (int(task["reads"]), 49)
                or requested != int(task["reads"])
                or returned != int(task["reads"])
                or not np.isin(samples, (-1, 1)).all()):
            raise ValueError("Saved response reads/shape/spins do not match frozen task")


def create_vs_platform_receipts(manifest_path, protocol_path, preparation,
                                verified_by="wsy"):
    """Preflight every platform record and screenshot before creating any receipt.

    Manifest entries contain only ``instance_id``, ``gain``, ``platform_record``
    and ``evidence_file``. File paths are relative to the manifest directory.
    The manifest must explicitly attest that a human reviewed the evidence.
    """
    manifest_path = Path(manifest_path).resolve()
    preparation = Path(preparation).resolve()
    protocol_path = Path(protocol_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("human_review_attested") is not True
            or verified_by != "wsy"
            or not isinstance(manifest.get("items"), list)
            or not manifest["items"]):
        raise ValueError("Batch manifest must be reviewed by wsy and contain schema_version=1 items")

    protocol = load_protocol(protocol_path)
    protocol["_path"] = str(protocol_path)
    _validate_preparation_identity(preparation, protocol)
    identity = _preparation_identity(preparation, protocol)
    if (manifest.get("protocol_sha256") != identity["protocol_sha256"]
            or manifest.get("preparation_manifest_sha256") != identity["preparation_manifest_sha256"]):
        raise ValueError("Batch receipt manifest is not bound to the frozen protocol/preparation")
    tasks = planned_tasks("vs", preparation, preparation / "real_responses", protocol)
    task_map = {(str(task["instance_id"]), float(task["gain"])): task for task in tasks}
    seen_conditions = set()
    seen_task_ids = set()
    prepared = []
    manifest_root = manifest_path.parent

    with tempfile.TemporaryDirectory(prefix="kaiwu-receipt-preflight-") as temp_dir:
        temp_dir = Path(temp_dir)
        for index, item in enumerate(manifest["items"]):
            try:
                key = (str(item["instance_id"]), float(item["gain"]))
                platform_record_path = (manifest_root / item["platform_record"]).resolve()
                evidence_path = (manifest_root / item["evidence_file"]).resolve()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Batch receipt item %d has invalid identity/paths" % index) from exc
            if key in seen_conditions or key not in task_map:
                raise ValueError("Batch receipt has duplicate or non-frozen VS condition: %s" % (key,))
            seen_conditions.add(key)
            task = task_map[key]
            response_path = (preparation / "real_responses" / ("gain_%d" % int(task["gain"]))
                             / (str(task["instance_id"]) + ".npz"))
            receipt_path = response_path.with_name(str(task["instance_id"]) + ".platform_verification.json")
            if not response_path.is_file() or not platform_record_path.is_file() or not evidence_path.is_file():
                raise FileNotFoundError("Response, platform record or reviewed evidence is missing for %s" % (key,))
            with np.load(task["matrix"], allow_pickle=False) as archive:
                if matrix_sha256(archive["hardware_matrix"]) != task["matrix_sha256"]:
                    raise ValueError("Frozen matrix contents do not match the VS task")
            if sha256_file(task["matrix"]) != task["hardware_matrix_file_sha256"]:
                raise ValueError("Frozen matrix file hash does not match the VS task")
            _check_saved_response(task, response_path)
            if receipt_path.is_file():
                receipt = validate_platform_verification(response_path, receipt_path, {
                    "instance_id": task["instance_id"], "task_name": task["task_name"],
                    "matrix_sha256": task["matrix_sha256"], "stage": "vs",
                    "gain": task["gain"], "requested_reads": task["reads"],
                    "returned_reads": task["reads"],
                })
                if receipt.get("verified_by") != verified_by:
                    raise ValueError("Existing receipt reviewer does not match the configured reviewer")
                if (receipt.get("platform_record_sha256") != sha256_file(platform_record_path)
                        or receipt.get("evidence_sha256") != sha256_file(evidence_path)):
                    raise ValueError("Batch inputs conflict with the sources bound by the existing receipt")
                seen_task_ids.add(str(receipt["task_id"]))
                prepared.append({"task": task, "receipt_path": receipt_path, "existing": True})
                continue
            temp_receipt = temp_dir / ("%04d.json" % index)
            _, receipt = create_platform_verification(
                response_path, task["matrix"], platform_record_path, evidence_path,
                verified_by, temp_receipt)
            if receipt["task_name"] != task["task_name"] or receipt["matrix_sha256"] != task["matrix_sha256"]:
                raise ValueError("Platform record differs from frozen task name/matrix")
            if receipt["platform_task_name"] != platform_task_name(task["task_name"]):
                raise ValueError("Platform task short name differs from frozen task")
            if receipt["task_id"] in seen_task_ids:
                raise ValueError("Duplicate platform task ID in batch receipt manifest")
            seen_task_ids.add(receipt["task_id"])
            prepared.append({
                "task": task, "response_path": response_path, "matrix_path": task["matrix"],
                "platform_record_path": platform_record_path, "evidence_path": evidence_path,
                "receipt_path": receipt_path, "existing": False,
            })

        results = []
        for row in prepared:
            if not row["existing"]:
                create_platform_verification(
                    row["response_path"], row["matrix_path"], row["platform_record_path"],
                    row["evidence_path"], verified_by, row["receipt_path"])
            task = row["task"]
            response_path = row.get("response_path") or (
                preparation / "real_responses" / ("gain_%d" % int(task["gain"]))
                / (str(task["instance_id"]) + ".npz"))
            receipt = validate_platform_verification(response_path, row["receipt_path"], {
                    "instance_id": task["instance_id"], "task_name": task["task_name"],
                    "matrix_sha256": task["matrix_sha256"], "stage": "vs",
                    "gain": task["gain"], "requested_reads": task["reads"],
                    "returned_reads": task["reads"],
                })
            results.append({
                "instance_id": task["instance_id"], "gain": task["gain"],
                "task_id": receipt["task_id"], "receipt_path": str(row["receipt_path"]),
                "receipt_sha256": sha256_file(row["receipt_path"]),
                "status": "already_verified" if row["existing"] else "created",
            })
    return {
        "verified_by": verified_by,
        "protocol_sha256": identity["protocol_sha256"],
        "preparation_manifest_sha256": identity["preparation_manifest_sha256"],
        "n_items": len(results),
        "items": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON manifest with human_review_attested=true and record/evidence paths")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--verified-by", choices=["wsy"], default="wsy")
    args = parser.parse_args()
    print(json.dumps(create_vs_platform_receipts(
        args.manifest, args.protocol, args.preparation, args.verified_by),
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
