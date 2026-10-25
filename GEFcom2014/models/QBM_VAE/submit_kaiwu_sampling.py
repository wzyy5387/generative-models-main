# -*- coding: utf-8 -*-

"""Submit one frozen FA-BM-VAE 49-spin matrix to Kaiwu in SAMPLING mode.

This command is intentionally opt-in. It requires an externally configured
Kaiwu SDK/license; project_no is optional for standard SPQC tasks.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import re
from pathlib import Path

import numpy as np

from .kaiwu_adapter import (
    KaiwuClient,
    build_task_name,
    kaiwu_sdk_versions,
    matrix_sha256,
    validate_hardware_matrix,
    validate_raw_hardware_spins,
    validate_sampling_reads,
)


FROZEN_TASK_AUDIT_FIELDS = (
    "protocol_sha256", "model_checkpoint_sha256", "preparation_manifest_sha256",
    "protocol_version", "hardware_matrix_path", "hardware_matrix_file_sha256",
    "source_problem_path", "source_problem_sha256",
    "reference_sample_path", "reference_sample_sha256", "reference_sampler",
)


def load_matrix(matrix_file):
    matrix_file = Path(matrix_file)
    with np.load(matrix_file, allow_pickle=False) as archive:
        if "hardware_matrix" not in archive:
            raise ValueError("%s does not contain hardware_matrix" % matrix_file)
        matrix = validate_hardware_matrix(archive["hardware_matrix"])
        expected_hash = None
        for key in ("matrix_sha256", "hardware_matrix_sha256"):
            if key in archive:
                expected_hash = str(np.asarray(archive[key]).reshape(-1)[0])
                break
    actual_hash = matrix_sha256(matrix)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("Matrix SHA-256 mismatch in %s" % matrix_file)
    return matrix, actual_hash


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"complex_repr": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_failure_evidence(output_dir, instance_id, task_name, matrix_hash,
                            stage, num_reads, hardware_gain, started_at,
                            error, raw_samples=None):
    """Persist known response/error evidence without promoting it to results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (str(instance_id) + ".failure.json")
    if path.exists():
        raise RuntimeError("Failure evidence already exists; refusing to overwrite: %s" % path)
    message = re.sub(
        r"(?i)(user[_ -]?id|sdk[_ -]?code|password|token|authorization)\s*[:=]\s*\S+",
        r"\1=[REDACTED]", str(error),
    )
    payload = {
        "status": "invalid_or_incomplete_sdk_response",
        "instance_id": str(instance_id),
        "task_name": str(task_name),
        "matrix_sha256": str(matrix_hash),
        "stage": str(stage),
        "requested_reads": int(num_reads),
        "hardware_gain": None if hardware_gain is None else float(hardware_gain),
        "started_at": started_at,
        "error_type": type(error).__name__,
        "error": message,
        "hardware_claim": False,
        "physical_platform_used": False,
    }
    if raw_samples is not None:
        try:
            raw = np.asarray(raw_samples)
            payload["raw_samples"] = _json_safe(raw)
            payload["raw_samples_dtype"] = str(raw.dtype)
            payload["raw_samples_shape"] = list(raw.shape)
        except Exception:
            payload["raw_samples_repr"] = repr(raw_samples)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
        handle.write("\n")
    return path


def submit_one(
    matrix_file,
    instance_id,
    output_dir,
    num_reads,
    project_no=None,
    checkpoint_dir=None,
    task_name=None,
    stage="vs-calibration",
    hardware_gain=None,
    wait=True,
    interval=1,
    frozen_task=None,
):
    num_reads = validate_sampling_reads(num_reads)
    if checkpoint_dir is None:
        raise ValueError(
            "checkpoint_dir is required; pass a task-specific --checkpoint-dir"
        )
    matrix, matrix_hash = load_matrix(matrix_file)
    task_name = task_name or build_task_name(instance_id, matrix_hash)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (str(instance_id) + ".npz")
    audit_path = output_dir / (str(instance_id) + ".json")
    if output_path.exists() or audit_path.exists():
        raise RuntimeError("Response artifact already exists; refusing duplicate submit or overwrite")
    checkpoint_path = (Path(checkpoint_dir).resolve() / task_name)
    if checkpoint_path.exists() and any(checkpoint_path.iterdir()):
        raise RuntimeError("Task checkpoint directory is non-empty; manual review required")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    submitted_at = datetime.now(timezone.utc).isoformat()
    try:
        client = KaiwuClient(project_no=project_no, wait=wait, interval=interval)
        kaiwu_common = getattr(client._kaiwu, "common", None)
        if kaiwu_common is None:
            kaiwu_common = importlib.import_module("kaiwu.common")
        checkpoint_manager = getattr(kaiwu_common, "CheckpointManager", None)
        if checkpoint_manager is None:
            raise RuntimeError(
                "Kaiwu SDK does not expose common.CheckpointManager; cannot isolate task checkpoints"
            )
        checkpoint_manager.save_dir = str(checkpoint_path)
        result = client.sample_hardware_matrix(
            matrix, num_reads=num_reads, task_name=task_name
        )
        samples = validate_raw_hardware_spins(
            result["samples"], num_reads, matrix.shape[0]
        )
    except Exception as exc:
        raw_samples = getattr(exc, "raw_samples", None)
        if raw_samples is None and "result" in locals() and isinstance(result, dict):
            raw_samples = result.get("samples")
        failure_path = _write_failure_evidence(
            output_dir, instance_id, task_name, matrix_hash, stage,
            num_reads, hardware_gain, submitted_at, exc, raw_samples,
        )
        raise RuntimeError(
            "Kaiwu call failed or returned invalid data; batch must stop without retry. "
            "Evidence: %s" % failure_path
        ) from exc
    metadata = result["metadata"]
    sdk_version = str(
        getattr(client._kaiwu, "__version__", "unknown")
    )
    sdk_versions = kaiwu_sdk_versions(client._kaiwu)
    platform_backend = "kaiwu.cim.CIMOptimizer"
    completed_at = datetime.now(timezone.utc).isoformat()
    returned_reads = int(samples.shape[0])
    np.savez_compressed(
        output_path,
        samples=samples,
        instance_id=np.asarray(instance_id),
        task_id=np.asarray(metadata.get("task_id", "")),
        task_name=np.asarray(metadata.get("task_name", task_name)),
        matrix_sha256=np.asarray(matrix_hash),
        bit_order=np.asarray("index_ascending"),
        backend=np.asarray(metadata.get("backend", "kaiwu_cim_sampling")),
        sdk_version=np.asarray(sdk_version),
        kaiwu_distribution_version=np.asarray(
            sdk_versions["importlib_metadata_version"] or ""
        ),
        kaiwu_module_version=np.asarray(sdk_versions["kaiwu_module_version"] or ""),
        platform_backend=np.asarray(platform_backend),
        submitted_at=np.asarray(submitted_at),
        completed_at=np.asarray(completed_at),
        requested_reads=np.asarray(num_reads),
        returned_reads=np.asarray(returned_reads),
        stage=np.asarray(stage),
        hardware_gain=np.asarray(-1.0 if hardware_gain is None else float(hardware_gain)),
        # Raw platform output is never itself a hardware conclusion.  The
        # external audit is the only place allowed to promote these flags.
        hardware_claim=np.asarray(False),
        physical_platform_used=np.asarray(False),
        checkpoint_dir=np.asarray(str(checkpoint_path)),
    )
    response_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    task_id = str(metadata.get("task_id", ""))
    platform_metadata_complete = bool(
        client.__class__.__module__ == "GEFcom2014.models.QBM_VAE.kaiwu_adapter" and
        task_id and task_name and sdk_version and platform_backend and
        str(metadata.get("task_name", "")) == str(task_name) and
        str(metadata.get("backend", "")).startswith("kaiwu") and
        sdk_versions["importlib_metadata_version"] and
        sdk_versions["kaiwu_module_version"] and
        submitted_at and completed_at and matrix_hash and response_hash and
        returned_reads == int(num_reads)
    )
    audit = {
        "method_name": "FA-BM-VAE",
        "training_backend": "classical",
        "sampling_stage": "post_training_conditional_ising_latent",
        "stage": str(stage),
        "gain": None if hardware_gain is None else float(hardware_gain),
        "hardware_claim": False,
        "physical_platform_used": False,
        "completed": False,
        "instance_id": str(instance_id),
        "task_id": str(metadata.get("task_id", "")),
        "task_name": str(metadata.get("task_name", task_name)),
        "matrix_sha256": matrix_hash,
        "sdk_version": sdk_version,
        "sdk_versions": sdk_versions,
        "platform_backend": platform_backend,
        "submitted_at": submitted_at,
        "completed_at": completed_at,
        "checkpoint_dir": str(checkpoint_path),
        "hardware_n_bits": int(matrix.shape[0]),
        "requested_reads": int(num_reads),
        "returned_reads": returned_reads,
        "response_file_sha256": response_hash,
        "response_path": str(output_path.resolve()),
        "hardware_metadata_complete": False,
    }
    # Persist provenance in the first audit, before returning to the driver.
    # A crash before its waiting marker must not leave an unusable base audit.
    audit.update({key: frozen_task[key] for key in FROZEN_TASK_AUDIT_FIELDS
                  if frozen_task and frozen_task.get(key)})
    # Validate the NPZ contents before promoting the audit flags.  A mock
    # client, incomplete metadata, malformed samples, or any hash mismatch
    # leaves the audit explicitly non-physical.
    if platform_metadata_complete:
        from .import_hardware_responses import strict_response_record_valid
        candidate = dict(audit, hardware_claim=True,
                         physical_platform_used=True, completed=True)
        candidate["hardware_metadata_complete"] = True
        if strict_response_record_valid(candidate):
            audit.update(candidate)
            audit["hardware_metadata_complete"] = True
    if audit_path.exists():
        raise RuntimeError("Response audit already exists; refusing to overwrite: %s" % audit_path)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output_path, audit_path, audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-file", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Base directory; a task-specific subdirectory is created.",
    )
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--project-no", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--confirm-real-submission",
        action="store_true",
        help="Required together with FA_BM_VAE_RUN_REAL_KAIWU=1 for paid tasks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval < 1:
        raise ValueError("--interval must be at least one minute")
    if not args.confirm_real_submission:
        raise RuntimeError(
            "CLI submission is disabled by default; pass --confirm-real-submission "
            "only after manual approval"
        )
    if os.environ.get("FA_BM_VAE_RUN_REAL_KAIWU") != "1":
        raise RuntimeError(
            "Set FA_BM_VAE_RUN_REAL_KAIWU=1 only in an approved hardware environment"
        )
    output_path, audit_path, _ = submit_one(
        matrix_file=args.matrix_file,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_reads=args.num_reads,
        project_no=args.project_no,
        task_name=args.task_name,
        wait=not args.no_wait,
        interval=args.interval,
    )
    print("Wrote raw Kaiwu response to %s" % output_path)
    print("Wrote submission audit to %s" % audit_path)


if __name__ == "__main__":
    main()
