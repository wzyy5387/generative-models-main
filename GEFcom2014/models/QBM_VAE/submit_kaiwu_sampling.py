# -*- coding: utf-8 -*-

"""Submit one frozen FA-BM-VAE 49-spin matrix to Kaiwu in SAMPLING mode.

This command is intentionally opt-in. It requires an externally configured
Kaiwu SDK/license/project number and is never called by tests or training.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .kaiwu_adapter import (
    KaiwuClient,
    build_task_name,
    matrix_sha256,
    validate_hardware_matrix,
    validate_sampling_reads,
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


def submit_one(
    matrix_file,
    instance_id,
    output_dir,
    num_reads,
    project_no=None,
    task_name=None,
    wait=True,
    interval=1,
):
    num_reads = validate_sampling_reads(num_reads)
    matrix, matrix_hash = load_matrix(matrix_file)
    task_name = task_name or build_task_name(instance_id, matrix_hash)
    client = KaiwuClient(project_no=project_no, wait=wait, interval=interval)
    result = client.sample_hardware_matrix(
        matrix, num_reads=num_reads, task_name=task_name
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    output_path = output_dir / (str(instance_id) + ".npz")
    np.savez_compressed(
        output_path,
        samples=np.asarray(result["samples"], dtype=np.int8),
        instance_id=np.asarray(instance_id),
        task_id=np.asarray(metadata.get("task_id", "")),
        task_name=np.asarray(metadata.get("task_name", task_name)),
        matrix_sha256=np.asarray(matrix_hash),
        bit_order=np.asarray("index_ascending"),
        backend=np.asarray(metadata.get("backend", "kaiwu_cim_sampling")),
    )
    audit = {
        "method_name": "FA-BM-VAE",
        "training_backend": "classical",
        "sampling_stage": "post_training_conditional_ising_latent",
        "hardware_claim": False,
        "instance_id": str(instance_id),
        "task_id": str(metadata.get("task_id", "")),
        "task_name": str(metadata.get("task_name", task_name)),
        "matrix_sha256": matrix_hash,
        "hardware_n_bits": int(matrix.shape[0]),
        "requested_reads": int(num_reads),
        "response_path": str(output_path.resolve()),
    }
    audit_path = output_dir / (str(instance_id) + ".json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output_path, audit_path, audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-file", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--project-no", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval < 1:
        raise ValueError("--interval must be at least one minute")
    output_path, audit_path, _ = submit_one(
        matrix_file=args.matrix_file,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
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
