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


def expand_samples(samples, n_bits, zero_spin=-1, reverse=False):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[1] != n_bits:
        raise ValueError(
            "Samples must have shape (n_reads, %d); got %s"
            % (n_bits, samples.shape)
        )
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
    counts = np.asarray(counts, dtype=np.int64)
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
            task_id = str(
                metadata.get(
                    "task_id", metadata.get("job_id", metadata.get("task_name", ""))
                )
            )
            if not task_id:
                raise ValueError(
                    "Kaiwu response %s must include task_id, task_name, or job_id"
                    % instance_id
                )
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
            "latency_s": float(metadata.get("latency_s", np.nan)),
            "applied_scale": float(metadata.get("applied_scale", np.nan)),
            "backend": str(metadata.get("backend", "bosonic_platform")),
            "job_id": str(metadata.get("job_id", metadata.get("task_id", ""))),
            "task_id": str(
                metadata.get(
                    "task_id", metadata.get("job_id", metadata.get("task_name", ""))
                )
            ),
            "raw_response_sha256": sha256_file(response_path),
        }
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
        records.append(
            {
                "instance_id": instance_id,
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
                        metadata.get("job_id", metadata.get("task_name", "")),
                    )
                ),
                "hardware_matrix_sha256": matrix_hash,
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
        "hardware_claim": bool(submission.get("hardware_claim", False)),
        "submission_stage": submission["stage"],
        "submission_manifest": str(Path(submission_manifest).resolve()),
        "n_expected": len(submission["instances"]),
        "n_imported": len(records),
        "missing_instance_ids": missing,
        "zero_spin": int(zero_spin),
        "bit_order": "index_descending" if reverse else "index_ascending",
        "records": records,
    }
    audit_path = output_dir / "response_import_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
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
