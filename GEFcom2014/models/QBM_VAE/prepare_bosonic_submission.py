# -*- coding: utf-8 -*-

"""Build a hash-locked, vendor-neutral Ising hardware submission bundle."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from .kaiwu_adapter import (
    HARDWARE_N_BITS,
    LOGICAL_N_BITS,
    matrix_sha256,
    validate_hardware_matrix,
    validate_sampling_reads,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "bosonic_submissions"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def validate_payload(payload, expected_id=None):
    if expected_id is not None and payload.get("instance_id") != expected_id:
        raise ValueError("Payload instance_id does not match manifest")
    h = np.asarray(payload["h"], dtype=np.float64)
    if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all():
        raise ValueError("Payload h must be a finite non-empty vector")
    n_bits = int(payload.get("expanded_n_bits", h.size))
    if n_bits != h.size:
        raise ValueError("expanded_n_bits does not match len(h)")
    hardware_matrix = payload.get("hardware_matrix")
    hardware_stats = None
    if hardware_matrix is not None:
        if int(payload.get("logical_n_bits", h.size)) != LOGICAL_N_BITS:
            raise ValueError("Kaiwu payload must declare logical_n_bits=48")
        if int(payload.get("hardware_n_bits", 0)) != HARDWARE_N_BITS:
            raise ValueError("Kaiwu payload must declare hardware_n_bits=49")
        if int(payload.get("auxiliary_spin_index", -1)) != LOGICAL_N_BITS:
            raise ValueError("Kaiwu payload must place the auxiliary spin at index 48")
        matrix = validate_hardware_matrix(np.asarray(hardware_matrix, dtype=np.int16))
        expected_hash = payload.get("hardware_matrix_sha256")
        if expected_hash != matrix_sha256(matrix):
            raise ValueError("Hardware matrix SHA-256 mismatch")
        hardware_edges = np.triu(np.abs(matrix) > 0, k=1)
        hardware_stats = {
            "logical_n_bits": LOGICAL_N_BITS,
            "hardware_n_bits": HARDWARE_N_BITS,
            "auxiliary_spin_index": LOGICAL_N_BITS,
            "hardware_matrix_sha256": matrix_sha256(matrix),
            "hardware_gain": float(payload["hardware_gain"]),
            "hardware_n_bits_declared": int(payload["hardware_n_bits"]),
        }
    seen_edges = set()
    coupling_values = []
    degrees = np.zeros(n_bits, dtype=np.int64)
    for coupling in payload.get("couplings", []):
        i = int(coupling["i"])
        j = int(coupling["j"])
        value = float(coupling["value"])
        if i < 0 or j < 0 or i >= n_bits or j >= n_bits or i >= j:
            raise ValueError("Couplings must use unique upper-triangle indices")
        if (i, j) in seen_edges or not np.isfinite(value):
            raise ValueError("Payload contains a duplicate or non-finite coupling")
        seen_edges.add((i, j))
        coupling_values.append(value)
        degrees[i] += 1
        degrees[j] += 1
    coefficients = np.concatenate(
        (np.abs(h), np.abs(np.asarray(coupling_values, dtype=np.float64)))
    )
    stats = {
        "n_bits": (
            int(hardware_stats["hardware_n_bits"])
            if hardware_stats is not None
            else n_bits
        ),
        "n_edges": (
            int(np.count_nonzero(hardware_edges))
            if hardware_stats is not None
            else len(seen_edges)
        ),
        "max_degree": int(degrees.max()) if degrees.size else 0,
        "max_abs_h": float(np.abs(h).max()),
        "max_abs_j": (
            float(np.abs(coupling_values).max()) if coupling_values else 0.0
        ),
        "max_abs_coefficient": float(coefficients.max()),
    }
    if hardware_stats is not None:
        hardware_degrees = np.count_nonzero(np.abs(matrix) > 0, axis=1)
        stats.update(hardware_stats)
        stats["max_degree"] = int(hardware_degrees.max())
        stats["max_abs_coefficient"] = int(np.abs(matrix).max())
        stats["source_n_edges"] = int(
            payload.get("quantization_audit", {}).get(
                "source_nonzero_coefficients", stats["n_edges"]
            )
        )
        stats["hardware_nonzero_edges"] = stats["n_edges"]
        # Keep the declared task size tied to the original 48-spin Ising
        # instance: 180 J edges plus 48 auxiliary-spin field edges. The
        # quantized matrix may have fewer nonzero entries, which is audited
        # separately rather than silently redefining the task.
        stats["n_edges"] = stats["source_n_edges"]
    return stats


def enforce_limits(stats, max_bits=None, max_edges=None, coefficient_limit=None):
    if max_bits is not None and stats["n_bits"] > max_bits:
        raise ValueError(
            "Problem has %d bits, above platform limit %d"
            % (stats["n_bits"], max_bits)
        )
    if max_edges is not None and stats["n_edges"] > max_edges:
        raise ValueError(
            "Problem has %d edges, above platform limit %d"
            % (stats["n_edges"], max_edges)
        )
    if (
        coefficient_limit is not None
        and stats["max_abs_coefficient"] > coefficient_limit + 1e-12
    ):
        raise ValueError(
            "Coefficient %.8g exceeds platform limit %.8g; do not clip. "
            "Choose a globally consistent scale and repeat VS calibration."
            % (stats["max_abs_coefficient"], coefficient_limit)
        )


def write_deterministic_zip(package_dir, archive_path):
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def prepare_submission(
    source_manifest,
    output_dir,
    stage,
    requested_reads,
    max_bits=None,
    max_edges=None,
    coefficient_limit=None,
):
    source_manifest = Path(source_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    requested_reads = validate_sampling_reads(requested_reads)
    package_dir = output_dir / "package"
    payload_dir = package_dir / "payloads"
    matrix_dir = package_dir / "hardware_matrices"
    payload_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for stale_payload in payload_dir.glob("*.json"):
        stale_payload.unlink()
    for stale_matrix in matrix_dir.glob("*.npz"):
        stale_matrix.unlink()
    instances = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected_split = "VS" if stage == "vs-calibration" else "TEST"
    records = []
    calibration_records = []
    hardware_gains = set()
    for instance in instances:
        if instance.get("split") != expected_split:
            raise ValueError(
                "Stage %s requires %s instances" % (stage, expected_split)
            )
        source_payload = resolve_repo_path(instance["platform_payload"])
        payload = json.loads(source_payload.read_text(encoding="utf-8"))
        if stage == "test-evaluation":
            calibration = payload.get("temperature_calibration")
            if calibration is None or calibration.get("fit_split") != "VS":
                raise ValueError(
                    "TEST payload %s is not frozen from VS calibration"
                    % instance["id"]
                )
            calibration_records.append(calibration)
        stats = validate_payload(payload, instance["id"])
        enforce_limits(stats, max_bits, max_edges, coefficient_limit)
        destination = payload_dir / (instance["id"] + ".json")
        shutil.copyfile(source_payload, destination)
        record = {
            "instance_id": instance["id"],
            "split": instance["split"],
            "payload": str(destination.relative_to(package_dir).as_posix()),
            "payload_sha256": sha256_file(destination),
            **stats,
        }
        if "hardware_matrix_file" in payload:
            matrix_source = resolve_repo_path(payload["hardware_matrix_file"])
            if not matrix_source.is_file():
                raise FileNotFoundError(
                    "Hardware matrix file is missing: %s" % matrix_source
                )
            with np.load(matrix_source, allow_pickle=False) as matrix_archive:
                if "hardware_matrix" not in matrix_archive:
                    raise ValueError(
                        "Hardware matrix file lacks hardware_matrix: %s"
                        % matrix_source
                    )
                matrix_from_file = validate_hardware_matrix(
                    matrix_archive["hardware_matrix"]
                )
            if matrix_sha256(matrix_from_file) != payload["hardware_matrix_sha256"]:
                raise ValueError(
                    "Hardware matrix file does not match payload hash for %s"
                    % instance["id"]
                )
            matrix_destination = matrix_dir / (instance["id"] + ".npz")
            shutil.copyfile(matrix_source, matrix_destination)
            record["hardware_matrix"] = str(
                matrix_destination.relative_to(package_dir).as_posix()
            )
            record["hardware_matrix_sha256_file"] = sha256_file(matrix_destination)
            hardware_gains.add(float(payload["hardware_gain"]))
        records.append(record)

    if hardware_gains and len(hardware_gains) != 1:
        raise ValueError("All instances in one submission must use one global hardware_gain")
    if hardware_gains and any("hardware_matrix" not in record for record in records):
        raise ValueError("A Kaiwu submission cannot mix hardware and logical-only instances")

    schema = {
        "required": {
            "instance_id": "must match the submitted instance",
            "samples": "2-D array with hardware_n_bits columns, using {-1,+1} or {0,1}",
        },
        "alternative": {
            "bitstrings": "list of index-ascending binary strings",
            "counts": "optional positive integer multiplicities",
        },
        "optional": [
            "job_id",
            "latency_s",
            "applied_scale",
            "backend",
            "platform_metadata",
            "task_id",
            "task_name",
        ],
        "spin_order": "index_ascending",
        "default_binary_mapping": {"0": -1, "1": 1},
    }
    (package_dir / "response_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    submission = {
        "schema_version": 1,
        "method_name": instances[0].get("method_name", "FA-BM-VAE")
        if instances
        else "FA-BM-VAE",
        "model_role": instances[0].get(
            "model_role", "forecast_anchor_conditional_bm_vae"
        ) if instances else "forecast_anchor_conditional_bm_vae",
        "training_backend": "classical",
        "sampling_stage": "post_training_conditional_ising_latent",
        "hardware_substitution": (
            "platform replaces only frozen conditional Ising latent sampling"
        ),
        "hardware_claim": False,
        "stage": stage,
        "split": expected_split,
        "frozen": True,
        "target_labels_included": False,
        "source_manifest_sha256": sha256_file(source_manifest),
        "requested_reads_per_instance": int(requested_reads),
        "hardware_gain": next(iter(hardware_gains)) if hardware_gains else None,
        "n_instances": len(records),
        "platform_limits_checked": {
            "max_bits": max_bits,
            "max_edges": max_edges,
            "coefficient_limit": coefficient_limit,
        },
        "problem_summary": {
            "n_bits_min": min(record["n_bits"] for record in records),
            "n_bits_max": max(record["n_bits"] for record in records),
            "n_edges_min": min(record["n_edges"] for record in records),
            "n_edges_max": max(record["n_edges"] for record in records),
            "source_n_edges_min": min(
                record.get("source_n_edges", record["n_edges"]) for record in records
            ),
            "source_n_edges_max": max(
                record.get("source_n_edges", record["n_edges"]) for record in records
            ),
            "hardware_nonzero_edges_min": min(
                record.get("hardware_nonzero_edges", record["n_edges"])
                for record in records
            ),
            "hardware_nonzero_edges_max": max(
                record.get("hardware_nonzero_edges", record["n_edges"])
                for record in records
            ),
            "max_degree": max(record["max_degree"] for record in records),
            "max_abs_coefficient": max(
                record["max_abs_coefficient"] for record in records
            ),
        },
        "response_schema": "response_schema.json",
        "instances": records,
    }
    if calibration_records:
        scales = {
            float(record["coefficient_scale"]) for record in calibration_records
        }
        if len(scales) != 1:
            raise ValueError("All TEST payloads must use one frozen global scale")
        submission["temperature_calibration"] = {
            "fit_split": "VS",
            "target_beta": float(calibration_records[0]["target_beta"]),
            "estimated_beta_eff": float(
                calibration_records[0]["estimated_beta_eff"]
            ),
            "coefficient_scale": scales.pop(),
        }
    manifest_path = package_dir / "submission_manifest.json"
    manifest_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    archive_path = output_dir / ("%s.zip" % stage)
    write_deterministic_zip(package_dir, archive_path)
    archive_hash = sha256_file(archive_path)
    (output_dir / ("%s.zip.sha256" % stage)).write_text(
        "%s  %s\n" % (archive_hash, archive_path.name), encoding="ascii"
    )
    return manifest_path, archive_path, archive_hash


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["vs-calibration", "test-evaluation"],
        required=True,
    )
    parser.add_argument("--requested-reads", type=int, default=1000)
    parser.add_argument("--max-bits", type=int)
    parser.add_argument("--max-edges", type=int)
    parser.add_argument("--coefficient-limit", type=float)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.requested_reads <= 0:
        raise ValueError("--requested-reads must be positive")
    manifest_path, archive_path, archive_hash = prepare_submission(
        args.manifest,
        args.output_dir,
        args.stage,
        args.requested_reads,
        max_bits=args.max_bits,
        max_edges=args.max_edges,
        coefficient_limit=args.coefficient_limit,
    )
    print("Submission manifest: %s" % manifest_path)
    print("Submission archive: %s" % archive_path)
    print("Archive SHA-256: %s" % archive_hash)


if __name__ == "__main__":
    main()
