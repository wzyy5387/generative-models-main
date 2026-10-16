# -*- coding: utf-8 -*-
"""Offline OPSD VS quality-gate audit.

This module only uses frozen exported instances and local SA.  It never imports
the Kaiwu SDK or submits a task.  Quantiles produced here are proposed
reference thresholds, not validated hardware acceptance limits.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .compare_hardware_responses import compare_response_pair
from .kaiwu_13k_protocol import (
    select_vs_gain,
    validate_vs_response_records,
)
from .samplers import build_sampler
from .kaiwu_adapter import normalize_hardware_spins


ROOT_DIR = Path(__file__).resolve().parents[3]
METRIC_NAMES = (
    "energy_wasserstein",
    "effective_beta_error",
    "edge_moment_mae",
    "magnetization_mae",
    "unique_state_fraction",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def _load_instances(preparation, gain):
    manifest = next(
        Path(preparation).joinpath("export_vs_gain_%d" % gain).glob("*_vs_manifest.json")
    )
    return json.loads(manifest.read_text(encoding="utf-8"))


def _load_logical_problem(instance):
    with np.load(_resolve(instance["path"]), allow_pickle=False) as archive:
        return np.asarray(archive["h"], dtype=np.float64), np.asarray(archive["J"], dtype=np.float64)


def _sample_float(h, j, reads, sweeps, seed):
    sampler = build_sampler("sa", sweeps=sweeps, seed=seed)
    return sampler.sample_ising(h, j, num_reads=reads, beta=1.0).samples.astype(np.int8)


def _sample_quantized(instance, reads, sweeps, seed):
    package = _resolve(instance["path"]).with_name(
        Path(instance["path"]).stem + "_hardware_matrix.npz"
    )
    with np.load(package, allow_pickle=False) as archive:
        matrix = np.asarray(archive["hardware_matrix"], dtype=np.float64)
        gain = float(np.asarray(archive["hardware_gain"]).reshape(-1)[0])
    h = np.zeros(matrix.shape[0], dtype=np.float64)
    j = -2.0 * matrix
    sampler = build_sampler("sa", sweeps=sweeps, beta_start=0.1 / gain, seed=seed)
    raw = sampler.sample_ising(
        h, j, num_reads=reads, beta=1.0 / gain
    ).samples.astype(np.int8)
    return normalize_hardware_spins(raw, logical_n_bits=48), raw


def _reference_thresholds(rows):
    values = {name: np.asarray([float(row[name]) for row in rows], dtype=float)
              for name in METRIC_NAMES if rows}
    thresholds = {
        name: float(np.quantile(values[name], 0.95))
        for name in METRIC_NAMES[:-1]
    }
    thresholds["unique_state_fraction_min"] = float(
        np.quantile(values["unique_state_fraction"], 0.05)
    )
    distributions = {}
    for name, array in values.items():
        distributions[name] = {
            "n": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "q05": float(np.quantile(array, 0.05)),
            "q50": float(np.quantile(array, 0.50)),
            "q95": float(np.quantile(array, 0.95)),
        }
    return thresholds, distributions


def generate_reference_audit(preparation, output_dir, repeats=5, reads=100,
                             sweeps=100, seed=202600):
    """Generate independent float/quantized SA repeats and proposed limits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    null_rows = []
    candidate_rows = []
    repeat_manifest = []
    for gain in (125, 150, 200):
        instances = _load_instances(preparation, gain)
        for index, instance in enumerate(instances):
            h, j = _load_logical_problem(instance)
            float_samples = []
            for repeat in range(repeats):
                current_seed = int(seed + gain + index * 100 + repeat)
                path = output_dir / "samples" / ("gain_%d" % gain) / ("repeat_%02d" % repeat) / (instance["id"] + ".npz")
                if path.exists():
                    with np.load(path, allow_pickle=False) as archive:
                        samples = np.asarray(archive["samples"], dtype=np.int8)
                else:
                    samples = _sample_float(h, j, reads, sweeps, current_seed)
                float_samples.append(samples)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    np.savez_compressed(path, samples=samples, hardware_claim=False,
                                        physical_platform_used=False, matrix_space="logical",
                                        gain=gain, repeat=repeat, seed=current_seed)
            for left in range(repeats):
                for right in range(left + 1, repeats):
                    row = compare_response_pair(h, j, float_samples[left], float_samples[right])
                    row["effective_beta_error"] = abs(float(row["candidate_beta_eff"]) - 1.0)
                    row["gain"] = gain
                    row["instance_id"] = instance["id"]
                    row["repeat_pair"] = [left, right]
                    row["unique_state_fraction"] = row["candidate_unique_state_fraction"]
                    null_rows.append(row)
            quant_path = output_dir / "samples" / ("gain_%d" % gain) / "candidate" / (instance["id"] + ".npz")
            quant_path.parent.mkdir(parents=True, exist_ok=True)
            if quant_path.exists():
                with np.load(quant_path, allow_pickle=False) as archive:
                    quantized = np.asarray(archive["samples"], dtype=np.int8)
            else:
                quantized, raw = _sample_quantized(
                    instance, reads, sweeps, int(seed + 100000 + gain + index)
                )
                np.savez_compressed(quant_path, samples=quantized, hardware_samples=raw,
                                    hardware_claim=False, physical_platform_used=False,
                                    matrix_space="hardware-quantized", gain=gain,
                                    reads=reads)
            row = compare_response_pair(h, j, float_samples[0], quantized)
            row["effective_beta_error"] = abs(float(row["candidate_beta_eff"]) - 1.0)
            row["gain"] = gain
            row["instance_id"] = instance["id"]
            row["unique_state_fraction"] = row["candidate_unique_state_fraction"]
            candidate_rows.append(row)
            repeat_manifest.append({"gain": gain, "instance_id": instance["id"],
                                    "float_repeats": repeats, "reads": reads,
                                    "sweeps": sweeps})
    thresholds, distributions = _reference_thresholds(null_rows)
    by_gain = []
    for gain in (125, 150, 200):
        rows = [row for row in candidate_rows if row["gain"] == gain]
        by_gain.append({
            "gain": gain,
            "n_instances": len(rows),
            "metrics": {name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES},
        })
    report = {
        "artifact_role": "offline_quality_gate_reference_audit",
        "dataset": "OPSD Wind",
        "preparation_source": str(Path(preparation).resolve()),
        "repeats": repeats,
        "reads_per_repeat": reads,
        "sweeps": sweeps,
        "seed_rule": "seed + gain + instance_index*100 + repeat",
        "independent_reference": "float logical SA repeats; pairwise null distribution",
        "null_distribution": distributions,
        "suggested_thresholds": thresholds,
        "threshold_basis": "95th percentile for errors and 5th percentile for unique-state fraction from local SA null pairs",
        "thresholds_approved": False,
        "thresholds_validated_on_physical_platform": False,
        "candidate_gain_metrics": by_gain,
        "reference_samples": repeat_manifest,
        "hardware_claim": False,
        "physical_platform_used": False,
        "test_data_used": False,
        "test_peeking": False,
        "quality_gate_status": "pending_manual_review",
        "quality_gate_action": "lock TEST until thresholds are explicitly approved",
    }
    (output_dir / "quality_gate_reference_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "quality_gate_null_rows.json").write_text(
        json.dumps(null_rows, indent=2), encoding="utf-8"
    )
    return report


def make_offline_response_records(preparation, output_dir, protocol, threshold_report):
    """Build format-valid, explicitly non-physical mock records for tests."""
    output_dir = Path(output_dir)
    records = []
    identity = {
        "protocol_sha256": sha256_file(protocol["_path"]),
        "model_checkpoint_sha256": json.loads(
            (Path(preparation) / "preparation_manifest.json").read_text(encoding="utf-8-sig")
        )["model_checkpoint_sha256"],
        "preparation_manifest_sha256": sha256_file(Path(preparation) / "preparation_manifest.json"),
    }
    for gain in (125, 150, 200):
        instances = _load_instances(preparation, gain)
        package_manifest = json.loads(
            (Path(preparation) / ("submission_gain_%d" % gain) / "package" / "submission_manifest.json").read_text(encoding="utf-8")
        )
        package_items = {item["instance_id"]: item for item in package_manifest["instances"]}
        for index, instance in enumerate(instances):
            h, j = _load_logical_problem(instance)
            float_path = output_dir / "samples" / ("gain_%d" % gain) / "repeat_00" / (instance["id"] + ".npz")
            candidate_path = output_dir / "samples" / ("gain_%d" % gain) / "candidate" / (instance["id"] + ".npz")
            with np.load(float_path) as ref, np.load(candidate_path) as candidate:
                ref_samples = np.asarray(ref["samples"], dtype=np.int8)
                raw = np.asarray(candidate["hardware_samples"], dtype=np.int8)
                logical = np.asarray(candidate["samples"], dtype=np.int8)
            response_path = output_dir / "mock_responses" / ("gain_%d" % gain) / (instance["id"] + ".npz")
            response_path.parent.mkdir(parents=True, exist_ok=True)
            task_id = "offline-task-%d-%s" % (gain, index)
            package_item = package_items[instance["id"]]
            task_name = json.loads(
                (Path(preparation) / ("submission_gain_%d" % gain) / "package" / package_item["payload"]).read_text(encoding="utf-8")
            )["task_name"]
            np.savez_compressed(
                response_path, samples=raw, hardware_samples=raw,
                instance_id=instance["id"], task_id=task_id, task_name=task_name,
                matrix_sha256=instance["hardware_matrix_sha256"], stage="vs",
                hardware_gain=gain, requested_reads=100, returned_reads=100,
            )
            metrics = compare_response_pair(h, j, ref_samples, logical)
            record = {
                **identity,
                "completed": True, "instance_id": instance["id"], "stage": "vs",
                "task_id": task_id, "task_name": task_name, "gain": gain,
                "matrix_sha256": instance["hardware_matrix_sha256"],
                "response_path": str(response_path.resolve()),
                "response_file_sha256": sha256_file(response_path),
                "requested_reads": 100, "returned_reads": 100,
                "sdk_version": "offline-mock", "platform_backend": "offline-mock",
                "submitted_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "hardware_metadata_complete": True,
                "hardware_claim": False, "physical_platform_used": False,
                "sdk_versions": {"importlib_metadata_version": "offline-mock",
                                  "kaiwu_module_version": "offline-mock"},
                "energy_wasserstein": metrics["energy_wasserstein"],
                "effective_beta_error": abs(float(metrics["candidate_beta_eff"]) - 1.0),
                "edge_moment_mae": metrics["edge_moment_mae"],
                "magnetization_mae": metrics["magnetization_mae"],
                "unique_state_fraction": metrics["candidate_unique_state_fraction"],
                "quality_assessment": {"status": "forged-and-ignored", "rule_version": "wrong"},
            }
            records.append(record)
    return records


def run_offline_chain(preparation, output_dir, protocol):
    audit = json.loads((Path(output_dir) / "quality_gate_reference_audit.json").read_text(encoding="utf-8"))
    records = make_offline_response_records(preparation, output_dir, protocol, audit)
    identity = {key: records[0][key] for key in
                ("protocol_sha256", "model_checkpoint_sha256", "preparation_manifest_sha256")}
    approved = {"status": "defined", "rule_version": "offline-test-only-v1",
                "thresholds_approved": True, "test_only": True,
                "thresholds": {"energy_wasserstein": 10.0,
                               "effective_beta_error": 10.0,
                               "edge_moment_mae": 10.0,
                               "magnetization_mae": 10.0,
                               "unique_state_fraction_min": 0.0}}
    expected_ids = sorted({row["instance_id"] for row in records})
    validated = validate_vs_response_records(
        records, (125, 150, 200), 10, expected_instance_ids=expected_ids,
        quality_policy=approved, require_quality_policy=False, require_physical=False)
    freeze_path = Path(output_dir) / "offline_mock_gain_selection.json"
    payload = select_vs_gain(
        validated, freeze_path, required_gains=(125, 150, 200),
        expected_instances_per_gain=10, expected_instance_ids=expected_ids,
        quality_policy=approved, expected_identity=identity, require_physical=False)
    payload["artifact_role"] = "offline_simulation"
    payload["hardware_claim"] = False
    payload["physical_platform_used"] = False
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (Path(output_dir) / "offline_chain_audit.json").write_text(json.dumps({
        "format_validation": "passed with physical flags false",
        "quality_recomputed": True,
        "forged_quality_field_used": False,
        "freeze_written": str(freeze_path.resolve()),
        "test_dry_run": "prepared only; no SDK call",
        "hardware_claim": False,
        "physical_platform_used": False,
    }, indent=2), encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reads", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202600)
    parser.add_argument("--offline-chain", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = generate_reference_audit(args.preparation, args.output_dir,
                                      args.repeats, args.reads, args.sweeps, args.seed)
    if args.offline_chain:
        from .kaiwu_13k_protocol import load_protocol
        protocol_path = ROOT_DIR / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json"
        protocol = load_protocol(protocol_path)
        protocol["_path"] = str(protocol_path.resolve())
        run_offline_chain(args.preparation, args.output_dir, protocol)
    print(json.dumps({"quality_gate_status": report["quality_gate_status"],
                      "suggested_thresholds": report["suggested_thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
