# -*- coding: utf-8 -*-

"""Offline FA-BM-VAE hardware preflight with no platform calls.

This workflow exports 50 stratified VS instances, evaluates fixed global gain
candidates, samples both logical and quantized matrices with local SA, writes
strictly shaped mock responses, and runs the real import/calibration/decoder
path. Every artifact is explicitly marked as non-physical.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibrate_hardware_temperature import evaluate_gain_candidate
from .compare_hardware_responses import compare_response_pair
from .import_hardware_responses import import_responses
from .kaiwu_adapter import (
    HARDWARE_N_BITS,
    LOGICAL_N_BITS,
    matrix_sha256,
    validate_hardware_matrix,
    validate_sampling_reads,
)
from .reconstruct_hardware_scenarios import reconstruct_scenarios


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "hardware_offline_preflight"
DEFAULT_GAIN_GRID = (25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0)


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def run_module(module, *args):
    command = [sys.executable, "-m", module] + [str(item) for item in args]
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_manifest(path):
    instances = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(instances, list) or len(instances) != 50:
        raise ValueError("Offline preflight requires exactly 50 VS instances")
    if any(instance.get("split") != "VS" for instance in instances):
        raise ValueError("Offline preflight is VS-only; TEST records are forbidden")
    return instances


def audit_export(manifest_path, gain):
    instances = load_manifest(manifest_path)
    audits = []
    for instance in instances:
        problem_path = resolve_repo_path(instance["path"])
        with np.load(problem_path, allow_pickle=False) as problem:
            if "hardware_matrix" not in problem:
                raise ValueError("Missing hardware_matrix for %s" % instance["id"])
            matrix = validate_hardware_matrix(problem["hardware_matrix"])
            if matrix.shape != (HARDWARE_N_BITS, HARDWARE_N_BITS):
                raise ValueError("Expected a 49x49 matrix for %s" % instance["id"])
            recorded_gain = float(np.asarray(problem["hardware_gain"]).reshape(-1)[0])
            recorded_hash = str(np.asarray(problem["hardware_matrix_sha256"]).reshape(-1)[0])
            if abs(recorded_gain - gain) > 1e-12:
                raise ValueError("Mixed global gains in %s" % instance["id"])
            if recorded_hash != matrix_sha256(matrix):
                raise ValueError("Matrix hash mismatch in %s" % instance["id"])
            audit = json.loads(str(np.asarray(problem["quantization_audit"]).reshape(-1)[0]))
            if audit["logical_n_bits"] != LOGICAL_N_BITS or audit["hardware_n_bits"] != HARDWARE_N_BITS:
                raise ValueError("Invalid bit dimensions in %s" % instance["id"])
            if audit["clipping"] or audit["precision_reducer"] or audit["variable_splitting"]:
                raise ValueError("Forbidden quantization operation in %s" % instance["id"])
            audits.append(audit)
    return {
        "n_instances": len(instances),
        "split": "VS",
        "logical_n_bits": LOGICAL_N_BITS,
        "hardware_n_bits": HARDWARE_N_BITS,
        "source_edges_min": min(item["source_nonzero_coefficients"] for item in audits),
        "source_edges_max": max(item["source_nonzero_coefficients"] for item in audits),
        "hardware_nonzero_edges_min": min(item["nonzero_edges"] for item in audits),
        "hardware_nonzero_edges_max": max(item["nonzero_edges"] for item in audits),
        "max_abs_integer": max(item["max_abs_integer"] for item in audits),
        "zeroed_coefficient_fraction_max": max(
            item["nonzero_quantized_to_zero_fraction"] for item in audits
        ),
        "relative_quantization_error_max": max(
            item["relative_quantization_error"] for item in audits
        ),
        "matrix_hashes": [item["matrix_sha256"] for item in audits],
    }


def write_mock_responses(submission_manifest, quantized_sa_dir, output_dir):
    submission_path = Path(submission_manifest).resolve()
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    mock_dir = Path(output_dir)
    mock_dir.mkdir(parents=True, exist_ok=True)
    for instance in submission["instances"]:
        instance_id = instance["instance_id"]
        source_path = Path(quantized_sa_dir) / (instance_id + ".npz")
        with np.load(source_path, allow_pickle=False) as source:
            raw = np.asarray(source["hardware_samples"], dtype=np.int8)
        if raw.shape[1] != HARDWARE_N_BITS or not np.isin(raw, (-1, 1)).all():
            raise ValueError("Quantized SA did not produce valid 49-spin samples")
        payload = json.loads(
            (submission_path.parent / instance["payload"]).read_text(encoding="utf-8")
        )
        np.savez_compressed(
            mock_dir / (instance_id + ".npz"),
            samples=raw,
            instance_id=np.asarray(instance_id),
            task_id=np.asarray("offline_mock_" + instance_id),
            task_name=np.asarray(payload["task_name"]),
            matrix_sha256=np.asarray(instance["hardware_matrix_sha256"]),
            bit_order=np.asarray("index_ascending"),
            backend=np.asarray("offline_quantized_sa_mock"),
            sdk_version=np.asarray("not_applicable"),
            platform_backend=np.asarray("none_offline_mock"),
            hardware_n_bits=np.asarray(HARDWARE_N_BITS),
            smoke_only=np.asarray(True),
            synthetic_data=np.asarray(True),
            eligible_for_paper=np.asarray(False),
            physical_platform_used=np.asarray(False),
        )
    return mock_dir


def evaluate_probability_metrics(source_manifest, float_sa_dir, canonical_dir):
    instances = load_manifest(source_manifest)
    rows = []
    for instance in instances:
        problem_path = resolve_repo_path(instance["path"])
        with np.load(problem_path, allow_pickle=False) as problem:
            h = np.asarray(problem["h"], dtype=np.float64)
            j = np.asarray(problem["J"], dtype=np.float64)
        with np.load(Path(float_sa_dir) / (instance["id"] + ".npz"), allow_pickle=False) as ref:
            reference = np.asarray(ref["samples"], dtype=np.int8)
        with np.load(Path(canonical_dir) / (instance["id"] + ".npz"), allow_pickle=False) as candidate:
            mock = np.asarray(candidate["samples"], dtype=np.int8)
        row = {"instance_id": instance["id"]}
        row.update(compare_response_pair(h, j, reference, mock))
        rows.append(row)
    metric_names = [
        "energy_wasserstein",
        "magnetization_mae",
        "edge_moment_mae",
        "beta_eff_difference",
    ]
    return {
        "n_instances": len(rows),
        "reference_label": "float_logical_sa",
        "candidate_label": "offline_quantized_sa_mock",
        "mean_metrics": {
            name: float(np.mean([row[name] for row in rows])) for name in metric_names
        },
        "max_metrics": {
            name: float(np.max([row[name] for row in rows])) for name in metric_names
        },
        "rows": rows,
    }


def run_candidate(args, output_dir, gain):
    label = "gain_%g" % gain
    candidate_dir = output_dir / label
    instance_dir = candidate_dir / "instances"
    export_args = [
        "--tag", args.tag,
        "--model-name", args.model_name,
        "--split", "VS",
        "--num-instances", "50",
        "--selection", "stratified",
        "--selection-seed", str(args.selection_seed),
        "--hardware-gain", str(gain),
        "--output-dir", instance_dir,
    ]
    if args.dataset_bundle:
        export_args += ["--dataset-bundle", args.dataset_bundle]
    if args.cpu:
        export_args.append("--cpu")
    run_module("GEFcom2014.models.QBM_VAE.export_ising_instances", *export_args)
    manifest_path = instance_dir / (
        "%s_%s_vs_manifest.json" % (args.tag, args.model_name)
    )
    export_audit = audit_export(manifest_path, gain)
    submission_dir = candidate_dir / "submission"
    run_module(
        "GEFcom2014.models.QBM_VAE.prepare_bosonic_submission",
        "--manifest", manifest_path,
        "--stage", "vs-calibration",
        "--requested-reads", str(args.reads),
        "--output-dir", submission_dir,
    )
    submission_manifest = submission_dir / "package" / "submission_manifest.json"
    float_dir = candidate_dir / "controls" / "float_logical_sa"
    quantized_dir = candidate_dir / "controls" / "quantized_matrix_sa"
    common_sample_args = [
        "--manifest", manifest_path,
        "--backend", "sa",
        "--num-reads", str(args.reads),
        "--beta", str(args.target_beta),
        "--sweeps", str(args.sweeps),
    ]
    run_module(
        "GEFcom2014.models.QBM_VAE.sample_exported_ising",
        *common_sample_args,
        "--matrix-space", "logical",
        "--output-dir", float_dir,
    )
    run_module(
        "GEFcom2014.models.QBM_VAE.sample_exported_ising",
        *common_sample_args,
        "--matrix-space", "hardware-quantized",
        "--output-dir", quantized_dir,
    )
    mock_dir = write_mock_responses(
        submission_manifest,
        quantized_dir / "sa",
        candidate_dir / "mock_responses",
    )
    canonical_dir = candidate_dir / "canonical_responses"
    _, import_audit = import_responses(
        submission_manifest, mock_dir, canonical_dir
    )
    candidate = evaluate_gain_candidate(
        {
            "gain": gain,
            "vs_manifest": str(manifest_path),
            "responses_dir": str(canonical_dir),
            "reference_dir": str(float_dir / "sa"),
        },
        target_beta=args.target_beta,
    )
    reconstruction = reconstruct_scenarios(
        manifest_path,
        canonical_dir,
        ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag) / (args.model_name + ".pickle"),
        ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag) / (args.model_name + ".json"),
        args.tag,
        dataset_bundle=args.dataset_bundle,
        output_dir=candidate_dir / "scenarios",
        output_label="OfflineMock",
        observation_noise=False,
        seed=args.selection_seed,
        cpu=args.cpu,
    )
    metrics = evaluate_probability_metrics(
        manifest_path, float_dir / "sa", canonical_dir
    )
    result = {
        "hardware_claim": False,
        "physical_platform_used": False,
        "gain": float(gain),
        "manifest": str(manifest_path.resolve()),
        "submission_manifest": str(submission_manifest.resolve()),
        "export_audit": export_audit,
        "import_audit": {
            "n_expected": import_audit["n_expected"],
            "n_imported": import_audit["n_imported"],
            "missing_instance_ids": import_audit["missing_instance_ids"],
        },
        "temperature_calibration": candidate,
        "probability_metrics": metrics,
        "reconstruction": {
            "scenario_path": str(reconstruction[0].resolve()),
            "metadata_path": str(reconstruction[1].resolve()),
        },
    }
    (candidate_dir / "candidate_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="wind", choices=["wind", "opsd-wind"])
    parser.add_argument("--model-name", default="wind_QBMVAE_2_lanchor_sa_0")
    parser.add_argument("--dataset-bundle", default=None)
    parser.add_argument("--gain-grid", type=float, nargs="+", default=list(DEFAULT_GAIN_GRID))
    parser.add_argument("--target-beta", type=float, default=1.0)
    parser.add_argument("--reads", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_beta <= 0 or args.sweeps < 1:
        raise ValueError("target-beta must be positive and sweeps must be positive")
    args.reads = validate_sampling_reads(args.reads)
    if len(args.gain_grid) == 0 or any(gain <= 0 for gain in args.gain_grid):
        raise ValueError("gain-grid must contain positive values")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty output directory: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    rejected = []
    for gain in args.gain_grid:
        try:
            candidates.append(run_candidate(args, output_dir, gain))
        except (subprocess.CalledProcessError, OverflowError, ValueError) as exc:
            rejected.append({"gain": float(gain), "reason": str(exc)})
    summary = {
        "experiment": "hardware_offline_preflight",
        "method_name": "FA-BM-VAE",
        "model_role": "forecast_anchor_conditional_bm_vae",
        "split": "VS",
        "n_instances": 50,
        "selection": "ten-zone stratified",
        "selection_seed": args.selection_seed,
        "gain_grid": [float(gain) for gain in args.gain_grid],
        "target_beta": float(args.target_beta),
        "reads_per_instance": int(args.reads),
        "hardware_claim": False,
        "physical_platform_used": False,
        "mock_response_backend": "offline_quantized_sa_mock",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_candidates": candidates,
        "rejected_candidates": rejected,
        "test_data_used": False,
    }
    (output_dir / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote offline FA-BM-VAE preflight to %s" % output_dir)
    print("Accepted gains: %s" % [item["gain"] for item in candidates])
    print("Rejected gains: %s" % rejected)


if __name__ == "__main__":
    main()
