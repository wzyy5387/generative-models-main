# -*- coding: utf-8 -*-

"""Fit VS-only effective temperature and scale frozen TEST Ising payloads."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .calibration import fit_shared_effective_temperature_pseudolikelihood
from .compare_hardware_responses import compare_response_pair, response_statistics


ROOT_DIR = Path(__file__).resolve().parents[3]


def load_calibration_problems(manifest_path, responses_dir):
    instances = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems = []
    response_paths = []
    for instance in instances:
        problem_path = resolve_repo_path(instance["path"])
        response_path = Path(responses_dir) / (instance["id"] + ".npz")
        if not response_path.is_file():
            raise FileNotFoundError(
                "Missing VS response for %s: %s" % (instance["id"], response_path)
            )
        with np.load(problem_path, allow_pickle=False) as problem:
            h = np.asarray(problem["h"], dtype=np.float64)
            j = np.asarray(problem["J"], dtype=np.float64)
        with np.load(response_path, allow_pickle=False) as response:
            if "samples" not in response:
                raise ValueError("%s does not contain samples" % response_path)
            samples = np.asarray(response["samples"], dtype=np.int8)
        problems.append((h, j, samples))
        response_paths.append(str(response_path.resolve()))
    return instances, problems, response_paths


def scale_payload(payload, coefficient_scale, calibration):
    if "hardware_matrix" in payload and abs(float(coefficient_scale) - 1.0) > 1e-12:
        raise ValueError(
            "Kaiwu payloads cannot be scaled after quantization; rebuild from "
            "the original h,J into a 49-spin matrix with a new global hardware_gain"
        )
    scaled = deepcopy(payload)
    scaled["h"] = [
        float(coefficient_scale * value) for value in payload["h"]
    ]
    scaled["couplings"] = [
        {
            **coupling,
            "value": float(coefficient_scale * coupling["value"]),
        }
        for coupling in payload["couplings"]
    ]
    scaled["temperature_calibration"] = {
        "fit_split": "VS",
        "target_beta": calibration["target_beta"],
        "estimated_beta_eff": calibration["beta_eff"],
        "coefficient_scale": coefficient_scale,
        "platform_has_no_direct_beta": bool(
            calibration.get("platform_has_no_direct_beta", False)
        ),
    }
    if "hardware_gain" in calibration:
        scaled["temperature_calibration"]["hardware_gain"] = calibration[
            "hardware_gain"
        ]
    return scaled


def evaluate_gain_candidate(candidate, target_beta=1.0, bootstrap_repetitions=0, seed=0):
    """Evaluate one already-sampled VS gain on the original logical h,J."""
    manifest_path = Path(candidate["vs_manifest"])
    responses_dir = Path(candidate["responses_dir"])
    reference_dir = (
        Path(candidate["reference_dir"]) if candidate.get("reference_dir") else None
    )
    instances = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    per_instance = []
    reference_metrics = []
    for instance in instances:
        problem_path = resolve_repo_path(instance["path"])
        response_path = responses_dir / (instance["id"] + ".npz")
        if not response_path.is_file():
            raise FileNotFoundError("Missing candidate response: %s" % response_path)
        with np.load(problem_path, allow_pickle=False) as problem:
            h = np.asarray(problem["h"], dtype=np.float64)
            j = np.asarray(problem["J"], dtype=np.float64)
        with np.load(response_path, allow_pickle=False) as response:
            if "samples" not in response:
                raise ValueError("%s does not contain logical samples" % response_path)
            samples = np.asarray(response["samples"], dtype=np.int8)
        if samples.shape[1] != h.size:
            raise ValueError("Candidate response is not in logical sample space")
        problems.append((h, j, samples))
        stats = response_statistics(h, j, samples)
        per_instance.append(stats)
        if reference_dir is not None:
            with np.load(reference_dir / (instance["id"] + ".npz"), allow_pickle=False) as reference:
                reference_samples = np.asarray(reference["samples"], dtype=np.int8)
            reference_metrics.append(compare_response_pair(h, j, reference_samples, samples))
    beta = fit_shared_effective_temperature_pseudolikelihood(
        problems,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return {
        "hardware_gain": float(candidate["gain"]),
        "vs_manifest": str(manifest_path.resolve()),
        "responses_dir": str(responses_dir.resolve()),
        "beta_eff": beta["beta_eff"],
        "beta_abs_error": abs(float(beta["beta_eff"]) - float(target_beta)),
        "mean_energy": float(np.mean([item["energies"].mean() for item in per_instance])),
        "mean_abs_magnetization": float(
            np.mean([np.abs(item["magnetization"]).mean() for item in per_instance])
        ),
        "mean_edge_moment_abs": float(
            np.mean([
                np.abs(item["edge_moments"]).mean()
                if item["edge_moments"].size else 0.0
                for item in per_instance
            ])
        ),
        "mean_unique_state_fraction": float(
            np.mean([item["unique_state_fraction"] for item in per_instance])
        ),
        "n_instances": len(problems),
        "n_reads_min": int(min(item[2].shape[0] for item in problems)),
        "n_reads_max": int(max(item[2].shape[0] for item in problems)),
    }
    if reference_metrics:
        result["reference_dir"] = str(reference_dir.resolve())
        result["reference_energy_wasserstein"] = float(
            np.mean([item["energy_wasserstein"] for item in reference_metrics])
        )
        result["reference_magnetization_mae"] = float(
            np.mean([item["magnetization_mae"] for item in reference_metrics])
        )
        result["reference_edge_moment_mae"] = float(
            np.mean([item["edge_moment_mae"] for item in reference_metrics])
        )
        result["reference_beta_eff_error"] = float(
            np.mean([abs(item["candidate_beta_eff"] - item["reference_beta_eff"]) for item in reference_metrics])
        )


def select_gain_candidate(candidates, target_beta=1.0):
    if not candidates:
        raise ValueError("At least one gain candidate is required")
    if all("reference_energy_wasserstein" in item for item in candidates):
        return min(
            candidates,
            key=lambda item: (
                float(item["reference_energy_wasserstein"]),
                float(item["reference_beta_eff_error"]),
                float(item["reference_edge_moment_mae"]),
                float(item["reference_magnetization_mae"]),
                -float(item["mean_unique_state_fraction"]),
            ),
        )
    return min(
        candidates,
        key=lambda item: (
            float(item["beta_abs_error"]),
            -float(item["mean_unique_state_fraction"]),
            float(abs(item["mean_abs_magnetization"])),
        ),
    )


def write_scaled_manifest(test_manifest, output_dir, coefficient_scale, calibration):
    instances = json.loads(Path(test_manifest).read_text(encoding="utf-8"))
    payload_dir = output_dir / "test_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    calibrated_instances = []
    for instance in instances:
        payload_path = resolve_repo_path(instance["platform_payload"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        output_path = payload_dir / (instance["id"] + ".json")
        output_path.write_text(
            json.dumps(
                scale_payload(payload, coefficient_scale, calibration),
                indent=2,
            ),
            encoding="utf-8",
        )
        calibrated = dict(instance)
        calibrated["uncalibrated_platform_payload"] = instance["platform_payload"]
        calibrated["platform_payload"] = str(output_path.resolve())
        calibrated["temperature_coefficient_scale"] = coefficient_scale
        if "hardware_matrix" in payload:
            expected_gain = float(calibration["hardware_gain"])
            if abs(float(payload["hardware_gain"]) - expected_gain) > 1e-12:
                raise ValueError(
                    "TEST payload gain does not match the selected VS global gain"
                )
            calibrated["hardware_gain"] = expected_gain
        calibrated_instances.append(calibrated)
    manifest_path = output_dir / "test_calibrated_manifest.json"
    manifest_path.write_text(
        json.dumps(calibrated_instances, indent=2), encoding="utf-8"
    )
    return manifest_path


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs-manifest", type=Path, default=None)
    parser.add_argument("--vs-responses-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--target-beta", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=3.0)
    parser.add_argument("--beta-points", type=int, default=296)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--gain-candidates",
        type=Path,
        default=None,
        help=(
            "JSON list of {gain, vs_manifest, responses_dir}; each candidate "
            "must already have physical VS responses."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "export" / "bosonic_calibration",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_beta <= 0 or args.beta_min <= 0:
        raise ValueError("Temperature parameters must be positive")
    if args.beta_max <= args.beta_min or args.beta_points < 2:
        raise ValueError("Invalid beta grid")
    if args.gain_candidates is not None:
        candidate_specs = json.loads(args.gain_candidates.read_text(encoding="utf-8"))
        evaluated = [
            evaluate_gain_candidate(
                candidate,
                target_beta=args.target_beta,
                bootstrap_repetitions=args.bootstrap_repetitions,
                seed=args.seed + offset,
            )
            for offset, candidate in enumerate(candidate_specs)
        ]
        selected = select_gain_candidate(evaluated, args.target_beta)
        args.vs_manifest = Path(selected["vs_manifest"])
        args.vs_responses_dir = Path(selected["responses_dir"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "gain_candidate_selection.json").write_text(
            json.dumps(
                {
                    "target_beta": args.target_beta,
                    "selected": selected,
                    "candidates": evaluated,
                    "selection_rule": (
                        "with a floating-point SA reference, minimize energy "
                        "Wasserstein, beta error, edge-moment MAE, and "
                        "magnetization MAE; otherwise minimize abs(beta_eff-" 
                        "target_beta), then maximize state diversity and "
                        "minimize absolute magnetization"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.vs_manifest is None or args.vs_responses_dir is None:
        raise ValueError("Provide --vs-manifest/--vs-responses-dir or --gain-candidates")
    _, problems, response_paths = load_calibration_problems(
        args.vs_manifest, args.vs_responses_dir
    )
    calibration = fit_shared_effective_temperature_pseudolikelihood(
        problems,
        beta_grid=np.linspace(args.beta_min, args.beta_max, args.beta_points),
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    with np.load(resolve_repo_path(json.loads(Path(args.vs_manifest).read_text(encoding="utf-8"))[0]["path"]), allow_pickle=False) as first_problem:
        is_hardware_problem = "hardware_matrix" in first_problem
    if is_hardware_problem:
        manifest_instances = json.loads(Path(args.vs_manifest).read_text(encoding="utf-8"))
        gains = set()
        for instance in manifest_instances:
            payload = json.loads(
                resolve_repo_path(instance["platform_payload"]).read_text(encoding="utf-8")
            )
            gains.add(float(payload["hardware_gain"]))
        if len(gains) != 1:
            raise ValueError("VS hardware manifest must use one global hardware_gain")
        coefficient_scale = 1.0
        calibration["platform_has_no_direct_beta"] = True
        calibration["hardware_gain"] = gains.pop()
        calibration["gain_selection_frozen"] = True
    else:
        coefficient_scale = float(args.target_beta / calibration["beta_eff"])
    calibration.update(
        {
            "fit_split": "VS",
            "test_responses_used": False,
            "target_beta": float(args.target_beta),
            "coefficient_scale": coefficient_scale,
            "vs_manifest": str(args.vs_manifest.resolve()),
            "vs_response_paths": response_paths,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "effective_temperature.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    print(
        "VS beta_eff %.4f [%.4f, %.4f] -> coefficient scale %.4f"
        % (
            calibration["beta_eff"],
            calibration.get("beta_eff_ci_2.5", calibration["beta_eff"]),
            calibration.get("beta_eff_ci_97.5", calibration["beta_eff"]),
            coefficient_scale,
        )
    )
    if args.test_manifest is not None:
        manifest_path = write_scaled_manifest(
            args.test_manifest,
            args.output_dir,
            coefficient_scale,
            calibration,
        )
        print("Wrote frozen TEST payload manifest to %s" % manifest_path.resolve())
    print("Wrote calibration to %s" % calibration_path.resolve())


if __name__ == "__main__":
    main()
