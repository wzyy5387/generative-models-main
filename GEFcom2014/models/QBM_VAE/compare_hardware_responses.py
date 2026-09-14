# -*- coding: utf-8 -*-

"""Compare two samplers on identical frozen Ising instances."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .calibration import fit_effective_temperature_pseudolikelihood
from .ising import ising_energy


ROOT_DIR = Path(__file__).resolve().parents[3]


def response_statistics(h, j, samples):
    energies = ising_energy(samples, h, j)
    edge_i, edge_j = np.where(np.triu(np.abs(j) > 1e-12, k=1))
    edge_moments = (
        (samples[:, edge_i] * samples[:, edge_j]).mean(axis=0)
        if edge_i.size
        else np.asarray([], dtype=np.float64)
    )
    calibration = fit_effective_temperature_pseudolikelihood(h, j, samples)
    return {
        "energies": energies,
        "magnetization": samples.mean(axis=0),
        "edge_moments": edge_moments,
        "beta_eff": calibration["beta_eff"],
        "unique_state_fraction": float(
            np.unique(samples, axis=0).shape[0] / samples.shape[0]
        ),
    }


def quantile_wasserstein(first, second, points=1001):
    quantiles = np.linspace(0.0, 1.0, points)
    return float(
        np.mean(
            np.abs(
                np.quantile(first, quantiles)
                - np.quantile(second, quantiles)
            )
        )
    )


def compare_response_pair(h, j, reference_samples, candidate_samples):
    reference = response_statistics(h, j, reference_samples)
    candidate = response_statistics(h, j, candidate_samples)
    edge_difference = np.abs(
        candidate["edge_moments"] - reference["edge_moments"]
    )
    return {
        "reference_mean_energy": float(reference["energies"].mean()),
        "candidate_mean_energy": float(candidate["energies"].mean()),
        "mean_energy_difference": float(
            candidate["energies"].mean() - reference["energies"].mean()
        ),
        "energy_wasserstein": quantile_wasserstein(
            reference["energies"], candidate["energies"]
        ),
        "magnetization_mae": float(
            np.abs(
                candidate["magnetization"] - reference["magnetization"]
            ).mean()
        ),
        "edge_moment_mae": float(edge_difference.mean()) if edge_difference.size else 0.0,
        "reference_beta_eff": float(reference["beta_eff"]),
        "candidate_beta_eff": float(candidate["beta_eff"]),
        "beta_eff_difference": float(
            candidate["beta_eff"] - reference["beta_eff"]
        ),
        "reference_unique_state_fraction": reference["unique_state_fraction"],
        "candidate_unique_state_fraction": candidate["unique_state_fraction"],
    }


def paired_improvement_statistics(
    baseline_values,
    candidate_values,
    repetitions=10000,
    seed=0,
):
    """Estimate paired error reduction and test a zero-improvement null."""
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    if baseline_values.shape != candidate_values.shape:
        raise ValueError("Paired metric arrays must have identical shapes")
    if baseline_values.ndim != 1 or baseline_values.size == 0:
        raise ValueError("Paired metric arrays must be non-empty vectors")
    improvements = baseline_values - candidate_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        improvements.size,
        size=(repetitions, improvements.size),
    )
    bootstrap_means = improvements[indices].mean(axis=1)
    observed = float(improvements.mean())
    signs = rng.choice(
        (-1.0, 1.0),
        size=(repetitions, improvements.size),
    )
    permuted_means = (signs * improvements).mean(axis=1)
    p_value = float(
        (1 + np.count_nonzero(np.abs(permuted_means) >= abs(observed)))
        / (repetitions + 1)
    )
    return {
        "mean_improvement": observed,
        "ci_2.5": float(np.quantile(bootstrap_means, 0.025)),
        "ci_97.5": float(np.quantile(bootstrap_means, 0.975)),
        "paired_sign_flip_p": p_value,
        "positive_means_candidate_is_closer": True,
    }


def load_samples(path, n_bits):
    with np.load(path, allow_pickle=False) as response:
        if "samples" not in response:
            raise ValueError("%s does not contain samples" % path)
        samples = np.asarray(response["samples"], dtype=np.int8)
        latency = (
            float(np.asarray(response["latency_s"]).reshape(-1)[0])
            if "latency_s" in response
            else float("nan")
        )
    if samples.ndim != 2 or samples.shape[1] != n_bits:
        raise ValueError("%s has incompatible sample shape" % path)
    if not np.isin(samples, (-1, 1)).all():
        raise ValueError("%s contains non-Ising samples" % path)
    return samples, latency


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-label", default="SA")
    parser.add_argument("--candidate-label", default="CIM")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Optional uncalibrated/control responses for paired improvement tests.",
    )
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--stat-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "export" / "bosonic_comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    instances = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for instance in instances:
        problem_path = Path(instance["path"])
        if not problem_path.is_absolute():
            problem_path = ROOT_DIR / problem_path
        with np.load(problem_path, allow_pickle=False) as problem:
            h = np.asarray(problem["h"], dtype=np.float64)
            j = np.asarray(problem["J"], dtype=np.float64)
        reference_path = args.reference_dir / (instance["id"] + ".npz")
        candidate_path = args.candidate_dir / (instance["id"] + ".npz")
        reference_samples, reference_latency = load_samples(reference_path, h.size)
        candidate_samples, candidate_latency = load_samples(candidate_path, h.size)
        row = {
            "instance_id": instance["id"],
            "reference_label": args.reference_label,
            "candidate_label": args.candidate_label,
            "reference_reads": int(reference_samples.shape[0]),
            "candidate_reads": int(candidate_samples.shape[0]),
            "reference_latency_s": reference_latency,
            "candidate_latency_s": candidate_latency,
        }
        row.update(
            compare_response_pair(
                h, j, reference_samples, candidate_samples
            )
        )
        if args.baseline_dir is not None:
            baseline_path = args.baseline_dir / (instance["id"] + ".npz")
            baseline_samples, baseline_latency = load_samples(
                baseline_path, h.size
            )
            baseline_metrics = compare_response_pair(
                h, j, reference_samples, baseline_samples
            )
            row["baseline_label"] = args.baseline_label
            row["baseline_reads"] = int(baseline_samples.shape[0])
            row["baseline_latency_s"] = baseline_latency
            for metric, value in baseline_metrics.items():
                row["baseline_" + metric] = value
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paired_sampler_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [
        "mean_energy_difference",
        "energy_wasserstein",
        "magnetization_mae",
        "edge_moment_mae",
        "beta_eff_difference",
    ]
    summary = {
        "manifest": str(args.manifest.resolve()),
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "n_instances": len(rows),
        "mean_metrics": {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in metric_names
        },
        "test_targets_used": False,
    }
    if args.baseline_dir is not None:
        error_metrics = {
            "absolute_mean_energy_error": lambda row, prefix="": abs(
                row[prefix + "mean_energy_difference"]
            ),
            "energy_wasserstein": lambda row, prefix="": row[
                prefix + "energy_wasserstein"
            ],
            "magnetization_mae": lambda row, prefix="": row[
                prefix + "magnetization_mae"
            ],
            "edge_moment_mae": lambda row, prefix="": row[
                prefix + "edge_moment_mae"
            ],
            "absolute_beta_eff_error": lambda row, prefix="": abs(
                row[prefix + "beta_eff_difference"]
            ),
        }
        summary["baseline_label"] = args.baseline_label
        summary["candidate_vs_baseline"] = {}
        for offset, (metric, accessor) in enumerate(error_metrics.items()):
            baseline_values = [
                accessor(row, "baseline_") for row in rows
            ]
            candidate_values = [accessor(row) for row in rows]
            summary["candidate_vs_baseline"][metric] = (
                paired_improvement_statistics(
                    baseline_values,
                    candidate_values,
                    repetitions=args.stat_repetitions,
                    seed=args.seed + offset,
                )
            )
        summary["stat_repetitions"] = args.stat_repetitions
    json_path = args.output_dir / "paired_sampler_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote %s and %s" % (csv_path.resolve(), json_path.resolve()))


if __name__ == "__main__":
    main()
