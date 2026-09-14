# -*- coding: utf-8 -*-

"""Aggregate multi-seed unified results without selecting on TEST."""

import argparse
import csv
import itertools
import re
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.compare_scenarios import (
    energy_score_by_day,
    load_scenarios,
    variogram_score_by_day,
)
from GEFcom2014.forecast_quality.unified_postprocessing import (
    ROOT_DIR,
    TRACK_GROUPS,
    VARIANTS,
    load_targets,
)


METRIC_COLUMNS = (
    "mean_qs",
    "mean_crps",
    "reliability_mae",
    "energy_score",
    "variogram_score",
    "corr_mae",
    "ramp_quantile_mae",
)


def main():
    args = parse_args()
    input_dir = args.input_dir or ROOT_DIR / "export" / "unified_postprocessing" / args.tag
    rows = read_metric_rows(input_dir / "metrics.csv")
    selected = select_variants_on_validation(rows)
    aggregate_rows = aggregate_selected_test_metrics(rows, selected)
    paired_rows = paired_crps_inference(
        rows,
        selected,
        tag=args.tag,
        bootstrap_repetitions=args.bootstrap_repetitions,
        permutation_repetitions=args.permutation_repetitions,
        seed=args.seed,
        dataset_bundle=args.dataset_bundle,
    )
    write_csv(input_dir / "selected_variants.csv", [
        {"family": family, "selected_variant": variant} for family, variant in selected.items()
    ])
    write_csv(input_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(input_dir / "paired_crps_tests.csv", paired_rows)
    score_rows = paired_score_inference(
        rows,
        selected,
        tag=args.tag,
        bootstrap_repetitions=args.bootstrap_repetitions,
        permutation_repetitions=args.permutation_repetitions,
        seed=args.seed,
        dataset_bundle=args.dataset_bundle,
    )
    write_csv(input_dir / "paired_score_tests.csv", score_rows)
    print("Selected on VS: %s" % selected)
    print("Wrote aggregate results to %s" % input_dir)


def read_metric_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["family"] = model_family(row["model"])
        for column in METRIC_COLUMNS:
            row[column] = float(row[column])
        row["n_scenarios"] = int(row["n_scenarios"])
    return rows


def model_family(model_name):
    return re.sub(r"\s*\(seed\s+\d+\)$", "", model_name)


def select_variants_on_validation(rows):
    families = sorted({row["family"] for row in rows})
    selected = {}
    priority = {variant: index for index, variant in enumerate(VARIANTS)}
    for family in families:
        scores = []
        for variant in VARIANTS:
            values = [
                row["mean_crps"] for row in rows
                if row["family"] == family and row["split"] == "VS" and row["variant"] == variant
            ]
            if values:
                scores.append((float(np.mean(values)), priority[variant], variant))
        best_score = min(score for score, _, _ in scores)
        numerically_tied = [
            item for item in scores if item[0] <= best_score + 1e-12
        ]
        selected[family] = min(numerically_tied, key=lambda item: item[1])[2]
    return selected


def aggregate_selected_test_metrics(rows, selected):
    output = []
    for family, variant in selected.items():
        family_rows = [
            row for row in rows
            if row["family"] == family and row["split"] == "TEST" and row["variant"] == variant
        ]
        result = {
            "family": family,
            "selected_variant": variant,
            "n_seeds": len(family_rows),
        }
        for metric in METRIC_COLUMNS:
            values = np.asarray([row[metric] for row in family_rows], dtype=np.float64)
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        output.append(result)
    return sorted(output, key=lambda row: row["mean_crps_mean"])


def paired_crps_inference(rows, selected, tag, bootstrap_repetitions, permutation_repetitions, seed,
                          dataset_bundle=None):
    target = load_targets(tag, "TEST", dataset_bundle).reshape(-1)
    n_groups = TRACK_GROUPS.get(tag, 1)
    family_scores = {}
    for family, variant in selected.items():
        paths = [
            Path(row["scenario_file"]) for row in rows
            if row["family"] == family and row["split"] == "TEST" and row["variant"] == variant
        ]
        seed_scores = []
        for path in paths:
            scenarios = load_scenarios(path, target.size)
            seed_scores.append(crps_by_date_block(scenarios, target, n_groups))
        family_scores[family] = np.mean(seed_scores, axis=0)

    rng = np.random.default_rng(seed)
    output = []
    for family_a, family_b in itertools.combinations(sorted(family_scores), 2):
        differences = family_scores[family_a] - family_scores[family_b]
        inference = paired_mean_inference(
            differences,
            bootstrap_repetitions=bootstrap_repetitions,
            permutation_repetitions=permutation_repetitions,
            rng=rng,
        )
        output.append({
            "model_a": family_a,
            "variant_a": selected[family_a],
            "model_b": family_b,
            "variant_b": selected[family_b],
            "mean_crps_difference_a_minus_b": inference["mean_difference"],
            "ci_2.5": inference["ci_2.5"],
            "ci_97.5": inference["ci_97.5"],
            "paired_permutation_p": inference["p_value"],
            "n_date_blocks": differences.size,
        })
    return output


def paired_score_inference(rows, selected, tag, bootstrap_repetitions, permutation_repetitions, seed,
                           dataset_bundle=None):
    target_matrix = load_targets(tag, "TEST", dataset_bundle)
    target = target_matrix.reshape(-1)
    n_groups = TRACK_GROUPS.get(tag, 1)
    family_scores = {}
    for family, variant in selected.items():
        paths = [
            Path(row["scenario_file"]) for row in rows
            if row["family"] == family and row["split"] == "TEST" and row["variant"] == variant
        ]
        score_sets = {"crps": [], "energy_score": [], "variogram_score": [], "ramp_crps": []}
        for path in paths:
            scenarios = load_scenarios(path, target.size)
            blocks = score_blocks_by_date(scenarios, target_matrix, n_groups)
            for metric, values in blocks.items():
                score_sets[metric].append(values)
        family_scores[family] = {
            metric: np.mean(seed_values, axis=0) for metric, seed_values in score_sets.items()
        }

    rng = np.random.default_rng(seed)
    output = []
    for family_a, family_b in itertools.combinations(sorted(family_scores), 2):
        for metric in ("crps", "energy_score", "variogram_score", "ramp_crps"):
            differences = family_scores[family_a][metric] - family_scores[family_b][metric]
            inference = paired_mean_inference(
                differences,
                bootstrap_repetitions=bootstrap_repetitions,
                permutation_repetitions=permutation_repetitions,
                rng=rng,
            )
            output.append({
                "metric": metric,
                "model_a": family_a,
                "variant_a": selected[family_a],
                "model_b": family_b,
                "variant_b": selected[family_b],
                "mean_difference_a_minus_b": inference["mean_difference"],
                "ci_2.5": inference["ci_2.5"],
                "ci_97.5": inference["ci_97.5"],
                "paired_permutation_p": inference["p_value"],
                "n_date_blocks": differences.size,
            })
    return output


def score_blocks_by_date(scenarios, target_matrix, n_groups):
    target_matrix = np.asarray(target_matrix, dtype=np.float64)
    n_days = target_matrix.shape[0]
    days_per_group = n_days // n_groups
    energy = energy_score_by_day(scenarios, target_matrix).reshape(
        n_groups, days_per_group
    ).mean(axis=0)
    variogram = variogram_score_by_day(scenarios, target_matrix).reshape(
        n_groups, days_per_group
    ).mean(axis=0)
    return {
        "crps": crps_by_date_block(scenarios, target_matrix.reshape(-1), n_groups),
        "energy_score": energy,
        "variogram_score": variogram,
        "ramp_crps": ramp_crps_by_date_block(scenarios, target_matrix, n_groups),
    }


def crps_by_date_block(scenarios, target, n_groups):
    samples = np.asarray(scenarios, dtype=np.float64)
    observations = np.asarray(target, dtype=np.float64).reshape(-1)
    sorted_samples = np.sort(samples, axis=1)
    n_scenarios = samples.shape[1]
    coefficients = 2 * np.arange(1, n_scenarios + 1) - n_scenarios - 1
    observation_term = np.mean(np.abs(samples - observations[:, None]), axis=1)
    ensemble_term = np.sum(sorted_samples * coefficients[None, :], axis=1) / n_scenarios ** 2
    period_scores = observation_term - ensemble_term
    days_per_group = period_scores.size // (n_groups * 24)
    return period_scores.reshape(n_groups, days_per_group, 24).mean(axis=(0, 2))


def ramp_crps_by_date_block(scenarios, target_matrix, n_groups):
    samples = np.asarray(scenarios, dtype=np.float64)
    observations = np.asarray(target_matrix, dtype=np.float64)
    n_days = observations.shape[0]
    n_scenarios = samples.shape[1]
    sample_ramps = np.diff(samples.reshape(n_days, 24, n_scenarios), axis=1)
    observed_ramps = np.diff(observations, axis=1)
    sorted_samples = np.sort(sample_ramps, axis=2)
    coefficients = 2 * np.arange(1, n_scenarios + 1) - n_scenarios - 1
    observation_term = np.mean(
        np.abs(sample_ramps - observed_ramps[:, :, None]), axis=2
    )
    ensemble_term = np.sum(
        sorted_samples * coefficients[None, None, :], axis=2
    ) / n_scenarios ** 2
    days_per_group = n_days // n_groups
    return (observation_term - ensemble_term).reshape(
        n_groups, days_per_group, 23
    ).mean(axis=(0, 2))


def paired_mean_inference(differences, bootstrap_repetitions, permutation_repetitions, rng):
    differences = np.asarray(differences, dtype=np.float64)
    observed = float(differences.mean())
    bootstrap_indices = rng.integers(
        0, differences.size, size=(bootstrap_repetitions, differences.size)
    )
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    signs = rng.choice(
        np.array([-1.0, 1.0]), size=(permutation_repetitions, differences.size)
    )
    null_means = (signs * differences[None, :]).mean(axis=1)
    p_value = (np.count_nonzero(np.abs(null_means) >= abs(observed)) + 1.0) / (
        permutation_repetitions + 1.0
    )
    return {
        "mean_difference": observed,
        "ci_2.5": float(np.quantile(bootstrap_means, 0.025)),
        "ci_97.5": float(np.quantile(bootstrap_means, 0.975)),
        "p_value": float(p_value),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate unified multi-seed results.")
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--permutation-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
