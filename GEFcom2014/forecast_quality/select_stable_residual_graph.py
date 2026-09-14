# -*- coding: utf-8 -*-

"""VS-only gate for the LS stability-selected residual-MI graph."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .aggregate_unified_results import paired_mean_inference, score_blocks_by_date
from .compare_scenarios import load_scenarios
from .unified_postprocessing import ROOT_DIR, TRACK_GROUPS, load_targets


FAMILIES = {
    "wind": {
        "j0": "abl_lanchor_ind",
        "temporal_mi": "lanchor",
    },
    "opsd-wind": {
        "j0": "abl_anchor_ind",
        "temporal_mi": "anchor",
    },
}
CANDIDATE = "candidate"
SCORE_METRICS = ("crps", "energy_score", "variogram_score", "ramp_crps")
DEPENDENCE_METRICS = ("variogram_score", "ramp_crps")
REFERENCES = ("j0", "temporal_mi")
MAX_RELATIVE_REGRESSION = 0.01


def scenario_path(track, label, seed):
    name = "scenarios_%s_QBMVAE_2_%s_sa_%d_100_VS.pickle" % (
        track,
        label,
        seed,
    )
    return ROOT_DIR / "export" / ("qbm_vae_%s" % track) / name


def load_score_blocks(track, label, seed, dataset_bundle):
    targets = load_targets(track, "VS", dataset_bundle)
    scenarios = load_scenarios(scenario_path(track, label, seed), targets.size)
    return score_blocks_by_date(scenarios, targets, TRACK_GROUPS[track])


def dependence_composite(metric_means, baseline_means):
    return float(np.mean([
        metric_means[metric] / baseline_means[metric]
        for metric in DEPENDENCE_METRICS
    ]))


def evaluate_gate(cell_means, per_seed_means, max_relative_regression=MAX_RELATIVE_REGRESSION):
    checks = []
    for track in sorted(cell_means):
        cells = cell_means[track]
        baseline = cells["j0"]
        candidate_composite = dependence_composite(
            cells[CANDIDATE], baseline
        )
        for reference in REFERENCES:
            reference_composite = dependence_composite(cells[reference], baseline)
            checks.append({
                "track": track,
                "check": "dependence_composite_vs_%s" % reference,
                "value": candidate_composite - reference_composite,
                "passed": candidate_composite < reference_composite,
            })

        best_reference = {
            metric: min(cells[reference][metric] for reference in REFERENCES)
            for metric in SCORE_METRICS
        }
        for metric in SCORE_METRICS:
            relative_regression = (
                cells[CANDIDATE][metric] / best_reference[metric] - 1.0
            )
            checks.append({
                "track": track,
                "check": "%s_relative_regression" % metric,
                "value": relative_regression,
                "passed": relative_regression <= max_relative_regression,
            })

        favorable_seeds = 0
        for seed, seed_cells in per_seed_means[track].items():
            candidate_seed_composite = dependence_composite(
                seed_cells[CANDIDATE], seed_cells["j0"]
            )
            temporal_seed_composite = dependence_composite(
                seed_cells["temporal_mi"], seed_cells["j0"]
            )
            favorable_seeds += candidate_seed_composite < temporal_seed_composite
        checks.append({
            "track": track,
            "check": "seed_direction_vs_temporal_mi",
            "value": favorable_seeds,
            "passed": favorable_seeds >= 2,
        })
    return checks, all(check["passed"] for check in checks)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.inference_seed)
    bundles = {"wind": None, "opsd-wind": Path(args.opsd_bundle)}
    cell_means = {}
    per_seed_means = {}
    cell_rows = []
    seed_rows = []
    contrast_rows = []

    for track, base_families in FAMILIES.items():
        families = dict(base_families)
        families[CANDIDATE] = args.candidate_label
        per_seed_means[track] = {seed: {} for seed in args.seeds}
        blocks = {}
        for family, label in families.items():
            blocks[family] = []
            for seed in args.seeds:
                score_blocks = load_score_blocks(
                    track, label, seed, bundles[track]
                )
                blocks[family].append(score_blocks)
                per_seed_means[track][seed][family] = {
                    metric: float(score_blocks[metric].mean())
                    for metric in SCORE_METRICS
                }
                for metric in SCORE_METRICS:
                    seed_rows.append({
                        "track": track,
                        "family": family,
                        "seed": seed,
                        "metric": metric,
                        "mean": per_seed_means[track][seed][family][metric],
                    })

        cell_means[track] = {}
        for family in families:
            cell_means[track][family] = {}
            for metric in SCORE_METRICS:
                values = np.asarray([
                    per_seed_means[track][seed][family][metric]
                    for seed in args.seeds
                ])
                cell_means[track][family][metric] = float(values.mean())
                cell_rows.append({
                    "track": track,
                    "family": family,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "seed_std": float(values.std(ddof=1)),
                })

        for reference in REFERENCES:
            for metric in SCORE_METRICS:
                candidate_blocks = np.mean(
                    [item[metric] for item in blocks[CANDIDATE]], axis=0
                )
                reference_blocks = np.mean(
                    [item[metric] for item in blocks[reference]], axis=0
                )
                inference = paired_mean_inference(
                    candidate_blocks - reference_blocks,
                    bootstrap_repetitions=args.repetitions,
                    permutation_repetitions=args.repetitions,
                    rng=rng,
                )
                contrast_rows.append({
                    "track": track,
                    "contrast": "%s_minus_%s" % (args.candidate_label, reference),
                    "metric": metric,
                    "mean_difference": inference["mean_difference"],
                    "ci_2.5": inference["ci_2.5"],
                    "ci_97.5": inference["ci_97.5"],
                    "paired_sign_permutation_p": inference["p_value"],
                    "n_vs_date_blocks": candidate_blocks.size,
                })

    checks, eligible = evaluate_gate(cell_means, per_seed_means)
    write_csv(output_dir / "vs_cell_metrics.csv", cell_rows)
    write_csv(output_dir / "vs_seed_metrics.csv", seed_rows)
    write_csv(output_dir / "vs_contrasts.csv", contrast_rows)
    write_csv(output_dir / "vs_gate_checks.csv", checks)
    with (output_dir / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_split": "VS",
                "candidate_label": args.candidate_label,
                "test_scenarios_generated": False,
                "seeds": args.seeds,
                "max_relative_regression": MAX_RELATIVE_REGRESSION,
                "eligible_for_random_control": eligible,
                "checks": checks,
            },
            handle,
            indent=2,
        )
    print("Candidate %s eligible for random control: %s" % (args.candidate_label, eligible))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--inference-seed", type=int, default=2026)
    parser.add_argument("--candidate-label", default="dev_sresmi")
    parser.add_argument(
        "--opsd-bundle",
        type=Path,
        default=ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "export" / "stable_residual_graph_selection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
