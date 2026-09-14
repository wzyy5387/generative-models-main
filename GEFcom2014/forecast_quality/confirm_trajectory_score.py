# -*- coding: utf-8 -*-

"""Independent VS-only confirmation for trajectory-score regularization."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .aggregate_unified_results import paired_mean_inference
from .anchor_coupling_ablation import add_holm_adjustment
from .select_stable_residual_graph import (
    DEPENDENCE_METRICS,
    SCORE_METRICS,
    load_score_blocks,
)
from .unified_postprocessing import ROOT_DIR


LABELS = {"candidate": "dev_tsr", "control": "dev_ctrl"}
MAX_RELATIVE_REGRESSION = 0.01


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = {
        "wind": None,
        "opsd-wind": Path(args.opsd_bundle),
    }
    rng = np.random.default_rng(args.inference_seed)
    seed_rows = []
    contrast_rows = []
    checks = []

    for track in ("wind", "opsd-wind"):
        blocks = {family: {} for family in LABELS}
        for family, label in LABELS.items():
            for seed in args.seeds:
                blocks[family][seed] = load_score_blocks(
                    track, label, seed, bundles[track]
                )

        favorable = 0
        seed_metric_means = {family: {} for family in LABELS}
        for seed in args.seeds:
            for family in LABELS:
                seed_metric_means[family][seed] = {
                    metric: float(blocks[family][seed][metric].mean())
                    for metric in SCORE_METRICS
                }
            dependence_ratio = float(np.mean([
                seed_metric_means["candidate"][seed][metric]
                / seed_metric_means["control"][seed][metric]
                for metric in DEPENDENCE_METRICS
            ]))
            favorable += dependence_ratio < 1.0
            seed_rows.append({
                "track": track,
                "seed": seed,
                "dependence_ratio_candidate_over_control": dependence_ratio,
                "favors_candidate": dependence_ratio < 1.0,
            })

        checks.append({
            "track": track,
            "check": "at_least_two_of_three_confirmation_seeds",
            "value": favorable,
            "passed": favorable >= 2,
        })
        candidate_composite = np.mean([
            np.mean([
                seed_metric_means["candidate"][seed][metric]
                / seed_metric_means["control"][seed][metric]
                for metric in DEPENDENCE_METRICS
            ])
            for seed in args.seeds
        ])
        checks.append({
            "track": track,
            "check": "mean_dependence_composite",
            "value": float(candidate_composite - 1.0),
            "passed": bool(candidate_composite < 1.0),
        })

        for metric in SCORE_METRICS:
            candidate_mean = np.mean([
                seed_metric_means["candidate"][seed][metric] for seed in args.seeds
            ])
            control_mean = np.mean([
                seed_metric_means["control"][seed][metric] for seed in args.seeds
            ])
            relative_difference = float(candidate_mean / control_mean - 1.0)
            checks.append({
                "track": track,
                "check": "%s_relative_regression" % metric,
                "value": relative_difference,
                "passed": bool(relative_difference <= MAX_RELATIVE_REGRESSION),
            })
            candidate_dates = np.mean(
                [blocks["candidate"][seed][metric] for seed in args.seeds], axis=0
            )
            control_dates = np.mean(
                [blocks["control"][seed][metric] for seed in args.seeds], axis=0
            )
            inference = paired_mean_inference(
                candidate_dates - control_dates,
                bootstrap_repetitions=args.repetitions,
                permutation_repetitions=args.repetitions,
                rng=rng,
            )
            contrast_rows.append({
                "track": track,
                "metric": metric,
                "mean_difference": inference["mean_difference"],
                "ci_2.5": inference["ci_2.5"],
                "ci_97.5": inference["ci_97.5"],
                "paired_sign_permutation_p": inference["p_value"],
            })

    add_holm_adjustment(contrast_rows)
    confirmed = bool(all(check["passed"] for check in checks))
    write_csv(output_dir / "confirmation_seed_directions.csv", seed_rows)
    write_csv(output_dir / "confirmation_contrasts.csv", contrast_rows)
    write_csv(output_dir / "confirmation_checks.csv", checks)
    with (output_dir / "confirmation.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": "VS",
                "test_scenarios_generated": False,
                "predeclared_confirmation_seeds": args.seeds,
                "required_favorable_seeds_per_track": 2,
                "max_relative_regression": MAX_RELATIVE_REGRESSION,
                "confirmed": confirmed,
                "checks": checks,
            },
            handle,
            indent=2,
        )
    print("Trajectory-score regularization confirmed: %s" % confirmed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--inference-seed", type=int, default=2027)
    parser.add_argument(
        "--opsd-bundle",
        type=Path,
        default=ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "export" / "trajectory_score_confirmation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
