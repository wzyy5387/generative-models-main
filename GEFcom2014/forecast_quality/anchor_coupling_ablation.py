# -*- coding: utf-8 -*-

"""Analyze the matched 2x2 forecast-anchor by Ising-coupling ablation."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from .aggregate_unified_results import (
    METRIC_COLUMNS,
    model_family,
    paired_mean_inference,
    score_blocks_by_date,
)
from .compare_scenarios import load_scenarios
from .unified_postprocessing import ROOT_DIR, TRACK_GROUPS, VARIANTS, load_targets


CELLS = {
    "no_anchor_independent": "Independent-prior VAE ablation (no anchor, J=0)",
    "no_anchor_coupled": "BM-VAE ablation (no anchor, Temporal-MI J)",
    "anchor_independent": "Forecast-Anchored Independent-prior VAE ablation (J=0)",
    "anchor_coupled": "QBM-VAE + Learned Forecast Anchor",
}

CONTRASTS = {
    "coupling_without_anchor": {
        "no_anchor_coupled": 1.0,
        "no_anchor_independent": -1.0,
    },
    "coupling_with_anchor": {
        "anchor_coupled": 1.0,
        "anchor_independent": -1.0,
    },
    "anchor_without_coupling": {
        "anchor_independent": 1.0,
        "no_anchor_independent": -1.0,
    },
    "anchor_with_coupling": {
        "anchor_coupled": 1.0,
        "no_anchor_coupled": -1.0,
    },
    "anchor_coupling_interaction": {
        "anchor_coupled": 1.0,
        "anchor_independent": -1.0,
        "no_anchor_coupled": -1.0,
        "no_anchor_independent": 1.0,
    },
}


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["family"] = model_family(row["model"])
        match = re.search(r"\(seed\s+(\d+)\)$", row["model"])
        row["seed"] = int(match.group(1)) if match else None
        for metric in METRIC_COLUMNS:
            row[metric] = float(row[metric])
    return rows


def common_seeds(rows):
    seed_sets = []
    for family in CELLS.values():
        seed_sets.append(
            {
                row["seed"]
                for row in rows
                if row["family"] == family and row["seed"] is not None
            }
        )
    common = set.intersection(*seed_sets)
    if not common:
        raise ValueError("The four ablation cells have no common completed seed")
    return sorted(common)


def select_variants(rows, seeds):
    priority = {variant: index for index, variant in enumerate(VARIANTS)}
    selected = {}
    for cell, family in CELLS.items():
        candidates = []
        for variant in VARIANTS:
            values = [
                row["mean_crps"]
                for row in rows
                if row["family"] == family
                and row["split"] == "VS"
                and row["variant"] == variant
                and row["seed"] in seeds
            ]
            if len(values) == len(seeds):
                candidates.append((float(np.mean(values)), priority[variant], variant))
        if not candidates:
            raise ValueError("No complete VS variant for %s" % family)
        best_score = min(score for score, _, _ in candidates)
        numerically_tied = [
            item for item in candidates if item[0] <= best_score + 1e-12
        ]
        selected[cell] = min(numerically_tied, key=lambda item: item[1])[2]
    return selected


def selected_row(rows, family, split, variant, seed):
    matches = [
        row for row in rows
        if row["family"] == family
        and row["split"] == split
        and row["variant"] == variant
        and row["seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one row for %s/%s/%s/seed%d; found %d"
            % (family, split, variant, seed, len(matches))
        )
    return matches[0]


def build_cell_blocks(rows, selected, seeds, tag):
    targets = load_targets(tag, "TEST")
    target_size = targets.size
    n_groups = TRACK_GROUPS.get(tag, 1)
    blocks = {}
    for cell, family in CELLS.items():
        seed_blocks = []
        for seed in seeds:
            row = selected_row(rows, family, "TEST", selected[cell], seed)
            scenarios = load_scenarios(Path(row["scenario_file"]), target_size)
            seed_blocks.append(score_blocks_by_date(scenarios, targets, n_groups))
        blocks[cell] = {
            metric: np.mean([item[metric] for item in seed_blocks], axis=0)
            for metric in seed_blocks[0]
        }
    return blocks


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_holm_adjustment(rows):
    """Apply one family-wise Holm correction across all reported contrasts."""
    p_values = np.asarray(
        [float(row["paired_sign_permutation_p"]) for row in rows]
    )
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (p_values.size - np.arange(p_values.size)) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
        row["significant_0.05_after_holm"] = bool(adjusted_p < 0.05)


def analyze(args):
    rows = read_rows(args.metrics)
    seeds = common_seeds(rows)
    selected = select_variants(rows, seeds)
    cell_rows = []
    for cell, family in CELLS.items():
        test_rows = [
            selected_row(rows, family, "TEST", selected[cell], seed)
            for seed in seeds
        ]
        result = {
            "cell": cell,
            "family": family,
            "selected_variant": selected[cell],
            "seeds": " ".join(map(str, seeds)),
            "n_seeds": len(seeds),
        }
        for metric in METRIC_COLUMNS:
            values = np.asarray([row[metric] for row in test_rows])
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        cell_rows.append(result)

    blocks = build_cell_blocks(rows, selected, seeds, args.tag)
    rng = np.random.default_rng(args.seed)
    contrast_rows = []
    for contrast, weights in CONTRASTS.items():
        for metric in ("crps", "energy_score", "variogram_score", "ramp_crps"):
            differences = sum(
                weight * blocks[cell][metric] for cell, weight in weights.items()
            )
            inference = paired_mean_inference(
                differences,
                bootstrap_repetitions=args.bootstrap_repetitions,
                permutation_repetitions=args.permutation_repetitions,
                rng=rng,
            )
            contrast_rows.append({
                "contrast": contrast,
                "metric": metric,
                "mean_difference": inference["mean_difference"],
                "ci_2.5": inference["ci_2.5"],
                "ci_97.5": inference["ci_97.5"],
                "paired_sign_permutation_p": inference["p_value"],
                "n_date_blocks": differences.size,
                "negative_favors_first_named_component": contrast != "anchor_coupling_interaction",
            })
    add_holm_adjustment(contrast_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cell_metrics.csv", cell_rows)
    write_csv(args.output_dir / "factorial_contrasts.csv", contrast_rows)
    audit = {
        "tag": args.tag,
        "common_seeds": seeds,
        "selected_variants_from_mean_vs_crps": selected,
        "test_used_for_selection": False,
        "date_block_inference": True,
        "cells": CELLS,
        "contrasts": CONTRASTS,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("Common completed seeds: %s" % seeds)
    print("Selected on VS: %s" % selected)
    print("Wrote 2x2 ablation results to %s" % args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="wind")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT_DIR / "export" / "unified_postprocessing" / "wind" / "metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "export" / "anchor_coupling_ablation" / "wind",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--permutation-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
