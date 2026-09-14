# -*- coding: utf-8 -*-

"""Cross-dataset Raw-primary inference for BM graph structure ablations."""

import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np

from .aggregate_unified_results import (
    METRIC_COLUMNS,
    paired_mean_inference,
    score_blocks_by_date,
)
from .compare_scenarios import load_scenarios
from .unified_postprocessing import ROOT_DIR, TRACK_GROUPS, load_targets


GRAPH_FAMILIES = {
    "wind": {
        "j0": "Forecast-Anchored Independent-prior VAE ablation (J=0)",
        "random": "Forecast-Anchored Same-edge Random-graph BM-VAE ablation",
        "temporal_mi": "QBM-VAE + Learned Forecast Anchor",
    },
    "opsd-wind": {
        "j0": "Forecast-Anchored Independent-prior VAE ablation (J=0)",
        "random": "Forecast-Anchored Same-edge Random-graph BM-VAE ablation",
        "temporal_mi": "QBM-VAE + Forecast Anchor",
    },
}

MODEL_LABELS = {
    "wind": {"j0": "abl_lanchor_ind", "random": "abl_lanchor_randw", "temporal_mi": "lanchor"},
    "opsd-wind": {"j0": "abl_anchor_ind", "random": "abl_anchor_randw", "temporal_mi": "anchor"},
}

CONTRASTS = {
    "temporal_mi_minus_j0": {"temporal_mi": 1.0, "j0": -1.0},
    "temporal_mi_minus_random": {"temporal_mi": 1.0, "random": -1.0},
}

SCORE_METRICS = ("crps", "energy_score", "variogram_score", "ramp_crps")
DEPENDENCE_METRICS = {"variogram_score", "ramp_crps"}
VARIANTS = ("Raw", "Cal")


def model_family(model_name):
    return re.sub(r"\s*\(seed\s+\d+\)$", "", model_name)


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


def common_seeds(rows, families):
    seed_sets = [
        {
            row["seed"]
            for row in rows
            if row["family"] == family and row["seed"] is not None
        }
        for family in families.values()
    ]
    common = set.intersection(*seed_sets)
    if not common:
        raise ValueError("Graph cells have no common completed seed")
    return sorted(common)


def selected_row(rows, family, split, variant, seed):
    matches = [
        row
        for row in rows
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


def multiplicity_family(variant, metric):
    scope = "dependence" if metric in DEPENDENCE_METRICS else "marginal"
    prefix = "primary" if variant == "Raw" else "sensitivity"
    return "%s_%s_%s" % (prefix, variant.lower(), scope)


def add_grouped_holm_adjustment(rows):
    groups = sorted({row["multiplicity_family"] for row in rows})
    for group in groups:
        indices = [
            index for index, row in enumerate(rows)
            if row["multiplicity_family"] == group
        ]
        p_values = np.asarray(
            [float(rows[index]["paired_sign_permutation_p"]) for index in indices]
        )
        order = np.argsort(p_values)
        adjusted_sorted = np.maximum.accumulate(
            (p_values.size - np.arange(p_values.size)) * p_values[order]
        )
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)
        for index, adjusted_p in zip(indices, adjusted):
            rows[index]["holm_adjusted_p"] = float(adjusted_p)
            rows[index]["significant_0.05_after_holm"] = bool(adjusted_p < 0.05)


def build_seed_blocks(rows, families, seeds, variant, tag, dataset_bundle):
    targets = load_targets(tag, "TEST", dataset_bundle)
    target_size = targets.size
    blocks = {}
    for graph, family in families.items():
        blocks[graph] = {}
        for seed in seeds:
            row = selected_row(rows, family, "TEST", variant, seed)
            scenarios = load_scenarios(Path(row["scenario_file"]), target_size)
            blocks[graph][seed] = score_blocks_by_date(
                scenarios, targets, TRACK_GROUPS[tag]
            )
    return blocks


def average_seed_blocks(seed_blocks, seeds):
    blocks = {}
    for graph, graph_blocks in seed_blocks.items():
        blocks[graph] = {
            metric: np.mean([graph_blocks[seed][metric] for seed in seeds], axis=0)
            for metric in SCORE_METRICS
        }
    return blocks


def seed_contrast_differences(seed_blocks, seeds, weights, metric):
    return {
        seed: sum(
            weight * seed_blocks[graph][seed][metric]
            for graph, weight in weights.items()
        )
        for seed in seeds
    }


def graph_model_path(tag, graph, seed):
    label = MODEL_LABELS[tag][graph]
    return (
        ROOT_DIR
        / "export"
        / ("qbm_vae_%s" % tag)
        / ("%s_QBMVAE_2_%s_sa_%d.pickle" % (tag, label, seed))
    )


def graph_mask_metadata(path):
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    mask = model.graph_mask
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float64)
    weights = mask[np.triu_indices(mask.shape[0], k=1)]
    weights = np.sort(weights[weights != 0])
    return {
        "n_edges": int(weights.size),
        "weight_sum": float(weights.sum()),
        "weights": weights,
    }


def graph_edge_audit(tag, seeds):
    rows = []
    for seed in seeds:
        configs = {}
        for graph in ("j0", "random", "temporal_mi"):
            configs[graph] = graph_mask_metadata(graph_model_path(tag, graph, seed))
        same_edge_count = (
            configs["random"]["n_edges"] == configs["temporal_mi"]["n_edges"]
        )
        same_weight_multiset = same_edge_count and np.allclose(
            configs["random"]["weights"],
            configs["temporal_mi"]["weights"],
            rtol=1e-6,
            atol=1e-7,
        )
        rows.append({
            "seed": seed,
            "j0_edges": configs["j0"]["n_edges"],
            "random_edges": configs["random"]["n_edges"],
            "temporal_mi_edges": configs["temporal_mi"]["n_edges"],
            "random_weight_sum": configs["random"]["weight_sum"],
            "temporal_mi_weight_sum": configs["temporal_mi"]["weight_sum"],
            "same_edge_count": bool(same_edge_count),
            "same_weight_multiset": bool(same_weight_multiset),
        })
    return rows


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
    rng = np.random.default_rng(args.seed)
    cell_rows = []
    contrast_rows = []
    seed_contrast_rows = []
    audit = {
        "primary_variant": "Raw",
        "primary_metrics": sorted(DEPENDENCE_METRICS),
        "ecc_excluded": True,
        "calibration_sensitivity_variant": "Cal",
        "tracks": {},
    }

    bundle_by_tag = {
        "wind": None,
        "opsd-wind": Path(args.opsd_bundle),
    }
    for tag in args.tags:
        metrics_path = ROOT_DIR / "export" / "unified_postprocessing" / tag / "metrics.csv"
        rows = read_rows(metrics_path)
        families = GRAPH_FAMILIES[tag]
        seeds = common_seeds(rows, families)
        edge_rows = graph_edge_audit(tag, seeds)
        if not all(
            row["same_edge_count"] and row["same_weight_multiset"]
            for row in edge_rows
        ):
            raise ValueError(
                "Random and Temporal-MI graph masks are not support/weight matched on %s"
                % tag
            )
        audit["tracks"][tag] = {"common_seeds": seeds, "edge_audit": edge_rows}

        for variant in VARIANTS:
            for graph, family in families.items():
                test_rows = [
                    selected_row(rows, family, "TEST", variant, seed)
                    for seed in seeds
                ]
                result = {
                    "track": tag,
                    "variant": variant,
                    "graph": graph,
                    "family": family,
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

            seed_blocks = build_seed_blocks(
                rows, families, seeds, variant, tag, bundle_by_tag[tag]
            )
            blocks = average_seed_blocks(seed_blocks, seeds)
            for contrast, weights in CONTRASTS.items():
                for metric in SCORE_METRICS:
                    per_seed_differences = seed_contrast_differences(
                        seed_blocks, seeds, weights, metric
                    )
                    per_seed_means = {
                        seed: float(differences.mean())
                        for seed, differences in per_seed_differences.items()
                    }
                    for seed in seeds:
                        seed_contrast_rows.append({
                            "track": tag,
                            "variant": variant,
                            "contrast": contrast,
                            "metric": metric,
                            "seed": seed,
                            "mean_difference": per_seed_means[seed],
                            "n_date_blocks": per_seed_differences[seed].size,
                            "negative_favors_temporal_mi": True,
                        })
                    differences = sum(
                        weight * blocks[graph][metric]
                        for graph, weight in weights.items()
                    )
                    inference = paired_mean_inference(
                        differences,
                        bootstrap_repetitions=args.bootstrap_repetitions,
                        permutation_repetitions=args.permutation_repetitions,
                        rng=rng,
                    )
                    contrast_rows.append({
                        "track": tag,
                        "variant": variant,
                        "contrast": contrast,
                        "metric": metric,
                        "mean_difference": inference["mean_difference"],
                        "ci_2.5": inference["ci_2.5"],
                        "ci_97.5": inference["ci_97.5"],
                        "paired_sign_permutation_p": inference["p_value"],
                        "n_date_blocks": differences.size,
                        "n_seeds_favor_temporal_mi": sum(
                            value < 0 for value in per_seed_means.values()
                        ),
                        "all_seeds_favor_temporal_mi": all(
                            value < 0 for value in per_seed_means.values()
                        ),
                        "multiplicity_family": multiplicity_family(variant, metric),
                        "negative_favors_temporal_mi": True,
                    })

    add_grouped_holm_adjustment(contrast_rows)
    write_csv(output_dir / "cell_metrics.csv", cell_rows)
    write_csv(output_dir / "seed_contrasts.csv", seed_contrast_rows)
    write_csv(output_dir / "graph_contrasts.csv", contrast_rows)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print("Wrote graph-structure ablation to %s" % output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", choices=sorted(GRAPH_FAMILIES), default=["wind", "opsd-wind"])
    parser.add_argument(
        "--opsd-bundle",
        type=Path,
        default=ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "export" / "graph_structure_ablation",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--permutation-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
