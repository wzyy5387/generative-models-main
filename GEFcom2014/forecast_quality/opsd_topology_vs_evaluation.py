# -*- coding: utf-8 -*-
"""Evaluate the formal OPSD matched-topology runs on VS only."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .aggregate_unified_results import (
    crps_by_date_block,
    ramp_crps_by_date_block,
)
from .compare_scenarios import (
    energy_score_by_day,
    load_scenarios,
    reshape_scenarios_by_day,
    variogram_score_by_day,
)
from .unified_postprocessing import ROOT_DIR, load_targets


SEEDS = (0, 1, 2)
ARMS = {
    "temporal": "matched_temporal",
    "random": "matched_random",
}
METRICS = (
    "crps",
    "energy_score",
    "variogram_score",
    "ramp_crps",
    "picp90",
    "interval_width90",
    "bias",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_daily_metrics(scenarios, targets):
    """Compute the frozen Raw VS metrics for one model/seed."""
    scenarios = np.asarray(scenarios, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 24:
        raise ValueError("OPSD targets must have shape (n_days, 24)")
    if scenarios.shape != (targets.size, 100):
        raise ValueError("Expected (n_days * 24, 100) VS scenarios")
    if not np.isfinite(scenarios).all() or not np.isfinite(targets).all():
        raise ValueError("VS scenarios and targets must be finite")

    samples_by_day = reshape_scenarios_by_day(
        scenarios, targets.shape[0], max_s=100, periods_per_day=24
    )
    lower = np.quantile(samples_by_day, 0.05, axis=2)
    upper = np.quantile(samples_by_day, 0.95, axis=2)
    return {
        "crps": crps_by_date_block(scenarios, targets.reshape(-1), 1),
        "energy_score": energy_score_by_day(scenarios, targets),
        "variogram_score": variogram_score_by_day(scenarios, targets, beta=0.5),
        "ramp_crps": ramp_crps_by_date_block(scenarios, targets, 1),
        "picp90": ((targets >= lower) & (targets <= upper)).mean(axis=1),
        "interval_width90": (upper - lower).mean(axis=1),
        "bias": (samples_by_day.mean(axis=2) - targets).mean(axis=1),
    }


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _block_means(values, block_size=7):
    values = np.asarray(values, dtype=np.float64)
    return np.asarray(
        [values[start:start + block_size].mean()
         for start in range(0, values.size, block_size)],
        dtype=np.float64,
    )


def _paired_block_summary(per_date, repetitions=5000, seed=20261020):
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        block_arrays = []
        observed_values = []
        for seed_id in SEEDS:
            difference = (
                per_date["temporal"][seed_id][metric]
                - per_date["random"][seed_id][metric]
            )
            block_arrays.append(_block_means(difference, 7))
            observed_values.extend(difference.tolist())
        bootstrap = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            bootstrap[index] = np.mean([
                blocks[rng.integers(0, blocks.size, size=blocks.size)].mean()
                for blocks in block_arrays
            ])
        observed = float(np.mean(observed_values))
        rows.append({
            "contrast": "temporal_minus_random",
            "metric": metric,
            "mean_difference": observed,
            "ci_2.5": float(np.quantile(bootstrap, 0.025)),
            "ci_97.5": float(np.quantile(bootstrap, 0.975)),
            "n_seeds": len(SEEDS),
            "n_date_blocks_per_seed": int(block_arrays[0].size),
            "bootstrap_block_days": 7,
        })
    return rows


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.dataset_bundle)
    targets = load_targets("opsd-wind", "VS", bundle)
    per_date = {arm: {} for arm in ARMS}
    provenance = {
        "artifact_role": "formal_topology_vs_raw_evaluation",
        "eligible_for_paper": True,
        "smoke_only": False,
        "synthetic_data": False,
        "hardware_claim": False,
        "physical_platform_used": False,
        "dataset": str(bundle.resolve()),
        "dataset_sha256": sha256_file(bundle),
        "split": "VS",
        "test_read": False,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "scenario_count": 100,
        "metrics": list(METRICS),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
    }
    date_rows = []
    seed_rows = []
    for arm, label in ARMS.items():
        for seed in SEEDS:
            run_dir = output_dir.parent / "formal" / arm / ("seed_%d" % seed)
            metadata_path = run_dir / (
                "opsd-wind_QBMVAE_2_%s_sa_%d.json" % (label, seed)
            )
            scenario_path = run_dir / (
                "scenarios_opsd-wind_QBMVAE_2_%s_sa_%d_100_VS.pickle" % (label, seed)
            )
            if not metadata_path.is_file() or not scenario_path.is_file():
                raise FileNotFoundError("Missing formal VS artifact for %s seed %d" % (arm, seed))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_split_seeds = {"VS": seed, "TEST": seed + 1000003}
            if metadata.get("training_seed") != seed:
                raise ValueError("training_seed mismatch for %s seed %d" % (arm, seed))
            if metadata.get("generation_sampler_seed") != seed:
                raise ValueError("generation_sampler_seed mismatch for %s seed %d" % (arm, seed))
            if metadata.get("generation_sampler_seeds") != expected_split_seeds:
                raise ValueError("split seed metadata mismatch for %s seed %d" % (arm, seed))
            if metadata.get("negative_phase_backend") != "sa" or metadata.get("sampler") != "sa":
                raise ValueError("SA backend mismatch for %s seed %d" % (arm, seed))
            scenarios = load_scenarios(scenario_path, targets.size)
            daily = compute_daily_metrics(scenarios, targets)
            per_date[arm][seed] = daily
            for day in range(targets.shape[0]):
                date_rows.append({
                    "arm": arm,
                    "seed": seed,
                    "date_index": day,
                    **{metric: float(daily[metric][day]) for metric in METRICS},
                })
            seed_rows.append({
                "arm": arm,
                "seed": seed,
                "best_epoch": metadata.get("best_epoch"),
                "checkpoint_selection_metric": metadata.get("checkpoint_selection_metric"),
                "checkpoint_sha256": sha256_file(metadata_path.with_suffix(".pickle")),
                "metadata_sha256": sha256_file(metadata_path),
                "scenario_sha256": sha256_file(scenario_path),
                **{metric + "_mean": float(np.mean(daily[metric])) for metric in METRICS},
                **{metric + "_std": float(np.std(daily[metric], ddof=1)) for metric in METRICS},
            })

    summary_rows = []
    for arm in ARMS:
        rows = [row for row in seed_rows if row["arm"] == arm]
        summary_rows.append({
            "arm": arm,
            "n_seeds": len(rows),
            **{metric + "_seed_mean": float(np.mean([row[metric + "_mean"] for row in rows])) for metric in METRICS},
            **{metric + "_seed_std": float(np.std([row[metric + "_mean"] for row in rows], ddof=1)) for metric in METRICS},
        })

    _write_csv(output_dir / "per_date_metrics.csv", date_rows)
    _write_csv(output_dir / "per_seed_metrics.csv", seed_rows)
    _write_csv(output_dir / "arm_summary.csv", summary_rows)
    _write_csv(output_dir / "paired_temporal_minus_random_7day.csv", _paired_block_summary(per_date))
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("Wrote VS-only topology evaluation to %s" % output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-bundle",
        type=Path,
        default=ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "export" / "opsd_sa_matched_random_graph_v1" / "formal_vs_evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
