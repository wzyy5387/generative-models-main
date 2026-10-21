# -*- coding: utf-8 -*-
"""Evaluate the formal OPSD matched-topology runs on VS only."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
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
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20261020


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


def _time_blocks(values, block_size=BOOTSTRAP_BLOCK_DAYS):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty one-dimensional array")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [values[start:start + block_size]
            for start in range(0, values.size, block_size)]


def _cross_seed_difference(per_date, metric):
    """Return one paired difference per common forecast date.

    The three training seeds are repeated measurements of the same date. The
    bootstrap unit is therefore the cross-seed daily mean, not a seed-specific
    block or an individual scenario read.
    """
    differences = []
    for seed_id in SEEDS:
        temporal = np.asarray(per_date["temporal"][seed_id][metric], dtype=np.float64)
        random = np.asarray(per_date["random"][seed_id][metric], dtype=np.float64)
        if temporal.ndim != 1 or random.ndim != 1 or temporal.shape != random.shape:
            raise ValueError("temporal/random date arrays must have identical one-dimensional shapes")
        differences.append(temporal - random)
    lengths = {values.size for values in differences}
    if len(lengths) != 1:
        raise ValueError("all seeds must share the same forecast-date set")
    return np.mean(np.stack(differences, axis=0), axis=0)


def _weighted_block_bootstrap(values, repetitions, rng, block_size=BOOTSTRAP_BLOCK_DAYS):
    """Bootstrap non-overlapping time blocks with date-proportional weights.

    The last 6-day block is sampled with probability 6/90, not as one of 13
    equally likely blocks. This keeps the bootstrap target equal to the daily
    mean while retaining contiguous time blocks.
    """
    blocks = _time_blocks(values, block_size)
    lengths = np.asarray([block.size for block in blocks], dtype=np.float64)
    probabilities = lengths / lengths.sum()
    draws = np.empty(int(repetitions), dtype=np.float64)
    for index in range(draws.size):
        selected = rng.choice(len(blocks), size=len(blocks), replace=True, p=probabilities)
        draws[index] = np.mean([blocks[item].mean() for item in selected])
    return draws, lengths


def _paired_block_summary(
    per_date,
    repetitions=BOOTSTRAP_REPETITIONS,
    seed=BOOTSTRAP_SEED,
    block_size=BOOTSTRAP_BLOCK_DAYS,
):
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        daily_difference = _cross_seed_difference(per_date, metric)
        bootstrap, block_lengths = _weighted_block_bootstrap(
            daily_difference, repetitions, rng, block_size=block_size
        )
        observed = float(np.mean(daily_difference))
        rows.append({
            "contrast": "temporal_minus_random",
            "metric": metric,
            "mean_difference": observed,
            "ci_2.5": float(np.quantile(bootstrap, 0.025)),
            "ci_97.5": float(np.quantile(bootstrap, 0.975)),
            "n_seeds": len(SEEDS),
            "n_dates": int(daily_difference.size),
            "n_date_blocks": int(block_lengths.size),
            "date_block_lengths": ";".join(str(int(length)) for length in block_lengths),
            "tail_block_days": int(block_lengths[-1]),
            "bootstrap_block_days": int(block_size),
            "bootstrap_method": "cross_seed_daily_mean_weighted_nonoverlap_blocks",
            "bootstrap_seed": int(seed),
            "bootstrap_repetitions": int(repetitions),
        })
    return rows


def _read_existing_daily_metrics(path):
    """Read and validate an existing VS daily-metric artifact without reloading scenarios."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(arm, seed, day) for arm in ARMS for seed in SEEDS for day in range(90)}
    records = {}
    for row in rows:
        key = (row.get("arm"), int(row.get("seed", -1)), int(row.get("date_index", -1)))
        if key in records or key not in expected:
            raise ValueError("existing daily metrics contain duplicate or unexpected date keys")
        record = {}
        for metric in METRICS:
            if metric not in row:
                raise ValueError("existing daily metrics missing %s" % metric)
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError("existing daily metrics contain non-finite values")
            record[metric] = value
        records[key] = record
    actual = set(records)
    if actual != expected:
        raise ValueError("existing daily metrics do not contain exactly 90 common dates per arm/seed")
    per_date = {arm: {seed: {} for seed in SEEDS} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            for metric in METRICS:
                values = np.asarray(
                    [records[(arm, seed, day)][metric] for day in range(90)],
                    dtype=np.float64,
                )
                if values.shape != (90,) or not np.isfinite(values).all():
                    raise ValueError("existing daily metrics are incomplete or non-finite")
                per_date[arm][seed][metric] = values
    return rows, per_date


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty():
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _posthoc_recompute_summary(args):
    """Recompute only the paired VS interval from an existing daily artifact."""
    source_dir = Path(args.source_evaluation_dir)
    output_dir = Path(args.output_dir)
    source_daily = source_dir / "per_date_metrics.csv"
    source_seed = source_dir / "per_seed_metrics.csv"
    source_summary = source_dir / "arm_summary.csv"
    required = (source_daily, source_seed, source_summary)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("source evaluation directory is incomplete")
    daily_rows, per_date = _read_existing_daily_metrics(source_daily)
    # Atomically reserve a new output directory; never overwrite an existing run.
    output_dir.mkdir(parents=True, exist_ok=False)
    for source_path in required:
        shutil.copyfile(source_path, output_dir / source_path.name)
    _write_csv(
        output_dir / "paired_temporal_minus_random_7day.csv",
        _paired_block_summary(per_date),
    )
    bundle = Path(args.dataset_bundle)
    provenance = {
        "artifact_role": "posthoc_recomputed_vs_summary",
        "eligible_for_paper": True,
        "smoke_only": False,
        "synthetic_data": False,
        "hardware_claim": False,
        "physical_platform_used": False,
        "test_read": False,
        "source_evaluation_dir": str(source_dir.resolve()),
        "source_daily_metrics_sha256": sha256_file(source_daily),
        "source_seed_metrics_sha256": sha256_file(source_seed),
        "source_arm_summary_sha256": sha256_file(source_summary),
        "source_daily_rows": len(daily_rows),
        "recomputed_from_existing_daily_metrics": True,
        "scenarios_reloaded": False,
        "dataset": str(bundle.resolve()),
        "dataset_sha256": sha256_file(bundle),
        "split": "VS",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_target": "cross-seed daily paired mean",
        "bootstrap_date_weighting": "daily mean; block draw probability proportional to block length",
        "source_git_commit": _git_commit(),
        "source_git_dirty": _git_dirty(),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("Wrote post-hoc VS summary to %s" % output_dir)


def evaluate(args):
    if args.source_evaluation_dir is not None:
        return _posthoc_recompute_summary(args)
    output_dir = Path(args.output_dir)
    # Atomically reserve a new output directory; never overwrite an existing run.
    output_dir.mkdir(parents=True, exist_ok=False)
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
        "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_target": "cross-seed daily paired mean",
        "bootstrap_date_weighting": "daily mean; block draw probability proportional to block length",
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "evaluator_sha256": sha256_file(Path(__file__)),
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
        help="New output directory; existing paths are rejected without overwriting results",
    )
    parser.add_argument(
        "--source-evaluation-dir",
        type=Path,
        default=None,
        help="Recompute only the paired VS summary from an existing daily-metric artifact",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
