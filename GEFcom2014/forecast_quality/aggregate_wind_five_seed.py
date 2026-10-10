# -*- coding: utf-8 -*-

"""Strict five-seed Raw comparison for the GEFCom Wind experiment.

This entry point deliberately does not discover arbitrary files.  It reads
only the six registered model families and seeds 0--4, so historical QBM,
ablation, smoke, and offline-hardware artifacts cannot enter the main table.
"""

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.aggregate_unified_results import score_blocks_by_date
from GEFcom2014.forecast_quality.temporal_rank_coupling import load_raw_track
from GEFcom2014.forecast_quality.unified_postprocessing import (
    ROOT_DIR,
    ScenarioPair,
    _write_metrics,
    run_pair,
    load_targets,
)
from GEFcom2014.forecast_quality.compare_scenarios import load_scenarios


SEEDS = tuple(range(5))
N_SCENARIOS = 100
TRACK_GROUPS = 10
MAIN_MODELS = {
    "FA-BM-VAE": {
        "directory": "qbm_vae_wind",
        "scenario": "scenarios_wind_QBMVAE_2_lanchor_sa_{seed}_100_{split}.pickle",
        "metadata": "wind_QBMVAE_2_lanchor_sa_{seed}.json",
        "checkpoint": "wind_QBMVAE_2_lanchor_sa_{seed}.pickle",
        "anchor": "wind_QBMVAE_2_lanchor_sa_{seed}_anchor.pt",
    },
    "Forecast-anchored Gaussian": {
        "directory": "anchor_gaussian_wind",
        "scenario": "scenarios_wind_AnchorGaussian_{seed}_100_{split}.pickle",
        "metadata": "wind_AnchorGaussian_{seed}.json",
        "checkpoint": "wind_AnchorGaussian_{seed}.pt",
    },
    "GMM-4": {
        "directory": "anchor_gmm4_wind",
        "scenario": "scenarios_wind_AnchorGMM4_{seed}_100_{split}.pickle",
        "metadata": "wind_AnchorGMM4_{seed}.json",
        "checkpoint": "wind_AnchorGMM4_{seed}.pt",
    },
    "Score-SDE": {
        "directory": "anchor_score_sde_wind",
        "scenario": "scenarios_wind_AnchorScoreSDE_{seed}_100_{split}.pickle",
        "metadata": "wind_AnchorScoreSDE_{seed}.json",
        "checkpoint": "wind_AnchorScoreSDE_{seed}.pt",
    },
    "D3U": {
        "directory": "d3u_wind",
        "scenario": "scenarios_wind_D3UAdapted_{seed}_100_{split}.pickle",
        "metadata": "wind_D3UAdapted_{seed}.json",
        "checkpoint": "wind_D3UAdapted_{seed}.pt",
    },
    "Treeffuser": {
        "directory": "treeffuser_wind",
        "scenario": "scenarios_wind_Treeffuser_{seed}_100_{split}.pickle",
        "metadata": "wind_Treeffuser_{seed}.json",
        "checkpoint": "wind_Treeffuser_{seed}.pickle",
    },
}

METRICS = ("crps", "energy_score", "variogram_score", "ramp_crps")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_model_specs():
    """Return the exact main-comparison registry, excluding historical QBM files."""
    return tuple(MAIN_MODELS)


def _artifact_paths(root, family, seed):
    spec = MAIN_MODELS[family]
    directory = root / "export" / spec["directory"]
    metadata = directory / spec["metadata"].format(seed=seed)
    checkpoint = directory / spec["checkpoint"].format(seed=seed)
    scenarios = {
        split: directory / spec["scenario"].format(seed=seed, split=split)
        for split in ("VS", "TEST")
    }
    anchor = directory / spec["anchor"].format(seed=seed) if "anchor" in spec else None
    return metadata, checkpoint, anchor, scenarios


def audit_artifacts(root=ROOT_DIR, seeds=SEEDS):
    targets = {split: load_targets("wind", split) for split in ("VS", "TEST")}
    rows = []
    for family in registered_model_specs():
        for seed in seeds:
            metadata_path, checkpoint_path, anchor_path, scenario_paths = _artifact_paths(
                root, family, seed
            )
            for path in (metadata_path, checkpoint_path):
                if not path.is_file():
                    raise FileNotFoundError("Missing %s artifact: %s" % (family, path))
            if anchor_path is not None and not anchor_path.is_file():
                raise FileNotFoundError("Missing learned-anchor checkpoint: %s" % anchor_path)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded_seed = metadata.get("seed")
            if recorded_seed is not None and int(recorded_seed) != seed:
                raise ValueError("Seed mismatch in %s" % metadata_path)
            selection_flag = metadata.get("test_used_for_selection", "NOT_RECORDED")
            if selection_flag != "NOT_RECORDED" and selection_flag is not False:
                raise ValueError("TEST selection flag is not false in %s" % metadata_path)

            row = {
                "model": family,
                "seed": seed,
                "metadata_file": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
                "checkpoint_file": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "anchor_checkpoint_file": str(anchor_path.resolve()) if anchor_path else "",
                "anchor_checkpoint_sha256": sha256_file(anchor_path) if anchor_path else "",
                "test_used_for_selection": selection_flag,
                "artifact_status": "VALID_SCENARIO_PAIR",
            }
            for split, target in targets.items():
                path = scenario_paths[split]
                if not path.is_file():
                    raise FileNotFoundError("Missing %s %s scenarios: %s" % (family, split, path))
                scenarios = load_scenarios(path, target.size)
                if scenarios.shape[1] != N_SCENARIOS:
                    raise ValueError("%s has %d scenarios, expected %d" % (
                        path, scenarios.shape[1], N_SCENARIOS
                    ))
                if not np.isfinite(scenarios).all():
                    raise ValueError("Non-finite values in %s" % path)
                row[split.lower() + "_scenario_file"] = str(path.resolve())
                row[split.lower() + "_scenario_sha256"] = sha256_file(path)
                row[split.lower() + "_shape"] = "%d x %d" % scenarios.shape
            rows.append(row)
    return rows


def _date_order(split):
    data, _ = load_raw_track("wind")
    frame = data[{"VS": 3, "TEST": 5}[split]]
    group_dates = list(frame.iloc[: frame.shape[0] // TRACK_GROUPS].index)
    if any(
        list(frame.iloc[group * len(group_dates):(group + 1) * len(group_dates)].index)
        != group_dates
        for group in range(1, TRACK_GROUPS)
    ):
        raise ValueError("Wind-zone date ordering is not shared across groups")
    order = sorted(range(len(group_dates)), key=lambda index: group_dates[index])
    labels = [group_dates[index].isoformat() for index in order]
    return np.asarray(order, dtype=int), labels


def _score_artifacts(root, audit_rows):
    targets = {split: load_targets("wind", split) for split in ("VS", "TEST")}
    date_orders = {split: _date_order(split) for split in targets}
    output = []
    daily = {}
    for audit in audit_rows:
        family = audit["model"]
        seed = int(audit["seed"])
        daily.setdefault((family, seed), {})
        for split, target in targets.items():
            scenarios = load_scenarios(
                Path(audit[split.lower() + "_scenario_file"]), target.size
            )
            blocks = score_blocks_by_date(scenarios, target, TRACK_GROUPS)
            order, labels = date_orders[split]
            values = {metric: np.asarray(blocks[metric])[order] for metric in METRICS}
            daily[(family, seed)][split] = values
            row = {
                "model": family,
                "seed": seed,
                "split": split,
                "statistical_unit": "forecast_date (10-zone daily block), not scenario read",
                "n_date_blocks": len(labels),
                "date_start": labels[0],
                "date_end": labels[-1],
                "n_scenarios": N_SCENARIOS,
            }
            row.update({metric: float(values[metric].mean()) for metric in METRICS})
            output.append(row)
    return output, daily, date_orders


def _contiguous_blocks(n, length):
    return [np.arange(start, min(start + length, n)) for start in range(0, n, length)]


def block_paired_inference(differences, block_length=7, repetitions=10000, seed=0):
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("differences must be a non-empty vector")
    blocks = _contiguous_blocks(differences.size, block_length)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[item] for item in chosen])[: differences.size]
        bootstrap[index] = differences[indices].mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, len(blocks)))
    signed = np.empty_like(signs)
    for index, block in enumerate(blocks):
        signed[:, index] = differences[block].sum()
    null = (signs * signed).sum(axis=1)
    null /= sum(block.size for block in blocks)
    observed = float(differences.mean())
    return {
        "mean_difference": observed,
        "ci_2.5": float(np.quantile(bootstrap, 0.025)),
        "ci_97.5": float(np.quantile(bootstrap, 0.975)),
        "block_sign_flip_p": float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (repetitions + 1)),
        "effect_size": float(observed / differences.std(ddof=1)) if differences.std(ddof=1) > 0 else 0.0,
        "n_date_blocks": len(blocks),
        "block_length_days": block_length,
    }


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (values.size - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _paired_rows(daily, seed, repetitions):
    output = []
    anchor = "FA-BM-VAE"
    for family in registered_model_specs():
        if family == anchor:
            continue
        for metric in METRICS:
            anchor_values = np.mean(
                [daily[(anchor, item)]["TEST"][metric] for item in SEEDS], axis=0
            )
            model_values = np.mean(
                [daily[(family, item)]["TEST"][metric] for item in SEEDS], axis=0
            )
            result = block_paired_inference(
                model_values - anchor_values,
                block_length=7,
                repetitions=repetitions,
                seed=seed + len(output),
            )
            output.append({
                "model_a": family,
                "model_b": anchor,
                "metric": metric,
                "comparison": "model_a_minus_FA-BM-VAE",
                "seed_aggregation": "mean of five seed daily scores",
                "statistical_unit": "forecast date block, not scenario read",
                **result,
            })
    for metric in METRICS:
        indices = [index for index, row in enumerate(output) if row["metric"] == metric]
        adjusted = holm_adjust([output[index]["block_sign_flip_p"] for index in indices])
        for index, value in zip(indices, adjusted):
            output[index]["holm_p"] = value
    return output


def _write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_state(root):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _run_sensitivity(root, output_root, audit_rows):
    sensitivity_root = output_root / "cal_ecc_sensitivity"
    track_dir = sensitivity_root / "wind"
    rows = []
    audits = []
    for family in registered_model_specs():
        for seed in SEEDS:
            _, _, _, paths = _artifact_paths(root, family, seed)
            pair = ScenarioPair(
                "%s (seed %d)" % (family, seed),
                paths["VS"],
                paths["TEST"],
            )
            model_rows, audit = run_pair(
                tag="wind",
                pair=pair,
                output_root=sensitivity_root,
                spread_grid=np.linspace(0.5, 2.0, 31).tolist(),
                harmonics=3,
                ridge=1.0,
                trc_seed=seed,
            )
            rows.extend(model_rows)
            audits.append(audit)
    track_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(track_dir / "metrics.csv", rows)
    summaries = []
    for model, variant in itertools.product(registered_model_specs(), ("Raw", "Cal", "TRC", "Cal+TRC")):
        values = [
            float(row["mean_crps"])
            for row in rows
            if row["model"].startswith(model + " (seed ")
            and row["variant"] == variant and row["split"] == "TEST"
        ]
        if values:
            summaries.append({
                "model": model,
                "variant": variant,
                "n_seeds": len(values),
                "mean_crps_mean": float(np.mean(values)),
                "mean_crps_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            })
    _write_rows(track_dir / "crps_sensitivity_summary.csv", summaries)
    (track_dir / "manifest.json").write_text(json.dumps({
        "artifact_role": "cal_ecc_sensitivity",
        "eligible_for_paper": False,
        "raw_is_primary": True,
        "variants": ["Cal", "TRC", "Cal+TRC"],
        "trc_template_split": "LS",
        "calibration_fit_split": "VS",
        "test_used_for_selection": False,
        "audits": audits,
    }, indent=2), encoding="utf-8")


def run(root=ROOT_DIR, output_dir=None, repetitions=10000, seed=2026, include_cal_ecc=False):
    root = Path(root)
    output_dir = Path(output_dir or root / "export" / "wind_five_seed_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = audit_artifacts(root)
    per_seed, daily, date_orders = _score_artifacts(root, audit_rows)
    summary = []
    for family in registered_model_specs():
        family_rows = [row for row in per_seed if row["model"] == family and row["split"] == "TEST"]
        for metric in METRICS:
            values = [row[metric] for row in family_rows]
            summary.append({
                "model": family,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(np.mean(values)),
                "seed_std": float(np.std(values, ddof=1)),
                "statistical_unit": "seed means over forecast dates; paired tests use dates",
            })
    paired = _paired_rows(daily, seed=seed, repetitions=repetitions)
    _write_rows(output_dir / "raw_per_seed_metrics.csv", per_seed)
    _write_rows(output_dir / "raw_five_seed_summary.csv", summary)
    _write_rows(output_dir / "raw_paired_date_tests.csv", paired)
    (output_dir / "artifact_audit.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    manifest = {
        "artifact_role": "formal_five_seed_raw_comparison",
        "eligible_for_paper": True,
        "synthetic_data": False,
        "hardware_claim": False,
        "physical_platform_used": False,
        "dataset": "GEFCom2014 Wind",
        "models": list(registered_model_specs()),
        "seeds": list(SEEDS),
        "n_scenarios": N_SCENARIOS,
        "raw_is_primary": True,
        "cal_ecc_is_sensitivity_only": True,
        "statistical_unit": "forecast date block, not scenario read",
        "bootstrap": {"block_length_days": 7, "repetitions": repetitions},
        "date_labels": {split: labels for split, (_, labels) in date_orders.items()},
        "git": _git_state(root),
        "python": sys.version,
        "platform": platform.platform(),
        "command": "python -m GEFcom2014.forecast_quality.aggregate_wind_five_seed",
        "outputs": {
            "artifact_audit": "artifact_audit.json",
            "per_seed_raw": "raw_per_seed_metrics.csv",
            "summary": "raw_five_seed_summary.csv",
            "paired_tests": "raw_paired_date_tests.csv",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if include_cal_ecc:
        _run_sensitivity(root, output_dir, audit_rows)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Strict GEFCom Wind five-seed Raw comparison.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-cal-ecc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(
        output_dir=args.output_dir,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        include_cal_ecc=args.include_cal_ecc,
    ))
