# -*- coding: utf-8 -*-
"""VS-only audit for the GEFCom Wind negative-phase Gibbs-step pilot."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from GEFcom2014 import wind_data
from GEFcom2014.forecast_quality.compare_scenarios import (
    evaluate_model_daily,
    load_scenarios,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "GEFcom2014" / "data" / "wind_data_all_zone.csv"
N_ZONES = 10
N_SCENARIOS = 100
METRICS = ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "mean_bias")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anchor_state_sha256(path):
    checkpoint = torch.load(path, map_location="cpu")
    digest = hashlib.sha256()
    for key, value in sorted(checkpoint["state_dict"].items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def load_variant(run_dir, expected_steps):
    run_dir = Path(run_dir)
    metadata_files = sorted(run_dir.glob("wind_QBMVAE_2_*_sa_0.json"))
    if len(metadata_files) != 1:
        raise ValueError("Expected one seed-0 QBM metadata file in %s" % run_dir)
    metadata_path = metadata_files[0]
    stem = metadata_path.stem
    checkpoint_path = run_dir / (stem + ".pickle")
    anchor_path = run_dir / (stem + "_anchor.pt")
    scenario_path = run_dir / ("scenarios_" + stem + "_100_VS.pickle")
    for path in (checkpoint_path, anchor_path, scenario_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = {
        "negative_steps": expected_steps,
        "negative_phase_backend": "internal",
        "sampler": "sa",
        "scenario_splits": ["VS"],
        "validation_only": True,
        "n_scenarios": N_SCENARIOS,
    }
    for key, expected in checks.items():
        if key == "n_scenarios" and key not in metadata:
            continue
        if metadata.get(key) != expected:
            raise ValueError(
                "%s has %s=%r, expected %r" %
                (metadata_path, key, metadata.get(key), expected)
            )
    return {
        "steps": expected_steps,
        "stem": stem,
        "metadata_path": metadata_path,
        "checkpoint_path": checkpoint_path,
        "anchor_path": anchor_path,
        "scenario_path": scenario_path,
        "metadata": metadata,
    }


def checkpoint_selection_metadata(path):
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    return {
        "best_epoch": getattr(model, "selection_epoch", "NOT_RECORDED"),
        "checkpoint_selection_metric": getattr(
            model, "selection_metric", "VS reconstruction"
        ),
    }


def score_variant(variant, y_vs):
    scenarios = load_scenarios(variant["scenario_path"], y_vs.size)
    if scenarios.shape[1] != N_SCENARIOS:
        raise ValueError("Expected %d scenarios, got %d" % (N_SCENARIOS, scenarios.shape[1]))
    if y_vs.shape[0] % N_ZONES != 0:
        raise ValueError("VS rows are not divisible by ten wind zones")

    days_per_zone = y_vs.shape[0] // N_ZONES
    rows = []
    arrays = {metric: [] for metric in METRICS}
    for zone in range(N_ZONES):
        row_start = zone * days_per_zone
        row_end = (zone + 1) * days_per_zone
        target = y_vs[row_start:row_end]
        period_start = row_start * target.shape[1]
        period_end = row_end * target.shape[1]
        daily = evaluate_model_daily(
            scenarios[period_start:period_end], target, max_s=N_SCENARIOS
        )
        scenario_days = scenarios[period_start:period_end].reshape(
            days_per_zone, target.shape[1], N_SCENARIOS
        )
        mean_bias = (scenario_days.mean(axis=2) - target).mean(axis=1)
        values = {
            "CRPS_raw": daily["CRPS_raw"],
            "Energy": daily["Energy"],
            "Variogram": daily["Variogram"],
            "ramp_CRPS": daily["ramp_CRPS"],
            "PICP90": daily["PICP90"],
            "mean_bias": mean_bias,
        }
        for metric, value in values.items():
            arrays[metric].append(np.asarray(value, dtype=np.float64))
        rows.append({
            "steps": variant["steps"],
            "zone": zone + 1,
            "n_dates": days_per_zone,
            "n_scenarios": N_SCENARIOS,
            **{metric: float(np.mean(values[metric])) for metric in METRICS},
        })

    overall = {
        "steps": variant["steps"],
        "zone": "all",
        "n_dates": y_vs.shape[0],
        "n_scenarios": N_SCENARIOS,
    }
    for metric in METRICS:
        overall[metric] = float(np.concatenate(arrays[metric]).mean())
    rows.append(overall)
    return rows, scenarios


def training_time(log_path):
    if not log_path or not Path(log_path).is_file():
        return None
    raw = Path(log_path).read_bytes()
    encoding = "utf-16" if b"\x00" in raw[:128] else "utf-8"
    text = raw.decode(encoding, errors="replace")
    matches = re.findall(r"Training time ([0-9.]+) s", text)
    return float(matches[-1]) if matches else None


def write_csv(path, rows):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_text(output_dir, steps, anchor_path=None):
    executable = r"D:\anaconda\envs\wsy\python.exe"
    command = (
        executable + " -u -m GEFcom2014.models.QBM_VAE.qbm_vae "
        r"--config configs\paper\fa_bm_vae_wind.json "
        "--output-dir %s --sampler sa --negative-phase-backend internal "
        "--negative-steps %d --seed 0 --n-scenarios 100 "
        "--validation-only --scenario-splits VS --skip-plots "
        "--run-label gibbs%d"
    ) % (output_dir, steps, steps)
    if anchor_path is None:
        command += " --learned-forecast-anchor"
    else:
        command += " --frozen-anchor-model %s" % anchor_path
    return command


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = wind_data(DATA_PATH, test_size=50, random_state=0)
    y_vs = np.asarray(data[3].values, dtype=np.float64)

    five = load_variant(args.steps5_dir, 5)
    twenty = load_variant(args.steps20_dir, 20)
    shared_anchor = Path(args.shared_anchor) if args.shared_anchor else five["anchor_path"]
    five_anchor_state_sha = anchor_state_sha256(five["anchor_path"])
    twenty_anchor_state_sha = anchor_state_sha256(twenty["anchor_path"])
    if five_anchor_state_sha != twenty_anchor_state_sha:
        raise ValueError("The two runs do not share the same frozen anchor")
    if anchor_state_sha256(shared_anchor) != five_anchor_state_sha:
        raise ValueError("The recorded shared anchor does not match the run checkpoints")

    rows = []
    for variant in (five, twenty):
        variant_rows, _ = score_variant(variant, y_vs)
        rows.extend(variant_rows)
    write_csv(
        output_dir / "vs_metrics_by_zone.csv",
        [row for row in rows if row["zone"] != "all"],
    )
    write_csv(
        output_dir / "vs_metrics_summary.csv",
        [row for row in rows if row["zone"] == "all"],
    )

    provenance = {
        "experiment_role": "single_seed_exploratory_pilot",
        "eligible_for_paper": False,
        "formal_five_seed_result": False,
        "test_metrics_computed": False,
        "test_data_used_for_selection": False,
        "dataset": "GEFCom2014 Wind",
        "dataset_file": str(DATA_PATH.resolve()),
        "dataset_sha256": sha256_file(DATA_PATH),
        "split": {"train": "LS", "evaluation": "VS", "test": "not read for metrics"},
        "seed": 0,
        "n_zones": N_ZONES,
        "n_scenarios": N_SCENARIOS,
        "generation_sampler": "sa",
        "negative_phase_backend": "internal persistent Gibbs",
        "variants": [],
        "shared_anchor_source": str(shared_anchor.resolve()),
        "shared_anchor_file_sha256": sha256_file(shared_anchor),
        "shared_anchor_state_sha256": five_anchor_state_sha,
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "torch": package_version("torch"),
        },
        "command": "python -m GEFcom2014.forecast_quality.gefcom_gibbs_steps_pilot",
    }
    for variant, log_path in ((five, args.log5), (twenty, args.log20)):
        metadata = variant["metadata"]
        selection = checkpoint_selection_metadata(variant["checkpoint_path"])
        provenance["variants"].append({
            "negative_steps": variant["steps"],
            "metadata_file": str(variant["metadata_path"].resolve()),
            "metadata_sha256": sha256_file(variant["metadata_path"]),
            "checkpoint_file": str(variant["checkpoint_path"].resolve()),
            "checkpoint_sha256": sha256_file(variant["checkpoint_path"]),
            "anchor_file": str(variant["anchor_path"].resolve()),
            "anchor_file_sha256": sha256_file(variant["anchor_path"]),
            "anchor_state_sha256": anchor_state_sha256(variant["anchor_path"]),
            "scenario_file": str(variant["scenario_path"].resolve()),
            "scenario_sha256": sha256_file(variant["scenario_path"]),
            "training_time_s": training_time(log_path),
            "best_epoch": metadata.get("best_epoch", selection["best_epoch"]),
            "checkpoint_selection_metric": metadata.get(
                "checkpoint_selection_metric", selection["checkpoint_selection_metric"]
            ),
            "configuration": metadata,
        })
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    commands = "# Exact training commands\n"
    commands += command_text(args.steps5_dir, 5, shared_anchor) + "\n"
    commands += command_text(args.steps20_dir, 20, shared_anchor) + "\n"
    (output_dir / "commands.txt").write_text(commands, encoding="utf-8")

    summary_by_steps = {row["steps"]: row for row in rows if row["zone"] == "all"}
    report_lines = [
        "# GEFCom Wind Gibbs-step pilot",
        "",
        "This is a single-seed exploratory experiment and is not a formal five-seed result.",
        "Only LS/VS processing and VS metrics were used; TEST metrics were not computed.",
        "",
        "| negative_steps | CRPS_raw | Energy | Variogram | ramp_CRPS | PICP90 | mean_bias |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for steps in (5, 20):
        row = summary_by_steps[steps]
        report_lines.append(
            "| %d | %.8f | %.8f | %.8f | %.8f | %.8f | %.8f |" %
            (steps, *(row[metric] for metric in METRICS))
        )
    report_lines.extend([
        "",
        "The two runs share one frozen MLP anchor state; normalized state SHA-256: `%s`." % five_anchor_state_sha,
        "Best checkpoint epochs: Gibbs-5 `%s`, Gibbs-20 `%s`; selection metric: VS reconstruction." % (
            provenance["variants"][0]["best_epoch"],
            provenance["variants"][1]["best_epoch"],
        ),
        "No statistical significance claim is made from this single seed.",
        "",
        "See `vs_metrics_by_zone.csv`, `vs_metrics_summary.csv`, `provenance.json`, and `commands.txt`.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(output_dir.resolve())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps5-dir", type=Path, required=True)
    parser.add_argument("--steps20-dir", type=Path, required=True)
    parser.add_argument("--log5", type=Path, required=True)
    parser.add_argument("--log20", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-anchor", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
