# -*- coding: utf-8 -*-
"""Protocol runner and inference utilities for the fixed OPSD origins.

The default command writes a manifest and a synthetic smoke result only.  It
does not train a model.  Formal execution requires an explicit confirmation
flag and a real OPSD bundle; TEST is never used for selection in this module.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from GEFcom2014.external_datasets import OPSD_ACTUAL, OPSD_FORECAST, OPSD_SHA256, sha256_file
from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily, load_scenarios
from GEFcom2014.forecast_quality.paper_artifact_guard import parse_bool


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "paper" / "opsd_rolling_origins.json"
DEFAULT_OUTPUT = ROOT_DIR / "export" / "opsd_rolling_origins"
METRICS = ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90",
           "interval_width90", "MAQCE")
REGISTERED_ORIGINS = ("origin_1", "origin_2", "origin_3")
REGISTERED_MODELS = ("FA-BM-VAE", "Forecast-anchored Gaussian", "GMM-4", "Score-SDE")
REGISTERED_SEEDS = (0, 1, 2, 3, 4)
MODEL_ENTRYPOINTS = {
    "FA-BM-VAE": ROOT_DIR / "GEFcom2014" / "models" / "QBM_VAE" / "qbm_vae.py",
    "Forecast-anchored Gaussian": ROOT_DIR / "GEFcom2014" / "models" / "QBM_VAE" / "anchor_gaussian_baseline.py",
    "GMM-4": ROOT_DIR / "GEFcom2014" / "models" / "QBM_VAE" / "anchor_gaussian_baseline.py",
    "Score-SDE": ROOT_DIR / "GEFcom2014" / "models" / "QBM_VAE" / "anchor_score_sde.py",
}


def _parse_day(value):
    return date.fromisoformat(str(value))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(path):
    return file_sha256(path)


def atomic_write_json(path, payload):
    """Write status/manifest JSON atomically so interruption cannot leave half JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_dirty():
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT_DIR), text=True)
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "torch", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        import torch
        versions["torch_cuda_build"] = torch.version.cuda or "none"
        versions["cuda"] = versions["torch_cuda_build"]
        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        versions["torch_cuda_build"] = "unavailable"
        versions["cuda"] = "unavailable"
        versions["cuda_available"] = False
        versions["cuda_device"] = "unavailable"
    return versions


def source_hashes():
    return {"rolling_runner": file_sha256(Path(__file__)),
            "model_entrypoints": {model: file_sha256(path) for model, path in MODEL_ENTRYPOINTS.items()}}


def build_provenance(config_path, source_csv, command=None):
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "source_hashes": source_hashes(),
        "protocol_sha256": file_sha256(config_path),
        "dataset_sha256": file_sha256(source_csv),
        "environment": environment_versions(),
        "command": command or [],
    }


def load_protocol(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    origins = protocol.get("origins", [])
    if len(origins) != 3:
        raise ValueError("The OPSD protocol must contain exactly three origins")
    previous_test_end = None
    for expected_index, origin in enumerate(origins, 1):
        if origin.get("origin_id") != "origin_%d" % expected_index:
            raise ValueError("Origin IDs must be origin_1, origin_2 and origin_3")
        vs_start, vs_end = _parse_day(origin["vs_start"]), _parse_day(origin["vs_end"])
        test_start, test_end = _parse_day(origin["test_start"]), _parse_day(origin["test_end"])
        if not (vs_start <= vs_end < test_start <= test_end):
            raise ValueError("VS/TEST windows must be ordered and non-overlapping")
        if previous_test_end is not None and vs_start <= previous_test_end:
            raise ValueError("Origins must be chronological and non-overlapping")
        previous_test_end = test_end
        ls_start = _parse_day(origin.get("ls_start", "2016-01-01"))
        if _parse_day(origin["ls_end"]) != vs_start - timedelta(days=1):
            raise ValueError("LS and VS boundary has a gap or overlap")
        if ls_start.isoformat() != "2016-01-01":
            raise ValueError("OPSD LS must start at 2016-01-01")
        expected_ls = (ls_end := vs_start - timedelta(days=1)) - ls_start
        if expected_ls.days + 1 != int(origin["ls_days"]):
            raise ValueError("OPSD LS length does not match registered dates")
    if protocol.get("models") != [
        "FA-BM-VAE", "Forecast-anchored Gaussian", "GMM-4", "Score-SDE"
    ]:
        raise ValueError("The four pre-registered OPSD models must not be changed")
    if protocol.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("The OPSD protocol requires seeds 0-4")
    if protocol.get("test_peeking") is not False:
        raise ValueError("TEST peeking must be disabled")
    if "CRPS_raw" not in protocol.get("metrics", []):
        raise ValueError("The registered metric name must be CRPS_raw")
    formal = protocol.get("training", {}).get("formal", {})
    if formal.get("negative_phase_backend") != "internal" or formal.get("sampler") != "sa":
        raise ValueError("Formal FA-BM-VAE must use internal negative phase and SA inference")
    if formal.get("epochs", {}).get("FA-BM-VAE") != 80 or formal.get("epochs", {}).get("Score-SDE") != 200:
        raise ValueError("Formal epoch configuration is not frozen")


def origin_windows(protocol):
    """Return exact date windows, including the expanding LS end date."""
    validate_protocol(protocol)
    origins = protocol["origins"]
    windows = []
    for index, origin in enumerate(origins):
        vs_start = _parse_day(origin["vs_start"])
        ls_end = vs_start - timedelta(days=1)
        ls_start = _parse_day(origin.get("ls_start", "2016-01-01"))
        windows.append({
            "origin_id": origin["origin_id"],
            "ls_start": None if ls_start is None else ls_start.isoformat(),
            "ls_end": ls_end.isoformat(),
            "ls_days": int(origin["ls_days"]),
            "vs_start": origin["vs_start"],
            "vs_end": origin["vs_end"],
            "test_start": origin["test_start"],
            "test_end": origin["test_end"],
        })
    return windows


def build_origin_bundle(protocol, origin_id, source_csv, output_path):
    """Build one origin from the verified raw OPSD CSV, fitting scale on LS only."""
    import pandas as pd
    from GEFcom2014.external_datasets import BUNDLE_ARRAY_KEYS

    source_csv = Path(source_csv)
    if sha256_file(source_csv) != OPSD_SHA256:
        raise ValueError("OPSD raw source SHA-256 mismatch")
    origin = next(item for item in protocol["origins"] if item["origin_id"] == origin_id)
    frame = pd.read_csv(source_csv, usecols=["utc_timestamp", OPSD_ACTUAL, OPSD_FORECAST],
                        parse_dates=["utc_timestamp"]).dropna()
    frame = frame.sort_values("utc_timestamp").drop_duplicates("utc_timestamp")
    frame["date"] = frame["utc_timestamp"].dt.tz_convert(None).dt.floor("D")
    frame["hour"] = frame["utc_timestamp"].dt.hour
    actual_days, forecast_days, dates = [], [], []
    for day, group in frame.groupby("date", sort=True):
        group = group.sort_values("hour")
        if len(group) == 24 and np.array_equal(group["hour"].to_numpy(), np.arange(24)):
            actual_days.append(group[OPSD_ACTUAL].to_numpy(float))
            forecast_days.append(group[OPSD_FORECAST].to_numpy(float))
            dates.append(pd.Timestamp(day))
    dates = pd.DatetimeIndex(dates)
    actual, forecast = np.asarray(actual_days), np.asarray(forecast_days)
    ls_start = pd.Timestamp(origin.get("ls_start", "2016-01-01"))
    ls_end = pd.Timestamp(origin["ls_end"])
    vs_start, vs_end = pd.Timestamp(origin["vs_start"]), pd.Timestamp(origin["vs_end"])
    test_start, test_end = pd.Timestamp(origin["test_start"]), pd.Timestamp(origin["test_end"])
    masks = [(dates >= ls_start) & (dates <= ls_end), (dates >= vs_start) & (dates <= vs_end),
             (dates >= test_start) & (dates <= test_end)]
    if any(int(mask.sum()) != expected for mask, expected in zip(masks, (origin["ls_days"], 90, 90))):
        raise ValueError("Raw OPSD does not contain the complete registered origin windows")
    scale_mw = float(np.max(actual[masks[0]]))
    target = np.clip(actual / scale_mw, 0.0, 1.0)
    context = np.concatenate((np.clip(forecast / scale_mw, 0.0, 1.0),
                              np.column_stack((np.sin(2*np.pi*dates.dayofyear.to_numpy()/365.25),
                                                np.cos(2*np.pi*dates.dayofyear.to_numpy()/365.25),
                                                np.eye(7)[dates.dayofweek.to_numpy()]))), axis=1)
    arrays, date_arrays = {}, {}
    for name, mask in zip(("ls", "vs", "test"), masks):
        arrays["x_" + name] = context[mask].astype(np.float32)
        arrays["y_" + name] = target[mask].astype(np.float32)
        date_arrays["dates_" + name] = np.asarray(dates[mask].strftime("%Y-%m-%d"), dtype="U10")
    metadata = {"dataset_name": "opsd-wind", "source": str(source_csv), "source_sha256": OPSD_SHA256,
                "origin_id": origin_id, "scale_fit_split": "LS", "scale_mw": scale_mw,
                "normalization": "target=clip(actual/scale_mw,0,1); context=clip(forecast/scale_mw,0,1)",
                "date_ranges": {name: [str(dates[mask][0].date()), str(dates[mask][-1].date())]
                                for name, mask in zip(("ls", "vs", "test"), masks)},
                "split_days": {name: int(len(arrays["y_"+name])) for name in ("ls", "vs", "test")},
                "smoke_only": False, "synthetic_data": False, "eligible_for_paper": True,
                "test_used_for_selection": False}
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays, **date_arrays, metadata=json.dumps(metadata))
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_run_manifest(protocol, dataset_bundle=None, artifact_role="formal_task"):
    """Build commands without launching them; each row is independently auditable."""
    bundle = str(dataset_bundle or protocol["dataset_bundle"])
    rows = []
    for window in origin_windows(protocol):
        for model in protocol["models"]:
            for seed in protocol["seeds"]:
                rows.append({
                    "origin_id": window["origin_id"],
                    "model": model,
                    "seed": int(seed),
                    "dataset_bundle": bundle,
                    "ls_end": window["ls_end"],
                    "vs_start": window["vs_start"],
                    "vs_end": window["vs_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "status": "PLANNED",
                    "artifact_role": artifact_role,
                    "eligible_for_paper": False,
                    "formal_matrix_complete": False,
                    "test_used_for_selection": False,
                })
    return rows


def _model_command(model, bundle, output_dir, seed, config, device):
    epochs = int(config["epochs"][model])
    score_steps = int(config.get("score_sampling_steps", 32))
    common = [sys.executable, "-m"]
    device = str(device)
    if model == "FA-BM-VAE":
        command = common + ["GEFcom2014.models.QBM_VAE.qbm_vae", "--tag", "opsd-wind",
            "--dataset-bundle", str(bundle), "--output-dir", str(output_dir),
            "--forecast-anchor", "--sampler", "sa", "--seed", str(seed),
            "--negative-phase-backend", "internal", "--epochs", str(epochs), "--n-scenarios", "100",
            "--scenario-splits", "TEST", "--skip-plots"]
        if device == "cpu": command.append("--cpu")
        return command
    if model == "Forecast-anchored Gaussian":
        command = common + ["GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline", "--tag", "opsd-wind",
            "--dataset-bundle", str(bundle), "--anchor-source", "context", "--seed", str(seed),
            "--epochs", str(epochs), "--n-scenarios", "100", "--output-dir", str(output_dir)]
        if device == "cpu": command.append("--cpu")
        return command
    if model == "GMM-4":
        command = common + ["GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline", "--tag", "opsd-wind",
            "--dataset-bundle", str(bundle), "--anchor-source", "context", "--components", "4",
            "--seed", str(seed), "--epochs", str(epochs), "--n-scenarios", "100",
            "--output-dir", str(output_dir)]
        if device == "cpu": command.append("--cpu")
        return command
    if model == "Score-SDE":
        device_arg = "cpu" if device == "cpu" else "auto" if device == "auto" else device
        return common + ["GEFcom2014.models.QBM_VAE.anchor_score_sde", "--tag", "opsd-wind",
            "--dataset-bundle", str(bundle), "--anchor-source", "context", "--seed", str(seed),
            "--epochs", str(epochs), "--sampling-steps", str(score_steps), "--n-scenarios", "100",
            "--output-dir", str(output_dir), "--device", device_arg]
    raise ValueError("Unknown OPSD pilot model: %s" % model)


def _pilot_command(model, bundle, output_dir, seed, epochs):
    return _model_command(
        model, bundle, output_dir, seed,
        {"epochs": {name: int(epochs) for name in (
            "FA-BM-VAE", "Forecast-anchored Gaussian", "GMM-4", "Score-SDE")},
         "score_sampling_steps": 4}, "cpu"
    )


def run_real_pilot(protocol, source_csv, output_dir, seed=0, epochs=3):
    """Run only origin_1/seed_0 on raw data and write auditable metrics."""
    from GEFcom2014.external_datasets import load_daily_bundle
    from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model, load_scenarios
    pilot_dir = Path(output_dir) / "origin_1" / "seed_0"
    bundle = pilot_dir / "data" / "origin_1.npz"
    metadata = build_origin_bundle(protocol, "origin_1", source_csv, bundle)
    arrays, _, dates = load_daily_bundle(bundle)
    target = arrays["y_test"]
    rows = []
    for model in protocol["models"]:
        model_dir = pilot_dir / model.lower().replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        stdout_path, stderr_path = model_dir / "stdout.log", model_dir / "stderr.log"
        status_path = model_dir / "status.json"
        command = _pilot_command(model, bundle, model_dir, seed, epochs)
        status = {"origin_id": "origin_1", "model": model, "seed": seed,
                  "synthetic_data": False, "smoke_only": False,
                  "eligible_for_paper": False, "pilot_only": True,
                  "pilot_purpose": "code_connectivity_only", "test_used_for_selection": False,
                  "command": command, "status": "RUNNING"}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=str(ROOT_DIR), stdout=stdout, stderr=stderr, text=True)
        if completed.returncode != 0:
            status.update({"status": "FAILED", "returncode": completed.returncode})
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            raise RuntimeError("OPSD pilot failed for %s; see %s" % (model, stderr_path))
        scenario_files = sorted(model_dir.glob("scenarios_*_TEST.pickle"))
        if len(scenario_files) != 1:
            raise RuntimeError("Expected one real TEST scenario file for %s, found %d" % (model, len(scenario_files)))
        scenarios = load_scenarios(scenario_files[0], len(target) * 24)
        daily = evaluate_model_daily(scenarios, target)
        for index, day in enumerate(dates["test"]):
            row = {"origin_id": "origin_1", "date": str(day), "model": model, "seed": seed,
                   "scenario_file": str(scenario_files[0]), "dataset_sha256": OPSD_SHA256,
                   "smoke_only": False, "synthetic_data": False, "eligible_for_paper": False,
                   "pilot_only": True, "pilot_purpose": "code_connectivity_only",
                   "test_used_for_selection": False,
                   **{metric: float(daily[metric][index]) for metric in METRICS},
                   "coverage": float(daily["coverage"][index])}
            rows.append(row)
        status.update({"status": "COMPLETED", "returncode": 0, "scenario_file": str(scenario_files[0]),
                       "metrics": {key: "per_date" for key in METRICS}})
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_csv(Path(output_dir) / "pilot_metrics.csv", rows)
    summary = {"protocol": "opsd-rolling-origin-v1", "origin_id": "origin_1", "seed": seed,
               "models": protocol["models"], "dataset_sha256": OPSD_SHA256,
               "smoke_only": False, "synthetic_data": False, "eligible_for_paper": False,
               "pilot_only": True, "pilot_purpose": "code_connectivity_only",
               "test_used_for_selection": False, "formal_5_seed_run": "NOT RUN",
               "rows": rows, "bundle_metadata": metadata}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _resume_mismatch_reasons(existing, row, provenance, run_dir):
    reasons = []
    expected = {
        "origin_id": row["origin_id"], "model": row["model"], "seed": int(row["seed"]),
        "protocol_sha256": provenance["protocol_sha256"],
        "dataset_sha256": provenance["dataset_sha256"],
        "runner_sha256": provenance["source_hashes"]["rolling_runner"],
        "model_source_sha256": provenance["source_hashes"]["model_entrypoints"][row["model"]],
    }
    for field, value in expected.items():
        if existing.get(field) != value:
            reasons.append("%s mismatch" % field)
    for path_field, hash_field in (("scenario_file", "scenario_sha256"),
                                   ("metrics_file", "metrics_sha256")):
        filename = existing.get(path_field)
        if not filename:
            reasons.append("missing %s" % path_field)
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.is_file():
            reasons.append("missing %s" % path_field)
        elif not existing.get(hash_field) or file_sha256(path) != existing.get(hash_field):
            reasons.append("%s hash mismatch" % path_field)
    return reasons


def _run_formal_task(row, bundle_dir, output_dir, protocol, config, device, retries,
                     config_hash, provenance, timeout_seconds=None, resume=True,
                     artifact_role="formal_task"):
    from datetime import datetime, timezone
    from GEFcom2014.external_datasets import load_daily_bundle
    run_dir = Path(output_dir) / row["origin_id"] / ("seed_%d" % row["seed"]) / row["model"].lower().replace(" ", "_")
    status_path = run_dir / "status.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        existing = {"status": "STALE", "stale_reasons": ["invalid status JSON"]}
    if resume and existing.get("status") == "COMPLETED":
        stale_reasons = _resume_mismatch_reasons(existing, row, provenance, run_dir)
        if not stale_reasons:
            return existing
        existing = {"status": "STALE", "stale_reasons": stale_reasons}
        atomic_write_json(status_path, {**existing, **row, "artifact_role": artifact_role,
                                        "eligible_for_paper": False,
                                        "formal_matrix_complete": False})
    command = _model_command(row["model"], bundle_dir / (row["origin_id"] + ".npz"),
                             run_dir, row["seed"], config, device)
    status = {**row, "status": "RUNNING", "command": command,
              "smoke_only": False, "synthetic_data": False,
              "artifact_role": artifact_role, "eligible_for_paper": False,
              "formal_matrix_complete": False, "test_used_for_selection": False,
              "analysis_primary": "raw", "attempts": 0,
              "config_sha256": config_hash, "training_config": config,
              "protocol_sha256": provenance["protocol_sha256"],
              "dataset_sha256": provenance["dataset_sha256"],
              "runner_sha256": provenance["source_hashes"]["rolling_runner"],
              "model_source_sha256": provenance["source_hashes"]["model_entrypoints"][row["model"]],
              "provenance": provenance,
              "resource_monitor": {"peak_memory": "not_instrumented"},
              "timeout_seconds": timeout_seconds or "NOT_SET",
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    if existing.get("stale_reasons"):
        status["stale_reasons"] = existing["stale_reasons"]
    atomic_write_json(status_path, status)
    started = time.perf_counter()
    for attempt in range(int(retries) + 1):
        status["attempts"] = attempt + 1
        try:
            with (run_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("a", encoding="utf-8") as stderr:
                completed = subprocess.run(command, cwd=str(ROOT_DIR), stdout=stdout, stderr=stderr,
                                           text=True, timeout=timeout_seconds or None)
        except subprocess.TimeoutExpired:
            status.update({"status": "FAILED", "timed_out": True,
                           "elapsed_seconds": time.perf_counter() - started,
                           "completed_at_utc": datetime.now(timezone.utc).isoformat()})
            atomic_write_json(status_path, status)
            return status
        if completed.returncode == 0:
            break
    if completed.returncode != 0:
        status.update({"status": "FAILED", "returncode": completed.returncode,
                       "elapsed_seconds": time.perf_counter() - started,
                       "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(status_path, status)
        return status
    scenario_files = sorted(run_dir.glob("scenarios_*_TEST.pickle"))
    if len(scenario_files) != 1:
        status.update({"status": "FAILED", "error": "expected exactly one TEST scenario file",
                       "elapsed_seconds": time.perf_counter() - started,
                       "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(status_path, status)
        return status
    arrays, metadata, dates = load_daily_bundle(bundle_dir / (row["origin_id"] + ".npz"))
    scenarios = load_scenarios(scenario_files[0], len(arrays["y_test"]) * 24)
    daily = evaluate_model_daily(scenarios, arrays["y_test"])
    metric_rows = []
    for index, day in enumerate(dates["test"]):
        metric_rows.append({"origin_id": row["origin_id"], "date": str(day), "model": row["model"],
                            "seed": int(row["seed"]), "scenario_file": str(scenario_files[0]),
                            "scenario_sha256": file_sha256(scenario_files[0]),
                            "dataset_sha256": metadata["source_sha256"], "scale_mw": metadata["scale_mw"],
                            "smoke_only": False, "synthetic_data": False,
                            "artifact_role": artifact_role, "eligible_for_paper": False,
                            "formal_matrix_complete": False,
                            "test_used_for_selection": False,
                            **{metric: float(daily[metric][index]) for metric in METRICS},
                            "coverage": float(daily["coverage"][index])})
    write_csv(run_dir / "metrics_per_date.csv", metric_rows)
    status.update({"status": "COMPLETED", "returncode": 0, "scenario_file": str(scenario_files[0]),
                   "scenario_sha256": file_sha256(scenario_files[0]), "metrics_file": str(run_dir / "metrics_per_date.csv"),
                   "metrics_sha256": file_sha256(run_dir / "metrics_per_date.csv"),
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": time.perf_counter() - started})
    atomic_write_json(status_path, status)
    return status


def run_formal(protocol, config_path, source_csv, output_dir, device="auto", workers=1,
               retries=0, origin_ids=None, seeds=None, models=None, resume=True,
               command=None, timeout_seconds=None, artifact_role="formal_task"):
    """Execute selected registered tasks; selection enables acceptance runs and resume."""
    output_dir = Path(output_dir)
    config = protocol["training"]["formal"]
    rows = build_run_manifest(protocol, artifact_role=artifact_role)
    if origin_ids:
        rows = [row for row in rows if row["origin_id"] in set(origin_ids)]
    if seeds is not None:
        rows = [row for row in rows if row["seed"] in set(seeds)]
    if models:
        rows = [row for row in rows if row["model"] in set(models)]
    bundle_dir = output_dir / "bundles"
    bundle_metadata = {}
    for origin in protocol["origins"]:
        if not origin_ids or origin["origin_id"] in set(origin_ids):
            bundle_metadata[origin["origin_id"]] = build_origin_bundle(
                protocol, origin["origin_id"], source_csv,
                bundle_dir / (origin["origin_id"] + ".npz"))
    formal_config_hash = config_sha256(config_path)
    provenance = {**build_provenance(config_path, source_csv, command),
                  "config_sha256": formal_config_hash,
                  "training_backend": "classical", "negative_phase_backend": "internal", "sampler": "sa",
                  "formal_training_config": config, "model_configs": config.get("model_configs", {}),
                  "origin_bundle_metadata": bundle_metadata,
                  "analysis_primary": "raw", "sensitivity_modes": ["calibrated", "temporal_ecc"],
                  "calibrated_status": "NOT RUN", "temporal_ecc_status": "NOT RUN",
                  "test_peeking": False, "n_tasks": len(rows), "device": device}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "task_manifest.json", {
        "provenance": provenance, "tasks": rows, "rows": rows,
        "formal_run_status": "NOT RUN", "artifact_role": artifact_role,
        "formal_matrix_complete": False,
    })
    statuses = []
    if workers <= 1:
        statuses = [_run_formal_task(row, bundle_dir, output_dir, protocol, config, device, retries,
                                     formal_config_hash, provenance, timeout_seconds=timeout_seconds,
                                     resume=resume, artifact_role=artifact_role) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_run_formal_task, row, bundle_dir, output_dir, protocol, config, device,
                                       retries, formal_config_hash, provenance, timeout_seconds, resume,
                                       artifact_role)
                       for row in rows]
            statuses = [future.result() for future in as_completed(futures)]
    metric_files = [Path(status["metrics_file"]) for status in statuses if status.get("status") == "COMPLETED" and status.get("metrics_file")]
    metric_rows = []
    for metric_file in metric_files:
        with metric_file.open(newline="", encoding="utf-8") as handle:
            metric_rows.extend(csv.DictReader(handle))
    write_csv(output_dir / "metrics_per_date.csv", metric_rows)
    complete_matrix = False
    matrix_summary = None
    completeness_error = None
    if artifact_role == "formal_task" and len(rows) == 60:
        try:
            matrix_summary = validate_formal_matrix(metric_rows, statuses, protocol)
            complete_matrix = True
        except ValueError as error:
            completeness_error = str(error)
    if artifact_role == "acceptance":
        run_status = "ACCEPTANCE_COMPLETED" if statuses and all(
            status.get("status") == "COMPLETED" for status in statuses) else "INCOMPLETE"
    else:
        run_status = "COMPLETED" if complete_matrix else "INCOMPLETE"
    result = {"provenance": provenance, "status": statuses,
              "n_completed": sum(s.get("status") == "COMPLETED" for s in statuses),
              "n_failed": sum(s.get("status") == "FAILED" for s in statuses),
              "formal_run_status": run_status,
              "artifact_role": "formal_complete" if complete_matrix else artifact_role,
              "formal_matrix_complete": complete_matrix,
              "smoke_only": False, "synthetic_data": False,
              "eligible_for_paper": complete_matrix,
              "matrix_summary": matrix_summary,
              "completeness_error": completeness_error}
    task_manifest = {"provenance": provenance, "tasks": [], "rows": [],
                     "formal_run_status": "NOT RUN", "artifact_role": artifact_role,
                     "formal_matrix_complete": False}
    status_by_key = {(status.get("origin_id"), status.get("model"), int(status.get("seed"))): status
                     for status in statuses}
    for row in rows:
        key = (row["origin_id"], row["model"], int(row["seed"]))
        task_manifest["tasks"].append({**row, **status_by_key.get(key, {"status": "NOT RUN"})})
    task_manifest["rows"] = task_manifest["tasks"]
    task_manifest["formal_run_status"] = run_status
    task_manifest["artifact_role"] = result["artifact_role"]
    task_manifest["formal_matrix_complete"] = complete_matrix
    atomic_write_json(output_dir / "task_manifest.json", task_manifest)
    atomic_write_json(output_dir / "run_manifest.json", {
        "protocol": protocol, "provenance": provenance,
        "tasks": task_manifest["tasks"], "rows": task_manifest["tasks"],
        "formal_run_status": run_status,
        "artifact_role": result["artifact_role"],
        "formal_matrix_complete": complete_matrix,
    })
    atomic_write_json(output_dir / "formal_status.json", result)
    result["metrics_sha256"] = file_sha256(output_dir / "metrics_per_date.csv") if metric_rows else None
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    atomic_write_json(output_dir / "formal_status.json", result)
    return result


def write_acceptance_report(output_dir, result):
    """Write a factual acceptance report without turning it into a paper result."""
    output_dir = Path(output_dir)
    statuses = result.get("status", [])
    lines = [
        "# OPSD Formal Acceptance Report",
        "",
        "This is an origin_1 / seed_0 code-and-data acceptance run. It is not the",
        "60-task paper experiment and must not be interpreted as a model ranking.",
        "",
        "## Acceptance criteria",
        "",
        "- Formal settings, not the 3-epoch pilot, were used.",
        "- Raw per-date metrics were written for every completed model.",
        "- The raw OPSD SHA-256 and LS-only normalization metadata were recorded.",
        "- Calibrated and temporal-ECC sensitivity analyses remain NOT RUN.",
        "- `artifact_role=acceptance`, `eligible_for_paper=false`, and "
        "`formal_matrix_complete=false` are enforced.",
        "",
        "## Task status",
        "",
        "| Model | Status | Attempts | Elapsed (s) | Output |",
        "|---|---:|---:|---:|---|",
    ]
    for status in statuses:
        elapsed = status.get("elapsed_seconds")
        elapsed_text = "{:.1f}".format(elapsed) if isinstance(elapsed, (int, float)) else "NOT RUN"
        lines.append("| {model} | {state} | {attempts} | {elapsed} | `{metrics}` |".format(
            model=status.get("model", ""), state=status.get("status", "UNKNOWN"),
            attempts=status.get("attempts", ""), elapsed=elapsed_text,
            metrics=status.get("metrics_file", "NOT WRITTEN")))
    lines.extend([
        "",
        "## Resource monitoring",
        "",
        "Wall-clock time is recorded per task. Peak child-process GPU/CPU memory was "
        "not instrumented by this Windows runner and is therefore reported as "
        "`NOT RECORDED`, not inferred.",
        "",
        "## Outputs",
        "",
        "- `formal_status.json`",
        "- `task_manifest.json`",
        "- `metrics_per_date.csv` (raw per-date metrics only)",
        "- per-task `stdout.log`, `stderr.log`, `status.json`, and scenario files",
        "- training curves are retained in each model directory when produced by the model",
        "",
        "## Result classification",
        "",
        "All acceptance files are ineligible for paper aggregation and must not be "
        "placed in the main comparison table. Only a complete 60-task matrix can "
        "produce `artifact_role=formal_complete`.",
    ])
    curve_files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*")
                         if path.is_file() and ("loss" in path.name.lower() or "history" in path.name.lower()))
    lines.extend(["", "## Training curves", ""])
    if curve_files:
        lines.extend("- `%s`" % path for path in curve_files)
    else:
        lines.append("- NOT FOUND")
    (output_dir / "acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _date_blocks(rows):
    keys = sorted({(str(row["origin_id"]), str(row["date"])) for row in rows})
    index = {key: idx for idx, key in enumerate(keys)}
    return keys, index


def _origin_blocks(values, origin_ids, block_length):
    values = np.asarray(values, dtype=np.float64)
    origin_ids = np.asarray(origin_ids, dtype=object)
    if values.ndim != 1 or origin_ids.shape != values.shape:
        raise ValueError("values and origin_ids must be aligned one-dimensional arrays")
    blocks = []
    for origin in dict.fromkeys(origin_ids.tolist()):
        indices = np.flatnonzero(origin_ids == origin)
        if not len(indices):
            continue
        blocks.extend(indices[start:start + block_length]
                     for start in range(0, len(indices), block_length))
    return blocks


def _block_resample(values, origin_ids, block_length, rng):
    values = np.asarray(values, dtype=np.float64)
    origin_ids = np.asarray(origin_ids, dtype=object)
    sampled = []
    for origin in dict.fromkeys(origin_ids.tolist()):
        indices = np.flatnonzero(origin_ids == origin)
        n_blocks = int(np.ceil(len(indices) / block_length))
        starts = rng.integers(0, len(indices), size=n_blocks)
        chosen = np.concatenate([
            indices[(start + np.arange(block_length)) % len(indices)] for start in starts
        ])[:len(indices)]
        sampled.append(values[chosen])
    return np.concatenate(sampled)


def paired_inference(differences, bootstrap_repetitions=2000,
                     permutation_repetitions=2000, seed=2026, block_length=7,
                     origin_ids=None):
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0 or not np.isfinite(differences).all():
        raise ValueError("paired differences must be a finite non-empty vector")
    if origin_ids is None:
        origin_ids = np.zeros(differences.size, dtype=object)
    origin_ids = np.asarray(origin_ids, dtype=object)
    if origin_ids.shape != differences.shape:
        raise ValueError("origin_ids must align with paired differences")
    blocks = _origin_blocks(differences, origin_ids, block_length)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        _block_resample(differences, origin_ids, block_length, rng).mean()
        for _ in range(bootstrap_repetitions)
    ])
    block_sums = np.asarray([differences[block].sum() for block in blocks])
    block_sizes = np.asarray([len(block) for block in blocks], dtype=np.float64)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutation_repetitions, len(blocks)))
    null = (signs * block_sums[None, :]).sum(axis=1) / block_sizes.sum()
    observed = float(differences.mean())
    return {
        "mean_difference": observed,
        "ci_2.5": float(np.quantile(boot, 0.025)),
        "ci_97.5": float(np.quantile(boot, 0.975)),
        "paired_sign_flip_p": float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) /
                                      (permutation_repetitions + 1)),
        "paired_cohen_d": float(observed / (differences.std(ddof=1) + 1e-12))
        if differences.size > 1 else 0.0,
        "n_dates": int(differences.size),
        "n_date_blocks": int(len(blocks)),
        "block_length_days": int(block_length),
        "block_origin_ids": sorted(set(str(value) for value in origin_ids.tolist())),
    }


def validate_formal_matrix(rows, statuses, protocol):
    """Fail closed unless all registered tasks and 90 test dates are present."""
    expected_origins = set(REGISTERED_ORIGINS)
    expected_models = set(REGISTERED_MODELS)
    expected_seeds = set(REGISTERED_SEEDS)
    status_keys = {(row.get("origin_id"), row.get("model"), int(row.get("seed")))
                   for row in statuses}
    expected_keys = {(origin, model, seed) for origin in expected_origins
                     for model in expected_models for seed in expected_seeds}
    if status_keys != expected_keys or len(statuses) != 60:
        raise ValueError("formal matrix requires exactly 60 registered task statuses")
    if any(row.get("status") != "COMPLETED" for row in statuses):
        raise ValueError("formal matrix requires every task status to be COMPLETED")
    grouped = {}
    for row in rows:
        if row.get("artifact_role") != "formal_task":
            raise ValueError("formal metric rows must have artifact_role=formal_task")
        key = (row.get("origin_id"), row.get("model"), int(row.get("seed")))
        grouped.setdefault(key, []).append(str(row.get("date")))
    if set(grouped) != expected_keys or len(rows) != 60 * 90:
        raise ValueError("formal matrix requires exactly 5400 metric rows")
    for origin in protocol["origins"]:
        expected_dates = {(date.fromisoformat(origin["test_start"]) + timedelta(days=index)).isoformat()
                          for index in range(90)}
        for model in expected_models:
            for seed in expected_seeds:
                dates = grouped[(origin["origin_id"], model, seed)]
                if len(dates) != 90 or len(set(dates)) != 90 or set(dates) != expected_dates:
                    raise ValueError("incomplete or duplicated TEST dates for %s/%s/seed_%s" %
                                     (origin["origin_id"], model, seed))
    return {"origins": sorted(expected_origins), "models": sorted(expected_models),
            "seeds": sorted(expected_seeds), "n_tasks": 60, "n_metric_rows": 5400}


def aggregate_metric_rows(rows, reference_model="FA-BM-VAE", bootstrap_repetitions=2000,
                          permutation_repetitions=2000, seed=2026):
    """Aggregate per-date rows and compare every model with FA-BM-VAE.

    Input rows must contain origin_id, date, model, seed and the six metric
    columns.  Seed means are formed before inference, while inference samples
    shared origin/date blocks, preserving the paired design.
    """
    if not rows:
        raise ValueError("No metric rows supplied")
    for row in rows:
        if parse_bool(row.get("smoke_only"), "smoke_only") is True or parse_bool(
            row.get("synthetic_data"), "synthetic_data"
        ) is True or ("eligible_for_paper" in row and parse_bool(
            row.get("eligible_for_paper"), "eligible_for_paper") is False):
            raise ValueError("Formal OPSD aggregation refuses smoke/synthetic rows")
        missing = {"origin_id", "date", "model", "seed"}.union(METRICS) - set(row)
        if missing:
            raise ValueError("Metric row missing %s" % sorted(missing))
    grouped = {}
    for row in rows:
        key = (str(row["origin_id"]), str(row["date"]), str(row["model"]))
        grouped.setdefault(key, {metric: [] for metric in METRICS})
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError("Non-finite %s" % metric)
            grouped[key][metric].append(value)
    means = {}
    per_run = []
    for key, values in grouped.items():
        means[key] = {metric: float(np.mean(values[metric])) for metric in METRICS}
        per_run.append({"origin_id": key[0], "date": key[1], "model": key[2],
                        "n_seeds": len(values[METRICS[0]]), **means[key]})
    models = sorted({key[2] for key in means})
    if reference_model not in models:
        raise ValueError("Reference model %s is absent" % reference_model)
    model_keys = {model: {(key[0], key[1]) for key in means if key[2] == model}
                  for model in models}
    reference_keys = model_keys[reference_model]
    if any(keys != reference_keys for model, keys in model_keys.items()
           if model != reference_model):
        raise ValueError("Incomplete model pairing: date keys must match the reference model")
    output = []
    for model in models:
        model_rows = [row for row in per_run if row["model"] == model]
        model_origin_ids = [row["origin_id"] for row in model_rows]
        n_blocks = len(_origin_blocks(np.zeros(len(model_rows)), model_origin_ids, 7))
        for metric in METRICS:
            output.append({
                "model": model,
                "metric": metric,
                "mean": float(np.mean([row[metric] for row in model_rows])),
                "std": float(np.std([row[metric] for row in model_rows], ddof=1))
                if len(model_rows) > 1 else 0.0,
                "n_dates": len(model_rows),
                "n_date_blocks": n_blocks,
            })
    comparisons = []
    keys = sorted({(row["origin_id"], row["date"]) for row in per_run})
    for model in models:
        if model == reference_model:
            continue
        for metric in METRICS:
            differences = []
            difference_origins = []
            for origin_id, day in keys:
                left = means.get((origin_id, day, model))
                right = means.get((origin_id, day, reference_model))
                if left is not None and right is not None:
                    differences.append(left[metric] - right[metric])
                    difference_origins.append(origin_id)
            if differences:
                primary = paired_inference(differences, bootstrap_repetitions,
                                           permutation_repetitions, seed, block_length=7,
                                           origin_ids=difference_origins)
                comparisons.append({
                    "model": model,
                    "reference_model": reference_model,
                    "metric": metric,
                    **primary,
                    "block_bootstrap_14": paired_inference(
                        differences, bootstrap_repetitions, permutation_repetitions,
                        seed, block_length=14, origin_ids=difference_origins),
                })
    for metric in METRICS:
        family = [row for row in comparisons if row["metric"] == metric]
        previous = 0.0
        for rank, row in enumerate(sorted(family, key=lambda item: item["paired_sign_flip_p"])):
            adjusted = min(1.0, row["paired_sign_flip_p"] * (len(family) - rank))
            row["holm_p"] = max(previous, adjusted)
            previous = row["holm_p"]
    return {"per_date_seed_means": per_run, "model_metrics": output,
            "paired_comparisons": comparisons}


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoke_rows(protocol):
    rng = np.random.default_rng(2026)
    rows = []
    models = protocol["models"]
    for origin_index, origin in enumerate(protocol["origins"]):
        start = _parse_day(origin["test_start"])
        for day_index in range(6):
            day = (start + timedelta(days=day_index)).isoformat()
            for model_index, model in enumerate(models):
                for seed in protocol["seeds"]:
                    base = 0.10 + 0.005 * model_index
                    rows.append({
                        "origin_id": origin["origin_id"], "date": day,
                        "model": model, "seed": seed,
                        "CRPS_raw": base + rng.normal(0, 0.001),
                        "Energy": base + 0.01 + rng.normal(0, 0.001),
                        "Variogram": base + 0.02 + rng.normal(0, 0.001),
                        "ramp_CRPS": base + 0.03 + rng.normal(0, 0.001),
                        "PICP90": 0.90 - 0.01 * model_index + rng.normal(0, 0.001),
                        "interval_width90": 0.20 + 0.01 * model_index + rng.normal(0, 0.001),
                        "MAQCE": 0.02 + 0.002 * model_index + rng.normal(0, 0.0005),
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--raw-source", type=Path,
                        default=ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_time_series_60min_2019-06-05.csv")
    parser.add_argument("--pilot", action="store_true", help="Build real origin bundles and emit pilot commands; does not train")
    parser.add_argument("--run-pilot", action="store_true", help="Run only real origin_1/seed_0 for the four registered models")
    parser.add_argument("--pilot-epochs", type=int, default=None, help="Optional explicit pilot override; never used by formal runs")
    parser.add_argument("--acceptance", action="store_true", help="Run origin_1/seed_0 for all four models with formal settings")
    parser.add_argument("--origin", dest="origins", action="append", choices=["origin_1", "origin_2", "origin_3"])
    parser.add_argument("--seed", dest="seeds", action="append", type=int)
    parser.add_argument("--model", dest="models", action="append", choices=["FA-BM-VAE", "Forecast-anchored Gaussian", "GMM-4", "Score-SDE"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=0,
                        help="Per-task timeout; 0 records NOT_SET and imposes no timeout.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.config)
    if args.execute and not args.confirm_formal:
        raise SystemExit("Formal execution requires --confirm-formal; no run was started")
    if args.execute and (args.pilot or args.run_pilot):
        raise SystemExit("Choose one of --execute and --pilot")
    if args.workers < 1 or args.retries < 0 or args.timeout_seconds < 0:
        raise SystemExit("--workers must be positive; retries/timeout must be non-negative")
    if args.workers > 1 and args.device in ("auto", "cuda"):
        raise SystemExit("GPU execution is limited to workers=1 on this single-GPU protocol")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(protocol, args.dataset_bundle,
                                  artifact_role="acceptance" if args.acceptance else "formal_task")
    atomic_write_json(output_dir / "run_manifest.json", {
        "protocol": protocol, "windows": origin_windows(protocol),
        "tasks": manifest, "rows": manifest, "formal_run_status": "NOT RUN",
        "artifact_role": "acceptance" if args.acceptance else "formal_task",
        "formal_matrix_complete": False,
    })
    if args.smoke:
        rows = smoke_rows(protocol)
        result = aggregate_metric_rows(rows, bootstrap_repetitions=300,
                                       permutation_repetitions=300)
        write_csv(output_dir / "smoke_per_date_seed_means.csv", result["per_date_seed_means"])
        write_csv(output_dir / "smoke_model_metrics.csv", result["model_metrics"])
        write_csv(output_dir / "smoke_paired_comparisons.csv", result["paired_comparisons"])
        (output_dir / "smoke_summary.json").write_text(json.dumps({
            "smoke_only": True, "synthetic_data": True,
            "eligible_for_paper": False,
            "hardware_claim": False, "physical_platform_used": False,
            "n_manifest_rows": len(manifest), "n_metric_rows": len(rows),
            "formal_run_status": "NOT RUN",
        }, indent=2), encoding="utf-8")
        print("Wrote OPSD rolling-origin smoke outputs to %s" % output_dir.resolve())
    elif args.run_pilot:
        summary = run_real_pilot(protocol, args.raw_source, output_dir, seed=0,
                                 epochs=args.pilot_epochs or 3)
        print("Completed real OPSD origin_1/seed_0 pilot: %s" % (output_dir / "pilot_summary.json"))
    elif args.pilot:
        bundle_dir = output_dir / "pilot_bundles"
        for origin in protocol["origins"]:
            metadata = build_origin_bundle(protocol, origin["origin_id"], args.raw_source,
                                           bundle_dir / (origin["origin_id"] + ".npz"))
            print(json.dumps({"origin_id": origin["origin_id"], "bundle": str(bundle_dir / (origin["origin_id"] + ".npz")),
                              "split_days": metadata["split_days"], "source_sha256": metadata["source_sha256"]}))
        print("Built real OPSD origin bundles; model training NOT RUN")
    elif args.execute:
        if args.acceptance:
            origins, seeds, models = ["origin_1"], [0], protocol["models"]
        else:
            origins, seeds, models = args.origins, args.seeds, args.models
        result = run_formal(protocol, args.config, args.raw_source, output_dir,
                            device=args.device, workers=args.workers, retries=args.retries,
                            origin_ids=origins, seeds=seeds, models=models,
                            resume=args.resume, command=[sys.executable, "-m",
                            "GEFcom2014.forecast_quality.opsd_rolling_origins"] + sys.argv[1:],
                            timeout_seconds=args.timeout_seconds or None,
                            artifact_role="acceptance" if args.acceptance else "formal_task")
        if args.acceptance:
            write_acceptance_report(output_dir, result)
        print(json.dumps({"output_dir": str(output_dir.resolve()),
                          "n_completed": result["n_completed"], "n_failed": result["n_failed"]}, indent=2))
    else:
        print("Wrote OPSD rolling-origin manifest to %s; formal training NOT RUN" % output_dir.resolve())


if __name__ == "__main__":
    main()
