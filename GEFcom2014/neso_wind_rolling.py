# -*- coding: utf-8 -*-
"""Freeze four chronological NESO rolling windows before any training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models.QBM_VAE.run_neso_wind_jobs import CONFIGS, jobs


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "export/fa_bm_vae_neso_wind_classical_v1"
V2 = ROOT / "export/fa_bm_vae_neso_wind_classical_v2"
MODELS = ("fa_bm_vae", "anchor_gaussian", "anchor_gmm4", "score_sde",
          "spline_flow", "residual_bootstrap", "j0")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_windows(eligible_dates, min_fraction=0.8, min_vs=60, min_ls=365):
    eligible = {pd.Timestamp(d) for d in eligible_dates}
    starts = pd.date_range(min(eligible).replace(day=1), max(eligible), freq="MS")
    feasible = []
    for start in starts:
        windows = []
        for quarter in range(4):
            test_start = start + pd.DateOffset(months=3 * quarter)
            test_end = test_start + pd.DateOffset(months=3)
            vs_start = test_start - pd.DateOffset(months=3)
            calendar = pd.date_range(test_start, test_end - pd.Timedelta(days=1), freq="D")
            test_dates = [d for d in calendar if d in eligible]
            vs_dates = [d for d in pd.date_range(vs_start, test_start - pd.Timedelta(days=1), freq="D") if d in eligible]
            ls_dates = sorted(d for d in eligible if d < vs_start)
            if len(test_dates) / len(calendar) < min_fraction or len(vs_dates) < min_vs or len(ls_dates) < min_ls:
                break
            windows.append({"test_start": str(test_start.date()), "test_end": str((test_end - pd.Timedelta(days=1)).date()),
                            "vs_start": str(vs_start.date()), "coverage": len(test_dates) / len(calendar),
                            "calendar_days": len(calendar), "ls": [str(d.date()) for d in ls_dates],
                            "vs": [str(d.date()) for d in vs_dates], "test": [str(d.date()) for d in test_dates]})
        if len(windows) == 4:
            feasible.append((start, windows))
    if not feasible:
        raise ValueError("No twelve-month four-quarter interval satisfies coverage and rolling LS/VS rules")
    return feasible[-1][1], len(feasible)


def prepare():
    source_bundle = V2 / "data/neso_wind_daily.npz"
    audit_csv = V2 / "date_eligibility.csv"
    if not source_bundle.is_file() or not audit_csv.is_file():
        raise FileNotFoundError("Complete the v2 source audit first")
    if (V2 / "protocol_v2.json").exists() or (V2 / "rolling").exists():
        raise FileExistsError("Rolling protocol or output already exists; refusing to overwrite")
    arrays, source_meta, date_arrays = load_daily_bundle(source_bundle)
    all_dates = np.concatenate([date_arrays[s] for s in ("ls", "vs", "test")])
    all_x = np.concatenate([arrays["x_" + s] for s in ("ls", "vs", "test")])
    all_y = np.concatenate([arrays["y_" + s] for s in ("ls", "vs", "test")])
    if len(set(all_dates)) != len(all_dates) or list(all_dates) != sorted(all_dates):
        raise ValueError("Source bundle dates must be unique and chronological")
    audit = pd.read_csv(audit_csv)
    eligible = audit.loc[audit.status == "eligible", "date"].tolist()
    if set(eligible) != set(all_dates):
        raise ValueError("Source audit and bundle eligible dates differ")
    windows, n_feasible = choose_windows(eligible)
    v1_dates = {s: set(date_arrays[s]) for s in ("ls", "vs", "test")}
    rolling = V2 / "rolling"
    rolling.mkdir(parents=True)
    mapping = {date: i for i, date in enumerate(all_dates)}
    frozen_windows, tasks = [], []
    for ordinal, window in enumerate(windows, 1):
        name = "quarter_%d" % ordinal
        home = rolling / name
        data_dir, config_dir = home / "data", home / "configs"
        data_dir.mkdir(parents=True)
        config_dir.mkdir()
        bundle = data_dir / "neso_wind_daily.npz"
        payload, split_dates = {}, {}
        for split in ("ls", "vs", "test"):
            dates = window[split]
            indices = np.asarray([mapping[d] for d in dates], dtype=int)
            payload["x_" + split] = all_x[indices]
            payload["y_" + split] = all_y[indices]
            payload["dates_" + split] = np.asarray(dates, dtype="U10")
            split_dates[split] = dates
        ls_y = payload["y_ls"]
        metadata = {key: source_meta[key] for key in ("dataset_name", "source", "source_urls", "source_files",
                     "forecast_fields", "actual_fields", "target_scope", "official_scope_note", "forecast_publication",
                     "timezone", "aggregation", "context_features", "anchor_contract", "target_features", "power_normalization")}
        metadata.update({"rolling_window": name, "split_days": {s: len(split_dates[s]) for s in split_dates},
                         "dates_by_split": split_dates, "ls_target_range_mw": [float(ls_y.min()), float(ls_y.max())],
                         "target_outside_ls_range_fraction": {
                             s: float(np.mean((payload["y_" + s] < ls_y.min()) | (payload["y_" + s] > ls_y.max())))
                             for s in split_dates}})
        payload["metadata"] = json.dumps(metadata)
        np.savez_compressed(bundle, **payload)
        bundle.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for key in ("fa_bm_vae", "j0"):
            config = json.loads(CONFIGS[key].read_text(encoding="utf-8"))
            config["protocol_version"] = "neso-rolling-four-season-v2"
            config["dataset_bundle"] = str(bundle.relative_to(ROOT))
            config["run_label"] = "neso_rolling_%s" % name
            config.pop("historical_graph_json_note", None)
            (config_dir / CONFIGS[key].name).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame({"date": window["ls"] + window["vs"] + window["test"],
                      "split": (["LS"] * len(window["ls"]) + ["VS"] * len(window["vs"]) + ["TEST"] * len(window["test"]))}).to_csv(
                          home / "dates.csv", index=False)
        task_list = jobs("formal", "formal", home, bundle, MODELS, config_dir)
        tasks.extend(task_list)
        frozen_windows.append({"name": name, "test_start": window["test_start"], "test_end": window["test_end"],
                               "vs_start": window["vs_start"], "coverage": window["coverage"],
                               "calendar_days": window["calendar_days"],
                               "split_days": {s: len(window[s]) for s in ("ls", "vs", "test")},
                               "dates": split_dates, "bundle": str(bundle.relative_to(ROOT)), "bundle_sha256": sha256(bundle),
                               "configs": {key: sha256(config_dir / CONFIGS[key].name) for key in ("fa_bm_vae", "j0")},
                               "overlap_with_old_splits_test_days": {
                                   s: len(set(window["test"]) & v1_dates[s]) for s in v1_dates}})
    if len(tasks) != 84 or len({task["output_dir"] for task in tasks}) != 84:
        raise ValueError("Expected 84 isolated task output paths")
    source_paths = ["GEFcom2014/neso_wind_dataset.py", "GEFcom2014/neso_wind_rolling.py",
                    "GEFcom2014/external_datasets.py", "GEFcom2014/models/utils_models.py",
                    "GEFcom2014/models/QBM_VAE/run_neso_wind_jobs.py",
                    "GEFcom2014/models/QBM_VAE/qbm_vae.py", "GEFcom2014/models/QBM_VAE/utils_qbm_vae.py",
                    "GEFcom2014/models/QBM_VAE/forecast_anchor.py",
                    "GEFcom2014/models/QBM_VAE/anchor_gaussian_baseline.py",
                    "GEFcom2014/models/QBM_VAE/anchor_score_sde.py",
                    "GEFcom2014/models/QBM_VAE/anchor_spline_flow.py",
                    "GEFcom2014/models/probabilistic_baselines.py",
                    "GEFcom2014/forecast_quality/temporal_rank_coupling.py",
                    "GEFcom2014/forecast_quality/compare_scenarios.py",
                    "GEFcom2014/forecast_quality/unified_postprocessing.py",
                    "GEFcom2014/forecast_quality/qbm_temporal_ecc.py",
                    "GEFcom2014/forecast_quality/neso_wind_results.py",
                    "GEFcom2014/forecast_quality/neso_v2_analysis.py"]
    protocol = {"protocol_version": "neso-rolling-four-season-v2", "status": "FROZEN_BEFORE_TRAINING",
                "selection_rule": "latest twelve consecutive calendar months with four nonoverlapping three-month test quarters, each >=80% eligible dates; each VS is preceding three calendar months with >=60 eligible dates; LS is all earlier eligible dates with >=365 days",
                "n_feasible_start_months": n_feasible, "windows": frozen_windows,
                "models": list(MODELS), "seeds": [0, 1, 2], "n_scenarios_per_date": 100,
                "primary_metric": "Cal CRPS_MW", "key_secondary_metric": "Cal Energy_Score_MW",
                "other_variants": ["Raw", "Cal+ECC"],
                "protocol_note": "Post-v1, first-round-result-informed historical repartition robustness analysis; v1 Raw primary protocol remains unchanged. Previously viewed dates are not an independent confirmation sample.",
                "statistics": "Per-date scores averaged across seeds, paired seven-calendar-day block bootstrap, 5000 replicates, segments stop at missing dates and window boundaries; no p-values",
                "calibration": "fit per model/seed/window on that window's VS only; no truth clipping",
                "ecc": "common FA-BM-VAE template from same seed and window; supplementary only",
                "source_bundle_sha256": sha256(source_bundle), "eligibility_sha256": sha256(audit_csv),
                "code_hashes": {path: sha256(ROOT / path) for path in source_paths}}
    protocol_path = V2 / "protocol_v2.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (V2 / "protocol_v2.sha256").write_text(sha256(protocol_path) + "\n", encoding="ascii")
    (V2 / "task_manifest.json").write_text(json.dumps({"protocol_sha256": sha256(protocol_path),
                   "status": "NOT RUN", "tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    return protocol


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    frozen = prepare()
    print(json.dumps({"windows": [{k: x[k] for k in ("name", "test_start", "test_end", "split_days", "coverage")}
                                   for x in frozen["windows"]], "tasks": 84}, indent=2))
