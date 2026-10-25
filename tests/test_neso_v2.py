import json
import numpy as np
import pandas as pd
import pytest

from GEFcom2014.forecast_quality import neso_v2_analysis as analysis
from GEFcom2014.forecast_quality.neso_v2_analysis import _validated_daily
from GEFcom2014.forecast_quality.neso_wind_results import DAILY_METRICS, MODEL_KEYS, _pair_bootstrap
from GEFcom2014.models.QBM_VAE.run_neso_wind_jobs import CONFIGS, SOURCES, jobs
from GEFcom2014.neso_wind_rolling import MODELS, choose_windows


def test_latest_four_season_window_is_chosen_from_dates_only():
    eligible = pd.date_range("2019-01-01", "2021-12-31", freq="D").strftime("%Y-%m-%d")
    windows, count = choose_windows(eligible)
    assert count > 0
    assert windows[0]["test_start"] == "2021-01-01"
    assert windows[-1]["test_end"] == "2021-12-31"
    assert all(len(w["vs"]) >= 60 and len(w["ls"]) >= 365 for w in windows)


def test_calendar_bootstrap_never_forms_a_block_across_a_gap():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03",
             "2024-07-01", "2024-07-02", "2024-07-03"]
    _, _, replicates = _pair_bootstrap([0, 0, 0, 10, 10, 10], [0] * 6,
                                       dates, repetitions=400, block_days=3,
                                       return_replicates=True)
    assert set(np.unique(replicates)) <= {0.0, 5.0, 10.0}


def test_incomplete_model_seed_date_grid_cannot_enter_table(tmp_path):
    dates = ["2024-01-01", "2024-01-02"]
    rows = [dict(model=model, seed=seed, variant=variant, date=date,
                 CRPS_MW=1.0, Energy_Score_MW=1.0, Variogram_Score=1.0,
                 Ramp_CRPS_MW=1.0, PICP90=0.9, interval_width90_MW=1.0,
                 MAQCE=0.1, mean_bias_MW=0.0, Reliability_MAE_pp=1.0,
                 Ramp_quantile_MAE_daily_MW=1.0)
            for model in ("FA-BM-VAE", "Forecast-anchored Gaussian")
            for seed in (0, 1, 2) for variant in ("Raw", "Cal", "Cal+ECC") for date in dates]
    path = tmp_path / "daily.csv"
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exact finite"):
        _validated_daily(path, dates, ("FA-BM-VAE", "Forecast-anchored Gaussian"))


def test_rolling_formal_jobs_have_21_distinct_output_paths(tmp_path):
    tasks = jobs("formal", "formal", tmp_path / "window", tmp_path / "bundle.npz", MODELS,
                 tmp_path / "configs")
    assert len(tasks) == 21
    assert len({task["output_dir"] for task in tasks}) == 21
    assert {(task["training_seed"], task["generation_seed"]) for task in tasks} == {(0, 0), (1, 1), (2, 2)}
    assert all(task["n_scenarios"] == 100 for task in tasks)


def _synthetic_formal_window(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "V2", tmp_path)
    home = tmp_path / "rolling/quarter_1"
    bundle = home / "data/neso_wind_daily.npz"
    bundle.parent.mkdir(parents=True)
    split_dates = {"ls": ["2024-01-01"], "vs": ["2024-01-02"], "test": ["2024-01-03", "2024-01-04"]}
    arrays = {}
    for split, dates in split_dates.items():
        arrays["x_" + split] = np.zeros((len(dates), 33), dtype=np.float32)
        arrays["y_" + split] = np.zeros((len(dates), 24), dtype=np.float32)
        arrays["dates_" + split] = np.asarray(dates)
    np.savez_compressed(bundle, **arrays, metadata="{}")
    config_hashes = {}
    for model in ("fa_bm_vae", "j0"):
        path = home / "configs" / CONFIGS[model].name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model, encoding="utf-8")
        config_hashes[model] = analysis.sha256(path)
    formal = home / "formal"
    result = formal / "results"
    result.mkdir(parents=True)
    model_keys = [MODEL_KEYS[model] for model in analysis.ROLLING_MODELS]
    tasks, completed, scenario_hashes = [], [], {}
    for model in model_keys:
        for seed in (0, 1, 2):
            out = formal / model / ("seed_%d" % seed)
            out.mkdir(parents=True)
            task = {"model": model, "training_seed": seed, "generation_seed": seed,
                    "n_scenarios": 100, "scenario_splits": ["VS", "TEST"],
                    "output_dir": str(out), "command": ["python", model, str(seed)]}
            tasks.append(task)
            completed.append({"model": model, "training_seed": seed, "status": "COMPLETED",
                              "return_code": 0, "command": task["command"], "output_dir": str(out)})
            for split in ("VS", "TEST"):
                path = out / ("synthetic_100_%s.pickle" % split)
                path.write_bytes(b"synthetic-only-" + model.encode() + str(seed).encode() + split.encode())
                scenario_hashes[str(path.relative_to(formal))] = analysis.sha256(path)
    code_hashes = {source: "synthetic-hash-" + source for source in SOURCES}
    code_hashes["GEFcom2014/forecast_quality/neso_wind_results.py"] = "synthetic-evaluator"
    protocol = {"models": model_keys, "code_hashes": code_hashes}
    window = {"name": "quarter_1", "bundle": str(bundle), "bundle_sha256": analysis.sha256(bundle),
              "configs": config_hashes, "dates": split_dates}
    (formal / "task_manifest.json").write_text(json.dumps({"status": "COMPLETED",
        "bundle": str(bundle), "bundle_sha256": window["bundle_sha256"],
        "config_hashes": config_hashes, "source_hashes": {s: code_hashes[s] for s in SOURCES},
        "jobs": tasks, "results": completed}), encoding="utf-8")
    daily = [dict(model=model, seed=seed, variant=variant, date=date,
                  **{metric: 1.0 for metric in DAILY_METRICS})
             for model in analysis.ROLLING_MODELS for seed in (0, 1, 2)
             for variant in analysis.VARIANTS for date in split_dates["test"]]
    pd.DataFrame(daily).to_csv(result / "metrics_daily.csv", index=False)
    paired = [dict(model="FA-BM-VAE", comparator=model, variant=variant, metric=metric,
                   mean_difference_main_minus_comparator=0.0,
                   block_bootstrap_ci95_low=-1.0, block_bootstrap_ci95_high=1.0)
              for model in analysis.ROLLING_MODELS[1:] for variant in analysis.VARIANTS
              for metric in analysis.KEY_METRICS]
    paired_path = result / "paired_comparisons.csv"
    pd.DataFrame(paired).to_csv(paired_path, index=False)
    (result / "metrics_per_seed.csv").write_text("synthetic only", encoding="utf-8")
    for stem in ("main_table_raw", "main_table_cal", "main_table_cal_ecc", "ablation_table"):
        for extension in ("csv", "md", "tex"):
            (result / (stem + "." + extension)).write_text("synthetic only", encoding="utf-8")
    identity = {"bundle_sha256": window["bundle_sha256"], "scenario_sha256": scenario_hashes,
                "metrics_daily_sha256": analysis.sha256(result / "metrics_daily.csv"),
                "code_sha256": "synthetic-evaluator", "model_keys": model_keys,
                "seeds": [0, 1, 2], "n_scenarios": 100, "splits": ["TEST"],
                "pair_variants": list(analysis.VARIANTS),
                "pair_comparators": list(analysis.ROLLING_MODELS[1:]),
                "pair_metrics": list(analysis.KEY_METRICS)}
    (result / "evaluation_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    return window, protocol, tasks, formal, paired_path


def test_rolling_preflight_accepts_complete_synthetic_window_and_rejects_tampering(tmp_path, monkeypatch):
    window, protocol, tasks, formal, paired_path = _synthetic_formal_window(tmp_path, monkeypatch)
    assert len(analysis._preflight_window(window, protocol, tasks)) == 126
    paired = pd.read_csv(paired_path)
    paired.iloc[:-1].to_csv(paired_path, index=False)
    with pytest.raises(ValueError, match="paired comparisons"):
        analysis._preflight_window(window, protocol, tasks)
    paired.to_csv(paired_path, index=False)
    scenario = next(formal.glob("*/seed_0/*_100_TEST.pickle"))
    scenario.write_bytes(b"altered synthetic response")
    with pytest.raises(ValueError, match="scenario file"):
        analysis._preflight_window(window, protocol, tasks)
