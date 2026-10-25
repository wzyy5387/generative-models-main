# -*- coding: utf-8 -*-
"""NESO v2 supplementary analysis and rolling result aggregation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.forecast_quality.compare_scenarios import load_scenarios
from GEFcom2014.forecast_quality.neso_wind_results import (
    DAILY_METRICS, MAIN_MODELS, MODEL_KEYS, SEEDS, VARIANTS, _aggregate_daily,
    _pair_bootstrap, _scenario_file, _table_rows, _write_csv, _write_table,
)
from GEFcom2014.forecast_quality.unified_postprocessing import HourlyAffineCalibrator
from GEFcom2014.models.QBM_VAE.run_neso_wind_jobs import CONFIGS, SOURCES


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "export/fa_bm_vae_neso_wind_classical_v1"
V2 = ROOT / "export/fa_bm_vae_neso_wind_classical_v2"
KEY_METRICS = ("CRPS_MW", "Energy_Score_MW", "Variogram_Score", "Ramp_CRPS_MW")
ROLLING_MODELS = MAIN_MODELS + ("J=0 / independent prior",)
NOMINALS = (0.5, 0.8, 0.9, 0.95)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validated_daily(path, dates, models):
    df = pd.read_csv(path)
    expected = {(m, s, v, d) for m in models for s in SEEDS for v in VARIANTS for d in dates}
    actual = set(zip(df.model, df.seed, df.variant, df.date))
    if actual != expected or len(df) != len(expected) or not np.isfinite(df[list(DAILY_METRICS)].to_numpy()).all():
        raise ValueError("Daily metrics do not contain the exact finite model/seed/variant/date grid")
    return df


def _preflight_window(window, protocol, planned_tasks):
    """Reject incomplete or changed formal inputs before annual files are created."""
    name = window["name"]
    home = V2 / "rolling" / name
    formal = home / "formal"
    result = formal / "results"
    bundle = home / "data/neso_wind_daily.npz"
    if (ROOT / window["bundle"]).resolve() != bundle.resolve() or sha256(bundle) != window["bundle_sha256"]:
        raise ValueError(name + ": frozen bundle identity or hash differs")
    _, _, dates = load_daily_bundle(bundle)
    if any(list(dates[split]) != window["dates"][split] for split in ("ls", "vs", "test")):
        raise ValueError(name + ": frozen split dates differ")
    for model, expected_hash in window["configs"].items():
        config = home / "configs" / CONFIGS[model].name
        if sha256(config) != expected_hash:
            raise ValueError(name + ": frozen config hash differs for " + model)

    manifest = json.loads((formal / "task_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or manifest.get("bundle_sha256") != window["bundle_sha256"]:
        raise ValueError(name + ": formal task manifest is incomplete or uses another bundle")
    if Path(manifest.get("bundle", "")).resolve() != bundle.resolve():
        raise ValueError(name + ": formal task manifest bundle path differs")
    if manifest.get("config_hashes") != window["configs"]:
        raise ValueError(name + ": formal task manifest config hashes differ")
    for source in SOURCES:
        if manifest.get("source_hashes", {}).get(source) != protocol["code_hashes"].get(source):
            raise ValueError(name + ": formal source hash differs for " + source)
    jobs, completed = manifest.get("jobs", []), manifest.get("results", [])
    expected = {(model, seed) for model in protocol["models"] for seed in SEEDS}
    job_keys = [(job.get("model"), job.get("training_seed")) for job in jobs]
    result_keys = [(row.get("model"), row.get("training_seed")) for row in completed]
    if (len(jobs) != 21 or len(completed) != 21 or set(job_keys) != expected or
            set(result_keys) != expected or len(set(job_keys)) != 21 or len(set(result_keys)) != 21):
        raise ValueError(name + ": expected 21 distinct completed formal tasks")
    plans = {(job["model"], job["training_seed"]): job for job in planned_tasks}
    for job in jobs:
        key = (job["model"], job["training_seed"])
        if job != plans.get(key) or job.get("generation_seed") != job["training_seed"] or job.get("n_scenarios") != 100 or job.get("scenario_splits") != ["VS", "TEST"]:
            raise ValueError(name + ": formal command or seed differs from frozen task list")
        if Path(job["output_dir"]).resolve() != (formal / job["model"] / ("seed_%d" % job["training_seed"])).resolve():
            raise ValueError(name + ": formal output path differs")
    for row in completed:
        key = (row["model"], row["training_seed"])
        if (row.get("status") != "COMPLETED" or row.get("return_code") != 0 or
                row.get("command") != plans[key]["command"] or
                Path(row.get("output_dir", "")).resolve() != Path(plans[key]["output_dir"]).resolve() or
                not Path(row.get("output_dir", "")).is_dir()):
            raise ValueError(name + ": formal output is absent or task did not complete")

    identity = json.loads((result / "evaluation_identity.json").read_text(encoding="utf-8"))
    if (identity.get("bundle_sha256") != window["bundle_sha256"] or
            identity.get("metrics_daily_sha256") != sha256(result / "metrics_daily.csv") or
            identity.get("code_sha256") != protocol["code_hashes"]["GEFcom2014/forecast_quality/neso_wind_results.py"] or
            identity.get("model_keys") != protocol["models"] or identity.get("seeds") != list(SEEDS) or
            identity.get("n_scenarios") != 100 or identity.get("splits") != ["TEST"] or
            identity.get("pair_variants") != list(VARIANTS) or
            identity.get("pair_comparators") != list(ROLLING_MODELS[1:]) or
            identity.get("pair_metrics") != list(KEY_METRICS)):
        raise ValueError(name + ": formal evaluation identity differs")
    scenario_hashes = {}
    for model in protocol["models"]:
        for seed in SEEDS:
            for split in ("VS", "TEST"):
                path = _scenario_file(formal, model, seed, split, 100)
                scenario_hashes[str(path.relative_to(formal))] = sha256(path)
    if identity.get("scenario_sha256") != scenario_hashes:
        raise ValueError(name + ": evaluated scenario file is absent or changed")
    daily = _validated_daily(result / "metrics_daily.csv", window["dates"]["test"], ROLLING_MODELS)
    paired = pd.read_csv(result / "paired_comparisons.csv")
    paired_expected = {(model, variant, metric) for model in ROLLING_MODELS[1:]
                       for variant in VARIANTS for metric in KEY_METRICS}
    paired_actual = set(zip(paired.comparator, paired.variant, paired.metric))
    if (len(paired) != len(paired_expected) or paired_actual != paired_expected or
            not (paired.model == "FA-BM-VAE").all() or
            not np.isfinite(paired[["mean_difference_main_minus_comparator",
                                    "block_bootstrap_ci95_low", "block_bootstrap_ci95_high"]].to_numpy()).all()):
        raise ValueError(name + ": paired comparisons lack the exact finite comparator/variant/metric grid")
    for filename in ("metrics_per_seed.csv", *(stem + "." + extension
                     for stem in ("main_table_raw", "main_table_cal", "main_table_cal_ecc", "ablation_table")
                     for extension in ("csv", "md", "tex"))):
        if not (result / filename).is_file():
            raise ValueError(name + ": formal evaluation output is missing: " + filename)
    return daily


def _preflight_rolling(protocol):
    if (protocol.get("protocol_version") != "neso-rolling-four-season-v2" or
            protocol.get("models") != [MODEL_KEYS[model] for model in ROLLING_MODELS] or
            protocol.get("seeds") != list(SEEDS) or protocol.get("n_scenarios_per_date") != 100 or
            [window["name"] for window in protocol.get("windows", [])] !=
            ["quarter_1", "quarter_2", "quarter_3", "quarter_4"]):
        raise ValueError("Frozen rolling protocol is incomplete or inconsistent")
    planned = json.loads((V2 / "task_manifest.json").read_text(encoding="utf-8"))
    if planned.get("protocol_sha256") != sha256(V2 / "protocol_v2.json") or len(planned.get("tasks", [])) != 84:
        raise ValueError("Frozen protocol and 84-task manifest differ")
    for source, expected_hash in protocol["code_hashes"].items():
        if sha256(ROOT / source) != expected_hash:
            raise ValueError("Frozen source hash differs for " + source)
    all_dates, daily = set(), []
    for window in protocol["windows"]:
        dates = window["dates"]["test"]
        if len(set(dates)) != len(dates) or all_dates.intersection(dates):
            raise ValueError("Rolling TEST dates duplicate or overlap")
        all_dates.update(dates)
        tasks = [job for job in planned["tasks"] if Path(job["output_dir"]).parent.parent ==
                 V2 / "rolling" / window["name"] / "formal"]
        if len(tasks) != 21:
            raise ValueError(window["name"] + ": frozen task list is incomplete")
        daily.append(_preflight_window(window, protocol, tasks))
    return daily


def paired_from_daily(df, dates, comparators, output_path, metrics=DAILY_METRICS, repetitions=5000):
    dates = list(dates)
    means = df.groupby(["model", "variant", "date"], sort=False)[list(metrics)].mean()
    rows = []
    for comparator in comparators:
        for variant in VARIANTS:
            for metric in metrics:
                left = np.asarray([means.loc[("FA-BM-VAE", variant, d), metric] for d in dates])
                right = np.asarray([means.loc[(comparator, variant, d), metric] for d in dates])
                estimate, ci = _pair_bootstrap(left, right, dates, repetitions=repetitions,
                                               seed=20261022 + len(rows))
                rows.append({"model": "FA-BM-VAE", "comparator": comparator, "variant": variant,
                             "metric": metric, "mean_difference_main_minus_comparator": estimate,
                             "block_bootstrap_ci95_low": ci[0], "block_bootstrap_ci95_high": ci[1],
                             "block_length_calendar_days": 7, "repetitions": repetitions,
                             "p_value": "NOT REPORTED", "statistical_unit": "forecast date; daily seed means"})
    _write_csv(output_path, rows)
    return pd.DataFrame(rows)


def coverage_from_scenarios(run_root, bundle, models, out_csv, window):
    arrays, _, dates = load_daily_bundle(bundle)
    y = np.asarray(arrays["y_test"], dtype=float)
    vs_target = np.asarray(arrays["y_vs"], dtype=float).reshape(-1)
    rows = []
    for model in models:
        key = MODEL_KEYS[model]
        for seed in SEEDS:
            raw = load_scenarios(_scenario_file(run_root, key, seed, "TEST", 100), len(y) * 24)
            vs = load_scenarios(_scenario_file(run_root, key, seed, "VS", 100), len(vs_target))
            cal = HourlyAffineCalibrator(limits=(0.0, float("inf"))).fit(vs, vs_target).transform(raw)
            for variant, samples in (("Raw", raw), ("Cal", cal), ("Cal+ECC", cal)):
                samples = samples.reshape(len(y), 24, 100)
                for nominal in NOMINALS:
                    alpha = (1.0 - nominal) / 2.0
                    low, high = np.quantile(samples, [alpha, 1.0 - alpha], axis=2)
                    rows.append({"window": window, "model": model, "seed": seed,
                                 "variant": variant, "nominal_coverage": nominal,
                                 "actual_coverage": float(np.mean((y >= low) & (y <= high))),
                                 "mean_interval_width_mw": float(np.mean(high - low)),
                                 "n_dates": len(dates["test"]),
                                 "scenario_negative_fraction": float(np.mean(samples < 0)),
                                 "ecc_note": "Cal+ECC has Cal marginals; dependence comes from same-seed FA-BM-VAE template"})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    return frame


def reliability_plots(frame, output_dir, prefix=""):
    output_dir = Path(output_dir)
    for variant in VARIANTS:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
        subset = frame[frame.variant == variant]
        for model in subset.model.unique():
            g = subset[subset.model == model].groupby("nominal_coverage").apply(
                lambda part: np.average(part.actual_coverage, weights=part.n_dates), include_groups=False)
            ax.plot(g.index, g.values, marker="o", label=model)
        ax.set(xlabel="Nominal coverage", ylabel="Actual coverage", xlim=(0.45, 1), ylim=(0.35, 1),
               title=(prefix + " " + variant).strip())
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / ("reliability_" + variant.lower().replace("+", "_") + ".png"), dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for model in subset.model.unique():
            g = subset[subset.model == model].groupby("nominal_coverage").apply(
                lambda part: np.average(part.mean_interval_width_mw, weights=part.n_dates), include_groups=False)
            ax.plot(g.index, g.values, marker="o", label=model)
        ax.set(xlabel="Nominal coverage", ylabel="Mean interval width (MW)",
               title=(prefix + " " + variant).strip())
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / ("interval_width_" + variant.lower().replace("+", "_") + ".png"), dpi=180)
        plt.close(fig)


def paired_plot(paired, output_path, variant="Cal"):
    selected = paired[(paired.variant == variant) & paired.metric.isin(KEY_METRICS)].copy()
    metrics = [m for m in KEY_METRICS if m in set(selected.metric)]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.6 * len(metrics)))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        part = selected[selected.metric == metric].reset_index(drop=True)
        center = part.mean_difference_main_minus_comparator.to_numpy()
        low = part.block_bootstrap_ci95_low.to_numpy()
        high = part.block_bootstrap_ci95_high.to_numpy()
        ax.errorbar(center, np.arange(len(part)), xerr=[center - low, high - center], fmt="o", capsize=2)
        ax.set_yticks(np.arange(len(part)), part.comparator, fontsize=7)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("FA-BM-VAE minus comparator (95% date-block CI)")
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def supplement_v1():
    output = V2 / "v1_supplement"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    bundle = V1 / "data/neso_wind_daily.npz"
    run_root = V1 / "formal_v1"
    _, _, dates = load_daily_bundle(bundle)
    test_dates = list(dates["test"])
    daily_path = run_root / "results/metrics_daily.csv"
    df = _validated_daily(daily_path, test_dates, tuple(MODEL_KEYS))
    paired = paired_from_daily(df, test_dates, tuple(MODEL_KEYS)[1:], output / "paired_comparisons.csv")
    coverage = coverage_from_scenarios(run_root, bundle, tuple(MODEL_KEYS),
                                       output / "coverage_by_nominal.csv", "old_90_day_TEST")
    reliability_plots(coverage[coverage.model.isin(MAIN_MODELS)], output, "Old 90 dates")
    daily_bias = df[["model", "seed", "variant", "date", "mean_bias_MW"]].copy()
    daily_bias.to_csv(output / "daily_bias.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    for model in ("FA-BM-VAE", "Forecast-anchored Gaussian"):
        part = daily_bias[(daily_bias.model == model) & (daily_bias.variant == "Cal")]
        g = part.groupby("date").mean_bias_MW.mean()
        ax.plot(pd.to_datetime(g.index), g.values, label=model)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Forecast date", ylabel="Daily mean bias (MW)", title="Old 90 dates, calibrated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "daily_bias_cal.png", dpi=180)
    plt.close(fig)
    paired_plot(paired, output / "paired_differences_cal.png")
    (output / "analysis_identity.json").write_text(json.dumps({"old_daily_sha256": sha256(daily_path),
          "old_bundle_sha256": sha256(bundle), "n_dates": len(test_dates), "paired_rows": len(paired),
          "coverage_rows": len(coverage), "raw_is_v1_primary": True,
          "cal_is_post_v1_followup": True, "bootstrap_seed_averaged_dates_only": True,
          "no_anchor_label": "No anchor addition retains forecast anchor in context input"}, indent=2), encoding="utf-8")
    return output


def aggregate_rolling():
    protocol = json.loads((V2 / "protocol_v2.json").read_text(encoding="utf-8"))
    output = V2 / "rolling_results"
    if output.exists():
        raise FileExistsError(output)
    verified_daily = _preflight_rolling(protocol)
    output.mkdir()
    all_daily, all_per_seed, all_pairs, all_coverage = [], [], [], []
    names = ROLLING_MODELS
    for window, df in zip(protocol["windows"], verified_daily):
        name = window["name"]
        home = V2 / "rolling" / name
        result = home / "formal/results"
        if not result.is_dir():
            raise FileNotFoundError(result)
        df = df.copy()
        df.insert(0, "window", name)
        all_daily.append(df)
        per_seed = pd.read_csv(result / "metrics_per_seed.csv")
        per_seed.insert(0, "window", name)
        all_per_seed.append(per_seed)
        paired = pd.read_csv(result / "paired_comparisons.csv")
        paired.insert(0, "window", name)
        all_pairs.append(paired)
        table_dir = output / name
        table_dir.mkdir()
        for table in ("main_table_raw", "main_table_cal", "main_table_cal_ecc", "ablation_table"):
            for extension in ("csv", "md", "tex"):
                shutil.copyfile(result / (table + "." + extension), table_dir / (table + "." + extension))
        cov = coverage_from_scenarios(home / "formal", home / "data/neso_wind_daily.npz", names,
                                      table_dir / "coverage_by_nominal.csv", name)
        all_coverage.append(cov)
    daily = pd.concat(all_daily, ignore_index=True)
    if daily.duplicated(["model", "seed", "variant", "date"]).any():
        raise ValueError("Rolling TEST dates overlap")
    daily.to_csv(output / "metrics_daily.csv", index=False)
    pd.concat(all_per_seed, ignore_index=True).to_csv(output / "metrics_per_seed.csv", index=False)
    summary_rows = daily.drop(columns=["ramp_quantile_mae_pooled_legacy_mw"], errors="ignore").to_dict("records")
    per_seed_year, summary = _aggregate_daily(summary_rows, names)
    _write_csv(output / "metrics_per_seed_full_year.csv", per_seed_year)
    _write_csv(output / "summary_full_year.csv", summary)
    for variant, stem in (("Raw", "main_table_raw"), ("Cal", "main_table_cal"),
                          ("Cal+ECC", "main_table_cal_ecc")):
        _write_table(output, stem, _table_rows(summary, MAIN_MODELS, variant), DAILY_METRICS)
    _write_table(output, "ablation_table", _table_rows(summary, ("FA-BM-VAE", "J=0 / independent prior"), "Cal"), DAILY_METRICS)
    # Re-sample within each retraining window and each contiguous date segment, then combine by actual day count.
    pair_rows = pd.concat(all_pairs, ignore_index=True)
    full_pairs = []
    for comparator in names[1:]:
        for variant in VARIANTS:
            for metric in KEY_METRICS:
                pieces, weights = [], []
                for i, window in enumerate(protocol["windows"]):
                    part = all_daily[i]
                    means = part.groupby(["model", "variant", "date"])[metric].mean()
                    dates = window["dates"]["test"]
                    left = np.asarray([means.loc[("FA-BM-VAE", variant, d)] for d in dates])
                    right = np.asarray([means.loc[(comparator, variant, d)] for d in dates])
                    _, _, replicates = _pair_bootstrap(left, right, dates, seed=20261022 + len(full_pairs) * 4 + i,
                                                        return_replicates=True)
                    pieces.append(replicates)
                    weights.append(len(dates))
                boot = np.average(np.stack(pieces), axis=0, weights=weights)
                point = float(np.average([pair_rows.loc[(pair_rows.window == window["name"]) &
                               (pair_rows.comparator == comparator) & (pair_rows.variant == variant) &
                               (pair_rows.metric == metric), "mean_difference_main_minus_comparator"].iloc[0]
                               for window in protocol["windows"]], weights=weights))
                ci = np.quantile(boot, [0.025, 0.975])
                full_pairs.append({"window": "full_year", "model": "FA-BM-VAE", "comparator": comparator,
                                   "variant": variant, "metric": metric,
                                   "mean_difference_main_minus_comparator": point,
                                   "block_bootstrap_ci95_low": float(ci[0]),
                                   "block_bootstrap_ci95_high": float(ci[1]),
                                   "block_length_calendar_days": 7, "repetitions": 5000,
                                   "p_value": "NOT REPORTED", "n_unique_dates": sum(weights),
                                   "statistical_unit": "forecast date, stratified by retraining window"})
    combined_pairs = pd.concat([pair_rows, pd.DataFrame(full_pairs)], ignore_index=True)
    combined_pairs.to_csv(output / "paired_comparisons.csv", index=False)
    coverage = pd.concat(all_coverage, ignore_index=True)
    coverage.to_csv(output / "coverage_by_nominal.csv", index=False)
    reliability_plots(coverage[coverage.model.isin(MAIN_MODELS)], output, "Rolling four seasons")
    paired_plot(pd.DataFrame(full_pairs), output / "paired_differences_cal.png")
    seasonal = daily.groupby(["window", "model", "variant"])[["CRPS_MW", "Energy_Score_MW"]].mean().reset_index()
    seasonal.to_csv(output / "seasonal_crps_energy.csv", index=False)
    for metric in ("CRPS_MW", "Energy_Score_MW"):
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        part = seasonal[(seasonal.variant == "Cal") & seasonal.model.isin(MAIN_MODELS)]
        for model in MAIN_MODELS:
            g = part[part.model == model].set_index("window").reindex([w["name"] for w in protocol["windows"]])
            ax.plot(g.index, g[metric], marker="o", label=model)
        ax.set(xlabel="Quarter", ylabel=metric, title="Cal, actual eligible days per quarter")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / ("seasonal_" + metric.lower() + ".png"), dpi=180)
        plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supplement-v1", "aggregate-rolling"))
    args = parser.parse_args()
    print(supplement_v1() if args.mode == "supplement-v1" else aggregate_rolling())
