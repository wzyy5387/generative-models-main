# -*- coding: utf-8 -*-
"""Strict common-date evaluation and table export for NESO Wind scenarios."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.forecast_quality.compare_scenarios import (
    evaluate_model_daily,
    load_scenarios,
    ramp_distribution_error,
)
from GEFcom2014.forecast_quality.qbm_temporal_ecc import ensemble_copula_coupling
from GEFcom2014.forecast_quality.unified_postprocessing import HourlyAffineCalibrator


ROOT = Path(__file__).resolve().parents[2]
MODEL_KEYS = {
    "FA-BM-VAE": "fa_bm_vae",
    "Forecast-anchored Gaussian": "anchor_gaussian",
    "Forecast-anchored GMM (K=4)": "anchor_gmm4",
    "Forecast-anchored Score-SDE": "score_sde",
    "Forecast-anchored Spline Flow": "spline_flow",
    "Daily residual bootstrap": "residual_bootstrap",
    "J=0 / independent prior": "j0",
    "No anchor addition": "no_anchor_addition",
}
MAIN_MODELS = tuple(list(MODEL_KEYS)[:6])
ABLATIONS = tuple(list(MODEL_KEYS)[6:])
SEEDS = (0, 1, 2)
VARIANTS = ("Raw", "Cal", "Cal+ECC")
DAILY_METRICS = (
    "CRPS_MW", "Energy_Score_MW", "Variogram_Score", "Ramp_CRPS_MW", "PICP90",
    "interval_width90_MW", "MAQCE", "mean_bias_MW", "Reliability_MAE_pp",
    "Ramp_quantile_MAE_daily_MW",
)
LOWER_IS_BETTER = set(DAILY_METRICS) - {"PICP90", "mean_bias_MW"}


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scenario_file(root, key, seed, split, n_scenarios):
    directory = Path(root) / key / ("seed_%d" % seed)
    paths = list(directory.glob("*_{}_{:s}.pickle".format(n_scenarios, split)))
    if len(paths) != 1:
        raise FileNotFoundError(
            "%s seed %d needs exactly one %d-scenario %s file; found %d" %
            (key, seed, n_scenarios, split, len(paths))
        )
    return paths[0]


def _daily_reliability_and_ramp_mae(scenarios, targets):
    n_days = len(targets)
    values = np.asarray(scenarios, dtype=np.float64).reshape(n_days, 24, -1)
    quantile_levels = np.arange(1, 100, dtype=np.float64) / 100.0
    reliability, ramp_mae = [], []
    for day in range(n_days):
        q = np.quantile(values[day], quantile_levels, axis=1).T
        observed = (targets[day, :, None] < q).mean(axis=0) * 100.0
        reliability.append(np.mean(np.abs(observed - quantile_levels * 100.0)))
        sample_ramps = np.diff(values[day], axis=0)
        target_ramps = np.diff(targets[day])
        ramp_quantiles = np.quantile(sample_ramps, quantile_levels, axis=1).T
        ramp_mae.append(np.mean(np.abs(ramp_quantiles - target_ramps[:, None])))
    return np.asarray(reliability), np.asarray(ramp_mae)


def _metrics_for(scenarios, targets):
    daily = evaluate_model_daily(scenarios, targets, max_s=100, periods_per_day=24)
    reliability, ramp_mae = _daily_reliability_and_ramp_mae(scenarios, targets)
    by_day_samples = np.asarray(scenarios).reshape(len(targets), 24, scenarios.shape[1])
    daily["mean_bias_MW"] = (by_day_samples.mean(axis=2) - targets).mean(axis=1)
    daily["Reliability_MAE_pp"] = reliability
    daily["Ramp_quantile_MAE_daily_MW"] = ramp_mae
    return {
        "CRPS_MW": daily["CRPS_raw"],
        "Energy_Score_MW": daily["Energy"],
        "Variogram_Score": daily["Variogram"],
        "Ramp_CRPS_MW": daily["ramp_CRPS"],
        "PICP90": daily["PICP90"],
        "interval_width90_MW": daily["interval_width90"],
        "MAQCE": daily["MAQCE"],
        "mean_bias_MW": daily["mean_bias_MW"],
        "Reliability_MAE_pp": daily["Reliability_MAE_pp"],
        "Ramp_quantile_MAE_daily_MW": daily["Ramp_quantile_MAE_daily_MW"],
    }


def _pair_bootstrap(left, right, dates, repetitions=5000, seed=20261022, block_days=7,
                    return_replicates=False):
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    dates = np.asarray(dates, dtype="datetime64[D]")
    if difference.shape != dates.shape or not np.isfinite(difference).all():
        raise ValueError("Paired bootstrap requires complete finite date pairs")
    if len(dates) < 2 or np.any(np.diff(dates).astype(int) <= 0):
        raise ValueError("Paired dates must be unique and increasing")
    segments, start = [], 0
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).astype(int) != 1:
            segments.append(np.arange(start, i))
            start = i
    segments.append(np.arange(start, len(dates)))
    blocks = []
    for segment in segments:
        if len(segment) < block_days:
            blocks.append(segment)
        else:
            blocks.extend(segment[i:i + block_days] for i in range(len(segment) - block_days + 1))
    if not blocks:
        raise ValueError("No calendar-contiguous blocks available")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for r in range(repetitions):
        indices = []
        while len(indices) < len(difference):
            indices.extend(blocks[int(rng.integers(len(blocks)))].tolist())
        estimates[r] = difference[np.asarray(indices[:len(difference)], dtype=int)].mean()
    result = float(difference.mean()), np.quantile(estimates, [0.025, 0.975]).tolist()
    return (*result, estimates) if return_replicates else result


def _aggregate_daily(daily_rows, models):
    per_seed, summary = [], []
    pooled_ramp = {}
    for row in daily_rows:
        if "ramp_quantile_mae_pooled_legacy_mw" in row:
            pooled_ramp[(row["model"], row["seed"], row["variant"])] = row["ramp_quantile_mae_pooled_legacy_mw"]
    for model in models:
        for seed in SEEDS:
            for variant in VARIANTS:
                selected = [r for r in daily_rows if r["model"] == model and r["seed"] == seed and r["variant"] == variant]
                if not selected:
                    raise ValueError("Missing model/seed/variant cell: %s %d %s" % (model, seed, variant))
                for metric in DAILY_METRICS:
                    values = np.asarray([r[metric] for r in selected], dtype=np.float64)
                    per_seed.append({"model": model, "seed": seed, "variant": variant, "metric": metric,
                                     "mean_over_dates": float(values.mean()), "sd_over_dates": float(values.std(ddof=1)),
                                     "n_dates": len(values), "n_scenarios_per_date": 100,
                                     "statistical_unit": "forecast date"})
                legacy = pooled_ramp.get((model, seed, variant))
                if legacy is not None:
                    per_seed.append({"model": model, "seed": seed, "variant": variant,
                                     "metric": "Ramp_quantile_MAE_pooled_legacy_MW", "mean_over_dates": float(legacy),
                                     "sd_over_dates": "", "n_dates": len(selected), "n_scenarios_per_date": 100,
                                     "statistical_unit": "pooled 23 within-day ramp positions across dates"})
    for model in models:
        for variant in VARIANTS:
            for metric in DAILY_METRICS + ("Ramp_quantile_MAE_pooled_legacy_MW",):
                rows = [r for r in per_seed if r["model"] == model and r["variant"] == variant and r["metric"] == metric]
                if not rows:
                    continue
                seed_values = np.asarray([float(r["mean_over_dates"]) for r in rows])
                summary.append({"model": model, "variant": variant, "metric": metric,
                                "mean_seed_means": float(seed_values.mean()),
                                "sd_between_seeds": float(seed_values.std(ddof=1)) if len(seed_values) > 1 else 0.0,
                                "n_seeds": len(seed_values), "n_dates_per_seed": int(rows[0]["n_dates"])})
    return per_seed, summary


def _table_rows(summary, models, variant):
    metric_names = list(DAILY_METRICS) + ["Ramp_quantile_MAE_pooled_legacy_MW"]
    rows = []
    for model in models:
        row = {"model": model}
        for metric in metric_names:
            found = next((r for r in summary if r["model"] == model and r["variant"] == variant and r["metric"] == metric), None)
            row[metric] = "" if found is None else "%.5f ± %.5f" % (found["mean_seed_means"], found["sd_between_seeds"])
            row[metric + "__mean"] = "" if found is None else found["mean_seed_means"]
            row[metric + "__sd"] = "" if found is None else found["sd_between_seeds"]
        rows.append(row)
    return rows


def _write_table(output_dir, stem, rows, metric_names):
    output_dir = Path(output_dir)
    columns = ["model"] + list(metric_names)
    _write_csv(output_dir / (stem + ".csv"), [{k: r.get(k, "") for k in columns} for r in rows])
    minima, best_by_metric = {}, {}
    for metric in metric_names:
        vals = [float(r[metric + "__mean"]) for r in rows if r.get(metric + "__mean") != ""]
        if vals:
            objective = (lambda x: abs(x - 0.9)) if metric == "PICP90" else (lambda x: abs(x)) if metric == "mean_bias_MW" else (lambda x: x)
            best = min(vals, key=objective)
            best_by_metric[metric] = best
    note = "Values are mean ± sample standard deviation across training seeds 0, 1, 2; date-paired bootstrap intervals are reported separately."
    md = [note, "", "| Model | " + " | ".join(metric_names) + " |", "|---|" + "|".join(["---:"] * len(metric_names)) + "|"]
    tex = ["% " + note, "\\begin{tabular}{l" + "r" * len(metric_names) + "}", "\\toprule", "Model & " + " & ".join(metric_names) + r" \\" , "\\midrule"]
    for row in rows:
        cells = []
        for metric in metric_names:
            value = row.get(metric, "")
            mean = row.get(metric + "__mean", "")
            if mean != "" and float(mean) == best_by_metric.get(metric):
                value = "**" + value + "**"
            cells.append(value)
        md.append("| " + row["model"] + " | " + " | ".join(cells) + " |")
        tex_cells = [row.get(metric, "") for metric in metric_names]
        tex.append(row["model"] + " & " + " & ".join(tex_cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (output_dir / (stem + ".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    (output_dir / (stem + ".tex")).write_text("\n".join(tex) + "\n", encoding="utf-8")


def evaluate(input_root, dataset_bundle, output_dir, n_scenarios=100, splits=("TEST",),
             repetitions=5000, seeds=SEEDS, model_keys=None, pair_variants=("Raw",),
             pair_comparators=None, pair_metrics=DAILY_METRICS):
    input_root, dataset_bundle, output_dir = Path(input_root), Path(dataset_bundle), Path(output_dir)
    model_keys = tuple(model_keys) if model_keys is not None else tuple(MODEL_KEYS.values())
    if len(set(model_keys)) != len(model_keys) or not set(model_keys) <= set(MODEL_KEYS.values()):
        raise ValueError("Unknown or duplicate model keys")
    models = tuple(name for name, key in MODEL_KEYS.items() if key in model_keys)
    if "FA-BM-VAE" not in models:
        raise ValueError("FA-BM-VAE is required as the common ECC template")
    if not set(pair_variants) <= set(VARIANTS):
        raise ValueError("Unknown paired variant")
    if not set(pair_metrics) <= set(DAILY_METRICS):
        raise ValueError("Unknown paired metric")
    pair_comparators = tuple(pair_comparators) if pair_comparators is not None else MAIN_MODELS[1:]
    if len(set(pair_comparators)) != len(pair_comparators) or not set(pair_comparators) <= (set(models) - {"FA-BM-VAE"}):
        raise ValueError("Paired comparator is absent or duplicated")
    arrays, metadata, dates = load_daily_bundle(dataset_bundle)
    rows, raw_by_cell, y_by_split, scenario_sha256 = [], {}, {}, {}
    for split in splits:
        split_key = split.lower()
        y = np.asarray(arrays["y_" + split_key], dtype=np.float64)
        split_dates = np.asarray(dates[split_key]).astype(str)
        if len(y) != len(split_dates) or len(set(split_dates)) != len(split_dates):
            raise ValueError("Date/target mapping is incomplete or duplicated for " + split)
        y_by_split[split] = (y, split_dates)
        targets = y
        for model in models:
            key = MODEL_KEYS[model]
            for seed in seeds:
                path = _scenario_file(input_root, key, seed, split, n_scenarios)
                scenario_sha256[str(path.relative_to(input_root))] = _sha256(path)
                scenario = load_scenarios(path, len(targets) * 24)
                if scenario.shape != (len(targets) * 24, n_scenarios):
                    raise ValueError("Unexpected scenario shape: " + str(path))
                if not np.isfinite(scenario).all():
                    raise ValueError("Non-finite scenarios: " + str(path))
                raw_by_cell[(model, seed, split)] = scenario
        template_by_seed = {seed: raw_by_cell[("FA-BM-VAE", seed, split)] for seed in seeds}
        for model in models:
            for seed in seeds:
                vs_path = _scenario_file(input_root, MODEL_KEYS[model], seed, "VS", n_scenarios)
                scenario_sha256[str(vs_path.relative_to(input_root))] = _sha256(vs_path)
                vs = load_scenarios(vs_path, len(arrays["y_vs"]) * 24)
                vs_targets = np.asarray(arrays["y_vs"], dtype=np.float64)
                calibrator = HourlyAffineCalibrator(limits=(0.0, float("inf"))).fit(vs, vs_targets.reshape(-1))
                raw = raw_by_cell[(model, seed, split)]
                calibrated = calibrator.transform(raw)
                coupled = ensemble_copula_coupling(calibrated, template_by_seed[seed])
                if not np.array_equal(np.sort(coupled, axis=1), np.sort(calibrated, axis=1)):
                    raise ValueError("ECC changed hourly marginal samples")
                for variant, scenario in (("Raw", raw), ("Cal", calibrated), ("Cal+ECC", coupled)):
                    metric_values = _metrics_for(scenario, targets)
                    for index, date in enumerate(split_dates):
                        daily_support = scenario[index * 24:(index + 1) * 24]
                        rows.append({"model": model, "seed": seed, "training_seed": seed,
                                     "generation_seed": seed, "date": date, "variant": variant,
                                     "n_scenarios": n_scenarios, "statistical_unit": "forecast date",
                                     "scenario_min_mw": float(np.min(daily_support)),
                                     "scenario_max_mw": float(np.max(daily_support)),
                                     "scenario_negative_fraction": float(np.mean(daily_support < 0)),
                                     **{name: float(value[index]) for name, value in metric_values.items()}})
                    rows[-1]["ramp_quantile_mae_pooled_legacy_mw"] = float(ramp_distribution_error(scenario, targets, max_s=n_scenarios))
    if "TEST" not in splits:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "smoke_metrics_daily.csv", rows)
        return output_dir

    dates_expected = y_by_split["TEST"][1].tolist()
    if tuple(seeds) != SEEDS:
        raise ValueError("Formal tables require the complete training seed set 0,1,2")
    expected_cells = {(model, seed, variant, date) for model in models for seed in SEEDS for variant in VARIANTS for date in dates_expected}
    actual_cells = {(r["model"], r["seed"], r["variant"], r["date"]) for r in rows if r["date"] in set(dates_expected)}
    if actual_cells != expected_cells:
        raise ValueError("Incomplete model/seed/date/variant grid; missing=%d extra=%d" % (len(expected_cells-actual_cells), len(actual_cells-expected_cells)))
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "metrics_daily.csv"
    _write_csv(daily_path, rows)
    per_seed, summary = _aggregate_daily([r for r in rows if r["date"] in set(dates_expected)], models)
    _write_csv(output_dir / "metrics_per_seed.csv", per_seed)
    _write_csv(output_dir / "summary.csv", summary)
    metric_names = list(DAILY_METRICS) + ["Ramp_quantile_MAE_pooled_legacy_MW"]
    for variant, stem in (("Raw", "main_table_raw"), ("Cal", "main_table_cal"), ("Cal+ECC", "main_table_cal_ecc")):
        _write_table(output_dir, stem, _table_rows(summary, tuple(m for m in MAIN_MODELS if m in models), variant), metric_names)
    selected_ablations = tuple(m for m in ABLATIONS if m in models)
    if selected_ablations:
        _write_table(output_dir, "ablation_table", _table_rows(summary, selected_ablations, "Raw"), metric_names)
    paired = []
    test_dates = y_by_split["TEST"][1]
    by_cell = {(r["model"], r["seed"], r["variant"], r["date"]): r for r in rows if r["date"] in set(dates_expected)}
    for comparator in pair_comparators:
        for variant in pair_variants:
            for metric in pair_metrics:
                left = np.stack([[by_cell[("FA-BM-VAE", seed, variant, date)][metric] for date in dates_expected]
                                 for seed in SEEDS]).mean(axis=0)
                right = np.stack([[by_cell[(comparator, seed, variant, date)][metric] for date in dates_expected]
                                  for seed in SEEDS]).mean(axis=0)
                estimate, ci = _pair_bootstrap(left, right, test_dates, repetitions=repetitions, seed=20261022 + len(paired))
                paired.append({"model": "FA-BM-VAE", "comparator": comparator, "variant": variant, "metric": metric,
                               "mean_difference_main_minus_comparator": estimate, "block_bootstrap_ci95_low": ci[0],
                               "block_bootstrap_ci95_high": ci[1], "block_length_calendar_days": 7,
                               "repetitions": repetitions, "p_value": "NOT REPORTED",
                               "statistical_unit": "forecast date; daily scores averaged over seeds before inference"})
    _write_csv(output_dir / "paired_comparisons.csv", paired)
    (output_dir / "evaluation_identity.json").write_text(json.dumps({
        "bundle_sha256": _sha256(dataset_bundle),
        "scenario_sha256": scenario_sha256,
        "metrics_daily_sha256": _sha256(daily_path),
        "code_sha256": _sha256(__file__),
        "model_keys": list(model_keys), "seeds": list(seeds),
        "n_scenarios": n_scenarios, "splits": list(splits),
        "pair_variants": list(pair_variants),
        "pair_comparators": list(pair_comparators),
        "pair_metrics": list(pair_metrics),
    }, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--splits", nargs="+", choices=["VS", "TEST"], default=["TEST"])
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_KEYS.values()))
    parser.add_argument("--pair-variants", nargs="+", choices=VARIANTS, default=["Raw"])
    parser.add_argument("--pair-comparators", nargs="+", choices=tuple(MODEL_KEYS))
    parser.add_argument("--pair-metrics", nargs="+", choices=DAILY_METRICS)
    args = parser.parse_args()
    print(evaluate(args.input_root, args.dataset_bundle, args.output_dir,
                   args.n_scenarios, tuple(args.splits), args.bootstrap_repetitions, tuple(args.seeds),
                   args.models, tuple(args.pair_variants), args.pair_comparators,
                   tuple(args.pair_metrics) if args.pair_metrics else DAILY_METRICS))


if __name__ == "__main__":
    main()
