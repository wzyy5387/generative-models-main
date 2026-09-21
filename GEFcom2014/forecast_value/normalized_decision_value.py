# -*- coding: utf-8 -*-
"""Model-agnostic normalized decision-value evaluation.

This is used when auditable market prices are unavailable.  It is deliberately
separate from the Gurobi day-ahead planner and must be reported as a
normalized asymmetric cost-loss sensitivity analysis.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.paper_artifact_guard import assert_paper_eligible


def validate_inputs(scenarios, actual, point_forecast, dates=None):
    scenarios = np.asarray(scenarios, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    point_forecast = np.asarray(point_forecast, dtype=np.float64)
    if scenarios.ndim != 3:
        raise ValueError("scenarios must have shape (dates, scenarios, horizon)")
    if actual.shape != (scenarios.shape[0], scenarios.shape[2]):
        raise ValueError("actual must have shape (dates, horizon)")
    if point_forecast.shape != actual.shape:
        raise ValueError("point_forecast must match actual shape")
    if dates is not None and len(dates) != actual.shape[0]:
        raise ValueError("dates must align with the first dimension")
    if not np.isfinite(np.concatenate([scenarios.reshape(-1), actual.reshape(-1),
                                       point_forecast.reshape(-1)])).all():
        raise ValueError("decision inputs must be finite")
    return scenarios, actual, point_forecast


def _cost(decision, actual, under_cost, over_cost):
    under = np.maximum(actual - decision, 0.0)
    over = np.maximum(decision - actual, 0.0)
    return under_cost * under + over_cost * over, under, over


def evaluate_normalized_decision_value(scenarios, actual, point_forecast,
                                       under_cost=2.0, over_cost=1.0):
    scenarios, actual, point_forecast = validate_inputs(scenarios, actual, point_forecast)
    if under_cost <= 0 or over_cost <= 0:
        raise ValueError("under_cost and over_cost must be positive")
    quantile = under_cost / (under_cost + over_cost)
    scenario_decision = np.quantile(scenarios, quantile, axis=1)
    scenario_cost, scenario_under, scenario_over = _cost(scenario_decision, actual,
                                                           under_cost, over_cost)
    point_cost, point_under, point_over = _cost(point_forecast, actual, under_cost, over_cost)
    oracle_cost = np.zeros_like(point_cost)
    return {
        "scenario_decision": scenario_decision,
        "realized_cost": scenario_cost.sum(axis=1),
        "oracle_regret": (scenario_cost - oracle_cost).sum(axis=1),
        "point_cost": point_cost.sum(axis=1),
        "cost_improvement_vs_point": point_cost.sum(axis=1) - scenario_cost.sum(axis=1),
        "underprediction_cost": (under_cost * scenario_under).sum(axis=1),
        "overprediction_cost": (over_cost * scenario_over).sum(axis=1),
        "point_underprediction_cost": (under_cost * point_under).sum(axis=1),
        "point_overprediction_cost": (over_cost * point_over).sum(axis=1),
        "quantile": float(quantile),
        "decision_value_type": "normalized_asymmetric_cost_loss",
        "real_market_prices_used": False,
    }


def block_bootstrap_mean(values, repetitions=2000, seed=2026):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, values.size), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci_2.5": float(np.quantile(samples, 0.025)),
            "ci_97.5": float(np.quantile(samples, 0.975)), "n_date_blocks": int(values.size)}


def run_synthetic_smoke(output_dir):
    rng = np.random.default_rng(11)
    dates, scenarios, horizon = 8, 100, 4
    actual = rng.uniform(0.2, 0.8, size=(dates, horizon))
    point = np.full_like(actual, 0.25)
    samples = actual[:, None, :] + rng.normal(0, 0.08, size=(dates, scenarios, horizon))
    samples = np.clip(samples, 0.0, 1.0)
    result = evaluate_normalized_decision_value(samples, actual, point)
    summary_keys = (
        "realized_cost", "oracle_regret", "point_cost", "cost_improvement_vs_point",
        "underprediction_cost", "overprediction_cost",
        "point_underprediction_cost", "point_overprediction_cost",
    )
    summary = {key: block_bootstrap_mean(result[key], repetitions=300)
               for key in summary_keys}
    payload = {"smoke_only": True, "synthetic_data": True, "eligible_for_paper": False,
               "hardware_claim": False, "physical_platform_used": False,
               "real_market_prices_used": False, "metrics": summary}
    path = Path(output_dir) / "normalized_decision_value_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, payload


def run_real_npz(input_path, output_dir, model_name, dataset_hash, seed=None):
    """Evaluate real precomputed scenarios; prices remain deliberately absent."""
    input_path = Path(input_path)
    with np.load(input_path, allow_pickle=False) as archive:
        required = {"scenarios", "actual", "point_forecast", "dates"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError("Decision NPZ missing %s" % sorted(missing))
        scenarios, actual, point = (archive[key] for key in ("scenarios", "actual", "point_forecast"))
        dates = np.asarray(archive["dates"]).astype(str)
    validate_inputs(scenarios, actual, point, dates)
    ratios = ((0.5, 1.0), (1.0, 1.0), (2.0, 1.0), (4.0, 1.0))
    records = []
    summaries = []
    for under, over in ratios:
        result = evaluate_normalized_decision_value(scenarios, actual, point, under, over)
        for index, day in enumerate(dates):
            records.append({"date": str(day), "under_cost": under, "over_cost": over,
                            "realized_cost": float(result["realized_cost"][index]),
                            "oracle_regret": float(result["oracle_regret"][index]),
                            "cost_improvement_vs_point": float(result["cost_improvement_vs_point"][index]),
                            "underprediction_cost": float(result["underprediction_cost"][index]),
                            "overprediction_cost": float(result["overprediction_cost"][index])})
        summaries.append({"under_cost": under, "over_cost": over,
                          "realized_cost": block_bootstrap_mean(result["realized_cost"]),
                          "oracle_regret": block_bootstrap_mean(result["oracle_regret"]),
                          "cost_improvement_vs_point": block_bootstrap_mean(result["cost_improvement_vs_point"]),
                          "underprediction_cost": block_bootstrap_mean(result["underprediction_cost"]),
                          "overprediction_cost": block_bootstrap_mean(result["overprediction_cost"])})
    summary = {"method_name": model_name, "model": model_name, "dataset_hash": dataset_hash,
               "seed": seed, "date_start": str(dates[0]), "date_end": str(dates[-1]),
               "n_dates": int(len(dates)), "n_scenarios": int(np.asarray(scenarios).shape[1]),
               "smoke_only": False, "synthetic_data": False, "eligible_for_paper": True,
               "real_market_prices_used": False,
               "decision_value_type": "normalized_asymmetric_cost_loss",
               "cost_ratios": ["%g:%g" % ratio for ratio in ratios], "date_block_bootstrap": summaries,
               "paired_comparison": "against point forecast within the same dates",
               "effect_size": "standardized date-level improvement; no market prices used"}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with (output_dir / "decision_value_per_date.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (output_dir / "decision_value_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("export/decision_value"))
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--model-name", default="unknown-model")
    parser.add_argument("--dataset-hash", required=False, default="unrecorded")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.smoke and args.input_npz:
        raise SystemExit("Choose one of --smoke and --input-npz")
    if args.input_npz:
        summary = run_real_npz(args.input_npz, args.output_dir, args.model_name, args.dataset_hash, args.seed)
        print("Wrote normalized decision-value real-input summary to %s" % Path(args.output_dir).resolve())
    elif args.smoke:
        path, _ = run_synthetic_smoke(args.output_dir)
        print("Wrote normalized decision-value smoke output to %s" % path.resolve())
    else:
        raise SystemExit("Pass --smoke or --input-npz")


if __name__ == "__main__":
    main()
