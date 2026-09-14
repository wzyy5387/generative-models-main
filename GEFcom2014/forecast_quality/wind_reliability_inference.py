# -*- coding: utf-8 -*-

"""Stratified wind reliability diagnostics with paired block inference.

The original aggregate reliability number pools every wind farm, test date,
and forecast horizon. This script reports mean absolute quantile calibration
error (MAQCE) by farm, date, and horizon. Statistical inference resamples
calendar-sorted test dates in circular moving blocks and QBM training seeds.

The GEFCom split is a random set of dates rather than a contiguous holdout.
Consequently, a block represents neighboring dates within the sorted held-out
set; this limitation is recorded in every summary row.
"""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from GEFcom2014.forecast_quality.compare_scenarios import (
    BASELINE_SCENARIOS_DIR,
    DATA_DIR,
    NF_UMNN_ID,
    ROOT_DIR,
    load_scenarios,
)
from GEFcom2014 import wind_data


OUTPUT_DIR = ROOT_DIR / "export" / "paper_summary"
N_QUANTILES = 99
N_ZONES = 10
QBM_VARIANT = "gibbs_calm_trc"


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    observations, dates = load_wind_test_tensor()
    order = np.argsort(dates.values)
    observations = observations[:, order, :]
    dates = dates[order]

    qbm_hits = load_qbm_hits(observations, order)
    details = []
    summaries = []
    plot_payloads = {}
    for comparator in args.comparators:
        baseline_hits = load_baseline_hits(comparator, observations, order)
        qbm_contributions = build_contributions(qbm_hits)
        baseline_contributions = build_contributions(baseline_hits)
        details.extend(
            detail_rows(
                comparator,
                qbm_contributions,
                baseline_contributions,
                dates,
            )
        )
        summaries.extend(
            inference_rows(
                comparator=comparator,
                qbm=qbm_contributions,
                baseline=baseline_contributions,
                n_bootstrap=args.n_bootstrap,
                block_length=args.block_length,
                rng=rng,
            )
        )
        plot_payloads[comparator] = layer_unit_metrics(qbm_contributions, baseline_contributions)

    write_csv(OUTPUT_DIR / "wind_reliability_strata_details.csv", details)
    write_csv(OUTPUT_DIR / "wind_reliability_inference.csv", summaries)
    write_latex(summaries)
    plot_details(plot_payloads, dates)
    print("Wind stratified reliability results written to %s" % OUTPUT_DIR)


def load_wind_test_tensor():
    data = wind_data(DATA_DIR / "wind_data_all_zone.csv", test_size=50, random_state=0)
    target = data[5].copy()
    if target.shape[0] % N_ZONES != 0:
        raise ValueError("Wind TEST rows are not divisible by the number of zones")
    n_dates = target.shape[0] // N_ZONES
    reference_dates = target.index[:n_dates]
    for zone in range(1, N_ZONES):
        if not reference_dates.equals(target.index[zone * n_dates:(zone + 1) * n_dates]):
            raise ValueError("Wind zones do not share the same TEST dates and row order")
    observations = target.values.reshape(N_ZONES, n_dates, 24)
    return observations, reference_dates


def qbm_paths():
    directory = ROOT_DIR / "export" / "qbm_vae_wind"
    prefix = "scenarios_wind_QBMVAE_2_%s_" % QBM_VARIANT
    paths = []
    for path in sorted(directory.glob(prefix + "*_100_TEST.pickle")):
        seed = path.stem.removesuffix("_100_TEST").removeprefix(prefix)
        if seed.isdigit():
            paths.append(path)
    if not paths:
        raise FileNotFoundError("No QBM seed scenarios found for %s" % QBM_VARIANT)
    return paths


def baseline_path(comparator):
    if comparator == "VAE":
        return BASELINE_SCENARIOS_DIR / "vae" / "scenarios_wind_VAElinear_1_0_100_TEST.pickle"
    if comparator == "NF-UMNN":
        return BASELINE_SCENARIOS_DIR / "nfs" / (
            "scenarios_wind_UMNN_M_%s_0_100_TEST.pickle" % NF_UMNN_ID["wind"]
        )
    raise ValueError("Unknown comparator: %s" % comparator)


def load_qbm_hits(observations, date_order):
    expected_periods = observations.size
    hits = []
    for path in qbm_paths():
        scenarios = load_scenarios(path, expected_periods=expected_periods)
        hits.append(scenario_hits(scenarios, observations, date_order))
    return np.asarray(hits, dtype=np.float64)


def load_baseline_hits(comparator, observations, date_order):
    scenarios = load_scenarios(baseline_path(comparator), expected_periods=observations.size)
    return scenario_hits(scenarios, observations, date_order)


def scenario_hits(scenarios, sorted_observations, date_order):
    q_grid = quantile_grid()
    quantiles = np.quantile(scenarios[:, :100], q=q_grid, axis=1).T
    n_dates = sorted_observations.shape[1]
    quantiles = quantiles.reshape(N_ZONES, n_dates, 24, N_QUANTILES)
    quantiles = quantiles[:, date_order, :, :]
    return sorted_observations[..., None] < quantiles


def build_contributions(hits):
    """Pre-aggregate hits while retaining the date axis for block inference."""
    hits = np.asarray(hits, dtype=np.float64)
    if hits.ndim == 5:
        return {
            "seeded": True,
            "field_day": hits.mean(axis=3).transpose(0, 2, 1, 3),  # seed, date, field, q
            "date": hits.mean(axis=(1, 3)),                         # seed, date, q
            "horizon_day": hits.mean(axis=1),                       # seed, date, horizon, q
        }
    if hits.ndim == 4:
        return {
            "seeded": False,
            "field_day": hits.mean(axis=2).transpose(1, 0, 2),     # date, field, q
            "date": hits.mean(axis=(0, 2)),                         # date, q
            "horizon_day": hits.mean(axis=0),                       # date, horizon, q
        }
    raise ValueError("hits must be field x date x horizon x quantile, optionally with seeds")


def quantile_grid():
    return np.arange(1, N_QUANTILES + 1, dtype=np.float64) / (N_QUANTILES + 1)


def maqce(coverage):
    coverage = np.asarray(coverage, dtype=np.float64)
    return np.abs(100.0 * coverage - 100.0 * quantile_grid()).mean(axis=-1)


def layer_unit_metrics(qbm, baseline):
    qbm_seeded = seeded_unit_metrics(qbm)
    q_field = qbm_seeded["field"].mean(axis=0)
    b_field = maqce(baseline["field_day"].mean(axis=0))
    q_date = qbm_seeded["date"].mean(axis=0)
    b_date = maqce(baseline["date"])
    q_horizon = qbm_seeded["horizon"].mean(axis=0)
    b_horizon = maqce(baseline["horizon_day"].mean(axis=0))
    q_global = float(qbm_seeded["global_pooled"].mean())
    b_global = float(maqce(baseline["date"].mean(axis=0)))
    return {
        "field": (q_field, b_field),
        "date": (q_date, b_date),
        "horizon": (q_horizon, b_horizon),
        "global_pooled": (np.asarray([q_global]), np.asarray([b_global])),
    }


def detail_rows(comparator, qbm, baseline, dates):
    mean_metrics = layer_unit_metrics(qbm, baseline)
    rows = []
    labels = {
        "field": ["ZONE_%s" % index for index in range(1, N_ZONES + 1)],
        "date": [str(date.date()) for date in dates],
        "horizon": ["H%02d" % hour for hour in range(1, 25)],
        "global_pooled": ["ALL"],
    }
    seeded_metrics = seeded_unit_metrics(qbm)
    for layer, (qbm_values, baseline_values) in mean_metrics.items():
        seed_std = seeded_metrics[layer].std(axis=0, ddof=1) if seeded_metrics[layer].shape[0] > 1 else np.zeros_like(qbm_values)
        for index, label in enumerate(labels[layer]):
            difference = float(qbm_values[index] - baseline_values[index])
            rows.append({
                "track": "wind",
                "comparator": comparator,
                "qbm_variant": QBM_VARIANT,
                "layer": layer,
                "unit_index": index,
                "unit_label": label,
                "qbm_maqce": float(qbm_values[index]),
                "qbm_seed_std": float(seed_std[index]),
                "comparator_maqce": float(baseline_values[index]),
                "qbm_minus_comparator": difference,
                "improvement_percent": percent_improvement(float(baseline_values[index]), float(qbm_values[index])),
            })
    return rows


def seeded_unit_metrics(qbm):
    return {
        "field": maqce(qbm["field_day"].mean(axis=1)),
        "date": maqce(qbm["date"]),
        "horizon": maqce(qbm["horizon_day"].mean(axis=1)),
        "global_pooled": maqce(qbm["date"].mean(axis=1))[:, None],
    }


def inference_rows(comparator, qbm, baseline, n_bootstrap, block_length, rng):
    n_seeds, n_dates = qbm["date"].shape[:2]
    observed_qbm = aggregate_layer_scores(qbm, np.full(n_seeds, 1.0 / n_seeds), np.full(n_dates, 1.0 / n_dates))
    observed_baseline = aggregate_layer_scores(
        baseline, None, np.full(n_dates, 1.0 / n_dates)
    )
    bootstrap_qbm = {layer: [] for layer in observed_qbm}
    bootstrap_baseline = {layer: [] for layer in observed_qbm}
    bootstrap_difference = {layer: [] for layer in observed_qbm}

    for _ in range(n_bootstrap):
        seed_draw = rng.integers(0, n_seeds, size=n_seeds)
        seed_weights = np.bincount(seed_draw, minlength=n_seeds) / n_seeds
        day_draw = circular_block_sample(n_dates, block_length, rng)
        day_weights = np.bincount(day_draw, minlength=n_dates) / n_dates
        qbm_scores = aggregate_layer_scores(qbm, seed_weights, day_weights)
        baseline_scores = aggregate_layer_scores(baseline, None, day_weights)
        for layer in observed_qbm:
            bootstrap_qbm[layer].append(qbm_scores[layer])
            bootstrap_baseline[layer].append(baseline_scores[layer])
            bootstrap_difference[layer].append(qbm_scores[layer] - baseline_scores[layer])

    p_values, n_permutations = paired_block_permutation_p(qbm, baseline, block_length, rng)
    rows = []
    for layer in observed_qbm:
        q_values = np.asarray(bootstrap_qbm[layer])
        b_values = np.asarray(bootstrap_baseline[layer])
        differences = np.asarray(bootstrap_difference[layer])
        q_ci = np.quantile(q_values, [0.025, 0.975])
        b_ci = np.quantile(b_values, [0.025, 0.975])
        d_ci = np.quantile(differences, [0.025, 0.975])
        observed_difference = observed_qbm[layer] - observed_baseline[layer]
        rows.append({
            "track": "wind",
            "comparator": comparator,
            "qbm_variant": QBM_VARIANT,
            "layer": layer,
            "n_qbm_seeds": n_seeds,
            "n_dates": n_dates,
            "n_units": {"field": N_ZONES, "date": n_dates, "horizon": 24, "global_pooled": 1}[layer],
            "block_length_dates": block_length,
            "n_bootstrap": n_bootstrap,
            "n_permutations": n_permutations,
            "qbm_maqce": observed_qbm[layer],
            "qbm_ci_low": float(q_ci[0]),
            "qbm_ci_high": float(q_ci[1]),
            "comparator_maqce": observed_baseline[layer],
            "comparator_ci_low": float(b_ci[0]),
            "comparator_ci_high": float(b_ci[1]),
            "qbm_minus_comparator": observed_difference,
            "difference_ci_low": float(d_ci[0]),
            "difference_ci_high": float(d_ci[1]),
            "difference_ci_excludes_zero": bool(d_ci[1] < 0.0 or d_ci[0] > 0.0),
            "improvement_percent": percent_improvement(observed_baseline[layer], observed_qbm[layer]),
            "paired_block_permutation_p_two_sided": p_values[layer],
            "permutation_significant_at_05": bool(p_values[layer] < 0.05),
            "test_dates_contiguous": False,
            "inference_note": "Blocks follow calendar-sorted dates from the random GEFCom TEST split.",
        })
    return rows


def aggregate_layer_scores(contributions, seed_weights, day_weights):
    if contributions["seeded"]:
        field_coverage = np.einsum("d,sdzq->szq", day_weights, contributions["field_day"])
        field_scores = maqce(field_coverage).mean(axis=1)
        date_scores = np.einsum("d,sd->s", day_weights, maqce(contributions["date"]))
        horizon_coverage = np.einsum("d,sdhq->shq", day_weights, contributions["horizon_day"])
        horizon_scores = maqce(horizon_coverage).mean(axis=1)
        global_coverage = np.einsum("d,sdq->sq", day_weights, contributions["date"])
        global_scores = maqce(global_coverage)
        return {
            "field": float(np.dot(seed_weights, field_scores)),
            "date": float(np.dot(seed_weights, date_scores)),
            "horizon": float(np.dot(seed_weights, horizon_scores)),
            "global_pooled": float(np.dot(seed_weights, global_scores)),
        }
    else:
        field_coverage = np.einsum("d,dzq->zq", day_weights, contributions["field_day"])
        date_coverage = contributions["date"]
        horizon_coverage = np.einsum("d,dhq->hq", day_weights, contributions["horizon_day"])
    return {
        "field": float(maqce(field_coverage).mean()),
        "date": float(np.dot(day_weights, maqce(date_coverage))),
        "horizon": float(maqce(horizon_coverage).mean()),
        "global_pooled": float(maqce(np.einsum("d,dq->q", day_weights, date_coverage))),
    }


def circular_block_sample(n_dates, block_length, rng):
    n_blocks = int(np.ceil(n_dates / block_length))
    starts = rng.integers(0, n_dates, size=n_blocks)
    indexes = np.concatenate([
        (start + np.arange(block_length)) % n_dates for start in starts
    ])
    return indexes[:n_dates]


def paired_block_permutation_p(qbm, baseline, block_length, rng, max_random=10000):
    n_seeds = qbm["date"].shape[0]
    n_dates = baseline["date"].shape[0]
    block_ids = np.arange(n_dates) // block_length
    n_blocks = int(block_ids.max() + 1)
    observed_q = aggregate_layer_scores(
        qbm, np.full(n_seeds, 1.0 / n_seeds), np.full(n_dates, 1.0 / n_dates)
    )
    observed_b = aggregate_layer_scores(baseline, None, np.full(n_dates, 1.0 / n_dates))
    observed = {layer: abs(observed_q[layer] - observed_b[layer]) for layer in observed_q}

    exact = n_blocks <= 15
    if exact:
        block_signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n_blocks)))
    else:
        block_signs = rng.choice([-1.0, 1.0], size=(max_random, n_blocks))
    exceedances = {layer: 0 for layer in observed}
    for signs_by_block in block_signs:
        signs = signs_by_block[block_ids]
        differences = {layer: [] for layer in observed}
        for seed in range(n_seeds):
            qbm_seed = seed_contributions(qbm, seed)
            permuted_q, permuted_b = swap_contributions_by_date(qbm_seed, baseline, signs)
            q_scores = aggregate_layer_scores(permuted_q, None, np.full(n_dates, 1.0 / n_dates))
            b_scores = aggregate_layer_scores(permuted_b, None, np.full(n_dates, 1.0 / n_dates))
            for layer in observed:
                differences[layer].append(q_scores[layer] - b_scores[layer])
        for layer in observed:
            if abs(float(np.mean(differences[layer]))) >= observed[layer] - 1e-12:
                exceedances[layer] += 1
    if exact:
        p_values = {layer: exceedances[layer] / len(block_signs) for layer in observed}
    else:
        p_values = {
            layer: (exceedances[layer] + 1) / (len(block_signs) + 1) for layer in observed
        }
    return p_values, len(block_signs)


def seed_contributions(qbm, seed):
    return {
        "seeded": False,
        "field_day": qbm["field_day"][seed],
        "date": qbm["date"][seed],
        "horizon_day": qbm["horizon_day"][seed],
    }


def swap_contributions_by_date(qbm, baseline, signs):
    q_out = {"seeded": False}
    b_out = {"seeded": False}
    for key in ("field_day", "date", "horizon_day"):
        expand = (slice(None),) + (None,) * (qbm[key].ndim - 1)
        sign_array = signs[expand]
        midpoint = 0.5 * (qbm[key] + baseline[key])
        half_difference = 0.5 * (qbm[key] - baseline[key])
        q_out[key] = midpoint + sign_array * half_difference
        b_out[key] = midpoint - sign_array * half_difference
    return q_out, b_out


def percent_improvement(baseline, candidate):
    if abs(baseline) < 1e-12:
        return float("nan")
    return 100.0 * (baseline - candidate) / baseline


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stratified wind calibration. MAQCE is measured in percentage points. Confidence intervals use a hierarchical moving-block bootstrap over sorted TEST dates and QBM seeds; p-values use paired date-block randomization. Negative differences favor QBM-VAE.}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Comparator & Layer & QBM & Baseline & Difference (95\% CI) & $p$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            "%s & %s & %.3f & %.3f & %.3f [%.3f, %.3f] & %.4f \\\\" % (
                row["comparator"],
                row["layer"].replace("_", " "),
                row["qbm_maqce"],
                row["comparator_maqce"],
                row["qbm_minus_comparator"],
                row["difference_ci_low"],
                row["difference_ci_high"],
                row["paired_block_permutation_p_two_sided"],
            )
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUTPUT_DIR / "wind_reliability_inference_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_details(payloads, dates):
    for comparator, metrics in payloads.items():
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0))
        x_field = np.arange(1, N_ZONES + 1)
        axes[0].plot(x_field, metrics["field"][0], marker="o", label="QBM-VAE")
        axes[0].plot(x_field, metrics["field"][1], marker="s", label=comparator)
        axes[0].set(xlabel="Wind farm", ylabel="MAQCE (pp)", xticks=x_field)
        axes[0].legend()

        x_date = np.arange(len(dates))
        axes[1].plot(x_date, metrics["date"][0], label="QBM-VAE")
        axes[1].plot(x_date, metrics["date"][1], label=comparator)
        axes[1].set(xlabel="Calendar-sorted TEST date", ylabel="MAQCE (pp)")

        x_horizon = np.arange(1, 25)
        axes[2].plot(x_horizon, metrics["horizon"][0], marker="o", label="QBM-VAE")
        axes[2].plot(x_horizon, metrics["horizon"][1], marker="s", label=comparator)
        axes[2].set(xlabel="Forecast horizon", ylabel="MAQCE (pp)", xticks=np.arange(1, 25, 2))
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe_name = comparator.lower().replace("-", "_")
        fig.savefig(OUTPUT_DIR / ("wind_reliability_strata_%s.pdf" % safe_name))
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified Wind MAQCE with paired date-block inference.")
    parser.add_argument("--comparators", nargs="+", default=["VAE", "NF-UMNN"], choices=["VAE", "NF-UMNN"])
    parser.add_argument("--block-length", type=int, default=7)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    return args


if __name__ == "__main__":
    main()
