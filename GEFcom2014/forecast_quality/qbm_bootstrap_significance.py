# -*- coding: utf-8 -*-

"""Paired hierarchical-bootstrap tests for the selected QBM-VAE variants.

The test resamples test days jointly for both models and resamples QBM training
seeds as a second level. A negative score difference means that QBM-VAE is
better than the comparator.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.compare_scenarios import (
    BASELINE_SCENARIOS_DIR,
    NF_UMNN_ID,
    ROOT_DIR,
    TRACKS,
    load_scenarios,
    load_track_data,
)


OUTPUT_DIR = ROOT_DIR / "export" / "paper_summary"
QBM_VARIANT_BY_TRACK = {
    "wind": "gibbs_calm_trc",
    "pv": "pv64_gibbs_calm_trc",
    "load": "gibbs_calx_trc",
}
N_QUANTILES = 99
MAX_SCENARIOS = 100


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    for tag in TRACKS:
        targets = load_track_data(tag).values
        qbm_scores = collect_qbm_scores(tag, targets)
        for comparator, path in baseline_paths(tag).items():
            baseline = load_scenarios(path, expected_periods=targets.size)
            baseline_scores = {
                "QS": daily_qs(baseline, targets),
                "CRPS": daily_crps(baseline, targets),
            }
            for metric, per_seed_scores in qbm_scores.items():
                rows.append(
                    bootstrap_row(
                        tag=tag,
                        metric=metric,
                        comparator=comparator,
                        qbm_daily=per_seed_scores,
                        baseline_daily=baseline_scores[metric],
                        n_bootstrap=args.n_bootstrap,
                        rng=rng,
                    )
                )

    write_csv(rows)
    write_latex(rows)
    print("QBM paired bootstrap results written to %s" % OUTPUT_DIR)


def baseline_paths(tag):
    return {
        "VAE": BASELINE_SCENARIOS_DIR / "vae" /
        ("scenarios_%s_VAElinear_1_0_100_TEST.pickle" % tag),
        "NF-UMNN": BASELINE_SCENARIOS_DIR / "nfs" /
        ("scenarios_%s_UMNN_M_%s_0_100_TEST.pickle" % (tag, NF_UMNN_ID[tag])),
    }


def collect_qbm_scores(tag, targets):
    variant = QBM_VARIANT_BY_TRACK[tag]
    directory = ROOT_DIR / "export" / ("qbm_vae_%s" % tag)
    prefix = "scenarios_%s_QBMVAE_2_%s_" % (tag, variant)
    paths = []
    for path in sorted(directory.glob(prefix + "*_100_TEST.pickle")):
        suffix = path.stem.removesuffix("_100_TEST").removeprefix(prefix)
        if suffix.isdigit():
            paths.append(path)
    if not paths:
        raise FileNotFoundError("No QBM scenarios found for %s (%s)" % (tag, variant))

    qs_scores = []
    crps_scores = []
    for path in paths:
        scenarios = load_scenarios(path, expected_periods=targets.size)
        qs_scores.append(daily_qs(scenarios, targets))
        crps_scores.append(daily_crps(scenarios, targets))
    return {
        "QS": np.asarray(qs_scores, dtype=np.float64),
        "CRPS": np.asarray(crps_scores, dtype=np.float64),
    }


def daily_qs(scenarios, targets):
    samples = np.asarray(scenarios[:, :MAX_SCENARIOS], dtype=np.float64)
    observations = np.asarray(targets, dtype=np.float64).reshape(-1)
    q_grid = np.arange(1, N_QUANTILES + 1, dtype=np.float64) / (N_QUANTILES + 1)
    quantiles = np.quantile(samples, q_grid, axis=1).T
    residual = observations[:, None] - quantiles
    pinball = np.maximum(q_grid[None, :] * residual, (q_grid[None, :] - 1.0) * residual)
    return 100.0 * pinball.mean(axis=1).reshape(-1, 24).mean(axis=1)


def daily_crps(scenarios, targets):
    samples = np.asarray(scenarios[:, :MAX_SCENARIOS], dtype=np.float64)
    observations = np.asarray(targets, dtype=np.float64).reshape(-1)
    n_scenarios = samples.shape[1]
    sorted_samples = np.sort(samples, axis=1)
    coefficients = 2 * np.arange(1, n_scenarios + 1) - n_scenarios - 1
    observation_term = np.abs(samples - observations[:, None]).mean(axis=1)
    ensemble_term = (sorted_samples * coefficients[None, :]).sum(axis=1) / (n_scenarios ** 2)
    return (observation_term - ensemble_term).reshape(-1, 24).mean(axis=1)


def bootstrap_row(tag, metric, comparator, qbm_daily, baseline_daily, n_bootstrap, rng):
    qbm_daily = np.asarray(qbm_daily, dtype=np.float64)
    baseline_daily = np.asarray(baseline_daily, dtype=np.float64)
    if qbm_daily.ndim != 2 or qbm_daily.shape[1] != baseline_daily.size:
        raise ValueError("QBM and baseline daily scores must align")

    n_seeds, n_days = qbm_daily.shape
    observed_qbm = float(qbm_daily.mean())
    observed_baseline = float(baseline_daily.mean())
    observed_difference = observed_qbm - observed_baseline

    seed_indexes = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    day_indexes = rng.integers(0, n_days, size=(n_bootstrap, n_days))
    qbm_resampled = qbm_daily[seed_indexes].mean(axis=1)
    differences = (
        np.take_along_axis(qbm_resampled, day_indexes, axis=1).mean(axis=1)
        - baseline_daily[day_indexes].mean(axis=1)
    )
    ci_low, ci_high = np.quantile(differences, [0.025, 0.975])
    permutation_p = paired_sign_flip_p(
        qbm_daily.mean(axis=0) - baseline_daily,
        n_permutations=n_bootstrap,
        rng=rng,
    )

    return {
        "track": tag,
        "metric": metric,
        "comparator": comparator,
        "qbm_variant": QBM_VARIANT_BY_TRACK[tag],
        "n_seeds": n_seeds,
        "n_days": n_days,
        "qbm_score": observed_qbm,
        "comparator_score": observed_baseline,
        "qbm_minus_comparator": observed_difference,
        "improvement_percent": 100.0 * (observed_baseline - observed_qbm) / observed_baseline,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_ci_excludes_zero": bool(ci_high < 0.0 or ci_low > 0.0),
        "paired_permutation_p_two_sided": permutation_p,
        "permutation_significant_at_05": bool(permutation_p < 0.05),
    }


def paired_sign_flip_p(daily_difference, n_permutations, rng):
    """Two-sided paired randomization p-value for the daily mean difference."""
    daily_difference = np.asarray(daily_difference, dtype=np.float64)
    observed = abs(float(daily_difference.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_permutations, daily_difference.size))
    null_statistics = np.abs((signs * daily_difference[None, :]).mean(axis=1))
    return float((np.count_nonzero(null_statistics >= observed) + 1) / (n_permutations + 1))


def write_csv(rows):
    path = OUTPUT_DIR / "qbm_bootstrap_significance.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired comparison of QBM-VAE against baseline models. Confidence intervals use a hierarchical bootstrap over QBM seeds and test days; p-values use a two-sided daily paired sign-flip test. Negative differences favor QBM-VAE.}",
        r"\begin{tabular}{lllrccc}",
        r"\hline",
        r"Track & Metric & Comparator & Difference & 95\% CI & Improvement (\%) & $p_{\mathrm{perm}}$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            "%s & %s & %s & %.4f & [%.4f, %.4f] & %.2f & %.4f \\\\" % (
                pretty_track(row["track"]),
                row["metric"],
                row["comparator"],
                row["qbm_minus_comparator"],
                row["ci_low"],
                row["ci_high"],
                row["improvement_percent"],
                row["paired_permutation_p_two_sided"],
            )
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUTPUT_DIR / "qbm_bootstrap_significance_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def pretty_track(tag):
    return {"wind": "Wind", "pv": "PV", "load": "Load"}[tag]


def parse_args():
    parser = argparse.ArgumentParser(description="Paired bootstrap tests for selected QBM-VAE variants.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    main()
