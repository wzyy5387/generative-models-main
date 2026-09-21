# -*- coding: utf-8 -*-

import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

from GEFcom2014 import load_data, pv_data, wind_data
from GEFcom2014.forecast_quality.utils_quality import (
    compute_reliability,
    plf_per_quantile,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / "GEFcom2014"
DATA_DIR = PACKAGE_DIR / "data"
BASELINE_SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
OUTPUT_DIR = ROOT_DIR / "export" / "scenario_comparison"

TRACKS = ("wind", "pv", "load")
N_QUANTILES = 99
MAX_CRPS_SCENARIOS = 100
VARIOGRAM_BETA = 0.5

NF_UMNN_ID = {"pv": 3, "wind": 1, "load": 1}

MODEL_COLORS = {
    "NF-UMNN": "tab:blue",
    "VAE": "tab:orange",
    "GAN": "tab:green",
    "GC": "tab:red",
    "RAND": "tab:gray",
}


def load_track_data(tag):
    if tag == "pv":
        data, indices = pv_data(DATA_DIR / "solar_new.csv", test_size=50, random_state=0)
    elif tag == "wind":
        data = wind_data(DATA_DIR / "wind_data_all_zone.csv", test_size=50, random_state=0)
        indices = []
    elif tag == "load":
        data = load_data(DATA_DIR / "load_data_track1.csv", test_size=50, random_state=0)
        indices = []
    else:
        raise ValueError("Unknown track: %s" % tag)

    df_y_test = data[5].copy()
    if tag == "pv":
        non_null_indexes = list(np.delete(np.arange(24), indices))
        df_y_test.columns = non_null_indexes
        for index in indices:
            df_y_test[index] = 0
        df_y_test = df_y_test.sort_index(axis=1)
    return df_y_test


def discover_scenario_files(tag):
    files = {
        "NF-UMNN": BASELINE_SCENARIOS_DIR / "nfs" /
        ("scenarios_%s_UMNN_M_%s_0_100_TEST.pickle" % (tag, NF_UMNN_ID[tag])),
        "VAE": BASELINE_SCENARIOS_DIR / "vae" /
        ("scenarios_%s_VAElinear_1_0_100_TEST.pickle" % tag),
        "GAN": BASELINE_SCENARIOS_DIR / "gan" /
        ("scenarios_%s_GAN_wasserstein_1_0_100_TEST.pickle" % tag),
        "GC": BASELINE_SCENARIOS_DIR / "gc" /
        ("scenarios_%s_gc_100_TEST.pickle" % tag),
        "RAND": BASELINE_SCENARIOS_DIR / "random" /
        ("scenarios_%s_random_100_TEST.pickle" % tag),
    }

    qbm_dir = ROOT_DIR / "export" / ("qbm_vae_%s" % tag)
    for path in sorted(qbm_dir.glob("scenarios_%s_QBMVAE_*_100_TEST.pickle" % tag)):
        suffix = path.stem.removeprefix("scenarios_%s_QBMVAE_" % tag)
        suffix = suffix.removesuffix("_100_TEST")
        parts = suffix.split("_")
        variant = "_".join(parts[1:-1]) if len(parts) >= 3 else suffix
        seed = parts[-1] if len(parts) >= 3 and parts[-1].isdigit() else "0"
        label = "QBM-VAE (%s)" % variant if seed == "0" else "QBM-VAE (%s seed %s)" % (variant, seed)
        files[label] = path

    return {label: path for label, path in files.items() if path.is_file()}


def load_scenarios(path, expected_periods):
    with path.open("rb") as handle:
        scenarios = pickle.load(handle)

    if isinstance(scenarios, dict):
        daily = []
        for key in sorted(scenarios, key=_sortable_key):
            values = np.asarray(scenarios[key], dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("%s contains a non-matrix day" % path)
            if values.shape[1] == 24:
                daily.append(values.transpose())
            elif values.shape[0] == 24:
                daily.append(values)
            else:
                raise ValueError("%s contains a day without 24 periods" % path)
        scenarios = np.concatenate(daily, axis=0)
    else:
        scenarios = np.asarray(scenarios, dtype=np.float64)

    if scenarios.ndim != 2:
        raise ValueError("%s must contain a 2-D scenario matrix" % path)
    if scenarios.shape[0] != expected_periods and scenarios.shape[1] == expected_periods:
        scenarios = scenarios.transpose()
    if scenarios.shape[0] != expected_periods:
        raise ValueError(
            "%s has %s periods; expected %s" % (path, scenarios.shape[0], expected_periods)
        )
    if not np.isfinite(scenarios).all():
        raise ValueError("%s contains NaN or infinite values" % path)
    return scenarios


def evaluate_model(scenarios, y_matrix, tag):
    y_true = y_matrix.reshape(-1)
    q_set = np.arange(1, N_QUANTILES + 1) / (N_QUANTILES + 1)
    quantiles = np.quantile(scenarios, q=q_set, axis=1).transpose()
    plf = plf_per_quantile(quantiles=quantiles, y_true=y_true)
    reliability = compute_reliability(y_true=y_true, y_quantile=quantiles, tag=tag)
    reliability_mae = mean_absolute_error(q_set * 100, reliability)
    crps = crps_per_hour_fast(scenarios, y_true, max_s=MAX_CRPS_SCENARIOS)
    es = energy_score_by_day(scenarios, y_matrix, max_s=MAX_CRPS_SCENARIOS)
    vs = variogram_score_by_day(scenarios, y_matrix, beta=VARIOGRAM_BETA, max_s=MAX_CRPS_SCENARIOS)
    corr_error = temporal_correlation_error(scenarios, y_matrix, max_s=MAX_CRPS_SCENARIOS)
    ramp_error = ramp_distribution_error(scenarios, y_matrix, max_s=MAX_CRPS_SCENARIOS)
    return {
        "plf": plf,
        "plf_mean": float(plf.mean()),
        "reliability": reliability,
        "reliability_mae": float(reliability_mae),
        "crps": crps,
        "crps_mean": float(crps.mean()),
        "energy_score": float(es.mean()),
        "variogram_score": float(vs.mean()),
        "corr_mae": float(corr_error),
        "ramp_quantile_mae": float(ramp_error),
        "n_scenarios": int(scenarios.shape[1]),
    }


def crps_per_hour_fast(scenarios, y_true, max_s=100):
    """
    Compute ensemble CRPS with the sorted-sample identity.

    This is mathematically equivalent to the pairwise formulation used by the
    original project, but avoids a Python loop over every sample pair.
    """
    samples = np.asarray(scenarios[:, :max_s], dtype=np.float64)
    observations = np.asarray(y_true, dtype=np.float64)
    n_s = samples.shape[1]
    sorted_samples = np.sort(samples, axis=1)
    coefficients = 2 * np.arange(1, n_s + 1) - n_s - 1
    observation_term = np.mean(np.abs(samples - observations[:, None]), axis=1)
    ensemble_term = np.sum(sorted_samples * coefficients[None, :], axis=1) / (n_s ** 2)
    crps_by_period = observation_term - ensemble_term
    return crps_by_period.reshape(-1, 24).mean(axis=0)


def energy_score_by_day(scenarios, y_matrix, max_s=100):
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    observations = np.asarray(y_matrix, dtype=np.float64)
    scores = []
    for day in range(observations.shape[0]):
        day_samples = samples[day]
        simple_term = np.linalg.norm(day_samples - observations[day, :, None], axis=0).mean()
        pairwise = day_samples[:, :, None] - day_samples[:, None, :]
        ensemble_term = np.linalg.norm(pairwise, axis=0).mean() / 2.0
        scores.append(simple_term - ensemble_term)
    return np.asarray(scores, dtype=np.float64)


def variogram_score_by_day(scenarios, y_matrix, beta=0.5, max_s=100):
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    observations = np.asarray(y_matrix, dtype=np.float64)
    scores = []
    for day in range(observations.shape[0]):
        true_diff = np.abs(observations[day, :, None] - observations[day, None, :]) ** beta
        sample_diff = np.abs(samples[day, :, None, :] - samples[day, None, :, :]) ** beta
        expected_diff = sample_diff.mean(axis=2)
        scores.append(np.square(true_diff - expected_diff).mean())
    return np.asarray(scores, dtype=np.float64)


def temporal_correlation_error(scenarios, y_matrix, max_s=100):
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    observations = np.asarray(y_matrix, dtype=np.float64)
    valid = np.std(observations, axis=0) > 1e-10
    if valid.sum() < 2:
        return 0.0

    true_corr = np.corrcoef(observations[:, valid], rowvar=False)
    scenario_corrs = []
    for scenario_index in range(samples.shape[2]):
        scenario_values = samples[:, valid, scenario_index]
        corr = np.corrcoef(scenario_values, rowvar=False)
        if np.isfinite(corr).all():
            scenario_corrs.append(corr)
    if not scenario_corrs:
        return float("nan")
    model_corr = np.mean(scenario_corrs, axis=0)
    off_diag = ~np.eye(valid.sum(), dtype=bool)
    return float(np.abs(model_corr[off_diag] - true_corr[off_diag]).mean())


def ramp_distribution_error(scenarios, y_matrix, max_s=100):
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    observations = np.asarray(y_matrix, dtype=np.float64)
    true_ramps = np.diff(observations, axis=1)
    sample_ramps = np.diff(samples, axis=1)
    q_grid = np.linspace(0.01, 0.99, 99)
    errors = []
    for hour in range(true_ramps.shape[1]):
        true_quantiles = np.quantile(true_ramps[:, hour], q_grid)
        sample_quantiles = np.quantile(sample_ramps[:, hour, :].reshape(-1), q_grid)
        errors.append(np.abs(sample_quantiles - true_quantiles).mean())
    return float(np.mean(errors))


def reshape_scenarios_by_day(scenarios, n_days, max_s=100):
    samples = np.asarray(scenarios[:, :max_s], dtype=np.float64)
    expected = n_days * 24
    if samples.shape[0] != expected:
        raise ValueError("Expected %s periods, got %s" % (expected, samples.shape[0]))
    return samples.reshape(n_days, 24, samples.shape[1])


def _ensemble_crps(samples, target):
    """CRPS for one vector-valued trajectory represented by ensemble samples."""
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if samples.ndim != 2 or target.shape != (samples.shape[1],):
        raise ValueError("samples/target must have shapes (n_scenarios, n_features)/(n_features,)")
    return float(np.mean(np.abs(samples - target[None, :])) -
                 0.5 * np.mean(np.abs(samples[:, None, :] - samples[None, :, :])))


def ramp_crps_by_day(scenarios, y_matrix, max_s=100):
    """Compute ramp CRPS per date from the scenario ramp trajectories."""
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    observations = np.asarray(y_matrix, dtype=np.float64)
    return np.asarray([
        _ensemble_crps(np.diff(samples[day], axis=0).T, np.diff(observations[day]))
        for day in range(observations.shape[0])
    ], dtype=np.float64)


def coverage_interval_by_day(scenarios, y_matrix, lower=0.05, upper=0.95, max_s=100):
    """Return 90% PICP and mean interval width for every date."""
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    observations = np.asarray(y_matrix, dtype=np.float64)
    low = np.quantile(samples, lower, axis=2)
    high = np.quantile(samples, upper, axis=2)
    return (np.mean((observations >= low) & (observations <= high), axis=1),
            np.mean(high - low, axis=1))


def maqce_by_day(scenarios, y_matrix, max_s=100):
    """Mean absolute quantile coverage error per date on a fixed grid."""
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    observations = np.asarray(y_matrix, dtype=np.float64)
    levels = np.linspace(0.05, 0.95, 19)
    observed = np.asarray([
        [np.mean(observations[day] <= np.quantile(samples[day], level, axis=1))
         for level in levels]
        for day in range(observations.shape[0])
    ])
    return np.mean(np.abs(observed - levels[None, :]), axis=1)


def evaluate_model_daily(scenarios, y_matrix, max_s=100):
    """Return the raw, per-date metrics used by the rolling-origin protocol."""
    scenarios = np.asarray(scenarios, dtype=np.float64)
    y_matrix = np.asarray(y_matrix, dtype=np.float64)
    if y_matrix.ndim != 2 or scenarios.shape[0] != y_matrix.shape[0] * 24:
        raise ValueError("Scenario periods must equal 24 times the number of dates")
    samples = reshape_scenarios_by_day(scenarios, y_matrix.shape[0], max_s)
    target = y_matrix.reshape(-1)
    flat = scenarios[:, :max_s]
    ordered = np.sort(flat, axis=1)
    n_scenarios = flat.shape[1]
    coefficients = 2 * np.arange(1, n_scenarios + 1) - n_scenarios - 1
    crps_period = (np.mean(np.abs(flat - target[:, None]), axis=1) -
                   np.sum(ordered * coefficients[None, :], axis=1) / n_scenarios ** 2)
    crps = crps_period.reshape(y_matrix.shape[0], 24).mean(axis=1)
    picp, width = coverage_interval_by_day(scenarios, y_matrix, max_s=max_s)
    return {
        "CRPS_raw": crps,
        "Energy": energy_score_by_day(scenarios, y_matrix, max_s=max_s),
        "Variogram": variogram_score_by_day(scenarios, y_matrix, max_s=max_s),
        "ramp_CRPS": ramp_crps_by_day(scenarios, y_matrix, max_s=max_s),
        "PICP90": picp,
        "coverage": picp,
        "interval_width90": width,
        "MAQCE": maqce_by_day(scenarios, y_matrix, max_s=max_s),
    }


def export_metrics(rows):
    path = OUTPUT_DIR / "metrics.csv"
    fieldnames = [
        "track",
        "model",
        "n_scenarios",
        "mean_qs",
        "mean_crps",
        "mean_crps_percent",
        "reliability_mae",
        "energy_score",
        "variogram_score",
        "corr_mae",
        "ramp_quantile_mae",
        "scenario_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_track(tag, results):
    labels = list(results)
    colors = [_model_color(label, index) for index, label in enumerate(labels)]
    quantile_axis = np.arange(1, N_QUANTILES + 1)
    hour_axis = np.arange(1, 25)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in zip(labels, colors):
        ax.plot(quantile_axis, results[label]["plf"], label=label, color=color, linewidth=2)
    ax.set(xlabel="Quantile (%)", ylabel="Quantile score", xlim=(1, 99))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / ("%s_qs.pdf" % tag))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in zip(labels, colors):
        ax.plot(hour_axis, 100 * results[label]["crps"], label=label, color=color, linewidth=2)
    ax.set(xlabel="Hour", ylabel="CRPS (%)", xlim=(1, 24))
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / ("%s_crps.pdf" % tag))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(quantile_axis, quantile_axis, color="black", linestyle="--", label="Ideal")
    for label, color in zip(labels, colors):
        ax.plot(quantile_axis, results[label]["reliability"], label=label, color=color, linewidth=2)
    ax.set(
        xlabel="Nominal quantile (%)",
        ylabel="Observed frequency (%)",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / ("%s_reliability.pdf" % tag))
    plt.close(fig)


def _model_color(label, index):
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    palette = ("tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:olive")
    return palette[index % len(palette)]


def _sortable_key(value):
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    for tag in TRACKS:
        y_matrix = load_track_data(tag).values
        y_true = y_matrix.reshape(-1)
        scenario_files = discover_scenario_files(tag)
        if not scenario_files:
            print("%s: no scenario files found" % tag)
            continue

        results = {}
        print("\n[%s]" % tag)
        for label, path in scenario_files.items():
            scenarios = load_scenarios(path, expected_periods=y_true.size)
            metrics = evaluate_model(scenarios, y_matrix, tag)
            results[label] = metrics
            metric_rows.append({
                "track": tag,
                "model": label,
                "n_scenarios": metrics["n_scenarios"],
                "mean_qs": "%.8f" % metrics["plf_mean"],
                "mean_crps": "%.8f" % metrics["crps_mean"],
                "mean_crps_percent": "%.4f" % (100 * metrics["crps_mean"]),
                "reliability_mae": "%.4f" % metrics["reliability_mae"],
                "energy_score": "%.8f" % metrics["energy_score"],
                "variogram_score": "%.8f" % metrics["variogram_score"],
                "corr_mae": "%.8f" % metrics["corr_mae"],
                "ramp_quantile_mae": "%.8f" % metrics["ramp_quantile_mae"],
                "scenario_file": str(path.relative_to(ROOT_DIR)),
            })
            print(
                "%-18s QS %.4f | CRPS %.2f%% | reliability MAE %.2f | ES %.4f | VS %.4f"
                % (
                    label,
                    metrics["plf_mean"],
                    100 * metrics["crps_mean"],
                    metrics["reliability_mae"],
                    metrics["energy_score"],
                    metrics["variogram_score"],
                )
            )
        plot_track(tag, results)

    metrics_path = export_metrics(metric_rows)
    print("\nMetrics written to %s" % metrics_path)


if __name__ == "__main__":
    main()
