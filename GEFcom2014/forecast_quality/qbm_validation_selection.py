# -*- coding: utf-8 -*-

import csv
from pathlib import Path

import numpy as np

from GEFcom2014 import load_data, pv_data, wind_data
from GEFcom2014.forecast_quality.compare_scenarios import (
    DATA_DIR,
    ROOT_DIR,
    evaluate_model,
    load_scenarios,
)


OUTPUT_DIR = ROOT_DIR / "export" / "paper_summary"
TRACKS = ("wind", "pv", "load")
RELIABILITY_WEIGHT = 2e-4


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in TRACKS:
        y_vs, y_test = load_targets(tag)
        for vs_path in discover_qbm_vs_files(tag):
            test_path = Path(str(vs_path).replace("_100_VS.pickle", "_100_TEST.pickle"))
            if not test_path.is_file():
                continue
            model = label_from_path(tag, vs_path)
            vs_scenarios = load_scenarios(vs_path, expected_periods=y_vs.size)
            test_scenarios = load_scenarios(test_path, expected_periods=y_test.size)
            vs_metrics = evaluate_model(vs_scenarios, y_vs, tag)
            test_metrics = evaluate_model(test_scenarios, y_test, tag)
            objective = vs_metrics["crps_mean"] + RELIABILITY_WEIGHT * vs_metrics["reliability_mae"]
            rows.append({
                "track": tag,
                "model": model,
                "vs_objective": objective,
                "vs_qs": vs_metrics["plf_mean"],
                "vs_crps_percent": 100 * vs_metrics["crps_mean"],
                "vs_reliability_mae": vs_metrics["reliability_mae"],
                "vs_variogram_score": vs_metrics["variogram_score"],
                "vs_ramp_quantile_mae": vs_metrics["ramp_quantile_mae"],
                "vs_corr_mae": vs_metrics["corr_mae"],
                "test_qs": test_metrics["plf_mean"],
                "test_crps_percent": 100 * test_metrics["crps_mean"],
                "test_reliability_mae": test_metrics["reliability_mae"],
                "test_energy_score": test_metrics["energy_score"],
                "test_variogram_score": test_metrics["variogram_score"],
                "test_corr_mae": test_metrics["corr_mae"],
                "test_ramp_quantile_mae": test_metrics["ramp_quantile_mae"],
                "scenario_file": str(test_path.relative_to(ROOT_DIR)),
            })

    rows.sort(key=lambda row: (
        row["track"],
        round(row["vs_objective"], 12),
        row["vs_variogram_score"],
        row["vs_ramp_quantile_mae"],
        row["vs_corr_mae"],
    ))
    add_ranks(rows)
    write_csv(rows)
    write_latex(rows)
    print("QBM validation selection written to %s" % OUTPUT_DIR)


def load_targets(tag):
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

    y_vs = data[3].copy()
    y_test = data[5].copy()
    if tag == "pv":
        non_null_indexes = list(np.delete(np.arange(24), indices))
        y_vs = rebuild_pv_targets(y_vs, indices, non_null_indexes)
        y_test = rebuild_pv_targets(y_test, indices, non_null_indexes)
    return y_vs.values, y_test.values


def rebuild_pv_targets(df_y, indices, non_null_indexes):
    df_y.columns = non_null_indexes
    for index in indices:
        df_y[index] = 0
    return df_y.sort_index(axis=1)


def discover_qbm_vs_files(tag):
    qbm_dir = ROOT_DIR / "export" / ("qbm_vae_%s" % tag)
    return sorted(qbm_dir.glob("scenarios_%s_QBMVAE_*_100_VS.pickle" % tag))


def label_from_path(tag, path):
    suffix = path.stem.removeprefix("scenarios_%s_QBMVAE_" % tag)
    suffix = suffix.removesuffix("_100_VS")
    parts = suffix.split("_")
    variant = "_".join(parts[1:-1]) if len(parts) >= 3 else suffix
    seed = parts[-1] if len(parts) >= 3 and parts[-1].isdigit() else "0"
    if seed == "0":
        return "QBM-VAE (%s)" % variant
    return "QBM-VAE (%s seed %s)" % (variant, seed)


def add_ranks(rows):
    for track in TRACKS:
        track_rows = [row for row in rows if row["track"] == track]
        for rank, row in enumerate(track_rows, start=1):
            row["vs_rank"] = rank


def write_csv(rows):
    path = OUTPUT_DIR / "qbm_validation_selection.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "track",
            "vs_rank",
            "model",
            "vs_objective",
            "vs_qs",
            "vs_crps_percent",
            "vs_reliability_mae",
            "vs_variogram_score",
            "vs_ramp_quantile_mae",
            "vs_corr_mae",
            "test_qs",
            "test_crps_percent",
            "test_reliability_mae",
            "test_energy_score",
            "test_variogram_score",
            "test_corr_mae",
            "test_ramp_quantile_mae",
            "scenario_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{QBM-VAE model selection on the validation set. Ties in the marginal objective are broken by validation dependence metrics. Test metrics are reported after selection.}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Track & Model & VS rank & VS obj. & Test QS & Test CRPS (\%) & Test Rel. \\",
        r"\hline",
    ]
    for track in TRACKS:
        top_rows = [row for row in rows if row["track"] == track and row["vs_rank"] <= 3]
        for index, row in enumerate(top_rows):
            track_label = pretty_track(track) if index == 0 else ""
            lines.append(
                "%s & %s & %d & %.4f & %.4f & %.4f & %.4f \\\\" % (
                    track_label,
                    compact_model(row["model"]),
                    row["vs_rank"],
                    row["vs_objective"],
                    row["test_qs"],
                    row["test_crps_percent"],
                    row["test_reliability_mae"],
                )
            )
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUTPUT_DIR / "qbm_validation_selection_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def compact_model(model):
    return model.replace("QBM-VAE ", "")


def pretty_track(track):
    return {"wind": "Wind", "pv": "PV", "load": "Load"}[track]


if __name__ == "__main__":
    main()
