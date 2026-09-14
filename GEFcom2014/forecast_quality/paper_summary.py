# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT_DIR / "export" / "scenario_comparison" / "metrics.csv"
OUTPUT_DIR = ROOT_DIR / "export" / "paper_summary"

TRACK_ORDER = ["wind", "pv", "load"]
MODEL_ORDER = ["NF-UMNN", "VAE", "GAN", "GC", "RAND", "QBM-VAE", "QBM-VAE calibrated"]
RAW_QBM_LABEL = "QBM-VAE (sa)"
BEST_QBM_BY_TRACK = {
    "wind": "QBM-VAE (gibbs_calm_trc)",
    "pv": "QBM-VAE (pv64_gibbs_calm_trc)",
    "load": "QBM-VAE (gibbs_calx_trc)",
}
CALIBRATION_JSON_BY_TRACK = {
    "wind": ROOT_DIR / "export" / "qbm_vae_wind" / "calibration_wind_QBMVAE_2_gibbs_calm_0.json",
    "pv": ROOT_DIR / "export" / "qbm_vae_pv" / "calibration_pv_QBMVAE_2_pv64_gibbs_calm_0.json",
    "load": ROOT_DIR / "export" / "qbm_vae_load" / "calibration_load_QBMVAE_2_gibbs_calx_0.json",
}


def main():
    rows = read_metrics()
    selected = select_paper_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_main_metrics(selected)
    write_latex_table(selected)
    write_qbm_config_summary(selected)
    write_improvement_summary(selected)
    write_graph_ablation_summary(rows)
    write_dependency_metrics_summary(rows)
    print("Paper summary written to %s" % OUTPUT_DIR)


def read_metrics():
    with METRICS_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["mean_qs"] = float(row["mean_qs"])
        row["mean_crps_percent"] = float(row["mean_crps_percent"])
        row["reliability_mae"] = float(row["reliability_mae"])
        row["n_scenarios"] = int(row["n_scenarios"])
        for optional_metric in ("energy_score", "variogram_score", "corr_mae", "ramp_quantile_mae"):
            if optional_metric in row and row[optional_metric] != "":
                row[optional_metric] = float(row[optional_metric])
    return rows


def select_paper_rows(rows):
    by_key = {(row["track"], row["model"]): row for row in rows}
    selected = []
    for track in TRACK_ORDER:
        for model in ["NF-UMNN", "VAE", "GAN", "GC", "RAND"]:
            selected.append(as_paper_row(by_key[(track, model)], model))
        selected.append(as_paper_row(by_key[(track, RAW_QBM_LABEL)], "QBM-VAE"))
        selected.append(as_paper_row(by_key[(track, BEST_QBM_BY_TRACK[track])], "QBM-VAE calibrated"))
    return selected


def as_paper_row(row, model):
    return {
        "track": row["track"],
        "model": model,
        "n_scenarios": row["n_scenarios"],
        "qs": row["mean_qs"],
        "crps_percent": row["mean_crps_percent"],
        "reliability_mae": row["reliability_mae"],
        "scenario_file": row["scenario_file"],
    }


def write_main_metrics(rows):
    path = OUTPUT_DIR / "main_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["track", "model", "n_scenarios", "qs", "crps_percent", "reliability_mae", "scenario_file"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows):
    best = best_values_by_track(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Probabilistic forecasting performance on GEFCom2014. Lower is better for all metrics.}",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Track & Model & QS & CRPS (\%) & Rel. MAE \\",
        r"\hline",
    ]
    for track in TRACK_ORDER:
        track_rows = [row for row in rows if row["track"] == track]
        for index, row in enumerate(track_rows):
            track_label = pretty_track(track) if index == 0 else ""
            lines.append(
                "%s & %s & %s & %s & %s \\\\" % (
                    track_label,
                    row["model"],
                    fmt_metric(row["qs"], best[track]["qs"]),
                    fmt_metric(row["crps_percent"], best[track]["crps_percent"]),
                    fmt_metric(row["reliability_mae"], best[track]["reliability_mae"]),
                )
            )
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUTPUT_DIR / "main_metrics_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def write_qbm_config_summary(rows):
    summary = []
    for track in TRACK_ORDER:
        calibration = read_json(CALIBRATION_JSON_BY_TRACK[track])
        qbm_row = next(row for row in rows if row["track"] == track and row["model"] == "QBM-VAE calibrated")
        summary.append({
            "track": track,
            "selected_model": BEST_QBM_BY_TRACK[track],
            "beta_eff": calibration.get("beta"),
            "scale_multiplier": calibration.get("scale_multiplier"),
            "vs_objective": calibration.get("objective"),
            "test_qs": qbm_row["qs"],
            "test_crps_percent": qbm_row["crps_percent"],
            "test_reliability_mae": qbm_row["reliability_mae"],
        })
    with (OUTPUT_DIR / "qbm_calibration_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def write_improvement_summary(rows):
    summary = []
    for track in TRACK_ORDER:
        vae = next(row for row in rows if row["track"] == track and row["model"] == "VAE")
        qbm = next(row for row in rows if row["track"] == track and row["model"] == "QBM-VAE calibrated")
        summary.append({
            "track": track,
            "vae_qs": vae["qs"],
            "qbm_qs": qbm["qs"],
            "qs_improvement_percent": percent_reduction(vae["qs"], qbm["qs"]),
            "vae_crps_percent": vae["crps_percent"],
            "qbm_crps_percent": qbm["crps_percent"],
            "crps_improvement_percent": percent_reduction(vae["crps_percent"], qbm["crps_percent"]),
            "vae_reliability_mae": vae["reliability_mae"],
            "qbm_reliability_mae": qbm["reliability_mae"],
            "reliability_improvement_percent": percent_reduction(vae["reliability_mae"], qbm["reliability_mae"]),
        })
    with (OUTPUT_DIR / "improvement_vs_vae.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Relative improvement of calibrated QBM-VAE over VAE. Positive values indicate lower error.}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Track & QS reduction (\%) & CRPS reduction (\%) & Rel. MAE reduction (\%) \\",
        r"\hline",
    ]
    for row in summary:
        lines.append(
            "%s & %.2f & %.2f & %.2f \\\\" % (
                pretty_track(row["track"]),
                row["qs_improvement_percent"],
                row["crps_improvement_percent"],
                row["reliability_improvement_percent"],
            )
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUTPUT_DIR / "improvement_vs_vae_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def write_graph_ablation_summary(rows):
    by_key = {(row["track"], row["model"]): row for row in rows}
    load_models = [
        ("VAE baseline", "VAE"),
        ("QBM-VAE without Ising edges", "QBM-VAE (loadnoj_gibbs_cal)"),
        ("QBM-VAE with random sparse graph", "QBM-VAE (loadrandgraph_gibbs_cal)"),
        ("QBM-VAE with Temporal-MI sparse graph", "QBM-VAE (gibbs_cal)"),
    ]
    temporal = by_key[("load", "QBM-VAE (gibbs_cal)")]
    ablation = []
    for display_name, model_name in load_models:
        row = by_key[("load", model_name)]
        ablation.append({
            "track": "load",
            "model": display_name,
            "qs": row["mean_qs"],
            "crps_percent": row["mean_crps_percent"],
            "reliability_mae": row["reliability_mae"],
            "qs_gap_to_temporal_mi_percent": 100.0 * (row["mean_qs"] - temporal["mean_qs"]) / temporal["mean_qs"],
            "crps_gap_to_temporal_mi_percent": 100.0 * (
                row["mean_crps_percent"] - temporal["mean_crps_percent"]
            ) / temporal["mean_crps_percent"],
        })
    with (OUTPUT_DIR / "load_graph_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(ablation[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation of the latent Ising graph on GEFCom2014 Load. Lower is better.}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Model & QS & CRPS (\%) & Rel. MAE & QS gap (\%) \\",
        r"\hline",
    ]
    best_qs = min(row["qs"] for row in ablation)
    for row in ablation:
        lines.append(
            "%s & %s & %.4f & %.4f & %.2f \\\\" % (
                row["model"],
                fmt_metric(row["qs"], best_qs),
                row["crps_percent"],
                row["reliability_mae"],
                row["qs_gap_to_temporal_mi_percent"],
            )
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUTPUT_DIR / "load_graph_ablation_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def write_dependency_metrics_summary(rows):
    by_key = {(row["track"], row["model"]): row for row in rows}
    selected = []
    for track in TRACK_ORDER:
        for display_name, model_name in [
            ("NF-UMNN", "NF-UMNN"),
            ("VAE", "VAE"),
            ("GAN", "GAN"),
            ("QBM-VAE calibrated", BEST_QBM_BY_TRACK[track]),
        ]:
            row = by_key[(track, model_name)]
            selected.append({
                "track": track,
                "model": display_name,
                "energy_score": row["energy_score"],
                "variogram_score": row["variogram_score"],
                "corr_mae": row["corr_mae"],
                "ramp_quantile_mae": row["ramp_quantile_mae"],
            })

    with (OUTPUT_DIR / "dependency_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(selected[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multivariate and temporal-dependence scenario metrics on GEFCom2014. Lower is better.}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Track & Model & ES & VS & Corr. MAE & Ramp MAE \\",
        r"\hline",
    ]
    for track in TRACK_ORDER:
        track_rows = [row for row in selected if row["track"] == track]
        best = {
            "energy_score": min(row["energy_score"] for row in track_rows),
            "variogram_score": min(row["variogram_score"] for row in track_rows),
            "corr_mae": min(row["corr_mae"] for row in track_rows),
            "ramp_quantile_mae": min(row["ramp_quantile_mae"] for row in track_rows),
        }
        for index, row in enumerate(track_rows):
            track_label = pretty_track(track) if index == 0 else ""
            lines.append(
                "%s & %s & %s & %s & %s & %s \\\\" % (
                    track_label,
                    row["model"],
                    fmt_metric(row["energy_score"], best["energy_score"]),
                    fmt_metric(row["variogram_score"], best["variogram_score"]),
                    fmt_metric(row["corr_mae"], best["corr_mae"]),
                    fmt_metric(row["ramp_quantile_mae"], best["ramp_quantile_mae"]),
                )
            )
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUTPUT_DIR / "dependency_metrics_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def percent_reduction(baseline, candidate):
    return 100.0 * (baseline - candidate) / baseline


def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def best_values_by_track(rows):
    best = {}
    for track in TRACK_ORDER:
        track_rows = [row for row in rows if row["track"] == track]
        best[track] = {
            "qs": min(row["qs"] for row in track_rows),
            "crps_percent": min(row["crps_percent"] for row in track_rows),
            "reliability_mae": min(row["reliability_mae"] for row in track_rows),
        }
    return best


def fmt_metric(value, best_value):
    text = "%.4f" % value
    if abs(value - best_value) < 1e-8:
        return r"\textbf{%s}" % text
    return text


def pretty_track(track):
    return {"wind": "Wind", "pv": "PV", "load": "Load"}[track]


if __name__ == "__main__":
    main()
