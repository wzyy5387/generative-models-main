# -*- coding: utf-8 -*-

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.compare_scenarios import (
    ROOT_DIR,
    TRACKS,
    evaluate_model,
    load_scenarios,
    load_track_data,
)


OUTPUT_DIR = ROOT_DIR / "export" / "paper_summary"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows = collect_seed_rows()
    write_raw(raw_rows)
    summary_rows = summarize(raw_rows)
    write_summary(summary_rows)
    write_latex(summary_rows)
    print("QBM seed summary written to %s" % OUTPUT_DIR)


def collect_seed_rows():
    rows = []
    for tag in TRACKS:
        y_matrix = load_track_data(tag).values
        expected_periods = y_matrix.size
        qbm_dir = ROOT_DIR / "export" / ("qbm_vae_%s" % tag)
        for path in sorted(qbm_dir.glob("scenarios_%s_QBMVAE_*_100_TEST.pickle" % tag)):
            parsed = parse_qbm_scenario_name(tag, path)
            if parsed is None:
                continue
            variant, seed = parsed
            scenarios = load_scenarios(path, expected_periods=expected_periods)
            metrics = evaluate_model(scenarios, y_matrix, tag)
            rows.append({
                "track": tag,
                "variant": variant,
                "seed": seed,
                "mean_qs": metrics["plf_mean"],
                "mean_crps_percent": 100 * metrics["crps_mean"],
                "reliability_mae": metrics["reliability_mae"],
                "energy_score": metrics["energy_score"],
                "variogram_score": metrics["variogram_score"],
                "corr_mae": metrics["corr_mae"],
                "ramp_quantile_mae": metrics["ramp_quantile_mae"],
                "scenario_file": str(path.relative_to(ROOT_DIR)),
            })
    return rows


def parse_qbm_scenario_name(tag, path):
    stem = path.stem
    prefix = "scenarios_%s_QBMVAE_" % tag
    suffix = "_100_TEST"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None
    body = stem[len(prefix):-len(suffix)]
    parts = body.split("_")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    seed = int(parts[-1])
    variant = "_".join(parts[1:-1])
    return variant, seed


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["track"], row["variant"])].append(row)

    summary = []
    metrics = [
        "mean_qs",
        "mean_crps_percent",
        "reliability_mae",
        "energy_score",
        "variogram_score",
        "corr_mae",
        "ramp_quantile_mae",
    ]
    for (track, variant), group_rows in sorted(groups.items()):
        item = {
            "track": track,
            "variant": variant,
            "n_seeds": len(group_rows),
            "seeds": " ".join(str(row["seed"]) for row in sorted(group_rows, key=lambda row: row["seed"])),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in group_rows], dtype=np.float64)
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def write_raw(rows):
    path = OUTPUT_DIR / "qbm_seed_metrics_raw.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "track",
            "variant",
            "seed",
            "mean_qs",
            "mean_crps_percent",
            "reliability_mae",
            "energy_score",
            "variogram_score",
            "corr_mae",
            "ramp_quantile_mae",
            "scenario_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows):
    path = OUTPUT_DIR / "qbm_seed_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows):
    selected = [
        row for row in rows
        if row["variant"] in {
            "gibbs_cal_trc",
            "gibbs_calm_trc",
            "gibbs_calx_trc",
            "pv64_gibbs_cal",
            "pv64_gibbs_cal_trc",
            "pv64_gibbs_calm",
            "pv64_gibbs_calm_trc",
        }
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-seed stability of selected QBM-VAE variants. Values are mean $\pm$ standard deviation.}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Track & Variant & Seeds & QS & CRPS (\%) & Rel. MAE \\",
        r"\hline",
    ]
    for row in selected:
        lines.append(
            "%s & %s & %s & %s & %s & %s \\\\" % (
                pretty_track(row["track"]),
                row["variant"],
                row["n_seeds"],
                fmt_pm(row["mean_qs_mean"], row["mean_qs_std"]),
                fmt_pm(row["mean_crps_percent_mean"], row["mean_crps_percent_std"]),
                fmt_pm(row["reliability_mae_mean"], row["reliability_mae_std"]),
            )
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUTPUT_DIR / "qbm_seed_summary_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def fmt_pm(mean, std):
    return "%.4f $\\pm$ %.4f" % (mean, std)


def pretty_track(track):
    return {"wind": "Wind", "pv": "PV", "load": "Load"}[track]


if __name__ == "__main__":
    main()
