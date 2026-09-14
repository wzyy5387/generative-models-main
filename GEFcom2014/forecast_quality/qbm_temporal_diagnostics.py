# -*- coding: utf-8 -*-

"""Export interpretable Temporal-AR and Ising summaries from QBM-VAE."""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR
from GEFcom2014.forecast_quality.temporal_rank_coupling import load_raw_track
from GEFcom2014.models import scale_data_multi


def main():
    args = parse_args()
    model_dir = args.model_dir or ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag)
    output_dir = args.output_dir or ROOT_DIR / "export" / ("qbm_ecc_%s" % args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, _ = load_raw_track(args.tag)
    scaled = scale_data_multi(
        data[0].values, data[1].values,
        data[2].values, data[3].values,
        data[4].values, data[5].values,
    )
    x_vs = torch.tensor(scaled[2], dtype=torch.float32)
    y_vs = torch.tensor(scaled[3], dtype=torch.float32)

    rho_rows = []
    edge_rows = []
    field_rows = []
    for seed in args.seeds:
        model_path = model_dir / (
            "%s_QBMVAE_2_%s_sa_%d.pickle" % (args.tag, args.model_label, seed)
        )
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        model.to(torch.device("cpu"))
        model.device = torch.device("cpu")
        model.eval()
        with torch.no_grad():
            logits = model.encode_logits(y_vs, x_vs)
            latent = (torch.sigmoid(logits) >= 0.5).float()
            _, _, rho = model.decode_temporal_parameters(latent, x_vs)
            conditional_h, j = model.qbm_params(x_vs)
        rho_rows.extend(summarize_rho(rho.numpy(), seed))
        edge_rows.extend(summarize_edges(j.numpy(), model.in_size, seed))
        field_rows.extend(summarize_fields(conditional_h.numpy(), model.in_size, seed))

    write_csv(output_dir / "temporal_ar_coefficients.csv", rho_rows)
    write_csv(output_dir / "ising_temporal_edges.csv", edge_rows)
    write_csv(output_dir / "conditional_ising_fields.csv", field_rows)
    print("Wrote temporal diagnostics to %s" % output_dir)


def summarize_rho(rho, seed):
    rows = []
    for period in range(rho.shape[1]):
        values = rho[:, period]
        rows.append({
            "seed": seed,
            "hour": period + 1,
            "mean_rho": float(values.mean()),
            "std_rho": float(values.std()),
            "q05_rho": float(np.quantile(values, 0.05)),
            "median_rho": float(np.median(values)),
            "q95_rho": float(np.quantile(values, 0.95)),
            "mean_abs_rho": float(np.abs(values).mean()),
            "near_boundary_fraction": float(np.mean(np.abs(values) > 0.94)),
        })
    return rows


def summarize_edges(j, n_periods, seed):
    latent_s = j.shape[0]
    if latent_s % n_periods:
        raise ValueError("latent size must be divisible by modeled periods")
    latent_per_period = latent_s // n_periods
    rows = []
    for source in range(n_periods):
        source_block = slice(source * latent_per_period, (source + 1) * latent_per_period)
        for target in range(source + 1, n_periods):
            target_block = slice(target * latent_per_period, (target + 1) * latent_per_period)
            block = j[source_block, target_block]
            if not np.any(block):
                continue
            rows.append({
                "seed": seed,
                "source_hour": source + 1,
                "target_hour": target + 1,
                "lag_hours": target - source,
                "mean_coupling": float(block.mean()),
                "mean_abs_coupling": float(np.abs(block).mean()),
                "max_abs_coupling": float(np.abs(block).max()),
                "active_latent_edges": int(np.count_nonzero(block)),
            })
    return rows


def summarize_fields(conditional_h, n_periods, seed):
    latent_s = conditional_h.shape[1]
    if latent_s % n_periods:
        raise ValueError("latent size must be divisible by modeled periods")
    latent_per_period = latent_s // n_periods
    rows = []
    for period in range(n_periods):
        block = conditional_h[:, period * latent_per_period:(period + 1) * latent_per_period]
        rows.append({
            "seed": seed,
            "hour": period + 1,
            "mean_field": float(block.mean()),
            "std_field": float(block.std()),
            "mean_abs_field": float(np.abs(block).mean()),
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Export QBM temporal interpretability tables.")
    parser.add_argument("--tag", default="wind", choices=["wind", "pv", "load"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model-label", default="tar1m")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
