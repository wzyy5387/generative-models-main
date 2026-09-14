# -*- coding: utf-8 -*-

import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from GEFcom2014 import load_data, pv_data, wind_data
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE import temporal_mi_sparse_graph


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "GEFcom2014" / "data"
OUTPUT_DIR = ROOT_DIR / "export" / "qbm_explainability"

DEFAULT_MODEL_BY_TRACK = {
    "load": "load_QBMVAE_2_sa_0",
    "pv": "pv_QBMVAE_2_pv64_sa_0",
    "wind": "wind_QBMVAE_2_sa_0",
}


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, _ = load_track(args.tag)
    x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, _ = scale_data_multi(
        x_LS=data[0].values,
        y_LS=data[1].values,
        x_VS=data[2].values,
        y_VS=data[3].values,
        x_TEST=data[4].values,
        y_TEST=data[5].values,
    )

    model_name = args.model_name or DEFAULT_MODEL_BY_TRACK[args.tag]
    model_path = ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag) / (model_name + ".pickle")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    model.device = device
    model.eval()

    x_explain = x_TEST_scaled[: min(args.max_instances, len(x_TEST_scaled))]
    with torch.no_grad():
        context = torch.tensor(x_explain, dtype=torch.float32, device=device)
        h_tensor, j_tensor = model.qbm_params(context if model.conditional_qbm else None)
        if h_tensor.ndim == 1:
            h_tensor = h_tensor[None, :].expand(len(x_explain), -1)
        h_all = h_tensor.detach().cpu().numpy()
        j = j_tensor.detach().cpu().numpy()
    h_first = h_all[0]
    h_mean = h_all.mean(axis=0)
    h_std = h_all.std(axis=0)
    graph_mask = model.graph_mask.detach().cpu().numpy()
    uses_path_integral = bool(
        hasattr(model, "uses_path_integral") and model.uses_path_integral()
    )
    gamma = (
        model.transverse_gamma().detach().cpu().numpy()
        if uses_path_integral
        else np.zeros(model.latent_s, dtype=np.float64)
    )
    longitudinal_strength = np.abs(h_all).mean(axis=0) + np.abs(j).sum(axis=1)
    quantum_ratio = gamma / np.maximum(longitudinal_strength, 1e-12)
    quantum_fraction = gamma / np.maximum(gamma + longitudinal_strength, 1e-12)
    replica_disagreement, transverse_magnetization, action_summary = path_integral_diagnostics(
        model, context, h_tensor, j_tensor, args.beta, args.sampling_steps
    )

    prefix = "%s_%s" % (args.tag, model_name)
    plot_heatmap(np.abs(j), OUTPUT_DIR / (prefix + "_abs_J.pdf"), "|J|", "Latent bit", "Latent bit")
    plot_heatmap(graph_mask, OUTPUT_DIR / (prefix + "_graph_mask.pdf"), "Ising graph mask", "Latent bit", "Latent bit")
    plot_bar(h_first, OUTPUT_DIR / (prefix + "_h_first_test.pdf"), "Conditional local field h(x)", "Latent bit", "h")
    plot_bar(h_mean, OUTPUT_DIR / (prefix + "_h_mean_test.pdf"), "Mean conditional local field on TEST", "Latent bit", "h")
    plot_bar(h_std, OUTPUT_DIR / (prefix + "_h_std_test.pdf"), "Conditional field variation on TEST", "Latent bit", "std(h)")

    bit_rows = []
    n_periods = y_LS_scaled.shape[1]
    latent_per_period = model.latent_s // n_periods if model.latent_s % n_periods == 0 else 0
    for bit in range(model.latent_s):
        bit_rows.append({
            "latent_bit": bit,
            "period": bit // latent_per_period if latent_per_period else -1,
            "h_mean": float(h_mean[bit]),
            "h_std": float(h_std[bit]),
            "coupling_l1": float(np.abs(j[bit]).sum()),
            "transverse_gamma": float(gamma[bit]),
            "quantum_to_longitudinal_ratio": float(quantum_ratio[bit]),
            "quantum_fraction": float(quantum_fraction[bit]),
            "replica_disagreement": float(replica_disagreement[bit]),
            "transverse_magnetization": float(transverse_magnetization[bit]),
        })
    write_csv(OUTPUT_DIR / (prefix + "_latent_metrics.csv"), bit_rows)

    edge_rows = build_edge_rows(j, latent_per_period)
    write_csv(OUTPUT_DIR / (prefix + "_ising_edges.csv"), edge_rows)

    sensitivity = conditional_field_sensitivity(model)
    np.save(OUTPUT_DIR / (prefix + "_conditional_field_sensitivity.npy"), sensitivity)
    if sensitivity.size:
        plot_heatmap(
            sensitivity,
            OUTPUT_DIR / (prefix + "_conditional_field_sensitivity.pdf"),
            "|dh/dx|: weather-to-latent field sensitivity",
            "Condition feature",
            "Latent bit",
        )

    if uses_path_integral:
        plot_bar(gamma, OUTPUT_DIR / (prefix + "_transverse_gamma.pdf"), "Transverse field", "Latent bit", "Gamma")
        plot_bar(
            quantum_fraction,
            OUTPUT_DIR / (prefix + "_quantum_fraction.pdf"),
            "Transverse contribution relative to longitudinal strength",
            "Latent bit",
            "Gamma / (Gamma + |h| + sum|J|)",
        )
        plot_bar(
            replica_disagreement,
            OUTPUT_DIR / (prefix + "_replica_disagreement.pdf"),
            "Imaginary-time replica disagreement",
            "Latent bit",
            "Disagreement probability",
        )
        plot_bar(
            transverse_magnetization,
            OUTPUT_DIR / (prefix + "_transverse_magnetization.pdf"),
            "Trotter estimator of transverse magnetization",
            "Latent bit",
            "<sigma_x>",
        )

    if latent_per_period:
        period_rows = aggregate_period_metrics(bit_rows, n_periods)
        write_csv(OUTPUT_DIR / (prefix + "_period_metrics.csv"), period_rows)

    if latent_per_period:
        mi_graph = temporal_mi_sparse_graph(y_LS_scaled, latent_per_period=latent_per_period, top_k=args.top_k)
        plot_heatmap(mi_graph, OUTPUT_DIR / (prefix + "_temporal_mi_graph.pdf"), "Temporal-MI sparse graph",
                     "Latent bit", "Latent bit")

    stability = seed_stability(model_path, model_name, model.latent_s, device, args.max_seed_models)
    if stability is not None:
        np.save(OUTPUT_DIR / (prefix + "_J_seed_std.npy"), stability["std"])
        plot_heatmap(
            stability["std"],
            OUTPUT_DIR / (prefix + "_J_seed_std.pdf"),
            "Coupling uncertainty across seeds",
            "Latent bit",
            "Latent bit",
        )

    summary = {
        "track": args.tag,
        "model": model_name,
        "prior_representation": "finite_trotter_tfim" if uses_path_integral else "classical_ising",
        "transverse_field_enabled": uses_path_integral,
        "beta": float(args.beta),
        "trotter_replicas": int(getattr(model, "trotter_replicas", 1) if uses_path_integral else 1),
        "mean_transverse_gamma": float(gamma.mean()),
        "mean_quantum_fraction": float(quantum_fraction.mean()),
        "mean_replica_disagreement": float(replica_disagreement.mean()),
        "mean_transverse_magnetization": float(transverse_magnetization.mean()),
        "action_decomposition": action_summary,
        "seed_models_found": 0 if stability is None else stability["n_models"],
        "claim_scope": (
            "Finite-replica Suzuki-Trotter approximation to a transverse-field quantum Gibbs prior; "
            "not native quantum Gibbs sampling."
            if uses_path_integral else
            "Classical Ising/Boltzmann prior without non-commuting terms."
        ),
    }
    (OUTPUT_DIR / (prefix + "_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Explainability plots written to %s" % OUTPUT_DIR)


def path_integral_diagnostics(model, context, h_tensor, j_tensor, beta, sampling_steps):
    if not (hasattr(model, "uses_path_integral") and model.uses_path_integral()):
        return np.zeros(model.latent_s), np.zeros(model.latent_s), {}
    with torch.no_grad():
        replicas_binary = model.sample_path_integral_negative(
            context, steps=sampling_steps, beta=beta, persistent=False
        )
        spins = 2.0 * replicas_binary - 1.0
        n_replicas = spins.shape[1]
        disagreement = 0.5 * (
            1.0 - (spins * torch.roll(spins, shifts=-1, dims=1)).mean(dim=(0, 1))
        )
        field = -(float(beta) / n_replicas) * (spins * h_tensor[:, None, :]).sum(dim=(1, 2))
        pair = -(0.5 * float(beta) / n_replicas) * torch.einsum(
            "bmi,ij,bmj->b", spins, j_tensor, spins
        )
        gamma = model.transverse_gamma()
        coupling = -0.5 * torch.log(
            torch.tanh(torch.clamp(float(beta) * gamma / n_replicas, min=1e-6)).clamp_min(1e-12)
        )
        transverse = -(
            spins * torch.roll(spins, shifts=-1, dims=1) * coupling[None, None, :]
        ).sum(dim=(1, 2))
        total = field + pair + transverse
        aligned_baseline = -n_replicas * coupling.sum()
        fluctuation_penalty = transverse - aligned_baseline
        adjacent_correlation = 1.0 - 2.0 * disagreement.double()
        twice_argument = 2.0 * float(beta) * gamma.double() / n_replicas
        transverse_magnetization = (
            torch.cosh(twice_argument) - adjacent_correlation
        ) / torch.sinh(twice_argument).clamp_min(1e-12)
        transverse_magnetization = transverse_magnetization.clamp(0.0, 1.0)
    return disagreement.cpu().numpy(), transverse_magnetization.cpu().numpy(), {
        "longitudinal_field_mean": float(field.mean().cpu()),
        "longitudinal_pair_mean": float(pair.mean().cpu()),
        "imaginary_time_mean": float(transverse.mean().cpu()),
        "imaginary_time_aligned_baseline": float(aligned_baseline.cpu()),
        "quantum_fluctuation_penalty_mean": float(fluctuation_penalty.mean().cpu()),
        "total_action_mean": float(total.mean().cpu()),
        "baseline_shifted_action_mean": float((field + pair + fluctuation_penalty).mean().cpu()),
    }


def conditional_field_sensitivity(model):
    layer = getattr(model, "cond_to_h", None)
    if layer is None:
        return np.empty((0, 0))
    return np.abs(layer.weight[:, : model.cond_in].detach().cpu().numpy())


def build_edge_rows(j, latent_per_period):
    rows = []
    for left, right in zip(*np.where(np.triu(np.abs(j) > 1e-12, k=1))):
        rows.append({
            "left_bit": int(left),
            "right_bit": int(right),
            "left_period": int(left // latent_per_period) if latent_per_period else -1,
            "right_period": int(right // latent_per_period) if latent_per_period else -1,
            "J": float(j[left, right]),
            "abs_J": float(abs(j[left, right])),
            "effect": "co-activation" if j[left, right] > 0 else "anti-activation",
        })
    return sorted(rows, key=lambda row: row["abs_J"], reverse=True)


def aggregate_period_metrics(bit_rows, n_periods):
    rows = []
    numeric_keys = [
        "h_mean", "h_std", "coupling_l1", "transverse_gamma",
        "quantum_to_longitudinal_ratio", "quantum_fraction", "replica_disagreement",
        "transverse_magnetization",
    ]
    for period in range(n_periods):
        selected = [row for row in bit_rows if row["period"] == period]
        row = {"period": period, "n_latent_bits": len(selected)}
        for key in numeric_keys:
            row[key] = float(np.mean([item[key] for item in selected]))
        rows.append(row)
    return rows


def seed_stability(model_path, model_name, latent_s, device, max_models):
    match = re.match(r"^(.*_)\d+$", model_name)
    if match is None:
        return None
    candidates = sorted(model_path.parent.glob(match.group(1) + "*.pickle"))[:max_models]
    couplings = []
    for candidate in candidates:
        try:
            with candidate.open("rb") as handle:
                candidate_model = pickle.load(handle)
            if candidate_model.latent_s != latent_s:
                continue
            candidate_model.to(device)
            candidate_model.device = device
            _, candidate_j = candidate_model.export_ising()
            couplings.append(candidate_j)
        except (AttributeError, RuntimeError, ValueError):
            continue
    if len(couplings) < 2:
        return None
    stack = np.stack(couplings)
    return {"n_models": len(couplings), "std": stack.std(axis=0)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_track(tag):
    if tag == "load":
        return load_data(DATA_DIR / "load_data_track1.csv", test_size=50, random_state=0), []
    if tag == "pv":
        data, indices = pv_data(DATA_DIR / "solar_new.csv", test_size=50, random_state=0)
        return data, indices
    if tag == "wind":
        return wind_data(DATA_DIR / "wind_data_all_zone.csv", test_size=50, random_state=0), []
    raise ValueError("Unknown tag: %s" % tag)


def plot_heatmap(matrix, path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bar(values, path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(np.arange(len(values)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot QBM-VAE Ising prior explainability figures.")
    parser.add_argument("--tag", default="load", choices=["load", "pv", "wind"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-instances", type=int, default=50)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--max-seed-models", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
