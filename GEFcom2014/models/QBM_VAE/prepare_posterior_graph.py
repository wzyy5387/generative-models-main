# -*- coding: utf-8 -*-

"""Prepare an LS-only conditional posterior graph from a frozen J=0 VAE."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from GEFcom2014 import wind_data
from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models.QBM_VAE import stable_partial_correlation_graph
from GEFcom2014.models.QBM_VAE.forecast_anchor import (
    DeterministicForecastAnchor,
    predict_deterministic_anchor,
)


ROOT_DIR = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ls(track, bundle=None):
    if track == "wind":
        data = wind_data(
            path_name=ROOT_DIR / "GEFcom2014" / "data" / "wind_data_all_zone.csv",
            test_size=50,
            random_state=0,
        )
        return data[0].values, data[1].values
    arrays, metadata, _ = load_daily_bundle(bundle)
    if metadata.get("dataset_name") not in (None, track):
        raise ValueError("Dataset bundle does not match track %s" % track)
    return arrays["x_ls"], arrays["y_ls"]


def load_anchor(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = DeterministicForecastAnchor(
        context_dim=checkpoint["context_dim"],
        target_dim=checkpoint["target_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        layers=checkpoint["layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def connected_components(mask):
    remaining = set(range(mask.shape[0]))
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in np.where(mask[node] != 0)[0]:
                neighbor = int(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def prepare(args):
    x_ls, y_ls = load_ls(args.track, args.dataset_bundle)
    x_scaler = StandardScaler().fit(x_ls)
    y_scaler = StandardScaler().fit(y_ls)
    x_scaled = x_scaler.transform(x_ls).astype(np.float32)
    y_scaled = y_scaler.transform(y_ls).astype(np.float32)

    if args.anchor_checkpoint:
        anchor = load_anchor(args.anchor_checkpoint)
        anchor_values = predict_deterministic_anchor(anchor, x_scaled, torch.device("cpu"))
    else:
        if x_ls.shape[1] < y_ls.shape[1]:
            raise ValueError("Context forecast anchor requires target-sized raw features")
        anchor_values = y_scaler.transform(x_ls[:, :y_ls.shape[1]]).astype(np.float32)
    conditional_input = np.concatenate((x_scaled, anchor_values), axis=1)

    with Path(args.source_model).open("rb") as handle:
        source_model = pickle.load(handle)
    source_model.to(torch.device("cpu"))
    source_model.device = torch.device("cpu")
    source_model.eval()
    with torch.no_grad():
        logits = source_model.encode_logits(
            torch.as_tensor(y_scaled),
            torch.as_tensor(conditional_input),
        ).cpu().numpy()
    if conditional_input.shape[1] != source_model.cond_in:
        raise ValueError("Reconstructed conditional input does not match source model")

    context_scaler = StandardScaler().fit(conditional_input)
    standardized_context = context_scaler.transform(conditional_input)
    context_model = Ridge(alpha=args.context_ridge).fit(standardized_context, logits)
    posterior_residuals = logits - context_model.predict(standardized_context)
    graph = stable_partial_correlation_graph(
        posterior_residuals,
        top_k=args.top_k,
        bootstrap_repetitions=args.bootstrap_repetitions,
        stability_threshold=args.stability_threshold,
        covariance_ridge=args.covariance_ridge,
        seed=args.seed,
    ).astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, graph, allow_pickle=False)
    upper = graph[np.triu_indices(graph.shape[0], k=1)]
    degrees = np.count_nonzero(graph, axis=1)
    components = connected_components(graph)
    metadata = {
        "method": "conditional_posterior_stable_partial_correlation",
        "selection_data": "LS only",
        "track": args.track,
        "source_model": str(Path(args.source_model).resolve()),
        "source_model_sha256": sha256(args.source_model),
        "anchor_checkpoint": str(Path(args.anchor_checkpoint).resolve()) if args.anchor_checkpoint else None,
        "anchor_checkpoint_sha256": sha256(args.anchor_checkpoint) if args.anchor_checkpoint else None,
        "n_ls_samples": int(y_ls.shape[0]),
        "latent_s": int(graph.shape[0]),
        "top_k": args.top_k,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "stability_threshold": args.stability_threshold,
        "context_ridge": args.context_ridge,
        "covariance_ridge": args.covariance_ridge,
        "seed": args.seed,
        "n_edges": int(np.count_nonzero(upper)),
        "density": float(np.count_nonzero(upper) / upper.size),
        "weight_sum": float(upper.sum()),
        "degree_min": int(degrees.min()),
        "degree_mean": float(degrees.mean()),
        "degree_max": int(degrees.max()),
        "n_connected_components": len(components),
        "component_sizes": sorted([len(component) for component in components], reverse=True),
        "graph_sha256": hashlib.sha256(graph.tobytes()).hexdigest(),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["wind", "opsd-wind"], required=True)
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--stability-threshold", type=float, default=0.7)
    parser.add_argument("--context-ridge", type=float, default=10.0)
    parser.add_argument("--covariance-ridge", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
