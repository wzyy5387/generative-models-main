# -*- coding: utf-8 -*-

"""Forecast-anchored conditional spline-flow residual ablation."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR
from GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline import (
    load_anchors,
    prepare_dataset,
)
from GEFcom2014.models.probabilistic_baselines import (
    build_spline_flow,
    sample_spline_flow,
    scenarios_to_period_matrix,
    set_seed,
    train_spline_flow,
)


def build_residual_arrays(arrays, anchors):
    residual_arrays = {}
    for split in ("ls", "vs", "test"):
        residual_arrays["x_" + split] = np.concatenate(
            (arrays["x_" + split], anchors[split]), axis=1
        ).astype(np.float32)
        residual_arrays["y_" + split] = (
            arrays["y_" + split] - anchors[split]
        ).astype(np.float32)
    return residual_arrays


def sample_anchor_spline_flow(
    model,
    context,
    anchor,
    n_scenarios,
    device,
    seed,
    day_batch_size=32,
):
    conditional_context = np.concatenate((context, anchor), axis=1).astype(np.float32)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        residuals = sample_spline_flow(
            model,
            conditional_context,
            n_scenarios,
            device,
            day_batch_size=day_batch_size,
        )
    return residuals + anchor[:, None, :]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--anchor-source", choices=["learned", "context"], default="learned")
    parser.add_argument("--anchor-dir", type=Path, default=None)
    parser.add_argument("--anchor-label", default="lanchor")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Using device: %s" % device)

    raw, arrays, target_scaler, zero_indices = prepare_dataset(
        args.tag, args.dataset_bundle
    )
    anchors, anchor_checkpoint = load_anchors(args, raw, arrays, device)
    residual_arrays = build_residual_arrays(arrays, anchors)
    data_dim = arrays["y_ls"].shape[1]
    context_dim = residual_arrays["x_ls"].shape[1]
    model = build_spline_flow(
        data_dim,
        context_dim,
        args.hidden_dim,
        args.num_transforms,
        args.num_bins,
    ).to(device)
    history, _ = train_spline_flow(model, residual_arrays, args, device)

    output_dir = args.output_dir or ROOT_DIR / "export" / (
        "anchor_spline_flow_%s" % args.tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "%s_AnchorSplineFlow_%d" % (args.tag, args.seed)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "data_dim": data_dim,
            "context_dim": context_dim,
            "hidden_dim": args.hidden_dim,
            "num_transforms": args.num_transforms,
            "num_bins": args.num_bins,
        },
        output_dir / (model_name + ".pt"),
    )
    audit = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": "forecast_anchored_conditional_rational_quadratic_spline_flow",
        "anchor_checkpoint": str(anchor_checkpoint) if anchor_checkpoint else None,
        "residual_definition": "scaled_target_minus_scaled_forecast_anchor",
        "fit_split": "LS",
        "selection_split": "VS",
        "test_used_for_selection": False,
        "history": history,
    }
    with (output_dir / (model_name + ".json")).open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    for split, offset in (("VS", 104729), ("TEST", 130363)):
        split_key = split.lower()
        samples = sample_anchor_spline_flow(
            model,
            arrays["x_" + split_key],
            anchors[split_key],
            args.n_scenarios,
            device,
            seed=args.seed + offset,
        )
        scenarios = scenarios_to_period_matrix(
            samples, target_scaler, args.tag, zero_indices
        )
        path = output_dir / (
            "scenarios_%s_AnchorSplineFlow_%d_%d_%s.pickle"
            % (args.tag, args.seed, args.n_scenarios, split)
        )
        with path.open("wb") as handle:
            pickle.dump(scenarios, handle)
        print("Wrote %s" % path)


if __name__ == "__main__":
    main()
