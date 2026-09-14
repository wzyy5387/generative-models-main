# -*- coding: utf-8 -*-

"""Forecast-anchored conditional Gaussian residual ablation."""

import argparse
import json
import pickle
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR
from GEFcom2014.forecast_quality.temporal_rank_coupling import load_raw_track
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE.forecast_anchor import (
    DeterministicForecastAnchor,
    predict_deterministic_anchor,
)
from GEFcom2014.models.probabilistic_baselines import scenarios_to_period_matrix


class ForecastAnchoredGaussian(nn.Module):
    def __init__(
        self,
        context_dim,
        target_dim,
        hidden_dim=256,
        layers=2,
        min_log_scale=-5.0,
        max_log_scale=1.0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.target_dim = int(target_dim)
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        widths = [context_dim + target_dim] + [hidden_dim] * layers + [2 * target_dim]
        modules = []
        for index, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
            modules.append(nn.Linear(in_features, out_features))
            if index < len(widths) - 2:
                modules.append(nn.SiLU())
        self.network = nn.Sequential(*modules)
        with torch.no_grad():
            self.network[-1].weight[:target_dim].zero_()
            self.network[-1].bias[:target_dim].zero_()

    def distribution_parameters(self, context, anchor):
        output = self.network(torch.cat((context, anchor), dim=1))
        correction, log_scale = torch.split(output, self.target_dim, dim=1)
        return anchor + correction, torch.clamp(
            log_scale, self.min_log_scale, self.max_log_scale
        )

    def nll(self, targets, context, anchor):
        mean, log_scale = self.distribution_parameters(context, anchor)
        inverse_variance = torch.exp(-2.0 * log_scale)
        return (0.5 * (targets - mean) ** 2 * inverse_variance + log_scale).sum(dim=1).mean()


class ForecastAnchoredGaussianMixture(nn.Module):
    """Conditional mixture over complete daily residual trajectories."""

    def __init__(
        self,
        context_dim,
        target_dim,
        components=4,
        hidden_dim=256,
        layers=2,
        min_log_scale=-5.0,
        max_log_scale=1.0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        if components < 2:
            raise ValueError("Gaussian mixture requires at least two components")
        self.target_dim = int(target_dim)
        self.components = int(components)
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        output_dim = self.components * (1 + 2 * self.target_dim)
        widths = [context_dim + target_dim] + [hidden_dim] * layers + [output_dim]
        modules = []
        for index, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
            modules.append(nn.Linear(in_features, out_features))
            if index < len(widths) - 2:
                modules.append(nn.SiLU())
        self.network = nn.Sequential(*modules)

        correction_start = self.components
        correction_stop = correction_start + self.components * self.target_dim
        with torch.no_grad():
            self.network[-1].weight[correction_start:correction_stop].zero_()
            self.network[-1].bias[correction_start:correction_stop].zero_()

    def distribution_parameters(self, context, anchor):
        batch_size = context.shape[0]
        output = self.network(torch.cat((context, anchor), dim=1))
        logits = output[:, :self.components]
        correction_start = self.components
        correction_stop = correction_start + self.components * self.target_dim
        corrections = output[:, correction_start:correction_stop].reshape(
            batch_size, self.components, self.target_dim
        )
        log_scale = output[:, correction_stop:].reshape(
            batch_size, self.components, self.target_dim
        )
        means = anchor[:, None, :] + corrections
        return logits, means, torch.clamp(
            log_scale, self.min_log_scale, self.max_log_scale
        )

    def nll(self, targets, context, anchor):
        logits, means, log_scale = self.distribution_parameters(context, anchor)
        centered = targets[:, None, :] - means
        component_log_prob = (
            -0.5 * centered ** 2 * torch.exp(-2.0 * log_scale) - log_scale
        ).sum(dim=2)
        log_prob = torch.logsumexp(
            torch.log_softmax(logits, dim=1) + component_log_prob,
            dim=1,
        )
        return -log_prob.mean()


def fit_anchor_gaussian(model, arrays, anchors, args, device):
    dataset = TensorDataset(
        torch.as_tensor(arrays["x_ls"], dtype=torch.float32),
        torch.as_tensor(anchors["ls"], dtype=torch.float32),
        torch.as_tensor(arrays["y_ls"], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    x_vs = torch.as_tensor(arrays["x_vs"], dtype=torch.float32, device=device)
    a_vs = torch.as_tensor(anchors["vs"], dtype=torch.float32, device=device)
    y_vs = torch.as_tensor(arrays["y_vs"], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_x, batch_anchor, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_anchor = batch_anchor.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.nll(batch_y, batch_x, batch_anchor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_sum += float(loss.detach()) * batch_x.shape[0]
            train_count += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            validation_loss = float(model.nll(y_vs, x_vs, a_vs))
        train_loss = train_sum / train_count
        history.append(
            {
                "epoch": int(epoch),
                "train_nll": float(train_loss),
                "validation_nll": validation_loss,
            }
        )
        if validation_loss < best_validation - args.min_delta:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                "Epoch %d | LS NLL %.6f VS NLL %.6f"
                % (epoch + 1, train_loss, validation_loss)
            )
        if stale_epochs >= args.patience:
            print("Early stopping at epoch %d" % (epoch + 1))
            break
    if best_state is None:
        raise RuntimeError("Conditional Gaussian did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return history


def sample_anchor_gaussian(model, context, anchor, n_scenarios, device, seed, batch_size=128):
    chunks = []
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for start in range(0, context.shape[0], batch_size):
            batch_x = torch.as_tensor(
                context[start:start + batch_size], dtype=torch.float32, device=device
            )
            batch_anchor = torch.as_tensor(
                anchor[start:start + batch_size], dtype=torch.float32, device=device
            )
            parameters = model.distribution_parameters(batch_x, batch_anchor)
            if isinstance(model, ForecastAnchoredGaussianMixture):
                logits, means, log_scale = parameters
                component = torch.multinomial(
                    torch.softmax(logits, dim=1),
                    n_scenarios,
                    replacement=True,
                    generator=generator,
                )
                gather_index = component[:, :, None].expand(
                    -1, -1, model.target_dim
                )
                selected_mean = torch.gather(means, 1, gather_index)
                selected_log_scale = torch.gather(log_scale, 1, gather_index)
                noise = torch.randn(
                    selected_mean.shape,
                    generator=generator,
                    device=device,
                )
                samples = selected_mean + torch.exp(selected_log_scale) * noise
            else:
                mean, log_scale = parameters
                noise = torch.randn(
                    mean.shape[0], n_scenarios, mean.shape[1],
                    generator=generator,
                    device=device,
                )
                samples = mean[:, None, :] + torch.exp(log_scale)[:, None, :] * noise
            chunks.append(samples.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def prepare_dataset(tag, dataset_bundle=None):
    if dataset_bundle is None:
        data, indices = load_raw_track(tag)
        raw = {
            "x_ls": data[0].values,
            "y_ls": data[1].values,
            "x_vs": data[2].values,
            "y_vs": data[3].values,
            "x_test": data[4].values,
            "y_test": data[5].values,
        }
    else:
        raw, metadata, _ = load_daily_bundle(dataset_bundle)
        expected_name = metadata.get("dataset_name")
        if expected_name and expected_name != tag:
            raise ValueError(
                "Bundle dataset_name %s does not match --tag %s" % (expected_name, tag)
            )
        indices = []
    scaled = scale_data_multi(
        x_LS=raw["x_ls"], y_LS=raw["y_ls"],
        x_VS=raw["x_vs"], y_VS=raw["y_vs"],
        x_TEST=raw["x_test"], y_TEST=raw["y_test"],
    )
    names = ("x_ls", "y_ls", "x_vs", "y_vs", "x_test", "y_test")
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in zip(names, scaled[:6])}
    return raw, arrays, scaled[6], np.asarray(indices, dtype=int)


def load_anchors(args, raw, arrays, device):
    target_dim = arrays["y_ls"].shape[1]
    if args.anchor_source == "context":
        if raw["x_ls"].shape[1] < target_dim:
            raise ValueError("Context anchor requires a target-sized leading forecast block")
        target_scaler = scale_data_multi(
            raw["x_ls"], raw["y_ls"], raw["x_vs"], raw["y_vs"],
            raw["x_test"], raw["y_test"],
        )[6]
        return {
            split: target_scaler.transform(raw["x_" + split][:, :target_dim]).astype(np.float32)
            for split in ("ls", "vs", "test")
        }, None

    anchor_dir = args.anchor_dir or ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag)
    checkpoint_path = anchor_dir / (
        "%s_QBMVAE_2_%s_sa_%d_anchor.pt" % (args.tag, args.anchor_label, args.seed)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    anchor_model = DeterministicForecastAnchor(
        checkpoint["context_dim"],
        checkpoint["target_dim"],
        checkpoint["hidden_dim"],
        checkpoint["layers"],
    ).to(device)
    anchor_model.load_state_dict(checkpoint["state_dict"])
    anchors = {
        split: predict_deterministic_anchor(anchor_model, arrays["x_" + split], device)
        for split in ("ls", "vs", "test")
    }
    return anchors, checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--anchor-source", choices=["learned", "context"], default="learned")
    parser.add_argument("--anchor-dir", type=Path, default=None)
    parser.add_argument("--anchor-label", default="lanchor")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu"
    )
    print("Using device: %s" % device)
    raw, arrays, target_scaler, zero_indices = prepare_dataset(
        args.tag, args.dataset_bundle
    )
    anchors, anchor_checkpoint = load_anchors(args, raw, arrays, device)
    model_class = (
        ForecastAnchoredGaussian
        if args.components == 1
        else ForecastAnchoredGaussianMixture
    )
    model_kwargs = {
        "context_dim": arrays["x_ls"].shape[1],
        "target_dim": arrays["y_ls"].shape[1],
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
    }
    if args.components != 1:
        model_kwargs["components"] = args.components
    model = model_class(**model_kwargs).to(device)
    history = fit_anchor_gaussian(model, arrays, anchors, args, device)

    model_label = (
        "AnchorGaussian" if args.components == 1 else "AnchorGMM%d" % args.components
    )
    default_directory = (
        "anchor_gaussian_%s" % args.tag
        if args.components == 1
        else "anchor_gmm%d_%s" % (args.components, args.tag)
    )
    output_dir = args.output_dir or ROOT_DIR / "export" / default_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "%s_%s_%d" % (args.tag, model_label, args.seed)
    torch.save(model.state_dict(), output_dir / (model_name + ".pt"))
    audit = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": (
            "forecast_anchored_conditional_diagonal_gaussian"
            if args.components == 1
            else "forecast_anchored_conditional_diagonal_gaussian_mixture"
        ),
        "anchor_checkpoint": str(anchor_checkpoint) if anchor_checkpoint else None,
        "fit_split": "LS",
        "selection_split": "VS",
        "test_used_for_selection": False,
        "history": history,
    }
    with (output_dir / (model_name + ".json")).open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    for split, offset in (("VS", 104729), ("TEST", 130363)):
        samples = sample_anchor_gaussian(
            model,
            arrays["x_" + split.lower()],
            anchors[split.lower()],
            args.n_scenarios,
            device,
            seed=args.seed + offset,
        )
        scenarios = scenarios_to_period_matrix(
            samples, target_scaler, args.tag, zero_indices
        )
        path = output_dir / (
            "scenarios_%s_%s_%d_%d_%s.pickle"
            % (args.tag, model_label, args.seed, args.n_scenarios, split)
        )
        with path.open("wb") as handle:
            pickle.dump(scenarios, handle)
        print("Wrote %s" % path)


if __name__ == "__main__":
    main()
