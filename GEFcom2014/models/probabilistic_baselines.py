# -*- coding: utf-8 -*-

"""Strong conditional scenario baselines for the GEFCom2014 tracks."""

import argparse
import json
import math
import pickle
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR
from GEFcom2014.forecast_quality.temporal_rank_coupling import load_raw_track
from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models import scale_data_multi


MODEL_LABELS = {
    "spline-nf": "SplineCNF",
    "ddpm": "ConditionalDDPM",
    "residual-bootstrap": "ResidualBootstrap",
}


class ConditionalDenoiser(nn.Module):
    def __init__(self, data_dim, context_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_layer = nn.Linear(data_dim + hidden_dim + time_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.output_layer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, data_dim))

    def forward(self, noisy_values, timesteps, context):
        time_features = sinusoidal_embedding(timesteps, self.time_dim)
        hidden = self.input_layer(
            torch.cat((noisy_values, self.context_net(context), time_features), dim=1)
        )
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output_layer(hidden)


def sinusoidal_embedding(timesteps, dimension):
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def build_spline_flow(data_dim, context_dim, hidden_dim, num_transforms, num_bins):
    transforms = []
    for _ in range(num_transforms):
        transforms.append(RandomPermutation(features=data_dim))
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=data_dim,
                hidden_features=hidden_dim,
                context_features=context_dim,
                num_bins=num_bins,
                tails="linear",
                tail_bound=4.0,
                num_blocks=2,
                use_residual_blocks=True,
                random_mask=False,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
    return Flow(CompositeTransform(transforms), StandardNormal([data_dim]))


def train_spline_flow(model, arrays, args, device):
    loader = make_loader(arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def loss_fn(batch_y, batch_x):
        return -model.log_prob(batch_y, context=batch_x).mean()

    history, best_state = fit_with_validation(
        model, loader, arrays, loss_fn, optimizer, args, device
    )
    return history, best_state


def train_ddpm(model, scheduler, arrays, args, device):
    loader = make_loader(arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def loss_fn(batch_y, batch_x):
        noise = torch.randn_like(batch_y)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch_y.shape[0],), device=device
        ).long()
        noisy = scheduler.add_noise(batch_y, noise, timesteps)
        prediction = model(noisy, timesteps, batch_x)
        if scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(batch_y, noise, timesteps)
        else:
            target = noise
        return torch.nn.functional.mse_loss(prediction, target)

    history, best_state = fit_with_validation(
        model, loader, arrays, loss_fn, optimizer, args, device
    )
    return history, best_state


def fit_with_validation(model, loader, arrays, loss_fn, optimizer, args, device):
    x_vs = torch.as_tensor(arrays["x_vs"], dtype=torch.float32, device=device)
    y_vs = torch.as_tensor(arrays["y_vs"], dtype=torch.float32, device=device)
    history = []
    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch_y, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_total += float(loss.detach()) * batch_y.shape[0]
            train_count += batch_y.shape[0]

        model.eval()
        validation_losses = []
        with torch.no_grad():
            cuda_devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(args.seed + 104729)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + 104729)
                for start in range(0, x_vs.shape[0], args.batch_size):
                    validation_losses.append(
                        float(loss_fn(y_vs[start:start + args.batch_size], x_vs[start:start + args.batch_size]))
                    )
        train_loss = train_total / train_count
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - args.min_delta:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print("Epoch %d | LS %.5f VS %.5f" % (epoch + 1, train_loss, validation_loss))
        if stale_epochs >= args.patience:
            print("Early stopping at epoch %d" % (epoch + 1))
            break
    model.load_state_dict(best_state)
    return history, best_state


def sample_spline_flow(model, context, n_scenarios, device, day_batch_size=32):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, context.shape[0], day_batch_size):
            batch_context = torch.as_tensor(
                context[start:start + day_batch_size], dtype=torch.float32, device=device
            )
            samples = model.sample(n_scenarios, context=batch_context)
            chunks.append(samples.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def sample_ddpm(model, scheduler, context, data_dim, n_scenarios, inference_steps, device,
                day_batch_size=16):
    chunks = []
    model.eval()
    scheduler.set_timesteps(inference_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for start in range(0, context.shape[0], day_batch_size):
            batch_context = torch.as_tensor(
                context[start:start + day_batch_size], dtype=torch.float32, device=device
            )
            expanded_context = batch_context.repeat_interleave(n_scenarios, dim=0)
            values = torch.randn(
                expanded_context.shape[0], data_dim, generator=generator, device=device
            )
            for timestep in scheduler.timesteps:
                time_batch = timestep.expand(values.shape[0])
                prediction = model(values, time_batch, expanded_context)
                values = scheduler.step(prediction, timestep, values, generator=generator).prev_sample
            shape = (batch_context.shape[0], n_scenarios, data_dim)
            chunks.append(values.reshape(shape).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def sample_residual_bootstrap(raw_arrays, split, n_scenarios, seed):
    """Add LS forecast-error days to each supplied day-ahead point forecast."""
    data_dim = raw_arrays["y_ls"].shape[1]
    if raw_arrays["x_ls"].shape[1] < data_dim:
        raise ValueError("Residual bootstrap requires a target-sized forecast context block")
    residual_days = raw_arrays["y_ls"] - raw_arrays["x_ls"][:, :data_dim]
    forecast = raw_arrays["x_" + split][:, :data_dim]
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, residual_days.shape[0], size=(forecast.shape[0], n_scenarios)
    )
    samples = forecast[:, None, :] + residual_days[indices]
    return np.clip(samples, 0.0, 1.0).astype(np.float32)


def prepare_track(tag):
    data, indices = load_raw_track(tag)
    scaled = scale_data_multi(
        x_LS=data[0].values,
        y_LS=data[1].values,
        x_VS=data[2].values,
        y_VS=data[3].values,
        x_TEST=data[4].values,
        y_TEST=data[5].values,
    )
    names = ("x_ls", "y_ls", "x_vs", "y_vs", "x_test", "y_test")
    arrays = {name: value.astype(np.float32) for name, value in zip(names, scaled[:6])}
    return arrays, scaled[6], np.asarray(indices, dtype=int)


def prepare_dataset(tag, dataset_bundle=None):
    if dataset_bundle is None:
        return prepare_track(tag)
    raw, metadata, _ = load_daily_bundle(dataset_bundle)
    scaled = scale_data_multi(
        x_LS=raw["x_ls"], y_LS=raw["y_ls"],
        x_VS=raw["x_vs"], y_VS=raw["y_vs"],
        x_TEST=raw["x_test"], y_TEST=raw["y_test"],
    )
    names = ("x_ls", "y_ls", "x_vs", "y_vs", "x_test", "y_test")
    arrays = {name: value.astype(np.float32) for name, value in zip(names, scaled[:6])}
    expected_name = metadata.get("dataset_name")
    if expected_name and expected_name != tag:
        raise ValueError("Bundle dataset_name %s does not match --tag %s" % (expected_name, tag))
    return arrays, scaled[6], np.asarray([], dtype=int)


def scenarios_to_period_matrix(samples, scaler, tag, zero_indices):
    n_days, n_scenarios, data_dim = samples.shape
    flat = scaler.inverse_transform(samples.reshape(-1, data_dim)).reshape(
        n_days, n_scenarios, data_dim
    )
    flat = np.clip(flat, 0.0, 1.0)
    if tag == "pv":
        rebuilt = np.zeros((n_days, n_scenarios, 24), dtype=np.float64)
        non_zero = np.setdiff1d(np.arange(24), zero_indices)
        rebuilt[:, :, non_zero] = flat
        flat = rebuilt
    return flat.transpose(0, 2, 1).reshape(n_days * 24, n_scenarios)


def make_loader(x, y, batch_size, seed):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def dump_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train strong conditional scenario baselines.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["spline-nf", "ddpm", "residual-bootstrap"],
    )
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Using device: %s" % device)
    if args.model == "residual-bootstrap":
        if args.dataset_bundle is None:
            raise ValueError("residual-bootstrap requires --dataset-bundle")
        raw_arrays, metadata, _ = load_daily_bundle(args.dataset_bundle)
        expected_name = metadata.get("dataset_name")
        if expected_name and expected_name != args.tag:
            raise ValueError(
                "Bundle dataset_name %s does not match --tag %s" % (expected_name, args.tag)
            )
        output_dir = args.output_dir or ROOT_DIR / "export" / (
            "residual_bootstrap_%s" % args.tag
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        label = MODEL_LABELS[args.model]
        model_name = "%s_%s_%s" % (args.tag, label, args.seed)
        samples_by_split = {
            "VS": sample_residual_bootstrap(
                raw_arrays, "vs", args.n_scenarios, args.seed + 15485863
            ),
            "TEST": sample_residual_bootstrap(
                raw_arrays, "test", args.n_scenarios, args.seed + 32452843
            ),
        }
        with (output_dir / (model_name + ".json")).open("w", encoding="utf-8") as handle:
            arguments = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            json.dump(
                {
                    **arguments,
                    "fit_split": "LS",
                    "method": "unconditional_daily_forecast_error_bootstrap",
                    "output_dir": str(output_dir),
                },
                handle,
                indent=2,
            )
        for split, samples in samples_by_split.items():
            scenarios = samples.transpose(0, 2, 1).reshape(-1, args.n_scenarios)
            path = output_dir / (
                "scenarios_%s_%s_%s_%s_%s.pickle"
                % (args.tag, label, args.seed, args.n_scenarios, split)
            )
            dump_pickle(path, scenarios)
            print("Wrote %s" % path)
        return

    arrays, target_scaler, zero_indices = prepare_dataset(args.tag, args.dataset_bundle)
    data_dim = arrays["y_ls"].shape[1]
    context_dim = arrays["x_ls"].shape[1]
    label = MODEL_LABELS[args.model]
    output_dir = args.output_dir or ROOT_DIR / "export" / ("%s_%s" % (args.model.replace("-", "_"), args.tag))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "spline-nf":
        model = build_spline_flow(
            data_dim, context_dim, args.hidden_dim, args.num_transforms, args.num_bins
        ).to(device)
        history, _ = train_spline_flow(model, arrays, args, device)
        validation_samples = sample_spline_flow(model, arrays["x_vs"], args.n_scenarios, device)
        test_samples = sample_spline_flow(model, arrays["x_test"], args.n_scenarios, device)
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            clip_sample=False,
        )
        model = ConditionalDenoiser(data_dim, context_dim, args.hidden_dim).to(device)
        history, _ = train_ddpm(model, scheduler, arrays, args, device)
        validation_samples = sample_ddpm(
            model, scheduler, arrays["x_vs"], data_dim, args.n_scenarios,
            args.inference_steps, device,
        )
        test_samples = sample_ddpm(
            model, scheduler, arrays["x_test"], data_dim, args.n_scenarios,
            args.inference_steps, device,
        )

    model_name = "%s_%s_%s" % (args.tag, label, args.seed)
    torch.save(model.state_dict(), output_dir / (model_name + ".pt"))
    with (output_dir / (model_name + ".json")).open("w", encoding="utf-8") as handle:
        arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        json.dump({**arguments, "output_dir": str(output_dir), "history": history}, handle, indent=2)
    with (output_dir / (model_name + "_scaler.pickle")).open("wb") as handle:
        pickle.dump(target_scaler, handle)

    for split, samples in (("VS", validation_samples), ("TEST", test_samples)):
        scenarios = scenarios_to_period_matrix(samples, target_scaler, args.tag, zero_indices)
        path = output_dir / (
            "scenarios_%s_%s_%s_%s_%s.pickle" % (
                args.tag, label, args.seed, args.n_scenarios, split
            )
        )
        dump_pickle(path, scenarios)
        print("Wrote %s" % path)


if __name__ == "__main__":
    main()
