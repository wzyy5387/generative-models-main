# -*- coding: utf-8 -*-

"""Forecast-anchored conditional EDM score model for daily residual trajectories."""

import argparse
import json
import math
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR
from GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline import (
    load_anchors,
    prepare_dataset,
)
from GEFcom2014.models.QBM_VAE.anchor_spline_flow import build_residual_arrays
from GEFcom2014.models.probabilistic_baselines import (
    scenarios_to_period_matrix,
    set_seed,
    sinusoidal_embedding,
)


class TemporalScoreBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.film = nn.Linear(channels, 2 * channels)
        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv_out = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, values, conditioning):
        hidden = self.norm(values)
        scale, shift = self.film(conditioning).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale[:, :, None]) + shift[:, :, None]
        hidden = torch.nn.functional.silu(hidden)
        hidden = self.conv_dilated(hidden)
        hidden = torch.nn.functional.silu(hidden)
        hidden = self.conv_out(hidden)
        return (values + hidden) / math.sqrt(2.0)


class ConditionalTemporalScoreNetwork(nn.Module):
    def __init__(
        self,
        data_dim,
        context_dim,
        channels=64,
        blocks=4,
        time_dim=64,
    ):
        super().__init__()
        self.data_dim = int(data_dim)
        self.time_dim = int(time_dim)
        self.input_projection = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalScoreBlock(channels, dilation=2 ** (index % 4))
                for index in range(blocks)
            ]
        )
        self.output_norm = nn.GroupNorm(min(8, channels), channels)
        self.output_projection = nn.Conv1d(channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, noisy_values, noise_condition, context):
        time_features = sinusoidal_embedding(noise_condition, self.time_dim)
        conditioning = self.context_projection(context) + self.time_projection(time_features)
        hidden = self.input_projection(noisy_values[:, None, :])
        for block in self.blocks:
            hidden = block(hidden, conditioning)
        hidden = torch.nn.functional.silu(self.output_norm(hidden))
        return self.output_projection(hidden).squeeze(1)


class EDMPreconditionedDenoiser(nn.Module):
    def __init__(self, network, sigma_data=0.5):
        super().__init__()
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        self.network = network
        self.sigma_data = float(sigma_data)

    def denoise(self, noisy_values, sigma, context):
        sigma = sigma.reshape(-1)
        sigma_data = self.sigma_data
        denominator = torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / denominator
        c_in = 1.0 / denominator
        c_noise = torch.log(sigma.clamp_min(1e-8)) / 4.0
        prediction = self.network(
            c_in[:, None] * noisy_values,
            c_noise,
            context,
        )
        return c_skip[:, None] * noisy_values + c_out[:, None] * prediction

    def training_loss(self, target, context, sigma, noise):
        noisy_values = target + sigma[:, None] * noise
        denoised = self.denoise(noisy_values, sigma, context)
        weight = (
            (sigma ** 2 + self.sigma_data ** 2)
            / (sigma * self.sigma_data) ** 2
        )
        return (weight[:, None] * (denoised - target) ** 2).mean()


def sample_training_sigmas(batch_size, args, device):
    sigma = torch.exp(
        args.p_mean
        + args.p_std * torch.randn(batch_size, device=device)
    )
    return sigma.clamp(args.sigma_min, args.sigma_max)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def fit_anchor_score_sde(model, arrays, args, device):
    dataset = TensorDataset(
        torch.as_tensor(arrays["x_ls"], dtype=torch.float32),
        torch.as_tensor(arrays["y_ls"], dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ema_model = deepcopy(model).eval()
    ema_model.requires_grad_(False)
    x_vs = torch.as_tensor(arrays["x_vs"], dtype=torch.float32, device=device)
    y_vs = torch.as_tensor(arrays["y_vs"], dtype=torch.float32, device=device)
    best_state = None
    best_validation = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_context, batch_target in loader:
            batch_context = batch_context.to(device)
            batch_target = batch_target.to(device)
            sigma = sample_training_sigmas(batch_target.shape[0], args, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(
                batch_target,
                batch_context,
                sigma,
                torch.randn_like(batch_target),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)
            train_total += float(loss.detach()) * batch_target.shape[0]
            train_count += batch_target.shape[0]

        validation_losses = []
        with torch.no_grad(), torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ):
            torch.manual_seed(args.seed + 104729)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 104729)
            for start in range(0, x_vs.shape[0], args.batch_size):
                batch_context = x_vs[start:start + args.batch_size]
                batch_target = y_vs[start:start + args.batch_size]
                sigma = sample_training_sigmas(batch_target.shape[0], args, device)
                validation_losses.append(
                    float(
                        ema_model.training_loss(
                            batch_target,
                            batch_context,
                            sigma,
                            torch.randn_like(batch_target),
                        )
                    )
                )
        train_loss = train_total / train_count
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation - args.min_delta:
            best_validation = validation_loss
            best_state = deepcopy(ema_model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                "Epoch %d | LS %.6f VS %.6f"
                % (epoch + 1, train_loss, validation_loss)
            )
        if stale_epochs >= args.patience:
            print("Early stopping at epoch %d" % (epoch + 1))
            break

    if best_state is None:
        raise RuntimeError("Score-SDE training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return history


def edm_noise_schedule(steps, sigma_min, sigma_max, rho, device):
    ramp = torch.linspace(0.0, 1.0, steps, device=device)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return torch.cat((sigmas, torch.zeros(1, device=device)))


def sample_anchor_score_sde(
    model,
    context,
    anchor,
    n_scenarios,
    device,
    seed,
    steps=32,
    sigma_min=0.002,
    sigma_max=1.0,
    rho=7.0,
    day_batch_size=32,
):
    if steps < 2:
        raise ValueError("steps must be at least two")
    conditional_context = np.concatenate((context, anchor), axis=1).astype(np.float32)
    schedule = edm_noise_schedule(
        steps, sigma_min, sigma_max, rho, device
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, context.shape[0], day_batch_size):
            batch_context = torch.as_tensor(
                conditional_context[start:start + day_batch_size],
                dtype=torch.float32,
                device=device,
            )
            batch_anchor = torch.as_tensor(
                anchor[start:start + day_batch_size],
                dtype=torch.float32,
                device=device,
            )
            expanded_context = batch_context.repeat_interleave(n_scenarios, dim=0)
            values = torch.randn(
                expanded_context.shape[0],
                batch_anchor.shape[1],
                generator=generator,
                device=device,
            ) * schedule[0]

            for index in range(schedule.shape[0] - 1):
                sigma = schedule[index]
                next_sigma = schedule[index + 1]
                sigma_batch = sigma.expand(values.shape[0])
                denoised = model.denoise(values, sigma_batch, expanded_context)
                derivative = (values - denoised) / sigma
                euler = values + (next_sigma - sigma) * derivative
                if float(next_sigma) == 0.0:
                    values = euler
                else:
                    next_batch = next_sigma.expand(values.shape[0])
                    next_denoised = model.denoise(
                        euler, next_batch, expanded_context
                    )
                    next_derivative = (euler - next_denoised) / next_sigma
                    values = values + (next_sigma - sigma) * (
                        derivative + next_derivative
                    ) / 2.0

            residuals = values.reshape(
                batch_context.shape[0], n_scenarios, batch_anchor.shape[1]
            )
            chunks.append((residuals + batch_anchor[:, None, :]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--anchor-source", choices=["learned", "context"], default="learned")
    parser.add_argument("--anchor-dir", type=Path, default=None)
    parser.add_argument("--anchor-label", default="lanchor")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--p-mean", type=float, default=-1.2)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sampling-steps", type=int, default=32)
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
    network = ConditionalTemporalScoreNetwork(
        data_dim=data_dim,
        context_dim=context_dim,
        channels=args.channels,
        blocks=args.blocks,
        time_dim=args.time_dim,
    )
    model = EDMPreconditionedDenoiser(
        network, sigma_data=args.sigma_data
    ).to(device)
    history = fit_anchor_score_sde(model, residual_arrays, args, device)

    output_dir = args.output_dir or ROOT_DIR / "export" / (
        "anchor_score_sde_%s" % args.tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "%s_AnchorScoreSDE_%d" % (args.tag, args.seed)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "data_dim": data_dim,
            "context_dim": context_dim,
            "channels": args.channels,
            "blocks": args.blocks,
            "time_dim": args.time_dim,
            "sigma_data": args.sigma_data,
        },
        output_dir / (model_name + ".pt"),
    )
    audit = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": "forecast_anchored_conditional_edm_score_sde",
        "network": "dilated_temporal_convolution",
        "sampler": "deterministic_heun_probability_flow",
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
        samples = sample_anchor_score_sde(
            model,
            arrays["x_" + split_key],
            anchors[split_key],
            args.n_scenarios,
            device,
            seed=args.seed + offset,
            steps=args.sampling_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
        )
        scenarios = scenarios_to_period_matrix(
            samples, target_scaler, args.tag, zero_indices
        )
        path = output_dir / (
            "scenarios_%s_AnchorScoreSDE_%d_%d_%s.pickle"
            % (args.tag, args.seed, args.n_scenarios, split)
        )
        with path.open("wb") as handle:
            pickle.dump(scenarios, handle)
        print("Wrote %s" % path)


if __name__ == "__main__":
    main()
