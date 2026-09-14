# -*- coding: utf-8 -*-

"""D3U-style and Treeffuser baselines for conditional daily scenarios.

The D3U adaptation preserves the published method's three defining components:
a pretrained deterministic conditioner, residual diffusion, and a patch-based
denoising transformer. The conditioner accepts future NWP covariates because the
GEFCom task does not expose the historical-window interface used by official D3U.
"""

import argparse
import json
import math
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR
from GEFcom2014.models.probabilistic_baselines import (
    prepare_dataset,
    scenarios_to_period_matrix,
    set_seed,
    sinusoidal_embedding,
)


MODEL_LABELS = {
    "d3u": "D3UAdapted",
    "treeffuser": "Treeffuser",
}


class DeterministicConditioner(nn.Module):
    """Point forecaster whose frozen representation conditions PatchDN."""

    def __init__(self, context_dim, data_dim, hidden_dim=256, layers=2):
        super().__init__()
        modules = []
        input_dim = context_dim
        for _ in range(layers):
            modules.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*modules)
        self.head = nn.Linear(hidden_dim, data_dim)

    def encode(self, context):
        return self.encoder(context)

    def forward(self, context):
        representation = self.encode(context)
        return self.head(representation), representation


def _modulate(values, shift, scale):
    return values * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaLNZeroBlock(nn.Module):
    """PatchDN transformer block with adaptive zero-initialized conditioning."""

    def __init__(self, hidden_dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, values, condition):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=1)
        attention_input = _modulate(self.norm_attention(values), shift_a, scale_a)
        attention_output = self.attention(
            attention_input, attention_input, attention_input, need_weights=False
        )[0]
        values = values + gate_a[:, None, :] * attention_output
        mlp_input = _modulate(self.norm_mlp(values), shift_m, scale_m)
        return values + gate_m[:, None, :] * self.mlp(mlp_input)


class PatchDenoiser(nn.Module):
    """One-dimensional PatchDN adapted to a 24-period daily trajectory."""

    def __init__(
        self,
        data_dim,
        condition_dim,
        hidden_dim=128,
        patch_size=4,
        depth=3,
        heads=4,
        time_dim=128,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.data_dim = int(data_dim)
        self.patch_size = int(patch_size)
        self.padded_dim = int(math.ceil(data_dim / patch_size) * patch_size)
        self.patch_count = self.padded_dim // patch_size
        self.time_dim = int(time_dim)
        self.patch_projection = nn.Linear(patch_size, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, hidden_dim))
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, heads) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        self.output_projection = nn.Linear(hidden_dim, patch_size)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def patchify(self, values):
        if values.shape[1] != self.data_dim:
            raise ValueError("trajectory dimension does not match PatchDN")
        if self.padded_dim != self.data_dim:
            values = torch.nn.functional.pad(values, (0, self.padded_dim - self.data_dim))
        return values.reshape(values.shape[0], self.patch_count, self.patch_size)

    def unpatchify(self, patches):
        values = patches.reshape(patches.shape[0], self.padded_dim)
        return values[:, :self.data_dim]

    def forward(self, noisy_values, timesteps, condition):
        patches = self.patch_projection(self.patchify(noisy_values)) + self.position
        time_features = sinusoidal_embedding(timesteps, self.time_dim)
        conditioning = self.time_projection(time_features) + self.condition_projection(
            condition
        )
        for block in self.blocks:
            patches = block(patches, conditioning)
        shift, scale = self.final_modulation(conditioning).chunk(2, dim=1)
        patches = _modulate(self.final_norm(patches), shift, scale)
        return self.unpatchify(self.output_projection(patches))


def _loader(x, y, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def fit_conditioner(model, arrays, args, device):
    loader = _loader(
        arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.anchor_learning_rate, weight_decay=args.weight_decay
    )
    x_vs = torch.as_tensor(arrays["x_vs"], dtype=torch.float32, device=device)
    y_vs = torch.as_tensor(arrays["y_vs"], dtype=torch.float32, device=device)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(args.anchor_epochs):
        model.train()
        total = 0.0
        count = 0
        for context, target in loader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(context)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            count += target.shape[0]
        model.eval()
        with torch.no_grad():
            validation = float(torch.nn.functional.mse_loss(model(x_vs)[0], y_vs))
        train_loss = total / count
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation}
        )
        if validation < best_loss - args.min_delta:
            best_loss = validation
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print("Anchor epoch %d | LS %.6f VS %.6f" % (epoch + 1, train_loss, validation))
        if stale >= args.anchor_patience:
            break
    if best_state is None:
        raise RuntimeError("D3U conditioner did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return history


def _d3u_condition(conditioner, context):
    anchor, representation = conditioner(context)
    return anchor, torch.cat((representation, anchor), dim=1)


def fit_d3u(denoiser, conditioner, scheduler, arrays, args, device):
    loader = _loader(
        arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed + 1
    )
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    x_vs = torch.as_tensor(arrays["x_vs"], dtype=torch.float32, device=device)
    y_vs = torch.as_tensor(arrays["y_vs"], dtype=torch.float32, device=device)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history = []

    def diffusion_loss(context, target):
        with torch.no_grad():
            anchor, condition = _d3u_condition(conditioner, context)
            residual = target - anchor
        noise = torch.randn_like(residual)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (target.shape[0],), device=device,
        ).long()
        noisy = scheduler.add_noise(residual, noise, timesteps)
        prediction = denoiser(noisy, timesteps, condition)
        objective = scheduler.get_velocity(residual, noise, timesteps)
        return torch.nn.functional.mse_loss(prediction, objective)

    for epoch in range(args.epochs):
        denoiser.train()
        total = 0.0
        count = 0
        for context, target in loader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(context, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            count += target.shape[0]
        validation_losses = []
        denoiser.eval()
        with torch.no_grad(), torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ):
            torch.manual_seed(args.seed + 104729)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 104729)
            for start in range(0, x_vs.shape[0], args.batch_size):
                validation_losses.append(
                    float(
                        diffusion_loss(
                            x_vs[start:start + args.batch_size],
                            y_vs[start:start + args.batch_size],
                        )
                    )
                )
        train_loss = total / count
        validation = float(np.mean(validation_losses))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation}
        )
        if validation < best_loss - args.min_delta:
            best_loss = validation
            best_state = deepcopy(denoiser.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print("D3U epoch %d | LS %.6f VS %.6f" % (epoch + 1, train_loss, validation))
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("D3U diffusion did not produce a checkpoint")
    denoiser.load_state_dict(best_state)
    denoiser.eval()
    return history


def sample_d3u(
    denoiser,
    conditioner,
    scheduler,
    context,
    n_scenarios,
    inference_steps,
    device,
    seed,
    day_batch_size=16,
):
    scheduler.set_timesteps(inference_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks = []
    denoiser.eval()
    conditioner.eval()
    with torch.no_grad():
        for start in range(0, context.shape[0], day_batch_size):
            batch = torch.as_tensor(
                context[start:start + day_batch_size], dtype=torch.float32, device=device
            )
            anchor, condition = _d3u_condition(conditioner, batch)
            expanded_condition = condition.repeat_interleave(n_scenarios, dim=0)
            residual = torch.randn(
                expanded_condition.shape[0], anchor.shape[1],
                generator=generator, device=device,
            )
            for timestep in scheduler.timesteps:
                prediction = denoiser(
                    residual, timestep.expand(residual.shape[0]), expanded_condition
                )
                residual = scheduler.step(
                    prediction, timestep, residual, generator=generator
                ).prev_sample
            residual = residual.reshape(
                batch.shape[0], n_scenarios, anchor.shape[1]
            )
            chunks.append((anchor[:, None, :] + residual).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_treeffuser(args):
    try:
        from treeffuser import Treeffuser
    except ImportError as error:
        raise ImportError(
            "Treeffuser baseline requires `pip install treeffuser==0.2.0`"
        ) from error
    return Treeffuser(
        n_repeats=args.tree_n_repeats,
        n_estimators=args.tree_n_estimators,
        early_stopping_rounds=args.tree_early_stopping_rounds,
        eval_percent=args.tree_eval_percent,
        num_leaves=args.tree_num_leaves,
        learning_rate=args.tree_learning_rate,
        n_jobs=args.tree_n_jobs,
        sde_name=args.tree_sde,
        sde_initialize_from_data=args.tree_sde_initialize_from_data,
        seed=args.seed,
        verbose=args.tree_verbose,
    )


def sample_treeffuser(model, context, n_scenarios, args, seed):
    samples = model.sample(
        context,
        n_samples=n_scenarios,
        n_parallel=args.tree_n_parallel,
        n_steps=args.tree_sampling_steps,
        seed=seed,
        verbose=bool(args.tree_verbose),
    )
    samples = np.asarray(samples, dtype=np.float32)
    expected = (n_scenarios, context.shape[0])
    if samples.shape[:2] != expected:
        raise RuntimeError("Unexpected Treeffuser sample shape %s" % (samples.shape,))
    return samples.transpose(1, 0, 2)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", choices=sorted(MODEL_LABELS), default=None)
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--scenario-splits", nargs="+", choices=["VS", "TEST"], default=["VS", "TEST"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--anchor-epochs", type=int, default=100)
    parser.add_argument("--anchor-hidden-dim", type=int, default=256)
    parser.add_argument("--anchor-layers", type=int, default=2)
    parser.add_argument("--anchor-learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor-patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--tree-n-repeats", type=int, default=30)
    parser.add_argument("--tree-n-estimators", type=int, default=3000)
    parser.add_argument("--tree-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--tree-eval-percent", type=float, default=0.1)
    parser.add_argument("--tree-num-leaves", type=int, default=31)
    parser.add_argument("--tree-learning-rate", type=float, default=0.1)
    parser.add_argument("--tree-n-jobs", type=int, default=-1)
    parser.add_argument("--tree-sde", choices=["vesde", "vpsde", "sub-vpsde"], default="vesde")
    parser.add_argument("--tree-sde-initialize-from-data", action="store_true")
    parser.add_argument("--tree-sampling-steps", type=int, default=50)
    parser.add_argument("--tree-n-parallel", type=int, default=10)
    parser.add_argument("--tree-verbose", type=int, default=0)
    return parser


def parse_args(argv=None):
    parser = _build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config is not None:
        with preliminary.config.open("r", encoding="utf-8") as handle:
            defaults = json.load(handle)
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - known)
        if unknown:
            parser.error("unknown config keys: %s" % ", ".join(unknown))
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.model is None:
        parser.error("--model is required, either directly or through --config")
    if isinstance(args.dataset_bundle, str):
        args.dataset_bundle = Path(args.dataset_bundle)
    if isinstance(args.output_dir, str):
        args.output_dir = Path(args.output_dir)
    if len(set(args.scenario_splits)) != len(args.scenario_splits):
        parser.error("--scenario-splits must not contain duplicates")
    if args.n_scenarios < 2:
        parser.error("--n-scenarios must be at least two")
    return args


def _json_arguments(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    arrays, target_scaler, zero_indices = prepare_dataset(
        args.tag, args.dataset_bundle
    )
    data_dim = arrays["y_ls"].shape[1]
    context_dim = arrays["x_ls"].shape[1]
    label = MODEL_LABELS[args.model]
    output_dir = args.output_dir or ROOT_DIR / "export" / ("%s_%s" % (args.model, args.tag))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "%s_%s_%d" % (args.tag, label, args.seed)
    started = time.perf_counter()

    if args.model == "d3u":
        conditioner = DeterministicConditioner(
            context_dim, data_dim, args.anchor_hidden_dim, args.anchor_layers
        ).to(device)
        anchor_history = fit_conditioner(conditioner, arrays, args, device)
        denoiser = PatchDenoiser(
            data_dim=data_dim,
            condition_dim=args.anchor_hidden_dim + data_dim,
            hidden_dim=args.hidden_dim,
            patch_size=args.patch_size,
            depth=args.depth,
            heads=args.heads,
            time_dim=args.time_dim,
        ).to(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            clip_sample=False,
        )
        diffusion_history = fit_d3u(
            denoiser, conditioner, scheduler, arrays, args, device
        )
        fit_seconds = time.perf_counter() - started
        torch.save(
            {
                "conditioner": conditioner.state_dict(),
                "denoiser": denoiser.state_dict(),
                "context_dim": context_dim,
                "data_dim": data_dim,
                "arguments": _json_arguments(args),
            },
            output_dir / (model_name + ".pt"),
        )

        def generate(split, offset):
            return sample_d3u(
                denoiser, conditioner, scheduler,
                arrays["x_" + split.lower()], args.n_scenarios,
                args.inference_steps, device, args.seed + offset,
            )

        audit = {
            "method": "nwp_adapted_d3u",
            "method_identity": [
                "pretrained_deterministic_conditioner",
                "frozen_condition_representation",
                "residual_diffusion",
                "patchdn_adaln_zero_transformer",
            ],
            "adaptation": "future NWP replaces the historical-window conditioner used by official D3U",
            "anchor_history": anchor_history,
            "diffusion_history": diffusion_history,
        }
        selection_split = "VS"
    else:
        if device.type != "cpu":
            print("Treeffuser uses the CPU LightGBM backend; --device is ignored")
        model = build_treeffuser(args)
        model.fit(arrays["x_ls"], arrays["y_ls"])
        fit_seconds = time.perf_counter() - started
        with (output_dir / (model_name + ".pickle")).open("wb") as handle:
            pickle.dump(model, handle)

        def generate(split, offset):
            return sample_treeffuser(
                model, arrays["x_" + split.lower()], args.n_scenarios,
                args, args.seed + offset,
            )

        audit = {
            "method": "official_treeffuser_0.2.0",
            "backend": "LightGBM conditional score model",
            "internal_early_stopping_split": "LS-only random holdout",
        }
        selection_split = None

    sample_seconds = {}
    for split, offset in (("VS", 104729), ("TEST", 130363)):
        if split not in args.scenario_splits:
            continue
        split_started = time.perf_counter()
        samples = generate(split, offset)
        sample_seconds[split] = time.perf_counter() - split_started
        scenarios = scenarios_to_period_matrix(
            samples, target_scaler, args.tag, zero_indices
        )
        path = output_dir / (
            "scenarios_%s_%s_%d_%d_%s.pickle"
            % (args.tag, label, args.seed, args.n_scenarios, split)
        )
        with path.open("wb") as handle:
            pickle.dump(scenarios, handle)
        print("Wrote %s" % path)

    with (output_dir / (model_name + ".json")).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **_json_arguments(args),
                **audit,
                "fit_split": "LS",
                "selection_split": selection_split,
                "test_used_for_selection": False,
                "fit_seconds": fit_seconds,
                "sampling_seconds": sample_seconds,
                "output_dir": str(output_dir),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
