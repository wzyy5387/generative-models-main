# -*- coding: utf-8 -*-

"""Leakage-free retraining for the project's original conditional baselines."""

import argparse
import json
import pickle
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR
from GEFcom2014.models.GAN.utils_gan_wasserstein import (
    Discriminator_wassertein,
    Generator_linear,
)
from GEFcom2014.models.VAE.utils_vae import VAElinear
from GEFcom2014.models.probabilistic_baselines import (
    prepare_track,
    scenarios_to_period_matrix,
)
from models import (
    AutoregressiveConditioner,
    MonotonicNormalizer,
    buildFCNormalizingFlow,
)


MODEL_LABELS = {
    "cvae": "LegacyCVAE",
    "wgan-gp": "LegacyWGANGP",
    "umnn": "LegacyUMNN",
}


TRACK_CONFIGS = {
    "cvae": {
        "pv": dict(epochs=300, latent_s=40, width=200, layers=2, lr=10 ** -3.3, wd=10 ** -3.5),
        "wind": dict(epochs=200, latent_s=20, width=200, layers=1, lr=10 ** -3.4, wd=10 ** -3.4),
        "load": dict(epochs=200, latent_s=5, width=500, layers=1, lr=10 ** -3.9, wd=10 ** -4),
    },
    "wgan-gp": {
        "pv": dict(epochs=500, latent_s=64, width=256, layers=3, lr=2e-4, wd=1e-4),
        "wind": dict(epochs=300, latent_s=64, width=256, layers=2, lr=2e-4, wd=1e-4),
        "load": dict(epochs=200, latent_s=256, width=1024, layers=2, lr=2e-4, wd=1e-4),
    },
    "umnn": {
        "pv": dict(epochs=600, steps=1, layers=4, width=300, out_size=20, lr=1e-4, wd=5e-4),
        "wind": dict(epochs=600, steps=1, layers=4, width=300, out_size=20, lr=1e-4, wd=5e-4),
        "load": dict(epochs=500, steps=1, layers=4, width=300, out_size=20, lr=1e-4, wd=5e-4),
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x, y, batch_size, seed):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def fixed_rng(device, seed):
    devices = [device.index or 0] if device.type == "cuda" else []
    return torch.random.fork_rng(devices=devices)


def validation_elbo(model, x, y, batch_size, device, seed):
    losses = []
    model.eval()
    with torch.no_grad(), fixed_rng(device, seed):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(y[start:start + batch_size], dtype=torch.float32, device=device)
            losses.append(float(model.loss(batch_y, batch_x)))
    return float(np.mean(losses))


def validation_nll(model, x, y, batch_size, device):
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(y[start:start + batch_size], dtype=torch.float32, device=device)
            ll, _ = model.compute_ll(batch_y, batch_x)
            total -= float(ll.sum())
            count += len(batch_y)
    return total / count


def energy_score(samples, observations):
    """Mean multivariate energy score for samples shaped (case, member, period)."""
    samples = np.asarray(samples, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observation_term = np.linalg.norm(samples - observations[:, None, :], axis=2).mean(axis=1)
    differences = samples[:, :, None, :] - samples[:, None, :, :]
    ensemble_term = 0.5 * np.linalg.norm(differences, axis=3).mean(axis=(1, 2))
    return float(np.mean(observation_term - ensemble_term))


def sample_decoder(model, context, n_scenarios, device, seed, day_batch_size=32):
    chunks = []
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for start in range(0, len(context), day_batch_size):
            batch_context = torch.as_tensor(
                context[start:start + day_batch_size], dtype=torch.float32, device=device
            )
            expanded = batch_context.repeat_interleave(n_scenarios, dim=0)
            noise = torch.randn(
                expanded.shape[0], model.latent_s, generator=generator, device=device
            )
            decoder = model.dec if hasattr(model, "dec") else model.gen
            values = decoder(torch.cat((noise, expanded), dim=1))
            chunks.append(values.reshape(len(batch_context), n_scenarios, -1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def sample_umnn(model, context, data_dim, n_scenarios, device, seed, day_batch_size=4,
                chunk_dir=None):
    chunks = []
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    if chunk_dir is not None:
        chunk_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for start in range(0, len(context), day_batch_size):
            stop = min(start + day_batch_size, len(context))
            chunk_path = None if chunk_dir is None else chunk_dir / (
                "chunk_%05d_%05d.npy" % (start, stop)
            )
            if chunk_path is not None and chunk_path.is_file():
                chunks.append(np.load(chunk_path))
                # Advance the deterministic generator exactly as if this chunk
                # had been sampled in the current process.
                torch.randn(
                    (stop - start) * n_scenarios, data_dim,
                    generator=generator, device=device,
                )
                continue
            batch_context = torch.as_tensor(
                context[start:stop], dtype=torch.float32, device=device
            )
            expanded = batch_context.repeat_interleave(n_scenarios, dim=0)
            z = torch.randn(expanded.shape[0], data_dim, generator=generator, device=device)
            values = model.invert(z, expanded)
            chunk = values.reshape(len(batch_context), n_scenarios, data_dim).cpu().numpy()
            if chunk_path is not None:
                np.save(chunk_path, chunk)
            chunks.append(chunk)
            print("Sampled UMNN cases %d:%d of %d" % (start, stop, len(context)), flush=True)
    return np.concatenate(chunks, axis=0)


def update_checkpoint(model, value, best_value, min_delta):
    if np.isfinite(value) and value < best_value - min_delta:
        return value, deepcopy(model.state_dict()), 0
    return best_value, None, 1


def train_cvae(arrays, config, args, device, checkpoint_path=None):
    model = VAElinear(
        latent_s=config["latent_s"], cond_in=arrays["x_ls"].shape[1],
        in_size=arrays["y_ls"].shape[1], enc_w=config["width"], enc_l=config["layers"],
        dec_w=config["width"], dec_l=config["layers"], gpu=device.type == "cuda",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    loader = make_loader(arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed)
    history, best_state, best_value, stale = [], None, float("inf"), 0
    for epoch in range(config["epochs"]):
        model.train()
        total = count = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch_y, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * len(batch_y)
            count += len(batch_y)
        train_loss = total / count
        val_loss = validation_elbo(
            model, arrays["x_vs"], arrays["y_vs"], args.batch_size, device,
            args.seed + 104729,
        )
        new_best, state, increment = update_checkpoint(model, val_loss, best_value, args.min_delta)
        if state is not None:
            best_value, best_state, stale = new_best, state, 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            stale += increment
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        report_epoch(epoch, train_loss, val_loss)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, history, "validation_elbo"


def build_umnn(arrays, config, args, device):
    data_dim, context_dim = arrays["y_ls"].shape[1], arrays["x_ls"].shape[1]
    conditioner_args = {
        "in_size": data_dim, "hidden": [config["width"]] * config["layers"],
        "out_size": config["out_size"], "cond_in": context_dim,
    }
    normalizer_args = {
        "integrand_net": [config["out_size"] * 2] * 3,
        "cond_size": config["out_size"], "nb_steps": args.umnn_integration_steps,
        "solver": "CCParallel", "hot_encoding": True,
    }
    return buildFCNormalizingFlow(
        nb_steps=config["steps"], conditioner_type=AutoregressiveConditioner,
        conditioner_args=conditioner_args, normalizer_type=MonotonicNormalizer,
        normalizer_args=normalizer_args,
    ).to(device)


def train_umnn(arrays, config, args, device, checkpoint_path=None):
    model = build_umnn(arrays, config, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    loader = make_loader(arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed)
    history, best_state, best_value, stale = [], None, float("inf"), 0
    for epoch in range(config["epochs"]):
        model.train()
        total = count = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            ll, _ = model.compute_ll(batch_y, batch_x)
            loss = -ll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * len(batch_y)
            count += len(batch_y)
        train_loss = total / count
        val_loss = validation_nll(model, arrays["x_vs"], arrays["y_vs"], args.batch_size, device)
        new_best, state, increment = update_checkpoint(model, val_loss, best_value, args.min_delta)
        if state is not None:
            best_value, best_state, stale = new_best, state, 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            stale += increment
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        report_epoch(epoch, train_loss, val_loss)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, history, "validation_nll"


def train_wgan(arrays, config, args, device, checkpoint_path=None):
    common = dict(
        latent_s=config["latent_s"], cond_in=arrays["x_ls"].shape[1],
        in_size=arrays["y_ls"].shape[1], gen_w=config["width"], gen_l=config["layers"],
        lambda_gp=args.lambda_gp, gpu=device.type == "cuda",
    )
    generator = Generator_linear(**common).to(device)
    critic = Discriminator_wassertein(**common).to(device)
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=config["lr"], betas=(0.0, 0.9), weight_decay=config["wd"]
    )
    opt_d = torch.optim.Adam(
        critic.parameters(), lr=config["lr"], betas=(0.0, 0.9), weight_decay=config["wd"]
    )
    loader = make_loader(arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed)
    history, best_state, best_value, stale, updates = [], None, float("inf"), 0, 0
    for epoch in range(config["epochs"]):
        generator.train()
        critic.train()
        d_total = g_total = 0.0
        d_count = g_count = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # The legacy gradient-penalty implementation differentiates with
            # respect to the interpolation itself. Keep that path explicit
            # while detaching generated samples from the generator update.
            batch_y.requires_grad_(True)
            noise = torch.randn(len(batch_y), generator.latent_s, device=device)
            fake = generator(noise, batch_x).detach()
            opt_d.zero_grad(set_to_none=True)
            d_loss = critic.loss(fake, batch_y, batch_x)
            d_loss.backward()
            opt_d.step()
            d_total += float(d_loss.detach())
            d_count += 1
            updates += 1
            if updates % args.n_critic == 0:
                opt_g.zero_grad(set_to_none=True)
                generated = generator(torch.randn_like(noise), batch_x)
                g_loss = -critic(generated, batch_x).mean()
                g_loss.backward()
                opt_g.step()
                g_total += float(g_loss.detach())
                g_count += 1
        val_samples = sample_decoder(
            generator, arrays["x_vs"], args.validation_scenarios, device,
            args.seed + 104729,
        )
        val_loss = energy_score(val_samples, arrays["y_vs"])
        train_loss = g_total / max(g_count, 1)
        new_best, state, increment = update_checkpoint(generator, val_loss, best_value, args.min_delta)
        if state is not None:
            best_value, best_state, stale = new_best, state, 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            stale += increment
        history.append({
            "epoch": epoch, "generator_loss": train_loss,
            "critic_loss": d_total / max(d_count, 1), "validation_loss": val_loss,
        })
        report_epoch(epoch, train_loss, val_loss)
        if stale >= args.patience:
            break
    generator.load_state_dict(best_state)
    return generator, history, "validation_energy_score"


def report_epoch(epoch, train_loss, validation_loss):
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(
            "Epoch %d | LS %.6f VS %.6f" % (epoch + 1, train_loss, validation_loss),
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_LABELS))
    parser.add_argument("--tag", default="wind", choices=["wind", "pv", "load"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--validation-scenarios", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--umnn-integration-steps", type=int, default=50)
    parser.add_argument("--umnn-day-batch-size", type=int, default=4)
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Load an existing UMNN best checkpoint and only export scenarios.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    arrays, target_scaler, zero_indices = prepare_track(args.tag)
    config = dict(TRACK_CONFIGS[args.model][args.tag])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    label = MODEL_LABELS[args.model]
    output_dir = args.output_dir or ROOT_DIR / "export" / ("legacy_%s_%s" % (
        args.model.replace("-", "_"), args.tag
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Training %s on %s with %s" % (label, args.tag, device))
    model_name = "%s_%s_%d" % (args.tag, label, args.seed)
    best_checkpoint_path = output_dir / (model_name + "_best.pt")

    training_started = time.perf_counter()
    if args.sample_only:
        if args.model != "umnn":
            raise ValueError("--sample-only currently supports only --model umnn")
        checkpoint_path = args.checkpoint or best_checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError("UMNN checkpoint not found: %s" % checkpoint_path)
        model = build_umnn(arrays, config, args, device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        history, selection_metric, sample = [], "validation_nll", None
    elif args.model == "cvae":
        model, history, selection_metric = train_cvae(
            arrays, config, args, device, best_checkpoint_path
        )
        sample = sample_decoder
    elif args.model == "wgan-gp":
        model, history, selection_metric = train_wgan(
            arrays, config, args, device, best_checkpoint_path
        )
        sample = sample_decoder
    else:
        model, history, selection_metric = train_umnn(
            arrays, config, args, device, best_checkpoint_path
        )
        sample = None
    training_seconds = time.perf_counter() - training_started

    sampling_started = time.perf_counter()
    if args.model == "umnn":
        validation_samples = sample_umnn(
            model, arrays["x_vs"], arrays["y_ls"].shape[1], args.n_scenarios,
            device, args.seed + 200003, args.umnn_day_batch_size,
            output_dir / (model_name + "_VS_chunks"),
        )
        test_samples = sample_umnn(
            model, arrays["x_test"], arrays["y_ls"].shape[1], args.n_scenarios,
            device, args.seed + 300007, args.umnn_day_batch_size,
            output_dir / (model_name + "_TEST_chunks"),
        )
    else:
        validation_samples = sample(
            model, arrays["x_vs"], args.n_scenarios, device, args.seed + 200003
        )
        test_samples = sample(
            model, arrays["x_test"], args.n_scenarios, device, args.seed + 300007
        )
    sampling_seconds = time.perf_counter() - sampling_started

    torch.save(model.state_dict(), output_dir / (model_name + ".pt"))
    metadata = {
        "model": args.model, "label": label, "track": args.tag, "seed": args.seed,
        "config": config, "selection_split": "VS", "selection_metric": selection_metric,
        "test_used_for_selection": False, "history": history,
        "runtime_seconds": {
            "training": training_seconds, "scenario_generation": sampling_seconds,
            "total": training_seconds + sampling_seconds,
        },
        "arguments": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "output_dir": str(output_dir),
        },
    }
    with (output_dir / (model_name + ".json")).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / (model_name + "_scaler.pickle")).open("wb") as handle:
        pickle.dump(target_scaler, handle)

    for split, samples in (("VS", validation_samples), ("TEST", test_samples)):
        scenarios = scenarios_to_period_matrix(samples, target_scaler, args.tag, zero_indices)
        path = output_dir / (
            "scenarios_%s_%s_%d_%d_%s.pickle" %
            (args.tag, label, args.seed, args.n_scenarios, split)
        )
        with path.open("wb") as handle:
            pickle.dump(scenarios, handle)
        print("Wrote %s" % path)


if __name__ == "__main__":
    main()
