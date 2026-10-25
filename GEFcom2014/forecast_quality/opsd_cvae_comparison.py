"""Train and compare standard CVAE, forecast-anchored CVAE, and FA-BM-VAE on frozen OPSD."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.forecast_quality.aggregate_wind_five_seed import block_paired_inference, holm_adjust
from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily, load_scenarios
from GEFcom2014.forecast_quality.opsd_sa_negative_evaluation import _calibrate, _daily_metrics
from GEFcom2014.forecast_quality.unified_postprocessing import HourlyAffineCalibrator
from GEFcom2014.models import probabilistic_baselines as pb
from GEFcom2014.models.QBM_VAE import ConditionalQBMVAE, build_qbm_vae_scenarios, build_sampler, fit_qbm_vae
from GEFcom2014.models.QBM_VAE.qbm_vae import load_frozen_graph_mask, build_generation_seed_metadata
from GEFcom2014.models.neso_extra_baselines import ConditionalVAE, configurations as neso_configurations, sample_cvae


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "export" / "opsd_cvae_comparison_v1"
BUNDLE = ROOT / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz"
FROZEN_CONFIG = ROOT / "configs" / "paper" / "fa_bm_vae_opsd_wind_sa_negative_v1.json"
OLD_FA_DIR = ROOT / "export" / "qbm_vae_opsd-wind"
OLD_FA_META = OLD_FA_DIR / "opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0.json"
OLD_FA_CHECKPOINT = OLD_FA_DIR / "opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0.pickle"
OLD_ACCEPTANCE = ROOT / "export" / "opsd_sa_negative_matched_v1" / "main_table_seed012_strict_no_anchor_v2" / "provenance_acceptance.json"
MODELS = ("standard_cvae", "fa_cvae", "fa_bm_vae")
LABELS = {"standard_cvae": "Standard CVAE", "fa_cvae": "FA-CVAE", "fa_bm_vae": "FA-BM-VAE"}
SEEDS = (0, 1, 2)
VARIANTS = ("Raw", "Cal")
METRICS = ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "interval_width90", "MAQCE", "mean_bias")
PRIMARY_COMPARISONS = (("fa_bm_vae", "standard_cvae"), ("fa_bm_vae", "fa_cvae"))
SPLIT_OFFSETS = {"VS": 0, "TEST": 1000003}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def read_protocol():
    return json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))


def read_manifest():
    return json.loads((OUT / "task_manifest.json").read_text(encoding="utf-8"))


def save_manifest(manifest):
    atomic_json(OUT / "task_manifest.json", manifest)


def _date_info():
    arrays, metadata, dates = load_daily_bundle(BUNDLE)
    if metadata.get("dataset_name") != "opsd-wind":
        raise ValueError("OPSD bundle identity mismatch")
    if {k: len(arrays["y_" + k]) for k in ("ls", "vs", "test")} != {"ls": 1036, "vs": 90, "test": 90}:
        raise ValueError("The frozen OPSD split must contain 1036/90/90 days")
    if len(dates["test"]) != 90 or np.any(np.diff(pd_to_days(dates["test"])) != 1):
        raise ValueError("Frozen TEST dates must be exactly 90 consecutive days")
    return arrays, metadata, dates


def _load_vs_evaluation_arrays():
    """Load only the VS truth needed by smoke; never open TEST values here."""
    with np.load(BUNDLE, allow_pickle=False) as data:
        x_vs = np.asarray(data["x_vs"], dtype=np.float32)
        y_vs = np.asarray(data["y_vs"], dtype=np.float32)
        dates_vs = np.asarray(data["dates_vs"]).astype(str)
    if x_vs.shape[0] != y_vs.shape[0] or len(dates_vs) != len(y_vs):
        raise ValueError("VS dates and arrays have inconsistent lengths")
    if not np.isfinite(x_vs).all() or not np.isfinite(y_vs).all():
        raise ValueError("VS arrays contain non-finite values")
    return {"x_vs": x_vs, "y_vs": y_vs}, {"vs": dates_vs}


def pd_to_days(date_strings):
    return np.asarray([np.datetime64(str(d), "D").astype(np.int64) for d in date_strings])


class ForecastAnchoredCVAE(nn.Module):
    """Diagonal-Gaussian conditional VAE with the frozen LS-scaled forecast added as anchor."""
    def __init__(self, context_dim, data_dim=24, latent_dim=48, hidden_dim=256, layers=2,
                 min_logvar=-8.0, max_logvar=6.0, min_log_std=-5.0, max_log_std=1.0):
        super().__init__()
        self.context_dim, self.data_dim, self.latent_dim = context_dim, data_dim, latent_dim
        self.min_logvar, self.max_logvar = float(min_logvar), float(max_logvar)
        self.min_log_std, self.max_log_std = float(min_log_std), float(max_log_std)

        def net(input_dim, output_dim):
            parts = []
            for _ in range(layers):
                parts.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
                input_dim = hidden_dim
            parts.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*parts)

        self.encoder = net(context_dim + data_dim, 2 * latent_dim)
        self.prior = net(context_dim, 2 * latent_dim)
        self.decoder = net(context_dim + latent_dim, 2 * data_dim)
        with torch.no_grad():
            self.decoder[-1].weight[:data_dim].zero_()
            self.decoder[-1].bias[:data_dim].zero_()

    def forward_parameters(self, target, context):
        q_mu, q_logvar = self.encoder(torch.cat((target, context), dim=1)).chunk(2, dim=1)
        p_mu, p_logvar = self.prior(context).chunk(2, dim=1)
        q_logvar = q_logvar.clamp(self.min_logvar, self.max_logvar)
        p_logvar = p_logvar.clamp(self.min_logvar, self.max_logvar)
        z = q_mu + torch.exp(0.5 * q_logvar) * torch.randn_like(q_mu)
        residual_mu, log_std = self.decoder(torch.cat((z, context), dim=1)).chunk(2, dim=1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        pred_mu = context[:, -self.data_dim:] + residual_mu
        return q_mu, q_logvar, p_mu, p_logvar, pred_mu, log_std

    def loss(self, target, context, kl_weight=0.05):
        q_mu, q_lv, p_mu, p_lv, pred_mu, log_std = self.forward_parameters(target, context)
        recon = (log_std + 0.5 * (target - pred_mu).square() * torch.exp(-2.0 * log_std)).sum(1)
        kl = diagonal_gaussian_kl(q_mu, q_lv, p_mu, p_lv)
        return (recon + float(kl_weight) * kl).mean(), {"reconstruction": recon.mean(), "kl": kl.mean()}

    def sample(self, context, n_scenarios, generator, batch_size=64):
        chunks = []
        self.eval()
        with torch.no_grad():
            for start in range(0, len(context), batch_size):
                x = torch.as_tensor(context[start:start + batch_size], dtype=torch.float32,
                                    device=next(self.parameters()).device)
                expanded = x.repeat_interleave(n_scenarios, 0)
                p_mu, p_lv = self.prior(expanded).chunk(2, dim=1)
                p_lv = p_lv.clamp(self.min_logvar, self.max_logvar)
                z = p_mu + torch.exp(0.5 * p_lv) * torch.randn(p_mu.shape, generator=generator,
                                                                device=p_mu.device, dtype=p_mu.dtype)
                residual_mu, log_std = self.decoder(torch.cat((z, expanded), dim=1)).chunk(2, dim=1)
                log_std = log_std.clamp(self.min_log_std, self.max_log_std)
                eps = torch.randn(residual_mu.shape, generator=generator, device=residual_mu.device,
                                  dtype=residual_mu.dtype)
                values = expanded[:, -self.data_dim:] + residual_mu + log_std.exp() * eps
                chunks.append(values.reshape(len(x), n_scenarios, self.data_dim).cpu().numpy())
        return np.concatenate(chunks, axis=0)


def diagonal_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar):
    """KL(q||p) for diagonal Gaussian tensors, reduced over latent dimensions only."""
    return 0.5 * (p_logvar - q_logvar +
                  (q_logvar.exp() + (q_mu - p_mu).square()) / p_logvar.exp() - 1.0).sum(dim=1)


def _fit_fa_cvae(model, arrays, args, kl_weight, warmup_epochs):
    loader = pb.make_loader(arrays["x_ls"], arrays["y_ls"], args.batch_size, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history, best_state, best_score, stale = [], None, float("inf"), 0
    x_vs = torch.as_tensor(arrays["x_vs"], dtype=torch.float32, device=args.device)
    y_vs = torch.as_tensor(arrays["y_vs"], dtype=torch.float32, device=args.device)
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_n, train_recon, train_kl = 0.0, 0, 0.0, 0.0
        effective_weight = kl_weight * min(1.0, (epoch + 1) / max(1, warmup_epochs))
        for context, target in loader:
            context, target = context.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = model.loss(target, context, effective_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n = len(target)
            train_loss += float(loss.detach()) * n
            train_recon += float(components["reconstruction"].detach()) * n
            train_kl += float(components["kl"].detach()) * n
            train_n += n
        model.eval()
        recon_values, objective_values = [], []
        with torch.no_grad(), torch.random.fork_rng(devices=[args.device.index or 0] if args.device.type == "cuda" else []):
            torch.manual_seed(args.seed + 104729)
            if args.device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 104729)
            for start in range(0, len(x_vs), args.batch_size):
                loss, components = model.loss(y_vs[start:start + args.batch_size], x_vs[start:start + args.batch_size], effective_weight)
                recon_values.append(float(components["reconstruction"]) * len(x_vs[start:start + args.batch_size]))
                objective_values.append(float(loss) * len(x_vs[start:start + args.batch_size]))
        val_recon = sum(recon_values) / len(x_vs)
        val_objective = sum(objective_values) / len(x_vs)
        history.append({"epoch": epoch, "training_objective": train_loss / train_n,
                        "training_reconstruction": train_recon / train_n, "training_kl": train_kl / train_n,
                        "kl_weight": effective_weight, "vs_reconstruction": val_recon,
                        "vs_objective": val_objective})
        if val_recon < best_score - args.min_delta:
            best_score, best_state, stale = val_recon, deepcopy(model.state_dict()), 0
            model.selection_epoch = epoch
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("FA-CVAE did not produce a finite VS-selected checkpoint")
    model.load_state_dict(best_state)
    return history, best_score


def _scale_ls_vs(x_ls, y_ls, x_vs, y_vs, model):
    x_scaler, y_scaler = StandardScaler().fit(x_ls), StandardScaler().fit(y_ls)
    x_ls_s, x_vs_s = x_scaler.transform(x_ls), x_scaler.transform(x_vs)
    y_ls_s, y_vs_s = y_scaler.transform(y_ls), y_scaler.transform(y_vs)
    if model in ("fa_cvae", "fa_bm_vae"):
        if x_ls.shape[1] < y_ls.shape[1]:
            raise ValueError("OPSD context does not contain the 24-hour anchor block")
        anchor_ls = y_scaler.transform(x_ls[:, :y_ls.shape[1]])
        anchor_vs = y_scaler.transform(x_vs[:, :y_vs.shape[1]])
        x_ls_s = np.concatenate((x_ls_s, anchor_ls), axis=1)
        x_vs_s = np.concatenate((x_vs_s, anchor_vs), axis=1)
        if not np.allclose(x_ls_s[:, -24:], anchor_ls, rtol=0, atol=0):
            raise AssertionError("FA anchor transform mismatch")
    arrays = {"x_ls": x_ls_s.astype(np.float32), "y_ls": y_ls_s.astype(np.float32),
              "x_vs": x_vs_s.astype(np.float32), "y_vs": y_vs_s.astype(np.float32)}
    return arrays, x_scaler, y_scaler


def _parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def _get_fa_bm_config():
    old = json.loads(OLD_FA_META.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    config = dict(old)
    config.update(frozen)
    config["graph_mask_model"] = str(OLD_FA_CHECKPOINT.resolve())
    config["graph_mask_sha256"] = sha(OLD_FA_CHECKPOINT)
    config["dataset_bundle"] = str(BUNDLE.resolve())
    config["test_used_for_selection"] = False
    config["checkpoint_selection"] = "minimum VS Gaussian reconstruction NLL; TEST arrays are not loaded during fit"
    return config


def _model_configs():
    standard = dict(neso_configurations()["cvae"])
    standard.update({"model": "standard_cvae", "checkpoint_selection": "VS negative ELBO",
                     "latent_prior": "standard_normal", "kl_weight": 1.0, "warmup_epochs": 0,
                     "input_transform": "LS-fit StandardScaler on 33 context columns",
                     "anchor_addition": False, "training_epochs": 200, "training_seed_policy": "seed 0,1,2",
                     "generation_seed_base": 10000, "generation_split_offsets": SPLIT_OFFSETS})
    matched = {"model": "fa_cvae", "latent_dim": 48, "hidden_dim": 256, "layers": 2,
               "epochs": 80, "batch_size": 256, "learning_rate": 3e-4, "weight_decay": 1e-5,
               "patience": 20, "min_delta": 1e-4, "grad_clip": 5.0,
               "kl_weight": 0.05, "warmup_epochs": 20, "decoder_log_std_range": [-5.0, 1.0],
               "posterior_prior_logvar_range": [-8.0, 6.0], "checkpoint_selection": "minimum VS Gaussian reconstruction NLL",
               "input_transform": "LS-fit StandardScaler on 33 context columns plus LS-fitted y scaler applied to first 24 raw forecast columns",
               "anchor_addition": "anchor + residual mean; forecast anchor is last 24 context columns",
               "training_seed_policy": "seed 0,1,2", "generation_seed_base": 10000,
               "generation_split_offsets": SPLIT_OFFSETS}
    bm = _get_fa_bm_config()
    bm.update({"model": "fa_bm_vae", "training_seed_policy": "seed 0,1,2",
               "generation_seed_base": 10000, "generation_split_offsets": SPLIT_OFFSETS,
               "checkpoint_selection": "minimum VS Gaussian reconstruction NLL; TEST arrays are not loaded during fit",
               "data_access_contract": "training reads x_ls/y_ls/x_vs/y_vs only; TEST arrays are opened after checkpoint freeze"})
    return {"standard_cvae": standard, "fa_cvae": matched, "fa_bm_vae": bm}


def _build_bm_model(config, graph_mask, device):
    return ConditionalQBMVAE(
        latent_s=int(config["latent_s"]), cond_in=int(config["cond_in"]), in_size=24,
        enc_w=int(config["enc_w"]), enc_l=int(config["enc_l"]),
        dec_w=int(config["dec_w"]), dec_l=int(config["dec_l"]), conditional_qbm=True,
        graph_mask=graph_mask, gpu=device.type == "cuda", decoder_covariance="diagonal",
        max_ar_coefficient=float(config.get("max_ar_coefficient", 0.95)), decoder_anchor=True,
        decoder_mean_zero_init=True)


def _command(model, seed, phase):
    return [sys.executable, "-m", "GEFcom2014.forecast_quality.opsd_cvae_comparison",
            phase, "--model", model, "--seed", str(seed)]


def freeze_protocol():
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite existing comparison directory: %s" % OUT)
    OUT.mkdir(parents=True)
    (OUT / "configs").mkdir()
    (OUT / "code_snapshot").mkdir()
    (OUT / "logs").mkdir()
    arrays, metadata, dates = _date_info()
    if not np.all(np.isfinite(arrays["x_ls"])) or not np.all(np.isfinite(arrays["y_ls"])):
        raise ValueError("Invalid LS arrays")
    standard, matched, bm = _model_configs().values()
    bm["cond_in"] = int(arrays["x_ls"].shape[1] + 24)
    bm["in_size"] = int(arrays["y_ls"].shape[1])
    bm["actual_graph_edges"] = None
    with OLD_FA_CHECKPOINT.open("rb") as handle:
        old_model = pickle.load(handle)
    graph_mask = old_model.graph_mask.detach().cpu().numpy().astype(np.float32)
    bm["actual_graph_edges"] = int(np.count_nonzero(np.triu(graph_mask, 1)))
    bm["graph_mask_sha256"] = hashlib.sha256(graph_mask.astype("<f4", copy=False).tobytes()).hexdigest()
    standard_model = ConditionalVAE(arrays["x_ls"].shape[1], 24, standard["latent_dim"], standard["hidden_dim"], standard["layers"])
    fa_model = ForecastAnchoredCVAE(arrays["x_ls"].shape[1] + 24, 24, matched["latent_dim"], matched["hidden_dim"], matched["layers"])
    bm_model = _build_bm_model(bm, graph_mask, torch.device("cpu"))
    model_parameters = {"standard_cvae": _parameter_count(standard_model),
                        "fa_cvae": _parameter_count(fa_model), "fa_bm_vae": _parameter_count(bm_model)}
    config_values = {"standard_cvae": standard, "fa_cvae": matched, "fa_bm_vae": bm}
    config_files, code_files = {}, {}
    for name, config in config_values.items():
        path = OUT / "configs" / (name + ".json")
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        config_files[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    source_paths = [
        Path(__file__), ROOT / "GEFcom2014" / "models" / "neso_extra_baselines.py",
        ROOT / "GEFcom2014" / "models" / "probabilistic_baselines.py",
        ROOT / "GEFcom2014" / "models" / "QBM_VAE" / "qbm_vae.py",
        ROOT / "GEFcom2014" / "models" / "QBM_VAE" / "utils_qbm_vae.py",
        ROOT / "GEFcom2014" / "models" / "utils_models.py",
        ROOT / "GEFcom2014" / "external_datasets.py",
        ROOT / "GEFcom2014" / "forecast_quality" / "compare_scenarios.py",
        ROOT / "GEFcom2014" / "forecast_quality" / "unified_postprocessing.py",
        ROOT / "GEFcom2014" / "forecast_quality" / "opsd_sa_negative_evaluation.py",
    ]
    for path in source_paths:
        destination = OUT / "code_snapshot" / path.name
        shutil.copy2(path, destination)
        code_files[str(path.relative_to(ROOT))] = {"snapshot": str(destination.relative_to(ROOT)), "sha256": sha(path)}
    acceptance = json.loads(OLD_ACCEPTANCE.read_text(encoding="utf-8"))
    if acceptance.get("acceptance") != "LIMITED_ACCEPTANCE":
        raise ValueError("Unexpected historical provenance acceptance state")
    protocol = {
        "protocol_version": "opsd-cvae-comparison-v1",
        "status": "FROZEN_BEFORE_NEW_TRAINING",
        "artifact_role": "post_TEST_viewed_appendix_research; new training provenance; not a correction of historical metadata",
        "research_context": {"existing_OPSD_TEST_results_previously_viewed": True,
                             "historical_provenance_acceptance": "LIMITED_ACCEPTANCE",
                             "historical_FA_BM_test_selection": "NOT_RECORDED; not inferred from defaults"},
        "dataset": {"path": str(BUNDLE.resolve()), "sha256": sha(BUNDLE),
                    "metadata_path": str(BUNDLE.with_suffix(".json").resolve()),
                    "metadata_sha256": sha(BUNDLE.with_suffix(".json")),
                    "source_metadata": metadata, "dates": {s: list(map(str, dates[s])) for s in ("ls", "vs", "test")},
                    "split_days": {s: len(dates[s]) for s in ("ls", "vs", "test")},
                    "test_dates": [str(dates["test"][0]), str(dates["test"][-1])],
                    "power_units": "normalized OPSD power in [0,1]; LS maximum 14,440 MW; original bundle clipping retained",
                    "input_columns": metadata["context_features"], "target_columns": metadata["target_features"],
                    "scale_fit_split": "LS only", "scenarios_per_date": 100, "hours_per_date": 24,
                    "postprocess": "inverse LS StandardScaler, then clip normalized output to [0,1]"},
        "models": config_values,
        "parameter_counts": model_parameters,
        "training": {"seeds": list(SEEDS), "test_arrays_available_to_fit_or_selection": False,
                     "test_loading_stage": "after all nine best checkpoints and VS scenarios are frozen",
                     "split_selection": "fixed existing LS/VS/TEST dates; no date changes",
                     "generation_seed": "10000 + training_seed + split offset; VS=0, TEST=1000003",
                     "selection": {"standard_cvae": "minimum fixed-seed VS negative ELBO",
                                   "fa_cvae": "minimum fixed-seed VS Gaussian reconstruction NLL; KL excluded from selection score",
                                   "fa_bm_vae": "minimum VS Gaussian reconstruction NLL"}},
        "metrics": {"primary": "Raw CRPS_raw", "supplement": "Cal; calibrator fit on each model/seed VS scenarios and targets only",
                    "metrics": list(METRICS), "statistical_unit": "forecast date",
                    "per_seed_first": True, "paired_block_bootstrap": {"block_days": 7, "repetitions": 5000,
                                                                           "seed_base": 20261025,
                                                                           "primary_comparisons": ["FA-BM-VAE - Standard CVAE", "FA-BM-VAE - FA-CVAE"],
                                                                           "holm_family": "two primary Raw CRPS comparisons"}},
        "comparability_note": "FA-CVAE vs FA-BM-VAE matches context/anchor, 48 latent dimensions, 2x256 encoder/decoder, diagonal Gaussian observation scale range, epoch cap, batch size, optimizer, VS reconstruction selection, and 0.05/20-epoch regularization schedule. Continuous Gaussian KL and binary Ising prior objectives are not identical; this does not isolate graph coupling alone. J=0 is the direct coupling ablation.",
        "provenance": {"historical_acceptance_path": str(OLD_ACCEPTANCE.relative_to(ROOT)),
                       "historical_acceptance_sha256": sha(OLD_ACCEPTANCE),
                       "old_seed0_checkpoint_sha256": sha(OLD_FA_CHECKPOINT),
                       "old_seed0_metadata_sha256": sha(OLD_FA_META),
                       "new_FA_BM_graph_edges_from_checkpoint": bm["actual_graph_edges"],
                       "new_FA_BM_graph_mask_sha256": bm["graph_mask_sha256"],
                       "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                       "git_status_at_freeze": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout},
        "config_files": config_files, "code_snapshot": code_files,
    }
    protocol_path = OUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    protocol["protocol_sha256"] = sha(protocol_path)
    manifest = {"protocol_sha256": protocol["protocol_sha256"], "status": "FROZEN",
                "smoke": [{"model": m, "seed": 0, "status": "NOT RUN", "command": _command(m, 0, "smoke")}
                          for m in MODELS],
                "tasks": [{"model": m, "seed": s, "status": "NOT RUN",
                           "training_seed": s, "generation_seed": 10000 + s,
                           "vs_generation_seed": 10000 + s,
                           "test_generation_seed": 10000 + s + SPLIT_OFFSETS["TEST"],
                           "n_scenarios": 100, "output_dir": str((OUT / "formal" / m / ("seed_%d" % s)).resolve()),
                           "command": _command(m, s, "train")}
                          for m in MODELS for s in SEEDS],
                "evaluation": {"status": "NOT RUN", "command": _command("all", 0, "generate-test") + ["--evaluate"]}}
    save_manifest(manifest)
    (OUT / "provenance_acceptance_source.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    (OUT / "freeze_commands.txt").write_text("\n".join(" ".join(_command(m, s, phase)) for phase, m, s in
        [("smoke", m, 0) for m in MODELS] + [("train", m, s) for m in MODELS for s in SEEDS]) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_sha256": protocol["protocol_sha256"], "parameter_counts": model_parameters,
                      "data_sha256": protocol["dataset"]["sha256"], "formal_tasks": len(manifest["tasks"])}, indent=2))


def _load_fit_arrays(model):
    # Deliberately never access x_test/y_test during training or VS selection.
    with np.load(BUNDLE, allow_pickle=False) as data:
        return tuple(np.asarray(data[k], dtype=np.float32) for k in ("x_ls", "y_ls", "x_vs", "y_vs"))


def _train_task(model_name, seed, phase="formal"):
    model_label = LABELS[model_name]
    output = OUT / phase / model_name / ("seed_%d" % seed)
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "training.log"
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
            config = _model_configs()[model_name]
            smoke = phase == "smoke"
            n_scenarios = 5 if smoke else 100
            epochs = 1 if smoke else int(config.get("epochs", config.get("training_epochs", 80)))
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            torch.set_num_threads(4)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x_ls, y_ls, x_vs, y_vs = _load_fit_arrays(model_name)
            arrays, x_scaler, y_scaler = _scale_ls_vs(x_ls, y_ls, x_vs, y_vs, model_name)
            args = SimpleNamespace(seed=seed, epochs=epochs, batch_size=int(config.get("batch_size", 256)),
                                   learning_rate=float(config.get("learning_rate", 3e-4)),
                                   weight_decay=float(config.get("weight_decay", 1e-5)),
                                   patience=int(config.get("patience", config.get("early_stopping_patience", 20))),
                                   min_delta=float(config.get("min_delta", 1e-4)), grad_clip=float(config.get("grad_clip", 5.0)),
                                   device=device)
            history, selection_score, fit_seconds = [], None, 0.0
            if model_name == "standard_cvae":
                model = ConditionalVAE(arrays["x_ls"].shape[1], 24, int(config["latent_dim"]),
                                       int(config["hidden_dim"]), int(config["layers"])).to(device)
                loader = pb.make_loader(arrays["x_ls"], arrays["y_ls"], args.batch_size, seed)
                opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                started_fit = time.perf_counter()
                history, best_state = pb.fit_with_validation(model, loader, arrays, model.loss, opt, args, device)
                fit_seconds = time.perf_counter() - started_fit
                model.load_state_dict(best_state)
                selection_score = min(row["validation_loss"] for row in history)
                best_epoch = min(history, key=lambda row: row["validation_loss"])["epoch"]
                checkpoint = {"state_dict": model.state_dict(), "model": model_name,
                              "context_dim": arrays["x_ls"].shape[1], "data_dim": 24,
                              "latent_dim": int(config["latent_dim"]), "hidden_dim": int(config["hidden_dim"]),
                              "layers": int(config["layers"])}
                selection_metric = "VS negative ELBO"
            elif model_name == "fa_cvae":
                model = ForecastAnchoredCVAE(arrays["x_ls"].shape[1], 24, int(config["latent_dim"]),
                                             int(config["hidden_dim"]), int(config["layers"])).to(device)
                started_fit = time.perf_counter()
                history, selection_score = _fit_fa_cvae(model, arrays, args, float(config["kl_weight"]),
                                                        int(config["warmup_epochs"]))
                fit_seconds = time.perf_counter() - started_fit
                best_epoch = int(model.selection_epoch)
                checkpoint = {"state_dict": model.state_dict(), "model": model_name,
                              "context_dim": arrays["x_ls"].shape[1], "data_dim": 24,
                              "latent_dim": int(config["latent_dim"]), "hidden_dim": int(config["hidden_dim"]),
                              "layers": int(config["layers"]), "min_logvar": model.min_logvar,
                              "max_logvar": model.max_logvar, "min_log_std": model.min_log_std,
                              "max_log_std": model.max_log_std}
                selection_metric = "VS Gaussian reconstruction NLL"
            else:
                config = _get_fa_bm_config()
                config["cond_in"], config["in_size"] = arrays["x_ls"].shape[1], 24
                graph_mask = load_frozen_graph_mask(OLD_FA_CHECKPOINT, int(config["latent_s"]))
                model = _build_bm_model(config, graph_mask, device)
                opt = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                                       weight_decay=float(config["weight_decay"]))
                negative_sampler = build_sampler("sa", seed=seed,
                    sweeps=int(config["negative_sampler_sweeps"]), beta_start=0.1,
                    beta_end=float(config["beta"]))
                started_fit = time.perf_counter()
                loss, best_model, _ = fit_qbm_vae(
                    nb_epoch=epochs, x_LS=arrays["x_ls"], y_LS=arrays["y_ls"],
                    x_VS=arrays["x_vs"], y_VS=arrays["y_vs"], x_TEST=None, y_TEST=None,
                    model=model, opt=opt, sampler=None, gpu=device.type == "cuda",
                    qbm_weight=float(config["qbm_weight"]), batch_size=int(config["batch_size"]),
                    warmup_epochs=int(config["warmup_epochs"]), negative_steps=int(config["negative_steps"]),
                    beta=float(config["beta"]), early_stopping_patience=int(config["early_stopping_patience"]),
                    temperature_start=float(config["temperature_start"]),
                    temperature_end=float(config["temperature_end"]), grad_clip=5.0, seed=seed,
                    evaluate_test_during_training=False, negative_sampler=negative_sampler,
                    negative_num_reads=int(config["negative_num_reads"]))
                fit_seconds = time.perf_counter() - started_fit
                model = best_model
                history = [{"epoch": i, "training_loss": float(row[0]), "vs_total": float(row[1]),
                            "test_loss": None} for i, row in enumerate(loss)]
                best_epoch = int(model.selection_epoch)
                selection_score = float(min(row["vs_total"] for row in history))
                selection_metric = "VS Gaussian reconstruction NLL (negative phase disabled for validation)"
                checkpoint = model
            with (output / "checkpoint.pickle").open("wb") as handle:
                pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with (output / "scalers.pickle").open("wb") as handle:
                pickle.dump({"context": x_scaler, "target": y_scaler}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if model_name == "fa_bm_vae":
                samples_vs = _generate_bm(model, arrays["x_vs"], y_scaler, seed, "VS", n_scenarios, device, config)
            elif model_name == "standard_cvae":
                raw = sample_cvae(model, arrays["x_vs"], n_scenarios, device, 10000 + seed, int(config.get("day_batch_size", 4)))
                samples_vs = _inverse_clip(raw, y_scaler, n_scenarios)
            else:
                gen = torch.Generator(device=device).manual_seed(10000 + seed)
                raw = model.sample(arrays["x_vs"], n_scenarios, gen, int(config.get("day_batch_size", 50)))
                samples_vs = _inverse_clip(raw, y_scaler, n_scenarios)
            if samples_vs.shape != (len(y_vs) * 24, n_scenarios) or not np.isfinite(samples_vs).all():
                raise ValueError("Invalid VS scenario shape or non-finite values")
            pb.dump_pickle(output / ("scenarios_%d_VS.pickle" % n_scenarios), samples_vs)
            metadata = {"model": model_name, "label": model_label, "seed": seed,
                        "training_seed": seed, "generation_seed": 10000 + seed,
                        "generation_split_seeds": {"VS": 10000 + seed, "TEST": 10000 + seed + SPLIT_OFFSETS["TEST"]},
                        "dataset_bundle": str(BUNDLE.resolve()), "dataset_sha256": sha(BUNDLE),
                        "fit_split": "LS", "selection_split": "VS", "test_used_for_selection": False,
                        "test_arrays_loaded_during_fit": False, "test_arrays_passed_to_selection": False,
                        "selection_metric": selection_metric, "selection_score": selection_score,
                        "best_epoch": int(best_epoch), "epochs_run": len(history), "history": history,
                        "parameters": _parameter_count(model), "fit_seconds": fit_seconds,
                        "vs_sampling_seconds": None, "n_scenarios": n_scenarios,
                        "dates_vs": _vs_dates(), "checkpoint_sha256": sha(output / "checkpoint.pickle"),
                        "scalers_sha256": sha(output / "scalers.pickle"),
                        "vs_scenarios_sha256": sha(output / ("scenarios_%d_VS.pickle" % n_scenarios)),
                        "device": str(device), "config": config}
            atomic_json(output / "metadata.json", metadata)
        elapsed = time.perf_counter() - started
        return {"status": "COMPLETED", "elapsed_seconds": elapsed,
                "metadata": str((output / "metadata.json").relative_to(ROOT)),
                "metadata_sha256": sha(output / "metadata.json"), "log": str(log_path.relative_to(ROOT))}
    except Exception as exc:
        atomic_json(output / "failure.json", {"status": "FAILED", "error": repr(exc), "elapsed_seconds": time.perf_counter() - started})
        raise


def _vs_dates():
    with np.load(BUNDLE, allow_pickle=False) as data:
        return np.asarray(data["dates_vs"]).astype(str).tolist()


def _inverse_clip(samples, scaler, n_scenarios):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != n_scenarios or values.shape[2] != 24:
        raise ValueError("VAE sampler must return (days, scenarios, 24)")
    inverse = scaler.inverse_transform(values.reshape(-1, 24)).reshape(values.shape)
    inverse = np.clip(inverse, 0.0, 1.0)
    return inverse.transpose(0, 2, 1).reshape(-1, n_scenarios)


def _generate_bm(model, context, y_scaler, seed, split, n_scenarios, device, config):
    split_seed = 10000 + int(seed) + SPLIT_OFFSETS[split]
    sampler = build_sampler("sa", sweeps=200, beta_start=0.1, beta_end=2.0, seed=split_seed)
    generator = torch.Generator(device=device).manual_seed(split_seed)
    model.device = device
    model.to(device)
    matrix = build_qbm_vae_scenarios(n_s=n_scenarios, x=context, y_scaler=y_scaler,
        model=model, sampler=sampler, max=1.0, gpu=device.type == "cuda", tag="opsd-wind",
        non_null_indexes=None, beta=float(config.get("beta", 1.0)), observation_noise=True,
        gibbs_steps=int(config.get("negative_steps", 5)), day_batch_size=50, generator=generator)
    return np.asarray(matrix, dtype=np.float64)


def _load_checkpoint(model_name, task_dir, context_dim, device):
    path = task_dir / "checkpoint.pickle"
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if model_name == "fa_bm_vae":
        model = checkpoint
        model.device = device
        model.to(device)
        model.eval()
    elif model_name == "standard_cvae":
        config = _model_configs()[model_name]
        model = ConditionalVAE(context_dim, 24, int(config["latent_dim"]), int(config["hidden_dim"]), int(config["layers"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
    else:
        config = _model_configs()[model_name]
        model = ForecastAnchoredCVAE(context_dim, 24, int(config["latent_dim"]), int(config["hidden_dim"]), int(config["layers"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
    return model


def _inference_context(raw_x, scalers, model_name):
    x = scalers["context"].transform(raw_x)
    if model_name in ("fa_cvae", "fa_bm_vae"):
        anchor = scalers["target"].transform(raw_x[:, :24])
        x = np.concatenate((x, anchor), axis=1)
    return x.astype(np.float32)


def run_smoke():
    manifest = read_manifest()
    if manifest["status"] not in ("FROZEN", "SMOKE_INTERRUPTED"):
        raise ValueError("Smoke requires a frozen protocol and no failed smoke task")
    results = []
    for item in manifest["smoke"]:
        model = item["model"]
        task_dir = OUT / "smoke" / model / "seed_0"
        if item["status"] == "COMPLETED":
            result = item["result"]
        elif task_dir.exists():
            # Resume only from a fully written, identity-matching local smoke artifact.
            metadata_path = task_dir / "metadata.json"
            scenario_path = task_dir / "scenarios_5_VS.pickle"
            checkpoint_path = task_dir / "checkpoint.pickle"
            if not all(p.is_file() for p in (metadata_path, scenario_path, checkpoint_path)):
                raise RuntimeError("Existing partial smoke output; refusing to retrain or overwrite: %s" % task_dir)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata.get("model") != model or metadata.get("seed") != 0 or
                    metadata.get("dataset_sha256") != sha(BUNDLE) or metadata.get("n_scenarios") != 5 or
                    metadata.get("fit_split") != "LS" or metadata.get("selection_split") != "VS"):
                raise RuntimeError("Existing smoke artifact identity mismatch: %s" % task_dir)
            result = {"status": "REUSED_VALIDATED_SMOKE", "metadata": str(metadata_path.relative_to(ROOT)),
                      "metadata_sha256": sha(metadata_path), "log": str((task_dir / "training.log").relative_to(ROOT))}
        elif item["status"] == "NOT RUN":
            item["status"] = "RUNNING"
            manifest["status"] = "SMOKE_IN_PROGRESS"
            save_manifest(manifest)
            try:
                result = _train_task(model, 0, "smoke")
            except Exception as exc:
                item.update(status="FAILED_NO_RETRY", failure=repr(exc))
                manifest.update(status="SMOKE_INTERRUPTED", failure=repr(exc))
                save_manifest(manifest)
                raise
        else:
            raise RuntimeError("Smoke task is not eligible for execution or verified reuse: %s" % item["status"])
        arrays, dates = _load_vs_evaluation_arrays()
        vs = load_scenarios(task_dir / "scenarios_5_VS.pickle", len(arrays["y_vs"]) * 24)
        metrics = evaluate_model_daily(vs, arrays["y_vs"], max_s=5)
        if any(not np.isfinite(v).all() for v in metrics.values()):
            raise ValueError("Smoke metrics contain non-finite values")
        calibrator = HourlyAffineCalibrator(limits=(0.0, 1.0)).fit(vs, arrays["y_vs"].reshape(-1))
        cal = calibrator.transform(vs)
        if cal.shape != vs.shape or not np.isfinite(cal).all():
            raise ValueError("Smoke calibration failed")
        item.update(status="COMPLETED")
        results.append({"model": model, **result, "scenario_shape": list(vs.shape),
                        "metrics_finite": True, "vs_only": True, "calibration_finite": True,
                        "date_count": len(dates["vs"]), "test_arrays_read": False})
        item["result"] = results[-1]
        manifest["status"] = "SMOKE_IN_PROGRESS"
        save_manifest(manifest)
    manifest["status"] = "SMOKE_PASSED"
    save_manifest(manifest)
    atomic_json(OUT / "smoke_validation.json", {"status": "PASSED", "test_arrays_read": False,
        "results": results, "protocol_sha256": manifest["protocol_sha256"]})
    print(json.dumps({"status": "SMOKE_PASSED", "tasks": len(results)}, indent=2))


def run_formal():
    manifest = read_manifest()
    if manifest["status"] != "SMOKE_PASSED":
        raise ValueError("Formal training is blocked until all LS/VS smoke checks pass")
    for entry in manifest["tasks"]:
        if entry["status"] != "NOT RUN":
            raise ValueError("Formal task is not in the pristine NOT RUN state; refusing retry/overwrite")
    manifest["status"] = "FORMAL_TRAINING"
    save_manifest(manifest)
    try:
        for i, entry in enumerate(manifest["tasks"]):
            entry["status"] = "RUNNING"
            entry["command"] = _command(entry["model"], entry["seed"], "train")
            save_manifest(manifest)
            result = _train_task(entry["model"], int(entry["seed"]), "formal")
            entry.update(status="COMPLETED", result=result)
            save_manifest(manifest)
        manifest["status"] = "ALL_NINE_TRAINED_VS_SELECTED"
        save_manifest(manifest)
    except Exception as exc:
        manifest["status"] = "FAILED_STOPPED_NO_RETRY"
        manifest["failure"] = repr(exc)
        save_manifest(manifest)
        raise
    print(json.dumps({"status": manifest["status"], "tasks": len(manifest["tasks"])}, indent=2))


def generate_test_and_evaluate():
    manifest = read_manifest()
    if manifest["status"] in ("ALL_TEST_SCENARIOS_GENERATED", "EVALUATING"):
        # Evaluation can be resumed only from the already verified, immutable scenario files.
        for entry in manifest["tasks"]:
            task_dir = OUT / "formal" / entry["model"] / ("seed_%d" % int(entry["seed"]))
            test_path = task_dir / "scenarios_100_TEST.pickle"
            expected = entry.get("test_generation", {}).get("scenario_sha256")
            if not test_path.is_file() or not expected or sha(test_path) != expected:
                raise RuntimeError("Cannot resume evaluation: TEST scenario identity/hash mismatch: %s" % test_path)
        evaluate_all()
        return
    if manifest["status"] != "ALL_NINE_TRAINED_VS_SELECTED":
        raise ValueError("TEST generation requires all nine checkpoints to be VS-selected and frozen")
    arrays, metadata, dates = _date_info()
    with np.load(BUNDLE, allow_pickle=False) as data:
        # TEST arrays are opened only here, after every checkpoint's selection is complete.
        raw_x_test = np.asarray(data["x_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=np.float32)
        dates_test = np.asarray(data["dates_test"]).astype(str).tolist()
    if dates_test != list(map(str, dates["test"])) or len(y_test) != 90:
        raise ValueError("TEST date identity changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for entry in manifest["tasks"]:
        model_name, seed = entry["model"], int(entry["seed"])
        task_dir = OUT / "formal" / model_name / ("seed_%d" % seed)
        test_path = task_dir / "scenarios_100_TEST.pickle"
        if test_path.exists():
            raise FileExistsError("Refusing to overwrite TEST scenarios: %s" % test_path)
        with (task_dir / "scalers.pickle").open("rb") as handle:
            scalers = pickle.load(handle)
        context = _inference_context(raw_x_test, scalers, model_name)
        model = _load_checkpoint(model_name, task_dir, context.shape[1], device)
        start = time.perf_counter()
        if model_name == "fa_bm_vae":
            config = _get_fa_bm_config()
            samples = _generate_bm(model, context, scalers["target"], seed, "TEST", 100, device, config)
        elif model_name == "standard_cvae":
            samples_raw = sample_cvae(model, context, 100, device, 10000 + seed + SPLIT_OFFSETS["TEST"],
                                      int(_model_configs()[model_name].get("day_batch_size", 4)))
            samples = _inverse_clip(samples_raw, scalers["target"], 100)
        else:
            generation_seed = 10000 + seed + SPLIT_OFFSETS["TEST"]
            rng = torch.Generator(device=device).manual_seed(generation_seed)
            samples_raw = model.sample(context, 100, rng, 50)
            samples = _inverse_clip(samples_raw, scalers["target"], 100)
        seconds = time.perf_counter() - start
        if samples.shape != (90 * 24, 100) or not np.isfinite(samples).all():
            raise ValueError("Invalid TEST scenario dimensions or non-finite values")
        pb.dump_pickle(test_path, samples)
        meta_path = task_dir / "metadata.json"
        task_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task_meta["test_sampling_seconds"] = seconds
        task_meta["dates_test"] = dates_test
        task_meta["test_scenarios_sha256"] = sha(test_path)
        task_meta["test_target_loaded_after_checkpoint_selection"] = True
        atomic_json(meta_path, task_meta)
        entry["test_generation"] = {"status": "COMPLETED", "seconds": seconds,
                                    "scenario_sha256": sha(test_path), "shape": list(samples.shape)}
        save_manifest(manifest)
    manifest["status"] = "ALL_TEST_SCENARIOS_GENERATED"
    save_manifest(manifest)
    evaluate_all()


def _metric_record(model, seed, split, variant, date, values):
    return {"model": LABELS[model], "model_id": model, "seed": seed, "split": split,
            "variant": variant, "date": str(date), "n_scenarios": 100,
            **{key: float(np.asarray(value)[0]) if np.asarray(value).ndim else float(value)
               for key, value in values.items()}}


def evaluate_all():
    manifest = read_manifest()
    if manifest["status"] not in ("ALL_TEST_SCENARIOS_GENERATED", "EVALUATING"):
        raise ValueError("Evaluation requires all formal TEST scenarios")
    manifest["status"] = "EVALUATING"
    save_manifest(manifest)
    arrays, _, dates = load_daily_bundle(BUNDLE)
    per_date, calibrations = [], []
    raw_arrays = {}
    for model in MODELS:
        for seed in SEEDS:
            task_dir = OUT / "formal" / model / ("seed_%d" % seed)
            for split in ("VS", "TEST"):
                target = arrays["y_" + split.lower()]
                scenario_path = task_dir / ("scenarios_100_%s.pickle" % split)
                raw = load_scenarios(scenario_path, len(target) * 24)
                if raw.shape != (len(target) * 24, 100):
                    raise ValueError("Scenario count/date shape mismatch: %s" % scenario_path)
                if split == "VS":
                    validation = raw
                    calibrator = HourlyAffineCalibrator(limits=(0.0, 1.0)).fit(validation, target.reshape(-1))
                    cal = calibrator.transform(raw)
                    calibrations.append({"model": LABELS[model], "model_id": model, "seed": seed,
                                         "fit_split": "VS", "spread": calibrator.parameters.spread,
                                         "hourly_bias": calibrator.parameters.hourly_bias,
                                         "zero_hours": calibrator.parameters.zero_hours,
                                         "harmonics": calibrator.parameters.harmonics,
                                         "ridge": calibrator.parameters.ridge})
                else:
                    vs_path = task_dir / "scenarios_100_VS.pickle"
                    vs = load_scenarios(vs_path, len(arrays["y_vs"]) * 24)
                    cal, cal_info = _calibrate(vs, arrays["y_vs"], raw)
                    cal_path = task_dir / "scenarios_100_TEST_Cal.pickle"
                    if cal_path.exists():
                        saved_cal = load_scenarios(cal_path, len(target) * 24)
                        if not np.allclose(saved_cal, cal, rtol=0.0, atol=1e-12):
                            raise RuntimeError("Existing Cal scenario differs; refusing overwrite: %s" % cal_path)
                    else:
                        cal_path.write_bytes(pickle.dumps(cal, protocol=pickle.HIGHEST_PROTOCOL))
                    cal_info_path = task_dir / "calibration.json"
                    if cal_info_path.exists():
                        old_info = json.loads(cal_info_path.read_text(encoding="utf-8"))
                        if old_info != cal_info:
                            raise RuntimeError("Existing calibration metadata differs; refusing overwrite")
                    else:
                        atomic_json(cal_info_path, cal_info)
                for variant, values in (("Raw", raw), ("Cal", cal)):
                    metrics = _daily_metrics(values, target)
                    labels = dates[split.lower()]
                    if len(labels) != len(target):
                        raise ValueError("Date/target length mismatch")
                    for i, date in enumerate(labels):
                        row = {"model": LABELS[model], "model_id": model, "seed": seed,
                               "split": split, "variant": variant, "date": str(date),
                               "n_scenarios": 100}
                        row.update({metric: float(np.asarray(result)[i]) for metric, result in metrics.items()})
                        per_date.append(row)
                    if split == "TEST":
                        raw_arrays[(model, seed, variant)] = {
                            metric: np.asarray(value, dtype=float) for metric, value in metrics.items()}
    _write_csv(OUT / "metrics_daily.csv", per_date)
    _write_csv(OUT / "calibration_parameters.csv", calibrations)
    _write_seed_tables(per_date)
    _write_paired(per_date)
    _write_main_tables(per_date)
    report = _build_report(per_date, calibrations)
    (OUT / "REPORT_zh.md").write_text(report, encoding="utf-8")
    manifest = read_manifest()
    manifest["status"] = "COMPLETED"
    manifest["outputs"] = {name: sha(OUT / name) for name in ("metrics_daily.csv", "metrics_per_seed.csv",
        "summary.csv", "paired_statistics.csv", "main_table_raw.csv", "main_table_cal.csv",
        "REPORT_zh.md", "calibration_parameters.csv")}
    save_manifest(manifest)
    print(json.dumps({"status": "COMPLETED", "daily_metric_rows": len(per_date),
                      "report": str((OUT / "REPORT_zh.md").relative_to(ROOT))}, indent=2))


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        raise ValueError("Refusing to write an empty result: %s" % path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _write_seed_tables(rows):
    test = [r for r in rows if r["split"] == "TEST"]
    per_seed = []
    group_keys = sorted({(r["model"], r["model_id"], r["seed"], r["variant"]) for r in test})
    for model, mid, seed, variant in group_keys:
        subset = [r for r in test if (r["model"], r["model_id"], r["seed"], r["variant"]) == (model, mid, seed, variant)]
        if len(subset) != 90 or len({r["date"] for r in subset}) != 90:
            raise ValueError("Incomplete TEST dates in per-seed metrics")
        for metric in METRICS:
            per_seed.append({"model": model, "model_id": mid, "seed": seed, "variant": variant,
                             "metric": metric, "mean": float(np.mean([r[metric] for r in subset])),
                             "n_test_dates": 90, "n_scenarios_per_date": 100})
    summary = []
    for model, mid, variant, metric in sorted({(r["model"], r["model_id"], r["variant"], r["metric"]) for r in per_seed}):
        values = [r["mean"] for r in per_seed if (r["model_id"], r["variant"], r["metric"]) == (mid, variant, metric)]
        if len(values) != 3:
            raise ValueError("Summary requires all three seeds")
        summary.append({"model": model, "model_id": mid, "variant": variant, "metric": metric,
                        "n_seeds": 3, "mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1)),
                        "statistical_unit": "seed means over 90 forecast dates"})
    _write_csv(OUT / "metrics_per_seed.csv", per_seed)
    _write_csv(OUT / "summary.csv", summary)


def _write_paired(rows):
    test = [r for r in rows if r["split"] == "TEST"]
    seed_differences, paired = [], []
    for a, b in PRIMARY_COMPARISONS:
        for variant in VARIANTS:
            for metric in METRICS:
                arrays_by_seed = {}
                for model in (a, b):
                    for seed in SEEDS:
                        cells = sorted([r for r in test if r["model_id"] == model and r["seed"] == seed and r["variant"] == variant], key=lambda r: r["date"])
                        if len(cells) != 90:
                            raise ValueError("Pair comparison has incomplete TEST dates")
                        arrays_by_seed[(model, seed)] = np.asarray([r[metric] for r in cells], dtype=float)
                for seed in SEEDS:
                    diffs = arrays_by_seed[(a, seed)] - arrays_by_seed[(b, seed)]
                    seed_differences.append({"model_a": LABELS[a], "model_b": LABELS[b],
                        "comparison": "FA-BM-VAE minus comparator", "seed": seed,
                        "variant": variant, "metric": metric, "mean_difference": float(diffs.mean()),
                        "n_paired_dates": 90})
                date_diff = np.mean([arrays_by_seed[(a, s)] - arrays_by_seed[(b, s)] for s in SEEDS], axis=0)
                seed_num = 20261025 + len(paired)
                result = block_paired_inference(date_diff, block_length=7, repetitions=5000, seed=seed_num)
                paired.append({"model_a": LABELS[a], "model_b": LABELS[b],
                    "comparison": "FA-BM-VAE minus comparator", "variant": variant,
                    "metric": metric, "point_estimate": float(date_diff.mean()),
                    "relative_difference_percent": float(100 * date_diff.mean() / max(abs(np.mean([arrays_by_seed[(b, s)].mean() for s in SEEDS])), 1e-12)),
                    "statistical_unit": "90 paired forecast dates; three seeds averaged within date",
                    "scenario_reads_are_independent": False, "bootstrap_seed": seed_num,
                    **result})
    primary = [r for r in paired if r["variant"] == "Raw" and r["metric"] == "CRPS_raw"]
    adjusted = holm_adjust([r["block_sign_flip_p"] for r in primary])
    for row, p in zip(primary, adjusted):
        row["holm_p"] = float(p)
        row["holm_family"] = "two predeclared Raw CRPS comparisons"
    _write_csv(OUT / "paired_differences_by_seed.csv", seed_differences)
    _write_csv(OUT / "paired_statistics.csv", paired)


def _write_main_tables(rows):
    for variant in VARIANTS:
        summary = [r for r in read_csv(OUT / "summary.csv") if r["variant"] == variant]
        models = []
        for model in MODELS:
            row = {"model": LABELS[model]}
            for metric in METRICS:
                found = next(x for x in summary if x["model_id"] == model and x["metric"] == metric)
                row[metric + "_mean"] = float(found["mean"])
                row[metric + "_seed_std"] = float(found["seed_std"])
            models.append(row)
        stem = "main_table_" + variant.lower()
        _write_csv(OUT / (stem + ".csv"), models)
        columns = ["model"] + [metric + " (mean ± seed SD)" for metric in METRICS]
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in models:
            values = [row["model"]] + ["%.4f ± %.4f" % (row[m + "_mean"], row[m + "_seed_std"]) for m in METRICS]
            lines.append("| " + " | ".join(values) + " |")
        (OUT / (stem + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_report(rows, calibrations):
    test_rows = [r for r in rows if r["split"] == "TEST" and r["variant"] == "Raw"]
    means = {model: {metric: float(np.mean([r[metric] for r in test_rows if r["model_id"] == model]))
                     for metric in METRICS} for model in MODELS}
    paired = read_csv(OUT / "paired_statistics.csv")
    primary = [r for r in paired if r["variant"] == "Raw" and r["metric"] == "CRPS_raw"]
    comp_lines = []
    for row in primary:
        comp_lines.append("| %s | %.6f | [%.6f, %.6f] | %.4g | %.4g |" % (
            row["model_b"], float(row["point_estimate"]), float(row["ci_2.5"]), float(row["ci_97.5"]),
            float(row.get("block_sign_flip_p", float("nan"))), float(row.get("holm_p", float("nan")))))
    raw_md = (OUT / "main_table_raw.md").read_text(encoding="utf-8")
    cal_md = (OUT / "main_table_cal.md").read_text(encoding="utf-8")
    runtime_rows = []
    for task in read_manifest()["tasks"]:
        meta_path = OUT / "formal" / task["model"] / ("seed_%d" % task["seed"]) / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        runtime_rows.append({"model": LABELS[task["model"]], "seed": task["seed"],
            "best_epoch": meta["best_epoch"], "epochs_run": meta["epochs_run"],
            "fit_seconds": meta["fit_seconds"], "task_elapsed_seconds": task["result"]["elapsed_seconds"],
            "test_sampling_seconds": meta.get("test_sampling_seconds", 0.0)})
    _write_csv(OUT / "runtime_costs.csv", runtime_rows)
    runtime_text = "\n".join("| {model} | {seed} | {best_epoch} | {epochs_run} | {fit_seconds:.1f} | {task_elapsed_seconds:.1f} | {test_sampling_seconds:.1f} |".format(**r) for r in runtime_rows)
    return f'''# OPSD 条件 VAE 补齐实验报告

## 实验身份与数据

本实验是已查看 OPSD TEST 后的追加研究。旧 FA-BM-VAE provenance 状态为 `LIMITED_ACCEPTANCE`；历史最佳 epoch、原始命令与 TEST 是否参与选择没有直接记录。新结果为独立来源，不回填历史元数据，不覆盖旧 checkpoint，不调用硬件。

冻结 bundle `{BUNDLE.relative_to(ROOT)}`，SHA-256 `{sha(BUNDLE)}`。LS/VS/TEST 天数为 1036/90/90；TEST 日期 `{_test_dates_string()}`。功率按已有 `[0,1]` 规则归一化（LS 最大值 14,440 MW），保留 bundle 原有截断。scaler 只在 LS 拟合。每个模型均使用相同日期、24 小时目标和每日 100 条场景。

## 模型与可解释范围

- **Standard CVAE**：复用 NESO `ConditionalVAE` 的对角高斯后验、高斯解码器、标准正态先验和负 ELBO；使用已冻结 NESO CVAE 配置，按 VS 负 ELBO 选模。
- **FA-CVAE**：使用 LS 缩放的日前风电预测锚点，输出为锚点加残差均值，并拟合条件对角高斯先验/后验与观测噪声。匹配 FA-BM-VAE 的 48 维潜变量、2x256 网络、80 epoch 上限、batch 256、Adam、VS 重建 NLL 选模和 0.05/20 epoch 预热权重。
- **FA-BM-VAE**：按冻结 SA-negative 配置在新目录重训 seed 0/1/2；图从旧 seed-0 checkpoint 加载，实际核验为 {read_protocol()["provenance"]["new_FA_BM_graph_edges_from_checkpoint"]} 条无向边。旧模型不修改。

FA-CVAE 与 FA-BM-VAE 不是只差是否有耦合：连续高斯 KL 和离散 Ising 负相目标不同。本对照回答的是匹配锚点与训练设置下 BM-VAE 是否优于条件高斯 VAE；J=0 才是直接的离散潜变量耦合消融。

## 数据隔离、选模和后处理

训练/VS 选模只读 LS/VS。九个 checkpoint 冻结后才加载 TEST。Standard CVAE 按 VS 负 ELBO 选择，FA-CVAE 与 FA-BM-VAE 按 VS 高斯重建 NLL 选择。每个模型/seed 仅用 VS 场景和真值拟合同一 `HourlyAffineCalibrator(limits=(0,1))`，再变换 TEST。Raw 为主结果，Cal 为补充，本次不含 Cal+ECC。

## Raw 主表

指标单位是 OPSD 归一化功率。PICP90 越接近 0.90 越好，mean bias 越接近 0 越好，其他损失型指标越低越好。先求每个 seed 的逐日平均，表中是跨 seed 平均 ? 样本标准差。

{raw_md}

## Cal 补充表

{cal_md}

## 预先指定主比较：Raw CRPS

差值是 FA-BM-VAE 减比较模型，负值代表 FA-BM-VAE CRPS 较低。先对三 seed 逐日分数取均值，再对 90 个配对日期做 7 日连续块 bootstrap（5000 次）。CI 条件于本次 checkpoint，不代表完整训练随机性；100 reads 不是独立测试样本。两项预先指定主比较的 sign-flip p 值用 Holm 校正。

| 比较模型 | 平均差 | 95% 日期块 CI | sign-flip p | Holm p |
|---|---:|---:|---:|---:|
{chr(10).join(comp_lines)}

## 结果回答

- FA-BM-VAE 对 Standard CVAE：{_verdict(means['fa_bm_vae']['CRPS_raw'], means['standard_cvae']['CRPS_raw'], primary, 'Standard CVAE')}。
- FA-BM-VAE 对 FA-CVAE：{_verdict(means['fa_bm_vae']['CRPS_raw'], means['fa_cvae']['CRPS_raw'], primary, 'FA-CVAE')}。
- 只依据预先指定 Raw CRPS 与日期块 CI 描述优势；区间包含 0 时不称为统计上优于对照。BM 与条件高斯的所有差异不能全部归因于图耦合。

## 运行成本

训练 fit 秒数、单任务总秒数和 TEST 生成时间保存于 `runtime_costs.csv`。

| 模型 | seed | 最佳 epoch | 训练 epoch | fit 秒 | 任务总秒 | TEST 生成秒 |
|---|---:|---:|---:|---:|---:|---:|
{runtime_text}

## 复现与来源

命令顺序为 smoke、train、generate-test，解释器 `D:\\anaconda\\envs\\wsy\\python.exe`。命令、数据和模型哈希保存于 `task_manifest.json` 和各 seed `metadata.json`。训练代码 SHA-256 `{read_manifest().get('tasks', [{}])[0].get('training_code_sha256', 'NOT RECORDED')}`；评价代码 SHA-256 `{read_manifest().get('analysis_code_sha256', 'NOT RECORDED')}`。smoke 与正式结果分开存储。所有数值来自本次独立实验，不与历史结果混用。'''


def _test_dates_string():
    _, _, dates = _date_info()
    return "%s 至 %s（90 日）" % (dates["test"][0], dates["test"][-1])


def _verdict(fa_value, comparator_value, primary_rows, comparator):
    row = next((r for r in primary_rows if r["model_b"] == LABELS.get(comparator, comparator)), None)
    diff = float(fa_value - comparator_value)
    if row is None:
        return "Raw CRPS 点差为 %.5f，统计信息缺失" % diff
    lo, hi = float(row["ci_2.5"]), float(row["ci_97.5"])
    if diff < 0 and hi < 0:
        return "Raw CRPS 更低，配对 95%% 区间不跨 0（差值 %.5f [%.5f, %.5f]）" % (diff, lo, hi)
    if diff < 0:
        return "Raw CRPS 点估计更低，但配对 95%% 区间包含 0（差值 %.5f [%.5f, %.5f]）" % (diff, lo, hi)
    if diff > 0 and lo > 0:
        return "Raw CRPS 更高，配对 95%% 区间不跨 0（差值 %.5f [%.5f, %.5f]）" % (diff, lo, hi)
    return "Raw CRPS 点估计未显示优势，配对 95%% 区间包含 0（差值 %.5f [%.5f, %.5f]）" % (diff, lo, hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("freeze")
    sub.add_parser("smoke")
    sub.add_parser("train")
    sub.add_parser("generate-test")
    args = parser.parse_args()
    if args.phase == "freeze": freeze_protocol()
    elif args.phase == "smoke": run_smoke()
    elif args.phase == "train": run_formal()
    else: generate_test_and_evaluate()


if __name__ == "__main__":
    main()
