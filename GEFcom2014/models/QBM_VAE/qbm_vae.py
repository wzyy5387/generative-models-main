# -*- coding: UTF-8 -*-

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from GEFcom2014 import wind_data, load_data, pv_data
from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.forecast_quality import quantiles_and_evaluation
from GEFcom2014.models import scale_data_multi, plot_loss
from GEFcom2014.models.QBM_VAE import (
    ConditionalQBMVAE,
    build_sampler,
    build_qbm_vae_scenarios,
    fit_qbm_vae,
    stable_temporal_mi_sparse_graph,
    temporal_mi_sparse_graph,
)
from GEFcom2014.models.QBM_VAE.forecast_anchor import (
    DeterministicForecastAnchor,
    fit_deterministic_anchor,
    predict_deterministic_anchor,
)
from GEFcom2014.utils import dump_file
from numpyencoder import NumpyEncoder
from torch.utils.benchmark import timer

GENERATION_SPLIT_SEED_OFFSETS = {"VS": 0, "TEST": 1000003}


def build_generation_seed_metadata(training_seed, generation_seed=None):
    """Return explicit generation seed provenance without changing legacy defaults."""
    explicit = generation_seed is not None
    base_seed = int(training_seed if generation_seed is None else generation_seed)
    split_seeds = (
        {split: base_seed + offset
         for split, offset in GENERATION_SPLIT_SEED_OFFSETS.items()}
        if explicit else None
    )
    return {
        "generation_sampler_seed": base_seed,
        "generation_sampler_seed_role": (
            "base_seed_for_explicit_split_samplers"
            if explicit
            else "legacy_sampler_seed; split seeds are not separately recorded"
        ),
        "generation_seed": int(generation_seed) if explicit else None,
        "generation_seed_explicit": explicit,
        "generation_split_seed_offsets": dict(GENERATION_SPLIT_SEED_OFFSETS),
        "generation_split_seeds": split_seeds,
        "generation_sampler_seeds": dict(split_seeds) if split_seeds is not None else None,
    }


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Train and evaluate a conditional QBM-VAE on GEFCom2014.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON defaults file. Explicit command-line arguments take precedence.",
    )
    parser.add_argument("--tag", default="load")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--sampler",
        default="sa",
        choices=["gibbs", "sa", "path-integral", "cim", "bosonic"],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--latent-s", type=int, default=48)
    parser.add_argument("--enc-w", type=int, default=256)
    parser.add_argument("--enc-l", type=int, default=2)
    parser.add_argument("--dec-w", type=int, default=256)
    parser.add_argument("--dec-l", type=int, default=2)
    parser.add_argument(
        "--decoder-covariance",
        default="diagonal",
        choices=["diagonal", "ar1"],
        help="Observation model: independent Gaussian or interpretable temporal AR(1).",
    )
    parser.add_argument("--max-ar-coefficient", type=float, default=0.95)
    parser.add_argument(
        "--forecast-anchor",
        action="store_true",
        help=(
            "Use the first target-sized raw context block as a fixed decoder mean "
            "anchor and learn a stochastic residual."
        ),
    )
    parser.add_argument(
        "--context-anchor-conditioning-only",
        action="store_true",
        help=(
            "Append the same fixed context anchor to the condition vector, but "
            "do not add it to the decoder output. This is the matched no-anchor-"
            "addition ablation."
        ),
    )
    decoder_init_group = parser.add_mutually_exclusive_group()
    decoder_init_group.add_argument(
        "--decoder-mean-zero-init",
        dest="decoder_mean_zero_init",
        action="store_true",
        default=None,
        help=(
            "Zero-initialize the decoder mean head independently of whether "
            "the context anchor is added to the mean."
        ),
    )
    decoder_init_group.add_argument(
        "--no-decoder-mean-zero-init",
        dest="decoder_mean_zero_init",
        action="store_false",
        help="Disable decoder mean-head zero initialization explicitly.",
    )
    parser.add_argument(
        "--learned-forecast-anchor",
        action="store_true",
        help="Fit an LS/VS-selected deterministic MLP and learn QBM residuals around it.",
    )
    parser.add_argument("--anchor-hidden-dim", type=int, default=256)
    parser.add_argument("--anchor-layers", type=int, default=2)
    parser.add_argument("--anchor-epochs", type=int, default=100)
    parser.add_argument("--anchor-learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor-patience", type=int, default=15)
    parser.add_argument(
        "--frozen-anchor-model",
        type=Path,
        default=None,
        help="Load an existing validation-selected MLP anchor instead of refitting it.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--qbm-weight", type=float, default=0.05)
    parser.add_argument("--trajectory-ensemble-size", type=int, default=4)
    parser.add_argument("--trajectory-energy-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-variogram-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-ramp-weight", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--negative-steps", type=int, default=5)
    parser.add_argument(
        "--negative-phase-backend",
        default="internal",
        choices=["internal", "gibbs", "sa", "exact", "cim", "bosonic"],
        help=(
            "Training-only negative-phase backend. 'internal' preserves the "
            "persistent Gibbs implementation used by existing experiments."
        ),
    )
    parser.add_argument("--negative-num-reads", type=int, default=1)
    parser.add_argument("--negative-sampler-sweeps", type=int, default=20)
    parser.add_argument("--negative-sampler-burn-in", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--transverse-field", type=float, default=0.0)
    parser.add_argument("--trotter-replicas", type=int, default=4)
    parser.add_argument("--learnable-transverse-field", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--graph-mode",
        default="temporal-mi",
        choices=[
            "temporal-mi",
            "stable-residual-mi",
            "stable-residual-random",
            "posterior-pc",
            "posterior-pc-random",
            "full",
            "none",
            "random",
        ],
        help="Latent Ising graph used by the QBM prior.",
    )
    parser.add_argument("--graph-seed", type=int, default=None)
    parser.add_argument(
        "--graph-mask-model",
        type=Path,
        default=None,
        help="Load and freeze graph_mask from an existing QBM-VAE model artifact.",
    )
    parser.add_argument(
        "--graph-mask-file",
        type=Path,
        default=None,
        help="Load and freeze a graph mask from a NumPy .npy artifact.",
    )
    parser.add_argument("--graph-bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--graph-stability-threshold", type=float, default=0.7)
    parser.add_argument("--graph-max-lag", type=int, default=6)
    parser.add_argument(
        "--random-graph-density",
        type=float,
        default=None,
        help=(
            "Exact edge density for --graph-mode random. By default both the "
            "edge count and nonzero weight multiset match the Temporal-MI graph."
        ),
    )
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument(
        "--scenario-splits",
        nargs="+",
        choices=["VS", "TEST"],
        default=["VS", "TEST"],
        help="Generate only these splits. Use VS alone while developing graph candidates.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=None,
                        help="Optional seed independent of training for scenario generation.")
    parser.add_argument("--run-label", default="", help="Optional suffix for model/scenario filenames.")
    parser.add_argument("--zone-embedding-dim", type=int, default=0)
    parser.add_argument("--nb-zones", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-scenarios", action="store_true", help="Train and save the model without sampling scenarios.")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Generate VS scenarios only; legacy default still generates VS and TEST.",
    )
    if config_args.config is not None:
        with config_args.config.open("r", encoding="utf-8") as handle:
            defaults = json.load(handle)
        fixed_generation_settings = {
            "generation_split_seed_offsets": GENERATION_SPLIT_SEED_OFFSETS,
            "generation_sa_settings": {
                "sweeps": 200,
                "beta_start": 0.1,
                "constructor_beta_end": 2.0,
                "beta_argument": 1.0,
                "effective_beta_end": 1.0,
            },
            "generation_observation_noise": True,
            "generation_day_batch_size": 50,
        }
        for key, expected in fixed_generation_settings.items():
            if key in defaults and defaults[key] != expected:
                raise ValueError("Config %s does not match the frozen generation settings" % key)
        valid_destinations = {action.dest for action in parser._actions}
        metadata_only = {
            "method_name",
            "model_role",
            "training_backend",
            "hardware_sampling_stage",
            "hardware_sampling_substitution",
            "quantum_advantage_claim",
            "protocol_version",
            "artifact_role",
            "checkpoint_selection",
            "test_used_for_selection",
            "couplings_fixed_zero",
            "conditioning_invariant",
            "initialization_invariant",
            "historical_graph_json_note",
            "training_seed_policy",
            "generation_seed_policy",
            "generation_split_seed_offsets",
            "generation_sa_settings",
            "generation_observation_noise",
            "generation_day_batch_size",
        }
        unknown = sorted(set(defaults) - valid_destinations - metadata_only)
        if unknown:
            raise ValueError(
                "Unknown QBM-VAE config key(s): %s" % ", ".join(unknown)
            )
        parser.set_defaults(
            **{
                key: value
                for key, value in defaults.items()
                if key in valid_destinations
            }
        )
    args = parser.parse_args()
    if args.validation_only:
        args.scenario_splits = ["VS"]
        args.skip_plots = True
    if args.generation_seed is not None and not (
            0 <= args.generation_seed <= 2**63 - 1 - max(GENERATION_SPLIT_SEED_OFFSETS.values())):
        parser.error("--generation-seed is outside the supported non-negative seed range")
    return args


def _random_graph_like(reference_mask, random_graph_density=None, seed=0):
    latent_s = reference_mask.shape[0]
    upper_rows, upper_cols = np.triu_indices(latent_s, k=1)
    n_possible_edges = upper_rows.size
    reference_weights = reference_mask[upper_rows, upper_cols]
    reference_weights = reference_weights[reference_weights != 0]
    if random_graph_density is None:
        n_edges = int(reference_weights.size)
    else:
        random_graph_density = float(np.clip(random_graph_density, 0.0, 1.0))
        n_edges = int(round(random_graph_density * n_possible_edges))
    n_edges = int(np.clip(n_edges, 0, n_possible_edges))
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_possible_edges, size=n_edges, replace=False)
    upper = np.zeros((latent_s, latent_s), dtype=np.float32)
    if n_edges:
        if reference_weights.size:
            weights = rng.choice(
                reference_weights,
                size=n_edges,
                replace=n_edges > reference_weights.size,
            )
        else:
            weights = np.ones(n_edges, dtype=np.float32)
        upper[upper_rows[selected], upper_cols[selected]] = weights
    return upper + upper.T


def load_frozen_graph_mask(path, latent_s):
    path = Path(path)
    with path.open("rb") as handle:
        model = pickle.load(handle)
    mask = model.graph_mask
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape != (latent_s, latent_s):
        raise ValueError(
            "Frozen graph mask has shape %s; expected (%d, %d)"
            % (mask.shape, latent_s, latent_s)
        )
    if not np.allclose(mask, mask.T) or not np.allclose(np.diag(mask), 0.0):
        raise ValueError("Frozen graph mask must be symmetric with a zero diagonal")
    return mask


def load_graph_mask_file(path, latent_s):
    mask = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float32)
    if mask.shape != (latent_s, latent_s):
        raise ValueError(
            "Graph mask file has shape %s; expected (%d, %d)"
            % (mask.shape, latent_s, latent_s)
        )
    if not np.allclose(mask, mask.T) or not np.allclose(np.diag(mask), 0.0):
        raise ValueError("Graph mask file must be symmetric with a zero diagonal")
    return mask


def build_graph_mask(
    y_train,
    latent_s,
    graph_mode,
    top_k,
    random_graph_density=None,
    seed=0,
    residual_train=None,
    graph_bootstrap_repetitions=100,
    graph_stability_threshold=0.7,
    graph_max_lag=6,
):
    if graph_mode == "none":
        return np.zeros((latent_s, latent_s), dtype=np.float32)

    if graph_mode == "full":
        return (np.ones((latent_s, latent_s), dtype=np.float32) - np.eye(latent_s, dtype=np.float32))

    if latent_s % y_train.shape[1] == 0:
        latent_per_period = max(1, latent_s // y_train.shape[1])
        temporal_mask = temporal_mi_sparse_graph(
            y_train,
            latent_per_period=latent_per_period,
            top_k=top_k,
            threshold=0.0,
        ).astype(np.float32)
    else:
        temporal_mask = np.ones((latent_s, latent_s), dtype=np.float32) - np.eye(latent_s, dtype=np.float32)

    if graph_mode == "temporal-mi":
        return temporal_mask

    if graph_mode == "random":
        return _random_graph_like(
            temporal_mask, random_graph_density=random_graph_density, seed=seed
        )

    if graph_mode in ("stable-residual-mi", "stable-residual-random"):
        if residual_train is None:
            raise ValueError("%s requires decoder-anchor LS residuals" % graph_mode)
        if latent_s % residual_train.shape[1] != 0:
            raise ValueError(
                "%s requires latent_s to be divisible by the number of periods"
                % graph_mode
            )
        stable_mask = stable_temporal_mi_sparse_graph(
            residual_train,
            latent_per_period=latent_s // residual_train.shape[1],
            top_k=top_k,
            bootstrap_repetitions=graph_bootstrap_repetitions,
            stability_threshold=graph_stability_threshold,
            max_lag=graph_max_lag,
            seed=seed,
        ).astype(np.float32)
        if graph_mode == "stable-residual-mi":
            return stable_mask
        return _random_graph_like(
            stable_mask, random_graph_density=random_graph_density, seed=seed
        )

    if graph_mode in ("posterior-pc", "posterior-pc-random"):
        raise ValueError("%s requires --graph-mask-file" % graph_mode)

    raise ValueError("Unknown graph mode: %s" % graph_mode)


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"

    tag = args.tag
    sampler_name = args.sampler
    gpu = not args.cpu
    if sampler_name == "bosonic" or args.negative_phase_backend == "bosonic":
        raise ValueError(
            "Bosonic platform sampling is post-training only for FA-BM-VAE. "
            "Train with classical Gibbs/SA, then run export_ising_instances -> "
            "prepare_bosonic_submission -> import_hardware_responses -> "
            "reconstruct_hardware_scenarios. Do not use the platform as the "
            "training negative phase."
        )
    if args.transverse_field > 0 and sampler_name != "path-integral":
        raise ValueError("A positive transverse field requires --sampler path-integral")
    if sampler_name == "path-integral" and args.transverse_field <= 0:
        raise ValueError("--sampler path-integral requires --transverse-field > 0")
    if args.beta <= 0:
        raise ValueError("--beta must be positive")
    if args.negative_num_reads < 1:
        raise ValueError("--negative-num-reads must be positive")
    if args.negative_sampler_sweeps < 1:
        raise ValueError("--negative-sampler-sweeps must be positive")
    if args.negative_sampler_burn_in < 0:
        raise ValueError("--negative-sampler-burn-in must be non-negative")
    if args.trajectory_ensemble_size < 2:
        raise ValueError("--trajectory-ensemble-size must be at least 2")
    if any(
        weight < 0
        for weight in (
            args.trajectory_energy_weight,
            args.trajectory_variogram_weight,
            args.trajectory_ramp_weight,
        )
    ):
        raise ValueError("Trajectory score weights must be non-negative")
    if args.transverse_field > 0 and args.negative_phase_backend != "internal":
        raise ValueError(
            "Transverse-field training currently requires the internal "
            "path-integral negative phase"
        )
    if args.forecast_anchor and (args.learned_forecast_anchor or args.frozen_anchor_model):
        raise ValueError("Choose only one forecast anchor mode")
    if args.context_anchor_conditioning_only and (
        args.forecast_anchor or args.learned_forecast_anchor or args.frozen_anchor_model
    ):
        raise ValueError("--context-anchor-conditioning-only cannot be combined with an anchor-addition mode")
    if args.graph_mask_model and args.graph_mask_file:
        raise ValueError("Choose only one of --graph-mask-model and --graph-mask-file")
    if len(set(args.scenario_splits)) != len(args.scenario_splits):
        raise ValueError("--scenario-splits must not contain duplicates")
    if not args.skip_plots and set(args.scenario_splits) != {"VS", "TEST"}:
        raise ValueError("Partial scenario generation requires --skip-plots")

    dir_path = str(args.output_dir or Path("export") / ("qbm_vae_" + tag)) + "/"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if args.dataset_bundle is not None:
        arrays, dataset_metadata, dates = load_daily_bundle(args.dataset_bundle)
        expected_name = dataset_metadata.get("dataset_name")
        if expected_name and expected_name != tag:
            raise ValueError("Bundle dataset_name %s does not match --tag %s" % (expected_name, tag))
        split_dates = {
            split: pd.to_datetime(dates.get(split, np.arange(len(arrays["y_" + split]))))
            for split in ("ls", "vs", "test")
        }
        data = (
            pd.DataFrame(arrays["x_ls"], index=split_dates["ls"]),
            pd.DataFrame(arrays["y_ls"], index=split_dates["ls"]),
            pd.DataFrame(arrays["x_vs"], index=split_dates["vs"]),
            pd.DataFrame(arrays["y_vs"], index=split_dates["vs"]),
            pd.DataFrame(arrays["x_test"], index=split_dates["test"]),
            pd.DataFrame(arrays["y_test"], index=split_dates["test"]),
        )
        indices = []
        ylim_loss = [-100, 100]
        ymax_plf = 8
        ylim_crps = [0, 12]
        nb_zones = 1
    elif tag == "pv":
        data, indices = pv_data(path_name=data_dir / "solar_new.csv", test_size=50, random_state=0)
        ylim_loss = [-100, 100]
        ymax_plf = 2.5
        ylim_crps = [0, 12]
        nb_zones = 3
    elif tag == "wind":
        data = wind_data(path_name=data_dir / "wind_data_all_zone.csv", test_size=50, random_state=0)
        ylim_loss = [-100, 100]
        ymax_plf = 8
        ylim_crps = [6, 12]
        nb_zones = 10
        indices = []
    elif tag == "load":
        data = load_data(path_name=data_dir / "load_data_track1.csv", test_size=50, random_state=0)
        ylim_loss = [-100, 100]
        ymax_plf = 2
        ylim_crps = [0, 5]
        nb_zones = 1
        indices = []
    else:
        raise ValueError("Unknown tag: %s" % tag)

    df_x_LS = data[0].copy()
    df_y_LS = data[1].copy()
    df_x_VS = data[2].copy()
    df_y_VS = data[3].copy()
    df_x_TEST = data[4].copy()
    df_y_TEST = data[5].copy()

    nb_days_LS = len(df_y_LS)
    nb_days_VS = len(df_y_VS)
    nb_days_TEST = len(df_y_TEST)
    print("#LS %s days #VS %s days # TEST %s days" %
          (nb_days_LS / nb_zones, nb_days_VS / nb_zones, nb_days_TEST / nb_zones))

    x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, y_LS_scaler = scale_data_multi(
        x_LS=df_x_LS.values,
        y_LS=df_y_LS.values,
        x_VS=df_x_VS.values,
        y_VS=df_y_VS.values,
        x_TEST=df_x_TEST.values,
        y_TEST=df_y_TEST.values,
    )

    anchor_model = None
    anchor_history = []
    if args.forecast_anchor or args.context_anchor_conditioning_only:
        target_size = y_LS_scaled.shape[1]
        if df_x_LS.shape[1] < target_size:
            raise ValueError(
                "--forecast-anchor requires at least %d raw context features" % target_size
            )
        anchor_splits = (
            y_LS_scaler.transform(df_x_LS.values[:, :target_size]),
            y_LS_scaler.transform(df_x_VS.values[:, :target_size]),
            y_LS_scaler.transform(df_x_TEST.values[:, :target_size]),
        )
        x_LS_scaled = np.concatenate((x_LS_scaled, anchor_splits[0]), axis=1)
        x_VS_scaled = np.concatenate((x_VS_scaled, anchor_splits[1]), axis=1)
        x_TEST_scaled = np.concatenate((x_TEST_scaled, anchor_splits[2]), axis=1)
    elif args.learned_forecast_anchor or args.frozen_anchor_model:
        anchor_device = torch.device(
            "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
        )
        if args.frozen_anchor_model:
            checkpoint = torch.load(
                args.frozen_anchor_model, map_location=anchor_device
            )
            expected_context_dim = int(df_x_LS.shape[1])
            expected_target_dim = int(y_LS_scaled.shape[1])
            if checkpoint.get("context_dim") != expected_context_dim:
                raise ValueError("Frozen anchor context dimension does not match GEFCom Wind")
            if checkpoint.get("target_dim") != expected_target_dim:
                raise ValueError("Frozen anchor target dimension does not match GEFCom Wind")
            anchor_model = DeterministicForecastAnchor(
                context_dim=expected_context_dim,
                target_dim=expected_target_dim,
                hidden_dim=int(checkpoint["hidden_dim"]),
                layers=int(checkpoint["layers"]),
            ).to(anchor_device)
            anchor_model.load_state_dict(checkpoint["state_dict"])
            anchor_model.eval()
            anchor_history = [{
                "source": "frozen_checkpoint",
                "path": str(args.frozen_anchor_model.resolve()),
                "sha256": hashlib.sha256(args.frozen_anchor_model.read_bytes()).hexdigest(),
            }]
        else:
            anchor_model, anchor_history = fit_deterministic_anchor(
                x_LS_scaled,
                y_LS_scaled,
                x_VS_scaled,
                y_VS_scaled,
                hidden_dim=args.anchor_hidden_dim,
                layers=args.anchor_layers,
                epochs=args.anchor_epochs,
                batch_size=args.batch_size,
                learning_rate=args.anchor_learning_rate,
                weight_decay=args.weight_decay,
                patience=args.anchor_patience,
                seed=args.seed,
                device=anchor_device,
            )
        anchor_splits = (
            predict_deterministic_anchor(anchor_model, x_LS_scaled, anchor_device),
            predict_deterministic_anchor(anchor_model, x_VS_scaled, anchor_device),
            predict_deterministic_anchor(anchor_model, x_TEST_scaled, anchor_device),
        )
        x_LS_scaled = np.concatenate((x_LS_scaled, anchor_splits[0]), axis=1)
        x_VS_scaled = np.concatenate((x_VS_scaled, anchor_splits[1]), axis=1)
        x_TEST_scaled = np.concatenate((x_TEST_scaled, anchor_splits[2]), axis=1)

    non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))
    if tag == "pv":
        df_y_TEST.columns = non_null_indexes
        for i in indices:
            df_y_TEST[i] = 0
        df_y_TEST = df_y_TEST.sort_index(axis=1)

        df_y_VS.columns = non_null_indexes
        for i in indices:
            df_y_VS[i] = 0
        df_y_VS = df_y_VS.sort_index(axis=1)

    nb_epoch = args.epochs
    model_family = "PIQBMVAE_1" if args.transverse_field > 0 else "QBMVAE_2"
    run_label = args.run_label
    if not run_label and args.forecast_anchor:
        run_label = "anchor"
    if not run_label and (args.learned_forecast_anchor or args.frozen_anchor_model):
        run_label = "lanchor"
    model_version = model_family + (("_" + run_label) if run_label else "")
    config = {"name": model_version, "latent_s": args.latent_s, "enc_w": args.enc_w, "enc_l": args.enc_l,
              "dec_w": args.dec_w, "dec_l": args.dec_l, "learning_rate": args.learning_rate,
              "weight_decay": args.weight_decay, "decoder_covariance": args.decoder_covariance,
              "max_ar_coefficient": args.max_ar_coefficient,
              "decoder_anchor": args.forecast_anchor or args.learned_forecast_anchor or args.frozen_anchor_model is not None,
              "context_anchor_conditioning_only": bool(args.context_anchor_conditioning_only),
              "anchor_source": (
                  "context" if args.forecast_anchor else
                  "context_conditioning_only" if args.context_anchor_conditioning_only else
                  "frozen_mlp" if args.frozen_anchor_model else
                  "learned_mlp" if args.learned_forecast_anchor else "none"
              ),
              "anchor_hidden_dim": args.anchor_hidden_dim,
              "anchor_layers": args.anchor_layers,
              "anchor_epochs": args.anchor_epochs,
              "anchor_learning_rate": args.anchor_learning_rate,
              "anchor_patience": args.anchor_patience,
              "anchor_history": anchor_history}
    config["decoder_mean_zero_init"] = (
        bool(config["decoder_anchor"])
        if args.decoder_mean_zero_init is None
        else bool(args.decoder_mean_zero_init)
    )
    config["frozen_anchor_model"] = (
        str(args.frozen_anchor_model.resolve()) if args.frozen_anchor_model else None
    )
    config["frozen_anchor_model_sha256"] = (
        hashlib.sha256(args.frozen_anchor_model.read_bytes()).hexdigest()
        if args.frozen_anchor_model else None
    )

    config["in_size"] = y_LS_scaled.shape[1]
    config["cond_in"] = x_LS_scaled.shape[1]
    config["sampler"] = sampler_name
    config["batch_size"] = args.batch_size
    config["qbm_weight"] = args.qbm_weight
    config["trajectory_ensemble_size"] = args.trajectory_ensemble_size
    config["trajectory_energy_weight"] = args.trajectory_energy_weight
    config["trajectory_variogram_weight"] = args.trajectory_variogram_weight
    config["trajectory_ramp_weight"] = args.trajectory_ramp_weight
    config["warmup_epochs"] = args.warmup_epochs
    config["negative_steps"] = args.negative_steps
    config["negative_phase_backend"] = args.negative_phase_backend
    config["negative_num_reads"] = args.negative_num_reads
    config["negative_sampler_sweeps"] = args.negative_sampler_sweeps
    config["negative_sampler_burn_in"] = args.negative_sampler_burn_in
    config["beta"] = args.beta
    config["prior_type"] = "trotter_bound" if args.transverse_field > 0 else "classical_ising"
    config["transverse_field"] = args.transverse_field
    config["trotter_replicas"] = args.trotter_replicas
    config["learnable_transverse_field"] = args.learnable_transverse_field
    config["method_name"] = (
        "FA-BM-VAE" if config["decoder_anchor"] else "BM-VAE ablation"
    )
    config["model_role"] = (
        "forecast_anchor_conditional_bm_vae"
        if config["decoder_anchor"]
        else "bm_vae_ablation"
    )
    config["training_backend"] = "classical"
    config["hardware_sampling_stage"] = "post_training_only"
    config["hardware_sampling_substitution"] = (
        "conditional_ising_latent"
    )
    config["quantum_advantage_claim"] = False
    config["early_stopping_patience"] = args.early_stopping_patience
    config["temperature_start"] = args.temperature_start
    config["temperature_end"] = args.temperature_end
    config["top_k"] = args.top_k
    config["graph_mode"] = args.graph_mode
    config["graph_seed"] = args.graph_seed if args.graph_seed is not None else args.seed
    config["graph_mask_model"] = (
        str(args.graph_mask_model.resolve()) if args.graph_mask_model else None
    )
    config["graph_mask_model_sha256"] = (
        hashlib.sha256(args.graph_mask_model.read_bytes()).hexdigest()
        if args.graph_mask_model else None
    )
    config["graph_mask_file"] = (
        str(args.graph_mask_file.resolve()) if args.graph_mask_file else None
    )
    config["graph_mask_file_sha256"] = (
        hashlib.sha256(args.graph_mask_file.read_bytes()).hexdigest()
        if args.graph_mask_file else None
    )
    config["graph_bootstrap_repetitions"] = args.graph_bootstrap_repetitions
    config["graph_stability_threshold"] = args.graph_stability_threshold
    config["graph_max_lag"] = args.graph_max_lag
    config["random_graph_density"] = args.random_graph_density
    config["scenario_splits"] = list(args.scenario_splits)
    config["validation_only"] = bool(args.validation_only)
    seed_metadata = build_generation_seed_metadata(args.seed, args.generation_seed)
    effective_generation_seed = seed_metadata["generation_sampler_seed"]
    config.update(seed_metadata)
    config["training_seed"] = int(args.seed)
    config["generation_seed_policy"] = (
        "explicit generation seed plus fixed split offsets"
        if args.generation_seed is not None else "legacy shared sequential RNG stream"
    )
    config["generation_sampler"] = sampler_name
    config["generation_sampler_settings"] = {
        "sa_sweeps": 200,
        "sa_beta_start": 0.1,
        "sa_constructor_beta_end": 2.0,
        "sa_beta_argument": float(config["beta"]),
        "sa_effective_beta_end": float(config["beta"]),
    }
    if sampler_name == "path-integral":
        config["generation_sampler_settings"]["trotter_replicas"] = int(
            args.trotter_replicas
        )
    config["generation_observation_noise"] = True
    config["generation_day_batch_size"] = 50
    config["zone_embedding_dim"] = args.zone_embedding_dim
    config["nb_zones"] = args.nb_zones
    config["dataset_bundle"] = str(args.dataset_bundle.resolve()) if args.dataset_bundle else None
    if args.transverse_field > 0 and config["latent_s"] % config["in_size"] != 0:
        raise ValueError(
            "Path-integral models require latent_s to be divisible by the modeled periods "
            "so each latent bit can be assigned to a period"
        )

    if args.graph_mask_file:
        graph_mask = load_graph_mask_file(args.graph_mask_file, config["latent_s"])
        if args.graph_mode == "posterior-pc-random":
            graph_mask = _random_graph_like(
                graph_mask,
                random_graph_density=config["random_graph_density"],
                seed=config["graph_seed"],
            )
    elif args.graph_mask_model:
        graph_mask = load_frozen_graph_mask(args.graph_mask_model, config["latent_s"])
    else:
        graph_mask = build_graph_mask(
            y_train=y_LS_scaled,
            latent_s=config["latent_s"],
            graph_mode=config["graph_mode"],
            top_k=config["top_k"],
            random_graph_density=config["random_graph_density"],
            seed=config["graph_seed"],
                residual_train=(
                y_LS_scaled - anchor_splits[0]
                if args.forecast_anchor or args.context_anchor_conditioning_only or args.learned_forecast_anchor or args.frozen_anchor_model
                else None
            ),
            graph_bootstrap_repetitions=config["graph_bootstrap_repetitions"],
            graph_stability_threshold=config["graph_stability_threshold"],
            graph_max_lag=config["graph_max_lag"],
        )
    graph_upper = np.triu(graph_mask, k=1)
    n_possible_edges = config["latent_s"] * (config["latent_s"] - 1) // 2
    config["graph_edges"] = int(np.count_nonzero(graph_upper))
    config["graph_density"] = float(config["graph_edges"] / max(1, n_possible_edges))
    config["graph_mask_weight_sum"] = float(graph_upper.sum())
    sampler_kwargs = {
        "sweeps": 200,
        "beta_start": 0.1,
        "beta_end": 2.0,
        "seed": args.seed,
    }
    if sampler_name == "bosonic":
        sampler_kwargs = {}
    if sampler_name == "path-integral":
        sampler_kwargs["replicas"] = args.trotter_replicas
    sampler = None if sampler_name == "gibbs" else build_sampler(sampler_name, **sampler_kwargs)
    negative_sampler = None
    if args.negative_phase_backend != "internal":
        negative_sampler_kwargs = (
            {} if args.negative_phase_backend == "bosonic" else {"seed": args.seed}
        )
        if args.negative_phase_backend in {"sa", "cim"}:
            negative_sampler_kwargs.update(
                {
                    "sweeps": args.negative_sampler_sweeps,
                    "beta_start": 0.1,
                    "beta_end": args.beta,
                }
            )
        elif args.negative_phase_backend == "gibbs":
            negative_sampler_kwargs.update(
                {
                    "sweeps": args.negative_sampler_sweeps,
                    "burn_in": args.negative_sampler_burn_in,
                }
            )
        negative_sampler = build_sampler(
            args.negative_phase_backend, **negative_sampler_kwargs
        )

    torch_seed = args.seed
    name = tag + "_" + config["name"] + "_" + sampler_name + "_" + str(torch_seed)
    print(name)
    if config["method_name"] == "FA-BM-VAE":
        print(
            "Method: FA-BM-VAE (classical training; hardware replaces only "
            "post-training conditional Ising latent sampling)"
        )

    if anchor_model is not None:
        torch.save(
            {
                "state_dict": anchor_model.state_dict(),
                "context_dim": int(df_x_LS.shape[1]),
                "target_dim": int(y_LS_scaled.shape[1]),
                "hidden_dim": int(args.anchor_hidden_dim),
                "layers": int(args.anchor_layers),
                "fit_split": "LS",
                "selection_split": "VS",
            },
            dir_path + name + "_anchor.pt",
        )

    with open(dir_path + name + ".json", "w") as file:
        json.dump(config, file, cls=NumpyEncoder)

    model = ConditionalQBMVAE(
        latent_s=config["latent_s"],
        cond_in=config["cond_in"],
        in_size=config["in_size"],
        enc_w=config["enc_w"],
        enc_l=config["enc_l"],
        dec_w=config["dec_w"],
        dec_l=config["dec_l"],
        conditional_qbm=True,
        graph_mask=graph_mask,
        zone_embedding_dim=config["zone_embedding_dim"],
        nb_zones=config["nb_zones"],
        transverse_field=config["transverse_field"],
        trotter_replicas=config["trotter_replicas"],
        learnable_transverse_field=config["learnable_transverse_field"],
        decoder_covariance=config["decoder_covariance"],
        max_ar_coefficient=config["max_ar_coefficient"],
        decoder_anchor=config["decoder_anchor"],
        decoder_mean_zero_init=config["decoder_mean_zero_init"],
        gpu=gpu,
    )
    if config["graph_edges"] == 0:
        config["couplings_fixed_zero"] = not model.raw_j.requires_grad
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    print(
        "Fit QBM-VAE with %s epochs, %s generation sampler, and %s negative phase"
        % (nb_epoch, sampler_name, args.negative_phase_backend)
    )
    start = timer()
    loss, best_model, last_model = fit_qbm_vae(
        nb_epoch=nb_epoch,
        x_LS=x_LS_scaled,
        y_LS=y_LS_scaled,
        x_VS=x_VS_scaled,
        y_VS=y_VS_scaled,
        x_TEST=x_TEST_scaled,
        y_TEST=y_TEST_scaled,
        model=model,
        opt=opt,
        sampler=sampler,
        gpu=gpu,
        qbm_weight=config["qbm_weight"],
        batch_size=config["batch_size"],
        warmup_epochs=config["warmup_epochs"],
        negative_steps=config["negative_steps"],
        beta=config["beta"],
        early_stopping_patience=config["early_stopping_patience"],
        temperature_start=config["temperature_start"],
        temperature_end=config["temperature_end"],
        seed=args.seed,
        negative_sampler=negative_sampler,
        negative_num_reads=config["negative_num_reads"],
        trajectory_ensemble_size=config["trajectory_ensemble_size"],
        trajectory_energy_weight=config["trajectory_energy_weight"],
        trajectory_variogram_weight=config["trajectory_variogram_weight"],
        trajectory_ramp_weight=config["trajectory_ramp_weight"],
    )
    config["best_epoch"] = int(getattr(best_model, "selection_epoch", -1))
    config["checkpoint_selection_metric"] = getattr(
        best_model, "selection_metric", "VS reconstruction"
    )
    with open(dir_path + name + ".json", "w") as file:
        json.dump(config, file, cls=NumpyEncoder)
    print("Training time %.2f s" % (timer() - start))

    dump_file(dir=dir_path, name="loss_" + name, file=loss)
    dump_file(dir=dir_path, name=name, file=best_model)
    if not args.skip_plots:
        plot_loss(loss=loss, nb_days=[nb_days_LS, nb_days_VS, nb_days_TEST], ylim=ylim_loss,
                  dir_path=dir_path, name="ll_" + name)

    if args.skip_scenarios:
        print("Skipping scenario generation")
        raise SystemExit(0)

    n_s = args.n_scenarios
    N_q = 99
    scenarios_by_split = {}
    split_inputs = {"VS": x_VS_scaled, "TEST": x_TEST_scaled}
    for split in args.scenario_splits:
        split_seed = (config["generation_split_seeds"][split]
                      if config["generation_split_seeds"] is not None else None)
        split_sampler = sampler
        noise_generator = None
        if args.generation_seed is not None:
            noise_generator = torch.Generator(device=best_model.device)
            noise_generator.manual_seed(split_seed)
            if sampler_name == "sa":
                generation_sampler_kwargs = dict(sampler_kwargs)
                generation_sampler_kwargs["seed"] = split_seed
                split_sampler = build_sampler("sa", **generation_sampler_kwargs)
        scenarios_by_split[split] = build_qbm_vae_scenarios(
            n_s=n_s,
            x=split_inputs[split],
            y_scaler=y_LS_scaler,
            model=best_model,
            sampler=split_sampler,
            max=None if tag == "neso-wind" else 1,
            gpu=gpu,
            tag=tag,
            non_null_indexes=non_null_indexes,
            beta=config["beta"],
            gibbs_steps=config["negative_steps"],
            generator=noise_generator,
        )
        dump_file(
            dir=dir_path,
            name="scenarios_" + name + "_" + str(n_s) + "_" + split,
            file=scenarios_by_split[split],
        )

    if not args.skip_plots:
        quantiles_and_evaluation(
            dir_path=dir_path,
            s_VS=scenarios_by_split["VS"],
            s_TEST=scenarios_by_split["TEST"],
            N_q=N_q,
            df_y_VS=df_y_VS,
            df_y_TEST=df_y_TEST,
            name=name,
            ymax_plf=ymax_plf,
            ylim_crps=ylim_crps,
            tag=tag,
            nb_zones=nb_zones,
        )
