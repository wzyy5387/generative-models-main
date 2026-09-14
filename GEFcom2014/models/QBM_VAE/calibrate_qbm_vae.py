# -*- coding: utf-8 -*-

import csv
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from GEFcom2014 import load_data, pv_data, wind_data
from GEFcom2014.forecast_quality.compare_scenarios import crps_per_hour_fast
from GEFcom2014.forecast_quality.utils_quality import compute_reliability, plf_per_quantile
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE import build_sampler
from GEFcom2014.models.QBM_VAE.ising import spin_to_binary
from GEFcom2014.models.QBM_VAE.utils_qbm_vae import sample_standardized_ar_residuals
from GEFcom2014.utils import dump_file


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "GEFcom2014" / "data"
N_QUANTILES = 99


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tag = args.tag
    model_name = args.model_name
    sampler_name = args.sampler
    if sampler_name == "bosonic":
        raise ValueError(
            "Bosonic sampling is post-training only for FA-BM-VAE. Use the "
            "export/import/reconstruct hardware workflow instead of this "
            "online calibration entry point."
        )
    n_scenarios = args.n_scenarios
    max_power = args.max_power
    beta_grid = args.beta_grid
    scale_grid = args.scale_grid
    if args.mode == "beta-only":
        scale_grid = [1.0]
    elif args.mode == "scale-only":
        beta_grid = [1.0]

    data, indices, nb_zones = load_track(tag)
    df_y_VS = data[3].copy()
    df_y_TEST = data[5].copy()

    x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, y_LS_scaler = scale_data_multi(
        x_LS=data[0].values,
        y_LS=data[1].values,
        x_VS=data[2].values,
        y_VS=data[3].values,
        x_TEST=data[4].values,
        y_TEST=data[5].values,
    )
    non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))
    if tag == "pv":
        df_y_VS, df_y_TEST = rebuild_pv_targets(df_y_VS, df_y_TEST, indices, non_null_indexes)

    export_dir = ROOT_DIR / "export" / ("qbm_vae_%s" % tag)
    with (export_dir / (model_name + ".pickle")).open("rb") as handle:
        model = pickle.load(handle)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.device = device
    model.eval()

    sampler_kwargs = {
        "sweeps": args.sweeps,
        "beta_start": 0.1,
        "beta_end": 2.0,
        "seed": args.seed,
    }
    if sampler_name == "path-integral":
        sampler_kwargs["replicas"] = getattr(model, "trotter_replicas", 4)
    if sampler_name == "gibbs":
        sampler = None
    elif sampler_name == "bosonic":
        sampler = build_sampler(sampler_name)
    else:
        sampler = build_sampler(sampler_name, **sampler_kwargs)
    calibration_rows = []
    best = None
    print("Calibrating %s on %s VS set" % (model_name, tag))

    for beta in beta_grid:
        scenarios_by_scale = build_scenario_grid_for_beta(
            n_s=n_scenarios,
            x=x_VS_scaled,
            y_scaler=y_LS_scaler,
            model=model,
            sampler=sampler,
            beta=beta,
            scale_grid=scale_grid,
            max_value=max_power,
            tag=tag,
            non_null_indexes=non_null_indexes,
            gibbs_steps=args.gibbs_steps,
            device=device,
        )
        for scale in scale_grid:
            scenarios_vs = scenarios_by_scale[scale]
            metrics = evaluate_scenarios(scenarios_vs, df_y_VS.values.reshape(-1), tag)
            objective = metrics["crps_mean"] + args.reliability_weight * metrics["reliability_mae"]
            row = {
                "beta": beta,
                "scale_multiplier": scale,
                "mean_qs": metrics["plf_mean"],
                "mean_crps": metrics["crps_mean"],
                "mean_crps_percent": 100 * metrics["crps_mean"],
                "reliability_mae": metrics["reliability_mae"],
                "reliability_weight": args.reliability_weight,
                "objective": objective,
            }
            calibration_rows.append(row)
            print(
                "beta %.2f scale %.2f | QS %.4f CRPS %.2f%% reliability MAE %.2f objective %.4f"
                % (beta, scale, row["mean_qs"], row["mean_crps_percent"], row["reliability_mae"], objective)
            )
            if best is None or objective < best["objective"]:
                best = row

    model_variant, source_seed = parse_model_variant_and_seed(model_name, tag)
    calibrated_name = "%s_%s_%s_%s_%s" % (tag, model_variant, sampler_name, args.output_label, source_seed)
    print("Best beta %.2f scale %.2f" % (best["beta"], best["scale_multiplier"]))

    scenarios_vs = build_scenario_grid_for_beta(
        n_s=n_scenarios,
        x=x_VS_scaled,
        y_scaler=y_LS_scaler,
        model=model,
        sampler=sampler,
        scale_grid=[best["scale_multiplier"]],
        max_value=max_power,
        tag=tag,
        non_null_indexes=non_null_indexes,
        beta=best["beta"],
        gibbs_steps=args.gibbs_steps,
        device=device,
    )[best["scale_multiplier"]]
    scenarios_test = build_scenario_grid_for_beta(
        n_s=n_scenarios,
        x=x_TEST_scaled,
        y_scaler=y_LS_scaler,
        model=model,
        sampler=sampler,
        scale_grid=[best["scale_multiplier"]],
        max_value=max_power,
        tag=tag,
        non_null_indexes=non_null_indexes,
        beta=best["beta"],
        gibbs_steps=args.gibbs_steps,
        device=device,
    )[best["scale_multiplier"]]

    dump_file(dir=str(export_dir) + "/", name="scenarios_" + calibrated_name + "_" + str(n_scenarios) + "_VS",
              file=scenarios_vs)
    dump_file(dir=str(export_dir) + "/", name="scenarios_" + calibrated_name + "_" + str(n_scenarios) + "_TEST",
              file=scenarios_test)

    with (export_dir / ("calibration_" + calibrated_name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(calibration_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calibration_rows)
    with (export_dir / ("calibration_" + calibrated_name + ".json")).open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    test_metrics = evaluate_scenarios(scenarios_test, df_y_TEST.values.reshape(-1), tag)
    print(
        "Calibrated TEST | QS %.4f CRPS %.2f%% reliability MAE %.2f"
        % (test_metrics["plf_mean"], 100 * test_metrics["crps_mean"], test_metrics["reliability_mae"])
    )


def load_track(tag):
    if tag == "pv":
        data, indices = pv_data(DATA_DIR / "solar_new.csv", test_size=50, random_state=0)
        return data, indices, 3
    if tag == "wind":
        return wind_data(DATA_DIR / "wind_data_all_zone.csv", test_size=50, random_state=0), [], 10
    if tag == "load":
        return load_data(DATA_DIR / "load_data_track1.csv", test_size=50, random_state=0), [], 1
    raise ValueError("Unknown track: %s" % tag)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate QBM-VAE sampler beta and decoder scale on the VS set."
    )
    parser.add_argument("--tag", default="load", choices=["load", "wind", "pv"])
    parser.add_argument("--model-name", default="load_QBMVAE_2_sa_0")
    parser.add_argument(
        "--sampler",
        default="gibbs",
        choices=["gibbs", "sa", "path-integral", "cim", "bosonic"],
    )
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--gibbs-steps", type=int, default=50)
    parser.add_argument("--max-power", type=float, default=1.0)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    parser.add_argument("--scale-grid", type=float, nargs="+", default=[0.85, 1.0, 1.15, 1.3, 1.45, 1.6])
    parser.add_argument("--mode", default="joint", choices=["joint", "beta-only", "scale-only"])
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reliability-weight", type=float, default=2e-4)
    args = parser.parse_args()
    if args.output_label is None:
        args.output_label = {
            "joint": "cal",
            "beta-only": "beta",
            "scale-only": "scale",
        }[args.mode]
    if any(beta <= 0 for beta in args.beta_grid):
        raise ValueError("All --beta-grid values must be positive")
    return args


def parse_model_variant_and_seed(model_name, tag):
    model_variant = model_name.removeprefix(tag + "_")
    parts = model_variant.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        source_seed = parts[1]
        model_variant = parts[0]
    else:
        source_seed = "0"
    for suffix in ("_path-integral", "_path_integral", "_sa", "_gibbs", "_cim", "_bosonic"):
        if model_variant.endswith(suffix):
            model_variant = model_variant[: -len(suffix)]
            break
    return model_variant, source_seed


def rebuild_pv_targets(df_y_VS, df_y_TEST, indices, non_null_indexes):
    df_y_TEST.columns = non_null_indexes
    df_y_VS.columns = non_null_indexes
    for index in indices:
        df_y_TEST[index] = 0
        df_y_VS[index] = 0
    return df_y_VS.sort_index(axis=1), df_y_TEST.sort_index(axis=1)


def build_scenario_grid_for_beta(n_s, x, y_scaler, model, sampler, beta, scale_grid,
                                 max_value, tag, non_null_indexes, gibbs_steps, device):
    scenario_chunks = {scale: [] for scale in scale_grid}
    model.eval()
    with torch.no_grad():
        for day in range(x.shape[0]):
            mean, residual = sample_decoder_base(
                model=model,
                x_cond=x[day],
                n_s=n_s,
                sampler=sampler,
                beta=beta,
                gibbs_steps=gibbs_steps,
                device=device,
            )
            mean_np = mean.cpu().numpy()
            residual_np = residual.cpu().numpy()
            for scale in scale_grid:
                samples = mean_np + scale * residual_np
                samples = y_scaler.inverse_transform(samples)
                samples = np.clip(samples, 0, max_value)
                if tag == "pv" and non_null_indexes is not None:
                    rebuilt = np.zeros((n_s, 24))
                    rebuilt[:, non_null_indexes] = samples
                    samples = rebuilt
                scenario_chunks[scale].append(samples.transpose())
    return {scale: np.concatenate(chunks, axis=0) for scale, chunks in scenario_chunks.items()}


def sample_decoder_base(model, x_cond, n_s, sampler, beta, gibbs_steps, device):
    context = torch.tensor(np.tile(x_cond, n_s).reshape(n_s, model.cond_in), device=device).float()
    if sampler is None:
        if model.uses_path_integral():
            replicas = model.sample_path_integral_negative(
                cond_in=context,
                steps=gibbs_steps,
                beta=beta,
                persistent=False,
            )
            z = replicas[:, 0, :]
        else:
            z = model.sample_conditional_negative(
                cond_in=context,
                steps=gibbs_steps,
                beta=beta,
                persistent=False,
            )
    else:
        h, j = model.export_ising(cond_in=x_cond[None, :] if model.conditional_qbm else None)
        sampler_kwargs = {}
        if model.uses_path_integral():
            sampler_kwargs = {
                "transverse_gamma": model.transverse_gamma().detach().cpu().numpy(),
                "trotter_replicas": model.trotter_replicas,
            }
        result = sampler.sample_ising(h, j, num_reads=n_s, beta=beta, **sampler_kwargs)
        z = torch.tensor(spin_to_binary(result.samples), dtype=torch.float32, device=device)
    mean, log_scale, rho = model.decode_temporal_parameters(z, context)
    residual = sample_standardized_ar_residuals(
        torch.randn_like(mean), torch.exp(log_scale), rho
    )
    return mean, residual


def evaluate_scenarios(scenarios, y_true, tag):
    q_set = np.arange(1, N_QUANTILES + 1) / (N_QUANTILES + 1)
    quantiles = np.quantile(scenarios, q=q_set, axis=1).transpose()
    plf = plf_per_quantile(quantiles=quantiles, y_true=y_true)
    reliability = compute_reliability(y_true=y_true, y_quantile=quantiles, tag=tag)
    return {
        "plf_mean": float(plf.mean()),
        "crps_mean": float(crps_per_hour_fast(scenarios, y_true).mean()),
        "reliability_mae": float(mean_absolute_error(q_set * 100, reliability)),
    }


if __name__ == "__main__":
    main()
