# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from GEFcom2014 import load_data, pv_data, wind_data
from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE.forecast_anchor import (
    DeterministicForecastAnchor,
    predict_deterministic_anchor,
)
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import (
    HARDWARE_N_BITS,
    LOGICAL_N_BITS,
    build_task_name,
    quantize_hardware_matrix,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "GEFcom2014" / "data"
OUTPUT_DIR = ROOT_DIR / "export" / "bosonic_instances"

DEFAULT_MODEL_BY_TRACK = {
    "load": "load_QBMVAE_2_sa_0",
    "pv": "pv_QBMVAE_2_pv64_sa_0",
    # Keep the historical artifact names for compatibility, but point the
    # hardware protocol at the current Forecast Anchor models.
    "wind": "wind_QBMVAE_2_lanchor_sa_0",
    "opsd-wind": "opsd-wind_QBMVAE_2_anchor_sa_0",
}


def main():
    args = parse_args()
    if args.beta <= 0:
        raise ValueError("--beta must be positive")
    if args.trotter_replicas is not None and args.trotter_replicas < 2:
        raise ValueError("--trotter-replicas must be at least 2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or DEFAULT_MODEL_BY_TRACK[args.tag]
    model_dir = ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag)
    model_path = model_dir / (model_name + ".pickle")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    model.device = device
    model.eval()

    data, indices = load_track(args.tag, args.dataset_bundle)
    x_LS_scaled, _, x_VS_scaled, _, x_TEST_scaled, _, y_scaler = scale_data_multi(
        x_LS=data[0].values,
        y_LS=data[1].values,
        x_VS=data[2].values,
        y_VS=data[3].values,
        x_TEST=data[4].values,
        y_TEST=data[5].values,
    )
    config_path = model_dir / (model_name + ".json")
    model_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    if args.hardware_gain is not None and args.hardware_gain <= 0:
        raise ValueError("--hardware-gain must be positive")
    is_forecast_anchor = bool(model_config.get("decoder_anchor", False))
    method_name = "FA-BM-VAE" if is_forecast_anchor else "BM-VAE ablation"
    model_role = (
        "forecast_anchor_conditional_bm_vae"
        if is_forecast_anchor
        else "bm_vae_ablation"
    )
    x_VS_scaled = append_anchor_conditions(
        model,
        model_config,
        model_dir / (model_name + "_anchor.pt"),
        x_VS_scaled,
        data[2].values,
        y_scaler,
        device,
    )
    x_TEST_scaled = append_anchor_conditions(
        model,
        model_config,
        model_dir / (model_name + "_anchor.pt"),
        x_TEST_scaled,
        data[4].values,
        y_scaler,
        device,
    )
    if x_VS_scaled.shape[1] != model.cond_in:
        raise ValueError(
            "Reconstructed condition width %d does not match model.cond_in=%d"
            % (x_VS_scaled.shape[1], model.cond_in)
        )

    split_to_x = {"VS": x_VS_scaled, "TEST": x_TEST_scaled}
    split_to_raw = {"VS": data[2], "TEST": data[4]}
    all_conditions = split_to_x[args.split]
    raw_frame = split_to_raw[args.split]
    selected_rows = select_instance_rows(
        args.tag,
        raw_frame.values,
        args.num_instances,
        args.selection,
        args.selection_seed,
    )
    instances = []
    for export_index, source_row in enumerate(selected_rows):
        x_cond = all_conditions[source_row]
        cond = x_cond[None, :] if model.conditional_qbm else None
        physical_h, physical_j = model.export_ising(cond_in=cond)
        uses_path_integral = bool(
            hasattr(model, "uses_path_integral") and model.uses_path_integral()
        )
        representation = "classical_ising"
        gamma = np.asarray([], dtype=np.float64)
        replicas = 1
        if uses_path_integral:
            replicas = args.trotter_replicas or model.trotter_replicas
            h, j = model.export_trotter_ising(
                cond_in=cond,
                beta=args.beta,
                replicas=replicas,
            )
            gamma = model.transverse_gamma().detach().cpu().numpy()
            representation = "suzuki_trotter_path_integral"
        else:
            h, j = physical_h, physical_j
        validate_ising(h, j)
        hardware_problem = None
        if args.hardware_gain is not None:
            if representation != "classical_ising":
                raise ValueError(
                    "Kaiwu 49-spin export currently accepts the classical "
                    "48-spin FA-BM-VAE Ising model only"
                )
            if h.shape != (LOGICAL_N_BITS,):
                raise ValueError(
                    "Kaiwu export requires a %d-dimensional logical model; got %s"
                    % (LOGICAL_N_BITS, h.shape)
                )
            hardware_problem = quantize_hardware_matrix(
                h, j, args.hardware_gain, logical_n_bits=LOGICAL_N_BITS
            )
        stem = "%s_%s_%s_%03d" % (
            args.tag,
            model_name,
            args.split.lower(),
            export_index,
        )
        problem_path = output_dir / (stem + ".npz")
        problem_arrays = {
            "h": h,
            "J": j,
            "x_cond": x_cond,
            "physical_h": physical_h,
            "physical_J": physical_j,
            "transverse_gamma": gamma,
            "beta": np.asarray(args.beta),
            "trotter_replicas": np.asarray(replicas),
            "representation": np.asarray(representation),
        }
        if hardware_problem is not None:
            problem_arrays.update(
                {
                    "hardware_matrix": hardware_problem["matrix"],
                    "hardware_source_matrix": hardware_problem["source_matrix"],
                    "hardware_gain": np.asarray(hardware_problem["hardware_gain"]),
                    "logical_n_bits": np.asarray(LOGICAL_N_BITS),
                    "hardware_n_bits": np.asarray(HARDWARE_N_BITS),
                    "auxiliary_spin_index": np.asarray(LOGICAL_N_BITS),
                    "hardware_matrix_sha256": np.asarray(
                        hardware_problem["audit"]["matrix_sha256"]
                    ),
                    "quantization_audit": np.asarray(
                        json.dumps(hardware_problem["audit"], sort_keys=True)
                    ),
                }
            )
        np.savez_compressed(problem_path, **problem_arrays)
        hardware_matrix_path = None
        if hardware_problem is not None:
            hardware_matrix_path = output_dir / (stem + "_hardware_matrix.npz")
            np.savez_compressed(
                hardware_matrix_path,
                hardware_matrix=hardware_problem["matrix"],
                hardware_gain=np.asarray(hardware_problem["hardware_gain"]),
                logical_n_bits=np.asarray(LOGICAL_N_BITS),
                hardware_n_bits=np.asarray(HARDWARE_N_BITS),
                auxiliary_spin_index=np.asarray(LOGICAL_N_BITS),
                matrix_sha256=np.asarray(hardware_problem["audit"]["matrix_sha256"]),
                quantization_audit=np.asarray(
                    json.dumps(hardware_problem["audit"], sort_keys=True)
                ),
            )
        json_path = output_dir / (stem + ".json")
        payload = build_platform_payload(
            stem,
            h,
            j,
            representation=representation,
            physical_n_bits=physical_h.shape[0],
            beta=args.beta,
            trotter_replicas=replicas,
            transverse_gamma=gamma,
            method_name=method_name,
            model_role=model_role,
            hardware_problem=hardware_problem,
        )
        if hardware_problem is not None:
            payload["hardware_matrix_file"] = relative_or_absolute(hardware_matrix_path)
            payload["task_name"] = build_task_name(
                stem, hardware_problem["audit"]["matrix_sha256"]
            )
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        scale_to_unit = 1.0 / max(float(np.abs(h).max()), float(np.abs(j).max()), 1e-12)
        instances.append({
            "id": stem,
            "track": args.tag,
            "model": model_name,
            "method_name": method_name,
            "model_role": model_role,
            "training_backend": "classical",
            "sampling_stage": "post_training_conditional_ising_latent",
            "hardware_substitution": (
                "platform replaces only frozen conditional Ising latent sampling"
            ),
            "hardware_claim": False,
            "anchor_source": model_config.get("anchor_source", "none"),
            "prior_type": model_config.get("prior_type", "classical_ising"),
            "logical_n_bits": int(h.shape[0]),
            "hardware_n_bits": (
                int(HARDWARE_N_BITS) if hardware_problem is not None else None
            ),
            "auxiliary_spin_index": (
                int(LOGICAL_N_BITS) if hardware_problem is not None else None
            ),
            "hardware_gain": (
                float(args.hardware_gain) if hardware_problem is not None else None
            ),
            "hardware_matrix_sha256": (
                hardware_problem["audit"]["matrix_sha256"]
                if hardware_problem is not None
                else None
            ),
            "quantization_audit": (
                hardware_problem["audit"] if hardware_problem is not None else None
            ),
            "split": args.split,
            "index": int(source_row),
            "export_index": export_index,
            "selection": args.selection,
            "selection_seed": args.selection_seed,
            "zone": infer_zone(args.tag, raw_frame.values[source_row]),
            "date": str(raw_frame.index[source_row]),
            "representation": representation,
            "physical_n_bits": int(physical_h.shape[0]),
            "n_bits": int(h.shape[0]),
            "beta": float(args.beta),
            "trotter_replicas": int(replicas),
            "transverse_gamma_mean": float(gamma.mean()) if gamma.size else 0.0,
            "h_l2": float(np.linalg.norm(h)),
            "J_l2": float(np.linalg.norm(j)),
            "J_density": float(np.count_nonzero(np.abs(j) > 1e-12) / j.size),
            "scale_to_unit": scale_to_unit,
            "condition_dim": int(x_cond.shape[0]),
            "path": relative_or_absolute(output_dir / (stem + ".npz")),
            "platform_payload": relative_or_absolute(json_path),
        })
    manifest_path = output_dir / (
        "%s_%s_%s_manifest.json" % (args.tag, model_name, args.split.lower())
    )
    manifest_path.write_text(json.dumps(instances, indent=2), encoding="utf-8")
    metadata_path = manifest_path.with_name(manifest_path.stem + "_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "method_name": method_name,
                "model_role": model_role,
                "training_backend": "classical",
                "sampling_stage": "post_training_conditional_ising_latent",
                "hardware_substitution": (
                    "platform replaces only frozen conditional Ising latent sampling"
                ),
                "hardware_claim": False,
                "model": model_name,
                "track": args.tag,
                "split": args.split,
                "n_instances": len(instances),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Exported %s Ising instances to %s" % (len(instances), output_dir))
    print("Wrote FA-BM-VAE export metadata to %s" % metadata_path)


def select_instance_rows(tag, raw_x, num_instances, strategy, seed):
    raw_x = np.asarray(raw_x)
    n_rows = raw_x.shape[0]
    if strategy == "all":
        return np.arange(n_rows, dtype=np.int64)
    if num_instances <= 0 or num_instances > n_rows:
        raise ValueError(
            "--num-instances must be between 1 and %d" % n_rows
        )
    if strategy == "head":
        return np.arange(num_instances, dtype=np.int64)
    if strategy != "stratified":
        raise ValueError("Unknown selection strategy: %s" % strategy)

    if tag == "wind":
        zone_values = raw_x[:, -10:]
        if zone_values.shape[1] != 10:
            raise ValueError("Wind context does not contain ten zone indicators")
        groups = np.argmax(zone_values, axis=1)
        if not np.allclose(zone_values[np.arange(n_rows), groups], 1.0):
            raise ValueError("Wind zone indicators are invalid")
    else:
        groups = np.zeros(n_rows, dtype=np.int64)

    unique_groups = np.unique(groups)
    base, remainder = divmod(num_instances, unique_groups.size)
    if base == 0:
        raise ValueError(
            "Stratified selection needs at least one instance per group"
        )
    rng = np.random.default_rng(seed)
    selected = []
    for offset, group in enumerate(unique_groups):
        group_rows = np.flatnonzero(groups == group)
        group_count = base + int(offset < remainder)
        if group_count > group_rows.size:
            raise ValueError("Not enough rows in group %s" % group)
        selected.extend(
            rng.choice(group_rows, size=group_count, replace=False).tolist()
        )
    return np.asarray(selected, dtype=np.int64)


def infer_zone(tag, raw_row):
    if tag != "wind":
        return None
    zone_values = np.asarray(raw_row)[-10:]
    return int(np.argmax(zone_values) + 1)


def append_anchor_conditions(
    model,
    config,
    anchor_checkpoint,
    x_scaled,
    raw_x,
    y_scaler,
    device,
):
    if not bool(getattr(model, "decoder_anchor", False)):
        return x_scaled
    anchor_source = config.get("anchor_source")
    target_size = int(model.in_size)
    if anchor_source == "context":
        if raw_x.shape[1] < target_size:
            raise ValueError("Raw context does not contain the forecast anchor")
        anchor = y_scaler.transform(raw_x[:, :target_size]).astype(np.float32)
    elif anchor_source == "learned_mlp":
        if not anchor_checkpoint.is_file():
            raise FileNotFoundError(
                "Learned-anchor checkpoint is missing: %s" % anchor_checkpoint
            )
        checkpoint = torch.load(anchor_checkpoint, map_location=device)
        anchor_model = DeterministicForecastAnchor(
            context_dim=int(checkpoint["context_dim"]),
            target_dim=int(checkpoint["target_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            layers=int(checkpoint["layers"]),
        ).to(device)
        anchor_model.load_state_dict(checkpoint["state_dict"])
        anchor = predict_deterministic_anchor(anchor_model, x_scaled, device)
    else:
        raise ValueError(
            "Decoder-anchor model requires anchor_source context or learned_mlp; got %r"
            % anchor_source
        )
    return np.concatenate((x_scaled, anchor), axis=1).astype(np.float32)


def relative_or_absolute(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def load_track(tag, dataset_bundle=None):
    if dataset_bundle is not None:
        arrays, metadata, dates = load_daily_bundle(dataset_bundle)
        if metadata.get("dataset_name") not in (None, tag):
            raise ValueError(
                "Bundle dataset_name %s does not match --tag %s"
                % (metadata.get("dataset_name"), tag)
            )
        import pandas as pd

        data = tuple(
            pd.DataFrame(arrays["%s_%s" % (kind, split)])
            for split in ("ls", "vs", "test")
            for kind in ("x", "y")
        )
        return data, []
    if tag == "load":
        return load_data(DATA_DIR / "load_data_track1.csv", test_size=50, random_state=0), []
    if tag == "pv":
        data, indices = pv_data(DATA_DIR / "solar_new.csv", test_size=50, random_state=0)
        return data, indices
    if tag == "wind":
        return wind_data(DATA_DIR / "wind_data_all_zone.csv", test_size=50, random_state=0), []
    raise ValueError("Unknown tag: %s" % tag)


def validate_ising(h, j):
    if h.ndim != 1:
        raise ValueError("h must be one-dimensional")
    if j.shape != (h.shape[0], h.shape[0]):
        raise ValueError("J must be square with shape (n_bits, n_bits)")
    if not np.allclose(j, j.T, atol=1e-6):
        raise ValueError("J must be symmetric")
    if not np.allclose(np.diag(j), 0.0, atol=1e-6):
        raise ValueError("J diagonal must be zero")


def build_platform_payload(instance_id, h, j, representation="classical_ising",
                           physical_n_bits=None, beta=1.0, trotter_replicas=1,
                           transverse_gamma=None, method_name="FA-BM-VAE",
                           model_role="forecast_anchor_conditional_bm_vae",
                           hardware_problem=None):
    """Create a vendor-neutral sparse Ising payload for hardware submission."""
    h = np.asarray(h, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    upper_i, upper_j = np.where(np.triu(np.abs(j) > 1e-12, k=1))
    scale_to_unit = 1.0 / max(float(np.abs(h).max()), float(np.abs(j).max()), 1e-12)
    payload = {
        "instance_id": instance_id,
        "method_name": method_name,
        "model_role": model_role,
        "training_backend": "classical",
        "sampling_stage": "post_training_conditional_ising_latent",
        "hardware_substitution": (
            "platform replaces only frozen conditional Ising latent sampling"
        ),
        "hardware_claim": False,
        "representation": representation,
        "physical_n_bits": int(physical_n_bits or h.shape[0]),
        "expanded_n_bits": int(h.shape[0]),
        "inverse_temperature": float(beta),
        "trotter_replicas": int(trotter_replicas),
        "transverse_gamma": np.asarray(
            [] if transverse_gamma is None else transverse_gamma, dtype=np.float64
        ).tolist(),
        "spin_values": [-1, 1],
        "energy_convention": "E(s) = -h^T s - 0.5 s^T J s",
        "h": h.tolist(),
        "couplings": [
            {"i": int(i), "j": int(k), "value": float(j[i, k])}
            for i, k in zip(upper_i, upper_j)
        ],
        "recommended_global_scale_to_unit": scale_to_unit,
        "scaling_note": (
            "If the platform applies this scale, return the applied scale so the "
            "effective-temperature calibration can account for it."
        ),
        "interpretation_note": (
            "A suzuki_trotter_path_integral payload is a finite-replica classical "
            "approximation of a transverse-field Gibbs state, not native quantum Gibbs sampling."
        ),
    }
    if hardware_problem is not None:
        audit = hardware_problem["audit"]
        hardware_matrix = np.asarray(hardware_problem["matrix"])
        payload.update(
            {
                "logical_n_bits": int(hardware_matrix.shape[0] - 1),
                "hardware_n_bits": int(hardware_matrix.shape[0]),
                "auxiliary_spin_index": int(hardware_matrix.shape[0] - 1),
                "hardware_matrix": hardware_matrix.tolist(),
                "hardware_gain": float(hardware_problem["hardware_gain"]),
                "hardware_matrix_sha256": audit["matrix_sha256"],
                "hardware_energy_convention": (
                    "u^T Q u = hardware_gain * E(s), u=[s,+1]"
                ),
                "hardware_matrix_dtype": "int8",
                "quantization_audit": audit,
                "source_energy_convention": "E(s) = -h^T s - 0.5 s^T J s",
                "quantization_rule": "Q = round-half-to-even(hardware_gain * M)",
                "hardware_sampling_reads": {
                    "minimum": 10,
                    "maximum": 2000,
                },
            }
        )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Export QBM-VAE conditional Ising instances for hardware samplers.")
    parser.add_argument("--tag", default="load")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--split", default="TEST", choices=["VS", "TEST"])
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument(
        "--selection",
        choices=["head", "stratified", "all"],
        default="head",
    )
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--hardware-gain",
        type=float,
        default=None,
        help=(
            "One global Kaiwu matrix gain. When supplied, export a 49-spin "
            "quantized hardware matrix from the original 48-spin h,J."
        ),
    )
    parser.add_argument("--trotter-replicas", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
