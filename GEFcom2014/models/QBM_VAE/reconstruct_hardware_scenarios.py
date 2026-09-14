# -*- coding: utf-8 -*-

"""Decode imported platform spins with a frozen FA-BM-VAE.

The model is trained classically and remains frozen.  The platform response
only substitutes the conditional Ising latent samples used by the decoder.
This module deliberately does not train a QBM or infer a quantum advantage.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE.export_ising_instances import (
    append_anchor_conditions,
    load_track,
)
from GEFcom2014.models.QBM_VAE.ising import collapse_trotter_samples, spin_to_binary
from GEFcom2014.models.QBM_VAE.utils_qbm_vae import (
    sample_standardized_ar_residuals,
)
from GEFcom2014.utils import dump_file


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "fa_bm_vae_hardware"


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def read_scalar_string(container, key, default):
    if key not in container:
        return default
    value = np.asarray(container[key]).reshape(-1)
    if value.size != 1:
        raise ValueError("%s must be scalar" % key)
    return str(value[0])


def load_platform_response(path, expected_bits):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing imported platform response: %s" % path)
    with np.load(path, allow_pickle=False) as response:
        if "samples" not in response:
            raise ValueError("%s does not contain samples" % path)
        samples = np.asarray(response["samples"], dtype=np.int8)
        if samples.ndim != 2 or samples.shape[1] != expected_bits:
            raise ValueError(
                "%s has samples shape %s; expected (n_reads, %d)"
                % (path, samples.shape, expected_bits)
            )
        if samples.shape[0] < 1 or not np.isin(samples, (-1, 1)).all():
            raise ValueError("%s samples must be non-empty spins in {-1,+1}" % path)
        metadata = {
            "backend": read_scalar_string(response, "backend", "bosonic_platform"),
            "job_id": read_scalar_string(response, "job_id", ""),
        }
        for key in ("latency_s", "applied_scale"):
            if key in response:
                value = np.asarray(response[key]).reshape(-1)
                if value.size != 1:
                    raise ValueError("%s must be scalar in %s" % (key, path))
                metadata[key] = float(value[0])
    return samples, metadata


def physical_latent_spins(samples, representation, physical_n_bits, trotter_replicas):
    if representation == "suzuki_trotter_path_integral":
        return collapse_trotter_samples(
            samples,
            physical_n_bits=int(physical_n_bits),
            replicas=int(trotter_replicas),
            selection="first",
        )
    if representation != "classical_ising":
        raise ValueError("Unsupported hardware representation: %s" % representation)
    return samples


def decode_platform_spins(
    model,
    spins,
    x_cond,
    y_scaler,
    max_value=1.0,
    scale_multiplier=1.0,
    observation_noise=True,
    seed=0,
):
    """Decode one condition's platform spins into shape (reads, periods)."""
    spins = np.asarray(spins, dtype=np.int8)
    if spins.ndim != 2 or spins.shape[1] != model.latent_s:
        raise ValueError(
            "Physical spins must have shape (n_reads, %d), got %s"
            % (model.latent_s, spins.shape)
        )
    if not np.isin(spins, (-1, 1)).all():
        raise ValueError("Physical spins must contain only -1 and +1")
    x_cond = np.asarray(x_cond, dtype=np.float32)
    if x_cond.ndim != 1 or x_cond.size != model.cond_in:
        raise ValueError(
            "x_cond must have shape (%d,), got %s" % (model.cond_in, x_cond.shape)
        )

    reads = spins.shape[0]
    context = torch.tensor(
        np.repeat(x_cond[None, :], reads, axis=0),
        dtype=torch.float32,
        device=model.device,
    )
    z = torch.tensor(
        spin_to_binary(spins), dtype=torch.float32, device=model.device
    )
    with torch.no_grad():
        mean, log_scale, rho = model.decode_temporal_parameters(z, context)
        if observation_noise:
            torch.manual_seed(int(seed))
            noise = scale_multiplier * torch.randn_like(mean)
            if getattr(model, "decoder_covariance", "diagonal") == "ar1":
                residual = sample_standardized_ar_residuals(
                    noise, torch.exp(log_scale), rho
                )
            else:
                residual = torch.exp(log_scale) * noise
            decoded = mean + residual
        else:
            decoded = mean
    scenarios = y_scaler.inverse_transform(decoded.cpu().numpy())
    return np.clip(scenarios, 0.0, max_value)


def reconstruct_scenarios(
    manifest_path,
    responses_dir,
    model_path,
    model_config_path,
    tag,
    dataset_bundle=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_label="BosonicSPQC",
    max_value=1.0,
    scale_multiplier=1.0,
    observation_noise=True,
    seed=2026,
    cpu=False,
):
    manifest_path = Path(manifest_path).resolve()
    responses_dir = Path(responses_dir).resolve()
    output_dir = Path(output_dir).resolve()
    instances = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(instances, list) or not instances:
        raise ValueError("The exported manifest must contain at least one instance")

    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not cpu else "cpu")
    model.to(device)
    model.device = device
    model.eval()

    model_config_path = Path(model_config_path)
    config = (
        json.loads(model_config_path.read_text(encoding="utf-8"))
        if model_config_path.is_file()
        else {}
    )
    if bool(config.get("decoder_anchor", False)) and config.get(
        "method_name", "FA-BM-VAE"
    ) != "FA-BM-VAE":
        raise ValueError("Forecast Anchor model metadata must use method_name FA-BM-VAE")
    if not bool(getattr(model, "decoder_anchor", False)):
        raise ValueError("Hardware reconstruction requires a Forecast Anchor model")

    data, indices = load_track(tag, dataset_bundle)
    _, _, _, _, _, _, y_scaler = scale_data_multi(
        x_LS=data[0].values,
        y_LS=data[1].values,
        x_VS=data[2].values,
        y_VS=data[3].values,
        x_TEST=data[4].values,
        y_TEST=data[5].values,
    )
    anchor_checkpoint = Path(model_path).with_name(
        Path(model_path).stem + "_anchor.pt"
    )
    conditions = {}
    for split, frame in (("VS", data[2]), ("TEST", data[4])):
        scaled_blocks = scale_data_multi(
            x_LS=data[0].values,
            y_LS=data[1].values,
            x_VS=frame.values,
            y_VS=data[3].values if split == "VS" else data[5].values,
            x_TEST=data[4].values,
            y_TEST=data[5].values,
        )
        # scale_data_multi returns LS, VS, TEST blocks; select the matching one.
        scaled = scaled_blocks[2] if split == "VS" else scaled_blocks[4]
        conditions[split] = append_anchor_conditions(
            model,
            config,
            anchor_checkpoint,
            scaled,
            frame.values,
            y_scaler,
            device,
        )

    chunks = []
    audit_records = []
    read_counts = set()
    for offset, instance in enumerate(instances):
        split = instance.get("split", "TEST")
        source_index = int(instance["index"])
        problem_path = resolve_repo_path(instance["path"])
        with np.load(problem_path, allow_pickle=False) as problem:
            x_cond = np.asarray(problem["x_cond"], dtype=np.float32)
            representation = read_scalar_string(
                problem, "representation", instance.get("representation", "classical_ising")
            )
            physical_n_bits = int(
                np.asarray(
                    problem["physical_h"] if "physical_h" in problem else problem["h"]
                ).shape[0]
            )
            replicas = int(
                np.asarray(
                    problem["trotter_replicas"]
                    if "trotter_replicas" in problem
                    else 1
                )
            )
        response_path = responses_dir / (instance["id"] + ".npz")
        samples, response_metadata = load_platform_response(
            response_path, int(instance["n_bits"])
        )
        physical_spins = physical_latent_spins(
            samples, representation, physical_n_bits, replicas
        )
        if physical_spins.shape[1] != model.latent_s:
            raise ValueError(
                "%s has %d physical latent bits; model expects %d"
                % (instance["id"], physical_spins.shape[1], model.latent_s)
            )
        if split in conditions and source_index < conditions[split].shape[0]:
            expected_cond = conditions[split][source_index]
            if not np.allclose(expected_cond, x_cond, atol=1e-5):
                raise ValueError(
                    "Condition mismatch for %s; export and reconstruction use different preprocessing"
                    % instance["id"]
                )
        decoded = decode_platform_spins(
            model,
            physical_spins,
            x_cond,
            y_scaler,
            max_value=max_value,
            scale_multiplier=scale_multiplier,
            observation_noise=observation_noise,
            seed=seed + offset,
        )
        if tag == "pv" and indices:
            non_null_indexes = list(np.delete(np.arange(24), indices))
            rebuilt = np.zeros((decoded.shape[0], 24), dtype=np.float64)
            rebuilt[:, non_null_indexes] = decoded
            decoded = rebuilt
        chunks.append(decoded.T)
        read_counts.add(decoded.shape[0])
        audit_records.append(
            {
                "instance_id": instance["id"],
                "split": split,
                "source_index": source_index,
                "n_reads": int(decoded.shape[0]),
                "backend": response_metadata["backend"],
                "job_id": response_metadata["job_id"],
                "representation": representation,
            }
        )
    if len(read_counts) != 1:
        raise ValueError("All platform instances must have the same number of reads")

    scenarios = np.concatenate(chunks, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    reads = next(iter(read_counts))
    output_name = "scenarios_FA-BM-VAE_%s_%d_%s" % (
        output_label,
        reads,
        str(instances[0].get("split", "TEST")).upper(),
    )
    dump_file(dir=str(output_dir) + "/", name=output_name, file=scenarios)
    metadata = {
        "method_name": "FA-BM-VAE",
        "model_role": "forecast_anchor_conditional_bm_vae",
        "training_backend": "classical",
        "sampling_backend": sorted({row["backend"] for row in audit_records}),
        "sampling_stage": "post_training_conditional_ising_latent",
        "hardware_substitution": (
            "platform replaces only frozen conditional Ising latent sampling"
        ),
        "hardware_claim": False,
        "forecast_anchor": True,
        "quantum_advantage_claim": False,
        "model_path": str(Path(model_path).resolve()),
        "model_config": str(model_config_path.resolve()),
        "source_manifest": str(manifest_path),
        "response_import_audit": str(
            responses_dir / "response_import_audit.json"
        ),
        "n_instances": len(instances),
        "n_reads_per_instance": reads,
        "scenario_shape": list(scenarios.shape),
        "observation_noise": bool(observation_noise),
        "seed": int(seed),
        "records": audit_records,
    }
    metadata_path = output_dir / (output_name + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir / (output_name + ".pickle"), metadata_path, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-label", default="BosonicSPQC")
    parser.add_argument("--max-value", type=float, default=1.0)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-observation-noise", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_value <= 0 or args.scale_multiplier < 0:
        raise ValueError("--max-value must be positive and --scale-multiplier non-negative")
    model_dir = ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag)
    result = reconstruct_scenarios(
        manifest_path=args.manifest,
        responses_dir=args.responses_dir,
        model_path=model_dir / (args.model_name + ".pickle"),
        model_config_path=model_dir / (args.model_name + ".json"),
        tag=args.tag,
        dataset_bundle=args.dataset_bundle,
        output_dir=args.output_dir,
        output_label=args.output_label,
        max_value=args.max_value,
        scale_multiplier=args.scale_multiplier,
        observation_noise=not args.no_observation_noise,
        seed=args.seed,
        cpu=args.cpu,
    )
    print("Wrote FA-BM-VAE hardware scenarios to %s" % result[0])
    print("Wrote reconstruction metadata to %s" % result[1])


if __name__ == "__main__":
    main()
