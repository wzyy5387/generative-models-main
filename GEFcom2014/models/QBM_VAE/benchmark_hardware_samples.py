# -*- coding: utf-8 -*-

"""Validate and summarize hardware Ising samples against exported instances.

Expected response layout
------------------------
For every instance ``<instance_id>.npz`` in a manifest, place a response file
with the same name in ``--responses-dir``. The response must contain:

* ``samples``: integer spins in {-1, +1}, shape (n_reads, n_bits)

Optional scalar fields are ``latency_s`` and ``applied_scale``. Optional string
fields are ``backend`` and ``job_id``. Energies are always recomputed locally
from the unscaled exported Ising problem, making backend comparisons auditable.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from GEFcom2014.models.QBM_VAE.calibration import fit_effective_temperature_pseudolikelihood
from GEFcom2014.models.QBM_VAE.ising import collapse_trotter_samples, ising_energy
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import normalize_hardware_spins


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "bosonic_benchmark"


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    missing = []
    for instance in instances:
        response_path = responses_dir / (instance["id"] + ".npz")
        if not response_path.is_file():
            missing.append(instance["id"])
            continue
        rows.append(summarize_response(instance, response_path))

    write_outputs(rows, missing, manifest_path, output_dir)
    print("Validated %s hardware responses; %s instances are missing" % (len(rows), len(missing)))


def summarize_response(instance, response_path):
    instance_path = ROOT_DIR / Path(instance["path"])
    with np.load(instance_path) as problem:
        h = np.asarray(problem["h"], dtype=np.float64)
        j = np.asarray(problem["J"], dtype=np.float64)
        representation = str(problem["representation"]) if "representation" in problem else "classical_ising"
        physical_n_bits = int(problem["physical_h"].shape[0]) if "physical_h" in problem else h.size
        trotter_replicas = int(problem["trotter_replicas"]) if "trotter_replicas" in problem else 1
    with np.load(response_path, allow_pickle=False) as response:
        if "samples" not in response:
            raise ValueError("%s does not contain samples" % response_path)
        samples = np.asarray(response["samples"], dtype=np.int8)
        if samples.ndim != 2 or samples.shape[1] != h.size:
            raise ValueError("%s has incompatible samples shape %s" % (response_path, samples.shape))
        if not np.isin(samples, [-1, 1]).all():
            raise ValueError("%s samples must contain only -1 and +1" % response_path)
        hardware_n_bits = ""
        if "hardware_samples" in response:
            hardware_samples = np.asarray(response["hardware_samples"], dtype=np.int8)
            if hardware_samples.shape[0] != samples.shape[0]:
                raise ValueError("%s hardware/logical read counts differ" % response_path)
            if not np.array_equal(normalize_hardware_spins(hardware_samples), samples):
                raise ValueError("%s logical samples do not match auxiliary-spin normalization" % response_path)
            hardware_n_bits = int(hardware_samples.shape[1])
        metadata = {
            "backend": read_optional_string(response, "backend", "bosonic_platform"),
            "job_id": read_optional_string(response, "job_id", ""),
            "latency_s": read_optional_scalar(response, "latency_s"),
            "applied_scale": read_optional_scalar(response, "applied_scale"),
        }

    energies = ising_energy(samples, h, j)
    calibration = fit_effective_temperature_pseudolikelihood(h, j, samples)
    replica_disagreement = ""
    physical_samples = samples
    if representation == "suzuki_trotter_path_integral":
        replicas = samples.reshape(samples.shape[0], trotter_replicas, physical_n_bits)
        adjacent_agreement = np.mean(replicas * np.roll(replicas, -1, axis=1))
        replica_disagreement = float(0.5 * (1.0 - adjacent_agreement))
        physical_samples = collapse_trotter_samples(
            samples, physical_n_bits, trotter_replicas, selection="first"
        )
    return {
        "method_name": instance.get("method_name", "FA-BM-VAE"),
        "model_role": instance.get(
            "model_role", "forecast_anchor_conditional_bm_vae"
        ),
        "training_backend": instance.get("training_backend", "classical"),
        "sampling_stage": instance.get(
            "sampling_stage", "post_training_conditional_ising_latent"
        ),
        "hardware_substitution": instance.get(
            "hardware_substitution",
            "platform replaces only frozen conditional Ising latent sampling",
        ),
        "hardware_claim": bool(instance.get("hardware_claim", False)),
        "instance_id": instance["id"],
        "track": instance["track"],
        "model": instance["model"],
        "split": instance["split"],
        "representation": representation,
        "physical_n_bits": physical_n_bits,
        "n_bits": h.size,
        "hardware_n_bits": hardware_n_bits,
        "n_reads": samples.shape[0],
        "backend": metadata["backend"],
        "job_id": metadata["job_id"],
        "latency_s": metadata["latency_s"],
        "applied_scale": metadata["applied_scale"],
        "mean_energy": float(energies.mean()),
        "std_energy": float(energies.std(ddof=1)) if energies.size > 1 else 0.0,
        "min_energy": float(energies.min()),
        "energy_q05": float(np.quantile(energies, 0.05)),
        "unique_state_fraction": float(np.unique(samples, axis=0).shape[0] / samples.shape[0]),
        "mean_abs_magnetization": float(np.abs(samples.mean(axis=0)).mean()),
        "physical_unique_state_fraction": float(
            np.unique(physical_samples, axis=0).shape[0] / physical_samples.shape[0]
        ),
        "replica_disagreement": replica_disagreement,
        "beta_eff_pseudolikelihood": calibration["beta_eff"],
        "beta_eff_nll": calibration["pseudolikelihood_nll"],
        "beta_eff_grid_boundary": calibration["grid_boundary"],
        "response_path": str(response_path),
    }


def read_optional_scalar(response, key):
    if key not in response:
        return ""
    value = np.asarray(response[key]).reshape(-1)
    if value.size != 1:
        raise ValueError("%s must be scalar" % key)
    return float(value[0])


def read_optional_string(response, key, default):
    if key not in response:
        return default
    value = np.asarray(response[key]).reshape(-1)
    if value.size != 1:
        raise ValueError("%s must be scalar" % key)
    return str(value[0])


def write_outputs(rows, missing, manifest_path, output_dir):
    summary_path = output_dir / (manifest_path.stem + "_hardware_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["instance_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "method_name": rows[0].get("method_name", "FA-BM-VAE") if rows else "FA-BM-VAE",
        "training_backend": "classical",
        "sampling_stage": "post_training_conditional_ising_latent",
        "hardware_claim": False,
        "manifest": str(manifest_path),
        "n_instances": len(rows) + len(missing),
        "n_validated": len(rows),
        "missing_instance_ids": missing,
        "response_schema": {
            "required": {"samples": "int8 array of shape (n_reads, n_bits) with values -1 or +1"},
            "optional": {
                "latency_s": "scalar float",
                "applied_scale": "scalar float",
                "backend": "scalar string",
                "job_id": "scalar string",
            },
        },
    }
    (output_dir / (manifest_path.stem + "_hardware_audit.json")).write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Validate bosonic-platform Ising response files.")
    parser.add_argument("--manifest", required=True, help="Path to an exported Ising manifest JSON.")
    parser.add_argument("--responses-dir", required=True, help="Directory containing <instance_id>.npz response files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    main()
