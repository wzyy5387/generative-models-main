# -*- coding: utf-8 -*-

"""Generate SA/CIM reference responses for a fixed exported Ising benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np

from GEFcom2014.models.QBM_VAE.samplers import build_sampler
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import (
    normalize_hardware_spins,
    validate_hardware_matrix,
    validate_sampling_reads,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "bosonic_responses"


def main():
    args = parse_args()
    if args.coefficient_scale <= 0:
        raise ValueError("--coefficient-scale must be positive")
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir) / args.backend
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = json.loads(manifest_path.read_text(encoding="utf-8"))

    for index, instance in enumerate(instances):
        problem_path = ROOT_DIR / Path(instance["path"])
        hardware_gain = None
        with np.load(problem_path) as problem:
            logical_h = np.asarray(problem["h"], dtype=np.float64)
            logical_j = np.asarray(problem["J"], dtype=np.float64)
            if args.matrix_space == "hardware-quantized":
                if "hardware_matrix" not in problem:
                    raise ValueError(
                        "Manifest instance %s has no quantized hardware matrix"
                        % instance["id"]
                    )
                hardware_matrix = validate_hardware_matrix(problem["hardware_matrix"])
                h = np.zeros(hardware_matrix.shape[0], dtype=np.float64)
                j = -2.0 * hardware_matrix.astype(np.float64)
                hardware_gain = float(np.asarray(problem["hardware_gain"]).reshape(-1)[0])
                if hardware_gain <= 0:
                    raise ValueError("Hardware gain must be positive")
            else:
                h = logical_h
                j = logical_j
        if args.matrix_space == "hardware-quantized":
            if abs(float(args.coefficient_scale) - 1.0) > 1e-12:
                raise ValueError(
                    "The quantized-matrix control must use coefficient-scale 1.0"
                )
            validate_sampling_reads(args.num_reads)
        logical_beta = float(args.beta)
        if args.matrix_space == "hardware-quantized":
            matrix_beta = logical_beta / hardware_gain
            beta_for_sampler = matrix_beta
        else:
            programmed_h = args.coefficient_scale * h
            programmed_j = args.coefficient_scale * j
            matrix_beta = logical_beta * float(args.coefficient_scale)
            beta_for_sampler = logical_beta
        if args.matrix_space == "hardware-quantized":
            programmed_h = h
            programmed_j = j
        sampler_kwargs = {"seed": args.seed + index}
        if args.backend in {"gibbs", "sa", "cim"}:
            sampler_kwargs["sweeps"] = args.sweeps
        if args.matrix_space == "hardware-quantized" and args.backend == "sa":
            sampler_kwargs["beta_start"] = 0.1 / hardware_gain
        sampler = build_sampler(args.backend, **sampler_kwargs)
        result = sampler.sample_ising(
            programmed_h,
            programmed_j,
            num_reads=args.num_reads,
            beta=beta_for_sampler,
        )
        raw_samples = result.samples.astype(np.int8)
        if args.matrix_space == "hardware-quantized":
            logical_n_bits = raw_samples.shape[1] - 1
            samples = normalize_hardware_spins(
                raw_samples, logical_n_bits=logical_n_bits
            )
        else:
            samples = raw_samples
        arrays = {
            "samples": samples,
            "latency_s": float(result.metadata.get("latency_s", np.nan)),
            "applied_scale": float(args.coefficient_scale),
            "sampler_beta": float(beta_for_sampler),
            "logical_beta": float(logical_beta),
            "matrix_beta": float(matrix_beta),
            "hardware_gain": float(hardware_gain) if hardware_gain is not None else np.nan,
            "backend": str(result.metadata.get("backend", args.backend)),
            "job_id": "local_%s" % args.backend,
            "matrix_space": args.matrix_space,
            "hardware_claim": False,
            "physical_platform_used": False,
        }
        if args.matrix_space == "hardware-quantized":
            arrays.update(
                {
                    "hardware_samples": raw_samples,
                    "hardware_n_bits": np.asarray(raw_samples.shape[1]),
                    "logical_n_bits": np.asarray(logical_n_bits),
                    "auxiliary_spin_index": np.asarray(logical_n_bits),
                    "hardware_matrix_sha256": np.asarray(
                        instance.get("hardware_matrix_sha256", "")
                    ),
                    "hardware_gain": np.asarray(
                        hardware_gain
                        if hardware_gain is not None
                        else instance.get("hardware_gain", np.nan)
                    ),
                }
            )
        np.savez_compressed(output_dir / (instance["id"] + ".npz"), **arrays)

    print("Wrote %s %s reference responses to %s" % (len(instances), args.backend, output_dir))


def parse_args():
    parser = argparse.ArgumentParser(description="Sample exported Ising instances with a local reference backend.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--backend", default="sa", choices=["exact", "gibbs", "sa", "cim"]
    )
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--coefficient-scale", type=float, default=1.0)
    parser.add_argument(
        "--matrix-space",
        choices=["logical", "hardware-quantized"],
        default="logical",
        help="Compare float logical Ising SA or the exact quantized 49-spin matrix SA.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    main()
