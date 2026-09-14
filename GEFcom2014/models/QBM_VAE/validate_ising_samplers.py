# -*- coding: utf-8 -*-

"""Validate Exact, Gibbs, and SA samplers against a small exact Ising law."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .ising import ising_energy, symmetrize_couplings
from .samplers import build_sampler


def exact_ising_distribution(h, j, beta):
    h = np.asarray(h, dtype=np.float64)
    j = symmetrize_couplings(j)
    n_bits = h.shape[0]
    if n_bits > 20:
        raise ValueError("Exact diagnostics support at most 20 bits")
    indices = np.arange(2 ** n_bits, dtype=np.uint64)
    bit_positions = np.arange(n_bits, dtype=np.uint64)
    bits = ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
    states = 2 * bits - 1
    energies = ising_energy(states, h, j)
    logits = -float(beta) * energies
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return states, energies, probabilities


def samples_to_indices(samples):
    samples = np.asarray(samples)
    if samples.ndim != 2 or not np.all(np.isin(samples, (-1, 1))):
        raise ValueError("samples must have shape (n_samples, n_bits) in {-1, +1}")
    bits = (samples > 0).astype(np.uint64)
    weights = np.left_shift(
        np.uint64(1), np.arange(samples.shape[1], dtype=np.uint64)
    )
    return bits @ weights


def distribution_diagnostics(samples, exact_energies, exact_probabilities):
    sample_indices = samples_to_indices(samples)
    empirical = np.bincount(
        sample_indices.astype(np.int64),
        minlength=exact_probabilities.shape[0],
    ).astype(np.float64)
    empirical /= empirical.sum()
    midpoint = 0.5 * (empirical + exact_probabilities)
    nonzero_empirical = empirical > 0
    nonzero_exact = exact_probabilities > 0
    js_divergence = 0.5 * (
        np.sum(
            empirical[nonzero_empirical]
            * np.log(empirical[nonzero_empirical] / midpoint[nonzero_empirical])
        )
        + np.sum(
            exact_probabilities[nonzero_exact]
            * np.log(
                exact_probabilities[nonzero_exact] / midpoint[nonzero_exact]
            )
        )
    )
    empirical_energy = float(np.sum(empirical * exact_energies))
    exact_energy = float(np.sum(exact_probabilities * exact_energies))
    return {
        "total_variation": float(
            0.5 * np.abs(empirical - exact_probabilities).sum()
        ),
        "js_divergence": float(js_divergence),
        "energy_mean": empirical_energy,
        "exact_energy_mean": exact_energy,
        "energy_mean_error": empirical_energy - exact_energy,
        "unique_state_ratio": float(np.unique(sample_indices).size / len(samples)),
    }


def build_problem(n_bits, density, seed):
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 0.25, size=n_bits)
    upper_mask = np.triu(rng.random((n_bits, n_bits)) < density, k=1)
    upper_j = np.triu(rng.normal(0.0, 0.15, size=(n_bits, n_bits)), k=1)
    upper_j *= upper_mask
    return h, upper_j + upper_j.T


def run_validation(args):
    rows = []
    for problem_seed in args.seeds:
        h, j = build_problem(args.n_bits, args.density, problem_seed)
        _, exact_energies, exact_probabilities = exact_ising_distribution(
            h, j, args.beta
        )
        samplers = {
            "Exact": build_sampler("exact", seed=problem_seed),
            "Gibbs": build_sampler(
                "gibbs",
                seed=problem_seed,
                sweeps=args.gibbs_sweeps,
                burn_in=args.gibbs_burn_in,
            ),
            "SA": build_sampler(
                "sa",
                seed=problem_seed,
                sweeps=args.sa_sweeps,
                beta_start=args.sa_beta_start,
                beta_end=args.beta,
            ),
        }
        for label, sampler in samplers.items():
            result = sampler.sample_ising(
                h, j, num_reads=args.num_reads, beta=args.beta
            )
            row = {
                "problem_seed": int(problem_seed),
                "backend": label,
                "n_bits": int(args.n_bits),
                "num_reads": int(args.num_reads),
                "beta": float(args.beta),
                "latency_s": float(result.metadata["latency_s"]),
            }
            row.update(
                distribution_diagnostics(
                    result.samples, exact_energies, exact_probabilities
                )
            )
            rows.append(row)
            print(
                "seed %d | %-5s | TV %.4f JS %.4f energy-error %+.4f time %.3fs"
                % (
                    problem_seed,
                    label,
                    row["total_variation"],
                    row["js_divergence"],
                    row["energy_mean_error"],
                    row["latency_s"],
                )
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bits", type=int, default=12)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-reads", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gibbs-sweeps", type=int, default=100)
    parser.add_argument("--gibbs-burn-in", type=int, default=100)
    parser.add_argument("--sa-sweeps", type=int, default=200)
    parser.add_argument("--sa-beta-start", type=float, default=0.1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("export/sampler_validation")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.n_bits <= 20:
        raise ValueError("--n-bits must be between 1 and 20")
    if not 0.0 <= args.density <= 1.0:
        raise ValueError("--density must be between zero and one")
    if args.beta <= 0 or args.num_reads < 1:
        raise ValueError("--beta and --num-reads must be positive")
    rows = run_validation(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ising_sampler_validation.csv"
    json_path = args.output_dir / "ising_sampler_validation.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": vars(args) | {"output_dir": str(args.output_dir)}, "rows": rows},
            handle,
            indent=2,
        )
    print("Wrote %s and %s" % (csv_path.resolve(), json_path.resolve()))


if __name__ == "__main__":
    main()
