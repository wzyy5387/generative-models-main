# -*- coding: utf-8 -*-

import argparse
import itertools
import json

import numpy as np

from .ising import expand_trotter_ising, ising_energy, symmetrize_couplings


def exact_quantum_gibbs_probabilities(h, j, transverse_gamma, beta):
    """Return diagonal probabilities of exp(-beta H) for a small TFIM."""
    h = np.asarray(h, dtype=np.float64)
    j = symmetrize_couplings(j)
    gamma = np.asarray(transverse_gamma, dtype=np.float64)
    if gamma.ndim == 0:
        gamma = np.full(h.shape[0], float(gamma))
    states = np.asarray(list(itertools.product([-1, 1], repeat=h.shape[0])), dtype=np.int8)
    hamiltonian = np.diag(ising_energy(states, h, j))
    state_to_index = {tuple(state): index for index, state in enumerate(states)}
    for state_index, state in enumerate(states):
        for bit, gamma_bit in enumerate(gamma):
            flipped = state.copy()
            flipped[bit] *= -1
            hamiltonian[state_index, state_to_index[tuple(flipped)]] -= gamma_bit

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    shifted_weights = np.exp(-beta * (eigenvalues - eigenvalues.min()))
    diagonal = (eigenvectors ** 2) @ shifted_weights
    return states, diagonal / diagonal.sum()


def exact_trotter_slice_probabilities(h, j, transverse_gamma, beta, replicas):
    """Enumerate the finite-M classical action and marginalize one replica."""
    h = np.asarray(h, dtype=np.float64)
    n_bits = h.shape[0]
    expanded_h, expanded_j = expand_trotter_ising(h, j, beta, transverse_gamma, replicas)
    expanded_states = np.asarray(
        list(itertools.product([-1, 1], repeat=n_bits * replicas)), dtype=np.int8
    )
    log_weights = -ising_energy(expanded_states, expanded_h, expanded_j)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()

    physical_states = np.asarray(list(itertools.product([-1, 1], repeat=n_bits)), dtype=np.int8)
    state_to_index = {tuple(state): index for index, state in enumerate(physical_states)}
    probabilities = np.zeros(physical_states.shape[0], dtype=np.float64)
    for state, weight in zip(expanded_states[:, :n_bits], weights):
        probabilities[state_to_index[tuple(state)]] += weight
    return physical_states, probabilities


def validate_trotter_mapping(h, j, transverse_gamma, beta, replicas_values=(2, 3, 4, 6)):
    states, exact_probabilities = exact_quantum_gibbs_probabilities(
        h, j, transverse_gamma, beta
    )
    rows = []
    for replicas in replicas_values:
        if len(h) * replicas > 20:
            raise ValueError("Exact Trotter validation is limited to 20 expanded bits")
        trotter_states, trotter_probabilities = exact_trotter_slice_probabilities(
            h, j, transverse_gamma, beta, replicas
        )
        if not np.array_equal(states, trotter_states):
            raise RuntimeError("State order mismatch in exact validation")
        rows.append({
            "replicas": int(replicas),
            "total_variation": float(0.5 * np.abs(exact_probabilities - trotter_probabilities).sum()),
            "max_probability_error": float(np.abs(exact_probabilities - trotter_probabilities).max()),
        })
    return {
        "beta": float(beta),
        "n_bits": int(len(h)),
        "exact_probabilities": exact_probabilities.tolist(),
        "states": states.tolist(),
        "trotter_error": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the finite-M TFIM Trotter mapping exactly.")
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--replicas", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    h = np.asarray([0.3, -0.2])
    j = np.asarray([[0.0, 0.35], [0.35, 0.0]])
    gamma = np.asarray([0.7, 0.5])
    result = validate_trotter_mapping(h, j, gamma, args.beta, args.replicas)
    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    print(output)


if __name__ == "__main__":
    main()
