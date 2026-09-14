# -*- coding: utf-8 -*-

import itertools

import numpy as np

from .ising import ising_energy


def fit_effective_temperature(h, j, samples, beta_grid=None, smoothing=1e-9):
    """
    Estimate beta_eff for hardware samples by matching empirical and Boltzmann distributions.

    This exact calibration is intended for small calibration graphs. For larger
    graphs, use held-out statistics such as mean energy instead of full states.
    """
    h = np.asarray(h, dtype=np.float64)
    n_bits = h.shape[0]
    if n_bits > 20:
        raise ValueError("Exact beta calibration is limited to <= 20 bits")
    if beta_grid is None:
        beta_grid = np.linspace(0.05, 5.0, 200)

    states = np.asarray(list(itertools.product([-1, 1], repeat=n_bits)), dtype=np.int8)
    energies = ising_energy(states, h, j)

    sample_keys, sample_counts = np.unique(np.asarray(samples, dtype=np.int8), axis=0, return_counts=True)
    empirical = np.full(states.shape[0], smoothing, dtype=np.float64)
    state_index = {tuple(state.tolist()): idx for idx, state in enumerate(states)}
    for state, count in zip(sample_keys, sample_counts):
        idx = state_index.get(tuple(state.tolist()))
        if idx is not None:
            empirical[idx] += count
    empirical /= empirical.sum()

    best_beta = None
    best_kl = np.inf
    for beta in beta_grid:
        logits = -beta * energies
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        kl = np.sum(empirical * (np.log(empirical + smoothing) - np.log(probs + smoothing)))
        if kl < best_kl:
            best_kl = float(kl)
            best_beta = float(beta)

    return {"beta_eff": best_beta, "kl": best_kl, "n_states": states.shape[0]}


def fit_effective_temperature_pseudolikelihood(h, j, samples, beta_grid=None):
    """Estimate effective beta without evaluating the Ising partition function.

    The estimator minimizes the negative conditional log-likelihood over all
    observed spins. It scales to the expanded Suzuki-Trotter graphs used for
    bosonic/CIM experiments.
    """
    h = np.asarray(h, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != h.shape[0]:
        raise ValueError("samples must have shape (n_samples, n_bits)")
    if not np.isin(samples, [-1, 1]).all():
        raise ValueError("samples must contain only -1 and +1")
    if j.shape != (h.shape[0], h.shape[0]):
        raise ValueError("j must have shape (n_bits, n_bits)")
    if beta_grid is None:
        beta_grid = np.linspace(0.02, 5.0, 300)
    beta_grid = np.asarray(beta_grid, dtype=np.float64)
    if beta_grid.ndim != 1 or beta_grid.size == 0 or np.any(beta_grid <= 0):
        raise ValueError("beta_grid must contain positive values")

    local_fields = h[None, :] + samples @ j
    signed_fields = samples * local_fields
    losses = np.asarray([
        np.logaddexp(0.0, -2.0 * beta * signed_fields).mean()
        for beta in beta_grid
    ])
    best_index = int(np.argmin(losses))
    return {
        "beta_eff": float(beta_grid[best_index]),
        "pseudolikelihood_nll": float(losses[best_index]),
        "n_samples": int(samples.shape[0]),
        "n_bits": int(samples.shape[1]),
        "grid_boundary": bool(best_index in (0, beta_grid.size - 1)),
    }


def fit_shared_effective_temperature_pseudolikelihood(
    problems,
    beta_grid=None,
    bootstrap_repetitions=0,
    seed=0,
):
    """Fit one effective beta across multiple held-out Ising instances.

    Every instance contributes equally, regardless of its number of reads.
    This is the calibration used for frozen VS-to-TEST hardware replay.
    """
    problems = list(problems)
    if not problems:
        raise ValueError("At least one Ising problem is required")
    if beta_grid is None:
        beta_grid = np.linspace(0.02, 5.0, 300)
    beta_grid = np.asarray(beta_grid, dtype=np.float64)
    if beta_grid.ndim != 1 or beta_grid.size == 0 or np.any(beta_grid <= 0):
        raise ValueError("beta_grid must contain positive values")
    loss_curves = []
    total_samples = 0
    n_bits_values = []
    for h, j, samples in problems:
        h = np.asarray(h, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != h.shape[0]:
            raise ValueError("samples must have shape (n_samples, n_bits)")
        if not np.isin(samples, [-1, 1]).all():
            raise ValueError("samples must contain only -1 and +1")
        if j.shape != (h.shape[0], h.shape[0]):
            raise ValueError("j must have shape (n_bits, n_bits)")
        signed_fields = samples * (h[None, :] + samples @ j)
        loss_curves.append(
            np.asarray(
                [
                    np.logaddexp(0.0, -2.0 * beta * signed_fields).mean()
                    for beta in beta_grid
                ]
            )
        )
        total_samples += samples.shape[0]
        n_bits_values.append(samples.shape[1])

    loss_curves = np.stack(loss_curves, axis=0)
    mean_losses = loss_curves.mean(axis=0)
    best_index = int(np.argmin(mean_losses))
    result = {
        "beta_eff": float(beta_grid[best_index]),
        "pseudolikelihood_nll": float(mean_losses[best_index]),
        "n_instances": int(len(problems)),
        "n_samples": int(total_samples),
        "n_bits_min": int(min(n_bits_values)),
        "n_bits_max": int(max(n_bits_values)),
        "grid_boundary": bool(best_index in (0, beta_grid.size - 1)),
    }
    if bootstrap_repetitions:
        rng = np.random.default_rng(seed)
        bootstrap_indices = rng.integers(
            0,
            loss_curves.shape[0],
            size=(int(bootstrap_repetitions), loss_curves.shape[0]),
        )
        bootstrap_losses = loss_curves[bootstrap_indices].mean(axis=1)
        bootstrap_beta = beta_grid[np.argmin(bootstrap_losses, axis=1)]
        result.update(
            {
                "bootstrap_repetitions": int(bootstrap_repetitions),
                "beta_eff_ci_2.5": float(np.quantile(bootstrap_beta, 0.025)),
                "beta_eff_ci_97.5": float(np.quantile(bootstrap_beta, 0.975)),
            }
        )
    return result
