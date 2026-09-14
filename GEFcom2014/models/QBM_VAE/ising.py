# -*- coding: utf-8 -*-

import numpy as np


def binary_to_spin(z):
    """Map binary variables {0, 1} to Ising spins {-1, +1}."""
    return 2 * np.asarray(z) - 1


def spin_to_binary(s):
    """Map Ising spins {-1, +1} to binary variables {0, 1}."""
    return ((np.asarray(s) + 1) / 2).astype(np.int64)


def ising_energy(samples, h, j):
    """
    Compute E(s) = - h^T s - 1/2 s^T J s.

    samples can be shape (n_bits,) or (n_samples, n_bits).
    """
    s = np.asarray(samples, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    squeeze = False
    if s.ndim == 1:
        s = s[None, :]
        squeeze = True
    linear = -s @ h
    quadratic = -0.5 * np.einsum("bi,ij,bj->b", s, j, s)
    energy = linear + quadratic
    return energy[0] if squeeze else energy


def symmetrize_couplings(j):
    """Return a zero-diagonal symmetric coupling matrix."""
    j = np.asarray(j, dtype=np.float64)
    j = 0.5 * (j + j.T)
    np.fill_diagonal(j, 0.0)
    return j


def trotter_coupling(beta, transverse_gamma, replicas, min_gamma=1e-6):
    """Return K = 0.5 log(coth(beta * Gamma / M)) for each latent bit."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    if replicas < 2:
        raise ValueError("replicas must be at least 2")
    gamma = np.asarray(transverse_gamma, dtype=np.float64)
    if np.any(gamma < 0):
        raise ValueError("transverse_gamma must be non-negative")
    argument = beta * np.maximum(gamma, min_gamma) / replicas
    return -0.5 * np.log(np.tanh(argument))


def expand_trotter_ising(h, j, beta, transverse_gamma, replicas):
    """Map a transverse-field Ising model to its finite-M path-integral graph.

    The returned coefficients define the dimensionless action through the
    existing energy convention E(s) = -h^T s - 0.5 s^T J s. Sampling this
    expanded graph at inverse temperature one approximates the quantum Gibbs
    distribution at the requested ``beta``.
    """
    h = np.asarray(h, dtype=np.float64)
    j = symmetrize_couplings(j)
    gamma = np.asarray(transverse_gamma, dtype=np.float64)
    if gamma.ndim == 0:
        gamma = np.full(h.shape[0], float(gamma), dtype=np.float64)
    if gamma.shape != h.shape:
        raise ValueError("transverse_gamma must be scalar or have shape (n_bits,)")

    n_bits = h.shape[0]
    expanded_h = np.tile((beta / replicas) * h, replicas)
    expanded_j = np.zeros((replicas * n_bits, replicas * n_bits), dtype=np.float64)
    intra_slice = (beta / replicas) * j
    inter_slice = trotter_coupling(beta, gamma, replicas)
    for replica in range(replicas):
        block = slice(replica * n_bits, (replica + 1) * n_bits)
        expanded_j[block, block] = intra_slice
        next_replica = (replica + 1) % replicas
        for bit in range(n_bits):
            source = replica * n_bits + bit
            target = next_replica * n_bits + bit
            # For M=2 the periodic ring contains two bonds between the same
            # replica pair, so accumulation (rather than assignment) matters.
            expanded_j[source, target] += inter_slice[bit]
            expanded_j[target, source] += inter_slice[bit]
    return expanded_h, symmetrize_couplings(expanded_j)


def collapse_trotter_samples(samples, physical_n_bits, replicas, selection="first", seed=None):
    """Extract physical z-basis samples from an expanded Trotter response."""
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != physical_n_bits * replicas:
        raise ValueError("samples do not match physical_n_bits * replicas")
    replica_samples = samples.reshape(samples.shape[0], replicas, physical_n_bits)
    if selection == "first":
        return replica_samples[:, 0, :]
    if selection == "random":
        rng = np.random.default_rng(seed)
        selected = rng.integers(0, replicas, size=samples.shape[0])
        return replica_samples[np.arange(samples.shape[0]), selected]
    raise ValueError("selection must be 'first' or 'random'")


def _period_mi_matrix(series, n_bins=10):
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("series must have shape (n_samples, n_periods)")
    if x.shape[0] < 2:
        raise ValueError("series must contain at least two samples")

    n_periods = x.shape[1]
    discretized = np.zeros_like(x, dtype=np.int64)
    for t in range(n_periods):
        edges = np.histogram_bin_edges(x[:, t], bins=n_bins)
        discretized[:, t] = np.digitize(x[:, t], edges[1:-1])

    mi = np.zeros((n_periods, n_periods), dtype=np.float64)
    for i in range(n_periods):
        for k in range(i + 1, n_periods):
            mi_value = _mutual_information(discretized[:, i], discretized[:, k])
            mi[i, k] = mi_value
            mi[k, i] = mi_value
    return mi


def _top_k_period_graph(mi, top_k, threshold=0.0, max_lag=None):
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    n_periods = mi.shape[0]
    adj_period = np.zeros_like(mi)
    for i in range(n_periods):
        candidates = np.argsort(mi[i])[::-1]
        selected = [
            k for k in candidates
            if k != i
            and mi[i, k] > threshold
            and (max_lag is None or abs(k - i) <= max_lag)
        ][:top_k]
        for k in selected:
            adj_period[i, k] = mi[i, k]
            adj_period[k, i] = mi[i, k]
    return adj_period


def _expand_period_graph(adj_period, latent_per_period):
    if latent_per_period == 1:
        return adj_period
    n_periods = adj_period.shape[0]
    n_bits = n_periods * latent_per_period
    adj = np.zeros((n_bits, n_bits), dtype=np.float64)
    for i in range(n_periods):
        for k in range(n_periods):
            if adj_period[i, k] == 0:
                continue
            block_i = slice(i * latent_per_period, (i + 1) * latent_per_period)
            block_k = slice(k * latent_per_period, (k + 1) * latent_per_period)
            adj[block_i, block_k] = adj_period[i, k] / latent_per_period
    np.fill_diagonal(adj, 0.0)
    return symmetrize_couplings(adj)


def temporal_mi_sparse_graph(series, latent_per_period=1, top_k=2, threshold=0.0, n_bins=10):
    """
    Build a sparse temporal Ising adjacency from mutual information between periods.

    series is expected to be shaped (n_days, n_periods), for example
    GEFCom daily targets with 24 hourly values. The returned matrix has
    size (n_periods * latent_per_period, n_periods * latent_per_period).
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("series must have shape (n_samples, n_periods)")
    if latent_per_period < 1:
        raise ValueError("latent_per_period must be >= 1")

    mi = _period_mi_matrix(x, n_bins=n_bins)
    adj_period = _top_k_period_graph(mi, top_k=top_k, threshold=threshold)
    return _expand_period_graph(adj_period, latent_per_period)


def stable_temporal_mi_sparse_graph(
    series,
    latent_per_period=1,
    top_k=2,
    threshold=0.0,
    n_bins=10,
    bootstrap_repetitions=100,
    stability_threshold=0.7,
    max_lag=None,
    seed=0,
):
    """Build an LS-only stability-selected temporal MI graph.

    Each bootstrap sample proposes a symmetric top-k period graph. An edge is
    retained only when its selection frequency reaches ``stability_threshold``;
    retained edges use the full-LS mutual information as their interpretable
    mask weight.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("series must have shape (n_samples, n_periods)")
    if latent_per_period < 1:
        raise ValueError("latent_per_period must be >= 1")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be >= 1")
    if not 0 < stability_threshold <= 1:
        raise ValueError("stability_threshold must be in (0, 1]")
    if max_lag is not None and max_lag < 1:
        raise ValueError("max_lag must be >= 1 when provided")

    full_mi = _period_mi_matrix(x, n_bins=n_bins)
    selection_count = np.zeros_like(full_mi, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_repetitions):
        indices = rng.integers(0, x.shape[0], size=x.shape[0])
        bootstrap_mi = _period_mi_matrix(x[indices], n_bins=n_bins)
        bootstrap_graph = _top_k_period_graph(
            bootstrap_mi,
            top_k=top_k,
            threshold=threshold,
            max_lag=max_lag,
        )
        selection_count += bootstrap_graph != 0

    selection_frequency = selection_count / float(bootstrap_repetitions)
    support = selection_frequency >= stability_threshold
    if max_lag is not None:
        periods = np.arange(x.shape[1])
        support &= np.abs(periods[:, None] - periods[None, :]) <= max_lag
    support &= full_mi > threshold
    adj_period = np.where(support, full_mi, 0.0)
    np.fill_diagonal(adj_period, 0.0)
    return _expand_period_graph(adj_period, latent_per_period)


def stable_partial_correlation_graph(
    features,
    top_k=3,
    bootstrap_repetitions=100,
    stability_threshold=0.7,
    covariance_ridge=0.1,
    seed=0,
):
    """Build a stability-selected graph from posterior residual dependence."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("features must have shape (n_samples, n_nodes)")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be >= 1")
    if not 0 < stability_threshold <= 1:
        raise ValueError("stability_threshold must be in (0, 1]")
    if covariance_ridge <= 0:
        raise ValueError("covariance_ridge must be positive")

    def partial_correlation(values):
        standardized = values - values.mean(axis=0, keepdims=True)
        scale = standardized.std(axis=0, keepdims=True)
        standardized /= np.where(scale > 1e-8, scale, 1.0)
        covariance = np.cov(standardized, rowvar=False)
        precision = np.linalg.pinv(
            covariance + covariance_ridge * np.eye(covariance.shape[0])
        )
        denominator = np.sqrt(
            np.maximum(np.diag(precision)[:, None] * np.diag(precision)[None, :], 1e-12)
        )
        partial = -precision / denominator
        np.fill_diagonal(partial, 0.0)
        return partial

    def top_k_support(partial):
        support = np.zeros_like(partial, dtype=bool)
        for node in range(partial.shape[0]):
            candidates = np.argsort(np.abs(partial[node]))[::-1]
            selected = [index for index in candidates if index != node][:top_k]
            support[node, selected] = True
        return support | support.T

    full_partial = partial_correlation(x)
    count = np.zeros_like(full_partial, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_repetitions):
        indices = rng.integers(0, x.shape[0], size=x.shape[0])
        count += top_k_support(partial_correlation(x[indices]))
    support = count / float(bootstrap_repetitions) >= stability_threshold
    weights = np.where(support, np.abs(full_partial), 0.0)
    np.fill_diagonal(weights, 0.0)
    return symmetrize_couplings(weights)


def _mutual_information(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    joint, _, _ = np.histogram2d(a, b, bins=(np.unique(a).size, np.unique(b).size))
    joint = joint / np.maximum(joint.sum(), 1.0)
    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)
    expected = p_a @ p_b
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
