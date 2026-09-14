# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import itertools
import time

import numpy as np

from .ising import expand_trotter_ising, ising_energy, symmetrize_couplings


@dataclass
class IsingSampleResult:
    samples: np.ndarray
    energies: np.ndarray
    metadata: dict = field(default_factory=dict)
    replicas: np.ndarray = None


class BaseIsingSampler:
    """Base class for interchangeable Ising sampler backends."""

    name = "base"

    def sample_ising(self, h, j, num_reads=100, beta=1.0, **kwargs):
        raise NotImplementedError


class ExactIsingSampler(BaseIsingSampler):
    """Exact Boltzmann sampler for small problems. Use only for tests/calibration."""

    name = "exact"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample_ising(self, h, j, num_reads=100, beta=1.0, **kwargs):
        start = time.perf_counter()
        h = np.asarray(h, dtype=np.float64)
        j = symmetrize_couplings(j)
        n_bits = h.shape[0]
        if n_bits > kwargs.get("max_exact_bits", 20):
            raise ValueError("Exact sampling is exponential; lower n_bits or increase max_exact_bits explicitly")

        states = np.asarray(list(itertools.product([-1, 1], repeat=n_bits)), dtype=np.int8)
        energies = ising_energy(states, h, j)
        logits = -beta * energies
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = self.rng.choice(states.shape[0], size=num_reads, replace=True, p=probs)
        samples = states[idx]
        sampled_energies = energies[idx]
        return IsingSampleResult(
            samples=samples,
            energies=sampled_energies,
            metadata={"backend": self.name, "beta": beta, "latency_s": time.perf_counter() - start},
        )


class GibbsIsingSampler(BaseIsingSampler):
    """Random-scan heat-bath sampler with independent persistent-free chains."""

    name = "gibbs"

    def __init__(self, sweeps=200, burn_in=100, seed=None):
        self.sweeps = int(sweeps)
        self.burn_in = int(burn_in)
        self.rng = np.random.default_rng(seed)

    def sample_ising(self, h, j, num_reads=100, beta=1.0, **kwargs):
        start = time.perf_counter()
        h = np.asarray(h, dtype=np.float64)
        j = symmetrize_couplings(j)
        n_bits = h.shape[0]
        sweeps = int(kwargs.get("sweeps", self.sweeps))
        burn_in = int(kwargs.get("burn_in", self.burn_in))
        samples = self.rng.choice([-1, 1], size=(num_reads, n_bits)).astype(np.int8)
        local_fields = h[None, :] + samples @ j

        for _ in range(burn_in + sweeps):
            for index in self.rng.permutation(n_bits):
                probability_up = 1.0 / (
                    1.0 + np.exp(-2.0 * beta * np.clip(local_fields[:, index], -30, 30))
                )
                old_spins = samples[:, index].copy()
                samples[:, index] = np.where(
                    self.rng.random(num_reads) < probability_up, 1, -1
                )
                spin_delta = samples[:, index] - old_spins
                changed = spin_delta != 0
                if np.any(changed):
                    local_fields[changed] += (
                        spin_delta[changed, None] * j[index][None, :]
                    )

        return IsingSampleResult(
            samples=samples,
            energies=ising_energy(samples, h, j),
            metadata={
                "backend": self.name,
                "beta": float(beta),
                "burn_in": burn_in,
                "sweeps": sweeps,
                "latency_s": time.perf_counter() - start,
            },
        )

    def sample_ising_batch(self, h, j, num_reads=100, beta=1.0, **kwargs):
        h = np.asarray(h, dtype=np.float64)
        if h.ndim != 2:
            raise ValueError("batched h must have shape (n_conditions, n_bits)")
        results = [
            self.sample_ising(field, j, num_reads=num_reads, beta=beta, **kwargs)
            for field in h
        ]
        return IsingSampleResult(
            samples=np.stack([result.samples for result in results], axis=0),
            energies=np.stack([result.energies for result in results], axis=0),
            metadata={
                "backend": self.name,
                "batched_conditions": int(h.shape[0]),
                "latency_s": float(
                    sum(result.metadata["latency_s"] for result in results)
                ),
            },
        )


class SimulatedAnnealingSampler(BaseIsingSampler):
    """Simple NumPy simulated annealing backend for QBM development."""

    name = "sa"

    def __init__(self, sweeps=200, beta_start=0.1, beta_end=2.0, seed=None):
        self.sweeps = sweeps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.rng = np.random.default_rng(seed)

    def sample_ising(self, h, j, num_reads=100, beta=None, **kwargs):
        start = time.perf_counter()
        h = np.asarray(h, dtype=np.float64)
        j = symmetrize_couplings(j)
        n_bits = h.shape[0]
        sweeps = int(kwargs.get("sweeps", self.sweeps))
        beta_start = float(kwargs.get("beta_start", self.beta_start))
        beta_end = float(kwargs.get("beta_end", self.beta_end if beta is None else beta))
        betas = np.linspace(beta_start, beta_end, sweeps)

        samples = self.rng.choice([-1, 1], size=(num_reads, n_bits)).astype(np.int8)
        local_fields = h[None, :] + samples @ j

        for b in betas:
            for i in self.rng.permutation(n_bits):
                delta_e = 2.0 * samples[:, i] * local_fields[:, i]
                accept = (delta_e <= 0.0) | (self.rng.random(num_reads) < np.exp(-b * delta_e))
                if not np.any(accept):
                    continue
                old_spin = samples[accept, i].copy()
                samples[accept, i] *= -1
                spin_delta = samples[accept, i] - old_spin
                local_fields[accept] += spin_delta[:, None] * j[i][None, :]

        energies = ising_energy(samples, h, j)
        return IsingSampleResult(
            samples=samples,
            energies=energies,
            metadata={
                "backend": self.name,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "sweeps": sweeps,
                "latency_s": time.perf_counter() - start,
            },
        )

    def sample_ising_batch(self, h, j, num_reads=100, beta=None, **kwargs):
        """Vectorized SA for multiple conditional fields sharing one coupling matrix."""
        start = time.perf_counter()
        h = np.asarray(h, dtype=np.float64)
        j = symmetrize_couplings(j)
        if h.ndim != 2:
            raise ValueError("batched h must have shape (n_conditions, n_bits)")
        n_conditions, n_bits = h.shape
        if j.shape != (n_bits, n_bits):
            raise ValueError("j shape does not match batched h")
        sweeps = int(kwargs.get("sweeps", self.sweeps))
        beta_start = float(kwargs.get("beta_start", self.beta_start))
        beta_end = float(kwargs.get("beta_end", self.beta_end if beta is None else beta))
        betas = np.linspace(beta_start, beta_end, sweeps)

        samples = self.rng.choice(
            [-1, 1], size=(n_conditions, num_reads, n_bits)
        ).astype(np.int8)
        local_fields = h[:, None, :] + np.einsum("brj,ji->bri", samples, j)
        for current_beta in betas:
            for index in self.rng.permutation(n_bits):
                current_spins = samples[:, :, index]
                delta_energy = 2.0 * current_spins * local_fields[:, :, index]
                accept = (delta_energy <= 0.0) | (
                    self.rng.random((n_conditions, num_reads))
                    < np.exp(-current_beta * delta_energy)
                )
                spin_delta = np.where(accept, -2 * current_spins, 0).astype(np.int8)
                samples[:, :, index] += spin_delta
                local_fields += spin_delta[:, :, None] * j[index][None, None, :]

        linear = -np.einsum("bri,bi->br", samples, h)
        quadratic = -0.5 * np.einsum("bri,ij,brj->br", samples, j, samples)
        return IsingSampleResult(
            samples=samples,
            energies=linear + quadratic,
            metadata={
                "backend": self.name,
                "batched_conditions": n_conditions,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "sweeps": sweeps,
                "latency_s": time.perf_counter() - start,
            },
        )


class PathIntegralSampler(BaseIsingSampler):
    """Finite-replica Suzuki-Trotter sampler for a transverse-field Ising prior."""

    name = "path_integral"

    def __init__(self, replicas=4, sweeps=100, seed=None, **kwargs):
        if replicas < 2:
            raise ValueError("replicas must be at least 2")
        self.replicas = int(replicas)
        self.sweeps = int(sweeps)
        self.rng = np.random.default_rng(seed)

    def sample_ising(self, h, j, num_reads=100, beta=1.0, **kwargs):
        start = time.perf_counter()
        h = np.asarray(h, dtype=np.float64)
        j = symmetrize_couplings(j)
        replicas = int(kwargs.get("trotter_replicas", self.replicas))
        gamma = kwargs.get("transverse_gamma")
        if gamma is None:
            raise ValueError("PathIntegralSampler requires transverse_gamma")
        expanded_h, expanded_j = expand_trotter_ising(h, j, beta, gamma, replicas)
        sweeps = int(kwargs.get("sweeps", self.sweeps))
        expanded_spins = self.rng.choice(
            [-1, 1], size=(num_reads, expanded_h.shape[0])
        ).astype(np.int8)
        local_fields = expanded_h[None, :] + expanded_spins @ expanded_j

        for _ in range(sweeps):
            for index in self.rng.permutation(expanded_h.shape[0]):
                probability_up = 1.0 / (1.0 + np.exp(-2.0 * np.clip(local_fields[:, index], -30, 30)))
                old_spin = expanded_spins[:, index].copy()
                expanded_spins[:, index] = np.where(
                    self.rng.random(num_reads) < probability_up, 1, -1
                )
                spin_delta = expanded_spins[:, index] - old_spin
                changed = spin_delta != 0
                if np.any(changed):
                    local_fields[changed] += spin_delta[changed, None] * expanded_j[index][None, :]

        replica_spins = expanded_spins.reshape(num_reads, replicas, h.shape[0])
        selected_slices = self.rng.integers(0, replicas, size=num_reads)
        samples = replica_spins[np.arange(num_reads), selected_slices]
        energies = ising_energy(samples, h, j)
        adjacent_agreement = np.mean(replica_spins * np.roll(replica_spins, -1, axis=1))
        return IsingSampleResult(
            samples=samples,
            energies=energies,
            replicas=replica_spins,
            metadata={
                "backend": self.name,
                "beta": float(beta),
                "trotter_replicas": replicas,
                "transverse_gamma_mean": float(np.asarray(gamma).mean()),
                "replica_disagreement": float(0.5 * (1.0 - adjacent_agreement)),
                "sweeps": sweeps,
                "latency_s": time.perf_counter() - start,
            },
        )


class CIMSamplingBackend(SimulatedAnnealingSampler):
    """
    CIM-compatible placeholder backend.

    It preserves the Ising sampler contract and metadata while using SA until a
    real coherent Ising machine client is connected.
    """

    name = "cim_placeholder"

    def sample_ising(self, h, j, num_reads=100, beta=None, **kwargs):
        result = super().sample_ising(h, j, num_reads=num_reads, beta=beta, **kwargs)
        result.metadata["backend"] = self.name
        result.metadata["hardware_ready"] = False
        return result

    def sample_ising_batch(self, h, j, num_reads=100, beta=None, **kwargs):
        result = super().sample_ising_batch(
            h, j, num_reads=num_reads, beta=beta, **kwargs
        )
        result.metadata["backend"] = self.name
        result.metadata["hardware_ready"] = False
        return result


class BosonicPlatformSampler(BaseIsingSampler):
    """
    Adapter for a post-training bosonic-platform FA-BM-VAE sampler.

    Pass a client object exposing sample_ising(h, j, num_reads, beta, **kwargs).
    The Kaiwu adapter accepts only the exported 49-spin h=0, j=-2Q form; the
    training CLI deliberately blocks this backend so it cannot become a
    training negative phase.
    """

    name = "bosonic_platform"

    def __init__(self, client=None):
        self.client = client

    def sample_ising(self, h, j, num_reads=100, beta=1.0, **kwargs):
        if self.client is None:
            raise NotImplementedError(
                "BosonicPlatformSampler needs a client with sample_ising(h, j, num_reads, beta, **kwargs)."
            )
        start = time.perf_counter()
        result = self.client.sample_ising(h=h, j=j, num_reads=num_reads, beta=beta, **kwargs)
        if isinstance(result, IsingSampleResult):
            result.metadata.setdefault("backend", self.name)
            result.metadata.setdefault("latency_s", time.perf_counter() - start)
            return result
        samples = np.asarray(result["samples"], dtype=np.int8)
        energies = np.asarray(result.get("energies", ising_energy(samples, h, j)), dtype=np.float64)
        metadata = dict(result.get("metadata", {}))
        metadata.setdefault("backend", self.name)
        metadata.setdefault("latency_s", time.perf_counter() - start)
        return IsingSampleResult(samples=samples, energies=energies, metadata=metadata)


def build_sampler(name, **kwargs):
    name = name.lower()
    if name == "exact":
        return ExactIsingSampler(**kwargs)
    if name in {"gibbs", "heat_bath"}:
        return GibbsIsingSampler(**kwargs)
    if name in {"sa", "simulated_annealing"}:
        return SimulatedAnnealingSampler(**kwargs)
    if name in {"path_integral", "path-integral", "pimc"}:
        return PathIntegralSampler(**kwargs)
    if name in {"cim", "cim_placeholder"}:
        return CIMSamplingBackend(**kwargs)
    if name in {"bosonic", "bosonic_platform"}:
        return BosonicPlatformSampler(**kwargs)
    raise ValueError("Unknown sampler backend: %s" % name)
