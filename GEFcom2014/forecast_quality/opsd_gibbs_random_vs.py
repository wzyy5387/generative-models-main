# -*- coding: utf-8 -*-
"""Offline VS comparison of Gibbs, saved quantized-SA, and random spins."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import scenario_metrics
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import (
    ENERGY_CONVENTION, build_task_name, hardware_matrix_from_ising, matrix_sha256,
    normalize_hardware_spins,
    platform_task_name,
)
from GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios import decode_platform_spins
from GEFcom2014.forecast_quality.opsd_sa_state_diversity_vs import (
    _date, _json, _path, _sha, _standard_noise, _validate_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json"
PREPARATION = ROOT / "export/opsd_sa_negative_kaiwu_preparation_10read_v1"
DATASET = ROOT / "GEFcom2014/data/external/opsd_wind_daily.npz"
PRIOR_ANALYSIS = ROOT / "export/opsd_sa_state_diversity_vs_v4"
DEFAULT_OUTPUT = ROOT / "export/opsd_gibbs_random_vs_v5"

# Frozen before sampling. One systematic sweep updates each hardware spin once;
# output samples are thinned by complete sweeps.
CHAINS = 4
WARMUP_SWEEPS = 2000
RETAINED_PER_CHAIN = 1000
THIN_SWEEPS = 5
GIBBS_SEED_BASE = 2026110600
RANDOM_SEED_BASE = 2026110700
BOOTSTRAP_SEED = 2026110800
BOOTSTRAP_REPETITIONS = 20000
RHAT_MAX = 1.05
ESS_MIN = 400
FLIP_RATE_RANGE = (0.01, 0.99)
N_BITS = 49
N_LOGICAL = 48
N_PREDICTIONS = 100


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def sdk_energy(matrix, spins):
    """Wide-precision SDK Hamiltonian H=-u.T K u."""
    k = np.asarray(matrix, dtype=np.float64)
    u = np.asarray(spins, dtype=np.float64)
    return -float(u @ k @ u)


def logical_energy(h, j, spins):
    s = np.asarray(spins, dtype=np.float64)
    return -float(np.asarray(h, dtype=np.float64) @ s) - 0.5 * float(
        s @ np.asarray(j, dtype=np.float64) @ s
    )


def _conditional_plus_probability(couplings, spins, beta, index):
    field = float((2.0 * couplings[index]) @ spins)
    return _sigmoid(2.0 * beta * field)


def _one_gibbs_update(couplings, spins, fields, beta, index, rng):
    probability = _sigmoid(2.0 * beta * float(fields[index]))
    new_spin = 1 if rng.random() < probability else -1
    old_spin = int(spins[index])
    if new_spin != old_spin:
        spins[index] = new_spin
        fields += (2.0 * couplings[:, index]) * (new_spin - old_spin)
        return 1
    return 0


def gibbs_chain(matrix, beta, seed, warmup, retained, thin, initial):
    """Sequential systematic-scan Gibbs for H=-u.T K u."""
    k = np.asarray(matrix, dtype=np.float64)
    if (k.shape != (N_BITS, N_BITS) or not np.isfinite(k).all()
            or not np.array_equal(k, k.T) or np.any(np.diag(k) != 0)):
        raise ValueError("Gibbs K must be finite, symmetric, zero-diagonal 49x49")
    if beta <= 0 or warmup < 0 or retained < 1 or thin < 1:
        raise ValueError("Invalid Gibbs configuration")
    couplings = k.copy()  # H=-1/2 u.T (2K) u
    rng = np.random.default_rng(int(seed))
    spins = np.asarray(initial, dtype=np.int8).copy()
    if spins.shape != (N_BITS,) or not np.isin(spins, (-1, 1)).all():
        raise ValueError("Gibbs initial state must contain 49 spins in {-1,+1}")
    fields = (2.0 * couplings) @ spins.astype(np.float64)
    total_sweeps = warmup + retained * thin
    states = np.empty((retained, N_BITS), dtype=np.int8)
    energy = np.empty(retained, dtype=np.float64)
    logical_mag = np.empty(retained, dtype=np.float64)
    flips = 0
    kept = 0
    for sweep in range(total_sweeps):
        for index in range(N_BITS):
            flips += _one_gibbs_update(couplings, spins, fields, beta, index, rng)
        if sweep >= warmup and (sweep - warmup + 1) % thin == 0:
            states[kept] = spins
            energy[kept] = sdk_energy(k, spins)
            logical = normalize_hardware_spins(spins[None, :], N_LOGICAL)[0]
            logical_mag[kept] = logical.mean()
            kept += 1
    if kept != retained:
        raise AssertionError("Gibbs retained sample count mismatch")
    return {
        "hardware_states": states,
        "energy": energy,
        "logical_magnetization": logical_mag,
        "flip_rate": flips / float(total_sweeps * N_BITS),
    }


def _rhat(chains):
    x = np.asarray(chains, dtype=np.float64)
    m, n = x.shape
    if m < 2 or n < 2:
        return float("inf")
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    between = n * float(np.var(np.mean(x, axis=1), ddof=1))
    if within == 0:
        return 1.0 if between == 0 else float("inf")
    variance = ((n - 1.0) / n) * within + between / n
    return float(math.sqrt(max(0.0, variance / within)))


def _ess(chains):
    x = np.asarray(chains, dtype=np.float64)
    m, n = x.shape
    if n < 2:
        return float(m * n)
    centered = x - x.mean(axis=1, keepdims=True)
    variance = float(np.mean(np.sum(centered * centered, axis=1) / n))
    if variance <= 1e-15:
        return float(m * n)
    rho = [1.0]
    # Direct autocorrelation is bounded by the fixed 1000 retained draws.
    for lag in range(1, n):
        covariance = np.mean(np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / (n - lag))
        rho.append(float(covariance / variance))
    positive_sum = 0.0
    for lag in range(1, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0:
            break
        positive_sum += pair
    tau = max(1.0, 1.0 + 2.0 * positive_sum)
    return float(min(m * n, (m * n) / tau))


def _diagnose(chains, edge_pairs, flips):
    observables = {}
    for key in ("energy", "logical_magnetization"):
        values = np.stack([chain[key] for chain in chains])
        observables[key] = {"rhat": _rhat(values), "ess": _ess(values),
                            "chain_means": values.mean(axis=1).astype(float).tolist()}
    logicals = [normalize_hardware_spins(chain["hardware_states"], N_LOGICAL) for chain in chains]
    for edge_index, (i, j) in enumerate(edge_pairs):
        values = np.stack([logical[:, i] * logical[:, j] for logical in logicals])
        observables["edge_%02d_%02d" % (i, j)] = {
            "rhat": _rhat(values), "ess": _ess(values),
            "chain_means": values.mean(axis=1).astype(float).tolist(),
        }
    rates = [float(chain["flip_rate"]) for chain in chains]
    repeated_fraction = [float(1.0 - np.unique(samples, axis=0).shape[0] / len(samples))
                         for samples in logicals]
    rhat_max = max(row["rhat"] for row in observables.values())
    ess_min = min(row["ess"] for row in observables.values())
    passed = (np.isfinite(rhat_max) and rhat_max <= RHAT_MAX and ess_min >= ESS_MIN
              and all(FLIP_RATE_RANGE[0] <= rate <= FLIP_RATE_RANGE[1] for rate in rates))
    return {
        "status": "PASS" if passed else "NOT_PASSED",
        "rhat_max": float(rhat_max), "ess_min": float(ess_min),
        "flip_rate_by_chain": rates, "repeated_state_fraction_by_chain": repeated_fraction,
        "observables": observables,
        "criteria": {"rhat_max": RHAT_MAX, "ess_min": ESS_MIN,
                     "flip_rate_range": list(FLIP_RATE_RANGE)},
    }


def _representative_edges(matrix, count=5):
    logical_block = np.asarray(matrix, dtype=np.float64)[:N_LOGICAL, :N_LOGICAL]
    candidates = [(abs(float(logical_block[i, j])), i, j)
                  for i in range(N_LOGICAL) for j in range(i + 1, N_LOGICAL)
                  if logical_block[i, j] != 0]
    if not candidates:
        raise ValueError("No logical couplings are available for edge diagnostics")
    candidates.sort(reverse=True)
    return [(i, j) for _, i, j in candidates[:count]]


def _observables(logical, h, j, edge_pairs):
    logical = np.asarray(logical, dtype=np.int8)
    energies = np.asarray([logical_energy(h, j, row) for row in logical], dtype=np.float64)
    return {
        "unique_states": int(np.unique(logical, axis=0).shape[0]),
        "dominant_state_count": int(np.unique(logical, axis=0, return_counts=True)[1].max()),
        "dominant_state_fraction": float(np.unique(logical, axis=0, return_counts=True)[1].max() / len(logical)),
        "logical_energy_mean": float(energies.mean()),
        "logical_energy_std": float(energies.std(ddof=1)) if len(energies) > 1 else 0.0,
        "mean_spin": logical.mean(axis=0).astype(float).tolist(),
        "mean_magnetization": float(logical.mean()),
        "representative_edge_correlations": {
            "%d-%d" % (i, j): float(np.mean(logical[:, i] * logical[:, j]))
            for i, j in edge_pairs
        },
    }


def exact_self_test():
    h = np.array([0.23, -0.31], dtype=np.float64)
    hardware_states = [np.array([a, b, aux], dtype=np.int8)
                       for a, b, aux in itertools.product((-1, 1), repeat=3)]
    for coupling in (0.41, -0.41):
        j = np.array([[0.0, coupling], [coupling, 0.0]], dtype=np.float64)
        k = hardware_matrix_from_ising(h, j, logical_n_bits=2)
        for u in hardware_states:
            s = normalize_hardware_spins(u[None, :], 2)[0]
            if not np.isclose(sdk_energy(k, u), logical_energy(h, j, s), atol=1e-12):
                raise AssertionError("Auxiliary-spin energy/gauge equivalence failed")

    # Enumerate a separate two-spin target and check its exact conditionals.
    small_k = np.array([[0.0, 0.7], [0.7, 0.0]])
    beta = 0.8
    all_states = [np.array(s, dtype=np.int8) for s in itertools.product((-1, 1), repeat=2)]
    energies = np.array([sdk_energy(small_k, s) for s in all_states])
    exact = np.exp(-beta * (energies - energies.min()))
    exact /= exact.sum()
    exact_pair_correlation = float(sum(p * state[0] * state[1] for p, state in zip(exact, all_states)))
    for state in all_states:
        conditional = _conditional_plus_probability(small_k, state.astype(float), beta, 0)
        plus = state.copy(); plus[0] = 1
        minus = state.copy(); minus[0] = -1
        exact_cond = math.exp(-beta * sdk_energy(small_k, plus))
        exact_cond /= exact_cond + math.exp(-beta * sdk_energy(small_k, minus))
        if not np.isclose(conditional, exact_cond, atol=1e-12):
            raise AssertionError("Gibbs conditional does not match exact enumeration")
    padded_k = np.pad(small_k, ((0, N_BITS - 2), (0, N_BITS - 2)))
    logical_j = np.zeros((N_LOGICAL, N_LOGICAL), dtype=np.float64)
    logical_j[0, 1] = logical_j[1, 0] = small_k[0, 1]
    synthetic = {"matrix": padded_k, "gain": 1.0 / beta, "export_index": 0,
                 "J": logical_j}
    batched = _sample_gibbs_batch([synthetic], warmup=250, retained=2000, thin=1, seed_base=917)
    sampled_logical, diagnostic, _, chain_meta = batched[0]
    if sampled_logical.shape != (10, N_LOGICAL) or len(chain_meta["selected_hardware_states"]) != 10:
        raise AssertionError("Gibbs sample shape self-test failed")
    empirical_pair_correlation = float(np.mean(
        diagnostic["observables"]["edge_00_01"]["chain_means"]
    ))
    if abs(empirical_pair_correlation - exact_pair_correlation) > 0.05:
        raise AssertionError("Gibbs sampler does not match enumerated small-system distribution")
    for aux in (-1, 1):
        u = np.array([[1, -1, aux]], dtype=np.int8)
        normalized = normalize_hardware_spins(u, 2)
        if normalized.shape != (1, 2) or not np.isin(normalized, (-1, 1)).all():
            raise AssertionError("Auxiliary spin normalization failed")
    return {"exact_states": len(all_states), "energy_and_conditionals": "PASS",
            "positive_and_negative_coupling_energy": "PASS",
            "auxiliary_gauge_equivalence": "PASS", "sampler_shape": "PASS",
            "exact_pair_correlation": exact_pair_correlation,
            "empirical_pair_correlation": empirical_pair_correlation}


def _load_frozen():
    protocol = _json(PROTOCOL)
    prep = _json(PREPARATION / "preparation_manifest.json")
    reference = _json(PREPARATION / "quality_reference_manifest.json")
    _validate_inputs(protocol, prep, reference)
    checkpoint = _path(protocol["model"]["checkpoint"])
    dataset_path = _path(protocol["dataset"]["source"])
    frozen_audit = _json(PREPARATION / "frozen_model_audit.json")
    if (_sha(checkpoint) != prep["model_checkpoint_sha256"]
            or _sha(checkpoint) != frozen_audit["files"]["checkpoint"]["sha256"]):
        raise ValueError("Frozen checkpoint identity/hash mismatch")
    if _sha(dataset_path) != prep["dataset_sha256"]:
        raise ValueError("Frozen data bundle hash mismatch")
    config_path = _path(protocol["model"]["model_config"])
    if _sha(config_path) != _sha(checkpoint.with_suffix(".json")):
        raise ValueError("Frozen model configuration hash mismatch")
    with np.load(DATASET, allow_pickle=False) as archive:
        x_ls = np.asarray(archive["x_ls"])
        y_ls = np.asarray(archive["y_ls"])
        x_vs = np.asarray(archive["x_vs"])
        y_vs = np.asarray(archive["y_vs"])
        dates = np.asarray(archive["dates_vs"])
    _, _, x_scaled, _, _, _, scaler = scale_data_multi(x_ls, y_ls, x_vs, y_vs)
    conditions = np.concatenate((x_scaled, scaler.transform(x_vs[:, :24])), axis=1).astype(np.float32)
    with Path(_path(protocol["model"]["checkpoint"])).open("rb") as handle:
        model = pickle.load(handle)
    model.to("cpu"); model.device = torch.device("cpu"); model.eval()
    return protocol, prep, reference, model, scaler, conditions, y_vs, dates


def _collect_conditions(reference, prep, protocol, conditions, y_vs, dates):
    selection = _json(PREPARATION / "vs_selection_manifest.json")
    if selection.get("test_data_used") is not False:
        raise ValueError("VS selection manifest must explicitly exclude TEST")
    if _sha(PREPARATION / "vs_selection_manifest.json") != prep["manifest_hashes"]["vs_selection_manifest"]:
        raise ValueError("Frozen VS selection manifest hash mismatch")
    index_to_export = {int(index): n for n, index in enumerate(selection["indices"])}
    manifest_hashes = prep["manifest_hashes"]["vs_manifests"]
    source_manifest = {}
    for path_value in prep["vs_manifests"]:
        path = _path(path_value)
        gain = int(float(_json(path)[0]["hardware_gain"]))
        if _sha(path) != manifest_hashes[str(gain)]:
            raise ValueError("Frozen logical source manifest hash mismatch")
        for row in _json(path):
            source_manifest[(float(gain), str(row["id"]))] = (row, path)

    results = []
    entries = sorted(reference["instances"], key=lambda x: (str(x["instance_id"]), int(x["gain"])))
    if len(entries) != 30:
        raise ValueError("Expected exactly 30 frozen VS SA references")
    for ref in entries:
        gain = int(ref["gain"]); instance = str(ref["instance_id"])
        source_entry = source_manifest.get((float(gain), instance))
        if source_entry is None:
            raise ValueError("No source row for %s gain %d" % (instance, gain))
        row, source_manifest_path = source_entry
        index, export_index = int(row["index"]), int(row["export_index"])
        date = _date(row["date"])
        if (index_to_export.get(index) != export_index or _date(dates[index]) != date
                or _date(ref["date"]) != date or row.get("split") != "VS"):
            raise ValueError("Frozen VS index/date mapping mismatch: %s" % instance)
        src_path = _path(row["path"])
        if src_path != _path(ref["source_problem_path"]) or _sha(src_path) != ref["source_problem_sha256"]:
            raise ValueError("Frozen condition source identity/hash mismatch")
        matrix_path = _path(ref["hardware_matrix_path"])
        if _sha(matrix_path) != ref["hardware_matrix_file_sha256"]:
            raise ValueError("Frozen matrix file hash mismatch")
        with np.load(matrix_path, allow_pickle=False) as archive:
            matrix = np.asarray(archive["hardware_matrix"])
            convention = str(np.asarray(archive["hardware_energy_convention"]).reshape(-1)[0])
        if convention != ENERGY_CONVENTION or matrix_sha256(matrix) != ref["hardware_matrix_sha256"]:
            raise ValueError("Frozen matrix convention/hash mismatch")
        sample_path = _path(ref["reference_sample_path"])
        if _sha(sample_path) != ref["reference_sample_sha256"]:
            raise ValueError("Frozen SA sample hash mismatch")
        with np.load(sample_path, allow_pickle=False) as archive:
            sa = np.asarray(archive["samples"])
            raw_hardware = np.asarray(archive["hardware_samples"])
            if (sa.shape != (10, N_LOGICAL) or not np.isin(sa, (-1, 1)).all()
                    or not np.array_equal(sa, normalize_hardware_spins(raw_hardware, N_LOGICAL))):
                raise ValueError("Frozen SA logical/hardware sample mismatch")
            expected_sample_metadata = {
                "instance_id": instance,
                "hardware_matrix_sha256": ref["hardware_matrix_sha256"],
                "protocol_sha256": _sha(PROTOCOL),
                "requested_reads": 10,
                "sweeps": int(ref["sampler"].get("sweeps", -1)),
                "random_seed": int(ref["sampler"].get("seed", -1)),
            }
            for field, expected in expected_sample_metadata.items():
                if field not in archive or np.asarray(archive[field]).reshape(-1)[0].item() != expected:
                    raise ValueError("Frozen SA response metadata mismatch: %s" % field)
        with np.load(src_path, allow_pickle=False) as archive:
            h, j = np.asarray(archive["h"], dtype=np.float64), np.asarray(archive["J"], dtype=np.float64)
            x_cond = np.asarray(archive["x_cond"], dtype=np.float32)
            source_matrix = np.asarray(archive["hardware_matrix"])
            if source_matrix.shape != (N_BITS, N_BITS):
                raise ValueError("Frozen logical source lacks 49-spin hardware matrix")
        if not np.array_equal(matrix, source_matrix):
            raise ValueError("Matrix arrays differ between source problem and reference manifest")
        if not np.allclose(x_cond, conditions[index], rtol=0.0, atol=1e-6):
            raise ValueError("Frozen condition input differs from VS data row")
        sampler = ref["sampler"]
        if (sampler.get("backend") != "sa" or sampler.get("matrix_space") != "hardware-quantized"
                or int(sampler.get("reads", -1)) != 10):
            raise ValueError("Reference is not a ten-read quantized-matrix SA sample")
        if matrix.shape != (N_BITS, N_BITS) or not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix)):
            raise ValueError("Invalid frozen 49-spin matrix")
        results.append({"instance_id": instance, "date": date, "index": index,
                        "export_index": export_index, "gain": gain,
                        "matrix": matrix.astype(np.int16), "matrix_sha256": matrix_sha256(matrix),
                        "matrix_file": matrix_path, "source_file": src_path,
                        "source_sha256": _sha(src_path), "sample_file": sample_path,
                        "sample_sha256": _sha(sample_path), "sa": sa, "h": h, "J": j,
                        "condition": conditions[index], "target": y_vs[index],
                        "sampler": sampler, "source_manifest_file": source_manifest_path})
    return results


def _sample_gibbs(item):
    matrix = item["matrix"]
    beta = 1.0 / float(item["gain"])
    edges = _representative_edges(matrix)
    chain_rows = []
    seed_base = GIBBS_SEED_BASE + item["export_index"] * 1000 + item["gain"]
    for chain_index in range(CHAINS):
        if chain_index == 0:
            initial = np.ones(N_BITS, dtype=np.int8)
        elif chain_index == 1:
            initial = -np.ones(N_BITS, dtype=np.int8)
        else:
            init_rng = np.random.default_rng(seed_base + 100 + chain_index)
            initial = init_rng.choice((-1, 1), size=N_BITS).astype(np.int8)
        chain_rows.append(gibbs_chain(matrix, beta, seed_base + chain_index,
                                      WARMUP_SWEEPS, RETAINED_PER_CHAIN,
                                      THIN_SWEEPS, initial))
    diagnostic = _diagnose(chain_rows, edges, [x["flip_rate"] for x in chain_rows])
    all_hardware = np.concatenate([x["hardware_states"] for x in chain_rows], axis=0)
    all_logical = normalize_hardware_spins(all_hardware, N_LOGICAL)
    selected_indices = np.linspace(0, len(all_logical) - 1, 10, dtype=np.int64)
    selected = all_logical[selected_indices]
    selected_chain_ids = (selected_indices // RETAINED_PER_CHAIN).astype(int)
    selected_within_chain = (selected_indices % RETAINED_PER_CHAIN).astype(int)
    return selected, diagnostic, edges, {
        "chain_seed_base": seed_base,
        "chain_seeds": [seed_base + i for i in range(CHAINS)],
        "initialization": ["all_plus", "all_minus", "independent_random", "independent_random"],
        "selected_chain_ids": selected_chain_ids.tolist(),
        "selected_within_chain_indices": selected_within_chain.tolist(),
        "selected_hardware_states": all_hardware[selected_indices].tolist(),
        "retained_samples_per_chain": RETAINED_PER_CHAIN,
        "total_diagnostic_draws": int(len(all_hardware)),
        "target_beta_for_integer_K": beta,
        "equivalent_target_energy": "H_sdk(K,u)/gain, where H_sdk=-u^T K u",
    }


def _sample_gibbs_batch(items, warmup=WARMUP_SWEEPS,
                        retained=RETAINED_PER_CHAIN, thin=THIN_SWEEPS,
                        seed_base=GIBBS_SEED_BASE):
    """Run all frozen conditions/chains together to keep the exact scan fast."""
    total = len(items) * CHAINS
    total_sweeps = warmup + retained * thin
    matrices = np.stack([items[i // CHAINS]["matrix"] for i in range(total)]).astype(np.float64)
    couplings = 2.0 * matrices
    betas = np.asarray([1.0 / items[i // CHAINS]["gain"] for i in range(total)], dtype=np.float64)
    seeds, inits, uniforms = [], np.empty((total, N_BITS), dtype=np.int8), np.empty(
        (total, total_sweeps, N_BITS), dtype=np.float32
    )
    for row in range(total):
        item_index, chain_index = divmod(row, CHAINS)
        item = items[item_index]
        seed = int(seed_base + item["export_index"] * 1000 + round(item["gain"]) + chain_index)
        rng = np.random.default_rng(seed)
        if chain_index == 0:
            initial = np.ones(N_BITS, dtype=np.int8)
        elif chain_index == 1:
            initial = -np.ones(N_BITS, dtype=np.int8)
        else:
            initial = rng.choice((-1, 1), size=N_BITS).astype(np.int8)
        inits[row] = initial
        uniforms[row] = rng.random((total_sweeps, N_BITS), dtype=np.float32)
        seeds.append(int(seed))

    spins = inits.copy()
    fields = np.einsum("bij,bj->bi", couplings, spins.astype(np.float64), optimize=True)
    retained_states = np.empty((total, retained, N_BITS), dtype=np.int8)
    retained_energy = np.empty((total, retained), dtype=np.float64)
    retained_magnetization = np.empty((total, retained), dtype=np.float64)
    flip_counts = np.zeros(total, dtype=np.int64)
    kept = 0
    for sweep in range(total_sweeps):
        for index in range(N_BITS):
            logits = np.clip(2.0 * betas * fields[:, index], -60.0, 60.0)
            probability_plus = 1.0 / (1.0 + np.exp(-logits))
            new_spins = np.where(uniforms[:, sweep, index] < probability_plus, 1, -1).astype(np.int8)
            delta = (new_spins - spins[:, index]).astype(np.float64)
            flip_counts += delta != 0
            spins[:, index] = new_spins
            fields += delta[:, None] * couplings[:, :, index]
        if sweep >= warmup and (sweep - warmup + 1) % thin == 0:
            retained_states[:, kept] = spins
            retained_energy[:, kept] = -0.5 * np.sum(spins.astype(np.float64) * fields, axis=1)
            logical = spins[:, :N_LOGICAL] * spins[:, N_LOGICAL, None]
            retained_magnetization[:, kept] = logical.mean(axis=1)
            kept += 1
    if kept != retained:
        raise AssertionError("Batched Gibbs retained sample count mismatch")

    results = []
    for item_index, item in enumerate(items):
        start = item_index * CHAINS
        chains = []
        for chain_index in range(CHAINS):
            row = start + chain_index
            chains.append({"hardware_states": retained_states[row],
                           "energy": retained_energy[row],
                           "logical_magnetization": retained_magnetization[row],
            "flip_rate": float(flip_counts[row] / (total_sweeps * N_BITS))})
        edge_pairs = _representative_edges(item["matrix"])
        diagnostic = _diagnose(chains, edge_pairs, [x["flip_rate"] for x in chains])
        all_hardware = np.concatenate([chain["hardware_states"] for chain in chains], axis=0)
        all_logical = normalize_hardware_spins(all_hardware, N_LOGICAL)
        selected_indices = np.linspace(0, len(all_logical) - 1, 10, dtype=np.int64)
        results.append((all_logical[selected_indices], diagnostic, edge_pairs, {
            "chain_seed_base": int(seed_base + item["export_index"] * 1000 + round(item["gain"])),
            "chain_seeds": seeds[start:start + CHAINS],
            "initialization": ["all_plus", "all_minus", "independent_random", "independent_random"],
            "selected_chain_ids": (selected_indices // retained).astype(int).tolist(),
            "selected_within_chain_indices": (selected_indices % retained).astype(int).tolist(),
            "selected_logical_states": all_logical[selected_indices].tolist(),
            "selected_hardware_states": all_hardware[selected_indices].tolist(),
            "retained_samples_per_chain": retained,
            "warmup_sweeps": warmup, "thinning_sweeps": thin,
            "total_diagnostic_draws": int(len(all_hardware)),
            "target_beta_for_integer_K": 1.0 / float(item["gain"]),
            "equivalent_target_energy": "H_sdk(K,u)/gain, where H_sdk=-u^T K u",
        }))
    return results


def _decode_and_score(item, samples, method, edge_pairs, noise):
    scenarios = decode_platform_spins(
        _MODEL, np.repeat(samples, 10, axis=0), item["condition"], _SCALER,
        observation_noise=True, observation_noise_values=noise,
    )
    if scenarios.shape != (N_PREDICTIONS, 24) or not np.isfinite(scenarios).all():
        raise ValueError("Decoded %s scenarios must be finite (100,24)" % method)
    metrics = scenario_metrics(scenarios, item["target"])
    row = {"instance_id": item["instance_id"], "date": item["date"],
           "index": item["index"], "export_index": item["export_index"],
           "gain": item["gain"], "method": method,
           "n_latent_samples": 10, "n_prediction_curves": 100,
           "matrix_sha256": item["matrix_sha256"],
           "sa_reference_sha256": item["sample_sha256"],
           "paired_noise_sha256": hashlib.sha256(noise.numpy().tobytes(order="C")).hexdigest()}
    row.update({"Raw_" + k: float(v) for k, v in metrics.items()})
    row.update(_observables(samples, item["h"], item["J"], edge_pairs))
    return row


def _paired_date_table(rows):
    lookup = {(r["instance_id"], r["gain"], r["method"]): r for r in rows}
    dates = sorted({r["date"] for r in rows})
    metrics = ["CRPS", "Energy", "Variogram", "PICP90", "interval_width90"]
    deltas = []
    for date in dates:
        day_items = [r for r in rows if r["date"] == date]
        gains = sorted({r["gain"] for r in day_items})
        row = {"date": date, "n_gains": len(gains)}
        for metric in metrics:
            for left, right in (("Gibbs", "SA"), ("Gibbs", "Random"), ("SA", "Random")):
                values = []
                for gain in gains:
                    instance = next(r["instance_id"] for r in day_items if r["gain"] == gain)
                    values.append(lookup[(instance, gain, left)]["Raw_" + metric]
                                  - lookup[(instance, gain, right)]["Raw_" + metric])
                row["%s_minus_%s_%s" % (left, right, metric)] = float(np.mean(values))
        deltas.append(row)
    return deltas


def _date_bootstrap(deltas):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {}
    for field in deltas[0]:
        if "_minus_" not in field:
            continue
        values = np.asarray([row[field] for row in deltas], dtype=np.float64)
        sample = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
        means = values[sample].mean(axis=1)
        output[field] = {"mean": float(values.mean()),
                         "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
                         "n_unique_dates": len(values)}
    return output


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _self_test_output(path):
    path.write_text(json.dumps(exact_self_test(), indent=2), encoding="utf-8")


def run(output_dir=DEFAULT_OUTPUT):
    global _MODEL, _SCALER
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite analysis output: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _self_test_output(output_dir / "sampler_self_test.json")
    protocol, prep, reference, _MODEL, _SCALER, conditions, y_vs, dates = _load_frozen()
    items = _collect_conditions(reference, prep, protocol, conditions, y_vs, dates)
    gibbs_results = _sample_gibbs_batch(items)
    rows, diagnostics, all_inputs = [], [], {}
    for item, gibbs_result in zip(items, gibbs_results):
        noise_seed = int(protocol["gain_selection"]["prediction_evaluation"]["noise_seed"]) + item["export_index"]
        noise = _standard_noise(noise_seed)
        if noise.shape != (N_PREDICTIONS, 24):
            raise ValueError("Frozen paired noise has invalid shape")
        gibbs, diag, edge_pairs, chain_meta = gibbs_result
        random_seed = RANDOM_SEED_BASE + item["export_index"] * 1000 + item["gain"]
        random_samples = np.random.default_rng(random_seed).choice((-1, 1), size=(10, N_LOGICAL)).astype(np.int8)
        method_rows = [
            _decode_and_score(item, item["sa"], "SA", edge_pairs, noise),
            _decode_and_score(item, gibbs, "Gibbs", edge_pairs, noise),
            _decode_and_score(item, random_samples, "Random", edge_pairs, noise),
        ]
        noise_hashes = {row["paired_noise_sha256"] for row in method_rows}
        if len(noise_hashes) != 1:
            raise ValueError("Methods did not receive identical ordered decoder noise")
        for row in method_rows:
            if row["n_prediction_curves"] != 100 or row["n_latent_samples"] != 10:
                raise AssertionError("Prediction/read count changed")
            rows.append(row)
        diagnostics.append({
            "instance_id": item["instance_id"], "date": item["date"],
            "gain": item["gain"], "matrix_sha256": item["matrix_sha256"],
            "energy_convention": ENERGY_CONVENTION,
            "diagnostic": diag, "gibbs_sampling": chain_meta,
            "representative_edges": [list(x) for x in edge_pairs],
            "sa_sampler_config": item["sampler"],
            "gibbs_logical_energy_mean": float(np.mean([logical_energy(item["h"], item["J"], s) for s in gibbs])),
            "gibbs_vs_sa_target_energy_max_abs_quantization_difference": float(max(
                abs(sdk_energy(item["matrix"], hw) / item["gain"]
                    - logical_energy(item["h"], item["J"], logical))
                for hw, logical in zip(chain_meta["selected_hardware_states"], gibbs)
            )),
        })
        for path in (item["source_manifest_file"], item["source_file"],
                     item["matrix_file"], item["sample_file"]):
            all_inputs[str(path.relative_to(ROOT))] = _sha(path)
    if len(rows) != 90 or len(diagnostics) != 30:
        raise AssertionError("Expected 30 conditions and 90 method rows")
    _write_csv(output_dir / "condition_metrics.csv", rows)
    (output_dir / "condition_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "gibbs_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    diagnostic_rows = [{"instance_id": d["instance_id"], "date": d["date"], "gain": d["gain"],
                        "matrix_sha256": d["matrix_sha256"], **{k: v for k, v in d["diagnostic"].items() if k != "observables"}}
                       for d in diagnostics]
    _write_csv(output_dir / "gibbs_diagnostics.csv", diagnostic_rows)
    date_rows = _paired_date_table(rows)
    _write_csv(output_dir / "paired_date_differences.csv", date_rows)
    bootstrap = _date_bootstrap(date_rows)
    (output_dir / "paired_date_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    prior_j0 = _json(PRIOR_ANALYSIS / "j0_comparability_audit.json")
    (output_dir / "j0_comparability_audit.json").write_text(json.dumps(prior_j0, indent=2, ensure_ascii=False), encoding="utf-8")
    prior_readiness = _json(PRIOR_ANALYSIS / "gibbs_hardware_readiness.json")
    probe = prior_readiness["gain_150_target"]
    probe["task_name"] = build_task_name(
        probe["instance_id"], probe["matrix_sha256"],
        protocol_id=_json(PROTOCOL)["protocol_version"],
        stage="vs-calibration", gain=150,
    )
    probe["platform_task_name"] = platform_task_name(probe["task_name"])
    probe["protocol_version"] = _json(PROTOCOL)["protocol_version"]
    (output_dir / "probe_readiness.json").write_text(json.dumps(prior_readiness, indent=2, ensure_ascii=False), encoding="utf-8")

    pass_count = sum(d["diagnostic"]["status"] == "PASS" for d in diagnostics)
    method_summary = {}
    for method in ("Gibbs", "SA", "Random"):
        method_rows = [r for r in rows if r["method"] == method]
        method_summary[method] = {
            metric: float(np.mean([r["Raw_" + metric] for r in method_rows]))
            for metric in ("CRPS", "Energy", "Variogram", "PICP90", "interval_width90")
        }
    # The full output manifest captures every frozen condition file plus primary sources.
    for path in (PROTOCOL, PREPARATION / "preparation_manifest.json",
                 PREPARATION / "quality_reference_manifest.json",
                 PREPARATION / "frozen_model_audit.json",
                 PREPARATION / "vs_selection_manifest.json", DATASET,
                 _path(protocol["model"]["checkpoint"]),
                 _path(protocol["model"]["model_config"]),
                 PRIOR_ANALYSIS / "j0_comparability_audit.json",
                 PRIOR_ANALYSIS / "gibbs_hardware_readiness.json"):
        all_inputs[str(path.relative_to(ROOT))] = _sha(path)
    source_copy = output_dir / "opsd_gibbs_random_vs.py"
    source_path = Path(__file__).resolve()
    source_copy.write_bytes(source_path.read_bytes())
    all_inputs[str(source_path.relative_to(ROOT))] = _sha(source_path)
    all_inputs[str(source_copy.relative_to(ROOT))] = _sha(source_copy)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "status": "COMPLETED_VS_ONLY",
        "scope": "10 frozen VS dates x 3 gains; saved SA + local Gibbs + independent random logical spins",
        "protocol_sha256": _sha(PROTOCOL), "preparation_manifest_sha256": _sha(PREPARATION / "preparation_manifest.json"),
        "quality_reference_manifest_sha256": _sha(PREPARATION / "quality_reference_manifest.json"),
        "inputs_sha256": dict(sorted(all_inputs.items())),
        "analysis_code_sha256": _sha(source_path), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        "git_commit": git_commit,
        "sampling": {"energy_convention": ENERGY_CONVENTION,
                     "target": "H_sdk(K,u)/gain; beta=1/gain for integer K",
                     "chains": CHAINS, "warmup_sweeps": WARMUP_SWEEPS,
                     "retained_draws_per_chain": RETAINED_PER_CHAIN, "thinning_sweeps": THIN_SWEEPS,
                     "gibbs_seed_base": GIBBS_SEED_BASE, "random_seed_base": RANDOM_SEED_BASE,
                     "rhat_max": RHAT_MAX, "ess_min": ESS_MIN, "flip_rate_range": list(FLIP_RATE_RANGE)},
        "decoder_noise": {"rule": "base_plus_export_index", "base": 2026,
                           "same_100x24_noise_for_all_methods": True,
                           "postprocessing": "raw_clip_0_1"},
        "gibbs_diagnostics": {"passed_conditions": pass_count, "total_conditions": 30,
                              "failed_condition_ids": [d["instance_id"] for d in diagnostics if d["diagnostic"]["status"] != "PASS"]},
        "prediction_summary": method_summary,
        "date_bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS,
                           "unit": "10 unique dates after averaging the three gains within date"},
        "j0_audit_source": {"path": str(PRIOR_ANALYSIS.relative_to(ROOT) / "j0_comparability_audit.json"),
                            "sha256": _sha(PRIOR_ANALYSIS / "j0_comparability_audit.json")},
        "test_arrays_accessed": False, "test_metrics_computed": False,
        "training_invoked": False, "physical_sampling_or_sdk_call": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(output_dir, manifest, diagnostics, bootstrap)
    plot_path = output_dir / "paired_date_crps_differences.png"
    x = np.arange(len(date_rows))
    for pair, color in (("Gibbs_minus_SA_CRPS", "tab:blue"), ("Gibbs_minus_Random_CRPS", "tab:orange"),
                        ("SA_minus_Random_CRPS", "tab:green")):
        # Names in the CSV are generated from pairwise labels below.
        field = pair
        plt.plot(x, [row[field] for row in date_rows], marker="o", label=pair.replace("_CRPS", ""), color=color)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, [r["date"][5:] for r in date_rows], rotation=45)
    plt.ylabel("Paired Raw CRPS difference")
    plt.legend(); plt.tight_layout(); plt.savefig(plot_path, dpi=160); plt.close()
    # Hash all user-facing outputs after creation.
    manifest["output_sha256"] = {p.name: _sha(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "run_manifest.json"}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _write_report(out, manifest, diagnostics, bootstrap):
    methods = manifest["prediction_summary"]
    pass_count = manifest["gibbs_diagnostics"]["passed_conditions"]
    lines = [
        "# OPSD VS：Gibbs、SA 与独立随机状态对照",
        "",
        "## 范围与数据身份",
        "",
        "只使用冻结 10-read 包的 30 个 VS 条件、冻结 seed-0 checkpoint、LS scaler、VS 条件和 VS 观测。没有读取 TEST 标签、计算 TEST 指标、训练模型、调用 SDK 或提交平台任务。SA 样本来自原始参考文件；Gibbs 与随机状态均为本地生成，不能称作真机样本。",
        "",
        "输入 SHA-256 与源码、环境、参数见 `run_manifest.json`；条件级实例、日期、gain、矩阵哈希和每种方法指标见 `condition_metrics.csv`。",
        "",
        "## Gibbs 目标与预先固定的链设置",
        "",
        "采样目标为 `H_sdk(K,u)/gain`，其中 `H_sdk=-uᵀKu`，等价于整数 K 上 `beta=1/gain`。第 49 个辅助自旋参与 49 维链状态；解码前按最后一位辅助自旋归一化成 48 维逻辑态。小型全状态枚举、条件概率、能量符号和辅助自旋等价性自检结果见 `sampler_self_test.json`。",
        "",
        "每条件 4 条独立链，预热 2000 sweep，每链保留 1000 个诊断样本、间隔 5 sweep；初值包括全正、全负和独立随机状态。每个条件用于预测的 10 条 Gibbs 状态从四链保留样本等距抽取。诊断阈值在采样前固定为所有观测 R-hat≤1.05、ESS≥400 且链翻转率在 0.01–0.99。",
        "",
        f"通过混合诊断：**{pass_count}/30** 条件。未通过的条件 ID 见 `run_manifest.json`；逐链能量、逻辑磁化、按整数 K 逻辑块绝对系数预先选出的五条代表边相关、R-hat、ESS、翻转率和 Gibbs 状态见 `gibbs_diagnostics.json`。未通过的条件不支持把相应预测分数解释为已混合 Gibbs 分布的比较。",
        "",
        "随机基线是每个逻辑自旋独立、等概率取 ±1；不以该条件的 Ising 能量生成。SA 和 Gibbs 的目标矩阵相同。所有方法每条件 10 个潜变量状态、每状态 10 组噪声生成 100 条曲线；同一日期的各 gain 以及各方法复用同一套 `100×24` 噪声和原始行顺序。",
        "",
        "## VS 预测结果",
        "",
        "下表是 30 个日期-gain 条件等权平均的描述统计；Gibbs 汇总需结合对应条件的诊断状态阅读。3 个 gain 重复使用同 10 个日期，配对差值先在日期内平均 gains，再对 10 个日期作探索性 bootstrap。",
        "",
        "| 方法 | Raw CRPS | Energy | Variogram | PICP90 | 90%区间宽度 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("Gibbs", "SA", "Random"):
        row = methods[method]
        lines.append("| %s | %.6f | %.6f | %.6f | %.4f | %.6f |" % (
            method, row["CRPS"], row["Energy"], row["Variogram"], row["PICP90"], row["interval_width90"]))
    lines.extend([
        "",
        "配对日期差值及 95% 区间见 `paired_date_differences.csv` 与 `paired_date_bootstrap.json`；图见 `paired_date_crps_differences.png`。差值统一为左侧方法减右侧方法。日期只有 10 个，区间仅作探索性描述。",
        "",
        "## 潜变量状态与 J=0",
        "",
        "每条件的不同状态数、最高频状态比例、基于原始 h/J 的逻辑能量、48 个自旋均值和代表边相关均列在 `condition_metrics.csv`。随机态的能量只作事后诊断，随机生成过程本身不使用 Ising 能量。",
        "",
        "完整 FA/J=0 checkpoint 和已有 VS 场景审计复用自 `opsd_sa_state_diversity_vs_v4/j0_comparability_audit.json`，本目录另存副本。J=0 是耦合消融；本节 Gibbs/SA/随机状态是在固定模型下比较采样方式，两者不是同一个估计对象。历史完整模型 JSON 声明 99 条图边但实际 checkpoint mask 有 172 条，详情以复制的 J=0 审计为准。",
        "",
        "## 真机探针准备状态与限制",
        "",
        "gain=150、10-read 的实例、协议版本、完整任务名、平台短任务名和矩阵哈希见 `probe_readiness.json`；本轮真实离线 dry-run 输出见 `probe_dry_run.txt`。完全替换 SDK 的模拟入口测试在 `tests/test_kaiwu_10read_inference.py`，本轮只读核对其断言，未运行会设置真实提交开关的测试。没有真实响应/回执/intent；旧 v9 100-read 回执不可替代。Kaiwu 授权、设备路由、Credits 换算和平台任务身份语义仍需外部确认。因此不能将本地一致性称为真机可提交条件已完整满足。",
        "",
        "结果仅支持对冻结 VS 条件的探索性比较。若 Gibbs 诊断未通过，不报告 Gibbs 与其他方法的预测优劣结论。即使通过，也不能由 VS 推断 TEST 表现、真机优势或精确 Boltzmann 抽样。",
        "",
        "## 复现",
        "",
        f"`D:\\anaconda\\envs\\wsy\\python.exe -m GEFcom2014.forecast_quality.opsd_gibbs_random_vs --output-dir {out.relative_to(ROOT).as_posix()}`",
        "",
        "脚本只读取固定 LS/VS 数组和已存档 SA 样本；输出目录非空时拒绝覆盖。",
    ])
    (out / "REPORT_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(exact_self_test(), ensure_ascii=False))
        return
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
