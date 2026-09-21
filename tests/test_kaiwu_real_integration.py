# -*- coding: utf-8 -*-

"""Opt-in real Kaiwu/SPQC integration tests for the FA-BM-VAE adapter.

These tests are skipped unless FA_BM_VAE_RUN_REAL_KAIWU=1. They are never
part of the default regression suite because they submit paid/platform tasks.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.calibrate_hardware_temperature import (
    evaluate_gain_candidate,
)
from GEFcom2014.models.QBM_VAE.compare_hardware_responses import (
    quantile_wasserstein,
)
from GEFcom2014.models.QBM_VAE.import_hardware_responses import import_responses
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import (
    HARDWARE_N_BITS,
    normalize_hardware_spins,
    quantize_hardware_matrix,
)
from GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios import (
    reconstruct_scenarios,
)
from GEFcom2014.models.QBM_VAE.submit_kaiwu_sampling import submit_one


pytestmark = pytest.mark.skipif(
    os.environ.get("FA_BM_VAE_RUN_REAL_KAIWU") != "1",
    reason="Set FA_BM_VAE_RUN_REAL_KAIWU=1 to submit real Kaiwu tasks",
)


def _project_no():
    value = os.environ.get("KAIWU_PROJECT_NO")
    if not value:
        pytest.skip("KAIWU_PROJECT_NO is not configured")
    return value


def _submit_matrix(tmp_path, matrix, instance_id, gain=None):
    matrix_path = tmp_path / (instance_id + "_matrix.npz")
    arrays = {"hardware_matrix": matrix}
    if gain is not None:
        arrays["hardware_gain"] = np.asarray(gain)
    np.savez_compressed(matrix_path, **arrays)
    return submit_one(
        matrix_path,
        instance_id,
        tmp_path / "raw",
        int(os.environ.get("FA_BM_VAE_REAL_READS", "100")),
        project_no=_project_no(),
        checkpoint_dir=tmp_path / "checkpoints",
    )


def _exact_logical_energies(h, j):
    n_bits = h.size
    states = np.asarray(
        [[1 if (index >> bit) & 1 else -1 for bit in range(n_bits)]
         for index in range(1 << n_bits)],
        dtype=np.int8,
    )
    energies = -states @ h - 0.5 * np.einsum("bi,ij,bj->b", states, j, states)
    return states, energies


def test_real_one_bit_positive_field_probe(tmp_path):
    """A positive one-bit field should produce more +1 than -1 samples."""
    quantized = quantize_hardware_matrix(
        np.array([1.0]), np.zeros((1, 1)), hardware_gain=32.0, logical_n_bits=1
    )
    _, _, audit = _submit_matrix(
        tmp_path, quantized["matrix"], "real_one_bit_probe", gain=32.0
    )
    with np.load(audit["response_path"], allow_pickle=False) as response:
        raw = np.asarray(response["samples"], dtype=np.int8)
    logical = normalize_hardware_spins(raw, logical_n_bits=1)
    assert logical.shape[1] == 1
    assert float(logical[:, 0].mean()) > 0.0


def test_real_small_matrix_matches_exact_energy_distribution(tmp_path):
    """Compare a 6-bit platform sample with its finite exact state law."""
    h = np.array([0.25, -0.15, 0.1, 0.0, 0.12, -0.08])
    j = np.zeros((6, 6))
    j[0, 1] = j[1, 0] = 0.12
    j[2, 3] = j[3, 2] = -0.10
    j[4, 5] = j[5, 4] = 0.08
    quantized = quantize_hardware_matrix(h, j, hardware_gain=64.0, logical_n_bits=6)
    _, _, audit = _submit_matrix(
        tmp_path, quantized["matrix"], "real_small_matrix", gain=64.0
    )
    with np.load(audit["response_path"], allow_pickle=False) as response:
        raw = np.asarray(response["samples"], dtype=np.int8)
    logical = normalize_hardware_spins(raw, logical_n_bits=6)
    assert raw.shape[1] == 7
    assert logical.shape[1] == 6

    _, exact_energies = _exact_logical_energies(h, j)
    beta = float(os.environ.get("FA_BM_VAE_REAL_REFERENCE_BETA", "1.0"))
    weights = np.exp(-beta * (exact_energies - exact_energies.min()))
    exact_samples = np.repeat(
        exact_energies,
        np.maximum(1, np.rint(2000 * weights / weights.sum()).astype(int)),
    )
    sample_energies = -logical @ h - 0.5 * np.einsum(
        "bi,ij,bj->b", logical, j, logical
    )
    assert np.isfinite(sample_energies).all()
    assert quantile_wasserstein(sample_energies, exact_samples) < 0.75


def test_real_49_bit_import_calibrate_and_reconstruct_pipeline(tmp_path):
    """Run all post-processing stages from one real 49-spin response."""
    submission_manifest = Path(os.environ.get("FA_BM_VAE_SUBMISSION_MANIFEST", ""))
    source_manifest = Path(os.environ.get("FA_BM_VAE_SOURCE_MANIFEST", ""))
    model_name = os.environ.get("FA_BM_VAE_MODEL_NAME")
    tag = os.environ.get("FA_BM_VAE_TAG", "wind")
    if not submission_manifest.is_file() or not source_manifest.is_file() or not model_name:
        pytest.skip(
            "Set FA_BM_VAE_SUBMISSION_MANIFEST, FA_BM_VAE_SOURCE_MANIFEST, "
            "and FA_BM_VAE_MODEL_NAME for the full pipeline test"
        )
    submission = json.loads(submission_manifest.read_text(encoding="utf-8"))
    if len(submission["instances"]) != 1:
        pytest.skip("Full pipeline test requires a one-instance submission manifest")
    item = submission["instances"][0]
    package_dir = submission_manifest.parent
    matrix_path = package_dir / item["hardware_matrix"]
    raw_path, _, _ = submit_one(
        matrix_path,
        item["instance_id"],
        tmp_path / "raw",
        int(submission["requested_reads_per_instance"]),
        project_no=_project_no(),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    with np.load(raw_path, allow_pickle=False) as raw:
        assert raw["samples"].shape[1] == HARDWARE_N_BITS

    canonical_dir = tmp_path / "canonical"
    _, import_audit = import_responses(
        submission_manifest, tmp_path / "raw", canonical_dir
    )
    assert import_audit["n_imported"] == 1
    with np.load(canonical_dir / (item["instance_id"] + ".npz"), allow_pickle=False) as canonical:
        assert canonical["hardware_samples"].shape[1] == 49
        assert canonical["samples"].shape[1] == 48

    candidate = evaluate_gain_candidate(
        {
            "gain": float(submission["hardware_gain"]),
            "vs_manifest": str(source_manifest),
            "responses_dir": str(canonical_dir),
        },
        target_beta=1.0,
    )
    assert np.isfinite(candidate["beta_eff"])

    model_dir = Path("export") / ("qbm_vae_%s" % tag)
    result = reconstruct_scenarios(
        source_manifest,
        canonical_dir,
        model_dir / (model_name + ".pickle"),
        model_dir / (model_name + ".json"),
        tag,
        output_dir=tmp_path / "scenarios",
        observation_noise=False,
        cpu=True,
    )
    assert result[0].is_file()
    assert result[1].is_file()
