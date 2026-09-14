import json

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.import_hardware_responses import import_responses
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import (
    KaiwuClient,
    binary_to_spin,
    hardware_matrix_from_ising,
    matrix_sha256,
    normalize_hardware_spins,
    quantize_hardware_matrix,
    spin_to_binary,
)
from GEFcom2014.models.QBM_VAE.export_ising_instances import build_platform_payload
from GEFcom2014.models.QBM_VAE.prepare_bosonic_submission import prepare_submission


def test_auxiliary_matrix_matches_original_energy_by_enumeration():
    h = np.array([0.7, -0.2, 0.4])
    j = np.array([[0.0, 0.3, -0.1], [0.3, 0.0, 0.25], [-0.1, 0.25, 0.0]])
    matrix = hardware_matrix_from_ising(h, j, logical_n_bits=3)
    states = np.asarray(
        [[-1 if (index >> bit) & 1 else 1 for bit in range(3)] for index in range(8)]
    )
    original = -states @ h - 0.5 * np.einsum("bi,ij,bj->b", states, j, states)
    augmented = np.concatenate((states, np.ones((states.shape[0], 1))), axis=1)
    np.testing.assert_allclose(np.einsum("bi,ij,bj->b", augmented, matrix, augmented), original)


def test_one_spin_sign_probe_is_exact():
    h = np.array([1.0])
    j = np.zeros((1, 1))
    matrix = hardware_matrix_from_ising(h, j, logical_n_bits=1)
    augmented = np.array([[1.0, 1.0], [-1.0, 1.0]])
    energy = np.einsum("bi,ij,bj->b", augmented, matrix, augmented)
    np.testing.assert_allclose(energy, np.array([-1.0, 1.0]))


def test_global_gain_quantization_is_audited_and_reproducible():
    h = np.linspace(-0.4, 0.4, 48)
    j = np.zeros((48, 48))
    j[np.arange(47), np.arange(1, 48)] = 0.1
    j += j.T
    first = quantize_hardware_matrix(h, j, hardware_gain=10.0)
    second = quantize_hardware_matrix(h, j, hardware_gain=10.0)
    np.testing.assert_array_equal(first["matrix"], second["matrix"])
    assert first["audit"]["hardware_n_bits"] == 49
    assert first["audit"]["matrix_sha256"] == matrix_sha256(first["matrix"])
    assert first["audit"]["clipping"] is False
    assert first["audit"]["precision_reducer"] is False
    assert first["audit"]["variable_splitting"] is False


def test_quantization_rejects_overflow_instead_of_clipping():
    h = np.ones(48)
    with pytest.raises(OverflowError, match="reduce the global gain"):
        quantize_hardware_matrix(h, np.zeros((48, 48)), hardware_gain=1000.0)


def test_kaiwu_client_uses_sampling_mode_without_importing_the_sdk():
    calls = {}

    class FakeTaskMode:
        SAMPLING = "sampling"

    class FakeOptimizer:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def solve(self, matrix):
            return np.ones((10, matrix.shape[0]), dtype=np.int8)

    client = KaiwuClient.__new__(KaiwuClient)
    client.project_no = "runtime-only-project"
    client.wait = True
    client.interval = 1
    client._optimizer_cls = FakeOptimizer
    client._task_mode = FakeTaskMode
    result = client.sample_hardware_matrix(
        np.zeros((49, 49), dtype=np.int8),
        num_reads=10,
        task_name="unique-test-task",
    )
    assert result["samples"].shape == (10, 49)
    assert calls["task_mode"] == "sampling"
    assert calls["sample_number"] == 10
    assert calls["task_name"] == "unique-test-task"


def test_gauge_normalization_and_binary_mapping_are_exact():
    logical = np.array([[1, -1, 1], [-1, -1, 1]], dtype=np.int8)
    raw = np.concatenate((logical, np.ones((2, 1), dtype=np.int8)), axis=1)
    flipped = -raw
    np.testing.assert_array_equal(
        normalize_hardware_spins(raw, logical_n_bits=3),
        normalize_hardware_spins(flipped, logical_n_bits=3),
    )
    np.testing.assert_array_equal(binary_to_spin(spin_to_binary(logical)), logical)
    with pytest.raises(ValueError):
        normalize_hardware_spins(np.ones((2, 3), dtype=np.int8), logical_n_bits=3)


def test_hardware_submission_and_import_keep_49_raw_and_48_logical(tmp_path):
    h = np.full(48, 0.2)
    j = np.zeros((48, 48))
    quantized = quantize_hardware_matrix(h, j, hardware_gain=20.0)
    matrix_path = tmp_path / "matrix.npz"
    np.savez_compressed(matrix_path, hardware_matrix=quantized["matrix"])
    payload = build_platform_payload(
        "example_000",
        h,
        j,
        hardware_problem=quantized,
    )
    payload["hardware_matrix_file"] = str(matrix_path)
    payload["task_name"] = "fa_bm_vae_example_task"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            [
                {
                    "id": "example_000",
                    "split": "VS",
                    "method_name": "FA-BM-VAE",
                    "platform_payload": str(payload_path),
                    "path": str(matrix_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    submission_path, _, _ = prepare_submission(
        source_manifest,
        tmp_path / "submission",
        stage="vs-calibration",
        requested_reads=10,
    )
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    record = submission["instances"][0]
    assert record["n_bits"] == 49
    assert record["n_edges"] == 48
    assert record["hardware_n_bits"] == 49

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_samples = np.tile(np.ones(49, dtype=np.int8), (10, 1))
    raw_samples[0, :48] *= -1
    np.savez_compressed(
        raw_dir / "example_000.npz",
        samples=raw_samples,
        instance_id="example_000",
        task_id="kaiwu-task-1",
        matrix_sha256=np.asarray(payload["hardware_matrix_sha256"]),
        bit_order="index_ascending",
        backend="kaiwu",
    )
    _, audit = import_responses(
        submission_path,
        raw_dir,
        tmp_path / "canonical",
    )
    with np.load(tmp_path / "canonical" / "example_000.npz") as response:
        assert response["hardware_samples"].shape == (10, 49)
        assert response["samples"].shape == (10, 48)
        np.testing.assert_array_equal(response["samples"][0], -np.ones(48, dtype=np.int8))
    assert audit["records"][0]["logical_n_bits"] == 48
