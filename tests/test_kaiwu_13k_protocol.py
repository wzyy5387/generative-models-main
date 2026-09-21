import json

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import (
    aggregate_test_metrics,
    audit_budget,
    compare_read_convergence,
    load_protocol,
    select_vs_gain,
    scenario_metrics,
    validate_test_manifest,
    validate_vs_manifests,
)


def _vs_item(gain, index, zone):
    return {
        "id": "vs_%03d" % index,
        "split": "VS",
        "selection": "stratified",
        "selection_seed": 2026,
        "hardware_gain": gain,
        "logical_n_bits": 48,
        "hardware_n_bits": 49,
        "zone": zone,
    }


def _test_item(index, zone):
    return {
        "id": "test_%03d" % index,
        "split": "TEST",
        "selection": "stratified",
        "selection_seed": 2026,
        "hardware_gain": None,
        "hardware_n_bits": None,
        "zone": zone,
    }


def test_protocol_budget_and_manifest_identity_are_frozen(tmp_path):
    protocol = load_protocol()
    vs_paths = []
    for gain in (125, 150, 200):
        path = tmp_path / ("vs_%d.json" % gain)
        path.write_text(json.dumps([_vs_item(gain, index, index + 1)
                                    for index in range(10)]), encoding="utf-8")
        vs_paths.append(path)
    test_path = tmp_path / "test.json"
    test_path.write_text(json.dumps([_test_item(index, index // 10 + 1)
                                     for index in range(100)]), encoding="utf-8")
    assert audit_budget(vs_paths, test_path, protocol)["planned_samples"] == 13000
    validate_vs_manifests(vs_paths, protocol)
    validate_test_manifest(test_path, protocol)


def test_protocol_rejects_preselected_test_gain(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "test.json"
    rows = [_test_item(index, index // 10 + 1) for index in range(100)]
    rows[0]["hardware_gain"] = 125
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="must not preselect"):
        validate_test_manifest(path, protocol)


def test_real_vs_selection_requires_verified_platform_metadata(tmp_path):
    with pytest.raises(ValueError, match="verified physical"):
        select_vs_gain([{
            "gain": 125,
            "hardware_claim": False,
            "physical_platform_used": False,
            "energy_wasserstein": 0.1,
            "effective_beta_error": 0.1,
            "edge_moment_mae": 0.1,
            "magnetization_mae": 0.1,
            "unique_state_fraction": 0.5,
        }], tmp_path / "frozen.json")

    record = {
        "gain": 150,
        "hardware_claim": True,
        "physical_platform_used": True,
        "energy_wasserstein": 0.1,
        "effective_beta_error": 0.01,
        "edge_moment_mae": 0.02,
        "magnetization_mae": 0.02,
        "unique_state_fraction": 0.9,
    }
    selected = select_vs_gain([record], tmp_path / "frozen.json")
    assert selected["selected_gain"] == 150.0
    assert selected["test_peeking"] is False


def test_read_convergence_and_test_metrics_keep_reads_within_instance():
    h = np.array([0.2, -0.1])
    j = np.array([[0.0, 0.1], [0.1, 0.0]])
    reference = np.tile(np.array([[1, -1]], dtype=np.int8), (100, 1))
    candidate = reference.copy()
    convergence = compare_read_convergence(h, j, reference, candidate)
    assert set(convergence) == {"reads_50", "reads_100"}
    assert convergence["reads_100"]["energy_wasserstein"] == 0.0

    samples = np.tile(np.array([[0.2, 0.4, 0.6]], dtype=float), (100, 1))
    metrics = scenario_metrics(samples, np.array([0.2, 0.4, 0.6]))
    rows = []
    for day in range(3):
        rows.append({"date": "2020-01-%02d" % (day + 1), "model": "hardware", **metrics})
        rows.append({"date": "2020-01-%02d" % (day + 1), "model": "float_logical_sa", **metrics})
    result = aggregate_test_metrics(rows, repetitions=100)
    assert result["reads_are_independent_test_units"] is False
    assert result["paired_comparisons"][0]["n_instance_dates"] == 3
