import json

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import (
    aggregate_test_metrics,
    audit_budget,
    compare_read_convergence,
    load_protocol,
    paired_date_inference,
    select_vs_gain,
    scenario_metrics,
    validate_vs_response_records,
    validate_test_manifest,
    validate_vs_manifests,
)
from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import build_task_name


def test_task_name_canonicalizes_integer_gain_types():
    args = ("instance", "0123456789abcdef",)
    common = {"protocol_id": "confirmed-v3", "stage": "vs-calibration"}
    assert build_task_name(*args, gain=125, **common) == build_task_name(*args, gain=125.0, **common)


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


def _approved_quality_policy():
    return {
        "status": "defined",
        "rule_version": "test-v1",
        "thresholds_approved": True,
        "thresholds": {
            "energy_wasserstein": 1.0,
            "effective_beta_error": 1.0,
            "edge_moment_mae": 1.0,
            "magnetization_mae": 1.0,
            "unique_state_fraction_min": 0.0,
        },
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


def test_vs_gain_selection_aggregates_fixed_panel_and_breaks_exact_ties_by_gain(tmp_path):
    records = []
    for gain in (125, 150, 200):
        for instance in range(2):
            response = tmp_path / ("response_%d_%d.npz" % (gain, instance))
            np.savez_compressed(
                response, samples=np.ones((10, 49), dtype=np.int8),
                instance_id="vs_%d" % instance, task_id="tid_%d_%d" % (gain, instance),
                task_name="tn_%d_%d" % (gain, instance), matrix_sha256="mh_%d_%d" % (gain, instance),
                stage="vs", hardware_gain=gain, requested_reads=10, returned_reads=10,
            )
            records.append({
                "gain": gain, "instance_id": "vs_%d" % instance,
                "stage": "vs", "task_id": "tid_%d_%d" % (gain, instance),
                "task_name": "tn_%d_%d" % (gain, instance),
                "matrix_sha256": "mh_%d_%d" % (gain, instance),
                "response_path": str(response),
                "response_file_sha256": __import__("hashlib").sha256(response.read_bytes()).hexdigest(),
                "requested_reads": 10, "returned_reads": 10,
                "sdk_version": "1.0.7", "platform_backend": "kaiwu.cim.CIMOptimizer",
                "submitted_at": "2026-01-01T00:00:00+00:00", "completed_at": "2026-01-01T00:01:00+00:00",
                "hardware_metadata_complete": True, "completed": True,
                "hardware_claim": True, "physical_platform_used": True,
                "sdk_versions": {"importlib_metadata_version": "1.4.1", "kaiwu_module_version": "1.0.7"},
                "energy_wasserstein": 0.1, "effective_beta_error": 0.01,
                "edge_moment_mae": 0.02, "magnetization_mae": 0.02,
                "unique_state_fraction": 0.9,
                "quality_assessment": {"status": "pass", "rule_version": "test-v1"},
            })
    selected = select_vs_gain(
        records, tmp_path / "frozen.json", required_gains=(125, 150, 200),
        expected_instances_per_gain=2, expected_instance_ids=("vs_0", "vs_1"),
        quality_policy=_approved_quality_policy(),
    )
    assert selected["selected_gain"] == 125.0
    assert len(selected["candidate_metrics"]) == 3


def _complete_vs_records(tmp_path):
    records = []
    for gain in (125, 150, 200):
        for instance in range(2):
            instance_id = "vs_%d" % instance
            response = tmp_path / ("valid_%d_%d.npz" % (gain, instance))
            np.savez_compressed(
                response, samples=np.ones((10, 49), dtype=np.int8),
                instance_id=instance_id, task_id="tid_%d_%d" % (gain, instance),
                task_name="tn_%d_%d" % (gain, instance), matrix_sha256="mh_%d_%d" % (gain, instance),
                stage="vs", hardware_gain=gain, requested_reads=10, returned_reads=10,
            )
            records.append({
                "completed": True, "instance_id": instance_id, "stage": "vs",
                "task_id": "tid_%d_%d" % (gain, instance), "task_name": "tn_%d_%d" % (gain, instance),
                "matrix_sha256": "mh_%d_%d" % (gain, instance),
                "response_file_sha256": __import__("hashlib").sha256(response.read_bytes()).hexdigest(),
                "requested_reads": 10, "returned_reads": 10, "response_path": str(response),
                "gain": gain, "sdk_version": "1.0.7", "platform_backend": "kaiwu.cim.CIMOptimizer",
                "submitted_at": "2026-01-01T00:00:00+00:00", "completed_at": "2026-01-01T00:01:00+00:00",
                "hardware_metadata_complete": True, "hardware_claim": True,
                "physical_platform_used": True,
                "sdk_versions": {"importlib_metadata_version": "1.4.1", "kaiwu_module_version": "1.0.7"},
                "energy_wasserstein": 0.1, "effective_beta_error": 0.01,
                "edge_moment_mae": 0.02, "magnetization_mae": 0.02,
                "unique_state_fraction": 0.9,
                "quality_assessment": {"status": "pass", "rule_version": "test-v1"},
            })
    return records


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(dict(rows[-1])),
    lambda rows: rows.__setitem__(-1, {**rows[-1], "instance_id": "vs_0"}),
    lambda rows: rows.__setitem__(0, {**rows[0], "matrix_sha256": "tampered"}),
    lambda rows: rows.__setitem__(0, {**rows[0], "energy_wasserstein": float("nan")}),
    lambda rows: rows.__setitem__(0, {**rows[0], "returned_reads": 9}),
])
def test_vs_freeze_rejects_incomplete_or_untrusted_records(tmp_path, mutation):
    records = _complete_vs_records(tmp_path)
    mutation(records)
    with pytest.raises(ValueError):
        validate_vs_response_records(
            records, required_gains=(125, 150, 200), expected_instances_per_gain=2,
            expected_instance_ids=("vs_0", "vs_1"),
            quality_policy=_approved_quality_policy(),
        )


def test_vs_freeze_rejects_missing_quality_policy(tmp_path):
    with pytest.raises(ValueError, match="quality policy"):
        validate_vs_response_records(
            _complete_vs_records(tmp_path), required_gains=(125, 150, 200),
            expected_instances_per_gain=2, expected_instance_ids=("vs_0", "vs_1"),
            quality_policy=None,
        )


def test_read_convergence_and_test_metrics_keep_reads_within_instance():
    h = np.array([0.2, -0.1])
    j = np.array([[0.0, 0.1], [0.1, 0.0]])
    reference = np.tile(np.array([[1, -1]], dtype=np.int8), (100, 1))
    candidate = reference.copy()
    convergence = compare_read_convergence(h, j, reference, candidate)
    assert set(convergence["comparison"]) == {"reads_50", "reads_100"}
    assert convergence["comparison"]["reads_100"]["energy_wasserstein"] == 0.0
    assert convergence["sample_overlap"] == 50
    assert "not a convergence proof" in convergence["interpretation"]

    samples = np.tile(np.array([[0.2, 0.4, 0.6]], dtype=float), (100, 1))
    metrics = scenario_metrics(samples, np.array([0.2, 0.4, 0.6]))
    rows = []
    dates = ["2020-01-%02d" % day for day in range(1, 4)]
    for day in range(3):
        for model in ("float_logical_sa", "int8_matrix_sa", "spqc_550"):
            rows.append({"date": dates[day], "model": model, **metrics})
    result = aggregate_test_metrics(rows, dates, repetitions=100)
    assert result["reads_are_independent_test_units"] is False
    assert result["primary_contrast"] == "B_vs_C"
    assert result["paired_comparisons"][0]["reference_model"] == "int8_matrix_sa"
    assert result["paired_comparisons"][0]["n_date_blocks"] == 1


def test_test_aggregation_rejects_missing_duplicate_and_unexpected_dates():
    metrics = {key: 0.1 for key in (
        "CRPS", "Energy", "Variogram", "Ramp_CRPS", "coverage", "PICP90",
        "interval_width90", "MAQCE",
    )}
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    rows = [{"date": date, "model": model, **metrics}
            for date in dates for model in ("float_logical_sa", "int8_matrix_sa", "spqc_550")]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_test_metrics(rows[:-1], dates, repetitions=20)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_test_metrics(rows + [rows[0]], dates, repetitions=20)
    with pytest.raises(ValueError, match="Unexpected"):
        aggregate_test_metrics(rows + [{**rows[0], "date": "2020-01-04"}], dates, repetitions=20)


def test_paired_inference_resamples_contiguous_date_blocks_reproducibly():
    differences = np.repeat(np.array([-1.0, 0.5, 1.5]), [7, 7, 1])
    first = paired_date_inference(differences, repetitions=200, seed=19, block_length_days=7)
    second = paired_date_inference(differences, repetitions=200, seed=19, block_length_days=7)
    assert first == second
    assert first["n_instance_dates"] == 15
    assert first["n_date_blocks"] == 3
    assert first["block_length_days"] == 7
    assert "exchangeable non-overlapping" in first["inference_assumption"]


def test_effective_beta_boundary_is_labeled_unresolved_with_range_and_loss():
    from GEFcom2014.models.QBM_VAE.calibration import fit_effective_temperature_pseudolikelihood
    samples = np.tile(np.array([[1, -1]], dtype=np.int8), (8, 1))
    result = fit_effective_temperature_pseudolikelihood(
        np.zeros(2), np.zeros((2, 2)), samples,
        beta_grid=np.array([0.02, 0.5, 1.0]))
    assert result["grid_boundary"] is True
    assert result["beta_eff_status"] == "current_range_unresolved"
    assert result["beta_grid_min"] == 0.02
    assert result["beta_grid_max"] == 1.0
    assert np.isfinite(result["pseudolikelihood_nll"])


def test_response_statistics_preserve_boundary_label_and_fit_loss():
    from GEFcom2014.models.QBM_VAE.compare_hardware_responses import response_statistics
    samples = np.tile(np.array([[1, -1]], dtype=np.int8), (10, 1))
    stats = response_statistics(np.zeros(2), np.zeros((2, 2)), samples)
    assert stats["beta_eff_grid_boundary"] is True
    assert stats["beta_eff_status"] == "current_range_unresolved"
    assert stats["beta_eff_grid_min"] == 0.02
    assert stats["beta_eff_grid_max"] == 5.0
    assert np.isfinite(stats["beta_eff_fit_loss"])


def test_hardware_metric_adapter_matches_classical_daily_definition():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(100, 24))
    target = rng.normal(size=24)
    classical = evaluate_model_daily(samples.T, target.reshape(1, -1), max_s=100)
    hardware = scenario_metrics(samples, target)
    mapping = {
        "CRPS": "CRPS_raw", "Energy": "Energy", "Variogram": "Variogram",
        "Ramp_CRPS": "ramp_CRPS", "coverage": "PICP90",
        "PICP90": "PICP90", "interval_width90": "interval_width90", "MAQCE": "MAQCE",
    }
    for hardware_name, classical_name in mapping.items():
        assert np.isclose(hardware[hardware_name], classical[classical_name][0], atol=1e-12)
