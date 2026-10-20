import pytest
import numpy as np
import json

from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import (
    aggregate_gain_metrics,
    evaluate_gain_quality,
    select_vs_gain,
)
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import matrix_sha256
from GEFcom2014.models.QBM_VAE import opsd_quality_gate as gate


def _policy(approved=True):
    return {
        "status": "defined" if approved else "reference_estimated_pending_manual_approval",
        "rule_version": "test-quality-v1",
        "thresholds_approved": approved,
        "thresholds": {
            "energy_wasserstein": 0.2,
            "effective_beta_error": 0.2,
            "edge_moment_mae": 0.2,
            "magnetization_mae": 0.2,
            "unique_state_fraction_min": 0.8,
        },
    }


def _metrics(gain, energy=0.1, unique=0.9):
    return {
        "gain": float(gain), "n_verified_instances": 10,
        "energy_wasserstein": energy, "effective_beta_error": 0.1,
        "edge_moment_mae": 0.1, "magnetization_mae": 0.1,
        "unique_state_fraction": unique,
    }


def test_quality_is_recomputed_by_gain_and_ignores_forged_status():
    rows = [_metrics(125), _metrics(150, energy=0.5), _metrics(200)]
    rows[0]["quality_assessment"] = {"status": "fail", "rule_version": "wrong"}
    result = evaluate_gain_quality(rows, _policy())
    assert [row["quality_status"] for row in result] == ["pass", "fail", "pass"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, None, True, "invalid"])
def test_invalid_threshold_cannot_authorize_even_an_empty_batch(value):
    policy = _policy()
    policy["thresholds"]["energy_wasserstein"] = value
    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluate_gain_quality([], policy)


def test_missing_threshold_and_invalid_diversity_limit_are_rejected():
    policy = _policy()
    policy["thresholds"].pop("edge_moment_mae")
    with pytest.raises(ValueError, match="edge_moment_mae"):
        evaluate_gain_quality([], policy)
    policy = _policy()
    policy["diversity_gate_approved"] = True
    policy["thresholds"]["unique_state_fraction_min"] = 1.1
    with pytest.raises(ValueError, match="within"):
        evaluate_gain_quality([], policy)


def test_quality_thresholds_without_approval_lock():
    with pytest.raises(ValueError, match="not approved"):
        evaluate_gain_quality([_metrics(125)], _policy(approved=False))


def test_unique_state_fraction_is_diagnostic_without_approved_diversity_gate():
    policy = _policy()
    policy["diversity_gate_approved"] = False
    row = _metrics(125, unique=0.0)
    result = evaluate_gain_quality([row], policy)[0]
    assert result["quality_status"] == "pass"
    assert result["quality_checks"]["unique_state_fraction"]["gate_enabled"] is False


def test_quality_aggregation_recomputes_raw_files_after_metric_tampering(tmp_path):
    rng = np.random.default_rng(11)
    h = rng.normal(size=48) * 0.01
    j = np.zeros((48, 48), dtype=float)
    j[0, 1] = j[1, 0] = 0.02
    problem = tmp_path / "problem.npz"
    reference = tmp_path / "reference.npz"
    response = tmp_path / "response.npz"
    matrix = tmp_path / "matrix.npz"
    np.savez(problem, h=h, J=j)
    reference_samples = rng.choice((-1, 1), size=(100, 48)).astype(np.int8)
    raw = np.concatenate([reference_samples, np.ones((100, 1), dtype=np.int8)], axis=1)
    np.savez(reference, samples=reference_samples)
    matrix_values = np.zeros((49, 49), dtype=np.int8)
    np.savez(matrix, hardware_matrix=matrix_values)
    np.savez(response, samples=raw)
    record = {
        "gain": 125, "response_path": str(response),
        "matrix_sha256": matrix_sha256(matrix_values),
        "hardware_matrix_path": str(matrix),
        "source_problem_path": str(problem),
        "source_problem_sha256": __import__("hashlib").sha256(problem.read_bytes()).hexdigest(),
        "reference_sample_path": str(reference),
        "reference_sample_sha256": __import__("hashlib").sha256(reference.read_bytes()).hexdigest(),
    }
    actual = aggregate_gain_metrics([record], recompute=True)[0]
    assert np.isfinite([actual[key] for key in (
        "energy_wasserstein", "effective_beta_error", "edge_moment_mae",
        "magnetization_mae", "unique_state_fraction")]).all()
    record.update({"energy_wasserstein": 999.0, "effective_beta_error": 999.0,
                   "edge_moment_mae": 999.0, "magnetization_mae": 999.0,
                   "unique_state_fraction": 0.0})
    record["energy_wasserstein"] = -123.0
    changed = aggregate_gain_metrics([record], recompute=True)[0]
    assert changed["energy_wasserstein"] == actual["energy_wasserstein"]


def test_new_policy_offline_chain_writes_fixture_report_without_freezing_gain(tmp_path, monkeypatch):
    (tmp_path / "prep").mkdir()
    (tmp_path / "prep" / "quality_reference_manifest.json").write_text("{}", encoding="utf-8")
    policy = {
        "status": "defined_pending_manual_approval",
        "rule_version": "opsd-sa-vs-quality-v2-diagnostic-raw-crps",
        "mode": "engineering_valid_prediction_selection",
        "policy_approved": False,
        "selection_metric": "raw_vs_crps",
        "tie_break": "lower_gain",
        "diversity_gate_approved": False,
        "thresholds": {key: 0.0 for key in (
            "energy_wasserstein", "effective_beta_error", "edge_moment_mae", "magnetization_mae")},
    }
    protocol = {"protocol_version": "fixture-protocol", "hardware": {"candidate_gains": [125, 150, 200]},
                "vs": {"instances": 1}, "gain_selection": {"quality_gate": policy}}
    tasks = [{"gain": gain, "instance_id": "d0", "task_name": "fixture-%s" % gain}
             for gain in (125, 150, 200)]
    rows = [{**_metrics(gain, energy=10.0), "raw_vs_crps": score,
             "vs_prediction_metrics": [{"metrics": {"CRPS": score}}]}
            for gain, score in ((125, 0.3), (150, 0.2), (200, 0.1))]
    monkeypatch.setattr(gate, "make_offline_response_records", lambda *args: [
        {"gain": gain, "instance_id": "d0", "hardware_claim": False,
         "physical_platform_used": False} for gain in (125, 150, 200)])
    monkeypatch.setattr("GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling._vs_tasks",
                        lambda *args: tasks)
    monkeypatch.setattr("GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling._task_expected",
                        lambda task: {"task_name": task["task_name"]})
    monkeypatch.setattr("GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling._preparation_identity",
                        lambda *args: {})
    monkeypatch.setattr(gate, "validate_vs_response_records", lambda records, *args, **kwargs: records)
    monkeypatch.setattr("GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol.aggregate_gain_metrics",
                        lambda *args, **kwargs: rows)

    result = gate.run_offline_chain(tmp_path / "prep", tmp_path / "out", protocol)
    report_path = tmp_path / "out" / "offline_prediction_policy_validation.json"
    audit = json.loads((tmp_path / "out" / "offline_chain_audit.json").read_text())
    assert result["selected_gain_diagnostic_only"] == 200.0
    assert result["artifact_role"] == "offline_test_fixture"
    assert result["unlock_real_test"] is False and result["physical_platform_used"] is False
    assert audit["new_prediction_policy_validation"] == "passed_offline_fixture_only"
    assert report_path.exists()
    assert not (tmp_path / "out" / "real_gain_selection.json").exists()
    assert protocol["gain_selection"]["quality_gate"]["policy_approved"] is False


def test_select_uses_only_quality_pass_gains_and_locks_when_all_fail(monkeypatch, tmp_path):
    rows = []
    for gain in (125, 150, 200):
        row = {**_metrics(gain), "hardware_claim": True, "physical_platform_used": True}
        rows.extend([dict(row) for _ in range(2)])
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol.validate_vs_response_records",
        lambda records, **kwargs: records,
    )
    selected = select_vs_gain(
        rows, tmp_path / "selected.json", required_gains=(125, 150, 200),
        expected_instances_per_gain=2, quality_policy=_policy(),
    )
    assert selected["selected_gain"] == 125.0
    failed = [{**row, "energy_wasserstein": 1.0} for row in rows]
    with pytest.raises(ValueError, match="All candidate gains failed"):
        select_vs_gain(
            failed, tmp_path / "failed.json", required_gains=(125, 150, 200),
            expected_instances_per_gain=2, quality_policy=_policy(),
        )


def test_select_raw_path_does_not_require_submitted_metric_fields(monkeypatch, tmp_path):
    rows = [{"gain": gain, "hardware_claim": True, "physical_platform_used": True}
            for gain in (125, 150, 200)]
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol.validate_vs_response_records",
        lambda records, **kwargs: records,
    )

    def recomputed(records, recompute=False, **kwargs):
        assert recompute is True
        return [{**_metrics(row["gain"]), "n_verified_instances": 1} for row in records]

    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol.aggregate_gain_metrics",
        recomputed,
    )
    selected = select_vs_gain(
        rows, tmp_path / "raw-only.json", required_gains=(125, 150, 200),
        expected_instances_per_gain=1, quality_policy=_policy(), recompute_from_raw=True,
    )
    assert selected["selected_gain"] == 125.0


def test_formal_select_does_not_fall_back_to_submitted_metrics(monkeypatch, tmp_path):
    rows = [{"gain": gain, "hardware_claim": True, "physical_platform_used": True,
             "energy_wasserstein": 0.1, "effective_beta_error": 0.1,
             "edge_moment_mae": 0.1, "magnetization_mae": 0.1,
             "unique_state_fraction": 0.9}
            for gain in (125, 150, 200)]
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol.validate_vs_response_records",
        lambda records, **kwargs: records,
    )
    with pytest.raises(ValueError, match="raw response metric inputs"):
        select_vs_gain(
            rows, tmp_path / "legacy-formal.json", required_gains=(125, 150, 200),
            expected_instances_per_gain=1,
            expected_task_map={(float(row["gain"]), "fixed"):
                               {**row, "source_problem_path": "frozen.npz"}
                               for row in rows},
            quality_policy=_policy(),
        )
