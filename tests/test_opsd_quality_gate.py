import pytest

from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import (
    evaluate_gain_quality,
    select_vs_gain,
)


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


def test_quality_thresholds_without_approval_lock():
    with pytest.raises(ValueError, match="not approved"):
        evaluate_gain_quality([_metrics(125)], _policy(approved=False))


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
