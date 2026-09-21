import numpy as np

from GEFcom2014.forecast_value.normalized_decision_value import (
    evaluate_normalized_decision_value,
    run_real_npz,
    run_synthetic_smoke,
)


def test_normalized_decision_value_has_aligned_shapes_and_cost_direction():
    actual = np.array([[0.8, 0.8], [0.7, 0.7]])
    scenarios = np.array([
        [[0.75, 0.75], [0.85, 0.85], [0.8, 0.8]],
        [[0.65, 0.65], [0.75, 0.75], [0.7, 0.7]],
    ])
    point = np.full_like(actual, 0.2)
    result = evaluate_normalized_decision_value(scenarios, actual, point,
                                                under_cost=2.0, over_cost=1.0)
    assert result["scenario_decision"].shape == actual.shape
    assert result["realized_cost"].shape == (2,)
    assert np.all(result["cost_improvement_vs_point"] > 0)
    assert result["real_market_prices_used"] is False


def test_decision_value_smoke_is_explicitly_synthetic(tmp_path):
    path, payload = run_synthetic_smoke(tmp_path)
    assert path.exists()
    assert payload["smoke_only"] is True
    assert payload["synthetic_data"] is True
    assert payload["eligible_for_paper"] is False
    assert payload["real_market_prices_used"] is False


def test_real_decision_cli_path_writes_date_block_summaries(tmp_path):
    scenarios = np.full((3, 4, 2), 0.6)
    actual = np.full((3, 2), 0.5)
    point = np.full((3, 2), 0.4)
    input_path = tmp_path / "real.npz"
    np.savez(input_path, scenarios=scenarios, actual=actual, point_forecast=point,
             dates=np.array(["2020-01-01", "2020-01-02", "2020-01-03"]))
    summary = run_real_npz(input_path, tmp_path / "out", "pilot", "abc")
    assert summary["eligible_for_paper"] is True
    assert len(summary["date_block_bootstrap"]) == 4
