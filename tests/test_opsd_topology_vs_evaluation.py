import numpy as np

from GEFcom2014.forecast_quality.opsd_topology_vs_evaluation import (
    compute_daily_metrics,
)


def test_compute_daily_metrics_returns_daily_raw_metrics():
    targets = np.zeros((2, 24), dtype=np.float64)
    scenarios = np.zeros((48, 100), dtype=np.float64)
    metrics = compute_daily_metrics(scenarios, targets)

    assert set(metrics) == {
        "crps",
        "energy_score",
        "variogram_score",
        "ramp_crps",
        "picp90",
        "interval_width90",
        "bias",
    }
    for values in metrics.values():
        assert values.shape == (2,)
        assert np.isfinite(values).all()
    np.testing.assert_allclose(metrics["picp90"], 1.0)
    np.testing.assert_allclose(metrics["interval_width90"], 0.0)
