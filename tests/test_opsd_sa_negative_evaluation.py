import numpy as np

from GEFcom2014.forecast_quality.opsd_sa_negative_evaluation import (
    _daily_metrics,
    scenario_specs,
)


def test_registry_keeps_sa_main_and_matched_ablations_separate():
    specs = scenario_specs()
    assert tuple(specs["FA-BM-VAE"]["seeds"]) == (0, 1, 2, 3, 4)
    assert tuple(specs["J=0 ablation"]["seeds"]) == (0, 1, 2)
    assert tuple(specs["No anchor addition ablation (matched zero-mean init)"]["seeds"]) == (0, 1, 2)
    assert "anchor_sa_neg" in specs["FA-BM-VAE"]["path"](0, "TEST").name
    assert "abl_j0_sa_neg" in specs["J=0 ablation"]["path"](0, "TEST").name
    assert "zero_mean" in specs["No anchor addition ablation (matched zero-mean init)"]["path"](0, "TEST").name


def test_daily_metrics_uses_trajectory_ramp_crps_and_date_level_outputs():
    rng = np.random.default_rng(7)
    targets = rng.uniform(0.0, 1.0, size=(2, 24))
    scenarios = np.clip(targets.reshape(-1, 1) + rng.normal(0.0, 0.1, size=(48, 100)), 0.0, 1.0)
    metrics = _daily_metrics(scenarios, targets)
    assert metrics["ramp_CRPS"].shape == (2,)
    assert metrics["CRPS_raw"].shape == (2,)
    assert metrics["mean_bias"].shape == (2,)
    assert np.isfinite(metrics["ramp_CRPS"]).all()
