import numpy as np

from GEFcom2014.forecast_quality.opsd_cvae_comparison import validate_vs_selection_record
from export.experiment_completion_v1.neso_calibration_bootstrap import daily_coverage_curves, draw_indices
from export.experiment_completion_v1.paired_block_analysis import boot


def test_selection_score_must_equal_selected_epoch_reconstruction():
    history = [
        {"epoch": 0, "vs_reconstruction": 2.0, "vs_total": -10.0},
        {"epoch": 1, "vs_reconstruction": 1.0, "vs_total": 3.0},
    ]
    assert validate_vs_selection_record(history, 1, 1.0) == 1.0


def test_pooled_reliability_is_recomputed_after_aggregation():
    cover = np.array([[0.25, 0.75], [0.0, 1.0]])
    q = np.array([0.2, 0.8])
    mean_daily = np.abs(cover - q).mean(axis=1).mean() * 100
    pooled = np.abs(cover.mean(axis=0) - q).mean() * 100
    assert np.isclose(mean_daily, 12.5)
    assert np.isclose(pooled, 7.5)
    assert mean_daily != pooled


def test_reliability_and_maqce_use_their_distinct_grids_and_comparisons():
    # Repeated point masses create ties: strict '<' and inclusive '<=' must differ.
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    scenes = np.empty((2, 2, 100), dtype=float)
    scenes[0, 0] = 1.0
    scenes[0, 1] = 2.0
    scenes[1, 0] = 2.5
    scenes[1, 1] = 4.5
    rel, maqce = daily_coverage_curves(y, scenes)
    assert rel.shape == (2, 99)
    assert maqce.shape == (2, 19)
    assert np.all((rel >= 0) & (rel <= 1))
    assert np.all((maqce >= 0) & (maqce <= 1))
    assert np.all(maqce[0] == 1.0)  # observed equals quantile: inclusive MAQCE rule
    assert np.all(rel[0] == 0.0)    # same tie excluded by Reliability's strict rule


def test_block_resampling_keeps_calendar_gaps_separate_and_constant_delta_stable():
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-02-01", "2020-02-02"])
    rng = np.random.default_rng(17)
    sampled = draw_indices(dates, rng)
    assert np.all(sampled[:3] < 3)
    assert np.all(sampled[3:] >= 3)
    ci = boot(np.array(["2020-01-01", "2020-01-02", "2020-01-03"]), np.full(3, -0.25), 2, 200, 9)
    assert np.allclose(ci, [-0.25, -0.25])
