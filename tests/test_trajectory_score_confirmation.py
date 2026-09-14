from GEFcom2014.forecast_quality.confirm_trajectory_score import MAX_RELATIVE_REGRESSION


def test_confirmation_regression_tolerance_is_fixed_at_one_percent():
    assert MAX_RELATIVE_REGRESSION == 0.01
