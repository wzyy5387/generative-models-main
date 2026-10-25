from pathlib import Path

import numpy as np
import pandas as pd

from GEFcom2014.neso_wind_dataset import _actual_days, _cutoff_utc, _eligibility_rows, _forecast_days
from GEFcom2014.models.probabilistic_baselines import scenarios_to_period_matrix


def _write_forecast(path, date, issue_utc, periods=48):
    local_start = pd.Timestamp(date).tz_localize("Europe/London")
    starts = pd.date_range(local_start, periods=periods, freq="30min")
    pd.DataFrame({
        "Datetime_GMT": starts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S"),
        "Date": [date] * periods,
        "Settlement_period": np.arange(1, periods + 1),
        "Capacity": [18000] * periods,
        "Incentive_forecast": np.arange(periods, dtype=float) + 100,
        "Forecast_Timestamp": [issue_utc] * periods,
    }).to_csv(path, index=False)


def test_forecast_issue_cutoff_uses_fixed_london_clock_and_full_24_hour_days(tmp_path):
    path = tmp_path / "forecast.csv"
    _write_forecast(path, "2024-07-01", "2024-06-30T08:10:00Z")
    days, reasons, _ = _forecast_days(path)
    assert len(days[ pd.Timestamp("2024-07-01").date() ]["values"]) == 24
    assert not reasons

    _write_forecast(path, "2024-07-01", "2024-06-30T08:20:00Z")
    days, reasons, _ = _forecast_days(path)
    assert not days
    assert reasons["forecast_after_information_cutoff"] == 1


def test_cutoff_uses_previous_london_calendar_day_across_both_dst_switches():
    assert _cutoff_utc(pd.Timestamp("2024-03-31").date()).isoformat() == "2024-03-30T09:15:00+00:00"
    assert _cutoff_utc(pd.Timestamp("2024-04-01").date()).isoformat() == "2024-03-31T08:15:00+00:00"
    assert _cutoff_utc(pd.Timestamp("2024-10-27").date()).isoformat() == "2024-10-26T08:15:00+00:00"
    assert _cutoff_utc(pd.Timestamp("2024-10-28").date()).isoformat() == "2024-10-27T09:15:00+00:00"


def test_forecast_and_actual_skip_dst_non_24_hour_dates_and_duplicates(tmp_path):
    forecast_path = tmp_path / "forecast.csv"
    _write_forecast(forecast_path, "2024-03-31", "2024-03-30T08:00:00Z", periods=46)
    days, reasons, _ = _forecast_days(forecast_path)
    assert not days
    assert reasons["forecast_incomplete_settlement_periods"] == 1

    actual_path = tmp_path / "actual.csv"
    pd.DataFrame({
        "Sett_Date": ["2024-03-31"] * 46,
        "Sett_Period": np.arange(1, 47),
        "Scottish Wind Output": np.ones(46),
        "England/Wales Wind Output": np.ones(46) * 2,
        "Total": np.ones(46) * 3,
    }).to_csv(actual_path, index=False)
    actual, reasons, _, _ = _actual_days([actual_path])
    assert not actual
    assert reasons["actual_incomplete_settlement_periods"] == 1


def test_daily_eligibility_keeps_specific_forecast_and_actual_failure_causes():
    days = [pd.Timestamp("2024-07-01").date(), pd.Timestamp("2024-07-02").date()]
    forecast_audit = {days[0]: {"forecast_rows": 48, "forecast_reason": "forecast_after_information_cutoff"},
                      days[1]: {"forecast_rows": 48, "forecast_reason": "eligible"}}
    actual_audit = {days[0]: {"actual_rows": 48, "actual_reason": "eligible"},
                    days[1]: {"actual_rows": 46, "actual_reason": "actual_incomplete_settlement_periods"}}
    rows = _eligibility_rows(forecast_audit, actual_audit, {days[1]: {}}, {days[0]: {}})
    assert rows.reason.tolist() == ["forecast_after_information_cutoff", "actual_incomplete_settlement_periods"]
    assert rows.status.tolist() == ["excluded", "excluded"]


def test_neso_scenario_inverse_scaling_keeps_mw_support_unclipped():
    class IdentityScaler:
        def inverse_transform(self, values):
            return values

    values = np.zeros((1, 2, 24), dtype=np.float32)
    values[0, :, 0] = [1500.0, -10.0]
    result = scenarios_to_period_matrix(values, IdentityScaler(), "neso-wind", np.array([], dtype=int))
    np.testing.assert_array_equal(result[0], [1500.0, -10.0])
