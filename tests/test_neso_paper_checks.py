import numpy as np
import pandas as pd
import pytest
from GEFcom2014.forecast_quality.neso_paper_checks import curves, block_ci_once
from GEFcom2014.forecast_quality.neso_paper_checks import seed_direction_tables
from GEFcom2014.forecast_quality.neso_paper_checks_finalize import (
    forecast_with_timezone, timezone_date_differences,
)


def test_pooled_reliability_and_maqce_constant_forecast_case():
    y=np.zeros((2,24)); s=np.zeros((48,100))
    rel,maq,cov,width=curves(s.reshape(2,24,100),y)
    assert rel == 50.0  # strict y < q: every empirical rate is zero
    assert np.isclose(maq,0.5)  # <= includes ties, all empirical rates are one
    assert cov == [1.,1.,1.,1.] and width == [0.,0.,0.,0.]


def test_block_bootstrap_constant_difference_and_gap_segments():
    dates=np.array(['2020-01-01','2020-01-02','2020-01-10','2020-01-11'],dtype='datetime64[D]')
    diff=np.full(4,-3.25)
    value=block_ci_once(diff,dates,14,np.random.default_rng(4))
    assert value == -3.25
    # sampled positions are confined by the implementation's segment boundary;
    # constant inputs also verify the statistic is invariant to resampling.


def test_seasonal_reversal_is_not_seed_disagreement_and_annual_weights_are_dates():
    rows = [dict(window=w, seed=s, model='main', comparator='other', variant='Cal',
                 metric='CRPS_MW', mean_difference=value)
            for w, value in [('quarter_1', -2.0), ('quarter_2', 3.0)]
            for s in (0, 1, 2)]
    differences, directions = seed_direction_tables(pd.DataFrame(rows), {'quarter_1':90, 'quarter_2':10})
    annual = differences[differences.window == 'FULL_YEAR']
    assert np.allclose(annual.mean_difference, -1.5)
    assert not directions.mixed_seed_signs.any()
    with pytest.raises(ValueError, match='Incomplete'):
        seed_direction_tables(pd.DataFrame(rows[:-1]), {'quarter_1':90, 'quarter_2':10})


def test_timezone_alternative_uses_complete_production_cleaning():
    starts = pd.date_range('2024-07-02', periods=48, freq='30min', tz='Europe/London').tz_convert('UTC')
    raw = pd.DataFrame({'Datetime_GMT':starts, 'Date':'2024-07-02',
                        'Settlement_period':np.arange(1,49), 'Capacity':1000.,
                        'Incentive_forecast':100., 'Forecast_Timestamp':'2024-07-01 09:00:00'})
    utc, _, _ = forecast_with_timezone(raw, 'UTC')
    local, _, _ = forecast_with_timezone(raw, 'Europe/London')
    assert not utc and len(local) == 1  # local 09:00 is UTC 08:00, before the cutoff
    for field, value, reason in [
        ('Incentive_forecast', -1., 'forecast_invalid_value_or_time_mapping'),
        ('Incentive_forecast', float('inf'), 'forecast_invalid_value_or_time_mapping'),
        ('Datetime_GMT', starts[0] + pd.Timedelta(minutes=5), 'forecast_noncontiguous_settlement_periods'),
    ]:
        bad = raw.copy()
        bad.loc[0, field] = value
        accepted, _, reasons = forecast_with_timezone(bad, 'Europe/London')
        assert not accepted and reasons[reason] == 1
    explicit = raw.copy()
    explicit['Forecast_Timestamp'] = '2024-07-01T09:00:00Z'
    accepted, _, _ = forecast_with_timezone(explicit, 'Europe/London')
    assert not accepted  # an explicit UTC offset must not be reinterpreted as local


def test_timezone_date_difference_keeps_excluded_calendar_dates_and_reasons():
    utc = pd.DataFrame([
        {'date':'2024-01-01','status':'eligible','reason':'eligible'},
        {'date':'2024-01-02','status':'excluded','reason':'forecast_after_information_cutoff'},
    ])
    local = pd.DataFrame([
        {'date':'2024-01-01','status':'excluded','reason':'forecast_after_information_cutoff'},
        {'date':'2024-01-02','status':'eligible','reason':'eligible'},
        {'date':'2024-01-03','status':'excluded','reason':'missing_forecast_rows'},
    ])
    rows = timezone_date_differences(utc, local).set_index('date')
    assert rows.index.tolist() == ['2024-01-01','2024-01-02','2024-01-03']
    assert rows.loc['2024-01-01','difference'] == 'UTC_only'
    assert rows.loc['2024-01-02','difference'] == 'London_only'
    assert rows.loc['2024-01-03','difference'] == 'neither'
    assert rows.loc['2024-01-02','utc_reason'] == 'forecast_after_information_cutoff'

