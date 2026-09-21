import numpy as np
import pandas as pd

from GEFcom2014.models.QBM_VAE.sdwpf5_experiment import (
    continuous_extrapolation_split,
    load_sdwpf_long_frame,
)


def test_sdwpf_long_etl_builds_utc_plus_8_wide_values_and_mask():
    frame = pd.DataFrame({
        "Day": ["2020-01-01"] * 4,
        "Tmstamp": ["00:00", "00:00", "01:00", "01:00"],
        "TurbID": ["T01", "T02", "T01", "T02"],
        "Patv": [1.0, np.nan, -1.0, 2.0],
        "shutdown": [False, True, False, False],
    })
    timestamps, values, valid, turbine_ids = load_sdwpf_long_frame(
        frame, shutdown_col="shutdown"
    )
    assert str(timestamps.tz) == "Asia/Shanghai"
    assert turbine_ids == ["T01", "T02"]
    assert values.shape == (2, 2)
    assert valid.tolist() == [[True, False], [False, True]]


def test_tz_aware_sdwpf_timestamps_can_be_split_chronologically():
    timestamps = pd.date_range("2020-01-01", periods=10, freq="h", tz="Asia/Shanghai")
    split = continuous_extrapolation_split(timestamps)
    assert split[0] == "LS"
    assert split[-1] == "TEST"
