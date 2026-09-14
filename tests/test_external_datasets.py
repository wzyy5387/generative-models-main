import numpy as np
import pandas as pd

from GEFcom2014.external_datasets import (
    OPSD_ACTUAL,
    OPSD_FORECAST,
    build_opsd_wind_bundle,
    load_daily_bundle,
)


def test_opsd_bundle_uses_chronological_splits_and_ls_only_scale(tmp_path):
    timestamps = pd.date_range("2016-01-01", periods=410 * 24, freq="h", tz="UTC")
    day = np.repeat(np.arange(410), 24)
    actual = 100.0 + day.astype(float)
    forecast = actual + 5.0
    frame = pd.DataFrame(
        {"utc_timestamp": timestamps, OPSD_ACTUAL: actual, OPSD_FORECAST: forecast}
    )
    source = tmp_path / "opsd.csv"
    output = tmp_path / "bundle.npz"
    frame.to_csv(source, index=False)

    metadata = build_opsd_wind_bundle(
        source, output, validation_days=20, test_days=20,
        scale_quantile=1.0, verify_source=False,
    )
    arrays, loaded_metadata, dates = load_daily_bundle(output)

    assert arrays["x_ls"].shape == (370, 33)
    assert arrays["y_ls"].shape == (370, 24)
    assert arrays["y_vs"].shape == (20, 24)
    assert arrays["y_test"].shape == (20, 24)
    assert metadata["scale_mw"] == 469.0
    assert loaded_metadata["scale_fit_split"] == "LS"
    assert dates["ls"][-1] < dates["vs"][0] < dates["test"][0]
    assert np.max(arrays["y_ls"]) == 1.0
    assert metadata["clipped_fraction"]["test"] == 1.0


def test_daily_bundle_rejects_missing_arrays(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, x_ls=np.zeros((2, 2)))
    try:
        load_daily_bundle(path)
    except ValueError as error:
        assert "missing arrays" in str(error)
    else:
        raise AssertionError("Expected invalid bundle to fail")
