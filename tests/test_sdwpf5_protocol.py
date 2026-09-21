import numpy as np

from GEFcom2014.models.QBM_VAE.sdwpf5_experiment import (
    build_forecast_windows,
    continuous_extrapolation_split,
    select_turbines_by_coordinates,
    split_time_boundaries,
)


def test_coordinate_selection_is_deterministic_and_coordinate_only():
    coordinates = [
        {"turbine_id": "T03", "x": 0.0, "y": 1.0},
        {"turbine_id": "T01", "x": 0.0, "y": 0.0},
        {"turbine_id": "T05", "x": 4.0, "y": 4.0},
        {"turbine_id": "T02", "x": 1.0, "y": 0.0},
        {"turbine_id": "T04", "x": 1.0, "y": 1.0},
        {"turbine_id": "T06", "x": 8.0, "y": 0.0},
    ]
    first = select_turbines_by_coordinates(coordinates, n_turbines=5)
    second = select_turbines_by_coordinates(list(reversed(coordinates)), n_turbines=5)
    assert first == second
    assert first == ["T01", "T06", "T05", "T04", "T03"]


def test_continuous_split_is_ordered_and_has_no_shuffle():
    timestamps = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-11"),
        np.timedelta64(1, "D"),
    )
    split = continuous_extrapolation_split(timestamps)
    np.testing.assert_array_equal(split, ["LS"] * 7 + ["VS"] + ["TEST"] * 2)
    boundaries = split_time_boundaries(timestamps, split)
    assert boundaries["LS"]["n_rows"] == 7
    assert boundaries["VS"]["n_rows"] == 1
    assert boundaries["TEST"]["n_rows"] == 2


def test_windows_discard_cross_boundary_contexts():
    timestamps = np.arange(
        np.datetime64("2020-01-01T00"),
        np.datetime64("2020-01-11T00"),
        np.timedelta64(1, "h"),
    )
    values = np.arange(timestamps.size * 2, dtype=np.float64).reshape(-1, 2)
    windows = build_forecast_windows(values, timestamps, context_steps=24, horizon_steps=24)
    for context, target in windows.values():
        assert context.shape[1:] == (24, 2)
        assert target.shape[1:] == (24, 2)
