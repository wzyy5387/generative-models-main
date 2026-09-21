# -*- coding: utf-8 -*-

"""Deterministic SDWPF-5 data protocol and smoke-test utilities.

The five turbines are selected from coordinates only. The smoke command uses
synthetic data solely to verify the pipeline when the SDWPF file is absent; it
is never a paper result and never reads TEST labels for model decisions.
"""

import argparse
import csv
import json
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "sdwpf5"


def select_turbines_by_coordinates(coordinates, n_turbines=5):
    """Select spatially covering turbines using coordinates and stable ties."""
    rows = list(coordinates)
    if len(rows) < n_turbines:
        raise ValueError("At least %d turbine coordinates are required" % n_turbines)
    normalized = []
    for row in rows:
        turbine_id = str(row["turbine_id"])
        x = float(row["x"])
        y = float(row["y"])
        if not np.isfinite([x, y]).all():
            raise ValueError("Coordinates must be finite")
        normalized.append((turbine_id, x, y))
    normalized.sort(key=lambda row: row[0])
    points = np.asarray([[row[1], row[2]] for row in normalized], dtype=np.float64)
    selected = [0]
    while len(selected) < n_turbines:
        remaining = [index for index in range(len(normalized)) if index not in selected]
        scores = []
        for index in remaining:
            distances = np.sum((points[index] - points[selected]) ** 2, axis=1)
            scores.append((float(np.min(distances)), normalized[index][0], index))
        selected.append(max(scores, key=lambda item: (item[0], item[1]))[2])
    return [normalized[index][0] for index in selected]


def load_coordinates(path, turbine_id_col="turbine_id", x_col="x", y_col="y"):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {turbine_id_col, x_col, y_col}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Coordinate CSV must contain %s" % sorted(required))
        return [
            {"turbine_id": row[turbine_id_col], "x": row[x_col], "y": row[y_col]}
            for row in reader
        ]


def load_wide_timeseries(path, timestamp_col="timestamp"):
    """Load a CSV with one timestamp column and one power column per turbine."""
    import pandas as pd

    frame = pd.read_csv(path)
    if timestamp_col not in frame.columns:
        raise ValueError("Data CSV must contain %s" % timestamp_col)
    timestamps = pd.to_datetime(frame[timestamp_col], errors="raise").to_numpy()
    turbine_ids = [str(column) for column in frame.columns if column != timestamp_col]
    if not turbine_ids:
        raise ValueError("Data CSV must contain at least one turbine column")
    values = frame[turbine_ids].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    return timestamps, values, turbine_ids


def load_sdwpf_long_frame(frame, day_col="Day", time_col="Tmstamp",
                          turbine_col="TurbID", power_col="Patv",
                          shutdown_col=None, anomaly_col=None):
    """Convert an official SDWPF long table to a UTC+08:00 wide table."""
    import pandas as pd

    required = {day_col, time_col, turbine_col, power_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("SDWPF long table missing %s" % sorted(missing))
    data = frame.copy()
    parsed = pd.to_datetime(
        data[day_col].astype(str).str.strip() + " " + data[time_col].astype(str).str.strip(),
        errors="coerce",
    )
    if parsed.isna().any():
        raise ValueError("Day/Tmstamp contains unparseable timestamps")
    data["_timestamp"] = parsed.dt.tz_localize("Asia/Shanghai")
    data[turbine_col] = data[turbine_col].astype(str)
    data["_power"] = pd.to_numeric(data[power_col], errors="coerce")
    if data.duplicated(["_timestamp", turbine_col]).any():
        raise ValueError("Duplicate timestamp/turbine rows are not auditable")
    numeric_power = data["_power"].to_numpy(dtype=np.float64)
    missing_mask = ~np.isfinite(numeric_power)
    anomaly_mask = np.isfinite(numeric_power) & (numeric_power < 0.0)
    shutdown_mask = np.zeros(len(data), dtype=bool)
    for column, target in ((shutdown_col, shutdown_mask), (anomaly_col, anomaly_mask)):
        if column is not None:
            if column not in data.columns:
                raise ValueError("Configured SDWPF mask column is absent: %s" % column)
            target[:] |= data[column].fillna(False).astype(bool).to_numpy()
    data["_valid"] = ~(missing_mask | shutdown_mask | anomaly_mask)
    turbine_ids = sorted(str(value) for value in data[turbine_col].unique())
    index = pd.DatetimeIndex(sorted(data["_timestamp"].unique()))
    value_table = data.pivot(index="_timestamp", columns=turbine_col, values="_power")
    valid_table = data.pivot(index="_timestamp", columns=turbine_col, values="_valid")
    value_table = value_table.reindex(index=index, columns=turbine_ids)
    valid_table = valid_table.reindex(index=index, columns=turbine_ids).fillna(False)
    values = value_table.to_numpy(dtype=np.float64)
    validity = valid_table.to_numpy(dtype=bool) & np.isfinite(values)
    return index, values, validity, turbine_ids


def load_sdwpf_long_parquet(path, **kwargs):
    import pandas as pd

    return load_sdwpf_long_frame(pd.read_parquet(path), **kwargs)


def continuous_extrapolation_split(timestamps, ls_fraction=0.70, vs_fraction=0.15):
    """Return chronological LS/VS/TEST masks with no shuffling or leakage."""
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 1 or timestamps.size < 3:
        raise ValueError("timestamps must be a one-dimensional non-empty sequence")
    try:
        import pandas as pd
        numeric = pd.DatetimeIndex(timestamps).asi8
    except (ImportError, TypeError, ValueError):
        numeric = timestamps.astype("datetime64[ns]").astype("int64")
    if np.any(numeric[1:] <= numeric[:-1]):
        raise ValueError("timestamps must be strictly increasing")
    if not 0 < ls_fraction < 1 or not 0 < vs_fraction < 1 or ls_fraction + vs_fraction >= 1:
        raise ValueError("Invalid LS/VS fractions")
    n_rows = timestamps.size
    ls_end = int(np.floor(n_rows * ls_fraction))
    vs_end = int(np.floor(n_rows * (ls_fraction + vs_fraction)))
    ls_end = max(1, min(ls_end, n_rows - 2))
    vs_end = max(ls_end + 1, min(vs_end, n_rows - 1))
    split = np.full(n_rows, "TEST", dtype="U4")
    split[:ls_end] = "LS"
    split[ls_end:vs_end] = "VS"
    return split


def split_time_boundaries(timestamps, split=None):
    timestamps = np.asarray(timestamps)
    split = continuous_extrapolation_split(timestamps) if split is None else np.asarray(split)
    return {
        name: {
            "start": str(timestamps[np.flatnonzero(split == name)[0]]),
            "end": str(timestamps[np.flatnonzero(split == name)[-1]]),
            "n_rows": int(np.count_nonzero(split == name)),
        }
        for name in ("LS", "VS", "TEST")
    }


def build_forecast_windows(values, timestamps, context_steps, horizon_steps):
    """Build target-end-labelled windows without crossing time boundaries."""
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps)
    if values.ndim != 2 or values.shape[0] != timestamps.size:
        raise ValueError("values must have shape (time, turbines)")
    split = continuous_extrapolation_split(timestamps)
    windows = {"LS": [], "VS": [], "TEST": []}
    for target_start in range(context_steps, values.shape[0] - horizon_steps + 1):
        target_end = target_start + horizon_steps - 1
        block = split[target_end]
        context = values[target_start - context_steps:target_start]
        target = values[target_start:target_end + 1]
        if split[target_start - context_steps] != block:
            continue
        windows[block].append((context, target))
    return {
        key: (
            np.stack([item[0] for item in block]),
            np.stack([item[1] for item in block]),
        ) if block else (np.empty((0, context_steps, values.shape[1])),
                         np.empty((0, horizon_steps, values.shape[1])))
        for key, block in windows.items()
    }


class ResourceMonitor:
    """Record wall time and Python allocation peak without extra dependencies."""

    def __enter__(self):
        tracemalloc.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.elapsed_seconds = time.perf_counter() - self.started
        self.peak_python_bytes = int(peak)


def run_smoke(output_dir, seed=0):
    rng = np.random.default_rng(seed)
    turbines = [
        {"turbine_id": "T01", "x": 0.0, "y": 0.0},
        {"turbine_id": "T02", "x": 1.0, "y": 0.0},
        {"turbine_id": "T03", "x": 0.0, "y": 1.0},
        {"turbine_id": "T04", "x": 1.0, "y": 1.0},
        {"turbine_id": "T05", "x": 4.0, "y": 4.0},
        {"turbine_id": "T06", "x": 8.0, "y": 0.0},
    ]
    selected = select_turbines_by_coordinates(turbines, n_turbines=5)
    timestamps = np.arange(
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-31T00:00"),
        np.timedelta64(1, "h"),
    )
    values = rng.normal(size=(timestamps.size, len(turbines)))
    selected_indices = [
        next(index for index, row in enumerate(turbines) if row["turbine_id"] == turbine)
        for turbine in selected
    ]
    with ResourceMonitor() as monitor:
        windows = build_forecast_windows(
            values[:, selected_indices], timestamps, context_steps=24, horizon_steps=24
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "sdwpf5_smoke",
        "smoke_only": True,
        "synthetic_data": True,
        "eligible_for_paper": False,
        "seed": int(seed),
        "selected_turbines": selected,
        "selection_basis": "coordinates_only_deterministic_farthest_point",
        "forecast_context_steps": 24,
        "forecast_horizon_steps": 24,
        "time_boundaries": split_time_boundaries(timestamps),
        "window_counts": {key: int(value[0].shape[0]) for key, value in windows.items()},
        "resource_monitor": {
            "elapsed_seconds": monitor.elapsed_seconds,
            "peak_python_bytes": monitor.peak_python_bytes,
        },
        "test_labels_used_for_selection_or_training": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "sdwpf5_smoke_seed0.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path, result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--data-format", choices=["wide-csv", "sdwpf-long-parquet"],
                        default="wide-csv")
    parser.add_argument("--shutdown-col", default=None)
    parser.add_argument("--anomaly-col", default=None)
    parser.add_argument("--context-steps", type=int, default=168)
    parser.add_argument("--horizon-steps", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        path, result = run_smoke(args.output_dir, seed=args.seed)
        print("Wrote SDWPF-5 synthetic smoke metadata to %s" % path.resolve())
        print("Selected turbines: %s" % result["selected_turbines"])
        return
    if args.coordinates is None or args.data is None:
        raise ValueError("Formal SDWPF-5 runs require --coordinates and --data")
    coordinates = load_coordinates(args.coordinates)
    if args.data_format == "sdwpf-long-parquet":
        timestamps, values, validity, turbine_ids = load_sdwpf_long_parquet(
            args.data, shutdown_col=args.shutdown_col, anomaly_col=args.anomaly_col
        )
    else:
        timestamps, values, turbine_ids = load_wide_timeseries(
            args.data, timestamp_col=args.timestamp_col
        )
        validity = np.isfinite(values)
    selected = select_turbines_by_coordinates(coordinates, n_turbines=5)
    missing = sorted(set(selected) - set(turbine_ids))
    if missing:
        raise ValueError("Selected turbines are missing from data: %s" % missing)
    selected_indices = [turbine_ids.index(turbine) for turbine in selected]
    with ResourceMonitor() as monitor:
        windows = build_forecast_windows(
            values[:, selected_indices], timestamps, args.context_steps, args.horizon_steps
        )
    result = {
        "experiment": "sdwpf5_protocol",
        "smoke_only": False,
        "synthetic_data": False,
        "data_format": args.data_format,
        "seed": int(args.seed),
        "selected_turbines": selected,
        "selection_basis": "coordinates_only_deterministic_farthest_point",
        "timestamps": {
            "first": str(timestamps[0]),
            "last": str(timestamps[-1]),
            "n_rows": int(timestamps.size),
        },
        "time_boundaries": split_time_boundaries(timestamps),
        "split_counts": {key: int(value[0].shape[0]) for key, value in windows.items()},
        "validity_fraction": float(np.mean(validity)),
        "mask_policy": "missing/non-finite, explicit shutdown/anomaly flags, and negative Patv are invalid",
        "forecast_context_steps": int(args.context_steps),
        "forecast_horizon_steps": int(args.horizon_steps),
        "resource_monitor": {
            "elapsed_seconds": monitor.elapsed_seconds,
            "peak_python_bytes": monitor.peak_python_bytes,
        },
        "test_labels_used_for_selection_or_training": False,
        "formal_model_training_executed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ("sdwpf5_protocol_seed%d.json" % args.seed)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Wrote SDWPF-5 protocol metadata to %s" % path.resolve())


if __name__ == "__main__":
    main()
