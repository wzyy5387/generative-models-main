# -*- coding: utf-8 -*-

"""Build and validate reproducible daily datasets outside GEFCom2014."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OPSD_RELEASE = "2019-06-05"
OPSD_DOI = "10.25832/time_series/2019-06-05"
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2019-06-05/"
    "time_series_60min_singleindex.csv"
)
OPSD_SHA256 = "659fe789af2672aabe989aebc8c5c21052a1a96e4da70b0fc941910a1cd4de9d"
OPSD_ACTUAL = "DE_50hertz_wind_onshore_generation_actual"
OPSD_FORECAST = "DE_50hertz_wind_onshore_generation_forecast"
BUNDLE_ARRAY_KEYS = ("x_ls", "y_ls", "x_vs", "y_vs", "x_test", "y_test")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_opsd_wind_bundle(
    source_csv,
    output_path,
    validation_days=90,
    test_days=90,
    scale_quantile=1.0,
    verify_source=True,
):
    source_csv = Path(source_csv)
    output_path = Path(output_path)
    if verify_source:
        source_hash = sha256_file(source_csv)
        if source_hash != OPSD_SHA256:
            raise ValueError("OPSD source SHA-256 mismatch: %s" % source_hash)
    else:
        source_hash = sha256_file(source_csv)

    frame = pd.read_csv(
        source_csv,
        usecols=["utc_timestamp", OPSD_ACTUAL, OPSD_FORECAST],
        parse_dates=["utc_timestamp"],
    ).dropna()
    frame = frame.sort_values("utc_timestamp").drop_duplicates("utc_timestamp")
    frame["date"] = frame["utc_timestamp"].dt.floor("D")
    frame["hour"] = frame["utc_timestamp"].dt.hour

    actual_days = []
    forecast_days = []
    dates = []
    expected_hours = np.arange(24)
    for date, group in frame.groupby("date", sort=True):
        group = group.sort_values("hour")
        if len(group) != 24 or not np.array_equal(group["hour"].to_numpy(), expected_hours):
            continue
        actual_days.append(group[OPSD_ACTUAL].to_numpy(dtype=np.float64))
        forecast_days.append(group[OPSD_FORECAST].to_numpy(dtype=np.float64))
        dates.append(pd.Timestamp(date))

    actual = np.asarray(actual_days)
    forecast = np.asarray(forecast_days)
    dates = pd.DatetimeIndex(dates)
    n_ls = len(dates) - int(validation_days) - int(test_days)
    if n_ls < 365:
        raise ValueError("At least 365 learning days are required after chronological splitting")
    if not 0.9 <= scale_quantile <= 1.0:
        raise ValueError("scale_quantile must be between 0.9 and 1.0")

    scale_mw = float(np.quantile(actual[:n_ls], scale_quantile))
    if not np.isfinite(scale_mw) or scale_mw <= 0:
        raise ValueError("Invalid LS-only wind normalization scale")
    target = np.clip(actual / scale_mw, 0.0, 1.0)
    forecast_normalized = np.clip(forecast / scale_mw, 0.0, 1.0)

    day_of_year = dates.dayofyear.to_numpy(dtype=np.float64)
    calendar = np.column_stack(
        (
            np.sin(2.0 * np.pi * day_of_year / 365.25),
            np.cos(2.0 * np.pi * day_of_year / 365.25),
            np.eye(7, dtype=np.float64)[dates.dayofweek.to_numpy()],
        )
    )
    context = np.concatenate((forecast_normalized, calendar), axis=1)
    boundaries = (n_ls, n_ls + int(validation_days))
    slices = (slice(0, boundaries[0]), slice(boundaries[0], boundaries[1]), slice(boundaries[1], None))
    split_names = ("ls", "vs", "test")
    arrays = {}
    date_arrays = {}
    for name, split_slice in zip(split_names, slices):
        arrays["x_" + name] = context[split_slice].astype(np.float32)
        arrays["y_" + name] = target[split_slice].astype(np.float32)
        date_arrays["dates_" + name] = np.asarray(
            dates[split_slice].strftime("%Y-%m-%d"), dtype="U10"
        )

    metadata = {
        "dataset_name": "opsd-wind",
        "source": "Open Power System Data Time Series",
        "release": OPSD_RELEASE,
        "doi": OPSD_DOI,
        "url": OPSD_URL,
        "source_sha256": source_hash,
        "actual_column": OPSD_ACTUAL,
        "forecast_column": OPSD_FORECAST,
        "timezone": "UTC",
        "split_strategy": "chronological",
        "validation_days": int(validation_days),
        "test_days": int(test_days),
        "scale_fit_split": "LS",
        "scale_quantile": float(scale_quantile),
        "scale_mw": scale_mw,
        "context_features": ["forecast_hour_%02d" % hour for hour in range(24)]
        + ["day_of_year_sin", "day_of_year_cos"]
        + ["weekday_%d" % day for day in range(7)],
        "target_features": ["wind_hour_%02d" % hour for hour in range(24)],
        "complete_days": int(len(dates)),
        "split_days": {name: int(len(arrays["y_" + name])) for name in split_names},
        "date_ranges": {
            name: [str(date_arrays["dates_" + name][0]), str(date_arrays["dates_" + name][-1])]
            for name in split_names
        },
        "clipped_fraction": {
            name: float(np.mean(actual[split_slice] > scale_mw))
            for name, split_slice in zip(split_names, slices)
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays, **date_arrays, metadata=json.dumps(metadata))
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def load_daily_bundle(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        missing = [key for key in BUNDLE_ARRAY_KEYS if key not in bundle]
        if missing:
            raise ValueError("Dataset bundle is missing arrays: %s" % ", ".join(missing))
        arrays = {key: np.asarray(bundle[key], dtype=np.float32) for key in BUNDLE_ARRAY_KEYS}
        metadata = json.loads(str(bundle["metadata"])) if "metadata" in bundle else {}
        dates = {
            split: np.asarray(bundle["dates_" + split]).astype(str)
            for split in ("ls", "vs", "test")
            if "dates_" + split in bundle
        }
    for split in ("ls", "vs", "test"):
        x, y = arrays["x_" + split], arrays["y_" + split]
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("Invalid %s arrays in %s" % (split, path))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite values in %s split" % split)
    return arrays, metadata, dates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "external" / "opsd_wind_daily.npz",
    )
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--scale-quantile", type=float, default=1.0)
    parser.add_argument("--skip-source-verification", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = build_opsd_wind_bundle(
        args.source_csv,
        args.output,
        validation_days=args.validation_days,
        test_days=args.test_days,
        scale_quantile=args.scale_quantile,
        verify_source=not args.skip_source_verification,
    )
    print("Wrote %s" % args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
