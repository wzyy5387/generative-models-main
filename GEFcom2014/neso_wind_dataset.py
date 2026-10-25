# -*- coding: utf-8 -*-
"""Build an LS/VS/TEST daily bundle from NESO day-ahead wind data."""

import argparse
from collections import Counter
import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LONDON = ZoneInfo("Europe/London")
FORECAST_URL = (
    "https://api.neso.energy/dataset/fbe3701d-1487-443e-abe9-47a6c01ecce2/"
    "resource/7524ec65-f782-4258-aaf8-5b926c17b966/download/"
    "7524ec65-f782-4258-aaf8-5b926c17b966-20261013084501.csv"
)
ACTUAL_BASE_URL = (
    "https://api.neso.energy/dataset/7f7fa642-6eff-4b8f-9e8b-7ede1ba50e20/"
    "resource/{resource}/download/{filename}"
)
ACTUAL_RESOURCES = {
    "2018-2019": "bf03c648-98d8-40f3-b5d9-e174cb2c1f81",
    "2019-2020": "c36868d9-9d43-47ce-ac3e-de69f0bb6222",
    "2020-2021": "043d5423-f877-4d28-864e-210dc68711d4",
    "2021-2022": "e5df03f3-a25b-4df4-a95c-8a709815924b",
    "2022-2023": "f732e9bb-b573-46a7-8767-3affbbb29b45",
    "2023-2024": "c47155bc-71b8-4b0a-aba3-0e2d1295daea",
    "2024-2025": "3ac8c742-d601-4b6c-9779-e69cea94e134",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cutoff_utc(date, hour=9, minute=15):
    """Previous *local calendar date* at the fixed London clock time."""
    previous = date - timedelta(days=1)
    return pd.Timestamp(datetime.combine(previous, time(hour, minute), tzinfo=LONDON)).tz_convert("UTC")


def _forecast_days(path, cutoff_hour=9, cutoff_minute=15, audit=None):
    frame = pd.read_csv(path, parse_dates=["Datetime_GMT", "Forecast_Timestamp"])
    required = {"Datetime_GMT", "Date", "Settlement_period", "Capacity", "Incentive_forecast", "Forecast_Timestamp"}
    if not required.issubset(frame.columns):
        raise ValueError("NESO forecast file lacks required columns")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.date
    frame["Forecast_Timestamp"] = pd.to_datetime(frame["Forecast_Timestamp"], utc=True, errors="coerce")
    frame["Datetime_GMT"] = pd.to_datetime(frame["Datetime_GMT"], utc=True, errors="coerce")
    result, reasons = {}, {}
    for date, day in frame.groupby("Date", sort=True):
        if pd.isna(date):
            reasons["invalid_date"] = reasons.get("invalid_date", 0) + 1
            continue
        cutoff = _cutoff_utc(date, cutoff_hour, cutoff_minute)
        detail = {"forecast_rows": len(day), "forecast_versions": int(day["Forecast_Timestamp"].nunique()),
                  "cutoff_utc": cutoff.isoformat(),
                  "latest_source_issue_utc": day["Forecast_Timestamp"].max().isoformat()
                  if day["Forecast_Timestamp"].notna().any() else ""}
        if audit is not None:
            audit[date] = detail

        def reject(reason):
            reasons[reason] = reasons.get(reason, 0) + 1
            detail["forecast_reason"] = reason

        eligible = day[day["Forecast_Timestamp"] <= cutoff]
        if eligible.empty:
            reject("invalid_forecast_timestamp" if day["Forecast_Timestamp"].isna().all()
                   else "forecast_after_information_cutoff")
            continue
        issue = eligible["Forecast_Timestamp"].max()
        detail["selected_issue_utc"] = issue.isoformat()
        selected = eligible[eligible["Forecast_Timestamp"] == issue].copy()
        periods = pd.to_numeric(selected["Settlement_period"], errors="coerce")
        values = pd.to_numeric(selected["Incentive_forecast"], errors="coerce")
        starts = selected["Datetime_GMT"].dt.tz_convert(LONDON)
        if periods.isna().any() or periods.duplicated().any():
            reject("forecast_duplicate_or_invalid_settlement_periods")
            continue
        if len(selected) != 48 or set(periods.astype(int)) != set(range(1, 49)):
            reject("forecast_incomplete_settlement_periods")
            continue
        if (not np.isfinite(values).all() or (values < 0).any() or starts.isna().any()
                or not (starts.dt.date == date).all()):
            reject("forecast_invalid_value_or_time_mapping")
            continue
        local_hours = starts.dt.hour.to_numpy()
        counts = np.bincount(local_hours, minlength=24)
        if not np.array_equal(counts, np.full(24, 2)):
            reject("forecast_not_24_local_hours_dst_or_bad_time")
            continue
        order = np.argsort(starts.astype("int64").to_numpy())
        local_sorted = starts.iloc[order]
        expected_starts = pd.date_range(
            pd.Timestamp(date).tz_localize(LONDON), periods=48, freq="30min"
        )
        if not np.array_equal(local_sorted.astype("int64").to_numpy(), expected_starts.astype("int64").to_numpy()):
            reject("forecast_noncontiguous_settlement_periods")
            continue
        ordered_values = values.to_numpy(dtype=np.float64)[order].reshape(24, 2)
        result[date] = {"values": ordered_values.mean(axis=1), "issue_time_utc": issue.isoformat(),
                        "capacity_mw": float(pd.to_numeric(selected["Capacity"], errors="coerce").median())}
        detail["forecast_reason"] = "eligible"
    return result, reasons, len(frame)


def _actual_days(paths, audit=None):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["Sett_Date"])
        frames.append(frame)
    if not frames:
        raise ValueError("No NESO operational-metered actual files supplied")
    frame = pd.concat(frames, ignore_index=True)
    required = {"Sett_Date", "Sett_Period", "Scottish Wind Output", "England/Wales Wind Output", "Total"}
    if not required.issubset(frame.columns):
        raise ValueError("NESO actual file lacks required columns")
    frame["Sett_Date"] = pd.to_datetime(frame["Sett_Date"], errors="coerce").dt.date
    result, reasons = {}, {}
    for date, day in frame.groupby("Sett_Date", sort=True):
        detail = {"actual_rows": len(day)}
        if audit is not None:
            audit[date] = detail

        def reject(reason):
            reasons[reason] = reasons.get(reason, 0) + 1
            detail["actual_reason"] = reason

        periods = pd.to_numeric(day["Sett_Period"], errors="coerce")
        values = pd.to_numeric(day["Total"], errors="coerce")
        components = day[["Scottish Wind Output", "England/Wales Wind Output"]].apply(pd.to_numeric, errors="coerce")
        if periods.isna().any() or periods.duplicated().any():
            reject("actual_duplicate_or_invalid_settlement_periods")
            continue
        if len(day) != 48 or set(periods.astype(int)) != set(range(1, 49)):
            reject("actual_incomplete_settlement_periods")
            continue
        if not np.isfinite(values).all() or not np.isfinite(components.to_numpy()).all() or (values < 0).any():
            reject("actual_invalid_values")
            continue
        if not np.allclose(values, components.sum(axis=1).to_numpy(), rtol=0, atol=0.01):
            reject("actual_total_component_mismatch")
            continue
        midnight = pd.Timestamp(date).tz_localize(LONDON)
        starts = pd.date_range(midnight, periods=48, freq="30min")
        if not np.array_equal(np.bincount(starts.hour, minlength=24), np.full(24, 2)):
            reject("actual_dst_day_not_24_hours")
            continue
        order = np.argsort(periods.to_numpy(dtype=int))
        result[date] = values.to_numpy(dtype=np.float64)[order].reshape(24, 2).mean(axis=1)
        detail["actual_reason"] = "eligible"
    return result, reasons, len(frame), int(frame.duplicated(["Sett_Date", "Sett_Period"]).sum())


def _eligibility_rows(forecast_audit, actual_audit, forecast, actual):
    """One reasoned record per calendar date in the raw source span."""
    first = min(min(forecast_audit), min(actual_audit))
    last = max(max(forecast_audit), max(actual_audit))
    rows = []
    for stamp in pd.date_range(first, last, freq="D"):
        date = stamp.date()
        f = forecast_audit.get(date, {})
        a = actual_audit.get(date, {})
        forecast_reason = f.get("forecast_reason", "missing_forecast_rows")
        actual_reason = a.get("actual_reason", "missing_actual_rows")
        reasons = [reason for reason in (forecast_reason, actual_reason) if reason != "eligible"]
        rows.append({"date": str(date), "month": stamp.strftime("%Y-%m"),
                     "status": "eligible" if date in forecast and date in actual else "excluded",
                     "reason": ";".join(reasons) or "eligible", "forecast_reason": forecast_reason,
                     "actual_reason": actual_reason, "forecast_rows": f.get("forecast_rows", 0),
                     "forecast_versions": f.get("forecast_versions", 0),
                     "actual_rows": a.get("actual_rows", 0), "cutoff_utc": f.get("cutoff_utc", ""),
                     "latest_source_issue_utc": f.get("latest_source_issue_utc", ""),
                     "selected_issue_utc": f.get("selected_issue_utc", "")})
    return pd.DataFrame(rows)


def _monthly_coverage(eligibility):
    rows = []
    for month, group in eligibility.groupby("month", sort=True):
        row = {"month": month, "calendar_days_in_source_span": len(group),
               "eligible_days": int((group.status == "eligible").sum()),
               "coverage_fraction": float((group.status == "eligible").mean())}
        for name in ("forecast_reason", "actual_reason"):
            for reason, count in group[name].value_counts().items():
                if reason != "eligible":
                    row[name + "__" + reason] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def build_neso_wind_bundle(forecast_csv, actual_csvs, output_path, validation_days=90, test_days=90,
                           min_ls_days=365, audit_dir=None):
    forecast_csv, actual_csvs, output_path = Path(forecast_csv), [Path(x) for x in actual_csvs], Path(output_path)
    forecast_audit, actual_audit = {}, {}
    forecast, forecast_reasons, forecast_rows = _forecast_days(forecast_csv, audit=forecast_audit)
    actual, actual_reasons, actual_rows, actual_duplicates = _actual_days(actual_csvs, audit=actual_audit)
    eligibility = _eligibility_rows(forecast_audit, actual_audit, forecast, actual)
    monthly = _monthly_coverage(eligibility)
    common = sorted(set(forecast) & set(actual))
    rejected = dict(Counter(eligibility.loc[eligibility.status == "excluded", "reason"]))
    if len(common) <= validation_days + test_days + min_ls_days:
        raise ValueError("Eligible daily coverage cannot satisfy the requested LS/VS/TEST split")
    dates = pd.DatetimeIndex(common)
    x_forecast = np.stack([forecast[d]["values"] for d in common])
    y = np.stack([actual[d] for d in common])
    doy = dates.dayofyear.to_numpy(dtype=float)
    calendar = np.column_stack((np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), np.eye(7)[dates.dayofweek]))
    x = np.concatenate((x_forecast, calendar), axis=1)
    n_ls = len(dates) - validation_days - test_days
    slices = {"ls": slice(0, n_ls), "vs": slice(n_ls, n_ls + validation_days), "test": slice(n_ls + validation_days, len(dates))}
    arrays, date_arrays = {}, {}
    for split, slc in slices.items():
        arrays["x_" + split] = x[slc].astype(np.float32)
        arrays["y_" + split] = y[slc].astype(np.float32)
        date_arrays["dates_" + split] = np.asarray(dates[slc].strftime("%Y-%m-%d"), dtype="U10")
    ls_values = y[slices["ls"]]
    metadata = {
        "dataset_name": "neso-wind", "source": "NESO Day Ahead Wind Forecast and Monthly Operational Metered Wind Output",
        "source_urls": {"forecast": FORECAST_URL, "actual_files": [ACTUAL_BASE_URL.format(resource=ACTUAL_RESOURCES[p.name.removeprefix("monthly-operational-metered-wind-output-").removesuffix(".csv")], filename=p.name) for p in actual_csvs]},
        "source_files": [{"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name,
                          "sha256": sha256(p)} for p in [forecast_csv, *actual_csvs]],
        "forecast_fields": {"target_date": "Date", "target_time_utc": "Datetime_GMT", "settlement_period": "Settlement_period", "value_mw": "Incentive_forecast", "published_at_utc": "Forecast_Timestamp", "capacity_mw": "Capacity"},
        "actual_fields": {"settlement_date": "Sett_Date", "settlement_period": "Sett_Period", "target_mw": "Total", "components": ["Scottish Wind Output", "England/Wales Wind Output"]},
        "target_scope": "national operational-metered wind output; Scotland + England/Wales Total; embedded wind excluded",
        "official_scope_note": "NESO states operational-metered output differs from settlement-metered outturn used in forecast-metric reporting; results use this operational-metered target consistently and are not those settlement-metered metrics.",
        "forecast_publication": {"official_frequency": "daily", "official_time": "09:00-09:15 Europe/London", "frozen_cutoff": "09:15 Europe/London on the day before target date", "historical_selection": "latest Forecast_Timestamp no later than frozen cutoff; later archive updates rejected", "all_selected_timestamps_verified": True},
        "timezone": "Europe/London Settlement Date/Period; Datetime_GMT and Forecast_Timestamp interpreted UTC; intervals mapped to UTC then aggregated by UK local hour",
        "aggregation": "Arithmetic time-weighted mean of two contiguous 30-minute average MW readings per UK local hour; not summed; DST 23/25-hour days excluded from 24-hour protocol.",
        "duplicate_handling": {"forecast": "latest eligible issue timestamp selected; duplicate periods invalidate target day", "actual_duplicate_period_rows": actual_duplicates, "actual": "duplicate periods invalidate target day"},
        "invalid_day_reasons": rejected,
        "forecast_source_reason_counts": forecast_reasons,
        "actual_source_reason_counts": actual_reasons,
        "forecast_source_rows": forecast_rows, "actual_source_rows": actual_rows,
        "complete_eligible_dates": len(dates), "date_ranges": {s: [str(v[0]), str(v[-1])] for s, v in date_arrays.items()},
        "split_strategy": "chronological eligible dates: last 90 TEST, preceding 90 VS, earlier LS; no split-crossing interpolation",
        "split_days": {s: len(arrays["y_" + s]) for s in slices},
        "dates_by_split": {s: date_arrays["dates_" + s].tolist() for s in slices},
        "missing_calendar_dates_by_split": {
            s: [str(d.date()) for d in pd.date_range(str(date_arrays["dates_" + s][0]), str(date_arrays["dates_" + s][-1]), freq="D")
                if str(d.date()) not in set(date_arrays["dates_" + s])]
            for s in slices
        },
        "context_features": ["day_ahead_forecast_hour_%02d_mw" % h for h in range(24)] + ["day_of_year_sin", "day_of_year_cos"] + ["weekday_%d" % d for d in range(7)],
        "anchor_contract": "first 24 raw context columns are the same-day day-ahead hourly forecast in MW",
        "target_features": ["operational_metered_wind_hour_%02d_mw" % h for h in range(24)],
        "power_normalization": "bundle stores MW without clipping; model scaler is fit on LS only and scenario outputs are inverse-transformed to MW",
        "ls_target_range_mw": [float(ls_values.min()), float(ls_values.max())],
        "target_outside_ls_range_fraction": {s: float(np.mean((y[slc] < ls_values.min()) | (y[slc] > ls_values.max()))) for s, slc in slices.items()},
        "target_above_ls_max_fraction": {s: float(np.mean(y[slc] > ls_values.max())) for s, slc in slices.items()},
        "forecast_issue_timestamps_utc": {str(d): forecast[d]["issue_time_utc"] for d in common},
        "forecast_capacity_mw_by_date": {str(d): forecast[d]["capacity_mw"] for d in common},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays, **date_arrays, metadata=json.dumps(metadata))
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        eligibility.to_csv(audit_dir / "date_eligibility.csv", index=False)
        monthly.to_csv(audit_dir / "monthly_coverage.csv", index=False)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-csv", type=Path, required=True)
    parser.add_argument("--actual-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_neso_wind_bundle(args.forecast_csv, args.actual_csv, args.output,
                                             audit_dir=args.audit_dir), indent=2))


if __name__ == "__main__":
    main()
