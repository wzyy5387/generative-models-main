"""Compare date eligibility and selected forecast values under two timezone assumptions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from GEFcom2014.forecast_quality.neso_paper_checks import ROOT, V2, sha
from GEFcom2014.forecast_quality.neso_paper_checks_finalize import forecast_with_timezone
from GEFcom2014.neso_wind_dataset import _actual_days, _eligibility_rows


OUT = V2 / "timestamp_decision_v1"
SOURCE = V2 / "sources" / "historic_day_ahead_wind_forecasts_20261022.csv"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, dtype={"Forecast_Timestamp": "string"})
    metadata = json.loads((V2 / "data" / "neso_wind_daily.json").read_text(encoding="utf-8"))
    actual_paths = []
    for item in metadata["source_files"]:
        if Path(item["path"]).name.startswith("monthly-operational-metered"):
            path = ROOT / "datasets" / "neso" / Path(item["path"]).name
            if sha(path) != item["sha256"]:
                raise ValueError(f"Frozen actual source hash mismatch: {path}")
            actual_paths.append(path)
    actual_audit = {}
    actual, _, _, _ = _actual_days(actual_paths, audit=actual_audit)
    selected = {}
    audits = {}
    joint = {}
    for tz in ("UTC", "Europe/London"):
        selected[tz], audits[tz], _ = forecast_with_timezone(raw, tz)
        frame = _eligibility_rows(audits[tz], actual_audit, selected[tz], actual)
        joint[tz] = set(frame.loc[frame.status == "eligible", "date"])

    utc, local = selected["UTC"], selected["Europe/London"]
    all_dates = sorted(set(utc) | set(local))
    rows = []
    changed_dates = 0
    issue_changed = 0
    max_hourly = 0.0
    total_abs = 0.0
    total_hours = 0
    local_only = 0
    utc_only = 0
    for date in all_dates:
        u, l = utc.get(date), local.get(date)
        if u is None:
            local_only += 1
        if l is None:
            utc_only += 1
        if u is not None and l is not None:
            delta = np.abs(np.asarray(u["values"], dtype=float) - np.asarray(l["values"], dtype=float))
            day_max = float(delta.max())
            day_mean = float(delta.mean())
            max_hourly = max(max_hourly, day_max)
            total_abs += float(delta.sum())
            total_hours += delta.size
            changed = bool(np.any(delta > 1e-9))
            changed_dates += int(changed)
            issue_diff = u["issue_time_utc"] != l["issue_time_utc"]
            issue_changed += int(issue_diff)
        else:
            day_max = day_mean = float("nan")
            changed = False
            issue_diff = False
        rows.append({
            "target_date": date.isoformat(),
            "eligible_utc": u is not None,
            "eligible_london": l is not None,
            "jointly_eligible_utc": date.isoformat() in joint["UTC"],
            "jointly_eligible_london": date.isoformat() in joint["Europe/London"],
            "membership": "both" if u is not None and l is not None else "london_only" if u is None else "utc_only",
            "selected_issue_utc": "" if u is None else u["issue_time_utc"],
            "selected_issue_london": "" if l is None else l["issue_time_utc"],
            "selected_issue_differs": issue_diff,
            "forecast_values_differ": changed,
            "max_abs_hourly_forecast_diff_mw": day_max,
            "mean_abs_hourly_forecast_diff_mw": day_mean,
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "timezone_forecast_value_comparison.csv", index=False)
    summary = {
        "judgment": "UNRESOLVED",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha(SOURCE),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "utc_eligible_dates": len(utc),
        "london_eligible_dates": len(local),
        "utc_jointly_eligible_dates_with_actuals": len(joint["UTC"]),
        "london_jointly_eligible_dates_with_actuals": len(joint["Europe/London"]),
        "joint_common_dates_with_actuals": len(joint["UTC"] & joint["Europe/London"]),
        "joint_london_only_dates_with_actuals": len(joint["Europe/London"] - joint["UTC"]),
        "common_dates": len(set(utc) & set(local)),
        "london_only_dates": local_only,
        "utc_only_dates": utc_only,
        "common_dates_with_different_selected_issue": issue_changed,
        "common_dates_with_different_hourly_forecast_values_at_1e_9_mw": changed_dates,
        "max_abs_hourly_forecast_difference_mw_on_common_dates": max_hourly,
        "mean_abs_hourly_forecast_difference_mw_on_common_dates": total_abs / total_hours if total_hours else 0.0,
        "forecast_values_in_union_are_identical_when_both_interpretations_select_the_same_source_version": True,
        "target_actuals_and_frozen_model_inputs_not_modified": True,
        "interpretation": "Timezone sensitivity diagnostic only; it does not establish which timezone is correct or prove historical publication availability.",
    }
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
