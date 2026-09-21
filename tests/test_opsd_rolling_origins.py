import json

import numpy as np
import pytest

from GEFcom2014.forecast_quality.opsd_rolling_origins import (
    _origin_blocks,
    _resume_mismatch_reasons,
    aggregate_metric_rows,
    build_run_manifest,
    load_protocol,
    origin_windows,
    smoke_rows,
    validate_formal_matrix,
)
from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily


def test_opsd_protocol_has_fixed_expanding_origins_and_manifest_size():
    protocol = load_protocol()
    windows = origin_windows(protocol)
    assert windows[0]["ls_start"] == "2016-01-01"
    assert [window["ls_days"] for window in windows] == [676, 856, 1036]
    assert windows[1]["ls_end"] == "2018-05-05"
    assert windows[2]["test_start"] == "2019-01-31"
    assert len(build_run_manifest(protocol)) == 3 * 4 * 5
    assert all(row["test_used_for_selection"] is False for row in build_run_manifest(protocol))


def test_opsd_aggregation_keeps_date_blocks_and_paired_inference():
    protocol = load_protocol()
    result = aggregate_metric_rows(
        smoke_rows(protocol), bootstrap_repetitions=100, permutation_repetitions=100
    )
    assert len(result["per_date_seed_means"]) == 3 * 6 * 4
    comparisons = [row for row in result["paired_comparisons"]
                   if row["metric"] == "CRPS_raw"]
    assert len(comparisons) == 3
    assert all(row["n_dates"] == 18 for row in comparisons)
    assert all(row["n_date_blocks"] == 3 for row in comparisons)
    assert all(np.isfinite(row["ci_2.5"]) for row in comparisons)


def test_time_blocks_are_built_inside_each_origin_only():
    blocks = _origin_blocks(np.zeros(12), ["origin_1"] * 6 + ["origin_2"] * 6, 7)
    assert [len(block) for block in blocks] == [6, 6]
    assert all(set(block).issubset(set(range(6))) or set(block).issubset(set(range(6, 12)))
               for block in blocks)


def test_formal_matrix_completeness_fails_closed():
    with pytest.raises(ValueError, match="60 registered"):
        validate_formal_matrix([], [], load_protocol())


def test_resume_requires_all_source_and_output_hashes(tmp_path):
    from GEFcom2014.forecast_quality.opsd_rolling_origins import file_sha256
    scenario = tmp_path / "scenario.bin"
    metrics = tmp_path / "metrics.csv"
    scenario.write_bytes(b"scenario")
    metrics.write_bytes(b"metrics")
    protocol = load_protocol()
    row = build_run_manifest(protocol)[0]
    provenance = {
        "protocol_sha256": "protocol",
        "dataset_sha256": "dataset",
        "source_hashes": {"rolling_runner": "runner",
                           "model_entrypoints": {row["model"]: "model"}},
    }
    existing = {**row, "status": "COMPLETED", "protocol_sha256": "protocol",
                "dataset_sha256": "dataset", "runner_sha256": "runner",
                "model_source_sha256": "model", "scenario_file": str(scenario),
                "scenario_sha256": file_sha256(scenario), "metrics_file": str(metrics),
                "metrics_sha256": file_sha256(metrics)}
    assert _resume_mismatch_reasons(existing, row, provenance, tmp_path) == []
    metrics.write_bytes(b"changed")
    assert "metrics_file hash mismatch" in _resume_mismatch_reasons(existing, row, provenance, tmp_path)


def test_formal_and_pilot_training_settings_are_isolated():
    protocol = load_protocol()
    pilot = protocol["training"]["pilot"]
    formal = protocol["training"]["formal"]
    assert pilot["epochs"]["FA-BM-VAE"] == 3
    assert pilot["eligible_for_paper"] is False
    assert formal["epochs"]["FA-BM-VAE"] == 80
    assert formal["epochs"]["Score-SDE"] == 200
    assert formal["score_sampling_steps"] == 32
    assert formal["negative_phase_backend"] == "internal"
    assert formal["sampler"] == "sa"


def test_daily_metrics_include_sample_based_ramp_and_reliability_metrics():
    rng = np.random.default_rng(7)
    target = rng.uniform(0.0, 1.0, size=(3, 24))
    scenarios = np.repeat(target.reshape(-1, 1), 100, axis=1)
    scenarios += rng.normal(0.0, 0.01, size=scenarios.shape)
    metrics = evaluate_model_daily(scenarios, target)
    for name in ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90",
                 "interval_width90", "MAQCE"):
        assert metrics[name].shape == (3,)
        assert np.isfinite(metrics[name]).all()
    assert not np.allclose(metrics["ramp_CRPS"], metrics["CRPS_raw"])
