import pytest
import numpy as np
import json

import GEFcom2014.models.QBM_VAE.validate_opsd_quality as validation

from GEFcom2014.models.QBM_VAE.validate_opsd_quality import (
    _candidate_metrics,
    _derived_seed,
    _evaluate_control,
    _check_thresholds,
    _legacy_derived_seed,
    _audit_normal_sample_files,
    audit_existing_validation_run,
    build_seed_plan,
    compare_validation_runs,
    validate_seed_plan,
)


def _entry_test_inputs(monkeypatch, history_seeds=None, audit_error=None):
    """Patch only data/model loading; the real run entry still performs the audit."""
    history_seeds = history_seeds or {}
    seen_versions = []
    sample_calls = []

    panels = {}
    for gain in (125, 150, 200):
        panels[gain] = [{
            "gain": gain,
            "instance_id": "condition_%03d" % index,
            "h": np.zeros(48, dtype=np.float64),
            "j": np.zeros((48, 48), dtype=np.float64),
            "reference": np.ones((100, 48), dtype=np.int8),
            "reference_path": "reference.npz",
            "reference_sha256": "reference",
            "source_problem_path": "problem.npz",
            "source_problem_sha256": "problem",
        } for index in range(10)]

    def fake_audit(root, version):
        seen_versions.append(version)
        if audit_error and version == audit_error[0]:
            raise audit_error[1]
        seeds = history_seeds.get(version, list(range(100000, 100300)))
        return {
            "version": version,
            "root": str(root),
            "normal_file_count": 300,
            "unique_seed_count": len(set(seeds)),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "seed_reuse_count": 300 - len(set(seeds)),
            "seed_mapping": [{"seed": seed} for seed in seeds],
            "artifact_class": "historical_run",
        }

    monkeypatch.setattr(validation, "_load_frozen_panel", lambda _: panels)
    monkeypatch.setattr(validation, "load_protocol",
                        lambda _: {"hardware": {"candidate_gains": [125, 150, 200]}})
    monkeypatch.setattr(validation, "_audit_reference_sample_files",
                        lambda *_: [{"seed": seed} for seed in range(200000, 200300)])
    monkeypatch.setattr(validation, "audit_existing_validation_run", fake_audit)
    monkeypatch.setattr(validation, "_sample_float",
                        lambda *args: sample_calls.append(args) or np.ones((100, 48), dtype=np.int8))
    return seen_versions, sample_calls


def _entry_test_files(tmp_path):
    threshold_source = tmp_path / "threshold.json"
    threshold_source.write_text(json.dumps({"suggested_thresholds": {
        "energy_wasserstein": 0.3,
        "effective_beta_error": 0.1,
        "edge_moment_mae": 0.1,
        "magnetization_mae": 0.1,
    }}), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    roots = {version: tmp_path / version for version in ("v1", "v2", "v3", "v4")}
    return threshold_source, protocol_path, roots


def test_legacy_formula_has_cross_batch_condition_collisions():
    old = [_legacy_derived_seed(303000 + batch, 125, condition)
           for batch in range(10) for condition in range(10)]
    assert len(old) > len(set(old))


def test_new_formula_allocates_300_unique_indexed_seeds():
    plan = build_seed_plan()
    actual = [row["random_seed"] for row in plan]
    assert len(plan) == 300
    assert len(set(actual)) == 300
    assert actual == list(range(303000, 303300))
    assert all(0 <= seed <= (2 ** 32) - 1 for seed in actual)


def test_recorded_seed_overlap_is_rejected_before_sampling():
    plan = build_seed_plan()
    recorded = {
        "threshold_reference": {"unique_seeds": [202725]},
        "validation_v1": {"unique_seeds": [plan[0]["random_seed"]]},
        "validation_v2": {"unique_seeds": []},
    }
    with pytest.raises(ValueError, match="overlaps recorded seeds"):
        validate_seed_plan(plan, recorded)


def test_run_entry_audits_v1_to_v4_and_rejects_overlap_before_sampling(tmp_path, monkeypatch):
    threshold, protocol, roots = _entry_test_files(tmp_path)
    seen, sample_calls = _entry_test_inputs(
        monkeypatch, history_seeds={"v3": [303000], "v4": [303001]}
    )
    with pytest.raises(ValueError, match="overlaps recorded seeds"):
        validation.run_independent_validation(
            tmp_path / "prep", protocol, threshold, tmp_path / "out",
            prior_validation_dirs=roots,
        )
    assert seen == ["v1", "v2", "v3", "v4"]
    assert sample_calls == []


def test_run_entry_rejects_missing_historical_directory_before_sampling(tmp_path, monkeypatch):
    threshold, protocol, roots = _entry_test_files(tmp_path)
    seen, sample_calls = _entry_test_inputs(monkeypatch)
    roots.pop("v4")
    with pytest.raises(ValueError, match="missing required validation directories"):
        validation.run_independent_validation(
            tmp_path / "prep", protocol, threshold, tmp_path / "out",
            prior_validation_dirs=roots,
        )
    assert seen == []
    assert sample_calls == []


@pytest.mark.parametrize("message", [
    "corrupt or incomplete normal sample",
    "normal sample has no recorded random seed",
])
def test_run_entry_rejects_corrupt_or_seedless_history_before_sampling(
    tmp_path, monkeypatch, message
):
    threshold, protocol, roots = _entry_test_files(tmp_path)
    seen, sample_calls = _entry_test_inputs(
        monkeypatch, audit_error=("v2", ValueError(message))
    )
    with pytest.raises(ValueError, match=message):
        validation.run_independent_validation(
            tmp_path / "prep", protocol, threshold, tmp_path / "out",
            prior_validation_dirs=roots,
        )
    assert seen == ["v1", "v2"]
    assert sample_calls == []


def _write_auditable_run(root, count=300, missing_seed_index=None, corrupt_index=None):
    for index in range(count):
        path = root / "normal_control" / ("gain_%d" % (125 + (index // 100) * 25)) \
            / ("batch_%02d" % ((index // 10) % 10))
        path.mkdir(parents=True, exist_ok=True)
        target = path / ("condition_%03d.npz" % (index % 10))
        if index == corrupt_index:
            target.write_bytes(b"not-an-npz")
            continue
        kwargs = {
            "samples": np.ones((100, 48), dtype=np.int8),
            "random_seed": 303000 + index,
            "gain": 125 + (index // 100) * 25,
            "gain_index": index // 100,
            "batch_index": (index // 10) % 10,
            "instance_id": "condition_%03d" % (index % 10),
            "reads": 100,
            "sweeps": 100,
        }
        if index == missing_seed_index:
            kwargs.pop("random_seed")
        np.savez_compressed(target, **kwargs)


def test_historical_missing_or_corrupt_normal_sample_fails_closed(tmp_path):
    missing = tmp_path / "missing"
    _write_auditable_run(missing, count=299)
    with pytest.raises(ValueError, match="exactly 300"):
        _audit_normal_sample_files(missing)

    corrupt = tmp_path / "corrupt"
    _write_auditable_run(corrupt, corrupt_index=7)
    with pytest.raises(ValueError, match="corrupt or incomplete"):
        _audit_normal_sample_files(corrupt)

    no_seed = tmp_path / "no_seed"
    _write_auditable_run(no_seed, missing_seed_index=7)
    with pytest.raises(ValueError, match="no recorded random seed"):
        _audit_normal_sample_files(no_seed)


def test_exact_reproduction_does_not_add_independent_sample_count(tmp_path):
    first_root = tmp_path / "v3"
    second_root = tmp_path / "v4"
    _write_auditable_run(first_root)
    _write_auditable_run(second_root)
    first = audit_existing_validation_run(first_root, "v3")
    second = audit_existing_validation_run(second_root, "v4")
    comparison = compare_validation_runs(first, second)
    assert comparison["exact_reproduction"] is True
    assert comparison["additional_independent_samples"] == 0


def test_threshold_candidates_are_read_only_and_four_metrics_are_required():
    thresholds = {
        "energy_wasserstein": 0.3,
        "effective_beta_error": 0.08,
        "edge_moment_mae": 0.12,
        "magnetization_mae": 0.12,
    }
    assert _check_thresholds(thresholds) == thresholds
    with pytest.raises(ValueError, match="missing"):
        _check_thresholds({"energy_wasserstein": 0.3})


def test_batch_quality_uses_four_metrics_and_keeps_unique_state_diagnostic():
    thresholds = {name: 0.2 for name in (
        "energy_wasserstein", "effective_beta_error", "edge_moment_mae", "magnetization_mae")}
    rows = []
    for gain in (125, 150, 200):
        rows.append({"control": "normal", "gain": gain, "batch_index": 0,
                     "random_seed": 303000, "n_conditions": 10,
                     "energy_wasserstein": 0.1, "effective_beta_error": 0.1,
                     "edge_moment_mae": 0.1, "magnetization_mae": 0.1,
                     "unique_state_fraction": 0.0})
    result = _evaluate_control(rows, thresholds)
    assert result["joint_pass_rate"] == 1.0
    assert result["metric_failure_counts"] == {name: 0 for name in thresholds}
