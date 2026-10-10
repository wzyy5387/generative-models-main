import subprocess
import sys
import json

import pytest
import hashlib

from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import (
    _load_completed,
    _query_existing_task,
    _recover_pending_intent,
    planned_tasks,
    select_pending_tasks,
    _validate_execution_limits,
)
from GEFcom2014.models.QBM_VAE.import_hardware_responses import strict_response_record_valid


def test_kaiwu_batch_default_dry_run_lists_frozen_budget():
    command = [sys.executable, "-m", "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling",
               "--stage", "vs", "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "tasks=30 reads=3000 dry_run=True" in result.stdout
    assert "matrix_sha256=" in result.stdout


def test_real_execution_checks_cpqc550_profile(tmp_path):
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(json.dumps({
        "platform_limits_checked": {
            "int8_coefficient_range": [-128, 127],
            "int8_range_verified": True,
            "machine_profile": {"name": "CPQC-550", "max_variables": 550,
                                "full_connectivity": True, "coupling_precision": "int8"},
        }
    }), encoding="utf-8")
    _validate_execution_limits([{"manifest": manifest}])


def test_resume_recognizes_completed_matrix_without_repeating_charge(tmp_path):
    path = tmp_path / "vs_batch_audit.json"
    path.write_text(json.dumps({
        "records": [{"completed": True, "instance_id": "vs_000", "matrix_sha256": "abc"}]
    }), encoding="utf-8")
    completed = _load_completed(tmp_path)
    assert ("vs_000", "abc") not in completed
    assert _load_completed(tmp_path) == completed


def test_test_stage_requires_verified_frozen_gain(tmp_path):
    with pytest.raises(RuntimeError, match="real_gain_selection"):
        planned_tasks("test", tmp_path, tmp_path)
    (tmp_path / "real_gain_selection.json").write_text(json.dumps({
        "hardware_claim": False,
        "physical_platform_used": False,
        "immutable": True,
        "selected_gain": 200,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="verified physical"):
        planned_tasks("test", tmp_path, tmp_path)


def test_limit_is_applied_after_completed_filter():
    tasks = [{"instance_id": "i%d" % i, "matrix_sha256": "h%d" % i} for i in range(3)]
    assert select_pending_tasks(tasks, set(), 1)[0]["instance_id"] == "i0"
    assert select_pending_tasks(tasks, {("i0", "h0")}, 1)[0]["instance_id"] == "i1"
    assert select_pending_tasks(tasks, {("i0", "h0"), ("i1", "h1")}, 1)[0]["instance_id"] == "i2"


def test_strict_response_rejects_missing_hash_reads_and_mock(tmp_path):
    response = tmp_path / "r.npz"
    npz_values = {
        "samples": __import__("numpy").ones((10, 49), dtype="int8"),
        "instance_id": "i", "task_id": "tid", "task_name": "tn",
        "matrix_sha256": "mh", "stage": "vs", "hardware_gain": 125,
        "requested_reads": 10, "returned_reads": 10,
    }
    __import__("numpy").savez_compressed(response, **npz_values)
    digest = hashlib.sha256(response.read_bytes()).hexdigest()
    record = {"completed": True, "instance_id": "i", "stage": "vs",
              "task_id": "tid", "task_name": "tn", "matrix_sha256": "mh",
              "response_file_sha256": digest, "requested_reads": 10,
              "returned_reads": 10, "response_path": str(response),
              "hardware_claim": True, "physical_platform_used": True,
              "gain": 125, "sdk_version": "1.0.7",
              "platform_backend": "kaiwu.cim.CIMOptimizer",
              "sdk_versions": {"importlib_metadata_version": "1.4.1",
                               "kaiwu_module_version": "1.0.7"}}
    assert strict_response_record_valid(record)
    for key in ("response_file_sha256", "response_path"):
        bad = dict(record); bad[key] = "missing"
        assert not strict_response_record_valid(bad)
    bad = dict(record); bad["returned_reads"] = 9
    assert not strict_response_record_valid(bad)
    bad = dict(record); bad["hardware_claim"] = False
    assert not strict_response_record_valid(bad)


def test_pending_intent_without_public_query_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling.official_task_query_capability",
        lambda: None,
    )
    with pytest.raises(RuntimeError, match="状态未知|manual platform"):
        _query_existing_task({"task_name": "t"}, {"instance_id": "i"})


def test_pending_intent_recovery_accepts_only_same_task_record(tmp_path, monkeypatch):
    import numpy as np
    response = tmp_path / "response.npz"
    np.savez_compressed(
        response, samples=np.ones((10, 49), dtype=np.int8), instance_id="i",
        task_id="tid", task_name="tn", matrix_sha256="mh", stage="vs",
        hardware_gain=125, requested_reads=10, returned_reads=10,
    )
    import hashlib
    record = {"completed": True, "instance_id": "i", "stage": "vs",
              "task_id": "tid", "task_name": "tn", "matrix_sha256": "mh",
              "gain": 125, "requested_reads": 10, "returned_reads": 10,
              "response_path": str(response),
              "response_file_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
              "hardware_claim": True, "physical_platform_used": True,
              "sdk_version": "1.0.7", "platform_backend": "kaiwu.cim.CIMOptimizer",
              "sdk_versions": {"importlib_metadata_version": "1.4.1", "kaiwu_module_version": "1.0.7"}}
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling._query_existing_task",
        lambda intent, task: {"audit": record},
    )
    recovered = _recover_pending_intent(
        {"task_name": "tn", "state": "pending"},
        {"stage": "vs", "instance_id": "i", "gain": 125,
         "matrix_sha256": "mh", "requested_reads": 10, "task_name": "tn"},
    )
    assert recovered["task_name"] == "tn"
