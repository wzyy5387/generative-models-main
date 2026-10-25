import subprocess
import sys
import json
from types import SimpleNamespace

import pytest
import hashlib

from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import (
    _load_completed,
    _load_pending_intents,
    _query_existing_task,
    _recover_pending_intent,
    planned_tasks,
    select_pending_tasks,
    _validate_execution_limits,
    _load_frozen_gain,
    run,
)
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import load_protocol
from GEFcom2014.models.QBM_VAE.import_hardware_responses import strict_response_record_valid


def test_kaiwu_batch_default_dry_run_lists_frozen_budget():
    command = [sys.executable, "-m", "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling",
               "--stage", "vs", "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "tasks=30 reads=3000 dry_run=True" in result.stdout
    assert '"hardware_claim": false' in result.stdout


def test_vs_execution_order_is_date_interleaved_and_bound_to_frozen_inputs(tmp_path):
    from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import (
        _vs_tasks, build_vs_execution_order, validate_vs_execution_order,
    )
    protocol = load_protocol("configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json")
    protocol["_path"] = "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json"
    prep = __import__("pathlib").Path("export/opsd_sa_negative_kaiwu_preparation_v8")
    tasks = _vs_tasks(prep, protocol)
    assert len(tasks) == 30
    assert [task["gain"] for task in tasks[:3]] == [125, 150, 200]
    assert [task["gain"] for task in tasks[3:6]] == [150, 200, 125]
    assert all(task["reads"] == 100 for task in tasks)
    payload = build_vs_execution_order(tasks, "a" * 64, "b" * 64)
    order_path = tmp_path / "vs_execution_order.json"
    order_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert validate_vs_execution_order(order_path, tasks, "a" * 64, "b" * 64) == payload
    payload["entries"][0]["gain"] = 200
    order_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match frozen"):
        validate_vs_execution_order(order_path, tasks, "a" * 64, "b" * 64)


def test_opsd_v5_stale_task_names_are_rejected_instead_of_rewritten():
    from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import _vs_tasks
    protocol = load_protocol("configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json")
    protocol["_path"] = "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json"
    with pytest.raises(FileNotFoundError, match="quality_reference_manifest.json"):
        _vs_tasks(__import__("pathlib").Path("export/opsd_sa_negative_kaiwu_preparation_v5"), protocol)


def test_opsd_v6_preparation_cannot_be_reused_under_new_protocol():
    from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import _vs_tasks
    protocol = load_protocol("configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json")
    protocol["_path"] = "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json"
    protocol["protocol_version"] = "fa-bm-vae-opsd-spqc-550-sa-negative-v2-quality-gate"
    with pytest.raises(ValueError, match="Quality reference manifest protocol_version mismatch"):
        _vs_tasks(__import__("pathlib").Path("export/opsd_sa_negative_kaiwu_preparation_v6"), protocol)


def test_canary_dry_run_is_one_frozen_gain_125_task():
    from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import _restrict_canary_tasks
    tasks = [{"stage": "vs", "instance_id": "fixed_vs", "gain": gain, "reads": 100}
             for gain in (125, 150, 200)]
    args = SimpleNamespace(stage="vs", canary=True, canary_instance_id="fixed_vs",
                           limit=1, execute=False, confirm_canary=False)
    selected = _restrict_canary_tasks(tasks, args, {"hardware": {"reads_per_instance": 100}})
    assert selected == [tasks[0]]


def test_canary_real_execution_requires_separate_approval():
    from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import _restrict_canary_tasks
    args = SimpleNamespace(stage="vs", canary=True, canary_instance_id="fixed_vs",
                           limit=1, execute=True, confirm_canary=False)
    with pytest.raises(RuntimeError, match="confirm-canary"):
        _restrict_canary_tasks(
            [{"stage": "vs", "instance_id": "fixed_vs", "gain": 125, "reads": 100}],
            args, {"hardware": {"reads_per_instance": 100}},
        )


@pytest.mark.parametrize("mode, message", [
    ("missing_batch_approval", "confirm-batch"),
    ("unapproved_quality", "not approved"),
    ("invalid_threshold", "finite and non-negative"),
    ("mock_policy", "test-only"),
    ("canary_flag_missing", "confirm-canary requires"),
    ("mixed_approval", "not both"),
])
def test_submission_scope_rejects_before_any_submit(tmp_path, monkeypatch, mode, message):
    from GEFcom2014.models.QBM_VAE import run_kaiwu_13k_sampling as driver
    from tests.test_kaiwu_13k_protocol import _approved_quality_policy
    policy = _approved_quality_policy()
    tasks = [{"stage": "vs", "instance_id": "i%d" % i, "gain": 125,
              "matrix_sha256": "h%d" % i, "reads": 100,
              "matrix": tmp_path / "matrix.npz"} for i in range(2)]
    monkeypatch.setattr(driver, "load_protocol", lambda _: {
        "protocol_version": "test-only", "gain_selection": {"quality_gate": policy}})
    monkeypatch.setattr(driver, "planned_tasks", lambda *a: tasks)
    monkeypatch.setattr(driver, "_validate_execution_limits", lambda _: None)
    calls = []
    monkeypatch.setattr(driver, "submit_one", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("FA_BM_VAE_RUN_REAL_KAIWU", "1")
    args = SimpleNamespace(stage="vs", protocol=tmp_path / "protocol.json",
                           preparation=tmp_path, output_dir=tmp_path / "responses",
                           checkpoint_dir=tmp_path / "checkpoints", resume=True,
                           limit=None, execute=True, confirm_real_submission=True,
                           confirm_batch=mode != "missing_batch_approval", project_no=None,
                           canary=False, confirm_canary=False)
    if mode == "unapproved_quality":
        policy["thresholds_approved"] = False
    elif mode == "invalid_threshold":
        policy["thresholds"]["energy_wasserstein"] = float("inf")
    elif mode == "mock_policy":
        policy["test_only"] = True
    elif mode == "canary_flag_missing":
        args.confirm_canary = True
    elif mode == "mixed_approval":
        args.canary = True
        args.confirm_canary = True
    with pytest.raises((RuntimeError, ValueError), match=message):
        run(args)
    assert calls == []
    assert not list(tmp_path.rglob("*.intent.json"))


def test_quality_approval_change_cannot_silently_rebind_frozen_protocol(tmp_path):
    from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import _validate_preparation_identity
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"thresholds_approved": False}), encoding="utf-8")
    manifest = tmp_path / "preparation_manifest.json"
    manifest.write_text(json.dumps({
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    }), encoding="utf-8")
    frozen_bytes = manifest.read_bytes()
    protocol = {"_path": str(protocol_path)}
    _validate_preparation_identity(tmp_path, protocol)
    protocol_path.write_text(json.dumps({"thresholds_approved": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different protocol JSON"):
        _validate_preparation_identity(tmp_path, protocol)
    assert manifest.read_bytes() == frozen_bytes


def test_spqc550_without_project_no_reaches_mocked_submit(tmp_path, monkeypatch):
    calls = []
    task = {"stage": "vs", "instance_id": "vs_000", "gain": 125,
            "matrix_sha256": "frozen-hash", "reads": 100, "matrix": tmp_path / "matrix.npz"}
    monkeypatch.delenv("KAIWU_PROJECT_NO", raising=False)
    monkeypatch.delenv("FA_BM_VAE_RUN_REAL_KAIWU", raising=False)
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling.load_protocol",
        lambda _: {"protocol_version": "test-protocol"},
    )
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling.planned_tasks",
        lambda *args: [task],
    )
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling._validate_execution_limits",
        lambda tasks: None,
    )

    def fake_submit(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("mock-only submit reached")

    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling.submit_one",
        fake_submit,
    )
    args = SimpleNamespace(stage="vs", protocol=tmp_path / "protocol.json",
                           preparation=tmp_path, output_dir=tmp_path / "responses",
                           checkpoint_dir=tmp_path / "checkpoints", resume=True,
                           limit=1, execute=True, confirm_real_submission=False,
                           project_no=None, canary=True, confirm_canary=True,
                           canary_instance_id="vs_000")
    with pytest.raises(RuntimeError, match="Real batch requires"):
        run(args)
    assert not calls
    args.confirm_real_submission = True
    monkeypatch.setenv("FA_BM_VAE_RUN_REAL_KAIWU", "1")
    with pytest.raises(RuntimeError, match="mock-only submit reached"):
        run(args)
    assert calls[0]["project_no"] is None
    assert list((tmp_path / "checkpoints").glob("*.intent.json"))


@pytest.mark.parametrize("stage", ["vs", "test"])
@pytest.mark.parametrize("failure", ["short_reads", "bad_hash", "wrong_task",
                                    "missing_metadata", "second_response", None])
def test_batch_stops_on_invalid_response_without_retry(tmp_path, monkeypatch, stage, failure):
    """Exercise the real validator and batch loop, with submission fully mocked."""
    import numpy as np
    from GEFcom2014.models.QBM_VAE import run_kaiwu_13k_sampling as driver
    from tests.test_kaiwu_13k_protocol import _approved_quality_policy

    tasks = [{"stage": stage, "instance_id": "i%d" % i, "gain": 125,
              "matrix_sha256": hashlib.sha256(("mh%d" % i).encode()).hexdigest(),
              "reads": 100, "requested_reads": 100,
              "matrix": tmp_path / ("matrix%d.npz" % i)} for i in range(3)]
    quality_policy = _approved_quality_policy()
    monkeypatch.setattr(driver, "load_protocol", lambda _: {
        "protocol_version": "test-only", "gain_selection": {"quality_gate": quality_policy}})
    monkeypatch.setattr(driver, "planned_tasks", lambda *args: tasks)
    monkeypatch.setattr(driver, "_validate_execution_limits", lambda _: None)
    monkeypatch.setattr(driver, "official_task_query_capability", lambda: None)
    real_validator = driver.strict_response_record_valid
    def fixture_validator(record, expected=None, require_physical=True):
        if record.get("platform_backend") == "offline-mock" and not record.get("prediction_validation_error"):
            record["completed"] = True
        return real_validator(record, expected, require_physical=False)
    monkeypatch.setattr(driver, "strict_response_record_valid", fixture_validator)
    monkeypatch.setenv("FA_BM_VAE_RUN_REAL_KAIWU", "1")
    monkeypatch.delenv("KAIWU_PROJECT_NO", raising=False)
    calls, raw_hashes = [], {}
    fail_index = 1 if failure == "second_response" else 0

    def fake_submit(matrix, instance_id, output_dir, reads, **kwargs):
        index = len(calls)
        calls.append(instance_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        response = output_dir / (instance_id + ".npz")
        fields = {"instance_id": instance_id, "task_id": "mock-tid-%d" % index,
                  "task_name": kwargs["task_name"], "matrix_sha256": tasks[index]["matrix_sha256"],
                  "stage": stage, "hardware_gain": 125,
                  "requested_reads": reads, "returned_reads": reads}
        np.savez_compressed(response, samples=np.ones((reads, 49), dtype=np.int8), **fields)
        digest = hashlib.sha256(response.read_bytes()).hexdigest()
        raw_hashes[response] = digest
        # Synthetic fixture is explicitly non-physical and never hardware evidence.
        record = {**fields, "gain": 125, "response_path": str(response),
                  "response_file_sha256": digest, "completed": True,
                  "hardware_claim": False, "physical_platform_used": False,
                  "hardware_metadata_complete": True, "sdk_version": "mock",
                  "platform_backend": "offline-mock",
                  "sdk_versions": {"importlib_metadata_version": "mock", "kaiwu_module_version": "mock"},
                  "submitted_at": "2026-01-01T00:00:00+00:00",
                  "completed_at": "2026-01-01T00:01:00+00:00"}
        if failure not in (None, "prediction_failure") and index == fail_index:
            if failure == "short_reads":
                record["returned_reads"] = reads - 1
            elif failure == "wrong_task":
                record["task_name"] = "another-task"
            elif failure == "missing_metadata":
                record.pop("task_id")
            else:
                record["response_file_sha256"] = "incorrect-hash"
        audit = output_dir / (instance_id + ".json")
        audit.write_text(json.dumps(record), encoding="utf-8")
        return response, audit, record

    monkeypatch.setattr(driver, "submit_one", fake_submit)
    args = SimpleNamespace(stage=stage, protocol=tmp_path / "protocol.json",
                           preparation=tmp_path, output_dir=tmp_path / "responses",
                           checkpoint_dir=tmp_path / "checkpoints", resume=True,
                           limit=3, execute=True, confirm_real_submission=True,
                           confirm_batch=True, confirm_test=True, project_no=None)
    if failure is None:
        result = run(args)
        assert result["completed"] == 3 and result["pending"] == 0
        assert len(calls) == 3
        assert run(args)["completed"] == 3
        assert len(calls) == 3  # Successful responses are not submitted twice either.
        assert not list(args.output_dir.rglob("*.validation_failure.json"))
        assert all(payload[1]["state"] == "completed"
                   for payload in _load_pending_intents(args.checkpoint_dir).values())
    else:
        with pytest.raises(RuntimeError, match="response failed (strict|full frozen VS) validation"):
            run(args)
        assert len(calls) == fail_index + 1
        failures = list(args.output_dir.rglob("*.validation_failure.json"))
        assert len(failures) == 1
        audit = json.loads(failures[0].read_text(encoding="utf-8"))
        assert audit["status"] == "stopped_invalid_response"
        assert audit["hardware_claim"] is False
        assert audit["reason"] and audit["expected"]["instance_id"] == tasks[fail_index]["instance_id"]
        assert audit["response_record"]["response_path"]
        completed = _load_completed(args.output_dir, tasks)
        assert len(completed) == fail_index
        # Pending intent must block both resume and accidental non-resume retries.
        for resume in (True, False):
            args.resume = resume
            with pytest.raises(RuntimeError, match="Prior response validation failure|manual platform|validation failure"):
                run(args)
            assert len(calls) == fail_index + 1
        assert len(list(args.checkpoint_dir.glob("*.intent.json"))) == fail_index + 1
    for path, digest in raw_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_nonfinite_prediction_stops_batch_and_blocks_resume(tmp_path, monkeypatch):
    from tests.test_kaiwu_response_recovery_validation import (
        test_new_prediction_failure_is_recorded_before_next_submit_and_restart_stops,
    )
    test_new_prediction_failure_is_recorded_before_next_submit_and_restart_stops(tmp_path, monkeypatch)


@pytest.mark.parametrize("contents", ["{truncated", json.dumps({"task_name": "x"})])
def test_corrupt_or_incomplete_pending_intent_fails_closed(tmp_path, contents):
    path = tmp_path / "x.intent.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError, match="intent"):
        _load_pending_intents(tmp_path)


def test_pending_intent_filename_and_task_identity_are_validated(tmp_path):
    payload = {"task_name": "another", "stage": "vs", "instance_id": "i",
               "gain": 125, "matrix_sha256": "a" * 64,
               "requested_reads": 100, "created_at": "2026-10-21T00:00:00Z",
               "state": "pending"}
    path = tmp_path / "expected.intent.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity or state"):
        _load_pending_intents(tmp_path)


def test_real_execution_checks_spqc550_profile(tmp_path):
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(json.dumps({
        "platform_limits_checked": {
            "int8_coefficient_range": [-128, 127],
            "int8_range_verified": True,
            "machine_profile": {"name": "SPQC-550", "max_variables": 550,
                                "full_connectivity": True, "coupling_precision": "int8"},
        }
    }), encoding="utf-8")
    _validate_execution_limits([{"manifest": manifest}])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["platform_limits_checked"]["machine_profile"]["name"] = "CPQC-550"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SPQC-550"):
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


def test_zero_limit_does_not_report_unsubmitted_tasks_as_complete(tmp_path, monkeypatch):
    task = {"stage": "vs", "instance_id": "vs_000", "gain": 125,
            "matrix_sha256": "frozen-hash", "reads": 100, "matrix": tmp_path / "matrix.npz"}
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling.load_protocol",
        lambda _: {"protocol_version": "test-protocol"},
    )
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling._validate_preparation_identity",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling.planned_tasks",
        lambda *args: [task],
    )
    result = run(SimpleNamespace(
        stage="vs", protocol=tmp_path / "protocol.json", preparation=tmp_path,
        output_dir=tmp_path / "responses", checkpoint_dir=tmp_path / "checkpoints",
        resume=True, limit=0, execute=True, confirm_real_submission=False, project_no=None,
    ))
    assert result["completed"] == 0
    assert result["pending"] == 1
    assert result["hardware_claim"] is False


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
              "submitted_at": "2026-01-01T00:00:00+00:00",
              "completed_at": "2026-01-01T00:01:00+00:00",
              "hardware_metadata_complete": True,
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
              "submitted_at": "2026-01-01T00:00:00+00:00",
              "completed_at": "2026-01-01T00:01:00+00:00",
              "hardware_metadata_complete": True,
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


def test_tampered_gain_freeze_is_rejected_before_test_planning(tmp_path, monkeypatch):
    from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import select_vs_gain
    from tests.test_kaiwu_13k_protocol import _complete_vs_records

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    prep_manifest = tmp_path / "preparation_manifest.json"
    prep_manifest.write_text(json.dumps({"model_checkpoint_sha256": "model-hash"}), encoding="utf-8")
    protocol = {
        "_path": str(protocol_path),
        "hardware": {"candidate_gains": [125, 150, 200]},
        "vs": {"instances": 2},
        "gain_selection": {"quality_gate": {
            "status": "defined", "rule_version": "test-v1",
            "thresholds_approved": True,
            "thresholds": {"energy_wasserstein": 1.0,
                            "effective_beta_error": 1.0,
                            "edge_moment_mae": 1.0,
                            "magnetization_mae": 1.0,
                            "unique_state_fraction_min": 0.0},
        }},
    }
    records = _complete_vs_records(tmp_path)
    identity = {
        "protocol_sha256": __import__("hashlib").sha256(protocol_path.read_bytes()).hexdigest(),
        "model_checkpoint_sha256": "model-hash",
        "preparation_manifest_sha256": __import__("hashlib").sha256(prep_manifest.read_bytes()).hexdigest(),
    }
    for row in records:
        row.update(identity)
    tasks = [{"stage": "vs", "instance_id": row["instance_id"], "gain": row["gain"],
              "matrix_sha256": row["matrix_sha256"], "requested_reads": row["requested_reads"],
              "task_name": row["task_name"], **identity} for row in records]
    monkeypatch.setattr(
        "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling._vs_tasks",
        lambda preparation, protocol: tasks,
    )
    freeze = select_vs_gain(
        records, tmp_path / "real_gain_selection.json",
        required_gains=(125, 150, 200), expected_instances_per_gain=2,
        expected_instance_ids=("vs_0", "vs_1"), expected_task_map={
            (float(row["gain"]), row["instance_id"]): {
                "stage": "vs", "instance_id": row["instance_id"], "gain": row["gain"],
                "matrix_sha256": row["matrix_sha256"], "requested_reads": row["requested_reads"],
                "task_name": row["task_name"], **identity,
            } for row in records}, expected_identity=identity,
        quality_policy=protocol["gain_selection"]["quality_gate"],
    )
    path = tmp_path / "real_gain_selection.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    assert _load_frozen_gain(tmp_path, protocol)["selected_gain"] == 125.0
    freeze["selected_gain"] = 150.0
    path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        _load_frozen_gain(tmp_path, protocol)
