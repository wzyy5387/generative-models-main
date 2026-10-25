"""Offline, temporary-directory acceptance for paid-response recovery.

The fixtures use the frozen VS inputs and a synthetic 100-read NPZ only.  The
SDK submit seam is replaced; all response identity, latent metric, and decoder
validation functions remain live (the physical-claim requirement is disabled
only for the explicitly synthetic fixture).
"""

import hashlib
import json
import atexit
import tempfile
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE import run_kaiwu_13k_sampling as driver
from GEFcom2014.models.QBM_VAE import opsd_vs_selection
from GEFcom2014.models.QBM_VAE.import_hardware_responses import strict_response_record_valid
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json"
PREPARATION = ROOT / "export/opsd_sa_negative_kaiwu_preparation_v8"
_CHECKPOINT_TEMP_DIRS = []


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in recovery tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@atexit.register
def _cleanup_short_checkpoint_dirs():
    for directory in _CHECKPOINT_TEMP_DIRS:
        directory.cleanup()


def _h(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen_protocol():
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["_path"] = str(PROTOCOL_PATH.resolve())
    return protocol


def _harness(tmp_path, monkeypatch, count=2, no_id_instances=()):
    protocol = _frozen_protocol()
    tasks = driver._vs_tasks(PREPARATION, protocol)[:count]
    # This test does not exercise the separate human-frozen order artifact.
    tasks = [{key: value for key, value in task.items() if key != "vs_task_order"}
             for task in tasks]
    monkeypatch.setattr(driver, "load_protocol", lambda _: dict(protocol))
    monkeypatch.setattr(driver, "planned_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(driver, "_validate_execution_limits", lambda _: None)
    monkeypatch.setattr(driver, "validate_quality_policy", lambda *_: None)

    # Retain the real strict checks for the synthetic fixture, relaxing only
    # the physical-platform flags. No credential or SDK call is fabricated.
    def synthetic_only_validator(record, expected=None, require_physical=True):
        physical = (record.get("hardware_claim") is True
                    and record.get("physical_platform_used") is True
                    and record.get("platform_backend") == "kaiwu.cim.CIMOptimizer")
        return strict_response_record_valid(record, expected, require_physical=physical)

    monkeypatch.setattr(driver, "strict_response_record_valid", synthetic_only_validator)
    import GEFcom2014.models.QBM_VAE.import_hardware_responses as response_checks
    monkeypatch.setattr(response_checks, "strict_response_record_valid", synthetic_only_validator)

    calls = []
    no_id_instances = set(no_id_instances)

    def fake_submit(matrix, instance_id, response_dir, reads, **kwargs):
        task = next(item for item in tasks if item["task_name"] == kwargs["task_name"])
        calls.append(instance_id)
        response_dir = Path(response_dir)
        response_dir.mkdir(parents=True, exist_ok=True)
        response_path = response_dir / (instance_id + ".npz")
        if response_path.exists():
            raise AssertionError("synthetic submit boundary must never overwrite an existing response")
        task_id = ("" if instance_id in no_id_instances or task["task_name"] in no_id_instances
                   else "synthetic-task-" + task["task_name"])
        timestamp = "2026-01-01T00:00:00+00:00"
        fields = {
            "samples": np.ones((100, 49), dtype=np.int8),
            "instance_id": instance_id, "task_id": task_id,
            "task_name": kwargs["task_name"],
            "matrix_sha256": task["matrix_sha256"], "stage": "vs",
            "hardware_gain": float(task["gain"]), "requested_reads": int(reads),
            "returned_reads": int(reads), "submitted_at": timestamp,
            "completed_at": "2026-01-01T00:01:00+00:00",
            "sdk_version": "synthetic", "kaiwu_distribution_version": "synthetic",
            "kaiwu_module_version": "synthetic",
        }
        np.savez_compressed(response_path, **fields)
        record = {
            **{key: task[key] for key in (
                "protocol_version", "stage", "instance_id", "gain", "matrix_sha256",
                "requested_reads", "task_name", "protocol_sha256",
                "model_checkpoint_sha256", "preparation_manifest_sha256",
                "hardware_matrix_path", "hardware_matrix_file_sha256",
                "source_problem_path", "source_problem_sha256",
                "reference_sample_path", "reference_sample_sha256", "reference_sampler")},
            "task_id": task_id, "task_name": kwargs["task_name"],
            "response_path": str(response_path.resolve()),
            "response_file_sha256": _h(response_path),
        "completed": True, "hardware_claim": False,
            "physical_platform_used": False, "hardware_metadata_complete": True,
            "sdk_version": "synthetic",
            # The missing-ID branch is selected using the installed SDK's
            # known adapter name; every artifact remains synthetic and temp-only.
            "platform_backend": ("kaiwu.cim.CIMOptimizer" if not task_id else "offline-synthetic"),
            "sdk_versions": {"importlib_metadata_version": "synthetic",
                              "kaiwu_module_version": "synthetic"},
            "submitted_at": timestamp, "completed_at": "2026-01-01T00:01:00+00:00",
            "requested_reads": int(reads), "returned_reads": int(reads),
            "gain": float(task["gain"]),
            "artifact_role": "temporary_offline_synthetic_response",
        }
        audit_path = response_dir / (instance_id + ".json")
        audit_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["audit_path"] = str(audit_path.resolve())
        return response_path, audit_path, record

    monkeypatch.setattr(driver, "submit_one", fake_submit)
    monkeypatch.setenv("FA_BM_VAE_RUN_REAL_KAIWU", "1")
    # Frozen task names are intentionally long. Put intents directly under the
    # short OS temp root so Windows path length does not distort the test.
    checkpoint_temp = tempfile.TemporaryDirectory(prefix="kvr_")
    _CHECKPOINT_TEMP_DIRS.append(checkpoint_temp)
    args = SimpleNamespace(
        stage="vs", protocol=PROTOCOL_PATH, preparation=PREPARATION,
        output_dir=tmp_path / "responses", checkpoint_dir=Path(checkpoint_temp.name),
        resume=True, limit=count, execute=True, confirm_real_submission=True,
        confirm_batch=True, confirm_canary=False, canary=False, project_no=None,
    )
    return tasks, calls, args


def _platform_receipt_for_waiting_response(tmp_path, tasks, task_name):
    from GEFcom2014.models.QBM_VAE.kaiwu_platform_receipt import create_platform_verification
    task = next(item for item in tasks if item["task_name"] == task_name)
    instance_id = task["instance_id"]
    response = tmp_path / "responses" / ("gain_%d" % task["gain"]) / (instance_id + ".npz")
    evidence_dir = tmp_path / ("synthetic_platform_gain_%d" % task["gain"]) / instance_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    platform_record = evidence_dir / "synthetic_platform_record.json"
    platform_record.write_text(json.dumps({
        "job_id": "synthetic-platform-job-" + instance_id,
        "task_name": task["task_name"], "matrix_sha256": task["matrix_sha256"],
        "status": "completed", "requested_reads": 100, "returned_reads": 100,
        "submitted_at": "2026-01-01T00:00:05Z",
        "completed_at": "2026-01-01T00:01:05Z",
        "platform_backend": "kaiwu.cim.CIMOptimizer",
    }), encoding="utf-8")
    evidence = evidence_dir / "synthetic_evidence.txt"
    evidence.write_text("synthetic offline fixture only", encoding="utf-8")
    receipt = response.with_name(instance_id + ".platform_verification.json")
    create_platform_verification(response, task["matrix"], platform_record,
                                 evidence, "synthetic-test-reviewer", receipt)


def _run_failure_restart(args, calls, expected_count):
    with pytest.raises(RuntimeError):
        driver.run(args)
    assert len(calls) == expected_count
    states = [json.loads(path.read_text(encoding="utf-8"))["state"]
              for path in args.checkpoint_dir.glob("*.intent.json")]
    assert "validation_failed" in states
    with pytest.raises(RuntimeError, match="Prior response validation failure|validation failure"):
        driver.run(args)
    assert len(calls) == expected_count


def _inject_real_prediction_path_failure(monkeypatch, message, error=ValueError):
    original = opsd_vs_selection.recompute_vs_predictions
    seen = []

    def tracked(records, *args, **kwargs):
        seen.extend(dict(record) for record in records)
        return original(records, *args, **kwargs)

    monkeypatch.setattr(opsd_vs_selection, "recompute_vs_predictions", tracked)
    monkeypatch.setattr(
        opsd_vs_selection, "decode_platform_spins",
        lambda *a, **k: (_ for _ in ()).throw(error(message)))
    return seen


def _assert_validation_received_100_read_response(seen, expected_instance):
    assert any(record.get("instance_id") == expected_instance for record in seen)
    record = next(record for record in seen if record.get("instance_id") == expected_instance)
    with np.load(record["response_path"], allow_pickle=False) as response:
        assert response["samples"].shape == (100, 49)
        assert np.isin(response["samples"], (-1, 1)).all()


def test_new_valid_response_is_experiment_validated_then_batch_continues(tmp_path, monkeypatch):
    from GEFcom2014.models.QBM_VAE import kaiwu_13k_protocol
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=3)
    latent_by_key = {}
    original = kaiwu_13k_protocol._recompute_response_metrics

    def capture(record, **kwargs):
        metrics = original(record, **kwargs)
        latent_by_key[(float(record["gain"]), record["instance_id"])] = metrics
        return metrics

    monkeypatch.setattr(kaiwu_13k_protocol, "_recompute_response_metrics", capture)
    result = driver.run(args)
    assert result["completed"] == 3
    assert calls == [task["instance_id"] for task in tasks]
    assert all(json.loads(path.read_text(encoding="utf-8")).get("state") == "completed"
               for path in args.checkpoint_dir.glob("*.intent.json"))
    batch = json.loads((args.output_dir / "vs_batch_audit.json").read_text(encoding="utf-8"))
    assert all(row["experiment_validation"]["status"] == "passed" for row in batch["records"])
    assert len({row["instance_id"] for row in batch["records"]}) == 1
    assert {row["gain"] for row in batch["records"]} == {125, 150, 200}
    for row in batch["records"]:
        validation = row["experiment_validation"]
        prediction = validation["prediction_metrics_recomputed"]
        assert (prediction["gain"], prediction["instance_id"]) == (row["gain"], row["instance_id"])
        assert validation["latent_metrics_recomputed"] == latent_by_key[(row["gain"], row["instance_id"])]


def test_new_prediction_failure_is_recorded_before_next_submit_and_restart_stops(tmp_path, monkeypatch):
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2)
    seen = _inject_real_prediction_path_failure(monkeypatch, "synthetic decoder failure")
    _run_failure_restart(args, calls, 1)
    _assert_validation_received_100_read_response(seen, tasks[0]["instance_id"])
    marker = next(args.output_dir.rglob("*.validation_failure.json"))
    assert "synthetic decoder failure" in marker.read_text(encoding="utf-8")
    # The call log records only the first paid-boundary simulation; therefore
    # the failure occurred before the second task's submit_one call.


def test_missing_task_id_pauses_then_receipt_revalidates_and_only_submits_remaining(tmp_path, monkeypatch):
    protocol = _frozen_protocol()
    tasks0 = driver._vs_tasks(PREPARATION, protocol)[:2]
    first = tasks0[0]["task_name"]
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2,
                                  no_id_instances=(first,))
    with pytest.raises(RuntimeError, match="Samples were saved"):
        driver.run(args)
    assert calls == [tasks[0]["instance_id"]]
    assert next(iter(driver._load_pending_intents(args.checkpoint_dir).values()))[1]["state"] == "pending"
    waiting = next(args.output_dir.rglob("*.awaiting_platform_verification.json"))
    assert json.loads(waiting.read_text(encoding="utf-8"))["status"] == "awaiting_platform_verification"
    _platform_receipt_for_waiting_response(tmp_path, tasks, first)
    # Receipt promotion is not enough: the live latent and prediction checks
    # execute before the remaining task reaches the submit boundary.
    result = driver.run(args)
    assert result["completed"] == 2
    assert calls == [tasks[0]["instance_id"], tasks[1]["instance_id"]]
    assert all(payload[1]["state"] == "completed"
               for payload in driver._load_pending_intents(args.checkpoint_dir).values())


def test_receipt_recovery_with_invalid_prediction_stops_before_remaining_submit(tmp_path, monkeypatch):
    protocol = _frozen_protocol()
    tasks0 = driver._vs_tasks(PREPARATION, protocol)[:2]
    first = tasks0[0]["task_name"]
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2,
                                  no_id_instances=(first,))
    with pytest.raises(RuntimeError, match="Samples were saved"):
        driver.run(args)
    _platform_receipt_for_waiting_response(tmp_path, tasks, first)
    seen = _inject_real_prediction_path_failure(monkeypatch, "synthetic recovered decode failure")
    _run_failure_restart(args, calls, 1)
    _assert_validation_received_100_read_response(seen, tasks[0]["instance_id"])


def test_all_responses_present_still_decode_before_complete_success(tmp_path, monkeypatch):
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=1)
    assert driver.run(args)["completed"] == 1
    assert len(calls) == 1
    seen = _inject_real_prediction_path_failure(monkeypatch, "synthetic existing decode failure")
    with pytest.raises(RuntimeError, match="before submission"):
        driver.run(args)
    assert len(calls) == 1
    _assert_validation_received_100_read_response(seen, tasks[0]["instance_id"])
    assert next(args.output_dir.rglob("*.validation_failure.json"))


def test_completed_intent_without_response_is_never_retried(tmp_path, monkeypatch):
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=1)
    task = tasks[0]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    intent = {"task_name": task["task_name"], "stage": "vs",
              "instance_id": task["instance_id"], "gain": task["gain"],
              "matrix_sha256": task["matrix_sha256"], "requested_reads": 100,
              "created_at": "2026-10-21T00:00:00Z", "state": "completed"}
    (args.checkpoint_dir / (task["task_name"] + ".intent.json")).write_text(
        json.dumps(intent), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Completed task intent has no valid response"):
        driver.run(args)
    assert calls == []


@pytest.mark.parametrize("tamper_field", ["response_file_sha256", "source_problem_sha256", "matrix_sha256"])
def test_existing_response_or_frozen_input_mismatch_stops_before_next_submit(
        tmp_path, monkeypatch, tamper_field):
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2)
    args.limit = 1
    assert driver.run(args)["completed"] == 1
    assert len(calls) == 1
    task = tasks[0]
    audit_path = args.output_dir / ("gain_%d" % task["gain"]) / (task["instance_id"] + ".json")
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    record[tamper_field] = "0" * 64
    audit_path.write_text(json.dumps(record), encoding="utf-8")
    args.limit = 2
    with pytest.raises(RuntimeError, match="before submission"):
        driver.run(args)
    assert len(calls) == 1
    assert next(args.output_dir.rglob("*.validation_failure.json"))


def test_platform_receipt_mismatch_stops_before_next_submit_and_preserves_raw(tmp_path, monkeypatch):
    protocol = _frozen_protocol()
    first_task = driver._vs_tasks(PREPARATION, protocol)[0]
    task_name = first_task["task_name"]
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2,
                                  no_id_instances=(task_name,))
    with pytest.raises(RuntimeError, match="Samples were saved"):
        driver.run(args)
    raw = args.output_dir / ("gain_%d" % tasks[0]["gain"]) / (tasks[0]["instance_id"] + ".npz")
    original_hash = _h(raw)
    _platform_receipt_for_waiting_response(tmp_path, tasks, task_name)
    receipt = raw.with_name(tasks[0]["instance_id"] + ".platform_verification.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["task_id"] = "different-synthetic-platform-job"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        driver.run(args)
    assert calls == [tasks[0]["instance_id"]]
    assert _h(raw) == original_hash


@pytest.mark.parametrize("interruption", ["after_receipt_before_validation", "after_validation_before_intent"])
def test_restart_after_interruption_never_repeats_submit(tmp_path, monkeypatch, interruption):
    original_decoder = opsd_vs_selection.decode_platform_spins
    if interruption == "after_receipt_before_validation":
        protocol = _frozen_protocol()
        first = driver._vs_tasks(PREPARATION, protocol)[0]["task_name"]
        tasks, calls, args = _harness(tmp_path, monkeypatch, count=1,
                                      no_id_instances=(first,))
        with pytest.raises(RuntimeError, match="Samples were saved"):
            driver.run(args)
        _platform_receipt_for_waiting_response(tmp_path, tasks, first)
        seen = _inject_real_prediction_path_failure(
            monkeypatch, "synthetic interruption", error=KeyboardInterrupt)
        with pytest.raises(KeyboardInterrupt):
            driver.run(args)
        _assert_validation_received_100_read_response(seen, tasks[0]["instance_id"])
        assert next(iter(driver._load_pending_intents(args.checkpoint_dir).values()))[1]["state"] == "pending"
    else:
        tasks, calls, args = _harness(tmp_path, monkeypatch, count=1)
        mark = driver._mark_intent_completed
        monkeypatch.setattr(driver, "_mark_intent_completed",
                            lambda *a: (_ for _ in ()).throw(KeyboardInterrupt("synthetic interruption")))
        with pytest.raises(KeyboardInterrupt):
            driver.run(args)
        monkeypatch.setattr(driver, "_mark_intent_completed", mark)
        assert next(iter(driver._load_pending_intents(args.checkpoint_dir).values()))[1]["state"] == "pending"
    assert len(calls) == 1
    monkeypatch.setattr(opsd_vs_selection, "decode_platform_spins", original_decoder)
    result = driver.run(args)
    assert result["completed"] == 1
    assert len(calls) == 1
    assert next(iter(driver._load_pending_intents(args.checkpoint_dir).values()))[1]["state"] == "completed"


@pytest.mark.parametrize("interruption", [None, "after_submit", "before_wait_marker"])
def test_actual_submit_audit_survives_receipt_pause_and_interruption(tmp_path, monkeypatch, interruption):
    """Use production NPZ/audit writes; replace only the hardware client."""
    from GEFcom2014.models.QBM_VAE import submit_kaiwu_sampling as submission
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2)

    class OfflineClient:
        def __init__(self, **kwargs):
            self._kaiwu = SimpleNamespace(
                __version__="synthetic",
                common=SimpleNamespace(CheckpointManager=SimpleNamespace(save_dir="")))

        def sample_hardware_matrix(self, matrix, num_reads, task_name):
            return {"samples": np.ones((num_reads, matrix.shape[0]), dtype=np.int8),
                    "metadata": {"task_name": task_name, "backend": "kaiwu_cim_sampling"}}

    monkeypatch.setattr(submission, "KaiwuClient", OfflineClient)

    def actual_submit(*args, **kwargs):
        calls.append(kwargs["task_name"])
        result = submission.submit_one(*args, **kwargs)
        if interruption == "after_submit" and len(calls) == 1:
            raise KeyboardInterrupt("after production audit was saved")
        return result

    monkeypatch.setattr(driver, "submit_one", actual_submit)
    original_atomic = driver._atomic_json

    def interrupted_marker(path, payload):
        if str(path).endswith(".awaiting_platform_verification.json"):
            raise KeyboardInterrupt("before waiting marker was saved")
        return original_atomic(path, payload)

    if interruption == "before_wait_marker":
        monkeypatch.setattr(driver, "_atomic_json", interrupted_marker)
    with pytest.raises(KeyboardInterrupt if interruption else RuntimeError):
        driver.run(args)
    monkeypatch.setattr(driver, "_atomic_json", original_atomic)
    assert calls == [tasks[0]["task_name"]]
    first = tasks[0]
    raw = args.output_dir / ("gain_%d" % first["gain"]) / (first["instance_id"] + ".npz")
    original_hash = _h(raw)
    base_audit = json.loads(raw.with_suffix(".json").read_text(encoding="utf-8"))
    for field in submission.FROZEN_TASK_AUDIT_FIELDS:
        assert base_audit[field] == first[field]
    # Without a receipt, restarting must pause before another SDK call.
    with pytest.raises(RuntimeError, match="awaiting platform identity"):
        driver.run(args)
    assert calls == [first["task_name"]]

    _platform_receipt_for_waiting_response(tmp_path, tasks, first["task_name"])
    with pytest.raises(RuntimeError, match="Samples were saved"):
        driver.run(args)
    assert calls == [task["task_name"] for task in tasks]
    assert _h(raw) == original_hash
    _platform_receipt_for_waiting_response(tmp_path, tasks, tasks[1]["task_name"])
    result = driver.run(args)
    assert result["completed"] == 2
    assert calls == [task["task_name"] for task in tasks]
    assert all(payload[1]["state"] == "completed"
               for payload in driver._load_pending_intents(args.checkpoint_dir).values())


@pytest.mark.parametrize("bad_predictions", ["missing_gain", "duplicate_gain", "unexpected_gain"])
def test_prediction_results_must_match_each_gain_once(tmp_path, monkeypatch, bad_predictions):
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2)
    assert driver.run(args)["completed"] == 2
    batch = json.loads((args.output_dir / "vs_batch_audit.json").read_text(encoding="utf-8"))
    predictions = [row["experiment_validation"]["prediction_metrics_recomputed"] for row in batch["records"]]
    if bad_predictions == "missing_gain":
        predictions.pop()
    elif bad_predictions == "duplicate_gain":
        predictions[1] = dict(predictions[0])
    else:
        predictions[1] = dict(predictions[1], gain=999)
    monkeypatch.setattr(opsd_vs_selection, "recompute_vs_predictions", lambda *a, **kw: predictions)
    with pytest.raises(RuntimeError, match="before submission"):
        driver.run(args)
    assert len(calls) == 2
    marker = json.loads((args.output_dir / "vs_batch.validation_failure.json").read_text(encoding="utf-8"))
    assert marker["failure_scope"] == "batch"
    assert marker["task_name"] is None and marker["expected"] == {}
    assert marker["response_record"] == {}
    assert all(payload[1]["state"] == "completed"
               for payload in driver._load_pending_intents(args.checkpoint_dir).values())
    with pytest.raises(RuntimeError, match="Prior response validation failure"):
        driver.run(args)
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["identity", "latent", "prediction"])
def test_second_gain_failure_is_attributed_to_its_own_task(tmp_path, monkeypatch, failure):
    from GEFcom2014.models.QBM_VAE import kaiwu_13k_protocol
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2)
    assert driver.run(args)["completed"] == 2
    first, second = tasks
    assert first["instance_id"] == second["instance_id"]
    first_intent = args.checkpoint_dir / (first["task_name"] + ".intent.json")
    first_before = first_intent.read_bytes()
    second_audit = args.output_dir / ("gain_%d" % second["gain"]) / (second["instance_id"] + ".json")
    if failure == "identity":
        record = json.loads(second_audit.read_text(encoding="utf-8"))
        record["source_problem_sha256"] = "0" * 64
        second_audit.write_text(json.dumps(record), encoding="utf-8")
    elif failure == "latent":
        original = kaiwu_13k_protocol._recompute_response_metrics

        def fail_second(record, **kwargs):
            if record["gain"] == second["gain"]:
                raise ValueError("synthetic second-gain latent failure")
            return original(record, **kwargs)

        monkeypatch.setattr(kaiwu_13k_protocol, "_recompute_response_metrics", fail_second)
    else:
        original = opsd_vs_selection.decode_platform_spins
        decoded = []

        def fail_second(*args, **kwargs):
            decoded.append(True)
            if len(decoded) == 2:
                raise ValueError("synthetic second-gain prediction failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(opsd_vs_selection, "decode_platform_spins", fail_second)
    with pytest.raises(RuntimeError, match="before submission"):
        driver.run(args)
    markers = list(args.output_dir.rglob("*.validation_failure.json"))
    assert len(markers) == 1
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    assert marker["failure_scope"] == "task"
    assert marker["expected"]["gain"] == second["gain"]
    assert marker["task_name"] == second["task_name"]
    assert marker["response_path"] == str(second_audit.with_suffix(".npz").resolve())
    assert first_intent.read_bytes() == first_before
    second_intent = args.checkpoint_dir / (second["task_name"] + ".intent.json")
    assert json.loads(second_intent.read_text(encoding="utf-8"))["state"] == "validation_failed"
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="Prior response validation failure"):
        driver.run(args)
    assert len(calls) == 2


def test_new_submit_does_not_steal_an_existing_responses_failure(tmp_path, monkeypatch):
    from GEFcom2014.models.QBM_VAE import kaiwu_13k_protocol
    tasks, calls, args = _harness(tmp_path, monkeypatch, count=2)
    original = kaiwu_13k_protocol._recompute_response_metrics

    def fail_first_after_second_submit(record, **kwargs):
        if len(calls) == 2 and record["gain"] == tasks[0]["gain"]:
            raise ValueError("synthetic first-response failure on batch revalidation")
        return original(record, **kwargs)

    monkeypatch.setattr(kaiwu_13k_protocol, "_recompute_response_metrics", fail_first_after_second_submit)
    with pytest.raises(RuntimeError, match="before another submission"):
        driver.run(args)
    marker = json.loads(next(args.output_dir.rglob("*.validation_failure.json")).read_text(encoding="utf-8"))
    assert marker["task_name"] == tasks[0]["task_name"]
    assert marker["expected"]["gain"] == 125
    second_intent = args.checkpoint_dir / (tasks[1]["task_name"] + ".intent.json")
    assert json.loads(second_intent.read_text(encoding="utf-8"))["state"] == "pending"
    with pytest.raises(RuntimeError, match="Prior response validation failure"):
        driver.run(args)
    assert len(calls) == 2
