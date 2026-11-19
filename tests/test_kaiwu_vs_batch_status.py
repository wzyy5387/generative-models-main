from GEFcom2014.models.QBM_VAE.kaiwu_vs_batch_status import build_vs_batch_status
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import platform_task_name


def _task(order, instance, gain, matrix):
    return {
        "vs_task_order": order, "date": "2018-11-%02d 00:00:00" % (order + 1),
        "instance_id": instance, "gain": gain, "reads": 10,
        "matrix_sha256": matrix, "task_name": "task-%s-%s" % (instance, gain),
    }


def test_status_counts_valid_response_and_pending_intent_without_double_counting():
    tasks = [_task(0, "i0", 125, "a" * 64), _task(1, "i1", 150, "b" * 64)]
    validated = [{"instance_id": "i0", "gain": 125}]
    records = [{"instance_id": "i0", "gain": 125, "matrix_sha256": "a" * 64,
                "task_name": "task-i0-125", "task_id": "real-id"}]
    intents = {"task-i0-125": (None, {"state": "pending"})}
    result = build_vs_batch_status(tasks, validated, records, intents, batch_size=4)
    assert result["counts"]["validated_complete"] == 1
    assert result["counts"]["not_submitted"] == 1
    assert result["verified_reads"] == 10
    assert result["remaining_reads"] == 10
    assert result["batch_tasks"] == 1
    assert result["batch_max_reads"] == 10
    assert result["tasks"][0]["status"] == "validated_complete"
    assert result["tasks"][0]["intent_state"] == "pending"
    assert result["tasks"][1]["platform_task_name"] == platform_task_name("task-i1-150")
    assert result["estimated_credits"] is None


def test_status_stops_batch_at_unresolved_identity_and_reports_it():
    tasks = [_task(0, "i0", 125, "a" * 64), _task(1, "i1", 150, "b" * 64),
             _task(2, "i2", 200, "c" * 64)]
    records = [{"instance_id": "i1", "gain": 150, "matrix_sha256": "b" * 64,
                "task_name": "task-i1-150", "status": "awaiting_platform_verification"}]
    result = build_vs_batch_status(tasks, [], records, {}, batch_size=3)
    assert result["counts"]["not_submitted"] == 2
    assert result["counts"]["awaiting_platform_verification"] == 1
    assert result["batch_tasks"] == 1
    assert result["batch_max_reads"] == 10
    assert result["batch_blocked_at_first_unresolved_task"] is True
