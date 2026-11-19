"""Read-only status report for a frozen Kaiwu VS batch."""

import argparse
import json
from pathlib import Path

from .kaiwu_13k_protocol import load_protocol
from .kaiwu_adapter import platform_task_name
from .run_kaiwu_13k_sampling import (
    _load_audit_records,
    _load_pending_intents,
    _preparation_identity,
    _validate_preparation_identity,
    _validate_vs_experiment_responses,
    planned_tasks,
    validate_vs_execution_order,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT_DIR / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json"
DEFAULT_PREPARATION = ROOT_DIR / "export/opsd_sa_negative_kaiwu_preparation_10read_v1"


def build_vs_batch_status(tasks, validated, records, intents, failures=(), batch_size=1):
    """Summarize identities already validated by the regular VS validator."""
    validated_keys = {(float(row["gain"]), str(row["instance_id"])) for row in validated}
    intent_by_task = {str(name): payload for name, (_, payload) in intents.items()}
    failures = list(failures)
    rows = []
    runnable = []
    blocked = False
    for position, task in enumerate(tasks):
        key = (float(task["gain"]), str(task["instance_id"]))
        task_records = [record for record in records
                        if (str(record.get("task_name", "")) == str(task.get("task_name", ""))
                            or (str(record.get("instance_id", "")) == str(task["instance_id"])
                                and float(record.get("gain", -1)) == float(task["gain"])
                                and str(record.get("matrix_sha256", "")) == str(task["matrix_sha256"]))) ]
        intent = intent_by_task.get(str(task["task_name"]))
        failure = next((item for item in failures
                        if str(item.get("task_name", "")) == str(task["task_name"])
                        or (str(item.get("instance_id", "")) == str(task["instance_id"])
                            and float(item.get("gain", -1)) == float(task["gain"])
                            and str(item.get("matrix_sha256", "")) == str(task["matrix_sha256"]))), None)
        if key in validated_keys:
            state = "validated_complete"
        elif failure:
            state = "failed_manual_review"
        elif any(not record.get("task_id") for record in task_records):
            state = "awaiting_platform_verification"
        elif intent and intent.get("state") == "completed":
            state = "completed_intent_missing_valid_response"
        elif intent:
            state = "pending_intent_manual_review"
        elif task_records:
            state = "unvalidated_response_manual_review"
        else:
            state = "not_submitted"

        if not blocked and state == "not_submitted" and len(runnable) < int(batch_size):
            runnable.append(task)
        elif state != "validated_complete":
            blocked = True

        validated_record = next((record for record in validated
                                 if (float(record.get("gain", -1)), str(record.get("instance_id", ""))) == key), None)
        derived = bool(validated_record and validated_record.get("record_kind") == "historical_100read_task_prefix_first10")
        rows.append({
            "order": int(task.get("vs_task_order", position)),
            "date": str(task["date"]),
            "instance_id": str(task["instance_id"]),
            "gain": float(task["gain"]),
            "reads": int(task["reads"]),
            "matrix_sha256": str(task["matrix_sha256"]),
            "task_name": str(task["task_name"]),
            "platform_task_name": platform_task_name(task["task_name"]),
            "status": state,
            "intent_state": intent.get("state") if intent else None,
            "response_source": ("historical 100-read task, original first 10 rows"
                                if derived else "new physical 10-read task" if state == "validated_complete" else None),
            "source_task_id": validated_record.get("source_task_id") if derived else
                              validated_record.get("task_id") if validated_record else None,
            "source_actual_reads": ([int(validated_record.get("source_requested_reads", 0)),
                                     int(validated_record.get("source_returned_reads", 0))]
                                    if derived else None),
            "analysis_reads": int(validated_record.get("analysis_reads", 10))
                              if validated_record else None,
        })

    counts = {state: sum(row["status"] == state for row in rows) for state in (
        "validated_complete", "not_submitted", "awaiting_platform_verification",
        "pending_intent_manual_review", "unvalidated_response_manual_review",
        "failed_manual_review", "completed_intent_missing_valid_response")}
    total_reads = sum(row["reads"] for row in rows)
    verified_reads = sum(row["reads"] for row in rows if row["status"] == "validated_complete")
    remaining_reads = total_reads - verified_reads
    historical_records = [record for record in validated
                          if record.get("record_kind") == "historical_100read_task_prefix_first10"]
    historical_source_reads = sum(int(record.get("source_returned_reads", 0)) for record in historical_records)
    new_physical_reads_returned = sum(int(record.get("returned_reads", 0)) for record in validated
                                      if record.get("record_kind") != "historical_100read_task_prefix_first10")
    batch_reads = sum(int(task["reads"]) for task in runnable)
    runnable_keys = {(str(task["instance_id"]), float(task["gain"])) for task in runnable}
    for row in rows:
        row["selected_in_batch"] = (
            str(row["instance_id"]), float(row["gain"])) in runnable_keys
    return {
        "stage": "vs",
        "protocol_sha256": "",
        "preparation_manifest_sha256": "",
        "total_tasks": len(rows),
        "total_reads": total_reads,
        "verified_reads": verified_reads,
        "remaining_reads": remaining_reads,
        "panel_analysis_samples_verified": verified_reads,
        "historical_source_reads_already_paid": historical_source_reads,
        "new_platform_reads_returned": new_physical_reads_returned,
        "new_platform_reads_remaining": remaining_reads,
        "estimated_credits": None,
        "credit_estimate_reason": "Sampling Credits per read has not been verified",
        "counts": counts,
        "batch_limit_tasks": int(batch_size),
        "batch_tasks": len(runnable),
        "batch_max_reads": batch_reads,
        "batch_blocked_at_first_unresolved_task": len(runnable) < min(
            int(batch_size), counts["not_submitted"]),
        "tasks": rows,
    }


def inspect_vs_batch(protocol_path=DEFAULT_PROTOCOL, preparation=DEFAULT_PREPARATION,
                     batch_size=1, panel_revision=None):
    protocol_path = Path(protocol_path).resolve()
    preparation = Path(preparation).resolve()
    protocol = load_protocol(protocol_path)
    protocol["_path"] = str(protocol_path)
    _validate_preparation_identity(preparation, protocol)
    output_dir = preparation / "real_responses"
    tasks = planned_tasks("vs", preparation, output_dir, protocol)
    identity = _preparation_identity(preparation, protocol)
    validate_vs_execution_order(
        preparation / "vs_execution_order.json", tasks,
        identity["protocol_sha256"], identity["preparation_manifest_sha256"])
    records = _load_audit_records(output_dir)
    intents = _load_pending_intents(preparation / "checkpoints")
    if panel_revision is not None:
        from .kaiwu_panel_revision import load_historical_panel_records
        records.extend(load_historical_panel_records(
            panel_revision, tasks, protocol, preparation,
            output_dir=output_dir, intents=intents, existing_records=records))
    validated = _validate_vs_experiment_responses(tasks, records, preparation, protocol)
    failures = []
    for path in output_dir.rglob("*.json"):
        if not (path.name.endswith(".failure.json")
                or path.name.endswith(".validation_failure.json")):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            failures.append(data)
    report = build_vs_batch_status(tasks, validated, records, intents, failures, batch_size)
    report["protocol_sha256"] = identity["protocol_sha256"]
    report["preparation_manifest_sha256"] = identity["preparation_manifest_sha256"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--panel-revision", type=Path, default=None,
                        help="Opt into a separately frozen, hash-bound panel addendum")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    report = inspect_vs_batch(args.protocol, args.preparation, args.batch_size, args.panel_revision)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
