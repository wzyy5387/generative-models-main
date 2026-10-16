# -*- coding: utf-8 -*-
"""Resumable, manually-gated VS/TEST Kaiwu batch driver.

The default is an offline dry-run.  This module never creates physical-platform
claims for plans, mocks, or local SA responses.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .kaiwu_13k_protocol import load_protocol
from .kaiwu_adapter import (build_task_name, matrix_sha256,
                             official_task_query_capability,
                             quantize_hardware_matrix, validate_hardware_matrix)
from .prepare_bosonic_submission import sha256_file, resolve_repo_path
from .import_hardware_responses import strict_response_record_valid
from .submit_kaiwu_sampling import submit_one


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREPARATION = ROOT_DIR / "export" / "kaiwu_13k_preparation"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _preparation_identity(preparation, protocol):
    manifest_path = Path(preparation) / "preparation_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = protocol.get("_path")
    return {
        "protocol_sha256": _sha256(protocol_path) if protocol_path and Path(protocol_path).is_file() else "",
        "model_checkpoint_sha256": str(manifest.get("model_checkpoint_sha256", "")),
        "preparation_manifest_sha256": _sha256(manifest_path),
    }


def _validate_preparation_identity(preparation, protocol):
    """Reject a preparation directory built for another protocol/model/data bundle."""
    manifest_path = Path(preparation) / "preparation_manifest.json"
    if not manifest_path.is_file():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = resolve_repo_path(protocol.get("_path", "")) if protocol.get("_path") else None
    if protocol_path is not None and protocol_path.is_file():
        expected_protocol_hash = sha256_file(protocol_path)
        if payload.get("protocol_sha256") != expected_protocol_hash:
            raise RuntimeError("Preparation directory was built from a different protocol JSON")
    expected_model_hash = protocol.get("model", {}).get("checkpoint")
    freeze_path = Path(preparation) / "frozen_model_audit.json"
    if expected_model_hash and freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        actual = freeze.get("files", {}).get("checkpoint", {}).get("path")
        if actual and Path(actual).name != Path(expected_model_hash).name:
            raise RuntimeError("Preparation directory model checkpoint does not match protocol")


def _vs_tasks(preparation, protocol=None):
    tasks = []
    gains = (protocol or {}).get("hardware", {}).get("candidate_gains", [125, 150, 200])
    expected_instances = int((protocol or {}).get("vs", {}).get("instances", 10))
    identity = _preparation_identity(preparation, protocol or {})
    ids_by_gain = {}
    for gain in gains:
        manifest_path = preparation / ("submission_gain_%d" % gain) / "package" / "submission_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Missing frozen VS submission manifest: %s" % manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("split") != "VS" or int(payload.get("n_instances", -1)) != expected_instances:
            raise ValueError("VS submission manifest does not match the protocol")
        expected_reads = int((protocol or {}).get("hardware", {}).get("reads_per_instance", 100))
        if int(payload.get("requested_reads_per_instance", -1)) != expected_reads:
            raise ValueError("VS submission manifest reads do not match the protocol")
        current_ids = []
        for item in payload.get("instances", []):
            matrix = manifest_path.parent / item["hardware_matrix"]
            current_ids.append(str(item["instance_id"]))
            task = {"gain": gain, "manifest": manifest_path, "stage": "vs",
                          "task_name": (json.loads((manifest_path.parent / item["payload"]).read_text(encoding="utf-8")).get("task_name") if item.get("payload") else None),
                          "instance_id": item["instance_id"],
                          "matrix": matrix, "matrix_sha256": item["hardware_matrix_sha256"],
                          "reads": int(payload["requested_reads_per_instance"])}
            task.update(identity)
            tasks.append(task)
        ids_by_gain[float(gain)] = current_ids
        if len([item for item in tasks if item["gain"] == gain]) != expected_instances:
            raise ValueError("Frozen VS plan has the wrong instance count for gain %s" % gain)
        if len(set(current_ids)) != expected_instances:
            raise ValueError("Frozen VS plan contains duplicate conditions for gain %s" % gain)
    if set(ids_by_gain) != {float(gain) for gain in gains} or len({tuple(sorted(ids)) for ids in ids_by_gain.values()}) != 1:
        raise ValueError("All VS gains must use the same frozen condition panel")
    if len(tasks) != len(gains) * expected_instances or len({(float(t["gain"]), str(t["instance_id"])) for t in tasks}) != len(tasks):
        raise ValueError("VS plan must contain exactly one task per gain-condition combination")
    return tasks


def _task_expected(task):
    return {key: task[key] for key in
            ("stage", "instance_id", "gain", "matrix_sha256", "requested_reads", "task_name",
             "protocol_sha256", "model_checkpoint_sha256", "preparation_manifest_sha256")
            if task.get(key) is not None and task.get(key) != ""}


def _load_audit_records(output_dir):
    records = []
    for path in Path(output_dir).glob("**/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        nested = data.get("records", []) if isinstance(data, dict) else []
        candidates = [data] + [item for item in nested if isinstance(item, dict)]
        records.extend(item for item in candidates if isinstance(item, dict))
    return records


def _load_completed(output_dir, tasks=None):
    records = _load_audit_records(output_dir)
    completed = set()
    if tasks is None:
        for record in records:
            if strict_response_record_valid(record) and record.get("matrix_sha256"):
                completed.add((str(record["instance_id"]), str(record["matrix_sha256"])))
        return completed
    for task in tasks:
        expected = _task_expected(task)
        if any(strict_response_record_valid(record, expected) for record in records):
            completed.add((task["instance_id"], task["matrix_sha256"]))
    return completed


def select_pending_tasks(tasks, completed, limit=None):
    """Filter validated records before applying --limit (resume-safe)."""
    pending = [item for item in tasks
               if (item["instance_id"], item["matrix_sha256"]) not in set(completed)]
    return pending if limit is None else pending[:max(0, int(limit))]


def _deterministic_task_name(task, protocol_id):
    return build_task_name(
        task["instance_id"], task["matrix_sha256"],
        protocol_id=protocol_id,
        stage=("vs-calibration" if task["stage"] == "vs" else "test-evaluation"),
        gain=task["gain"],
    )


def _validated_records_for_tasks(tasks, records):
    validated = []
    for task in tasks:
        expected = _task_expected(task)
        matches = [record for record in records
                   if strict_response_record_valid(record, expected)]
        if matches:
            validated.append(matches[0])
    return validated


def _validate_execution_limits(tasks):
    """Validate every frozen matrix; SPQC-550 has no guessed edge cap."""
    checked = set()
    for item in tasks:
        key = str(item["manifest"])
        if key in checked:
            continue
        checked.add(key)
        payload = json.loads(Path(item["manifest"]).read_text(encoding="utf-8"))
        limits = payload.get("platform_limits_checked", {})
        if limits.get("int8_coefficient_range") != [-128, 127]:
            raise RuntimeError("Real execution requires verified int8 coefficient range")
        if limits.get("int8_range_verified") is not True:
            raise RuntimeError("Real execution requires verified int8 coefficient range")
        if limits.get("machine_profile") != {"name": "SPQC-550", "max_variables": 550,
                                              "full_connectivity": True,
                                              "coupling_precision": "int8"}:
            raise RuntimeError("Real execution requires the SPQC-550 machine profile")
        for instance in payload.get("instances", []):
            if instance.get("clipping") is True or instance.get("quantization_audit", {}).get("clipping") is True:
                raise RuntimeError("Real execution refuses a clipped matrix")
            path = Path(item["manifest"]).parent / instance["hardware_matrix"]
            with np.load(path, allow_pickle=False) as archive:
                matrix = validate_hardware_matrix(archive["hardware_matrix"], 49)
            if matrix_sha256(matrix) != instance.get("hardware_matrix_sha256"):
                raise RuntimeError("Frozen matrix SHA-256 mismatch")


def _load_frozen_gain(preparation, protocol=None, allow_offline=False, offline_freeze_path=None):
    path = (Path(offline_freeze_path) if offline_freeze_path is not None else
            preparation / ("offline_mock_gain_selection.json" if allow_offline else "real_gain_selection.json"))
    if not path.is_file():
        raise RuntimeError("TEST is locked until real_gain_selection.json is frozen from verified VS responses")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        selected_gain = float(payload.get("selected_gain"))
    except (TypeError, ValueError):
        selected_gain = None
    physical_ok = payload.get("hardware_claim") is True and payload.get("physical_platform_used") is True
    offline_ok = allow_offline and payload.get("artifact_role") == "offline_simulation" and not physical_ok
    if ((not physical_ok and not offline_ok) or payload.get("immutable") is not True
            or selected_gain not in (125.0, 150.0, 200.0)):
        raise RuntimeError("TEST is locked until gain freeze records verified physical VS responses")
    verified = payload.get("verified_vs_records")
    if not verified:
        raise RuntimeError("TEST is locked until verified VS records are listed in the gain freeze")
    if protocol is not None:
        from .kaiwu_13k_protocol import (aggregate_gain_metrics,
                                         evaluate_gain_quality,
                                         validate_vs_response_records)
        vs_tasks = _vs_tasks(preparation, protocol)
        expected_ids = sorted({str(item["instance_id"]) for item in vs_tasks})
        expected_map = {(float(item["gain"]), str(item["instance_id"])): _task_expected(item)
                        for item in vs_tasks}
        identity = _preparation_identity(preparation, protocol)
        quality_policy = (payload.get("quality_gate") if allow_offline
                          else protocol.get("gain_selection", {}).get("quality_gate"))
        if not isinstance(quality_policy, dict) or quality_policy.get("status") != "defined":
            raise RuntimeError("TEST is locked until a pre-defined VS quality policy is frozen")
        validate_vs_response_records(
            verified,
            required_gains=protocol["hardware"]["candidate_gains"],
            expected_instances_per_gain=int(protocol["vs"]["instances"]),
            expected_instance_ids=expected_ids,
            expected_task_map=expected_map,
            expected_identity=identity,
            quality_policy=quality_policy,
            require_quality_policy=False,
            require_physical=not allow_offline,
        )
        candidate_metrics = payload.get("candidate_metrics")
        if not isinstance(candidate_metrics, list) or {float(row.get("gain", -1)) for row in candidate_metrics} != set(float(gain) for gain in protocol["hardware"]["candidate_gains"]):
            raise RuntimeError("TEST gain freeze lacks the complete candidate metric grid")
        if any(not all(np.isfinite(float(row[key])) for key in
                       ("energy_wasserstein", "effective_beta_error", "edge_moment_mae",
                        "magnetization_mae", "unique_state_fraction"))
               for row in candidate_metrics):
            raise RuntimeError("TEST gain freeze contains non-finite candidate metrics")
        recomputed = evaluate_gain_quality(
            aggregate_gain_metrics(verified), quality_policy
        )
        for stored, current in zip(candidate_metrics, recomputed):
            for key in ("gain", "n_verified_instances", "energy_wasserstein",
                        "effective_beta_error", "edge_moment_mae",
                        "magnetization_mae", "unique_state_fraction"):
                if not np.isclose(float(stored[key]), float(current[key]), rtol=0.0, atol=1e-12):
                    raise RuntimeError("TEST gain freeze candidate metrics are not reproducible")
        if payload.get("quality_summary_recomputed") is not True:
            raise RuntimeError("TEST gain freeze lacks a recomputed quality summary")
        if any(row.get("quality_status") != recomputed[index].get("quality_status")
               for index, row in enumerate(candidate_metrics)):
            raise RuntimeError("TEST gain freeze quality status is not reproducible")
        qualified = [row for row in recomputed if row.get("quality_status") == "pass"]
        if not qualified:
            raise RuntimeError("All candidate gains failed the VS quality gate; TEST remains locked")
        chosen = min(qualified, key=lambda row: (
            float(row["energy_wasserstein"]), float(row["effective_beta_error"]),
            float(row["edge_moment_mae"]), float(row["magnetization_mae"]),
            -float(row["unique_state_fraction"]), float(row["gain"]),
        ))
        if float(payload.get("selected_gain", -1)) != float(chosen["gain"]):
            raise RuntimeError("TEST gain freeze selected gain does not match its candidate metrics")
        for key, value in identity.items():
            if str(payload.get(key, "")) != str(value):
                raise RuntimeError("TEST gain freeze is bound to a different %s" % key)
        if payload.get("test_data_used") is not False or payload.get("test_peeking") is not False:
            raise RuntimeError("TEST gain freeze declares TEST data use")
    return payload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_pending_intents(checkpoint_dir):
    intents = {}
    for path in Path(checkpoint_dir).glob("*.intent.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        required = ("task_name", "stage", "instance_id", "gain",
                    "matrix_sha256", "requested_reads", "created_at", "state")
        if all(key in payload for key in required) and payload.get("state") != "completed":
            intents[str(payload["task_name"])] = (path, payload)
    return intents


def _query_existing_task(intent, task):
    """Query only a public task-name/task-id SDK capability; otherwise stop."""
    if official_task_query_capability() is None:
        raise RuntimeError("任务状态未知，需要人工在平台确认 (task status unknown; manual platform confirmation required)")
    # The installed SDK currently has no supported task-name/task-id query
    # adapter.  Do not guess a call signature or risk a duplicate charge.
    raise RuntimeError("任务状态未知，需要人工在平台确认 (task status unknown; manual platform confirmation required)")


def _recover_pending_intent(intent, task):
    result = _query_existing_task(intent, task)
    if not isinstance(result, dict):
        raise RuntimeError("任务状态未知，需要人工在平台确认 (incomplete task query result)")
    record = result.get("audit", result)
    if not strict_response_record_valid(record, _task_expected(task)):
        raise RuntimeError("任务状态未知，需要人工在平台确认 (queried response failed strict validation)")
    return record


def _mark_intent_completed(intent_path, intent):
    completed = dict(intent)
    completed["state"] = "completed"
    completed["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(intent_path, completed)


def _test_source_manifest(preparation):
    matches = sorted((Path(preparation) / "test_logical").glob("*_test_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError("TEST requires exactly one frozen logical TEST manifest")
    return matches[0]


def _build_test_manifest(preparation, selected_gain, protocol_id, protocol=None):
    source = _test_source_manifest(preparation)
    out = Path(preparation) / ("submission_test_gain_%d" % int(selected_gain)) / "package"
    manifest_path = out / "submission_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("hardware_gain") != float(selected_gain) or existing.get("source_manifest_sha256") != sha256_file(source):
            raise RuntimeError("Existing TEST manifest is immutable and does not match the frozen gain")
        return manifest_path
    source_items = json.loads(source.read_text(encoding="utf-8"))
    out.joinpath("payloads").mkdir(parents=True, exist_ok=True)
    out.joinpath("hardware_matrices").mkdir(parents=True, exist_ok=True)
    records = []
    expected_reads = int((protocol or {}).get("hardware", {}).get("reads_per_instance", 100))
    expected_instances = int((protocol or {}).get("test", {}).get("instances", 100))
    if len(source_items) != expected_instances:
        raise RuntimeError("Frozen TEST manifest has %d instances; protocol requires %d" % (len(source_items), expected_instances))
    for item in source_items:
        payload_path = resolve_repo_path(item["platform_payload"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if payload.get("instance_id") != item["id"] or item.get("split") != "TEST":
            raise RuntimeError("TEST source manifest/payload mismatch")
        h = np.asarray(payload["h"], dtype=float)
        j = np.zeros((h.size, h.size), dtype=float)
        for edge in payload.get("couplings", []):
            j[int(edge["i"]), int(edge["j"])] = float(edge["value"])
            j[int(edge["j"]), int(edge["i"])] = float(edge["value"])
        quantized = quantize_hardware_matrix(h, j, selected_gain, logical_n_bits=48)
        matrix = quantized["matrix"]
        matrix_name = item["id"] + ".npz"
        payload_name = item["id"] + ".json"
        matrix_path = out / "hardware_matrices" / matrix_name
        payload_out = out / "payloads" / payload_name
        np.savez_compressed(matrix_path, hardware_matrix=matrix,
                            matrix_sha256=np.asarray(matrix_sha256(matrix)),
                            hardware_gain=np.asarray(selected_gain))
        payload.update({"hardware_matrix": matrix.tolist(),
                        "hardware_matrix_file": "hardware_matrices/" + matrix_name,
                        "hardware_matrix_sha256": matrix_sha256(matrix),
                        "hardware_gain": float(selected_gain),
                        "hardware_n_bits": 49, "auxiliary_spin_index": 48,
                        "task_name": build_task_name(item["id"], matrix_sha256(matrix),
                                                      protocol_id=protocol_id,
                                                      stage="test-evaluation", gain=selected_gain),
                        "hardware_claim": False, "physical_platform_used": False})
        payload_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        records.append({"instance_id": item["id"], "split": "TEST",
                        "payload": "payloads/" + payload_name,
                        "payload_sha256": sha256_file(payload_out),
                        "hardware_matrix": "hardware_matrices/" + matrix_name,
                        "hardware_matrix_sha256_file": sha256_file(matrix_path),
                        "hardware_matrix_sha256": matrix_sha256(matrix),
                        "hardware_gain": float(selected_gain), "hardware_n_bits": 49,
                        "logical_n_bits": 48, "auxiliary_spin_index": 48,
                        "n_bits": 49, "n_edges": int(np.count_nonzero(np.triu(matrix != 0, 1))),
                        "source_n_edges": int(sum(1 for e in payload.get("couplings", []) if float(e["value"]) != 0)),
                        "max_abs_coefficient": int(np.abs(matrix).max())})
    manifest = {"schema_version": 1, "stage": "test-evaluation", "split": "TEST",
                "hardware_claim": False, "physical_platform_used": False,
                "frozen": True, "hardware_gain": float(selected_gain),
                "requested_reads_per_instance": expected_reads, "n_instances": len(records),
                "source_manifest_sha256": sha256_file(source),
                "machine_profile": {"name": "SPQC-550", "max_variables": 550,
                                    "full_connectivity": True, "coupling_precision": "int8"},
                "platform_limits_checked": {"int8_coefficient_range": [-128, 127],
                    "int8_range_verified": True, "machine_profile": {"name": "SPQC-550", "max_variables": 550,
                    "full_connectivity": True, "coupling_precision": "int8"}},
                "instances": records}
    _atomic_json(manifest_path, manifest)
    return manifest_path


def planned_tasks(stage, preparation, output_dir, protocol=None, offline_simulation=False,
                  offline_freeze_path=None):
    if stage == "vs":
        return _vs_tasks(preparation, protocol)
    freeze = _load_frozen_gain(preparation, protocol, allow_offline=offline_simulation,
                               offline_freeze_path=offline_freeze_path)
    manifest_path = _build_test_manifest(preparation, float(freeze["selected_gain"]),
                                         (protocol or {}).get("protocol_version", "fa-bm-vae-kaiwu-13k-v1"),
                                         protocol)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [{"gain": float(payload["hardware_gain"]), "manifest": manifest_path,
             "stage": "test", "task_name": (json.loads((manifest_path.parent / item["payload"]).read_text(encoding="utf-8")).get("task_name") if item.get("payload") else None),
             "instance_id": item["instance_id"],
             "matrix": manifest_path.parent / item["hardware_matrix"],
             "matrix_sha256": item["hardware_matrix_sha256"],
             "reads": int(payload["requested_reads_per_instance"])} for item in payload["instances"]]


def run(args):
    protocol = load_protocol(args.protocol)
    protocol["_path"] = str(Path(args.protocol).resolve())
    preparation = Path(args.preparation).resolve()
    _validate_preparation_identity(preparation, protocol)
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not hasattr(args, "offline_simulation") and not hasattr(args, "offline_freeze"):
        all_tasks = planned_tasks(args.stage, preparation, output_dir, protocol)
    else:
        all_tasks = planned_tasks(
            args.stage, preparation, output_dir, protocol,
            offline_simulation=getattr(args, "offline_simulation", False),
            offline_freeze_path=getattr(args, "offline_freeze", None),
        )
    completed = _load_completed(output_dir, all_tasks) if args.resume else set()
    recovered_records = []
    intents = _load_pending_intents(checkpoint_dir)
    if args.execute:
        for task in all_tasks:
            key = (task["instance_id"], task["matrix_sha256"])
            task_name = _deterministic_task_name(task, protocol.get("protocol_version"))
            intent_entry = intents.get(task_name)
            if intent_entry is None:
                for candidate_path, candidate in intents.values():
                    if (str(candidate.get("instance_id")) == str(task["instance_id"]) and
                            str(candidate.get("matrix_sha256")) == str(task["matrix_sha256"]) and
                            float(candidate.get("gain")) == float(task["gain"])):
                        raise RuntimeError("任务状态未知，需要人工在平台确认 (pending intent task name mismatch)")
            if key in completed:
                continue
            if intent_entry is not None:
                if not args.resume:
                    raise RuntimeError("任务状态未知，需要人工在平台确认 (resume is required for a pending intent)")
                intent_path, intent = intent_entry
                record = _recover_pending_intent(intent, task)
                recovered_records.append(record)
                completed.add(key)
                _mark_intent_completed(intent_path, intent)
    tasks = select_pending_tasks(all_tasks, completed, args.limit)
    total_reads = sum(item["reads"] for item in tasks)
    max_tasks = len(all_tasks)
    max_reads = sum(item["reads"] for item in all_tasks)
    if len(tasks) > max_tasks or total_reads > max_reads:
        raise ValueError("Requested batch exceeds frozen %s budget" % args.stage.upper())
    print("stage=%s tasks=%d reads=%d dry_run=%s" % (args.stage, len(tasks), total_reads, not args.execute))
    for item in tasks:
        print("%s gain=%s reads=%d matrix_sha256=%s" %
              (item["instance_id"], item["gain"], item["reads"], item["matrix_sha256"]))
    if not args.execute:
        return {"stage": args.stage, "tasks": len(tasks), "reads": total_reads, "hardware_claim": False,
                "physical_platform_used": False, "dry_run": True}
    audit_path = output_dir / (args.stage + "_batch_audit.json")
    audit_records = _load_audit_records(output_dir)
    audit_records.extend(recovered_records)
    if not tasks:
        validated = _validated_records_for_tasks(all_tasks, audit_records)
        complete = bool(all_tasks) and len(validated) == len(all_tasks)
        pending_count = len(all_tasks) - len(validated)
        if recovered_records:
            _atomic_json(audit_path, {"stage": args.stage, "records": validated,
                "completed": len(validated), "skipped_validated": len(completed),
                "pending": len(all_tasks) - len(validated),
                "requested_reads": sum(item["reads"] for item in all_tasks),
                "returned_reads": sum(int(r["returned_reads"]) for r in validated),
                "hardware_claim": complete, "physical_platform_used": complete})
        return {"stage": args.stage, "completed": len(validated),
                "skipped_validated": len(completed), "pending": pending_count,
                "requested_reads": sum(item["reads"] for item in all_tasks),
                "returned_reads": sum(int(r["returned_reads"]) for r in validated), "hardware_claim": complete,
                "physical_platform_used": complete}
    _validate_execution_limits(tasks)
    if not args.confirm_real_submission or os.environ.get("FA_BM_VAE_RUN_REAL_KAIWU") != "1":
        raise RuntimeError("Real batch requires --confirm-real-submission and FA_BM_VAE_RUN_REAL_KAIWU=1")
    project_no = args.project_no or os.environ.get("KAIWU_PROJECT_NO")
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in tasks:
        response_dir = output_dir / ("gain_%d" % item["gain"])
        task_name = _deterministic_task_name(item, protocol.get("protocol_version"))
        _atomic_json(checkpoint_dir / (task_name + ".intent.json"), {
            "task_name": task_name, "stage": args.stage, "instance_id": item["instance_id"],
            "gain": item["gain"], "matrix_sha256": item["matrix_sha256"],
            "requested_reads": item["reads"], "created_at": datetime.now(timezone.utc).isoformat(),
            "state": "pending"})
        result = submit_one(item["matrix"], item["instance_id"], response_dir, item["reads"],
                            project_no=project_no, checkpoint_dir=checkpoint_dir,
                            task_name=task_name, stage=args.stage, hardware_gain=item["gain"])
        record = result[2]
        record.update({key: item[key] for key in
                       ("protocol_sha256", "model_checkpoint_sha256", "preparation_manifest_sha256")
                       if item.get(key)})
        record["completed"] = bool(record.get("hardware_claim") and record.get("returned_reads") == item["reads"])
        audit_records.append(record)
        validated = _validated_records_for_tasks(all_tasks, audit_records)
        completed_count = len(validated)
        returned_total = sum(int(r.get("returned_reads", 0)) for r in validated)
        _atomic_json(audit_path, {"stage": args.stage, "records": validated,
            "completed": completed_count,
            "skipped_validated": len(completed),
            "pending": len(all_tasks) - completed_count,
            "requested_reads": sum(item["reads"] for item in all_tasks),
            "returned_reads": returned_total,
            "hardware_claim": bool(all_tasks) and completed_count == len(all_tasks),
            "physical_platform_used": bool(all_tasks) and completed_count == len(all_tasks)})
    validated = _validated_records_for_tasks(all_tasks, audit_records)
    verified = len(validated)
    return {"stage": args.stage, "completed": verified, "skipped_validated": len(completed),
            "pending": len(all_tasks) - verified,
            "requested_reads": sum(item["reads"] for item in all_tasks),
            "returned_reads": sum(int(r.get("returned_reads", 0)) for r in validated),
            "hardware_claim": bool(all_tasks) and verified == len(all_tasks),
            "physical_platform_used": bool(all_tasks) and verified == len(all_tasks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["vs", "test"], required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT_DIR / "configs" / "paper" / "fa_bm_vae_kaiwu_13k_protocol.json")
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARATION / "real_responses")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_PREPARATION / "checkpoints")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Default; print plan without SDK calls")
    parser.add_argument("--execute", action="store_true", help="Still requires all real-submission gates")
    parser.add_argument("--confirm-real-submission", action="store_true")
    parser.add_argument("--project-no", default=None)
    parser.add_argument("--offline-simulation", action="store_true",
                        help="Allow only dry-run planning from non-physical mock freeze records")
    parser.add_argument("--offline-freeze", type=Path, default=None,
                        help="Explicit offline mock freeze; only accepted with --offline-simulation")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        raise SystemExit("Choose only one of --dry-run and --execute")
    if args.offline_simulation and (not args.dry_run or args.execute):
        raise SystemExit("--offline-simulation is restricted to --dry-run")
    if args.offline_freeze is not None and not args.offline_simulation:
        raise SystemExit("--offline-freeze requires --offline-simulation")
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
