"""Explicit hash-bound addendum for one historical response prefix in VS."""

import argparse
import json
from pathlib import Path

from .kaiwu_13k_protocol import load_protocol, sha256_file
from .derive_historical_kaiwu_prefix import validate_bundle


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREFIX_BUNDLE = ROOT / "export/opsd_sa_negative_kaiwu_historical_prefix_v3"
DEFAULT_PROTOCOL = ROOT / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json"
DEFAULT_PREPARATION = ROOT / "export/opsd_sa_negative_kaiwu_preparation_10read_v1"
DEFAULT_REVISION_DIR = ROOT / "export/opsd_sa_negative_kaiwu_panel_revision_historical_prefix_v1"
RECORD_KIND = "historical_100read_task_prefix_first10"
_VALIDATION_TOKEN = object()


def _frozen_inputs(protocol_path, preparation):
    protocol_path = Path(protocol_path).resolve()
    preparation = Path(preparation).resolve()
    protocol = load_protocol(protocol_path)
    protocol["_path"] = str(protocol_path)
    return protocol_path, protocol, preparation


def create_panel_revision(prefix_bundle, protocol_path, preparation, output_dir):
    prefix_bundle = Path(prefix_bundle).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite panel revision: %s" % output_dir)
    prefix = validate_bundle(prefix_bundle)
    protocol_path, protocol, preparation = _frozen_inputs(protocol_path, preparation)
    preparation_manifest = preparation / "preparation_manifest.json"
    order_path = preparation / "vs_execution_order.json"
    if (sha256_file(protocol_path) != prefix["base_protocol"]["sha256"]
            or sha256_file(preparation_manifest) != prefix["base_protocol"]["preparation_manifest_sha256"]
            or sha256_file(order_path) != prefix["base_protocol"]["vs_execution_order_sha256"]):
        raise ValueError("Historical-prefix bundle is bound to different base VS inputs")
    if int(protocol["hardware"]["reads_per_instance"]) != 10:
        raise ValueError("Panel revision applies only to the frozen 10-read protocol")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = output_dir.parent / (output_dir.name + ".tmp")
    if temp.exists():
        raise FileExistsError("Refusing to reuse temporary panel-revision path")
    temp.mkdir()
    revision = {
        "schema_version": 1,
        "revision_id": "opsd-vs-historical-100read-prefix-first10-v1",
        "revision_status": "explicit_posthoc_panel_addendum",
        "base_protocol": {
            "path": str(protocol_path), "sha256": sha256_file(protocol_path),
            "preparation_path": str(preparation),
            "preparation_manifest_sha256": sha256_file(preparation_manifest),
            "vs_execution_order_sha256": sha256_file(order_path),
        },
        "historical_prefix_bundle": {
            "path": str(prefix_bundle),
            "manifest_path": str((prefix_bundle / "historical_prefix_amendment.json").resolve()),
            "manifest_sha256": sha256_file(prefix_bundle / "historical_prefix_amendment.json"),
            "manifest_lock_path": str((prefix_bundle / "historical_prefix_amendment.sha256").resolve()),
            "manifest_lock_sha256": sha256_file(prefix_bundle / "historical_prefix_amendment.sha256"),
            "derived_samples_path": str((prefix_bundle / prefix["derived_data"]["path"]).resolve()),
            "derived_samples_sha256": prefix["derived_data"]["sha256"],
        },
        "panel_mapping": {
            "instance_id": prefix["new_panel_entry"]["instance_id"],
            "date": prefix["new_panel_entry"]["date"],
            "gain": prefix["new_panel_entry"]["gain"],
            "analysis_reads": prefix["new_panel_entry"]["analysis_reads"],
            "panel_task_name": prefix["new_panel_entry"]["target_task_name"],
            "matrix_content_sha256": prefix["new_panel_entry"]["target_matrix_content_sha256"],
            "source_task_id": prefix["source_physical_task"]["task_id"],
            "source_task_name": prefix["source_physical_task"]["task_name"],
            "source_requested_reads": prefix["source_physical_task"]["requested_reads"],
            "source_returned_reads": prefix["source_physical_task"]["returned_reads"],
            "source_row_indices": prefix["derived_data"]["source_row_indices"],
            "derived_samples_sha256": prefix["derived_data"]["sha256"],
            "source_response_sha256": prefix["source_physical_task"]["response_sha256"],
            "source_receipt_sha256": prefix["source_physical_task"]["receipt_sha256"],
            "source_matrix_file_sha256": prefix["source_physical_task"]["hardware_matrix_file_sha256"],
            "energy_convention": prefix["source_physical_task"]["hardware_energy_convention"],
        },
        "posthoc_decision": prefix["posthoc_decision_disclosure"],
        "accounting": {
            "historical_physical_reads_already_paid": 100,
            "historical_task_actual_requested_returned": [100, 100],
            "panel_analysis_samples_reused": 10,
            "new_physical_reads_created_by_reuse": 0,
            "base_panel_conditions": 30,
            "verified_existing_10read_conditions": 3,
            "historical_prefix_conditions": 1,
            "new_platform_tasks_remaining": 26,
            "new_platform_reads_remaining": 260,
            "sampling_credits_conversion_verified": False,
        },
        "rules": {
            "count_source_task_once": True,
            "never_claim_historical_task_requested_10_reads": True,
            "never_create_new_platform_task_id_or_receipt": True,
            "duplicate_new_response_or_intent_action": "stop_for_manual_review",
            "missing_or_changed_source_action": "stop_for_manual_review",
            "base_protocol_without_this_revision": "continue_to_reject_historical_100read_task",
            "test_unlock": "unchanged; requires all 30 panel conditions validated and a separately frozen VS gain",
            "posthoc_status": "reuse decision followed observation of one logical state across all 100 returned reads",
        },
    }
    revision_path = temp / "panel_revision.json"
    revision_path.write_text(json.dumps(revision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (temp / "panel_revision.sha256").write_text(
        sha256_file(revision_path) + "  panel_revision.json\n", encoding="ascii")
    report = (
        "# VS panel revision: historical prefix\n\n"
        "This explicit post-hoc addendum maps the first 10 original rows of the validated historical 100/100 task to one analysis slot in the 10-read VS panel. The original task ID, task name, request/return counts, response, receipt, and evidence remain unchanged. The derived record is not itself a Kaiwu task.\n\n"
        "The selection decision followed observation that all 100 original reads normalize to one logical state. The addendum binds the base 10-read protocol, preparation, execution order, historical source bundle, response, receipt, screenshot, matrix, and derived sample hashes. Without the addendum, the base protocol continues to reject this source.\n\n"
        "Accounting: the old task consumed 100 actual reads; the panel analyzes its first 10; reuse adds zero reads; 26 new tasks (260 reads) remain. Credits conversion is unknown. TEST stays locked until all 30 panel entries are validated and a gain is frozen under the existing rule.\n"
    )
    (temp / "PANEL_REVISION.md").write_text(report, encoding="utf-8")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp.replace(output_dir)
    return output_dir / "panel_revision.json"


def _read_revision(path):
    path = Path(path).resolve()
    lock_path = path.with_suffix(".sha256")
    if not path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("Panel revision JSON or hash lock is missing")
    expected = lock_path.read_text(encoding="ascii").split()[0]
    if sha256_file(path) != expected:
        raise ValueError("Panel revision hash mismatch")
    revision = json.loads(path.read_text(encoding="utf-8"))
    if revision.get("revision_status") != "explicit_posthoc_panel_addendum":
        raise ValueError("Unsupported panel revision status")
    return path, revision


def validate_panel_revision(path, protocol, preparation):
    path, revision = _read_revision(path)
    protocol_path = Path(protocol.get("_path", "")).resolve()
    preparation = Path(preparation).resolve()
    base = revision["base_protocol"]
    if (not protocol_path.is_file() or protocol_path != Path(base["path"]).resolve()
            or sha256_file(protocol_path) != base["sha256"]):
        raise ValueError("Panel revision does not bind the active base protocol")
    if (preparation != Path(base["preparation_path"]).resolve()
            or sha256_file(preparation / "preparation_manifest.json") != base["preparation_manifest_sha256"]
            or sha256_file(preparation / "vs_execution_order.json") != base["vs_execution_order_sha256"]):
        raise ValueError("Panel revision does not bind the active frozen preparation/order")
    prefix_ref = revision["historical_prefix_bundle"]
    prefix_dir = Path(prefix_ref["path"])
    prefix = validate_bundle(prefix_dir)
    for path_key, hash_key in (("manifest_path", "manifest_sha256"),
                               ("manifest_lock_path", "manifest_lock_sha256"),
                               ("derived_samples_path", "derived_samples_sha256")):
        if sha256_file(prefix_ref[path_key]) != prefix_ref[hash_key]:
            raise ValueError("Historical-prefix revision source hash mismatch: %s" % path_key)
    mapping = revision["panel_mapping"]
    source = prefix["source_physical_task"]
    target = prefix["new_panel_entry"]
    expected_pairs = {
        "instance_id": target["instance_id"], "date": target["date"], "gain": target["gain"],
        "analysis_reads": target["analysis_reads"], "panel_task_name": target["target_task_name"],
        "matrix_content_sha256": target["target_matrix_content_sha256"],
        "source_task_id": source["task_id"], "source_task_name": source["task_name"],
        "source_requested_reads": source["requested_reads"], "source_returned_reads": source["returned_reads"],
        "source_row_indices": prefix["derived_data"]["source_row_indices"],
        "derived_samples_sha256": prefix["derived_data"]["sha256"],
        "source_response_sha256": source["response_sha256"], "source_receipt_sha256": source["receipt_sha256"],
        "source_matrix_file_sha256": source["hardware_matrix_file_sha256"],
        "energy_convention": source["hardware_energy_convention"],
    }
    if mapping != expected_pairs:
        raise ValueError("Panel revision mapping differs from its source derivation")
    if protocol["hardware"]["reads_per_instance"] != 10:
        raise ValueError("Panel revision is only valid for the 10-read protocol")
    return path, revision, prefix


def _analysis_record(task, revision_path, revision, prefix):
    source = prefix["source_physical_task"]
    target = prefix["new_panel_entry"]
    prep = Path(revision["base_protocol"]["preparation_path"])
    # Only a 10-read projection is used by the decoder/metrics. Actual task
    # counts remain separately named and task_id/task_name remain the originals.
    record = dict(task)
    record.update({
        "record_kind": RECORD_KIND,
        "task_id": source["task_id"],
        "task_name": source["task_name"],
        "platform_task_name": source["platform_task_name"],
        "panel_task_name": target["target_task_name"],
        "requested_reads": int(source["requested_reads"]),
        "returned_reads": int(source["returned_reads"]),
        "analysis_reads": int(target["analysis_reads"]),
        "source_task_id": source["task_id"],
        "source_task_name": source["task_name"],
        "source_requested_reads": int(source["requested_reads"]),
        "source_returned_reads": int(source["returned_reads"]),
        "source_response_path": source["response_path"],
        "source_response_sha256": source["response_sha256"],
        "source_response_audit_path": source["response_audit_path"],
        "source_response_audit_sha256": source["response_audit_sha256"],
        "platform_verification_path": source["receipt_path"],
        "platform_verification_sha256": source["receipt_sha256"],
        "source_platform_record_path": source["platform_record_path"],
        "source_platform_record_sha256": source["platform_record_sha256"],
        "source_screenshot_path": source["screenshot_evidence_path"],
        "source_screenshot_sha256": source["screenshot_evidence_sha256"],
        "source_matrix_path": source["hardware_matrix_path"],
        "source_matrix_file_sha256": source["hardware_matrix_file_sha256"],
        "source_energy_convention": source["hardware_energy_convention"],
        "response_path": revision["historical_prefix_bundle"]["derived_samples_path"],
        "response_file_sha256": revision["historical_prefix_bundle"]["derived_samples_sha256"],
        "hardware_claim": True,
        "physical_platform_used": True,
        "hardware_metadata_complete": True,
        "completed": True,
        "platform_identity_verified": True,
        "platform_response_complete": True,
        "experiment_validated": False,
        "protocol_sha256": revision["base_protocol"]["sha256"],
        "preparation_manifest_sha256": revision["base_protocol"]["preparation_manifest_sha256"],
        "protocol_version": task["protocol_version"],
        "panel_revision_id": revision["revision_id"],
        "panel_revision_path": str(revision_path),
        "panel_revision_sha256": sha256_file(revision_path),
        "source_logical_states_in_all_100": source["distinct_logical_states_in_all_100"],
        "source_state_frequency": source["single_state_frequency"],
        "posthoc_reuse_disclosure": prefix["posthoc_decision_disclosure"],
        "actual_reads_added": 0,
        "analysis_record_only": True,
        "reference_sample_path": task["reference_sample_path"],
        "reference_sample_sha256": task["reference_sample_sha256"],
        "reference_sampler": task["reference_sampler"],
        "hardware_matrix_path": task["hardware_matrix_path"],
        "hardware_matrix_file_sha256": task["hardware_matrix_file_sha256"],
        "source_problem_path": task["source_problem_path"],
        "source_problem_sha256": task["source_problem_sha256"],
        "model_checkpoint_sha256": revision["panel_mapping"].get("target_model_checkpoint_sha256", task.get("model_checkpoint_sha256")),
        "dataset_sha256": revision["panel_mapping"].get("target_dataset_sha256", ""),
        "_historical_panel_validation_token": _VALIDATION_TOKEN,
    })
    return record


def load_historical_panel_records(revision_path, tasks, protocol, preparation,
                                  output_dir=None, intents=None, existing_records=()):
    revision_path, revision, prefix = validate_panel_revision(revision_path, protocol, preparation)
    mapping = revision["panel_mapping"]
    matches = [task for task in tasks
               if str(task.get("instance_id")) == str(mapping["instance_id"])
               and float(task.get("gain", -1)) == float(mapping["gain"])
               and str(task.get("matrix_sha256")) == str(mapping["matrix_content_sha256"])]
    if len(matches) != 1 or int(matches[0].get("reads", -1)) != mapping["analysis_reads"]:
        raise ValueError("Historical prefix does not map to exactly one frozen panel task")
    task = matches[0]
    for record in existing_records:
        if (str(record.get("instance_id")) == str(task["instance_id"])
                and float(record.get("gain", -1)) == float(task["gain"])
                and str(record.get("matrix_sha256")) == str(task["matrix_sha256"])):
            if record.get("record_kind") != RECORD_KIND:
                raise ValueError("A physical/new response conflicts with the historical derived panel entry")
            if (str(record.get("panel_revision_sha256", "")) != sha256_file(revision_path)
                    or str(record.get("response_file_sha256", "")) != mapping.get("derived_samples_sha256", "")):
                raise ValueError("Existing historical derived record conflicts with the frozen revision")
    if intents:
        for _, intent in intents.values():
            if (str(intent.get("instance_id")) == str(task["instance_id"])
                    and float(intent.get("gain", -1)) == float(task["gain"])
                    and str(intent.get("matrix_sha256")) == str(task["matrix_sha256"])):
                raise RuntimeError("A task intent conflicts with the historical prefix; manual review required")
    if output_dir is not None:
        for path in Path(output_dir).rglob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError("Unparseable response record blocks historical reuse: %s" % path) from exc
            for row in ([item] + item.get("records", []) if isinstance(item, dict) else []):
                if (isinstance(row, dict)
                        and str(row.get("instance_id")) == str(task["instance_id"])
                        and float(row.get("gain", -1)) == float(task["gain"])
                        and str(row.get("matrix_sha256")) == str(task["matrix_sha256"])
                        and row.get("record_kind") != RECORD_KIND):
                    raise ValueError("A response/receipt already claims the same condition; manual review required")
    record = _analysis_record(task, revision_path, revision, prefix)
    return [record]


def validate_historical_panel_record(record, expected, require_physical=True):
    """Strictly revalidate a panel projection; never equate source and panel reads."""
    if not isinstance(record, dict) or record.get("record_kind") != RECORD_KIND:
        return False
    if record.get("_historical_panel_validation_token") is _VALIDATION_TOKEN:
        if require_physical and (record.get("hardware_claim") is not True
                                 or record.get("physical_platform_used") is not True):
            return False
        if (record.get("task_id") != record.get("source_task_id")
                or record.get("task_name") != record.get("source_task_name")
                or int(record.get("requested_reads", -1)) != 100
                or int(record.get("returned_reads", -1)) != 100
                or int(record.get("source_requested_reads", -1)) != 100
                or int(record.get("source_returned_reads", -1)) != 100
                or int(record.get("analysis_reads", -1)) != 10):
            return False
        bound_files = (
            (record.get("response_path"), record.get("response_file_sha256")),
            (record.get("source_response_path"), record.get("source_response_sha256")),
            (record.get("source_response_audit_path"), record.get("source_response_audit_sha256")),
            (record.get("platform_verification_path"), record.get("platform_verification_sha256")),
            (record.get("source_platform_record_path"), record.get("source_platform_record_sha256")),
            (record.get("source_screenshot_path"), record.get("source_screenshot_sha256")),
            (record.get("source_matrix_path"), record.get("source_matrix_file_sha256")),
        )
        try:
            if any(not path or not expected_hash or sha256_file(path) != expected_hash
                   for path, expected_hash in bound_files):
                return False
        except (OSError, TypeError, ValueError):
            return False
        for key, value in expected.items():
            record_key = "panel_task_name" if key == "task_name" else key
            if key == "requested_reads":
                if int(record.get("analysis_reads", -1)) != int(value):
                    return False
            elif key == "gain":
                if float(record.get(record_key, -1)) != float(value):
                    return False
            elif record_key not in record or str(record.get(record_key)) != str(value):
                return False
        return True
    try:
        revision_path, revision = _read_revision(record["panel_revision_path"])
        base_protocol_path = revision["base_protocol"]["path"]
        base_preparation = revision["base_protocol"]["preparation_path"]
        protocol_path, protocol, preparation = _frozen_inputs(base_protocol_path, base_preparation)
    except (OSError, KeyError, TypeError, ValueError):
        return False
    try:
        tasks = __import__("GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling", fromlist=["_vs_tasks"])._vs_tasks(preparation, protocol)
        records = load_historical_panel_records(
            revision_path, tasks, protocol, preparation,
            output_dir=preparation / "real_responses", intents=None, existing_records=())
        canonical = records[0]
        identity_fields = (
            "instance_id", "stage", "gain", "matrix_sha256", "analysis_reads",
            "task_id", "task_name", "panel_task_name", "requested_reads", "returned_reads",
            "source_task_id", "source_task_name", "source_requested_reads", "source_returned_reads",
            "response_path", "response_file_sha256", "source_response_path", "source_response_sha256",
            "platform_verification_path", "platform_verification_sha256",
            "source_response_audit_path", "source_response_audit_sha256",
            "source_platform_record_path", "source_platform_record_sha256",
            "source_screenshot_path", "source_screenshot_sha256",
            "source_matrix_path", "source_matrix_file_sha256", "source_energy_convention",
            "reference_sample_path", "reference_sample_sha256", "source_problem_path",
            "source_problem_sha256", "hardware_matrix_path", "hardware_matrix_file_sha256",
            "hardware_energy_convention", "protocol_sha256", "preparation_manifest_sha256",
            "panel_revision_sha256",
        )
        for key in identity_fields:
            left, right = record.get(key), canonical.get(key)
            if isinstance(left, (str, Path)) and isinstance(right, (str, Path)) and key.endswith("_path"):
                if Path(left).resolve() != Path(right).resolve():
                    return False
            elif str(left) != str(right):
                return False
        for key, digest_key in (
            ("response_path", "response_file_sha256"),
            ("source_response_path", "source_response_sha256"),
            ("source_response_audit_path", "source_response_audit_sha256"),
            ("platform_verification_path", "platform_verification_sha256"),
            ("source_platform_record_path", "source_platform_record_sha256"),
            ("source_screenshot_path", "source_screenshot_sha256"),
            ("source_matrix_path", "source_matrix_file_sha256"),
        ):
            if sha256_file(record[key]) != record[digest_key]:
                return False
        for key, value in expected.items():
            if key == "task_name":
                if str(canonical.get("panel_task_name")) != str(value):
                    return False
            elif key == "requested_reads":
                if int(canonical.get("analysis_reads", -1)) != int(value):
                    return False
            elif key == "gain":
                if float(canonical.get(key, -1)) != float(value):
                    return False
            elif key not in canonical or str(canonical.get(key)) != str(value):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-bundle", type=Path, default=DEFAULT_PREFIX_BUNDLE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REVISION_DIR)
    args = parser.parse_args()
    path = create_panel_revision(args.prefix_bundle, args.protocol, args.preparation, args.output_dir)
    print(json.dumps({"status": "panel_revision_frozen", "path": str(path),
                      "sha256": sha256_file(path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
