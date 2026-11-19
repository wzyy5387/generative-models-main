import json
import shutil
from pathlib import Path

import pytest

from GEFcom2014.models.QBM_VAE import run_kaiwu_13k_sampling as driver
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import load_protocol, sha256_file
from GEFcom2014.models.QBM_VAE.kaiwu_panel_revision import (
    load_historical_panel_records,
    validate_panel_revision,
)
from GEFcom2014.models.QBM_VAE.kaiwu_vs_batch_status import inspect_vs_batch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json"
PREPARATION = ROOT / "export/opsd_sa_negative_kaiwu_preparation_10read_v1"
PREFIX = ROOT / "export/opsd_sa_negative_kaiwu_historical_prefix_v3"
REVISION = ROOT / "export/opsd_sa_negative_kaiwu_panel_revision_historical_prefix_v1/panel_revision.json"
SOURCE_TASK_ID = "b1331cc8139e4395a72e2562db61015d"
INSTANCE_ID = "opsd-wind_opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0_vs_000"
MATRIX_SHA256 = "0487ba43a0865f4d721f2ac435fb77eb1ba99638625448c944fee6fcd79d6265"


def _active_inputs():
    protocol_path = PROTOCOL.resolve()
    protocol = load_protocol(protocol_path)
    protocol["_path"] = str(protocol_path)
    return protocol, PREPARATION.resolve()


def _write_lock(path, target_name):
    path.with_suffix(".sha256").write_text(
        sha256_file(path) + "  " + target_name + "\n", encoding="ascii")


def test_revision_is_opt_in_and_counts_historical_prefix_without_faking_reads():
    base = inspect_vs_batch(PROTOCOL, PREPARATION, batch_size=1)
    revised = inspect_vs_batch(PROTOCOL, PREPARATION, batch_size=1,
                               panel_revision=REVISION)

    assert base["counts"]["validated_complete"] == 3
    assert base["counts"]["not_submitted"] == 27
    assert base["remaining_reads"] == 270
    assert not any(row.get("source_task_id") == SOURCE_TASK_ID for row in base["tasks"])

    assert revised["counts"]["validated_complete"] == 4
    assert revised["counts"]["not_submitted"] == 26
    assert revised["panel_analysis_samples_verified"] == 40
    assert revised["historical_source_reads_already_paid"] == 100
    assert revised["new_platform_reads_returned"] == 30
    assert revised["new_platform_reads_remaining"] == 260
    row = next(row for row in revised["tasks"] if row.get("source_task_id") == SOURCE_TASK_ID)
    assert row["response_source"] == "historical 100-read task, original first 10 rows"
    assert row["source_actual_reads"] == [100, 100]
    assert row["analysis_reads"] == 10
    assert row["source_task_id"] == SOURCE_TASK_ID
    assert revised["counts"]["awaiting_platform_verification"] == 0
    assert revised["counts"]["pending_intent_manual_review"] == 0


def test_revision_lock_rejects_changed_revision_and_derived_sample_copy(tmp_path):
    protocol, preparation = _active_inputs()
    revision_copy_dir = tmp_path / "revision"
    shutil.copytree(REVISION.parent, revision_copy_dir)
    changed_revision = revision_copy_dir / "panel_revision.json"
    changed = json.loads(changed_revision.read_text(encoding="utf-8"))
    changed["panel_mapping"]["gain"] = 150
    changed_revision.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_panel_revision(changed_revision, protocol, preparation)

    mismatched_revision_dir = tmp_path / "revision-source"
    shutil.copytree(REVISION.parent, mismatched_revision_dir)
    mismatched_revision = mismatched_revision_dir / "panel_revision.json"
    data = json.loads(mismatched_revision.read_text(encoding="utf-8"))
    data["panel_mapping"]["source_task_id"] = "not-the-platform-task-id"
    mismatched_revision.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _write_lock(mismatched_revision, "panel_revision.json")
    with pytest.raises(ValueError, match="mapping differs"):
        validate_panel_revision(mismatched_revision, protocol, preparation)

    bundle_copy = tmp_path / "prefix"
    shutil.copytree(PREFIX, bundle_copy)
    revision_copy_dir2 = tmp_path / "revision-derived"
    shutil.copytree(REVISION.parent, revision_copy_dir2)
    derived_revision = revision_copy_dir2 / "panel_revision.json"
    data = json.loads(derived_revision.read_text(encoding="utf-8"))
    data["historical_prefix_bundle"].update({
        "path": str(bundle_copy.resolve()),
        "manifest_path": str((bundle_copy / "historical_prefix_amendment.json").resolve()),
        "manifest_lock_path": str((bundle_copy / "historical_prefix_amendment.sha256").resolve()),
        "derived_samples_path": str((bundle_copy / "derived_prefix.npz").resolve()),
    })
    derived_revision.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _write_lock(derived_revision, "panel_revision.json")
    derived_file = bundle_copy / "derived_prefix.npz"
    derived_file.write_bytes(derived_file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Derived prefix file hash mismatch"):
        validate_panel_revision(derived_revision, protocol, preparation)


def test_revision_rejects_missing_receipt_or_screenshot_evidence(tmp_path):
    from GEFcom2014.models.QBM_VAE.derive_historical_kaiwu_prefix import validate_bundle

    for field in ("receipt_path", "screenshot_evidence_path"):
        copied = tmp_path / field
        shutil.copytree(PREFIX, copied)
        manifest_path = copied / "historical_prefix_amendment.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_physical_task"][field] = str(copied / "missing-evidence.bin")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _write_lock(manifest_path, "historical_prefix_amendment.json")
        with pytest.raises((FileNotFoundError, ValueError)):
            validate_bundle(copied)


@pytest.mark.parametrize("source_path_key", [
    "response_path", "receipt_path", "screenshot_evidence_path", "hardware_matrix_path",
])
def test_tampered_original_source_copy_is_rejected_without_touching_evidence(tmp_path, source_path_key):
    from GEFcom2014.models.QBM_VAE.derive_historical_kaiwu_prefix import validate_bundle

    copied = tmp_path / source_path_key
    shutil.copytree(PREFIX, copied)
    manifest_path = copied / "historical_prefix_amendment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = Path(manifest["source_physical_task"][source_path_key])
    replacement = tmp_path / (source_path_key + original.suffix)
    shutil.copy2(original, replacement)
    manifest["source_physical_task"][source_path_key] = str(replacement)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_lock(manifest_path, "historical_prefix_amendment.json")
    replacement.write_bytes(replacement.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle(copied)


def test_existing_response_or_intent_for_historical_slot_stops_panel_loading(tmp_path):
    protocol, preparation = _active_inputs()
    tasks = driver.planned_tasks("vs", preparation, preparation / "real_responses", protocol)
    target = next(task for task in tasks if task["instance_id"] == INSTANCE_ID
                  and float(task["gain"]) == 125)
    physical_duplicate = {
        "instance_id": INSTANCE_ID, "gain": 125, "matrix_sha256": MATRIX_SHA256,
        "task_id": "different-real-task", "record_kind": "physical_response",
    }
    with pytest.raises(ValueError, match="conflicts with the historical derived panel entry"):
        load_historical_panel_records(REVISION, tasks, protocol, preparation,
                                      existing_records=[physical_duplicate])

    intent = {
        "instance_id": INSTANCE_ID, "gain": 125, "matrix_sha256": MATRIX_SHA256,
        "task_name": target["task_name"], "state": "pending",
    }
    with pytest.raises(RuntimeError, match="intent conflicts"):
        load_historical_panel_records(REVISION, tasks, protocol, preparation,
                                      intents={target["task_name"]: (tmp_path / "intent", intent)})


def test_addendum_only_marks_one_slot_and_keeps_next_submission_order_frozen():
    protocol, preparation = _active_inputs()
    tasks = driver.planned_tasks("vs", preparation, preparation / "real_responses", protocol)
    records = load_historical_panel_records(REVISION, tasks, protocol, preparation)
    validated = driver._validate_vs_experiment_responses(tasks, records, preparation, protocol)
    assert len(validated) == 1
    assert validated[0]["task_id"] == SOURCE_TASK_ID
    assert validated[0]["requested_reads"] == 100
    assert validated[0]["returned_reads"] == 100
    assert validated[0]["analysis_reads"] == 10
    # The canonical addendum contributes one panel key; selection then follows
    # the existing frozen order and cannot schedule that condition again.
    complete_keys = {(row["instance_id"], row["matrix_sha256"]) for row in validated}
    pending = driver.select_pending_tasks(tasks, complete_keys, limit=1)
    assert len(pending) == 1
    assert pending[0]["vs_task_order"] == 1
    assert not (pending[0]["instance_id"] == INSTANCE_ID
                and float(pending[0]["gain"]) == 125)


def test_persisted_historical_projection_rechecks_paths_and_source_identity(tmp_path):
    from GEFcom2014.models.QBM_VAE.kaiwu_panel_revision import validate_historical_panel_record

    protocol, preparation = _active_inputs()
    tasks = driver.planned_tasks("vs", preparation, preparation / "real_responses", protocol)
    task = next(task for task in tasks if task["instance_id"] == INSTANCE_ID
                and float(task["gain"]) == 125)
    canonical = load_historical_panel_records(REVISION, tasks, protocol, preparation)[0]
    persisted = {key: value for key, value in canonical.items() if not key.startswith("_")}
    expected = driver._task_expected(task)
    expected.update({"requested_reads": 10, "task_name": task["task_name"]})
    assert validate_historical_panel_record(persisted, expected)

    copied_samples = tmp_path / "same-bytes-different-source.npz"
    shutil.copy2(persisted["response_path"], copied_samples)
    changed_path = dict(persisted, response_path=str(copied_samples))
    assert not validate_historical_panel_record(changed_path, expected)
    changed_source = dict(persisted, source_task_id="invented-task")
    assert not validate_historical_panel_record(changed_source, expected)


def test_restore_validated_panel_entries_never_calls_submit_and_updates_only_temp_intents(tmp_path, monkeypatch):
    output_copy = tmp_path / "real_responses"
    checkpoint_copy = tmp_path / "checkpoints"
    shutil.copytree(PREPARATION / "real_responses", output_copy)
    shutil.copytree(PREPARATION / "checkpoints", checkpoint_copy)
    submit_calls = []
    validation_calls = []
    monkeypatch.setattr(driver, "submit_one", lambda *args, **kwargs: submit_calls.append((args, kwargs)))
    actual_validator = driver._validate_vs_experiment_responses

    def observe_validator(tasks, *args, **kwargs):
        validated = actual_validator(tasks, *args, **kwargs)
        validation_calls.append((len(tasks), len(validated)))
        return validated

    monkeypatch.setattr(driver, "_validate_vs_experiment_responses", observe_validator)
    args = type("Args", (), {
        "stage": "vs", "protocol": PROTOCOL, "preparation": PREPARATION,
        "output_dir": output_copy, "checkpoint_dir": checkpoint_copy,
        "execute": True, "dry_run": False, "limit": 1, "resume": True,
        "vs_order_manifest": PREPARATION / "vs_execution_order.json",
        "panel_revision": REVISION, "historical_prefix_amendment": PREFIX,
        "canary": True, "confirm_canary": True, "confirm_batch": False,
        "canary_instance_id": INSTANCE_ID, "canary_gain": 125,
        "canary_matrix_sha256": MATRIX_SHA256,
        "confirm_real_submission": True, "confirm_test": False, "project_no": None,
    })()

    result = driver.run(args)
    assert result["completed"] == 1
    assert result["skipped_validated"] == 1
    assert result["returned_reads"] == 100  # actual source count remains honest
    assert validation_calls == [(30, 4)]
    assert submit_calls == []
    temp_intents = list(checkpoint_copy.glob("*.intent.json"))
    assert len(temp_intents) == 3
    assert all(json.loads(path.read_text(encoding="utf-8"))["state"] == "completed"
               for path in temp_intents)
    formal_intents = list((PREPARATION / "checkpoints").glob("*.intent.json"))
    assert all(json.loads(path.read_text(encoding="utf-8"))["state"] == "pending"
               for path in formal_intents)
