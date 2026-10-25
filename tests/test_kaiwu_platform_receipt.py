import hashlib
import json

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.kaiwu_adapter import matrix_sha256
from GEFcom2014.models.QBM_VAE.kaiwu_platform_receipt import (
    create_platform_verification,
    validate_platform_verification,
    _same_instant,
)
from GEFcom2014.models.QBM_VAE.import_hardware_responses import strict_response_record_valid
from GEFcom2014.models.QBM_VAE.import_hardware_responses import import_responses


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path, identifier_field="task_id", platform_submitted_at="2026-01-01T00:00:00+00:00"):
    instance = "vs_frozen_000"
    task_name = "official_task_name_000"
    matrix = np.zeros((49, 49), dtype=np.int8)
    matrix[0, 1] = matrix[1, 0] = 2
    matrix_path = tmp_path / "matrix.npz"
    np.savez(matrix_path, hardware_matrix=matrix)
    matrix_hash = matrix_sha256(matrix)
    samples = np.ones((4, 49), dtype=np.int8)
    raw_path = tmp_path / (instance + ".npz")
    np.savez(
        raw_path, samples=samples, instance_id=instance, task_id="",
        task_name=task_name, matrix_sha256=matrix_hash, requested_reads=4,
        returned_reads=4, stage="vs", hardware_gain=125.0,
        submitted_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-01T00:00:01+00:00",
        sdk_version="1.0.7", kaiwu_distribution_version="1.4.1",
        kaiwu_module_version="1.0.7", platform_backend="kaiwu.cim.CIMOptimizer",
    )
    record = {
        identifier_field: "platform-92731", "task_name": task_name,
        "matrix_sha256": matrix_hash, "status": "completed",
        "requested_reads": 4, "returned_reads": 4,
        "submitted_at": platform_submitted_at, "completed_at": "2026-01-01T00:01:00+00:00",
        "platform_backend": "kaiwu.cim.CIMOptimizer",
    }
    record_path = tmp_path / "platform_record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    evidence_path = tmp_path / "platform_evidence.txt"
    evidence_path.write_text("operator reviewed platform task entry", encoding="utf-8")
    return instance, task_name, matrix_hash, matrix_path, raw_path, record_path, evidence_path


@pytest.mark.parametrize("identifier_field", ["task_id", "job_id"])
def test_manual_platform_receipt_binds_id_matrix_response_and_preserves_raw(tmp_path, identifier_field):
    instance, task_name, matrix_hash, matrix, raw, platform_record, evidence = _artifacts(
        tmp_path, identifier_field=identifier_field,
        platform_submitted_at="2026-01-01T00:00:05Z")
    original_hash = _sha(raw)
    receipt_path = tmp_path / (instance + ".platform_verification.json")
    _, receipt = create_platform_verification(
        raw, matrix, platform_record, evidence, "reviewer-1", receipt_path)
    assert _sha(raw) == original_hash
    assert receipt["task_id"] == "platform-92731"
    assert receipt["platform_identifier_field"] == identifier_field
    assert receipt["platform_identifier_value"] == "platform-92731"
    assert receipt["client_called_at"] == "2026-01-01T00:00:00+00:00"
    assert receipt["platform_submitted_at"] == "2026-01-01T00:00:05Z"
    assert receipt["task_name"] == task_name
    assert receipt["matrix_sha256"] == matrix_hash
    assert receipt["raw_response_sha256"] == original_hash
    checked = validate_platform_verification(raw, receipt_path, {
        "instance_id": instance, "task_id": "platform-92731",
        "task_name": task_name, "matrix_sha256": matrix_hash,
        "stage": "vs", "gain": 125, "requested_reads": 4, "returned_reads": 4,
    })
    record = {
        "completed": True, "hardware_claim": True, "physical_platform_used": True,
        "hardware_metadata_complete": True,
        "instance_id": instance, "task_id": checked["task_id"],
        "task_name": task_name, "matrix_sha256": matrix_hash,
        "response_file_sha256": original_hash, "response_path": str(raw),
        "platform_verification_path": str(receipt_path),
        "platform_verification_sha256": _sha(receipt_path),
        "stage": "vs", "gain": 125, "requested_reads": 4, "returned_reads": 4,
        "sdk_version": "1.0.7", "sdk_versions": {
            "importlib_metadata_version": "1.4.1", "kaiwu_module_version": "1.0.7",
        }, "platform_backend": "kaiwu.cim.CIMOptimizer",
        "submitted_at": checked["submitted_at"], "completed_at": checked["completed_at"],
        "client_called_at": checked["client_called_at"],
        "client_returned_at": checked["client_returned_at"],
        "platform_submitted_at": checked["platform_submitted_at"],
        "platform_completed_at": checked["platform_completed_at"],
        "platform_identifier_field": checked["platform_identifier_field"],
        "platform_identifier_value": checked["platform_identifier_value"],
    }
    assert strict_response_record_valid(record)


def test_platform_receipt_rejects_wrong_matrix_or_response_and_never_overwrites(tmp_path):
    instance, _, _, matrix, raw, record, evidence = _artifacts(tmp_path)
    receipt_path = tmp_path / (instance + ".platform_verification.json")
    create_platform_verification(raw, matrix, record, evidence, "reviewer-1", receipt_path)
    with np.load(raw, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["samples"] = np.zeros((4, 49), dtype=np.int8)
    changed = tmp_path / "changed.npz"
    np.savez(changed, **arrays)
    with pytest.raises(ValueError, match="changed after"):
        validate_platform_verification(changed, receipt_path)
    with pytest.raises(FileExistsError):
        create_platform_verification(raw, matrix, record, evidence, "reviewer-1", receipt_path)


def test_platform_receipt_requires_real_platform_id_and_matching_task_name(tmp_path):
    instance, _, _, matrix, raw, record_path, evidence = _artifacts(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["task_name"] = "pretend-id"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="task_name"):
        create_platform_verification(raw, matrix, record_path, evidence, "reviewer-1",
                                     tmp_path / "out.json")
    record.pop("task_id")
    record.pop("job_id", None)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="real task_id or job_id"):
        create_platform_verification(raw, matrix, record_path, evidence, "reviewer-1",
                                     tmp_path / "out.json")


def test_platform_receipt_rejects_identifier_mismatch(tmp_path):
    instance, _, _, matrix, raw, record, evidence = _artifacts(tmp_path, identifier_field="job_id")
    receipt_path = tmp_path / (instance + ".platform_verification.json")
    create_platform_verification(raw, matrix, record, evidence, "reviewer-1", receipt_path)
    with pytest.raises(ValueError, match="task_id mismatch"):
        validate_platform_verification(raw, receipt_path, {"task_id": "different-platform-id"})


def test_platform_receipt_rejects_conflicting_task_and_job_ids(tmp_path):
    instance, _, _, matrix, raw, record_path, evidence = _artifacts(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["job_id"] = "different-platform-id"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="task_id and job_id disagree"):
        create_platform_verification(raw, matrix, record_path, evidence, "reviewer-1",
                                     tmp_path / (instance + ".verification.json"))


def test_platform_and_client_event_times_are_distinct_and_offsets_parse(tmp_path):
    instance, _, _, matrix, raw, record, evidence = _artifacts(
        tmp_path, platform_submitted_at="2026-01-01T00:00:05Z")
    receipt_path = tmp_path / (instance + ".platform_verification.json")
    create_platform_verification(raw, matrix, record, evidence, "reviewer-1", receipt_path)
    receipt = validate_platform_verification(raw, receipt_path)
    assert receipt["client_called_at"] == "2026-01-01T00:00:00+00:00"
    assert receipt["platform_submitted_at"] == "2026-01-01T00:00:05Z"
    assert "cross-system event equality is not assumed" in receipt["platform_time_semantics"]
    assert _same_instant("2026-01-01T00:00:00Z", "2026-01-01T00:00:00+00:00", "test_time")
    with pytest.raises(ValueError, match="timezone offset"):
        _same_instant("2026-01-01T00:00:00", "2026-01-01T00:00:00Z", "test_time")


@pytest.mark.parametrize("identifier_field", ["task_id", "job_id"])
def test_verified_sidecar_enters_formal_import_without_rewriting_raw_response(tmp_path, identifier_field):
    instance, task_name, matrix_hash, matrix_path, raw, record_path, evidence = _artifacts(
        tmp_path, identifier_field=identifier_field)
    input_dir = tmp_path / "raw_inputs"
    input_dir.mkdir()
    raw_in_input = input_dir / raw.name
    raw_in_input.write_bytes(raw.read_bytes())
    original_hash = _sha(raw_in_input)
    verification = input_dir / (instance + ".platform_verification.json")
    create_platform_verification(
        raw_in_input, matrix_path, record_path, evidence, "reviewer-1", verification)

    package = tmp_path / "package"
    (package / "hardware_matrices").mkdir(parents=True)
    (package / "payloads").mkdir()
    packaged_matrix = package / "hardware_matrices" / "matrix.npz"
    packaged_matrix.write_bytes(matrix_path.read_bytes())
    payload_path = package / "payloads" / "instance.json"
    payload_path.write_text(json.dumps({"task_name": task_name}), encoding="utf-8")
    manifest = {
        "stage": "vs", "requested_reads_per_instance": 4,
        "instances": [{
            "instance_id": instance, "split": "VS", "hardware_n_bits": 49,
            "logical_n_bits": 48, "n_bits": 49, "auxiliary_spin_index": 48,
            "hardware_gain": 125.0, "hardware_matrix_sha256": matrix_hash,
            "hardware_matrix": "hardware_matrices/matrix.npz",
            "hardware_matrix_sha256_file": _sha(packaged_matrix),
            "payload": "payloads/instance.json", "payload_sha256": _sha(payload_path),
        }],
    }
    manifest_path = package / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit_path, audit = import_responses(manifest_path, input_dir, tmp_path / "formal_import")
    assert audit["hardware_claim"] is True
    assert _sha(raw_in_input) == original_hash
    record = audit["records"][0]
    assert record["task_id"] == "platform-92731"
    assert record["platform_verification_path"] == str(verification.resolve())
    assert strict_response_record_valid(record)
    assert json.loads(audit_path.read_text(encoding="utf-8"))["hardware_claim"] is True
    canonical_hash = _sha(tmp_path / "formal_import" / (instance + ".npz"))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        import_responses(manifest_path, input_dir, tmp_path / "formal_import")
    assert _sha(tmp_path / "formal_import" / (instance + ".npz")) == canonical_hash


@pytest.mark.parametrize("identifier_field", ["task_id", "job_id"])
def test_verified_receipt_is_used_on_resume_without_rewriting_raw_audit(tmp_path, identifier_field):
    from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import _load_audit_records, _load_completed
    instance, task_name, matrix_hash, matrix, raw, platform_record, evidence = _artifacts(
        tmp_path, identifier_field=identifier_field)
    verification = tmp_path / (instance + ".platform_verification.json")
    create_platform_verification(raw, matrix, platform_record, evidence, "reviewer-1", verification)
    raw_hash = _sha(raw)
    audit = {
        "instance_id": instance, "task_id": "", "task_name": task_name,
        "matrix_sha256": matrix_hash, "response_file_sha256": raw_hash,
        "response_path": str(raw), "stage": "vs", "gain": 125,
        "requested_reads": 4, "returned_reads": 4, "completed": False,
        "hardware_claim": False, "physical_platform_used": False,
        "hardware_metadata_complete": False, "sdk_version": "1.0.7",
        "sdk_versions": {"importlib_metadata_version": "1.4.1", "kaiwu_module_version": "1.0.7"},
        "platform_backend": "kaiwu.cim.CIMOptimizer",
        "submitted_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "protocol_version": "frozen-v1", "protocol_sha256": "a" * 64,
        "model_checkpoint_sha256": "b" * 64, "preparation_manifest_sha256": "c" * 64,
        "hardware_matrix_path": str(matrix), "hardware_matrix_file_sha256": _sha(matrix),
        "source_problem_path": str(matrix), "source_problem_sha256": "d" * 64,
        "reference_sample_path": str(matrix), "reference_sample_sha256": "e" * 64,
        "reference_sampler": {"backend": "sa"},
    }
    original_audit = json.dumps(audit, indent=2)
    (tmp_path / (instance + ".json")).write_text(original_audit, encoding="utf-8")
    records = _load_audit_records(tmp_path)
    promoted = next(row for row in records if row.get("response_path") == str(raw))
    assert promoted["task_id"] == "platform-92731"
    assert promoted["platform_identifier_field"] == identifier_field
    assert promoted["hardware_claim"] is True
    assert strict_response_record_valid(promoted)
    task = {key: audit[key] for key in (
        "instance_id", "matrix_sha256", "gain", "requested_reads", "task_name",
        "protocol_version", "protocol_sha256", "model_checkpoint_sha256",
        "preparation_manifest_sha256", "hardware_matrix_path",
        "hardware_matrix_file_sha256", "source_problem_path", "source_problem_sha256",
        "reference_sample_path", "reference_sample_sha256", "reference_sampler")}
    assert _load_completed(tmp_path, [task]) == {(instance, matrix_hash)}
    assert _sha(raw) == raw_hash
    assert (tmp_path / (instance + ".json")).read_text(encoding="utf-8") == original_audit
