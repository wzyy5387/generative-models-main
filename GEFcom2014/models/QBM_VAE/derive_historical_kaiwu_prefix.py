"""Create an auditable first-10 prefix from a validated historical 100-read task.

This emits a derived analysis artifact and a protocol-amendment record. It does
not convert the historical task into a new 10-read physical submission.
"""

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.models import scale_data_multi
from .kaiwu_13k_protocol import load_protocol, scenario_metrics, sha256_file
from .kaiwu_adapter import normalize_hardware_spins, matrix_sha256
from .opsd_vs_selection import decode_latent_samples_with_paired_noise
from .run_kaiwu_13k_sampling import (
    _load_audit_records,
    _validate_preparation_identity,
    _validate_vs_experiment_responses,
    _vs_tasks,
)


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_INSTANCE = "opsd-wind_opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0_vs_000"
EXPECTED_TASK_ID = "b1331cc8139e4395a72e2562db61015d"
ENERGY_CONVENTION = "kaiwu_sdk_hamiltonian_matches_fa_bm_vae_energy_v1"
DEFAULT_OUTPUT = ROOT / "export/opsd_sa_negative_kaiwu_historical_prefix_v3"


def _load_frozen(protocol_path, preparation_path):
    protocol_path = Path(protocol_path).resolve()
    preparation_path = Path(preparation_path).resolve()
    protocol = load_protocol(protocol_path)
    protocol["_path"] = str(protocol_path)
    _validate_preparation_identity(preparation_path, protocol)
    return protocol, preparation_path, _vs_tasks(preparation_path, protocol)


def _one_task(tasks, reads):
    matches = [task for task in tasks
               if task.get("instance_id") == EXPECTED_INSTANCE
               and float(task.get("gain", -1)) == 125.0
               and int(task.get("reads", -1)) == reads]
    if len(matches) != 1:
        raise ValueError("Frozen task identity does not resolve uniquely")
    return matches[0]


def assert_matching_task_identity(source, target):
    for field in ("instance_id", "date", "gain", "matrix_sha256",
                  "hardware_energy_convention", "source_problem_sha256",
                  "hardware_matrix_file_sha256"):
        if source.get(field) != target.get(field):
            raise ValueError("Historical source differs from new panel task: %s" % field)
    if source.get("hardware_energy_convention") != ENERGY_CONVENTION:
        raise ValueError("Historical source uses a different energy convention")


def _assert_matching_source_and_target(source, target, source_preparation,
                                       target_preparation, source_protocol,
                                       target_protocol):
    assert_matching_task_identity(source, target)
    source_prep = json.loads((source_preparation / "preparation_manifest.json").read_text(encoding="utf-8"))
    target_prep = json.loads((target_preparation / "preparation_manifest.json").read_text(encoding="utf-8"))
    for field in ("model_checkpoint_sha256", "dataset_sha256"):
        if source_prep.get(field) != target_prep.get(field):
            raise ValueError("Historical source differs from new panel frozen input: %s" % field)
    if source_protocol.get("model") != target_protocol.get("model"):
        raise ValueError("Historical source and new panel model configurations differ")
    if source_protocol.get("dataset") != target_protocol.get("dataset"):
        raise ValueError("Historical source and new panel dataset definitions differ")
    source_matrix = Path(source["hardware_matrix_path"])
    target_matrix = Path(target["hardware_matrix_path"])
    if not source_matrix.is_file() or not target_matrix.is_file():
        raise FileNotFoundError("Frozen source or target matrix is missing")
    if (sha256_file(source_matrix) != source["hardware_matrix_file_sha256"]
            or sha256_file(target_matrix) != target["hardware_matrix_file_sha256"]
            or sha256_file(source_matrix) != sha256_file(target_matrix)):
        raise ValueError("Frozen source and target matrix files differ")
    with np.load(source_matrix, allow_pickle=False) as source_npz:
        sm = np.asarray(source_npz["hardware_matrix"])
    with np.load(target_matrix, allow_pickle=False) as target_npz:
        tm = np.asarray(target_npz["hardware_matrix"])
    if (sm.shape != (49, 49) or not np.array_equal(sm, tm)
            or matrix_sha256(sm) != target["matrix_sha256"]):
        raise ValueError("Frozen source and target matrix contents differ")
    source_problem = Path(source["source_problem_path"])
    target_problem = Path(target["source_problem_path"])
    if sha256_file(source_problem) != source["source_problem_sha256"]:
        raise ValueError("Historical logical h/J and condition source hash mismatch")
    if sha256_file(target_problem) != target["source_problem_sha256"]:
        raise ValueError("New panel logical h/J and condition source hash mismatch")
    with np.load(source_problem, allow_pickle=False) as a, np.load(target_problem, allow_pickle=False) as b:
        if not np.array_equal(np.asarray(a["x_cond"]), np.asarray(b["x_cond"])):
            raise ValueError("Historical and new panel condition inputs differ")


def _validated_source(source_protocol_path, source_preparation, source_tasks):
    response_dir = source_preparation / "real_responses"
    records = _load_audit_records(response_dir)
    matches = [record for record in records
               if record.get("task_id") == EXPECTED_TASK_ID
               and record.get("instance_id") == EXPECTED_INSTANCE
               and float(record.get("gain", -1)) == 125.0
               and record.get("status") != "awaiting_platform_verification"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one receipt-verified historical source task")
    source_task = _one_task(source_tasks, 100)
    if matches[0].get("task_name") != source_task.get("task_name"):
        raise ValueError("Historical platform task name differs from frozen v9 task")
    if int(matches[0].get("requested_reads", -1)) != 100 or int(matches[0].get("returned_reads", -1)) != 100:
        raise ValueError("Historical platform task is not recorded as 100/100 reads")
    source_protocol = load_protocol(source_protocol_path)
    source_protocol["_path"] = str(Path(source_protocol_path).resolve())
    validated = _validate_vs_experiment_responses(
        [source_task], records, source_preparation,
        source_protocol, require_physical=True)
    return source_task, validated[0]


def _verify_hash(path, expected_hash, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("%s file is missing: %s" % (description, path))
    if sha256_file(path) != str(expected_hash):
        raise ValueError("%s hash mismatch: %s" % (description, path))
    return path


def _replay_prediction(prefix49, target_task, target_protocol, target_preparation,
                       stored_replay_path):
    data_path = Path(target_protocol["dataset"]["source"])
    with np.load(data_path, allow_pickle=False) as data:
        x_ls, y_ls = np.asarray(data["x_ls"]), np.asarray(data["y_ls"])
        x_vs, y_vs = np.asarray(data["x_vs"]), np.asarray(data["y_vs"])
    _, _, x_scaled, _, _, _, scaler = scale_data_multi(x_ls, y_ls, x_vs, y_vs)
    conditions = np.concatenate((x_scaled, scaler.transform(x_vs[:, :24])), axis=1).astype(np.float32)
    index, offset = int(target_task["index"]), int(target_task["export_index"])
    condition = np.asarray(conditions[index], dtype=np.float32)
    with np.load(target_task["source_problem_path"], allow_pickle=False) as problem:
        frozen_condition = np.asarray(problem["x_cond"], dtype=np.float32)
    if condition.shape != frozen_condition.shape or not np.allclose(condition, frozen_condition, rtol=0, atol=1e-6):
        raise ValueError("Frozen target date condition does not match the dataset row")
    with Path(target_protocol["model"]["checkpoint"]).open("rb") as handle:
        model = pickle.load(handle)
    model.device = torch.device("cpu")
    model.to("cpu")
    model.eval()
    curves = decode_latent_samples_with_paired_noise(
        model, normalize_hardware_spins(prefix49, logical_n_bits=48), condition,
        scaler, prediction_curves=100,
        seed=int(target_protocol["gain_selection"]["prediction_evaluation"]["noise_seed"]) + offset,
    )
    if curves.shape != (100, 24) or not np.isfinite(curves).all():
        raise ValueError("Paired decoder did not produce finite 100x24 scenarios")
    with np.load(stored_replay_path, allow_pickle=False) as saved:
        baseline = np.asarray(saved["original_real_100read_curves"])
        replay10 = np.asarray(saved["real_10read_curves"])
        target = np.asarray(saved["target_vs"])
    if baseline.shape != (100, 24) or replay10.shape != (100, 24) or target.shape != (24,):
        raise ValueError("Stored historical replay has unexpected shapes")
    baseline_error = float(np.max(np.abs(curves - baseline)))
    replay_error = float(np.max(np.abs(curves - replay10)))
    metrics = scenario_metrics(curves, y_vs[index])
    if not np.allclose(y_vs[index], target, rtol=0, atol=0):
        raise ValueError("Stored paired replay target differs from frozen VS observation")
    expected = float(metrics["CRPS"])
    if baseline_error > 1e-7 or replay_error > 1e-7:
        raise ValueError("Historical prefix prediction replay exceeds numeric tolerance")
    if abs(expected - 0.012927510064) > 1e-10:
        raise ValueError("Historical prefix Raw CRPS does not reproduce the stored value")
    return curves, {"raw_crps": expected, "max_abs_error_vs_original_100read_curves": baseline_error,
                    "max_abs_error_vs_existing_10read_replay": replay_error,
                    "noise_seed": int(target_protocol["gain_selection"]["prediction_evaluation"]["noise_seed"]) + offset,
                    "noise_rows": 100, "postprocessing": "raw_clip_0_1"}


def verify_prefix_arrays(parent, derived, row_indices):
    parent = np.asarray(parent)
    derived = np.asarray(derived)
    row_indices = np.asarray(row_indices)
    if parent.shape != (100, 49) or derived.shape != (10, 49):
        raise ValueError("Historical parent and derived arrays must be 100x49 and 10x49")
    if not np.array_equal(row_indices, np.arange(10, dtype=np.int64)):
        raise ValueError("Derived row order must be the original rows 0 through 9")
    if not np.array_equal(derived, parent[:10]):
        raise ValueError("Derived spins are not the exact original-order first ten rows")
    if not np.isin(parent, (-1, 1)).all():
        raise ValueError("Historical raw response contains invalid spins")
    logical = normalize_hardware_spins(parent, logical_n_bits=48)
    unique_count = np.unique(logical, axis=0).shape[0]
    if unique_count != 1:
        raise ValueError("Post-hoc prefix reuse criterion is not met: 100 logical states are not identical")
    return logical, unique_count


def build_historical_prefix(source_protocol_path, source_preparation,
                            target_protocol_path, target_preparation, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite historical-prefix output: %s" % output_dir)
    source_protocol, source_preparation, source_tasks = _load_frozen(
        source_protocol_path, source_preparation)
    target_protocol, target_preparation, target_tasks = _load_frozen(
        target_protocol_path, target_preparation)
    source_task, source_record = _validated_source(
        source_protocol_path, source_preparation, source_tasks)
    target_task = _one_task(target_tasks, 10)
    _assert_matching_source_and_target(
        source_task, target_task, source_preparation, target_preparation,
        source_protocol, target_protocol)
    response_path = Path(source_record["response_path"])
    receipt_path = Path(source_record["platform_verification_path"])
    response_sha = sha256_file(response_path)
    receipt_sha = sha256_file(receipt_path)
    if response_sha != source_record["response_file_sha256"]:
        raise ValueError("Historical raw response hash mismatch")
    if receipt_sha != source_record["platform_verification_sha256"]:
        raise ValueError("Historical platform receipt hash mismatch")
    with np.load(response_path, allow_pickle=False) as archive:
        parent = np.asarray(archive["samples"])
    logical, unique_states = verify_prefix_arrays(parent, parent[:10].copy(), np.arange(10))
    distinct, counts = np.unique(logical, axis=0, return_counts=True)
    matrix_path = Path(target_task["hardware_matrix_path"])
    replay_path = target_preparation / "offline_replay_probe_2019-01-06_gain_125" / "paired_curves.npz"
    curves, replay_metrics = _replay_prediction(parent[:10], target_task, target_protocol,
                                                target_preparation, replay_path)
    from .kaiwu_platform_receipt import validate_platform_verification
    receipt = validate_platform_verification(response_path, receipt_path, {
        "instance_id": EXPECTED_INSTANCE, "task_id": EXPECTED_TASK_ID,
        "task_name": source_task["task_name"], "matrix_sha256": target_task["matrix_sha256"],
        "stage": "vs", "gain": 125, "requested_reads": 100, "returned_reads": 100,
    })
    if receipt.get("verified_by") != "wsy":
        raise ValueError("Historical receipt reviewer is not the specified wsy")
    platform_record_path = Path(receipt["platform_record_path"])
    evidence_path = Path(receipt["evidence_path"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaiwu-prefix-", dir=output_dir.parent) as temp_name:
        temp = Path(temp_name)
        derived_path = temp / "derived_prefix.npz"
        np.savez_compressed(derived_path,
                            samples=np.asarray(parent[:10], dtype=np.int8),
                            source_row_indices=np.arange(10, dtype=np.int64))
        derived_sha = sha256_file(derived_path)
        derived_samples_hash = hashlib.sha256(np.asarray(parent[:10], dtype=np.int8).tobytes(order="C")).hexdigest()
        replay_hash = sha256_file(replay_path)
        manifest = {
            "schema_version": 1,
            "artifact_role": "derived_analysis_prefix_from_historical_physical_task",
            "panel_amendment_status": "applied_posthoc_addendum; base protocol remains unchanged",
            "base_protocol": {
                "path": str(Path(target_protocol_path).resolve()),
                "sha256": sha256_file(target_protocol_path),
                "preparation_manifest_path": str((target_preparation / "preparation_manifest.json").resolve()),
                "preparation_manifest_sha256": sha256_file(target_preparation / "preparation_manifest.json"),
                "vs_execution_order_sha256": sha256_file(target_preparation / "vs_execution_order.json"),
            },
            "source_protocol": {
                "path": str(Path(source_protocol_path).resolve()),
                "sha256": sha256_file(source_protocol_path),
                "preparation_manifest_path": str((source_preparation / "preparation_manifest.json").resolve()),
                "preparation_manifest_sha256": sha256_file(source_preparation / "preparation_manifest.json"),
            },
            "source_physical_task": {
                "task_id": EXPECTED_TASK_ID,
                "task_name": source_record["task_name"],
                "platform_task_name": receipt["platform_task_name"],
                "instance_id": EXPECTED_INSTANCE,
                "date": str(source_task["date"]),
                "gain": 125,
                "requested_reads": 100,
                "returned_reads": 100,
                "response_path": str(response_path.resolve()),
                "response_sha256": response_sha,
                "response_audit_path": str(Path(source_record["audit_path"]).resolve()),
                "response_audit_sha256": sha256_file(source_record["audit_path"]),
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": receipt_sha,
                "platform_record_path": str(platform_record_path.resolve()),
                "platform_record_sha256": receipt["platform_record_sha256"],
                "screenshot_evidence_path": str(evidence_path.resolve()),
                "screenshot_evidence_sha256": receipt["evidence_sha256"],
                "verified_by": "wsy",
                "hardware_matrix_path": str(Path(source_task["hardware_matrix_path"]).resolve()),
                "hardware_matrix_file_sha256": source_task["hardware_matrix_file_sha256"],
                "matrix_content_sha256": source_task["matrix_sha256"],
                "hardware_energy_convention": source_task["hardware_energy_convention"],
                "distinct_logical_states_in_all_100": int(unique_states),
                "single_state_frequency": int(counts.max()),
            },
            "new_panel_entry": {
                "instance_id": EXPECTED_INSTANCE,
                "date": str(target_task["date"]),
                "gain": 125,
                "target_task_name": target_task["task_name"],
                "target_matrix_content_sha256": target_task["matrix_sha256"],
                "target_hardware_matrix_file_sha256": target_task["hardware_matrix_file_sha256"],
                "target_hardware_matrix_path": str(Path(target_task["hardware_matrix_path"]).resolve()),
                "target_source_problem_path": str(Path(target_task["source_problem_path"]).resolve()),
                "target_source_problem_sha256": target_task["source_problem_sha256"],
                "target_model_checkpoint_sha256": target_task["model_checkpoint_sha256"],
                "target_dataset_sha256": json.loads((target_preparation / "preparation_manifest.json").read_text(encoding="utf-8"))["dataset_sha256"],
                "analysis_reads": 10,
                "source_row_indices": list(range(10)),
                "target_protocol_sha256": sha256_file(target_protocol_path),
                "target_preparation_manifest_sha256": sha256_file(target_preparation / "preparation_manifest.json"),
                "countable_under_unamended_base_protocol": False,
            },
            "posthoc_decision_disclosure": {
                "decision_timing": "Reuse was decided after observing that all 100 historical returned reads normalize to the same logical state.",
                "selection_rule": "Use exactly original row order 0..9; no outcome-based selection within the prefix.",
                "interpretation": "Derived analysis prefix from a 100/100 physical task; not a new 10-read submission and not a new platform task ID.",
            },
            "derived_data": {
                "path": "derived_prefix.npz",
                "sha256": derived_sha,
                "samples_shape": [10, 49],
                "samples_dtype": "int8",
                "samples_sha256": derived_samples_hash,
                "source_row_indices": list(range(10)),
                "physical_reads_added": 0,
            },
            "prediction_replay": {
                **replay_metrics,
                "stored_replay_curves_sha256": replay_hash,
                "paired_curves_path": str(replay_path.resolve()),
                "paired_curves_max_abs_error": replay_metrics["max_abs_error_vs_existing_10read_replay"],
            },
            "formal_panel_effect": {
                "unamended_panel_verified_tasks": 3,
                "unamended_panel_remaining_tasks": 27,
                "amendment_required_to_count_prefix": True,
                "no_change_to_existing_protocol_or_manifests": True,
            },
        }
        manifest_path = temp / "historical_prefix_amendment.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lock_path = temp / "historical_prefix_amendment.sha256"
        lock_path.write_text(sha256_file(manifest_path) + "  historical_prefix_amendment.json\n", encoding="ascii")
        (temp / "REPORT.md").write_text(
            "# Historical 100-read response prefix\n\n"
            "The first 10 raw rows were derived from a validated 100/100 physical task. The reuse decision was made after observing that all 100 reads normalize to the same logical state. The source task ID and actual request/return counts remain unchanged; no new platform task ID or read was created.\n\n"
            "Date, instance, gain, model, dataset, frozen condition, matrix contents and energy convention were checked against the target 10-read entry. The paired-noise decoder replay reproduces Raw CRPS 0.012927510063860267 and the saved 100×24 curves within the recorded tolerance.\n\n"
            "The unamended 10-read protocol excludes the historical 100-read task. This bundle is a separately hashed post-hoc amendment artifact; the normal panel runner does not consume it. Formal panel progress therefore remains 3 verified and 27 remaining until a reviewed protocol revision and runner support are added. Do not resubmit this target condition before that reconciliation is implemented.\n",
            encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Refusing to overwrite historical-prefix output: %s" % output_dir)
        validate_bundle(temp)
        os.replace(temp, output_dir)
    return json.loads((output_dir / "historical_prefix_amendment.json").read_text(encoding="utf-8"))


def validate_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "historical_prefix_amendment.json"
    lock_path = bundle_dir / "historical_prefix_amendment.sha256"
    expected_manifest_sha = lock_path.read_text(encoding="ascii").split()[0]
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("Historical-prefix amendment manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source_physical_task"]
    target = manifest["new_panel_entry"]
    source_protocol, source_preparation, source_tasks = _load_frozen(
        manifest["source_protocol"]["path"], Path(manifest["source_protocol"]["preparation_manifest_path"]).parent)
    target_protocol, target_preparation, target_tasks = _load_frozen(
        manifest["base_protocol"]["path"], Path(manifest["base_protocol"]["preparation_manifest_path"]).parent)
    source_task, source_record = _validated_source(
        manifest["source_protocol"]["path"], source_preparation, source_tasks)
    target_task = _one_task(target_tasks, 10)
    _assert_matching_source_and_target(source_task, target_task, source_preparation,
                                       target_preparation, source_protocol, target_protocol)
    for key, actual in (("task_id", source_record.get("task_id")),
                        ("response_sha256", source_record.get("response_file_sha256")),
                        ("receipt_sha256", source_record.get("platform_verification_sha256")),
                        ("instance_id", source_task.get("instance_id")),
                        ("matrix_content_sha256", source_task.get("matrix_sha256"))):
        if str(source.get(key)) != str(actual):
            raise ValueError("Historical source identity changed: %s" % key)
    for key, actual in (("instance_id", target_task.get("instance_id")),
                        ("date", str(target_task.get("date"))),
                        ("target_task_name", target_task.get("task_name")),
                        ("target_matrix_content_sha256", target_task.get("matrix_sha256")),
                        ("target_hardware_matrix_file_sha256", target_task.get("hardware_matrix_file_sha256"))):
        if str(target.get(key)) != str(actual):
            raise ValueError("New panel identity changed: %s" % key)
    _verify_hash(source["response_path"], source["response_sha256"], "Historical original response")
    _verify_hash(source["response_audit_path"], source["response_audit_sha256"], "Historical response audit")
    _verify_hash(source["receipt_path"], source["receipt_sha256"], "Historical platform receipt")
    _verify_hash(source["hardware_matrix_path"], source["hardware_matrix_file_sha256"], "Historical frozen matrix")
    _verify_hash(source["platform_record_path"], source["platform_record_sha256"], "Historical platform record")
    _verify_hash(source["screenshot_evidence_path"], source["screenshot_evidence_sha256"], "Historical screenshot evidence")
    _verify_hash(target["target_hardware_matrix_path"], target["target_hardware_matrix_file_sha256"], "Target frozen matrix")
    _verify_hash(target["target_source_problem_path"], target["target_source_problem_sha256"], "Target condition source")
    if source["matrix_content_sha256"] != target["target_matrix_content_sha256"]:
        raise ValueError("Historical and target matrix content identities differ")
    with np.load(source["hardware_matrix_path"], allow_pickle=False) as a, np.load(target["target_hardware_matrix_path"], allow_pickle=False) as b:
        source_matrix = np.asarray(a["hardware_matrix"])
        target_matrix = np.asarray(b["hardware_matrix"])
        if (not np.array_equal(source_matrix, target_matrix)
                or matrix_sha256(source_matrix) != source["matrix_content_sha256"]):
            raise ValueError("Historical and target frozen matrix contents differ")
    data_path = bundle_dir / manifest["derived_data"]["path"]
    if sha256_file(data_path) != manifest["derived_data"]["sha256"]:
        raise ValueError("Derived prefix file hash mismatch")
    with np.load(source["response_path"], allow_pickle=False) as original, np.load(data_path, allow_pickle=False) as derived:
        original_samples = np.asarray(original["samples"])
        derived_samples = np.asarray(derived["samples"])
        _, unique_count = verify_prefix_arrays(
            original_samples, derived_samples, np.asarray(derived["source_row_indices"]))
    if unique_count != source["distinct_logical_states_in_all_100"]:
        raise ValueError("Historical logical-state summary changed")
    from .kaiwu_platform_receipt import validate_platform_verification
    receipt = validate_platform_verification(source["response_path"], source["receipt_path"], {
        "instance_id": source["instance_id"], "task_id": source["task_id"],
        "task_name": source["task_name"], "matrix_sha256": source["matrix_content_sha256"],
        "stage": "vs", "gain": source["gain"],
        "requested_reads": source["requested_reads"], "returned_reads": source["returned_reads"],
    })
    if receipt.get("verified_by") != "wsy":
        raise ValueError("Historical receipt reviewer mismatch")
    _verify_hash(manifest["prediction_replay"]["paired_curves_path"],
                 manifest["prediction_replay"]["stored_replay_curves_sha256"],
                 "Stored paired replay")
    if sha256_file(manifest["base_protocol"]["path"]) != manifest["base_protocol"]["sha256"]:
        raise ValueError("Base target protocol hash mismatch")
    if sha256_file(manifest["base_protocol"]["preparation_manifest_path"]) != manifest["base_protocol"]["preparation_manifest_sha256"]:
        raise ValueError("Base target preparation manifest hash mismatch")
    if sha256_file(Path(manifest["base_protocol"]["preparation_manifest_path"]).parent / "vs_execution_order.json") != manifest["base_protocol"]["vs_execution_order_sha256"]:
        raise ValueError("Base target VS execution order hash mismatch")
    if sha256_file(manifest["source_protocol"]["path"]) != manifest["source_protocol"]["sha256"]:
        raise ValueError("Source protocol hash mismatch")
    if sha256_file(manifest["source_protocol"]["preparation_manifest_path"]) != manifest["source_protocol"]["preparation_manifest_sha256"]:
        raise ValueError("Source preparation manifest hash mismatch")
    replay_path = manifest["prediction_replay"]["paired_curves_path"]
    _, replay_metrics = _replay_prediction(derived_samples, target_task, target_protocol,
                                           target_preparation, replay_path)
    if (abs(replay_metrics["raw_crps"] - float(manifest["prediction_replay"]["raw_crps"])) > 1e-12
            or replay_metrics["max_abs_error_vs_original_100read_curves"] > 1e-7):
        raise ValueError("Historical prefix decoder replay changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, default=ROOT / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_v9.json")
    parser.add_argument("--source-preparation", type=Path, default=ROOT / "export/opsd_sa_negative_kaiwu_preparation_v9")
    parser.add_argument("--target-protocol", type=Path, default=ROOT / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json")
    parser.add_argument("--target-preparation", type=Path, default=ROOT / "export/opsd_sa_negative_kaiwu_preparation_10read_v1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_historical_prefix(args.source_protocol, args.source_preparation,
                                       args.target_protocol, args.target_preparation,
                                       args.output)
    print(json.dumps({"status": "derived_prefix_created",
                      "output": str(args.output.resolve()),
                      "derived_sha256": manifest["derived_data"]["sha256"],
                      "raw_crps": manifest["prediction_replay"]["raw_crps"],
                      "base_protocol_can_count": manifest["new_panel_entry"]["countable_under_unamended_base_protocol"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
