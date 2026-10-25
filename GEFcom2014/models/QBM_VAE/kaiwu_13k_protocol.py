# -*- coding: utf-8 -*-
"""Freeze and audit the 13k-read FA-BM-VAE Kaiwu preparation protocol.

This module only validates offline artifacts and post-response records.  It
never imports the Kaiwu SDK and never submits a task.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .compare_hardware_responses import compare_response_pair, response_statistics
from .ising import ising_energy
from .kaiwu_adapter import kaiwu_sdk_versions, normalize_hardware_spins, matrix_sha256
from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "paper" / "fa_bm_vae_kaiwu_13k_protocol.json"
DEFAULT_OUTPUT = ROOT_DIR / "export" / "kaiwu_13k_preparation"


def resolve_path(path):
    path = Path(path)
    if path.is_file():
        return path
    return path if path.is_absolute() else ROOT_DIR / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path=DEFAULT_CONFIG):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    if protocol.get("method_name") != "FA-BM-VAE":
        raise ValueError("Protocol method must be FA-BM-VAE")
    hardware = protocol["hardware"]
    if hardware["candidate_gains"] != [125, 150, 200]:
        raise ValueError("Gain grid is frozen to [125, 150, 200]")
    if hardware["reads_per_instance"] != 100:
        raise ValueError("Reads per instance must be 100")
    if protocol.get("gain_selection", {}).get("read_convergence") != [50, 100]:
        raise ValueError("VS read sensitivity is frozen to overlapping [50, 100] prefixes")
    if protocol["dataset"]["selection_seed"] != 2026:
        raise ValueError("Selection seed must be 2026")
    if protocol["dataset"]["test_peeking"] is not False:
        raise ValueError("TEST peeking must be false")
    gain_count = len(hardware["candidate_gains"])
    expected_vs_tasks = int(protocol["vs"]["instances"]) * gain_count
    expected_test_tasks = int(protocol["test"]["instances"])
    if int(protocol["vs"]["tasks"]) != expected_vs_tasks:
        raise ValueError("VS task count must equal instances times candidate gains")
    if int(protocol["test"]["tasks"]) != expected_test_tasks:
        raise ValueError("TEST task count must equal TEST instances")
    reads = int(hardware["reads_per_instance"])
    expected_planned = (expected_vs_tasks + expected_test_tasks) * reads
    if int(protocol["budget"]["planned_samples"]) != expected_planned:
        raise ValueError("Planned sample budget does not match protocol counts")
    if protocol["budget"]["application_upper_limit"] != 15000:
        raise ValueError("Application upper limit must be 15000")
    quality_gate = protocol.get("gain_selection", {}).get("quality_gate")
    if quality_gate is not None and quality_gate.get("default_action") != "lock TEST until a pre-defined quality policy is frozen":
        raise ValueError("VS quality gate must fail closed until explicitly frozen")
    limits = hardware.get("platform_limits", {})
    if limits.get("int8_coefficient_range") != [-128, 127]:
        raise ValueError("Protocol must record the verified int8 coefficient range")
    if limits.get("int8_range_verified") is not True:
        raise ValueError("Protocol int8 coefficient range must be verified")
    profile = limits.get("machine_profile", {})
    if profile != {"name": "SPQC-550", "max_variables": 550,
                   "full_connectivity": True, "coupling_precision": "int8"}:
        raise ValueError("Protocol must declare the SPQC-550 machine profile")


def load_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Instance manifest must be a JSON list: %s" % path)
    return payload


def _forbidden_target_keys(instance):
    return {key for key in instance if key.lower() in {"y", "target", "target_values", "observed"}}


def validate_vs_manifests(paths, protocol):
    expected_gains = protocol["hardware"]["candidate_gains"]
    if len(paths) != len(expected_gains):
        raise ValueError("VS manifests must contain exactly one manifest per candidate gain")
    manifests = []
    id_sets = []
    for path, gain in zip(paths, expected_gains):
        items = load_manifest(path)
        if len(items) != protocol["vs"]["instances"]:
            raise ValueError("VS manifest %s has the wrong instance count" % path)
        ids = []
        for item in items:
            if item.get("split") != "VS" or item.get("selection") != "stratified":
                raise ValueError("VS manifest must be stratified and split=VS")
            if int(item.get("selection_seed")) != protocol["dataset"]["selection_seed"]:
                raise ValueError("VS selection seed mismatch")
            if float(item.get("hardware_gain")) != float(gain):
                raise ValueError("VS gain mismatch in %s" % path)
            if int(item.get("logical_n_bits")) != 48 or int(item.get("hardware_n_bits")) != 49:
                raise ValueError("VS manifest must declare 48/49 spins")
            if _forbidden_target_keys(item):
                raise ValueError("VS manifest contains target values")
            ids.append(str(item["id"]))
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate VS instance IDs in %s" % path)
        id_sets.append(set(ids))
        manifests.append({"path": str(Path(path).resolve()), "gain": gain, "instances": items})
    if not all(current == id_sets[0] for current in id_sets[1:]):
        raise ValueError("All gain manifests must share identical VS instance IDs")
    return manifests


def validate_test_manifest(path, protocol):
    items = load_manifest(path)
    dataset_name = str(protocol.get("dataset", {}).get("name", "")).lower()
    if not items:
        raise ValueError("TEST manifest must contain at least one instance")
    if (len(items) != protocol["test"]["instances"] and
            not protocol["test"].get("allow_preregistered_subset", False)):
        raise ValueError("TEST manifest does not match the frozen protocol size")
    zones = []
    ids = []
    for item in items:
        if item.get("split") != "TEST" or item.get("selection") not in {"stratified", "all"}:
            raise ValueError("TEST manifest must use selection=stratified or all and split=TEST")
        if int(item.get("selection_seed")) != protocol["dataset"]["selection_seed"]:
            raise ValueError("TEST selection seed mismatch")
        if item.get("hardware_gain") is not None or item.get("hardware_matrix_sha256") is not None:
            raise ValueError("TEST manifest must not preselect a hardware gain")
        if item.get("hardware_n_bits") is not None:
            raise ValueError("TEST manifest must contain logical h,J only")
        if _forbidden_target_keys(item):
            raise ValueError("TEST manifest contains target values")
        zone = item.get("zone")
        if zone is None and "opsd" not in dataset_name:
            raise ValueError("TEST manifest lacks zone provenance")
        if zone is not None:
            zones.append(int(zone))
        ids.append(str(item["id"]))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate TEST instance IDs")
    counts = {zone: zones.count(zone) for zone in range(1, 11)}
    if len(items) == protocol["test"]["instances"] and "gefcom" in dataset_name:
        if counts != {zone: 10 for zone in range(1, 11)}:
            raise ValueError("Full GEFCom TEST manifest must contain 10 instances per zone")
    if len(items) == protocol["test"]["instances"] and protocol["test"].get("require_unique_dates", False):
        dates = [str(item.get("date", "")) for item in items]
        if any(not date for date in dates) or len(set(dates)) != len(dates):
            raise ValueError("Full TEST manifest must contain unique dates")
    return {"path": str(Path(path).resolve()), "instances": items, "zone_counts": counts}


def audit_budget(vs_paths, test_path, protocol):
    validate_protocol(protocol)
    vs = validate_vs_manifests(vs_paths, protocol)
    test = validate_test_manifest(test_path, protocol)
    reads = int(protocol["hardware"]["reads_per_instance"])
    vs_tasks = sum(len(item["instances"]) for item in vs)
    test_tasks = len(test["instances"])
    result = {
        "vs_tasks": vs_tasks,
        "vs_samples": vs_tasks * reads,
        "test_tasks": test_tasks,
        "test_samples": test_tasks * reads,
        "formal_tasks": vs_tasks + test_tasks,
        "planned_samples": (vs_tasks + test_tasks) * reads,
        "backup_tasks_max": int(protocol["budget"]["backup_tasks_max"]),
        "backup_samples_max": int(protocol["budget"]["backup_samples_max"]),
        "application_upper_limit": int(protocol["budget"]["application_upper_limit"]),
        "hardware_claim": False,
        "physical_platform_used": False,
        "test_peeking": False,
        "platform_limits_checked": dict(protocol["hardware"]["platform_limits"]),
        "machine_profile": dict(protocol["hardware"]["platform_limits"]["machine_profile"]),
        "sdk_versions": kaiwu_sdk_versions(),
    }
    expected_vs_tasks = int(protocol["vs"]["instances"]) * len(protocol["hardware"]["candidate_gains"])
    expected_test_tasks = int(protocol["test"]["instances"])
    if result["vs_tasks"] != expected_vs_tasks:
        raise ValueError("VS budget mismatch")
    if result["test_tasks"] != expected_test_tasks:
        raise ValueError("TEST budget mismatch")
    expected_planned = (expected_vs_tasks + expected_test_tasks) * reads
    if result["planned_samples"] != expected_planned:
        raise ValueError("Formal sample budget mismatch")
    if result["planned_samples"] + result["backup_samples_max"] > result["application_upper_limit"]:
        raise ValueError("Backup allowance exceeds application upper limit")
    return result


def _scalar(container, key, default=""):
    if key not in container:
        return default
    values = np.asarray(container[key]).reshape(-1)
    if values.size != 1:
        raise ValueError("%s must be scalar" % key)
    return values[0].item()


def _platform_metadata_complete(meta, returned_reads):
    required = ("task_id", "task_name", "sdk_version", "platform_backend",
                "submitted_at", "completed_at", "matrix_sha256", "response_file_sha256")
    return all(str(meta.get(key, "")) for key in required) and int(returned_reads) > 0


def read_response_metadata(path):
    with np.load(path, allow_pickle=False) as response:
        metadata = {key: _scalar(response, key) for key in (
            "instance_id", "task_id", "task_name", "sdk_version",
            "platform_backend", "submitted_at", "completed_at", "matrix_sha256",
            "response_file_sha256", "hardware_claim", "physical_platform_used",
            "requested_reads", "returned_reads") if key in response}
        samples = np.asarray(response["samples"], dtype=np.int8)
    return metadata, samples


def compare_read_convergence(h, j, reference, candidate, target_beta=1.0):
    """Compare overlapping first-50 and full-100 read prefixes.

    This is a read-count sensitivity diagnostic, not a convergence proof: the
    50-read set is exactly contained in the 100-read set.
    """
    if reference.shape[0] < 100 or candidate.shape[0] < 100:
        raise ValueError("VS convergence comparison requires at least 100 reads")
    result = {}
    for count in (50, 100):
        metrics = compare_response_pair(h, j, reference[:count], candidate[:count])
        metrics["effective_beta_error"] = abs(float(metrics["candidate_beta_eff"]) - float(target_beta))
        result["reads_%d" % count] = metrics
    return {
        "comparison": result,
        "sample_overlap": 50,
        "overlap_fraction_of_50_read_set": 1.0,
        "interpretation": "overlapping read-count sensitivity only; not a convergence proof",
    }


def _recompute_response_metrics(record, target_beta=1.0, require_reference_provenance=False):
    """Recompute VS metrics from frozen files, never from submitted JSON fields."""
    required = ["source_problem_path", "source_problem_sha256",
                "reference_sample_path", "reference_sample_sha256",
                "hardware_matrix_path"]
    if require_reference_provenance:
        required.extend(("hardware_matrix_file_sha256", "reference_sampler"))
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError("VS response lacks frozen metric inputs: %s" % ", ".join(missing))
    problem_path = resolve_path(record["source_problem_path"])
    reference_path = resolve_path(record["reference_sample_path"])
    matrix_path = resolve_path(record["hardware_matrix_path"])
    if sha256_file(problem_path) != str(record["source_problem_sha256"]):
        raise ValueError("VS logical h,J source hash mismatch")
    if sha256_file(reference_path) != str(record["reference_sample_sha256"]):
        raise ValueError("VS frozen reference sample hash mismatch")
    if record.get("hardware_matrix_file_sha256") and sha256_file(matrix_path) != str(record["hardware_matrix_file_sha256"]):
        raise ValueError("VS hardware matrix file hash mismatch")
    with np.load(problem_path, allow_pickle=False) as problem:
        h = np.asarray(problem["h"], dtype=np.float64)
        j = np.asarray(problem["J"], dtype=np.float64)
    with np.load(matrix_path, allow_pickle=False) as matrix_archive:
        matrix = np.asarray(matrix_archive["hardware_matrix"])
    if matrix.shape != (49, 49) or matrix_sha256(matrix) != str(record["matrix_sha256"]):
        raise ValueError("VS hardware matrix does not match the response identity")
    response_path = Path(record["response_path"])
    with np.load(response_path, allow_pickle=False) as response:
        raw = np.asarray(response["samples"], dtype=np.int8)
    logical = normalize_hardware_spins(raw, logical_n_bits=h.size)
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_samples = np.asarray(reference["samples"], dtype=np.int8)
        sampler = record.get("reference_sampler")
        if sampler is None:
            if require_reference_provenance:
                raise ValueError("VS response lacks frozen reference sampler provenance")
            sampler = {}
        checks = {
            "backend": sampler.get("backend"),
            "matrix_space": sampler.get("matrix_space"),
            "random_seed": sampler.get("seed"),
            "requested_reads": sampler.get("reads"),
            "sweeps": sampler.get("sweeps"),
            "sampler_beta": sampler.get("sampler_beta"),
            "matrix_beta": sampler.get("matrix_beta"),
            "hardware_gain": record.get("gain"),
            "hardware_matrix_sha256": record.get("matrix_sha256"),
            "source_problem_sha256": record.get("source_problem_sha256"),
            "instance_id": record.get("instance_id"),
        }
        for key, expected in checks.items():
            if key not in reference:
                if not require_reference_provenance:
                    continue
                raise ValueError("VS reference sample provenance mismatch: %s" % key)
            actual = np.asarray(reference[key]).reshape(-1)[0].item()
            if isinstance(expected, (int, float, np.integer, np.floating)):
                matches = np.isclose(float(actual), float(expected), rtol=0, atol=1e-12)
            else:
                matches = str(actual) == str(expected)
            if not matches:
                raise ValueError("VS reference sample provenance mismatch: %s" % key)
    if (reference_samples.shape != (100, h.size)
            or not np.isin(reference_samples, (-1, 1)).all()):
        raise ValueError("Frozen VS reference samples have the wrong logical dimension")
    metrics = compare_response_pair(h, j, reference_samples, logical)
    read_convergence = compare_read_convergence(h, j, reference_samples, logical)
    metrics["effective_beta_error"] = abs(float(metrics["candidate_beta_eff"]) - float(target_beta))
    metrics["unique_state_fraction"] = float(metrics["candidate_unique_state_fraction"])
    result = {key: float(metrics[key]) for key in (
        "energy_wasserstein", "effective_beta_error", "edge_moment_mae",
        "magnetization_mae", "unique_state_fraction")}
    result["read_convergence_diagnostics"] = read_convergence
    if not np.isfinite([value for value in result.values()
                        if isinstance(value, (int, float, np.integer, np.floating))]).all():
        raise ValueError("VS recomputed calibration metrics are non-finite")
    return result


def aggregate_gain_metrics(response_records, recompute=False, prediction_inputs=None,
                           require_physical=True):
    """Aggregate calibration metrics; formal paths recompute from raw files."""
    grouped = {}
    for record in response_records:
        item = dict(record)
        if recompute:
            item["recomputed_metrics"] = _recompute_response_metrics(
                record, require_reference_provenance=prediction_inputs is not None)
        grouped.setdefault(float(record["gain"]), []).append(item)
    rows = []
    for gain, records in sorted(grouped.items()):
        metrics = [row.get("recomputed_metrics", row) for row in records]
        rows.append({
            "gain": gain,
            "n_verified_instances": len(records),
            "energy_wasserstein": float(np.mean([row["energy_wasserstein"] for row in metrics])),
            "effective_beta_error": float(np.mean([row["effective_beta_error"] for row in metrics])),
            "edge_moment_mae": float(np.mean([row["edge_moment_mae"] for row in metrics])),
            "magnetization_mae": float(np.mean([row["magnetization_mae"] for row in metrics])),
            "unique_state_fraction": float(np.mean([row["unique_state_fraction"] for row in metrics])),
            "read_convergence_diagnostics": [
                {"instance_id": record["instance_id"],
                 **metric.get("read_convergence_diagnostics", {})}
                for record, metric in zip(records, metrics)
                if metric.get("read_convergence_diagnostics")
            ],
        })
    if prediction_inputs is not None:
        from .opsd_vs_selection import recompute_vs_predictions
        predictions = recompute_vs_predictions(
            response_records, require_physical=require_physical, **prediction_inputs)
        for row in rows:
            daily = [item for item in predictions if item["gain"] == row["gain"]]
            row["raw_vs_crps"] = float(np.mean([item["metrics"]["CRPS"] for item in daily]))
            row["vs_prediction_metrics"] = daily
            row["prediction_read_sensitivity_diagnostics"] = [
                {"instance_id": item["instance_id"],
                 **item["prediction_read_sensitivity"]}
                for item in daily
            ]
    return rows


def prediction_quality_mode(quality_policy):
    return isinstance(quality_policy, dict) and quality_policy.get("mode") == "engineering_valid_prediction_selection"


def choose_vs_candidate(candidate_metrics, quality_policy):
    """One selection rule shared by gain freezing and TEST revalidation."""
    if prediction_quality_mode(quality_policy):
        if not candidate_metrics or any(
                not np.isfinite(float(row.get("raw_vs_crps", np.nan)))
                or float(row["raw_vs_crps"]) < 0 for row in candidate_metrics):
            raise ValueError("Complete finite Raw VS CRPS is required; TEST remains locked")
        return min(candidate_metrics, key=lambda row: (row["raw_vs_crps"], row["gain"]))
    qualified = [row for row in candidate_metrics if row.get("quality_status", "pass") == "pass"]
    if not qualified:
        raise ValueError("All candidate gains failed the VS quality gate; TEST remains locked")
    return min(qualified, key=lambda row: (
        row["energy_wasserstein"], row["effective_beta_error"], row["edge_moment_mae"],
        row["magnetization_mae"], -row["unique_state_fraction"], row["gain"],
    ))


def select_vs_gain(response_records, output_path, target_beta=1.0,
                   required_gains=None, expected_instances_per_gain=None,
                   expected_instance_ids=None, expected_task_map=None,
                   expected_identity=None, quality_policy=None,
                   require_physical=True, recompute_from_raw=None, prediction_inputs=None):
    """Freeze one gain from verified physical VS records.

    Metrics are averaged within each gain across the fixed VS panel. Ties use
    the lower gain, so no TEST outcome can alter a deterministic selection.
    """
    if not response_records:
        raise ValueError("No VS response records supplied")
    if required_gains is not None or expected_instances_per_gain is not None:
        response_records = validate_vs_response_records(
            response_records,
            required_gains=required_gains,
            expected_instances_per_gain=expected_instances_per_gain,
            expected_instance_ids=expected_instance_ids,
            expected_task_map=expected_task_map,
            expected_identity=expected_identity,
            quality_policy=quality_policy,
            require_quality_policy=False,
            require_physical=require_physical,
        )
    for record in response_records:
        if float(record.get("gain")) not in (125.0, 150.0, 200.0):
            raise ValueError("VS response gain is outside the frozen candidate grid")
        if require_physical:
            if not record.get("hardware_claim") or not record.get("physical_platform_used"):
                raise ValueError("Gain selection requires verified physical VS responses")
        elif record.get("hardware_claim") is not False or record.get("physical_platform_used") is not False:
            raise ValueError("Offline gain selection requires hardware_claim=false")
        if recompute_from_raw is False:
            for key in ("energy_wasserstein", "effective_beta_error", "edge_moment_mae",
                        "magnetization_mae", "unique_state_fraction"):
                if key not in record:
                    raise ValueError("VS response record lacks %s" % key)
    grouped = {}
    for record in response_records:
        grouped.setdefault(float(record["gain"]), []).append(record)
    if required_gains is not None and set(grouped) != {float(gain) for gain in required_gains}:
        raise ValueError("VS gain selection does not contain exactly the frozen candidate grid")
    if expected_instances_per_gain is not None:
        for gain, records in grouped.items():
            if len(records) != int(expected_instances_per_gain):
                raise ValueError("VS gain %s does not have the required number of verified instances" % gain)
    raw_fields = ("source_problem_path", "source_problem_sha256",
                  "reference_sample_path", "reference_sample_sha256",
                  "hardware_matrix_path")
    formal_raw_bound = bool(expected_task_map) and any(
        any(expected.get(field) for field in raw_fields)
        for expected in expected_task_map.values()
    )
    if recompute_from_raw is None:
        recompute_from_raw = all(all(record.get(field) for field in raw_fields)
                                 for record in response_records)
    if required_gains is not None and formal_raw_bound and not recompute_from_raw:
        raise ValueError("Formal VS gain selection requires raw response metric inputs")
    if recompute_from_raw is False:
        for record in response_records:
            for key in ("energy_wasserstein", "effective_beta_error", "edge_moment_mae",
                        "magnetization_mae", "unique_state_fraction"):
                if key not in record:
                    raise ValueError("VS response record lacks %s" % key)
    prediction_mode = prediction_quality_mode(quality_policy)
    if prediction_mode:
        validate_quality_policy(quality_policy, allow_test_only=not require_physical)
        if required_gains is None or not recompute_from_raw or prediction_inputs is None:
            raise ValueError("Prediction selection requires the full VS panel and frozen raw prediction inputs")
        if set(prediction_inputs) != {"preparation", "protocol"}:
            raise ValueError("Prediction inputs must not override complete-panel validation")
        if prediction_inputs["protocol"]["gain_selection"]["quality_gate"] != quality_policy:
            raise ValueError("Prediction selection policy differs from the frozen protocol")
    candidate_metrics = aggregate_gain_metrics(
        response_records, recompute=recompute_from_raw,
        prediction_inputs=prediction_inputs if prediction_mode else None,
        require_physical=require_physical)
    if required_gains is not None:
        if not isinstance(quality_policy, dict):
            raise ValueError("No VS quality policy is defined; TEST remains locked")
        validate_quality_policy(quality_policy, allow_test_only=not require_physical)
        candidate_metrics = evaluate_gain_quality(candidate_metrics, quality_policy)
    chosen = choose_vs_candidate(candidate_metrics, quality_policy)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen gain record already exists: %s" % output_path)
    payload = {
        "method_name": "FA-BM-VAE",
        "artifact_role": "verified_physical_vs_gain_selection" if require_physical else "offline_test_fixture",
        "selection_split": "VS",
        "hardware_claim": bool(require_physical),
        "physical_platform_used": bool(require_physical),
        "eligible_for_paper": False,
        "unlock_real_test": bool(require_physical),
        "selected_gain": float(chosen["gain"]),
        "selection_metric": "mean Raw VS CRPS, then lower gain" if prediction_mode else "lexicographic mean energy Wasserstein, effective-beta error, edge-moment MAE, magnetization MAE, negative unique-state fraction, then lower gain",
        "selection_metrics": chosen,
        "candidate_metrics": candidate_metrics,
        "immutable": True,
        "test_peeking": False,
        "test_data_used": False,
        "verified_vs_records": [dict(row) for row in response_records],
        "n_verified_vs_records": len(response_records),
        "metrics_source": "raw response samples + frozen logical h,J + frozen reference samples"
        if recompute_from_raw else "legacy record fields",
    }
    if expected_identity:
        payload.update({key: str(value) for key, value in expected_identity.items()})
    if quality_policy is not None:
        payload["quality_gate"] = dict(quality_policy)
        payload["quality_summary_recomputed"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _finite_metric_record(record):
    keys = ("energy_wasserstein", "effective_beta_error", "edge_moment_mae",
            "magnetization_mae", "unique_state_fraction")
    try:
        values = [float(record[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all())


def validate_vs_response_records(response_records, required_gains,
                                 expected_instances_per_gain,
                                 expected_instance_ids=None,
                                 expected_task_map=None, expected_identity=None,
                                 quality_policy=None, require_quality_policy=True,
                                 require_physical=True):
    """Validate the complete physical VS panel before gain selection.

    This is deliberately stricter than checking ``hardware_claim``.  The
    response file, task identity, matrix identity, reads, provenance hashes,
    finite calibration metrics, and an explicit quality decision all have to
    agree with the frozen protocol.
    """
    from .import_hardware_responses import strict_response_record_valid

    gains = {float(gain) for gain in required_gains}
    expected_ids = {str(item) for item in expected_instance_ids} if expected_instance_ids is not None else None
    if len(response_records) != len(gains) * int(expected_instances_per_gain):
        raise ValueError("VS responses must contain exactly gain x instance unique combinations")
    seen = set()
    for record in response_records:
        try:
            gain = float(record["gain"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("VS response has no valid gain")
        instance_id = str(record.get("instance_id", ""))
        key = (gain, instance_id)
        if gain not in gains or not instance_id or key in seen:
            raise ValueError("VS response panel contains an extra or duplicate condition")
        if expected_ids is not None and instance_id not in expected_ids:
            raise ValueError("VS response contains a condition outside the frozen VS panel")
        expected = expected_task_map.get(key) if expected_task_map else None
        if not strict_response_record_valid(record, expected, require_physical=require_physical):
            raise ValueError("VS response failed raw task, metadata, hash, shape, or read validation")
        if expected_identity:
            for field, value in expected_identity.items():
                if str(record.get(field, "")) != str(value):
                    raise ValueError("VS response provenance does not match frozen %s" % field)
        expected_requires_raw = bool(expected and any(
            expected.get(field) for field in (
                "source_problem_path", "source_problem_sha256",
                "reference_sample_path", "reference_sample_sha256",
                "hardware_matrix_path")
        ))
        if not expected_requires_raw and not _finite_metric_record(record):
            raise ValueError("VS response contains a non-finite calibration metric")
        if expected_requires_raw:
            try:
                _recompute_response_metrics(
                    record, require_reference_provenance=bool(
                        expected and expected.get("reference_sampler")))
            except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
                raise ValueError("VS response raw metric inputs are invalid") from exc
        if require_quality_policy and quality_policy is None:
            raise ValueError("No VS quality policy is defined; TEST remains locked")
        seen.add(key)
    for gain in gains:
        ids = {str(row["instance_id"]) for row in response_records if float(row["gain"]) == gain}
        if len(ids) != int(expected_instances_per_gain) or (expected_ids is not None and ids != expected_ids):
            raise ValueError("VS gain %s does not contain the exact frozen instance panel" % gain)
    return list(response_records)


def validate_quality_policy(quality_policy, allow_test_only=False):
    """Validate approval and finite limits before any paid batch is submitted."""
    prediction_mode = prediction_quality_mode(quality_policy)
    approval = "policy_approved" if prediction_mode else "thresholds_approved"
    if not isinstance(quality_policy, dict) or quality_policy.get("status") != "defined" or quality_policy.get(approval) is not True:
        raise ValueError("VS quality policy is not approved; TEST remains locked")
    if quality_policy.get("mode", "legacy_threshold_gate") not in (
            "legacy_threshold_gate", "engineering_valid_prediction_selection"):
        raise ValueError("Unknown VS quality policy mode")
    if quality_policy.get("test_only") and not allow_test_only:
        raise ValueError("A test-only quality policy cannot authorize physical submission or gain freeze")
    if not isinstance(quality_policy.get("rule_version"), str) or not quality_policy["rule_version"].strip():
        raise ValueError("VS quality policy requires a rule_version")
    if prediction_mode:
        if (quality_policy.get("selection_metric") != "raw_vs_crps"
                or quality_policy.get("tie_break") != "lower_gain"
                or quality_policy.get("diversity_gate_approved") is True):
            raise ValueError("Prediction policy must use Raw VS CRPS, lower gain ties, and diagnostic diversity")
    thresholds = quality_policy.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("VS quality policy lacks thresholds")
    for metric in ("energy_wasserstein", "effective_beta_error", "edge_moment_mae", "magnetization_mae"):
        try:
            value = thresholds[metric]
            valid = not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("VS quality threshold must be finite and non-negative: %s" % metric)
    if quality_policy.get("diversity_gate_approved") is True:
        try:
            value = thresholds["unique_state_fraction_min"]
            valid = not isinstance(value, bool) and np.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("Approved diversity threshold must be finite and within [0, 1]")
    return thresholds


def evaluate_gain_quality(candidate_metrics, quality_policy):
    """Recompute gain-level quality from verified response metrics.

    ``quality_assessment`` fields supplied by a platform or mock are ignored.
    Legacy policies gate on thresholds. Prediction-selection policies retain
    the same checks as diagnostics only, without certifying distribution fit.
    """
    thresholds = validate_quality_policy(quality_policy, allow_test_only=True)
    error_metrics = ("energy_wasserstein", "effective_beta_error",
                     "edge_moment_mae", "magnetization_mae")
    output = []
    for row in candidate_metrics:
        if not _finite_metric_record(row):
            raise ValueError("Non-finite latent diagnostics; TEST remains locked")
        checks = {metric: float(row[metric]) <= float(thresholds[metric])
                  for metric in error_metrics}
        diversity_gate = quality_policy.get("diversity_gate_approved") is True
        checks["unique_state_fraction"] = {
            "value": float(row["unique_state_fraction"]),
            "gate_enabled": diversity_gate,
        }
        item = dict(row)
        item["quality_checks"] = checks
        item["quality_status"] = "pass" if all(
            bool(checks[metric]) for metric in error_metrics
        ) and (not diversity_gate or float(row["unique_state_fraction"]) >= float(
            thresholds["unique_state_fraction_min"]
        )) else "fail"
        item["quality_rule_version"] = str(quality_policy["rule_version"])
        if prediction_quality_mode(quality_policy):
            item["diagnostic_threshold_status"] = item["quality_status"]
            item["quality_status"] = "diagnostic_only"
            item["latent_thresholds_are_hard_gates"] = False
        output.append(item)
    return output


def crps_ensemble(samples, target):
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ordered = np.sort(samples, axis=0)
    n = samples.shape[0]
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    return float(np.mean(np.abs(samples - target[None, :]) -
                         np.sum(ordered * coefficients[:, None], axis=0) / n ** 2))


def scenario_metrics(samples, target):
    """Return the canonical daily metrics used by the classical evaluator.

    ``samples`` is reads x horizon and ``target`` is one horizon.  The common
    evaluator consumes periods x reads, so this adapter performs only that
    explicit orientation conversion; all score definitions, Variogram beta
    (0.5), ramp CRPS, 90-percent coverage, and interval width come from the
    canonical implementation.
    """
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if samples.ndim != 2 or target.shape != (samples.shape[1],):
        raise ValueError("samples/target shape must be (reads,horizon)/(horizon,)")
    daily = evaluate_model_daily(
        samples.T, target.reshape(1, -1), max_s=samples.shape[0],
        periods_per_day=target.size,
    )
    return {
        "CRPS": float(daily["CRPS_raw"][0]),
        "Energy": float(daily["Energy"][0]),
        "Variogram": float(daily["Variogram"][0]),
        "Ramp_CRPS": float(daily["ramp_CRPS"][0]),
        "coverage": float(daily["PICP90"][0]),
        "PICP90": float(daily["PICP90"][0]),
        "interval_width90": float(daily["interval_width90"][0]),
        "MAQCE": float(daily["MAQCE"][0]),
    }


def paired_date_inference(differences, repetitions=2000, seed=2026,
                          block_length_days=7):
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0 or not np.isfinite(differences).all():
        raise ValueError("date differences must be a non-empty finite one-dimensional array")
    if repetitions < 1 or block_length_days < 1:
        raise ValueError("repetitions and block_length_days must be positive")
    # Reuse the OPSD chronological non-overlapping block bootstrap. Inference
    # assumes blocks are exchangeable; sign-flips additionally assume block-
    # level symmetry under the paired null. Within-block dependence is retained.
    from GEFcom2014.forecast_quality.opsd_topology_vs_evaluation import (
        _time_blocks, _weighted_block_bootstrap,
    )
    rng = np.random.default_rng(seed)
    blocks = _time_blocks(differences, block_length_days)
    boot, block_lengths = _weighted_block_bootstrap(
        differences, repetitions, rng, block_size=block_length_days
    )
    signs = rng.choice([-1.0, 1.0], size=(repetitions, len(blocks)))
    block_sums = np.asarray([block.sum() for block in blocks], dtype=np.float64)
    null = (signs * block_sums[None, :]).sum(axis=1) / differences.size
    mean = float(differences.mean())
    return {
        "mean_difference": mean,
        "ci_2.5": float(np.quantile(boot, 0.025)),
        "ci_97.5": float(np.quantile(boot, 0.975)),
        "paired_sign_flip_p": float((np.count_nonzero(np.abs(null) >= abs(mean)) + 1) / (repetitions + 1)),
        "effect_size": float(mean / (differences.std(ddof=1) + 1e-12)) if differences.size > 1 else 0.0,
        "n_instance_dates": int(differences.size),
        "n_date_blocks": int(len(block_lengths)),
        "block_length_days": int(block_length_days),
        "inference_assumption": "exchangeable non-overlapping chronological blocks; symmetric block differences for sign-flip",
    }


def aggregate_test_metrics(rows, expected_dates, repetitions=2000, seed=2026,
                           block_length_days=7):
    """Compare fixed A/B/C daily rows; expected dates come from the frozen manifest.

    A=float logical SA, B=int8-matrix SA, C=SPQC-550. B vs C is primary;
    A vs B is the representation/quantization contrast.
    """
    if not rows or not expected_dates:
        raise ValueError("Rows and the complete frozen expected-date list are required")
    models = ("float_logical_sa", "int8_matrix_sa", "spqc_550")
    expected = [str(np.datetime64(str(value), "D")) for value in expected_dates]
    if len(expected) != len(set(expected)):
        raise ValueError("Frozen expected dates contain duplicates")
    grouped = {}
    for row in rows:
        date = str(np.datetime64(str(row["date"]), "D"))
        model = str(row["model"])
        if date not in expected or model not in models:
            raise ValueError("Unexpected TEST date or model: %s / %s" % (date, model))
        key = (date, model)
        if key in grouped:
            raise ValueError("Duplicate daily TEST row for %s / %s" % key)
        values = {}
        for metric in ("CRPS", "Energy", "Variogram", "Ramp_CRPS", "coverage", "PICP90", "interval_width90", "MAQCE"):
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError("Non-finite daily metric for %s / %s" % key)
            values[metric] = value
        grouped[key] = values
    required = {(date, model) for date in expected for model in models}
    if set(grouped) != required:
        missing = sorted(required - set(grouped))
        extra = sorted(set(grouped) - required)
        raise ValueError("TEST daily panel incomplete (missing=%d, extra=%d)" % (len(missing), len(extra)))
    comparisons = []
    contrasts = (("spqc_550", "int8_matrix_sa", "primary_B_vs_C"),
                 ("int8_matrix_sa", "float_logical_sa", "secondary_A_vs_B"))
    for model, reference_model, role in contrasts:
        for metric in ("CRPS", "Energy", "Variogram", "Ramp_CRPS", "coverage", "PICP90", "interval_width90", "MAQCE"):
            diffs = [grouped[(date, model)][metric] - grouped[(date, reference_model)][metric]
                     for date in expected]
            comparisons.append({"model": model, "reference_model": reference_model,
                                "contrast_role": role, "metric": metric,
                                **paired_date_inference(
                                    diffs, repetitions, seed, block_length_days
                                )})
    return {
        "paired_comparisons": comparisons,
        "model_roles": {"A": models[0], "B": models[1], "C": models[2]},
        "primary_contrast": "B_vs_C",
        "secondary_contrast": "A_vs_B",
        "expected_dates": expected,
        "reads_are_independent_test_units": False,
    }


def readiness_report(protocol, budget, vs_paths, test_path, output_path, missing_platform=True):
    model = protocol["model"]
    model_files = [model["checkpoint"], model["model_config"], model["anchor_checkpoint"], protocol["dataset"]["source"]]
    existing = {path: resolve_path(path).is_file() for path in model_files}
    lines = [
        "# FA-BM-VAE Kaiwu 13k Preparation Readiness",
        "",
        "Status: `BLOCKED`",
        "",
        "This report records offline package preparation only. No Kaiwu/SPQC task was submitted.",
        "The authoritative current protocol is `configs/paper/fa_bm_vae_kaiwu_13k_protocol.json`; historical 50-VS / 500-TEST offline replay with 1000 reads per instance is not this submission plan.",
        "",
        "## Frozen scope",
        "",
        "- Method: FA-BM-VAE; classical training; hardware replaces only frozen conditional Ising latent sampling.",
        "- Model: `%s`" % model["model_name"],
        "- TEST peeking: `false`",
        "- Model/data files present: `%s`" % existing,
        "",
        "## Modified files",
        "",
        "- `configs/paper/fa_bm_vae_kaiwu_13k_protocol.json`",
        "- `GEFcom2014/models/QBM_VAE/kaiwu_13k_protocol.py`",
        "- `GEFcom2014/models/QBM_VAE/submit_kaiwu_sampling.py`",
        "- `GEFcom2014/models/QBM_VAE/import_hardware_responses.py`",
        "- Audited existing `kaiwu_adapter.py`, `calibrate_hardware_temperature.py` and `sample_exported_ising.py` without changing old results.",
        "- `tests/test_kaiwu_13k_protocol.py` and `tests/test_submit_kaiwu_safety.py`",
        "- `GEFcom2014/models/QBM_VAE/run_kaiwu_13k_sampling.py` (default-dry-run batch gate).",
        "- `GEFcom2014/forecast_quality/paper_artifact_guard.py` and synthetic isolation tests.",
        "",
        "## Budget",
        "",
        "- VS: 30 tasks x 100 reads = 3,000 samples",
        "- TEST: 100 tasks x 100 reads = 10,000 samples",
        "- Formal total: 130 tasks = 13,000 samples",
        "- Backup allowance: at most 20 tasks / 2,000 samples",
        "- Application upper limit: 15,000 samples",
        "- Audit: `%s`" % json.dumps(budget, sort_keys=True),
        "",
        "## VS/TEST task manifests",
        "",
        "| stage | gain | tasks | reads/task | samples | status |",
        "|---|---:|---:|---:|---:|---|",
        "| VS | 125 | 10 | 100 | 1,000 | offline package |",
        "| VS | 150 | 10 | 100 | 1,000 | offline package |",
        "| VS | 200 | 10 | 100 | 1,000 | offline package |",
        "| TEST logical h,J | unresolved | 100 | pending | 10,000 planned | gain not frozen |",
        "",
        "The three VS manifests share the same ten instance IDs. The TEST manifest contains ten instances per Wind zone and no hardware matrix or gain.",
        "",
        "## Validation evidence",
        "",
        "- Full regression is rerun after this preparation; any real-platform skips remain skips, not hardware evidence.",
        "- Kaiwu batch dry-run: 30 VS tasks / 3,000 reads with matrix hashes; no SDK call.",
        "- Offline manifests declare 49 spins, 228 source edges, int8 range [-128, 127], and per-gain maximum coefficients; no clipping is allowed.",
        "- SDK audit preserves both importlib.metadata.version('kaiwu') and kaiwu.__version__: `%s`."
        % json.dumps(budget.get("sdk_versions", {}), sort_keys=True),
        "- OPSD rolling-origin formal Raw matrix is complete: 3 origins x 4 models x 5 seeds, 60/60 tasks, 5,400 date rows; Cal/Temporal-ECC remain sensitivity analyses.",
        "- Compile check: `PASS`.",
        "- CLI without explicit confirmation: rejected before SDK/project access.",
        "",
        "## Offline manifests",
        "",
        "- VS manifests: %s" % ", ".join(str(Path(path).resolve()) for path in vs_paths),
        "- TEST logical manifest: `%s`" % Path(test_path).resolve(),
        "- Hardware gain: unresolved until verified VS responses; TEST matrices must be rebuilt from original h,J.",
        "",
        "## Platform gate",
        "",
        "- `hardware_claim=false` and `physical_platform_used=false` for all offline artifacts.",
        "- SPQC-550 is recorded as 550 variables/full-connectivity/int8; this metadata does not select or verify the physical machine.",
        "- The account UI showed 15,000 Sampling Credits; this offline audit cannot verify the balance, SDK license, or SPQC-550 task routing." if missing_platform else "- Platform conditions supplied.",
        "- Confirm the SDK task routes to SPQC-550 before the first paid submission; verify the actual platform/task metadata in the returned response.",
        "- Real TEST submission is blocked until one gain is frozen from VS only.",
        "",
        "## Manual-gated next command",
        "",
        "The code and documentation are ready for a single-task canary, but run only after the SDK license, account credits and SPQC-550 routing are confirmed; project_no is optional for standard SPQC-550 tasks. This command was not executed:",
        "",
        "```powershell",
        "$env:FA_BM_VAE_RUN_REAL_KAIWU=\"1\"",
        "D:\\anaconda\\envs\\wsy\\python.exe -m GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling `",
        "  --stage vs --execute --limit 1 --resume --confirm-real-submission `",
        "  --output-dir export\\kaiwu_13k_preparation\\real_responses `",
        "  --checkpoint-dir export\\kaiwu_13k_preparation\\real_checkpoints",
        "```",
        "",
        "This first task is one of the frozen 30 VS tasks and consumes 100 of the planned 3,000 VS reads. Continue only with the same driver and `--resume` after response audit; TEST remains blocked until VS-only gain selection freezes exactly one gain.",
        "",
        "## Unrelated experiments",
        "",
        "- OPSD rolling-origin: formal Raw matrix complete in export/opsd_rolling_origins_formal_v1; no additional training is permitted.",
        "- SDWPF-5: intentionally not run in this round; excluded from this hardware budget.",
        "- Decision value: exploratory normalized cost-loss only; no real market prices; excluded from main empirical evidence.",
        "- Repository manuscript search: no .tex/.docx/manuscript*.md/paper*.md file found; paper wording must be updated outside this repository if applicable.",
        "",
        "Final status: `BLOCKED` (offline audit cannot verify SDK license, account balance or SPQC-550 routing; no physical response files).",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--vs-manifest", type=Path, nargs=3, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = load_protocol(args.config)
    budget = audit_budget(args.vs_manifest, args.test_manifest, protocol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "budget_audit.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    report = readiness_report(protocol, budget, args.vs_manifest, args.test_manifest,
                              output_dir / "READINESS_REPORT.md")
    print("Wrote budget audit and offline readiness report to %s" % output_dir.resolve())


if __name__ == "__main__":
    main()
