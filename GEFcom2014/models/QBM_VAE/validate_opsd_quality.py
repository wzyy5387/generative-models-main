# -*- coding: utf-8 -*-
"""Independent, zero-credit validation of the frozen OPSD quality rule.

This module deliberately does not update thresholds, read TEST data, import the
Kaiwu SDK, or submit tasks.  Normal controls are compared with the same frozen
reference samples used by the formal metric path.  Anomaly controls are
deterministic transformations of normal samples and are never experimental
results.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .compare_hardware_responses import compare_response_pair
from .kaiwu_13k_protocol import load_protocol
from .opsd_quality_gate import ROOT_DIR, _load_instances, _load_logical_problem, _sample_float, sha256_file


GAINS = (125, 150, 200)
ERROR_METRICS = (
    "energy_wasserstein",
    "effective_beta_error",
    "edge_moment_mae",
    "magnetization_mae",
)
DEFAULT_SEED_BASE = 303000
N_BATCHES = 10
N_CONDITIONS = 10
MAX_SAMPLER_SEED = (2 ** 32) - 1
REQUIRED_VALIDATION_VERSIONS = ("v1", "v2", "v3", "v4")


def _finite_metric_row(row):
    return all(np.isfinite(float(row[name])) for name in ERROR_METRICS)


def _candidate_metrics(h, j, reference, candidate):
    metrics = compare_response_pair(h, j, reference, candidate)
    metrics["effective_beta_error"] = abs(float(metrics["candidate_beta_eff"]) - 1.0)
    metrics["unique_state_fraction"] = float(metrics["candidate_unique_state_fraction"])
    if not _finite_metric_row(metrics):
        raise ValueError("non-finite quality metric in independent validation")
    return {name: float(metrics[name]) for name in ERROR_METRICS + ("unique_state_fraction",)}


def _check_thresholds(thresholds):
    missing = [name for name in ERROR_METRICS if name not in thresholds]
    if missing:
        raise ValueError("threshold source is missing: " + ", ".join(missing))
    values = {name: float(thresholds[name]) for name in ERROR_METRICS}
    if not all(np.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError("threshold candidates must be finite and non-negative")
    return values


def _load_frozen_panel(preparation):
    preparation = Path(preparation)
    manifest_path = preparation / "quality_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {(int(row["gain"]), str(row["instance_id"])): row for row in manifest["instances"]}
    if len(entries) != 30:
        raise ValueError("frozen reference panel must contain 30 gain-condition entries")
    panels = {}
    for gain in GAINS:
        instances = _load_instances(preparation, gain)
        ids = {str(instance["id"]) for instance in instances}
        if len(ids) != 10:
            raise ValueError("each frozen gain panel must contain ten unique conditions")
        panels[gain] = []
        for instance in instances:
            key = (gain, str(instance["id"]))
            if key not in entries:
                raise ValueError("missing frozen reference entry: %s" % (key,))
            entry = entries[key]
            source_path = Path(instance["path"])
            if not source_path.is_absolute():
                source_path = ROOT_DIR / source_path
            reference_path = Path(entry["reference_sample_path"])
            if not reference_path.is_absolute():
                reference_path = ROOT_DIR / reference_path
            if sha256_file(source_path) != str(entry["source_problem_sha256"]):
                raise ValueError("frozen source problem hash mismatch: %s" % instance["id"])
            if sha256_file(reference_path) != str(entry["reference_sample_sha256"]):
                raise ValueError("frozen reference sample hash mismatch: %s" % instance["id"])
            h, j = _load_logical_problem(instance)
            with np.load(reference_path, allow_pickle=False) as archive:
                reference = np.asarray(archive["samples"], dtype=np.int8)
            if reference.shape != (100, h.size) or not np.isin(reference, (-1, 1)).all():
                raise ValueError("invalid frozen reference sample: %s" % instance["id"])
            panels[gain].append({
                "gain": gain,
                "instance_id": str(instance["id"]),
                "h": h,
                "j": j,
                "reference": reference,
                "reference_path": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
                "source_problem_path": str(source_path),
                "source_problem_sha256": sha256_file(source_path),
            })
    base_ids = {row["instance_id"] for row in panels[GAINS[0]]}
    if any({row["instance_id"] for row in panels[gain]} != base_ids for gain in GAINS[1:]):
        raise ValueError("gain panels do not share the same ten VS conditions")
    return panels


def _legacy_derived_seed(batch_seed, gain, condition_index):
    """The old allocation, retained only to test its collision behavior."""
    return int(batch_seed) + int(gain) * 1000 + int(condition_index)


def _derived_seed(gain_index, batch_index, condition_index, seed_base=DEFAULT_SEED_BASE):
    """Allocate one actual random seed per indexed gain/batch/condition."""
    if not (0 <= int(gain_index) < len(GAINS)):
        raise ValueError("gain_index is outside the frozen gain panel")
    if not (0 <= int(batch_index) < N_BATCHES):
        raise ValueError("batch_index is outside the fixed ten-batch plan")
    if not (0 <= int(condition_index) < N_CONDITIONS):
        raise ValueError("condition_index is outside the fixed ten-condition panel")
    seed = int(seed_base) + (
        (int(gain_index) * N_BATCHES + int(batch_index)) * N_CONDITIONS
        + int(condition_index)
    )
    if not (0 <= seed <= MAX_SAMPLER_SEED):
        raise ValueError("allocated seed is outside the sampler's supported range")
    return seed


def build_seed_plan(batch_count=N_BATCHES, condition_count=N_CONDITIONS,
                    seed_base=DEFAULT_SEED_BASE):
    if int(batch_count) != N_BATCHES or int(condition_count) != N_CONDITIONS:
        raise ValueError("the independent validation plan is fixed at 3x10x10")
    rows = []
    for gain_index, gain in enumerate(GAINS):
        for batch_index in range(N_BATCHES):
            for condition_index in range(N_CONDITIONS):
                rows.append({
                    "gain_index": gain_index,
                    "gain": gain,
                    "batch_index": batch_index,
                    "condition_index": condition_index,
                    "random_seed": _derived_seed(
                        gain_index, batch_index, condition_index, seed_base
                    ),
                })
    return rows


def _recorded_seeds(root):
    root = Path(root)
    seeds = set()
    files = 0
    if not root.exists():
        return {"root": str(root), "exists": False, "files_with_seed": 0,
                "unique_seeds": [], "seed_count": 0}
    for path in root.rglob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as archive:
                for key in ("random_seed", "derived_seed", "seed"):
                    if key in archive:
                        value = int(np.asarray(archive[key]).reshape(-1)[0])
                        seeds.add(value)
                        files += 1
                        break
        except (OSError, ValueError, KeyError):
            continue
    return {"root": str(root), "exists": True, "files_with_seed": files,
            "unique_seeds": sorted(seeds), "seed_count": len(seeds)}


def _array_sha256(samples):
    digest = hashlib.sha256()
    digest.update(str(samples.dtype).encode("ascii"))
    digest.update(repr(tuple(samples.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(samples).tobytes())
    return digest.hexdigest()


def _audit_normal_sample_files(root, expected_count=300):
    """Audit recorded normal samples; do not count deterministic anomalies."""
    root = Path(root)
    normal_root = root / "normal_control"
    if not root.is_dir() or not normal_root.is_dir():
        raise FileNotFoundError("missing required normal-control directory: %s" % normal_root)
    files = sorted(normal_root.rglob("*.npz"))
    if len(files) != expected_count:
        raise ValueError("%s must contain exactly %d normal sample files; found %d" %
                         (normal_root, expected_count, len(files)))
    records = []
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "samples" not in archive:
                    raise ValueError("missing samples array")
                samples = np.asarray(archive["samples"], dtype=np.int8)
                if samples.shape != (100, 48) or not np.isin(samples, (-1, 1)).all():
                    raise ValueError("invalid normal sample shape or spin values")
                seed_key = next((key for key in ("random_seed", "derived_seed", "seed")
                                 if key in archive), None)
                if seed_key is None:
                    raise ValueError("normal sample has no recorded random seed")
                random_seed = int(np.asarray(archive[seed_key]).reshape(-1)[0])
                metadata = {
                    "gain": int(np.asarray(archive["gain"]).reshape(-1)[0])
                    if "gain" in archive else None,
                    "gain_index": int(np.asarray(archive["gain_index"]).reshape(-1)[0])
                    if "gain_index" in archive else None,
                    "batch_index": int(np.asarray(archive["batch_index"]).reshape(-1)[0])
                    if "batch_index" in archive else None,
                    "instance_id": str(np.asarray(archive["instance_id"]).reshape(-1)[0])
                    if "instance_id" in archive else None,
                    "reads": int(np.asarray(archive["reads"]).reshape(-1)[0])
                    if "reads" in archive else None,
                    "sweeps": int(np.asarray(archive["sweeps"]).reshape(-1)[0])
                    if "sweeps" in archive else None,
                }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("corrupt or incomplete normal sample %s: %s" % (path, exc)) from exc
        records.append({
            "path": str(path), "relative_path": str(path.relative_to(normal_root)),
            "sha256": sha256_file(path), "array_sha256": _array_sha256(samples),
            "seed": random_seed, "seed_key": seed_key, **metadata,
        })
    if len({row["path"] for row in records}) != expected_count:
        raise ValueError("normal sample paths are not unique")
    return records


def _audit_reference_sample_files(preparation, threshold_source):
    """Audit all seed-bearing files used for the threshold reference."""
    threshold_root = Path(threshold_source).parent
    reference_files = sorted((threshold_root / "samples").glob("gain_*/repeat_*/*.npz"))
    if len(reference_files) != 300:
        raise ValueError("threshold reference must contain exactly 300 repeat sample files")
    records = []
    for path in reference_files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "samples" not in archive:
                    raise ValueError("missing samples array")
                samples = np.asarray(archive["samples"], dtype=np.int8)
                if samples.shape != (100, 48) or not np.isin(samples, (-1, 1)).all():
                    raise ValueError("invalid reference sample shape or spin values")
                if "seed" not in archive:
                    raise ValueError("reference sample has no recorded seed")
                seed = int(np.asarray(archive["seed"]).reshape(-1)[0])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("corrupt or incomplete threshold reference %s: %s" % (path, exc)) from exc
        records.append({"path": str(path), "sha256": sha256_file(path),
                        "array_sha256": _array_sha256(samples), "seed": seed})
    return records


def audit_existing_validation_run(root, version):
    """Audit one historical run without treating it as a new experiment."""
    records = _audit_normal_sample_files(root)
    seeds = [row["seed"] for row in records]
    keys = {(row["gain"], row["batch_index"], row["instance_id"]) for row in records}
    if None in {row["gain"] for row in records} or None in {row["batch_index"] for row in records}:
        raise ValueError("%s normal samples lack gain or batch metadata" % root)
    if len(keys) != 300:
        raise ValueError("%s normal sample metadata has duplicate condition keys" % root)
    manifest_path = Path(root) / "validation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    return {
        "version": version,
        "root": str(Path(root)),
        "normal_file_count": len(records),
        "unique_seed_count": len(set(seeds)),
        "seed_min": min(seeds), "seed_max": max(seeds),
        "seed_reuse_count": len(seeds) - len(set(seeds)),
        "normal_sample_file_hash_digest": hashlib.sha256(
            "".join(row["sha256"] for row in records).encode("ascii")
        ).hexdigest(),
        "normal_sample_array_digest": hashlib.sha256(
            "".join(row["array_sha256"] for row in records).encode("ascii")
        ).hexdigest(),
        "seed_mapping": [
            {key: row[key] for key in ("relative_path", "seed", "seed_key",
                                        "gain", "gain_index", "batch_index", "instance_id")}
            for row in records
        ],
        "manifest_present": manifest is not None,
        "manifest": manifest or {},
        "independent_sample_count": 0 if version == "v4" else len(records),
        "artifact_class": "historical_run",
    }


def compare_validation_runs(first, second):
    """Compare recorded runs and identify exact reproduction."""
    same_arrays = first["normal_sample_array_digest"] == second["normal_sample_array_digest"]
    same_files = first["normal_sample_file_hash_digest"] == second["normal_sample_file_hash_digest"]
    same_seeds = first["seed_mapping"] == second["seed_mapping"]
    first_manifest, second_manifest = first.get("manifest", {}), second.get("manifest", {})
    comparable_manifest_keys = (
        "protocol_sha256", "preparation_manifest_sha256", "threshold_source_sha256",
        "seed_base", "seed_rule", "gains", "conditions_per_gain",
        "reads_per_condition", "sweeps", "thresholds",
    )
    manifest_comparison = {
        key: first_manifest.get(key) == second_manifest.get(key)
        for key in comparable_manifest_keys
    }
    exact_reproduction = same_arrays and same_files and same_seeds and all(manifest_comparison.values())
    return {
        "same_sample_arrays": same_arrays,
        "same_sample_files": same_files,
        "same_seed_mapping": same_seeds,
        "manifest_comparison": manifest_comparison,
        "exact_reproduction": exact_reproduction,
        "additional_independent_samples": 0 if exact_reproduction else second["normal_file_count"],
    }


def audit_existing_validation_runs(preparation, threshold_source, output_dir,
                                   validation_roots=None):
    """Fail closed while auditing the threshold reference and v1-v4 history."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("audit output must be a new, empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_records = _audit_reference_sample_files(preparation, threshold_source)
    roots = _resolve_validation_roots(threshold_source, validation_roots)
    runs = {version: audit_existing_validation_run(roots[version], version)
            for version in REQUIRED_VALIDATION_VERSIONS}
    v3_v4 = compare_validation_runs(runs["v3"], runs["v4"])
    if not v3_v4["exact_reproduction"]:
        raise ValueError("v4 is not an exact reproduction/audit completion of v3")
    historical_seed_sets = {
        "threshold_reference": set(row["seed"] for row in threshold_records),
        **{
            "validation_" + version: set(
                row["seed"] for row in _audit_normal_sample_files(roots[version])
            )
            for version in REQUIRED_VALIDATION_VERSIONS
        },
    }
    historical_intersections = {}
    for version in REQUIRED_VALIDATION_VERSIONS:
        own = set(row["seed"] for row in _audit_normal_sample_files(roots[version]))
        historical_intersections[version] = {
            source: sorted(own.intersection(seeds))
            for source, seeds in historical_seed_sets.items()
            if source != "validation_" + version
        }
    report = {
        "artifact_role": "historical_validation_audit",
        "threshold_reference": {
            "file_count": len(threshold_records),
            "unique_seed_count": len({row["seed"] for row in threshold_records}),
            "seed_min": min(row["seed"] for row in threshold_records),
            "seed_max": max(row["seed"] for row in threshold_records),
        },
        "runs": runs,
        "v3_v4_comparison": v3_v4,
        "historical_seed_intersections": historical_intersections,
        "independent_evidence_count": {
            "v1": 0, "v2": 0, "v3": runs["v3"]["normal_file_count"], "v4": 0,
            "reason": "v1/v2 reuse seeds; v4 exactly reproduces v3 and is audit completion only",
        },
        "thresholds_changed_in_artifacts": False,
        "model_or_sampling_change_during_period": "not_recoverable_from_artifacts; no difference found between v3/v4 manifests",
        "selective_retention_based_on_results": "not_recoverable from stored artifacts",
        "test_data_used": False,
        "hardware_claim": False,
        "physical_platform_used": False,
    }
    (output_dir / "historical_seed_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "historical_seed_audit.md").write_text(
        "# Historical seed audit\n\n"
        "The threshold reference and validation directories v1-v4 were audited fail-closed. "
        "v1/v2 contain seed reuse and are not independent evidence. v3 is the corrected "
        "independent run. v4 has identical sample arrays and seed mapping to v3 and is "
        "classified as reproduction/audit completion; it adds zero independent samples.\n\n"
        "`thresholds_changed=false`, `test_data_used=false`, `hardware_claim=false`, "
        "and `physical_platform_used=false`.\n",
        encoding="utf-8",
    )
    return report


def _resolve_validation_roots(threshold_source, prior_validation_dirs=None):
    """Resolve the required v1-v4 history without silently filling gaps."""
    if prior_validation_dirs is None:
        base = Path(threshold_source).parents[2]
        return {
            version: base / ("opsd_sa_quality_validation_20261018_" + version)
            for version in REQUIRED_VALIDATION_VERSIONS
        }
    if isinstance(prior_validation_dirs, dict):
        missing = [version for version in REQUIRED_VALIDATION_VERSIONS
                   if version not in prior_validation_dirs]
        if missing:
            raise ValueError(
                "strict historical audit is missing required validation directories: "
                + ", ".join(missing)
            )
        return {version: Path(prior_validation_dirs[version])
                for version in REQUIRED_VALIDATION_VERSIONS}
    paths = list(prior_validation_dirs)
    if len(paths) != len(REQUIRED_VALIDATION_VERSIONS):
        raise ValueError(
            "strict historical audit requires validation directories v1-v4"
        )
    return {
        version: Path(path)
        for version, path in zip(REQUIRED_VALIDATION_VERSIONS, paths)
    }


def strict_recorded_seed_audit(threshold_source, prior_validation_dirs=None):
    """Audit all required history before a new validation can sample.

    The returned ``sources`` mapping is intentionally shaped for
    ``validate_seed_plan``.  Unlike the legacy audit, missing, damaged, or
    seed-less historical artifacts raise before the caller can sample.
    """
    threshold_records = _audit_reference_sample_files(None, threshold_source)
    roots = _resolve_validation_roots(threshold_source, prior_validation_dirs)
    runs = {
        version: audit_existing_validation_run(roots[version], version)
        for version in REQUIRED_VALIDATION_VERSIONS
    }
    sources = {
        "threshold_reference": {
            "root": str(Path(threshold_source).parent),
            "exists": True,
            "files_with_seed": len(threshold_records),
            "unique_seeds": sorted({row["seed"] for row in threshold_records}),
            "seed_count": len({row["seed"] for row in threshold_records}),
        },
    }
    for version in REQUIRED_VALIDATION_VERSIONS:
        run = runs[version]
        sources["validation_" + version] = {
            "root": run["root"],
            "exists": True,
            "files_with_seed": run["normal_file_count"],
            "unique_seeds": sorted({row["seed"] for row in run["seed_mapping"]}),
            "seed_count": run["unique_seed_count"],
            "seed_reuse_count": run["seed_reuse_count"],
        }
    return {
        "required_versions": list(REQUIRED_VALIDATION_VERSIONS),
        "threshold_reference": sources["threshold_reference"],
        "runs": {
            version: {
                "root": runs[version]["root"],
                "normal_file_count": runs[version]["normal_file_count"],
                "unique_seed_count": runs[version]["unique_seed_count"],
                "seed_reuse_count": runs[version]["seed_reuse_count"],
                "artifact_class": runs[version]["artifact_class"],
            }
            for version in REQUIRED_VALIDATION_VERSIONS
        },
        "sources": sources,
        "status": "passed",
    }


def recorded_seed_audit(threshold_source, prior_validation_dirs=None):
    """Compatibility wrapper for the strict pre-sampling audit."""
    return strict_recorded_seed_audit(
        threshold_source, prior_validation_dirs
    )["sources"]


def validate_seed_plan(seed_plan, recorded_sources):
    """Fail before sampling if the complete allocation is invalid or reused."""
    if len(seed_plan) != 300:
        raise ValueError("seed plan must contain exactly 300 entries")
    actual = [int(row["random_seed"]) for row in seed_plan]
    if len(set(actual)) != len(actual):
        raise ValueError("new seed plan contains duplicate random seeds")
    expected = {(gi, bi, ci) for gi in range(3)
                for bi in range(N_BATCHES) for ci in range(N_CONDITIONS)}
    observed = {(int(row["gain_index"]), int(row["batch_index"]),
                 int(row["condition_index"])) for row in seed_plan}
    if observed != expected:
        raise ValueError("seed plan indices are incomplete or invalid")
    overlaps = {}
    for name, source in recorded_sources.items():
        overlap = sorted(set(actual).intersection(source["unique_seeds"]))
        if overlap:
            overlaps[name] = overlap
    if overlaps:
        raise ValueError("new seed plan overlaps recorded seeds: " + json.dumps(overlaps))
    return {
        "n_entries": len(seed_plan),
        "unique_random_seeds": len(set(actual)),
        "min_random_seed": min(actual),
        "max_random_seed": max(actual),
        "overlaps": overlaps,
        "status": "passed",
    }


def _save_samples(path, samples, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, samples=np.asarray(samples, dtype=np.int8), **metadata)


def _evaluate_control(rows, thresholds):
    batch_rows = []
    grouped = {}
    for row in rows:
        grouped.setdefault((row["control"], row["gain"], row["batch_index"]), []).append(row)
    for key, group in sorted(grouped.items()):
        metrics = {name: float(np.mean([item[name] for item in group])) for name in ERROR_METRICS}
        failed = [name for name in ERROR_METRICS if metrics[name] > thresholds[name]]
        batch_rows.append({
            "control": key[0], "gain": int(key[1]), "batch_index": int(key[2]),
            "random_seed_range": [
                int(min(item["random_seed"] for item in group)),
                int(max(item["random_seed"] for item in group)),
            ], "n_conditions": len(group),
            **metrics, "failed_metrics": failed, "joint_pass": not failed,
        })
    failure_counts = {
        name: int(sum(name in row["failed_metrics"] for row in batch_rows))
        for name in ERROR_METRICS
    }
    passed = int(sum(bool(row["joint_pass"]) for row in batch_rows))
    return {
        "n_gain_batches": len(batch_rows),
        "joint_pass_count": passed,
        "joint_pass_rate": float(passed / len(batch_rows)) if batch_rows else 0.0,
        "metric_failure_counts": failure_counts,
        "batch_rows": batch_rows,
    }


def _condition_diagnostics(rows, thresholds):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["control"], row["gain"], row["instance_id"]), []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        means = {name: float(np.mean([item[name] for item in group])) for name in ERROR_METRICS}
        row_failed = [
            [name for name in ERROR_METRICS if float(item[name]) > thresholds[name]]
            for item in group
        ]
        failure_counts = {name: int(sum(name in failed for failed in row_failed))
                          for name in ERROR_METRICS}
        output.append({
            "control": key[0], "gain": int(key[1]), "instance_id": key[2],
            "n_batches": len(group), "mean_metrics": means,
            "metric_failure_counts": failure_counts,
            "joint_pass_count": int(sum(not failed for failed in row_failed)),
            "joint_pass_rate": float(sum(not failed for failed in row_failed) / len(group)),
        })
    return output


def run_independent_validation(preparation, protocol_path, threshold_source, output_dir,
                               seed_base=DEFAULT_SEED_BASE, reads=100, sweeps=100,
                               prior_validation_dirs=None):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("independent validation output must be a new, empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_payload = json.loads(Path(threshold_source).read_text(encoding="utf-8"))
    thresholds = _check_thresholds(thresholds_payload["suggested_thresholds"])
    panels = _load_frozen_panel(preparation)
    protocol = load_protocol(protocol_path)
    if tuple(int(x) for x in protocol["hardware"]["candidate_gains"]) != GAINS:
        raise ValueError("protocol candidate gains do not match the frozen validation panel")
    seed_plan = build_seed_plan(seed_base=seed_base)
    history_audit = strict_recorded_seed_audit(
        threshold_source, prior_validation_dirs
    )
    seed_sources = history_audit["sources"]
    seed_audit = validate_seed_plan(seed_plan, seed_sources)
    seed_config = {
        "seed_allocation": "seed_base + ((gain_index*10 + batch_index)*10 + condition_index)",
        "seed_base": int(seed_base),
        "gains": list(GAINS), "batch_count": N_BATCHES,
        "condition_count": N_CONDITIONS,
        "entries": seed_plan,
        "recorded_seed_sources": seed_sources,
        "historical_seed_audit": history_audit,
        "audit": seed_audit,
    }
    (output_dir / "seed_plan.json").write_text(
        json.dumps(seed_config, indent=2), encoding="utf-8"
    )
    normal_rows = []
    anomaly_rows = []
    anomaly_kinds = ("collapse", "reverse_bit_order")
    for gain_index, gain in enumerate(GAINS):
        for batch_index in range(N_BATCHES):
            for condition_index, item in enumerate(panels[gain]):
                seed = _derived_seed(gain_index, batch_index, condition_index, seed_base)
                normal = _sample_float(item["h"], item["j"], reads, sweeps, seed)
                normal_path = output_dir / "normal_control" / ("gain_%d" % gain) / ("batch_%02d" % batch_index) / (item["instance_id"] + ".npz")
                _save_samples(normal_path, normal, {
                    "validation_role": "normal_control", "batch_index": batch_index,
                    "random_seed": seed, "gain_index": gain_index, "gain": gain,
                    "instance_id": item["instance_id"], "reads": reads, "sweeps": sweeps,
                    "hardware_claim": False, "physical_platform_used": False,
                })
                normal_metrics = _candidate_metrics(item["h"], item["j"], item["reference"], normal)
                normal_rows.append({
                    "control": "normal", "batch_index": batch_index, "random_seed": seed,
                    "gain_index": gain_index, "gain": gain,
                    "instance_id": item["instance_id"],
                    "sample_path": str(normal_path), "sample_sha256": sha256_file(normal_path),
                    "reference_sample_path": item["reference_path"],
                    "reference_sample_sha256": item["reference_sha256"],
                    **normal_metrics,
                })
                anomalies = {
                    "collapse": np.repeat(normal[:1], normal.shape[0], axis=0),
                    "reverse_bit_order": normal[:, ::-1].copy(),
                }
                for anomaly_kind, anomaly in anomalies.items():
                    anomaly_path = output_dir / "anomaly_control" / anomaly_kind / ("gain_%d" % gain) / ("batch_%02d" % batch_index) / (item["instance_id"] + ".npz")
                    _save_samples(anomaly_path, anomaly, {
                        "validation_role": "anomaly_control", "anomaly_kind": anomaly_kind,
                        "source_normal_sample_sha256": sha256_file(normal_path),
                        "batch_index": batch_index, "random_seed": seed,
                        "gain_index": gain_index, "gain": gain,
                        "instance_id": item["instance_id"], "reads": reads,
                        "hardware_claim": False, "physical_platform_used": False,
                    })
                    anomaly_metrics = _candidate_metrics(item["h"], item["j"], item["reference"], anomaly)
                    anomaly_rows.append({
                        "control": anomaly_kind, "batch_index": batch_index, "random_seed": seed,
                        "gain_index": gain_index, "gain": gain,
                        "instance_id": item["instance_id"],
                        "sample_path": str(anomaly_path), "sample_sha256": sha256_file(anomaly_path),
                        "reference_sample_path": item["reference_path"],
                        "reference_sample_sha256": item["reference_sha256"],
                        **anomaly_metrics,
                    })
    controls = {"normal": normal_rows}
    for anomaly_kind in anomaly_kinds:
        controls[anomaly_kind] = [row for row in anomaly_rows if row["control"] == anomaly_kind]
    summaries = {name: _evaluate_control(rows, thresholds) for name, rows in controls.items()}
    diagnostics = []
    for rows in controls.values():
        diagnostics.extend(_condition_diagnostics(rows, thresholds))
    manifest = {
        "artifact_role": "independent_zero_credit_quality_validation",
        "validation_status": "completed_locally",
        "preparation": str(Path(preparation)),
        "protocol": str(Path(protocol_path)),
        "threshold_source": str(Path(threshold_source)),
        "protocol_sha256": sha256_file(protocol_path),
        "preparation_manifest_sha256": sha256_file(Path(preparation) / "preparation_manifest.json"),
        "threshold_source_sha256": sha256_file(threshold_source),
        "seed_base": int(seed_base),
        "seed_rule": seed_config["seed_allocation"],
        "seed_plan_path": str(output_dir / "seed_plan.json"),
        "seed_audit": seed_audit,
        "recorded_seed_sources": seed_sources,
        "historical_seed_audit": history_audit,
        "gains": list(GAINS), "conditions_per_gain": 10,
        "reads_per_condition": reads, "sweeps": sweeps,
        "normal_reference_sample_rule": "same frozen reference_sample_path from quality_reference_manifest; repeat_00",
        "thresholds": thresholds,
        "thresholds_changed": False, "thresholds_approved": False,
        "test_data_used": False, "test_peeking": False,
        "anomaly_controls": list(anomaly_kinds),
        "anomaly_controls_are_results": False,
        "hardware_claim": False, "physical_platform_used": False,
    }
    (output_dir / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "normal_rows.json").write_text(json.dumps(normal_rows, indent=2), encoding="utf-8")
    (output_dir / "anomaly_rows.json").write_text(json.dumps(anomaly_rows, indent=2), encoding="utf-8")
    (output_dir / "condition_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    summary = {
        "artifact_role": manifest["artifact_role"],
        "thresholds": thresholds,
        "controls": summaries,
        "condition_diagnostics_path": str(output_dir / "condition_diagnostics.json"),
        "normal_joint_pass_rate": summaries["normal"]["joint_pass_rate"],
        "anomaly_detection": {
            name: {"joint_rejection_rate": 1.0 - summaries[name]["joint_pass_rate"],
                   "joint_pass_rate": summaries[name]["joint_pass_rate"]}
            for name in anomaly_kinds
        },
        "hardware_claim": False, "physical_platform_used": False,
        "eligible_for_paper": False,
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--threshold-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reads", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--audit-existing", action="store_true",
                        help="audit threshold reference and existing v1-v4 runs without sampling")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.audit_existing:
        report = audit_existing_validation_runs(
            args.preparation, args.threshold_source, args.output_dir
        )
        print(json.dumps({
            "artifact_role": report["artifact_role"],
            "v3_v4_exact_reproduction": report["v3_v4_comparison"]["exact_reproduction"],
            "independent_evidence_count": report["independent_evidence_count"],
        }, indent=2))
        return
    summary = run_independent_validation(
        args.preparation, args.protocol, args.threshold_source, args.output_dir,
        reads=args.reads, sweeps=args.sweeps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
