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


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "paper" / "fa_bm_vae_kaiwu_13k_protocol.json"
DEFAULT_OUTPUT = ROOT_DIR / "export" / "kaiwu_13k_preparation"


def resolve_path(path):
    path = Path(path)
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
    if protocol["dataset"]["selection_seed"] != 2026:
        raise ValueError("Selection seed must be 2026")
    if protocol["dataset"]["test_peeking"] is not False:
        raise ValueError("TEST peeking must be false")
    if protocol["vs"]["instances"] != 10 or protocol["vs"]["tasks"] != 30:
        raise ValueError("VS protocol must be 10 instances and 30 tasks")
    if protocol["test"]["instances"] != 100 or protocol["test"]["tasks"] != 100:
        raise ValueError("TEST protocol must be 100 instances and 100 tasks")
    if protocol["budget"]["planned_samples"] != 13000:
        raise ValueError("Planned sample budget must be 13000")
    if protocol["budget"]["application_upper_limit"] != 15000:
        raise ValueError("Application upper limit must be 15000")


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
    manifests = []
    id_sets = []
    for path, gain in zip(paths, expected_gains):
        items = load_manifest(path)
        if len(items) != protocol["vs"]["instances"]:
            raise ValueError("VS manifest %s must contain 10 instances" % path)
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
    if len(items) != protocol["test"]["instances"]:
        raise ValueError("TEST manifest must contain 100 instances")
    zones = []
    ids = []
    for item in items:
        if item.get("split") != "TEST" or item.get("selection") != "stratified":
            raise ValueError("TEST manifest must be stratified and split=TEST")
        if int(item.get("selection_seed")) != protocol["dataset"]["selection_seed"]:
            raise ValueError("TEST selection seed mismatch")
        if item.get("hardware_gain") is not None or item.get("hardware_matrix_sha256") is not None:
            raise ValueError("TEST manifest must not preselect a hardware gain")
        if item.get("hardware_n_bits") is not None:
            raise ValueError("TEST manifest must contain logical h,J only")
        if _forbidden_target_keys(item):
            raise ValueError("TEST manifest contains target values")
        zone = item.get("zone")
        if zone is None:
            raise ValueError("TEST manifest lacks zone provenance")
        zones.append(int(zone))
        ids.append(str(item["id"]))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate TEST instance IDs")
    counts = {zone: zones.count(zone) for zone in range(1, 11)}
    if counts != {zone: 10 for zone in range(1, 11)}:
        raise ValueError("TEST manifest must contain 10 instances per zone")
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
    }
    if result["vs_tasks"] != 30 or result["vs_samples"] != 3000:
        raise ValueError("VS budget mismatch")
    if result["test_tasks"] != 100 or result["test_samples"] != 10000:
        raise ValueError("TEST budget mismatch")
    if result["planned_samples"] != 13000:
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
    if reference.shape[0] < 100 or candidate.shape[0] < 100:
        raise ValueError("VS convergence comparison requires at least 100 reads")
    result = {}
    for count in (50, 100):
        metrics = compare_response_pair(h, j, reference[:count], candidate[:count])
        metrics["effective_beta_error"] = abs(float(metrics["candidate_beta_eff"]) - float(target_beta))
        result["reads_%d" % count] = metrics
    return result


def select_vs_gain(response_records, output_path, target_beta=1.0):
    """Select one gain from verified real VS records and write once."""
    if not response_records:
        raise ValueError("No VS response records supplied")
    for record in response_records:
        if float(record.get("gain")) not in (125.0, 150.0, 200.0):
            raise ValueError("VS response gain is outside the frozen candidate grid")
        if not record.get("hardware_claim") or not record.get("physical_platform_used"):
            raise ValueError("Gain selection requires verified physical VS responses")
        for key in ("energy_wasserstein", "effective_beta_error", "edge_moment_mae",
                    "magnetization_mae", "unique_state_fraction"):
            if key not in record:
                raise ValueError("VS response record lacks %s" % key)
    chosen = min(response_records, key=lambda row: (
        float(row["energy_wasserstein"]), float(row["effective_beta_error"]),
        float(row["edge_moment_mae"]), float(row["magnetization_mae"]),
        -float(row["unique_state_fraction"]),
    ))
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Frozen gain record already exists: %s" % output_path)
    payload = {
        "method_name": "FA-BM-VAE",
        "selection_split": "VS",
        "hardware_claim": True,
        "physical_platform_used": True,
        "selected_gain": float(chosen["gain"]),
        "selection_metrics": chosen,
        "immutable": True,
        "test_peeking": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def crps_ensemble(samples, target):
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ordered = np.sort(samples, axis=0)
    n = samples.shape[0]
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    return float(np.mean(np.abs(samples - target[None, :]) -
                         np.sum(ordered * coefficients[:, None], axis=0) / n ** 2))


def scenario_metrics(samples, target):
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if samples.ndim != 2 or target.shape != (samples.shape[1],):
        raise ValueError("samples/target shape must be (reads,horizon)/(horizon,)")
    crps = crps_ensemble(samples, target)
    energy = float(np.mean(np.linalg.norm(samples - target[None, :], axis=1)) -
                   0.5 * np.mean(np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2)))
    ramps = np.diff(samples, axis=1)
    target_ramp = np.diff(target)
    ramp_crps = crps_ensemble(ramps, target_ramp) if ramps.shape[1] else 0.0
    pairwise = np.abs(samples[:, :, None] - samples[:, None, :])
    target_pairwise = np.abs(target[:, None] - target[None, :])
    variogram = float(np.mean((pairwise.mean(axis=0) - target_pairwise) ** 2))
    coverages = np.asarray([
        np.mean(target <= np.quantile(samples, level, axis=0))
        for level in np.linspace(0.05, 0.95, 19)
    ])
    levels = np.linspace(0.05, 0.95, 19)
    return {
        "CRPS": crps,
        "Energy": energy,
        "Variogram": variogram,
        "Ramp_CRPS": ramp_crps,
        "coverage": float(np.mean((target >= np.quantile(samples, 0.05, axis=0)) &
                                   (target <= np.quantile(samples, 0.95, axis=0)))),
        "MAQCE": float(np.mean(np.abs(coverages - levels))),
    }


def paired_date_inference(differences, repetitions=2000, seed=2026):
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = rng.choice(differences, size=(repetitions, differences.size), replace=True).mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(repetitions, differences.size))
    null = (signs * differences[None, :]).mean(axis=1)
    mean = float(differences.mean())
    return {
        "mean_difference": mean,
        "ci_2.5": float(np.quantile(boot, 0.025)),
        "ci_97.5": float(np.quantile(boot, 0.975)),
        "paired_sign_flip_p": float((np.count_nonzero(np.abs(null) >= abs(mean)) + 1) / (repetitions + 1)),
        "effect_size": float(mean / (differences.std(ddof=1) + 1e-12)) if differences.size > 1 else 0.0,
        "n_instance_dates": int(differences.size),
    }


def aggregate_test_metrics(rows, reference_model="float_logical_sa", repetitions=2000, seed=2026):
    """Aggregate one metric row per instance/date; reads remain within each row."""
    if not rows:
        raise ValueError("No TEST metric rows supplied")
    grouped = {}
    for row in rows:
        key = (str(row["date"]), str(row["model"]))
        grouped.setdefault(key, {metric: [] for metric in ("CRPS", "Energy", "Variogram", "Ramp_CRPS", "coverage", "MAQCE")})
        for metric in grouped[key]:
            grouped[key][metric].append(float(row[metric]))
    means = {(date, model): {metric: float(np.mean(values)) for metric, values in data.items()}
             for (date, model), data in grouped.items()}
    models = sorted({model for _, model in means})
    if reference_model not in models:
        raise ValueError("Reference TEST model is absent")
    comparisons = []
    for model in models:
        if model == reference_model:
            continue
        for metric in ("CRPS", "Energy", "Variogram", "Ramp_CRPS", "coverage", "MAQCE"):
            diffs = [means[(date, model)][metric] - means[(date, reference_model)][metric]
                     for date in sorted({date for date, _ in means})
                     if (date, model) in means and (date, reference_model) in means]
            if diffs:
                comparisons.append({"model": model, "reference_model": reference_model,
                                    "metric": metric, **paired_date_inference(diffs, repetitions, seed)})
    return {"paired_comparisons": comparisons, "reads_are_independent_test_units": False}


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
        "- OPSD real origin_1/seed_0 four-model pilot completed in a separate export directory; formal 5-seed run remains NOT RUN.",
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
        "- Missing platform conditions: Sampling Credits, project number, licensed SDK, account environment, and real response files." if missing_platform else "- Platform conditions supplied.",
        "- Real TEST submission is blocked until one gain is frozen from VS only.",
        "",
        "## Manual-gated next command",
        "",
        "Run only after the user has supplied Sampling Credits, a project number, and a licensed account environment; this command was not executed:",
        "",
        "```powershell",
        "$env:FA_BM_VAE_RUN_REAL_KAIWU=\"1\"",
        "$env:KAIWU_PROJECT_NO=\"<manually-provided-project-no>\"",
        "$matrices = Get-ChildItem export\\kaiwu_13k_preparation\\submission_gain_125\\package\\hardware_matrices\\*.npz | Select-Object -First 10",
        "foreach ($matrix in $matrices) {",
        "  D:\\anaconda\\envs\\wsy\\python.exe -m GEFcom2014.models.QBM_VAE.submit_kaiwu_sampling `",
        "    --matrix-file $matrix.FullName --instance-id $matrix.BaseName --num-reads 100 `",
        "    --output-dir export\\kaiwu_13k_preparation\\real_vs_gain_125 `",
        "    --checkpoint-dir export\\kaiwu_13k_preparation\\real_checkpoints `",
        "    --confirm-real-submission",
        "}",
        "```",
        "",
        "## Unrelated experiments",
        "",
        "- OPSD rolling-origin: real origin_1/seed_0 pilot only; 3-origin x 5-seed formal run NOT RUN.",
        "- SDWPF-5: smoke/NOT RUN; excluded from this hardware budget.",
        "- Decision value: normalized asymmetric cost-loss real-input pilot completed; auditable market prices NOT USED.",
        "- Repository manuscript search: no .tex/.docx/manuscript*.md/paper*.md file found; paper wording must be updated outside this repository if applicable.",
        "",
        "Final status: `BLOCKED` (missing Sampling Credits, KAIWU_PROJECT_NO, licensed SDK/account environment, and verified physical response files).",
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
