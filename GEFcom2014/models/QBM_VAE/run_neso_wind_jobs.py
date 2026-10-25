# -*- coding: utf-8 -*-
"""Run isolated NESO Wind smoke or formal classical jobs with provenance logs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "export" / "fa_bm_vae_neso_wind_classical_v1"
BUNDLE = OUTPUT / "data" / "neso_wind_daily.npz"
CONFIGS = {
    "fa_bm_vae": ROOT / "configs/paper/fa_bm_vae_neso_wind_sa_negative_v1.json",
    "j0": ROOT / "configs/paper/fa_bm_vae_neso_wind_sa_negative_j0_v1.json",
    "no_anchor_addition": ROOT / "configs/paper/fa_bm_vae_neso_wind_sa_negative_no_anchor_addition_v1.json",
}
SOURCES = (
    "GEFcom2014/neso_wind_dataset.py",
    "GEFcom2014/external_datasets.py",
    "GEFcom2014/models/utils_models.py",
    "GEFcom2014/models/QBM_VAE/qbm_vae.py",
    "GEFcom2014/models/QBM_VAE/utils_qbm_vae.py",
    "GEFcom2014/models/QBM_VAE/forecast_anchor.py",
    "GEFcom2014/models/QBM_VAE/anchor_gaussian_baseline.py",
    "GEFcom2014/models/QBM_VAE/anchor_score_sde.py",
    "GEFcom2014/models/QBM_VAE/anchor_spline_flow.py",
    "GEFcom2014/models/probabilistic_baselines.py",
    "GEFcom2014/forecast_quality/temporal_rank_coupling.py",
    "GEFcom2014/forecast_quality/compare_scenarios.py",
    "GEFcom2014/forecast_quality/unified_postprocessing.py",
    "GEFcom2014/forecast_quality/qbm_temporal_ecc.py",
    "GEFcom2014/forecast_quality/neso_wind_results.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _output_hashes(job, bundle):
    """Check a finished task before it can be counted or reused."""
    out = Path(job["output_dir"])
    metadata_files = list(out.glob("*.json"))
    if len(metadata_files) != 1:
        raise ValueError("Expected exactly one task metadata JSON in " + str(out))
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    seed = job["training_seed"]
    if (int(metadata.get("training_seed", metadata.get("seed", -1))) != seed
            or int(metadata.get("generation_seed", -1)) != job["generation_seed"]
            or Path(metadata.get("dataset_bundle", "")).resolve() != Path(bundle).resolve()
            or list(metadata.get("scenario_splits", [])) != job["scenario_splits"]):
        raise ValueError("Task metadata does not match frozen bundle, seeds or splits: " + str(out))
    if "n_scenarios" in metadata and int(metadata["n_scenarios"]) != job["n_scenarios"]:
        raise ValueError("Task metadata has the wrong scenario count: " + str(out))
    if job["graph_seed"] is not None and int(metadata.get("graph_seed", -1)) != job["graph_seed"]:
        raise ValueError("Task metadata has the wrong graph seed: " + str(out))
    if job["model"] in CONFIGS:
        command = job["command"]
        expected_config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
        if metadata.get("graph_mode") != expected_config.get("graph_mode"):
            raise ValueError("Task graph mode differs from frozen configuration: " + str(out))
    else:
        expected_models = {"anchor_gaussian": "forecast_anchored_conditional_diagonal_gaussian",
                           "anchor_gmm4": "forecast_anchored_conditional_diagonal_gaussian_mixture",
                           "score_sde": "forecast_anchored_conditional_edm_score_sde",
                           "spline_flow": "forecast_anchored_conditional_rational_quadratic_spline_flow",
                           "residual_bootstrap": "residual-bootstrap"}
        if metadata.get("model") != expected_models[job["model"]]:
            raise ValueError("Task model identity differs from requested model: " + str(out))
    if job["model"] == "j0" and (metadata.get("graph_mode") != "none" or metadata.get("graph_edges") != 0):
        raise ValueError("J=0 task did not use a zero-coupling graph: " + str(out))
    if job["model"] == "anchor_gmm4" and metadata.get("components") != 4:
        raise ValueError("GMM task did not use four components: " + str(out))
    with np.load(bundle, allow_pickle=False) as arrays:
        for split in job["scenario_splits"]:
            matches = list(out.glob("*_{}_{}.pickle".format(job["n_scenarios"], split)))
            if len(matches) != 1:
                raise ValueError("Expected one %s scenario file in %s" % (split, out))
            with matches[0].open("rb") as handle:
                samples = np.asarray(pickle.load(handle), dtype=np.float64)
            expected = (len(arrays["y_" + split.lower()]) * 24, job["n_scenarios"])
            if samples.shape != expected or not np.isfinite(samples).all():
                raise ValueError("Invalid %s scenario matrix in %s; expected %s" % (split, out, expected))
    files = sorted(path for path in out.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Task produced no files: " + str(out))
    return {str(path.relative_to(out)): sha256(path) for path in files}


def _validated_history(run_root, manifest, task_list, bundle):
    history_path = run_root / "job_results.json"
    if not history_path.is_file():
        raise ValueError("Existing run has no task state; refusing automatic resume")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) > len(task_list):
        raise ValueError("Invalid task-state history")
    if manifest.get("status") == "COMPLETED" and len(history) != len(task_list):
        raise ValueError("Completed manifest has an incomplete task-state history")
    for index, record in enumerate(history):
        job = task_list[index]
        if (record.get("model") != job["model"] or record.get("training_seed") != job["training_seed"]
                or record.get("generation_seed") != job["generation_seed"]
                or record.get("command") != job["command"]
                or record.get("output_dir") != str(Path(job["output_dir"]).resolve())):
            raise ValueError("Task-state identity mismatch at task %d" % (index + 1))
        if record.get("status") != "COMPLETED":
            raise ValueError("Task %d is %s; refusing automatic resubmission" % (index + 1, record.get("status")))
        recorded_hashes = record.get("artifact_hashes")
        if not isinstance(recorded_hashes, dict) or not recorded_hashes:
            raise ValueError("Completed task lacks output hashes: %d" % (index + 1))
        log = Path(record.get("log", ""))
        expected_log = (run_root / "run_logs" / ("%02d_%s_seed_%d.log" %
                        (index + 1, job["model"], job["training_seed"]))).resolve()
        if log.resolve() != expected_log or not log.is_file():
            raise ValueError("Completed task log identity is missing: %d" % (index + 1))
        if sha256(log) != record.get("log_sha256"):
            raise ValueError("Completed task log changed: %d" % (index + 1))
        if _output_hashes(job, bundle) != recorded_hashes:
            raise ValueError("Completed task output changed: %d" % (index + 1))
    return history


def jobs(mode, run_label=None, output_root=OUTPUT, bundle=BUNDLE, models=None, config_dir=None):
    smoke = mode == "smoke"
    root = Path(output_root) / (run_label or ("smoke" if smoke else "formal"))
    bundle = Path(bundle)
    configs = {key: Path(config_dir) / path.name if config_dir else path for key, path in CONFIGS.items()}
    models = tuple(models) if models is not None else tuple((*MODEL_KEYS,))
    seeds = (0,) if smoke else (0, 1, 2)
    n_scenarios = 5 if smoke else 100
    split_args = ["--scenario-splits", "VS"] if smoke else ["--scenario-splits", "VS", "TEST"]
    jobs_out = []
    for model in models:
        if model not in MODEL_KEYS:
            raise ValueError("Unknown NESO model: " + model)
        for seed in seeds:
            out = root / model / ("seed_%d" % seed)
            common = ["--tag", "neso-wind", "--dataset-bundle", str(bundle), "--seed", str(seed),
                      "--generation-seed", str(seed), "--n-scenarios", str(n_scenarios), *split_args,
                      "--output-dir", str(out)]
            if model in configs:
                cmd = [sys.executable, "-m", "GEFcom2014.models.QBM_VAE.qbm_vae", "--config", str(configs[model]),
                       "--tag", "neso-wind", "--dataset-bundle", str(bundle), "--seed", str(seed),
                       "--generation-seed", str(seed), "--graph-seed", "20261020", "--n-scenarios", str(n_scenarios),
                       *split_args, "--output-dir", str(out), "--skip-plots"]
                if smoke:
                    cmd += ["--epochs", "1"]
            elif model in ("anchor_gaussian", "anchor_gmm4"):
                cmd = [sys.executable, "-m", "GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline",
                       *common, "--anchor-source", "context", "--epochs", "1" if smoke else "80",
                       "--hidden-dim", "256", "--layers", "2", "--batch-size", "256", "--patience", "20",
                       "--components", "1" if model == "anchor_gaussian" else "4"]
            elif model == "score_sde":
                cmd = [sys.executable, "-m", "GEFcom2014.models.QBM_VAE.anchor_score_sde",
                       *common, "--anchor-source", "context", "--epochs", "1" if smoke else "200",
                       "--batch-size", "128", "--channels", "64", "--blocks", "4", "--device", "auto"]
            elif model == "spline_flow":
                cmd = [sys.executable, "-m", "GEFcom2014.models.QBM_VAE.anchor_spline_flow",
                       *common, "--anchor-source", "context", "--epochs", "1" if smoke else "200",
                       "--batch-size", "128", "--hidden-dim", "256", "--num-transforms", "5", "--num-bins", "8", "--device", "auto"]
            else:
                cmd = [sys.executable, "-m", "GEFcom2014.models.probabilistic_baselines", "--model", "residual-bootstrap",
                       *common]
            jobs_out.append({"model": model, "training_seed": seed, "generation_seed": seed,
                             "graph_seed": 20261020 if model in CONFIGS else None,
                             "scenario_splits": ["VS"] if smoke else ["VS", "TEST"],
                             "n_scenarios": n_scenarios, "output_dir": str(out), "command": cmd})
    return jobs_out


MODEL_KEYS = ("fa_bm_vae", "anchor_gaussian", "anchor_gmm4", "score_sde", "spline_flow",
              "residual_bootstrap", "j0", "no_anchor_addition")


def run(mode, run_label=None, output_root=OUTPUT, bundle=BUNDLE, models=None, config_dir=None,
        resume=False):
    bundle = Path(bundle)
    output_root = Path(output_root)
    if not bundle.is_file():
        raise FileNotFoundError("Build the frozen NESO bundle before training: %s" % bundle)
    run_label = run_label or ("smoke" if mode == "smoke" else "formal")
    run_root = output_root / run_label
    task_list = jobs(mode, run_label, output_root, bundle, models, config_dir)
    manifest = {
        "mode": mode, "artifact_role": "smoke_NOT_FOR_TABLES" if mode == "smoke" else "formal_classical_training_and_generation",
        "bundle": str(bundle.resolve()), "bundle_sha256": sha256(bundle),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout,
        "source_hashes": {p: sha256(ROOT / p) for p in SOURCES},
        "config_hashes": {key: sha256(Path(config_dir) / path.name if config_dir else path)
                          for key, path in CONFIGS.items() if key in {job["model"] for job in task_list}},
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "nflows", "diffusers") if _has_package(name)},
        "jobs": task_list,
    }
    manifest_path = run_root / "task_manifest.json"
    history_path = run_root / "job_results.json"
    log_root = run_root / "run_logs"
    if run_root.exists():
        if not resume:
            raise FileExistsError("Refusing to overwrite existing run directory: %s" % run_root)
        if not manifest_path.is_file() or not log_root.is_dir():
            raise ValueError("Existing run lacks a complete frozen manifest or log directory")
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity_fields = ("mode", "artifact_role", "bundle", "bundle_sha256", "source_hashes",
                           "config_hashes", "python", "packages", "jobs")
        if any(frozen.get(field) != manifest[field] for field in identity_fields):
            raise ValueError("Frozen run identity differs from current bundle, code, configuration or task list")
        results = _validated_history(run_root, frozen, task_list, bundle)
        manifest = frozen
    else:
        if resume:
            raise FileNotFoundError("Cannot resume a run that does not exist: %s" % run_root)
        run_root.mkdir(parents=True)
        log_root.mkdir()
        results = []
        manifest["status"] = "RUNNING"
        _write_json(manifest_path, manifest)
        _write_json(history_path, results)
    for index in range(len(results), len(task_list)):
        job = task_list[index]
        out = Path(job["output_dir"])
        log = log_root / ("%02d_%s_seed_%d.log" % (index + 1, job["model"], job["training_seed"]))
        if out.exists():
            raise FileExistsError("Unrecorded task output exists; refusing automatic retraining: %s" % out)
        if log.exists():
            raise FileExistsError("Unrecorded task log exists; refusing automatic retraining: %s" % log)
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result = {"model": job["model"], "training_seed": job["training_seed"], "generation_seed": job["generation_seed"],
                  "status": "RUNNING", "started_at_unix": started, "command": job["command"], "log": str(log.resolve()),
                  "output_dir": str(out.resolve())}
        results.append(result)
        _write_json(history_path, results)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            with log.open("xb") as handle:
                completed = subprocess.run(job["command"], cwd=ROOT, stdout=handle,
                                           stderr=subprocess.STDOUT, env=environment)
        except BaseException as exc:
            result.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED_TO_LAUNCH",
                          elapsed_seconds=time.time() - started, error=repr(exc))
            _write_json(history_path, results)
            print("[%d/%d] %s seed %d: %s (%.1fs)" %
                  (index + 1, len(task_list), job["model"], job["training_seed"], result["status"],
                   result["elapsed_seconds"]), flush=True)
            raise
        result.update(return_code=completed.returncode, elapsed_seconds=time.time() - started,
                      log_sha256=sha256(log))
        if completed.returncode:
            result["status"] = "FAILED"
        else:
            try:
                result["artifact_hashes"] = _output_hashes(job, bundle)
                result["status"] = "COMPLETED"
            except Exception as exc:
                result.update(status="INVALID_OUTPUT", error=str(exc))
        _write_json(history_path, results)
        print("[%d/%d] %s seed %d: %s (%.1fs)" %
              (index + 1, len(task_list), job["model"], job["training_seed"], result["status"],
               result["elapsed_seconds"]), flush=True)
        if result["status"] != "COMPLETED":
            break
    status = "COMPLETED" if len(results) == len(task_list) and all(
        row["status"] == "COMPLETED" for row in results) else "INCOMPLETE"
    manifest["status"] = status
    manifest["results"] = results
    _write_json(manifest_path, manifest)
    return manifest


def _has_package(name):
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--dataset-bundle", type=Path, default=BUNDLE)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume after verifying every completed task; never rerun an interrupted task.")
    args = parser.parse_args()
    result = run(args.mode, args.run_label, args.output_root, args.dataset_bundle,
                 args.models, args.config_dir, args.resume)
    print(json.dumps(result, indent=2))
    if result["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
