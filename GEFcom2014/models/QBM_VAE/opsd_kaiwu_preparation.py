# -*- coding: utf-8 -*-
"""Prepare and locally validate the frozen fixed-window OPSD SPQC-550 plan.

This command never trains a model and never calls the Kaiwu SDK.  It exports
post-training Ising instances, builds offline packages, runs local SA controls,
and writes an auditable zero-credit preparation directory.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE.export_ising_instances import (
    append_anchor_conditions,
)
from GEFcom2014.models.QBM_VAE.import_hardware_responses import import_responses
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import (
    audit_budget,
    load_protocol,
    scenario_metrics,
)
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import matrix_sha256
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import hardware_matrix_from_ising
from GEFcom2014.models.QBM_VAE.ising import ising_energy
from GEFcom2014.models.QBM_VAE.prepare_bosonic_submission import prepare_submission
from GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios import decode_platform_spins


ROOT_DIR = Path(__file__).resolve().parents[3]
# The current fixed-window OPSD hardware case study is the directly recorded
# SA-negative seed-0 checkpoint.  Keep the legacy internal-Gibbs protocol as a
# separate, explicit input rather than making it the accidental CLI default.
DEFAULT_PROTOCOL = ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json"
DEFAULT_OUTPUT = ROOT_DIR / "export" / "opsd_sa_negative_kaiwu_preparation_v8"
MODEL_NAME = "opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def _git(command):
    try:
        return subprocess.check_output(["git"] + command, cwd=ROOT_DIR, text=True,
                                       stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return "ERROR: %s" % exc.output.strip()


def environment_snapshot():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        import kaiwu
        kaiwu_module_version = getattr(kaiwu, "__version__", None)
    except ImportError:
        kaiwu_module_version = None
    kaiwu_distribution_version = version("kaiwu")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "kaiwu_distribution_version": kaiwu_distribution_version,
        "kaiwu_module_version": kaiwu_module_version,
        "kaiwu_version_mismatch": (
            kaiwu_distribution_version is not None
            and kaiwu_module_version is not None
            and kaiwu_distribution_version != kaiwu_module_version
        ),
    }


def _model_paths(protocol):
    model = protocol["model"]
    return {
        "checkpoint": _repo_path(model["checkpoint"]),
        "config": _repo_path(model["model_config"]),
        "dataset": _repo_path(protocol["dataset"]["source"]),
    }


def select_vs_rows(x_vs, selection_seed=2026, n_instances=10):
    """Select context-only VS rows from ten deterministic forecast strata."""
    x_vs = np.asarray(x_vs)
    if x_vs.ndim != 2 or n_instances != 10 or x_vs.shape[0] < n_instances:
        raise ValueError("OPSD VS context must contain at least ten rows")
    score = np.mean(x_vs[:, :24], axis=1)
    strata = np.array_split(np.argsort(score, kind="mergesort"), n_instances)
    rng = np.random.default_rng(selection_seed)
    rows = np.asarray([int(rng.choice(group)) for group in strata], dtype=np.int64)
    if len(np.unique(rows)) != n_instances:
        raise ValueError("Context-stratified VS selection produced duplicate rows")
    return rows


def historical_checkpoint_selection_evidence():
    """State only what legacy checkpoint metadata can actually establish."""
    return {
        "value": "historical VS reconstruction selection rule",
        "status": "可由同期源代码核实",
        # The checkpoint JSON does not preserve an immutable record of this
        # historical run's selection inputs. The contemporaneous source
        # documents the intended VS rule, but cannot establish the absence of
        # TEST use for the already-produced checkpoint.
        "test_used": "NOT_RECORDED",
        "best_epoch": "NOT_RECORDED",
        "basis": "fit_qbm_vae source defines a VS reconstruction checkpoint rule; old checkpoint JSON stores neither selected epoch nor a direct TEST-selection field",
    }


def audit_frozen_model(protocol, output_dir):
    paths = _model_paths(protocol)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError("Missing frozen OPSD %s: %s" % (label, path))
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    required = {
        "latent_s": 48,
        "cond_in": 57,
        "in_size": 24,
        "decoder_anchor": True,
        "sampler": "sa",
        "prior_type": "classical_ising",
        "transverse_field": 0.0,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError("Frozen OPSD config %s=%r, expected %r" % (key, config.get(key), expected))
    arrays, dataset_metadata, dates = load_daily_bundle(paths["dataset"])
    if tuple(arrays[key].shape[0] for key in ("x_ls", "x_vs", "x_test")) != (1036, 90, 90):
        raise ValueError("Frozen OPSD bundle does not have 1036/90/90 split")
    scaled = scale_data_multi(
        arrays["x_ls"], arrays["y_ls"], arrays["x_vs"], arrays["y_vs"],
        arrays["x_test"], arrays["y_test"],
    )
    x_ls_scaled, _, x_vs_scaled, _, _, _, y_scaler = scaled
    # Record the actual LS-fitted x scaler parameters, rather than a hash of
    # the transformed sample mean (which is close to zero by construction).
    x_mean = np.asarray(arrays["x_ls"], dtype=np.float64).mean(axis=0)
    x_scale = np.asarray(arrays["x_ls"], dtype=np.float64).std(axis=0, ddof=0)
    context = np.asarray(arrays["x_vs"][0, :24], dtype=np.float64)
    expected_anchor = y_scaler.transform(context[None, :])[0]
    condition = np.concatenate((x_vs_scaled[0], expected_anchor)).astype(np.float32)
    if condition.size != 57:
        raise ValueError("Historical context-anchor condition is not 57-dimensional")
    code_files = [
        ROOT_DIR / "GEFcom2014/models/QBM_VAE/qbm_vae.py",
        ROOT_DIR / "GEFcom2014/models/QBM_VAE/utils_qbm_vae.py",
        ROOT_DIR / "GEFcom2014/models/QBM_VAE/export_ising_instances.py",
        ROOT_DIR / "GEFcom2014/models/QBM_VAE/kaiwu_adapter.py",
        ROOT_DIR / "GEFcom2014/models/QBM_VAE/opsd_kaiwu_preparation.py",
        ROOT_DIR / "GEFcom2014/models/utils_models.py",
    ]
    code_hashes = {str(path.relative_to(ROOT_DIR)): sha256_file(path)
                   for path in code_files if path.is_file()}
    model_config_hash = sha256_file(paths["config"])
    evidence = {
        "negative_phase_backend": {
            "value": "internal",
            "status": "可由同期源代码和默认参数核实",
            "direct_json_field": config.get("negative_phase_backend", "NOT_RECORDED"),
            "basis": [
                "historical qbm_vae.py defaulted --negative-phase-backend to internal",
                "the frozen JSON records negative_steps=5 but not the backend field",
            ],
        },
        "anchor": {
            "value": "context",
            "status": "有直接证据证实",
            "basis": "decoder_anchor=true, cond_in=57, in_size=24 and historical source concatenates the first 24 raw context columns after LS scaling",
            "anchor_checkpoint": "NOT_APPLICABLE_HISTORICAL_CONTEXT_ANCHOR",
        },
        "scaler": {
            "value": "StandardScaler fit on LS x/y only",
            "status": "可由同期源代码核实",
            "fit_split": "LS",
            "x_mean_sha256": hashlib.sha256(np.asarray(x_mean, dtype="<f8").tobytes()).hexdigest(),
            "x_scale_sha256": hashlib.sha256(np.asarray(x_scale, dtype="<f8").tobytes()).hexdigest(),
            "y_mean_sha256": hashlib.sha256(np.asarray(y_scaler.mean_, dtype="<f8").tobytes()).hexdigest(),
            "y_scale_sha256": hashlib.sha256(np.asarray(y_scaler.scale_, dtype="<f8").tobytes()).hexdigest(),
        },
        "conditioning_order": {
            "value": "standardized 33-column x followed by 24-column standardized context anchor",
            "status": "可由同期源代码核实",
        },
        "decoder": {
            "value": {key: config.get(key) for key in ("decoder_covariance", "max_ar_coefficient", "in_size", "cond_in")},
            "status": "有直接证据证实",
        },
        "checkpoint_selection": historical_checkpoint_selection_evidence(),
    }
    # Old checkpoints may lack this field, but a directly recorded backend must
    # take precedence over any source/default inference used for legacy inputs.
    if config.get("negative_phase_backend") is not None:
        evidence["negative_phase_backend"] = {
            "value": config["negative_phase_backend"],
            "status": "direct_checkpoint_json",
            "direct_json_field": config["negative_phase_backend"],
            "basis": ["frozen checkpoint JSON explicitly records negative_phase_backend"],
        }
    payload = {
        "artifact_role": "offline_frozen_input_audit",
        "eligible_for_paper": False,
        "hardware_claim": False,
        "physical_platform_used": False,
        "training_invoked": False,
        "real_submission_executed": False,
        "protocol": str(Path(protocol.get("_path", DEFAULT_PROTOCOL)).resolve()),
        "protocol_sha256": sha256_file(Path(protocol.get("_path", DEFAULT_PROTOCOL))),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "git_status": _git(["status", "--short"]),
        "environment": environment_snapshot(),
        "files": {key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                  for key, path in paths.items()},
        "model_config": config,
        "dataset_metadata": dataset_metadata,
        "date_ranges": {split: [str(values[0]), str(values[-1])] for split, values in dates.items()},
        "provenance_code_sha256": code_hashes,
        "evidence": evidence,
        "historical_scope": "old fixed-window OPSD result; independent from rolling-origin and GEFCom experiments",
    }
    _json_dump(Path(output_dir) / "frozen_model_audit.json", payload)
    return payload, arrays, y_scaler


def _run_export(protocol, output_dir, split, hardware_gain=None, indices_file=None):
    model = protocol["model"]
    command = [
        sys.executable, "-m", "GEFcom2014.models.QBM_VAE.export_ising_instances",
        "--tag", "opsd-wind", "--dataset-bundle", str(_repo_path(protocol["dataset"]["source"])),
        "--model-name", model["model_name"], "--split", split,
        "--num-instances", str(protocol["vs"]["instances"] if split == "VS" else protocol["test"]["instances"]),
        "--selection", "stratified" if split == "VS" else "all",
        "--selection-seed", str(protocol["dataset"]["selection_seed"]),
        "--protocol-version", protocol["protocol_version"],
        "--protocol-sha256", sha256_file(protocol["_path"]),
        "--output-dir", str(output_dir), "--cpu",
    ]
    if hardware_gain is not None:
        command.extend(["--hardware-gain", str(hardware_gain)])
    if indices_file is not None:
        command.extend(["--indices-file", str(indices_file)])
    subprocess.run(command, cwd=ROOT_DIR, check=True)
    manifests = sorted(Path(output_dir).glob("*_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("Expected one exporter manifest in %s" % output_dir)
    return manifests[0], command


def _build_quality_reference_manifest(output_dir, protocol, vs_manifests):
    entries = []
    for gain, manifest_path in zip(protocol["hardware"]["candidate_gains"], vs_manifests):
        package = output_dir / ("submission_gain_%d" % gain) / "package"
        package_manifest = json.loads((package / "submission_manifest.json").read_text(encoding="utf-8"))
        package_items = {row["instance_id"]: row for row in package_manifest["instances"]}
        for item in json.loads(Path(manifest_path).read_text(encoding="utf-8")):
            problem_path = _repo_path(item["path"])
            package_item = package_items[str(item["id"])]
            matrix_path = package / package_item["hardware_matrix"]
            sample_path = output_dir / "local_controls" / ("gain_%d" % gain) / "quantized" / "sa" / (str(item["id"]) + ".npz")
            with np.load(problem_path, allow_pickle=False) as problem:
                h = np.asarray(problem["h"], dtype=np.float64)
                j = np.asarray(problem["J"], dtype=np.float64)
            with np.load(matrix_path, allow_pickle=False) as matrix_archive:
                matrix = np.asarray(matrix_archive["hardware_matrix"], dtype=np.int8)
            with np.load(sample_path, allow_pickle=False) as sample:
                required = ("samples", "hardware_samples", "backend", "matrix_space",
                            "instance_id", "random_seed", "requested_reads", "sweeps",
                            "sampler_beta", "matrix_beta", "logical_beta", "hardware_gain",
                            "protocol_sha256", "hardware_matrix_sha256", "source_problem_sha256")
                missing = [key for key in required if key not in sample]
                if missing:
                    raise ValueError("SA reference lacks provenance fields: %s" % ", ".join(missing))
                logical = np.asarray(sample["samples"], dtype=np.int8)
                raw = np.asarray(sample["hardware_samples"], dtype=np.int8)
                metadata = {key: np.asarray(sample[key]).reshape(-1)[0].item()
                            for key in required if key not in ("samples", "hardware_samples")}
            expected_beta = float(protocol["gain_selection"]["prediction_evaluation"].get("logical_beta", 1.0)) / float(gain)
            if (h.shape != (48,) or j.shape != (48, 48) or matrix.shape != (49, 49)
                    or logical.shape != (100, 48) or raw.shape != (100, 49)
                    or not np.isin(logical, (-1, 1)).all() or not np.isin(raw, (-1, 1)).all()
                    or metadata["backend"] != "sa" or metadata["matrix_space"] != "hardware-quantized"
                    or str(metadata["instance_id"]) != str(item["id"])
                    or int(metadata["requested_reads"]) != 100
                    or int(metadata["sweeps"]) != 100
                    or int(metadata["hardware_gain"]) != int(gain)
                    or not np.isclose(float(metadata["sampler_beta"]), expected_beta, rtol=0, atol=1e-12)
                    or not np.isclose(float(metadata["matrix_beta"]), expected_beta, rtol=0, atol=1e-12)
                    or str(metadata["hardware_matrix_sha256"]) != matrix_sha256(matrix)
                    or str(metadata["source_problem_sha256"]) != sha256_file(problem_path)
                    or str(metadata["protocol_sha256"]) != sha256_file(protocol["_path"])
                    or not np.isclose(float(metadata["logical_beta"]),
                                      float(protocol["gain_selection"]["prediction_evaluation"].get("logical_beta", 1.0)),
                                      rtol=0, atol=1e-12)):
                raise ValueError("SA reference provenance/configuration mismatch for %s" % item["id"])
            entries.append({
                "gain": int(gain), "instance_id": str(item["id"]),
                "split": item["split"], "date": item["date"],
                "protocol_sha256": sha256_file(protocol["_path"]),
                "source_problem_path": str(problem_path.resolve()),
                "source_problem_sha256": sha256_file(problem_path),
                "hardware_matrix_path": str(matrix_path.resolve()),
                "hardware_matrix_file_sha256": sha256_file(matrix_path),
                "hardware_matrix_sha256": matrix_sha256(matrix),
                "reference_sample_path": str(sample_path.resolve()),
                "reference_sample_sha256": sha256_file(sample_path),
                "sampler": {"backend": "sa", "matrix_space": "hardware-quantized",
                            "reads": 100, "sweeps": 100,
                            "seed": int(metadata["random_seed"]),
                            "sampler_beta": float(metadata["sampler_beta"]),
                            "matrix_beta": float(metadata["matrix_beta"]),
                            "logical_beta": float(metadata["logical_beta"])},
                "hardware_claim": False, "physical_platform_used": False,
            })
    expected = len(protocol["hardware"]["candidate_gains"]) * int(protocol["vs"]["instances"])
    keys = [(row["gain"], row["instance_id"]) for row in entries]
    if len(entries) != expected or len(set(keys)) != expected:
        raise ValueError("Quality reference must cover every unique gain-condition pair")
    target = Path(output_dir) / "quality_reference_manifest.json"
    _json_dump(target, {
        "artifact_role": "offline_classical_sa_quality_reference",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol["_path"]),
        "sampling_configuration": {"backend": "sa", "matrix_space": "hardware-quantized",
                                   "reads": 100, "sweeps": 100, "logical_beta": 1.0},
        "instances": entries,
        "hardware_claim": False, "physical_platform_used": False,
        "eligible_for_paper": False,
    })
    return target


def _make_mock_responses(package_manifest, quantized_response_dir, output_dir):
    package_manifest = Path(package_manifest)
    package = json.loads(package_manifest.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in package["instances"]:
        source = Path(quantized_response_dir) / (item["instance_id"] + ".npz")
        with np.load(source, allow_pickle=False) as archive:
            raw = np.asarray(archive["hardware_samples"], dtype=np.int8)
        payload = json.loads((package_manifest.parent / item["payload"]).read_text(encoding="utf-8"))
        np.savez_compressed(
            output_dir / (item["instance_id"] + ".npz"),
            samples=raw,
            instance_id=item["instance_id"],
            task_id="offline_mock_%s" % item["instance_id"],
            task_name=payload["task_name"],
            matrix_sha256=item["hardware_matrix_sha256"],
            hardware_matrix_sha256=item["hardware_matrix_sha256"],
            bit_order="index_ascending",
            backend="offline_sa_mock",
            protocol_sha256=payload["protocol_sha256"],
            requested_reads=100,
            returned_reads=int(raw.shape[0]),
            hardware_claim=False,
            physical_platform_used=False,
        )


def audit_exported_matrices(manifests, output_path, protocol_sha256=None):
    records = []
    for manifest_path in manifests:
        for item in json.loads(Path(manifest_path).read_text(encoding="utf-8")):
            with np.load(_repo_path(item["path"]), allow_pickle=False) as problem:
                h = np.asarray(problem["h"], dtype=np.float64)
                j = np.asarray(problem["J"], dtype=np.float64)
                q = np.asarray(problem["hardware_matrix"], dtype=np.int8)
                source_hash = str(np.asarray(problem.get(
                    "protocol_sha256", np.asarray(""))).reshape(-1)[0])
            if protocol_sha256 is not None and source_hash != protocol_sha256:
                raise ValueError("Exported Ising problem protocol hash mismatch for %s" % item["id"])
            rng = np.random.default_rng(2026 + int(item["index"]))
            spins = rng.choice(np.asarray([-1.0, 1.0]), size=(8, 48))
            augmented = np.concatenate((spins, np.ones((8, 1))), axis=1)
            source = hardware_matrix_from_ising(h, j)
            energy_error = float(np.max(np.abs(
                np.einsum("bi,ij,bj->b", augmented, source, augmented)
                - ising_energy(spins, h, j)
            )))
            if energy_error > 1e-10:
                raise ValueError("Auxiliary-spin energy mapping failed for %s" % item["id"])
            if q.shape != (49, 49) or not np.array_equal(q, q.T) or np.any(np.diag(q)):
                raise ValueError("Invalid 49-spin matrix for %s" % item["id"])
            audit = item.get("quantization_audit") or {}
            records.append({
                "instance_id": item["id"],
                "gain": item.get("hardware_gain"),
                "logical_n_bits": int(h.size),
                "hardware_n_bits": int(q.shape[0]),
                "source_nonzero_coefficients": audit.get("source_nonzero_coefficients"),
                "nonzero_edges": audit.get("nonzero_edges"),
                "max_abs_integer": audit.get("max_abs_integer"),
                "relative_quantization_error": audit.get("relative_quantization_error"),
                "zeroed_coefficient_fraction": audit.get("nonzero_quantized_to_zero_fraction"),
                "clipping": audit.get("clipping"),
                "matrix_sha256": matrix_sha256(q),
                "energy_mapping_max_abs_error": energy_error,
            })
    _json_dump(output_path, {
        "protocol_sha256": protocol_sha256,
        "hardware_claim": False,
        "physical_platform_used": False,
        "records": records,
    })
    return records


def _decode_metrics(manifest_path, canonical_dir, float_dir, arrays, y_scaler,
                    model_path, model_config_path, seed=2026):
    import pickle
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    model.device = torch.device("cpu")
    model.to("cpu")
    model.eval()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = []
    for offset, item in enumerate(manifest):
        with np.load(_repo_path(item["path"]), allow_pickle=False) as problem:
            condition = np.asarray(problem["x_cond"], dtype=np.float32)
        target = np.asarray(arrays["y_vs"][int(item["index"])], dtype=np.float64)
        with np.load(Path(canonical_dir) / (item["id"] + ".npz"), allow_pickle=False) as imported:
            quantized_spins = np.asarray(imported["samples"], dtype=np.int8)
        with np.load(Path(float_dir) / (item["id"] + ".npz"), allow_pickle=False) as floating:
            float_spins = np.asarray(floating["samples"], dtype=np.int8)
        quantized_scenarios = decode_platform_spins(
            model, quantized_spins, condition, y_scaler,
            observation_noise=True, seed=seed + offset,
        )
        float_scenarios = decode_platform_spins(
            model, float_spins, condition, y_scaler,
            observation_noise=True, seed=seed + offset,
        )
        rows.append({
            "instance_id": item["id"], "date": str(item.get("date")),
            "float_logical_sa": scenario_metrics(float_scenarios, target),
            "quantized_sa_from_mock_response": scenario_metrics(quantized_scenarios, target),
            "sample_audit": {
                "requested_reads": 100,
                "float_logical_shape": list(float_spins.shape),
                "quantized_logical_shape": list(quantized_spins.shape),
                "float_spins_valid": bool(np.isin(float_spins, (-1, 1)).all()),
                "quantized_spins_valid": bool(np.isin(quantized_spins, (-1, 1)).all()),
                "float_unique_state_fraction": float(np.unique(float_spins, axis=0).shape[0] / float_spins.shape[0]),
                "quantized_unique_state_fraction": float(np.unique(quantized_spins, axis=0).shape[0] / quantized_spins.shape[0]),
                "decoded_scenario_shape": list(quantized_scenarios.shape),
            },
            "hardware_claim": False, "physical_platform_used": False,
        })
    return rows


def prepare(protocol_path=DEFAULT_PROTOCOL, output_dir=DEFAULT_OUTPUT):
    protocol_path = Path(protocol_path).resolve()
    output_dir = Path(output_dir).resolve()
    protocol = load_protocol(protocol_path)
    protocol["_path"] = str(protocol_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Preparation output must be a new empty directory: %s" % output_dir)
    freeze, arrays, y_scaler = audit_frozen_model(protocol, output_dir)
    _, _, bundle_dates = load_daily_bundle(_model_paths(protocol)["dataset"])
    rows = select_vs_rows(arrays["x_vs"], protocol["dataset"]["selection_seed"], protocol["vs"]["instances"])
    indices_file = output_dir / "vs_selection_indices.json"
    _json_dump(indices_file, rows.tolist())
    _json_dump(output_dir / "vs_selection_manifest.json", {
        "selection_seed": protocol["dataset"]["selection_seed"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "selection_rule": "mean of first 24 forecast-context columns, ten deterministic strata, no y access",
        "indices": rows.tolist(), "dates": [str(bundle_dates["vs"][i]) for i in rows],
        "test_data_used": False, "hardware_claim": False, "physical_platform_used": False,
    })

    vs_manifests = []
    commands = []
    for gain in protocol["hardware"]["candidate_gains"]:
        export_dir = output_dir / ("export_vs_gain_%d" % gain)
        manifest, command = _run_export(protocol, export_dir, "VS", gain, indices_file)
        vs_manifests.append(manifest)
        commands.append(command)
        prepare_submission(manifest, output_dir / ("submission_gain_%d" % gain),
                           "vs-calibration", protocol["hardware"]["reads_per_instance"])

    matrix_audit = audit_exported_matrices(
        vs_manifests, output_dir / "matrix_audit.json", sha256_file(protocol_path))

    test_export_dir = output_dir / "test_logical"
    test_manifest, command = _run_export(protocol, test_export_dir, "TEST")
    commands.append(command)
    test_manifest_out = output_dir / "test_logical" / "opsd-wind_test_manifest.json"
    if test_manifest.resolve() != test_manifest_out.resolve():
        test_manifest_out = test_manifest
    budget = audit_budget(vs_manifests, test_manifest_out, protocol)
    budget["protocol_version"] = protocol["protocol_version"]
    budget["protocol_sha256"] = sha256_file(protocol_path)
    _json_dump(output_dir / "budget_audit.json", budget)

    manifest_hashes = {
        "vs_selection_manifest": sha256_file(output_dir / "vs_selection_manifest.json"),
        "test_logical_manifest": sha256_file(test_manifest_out),
        "vs_manifests": {
            str(gain): sha256_file(manifest)
            for gain, manifest in zip(protocol["hardware"]["candidate_gains"], vs_manifests)
        },
    }

    local_rows = []
    for gain, manifest in zip(protocol["hardware"]["candidate_gains"], vs_manifests):
        base = output_dir / "local_controls" / ("gain_%d" % gain)
        float_dir = base / "float"
        quantized_root = base / "quantized"
        for matrix_space, target_dir in (("logical", float_dir), ("hardware-quantized", quantized_root)):
            subprocess.run([
                sys.executable, "-m", "GEFcom2014.models.QBM_VAE.sample_exported_ising",
                "--manifest", str(manifest), "--backend", "sa", "--num-reads", "100",
                "--beta", "1.0", "--sweeps", "100", "--seed", "2026",
                "--matrix-space", matrix_space, "--output-dir", str(target_dir),
                "--protocol-sha256", sha256_file(protocol_path),
            ], cwd=ROOT_DIR, check=True)
        quantized_response_dir = quantized_root / "sa"
        package_manifest = output_dir / ("submission_gain_%d" % gain) / "package" / "submission_manifest.json"
        mock_dir = output_dir / "mock_responses" / ("gain_%d" % gain)
        _make_mock_responses(package_manifest, quantized_response_dir, mock_dir)
        canonical_dir = output_dir / "imported_mock" / ("gain_%d" % gain)
        import_responses(package_manifest, mock_dir, canonical_dir)
        local_rows.extend(_decode_metrics(
            manifest, canonical_dir, float_dir / "sa", arrays, y_scaler,
            _repo_path(protocol["model"]["checkpoint"]), _repo_path(protocol["model"]["model_config"]),
        ))

    quality_reference_manifest = _build_quality_reference_manifest(output_dir, protocol, vs_manifests)

    prep_manifest = {
        "artifact_role": "offline_hardware_preparation",
        "protocol_version": protocol["protocol_version"],
        "eligible_for_paper": False,
        "hardware_claim": False,
        "physical_platform_used": False,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "commands": commands,
        "vs_manifests": [str(path.resolve()) for path in vs_manifests],
        "test_logical_manifest": str(Path(test_manifest_out).resolve()),
        "candidate_gains": protocol["hardware"]["candidate_gains"],
        "budget": budget,
        "model_checkpoint_sha256": freeze["files"]["checkpoint"]["sha256"],
        "dataset_sha256": freeze["files"]["dataset"]["sha256"],
        "matrix_audit_sha256": sha256_file(output_dir / "matrix_audit.json"),
        "matrix_audit_records": len(matrix_audit),
        "manifest_hashes": manifest_hashes,
        "quality_reference_manifest_sha256": sha256_file(quality_reference_manifest),
        "test_data_used_for_gain_selection": False,
        "real_platform_submitted": False,
        "local_validation": {
            "backend": "SA/mock response",
            "rows": local_rows,
            "raw_metrics_only": True,
            "observation_noise": True,
            "seed": 2026,
        },
        "hashes": {
            "frozen_model_audit": sha256_file(output_dir / "frozen_model_audit.json"),
            "budget_audit": sha256_file(output_dir / "budget_audit.json"),
        },
    }
    _json_dump(output_dir / "preparation_manifest.json", prep_manifest)
    report = output_dir / "READINESS_REPORT.md"
    report.write_text(
        "# OPSD SPQC-550 Offline Preparation\n\n"
        "Status: `BLOCKED` for real submission; zero-credit local preparation completed.\n\n"
        "This directory uses the frozen fixed-window OPSD FA-BM-VAE checkpoint. "
        "Training was not invoked and no Kaiwu/SPQC task was submitted. All local SA/mock "
        "artifacts have `hardware_claim=false` and `physical_platform_used=false`.\n\n"
        "## Budget\n\n"
        "- VS: 10 conditions x 3 gains x 100 reads = 3,000 requested reads.\n"
        "- TEST: 90 frozen dates x 100 reads = 9,000 requested reads.\n"
        "- Planned total: 12,000 requested reads; account 15,000 Sampling Credits conversion is pending the first real task.\n"
        "- TEST hardware matrices remain blocked until a unique gain is selected from verified real VS responses.\n\n"
        "## Frozen model\n\n"
        "The historical checkpoint uses a 48-spin classical Ising prior and a context anchor: "
        "33 standardized context features followed by 24 standardized forecast columns. "
        "The frozen JSON records the training negative-phase backend in the audit. "
        "It is not inferred from the post-training sampler name.\n\n"
        "## Missing external conditions\n\n"
        "Sampling Credits, authorized SPQC-550 routing, Kaiwu SDK/license, and account/project "
        "environment are not verified. This report is not a physical-platform result and makes "
        "no quantum-advantage claim.\n\n"
        "The local Kaiwu distribution/module versions and their mismatch status are recorded in "
        "`frozen_model_audit.json`. A reported mismatch must be resolved with the platform "
        "administrator before the first paid task.\n\n"
        "## First probe command (not executed)\n\n"
        "```powershell\n"
        "$env:FA_BM_VAE_RUN_REAL_KAIWU=\"1\"\n"
        "D:\\anaconda\\envs\\wsy\\python.exe -m GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling `\n"
        "  --stage vs --protocol %s `\n"
        "  --preparation %s `\n"
        "  --output-dir %s\\real_responses `\n"
        "  --checkpoint-dir %s\\checkpoints `\n"
        "  --limit 1 --resume --execute --confirm-real-submission\n"
        "```\n" % (
            str(protocol_path.relative_to(ROOT_DIR)).replace("/", "\\\\"),
            str(output_dir.relative_to(ROOT_DIR)).replace("/", "\\\\"),
            str(output_dir.relative_to(ROOT_DIR)).replace("/", "\\\\"),
            str(output_dir.relative_to(ROOT_DIR)).replace("/", "\\\\"),
        ),
        encoding="utf-8",
    )
    return prep_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    result = prepare(args.protocol, args.output_dir)
    result["elapsed_seconds"] = time.perf_counter() - started
    _json_dump(Path(args.output_dir) / "preparation_manifest.json", result)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()),
                      "vs_tasks": result["budget"]["vs_tasks"],
                      "test_tasks": result["budget"]["test_tasks"],
                      "planned_samples": result["budget"]["planned_samples"],
                      "hardware_claim": False, "physical_platform_used": False,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
