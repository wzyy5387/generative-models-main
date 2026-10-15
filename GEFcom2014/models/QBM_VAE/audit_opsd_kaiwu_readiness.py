# -*- coding: utf-8 -*-
"""Read-only zero-credit audit for a frozen OPSD Kaiwu preparation package."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE.ising import ising_energy
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import scenario_metrics
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import normalize_hardware_spins
from GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios import decode_platform_spins


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREPARATION = ROOT_DIR / "export" / "opsd_sa_negative_kaiwu_preparation_v4"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar(archive, key):
    return np.asarray(archive[key]).reshape(-1)[0].item()


def inspect_sdk():
    import kaiwu
    import kaiwu.cim as cim
    import kaiwu.common as common

    solve_parameters = inspect.signature(cim.CIMOptimizer.solve).parameters
    return {
        "distribution_version": importlib.metadata.version("kaiwu"),
        "module_version": getattr(kaiwu, "__version__", None),
        "module_path": str(Path(kaiwu.__file__).resolve()),
        "cim_module_path": str(Path(cim.__file__).resolve()),
        "optimizer_init_signature": str(inspect.signature(cim.CIMOptimizer)),
        "optimizer_solve_signature": str(inspect.signature(cim.CIMOptimizer.solve)),
        "sampling_mode": str(cim.TaskMode.SAMPLING),
        "checkpoint_manager_present": hasattr(common, "CheckpointManager"),
        "api_signature_compatible": {
            "negtail_flip": "negtail_flip" in solve_parameters,
            "sort_solutions": "sort_solutions" in solve_parameters,
        },
        "version_mismatch": importlib.metadata.version("kaiwu") != getattr(kaiwu, "__version__", None),
        "interpretation": (
            "Static API inspection supports the configured solve(matrix, negtail_flip=False, "
            "sort_solutions=False) call. Distribution/module version mismatch alone is not an "
            "observed API incompatibility; account routing, license and charged canary remain unverified."
        ),
    }


def _load_model_and_scaler(preparation):
    frozen = json.loads((preparation / "frozen_model_audit.json").read_text(encoding="utf-8"))
    checkpoint = Path(frozen["files"]["checkpoint"]["path"])
    dataset = Path(frozen["files"]["dataset"]["path"])
    arrays, _, _ = load_daily_bundle(dataset)
    scaled = scale_data_multi(
        arrays["x_ls"], arrays["y_ls"], arrays["x_vs"], arrays["y_vs"],
        arrays["x_test"], arrays["y_test"],
    )
    with checkpoint.open("rb") as handle:
        model = pickle.load(handle)
    model.device = torch.device("cpu")
    model.to("cpu").eval()
    return frozen, arrays, scaled[-1], model


def audit_local_controls(preparation):
    frozen, arrays, y_scaler, model = _load_model_and_scaler(preparation)
    matrix_audit = json.loads((preparation / "matrix_audit.json").read_text(encoding="utf-8"))["records"]
    outputs = []
    for gain in (125, 150, 200):
        manifests = list((preparation / ("export_vs_gain_%d" % gain)).glob("*_manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("Expected exactly one VS export manifest for gain %d" % gain)
        items = json.loads(manifests[0].read_text(encoding="utf-8"))
        deltas, sample_audits = [], []
        for offset, item in enumerate(items):
            instance_id = item["id"]
            with np.load(item["path"], allow_pickle=False) as problem:
                h, j = np.asarray(problem["h"]), np.asarray(problem["J"])
                condition = np.asarray(problem["x_cond"], dtype=np.float32)
            float_path = preparation / "local_controls" / ("gain_%d" % gain) / "float" / "sa" / (instance_id + ".npz")
            imported_path = preparation / "imported_mock" / ("gain_%d" % gain) / (instance_id + ".npz")
            with np.load(float_path, allow_pickle=False) as payload:
                float_spins = np.asarray(payload["samples"], dtype=np.int8)
            with np.load(imported_path, allow_pickle=False) as payload:
                logical_spins = np.asarray(payload["samples"], dtype=np.int8)
                hardware_spins = np.asarray(payload["hardware_samples"], dtype=np.int8)
                requested = int(_scalar(payload, "requested_reads"))
                returned = int(_scalar(payload, "returned_reads"))
                claim = bool(_scalar(payload, "hardware_claim"))
                physical = bool(_scalar(payload, "physical_platform_used"))
            normalized = normalize_hardware_spins(hardware_spins)
            if not np.array_equal(normalized, logical_spins):
                raise AssertionError("Auxiliary-spin normalization mismatch for %s" % instance_id)
            if not (requested == returned == 100 and float_spins.shape == logical_spins.shape == (100, 48)):
                raise AssertionError("Read count or logical sample shape mismatch for %s" % instance_id)
            if claim or physical:
                raise AssertionError("Offline mock must not be promoted to physical hardware")
            target = np.asarray(arrays["y_vs"][int(item["index"])], dtype=float)
            float_scenarios = decode_platform_spins(model, float_spins, condition, y_scaler,
                                                     observation_noise=True, seed=2026 + offset)
            quantized_scenarios = decode_platform_spins(model, logical_spins, condition, y_scaler,
                                                         observation_noise=True, seed=2026 + offset)
            float_metrics = scenario_metrics(float_scenarios, target)
            quantized_metrics = scenario_metrics(quantized_scenarios, target)
            deltas.append({key: float(quantized_metrics[key] - float_metrics[key]) for key in float_metrics})
            unique, counts = np.unique(logical_spins, axis=0, return_counts=True)
            sample_audits.append({
                "instance_id": instance_id, "requested_reads": requested, "returned_reads": returned,
                "hardware_shape": list(hardware_spins.shape), "logical_shape": list(logical_spins.shape),
                "values_valid": bool(np.isin(hardware_spins, (-1, 1)).all() and np.isin(logical_spins, (-1, 1)).all()),
                "frequency_preserved": int(counts.sum()) == returned,
                "unique_state_fraction": float(unique.shape[0] / returned),
                "offline_claims_false": not claim and not physical,
                "logical_energy_mean": float(np.mean(ising_energy(logical_spins, h, j))),
            })
        records = [row for row in matrix_audit if int(row["gain"]) == gain]
        outputs.append({
            "gain": gain, "n_instances": len(items),
            "max_relative_quantization_error": float(max(row["relative_quantization_error"] for row in records)),
            "max_zeroed_coefficient_fraction": float(max(row["zeroed_coefficient_fraction"] for row in records)),
            "max_abs_integer": int(max(row["max_abs_integer"] for row in records)),
            "any_clipping": bool(any(row["clipping"] for row in records)),
            "scenario_metric_delta_quantized_minus_float_sa": {
                key: float(np.mean([row[key] for row in deltas])) for key in deltas[0]
            },
            "sample_audit": sample_audits,
        })
    return frozen, outputs


def run(preparation=DEFAULT_PREPARATION):
    preparation = Path(preparation).resolve()
    manifest = json.loads((preparation / "preparation_manifest.json").read_text(encoding="utf-8"))
    frozen, controls = audit_local_controls(preparation)
    return {
        "artifact_role": "zero_credit_kaiwu_readiness_audit",
        "hardware_claim": False, "physical_platform_used": False,
        "preparation": str(preparation), "preparation_manifest_sha256": _sha256(preparation / "preparation_manifest.json"),
        "protocol_sha256": manifest["protocol_sha256"],
        "model_checkpoint_sha256": manifest["model_checkpoint_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "sdk": inspect_sdk(),
        "local_controls": controls,
        "budget": manifest["budget"],
        "gain_selection_protocol": {
            "split": "VS only; TEST data and TEST metrics are prohibited",
            "required_candidates": [125, 150, 200], "instances_per_gain": 10,
            "reads_per_instance": 100, "convergence_reads": [50, 100],
            "reference": "same-instance floating logical SA and 8-bit matrix SA with fixed seeds",
            "selection": "aggregate each metric across the ten verified VS instances; lexicographically minimize energy Wasserstein, effective-beta error, edge-moment MAE, magnetization MAE, then maximize unique-state fraction; exact tie selects lower gain",
            "freeze": "write immutable real_gain_selection.json only after all 30 verified physical VS responses; TEST remains locked otherwise",
        },
        "canary_acceptance": {
            "first_task": "one frozen VS condition at gain 125 with 100 reads, counted within the 3,000 VS budget",
            "must_pass": ["task ID and UTC timestamps", "SDK/backend/matrix/response hashes", "49 valid spins and 48-spin normalization", "requested=returned=100", "response import and decoder reconstruction"],
            "stop_conditions": ["SDK/license/routing failure", "missing metadata or hash mismatch", "invalid spin values or wrong shape", "unknown pending task status", "unexpected billing policy"],
        },
        "external_blockers": ["Sampling Credits", "licensed account and SPQC-550 routing", "vendor confirmation of credit/read and failed-task charging"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.preparation / "ZERO_CREDIT_ACCEPTANCE.json")
    payload = run(args.preparation)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
