"""Offline Raw-VS scoring and gain freeze. Never submits hardware tasks."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.models import scale_data_multi
from .kaiwu_13k_protocol import (load_protocol, resolve_path, sha256_file,
    scenario_metrics, select_vs_gain, validate_vs_response_records)
from .kaiwu_adapter import normalize_hardware_spins
from .reconstruct_hardware_scenarios import decode_platform_spins


def recompute_vs_predictions(records, preparation, protocol, require_physical=True,
                             require_complete=True):
    """Recreate scores from hash-bound inputs, without opening any TEST arrays.

    Noise seed = frozen base seed + export_index, shared across gains/backends.
    No calibration/ECC, target-based selection, refitting, or TEST evaluation.
    """
    from .run_kaiwu_13k_sampling import (
        _validate_preparation_identity, _vs_tasks, _task_expected, _preparation_identity)
    preparation = Path(preparation)
    if not protocol.get("_path") or load_protocol(protocol["_path"]) != {
            key: value for key, value in protocol.items() if key != "_path"}:
        raise ValueError("Prediction protocol must match the frozen on-disk JSON")
    _validate_preparation_identity(preparation, protocol)
    tasks = _vs_tasks(preparation, protocol)
    expected_map = {(float(t["gain"]), str(t["instance_id"])): _task_expected(t) for t in tasks}
    if require_complete:
        validate_vs_response_records(
            records, protocol["hardware"]["candidate_gains"], protocol["vs"]["instances"],
            expected_instance_ids=sorted({t["instance_id"] for t in tasks}),
            expected_task_map=expected_map, expected_identity=_preparation_identity(preparation, protocol),
            require_quality_policy=False, require_physical=require_physical)
    else:
        from .import_hardware_responses import strict_response_record_valid
        for record in records:
            expected = expected_map.get((float(record["gain"]), str(record["instance_id"])))
            if expected is None or not strict_response_record_valid(record, expected, require_physical):
                raise ValueError("Invalid single VS response")
    prep = json.loads((preparation / "preparation_manifest.json").read_text(encoding="utf-8"))
    model_path = resolve_path(protocol["model"]["checkpoint"])
    dataset_path = resolve_path(protocol["dataset"]["source"])
    if sha256_file(model_path) != prep["model_checkpoint_sha256"]:
        raise ValueError("Prediction checkpoint hash mismatch")
    if sha256_file(dataset_path) != prep["dataset_sha256"]:
        raise ValueError("Prediction dataset hash mismatch")
    # npz is lazy: only these four LS/VS arrays are accessed.
    with np.load(dataset_path, allow_pickle=False) as data:
        x_ls, y_ls, x_vs, y_vs = [np.asarray(data[key]) for key in ("x_ls", "y_ls", "x_vs", "y_vs")]
    _, _, x_scaled, _, _, _, scaler = scale_data_multi(x_ls, y_ls, x_vs, y_vs)
    conditions = np.concatenate((x_scaled, scaler.transform(x_vs[:, :24])), axis=1).astype(np.float32)
    by_key = {}
    panels = []
    for path in prep["vs_manifests"]:
        path = resolve_path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        gain = float(manifest[0]["hardware_gain"])
        if sha256_file(path) != prep["manifest_hashes"]["vs_manifests"][str(int(gain))]:
            raise ValueError("Prediction VS manifest hash mismatch")
        panel = []
        for item in manifest:
            key = (gain, str(item["id"]))
            if item["split"] != "VS" or key in by_key or key not in expected_map:
                raise ValueError("Prediction manifest must contain unique frozen VS conditions")
            index, offset = int(item["index"]), int(item["export_index"])
            if not 0 <= index < len(y_vs) or not 0 <= offset < int(protocol["vs"]["instances"]):
                raise ValueError("Prediction VS index/seed offset out of range")
            if sha256_file(resolve_path(item["path"])) != expected_map[key]["source_problem_sha256"]:
                raise ValueError("Prediction condition source hash mismatch")
            expected = expected_map[key]
            if sha256_file(resolve_path(expected["reference_sample_path"])) != expected["reference_sample_sha256"]:
                raise ValueError("Prediction reference sample hash mismatch")
            by_key[key] = (index, offset)
            panel.append((str(item["id"]), index, offset))
        panels.append(sorted(panel))
    if set(by_key) != set(expected_map) or any(panel != panels[0] for panel in panels):
        raise ValueError("Prediction gains must share the complete frozen VS panel")
    if len({index for _, index, _ in panels[0]}) != len(panels[0]) or len({offset for _, _, offset in panels[0]}) != len(panels[0]):
        raise ValueError("Prediction panel repeats a VS date or noise seed offset")
    settings = protocol["gain_selection"]["prediction_evaluation"]
    if settings != {"noise_seed": 2026, "noise_seed_rule": "base_plus_export_index",
                    "observation_noise": True, "postprocessing": "raw_clip_0_1"}:
        raise ValueError("Unsupported frozen VS prediction settings")
    # Trusted local checkpoint, checked against the preparation hash above.
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    model.device = torch.device("cpu")
    model.to("cpu")
    model.eval()
    output = []
    for record in sorted(records, key=lambda r: (float(r["gain"]), str(r["instance_id"]))):
        index, offset = by_key[(float(record["gain"]), str(record["instance_id"]))]
        with np.load(resolve_path(record["source_problem_path"]), allow_pickle=False) as problem:
            condition = np.asarray(problem["x_cond"], dtype=np.float32)
        if condition.shape != conditions[index].shape or not np.allclose(condition, conditions[index], rtol=0, atol=1e-6):
            raise ValueError("Frozen condition does not match the LS-scaled VS row")
        with np.load(record["response_path"], allow_pickle=False) as response:
            spins = normalize_hardware_spins(response["samples"], logical_n_bits=48)
        seed = settings["noise_seed"] + offset
        samples = decode_platform_spins(model, spins, condition, scaler,
                                       observation_noise=True, seed=seed)
        if samples.shape != (100, 24) or not np.isfinite(samples).all() or not np.isfinite(y_vs[index]).all():
            raise ValueError("Non-finite or invalid VS predictions/targets")
        metrics = scenario_metrics(samples, y_vs[index])
        metrics["mean_bias"] = float(np.mean(samples - y_vs[index]))
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError("Non-finite VS prediction metrics")
        output.append({"instance_id": record["instance_id"], "gain": float(record["gain"]),
                       "vs_index": index, "noise_seed": seed, "metrics": metrics})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--response-audit", type=Path, required=True,
                        help="Complete real vs_batch_audit.json (not mock responses)")
    args = parser.parse_args()
    from .run_kaiwu_13k_sampling import _vs_tasks, _task_expected, _preparation_identity
    protocol = load_protocol(args.config)
    protocol["_path"] = str(args.config.resolve())
    tasks = _vs_tasks(args.preparation, protocol)
    records = json.loads(args.response_audit.read_text(encoding="utf-8"))["records"]
    payload = select_vs_gain(
        records, args.preparation / "real_gain_selection.json",
        required_gains=protocol["hardware"]["candidate_gains"],
        expected_instances_per_gain=protocol["vs"]["instances"],
        expected_instance_ids=sorted({t["instance_id"] for t in tasks}),
        expected_task_map={(float(t["gain"]), str(t["instance_id"])): _task_expected(t) for t in tasks},
        expected_identity=_preparation_identity(args.preparation, protocol),
        quality_policy=protocol["gain_selection"]["quality_gate"], recompute_from_raw=True,
        prediction_inputs={"preparation": args.preparation, "protocol": protocol})
    print(json.dumps({"selected_gain": payload["selected_gain"], "submission_executed": False}))


if __name__ == "__main__":
    main()
