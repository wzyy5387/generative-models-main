"""Local decoder sensitivity diagnostic using only frozen VS SA samples."""

import argparse
import json
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from GEFcom2014.models import scale_data_multi

from .kaiwu_13k_protocol import load_protocol, resolve_path, sha256_file
from .run_kaiwu_13k_sampling import _vs_tasks, _validate_preparation_identity


def _decode_parameters(model, binary_z, condition, device):
    z = torch.as_tensor(binary_z, dtype=torch.float32, device=device)
    context = torch.as_tensor(condition, dtype=torch.float32, device=device)
    context = context.unsqueeze(0).expand(z.shape[0], -1)
    with torch.inference_mode():
        mean, log_scale, _ = model.decode_temporal_parameters(z, context)
    mean = mean.detach().cpu().numpy()
    scale = np.exp(log_scale.detach().cpu().numpy())
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Non-finite decoder parameter in latent sensitivity diagnostic")
    return mean, scale


def summarize_flip_sensitivity(base_mean, changed_mean, base_scale, changed_scale,
                               scaler_scale, scale_mw):
    """Return per-sample/bit RMS diagnostics and distribution summaries.

    Axes are explicit: sample, latent-bit flip, forecast hour. Decoder outputs
    enter in standardized-target units; StandardScaler scale and dataset
    scale_mw are multiplicative factors. No scaler mean is applied to deltas
    or standard deviations.
    """
    base_mean = np.asarray(base_mean, dtype=np.float64)
    changed_mean = np.asarray(changed_mean, dtype=np.float64)
    base_scale = np.asarray(base_scale, dtype=np.float64)
    changed_scale = np.asarray(changed_scale, dtype=np.float64)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float64).reshape(-1)
    scale_mw = float(scale_mw)
    if (base_mean.ndim != 2 or base_scale.shape != base_mean.shape
            or changed_mean.ndim != 3 or changed_scale.shape != changed_mean.shape
            or changed_mean.shape[0] != base_mean.shape[0]
            or changed_mean.shape[2] != base_mean.shape[1]
            or scaler_scale.size != base_mean.shape[1]):
        raise ValueError("Expected base (sample,hour) and changed (sample,flip,hour) arrays")
    if (not np.isfinite([scale_mw]).all() or scale_mw <= 0
            or not np.isfinite(scaler_scale).all() or np.any(scaler_scale <= 0)
            or not np.isfinite(base_mean).all() or not np.isfinite(changed_mean).all()
            or not np.isfinite(base_scale).all() or not np.isfinite(changed_scale).all()
            or np.any(base_scale <= 0) or np.any(changed_scale <= 0)):
        raise ValueError("Sensitivity inputs and physical scales must be finite and positive")

    # sample x flip x hour; scaler mean is deliberately absent for deltas/sigma.
    mw_scale = scaler_scale.reshape(1, 1, -1) * scale_mw
    mean_delta_mw = (changed_mean - base_mean[:, None, :]) * mw_scale
    base_sigma_mw = base_scale * (scaler_scale * scale_mw)[None, :]
    changed_sigma_mw = changed_scale * mw_scale
    mean_shift_rms = np.sqrt(np.mean(mean_delta_mw ** 2, axis=2))
    observation_sigma_rms = np.sqrt(np.mean(base_sigma_mw ** 2, axis=1))
    mean_to_noise = mean_shift_rms / np.maximum(observation_sigma_rms[:, None], 1e-12)
    relative_scale_change_rms = np.sqrt(np.mean(
        ((changed_sigma_mw - base_sigma_mw[:, None, :])
         / np.maximum(base_sigma_mw[:, None, :], 1e-12)) ** 2,
        axis=2,
    ))

    def summary(values):
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        return {"median": float(np.median(flattened)),
                "p95": float(np.quantile(flattened, 0.95))}

    return {
        "mean_shift_rms_mw": mean_shift_rms,
        "observation_sigma_rms_mw_by_sample": observation_sigma_rms,
        "mean_shift_to_observation_sigma": mean_to_noise,
        "relative_observation_scale_change_rms": relative_scale_change_rms,
        "summary": {
            "mean_shift_rms_mw": summary(mean_shift_rms),
            "observation_sigma_rms_mw": summary(observation_sigma_rms),
            "mean_shift_to_observation_sigma": summary(mean_to_noise),
            "relative_observation_scale_change_rms": summary(relative_scale_change_rms),
        },
    }


def frozen_vs_condition(task, conditions, vs_start_date):
    """Load a frozen problem condition and verify it against its source VS row."""
    index = int(task["index"])
    export_index = int(task["export_index"])
    if not 0 <= index < len(conditions):
        raise ValueError("VS source index is outside the full VS condition array")
    if export_index < 0:
        raise ValueError("VS export_index must be non-negative")
    expected_date = date.fromisoformat(str(vs_start_date)) + timedelta(days=index)
    actual_date = datetime.fromisoformat(str(task["date"])).date()
    if actual_date != expected_date:
        raise ValueError("Frozen VS task date does not match its original VS row index")

    source_path = Path(task["source_problem_path"])
    expected_hash = str(task["source_problem_sha256"])
    if not source_path.is_file() or sha256_file(source_path) != expected_hash:
        raise ValueError("Frozen VS source problem is missing or its hash differs")
    with np.load(source_path, allow_pickle=False) as problem:
        if "x_cond" not in problem:
            raise ValueError("Frozen VS source problem lacks x_cond")
        condition = np.asarray(problem["x_cond"], dtype=np.float32)
    full_vs_condition = np.asarray(conditions[index], dtype=np.float32)
    if (condition.shape != full_vs_condition.shape
            or not np.allclose(condition, full_vs_condition, rtol=0, atol=1e-6)):
        raise ValueError("Frozen source x_cond does not match the indexed full VS condition")
    return condition, float(np.max(np.abs(condition - full_vs_condition)))


def run_diagnostic(protocol_path, preparation, max_samples_per_condition=20):
    protocol = load_protocol(protocol_path)
    protocol["_path"] = str(Path(protocol_path).resolve())
    _validate_preparation_identity(preparation, protocol)
    preparation = Path(preparation).resolve()
    prep = json.loads((preparation / "preparation_manifest.json").read_text(encoding="utf-8"))
    model_path = resolve_path(protocol["model"]["checkpoint"])
    dataset_path = resolve_path(protocol["dataset"]["source"])
    if sha256_file(model_path) != prep["model_checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash does not match v8 manifest")
    if sha256_file(dataset_path) != prep["dataset_sha256"]:
        raise ValueError("Frozen OPSD data hash does not match v8 manifest")
    metadata_path = dataset_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError("Dataset metadata with scale_mw is required: %s" % metadata_path)
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        scale_mw = float(dataset_metadata["scale_mw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Dataset metadata scale_mw must be numeric") from exc
    if not np.isfinite(scale_mw) or scale_mw <= 0:
        raise ValueError("Dataset metadata scale_mw must be finite and positive")
    # Access only the LS and VS arrays. No TEST targets are loaded.
    with np.load(dataset_path, allow_pickle=False) as data:
        x_ls = np.asarray(data["x_ls"])
        y_ls = np.asarray(data["y_ls"])
        x_vs = np.asarray(data["x_vs"])
        y_vs = np.asarray(data["y_vs"])
    _, _, x_vs_scaled, _, _, _, scaler = scale_data_multi(x_ls, y_ls, x_vs, y_vs)
    conditions = np.concatenate((x_vs_scaled, scaler.transform(x_vs[:, :24])), axis=1).astype(np.float32)
    vs_start_date = dataset_metadata.get("date_ranges", {}).get("vs", [None])[0]
    if not vs_start_date:
        raise ValueError("Dataset metadata lacks the frozen VS start date")
    if int(dataset_metadata.get("split_days", {}).get("vs", -1)) != len(x_vs):
        raise ValueError("Dataset metadata VS row count differs from the frozen split")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    device = torch.device("cpu")
    model.to(device)
    model.device = device
    model.eval()

    tasks = _vs_tasks(preparation, protocol)
    rows = []
    for task in tasks:
        source_index = int(task["index"])
        export_index = int(task["export_index"])
        task_for_condition = dict(task)
        task_for_condition["source_problem_path"] = resolve_path(task["source_problem_path"])
        condition, condition_max_abs_error = frozen_vs_condition(
            task_for_condition, conditions, vs_start_date)
        sample_path = resolve_path(task["reference_sample_path"])
        if sha256_file(sample_path) != task["reference_sample_sha256"]:
            raise ValueError("Frozen VS SA sample hash mismatch: %s" % sample_path)
        with np.load(sample_path, allow_pickle=False) as archive:
            spins = np.asarray(archive["samples"])
        if spins.shape != (100, int(model.latent_s)) or not np.isin(spins, (-1, 1)).all():
            raise ValueError("Expected 100 valid archived 48-bit VS SA samples")
        n = min(int(max_samples_per_condition), spins.shape[0])
        binary = ((spins[:n].astype(np.int8) + 1) // 2).astype(np.float32)
        base_mean, base_scale = _decode_parameters(model, binary, condition, device)
        flip_z = np.repeat(binary, int(model.latent_s), axis=0)
        flip_bits = np.tile(np.arange(int(model.latent_s)), n)
        flip_z[np.arange(len(flip_z)), flip_bits] = 1.0 - flip_z[
            np.arange(len(flip_z)), flip_bits]
        changed_mean, changed_scale = _decode_parameters(model, flip_z, condition, device)

        flip_summary = summarize_flip_sensitivity(
            base_mean,
            changed_mean.reshape(n, int(model.latent_s), -1),
            base_scale,
            changed_scale.reshape(n, int(model.latent_s), -1),
            scaler.scale_,
            scale_mw,
        )
        mean_shift_rms = flip_summary["mean_shift_rms_mw"]
        observation_sigma = flip_summary["observation_sigma_rms_mw_by_sample"]
        mean_to_noise = flip_summary["mean_shift_to_observation_sigma"]
        scale_relative_change = flip_summary["relative_observation_scale_change_rms"]
        per_flip = []
        for sample_index in range(n):
            for latent_bit in range(int(model.latent_s)):
                per_flip.append({
                    "sample_index": sample_index,
                    "latent_bit": latent_bit,
                    "mean_shift_rms_mw": float(mean_shift_rms[sample_index, latent_bit]),
                    "observation_sigma_rms_mw": float(observation_sigma[sample_index]),
                    "mean_shift_to_observation_sigma": float(mean_to_noise[sample_index, latent_bit]),
                    "relative_observation_scale_change_rms": float(scale_relative_change[sample_index, latent_bit]),
                })
        rows.append({
            "instance_id": str(task["instance_id"]),
            "date": str(task.get("date", "")),
            "gain": float(task["gain"]),
            "index": source_index,
            "export_index": export_index,
            "condition_source_sha256": task["source_problem_sha256"],
            "condition_matches_indexed_vs_row": True,
            "condition_max_abs_error": condition_max_abs_error,
            "condition_date_verified": True,
            "source_sample_sha256": task["reference_sample_sha256"],
            "n_archived_samples": int(n),
            "single_latent_bit_flips_per_sample": int(model.latent_s),
            "mean_shift_rms_mw": flip_summary["summary"]["mean_shift_rms_mw"],
            "observation_sigma_rms_mw": flip_summary["summary"]["observation_sigma_rms_mw"],
            "mean_shift_to_observation_sigma": flip_summary["summary"]["mean_shift_to_observation_sigma"],
            "relative_observation_scale_change_rms": flip_summary["summary"]["relative_observation_scale_change_rms"],
            "per_flip": per_flip,
        })
    return {
        "artifact_role": "frozen_vs_latent_local_sensitivity_diagnostic",
        "method": "For each frozen problem, verify its source hash, x_cond and date/index mapping against the full VS split using the original source index; export_index is metadata/order only. Decode the first N archived quantized-matrix SA samples, flip each latent bit separately, and compare paired decoder parameters without observation noise. Arrays are explicitly sample x flip-bit x hour; for every flip, RMS is computed across 24 hours, then median and P95 across all sample/bit flips. Every flip-level summary is retained.",
        "unit_conversion": {
            "decoder_unit": "LS-fitted StandardScaler target units over normalized power",
            "formula_for_differences_and_sigma": "standardized_delta_or_sigma * target_scaler.scale_[hour] * dataset_metadata.scale_mw",
            "scaler_mean_applied": False,
            "scale_mw": scale_mw,
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "reported_unit": "MW",
        },
        "noise_comparison": "Per-sample decoder marginal sigma is mapped from standardized target units through target_scaler.scale_ and dataset scale_mw; each 24-hour vector is summarized by RMS, then compared with each bit-flip mean RMS.",
        "limitations": [
            "Local one-bit flips may leave the learned latent/Ising sample distribution.",
            "Archived samples are classical SA references, not SPQC-550 samples.",
            "Only the first 20 of 100 archived reads per condition are used for this diagnostic.",
            "This describes decoder sensitivity and is not predictive skill, hardware quality, or evidence for changing the frozen checkpoint.",
            "No TEST arrays or targets are read; no sampling or training is performed."
        ],
        "checkpoint_sha256": prep["model_checkpoint_sha256"],
        "dataset_sha256": prep["dataset_sha256"],
        "dataset_metadata_sha256": sha256_file(metadata_path),
        "protocol_sha256": sha256_file(protocol_path),
        "n_conditions": len(rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-condition", type=int, default=20)
    args = parser.parse_args()
    result = run_diagnostic(args.protocol, args.preparation, args.samples_per_condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"path": str(args.output.resolve()), "conditions": result["n_conditions"],
                      "checkpoint_sha256": result["checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
