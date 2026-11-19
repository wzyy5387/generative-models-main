# -*- coding: utf-8 -*-
"""VS-only paired SA state-diversity and existing J=0 audit for OPSD."""

import argparse
import csv
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily
from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import scenario_metrics
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import (
    ENERGY_CONVENTION, matrix_sha256, normalize_hardware_spins,
)
from GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios import (
    decode_latent_samples_with_paired_noise, decode_platform_spins,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json"
PREPARATION = ROOT / "export/opsd_sa_negative_kaiwu_preparation_10read_v1"
DATASET = ROOT / "GEFcom2014/data/external/opsd_wind_daily.npz"
DEFAULT_OUTPUT = ROOT / "export/opsd_sa_state_diversity_vs_v4"
METRICS = ("CRPS", "Energy", "Variogram", "PICP90", "interval_width90")
VS_CONFIG_FIELDS = (
    "dataset_bundle", "sampler", "epochs", "latent_s", "enc_w", "enc_l",
    "dec_w", "dec_l", "decoder_covariance", "forecast_anchor",
    "learning_rate", "weight_decay", "batch_size", "qbm_weight",
    "warmup_epochs", "negative_steps", "negative_phase_backend",
    "negative_num_reads", "negative_sampler_sweeps", "negative_sampler_burn_in",
    "beta", "temperature_start", "temperature_end", "early_stopping_patience",
    "n_scenarios", "scenario_splits", "checkpoint_selection",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _scalar(archive, key):
    value = np.asarray(archive[key]).reshape(-1)
    if value.size != 1:
        raise ValueError("Expected scalar field %s" % key)
    return value[0].item()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _date(value):
    return str(value).replace("T", " ").split(" ")[0]


def _standard_noise(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((100, 24), generator=generator, dtype=torch.float32)


def _validate_inputs(protocol, prep, reference_manifest):
    if _sha(PROTOCOL) != prep.get("protocol_sha256"):
        raise ValueError("10-read protocol hash does not match preparation")
    if prep.get("quality_reference_manifest_sha256") != _sha(PREPARATION / "quality_reference_manifest.json"):
        raise ValueError("Quality reference manifest hash mismatch")
    if prep.get("protocol_version") != protocol.get("protocol_version"):
        raise ValueError("Preparation protocol version mismatch")
    if reference_manifest.get("protocol_sha256") != _sha(PROTOCOL):
        raise ValueError("Quality reference protocol hash mismatch")
    if (reference_manifest.get("hardware_claim") is not False
            or reference_manifest.get("physical_platform_used") is not False
            or len(reference_manifest.get("instances", [])) != 30):
        raise ValueError("Expected 30 explicitly non-physical SA reference conditions")
    if protocol["hardware"]["reads_per_instance"] != 10:
        raise ValueError("This analysis requires the frozen 10-read protocol")
    if protocol["inference"]["prediction_curves_per_date"] != 100:
        raise ValueError("This analysis requires 100 prediction curves per date")
    if protocol["gain_selection"]["prediction_evaluation"] != {
        "noise_seed": 2026, "noise_seed_rule": "base_plus_export_index",
        "observation_noise": True, "postprocessing": "raw_clip_0_1",
    }:
        raise ValueError("Unsupported frozen VS decoder/noise/postprocessing settings")


def _state_diversity(out, protocol, prep, reference_manifest):
    model_path = _path(protocol["model"]["checkpoint"])
    config_path = _path(protocol["model"]["model_config"])
    dataset_path = _path(protocol["dataset"]["source"])
    frozen_audit = _json(PREPARATION / "frozen_model_audit.json")
    if _sha(model_path) != prep["model_checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash mismatch")
    if _sha(model_path) != frozen_audit["files"]["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from frozen model audit")
    if _sha(dataset_path) != prep["dataset_sha256"]:
        raise ValueError("Dataset bundle hash mismatch")
    if _sha(config_path) != _sha(model_path.with_suffix(".json")):
        raise ValueError("Model configuration path/hash does not match checkpoint family")

    # The NPZ is accessed only for LS and VS arrays. No TEST key is read.
    with np.load(dataset_path, allow_pickle=False) as dataset:
        x_ls = np.asarray(dataset["x_ls"])
        y_ls = np.asarray(dataset["y_ls"])
        x_vs = np.asarray(dataset["x_vs"])
        y_vs = np.asarray(dataset["y_vs"])
        dates_vs = np.asarray(dataset["dates_vs"])
    if x_vs.ndim != 2 or y_vs.shape != (len(x_vs), 24) or len(dates_vs) != len(x_vs):
        raise ValueError("Frozen VS array dimensions are inconsistent")
    _, _, x_scaled, _, _, _, scaler = scale_data_multi(x_ls, y_ls, x_vs, y_vs)
    conditions = np.concatenate((x_scaled, scaler.transform(x_vs[:, :24])), axis=1).astype(np.float32)
    selection = _json(PREPARATION / "vs_selection_manifest.json")
    if (selection.get("test_data_used") is not False
            or selection.get("indices") != [65, 7, 40, 33, 14, 26, 47, 17, 35, 74]
            or len(selection.get("dates", [])) != 10):
        raise ValueError("Frozen VS selection manifest is unexpected")
    export_by_index = {int(index): offset for offset, index in enumerate(selection["indices"])}

    # Read and hash the three frozen logical condition manifests; no TEST arrays.
    source_rows = {}
    manifest_hashes = prep["manifest_hashes"]["vs_manifests"]
    for manifest_value in prep["vs_manifests"]:
        manifest_path = _path(manifest_value)
        rows = _json(manifest_path)
        if len(rows) != 10:
            raise ValueError("A frozen VS source manifest must contain ten dates")
        gain = int(float(rows[0]["hardware_gain"]))
        if _sha(manifest_path) != manifest_hashes[str(gain)]:
            raise ValueError("Frozen logical VS manifest hash mismatch for gain %d" % gain)
        for row in rows:
            key = (float(gain), str(row["id"]))
            if key in source_rows or row.get("split") != "VS":
                raise ValueError("Duplicate/non-VS logical condition manifest entry")
            source_rows[key] = (row, manifest_path)

    model_config = _json(config_path)
    if model_config.get("negative_phase_backend") != "sa" or model_config.get("sampler") != "sa":
        raise ValueError("Frozen seed-0 checkpoint metadata does not declare SA")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    model.to("cpu")
    model.device = torch.device("cpu")
    model.eval()

    noise_seed_base = int(protocol["gain_selection"]["prediction_evaluation"]["noise_seed"])
    noise_hash_by_date = {}
    rows_out, json_out, input_files = [], [], {}
    entries = {(float(item["gain"]), str(item["instance_id"])): item
               for item in reference_manifest["instances"]}
    if len(entries) != 30:
        raise ValueError("Reference manifest has duplicate conditions")

    for (gain, instance_id), ref in sorted(entries.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        key = (gain, instance_id)
        if key not in source_rows:
            raise ValueError("No frozen source manifest entry for %s gain %s" % key)
        item, manifest_path = source_rows[key]
        index, export_index = int(item["index"]), int(item["export_index"])
        expected_export_index = export_by_index.get(index)
        date = _date(item["date"])
        if (expected_export_index is None or expected_export_index != export_index
                or _date(dates_vs[index]) != date or _date(ref["date"]) != date):
            raise ValueError("Frozen VS date/index/export_index mapping mismatch: %s" % instance_id)
        source_path = _path(item["path"])
        if source_path != _path(ref["source_problem_path"]):
            raise ValueError("Reference manifest points to a different frozen condition source")
        if _sha(source_path) != ref["source_problem_sha256"]:
            raise ValueError("Frozen condition source hash mismatch")
        matrix_path = _path(ref["hardware_matrix_path"])
        if _sha(matrix_path) != ref["hardware_matrix_file_sha256"]:
            raise ValueError("Frozen hardware matrix file hash mismatch")
        with np.load(matrix_path, allow_pickle=False) as matrix_archive:
            matrix = np.asarray(matrix_archive["hardware_matrix"])
            if str(_scalar(matrix_archive, "hardware_energy_convention")) != ENERGY_CONVENTION:
                raise ValueError("Hardware matrix energy convention mismatch")
        if matrix_sha256(matrix) != ref["hardware_matrix_sha256"] or matrix_sha256(matrix) != item["hardware_matrix_sha256"]:
            raise ValueError("Hardware matrix identity hash mismatch")
        sample_path = _path(ref["reference_sample_path"])
        if _sha(sample_path) != ref["reference_sample_sha256"]:
            raise ValueError("SA reference file hash mismatch")
        if _sha(sample_path) != entries[key]["reference_sample_sha256"]:
            raise ValueError("SA reference identity changed during audit")
        with np.load(source_path, allow_pickle=False) as source:
            condition = np.asarray(source["x_cond"], dtype=np.float32)
            if (source["h"].shape != (48,) or source["J"].shape != (48, 48)
                    or np.asarray(source["J"]).dtype.kind not in "fiu"):
                raise ValueError("Frozen logical source has invalid h/J dimensions")
            if not np.allclose(condition, conditions[index], rtol=0.0, atol=1e-6):
                raise ValueError("Frozen x_cond differs from LS-scaled VS row %d" % index)
            if np.asarray(source["hardware_matrix"]).shape != (49, 49):
                raise ValueError("Frozen condition source lacks its 49x49 matrix")
        sampler = ref["sampler"]
        if (sampler.get("backend") != "sa" or sampler.get("matrix_space") != "hardware-quantized"
                or int(sampler.get("reads", -1)) != 10 or int(sampler.get("sweeps", -1)) != 100
                or int(sampler.get("seed", -1)) != noise_seed_base + export_index
                or ref.get("hardware_claim") is not False
                or ref.get("physical_platform_used") is not False):
            raise ValueError("SA reference sampling provenance mismatch")
        with np.load(sample_path, allow_pickle=False) as ref_archive:
            spins = np.asarray(ref_archive["samples"])
            if spins.shape != (10, 48) or not np.isin(spins, (-1, 1)).all():
                raise ValueError("SA reference must retain ten ordered 48-bit latent states")
            logical = normalize_hardware_spins(ref_archive["hardware_samples"], logical_n_bits=48)
            if not np.array_equal(logical, spins):
                raise ValueError("Saved logical and 49-spin SA samples disagree")
            for field, expected in (("instance_id", instance_id), ("hardware_matrix_sha256", ref["hardware_matrix_sha256"]),
                                    ("protocol_sha256", _sha(PROTOCOL)), ("requested_reads", 10),
                                    ("sweeps", 100), ("random_seed", int(sampler["seed"]))):
                if _scalar(ref_archive, field) != expected:
                    raise ValueError("SA reference metadata mismatch: %s" % field)
        if not np.array_equal(matrix, np.asarray(np.load(source_path, allow_pickle=False)["hardware_matrix"])):
            raise ValueError("Matrix array differs between package and source condition")

        seed = noise_seed_base + export_index
        noise = _standard_noise(seed)
        noise_digest = hashlib.sha256(noise.numpy().tobytes(order="C")).hexdigest()
        if date in noise_hash_by_date and noise_hash_by_date[date] != noise_digest:
            raise ValueError("Different gains did not receive identical date-paired noise")
        noise_hash_by_date[date] = noise_digest

        # A: retain original latent order/duplicates, ten curves per state.
        multi = decode_latent_samples_with_paired_noise(
            model, spins, condition, scaler, prediction_curves=100, seed=seed,
        )
        expanded_multi = np.repeat(spins, 10, axis=0)
        multi_explicit = decode_platform_spins(
            model, expanded_multi, condition, scaler,
            observation_noise=True, observation_noise_values=noise,
        )
        if not np.array_equal(multi, multi_explicit):
            raise ValueError("Paired-noise helper and explicit shared-noise decode differ")

        # B: all ten fixed-state forecasts use the same 100x24 noise in order.
        fixed_spins = np.repeat(spins, 100, axis=0)
        fixed_noise = noise.repeat(10, 1)
        fixed_curves = decode_platform_spins(
            model, fixed_spins, condition, scaler,
            observation_noise=True, observation_noise_values=fixed_noise,
        ).reshape(10, 100, 24)
        target = np.asarray(y_vs[index], dtype=np.float64)
        if (multi.shape != (100, 24) or fixed_curves.shape != (10, 100, 24)
                or target.shape != (24,) or not np.isfinite(multi).all()
                or not np.isfinite(fixed_curves).all() or not np.isfinite(target).all()):
            raise ValueError("Invalid/non-finite decoded VS curves or target")
        multi_metrics = scenario_metrics(multi, target)
        fixed_state_metrics = [scenario_metrics(fixed_curves[i], target) for i in range(10)]
        fixed_mean = {name: float(np.mean([m[name] for m in fixed_state_metrics]))
                      for name in fixed_state_metrics[0]}
        unique, counts = np.unique(spins, axis=0, return_counts=True)
        counts_list = [int(v) for v in counts]
        output_row = {
            "instance_id": instance_id, "date": date, "index": index,
            "export_index": export_index, "gain": int(gain),
            "n_latent_samples": int(len(spins)), "n_prediction_curves": 100,
            "unique_latent_states": int(len(unique)),
            "max_state_count": int(counts.max()),
            "max_state_fraction": float(counts.max() / len(spins)),
            "noise_seed": seed, "paired_noise_sha256": noise_digest,
            "source_problem_sha256": _sha(source_path),
            "matrix_sha256": matrix_sha256(matrix),
            "sa_reference_sha256": _sha(sample_path),
        }
        for metric in multi_metrics:
            output_row["multi_" + metric] = float(multi_metrics[metric])
            output_row["fixed_state_mean_" + metric] = fixed_mean[metric]
            output_row["multi_minus_fixed_" + metric] = float(multi_metrics[metric] - fixed_mean[metric])
        rows_out.append(output_row)
        json_out.append({
            **output_row, "state_counts_in_original_order": counts_list,
            "fixed_state_scores": fixed_state_metrics,
            "interpretation": "fixed-state scores averaged over all ten observed SA states; no best-state selection",
        })
        for path in (manifest_path, source_path, matrix_path, sample_path):
            input_files[str(path.relative_to(ROOT))] = _sha(path)

    if len(rows_out) != 30 or len({row["date"] for row in rows_out}) != 10:
        raise ValueError("Expected exactly 30 conditions over ten VS dates")
    for metric in ("multi_CRPS", "fixed_state_mean_CRPS"):
        if not np.isfinite([row[metric] for row in rows_out]).all():
            raise ValueError("Non-finite summary metric: %s" % metric)
    _save_csv(out / "condition_metrics.csv", rows_out)
    (out / "condition_metrics.json").write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")

    date_rows = []
    date_groups = {}
    for row in rows_out:
        date_groups.setdefault(row["date"], []).append(row)
    for date, group in sorted(date_groups.items()):
        date_row = {"date": date, "n_gains": len(group)}
        for metric in multi_metrics:
            date_row["multi_" + metric] = float(np.mean([r["multi_" + metric] for r in group]))
            date_row["fixed_state_mean_" + metric] = float(np.mean([r["fixed_state_mean_" + metric] for r in group]))
            date_row["multi_minus_fixed_" + metric] = float(np.mean([r["multi_minus_fixed_" + metric] for r in group]))
        date_rows.append(date_row)
    _save_csv(out / "paired_date_differences.csv", date_rows)

    crps_diffs = np.asarray([row["multi_minus_fixed_CRPS"] for row in date_rows], dtype=np.float64)
    rng = np.random.default_rng(20261105)
    indices = rng.integers(0, len(crps_diffs), size=(20000, len(crps_diffs)))
    bootstrap = crps_diffs[indices].mean(axis=1)
    summary = {}
    for metric in multi_metrics:
        multi_values = np.asarray([row["multi_" + metric] for row in rows_out])
        fixed_values = np.asarray([row["fixed_state_mean_" + metric] for row in rows_out])
        summary[metric] = {
            "multi_state_mean_over_30_conditions": float(multi_values.mean()),
            "fixed_state_mean_over_30_conditions": float(fixed_values.mean()),
            "multi_minus_fixed_mean_over_30_conditions": float((multi_values-fixed_values).mean()),
            "date_level_mean_difference_over_10_dates": float(np.mean([r["multi_minus_fixed_"+metric] for r in date_rows])),
        }
    summary["CRPS"]["date_resampling_ci_95"] = [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]
    summary["CRPS"]["date_resampling_seed"] = 20261105
    summary["CRPS"]["date_resampling_repetitions"] = 20000
    summary["design"] = {
        "conditions": 30, "unique_prediction_dates": 10,
        "date_unit_not_date_by_gain": True,
        "gains": protocol["hardware"]["candidate_gains"],
        "noise_rule": "torch CPU float32 standard normal; seed=2026+export_index; common 100x24 rows",
        "postprocessing": "raw_clip_0_1",
        "test_data_accessed": False,
    }
    (out / "state_diversity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    dates = [r["date"] for r in date_rows]
    for gain in protocol["hardware"]["candidate_gains"]:
        group = sorted((r for r in rows_out if r["gain"] == gain), key=lambda r: r["date"])
        ax.plot(dates, [r["multi_minus_fixed_CRPS"] for r in group], marker="o", alpha=0.7,
                label="gain=%s" % gain)
    ax.plot(dates, [r["multi_minus_fixed_CRPS"] for r in date_rows], color="black", marker="s",
            linewidth=2.2, label="date mean across gains")
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_ylabel("Raw CRPS difference (multi-state − fixed-state)")
    ax.set_xlabel("Frozen VS date")
    ax.set_title("VS exploratory paired comparison; 10 dates, gains are repeated conditions")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "paired_date_crps_difference.png", dpi=180)
    plt.close(fig)
    return summary, input_files, rows_out


def _j0_vs_audit(out, input_files):
    """Reuse only saved VS scenes/targets for full FA vs fixed-zero J audit."""
    full_root = ROOT / "export/qbm_vae_opsd-wind"
    j0_root = ROOT / "export/opsd_sa_negative_matched_v1/j0"
    full_config = _json(ROOT / "configs/paper/fa_bm_vae_opsd_wind_sa_negative_v1.json")
    j0_config = _json(ROOT / "configs/paper/fa_bm_vae_opsd_wind_sa_negative_j0_v1.json")
    with np.load(DATASET, allow_pickle=False) as dataset:
        y_vs = np.asarray(dataset["y_vs"], dtype=np.float64)
        dates_vs = [_date(x) for x in np.asarray(dataset["dates_vs"])]
    if y_vs.shape != (90, 24) or len(set(dates_vs)) != 90:
        raise ValueError("Existing OPSD VS target/date panel must have 90 unique days")
    audit_path = ROOT / "export/opsd_sa_negative_matched_v1/evaluation/ablation_parameter_audit.json"
    audit_rows = _json(audit_path)
    zero_audit = {int(r["seed"]): r for r in audit_rows if r.get("model") == "J=0 ablation"}
    if set(zero_audit) != {0, 1, 2}:
        raise ValueError("Existing J=0 checkpoint audit lacks seed 0/1/2")

    metadata_rows, daily_rows = [], []
    comparisons = []
    for model_name, root, template, config in (
        ("FA-BM-VAE", full_root, "opsd-wind_QBMVAE_2_anchor_sa_neg_sa_{seed}", full_config),
        ("J=0 / independent prior", j0_root, "opsd-wind_QBMVAE_2_abl_j0_sa_neg_v1_sa_{seed}", j0_config),
    ):
        for seed in (0, 1, 2):
            base = template.format(seed=seed)
            checkpoint = root / (base + ".pickle")
            metadata_path = root / (base + ".json")
            scenario_path = root / ("scenarios_" + base + "_100_VS.pickle")
            for p in (checkpoint, metadata_path, scenario_path):
                if not p.is_file():
                    raise FileNotFoundError("Missing existing VS artifact: %s" % p)
                input_files[str(p.relative_to(ROOT))] = _sha(p)
            meta = _json(metadata_path)
            if meta.get("sampler") != "sa" or meta.get("negative_phase_backend") != "sa":
                raise ValueError("Existing %s seed %d is not SA-based" % (model_name, seed))
            if meta.get("decoder_anchor") is not True or meta.get("anchor_source") != "context":
                raise ValueError("Existing %s seed %d does not use the forecast context anchor" % (model_name, seed))
            with checkpoint.open("rb") as handle:
                model = pickle.load(handle)
            if model_name.startswith("J=0"):
                raw_j = model.raw_j.detach().cpu().numpy()
                effective_j = model.qbm_params()[1].detach().cpu().numpy()
                if (np.max(np.abs(raw_j)) != 0.0 or np.max(np.abs(effective_j)) != 0.0
                        or model.raw_j.requires_grad
                        or _sha(checkpoint) != zero_audit[seed]["checkpoint_sha256"]):
                    raise ValueError("J=0 checkpoint is not structurally fixed at zero")
                graph_state = {"mode": meta.get("graph_mode"), "edges": meta.get("graph_edges"),
                               "raw_j_absmax": float(np.max(np.abs(raw_j))),
                               "effective_j_absmax": float(np.max(np.abs(effective_j))),
                               "raw_j_requires_grad": bool(model.raw_j.requires_grad)}
            else:
                graph_mask = model.graph_mask.detach().cpu().numpy()
                if (graph_mask.ndim != 2 or graph_mask.shape[0] != graph_mask.shape[1]
                        or not np.array_equal(graph_mask, graph_mask.T)
                        or np.any(np.diag(graph_mask) != 0)):
                    raise ValueError("Frozen FA graph mask is not symmetric with a zero diagonal")
                graph_state = {
                    "mode": meta.get("graph_mode"),
                    "metadata_graph_edges_declared": meta.get("graph_edges"),
                    "actual_nonzero_undirected_edges": int(np.count_nonzero(np.triu(graph_mask, k=1))),
                    "graph_mask_sha256": hashlib.sha256(np.asarray(graph_mask).tobytes()).hexdigest(),
                }
            with scenario_path.open("rb") as handle:
                scenarios = np.asarray(pickle.load(handle), dtype=np.float64)
            if scenarios.shape != (90 * 24, 100) or not np.isfinite(scenarios).all():
                raise ValueError("Saved %s seed %d VS scenarios must be finite (2160,100)" % (model_name, seed))
            if scenarios.min() < -1e-8 or scenarios.max() > 1.0 + 1e-8:
                raise ValueError("Saved Raw VS scenarios do not match the raw_clip_0_1 range")
            daily = evaluate_model_daily(scenarios, y_vs, max_s=100, periods_per_day=24)
            sample_cube = scenarios.T.reshape(100, 90, 24).transpose(1, 2, 0)
            bias = (sample_cube.mean(axis=2) - y_vs).mean(axis=1)
            for d, date in enumerate(dates_vs):
                daily_rows.append({
                    "model": model_name, "training_seed": seed, "date": date,
                    "n_scenarios": 100, "CRPS_raw": float(daily["CRPS_raw"][d]),
                    "Energy": float(daily["Energy"][d]), "Variogram": float(daily["Variogram"][d]),
                    "ramp_CRPS": float(daily["ramp_CRPS"][d]), "PICP90": float(daily["PICP90"][d]),
                    "interval_width90": float(daily["interval_width90"][d]),
                    "MAQCE": float(daily["MAQCE"][d]), "mean_bias": float(bias[d]),
                })
            metadata_rows.append({
                "model": model_name, "seed": seed,
                "checkpoint_sha256": _sha(checkpoint), "metadata_sha256": _sha(metadata_path),
                "vs_scenarios_sha256": _sha(scenario_path), "best_epoch": meta.get("best_epoch"),
                "checkpoint_selection_metric": meta.get("checkpoint_selection_metric"),
                "generation_seed": meta.get("generation_seed"),
                "generation_seed_explicit": meta.get("generation_seed_explicit"),
                "scenario_count": 100, "scenario_shape": list(scenarios.shape),
                "scenario_range": [float(scenarios.min()), float(scenarios.max())],
                "graph_state": graph_state,
                "training_backend": meta.get("training_backend"),
                "model_metadata_seed": meta.get("seed"),
                "dataset_bundle": meta.get("dataset_bundle"),
            })

    full_fields, j0_fields = {}, {}
    for field in VS_CONFIG_FIELDS:
        full_fields[field] = full_config.get(field)
        j0_fields[field] = j0_config.get(field)
    differing = [field for field in VS_CONFIG_FIELDS if full_fields[field] != j0_fields[field]]
    expected_differences = ["checkpoint_selection"] if full_fields["checkpoint_selection"] != j0_fields["checkpoint_selection"] else []
    # Graph identity is intentionally excluded from shared training knobs.
    config_equal = not differing

    daily_map = {(r["model"], r["training_seed"], r["date"]): r for r in daily_rows}
    paired_rows = []
    for date in dates_vs:
        for seed in (0, 1, 2):
            full = daily_map[("FA-BM-VAE", seed, date)]
            j0 = daily_map[("J=0 / independent prior", seed, date)]
            paired = {"date": date, "training_seed": seed}
            for metric in ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "interval_width90", "MAQCE", "mean_bias"):
                paired[metric + "_FA_minus_J0"] = full[metric] - j0[metric]
            paired_rows.append(paired)
    mean_by_date = []
    for date in dates_vs:
        group = [r for r in paired_rows if r["date"] == date]
        row = {"date": date, "training_seeds_averaged_first": "0,1,2"}
        for metric in ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "interval_width90", "MAQCE", "mean_bias"):
            row[metric + "_FA_minus_J0"] = float(np.mean([r[metric + "_FA_minus_J0"] for r in group]))
        mean_by_date.append(row)
    summary_rows = []
    for model_name in ("FA-BM-VAE", "J=0 / independent prior"):
        for metric in ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "interval_width90", "MAQCE", "mean_bias"):
            seed_values = []
            for seed in (0, 1, 2):
                values = [r[metric] for r in daily_rows if r["model"] == model_name and r["training_seed"] == seed]
                seed_values.append(float(np.mean(values)))
            summary_rows.append({"model": model_name, "metric": metric,
                                 "mean_across_seed_means": float(np.mean(seed_values)),
                                 "sample_std_across_seed_means": float(np.std(seed_values, ddof=1)),
                                 "seed_0_mean": seed_values[0], "seed_1_mean": seed_values[1],
                                 "seed_2_mean": seed_values[2], "n_vs_dates": 90})
    paired_summary = {}
    for metric in ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "interval_width90", "MAQCE", "mean_bias"):
        seed_diffs = []
        for seed in (0, 1, 2):
            seed_diffs.append(float(np.mean([r[metric + "_FA_minus_J0"] for r in paired_rows if r["training_seed"] == seed])))
        paired_summary[metric] = {
            "FA_minus_J0_mean_across_seed_means": float(np.mean(seed_diffs)),
            "per_seed_date_mean_differences": seed_diffs,
            "date_unit": "90 unique VS dates; average seeds within date for date-level view",
        }
    _save_csv(out / "j0_vs_per_seed_date.csv", daily_rows)
    _save_csv(out / "j0_vs_paired_date_by_seed.csv", paired_rows)
    _save_csv(out / "j0_vs_paired_date_seed_averaged.csv", mean_by_date)
    _save_csv(out / "j0_vs_summary.csv", summary_rows)
    audit = {
        "scope": "VS only; existing saved scenarios and VS targets; no TEST arrays/labels read",
        "dataset_sha256": _sha(DATASET), "vs_dates": dates_vs,
        "shared_config_fields_match": config_equal,
        "shared_config_differences": differing,
        "full_config_protocol": full_config.get("protocol_version"),
        "j0_config_protocol": j0_config.get("protocol_version"),
        "full_graph": full_config.get("graph_mode"), "j0_graph": j0_config.get("graph_mode"),
        "direct_coupling_ablation": True,
        "j0_checkpoints_fixed_zero_verified_seeds": [0, 1, 2],
        "metadata_and_scenario_provenance": metadata_rows,
        "paired_summary_FA_minus_J0": paired_summary,
        "selection_rule_note": "Historical main seed-0..2 checkpoint selection is reported in source audit as minimum VS reconstruction; old model JSON does not record selected epochs. J0 metadata records VS reconstruction and best_epoch.",
        "generation_seed_note": "generation_seed metadata is recorded per artifact below; missing legacy values limit exact scenario regeneration but do not change the saved-scenario VS arithmetic.",
        "non_comparability_caveat": "The comparison is a saved-scenario VS comparison under matched anchor/data/architecture/training knobs. Different parameter seeds remain model replicates; 100 curves are ensemble members, not independent dates.",
        "expected_non_graph_config_differences": expected_differences,
    }
    (out / "j0_comparability_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit, input_files


def _readiness_audit(out):
    prep = PREPARATION
    refs = _json(prep / "quality_reference_manifest.json")
    gibbs_files = [str(p.relative_to(ROOT)) for p in prep.rglob("*")
                   if p.is_file() and any(t in p.name.lower() for t in ("gibbs", "heatbath"))]
    response_dir = prep / "real_responses"
    checkpoint_dir = prep / "checkpoints"
    intent_files = list(checkpoint_dir.glob("*.intent.json")) if checkpoint_dir.exists() else []
    hardware_files = []
    for entry in refs["instances"]:
        if entry["gain"] == 150:
            hardware_files.append({"instance_id": entry["instance_id"], "matrix_sha256": entry["hardware_matrix_sha256"]})
    if len(hardware_files) != 10:
        raise ValueError("The 10-read gain-150 frozen condition panel is incomplete")
    target = next(x for x in hardware_files if x["instance_id"].endswith("_vs_000"))
    readiness = {
        "physical_sdk_calls_made_by_this_analysis": 0,
        "gibbs_references_for_frozen_30_conditions_found": bool(gibbs_files),
        "gibbs_matching_files": gibbs_files,
        "quality_reference_backends": sorted({str(x["sampler"]["backend"]) for x in refs["instances"]}),
        "gain_150_target": target,
        "target_response_dir_exists": response_dir.exists(),
        "target_checkpoint_dir_exists": checkpoint_dir.exists(),
        "pending_intents": [str(p) for p in intent_files],
        "matching_10read_platform_receipt_found": False,
        "historical_v9_100read_receipt_is_different_protocol": True,
        "local_identity_selection_consistent": True,
        "dry_run_evidence": "previously recorded exact dry-run task=1, reads=10, gain=150, matrix hash above; no response/checkpoint directory was created",
        "mock_sdk_evidence": "tests/test_kaiwu_10read_inference.py exercises normal driver with a fully replaced client and checks task name, reads and matrix hash; this is not hardware evidence",
        "submission_entry": "--canary plus explicit --canary-instance-id/--canary-gain/--canary-matrix-sha256/--limit 1/--execute/--confirm-real-submission/--confirm-canary; normal protocol/hash/intent safety remains active",
        "external_preconditions_unverified": ["Kaiwu license/auth", "SPQC-550 route", "current Credits and reads-to-Credits conversion", "real platform task ID/receipt semantics"],
    }
    (out / "gibbs_hardware_readiness.json").write_text(json.dumps(readiness, indent=2, ensure_ascii=False), encoding="utf-8")
    return readiness


def _write_report(out, state_summary, j0_audit, readiness):
    crps = state_summary["CRPS"]
    full = j0_audit["paired_summary_FA_minus_J0"]["CRPS_raw"]
    full_graph = next(
        row["graph_state"] for row in j0_audit["metadata_and_scenario_provenance"]
        if row["model"] == "FA-BM-VAE" and row["seed"] == 0
    )
    report = f"""# OPSD SA 潜在状态与耦合贡献：VS 离线核查

## 范围和身份

本报告只使用冻结 seed-0 FA-BM-VAE、10-read 准备包中的 30 个 VS 条件与量化矩阵 SA 参考、以及已有完整模型/J=0 checkpoint 和保存的 VS 场景。没有读取 TEST 目标，没有计算 TEST 指标，没有调用 SDK、提交任务或生成新潜变量样本。所有新推理使用固定的配对解码噪声，不改冻结协议和输入。

协议 SHA-256：`{_sha(PROTOCOL)}`。准备包 manifest SHA-256：`{_sha(PREPARATION / 'preparation_manifest.json')}`。数据 SHA-256：`{_sha(DATASET)}`。冻结 seed-0 checkpoint SHA-256：`{_sha(_path(_json(PROTOCOL)['model']['checkpoint']))}`。

## 多状态与固定状态

A 使用每条件原始顺序的 10 条 SA 逻辑状态（保留重复），每个状态重复 10 次；B 对这 10 条状态逐一固定，每个固定状态使用同一日期的同一组 100×24 标准正态噪声，逐状态评分后取十个分数的平均。两组均使用相同模型、条件/scaler、100 条解码曲线和 `raw_clip_0_1` 后处理。噪声种子固定为 `2026 + export_index`；`index` 只定位 VS 原始行，`export_index` 只控制噪声种子。

30 个“日期×gain”条件的平均 Raw CRPS：多状态 `{crps['multi_state_mean_over_30_conditions']:.6f}`；固定状态 `{crps['fixed_state_mean_over_30_conditions']:.6f}`；差值（多状态−固定状态）`{crps['multi_minus_fixed_mean_over_30_conditions']:.6f}`。先对每日期的三个 gain 求平均，再以 10 个日期为单位作探索性重采样，95% 区间 `[{crps['date_resampling_ci_95'][0]:.6f}, {crps['date_resampling_ci_95'][1]:.6f}]`。3 个 gain 复用相同 10 天，未作为 30 个独立日期。预期内存复算参考约 0.032280/0.032710；本次是否数值复现由上面实际结果决定，未据参考数调参。

辅助指标（Energy、Variogram、PICP90、90%区间宽度）、每条件状态多样性、各固定状态的完整得分见 `condition_metrics.csv`、`condition_metrics.json` 和 `state_diversity_summary.json`。逐日期差值图：`paired_date_crps_difference.png`。

这只是 10 个冻结 VS 日期上的状态集合敏感性探索。它不证明真机优于 SA，不证明有限 SA 样本等于完整 Boltzmann 分布，也不用于 gain 选择。

## 已有完整 FA 与 J=0 VS 场景

J=0 checkpoint 的 0/1/2 种子均经当前 checkpoint 重新核验 `raw_j=0`、`effective_j=0`、`raw_j` 不可训练；只差图耦合的条件模型设置在共享的锚点、网络、SA 训练及 VS reconstruction 选模配置上匹配。使用已有的三个训练种子、每日期 100 条 VS 场景，在冻结的 90 个 VS 日期上按同一 Raw 指标重算；没有重新训练或采样。

图来源复核：完整模型 checkpoint 的 48×48 `graph_mask` 实际有 {full_graph['actual_nonzero_undirected_edges']} 条非零无向边，SHA-256 `{full_graph['graph_mask_sha256']}`；旧模型 JSON 的 `graph_edges` 字段声明 {full_graph['metadata_graph_edges_declared']}，与 checkpoint 不一致。边数和哈希以 checkpoint 实际掩码为准；J=0 checkpoint 的实际耦合均为零。

FA-BM-VAE 相对 J=0 的全年 VS CRPS 差（FA−J=0），先在每个 seed 的日期内求均值，再平均三个 seed：`{full['FA_minus_J0_mean_across_seed_means']:.6f}`；逐 seed 差：`{', '.join(f'{x:.6f}' for x in full['per_seed_date_mean_differences'])}`。负数表示 FA-BM-VAE CRPS 较低。其它 VS 指标以及日期级结果见 `j0_vs_summary.csv`、`j0_vs_per_seed_date.csv`、`j0_vs_paired_date_by_seed.csv`。

这是已有场景的 VS 复算，不是 TEST 结论。历史完整模型 seed-0..2 的部分选中 epoch/生成 seed 元数据不完整，不能声称逐随机流完全配对；J=0 对照回答离散潜变量耦合是否有贡献，区别于前一节的多状态/单状态比较。

## Gibbs 与单任务真机准备

Gibbs 对照：{'已发现匹配参考' if readiness['gibbs_references_for_frozen_30_conditions_found'] else '未发现与 30 个冻结条件匹配的 Gibbs 样本'}；当前参考 manifest 只有 SA 后端。后续 Gibbs 需以 `H_sdk(K,u)=-uᵀKu` 为提交矩阵能量，按 `H_sdk/gain`、逻辑 beta=1 采样（等价整数矩阵 beta=1/gain）；固定随机初始化和多链种子，事先冻结预热、采样长度，并诊断能量/磁化和代表性边相关的自相关、ESS、链间一致性、翻转率及状态重复。有限 Gibbs 样本仍只是数值参考，不是精确分布。

gain=150 单任务的 dry-run、完全替换 SDK 的模拟提交测试、以及正式 canary 入口的实例/gain/reads/矩阵选择逻辑一致；对应矩阵 `{readiness['gain_150_target']['matrix_sha256']}`。本目录未找到匹配 10-read 响应、回执或 pending intent。既有 v9 100-read 回执不匹配本 10-read 任务。代码本地一致性不代表 Kaiwu license、SPQC-550 路由、Credits 或平台身份条件已确认。

## 文件与复现

- `analyze_opsd_sa_state_diversity_vs.py`：本次分析脚本副本；源码位于 `GEFcom2014/forecast_quality/opsd_sa_state_diversity_vs.py`。
- `run_manifest.json`：输入、脚本和环境指纹。
- `condition_metrics.csv/.json`、`state_diversity_summary.json`、`paired_date_differences.csv`。
- `j0_comparability_audit.json`、`j0_vs_*.csv`。
- `gibbs_hardware_readiness.json`。

重跑命令：`D:\\anaconda\\envs\\wsy\\python.exe -m GEFcom2014.forecast_quality.opsd_sa_state_diversity_vs --output-dir {out.relative_to(ROOT).as_posix()}`。脚本若输出目录非空会停止，避免覆盖。
"""
    (out / "REPORT_zh.md").write_text(report, encoding="utf-8")


def run(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir).resolve()
    partial_files = {
        "analyze_opsd_sa_state_diversity_vs.py", "condition_metrics.csv",
        "condition_metrics.json", "state_diversity_summary.json",
        "paired_date_differences.csv", "paired_date_crps_difference.png",
        "j0_comparability_audit.json", "j0_vs_per_seed_date.csv",
        "j0_vs_paired_date_by_seed.csv", "j0_vs_paired_date_seed_averaged.csv",
        "j0_vs_summary.csv", "gibbs_hardware_readiness.json", "REPORT_zh.md",
        "run_manifest.json",
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = {p.name for p in output_dir.iterdir()}
        if "run_manifest.json" in existing or not existing.issubset(partial_files):
            raise FileExistsError("Refusing to overwrite existing analysis directory: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _json(PROTOCOL)
    prep = _json(PREPARATION / "preparation_manifest.json")
    reference_manifest = _json(PREPARATION / "quality_reference_manifest.json")
    _validate_inputs(protocol, prep, reference_manifest)
    state_summary, input_files, _ = _state_diversity(output_dir, protocol, prep, reference_manifest)
    j0_audit, j0_files = _j0_vs_audit(output_dir, input_files)
    input_files.update(j0_files)
    readiness = _readiness_audit(output_dir)
    source_copy = output_dir / "analyze_opsd_sa_state_diversity_vs.py"
    shutil.copy2(Path(__file__), source_copy)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    try:
        git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        git_status = "unavailable"
    input_files.update({
        str(PROTOCOL.relative_to(ROOT)): _sha(PROTOCOL),
        str((PREPARATION / "preparation_manifest.json").relative_to(ROOT)): _sha(PREPARATION / "preparation_manifest.json"),
        str((PREPARATION / "quality_reference_manifest.json").relative_to(ROOT)): _sha(PREPARATION / "quality_reference_manifest.json"),
        str((PREPARATION / "frozen_model_audit.json").relative_to(ROOT)): _sha(PREPARATION / "frozen_model_audit.json"),
        str((PREPARATION / "vs_selection_manifest.json").relative_to(ROOT)): _sha(PREPARATION / "vs_selection_manifest.json"),
        str(DATASET.relative_to(ROOT)): _sha(DATASET),
        str(Path(__file__).relative_to(ROOT)): _sha(Path(__file__)),
        str(source_copy.relative_to(ROOT)): _sha(source_copy),
    })
    run_manifest = {
        "analysis": "OPSD SA multi-state versus fixed-state VS; existing FA versus J=0 VS audit",
        "status": "COMPLETED_VS_ONLY",
        "protocol_sha256": _sha(PROTOCOL), "preparation_manifest_sha256": _sha(PREPARATION / "preparation_manifest.json"),
        "quality_reference_manifest_sha256": _sha(PREPARATION / "quality_reference_manifest.json"),
        "inputs_sha256": dict(sorted(input_files.items())),
        "analysis_code_sha256": _sha(Path(__file__)),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "git_commit": git_commit, "git_status_at_run": git_status,
        "training_invoked": False, "new_sampling_invoked": False,
        "test_arrays_accessed": False, "test_metrics_computed": False,
        "sdk_import_or_call": False, "physical_platform_used": False,
        "state_analysis_seeds": {"decoder_noise_base": 2026, "bootstrap": 20261105},
        "state_summary": state_summary,
        "j0_vs_scope": {"dates": 90, "seeds": [0, 1, 2], "scenarios_per_date": 100},
        "output_files": ["condition_metrics.csv", "condition_metrics.json", "state_diversity_summary.json",
                         "paired_date_differences.csv", "paired_date_crps_difference.png",
                         "j0_comparability_audit.json", "j0_vs_per_seed_date.csv",
                         "j0_vs_paired_date_by_seed.csv", "j0_vs_paired_date_seed_averaged.csv",
                         "j0_vs_summary.csv", "gibbs_hardware_readiness.json", "REPORT_zh.md"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(output_dir, state_summary, j0_audit, readiness)
    return run_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir),
                      "conditions": 30, "vs_dates": 10, "test_arrays_accessed": False,
                      "sdk_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
