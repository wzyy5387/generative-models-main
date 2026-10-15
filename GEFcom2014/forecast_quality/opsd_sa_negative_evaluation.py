# -*- coding: utf-8 -*-
"""Audit and evaluate the fixed-window OPSD SA-negative experiment family.

This module only reads frozen VS/TEST scenarios.  Raw is the primary result.
Calibration is fit on VS only.  Cal+ECC is produced only when the pre-existing
same-seed TAR1M rank template exists; absent templates are reported as missing,
never estimated from TEST observations.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.forecast_quality.aggregate_wind_five_seed import block_paired_inference
from GEFcom2014.forecast_quality.compare_scenarios import evaluate_model_daily, load_scenarios
from GEFcom2014.forecast_quality.qbm_temporal_ecc import ensemble_copula_coupling
from GEFcom2014.forecast_quality.unified_postprocessing import HourlyAffineCalibrator, load_targets


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET = ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz"
OLD_QBM = ROOT_DIR / "export" / "qbm_vae_opsd-wind"
NEW_ROOT = ROOT_DIR / "export" / "opsd_sa_negative_matched_v1"
STRICT_NO_ANCHOR_ROOT = NEW_ROOT / "no_anchor_addition_zero_mean_init"
OUTPUT = NEW_ROOT / "evaluation_strict_no_anchor_v2"
N_SCENARIOS = 100
METRICS = ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "interval_width90", "MAQCE", "mean_bias")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root, template, seed, split):
    return root / template.format(seed=seed, split=split)


def scenario_specs():
    """The strict registry excludes rolling-origin, smoke, and hardware outputs."""
    return {
        "FA-BM-VAE": {
            "seeds": range(5),
            "path": lambda seed, split: _path(
                OLD_QBM if seed < 3 else NEW_ROOT / "fa_bm_vae",
                "scenarios_opsd-wind_QBMVAE_2_anchor_sa_neg%s_sa_{seed}_100_{split}.pickle"
                % ("" if seed < 3 else "_v1"), seed, split),
            "metadata": lambda seed: (OLD_QBM if seed < 3 else NEW_ROOT / "fa_bm_vae") /
            ("opsd-wind_QBMVAE_2_anchor_sa_neg%s_sa_%d.json" % ("" if seed < 3 else "_v1", seed)),
            "checkpoint": lambda seed: (OLD_QBM if seed < 3 else NEW_ROOT / "fa_bm_vae") /
            ("opsd-wind_QBMVAE_2_anchor_sa_neg%s_sa_%d.pickle" % ("" if seed < 3 else "_v1", seed)),
        },
        "Forecast-anchored Gaussian": {
            "seeds": range(5),
            "path": lambda seed, split: _path(
                ROOT_DIR / "export" / "anchor_gaussian_opsd-wind" if seed < 3 else NEW_ROOT / "gaussian",
                "scenarios_opsd-wind_AnchorGaussian_{seed}_100_{split}.pickle", seed, split),
            "metadata": lambda seed: (ROOT_DIR / "export" / "anchor_gaussian_opsd-wind" if seed < 3 else NEW_ROOT / "gaussian") / ("opsd-wind_AnchorGaussian_%d.json" % seed),
            "checkpoint": lambda seed: (ROOT_DIR / "export" / "anchor_gaussian_opsd-wind" if seed < 3 else NEW_ROOT / "gaussian") / ("opsd-wind_AnchorGaussian_%d.pt" % seed),
        },
        "GMM-4": {
            "seeds": range(5),
            "path": lambda seed, split: _path(
                ROOT_DIR / "export" / "anchor_gmm4_opsd-wind" if seed < 3 else NEW_ROOT / "gmm4",
                "scenarios_opsd-wind_AnchorGMM4_{seed}_100_{split}.pickle", seed, split),
            "metadata": lambda seed: (ROOT_DIR / "export" / "anchor_gmm4_opsd-wind" if seed < 3 else NEW_ROOT / "gmm4") / ("opsd-wind_AnchorGMM4_%d.json" % seed),
            "checkpoint": lambda seed: (ROOT_DIR / "export" / "anchor_gmm4_opsd-wind" if seed < 3 else NEW_ROOT / "gmm4") / ("opsd-wind_AnchorGMM4_%d.pt" % seed),
        },
        "Score-SDE": {
            "seeds": range(5),
            "path": lambda seed, split: _path(
                ROOT_DIR / "export" / "anchor_score_sde_opsd-wind" if seed < 3 else NEW_ROOT / "score_sde",
                "scenarios_opsd-wind_AnchorScoreSDE_{seed}_100_{split}.pickle", seed, split),
            "metadata": lambda seed: (ROOT_DIR / "export" / "anchor_score_sde_opsd-wind" if seed < 3 else NEW_ROOT / "score_sde") / ("opsd-wind_AnchorScoreSDE_%d.json" % seed),
            "checkpoint": lambda seed: (ROOT_DIR / "export" / "anchor_score_sde_opsd-wind" if seed < 3 else NEW_ROOT / "score_sde") / ("opsd-wind_AnchorScoreSDE_%d.pt" % seed),
        },
        "J=0 ablation": {
            "seeds": range(3),
            "path": lambda seed, split: _path(NEW_ROOT / "j0", "scenarios_opsd-wind_QBMVAE_2_abl_j0_sa_neg_v1_sa_{seed}_100_{split}.pickle", seed, split),
            "metadata": lambda seed: NEW_ROOT / "j0" / ("opsd-wind_QBMVAE_2_abl_j0_sa_neg_v1_sa_%d.json" % seed),
            "checkpoint": lambda seed: NEW_ROOT / "j0" / ("opsd-wind_QBMVAE_2_abl_j0_sa_neg_v1_sa_%d.pickle" % seed),
        },
        "No anchor addition ablation (matched zero-mean init)": {
            "seeds": range(3),
            "path": lambda seed, split: _path(STRICT_NO_ANCHOR_ROOT, "scenarios_opsd-wind_QBMVAE_2_abl_no_anchor_add_zero_mean_sa_neg_v2_sa_{seed}_100_{split}.pickle", seed, split),
            "metadata": lambda seed: STRICT_NO_ANCHOR_ROOT / ("opsd-wind_QBMVAE_2_abl_no_anchor_add_zero_mean_sa_neg_v2_sa_%d.json" % seed),
            "checkpoint": lambda seed: STRICT_NO_ANCHOR_ROOT / ("opsd-wind_QBMVAE_2_abl_no_anchor_add_zero_mean_sa_neg_v2_sa_%d.pickle" % seed),
        },
        "Historical no-anchor ablation (initialization-confounded)": {
            "seeds": range(3),
            "path": lambda seed, split: _path(NEW_ROOT / "no_anchor_addition", "scenarios_opsd-wind_QBMVAE_2_abl_no_anchor_add_sa_neg_v1_sa_{seed}_100_{split}.pickle", seed, split),
            "metadata": lambda seed: NEW_ROOT / "no_anchor_addition" / ("opsd-wind_QBMVAE_2_abl_no_anchor_add_sa_neg_v1_sa_%d.json" % seed),
            "checkpoint": lambda seed: NEW_ROOT / "no_anchor_addition" / ("opsd-wind_QBMVAE_2_abl_no_anchor_add_sa_neg_v1_sa_%d.pickle" % seed),
        },
    }


def _dates(split):
    from GEFcom2014.external_datasets import load_daily_bundle
    _, _, date_sets = load_daily_bundle(DATASET)
    return [str(date) for date in date_sets[split.lower()]]


def _daily_metrics(scenarios, targets):
    values = dict(evaluate_model_daily(scenarios, targets, max_s=N_SCENARIOS))
    samples = scenarios.reshape(targets.shape[0], 24, N_SCENARIOS)
    values["mean_bias"] = (samples.mean(axis=2) - targets).mean(axis=1)
    return values


def _calibrate(validation, validation_targets, test):
    calibrator = HourlyAffineCalibrator(limits=(0.0, 1.0)).fit(validation, validation_targets.reshape(-1))
    return calibrator.transform(test), {
        "fit_split": "VS",
        "selection_metric": "validation_crps",
        "parameters": {
            "spread": calibrator.parameters.spread,
            "harmonics": calibrator.parameters.harmonics,
            "ridge": calibrator.parameters.ridge,
        },
    }


def _template(seed, split):
    return OLD_QBM / ("scenarios_opsd-wind_QBMVAE_2_tar1m_sa_%d_100_%s.pickle" % (seed, split))


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_state():
    def call(args):
        return subprocess.run(args, cwd=ROOT_DIR, capture_output=True, text=True, check=True).stdout.strip()
    try:
        status = call(["git", "status", "--short"])
        return {"commit": call(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "NOT_RECORDED", "dirty": "NOT_RECORDED", "status": "NOT_RECORDED"}


def _audit_entry(name, seed, spec):
    paths = {"metadata": spec["metadata"](seed), "checkpoint": spec["checkpoint"](seed)}
    paths.update({split: spec["path"](seed, split) for split in ("VS", "TEST")})
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("%s seed %d missing %s" % (name, seed, ", ".join(missing)))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return {
        "model": name, "seed": seed, "metadata": str(paths["metadata"].resolve()),
        "metadata_sha256": sha256_file(paths["metadata"]),
        "checkpoint": str(paths["checkpoint"].resolve()), "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "vs_scenario": str(paths["VS"].resolve()), "vs_scenario_sha256": sha256_file(paths["VS"]),
        "test_scenario": str(paths["TEST"].resolve()), "test_scenario_sha256": sha256_file(paths["TEST"]),
        "negative_phase_backend": metadata.get("negative_phase_backend", "NOT_APPLICABLE_OR_NOT_RECORDED"),
        "sampler": metadata.get("sampler", "NOT_RECORDED"),
        "checkpoint_selection": metadata.get("checkpoint_selection", "NOT_RECORDED"),
        "test_used_for_selection": metadata.get("test_used_for_selection", "NOT_RECORDED"),
    }


def _qbm_graph_audit(seed):
    path = OLD_QBM / ("opsd-wind_QBMVAE_2_anchor_sa_neg_sa_%d.pickle" % seed)
    with path.open("rb") as handle:
        model = pickle.load(handle)
    mask = model.graph_mask.detach().cpu().numpy().astype("<f4", copy=False)
    upper = np.triu(mask, k=1)
    return {
        "seed": seed,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "graph_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "nonzero_undirected_edges": int(np.count_nonzero(upper)),
        "upper_mask_weight_sum": float(upper.sum()),
    }


def _protocol_freeze_audit(audit_rows):
    historical = []
    for seed in range(3):
        entry = next(row for row in audit_rows if row["model"] == "FA-BM-VAE" and row["seed"] == seed)
        config = json.loads(Path(entry["metadata"]).read_text(encoding="utf-8"))
        historical.append({
            "seed": seed,
            "model_and_scenario_hashes": "direct_checkpoint_and_scenario_files",
            "negative_phase_backend": {"value": config.get("negative_phase_backend", "NOT_RECORDED"), "evidence": "direct_checkpoint_json"},
            "sampler": {"value": config.get("sampler", "NOT_RECORDED"), "evidence": "direct_checkpoint_json"},
            "checkpoint_selection": {
                "value": "minimum VS reconstruction loss",
                "evidence": "contemporaneous_training_code; selected_epoch_NOT_RECORDED_in_old_json",
                "test_used_for_selection": "NOT_RECORDED",
            },
            "graph": _qbm_graph_audit(seed),
            "historical_json_graph_edges": config.get("graph_edges", "NOT_RECORDED"),
        })
    masks = {row["graph"]["graph_mask_sha256"] for row in historical}
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "NOT_RECORDED"
    return {
        "protocol": "old fixed-window OPSD; SA negative phase; post-training SA generation",
        "dataset": str(DATASET.resolve()), "dataset_sha256": sha256_file(DATASET),
        "git": _git_state(),
        "environment": {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "pandas": version("pandas"), "scikit_learn": version("scikit-learn"),
        },
        "historical_sa_main_seeds_0_2": historical,
        "graph_json_discrepancy": {
            "old_json_value": 99,
            "direct_checkpoint_value": 172,
            "all_seed_mask_hashes_identical": len(masks) == 1,
            "interpretation": "The old JSON field is retained as historical metadata. Direct checkpoint graph masks are authoritative for the matched extension.",
        },
        "new_training_commands": {
            "fa_bm_vae_seeds_3_4": "D:\\anaconda\\envs\\wsy\\python.exe -u -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs/paper/fa_bm_vae_opsd_wind_sa_negative_v1.json --seed {3|4} --output-dir export/opsd_sa_negative_matched_v1/fa_bm_vae",
            "j0_seeds_0_2": "D:\\anaconda\\envs\\wsy\\python.exe -u -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs/paper/fa_bm_vae_opsd_wind_sa_negative_j0_v1.json --seed {0|1|2} --output-dir export/opsd_sa_negative_matched_v1/j0",
            "no_anchor_addition_seeds_0_2": "D:\\anaconda\\envs\\wsy\\python.exe -u -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs/paper/fa_bm_vae_opsd_wind_sa_negative_no_anchor_addition_v1.json --seed {0|1|2} --output-dir export/opsd_sa_negative_matched_v1/no_anchor_addition",
            "gaussian_seeds_3_4": "D:\\anaconda\\envs\\wsy\\python.exe -u -m GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline --tag opsd-wind --dataset-bundle GEFcom2014/data/external/opsd_wind_daily.npz --anchor-source context --epochs 80 --hidden-dim 256 --layers 2 --components 1 --batch-size 256 --learning-rate 0.0003 --weight-decay 0.00001 --patience 20 --min-delta 0.00001 --grad-clip 5 --n-scenarios 100 --seed {3|4} --output-dir export/opsd_sa_negative_matched_v1/gaussian",
            "gmm4_seeds_3_4": "D:\\anaconda\\envs\\wsy\\python.exe -u -m GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline --tag opsd-wind --dataset-bundle GEFcom2014/data/external/opsd_wind_daily.npz --anchor-source context --epochs 80 --hidden-dim 256 --layers 2 --components 4 --batch-size 256 --learning-rate 0.0003 --weight-decay 0.00001 --patience 20 --min-delta 0.00001 --grad-clip 5 --n-scenarios 100 --seed {3|4} --output-dir export/opsd_sa_negative_matched_v1/gmm4",
            "score_sde_seeds_3_4": "D:\\anaconda\\envs\\wsy\\python.exe -u -m GEFcom2014.models.QBM_VAE.anchor_score_sde --tag opsd-wind --dataset-bundle GEFcom2014/data/external/opsd_wind_daily.npz --anchor-source context --epochs 200 --batch-size 128 --channels 64 --blocks 4 --time-dim 64 --learning-rate 0.0002 --weight-decay 0.00001 --patience 25 --min-delta 0.0001 --grad-clip 5 --ema-decay 0.999 --sigma-data 0.5 --sigma-min 0.002 --sigma-max 1.0 --p-mean -1.2 --p-std 1.2 --rho 7 --sampling-steps 32 --n-scenarios 100 --seed {3|4} --device cuda --output-dir export/opsd_sa_negative_matched_v1/score_sde",
        },
        "historical_command_seeds_0_2": "NOT_RECORDED: no immutable command log was found; current defaults were not used to infer it.",
    }


def _ablation_parameter_audit(specs):
    rows = []
    for name in ("J=0 ablation", "No anchor addition ablation (matched zero-mean init)"):
        spec = specs[name]
        for seed in spec["seeds"]:
            with spec["checkpoint"](seed).open("rb") as handle:
                model = pickle.load(handle)
            raw = model.raw_j.detach().cpu().numpy()
            effective = model.qbm_params()[1].detach().cpu().numpy()
            rows.append({
                "model": name, "seed": seed,
                "checkpoint": str(spec["checkpoint"](seed).resolve()),
                "checkpoint_sha256": sha256_file(spec["checkpoint"](seed)),
                "graph_nonzero_undirected_edges": int(np.count_nonzero(np.triu(model.graph_mask.detach().cpu().numpy(), k=1))),
                "raw_j_absmax": float(np.abs(raw).max()),
                "effective_j_absmax": float(np.abs(effective).max()),
                "raw_j_requires_grad": bool(model.raw_j.requires_grad),
                "decoder_anchor": bool(getattr(model, "decoder_anchor", False)),
                "decoder_mean_zero_init": bool(getattr(model, "decoder_mean_zero_init", False)),
                "mean_head_zero_initialized": bool(np.array_equal(
                    model.dec[-1].weight[:model.in_size].detach().cpu().numpy(),
                    np.zeros_like(model.dec[-1].weight[:model.in_size].detach().cpu().numpy())
                ) and np.array_equal(
                    model.dec[-1].bias[:model.in_size].detach().cpu().numpy(),
                    np.zeros_like(model.dec[-1].bias[:model.in_size].detach().cpu().numpy())
                )),
                "verification": (
                    "fixed_zero_verified" if name == "J=0 ablation" and np.abs(raw).max() == 0.0
                    and np.abs(effective).max() == 0.0 and not model.raw_j.requires_grad else "not_applicable"
                ),
            })
    return rows


def _paired_rows(daily, models, seed_sets, repetitions):
    rows = []
    reference = "FA-BM-VAE"
    for model in models:
        if model == reference:
            continue
        common = sorted(set(seed_sets[reference]) & set(seed_sets[model]))
        if not common:
            continue
        for metric in METRICS:
            reference_values = np.mean([daily[(reference, seed, "Raw")][metric] for seed in common], axis=0)
            candidate_values = np.mean([daily[(model, seed, "Raw")][metric] for seed in common], axis=0)
            result = block_paired_inference(candidate_values - reference_values, block_length=7,
                                            repetitions=repetitions, seed=2026 + len(rows))
            rows.append({"model_a": model, "model_b": reference, "variant": "Raw", "metric": metric,
                         "comparison": "model_a_minus_FA-BM-VAE", "seeds": ",".join(map(str, common)),
                         "statistical_unit": "forecast date; scenarios remain within-date ensemble members", **result})
    return rows


def evaluate(output_dir=OUTPUT, repetitions=5000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {split: load_targets("opsd-wind", split, DATASET) for split in ("VS", "TEST")}
    dates = {split: _dates(split) for split in targets}
    specs = scenario_specs()
    audit, daily_rows, summaries, daily = [], [], [], {}
    unavailable_ecc = []
    for name, spec in specs.items():
        for seed in spec["seeds"]:
            audit.append(_audit_entry(name, seed, spec))
            vs = load_scenarios(spec["path"](seed, "VS"), targets["VS"].size)
            test = load_scenarios(spec["path"](seed, "TEST"), targets["TEST"].size)
            if vs.shape[1] != N_SCENARIOS or test.shape[1] != N_SCENARIOS:
                raise ValueError("%s seed %d does not contain exactly 100 scenarios" % (name, seed))
            variants = {"Raw": test}
            cal, cal_audit = _calibrate(vs, targets["VS"], test)
            variants["Cal"] = cal
            template_vs, template_test = _template(seed, "VS"), _template(seed, "TEST")
            if template_vs.is_file() and template_test.is_file():
                # The template is a frozen scenario-only rank template; no TEST y is read here.
                variants["Cal+ECC"] = ensemble_copula_coupling(
                    cal, load_scenarios(template_test, targets["TEST"].size))
            else:
                unavailable_ecc.append({"model": name, "seed": seed, "reason": "same-seed frozen TAR1M template unavailable"})
            for variant, scenarios in variants.items():
                metric_values = _daily_metrics(scenarios, targets["TEST"])
                daily[(name, seed, variant)] = metric_values
                for day, date in enumerate(dates["TEST"]):
                    daily_rows.append({"model": name, "seed": seed, "variant": variant, "date": date,
                                       "n_scenarios": N_SCENARIOS,
                                       "statistical_unit": "forecast date; scenarios are ensemble members, not independent dates",
                                       **{metric: float(metric_values[metric][day]) for metric in METRICS}})
                for metric in METRICS:
                    summaries.append({"model": name, "seed": seed, "variant": variant, "metric": metric,
                                      "mean": float(np.mean(metric_values[metric])), "n_dates": len(dates["TEST"]),
                                      "n_scenarios": N_SCENARIOS})
            audit[-1]["calibration"] = cal_audit
    grouped = []
    for name, spec in specs.items():
        for variant in ("Raw", "Cal", "Cal+ECC"):
            for metric in METRICS:
                values = [row["mean"] for row in summaries if row["model"] == name and row["variant"] == variant and row["metric"] == metric]
                if values:
                    grouped.append({"model": name, "variant": variant, "metric": metric, "n_seeds": len(values),
                                    "mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                    "statistical_unit": "seed means over 90 forecast dates"})
    core = ("FA-BM-VAE", "Forecast-anchored Gaussian", "GMM-4", "Score-SDE", "J=0 ablation", "No anchor addition ablation (matched zero-mean init)")
    seed_sets = {name: tuple(specs[name]["seeds"]) for name in core}
    paired = _paired_rows(daily, core, seed_sets, repetitions)
    _write_csv(output_dir / "per_date_metrics.csv", daily_rows)
    _write_csv(output_dir / "per_seed_metrics.csv", summaries)
    _write_csv(output_dir / "summary.csv", grouped)
    _write_csv(output_dir / "raw_paired_date_statistics.csv", paired)
    (output_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output_dir / "protocol_freeze_audit.json").write_text(
        json.dumps(_protocol_freeze_audit(audit), indent=2), encoding="utf-8"
    )
    (output_dir / "ablation_parameter_audit.json").write_text(
        json.dumps(_ablation_parameter_audit(specs), indent=2), encoding="utf-8"
    )
    manifest = {
        "artifact_role": "supplementary_fixed_window_opsd_evaluation_strict_no_anchor_v2",
        "eligible_for_paper": False,
        "synthetic_data": False,
        "hardware_claim": False,
        "physical_platform_used": False,
        "raw_is_primary": True,
        "historical_no_anchor_ablation": "retained separately as initialization-confounded and excluded from the matched main-table contrast",
        "calibration_fit_split": "VS",
        "ecc_template_split": "frozen scenario template; no targets used",
        "cal_ecc_status": "partial: only seeds with same-seed frozen TAR1M template",
        "cal_ecc_unavailable": unavailable_ecc,
        "metrics": list(METRICS),
        "ramp_metric": "trajectory ensemble CRPS of 23 within-day ramp values; not historical ramp quantile MAE",
        "n_test_dates": len(dates["TEST"]), "n_scenarios_per_date": N_SCENARIOS,
        "statistical_unit": "forecast date, not scenario draw or date-by-seed pseudo-replicate",
        "block_statistics": {"block_length_days": 7, "bootstrap_repetitions": repetitions},
        "dataset": str(DATASET.resolve()), "dataset_sha256": sha256_file(DATASET),
        "git": _git_state(),
        "python": sys.version, "platform": platform.platform(),
        "source_sha256": sha256_file(__file__),
        "command": "python -m GEFcom2014.forecast_quality.opsd_sa_negative_evaluation",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    print(evaluate(args.output_dir, args.bootstrap_repetitions))


if __name__ == "__main__":
    main()
