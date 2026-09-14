# -*- coding: utf-8 -*-

"""Validation-only calibration and temporal coupling for scenario models.

The module applies one predeclared post-processing protocol to every model:
Raw, Cal, TRC, and Cal+TRC. Calibration parameters are fitted exclusively on
the validation split and then transferred unchanged to the test split.
"""

import argparse
import csv
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.compare_scenarios import (
    BASELINE_SCENARIOS_DIR,
    ROOT_DIR,
    crps_per_hour_fast,
    evaluate_model,
    load_scenarios,
)
from GEFcom2014.forecast_quality.temporal_rank_coupling import (
    grouped_temporal_rank_coupling,
    load_raw_track,
    load_templates,
)
from GEFcom2014.external_datasets import load_daily_bundle


TRACK_GROUPS = {"wind": 10, "pv": 3, "load": 1, "opsd-wind": 1}
TRACK_LIMITS = {tag: (0.0, 1.0) for tag in TRACK_GROUPS}
VARIANTS = ("Raw", "Cal", "TRC", "Cal+TRC")


@dataclass(frozen=True)
class CalibrationParameters:
    spread: float
    hourly_bias: list
    zero_hours: list
    lower: float
    upper: float
    harmonics: int
    ridge: float
    selection_metric: str = "validation_crps"


@dataclass(frozen=True)
class ScenarioPair:
    name: str
    validation_path: Path
    test_path: Path


class HourlyAffineCalibrator:
    """Low-dimensional location/spread calibration for ensemble scenarios."""

    def __init__(self, spread_grid=None, harmonics=3, ridge=1.0, limits=(0.0, 1.0)):
        if spread_grid is None:
            spread_grid = np.linspace(0.5, 2.0, 31)
        self.spread_grid = np.asarray(spread_grid, dtype=np.float64)
        self.harmonics = int(harmonics)
        self.ridge = float(ridge)
        self.limits = tuple(float(value) for value in limits)
        self.parameters = None
        self.validation_candidates = None

    def fit(self, validation_scenarios, validation_targets):
        scenarios, targets = _validate_scenarios_targets(validation_scenarios, validation_targets)
        location = np.median(scenarios, axis=1)
        residual = (targets - location).reshape(-1, 24)
        hourly_residual = residual.mean(axis=0)
        target_by_day = targets.reshape(-1, 24)
        zero_hours = np.flatnonzero(np.max(np.abs(target_by_day), axis=0) <= 1e-12)
        active_hours = np.setdiff1d(np.arange(24), zero_hours)

        design = _fourier_design(self.harmonics)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.ridge
        penalty[0, 0] = 0.0
        active_design = design[active_hours]
        coefficients = np.linalg.solve(
            active_design.T @ active_design + penalty,
            active_design.T @ hourly_residual[active_hours],
        )
        hourly_bias = design @ coefficients
        hourly_bias[zero_hours] = 0.0

        candidates = []
        best = None
        for spread in self.spread_grid:
            calibrated = self._apply_values(scenarios, hourly_bias, float(spread), zero_hours)
            score = float(crps_per_hour_fast(calibrated, targets).mean())
            row = {"spread": float(spread), "validation_crps": score}
            candidates.append(row)
            if best is None or (score, abs(float(spread) - 1.0)) < (
                best["validation_crps"],
                abs(best["spread"] - 1.0),
            ):
                best = row

        self.parameters = CalibrationParameters(
            spread=best["spread"],
            hourly_bias=hourly_bias.tolist(),
            zero_hours=zero_hours.astype(int).tolist(),
            lower=self.limits[0],
            upper=self.limits[1],
            harmonics=self.harmonics,
            ridge=self.ridge,
        )
        self.validation_candidates = candidates
        return self

    def transform(self, scenarios):
        if self.parameters is None:
            raise RuntimeError("fit must be called before transform")
        values = np.asarray(scenarios, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] % 24 != 0:
            raise ValueError("scenarios must have shape (n_days * 24, n_scenarios)")
        return self._apply_values(
            values,
            np.asarray(self.parameters.hourly_bias),
            self.parameters.spread,
            np.asarray(self.parameters.zero_hours, dtype=int),
        )

    def _apply_values(self, scenarios, hourly_bias, spread, zero_hours):
        location = np.median(scenarios, axis=1, keepdims=True)
        period_bias = np.tile(hourly_bias, scenarios.shape[0] // 24)[:, None]
        calibrated = location + period_bias + spread * (scenarios - location)
        calibrated = np.clip(calibrated, self.limits[0], self.limits[1])
        if zero_hours.size:
            rows = np.tile(np.isin(np.arange(24), zero_hours), scenarios.shape[0] // 24)
            calibrated[rows] = 0.0
        return calibrated


def run_pair(tag, pair, output_root, spread_grid, harmonics, ridge, trc_seed,
             dataset_bundle=None):
    validation_targets = load_targets(tag, "VS", dataset_bundle)
    test_targets = load_targets(tag, "TEST", dataset_bundle)
    validation_periods = validation_targets.size
    test_periods = test_targets.size
    validation_raw = load_scenarios(pair.validation_path, validation_periods)
    test_raw = load_scenarios(pair.test_path, test_periods)

    calibrator = HourlyAffineCalibrator(
        spread_grid=spread_grid,
        harmonics=harmonics,
        ridge=ridge,
        limits=TRACK_LIMITS[tag],
    ).fit(validation_raw, validation_targets.reshape(-1))
    validation_cal = calibrator.transform(validation_raw)
    test_cal = calibrator.transform(test_raw)

    if dataset_bundle is None:
        templates = load_templates(tag, split="LS")
    else:
        bundle_arrays, _, _ = load_daily_bundle(dataset_bundle)
        templates = bundle_arrays["y_ls"]
    trc_kwargs = {
        "template_matrix": templates,
        "seed": trc_seed,
        "n_groups": TRACK_GROUPS[tag],
    }
    validation_variants = {
        "Raw": validation_raw,
        "Cal": validation_cal,
        "TRC": grouped_temporal_rank_coupling(validation_raw, **trc_kwargs),
        "Cal+TRC": grouped_temporal_rank_coupling(validation_cal, **trc_kwargs),
    }
    test_variants = {
        "Raw": test_raw,
        "Cal": test_cal,
        "TRC": grouped_temporal_rank_coupling(test_raw, **trc_kwargs),
        "Cal+TRC": grouped_temporal_rank_coupling(test_cal, **trc_kwargs),
    }

    model_dir = output_root / tag / _slug(pair.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, variants, targets in (
        ("VS", validation_variants, validation_targets),
        ("TEST", test_variants, test_targets),
    ):
        for variant in VARIANTS:
            scenario_path = model_dir / ("scenarios_%s_%s.pickle" % (_slug(variant), split))
            _dump_pickle(scenario_path, variants[variant])
            metrics = evaluate_model(variants[variant], targets, tag)
            rows.append(_metric_row(tag, pair.name, variant, split, scenario_path, metrics))

    audit = {
        "track": tag,
        "model": pair.name,
        "validation_source": str(pair.validation_path.resolve()),
        "test_source": str(pair.test_path.resolve()),
        "fit_split": "VS",
        "evaluation_split": "TEST",
        "calibration": asdict(calibrator.parameters),
        "spread_candidates": calibrator.validation_candidates,
        "trc": {
            "template_split": "LS",
            "seed": int(trc_seed),
            "n_groups": TRACK_GROUPS[tag],
            "preserves_hourly_marginals": True,
        },
    }
    with (model_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    _write_metrics(model_dir / "metrics.csv", rows)
    return rows, audit


def discover_default_pairs(tag):
    nf_ids = {"wind": 1, "pv": 3, "load": 1}
    baseline_specs = {}
    if tag in nf_ids:
        baseline_specs = {
            "CVAE": BASELINE_SCENARIOS_DIR / "vae" / ("scenarios_%s_VAElinear_1_0_100_{split}.pickle" % tag),
            "UMNN-NF": BASELINE_SCENARIOS_DIR / "nfs" /
            ("scenarios_%s_UMNN_M_%s_0_100_{split}.pickle" % (tag, nf_ids[tag])),
            "WGAN-GP": BASELINE_SCENARIOS_DIR / "gan" /
            ("scenarios_%s_GAN_wasserstein_1_0_100_{split}.pickle" % tag),
        }

    pairs = []
    missing = []
    for name, pattern in baseline_specs.items():
        validation_path = Path(str(pattern).format(split="VS"))
        test_path = Path(str(pattern).format(split="TEST"))
        if validation_path.is_file() and test_path.is_file():
            pairs.append(ScenarioPair(name, validation_path, test_path))
        else:
            missing.append({
                "model": name,
                "validation_path": str(validation_path),
                "test_path": str(test_path),
                "missing_validation": not validation_path.is_file(),
                "missing_test": not test_path.is_file(),
            })

    generated_families = (
        (
            "BM-VAE ablation (no anchor, Temporal-MI J)",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_abl_noanchor_mi_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Independent-prior VAE ablation (no anchor, J=0)",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_abl_noanchor_ind_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Independent-prior VAE ablation (J=0)",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_abl_(?:lanchor|anchor)_ind_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Same-edge Random-graph BM-VAE ablation",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_abl_(?:lanchor|anchor)_randw_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "QBM-VAE",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2(?:_pv64)?_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "QBM-VAE + Forecast Anchor",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_anchor_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "QBM-VAE + Learned Forecast Anchor",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_lanchor_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "FA-BM-VAE + Learned Forecast Anchor (SA negative phase)",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_lanchor_sa_neg_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "FA-BM-VAE + Forecast Anchor (SA negative phase)",
            ROOT_DIR / "export" / ("qbm_vae_%s" % tag),
            r"scenarios_%s_QBMVAE_2_anchor_sa_neg_sa_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Spline Conditional NF",
            ROOT_DIR / "export" / ("spline_nf_%s" % tag),
            r"scenarios_%s_SplineCNF_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Conditional DDPM",
            ROOT_DIR / "export" / ("ddpm_%s" % tag),
            r"scenarios_%s_ConditionalDDPM_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "D3U (NWP-adapted)",
            ROOT_DIR / "export" / ("d3u_%s" % tag),
            r"scenarios_%s_D3UAdapted_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Treeffuser",
            ROOT_DIR / "export" / ("treeffuser_%s" % tag),
            r"scenarios_%s_Treeffuser_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Daily Residual Bootstrap",
            ROOT_DIR / "export" / ("residual_bootstrap_%s" % tag),
            r"scenarios_%s_ResidualBootstrap_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Conditional Gaussian",
            ROOT_DIR / "export" / ("anchor_gaussian_%s" % tag),
            r"scenarios_%s_AnchorGaussian_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Gaussian Mixture (K=4)",
            ROOT_DIR / "export" / ("anchor_gmm4_%s" % tag),
            r"scenarios_%s_AnchorGMM4_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Conditional Spline Flow",
            ROOT_DIR / "export" / ("anchor_spline_flow_%s" % tag),
            r"scenarios_%s_AnchorSplineFlow_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Conditional Score-SDE",
            ROOT_DIR / "export" / ("anchor_score_sde_%s" % tag),
            r"scenarios_%s_AnchorScoreSDE_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "QBM-VAE + Temporal ECC",
            ROOT_DIR / "export" / ("qbm_ecc_%s" % tag),
            r"scenarios_%s_QBMECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "QBM-VAE + Forecast Anchor + Temporal ECC",
            ROOT_DIR / "export" / ("qbm_ecc_%s" % tag),
            r"scenarios_%s_QBMAnchorECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "QBM-VAE + Learned Forecast Anchor + Temporal ECC",
            ROOT_DIR / "export" / ("qbm_ecc_%s" % tag),
            r"scenarios_%s_QBMLearnedAnchorECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "FA-BM-VAE + Learned Forecast Anchor (SA negative phase) + Temporal ECC",
            ROOT_DIR / "export" / ("qbm_ecc_%s" % tag),
            r"scenarios_%s_QBMLearnedAnchorSANegECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "FA-BM-VAE + Forecast Anchor (SA negative phase) + Temporal ECC",
            ROOT_DIR / "export" / ("qbm_ecc_%s" % tag),
            r"scenarios_%s_QBMAnchorSANegECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Conditional Gaussian + Temporal ECC",
            ROOT_DIR / "export" / ("anchor_gaussian_ecc_%s" % tag),
            r"scenarios_%s_AnchorGaussianECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Gaussian Mixture (K=4) + Temporal ECC",
            ROOT_DIR / "export" / ("anchor_gmm4_ecc_%s" % tag),
            r"scenarios_%s_AnchorGMM4ECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Conditional Spline Flow + Temporal ECC",
            ROOT_DIR / "export" / ("anchor_spline_flow_ecc_%s" % tag),
            r"scenarios_%s_AnchorSplineFlowECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Forecast-Anchored Conditional Score-SDE + Temporal ECC",
            ROOT_DIR / "export" / ("anchor_score_sde_ecc_%s" % tag),
            r"scenarios_%s_AnchorScoreSDEECC_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Legacy CVAE (VS retrained)",
            ROOT_DIR / "export" / ("legacy_cvae_%s" % tag),
            r"scenarios_%s_LegacyCVAE_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Legacy WGAN-GP (VS retrained)",
            ROOT_DIR / "export" / ("legacy_wgan_gp_%s" % tag),
            r"scenarios_%s_LegacyWGANGP_(\d+)_100_VS\.pickle" % tag,
        ),
        (
            "Legacy UMNN (VS retrained)",
            ROOT_DIR / "export" / ("legacy_umnn_%s" % tag),
            r"scenarios_%s_LegacyUMNN_(\d+)_100_VS\.pickle" % tag,
        ),
    )
    for family, directory, pattern in generated_families:
        expression = re.compile(pattern)
        for validation_path in sorted(directory.glob("*_100_VS.pickle")):
            match = expression.fullmatch(validation_path.name)
            if match is None:
                continue
            test_path = validation_path.with_name(validation_path.name.replace("_VS.pickle", "_TEST.pickle"))
            seed = int(match.group(1))
            if test_path.is_file():
                pairs.append(ScenarioPair("%s (seed %d)" % (family, seed), validation_path, test_path))
            else:
                missing.append({
                    "model": "%s (seed %d)" % (family, seed),
                    "validation_path": str(validation_path),
                    "test_path": str(test_path),
                    "missing_validation": False,
                    "missing_test": True,
                })
    return pairs, missing


def load_targets(tag, split, dataset_bundle=None):
    if dataset_bundle is not None:
        arrays, metadata, _ = load_daily_bundle(dataset_bundle)
        expected_name = metadata.get("dataset_name")
        if expected_name and expected_name != tag:
            raise ValueError("Bundle dataset_name %s does not match tag %s" % (expected_name, tag))
        return arrays[{"VS": "y_vs", "TEST": "y_test"}[split]].astype(np.float64)
    data, indices = load_raw_track(tag)
    split_index = {"VS": 3, "TEST": 5}[split]
    targets = data[split_index].copy()
    if tag == "pv":
        non_null = list(np.delete(np.arange(24), indices))
        targets.columns = non_null
        for index in indices:
            targets[index] = 0.0
        targets = targets.sort_index(axis=1)
    return targets.values.astype(np.float64)


def _fourier_design(harmonics):
    hours = np.arange(24, dtype=np.float64)
    columns = [np.ones(24, dtype=np.float64)]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * np.pi * harmonic * hours / 24.0
        columns.extend((np.cos(angle), np.sin(angle)))
    return np.column_stack(columns)


def _validate_scenarios_targets(scenarios, targets):
    scenarios = np.asarray(scenarios, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if scenarios.ndim != 2 or scenarios.shape[0] != targets.size:
        raise ValueError("scenario rows must equal the number of target periods")
    if scenarios.shape[0] % 24 != 0:
        raise ValueError("target periods must be divisible by 24")
    if not np.isfinite(scenarios).all() or not np.isfinite(targets).all():
        raise ValueError("scenarios and targets must be finite")
    return scenarios, targets


def _metric_row(tag, model, variant, split, scenario_path, metrics):
    return {
        "track": tag,
        "model": model,
        "variant": variant,
        "split": split,
        "n_scenarios": metrics["n_scenarios"],
        "mean_qs": metrics["plf_mean"],
        "mean_crps": metrics["crps_mean"],
        "reliability_mae": metrics["reliability_mae"],
        "energy_score": metrics["energy_score"],
        "variogram_score": metrics["variogram_score"],
        "corr_mae": metrics["corr_mae"],
        "ramp_quantile_mae": metrics["ramp_quantile_mae"],
        "scenario_file": str(scenario_path.resolve()),
    }


def _write_metrics(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dump_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply validation-only calibration and grouped TRC uniformly to scenario models."
    )
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--vs-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "export" / "unified_postprocessing")
    parser.add_argument("--spread-grid", type=float, nargs="+", default=np.linspace(0.5, 2.0, 31).tolist())
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--trc-seed", type=int, default=0)
    parser.add_argument(
        "--reuse-completed", action="store_true",
        help="Reuse per-model metrics/audit files and process only missing model pairs.",
    )
    args = parser.parse_args()
    supplied = (args.model_name is not None, args.vs_path is not None, args.test_path is not None)
    if any(supplied) and not all(supplied):
        parser.error("--model-name, --vs-path, and --test-path must be supplied together")
    if args.harmonics < 0 or args.harmonics > 11:
        parser.error("--harmonics must be between 0 and 11")
    if args.ridge < 0 or any(value <= 0 for value in args.spread_grid):
        parser.error("ridge must be non-negative and spread values must be positive")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model_name is not None:
        pairs = [ScenarioPair(args.model_name, args.vs_path, args.test_path)]
        missing = []
    else:
        pairs, missing = discover_default_pairs(args.tag)
    if not pairs:
        raise FileNotFoundError("No complete VS/TEST scenario pair was found; regenerate VS scenarios first")

    all_rows = []
    audits = []
    for pair in pairs:
        model_dir = args.output_dir / args.tag / _slug(pair.name)
        model_metrics = model_dir / "metrics.csv"
        model_audit = model_dir / "audit.json"
        if args.reuse_completed and model_metrics.is_file() and model_audit.is_file():
            print("Reusing completed %s on %s" % (pair.name, args.tag))
            with model_metrics.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with model_audit.open("r", encoding="utf-8") as handle:
                audit = json.load(handle)
        else:
            print("Processing %s on %s" % (pair.name, args.tag))
            rows, audit = run_pair(
                tag=args.tag,
                pair=pair,
                output_root=args.output_dir,
                spread_grid=args.spread_grid,
                harmonics=args.harmonics,
                ridge=args.ridge,
                trc_seed=args.trc_seed,
                dataset_bundle=args.dataset_bundle,
            )
        all_rows.extend(rows)
        audits.append(audit)

    track_dir = args.output_dir / args.tag
    track_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(track_dir / "metrics.csv", all_rows)
    manifest = {
        "track": args.tag,
        "dataset_bundle": str(args.dataset_bundle.resolve()) if args.dataset_bundle else None,
        "protocol": ["Raw", "Cal", "TRC", "Cal+TRC"],
        "calibration_fit_split": "VS",
        "trc_template_split": "LS",
        "processed_models": [pair.name for pair in pairs],
        "missing_default_models": missing,
        "audits": audits,
    }
    with (track_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Wrote unified results to %s" % track_dir)
    if missing:
        print("Skipped %s default model(s) without paired VS/TEST scenarios" % len(missing))


if __name__ == "__main__":
    main()
