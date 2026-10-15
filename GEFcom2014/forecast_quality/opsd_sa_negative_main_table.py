# -*- coding: utf-8 -*-
"""Build the fixed seed 0--2 OPSD main-table candidate from frozen metrics."""

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.aggregate_wind_five_seed import (
    block_paired_inference,
    holm_adjust,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
OLD_QBM_DIR = ROOT_DIR / "export" / "qbm_vae_opsd-wind"
SOURCE_DIR = ROOT_DIR / "export" / "opsd_sa_negative_matched_v1" / "evaluation_strict_no_anchor_v2"
OUTPUT_DIR = ROOT_DIR / "export" / "opsd_sa_negative_matched_v1" / "main_table_seed012_strict_no_anchor_v2"
SEEDS = (0, 1, 2)
VARIANTS = ("Raw", "Cal", "Cal+ECC")
METRICS = ("CRPS_raw", "Energy", "Variogram", "ramp_CRPS", "PICP90", "interval_width90", "MAQCE", "mean_bias")
CORE_MODELS = ("FA-BM-VAE", "Forecast-anchored Gaussian", "GMM-4", "Score-SDE")
ABLATIONS = ("J=0 ablation", "No anchor addition ablation (matched zero-mean init)")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_rows(rows):
    valid_models = set(CORE_MODELS + ABLATIONS)
    selected = []
    for row in rows:
        if (row["model"] in valid_models and int(row["seed"]) in SEEDS
                and row["variant"] in VARIANTS):
            selected.append(row)
    return selected


def validate_complete_cells(rows, expected_dates=90):
    """Reject incomplete seed/model/variant cells instead of using an intersection."""
    dates = {}
    for row in rows:
        key = (row["model"], int(row["seed"]), row["variant"])
        dates.setdefault(key, []).append(row["date"])
        if int(row["n_scenarios"]) != 100:
            raise ValueError("Main table requires exactly 100 scenarios per forecast date")
    expected = {(model, seed, variant) for model in CORE_MODELS + ABLATIONS
                for seed in SEEDS for variant in VARIANTS}
    if set(dates) != expected:
        missing = sorted(expected - set(dates))
        extra = sorted(set(dates) - expected)
        raise ValueError("Incomplete main-table cells; missing=%s extra=%s" % (missing, extra))
    reference_dates = None
    for key, values in dates.items():
        if len(values) != expected_dates or len(set(values)) != expected_dates:
            raise ValueError("%s must have %d unique TEST dates" % (key, expected_dates))
        ordered = tuple(sorted(values))
        if reference_dates is None:
            reference_dates = ordered
        elif ordered != reference_dates:
            raise ValueError("Main-table cells do not have an identical TEST date set")
    return reference_dates


def _daily_arrays(rows, dates):
    indexed = {}
    for row in rows:
        indexed[(row["model"], int(row["seed"]), row["variant"], row["date"])] = row
    arrays = {}
    for model in CORE_MODELS + ABLATIONS:
        for seed in SEEDS:
            for variant in VARIANTS:
                for metric in METRICS:
                    arrays[(model, seed, variant, metric)] = np.asarray([
                        float(indexed[(model, seed, variant, date)][metric])
                        for date in dates
                    ], dtype=float)
    return arrays


def _per_seed_and_summary(arrays):
    per_seed, summary = [], []
    for model in CORE_MODELS + ABLATIONS:
        for variant in VARIANTS:
            for metric in METRICS:
                values = []
                for seed in SEEDS:
                    mean = float(arrays[(model, seed, variant, metric)].mean())
                    values.append(mean)
                    per_seed.append({"model": model, "seed": seed, "variant": variant,
                                     "metric": metric, "mean": mean, "n_test_dates": 90,
                                     "n_scenarios_per_date": 100})
                summary.append({"model": model, "variant": variant, "metric": metric,
                                "n_seeds": len(values), "mean": float(np.mean(values)),
                                "seed_std": float(np.std(values, ddof=1)),
                                "statistical_unit": "seed means over 90 forecast dates"})
    return per_seed, summary


def _paired_rows(arrays, models, family, repetitions):
    output = []
    for variant in VARIANTS:
        for metric in METRICS:
            reference = np.mean([arrays[("FA-BM-VAE", seed, variant, metric)] for seed in SEEDS], axis=0)
            for model in models:
                candidate = np.mean([arrays[(model, seed, variant, metric)] for seed in SEEDS], axis=0)
                result = block_paired_inference(candidate - reference, block_length=7,
                                                repetitions=repetitions, seed=2026 + len(output))
                output.append({"family": family, "model_a": model, "model_b": "FA-BM-VAE",
                               "variant": variant, "metric": metric,
                               "comparison": "model_a_minus_FA-BM-VAE", "seeds": "0,1,2",
                               "statistical_unit": "forecast date block; scenario draws are not independent dates",
                               **result})
    for variant in VARIANTS:
        for metric in METRICS:
            indices = [index for index, row in enumerate(output)
                       if row["variant"] == variant and row["metric"] == metric]
            adjusted = holm_adjust([output[index]["block_sign_flip_p"] for index in indices])
            for index, p_value in zip(indices, adjusted):
                output[index]["holm_p"] = p_value
                output[index]["holm_family"] = "%s/%s/%s" % (family, variant, metric)
    return output


def provenance_acceptance(source_dir):
    audit = json.loads((Path(source_dir) / "artifact_audit.json").read_text(encoding="utf-8"))
    rows = [row for row in audit if row["model"] in CORE_MODELS + ABLATIONS and int(row["seed"]) in SEEDS]
    missing = []
    for row in rows:
        for key in ("metadata_sha256", "checkpoint_sha256", "vs_scenario_sha256", "test_scenario_sha256"):
            if not row.get(key):
                missing.append({"model": row["model"], "seed": row["seed"], "field": key})
    fa_historical = [row for row in rows if row["model"] == "FA-BM-VAE"]
    return {
        "artifact_role": "main_table_provenance_acceptance",
        "main_table_seeds": list(SEEDS),
        "direct_file_hashes_complete": not missing,
        "missing_hash_fields": missing,
        "fa_bm_negative_phase_backend": sorted(set(row["negative_phase_backend"] for row in fa_historical)),
        "fa_bm_generation_sampler": sorted(set(row["sampler"] for row in fa_historical)),
        "checkpoint_selection_evidence": "contemporaneous VS reconstruction rule; historical selected epoch not recorded",
        "test_selection_evidence": "NOT_RECORDED in historical FA-BM-VAE metadata; do not infer false from current defaults",
        "acceptance": "LIMITED_ACCEPTANCE",
        "paper_eligibility": False,
        "reason": "Hashes and direct SA backend evidence are complete, but the old FA-BM-VAE metadata lacks a direct TEST-selection field and immutable original command log.",
        "historical_evidence_table": historical_evidence_table(),
    }


def _git_state():
    try:
        status = subprocess.run(["git", "status", "--short"], cwd=ROOT_DIR,
                                capture_output=True, text=True, check=True).stdout.strip()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR,
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "NOT_RECORDED", "dirty": "NOT_RECORDED", "status": "NOT_RECORDED"}


def historical_evidence_table():
    """Distinguish direct legacy artifacts from source-based inferences."""
    artifacts, graph_edges, graph_hashes, graph_sums, loss_hashes = [], [], [], [], []
    for seed in SEEDS:
        stem = "opsd-wind_QBMVAE_2_anchor_sa_neg_sa_%d" % seed
        metadata = OLD_QBM_DIR / (stem + ".json")
        checkpoint = OLD_QBM_DIR / (stem + ".pickle")
        loss = OLD_QBM_DIR / ("loss_" + stem + ".pickle")
        test = OLD_QBM_DIR / ("scenarios_" + stem + "_100_TEST.pickle")
        validation = OLD_QBM_DIR / ("scenarios_" + stem + "_100_VS.pickle")
        config = json.loads(metadata.read_text(encoding="utf-8"))
        with checkpoint.open("rb") as handle:
            model = pickle.load(handle)
        with loss.open("rb") as handle:
            loss_history = pickle.load(handle)
        mask = model.graph_mask.detach().cpu().numpy().astype("<f4", copy=False)
        upper = np.triu(mask, k=1)
        graph_edges.append(int(np.count_nonzero(upper)))
        graph_hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
        graph_sums.append(float(upper.sum()))
        loss_hashes.append({"seed": seed, "path": str(loss.resolve()), "sha256": sha256_file(loss),
                            "history_length": len(loss_history)})
        artifacts.append({
            "seed": seed,
            "metadata_sha256": sha256_file(metadata), "checkpoint_sha256": sha256_file(checkpoint),
            "vs_scenarios_sha256": sha256_file(validation), "test_scenarios_sha256": sha256_file(test),
            "negative_phase_backend": config.get("negative_phase_backend", "NOT_RECORDED"),
            "generation_sampler": config.get("sampler", "NOT_RECORDED"),
            "json_graph_edges": config.get("graph_edges", "NOT_RECORDED"),
        })
    return {
        "evidence_levels": {
            "direct": "direct immutable artifact evidence",
            "indirect": "contemporaneous source or protocol support, not an immutable record of the historical run",
            "unrecoverable": "no artifact found; no inference made",
        },
        "items": [
            {"item": "checkpoint/configuration/VS and TEST scenario artifacts", "level": "direct", "evidence": artifacts},
            {"item": "negative-phase backend", "level": "direct", "evidence": "all historical JSON files record sa"},
            {"item": "post-training generation sampler", "level": "direct", "evidence": "all historical JSON files record sa"},
            {"item": "post-training generation RNG seed", "level": "unrecoverable", "evidence": "historical JSON records sampler=sa but has no independent generation RNG seed; it is therefore not inferred from the training-seed filename"},
            {"item": "loss files", "level": "direct", "evidence": loss_hashes},
            {"item": "checkpoint selection rule", "level": "indirect", "evidence": "contemporaneous fit_qbm_vae source uses VS reconstruction; historical selected epoch is absent"},
            {"item": "TEST use for historical checkpoint selection", "level": "unrecoverable", "evidence": "NOT_RECORDED; current defaults and later configurations were not used to infer false"},
            {"item": "original commands and exact historical source commit", "level": "unrecoverable", "evidence": "no immutable command log or source revision record found"},
            {"item": "graph-edge discrepancy", "level": "direct", "evidence": {
                "historical_json_graph_edges": sorted(set(item["json_graph_edges"] for item in artifacts)),
                "checkpoint_nonzero_undirected_edges": sorted(set(graph_edges)),
                "checkpoint_mask_sha256": sorted(set(graph_hashes)),
                "checkpoint_upper_mask_weight_sum": sorted(set(graph_sums)),
                "interpretation": "checkpoint masks are direct model state and authoritative for matched extensions; the cause of the stale JSON count is unrecoverable",
            }},
        ],
        "current_audit_context": {
            "git": _git_state(),
            "main_table_source_sha256": sha256_file(__file__),
            "historical_training_source_version": "NOT_RECORDED",
        },
    }


def run(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, repetitions=10000):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source = source_dir / "per_date_metrics.csv"
    rows = _select_rows(read_csv(source))
    dates = validate_complete_cells(rows)
    arrays = _daily_arrays(rows, dates)
    per_seed, summary = _per_seed_and_summary(arrays)
    paired = _paired_rows(arrays, CORE_MODELS[1:], "core_baselines", repetitions)
    paired += _paired_rows(arrays, ABLATIONS, "matched_ablations", repetitions)
    acceptance = provenance_acceptance(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_metrics.csv", per_seed)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_statistics_holm.csv", paired)
    (output_dir / "provenance_acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    manifest = {
        "artifact_role": "main_table_seed012_candidate",
        "eligible_for_paper": False,
        "provenance_acceptance": acceptance["acceptance"],
        "main_table_protocol": "fixed-window OPSD, seeds 0-2, 100 scenarios/date",
        "variants": list(VARIANTS), "raw_is_primary": True,
        "calibration_fit_split": "VS", "ecc_template": "frozen same-seed TAR1M scenarios",
        "statistical_unit": "forecast date; not scenario reads or date-by-seed pseudo-replicates",
        "block_bootstrap": {"block_length_days": 7, "repetitions": repetitions},
        "holm": "within family/variant/metric across the registered comparators",
        "source_per_date_metrics": str(source.resolve()), "source_sha256": sha256_file(source),
        "source_evaluation_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "five_seed_and_rolling_origin": "supplementary evidence only",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    print(run(args.source_dir, args.output_dir, args.bootstrap_repetitions))


if __name__ == "__main__":
    main()
