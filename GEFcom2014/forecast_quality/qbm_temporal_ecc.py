# -*- coding: utf-8 -*-

"""Conditional ensemble copula coupling for QBM-VAE temporal dependence."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR, load_scenarios
from GEFcom2014.forecast_quality.unified_postprocessing import load_targets


def ensemble_copula_coupling(marginal_scenarios, dependence_template):
    """Import template ranks while preserving every period's samples exactly."""
    marginal = np.asarray(marginal_scenarios, dtype=np.float64)
    template = np.asarray(dependence_template, dtype=np.float64)
    if marginal.ndim != 2 or template.ndim != 2:
        raise ValueError("marginal_scenarios and dependence_template must be 2-D")
    if marginal.shape != template.shape:
        raise ValueError("marginal scenarios and dependence template must have equal shapes")

    coupled = np.empty_like(marginal)
    for period in range(marginal.shape[0]):
        rank_order = np.argsort(template[period], kind="mergesort")
        coupled[period, rank_order] = np.sort(marginal[period], kind="mergesort")
    return coupled


def build_seed_pair(
    tag,
    seed,
    source_dir,
    template_dir,
    output_dir,
    template_label,
    dataset_bundle=None,
    source_label="",
    source_version=None,
    output_label="QBMECC",
):
    expected = {
        split: load_targets(tag, split, dataset_bundle=dataset_bundle).size
        for split in ("VS", "TEST")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    sources = {}
    if source_version is None:
        source_version = (
            "QBMVAE_2" + (("_" + source_label) if source_label else "") + "_sa"
        )
    for split in ("VS", "TEST"):
        marginal_path = source_dir / (
            "scenarios_%s_%s_%d_100_%s.pickle"
            % (tag, source_version, seed, split)
        )
        template_path = template_dir / (
            "scenarios_%s_QBMVAE_2_%s_sa_%d_100_%s.pickle"
            % (tag, template_label, seed, split)
        )
        marginal = load_scenarios(marginal_path, expected[split])
        template = load_scenarios(template_path, expected[split])
        coupled = ensemble_copula_coupling(marginal, template)
        output_path = output_dir / (
            "scenarios_%s_%s_%d_100_%s.pickle" % (tag, output_label, seed, split)
        )
        with output_path.open("wb") as handle:
            pickle.dump(coupled, handle)
        outputs[split] = str(output_path.resolve())
        sources[split] = {
            "marginal": str(marginal_path.resolve()),
            "dependence_template": str(template_path.resolve()),
        }

    audit = {
        "track": tag,
        "seed": seed,
        "method": "conditional_ensemble_copula_coupling",
        "source_label": source_label,
        "source_version": source_version,
        "template_label": template_label,
        "output_label": output_label,
        "marginals_preserved_exactly": True,
        "source": sources,
        "output": outputs,
        "test_used_for_selection": False,
        "dataset_bundle": (
            str(Path(dataset_bundle).resolve()) if dataset_bundle is not None else None
        ),
    }
    with (output_dir / ("qbm_ecc_%s_%s_seed_%d.json" % (tag, output_label, seed))).open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Couple QBM-VAE hourly marginals using conditional Temporal-AR ranks."
    )
    parser.add_argument("--tag", default="wind")
    parser.add_argument("--dataset-bundle", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--template-label", default="tar1m")
    parser.add_argument("--source-label", default="")
    parser.add_argument("--source-version", default=None)
    parser.add_argument("--output-label", default="QBMECC")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = args.source_dir or ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag)
    template_dir = args.template_dir or source_dir
    output_dir = args.output_dir or ROOT_DIR / "export" / ("qbm_ecc_%s" % args.tag)
    for seed in args.seeds:
        outputs = build_seed_pair(
            args.tag,
            seed,
            source_dir,
            template_dir,
            output_dir,
            args.template_label,
            dataset_bundle=args.dataset_bundle,
            source_label=args.source_label,
            source_version=args.source_version,
            output_label=args.output_label,
        )
        print("Seed %d | VS %s | TEST %s" % (seed, outputs["VS"], outputs["TEST"]))


if __name__ == "__main__":
    main()
