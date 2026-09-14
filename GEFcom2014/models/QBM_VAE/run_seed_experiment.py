# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys


PROFILE_ARGS = {
    "wind": [],
    "load": [],
    "pv": [
        "--run-label", "pv64",
        "--latent-s", "64",
        "--enc-w", "512",
        "--dec-w", "512",
        "--epochs", "120",
    ],
}


CALIBRATION_ARGS = {
    # Keep the selected Wind scale (1.6) away from the search boundary.
    "wind": [
        "--scale-grid", "0.85", "1.0", "1.15", "1.3", "1.45", "1.6", "1.75", "1.9",
    ],
    "load": [],
    "pv": [],
}


RAW_MODEL_BY_TRACK = {
    "wind": "wind_QBMVAE_2_sa_{seed}",
    "load": "load_QBMVAE_2_sa_{seed}",
    "pv": "pv_QBMVAE_2_pv64_sa_{seed}",
}


CALIBRATED_MODEL_BY_TRACK = {
    "wind": "QBMVAE_2_gibbs_{label}_{seed}",
    "load": "QBMVAE_2_gibbs_{label}_{seed}",
    "pv": "QBMVAE_2_pv64_gibbs_{label}_{seed}",
}

DEFAULT_CALIBRATION_LABEL_BY_TRACK = {
    "wind": "calm",
    "load": "calx",
    "pv": "calm",
}


def main():
    args = parse_args()
    for seed in args.seeds:
        run_seed(args.tag, seed, args)


def run_seed(tag, seed, args):
    calibration_label = args.calibration_label or DEFAULT_CALIBRATION_LABEL_BY_TRACK[tag]
    train_cmd = [
        sys.executable,
        "-u",
        "-m",
        "GEFcom2014.models.QBM_VAE.qbm_vae",
        "--tag", tag,
        "--sampler", "sa",
        "--seed", str(seed),
        "--n-scenarios", str(args.n_scenarios),
        "--skip-plots",
    ] + PROFILE_ARGS[tag]
    if args.cpu:
        train_cmd.append("--cpu")
    if args.epochs is not None:
        train_cmd += ["--epochs", str(args.epochs)]
    run(train_cmd, dry_run=args.dry_run)

    raw_model = RAW_MODEL_BY_TRACK[tag].format(seed=seed)
    calibrate_cmd = [
        sys.executable,
        "-u",
        "-m",
        "GEFcom2014.models.QBM_VAE.calibrate_qbm_vae",
        "--tag", tag,
        "--model-name", raw_model,
        "--sampler", "gibbs",
        "--mode", "joint",
        "--n-scenarios", str(args.n_scenarios),
        "--gibbs-steps", str(args.gibbs_steps),
        "--seed", str(seed),
        "--output-label", calibration_label,
    ] + CALIBRATION_ARGS[tag]
    run(calibrate_cmd, dry_run=args.dry_run)

    trc_model = CALIBRATED_MODEL_BY_TRACK[tag].format(label=calibration_label, seed=seed)
    for split in ("VS", "TEST"):
        trc_cmd = [
            sys.executable,
            "-u",
            "-m",
            "GEFcom2014.forecast_quality.temporal_rank_coupling",
            "--tag", tag,
            "--model-name", trc_model,
            "--split", split,
            "--n-scenarios", str(args.n_scenarios),
            "--seed", str(seed),
        ]
        run(trc_cmd, dry_run=args.dry_run)


def run(cmd, dry_run=False):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run one or more QBM-VAE seeds end-to-end.")
    parser.add_argument("--tag", default="load", choices=["wind", "pv", "load"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--gibbs-steps", type=int, default=10)
    parser.add_argument("--calibration-label", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override profile epochs for quick tests.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
