# -*- coding: utf-8 -*-

"""Run independent VS-only seeds for trajectory-score confirmation."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
EXPERIMENTS = (
    ("wind", "dev_ctrl", ROOT_DIR / "configs" / "paper" / "fa_bm_vae_wind_lanchor_vs_control.json"),
    ("wind", "dev_tsr", ROOT_DIR / "configs" / "paper" / "fa_bm_vae_wind_lanchor_trajectory_score.json"),
    ("opsd-wind", "dev_ctrl", ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_wind_anchor_vs_control.json"),
    ("opsd-wind", "dev_tsr", ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_wind_anchor_trajectory_score.json"),
)


def scenario_path(track, label, seed):
    name = "scenarios_%s_QBMVAE_2_%s_sa_%d_100_VS.pickle" % (track, label, seed)
    return ROOT_DIR / "export" / ("qbm_vae_%s" % track) / name


def run_logged(command, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT_DIR, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Command failed; see %s" % stderr_path)


def main():
    args = parse_args()
    log_dir = ROOT_DIR / "export" / "experiment_logs" / "trajectory_score_confirmation"
    for seed in args.seeds:
        for track, label, config in EXPERIMENTS:
            output = scenario_path(track, label, seed)
            short_name = "%s_%s_seed%d" % (track, label, seed)
            if output.is_file() and not args.force:
                print("SKIP %s" % short_name, flush=True)
                continue
            print("RUN %s (VS only)" % short_name, flush=True)
            run_logged(
                [sys.executable, "-u", "-m", "GEFcom2014.models.QBM_VAE.qbm_vae", "--config", str(config), "--seed", str(seed)],
                log_dir / (short_name + ".out.log"),
                log_dir / (short_name + ".err.log"),
            )
            if not output.is_file():
                raise RuntimeError("Missing VS scenarios: %s" % output)
    run_logged(
        [sys.executable, "-u", "-m", "GEFcom2014.forecast_quality.confirm_trajectory_score", "--seeds", *[str(seed) for seed in args.seeds], "--repetitions", str(args.repetitions)],
        log_dir / "confirmation.out.log",
        log_dir / "confirmation.err.log",
    )
    print("Trajectory-score confirmation complete", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
