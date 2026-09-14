# -*- coding: utf-8 -*-

"""Run the matched Wind anchor-by-coupling ablation with resumable logs."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIGS = {
    "noanchor_mi": (
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_wind_noanchor_mi.json",
        "abl_noanchor_mi",
    ),
    "noanchor_ind": (
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_wind_noanchor_independent.json",
        "abl_noanchor_ind",
    ),
    "lanchor_ind": (
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_wind_lanchor_independent.json",
        "abl_lanchor_ind",
    ),
}


def scenario_paths(label, seed):
    directory = ROOT_DIR / "export" / "qbm_vae_wind"
    stem = "scenarios_wind_QBMVAE_2_%s_sa_%d_100" % (label, seed)
    return directory / (stem + "_VS.pickle"), directory / (stem + "_TEST.pickle")


def run_logged(command, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code %d; see %s"
            % (completed.returncode, stderr_path)
        )


def main():
    args = parse_args()
    log_dir = ROOT_DIR / "export" / "experiment_logs" / "anchor_coupling_ablation"
    for seed in args.seeds:
        for short_name, (config, label) in CONFIGS.items():
            validation_path, test_path = scenario_paths(label, seed)
            if validation_path.is_file() and test_path.is_file() and not args.force:
                print("SKIP complete %s seed %d" % (short_name, seed), flush=True)
                continue
            print("RUN %s seed %d" % (short_name, seed), flush=True)
            run_logged(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "GEFcom2014.models.QBM_VAE.qbm_vae",
                    "--config",
                    str(config),
                    "--seed",
                    str(seed),
                ],
                log_dir / ("%s_seed%d.out.log" % (short_name, seed)),
                log_dir / ("%s_seed%d.err.log" % (short_name, seed)),
            )
            if not validation_path.is_file() or not test_path.is_file():
                raise RuntimeError("Training completed without paired scenario artifacts")
            print("DONE %s seed %d" % (short_name, seed), flush=True)

    if args.evaluate:
        print("RUN unified post-processing", flush=True)
        run_logged(
            [
                sys.executable,
                "-u",
                "-m",
                "GEFcom2014.forecast_quality.unified_postprocessing",
                "--tag",
                "wind",
            ],
            log_dir / "unified_eval_final.out.log",
            log_dir / "unified_eval_final.err.log",
        )
        print("RUN factorial inference", flush=True)
        run_logged(
            [
                sys.executable,
                "-u",
                "-m",
                "GEFcom2014.forecast_quality.anchor_coupling_ablation",
                "--tag",
                "wind",
                "--bootstrap-repetitions",
                str(args.repetitions),
                "--permutation-repetitions",
                str(args.repetitions),
                "--seed",
                "2026",
            ],
            log_dir / "factorial_eval_final.out.log",
            log_dir / "factorial_eval_final.err.log",
        )
    print("Anchor-coupling ablation complete", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--repetitions", type=int, default=20000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
