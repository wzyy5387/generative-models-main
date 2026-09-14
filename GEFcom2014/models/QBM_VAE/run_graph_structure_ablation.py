# -*- coding: utf-8 -*-

"""Run missing cross-dataset J=0 and same-edge random-graph controls."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
OPSD_BUNDLE = ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz"
EXPERIMENTS = (
    (
        "wind_random",
        "wind",
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_wind_lanchor_random.json",
        "abl_lanchor_randw",
    ),
    (
        "opsd_j0",
        "opsd-wind",
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_wind_anchor_independent.json",
        "abl_anchor_ind",
    ),
    (
        "opsd_random",
        "opsd-wind",
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_wind_anchor_random.json",
        "abl_anchor_randw",
    ),
)


def scenario_paths(tag, label, seed):
    directory = ROOT_DIR / "export" / ("qbm_vae_%s" % tag)
    stem = "scenarios_%s_QBMVAE_2_%s_sa_%d_100" % (tag, label, seed)
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
    log_dir = ROOT_DIR / "export" / "experiment_logs" / "graph_structure_ablation"
    for seed in args.seeds:
        for short_name, tag, config, label in EXPERIMENTS:
            validation_path, test_path = scenario_paths(tag, label, seed)
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
        for tag in ("wind", "opsd-wind"):
            print("RUN unified post-processing for %s" % tag, flush=True)
            command = [
                sys.executable,
                "-u",
                "-m",
                "GEFcom2014.forecast_quality.unified_postprocessing",
                "--tag",
                tag,
                "--reuse-completed",
            ]
            if tag == "opsd-wind":
                command.extend(["--dataset-bundle", str(OPSD_BUNDLE)])
            run_logged(
                command,
                log_dir / ("unified_%s.out.log" % tag),
                log_dir / ("unified_%s.err.log" % tag),
            )

        print("RUN cross-dataset graph inference", flush=True)
        run_logged(
            [
                sys.executable,
                "-u",
                "-m",
                "GEFcom2014.forecast_quality.graph_structure_ablation",
                "--bootstrap-repetitions",
                str(args.repetitions),
                "--permutation-repetitions",
                str(args.repetitions),
                "--seed",
                "2026",
            ],
            log_dir / "graph_inference.out.log",
            log_dir / "graph_inference.err.log",
        )
    print("Graph-structure ablation complete", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--repetitions", type=int, default=20000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
