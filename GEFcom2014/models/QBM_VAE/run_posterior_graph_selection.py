# -*- coding: utf-8 -*-

"""Run the VS-only conditional posterior graph experiment."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
EXPERIMENTS = (
    (
        "wind",
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_wind_lanchor_posterior_pc.json",
    ),
    (
        "opsd-wind",
        ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_wind_anchor_posterior_pc.json",
    ),
)
LABEL = "dev_ppc"


def scenario_path(track, seed):
    stem = "scenarios_%s_QBMVAE_2_%s_sa_%d_100_VS.pickle" % (
        track,
        LABEL,
        seed,
    )
    return ROOT_DIR / "export" / ("qbm_vae_%s" % track) / stem


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
        raise RuntimeError("Command failed; see %s" % stderr_path)


def main():
    args = parse_args()
    log_dir = ROOT_DIR / "export" / "experiment_logs" / "posterior_graph_selection"
    for seed in args.seeds:
        for track, config in EXPERIMENTS:
            output = scenario_path(track, seed)
            if output.is_file() and not args.force:
                print("SKIP complete %s seed %d" % (track, seed), flush=True)
                continue
            print("RUN %s seed %d (VS only)" % (track, seed), flush=True)
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
                log_dir / ("%s_seed%d.out.log" % (track, seed)),
                log_dir / ("%s_seed%d.err.log" % (track, seed)),
            )
            if not output.is_file():
                raise RuntimeError("Training completed without VS scenarios: %s" % output)
            print("DONE %s seed %d" % (track, seed), flush=True)

    print("RUN VS selection inference", flush=True)
    run_logged(
        [
            sys.executable,
            "-u",
            "-m",
            "GEFcom2014.forecast_quality.select_stable_residual_graph",
            "--candidate-label",
            LABEL,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--repetitions",
            str(args.repetitions),
            "--output-dir",
            str(ROOT_DIR / "export" / "posterior_graph_selection"),
        ],
        log_dir / "selection.out.log",
        log_dir / "selection.err.log",
    )
    print("VS-only conditional posterior graph selection complete", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
