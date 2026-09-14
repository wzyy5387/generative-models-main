# -*- coding: utf-8 -*-

"""Run the frozen five-seed D3U and Treeffuser comparison."""

import argparse
import subprocess
import sys
from pathlib import Path

from GEFcom2014.forecast_quality.compare_scenarios import ROOT_DIR


CONFIGS = {
    ("wind", "d3u"): ROOT_DIR / "configs" / "paper" / "d3u_wind.json",
    ("wind", "treeffuser"): ROOT_DIR / "configs" / "paper" / "treeffuser_wind.json",
    ("opsd-wind", "d3u"): ROOT_DIR / "configs" / "paper" / "d3u_opsd_wind.json",
    ("opsd-wind", "treeffuser"): ROOT_DIR / "configs" / "paper" / "treeffuser_opsd_wind.json",
}
LABELS = {"d3u": "D3UAdapted", "treeffuser": "Treeffuser"}


def scenario_paths(track, model, seed):
    directory = ROOT_DIR / "export" / ("%s_%s" % (model, track))
    stem = "scenarios_%s_%s_%d_100" % (track, LABELS[model], seed)
    return directory / (stem + "_VS.pickle"), directory / (stem + "_TEST.pickle")


def run_logged(command, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            command, cwd=ROOT_DIR, stdout=stdout, stderr=stderr, check=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=["wind", "opsd-wind"], default=["wind", "opsd-wind"])
    parser.add_argument("--models", nargs="+", choices=["d3u", "treeffuser"], default=["d3u", "treeffuser"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    log_dir = ROOT_DIR / "export" / "experiment_logs" / "strong_baselines"

    for track in args.tracks:
        for model in args.models:
            for seed in args.seeds:
                vs_path, test_path = scenario_paths(track, model, seed)
                run_id = "%s_%s_seed%d" % (track, model, seed)
                if not args.force and vs_path.is_file() and test_path.is_file():
                    print("SKIP %s" % run_id, flush=True)
                    continue
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "GEFcom2014.models.strong_probabilistic_baselines",
                    "--config",
                    str(CONFIGS[(track, model)]),
                    "--seed",
                    str(seed),
                ]
                print("RUN %s" % run_id, flush=True)
                run_logged(
                    command,
                    log_dir / (run_id + ".out.log"),
                    log_dir / (run_id + ".err.log"),
                )

    if args.evaluate:
        for track in args.tracks:
            postprocess = [
                sys.executable,
                "-u",
                "-m",
                "GEFcom2014.forecast_quality.unified_postprocessing",
                "--tag",
                track,
                "--reuse-completed",
            ]
            aggregate = [
                sys.executable,
                "-u",
                "-m",
                "GEFcom2014.forecast_quality.aggregate_unified_results",
                "--tag",
                track,
                "--bootstrap-repetitions",
                str(args.repetitions),
                "--permutation-repetitions",
                str(args.repetitions),
            ]
            if track == "opsd-wind":
                bundle = ROOT_DIR / "GEFcom2014" / "data" / "external" / "opsd_wind_daily.npz"
                postprocess.extend(("--dataset-bundle", str(bundle)))
                aggregate.extend(("--dataset-bundle", str(bundle)))
            run_logged(
                postprocess,
                log_dir / (track + "_postprocess.out.log"),
                log_dir / (track + "_postprocess.err.log"),
            )
            run_logged(
                aggregate,
                log_dir / (track + "_aggregate.out.log"),
                log_dir / (track + "_aggregate.err.log"),
            )


if __name__ == "__main__":
    main()
