# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path

import numpy as np

from GEFcom2014 import load_data, pv_data, wind_data
from GEFcom2014.forecast_quality.compare_scenarios import DATA_DIR, ROOT_DIR, load_scenarios
from GEFcom2014.utils import dump_file


DEFAULT_QBM_MODEL_BY_TRACK = {
    "wind": "QBMVAE_2_gibbs_cal_0",
    "pv": "QBMVAE_2_pv64_gibbs_cal_0",
    "load": "QBMVAE_2_gibbs_cal_0",
}


def main():
    args = parse_args()
    model_name = args.model_name or DEFAULT_QBM_MODEL_BY_TRACK[args.tag]
    export_dir = ROOT_DIR / "export" / ("qbm_vae_%s" % args.tag)
    scenario_path = export_dir / ("scenarios_%s_%s_%s_%s.pickle" % (
        args.tag,
        model_name,
        args.n_scenarios,
        args.split,
    ))
    scenarios = load_scenarios(scenario_path, expected_periods=load_target_rows(args.tag, args.split) * 24)
    templates = load_templates(args.tag, split=args.template_split)

    coupled = temporal_rank_coupling(
        scenarios=scenarios,
        template_matrix=templates,
        seed=args.seed,
    )
    output_model_name = insert_variant_suffix(model_name, "trc")
    output_name = "scenarios_%s_%s_%s_%s" % (args.tag, output_model_name, args.n_scenarios, args.split)
    dump_file(dir=str(export_dir) + "/", name=output_name, file=coupled)
    print("Temporal rank coupling written to %s" % (export_dir / (output_name + ".pickle")))


def temporal_rank_coupling(scenarios, template_matrix, seed=0):
    return grouped_temporal_rank_coupling(
        scenarios=scenarios,
        template_matrix=template_matrix,
        seed=seed,
        n_groups=1,
    )


def grouped_temporal_rank_coupling(scenarios, template_matrix, seed=0, n_groups=1):
    """Reorder scenarios using historical daily ranks within each site group.

    The operation changes only the dependence between hours. Every hourly
    ensemble is preserved exactly. GEFCom rows are stored group-major, so
    restricting templates to the matching group prevents cross-site leakage.
    """
    scenarios = np.asarray(scenarios, dtype=np.float64)
    template_matrix = np.asarray(template_matrix, dtype=np.float64)
    if scenarios.ndim != 2:
        raise ValueError("scenarios must be a 2-D matrix")
    if template_matrix.ndim != 2 or template_matrix.shape[1] != 24:
        raise ValueError("template_matrix must have shape (n_days, 24)")
    if scenarios.shape[0] % 24 != 0:
        raise ValueError("scenario periods must be divisible by 24")
    if n_groups < 1:
        raise ValueError("n_groups must be positive")

    n_periods, n_scenarios = scenarios.shape
    n_days = n_periods // 24
    if n_days % n_groups != 0 or template_matrix.shape[0] % n_groups != 0:
        raise ValueError("scenario days and template days must be divisible by n_groups")

    daily = scenarios.reshape(n_days, 24, n_scenarios)
    coupled = np.empty_like(daily)
    rng = np.random.default_rng(seed)
    scenario_days_per_group = n_days // n_groups
    template_days_per_group = template_matrix.shape[0] // n_groups

    for day in range(n_days):
        group = day // scenario_days_per_group
        group_start = group * template_days_per_group
        group_stop = group_start + template_days_per_group
        group_templates = template_matrix[group_start:group_stop]
        replace = group_templates.shape[0] < n_scenarios
        template_indices = rng.choice(group_templates.shape[0], size=n_scenarios, replace=replace)
        template = group_templates[template_indices]
        for hour in range(24):
            rank_order = np.argsort(template[:, hour], kind="mergesort")
            sorted_values = np.sort(daily[day, hour, :], kind="mergesort")
            coupled[day, hour, rank_order] = sorted_values
    return coupled.reshape(n_periods, n_scenarios)


def insert_variant_suffix(model_name, suffix):
    parts = model_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return "%s_%s_%s" % (parts[0], suffix, parts[1])
    return "%s_%s" % (model_name, suffix)


def load_templates(tag, split):
    data, indices = load_raw_track(tag)
    split_to_index = {"LS": 1, "VS": 3}
    df_y = data[split_to_index[split]].copy()
    if tag == "pv":
        non_null_indexes = list(np.delete(np.arange(24), indices))
        df_y.columns = non_null_indexes
        for index in indices:
            df_y[index] = 0
        df_y = df_y.sort_index(axis=1)
    return df_y.values


def load_target_rows(tag, split):
    data, indices = load_raw_track(tag)
    split_to_index = {"VS": 3, "TEST": 5}
    df_y = data[split_to_index[split]].copy()
    if tag == "pv":
        non_null_indexes = list(np.delete(np.arange(24), indices))
        df_y.columns = non_null_indexes
        for index in indices:
            df_y[index] = 0
        df_y = df_y.sort_index(axis=1)
    return len(df_y)


def load_raw_track(tag):
    if tag == "pv":
        return pv_data(DATA_DIR / "solar_new.csv", test_size=50, random_state=0)
    if tag == "wind":
        return wind_data(DATA_DIR / "wind_data_all_zone.csv", test_size=50, random_state=0), []
    if tag == "load":
        return load_data(DATA_DIR / "load_data_track1.csv", test_size=50, random_state=0), []
    raise ValueError("Unknown track: %s" % tag)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply LS-based temporal rank coupling to scenario samples without changing hourly marginals."
    )
    parser.add_argument("--tag", default="load", choices=["wind", "pv", "load"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--split", default="TEST", choices=["VS", "TEST"])
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--template-split", default="LS", choices=["LS", "VS"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
