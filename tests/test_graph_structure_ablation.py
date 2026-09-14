import numpy as np

from GEFcom2014.forecast_quality.graph_structure_ablation import (
    add_grouped_holm_adjustment,
    average_seed_blocks,
    multiplicity_family,
    seed_contrast_differences,
)


def test_raw_dependence_metrics_are_the_primary_test_family():
    assert multiplicity_family("Raw", "variogram_score") == "primary_raw_dependence"
    assert multiplicity_family("Raw", "ramp_crps") == "primary_raw_dependence"
    assert multiplicity_family("Raw", "crps") == "primary_raw_marginal"
    assert multiplicity_family("Cal", "variogram_score") == "sensitivity_cal_dependence"


def test_holm_adjustment_is_applied_within_predeclared_families():
    rows = [
        {"multiplicity_family": "primary_raw_dependence", "paired_sign_permutation_p": 0.01},
        {"multiplicity_family": "primary_raw_dependence", "paired_sign_permutation_p": 0.04},
        {"multiplicity_family": "sensitivity_cal_dependence", "paired_sign_permutation_p": 0.03},
    ]

    add_grouped_holm_adjustment(rows)

    np.testing.assert_allclose(
        [row["holm_adjusted_p"] for row in rows],
        [0.02, 0.04, 0.03],
    )
    assert all(row["significant_0.05_after_holm"] for row in rows)


def test_seed_contrasts_preserve_training_seed_directions_before_averaging():
    seed_blocks = {
        "temporal_mi": {
            0: {"crps": np.array([1.0, 2.0])},
            1: {"crps": np.array([2.0, 3.0])},
        },
        "j0": {
            0: {"crps": np.array([2.0, 4.0])},
            1: {"crps": np.array([1.0, 1.0])},
        },
    }
    weights = {"temporal_mi": 1.0, "j0": -1.0}

    differences = seed_contrast_differences(
        seed_blocks, [0, 1], weights, "crps"
    )

    np.testing.assert_allclose(differences[0], [-1.0, -2.0])
    np.testing.assert_allclose(differences[1], [1.0, 2.0])


def test_seed_block_averaging_matches_the_mean_of_seed_scores():
    seed_blocks = {
        "temporal_mi": {
            0: {metric: np.array([1.0, 3.0]) for metric in ("crps", "energy_score", "variogram_score", "ramp_crps")},
            1: {metric: np.array([3.0, 5.0]) for metric in ("crps", "energy_score", "variogram_score", "ramp_crps")},
        }
    }

    averaged = average_seed_blocks(seed_blocks, [0, 1])

    np.testing.assert_allclose(averaged["temporal_mi"]["ramp_crps"], [2.0, 4.0])
