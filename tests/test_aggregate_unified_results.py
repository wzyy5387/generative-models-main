import numpy as np

from GEFcom2014.forecast_quality.aggregate_unified_results import (
    crps_by_date_block,
    model_family,
    paired_mean_inference,
    ramp_crps_by_date_block,
    score_blocks_by_date,
    select_variants_on_validation,
)


def test_model_family_removes_only_seed_suffix():
    assert model_family("QBM-VAE (seed 2)") == "QBM-VAE"
    assert model_family("Conditional DDPM") == "Conditional DDPM"


def test_variant_selection_uses_validation_and_prefers_simpler_tie():
    rows = []
    for seed in range(2):
        for variant, score in (("Raw", 0.2), ("Cal", 0.1), ("TRC", 0.2), ("Cal+TRC", 0.1)):
            rows.append({
                "family": "Model",
                "split": "VS",
                "variant": variant,
                "mean_crps": score + seed * 0.001,
            })
    assert select_variants_on_validation(rows) == {"Model": "Cal"}


def test_crps_date_blocks_average_sites_for_matching_dates():
    scenarios = np.zeros((2 * 3 * 24, 10))
    targets = np.concatenate((np.ones(3 * 24), np.full(3 * 24, 3.0)))
    blocks = crps_by_date_block(scenarios, targets, n_groups=2)
    np.testing.assert_allclose(blocks, 2.0)


def test_paired_inference_detects_consistent_difference():
    inference = paired_mean_inference(
        np.full(50, 0.02),
        bootstrap_repetitions=1000,
        permutation_repetitions=1000,
        rng=np.random.default_rng(3),
    )
    assert inference["ci_2.5"] > 0
    assert inference["p_value"] < 0.01


def test_score_blocks_return_one_value_per_shared_date():
    rng = np.random.default_rng(21)
    target = rng.normal(size=(6, 24))
    scenarios = rng.normal(size=(6 * 24, 20))

    blocks = score_blocks_by_date(scenarios, target, n_groups=2)

    assert set(blocks) == {"crps", "energy_score", "variogram_score", "ramp_crps"}
    assert all(values.shape == (3,) for values in blocks.values())


def test_ramp_crps_is_zero_for_perfect_deterministic_ramps():
    target = np.tile(np.arange(24, dtype=float), (4, 1))
    scenarios = np.repeat(target.reshape(-1, 1), 10, axis=1)

    scores = ramp_crps_by_date_block(scenarios, target, n_groups=2)

    np.testing.assert_allclose(scores, 0.0)
