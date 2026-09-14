import numpy as np

from GEFcom2014.forecast_quality.anchor_coupling_ablation import (
    CELLS,
    add_holm_adjustment,
    select_variants,
)
from GEFcom2014.forecast_quality.compare_scenarios import crps_per_hour_fast
from GEFcom2014.forecast_quality.temporal_rank_coupling import (
    grouped_temporal_rank_coupling,
)
from GEFcom2014.forecast_quality.unified_postprocessing import (
    HourlyAffineCalibrator,
)


def test_affine_calibration_improves_underdispersed_validation_ensemble():
    rng = np.random.default_rng(7)
    n_days = 80
    n_scenarios = 100
    hours = np.tile(np.arange(24), n_days)
    center = 0.5 + 0.15 * np.sin(2.0 * np.pi * hours / 24.0)
    targets = np.clip(center + rng.normal(0.0, 0.12, center.size), 0.0, 1.0)
    scenarios = np.clip(
        center[:, None] + rng.normal(0.0, 0.035, (center.size, n_scenarios)),
        0.0,
        1.0,
    )

    calibrator = HourlyAffineCalibrator(
        spread_grid=[0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
        harmonics=2,
        ridge=1.0,
    ).fit(scenarios, targets)
    calibrated = calibrator.transform(scenarios)

    raw_crps = crps_per_hour_fast(scenarios, targets).mean()
    calibrated_crps = crps_per_hour_fast(calibrated, targets).mean()
    assert calibrator.parameters.spread > 1.0
    assert calibrated_crps < raw_crps


def test_affine_calibration_preserves_structural_zero_hours():
    rng = np.random.default_rng(2)
    scenarios = rng.uniform(0.0, 0.4, size=(12 * 24, 30))
    targets = rng.uniform(0.0, 0.4, size=(12, 24))
    targets[:, [0, 1, 22, 23]] = 0.0

    calibrator = HourlyAffineCalibrator(spread_grid=[1.0], harmonics=2).fit(
        scenarios,
        targets.reshape(-1),
    )
    calibrated = calibrator.transform(scenarios).reshape(12, 24, 30)

    assert calibrator.parameters.zero_hours == [0, 1, 22, 23]
    np.testing.assert_array_equal(calibrated[:, [0, 1, 22, 23], :], 0.0)


def test_grouped_trc_is_deterministic_and_preserves_hourly_marginals():
    rng = np.random.default_rng(11)
    scenarios = rng.normal(size=(4 * 24, 20))
    templates = np.vstack(
        [
            rng.normal(loc=-2.0, size=(8, 24)),
            rng.normal(loc=2.0, size=(8, 24)),
        ]
    )

    first = grouped_temporal_rank_coupling(scenarios, templates, seed=5, n_groups=2)
    second = grouped_temporal_rank_coupling(scenarios, templates, seed=5, n_groups=2)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.sort(first, axis=1), np.sort(scenarios, axis=1))


def test_grouped_trc_rejects_incompatible_group_layout():
    scenarios = np.zeros((3 * 24, 10))
    templates = np.zeros((8, 24))

    try:
        grouped_temporal_rank_coupling(scenarios, templates, n_groups=2)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected an incompatible group layout to fail")


def test_holm_adjustment_controls_ordered_familywise_p_values():
    rows = [
        {"paired_sign_permutation_p": value}
        for value in (0.001, 0.01, 0.04)
    ]

    add_holm_adjustment(rows)

    np.testing.assert_allclose(
        [row["holm_adjusted_p"] for row in rows],
        [0.003, 0.02, 0.04],
    )
    assert all(row["significant_0.05_after_holm"] for row in rows)


def test_factorial_variant_selection_uses_priority_for_numerical_crps_ties():
    rows = []
    for cell, family in CELLS.items():
        rows.extend([
            {
                "family": family,
                "split": "VS",
                "variant": "Cal",
                "seed": 0,
                "mean_crps": 0.1,
            },
            {
                "family": family,
                "split": "VS",
                "variant": "Cal+TRC",
                "seed": 0,
                "mean_crps": 0.1 - 1e-15,
            },
        ])

    selected = select_variants(rows, seeds=[0])

    assert selected == {cell: "Cal" for cell in CELLS}
