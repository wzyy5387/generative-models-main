import unittest

import numpy as np

from GEFcom2014.forecast_quality.wind_reliability_inference import (
    aggregate_layer_scores,
    circular_block_sample,
    maqce,
    quantile_grid,
)


class WindReliabilityInferenceTest(unittest.TestCase):
    def test_maqce_is_zero_on_ideal_reliability_diagonal(self):
        self.assertAlmostEqual(float(maqce(quantile_grid())), 0.0, places=12)

    def test_seed_scores_are_computed_before_seed_averaging(self):
        first_seed = np.zeros(99)
        second_seed = np.ones(99)
        seeded_date = np.stack([first_seed, second_seed])[:, None, :]
        contributions = {
            "seeded": True,
            "field_day": seeded_date[:, :, None, :],
            "date": seeded_date,
            "horizon_day": seeded_date[:, :, None, :],
        }
        scores = aggregate_layer_scores(
            contributions,
            seed_weights=np.asarray([0.5, 0.5]),
            day_weights=np.asarray([1.0]),
        )
        expected = 0.5 * float(maqce(first_seed)) + 0.5 * float(maqce(second_seed))
        for score in scores.values():
            self.assertAlmostEqual(score, expected, places=12)

    def test_circular_block_sample_has_requested_size(self):
        indexes = circular_block_sample(50, 7, np.random.default_rng(0))
        self.assertEqual(indexes.shape, (50,))
        self.assertTrue(((indexes >= 0) & (indexes < 50)).all())


if __name__ == "__main__":
    unittest.main()
