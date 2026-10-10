import numpy as np

from GEFcom2014.forecast_quality.aggregate_wind_five_seed import (
    MAIN_MODELS,
    SEEDS,
    block_paired_inference,
    _contiguous_blocks,
    holm_adjust,
    registered_model_specs,
)


def test_registry_is_exactly_the_six_main_families_and_five_seeds():
    assert registered_model_specs() == tuple(MAIN_MODELS)
    assert SEEDS == (0, 1, 2, 3, 4)
    assert "FA-BM-VAE" in registered_model_specs()
    assert all("QBM-VAE" not in name for name in registered_model_specs())


def test_date_blocks_are_contiguous_and_holm_is_monotone():
    blocks = _contiguous_blocks(15, 7)
    assert [block.tolist() for block in blocks] == [list(range(7)), list(range(7, 14)), [14]]
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert np.all(np.asarray(adjusted) <= 1.0)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_sign_flip_weights_short_final_block_by_date_count():
    result = block_paired_inference(np.ones(15), block_length=7, repetitions=2000, seed=4)
    assert result["mean_difference"] == 1.0
    assert 0.15 < result["block_sign_flip_p"] < 0.35
