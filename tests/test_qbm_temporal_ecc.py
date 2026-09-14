import numpy as np

from GEFcom2014.forecast_quality.qbm_temporal_ecc import ensemble_copula_coupling


def test_ecc_preserves_every_period_marginal_exactly():
    rng = np.random.default_rng(13)
    marginal = rng.normal(size=(48, 30))
    template = rng.normal(size=(48, 30))

    coupled = ensemble_copula_coupling(marginal, template)

    np.testing.assert_array_equal(
        np.sort(coupled, axis=1),
        np.sort(marginal, axis=1),
    )


def test_ecc_imports_template_rank_order():
    marginal = np.array([[30.0, 10.0, 20.0], [4.0, 6.0, 5.0]])
    template = np.array([[0.2, 0.3, 0.1], [9.0, 7.0, 8.0]])

    coupled = ensemble_copula_coupling(marginal, template)

    np.testing.assert_array_equal(np.argsort(coupled, axis=1), np.argsort(template, axis=1))


def test_ecc_rejects_shape_mismatch():
    try:
        ensemble_copula_coupling(np.zeros((24, 10)), np.zeros((24, 9)))
    except ValueError as exc:
        assert "equal shapes" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to fail")
