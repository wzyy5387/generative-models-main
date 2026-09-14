import numpy as np
import torch

from GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline import (
    ForecastAnchoredGaussian,
    ForecastAnchoredGaussianMixture,
    sample_anchor_gaussian,
)


def test_anchor_gaussian_initializes_as_anchor_centered_distribution():
    model = ForecastAnchoredGaussian(5, 3, hidden_dim=8, layers=1)
    context = torch.randn(4, 5)
    anchor = torch.randn(4, 3)

    mean, log_scale = model.distribution_parameters(context, anchor)

    torch.testing.assert_close(mean, anchor)
    assert mean.shape == log_scale.shape == (4, 3)
    assert torch.isfinite(model.nll(anchor, context, anchor))


def test_anchor_gaussian_sampling_is_reproducible_and_finite():
    model = ForecastAnchoredGaussian(4, 2, hidden_dim=8, layers=1)
    context = np.zeros((3, 4), dtype=np.float32)
    anchor = np.full((3, 2), 0.25, dtype=np.float32)

    first = sample_anchor_gaussian(
        model, context, anchor, 5, torch.device("cpu"), seed=9, batch_size=2
    )
    second = sample_anchor_gaussian(
        model, context, anchor, 5, torch.device("cpu"), seed=9, batch_size=2
    )

    assert first.shape == (3, 5, 2)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_anchor_gaussian_mixture_is_anchor_centered_and_has_finite_nll():
    model = ForecastAnchoredGaussianMixture(
        5, 3, components=4, hidden_dim=8, layers=1
    )
    context = torch.randn(4, 5)
    anchor = torch.randn(4, 3)

    logits, means, log_scale = model.distribution_parameters(context, anchor)

    torch.testing.assert_close(means, anchor[:, None, :].expand(-1, 4, -1))
    assert logits.shape == (4, 4)
    assert log_scale.shape == (4, 4, 3)
    assert torch.isfinite(model.nll(anchor, context, anchor))


def test_anchor_gaussian_mixture_sampling_is_reproducible_and_finite():
    model = ForecastAnchoredGaussianMixture(
        4, 2, components=4, hidden_dim=8, layers=1
    )
    context = np.zeros((3, 4), dtype=np.float32)
    anchor = np.full((3, 2), 0.25, dtype=np.float32)

    first = sample_anchor_gaussian(
        model, context, anchor, 5, torch.device("cpu"), seed=11, batch_size=2
    )
    second = sample_anchor_gaussian(
        model, context, anchor, 5, torch.device("cpu"), seed=11, batch_size=2
    )

    assert first.shape == (3, 5, 2)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
