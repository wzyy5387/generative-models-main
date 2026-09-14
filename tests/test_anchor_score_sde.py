from types import SimpleNamespace

import numpy as np
import torch

from GEFcom2014.models.QBM_VAE.anchor_score_sde import (
    ConditionalTemporalScoreNetwork,
    EDMPreconditionedDenoiser,
    edm_noise_schedule,
    sample_anchor_score_sde,
    sample_training_sigmas,
)


def build_model():
    network = ConditionalTemporalScoreNetwork(
        data_dim=6,
        context_dim=10,
        channels=8,
        blocks=2,
        time_dim=8,
    )
    return EDMPreconditionedDenoiser(network, sigma_data=0.5)


def test_score_sde_denoiser_and_loss_are_finite():
    model = build_model()
    target = torch.randn(4, 6)
    context = torch.randn(4, 10)
    sigma = torch.full((4,), 0.2)
    denoised = model.denoise(target, sigma, context)
    loss = model.training_loss(target, context, sigma, torch.randn_like(target))

    assert denoised.shape == target.shape
    assert torch.isfinite(denoised).all()
    assert torch.isfinite(loss)
    loss.backward()


def test_score_sde_noise_schedule_is_monotone_and_ends_at_zero():
    schedule = edm_noise_schedule(
        steps=5,
        sigma_min=0.002,
        sigma_max=1.0,
        rho=7.0,
        device=torch.device("cpu"),
    )

    assert schedule.shape == (6,)
    assert torch.all(schedule[:-2] > schedule[1:-1])
    assert schedule[-1] == 0


def test_score_sde_training_sigmas_respect_bounds():
    args = SimpleNamespace(
        p_mean=-1.2,
        p_std=1.2,
        sigma_min=0.01,
        sigma_max=0.5,
    )
    sigma = sample_training_sigmas(100, args, torch.device("cpu"))

    assert torch.all(sigma >= 0.01)
    assert torch.all(sigma <= 0.5)


def test_anchor_score_sde_sampling_is_reproducible():
    model = build_model().eval()
    context = np.zeros((3, 4), dtype=np.float32)
    anchor = np.full((3, 6), 0.25, dtype=np.float32)

    first = sample_anchor_score_sde(
        model,
        context,
        anchor,
        n_scenarios=4,
        device=torch.device("cpu"),
        seed=17,
        steps=3,
        day_batch_size=2,
    )
    second = sample_anchor_score_sde(
        model,
        context,
        anchor,
        n_scenarios=4,
        device=torch.device("cpu"),
        seed=17,
        steps=3,
        day_batch_size=2,
    )

    assert first.shape == (3, 4, 6)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
