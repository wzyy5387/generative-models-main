import numpy as np
import torch
from diffusers import DDPMScheduler

from GEFcom2014.models.probabilistic_baselines import (
    ConditionalDenoiser,
    build_spline_flow,
    sample_ddpm,
    sample_residual_bootstrap,
    scenarios_to_period_matrix,
)


def test_conditional_spline_flow_shapes():
    flow = build_spline_flow(
        data_dim=4,
        context_dim=6,
        hidden_dim=16,
        num_transforms=2,
        num_bins=4,
    )
    context = torch.randn(3, 6)
    observations = torch.randn(3, 4)

    assert flow.log_prob(observations, context=context).shape == (3,)
    assert flow.sample(5, context=context).shape == (3, 5, 4)


def test_conditional_ddpm_sampling_shape():
    denoiser = ConditionalDenoiser(data_dim=4, context_dim=6, hidden_dim=16, time_dim=8)
    scheduler = DDPMScheduler(num_train_timesteps=5, clip_sample=False)
    context = np.zeros((3, 6), dtype=np.float32)

    samples = sample_ddpm(
        denoiser,
        scheduler,
        context,
        data_dim=4,
        n_scenarios=3,
        inference_steps=3,
        device=torch.device("cpu"),
        day_batch_size=2,
    )

    assert samples.shape == (3, 3, 4)
    assert np.isfinite(samples).all()


def test_pv_scenario_rebuild_restores_zero_hours():
    class IdentityScaler:
        @staticmethod
        def inverse_transform(values):
            return values

    zero_hours = np.array([0, 1, 22, 23])
    samples = np.full((2, 5, 20), 0.4)
    matrix = scenarios_to_period_matrix(samples, IdentityScaler(), "pv", zero_hours)
    daily = matrix.reshape(2, 24, 5)

    assert matrix.shape == (48, 5)
    np.testing.assert_array_equal(daily[:, zero_hours, :], 0.0)
    np.testing.assert_allclose(daily[:, 2:22, :], 0.4)


def test_residual_bootstrap_uses_only_ls_error_days_and_is_reproducible():
    raw = {
        "x_ls": np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float32),
        "y_ls": np.array([[0.2, 0.3], [0.3, 0.4]], dtype=np.float32),
        "x_vs": np.array([[0.5, 0.6]], dtype=np.float32),
        "y_vs": np.array([[0.0, 0.0]], dtype=np.float32),
    }

    first = sample_residual_bootstrap(raw, "vs", n_scenarios=8, seed=11)
    second = sample_residual_bootstrap(raw, "vs", n_scenarios=8, seed=11)

    assert first.shape == (1, 8, 2)
    np.testing.assert_array_equal(first, second)
    possible = np.array([[0.6, 0.7], [0.4, 0.5]], dtype=np.float32)
    for sample in first[0]:
        assert np.any(np.all(np.isclose(possible, sample), axis=1))
