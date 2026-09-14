import numpy as np
import torch

from GEFcom2014.models.QBM_VAE.utils_qbm_vae import (
    ConditionalQBMVAE,
    ensemble_trajectory_scores,
    sample_temporal_ar_residuals,
    sample_standardized_ar_residuals,
    standardized_ar_innovations,
    temporal_ar_innovations,
)
from GEFcom2014.models.QBM_VAE.samplers import SimulatedAnnealingSampler


def test_temporal_ar_innovation_transform_is_invertible():
    generator = torch.Generator().manual_seed(5)
    innovations = torch.randn(8, 24, generator=generator)
    rho = torch.zeros(8, 24)
    rho[:, 1:] = 0.7

    residuals = sample_temporal_ar_residuals(innovations, rho)
    recovered = temporal_ar_innovations(residuals, rho)

    torch.testing.assert_close(recovered, innovations)


def test_ar1_decoder_has_bounded_coefficients_and_finite_loss():
    model = ConditionalQBMVAE(
        latent_s=8,
        cond_in=6,
        in_size=4,
        enc_w=12,
        enc_l=1,
        dec_w=12,
        dec_l=1,
        decoder_covariance="ar1",
        max_ar_coefficient=0.8,
        gpu=False,
    )
    context = torch.randn(10, 6)
    observations = torch.randn(10, 4)
    latent = torch.randint(0, 2, (10, 8)).float()

    mean, log_scale, rho = model.decode_temporal_parameters(latent, context)
    loss = model.loss(
        observations,
        cond_in=context,
        use_negative_phase=False,
    )

    assert mean.shape == log_scale.shape == rho.shape == (10, 4)
    torch.testing.assert_close(rho[:, 0], torch.zeros(10))
    assert torch.max(torch.abs(rho[:, 1:])) < 0.8
    assert torch.isfinite(loss)


def test_standardized_ar_preserves_marginal_scale_and_is_invertible():
    generator = torch.Generator().manual_seed(17)
    noise = torch.randn(30000, 4, generator=generator)
    marginal_scale = torch.tensor([[0.5, 1.0, 1.5, 2.0]]).expand_as(noise)
    rho = torch.zeros_like(noise)
    rho[:, 1:] = torch.tensor([0.3, 0.7, -0.5])

    residuals = sample_standardized_ar_residuals(noise, marginal_scale, rho)
    recovered, log_det = standardized_ar_innovations(residuals, marginal_scale, rho)

    torch.testing.assert_close(recovered, noise, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(residuals.std(dim=0), marginal_scale[0], rtol=0.02, atol=0.02)
    assert torch.all(log_det[:, 1:] <= 0)


def test_diagonal_decoder_remains_backward_compatible():
    model = ConditionalQBMVAE(
        latent_s=4,
        cond_in=3,
        in_size=2,
        enc_w=8,
        enc_l=1,
        dec_w=8,
        dec_l=1,
        decoder_covariance="diagonal",
        gpu=False,
    )
    context = torch.zeros(5, 3)
    latent = torch.zeros(5, 4)

    mean, log_scale = model.decode_distribution(latent, context)
    _, _, rho = model.decode_temporal_parameters(latent, context)

    assert mean.shape == log_scale.shape == (5, 2)
    np.testing.assert_array_equal(rho.detach().numpy(), 0.0)


def test_forecast_anchor_initializes_decoder_as_zero_residual_model():
    model = ConditionalQBMVAE(
        latent_s=4,
        cond_in=5,
        in_size=2,
        enc_w=8,
        enc_l=1,
        dec_w=8,
        dec_l=1,
        decoder_anchor=True,
        gpu=False,
    )
    context = torch.tensor(
        [[10.0, 20.0, 30.0, 0.25, -0.75], [4.0, 5.0, 6.0, -1.0, 1.5]]
    )
    latent = torch.zeros(2, 4)

    mean, _ = model.decode_distribution(latent, context)

    torch.testing.assert_close(mean, context[:, -2:])


def test_forecast_anchor_rejects_context_smaller_than_target():
    with np.testing.assert_raises(ValueError):
        ConditionalQBMVAE(
            latent_s=4,
            cond_in=1,
            in_size=2,
            decoder_anchor=True,
            gpu=False,
        )


def test_batched_sa_supports_conditional_fields_and_reports_exact_energies():
    sampler = SimulatedAnnealingSampler(sweeps=4, seed=9)
    h = np.array([[0.2, -0.1, 0.3], [-0.4, 0.2, 0.1]])
    j = np.array([[0.0, 0.1, 0.0], [0.1, 0.0, -0.2], [0.0, -0.2, 0.0]])

    result = sampler.sample_ising_batch(h, j, num_reads=7, beta=1.0)

    assert result.samples.shape == result.energies.shape + (3,)
    expected = -np.einsum("bri,bi->br", result.samples, h)
    expected -= 0.5 * np.einsum("bri,ij,brj->br", result.samples, j, result.samples)
    np.testing.assert_allclose(result.energies, expected)


def test_trajectory_scores_are_finite_and_zero_for_perfect_ensemble():
    target = torch.tensor([[0.0, 1.0, 3.0], [2.0, 1.0, 0.0]])
    samples = target[:, None, :].expand(-1, 4, -1).clone()

    energy, variogram, ramp = ensemble_trajectory_scores(samples, target)

    torch.testing.assert_close(energy, torch.tensor(0.0))
    torch.testing.assert_close(variogram, torch.tensor(0.0))
    torch.testing.assert_close(ramp, torch.tensor(0.0))


def test_trajectory_scores_have_finite_gradients_for_flat_trajectories():
    target = torch.zeros(2, 4)
    samples = torch.zeros(2, 3, 4, requires_grad=True)

    scores = ensemble_trajectory_scores(samples, target)
    sum(scores).backward()

    assert torch.isfinite(samples.grad).all()


def test_trajectory_regularization_backpropagates_through_encoder_and_decoder():
    model = ConditionalQBMVAE(
        latent_s=4,
        cond_in=3,
        in_size=3,
        enc_w=8,
        enc_l=1,
        dec_w=8,
        dec_l=1,
        gpu=False,
    )
    context = torch.randn(6, 3)
    observations = torch.randn(6, 3)

    loss, components = model.loss(
        observations,
        cond_in=context,
        use_negative_phase=True,
        negative_steps=1,
        return_components=True,
        trajectory_ensemble_size=3,
        trajectory_energy_weight=0.25,
        trajectory_variogram_weight=0.25,
        trajectory_ramp_weight=0.5,
    )
    loss.backward()

    assert components["trajectory_objective"] > 0
    assert model.enc[0].weight.grad is not None
    assert model.dec[0].weight.grad is not None
