import numpy as np
import torch

from GEFcom2014.forecast_quality.opsd_cvae_comparison import (
    ForecastAnchoredCVAE,
    _load_vs_evaluation_arrays,
    _scale_ls_vs,
    diagonal_gaussian_kl,
)


def test_diagonal_gaussian_kl_matches_known_case():
    q_mu = torch.tensor([[1.0, 0.0]])
    q_logvar = torch.zeros_like(q_mu)
    p_mu = torch.zeros_like(q_mu)
    p_logvar = torch.zeros_like(q_mu)
    # Equal unit covariance: KL = 1/2 * ||mu_q - mu_p||^2.
    assert torch.allclose(diagonal_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar), torch.tensor([0.5]))


def test_fa_cvae_anchor_variance_and_sampling_noise():
    torch.manual_seed(1)
    model = ForecastAnchoredCVAE(context_dim=5, data_dim=2, latent_dim=3, hidden_dim=8, layers=1)
    # Set decoder mean to exactly zero residual and observation log standard deviation to zero.
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.zero_()
        model.decoder[-1].bias[2:] = 0.0
    context = torch.tensor([[0.2, -0.1, 0.0, 0.35, 0.7]])
    generator = torch.Generator().manual_seed(71)
    samples = model.sample(context, 4000, generator, batch_size=4000)
    assert samples.shape == (1, 4000, 2)
    # The last two context coordinates are the anchor, and unit log_std yields unit noise.
    assert np.allclose(samples.mean(axis=1)[0], [0.35, 0.7], atol=0.06)
    assert np.allclose(samples.std(axis=1)[0], [1.0, 1.0], atol=0.06)
    assert np.isfinite(samples).all()


def test_ls_scaler_anchor_transform_is_explicit():
    rng = np.random.default_rng(9)
    x_ls = rng.normal(size=(12, 27)).astype(np.float32)
    y_ls = rng.normal(size=(12, 24)).astype(np.float32)
    x_vs = rng.normal(size=(4, 27)).astype(np.float32)
    y_vs = rng.normal(size=(4, 24)).astype(np.float32)
    arrays, x_scaler, y_scaler = _scale_ls_vs(x_ls, y_ls, x_vs, y_vs, "fa_cvae")
    assert arrays["x_ls"].shape == (12, 51)
    assert np.allclose(arrays["x_ls"][:, -24:], y_scaler.transform(x_ls[:, :24]))
    assert np.allclose(arrays["x_vs"][:, -24:], y_scaler.transform(x_vs[:, :24]))
    assert np.allclose(arrays["x_ls"][:, :27], x_scaler.transform(x_ls))


def test_vs_only_loader_returns_complete_dates():
    arrays, dates = _load_vs_evaluation_arrays()
    assert set(arrays) == {"x_vs", "y_vs"}
    assert set(dates) == {"vs"}
    assert arrays["x_vs"].shape[0] == arrays["y_vs"].shape[0] == len(dates["vs"]) == 90
