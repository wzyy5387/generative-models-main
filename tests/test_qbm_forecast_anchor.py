import numpy as np
import torch

from GEFcom2014.models.QBM_VAE.forecast_anchor import (
    DeterministicForecastAnchor,
    fit_deterministic_anchor,
    predict_deterministic_anchor,
)


def test_deterministic_anchor_shapes():
    model = DeterministicForecastAnchor(5, 3, hidden_dim=8, layers=2)
    output = model(torch.zeros(4, 5))
    assert output.shape == (4, 3)


def test_anchor_fit_uses_validation_checkpoint_and_predicts_finite_values():
    rng = np.random.default_rng(4)
    x_ls = rng.normal(size=(64, 3)).astype(np.float32)
    x_vs = rng.normal(size=(16, 3)).astype(np.float32)
    weights = np.array([[1.0, -0.5], [0.2, 0.4], [-0.3, 0.8]], dtype=np.float32)
    y_ls = x_ls @ weights
    y_vs = x_vs @ weights

    model, history = fit_deterministic_anchor(
        x_ls,
        y_ls,
        x_vs,
        y_vs,
        hidden_dim=12,
        layers=1,
        epochs=15,
        batch_size=16,
        learning_rate=5e-3,
        patience=5,
        seed=7,
        device=torch.device("cpu"),
    )
    predictions = predict_deterministic_anchor(model, x_vs, torch.device("cpu"))

    assert history
    assert predictions.shape == y_vs.shape
    assert np.isfinite(predictions).all()
    assert min(row["validation_mse"] for row in history) < 1.0
