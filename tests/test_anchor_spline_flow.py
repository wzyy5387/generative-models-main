import numpy as np
import torch

from GEFcom2014.models.QBM_VAE.anchor_spline_flow import (
    build_residual_arrays,
    sample_anchor_spline_flow,
)
from GEFcom2014.models.probabilistic_baselines import build_spline_flow


def test_build_residual_arrays_appends_anchor_and_subtracts_it():
    arrays = {}
    anchors = {}
    for split in ("ls", "vs", "test"):
        arrays["x_" + split] = np.ones((3, 4), dtype=np.float32)
        arrays["y_" + split] = np.full((3, 2), 0.75, dtype=np.float32)
        anchors[split] = np.full((3, 2), 0.25, dtype=np.float32)

    residual = build_residual_arrays(arrays, anchors)

    assert residual["x_ls"].shape == (3, 6)
    np.testing.assert_array_equal(residual["x_ls"][:, -2:], anchors["ls"])
    np.testing.assert_array_equal(
        residual["y_ls"], np.full((3, 2), 0.5, dtype=np.float32)
    )


def test_anchor_spline_flow_sampling_is_reproducible_and_finite():
    model = build_spline_flow(
        data_dim=2,
        context_dim=6,
        hidden_dim=8,
        num_transforms=1,
        num_bins=4,
    )
    context = np.zeros((3, 4), dtype=np.float32)
    anchor = np.full((3, 2), 0.25, dtype=np.float32)

    first = sample_anchor_spline_flow(
        model, context, anchor, 5, torch.device("cpu"), seed=13, day_batch_size=2
    )
    second = sample_anchor_spline_flow(
        model, context, anchor, 5, torch.device("cpu"), seed=13, day_batch_size=2
    )

    assert first.shape == (3, 5, 2)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
