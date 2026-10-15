import sys

import numpy as np
import torch

from GEFcom2014.models.QBM_VAE import qbm_vae
from GEFcom2014.models.QBM_VAE.utils_qbm_vae import ConditionalQBMVAE


def test_validation_only_forces_vs_without_changing_legacy_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qbm_vae"])
    legacy = qbm_vae.parse_args()
    assert legacy.scenario_splits == ["VS", "TEST"]
    assert legacy.skip_plots is False

    monkeypatch.setattr(sys, "argv", ["qbm_vae", "--validation-only"])
    validation_only = qbm_vae.parse_args()
    assert validation_only.scenario_splits == ["VS"]
    assert validation_only.skip_plots is True


def test_frozen_anchor_argument_is_available(monkeypatch, tmp_path):
    anchor = tmp_path / "anchor.pt"
    anchor.write_bytes(b"placeholder")
    monkeypatch.setattr(
        sys, "argv", ["qbm_vae", "--frozen-anchor-model", str(anchor)]
    )
    args = qbm_vae.parse_args()
    assert args.frozen_anchor_model == anchor


def test_context_anchor_conditioning_only_is_a_distinct_ablation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qbm_vae", "--context-anchor-conditioning-only"])
    args = qbm_vae.parse_args()
    assert args.context_anchor_conditioning_only is True
    assert args.forecast_anchor is False


def test_zero_graph_mask_keeps_effective_ising_couplings_zero():
    model = ConditionalQBMVAE(
        latent_s=4, cond_in=3, in_size=2, graph_mask=np.zeros((4, 4)), gpu=False
    )
    _, couplings = model.export_ising()
    np.testing.assert_array_equal(couplings, np.zeros((4, 4)))
    assert model.raw_j.requires_grad is False
    np.testing.assert_array_equal(model.raw_j.detach().numpy(), np.zeros((4, 4)))
