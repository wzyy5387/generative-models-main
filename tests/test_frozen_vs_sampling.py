import json
import pickle

import numpy as np

from GEFcom2014.models.QBM_VAE import frozen_vs_sampling


class _FrozenModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def test_frozen_vs_sampling_reads_only_ls_vs_and_never_trains(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pickle"
    model_path.write_bytes(pickle.dumps(_FrozenModel()))
    (tmp_path / "model.json").write_text(
        json.dumps({"decoder_anchor": True, "sampler": "sa", "beta": 1.0,
                    "negative_steps": 5}), encoding="utf-8"
    )
    bundle = tmp_path / "bundle.npz"
    np.savez(
        bundle,
        x_ls=np.zeros((4, 48), dtype=np.float32),
        y_ls=np.zeros((4, 24), dtype=np.float32),
        x_vs=np.ones((90, 48), dtype=np.float32),
        y_vs=np.ones((90, 24), dtype=np.float32),
        x_test=np.full((90, 48), np.nan, dtype=np.float32),
        y_test=np.full((90, 24), np.nan, dtype=np.float32),
    )
    monkeypatch.setattr(frozen_vs_sampling, "build_sampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        frozen_vs_sampling,
        "build_qbm_vae_scenarios",
        lambda **kwargs: np.zeros((90 * 24, 100), dtype=np.float64),
    )
    metadata = frozen_vs_sampling.generate_frozen_vs(
        model_path, bundle, tmp_path / "out", "origin_1", 0, device="cpu",
        source_commit="test-commit",
    )
    assert metadata["test_data_accessed"] is False
    assert metadata["training_invoked"] is False
    assert metadata["model_parameters_updated"] is False
    assert metadata["shape"] == [2160, 100]
    assert (tmp_path / "out" / "scenarios_opsd-wind_QBMVAE_2_anchor_sa_0_100_VS.pickle").exists()
