import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from GEFcom2014.models.QBM_VAE.export_ising_instances import (
    DEFAULT_MODEL_BY_TRACK,
    build_platform_payload,
)
from GEFcom2014.models.QBM_VAE.prepare_bosonic_submission import prepare_submission
from GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios import (
    decode_platform_spins,
)


class DummyDecoder:
    latent_s = 2
    cond_in = 3
    decoder_covariance = "diagonal"
    device = torch.device("cpu")

    def decode_temporal_parameters(self, z, context):
        mean = torch.zeros((z.shape[0], 4), device=self.device)
        log_scale = torch.zeros_like(mean)
        rho = torch.zeros_like(mean)
        return mean, log_scale, rho


def test_hardware_defaults_point_to_current_forecast_anchor_models():
    assert DEFAULT_MODEL_BY_TRACK["wind"] == "wind_QBMVAE_2_lanchor_sa_0"
    assert DEFAULT_MODEL_BY_TRACK["opsd-wind"] == "opsd-wind_QBMVAE_2_anchor_sa_0"


def test_platform_payload_declares_post_training_fa_bm_vae_role():
    payload = build_platform_payload(
        "example",
        np.array([0.2, -0.1]),
        np.array([[0.0, 0.3], [0.3, 0.0]]),
    )

    assert payload["method_name"] == "FA-BM-VAE"
    assert payload["training_backend"] == "classical"
    assert payload["sampling_stage"] == "post_training_conditional_ising_latent"
    assert payload["hardware_claim"] is False


def test_platform_spins_decode_to_bounded_scenarios():
    scaler = StandardScaler().fit(np.zeros((3, 4)))
    spins = np.array([[1, -1], [-1, 1], [1, 1]], dtype=np.int8)
    scenarios = decode_platform_spins(
        DummyDecoder(),
        spins,
        np.zeros(3, dtype=np.float32),
        scaler,
        max_value=0.25,
        observation_noise=False,
    )

    assert scenarios.shape == (3, 4)
    assert np.isfinite(scenarios).all()
    assert np.all((scenarios >= 0.0) & (scenarios <= 0.25))


def test_submission_metadata_preserves_fa_bm_vae_scope(tmp_path):
    payload = build_platform_payload(
        "example_000",
        np.array([0.2, -0.1]),
        np.array([[0.0, 0.3], [0.3, 0.0]]),
    )
    payload_path = tmp_path / "example_000.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "id": "example_000",
                    "split": "VS",
                    "method_name": "FA-BM-VAE",
                    "model_role": "forecast_anchor_conditional_bm_vae",
                    "platform_payload": str(payload_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    submission_path, _, _ = prepare_submission(
        manifest_path,
        tmp_path / "submission",
        stage="vs-calibration",
        requested_reads=10,
    )
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    assert submission["method_name"] == "FA-BM-VAE"
    assert submission["training_backend"] == "classical"
    assert submission["hardware_claim"] is False
