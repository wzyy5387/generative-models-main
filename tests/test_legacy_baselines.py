import numpy as np
import torch

from GEFcom2014.models.GAN.utils_gan_wasserstein import Generator_linear
from GEFcom2014.models.VAE.utils_vae import VAElinear
from GEFcom2014.models.legacy_baselines import energy_score, sample_decoder, sample_umnn


def test_energy_score_is_zero_for_perfect_deterministic_ensemble():
    observations = np.arange(12, dtype=np.float32).reshape(3, 4)
    samples = np.repeat(observations[:, None, :], 5, axis=1)
    assert energy_score(samples, observations) == 0.0


def test_sample_decoder_supports_original_cvae_and_wgan():
    common = dict(latent_s=3, cond_in=5, in_size=4, gpu=False)
    cvae = VAElinear(**common, enc_w=8, enc_l=1, dec_w=8, dec_l=1)
    wgan = Generator_linear(**common, gen_w=8, gen_l=1)
    context = np.zeros((2, 5), dtype=np.float32)

    cvae_samples = sample_decoder(cvae, context, 6, torch.device("cpu"), seed=7)
    wgan_samples = sample_decoder(wgan, context, 6, torch.device("cpu"), seed=7)

    assert cvae_samples.shape == (2, 6, 4)
    assert wgan_samples.shape == (2, 6, 4)
    assert np.isfinite(cvae_samples).all()
    assert np.isfinite(wgan_samples).all()


def test_umnn_chunk_resume_preserves_deterministic_samples(tmp_path):
    class IdentityFlow:
        @staticmethod
        def eval():
            return None

        @staticmethod
        def invert(z, context):
            return z

    context = np.zeros((5, 2), dtype=np.float32)
    first = sample_umnn(
        IdentityFlow(), context, data_dim=3, n_scenarios=4,
        device=torch.device("cpu"), seed=11, day_batch_size=2, chunk_dir=tmp_path,
    )
    resumed = sample_umnn(
        IdentityFlow(), context, data_dim=3, n_scenarios=4,
        device=torch.device("cpu"), seed=11, day_batch_size=2, chunk_dir=tmp_path,
    )

    np.testing.assert_array_equal(first, resumed)
