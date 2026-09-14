import numpy as np
import torch

from GEFcom2014.models.QBM_VAE.samplers import (
    IsingSampleResult,
    build_sampler,
)
from GEFcom2014.models.QBM_VAE.utils_qbm_vae import ConditionalQBMVAE
from GEFcom2014.models.QBM_VAE.validate_ising_samplers import (
    distribution_diagnostics,
    exact_ising_distribution,
)


class CountingBatchSampler:
    def __init__(self):
        self.calls = 0

    def sample_ising_batch(self, h, j, num_reads=1, beta=1.0, **kwargs):
        self.calls += 1
        samples = np.ones(
            (h.shape[0], num_reads, h.shape[1]), dtype=np.int8
        )
        return IsingSampleResult(
            samples=samples,
            energies=np.zeros((h.shape[0], num_reads)),
            metadata={"backend": "counting"},
        )


def test_external_negative_sampler_is_used_by_qbm_loss():
    model = ConditionalQBMVAE(
        latent_s=4,
        cond_in=3,
        in_size=2,
        enc_w=8,
        dec_w=8,
        gpu=False,
    )
    sampler = CountingBatchSampler()
    target = torch.randn(5, 2)
    context = torch.randn(5, 3)

    loss, components = model.loss(
        target,
        cond_in=context,
        negative_steps=2,
        negative_sampler=sampler,
        negative_num_reads=2,
        return_components=True,
    )
    loss.backward()

    assert sampler.calls == 1
    assert model.last_negative_sampler_metadata["backend"] == "counting"
    assert torch.isfinite(loss)
    assert torch.isfinite(components["negative_energy"])
    assert model.h.grad is not None
    assert model.raw_j.grad is not None


def test_exact_gibbs_and_sa_match_small_ising_distribution():
    h = np.array([0.25, -0.15, 0.1])
    j = np.array(
        [
            [0.0, 0.18, 0.0],
            [0.18, 0.0, -0.12],
            [0.0, -0.12, 0.0],
        ]
    )
    _, exact_energies, exact_probabilities = exact_ising_distribution(
        h, j, beta=1.0
    )
    samplers = [
        build_sampler("exact", seed=7),
        build_sampler("gibbs", seed=7, sweeps=40, burn_in=40),
        build_sampler(
            "sa", seed=7, sweeps=80, beta_start=0.1, beta_end=1.0
        ),
    ]

    diagnostics = [
        distribution_diagnostics(
            sampler.sample_ising(h, j, num_reads=4000, beta=1.0).samples,
            exact_energies,
            exact_probabilities,
        )
        for sampler in samplers
    ]

    assert all(item["total_variation"] < 0.05 for item in diagnostics)
    assert all(abs(item["energy_mean_error"]) < 0.04 for item in diagnostics)


def test_external_negative_sampler_rejects_binary_zero_one_samples():
    class InvalidSampler:
        def sample_ising_batch(self, h, j, num_reads=1, beta=1.0, **kwargs):
            return IsingSampleResult(
                samples=np.zeros((h.shape[0], num_reads, h.shape[1])),
                energies=np.zeros((h.shape[0], num_reads)),
            )

    model = ConditionalQBMVAE(
        latent_s=3,
        cond_in=2,
        in_size=2,
        enc_w=6,
        dec_w=6,
        gpu=False,
    )

    with np.testing.assert_raises_regex(ValueError, r"\{-1, \+1\}"):
        model.sample_external_negative(
            torch.zeros(2, 2),
            InvalidSampler(),
            num_reads=1,
        )
