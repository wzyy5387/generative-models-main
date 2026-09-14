import unittest

import numpy as np
import torch

from GEFcom2014.models.QBM_VAE import ConditionalQBMVAE, PathIntegralSampler
from GEFcom2014.models.QBM_VAE.calibration import fit_effective_temperature_pseudolikelihood
from GEFcom2014.models.QBM_VAE.ising import collapse_trotter_samples, expand_trotter_ising, ising_energy
from GEFcom2014.models.QBM_VAE.validate_trotter_mapping import validate_trotter_mapping


class TrotterMappingTest(unittest.TestCase):
    def test_expanded_energy_matches_manual_action_for_two_replicas(self):
        h = np.asarray([0.2, -0.1])
        j = np.asarray([[0.0, 0.4], [0.4, 0.0]])
        gamma = np.asarray([0.6, 0.8])
        beta = 0.9
        replicas = 2
        spins = np.asarray([[1, -1], [-1, -1]], dtype=np.int8)
        expanded_h, expanded_j = expand_trotter_ising(h, j, beta, gamma, replicas)
        expanded_energy = ising_energy(spins.reshape(-1), expanded_h, expanded_j)

        longitudinal = sum(ising_energy(replica, h, j) for replica in spins) * beta / replicas
        coupling = -0.5 * np.log(np.tanh(beta * gamma / replicas))
        transverse = -sum(
            np.dot(coupling, spins[m] * spins[(m + 1) % replicas])
            for m in range(replicas)
        )
        self.assertAlmostEqual(expanded_energy, longitudinal + transverse, places=10)

    def test_trotter_error_decreases_with_replica_count(self):
        result = validate_trotter_mapping(
            h=np.asarray([0.3, -0.2]),
            j=np.asarray([[0.0, 0.35], [0.35, 0.0]]),
            transverse_gamma=np.asarray([0.7, 0.5]),
            beta=0.8,
            replicas_values=(2, 4, 6),
        )
        errors = [row["total_variation"] for row in result["trotter_error"]]
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 0.01)

    def test_sampler_and_model_backward(self):
        sampler = PathIntegralSampler(replicas=3, sweeps=5, seed=0)
        result = sampler.sample_ising(
            np.zeros(4),
            np.zeros((4, 4)),
            num_reads=7,
            beta=1.0,
            transverse_gamma=np.full(4, 0.5),
        )
        self.assertEqual(result.samples.shape, (7, 4))
        self.assertEqual(result.replicas.shape, (7, 3, 4))
        self.assertTrue(np.isin(result.samples, [-1, 1]).all())

        model = ConditionalQBMVAE(
            latent_s=4,
            cond_in=3,
            in_size=2,
            enc_w=8,
            dec_w=8,
            transverse_field=0.5,
            trotter_replicas=3,
            learnable_transverse_field=True,
            gpu=False,
        )
        loss, components = model.loss(
            torch.randn(5, 2),
            cond_in=torch.randn(5, 3),
            negative_steps=2,
            return_components=True,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.raw_transverse_gamma.grad).all())
        self.assertGreaterEqual(components["replica_disagreement"].item(), 0.0)

    def test_trotter_collapse_and_scalable_temperature_calibration(self):
        expanded = np.asarray([
            [1, -1, -1, 1],
            [-1, -1, 1, 1],
        ])
        collapsed = collapse_trotter_samples(expanded, physical_n_bits=2, replicas=2)
        np.testing.assert_array_equal(collapsed, expanded[:, :2])

        rng = np.random.default_rng(7)
        h = np.asarray([0.4, -0.7, 1.0])
        true_beta = 1.3
        probability_up = 1.0 / (1.0 + np.exp(-2.0 * true_beta * h))
        samples = np.where(rng.random((20000, h.size)) < probability_up, 1, -1)
        calibration = fit_effective_temperature_pseudolikelihood(
            h, np.zeros((h.size, h.size)), samples,
            beta_grid=np.linspace(0.5, 2.0, 151),
        )
        self.assertAlmostEqual(calibration["beta_eff"], true_beta, delta=0.08)


if __name__ == "__main__":
    unittest.main()
