# -*- coding: utf-8 -*-

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .ising import expand_trotter_ising, spin_to_binary


def ensemble_trajectory_scores(samples, target, variogram_power=0.5):
    """Differentiable dimension-normalized Energy, Variogram, and ramp scores."""
    if samples.ndim != 3:
        raise ValueError("samples must have shape (batch, ensemble, periods)")
    if target.ndim != 2 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("target must have shape (batch, periods)")
    if samples.shape[1] < 2:
        raise ValueError("trajectory scores require at least two ensemble members")

    periods = samples.shape[2]
    observation_distance = torch.linalg.vector_norm(
        samples - target[:, None, :], dim=2
    ).mean(dim=1)
    pairwise_distance = torch.cdist(samples, samples, p=2).mean(dim=(1, 2))
    energy = ((observation_distance - 0.5 * pairwise_distance) / periods ** 0.5).mean()

    epsilon = torch.finfo(samples.dtype).eps
    observed_differences = (
        torch.abs(target[:, :, None] - target[:, None, :]) + epsilon
    ) ** variogram_power
    sample_differences = (
        torch.abs(samples[:, :, :, None] - samples[:, :, None, :]) + epsilon
    ) ** variogram_power
    expected_differences = sample_differences.mean(dim=1)
    upper = torch.triu(
        torch.ones(periods, periods, dtype=torch.bool, device=samples.device),
        diagonal=1,
    )
    variogram = (
        (observed_differences - expected_differences)[:, upper] ** 2
    ).mean()

    sample_ramps = torch.diff(samples, dim=2)
    target_ramps = torch.diff(target, dim=1)
    ramp_observation_distance = torch.linalg.vector_norm(
        sample_ramps - target_ramps[:, None, :], dim=2
    ).mean(dim=1)
    ramp_pairwise_distance = torch.cdist(sample_ramps, sample_ramps, p=2).mean(dim=(1, 2))
    ramp_energy = (
        (ramp_observation_distance - 0.5 * ramp_pairwise_distance)
        / max(periods - 1, 1) ** 0.5
    ).mean()
    return energy, variogram, ramp_energy


class ConditionalQBMVAE(nn.Module):
    """
    Conditional VAE with a binary QBM prior interface.

    The encoder uses Bernoulli latent logits. The QBM prior is represented as
    Ising parameters (h, J), with an optional conditional field correction h(x).
    """

    def __init__(self, latent_s, cond_in, in_size, enc_w=200, enc_l=1, dec_w=200, dec_l=1,
                 conditional_qbm=True, graph_mask=None, gpu=True, min_log_scale=-5.0,
                 max_log_scale=1.0, zone_embedding_dim=0, nb_zones=0,
                 transverse_field=0.0, trotter_replicas=4,
                 learnable_transverse_field=False, min_transverse_field=0.02,
                 max_transverse_field=5.0, decoder_covariance="diagonal",
                 max_ar_coefficient=0.95, decoder_anchor=False):
        super().__init__()
        self.latent_s = latent_s
        self.cond_in = cond_in
        self.in_size = in_size
        self.conditional_qbm = conditional_qbm
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale
        self.zone_embedding_dim = zone_embedding_dim
        self.nb_zones = nb_zones
        self.trotter_replicas = int(trotter_replicas)
        self.learnable_transverse_field = bool(learnable_transverse_field)
        self.min_transverse_field = float(min_transverse_field)
        self.max_transverse_field = float(max_transverse_field)
        if decoder_covariance not in ("diagonal", "ar1"):
            raise ValueError("decoder_covariance must be 'diagonal' or 'ar1'")
        if not 0.0 < max_ar_coefficient < 1.0:
            raise ValueError("max_ar_coefficient must be between zero and one")
        self.decoder_covariance = decoder_covariance
        self.max_ar_coefficient = float(max_ar_coefficient)
        self.decoder_anchor = bool(decoder_anchor)
        if self.decoder_anchor and cond_in < in_size:
            raise ValueError("decoder_anchor requires at least in_size context features")
        self.transverse_field_enabled = float(transverse_field) > 0.0
        self.prior_type = "trotter_bound" if self.transverse_field_enabled else "classical_ising"
        if self.transverse_field_enabled and self.trotter_replicas < 2:
            raise ValueError("trotter_replicas must be at least 2 when transverse field is enabled")
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        context_size = cond_in + zone_embedding_dim
        if zone_embedding_dim > 0:
            if nb_zones <= 0:
                raise ValueError("nb_zones must be positive when zone_embedding_dim > 0")
            self.zone_embedding = nn.Embedding(nb_zones, zone_embedding_dim)
        else:
            self.zone_embedding = None

        enc_layers = [in_size + context_size] + [enc_w] * enc_l + [latent_s]
        decoder_multiplier = 3 if decoder_covariance == "ar1" else 2
        dec_layers = [latent_s + context_size] + [dec_w] * dec_l + [decoder_multiplier * in_size]
        self.enc = _build_mlp(enc_layers)
        self.dec = _build_mlp(dec_layers)
        if self.decoder_anchor:
            output_layer = self.dec[-1]
            with torch.no_grad():
                output_layer.weight[:in_size].zero_()
                output_layer.bias[:in_size].zero_()

        self.h = nn.Parameter(torch.zeros(latent_s))
        self.raw_j = nn.Parameter(0.01 * torch.randn(latent_s, latent_s))
        initial_gamma = torch.full((latent_s,), max(float(transverse_field), self.min_transverse_field))
        self.register_buffer("fixed_transverse_gamma", initial_gamma)
        if self.transverse_field_enabled and self.learnable_transverse_field:
            shifted = torch.clamp(initial_gamma - self.min_transverse_field, min=1e-6)
            self.raw_transverse_gamma = nn.Parameter(torch.log(torch.expm1(shifted)))
        else:
            self.register_parameter("raw_transverse_gamma", None)
        if conditional_qbm:
            self.cond_to_h = nn.Linear(context_size, latent_s)
        else:
            self.cond_to_h = None

        if graph_mask is None:
            graph_mask = torch.ones(latent_s, latent_s) - torch.eye(latent_s)
        else:
            graph_mask = torch.tensor(graph_mask, dtype=torch.float32)
        self.register_buffer("graph_mask", graph_mask)
        self.register_buffer("persistent_spins", torch.empty(0, latent_s))
        self.register_buffer("persistent_replicas", torch.empty(0, self.trotter_replicas, latent_s))
        self.to(self.device)

    def uses_path_integral(self):
        return bool(getattr(self, "transverse_field_enabled", False))

    def transverse_gamma(self):
        if not self.uses_path_integral():
            return torch.zeros(self.latent_s, device=self.device)
        raw_gamma = getattr(self, "raw_transverse_gamma", None)
        if raw_gamma is None:
            gamma = self.fixed_transverse_gamma
        else:
            gamma = self.min_transverse_field + torch.nn.functional.softplus(raw_gamma)
        return torch.clamp(gamma, max=self.max_transverse_field)

    def qbm_params(self, cond_in=None):
        j = 0.5 * (self.raw_j + self.raw_j.t()) * self.graph_mask
        j = j - torch.diag(torch.diag(j))
        h = self.h
        if cond_in is not None and self.cond_to_h is not None:
            h = h.unsqueeze(0) + self.cond_to_h(self.context_features(cond_in))
        return h, j

    def encode_logits(self, x0, cond_in):
        return self.enc(torch.cat((x0, self.context_features(cond_in)), dim=1))

    def context_features(self, cond_in):
        zone_embedding = getattr(self, "zone_embedding", None)
        zone_embedding_dim = getattr(self, "zone_embedding_dim", 0)
        nb_zones = getattr(self, "nb_zones", 0)
        if zone_embedding is None or zone_embedding_dim <= 0:
            return cond_in
        zone_logits = cond_in[:, -nb_zones:]
        zone_ids = torch.argmax(zone_logits, dim=1)
        return torch.cat((cond_in, zone_embedding(zone_ids)), dim=1)

    def sample_binary_latent(self, logits, temperature=0.67):
        noise = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        logistic = torch.log(noise) - torch.log1p(-noise)
        relaxed = torch.sigmoid((logits + logistic) / temperature)
        hard = (relaxed > 0.5).float()
        return hard.detach() - relaxed.detach() + relaxed

    def posterior_predictive_ensemble(self, logits, cond_in, ensemble_size, temperature):
        batch_size = logits.shape[0]
        expanded_logits = logits[:, None, :].expand(-1, ensemble_size, -1)
        latent = self.sample_binary_latent(
            expanded_logits.reshape(-1, self.latent_s), temperature=temperature
        )
        expanded_context = cond_in[:, None, :].expand(-1, ensemble_size, -1)
        expanded_context = expanded_context.reshape(-1, self.cond_in)
        mean, log_scale, rho = self.decode_temporal_parameters(latent, expanded_context)
        noise = torch.randn_like(mean)
        if getattr(self, "decoder_covariance", "diagonal") == "ar1":
            residual = sample_standardized_ar_residuals(
                noise, torch.exp(log_scale), rho
            )
        else:
            residual = torch.exp(log_scale) * noise
        return (mean + residual).reshape(batch_size, ensemble_size, self.in_size)

    def decode_distribution(self, z, cond_in):
        mean, log_scale, _ = self.decode_temporal_parameters(z, cond_in)
        return mean, log_scale

    def decode_temporal_parameters(self, z, cond_in):
        output = self.dec(torch.cat((z, self.context_features(cond_in)), dim=1))
        decoder_covariance = getattr(self, "decoder_covariance", "diagonal")
        if decoder_covariance == "ar1":
            mean, log_scale, raw_rho = torch.split(output, self.in_size, dim=1)
            rho = getattr(self, "max_ar_coefficient", 0.95) * torch.tanh(raw_rho)
            rho = torch.cat((torch.zeros_like(rho[:, :1]), rho[:, 1:]), dim=1)
        else:
            mean, log_scale = torch.split(output, self.in_size, dim=1)
            rho = torch.zeros_like(mean)
        if getattr(self, "decoder_anchor", False):
            mean = mean + cond_in[:, -self.in_size:]
        log_scale = torch.clamp(log_scale, self.min_log_scale, self.max_log_scale)
        return mean, log_scale, rho

    def qbm_energy_torch(self, z_binary, cond_in=None):
        spin = 2.0 * z_binary - 1.0
        h, j = self.qbm_params(cond_in)
        if h.ndim == 1:
            linear = -(spin * h.unsqueeze(0)).sum(dim=1)
        else:
            linear = -(spin * h).sum(dim=1)
        quadratic = -0.5 * torch.einsum("bi,ij,bj->b", spin, j, spin)
        return linear + quadratic

    def path_integral_action_torch(self, replicas_binary, cond_in=None, beta=1.0):
        """Dimensionless finite-M Suzuki-Trotter action for binary replicas."""
        if not self.uses_path_integral():
            raise RuntimeError("Path-integral action requires a positive transverse field")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if replicas_binary.ndim != 3:
            raise ValueError("replicas_binary must have shape (batch, replicas, latent_s)")
        spins = 2.0 * replicas_binary - 1.0
        batch_size, replicas, latent_s = spins.shape
        if replicas != self.trotter_replicas or latent_s != self.latent_s:
            raise ValueError("Replica tensor does not match the configured path-integral shape")
        h, j = self.qbm_params(cond_in)
        if h.ndim == 1:
            h = h.unsqueeze(0).expand(batch_size, -1)
        linear = -(spins * h[:, None, :]).sum(dim=2)
        quadratic = -0.5 * torch.einsum("bmi,ij,bmj->bm", spins, j, spins)
        longitudinal_action = (float(beta) / replicas) * (linear + quadratic).sum(dim=1)

        gamma = self.transverse_gamma()
        argument = torch.clamp(float(beta) * gamma / replicas, min=1e-6)
        coupling = -0.5 * torch.log(torch.tanh(argument).clamp_min(1e-12))
        adjacent = spins * torch.roll(spins, shifts=-1, dims=1)
        transverse_action = -(adjacent * coupling[None, None, :]).sum(dim=(1, 2))
        return longitudinal_action + transverse_action

    @torch.no_grad()
    def sample_path_integral_negative(self, cond_in, steps=20, beta=1.0, persistent=True):
        """Sample conditional Trotter replicas at fixed dimensionless temperature."""
        if not self.uses_path_integral():
            raise RuntimeError("Path-integral sampling requires a positive transverse field")
        if beta <= 0:
            raise ValueError("beta must be positive")
        batch_size = cond_in.shape[0]
        replicas = self.trotter_replicas
        h, j = self.qbm_params(cond_in if self.conditional_qbm else None)
        if h.ndim == 1:
            h = h.unsqueeze(0).expand(batch_size, -1)
        if persistent and self.persistent_replicas.shape == (batch_size, replicas, self.latent_s):
            spins = self.persistent_replicas.clone()
        else:
            spins = torch.where(
                torch.rand(batch_size, replicas, self.latent_s, device=self.device) < 0.5,
                -torch.ones(1, device=self.device),
                torch.ones(1, device=self.device),
            )

        gamma = self.transverse_gamma().detach()
        argument = torch.clamp(float(beta) * gamma / replicas, min=1e-6)
        coupling = -0.5 * torch.log(torch.tanh(argument).clamp_min(1e-12))
        for _ in range(steps):
            for replica in torch.randperm(replicas, device=self.device):
                replica_index = int(replica.item())
                previous_index = (replica_index - 1) % replicas
                next_index = (replica_index + 1) % replicas
                for bit in torch.randperm(self.latent_s, device=self.device):
                    bit_index = int(bit.item())
                    longitudinal_field = h[:, bit_index] + spins[:, replica_index, :] @ j[:, bit_index]
                    imaginary_time_field = coupling[bit_index] * (
                        spins[:, previous_index, bit_index] + spins[:, next_index, bit_index]
                    )
                    local_field = (float(beta) / replicas) * longitudinal_field + imaginary_time_field
                    probability_up = torch.sigmoid(2.0 * local_field)
                    spins[:, replica_index, bit_index] = torch.where(
                        torch.rand(batch_size, device=self.device) < probability_up,
                        torch.ones(batch_size, device=self.device),
                        -torch.ones(batch_size, device=self.device),
                    )
        if persistent:
            self.persistent_replicas = spins.detach()
        return 0.5 * (spins + 1.0)

    @torch.no_grad()
    def sample_conditional_negative(self, cond_in, steps=20, beta=1.0, persistent=True):
        """
        Draw one negative-phase sample for every conditional input.

        Each row uses its own conditional field h(x), while all rows share J.
        Samples are treated as constants when the contrastive energy is
        differentiated, which gives the standard positive-minus-negative phase.
        """
        batch_size = cond_in.shape[0]
        h, j = self.qbm_params(cond_in if self.conditional_qbm else None)
        if h.ndim == 1:
            h = h.unsqueeze(0).expand(batch_size, -1)

        if persistent and self.persistent_spins.shape == (batch_size, self.latent_s):
            spins = self.persistent_spins.clone()
        else:
            spins = torch.where(
                torch.rand(batch_size, self.latent_s, device=self.device) < 0.5,
                -torch.ones(1, device=self.device),
                torch.ones(1, device=self.device),
            )

        for _ in range(steps):
            for index in torch.randperm(self.latent_s, device=self.device):
                local_field = h[:, index] + spins @ j[:, index]
                probability_up = torch.sigmoid(2.0 * beta * local_field)
                spins[:, index] = torch.where(
                    torch.rand(batch_size, device=self.device) < probability_up,
                    torch.ones(batch_size, device=self.device),
                    -torch.ones(batch_size, device=self.device),
                )

        if persistent:
            self.persistent_spins = spins.detach()
        return 0.5 * (spins + 1.0)

    @torch.no_grad()
    def sample_external_negative(
        self,
        cond_in,
        sampler,
        num_reads=1,
        beta=1.0,
        sampler_kwargs=None,
    ):
        """Draw conditional negative-phase samples through an external Ising backend."""
        if self.uses_path_integral():
            raise ValueError(
                "External classical Ising negative sampling is not valid for a "
                "transverse-field path-integral prior"
            )
        if sampler is None:
            raise ValueError("sampler is required for external negative sampling")
        if num_reads < 1:
            raise ValueError("num_reads must be positive")
        sampler_kwargs = dict(sampler_kwargs or {})
        batch_size = cond_in.shape[0]
        h, j = self.qbm_params(cond_in if self.conditional_qbm else None)
        h_numpy = h.detach().cpu().numpy()
        j_numpy = j.detach().cpu().numpy()

        if h.ndim == 1:
            result = sampler.sample_ising(
                h_numpy,
                j_numpy,
                num_reads=batch_size * num_reads,
                beta=beta,
                **sampler_kwargs,
            )
            samples = np.asarray(result.samples).reshape(
                batch_size, num_reads, self.latent_s
            )
        elif hasattr(sampler, "sample_ising_batch"):
            result = sampler.sample_ising_batch(
                h_numpy,
                j_numpy,
                num_reads=num_reads,
                beta=beta,
                **sampler_kwargs,
            )
            samples = np.asarray(result.samples)
        else:
            results = [
                sampler.sample_ising(
                    field,
                    j_numpy,
                    num_reads=num_reads,
                    beta=beta,
                    **sampler_kwargs,
                )
                for field in h_numpy
            ]
            samples = np.stack(
                [np.asarray(item.samples) for item in results], axis=0
            )
            result = results[-1]

        expected_shape = (batch_size, num_reads, self.latent_s)
        if samples.shape != expected_shape:
            raise ValueError(
                "External sampler returned shape %s; expected %s"
                % (samples.shape, expected_shape)
            )
        if not np.all(np.isin(samples, (-1, 1))):
            raise ValueError("External sampler must return Ising spins in {-1, +1}")
        self.last_negative_sampler_metadata = dict(
            getattr(result, "metadata", {})
        )
        return torch.as_tensor(
            0.5 * (samples + 1.0),
            dtype=torch.float32,
            device=self.device,
        )

    def loss(self, x0, cond_in=None, qbm_weight=1e-2, temperature=0.67,
             negative_steps=20, beta=1.0, use_negative_phase=True,
             persistent=True, return_components=False, negative_sampler=None,
             negative_num_reads=1, negative_sampler_kwargs=None,
             trajectory_ensemble_size=4, trajectory_energy_weight=0.0,
             trajectory_variogram_weight=0.0, trajectory_ramp_weight=0.0):
        bs = x0.shape[0]
        if cond_in is None:
            cond_in = torch.zeros(bs, self.cond_in, device=self.device)

        logits = self.encode_logits(x0, cond_in)
        z = self.sample_binary_latent(logits, temperature=temperature)
        mean, log_scale, rho = self.decode_temporal_parameters(z, cond_in)
        residual = x0 - mean
        if getattr(self, "decoder_covariance", "diagonal") == "ar1":
            marginal_scale = torch.exp(log_scale)
            normalized_innovations, correlation_log_det = standardized_ar_innovations(
                residual, marginal_scale, rho
            )
            recon = (
                0.5 * normalized_innovations ** 2 + log_scale + correlation_log_det
            ).sum(dim=1).mean()
        else:
            inverse_variance = torch.exp(-2.0 * log_scale)
            recon = (0.5 * residual ** 2 * inverse_variance + log_scale).sum(dim=1).mean()

        if self.uses_path_integral():
            positive_replicas = z[:, None, :].expand(-1, self.trotter_replicas, -1)
            positive_energy = self.path_integral_action_torch(
                positive_replicas,
                cond_in=cond_in if self.conditional_qbm else None,
                beta=beta,
            ).mean()
        else:
            positive_energy = self.qbm_energy_torch(
                z, cond_in=cond_in if self.conditional_qbm else None
            ).mean()
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).sum(dim=1).mean()

        negative_energy = torch.tensor(0.0, device=self.device)
        replica_disagreement = torch.tensor(0.0, device=self.device)
        if use_negative_phase:
            if self.uses_path_integral():
                negative_replicas = self.sample_path_integral_negative(
                    cond_in=cond_in,
                    steps=negative_steps,
                    beta=beta,
                    persistent=persistent,
                )
                negative_energy = self.path_integral_action_torch(
                    negative_replicas,
                    cond_in=cond_in if self.conditional_qbm else None,
                    beta=beta,
                ).mean()
                negative_spins = 2.0 * negative_replicas - 1.0
                agreement = (
                    negative_spins * torch.roll(negative_spins, shifts=-1, dims=1)
                ).mean()
                replica_disagreement = 0.5 * (1.0 - agreement)
            else:
                negative_cond_in = cond_in
                if negative_sampler is None:
                    z_neg = self.sample_conditional_negative(
                        cond_in=cond_in,
                        steps=negative_steps,
                        beta=beta,
                        persistent=persistent,
                    )
                else:
                    z_neg = self.sample_external_negative(
                        cond_in=cond_in,
                        sampler=negative_sampler,
                        num_reads=negative_num_reads,
                        beta=beta,
                        sampler_kwargs=negative_sampler_kwargs,
                    )
                    z_neg = z_neg.reshape(-1, self.latent_s)
                    negative_cond_in = cond_in.repeat_interleave(
                        negative_num_reads, dim=0
                    )
                negative_energy = self.qbm_energy_torch(
                    z_neg,
                    cond_in=negative_cond_in if self.conditional_qbm else None,
                ).mean()

        qbm_objective = positive_energy - negative_energy - entropy
        trajectory_energy = torch.tensor(0.0, device=self.device)
        trajectory_variogram = torch.tensor(0.0, device=self.device)
        trajectory_ramp = torch.tensor(0.0, device=self.device)
        trajectory_enabled = use_negative_phase and any(
            weight > 0
            for weight in (
                trajectory_energy_weight,
                trajectory_variogram_weight,
                trajectory_ramp_weight,
            )
        )
        if trajectory_enabled:
            ensemble = self.posterior_predictive_ensemble(
                logits,
                cond_in,
                ensemble_size=trajectory_ensemble_size,
                temperature=temperature,
            )
            trajectory_energy, trajectory_variogram, trajectory_ramp = (
                ensemble_trajectory_scores(ensemble, x0)
            )
        trajectory_objective = (
            trajectory_energy_weight * trajectory_energy
            + trajectory_variogram_weight * trajectory_variogram
            + trajectory_ramp_weight * trajectory_ramp
        )
        total = recon + qbm_weight * qbm_objective + trajectory_objective
        if return_components:
            return total, {
                "recon": recon.detach(),
                "positive_energy": positive_energy.detach(),
                "negative_energy": negative_energy.detach(),
                "entropy": entropy.detach(),
                "qbm_objective": qbm_objective.detach(),
                "trajectory_energy": trajectory_energy.detach(),
                "trajectory_variogram": trajectory_variogram.detach(),
                "trajectory_ramp": trajectory_ramp.detach(),
                "trajectory_objective": trajectory_objective.detach(),
                "transverse_gamma_mean": self.transverse_gamma().mean().detach(),
                "replica_disagreement": replica_disagreement.detach(),
            }
        return total

    def export_ising(self, cond_in=None):
        with torch.no_grad():
            if cond_in is not None:
                cond_tensor = torch.tensor(cond_in, dtype=torch.float32, device=self.device)
                h, j = self.qbm_params(cond_tensor)
                h = h.mean(dim=0)
            else:
                h, j = self.qbm_params(None)
            return h.detach().cpu().numpy(), j.detach().cpu().numpy()

    def export_trotter_ising(self, cond_in=None, beta=1.0, replicas=None):
        """Export the finite-replica classical action used by path-integral sampling."""
        if not self.uses_path_integral():
            raise RuntimeError("Trotter export requires a positive transverse field")
        h, j = self.export_ising(cond_in=cond_in)
        gamma = self.transverse_gamma().detach().cpu().numpy()
        return expand_trotter_ising(
            h,
            j,
            beta=float(beta),
            transverse_gamma=gamma,
            replicas=int(replicas or self.trotter_replicas),
        )

    def sample(self, n_s=1, x_cond=None, sampler=None, beta=1.0, observation_noise=True,
               scale_multiplier=1.0, gibbs_steps=50):
        if x_cond is None:
            x_cond = np.zeros(self.cond_in, dtype=np.float32)
        context = torch.tensor(np.tile(x_cond, n_s).reshape(n_s, self.cond_in), device=self.device).float()

        if sampler is None:
            if self.uses_path_integral():
                replicas = self.sample_path_integral_negative(
                    cond_in=context,
                    steps=gibbs_steps,
                    beta=beta,
                    persistent=False,
                )
                z = replicas[:, 0, :]
            else:
                z = self.sample_conditional_negative(
                    cond_in=context,
                    steps=gibbs_steps,
                    beta=beta,
                    persistent=False,
                )
        else:
            h, j = self.export_ising(cond_in=x_cond[None, :] if self.conditional_qbm else None)
            sampler_kwargs = {}
            if self.uses_path_integral():
                sampler_kwargs = {
                    "transverse_gamma": self.transverse_gamma().detach().cpu().numpy(),
                    "trotter_replicas": self.trotter_replicas,
                }
            result = sampler.sample_ising(h, j, num_reads=n_s, beta=beta, **sampler_kwargs)
            z = torch.tensor(spin_to_binary(result.samples), dtype=torch.float32, device=self.device)

        mean, log_scale, rho = self.decode_temporal_parameters(z, context)
        if observation_noise:
            noise = scale_multiplier * torch.randn_like(mean)
            if getattr(self, "decoder_covariance", "diagonal") == "ar1":
                residual = sample_standardized_ar_residuals(
                    noise, torch.exp(log_scale), rho
                )
            else:
                residual = torch.exp(log_scale) * noise
            scenarios = mean + residual
        else:
            scenarios = mean
        scenarios = scenarios.view(n_s, -1).cpu().detach().numpy()
        return scenarios


def fit_qbm_vae(nb_epoch, x_LS, y_LS, x_VS, y_VS, x_TEST, y_TEST, model, opt, sampler=None,
                gpu=True, qbm_weight=0.1, batch_size=128, warmup_epochs=30,
                negative_steps=20, beta=1.0, early_stopping_patience=30,
                temperature_start=1.0, temperature_end=0.5, grad_clip=5.0, seed=0,
                evaluate_test_during_training=False, negative_sampler=None,
                negative_num_reads=1, negative_sampler_kwargs=None,
                trajectory_ensemble_size=4, trajectory_energy_weight=0.0,
                trajectory_variogram_weight=0.0, trajectory_ramp_weight=0.0):
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.device = device
    x_LS_t = torch.tensor(x_LS, dtype=torch.float32)
    y_LS_t = torch.tensor(y_LS, dtype=torch.float32)
    x_VS_t = torch.tensor(x_VS, dtype=torch.float32, device=device)
    y_VS_t = torch.tensor(y_VS, dtype=torch.float32, device=device)
    if evaluate_test_during_training:
        x_TEST_t = torch.tensor(x_TEST, dtype=torch.float32, device=device)
        y_TEST_t = torch.tensor(y_TEST, dtype=torch.float32, device=device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_LS_t, y_LS_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    loss_rows = []
    best_model = None
    best_vs = np.inf
    epochs_without_improvement = 0
    for epoch in range(nb_epoch):
        model.train()
        epoch_fraction = epoch / max(nb_epoch - 1, 1)
        temperature = temperature_start + epoch_fraction * (temperature_end - temperature_start)
        current_qbm_weight = qbm_weight * min(1.0, (epoch + 1) / max(warmup_epochs, 1))
        train_total = 0.0
        train_recon = 0.0
        train_qbm = 0.0
        train_gamma = 0.0
        train_replica_disagreement = 0.0
        train_trajectory = 0.0
        n_train = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            batch_loss, components = model.loss(
                y_batch,
                cond_in=x_batch,
                qbm_weight=current_qbm_weight,
                temperature=temperature,
                negative_steps=negative_steps,
                beta=beta,
                use_negative_phase=True,
                persistent=True,
                return_components=True,
                negative_sampler=negative_sampler,
                negative_num_reads=negative_num_reads,
                negative_sampler_kwargs=negative_sampler_kwargs,
                trajectory_ensemble_size=trajectory_ensemble_size,
                trajectory_energy_weight=trajectory_energy_weight,
                trajectory_variogram_weight=trajectory_variogram_weight,
                trajectory_ramp_weight=trajectory_ramp_weight,
            )
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            batch_n = y_batch.shape[0]
            train_total += batch_loss.item() * batch_n
            train_recon += components["recon"].item() * batch_n
            train_qbm += components["qbm_objective"].item() * batch_n
            train_gamma += components["transverse_gamma_mean"].item() * batch_n
            train_replica_disagreement += components["replica_disagreement"].item() * batch_n
            train_trajectory += components["trajectory_objective"].item() * batch_n
            n_train += batch_n

        model.eval()
        with torch.no_grad():
            vs_loss, vs_components = model.loss(
                y_VS_t,
                cond_in=x_VS_t,
                qbm_weight=current_qbm_weight,
                temperature=temperature,
                use_negative_phase=False,
                persistent=False,
                return_components=True,
            )
            if evaluate_test_during_training:
                test_loss, _ = model.loss(
                    y_TEST_t,
                    cond_in=x_TEST_t,
                    qbm_weight=current_qbm_weight,
                    temperature=temperature,
                    use_negative_phase=False,
                    persistent=False,
                    return_components=True,
                )
                test_loss_value = test_loss.item()
            else:
                test_loss_value = float("nan")

        train_loss = train_total / n_train
        loss_rows.append([train_loss, vs_loss.item(), test_loss_value])
        validation_score = vs_components["recon"].item()
        if validation_score < best_vs:
            best_vs = validation_score
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            print(
                "Epoch %s - LS %.4f VS %.4f | recon %.4f qbm %.4f | "
                "lambda %.4f temp %.3f trajectory %.4f gamma %.3f replica-disagreement %.3f"
                % (
                    epoch,
                    train_loss,
                    vs_loss.item(),
                    train_recon / n_train,
                    train_qbm / n_train,
                    current_qbm_weight,
                    temperature,
                    train_trajectory / n_train,
                    train_gamma / n_train,
                    train_replica_disagreement / n_train,
                )
            )
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping at epoch %s (best VS reconstruction %.6f)" % (epoch, best_vs))
            break
    return np.asarray(loss_rows), best_model, model


def build_qbm_vae_scenarios(n_s, x, y_scaler, model, sampler=None, max=1, gpu=True, tag=None,
                            non_null_indexes=None, beta=1.0, scale_multiplier=1.0,
                            observation_noise=True, gibbs_steps=50, day_batch_size=50):
    if sampler is not None and hasattr(sampler, "sample_ising_batch") and not model.uses_path_integral():
        return _build_batched_qbm_vae_scenarios(
            n_s=n_s,
            x=x,
            y_scaler=y_scaler,
            model=model,
            sampler=sampler,
            max_value=max,
            tag=tag,
            non_null_indexes=non_null_indexes,
            beta=beta,
            scale_multiplier=scale_multiplier,
            observation_noise=observation_noise,
            day_batch_size=day_batch_size,
        )
    scenarios = []
    for i in range(x.shape[0]):
        samples = model.sample(
            n_s=n_s,
            x_cond=x[i],
            sampler=sampler,
            beta=beta,
            scale_multiplier=scale_multiplier,
            observation_noise=observation_noise,
            gibbs_steps=gibbs_steps,
        )
        samples = y_scaler.inverse_transform(samples)
        samples = np.clip(samples, 0, max)
        if tag == "pv" and non_null_indexes is not None:
            rebuilt = np.zeros((n_s, 24))
            rebuilt[:, non_null_indexes] = samples
            samples = rebuilt
        scenarios.append(samples.transpose())
    return np.concatenate(scenarios, axis=0)


def _build_batched_qbm_vae_scenarios(n_s, x, y_scaler, model, sampler, max_value, tag,
                                     non_null_indexes, beta, scale_multiplier,
                                     observation_noise, day_batch_size):
    scenario_chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], day_batch_size):
            x_chunk = np.asarray(x[start:start + day_batch_size], dtype=np.float32)
            context_days = torch.tensor(x_chunk, dtype=torch.float32, device=model.device)
            h, j = model.qbm_params(context_days if model.conditional_qbm else None)
            if h.ndim == 1:
                h = h.unsqueeze(0).expand(context_days.shape[0], -1)
            result = sampler.sample_ising_batch(
                h.detach().cpu().numpy(),
                j.detach().cpu().numpy(),
                num_reads=n_s,
                beta=beta,
            )
            z = torch.tensor(
                spin_to_binary(result.samples).reshape(-1, model.latent_s),
                dtype=torch.float32,
                device=model.device,
            )
            context = context_days.repeat_interleave(n_s, dim=0)
            mean, log_scale, rho = model.decode_temporal_parameters(z, context)
            if observation_noise:
                noise = scale_multiplier * torch.randn_like(mean)
                if getattr(model, "decoder_covariance", "diagonal") == "ar1":
                    residual = sample_standardized_ar_residuals(
                        noise, torch.exp(log_scale), rho
                    )
                else:
                    residual = torch.exp(log_scale) * noise
                samples = mean + residual
            else:
                samples = mean
            samples = samples.cpu().numpy()
            samples = y_scaler.inverse_transform(samples)
            samples = np.clip(samples, 0, max_value)
            n_days = x_chunk.shape[0]
            samples = samples.reshape(n_days, n_s, -1)
            if tag == "pv" and non_null_indexes is not None:
                rebuilt = np.zeros((n_days, n_s, 24), dtype=np.float64)
                rebuilt[:, :, non_null_indexes] = samples
                samples = rebuilt
            scenario_chunks.append(samples.transpose(0, 2, 1).reshape(n_days * 24, n_s))
    return np.concatenate(scenario_chunks, axis=0)


def _build_mlp(sizes):
    layers = []
    for in_features, out_features in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*layers)


def temporal_ar_innovations(residuals, rho):
    """Map temporally correlated residuals to AR(1) innovations."""
    innovations = residuals.clone()
    innovations[:, 1:] = residuals[:, 1:] - rho[:, 1:] * residuals[:, :-1]
    return innovations


def sample_temporal_ar_residuals(innovations, rho):
    """Generate residual paths from AR(1) innovations and conditional rho."""
    residuals = torch.zeros_like(innovations)
    residuals[:, 0] = innovations[:, 0]
    for period in range(1, innovations.shape[1]):
        residuals[:, period] = (
            rho[:, period] * residuals[:, period - 1] + innovations[:, period]
        )
    return residuals


def standardized_ar_innovations(residuals, marginal_scale, rho):
    """Whiten residuals under a variance-preserving heteroscedastic AR(1)."""
    standardized = residuals / marginal_scale
    innovation_scale = torch.sqrt(torch.clamp(1.0 - rho ** 2, min=1e-6))
    innovations = standardized.clone()
    innovations[:, 1:] = (
        standardized[:, 1:] - rho[:, 1:] * standardized[:, :-1]
    ) / innovation_scale[:, 1:]
    correlation_log_det = torch.zeros_like(residuals)
    correlation_log_det[:, 1:] = torch.log(innovation_scale[:, 1:])
    return innovations, correlation_log_det


def sample_standardized_ar_residuals(noise, marginal_scale, rho):
    """Sample AR(1) residuals while preserving each period's marginal scale."""
    innovation_scale = torch.sqrt(torch.clamp(1.0 - rho ** 2, min=1e-6))
    standardized = torch.zeros_like(noise)
    standardized[:, 0] = noise[:, 0]
    for period in range(1, noise.shape[1]):
        standardized[:, period] = (
            rho[:, period] * standardized[:, period - 1]
            + innovation_scale[:, period] * noise[:, period]
        )
    return marginal_scale * standardized
