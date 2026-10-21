import json
import sys

import numpy as np
import pytest
import torch

from GEFcom2014.models.QBM_VAE.qbm_vae import (
    GENERATION_SPLIT_SEED_OFFSETS,
    build_generation_seed_metadata,
    parse_args,
)
from GEFcom2014.models.QBM_VAE.samplers import SimulatedAnnealingSampler
from GEFcom2014.models.QBM_VAE.utils_qbm_vae import (
    ConditionalQBMVAE,
    build_qbm_vae_scenarios,
)


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values)


class _ScalarOnlySampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def sample_ising(self, *args, **kwargs):
        return self.sampler.sample_ising(*args, **kwargs)


def _fixed_model():
    torch.manual_seed(82)
    return ConditionalQBMVAE(
        latent_s=8,
        cond_in=24,
        in_size=24,
        enc_w=8,
        enc_l=1,
        dec_w=8,
        dec_l=1,
        conditional_qbm=True,
        graph_mask=np.ones((8, 8), dtype=np.float32) - np.eye(8, dtype=np.float32),
        gpu=False,
    )


def _generate(model, generation_seed, batched=True):
    sampler = SimulatedAnnealingSampler(
        sweeps=8, beta_start=0.1, beta_end=2.0, seed=generation_seed
    )
    if not batched:
        sampler = _ScalarOnlySampler(sampler)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(generation_seed)
    return build_qbm_vae_scenarios(
        n_s=32,
        x=np.stack([np.linspace(i / 10, 1 + i / 10, 24) for i in range(3)]).astype(np.float32),
        y_scaler=_IdentityScaler(),
        model=model,
        sampler=sampler,
        gpu=False,
        beta=1.0,
        observation_noise=True,
        day_batch_size=2,
        generator=generator,
    )


@pytest.mark.parametrize("batched", [True, False])
def test_generation_is_independent_of_global_rng_and_prior_calls(batched):
    model = _fixed_model()
    weights_before = {key: value.clone() for key, value in model.state_dict().items()}
    torch.rand(3)
    first = _generate(model, generation_seed=17, batched=batched)
    torch.rand(997)
    np.random.random(31)
    second = _generate(model, generation_seed=17, batched=batched)
    np.testing.assert_array_equal(first, second)
    third = _generate(model, generation_seed=17, batched=batched)
    np.testing.assert_array_equal(second, third)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights_before[key])


@pytest.mark.parametrize("batched", [True, False])
def test_non_degenerate_generation_changes_with_seed(batched):
    model = _fixed_model()
    first = _generate(model, generation_seed=17, batched=batched)
    second = _generate(model, generation_seed=18, batched=batched)
    assert not np.array_equal(first, second)


def test_paired_seed_does_not_make_different_models_share_scenarios():
    first_model = _fixed_model()
    second_model = _fixed_model()
    with torch.no_grad():
        second_model.dec[-1].bias.add_(0.2)
    first = _generate(first_model, generation_seed=17)
    second = _generate(second_model, generation_seed=17)
    assert not np.array_equal(first, second)


def test_split_seed_mapping_is_independent_of_execution_order():
    model = _fixed_model()

    def generate_split(split):
        split_seed = 20261020 + GENERATION_SPLIT_SEED_OFFSETS[split]
        return _generate(model, generation_seed=split_seed)

    forward = {split: generate_split(split) for split in ("VS", "TEST")}
    reverse = {split: generate_split(split) for split in ("TEST", "VS")}
    for split in ("VS", "TEST"):
        np.testing.assert_array_equal(forward[split], reverse[split])


def test_generation_seed_cli_is_optional_and_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qbm_vae"])
    assert parse_args().generation_seed is None
    monkeypatch.setattr(sys, "argv", ["qbm_vae", "--generation-seed", "2"])
    assert parse_args().generation_seed == 2


def test_explicit_generation_seed_metadata_separates_base_and_split_seeds():
    metadata = build_generation_seed_metadata(training_seed=2, generation_seed=7)

    assert metadata["generation_sampler_seed"] != 2
    assert metadata["generation_sampler_seed_role"].startswith("base_seed")
    assert metadata["generation_sampler_seeds"] == {"VS": 7, "TEST": 1000010}


def test_legacy_generation_seed_metadata_remains_backward_compatible():
    metadata = build_generation_seed_metadata(training_seed=2)

    assert metadata["generation_sampler_seed"] == 2
    assert metadata["generation_split_seeds"] is None


@pytest.mark.parametrize("label", ["matched_temporal", "matched_random"])
def test_minimal_graph_config_accepts_fixed_generation_policy(monkeypatch, tmp_path, label):
    config = tmp_path / (label + ".json")
    config.write_text(json.dumps({
        "run_label": label,
        "sampler": "sa",
        "scenario_splits": ["VS"],
        "n_scenarios": 100,
        "generation_split_seed_offsets": {"VS": 0, "TEST": 1000003},
        "generation_sa_settings": {
            "sweeps": 200,
            "beta_start": 0.1,
            "constructor_beta_end": 2.0,
            "beta_argument": 1.0,
            "effective_beta_end": 1.0,
        },
        "generation_observation_noise": True,
        "generation_day_batch_size": 50,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["qbm_vae", "--config", str(config)])
    args = parse_args()
    assert args.run_label == label
    assert args.sampler == "sa"
    assert args.scenario_splits == ["VS"]
    assert args.n_scenarios == 100
    assert args.generation_seed is None


def test_legacy_generation_call_without_generator_still_works():
    model = _fixed_model()
    result = build_qbm_vae_scenarios(
        n_s=3,
        x=np.zeros((2, 24), dtype=np.float32),
        y_scaler=_IdentityScaler(),
        model=model,
        sampler=SimulatedAnnealingSampler(sweeps=2, seed=3),
        gpu=False,
    )
    assert result.shape == (48, 3)
