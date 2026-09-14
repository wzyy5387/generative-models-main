import json

import numpy as np
import torch
from diffusers import DDPMScheduler

from GEFcom2014.models.run_strong_baselines import scenario_paths
from GEFcom2014.models.strong_probabilistic_baselines import (
    DeterministicConditioner,
    PatchDenoiser,
    parse_args,
    sample_d3u,
    sample_treeffuser,
)
from GEFcom2014.forecast_quality import unified_postprocessing


def test_patch_denoiser_preserves_nondivisible_trajectory_shape_and_gradients():
    model = PatchDenoiser(
        data_dim=10,
        condition_dim=12,
        hidden_dim=16,
        patch_size=4,
        depth=2,
        heads=4,
        time_dim=16,
    )
    values = torch.randn(3, 10, requires_grad=True)
    output = model(values, torch.tensor([0, 1, 2]), torch.randn(3, 12))
    assert output.shape == values.shape
    output.square().mean().backward()
    assert torch.isfinite(values.grad).all()


def test_d3u_sampler_returns_joint_daily_ensemble():
    conditioner = DeterministicConditioner(5, 8, hidden_dim=6, layers=1)
    denoiser = PatchDenoiser(
        data_dim=8,
        condition_dim=14,
        hidden_dim=16,
        patch_size=4,
        depth=1,
        heads=4,
        time_dim=16,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=4,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False,
    )
    samples = sample_d3u(
        denoiser,
        conditioner,
        scheduler,
        np.zeros((2, 5), dtype=np.float32),
        n_scenarios=3,
        inference_steps=2,
        device=torch.device("cpu"),
        seed=7,
        day_batch_size=2,
    )
    assert samples.shape == (2, 3, 8)
    assert np.isfinite(samples).all()


def test_treeffuser_sample_axis_order_is_normalized():
    class FakeTreeffuser:
        def sample(self, context, n_samples, **kwargs):
            return np.zeros((n_samples, context.shape[0], 4), dtype=np.float32)

    class Args:
        tree_n_parallel = 2
        tree_sampling_steps = 3
        tree_verbose = 0

    samples = sample_treeffuser(
        FakeTreeffuser(), np.zeros((5, 2), dtype=np.float32), 7, Args(), 11
    )
    assert samples.shape == (5, 7, 4)


def test_paper_config_can_supply_required_model(tmp_path):
    config = tmp_path / "baseline.json"
    config.write_text(
        json.dumps({"model": "d3u", "tag": "wind", "scenario_splits": ["VS"]}),
        encoding="utf-8",
    )
    args = parse_args(["--config", str(config), "--seed", "4"])
    assert args.model == "d3u"
    assert args.seed == 4
    assert args.scenario_splits == ["VS"]


def test_five_seed_runner_uses_unified_scenario_names():
    validation, test = scenario_paths("wind", "treeffuser", 4)
    assert validation.name == "scenarios_wind_Treeffuser_4_100_VS.pickle"
    assert test.name == "scenarios_wind_Treeffuser_4_100_TEST.pickle"


def test_unified_postprocessing_discovers_new_strong_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(unified_postprocessing, "ROOT_DIR", tmp_path)
    expected = {
        "D3U (NWP-adapted) (seed 0)",
        "Treeffuser (seed 0)",
    }
    for model, label in (("d3u", "D3UAdapted"), ("treeffuser", "Treeffuser")):
        directory = tmp_path / "export" / (model + "_wind")
        directory.mkdir(parents=True)
        for split in ("VS", "TEST"):
            (directory / ("scenarios_wind_%s_0_100_%s.pickle" % (label, split))).touch()
    pairs, _ = unified_postprocessing.discover_default_pairs("wind")
    assert expected.issubset({pair.name for pair in pairs})
