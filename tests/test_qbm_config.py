import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.qbm_vae import (
    build_graph_mask,
    load_frozen_graph_mask,
    load_graph_mask_file,
    parse_args,
)
from GEFcom2014.models.QBM_VAE.ising import stable_partial_correlation_graph


def test_qbm_json_config_is_loaded_and_cli_overrides_it(tmp_path, monkeypatch):
    config_path = tmp_path / "qbm.json"
    config_path.write_text(
        json.dumps(
            {
                "tag": "wind",
                "epochs": 80,
                "learned_forecast_anchor": True,
                "run_label": "lanchor",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["qbm_vae", "--config", str(config_path), "--epochs", "12", "--seed", "2"],
    )

    args = parse_args()

    assert args.tag == "wind"
    assert args.epochs == 12
    assert args.seed == 2
    assert args.learned_forecast_anchor is True
    assert args.run_label == "lanchor"


def test_qbm_json_config_rejects_unknown_keys(tmp_path, monkeypatch):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps({"not_an_argument": 1}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["qbm_vae", "--config", str(config_path)])

    with pytest.raises(ValueError, match="not_an_argument"):
        parse_args()


def test_random_graph_matches_temporal_graph_edge_count_exactly():
    rng = np.random.default_rng(4)
    targets = rng.normal(size=(80, 24))
    temporal = build_graph_mask(
        targets, latent_s=48, graph_mode="temporal-mi", top_k=3, seed=7
    )
    random_graph = build_graph_mask(
        targets, latent_s=48, graph_mode="random", top_k=3, seed=7
    )

    assert np.count_nonzero(np.triu(random_graph, k=1)) == np.count_nonzero(
        np.triu(temporal, k=1)
    )
    np.testing.assert_allclose(
        np.sort(random_graph[np.triu_indices(48, k=1)][random_graph[np.triu_indices(48, k=1)] != 0]),
        np.sort(temporal[np.triu_indices(48, k=1)][temporal[np.triu_indices(48, k=1)] != 0]),
    )
    np.testing.assert_array_equal(random_graph, random_graph.T)
    np.testing.assert_array_equal(np.diag(random_graph), np.zeros(48))
    np.testing.assert_array_equal(
        random_graph,
        build_graph_mask(
            targets, latent_s=48, graph_mode="random", top_k=3, seed=7
        ),
    )


def test_explicit_random_graph_density_uses_nearest_exact_edge_count():
    targets = np.zeros((20, 4))
    graph = build_graph_mask(
        targets,
        latent_s=8,
        graph_mode="random",
        top_k=1,
        random_graph_density=0.25,
        seed=3,
    )

    assert np.count_nonzero(np.triu(graph, k=1)) == round(0.25 * 28)


def test_stable_residual_graph_is_deterministic_and_lag_constrained():
    rng = np.random.default_rng(12)
    base = rng.normal(size=(240, 4))
    residuals = np.column_stack(
        [base[:, 0], base[:, 0] + 0.05 * base[:, 1], base[:, 2], base[:, 2] + 0.05 * base[:, 3]]
    )
    graph = build_graph_mask(
        residuals,
        residual_train=residuals,
        latent_s=8,
        graph_mode="stable-residual-mi",
        top_k=1,
        graph_bootstrap_repetitions=30,
        graph_stability_threshold=0.7,
        graph_max_lag=1,
        seed=2026,
    )

    np.testing.assert_array_equal(graph, graph.T)
    assert np.count_nonzero(np.triu(graph, k=1)) > 0
    for source, target in zip(*np.where(np.triu(graph, k=1) != 0)):
        assert abs(source // 2 - target // 2) <= 1
    np.testing.assert_array_equal(
        graph,
        build_graph_mask(
            residuals,
            residual_train=residuals,
            latent_s=8,
            graph_mode="stable-residual-mi",
            top_k=1,
            graph_bootstrap_repetitions=30,
            graph_stability_threshold=0.7,
            graph_max_lag=1,
            seed=2026,
        ),
    )


def test_stable_residual_random_matches_reference_support_and_weights():
    rng = np.random.default_rng(8)
    residuals = rng.normal(size=(200, 6))
    residuals[:, 1:] += 0.8 * residuals[:, :-1]
    kwargs = dict(
        y_train=residuals,
        residual_train=residuals,
        latent_s=12,
        top_k=2,
        graph_bootstrap_repetitions=20,
        graph_stability_threshold=0.5,
        graph_max_lag=2,
        seed=77,
    )
    learned = build_graph_mask(graph_mode="stable-residual-mi", **kwargs)
    random_graph = build_graph_mask(graph_mode="stable-residual-random", **kwargs)
    upper = np.triu_indices(12, k=1)
    learned_weights = np.sort(learned[upper][learned[upper] != 0])
    random_weights = np.sort(random_graph[upper][random_graph[upper] != 0])

    np.testing.assert_allclose(random_weights, learned_weights)


def test_stable_residual_graph_requires_anchor_residuals():
    with pytest.raises(ValueError, match="requires decoder-anchor LS residuals"):
        build_graph_mask(
            np.zeros((20, 4)),
            latent_s=8,
            graph_mode="stable-residual-mi",
            top_k=1,
        )


def test_frozen_graph_mask_loader_validates_and_returns_the_saved_mask(tmp_path):
    mask = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    path = tmp_path / "model.pickle"
    with path.open("wb") as handle:
        pickle.dump(SimpleNamespace(graph_mask=mask), handle)

    np.testing.assert_array_equal(load_frozen_graph_mask(path, latent_s=2), mask)

    with pytest.raises(ValueError, match="expected"):
        load_frozen_graph_mask(path, latent_s=3)


def test_numpy_graph_mask_loader_validates_the_artifact(tmp_path):
    mask = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float32)
    path = tmp_path / "mask.npy"
    np.save(path, mask)

    np.testing.assert_array_equal(load_graph_mask_file(path, latent_s=2), mask)


def test_stable_partial_correlation_graph_recovers_repeated_dependence():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(300, 6))
    values[:, 1] = values[:, 0] + 0.05 * rng.normal(size=300)
    values[:, 3] = values[:, 2] + 0.05 * rng.normal(size=300)
    graph = stable_partial_correlation_graph(
        values,
        top_k=1,
        bootstrap_repetitions=30,
        stability_threshold=0.7,
        covariance_ridge=0.1,
        seed=9,
    )

    assert graph[0, 1] > 0
    assert graph[2, 3] > 0
    np.testing.assert_array_equal(graph, graph.T)
