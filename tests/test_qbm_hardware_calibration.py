import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from GEFcom2014.models.QBM_VAE.calibrate_hardware_temperature import (
    evaluate_gain_candidate,
    select_gain_candidate,
    scale_payload,
)
from GEFcom2014.models.QBM_VAE.calibration import (
    fit_shared_effective_temperature_pseudolikelihood,
)
from GEFcom2014.models.QBM_VAE.compare_hardware_responses import (
    compare_response_pair,
    paired_improvement_statistics,
)
from GEFcom2014.models.QBM_VAE.export_ising_instances import (
    append_anchor_conditions,
    select_instance_rows,
)
from GEFcom2014.models.QBM_VAE.import_hardware_responses import (
    import_responses,
)
from GEFcom2014.models.QBM_VAE.prepare_bosonic_submission import (
    prepare_submission,
)


def draw_independent_samples(h, beta, reads, seed):
    rng = np.random.default_rng(seed)
    probability_up = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
    return np.where(rng.random((reads, h.size)) < probability_up, 1, -1)


def test_shared_effective_temperature_recovers_known_beta():
    true_beta = 1.35
    problems = []
    for seed in range(4):
        h = np.random.default_rng(seed).normal(0.0, 0.7, size=6)
        j = np.zeros((6, 6))
        samples = draw_independent_samples(h, true_beta, 8000, seed + 10)
        problems.append((h, j, samples))

    result = fit_shared_effective_temperature_pseudolikelihood(
        problems,
        beta_grid=np.linspace(0.8, 1.8, 101),
        bootstrap_repetitions=200,
        seed=3,
    )

    assert abs(result["beta_eff"] - true_beta) < 0.06
    assert result["beta_eff_ci_2.5"] <= true_beta
    assert result["beta_eff_ci_97.5"] >= true_beta


def test_context_anchor_conditions_match_training_transform(tmp_path):
    raw_x = np.array([[10.0, 20.0, 3.0], [20.0, 40.0, 4.0]])
    x_scaled = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    y_scaler = StandardScaler().fit(
        np.array([[0.0, 10.0], [20.0, 50.0]])
    )
    model = SimpleNamespace(decoder_anchor=True, in_size=2)

    conditions = append_anchor_conditions(
        model,
        {"anchor_source": "context"},
        tmp_path / "unused.pt",
        x_scaled,
        raw_x,
        y_scaler,
        device=None,
    )

    expected_anchor = y_scaler.transform(raw_x[:, :2])
    np.testing.assert_allclose(conditions[:, :3], x_scaled)
    np.testing.assert_allclose(conditions[:, 3:], expected_anchor)


def test_wind_instance_selection_is_balanced_across_zones():
    raw_x = np.zeros((50, 12))
    for zone in range(10):
        raw_x[zone * 5:(zone + 1) * 5, -10 + zone] = 1.0

    rows = select_instance_rows(
        "wind",
        raw_x,
        num_instances=20,
        strategy="stratified",
        seed=4,
    )
    selected_zones = np.argmax(raw_x[rows, -10:], axis=1)

    np.testing.assert_array_equal(
        np.bincount(selected_zones, minlength=10),
        np.full(10, 2),
    )


def test_payload_scaling_records_vs_calibration():
    payload = {
        "h": [0.5, -0.25],
        "couplings": [{"i": 0, "j": 1, "value": 0.4}],
    }
    calibration = {"target_beta": 1.0, "beta_eff": 0.5}

    scaled = scale_payload(payload, 2.0, calibration)

    assert scaled["h"] == [1.0, -0.5]
    assert scaled["couplings"][0]["value"] == 0.8
    assert scaled["temperature_calibration"]["fit_split"] == "VS"


def test_hardware_payload_cannot_be_rescaled_after_quantization():
    payload = {
        "h": [0.5, -0.25],
        "couplings": [{"i": 0, "j": 1, "value": 0.4}],
        "hardware_matrix": [[0, 1], [1, 0]],
    }
    with pytest.raises(ValueError, match="rebuild from the original h,J"):
        scale_payload(payload, 2.0, {"target_beta": 1.0, "beta_eff": 0.5})


def test_gain_candidate_selection_is_vs_only_and_deterministic(tmp_path):
    problem_path = tmp_path / "problem.npz"
    h = np.array([0.3, -0.2])
    j = np.array([[0.0, 0.1], [0.1, 0.0]])
    np.savez_compressed(problem_path, h=h, J=j)
    responses = tmp_path / "responses"
    responses.mkdir()
    rng = np.random.default_rng(8)
    np.savez_compressed(
        responses / "example_000.npz",
        samples=rng.choice([-1, 1], size=(30, 2)).astype(np.int8),
    )
    manifest = tmp_path / "vs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "example_000",
                    "path": str(problem_path),
                    "split": "VS",
                }
            ]
        ),
        encoding="utf-8",
    )
    candidate = evaluate_gain_candidate(
        {"gain": 11.0, "vs_manifest": str(manifest), "responses_dir": str(responses)},
        target_beta=1.0,
    )
    assert candidate["hardware_gain"] == 11.0
    assert candidate["n_instances"] == 1
    assert select_gain_candidate([candidate])["hardware_gain"] == 11.0


def test_paired_response_metrics_are_zero_for_identical_samples():
    h = np.array([0.2, -0.3])
    j = np.array([[0.0, 0.1], [0.1, 0.0]])
    samples = draw_independent_samples(h, 1.0, 500, 8)

    metrics = compare_response_pair(h, j, samples, samples.copy())

    assert metrics["energy_wasserstein"] == 0.0
    assert metrics["magnetization_mae"] == 0.0
    assert metrics["edge_moment_mae"] == 0.0


def test_paired_improvement_statistics_detect_consistent_reduction():
    baseline = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    candidate = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

    result = paired_improvement_statistics(
        baseline,
        candidate,
        repetitions=2000,
        seed=7,
    )

    assert result["mean_improvement"] > 2.0
    assert result["ci_2.5"] > 0.0
    assert result["paired_sign_flip_p"] < 0.1


def make_submission_source(tmp_path, split="VS"):
    payload = {
        "instance_id": "example_000",
        "expanded_n_bits": 3,
        "h": [0.2, -0.3, 0.1],
        "couplings": [{"i": 0, "j": 2, "value": 0.4}],
    }
    payload_path = tmp_path / "example_000.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            [
                {
                    "id": "example_000",
                    "split": split,
                    "platform_payload": str(payload_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    return source_manifest


def test_submission_bundle_is_hash_locked_and_deterministic(tmp_path):
    source_manifest = make_submission_source(tmp_path)
    output_dir = tmp_path / "submission"

    manifest_path, _, first_hash = prepare_submission(
        source_manifest,
        output_dir,
        stage="vs-calibration",
        requested_reads=10,
        max_bits=3,
        max_edges=1,
        coefficient_limit=0.5,
    )
    _, _, second_hash = prepare_submission(
        source_manifest,
        output_dir,
        stage="vs-calibration",
            requested_reads=10,
        max_bits=3,
        max_edges=1,
        coefficient_limit=0.5,
    )

    submission = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first_hash == second_hash
    assert submission["target_labels_included"] is False
    assert submission["instances"][0]["payload_sha256"]


def test_submission_rejects_coefficient_clipping(tmp_path):
    source_manifest = make_submission_source(tmp_path)

    with pytest.raises(ValueError, match="do not clip"):
        prepare_submission(
            source_manifest,
            tmp_path / "submission",
            stage="vs-calibration",
            requested_reads=10,
            coefficient_limit=0.2,
        )


def test_test_submission_requires_vs_calibration(tmp_path):
    source_manifest = make_submission_source(tmp_path, split="TEST")

    with pytest.raises(ValueError, match="not frozen from VS calibration"):
        prepare_submission(
            source_manifest,
            tmp_path / "submission",
            stage="test-evaluation",
            requested_reads=10,
        )


def test_response_import_expands_bitstring_counts(tmp_path):
    source_manifest = make_submission_source(tmp_path)
    submission_manifest, _, _ = prepare_submission(
        source_manifest,
        tmp_path / "submission",
        stage="vs-calibration",
        requested_reads=10,
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "example_000.json").write_text(
        json.dumps(
            {
                "instance_id": "example_000",
                "bitstrings": ["001", "110"],
                "counts": [7, 3],
                "job_id": "job-1",
                "latency_s": 0.5,
            }
        ),
        encoding="utf-8",
    )

    _, audit = import_responses(
        submission_manifest,
        raw_dir,
        tmp_path / "canonical",
    )

    with np.load(tmp_path / "canonical" / "example_000.npz") as response:
        samples = response["samples"]
    assert samples.shape == (10, 3)
    np.testing.assert_array_equal(samples[:7], np.tile([[-1, -1, 1]], (7, 1)))
    np.testing.assert_array_equal(samples[7:], np.tile([[1, 1, -1]], (3, 1)))
    assert audit["records"][0]["read_count_matches"] is True
