import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.latent_sensitivity_diagnostic import (
    frozen_vs_condition,
    summarize_flip_sensitivity,
)


def test_sensitivity_uses_scaler_scale_and_dataset_mw_scale_per_flip():
    # Axes: one source sample x three flipped bits x two forecast hours.
    base_mean = np.zeros((1, 2))
    changed_mean = np.array([[[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]]])
    base_scale = np.ones((1, 2))
    changed_scale = np.array([[[2.0, 1.0], [1.0, 3.0], [2.0, 3.0]]])

    result = summarize_flip_sensitivity(
        base_mean, changed_mean, base_scale, changed_scale,
        scaler_scale=np.array([2.0, 3.0]), scale_mw=10.0)

    expected_mean_rms = np.array([np.sqrt(200.0), np.sqrt(1800.0), np.sqrt(2000.0)])
    expected_noise_rms = np.sqrt(650.0)
    expected_scale_change_rms = np.array([np.sqrt(0.5), np.sqrt(2.0), np.sqrt(2.5)])
    np.testing.assert_allclose(result["mean_shift_rms_mw"][0], expected_mean_rms)
    np.testing.assert_allclose(
        result["observation_sigma_rms_mw_by_sample"], [expected_noise_rms])
    np.testing.assert_allclose(
        result["mean_shift_to_observation_sigma"][0], expected_mean_rms / expected_noise_rms)
    np.testing.assert_allclose(
        result["relative_observation_scale_change_rms"][0], expected_scale_change_rms)

    summary = result["summary"]
    np.testing.assert_allclose(summary["mean_shift_rms_mw"]["median"], np.median(expected_mean_rms))
    np.testing.assert_allclose(summary["mean_shift_rms_mw"]["p95"], np.quantile(expected_mean_rms, .95))
    np.testing.assert_allclose(
        summary["relative_observation_scale_change_rms"]["median"],
        np.median(expected_scale_change_rms))
    np.testing.assert_allclose(
        summary["relative_observation_scale_change_rms"]["p95"],
        np.quantile(expected_scale_change_rms, .95))


def test_sensitivity_rejects_invalid_power_scale():
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_flip_sensitivity(
            np.zeros((1, 2)), np.ones((1, 1, 2)), np.ones((1, 2)),
            np.ones((1, 1, 2)), np.ones(2), scale_mw=float("nan"))


def test_frozen_condition_uses_source_index_not_export_order(tmp_path):
    conditions = np.arange(90 * 3, dtype=np.float32).reshape(90, 3)
    frozen_path = tmp_path / "condition.npz"
    np.savez(frozen_path, x_cond=conditions[65])
    from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import sha256_file
    task = {
        "index": 65, "export_index": 0, "date": "2019-01-06 00:00:00",
        "source_problem_path": str(frozen_path),
        "source_problem_sha256": sha256_file(frozen_path),
    }
    condition, error = frozen_vs_condition(task, conditions, "2018-11-02")
    np.testing.assert_array_equal(condition, conditions[65])
    assert error == 0.0

    # A frozen file carrying the export-order row is rejected for this date.
    np.savez(frozen_path, x_cond=conditions[0])
    task["source_problem_sha256"] = sha256_file(frozen_path)
    with pytest.raises(ValueError, match="does not match the indexed full VS condition"):
        frozen_vs_condition(task, conditions, "2018-11-02")

    np.savez(frozen_path, x_cond=conditions[65])
    task["source_problem_sha256"] = sha256_file(frozen_path)
    task["date"] = "2018-11-02 00:00:00"
    with pytest.raises(ValueError, match="date does not match"):
        frozen_vs_condition(task, conditions, "2018-11-02")
