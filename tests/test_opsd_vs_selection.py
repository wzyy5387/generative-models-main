"""Synthetic unit fixtures only: no SDK, no hardware result or quota usage."""
import json
import pickle

import numpy as np
import pytest
import torch

from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE import kaiwu_13k_protocol as policy
from GEFcom2014.models.QBM_VAE import run_kaiwu_13k_sampling as driver
from GEFcom2014.models.QBM_VAE import opsd_vs_selection as scoring
from GEFcom2014.models.QBM_VAE.kaiwu_adapter import matrix_sha256
from tests.test_kaiwu_13k_protocol import _complete_vs_records


class ToyDecoder(torch.nn.Module):
    latent_s = 48
    cond_in = 57

    def decode_temporal_parameters(self, z, context):
        return context[:, -24:] + z[:, :24] * 0.1, torch.full_like(z[:, :24], -2), None


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    protocol = policy.load_protocol("configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json")
    protocol["vs"].update(instances=2, tasks=6, samples=600)
    protocol["budget"]["planned_samples"] = 9600
    protocol["gain_selection"]["quality_gate"].update(status="defined", policy_approved=True)
    protocol["gain_selection"]["quality_gate"]["thresholds"] = dict.fromkeys(
        ("energy_wasserstein", "effective_beta_error", "edge_moment_mae", "magnetization_mae"), 0.0)
    protocol["gain_selection"]["quality_gate"]["test_only"] = True
    rng = np.random.default_rng(19)
    x_ls, y_ls = rng.random((20, 33)), rng.random((20, 24))
    x_vs, y_vs = rng.random((2, 33)), rng.random((2, 24))
    data_path, model_path = tmp_path / "data.npz", tmp_path / "model.pickle"
    # Intentionally no TEST arrays: any access to TEST would fail this test.
    np.savez(data_path, x_ls=x_ls, y_ls=y_ls, x_vs=x_vs, y_vs=y_vs)
    model_path.write_bytes(pickle.dumps(ToyDecoder()))
    protocol["model"]["checkpoint"] = str(model_path)
    protocol["dataset"]["source"] = str(data_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol["_path"] = str(protocol_path)
    _, _, scaled, _, _, _, scaler = scale_data_multi(x_ls, y_ls, x_vs, y_vs)
    conditions = np.concatenate((scaled, scaler.transform(x_vs[:, :24])), axis=1).astype(np.float32)
    records = _complete_vs_records(tmp_path)
    h, j = np.full(48, 0.01), np.zeros((48, 48))
    j[0, 1] = j[1, 0] = 0.02
    manifests = []
    manifest_hashes = {}
    quality_entries = []
    for gain in (125, 150, 200):
        items = []
        for index in range(2):
            record = next(r for r in records if r["gain"] == gain and r["instance_id"] == "vs_%d" % index)
            problem, reference, matrix = [tmp_path / ("%s_%d_%d.npz" % (name, gain, index))
                                          for name in ("problem", "reference", "matrix")]
            np.savez(problem, h=h, J=j, x_cond=conditions[index])
            spins = rng.choice([-1, 1], size=(100, 48)).astype(np.int8)
            reference_samples = rng.choice([-1, 1], size=(100, 48)).astype(np.int8)
            values = np.zeros((49, 49), dtype=np.int8)
            np.savez(matrix, hardware_matrix=values)
            sampler = {"backend": "sa", "matrix_space": "hardware-quantized", "reads": 100,
                       "sweeps": 100, "seed": 2026 + index, "sampler_beta": 1 / gain,
                       "matrix_beta": 1 / gain, "logical_beta": 1.0}
            np.savez(reference, samples=reference_samples, backend="sa",
                     matrix_space="hardware-quantized", random_seed=2026 + index,
                     requested_reads=100, sweeps=100, sampler_beta=1 / gain,
                     matrix_beta=1 / gain, logical_beta=1.0, hardware_gain=gain,
                     hardware_matrix_sha256=matrix_sha256(values),
                     source_problem_sha256=policy.sha256_file(problem),
                     instance_id="vs_%d" % index)
            record.update(requested_reads=100, returned_reads=100, matrix_sha256=matrix_sha256(values))
            record.update(hardware_claim=False, physical_platform_used=False,
                          hardware_matrix_path=str(matrix),
                          hardware_matrix_file_sha256=policy.sha256_file(matrix),
                          reference_sample_path=str(reference),
                          reference_sample_sha256=policy.sha256_file(reference),
                          reference_sampler=sampler)
            np.savez(record["response_path"], samples=np.column_stack((spins, np.ones(100, dtype=np.int8))),
                     **{key: record[key] for key in ("instance_id", "task_id", "task_name", "matrix_sha256", "stage", "requested_reads", "returned_reads")},
                     hardware_gain=gain)
            record.update(source_problem_path=str(problem), source_problem_sha256=policy.sha256_file(problem),
                          response_file_sha256=policy.sha256_file(record["response_path"]))
            quality_entries.append({
                "gain": gain, "instance_id": record["instance_id"],
                "source_problem_path": str(problem),
                "source_problem_sha256": record["source_problem_sha256"],
                "hardware_matrix_path": str(matrix),
                "hardware_matrix_file_sha256": record["hardware_matrix_file_sha256"],
                "hardware_matrix_sha256": record["matrix_sha256"],
                "reference_sample_path": str(reference),
                "reference_sample_sha256": record["reference_sample_sha256"],
                "sampler": sampler,
            })
            items.append({"id": record["instance_id"], "hardware_gain": gain, "split": "VS",
                          "index": index, "export_index": index, "path": str(problem)})
        path = tmp_path / ("manifest_%d.json" % gain)
        path.write_text(json.dumps(items), encoding="utf-8")
        manifests.append(str(path))
        manifest_hashes[str(gain)] = policy.sha256_file(path)
    prep = {"protocol_sha256": policy.sha256_file(protocol_path),
            "protocol_version": protocol["protocol_version"],
            "model_checkpoint_sha256": policy.sha256_file(model_path),
            "dataset_sha256": policy.sha256_file(data_path), "vs_manifests": manifests,
            "manifest_hashes": {"vs_manifests": manifest_hashes}}
    (tmp_path / "preparation_manifest.json").write_text(json.dumps(prep), encoding="utf-8")
    quality_reference = {"protocol_version": protocol["protocol_version"],
                         "hardware_claim": False, "physical_platform_used": False,
                         "instances": quality_entries}
    quality_reference_path = tmp_path / "quality_reference_manifest.json"
    quality_reference_path.write_text(json.dumps(quality_reference), encoding="utf-8")
    prep["quality_reference_manifest_sha256"] = policy.sha256_file(quality_reference_path)
    (tmp_path / "preparation_manifest.json").write_text(json.dumps(prep), encoding="utf-8")
    identity = driver._preparation_identity(tmp_path, protocol)
    tasks = []
    for record in records:
        record.update(identity)
        tasks.append({**record, "reads": 100})
    monkeypatch.setattr(driver, "_vs_tasks", lambda preparation, protocol: tasks)
    kwargs = dict(required_gains=(125, 150, 200), expected_instances_per_gain=2,
                  expected_instance_ids=["vs_0", "vs_1"], expected_identity=identity,
                  expected_task_map={(float(t["gain"]), t["instance_id"]): driver._task_expected(t) for t in tasks},
                  quality_policy=protocol["gain_selection"]["quality_gate"], recompute_from_raw=True,
                  prediction_inputs={"preparation": tmp_path, "protocol": protocol},
                  require_physical=False)
    return records, kwargs, protocol


def test_raw_crps_freeze_and_test_revalidation_ignore_forged_scores(bundle, tmp_path):
    records, kwargs, protocol = bundle
    records[0]["raw_vs_crps"] = -999
    path = tmp_path / "real_gain_selection.json"
    frozen = policy.select_vs_gain(records, path, **kwargs)
    assert frozen["artifact_role"] == "offline_test_fixture"
    assert frozen["hardware_claim"] is False and frozen["physical_platform_used"] is False
    assert frozen["unlock_real_test"] is False
    candidates = frozen["candidate_metrics"]
    assert all(r["quality_status"] == "diagnostic_only" for r in candidates)
    assert all(r["diagnostic_threshold_status"] == "fail" for r in candidates)
    assert frozen["selected_gain"] == min(candidates, key=lambda r: (r["raw_vs_crps"], r["gain"]))["gain"]
    with pytest.raises(RuntimeError, match="physical VS responses"):
        driver._load_frozen_gain(tmp_path, protocol)


@pytest.mark.parametrize("mutation", ["missing", "mock", "bad_reads", "nan_samples",
                                       "missing_reference", "tampered_reference",
                                       "wrong_task_name", "changed_model", "changed_data", "changed_protocol"])
def test_invalid_inputs_cannot_freeze(bundle, tmp_path, mutation):
    records, kwargs, protocol = bundle
    if mutation == "missing":
        records.pop()
    elif mutation == "mock":
        records[0].update(hardware_claim=True, physical_platform_used=True)
    elif mutation == "bad_reads":
        records[0]["returned_reads"] = 99
    elif mutation == "nan_samples":
        path = records[0]["response_path"]
        with np.load(path) as archive:
            fields = dict(archive)
        fields["samples"] = fields["samples"].astype(float)
        fields["samples"][0, 0] = np.nan
        np.savez(path, **fields)
        records[0]["response_file_sha256"] = policy.sha256_file(path)
    elif mutation in ("missing_reference", "tampered_reference"):
        path = records[0]["reference_sample_path"]
        if mutation == "missing_reference":
            __import__("pathlib").Path(path).unlink()
        else:
            with open(path, "ab") as handle:
                handle.write(b"tampered")
    elif mutation == "wrong_task_name":
        records[0]["task_name"] = "wrong-protocol-task"
    else:
        path = {"changed_model": "model.pickle", "changed_data": "data.npz", "changed_protocol": "protocol.json"}[mutation]
        with (tmp_path / path).open("ab") as handle:
            handle.write(b" ")
    with pytest.raises((ValueError, RuntimeError)):
        policy.select_vs_gain(records, tmp_path / "not_created.json", **kwargs)
    assert not (tmp_path / "not_created.json").exists()


def test_ties_nonfinite_scores_and_explicit_approval(bundle):
    _, kwargs, _ = bundle
    q = kwargs["quality_policy"]
    assert policy.choose_vs_candidate([{"gain": 200, "raw_vs_crps": .1}, {"gain": 125, "raw_vs_crps": .1}], q)["gain"] == 125
    with pytest.raises(ValueError, match="finite Raw VS CRPS"):
        policy.choose_vs_candidate([{"gain": 125, "raw_vs_crps": float("nan")}], q)
    q["policy_approved"] = False
    with pytest.raises(ValueError, match="not approved"):
        policy.validate_quality_policy(q)
    q.update(policy_approved=True, test_only=True)
    with pytest.raises(ValueError, match="test-only"):
        policy.validate_quality_policy(q)


def test_nonfinite_decoding_hard_fails(bundle, monkeypatch):
    records, kwargs, _ = bundle
    monkeypatch.setattr(scoring, "decode_platform_spins", lambda *a, **k: np.full((100, 24), np.nan))
    with pytest.raises(ValueError, match="Non-finite"):
        scoring.recompute_vs_predictions(records, require_physical=False,
                                         **kwargs["prediction_inputs"])


def test_new_policy_canary_cannot_bypass_policy_approval(tmp_path):
    from types import SimpleNamespace
    protocol = policy.load_protocol("configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json")
    protocol["gain_selection"]["quality_gate"].update(
        status="defined_pending_manual_approval", policy_approved=False)
    protocol_path = tmp_path / "unapproved_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    args = SimpleNamespace(
        protocol=protocol_path,
        preparation=tmp_path, output_dir=tmp_path / "responses",
        checkpoint_dir=tmp_path / "checkpoints", execute=True,
        canary=True, confirm_canary=True)
    with pytest.raises(ValueError, match="not approved"):
        driver.run(args)
    assert not args.output_dir.exists()
