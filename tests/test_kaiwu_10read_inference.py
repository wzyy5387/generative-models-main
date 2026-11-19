import json
import numpy as np
import pytest
import torch
from argparse import Namespace

from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import validate_protocol
from GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios import (
    decode_latent_samples_with_paired_noise,
    decode_platform_spins,
)
from GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling import (
    _restrict_canary_tasks, _restrict_dry_run_task,
)


def test_gain_aggregation_handles_unavailable_read_sensitivity(monkeypatch):
    from GEFcom2014.models.QBM_VAE import kaiwu_13k_protocol, opsd_vs_selection

    metrics = {key: 0.0 for key in (
        "energy_wasserstein", "effective_beta_error", "edge_moment_mae",
        "magnetization_mae", "unique_state_fraction")}
    records = [dict(metrics, gain=125, instance_id="ten"),
               dict(metrics, gain=125, instance_id="hundred")]
    predictions = [
        {"instance_id": "ten", "gain": 125, "metrics": {"CRPS": 0.1},
         "prediction_read_sensitivity": None},
        {"instance_id": "hundred", "gain": 125, "metrics": {"CRPS": 0.3},
         "prediction_read_sensitivity": {"sample_overlap": 50}},
    ]
    monkeypatch.setattr(opsd_vs_selection, "recompute_vs_predictions",
                        lambda *args, **kwargs: predictions)
    result = kaiwu_13k_protocol.aggregate_gain_metrics(records, prediction_inputs={})[0]
    assert result["raw_vs_crps"] == pytest.approx(0.2)
    assert result["prediction_read_sensitivity_diagnostics"] == [
        {"instance_id": "hundred", "sample_overlap": 50}]


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values)


class _SmallDecoder:
    latent_s = 2
    cond_in = 1
    device = torch.device("cpu")
    decoder_covariance = "diagonal"

    def decode_temporal_parameters(self, z, context):
        mean = torch.full((len(z), 24), 0.5, dtype=z.dtype, device=z.device)
        log_scale = torch.full_like(mean, -5.0)
        rho = torch.zeros(len(z), dtype=z.dtype, device=z.device)
        return mean, log_scale, rho


def _protocol(reads):
    payload = {
        "method_name": "FA-BM-VAE",
        "hardware": {
            "candidate_gains": [125, 150, 200],
            "reads_per_instance": reads,
            "platform_limits": {
                "int8_coefficient_range": [-128, 127],
                "int8_range_verified": True,
                "machine_profile": {
                    "name": "SPQC-550", "max_variables": 550,
                    "full_connectivity": True, "coupling_precision": "int8",
                },
            },
        },
        "dataset": {"selection_seed": 2026, "test_peeking": False},
        "vs": {"instances": 10, "tasks": 30},
        "test": {"instances": 90, "tasks": 90},
        "budget": {
            "planned_samples": 120 * reads,
            "application_upper_limit": 15000,
        },
        "gain_selection": {
            "read_convergence": [50, 100] if reads == 100 else [],
            "quality_gate": {
                "default_action": "lock TEST until a pre-defined quality policy is frozen",
            },
        },
    }
    if reads == 10:
        payload["inference"] = {
            "latent_samples_per_date": 10,
            "prediction_curves_per_date": 100,
            "noise_pairing": "same_seeded_row_major_100x24_for_all_methods",
        }
    return payload


def test_ten_latent_reads_expand_to_100_paired_noise_curves_and_replay_100_read_path():
    model = _SmallDecoder()
    scaler = _IdentityScaler()
    condition = np.zeros(1, dtype=np.float32)
    one_state = np.ones((100, 2), dtype=np.int8)
    original = decode_platform_spins(model, one_state, condition, scaler, seed=2026)
    replay = decode_latent_samples_with_paired_noise(
        model, one_state[:10], condition, scaler, prediction_curves=100, seed=2026,
    )
    assert replay.shape == (100, 24)
    np.testing.assert_array_equal(replay, original)


def test_different_methods_receive_same_seeded_noise_in_fixed_row_mapping():
    model = _SmallDecoder()
    scaler = _IdentityScaler()
    condition = np.zeros(1, dtype=np.float32)
    samples = np.array([[1, -1]] * 10, dtype=np.int8)
    first = decode_latent_samples_with_paired_noise(
        model, samples, condition, scaler, prediction_curves=100, seed=17,
    )
    second = decode_latent_samples_with_paired_noise(
        model, samples.copy(), condition, scaler, prediction_curves=100, seed=17,
    )
    np.testing.assert_array_equal(first, second)
    for i in range(10):
        np.testing.assert_array_equal(first[i * 10:(i + 1) * 10], second[i * 10:(i + 1) * 10])


def test_10read_and_legacy_100read_protocol_validation_are_separate():
    validate_protocol(_protocol(10))
    validate_protocol(_protocol(100))
    missing_semantics = _protocol(10)
    del missing_semantics["inference"]
    with pytest.raises(ValueError, match="declare latent samples"):
        validate_protocol(missing_semantics)
    invalid = _protocol(10)
    invalid["inference"]["prediction_curves_per_date"] = 10
    with pytest.raises(ValueError, match="100 prediction curves"):
        validate_protocol(invalid)


def test_dry_run_can_select_gain_150_but_never_for_execution():
    tasks = [
        {"instance_id": "date-0", "gain": gain, "reads": 10}
        for gain in (125, 150, 200)
    ]
    args = Namespace(
        dry_run=True, execute=False, stage="vs",
        dry_run_instance_id="date-0", dry_run_gain=150,
    )
    assert _restrict_dry_run_task(tasks, args) == [tasks[1]]
    args.execute = True
    with pytest.raises(RuntimeError, match="dry-runs"):
        _restrict_dry_run_task(tasks, args)


def test_10read_dry_run_and_canary_share_exact_task_selection():
    task = {"stage": "vs", "instance_id": "fixed", "gain": 150,
            "reads": 10, "matrix_sha256": "a" * 64}
    protocol = {"hardware": {"reads_per_instance": 10}}
    dry = Namespace(dry_run=True, execute=False, stage="vs",
                    dry_run_instance_id="fixed", dry_run_gain=150,
                    dry_run_matrix_sha256="a" * 64)
    canary = Namespace(stage="vs", canary=True, confirm_batch=False,
                       confirm_canary=False, canary_instance_id="fixed",
                       canary_gain=150, canary_matrix_sha256="a" * 64,
                       limit=1, execute=False)
    selected_dry = _restrict_dry_run_task([task], dry, protocol)
    selected_canary = _restrict_canary_tasks([task], canary, protocol)
    assert selected_dry == selected_canary == [task]

    for bad in (
        Namespace(**{**vars(canary), "canary_gain": None}),
        Namespace(**{**vars(canary), "canary_matrix_sha256": None}),
        Namespace(**{**vars(canary), "canary_gain": 125}),
        Namespace(**{**vars(canary), "canary_matrix_sha256": "b" * 64}),
    ):
        with pytest.raises(RuntimeError):
            _restrict_canary_tasks([task], bad, protocol)


def test_historical_offline_metric_record_may_omit_read_metadata(tmp_path):
    from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import _recompute_response_metrics
    rng = np.random.default_rng(111)
    h = rng.normal(size=48) * 0.01
    j = np.zeros((48, 48), dtype=np.float64)
    problem = tmp_path / "problem.npz"
    reference = tmp_path / "reference.npz"
    response = tmp_path / "response.npz"
    matrix_path = tmp_path / "matrix.npz"
    samples = rng.choice((-1, 1), size=(100, 48)).astype(np.int8)
    np.savez(problem, h=h, J=j)
    np.savez(reference, samples=samples)
    np.savez(response, samples=np.c_[samples, np.ones(100, dtype=np.int8)])
    matrix = np.zeros((49, 49), dtype=np.int8)
    np.savez(matrix_path, hardware_matrix=matrix)
    import hashlib
    record = {
        "gain": 125, "response_path": str(response),
        "matrix_sha256": __import__(
            "GEFcom2014.models.QBM_VAE.kaiwu_adapter", fromlist=["matrix_sha256"]
        ).matrix_sha256(matrix),
        "hardware_matrix_path": str(matrix_path),
        "source_problem_path": str(problem),
        "source_problem_sha256": hashlib.sha256(problem.read_bytes()).hexdigest(),
        "reference_sample_path": str(reference),
        "reference_sample_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    }
    result = _recompute_response_metrics(record, require_reference_provenance=False)
    assert np.isfinite(result["energy_wasserstein"])


@pytest.mark.parametrize(("gain", "matrix_sha"), [
    (150, "aa815452b06c6a9794a030679f673a3f4a6d9ab6eaf72d892ec38e0a2fe374aa"),
    (200, "6f500cd03ea2c5b755bee5cb948c6546e5f2d44e04585b9cb2b3ee93c07d28a4"),
])
def test_real_entry_selection_matches_dry_run_with_fully_replaced_sdk(
        tmp_path, monkeypatch, capsys, gain, matrix_sha):
    """Run the normal driver path while the only SDK client is a local fake."""
    from pathlib import Path
    from types import SimpleNamespace
    from GEFcom2014.models.QBM_VAE import run_kaiwu_13k_sampling as driver
    from GEFcom2014.models.QBM_VAE import submit_kaiwu_sampling as submitter
    from GEFcom2014.models.QBM_VAE.kaiwu_adapter import matrix_sha256, platform_task_name

    protocol_path = Path("configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json")
    preparation = Path("export/opsd_sa_negative_kaiwu_preparation_10read_v1")
    output_dir, checkpoint_dir = tmp_path / "responses", tmp_path / "checkpoints"
    order_path = preparation / "vs_execution_order.json"
    instance_id = "opsd-wind_opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0_vs_000"
    calls = []

    class FakeClient:
        __module__ = "GEFcom2014.models.QBM_VAE.kaiwu_adapter"

        def __init__(self, **kwargs):
            self._kaiwu = SimpleNamespace(
                __version__="synthetic-sdk",
                common=SimpleNamespace(CheckpointManager=SimpleNamespace(save_dir="")),
            )

        def sample_hardware_matrix(self, matrix, num_reads, task_name=None,
                                   energy_convention=None):
            from GEFcom2014.models.QBM_VAE.kaiwu_adapter import ENERGY_CONVENTION
            calls.append({
                "matrix": np.asarray(matrix).copy(), "reads": num_reads,
                "task_name": task_name,
                "platform_task_name": platform_task_name(task_name),
                "energy_convention": energy_convention,
            })
            return {
                "samples": np.ones((num_reads, 49), dtype=np.int8),
                "metadata": {
                    "backend": "kaiwu_cim_sampling", "task_id": "SYNTHETIC-TASK-ID",
                    "task_name": task_name,
                    "platform_task_name": platform_task_name(task_name),
                },
            }

    monkeypatch.setattr(submitter, "KaiwuClient", FakeClient)
    monkeypatch.setattr(submitter, "kaiwu_sdk_versions", lambda _: {
        "importlib_metadata_version": "synthetic-sdk",
        "kaiwu_module_version": "synthetic-sdk",
        "version_mismatch": False,
    })
    monkeypatch.setattr(driver, "validate_quality_policy", lambda *a, **k: None)
    monkeypatch.setattr(driver, "prediction_quality_mode", lambda _: False)
    environment_get = driver.os.environ.get
    monkeypatch.setattr(
        driver.os.environ, "get",
        lambda key, default=None: "1" if key == "FA_BM_VAE_RUN_REAL_KAIWU"
        else environment_get(key, default),
    )

    def args(**overrides):
        values = dict(
            stage="vs", protocol=protocol_path, preparation=preparation,
            output_dir=output_dir, checkpoint_dir=checkpoint_dir,
            vs_order_manifest=order_path, limit=1, resume=True,
            execute=False, dry_run=True, confirm_real_submission=False,
            confirm_canary=False, confirm_batch=False, canary=False,
            canary_instance_id=None, canary_gain=None,
            canary_matrix_sha256=None, dry_run_instance_id=instance_id,
            dry_run_gain=gain, dry_run_matrix_sha256=matrix_sha,
            project_no=None, offline_simulation=False, offline_freeze=None,
        )
        values.update(overrides)
        return Namespace(**values)

    preview = driver.run(args())
    assert preview == {"stage": "vs", "tasks": 1, "reads": 10,
                       "hardware_claim": False, "physical_platform_used": False,
                       "dry_run": True}
    preview_output = capsys.readouterr().out
    assert "tasks=1 reads=10 dry_run=True" in preview_output
    assert "gain=%d reads=10 matrix_sha256=" % gain + matrix_sha in preview_output
    assert calls == []

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    simulated = driver.run(args(
        dry_run=False, execute=True, confirm_real_submission=True,
        canary=True, confirm_canary=True, canary_instance_id=instance_id,
        canary_gain=gain, canary_matrix_sha256=matrix_sha,
        dry_run_instance_id=None, dry_run_gain=None,
        dry_run_matrix_sha256=None,
    ))
    assert simulated["completed"] == 1
    assert len(calls) == 1
    assert calls[0]["reads"] == 10
    assert calls[0]["energy_convention"]
    assert matrix_sha256(calls[0]["matrix"]) == matrix_sha
    assert calls[0]["platform_task_name"] == platform_task_name(calls[0]["task_name"])
    record = json.loads(next(output_dir.glob("gain_%d/*.json" % gain)).read_text(encoding="utf-8"))
    assert record["instance_id"] == instance_id
    assert record["gain"] == gain
    assert record["requested_reads"] == record["returned_reads"] == 10
    assert record["matrix_sha256"] == matrix_sha
    assert record["task_name"] == calls[0]["task_name"]
    assert record["platform_task_name"] == calls[0]["platform_task_name"]
    assert list(checkpoint_dir.glob("*.intent.json"))
    recovered_intents = driver._load_pending_intents(checkpoint_dir)
    assert calls[0]["task_name"] in recovered_intents

    intents_before = sorted(path.name for path in checkpoint_dir.glob("*.intent.json"))
    for bad_overrides in (
        {"canary_gain": None},
        {"canary_matrix_sha256": None},
        {"canary_gain": 125},
        {"canary_matrix_sha256": "b" * 64},
        {"canary_instance_id": "no-such-instance"},
    ):
        bad_args = args(
            dry_run=False, execute=True, confirm_real_submission=True,
            canary=True, confirm_canary=True, canary_instance_id=instance_id,
            canary_gain=gain, canary_matrix_sha256=matrix_sha,
            dry_run_instance_id=None, dry_run_gain=None,
            dry_run_matrix_sha256=None,
        )
        for key, value in bad_overrides.items():
            setattr(bad_args, key, value)
        with pytest.raises(RuntimeError):
            driver.run(bad_args)
        assert len(calls) == 1
        assert sorted(path.name for path in checkpoint_dir.glob("*.intent.json")) == intents_before

    # A malformed frozen task count must fail before an intent or SDK call.
    original_planned_tasks = driver.planned_tasks

    def wrong_read_count(*call_args, **call_kwargs):
        tasks = original_planned_tasks(*call_args, **call_kwargs)
        for task in tasks:
            if task.get("instance_id") == instance_id:
                task["reads"] = 11
        return tasks

    monkeypatch.setattr(driver, "planned_tasks", wrong_read_count)
    bad_reads = args(
        dry_run=False, execute=True, confirm_real_submission=True,
        canary=True, confirm_canary=True, canary_instance_id=instance_id,
        canary_gain=gain, canary_matrix_sha256=matrix_sha,
        dry_run_instance_id=None, dry_run_gain=None,
        dry_run_matrix_sha256=None,
    )
    with pytest.raises(RuntimeError):
        driver.run(bad_reads)
    assert len(calls) == 1
    assert sorted(path.name for path in checkpoint_dir.glob("*.intent.json")) == intents_before

