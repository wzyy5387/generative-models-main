import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE import run_kaiwu_13k_sampling as driver
from GEFcom2014.models.QBM_VAE.derive_historical_kaiwu_prefix import (
    _verify_hash,
    assert_matching_task_identity,
    validate_bundle,
    verify_prefix_arrays,
)
from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import sha256_file


def test_prefix_arrays_require_original_order_and_same_logical_state():
    parent = np.ones((100, 49), dtype=np.int8)
    parent[5:10, :48] = -1
    parent[5:10, 48] = -1
    logical, unique = verify_prefix_arrays(parent, parent[:10], np.arange(10))
    assert logical.shape == (100, 48)
    assert unique == 1
    with pytest.raises(ValueError, match="row order"):
        verify_prefix_arrays(parent, parent[1:11], np.arange(1, 11))
    changed = parent[:10].copy()
    changed[[0, 5]] = changed[[5, 0]]
    with pytest.raises(ValueError, match="exact original-order"):
        verify_prefix_arrays(parent, changed, np.arange(10))
    mixed = parent.copy()
    mixed[99, 0] = -1
    with pytest.raises(ValueError, match="reuse criterion"):
        verify_prefix_arrays(mixed, mixed[:10], np.arange(10))


@pytest.mark.parametrize("artifact_name", ["response", "receipt", "matrix"])
def test_changed_source_response_receipt_or_matrix_hash_is_rejected(tmp_path, artifact_name):
    path = tmp_path / (artifact_name + ".bin")
    path.write_bytes(b"source bytes")
    expected = sha256_file(path)
    assert _verify_hash(path, expected, artifact_name).is_file()
    path.write_bytes(b"tampered bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_hash(path, expected, artifact_name)


def test_matrix_or_energy_identity_mismatch_is_rejected():
    source = {"instance_id": "i", "date": "2019-01-06", "gain": 125,
              "matrix_sha256": "a" * 64,
              "hardware_energy_convention": "kaiwu_sdk_hamiltonian_matches_fa_bm_vae_energy_v1",
              "source_problem_sha256": "b" * 64,
              "hardware_matrix_file_sha256": "c" * 64}
    assert_matching_task_identity(source, dict(source))
    for field in ("matrix_sha256", "hardware_energy_convention"):
        changed = dict(source)
        changed[field] = "mismatch"
        with pytest.raises(ValueError, match=field):
            assert_matching_task_identity(source, changed)


def test_historical_prefix_blocks_execution_before_submit_or_intent(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    bundle = repo / "export/opsd_sa_negative_kaiwu_historical_prefix_v3"
    validate_bundle(bundle)
    copied_bundle = tmp_path / "amendment"
    shutil.copytree(bundle, copied_bundle)

    calls = []
    monkeypatch.setattr(driver, "submit_one", lambda *a, **k: calls.append((a, k)))
    output_dir = repo / "export/opsd_sa_negative_kaiwu_preparation_10read_v1/real_responses"
    checkpoint_dir = repo / "export/opsd_sa_negative_kaiwu_preparation_10read_v1/checkpoints"
    intent_before = {p.name: sha256_file(p) for p in checkpoint_dir.glob("*.intent.json")}
    args = SimpleNamespace(
        stage="vs",
        protocol=repo / "configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json",
        preparation=repo / "export/opsd_sa_negative_kaiwu_preparation_10read_v1",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        execute=True,
        dry_run=False,
        limit=None,
        resume=True,
        vs_order_manifest=repo / "export/opsd_sa_negative_kaiwu_preparation_10read_v1/vs_execution_order.json",
        historical_prefix_amendment=copied_bundle,
    )
    with pytest.raises(RuntimeError, match="no intent or SDK submission was made"):
        driver.run(args)
    intent_after = {p.name: sha256_file(p) for p in checkpoint_dir.glob("*.intent.json")}
    assert intent_after == intent_before
    assert calls == []

    # A changed receipt in the temporary copy is rejected before the same guard.
    manifest = json.loads((copied_bundle / "historical_prefix_amendment.json").read_text(encoding="utf-8"))
    receipt = Path(manifest["source_physical_task"]["receipt_path"])
    receipt_copy = tmp_path / "receipt.json"
    shutil.copy2(receipt, receipt_copy)
    manifest["source_physical_task"]["receipt_path"] = str(receipt_copy)
    (copied_bundle / "historical_prefix_amendment.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        driver._guard_historical_prefix_resubmission(
            [{"instance_id": manifest["new_panel_entry"]["instance_id"],
              "gain": 125, "matrix_sha256": manifest["new_panel_entry"]["target_matrix_content_sha256"]}],
            copied_bundle)
    receipt_copy.write_bytes(b"altered receipt")
    lock = copied_bundle / "historical_prefix_amendment.sha256"
    lock.write_text(sha256_file(copied_bundle / "historical_prefix_amendment.json") +
                    "  historical_prefix_amendment.json\n", encoding="ascii")
    with pytest.raises(ValueError, match="Historical platform receipt hash mismatch"):
        driver._guard_historical_prefix_resubmission(
            [{"instance_id": manifest["new_panel_entry"]["instance_id"],
              "gain": 125, "matrix_sha256": manifest["new_panel_entry"]["target_matrix_content_sha256"]}],
            copied_bundle)
    assert calls == []
