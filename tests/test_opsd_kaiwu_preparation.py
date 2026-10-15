import json

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE.kaiwu_13k_protocol import (
    audit_budget,
    load_protocol,
    validate_test_manifest,
)
from GEFcom2014.models.QBM_VAE.opsd_kaiwu_preparation import (
    DEFAULT_PROTOCOL,
    historical_checkpoint_selection_evidence,
    select_vs_rows,
)


def _test_item(index):
    return {
        "id": "opsd_test_%03d" % index,
        "split": "TEST",
        "selection": "all",
        "selection_seed": 2026,
        "hardware_gain": None,
        "hardware_n_bits": None,
        "date": "2019-01-%02d" % (index + 1),
    }


def _vs_item(gain, index):
    return {
        "id": "opsd_vs_%03d" % index,
        "split": "VS",
        "selection": "stratified",
        "selection_seed": 2026,
        "hardware_gain": gain,
        "logical_n_bits": 48,
        "hardware_n_bits": 49,
    }


def test_opsd_protocol_budget_is_10x3_plus_90(tmp_path):
    protocol = load_protocol("configs/paper/fa_bm_vae_opsd_kaiwu_protocol.json")
    vs_paths = []
    for gain in protocol["hardware"]["candidate_gains"]:
        path = tmp_path / ("vs_%d.json" % gain)
        path.write_text(json.dumps([_vs_item(gain, index) for index in range(10)]), encoding="utf-8")
        vs_paths.append(path)
    test_path = tmp_path / "test.json"
    test_path.write_text(json.dumps([_test_item(index) for index in range(90)]), encoding="utf-8")
    budget = audit_budget(vs_paths, test_path, protocol)
    assert budget["vs_tasks"] == 30
    assert budget["test_tasks"] == 90
    assert budget["planned_samples"] == 12000


def test_opsd_protocol_rejects_wrong_test_count(tmp_path):
    protocol = load_protocol("configs/paper/fa_bm_vae_opsd_kaiwu_protocol.json")
    path = tmp_path / "test.json"
    path.write_text(json.dumps([_test_item(index) for index in range(89)]), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol size"):
        validate_test_manifest(path, protocol)


def test_sa_negative_protocol_keeps_the_fixed_12k_budget_and_checkpoint():
    protocol = load_protocol("configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json")
    assert protocol["model"]["model_name"] == "opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0"
    assert protocol["model"]["negative_phase_backend"].startswith("sa")
    assert protocol["budget"]["planned_samples"] == 12000


def test_opsd_preparation_default_protocol_is_the_sa_negative_case_study():
    assert DEFAULT_PROTOCOL.name == "fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json"


def test_legacy_checkpoint_audit_does_not_infer_test_non_use():
    evidence = historical_checkpoint_selection_evidence()
    assert evidence["test_used"] == "NOT_RECORDED"
    assert evidence["best_epoch"] == "NOT_RECORDED"


def test_vs_selection_is_context_only_and_reproducible():
    context = np.arange(90 * 33, dtype=np.float32).reshape(90, 33)
    first = select_vs_rows(context, 2026, 10)
    second = select_vs_rows(context, 2026, 10)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 10
