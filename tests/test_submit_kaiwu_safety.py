import sys

import pytest

from GEFcom2014.models.QBM_VAE import submit_kaiwu_sampling


def test_real_cli_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_kaiwu_sampling",
            "--matrix-file", "matrix.npz",
            "--instance-id", "instance",
            "--output-dir", "out",
            "--checkpoint-dir", "checkpoints",
        ],
    )
    with pytest.raises(RuntimeError, match="disabled by default"):
        submit_kaiwu_sampling.main()
