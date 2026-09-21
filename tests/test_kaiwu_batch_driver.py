import subprocess
import sys


def test_kaiwu_batch_default_dry_run_lists_frozen_budget():
    command = [sys.executable, "-m", "GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling",
               "--stage", "vs", "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "tasks=30 reads=3000 dry_run=True" in result.stdout
    assert "matrix_sha256=" in result.stdout
