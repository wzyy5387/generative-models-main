"""Offline checks for formal task logging and safe resume; no model training."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from GEFcom2014.models.QBM_VAE import run_neso_wind_jobs as runner


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.npz"
    np.savez(bundle, y_ls=np.zeros((1, 24)), y_vs=np.zeros((1, 24)), y_test=np.zeros((1, 24)))
    script = tmp_path / "fake_task.py"
    script.write_text(
        "import json, pathlib, pickle, sys\n"
        "import numpy as np\n"
        "out, bundle, seed, counter, fail = sys.argv[1:]\n"
        "out, counter, fail = pathlib.Path(out), pathlib.Path(counter), pathlib.Path(fail)\n"
        "counter.write_text(counter.read_text() + seed + '\\n' if counter.exists() else seed + '\\n')\n"
        "print('synthetic seed ' + seed, flush=True)\n"
        "if fail.exists() and seed == '1': sys.exit(9)\n"
        "out.mkdir(parents=True)\n"
        "(out / 'task.json').write_text(json.dumps(dict(model='residual-bootstrap', seed=int(seed), "
        "generation_seed=int(seed), dataset_bundle=bundle, scenario_splits=['VS', 'TEST'], "
        "n_scenarios=100)))\n"
        "for split in ('VS', 'TEST'):\n"
        " with (out / ('scenarios_100_' + split + '.pickle')).open('wb') as f:\n"
        "  pickle.dump(np.zeros((24, 100)), f)\n",
        encoding="utf-8",
    )
    root, counter, fail = tmp_path / "output", tmp_path / "counter.txt", tmp_path / "fail.flag"

    def fake_jobs(mode, run_label, output_root, dataset_bundle, models, config_dir):
        assert mode == "formal"
        return [dict(model="residual_bootstrap", training_seed=seed, generation_seed=seed,
                     graph_seed=None, scenario_splits=["VS", "TEST"], n_scenarios=100,
                     output_dir=str(Path(output_root) / run_label / "residual_bootstrap" / ("seed_%d" % seed)),
                     command=[sys.executable, str(script),
                              str(Path(output_root) / run_label / "residual_bootstrap" / ("seed_%d" % seed)),
                              str(dataset_bundle), str(seed), str(counter), str(fail)])
                for seed in (0, 1)]

    monkeypatch.setattr(runner, "jobs", fake_jobs)
    return root, bundle, counter, fail


def test_completed_outputs_are_verified_and_not_rerun(synthetic_run):
    root, bundle, counter, _ = synthetic_run
    result = runner.run("formal", output_root=root, bundle=bundle)
    assert result["status"] == "COMPLETED"
    assert counter.read_text().splitlines() == ["0", "1"]
    assert all(row["artifact_hashes"] and row["log_sha256"] for row in result["results"])
    assert "synthetic seed 0" in (root / "formal/run_logs/01_residual_bootstrap_seed_0.log").read_text()
    assert runner.run("formal", output_root=root, bundle=bundle, resume=True)["status"] == "COMPLETED"
    assert counter.read_text().splitlines() == ["0", "1"]
    scenario = root / "formal/residual_bootstrap/seed_0/scenarios_100_TEST.pickle"
    scenario.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed|Invalid"):
        runner.run("formal", output_root=root, bundle=bundle, resume=True)
    assert counter.read_text().splitlines() == ["0", "1"]


def test_resume_only_skips_a_verified_completed_prefix(synthetic_run):
    root, bundle, counter, _ = synthetic_run
    runner.run("formal", output_root=root, bundle=bundle)
    run_root = root / "formal"
    history = json.loads((run_root / "job_results.json").read_text())
    (run_root / "job_results.json").write_text(json.dumps(history[:1]))
    manifest = json.loads((run_root / "task_manifest.json").read_text())
    manifest["status"] = "RUNNING"
    (run_root / "task_manifest.json").write_text(json.dumps(manifest))
    shutil.rmtree(run_root / "residual_bootstrap/seed_1")
    (run_root / "run_logs/02_residual_bootstrap_seed_1.log").unlink()
    assert runner.run("formal", output_root=root, bundle=bundle, resume=True)["status"] == "COMPLETED"
    assert counter.read_text().splitlines() == ["0", "1", "1"]


def test_failed_task_and_identity_change_block_resume(synthetic_run):
    root, bundle, counter, fail = synthetic_run
    fail.write_text("fail")
    result = runner.run("formal", output_root=root, bundle=bundle)
    assert result["status"] == "INCOMPLETE"
    assert [row["status"] for row in result["results"]] == ["COMPLETED", "FAILED"]
    with pytest.raises(ValueError, match="refusing automatic resubmission"):
        runner.run("formal", output_root=root, bundle=bundle, resume=True)
    assert counter.read_text().splitlines() == ["0", "1"]
    bundle.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen run identity differs"):
        runner.run("formal", output_root=root, bundle=bundle, resume=True)
    assert counter.read_text().splitlines() == ["0", "1"]
