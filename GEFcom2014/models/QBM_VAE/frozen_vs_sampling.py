"""Generate VS scenarios from an already-fitted FA-BM-VAE pickle only."""

import argparse
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.models import scale_data_multi
from GEFcom2014.models.QBM_VAE import build_qbm_vae_scenarios, build_sampler
from GEFcom2014.utils import dump_file


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ls_vs_bundle(path):
    with np.load(path, allow_pickle=False) as bundle:
        keys = ("x_ls", "y_ls", "x_vs", "y_vs")
        arrays = {key: np.asarray(bundle[key], dtype=np.float32) for key in keys}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("LS/VS bundle contains non-finite values")
    return arrays


def generate_frozen_vs(model_path, dataset_bundle, output_dir, origin_id, seed,
                       n_scenarios=100, device="cuda", source_commit=None):
    model_path = Path(model_path)
    dataset_bundle = Path(dataset_bundle)
    output_dir = Path(output_dir)
    model_config_path = model_path.with_suffix(".json")
    config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if not config.get("decoder_anchor"):
        raise ValueError("Frozen VS sampling requires a forecast-anchored FA-BM-VAE")
    if config.get("sampler") != "sa":
        raise ValueError("Frozen VS sampling requires sampler=sa")

    arrays = _load_ls_vs_bundle(dataset_bundle)
    x_ls, y_ls = arrays["x_ls"], arrays["y_ls"]
    x_vs, y_vs = arrays["x_vs"], arrays["y_vs"]
    if len(y_vs) != 90 or y_vs.shape[1] != 24:
        raise ValueError("Frozen VS sampling requires a 90-day, 24-hour VS split")
    x_ls_scaled, y_ls_scaled, x_vs_scaled, _, _, _, y_scaler = scale_data_multi(
        x_LS=x_ls, y_LS=y_ls, x_VS=x_vs, y_VS=y_vs,
        x_TEST=None, y_TEST=None,
    )
    target_size = y_ls_scaled.shape[1]
    if x_ls.shape[1] < target_size:
        raise ValueError("Frozen FA-BM-VAE context has fewer forecast features than targets")
    x_vs_scaled = np.concatenate((
        x_vs_scaled,
        y_scaler.transform(x_vs[:, :target_size]),
    ), axis=1)

    torch_device = torch.device("cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    model.device = torch_device
    model.to(torch_device)
    model.eval()
    torch.manual_seed(int(seed))
    sampler = build_sampler("sa", sweeps=200, beta_start=0.1, beta_end=2.0, seed=int(seed))
    scenarios = build_qbm_vae_scenarios(
        n_s=int(n_scenarios), x=x_vs_scaled, y_scaler=y_scaler, model=model,
        sampler=sampler, max=1, gpu=torch_device.type == "cuda", tag="opsd-wind",
        beta=float(config.get("beta", 1.0)), gibbs_steps=int(config.get("negative_steps", 5)),
    )
    scenarios = np.asarray(scenarios, dtype=np.float64)
    expected_shape = (90 * 24, int(n_scenarios))
    if scenarios.shape != expected_shape or not np.isfinite(scenarios).all():
        raise ValueError("Unexpected frozen VS scenario shape or non-finite value")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "scenarios_opsd-wind_QBMVAE_2_anchor_sa_%d_%d_VS" % (int(seed), int(n_scenarios))
    scenario_path = output_dir / (stem + ".pickle")
    dump_file(str(output_dir) + "/", stem, scenarios)
    metadata = {
        "artifact_role": "frozen_vs_scenarios",
        "origin_id": origin_id,
        "seed": int(seed),
        "model_path": str(model_path.resolve()),
        "model_sha256": file_sha256(model_path),
        "dataset_bundle": str(dataset_bundle.resolve()),
        "dataset_bundle_sha256": file_sha256(dataset_bundle),
        "sampler": "sa",
        "n_scenarios": int(n_scenarios),
        "scenario_file": str(scenario_path.resolve()),
        "scenario_sha256": file_sha256(scenario_path),
        "shape": list(scenarios.shape),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "test_data_accessed": False,
        "training_invoked": False,
        "model_parameters_updated": False,
    }
    (output_dir / (stem + ".json")).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_path", type=Path, required=True)
    parser.add_argument("--dataset-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--origin", dest="origin_id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(generate_frozen_vs(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
