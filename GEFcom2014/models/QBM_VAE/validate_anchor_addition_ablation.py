# -*- coding: utf-8 -*-
"""Verify the strict no-anchor-addition initialization before training."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models.QBM_VAE.qbm_vae import load_frozen_graph_mask
from GEFcom2014.models.QBM_VAE.utils_qbm_vae import ConditionalQBMVAE


ROOT_DIR = Path(__file__).resolve().parents[3]
FULL_CONFIG = ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_wind_sa_negative_v1.json"
ABLATION_CONFIG = ROOT_DIR / "configs" / "paper" / "fa_bm_vae_opsd_wind_sa_negative_no_anchor_addition_zero_mean_init_v2.json"
DEFAULT_OUTPUT = ROOT_DIR / "export" / "opsd_sa_negative_matched_v1" / "no_anchor_addition_zero_mean_init" / "initialization_audit.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state():
    try:
        status = subprocess.run(["git", "status", "--short"], cwd=ROOT_DIR,
                                capture_output=True, text=True, check=True).stdout.strip()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR,
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "NOT_RECORDED", "dirty": "NOT_RECORDED", "status": "NOT_RECORDED"}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_RECORDED"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _model(config, graph_mask, cond_in, in_size, decoder_anchor):
    return ConditionalQBMVAE(
        latent_s=int(config["latent_s"]), cond_in=cond_in, in_size=in_size,
        enc_w=int(config["enc_w"]), enc_l=int(config["enc_l"]),
        dec_w=int(config["dec_w"]), dec_l=int(config["dec_l"]),
        graph_mask=graph_mask, gpu=False,
        decoder_covariance=config["decoder_covariance"],
        decoder_anchor=decoder_anchor, decoder_mean_zero_init=True,
    )


def verify(full_config=FULL_CONFIG, ablation_config=ABLATION_CONFIG, seeds=(0, 1, 2)):
    full, ablation = _load(full_config), _load(ablation_config)
    if not full.get("forecast_anchor") or not ablation.get("context_anchor_conditioning_only"):
        raise ValueError("Configurations do not define the requested anchor-addition contrast")
    if ablation.get("decoder_mean_zero_init") is not True:
        raise ValueError("Strict no-anchor-addition ablation requires decoder_mean_zero_init=true")
    if full["graph_mask_model"] != ablation["graph_mask_model"]:
        raise ValueError("Full and ablation configurations must freeze the same graph mask")
    bundle = ROOT_DIR / full["dataset_bundle"]
    arrays, _, _ = load_daily_bundle(bundle)
    in_size = int(arrays["y_ls"].shape[1])
    cond_in = int(arrays["x_ls"].shape[1] + in_size)
    graph_path = ROOT_DIR / full["graph_mask_model"]
    graph_mask = load_frozen_graph_mask(graph_path, int(full["latent_s"]))
    graph_sha256 = hashlib.sha256(np.asarray(graph_mask, dtype="<f4").tobytes()).hexdigest()

    records = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        anchored = _model(full, graph_mask, cond_in, in_size, decoder_anchor=True)
        torch.manual_seed(int(seed))
        ablated = _model(ablation, graph_mask, cond_in, in_size, decoder_anchor=False)
        state_equal = all(torch.equal(value, ablated.state_dict()[key])
                          for key, value in anchored.state_dict().items())
        if not state_equal:
            raise AssertionError("Initial parameter or graph-mask mismatch for seed %s" % seed)
        generator = torch.Generator().manual_seed(1000 + int(seed))
        latent = torch.randn(3, int(full["latent_s"]), generator=generator)
        condition = torch.randn(3, cond_in, generator=generator)
        anchored_mean, anchored_scale = anchored.decode_distribution(latent, condition)
        ablated_mean, ablated_scale = ablated.decode_distribution(latent, condition)
        mean_difference = anchored_mean - ablated_mean
        if not torch.equal(mean_difference, condition[:, -in_size:]):
            raise AssertionError("Decoder means differ by more than anchor addition for seed %s" % seed)
        if not torch.equal(anchored_scale, ablated_scale):
            raise AssertionError("Decoder scales differ at initialization for seed %s" % seed)
        records.append({
            "seed": int(seed), "initial_state_equal": True,
            "graph_mask_sha256": graph_sha256,
            "graph_edges": int(np.count_nonzero(np.triu(graph_mask, k=1))),
            "mean_difference_equals_anchor": True,
            "scale_output_equal": True,
            "decoder_mean_zero_init": True,
        })
    return {
        "artifact_role": "strict_no_anchor_addition_initialization_audit",
        "eligible_for_paper": False,
        "full_config": str(Path(full_config).resolve()),
        "full_config_sha256": sha256_file(full_config),
        "ablation_config": str(Path(ablation_config).resolve()),
        "ablation_config_sha256": sha256_file(ablation_config),
        "frozen_graph_checkpoint": str(graph_path.resolve()),
        "frozen_graph_checkpoint_sha256": sha256_file(graph_path),
        "dataset": str(bundle.resolve()), "dataset_sha256": sha256_file(bundle),
        "code_sha256": {
            "qbm_vae": sha256_file(ROOT_DIR / "GEFcom2014/models/QBM_VAE/qbm_vae.py"),
            "utils_qbm_vae": sha256_file(ROOT_DIR / "GEFcom2014/models/QBM_VAE/utils_qbm_vae.py"),
            "validator": sha256_file(__file__),
        },
        "git": _git_state(),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "pandas": _version("pandas"), "scikit_learn": _version("scikit-learn"),
        },
        "commands": [
            "%s -u -m GEFcom2014.models.QBM_VAE.qbm_vae --config %s --seed %d --output-dir export/opsd_sa_negative_matched_v1/no_anchor_addition_zero_mean_init"
            % (sys.executable, Path(ablation_config).relative_to(ROOT_DIR), seed)
            for seed in seeds
        ],
        "comparison": "same seed, same parameters and graph mask; decoder_anchor is the only behavioral difference",
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-config", type=Path, default=FULL_CONFIG)
    parser.add_argument("--ablation-config", type=Path, default=ABLATION_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    args = parser.parse_args()
    payload = verify(args.full_config, args.ablation_config, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
