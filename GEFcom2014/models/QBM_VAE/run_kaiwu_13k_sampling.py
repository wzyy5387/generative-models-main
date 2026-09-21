# -*- coding: utf-8 -*-
"""Resumable, manually-gated VS/TEST Kaiwu batch driver.

The default is an offline dry-run.  This module never creates physical-platform
claims for plans, mocks, or local SA responses.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .kaiwu_13k_protocol import load_protocol
from .submit_kaiwu_sampling import submit_one


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREPARATION = ROOT_DIR / "export" / "kaiwu_13k_preparation"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vs_tasks(preparation):
    tasks = []
    for gain in (125, 150, 200):
        manifest_path = preparation / ("submission_gain_%d" % gain) / "package" / "submission_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Missing frozen VS submission manifest: %s" % manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("instances", []):
            matrix = manifest_path.parent / item["hardware_matrix"]
            tasks.append({"gain": gain, "manifest": manifest_path, "instance_id": item["instance_id"],
                          "matrix": matrix, "matrix_sha256": item["hardware_matrix_sha256"],
                          "reads": int(payload["requested_reads_per_instance"])})
    if len(tasks) != 30:
        raise ValueError("Frozen VS plan must contain exactly 30 tasks")
    return tasks


def _load_completed(output_dir):
    completed = set()
    for path in Path(output_dir).glob("**/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("completed") is True and data.get("matrix_sha256"):
            completed.add((str(data.get("instance_id")), str(data.get("matrix_sha256"))))
    return completed


def planned_tasks(stage, preparation, output_dir):
    if stage == "vs":
        return _vs_tasks(preparation)
    freeze = preparation / "real_gain_selection.json"
    if not freeze.is_file():
        raise RuntimeError("TEST is locked until real_gain_selection.json is frozen from verified VS responses")
    raise RuntimeError("TEST matrix manifest is intentionally not generated before VS gain freeze")


def run(args):
    protocol = load_protocol(args.protocol)
    preparation = Path(args.preparation).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    tasks = planned_tasks(args.stage, preparation, output_dir)
    limit = len(tasks) if args.limit is None else max(0, min(int(args.limit), len(tasks)))
    tasks = tasks[:limit]
    total_reads = sum(item["reads"] for item in tasks)
    max_tasks, max_reads = ((30, 3000) if args.stage == "vs" else (100, 10000))
    if len(tasks) > max_tasks or total_reads > max_reads:
        raise ValueError("Requested batch exceeds frozen %s budget" % args.stage.upper())
    print("stage=%s tasks=%d reads=%d dry_run=%s" % (args.stage, len(tasks), total_reads, not args.execute))
    for item in tasks:
        print("%s gain=%s reads=%d matrix_sha256=%s" %
              (item["instance_id"], item["gain"], item["reads"], item["matrix_sha256"]))
    if not args.execute:
        return {"stage": args.stage, "tasks": len(tasks), "reads": total_reads, "hardware_claim": False,
                "physical_platform_used": False, "dry_run": True}
    if not args.confirm_real_submission or os.environ.get("FA_BM_VAE_RUN_REAL_KAIWU") != "1":
        raise RuntimeError("Real batch requires --confirm-real-submission and FA_BM_VAE_RUN_REAL_KAIWU=1")
    project_no = args.project_no or os.environ.get("KAIWU_PROJECT_NO")
    if not project_no:
        raise RuntimeError("KAIWU_PROJECT_NO/project number is required")
    completed = _load_completed(output_dir) if args.resume else set()
    audit_path = output_dir / (args.stage + "_batch_audit.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in tasks:
        key = (item["instance_id"], item["matrix_sha256"])
        if key in completed:
            print("skip verified response %s" % item["instance_id"])
            continue
        response_dir = output_dir / ("gain_%d" % item["gain"])
        result = submit_one(item["matrix"], item["instance_id"], response_dir, item["reads"],
                            project_no=project_no, checkpoint_dir=checkpoint_dir)
        record = result[2]
        record["completed"] = True
        records.append(record)
        audit_path.write_text(json.dumps({"stage": args.stage, "records": records,
                                          "hardware_claim": all(r.get("hardware_claim") for r in records),
                                          "physical_platform_used": all(r.get("physical_platform_used") for r in records)}, indent=2),
                              encoding="utf-8")
    return {"stage": args.stage, "completed": len(records), "hardware_claim": True,
            "physical_platform_used": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["vs", "test"], required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT_DIR / "configs" / "paper" / "fa_bm_vae_kaiwu_13k_protocol.json")
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARATION / "real_responses")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_PREPARATION / "checkpoints")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Default; print plan without SDK calls")
    parser.add_argument("--execute", action="store_true", help="Still requires all real-submission gates")
    parser.add_argument("--confirm-real-submission", action="store_true")
    parser.add_argument("--project-no", default=None)
    args = parser.parse_args()
    if args.dry_run and args.execute:
        raise SystemExit("Choose only one of --dry-run and --execute")
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
