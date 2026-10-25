"""Freeze a date-interleaved VS order beside (not inside) the v8 package."""

import argparse
import json
from pathlib import Path

from .kaiwu_13k_protocol import load_protocol
from .run_kaiwu_13k_sampling import (
    _preparation_identity,
    _validate_preparation_identity,
    _vs_tasks,
    build_vs_execution_order,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    protocol["_path"] = str(args.protocol.resolve())
    _validate_preparation_identity(args.preparation, protocol)
    identity = _preparation_identity(args.preparation, protocol)
    tasks = _vs_tasks(args.preparation, protocol)
    payload = build_vs_execution_order(
        tasks, identity["protocol_sha256"], identity["preparation_manifest_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"path": str(args.output.resolve()), "tasks": len(payload["entries"]),
                      "protocol_sha256": payload["protocol_sha256"]}))


if __name__ == "__main__":
    main()
