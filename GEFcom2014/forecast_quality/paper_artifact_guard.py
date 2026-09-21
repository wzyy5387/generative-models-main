# -*- coding: utf-8 -*-
"""Fail-closed guards for artifacts that may enter paper summaries."""

import json
from pathlib import Path


def parse_bool(value, field="boolean"):
    """Parse JSON booleans and CSV boolean strings without truthiness traps."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError("Invalid %s value: %r" % (field, value))


def assert_paper_eligible(metadata, source="artifact"):
    metadata = metadata or {}
    if parse_bool(metadata.get("smoke_only"), "smoke_only") is True or parse_bool(
        metadata.get("synthetic_data"), "synthetic_data"
    ) is True:
        raise ValueError("Paper aggregation refused synthetic/smoke artifact: %s" % source)
    eligible = parse_bool(metadata.get("eligible_for_paper"), "eligible_for_paper")
    if eligible is not True:
        raise ValueError("Paper aggregation requires eligible_for_paper=true: %s" % source)
    role = metadata.get("artifact_role")
    if role in {"acceptance", "pilot", "smoke", "formal_task"}:
        raise ValueError("Paper aggregation refused non-complete artifact role %s: %s" % (role, source))
    return metadata


def read_metadata(path):
    path = Path(path)
    if path.suffix.lower() != ".json":
        return {}
    return assert_paper_eligible(json.loads(path.read_text(encoding="utf-8")), str(path))
