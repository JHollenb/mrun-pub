"""Deterministic ids — same config => same id (ported from discovery lab/runner.py).

Used as ``client_run_id`` so a re-submit of an identical run can be found/attached, and
as the estimates-table key so the second run of any config gets a calibrated reservation.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def stable_hash(payload: dict[str, Any], *, n: int = 16) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


def config_run_id(kind: str, config: dict[str, Any], cmd: list[str] | None = None) -> str:
    return stable_hash({"kind": kind, "config": config, "cmd": cmd or []})


def new_job_id() -> str:
    return "job-" + uuid.uuid4().hex[:12]
