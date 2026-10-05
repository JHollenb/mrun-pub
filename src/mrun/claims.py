"""Typed claim-row export — mrun's side of the shared claim graph.

Every promotion or refusal decision that mrun makes (workplan gates, CUDA Graph
campaign selection) is exported as one content-addressed claim row. manalysis
owns the full claim store and its supersession semantics; mrun only emits rows
so no decision is trapped inside a one-off result JSON.

Export must never break the primary work: any failure degrades to a stderr
warning and an ``{"error": ...}`` return (the tapes-store doctrine).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import compact_json, stable_json

CLAIM_ROW_SCHEMA = "mrun-claim-row-v1"

CLAIM_VERDICTS = ("promotable", "blocked", "promoted", "refused")


def _sha256_hex(value: object, field_name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class ClaimRow:
    claim_type: str
    verdict: str
    scope: dict[str, Any] = field(default_factory=dict)
    custody: dict[str, str] = field(default_factory=dict)
    source_job_ids: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    detail: str = ""
    created_ts: float = 0.0
    schema_version: str = CLAIM_ROW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CLAIM_ROW_SCHEMA:
            raise ValueError(f"unsupported claim row schema: {self.schema_version}")
        if not self.claim_type:
            raise ValueError("claim_type must be non-empty")
        if self.verdict not in CLAIM_VERDICTS:
            raise ValueError(f"unknown claim verdict: {self.verdict!r}")
        custody = {
            str(k): _sha256_hex(v, f"custody[{k}]") for k, v in self.custody.items()
        }
        object.__setattr__(self, "custody", custody)
        object.__setattr__(
            self, "source_job_ids", tuple(str(j) for j in self.source_job_ids)
        )
        object.__setattr__(self, "supersedes", tuple(str(s) for s in self.supersedes))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim_type": self.claim_type,
            "verdict": self.verdict,
            "scope": self.scope,
            "custody": self.custody,
            "source_job_ids": list(self.source_job_ids),
            "supersedes": list(self.supersedes),
            "detail": self.detail,
        }

    @property
    def claim_id(self) -> str:
        return hashlib.sha256(compact_json(self._payload()).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["claim_id"] = self.claim_id
        payload["created_ts"] = self.created_ts
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClaimRow:
        row = cls(
            claim_type=payload["claim_type"],
            verdict=payload["verdict"],
            scope=dict(payload.get("scope") or {}),
            custody=dict(payload.get("custody") or {}),
            source_job_ids=tuple(payload.get("source_job_ids") or ()),
            supersedes=tuple(payload.get("supersedes") or ()),
            detail=str(payload.get("detail", "")),
            created_ts=float(payload.get("created_ts") or 0.0),
            schema_version=payload.get("schema_version", CLAIM_ROW_SCHEMA),
        )
        claimed = payload.get("claim_id")
        if claimed is not None and claimed != row.claim_id:
            raise ValueError("claim row id mismatch: content drifted")
        return row


def claims_dir() -> Path:
    env = os.environ.get("MRUN_CLAIMS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".local" / "state" / "mrun" / "claims"


def _current_job_ids() -> tuple[str, ...]:
    job_id = os.environ.get("MRUN_JOB_ID")
    return (job_id,) if job_id else ()


def export_claim(row: ClaimRow) -> dict[str, Any]:
    """Idempotently write ``rows/{claim_id}.json`` and append one log line.

    Never raises: claim export is an observer of a decision that has already
    been recorded in the primary payload."""

    try:
        directory = claims_dir()
        rows = directory / "rows"
        rows.mkdir(parents=True, exist_ok=True)
        target = rows / f"{row.claim_id}.json"
        created = False
        if not target.exists():
            # Publish through an atomic hard-link.  Unlike check-then-replace,
            # this makes the first concurrent exporter the sole owner of the
            # row and avoids duplicate log entries for the same claim.
            tmp: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=rows,
                    prefix=f".{row.claim_id}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    tmp = Path(handle.name)
                    handle.write(stable_json(row.as_dict()) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(tmp, target)
                except FileExistsError:
                    pass
                else:
                    created = True
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
        if created:
            line = compact_json(
                {
                    "claim_id": row.claim_id,
                    "ts": time.time(),
                    "claim_type": row.claim_type,
                    "verdict": row.verdict,
                }
            )
            with open(directory / "log.jsonl", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return {"claim_id": row.claim_id, "path": str(target), "created": created}
    except Exception as exc:  # noqa: BLE001 — never break the primary work
        print(f"mrun: claim export skipped ({exc})", file=sys.stderr)
        return {"error": str(exc)}




# -- builders ----------------------------------------------------------------


def claim_from_promotion_report(
    report: Any,
    bundle_fingerprint: str,
    *,
    scope: dict[str, Any] | None = None,
    artifact_sha256: str | None = None,
) -> ClaimRow:
    custody = {"bundle_fingerprint": bundle_fingerprint}
    if artifact_sha256:
        custody["artifact_sha256"] = artifact_sha256
    return ClaimRow(
        claim_type="workplan-promotion",
        verdict="promotable" if report.promotable else "blocked",
        scope={**(scope or {}), "target": report.target},
        custody=custody,
        source_job_ids=_current_job_ids(),
        detail="; ".join(report.blockers)
        if report.blockers
        else ", ".join(report.satisfied_requirements),
        created_ts=time.time(),
    )


def claim_from_cuda_graph_decision(
    decision: Any, *, scope: dict[str, Any] | None = None
) -> ClaimRow:
    custody: dict[str, str] = {}
    if decision.promotion is not None:
        custody["promotion_fingerprint"] = decision.promotion.fingerprint
    return ClaimRow(
        claim_type="cuda-graph-promotion-selection",
        verdict="promoted" if decision.selected else "refused",
        scope=dict(scope or {}),
        custody=custody,
        source_job_ids=_current_job_ids(),
        detail=decision.reason
        + ("; blockers: " + "; ".join(decision.blockers) if decision.blockers else ""),
        created_ts=time.time(),
    )
