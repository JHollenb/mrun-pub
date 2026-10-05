"""Fail-closed evidence gates for WorkPlan promotion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .bundle import CompilationBundle


@dataclass(frozen=True)
class EvidenceRecord:
    requirement: str
    satisfied: bool
    source: str
    detail: str = ""
    subject_fingerprint: str | None = None
    result_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.requirement or not self.source:
            raise ValueError("evidence requirement and source must be non-empty")
        for field_name in ("subject_fingerprint", "result_fingerprint"):
            value = getattr(self, field_name)
            if value is not None and (
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "satisfied": self.satisfied,
            "source": self.source,
            "detail": self.detail,
            "subject_fingerprint": self.subject_fingerprint,
            "result_fingerprint": self.result_fingerprint,
        }


def evidence_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_is_bound(record: EvidenceRecord, bundle: CompilationBundle) -> bool:
    return (
        record.subject_fingerprint == bundle.fingerprint and record.result_fingerprint is not None
    )


@dataclass(frozen=True)
class PromotionReport:
    target: str
    promotable: bool
    satisfied_requirements: tuple[str, ...]
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "promotable": self.promotable,
            "satisfied_requirements": list(self.satisfied_requirements),
            "blockers": list(self.blockers),
        }


def evaluate_promotion(
    bundle: CompilationBundle,
    evidence: tuple[EvidenceRecord, ...] | list[EvidenceRecord] = (),
    *,
    target: str = "candidate",
) -> PromotionReport:
    """Evaluate reference/candidate/production gates without inferring missing evidence."""

    if target not in {"reference", "candidate", "production"}:
        raise ValueError("target must be one of: reference, candidate, production")
    records: dict[str, EvidenceRecord] = {}
    duplicate_requirements: set[str] = set()
    for record in evidence:
        if record.requirement in records:
            duplicate_requirements.add(record.requirement)
        records[record.requirement] = record

    blockers: list[str] = []
    satisfied: list[str] = []
    if duplicate_requirements:
        blockers.extend(
            f"duplicate evidence record: {requirement}"
            for requirement in sorted(duplicate_requirements)
        )
    if bundle.lowered.implementation_status != "eager-adapter":
        blockers.append(
            f"backend implementation is {bundle.lowered.implementation_status}, not executable"
        )
    if not bundle.lowered.content_identity_verified:
        blockers.append("model/store content identity is not verified")
    else:
        satisfied.append("content-identity-verified")
    if not bundle.lowered.placement_verified:
        blockers.append("execution placement is not verified")
    else:
        satisfied.append("placement-verified")

    if target in {"candidate", "production"}:
        for requirement in bundle.lowered.evidence_requirements:
            record = records.get(requirement)
            if record is None:
                blockers.append(f"missing evidence: {requirement}")
            elif not record.satisfied:
                blockers.append(f"failed evidence: {requirement} ({record.detail})")
            elif not _record_is_bound(record, bundle):
                blockers.append(f"unbound evidence: {requirement}")
            else:
                satisfied.append(requirement)
    if target == "production":
        artifact = records.get("compilation-artifact-checksum")
        if artifact is None:
            blockers.append("missing evidence: compilation-artifact-checksum")
        elif not artifact.satisfied:
            blockers.append(f"failed evidence: compilation-artifact-checksum ({artifact.detail})")
        elif not _record_is_bound(artifact, bundle):
            blockers.append("unbound evidence: compilation-artifact-checksum")
        else:
            satisfied.append("compilation-artifact-checksum")
        # ``capture_executed`` is deliberately never serialized into a compilation
        # artifact: it is a runtime verdict.  A captured production route must carry
        # an explicit execution record instead of promoting a compiler-side claim.
        if (
            bundle.lowered.capture_requested
            and "cuda-graph-capture-executed" not in bundle.lowered.evidence_requirements
        ):
            capture = records.get("cuda-graph-capture-executed")
            if capture is None:
                blockers.append("missing evidence: cuda-graph-capture-executed")
            elif not capture.satisfied:
                blockers.append(f"failed evidence: cuda-graph-capture-executed ({capture.detail})")
            elif not _record_is_bound(capture, bundle):
                blockers.append("unbound evidence: cuda-graph-capture-executed")
            else:
                satisfied.append("cuda-graph-capture-executed")

    return PromotionReport(
        target=target,
        promotable=not blockers,
        satisfied_requirements=tuple(dict.fromkeys(satisfied)),
        blockers=tuple(blockers),
    )
