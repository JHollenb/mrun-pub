"""Exact compatibility classes and deadline-bounded shape cohorts."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

COHORT_PLAN_SCHEMA = "mrun-sciencegraph-cohort-plan-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CohortRequest:
    request_id: str
    arena_fingerprint: str
    numerical_contract: str
    output_contract: str
    sequence_length: int
    selected_row_count: int
    deadline_ns: int

    def __post_init__(self) -> None:
        for field in (
            "request_id",
            "arena_fingerprint",
            "numerical_contract",
            "output_contract",
        ):
            if not str(getattr(self, field)):
                raise ValueError(f"{field} must be non-empty")
        for field in ("sequence_length", "selected_row_count", "deadline_ns"):
            if isinstance(getattr(self, field), bool) or int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")

    @property
    def compatibility_key(self) -> tuple[str, str, str]:
        return self.arena_fingerprint, self.numerical_contract, self.output_contract


@dataclass(frozen=True)
class ShapeBucketPolicy:
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    sequence_lengths: tuple[int, ...] = (1, 5, 16, 32, 128)
    selected_row_counts: tuple[int, ...] = (2, 6, 16, 64)
    coalescing_window_ns: int = 250_000

    def __post_init__(self) -> None:
        for field in ("batch_sizes", "sequence_lengths", "selected_row_counts"):
            values = tuple(int(value) for value in getattr(self, field))
            if not values or values != tuple(sorted(set(values))) or min(values) <= 0:
                raise ValueError(f"{field} must be strictly increasing positive buckets")
            object.__setattr__(self, field, values)
        if self.coalescing_window_ns < 0:
            raise ValueError("coalescing window must be non-negative")

    @staticmethod
    def _admit(value: int, buckets: tuple[int, ...], field: str) -> int:
        for bucket in buckets:
            if value <= bucket:
                return bucket
        raise OverflowError(f"{field} exceeds every admitted shape bucket")

    def bucket_for(self, *, batch: int, sequence: int, selected_rows: int) -> tuple[int, int, int]:
        return (
            self._admit(batch, self.batch_sizes, "batch"),
            self._admit(sequence, self.sequence_lengths, "sequence"),
            self._admit(selected_rows, self.selected_row_counts, "selected rows"),
        )


@dataclass(frozen=True)
class AmortizationGroup:
    request_ids: tuple[str, ...]
    compatibility_key: tuple[str, str, str]
    bucket: tuple[int, int, int]
    earliest_deadline_ns: int
    padded_sequence_tokens: int
    padded_selected_rows: int

    def __post_init__(self) -> None:
        if not self.request_ids or len(self.request_ids) != len(set(self.request_ids)):
            raise ValueError("cohort request IDs must be non-empty and unique")
        if self.bucket[0] < len(self.request_ids):
            raise ValueError("cohort batch bucket is smaller than its membership")


@dataclass(frozen=True)
class CohortPlan:
    groups: tuple[AmortizationGroup, ...]
    logical_requests: int
    physical_cohorts: int
    schema: str = COHORT_PLAN_SCHEMA

    @property
    def fingerprint(self) -> str:
        return _digest(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "groups": [asdict(group) for group in self.groups],
            "logical_requests": self.logical_requests,
            "physical_cohorts": self.physical_cohorts,
            "semantic_work_deleted": 0,
            "weight_traversals_amortized": self.logical_requests - self.physical_cohorts,
            "residency_savings_claimed": 0,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


def form_amortization_groups(
    requests: tuple[CohortRequest, ...],
    policy: ShapeBucketPolicy | None = None,
) -> CohortPlan:
    """Form stable compatible cohorts without allowing queueing beyond a deadline."""

    policy = ShapeBucketPolicy() if policy is None else policy
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("cohort request IDs must be globally unique")
    compatible: OrderedDict[tuple[str, str, str], list[CohortRequest]] = OrderedDict()
    for request in requests:
        compatible.setdefault(request.compatibility_key, []).append(request)
    groups: list[AmortizationGroup] = []
    for compatibility_key, members in compatible.items():
        members.sort(key=lambda item: item.deadline_ns)
        pending: list[CohortRequest] = []
        first_deadline = 0
        for member in members:
            if not pending:
                first_deadline = member.deadline_ns
            exceeds_window = member.deadline_ns - first_deadline > policy.coalescing_window_ns
            exceeds_batch = len(pending) >= policy.batch_sizes[-1]
            if pending and (exceeds_window or exceeds_batch):
                groups.append(_build_group(pending, compatibility_key, policy))
                pending = []
                first_deadline = member.deadline_ns
            pending.append(member)
        if pending:
            groups.append(_build_group(pending, compatibility_key, policy))
    return CohortPlan(
        groups=tuple(groups),
        logical_requests=len(requests),
        physical_cohorts=len(groups),
    )


def _build_group(
    members: list[CohortRequest],
    compatibility_key: tuple[str, str, str],
    policy: ShapeBucketPolicy,
) -> AmortizationGroup:
    max_sequence = max(member.sequence_length for member in members)
    max_rows = max(member.selected_row_count for member in members)
    bucket = policy.bucket_for(
        batch=len(members),
        sequence=max_sequence,
        selected_rows=max_rows,
    )
    return AmortizationGroup(
        request_ids=tuple(member.request_id for member in members),
        compatibility_key=compatibility_key,
        bucket=bucket,
        earliest_deadline_ns=min(member.deadline_ns for member in members),
        padded_sequence_tokens=sum(bucket[1] - member.sequence_length for member in members),
        padded_selected_rows=sum(bucket[2] - member.selected_row_count for member in members),
    )


__all__ = [
    "COHORT_PLAN_SCHEMA",
    "AmortizationGroup",
    "CohortPlan",
    "CohortRequest",
    "ShapeBucketPolicy",
    "form_amortization_groups",
]
