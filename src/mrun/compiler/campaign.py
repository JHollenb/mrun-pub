"""Graph-compiled candidate campaigns for one immutable model input.

A candidate campaign is a deliberately narrow multi-query supergraph.  Several
logical readouts ask different candidate questions about the *same* token sequence,
model/store identity, and numerical contract.  The compiler evaluates the transformer
body once, evaluates every distinct vocabulary row once, and projects those scores
back into each query's original candidate order before applying the established
top-two/margin reduction.

This module does not merge stateful requests, different token sequences, patches,
RNG-dependent work, or merely similar graphs.  Those require stronger temporal and
effect proofs than WorkPlan v2 currently carries.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from .bundle import CompilationBundle, compile_work_plan
from .executable import (
    _validate_runtime_configuration,
    candidate_outputs_from_values,
)
from .identity import bind_loaded_qstore_identity
from .ir import ExecutionMode, OutputContract

CANDIDATE_CAMPAIGN_SCHEMA = "mrun-candidate-campaign-v1"
CAMPAIGN_INPUT_BINDING_SCHEMA = "mrun-campaign-input-binding-v1"
CAMPAIGN_SHARING_SCHEMA = "mrun-campaign-sharing-certificate-v1"
_CAMPAIGN_LOWERER_BY_ENGINE = {
    "paged": "paged-qstore",
    "dense-qstore-cuda": "cuda-qstore",
}
_CAMPAIGN_ENGINE_BY_LOWERER = {
    lowerer: engine for engine, lowerer in _CAMPAIGN_LOWERER_BY_ENGINE.items()
}


def _require_name(value: Any, field_name: str) -> str:
    result = str(value)
    if not result or result.strip() != result:
        raise ValueError(f"{field_name} must be a non-empty string without outer whitespace")
    return result


def _require_sha256(value: Any, field_name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return result


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_evidence(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("campaign execution evidence cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _require_name(key, "campaign evidence key"): _freeze_evidence(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(_freeze_evidence(item) for item in value)
    raise TypeError("campaign execution evidence must contain only JSON-compatible values")


def _thaw_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_evidence(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_evidence(item) for item in value]
    return value


def _token_array(token_ids: np.ndarray | Sequence[int]) -> np.ndarray:
    result = np.asarray(token_ids, dtype=np.int64)
    if result.ndim != 1 or not result.size:
        raise ValueError("campaign token IDs must be a non-empty one-dimensional array")
    if np.any(result < 0):
        raise ValueError("campaign token IDs must be non-negative")
    return np.ascontiguousarray(result)


def _token_digest(token_ids: np.ndarray) -> str:
    canonical = np.asarray(token_ids, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(b"mrun-candidate-campaign-token-ids-v1\0")
    digest.update(int(canonical.size).to_bytes(8, "little", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateReadout:
    """One logical candidate question, preserving caller-specified candidate order."""

    query_id: str
    candidate_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _require_name(self.query_id, "query_id"))
        candidates = tuple(int(value) for value in self.candidate_token_ids)
        if len(candidates) < 2:
            raise ValueError("candidate readouts require at least two candidate token IDs")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate token IDs must be unique inside each readout")
        if any(value < 0 for value in candidates):
            raise ValueError("candidate token IDs must be non-negative")
        object.__setattr__(self, "candidate_token_ids", candidates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "candidate_token_ids": list(self.candidate_token_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateReadout:
        return cls(
            query_id=str(payload["query_id"]),
            candidate_token_ids=tuple(int(value) for value in payload["candidate_token_ids"]),
        )


@dataclass(frozen=True)
class CampaignInputBinding:
    """Content binding for the one immutable token sequence shared by all readouts."""

    token_sha256: str
    token_count: int
    token_dtype: str = "int64-le"
    schema_version: str = CAMPAIGN_INPUT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_INPUT_BINDING_SCHEMA:
            raise ValueError(f"unsupported campaign input binding schema: {self.schema_version}")
        object.__setattr__(
            self,
            "token_sha256",
            _require_sha256(self.token_sha256, "token_sha256"),
        )
        object.__setattr__(self, "token_count", int(self.token_count))
        if self.token_count <= 0:
            raise ValueError("campaign input binding token count must be positive")
        if self.token_dtype != "int64-le":
            raise ValueError("campaign input binding dtype must be int64-le")

    @classmethod
    def from_token_ids(
        cls,
        token_ids: np.ndarray | Sequence[int],
    ) -> CampaignInputBinding:
        values = _token_array(token_ids)
        return cls(token_sha256=_token_digest(values), token_count=int(values.size))

    def matches(self, token_ids: np.ndarray | Sequence[int]) -> bool:
        values = _token_array(token_ids)
        return int(values.size) == self.token_count and _token_digest(values) == (self.token_sha256)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "token_sha256": self.token_sha256,
            "token_count": self.token_count,
            "token_dtype": self.token_dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CampaignInputBinding:
        return cls(
            token_sha256=str(payload["token_sha256"]),
            token_count=int(payload["token_count"]),
            token_dtype=str(payload.get("token_dtype", "int64-le")),
            schema_version=str(payload.get("schema_version", CAMPAIGN_INPUT_BINDING_SCHEMA)),
        )


@dataclass(frozen=True)
class CandidateOverlapEdge:
    """An edge in the query-overlap graph, labelled by shared vocabulary rows."""

    left_query_id: str
    right_query_id: str
    shared_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        left = _require_name(self.left_query_id, "left_query_id")
        right = _require_name(self.right_query_id, "right_query_id")
        if left >= right:
            raise ValueError("candidate overlap edge query IDs must be canonical")
        shared = tuple(int(value) for value in self.shared_token_ids)
        if not shared or len(shared) != len(set(shared)) or any(value < 0 for value in shared):
            raise ValueError("candidate overlap edge tokens must be unique and non-negative")
        object.__setattr__(self, "left_query_id", left)
        object.__setattr__(self, "right_query_id", right)
        object.__setattr__(self, "shared_token_ids", shared)

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_query_id": self.left_query_id,
            "right_query_id": self.right_query_id,
            "shared_token_ids": list(self.shared_token_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateOverlapEdge:
        return cls(
            left_query_id=str(payload["left_query_id"]),
            right_query_id=str(payload["right_query_id"]),
            shared_token_ids=tuple(int(value) for value in payload["shared_token_ids"]),
        )


@dataclass(frozen=True)
class CandidateQueryProjection:
    """Total projection from one query's candidate order into stable union offsets."""

    query_id: str
    union_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _require_name(self.query_id, "query_id"))
        offsets = tuple(int(value) for value in self.union_offsets)
        if len(offsets) < 2 or len(offsets) != len(set(offsets)):
            raise ValueError("candidate projection offsets must be unique with length >= 2")
        if any(value < 0 for value in offsets):
            raise ValueError("candidate projection offsets must be non-negative")
        object.__setattr__(self, "union_offsets", offsets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "union_offsets": list(self.union_offsets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateQueryProjection:
        return cls(
            query_id=str(payload["query_id"]),
            union_offsets=tuple(int(value) for value in payload["union_offsets"]),
        )


@dataclass(frozen=True)
class CandidateSharingCertificate:
    """Structural savings certificate for the campaign's shared-value quotient."""

    query_count: int
    candidate_reference_count: int
    union_candidate_count: int
    duplicate_candidate_reference_count: int
    shared_candidate_token_count: int
    max_candidate_multiplicity: int
    overlap_edges: tuple[CandidateOverlapEdge, ...]
    overlap_component_count: int
    query_projections: tuple[CandidateQueryProjection, ...]
    independent_body_evaluations: int
    campaign_body_evaluations: int
    independent_head_score_evaluations: int
    campaign_head_score_evaluations: int
    assumptions: tuple[str, ...] = (
        "all-queries-bind-the-identical-token-sequence",
        "model-store-and-numerical-contract-are-identical",
        "compiled-graph-is-pure-and-stateless",
        "candidate-logits-are-deterministic-functions-of-shared-hidden-state-and-head-row",
    )
    limits: tuple[str, ...] = (
        "structural-counts-not-runtime-traffic",
        "structural-counts-not-latency-or-energy",
        "syntactic-typed-sharing-not-general-semantic-equivalence",
        "no-stateful-decode-patches-rng-or-kv-sharing",
    )
    schema_version: str = CAMPAIGN_SHARING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_SHARING_SCHEMA:
            raise ValueError(f"unsupported campaign sharing schema: {self.schema_version}")
        for field_name in (
            "query_count",
            "candidate_reference_count",
            "union_candidate_count",
            "duplicate_candidate_reference_count",
            "shared_candidate_token_count",
            "max_candidate_multiplicity",
            "overlap_component_count",
            "independent_body_evaluations",
            "campaign_body_evaluations",
            "independent_head_score_evaluations",
            "campaign_head_score_evaluations",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
            if getattr(self, field_name) < 0:
                raise ValueError(f"campaign sharing {field_name} cannot be negative")
        if self.query_count < 2:
            raise ValueError("candidate campaigns require at least two logical queries")
        if self.union_candidate_count <= 0:
            raise ValueError("candidate campaign union cannot be empty")
        if (
            self.duplicate_candidate_reference_count
            != self.candidate_reference_count - self.union_candidate_count
            or self.independent_body_evaluations != self.query_count
            or self.campaign_body_evaluations != 1
            or self.independent_head_score_evaluations != self.candidate_reference_count
            or self.campaign_head_score_evaluations != self.union_candidate_count
        ):
            raise ValueError("candidate sharing structural counts are inconsistent")
        if not 1 <= self.overlap_component_count <= self.query_count:
            raise ValueError("candidate overlap component count is inconsistent")
        edge_keys = [(edge.left_query_id, edge.right_query_id) for edge in self.overlap_edges]
        if len(edge_keys) != len(set(edge_keys)) or edge_keys != sorted(edge_keys):
            raise ValueError("candidate overlap edges must be unique and canonical")
        projection_ids = [projection.query_id for projection in self.query_projections]
        if (
            len(projection_ids) != self.query_count
            or len(projection_ids) != len(set(projection_ids))
            or sum(len(projection.union_offsets) for projection in self.query_projections)
            != self.candidate_reference_count
            or any(
                offset >= self.union_candidate_count
                for projection in self.query_projections
                for offset in projection.union_offsets
            )
        ):
            raise ValueError("candidate query projections are incomplete or inconsistent")
        for field_name in ("assumptions", "limits"):
            values = tuple(str(value) for value in getattr(self, field_name))
            if not values or len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"candidate sharing {field_name} must be unique and non-empty")
            object.__setattr__(self, field_name, values)

    @property
    def eliminated_body_evaluations(self) -> int:
        return self.independent_body_evaluations - self.campaign_body_evaluations

    @property
    def eliminated_head_score_evaluations(self) -> int:
        return self.independent_head_score_evaluations - self.campaign_head_score_evaluations

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_count": self.query_count,
            "candidate_reference_count": self.candidate_reference_count,
            "union_candidate_count": self.union_candidate_count,
            "duplicate_candidate_reference_count": (self.duplicate_candidate_reference_count),
            "shared_candidate_token_count": self.shared_candidate_token_count,
            "max_candidate_multiplicity": self.max_candidate_multiplicity,
            "overlap_edges": [edge.as_dict() for edge in self.overlap_edges],
            "overlap_component_count": self.overlap_component_count,
            "query_projections": [projection.as_dict() for projection in self.query_projections],
            "independent_body_evaluations": self.independent_body_evaluations,
            "campaign_body_evaluations": self.campaign_body_evaluations,
            "eliminated_body_evaluations": self.eliminated_body_evaluations,
            "independent_head_score_evaluations": (self.independent_head_score_evaluations),
            "campaign_head_score_evaluations": self.campaign_head_score_evaluations,
            "eliminated_head_score_evaluations": (self.eliminated_head_score_evaluations),
            "assumptions": list(self.assumptions),
            "limits": list(self.limits),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateSharingCertificate:
        result = cls(
            query_count=int(payload["query_count"]),
            candidate_reference_count=int(payload["candidate_reference_count"]),
            union_candidate_count=int(payload["union_candidate_count"]),
            duplicate_candidate_reference_count=int(payload["duplicate_candidate_reference_count"]),
            shared_candidate_token_count=int(payload["shared_candidate_token_count"]),
            max_candidate_multiplicity=int(payload["max_candidate_multiplicity"]),
            overlap_edges=tuple(
                CandidateOverlapEdge.from_dict(value) for value in payload.get("overlap_edges", ())
            ),
            overlap_component_count=int(payload["overlap_component_count"]),
            query_projections=tuple(
                CandidateQueryProjection.from_dict(value)
                for value in payload.get("query_projections", ())
            ),
            independent_body_evaluations=int(payload["independent_body_evaluations"]),
            campaign_body_evaluations=int(payload["campaign_body_evaluations"]),
            independent_head_score_evaluations=int(payload["independent_head_score_evaluations"]),
            campaign_head_score_evaluations=int(payload["campaign_head_score_evaluations"]),
            assumptions=tuple(str(value) for value in payload.get("assumptions", ())),
            limits=tuple(str(value) for value in payload.get("limits", ())),
            schema_version=str(payload.get("schema_version", CAMPAIGN_SHARING_SCHEMA)),
        )
        if (
            payload.get("eliminated_body_evaluations") is not None
            and int(payload["eliminated_body_evaluations"]) != result.eliminated_body_evaluations
        ):
            raise ValueError("campaign eliminated body count mismatch")
        if (
            payload.get("eliminated_head_score_evaluations") is not None
            and int(payload["eliminated_head_score_evaluations"])
            != result.eliminated_head_score_evaluations
        ):
            raise ValueError("campaign eliminated head-score count mismatch")
        return result


def _overlap_component_count(
    query_ids: tuple[str, ...],
    edges: tuple[CandidateOverlapEdge, ...],
) -> int:
    parents = {query_id: query_id for query_id in query_ids}

    def find(query_id: str) -> str:
        while parents[query_id] != query_id:
            parents[query_id] = parents[parents[query_id]]
            query_id = parents[query_id]
        return query_id

    for edge in edges:
        left = find(edge.left_query_id)
        right = find(edge.right_query_id)
        if left != right:
            parents[max(left, right)] = min(left, right)
    return len({find(query_id) for query_id in query_ids})


def _sharing_certificate(
    readouts: tuple[CandidateReadout, ...],
) -> CandidateSharingCertificate:
    query_ids = tuple(readout.query_id for readout in readouts)
    multiplicities = Counter(token for readout in readouts for token in readout.candidate_token_ids)
    union = tuple(multiplicities)
    by_query = {readout.query_id: set(readout.candidate_token_ids) for readout in readouts}
    edges: list[CandidateOverlapEdge] = []
    for left_index, left_query_id in enumerate(sorted(query_ids)):
        for right_query_id in sorted(query_ids)[left_index + 1 :]:
            shared = tuple(sorted(by_query[left_query_id].intersection(by_query[right_query_id])))
            if shared:
                edges.append(
                    CandidateOverlapEdge(
                        left_query_id=left_query_id,
                        right_query_id=right_query_id,
                        shared_token_ids=shared,
                    )
                )
    materialized_edges = tuple(edges)
    references = sum(len(readout.candidate_token_ids) for readout in readouts)
    stable_union = _stable_candidate_union(readouts)
    union_offsets = {token: index for index, token in enumerate(stable_union)}
    return CandidateSharingCertificate(
        query_count=len(readouts),
        candidate_reference_count=references,
        union_candidate_count=len(union),
        duplicate_candidate_reference_count=references - len(union),
        shared_candidate_token_count=sum(count > 1 for count in multiplicities.values()),
        max_candidate_multiplicity=max(multiplicities.values()),
        overlap_edges=materialized_edges,
        overlap_component_count=_overlap_component_count(
            query_ids,
            materialized_edges,
        ),
        query_projections=tuple(
            CandidateQueryProjection(
                query_id=readout.query_id,
                union_offsets=tuple(union_offsets[token] for token in readout.candidate_token_ids),
            )
            for readout in readouts
        ),
        independent_body_evaluations=len(readouts),
        campaign_body_evaluations=1,
        independent_head_score_evaluations=references,
        campaign_head_score_evaluations=len(union),
    )


def _stable_candidate_union(
    readouts: Sequence[CandidateReadout],
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(token for readout in readouts for token in readout.candidate_token_ids)
    )


@dataclass(frozen=True)
class CandidateCampaign:
    """Serializable compiled supergraph for candidate readouts over one input."""

    input_binding: CampaignInputBinding
    readouts: tuple[CandidateReadout, ...]
    union_token_ids: tuple[int, ...]
    base_bundle: CompilationBundle
    sharing: CandidateSharingCertificate
    schema_version: str = CANDIDATE_CAMPAIGN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_CAMPAIGN_SCHEMA:
            raise ValueError(f"unsupported candidate campaign schema: {self.schema_version}")
        readouts = tuple(self.readouts)
        if len(readouts) < 2:
            raise ValueError("candidate campaigns require at least two readouts")
        query_ids = [readout.query_id for readout in readouts]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("candidate campaign query IDs must be unique")
        object.__setattr__(self, "readouts", readouts)
        union = tuple(int(value) for value in self.union_token_ids)
        expected_union = _stable_candidate_union(readouts)
        if union != expected_union:
            raise ValueError("candidate campaign union is not stable first-use order")
        object.__setattr__(self, "union_token_ids", union)
        expected_sharing = _sharing_certificate(readouts)
        if self.sharing != expected_sharing:
            raise ValueError("candidate campaign sharing certificate is inconsistent")

        plan = self.base_bundle.plan
        lowered = self.base_bundle.lowered
        expected_engine_backend = _CAMPAIGN_ENGINE_BY_LOWERER.get(lowered.backend)
        if expected_engine_backend is None:
            raise ValueError("candidate campaign bundle uses an unsupported lowerer")
        if dict(plan.metadata).get("engine_backend") != expected_engine_backend:
            raise ValueError("candidate campaign plan engine backend does not match its lowerer")
        if lowered.implementation_status != "eager-adapter":
            raise ValueError("candidate campaign requires an executable eager adapter")
        if lowered.content_identity_verified != plan.content_identity_verified:
            raise ValueError("candidate campaign lowering content identity is inconsistent")
        if lowered.capture_executed:
            raise ValueError("capture execution is a runtime verdict, not a serialized claim")
        if plan.capture.requested and not lowered.capture_ready:
            raise ValueError("candidate campaign requested capture but its lowerer is not ready")
        if plan.output_contract is not OutputContract.SELECTED_TOKEN_ROWS:
            raise ValueError("candidate campaign base plan must return selected token rows")
        if plan.execution_mode is not ExecutionMode.SCORE:
            raise ValueError("candidate campaign base plan must use stateless score mode")
        if plan.shape.actual_batch != 1:
            raise ValueError("candidate campaign currently requires exactly one token input")
        if plan.shape.sequence_length != self.input_binding.token_count:
            raise ValueError("candidate campaign input binding does not match plan shape")
        if plan.required_output_rows != union:
            raise ValueError("candidate campaign base plan does not bind the stable union")
        if plan.candidate_token_ids:
            raise ValueError("candidate campaign base plan cannot carry row-local candidates")
        if plan.prefix_state_ids or plan.kv_read_handles or plan.kv_write_handles:
            raise ValueError("candidate campaign cannot bind prefix or KV state")
        if not bool(dict(plan.metadata).get("output_pushdown", False)):
            raise ValueError("candidate campaign requires selected-row output pushdown")

        compilation = self.base_bundle.graph
        if compilation is None or compilation.rewrite_certificate is None:
            raise ValueError("candidate campaign requires a certified compiled graph")
        if compilation.rewrite_certificate.output_contract != (
            OutputContract.SELECTED_TOKEN_ROWS.value
        ):
            raise ValueError("candidate campaign graph has the wrong output contract")
        if any(node.effects for node in compilation.graph.nodes):
            raise ValueError("candidate campaign graph must be effect-free")
        head_nodes = [
            node for node in compilation.graph.nodes if node.node_id == "output.vocab_projection"
        ]
        if len(head_nodes) != 1 or len(head_nodes[0].parameters) != 1:
            raise ValueError("candidate campaign graph must contain one vocabulary head")
        head = head_nodes[0].parameters[0]
        if head.access != "rows" or head.row_indices != union:
            raise ValueError("candidate campaign graph head does not bind the stable union")
        if any(token >= head.shape[0] for token in union):
            raise ValueError("candidate campaign token exceeds compiled vocabulary size")

        work_floor = self.base_bundle.work_floor
        if work_floor is None:
            raise ValueError("candidate campaign requires a work-floor comparison")
        head_floor = work_floor.rewritten.certificate.head_demand
        if not head_floor.complete or not head_floor.attained:
            raise ValueError(
                "candidate campaign selected-row head has not attained its scoped work floor"
            )

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_binding": self.input_binding.as_dict(),
            "readouts": [readout.as_dict() for readout in self.readouts],
            "union_token_ids": list(self.union_token_ids),
            "base_bundle": self.base_bundle.as_dict(),
            "sharing": self.sharing.as_dict(),
        }

    @cached_property
    def fingerprint(self) -> str:
        return _hash_payload(self._payload_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload_dict(),
            "campaign_fingerprint": self.fingerprint,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateCampaign:
        result = cls(
            input_binding=CampaignInputBinding.from_dict(payload["input_binding"]),
            readouts=tuple(
                CandidateReadout.from_dict(value) for value in payload.get("readouts", ())
            ),
            union_token_ids=tuple(int(value) for value in payload["union_token_ids"]),
            base_bundle=CompilationBundle.from_dict(payload["base_bundle"]),
            sharing=CandidateSharingCertificate.from_dict(payload["sharing"]),
            schema_version=str(payload.get("schema_version", CANDIDATE_CAMPAIGN_SCHEMA)),
        )
        claimed = payload.get("campaign_fingerprint")
        if claimed is not None and str(claimed) != result.fingerprint:
            raise ValueError("candidate campaign fingerprint mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> CandidateCampaign:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("serialized candidate campaign must be an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class CandidateReadoutResult:
    query_id: str
    candidate_token_ids: tuple[int, ...]
    candidate_logits: tuple[float, ...]
    winner_token_id: int
    runner_up_token_id: int
    winner_logit: float
    runner_up_logit: float
    margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _require_name(self.query_id, "query_id"))
        candidates = tuple(int(value) for value in self.candidate_token_ids)
        logits = tuple(float(value) for value in self.candidate_logits)
        if len(candidates) < 2 or len(candidates) != len(logits):
            raise ValueError("candidate result IDs and logits must have equal length >= 2")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate result IDs must be unique")
        if any(not math.isfinite(value) for value in logits):
            raise ValueError("candidate result logits must be finite")
        object.__setattr__(self, "candidate_token_ids", candidates)
        object.__setattr__(self, "candidate_logits", logits)
        for field_name in (
            "winner_token_id",
            "runner_up_token_id",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
            if getattr(self, field_name) not in candidates:
                raise ValueError(f"{field_name} is not a candidate token")
        if self.winner_token_id == self.runner_up_token_id:
            raise ValueError("winner and runner-up token IDs must differ")
        for field_name in ("winner_logit", "runner_up_logit", "margin"):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if not math.isclose(
            self.margin,
            self.winner_logit - self.runner_up_logit,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("candidate result margin is inconsistent")
        winner_value = logits[candidates.index(self.winner_token_id)]
        runner_value = logits[candidates.index(self.runner_up_token_id)]
        if not math.isclose(
            self.winner_logit,
            winner_value,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ) or not math.isclose(
            self.runner_up_logit,
            runner_value,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError("candidate result top-two logits do not match candidate values")
        if self.winner_logit < self.runner_up_logit or any(
            value > self.runner_up_logit
            for index, value in enumerate(logits)
            if candidates[index] != self.winner_token_id
        ):
            raise ValueError("candidate result winner/runner-up are not a valid top two")

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "candidate_token_ids": list(self.candidate_token_ids),
            "candidate_logits": list(self.candidate_logits),
            "winner_token_id": self.winner_token_id,
            "runner_up_token_id": self.runner_up_token_id,
            "winner_logit": self.winner_logit,
            "runner_up_logit": self.runner_up_logit,
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateReadoutResult:
        return cls(
            query_id=str(payload["query_id"]),
            candidate_token_ids=tuple(int(value) for value in payload["candidate_token_ids"]),
            candidate_logits=tuple(float(value) for value in payload["candidate_logits"]),
            winner_token_id=int(payload["winner_token_id"]),
            runner_up_token_id=int(payload["runner_up_token_id"]),
            winner_logit=float(payload["winner_logit"]),
            runner_up_logit=float(payload["runner_up_logit"]),
            margin=float(payload["margin"]),
        )


@dataclass(frozen=True)
class CandidateCampaignExecutionResult:
    campaign_fingerprint: str
    union_token_ids: tuple[int, ...]
    union_logits: tuple[float, ...]
    readouts: tuple[CandidateReadoutResult, ...]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "campaign_fingerprint",
            _require_sha256(self.campaign_fingerprint, "campaign_fingerprint"),
        )
        union = tuple(int(value) for value in self.union_token_ids)
        logits = tuple(float(value) for value in self.union_logits)
        if not union or len(union) != len(logits) or len(union) != len(set(union)):
            raise ValueError("campaign execution union IDs/logits are inconsistent")
        if any(not math.isfinite(value) for value in logits):
            raise ValueError("campaign execution union logits must be finite")
        object.__setattr__(self, "union_token_ids", union)
        object.__setattr__(self, "union_logits", logits)
        readouts = tuple(self.readouts)
        if len(readouts) < 2:
            raise ValueError("campaign execution must contain at least two readouts")
        if len({readout.query_id for readout in readouts}) != len(readouts):
            raise ValueError("campaign execution readout query IDs must be unique")
        object.__setattr__(self, "readouts", readouts)
        if not isinstance(self.evidence, Mapping):
            raise TypeError("campaign execution evidence must be an object")
        evidence = _freeze_evidence(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if int(evidence.get("body_execution_count", -1)) != 1:
            raise ValueError("campaign execution evidence must record one body execution")

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_fingerprint": self.campaign_fingerprint,
            "union_token_ids": list(self.union_token_ids),
            "union_logits": list(self.union_logits),
            "readouts": [readout.as_dict() for readout in self.readouts],
            "evidence": _thaw_evidence(self.evidence),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> CandidateCampaignExecutionResult:
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TypeError("campaign execution evidence must be an object")
        return cls(
            campaign_fingerprint=str(payload["campaign_fingerprint"]),
            union_token_ids=tuple(int(value) for value in payload["union_token_ids"]),
            union_logits=tuple(float(value) for value in payload["union_logits"]),
            readouts=tuple(
                CandidateReadoutResult.from_dict(value) for value in payload.get("readouts", ())
            ),
            evidence=evidence,
        )


def _manifest_vocab_size(manifest: Mapping[str, Any]) -> int:
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("candidate campaign requires a manifest config")
    vocab_size = int(config.get("vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("candidate campaign requires a positive manifest vocabulary size")
    return vocab_size


def compile_candidate_campaign(
    engine: Any,
    token_ids: np.ndarray | Sequence[int],
    readouts: Sequence[CandidateReadout],
    *,
    capture_requested: bool = False,
) -> CandidateCampaign:
    """Compile a pure, stateless, one-input candidate-readout supergraph."""

    engine_backend = str(getattr(engine, "backend", ""))
    lowering_backend = _CAMPAIGN_LOWERER_BY_ENGINE.get(engine_backend)
    if lowering_backend is None:
        raise NotImplementedError(
            "candidate campaign execution requires paged or dense-qstore-cuda"
        )
    if capture_requested and engine_backend != "dense-qstore-cuda":
        raise ValueError("candidate campaign graph capture requires dense-qstore-cuda")
    selected = getattr(engine, "selected_last_logits_batch", None)
    if not callable(selected):
        raise ValueError("engine does not support selected-row output pushdown")
    plan_builder = getattr(engine, "build_work_plan", None)
    if not callable(plan_builder):
        raise ValueError("engine does not expose WorkPlan compilation")
    store = getattr(engine, "store", None)
    manifest = getattr(store, "man", None)
    if not isinstance(manifest, dict):
        raise ValueError("candidate campaign requires an in-scope QStore manifest")

    values = _token_array(token_ids)
    materialized = tuple(readouts)
    if any(not isinstance(readout, CandidateReadout) for readout in materialized):
        raise TypeError("candidate campaign readouts must be CandidateReadout objects")
    if len(materialized) < 2:
        raise ValueError("candidate campaigns require at least two readouts")
    query_ids = [readout.query_id for readout in materialized]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("candidate campaign query IDs must be unique")
    union = _stable_candidate_union(materialized)
    vocab_size = _manifest_vocab_size(manifest)
    if any(token >= vocab_size for token in values):
        raise ValueError("campaign input token exceeds the manifest vocabulary size")
    if any(token >= vocab_size for token in union):
        raise ValueError("candidate token exceeds the manifest vocabulary size")

    input_binding = CampaignInputBinding.from_token_ids(values)
    plan_kwargs: dict[str, Any] = {
        "execution_mode": ExecutionMode.SCORE,
        "output_contract": OutputContract.SELECTED_TOKEN_ROWS,
        "required_output_rows": union,
        "request_ids": (f"campaign-{input_binding.token_sha256[:16]}",),
        "capture_requested": capture_requested,
    }
    if engine_backend == "dense-qstore-cuda":
        plan_kwargs.update(
            stable_addresses=capture_requested,
            graph_safe=capture_requested,
        )
    plan = plan_builder([values], **plan_kwargs)
    if dict(plan.metadata).get("engine_backend") != engine_backend:
        raise RuntimeError("candidate campaign plan builder changed the engine backend")
    if capture_requested and not plan.content_identity_verified:
        raise ValueError("CUDA Graph campaigns require a semantically and blob-verified QStore")
    bundle = compile_work_plan(
        plan,
        lowering_backend,
        manifest=manifest,
    )
    return CandidateCampaign(
        input_binding=input_binding,
        readouts=materialized,
        union_token_ids=union,
        base_bundle=bundle,
        sharing=_sharing_certificate(materialized),
    )


def validate_candidate_campaign_engine_binding(
    engine: Any,
    campaign: CandidateCampaign,
) -> dict[str, str]:
    expected_engine_backend = _CAMPAIGN_ENGINE_BY_LOWERER.get(campaign.base_bundle.lowered.backend)
    if expected_engine_backend is None:
        raise RuntimeError("candidate campaign bundle has an unsupported lowerer")
    if str(getattr(engine, "backend", "")) != expected_engine_backend:
        raise RuntimeError("runtime engine backend does not match the compiled campaign lowerer")
    selected = getattr(engine, "selected_last_logits_batch", None)
    if not callable(selected):
        raise RuntimeError("engine cannot execute selected vocabulary rows")
    store = getattr(engine, "store", None)
    manifest = getattr(store, "man", None)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("engine has no QStore manifest")
    plan = campaign.base_bundle.plan
    identity = bind_loaded_qstore_identity(engine)
    if (
        plan.model_name != identity.model_name
        or plan.model_revision != identity.model_revision
        or plan.store_fingerprint != identity.store_fingerprint
    ):
        raise RuntimeError("runtime engine identity does not match the compiled campaign")
    supported_contracts = tuple(
        str(value)
        for value in getattr(
            engine,
            "supported_numerical_contracts",
            (getattr(engine, "numerical_contract", ""),),
        )
    )
    if plan.numerical_contract not in supported_contracts:
        raise RuntimeError("runtime engine does not support the campaign numerical contract")
    vocab_size = _manifest_vocab_size(manifest)
    if any(token >= vocab_size for token in campaign.union_token_ids):
        raise RuntimeError("runtime engine vocabulary is smaller than the campaign union")
    return _validate_runtime_configuration(
        engine,
        plan,
        reported_fabric=campaign.base_bundle.lowered.reported_fabric,
    )


def _campaign_result_from_union_scores(
    campaign: CandidateCampaign,
    scores: torch.Tensor,
    evidence: Mapping[str, Any],
) -> CandidateCampaignExecutionResult:
    if scores.ndim != 2 or tuple(scores.shape) != (1, len(campaign.union_token_ids)):
        raise RuntimeError("candidate campaign engine returned an invalid selected-row score shape")
    union_scores = scores[0]
    local_values = [
        union_scores.index_select(
            0,
            torch.as_tensor(
                projection.union_offsets,
                dtype=torch.long,
                device=union_scores.device,
            ),
        )
        for _readout, projection in zip(
            campaign.readouts,
            campaign.sharing.query_projections,
            strict=True,
        )
    ]
    summaries = candidate_outputs_from_values(
        local_values,
        tuple(readout.candidate_token_ids for readout in campaign.readouts),
    )
    readout_results = tuple(
        CandidateReadoutResult(
            query_id=readout.query_id,
            candidate_token_ids=readout.candidate_token_ids,
            candidate_logits=tuple(float(value) for value in local.detach().float().cpu()),
            winner_token_id=int(summary["winner_token_id"]),
            runner_up_token_id=int(summary["runner_up_token_id"]),
            winner_logit=float(summary["winner_logit"]),
            runner_up_logit=float(summary["runner_up_logit"]),
            margin=float(summary["margin"]),
        )
        for readout, local, summary in zip(
            campaign.readouts,
            local_values,
            summaries,
            strict=True,
        )
    )
    return CandidateCampaignExecutionResult(
        campaign_fingerprint=campaign.fingerprint,
        union_token_ids=campaign.union_token_ids,
        union_logits=tuple(float(value) for value in union_scores.detach().cpu()),
        readouts=readout_results,
        evidence=evidence,
    )


class PreparedCandidateCampaign:
    """Validated hot executor for compile-once, dispatch-many campaign replay.

    Preparation binds the engine identity, selected-row entry point, immutable token
    content, bundle, graph certificate, and WorkFloor outside the hot path.  Replay
    still validates the returned tensor shape and reconstructs every query-local
    public result.
    """

    __slots__ = (
        "_campaign",
        "_capture_executor",
        "_closed",
        "_evidence",
        "_selected",
        "_token_ids",
    )

    def __init__(
        self,
        engine: Any,
        campaign: CandidateCampaign,
        token_ids: np.ndarray | Sequence[int],
    ) -> None:
        values = _token_array(token_ids)
        if not campaign.input_binding.matches(values):
            raise ValueError("runtime token IDs do not match the compiled campaign input binding")
        runtime_configuration = validate_candidate_campaign_engine_binding(engine, campaign)
        bundle = campaign.base_bundle
        graph = bundle.graph
        if graph is None or graph.rewrite_certificate is None:
            raise RuntimeError("candidate campaign has no certified compiled graph")
        selected = getattr(engine, "selected_last_logits_batch", None)
        if not callable(selected):
            raise RuntimeError("engine cannot execute selected vocabulary rows")
        bound_values = values.copy()
        bound_values.setflags(write=False)
        capture_executor: Any | None = None
        capture_metadata: Mapping[str, Any] | None = None
        if bundle.plan.capture.requested:
            if not bundle.lowered.capture_ready:
                raise RuntimeError("campaign requested CUDA Graph capture but is not capture-ready")
            prepare_capture = getattr(engine, "prepare_selected_last_cuda_graph", None)
            if not callable(prepare_capture):
                raise RuntimeError("engine cannot prepare the requested CUDA Graph executor")
            capture_executor = prepare_capture(
                [bound_values],
                campaign.union_token_ids,
            )
            if not callable(getattr(capture_executor, "execute", None)):
                close = getattr(capture_executor, "close", None)
                if callable(close):
                    close()
                raise RuntimeError("CUDA Graph executor does not expose execute()")
            raw_capture_metadata = getattr(capture_executor, "evidence", None)
            if not isinstance(raw_capture_metadata, Mapping):
                close = getattr(capture_executor, "close", None)
                if callable(close):
                    close()
                raise RuntimeError("CUDA Graph executor does not expose evidence")
            capture_metadata = dict(raw_capture_metadata)
        work_floor = bundle.work_floor
        evidence = {
            "implementation_status": bundle.lowered.implementation_status,
            "runtime_implementation_status": (
                "cuda-graph" if capture_executor is not None else "eager-adapter"
            ),
            "engine_backend": str(getattr(engine, "backend", "unknown")),
            "reported_fabric": runtime_configuration["runtime_fabric"],
            "graph_replay": capture_executor is not None,
            "capture_requested": bundle.plan.capture.requested,
            "capture_ready": bundle.lowered.capture_ready,
            "capture_executed": capture_executor is not None,
            "capture_metadata": capture_metadata,
            "placement_verified": bundle.lowered.placement_verified,
            "actual_batch": 1,
            "sequence_length": bundle.plan.shape.sequence_length,
            "numerical_contract": bundle.plan.numerical_contract,
            "output_pushdown": True,
            "graph_compilation_fingerprint": graph.fingerprint,
            "graph_rewrite_ids": graph.rewrite_certificate.rewrite_ids,
            "graph_dispatch_verified": True,
            "dispatch_prepared": True,
            "body_execution_count": 1,
            "independent_equivalent_body_execution_count": len(campaign.readouts),
            "candidate_reference_count": campaign.sharing.candidate_reference_count,
            "union_candidate_count": campaign.sharing.union_candidate_count,
            "eliminated_body_evaluations": (campaign.sharing.eliminated_body_evaluations),
            "eliminated_head_score_evaluations": (
                campaign.sharing.eliminated_head_score_evaluations
            ),
            "campaign_sharing_schema": campaign.sharing.schema_version,
            "compilation_bundle_fingerprint": bundle.fingerprint,
            "work_floor_fingerprint": (
                None if work_floor is None else work_floor.rewritten.certificate.fingerprint
            ),
            "content_identity_verified": bundle.plan.content_identity_verified,
            "runtime_identity_bound": True,
            "runtime_configuration_bound": True,
            **runtime_configuration,
            "input_token_sha256": campaign.input_binding.token_sha256,
            "query_reduction_semantics": "local-candidate-order-then-torch-topk-2",
        }
        self._campaign = campaign
        self._capture_executor = capture_executor
        self._closed = False
        self._token_ids = bound_values
        self._selected: Callable[..., Any] = selected
        self._evidence = MappingProxyType(evidence)

    @property
    def campaign(self) -> CandidateCampaign:
        return self._campaign

    @property
    def input_token_sha256(self) -> str:
        return self._campaign.input_binding.token_sha256

    def execute(self) -> CandidateCampaignExecutionResult:
        if self._closed:
            raise RuntimeError("prepared candidate campaign is closed")
        if self._capture_executor is None:
            raw_scores = self._selected(
                [self._token_ids],
                self._campaign.union_token_ids,
            )
            evidence: Mapping[str, Any] = self._evidence
        else:
            raw_scores = self._capture_executor.execute()
            raw_capture_metadata = getattr(self._capture_executor, "evidence", None)
            if not isinstance(raw_capture_metadata, Mapping):
                raise RuntimeError("CUDA Graph executor lost its runtime evidence")
            evidence = {
                **self._evidence,
                "capture_metadata": dict(raw_capture_metadata),
            }
        scores = torch.as_tensor(raw_scores)
        return _campaign_result_from_union_scores(
            self._campaign,
            scores,
            evidence,
        )

    def close(self) -> None:
        if self._closed:
            return
        executor = self._capture_executor
        self._capture_executor = None
        self._closed = True
        close = getattr(executor, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> PreparedCandidateCampaign:
        if self._closed:
            raise RuntimeError("prepared candidate campaign is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def prepare_candidate_campaign(
    engine: Any,
    campaign: CandidateCampaign,
    token_ids: np.ndarray | Sequence[int],
) -> PreparedCandidateCampaign:
    """Validate and bind a campaign once for repeated hot dispatch."""

    return PreparedCandidateCampaign(engine, campaign, token_ids)


def execute_candidate_campaign(
    engine: Any,
    campaign: CandidateCampaign,
    token_ids: np.ndarray | Sequence[int],
) -> CandidateCampaignExecutionResult:
    """Prepare and execute one compiled campaign.

    Repeated callers should use :func:`prepare_candidate_campaign` once and replay
    :meth:`PreparedCandidateCampaign.execute` so certificate/identity checks remain
    outside the measured hot path.
    """

    with prepare_candidate_campaign(engine, campaign, token_ids) as prepared:
        return prepared.execute()
