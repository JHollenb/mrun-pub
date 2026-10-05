"""Deterministic architecture-adapter matching and ambiguity rejection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._json import (
    canonical_sha256,
    require_bool,
    require_dict,
    require_exact_keys,
    require_int,
    require_list,
    require_sha256,
    require_str,
)
from .errors import AmbiguousAdapterError, NoAdapterError, UnsupportedVariantError
from .ir import IRBundle
from .source import FrozenSourceBundle
from .tensor_index import TensorIndex

MATCH_RESULT_SCHEMA = "mrun-architecture-match-v1"


@dataclass(frozen=True, slots=True)
class MatchEvidence:
    predicate: str
    expected: str
    observed: str
    matched: bool

    def __post_init__(self) -> None:
        for field_name in ("predicate", "expected", "observed"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if type(self.matched) is not bool:
            raise TypeError("match evidence matched must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "expected": self.expected,
            "observed": self.observed,
            "matched": self.matched,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> MatchEvidence:
        value = require_dict(payload, field="match evidence")
        require_exact_keys(
            value, {"predicate", "expected", "observed", "matched"}, field="match evidence"
        )
        return cls(
            predicate=require_str(value["predicate"], field="predicate"),
            expected=require_str(value["expected"], field="expected"),
            observed=require_str(value["observed"], field="observed"),
            matched=require_bool(value["matched"], field="matched"),
        )


@dataclass(frozen=True, slots=True)
class UnsupportedFeature:
    code: str
    field: str
    observed: str
    reason: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )

    def as_dict(self) -> dict[str, str]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Any) -> UnsupportedFeature:
        value = require_dict(payload, field="unsupported feature")
        expected = set(cls.__dataclass_fields__)
        require_exact_keys(value, expected, field="unsupported feature")
        return cls(**{name: require_str(value[name], field=name) for name in expected})


@dataclass(frozen=True, slots=True)
class MatchResult:
    adapter_id: str
    adapter_version: str
    adapter_fingerprint: str
    matched: bool
    supported: bool
    strength: int
    evidence: tuple[MatchEvidence, ...]
    rejected_reasons: tuple[str, ...]
    required_tensor_patterns: tuple[str, ...]
    forbidden_tensor_patterns: tuple[str, ...]
    source_codec_candidates: tuple[str, ...]
    unsupported_features: tuple[UnsupportedFeature, ...]
    fingerprint: str
    schema_version: str = MATCH_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MATCH_RESULT_SCHEMA:
            raise ValueError(f"unsupported match-result schema: {self.schema_version!r}")
        for field_name in ("adapter_id", "adapter_version"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        for field_name in ("adapter_fingerprint", "fingerprint"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if type(self.matched) is not bool or type(self.supported) is not bool:
            raise TypeError("matched and supported must be booleans")
        if self.supported and not self.matched:
            raise ValueError("an unmatched adapter cannot be supported")
        if type(self.strength) is not int or not 0 <= self.strength <= 100:
            raise ValueError("match strength must be an integer from 0 through 100")
        if not self.matched and self.strength != 0:
            raise ValueError("an unmatched adapter must have zero strength")
        evidence = tuple(self.evidence)
        if evidence != tuple(sorted(evidence, key=lambda item: item.predicate)):
            raise ValueError("match evidence must be sorted by predicate")
        if len({item.predicate for item in evidence}) != len(evidence):
            raise ValueError("match evidence predicates must be unique")
        object.__setattr__(self, "evidence", evidence)
        for field_name in (
            "rejected_reasons",
            "required_tensor_patterns",
            "forbidden_tensor_patterns",
            "source_codec_candidates",
        ):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))) or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"{field_name} must contain sorted unique strings")
            object.__setattr__(self, field_name, values)
        unsupported = tuple(self.unsupported_features)
        if unsupported != tuple(sorted(unsupported, key=lambda item: (item.code, item.field))):
            raise ValueError("unsupported features must be sorted by code and field")
        if len({(item.code, item.field) for item in unsupported}) != len(unsupported):
            raise ValueError("unsupported features must have unique code/field identities")
        if not self.matched and unsupported:
            raise ValueError(
                "an unmatched adapter cannot claim variant-specific unsupported features"
            )
        object.__setattr__(self, "unsupported_features", unsupported)
        if self.matched and self.supported == bool(unsupported):
            raise ValueError("supported must be true exactly when unsupported_features is empty")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("match-result fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "adapter_fingerprint": self.adapter_fingerprint,
            "matched": self.matched,
            "supported": self.supported,
            "strength": self.strength,
            "evidence": [item.as_dict() for item in self.evidence],
            "rejected_reasons": list(self.rejected_reasons),
            "required_tensor_patterns": list(self.required_tensor_patterns),
            "forbidden_tensor_patterns": list(self.forbidden_tensor_patterns),
            "source_codec_candidates": list(self.source_codec_candidates),
            "unsupported_features": [item.as_dict() for item in self.unsupported_features],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **kwargs: Any) -> MatchResult:
        payload = {
            "schema_version": MATCH_RESULT_SCHEMA,
            "adapter_id": kwargs["adapter_id"],
            "adapter_version": kwargs["adapter_version"],
            "adapter_fingerprint": kwargs["adapter_fingerprint"],
            "matched": kwargs["matched"],
            "supported": kwargs["supported"],
            "strength": kwargs["strength"],
            "evidence": [item.as_dict() for item in kwargs["evidence"]],
            "rejected_reasons": list(kwargs["rejected_reasons"]),
            "required_tensor_patterns": list(kwargs["required_tensor_patterns"]),
            "forbidden_tensor_patterns": list(kwargs["forbidden_tensor_patterns"]),
            "source_codec_candidates": list(kwargs["source_codec_candidates"]),
            "unsupported_features": [item.as_dict() for item in kwargs["unsupported_features"]],
        }
        return cls(**kwargs, fingerprint=canonical_sha256(payload))

    @classmethod
    def from_dict(cls, payload: Any) -> MatchResult:
        value = require_dict(payload, field="adapter match")
        require_exact_keys(
            value,
            {
                "schema_version",
                "adapter_id",
                "adapter_version",
                "adapter_fingerprint",
                "matched",
                "supported",
                "strength",
                "evidence",
                "rejected_reasons",
                "required_tensor_patterns",
                "forbidden_tensor_patterns",
                "source_codec_candidates",
                "unsupported_features",
                "fingerprint",
            },
            field="adapter match",
        )
        strings = (
            "rejected_reasons",
            "required_tensor_patterns",
            "forbidden_tensor_patterns",
            "source_codec_candidates",
        )
        parsed_strings: dict[str, tuple[str, ...]] = {}
        for field_name in strings:
            parsed_strings[field_name] = tuple(
                require_str(item, field=f"{field_name}[]")
                for item in require_list(value[field_name], field=field_name)
            )
        return cls(
            schema_version=require_str(value["schema_version"], field="match schema"),
            adapter_id=require_str(value["adapter_id"], field="adapter_id"),
            adapter_version=require_str(value["adapter_version"], field="adapter_version"),
            adapter_fingerprint=require_str(
                value["adapter_fingerprint"], field="adapter_fingerprint"
            ),
            matched=require_bool(value["matched"], field="matched"),
            supported=require_bool(value["supported"], field="supported"),
            strength=require_int(value["strength"], field="strength", minimum=0),
            evidence=tuple(
                MatchEvidence.from_dict(item)
                for item in require_list(value["evidence"], field="evidence")
            ),
            unsupported_features=tuple(
                UnsupportedFeature.from_dict(item)
                for item in require_list(
                    value["unsupported_features"], field="unsupported_features"
                )
            ),
            fingerprint=require_str(value["fingerprint"], field="match fingerprint"),
            **parsed_strings,
        )


@runtime_checkable
class ArchitectureAdapter(Protocol):
    adapter_id: str
    adapter_version: str
    adapter_fingerprint: str

    def match(self, source: FrozenSourceBundle, index: TensorIndex) -> MatchResult: ...

    def compile_ir(self, source: FrozenSourceBundle, index: TensorIndex) -> IRBundle: ...


@dataclass(frozen=True, slots=True)
class AdapterSelection:
    adapter: ArchitectureAdapter
    selected_match: MatchResult
    all_matches: tuple[MatchResult, ...]


class AdapterRegistry:
    """Order-independent registry; ties and recognized unsupported variants fail closed."""

    def __init__(
        self,
        adapters: tuple[ArchitectureAdapter, ...] | list[ArchitectureAdapter] = (),
        *,
        minimum_strength: int = 80,
    ) -> None:
        if type(minimum_strength) is not int or not 1 <= minimum_strength <= 100:
            raise ValueError("minimum_strength must be an integer from 1 through 100")
        normalized = tuple(adapters)
        identities: set[tuple[str, str]] = set()
        for adapter in normalized:
            required = (
                "adapter_id",
                "adapter_version",
                "adapter_fingerprint",
                "match",
                "compile_ir",
            )
            if any(not hasattr(adapter, field_name) for field_name in required):
                raise TypeError("registry entries must implement ArchitectureAdapter")
            if not callable(adapter.match) or not callable(adapter.compile_ir):
                raise TypeError("adapter match and compile_ir attributes must be callable")
            identity = (adapter.adapter_id, adapter.adapter_version)
            if identity in identities:
                raise ValueError(f"duplicate adapter identity: {identity!r}")
            identities.add(identity)
        self._adapters = tuple(
            sorted(normalized, key=lambda item: (item.adapter_id, item.adapter_version))
        )
        self.minimum_strength = minimum_strength

    @property
    def adapters(self) -> tuple[ArchitectureAdapter, ...]:
        return self._adapters

    def match_all(self, source: FrozenSourceBundle, index: TensorIndex) -> tuple[MatchResult, ...]:
        results = tuple(adapter.match(source, index) for adapter in self._adapters)
        for adapter, result in zip(self._adapters, results, strict=True):
            if (
                result.adapter_id != adapter.adapter_id
                or result.adapter_version != adapter.adapter_version
                or result.adapter_fingerprint != adapter.adapter_fingerprint
            ):
                raise RuntimeError(
                    f"adapter {adapter.adapter_id!r} returned a foreign match result"
                )
        return results

    def select(self, source: FrozenSourceBundle, index: TensorIndex) -> AdapterSelection:
        matches = self.match_all(source, index)
        candidates = [
            (adapter, result)
            for adapter, result in zip(self._adapters, matches, strict=True)
            if result.matched and result.strength >= self.minimum_strength
        ]
        if not candidates:
            raise NoAdapterError(
                "no installed architecture adapter matched the frozen source",
                details={"matches": [result.as_dict() for result in matches]},
            )
        maximum = max(result.strength for _, result in candidates)
        winners = [
            (adapter, result) for adapter, result in candidates if result.strength == maximum
        ]
        if len(winners) != 1:
            raise AmbiguousAdapterError(
                "multiple architecture adapters have the same winning strength",
                details={
                    "strength": maximum,
                    "adapters": [result.adapter_id for _, result in winners],
                    "matches": [result.as_dict() for result in matches],
                },
            )
        adapter, result = winners[0]
        if not result.supported:
            raise UnsupportedVariantError(
                f"adapter {adapter.adapter_id!r} recognized an unsupported variant",
                details={
                    "selected_match": result.as_dict(),
                    "unsupported_features": [
                        feature.as_dict() for feature in result.unsupported_features
                    ],
                },
            )
        return AdapterSelection(adapter=adapter, selected_match=result, all_matches=matches)
