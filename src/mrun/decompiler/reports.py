"""Canonical decompilation coverage and support reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._json import (
    canonical_json,
    canonical_sha256,
    require_bool,
    require_dict,
    require_exact_keys,
    require_int,
    require_list,
    require_sha256,
    require_str,
    strict_json_loads,
)
from .errors import DecompilerError
from .ir import IRBundle, PhysicalWeightIR
from .matching import MatchResult
from .source import FrozenSourceBundle
from .tensor_index import TensorIndex

COVERAGE_REPORT_SCHEMA = "mrun-source-coverage-report-v1"
DECOMPILE_REPORT_SCHEMA = "mrun-decompile-report-v1"
DECOMPILE_RESULT_SCHEMA = "mrun-decompile-result-v1"


@dataclass(frozen=True, slots=True)
class FailureRecord:
    code: str
    gate: str
    message: str
    _details_json: str

    def __post_init__(self) -> None:
        for field_name in ("code", "gate", "message"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        details = strict_json_loads(self._details_json, field="failure details")
        if not isinstance(details, dict):
            raise TypeError("failure details must be a JSON object")
        object.__setattr__(self, "_details_json", canonical_json(details))

    @property
    def details(self) -> dict[str, Any]:
        value = strict_json_loads(self._details_json, field="failure details")
        assert isinstance(value, dict)
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "gate": self.gate,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_error(cls, error: DecompilerError) -> FailureRecord:
        return cls(
            code=error.code,
            gate=error.gate,
            message=error.message,
            _details_json=canonical_json(error.details),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> FailureRecord:
        value = require_dict(payload, field="failure record")
        require_exact_keys(value, {"code", "gate", "message", "details"}, field="failure record")
        return cls(
            code=require_str(value["code"], field="failure code"),
            gate=require_str(value["gate"], field="failure gate"),
            message=require_str(value["message"], field="failure message"),
            _details_json=canonical_json(require_dict(value["details"], field="failure details")),
        )


@dataclass(frozen=True, slots=True)
class CoverageReport:
    source_tensor_count: int
    source_tensor_bytes: int
    classified_tensor_count: int
    classified_tensor_bytes: int
    parameter_tensors: tuple[str, ...]
    buffer_tensors: tuple[str, ...]
    codec_metadata_tensors: tuple[str, ...]
    ignored_tensors: tuple[str, ...]
    unexplained_tensors: tuple[str, ...]
    orphan_logical_views: tuple[str, ...]
    complete: bool
    fingerprint: str
    schema_version: str = COVERAGE_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COVERAGE_REPORT_SCHEMA:
            raise ValueError(f"unsupported coverage schema: {self.schema_version!r}")
        for field_name in (
            "source_tensor_count",
            "source_tensor_bytes",
            "classified_tensor_count",
            "classified_tensor_bytes",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in (
            "parameter_tensors",
            "buffer_tensors",
            "codec_metadata_tensors",
            "ignored_tensors",
            "unexplained_tensors",
            "orphan_logical_views",
        ):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))) or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"{field_name} must contain sorted unique strings")
            object.__setattr__(self, field_name, values)
        if type(self.complete) is not bool:
            raise TypeError("coverage complete must be a boolean")
        expected_complete = (
            self.source_tensor_count == self.classified_tensor_count
            and self.source_tensor_bytes == self.classified_tensor_bytes
            and not self.unexplained_tensors
            and not self.orphan_logical_views
        )
        if self.complete != expected_complete:
            raise ValueError("coverage complete flag does not match the coverage counts")
        object.__setattr__(
            self,
            "fingerprint",
            require_sha256(self.fingerprint, field="coverage fingerprint"),
        )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("coverage fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_tensor_count": self.source_tensor_count,
            "source_tensor_bytes": self.source_tensor_bytes,
            "classified_tensor_count": self.classified_tensor_count,
            "classified_tensor_bytes": self.classified_tensor_bytes,
            "parameter_tensors": list(self.parameter_tensors),
            "buffer_tensors": list(self.buffer_tensors),
            "codec_metadata_tensors": list(self.codec_metadata_tensors),
            "ignored_tensors": list(self.ignored_tensors),
            "unexplained_tensors": list(self.unexplained_tensors),
            "orphan_logical_views": list(self.orphan_logical_views),
            "complete": self.complete,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, index: TensorIndex, weights: PhysicalWeightIR) -> CoverageReport:
        source = index.by_name()
        classifications = {item.source_name: item for item in weights.classifications}
        classified_names = set(source) & set(classifications)
        unexplained = tuple(sorted(set(source) - set(classifications)))
        disposition_names: dict[str, tuple[str, ...]] = {}
        for disposition in ("parameter", "buffer", "codec-metadata", "ignored"):
            disposition_names[disposition] = tuple(
                sorted(
                    name
                    for name, classification in classifications.items()
                    if classification.disposition == disposition and name in source
                )
            )
        classified_views = {
            view_id
            for classification in classifications.values()
            for view_id in classification.logical_view_ids
        }
        orphan_views = tuple(
            sorted(view.view_id for view in weights.views if view.view_id not in classified_views)
        )
        payload = {
            "schema_version": COVERAGE_REPORT_SCHEMA,
            "source_tensor_count": len(source),
            "source_tensor_bytes": index.total_tensor_bytes,
            "classified_tensor_count": len(classified_names),
            "classified_tensor_bytes": sum(source[name].byte_length for name in classified_names),
            "parameter_tensors": list(disposition_names["parameter"]),
            "buffer_tensors": list(disposition_names["buffer"]),
            "codec_metadata_tensors": list(disposition_names["codec-metadata"]),
            "ignored_tensors": list(disposition_names["ignored"]),
            "unexplained_tensors": list(unexplained),
            "orphan_logical_views": list(orphan_views),
            "complete": (
                len(classified_names) == len(source)
                and sum(source[name].byte_length for name in classified_names)
                == index.total_tensor_bytes
                and not unexplained
                and not orphan_views
            ),
        }
        return cls(
            source_tensor_count=payload["source_tensor_count"],
            source_tensor_bytes=payload["source_tensor_bytes"],
            classified_tensor_count=payload["classified_tensor_count"],
            classified_tensor_bytes=payload["classified_tensor_bytes"],
            parameter_tensors=disposition_names["parameter"],
            buffer_tensors=disposition_names["buffer"],
            codec_metadata_tensors=disposition_names["codec-metadata"],
            ignored_tensors=disposition_names["ignored"],
            unexplained_tensors=unexplained,
            orphan_logical_views=orphan_views,
            complete=payload["complete"],
            fingerprint=canonical_sha256(payload),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> CoverageReport:
        value = require_dict(payload, field="coverage report")
        expected = {
            "schema_version",
            "source_tensor_count",
            "source_tensor_bytes",
            "classified_tensor_count",
            "classified_tensor_bytes",
            "parameter_tensors",
            "buffer_tensors",
            "codec_metadata_tensors",
            "ignored_tensors",
            "unexplained_tensors",
            "orphan_logical_views",
            "complete",
            "fingerprint",
        }
        require_exact_keys(value, expected, field="coverage report")
        string_fields = {
            field_name: tuple(
                require_str(item, field=f"{field_name}[]")
                for item in require_list(value[field_name], field=field_name)
            )
            for field_name in (
                "parameter_tensors",
                "buffer_tensors",
                "codec_metadata_tensors",
                "ignored_tensors",
                "unexplained_tensors",
                "orphan_logical_views",
            )
        }
        return cls(
            schema_version=require_str(value["schema_version"], field="coverage schema"),
            source_tensor_count=require_int(
                value["source_tensor_count"], field="source_tensor_count", minimum=0
            ),
            source_tensor_bytes=require_int(
                value["source_tensor_bytes"], field="source_tensor_bytes", minimum=0
            ),
            classified_tensor_count=require_int(
                value["classified_tensor_count"], field="classified_tensor_count", minimum=0
            ),
            classified_tensor_bytes=require_int(
                value["classified_tensor_bytes"], field="classified_tensor_bytes", minimum=0
            ),
            complete=require_bool(value["complete"], field="complete"),
            fingerprint=require_str(value["fingerprint"], field="coverage fingerprint"),
            **string_fields,
        )


@dataclass(frozen=True, slots=True)
class DecompileReport:
    status: str
    universal_level: str
    source_fingerprint: str | None
    tensor_index_fingerprint: str | None
    source_asset_count: int
    source_asset_bytes: int
    selected_adapter_id: str | None
    selected_adapter_version: str | None
    selected_adapter_fingerprint: str | None
    match_results: tuple[MatchResult, ...]
    coverage: CoverageReport | None
    ir_fingerprints: tuple[tuple[str, str], ...]
    pending_gates: tuple[str, ...]
    failures: tuple[FailureRecord, ...]
    fingerprint: str
    schema_version: str = DECOMPILE_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DECOMPILE_REPORT_SCHEMA:
            raise ValueError(f"unsupported decompile-report schema: {self.schema_version!r}")
        if self.status not in {"decoded", "unsupported", "rejected"}:
            raise ValueError(f"invalid decompile status: {self.status!r}")
        if self.universal_level not in {"U_NONE", "U0", "U1", "U2"}:
            raise ValueError(f"invalid universal level: {self.universal_level!r}")
        for field_name in ("source_fingerprint", "tensor_index_fingerprint"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_sha256(value, field=field_name))
        for field_name in ("source_asset_count", "source_asset_bytes"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        selected = (
            self.selected_adapter_id,
            self.selected_adapter_version,
            self.selected_adapter_fingerprint,
        )
        partly_missing = any(value is None for value in selected) and any(
            value is not None for value in selected
        )
        if partly_missing:
            raise ValueError("selected adapter identity must be entirely present or absent")
        if self.selected_adapter_id is not None:
            object.__setattr__(
                self,
                "selected_adapter_id",
                require_str(self.selected_adapter_id, field="selected_adapter_id"),
            )
            object.__setattr__(
                self,
                "selected_adapter_version",
                require_str(self.selected_adapter_version, field="selected_adapter_version"),
            )
            object.__setattr__(
                self,
                "selected_adapter_fingerprint",
                require_sha256(
                    self.selected_adapter_fingerprint,
                    field="selected_adapter_fingerprint",
                ),
            )
        matches = tuple(self.match_results)
        identities = [(item.adapter_id, item.adapter_version) for item in matches]
        if identities != sorted(set(identities)):
            raise ValueError("match results must be sorted with unique adapter identities")
        object.__setattr__(self, "match_results", matches)
        fingerprints = tuple(self.ir_fingerprints)
        if fingerprints != tuple(sorted(fingerprints)):
            raise ValueError("IR fingerprints must be sorted by IR name")
        if len({name for name, _ in fingerprints}) != len(fingerprints):
            raise ValueError("IR fingerprint names must be unique")
        for name, digest in fingerprints:
            require_str(name, field="IR fingerprint name")
            require_sha256(digest, field=f"{name} fingerprint")
        object.__setattr__(self, "ir_fingerprints", fingerprints)
        for field_name in ("pending_gates",):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must contain sorted unique strings")
            object.__setattr__(self, field_name, values)
        if self.status == "decoded":
            if self.universal_level != "U2" or self.failures:
                raise ValueError("decoded report must attain U2 without failures")
            if self.coverage is None or not self.coverage.complete:
                raise ValueError("decoded report requires complete source coverage")
            if self.selected_adapter_id is None or len(fingerprints) != 5:
                raise ValueError("decoded report requires adapter and all IR fingerprints")
        elif not self.failures:
            raise ValueError("unsupported/rejected report requires a failure record")
        object.__setattr__(
            self,
            "fingerprint",
            require_sha256(self.fingerprint, field="report fingerprint"),
        )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("decompile-report fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "universal_level": self.universal_level,
            "source_fingerprint": self.source_fingerprint,
            "tensor_index_fingerprint": self.tensor_index_fingerprint,
            "source_asset_count": self.source_asset_count,
            "source_asset_bytes": self.source_asset_bytes,
            "selected_adapter_id": self.selected_adapter_id,
            "selected_adapter_version": self.selected_adapter_version,
            "selected_adapter_fingerprint": self.selected_adapter_fingerprint,
            "match_results": [item.as_dict() for item in self.match_results],
            "coverage": self.coverage.as_dict() if self.coverage is not None else None,
            "ir_fingerprints": [
                {"name": name, "sha256": digest} for name, digest in self.ir_fingerprints
            ],
            "pending_gates": list(self.pending_gates),
            "failures": [item.as_dict() for item in self.failures],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **kwargs: Any) -> DecompileReport:
        payload = {
            "schema_version": DECOMPILE_REPORT_SCHEMA,
            "status": kwargs["status"],
            "universal_level": kwargs["universal_level"],
            "source_fingerprint": kwargs["source_fingerprint"],
            "tensor_index_fingerprint": kwargs["tensor_index_fingerprint"],
            "source_asset_count": kwargs["source_asset_count"],
            "source_asset_bytes": kwargs["source_asset_bytes"],
            "selected_adapter_id": kwargs["selected_adapter_id"],
            "selected_adapter_version": kwargs["selected_adapter_version"],
            "selected_adapter_fingerprint": kwargs["selected_adapter_fingerprint"],
            "match_results": [item.as_dict() for item in kwargs["match_results"]],
            "coverage": (kwargs["coverage"].as_dict() if kwargs["coverage"] is not None else None),
            "ir_fingerprints": [
                {"name": name, "sha256": digest} for name, digest in kwargs["ir_fingerprints"]
            ],
            "pending_gates": list(kwargs["pending_gates"]),
            "failures": [item.as_dict() for item in kwargs["failures"]],
        }
        return cls(**kwargs, fingerprint=canonical_sha256(payload))

    @classmethod
    def from_dict(cls, payload: Any) -> DecompileReport:
        value = require_dict(payload, field="decompile report")
        expected = {
            "schema_version",
            "status",
            "universal_level",
            "source_fingerprint",
            "tensor_index_fingerprint",
            "source_asset_count",
            "source_asset_bytes",
            "selected_adapter_id",
            "selected_adapter_version",
            "selected_adapter_fingerprint",
            "match_results",
            "coverage",
            "ir_fingerprints",
            "pending_gates",
            "failures",
            "fingerprint",
        }
        require_exact_keys(value, expected, field="decompile report")
        nullable_strings: dict[str, str | None] = {}
        for field_name in (
            "source_fingerprint",
            "tensor_index_fingerprint",
            "selected_adapter_id",
            "selected_adapter_version",
            "selected_adapter_fingerprint",
        ):
            raw = value[field_name]
            nullable_strings[field_name] = (
                None if raw is None else require_str(raw, field=field_name)
            )
        raw_coverage = value["coverage"]
        raw_ir = require_list(value["ir_fingerprints"], field="IR fingerprints")
        fingerprints: list[tuple[str, str]] = []
        for item in raw_ir:
            record = require_dict(item, field="IR fingerprint")
            require_exact_keys(record, {"name", "sha256"}, field="IR fingerprint")
            fingerprints.append(
                (
                    require_str(record["name"], field="IR fingerprint name"),
                    require_str(record["sha256"], field="IR fingerprint sha256"),
                )
            )
        return cls(
            schema_version=require_str(value["schema_version"], field="report schema"),
            status=require_str(value["status"], field="status"),
            universal_level=require_str(value["universal_level"], field="universal_level"),
            source_asset_count=require_int(
                value["source_asset_count"], field="source_asset_count", minimum=0
            ),
            source_asset_bytes=require_int(
                value["source_asset_bytes"], field="source_asset_bytes", minimum=0
            ),
            match_results=tuple(
                MatchResult.from_dict(item)
                for item in require_list(value["match_results"], field="match_results")
            ),
            coverage=(None if raw_coverage is None else CoverageReport.from_dict(raw_coverage)),
            ir_fingerprints=tuple(fingerprints),
            pending_gates=tuple(
                require_str(item, field="pending gate")
                for item in require_list(value["pending_gates"], field="pending_gates")
            ),
            failures=tuple(
                FailureRecord.from_dict(item)
                for item in require_list(value["failures"], field="failures")
            ),
            fingerprint=require_str(value["fingerprint"], field="report fingerprint"),
            **nullable_strings,
        )


@dataclass(frozen=True, slots=True)
class DecompileResult:
    report: DecompileReport
    source: FrozenSourceBundle | None
    tensor_index: TensorIndex | None
    ir_bundle: IRBundle | None
    schema_version: str = DECOMPILE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DECOMPILE_RESULT_SCHEMA:
            raise ValueError(f"unsupported decompile-result schema: {self.schema_version!r}")
        if not isinstance(self.report, DecompileReport):
            raise TypeError("report must be a DecompileReport")
        if self.report.status == "decoded":
            if self.source is None or self.tensor_index is None or self.ir_bundle is None:
                raise ValueError("decoded result requires source, tensor index, and IR bundle")
        elif self.ir_bundle is not None:
            raise ValueError("failed result cannot expose an executable IR bundle")
        if self.source is not None:
            if self.report.source_fingerprint != self.source.fingerprint:
                raise ValueError("result report does not bind its frozen source")
            if self.report.source_asset_count != len(self.source.files) or (
                self.report.source_asset_bytes
                != sum(record.byte_count for record in self.source.files)
            ):
                raise ValueError("result report source asset totals are inconsistent")
        elif self.report.source_fingerprint is not None:
            raise ValueError("result report binds a source that is not present")
        if self.tensor_index is not None:
            if self.source is None:
                raise ValueError("tensor index cannot exist without its frozen source")
            if self.tensor_index.source_fingerprint != self.source.fingerprint:
                raise ValueError("tensor index does not bind the result's frozen source")
            if self.tensor_index.config_features != self.source.config:
                raise ValueError("tensor index config does not match the frozen source")
            if self.report.tensor_index_fingerprint != self.tensor_index.fingerprint:
                raise ValueError("result report does not bind its tensor index")
        elif self.report.tensor_index_fingerprint is not None:
            raise ValueError("result report binds a tensor index that is not present")
        if self.ir_bundle is not None:
            assert self.source is not None and self.tensor_index is not None
            weights = self.ir_bundle.physical_weights
            if weights.tensor_index_fingerprint != self.tensor_index.fingerprint:
                raise ValueError("PhysicalWeightIR does not bind the result's tensor index")
            indexed = self.tensor_index.by_name()
            allocations = {item.source_tensor: item for item in weights.allocations}
            if set(indexed) != set(allocations):
                raise ValueError("PhysicalWeightIR allocations do not cover the tensor index")
            for source_name, record in indexed.items():
                allocation = allocations[source_name]
                if (
                    allocation.source_file != record.source_file
                    or allocation.byte_offset != record.byte_offset
                    or allocation.byte_length != record.byte_length
                    or allocation.stored_shape != record.shape
                    or allocation.stored_dtype != record.storage_dtype
                    or allocation.content_fingerprint != record.range_identity
                ):
                    raise ValueError(
                        f"physical allocation does not match tensor index: {source_name}"
                    )
            expected_coverage = CoverageReport.build(self.tensor_index, weights)
            if self.report.coverage != expected_coverage:
                raise ValueError("result report coverage does not match the coordinated IR")
            expected_ir = tuple(
                sorted(
                    (
                        ("bundle", self.ir_bundle.fingerprint),
                        ("io", self.ir_bundle.io.fingerprint),
                        ("model", self.ir_bundle.model.fingerprint),
                        ("physical_weights", weights.fingerprint),
                        ("state", self.ir_bundle.state.fingerprint),
                    )
                )
            )
            if self.report.ir_fingerprints != expected_ir:
                raise ValueError("result report does not bind every coordinated IR")
            selected = (
                self.report.selected_adapter_id,
                self.report.selected_adapter_version,
                self.report.selected_adapter_fingerprint,
            )
            model_adapter = (
                self.ir_bundle.model.adapter_id,
                self.ir_bundle.model.adapter_version,
                self.ir_bundle.model.adapter_fingerprint,
            )
            if selected != model_adapter:
                raise ValueError("result report selected adapter does not match ModelIR")
            source_assets = {record.path: record for record in self.source.files}
            for asset in (
                *self.ir_bundle.io.tokenizer_assets,
                *self.ir_bundle.io.processor_assets,
            ):
                record = source_assets.get(asset.path)
                if record is None or (
                    asset.sha256 != record.sha256
                    or asset.byte_count != record.byte_count
                    or asset.role != record.role
                ):
                    raise ValueError("IOIR contains a foreign or stale bound asset")

    @property
    def succeeded(self) -> bool:
        return self.report.status == "decoded" and self.ir_bundle is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report": self.report.as_dict(),
            "source": self.source.as_dict() if self.source is not None else None,
            "tensor_index": self.tensor_index.as_dict() if self.tensor_index is not None else None,
            "ir_bundle": self.ir_bundle.as_dict() if self.ir_bundle is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DecompileResult:
        value = require_dict(payload, field="decompile result")
        require_exact_keys(
            value,
            {"schema_version", "report", "source", "tensor_index", "ir_bundle"},
            field="decompile result",
        )
        return cls(
            schema_version=require_str(value["schema_version"], field="result schema"),
            report=DecompileReport.from_dict(value["report"]),
            source=(
                None if value["source"] is None else FrozenSourceBundle.from_dict(value["source"])
            ),
            tensor_index=(
                None
                if value["tensor_index"] is None
                else TensorIndex.from_dict(value["tensor_index"])
            ),
            ir_bundle=(
                None if value["ir_bundle"] is None else IRBundle.from_dict(value["ir_bundle"])
            ),
        )
