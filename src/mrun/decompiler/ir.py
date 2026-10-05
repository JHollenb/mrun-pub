"""Versioned, canonical semantic/physical/state/IO intermediate representations.

The records in this module contain JSON-native data only.  Backend tensors, Python callables,
opaque transforms, and host paths are deliberately impossible to encode here.
"""

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
from .errors import IRValidationError

PHYSICAL_WEIGHT_IR_SCHEMA = "mrun-physical-weight-ir-v1"
MODEL_IR_SCHEMA = "mrun-model-ir-v1"
STATE_IR_SCHEMA = "mrun-state-ir-v1"
IO_IR_SCHEMA = "mrun-io-ir-v1"
IR_BUNDLE_SCHEMA = "mrun-decoded-ir-bundle-v1"


def _strings(values: Any, *, field: str, sorted_unique: bool = False) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field} must be an array")
    result = tuple(require_str(value, field=f"{field}[]") for value in values)
    if sorted_unique and result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must contain sorted unique strings")
    return result


def _shape(values: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field} must be an array")
    return tuple(require_int(value, field=f"{field}[]", minimum=0) for value in values)


def _json_object(raw: str, *, field: str) -> dict[str, Any]:
    value = strict_json_loads(raw, field=field)
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _transformed_shape(
    stored_shape: tuple[int, ...], transforms: tuple[ViewTransformIR, ...]
) -> tuple[int, ...]:
    shape = stored_shape
    if not transforms:
        raise IRValidationError("a tensor view must declare at least one typed transform")
    for transform in transforms:
        parameters = transform.parameters
        if transform.kind == "identity":
            if parameters:
                raise IRValidationError("identity transform cannot have parameters")
            continue
        if transform.kind == "transpose":
            if set(parameters) != {"axes"} or not isinstance(parameters["axes"], list):
                raise IRValidationError("transpose transform requires one axes array")
            axes = parameters["axes"]
            if any(type(axis) is not int for axis in axes) or sorted(axes) != list(
                range(len(shape))
            ):
                raise IRValidationError("transpose axes must be a dimension permutation")
            shape = tuple(shape[axis] for axis in axes)
            continue
        if transform.kind == "reshape":
            if set(parameters) != {"shape"} or not isinstance(parameters["shape"], list):
                raise IRValidationError("reshape transform requires one shape array")
            reshaped = tuple(parameters["shape"])
            if any(type(dimension) is not int or dimension < 0 for dimension in reshaped):
                raise IRValidationError("reshape dimensions must be non-negative integers")
            if _element_count(reshaped) != _element_count(shape):
                raise IRValidationError("reshape transform changes the element count")
            shape = reshaped
            continue
        if transform.kind in {"slice", "split"}:
            expected = {"axis", "start", "stop"}
            if set(parameters) != expected:
                raise IRValidationError(f"{transform.kind} transform requires axis/start/stop")
            axis = parameters["axis"]
            start = parameters["start"]
            stop = parameters["stop"]
            if any(type(value) is not int for value in (axis, start, stop)):
                raise IRValidationError("slice/split bounds must be integers")
            if not 0 <= axis < len(shape) or not 0 <= start <= stop <= shape[axis]:
                raise IRValidationError("slice/split bounds are outside the stored shape")
            mutable = list(shape)
            mutable[axis] = stop - start
            shape = tuple(mutable)
            continue
        raise IRValidationError(
            f"transform {transform.kind!r} is not legal for a single-allocation view"
        )
    return shape


@dataclass(frozen=True, slots=True)
class CodecBindingIR:
    codec_id: str
    codec_version: str
    stored_dtype: str
    _parameters_json: str

    def __post_init__(self) -> None:
        for field_name in ("codec_id", "codec_version", "stored_dtype"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        object.__setattr__(
            self,
            "_parameters_json",
            canonical_json(_json_object(self._parameters_json, field="codec parameters")),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _json_object(self._parameters_json, field="codec parameters")

    def as_dict(self) -> dict[str, Any]:
        return {
            "codec_id": self.codec_id,
            "codec_version": self.codec_version,
            "stored_dtype": self.stored_dtype,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> CodecBindingIR:
        value = require_dict(payload, field="codec binding")
        require_exact_keys(
            value,
            {"codec_id", "codec_version", "stored_dtype", "parameters"},
            field="codec binding",
        )
        return cls(
            codec_id=require_str(value["codec_id"], field="codec_id"),
            codec_version=require_str(value["codec_version"], field="codec_version"),
            stored_dtype=require_str(value["stored_dtype"], field="stored_dtype"),
            _parameters_json=canonical_json(
                require_dict(value["parameters"], field="codec parameters")
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalAllocationIR:
    allocation_id: str
    source_tensor: str
    source_file: str
    byte_offset: int
    byte_length: int
    stored_shape: tuple[int, ...]
    stored_dtype: str
    codec: CodecBindingIR
    content_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("allocation_id", "source_tensor", "source_file", "stored_dtype"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if type(self.byte_offset) is not int or self.byte_offset < 0:
            raise ValueError("allocation byte_offset must be non-negative")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ValueError("allocation byte_length must be non-negative")
        object.__setattr__(self, "stored_shape", _shape(self.stored_shape, field="stored_shape"))
        if not isinstance(self.codec, CodecBindingIR):
            raise TypeError("allocation codec must be a CodecBindingIR")
        object.__setattr__(
            self,
            "content_fingerprint",
            require_sha256(self.content_fingerprint, field="allocation content fingerprint"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "source_tensor": self.source_tensor,
            "source_file": self.source_file,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "stored_shape": list(self.stored_shape),
            "stored_dtype": self.stored_dtype,
            "codec": self.codec.as_dict(),
            "content_fingerprint": self.content_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> PhysicalAllocationIR:
        value = require_dict(payload, field="physical allocation")
        require_exact_keys(
            value,
            {
                "allocation_id",
                "source_tensor",
                "source_file",
                "byte_offset",
                "byte_length",
                "stored_shape",
                "stored_dtype",
                "codec",
                "content_fingerprint",
            },
            field="physical allocation",
        )
        return cls(
            allocation_id=require_str(value["allocation_id"], field="allocation_id"),
            source_tensor=require_str(value["source_tensor"], field="source_tensor"),
            source_file=require_str(value["source_file"], field="source_file"),
            byte_offset=require_int(value["byte_offset"], field="byte_offset", minimum=0),
            byte_length=require_int(value["byte_length"], field="byte_length", minimum=0),
            stored_shape=_shape(value["stored_shape"], field="stored_shape"),
            stored_dtype=require_str(value["stored_dtype"], field="stored_dtype"),
            codec=CodecBindingIR.from_dict(value["codec"]),
            content_fingerprint=require_str(
                value["content_fingerprint"], field="content_fingerprint"
            ),
        )


@dataclass(frozen=True, slots=True)
class ViewTransformIR:
    kind: str
    _parameters_json: str

    def __post_init__(self) -> None:
        allowed = {"identity", "slice", "split", "transpose", "reshape", "concatenate"}
        object.__setattr__(self, "kind", require_str(self.kind, field="transform kind"))
        if self.kind not in allowed:
            raise ValueError(f"unsupported typed view transform: {self.kind!r}")
        object.__setattr__(
            self,
            "_parameters_json",
            canonical_json(_json_object(self._parameters_json, field="transform parameters")),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _json_object(self._parameters_json, field="transform parameters")

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "parameters": self.parameters}

    @classmethod
    def from_dict(cls, payload: Any) -> ViewTransformIR:
        value = require_dict(payload, field="view transform")
        require_exact_keys(value, {"kind", "parameters"}, field="view transform")
        return cls(
            kind=require_str(value["kind"], field="transform kind"),
            _parameters_json=canonical_json(
                require_dict(value["parameters"], field="transform parameters")
            ),
        )


@dataclass(frozen=True, slots=True)
class TensorViewIR:
    view_id: str
    logical_name: str
    allocation_id: str
    logical_shape: tuple[int, ...]
    transforms: tuple[ViewTransformIR, ...]
    semantic_role: str
    parameter_kind: str = "parameter"

    def __post_init__(self) -> None:
        for field_name in ("view_id", "logical_name", "allocation_id", "semantic_role"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if self.parameter_kind not in {"parameter", "buffer"}:
            raise ValueError("parameter_kind must be 'parameter' or 'buffer'")
        object.__setattr__(self, "logical_shape", _shape(self.logical_shape, field="logical_shape"))
        transforms = tuple(self.transforms)
        if not transforms or any(not isinstance(item, ViewTransformIR) for item in transforms):
            raise TypeError("view transforms must be ViewTransformIR records")
        object.__setattr__(self, "transforms", transforms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "logical_name": self.logical_name,
            "allocation_id": self.allocation_id,
            "logical_shape": list(self.logical_shape),
            "transforms": [transform.as_dict() for transform in self.transforms],
            "semantic_role": self.semantic_role,
            "parameter_kind": self.parameter_kind,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TensorViewIR:
        value = require_dict(payload, field="tensor view")
        require_exact_keys(
            value,
            {
                "view_id",
                "logical_name",
                "allocation_id",
                "logical_shape",
                "transforms",
                "semantic_role",
                "parameter_kind",
            },
            field="tensor view",
        )
        transforms = require_list(value["transforms"], field="view transforms")
        return cls(
            view_id=require_str(value["view_id"], field="view_id"),
            logical_name=require_str(value["logical_name"], field="logical_name"),
            allocation_id=require_str(value["allocation_id"], field="allocation_id"),
            logical_shape=_shape(value["logical_shape"], field="logical_shape"),
            transforms=tuple(ViewTransformIR.from_dict(item) for item in transforms),
            semantic_role=require_str(value["semantic_role"], field="semantic_role"),
            parameter_kind=require_str(value["parameter_kind"], field="parameter_kind"),
        )


@dataclass(frozen=True, slots=True)
class AliasEvidenceIR:
    kind: str
    config_fields: tuple[str, ...]
    adapter_rule: str
    certification_status: str
    required_followup: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("kind", "adapter_rule", "certification_status"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        object.__setattr__(
            self,
            "config_fields",
            _strings(self.config_fields, field="config_fields", sorted_unique=True),
        )
        object.__setattr__(
            self,
            "required_followup",
            _strings(self.required_followup, field="required_followup", sorted_unique=True),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "config_fields": list(self.config_fields),
            "adapter_rule": self.adapter_rule,
            "certification_status": self.certification_status,
            "required_followup": list(self.required_followup),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> AliasEvidenceIR:
        value = require_dict(payload, field="alias evidence")
        require_exact_keys(
            value,
            {
                "kind",
                "config_fields",
                "adapter_rule",
                "certification_status",
                "required_followup",
            },
            field="alias evidence",
        )
        return cls(
            kind=require_str(value["kind"], field="alias evidence kind"),
            config_fields=_strings(
                value["config_fields"], field="config_fields", sorted_unique=True
            ),
            adapter_rule=require_str(value["adapter_rule"], field="adapter_rule"),
            certification_status=require_str(
                value["certification_status"], field="certification_status"
            ),
            required_followup=_strings(
                value["required_followup"], field="required_followup", sorted_unique=True
            ),
        )


@dataclass(frozen=True, slots=True)
class AliasClassIR:
    class_id: str
    allocation_id: str
    logical_names: tuple[str, ...]
    evidence: AliasEvidenceIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_id", require_str(self.class_id, field="class_id"))
        object.__setattr__(
            self, "allocation_id", require_str(self.allocation_id, field="allocation_id")
        )
        logical_names = _strings(
            self.logical_names, field="alias logical_names", sorted_unique=True
        )
        if len(logical_names) < 2:
            raise ValueError("an alias class must contain at least two logical names")
        object.__setattr__(self, "logical_names", logical_names)
        if not isinstance(self.evidence, AliasEvidenceIR):
            raise TypeError("alias evidence must be AliasEvidenceIR")

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "allocation_id": self.allocation_id,
            "logical_names": list(self.logical_names),
            "evidence": self.evidence.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> AliasClassIR:
        value = require_dict(payload, field="alias class")
        require_exact_keys(
            value,
            {"class_id", "allocation_id", "logical_names", "evidence"},
            field="alias class",
        )
        return cls(
            class_id=require_str(value["class_id"], field="class_id"),
            allocation_id=require_str(value["allocation_id"], field="allocation_id"),
            logical_names=_strings(
                value["logical_names"], field="logical_names", sorted_unique=True
            ),
            evidence=AliasEvidenceIR.from_dict(value["evidence"]),
        )


@dataclass(frozen=True, slots=True)
class TensorClassificationIR:
    source_name: str
    allocation_id: str
    disposition: str
    logical_view_ids: tuple[str, ...]
    rule_id: str
    reason: str

    def __post_init__(self) -> None:
        for field_name in ("source_name", "allocation_id", "rule_id", "reason"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        allowed = {"parameter", "buffer", "codec-metadata", "ignored"}
        if self.disposition not in allowed:
            raise ValueError(f"unsupported tensor disposition: {self.disposition!r}")
        views = _strings(self.logical_view_ids, field="logical_view_ids", sorted_unique=True)
        if self.disposition in {"parameter", "buffer"} and not views:
            raise ValueError(f"{self.disposition} classification requires a logical view")
        if self.disposition in {"ignored", "codec-metadata"} and views:
            raise ValueError(f"{self.disposition} classification cannot own logical views")
        object.__setattr__(self, "logical_view_ids", views)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "allocation_id": self.allocation_id,
            "disposition": self.disposition,
            "logical_view_ids": list(self.logical_view_ids),
            "rule_id": self.rule_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TensorClassificationIR:
        value = require_dict(payload, field="tensor classification")
        require_exact_keys(
            value,
            {
                "source_name",
                "allocation_id",
                "disposition",
                "logical_view_ids",
                "rule_id",
                "reason",
            },
            field="tensor classification",
        )
        return cls(
            source_name=require_str(value["source_name"], field="source_name"),
            allocation_id=require_str(value["allocation_id"], field="allocation_id"),
            disposition=require_str(value["disposition"], field="disposition"),
            logical_view_ids=_strings(
                value["logical_view_ids"], field="logical_view_ids", sorted_unique=True
            ),
            rule_id=require_str(value["rule_id"], field="rule_id"),
            reason=require_str(value["reason"], field="reason"),
        )


@dataclass(frozen=True, slots=True)
class PhysicalWeightIR:
    source_fingerprint: str
    tensor_index_fingerprint: str
    adapter_fingerprint: str
    allocations: tuple[PhysicalAllocationIR, ...]
    views: tuple[TensorViewIR, ...]
    alias_classes: tuple[AliasClassIR, ...]
    classifications: tuple[TensorClassificationIR, ...]
    fingerprint: str
    schema_version: str = PHYSICAL_WEIGHT_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_WEIGHT_IR_SCHEMA:
            raise ValueError(f"unsupported physical-weight schema: {self.schema_version!r}")
        for field_name in (
            "source_fingerprint",
            "tensor_index_fingerprint",
            "adapter_fingerprint",
            "fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        allocations = tuple(self.allocations)
        views = tuple(self.views)
        aliases = tuple(self.alias_classes)
        classifications = tuple(self.classifications)
        if allocations != tuple(sorted(allocations, key=lambda item: item.allocation_id)):
            raise IRValidationError("physical allocations must be sorted by allocation_id")
        if views != tuple(sorted(views, key=lambda item: item.logical_name)):
            raise IRValidationError("tensor views must be sorted by logical_name")
        if aliases != tuple(sorted(aliases, key=lambda item: item.class_id)):
            raise IRValidationError("alias classes must be sorted by class_id")
        if classifications != tuple(sorted(classifications, key=lambda item: item.source_name)):
            raise IRValidationError("tensor classifications must be sorted by source_name")
        for collection, key, label in (
            (allocations, lambda item: item.allocation_id, "allocation IDs"),
            (allocations, lambda item: item.source_tensor, "allocation source tensors"),
            (views, lambda item: item.view_id, "view IDs"),
            (views, lambda item: item.logical_name, "logical names"),
            (aliases, lambda item: item.class_id, "alias class IDs"),
            (classifications, lambda item: item.source_name, "classification source names"),
        ):
            values = [key(item) for item in collection]
            if len(values) != len(set(values)):
                raise IRValidationError(f"physical IR contains duplicate {label}")
        allocations_by_id = {item.allocation_id: item for item in allocations}
        views_by_id = {item.view_id: item for item in views}
        classifications_by_source = {item.source_name: item for item in classifications}
        if set(item.source_tensor for item in allocations) != set(classifications_by_source):
            raise IRValidationError("every physical allocation must have one source classification")
        classified_views: set[str] = set()
        for allocation in allocations:
            if allocation.codec.stored_dtype != allocation.stored_dtype:
                raise IRValidationError("allocation dtype disagrees with its codec binding")
        for view in views:
            allocation = allocations_by_id.get(view.allocation_id)
            if allocation is None:
                raise IRValidationError("logical view references a missing allocation")
            if _transformed_shape(allocation.stored_shape, view.transforms) != view.logical_shape:
                raise IRValidationError("logical view shape does not match its typed transforms")
        for classification in classifications:
            allocation = allocations_by_id.get(classification.allocation_id)
            if allocation is None or allocation.source_tensor != classification.source_name:
                raise IRValidationError(
                    "classification allocation/source ownership is inconsistent"
                )
            for view_id in classification.logical_view_ids:
                view = views_by_id.get(view_id)
                if view is None or view.allocation_id != classification.allocation_id:
                    raise IRValidationError("classification references an invalid logical view")
                if view.parameter_kind != classification.disposition:
                    raise IRValidationError(
                        "classification disposition disagrees with its logical view kind"
                    )
                if view_id in classified_views:
                    raise IRValidationError("a logical view has multiple source owners")
                classified_views.add(view_id)
        if classified_views != set(views_by_id):
            raise IRValidationError("every logical view must have exactly one source owner")
        aliases_by_allocation: dict[str, AliasClassIR] = {}
        for alias in aliases:
            if alias.allocation_id not in allocations_by_id:
                raise IRValidationError("alias class references a missing allocation")
            if alias.allocation_id in aliases_by_allocation:
                raise IRValidationError("an allocation has multiple alias classes")
            actual = tuple(
                sorted(
                    item.logical_name for item in views if item.allocation_id == alias.allocation_id
                )
            )
            if actual != alias.logical_names:
                raise IRValidationError("alias class does not equal its allocation's logical views")
            aliases_by_allocation[alias.allocation_id] = alias
        for allocation_id in allocations_by_id:
            logical_count = sum(item.allocation_id == allocation_id for item in views)
            if logical_count > 1 and allocation_id not in aliases_by_allocation:
                raise IRValidationError("shared physical allocation lacks an explicit alias class")
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "alias_classes", aliases)
        object.__setattr__(self, "classifications", classifications)
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("physical-weight IR fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "tensor_index_fingerprint": self.tensor_index_fingerprint,
            "adapter_fingerprint": self.adapter_fingerprint,
            "allocations": [item.as_dict() for item in self.allocations],
            "views": [item.as_dict() for item in self.views],
            "alias_classes": [item.as_dict() for item in self.alias_classes],
            "classifications": [item.as_dict() for item in self.classifications],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(
        cls,
        *,
        source_fingerprint: str,
        tensor_index_fingerprint: str,
        adapter_fingerprint: str,
        allocations: tuple[PhysicalAllocationIR, ...],
        views: tuple[TensorViewIR, ...],
        alias_classes: tuple[AliasClassIR, ...],
        classifications: tuple[TensorClassificationIR, ...],
    ) -> PhysicalWeightIR:
        payload = {
            "schema_version": PHYSICAL_WEIGHT_IR_SCHEMA,
            "source_fingerprint": source_fingerprint,
            "tensor_index_fingerprint": tensor_index_fingerprint,
            "adapter_fingerprint": adapter_fingerprint,
            "allocations": [item.as_dict() for item in allocations],
            "views": [item.as_dict() for item in views],
            "alias_classes": [item.as_dict() for item in alias_classes],
            "classifications": [item.as_dict() for item in classifications],
        }
        return cls(
            source_fingerprint=source_fingerprint,
            tensor_index_fingerprint=tensor_index_fingerprint,
            adapter_fingerprint=adapter_fingerprint,
            allocations=allocations,
            views=views,
            alias_classes=alias_classes,
            classifications=classifications,
            fingerprint=canonical_sha256(payload),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> PhysicalWeightIR:
        value = require_dict(payload, field="physical-weight IR")
        require_exact_keys(
            value,
            {
                "schema_version",
                "source_fingerprint",
                "tensor_index_fingerprint",
                "adapter_fingerprint",
                "allocations",
                "views",
                "alias_classes",
                "classifications",
                "fingerprint",
            },
            field="physical-weight IR",
        )
        return cls(
            schema_version=require_str(value["schema_version"], field="physical schema"),
            source_fingerprint=require_str(value["source_fingerprint"], field="source_fingerprint"),
            tensor_index_fingerprint=require_str(
                value["tensor_index_fingerprint"], field="tensor_index_fingerprint"
            ),
            adapter_fingerprint=require_str(
                value["adapter_fingerprint"], field="adapter_fingerprint"
            ),
            allocations=tuple(
                PhysicalAllocationIR.from_dict(item)
                for item in require_list(value["allocations"], field="allocations")
            ),
            views=tuple(
                TensorViewIR.from_dict(item) for item in require_list(value["views"], field="views")
            ),
            alias_classes=tuple(
                AliasClassIR.from_dict(item)
                for item in require_list(value["alias_classes"], field="alias_classes")
            ),
            classifications=tuple(
                TensorClassificationIR.from_dict(item)
                for item in require_list(value["classifications"], field="classifications")
            ),
            fingerprint=require_str(value["fingerprint"], field="physical fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class ModelDimensionsIR:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    physical_vocab_rows: int
    max_position_embeddings: int

    def __post_init__(self) -> None:
        architecture_neutral_fields = {
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        }
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            minimum = 0 if field_name in architecture_neutral_fields else 1
            if type(value) is not int or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{field_name} must be a {qualifier} integer")
        attention_shape = (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        if any(attention_shape) and not all(attention_shape):
            raise ValueError(
                "attention dimensions must either all be zero or all be positive integers"
            )
        if self.num_key_value_heads and self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.physical_vocab_rows < self.vocab_size:
            raise ValueError("physical vocabulary rows cannot be smaller than vocab_size")

    def as_dict(self) -> dict[str, int]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Any) -> ModelDimensionsIR:
        value = require_dict(payload, field="model dimensions")
        expected = set(cls.__dataclass_fields__)
        require_exact_keys(value, expected, field="model dimensions")
        architecture_neutral_fields = {
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        }
        return cls(
            **{
                field_name: require_int(
                    value[field_name],
                    field=field_name,
                    minimum=0 if field_name in architecture_neutral_fields else 1,
                )
                for field_name in expected
            }
        )


@dataclass(frozen=True, slots=True)
class LogicalParameterRefIR:
    logical_name: str
    view_id: str
    semantic_role: str
    parameter_kind: str

    def __post_init__(self) -> None:
        for field_name in ("logical_name", "view_id", "semantic_role"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if self.parameter_kind not in {"parameter", "buffer"}:
            raise ValueError("parameter_kind must be 'parameter' or 'buffer'")

    def as_dict(self) -> dict[str, str]:
        return {
            "logical_name": self.logical_name,
            "view_id": self.view_id,
            "semantic_role": self.semantic_role,
            "parameter_kind": self.parameter_kind,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> LogicalParameterRefIR:
        value = require_dict(payload, field="logical parameter reference")
        require_exact_keys(
            value,
            {"logical_name", "view_id", "semantic_role", "parameter_kind"},
            field="logical parameter reference",
        )
        return cls(
            logical_name=require_str(value["logical_name"], field="logical_name"),
            view_id=require_str(value["view_id"], field="view_id"),
            semantic_role=require_str(value["semantic_role"], field="semantic_role"),
            parameter_kind=require_str(value["parameter_kind"], field="parameter_kind"),
        )


@dataclass(frozen=True, slots=True)
class OperationIR:
    operation_id: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    parameters: tuple[str, ...]
    _attributes_json: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", require_str(self.operation_id, field="operation_id")
        )
        object.__setattr__(self, "kind", require_str(self.kind, field="operation kind"))
        object.__setattr__(self, "inputs", _strings(self.inputs, field="operation inputs"))
        outputs = _strings(self.outputs, field="operation outputs")
        if not outputs or len(outputs) != len(set(outputs)):
            raise ValueError("operation outputs must be non-empty and unique")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(
            self,
            "parameters",
            _strings(self.parameters, field="operation parameters", sorted_unique=True),
        )
        object.__setattr__(
            self,
            "_attributes_json",
            canonical_json(_json_object(self._attributes_json, field="operation attributes")),
        )

    @property
    def attributes(self) -> dict[str, Any]:
        return _json_object(self._attributes_json, field="operation attributes")

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "parameters": list(self.parameters),
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> OperationIR:
        value = require_dict(payload, field="operation")
        require_exact_keys(
            value,
            {"operation_id", "kind", "inputs", "outputs", "parameters", "attributes"},
            field="operation",
        )
        return cls(
            operation_id=require_str(value["operation_id"], field="operation_id"),
            kind=require_str(value["kind"], field="operation kind"),
            inputs=_strings(value["inputs"], field="operation inputs"),
            outputs=_strings(value["outputs"], field="operation outputs"),
            parameters=_strings(
                value["parameters"], field="operation parameters", sorted_unique=True
            ),
            _attributes_json=canonical_json(
                require_dict(value["attributes"], field="operation attributes")
            ),
        )


@dataclass(frozen=True, slots=True)
class PortIR:
    name: str
    semantic: str
    dtype: str
    shape: tuple[str, ...]
    value: str

    def __post_init__(self) -> None:
        for field_name in ("name", "semantic", "dtype", "value"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        object.__setattr__(self, "shape", _strings(self.shape, field="port shape"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic": self.semantic,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> PortIR:
        value = require_dict(payload, field="port")
        require_exact_keys(value, {"name", "semantic", "dtype", "shape", "value"}, field="port")
        return cls(
            name=require_str(value["name"], field="port name"),
            semantic=require_str(value["semantic"], field="port semantic"),
            dtype=require_str(value["dtype"], field="port dtype"),
            shape=_strings(value["shape"], field="port shape"),
            value=require_str(value["value"], field="port value"),
        )


@dataclass(frozen=True, slots=True)
class NumericalSemanticsIR:
    reference_contract: str
    accumulation: str
    softmax: str
    positional_arithmetic: str
    optimization_contract: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )

    def as_dict(self) -> dict[str, str]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Any) -> NumericalSemanticsIR:
        value = require_dict(payload, field="numerical semantics")
        expected = set(cls.__dataclass_fields__)
        require_exact_keys(value, expected, field="numerical semantics")
        return cls(**{name: require_str(value[name], field=name) for name in expected})


@dataclass(frozen=True, slots=True)
class ModelIR:
    source_fingerprint: str
    adapter_id: str
    adapter_version: str
    adapter_fingerprint: str
    architecture_id: str
    physical_weights_fingerprint: str
    dimensions: ModelDimensionsIR
    parameters: tuple[LogicalParameterRefIR, ...]
    operations: tuple[OperationIR, ...]
    state_refs: tuple[str, ...]
    input_ports: tuple[PortIR, ...]
    output_ports: tuple[PortIR, ...]
    numerical_semantics: NumericalSemanticsIR
    fingerprint: str
    schema_version: str = MODEL_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_IR_SCHEMA:
            raise ValueError(f"unsupported model IR schema: {self.schema_version!r}")
        for field_name in (
            "source_fingerprint",
            "adapter_fingerprint",
            "physical_weights_fingerprint",
            "fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in ("adapter_id", "adapter_version", "architecture_id"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if not isinstance(self.dimensions, ModelDimensionsIR):
            raise TypeError("dimensions must be ModelDimensionsIR")
        parameters = tuple(self.parameters)
        if parameters != tuple(sorted(parameters, key=lambda item: item.logical_name)):
            raise IRValidationError("model parameters must be sorted by logical_name")
        parameter_names = [item.logical_name for item in parameters]
        view_ids = [item.view_id for item in parameters]
        if len(parameter_names) != len(set(parameter_names)) or len(view_ids) != len(set(view_ids)):
            raise IRValidationError("model parameter names and view IDs must be unique")
        operations = tuple(self.operations)
        if not operations:
            raise IRValidationError("model IR must contain operations")
        operation_ids = [item.operation_id for item in operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise IRValidationError("model operation IDs must be unique")
        available_values = {port.value for port in self.input_ports}
        produced_values: set[str] = set()
        known_parameters = set(parameter_names)
        used_parameters: set[str] = set()
        for operation in operations:
            unknown_inputs = set(operation.inputs) - available_values
            if unknown_inputs:
                raise IRValidationError(
                    f"operation {operation.operation_id} has unavailable inputs: "
                    f"{sorted(unknown_inputs)}"
                )
            unknown_parameters = set(operation.parameters) - known_parameters
            if unknown_parameters:
                raise IRValidationError(
                    f"operation {operation.operation_id} has unknown parameters: "
                    f"{sorted(unknown_parameters)}"
                )
            used_parameters.update(operation.parameters)
            duplicate_outputs = set(operation.outputs) & produced_values
            if duplicate_outputs:
                raise IRValidationError(
                    f"operation {operation.operation_id} redefines values: "
                    f"{sorted(duplicate_outputs)}"
                )
            available_values.update(operation.outputs)
            produced_values.update(operation.outputs)
        if used_parameters != known_parameters:
            raise IRValidationError(
                "model operations do not exactly consume every declared parameter"
            )
        if any(port.value not in available_values for port in self.output_ports):
            raise IRValidationError("model output port references an unavailable value")
        input_names = [port.name for port in self.input_ports]
        output_names = [port.name for port in self.output_ports]
        if not input_names or not output_names:
            raise IRValidationError("model must have input and output ports")
        if len(input_names) != len(set(input_names)) or len(output_names) != len(set(output_names)):
            raise IRValidationError("model port names must be unique")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self, "state_refs", _strings(self.state_refs, field="state_refs", sorted_unique=True)
        )
        if not isinstance(self.numerical_semantics, NumericalSemanticsIR):
            raise TypeError("numerical_semantics must be NumericalSemanticsIR")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("model IR fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "adapter_fingerprint": self.adapter_fingerprint,
            "architecture_id": self.architecture_id,
            "physical_weights_fingerprint": self.physical_weights_fingerprint,
            "dimensions": self.dimensions.as_dict(),
            "parameters": [item.as_dict() for item in self.parameters],
            "operations": [item.as_dict() for item in self.operations],
            "state_refs": list(self.state_refs),
            "input_ports": [item.as_dict() for item in self.input_ports],
            "output_ports": [item.as_dict() for item in self.output_ports],
            "numerical_semantics": self.numerical_semantics.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **kwargs: Any) -> ModelIR:
        payload = {
            "schema_version": MODEL_IR_SCHEMA,
            "source_fingerprint": kwargs["source_fingerprint"],
            "adapter_id": kwargs["adapter_id"],
            "adapter_version": kwargs["adapter_version"],
            "adapter_fingerprint": kwargs["adapter_fingerprint"],
            "architecture_id": kwargs["architecture_id"],
            "physical_weights_fingerprint": kwargs["physical_weights_fingerprint"],
            "dimensions": kwargs["dimensions"].as_dict(),
            "parameters": [item.as_dict() for item in kwargs["parameters"]],
            "operations": [item.as_dict() for item in kwargs["operations"]],
            "state_refs": list(kwargs["state_refs"]),
            "input_ports": [item.as_dict() for item in kwargs["input_ports"]],
            "output_ports": [item.as_dict() for item in kwargs["output_ports"]],
            "numerical_semantics": kwargs["numerical_semantics"].as_dict(),
        }
        return cls(**kwargs, fingerprint=canonical_sha256(payload))

    @classmethod
    def from_dict(cls, payload: Any) -> ModelIR:
        value = require_dict(payload, field="model IR")
        require_exact_keys(
            value,
            {
                "schema_version",
                "source_fingerprint",
                "adapter_id",
                "adapter_version",
                "adapter_fingerprint",
                "architecture_id",
                "physical_weights_fingerprint",
                "dimensions",
                "parameters",
                "operations",
                "state_refs",
                "input_ports",
                "output_ports",
                "numerical_semantics",
                "fingerprint",
            },
            field="model IR",
        )
        return cls(
            schema_version=require_str(value["schema_version"], field="model schema"),
            source_fingerprint=require_str(value["source_fingerprint"], field="source_fingerprint"),
            adapter_id=require_str(value["adapter_id"], field="adapter_id"),
            adapter_version=require_str(value["adapter_version"], field="adapter_version"),
            adapter_fingerprint=require_str(
                value["adapter_fingerprint"], field="adapter_fingerprint"
            ),
            architecture_id=require_str(value["architecture_id"], field="architecture_id"),
            physical_weights_fingerprint=require_str(
                value["physical_weights_fingerprint"], field="physical_weights_fingerprint"
            ),
            dimensions=ModelDimensionsIR.from_dict(value["dimensions"]),
            parameters=tuple(
                LogicalParameterRefIR.from_dict(item)
                for item in require_list(value["parameters"], field="parameters")
            ),
            operations=tuple(
                OperationIR.from_dict(item)
                for item in require_list(value["operations"], field="operations")
            ),
            state_refs=_strings(value["state_refs"], field="state_refs", sorted_unique=True),
            input_ports=tuple(
                PortIR.from_dict(item)
                for item in require_list(value["input_ports"], field="input_ports")
            ),
            output_ports=tuple(
                PortIR.from_dict(item)
                for item in require_list(value["output_ports"], field="output_ports")
            ),
            numerical_semantics=NumericalSemanticsIR.from_dict(value["numerical_semantics"]),
            fingerprint=require_str(value["fingerprint"], field="model fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class StateSlotIR:
    slot_id: str
    kind: str
    dtype: str
    shape_expression: tuple[str, ...]
    ownership: str
    lease_behavior: str
    provisional_representation: str
    commit_rule: str
    rollback_rule: str
    memory_charge_expression: str

    def __post_init__(self) -> None:
        for field_name in (
            "slot_id",
            "kind",
            "dtype",
            "ownership",
            "lease_behavior",
            "provisional_representation",
            "commit_rule",
            "rollback_rule",
            "memory_charge_expression",
        ):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        object.__setattr__(
            self, "shape_expression", _strings(self.shape_expression, field="shape_expression")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "kind": self.kind,
            "dtype": self.dtype,
            "shape_expression": list(self.shape_expression),
            "ownership": self.ownership,
            "lease_behavior": self.lease_behavior,
            "provisional_representation": self.provisional_representation,
            "commit_rule": self.commit_rule,
            "rollback_rule": self.rollback_rule,
            "memory_charge_expression": self.memory_charge_expression,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> StateSlotIR:
        value = require_dict(payload, field="state slot")
        expected = {
            "slot_id",
            "kind",
            "dtype",
            "shape_expression",
            "ownership",
            "lease_behavior",
            "provisional_representation",
            "commit_rule",
            "rollback_rule",
            "memory_charge_expression",
        }
        require_exact_keys(value, expected, field="state slot")
        return cls(
            slot_id=require_str(value["slot_id"], field="slot_id"),
            kind=require_str(value["kind"], field="state kind"),
            dtype=require_str(value["dtype"], field="state dtype"),
            shape_expression=_strings(value["shape_expression"], field="shape_expression"),
            ownership=require_str(value["ownership"], field="ownership"),
            lease_behavior=require_str(value["lease_behavior"], field="lease_behavior"),
            provisional_representation=require_str(
                value["provisional_representation"], field="provisional_representation"
            ),
            commit_rule=require_str(value["commit_rule"], field="commit_rule"),
            rollback_rule=require_str(value["rollback_rule"], field="rollback_rule"),
            memory_charge_expression=require_str(
                value["memory_charge_expression"], field="memory_charge_expression"
            ),
        )


@dataclass(frozen=True, slots=True)
class StateOperationIR:
    operation_id: str
    kind: str
    slots: tuple[str, ...]
    _attributes_json: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", require_str(self.operation_id, field="state operation_id")
        )
        object.__setattr__(self, "kind", require_str(self.kind, field="state operation kind"))
        object.__setattr__(self, "slots", _strings(self.slots, field="state operation slots"))
        object.__setattr__(
            self,
            "_attributes_json",
            canonical_json(_json_object(self._attributes_json, field="state attributes")),
        )

    @property
    def attributes(self) -> dict[str, Any]:
        return _json_object(self._attributes_json, field="state attributes")

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "slots": list(self.slots),
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> StateOperationIR:
        value = require_dict(payload, field="state operation")
        require_exact_keys(
            value,
            {"operation_id", "kind", "slots", "attributes"},
            field="state operation",
        )
        return cls(
            operation_id=require_str(value["operation_id"], field="state operation_id"),
            kind=require_str(value["kind"], field="state operation kind"),
            slots=_strings(value["slots"], field="state operation slots"),
            _attributes_json=canonical_json(
                require_dict(value["attributes"], field="state attributes")
            ),
        )


@dataclass(frozen=True, slots=True)
class CommitProtocolIR:
    protocol_id: str
    authority: str
    atomicity: str
    accepted_prefix_rule: str
    rollback: str
    stale_epoch_rule: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )

    def as_dict(self) -> dict[str, str]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Any) -> CommitProtocolIR:
        value = require_dict(payload, field="commit protocol")
        expected = set(cls.__dataclass_fields__)
        require_exact_keys(value, expected, field="commit protocol")
        return cls(**{name: require_str(value[name], field=name) for name in expected})


@dataclass(frozen=True, slots=True)
class CapacityEquationIR:
    quantity: str
    expression: str
    units: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )

    def as_dict(self) -> dict[str, str]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Any) -> CapacityEquationIR:
        value = require_dict(payload, field="capacity equation")
        expected = set(cls.__dataclass_fields__)
        require_exact_keys(value, expected, field="capacity equation")
        return cls(**{name: require_str(value[name], field=name) for name in expected})


@dataclass(frozen=True, slots=True)
class StateIR:
    source_fingerprint: str
    adapter_fingerprint: str
    model_fingerprint: str
    slots: tuple[StateSlotIR, ...]
    initialization: tuple[StateOperationIR, ...]
    prefill_updates: tuple[StateOperationIR, ...]
    decode_updates: tuple[StateOperationIR, ...]
    commit_protocol: CommitProtocolIR
    capacity_equations: tuple[CapacityEquationIR, ...]
    fingerprint: str
    schema_version: str = STATE_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATE_IR_SCHEMA:
            raise ValueError(f"unsupported state IR schema: {self.schema_version!r}")
        for field_name in (
            "source_fingerprint",
            "adapter_fingerprint",
            "model_fingerprint",
            "fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        slots = tuple(self.slots)
        if slots != tuple(sorted(slots, key=lambda item: item.slot_id)):
            raise IRValidationError("state slots must be sorted by slot_id")
        slot_ids = [item.slot_id for item in slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise IRValidationError("state slot IDs must be unique")
        known_slots = set(slot_ids)
        operation_ids: set[str] = set()
        for operation in (*self.initialization, *self.prefill_updates, *self.decode_updates):
            if operation.operation_id in operation_ids:
                raise IRValidationError("state operation IDs must be unique")
            operation_ids.add(operation.operation_id)
            if set(operation.slots) - known_slots:
                raise IRValidationError("state operation references an unknown slot")
        if not isinstance(self.commit_protocol, CommitProtocolIR):
            raise TypeError("commit_protocol must be CommitProtocolIR")
        equations = tuple(self.capacity_equations)
        quantities = [item.quantity for item in equations]
        if quantities != sorted(set(quantities)):
            raise IRValidationError("capacity equations must be sorted and uniquely named")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "capacity_equations", equations)
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("state IR fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "adapter_fingerprint": self.adapter_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "slots": [item.as_dict() for item in self.slots],
            "initialization": [item.as_dict() for item in self.initialization],
            "prefill_updates": [item.as_dict() for item in self.prefill_updates],
            "decode_updates": [item.as_dict() for item in self.decode_updates],
            "commit_protocol": self.commit_protocol.as_dict(),
            "capacity_equations": [item.as_dict() for item in self.capacity_equations],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **kwargs: Any) -> StateIR:
        payload = {
            "schema_version": STATE_IR_SCHEMA,
            "source_fingerprint": kwargs["source_fingerprint"],
            "adapter_fingerprint": kwargs["adapter_fingerprint"],
            "model_fingerprint": kwargs["model_fingerprint"],
            "slots": [item.as_dict() for item in kwargs["slots"]],
            "initialization": [item.as_dict() for item in kwargs["initialization"]],
            "prefill_updates": [item.as_dict() for item in kwargs["prefill_updates"]],
            "decode_updates": [item.as_dict() for item in kwargs["decode_updates"]],
            "commit_protocol": kwargs["commit_protocol"].as_dict(),
            "capacity_equations": [item.as_dict() for item in kwargs["capacity_equations"]],
        }
        return cls(**kwargs, fingerprint=canonical_sha256(payload))

    @classmethod
    def from_dict(cls, payload: Any) -> StateIR:
        value = require_dict(payload, field="state IR")
        require_exact_keys(
            value,
            {
                "schema_version",
                "source_fingerprint",
                "adapter_fingerprint",
                "model_fingerprint",
                "slots",
                "initialization",
                "prefill_updates",
                "decode_updates",
                "commit_protocol",
                "capacity_equations",
                "fingerprint",
            },
            field="state IR",
        )
        return cls(
            schema_version=require_str(value["schema_version"], field="state schema"),
            source_fingerprint=require_str(value["source_fingerprint"], field="source_fingerprint"),
            adapter_fingerprint=require_str(
                value["adapter_fingerprint"], field="adapter_fingerprint"
            ),
            model_fingerprint=require_str(value["model_fingerprint"], field="model_fingerprint"),
            slots=tuple(
                StateSlotIR.from_dict(item)
                for item in require_list(value["slots"], field="state slots")
            ),
            initialization=tuple(
                StateOperationIR.from_dict(item)
                for item in require_list(value["initialization"], field="initialization")
            ),
            prefill_updates=tuple(
                StateOperationIR.from_dict(item)
                for item in require_list(value["prefill_updates"], field="prefill updates")
            ),
            decode_updates=tuple(
                StateOperationIR.from_dict(item)
                for item in require_list(value["decode_updates"], field="decode updates")
            ),
            commit_protocol=CommitProtocolIR.from_dict(value["commit_protocol"]),
            capacity_equations=tuple(
                CapacityEquationIR.from_dict(item)
                for item in require_list(value["capacity_equations"], field="capacity equations")
            ),
            fingerprint=require_str(value["fingerprint"], field="state fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class BoundAssetIR:
    path: str
    sha256: str
    byte_count: int
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", require_str(self.path, field="asset path"))
        object.__setattr__(self, "sha256", require_sha256(self.sha256, field="asset sha256"))
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("bound asset byte_count must be positive")
        object.__setattr__(self, "role", require_str(self.role, field="asset role"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> BoundAssetIR:
        value = require_dict(payload, field="bound asset")
        require_exact_keys(value, {"path", "sha256", "byte_count", "role"}, field="bound asset")
        return cls(
            path=require_str(value["path"], field="asset path"),
            sha256=require_str(value["sha256"], field="asset sha256"),
            byte_count=require_int(value["byte_count"], field="asset bytes", minimum=1),
            role=require_str(value["role"], field="asset role"),
        )


@dataclass(frozen=True, slots=True)
class TokenSpaceIR:
    space_id: str
    token_count: int
    physical_row_count: int
    ordering_contract: str
    tokenizer_status: str

    def __post_init__(self) -> None:
        for field_name in ("space_id", "ordering_contract", "tokenizer_status"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if type(self.physical_row_count) is not int or self.physical_row_count < self.token_count:
            raise ValueError("physical_row_count must address every token")

    def as_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "token_count": self.token_count,
            "physical_row_count": self.physical_row_count,
            "ordering_contract": self.ordering_contract,
            "tokenizer_status": self.tokenizer_status,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TokenSpaceIR:
        value = require_dict(payload, field="token space")
        require_exact_keys(
            value,
            {
                "space_id",
                "token_count",
                "physical_row_count",
                "ordering_contract",
                "tokenizer_status",
            },
            field="token space",
        )
        return cls(
            space_id=require_str(value["space_id"], field="space_id"),
            token_count=require_int(value["token_count"], field="token_count", minimum=1),
            physical_row_count=require_int(
                value["physical_row_count"], field="physical_row_count", minimum=1
            ),
            ordering_contract=require_str(value["ordering_contract"], field="ordering_contract"),
            tokenizer_status=require_str(value["tokenizer_status"], field="tokenizer_status"),
        )


@dataclass(frozen=True, slots=True)
class RowMapperIR:
    mapper_id: str
    source_space: str
    output_rows: str
    kind: str
    token_count: int
    row_count: int
    unreachable_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("mapper_id", "source_space", "output_rows", "kind"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if self.kind not in {"identity", "padded-identity", "permutation", "sparse", "projected"}:
            raise ValueError(f"unsupported row mapper kind: {self.kind!r}")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("row mapper token_count must be positive")
        if type(self.row_count) is not int or self.row_count < self.token_count:
            raise ValueError("row mapper row_count must cover the token space")
        unreachable = tuple(self.unreachable_rows)
        if unreachable != tuple(sorted(set(unreachable))) or any(
            type(item) is not int or item < self.token_count or item >= self.row_count
            for item in unreachable
        ):
            raise ValueError("unreachable rows must be sorted unique padded row IDs")
        expected = tuple(range(self.token_count, self.row_count))
        if self.kind == "identity" and (self.row_count != self.token_count or unreachable):
            raise ValueError("identity mapper requires equal token/row domains")
        if self.kind == "padded-identity" and unreachable != expected:
            raise ValueError("padded-identity must enumerate every unreachable padded row")
        object.__setattr__(self, "unreachable_rows", unreachable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapper_id": self.mapper_id,
            "source_space": self.source_space,
            "output_rows": self.output_rows,
            "kind": self.kind,
            "token_count": self.token_count,
            "row_count": self.row_count,
            "unreachable_rows": list(self.unreachable_rows),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RowMapperIR:
        value = require_dict(payload, field="row mapper")
        require_exact_keys(
            value,
            {
                "mapper_id",
                "source_space",
                "output_rows",
                "kind",
                "token_count",
                "row_count",
                "unreachable_rows",
            },
            field="row mapper",
        )
        rows = require_list(value["unreachable_rows"], field="unreachable_rows")
        return cls(
            mapper_id=require_str(value["mapper_id"], field="mapper_id"),
            source_space=require_str(value["source_space"], field="source_space"),
            output_rows=require_str(value["output_rows"], field="output_rows"),
            kind=require_str(value["kind"], field="mapper kind"),
            token_count=require_int(value["token_count"], field="token_count", minimum=1),
            row_count=require_int(value["row_count"], field="row_count", minimum=1),
            unreachable_rows=tuple(
                require_int(item, field="unreachable row", minimum=0) for item in rows
            ),
        )


@dataclass(frozen=True, slots=True)
class SpecialTokenIR:
    name: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_str(self.name, field="special token name"))
        values = tuple(self.token_ids)
        if values != tuple(sorted(set(values))) or any(
            type(value) is not int or value < 0 for value in values
        ):
            raise ValueError("special token IDs must be sorted unique non-negative integers")
        object.__setattr__(self, "token_ids", values)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "token_ids": list(self.token_ids)}

    @classmethod
    def from_dict(cls, payload: Any) -> SpecialTokenIR:
        value = require_dict(payload, field="special token")
        require_exact_keys(value, {"name", "token_ids"}, field="special token")
        return cls(
            name=require_str(value["name"], field="special token name"),
            token_ids=tuple(
                require_int(item, field="special token ID", minimum=0)
                for item in require_list(value["token_ids"], field="special token IDs")
            ),
        )


@dataclass(frozen=True, slots=True)
class ChatTemplateIR:
    template_id: str
    source_asset: str
    content: str
    sha256: str

    def __post_init__(self) -> None:
        for field_name in ("template_id", "source_asset"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if type(self.content) is not str or not self.content:
            raise ValueError("chat template content must be non-empty")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, field="template sha256"))
        if canonical_sha256({"content": self.content}) != self.sha256:
            raise ValueError("chat template digest does not match content")

    def as_dict(self) -> dict[str, str]:
        return {
            "template_id": self.template_id,
            "source_asset": self.source_asset,
            "content": self.content,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ChatTemplateIR:
        value = require_dict(payload, field="chat template")
        require_exact_keys(
            value, {"template_id", "source_asset", "content", "sha256"}, field="chat template"
        )
        return cls(
            template_id=require_str(value["template_id"], field="template_id"),
            source_asset=require_str(value["source_asset"], field="source_asset"),
            content=require_str(value["content"], field="template content"),
            sha256=require_str(value["sha256"], field="template sha256"),
        )


@dataclass(frozen=True, slots=True)
class OutputSpaceIR:
    space_id: str
    semantic: str
    row_mapper_id: str
    row_count: int

    def __post_init__(self) -> None:
        for field_name in ("space_id", "semantic", "row_mapper_id"):
            object.__setattr__(
                self, field_name, require_str(getattr(self, field_name), field=field_name)
            )
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("output-space row_count must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "semantic": self.semantic,
            "row_mapper_id": self.row_mapper_id,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> OutputSpaceIR:
        value = require_dict(payload, field="output space")
        require_exact_keys(
            value, {"space_id", "semantic", "row_mapper_id", "row_count"}, field="output space"
        )
        return cls(
            space_id=require_str(value["space_id"], field="space_id"),
            semantic=require_str(value["semantic"], field="semantic"),
            row_mapper_id=require_str(value["row_mapper_id"], field="row_mapper_id"),
            row_count=require_int(value["row_count"], field="row_count", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class IOIR:
    source_fingerprint: str
    adapter_fingerprint: str
    model_fingerprint: str
    text_spaces: tuple[TokenSpaceIR, ...]
    row_mappers: tuple[RowMapperIR, ...]
    special_tokens: tuple[SpecialTokenIR, ...]
    tokenizer_assets: tuple[BoundAssetIR, ...]
    processor_assets: tuple[BoundAssetIR, ...]
    chat_templates: tuple[ChatTemplateIR, ...]
    _generation_defaults_json: str
    output_spaces: tuple[OutputSpaceIR, ...]
    missing_requirements: tuple[str, ...]
    portable: bool
    fingerprint: str
    schema_version: str = IO_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != IO_IR_SCHEMA:
            raise ValueError(f"unsupported IO IR schema: {self.schema_version!r}")
        for field_name in (
            "source_fingerprint",
            "adapter_fingerprint",
            "model_fingerprint",
            "fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for values, key, label in (
            (self.text_spaces, lambda item: item.space_id, "text spaces"),
            (self.row_mappers, lambda item: item.mapper_id, "row mappers"),
            (self.special_tokens, lambda item: item.name, "special tokens"),
            (self.tokenizer_assets, lambda item: item.path, "tokenizer assets"),
            (self.processor_assets, lambda item: item.path, "processor assets"),
            (self.chat_templates, lambda item: item.template_id, "chat templates"),
            (self.output_spaces, lambda item: item.space_id, "output spaces"),
        ):
            keys = [key(item) for item in values]
            if keys != sorted(set(keys)):
                raise IRValidationError(f"{label} must be sorted with unique identities")
        if not self.text_spaces or not self.row_mappers or not self.output_spaces:
            raise IRValidationError("IOIR requires text, row-mapper, and output spaces")
        mapper_ids = {item.mapper_id for item in self.row_mappers}
        text_ids = {item.space_id for item in self.text_spaces}
        maximum_token_id = max(item.token_count for item in self.text_spaces) - 1
        for special in self.special_tokens:
            if any(token_id > maximum_token_id for token_id in special.token_ids):
                raise IRValidationError("special token ID is outside every declared token space")
        for mapper in self.row_mappers:
            if mapper.source_space not in text_ids:
                raise IRValidationError("row mapper references an unknown token space")
        for output in self.output_spaces:
            if output.row_mapper_id not in mapper_ids:
                raise IRValidationError("output space references an unknown row mapper")
        tokenizer_paths = {item.path for item in self.tokenizer_assets}
        processor_paths = {item.path for item in self.processor_assets}
        if tokenizer_paths & processor_paths:
            raise IRValidationError("an IO asset cannot have tokenizer and processor ownership")
        object.__setattr__(
            self,
            "_generation_defaults_json",
            canonical_json(
                _json_object(self._generation_defaults_json, field="generation defaults")
            ),
        )
        missing = _strings(
            self.missing_requirements, field="missing_requirements", sorted_unique=True
        )
        object.__setattr__(self, "missing_requirements", missing)
        if type(self.portable) is not bool:
            raise TypeError("portable must be a boolean")
        if self.portable and missing:
            raise IRValidationError("portable IOIR cannot have missing requirements")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("IO IR fingerprint does not match its payload")

    @property
    def generation_defaults(self) -> dict[str, Any]:
        return _json_object(self._generation_defaults_json, field="generation defaults")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "adapter_fingerprint": self.adapter_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "text_spaces": [item.as_dict() for item in self.text_spaces],
            "row_mappers": [item.as_dict() for item in self.row_mappers],
            "special_tokens": [item.as_dict() for item in self.special_tokens],
            "tokenizer_assets": [item.as_dict() for item in self.tokenizer_assets],
            "processor_assets": [item.as_dict() for item in self.processor_assets],
            "chat_templates": [item.as_dict() for item in self.chat_templates],
            "generation_defaults": self.generation_defaults,
            "output_spaces": [item.as_dict() for item in self.output_spaces],
            "missing_requirements": list(self.missing_requirements),
            "portable": self.portable,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **kwargs: Any) -> IOIR:
        payload = {
            "schema_version": IO_IR_SCHEMA,
            "source_fingerprint": kwargs["source_fingerprint"],
            "adapter_fingerprint": kwargs["adapter_fingerprint"],
            "model_fingerprint": kwargs["model_fingerprint"],
            "text_spaces": [item.as_dict() for item in kwargs["text_spaces"]],
            "row_mappers": [item.as_dict() for item in kwargs["row_mappers"]],
            "special_tokens": [item.as_dict() for item in kwargs["special_tokens"]],
            "tokenizer_assets": [item.as_dict() for item in kwargs["tokenizer_assets"]],
            "processor_assets": [item.as_dict() for item in kwargs["processor_assets"]],
            "chat_templates": [item.as_dict() for item in kwargs["chat_templates"]],
            "generation_defaults": _json_object(
                kwargs["_generation_defaults_json"], field="generation defaults"
            ),
            "output_spaces": [item.as_dict() for item in kwargs["output_spaces"]],
            "missing_requirements": list(kwargs["missing_requirements"]),
            "portable": kwargs["portable"],
        }
        return cls(**kwargs, fingerprint=canonical_sha256(payload))

    @classmethod
    def from_dict(cls, payload: Any) -> IOIR:
        value = require_dict(payload, field="IO IR")
        require_exact_keys(
            value,
            {
                "schema_version",
                "source_fingerprint",
                "adapter_fingerprint",
                "model_fingerprint",
                "text_spaces",
                "row_mappers",
                "special_tokens",
                "tokenizer_assets",
                "processor_assets",
                "chat_templates",
                "generation_defaults",
                "output_spaces",
                "missing_requirements",
                "portable",
                "fingerprint",
            },
            field="IO IR",
        )
        return cls(
            schema_version=require_str(value["schema_version"], field="IO schema"),
            source_fingerprint=require_str(value["source_fingerprint"], field="source_fingerprint"),
            adapter_fingerprint=require_str(
                value["adapter_fingerprint"], field="adapter_fingerprint"
            ),
            model_fingerprint=require_str(value["model_fingerprint"], field="model_fingerprint"),
            text_spaces=tuple(
                TokenSpaceIR.from_dict(item)
                for item in require_list(value["text_spaces"], field="text spaces")
            ),
            row_mappers=tuple(
                RowMapperIR.from_dict(item)
                for item in require_list(value["row_mappers"], field="row mappers")
            ),
            special_tokens=tuple(
                SpecialTokenIR.from_dict(item)
                for item in require_list(value["special_tokens"], field="special tokens")
            ),
            tokenizer_assets=tuple(
                BoundAssetIR.from_dict(item)
                for item in require_list(value["tokenizer_assets"], field="tokenizer assets")
            ),
            processor_assets=tuple(
                BoundAssetIR.from_dict(item)
                for item in require_list(value["processor_assets"], field="processor assets")
            ),
            chat_templates=tuple(
                ChatTemplateIR.from_dict(item)
                for item in require_list(value["chat_templates"], field="chat templates")
            ),
            _generation_defaults_json=canonical_json(
                require_dict(value["generation_defaults"], field="generation defaults")
            ),
            output_spaces=tuple(
                OutputSpaceIR.from_dict(item)
                for item in require_list(value["output_spaces"], field="output spaces")
            ),
            missing_requirements=_strings(
                value["missing_requirements"],
                field="missing_requirements",
                sorted_unique=True,
            ),
            portable=require_bool(value["portable"], field="portable"),
            fingerprint=require_str(value["fingerprint"], field="IO fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class IRBundle:
    physical_weights: PhysicalWeightIR
    model: ModelIR
    state: StateIR
    io: IOIR
    fingerprint: str
    schema_version: str = IR_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != IR_BUNDLE_SCHEMA:
            raise ValueError(f"unsupported IR bundle schema: {self.schema_version!r}")
        if not isinstance(self.physical_weights, PhysicalWeightIR):
            raise TypeError("physical_weights must be PhysicalWeightIR")
        if not isinstance(self.model, ModelIR) or not isinstance(self.state, StateIR):
            raise TypeError("model and state must be their corresponding IR types")
        if not isinstance(self.io, IOIR):
            raise TypeError("io must be IOIR")
        source_fingerprints = {
            self.physical_weights.source_fingerprint,
            self.model.source_fingerprint,
            self.state.source_fingerprint,
            self.io.source_fingerprint,
        }
        adapter_fingerprints = {
            self.physical_weights.adapter_fingerprint,
            self.model.adapter_fingerprint,
            self.state.adapter_fingerprint,
            self.io.adapter_fingerprint,
        }
        if len(source_fingerprints) != 1 or len(adapter_fingerprints) != 1:
            raise IRValidationError("all coordinated IRs must share source and adapter identity")
        if self.model.physical_weights_fingerprint != self.physical_weights.fingerprint:
            raise IRValidationError("ModelIR does not bind the supplied PhysicalWeightIR")
        if self.state.model_fingerprint != self.model.fingerprint:
            raise IRValidationError("StateIR does not bind the supplied ModelIR")
        if self.io.model_fingerprint != self.model.fingerprint:
            raise IRValidationError("IOIR does not bind the supplied ModelIR")
        physical_views = {item.view_id for item in self.physical_weights.views}
        model_views = {item.view_id for item in self.model.parameters}
        if physical_views != model_views:
            raise IRValidationError(
                "ModelIR parameters do not exactly cover physical logical views"
            )
        state_refs = {item.slot_id for item in self.state.slots}
        if set(self.model.state_refs) != state_refs:
            raise IRValidationError("ModelIR state references do not exactly cover StateIR slots")
        object.__setattr__(
            self,
            "fingerprint",
            require_sha256(self.fingerprint, field="IR fingerprint"),
        )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("IR bundle fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "physical_weights": self.physical_weights.as_dict(),
            "model": self.model.as_dict(),
            "state": self.state.as_dict(),
            "io": self.io.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(
        cls,
        *,
        physical_weights: PhysicalWeightIR,
        model: ModelIR,
        state: StateIR,
        io: IOIR,
    ) -> IRBundle:
        payload = {
            "schema_version": IR_BUNDLE_SCHEMA,
            "physical_weights": physical_weights.as_dict(),
            "model": model.as_dict(),
            "state": state.as_dict(),
            "io": io.as_dict(),
        }
        return cls(
            physical_weights=physical_weights,
            model=model,
            state=state,
            io=io,
            fingerprint=canonical_sha256(payload),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> IRBundle:
        value = require_dict(payload, field="IR bundle")
        require_exact_keys(
            value,
            {"schema_version", "physical_weights", "model", "state", "io", "fingerprint"},
            field="IR bundle",
        )
        return cls(
            schema_version=require_str(value["schema_version"], field="IR bundle schema"),
            physical_weights=PhysicalWeightIR.from_dict(value["physical_weights"]),
            model=ModelIR.from_dict(value["model"]),
            state=StateIR.from_dict(value["state"]),
            io=IOIR.from_dict(value["io"]),
            fingerprint=require_str(value["fingerprint"], field="IR bundle fingerprint"),
        )
