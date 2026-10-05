"""Deterministic backend schedule lowering for WorkPlan v3.

Lowering describes legal work and evidence requirements. It does not claim that a CUDA
or HIP graph has been captured, or that Core ML placed work on the Neural Engine.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .ir import (
    DISPATCH_BINDING_SCHEMA,
    DenseWorkPlan,
    ExecutionMode,
    OutputContract,
    WorkTemplate,
    _is_sha256_digest,
    _require_exact_keys,
    _strict_bool,
    _strict_json_loads,
    _strict_string,
    _strict_string_list,
)

LOWERED_WORK_TEMPLATE_SCHEMA = "mrun-lowered-work-template-v1"
LOWERING_ABI = "mrun-work-template-lowering-abi-v1"
LOWERED_TEMPLATE_LOOKUP_NAMESPACE = "mrun-lowered-template-lookup-v1"


@dataclass(frozen=True)
class _FrozenJsonObject:
    items: tuple[tuple[str, Any], ...]


def _freeze_param(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} cannot contain non-finite floats")
        return value
    if isinstance(value, _FrozenJsonObject):
        return value
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for raw_key, raw_value in value.items():
            if type(raw_key) is not str or not raw_key or raw_key.strip() != raw_key:
                raise ValueError(f"{path} keys must be canonical non-empty strings")
            items.append((raw_key, _freeze_param(raw_value, path=f"{path}.{raw_key}")))
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{path} contains duplicate keys")
        return _FrozenJsonObject(tuple(sorted(items)))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_param(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible canonical values")


def _thaw_param(value: Any) -> Any:
    if isinstance(value, _FrozenJsonObject):
        return {key: _thaw_param(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [_thaw_param(item) for item in value]
    return value


def _canonical_params(
    values: Mapping[str, Any] | Sequence[tuple[str, Any]],
) -> tuple[tuple[str, Any], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized: list[tuple[str, Any]] = []
    for raw_key, raw_value in items:
        if type(raw_key) is not str or not raw_key or raw_key.strip() != raw_key:
            raise ValueError("schedule parameter names must be canonical non-empty strings")
        normalized.append((raw_key, _freeze_param(raw_value, path=f"params.{raw_key}")))
    keys = [key for key, _ in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("schedule parameters must have unique names")
    return tuple(sorted(normalized))


def _params(**values: Any) -> tuple[tuple[str, Any], ...]:
    return _canonical_params(values)


@dataclass(frozen=True)
class ScheduleStep:
    target: str
    operation: str
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("target", "operation"):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"schedule step {field_name} must be a canonical string")
        object.__setattr__(self, "params", _canonical_params(self.params))

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "operation": self.operation,
            "params": {key: _thaw_param(value) for key, value in self.params},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScheduleStep:
        _require_exact_keys(payload, {"target", "operation", "params"}, field="ScheduleStep")
        params = payload.get("params")
        if type(params) is not dict:
            raise TypeError("schedule step params must be an object")
        return cls(
            target=_strict_string(payload["target"], field="ScheduleStep.target"),
            operation=_strict_string(payload["operation"], field="ScheduleStep.operation"),
            params=_canonical_params(params),
        )


@dataclass(frozen=True)
class LoweredWorkPlan:
    backend: str
    plan_fingerprint: str
    executable_key: str
    implementation_status: str
    reported_fabric: str
    placement_verified: bool
    content_identity_verified: bool
    capture_requested: bool
    capture_ready: bool
    capture_executed: bool
    capture_refusal_reasons: tuple[str, ...]
    steps: tuple[ScheduleStep, ...]
    evidence_requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "backend",
            "plan_fingerprint",
            "executable_key",
            "implementation_status",
            "reported_fabric",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"lowered WorkPlan {field_name} must be canonical")
        for field_name in (
            "placement_verified",
            "content_identity_verified",
            "capture_requested",
            "capture_ready",
            "capture_executed",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"lowered WorkPlan {field_name} must be boolean")
        if not _is_sha256_digest(self.plan_fingerprint):
            raise ValueError("lowered WorkPlan requires a canonical plan fingerprint")
        if not _is_sha256_digest(self.executable_key):
            raise ValueError("lowered WorkPlan requires a canonical executable key")
        if not isinstance(self.steps, (tuple, list)) or any(
            not isinstance(step, ScheduleStep) for step in self.steps
        ):
            raise TypeError("lowered WorkPlan steps must be ScheduleStep values")
        object.__setattr__(self, "steps", tuple(self.steps))
        for field_name in ("capture_refusal_reasons", "evidence_requirements"):
            values = getattr(self, field_name)
            if not isinstance(values, (tuple, list)) or any(
                type(value) is not str for value in values
            ):
                raise TypeError(f"lowered WorkPlan {field_name} must contain only strings")
            object.__setattr__(self, field_name, tuple(values))
        if self.capture_executed:
            raise ValueError("serialized lowered WorkPlans cannot claim capture execution")

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "plan_fingerprint": self.plan_fingerprint,
            "executable_key": self.executable_key,
            "implementation_status": self.implementation_status,
            "reported_fabric": self.reported_fabric,
            "placement_verified": self.placement_verified,
            "content_identity_verified": self.content_identity_verified,
            "capture_requested": self.capture_requested,
            "capture_ready": self.capture_ready,
            "capture_executed": self.capture_executed,
            "capture_refusal_reasons": list(self.capture_refusal_reasons),
            "steps": [step.as_dict() for step in self.steps],
            "evidence_requirements": list(self.evidence_requirements),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LoweredWorkPlan:
        _require_exact_keys(
            payload,
            {
                "backend",
                "plan_fingerprint",
                "executable_key",
                "implementation_status",
                "reported_fabric",
                "placement_verified",
                "content_identity_verified",
                "capture_requested",
                "capture_ready",
                "capture_executed",
                "capture_refusal_reasons",
                "steps",
                "evidence_requirements",
            },
            field="LoweredWorkPlan",
        )
        steps = payload.get("steps")
        if type(steps) is not list:
            raise TypeError("lowered steps must be a list")
        if any(type(step) is not dict for step in steps):
            raise TypeError("lowered steps must contain only objects")
        return cls(
            backend=_strict_string(payload["backend"], field="LoweredWorkPlan.backend"),
            plan_fingerprint=_strict_string(
                payload["plan_fingerprint"], field="LoweredWorkPlan.plan_fingerprint"
            ),
            executable_key=_strict_string(
                payload["executable_key"], field="LoweredWorkPlan.executable_key"
            ),
            implementation_status=_strict_string(
                payload["implementation_status"],
                field="LoweredWorkPlan.implementation_status",
            ),
            reported_fabric=_strict_string(
                payload["reported_fabric"], field="LoweredWorkPlan.reported_fabric"
            ),
            placement_verified=_strict_bool(
                payload["placement_verified"], field="LoweredWorkPlan.placement_verified"
            ),
            content_identity_verified=_strict_bool(
                payload["content_identity_verified"],
                field="LoweredWorkPlan.content_identity_verified",
            ),
            capture_requested=_strict_bool(
                payload["capture_requested"], field="LoweredWorkPlan.capture_requested"
            ),
            capture_ready=_strict_bool(
                payload["capture_ready"], field="LoweredWorkPlan.capture_ready"
            ),
            capture_executed=_strict_bool(
                payload["capture_executed"], field="LoweredWorkPlan.capture_executed"
            ),
            capture_refusal_reasons=_strict_string_list(
                payload["capture_refusal_reasons"],
                field="LoweredWorkPlan.capture_refusal_reasons",
            ),
            steps=tuple(ScheduleStep.from_dict(step) for step in steps),
            evidence_requirements=_strict_string_list(
                payload["evidence_requirements"],
                field="LoweredWorkPlan.evidence_requirements",
            ),
        )

    @property
    def estimated_bytes(self) -> int:
        return len(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _lowered_template_lookup_key(
    backend: str,
    template_fingerprint: str,
    lowering_abi: str = LOWERING_ABI,
) -> str:
    return _canonical_sha256(
        {
            "namespace": LOWERED_TEMPLATE_LOOKUP_NAMESPACE,
            "lowering_abi": lowering_abi,
            "backend": backend,
            "template_fingerprint": template_fingerprint,
        }
    )


@dataclass(frozen=True)
class LoweredWorkTemplate:
    """A backend schedule with no request IDs, state handles, or vocabulary values."""

    backend: str
    template_fingerprint: str
    executable_key: str
    implementation_status: str
    reported_fabric: str
    placement_verified: bool
    content_identity_verified: bool
    capture_requested: bool
    capture_ready: bool
    capture_executed: bool
    capture_refusal_reasons: tuple[str, ...]
    steps: tuple[ScheduleStep, ...]
    evidence_requirements: tuple[str, ...]
    artifact_sha256: str
    lowering_abi: str = LOWERING_ABI
    schema_version: str = LOWERED_WORK_TEMPLATE_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("capture_refusal_reasons", "evidence_requirements"):
            raw_values = getattr(self, field_name)
            if not isinstance(raw_values, (tuple, list)) or any(
                type(value) is not str for value in raw_values
            ):
                raise TypeError(f"lowered template {field_name} must contain only strings")
            object.__setattr__(self, field_name, tuple(raw_values))
        if not isinstance(self.steps, (tuple, list)) or any(
            not isinstance(step, ScheduleStep) for step in self.steps
        ):
            raise TypeError("lowered-template steps must be ScheduleStep values")
        object.__setattr__(self, "steps", tuple(self.steps))
        self.verify_integrity()

    def _artifact_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lowering_abi": self.lowering_abi,
            "backend": self.backend,
            "template_fingerprint": self.template_fingerprint,
            "executable_key": self.executable_key,
            "implementation_status": self.implementation_status,
            "reported_fabric": self.reported_fabric,
            "placement_verified": self.placement_verified,
            "content_identity_verified": self.content_identity_verified,
            "capture_requested": self.capture_requested,
            "capture_ready": self.capture_ready,
            "capture_executed": self.capture_executed,
            "capture_refusal_reasons": list(self.capture_refusal_reasons),
            "steps": [step.as_dict() for step in self.steps],
            "evidence_requirements": list(self.evidence_requirements),
        }

    def _computed_artifact_sha256(self) -> str:
        return _canonical_sha256(self._artifact_payload())

    def verify_integrity(self) -> None:
        """Revalidate lookup identity and the digest of every lowered artifact field."""

        for field_name in ("capture_refusal_reasons", "evidence_requirements"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(type(value) is not str for value in values):
                raise TypeError(f"lowered template {field_name} is not canonical")
        if type(self.steps) is not tuple or any(
            not isinstance(step, ScheduleStep) for step in self.steps
        ):
            raise TypeError("lowered-template steps are not canonical")
        if self.schema_version != LOWERED_WORK_TEMPLATE_SCHEMA:
            raise ValueError(f"unsupported lowered-template schema: {self.schema_version}")
        if self.lowering_abi != LOWERING_ABI:
            raise ValueError(f"unsupported lowered-template ABI: {self.lowering_abi}")
        for field_name in (
            "backend",
            "template_fingerprint",
            "executable_key",
            "implementation_status",
            "reported_fabric",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"lowered template {field_name} must be canonical")
        for field_name in (
            "placement_verified",
            "content_identity_verified",
            "capture_requested",
            "capture_ready",
            "capture_executed",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"lowered template {field_name} must be boolean")
        if not _is_sha256_digest(self.template_fingerprint):
            raise ValueError("lowered template requires a canonical template fingerprint")
        expected_key = _lowered_template_lookup_key(
            self.backend,
            self.template_fingerprint,
            self.lowering_abi,
        )
        if self.executable_key != expected_key:
            raise ValueError("lowered-template lookup key does not match its ABI identity")
        if self.capture_executed:
            raise ValueError("serialized lowered templates cannot claim runtime capture execution")
        if not _is_sha256_digest(self.artifact_sha256):
            raise ValueError("lowered template requires a canonical artifact digest")
        if self.artifact_sha256 != self._computed_artifact_sha256():
            raise ValueError("lowered-template artifact digest mismatch")

    def bind(self, plan: DenseWorkPlan) -> LoweredWorkPlan:
        """Attach concrete provenance after proving structural compatibility."""

        self.verify_integrity()
        template = WorkTemplate.from_plan(plan)
        if template.fingerprint != self.template_fingerprint:
            raise ValueError("WorkPlan does not match the lowered WorkTemplate")
        if self.content_identity_verified != plan.content_identity_verified:
            raise ValueError("lowered-template content identity verdict does not match WorkPlan")
        if self.capture_requested != plan.capture.requested:
            raise ValueError("lowered-template capture contract does not match WorkPlan")
        return LoweredWorkPlan(
            backend=self.backend,
            plan_fingerprint=plan.fingerprint,
            executable_key=self.executable_key,
            implementation_status=self.implementation_status,
            reported_fabric=self.reported_fabric,
            placement_verified=self.placement_verified,
            content_identity_verified=self.content_identity_verified,
            capture_requested=self.capture_requested,
            capture_ready=self.capture_ready,
            capture_executed=False,
            capture_refusal_reasons=self.capture_refusal_reasons,
            steps=self.steps,
            evidence_requirements=self.evidence_requirements,
        )

    def as_dict(self) -> dict[str, Any]:
        self.verify_integrity()
        return {**self._artifact_payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LoweredWorkTemplate:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "lowering_abi",
                "artifact_sha256",
                "backend",
                "template_fingerprint",
                "executable_key",
                "implementation_status",
                "reported_fabric",
                "placement_verified",
                "content_identity_verified",
                "capture_requested",
                "capture_ready",
                "capture_executed",
                "capture_refusal_reasons",
                "steps",
                "evidence_requirements",
            },
            field="LoweredWorkTemplate",
        )
        steps = payload.get("steps")
        if type(steps) is not list:
            raise TypeError("lowered-template steps must be a list")
        if any(type(step) is not dict for step in steps):
            raise TypeError("lowered-template steps must contain only objects")
        return cls(
            backend=_strict_string(payload["backend"], field="LoweredWorkTemplate.backend"),
            template_fingerprint=_strict_string(
                payload["template_fingerprint"],
                field="LoweredWorkTemplate.template_fingerprint",
            ),
            executable_key=_strict_string(
                payload["executable_key"], field="LoweredWorkTemplate.executable_key"
            ),
            implementation_status=_strict_string(
                payload["implementation_status"],
                field="LoweredWorkTemplate.implementation_status",
            ),
            reported_fabric=_strict_string(
                payload["reported_fabric"], field="LoweredWorkTemplate.reported_fabric"
            ),
            placement_verified=_strict_bool(
                payload["placement_verified"],
                field="LoweredWorkTemplate.placement_verified",
            ),
            content_identity_verified=_strict_bool(
                payload["content_identity_verified"],
                field="LoweredWorkTemplate.content_identity_verified",
            ),
            capture_requested=_strict_bool(
                payload["capture_requested"], field="LoweredWorkTemplate.capture_requested"
            ),
            capture_ready=_strict_bool(
                payload["capture_ready"], field="LoweredWorkTemplate.capture_ready"
            ),
            capture_executed=_strict_bool(
                payload["capture_executed"], field="LoweredWorkTemplate.capture_executed"
            ),
            capture_refusal_reasons=_strict_string_list(
                payload["capture_refusal_reasons"],
                field="LoweredWorkTemplate.capture_refusal_reasons",
            ),
            steps=tuple(ScheduleStep.from_dict(step) for step in steps),
            evidence_requirements=_strict_string_list(
                payload["evidence_requirements"],
                field="LoweredWorkTemplate.evidence_requirements",
            ),
            lowering_abi=_strict_string(
                payload["lowering_abi"], field="LoweredWorkTemplate.lowering_abi"
            ),
            artifact_sha256=_strict_string(
                payload["artifact_sha256"], field="LoweredWorkTemplate.artifact_sha256"
            ),
            schema_version=_strict_string(
                payload["schema_version"], field="LoweredWorkTemplate.schema_version"
            ),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> LoweredWorkTemplate:
        return cls.from_dict(_strict_json_loads(payload, field="LoweredWorkTemplate"))

    @property
    def estimated_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))


def _executable_key(plan: DenseWorkPlan, backend: str) -> str:
    # LoweredWorkPlan currently retains the concrete plan fingerprint (and stateful
    # schedules also embed exact KV bindings). A narrower reusable key would return an
    # artifact that execution must reject as belonging to another plan. Use full provenance
    # until lowering is split into a binding-free template plus per-plan attachment.
    payload = f"{backend}:{plan.fingerprint}".encode()
    return hashlib.sha256(payload).hexdigest()


def _template_executable_key(template: WorkTemplate, backend: str) -> str:
    return _lowered_template_lookup_key(backend, template.fingerprint)


def _head_operation(plan: DenseWorkPlan) -> str:
    return {
        OutputContract.FULL_LOGITS: "full_vocabulary_head",
        OutputContract.LAST_TOKEN_LOGITS: "last_token_full_vocabulary_head",
        OutputContract.SELECTED_TOKEN_ROWS: "selected_vocabulary_rows",
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN: "candidate_rows_argmax_margin",
        OutputContract.LOSS_ONLY: "loss_only_head",
        OutputContract.HIDDEN_STATE_ONLY: "hidden_state_output",
        OutputContract.SELECTED_CAPTURE: "selected_capture_output",
    }[plan.output_contract]


def _canonical_backend(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    if normalized in {"paged", "paged-qstore", "cpu-paged"}:
        return "paged-qstore"
    if normalized in {"cuda", "cuda-qstore", "dense-qstore-cuda", "dense-cuda"}:
        return "cuda-qstore"
    if normalized in {"rocm", "hip"}:
        return "rocm"
    if normalized in {"coreml", "ane"}:
        return "coreml"
    raise ValueError("backend must be one of: paged-qstore, cuda-qstore, rocm, coreml")


def _typed_capture_params(plan: DenseWorkPlan, backend: str) -> dict[str, Any]:
    if plan.output_contract is not OutputContract.SELECTED_CAPTURE:
        return {"output_contract": plan.output_contract.value}
    capability = plan.capture_capability
    if capability is None:
        raise ValueError("selected_capture lowering requires an exact capture capability")
    if _canonical_backend(capability.backend) != backend:
        raise ValueError(
            "selected_capture capability backend does not match the requested lowerer "
            f"({capability.backend!r} != {backend!r})"
        )
    capability.validate_specs(plan.capture_specs)
    return {
        "output_contract": plan.output_contract.value,
        "runtime_entrypoint": capability.runtime_entrypoint,
        "capability_fingerprint": capability.fingerprint,
        "capture_spec_fingerprints": [spec.fingerprint for spec in plan.capture_specs],
        "capture_count": len(plan.capture_specs),
        "aggregate_device_bytes": sum(
            spec.resource_estimate.device_bytes for spec in plan.capture_specs
        ),
        "aggregate_retained_artifact_bytes": sum(
            spec.resource_estimate.retained_artifact_bytes for spec in plan.capture_specs
        ),
        "aggregate_retained_byte_cap": sum(spec.max_retained_bytes for spec in plan.capture_specs),
        "retention_policies": sorted({spec.retention.value for spec in plan.capture_specs}),
        "payload_storage": "external-content-addressed-runtime-inputs-only",
    }


def _typed_capture_evidence(plan: DenseWorkPlan) -> tuple[str, ...]:
    if plan.output_contract is not OutputContract.SELECTED_CAPTURE:
        return ()
    return (
        "typed-capture-capability-bound",
        "capture-row-identity-preserved",
        "capture-on-device-reduction",
        "capture-retained-byte-cap-enforced",
        "capture-payloads-external-to-workplan",
    )


def _capture_fields(
    plan: DenseWorkPlan,
    *,
    executor_available: bool,
) -> tuple[bool, tuple[str, ...]]:
    reasons = list(plan.capture.refusal_reasons)
    if not executor_available:
        reasons.append("backend graph executor is not implemented")
    if plan.execution_mode is not ExecutionMode.SCORE:
        reasons.append("CUDA Graph runtime supports only stateless score mode")
    if plan.output_contract is not OutputContract.SELECTED_TOKEN_ROWS:
        reasons.append("CUDA Graph runtime supports only selected token rows")
    if plan.prefix_state_ids or plan.kv_read_handles or plan.kv_write_handles:
        reasons.append("CUDA Graph runtime does not support bound state or KV handles")
    if not plan.content_identity_verified:
        reasons.append("CUDA Graph runtime requires a loaded-store identity certificate")
    metadata = dict(plan.metadata)
    if metadata.get("engine_backend") != "dense-qstore-cuda":
        reasons.append("CUDA Graph runtime requires the dense CUDA QStore backend")
    if str(metadata.get("engine_device", "")).split(":", 1)[0] != "cuda":
        reasons.append("CUDA Graph runtime requires CUDA placement")
    if metadata.get("head_output_pushdown") is not True:
        reasons.append("CUDA Graph runtime requires selected-head output pushdown")
    ready = plan.capture.eligible and executor_available and not reasons
    return ready, tuple(dict.fromkeys(reasons))


def _lower_cuda_qstore(plan: DenseWorkPlan) -> LoweredWorkPlan:
    if plan.execution_mode is not ExecutionMode.SCORE:
        raise NotImplementedError(
            "dense CUDA stateful WorkPlans are unavailable until the engine exposes the "
            "versioned provisional-delta executor"
        )
    if plan.precision.weight_dtype != "int8":
        raise ValueError("the current CUDA QStore lowerer requires int8 weights")
    if plan.precision.activation_dtype not in {"fp16", "bf16"}:
        raise ValueError("the current CUDA QStore lowerer requires fp16 or bf16 activations")
    capture_ready, capture_reasons = _capture_fields(plan, executor_available=True)
    head_output_pushdown = bool(dict(plan.metadata).get("head_output_pushdown", False))
    steps = (
        ScheduleStep(
            "host",
            "validate_content_fingerprints",
            _params(
                model_revision=plan.model_revision,
                store_fingerprint=plan.store_fingerprint,
            ),
        ),
        ScheduleStep(
            "cuda",
            "bind_compact_qstore_pages",
            _params(
                cache_admission=plan.cache_admission,
                page_count=len(plan.page_sequence),
            ),
        ),
        ScheduleStep(
            "cuda",
            "allocate_or_reuse_persistent_kv",
            _params(
                batch_bucket=plan.shape.batch_bucket,
                sequence_bucket=plan.shape.sequence_bucket,
            ),
        ),
        ScheduleStep(
            "cuda",
            "dense_transformer_region",
            _params(
                activation_dtype=plan.precision.activation_dtype,
                accumulator_dtype=plan.precision.accumulator_dtype,
                layout_ids=list(plan.compute_layout_ids),
            ),
        ),
        ScheduleStep(
            "cuda", _head_operation(plan), _params(**_typed_capture_params(plan, "cuda-qstore"))
        ),
    )
    return LoweredWorkPlan(
        backend="cuda-qstore",
        plan_fingerprint=plan.fingerprint,
        executable_key=_executable_key(plan, "cuda-qstore"),
        implementation_status="eager-adapter",
        reported_fabric="cuda",
        placement_verified=True,
        content_identity_verified=plan.content_identity_verified,
        capture_requested=plan.capture.requested,
        capture_ready=capture_ready,
        capture_executed=False,
        capture_refusal_reasons=capture_reasons,
        steps=steps,
        evidence_requirements=(
            "content-addressed-model-and-store",
            "same-qstore-eager-parity",
            *(("full-head-output-contract-parity",) if head_output_pushdown else ()),
            "canonical-hf-quality",
            "equal-batch-baseline",
            *_typed_capture_evidence(plan),
            *(
                (
                    "cuda-graph-eager-replay-parity",
                    "cuda-graph-capture-executed",
                )
                if plan.capture.requested
                else ()
            ),
        ),
    )


def _lower_paged_qstore(plan: DenseWorkPlan) -> LoweredWorkPlan:
    if plan.precision.weight_dtype not in {"int8", "int4", "int3", "int2"}:
        raise ValueError("the paged QStore lowerer requires int2/int3/int4/int8 weights")
    if plan.precision.activation_dtype not in {"fp16", "bf16", "fp32"}:
        raise ValueError("the paged QStore lowerer requires a floating-point activation type")
    capture_ready, capture_reasons = _capture_fields(plan, executor_available=False)
    metadata = dict(plan.metadata)
    fabric = str(metadata.get("engine_device", "cpu"))
    head_output_pushdown = bool(metadata.get("head_output_pushdown", False))
    stateful = plan.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}
    steps = [
        ScheduleStep(
            "host",
            "validate_content_fingerprints",
            _params(
                model_revision=plan.model_revision,
                store_fingerprint=plan.store_fingerprint,
            ),
        ),
        ScheduleStep(
            fabric,
            "bind_memory_mapped_qstore_pages",
            _params(
                cache_admission=plan.cache_admission,
                page_count=len(plan.page_sequence),
                weight_dtype=plan.precision.weight_dtype,
            ),
        ),
    ]
    if stateful:
        steps.append(
            ScheduleStep(
                fabric,
                "bind_versioned_kv_state",
                _params(
                    adapter_abi="mrun-paged-scratch-only-v1",
                    cache_identity_source="VersionedKVStateBinding.cache_id",
                    capacity=int(metadata["kv_capacity"]),
                    identity_fields=["cache_id", "storage_signature", "epoch", "lengths"],
                    read_handles=list(plan.kv_read_handles),
                    request_slots=list(plan.request_slots),
                    write_handles=list(plan.kv_write_handles),
                ),
            )
        )
    steps.extend(
        [
            ScheduleStep(
                fabric,
                "fused_batch_paged_transformer_region",
                _params(
                    actual_batch=plan.shape.actual_batch,
                    sequence_length=plan.shape.sequence_length,
                    activation_dtype=plan.precision.activation_dtype,
                    accumulator_dtype=plan.precision.accumulator_dtype,
                ),
            ),
            ScheduleStep(
                fabric,
                _head_operation(plan),
                _params(
                    **_typed_capture_params(plan, "paged-qstore"),
                    logical_head_access=metadata.get("logical_head_access", "unknown"),
                    logical_head_row_count=int(metadata.get("logical_head_row_count", 0)),
                ),
            ),
        ]
    )
    if stateful:
        steps.append(
            ScheduleStep(
                fabric,
                "emit_provisional_kv_delta",
                _params(
                    commit_policy="explicit-only",
                    parent_version_fields=[
                        "cache_id",
                        "storage_signature",
                        "epoch",
                        "lengths",
                    ],
                    token_count=plan.shape.sequence_length,
                ),
            )
        )
    steps.append(ScheduleStep("host", "record_max_dequantized_block_evidence"))
    return LoweredWorkPlan(
        backend="paged-qstore",
        plan_fingerprint=plan.fingerprint,
        executable_key=_executable_key(plan, "paged-qstore"),
        implementation_status="eager-adapter",
        reported_fabric=fabric,
        placement_verified=True,
        content_identity_verified=plan.content_identity_verified,
        capture_requested=plan.capture.requested,
        capture_ready=capture_ready,
        capture_executed=False,
        capture_refusal_reasons=capture_reasons,
        steps=tuple(steps),
        evidence_requirements=(
            "content-addressed-model-and-store",
            "same-qstore-direct-parity",
            *(("full-head-output-contract-parity",) if head_output_pushdown else ()),
            "canonical-hf-quality",
            "equal-batch-baseline",
            "observed-max-dequantized-block",
            *_typed_capture_evidence(plan),
            *(("versioned-kv-state-binding",) if stateful else ()),
            *(("provisional-kv-no-auto-commit",) if stateful else ()),
        ),
    )


def _lower_rocm(plan: DenseWorkPlan) -> LoweredWorkPlan:
    if plan.precision.activation_dtype not in {"fp16", "bf16"}:
        raise ValueError("the ROCm candidate lowerer requires fp16 or bf16 activations")
    capture_ready, capture_reasons = _capture_fields(plan, executor_available=False)
    steps = (
        ScheduleStep("host", "validate_content_fingerprints"),
        ScheduleStep(
            "rocm",
            "bind_compact_weight_pages",
            _params(weight_dtype=plan.precision.weight_dtype),
        ),
        ScheduleStep(
            "rocm",
            "hipblaslt_or_triton_transformer_region",
            _params(
                activation_dtype=plan.precision.activation_dtype,
                accumulator_dtype=plan.precision.accumulator_dtype,
            ),
        ),
        ScheduleStep("rocm", _head_operation(plan)),
        ScheduleStep("host", "record_rocm_device_and_runtime_evidence"),
    )
    return LoweredWorkPlan(
        backend="rocm",
        plan_fingerprint=plan.fingerprint,
        executable_key=_executable_key(plan, "rocm"),
        implementation_status="schedule-only",
        reported_fabric="rocm-unmeasured",
        placement_verified=False,
        content_identity_verified=plan.content_identity_verified,
        capture_requested=plan.capture.requested,
        capture_ready=capture_ready,
        capture_executed=False,
        capture_refusal_reasons=capture_reasons,
        steps=steps,
        evidence_requirements=(
            "content-addressed-model-and-store",
            "amd-device-capability",
            "dense-projection-parity",
            "complete-block-parity",
            "hip-graph-eager-replay-parity",
            "equal-batch-baseline",
        ),
    )


def _lower_coreml(plan: DenseWorkPlan) -> LoweredWorkPlan:
    if plan.precision.activation_dtype != "fp16" or plan.precision.weight_dtype != "fp16":
        raise ValueError("the Core ML candidate lowerer requires fp16 activations and weights")
    if plan.shape.sequence_bucket != plan.shape.sequence_length:
        raise ValueError("Core ML exact-mask plans require an exact sequence bucket")
    steps = (
        ScheduleStep(
            "host",
            "compile_exact_shape_mlprogram",
            _params(
                batch_bucket=plan.shape.batch_bucket,
                sequence_length=plan.shape.sequence_length,
                weights_baked=True,
                mask_baked=True,
            ),
        ),
        ScheduleStep(
            "coreml",
            "predict",
            _params(
                compute_units_request="ALL",
                output_contract=plan.output_contract.value,
            ),
        ),
        ScheduleStep("host", "inspect_mlcomputeplan_preferred_devices"),
        ScheduleStep("host", "record_compile_first_run_and_warm_run"),
    )
    reasons = tuple(
        dict.fromkeys((*plan.capture.refusal_reasons, "Core ML uses compiled artifact dispatch"))
    )
    return LoweredWorkPlan(
        backend="coreml",
        plan_fingerprint=plan.fingerprint,
        executable_key=_executable_key(plan, "coreml"),
        implementation_status="schedule-only",
        reported_fabric="coreml-unverified",
        placement_verified=False,
        content_identity_verified=plan.content_identity_verified,
        capture_requested=plan.capture.requested,
        capture_ready=False,
        capture_executed=False,
        capture_refusal_reasons=reasons,
        steps=steps,
        evidence_requirements=(
            "content-addressed-model-and-store",
            "coreml-output-parity",
            "mlcomputeplan-placement",
            "cpu-only-vs-cpu-and-ne-vs-all",
            "compile-first-run-warm-run",
        ),
    )


def lower_work_plan(plan: DenseWorkPlan, backend: str) -> LoweredWorkPlan:
    """Lower a validated plan, refusing unsupported backend/precision combinations."""

    normalized = _canonical_backend(backend)
    if plan.output_contract is OutputContract.SELECTED_CAPTURE:
        _typed_capture_params(plan, normalized)
    if normalized == "paged-qstore":
        return _lower_paged_qstore(plan)
    if normalized == "cuda-qstore":
        return _lower_cuda_qstore(plan)
    if normalized == "rocm":
        return _lower_rocm(plan)
    if normalized == "coreml":
        return _lower_coreml(plan)
    raise AssertionError(f"unreachable canonical backend: {normalized}")


def _binding_free_steps(
    steps: tuple[ScheduleStep, ...],
    template: WorkTemplate,
) -> tuple[ScheduleStep, ...]:
    """Erase concrete state handles while retaining their structural dispatch contract."""

    result: list[ScheduleStep] = []
    for step in steps:
        if step.operation != "bind_versioned_kv_state":
            result.append(step)
            continue
        params = dict(step.params)
        params.pop("read_handles", None)
        params.pop("write_handles", None)
        params.pop("request_slots", None)
        params.update(
            {
                "binding_schema": DISPATCH_BINDING_SCHEMA,
                "handle_count": template.kv_read_handle_count,
                "request_slot_count": template.shape.actual_batch,
            }
        )
        result.append(ScheduleStep(step.target, step.operation, _params(**params)))
    binding_keys = {
        "request_ids",
        "request_slots",
        "prefix_state_ids",
        "kv_read_handles",
        "kv_write_handles",
        "read_handles",
        "write_handles",
        "required_output_rows",
        "candidate_token_ids",
        "output_token_ids",
    }
    for step in result:
        leaked = binding_keys.intersection(dict(step.params))
        if leaked:
            raise RuntimeError(f"lowered WorkTemplate retained dispatch fields: {sorted(leaked)!r}")
    return tuple(result)


def lower_work_template(template: WorkTemplate, backend: str) -> LoweredWorkTemplate:
    """Lower one structural template exactly once for all compatible bindings."""

    representative = template._materialize(  # noqa: SLF001 - compiler schema handshake
        template._representative_binding()  # noqa: SLF001
    )
    concrete = lower_work_plan(representative, backend)
    steps = _binding_free_steps(concrete.steps, template)
    payload = {
        "schema_version": LOWERED_WORK_TEMPLATE_SCHEMA,
        "lowering_abi": LOWERING_ABI,
        "backend": concrete.backend,
        "template_fingerprint": template.fingerprint,
        "executable_key": _template_executable_key(template, concrete.backend),
        "implementation_status": concrete.implementation_status,
        "reported_fabric": concrete.reported_fabric,
        "placement_verified": concrete.placement_verified,
        "content_identity_verified": concrete.content_identity_verified,
        "capture_requested": concrete.capture_requested,
        "capture_ready": concrete.capture_ready,
        "capture_executed": False,
        "capture_refusal_reasons": list(concrete.capture_refusal_reasons),
        "steps": [step.as_dict() for step in steps],
        "evidence_requirements": list(concrete.evidence_requirements),
    }
    return LoweredWorkTemplate.from_dict({**payload, "artifact_sha256": _canonical_sha256(payload)})
