"""Executable, fail-closed reference target for canonical source-component artifacts.

This is the first U3/G8 target for ``mrun-native-source-component-v1``.  It deliberately is not a
Metal or CUDA lowering: tensors are read directly from the artifact's immutable allocation blobs
and a small Torch/CPU interpreter executes the typed decoder IR.  Dense rotary decoders, the strict
routed-only Mixtral contract, and exact stateless full-sequence Mamba1 recurrence are supported.  A
separate source-schema runner wires the same verified allocations by their original checkpoint
names.  Comparing those two paths catches mapping, alias, topology, shape, state, routing, and
execution mistakes without reconstructing a Transformers model or importing repository code.

The promotion boundary is explicit:

* opening an artifact proves custody and structure only;
* lowering proves that the artifact is in this target's closed support set;
* G8 proves source-schema versus IR forward parity for the supplied cases;
* promotion is reference-only and never creates a production ``CompiledModelIdentity``.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ._json import canonical_sha256, require_sha256
from .emitter import COMPONENT_ARTIFACT_SCHEMA, ComponentArtifact, open_component_artifact
from .errors import DecompilerError
from .ir import ModelDimensionsIR, OperationIR, StateIR

REFERENCE_TARGET_SCHEMA = "mrun-artifact-reference-target-v1"
REFERENCE_CERTIFICATION_SCHEMA = "mrun-reference-execution-certification-v1"
MIXTRAL_SEMANTIC_CERTIFICATION_SCHEMA = "mrun-mixtral-semantic-certification-v1"
REFERENCE_PROMOTION_SCHEMA = "mrun-reference-promotion-v1"
REFERENCE_TARGET_ID = "mrun.torch-cpu.decoder-ir-reference"
REFERENCE_TARGET_VERSION = "1.2.0"

_ARCHITECTURES = {
    "gemma1-dense-causal-decoder": ("mrun.hf.gemma1-dense", "gemma"),
    "gpt2-causal-decoder": ("mrun.hf.gpt2", "gpt2"),
    "gpt-neox-pythia-causal-decoder": ("mrun.hf.gpt-neox-pythia", "gpt_neox"),
    "qwen2-dense-causal-decoder": ("mrun.hf.qwen2-dense", "qwen2"),
    "qwen3-dense-causal-decoder": ("mrun.hf.qwen3-dense", "qwen3"),
    "qwen3_5-hybrid-text-causal-decoder": (
        "mrun.hf.qwen3_5-hybrid-text",
        "qwen3_5",
    ),
    "llama-dense-causal-decoder": ("mrun.hf.llama-dense-baseline", "llama"),
    "mistral-dense-causal-decoder": ("mrun.hf.mistral-dense", "mistral"),
    "mixtral-sparse-moe-causal-decoder": ("mrun.hf.mixtral-sparse-moe", "mixtral"),
    "mamba1-selective-state-space-causal-decoder": ("mrun.hf.mamba1", "mamba"),
    "phi-causal-decoder": ("mrun.hf.phi", "phi"),
}
_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
_ALLOWED_OPERATION_KINDS = frozenset(
    {
        "token-embedding",
        "rms-norm",
        "head-rms-norm",
        "linear",
        "rotary-default",
        "causal-grouped-query-attention",
        "residual-add",
        "silu",
        "elementwise-multiply",
        "absolute-position-embedding-add",
        "conv1d-linear",
        "fused-qkv-split",
        "gelu-tanh",
        "gemma-rms-norm",
        "scalar-multiply",
        "gelu-erf",
        "gpt-neox-qkv-unpack",
        "layer-norm",
        "linear-readout",
        "rotary-partial-default",
        "moe-router-linear",
        "moe-top-k-softmax",
        "moe-token-dispatch",
        "moe-routed-expert-linear",
        "moe-weighted-scatter-add",
        "mamba-rms-norm",
        "mamba-input-gate-split",
        "mamba-causal-depthwise-convolution",
        "mamba-selection-split",
        "mamba-selective-scan",
        "qwen35-attention-output-gate",
        "qwen35-causal-depthwise-convolution",
        "qwen35-gated-delta-recurrence",
        "qwen35-head-rms-norm",
        "qwen35-linear-qkv-split",
        "qwen35-partial-interleaved-mrope",
        "qwen35-query-gate-split",
        "qwen35-rms-norm",
        "qwen35-rms-norm-gated",
    }
)


class ReferenceLoweringError(DecompilerError):
    """The artifact is outside the registered reference target's support set."""

    code = "reference_lowering_rejection"
    gate = "G8"


class ReferenceExecutionError(DecompilerError):
    """A lowered reference program rejected an input or failed execution."""

    code = "reference_execution_failure"
    gate = "G8"


class ReferenceParityError(DecompilerError):
    """Source-schema and IR executions did not agree."""

    code = "reference_parity_failure"
    gate = "G8"


class MixtralSemanticParityError(DecompilerError):
    """The routed-only Mixtral IR disagreed with the independent source semantics."""

    code = "mixtral_semantic_parity_failure"
    gate = "G13"


class NativeLoweringUnavailable(DecompilerError):
    """The reference target cannot be represented as a production native runtime."""

    code = "native_lowering_unavailable"
    gate = "G9"


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _finite_positive(value: Any, *, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0:
        raise ReferenceLoweringError(f"{field} must be a positive finite number")
    return float(value)


def _tensor_fingerprint(value: torch.Tensor) -> str:
    normalized = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = normalized.numpy().astype("<f4", copy=False).tobytes()
    return canonical_sha256(
        {
            "dtype": "F32-canonical-observation",
            "shape": list(normalized.shape),
            "data_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )


def _normalize_cases(
    cases: Sequence[Sequence[Sequence[int]] | torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if not cases:
        raise ReferenceExecutionError("G8 certification requires at least one token case")
    normalized: list[torch.Tensor] = []
    for case_index, raw in enumerate(cases):
        value = (
            raw.detach().to(device="cpu") if isinstance(raw, torch.Tensor) else torch.tensor(raw)
        )
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.numel() == 0:
            raise ReferenceExecutionError(
                "token cases must be non-empty rank-1 or rank-2 arrays",
                details={"case_index": case_index, "shape": list(value.shape)},
            )
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise ReferenceExecutionError(
                "token cases must contain integers", details={"case_index": case_index}
            )
        normalized.append(value.to(dtype=torch.int64).contiguous())
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ReferenceTargetIdentity:
    artifact_id: str
    manifest_sha256: str
    ir_bundle_fingerprint: str
    model_fingerprint: str
    architecture_id: str
    source_model_type: str
    stored_parameter_dtype: str
    operation_kinds: tuple[str, ...]
    state_mode: str
    execution_certified: bool
    production_runtime_eligible: bool
    native_lowering_status: str
    fingerprint: str
    schema_version: str = REFERENCE_TARGET_SCHEMA
    target_id: str = REFERENCE_TARGET_ID
    target_version: str = REFERENCE_TARGET_VERSION

    def __post_init__(self) -> None:
        for value, field in (
            (self.artifact_id, "artifact_id"),
            (self.manifest_sha256, "manifest_sha256"),
            (self.ir_bundle_fingerprint, "ir_bundle_fingerprint"),
            (self.model_fingerprint, "model_fingerprint"),
            (self.fingerprint, "reference target fingerprint"),
        ):
            require_sha256(value, field=field)
        if self.schema_version != REFERENCE_TARGET_SCHEMA:
            raise ValueError("unsupported reference target schema")
        if self.target_id != REFERENCE_TARGET_ID or self.target_version != REFERENCE_TARGET_VERSION:
            raise ValueError("unsupported reference target implementation")
        if self.execution_certified or self.production_runtime_eligible:
            raise ValueError("an unexecuted reference target cannot claim promotion or production")
        if (
            self.state_mode != "stateless-full-sequence"
            or self.native_lowering_status != "not-lowered"
        ):
            raise ValueError("reference target boundary is invalid")
        if self.operation_kinds != tuple(sorted(set(self.operation_kinds))):
            raise ValueError("operation kinds must be sorted and unique")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("reference target fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "target_version": self.target_version,
            "artifact_id": self.artifact_id,
            "manifest_sha256": self.manifest_sha256,
            "ir_bundle_fingerprint": self.ir_bundle_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "architecture_id": self.architecture_id,
            "source_model_type": self.source_model_type,
            "stored_parameter_dtype": self.stored_parameter_dtype,
            "operation_kinds": list(self.operation_kinds),
            "state_mode": self.state_mode,
            "execution_certified": self.execution_certified,
            "production_runtime_eligible": self.production_runtime_eligible,
            "native_lowering_status": self.native_lowering_status,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **kwargs: Any) -> ReferenceTargetIdentity:
        payload = {
            "schema_version": REFERENCE_TARGET_SCHEMA,
            "target_id": REFERENCE_TARGET_ID,
            "target_version": REFERENCE_TARGET_VERSION,
            **kwargs,
            "operation_kinds": list(kwargs["operation_kinds"]),
        }
        return cls(**kwargs, fingerprint=canonical_sha256(payload))


@dataclass(frozen=True, slots=True)
class ReferenceForwardResult:
    logits: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReferenceExecutionCertification:
    artifact_id: str
    target_fingerprint: str
    case_input_fingerprints: tuple[str, ...]
    source_logits_fingerprints: tuple[str, ...]
    ir_logits_fingerprints: tuple[str, ...]
    maximum_absolute_error: float
    maximum_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    checks: tuple[str, ...]
    fingerprint: str
    schema_version: str = REFERENCE_CERTIFICATION_SCHEMA
    status: str = "passed"
    scope: str = "reference-target-source-schema-forward-parity"
    execution_performed: bool = True
    execution_certified: bool = True
    production_runtime_eligible: bool = False
    native_lowering_status: str = "not-lowered"

    def __post_init__(self) -> None:
        for value, field in (
            (self.artifact_id, "artifact_id"),
            (self.target_fingerprint, "target_fingerprint"),
            (self.fingerprint, "reference certification fingerprint"),
        ):
            require_sha256(value, field=field)
        if self.schema_version != REFERENCE_CERTIFICATION_SCHEMA:
            raise ValueError("unsupported reference execution certification schema")
        if self.status != "passed" or self.scope != "reference-target-source-schema-forward-parity":
            raise ValueError("reference certification status or scope is invalid")
        if not self.execution_performed or not self.execution_certified:
            raise ValueError("a passing execution certification must record execution")
        if self.production_runtime_eligible or self.native_lowering_status != "not-lowered":
            raise ValueError("reference certification cannot claim native production readiness")
        count = len(self.case_input_fingerprints)
        if (
            not count
            or len(self.source_logits_fingerprints) != count
            or len(self.ir_logits_fingerprints) != count
        ):
            raise ValueError("reference certification case records are incomplete")
        for values in (
            self.case_input_fingerprints,
            self.source_logits_fingerprints,
            self.ir_logits_fingerprints,
        ):
            for value in values:
                require_sha256(value, field="case fingerprint")
        if self.source_logits_fingerprints != self.ir_logits_fingerprints:
            raise ValueError("a passing exact reference certification must bind equal observations")
        if self.checks != tuple(sorted(set(self.checks))) or not self.checks:
            raise ValueError("reference checks must be sorted and unique")
        for value in (
            self.maximum_absolute_error,
            self.maximum_relative_error,
            self.absolute_tolerance,
            self.relative_tolerance,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "reference error and tolerance values must be finite and non-negative"
                )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("reference certification fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "scope": self.scope,
            "artifact_id": self.artifact_id,
            "target_fingerprint": self.target_fingerprint,
            "case_input_fingerprints": list(self.case_input_fingerprints),
            "source_logits_fingerprints": list(self.source_logits_fingerprints),
            "ir_logits_fingerprints": list(self.ir_logits_fingerprints),
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "checks": list(self.checks),
            "execution_performed": self.execution_performed,
            "execution_certified": self.execution_certified,
            "production_runtime_eligible": self.production_runtime_eligible,
            "native_lowering_status": self.native_lowering_status,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class MixtralSemanticCertification:
    artifact_id: str
    target_fingerprint: str
    case_input_fingerprints: tuple[str, ...]
    observation_names: tuple[str, ...]
    source_observation_fingerprints: tuple[str, ...]
    ir_observation_fingerprints: tuple[str, ...]
    maximum_absolute_error: float
    maximum_relative_error: float
    checks: tuple[str, ...]
    fingerprint: str
    schema_version: str = MIXTRAL_SEMANTIC_CERTIFICATION_SCHEMA
    status: str = "passed"
    gate: str = "G13-mixtral-routed-only-semantics"
    execution_performed: bool = True
    execution_certified: bool = True
    production_runtime_eligible: bool = False

    def __post_init__(self) -> None:
        for value, field in (
            (self.artifact_id, "artifact_id"),
            (self.target_fingerprint, "target_fingerprint"),
            (self.fingerprint, "Mixtral semantic certification fingerprint"),
            *((value, "case input fingerprint") for value in self.case_input_fingerprints),
            *(
                (value, "source observation fingerprint")
                for value in self.source_observation_fingerprints
            ),
            *((value, "IR observation fingerprint") for value in self.ir_observation_fingerprints),
        ):
            require_sha256(value, field=field)
        if (
            self.schema_version != MIXTRAL_SEMANTIC_CERTIFICATION_SCHEMA
            or self.status != "passed"
            or self.gate != "G13-mixtral-routed-only-semantics"
            or not self.execution_performed
            or not self.execution_certified
            or self.production_runtime_eligible
        ):
            raise ValueError("unsupported Mixtral semantic certification boundary")
        if not self.case_input_fingerprints or not self.observation_names:
            raise ValueError("Mixtral semantic certification must bind cases and observations")
        expected_observations = len(self.case_input_fingerprints) * len(self.observation_names)
        if (
            len(self.source_observation_fingerprints) != expected_observations
            or len(self.ir_observation_fingerprints) != expected_observations
            or self.source_observation_fingerprints != self.ir_observation_fingerprints
        ):
            raise ValueError("Mixtral semantic observation fingerprints are inconsistent")
        if self.observation_names != tuple(sorted(set(self.observation_names))):
            raise ValueError("Mixtral observation names must be sorted and unique")
        if self.checks != tuple(sorted(set(self.checks))):
            raise ValueError("Mixtral semantic checks must be sorted and unique")
        if (
            not math.isfinite(self.maximum_absolute_error)
            or self.maximum_absolute_error < 0
            or not math.isfinite(self.maximum_relative_error)
            or self.maximum_relative_error < 0
        ):
            raise ValueError("Mixtral semantic errors must be finite and non-negative")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("Mixtral semantic certification fingerprint does not match")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "gate": self.gate,
            "artifact_id": self.artifact_id,
            "target_fingerprint": self.target_fingerprint,
            "case_input_fingerprints": list(self.case_input_fingerprints),
            "observation_names": list(self.observation_names),
            "source_observation_fingerprints": list(self.source_observation_fingerprints),
            "ir_observation_fingerprints": list(self.ir_observation_fingerprints),
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "checks": list(self.checks),
            "execution_performed": self.execution_performed,
            "execution_certified": self.execution_certified,
            "production_runtime_eligible": self.production_runtime_eligible,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class ReferencePromotionRecord:
    artifact_id: str
    target_fingerprint: str
    execution_certification_fingerprint: str
    fingerprint: str
    schema_version: str = REFERENCE_PROMOTION_SCHEMA
    status: str = "promoted-reference-only"
    execution_certified: bool = True
    production_runtime_eligible: bool = False
    native_lowering_status: str = "required-separate-target-lowering"

    def __post_init__(self) -> None:
        for value, field in (
            (self.artifact_id, "artifact_id"),
            (self.target_fingerprint, "target_fingerprint"),
            (self.execution_certification_fingerprint, "execution_certification_fingerprint"),
            (self.fingerprint, "reference promotion fingerprint"),
        ):
            require_sha256(value, field=field)
        if (
            self.schema_version != REFERENCE_PROMOTION_SCHEMA
            or self.status != "promoted-reference-only"
        ):
            raise ValueError("unsupported reference promotion record")
        if not self.execution_certified or self.production_runtime_eligible:
            raise ValueError("reference promotion cannot claim production eligibility")
        if self.native_lowering_status != "required-separate-target-lowering":
            raise ValueError("reference promotion native boundary is invalid")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("reference promotion fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "artifact_id": self.artifact_id,
            "target_fingerprint": self.target_fingerprint,
            "execution_certification_fingerprint": self.execution_certification_fingerprint,
            "execution_certified": self.execution_certified,
            "production_runtime_eligible": self.production_runtime_eligible,
            "native_lowering_status": self.native_lowering_status,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}


class ArtifactTensorStore:
    """Lazy, hash-rechecked tensor access over canonical allocation blobs."""

    def __init__(self, artifact: ComponentArtifact) -> None:
        self.artifact = artifact
        self._allocation_cache: dict[str, torch.Tensor] = {}
        self._allocation_ir = {
            item.allocation_id: item for item in artifact.ir_bundle.physical_weights.allocations
        }
        self._view_ir = {
            item.logical_name: item for item in artifact.ir_bundle.physical_weights.views
        }
        self._source_allocation = {
            item.source_tensor: item.allocation_id
            for item in artifact.ir_bundle.physical_weights.allocations
        }
        self._manifest_allocations = {
            item["allocation_id"]: item for item in artifact.manifest["allocations"]
        }

    def _load_allocation(self, allocation_id: str) -> torch.Tensor:
        cached = self._allocation_cache.get(allocation_id)
        if cached is not None:
            return cached
        allocation = self._allocation_ir.get(allocation_id)
        manifest = self._manifest_allocations.get(allocation_id)
        if allocation is None or manifest is None:
            raise ReferenceExecutionError(
                "tensor view references an absent allocation",
                details={"allocation_id": allocation_id},
            )
        dtype = _TORCH_DTYPES.get(allocation.stored_dtype)
        dtype_bytes = _DTYPE_BYTES.get(allocation.stored_dtype)
        if dtype is None or dtype_bytes is None:
            raise ReferenceExecutionError(
                "allocation dtype is not registered by the reference target",
                details={"allocation_id": allocation_id, "dtype": allocation.stored_dtype},
            )
        expected_bytes = _product(allocation.stored_shape) * dtype_bytes
        if expected_bytes != allocation.byte_length or expected_bytes <= 0:
            raise ReferenceExecutionError(
                "allocation shape/dtype byte count is inconsistent",
                details={
                    "allocation_id": allocation_id,
                    "shape_dtype_bytes": expected_bytes,
                    "declared_bytes": allocation.byte_length,
                },
            )
        blob = manifest["blob"]
        path = self.artifact.directory / blob["path"]
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ReferenceExecutionError(
                "cannot open an allocation blob without following links",
                details={"allocation_id": allocation_id},
            ) from exc
        digest = hashlib.sha256()
        payload = bytearray()
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
                raise ReferenceExecutionError(
                    "allocation blob identity or length changed after artifact reopen",
                    details={"allocation_id": allocation_id},
                )
            while True:
                block = os.read(descriptor, 8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                payload.extend(block)
            after = os.fstat(descriptor)
            if _stat_signature(before) != _stat_signature(after):
                raise ReferenceExecutionError(
                    "allocation blob changed while loading",
                    details={"allocation_id": allocation_id},
                )
        finally:
            os.close(descriptor)
        if len(payload) != expected_bytes or digest.hexdigest() != blob["sha256"]:
            raise ReferenceExecutionError(
                "allocation blob failed its execution-time content check",
                details={"allocation_id": allocation_id},
            )
        tensor = torch.frombuffer(payload, dtype=dtype, count=_product(allocation.stored_shape))
        tensor = tensor.reshape(allocation.stored_shape).clone()
        self._allocation_cache[allocation_id] = tensor
        return tensor

    def tensor(self, logical_name: str) -> torch.Tensor:
        view = self._view_ir.get(logical_name)
        if view is None:
            raise ReferenceExecutionError(
                "IR program requested an undeclared logical tensor",
                details={"logical_name": logical_name},
            )
        # The registered target intentionally supports identity views only.  The validation pass
        # guarantees this before execution; repeat the check here to fail closed after mutation.
        if (
            len(view.transforms) != 1
            or view.transforms[0].kind != "identity"
            or view.transforms[0].parameters
        ):
            raise ReferenceExecutionError(
                "reference target encountered an unregistered view transform",
                details={"logical_name": logical_name},
            )
        result = self._load_allocation(view.allocation_id)
        if tuple(result.shape) != view.logical_shape:
            raise ReferenceExecutionError(
                "logical tensor shape differs from its allocation",
                details={"logical_name": logical_name},
            )
        return result

    def source_tensor(self, source_name: str) -> torch.Tensor:
        allocation_id = self._source_allocation.get(source_name)
        if allocation_id is None:
            raise ReferenceExecutionError(
                "source-schema runner requested an absent tensor",
                details={"source_tensor": source_name},
            )
        return self._load_allocation(allocation_id)


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    input_dtype = value.dtype
    normalized = value.to(torch.float32)
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(input_dtype) * weight.to(input_dtype)


def _gemma_rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    input_dtype = value.dtype
    normalized = value.to(torch.float32)
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + epsilon)
    return (normalized * (1.0 + weight.to(torch.float32))).to(input_dtype)


def _qwen35_rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Qwen3.5's zero-centered RMSNorm: normalize and multiply by ``1 + weight`` in FP32."""

    input_dtype = value.dtype
    normalized = value.to(torch.float32)
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + epsilon)
    return (normalized * (1.0 + weight.to(torch.float32))).to(input_dtype)


def _head_rms_norm(
    value: torch.Tensor, weight: torch.Tensor, epsilon: float, head_dim: int
) -> torch.Tensor:
    if value.shape[-1] % head_dim:
        raise ReferenceExecutionError("head RMSNorm input width is not divisible by head_dim")
    original = value.shape
    reshaped = value.reshape(*original[:-1], original[-1] // head_dim, head_dim)
    return _rms_norm(reshaped, weight, epsilon).reshape(original)


def _qwen35_head_rms_norm(
    value: torch.Tensor, weight: torch.Tensor, epsilon: float, head_dim: int
) -> torch.Tensor:
    if value.shape[-1] % head_dim:
        raise ReferenceExecutionError("Qwen3.5 head RMSNorm width is not divisible by head_dim")
    original = value.shape
    reshaped = value.reshape(*original[:-1], original[-1] // head_dim, head_dim)
    return _qwen35_rms_norm(reshaped, weight, epsilon).reshape(original)


def _qwen35_causal_depthwise_convolution(
    value: torch.Tensor, weight: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    if value.ndim != 3 or weight.shape != (value.shape[-1], 1, kernel_size):
        raise ReferenceExecutionError("Qwen3.5 causal convolution shape is invalid")
    convolved = F.conv1d(
        value.transpose(1, 2),
        weight.to(value.dtype),
        padding=kernel_size - 1,
        groups=value.shape[-1],
    )[..., : value.shape[1]]
    return F.silu(convolved).transpose(1, 2).contiguous()


def _qwen35_query_gate_split(
    packed: torch.Tensor, *, num_attention_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = 2 * num_attention_heads * head_dim
    if packed.shape[-1] != expected:
        raise ReferenceExecutionError("Qwen3.5 packed query/gate width is invalid")
    shape = (*packed.shape[:-1], num_attention_heads, 2 * head_dim)
    query, gate = packed.reshape(shape).chunk(2, dim=-1)
    return query.reshape(*packed.shape[:-1], -1), gate.reshape(*packed.shape[:-1], -1)


def _qwen35_partial_mrope(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only interleaved mRoPE; T/H/W coordinates coincide for ordinary token positions."""

    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ReferenceExecutionError("Qwen3.5 partial mRoPE dimension is invalid")
    batch, sequence, query_width = query.shape
    if query_width % head_dim or key.shape[-1] % head_dim:
        raise ReferenceExecutionError("Qwen3.5 mRoPE width is not divisible by head_dim")
    q = query.reshape(batch, sequence, query_width // head_dim, head_dim).transpose(1, 2)
    k = key.reshape(batch, sequence, key.shape[-1] // head_dim, head_dim).transpose(1, 2)
    indices = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
    frequencies = 1.0 / (rope_theta ** (indices / rotary_dim))
    if type(position_offset) is not int or position_offset < 0:
        raise ReferenceExecutionError("Qwen3.5 mRoPE position offset is invalid")
    positions = torch.arange(position_offset, position_offset + sequence, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    embedding = torch.cat((angles, angles), dim=-1)
    cosine = embedding.cos().to(q.dtype)[None, None]
    sine = embedding.sin().to(q.dtype)[None, None]
    q_rotary = q[..., :rotary_dim]
    k_rotary = k[..., :rotary_dim]
    q = torch.cat((q_rotary * cosine + _rotate_half(q_rotary) * sine, q[..., rotary_dim:]), dim=-1)
    k = torch.cat((k_rotary * cosine + _rotate_half(k_rotary) * sine, k[..., rotary_dim:]), dim=-1)
    return q.transpose(1, 2).reshape(query.shape), k.transpose(1, 2).reshape(key.shape)


def _qwen35_gated_delta_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, _ = query.shape
    if query.shape[-1] != num_key_heads * key_head_dim or key.shape != query.shape:
        raise ReferenceExecutionError("Qwen3.5 gated-delta query/key geometry is invalid")
    if value.shape != (batch, sequence, num_value_heads * value_head_dim):
        raise ReferenceExecutionError("Qwen3.5 gated-delta value geometry is invalid")
    if a.shape != (batch, sequence, num_value_heads) or b.shape != a.shape:
        raise ReferenceExecutionError("Qwen3.5 gated-delta decay/gate geometry is invalid")
    if num_value_heads % num_key_heads:
        raise ReferenceExecutionError("Qwen3.5 gated-delta head replication is invalid")
    q = query.reshape(batch, sequence, num_key_heads, key_head_dim)
    k = key.reshape(batch, sequence, num_key_heads, key_head_dim)
    v = value.reshape(batch, sequence, num_value_heads, value_head_dim)
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    repeats = num_value_heads // num_key_heads
    if repeats > 1:
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)
    q = q.to(torch.float32) * (key_head_dim**-0.5)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = torch.sigmoid(b).to(torch.float32)
    decay = -torch.exp(a_log.to(torch.float32)) * F.softplus(
        a.to(torch.float32) + dt_bias.to(torch.float32)
    )
    state_shape = (batch, num_value_heads, key_head_dim, value_head_dim)
    state = (
        torch.zeros(state_shape, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32).clone()
    )
    if state.shape != state_shape:
        raise ReferenceExecutionError("Qwen3.5 initial recurrent state shape is invalid")
    outputs = torch.empty((batch, sequence, num_value_heads, value_head_dim), dtype=torch.float32)
    for position in range(sequence):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
        state = state * decay[:, position].exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - memory) * beta[:, position].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, position] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return outputs.to(query.dtype).reshape(batch, sequence, -1), state


def _qwen35_rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
    head_dim: int,
) -> torch.Tensor:
    if value.shape != gate.shape or value.shape[-1] % head_dim:
        raise ReferenceExecutionError("Qwen3.5 gated RMSNorm geometry is invalid")
    original = value.shape
    value_2d = value.reshape(-1, head_dim)
    gate_2d = gate.reshape(-1, head_dim)
    normalized = value_2d.to(torch.float32)
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + epsilon)
    normalized = weight.to(value.dtype) * normalized.to(value.dtype)
    normalized = normalized * F.silu(gate_2d.to(torch.float32))
    return normalized.to(value.dtype).reshape(original)


def _layer_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if weight.shape != bias.shape or tuple(weight.shape) != (value.shape[-1],):
        raise ReferenceExecutionError("LayerNorm parameters do not match the hidden width")
    return F.layer_norm(
        value,
        (value.shape[-1],),
        weight.to(value.dtype),
        bias.to(value.dtype),
        epsilon,
    )


def _linear(value: torch.Tensor, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = [item for name, item in parameters.items() if name.endswith(".weight")]
    biases = [item for name, item in parameters.items() if name.endswith(".bias")]
    if len(weights) != 1 or len(biases) > 1:
        raise ReferenceExecutionError("linear operation has an invalid parameter binding")
    weight = weights[0].to(value.dtype)
    bias = biases[0].to(value.dtype) if biases else None
    return F.linear(value, weight, bias)


def _moe_top_k(
    router_logits: torch.Tensor,
    *,
    num_experts_per_token: int,
    num_routed_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if router_logits.ndim != 3 or router_logits.shape[-1] != num_routed_experts:
        raise ReferenceExecutionError("MoE router logits do not match the routed-expert domain")
    if not 0 < num_experts_per_token < num_routed_experts:
        raise ReferenceExecutionError("MoE top-k is outside the routed-expert domain")
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, num_experts_per_token, dim=-1)
    denominator = routing_weights.sum(dim=-1, keepdim=True)
    if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
        raise ReferenceExecutionError("MoE selected routing mass is not finite and positive")
    routing_weights = (routing_weights / denominator).to(router_logits.dtype)
    return routing_weights, selected_experts


def _moe_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    num_routed_experts: int,
) -> tuple[torch.Tensor, ...]:
    if hidden_states.ndim != 3 or selected_experts.shape[:2] != hidden_states.shape[:2]:
        raise ReferenceExecutionError("MoE dispatch inputs have inconsistent token domains")
    if selected_experts.ndim != 3 or selected_experts.shape[-1] <= 0:
        raise ReferenceExecutionError("MoE dispatch requires a non-empty selected-expert axis")
    if selected_experts.dtype != torch.int64:
        raise ReferenceExecutionError("MoE selected experts must use int64 indices")
    if (
        int(selected_experts.min().item()) < 0
        or int(selected_experts.max().item()) >= num_routed_experts
    ):
        raise ReferenceExecutionError("MoE dispatch selected an expert outside its domain")
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    selected = selected_experts.reshape(-1, selected_experts.shape[-1])
    outputs: list[torch.Tensor] = []
    for expert in range(num_routed_experts):
        # Match the source framework's expert-mask traversal: top-k slot first, then flattened
        # source-token index.  This order is part of the numerical accumulation contract.
        _, token_indices = torch.where(selected.transpose(0, 1) == expert)
        outputs.append(flattened[token_indices])
    return tuple(outputs)


def _moe_weighted_scatter_add(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_outputs: Sequence[torch.Tensor],
    *,
    num_routed_experts: int,
) -> torch.Tensor:
    if len(expert_outputs) != num_routed_experts:
        raise ReferenceExecutionError("MoE combine received the wrong expert-output arity")
    if routing_weights.shape != selected_experts.shape or routing_weights.ndim != 3:
        raise ReferenceExecutionError("MoE routing weights and selected experts disagree")
    selected = selected_experts.reshape(-1, selected_experts.shape[-1])
    weights = routing_weights.reshape(-1, routing_weights.shape[-1])
    hidden_size = expert_outputs[0].shape[-1] if expert_outputs else 0
    if hidden_size <= 0:
        raise ReferenceExecutionError("MoE expert outputs have no hidden width")
    combined = torch.zeros(
        (selected.shape[0], hidden_size), dtype=expert_outputs[0].dtype, device="cpu"
    )
    for expert, expert_output in enumerate(expert_outputs):
        if expert_output.ndim != 2 or expert_output.shape[-1] != hidden_size:
            raise ReferenceExecutionError("MoE expert output has an inconsistent hidden width")
        top_k_indices, token_indices = torch.where(selected.transpose(0, 1) == expert)
        if expert_output.shape[0] != token_indices.numel():
            raise ReferenceExecutionError("MoE expert output row count differs from dispatch")
        weighted = expert_output * weights[token_indices, top_k_indices, None].to(
            expert_output.dtype
        )
        combined.index_add_(0, token_indices, weighted.to(combined.dtype))
    return combined.reshape(*routing_weights.shape[:2], hidden_size)


def _conv1d_linear(value: torch.Tensor, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = [item for name, item in parameters.items() if name.endswith(".weight")]
    biases = [item for name, item in parameters.items() if name.endswith(".bias")]
    if len(weights) != 1 or len(biases) != 1:
        raise ReferenceExecutionError("GPT-2 Conv1D operation has invalid parameter bindings")
    weight = weights[0].to(value.dtype)
    bias = biases[0].to(value.dtype)
    if weight.ndim != 2 or weight.shape[0] != value.shape[-1] or bias.shape != weight.shape[1:]:
        raise ReferenceExecutionError("GPT-2 Conv1D parameters do not match the activation width")
    flattened = value.reshape(-1, value.shape[-1])
    return torch.addmm(bias, flattened, weight).reshape(*value.shape[:-1], weight.shape[1])


def _mamba_causal_depthwise_convolution(
    value: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    kernel_size: int,
) -> torch.Tensor:
    """Execute the unfused HF Mamba causal Conv1d path over ``[batch, sequence, width]``."""

    weights = [item for name, item in parameters.items() if name.endswith(".conv1d.weight")]
    biases = [item for name, item in parameters.items() if name.endswith(".conv1d.bias")]
    if len(weights) != 1 or len(biases) > 1:
        raise ReferenceExecutionError("Mamba causal convolution has invalid parameter bindings")
    if value.ndim != 3:
        raise ReferenceExecutionError("Mamba causal convolution requires rank-3 activations")
    width = value.shape[-1]
    weight = weights[0]
    bias = biases[0] if biases else None
    if tuple(weight.shape) != (width, 1, kernel_size) or (
        bias is not None and tuple(bias.shape) != (width,)
    ):
        raise ReferenceExecutionError("Mamba causal convolution parameters have invalid shapes")
    sequence = value.shape[1]
    convolved = F.conv1d(
        value.transpose(1, 2),
        weight.to(value.dtype),
        None if bias is None else bias.to(value.dtype),
        padding=kernel_size - 1,
        groups=width,
    )[..., :sequence]
    return F.silu(convolved).transpose(1, 2)


def _mamba_selective_scan(
    hidden_states: torch.Tensor,
    dt_pre_softplus: torch.Tensor,
    input_b: torch.Tensor,
    input_c: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    state_size: int,
) -> torch.Tensor:
    """Execute the exact sequential HF Mamba recurrence from a zero initial state."""

    a_values = [item for name, item in parameters.items() if name.endswith(".A_log")]
    d_values = [item for name, item in parameters.items() if name.endswith(".D")]
    if len(a_values) != 1 or len(d_values) != 1:
        raise ReferenceExecutionError("Mamba selective scan has invalid parameter bindings")
    if hidden_states.ndim != 3:
        raise ReferenceExecutionError("Mamba selective scan requires rank-3 activations")
    batch, sequence, width = hidden_states.shape
    if (
        tuple(dt_pre_softplus.shape) != (batch, sequence, width)
        or tuple(input_b.shape) != (batch, sequence, state_size)
        or tuple(input_c.shape) != (batch, sequence, state_size)
        or tuple(a_values[0].shape) != (width, state_size)
        or tuple(d_values[0].shape) != (width,)
    ):
        raise ReferenceExecutionError("Mamba selective scan tensor shapes are inconsistent")

    dtype = hidden_states.dtype
    channel_first = hidden_states.transpose(1, 2)
    discrete_time_step = F.softplus(dt_pre_softplus).transpose(1, 2)
    continuous_a = -torch.exp(a_values[0].to(torch.float32))
    discrete_a = torch.exp(continuous_a[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_b = discrete_time_step[:, :, :, None] * input_b[:, None, :, :].to(torch.float32)
    delta_b_u = discrete_b * channel_first[:, :, :, None].to(torch.float32)
    state = torch.zeros((batch, width, state_size), dtype=dtype, device=hidden_states.device)
    outputs: list[torch.Tensor] = []
    for position in range(sequence):
        state = discrete_a[:, :, position, :] * state + delta_b_u[:, :, position, :]
        output = torch.matmul(state.to(dtype), input_c[:, position, :].unsqueeze(-1))
        outputs.append(output[:, :, 0])
    scan_output = torch.stack(outputs, dim=-1)
    scan_output = scan_output + channel_first * d_values[0][None, :, None]
    return scan_output.transpose(1, 2)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _unpack_gpt_neox_qkv(
    packed: torch.Tensor, *, num_attention_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected = 3 * num_attention_heads * head_dim
    if packed.shape[-1] != expected:
        raise ReferenceExecutionError(
            "GPT-NeoX fused QKV width differs from the per-head packing contract"
        )
    shaped = packed.reshape(*packed.shape[:-1], num_attention_heads, 3 * head_dim)
    query, key, value = shaped.split(head_dim, dim=-1)
    flattened = packed.shape[:-1] + (num_attention_heads * head_dim,)
    return (
        query.reshape(flattened),
        key.reshape(flattened),
        value.reshape(flattened),
    )


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float,
    inv_freq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, query_width = query.shape
    del batch
    if query_width % head_dim or key.shape[-1] % head_dim:
        raise ReferenceExecutionError("rotary input width is not divisible by head_dim")
    query_heads = query_width // head_dim
    key_heads = key.shape[-1] // head_dim
    q = query.reshape(query.shape[0], sequence, query_heads, head_dim).transpose(1, 2)
    k = key.reshape(key.shape[0], sequence, key_heads, head_dim).transpose(1, 2)
    if inv_freq is None:
        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        frequencies = 1.0 / (rope_theta ** (indices / head_dim))
    else:
        if tuple(inv_freq.shape) != (head_dim // 2,):
            raise ReferenceExecutionError("serialized rotary frequency has the wrong shape")
        frequencies = inv_freq.to(torch.float32)
    positions = torch.arange(sequence, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    embedding = torch.cat((angles, angles), dim=-1)
    cosine = embedding.cos().to(q.dtype)[None, None, :, :]
    sine = embedding.sin().to(q.dtype)[None, None, :, :]
    q = q * cosine + _rotate_half(q) * sine
    k = k * cosine + _rotate_half(k) * sine
    return q.transpose(1, 2).reshape(query.shape), k.transpose(1, 2).reshape(key.shape)


def _apply_partial_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    head_dim: int,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ReferenceExecutionError("partial rotary dimension is outside the GPT-NeoX contract")
    batch, sequence, query_width = query.shape
    if query_width % head_dim or key.shape[-1] % head_dim:
        raise ReferenceExecutionError("partial rotary input width is not divisible by head_dim")
    query_heads = query_width // head_dim
    key_heads = key.shape[-1] // head_dim
    q = query.reshape(batch, sequence, query_heads, head_dim).transpose(1, 2)
    k = key.reshape(key.shape[0], sequence, key_heads, head_dim).transpose(1, 2)
    indices = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
    frequencies = 1.0 / (rope_theta ** (indices / rotary_dim))
    positions = torch.arange(sequence, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    embedding = torch.cat((angles, angles), dim=-1)
    cosine = embedding.cos().to(q.dtype)[None, None, :, :]
    sine = embedding.sin().to(q.dtype)[None, None, :, :]
    q_rotary = q[..., :rotary_dim]
    k_rotary = k[..., :rotary_dim]
    q = torch.cat((q_rotary * cosine + _rotate_half(q_rotary) * sine, q[..., rotary_dim:]), dim=-1)
    k = torch.cat((k_rotary * cosine + _rotate_half(k_rotary) * sine, k[..., rotary_dim:]), dim=-1)
    return q.transpose(1, 2).reshape(query.shape), k.transpose(1, 2).reshape(key.shape)


def _gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    scale: float,
    causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, sequence, _ = query.shape
    q = query.reshape(batch, sequence, num_attention_heads, head_dim).transpose(1, 2)
    k = key.reshape(batch, sequence, num_key_value_heads, head_dim).transpose(1, 2)
    v = value.reshape(batch, sequence, num_key_value_heads, head_dim).transpose(1, 2)
    repeats = num_attention_heads // num_key_value_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal_mask is None:
        disallowed = torch.ones((sequence, sequence), dtype=torch.bool).triu(diagonal=1)
        disallowed = disallowed[None, None, :, :]
    else:
        if (
            causal_mask.ndim != 4
            or causal_mask.shape[0:2] != (1, 1)
            or causal_mask.shape[-2] < sequence
            or causal_mask.shape[-1] < sequence
        ):
            raise ReferenceExecutionError("serialized causal mask is outside the GPT-2 contract")
        allowed = causal_mask[:, :, :sequence, :sequence].to(torch.bool)
        disallowed = ~allowed
    scores = scores.masked_fill(disallowed, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
    context = torch.matmul(probabilities, v)
    return context.transpose(1, 2).contiguous().reshape(batch, sequence, -1)


def _rotary_parameter(parameters: dict[str, torch.Tensor]) -> torch.Tensor | None:
    direct = [
        value
        for name, value in parameters.items()
        if name.endswith(".inv_freq") and not name.endswith(".original_inv_freq")
    ]
    original = [value for name, value in parameters.items() if name.endswith(".original_inv_freq")]
    if len(direct) > 1 or len(original) > 1:
        raise ReferenceExecutionError("rotary operation has ambiguous serialized frequencies")
    if (
        direct
        and original
        and not torch.equal(direct[0].to(torch.float32), original[0].to(torch.float32))
    ):
        raise ReferenceExecutionError("default RoPE inv_freq and original_inv_freq disagree")
    return direct[0] if direct else (original[0] if original else None)


class ReferenceExecutable:
    """Lowered stateless full-sequence IR executable."""

    def __init__(
        self,
        artifact: ComponentArtifact,
        identity: ReferenceTargetIdentity,
        store: ArtifactTensorStore,
    ) -> None:
        self.artifact = artifact
        self.identity = identity
        self.store = store

    def _validate_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        value = token_ids.detach().to(device="cpu")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.numel() == 0:
            raise ReferenceExecutionError("token_ids must be a non-empty rank-1 or rank-2 tensor")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise ReferenceExecutionError("token_ids must use an integer dtype")
        value = value.to(torch.int64).contiguous()
        dimensions = self.artifact.ir_bundle.model.dimensions
        minimum = int(value.min().item())
        maximum = int(value.max().item())
        if minimum < 0 or maximum >= dimensions.vocab_size:
            raise ReferenceExecutionError(
                "token ID is outside the model token space",
                details={
                    "minimum": minimum,
                    "maximum": maximum,
                    "vocab_size": dimensions.vocab_size,
                },
            )
        if (
            dimensions.max_position_embeddings > 0
            and value.shape[1] > dimensions.max_position_embeddings
        ):
            raise ReferenceExecutionError(
                "sequence exceeds the registered default-RoPE context",
                details={
                    "sequence": value.shape[1],
                    "max_position_embeddings": dimensions.max_position_embeddings,
                },
            )
        return value

    @torch.inference_mode()
    def _execute_values(
        self, token_ids: torch.Tensor | Sequence[Sequence[int]]
    ) -> dict[str, torch.Tensor]:
        raw = token_ids if isinstance(token_ids, torch.Tensor) else torch.tensor(token_ids)
        values: dict[str, torch.Tensor] = {"token_ids": self._validate_tokens(raw)}
        for operation in self.artifact.ir_bundle.model.operations:
            inputs = [values[name] for name in operation.inputs]
            parameters = {name: self.store.tensor(name) for name in operation.parameters}
            outputs = self._execute_operation(operation, inputs, parameters)
            if len(outputs) != len(operation.outputs):
                raise ReferenceExecutionError(
                    "operation returned the wrong output arity",
                    details={"operation_id": operation.operation_id},
                )
            values.update(zip(operation.outputs, outputs, strict=True))
        return values

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor | Sequence[Sequence[int]]) -> ReferenceForwardResult:
        values = self._execute_values(token_ids)
        model = self.artifact.ir_bundle.model
        ports = {port.name: values[port.value] for port in model.output_ports}
        logits = ports["logits"]
        if logits.shape[-1] < model.dimensions.vocab_size:
            raise ReferenceExecutionError("readout has fewer rows than the model token domain")
        # The only registered output mappings are identity and padded identity.  Physical padding
        # rows are unreachable implementation detail and must never escape as token logits.
        logits = logits[..., : model.dimensions.vocab_size]
        return ReferenceForwardResult(logits=logits, hidden_states=ports["hidden_states"])

    @torch.inference_mode()
    def trace(
        self,
        token_ids: torch.Tensor | Sequence[Sequence[int]],
        value_names: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Return explicitly requested intermediate values from one reference execution."""

        requested = tuple(value_names)
        if not requested or len(set(requested)) != len(requested):
            raise ReferenceExecutionError("trace value names must be non-empty and unique")
        values = self._execute_values(token_ids)
        missing = sorted(set(requested) - set(values))
        if missing:
            raise ReferenceExecutionError(
                "trace requested values outside the executable dataflow",
                details={"missing": missing},
            )
        return {name: values[name].detach().clone() for name in requested}

    def _execute_operation(
        self,
        operation: OperationIR,
        inputs: list[torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        attributes = operation.attributes
        if operation.kind == "token-embedding":
            return (F.embedding(inputs[0], next(iter(parameters.values()))),)
        if operation.kind == "absolute-position-embedding-add":
            weight = next(iter(parameters.values())).to(inputs[0].dtype)
            sequence = inputs[0].shape[1]
            if sequence > weight.shape[0]:
                raise ReferenceExecutionError("absolute position table is shorter than the input")
            positions = torch.arange(sequence, dtype=torch.int64)
            return (inputs[0] + F.embedding(positions, weight)[None, :, :],)
        if operation.kind == "rms-norm":
            return (
                _rms_norm(inputs[0], next(iter(parameters.values())), float(attributes["epsilon"])),
            )
        if operation.kind == "gemma-rms-norm":
            return (
                _gemma_rms_norm(
                    inputs[0], next(iter(parameters.values())), float(attributes["epsilon"])
                ),
            )
        if operation.kind == "qwen35-rms-norm":
            return (
                _qwen35_rms_norm(
                    inputs[0], next(iter(parameters.values())), float(attributes["epsilon"])
                ),
            )
        if operation.kind == "qwen35-head-rms-norm":
            return (
                _qwen35_head_rms_norm(
                    inputs[0],
                    next(iter(parameters.values())),
                    float(attributes["epsilon"]),
                    int(attributes["head_dim"]),
                ),
            )
        if operation.kind == "head-rms-norm":
            return (
                _head_rms_norm(
                    inputs[0],
                    next(iter(parameters.values())),
                    float(attributes["epsilon"]),
                    int(attributes["head_dim"]),
                ),
            )
        if operation.kind == "layer-norm":
            weights = [value for name, value in parameters.items() if name.endswith(".weight")]
            biases = [value for name, value in parameters.items() if name.endswith(".bias")]
            if len(weights) != 1 or len(biases) != 1:
                raise ReferenceExecutionError("LayerNorm operation has invalid parameter bindings")
            return (_layer_norm(inputs[0], weights[0], biases[0], float(attributes["epsilon"])),)
        if operation.kind == "linear-readout" and self.identity.source_model_type == "mamba":
            weight = next(iter(parameters.values()))
            return (F.linear(inputs[0].to(weight.dtype), weight).to(torch.float32),)
        if operation.kind in {"linear", "linear-readout"}:
            return (_linear(inputs[0], parameters),)
        if operation.kind in {"moe-router-linear", "moe-routed-expert-linear"}:
            return (_linear(inputs[0], parameters),)
        if operation.kind == "moe-top-k-softmax":
            return _moe_top_k(
                inputs[0],
                num_experts_per_token=int(attributes["num_experts_per_token"]),
                num_routed_experts=int(attributes["num_routed_experts"]),
            )
        if operation.kind == "moe-token-dispatch":
            return _moe_dispatch(
                inputs[0],
                inputs[1],
                num_routed_experts=int(attributes["num_routed_experts"]),
            )
        if operation.kind == "moe-weighted-scatter-add":
            return (
                _moe_weighted_scatter_add(
                    inputs[0],
                    inputs[1],
                    inputs[2:],
                    num_routed_experts=int(attributes["num_routed_experts"]),
                ),
            )
        if operation.kind == "conv1d-linear":
            return (_conv1d_linear(inputs[0], parameters),)
        if operation.kind == "fused-qkv-split":
            width = int(attributes["split_width"])
            if inputs[0].shape[-1] != 3 * width:
                raise ReferenceExecutionError("fused QKV tensor has the wrong width")
            return tuple(inputs[0].split(width, dim=-1))
        if operation.kind == "mamba-input-gate-split":
            width = int(attributes["split_width"])
            if inputs[0].shape[-1] != 2 * width:
                raise ReferenceExecutionError("Mamba input/gate tensor has the wrong width")
            return tuple(inputs[0].split(width, dim=-1))
        if operation.kind == "mamba-causal-depthwise-convolution":
            return (
                _mamba_causal_depthwise_convolution(
                    inputs[0], parameters, kernel_size=int(attributes["kernel_size"])
                ),
            )
        if operation.kind == "qwen35-causal-depthwise-convolution":
            weight = next(iter(parameters.values()))
            return (
                _qwen35_causal_depthwise_convolution(
                    inputs[0], weight, int(attributes["kernel_size"])
                ),
            )
        if operation.kind == "qwen35-linear-qkv-split":
            key_width = int(attributes["key_width"])
            value_width = int(attributes["value_width"])
            if inputs[0].shape[-1] != 2 * key_width + value_width:
                raise ReferenceExecutionError("Qwen3.5 linear QKV tensor has the wrong width")
            return tuple(inputs[0].split((key_width, key_width, value_width), dim=-1))
        if operation.kind == "qwen35-query-gate-split":
            return _qwen35_query_gate_split(
                inputs[0],
                num_attention_heads=int(attributes["num_attention_heads"]),
                head_dim=int(attributes["head_dim"]),
            )
        if operation.kind == "qwen35-gated-delta-recurrence":
            a_logs = [value for name, value in parameters.items() if name.endswith(".A_log")]
            biases = [value for name, value in parameters.items() if name.endswith(".dt_bias")]
            if len(a_logs) != 1 or len(biases) != 1:
                raise ReferenceExecutionError("Qwen3.5 recurrence parameter binding is invalid")
            output, _state = _qwen35_gated_delta_recurrence(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                a_log=a_logs[0],
                dt_bias=biases[0],
                num_key_heads=int(attributes["num_key_heads"]),
                num_value_heads=int(attributes["num_value_heads"]),
                key_head_dim=int(attributes["key_head_dim"]),
                value_head_dim=int(attributes["value_head_dim"]),
            )
            return (output,)
        if operation.kind == "qwen35-rms-norm-gated":
            return (
                _qwen35_rms_norm_gated(
                    inputs[0],
                    inputs[1],
                    next(iter(parameters.values())),
                    epsilon=float(attributes["epsilon"]),
                    head_dim=int(attributes["head_dim"]),
                ),
            )
        if operation.kind == "mamba-selection-split":
            rank = int(attributes["time_step_rank"])
            state_size = int(attributes["state_size"])
            if inputs[0].shape[-1] != rank + 2 * state_size:
                raise ReferenceExecutionError("Mamba selection tensor has the wrong width")
            return tuple(inputs[0].split((rank, state_size, state_size), dim=-1))
        if operation.kind == "mamba-selective-scan":
            return (
                _mamba_selective_scan(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    parameters,
                    state_size=int(attributes["state_size"]),
                ),
            )
        if operation.kind == "gpt-neox-qkv-unpack":
            return _unpack_gpt_neox_qkv(
                inputs[0],
                num_attention_heads=int(attributes["num_attention_heads"]),
                head_dim=int(attributes["head_dim"]),
            )
        if operation.kind == "rotary-default":
            return _apply_rotary(
                inputs[0],
                inputs[1],
                head_dim=int(attributes["head_dim"]),
                rope_theta=float(attributes["rope_theta"]),
                inv_freq=_rotary_parameter(parameters),
            )
        if operation.kind == "rotary-partial-default":
            return _apply_partial_rotary(
                inputs[0],
                inputs[1],
                head_dim=int(attributes["head_dim"]),
                rotary_dim=int(attributes["rotary_dim"]),
                rope_theta=float(attributes["rope_theta"]),
            )
        if operation.kind == "qwen35-partial-interleaved-mrope":
            return _qwen35_partial_mrope(
                inputs[0],
                inputs[1],
                head_dim=int(attributes["head_dim"]),
                rotary_dim=int(attributes["rotary_dim"]),
                rope_theta=float(attributes["rope_theta"]),
            )
        if operation.kind == "causal-grouped-query-attention":
            causal_masks = [
                value for name, value in parameters.items() if name.endswith(".causal_mask")
            ]
            if len(causal_masks) > 1:
                raise ReferenceExecutionError("attention has multiple serialized causal masks")
            return (
                _gqa(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    num_attention_heads=int(attributes["num_attention_heads"]),
                    num_key_value_heads=int(attributes["num_key_value_heads"]),
                    head_dim=int(attributes["head_dim"]),
                    scale=float(attributes["scale"]),
                    causal_mask=causal_masks[0] if causal_masks else None,
                ),
            )
        if operation.kind == "residual-add":
            if attributes.get("residual_accumulation") == "float32":
                return (inputs[0].to(torch.float32) + inputs[1].to(torch.float32),)
            return (inputs[0] + inputs[1],)
        if operation.kind == "mamba-rms-norm":
            weight = next(iter(parameters.values()))
            value = inputs[0]
            if operation.operation_id.startswith("layers."):
                value = value.to(weight.dtype)
            return (_rms_norm(value, weight, float(attributes["epsilon"])),)
        if operation.kind == "scalar-multiply":
            return (inputs[0] * float(attributes["scalar"]),)
        if operation.kind == "silu":
            return (F.silu(inputs[0]),)
        if operation.kind == "gelu-erf":
            return (F.gelu(inputs[0], approximate="none"),)
        if operation.kind == "gelu-tanh":
            return (F.gelu(inputs[0], approximate="tanh"),)
        if operation.kind == "elementwise-multiply":
            return (inputs[0] * inputs[1],)
        if operation.kind == "qwen35-attention-output-gate":
            if inputs[0].shape != inputs[1].shape:
                raise ReferenceExecutionError("Qwen3.5 attention gate shape is invalid")
            return (inputs[0] * torch.sigmoid(inputs[1]),)
        raise ReferenceExecutionError(
            "IR operation kind is not registered by the reference target",
            details={"operation_id": operation.operation_id, "kind": operation.kind},
        )


class _SourceSchemaRunner:
    """Independent architecture wiring over original source tensor names."""

    def __init__(self, executable: ReferenceExecutable) -> None:
        self.executable = executable
        self.store = executable.store
        self.config = executable.artifact.source.config
        self.family = executable.identity.source_model_type
        self.dimensions = executable.artifact.ir_bundle.model.dimensions

    def _tensor(self, name: str) -> torch.Tensor:
        return self.store.source_tensor(name)

    def _maybe_tensor(self, name: str) -> torch.Tensor | None:
        try:
            return self._tensor(name)
        except ReferenceExecutionError:
            return None

    def _linear(self, value: torch.Tensor, prefix: str, *, bias: bool) -> torch.Tensor:
        parameters = {f"{prefix}.weight": self._tensor(f"{prefix}.weight")}
        if bias:
            parameters[f"{prefix}.bias"] = self._tensor(f"{prefix}.bias")
        return _linear(value, parameters)

    def _inv_freq(self, layer: int) -> torch.Tensor | None:
        direct = self._maybe_tensor(f"model.layers.{layer}.self_attn.rotary_emb.inv_freq")
        if direct is None:
            direct = self._maybe_tensor("model.rotary_emb.inv_freq")
        original = self._maybe_tensor(
            f"model.layers.{layer}.self_attn.rotary_emb.original_inv_freq"
        )
        if original is None:
            original = self._maybe_tensor("model.rotary_emb.original_inv_freq")
        values: dict[str, torch.Tensor] = {}
        if direct is not None:
            values["rotary.inv_freq"] = direct
        if original is not None:
            values["rotary.original_inv_freq"] = original
        return _rotary_parameter(values)

    def _token_domain_result(self, result: ReferenceForwardResult) -> ReferenceForwardResult:
        vocab_size = self.dimensions.vocab_size
        if result.logits.shape[-1] < vocab_size:
            raise ReferenceExecutionError("source readout has fewer rows than the token domain")
        return ReferenceForwardResult(
            logits=result.logits[..., :vocab_size], hidden_states=result.hidden_states
        )

    def _forward_gpt2(self, token_ids: torch.Tensor) -> ReferenceForwardResult:
        d = self.dimensions
        positions = torch.arange(token_ids.shape[1], dtype=torch.int64)
        hidden = F.embedding(token_ids, self._tensor("transformer.wte.weight"))
        hidden = hidden + F.embedding(positions, self._tensor("transformer.wpe.weight"))[None]
        epsilon = float(self.config.get("layer_norm_epsilon", 1e-5))
        for layer in range(d.num_hidden_layers):
            source = f"transformer.h.{layer}"
            residual = hidden
            normed = _layer_norm(
                hidden,
                self._tensor(f"{source}.ln_1.weight"),
                self._tensor(f"{source}.ln_1.bias"),
                epsilon,
            )
            packed = _conv1d_linear(
                normed,
                {
                    f"{source}.attn.c_attn.weight": self._tensor(f"{source}.attn.c_attn.weight"),
                    f"{source}.attn.c_attn.bias": self._tensor(f"{source}.attn.c_attn.bias"),
                },
            )
            query, key, value = packed.split(d.hidden_size, dim=-1)
            context = _gqa(
                query,
                key,
                value,
                num_attention_heads=d.num_attention_heads,
                num_key_value_heads=d.num_key_value_heads,
                head_dim=d.head_dim,
                scale=d.head_dim**-0.5,
                causal_mask=self._tensor(f"{source}.attn.bias"),
            )
            attention = _conv1d_linear(
                context,
                {
                    f"{source}.attn.c_proj.weight": self._tensor(f"{source}.attn.c_proj.weight"),
                    f"{source}.attn.c_proj.bias": self._tensor(f"{source}.attn.c_proj.bias"),
                },
            )
            hidden = residual + attention
            residual = hidden
            normed = _layer_norm(
                hidden,
                self._tensor(f"{source}.ln_2.weight"),
                self._tensor(f"{source}.ln_2.bias"),
                epsilon,
            )
            expanded = _conv1d_linear(
                normed,
                {
                    f"{source}.mlp.c_fc.weight": self._tensor(f"{source}.mlp.c_fc.weight"),
                    f"{source}.mlp.c_fc.bias": self._tensor(f"{source}.mlp.c_fc.bias"),
                },
            )
            activated = F.gelu(expanded, approximate="tanh")
            projected = _conv1d_linear(
                activated,
                {
                    f"{source}.mlp.c_proj.weight": self._tensor(f"{source}.mlp.c_proj.weight"),
                    f"{source}.mlp.c_proj.bias": self._tensor(f"{source}.mlp.c_proj.bias"),
                },
            )
            hidden = residual + projected
        hidden = _layer_norm(
            hidden,
            self._tensor("transformer.ln_f.weight"),
            self._tensor("transformer.ln_f.bias"),
            epsilon,
        )
        head = (
            self._tensor("transformer.wte.weight")
            if bool(self.config.get("tie_word_embeddings", True))
            else self._tensor("lm_head.weight")
        )
        return ReferenceForwardResult(
            logits=F.linear(hidden, head.to(hidden.dtype)),
            hidden_states=hidden,
        )

    def _forward_gpt_neox(self, token_ids: torch.Tensor) -> ReferenceForwardResult:
        d = self.dimensions
        hidden = F.embedding(token_ids, self._tensor("gpt_neox.embed_in.weight"))
        epsilon = float(self.config.get("layer_norm_eps", 1e-5))
        rope = self.config.get("rope_parameters") or {}
        rope_theta = float(rope.get("rope_theta", self.config.get("rotary_emb_base", 10_000.0)))
        rotary_pct = float(rope.get("partial_rotary_factor", self.config.get("rotary_pct", 0.25)))
        rotary_dim = int(d.head_dim * rotary_pct)
        attention_bias = bool(self.config.get("attention_bias", True))
        parallel = bool(self.config.get("use_parallel_residual", True))
        for layer in range(d.num_hidden_layers):
            source = f"gpt_neox.layers.{layer}"
            residual = hidden
            normed = _layer_norm(
                hidden,
                self._tensor(f"{source}.input_layernorm.weight"),
                self._tensor(f"{source}.input_layernorm.bias"),
                epsilon,
            )
            packed = self._linear(
                normed,
                f"{source}.attention.query_key_value",
                bias=attention_bias,
            )
            query, key, value = _unpack_gpt_neox_qkv(
                packed,
                num_attention_heads=d.num_attention_heads,
                head_dim=d.head_dim,
            )
            query, key = _apply_partial_rotary(
                query,
                key,
                head_dim=d.head_dim,
                rotary_dim=rotary_dim,
                rope_theta=rope_theta,
            )
            context = _gqa(
                query,
                key,
                value,
                num_attention_heads=d.num_attention_heads,
                num_key_value_heads=d.num_key_value_heads,
                head_dim=d.head_dim,
                scale=d.head_dim**-0.5,
            )
            attention_output = self._linear(
                context,
                f"{source}.attention.dense",
                bias=attention_bias,
            )
            attention_residual = residual + attention_output
            mlp_base = residual if parallel else attention_residual
            mlp_input = _layer_norm(
                mlp_base,
                self._tensor(f"{source}.post_attention_layernorm.weight"),
                self._tensor(f"{source}.post_attention_layernorm.bias"),
                epsilon,
            )
            expanded = self._linear(
                mlp_input,
                f"{source}.mlp.dense_h_to_4h",
                bias=True,
            )
            activated = F.gelu(expanded, approximate="none")
            mlp_output = self._linear(
                activated,
                f"{source}.mlp.dense_4h_to_h",
                bias=True,
            )
            hidden = attention_residual + mlp_output
        hidden = _layer_norm(
            hidden,
            self._tensor("gpt_neox.final_layer_norm.weight"),
            self._tensor("gpt_neox.final_layer_norm.bias"),
            epsilon,
        )
        logits = F.linear(hidden, self._tensor("embed_out.weight").to(hidden.dtype))
        return ReferenceForwardResult(logits=logits, hidden_states=hidden)

    def _forward_phi(self, token_ids: torch.Tensor) -> ReferenceForwardResult:
        d = self.dimensions
        hidden = F.embedding(token_ids, self._tensor("model.embed_tokens.weight"))
        epsilon = float(self.config.get("layer_norm_eps", 1e-5))
        rope = self.config.get("rope_parameters") or {}
        theta = float(rope.get("rope_theta", self.config.get("rope_theta", 10_000.0)))
        partial = float(self.config.get("partial_rotary_factor", 0.5))
        rotary_dim = int(d.head_dim * partial)
        for layer in range(d.num_hidden_layers):
            source = f"model.layers.{layer}"
            residual = hidden
            normed = _layer_norm(
                hidden,
                self._tensor(f"{source}.input_layernorm.weight"),
                self._tensor(f"{source}.input_layernorm.bias"),
                epsilon,
            )
            query = self._linear(normed, f"{source}.self_attn.q_proj", bias=True)
            key = self._linear(normed, f"{source}.self_attn.k_proj", bias=True)
            value = self._linear(normed, f"{source}.self_attn.v_proj", bias=True)
            query, key = _apply_partial_rotary(
                query,
                key,
                head_dim=d.head_dim,
                rotary_dim=rotary_dim,
                rope_theta=theta,
            )
            context = _gqa(
                query,
                key,
                value,
                num_attention_heads=d.num_attention_heads,
                num_key_value_heads=d.num_key_value_heads,
                head_dim=d.head_dim,
                scale=d.head_dim**-0.5,
            )
            attention = self._linear(context, f"{source}.self_attn.dense", bias=True)
            expanded = self._linear(normed, f"{source}.mlp.fc1", bias=True)
            activated = F.gelu(expanded, approximate="tanh")
            feed_forward = self._linear(activated, f"{source}.mlp.fc2", bias=True)
            hidden = attention + feed_forward + residual
        hidden = _layer_norm(
            hidden,
            self._tensor("model.final_layernorm.weight"),
            self._tensor("model.final_layernorm.bias"),
            epsilon,
        )
        logits = F.linear(
            hidden,
            self._tensor("lm_head.weight").to(hidden.dtype),
            self._tensor("lm_head.bias").to(hidden.dtype),
        )
        return ReferenceForwardResult(logits=logits, hidden_states=hidden)

    def _forward_mamba(self, token_ids: torch.Tensor) -> ReferenceForwardResult:
        """Independent source-name wiring for stateless full-sequence Mamba execution."""

        d = self.dimensions
        hidden = F.embedding(token_ids, self._tensor("backbone.embeddings.weight"))
        epsilon = float(self.config.get("layer_norm_epsilon", 1e-5))
        state_size = int(self.config["state_size"])
        conv_kernel = int(self.config["conv_kernel"])
        time_step_rank = self.config.get("time_step_rank", "auto")
        rank = math.ceil(d.hidden_size / 16) if time_step_rank == "auto" else int(time_step_rank)
        use_bias = bool(self.config.get("use_bias", False))
        use_conv_bias = bool(self.config.get("use_conv_bias", True))
        residual_in_fp32 = bool(self.config.get("residual_in_fp32", True))
        for layer in range(d.num_hidden_layers):
            source = f"backbone.layers.{layer}"
            mixer = f"{source}.mixer"
            residual = hidden
            norm_weight = self._tensor(f"{source}.norm.weight")
            normed = _rms_norm(hidden.to(norm_weight.dtype), norm_weight, epsilon)
            packed = self._linear(normed, f"{mixer}.in_proj", bias=use_bias)
            x_value, gate = packed.split(d.intermediate_size, dim=-1)

            convolution_parameters = {
                f"{mixer}.conv1d.weight": self._tensor(f"{mixer}.conv1d.weight")
            }
            if use_conv_bias:
                convolution_parameters[f"{mixer}.conv1d.bias"] = self._tensor(
                    f"{mixer}.conv1d.bias"
                )
            convolved = _mamba_causal_depthwise_convolution(
                x_value, convolution_parameters, kernel_size=conv_kernel
            )
            selection = self._linear(convolved, f"{mixer}.x_proj", bias=False)
            dt_input, input_b, input_c = selection.split((rank, state_size, state_size), dim=-1)
            dt_pre_softplus = self._linear(dt_input, f"{mixer}.dt_proj", bias=True)
            scan_output = _mamba_selective_scan(
                convolved,
                dt_pre_softplus,
                input_b,
                input_c,
                {
                    f"{mixer}.A_log": self._tensor(f"{mixer}.A_log"),
                    f"{mixer}.D": self._tensor(f"{mixer}.D"),
                },
                state_size=state_size,
            )
            mixer_output = self._linear(
                scan_output * F.silu(gate), f"{mixer}.out_proj", bias=use_bias
            )
            hidden = (
                residual.to(torch.float32) + mixer_output.to(torch.float32)
                if residual_in_fp32
                else residual + mixer_output
            )

        hidden = _rms_norm(
            hidden,
            self._tensor("backbone.norm_f.weight"),
            epsilon,
        )
        head = (
            self._tensor("backbone.embeddings.weight")
            if bool(self.config.get("tie_word_embeddings", True))
            else self._tensor("lm_head.weight")
        )
        logits = F.linear(hidden.to(head.dtype), head).to(torch.float32)
        return ReferenceForwardResult(logits=logits, hidden_states=hidden)

    def _forward_mixtral(
        self,
        token_ids: torch.Tensor,
        *,
        trace: dict[str, torch.Tensor] | None = None,
    ) -> ReferenceForwardResult:
        """Independent wiring for the serialized routed-only HF Mixtral schema."""

        d = self.dimensions
        hidden = F.embedding(token_ids, self._tensor("model.embed_tokens.weight"))
        epsilon = float(self.config["rms_norm_eps"])
        rope = self.config.get("rope_parameters") or {}
        rope_theta = float(rope.get("rope_theta", self.config.get("rope_theta", 10_000.0)))
        num_experts = int(self.config["num_local_experts"])
        top_k = int(self.config["num_experts_per_tok"])
        for layer in range(d.num_hidden_layers):
            source = f"model.layers.{layer}"
            residual = hidden
            normed = _rms_norm(hidden, self._tensor(f"{source}.input_layernorm.weight"), epsilon)
            query = self._linear(normed, f"{source}.self_attn.q_proj", bias=False)
            key = self._linear(normed, f"{source}.self_attn.k_proj", bias=False)
            value = self._linear(normed, f"{source}.self_attn.v_proj", bias=False)
            query, key = _apply_rotary(
                query,
                key,
                head_dim=d.head_dim,
                rope_theta=rope_theta,
                inv_freq=self._inv_freq(layer),
            )
            context = _gqa(
                query,
                key,
                value,
                num_attention_heads=d.num_attention_heads,
                num_key_value_heads=d.num_key_value_heads,
                head_dim=d.head_dim,
                scale=d.head_dim**-0.5,
            )
            hidden = residual + self._linear(context, f"{source}.self_attn.o_proj", bias=False)

            residual = hidden
            routed_input = _rms_norm(
                hidden, self._tensor(f"{source}.post_attention_layernorm.weight"), epsilon
            )
            original_shape = routed_input.shape
            flattened = routed_input.reshape(-1, d.hidden_size)
            router_logits = F.linear(
                flattened,
                self._tensor(f"{source}.block_sparse_moe.gate.weight").to(flattened.dtype),
            )
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(flattened.dtype)
            moe = f"layers.{layer}.moe"
            if trace is not None:
                trace[f"{moe}.router_logits"] = (
                    router_logits.reshape(*original_shape[:2], num_experts).detach().clone()
                )
                trace[f"{moe}.routing_weights"] = (
                    routing_weights.reshape(*original_shape[:2], top_k).detach().clone()
                )
                trace[f"{moe}.selected_experts"] = (
                    selected_experts.reshape(*original_shape[:2], top_k).detach().clone()
                )
            final_hidden = torch.zeros_like(flattened)
            expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
            for expert in range(num_experts):
                top_k_indices, token_indices = torch.where(expert_mask[expert])
                if token_indices.numel() == 0:
                    if trace is not None:
                        trace[f"{moe}.routed_experts.{expert}.output"] = flattened.new_empty(
                            (0, d.hidden_size)
                        )
                    continue
                current = flattened[None, token_indices].reshape(-1, d.hidden_size)
                expert_source = f"{source}.block_sparse_moe.experts.{expert}"
                gate = F.linear(current, self._tensor(f"{expert_source}.w1.weight"))
                up = F.linear(current, self._tensor(f"{expert_source}.w3.weight"))
                output = F.linear(
                    F.silu(gate) * up,
                    self._tensor(f"{expert_source}.w2.weight"),
                )
                if trace is not None:
                    trace[f"{moe}.routed_experts.{expert}.output"] = output.detach().clone()
                output *= routing_weights[token_indices, top_k_indices, None]
                final_hidden.index_add_(0, token_indices, output.to(final_hidden.dtype))
            if trace is not None:
                trace[f"{moe}.output"] = final_hidden.reshape(original_shape).detach().clone()
            hidden = residual + final_hidden.reshape(original_shape)
        hidden = _rms_norm(hidden, self._tensor("model.norm.weight"), epsilon)
        head = (
            self._tensor("model.embed_tokens.weight")
            if bool(self.config.get("tie_word_embeddings", False))
            else self._tensor("lm_head.weight")
        )
        return ReferenceForwardResult(
            logits=F.linear(hidden, head.to(hidden.dtype)),
            hidden_states=hidden,
        )

    def _forward_qwen35(self, token_ids: torch.Tensor) -> ReferenceForwardResult:
        """Independent nested-source wiring for the Qwen3.5 text decoder."""

        text = self.config.get("text_config")
        if not isinstance(text, dict):
            raise ReferenceExecutionError("Qwen3.5 source has no nested text_config")
        d = self.dimensions
        root = "model.language_model"
        hidden = F.embedding(token_ids, self._tensor(f"{root}.embed_tokens.weight"))
        epsilon = float(text["rms_norm_eps"])
        rope = text["rope_parameters"]
        rotary_dim = int(d.head_dim * float(rope["partial_rotary_factor"]))
        key_heads = int(text["linear_num_key_heads"])
        value_heads = int(text["linear_num_value_heads"])
        key_dim = int(text["linear_key_head_dim"])
        value_dim = int(text["linear_value_head_dim"])
        conv_kernel = int(text["linear_conv_kernel_dim"])
        layer_types = tuple(text["layer_types"])
        for layer, layer_type in enumerate(layer_types):
            source = f"{root}.layers.{layer}"
            residual = hidden
            normed = _qwen35_rms_norm(
                hidden, self._tensor(f"{source}.input_layernorm.weight"), epsilon
            )
            if layer_type == "linear_attention":
                mixer = f"{source}.linear_attn"
                packed = self._linear(normed, f"{mixer}.in_proj_qkv", bias=False)
                convolved = _qwen35_causal_depthwise_convolution(
                    packed, self._tensor(f"{mixer}.conv1d.weight"), conv_kernel
                )
                key_width = key_heads * key_dim
                value_width = value_heads * value_dim
                query, key, value = convolved.split((key_width, key_width, value_width), dim=-1)
                z = self._linear(normed, f"{mixer}.in_proj_z", bias=False)
                a = self._linear(normed, f"{mixer}.in_proj_a", bias=False)
                b = self._linear(normed, f"{mixer}.in_proj_b", bias=False)
                core, _state = _qwen35_gated_delta_recurrence(
                    query,
                    key,
                    value,
                    a,
                    b,
                    a_log=self._tensor(f"{mixer}.A_log"),
                    dt_bias=self._tensor(f"{mixer}.dt_bias"),
                    num_key_heads=key_heads,
                    num_value_heads=value_heads,
                    key_head_dim=key_dim,
                    value_head_dim=value_dim,
                )
                core = _qwen35_rms_norm_gated(
                    core,
                    z,
                    self._tensor(f"{mixer}.norm.weight"),
                    epsilon=epsilon,
                    head_dim=value_dim,
                )
                mixer_output = self._linear(core, f"{mixer}.out_proj", bias=False)
            elif layer_type == "full_attention":
                attention = f"{source}.self_attn"
                packed_query = self._linear(normed, f"{attention}.q_proj", bias=False)
                query, gate = _qwen35_query_gate_split(
                    packed_query,
                    num_attention_heads=d.num_attention_heads,
                    head_dim=d.head_dim,
                )
                key = self._linear(normed, f"{attention}.k_proj", bias=False)
                value = self._linear(normed, f"{attention}.v_proj", bias=False)
                query = _qwen35_head_rms_norm(
                    query, self._tensor(f"{attention}.q_norm.weight"), epsilon, d.head_dim
                )
                key = _qwen35_head_rms_norm(
                    key, self._tensor(f"{attention}.k_norm.weight"), epsilon, d.head_dim
                )
                query, key = _qwen35_partial_mrope(
                    query,
                    key,
                    head_dim=d.head_dim,
                    rotary_dim=rotary_dim,
                    rope_theta=float(rope["rope_theta"]),
                )
                context = _gqa(
                    query,
                    key,
                    value,
                    num_attention_heads=d.num_attention_heads,
                    num_key_value_heads=d.num_key_value_heads,
                    head_dim=d.head_dim,
                    scale=d.head_dim**-0.5,
                )
                mixer_output = self._linear(
                    context * torch.sigmoid(gate), f"{attention}.o_proj", bias=False
                )
            else:
                raise ReferenceExecutionError("Qwen3.5 source layer has an unknown mixer")
            hidden = residual + mixer_output
            residual = hidden
            normed = _qwen35_rms_norm(
                hidden, self._tensor(f"{source}.post_attention_layernorm.weight"), epsilon
            )
            gate = self._linear(normed, f"{source}.mlp.gate_proj", bias=False)
            up = self._linear(normed, f"{source}.mlp.up_proj", bias=False)
            hidden = residual + self._linear(
                F.silu(gate) * up, f"{source}.mlp.down_proj", bias=False
            )
        hidden = _qwen35_rms_norm(hidden, self._tensor(f"{root}.norm.weight"), epsilon)
        tied = bool(self.config.get("tie_word_embeddings", True))
        head = (
            self._tensor(f"{root}.embed_tokens.weight") if tied else self._tensor("lm_head.weight")
        )
        return ReferenceForwardResult(
            logits=F.linear(hidden, head.to(hidden.dtype)), hidden_states=hidden
        )

    @torch.inference_mode()
    def mixtral_trace(
        self, token_ids: torch.Tensor
    ) -> tuple[ReferenceForwardResult, dict[str, torch.Tensor]]:
        if self.family != "mixtral":
            raise ReferenceExecutionError("Mixtral semantic trace requires a Mixtral artifact")
        normalized = self.executable._validate_tokens(token_ids)
        trace: dict[str, torch.Tensor] = {}
        result = self._token_domain_result(self._forward_mixtral(normalized, trace=trace))
        return result, trace

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> ReferenceForwardResult:
        token_ids = self.executable._validate_tokens(token_ids)
        if self.family == "gpt2":
            return self._token_domain_result(self._forward_gpt2(token_ids))
        if self.family == "gpt_neox":
            return self._token_domain_result(self._forward_gpt_neox(token_ids))
        if self.family == "phi":
            return self._token_domain_result(self._forward_phi(token_ids))
        if self.family == "mamba":
            return self._token_domain_result(self._forward_mamba(token_ids))
        if self.family == "mixtral":
            return self._token_domain_result(self._forward_mixtral(token_ids))
        if self.family == "qwen3_5":
            return self._token_domain_result(self._forward_qwen35(token_ids))
        d = self.dimensions
        hidden = F.embedding(token_ids, self._tensor("model.embed_tokens.weight"))
        if self.family == "gemma":
            hidden = hidden * math.sqrt(d.hidden_size)
        attention_bias = bool(self.config.get("attention_bias", False))
        mlp_bias = bool(self.config.get("mlp_bias", False))
        epsilon = float(self.config.get("rms_norm_eps", 1e-6))
        rope = self.config.get("rope_parameters") or {}
        rope_theta = float(rope.get("rope_theta", self.config.get("rope_theta", 10_000.0)))
        for layer in range(d.num_hidden_layers):
            source = f"model.layers.{layer}"
            residual = hidden
            if self.family == "gemma":
                normed = _gemma_rms_norm(
                    hidden, self._tensor(f"{source}.input_layernorm.weight"), epsilon
                )
            else:
                normed = _rms_norm(
                    hidden, self._tensor(f"{source}.input_layernorm.weight"), epsilon
                )
            qkv_bias = True if self.family == "qwen2" else attention_bias
            query = self._linear(normed, f"{source}.self_attn.q_proj", bias=qkv_bias)
            key = self._linear(normed, f"{source}.self_attn.k_proj", bias=qkv_bias)
            value = self._linear(normed, f"{source}.self_attn.v_proj", bias=qkv_bias)
            if self.family == "qwen3":
                query = _head_rms_norm(
                    query, self._tensor(f"{source}.self_attn.q_norm.weight"), epsilon, d.head_dim
                )
                key = _head_rms_norm(
                    key, self._tensor(f"{source}.self_attn.k_norm.weight"), epsilon, d.head_dim
                )
            query, key = _apply_rotary(
                query,
                key,
                head_dim=d.head_dim,
                rope_theta=rope_theta,
                inv_freq=self._inv_freq(layer),
            )
            context = _gqa(
                query,
                key,
                value,
                num_attention_heads=d.num_attention_heads,
                num_key_value_heads=d.num_key_value_heads,
                head_dim=d.head_dim,
                scale=d.head_dim**-0.5,
            )
            output_bias = attention_bias and self.family != "qwen2"
            hidden = residual + self._linear(
                context, f"{source}.self_attn.o_proj", bias=output_bias
            )
            residual = hidden
            if self.family == "gemma":
                normed = _gemma_rms_norm(
                    hidden,
                    self._tensor(f"{source}.post_attention_layernorm.weight"),
                    epsilon,
                )
            else:
                normed = _rms_norm(
                    hidden,
                    self._tensor(f"{source}.post_attention_layernorm.weight"),
                    epsilon,
                )
            gate = self._linear(normed, f"{source}.mlp.gate_proj", bias=mlp_bias)
            up = self._linear(normed, f"{source}.mlp.up_proj", bias=mlp_bias)
            activated = F.gelu(gate, approximate="tanh") if self.family == "gemma" else F.silu(gate)
            intermediate = activated * up
            hidden = residual + self._linear(intermediate, f"{source}.mlp.down_proj", bias=mlp_bias)
        hidden = (
            _gemma_rms_norm(hidden, self._tensor("model.norm.weight"), epsilon)
            if self.family == "gemma"
            else _rms_norm(hidden, self._tensor("model.norm.weight"), epsilon)
        )
        head = (
            self._tensor("model.embed_tokens.weight")
            if bool(self.config.get("tie_word_embeddings", False))
            else self._tensor("lm_head.weight")
        )
        logits = F.linear(hidden, head.to(hidden.dtype))
        return self._token_domain_result(
            ReferenceForwardResult(logits=logits, hidden_states=hidden)
        )


def _expected_program(
    dimensions: ModelDimensionsIR, architecture_id: str, config: dict[str, Any]
) -> tuple[tuple[str, str], ...]:
    program: list[tuple[str, str]] = [("embedding", "token-embedding")]
    if architecture_id == "qwen3_5-hybrid-text-causal-decoder":
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise ReferenceLoweringError("Qwen3.5 text_config is absent")
        layer_types = text.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != dimensions.num_hidden_layers:
            raise ReferenceLoweringError("Qwen3.5 layer_types differs from ModelIR")
        for layer, layer_type in enumerate(layer_types):
            prefix = f"layers.{layer}"
            program.append((f"{prefix}.mixer_norm", "qwen35-rms-norm"))
            if layer_type == "linear_attention":
                mixer = f"{prefix}.gated_delta"
                program.extend(
                    [
                        (f"{mixer}.in_proj_qkv", "linear"),
                        (f"{mixer}.in_proj_z", "linear"),
                        (f"{mixer}.in_proj_a", "linear"),
                        (f"{mixer}.in_proj_b", "linear"),
                        (f"{mixer}.causal_conv", "qwen35-causal-depthwise-convolution"),
                        (f"{mixer}.split_qkv", "qwen35-linear-qkv-split"),
                        (f"{mixer}.recurrence", "qwen35-gated-delta-recurrence"),
                        (f"{mixer}.gated_norm", "qwen35-rms-norm-gated"),
                        (f"{mixer}.out_proj", "linear"),
                    ]
                )
            elif layer_type == "full_attention":
                attention = f"{prefix}.attention"
                program.extend(
                    [
                        (f"{attention}.q_proj", "linear"),
                        (f"{attention}.split_query_gate", "qwen35-query-gate-split"),
                        (f"{attention}.k_proj", "linear"),
                        (f"{attention}.v_proj", "linear"),
                        (f"{attention}.q_norm", "qwen35-head-rms-norm"),
                        (f"{attention}.k_norm", "qwen35-head-rms-norm"),
                        (f"{attention}.mrope", "qwen35-partial-interleaved-mrope"),
                        (f"{attention}.gqa", "causal-grouped-query-attention"),
                        (f"{attention}.output_gate", "qwen35-attention-output-gate"),
                        (f"{attention}.o_proj", "linear"),
                    ]
                )
            else:
                raise ReferenceLoweringError("Qwen3.5 layer_types contains an unknown mixer")
            program.extend(
                [
                    (f"{prefix}.mixer_residual", "residual-add"),
                    (f"{prefix}.mlp_norm", "qwen35-rms-norm"),
                    (f"{prefix}.mlp.gate_proj", "linear"),
                    (f"{prefix}.mlp.up_proj", "linear"),
                    (f"{prefix}.mlp.silu", "silu"),
                    (f"{prefix}.mlp.multiply", "elementwise-multiply"),
                    (f"{prefix}.mlp.down_proj", "linear"),
                    (f"{prefix}.mlp_residual", "residual-add"),
                ]
            )
        program.extend([("final_norm", "qwen35-rms-norm"), ("lm_head", "linear-readout")])
        return tuple(program)
    if architecture_id == "mamba1-selective-state-space-causal-decoder":
        for layer in range(dimensions.num_hidden_layers):
            prefix = f"layers.{layer}"
            mixer = f"{prefix}.mixer"
            program.extend(
                [
                    (f"{prefix}.norm", "mamba-rms-norm"),
                    (f"{mixer}.in_proj", "linear"),
                    (f"{mixer}.split_xz", "mamba-input-gate-split"),
                    (f"{mixer}.causal_conv", "mamba-causal-depthwise-convolution"),
                    (f"{mixer}.x_proj", "linear"),
                    (f"{mixer}.split_selection", "mamba-selection-split"),
                    (f"{mixer}.dt_proj", "linear"),
                    (f"{mixer}.selective_scan", "mamba-selective-scan"),
                    (f"{mixer}.gate_activation", "silu"),
                    (f"{mixer}.gate_scan", "elementwise-multiply"),
                    (f"{mixer}.out_proj", "linear"),
                    (f"{prefix}.residual", "residual-add"),
                ]
            )
        program.extend([("final_norm", "mamba-rms-norm"), ("lm_head", "linear-readout")])
        return tuple(program)
    if architecture_id == "gpt2-causal-decoder":
        program.append(("position_embedding", "absolute-position-embedding-add"))
        for layer in range(dimensions.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            program.extend(
                [
                    (f"{prefix}.attention_norm", "layer-norm"),
                    (f"{attention}.c_attn", "conv1d-linear"),
                    (f"{attention}.split_qkv", "fused-qkv-split"),
                    (f"{attention}.mha", "causal-grouped-query-attention"),
                    (f"{attention}.c_proj", "conv1d-linear"),
                    (f"{prefix}.attention_residual", "residual-add"),
                    (f"{prefix}.mlp_norm", "layer-norm"),
                    (f"{prefix}.mlp.c_fc", "conv1d-linear"),
                    (f"{prefix}.mlp.gelu", "gelu-tanh"),
                    (f"{prefix}.mlp.c_proj", "conv1d-linear"),
                    (f"{prefix}.mlp_residual", "residual-add"),
                ]
            )
        program.extend([("final_norm", "layer-norm"), ("lm_head", "linear-readout")])
        return tuple(program)
    if architecture_id == "phi-causal-decoder":
        for layer in range(dimensions.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            program.extend(
                [
                    (f"{prefix}.input_norm", "layer-norm"),
                    (f"{attention}.q_proj", "linear"),
                    (f"{attention}.k_proj", "linear"),
                    (f"{attention}.v_proj", "linear"),
                    (f"{attention}.rotary", "rotary-partial-default"),
                    (f"{attention}.gqa", "causal-grouped-query-attention"),
                    (f"{attention}.dense", "linear"),
                    (f"{prefix}.mlp.fc1", "linear"),
                    (f"{prefix}.mlp.gelu", "gelu-tanh"),
                    (f"{prefix}.mlp.fc2", "linear"),
                    (f"{prefix}.parallel_sum", "residual-add"),
                    (f"{prefix}.residual", "residual-add"),
                ]
            )
        program.extend([("final_norm", "layer-norm"), ("lm_head", "linear-readout")])
        return tuple(program)
    if architecture_id == "gpt-neox-pythia-causal-decoder":
        for layer in range(dimensions.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            program.extend(
                [
                    (f"{prefix}.attention_norm", "layer-norm"),
                    (f"{attention}.query_key_value", "linear"),
                    (f"{attention}.unpack_qkv", "gpt-neox-qkv-unpack"),
                    (f"{attention}.rotary", "rotary-partial-default"),
                    (f"{attention}.mha", "causal-grouped-query-attention"),
                    (f"{attention}.o_proj", "linear"),
                    (f"{prefix}.attention_residual", "residual-add"),
                    (f"{prefix}.mlp_norm", "layer-norm"),
                    (f"{prefix}.mlp.dense_h_to_4h", "linear"),
                    (f"{prefix}.mlp.gelu", "gelu-erf"),
                    (f"{prefix}.mlp.dense_4h_to_h", "linear"),
                    (f"{prefix}.mlp_residual", "residual-add"),
                ]
            )
        program.extend([("final_norm", "layer-norm"), ("lm_head", "linear-readout")])
        return tuple(program)
    if architecture_id == "mixtral-sparse-moe-causal-decoder":
        experts = config.get("num_local_experts")
        if type(experts) is not int or experts <= 1:
            raise ReferenceLoweringError("Mixtral num_local_experts is invalid")
        for layer in range(dimensions.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            moe = f"{prefix}.moe"
            program.extend(
                [
                    (f"{prefix}.attention_norm", "rms-norm"),
                    (f"{attention}.q_proj", "linear"),
                    (f"{attention}.k_proj", "linear"),
                    (f"{attention}.v_proj", "linear"),
                    (f"{attention}.rotary", "rotary-default"),
                    (f"{attention}.gqa", "causal-grouped-query-attention"),
                    (f"{attention}.o_proj", "linear"),
                    (f"{prefix}.attention_residual", "residual-add"),
                    (f"{prefix}.moe_norm", "rms-norm"),
                    (f"{moe}.router", "moe-router-linear"),
                    (f"{moe}.top_k", "moe-top-k-softmax"),
                    (f"{moe}.dispatch", "moe-token-dispatch"),
                ]
            )
            for expert in range(experts):
                expert_prefix = f"{moe}.routed_experts.{expert}"
                program.extend(
                    [
                        (f"{expert_prefix}.gate_proj", "moe-routed-expert-linear"),
                        (f"{expert_prefix}.up_proj", "moe-routed-expert-linear"),
                        (f"{expert_prefix}.silu", "silu"),
                        (f"{expert_prefix}.multiply", "elementwise-multiply"),
                        (f"{expert_prefix}.down_proj", "moe-routed-expert-linear"),
                    ]
                )
            program.extend(
                [
                    (f"{moe}.combine", "moe-weighted-scatter-add"),
                    (f"{prefix}.moe_residual", "residual-add"),
                ]
            )
        program.extend([("final_norm", "rms-norm"), ("lm_head", "linear-readout")])
        return tuple(program)
    gemma = architecture_id == "gemma1-dense-causal-decoder"
    if gemma:
        program.append(("embedding_scale", "scalar-multiply"))
    qk_norm = architecture_id == "qwen3-dense-causal-decoder"
    norm_kind = "gemma-rms-norm" if gemma else "rms-norm"
    activation_kind = "gelu-tanh" if gemma else "silu"
    for layer in range(dimensions.num_hidden_layers):
        prefix = f"layers.{layer}"
        attention = f"{prefix}.attention"
        program.extend(
            [
                (f"{prefix}.attention_norm", norm_kind),
                (f"{attention}.q_proj", "linear"),
                (f"{attention}.k_proj", "linear"),
                (f"{attention}.v_proj", "linear"),
            ]
        )
        if qk_norm:
            program.extend(
                [
                    (f"{attention}.q_norm", "head-rms-norm"),
                    (f"{attention}.k_norm", "head-rms-norm"),
                ]
            )
        program.extend(
            [
                (f"{attention}.rotary", "rotary-default"),
                (f"{attention}.gqa", "causal-grouped-query-attention"),
                (f"{attention}.o_proj", "linear"),
                (f"{prefix}.attention_residual", "residual-add"),
                (f"{prefix}.mlp_norm", norm_kind),
                (f"{prefix}.mlp.gate_proj", "linear"),
                (f"{prefix}.mlp.up_proj", "linear"),
                (f"{prefix}.mlp.gelu" if gemma else f"{prefix}.mlp.silu", activation_kind),
                (f"{prefix}.mlp.multiply", "elementwise-multiply"),
                (f"{prefix}.mlp.down_proj", "linear"),
                (f"{prefix}.mlp_residual", "residual-add"),
            ]
        )
    program.extend([("final_norm", norm_kind), ("lm_head", "linear-readout")])
    return tuple(program)


def _gpt2_config_semantics(
    dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> tuple[float, bool]:
    expected = {
        "n_embd": dimensions.hidden_size,
        "n_layer": dimensions.num_hidden_layers,
        "n_head": dimensions.num_attention_heads,
        "n_positions": dimensions.max_position_embeddings,
        "vocab_size": dimensions.vocab_size,
    }
    for field, value in expected.items():
        if type(config.get(field)) is not int or config[field] != value:
            raise ReferenceLoweringError(f"GPT-2 config field {field!r} differs from ModelIR")
    if (
        config.get("n_ctx", dimensions.max_position_embeddings)
        != dimensions.max_position_embeddings
    ):
        raise ReferenceLoweringError("GPT-2 n_ctx differs from ModelIR position capacity")
    inner = config.get("n_inner")
    if (4 * dimensions.hidden_size if inner is None else inner) != dimensions.intermediate_size:
        raise ReferenceLoweringError("GPT-2 MLP width differs from ModelIR")
    if (
        dimensions.num_key_value_heads != dimensions.num_attention_heads
        or dimensions.head_dim * dimensions.num_attention_heads != dimensions.hidden_size
    ):
        raise ReferenceLoweringError("GPT-2 ModelIR attention dimensions are inconsistent")
    if config.get("activation_function", "gelu_new") != "gelu_new":
        raise ReferenceLoweringError("GPT-2 reference target requires gelu_new")
    for field, default in (
        ("scale_attn_weights", True),
        ("scale_attn_by_inverse_layer_idx", False),
        ("reorder_and_upcast_attn", False),
    ):
        if config.get(field, default) is not default:
            raise ReferenceLoweringError(f"GPT-2 config field {field!r} is outside the target")
    epsilon = _finite_positive(config.get("layer_norm_epsilon", 1e-5), field="layer_norm_epsilon")
    tied = config.get("tie_word_embeddings", True)
    if type(tied) is not bool:
        raise ReferenceLoweringError("GPT-2 tie_word_embeddings must be boolean")
    return epsilon, tied


def _validate_gpt2_bindings(
    operations: tuple[OperationIR, ...], dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> None:
    epsilon, _tied = _gpt2_config_semantics(dimensions, config)
    expected: list[
        tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]
    ] = []

    def add(
        operation_id: str,
        kind: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        parameters: tuple[str, ...] = (),
        attributes: dict[str, Any] | None = None,
    ) -> None:
        expected.append(
            (
                operation_id,
                kind,
                inputs,
                outputs,
                tuple(sorted(parameters)),
                attributes or {},
            )
        )

    add(
        "embedding",
        "token-embedding",
        ("token_ids",),
        ("embedding.token",),
        ("token_embedding.weight",),
        {"padding_policy": "IOIR-row-mapper"},
    )
    add(
        "position_embedding",
        "absolute-position-embedding-add",
        ("embedding.token",),
        ("embedding.hidden",),
        ("position_embedding.weight",),
        {"max_position_embeddings": dimensions.max_position_embeddings},
    )
    current = "embedding.hidden"
    for layer in range(dimensions.num_hidden_layers):
        prefix = f"layers.{layer}"
        attention = f"{prefix}.attention"
        add(
            f"{prefix}.attention_norm",
            "layer-norm",
            (current,),
            (f"{prefix}.attention_norm.hidden",),
            (f"{prefix}.attention_norm.weight", f"{prefix}.attention_norm.bias"),
            {"epsilon": epsilon},
        )
        add(
            f"{attention}.c_attn",
            "conv1d-linear",
            (f"{prefix}.attention_norm.hidden",),
            (f"{attention}.packed_qkv",),
            (f"{prefix}.attn.c_attn.weight", f"{prefix}.attn.c_attn.bias"),
            {"weight_orientation": "in-out"},
        )
        add(
            f"{attention}.split_qkv",
            "fused-qkv-split",
            (f"{attention}.packed_qkv",),
            (f"{attention}.q", f"{attention}.k", f"{attention}.v"),
            attributes={"split_width": dimensions.hidden_size},
        )
        add(
            f"{attention}.mha",
            "causal-grouped-query-attention",
            (f"{attention}.q", f"{attention}.k", f"{attention}.v"),
            (f"{attention}.context",),
            (f"{attention}.causal_mask",),
            {
                "head_dim": dimensions.head_dim,
                "num_attention_heads": dimensions.num_attention_heads,
                "num_key_value_heads": dimensions.num_key_value_heads,
                "scale": dimensions.head_dim**-0.5,
                "state_slots": [
                    f"layers.{layer}.k_cache",
                    f"layers.{layer}.v_cache",
                    "position",
                ],
            },
        )
        add(
            f"{attention}.c_proj",
            "conv1d-linear",
            (f"{attention}.context",),
            (f"{attention}.output",),
            (f"{prefix}.attn.c_proj.weight", f"{prefix}.attn.c_proj.bias"),
            {"weight_orientation": "in-out"},
        )
        add(
            f"{prefix}.attention_residual",
            "residual-add",
            (current, f"{attention}.output"),
            (f"{prefix}.attention_residual.hidden",),
        )
        add(
            f"{prefix}.mlp_norm",
            "layer-norm",
            (f"{prefix}.attention_residual.hidden",),
            (f"{prefix}.mlp_norm.hidden",),
            (f"{prefix}.mlp_norm.weight", f"{prefix}.mlp_norm.bias"),
            {"epsilon": epsilon},
        )
        add(
            f"{prefix}.mlp.c_fc",
            "conv1d-linear",
            (f"{prefix}.mlp_norm.hidden",),
            (f"{prefix}.mlp.expanded",),
            (f"{prefix}.mlp.c_fc.weight", f"{prefix}.mlp.c_fc.bias"),
            {"weight_orientation": "in-out"},
        )
        add(
            f"{prefix}.mlp.gelu",
            "gelu-tanh",
            (f"{prefix}.mlp.expanded",),
            (f"{prefix}.mlp.activated",),
        )
        add(
            f"{prefix}.mlp.c_proj",
            "conv1d-linear",
            (f"{prefix}.mlp.activated",),
            (f"{prefix}.mlp.output",),
            (f"{prefix}.mlp.c_proj.weight", f"{prefix}.mlp.c_proj.bias"),
            {"weight_orientation": "in-out"},
        )
        next_hidden = f"{prefix}.output"
        add(
            f"{prefix}.mlp_residual",
            "residual-add",
            (f"{prefix}.attention_residual.hidden", f"{prefix}.mlp.output"),
            (next_hidden,),
        )
        current = next_hidden
    add(
        "final_norm",
        "layer-norm",
        (current,),
        ("final.hidden",),
        ("final_norm.weight", "final_norm.bias"),
        {"epsilon": epsilon},
    )
    add(
        "lm_head",
        "linear-readout",
        ("final.hidden",),
        ("logits",),
        ("lm_head.weight",),
        {"weight_orientation": "rows-hidden"},
    )
    observed = [
        (
            item.operation_id,
            item.kind,
            item.inputs,
            item.outputs,
            item.parameters,
            item.attributes,
        )
        for item in operations
    ]
    if observed != expected:
        raise ReferenceLoweringError(
            "GPT-2 IR dataflow or parameter bindings differ from the exact target contract"
        )


def _gpt_neox_config_semantics(
    dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> tuple[bool, bool, float, float, int]:
    integer_fields = {
        "hidden_size": dimensions.hidden_size,
        "intermediate_size": dimensions.intermediate_size,
        "num_hidden_layers": dimensions.num_hidden_layers,
        "num_attention_heads": dimensions.num_attention_heads,
        "vocab_size": dimensions.vocab_size,
        "max_position_embeddings": dimensions.max_position_embeddings,
    }
    for field, expected in integer_fields.items():
        if type(config.get(field)) is not int or config[field] != expected:
            raise ReferenceLoweringError(f"GPT-NeoX config field {field!r} differs from ModelIR")
    if dimensions.head_dim != dimensions.hidden_size // dimensions.num_attention_heads:
        raise ReferenceLoweringError("GPT-NeoX ModelIR head width is inconsistent")
    kv_heads = config.get("num_key_value_heads", dimensions.num_attention_heads)
    if kv_heads is None:
        kv_heads = dimensions.num_attention_heads
    if kv_heads != dimensions.num_attention_heads or dimensions.num_key_value_heads != kv_heads:
        raise ReferenceLoweringError("GPT-NeoX fused QKV requires symmetric attention heads")
    if config.get("hidden_act", "gelu") != "gelu":
        raise ReferenceLoweringError("GPT-NeoX reference target requires exact erf GELU")
    if config.get("tie_word_embeddings", False) is not False:
        raise ReferenceLoweringError("GPT-NeoX reference target requires untied lexical matrices")
    attention_bias = config.get("attention_bias", True)
    parallel = config.get("use_parallel_residual", True)
    if type(attention_bias) is not bool or type(parallel) is not bool:
        raise ReferenceLoweringError("GPT-NeoX bias and residual flags must be booleans")
    epsilon = _finite_positive(config.get("layer_norm_eps", 1e-5), field="layer_norm_eps")
    if config.get("rope_scaling") is not None:
        raise ReferenceLoweringError("GPT-NeoX reference target requires unscaled default RoPE")
    rope = config.get("rope_parameters") or {}
    if not isinstance(rope, dict) or set(rope) - {
        "partial_rotary_factor",
        "rope_theta",
        "rope_type",
    }:
        raise ReferenceLoweringError("GPT-NeoX rope_parameters are outside the target contract")
    if rope.get("rope_type", "default") != "default":
        raise ReferenceLoweringError("GPT-NeoX reference target requires default RoPE")
    theta = _finite_positive(
        rope.get("rope_theta", config.get("rotary_emb_base", 10_000.0)),
        field="rotary_emb_base",
    )
    pct = rope.get("partial_rotary_factor", config.get("rotary_pct", 0.25))
    if type(pct) not in {int, float} or not math.isfinite(float(pct)) or not 0 < float(pct) <= 1:
        raise ReferenceLoweringError("GPT-NeoX rotary fraction is outside the target contract")
    rotary_dim = int(dimensions.head_dim * float(pct))
    if rotary_dim <= 0 or rotary_dim > dimensions.head_dim or rotary_dim % 2:
        raise ReferenceLoweringError("GPT-NeoX rotary dimension is invalid")
    return attention_bias, parallel, epsilon, theta, rotary_dim


def _validate_gpt_neox_bindings(
    operations: tuple[OperationIR, ...], dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> None:
    attention_bias, parallel, epsilon, theta, rotary_dim = _gpt_neox_config_semantics(
        dimensions, config
    )
    expected: list[
        tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]
    ] = []

    def add(
        operation_id: str,
        kind: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        parameters: tuple[str, ...] = (),
        attributes: dict[str, Any] | None = None,
    ) -> None:
        expected.append(
            (
                operation_id,
                kind,
                inputs,
                outputs,
                tuple(sorted(parameters)),
                attributes or {},
            )
        )

    add(
        "embedding",
        "token-embedding",
        ("token_ids",),
        ("embedding.hidden",),
        ("token_embedding.weight",),
        {"padding_policy": "IOIR-row-mapper"},
    )
    current = "embedding.hidden"
    for layer in range(dimensions.num_hidden_layers):
        prefix = f"layers.{layer}"
        attention = f"{prefix}.attention"
        add(
            f"{prefix}.attention_norm",
            "layer-norm",
            (current,),
            (f"{prefix}.attention_norm.hidden",),
            (f"{prefix}.attention_norm.weight", f"{prefix}.attention_norm.bias"),
            {"epsilon": epsilon},
        )
        qkv_parameters = [f"{attention}.query_key_value.weight"]
        if attention_bias:
            qkv_parameters.append(f"{attention}.query_key_value.bias")
        add(
            f"{attention}.query_key_value",
            "linear",
            (f"{prefix}.attention_norm.hidden",),
            (f"{attention}.packed_qkv",),
            tuple(qkv_parameters),
            {"weight_orientation": "out-in"},
        )
        add(
            f"{attention}.unpack_qkv",
            "gpt-neox-qkv-unpack",
            (f"{attention}.packed_qkv",),
            (f"{attention}.q", f"{attention}.k", f"{attention}.v"),
            attributes={
                "head_dim": dimensions.head_dim,
                "num_attention_heads": dimensions.num_attention_heads,
                "packing": "per-head-qkv",
            },
        )
        add(
            f"{attention}.rotary",
            "rotary-partial-default",
            (f"{attention}.q", f"{attention}.k"),
            (f"{attention}.q_rotary", f"{attention}.k_rotary"),
            attributes={
                "head_dim": dimensions.head_dim,
                "max_position_embeddings": dimensions.max_position_embeddings,
                "rope_theta": theta,
                "rotary_dim": rotary_dim,
            },
        )
        add(
            f"{attention}.mha",
            "causal-grouped-query-attention",
            (f"{attention}.q_rotary", f"{attention}.k_rotary", f"{attention}.v"),
            (f"{attention}.context",),
            attributes={
                "head_dim": dimensions.head_dim,
                "num_attention_heads": dimensions.num_attention_heads,
                "num_key_value_heads": dimensions.num_key_value_heads,
                "scale": dimensions.head_dim**-0.5,
                "state_slots": [
                    f"layers.{layer}.k_cache",
                    f"layers.{layer}.v_cache",
                    "position",
                ],
            },
        )
        output_parameters = [f"{attention}.o_proj.weight"]
        if attention_bias:
            output_parameters.append(f"{attention}.o_proj.bias")
        add(
            f"{attention}.o_proj",
            "linear",
            (f"{attention}.context",),
            (f"{attention}.output",),
            tuple(output_parameters),
            {"weight_orientation": "out-in"},
        )
        attention_residual = f"{prefix}.attention_residual.hidden"
        add(
            f"{prefix}.attention_residual",
            "residual-add",
            (current, f"{attention}.output"),
            (attention_residual,),
        )
        add(
            f"{prefix}.mlp_norm",
            "layer-norm",
            (current if parallel else attention_residual,),
            (f"{prefix}.mlp_norm.hidden",),
            (f"{prefix}.mlp_norm.weight", f"{prefix}.mlp_norm.bias"),
            {"epsilon": epsilon},
        )
        add(
            f"{prefix}.mlp.dense_h_to_4h",
            "linear",
            (f"{prefix}.mlp_norm.hidden",),
            (f"{prefix}.mlp.expanded",),
            (
                f"{prefix}.mlp.dense_h_to_4h.weight",
                f"{prefix}.mlp.dense_h_to_4h.bias",
            ),
            {"weight_orientation": "out-in"},
        )
        add(
            f"{prefix}.mlp.gelu",
            "gelu-erf",
            (f"{prefix}.mlp.expanded",),
            (f"{prefix}.mlp.activated",),
        )
        add(
            f"{prefix}.mlp.dense_4h_to_h",
            "linear",
            (f"{prefix}.mlp.activated",),
            (f"{prefix}.mlp.output",),
            (
                f"{prefix}.mlp.dense_4h_to_h.weight",
                f"{prefix}.mlp.dense_4h_to_h.bias",
            ),
            {"weight_orientation": "out-in"},
        )
        current = f"{prefix}.output"
        add(
            f"{prefix}.mlp_residual",
            "residual-add",
            (attention_residual, f"{prefix}.mlp.output"),
            (current,),
        )
    add(
        "final_norm",
        "layer-norm",
        (current,),
        ("final.hidden",),
        ("final_norm.weight", "final_norm.bias"),
        {"epsilon": epsilon},
    )
    add(
        "lm_head",
        "linear-readout",
        ("final.hidden",),
        ("logits",),
        ("lm_head.weight",),
        {"weight_orientation": "rows-hidden"},
    )
    observed = [
        (
            item.operation_id,
            item.kind,
            item.inputs,
            item.outputs,
            item.parameters,
            item.attributes,
        )
        for item in operations
    ]
    if observed != expected:
        raise ReferenceLoweringError(
            "GPT-NeoX IR dataflow or parameter bindings differ from the exact target contract"
        )


def _mamba_config_semantics(
    dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> tuple[int, int, int, float, bool, bool, bool, bool]:
    for field, expected in (
        ("hidden_size", dimensions.hidden_size),
        ("num_hidden_layers", dimensions.num_hidden_layers),
        ("vocab_size", dimensions.vocab_size),
    ):
        if type(config.get(field)) is not int or config[field] != expected:
            raise ReferenceLoweringError(
                "Mamba source config differs from ModelIR dimensions",
                details={"field": field, "expected": expected, "actual": config.get(field)},
            )
    if (
        dimensions.num_attention_heads != 0
        or dimensions.num_key_value_heads != 0
        or dimensions.head_dim != 0
        or dimensions.max_position_embeddings != 0
    ):
        raise ReferenceLoweringError("Mamba ModelIR invents attention or positional dimensions")
    for alias, expected in (
        ("d_model", dimensions.hidden_size),
        ("n_layer", dimensions.num_hidden_layers),
    ):
        if alias in config and (type(config[alias]) is not int or config[alias] != expected):
            raise ReferenceLoweringError(f"Mamba config field {alias!r} differs from ModelIR")

    expand = config.get("expand", 2)
    if type(expand) is not int or expand <= 0:
        raise ReferenceLoweringError("Mamba expand must be a positive integer")
    raw_intermediate = config.get("intermediate_size", config.get("d_inner"))
    intermediate = dimensions.hidden_size * expand if raw_intermediate is None else raw_intermediate
    if type(intermediate) is not int or intermediate != dimensions.intermediate_size:
        raise ReferenceLoweringError("Mamba intermediate width differs from ModelIR")
    if "d_inner" in config and (
        type(config["d_inner"]) is not int or config["d_inner"] != dimensions.intermediate_size
    ):
        raise ReferenceLoweringError("Mamba d_inner differs from ModelIR")
    if dimensions.intermediate_size != dimensions.hidden_size * expand:
        raise ReferenceLoweringError("Mamba ModelIR intermediate width differs from expand")

    state_size = config.get("state_size")
    conv_kernel = config.get("conv_kernel")
    if type(state_size) is not int or state_size <= 0:
        raise ReferenceLoweringError("Mamba state_size must be a positive integer")
    if type(conv_kernel) is not int or conv_kernel <= 0:
        raise ReferenceLoweringError("Mamba conv_kernel must be a positive integer")
    raw_rank = config.get("time_step_rank", "auto")
    rank = math.ceil(dimensions.hidden_size / 16) if raw_rank == "auto" else raw_rank
    if type(rank) is not int or rank <= 0:
        raise ReferenceLoweringError("Mamba time_step_rank is outside the target contract")
    epsilon = _finite_positive(
        config.get("layer_norm_epsilon", 1e-5), field="Mamba layer_norm_epsilon"
    )
    if (
        config.get("hidden_act", "silu") != "silu"
        or config.get("rms_norm", True) is not True
        or config.get("ssm_cfg", {}) not in ({}, None)
        or config.get("mixer_rms_eps") is not None
        or config.get("use_mambapy", False) is not False
    ):
        raise ReferenceLoweringError("Mamba source config crosses the registered semantic lane")
    if config.get("architectures") != ["MambaForCausalLM"]:
        raise ReferenceLoweringError("Mamba architecture declaration is outside the target")

    boolean_values: dict[str, bool] = {}
    for field, default in (
        ("tie_word_embeddings", True),
        ("use_bias", False),
        ("use_conv_bias", True),
        ("residual_in_fp32", True),
        ("fused_add_norm", False),
        ("rescale_prenorm_residual", False),
        ("use_associative_scan", True),
        ("use_cache", True),
    ):
        value = config.get(field, default)
        if type(value) is not bool:
            raise ReferenceLoweringError(f"Mamba config field {field!r} must be boolean")
        boolean_values[field] = value
    if config.get("time_step_init_scheme", "random") not in {"random", "constant"}:
        raise ReferenceLoweringError("Mamba time-step initialization scheme is invalid")
    time_step_min = _finite_positive(
        config.get("time_step_min", 0.001), field="Mamba time_step_min"
    )
    time_step_max = _finite_positive(config.get("time_step_max", 0.1), field="Mamba time_step_max")
    if time_step_min > time_step_max:
        raise ReferenceLoweringError("Mamba time_step_min exceeds time_step_max")
    _finite_positive(config.get("time_step_floor", 0.0001), field="Mamba time_step_floor")
    _finite_positive(config.get("time_step_scale", 1.0), field="Mamba time_step_scale")
    pad_multiple = config.get("pad_vocab_size_multiple", 1)
    if type(pad_multiple) is not int or pad_multiple <= 0:
        raise ReferenceLoweringError("Mamba pad_vocab_size_multiple must be positive")
    return (
        state_size,
        conv_kernel,
        rank,
        epsilon,
        boolean_values["tie_word_embeddings"],
        boolean_values["use_bias"],
        boolean_values["use_conv_bias"],
        boolean_values["residual_in_fp32"],
    )


def _validate_mamba_bindings(
    operations: tuple[OperationIR, ...], dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> None:
    state_size, conv_kernel, rank, epsilon, _, use_bias, use_conv_bias, residual_fp32 = (
        _mamba_config_semantics(dimensions, config)
    )
    expected: list[
        tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]
    ] = []

    def add(
        operation_id: str,
        kind: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        parameters: tuple[str, ...] = (),
        attributes: dict[str, Any] | None = None,
    ) -> None:
        expected.append(
            (operation_id, kind, inputs, outputs, tuple(sorted(parameters)), attributes or {})
        )

    add(
        "embedding",
        "token-embedding",
        ("token_ids",),
        ("embedding.hidden",),
        ("token_embedding.weight",),
        {"padding_policy": "IOIR-row-mapper"},
    )
    current = "embedding.hidden"
    for layer in range(dimensions.num_hidden_layers):
        prefix = f"layers.{layer}"
        mixer = f"{prefix}.mixer"
        add(
            f"{prefix}.norm",
            "mamba-rms-norm",
            (current,),
            (f"{prefix}.norm.hidden",),
            (f"{prefix}.norm.weight",),
            {"epsilon": epsilon, "accumulation": "float32"},
        )
        in_projection = [f"{mixer}.in_proj.weight"]
        if use_bias:
            in_projection.append(f"{mixer}.in_proj.bias")
        add(
            f"{mixer}.in_proj",
            "linear",
            (f"{prefix}.norm.hidden",),
            (f"{mixer}.packed_xz",),
            tuple(in_projection),
            {"weight_orientation": "out-in"},
        )
        add(
            f"{mixer}.split_xz",
            "mamba-input-gate-split",
            (f"{mixer}.packed_xz",),
            (f"{mixer}.x", f"{mixer}.gate"),
            attributes={"split_width": dimensions.intermediate_size},
        )
        convolution = [f"{mixer}.conv1d.weight"]
        if use_conv_bias:
            convolution.append(f"{mixer}.conv1d.bias")
        add(
            f"{mixer}.causal_conv",
            "mamba-causal-depthwise-convolution",
            (f"{mixer}.x",),
            (f"{mixer}.convolved",),
            tuple(convolution),
            {
                "activation": "silu",
                "kernel_size": conv_kernel,
                "state_slot": f"{prefix}.conv_state",
            },
        )
        add(
            f"{mixer}.x_proj",
            "linear",
            (f"{mixer}.convolved",),
            (f"{mixer}.packed_selection",),
            (f"{mixer}.x_proj.weight",),
            {"weight_orientation": "out-in"},
        )
        add(
            f"{mixer}.split_selection",
            "mamba-selection-split",
            (f"{mixer}.packed_selection",),
            (f"{mixer}.dt_input", f"{mixer}.B", f"{mixer}.C"),
            attributes={"state_size": state_size, "time_step_rank": rank},
        )
        add(
            f"{mixer}.dt_proj",
            "linear",
            (f"{mixer}.dt_input",),
            (f"{mixer}.dt_pre_softplus",),
            (f"{mixer}.dt_proj.weight", f"{mixer}.dt_proj.bias"),
            {"weight_orientation": "out-in"},
        )
        add(
            f"{mixer}.selective_scan",
            "mamba-selective-scan",
            (
                f"{mixer}.convolved",
                f"{mixer}.dt_pre_softplus",
                f"{mixer}.B",
                f"{mixer}.C",
            ),
            (f"{mixer}.scan_output",),
            (f"{mixer}.A_log", f"{mixer}.D"),
            {
                "dt_activation": "softplus",
                "state_size": state_size,
                "state_slot": f"{prefix}.recurrent_state",
                "state_update": "prefix-indexed-provisional",
            },
        )
        add(
            f"{mixer}.gate_activation",
            "silu",
            (f"{mixer}.gate",),
            (f"{mixer}.activated_gate",),
        )
        add(
            f"{mixer}.gate_scan",
            "elementwise-multiply",
            (f"{mixer}.scan_output", f"{mixer}.activated_gate"),
            (f"{mixer}.gated_scan",),
        )
        output_projection = [f"{mixer}.out_proj.weight"]
        if use_bias:
            output_projection.append(f"{mixer}.out_proj.bias")
        add(
            f"{mixer}.out_proj",
            "linear",
            (f"{mixer}.gated_scan",),
            (f"{mixer}.output",),
            tuple(output_projection),
            {"weight_orientation": "out-in"},
        )
        next_hidden = f"{prefix}.output"
        add(
            f"{prefix}.residual",
            "residual-add",
            (current, f"{mixer}.output"),
            (next_hidden,),
            attributes={
                "residual_accumulation": "float32" if residual_fp32 else "activation-dtype"
            },
        )
        current = next_hidden
    add(
        "final_norm",
        "mamba-rms-norm",
        (current,),
        ("final.hidden",),
        ("final_norm.weight",),
        {"epsilon": epsilon, "accumulation": "float32"},
    )
    add(
        "lm_head",
        "linear-readout",
        ("final.hidden",),
        ("logits",),
        ("lm_head.weight",),
        {"weight_orientation": "rows-hidden"},
    )
    observed = [
        (
            item.operation_id,
            item.kind,
            item.inputs,
            item.outputs,
            item.parameters,
            item.attributes,
        )
        for item in operations
    ]
    if observed != expected:
        raise ReferenceLoweringError(
            "Mamba IR dataflow or parameter bindings differ from the exact target contract"
        )


def _mixtral_config_semantics(
    dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> tuple[int, int, float, float]:
    expected_dimensions = {
        "hidden_size": dimensions.hidden_size,
        "intermediate_size": dimensions.intermediate_size,
        "num_hidden_layers": dimensions.num_hidden_layers,
        "num_attention_heads": dimensions.num_attention_heads,
        "num_key_value_heads": dimensions.num_key_value_heads,
        "head_dim": dimensions.head_dim,
        "max_position_embeddings": dimensions.max_position_embeddings,
        "vocab_size": dimensions.vocab_size,
    }
    for field, expected in expected_dimensions.items():
        if type(config.get(field)) is not int or config[field] != expected:
            raise ReferenceLoweringError(
                "Mixtral source config differs from ModelIR dimensions",
                details={"field": field, "expected": expected, "actual": config.get(field)},
            )
    experts = config.get("num_local_experts")
    top_k = config.get("num_experts_per_tok")
    if (
        type(experts) is not int
        or type(top_k) is not int
        or experts <= 1
        or not 0 < top_k < experts
    ):
        raise ReferenceLoweringError("Mixtral source config has an invalid expert topology")
    if (
        config.get("attention_bias") is not False
        or config.get("hidden_act") != "silu"
        or config.get("output_router_logits") is not False
        or config.get("sliding_window") is not None
    ):
        raise ReferenceLoweringError("Mixtral source config crosses the registered semantic lane")
    jitter = config.get("router_jitter_noise")
    if type(jitter) not in {int, float} or float(jitter) != 0.0:
        raise ReferenceLoweringError("Mixtral reference execution requires deterministic routing")
    epsilon = _finite_positive(config.get("rms_norm_eps"), field="Mixtral RMSNorm epsilon")
    rope = config.get("rope_parameters") or {}
    if not isinstance(rope, dict):
        raise ReferenceLoweringError("Mixtral rope_parameters must be an object")
    theta = _finite_positive(
        rope.get("rope_theta", config.get("rope_theta", 10_000.0)),
        field="Mixtral rope_theta",
    )
    return experts, top_k, epsilon, theta


def _validate_mixtral_bindings(
    operations: Sequence[OperationIR], dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> None:
    experts, top_k, epsilon, _ = _mixtral_config_semantics(dimensions, config)
    by_id = {operation.operation_id: operation for operation in operations}

    def require(
        operation_id: str,
        *,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        parameters: tuple[str, ...],
        attributes: dict[str, Any] | None = None,
    ) -> None:
        operation = by_id[operation_id]
        if (
            operation.inputs != inputs
            or operation.outputs != outputs
            or operation.parameters != parameters
            or (attributes is not None and operation.attributes != attributes)
        ):
            raise ReferenceLoweringError(
                "Mixtral IR dataflow or parameter bindings differ from the exact target contract",
                details={"operation_id": operation_id},
            )

    for layer in range(dimensions.num_hidden_layers):
        prefix = f"layers.{layer}"
        residual = f"{prefix}.attention_residual.hidden"
        moe = f"{prefix}.moe"
        require(
            f"{prefix}.moe_norm",
            inputs=(residual,),
            outputs=(f"{moe}.input",),
            parameters=(f"{prefix}.moe_norm.weight",),
            attributes={"epsilon": epsilon},
        )
        require(
            f"{moe}.router",
            inputs=(f"{moe}.input",),
            outputs=(f"{moe}.router_logits",),
            parameters=(f"{moe}.router.weight",),
            attributes={
                "expert_scope": "routed-only",
                "num_routed_experts": experts,
                "num_shared_experts": 0,
                "router_bias": False,
                "weight_orientation": "experts-hidden",
            },
        )
        require(
            f"{moe}.top_k",
            inputs=(f"{moe}.router_logits",),
            outputs=(f"{moe}.routing_weights", f"{moe}.selected_experts"),
            parameters=(),
            attributes={
                "jitter_noise": 0.0,
                "num_experts_per_token": top_k,
                "num_routed_experts": experts,
                "renormalize_selected_probabilities": True,
                "selection": "top-k-after-softmax",
                "softmax_dtype": "float32",
                "tie_breaking": "source-framework-defined",
            },
        )
        expert_inputs = tuple(f"{moe}.routed_experts.{expert}.input" for expert in range(experts))
        require(
            f"{moe}.dispatch",
            inputs=(f"{moe}.input", f"{moe}.selected_experts"),
            outputs=expert_inputs,
            parameters=(),
            attributes={
                "expert_scope": "routed-only",
                "num_routed_experts": experts,
                "num_shared_experts": 0,
            },
        )
        expert_outputs: list[str] = []
        for expert in range(experts):
            expert_prefix = f"{moe}.routed_experts.{expert}"
            expert_input = f"{expert_prefix}.input"
            for projection in ("gate_proj", "up_proj"):
                require(
                    f"{expert_prefix}.{projection}",
                    inputs=(expert_input,),
                    outputs=(f"{expert_prefix}.{projection}.hidden",),
                    parameters=(f"{expert_prefix}.{projection}.weight",),
                    attributes={
                        "expert_id": expert,
                        "expert_scope": "routed",
                        "weight_orientation": "out-in",
                    },
                )
            require(
                f"{expert_prefix}.silu",
                inputs=(f"{expert_prefix}.gate_proj.hidden",),
                outputs=(f"{expert_prefix}.gate_activated",),
                parameters=(),
                attributes={},
            )
            require(
                f"{expert_prefix}.multiply",
                inputs=(
                    f"{expert_prefix}.gate_activated",
                    f"{expert_prefix}.up_proj.hidden",
                ),
                outputs=(f"{expert_prefix}.intermediate",),
                parameters=(),
                attributes={},
            )
            expert_output = f"{expert_prefix}.output"
            require(
                f"{expert_prefix}.down_proj",
                inputs=(f"{expert_prefix}.intermediate",),
                outputs=(expert_output,),
                parameters=(f"{expert_prefix}.down_proj.weight",),
                attributes={
                    "expert_id": expert,
                    "expert_scope": "routed",
                    "weight_orientation": "out-in",
                },
            )
            expert_outputs.append(expert_output)
        require(
            f"{moe}.combine",
            inputs=(
                f"{moe}.routing_weights",
                f"{moe}.selected_experts",
                *expert_outputs,
            ),
            outputs=(f"{moe}.output",),
            parameters=(),
            attributes={
                "accumulation_order": "source-expert-index-order",
                "expert_scope": "routed-only",
                "num_routed_experts": experts,
                "num_shared_experts": 0,
            },
        )
        require(
            f"{prefix}.moe_residual",
            inputs=(residual, f"{moe}.output"),
            outputs=(f"{prefix}.output",),
            parameters=(),
            attributes={},
        )


def _expected_parameter_shapes(
    dimensions: ModelDimensionsIR,
    *,
    architecture_id: str,
    config: dict[str, Any],
    input_rows: int,
    output_rows: int,
) -> dict[str, tuple[int, ...]]:
    d = dimensions
    if architecture_id == "qwen3_5-hybrid-text-causal-decoder":
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise ReferenceLoweringError("Qwen3.5 text_config is absent")
        layer_types = text.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != d.num_hidden_layers:
            raise ReferenceLoweringError("Qwen3.5 layer_types differs from ModelIR")
        key_heads = int(text["linear_num_key_heads"])
        value_heads = int(text["linear_num_value_heads"])
        key_dim = int(text["linear_key_head_dim"])
        value_dim = int(text["linear_value_head_dim"])
        conv_kernel = int(text["linear_conv_kernel_dim"])
        key_width = key_heads * key_dim
        value_width = value_heads * value_dim
        conv_width = 2 * key_width + value_width
        shapes: dict[str, tuple[int, ...]] = {
            "token_embedding.weight": (input_rows, d.hidden_size),
            "lm_head.weight": (output_rows, d.hidden_size),
            "final_norm.weight": (d.hidden_size,),
        }
        for layer, layer_type in enumerate(layer_types):
            prefix = f"layers.{layer}"
            shapes[f"{prefix}.mixer_norm.weight"] = (d.hidden_size,)
            shapes[f"{prefix}.mlp_norm.weight"] = (d.hidden_size,)
            if layer_type == "linear_attention":
                mixer = f"{prefix}.gated_delta"
                shapes[f"{mixer}.in_proj_qkv.weight"] = (conv_width, d.hidden_size)
                shapes[f"{mixer}.in_proj_z.weight"] = (value_width, d.hidden_size)
                shapes[f"{mixer}.in_proj_a.weight"] = (value_heads, d.hidden_size)
                shapes[f"{mixer}.in_proj_b.weight"] = (value_heads, d.hidden_size)
                shapes[f"{mixer}.conv1d.weight"] = (conv_width, 1, conv_kernel)
                shapes[f"{mixer}.A_log"] = (value_heads,)
                shapes[f"{mixer}.dt_bias"] = (value_heads,)
                shapes[f"{mixer}.norm.weight"] = (value_dim,)
                shapes[f"{mixer}.out_proj.weight"] = (d.hidden_size, value_width)
            elif layer_type == "full_attention":
                attention = f"{prefix}.attention"
                shapes[f"{attention}.q_proj.weight"] = (
                    2 * d.num_attention_heads * d.head_dim,
                    d.hidden_size,
                )
                shapes[f"{attention}.k_proj.weight"] = (
                    d.num_key_value_heads * d.head_dim,
                    d.hidden_size,
                )
                shapes[f"{attention}.v_proj.weight"] = (
                    d.num_key_value_heads * d.head_dim,
                    d.hidden_size,
                )
                shapes[f"{attention}.o_proj.weight"] = (
                    d.hidden_size,
                    d.num_attention_heads * d.head_dim,
                )
                shapes[f"{attention}.q_norm.weight"] = (d.head_dim,)
                shapes[f"{attention}.k_norm.weight"] = (d.head_dim,)
            else:
                raise ReferenceLoweringError("Qwen3.5 layer_types contains an unknown mixer")
            shapes[f"{prefix}.mlp.gate_proj.weight"] = (d.intermediate_size, d.hidden_size)
            shapes[f"{prefix}.mlp.up_proj.weight"] = (d.intermediate_size, d.hidden_size)
            shapes[f"{prefix}.mlp.down_proj.weight"] = (d.hidden_size, d.intermediate_size)
        return shapes
    if architecture_id == "mamba1-selective-state-space-causal-decoder":
        state_size, conv_kernel, rank, _, _, use_bias, use_conv_bias, _ = _mamba_config_semantics(
            d, config
        )
        shapes: dict[str, tuple[int, ...]] = {
            "token_embedding.weight": (input_rows, d.hidden_size),
            "lm_head.weight": (output_rows, d.hidden_size),
            "final_norm.weight": (d.hidden_size,),
        }
        for layer in range(d.num_hidden_layers):
            prefix = f"layers.{layer}"
            mixer = f"{prefix}.mixer"
            shapes[f"{prefix}.norm.weight"] = (d.hidden_size,)
            shapes[f"{mixer}.in_proj.weight"] = (2 * d.intermediate_size, d.hidden_size)
            if use_bias:
                shapes[f"{mixer}.in_proj.bias"] = (2 * d.intermediate_size,)
            shapes[f"{mixer}.conv1d.weight"] = (
                d.intermediate_size,
                1,
                conv_kernel,
            )
            if use_conv_bias:
                shapes[f"{mixer}.conv1d.bias"] = (d.intermediate_size,)
            shapes[f"{mixer}.x_proj.weight"] = (
                rank + 2 * state_size,
                d.intermediate_size,
            )
            shapes[f"{mixer}.dt_proj.weight"] = (d.intermediate_size, rank)
            shapes[f"{mixer}.dt_proj.bias"] = (d.intermediate_size,)
            shapes[f"{mixer}.A_log"] = (d.intermediate_size, state_size)
            shapes[f"{mixer}.D"] = (d.intermediate_size,)
            shapes[f"{mixer}.out_proj.weight"] = (d.hidden_size, d.intermediate_size)
            if use_bias:
                shapes[f"{mixer}.out_proj.bias"] = (d.hidden_size,)
        return shapes
    if architecture_id == "gpt2-causal-decoder":
        shapes: dict[str, tuple[int, ...]] = {
            "token_embedding.weight": (input_rows, d.hidden_size),
            "position_embedding.weight": (d.max_position_embeddings, d.hidden_size),
            "lm_head.weight": (output_rows, d.hidden_size),
            "final_norm.weight": (d.hidden_size,),
            "final_norm.bias": (d.hidden_size,),
        }
        for layer in range(d.num_hidden_layers):
            prefix = f"layers.{layer}"
            for norm in ("attention_norm", "mlp_norm"):
                shapes[f"{prefix}.{norm}.weight"] = (d.hidden_size,)
                shapes[f"{prefix}.{norm}.bias"] = (d.hidden_size,)
            shapes[f"{prefix}.attention.causal_mask"] = (
                1,
                1,
                d.max_position_embeddings,
                d.max_position_embeddings,
            )
            shapes[f"{prefix}.attn.c_attn.weight"] = (d.hidden_size, 3 * d.hidden_size)
            shapes[f"{prefix}.attn.c_attn.bias"] = (3 * d.hidden_size,)
            shapes[f"{prefix}.attn.c_proj.weight"] = (d.hidden_size, d.hidden_size)
            shapes[f"{prefix}.attn.c_proj.bias"] = (d.hidden_size,)
            shapes[f"{prefix}.mlp.c_fc.weight"] = (d.hidden_size, d.intermediate_size)
            shapes[f"{prefix}.mlp.c_fc.bias"] = (d.intermediate_size,)
            shapes[f"{prefix}.mlp.c_proj.weight"] = (d.intermediate_size, d.hidden_size)
            shapes[f"{prefix}.mlp.c_proj.bias"] = (d.hidden_size,)
        return shapes
    if architecture_id == "phi-causal-decoder":
        shapes = {
            "token_embedding.weight": (input_rows, d.hidden_size),
            "lm_head.weight": (output_rows, d.hidden_size),
            "lm_head.bias": (output_rows,),
            "final_norm.weight": (d.hidden_size,),
            "final_norm.bias": (d.hidden_size,),
        }
        for layer in range(d.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            shapes[f"{prefix}.input_norm.weight"] = (d.hidden_size,)
            shapes[f"{prefix}.input_norm.bias"] = (d.hidden_size,)
            for projection, width in (
                ("q_proj", d.num_attention_heads * d.head_dim),
                ("k_proj", d.num_key_value_heads * d.head_dim),
                ("v_proj", d.num_key_value_heads * d.head_dim),
            ):
                shapes[f"{attention}.{projection}.weight"] = (width, d.hidden_size)
                shapes[f"{attention}.{projection}.bias"] = (width,)
            shapes[f"{attention}.dense.weight"] = (
                d.hidden_size,
                d.num_attention_heads * d.head_dim,
            )
            shapes[f"{attention}.dense.bias"] = (d.hidden_size,)
            shapes[f"{prefix}.mlp.fc1.weight"] = (d.intermediate_size, d.hidden_size)
            shapes[f"{prefix}.mlp.fc1.bias"] = (d.intermediate_size,)
            shapes[f"{prefix}.mlp.fc2.weight"] = (d.hidden_size, d.intermediate_size)
            shapes[f"{prefix}.mlp.fc2.bias"] = (d.hidden_size,)
        return shapes
    if architecture_id == "gpt-neox-pythia-causal-decoder":
        shapes: dict[str, tuple[int, ...]] = {
            "token_embedding.weight": (input_rows, d.hidden_size),
            "lm_head.weight": (output_rows, d.hidden_size),
            "final_norm.weight": (d.hidden_size,),
            "final_norm.bias": (d.hidden_size,),
        }
        attention_bias = bool(config.get("attention_bias", True))
        for layer in range(d.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            for norm in ("attention_norm", "mlp_norm"):
                shapes[f"{prefix}.{norm}.weight"] = (d.hidden_size,)
                shapes[f"{prefix}.{norm}.bias"] = (d.hidden_size,)
            shapes[f"{attention}.query_key_value.weight"] = (
                3 * d.hidden_size,
                d.hidden_size,
            )
            shapes[f"{attention}.o_proj.weight"] = (d.hidden_size, d.hidden_size)
            if attention_bias:
                shapes[f"{attention}.query_key_value.bias"] = (3 * d.hidden_size,)
                shapes[f"{attention}.o_proj.bias"] = (d.hidden_size,)
            shapes[f"{prefix}.mlp.dense_h_to_4h.weight"] = (
                d.intermediate_size,
                d.hidden_size,
            )
            shapes[f"{prefix}.mlp.dense_h_to_4h.bias"] = (d.intermediate_size,)
            shapes[f"{prefix}.mlp.dense_4h_to_h.weight"] = (
                d.hidden_size,
                d.intermediate_size,
            )
            shapes[f"{prefix}.mlp.dense_4h_to_h.bias"] = (d.hidden_size,)
        return shapes
    if architecture_id == "mixtral-sparse-moe-causal-decoder":
        experts = config.get("num_local_experts")
        top_k = config.get("num_experts_per_tok")
        if (
            type(experts) is not int
            or type(top_k) is not int
            or experts <= 1
            or not 0 < top_k < experts
        ):
            raise ReferenceLoweringError("Mixtral expert topology is invalid")
        shapes = {
            "token_embedding.weight": (input_rows, d.hidden_size),
            "lm_head.weight": (output_rows, d.hidden_size),
            "final_norm.weight": (d.hidden_size,),
        }
        for layer in range(d.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            moe = f"{prefix}.moe"
            shapes[f"{prefix}.attention_norm.weight"] = (d.hidden_size,)
            shapes[f"{prefix}.moe_norm.weight"] = (d.hidden_size,)
            for projection, width in (
                ("q_proj", d.num_attention_heads * d.head_dim),
                ("k_proj", d.num_key_value_heads * d.head_dim),
                ("v_proj", d.num_key_value_heads * d.head_dim),
            ):
                shapes[f"{attention}.{projection}.weight"] = (width, d.hidden_size)
            shapes[f"{attention}.o_proj.weight"] = (
                d.hidden_size,
                d.num_attention_heads * d.head_dim,
            )
            shapes[f"{moe}.router.weight"] = (experts, d.hidden_size)
            for expert in range(experts):
                expert_prefix = f"{moe}.routed_experts.{expert}"
                shapes[f"{expert_prefix}.gate_proj.weight"] = (
                    d.intermediate_size,
                    d.hidden_size,
                )
                shapes[f"{expert_prefix}.up_proj.weight"] = (
                    d.intermediate_size,
                    d.hidden_size,
                )
                shapes[f"{expert_prefix}.down_proj.weight"] = (
                    d.hidden_size,
                    d.intermediate_size,
                )
        return shapes
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (input_rows, d.hidden_size),
        "lm_head.weight": (output_rows, d.hidden_size),
        "final_norm.weight": (d.hidden_size,),
    }
    attention_bias = bool(config.get("attention_bias", False))
    mlp_bias = bool(config.get("mlp_bias", False))
    for layer in range(d.num_hidden_layers):
        prefix = f"layers.{layer}"
        attention = f"{prefix}.attention"
        shapes[f"{prefix}.attention_norm.weight"] = (d.hidden_size,)
        shapes[f"{prefix}.mlp_norm.weight"] = (d.hidden_size,)
        for projection, width in (
            ("q_proj", d.num_attention_heads * d.head_dim),
            ("k_proj", d.num_key_value_heads * d.head_dim),
            ("v_proj", d.num_key_value_heads * d.head_dim),
        ):
            shapes[f"{attention}.{projection}.weight"] = (width, d.hidden_size)
            if architecture_id == "qwen2-dense-causal-decoder" or attention_bias:
                shapes[f"{attention}.{projection}.bias"] = (width,)
        shapes[f"{attention}.o_proj.weight"] = (
            d.hidden_size,
            d.num_attention_heads * d.head_dim,
        )
        if attention_bias and architecture_id != "qwen2-dense-causal-decoder":
            shapes[f"{attention}.o_proj.bias"] = (d.hidden_size,)
        if architecture_id == "qwen3-dense-causal-decoder":
            shapes[f"{attention}.q_norm.weight"] = (d.head_dim,)
            shapes[f"{attention}.k_norm.weight"] = (d.head_dim,)
        for projection, shape in (
            ("gate_proj", (d.intermediate_size, d.hidden_size)),
            ("up_proj", (d.intermediate_size, d.hidden_size)),
            ("down_proj", (d.hidden_size, d.intermediate_size)),
        ):
            shapes[f"{prefix}.mlp.{projection}.weight"] = shape
            if mlp_bias:
                width = d.hidden_size if projection == "down_proj" else d.intermediate_size
                shapes[f"{prefix}.mlp.{projection}.bias"] = (width,)
    return shapes


def _validate_operation(operation: OperationIR, dimensions: ModelDimensionsIR) -> None:
    if operation.kind not in _ALLOWED_OPERATION_KINDS:
        raise ReferenceLoweringError(
            "IR operation kind is outside the registered reference target",
            details={"operation_id": operation.operation_id, "kind": operation.kind},
        )
    attributes = operation.attributes
    empty = {
        "silu",
        "elementwise-multiply",
        "gelu-erf",
        "gelu-tanh",
        "qwen35-attention-output-gate",
    }
    if operation.kind in empty and attributes:
        raise ReferenceLoweringError("parameter-free operation has unexpected attributes")
    if operation.kind == "residual-add" and attributes not in (
        {},
        {"residual_accumulation": "float32"},
        {"residual_accumulation": "activation-dtype"},
    ):
        raise ReferenceLoweringError("residual accumulation is outside the target contract")
    if operation.kind == "token-embedding" and attributes != {"padding_policy": "IOIR-row-mapper"}:
        raise ReferenceLoweringError("token embedding attributes are outside the target contract")
    if operation.kind == "absolute-position-embedding-add":
        if attributes != {"max_position_embeddings": dimensions.max_position_embeddings}:
            raise ReferenceLoweringError(
                "absolute position embedding attributes differ from ModelIR"
            )
    if operation.kind in {"rms-norm", "gemma-rms-norm", "head-rms-norm"}:
        expected = (
            {"epsilon"}
            if operation.kind in {"rms-norm", "gemma-rms-norm"}
            else {"epsilon", "head_dim"}
        )
        if set(attributes) != expected:
            raise ReferenceLoweringError("RMSNorm attributes are outside the target contract")
        _finite_positive(attributes["epsilon"], field="RMSNorm epsilon")
        if operation.kind == "head-rms-norm" and attributes["head_dim"] != dimensions.head_dim:
            raise ReferenceLoweringError("head RMSNorm head_dim differs from ModelIR")
    if operation.kind in {"qwen35-rms-norm", "qwen35-head-rms-norm"}:
        expected = (
            {"epsilon", "weight_center"}
            if operation.kind == "qwen35-rms-norm"
            else {"epsilon", "head_dim", "weight_center"}
        )
        if set(attributes) != expected or float(attributes["weight_center"]) != 1.0:
            raise ReferenceLoweringError("Qwen3.5 RMSNorm attributes are outside the target")
        _finite_positive(attributes["epsilon"], field="Qwen3.5 RMSNorm epsilon")
        if (
            operation.kind == "qwen35-head-rms-norm"
            and attributes["head_dim"] != dimensions.head_dim
        ):
            raise ReferenceLoweringError("Qwen3.5 head RMSNorm differs from ModelIR")
    if operation.kind == "qwen35-rms-norm-gated":
        if set(attributes) != {"epsilon", "head_dim"}:
            raise ReferenceLoweringError("Qwen3.5 gated RMSNorm attributes are invalid")
        _finite_positive(attributes["epsilon"], field="Qwen3.5 gated RMSNorm epsilon")
        if type(attributes["head_dim"]) is not int or attributes["head_dim"] <= 0:
            raise ReferenceLoweringError("Qwen3.5 gated RMSNorm head width is invalid")
    if operation.kind == "mamba-rms-norm":
        if set(attributes) != {"epsilon", "accumulation"}:
            raise ReferenceLoweringError("Mamba RMSNorm attributes are outside the target")
        _finite_positive(attributes["epsilon"], field="Mamba RMSNorm epsilon")
        if attributes["accumulation"] != "float32":
            raise ReferenceLoweringError("Mamba RMSNorm must accumulate in float32")
    if operation.kind == "scalar-multiply":
        if set(attributes) != {"scalar"}:
            raise ReferenceLoweringError("scalar multiply attributes are outside the target")
        scalar = _finite_positive(attributes["scalar"], field="scalar multiply value")
        if scalar != math.sqrt(dimensions.hidden_size):
            raise ReferenceLoweringError("embedding scale differs from ModelIR hidden width")
    if operation.kind == "layer-norm":
        if set(attributes) != {"epsilon"}:
            raise ReferenceLoweringError("LayerNorm attributes are outside the target contract")
        _finite_positive(attributes["epsilon"], field="LayerNorm epsilon")
    if operation.kind == "linear" and attributes != {"weight_orientation": "out-in"}:
        raise ReferenceLoweringError("linear orientation is outside the target contract")
    if operation.kind == "moe-router-linear":
        expected = {
            "expert_scope",
            "num_routed_experts",
            "num_shared_experts",
            "router_bias",
            "weight_orientation",
        }
        if (
            set(attributes) != expected
            or attributes["expert_scope"] != "routed-only"
            or type(attributes["num_routed_experts"]) is not int
            or attributes["num_routed_experts"] <= 1
            or attributes["num_shared_experts"] != 0
            or attributes["router_bias"] is not False
            or attributes["weight_orientation"] != "experts-hidden"
        ):
            raise ReferenceLoweringError("MoE router attributes are outside the target contract")
    if operation.kind == "moe-top-k-softmax":
        expected = {
            "jitter_noise",
            "num_experts_per_token",
            "num_routed_experts",
            "renormalize_selected_probabilities",
            "selection",
            "softmax_dtype",
            "tie_breaking",
        }
        experts = attributes.get("num_routed_experts")
        top_k = attributes.get("num_experts_per_token")
        if (
            set(attributes) != expected
            or type(experts) is not int
            or type(top_k) is not int
            or experts <= 1
            or not 0 < top_k < experts
            or float(attributes["jitter_noise"]) != 0.0
            or attributes["renormalize_selected_probabilities"] is not True
            or attributes["selection"] != "top-k-after-softmax"
            or attributes["softmax_dtype"] != "float32"
            or attributes["tie_breaking"] != "source-framework-defined"
        ):
            raise ReferenceLoweringError("MoE top-k attributes are outside the target contract")
    if operation.kind == "moe-token-dispatch":
        expected = {"expert_scope", "num_routed_experts", "num_shared_experts"}
        if (
            set(attributes) != expected
            or attributes["expert_scope"] != "routed-only"
            or type(attributes["num_routed_experts"]) is not int
            or attributes["num_routed_experts"] <= 1
            or attributes["num_shared_experts"] != 0
        ):
            raise ReferenceLoweringError("MoE dispatch attributes are outside the target contract")
    if operation.kind == "moe-routed-expert-linear":
        expected = {"expert_id", "expert_scope", "weight_orientation"}
        if (
            set(attributes) != expected
            or type(attributes["expert_id"]) is not int
            or attributes["expert_id"] < 0
            or attributes["expert_scope"] != "routed"
            or attributes["weight_orientation"] != "out-in"
        ):
            raise ReferenceLoweringError(
                "MoE routed-expert linear attributes are outside the target contract"
            )
    if operation.kind == "moe-weighted-scatter-add":
        expected = {
            "accumulation_order",
            "expert_scope",
            "num_routed_experts",
            "num_shared_experts",
        }
        if (
            set(attributes) != expected
            or attributes["accumulation_order"] != "source-expert-index-order"
            or attributes["expert_scope"] != "routed-only"
            or type(attributes["num_routed_experts"]) is not int
            or attributes["num_routed_experts"] <= 1
            or attributes["num_shared_experts"] != 0
        ):
            raise ReferenceLoweringError("MoE combine attributes are outside the target contract")
    if operation.kind == "conv1d-linear" and attributes != {"weight_orientation": "in-out"}:
        raise ReferenceLoweringError("GPT-2 Conv1D orientation is outside the target contract")
    if operation.kind == "linear-readout" and attributes != {"weight_orientation": "rows-hidden"}:
        raise ReferenceLoweringError("readout orientation is outside the target contract")
    if operation.kind == "rotary-default":
        if set(attributes) != {"head_dim", "max_position_embeddings", "rope_theta"}:
            raise ReferenceLoweringError("default RoPE attributes are outside the target contract")
        if (
            attributes["head_dim"] != dimensions.head_dim
            or attributes["max_position_embeddings"] != dimensions.max_position_embeddings
        ):
            raise ReferenceLoweringError("default RoPE dimensions differ from ModelIR")
        _finite_positive(attributes["rope_theta"], field="rope_theta")
    if operation.kind == "rotary-partial-default":
        expected = {"head_dim", "max_position_embeddings", "rope_theta", "rotary_dim"}
        if set(attributes) != expected:
            raise ReferenceLoweringError("partial RoPE attributes are outside the target contract")
        rotary_dim = attributes["rotary_dim"]
        if (
            attributes["head_dim"] != dimensions.head_dim
            or attributes["max_position_embeddings"] != dimensions.max_position_embeddings
            or type(rotary_dim) is not int
            or rotary_dim <= 0
            or rotary_dim > dimensions.head_dim
            or rotary_dim % 2
        ):
            raise ReferenceLoweringError("partial RoPE dimensions differ from ModelIR")
        _finite_positive(attributes["rope_theta"], field="rope_theta")
    if operation.kind == "gpt-neox-qkv-unpack":
        expected = {"head_dim", "num_attention_heads", "packing"}
        if set(attributes) != expected or attributes["packing"] != "per-head-qkv":
            raise ReferenceLoweringError("GPT-NeoX QKV packing attributes are outside the target")
        if (
            attributes["head_dim"] != dimensions.head_dim
            or attributes["num_attention_heads"] != dimensions.num_attention_heads
        ):
            raise ReferenceLoweringError("GPT-NeoX QKV packing dimensions differ from ModelIR")
    if operation.kind == "fused-qkv-split":
        if attributes != {"split_width": dimensions.hidden_size}:
            raise ReferenceLoweringError("fused QKV split width differs from ModelIR")
    if operation.kind == "mamba-input-gate-split":
        if attributes != {"split_width": dimensions.intermediate_size}:
            raise ReferenceLoweringError("Mamba input/gate split width differs from ModelIR")
    if operation.kind == "mamba-causal-depthwise-convolution":
        expected = {"activation", "kernel_size", "state_slot"}
        if (
            set(attributes) != expected
            or attributes["activation"] != "silu"
            or type(attributes["kernel_size"]) is not int
            or attributes["kernel_size"] <= 0
            or type(attributes["state_slot"]) is not str
            or not attributes["state_slot"].endswith(".conv_state")
        ):
            raise ReferenceLoweringError("Mamba causal convolution attributes are invalid")
    if operation.kind == "qwen35-causal-depthwise-convolution":
        expected = {"activation", "kernel_size", "state_slot"}
        if (
            set(attributes) != expected
            or attributes["activation"] != "silu"
            or type(attributes["kernel_size"]) is not int
            or attributes["kernel_size"] <= 0
            or type(attributes["state_slot"]) is not str
            or not attributes["state_slot"].endswith(".conv_state")
        ):
            raise ReferenceLoweringError("Qwen3.5 causal convolution attributes are invalid")
    if operation.kind == "qwen35-linear-qkv-split":
        if (
            set(attributes) != {"key_width", "value_width"}
            or type(attributes["key_width"]) is not int
            or attributes["key_width"] <= 0
            or type(attributes["value_width"]) is not int
            or attributes["value_width"] <= 0
        ):
            raise ReferenceLoweringError("Qwen3.5 linear QKV split attributes are invalid")
    if operation.kind == "qwen35-query-gate-split":
        if attributes != {
            "head_dim": dimensions.head_dim,
            "num_attention_heads": dimensions.num_attention_heads,
        }:
            raise ReferenceLoweringError("Qwen3.5 query/gate split differs from ModelIR")
    if operation.kind == "qwen35-gated-delta-recurrence":
        expected = {
            "key_head_dim",
            "num_key_heads",
            "num_value_heads",
            "state_slot",
            "value_head_dim",
        }
        if (
            set(attributes) != expected
            or any(
                type(attributes[field]) is not int or attributes[field] <= 0
                for field in (
                    "key_head_dim",
                    "num_key_heads",
                    "num_value_heads",
                    "value_head_dim",
                )
            )
            or attributes["num_value_heads"] % attributes["num_key_heads"]
            or type(attributes["state_slot"]) is not str
            or not attributes["state_slot"].endswith(".recurrent_state")
        ):
            raise ReferenceLoweringError("Qwen3.5 gated-delta recurrence attributes are invalid")
    if operation.kind == "qwen35-partial-interleaved-mrope":
        expected = {
            "head_dim",
            "mrope_interleaved",
            "mrope_section",
            "rope_theta",
            "rotary_dim",
        }
        section = attributes.get("mrope_section")
        rotary_dim = attributes.get("rotary_dim")
        if (
            set(attributes) != expected
            or attributes["head_dim"] != dimensions.head_dim
            or attributes["mrope_interleaved"] is not True
            or not isinstance(section, list)
            or len(section) != 3
            or any(type(value) is not int or value <= 0 for value in section)
            or type(rotary_dim) is not int
            or rotary_dim <= 0
            or rotary_dim > dimensions.head_dim
            or rotary_dim % 2
            or sum(section) != rotary_dim // 2
        ):
            raise ReferenceLoweringError("Qwen3.5 partial interleaved mRoPE is invalid")
        _finite_positive(attributes["rope_theta"], field="Qwen3.5 rope_theta")
    if operation.kind == "mamba-selection-split":
        if (
            set(attributes) != {"state_size", "time_step_rank"}
            or type(attributes["state_size"]) is not int
            or attributes["state_size"] <= 0
            or type(attributes["time_step_rank"]) is not int
            or attributes["time_step_rank"] <= 0
        ):
            raise ReferenceLoweringError("Mamba selection split attributes are invalid")
    if operation.kind == "mamba-selective-scan":
        expected = {"dt_activation", "state_size", "state_slot", "state_update"}
        if (
            set(attributes) != expected
            or attributes["dt_activation"] != "softplus"
            or type(attributes["state_size"]) is not int
            or attributes["state_size"] <= 0
            or type(attributes["state_slot"]) is not str
            or not attributes["state_slot"].endswith(".recurrent_state")
            or attributes["state_update"] != "prefix-indexed-provisional"
        ):
            raise ReferenceLoweringError("Mamba selective scan attributes are invalid")
    if operation.kind == "causal-grouped-query-attention":
        expected = {
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "scale",
            "state_slots",
        }
        if set(attributes) != expected:
            raise ReferenceLoweringError("GQA attributes are outside the target contract")
        if (
            attributes["head_dim"] != dimensions.head_dim
            or attributes["num_attention_heads"] != dimensions.num_attention_heads
            or attributes["num_key_value_heads"] != dimensions.num_key_value_heads
            or float(attributes["scale"]) != dimensions.head_dim**-0.5
        ):
            raise ReferenceLoweringError("GQA dimensions or scale differ from ModelIR")


def _validate_mamba_state(
    state: StateIR, dimensions: ModelDimensionsIR, config: dict[str, Any]
) -> None:
    state_size, conv_kernel, _, _, _, _, _, _ = _mamba_config_semantics(dimensions, config)
    expected_slots: list[dict[str, Any]] = [
        {
            "slot_id": "sequence_length",
            "kind": "committed-sequence-length",
            "dtype": "int64",
            "shape_expression": ["batch"],
            "ownership": "request",
            "lease_behavior": "exclusive-epoch-bound",
            "provisional_representation": "accepted-prefix-length-delta",
            "commit_rule": "atomic-select-accepted-prefix",
            "rollback_rule": "discard-provisional",
            "memory_charge_expression": "batch * sizeof(int64)",
        }
    ]
    for layer in range(dimensions.num_hidden_layers):
        expected_slots.extend(
            [
                {
                    "slot_id": f"layers.{layer}.conv_state",
                    "kind": "mamba-causal-convolution-window",
                    "dtype": "activation",
                    "shape_expression": ["batch", "intermediate_size", "conv_kernel"],
                    "ownership": "request",
                    "lease_behavior": "exclusive-epoch-bound",
                    "provisional_representation": "prefix-indexed-state-snapshots",
                    "commit_rule": "atomic-select-accepted-prefix-snapshot",
                    "rollback_rule": "discard-provisional-snapshots",
                    "memory_charge_expression": (
                        "batch * intermediate_size * conv_kernel * dtype_bytes"
                    ),
                },
                {
                    "slot_id": f"layers.{layer}.recurrent_state",
                    "kind": "mamba-selective-scan-state",
                    "dtype": "activation",
                    "shape_expression": ["batch", "intermediate_size", "state_size"],
                    "ownership": "request",
                    "lease_behavior": "exclusive-epoch-bound",
                    "provisional_representation": "prefix-indexed-state-snapshots",
                    "commit_rule": "atomic-select-accepted-prefix-snapshot",
                    "rollback_rule": "discard-provisional-snapshots",
                    "memory_charge_expression": (
                        "batch * intermediate_size * state_size * dtype_bytes"
                    ),
                },
            ]
        )
    expected_slots.sort(key=lambda item: item["slot_id"])
    if [item.as_dict() for item in state.slots] != expected_slots:
        raise ReferenceLoweringError("Mamba StateIR slots differ from the exact target contract")
    slot_ids = tuple(item["slot_id"] for item in expected_slots)
    update_attributes = {
        "accepted_prefix": "select-snapshot-at-prefix-boundary",
        "committed_mutation_before_commit": False,
        "conv_kernel": conv_kernel,
        "state_size": state_size,
    }
    expected_initialization = [
        {
            "operation_id": "initialize-mamba-request-state",
            "kind": "zero-recurrent-state-and-bind-epoch",
            "slots": list(slot_ids),
            "attributes": {"epoch": 0, "sequence_length": 0},
        }
    ]
    expected_prefill = [
        {
            "operation_id": "prefill-mamba-provisional-scan",
            "kind": "scan-prefix-to-provisional-state-snapshots",
            "slots": list(slot_ids),
            "attributes": update_attributes,
        }
    ]
    expected_decode = [
        {
            "operation_id": "decode-mamba-provisional-scan",
            "kind": "scan-decode-block-to-provisional-state-snapshots",
            "slots": list(slot_ids),
            "attributes": update_attributes,
        }
    ]
    if (
        [item.as_dict() for item in state.initialization] != expected_initialization
        or [item.as_dict() for item in state.prefill_updates] != expected_prefill
        or [item.as_dict() for item in state.decode_updates] != expected_decode
    ):
        raise ReferenceLoweringError(
            "Mamba StateIR operations differ from the exact target contract"
        )
    if state.commit_protocol.as_dict() != {
        "protocol_id": "mrun-transactional-recurrent-state-v1",
        "authority": "cache-issued-exclusive-lease",
        "atomicity": "all-convolution-recurrence-and-length-slots",
        "accepted_prefix_rule": ("select-exact-prefix-snapshot-zero-through-provisional-length"),
        "rollback": "discard-snapshots-without-committed-mutation",
        "stale_epoch_rule": "reject-before-write",
    }:
        raise ReferenceLoweringError(
            "Mamba StateIR commit protocol differs from the exact target contract"
        )
    expected_equations = [
        {
            "quantity": "committed_conv_state_bytes",
            "expression": (
                "batch * num_hidden_layers * intermediate_size * conv_kernel * dtype_bytes"
            ),
            "units": "bytes",
        },
        {
            "quantity": "committed_recurrent_state_bytes",
            "expression": (
                "batch * num_hidden_layers * intermediate_size * state_size * dtype_bytes"
            ),
            "units": "bytes",
        },
        {
            "quantity": "length_bytes",
            "expression": "batch * sizeof(int64)",
            "units": "bytes",
        },
        {
            "quantity": "provisional_snapshot_bytes",
            "expression": (
                "provisional_tokens * batch * num_hidden_layers * intermediate_size * "
                "(conv_kernel + state_size) * dtype_bytes"
            ),
            "units": "bytes",
        },
    ]
    if [item.as_dict() for item in state.capacity_equations] != expected_equations:
        raise ReferenceLoweringError(
            "Mamba StateIR capacity equations differ from the exact target contract"
        )


def _reopen_artifact(value: ComponentArtifact | str | Path) -> ComponentArtifact:
    if isinstance(value, ComponentArtifact):
        reopened = open_component_artifact(value.directory)
        if (
            reopened.artifact_id != value.artifact_id
            or reopened.manifest_sha256 != value.manifest_sha256
        ):
            raise ReferenceLoweringError("component artifact changed before reference lowering")
        return reopened
    return open_component_artifact(value)


def lower_component_artifact_to_reference(
    artifact: ComponentArtifact | str | Path,
) -> ReferenceExecutable:
    """Validate and lower a canonical artifact into the registered CPU reference target."""

    if sys.byteorder != "little":
        raise ReferenceLoweringError("reference target currently requires a little-endian host")
    verified = _reopen_artifact(artifact)
    manifest = verified.manifest
    if (
        manifest["schema_version"] != COMPONENT_ARTIFACT_SCHEMA
        or manifest["status"] != "built-unexecuted"
        or manifest["execution_certified"] is not False
    ):
        raise ReferenceLoweringError("artifact manifest crosses the source/reference boundary")
    model = verified.ir_bundle.model
    architecture = _ARCHITECTURES.get(model.architecture_id)
    if architecture is None:
        raise ReferenceLoweringError(
            "architecture has no registered reference target",
            details={"architecture_id": model.architecture_id},
        )
    expected_adapter, source_model_type = architecture
    if (
        model.adapter_id != expected_adapter
        or model.adapter_version != "1.0.0"
        or verified.source.config.get("model_type") != source_model_type
    ):
        raise ReferenceLoweringError(
            "adapter, architecture, and source model type are inconsistent"
        )
    expected_program = _expected_program(
        model.dimensions, model.architecture_id, verified.source.config
    )
    observed_program = tuple((item.operation_id, item.kind) for item in model.operations)
    if observed_program != expected_program:
        raise ReferenceLoweringError(
            "IR operation topology is outside the registered reference target"
        )
    if model.architecture_id == "gpt2-causal-decoder":
        _validate_gpt2_bindings(model.operations, model.dimensions, verified.source.config)
    if model.architecture_id == "gpt-neox-pythia-causal-decoder":
        _validate_gpt_neox_bindings(model.operations, model.dimensions, verified.source.config)
    if model.architecture_id == "mamba1-selective-state-space-causal-decoder":
        _validate_mamba_bindings(model.operations, model.dimensions, verified.source.config)
        _validate_mamba_state(verified.ir_bundle.state, model.dimensions, verified.source.config)
        expected_state_refs = tuple(
            sorted(
                (
                    "sequence_length",
                    *(
                        slot
                        for layer in range(model.dimensions.num_hidden_layers)
                        for slot in (
                            f"layers.{layer}.conv_state",
                            f"layers.{layer}.recurrent_state",
                        )
                    ),
                )
            )
        )
        if model.state_refs != expected_state_refs:
            raise ReferenceLoweringError("Mamba ModelIR state references are not exact")
        residual_fp32 = bool(verified.source.config.get("residual_in_fp32", True))
        if model.numerical_semantics.as_dict() != {
            "reference_contract": "source-reference-required-before-u3",
            "accumulation": (
                "float32-residual-and-ssm"
                if residual_fp32
                else "ssm-float32-residual-activation-dtype"
            ),
            "softmax": "not-applicable-no-attention",
            "positional_arithmetic": "none-causal-recurrence-order",
            "optimization_contract": "preregister-before-target-execution",
        }:
            raise ReferenceLoweringError("Mamba numerical semantics are outside the target")
    if model.architecture_id == "mixtral-sparse-moe-causal-decoder":
        _validate_mixtral_bindings(model.operations, model.dimensions, verified.source.config)
    if model.architecture_id == "qwen3_5-hybrid-text-causal-decoder":
        text = verified.source.config.get("text_config")
        if not isinstance(text, dict):
            raise ReferenceLoweringError("Qwen3.5 text_config is absent")
        layer_types = tuple(text.get("layer_types", ()))
        expected_refs = ["position"]
        expected_slot_kinds = {"position": "position-counter"}
        for layer, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                for kind in ("k", "v"):
                    slot = f"layers.{layer}.{kind}_cache"
                    expected_refs.append(slot)
                    expected_slot_kinds[slot] = f"paged-{kind}-cache"
            elif layer_type == "linear_attention":
                conv = f"layers.{layer}.conv_state"
                recurrent = f"layers.{layer}.recurrent_state"
                expected_refs.extend((conv, recurrent))
                expected_slot_kinds[conv] = "causal-depthwise-convolution-state"
                expected_slot_kinds[recurrent] = "gated-delta-recurrent-matrix"
            else:
                raise ReferenceLoweringError("Qwen3.5 StateIR has an unknown layer mixer")
        if model.state_refs != tuple(sorted(expected_refs)):
            raise ReferenceLoweringError("Qwen3.5 ModelIR hybrid state references are not exact")
        observed_slots = {slot.slot_id: slot.kind for slot in verified.ir_bundle.state.slots}
        if observed_slots != expected_slot_kinds:
            raise ReferenceLoweringError("Qwen3.5 StateIR hybrid slot inventory is not exact")
        if verified.ir_bundle.state.commit_protocol.protocol_id != (
            "mrun-transactional-hybrid-state-v1"
        ):
            raise ReferenceLoweringError("Qwen3.5 StateIR commit protocol is not recognized")
        if model.numerical_semantics.as_dict() != {
            "reference_contract": "source-reference-required-before-u3",
            "accumulation": "qwen35-fp32-norm-and-recurrence",
            "softmax": "stable-causal-softmax-fp32",
            "positional_arithmetic": "partial-interleaved-mrope-fp32",
            "optimization_contract": "fused-kernels-require-capability-and-parity-proof",
        }:
            raise ReferenceLoweringError("Qwen3.5 numerical semantics are outside the target")
    for operation in model.operations:
        _validate_operation(operation, model.dimensions)

    views = verified.ir_bundle.physical_weights.views
    view_by_name = {item.logical_name: item for item in views}
    for view in views:
        if (
            len(view.transforms) != 1
            or view.transforms[0].kind != "identity"
            or view.transforms[0].parameters
        ):
            raise ReferenceLoweringError(
                "reference target supports identity logical views only",
                details={"logical_name": view.logical_name},
            )
    grouped: dict[str, list[str]] = {}
    for view in views:
        grouped.setdefault(view.allocation_id, []).append(view.logical_name)
    actual_aliases = {
        item.allocation_id: tuple(item.logical_names)
        for item in verified.ir_bundle.physical_weights.alias_classes
    }
    expected_aliases = {
        allocation_id: tuple(sorted(names))
        for allocation_id, names in grouped.items()
        if len(names) > 1
    }
    if actual_aliases != expected_aliases:
        raise ReferenceLoweringError("logical aliases do not exactly describe shared allocations")

    row_mappers = {item.mapper_id: item for item in verified.ir_bundle.io.row_mappers}
    if set(row_mappers) != {"tokens-to-input-rows", "tokens-to-output-rows"}:
        raise ReferenceLoweringError("reference target requires exact input/output row mappers")
    input_mapper = row_mappers["tokens-to-input-rows"]
    output_mapper = row_mappers["tokens-to-output-rows"]
    if input_mapper.kind not in {"identity", "padded-identity"} or output_mapper.kind not in {
        "identity",
        "padded-identity",
    }:
        raise ReferenceLoweringError("reference target does not implement non-identity row mapping")
    if (
        input_mapper.token_count != model.dimensions.vocab_size
        or output_mapper.token_count != model.dimensions.vocab_size
        or input_mapper.row_count != model.dimensions.physical_vocab_rows
    ):
        raise ReferenceLoweringError("row mapper domains differ from ModelIR dimensions")
    expected_shapes = _expected_parameter_shapes(
        model.dimensions,
        architecture_id=model.architecture_id,
        config=verified.source.config,
        input_rows=input_mapper.row_count,
        output_rows=output_mapper.row_count,
    )
    rotary_names = {
        name
        for operation in model.operations
        if operation.kind == "rotary-default"
        for name in operation.parameters
    }
    if set(view_by_name) != set(expected_shapes) | rotary_names:
        raise ReferenceLoweringError(
            "logical parameter inventory is outside the exact dense reference schema",
            details={
                "missing": sorted((set(expected_shapes) | rotary_names) - set(view_by_name)),
                "extra": sorted(set(view_by_name) - set(expected_shapes) - rotary_names),
            },
        )
    for name, shape in expected_shapes.items():
        if view_by_name[name].logical_shape != shape:
            raise ReferenceLoweringError(
                "logical parameter shape differs from the dense reference schema",
                details={
                    "logical_name": name,
                    "expected": list(shape),
                    "actual": list(view_by_name[name].logical_shape),
                },
            )
    allocation_by_id = {
        item.allocation_id: item for item in verified.ir_bundle.physical_weights.allocations
    }
    parameter_dtypes = {
        allocation_by_id[view_by_name[name].allocation_id].stored_dtype for name in expected_shapes
    }
    if model.architecture_id == "qwen3_5-hybrid-text-causal-decoder":
        if not parameter_dtypes or not parameter_dtypes <= set(_TORCH_DTYPES):
            raise ReferenceLoweringError(
                "Qwen3.5 stored parameter dtype has no registered Torch decoder",
                details={"dtypes": sorted(parameter_dtypes)},
            )
        stored_dtype = "mixed[" + ",".join(sorted(parameter_dtypes)) + "]"
    else:
        if len(parameter_dtypes) != 1:
            raise ReferenceLoweringError(
                "reference target requires one stored dtype across executable parameters",
                details={"dtypes": sorted(parameter_dtypes)},
            )
        stored_dtype = next(iter(parameter_dtypes))
        if stored_dtype not in _TORCH_DTYPES:
            raise ReferenceLoweringError("stored parameter dtype has no registered Torch decoder")
    if model.numerical_semantics.reference_contract != "source-reference-required-before-u3":
        raise ReferenceLoweringError("ModelIR numerical reference contract is not recognized")
    state_protocol = verified.ir_bundle.state.commit_protocol.protocol_id
    if model.architecture_id == "mamba1-selective-state-space-causal-decoder":
        if state_protocol != "mrun-transactional-recurrent-state-v1":
            raise ReferenceLoweringError("Mamba StateIR commit protocol is not recognized")
    elif model.architecture_id == "qwen3_5-hybrid-text-causal-decoder":
        if state_protocol != "mrun-transactional-hybrid-state-v1":
            raise ReferenceLoweringError("Qwen3.5 StateIR commit protocol is not recognized")
    elif state_protocol != "mrun-transactional-state-v1":
        raise ReferenceLoweringError("StateIR commit protocol is not recognized")
    kwargs = {
        "artifact_id": verified.artifact_id,
        "manifest_sha256": verified.manifest_sha256,
        "ir_bundle_fingerprint": verified.ir_bundle.fingerprint,
        "model_fingerprint": model.fingerprint,
        "architecture_id": model.architecture_id,
        "source_model_type": source_model_type,
        "stored_parameter_dtype": stored_dtype,
        "operation_kinds": tuple(sorted({item.kind for item in model.operations})),
        "state_mode": "stateless-full-sequence",
        "execution_certified": False,
        "production_runtime_eligible": False,
        "native_lowering_status": "not-lowered",
    }
    identity = ReferenceTargetIdentity.build(**kwargs)
    return ReferenceExecutable(verified, identity, ArtifactTensorStore(verified))


def run_g8_reference_parity(
    artifact: ComponentArtifact | str | Path | ReferenceExecutable,
    cases: Sequence[Sequence[Sequence[int]] | torch.Tensor],
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> ReferenceExecutionCertification:
    """Execute independent source-schema and IR paths and return a scoped G8 record."""

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    executable = (
        artifact
        if isinstance(artifact, ReferenceExecutable)
        else lower_component_artifact_to_reference(artifact)
    )
    normalized = _normalize_cases(cases)
    source = _SourceSchemaRunner(executable)
    input_fingerprints: list[str] = []
    source_fingerprints: list[str] = []
    ir_fingerprints: list[str] = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for case_index, token_ids in enumerate(normalized):
        source_result = source.forward(token_ids)
        ir_result = executable.forward(token_ids)
        if source_result.logits.shape != ir_result.logits.shape:
            raise ReferenceParityError(
                "source-schema and IR logits have different shapes",
                details={"case_index": case_index},
            )
        source_logits = source_result.logits.to(torch.float32)
        ir_logits = ir_result.logits.to(torch.float32)
        if not torch.isfinite(source_logits).all() or not torch.isfinite(ir_logits).all():
            raise ReferenceParityError(
                "reference execution produced non-finite logits",
                details={"case_index": case_index},
            )
        difference = (source_logits - ir_logits).abs()
        absolute = float(difference.max().item()) if difference.numel() else 0.0
        denominator = source_logits.abs().clamp_min(torch.finfo(torch.float32).tiny)
        relative = float((difference / denominator).max().item()) if difference.numel() else 0.0
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_relative_error = max(maximum_relative_error, relative)
        if not torch.allclose(
            source_logits,
            ir_logits,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            equal_nan=False,
        ):
            raise ReferenceParityError(
                "source-schema and IR logits differ",
                details={
                    "case_index": case_index,
                    "maximum_absolute_error": absolute,
                    "maximum_relative_error": relative,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                },
            )
        source_fingerprint = _tensor_fingerprint(source_logits)
        ir_fingerprint = _tensor_fingerprint(ir_logits)
        if source_fingerprint != ir_fingerprint:
            # Tolerant comparisons are useful diagnostically, but a promotion record binds exact
            # observations so it cannot hide a target-dependent numerical delta.
            raise ReferenceParityError(
                "tolerant parity passed but exact observation fingerprints differ",
                details={"case_index": case_index},
            )
        input_fingerprints.append(
            canonical_sha256(
                {"dtype": "int64", "shape": list(token_ids.shape), "values": token_ids.tolist()}
            )
        )
        source_fingerprints.append(source_fingerprint)
        ir_fingerprints.append(ir_fingerprint)
    kwargs = {
        "artifact_id": executable.identity.artifact_id,
        "target_fingerprint": executable.identity.fingerprint,
        "case_input_fingerprints": tuple(input_fingerprints),
        "source_logits_fingerprints": tuple(source_fingerprints),
        "ir_logits_fingerprints": tuple(ir_fingerprints),
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "checks": tuple(
            sorted(
                {
                    "allocation-execution-time-hashes",
                    "exact-logit-observation-fingerprints",
                    "finite-logits",
                    "logical-alias-identity",
                    "source-schema-vs-ir-forward",
                    "strict-operation-topology",
                    "strict-shape-and-dtype-contract",
                }
            )
        ),
    }
    payload = {
        "schema_version": REFERENCE_CERTIFICATION_SCHEMA,
        "status": "passed",
        "scope": "reference-target-source-schema-forward-parity",
        **kwargs,
        "case_input_fingerprints": list(kwargs["case_input_fingerprints"]),
        "source_logits_fingerprints": list(kwargs["source_logits_fingerprints"]),
        "ir_logits_fingerprints": list(kwargs["ir_logits_fingerprints"]),
        "checks": list(kwargs["checks"]),
        "execution_performed": True,
        "execution_certified": True,
        "production_runtime_eligible": False,
        "native_lowering_status": "not-lowered",
    }
    return ReferenceExecutionCertification(**kwargs, fingerprint=canonical_sha256(payload))


def run_g13_mixtral_semantic_parity(
    artifact: ComponentArtifact | str | Path | ReferenceExecutable,
    cases: Sequence[Sequence[Sequence[int]] | torch.Tensor],
) -> MixtralSemanticCertification:
    """Certify exact router, selected-expert, expert-output, combine, and logit observations."""

    executable = (
        artifact
        if isinstance(artifact, ReferenceExecutable)
        else lower_component_artifact_to_reference(artifact)
    )
    if executable.identity.architecture_id != "mixtral-sparse-moe-causal-decoder":
        raise MixtralSemanticParityError("G13 Mixtral certification requires a Mixtral target")
    normalized = _normalize_cases(cases)
    experts, _, _, _ = _mixtral_config_semantics(
        executable.artifact.ir_bundle.model.dimensions,
        executable.artifact.source.config,
    )
    names: list[str] = ["logits"]
    for layer in range(executable.artifact.ir_bundle.model.dimensions.num_hidden_layers):
        moe = f"layers.{layer}.moe"
        names.extend(
            [
                f"{moe}.router_logits",
                f"{moe}.routing_weights",
                f"{moe}.selected_experts",
                *(f"{moe}.routed_experts.{expert}.output" for expert in range(experts)),
                f"{moe}.output",
            ]
        )
    observation_names = tuple(sorted(names))
    trace_names = tuple(name for name in observation_names if name != "logits")
    source_runner = _SourceSchemaRunner(executable)
    input_fingerprints: list[str] = []
    source_fingerprints: list[str] = []
    ir_fingerprints: list[str] = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for case_index, token_ids in enumerate(normalized):
        source_result, source_trace = source_runner.mixtral_trace(token_ids)
        ir_result = executable.forward(token_ids)
        ir_trace = executable.trace(token_ids, trace_names)
        source_trace["logits"] = source_result.logits
        ir_trace["logits"] = ir_result.logits
        if set(source_trace) != set(observation_names) or set(ir_trace) != set(observation_names):
            raise MixtralSemanticParityError(
                "Mixtral trace inventory differs from the frozen G13 contract",
                details={"case_index": case_index},
            )
        input_fingerprints.append(
            canonical_sha256(
                {"dtype": "int64", "shape": list(token_ids.shape), "values": token_ids.tolist()}
            )
        )
        for name in observation_names:
            source_value = source_trace[name].detach().to(device="cpu")
            ir_value = ir_trace[name].detach().to(device="cpu")
            if source_value.shape != ir_value.shape or source_value.dtype != ir_value.dtype:
                raise MixtralSemanticParityError(
                    "Mixtral source and IR observations have different tensor contracts",
                    details={
                        "case_index": case_index,
                        "observation": name,
                        "source_shape": list(source_value.shape),
                        "ir_shape": list(ir_value.shape),
                        "source_dtype": str(source_value.dtype),
                        "ir_dtype": str(ir_value.dtype),
                    },
                )
            if source_value.is_floating_point():
                if not torch.isfinite(source_value).all() or not torch.isfinite(ir_value).all():
                    raise MixtralSemanticParityError(
                        "Mixtral semantic observation is non-finite",
                        details={"case_index": case_index, "observation": name},
                    )
                if source_value.numel():
                    difference = (source_value.to(torch.float32) - ir_value.to(torch.float32)).abs()
                    absolute = float(difference.max().item())
                    denominator = (
                        source_value.to(torch.float32)
                        .abs()
                        .clamp_min(torch.finfo(torch.float32).tiny)
                    )
                    relative = float((difference / denominator).max().item())
                    maximum_absolute_error = max(maximum_absolute_error, absolute)
                    maximum_relative_error = max(maximum_relative_error, relative)
            if not torch.equal(source_value, ir_value):
                raise MixtralSemanticParityError(
                    "Mixtral source and IR semantic observations differ",
                    details={"case_index": case_index, "observation": name},
                )
            source_fingerprint = _tensor_fingerprint(source_value)
            ir_fingerprint = _tensor_fingerprint(ir_value)
            if source_fingerprint != ir_fingerprint:
                raise MixtralSemanticParityError(
                    "Mixtral exact observation fingerprints differ",
                    details={"case_index": case_index, "observation": name},
                )
            source_fingerprints.append(source_fingerprint)
            ir_fingerprints.append(ir_fingerprint)
    checks = tuple(
        sorted(
            {
                "float32-router-softmax",
                "source-expert-index-accumulation-order",
                "exact-combined-moe-output",
                "exact-expert-output",
                "exact-logits",
                "exact-router-logits",
                "exact-routing-weights",
                "exact-selected-expert-indices",
                "finite-floating-observations",
            }
        )
    )
    payload = {
        "schema_version": MIXTRAL_SEMANTIC_CERTIFICATION_SCHEMA,
        "status": "passed",
        "gate": "G13-mixtral-routed-only-semantics",
        "artifact_id": executable.identity.artifact_id,
        "target_fingerprint": executable.identity.fingerprint,
        "case_input_fingerprints": input_fingerprints,
        "observation_names": list(observation_names),
        "source_observation_fingerprints": source_fingerprints,
        "ir_observation_fingerprints": ir_fingerprints,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "checks": list(checks),
        "execution_performed": True,
        "execution_certified": True,
        "production_runtime_eligible": False,
    }
    return MixtralSemanticCertification(
        artifact_id=executable.identity.artifact_id,
        target_fingerprint=executable.identity.fingerprint,
        case_input_fingerprints=tuple(input_fingerprints),
        observation_names=observation_names,
        source_observation_fingerprints=tuple(source_fingerprints),
        ir_observation_fingerprints=tuple(ir_fingerprints),
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        checks=checks,
        fingerprint=canonical_sha256(payload),
    )


def promote_reference_target(
    executable: ReferenceExecutable,
    certification: ReferenceExecutionCertification,
) -> ReferencePromotionRecord:
    """Promote one exact target/certification pair, while retaining the native boundary."""

    if not isinstance(certification, ReferenceExecutionCertification):
        raise TypeError("reference promotion requires a reference execution certification")
    if (
        certification.artifact_id != executable.identity.artifact_id
        or certification.target_fingerprint != executable.identity.fingerprint
        or not certification.execution_certified
    ):
        raise ReferenceParityError("execution certification is not bound to this reference target")
    payload = {
        "schema_version": REFERENCE_PROMOTION_SCHEMA,
        "status": "promoted-reference-only",
        "artifact_id": executable.identity.artifact_id,
        "target_fingerprint": executable.identity.fingerprint,
        "execution_certification_fingerprint": certification.fingerprint,
        "execution_certified": True,
        "production_runtime_eligible": False,
        "native_lowering_status": "required-separate-target-lowering",
    }
    return ReferencePromotionRecord(
        artifact_id=executable.identity.artifact_id,
        target_fingerprint=executable.identity.fingerprint,
        execution_certification_fingerprint=certification.fingerprint,
        fingerprint=canonical_sha256(payload),
    )


def require_native_lowering(_: ReferencePromotionRecord) -> None:
    """Explicitly reject treating a CPU reference promotion as a native runtime build."""

    raise NativeLoweringUnavailable(
        "reference execution does not lower allocation layouts, kernels, or state to MLX/CUDA",
        details={
            "required_next_gate": "registered-target-lowering-with-transactional-state",
            "reference_target": REFERENCE_TARGET_ID,
        },
    )


__all__ = [
    "ArtifactTensorStore",
    "MIXTRAL_SEMANTIC_CERTIFICATION_SCHEMA",
    "MixtralSemanticCertification",
    "MixtralSemanticParityError",
    "NativeLoweringUnavailable",
    "REFERENCE_CERTIFICATION_SCHEMA",
    "REFERENCE_PROMOTION_SCHEMA",
    "REFERENCE_TARGET_ID",
    "REFERENCE_TARGET_SCHEMA",
    "REFERENCE_TARGET_VERSION",
    "ReferenceExecutable",
    "ReferenceExecutionCertification",
    "ReferenceExecutionError",
    "ReferenceForwardResult",
    "ReferenceLoweringError",
    "ReferenceParityError",
    "ReferencePromotionRecord",
    "ReferenceTargetIdentity",
    "lower_component_artifact_to_reference",
    "promote_reference_target",
    "require_native_lowering",
    "run_g8_reference_parity",
    "run_g13_mixtral_semantic_parity",
]
