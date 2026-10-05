"""Fail-closed promotions for resident templates and later runtime capabilities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

RESIDENT_TEMPLATE_PROMOTION_SCHEMA = "mrun-resident-template-promotion-v1"
RUNTIME_CAPABILITY_PROMOTION_SCHEMA = "mrun-runtime-capability-promotion-v1"


def _sha256(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ResidentTemplatePromotion:
    promotion_id: str
    arena_fingerprint: str
    template_implementation_sha256: str
    mutable_binding_schema_sha256: str
    numerical_contract: str
    output_contract: str
    shape_bucket: tuple[int, int, int]
    device_identity: str
    runtime_identity_sha256: str
    arena_bytes: int
    template_bytes: int
    setup_ms: float
    rebind_ms: float
    replay_speedup_lower_95: float
    lane_count: int
    contamination_trials: int
    cancellation_passed: bool
    exact_parity: bool
    evidence_sha256: str
    wheel_sha256: str
    schema: str = RESIDENT_TEMPLATE_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESIDENT_TEMPLATE_PROMOTION_SCHEMA or not self.promotion_id:
            raise ValueError("resident-template promotion identity is invalid")
        for field in (
            "arena_fingerprint",
            "template_implementation_sha256",
            "mutable_binding_schema_sha256",
            "runtime_identity_sha256",
            "evidence_sha256",
            "wheel_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if not self.numerical_contract or not self.output_contract or not self.device_identity:
            raise ValueError("resident-template promotion contract is incomplete")
        bucket = tuple(int(value) for value in self.shape_bucket)
        if len(bucket) != 3 or min(bucket) <= 0:
            raise ValueError("resident-template shape bucket is invalid")
        object.__setattr__(self, "shape_bucket", bucket)
        for field in ("arena_bytes", "lane_count", "contamination_trials"):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.template_bytes < 0:
            raise ValueError("template bytes must be non-negative")
        for field in ("setup_ms", "rebind_ms"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        if not math.isfinite(self.replay_speedup_lower_95) or self.replay_speedup_lower_95 <= 1:
            raise ValueError("resident-template promotion requires a replay lower bound above one")
        if self.contamination_trials < 1_000:
            raise ValueError("resident-template promotion requires 1,000 contamination trials")
        if not self.cancellation_passed or not self.exact_parity:
            raise ValueError("resident-template promotion requires cancellation and exact parity")
        if self.template_bytes > self.arena_bytes // 10:
            raise ValueError("template duplicates more than 10% of resident arena bytes")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True)
class RuntimeCapabilityPromotion:
    capability: str
    implementation_sha256: str
    evidence_sha256: str
    numerical_contract: str
    device_identity: str
    parity_passed: bool
    performance_lower_95: float
    schema: str = RUNTIME_CAPABILITY_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        allowed = {
            "continuous_batch_capture",
            "profile_driven_fusion",
            "stateful_decode_capture",
            "target_aligned_k4",
            "fp8_dense",
            "int4_dense",
            "route_first_moe",
        }
        if self.schema != RUNTIME_CAPABILITY_PROMOTION_SCHEMA or self.capability not in allowed:
            raise ValueError("unsupported runtime capability promotion")
        for field in ("implementation_sha256", "evidence_sha256"):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if not self.numerical_contract or not self.device_identity or not self.parity_passed:
            raise ValueError("runtime capability promotion contract is incomplete")
        minimum = 1.05 if self.capability == "profile_driven_fusion" else 1.0
        if not math.isfinite(self.performance_lower_95) or self.performance_lower_95 <= minimum:
            raise ValueError("runtime capability promotion did not clear its performance gate")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


BUILTIN_RUNTIME_CAPABILITY_PROMOTIONS = (
    RuntimeCapabilityPromotion(
        capability="continuous_batch_capture",
        implementation_sha256=(
            "8314d90d4822b78024375023099f95b2534f7f37cd297938773e730ef803223b"
        ),
        evidence_sha256="df251ac23172a3d4ed16653f41bc33450c8937f7e12a422ec8f07a847391afb4",
        numerical_contract="row-stable-triton-v1",
        device_identity="NVIDIA GeForce RTX 4080|cc8.9|vram16718168064",
        parity_passed=True,
        performance_lower_95=13.138606786610225,
    ),
)


def select_runtime_capability_promotion(
    records: tuple[RuntimeCapabilityPromotion, ...],
    *,
    capability: str,
    numerical_contract: str,
    device_identity: str,
) -> RuntimeCapabilityPromotion | None:
    matches = tuple(
        record
        for record in records
        if record.capability == capability
        and record.numerical_contract == numerical_contract
        and record.device_identity == device_identity
    )
    if len(matches) > 1:
        raise RuntimeError("runtime capability promotion registry is ambiguous")
    return matches[0] if matches else None


def select_resident_template_promotion(
    records: tuple[ResidentTemplatePromotion, ...],
    *,
    arena_fingerprint: str,
    numerical_contract: str,
    output_contract: str,
    shape_bucket: tuple[int, int, int],
    device_identity: str,
) -> ResidentTemplatePromotion | None:
    matches = tuple(
        record
        for record in records
        if record.arena_fingerprint == arena_fingerprint
        and record.numerical_contract == numerical_contract
        and record.output_contract == output_contract
        and record.shape_bucket == shape_bucket
        and record.device_identity == device_identity
    )
    if len(matches) > 1:
        raise RuntimeError("resident-template promotion registry is ambiguous")
    return matches[0] if matches else None


__all__ = [
    "BUILTIN_RUNTIME_CAPABILITY_PROMOTIONS",
    "RESIDENT_TEMPLATE_PROMOTION_SCHEMA",
    "RUNTIME_CAPABILITY_PROMOTION_SCHEMA",
    "ResidentTemplatePromotion",
    "RuntimeCapabilityPromotion",
    "select_resident_template_promotion",
    "select_runtime_capability_promotion",
]
