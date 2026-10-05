"""Production assembly for one warm, native decomposed-model inference process.

This module is deliberately the only place where the native engine, capability/placement
contract, transactional runtime, tokenizer boundary, optional exact-prefix store, generation
coordinator, and HTTP host are assembled.  Keeping assembly explicit prevents the public server
from silently selecting a compatibility backend or crossing a model/tokenizer identity boundary.
"""

from __future__ import annotations

import math
import os
import stat
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from mrun.decompiler import VerifiedMixtralExpertStore
from mrun.runtime import (
    CompatibleBatchLane,
    DenseCudaCompatibleBatchLane,
    DenseCudaComponentExecutionBackend,
    FallbackPolicy,
    MixtralExpertPagedHardwareEvidence,
    MixtralExpertPagedRuntimeSlice,
    MixtralTieredPlacement,
    MlxAffineKVCodec,
    MlxCompatibleBatchLane,
    MlxComponentExecutionBackend,
    MlxMambaComponentExecutionBackend,
    OutputMode,
    WorkloadSpec,
    describe_cuda_device,
    describe_mlx_device,
    plan_mixtral_tiered_placement,
    probe_mixtral_expert_paged_hardware,
)
from mrun.runtime.inference import (
    NativeGenerationService,
    NativeSessionStore,
    SessionIdentity,
    ShutdownMode,
)

from .app import InferenceAppConfig, InferenceHost, create_inference_app
from .chat import BoundChatTokenizer, canonical_chat_template_sha256

NATIVE_INFERENCE_BACKENDS = (
    "mlx-source",
    "mlx-source-q4",
    "mlx-source-q3",
    "mlx-source-q2",
    "mlx-source-hybrid-q8",
    "mlx-source-hybrid-bf16",
    "cuda-source-int8",
    "cuda-source-int8-compact-head",
    "mlx-q4",
    "mlx-q8",
    "dense-cuda",
)
_SOURCE_INFERENCE_BACKENDS = frozenset(
    {
        "mlx-source",
        "mlx-source-q4",
        "mlx-source-q3",
        "mlx-source-q2",
        "mlx-source-hybrid-q8",
        "mlx-source-hybrid-bf16",
        "cuda-source-int8",
        "cuda-source-int8-compact-head",
    }
)
_MLX_INFERENCE_BACKENDS = frozenset(
    {
        "mlx-source",
        "mlx-source-q4",
        "mlx-source-q3",
        "mlx-source-q2",
        "mlx-source-hybrid-q8",
        "mlx-source-hybrid-bf16",
        "mlx-q4",
        "mlx-q8",
    }
)


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _finite_positive(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return normalized


def _finite_nonnegative(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class MixtralExpertPagedSliceConfig:
    """Explicit non-serving configuration for the experimental Mixtral MoE slice."""

    expert_store: Path
    expert_cache_capacity_bytes: int
    memory_budget_bytes: int | None = None
    workspace_bytes: int = 0
    headroom_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_store", Path(self.expert_store).expanduser())
        _positive_int(self.expert_cache_capacity_bytes, "expert_cache_capacity_bytes")
        if self.memory_budget_bytes is not None:
            _positive_int(self.memory_budget_bytes, "memory_budget_bytes")
        _nonnegative_int(self.workspace_bytes, "workspace_bytes")
        _nonnegative_int(self.headroom_bytes, "headroom_bytes")

    def as_dict(self) -> dict[str, Any]:
        return {
            "expert_store": str(self.expert_store),
            "expert_cache_capacity_bytes": self.expert_cache_capacity_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "workspace_bytes": self.workspace_bytes,
            "headroom_bytes": self.headroom_bytes,
        }


@dataclass(frozen=True, slots=True)
class NativeInferenceConfig:
    """Complete non-secret configuration for one resident inference route."""

    backend: str
    model_id: str
    component_graph: Path | None = None
    source_artifact: Path | None = None
    native_artifact: Path | None = None
    native_artifact_root: Path | None = None
    chat_template_file: Path | None = None
    chat_template_sha256: str | None = None
    context_tokens: int = 4096
    headroom_bytes: int = 256 * 1024**2
    memory_budget_bytes: int | None = None
    max_active_requests: int = 32
    max_new_tokens: int = 4096
    event_queue_capacity: int = 4098
    max_http_requests: int = 64
    max_request_bytes: int = 1_048_576
    default_max_tokens: int = 256
    request_timeout_seconds: float | None = 120.0
    expose_metrics: bool = True
    cuda_device_index: int = 0
    cuda_compute_dtype: str = "bf16"
    cuda_component_cache_mb: tuple[tuple[str, float], ...] = ()
    cuda_resident_head_mb: float | None = None
    cuda_require_triton: bool = True
    cuda_decode_attention_mode: str = "established"
    cuda_decode_attention_tile: int = 64
    cuda_body_fusion_mode: str = "established"
    cuda_compatible_batch_size: int = 1
    cuda_batch_queue_delay_seconds: float = 0.002
    cuda_batch_scratch_bytes: int | None = None
    session_cache_bytes: int = 0
    session_max_entries: int = 128
    session_ttl_seconds: float = 900.0
    mlx_prefill_chunk_size: int | None = None
    mlx_compatible_batch_size: int = 1
    mlx_batch_queue_delay_seconds: float = 0.002
    mlx_batch_scratch_bytes: int | None = None
    mlx_kv_bits: int | None = None
    mlx_kv_group_size: int = 64
    mlx_paged_kv_page_size: int | None = None
    mlx_paged_kv_page_count: int | None = None
    mlx_paged_decode_attention: bool = False

    def __post_init__(self) -> None:
        if self.backend not in NATIVE_INFERENCE_BACKENDS:
            raise ValueError("backend must be one of: " + ", ".join(NATIVE_INFERENCE_BACKENDS))
        if (
            type(self.model_id) is not str
            or not self.model_id
            or self.model_id.strip() != self.model_id
        ):
            raise ValueError("model_id must be a canonical non-empty string")
        for field in (
            "component_graph",
            "source_artifact",
            "native_artifact",
            "native_artifact_root",
            "chat_template_file",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, Path(value).expanduser())
        if self.backend in _SOURCE_INFERENCE_BACKENDS:
            if self.source_artifact is None or self.component_graph is not None:
                raise ValueError(
                    "each direct-source backend requires source_artifact and forbids a "
                    "transitional component_graph"
                )
        elif self.component_graph is None or self.source_artifact is not None:
            raise ValueError(
                "QStore-derived backends require component_graph and forbid source_artifact"
            )
        if self.native_artifact is not None and self.native_artifact_root is not None:
            raise ValueError("native_artifact and native_artifact_root are mutually exclusive")
        if self.chat_template_sha256 is not None:
            digest = self.chat_template_sha256
            if (
                type(digest) is not str
                or len(digest) != 64
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("chat_template_sha256 must be a lowercase SHA-256 digest")
            if self.chat_template_file is None:
                raise ValueError("chat_template_sha256 requires chat_template_file")
        for field in (
            "context_tokens",
            "max_active_requests",
            "max_new_tokens",
            "event_queue_capacity",
            "max_http_requests",
            "max_request_bytes",
            "default_max_tokens",
            "session_max_entries",
        ):
            object.__setattr__(self, field, _positive_int(getattr(self, field), field))
        for field in ("headroom_bytes", "session_cache_bytes", "cuda_device_index"):
            object.__setattr__(self, field, _nonnegative_int(getattr(self, field), field))
        if self.memory_budget_bytes is not None:
            object.__setattr__(
                self,
                "memory_budget_bytes",
                _positive_int(self.memory_budget_bytes, "memory_budget_bytes"),
            )
        if self.mlx_prefill_chunk_size is not None:
            object.__setattr__(
                self,
                "mlx_prefill_chunk_size",
                _positive_int(self.mlx_prefill_chunk_size, "mlx_prefill_chunk_size"),
            )
            if self.mlx_prefill_chunk_size > self.context_tokens:
                raise ValueError("mlx_prefill_chunk_size cannot exceed context_tokens")
        object.__setattr__(
            self,
            "mlx_compatible_batch_size",
            _positive_int(self.mlx_compatible_batch_size, "mlx_compatible_batch_size"),
        )
        if self.mlx_compatible_batch_size > self.max_active_requests:
            raise ValueError("mlx_compatible_batch_size cannot exceed max_active_requests")
        object.__setattr__(
            self,
            "mlx_batch_queue_delay_seconds",
            _finite_nonnegative(
                self.mlx_batch_queue_delay_seconds,
                "mlx_batch_queue_delay_seconds",
            ),
        )
        if self.mlx_batch_scratch_bytes is not None:
            object.__setattr__(
                self,
                "mlx_batch_scratch_bytes",
                _positive_int(self.mlx_batch_scratch_bytes, "mlx_batch_scratch_bytes"),
            )
            if self.mlx_compatible_batch_size == 1:
                raise ValueError(
                    "mlx_batch_scratch_bytes requires mlx_compatible_batch_size greater than 1"
                )
        object.__setattr__(
            self,
            "mlx_kv_group_size",
            _positive_int(self.mlx_kv_group_size, "mlx_kv_group_size"),
        )
        if self.mlx_kv_bits is not None:
            object.__setattr__(
                self,
                "mlx_kv_bits",
                _positive_int(self.mlx_kv_bits, "mlx_kv_bits"),
            )
            if self.mlx_kv_bits != 4 or self.mlx_kv_group_size != 64:
                raise ValueError("the production service currently admits only MLX KV4 group-64")
            if self.mlx_compatible_batch_size != 1:
                raise ValueError("MLX KV4 currently requires the transactional B1 execution lane")
            if self.session_cache_bytes:
                raise ValueError(
                    "MLX KV4 retained sessions are disabled until full-versus-segmented "
                    "execution passes the numerical promotion gate"
                )
        elif self.mlx_kv_group_size != 64:
            raise ValueError("mlx_kv_group_size requires mlx_kv_bits")
        if (self.mlx_paged_kv_page_size is None) != (self.mlx_paged_kv_page_count is None):
            raise ValueError(
                "MLX paged K/V requires both mlx_paged_kv_page_size and mlx_paged_kv_page_count"
            )
        if self.mlx_paged_kv_page_size is not None:
            object.__setattr__(
                self,
                "mlx_paged_kv_page_size",
                _positive_int(self.mlx_paged_kv_page_size, "mlx_paged_kv_page_size"),
            )
            object.__setattr__(
                self,
                "mlx_paged_kv_page_count",
                _positive_int(self.mlx_paged_kv_page_count, "mlx_paged_kv_page_count"),
            )
            if self.mlx_paged_kv_page_size & (self.mlx_paged_kv_page_size - 1):
                raise ValueError("mlx_paged_kv_page_size must be a power of two")
            if self.mlx_paged_kv_page_size * self.mlx_paged_kv_page_count < self.context_tokens:
                raise ValueError("MLX paged K/V pool must hold at least one maximum-context state")
            if self.mlx_kv_bits is not None:
                raise ValueError("MLX paged BF16 K/V cannot be combined with MLX KV4")
            if self.mlx_compatible_batch_size != 1:
                raise ValueError(
                    "MLX paged K/V currently requires the transactional B1 execution lane"
                )
        if type(self.mlx_paged_decode_attention) is not bool:
            raise TypeError("mlx_paged_decode_attention must be boolean")
        if self.mlx_paged_decode_attention:
            if self.mlx_paged_kv_page_size is None:
                raise ValueError(
                    "MLX paged decode attention requires explicit BF16 page size and page count"
                )
            if self.mlx_prefill_chunk_size is not None:
                raise ValueError(
                    "MLX paged decode attention and chunked prefill require separate routes"
                )
        if self.event_queue_capacity < 2:
            raise ValueError("event_queue_capacity must reserve token and terminal events")
        if self.default_max_tokens > self.max_new_tokens:
            raise ValueError("default_max_tokens cannot exceed max_new_tokens")
        if self.cuda_compute_dtype not in {"bf16", "fp16"}:
            raise ValueError("cuda_compute_dtype must be bf16 or fp16")
        if self.cuda_decode_attention_mode not in {
            "established",
            "segmented-flash-gqa-decode-v1",
        }:
            raise ValueError(
                "cuda_decode_attention_mode must be 'established' or "
                "'segmented-flash-gqa-decode-v1'"
            )
        if (
            isinstance(self.cuda_decode_attention_tile, bool)
            or not isinstance(self.cuda_decode_attention_tile, int)
            or self.cuda_decode_attention_tile < 16
            or self.cuda_decode_attention_tile > 256
            or self.cuda_decode_attention_tile & (self.cuda_decode_attention_tile - 1)
        ):
            raise ValueError("cuda_decode_attention_tile must be a power of two in [16, 256]")
        if (
            self.cuda_decode_attention_mode == "segmented-flash-gqa-decode-v1"
            and not self.cuda_require_triton
        ):
            raise ValueError("segmented CUDA decode attention requires cuda_require_triton")
        if self.cuda_body_fusion_mode not in {
            "established",
            "residual-rms-swiglu-v1",
        }:
            raise ValueError(
                "cuda_body_fusion_mode must be 'established' or 'residual-rms-swiglu-v1'"
            )
        if self.cuda_body_fusion_mode != "established" and not self.cuda_require_triton:
            raise ValueError("fused CUDA transformer body requires cuda_require_triton")
        if self.cuda_body_fusion_mode != "established" and self.backend not in {
            "cuda-source-int8",
            "cuda-source-int8-compact-head",
        }:
            raise ValueError("fused CUDA transformer body currently requires a direct-source route")
        object.__setattr__(
            self,
            "cuda_compatible_batch_size",
            _positive_int(self.cuda_compatible_batch_size, "cuda_compatible_batch_size"),
        )
        if self.cuda_compatible_batch_size > self.max_active_requests:
            raise ValueError("cuda_compatible_batch_size cannot exceed max_active_requests")
        object.__setattr__(
            self,
            "cuda_batch_queue_delay_seconds",
            _finite_nonnegative(
                self.cuda_batch_queue_delay_seconds,
                "cuda_batch_queue_delay_seconds",
            ),
        )
        if self.cuda_batch_scratch_bytes is not None:
            object.__setattr__(
                self,
                "cuda_batch_scratch_bytes",
                _positive_int(self.cuda_batch_scratch_bytes, "cuda_batch_scratch_bytes"),
            )
            if self.cuda_compatible_batch_size == 1:
                raise ValueError(
                    "cuda_batch_scratch_bytes requires cuda_compatible_batch_size greater than 1"
                )
        if (
            self.cuda_compatible_batch_size > 1
            and self.cuda_decode_attention_mode != "segmented-flash-gqa-decode-v1"
        ):
            raise ValueError(
                "CUDA compatible batching requires segmented-flash-gqa-decode-v1 attention"
            )
        if self.cuda_compatible_batch_size > 1 and self.session_cache_bytes:
            raise ValueError(
                "CUDA slot-pooled batching does not yet admit retained sessions; the fixed "
                "slot count currently belongs to active requests only"
            )
        budgets: list[tuple[str, float]] = []
        seen: set[str] = set()
        for role, value in self.cuda_component_cache_mb:
            if type(role) is not str or not role or role.strip() != role or role in seen:
                raise ValueError("CUDA component-cache roles must be unique canonical strings")
            seen.add(role)
            budgets.append((role, _finite_positive(value, f"cuda_component_cache_mb[{role}]")))
        object.__setattr__(self, "cuda_component_cache_mb", tuple(sorted(budgets)))
        if self.cuda_resident_head_mb is not None:
            object.__setattr__(
                self,
                "cuda_resident_head_mb",
                _finite_positive(self.cuda_resident_head_mb, "cuda_resident_head_mb"),
            )
        if (
            self.backend == "cuda-source-int8-compact-head"
            and self.cuda_resident_head_mb is not None
        ):
            raise ValueError("cuda-source-int8-compact-head forbids an expanded resident FP32 head")
        object.__setattr__(
            self,
            "session_ttl_seconds",
            _finite_positive(self.session_ttl_seconds, "session_ttl_seconds"),
        )
        if self.request_timeout_seconds is not None:
            object.__setattr__(
                self,
                "request_timeout_seconds",
                _finite_positive(self.request_timeout_seconds, "request_timeout_seconds"),
            )
        for field in ("expose_metrics", "cuda_require_triton"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be boolean")
        if self.backend.startswith("mlx-"):
            if (
                self.cuda_component_cache_mb
                or self.cuda_resident_head_mb is not None
                or self.cuda_decode_attention_mode != "established"
                or self.cuda_decode_attention_tile != 64
                or self.cuda_body_fusion_mode != "established"
                or self.cuda_compatible_batch_size != 1
                or self.cuda_batch_queue_delay_seconds != 0.002
                or self.cuda_batch_scratch_bytes is not None
            ):
                raise ValueError("CUDA residency options cannot be used with an MLX backend")
        else:
            if (
                self.mlx_prefill_chunk_size is not None
                or self.mlx_compatible_batch_size != 1
                or self.mlx_batch_queue_delay_seconds != 0.002
                or self.mlx_batch_scratch_bytes is not None
                or self.mlx_kv_bits is not None
                or self.mlx_kv_group_size != 64
                or self.mlx_paged_kv_page_size is not None
                or self.mlx_paged_kv_page_count is not None
                or self.mlx_paged_decode_attention
            ):
                raise ValueError("MLX execution options require an MLX backend")
            if self.backend not in {
                "cuda-source-int8",
                "cuda-source-int8-compact-head",
            } and (self.native_artifact is not None or self.native_artifact_root is not None):
                raise ValueError(
                    "native_artifact options apply only to direct-source or MLX backends"
                )

    @property
    def sessions_enabled(self) -> bool:
        return self.session_cache_bytes > 0

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _auto_cuda_residency(graph_path: Path) -> tuple[dict[str, float], float]:
    """Derive conservative full-residency budgets from a verified component graph.

    Component cache budgets use decimal MiB in the existing CUDA engine.  One extra byte is
    included before division so floating-point roundoff cannot truncate an exactly-sized cache.
    The separately resident FP32 output head is charged from its logical matrix shape.
    """

    from mrun.engine.kernels.composite_qstore import ComponentGraph, _resolve_alias

    graph = ComponentGraph(graph_path)
    budgets: dict[str, float] = {}
    for role, record in graph.components.items():
        blobs = record["blobs"]
        required = sum(int(blob["bytes"]) for blob in blobs.values())
        budgets[role] = (required + 1) / 1e6
    head = graph.logical_blocks[_resolve_alias(graph.logical_blocks, "lm_head")]
    shape = head.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("component graph has no matrix-shaped output head")
    head_bytes = int(shape[0]) * int(shape[1]) * 4
    return budgets, (head_bytes + 1) / 1e6


def _auto_direct_cuda_residency(artifact: Any) -> tuple[dict[str, float], float]:
    """Size complete role residency and the contract-exact dequantized FP32 head."""

    budgets: dict[str, float] = {}
    for role, component in artifact.components.items():
        required = sum(int(record["bytes"]) for record in component["blobs"].values())
        budgets[str(role)] = (required + 1) / 1e6
    blocks = artifact.blocks
    head = blocks.get("lm_head")
    seen = {"lm_head"}
    while isinstance(head, Mapping) and "alias" in head:
        target = str(head["alias"])
        if target in seen:
            raise ValueError("direct CUDA output-head alias is cyclic")
        seen.add(target)
        head = blocks.get(target)
    if not isinstance(head, Mapping) or head.get("kind") != "qrow":
        raise ValueError("direct CUDA artifact has no matrix output head")
    shape = head.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("direct CUDA output head has no matrix shape")
    head_bytes = int(shape[0]) * int(shape[1]) * 4
    return budgets, (head_bytes + 1) / 1e6


def _legacy_template_sha256(engine: Any) -> str:
    graph = getattr(engine, "graph", None)
    if graph is not None:
        descriptor = graph.raw.get("tokenizer", {})
        value = descriptor.get("chat_template_sha256")
    else:
        composite = getattr(engine, "composite_store", None)
        value = None if composite is None else composite.vocab.chat_template_sha256
    if type(value) is not str:
        raise RuntimeError("native component route has no graph-bound chat-template identity")
    return value


def _read_external_chat_template(path: Path) -> str:
    """Read one explicitly selected chat template from a stable file descriptor.

    The template is copied into the immutable route identity at startup, so the path is never
    consulted again while serving.  Symlinks and non-regular files are rejected to keep startup
    custody independent of mutable path redirection.
    """

    resolved = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError("chat_template_file must be a readable non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("chat_template_file must be a regular file")
        if before.st_size <= 0 or before.st_size > 1_048_576:
            raise ValueError("chat_template_file must contain between 1 and 1048576 bytes")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            len(raw) != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError("chat_template_file changed while it was being read")
    except OSError as exc:
        raise ValueError("chat_template_file could not be read") from exc
    finally:
        os.close(descriptor)
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("chat_template_file must contain UTF-8 text") from exc
    if not content or "\x00" in content:
        raise ValueError("chat_template_file must contain non-empty text without NUL bytes")
    return content


def _serving_workspace_bytes(config: NativeInferenceConfig, model: Any) -> int:
    """Charge live model-state rows beyond those already reserved by placement.

    ``WorkloadSpec`` describes the width of one native dispatch, while the coordinator may own
    several independent B1 states. Placement reserves one row normally or the complete physical
    CUDA decode width for the fixed slot pool; workspace carries the remaining active-request
    rows plus the separately retained exact-prefix session budget. Service admission enforces the
    corresponding count and size caps.
    """

    bytes_per_token = _nonnegative_int(
        model.state_bytes_per_token,
        "state_bytes_per_token",
    )
    fixed_bytes = _nonnegative_int(
        getattr(model, "state_fixed_bytes_per_row", 0),
        "state_fixed_bytes_per_row",
    )
    if bytes_per_token == 0 and fixed_bytes == 0:
        raise ValueError("native model state must have a per-token or fixed byte charge")
    if config.mlx_paged_kv_page_size is not None:
        if fixed_bytes:
            raise ValueError("paged K/V cannot be combined with fixed recurrent state")
        # The backend charges the fixed physical pool exactly once by replacing the placement's
        # ordinary B1 arena with (page_size * page_count) token slots. Active and retained logical
        # states draw from that same bounded pool and must not be charged as duplicate storage.
        return 0
    state_charge = fixed_bytes + config.context_tokens * bytes_per_token
    placement_rows = (
        config.cuda_compatible_batch_size
        if config.backend.startswith("cuda-") or config.backend == "dense-cuda"
        else 1
    )
    additional_active = max(config.max_active_requests - placement_rows, 0) * state_charge
    compatible_batch_scratch = 0
    if config.mlx_compatible_batch_size > 1:
        compatible_batch_scratch = config.mlx_batch_scratch_bytes or (
            state_charge * config.mlx_compatible_batch_size
        )
    return additional_active + compatible_batch_scratch + config.session_cache_bytes


def _validate_mlx_architecture_options(
    config: NativeInferenceConfig,
    architecture: str,
) -> None:
    """Reject transformer-state controls before binding a fixed-state recurrent backend."""

    if architecture != "mamba":
        return
    incompatible = {
        "mlx_compatible_batch_size": config.mlx_compatible_batch_size != 1,
        "mlx_batch_scratch_bytes": config.mlx_batch_scratch_bytes is not None,
        "mlx_kv_bits": config.mlx_kv_bits is not None,
        "mlx_paged_kv": config.mlx_paged_kv_page_size is not None,
        "mlx_paged_decode_attention": config.mlx_paged_decode_attention,
    }
    selected = tuple(name for name, enabled in incompatible.items() if enabled)
    if selected:
        raise ValueError(
            "fixed-state Mamba forbids transformer K/V and compatible-batch options: "
            + ", ".join(selected)
        )


class LoadedNativeInference:
    """Owned native route with deterministic drain and resource-release ordering."""

    def __init__(
        self,
        *,
        config: NativeInferenceConfig,
        engine: Any,
        device: Any,
        backend: Any,
        model: Any,
        workload: WorkloadSpec,
        placement: Any,
        runtime: Any,
        tokenizer: BoundChatTokenizer,
        session_store: NativeSessionStore | None,
        compatible_batch_lane: CompatibleBatchLane | None,
        service: NativeGenerationService,
        host: InferenceHost,
        app: Any,
    ) -> None:
        self.config = config
        self.engine = engine
        self.device = device
        self.backend = backend
        self.model = model
        self.workload = workload
        self.placement = placement
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.session_store = session_store
        self.compatible_batch_lane = compatible_batch_lane
        self.service = service
        self.host = host
        self.app = app
        self._closed = False
        self._close_lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        engine_report = getattr(self.engine, "runtime_report", None)
        if not callable(engine_report):
            engine_report = getattr(self.engine, "runtime_stats", None)
        session_telemetry = None if self.session_store is None else self.session_store.telemetry()
        capabilities = self.backend.capabilities(self.device)
        return {
            "schema": "mrun-loaded-native-inference-v1",
            "config": self.config.as_dict(),
            "model": self.model.as_dict(),
            "model_fingerprint": self.model.fingerprint,
            "device": self.device.as_dict(),
            "device_fingerprint": self.device.fingerprint,
            "capabilities": capabilities.as_dict(),
            "workload": self.workload.as_dict(),
            "placement": self.placement.as_dict(),
            "placement_fingerprint": self.placement.fingerprint,
            "route": _jsonable(self.runtime.route),
            "route_id": self.runtime.route.runtime_id,
            "tokenizer": {
                "model_id": self.tokenizer.model_id,
                "semantic_token_count": self.tokenizer.semantic_token_count,
                "context_size": self.tokenizer.context_size,
                "chat_template_sha256": self.tokenizer.chat_template_sha256,
                "legacy_raw_chat_template_sha256": (self.tokenizer.legacy_raw_chat_template_sha256),
            },
            "sessions": {
                "enabled": self.session_store is not None,
                "store": (
                    None
                    if session_telemetry is None
                    else {
                        "identity_fingerprint": session_telemetry.identity_fingerprint,
                        "state_abi": self.session_store.identity.state_abi,
                        "entries": session_telemetry.entries,
                        "retired_entries": session_telemetry.retired_entries,
                        "active_leases": session_telemetry.active_leases,
                        "pinned_sources": session_telemetry.pinned_sources,
                        "stored_bytes": session_telemetry.stored_bytes,
                        "reserved_bytes": session_telemetry.reserved_bytes,
                        "reserved_slots": session_telemetry.reserved_slots,
                        "max_entries": session_telemetry.max_entries,
                        "max_bytes": session_telemetry.max_bytes,
                        "budget_reconciled": session_telemetry.budget_reconciled,
                        "accepting": session_telemetry.accepting,
                        "poisoned": session_telemetry.poisoned,
                        "cross_session_prefix": {
                            "hits": session_telemetry.cross_session_prefix_hits,
                            "tokens": session_telemetry.cross_session_prefix_tokens,
                            "bytes": session_telemetry.cross_session_prefix_bytes,
                        },
                    }
                ),
            },
            "execution": {
                "kv_state": (
                    {
                        "codec_id": self.backend.kv_codec.codec_id,
                        "bits": self.backend.kv_codec.bits,
                        "group_size": self.backend.kv_codec.group_size,
                        "state_abi": self.model.state_abi,
                        "bytes_per_token": self.model.state_bytes_per_token,
                        "promotion_status": self.backend.capabilities(
                            self.device
                        ).promotion_status.value,
                    }
                    if getattr(self.backend, "kv_codec", None) is not None
                    else (
                        {
                            "storage": "block-paged-device-bf16",
                            "cache_abi": self.model.state_abi,
                            "page_size": self.backend.paged_kv_page_size,
                            "page_count": self.backend.paged_kv_page_count,
                            "physical_bytes": self.backend.paged_kv_physical_bytes,
                            "bytes_per_token": self.model.state_bytes_per_token,
                            "attention_materialization": (
                                "metal-block-paged-k1"
                                if getattr(
                                    self.runtime,
                                    "paged_decode_attention_identity",
                                    None,
                                )
                                is not None
                                else "device-concatenate"
                            ),
                            "promotion_status": self.backend.capabilities(
                                self.device
                            ).promotion_status.value,
                        }
                        if getattr(self.backend, "paged_kv_page_size", None) is not None
                        else None
                    )
                ),
                "recurrent_state": (
                    None
                    if getattr(self.model, "state_fixed_bytes_per_row", 0) == 0
                    else {
                        "storage": "fixed-device-recurrence",
                        "state_abi": self.model.state_abi,
                        "fixed_bytes_per_row": self.model.state_fixed_bytes_per_row,
                        "bytes_per_token": self.model.state_bytes_per_token,
                        "promotion_status": self.backend.capabilities(
                            self.device
                        ).promotion_status.value,
                    }
                ),
                "prefill_shape": _jsonable(getattr(self.runtime, "prefill_execution_shape", None)),
                "paged_decode_attention": (
                    None
                    if getattr(self.runtime, "paged_decode_attention_identity", None) is None
                    else {
                        "identity": _jsonable(self.runtime.paged_decode_attention_identity),
                        "telemetry": _jsonable(self.runtime.paged_decode_attention_telemetry()),
                    }
                ),
                "compatible_batch": (
                    None
                    if self.compatible_batch_lane is None
                    else {
                        "identity": _jsonable(self.compatible_batch_lane.identity),
                        "telemetry": _jsonable(self.compatible_batch_lane.telemetry()),
                    }
                ),
            },
            "engine": _jsonable(engine_report() if callable(engine_report) else {}),
        }

    def close(self, *, drain_timeout_seconds: float = 30.0) -> None:
        timeout = _finite_positive(drain_timeout_seconds, "drain_timeout_seconds")
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        errors: list[BaseException] = []
        self.host.begin_drain()
        try:
            stopped = self.service.shutdown(ShutdownMode.DRAIN, wait=True, timeout=timeout)
            if not stopped:
                self.service.shutdown(ShutdownMode.ABORT, wait=True, timeout=timeout)
        except BaseException as exc:
            errors.append(exc)
            try:
                self.service.shutdown(ShutdownMode.ABORT, wait=True, timeout=timeout)
            except BaseException as cleanup_exc:
                errors.append(cleanup_exc)
        self.host.mark_unready()
        if self.session_store is not None:
            try:
                self.session_store.close()
            except BaseException as exc:
                errors.append(exc)
        try:
            self.runtime.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            summary = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            raise RuntimeError(f"native inference shutdown failed: {summary}") from errors[0]

    def __enter__(self) -> LoadedNativeInference:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class LoadedMixtralExpertPagedSlice:
    """Owned, hardware-admitted Mixtral skeleton/cache slice without a chat data plane."""

    def __init__(
        self,
        *,
        config: MixtralExpertPagedSliceConfig,
        artifact: VerifiedMixtralExpertStore,
        hardware_evidence: MixtralExpertPagedHardwareEvidence,
        admission_device: Any,
        admission_ceiling_bytes: int,
        placement: MixtralTieredPlacement,
        runtime: MixtralExpertPagedRuntimeSlice,
    ) -> None:
        self.config = config
        self.artifact = artifact
        self.hardware_evidence = hardware_evidence
        self.admission_device = admission_device
        self.admission_ceiling_bytes = admission_ceiling_bytes
        self.placement = placement
        self.runtime = runtime
        self._closed = False
        self._close_lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Mixtral expert-paged slice is closed")
        return {
            "schema": "mrun-loaded-mixtral-expert-paged-slice-v1",
            "config": self.config.as_dict(),
            "artifact": {
                "schema": self.artifact.manifest["schema"],
                "artifact_sha256": self.artifact.artifact_sha256,
                "build_key_sha256": self.artifact.build_key_sha256,
                "source": self.artifact.source,
                "topology": self.artifact.topology,
                "skeleton_tensor_bytes": self.artifact.skeleton_tensor_bytes,
                "expert_page_tensor_bytes": self.artifact.expert_page_tensor_bytes,
                "expert_store_tensor_bytes": self.artifact.expert_store_tensor_bytes,
            },
            "hardware_evidence": self.hardware_evidence.as_dict(),
            "admission_device": self.admission_device.as_dict(),
            "admission_ceiling_bytes": self.admission_ceiling_bytes,
            "placement": self.placement.as_dict(),
            "accounting": self.runtime.accounting(),
            "claim_boundary": {
                "explicit_opt_in": True,
                "full_model_runtime": False,
                "attention_implemented": False,
                "kv_state_implemented": False,
                "token_io_implemented": False,
                "real_checkpoint_execution_certified": False,
                "performance_claim_valid": False,
                "production_runtime_eligible": False,
            },
        }

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self.runtime.close()
            self._closed = True

    def __enter__(self) -> LoadedMixtralExpertPagedSlice:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def load_mixtral_expert_paged_slice(
    config: MixtralExpertPagedSliceConfig,
) -> LoadedMixtralExpertPagedSlice:
    """Open the isolated slice; never register it as a full native inference backend."""

    if not isinstance(config, MixtralExpertPagedSliceConfig):
        raise TypeError("config must be MixtralExpertPagedSliceConfig")
    if not config.expert_store.is_dir():
        raise FileNotFoundError(f"Mixtral expert store is not a directory: {config.expert_store}")
    artifact = VerifiedMixtralExpertStore(config.expert_store)
    artifact.assert_unchanged()
    evidence = probe_mixtral_expert_paged_hardware()
    admission_device = describe_mlx_device()
    admission_ceiling = min(
        evidence.max_recommended_working_set_size_bytes,
        admission_device.available_bytes,
    )
    memory_budget = (
        admission_ceiling if config.memory_budget_bytes is None else config.memory_budget_bytes
    )
    if memory_budget > admission_ceiling:
        raise ValueError(
            "Mixtral slice memory budget exceeds live available unified-memory admission"
        )
    placement = plan_mixtral_tiered_placement(
        artifact,
        hardware_evidence=evidence,
        expert_cache_capacity_bytes=config.expert_cache_capacity_bytes,
        memory_budget_bytes=memory_budget,
        workspace_bytes=config.workspace_bytes,
        headroom_bytes=config.headroom_bytes,
    )
    runtime: MixtralExpertPagedRuntimeSlice | None = None
    try:
        runtime = MixtralExpertPagedRuntimeSlice(
            artifact,
            placement=placement,
            hardware_evidence=evidence,
        )
        artifact.assert_unchanged()
        return LoadedMixtralExpertPagedSlice(
            config=config,
            artifact=artifact,
            hardware_evidence=evidence,
            admission_device=admission_device,
            admission_ceiling_bytes=admission_ceiling,
            placement=placement,
            runtime=runtime,
        )
    except BaseException:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException:
                pass
        raise


def load_native_inference(
    config: NativeInferenceConfig,
    *,
    bearer_token: str | None,
) -> LoadedNativeInference:
    """Open one fail-closed decomposed native route and its public chat data plane."""

    if not isinstance(config, NativeInferenceConfig):
        raise TypeError("config must be NativeInferenceConfig")
    if config.backend in _SOURCE_INFERENCE_BACKENDS:
        assert config.source_artifact is not None
        if not config.source_artifact.is_dir():
            raise FileNotFoundError(
                f"canonical source artifact is not a directory: {config.source_artifact}"
            )
    else:
        assert config.component_graph is not None
        if not config.component_graph.is_file():
            raise FileNotFoundError(f"component graph is not a file: {config.component_graph}")
    if bearer_token is not None and (type(bearer_token) is not str or not bearer_token):
        raise ValueError("bearer_token must be a non-empty string or None")

    engine: Any | None = None
    runtime: Any | None = None
    service: NativeGenerationService | None = None
    session_store: NativeSessionStore | None = None
    compatible_batch_lane: CompatibleBatchLane | None = None
    try:
        if config.backend in _MLX_INFERENCE_BACKENDS:
            from mrun.engine.mlx_component import MLXComponentEngine, MLXComponentQ4Engine

            if config.backend in _SOURCE_INFERENCE_BACKENDS:
                if config.backend in {"mlx-source-hybrid-q8", "mlx-source-hybrid-bf16"}:
                    from mrun.decompiler.mlx_hybrid import (
                        MLXSourceHybridBF16Engine,
                        MLXSourceHybridQ8Engine,
                    )

                    engine_type = {
                        "mlx-source-hybrid-q8": MLXSourceHybridQ8Engine,
                        "mlx-source-hybrid-bf16": MLXSourceHybridBF16Engine,
                    }[config.backend]
                elif config.backend in {"mlx-source-q3", "mlx-source-q2"}:
                    from mrun.decompiler.mlx_lowbit import (
                        MLXSourceQ2Engine,
                        MLXSourceQ3Engine,
                    )

                    engine_type = {
                        "mlx-source-q3": MLXSourceQ3Engine,
                        "mlx-source-q2": MLXSourceQ2Engine,
                    }[config.backend]
                elif config.backend in {"mlx-source", "mlx-source-q4"}:
                    from mrun.decompiler.mlx_native import (
                        MLXSourceComponentEngine,
                        MLXSourceQ4Engine,
                    )

                    engine_type = {
                        "mlx-source": MLXSourceComponentEngine,
                        "mlx-source-q4": MLXSourceQ4Engine,
                    }[config.backend]
                else:
                    raise RuntimeError(
                        f"source backend has no exact engine binding: {config.backend!r}"
                    )
                engine = engine_type(
                    config.model_id,
                    source_artifact=config.source_artifact,
                    native_artifact=config.native_artifact,
                    native_root=config.native_artifact_root,
                )
            else:
                engine_type = {
                    "mlx-q4": MLXComponentQ4Engine,
                    "mlx-q8": MLXComponentEngine,
                }[config.backend]
                engine = engine_type(
                    config.model_id,
                    component_graph=config.component_graph,
                    native_artifact=config.native_artifact,
                    native_root=config.native_artifact_root,
                )
            device = describe_mlx_device()
            if str(getattr(engine, "arch", "")) == "mamba":
                _validate_mlx_architecture_options(config, "mamba")
                backend = MlxMambaComponentExecutionBackend(
                    engine,
                    device,
                    prefill_chunk_size=config.mlx_prefill_chunk_size,
                    owns_engine=True,
                )
            else:
                backend = MlxComponentExecutionBackend(
                    engine,
                    device,
                    kv_codec=(
                        None
                        if config.mlx_kv_bits is None
                        else MlxAffineKVCodec(
                            bits=config.mlx_kv_bits,
                            group_size=config.mlx_kv_group_size,
                        )
                    ),
                    prefill_chunk_size=config.mlx_prefill_chunk_size,
                    paged_kv_page_size=config.mlx_paged_kv_page_size,
                    paged_kv_page_count=config.mlx_paged_kv_page_count,
                    paged_decode_attention=config.mlx_paged_decode_attention,
                    owns_engine=True,
                )
        else:
            from mrun.engine.dense_qstore_cuda import (
                DenseQStoreCudaEngine,
                DenseSourceCudaInt8CompactHeadEngine,
                DenseSourceCudaInt8Engine,
            )

            device = describe_cuda_device(config.cuda_device_index)
            if config.backend in {
                "cuda-source-int8",
                "cuda-source-int8-compact-head",
            }:
                from mrun.decompiler.cuda_native import (
                    VerifiedSourceCudaInt8Artifact,
                    build_source_cuda_int8_artifact,
                )

                assert config.source_artifact is not None
                if config.native_artifact is None:
                    native_root = (
                        config.native_artifact_root
                        or Path("~/.cache/mrun/cuda-source-int8").expanduser()
                    )
                    native_path = build_source_cuda_int8_artifact(
                        config.source_artifact, native_root
                    ).path
                else:
                    native_path = config.native_artifact
                artifact = VerifiedSourceCudaInt8Artifact(
                    native_path, source_artifact=config.source_artifact
                )
                auto_budgets, auto_head_mb = _auto_direct_cuda_residency(artifact)
                budgets = dict(config.cuda_component_cache_mb) or auto_budgets
                compact_head = config.backend == "cuda-source-int8-compact-head"
                head_mb = None if compact_head else config.cuda_resident_head_mb or auto_head_mb
                engine_type = (
                    DenseSourceCudaInt8CompactHeadEngine
                    if compact_head
                    else DenseSourceCudaInt8Engine
                )
                engine = engine_type(
                    config.model_id,
                    source_artifact=config.source_artifact,
                    native_artifact=native_path,
                    output_contract="full_logits",
                    component_cache_mb=budgets,
                    resident_exact_head_mb=head_mb,
                    device=f"cuda:{config.cuda_device_index}",
                    compute_dtype=config.cuda_compute_dtype,
                    max_seq_len=config.context_tokens,
                    require_triton=config.cuda_require_triton,
                    decode_attention_mode=config.cuda_decode_attention_mode,
                    decode_attention_tile=config.cuda_decode_attention_tile,
                    body_fusion_mode=config.cuda_body_fusion_mode,
                )
            else:
                assert config.component_graph is not None
                auto_budgets, auto_head_mb = _auto_cuda_residency(config.component_graph)
                budgets = dict(config.cuda_component_cache_mb) or auto_budgets
                head_mb = config.cuda_resident_head_mb or auto_head_mb
                engine = DenseQStoreCudaEngine(
                    config.model_id,
                    component_graph=config.component_graph,
                    output_contract="full_logits",
                    component_cache_mb=budgets,
                    resident_exact_head_mb=head_mb,
                    device=f"cuda:{config.cuda_device_index}",
                    compute_dtype=config.cuda_compute_dtype,
                    max_seq_len=config.context_tokens,
                    require_triton=config.cuda_require_triton,
                    decode_attention_mode=config.cuda_decode_attention_mode,
                    decode_attention_tile=config.cuda_decode_attention_tile,
                    body_fusion_mode=config.cuda_body_fusion_mode,
                )
            backend = DenseCudaComponentExecutionBackend(
                engine,
                device,
                max_batch_size=config.cuda_compatible_batch_size,
                owns_engine=True,
            )

        capabilities = backend.capabilities(device)
        model = backend.model
        serving_workspace_bytes = _serving_workspace_bytes(config, model)
        workload = WorkloadSpec(
            max_batch_size=(
                config.cuda_compatible_batch_size
                if not config.backend.startswith("mlx-")
                else 1
            ),
            max_context_tokens=config.context_tokens,
            verify_tokens=1,
            output_mode=OutputMode.NEXT_TOKEN_ARGMAX,
            numerical_contract=engine.numerical_contract,
            state_abi=model.state_abi,
            required_component_roles=tuple(
                sorted({component.role for component in model.components})
            ),
            workspace_bytes=serving_workspace_bytes,
            headroom_bytes=config.headroom_bytes,
            fallback_policy=FallbackPolicy.DENY,
        )
        placement = backend.plan(
            model,
            workload,
            device,
            memory_budget_bytes=config.memory_budget_bytes,
        )
        runtime = backend.open(model, placement)
        if config.cuda_compatible_batch_size > 1:
            compatible_batch_lane = DenseCudaCompatibleBatchLane(
                runtime,
                max_batch_size=config.cuda_compatible_batch_size,
                max_slots=config.max_active_requests,
                max_queue_delay_seconds=config.cuda_batch_queue_delay_seconds,
                max_scratch_bytes=config.cuda_batch_scratch_bytes,
            )
        if config.mlx_compatible_batch_size > 1:
            max_scratch_bytes = config.mlx_batch_scratch_bytes or (
                model.state_bytes_per_token
                * config.context_tokens
                * config.mlx_compatible_batch_size
            )
            compatible_batch_lane = MlxCompatibleBatchLane(
                runtime,
                max_batch_size=config.mlx_compatible_batch_size,
                max_queue_delay_seconds=config.mlx_batch_queue_delay_seconds,
                max_scratch_bytes=max_scratch_bytes,
            )
        external_chat_template = (
            None
            if config.chat_template_file is None
            else _read_external_chat_template(config.chat_template_file)
        )
        if external_chat_template is not None:
            built_in_template = str(getattr(engine.tokenizer, "chat_template", "") or "")
            if built_in_template:
                raise ValueError(
                    "chat_template_file cannot override a source-bound tokenizer chat template"
                )
            observed_template_sha256 = canonical_chat_template_sha256(external_chat_template)
            if (
                config.chat_template_sha256 is not None
                and observed_template_sha256 != config.chat_template_sha256
            ):
                raise ValueError("external chat template content differs from chat_template_sha256")
        tokenizer = BoundChatTokenizer(
            engine.tokenizer,
            model_id=engine.name,
            semantic_token_count=engine.semantic_token_count,
            context_size=config.context_tokens,
            expected_legacy_raw_chat_template_sha256=(
                None if external_chat_template is not None else _legacy_template_sha256(engine)
            ),
            expected_chat_template_sha256=(
                config.chat_template_sha256 if external_chat_template is not None else None
            ),
            template_content=external_chat_template,
        )
        if config.sessions_enabled:
            route = runtime.route
            identity = SessionIdentity(
                model_id=tokenizer.model_id,
                model_fingerprint=model.fingerprint,
                chat_template_sha256=tokenizer.chat_template_sha256,
                semantic_token_count=tokenizer.semantic_token_count,
                route_id=route.runtime_id,
                runtime_id=route.runtime_id,
                capability_fingerprint=route.capability_fingerprint,
                placement_fingerprint=route.placement_fingerprint,
                backend_id=route.backend_id,
                device_id=route.device_id,
                state_abi=model.state_abi,
                execution_shape_fingerprint=route.execution_shape_fingerprint,
            )
            session_store = NativeSessionStore(
                runtime,
                identity=identity,
                state_bytes_per_token=model.state_bytes_per_token,
                state_fixed_bytes_per_row=model.state_fixed_bytes_per_row,
                ttl_seconds=config.session_ttl_seconds,
                max_entries=config.session_max_entries,
                max_bytes=config.session_cache_bytes,
            )
        service = NativeGenerationService(
            runtime,
            max_context_tokens=config.context_tokens,
            semantic_token_count=engine.semantic_token_count,
            max_active_requests=config.max_active_requests,
            supported_output_modes=capabilities.output_modes,
            max_new_tokens=config.max_new_tokens,
            event_queue_capacity=config.event_queue_capacity,
            owns_runtime=False,
            session_store=session_store,
            compatible_batch_lane=compatible_batch_lane,
        )
        host = InferenceHost(
            tokenizer=tokenizer,
            generation_service=service,
            route_id=runtime.route.runtime_id,
            promotion_status=capabilities.promotion_status,
            config=InferenceAppConfig(
                max_request_bytes=config.max_request_bytes,
                max_http_requests=config.max_http_requests,
                default_max_tokens=config.default_max_tokens,
                request_timeout_seconds=config.request_timeout_seconds,
                expose_metrics=config.expose_metrics,
            ),
        )
        app = create_inference_app(host, bearer_token=bearer_token)
        return LoadedNativeInference(
            config=config,
            engine=engine,
            device=device,
            backend=backend,
            model=model,
            workload=workload,
            placement=placement,
            runtime=runtime,
            tokenizer=tokenizer,
            session_store=session_store,
            compatible_batch_lane=compatible_batch_lane,
            service=service,
            host=host,
            app=app,
        )
    except BaseException:
        if service is not None:
            try:
                service.shutdown(ShutdownMode.ABORT, wait=True)
            except BaseException:
                pass
        if session_store is not None:
            try:
                session_store.close()
            except BaseException:
                pass
        if runtime is not None:
            try:
                runtime.close()
            except BaseException:
                pass
        elif engine is not None:
            try:
                engine.close()
            except BaseException:
                pass
        raise


__all__ = [
    "NATIVE_INFERENCE_BACKENDS",
    "LoadedMixtralExpertPagedSlice",
    "LoadedNativeInference",
    "MixtralExpertPagedSliceConfig",
    "NativeInferenceConfig",
    "load_mixtral_expert_paged_slice",
    "load_native_inference",
]
