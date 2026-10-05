"""Explicit Metal block-paged K=1 decode attention for Qwen2 and Mixtral.

The lane is intentionally narrow.  It accepts only B=1, K=1, BF16 global grouped-query
attention whose K and V head widths match and are multiples of one Metal SIMD group.  Prefill
continues through mlx-lm's ordinary dense attention.  Eligible decode calls append directly to
the authoritative page table and read the physical slabs through a generation-validated logical
page-slot vector; no dense K/V sequence is constructed.

Installing the lane changes execution arithmetic (online softmax with Metal fast exponential),
so it has an explicit experimental numerical identity.  Installation dynamically subclasses the
already-loaded attention objects rather than replacing them: projection/RoPE/output modules and
all parameter objects retain their exact identities and values.  ``close()`` restores the
original classes before the model owner is closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from .mlx_paged_kv import FixedMlxPagedKVCache, MlxPagedAttentionInput

MLX_PAGED_DECODE_ATTENTION_ABI = "mrun-mlx-metal-block-paged-decode-attention-v1"
MLX_PAGED_DECODE_NUMERICAL_CONTRACT = (
    "mlx-metal-block-paged-k1-online-softmax-fast-exp-bf16-bounded-v1"
)

_SUPPORTED_ARCHITECTURES = frozenset({"qwen2", "mixtral"})
_BINDING_ATTRIBUTE = "_mrun_paged_decode_attention_binding"

_METAL_SOURCE = r"""
    constexpr int VECTORS_PER_LANE = HEAD_DIM / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint query_head = threadgroup_position_in_grid.y;
    const uint kv_head = query_head / QUERY_HEADS_PER_KV_HEAD;
    threadgroup float partial_maxima[SIMD_GROUPS];
    threadgroup float partial_sums[SIMD_GROUPS];
    threadgroup float partial_outputs[SIMD_GROUPS * HEAD_DIM];

    device const T* query_row = queries + query_head * HEAD_DIM;
    device T* output_row = output + query_head * HEAD_DIM;
    float accum[VECTORS_PER_LANE];
    for (int vector = 0; vector < VECTORS_PER_LANE; ++vector) {
        accum[vector] = 0.0f;
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const int token_count = lengths[0];

    for (int token = simd_group; token < token_count; token += SIMD_GROUPS) {
        const int logical_page = token / PAGE_SIZE;
        const int in_page = token - logical_page * PAGE_SIZE;
        const int physical_page = page_slots[logical_page];
        const int row_base =
            ((physical_page * KV_HEADS + kv_head) * PAGE_SIZE + in_page) * HEAD_DIM;
        float partial = 0.0f;
        for (int vector = 0; vector < VECTORS_PER_LANE; ++vector) {
            const int dimension = lane + vector * 32;
            partial += static_cast<float>(query_row[dimension])
                * static_cast<float>(keys[row_base + dimension]);
        }
        const float score = simd_sum(partial) * scales[0];
        const float next_max = max(running_max, score);
        const float old_weight = metal::fast::exp(running_max - next_max);
        const float new_weight = metal::fast::exp(score - next_max);
        for (int vector = 0; vector < VECTORS_PER_LANE; ++vector) {
            const int dimension = lane + vector * 32;
            accum[vector] = accum[vector] * old_weight
                + static_cast<float>(values[row_base + dimension]) * new_weight;
        }
        running_sum = running_sum * old_weight + new_weight;
        running_max = next_max;
    }

    if (lane == 0) {
        partial_maxima[simd_group] = running_max;
        partial_sums[simd_group] = running_sum;
    }
    for (int vector = 0; vector < VECTORS_PER_LANE; ++vector) {
        const int dimension = lane + vector * 32;
        partial_outputs[simd_group * HEAD_DIM + dimension] = accum[vector];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float global_max = -INFINITY;
        for (int group = 0; group < SIMD_GROUPS; ++group) {
            global_max = max(global_max, partial_maxima[group]);
        }
        float global_sum = 0.0f;
        for (int group = 0; group < SIMD_GROUPS; ++group) {
            global_sum += partial_sums[group]
                * metal::fast::exp(partial_maxima[group] - global_max);
        }
        for (int vector = 0; vector < VECTORS_PER_LANE; ++vector) {
            const int dimension = lane + vector * 32;
            float combined = 0.0f;
            for (int group = 0; group < SIMD_GROUPS; ++group) {
                combined += partial_outputs[group * HEAD_DIM + dimension]
                    * metal::fast::exp(partial_maxima[group] - global_max);
            }
            output_row[dimension] = static_cast<T>(combined / global_sum);
        }
    }
"""


class MlxPagedDecodeAttentionError(RuntimeError):
    """The requested model/cache/call lies outside the explicit Metal lane."""


def _strict_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MlxPagedDecodeAttentionError(f"{field_name} must be a positive integer")
    return int(value)


def _dtype_name(value: Any) -> str:
    return str(value).lower().rsplit(".", maxsplit=1)[-1]


def _is_bfloat16(value: Any) -> bool:
    return _dtype_name(value) in {"bf16", "bfloat16"}


def _projection_width(projection: Any, field_name: str) -> int:
    output_dims = getattr(projection, "output_dims", None)
    if output_dims is None:
        weight = getattr(projection, "weight", None)
        if int(getattr(weight, "ndim", -1)) != 2:
            raise MlxPagedDecodeAttentionError(f"{field_name} has no rank-two weight")
        output_dims = int(weight.shape[0])
    return _strict_positive_int(output_dims, field_name)


def _projection_input_width(projection: Any, field_name: str) -> int:
    input_dims = getattr(projection, "input_dims", None)
    if input_dims is None:
        weight = getattr(projection, "weight", None)
        if int(getattr(weight, "ndim", -1)) != 2:
            raise MlxPagedDecodeAttentionError(f"{field_name} has no rank-two weight")
        bits = getattr(projection, "bits", None)
        if bits is None:
            input_dims = int(weight.shape[1])
        else:
            bits = _strict_positive_int(bits, f"{field_name} quantization bits")
            packed_bits = int(weight.shape[1]) * 32
            if bits > 8 or packed_bits % bits:
                raise MlxPagedDecodeAttentionError(
                    f"{field_name} has unsupported packed quantization geometry"
                )
            input_dims = packed_bits // bits
    return _strict_positive_int(input_dims, field_name)


def _attention_geometry(attention: Any, architecture: str) -> tuple[int, int, int]:
    if architecture == "qwen2":
        query_heads = _strict_positive_int(getattr(attention, "n_heads", None), "query heads")
        kv_heads = _strict_positive_int(getattr(attention, "n_kv_heads", None), "K/V heads")
        query_width = _projection_width(getattr(attention, "q_proj", None), "Q projection width")
        if query_width % query_heads:
            raise MlxPagedDecodeAttentionError(
                "Q projection width must be divisible by the query-head count"
            )
        head_dim = query_width // query_heads
    elif architecture == "mixtral":
        query_heads = _strict_positive_int(
            getattr(attention, "num_heads", None),
            "query heads",
        )
        kv_heads = _strict_positive_int(
            getattr(attention, "num_key_value_heads", None),
            "K/V heads",
        )
        head_dim = _strict_positive_int(getattr(attention, "head_dim", None), "head dimension")
    else:
        raise MlxPagedDecodeAttentionError(
            f"Metal paged decode does not support architecture {architecture!r}"
        )
    if query_heads % kv_heads:
        raise MlxPagedDecodeAttentionError("query heads must be divisible by K/V heads")
    if head_dim % 32 or not 32 <= head_dim <= 256:
        raise MlxPagedDecodeAttentionError(
            "head dimension must be a multiple of 32 in the closed interval [32, 256]"
        )
    expected_widths = {
        "Q": query_heads * head_dim,
        "K": kv_heads * head_dim,
        "V": kv_heads * head_dim,
    }
    for label, name in (("Q", "q_proj"), ("K", "k_proj"), ("V", "v_proj")):
        if (
            _projection_width(getattr(attention, name, None), f"{label} projection width")
            != (expected_widths[label])
        ):
            raise MlxPagedDecodeAttentionError(
                f"{label} projection width differs from the declared attention geometry"
            )
    if _projection_input_width(
        getattr(attention, "o_proj", None),
        "output projection input width",
    ) != (query_heads * head_dim):
        raise MlxPagedDecodeAttentionError(
            "output projection width differs from the declared attention geometry"
        )
    return query_heads, kv_heads, head_dim


@dataclass(frozen=True, slots=True)
class MlxPagedDecodeAttentionIdentity:
    architecture: str
    layer_count: int
    layer_geometries: tuple[tuple[int, int, int], ...]
    page_size: int
    page_count: int
    simd_groups: int
    base_numerical_contract: str
    numerical_contract: str = MLX_PAGED_DECODE_NUMERICAL_CONTRACT
    execution_abi: str = MLX_PAGED_DECODE_ATTENTION_ABI
    promotion_status: str = "experimental"

    def __post_init__(self) -> None:
        if self.execution_abi != MLX_PAGED_DECODE_ATTENTION_ABI:
            raise ValueError("unsupported Metal paged-decode execution ABI")
        if self.architecture not in _SUPPORTED_ARCHITECTURES:
            raise ValueError("unsupported Metal paged-decode architecture")
        for field_name in ("layer_count", "page_size", "page_count", "simd_groups"):
            _strict_positive_int(getattr(self, field_name), field_name)
        if self.page_size & (self.page_size - 1):
            raise ValueError("Metal paged-decode page size must be a power of two")
        if self.simd_groups > 32:
            raise ValueError("Metal paged-decode supports at most 32 SIMD groups")
        if self.promotion_status != "experimental":
            raise ValueError("Metal paged-decode is not a promoted execution lane")
        if len(self.layer_geometries) != self.layer_count:
            raise ValueError("Metal paged-decode identity has the wrong layer-geometry count")
        for query_heads, kv_heads, head_dim in self.layer_geometries:
            for value in (query_heads, kv_heads, head_dim):
                _strict_positive_int(value, "layer geometry")
            if query_heads % kv_heads or head_dim % 32 or not 32 <= head_dim <= 256:
                raise ValueError("Metal paged-decode identity has unsupported layer geometry")
        for field_name in ("base_numerical_contract", "numerical_contract"):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a canonical non-empty string")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "architecture": self.architecture,
                "layer_count": self.layer_count,
                "layer_geometries": self.layer_geometries,
                "page_size": self.page_size,
                "page_count": self.page_count,
                "simd_groups": self.simd_groups,
                "base_numerical_contract": self.base_numerical_contract,
                "numerical_contract": self.numerical_contract,
                "execution_abi": self.execution_abi,
                "promotion_status": self.promotion_status,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MlxPagedDecodeAttentionTelemetry:
    identity_fingerprint: str
    installed_layers: int
    paged_decode_calls: int
    attended_tokens: int
    logical_pages_read: int
    page_table_entries_uploaded: int
    dense_materializations_avoided: int
    prefill_dense_calls: int
    uncached_dense_calls: int
    qwen2_calls: int
    mixtral_calls: int
    failures: int
    max_attended_tokens: int
    closed: bool


class MlxMetalPagedDecodeAttention:
    """Validated low-level dispatch of the Metal kernel from the frozen prototype."""

    def __init__(self, mx: Any, *, simd_groups: int = 8) -> None:
        self._mx = mx
        self.simd_groups = _strict_positive_int(simd_groups, "simd_groups")
        if self.simd_groups > 32 or self.simd_groups * 32 > 1024:
            raise MlxPagedDecodeAttentionError("Metal threadgroup exceeds 1024 threads")
        fast = getattr(mx, "fast", None)
        metal_kernel = None if fast is None else getattr(fast, "metal_kernel", None)
        if not callable(metal_kernel):
            raise MlxPagedDecodeAttentionError("MLX execution context has no Metal kernel API")
        self._kernel = metal_kernel(
            name="mrun_block_paged_decode_attention_v1",
            input_names=["queries", "keys", "values", "page_slots", "lengths", "scales"],
            output_names=["output"],
            source=_METAL_SOURCE,
            ensure_row_contiguous=True,
        )

    def __call__(
        self,
        queries: Any,
        view: MlxPagedAttentionInput,
        *,
        scale: float,
        query_heads: int,
    ) -> Any:
        if not isinstance(view, MlxPagedAttentionInput):
            raise TypeError("paged decode requires MlxPagedAttentionInput")
        query_heads = _strict_positive_int(query_heads, "query_heads")
        if int(getattr(queries, "ndim", -1)) != 4 or tuple(
            int(value) for value in queries.shape[:3]
        ) != (1, query_heads, 1):
            raise MlxPagedDecodeAttentionError("paged decode queries must have shape [1,H,1,D]")
        head_dim = int(queries.shape[3])
        if head_dim != view.key_head_dim or head_dim != view.value_head_dim:
            raise MlxPagedDecodeAttentionError("query, key, and value head dimensions must match")
        if head_dim % 32 or not 32 <= head_dim <= 256:
            raise MlxPagedDecodeAttentionError("unsupported Metal paged-decode head dimension")
        if query_heads % view.kv_heads:
            raise MlxPagedDecodeAttentionError("query heads must be divisible by K/V heads")
        if not (_is_bfloat16(queries.dtype) and _is_bfloat16(view.dtype)):
            raise MlxPagedDecodeAttentionError("Metal paged decode requires BF16 Q/K/V")
        if not math.isfinite(scale) or scale <= 0:
            raise MlxPagedDecodeAttentionError("attention scale must be finite and positive")
        if view.length <= 0 or view.logical_page_count <= 0:
            raise MlxPagedDecodeAttentionError("paged decode requires a non-empty K/V sequence")
        if self.simd_groups * (head_dim + 2) * 4 > 32 * 1024:
            raise MlxPagedDecodeAttentionError("Metal threadgroup memory exceeds 32 KiB")
        lengths = self._mx.array([view.length], dtype=self._mx.int32)
        scales = self._mx.array([float(scale)], dtype=self._mx.float32)
        return self._kernel(
            inputs=[queries, view.keys, view.values, view.page_slots, lengths, scales],
            output_shapes=[queries.shape],
            output_dtypes=[queries.dtype],
            template=[
                ("T", queries.dtype),
                ("HEAD_DIM", head_dim),
                ("KV_HEADS", view.kv_heads),
                ("PAGE_SIZE", view.page_size),
                ("QUERY_HEADS_PER_KV_HEAD", query_heads // view.kv_heads),
                ("SIMD_GROUPS", self.simd_groups),
            ],
            grid=(32 * self.simd_groups, query_heads, 1),
            threadgroup=(32 * self.simd_groups, 1, 1),
        )[0]


@dataclass(frozen=True, slots=True)
class _LayerBinding:
    lane: MlxPagedDecodeAttentionLane
    layer_index: int
    architecture: str
    original_class: type[Any]
    original_call: Any
    query_heads: int
    kv_heads: int
    head_dim: int

    def __call__(self, attention: Any, x: Any, mask: Any, cache: Any) -> Any:
        return self.lane._attention_call(  # noqa: SLF001 - binding and lane are one owner
            self,
            attention,
            x,
            mask,
            cache,
        )


class _PagedDecodeAttentionMixin:
    def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
        binding = object.__getattribute__(self, _BINDING_ATTRIBUTE)
        return binding(self, x, mask, cache)


class MlxPagedDecodeAttentionLane:
    """Install and own an explicit paged K=1 attention execution identity."""

    def __init__(
        self,
        model: Any,
        *,
        mx: Any,
        architecture: str,
        state_layout: Any,
        page_size: int,
        page_count: int,
        base_numerical_contract: str,
        simd_groups: int = 8,
    ) -> None:
        if architecture not in _SUPPORTED_ARCHITECTURES:
            raise MlxPagedDecodeAttentionError(
                "Metal paged decode supports only Qwen2 and classic Mixtral"
            )
        layers = tuple(getattr(model, "layers", ()))
        if not layers:
            body = getattr(model, "model", None)
            layers = tuple(getattr(body, "layers", ()))
        specs = tuple(getattr(state_layout, "layers", ()))
        if not layers or len(layers) != len(specs):
            raise MlxPagedDecodeAttentionError(
                "model decoder layers differ from the measured K/V state layout"
            )
        if str(getattr(state_layout, "dtype_name", "")).lower() not in {"bf16", "bfloat16"}:
            raise MlxPagedDecodeAttentionError("Metal paged decode requires BF16 K/V state")
        page_size = _strict_positive_int(page_size, "page_size")
        page_count = _strict_positive_int(page_count, "page_count")
        if page_size & (page_size - 1):
            raise MlxPagedDecodeAttentionError(
                "Metal paged-decode page size must be a power of two"
            )
        prepared: list[tuple[Any, tuple[int, int, int]]] = []
        for layer_index, (layer, spec) in enumerate(zip(layers, specs, strict=True)):
            attention = getattr(layer, "self_attn", None)
            if attention is None or hasattr(attention, _BINDING_ATTRIBUTE):
                raise MlxPagedDecodeAttentionError(
                    "decoder layer lacks an unwrapped self-attention module"
                )
            geometry = _attention_geometry(attention, architecture)
            _query_heads, kv_heads, head_dim = geometry
            if (
                int(getattr(spec, "kv_heads", -1)) != kv_heads
                or int(getattr(spec, "key_head_dim", -1)) != head_dim
                or int(getattr(spec, "value_head_dim", -1)) != head_dim
                or not _is_bfloat16(getattr(spec, "key_dtype", None))
                or not _is_bfloat16(getattr(spec, "value_dtype", None))
            ):
                raise MlxPagedDecodeAttentionError(
                    f"layer {layer_index} attention differs from measured BF16 K/V geometry"
                )
            prepared.append((attention, geometry))
        self._mx = mx
        self._kernel = MlxMetalPagedDecodeAttention(mx, simd_groups=simd_groups)
        self._identity = MlxPagedDecodeAttentionIdentity(
            architecture=architecture,
            layer_count=len(layers),
            layer_geometries=tuple(geometry for _attention, geometry in prepared),
            page_size=page_size,
            page_count=page_count,
            simd_groups=simd_groups,
            base_numerical_contract=base_numerical_contract,
        )
        self._lock = threading.RLock()
        self._installed: list[tuple[Any, type[Any]]] = []
        self._closed = False
        self._paged_decode_calls = 0
        self._attended_tokens = 0
        self._logical_pages_read = 0
        self._page_table_entries_uploaded = 0
        self._dense_materializations_avoided = 0
        self._prefill_dense_calls = 0
        self._uncached_dense_calls = 0
        self._qwen2_calls = 0
        self._mixtral_calls = 0
        self._failures = 0
        self._max_attended_tokens = 0
        try:
            for layer_index, (attention, geometry) in enumerate(prepared):
                query_heads, kv_heads, head_dim = geometry
                original_class = type(attention)
                binding = _LayerBinding(
                    lane=self,
                    layer_index=layer_index,
                    architecture=architecture,
                    original_class=original_class,
                    original_call=original_class.__call__,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                )
                patched_class = type(
                    f"MrunPagedDecode{original_class.__name__}Layer{layer_index}",
                    (_PagedDecodeAttentionMixin, original_class),
                    {"__module__": __name__},
                )
                object.__setattr__(attention, _BINDING_ATTRIBUTE, binding)
                attention.__class__ = patched_class
                self._installed.append((attention, original_class))
        except BaseException:
            self.close()
            raise

    @property
    def identity(self) -> MlxPagedDecodeAttentionIdentity:
        return self._identity

    def _record_dense(self, *, uncached: bool) -> None:
        with self._lock:
            if uncached:
                self._uncached_dense_calls += 1
            else:
                self._prefill_dense_calls += 1

    def _attention_call(
        self,
        binding: _LayerBinding,
        attention: Any,
        x: Any,
        mask: Any,
        cache: Any,
    ) -> Any:
        if cache is None:
            self._record_dense(uncached=True)
            return binding.original_call(attention, x, mask, cache)
        if int(getattr(x, "ndim", -1)) != 3:
            raise MlxPagedDecodeAttentionError("attention input must have shape [B,L,D]")
        batch, length, hidden = (int(value) for value in x.shape)
        if length != 1:
            self._record_dense(uncached=False)
            return binding.original_call(attention, x, mask, cache)
        if batch != 1:
            raise MlxPagedDecodeAttentionError("Metal paged decode is strictly B=1")
        if mask is not None:
            raise MlxPagedDecodeAttentionError("K=1 global decode must not carry a dense mask")
        if not isinstance(cache, FixedMlxPagedKVCache):
            raise MlxPagedDecodeAttentionError(
                "Metal paged decode requires the authoritative BF16 page cache"
            )
        if not cache.paged_decode_attention_enabled:
            raise MlxPagedDecodeAttentionError("paged cache was not admitted for Metal decode")
        if hidden != binding.query_heads * binding.head_dim:
            raise MlxPagedDecodeAttentionError("attention hidden width differs from head geometry")
        try:
            queries = attention.q_proj(x)
            keys = attention.k_proj(x)
            values = attention.v_proj(x)
            queries = queries.reshape(1, 1, binding.query_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(1, 1, binding.kv_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(1, 1, binding.kv_heads, -1).transpose(0, 2, 1, 3)
            prior_offset = cache.offset
            queries = attention.rope(queries, offset=prior_offset)
            keys = attention.rope(keys, offset=prior_offset)
            view = cache.update_for_paged_decode(keys, values)
            output = self._kernel(
                queries,
                view,
                scale=float(attention.scale),
                query_heads=binding.query_heads,
            )
            output = output.transpose(0, 2, 1, 3).reshape(1, 1, -1)
            result = attention.o_proj(output)
        except BaseException:
            with self._lock:
                self._failures += 1
            raise
        with self._lock:
            if self._closed:
                raise MlxPagedDecodeAttentionError("Metal paged-decode lane is closed")
            self._paged_decode_calls += 1
            self._attended_tokens += view.length
            self._logical_pages_read += view.logical_page_count
            self._page_table_entries_uploaded += view.logical_page_count
            self._dense_materializations_avoided += 1
            self._max_attended_tokens = max(self._max_attended_tokens, view.length)
            if binding.architecture == "qwen2":
                self._qwen2_calls += 1
            else:
                self._mixtral_calls += 1
        return result

    def telemetry(self) -> MlxPagedDecodeAttentionTelemetry:
        with self._lock:
            return MlxPagedDecodeAttentionTelemetry(
                identity_fingerprint=self._identity.fingerprint,
                installed_layers=len(self._installed),
                paged_decode_calls=self._paged_decode_calls,
                attended_tokens=self._attended_tokens,
                logical_pages_read=self._logical_pages_read,
                page_table_entries_uploaded=self._page_table_entries_uploaded,
                dense_materializations_avoided=self._dense_materializations_avoided,
                prefill_dense_calls=self._prefill_dense_calls,
                uncached_dense_calls=self._uncached_dense_calls,
                qwen2_calls=self._qwen2_calls,
                mixtral_calls=self._mixtral_calls,
                failures=self._failures,
                max_attended_tokens=self._max_attended_tokens,
                closed=self._closed,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for attention, original_class in reversed(self._installed):
                attention.__class__ = original_class
                if hasattr(attention, _BINDING_ATTRIBUTE):
                    object.__delattr__(attention, _BINDING_ATTRIBUTE)
            self._installed.clear()
            self._closed = True


@dataclass(frozen=True, slots=True)
class MlxPagedDecodeBenchmarkPoint:
    tokens: int
    page_size: int
    pages: int
    paged_median_ms: float
    dense_materialize_median_ms: float
    paged_over_dense: float
    max_abs_error: float
    mean_abs_error: float


def benchmark_mlx_paged_decode_attention(
    mx: Any,
    *,
    token_counts: tuple[int, ...] = (128, 1024, 8192, 32768),
    page_size: int = 64,
    query_heads: int = 14,
    kv_heads: int = 2,
    head_dim: int = 64,
    simd_groups: int = 8,
    trials: int = 7,
) -> tuple[MlxPagedDecodeBenchmarkPoint, ...]:
    """Measure the crossover against dense materialization plus MLX SDPA.

    This is a local microbenchmark, not a model token/s claim.  Alternating invocation order
    includes page concatenation in the dense control because that is the cost the lane removes.
    """

    counts = tuple(_strict_positive_int(value, "token_counts[]") for value in token_counts)
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("token_counts must be strictly increasing and unique")
    page_size = _strict_positive_int(page_size, "page_size")
    query_heads = _strict_positive_int(query_heads, "query_heads")
    kv_heads = _strict_positive_int(kv_heads, "kv_heads")
    head_dim = _strict_positive_int(head_dim, "head_dim")
    trials = _strict_positive_int(trials, "trials")
    if query_heads % kv_heads or head_dim % 32:
        raise ValueError("benchmark requires divisible heads and a SIMD-aligned head dimension")
    kernel = MlxMetalPagedDecodeAttention(mx, simd_groups=simd_groups)
    results: list[MlxPagedDecodeBenchmarkPoint] = []
    mx.random.seed(17)
    for tokens in counts:
        pages = (tokens + page_size - 1) // page_size
        queries = mx.random.normal((1, query_heads, 1, head_dim)).astype(mx.bfloat16)
        keys = mx.random.normal((pages, kv_heads, page_size, head_dim)).astype(mx.bfloat16)
        values = mx.random.normal((pages, kv_heads, page_size, head_dim)).astype(mx.bfloat16)
        slots = mx.arange(pages, dtype=mx.int32)
        view = MlxPagedAttentionInput(
            keys=keys,
            values=values,
            page_slots=slots,
            length=tokens,
            logical_page_count=pages,
            page_size=page_size,
            kv_heads=kv_heads,
            key_head_dim=head_dim,
            value_head_dim=head_dim,
            dtype=mx.bfloat16,
        )

        def paged(queries: Any = queries, view: Any = view) -> Any:
            return kernel(
                queries,
                view,
                scale=head_dim**-0.5,
                query_heads=query_heads,
            )

        def dense(
            queries: Any = queries,
            keys: Any = keys,
            values: Any = values,
            pages: int = pages,
            tokens: int = tokens,
        ) -> Any:
            dense_keys = mx.concatenate(
                tuple(keys[index : index + 1] for index in range(pages)),
                axis=2,
            )[:, :, :tokens, :]
            dense_values = mx.concatenate(
                tuple(values[index : index + 1] for index in range(pages)), axis=2
            )[:, :, :tokens, :]
            return mx.fast.scaled_dot_product_attention(
                queries,
                dense_keys,
                dense_values,
                scale=head_dim**-0.5,
            )

        candidate = paged()
        reference = dense()
        mx.eval(candidate, reference)
        difference = mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
        mx.eval(difference)
        paged_times: list[float] = []
        dense_times: list[float] = []
        for _index in range(3):
            mx.eval(paged(), dense())
        for trial in range(trials):
            order = ((paged, paged_times), (dense, dense_times))
            if trial % 2:
                order = tuple(reversed(order))
            for function, samples in order:
                started = time.perf_counter_ns()
                mx.eval(function())
                samples.append((time.perf_counter_ns() - started) / 1e6)
        paged_median = sorted(paged_times)[len(paged_times) // 2]
        dense_median = sorted(dense_times)[len(dense_times) // 2]
        results.append(
            MlxPagedDecodeBenchmarkPoint(
                tokens=tokens,
                page_size=page_size,
                pages=pages,
                paged_median_ms=paged_median,
                dense_materialize_median_ms=dense_median,
                paged_over_dense=paged_median / dense_median,
                max_abs_error=float(mx.max(difference).item()),
                mean_abs_error=float(mx.mean(difference).item()),
            )
        )
    return tuple(results)


__all__ = [
    "MLX_PAGED_DECODE_ATTENTION_ABI",
    "MLX_PAGED_DECODE_NUMERICAL_CONTRACT",
    "MlxMetalPagedDecodeAttention",
    "MlxPagedDecodeAttentionError",
    "MlxPagedDecodeAttentionIdentity",
    "MlxPagedDecodeAttentionLane",
    "MlxPagedDecodeAttentionTelemetry",
    "MlxPagedDecodeBenchmarkPoint",
    "benchmark_mlx_paged_decode_attention",
]
