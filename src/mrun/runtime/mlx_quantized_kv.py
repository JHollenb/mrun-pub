"""Fixed, transactional affine low-bit K/V storage for MLX global attention.

``mlx-lm`` exposes a dynamically growing :class:`QuantizedKVCache`, but the native mrun
service admits complete request capacities before execution and requires fixed backing identity
for provisional commit, rollback, and cross-session forks.  This module provides the physical
cache primitive for that contract.  It deliberately does not select a policy or mutate model
identity; the native backend binding owns those decisions.

The packed payload, scales, and biases stay as MLX arrays in unified memory.  Prefix copies are
device-to-device and preserve the exact packed representation.  Returning the same three-array
tuples as ``mlx-lm`` makes its quantized scaled-dot-product attention consume the cache directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MLX_AFFINE_KV_CACHE_ABI = "mrun-mlx-fixed-affine-kv-v1"
_SUPPORTED_BITS = frozenset({2, 4, 8})
_SUPPORTED_GROUP_SIZES = frozenset({32, 64, 128})


def _strict_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _dtype_name(value: Any) -> str:
    return str(value).lower().rsplit(".", maxsplit=1)[-1]


def _array_signature(value: Any) -> tuple[Any, ...]:
    return (
        id(value),
        tuple(int(dimension) for dimension in value.shape),
        str(value.dtype),
    )


@dataclass(frozen=True, slots=True)
class MlxAffineKVCodec:
    """Physical identity and exact byte formula for one affine K/V codec."""

    bits: int = 4
    group_size: int = 64
    mode: str = "affine"
    cache_abi: str = MLX_AFFINE_KV_CACHE_ABI

    def __post_init__(self) -> None:
        bits = _strict_positive_int(self.bits, "KV bits")
        group_size = _strict_positive_int(self.group_size, "KV group_size")
        if bits not in _SUPPORTED_BITS or 32 % bits != 0:
            # MLX packs an integer number of values into each uint32.  Although
            # mx.quantize supports 3/5/6-bit weights, their packed last dimension is
            # not a simple fixed-arena division and is intentionally not admitted here.
            raise ValueError("fixed affine KV bits must divide 32 (2, 4, or 8)")
        if group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError("fixed affine KV group_size must be 32, 64, or 128")
        if self.mode != "affine" or self.cache_abi != MLX_AFFINE_KV_CACHE_ABI:
            raise ValueError("unsupported fixed MLX K/V codec identity")

    @property
    def values_per_word(self) -> int:
        return 32 // self.bits

    @property
    def codec_id(self) -> str:
        return f"mlx-affine-kv{self.bits}-g{self.group_size}"

    def state_abi(self, source_dtype: str) -> str:
        normalized = source_dtype.lower().rsplit(".", maxsplit=1)[-1]
        if normalized not in {"bfloat16", "float16", "float32"}:
            raise ValueError("fixed affine K/V source dtype is unsupported")
        return f"mrun-transactional-gqa-{self.codec_id}-{normalized}-global-attention-v1"

    def vector_bytes(self, width: int, *, source_element_bytes: int) -> int:
        width = _strict_positive_int(width, "K/V vector width")
        source_element_bytes = _strict_positive_int(
            source_element_bytes,
            "K/V source element bytes",
        )
        if width % self.values_per_word:
            raise ValueError("K/V vector width is not packable into uint32 words")
        if width % self.group_size:
            raise ValueError("K/V vector width is not divisible by group_size")
        payload = width // self.values_per_word * 4
        affine_metadata = 2 * (width // self.group_size) * source_element_bytes
        return payload + affine_metadata


class FixedMlxQuantizedKVCache:
    """``mlx-lm`` compatible fixed affine K/V cache with transactional primitives."""

    __slots__ = (
        "_capacity",
        "_create_attention_mask",
        "_key_dtype",
        "_key_head_dim",
        "_kv_heads",
        "_mx",
        "_value_dtype",
        "_value_head_dim",
        "bits",
        "group_size",
        "keys",
        "offset",
        "values",
    )

    def __init__(
        self,
        *,
        mx: Any,
        create_attention_mask: Any,
        capacity: int,
        kv_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        key_dtype: Any,
        value_dtype: Any,
        codec: MlxAffineKVCodec | None = None,
    ) -> None:
        codec = codec or MlxAffineKVCodec()
        self._capacity = _strict_positive_int(capacity, "K/V capacity")
        self._kv_heads = _strict_positive_int(kv_heads, "K/V head count")
        self._key_head_dim = _strict_positive_int(key_head_dim, "key head width")
        self._value_head_dim = _strict_positive_int(value_head_dim, "value head width")
        # Validate both physical vectors before allocating any unified memory.
        key_element_bytes = int(getattr(key_dtype, "size", 0))
        value_element_bytes = int(getattr(value_dtype, "size", 0))
        codec.vector_bytes(self._key_head_dim, source_element_bytes=key_element_bytes)
        codec.vector_bytes(self._value_head_dim, source_element_bytes=value_element_bytes)
        self._mx = mx
        self._create_attention_mask = create_attention_mask
        self._key_dtype = key_dtype
        self._value_dtype = value_dtype
        self.bits = codec.bits
        self.group_size = codec.group_size
        self.keys = self._allocate(self._key_head_dim, key_dtype)
        self.values = self._allocate(self._value_head_dim, value_dtype)
        self.offset = 0

    def _allocate(self, width: int, dtype: Any) -> tuple[Any, Any, Any]:
        prefix = (1, self._kv_heads, self._capacity)
        return (
            self._mx.zeros((*prefix, width // (32 // self.bits)), dtype=self._mx.uint32),
            self._mx.zeros((*prefix, width // self.group_size), dtype=dtype),
            self._mx.zeros((*prefix, width // self.group_size), dtype=dtype),
        )

    @staticmethod
    def _slice(parts: tuple[Any, Any, Any], stop: int) -> tuple[Any, Any, Any]:
        return tuple(part[..., :stop, :] for part in parts)  # type: ignore[return-value]

    def _validate_source(self, keys: Any, values: Any) -> int:
        if int(getattr(keys, "ndim", -1)) != 4 or int(getattr(values, "ndim", -1)) != 4:
            raise ValueError("model K/V output must be rank-four MLX arrays")
        token_count = int(keys.shape[2])
        if token_count <= 0 or self.offset + token_count > self._capacity:
            raise OverflowError("native MLX quantized K/V append exceeds the fixed arena")
        if (
            int(keys.shape[0]) != 1
            or int(values.shape[0]) != 1
            or int(keys.shape[1]) != self._kv_heads
            or int(values.shape[1]) != self._kv_heads
            or int(keys.shape[3]) != self._key_head_dim
            or int(values.shape[3]) != self._value_head_dim
            or int(values.shape[2]) != token_count
            or keys.dtype != self._key_dtype
            or values.dtype != self._value_dtype
        ):
            raise ValueError("model K/V output differs from the quantized arena geometry")
        return token_count

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        token_count = self._validate_source(keys, values)
        start = self.offset
        stop = start + token_count
        quantized_keys = self._mx.quantize(
            keys,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )
        quantized_values = self._mx.quantize(
            values,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )
        if len(quantized_keys) != 3 or len(quantized_values) != 3:
            raise RuntimeError("MLX affine K/V quantizer returned an invalid payload")
        for target, source in zip(self.keys, quantized_keys, strict=True):
            target[..., start:stop, :] = source
        for target, source in zip(self.values, quantized_values, strict=True):
            target[..., start:stop, :] = source
        self.offset = stop
        return self._slice(self.keys, stop), self._slice(self.values, stop)

    def make_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._create_attention_mask(*args, offset=self.offset, **kwargs)

    def trim(self, count: int) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("trim count must be a non-negative integer")
        trimmed = min(self.offset, count)
        self.offset -= trimmed
        return trimmed

    def reset(self, offset: int) -> None:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("reset offset must be an integer")
        if offset < 0 or offset > self._capacity:
            raise ValueError("reset offset lies outside the fixed quantized arena")
        self.offset = int(offset)

    def copy_committed_prefix_from(
        self,
        source: FixedMlxQuantizedKVCache,
        length: int,
    ) -> None:
        if not isinstance(source, FixedMlxQuantizedKVCache):
            raise TypeError("quantized K/V fork source uses another cache ABI")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("quantized K/V fork length must be non-negative")
        if self.offset != 0:
            raise RuntimeError("quantized K/V fork target must be empty")
        if self._mx is not source._mx:
            raise RuntimeError("quantized K/V fork crosses an execution context")
        if length > source.offset or length > self._capacity:
            raise OverflowError("quantized K/V fork exceeds source or target capacity")
        if (
            self.bits != source.bits
            or self.group_size != source.group_size
            or self._kv_heads != source._kv_heads
            or self._key_head_dim != source._key_head_dim
            or self._value_head_dim != source._value_head_dim
            or self._key_dtype != source._key_dtype
            or self._value_dtype != source._value_dtype
        ):
            raise RuntimeError("quantized K/V fork crosses a physical codec boundary")
        if {id(value) for value in self.storage_arrays()}.intersection(
            id(value) for value in source.storage_arrays()
        ):
            raise RuntimeError("quantized K/V fork target aliases source storage")
        if length:
            for target, value in zip(self.keys, source.keys, strict=True):
                target[..., :length, :] = value[..., :length, :]
            for target, value in zip(self.values, source.values, strict=True):
                target[..., :length, :] = value[..., :length, :]
            self._mx.eval(*self.storage_arrays())
        self.offset = length

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def state(self) -> tuple[Any, Any]:
        return self._slice(self.keys, self.offset), self._slice(self.values, self.offset)

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.storage_arrays())

    def storage_arrays(self) -> tuple[Any, ...]:
        return (*self.keys, *self.values)

    def storage_signature(self) -> tuple[Any, ...]:
        return (
            MLX_AFFINE_KV_CACHE_ABI,
            self.bits,
            self.group_size,
            self._capacity,
            self._kv_heads,
            self._key_head_dim,
            self._value_head_dim,
            _dtype_name(self._key_dtype),
            _dtype_name(self._value_dtype),
            tuple(_array_signature(value) for value in self.storage_arrays()),
        )


__all__ = [
    "MLX_AFFINE_KV_CACHE_ABI",
    "FixedMlxQuantizedKVCache",
    "MlxAffineKVCodec",
]
