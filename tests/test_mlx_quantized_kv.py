from __future__ import annotations

from typing import Any

import pytest

from mrun.runtime.mlx_quantized_kv import FixedMlxQuantizedKVCache, MlxAffineKVCodec


def _mlx() -> Any:
    return pytest.importorskip("mlx.core")


def _cache(*, capacity: int = 8, bits: int = 4) -> FixedMlxQuantizedKVCache:
    mx = _mlx()
    return FixedMlxQuantizedKVCache(
        mx=mx,
        create_attention_mask=lambda *args, offset, **kwargs: (args, offset, kwargs),
        capacity=capacity,
        kv_heads=2,
        key_head_dim=64,
        value_head_dim=64,
        key_dtype=mx.bfloat16,
        value_dtype=mx.bfloat16,
        codec=MlxAffineKVCodec(bits=bits, group_size=64),
    )


def test_fixed_quantized_kv_has_exact_capacity_bytes_and_mlx_tuple_shape() -> None:
    mx = _mlx()
    cache = _cache(capacity=8)
    # Per token: two heads * (K + V) * (32 packed bytes + 2 scales/biases * 2 bytes).
    assert cache.nbytes == 8 * 2 * 2 * (32 + 4)
    keys = mx.arange(2 * 3 * 64).reshape(1, 2, 3, 64).astype(mx.bfloat16)
    values = (keys + 1).astype(mx.bfloat16)
    quantized_keys, quantized_values = cache.update_and_fetch(keys, values)
    mx.eval(*cache.storage_arrays())
    assert cache.offset == 3
    assert [tuple(value.shape) for value in quantized_keys] == [
        (1, 2, 3, 8),
        (1, 2, 3, 1),
        (1, 2, 3, 1),
    ]
    assert [tuple(value.shape) for value in quantized_values] == [
        (1, 2, 3, 8),
        (1, 2, 3, 1),
        (1, 2, 3, 1),
    ]
    assert cache.make_mask(1, return_array=True) == ((1,), 3, {"return_array": True})


def test_fixed_quantized_kv_trim_reset_and_overflow_are_transactional() -> None:
    mx = _mlx()
    cache = _cache(capacity=4)
    row = mx.ones((1, 2, 3, 64), dtype=mx.bfloat16)
    cache.update_and_fetch(row, row)
    assert cache.trim(1) == 1
    assert cache.offset == 2
    cache.reset(1)
    assert cache.offset == 1
    overflow = mx.ones((1, 2, 4, 64), dtype=mx.bfloat16)
    with pytest.raises(OverflowError):
        cache.update_and_fetch(overflow, overflow)
    assert cache.offset == 1
    with pytest.raises(ValueError):
        cache.reset(5)


def test_fixed_quantized_kv_fork_copies_exact_packed_prefix_without_aliasing() -> None:
    mx = _mlx()
    source = _cache(capacity=8)
    target = _cache(capacity=6)
    keys = mx.arange(2 * 4 * 64).reshape(1, 2, 4, 64).astype(mx.bfloat16)
    values = (keys * 0.5).astype(mx.bfloat16)
    source.update_and_fetch(keys, values)
    target.copy_committed_prefix_from(source, 3)
    assert target.offset == 3
    assert not {id(value) for value in target.storage_arrays()}.intersection(
        id(value) for value in source.storage_arrays()
    )
    equality = [
        mx.array_equal(left[..., :3, :], right[..., :3, :])
        for left, right in zip(target.storage_arrays(), source.storage_arrays(), strict=True)
    ]
    mx.eval(*equality)
    assert all(bool(value.item()) for value in equality)
    assert target.storage_signature() != source.storage_signature()


def test_affine_kv_codec_identity_and_geometry_fail_closed() -> None:
    codec = MlxAffineKVCodec(bits=4, group_size=64)
    assert codec.codec_id == "mlx-affine-kv4-g64"
    assert codec.vector_bytes(128, source_element_bytes=2) == 72
    assert codec.state_abi("bfloat16").endswith("bfloat16-global-attention-v1")
    with pytest.raises(ValueError, match="divide 32"):
        MlxAffineKVCodec(bits=3, group_size=64)
    with pytest.raises(ValueError, match="divisible by group_size"):
        codec.vector_bytes(96, source_element_bytes=2)


@pytest.mark.parametrize(
    ("bits", "packed_columns", "vector_bytes"),
    ((2, 4, 20), (4, 8, 36), (8, 16, 68)),
)
def test_fixed_affine_codec_supported_widths_match_physical_mlx_arrays(
    bits: int,
    packed_columns: int,
    vector_bytes: int,
) -> None:
    mx = _mlx()
    cache = _cache(capacity=3, bits=bits)
    assert cache.keys[0].shape == (1, 2, 3, packed_columns)
    assert cache.nbytes == 3 * 2 * 2 * vector_bytes
    row = mx.ones((1, 2, 1, 64), dtype=mx.bfloat16)
    cache.update_and_fetch(row, row)
    mx.eval(*cache.storage_arrays())
    assert cache.offset == 1


def test_fixed_affine_cache_rejects_malformed_source_before_indexing_shape() -> None:
    mx = _mlx()
    cache = _cache()
    malformed = mx.ones((2, 64), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="rank-four"):
        cache.update_and_fetch(malformed, malformed)
    assert cache.offset == 0
