"""Compact row-segmented grouped-query attention for one-token CUDA decode.

Each logical decode row names a physical K/V slot and its own sequence length.  The CUDA kernel
reads that prefix directly, maps each query head to its grouped K/V head, and performs an online
softmax without padding, prefix gathering, or physical K/V-head expansion.  A separate tiny
scatter writes exactly one provisional K/V position per logical row and K/V head.

The CPU implementation is deliberately a reference, not a production fallback.  CUDA fails
closed when Triton is unavailable so a serving route cannot silently acquire padded or expanded
attention arithmetic.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


class SegmentedGQADecodeError(RuntimeError):
    """The compact segmented-GQA execution contract was violated."""


def _validate_shapes(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    physical_slots: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 3:
        raise ValueError(f"query must have shape [rows,query_heads,head_dim], got {query.shape}")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("segmented K/V caches must have shape [slots,kv_heads,capacity,head_dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("segmented key and value cache shapes must match")
    rows, query_heads, head_dim = map(int, query.shape)
    slots, kv_heads, capacity, cache_head_dim = map(int, key_cache.shape)
    if rows <= 0 or query_heads <= 0 or kv_heads <= 0 or head_dim <= 0 or capacity <= 0:
        raise ValueError("segmented GQA dimensions must be positive")
    if head_dim != cache_head_dim:
        raise ValueError("segmented query and K/V head dimensions must match")
    if query_heads % kv_heads:
        raise ValueError("query heads must be an exact multiple of K/V heads")
    if physical_slots.ndim != 1 or tuple(physical_slots.shape) != (rows,):
        raise ValueError("physical_slots must contain one slot per query row")
    if sequence_lengths.ndim != 1 or tuple(sequence_lengths.shape) != (rows,):
        raise ValueError("sequence_lengths must contain one length per query row")
    if physical_slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("physical_slots must use an integer tensor dtype")
    if sequence_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("sequence_lengths must use an integer tensor dtype")
    tensors = (query, key_cache, value_cache, physical_slots, sequence_lengths)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("segmented GQA tensors must share one device")
    if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
        raise TypeError("segmented query and K/V cache dtypes must match")
    return rows, query_heads, kv_heads, capacity, head_dim


def _validate_host_indices(
    physical_slots: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    slots: int,
    capacity: int,
) -> None:
    slot_values = tuple(int(value) for value in physical_slots.tolist())
    length_values = tuple(int(value) for value in sequence_lengths.tolist())
    if len(set(slot_values)) != len(slot_values):
        raise ValueError("segmented GQA physical slots must be unique")
    if any(value < 0 or value >= slots for value in slot_values):
        raise IndexError("segmented GQA physical slot is out of range")
    if any(value <= 0 or value > capacity for value in length_values):
        raise ValueError("segmented GQA sequence length is outside cache capacity")


def segmented_gqa_decode_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    physical_slots: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Literal compact-GQA reference with no head expansion or rectangular padding."""

    rows, query_heads, kv_heads, capacity, head_dim = _validate_shapes(
        query,
        key_cache,
        value_cache,
        physical_slots,
        sequence_lengths,
    )
    _validate_host_indices(
        physical_slots,
        sequence_lengths,
        slots=int(key_cache.shape[0]),
        capacity=capacity,
    )
    attention_scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("segmented GQA scale must be finite and positive")
    heads_per_kv = query_heads // kv_heads
    row_outputs: list[torch.Tensor] = []
    for row in range(rows):
        slot = int(physical_slots[row].item())
        length = int(sequence_lengths[row].item())
        head_outputs: list[torch.Tensor] = []
        for query_head in range(query_heads):
            kv_head = query_head // heads_per_kv
            keys = key_cache[slot, kv_head, :length].float()
            values = value_cache[slot, kv_head, :length].float()
            scores = (keys @ query[row, query_head].float()) * attention_scale
            probabilities = torch.softmax(scores, dim=0)
            head_outputs.append(probabilities @ values)
        row_outputs.append(torch.stack(head_outputs, dim=0))
    return torch.stack(row_outputs, dim=0).to(query.dtype)


def segmented_gqa_decode_cow_reference(
    query: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    branch_key: torch.Tensor,
    branch_value: torch.Tensor,
    branch_lengths: torch.Tensor,
    *,
    parent_length: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Literal immutable-parent plus branch-delta GQA reference.

    The parent has one physical row and is broadcast logically to every branch. Branch K/V owns
    only tokens written after the cut. This is the semantic reference for the CUDA COW kernel;
    concatenation exists only inside this CPU oracle and never in the serving path.
    """

    rows, query_heads, head_dim = map(int, query.shape)
    if parent_key.ndim != 4 or parent_value.shape != parent_key.shape:
        raise ValueError("COW parent K/V must have matching [1,kv_heads,capacity,head_dim] shapes")
    if branch_key.ndim != 4 or branch_value.shape != branch_key.shape:
        raise ValueError(
            "COW branch K/V must have matching [rows,kv_heads,capacity,head_dim] shapes"
        )
    if int(parent_key.shape[0]) != 1 or int(branch_key.shape[0]) != rows:
        raise ValueError("COW attention requires one parent row and one delta row per query")
    kv_heads = int(parent_key.shape[1])
    if tuple(branch_key.shape[1::2]) != (kv_heads, head_dim):
        raise ValueError("COW parent, branch, and query head geometry must agree")
    if int(parent_key.shape[-1]) != head_dim or query_heads % kv_heads:
        raise ValueError("COW query heads must group exactly over parent K/V heads")
    tensors = (query, parent_key, parent_value, branch_key, branch_value, branch_lengths)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("COW attention tensors must share one device")
    if len({query.dtype, parent_key.dtype, parent_value.dtype, branch_key.dtype, branch_value.dtype}) != 1:
        raise TypeError("COW query and K/V tensors must share one dtype")
    if branch_lengths.ndim != 1 or tuple(branch_lengths.shape) != (rows,):
        raise ValueError("COW branch_lengths must contain one length per query row")
    if branch_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("COW branch_lengths must use an integer dtype")
    if isinstance(parent_length, bool) or not isinstance(parent_length, int):
        raise TypeError("COW parent_length must be an integer")
    if parent_length <= 0 or parent_length > int(parent_key.shape[2]):
        raise ValueError("COW parent_length is outside parent K/V capacity")
    lengths = tuple(int(value) for value in branch_lengths.tolist())
    if any(length < 0 or length > int(branch_key.shape[2]) for length in lengths):
        raise ValueError("COW branch length is outside branch K/V capacity")
    attention_scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("COW GQA scale must be finite and positive")
    heads_per_kv = query_heads // kv_heads
    row_outputs: list[torch.Tensor] = []
    for row, branch_length in enumerate(lengths):
        head_outputs: list[torch.Tensor] = []
        for query_head in range(query_heads):
            kv_head = query_head // heads_per_kv
            keys = torch.cat(
                (
                    parent_key[0, kv_head, :parent_length],
                    branch_key[row, kv_head, :branch_length],
                ),
                dim=0,
            ).float()
            values = torch.cat(
                (
                    parent_value[0, kv_head, :parent_length],
                    branch_value[row, kv_head, :branch_length],
                ),
                dim=0,
            ).float()
            scores = (keys @ query[row, query_head].float()) * attention_scale
            head_outputs.append(torch.softmax(scores, dim=0) @ values)
        row_outputs.append(torch.stack(head_outputs, dim=0))
    return torch.stack(row_outputs, dim=0).to(query.dtype)


if triton is not None:

    @triton.jit
    def _scatter_segmented_kv_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        slots_ptr,
        positions_ptr,
        key_row_stride,
        key_head_stride,
        key_dim_stride,
        value_row_stride,
        value_head_stride,
        value_dim_stride,
        cache_slot_stride,
        cache_head_stride,
        cache_token_stride,
        cache_dim_stride,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        kv_head = tl.program_id(1)
        offsets = tl.arange(0, block_dim)
        mask = offsets < head_dim
        slot = tl.load(slots_ptr + row)
        position = tl.load(positions_ptr + row)
        key_source = (
            row * key_row_stride + kv_head * key_head_stride + offsets * key_dim_stride
        )
        value_source = (
            row * value_row_stride
            + kv_head * value_head_stride
            + offsets * value_dim_stride
        )
        target = (
            slot * cache_slot_stride
            + kv_head * cache_head_stride
            + position * cache_token_stride
            + offsets * cache_dim_stride
        )
        key = tl.load(key_ptr + key_source, mask=mask, other=0.0)
        value = tl.load(value_ptr + value_source, mask=mask, other=0.0)
        tl.store(key_cache_ptr + target, key, mask=mask)
        tl.store(value_cache_ptr + target, value, mask=mask)

    @triton.jit
    def _segmented_gqa_decode_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        slots_ptr,
        lengths_ptr,
        output_ptr,
        query_row_stride,
        query_head_stride,
        query_dim_stride,
        cache_slot_stride,
        cache_head_stride,
        cache_token_stride,
        cache_dim_stride,
        output_row_stride,
        output_head_stride,
        output_dim_stride,
        attention_scale,
        query_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        sequence_bucket: tl.constexpr,
        block_tokens: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        query_head = tl.program_id(1)
        heads_per_kv: tl.constexpr = query_heads // kv_heads
        kv_head = query_head // heads_per_kv
        slot = tl.load(slots_ptr + row)
        length = tl.load(lengths_ptr + row)
        offsets_d = tl.arange(0, block_dim)
        mask_d = offsets_d < head_dim
        query = tl.load(
            query_ptr
            + row * query_row_stride
            + query_head * query_head_stride
            + offsets_d * query_dim_stride,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((block_dim,), dtype=tl.float32)
        log2e: tl.constexpr = 1.4426950408889634

        for token_start in range(0, sequence_bucket, block_tokens):
            offsets_n = token_start + tl.arange(0, block_tokens)
            mask_n = offsets_n < length
            cache_offsets = (
                slot * cache_slot_stride
                + kv_head * cache_head_stride
                + offsets_n[:, None] * cache_token_stride
                + offsets_d[None, :] * cache_dim_stride
            )
            keys = tl.load(
                key_cache_ptr + cache_offsets,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * attention_scale
            scores = tl.where(mask_n, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            correction = tl.exp2((running_max - next_max) * log2e)
            probabilities = tl.exp2((scores - next_max) * log2e)
            values = tl.load(
                value_cache_ptr + cache_offsets,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(
                probabilities[:, None] * values,
                axis=0,
            )
            running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
            running_max = next_max

        output = accumulator / running_sum
        tl.store(
            output_ptr
            + row * output_row_stride
            + query_head * output_head_stride
            + offsets_d * output_dim_stride,
            output,
            mask=mask_d,
        )

    @triton.jit
    def _segmented_gqa_decode_cow_kernel(
        query_ptr,
        parent_key_ptr,
        parent_value_ptr,
        branch_key_ptr,
        branch_value_ptr,
        branch_lengths_ptr,
        output_ptr,
        query_row_stride,
        query_head_stride,
        query_dim_stride,
        parent_head_stride,
        parent_token_stride,
        parent_dim_stride,
        branch_row_stride,
        branch_head_stride,
        branch_token_stride,
        branch_dim_stride,
        output_row_stride,
        output_head_stride,
        output_dim_stride,
        attention_scale,
        parent_length: tl.constexpr,
        query_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        sequence_bucket: tl.constexpr,
        block_tokens: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        query_head = tl.program_id(1)
        heads_per_kv: tl.constexpr = query_heads // kv_heads
        kv_head = query_head // heads_per_kv
        branch_length = tl.load(branch_lengths_ptr + row)
        total_length = parent_length + branch_length
        offsets_d = tl.arange(0, block_dim)
        mask_d = offsets_d < head_dim
        query = tl.load(
            query_ptr
            + row * query_row_stride
            + query_head * query_head_stride
            + offsets_d * query_dim_stride,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((block_dim,), dtype=tl.float32)
        log2e: tl.constexpr = 1.4426950408889634

        for token_start in range(0, sequence_bucket, block_tokens):
            logical_tokens = token_start + tl.arange(0, block_tokens)
            live_mask = logical_tokens < total_length
            parent_mask = live_mask & (logical_tokens < parent_length)
            branch_tokens = logical_tokens - parent_length
            branch_mask = live_mask & (logical_tokens >= parent_length)
            parent_offsets = (
                kv_head * parent_head_stride
                + logical_tokens[:, None] * parent_token_stride
                + offsets_d[None, :] * parent_dim_stride
            )
            branch_offsets = (
                row * branch_row_stride
                + kv_head * branch_head_stride
                + branch_tokens[:, None] * branch_token_stride
                + offsets_d[None, :] * branch_dim_stride
            )
            parent_keys = tl.load(
                parent_key_ptr + parent_offsets,
                mask=parent_mask[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            branch_keys = tl.load(
                branch_key_ptr + branch_offsets,
                mask=branch_mask[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            keys = tl.where(parent_mask[:, None], parent_keys, branch_keys)
            scores = tl.sum(keys * query[None, :], axis=1) * attention_scale
            scores = tl.where(live_mask, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            correction = tl.exp2((running_max - next_max) * log2e)
            probabilities = tl.exp2((scores - next_max) * log2e)
            parent_values = tl.load(
                parent_value_ptr + parent_offsets,
                mask=parent_mask[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            branch_values = tl.load(
                branch_value_ptr + branch_offsets,
                mask=branch_mask[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            values = tl.where(parent_mask[:, None], parent_values, branch_values)
            accumulator = accumulator * correction + tl.sum(
                probabilities[:, None] * values,
                axis=0,
            )
            running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
            running_max = next_max

        output = accumulator / running_sum
        tl.store(
            output_ptr
            + row * output_row_stride
            + query_head * output_head_stride
            + offsets_d * output_dim_stride,
            output,
            mask=mask_d,
        )


def scatter_segmented_decode_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    physical_slots: torch.Tensor,
    parent_lengths: torch.Tensor,
) -> None:
    """Write one provisional key/value position for every row and compact K/V head."""

    rows, _query_heads, kv_heads, capacity, head_dim = _validate_shapes(
        key,
        key_cache,
        value_cache,
        physical_slots,
        parent_lengths + 1,
    )
    if key.shape != value.shape:
        raise ValueError("provisional key and value shapes must match")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError("provisional key and value must share one device and dtype")
    if int(key.shape[1]) != kv_heads:
        raise ValueError("provisional key/value head count must match the cache")
    if physical_slots.device.type == "cpu":
        _validate_host_indices(
            physical_slots,
            parent_lengths + 1,
            slots=int(key_cache.shape[0]),
            capacity=capacity,
        )
        for row in range(rows):
            slot = int(physical_slots[row].item())
            position = int(parent_lengths[row].item())
            key_cache[slot, :, position].copy_(key[row])
            value_cache[slot, :, position].copy_(value[row])
        return
    if physical_slots.device.type != "cuda":
        raise SegmentedGQADecodeError("segmented K/V scatter supports CPU reference or CUDA")
    if triton is None:
        raise SegmentedGQADecodeError("CUDA segmented K/V scatter requires Triton")
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("CUDA segmented K/V scatter supports fp16 or bf16")
    if key_cache.stride() != value_cache.stride():
        raise ValueError("CUDA segmented K/V scatter requires matching cache strides")
    block_dim = triton.next_power_of_2(head_dim)
    _scatter_segmented_kv_kernel[(rows, kv_heads)](
        key,
        value,
        key_cache,
        value_cache,
        physical_slots,
        parent_lengths,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        head_dim=head_dim,
        block_dim=block_dim,
        num_warps=1,
    )


def segmented_gqa_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    physical_slots: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    max_sequence_length: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Execute one compact, ragged GQA decode attention traversal."""

    rows, query_heads, kv_heads, capacity, head_dim = _validate_shapes(
        query,
        key_cache,
        value_cache,
        physical_slots,
        sequence_lengths,
    )
    attention_scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("segmented GQA scale must be finite and positive")
    if query.device.type == "cpu":
        return segmented_gqa_decode_reference(
            query,
            key_cache,
            value_cache,
            physical_slots,
            sequence_lengths,
            scale=attention_scale,
        )
    if query.device.type != "cuda":
        raise SegmentedGQADecodeError("segmented GQA supports CPU reference or CUDA")
    if triton is None:
        raise SegmentedGQADecodeError("CUDA segmented GQA requires Triton")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("CUDA segmented GQA supports fp16 or bf16")
    if key_cache.stride() != value_cache.stride():
        raise ValueError("CUDA segmented GQA requires matching key/value cache strides")
    if max_sequence_length is None:
        raise ValueError("CUDA segmented GQA requires an explicit host max_sequence_length")
    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or max_sequence_length <= 0
        or max_sequence_length > capacity
    ):
        raise ValueError("max_sequence_length must be in 1..cache capacity")
    # Power-of-two bucketing bounds the compile-cache cardinality as context grows. The Triton
    # token loop is still specialized to this bucket: first-use JIT latency and long-context
    # instruction/register behavior are hardware-gated rather than inferred from the CPU path.
    sequence_bucket = max(32, triton.next_power_of_2(max_sequence_length))
    block_dim = triton.next_power_of_2(head_dim)
    block_tokens = 32
    output = torch.empty_like(query)
    _segmented_gqa_decode_kernel[(rows, query_heads)](
        query,
        key_cache,
        value_cache,
        physical_slots,
        sequence_lengths,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        attention_scale,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        sequence_bucket=sequence_bucket,
        block_tokens=block_tokens,
        block_dim=block_dim,
        num_warps=4 if head_dim >= 64 else 2,
    )
    return output


def segmented_gqa_decode_cow(
    query: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    branch_key: torch.Tensor,
    branch_value: torch.Tensor,
    branch_lengths: torch.Tensor,
    *,
    parent_length: int,
    max_branch_length: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode from one immutable parent plus compact branch-local K/V deltas.

    CUDA reads both segments directly in one online-softmax traversal. It never broadcasts,
    concatenates, or copies parent K/V into branch storage.
    """

    rows, query_heads, head_dim = map(int, query.shape)
    if parent_key.ndim != 4 or parent_value.shape != parent_key.shape:
        raise ValueError("COW parent K/V must have matching rank-four shapes")
    if branch_key.ndim != 4 or branch_value.shape != branch_key.shape:
        raise ValueError("COW branch K/V must have matching rank-four shapes")
    if int(parent_key.shape[0]) != 1 or int(branch_key.shape[0]) != rows:
        raise ValueError("COW attention requires one parent row and one branch row per query")
    kv_heads = int(parent_key.shape[1])
    if (
        int(parent_key.shape[-1]) != head_dim
        or int(branch_key.shape[1]) != kv_heads
        or int(branch_key.shape[-1]) != head_dim
        or query_heads % kv_heads
    ):
        raise ValueError("COW attention head geometry does not agree")
    tensors = (query, parent_key, parent_value, branch_key, branch_value, branch_lengths)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("COW attention tensors must share one device")
    if len({query.dtype, parent_key.dtype, parent_value.dtype, branch_key.dtype, branch_value.dtype}) != 1:
        raise TypeError("COW query and K/V tensors must share one dtype")
    if branch_lengths.ndim != 1 or tuple(branch_lengths.shape) != (rows,):
        raise ValueError("COW branch_lengths must contain one value per query row")
    if branch_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("COW branch_lengths must use an integer dtype")
    if isinstance(parent_length, bool) or not isinstance(parent_length, int):
        raise TypeError("COW parent_length must be an integer")
    if parent_length <= 0 or parent_length > int(parent_key.shape[2]):
        raise ValueError("COW parent_length is outside parent capacity")
    attention_scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("COW GQA scale must be finite and positive")
    if query.device.type == "cpu":
        return segmented_gqa_decode_cow_reference(
            query,
            parent_key,
            parent_value,
            branch_key,
            branch_value,
            branch_lengths,
            parent_length=parent_length,
            scale=attention_scale,
        )
    if query.device.type != "cuda":
        raise SegmentedGQADecodeError("COW segmented GQA supports CPU reference or CUDA")
    if triton is None:
        raise SegmentedGQADecodeError("CUDA COW segmented GQA requires Triton")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("CUDA COW segmented GQA supports fp16 or bf16")
    if parent_key.stride() != parent_value.stride() or branch_key.stride() != branch_value.stride():
        raise ValueError("CUDA COW segmented GQA requires matching K/V strides per segment")
    if (
        max_branch_length is None
        or isinstance(max_branch_length, bool)
        or not isinstance(max_branch_length, int)
        or max_branch_length < 0
        or max_branch_length > int(branch_key.shape[2])
    ):
        raise ValueError("max_branch_length must be in 0..branch capacity")
    sequence_bucket = max(32, triton.next_power_of_2(parent_length + max_branch_length))
    block_dim = triton.next_power_of_2(head_dim)
    output = torch.empty_like(query)
    _segmented_gqa_decode_cow_kernel[(rows, query_heads)](
        query,
        parent_key,
        parent_value,
        branch_key,
        branch_value,
        branch_lengths,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        parent_key.stride(1),
        parent_key.stride(2),
        parent_key.stride(3),
        branch_key.stride(0),
        branch_key.stride(1),
        branch_key.stride(2),
        branch_key.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        attention_scale,
        parent_length=parent_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        sequence_bucket=sequence_bucket,
        block_tokens=32,
        block_dim=block_dim,
        num_warps=4 if head_dim >= 64 else 2,
    )
    return output


__all__ = [
    "SegmentedGQADecodeError",
    "scatter_segmented_decode_kv",
    "segmented_gqa_decode",
    "segmented_gqa_decode_cow",
    "segmented_gqa_decode_cow_reference",
    "segmented_gqa_decode_reference",
]
