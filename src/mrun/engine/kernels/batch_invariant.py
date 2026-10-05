"""Batch-invariant FP32 kernels: every reduction is bracketed by the row and absolute position.

A float reduction is not associative, so the bits of ``sum(x_k)`` depend on the bracket in which
the terms are combined.  cuBLAS and PyTorch choose that bracket from the whole call shape (the
number of rows, split-K heuristics, vectorization), which is why a request's logits change when
other requests share its batch.  The mrun row-stable lanes avoid this by running one B=1 call per
request, paying one weight traversal and one kernel launch per row.

The kernels here fix the bracket instead.  Tile sizes are compile-time constants, there is no
split-K, every output element accumulates its K tiles in ascending order, and attention walks key
tiles aligned to absolute position 0 with an online softmax.  An output row is therefore a pure
function of that row's inputs: packing B rows into one launch reads each weight once and returns
the same bits as B independent launches.  Padding rows or padding key positions contribute exact
zeros and cannot perturb a real row.

Named contract: ``batch-invariant-triton-fp32-v1``.  It is a distinct numerical contract, not a
reproduction of the cuBLAS B=1 bits that ``row_stable`` preserves.  CPU and non-Triton
installations fall back to per-row reference arithmetic, which is batch-invariant by construction
but is not bit-identical to the CUDA kernels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU and Apple installations do not require Triton.
    triton = None
    tl = None


BATCH_INVARIANT_CONTRACT = "batch-invariant-triton-fp32-v1"

_MATMUL_BLOCK_M = 16
_MATMUL_BLOCK_N = 64
_MATMUL_BLOCK_K = 32
_ATTENTION_BLOCK_S = 64

__all__ = [
    "BATCH_INVARIANT_CONTRACT",
    "batch_invariant_attention",
    "batch_invariant_attention_reference",
    "batch_invariant_available",
    "batch_invariant_matmul",
    "batch_invariant_rms_norm",
]


def batch_invariant_available(device: torch.device | str) -> bool:
    return triton is not None and torch.device(device).type == "cuda"


if triton is not None:

    @triton.jit
    def _bi_matmul_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        y_ptr,
        rows,
        out_features,
        in_features,
        stride_xm,
        stride_wn,
        stride_ym,
        HAS_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start_k in range(0, in_features, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            activations = tl.load(
                x_ptr + offsets_m[:, None] * stride_xm + offsets_k[None, :],
                mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < in_features),
                other=0.0,
            ).to(tl.float32)
            weights = tl.load(
                w_ptr + offsets_n[:, None] * stride_wn + offsets_k[None, :],
                mask=(offsets_n[:, None] < out_features) & (offsets_k[None, :] < in_features),
                other=0.0,
            ).to(tl.float32)
            accumulator = tl.dot(
                activations,
                tl.trans(weights),
                accumulator,
                input_precision="ieee",
            )
        if HAS_SCALE:
            scales = tl.load(scale_ptr + offsets_n, mask=offsets_n < out_features, other=0.0)
            accumulator = accumulator * scales[None, :]
        tl.store(
            y_ptr + offsets_m[:, None] * stride_ym + offsets_n[None, :],
            accumulator,
            mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < out_features),
        )

    @triton.jit
    def _bi_rms_norm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        width,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_D)
        valid = offsets < width
        values = tl.load(x_ptr + row * width + offsets, mask=valid, other=0.0).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / width
        inverse_rms = 1.0 / tl.sqrt(mean_square + eps)
        weights = tl.load(w_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * width + offsets, values * inverse_rms * weights, mask=valid)

    @triton.jit
    def _bi_attention_kernel(
        q_ptr,
        new_k_ptr,
        new_v_ptr,
        src_k_ptr,
        src_v_ptr,
        row_ids_ptr,
        src_rows_ptr,
        past_ptr,
        out_ptr,
        stride_qr,
        stride_qt,
        stride_qh,
        stride_nr,
        stride_nt,
        stride_nh,
        stride_sb,
        stride_ss,
        stride_sh,
        stride_or,
        stride_ot,
        stride_oh,
        token_count,
        num_heads,
        group_size,
        head_dim,
        scale,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % num_heads
        item = program // num_heads
        token = item % token_count
        slot = item // token_count
        row = tl.load(row_ids_ptr + slot)
        src_row = tl.load(src_rows_ptr + slot)
        past = tl.load(past_ptr + slot)
        kv_head = head // group_size
        total = past + token + 1

        offsets_d = tl.arange(0, BLOCK_D)
        valid_d = offsets_d < head_dim
        query = tl.load(
            q_ptr + row * stride_qr + token * stride_qt + head * stride_qh + offsets_d,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        running_max = tl.full([1], float("-inf"), tl.float32)
        running_sum = tl.zeros([1], tl.float32)
        context = tl.zeros([BLOCK_D], tl.float32)
        for start in range(0, total, BLOCK_S):
            positions = start + tl.arange(0, BLOCK_S)
            valid_s = positions < total
            from_source = positions < past
            source_mask = (valid_s & from_source)[:, None] & valid_d[None, :]
            new_mask = (valid_s & (positions >= past))[:, None] & valid_d[None, :]
            source_offsets = (
                src_row * stride_sb
                + positions[:, None] * stride_ss
                + kv_head * stride_sh
                + offsets_d[None, :]
            )
            new_offsets = (
                row * stride_nr
                + (positions - past)[:, None] * stride_nt
                + kv_head * stride_nh
                + offsets_d[None, :]
            )
            keys = tl.where(
                from_source[:, None],
                tl.load(src_k_ptr + source_offsets, mask=source_mask, other=0.0),
                tl.load(new_k_ptr + new_offsets, mask=new_mask, other=0.0),
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * scale
            scores = tl.where(valid_s, scores, float("-inf"))
            tile_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, tile_max)
            correction = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max)
            weights = tl.where(valid_s, weights, 0.0)
            values = tl.where(
                from_source[:, None],
                tl.load(src_v_ptr + source_offsets, mask=source_mask, other=0.0),
                tl.load(new_v_ptr + new_offsets, mask=new_mask, other=0.0),
            ).to(tl.float32)
            running_sum = running_sum * correction + tl.sum(weights, axis=0)
            context = context * correction + tl.sum(weights[:, None] * values, axis=0)
            running_max = new_max
        tl.store(
            out_ptr + row * stride_or + token * stride_ot + head * stride_oh + offsets_d,
            context / running_sum,
            mask=valid_d,
        )


def _device_index(values: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor) and values.device == device and values.dtype == torch.int32:
        return values.contiguous()
    return torch.as_tensor([int(value) for value in values], dtype=torch.int32, device=device)


def _per_row_reference_matmul(
    flattened: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor | None,
) -> torch.Tensor:
    dense = weight.float() if scales is None else weight.float() * scales.float()[:, None]
    return torch.cat(
        tuple(flattened[row : row + 1].float() @ dense.T for row in range(flattened.shape[0])),
        dim=0,
    )


@torch.inference_mode()
def batch_invariant_matmul(
    activations: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``activations @ (weight * scales[:, None]).T`` in FP32 with a fixed bracket.

    ``weight`` is ``[out, in]`` (FP32/BF16/FP16, or int8 codes when ``scales`` holds one FP32
    scale per output row).  Output is FP32 with the activation prefix shape.
    """

    if weight.ndim != 2 or activations.shape[-1] != weight.shape[1]:
        raise ValueError("batch-invariant matmul needs activations [..., in] and weight [out, in]")
    if scales is not None and tuple(scales.shape) != (int(weight.shape[0]),):
        raise ValueError("scales must hold one value per output row")
    if activations.device != weight.device:
        raise ValueError("activations and weight must share one device")
    out_features, in_features = (int(value) for value in weight.shape)
    flattened = activations.reshape(-1, in_features)
    rows = int(flattened.shape[0])
    if rows == 0:
        return torch.empty((*activations.shape[:-1], out_features), device=weight.device)
    if not batch_invariant_available(weight.device):
        output = _per_row_reference_matmul(flattened, weight, scales)
        return output.reshape(*activations.shape[:-1], out_features)
    flattened = flattened.contiguous()
    weight = weight.contiguous()
    output = torch.empty((rows, out_features), dtype=torch.float32, device=weight.device)
    grid = (triton.cdiv(rows, _MATMUL_BLOCK_M), triton.cdiv(out_features, _MATMUL_BLOCK_N))
    _bi_matmul_kernel[grid](
        flattened,
        weight,
        scales.contiguous().float() if scales is not None else weight,
        output,
        rows,
        out_features,
        in_features,
        flattened.stride(0),
        weight.stride(0),
        output.stride(0),
        HAS_SCALE=scales is not None,
        BLOCK_M=_MATMUL_BLOCK_M,
        BLOCK_N=_MATMUL_BLOCK_N,
        BLOCK_K=_MATMUL_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*activations.shape[:-1], out_features)


@torch.inference_mode()
def batch_invariant_rms_norm(
    activations: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Row-local FP32 RMSNorm with a width-only reduction bracket."""

    width = int(activations.shape[-1])
    if weight.ndim != 1 or int(weight.numel()) != width:
        raise ValueError("RMSNorm weight width does not match activations")
    flattened = activations.reshape(-1, width)
    if not batch_invariant_available(activations.device):
        output = torch.cat(
            tuple(
                flattened[row : row + 1].float()
                * torch.rsqrt(flattened[row : row + 1].float().pow(2).mean(-1, keepdim=True) + eps)
                * weight.float()
                for row in range(flattened.shape[0])
            ),
            dim=0,
        )
        return output.reshape(activations.shape)
    flattened = flattened.contiguous()
    output = torch.empty(flattened.shape, dtype=torch.float32, device=flattened.device)
    if int(flattened.shape[0]):
        _bi_rms_norm_kernel[(int(flattened.shape[0]),)](
            flattened,
            weight.contiguous(),
            output,
            width,
            float(eps),
            BLOCK_D=triton.next_power_of_2(width),
            num_warps=4,
        )
    return output.reshape(activations.shape)


def batch_invariant_attention_reference(
    query: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    source_keys: torch.Tensor,
    source_values: torch.Tensor,
    row_ids: Sequence[int],
    source_rows: Sequence[int],
    past_lengths: Sequence[int],
) -> torch.Tensor:
    """Reference for :func:`batch_invariant_attention`; returns ``[len(row_ids), T, H, D]``."""

    token_count, num_heads, head_dim = (int(value) for value in query.shape[1:])
    group = num_heads // int(new_keys.shape[2])
    outputs = []
    for row, source_row, past in zip(row_ids, source_rows, past_lengths, strict=True):
        keys = torch.cat((source_keys[int(source_row), : int(past)], new_keys[int(row)]), dim=0)
        values = torch.cat(
            (source_values[int(source_row), : int(past)], new_values[int(row)]), dim=0
        )
        keys = keys.float().repeat_interleave(group, dim=1).transpose(0, 1)
        values = values.float().repeat_interleave(group, dim=1).transpose(0, 1)
        queries = query[int(row)].float().transpose(0, 1)
        scores = queries @ keys.transpose(-1, -2) * head_dim**-0.5
        positions = torch.arange(keys.shape[1], device=query.device)
        allowed = positions[None, :] <= int(past) + torch.arange(token_count, device=query.device)[:, None]
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        outputs.append((torch.softmax(scores, dim=-1) @ values).transpose(0, 1))
    return torch.stack(outputs, dim=0)


@torch.inference_mode()
def batch_invariant_attention(
    query: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    source_keys: torch.Tensor,
    source_values: torch.Tensor,
    row_ids: Sequence[int] | torch.Tensor,
    source_rows: Sequence[int] | torch.Tensor,
    past_lengths: Sequence[int] | torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal GQA attention over ``[source[:past] ‖ new]`` without copying or expanding K/V.

    ``query`` is ``[R, T, H, D]``; ``new_keys/new_values`` are ``[R, T, Hkv, D]`` (post-RoPE
    provisional rows); ``source_keys/values`` are one layer of a committed arena
    ``[Bsrc, capacity, Hkv, D]`` read in place.  For each listed ``row_ids[i]`` the kernel reads
    ``source_rows[i]`` up to ``past_lengths[i]`` and writes ``out[row_ids[i]]``.  Key tiles are
    aligned to absolute position 0, so a row's bits depend on nothing but that row.

    The three index arguments may be int32 device tensors; callers that launch once per layer
    should upload them once per forward, because a pageable host-to-device copy synchronizes
    the stream and a per-layer upload serializes the whole forward.
    """

    if query.ndim != 4 or new_keys.ndim != 4 or source_keys.ndim != 4:
        raise ValueError("attention tensors must be rank four")
    token_count, num_heads, head_dim = (int(value) for value in query.shape[1:])
    num_kv_heads = int(new_keys.shape[2])
    if num_heads % num_kv_heads:
        raise ValueError("query heads must be a multiple of key/value heads")
    if not (len(row_ids) == len(source_rows) == len(past_lengths)):
        raise ValueError("row_ids, source_rows, and past_lengths must align")
    if out is None:
        out = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    if len(row_ids) == 0:
        return out
    if not batch_invariant_available(query.device):
        rows_list = [int(value) for value in row_ids]
        reference = batch_invariant_attention_reference(
            query,
            new_keys,
            new_values,
            source_keys,
            source_values,
            rows_list,
            [int(value) for value in source_rows],
            [int(value) for value in past_lengths],
        )
        out[torch.as_tensor(rows_list, dtype=torch.long, device=out.device)] = reference
        return out
    for tensor in (query, new_keys, new_values, source_keys, source_values, out):
        if tensor.stride(-1) != 1:
            raise ValueError("attention tensors must be contiguous in the head dimension")
    if source_keys.stride() != source_values.stride() or new_keys.stride() != new_values.stride():
        raise ValueError("key and value tensors must share strides")
    device = query.device
    row_tensor = _device_index(row_ids, device)
    source_tensor = _device_index(source_rows, device)
    past_tensor = _device_index(past_lengths, device)
    _bi_attention_kernel[(len(row_ids) * token_count * num_heads,)](
        query,
        new_keys,
        new_values,
        source_keys,
        source_values,
        row_tensor,
        source_tensor,
        past_tensor,
        out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        new_keys.stride(0),
        new_keys.stride(1),
        new_keys.stride(2),
        source_keys.stride(0),
        source_keys.stride(1),
        source_keys.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        token_count,
        num_heads,
        num_heads // num_kv_heads,
        head_dim,
        head_dim**-0.5,
        BLOCK_S=_ATTENTION_BLOCK_S,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out
