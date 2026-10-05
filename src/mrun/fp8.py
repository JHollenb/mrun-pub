"""Block/row/tensor-scaled FP8 weight dequantization, shared by every checkpoint reader.

An FP8 checkpoint stores ``<key>.weight`` as raw e4m3 CODES plus a companion
``<key>.weight_scale_inv`` (DeepSeek/Qwen block layout) or ``<key>.weight_scale``. Casting the
codes straight to bf16/fp32 — which is what a reader that only knows about dtypes does — yields
numbers off by the block scale: plausible-looking, wrong everywhere. Both of mrun's
name-addressed readers (``engine.qwen3_moe_cuda.TensorReader`` for the fused expert-store build,
``mri.moe_stream.PagedSafetensors`` for the mmap streaming forward) need the same expansion, so
it lives here once rather than being copied into each.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "FP8_SCALE_SUFFIXES",
    "DEQUANT_ROW_BUDGET",
    "FP4_E2M1_VALUES",
    "declared_block_size",
    "dequantize_block_fp8",
    "dequantize_fp8_rows",
    "dequantize_fp4_packed",
    "is_fp4_packed",
    "resolve_block_size",
    "scale_key_for",
    "scale_to_float32",
]

#: Suffixes a checkpoint appends to a weight name to store its dequantization scale.
FP8_SCALE_SUFFIXES = ("_scale_inv", "_scale")

#: fp32 elements held per dequantization chunk (16 MB). See :func:`dequantize_block_fp8`.
DEQUANT_ROW_BUDGET = 1 << 22

#: e2m1 (FP4) code -> value lookup, 4-bit code 0..15: sign(1) exp(2) mant(1).
#: Same table as ``transformers.integrations.finegrained_fp8.Fp8Dequantize._FP4_E2M1_LUT``
#: and the DeepSeek reference kernel — the independent implementations agree that the LOW
#: nibble of each packed byte is the EVEN (lower-index) element along the last dim.
FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def scale_key_for(weight_map: Any, key: str) -> str | None:
    """Name of ``key``'s companion FP8 scale tensor, or ``None`` when it is stored unquantized.

    ``weight_map`` is anything supporting ``in`` (a dict of tensor name -> shard, a set of keys).

    Two naming families exist: HF-style ``<key>_scale_inv`` / ``<key>_scale`` appended to the
    full tensor name, and the DeepSeek-native release layout where ``<module>.weight`` pairs
    with ``<module>.scale`` (the ``.weight`` suffix is REPLACED, not appended).
    """
    for suffix in FP8_SCALE_SUFFIXES:
        if f"{key}{suffix}" in weight_map:
            return f"{key}{suffix}"
    if key.endswith(".weight"):
        native = key[: -len(".weight")] + ".scale"
        if native in weight_map:
            return native
    return None


def scale_to_float32(scale: torch.Tensor) -> torch.Tensor:
    """A scale tensor as fp32 real numbers, whatever dtype the checkpoint stored.

    ``float8_e8m0fnu`` (power-of-two exponent bytes, DeepSeek ``ue8m0``) casts exactly via
    torch (measured: ``.to(float32)`` == ``2^(byte-127)`` for all 255 finite codes); a raw
    ``uint8``-typed e8m0 save needs the explicit exp2 — interpreting the bytes as numbers
    would be silently wrong everywhere.
    """
    if scale.dtype == torch.uint8:
        return torch.exp2(scale.to(torch.float32) - 127.0)
    return scale.to(torch.float32)


def is_fp4_packed(codes: torch.Tensor, scale: torch.Tensor | None) -> bool:
    """True when ``codes`` is a packed-fp4 weight (two e2m1 nibbles per byte, group scale).

    FP8 codes come out of safetensors as ``float8_e4m3fn`` so the dtypes are disjoint: a
    quantized weight stored as int8/uint8/``float4_e2m1fn_x2`` with a per-row scale grid is
    the DeepSeek fp4 expert layout ``[out, in//2]`` + e8m0 ``[out, in//group]``.
    """
    if scale is None or codes.ndim != 2:
        return False
    fp4_x2 = getattr(torch, "float4_e2m1fn_x2", None)
    if codes.dtype not in (torch.int8, torch.uint8) and (
        fp4_x2 is None or codes.dtype != fp4_x2
    ):
        return False
    return scale.ndim == 2 and int(scale.shape[0]) == int(codes.shape[0])


def dequantize_fp4_packed(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand a packed-fp4 weight ``[out, in//2]`` (+ per-row group scale) to real numbers.

    Layout (DeepSeek-V4 experts, verified Stage-0 and against transformers'
    ``Fp8Dequantize._unpack_fp4``): each byte packs TWO e2m1 values along the input dim —
    low nibble = even column, high nibble = odd column; ``scale`` is ``[out, in//group]``
    e8m0 with group along the input dim (group 32 for the V4 checkpoints).
    """
    if codes.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"unsupported fp4 dequant shapes: {tuple(codes.shape)} / {tuple(scale.shape)}"
        )
    rows, half = int(codes.shape[0]), int(codes.shape[1])
    cols = 2 * half
    if int(scale.shape[0]) != rows or cols % int(scale.shape[1]):
        raise ValueError(
            f"fp4 scale grid {tuple(scale.shape)} does not tile weight [{rows}, {cols}]"
        )
    group = cols // int(scale.shape[1])
    lut = torch.tensor(FP4_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    u8 = codes.contiguous().view(torch.uint8)
    values = torch.empty((rows, cols), dtype=torch.float32, device=codes.device)
    values[:, 0::2] = lut[(u8 & 0xF).long()]
    values[:, 1::2] = lut[(u8 >> 4).long()]
    scale32 = scale_to_float32(scale)
    col_index = torch.arange(cols, device=codes.device) // group
    values *= scale32.index_select(1, col_index)
    return values.to(dtype)


def declared_block_size(cfg: Any) -> tuple[int, int] | None:
    """``quantization_config.weight_block_size`` from a parsed ``config.json``, or ``None``.

    This is the checkpoint's own statement of its block geometry and it OUTRANKS inference,
    which cannot always recover the block from shapes alone (see :func:`_infer_block`).
    """
    if not isinstance(cfg, dict):
        return None
    quantization = cfg.get("quantization_config")
    if not isinstance(quantization, dict):
        return None
    declared = quantization.get("weight_block_size")
    if isinstance(declared, (list, tuple)) and len(declared) >= 2:
        return (int(declared[0]), int(declared[1]))
    return None


def _infer_block(length: int, groups: int, declared: int | None, axis: str) -> int:
    """Block size along one axis: the checkpoint's DECLARED value when there is one, else a
    conservatively inferred quotient.

    Shapes alone do not determine the block, and getting it wrong is silent: a 300-row weight
    with 3 scale groups is 128-blocked (128/128/44), but ``300 // 3`` "infers" 100 and then
    multiplies most rows by a neighbouring block's scale — plausible-looking numbers, wrong
    everywhere. Neither ``ceil(length/groups)`` nor exact division rescues that case, so
    inference is accepted only when the quotient is exact AND a power of two (every FP8 block
    quantizer in use blocks by 64/128/256); anything else refuses and asks for the declared
    ``quantization_config.weight_block_size``.
    """
    if declared:
        if -(-length // declared) != groups:
            raise ValueError(
                f"declared {axis} block {declared} implies {-(-length // declared)} scale "
                f"groups, checkpoint stores {groups}"
            )
        return declared
    if groups == 1:
        return length
    quotient = length // groups
    if length % groups or quotient & (quotient - 1):
        raise ValueError(
            f"cannot infer the fp8 {axis} block size from {length} elements in {groups} scale "
            f"groups (quotient {length / groups:g} is not an exact power of two). Pass "
            f"block_size=(rows, cols) from the checkpoint's "
            f"quantization_config.weight_block_size."
        )
    return quotient


def resolve_block_size(
    rows: int,
    cols: int,
    scale_shape: Sequence[int],
    declared: Sequence[int] | None = None,
) -> tuple[int, int]:
    """``(block_rows, block_cols)`` for a 2-D block scale, declared-value-first."""
    declared_rows, declared_cols = (declared or (None, None))[:2]
    return (
        _infer_block(rows, int(scale_shape[0]), declared_rows, "row"),
        _infer_block(cols, int(scale_shape[1]), declared_cols, "column"),
    )


def dequantize_fp8_rows(
    code_rows: torch.Tensor,
    scale_rows: torch.Tensor,
    *,
    block_cols: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a ROW-GATHERED slice of a block-scaled fp8 weight.

    Separate from :func:`dequantize_block_fp8` because that one derives each row's scale block
    from the row's position in the FULL tensor. After a scattered gather those positions are
    gone, so the caller resolves each gathered row's scale block itself and passes the matching
    scale rows here — otherwise a subset read would silently pick up the wrong block's scale,
    which is the same class of error as skipping the scale entirely.

    ``code_rows``: ``[k, cols]`` fp8. ``scale_rows``: ``[k, n_col_blocks]`` fp32, already
    row-aligned to ``code_rows``.
    """
    if code_rows.ndim != 2 or scale_rows.ndim != 2:
        raise ValueError(
            f"expected 2-D rows and scales, got {tuple(code_rows.shape)} / "
            f"{tuple(scale_rows.shape)}"
        )
    if code_rows.shape[0] != scale_rows.shape[0]:
        raise ValueError("scale rows are not aligned to the gathered code rows")
    cols = int(code_rows.shape[1])
    col_index = torch.arange(cols) // block_cols
    if int(col_index.max()) >= scale_rows.shape[1]:
        raise ValueError(
            f"column block {block_cols} needs {int(col_index.max()) + 1} scale groups, "
            f"got {scale_rows.shape[1]}"
        )
    return (code_rows.to(torch.float32) * scale_rows.index_select(1, col_index)).to(dtype)


def dequantize_block_fp8(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Expand a block/row/tensor-scaled FP8 weight back to real numbers.

    Covers the three scale ranks an FP8 checkpoint uses: 0-d (one scale for the tensor), 1-d
    (per output row — the codec ``build_fp8_expert_store`` writes), and 2-d (the DeepSeek/Qwen
    ``weight_scale_inv`` block layout). ``block_size`` is the checkpoint's declared
    ``quantization_config.weight_block_size``; without it the block is inferred, which is exact
    only when both axes divide evenly into a power of two and raises otherwise
    (see :func:`_infer_block`).

    Chunked over rows: a naive ``codes.float() * expanded_scale`` allocates two fp32 temporaries
    the size of the weight, which for a 151936x6144 embedding is 7.4 GB of transient RSS — the
    exact class of blowup a streaming reader exists to avoid.
    """
    if is_fp4_packed(codes, scale):
        return dequantize_fp4_packed(codes, scale, dtype=dtype)
    if scale.ndim == 0:
        return (codes.to(torch.float32) * scale_to_float32(scale)).to(dtype)
    if codes.ndim != 2 or scale.ndim > 2:
        raise ValueError(
            f"unsupported fp8 dequant shapes: {tuple(codes.shape)} / {tuple(scale.shape)}"
        )
    rows, cols = int(codes.shape[0]), int(codes.shape[1])
    declared_rows, declared_cols = (block_size or (None, None))[:2]
    scale32 = scale_to_float32(scale)
    if scale32.ndim == 1:
        if scale32.shape[0] != rows:
            raise ValueError(f"row scale {tuple(scale.shape)} does not match weight rows {rows}")
        scale32 = scale32[:, None]
        col_index = torch.zeros(cols, dtype=torch.long)
    else:
        col_index = torch.arange(cols) // _infer_block(
            cols, int(scale32.shape[1]), declared_cols, "column"
        )
    row_index = torch.arange(rows) // _infer_block(
        rows, int(scale32.shape[0]), declared_rows, "row"
    )
    out = torch.empty((rows, cols), dtype=dtype)
    chunk = max(1, DEQUANT_ROW_BUDGET // max(1, cols))
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        block = codes[start:stop].to(torch.float32)
        block *= scale32.index_select(0, row_index[start:stop]).index_select(1, col_index)
        out[start:stop] = block.to(dtype)
    return out
