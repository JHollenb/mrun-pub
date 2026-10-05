"""Small CUDA kernels used by the SATURN routed latent student.

The first kernel is intentionally narrow: it fuses the fixed-route residual
write that was previously expressed as ``zeros_like`` + ``index_copy``.  The
route itself remains an explicit, testable ``topk`` plan.  This lets us compare
the execution mechanism at an identical route before attempting a monolithic
expert/router kernel.

Triton is optional.  CPU and non-CUDA callers use an exact PyTorch fallback so
the public operation remains importable in the normal mrun test environment.
"""

from __future__ import annotations

from typing import Any

import torch


def backend_name() -> str:
    """Return the selected implementation name without importing Triton eagerly."""

    try:
        import triton  # noqa: F401

        return "triton-active-residual"
    except Exception:  # noqa: BLE001 — optional dependency
        return "torch-index-copy-fallback"


def fused_active_residual(
    base: Any,
    active_delta: Any,
    active_indices: Any,
) -> Any:
    """Return ``base`` with ``active_delta`` added at ``active_indices``.

    ``base`` is ``[N, D]``; ``active_delta`` is ``[K, D]``; indices are unique
    rows in ``[0, N)``.  The Triton path copies the base once and performs the
    active row add in one kernel launch.  The fallback has the same numerical
    ordering as the old implementation and is used for unsupported dtypes,
    devices, or missing Triton.
    """

    if base.ndim != 2 or active_delta.ndim != 2 or active_indices.ndim != 1:
        raise ValueError("SATURN active residual expects [N,D], [K,D], and [K] tensors")
    if base.shape[1] != active_delta.shape[1] or active_delta.shape[0] != active_indices.shape[0]:
        raise ValueError("SATURN active residual shapes do not agree")
    if base.device != active_delta.device or base.device != active_indices.device:
        raise ValueError("SATURN active residual tensors must share a device")

    output = base.clone()
    if active_delta.shape[0] == 0:
        return output
    if not base.is_cuda or active_indices.dtype not in (torch.int32, torch.int64):
        return output.index_copy(0, active_indices, output.index_select(0, active_indices) + active_delta)

    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 — optional CUDA dependency
        return output.index_copy(0, active_indices, output.index_select(0, active_indices) + active_delta)

    if base.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return output.index_copy(0, active_indices, output.index_select(0, active_indices) + active_delta)

    @triton.jit
    def _active_add_kernel(
        base_ptr,
        active_ptr,
        index_ptr,
        output_ptr,
        n_cols,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        source_row = tl.load(index_ptr + row)
        base_offset = source_row * n_cols + cols
        active_offset = row * n_cols + cols
        values = tl.load(base_ptr + base_offset, mask=mask, other=0.0)
        delta = tl.load(active_ptr + active_offset, mask=mask, other=0.0)
        tl.store(output_ptr + base_offset, values + delta, mask=mask)

    grid = (active_delta.shape[0],)
    block = min(next_power_of_two(int(base.shape[1])), 1024)
    _active_add_kernel[grid](
        base,
        active_delta,
        active_indices,
        output,
        int(base.shape[1]),
        BLOCK=block,
    )
    return output


def next_power_of_two(value: int) -> int:
    """Small local helper kept dependency-free for kernel block selection."""

    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


__all__ = ["backend_name", "fused_active_residual", "next_power_of_two"]
