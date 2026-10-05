"""mri.moe_stream banded on-device dequant — bit-exactness + transient-footprint gate.

``PagedSafetensors.get_on`` used to widen a whole FP8 weight to fp32 and multiply that into a
second full-size fp32 tensor before narrowing: ~10 bytes/param of transient to produce a
2 bytes/param bf16 weight, and ~14 for the ragged-block branch (which additionally materialized
a full ``[rows, cols]`` expanded scale — the exact blowup the aligned branch's grid broadcast
was written to avoid). That cost is why FP8 paging measured 1.3x on the mstack FLUX DiT instead
of the 2.0x its halved byte count predicts.

The replacement bands over rows. These tests pin the property that makes the change safe:
banding changes no element's arithmetic, so the output is ``torch.equal`` to the naive
expression — not close to it. The band size is forced small so the multi-band path is the one
under test rather than a single-band degenerate case.
"""
from __future__ import annotations

import pytest
import torch

# `mrun.mri.__init__` is a facade that forwards unknown attributes to `manalysis.mri`, so
# `from mrun.mri import moe_stream` would try to import manalysis. Import the submodule directly,
# exactly as `mstack/experiments/paged_dit/paged_dit.py` does.
import mrun.engine.moe_safetensors as moe_stream
from mrun.engine.moe_safetensors import _dequant_banded_on_device, _dequant_blocked_on_device

DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.fixture
def tiny_band(monkeypatch):
    """Force several bands per tensor so the loop, not a single pass, is exercised."""
    monkeypatch.setattr(moe_stream, "_DEVICE_DEQUANT_ELEMS", 64)


def _codes(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    values = (torch.randn(rows, cols, generator=g) * 6.0).clamp(-448.0, 448.0)
    return values.to(torch.float8_e4m3fn)


def _naive_scalar(codes, scale, dtype):
    return (codes.to(torch.float32) * scale).to(dtype)


def _naive_rows(codes, scale, dtype):
    return (codes.to(torch.float32) * scale[:, None]).to(dtype)


def _naive_blocked_aligned(codes, scale, dtype, block_rows, block_cols):
    rows, cols = codes.shape
    values = codes.to(torch.float32)
    grid = values.reshape(rows // block_rows, block_rows, cols // block_cols, block_cols)
    grid = grid * scale.reshape(scale.shape[0], 1, scale.shape[1], 1)
    return grid.reshape(rows, cols).to(dtype)


def _naive_blocked_ragged(codes, scale, dtype, block_rows, block_cols):
    rows, cols = codes.shape
    values = codes.to(torch.float32)
    row_index = torch.arange(rows) // block_rows
    col_index = torch.arange(cols) // block_cols
    return (values * scale.index_select(0, row_index).index_select(1, col_index)).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_scale_bit_exact(dtype, tiny_band):
    codes = _codes(37, 23)
    scale = torch.tensor(0.031_25)
    got = _dequant_banded_on_device(codes, scale, dtype=dtype)
    assert torch.equal(got, _naive_scalar(codes, scale, dtype))
    assert got.dtype is dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_row_scale_bit_exact(dtype, tiny_band):
    codes = _codes(41, 29, seed=1)
    scale = torch.rand(41) * 0.5 + 1e-3
    got = _dequant_banded_on_device(codes, scale[:, None], dtype=dtype)
    assert torch.equal(got, _naive_rows(codes, scale, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,block", [((64, 48), (16, 16)), ((128, 128), (128, 128))])
def test_blocked_aligned_bit_exact(dtype, shape, block, tiny_band):
    rows, cols = shape
    block_rows, block_cols = block
    codes = _codes(rows, cols, seed=2)
    scale = torch.rand(rows // block_rows, cols // block_cols) * 0.25 + 1e-3
    got = _dequant_blocked_on_device(
        codes, scale, dtype=dtype, block_rows=block_rows, block_cols=block_cols
    )
    assert torch.equal(got, _naive_blocked_aligned(codes, scale, dtype, block_rows, block_cols))


@pytest.mark.parametrize("dtype", DTYPES)
def test_blocked_ragged_bit_exact(dtype, tiny_band):
    """rows/cols not multiples of the block — the branch that used to expand a full-size scale."""
    rows, cols, block_rows, block_cols = 70, 50, 16, 16
    codes = _codes(rows, cols, seed=3)
    scale = torch.rand((rows + block_rows - 1) // block_rows, (cols + block_cols - 1) // block_cols)
    scale = scale * 0.25 + 1e-3
    got = _dequant_blocked_on_device(
        codes, scale, dtype=dtype, block_rows=block_rows, block_cols=block_cols
    )
    assert torch.equal(got, _naive_blocked_ragged(codes, scale, dtype, block_rows, block_cols))


def test_band_size_does_not_change_the_result(monkeypatch):
    """The band is a scheduling knob, not a numerical one: every band size gives one answer."""
    codes = _codes(96, 64, seed=4)
    scale = torch.rand(6, 4) * 0.25 + 1e-3
    results = []
    for elems in (16, 256, 4096, 1 << 24):
        monkeypatch.setattr(moe_stream, "_DEVICE_DEQUANT_ELEMS", elems)
        results.append(
            _dequant_blocked_on_device(
                codes, scale, dtype=torch.bfloat16, block_rows=16, block_cols=16
            )
        )
    for other in results[1:]:
        assert torch.equal(results[0], other)


def test_non_2d_codes_fall_back_without_banding(tiny_band):
    codes = torch.arange(12, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.5)
    got = _dequant_banded_on_device(codes, scale, dtype=torch.float32)
    assert torch.equal(got, _naive_scalar(codes, scale, torch.float32))


def test_single_band_covers_whole_tensor_by_default():
    """The production band must not fragment an ordinary weight into many launches."""
    cols = 6144
    band = max(1, moe_stream._DEVICE_DEQUANT_ELEMS // cols)
    assert band >= 2048, "a 12288x6144 weight should dequantize in a handful of bands, not hundreds"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="transient footprint is a CUDA claim")
@pytest.mark.parametrize("block", [(128, 128)])
def test_cuda_transient_footprint_bounded(block):
    """Peak transient must be a band, not the whole tensor: the point of the change."""
    rows, cols = 4096, 4096
    block_rows, block_cols = block
    codes = _codes(rows, cols, seed=5).cuda()
    scale = (torch.rand(rows // block_rows, cols // block_cols) * 0.25 + 1e-3).cuda()
    output_bytes = rows * cols * 2  # bf16

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    got = _dequant_blocked_on_device(
        codes, scale, dtype=torch.bfloat16, block_rows=block_rows, block_cols=block_cols
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base

    assert torch.equal(
        got, _naive_blocked_aligned(codes, scale, torch.bfloat16, block_rows, block_cols)
    )
    assert peak <= 1.4 * output_bytes, f"transient {peak} > 1.4x output {output_bytes}"
