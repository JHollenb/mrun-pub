from __future__ import annotations

import torch

from mrun.diffusion.saturn_kernels import fused_active_residual, next_power_of_two


def test_next_power_of_two():
    assert next_power_of_two(1) == 1
    assert next_power_of_two(129) == 256


def test_fused_active_residual_matches_index_copy_on_cpu():
    base = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    active_indices = torch.tensor([1, 4], dtype=torch.int64)
    active_delta = torch.full((2, 4), 0.5)
    expected = base.clone().index_copy(
        0, active_indices, base.index_select(0, active_indices) + active_delta
    )
    actual = fused_active_residual(base, active_delta, active_indices)
    torch.testing.assert_close(actual, expected)


def test_fused_active_residual_empty_route():
    base = torch.ones((3, 2), dtype=torch.float32)
    actual = fused_active_residual(
        base,
        torch.empty((0, 2), dtype=base.dtype),
        torch.empty((0,), dtype=torch.int64),
    )
    torch.testing.assert_close(actual, base)
