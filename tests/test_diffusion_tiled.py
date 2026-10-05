"""Tests for MARS tiled VAE/renderer boundary and dirty-region semantics."""

from __future__ import annotations

import torch
from torch import nn

from mrun.diffusion import (
    TiledDecodeError,
    decode_dirty_tiled,
    decode_tiled,
    dirty_window_indices,
    plan_tiles,
)


def test_tiled_decode_with_halo_matches_local_full_decode() -> None:
    torch.manual_seed(51)
    decoder = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True).eval()
    latents = torch.randn(1, 2, 16, 16)
    full = decoder(latents)
    tiled = decode_tiled(decoder, latents, tile_height=8, tile_width=8, halo=1)
    torch.testing.assert_close(tiled.output, full, rtol=1e-6, atol=1e-7)
    assert tiled.telemetry["tile_count"] == 4
    assert tiled.telemetry["expanded_pixels"] > tiled.telemetry["core_pixels"]


def test_dirty_tiled_decode_reuses_untouched_output() -> None:
    torch.manual_seed(52)
    decoder = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True).eval()
    old_latents = torch.randn(1, 2, 16, 16)
    new_latents = old_latents.clone()
    new_latents[:, :, 2, 2] += 0.75
    changed = torch.zeros(16, 16, dtype=torch.bool)
    changed[2, 2] = True
    # Full and expanded-tile shapes must use the same CPU oracle backend;
    # oneDNN's shape-specific reduction order is not a byte-parity oracle.
    with torch.backends.mkldnn.flags(enabled=False):
        previous = decoder(old_latents)
        expected = decoder(new_latents)
        dirty = decode_dirty_tiled(
            decoder,
            new_latents,
            previous,
            changed,
            tile_height=8,
            tile_width=8,
            halo=1,
        )
    assert torch.equal(dirty.output, expected)
    untouched = torch.ones_like(changed)
    untouched[:8, :8] = False
    assert torch.equal(dirty.output[..., untouched], previous[..., untouched])
    assert dirty.telemetry["dirty_tile_count"] == 1
    assert dirty.telemetry["tile_count"] == 4


def test_dirty_tile_selection_and_noop_preservation() -> None:
    windows = plan_tiles(16, 16, tile_height=8, tile_width=8, halo=1)
    changed = torch.zeros(16, 16, dtype=torch.bool)
    assert dirty_window_indices(changed, windows, halo=1) == ()
    changed[8, 8] = True
    assert dirty_window_indices(changed, windows, halo=1) == (0, 1, 2, 3)


def test_tiled_decode_rejects_wrong_latent_rank() -> None:
    with torch.no_grad():
        try:
            decode_tiled(lambda value: value, torch.zeros(2, 2, 2), tile_height=2, tile_width=2)
        except TiledDecodeError:
            pass
        else:  # pragma: no cover - assertion kept explicit for the contract
            raise AssertionError("rank mismatch must fail closed")
