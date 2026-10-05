"""Tiled decoder and dirty-region planning primitives for the MARS VAE boundary.

The decoder is treated as a callable component. A halo is explicit because a
tile cannot be independently decoded safely unless its receptive field is
available at the tile edge. The implementation reports tile identity and
coverage as debugger metadata; it does not assume that every VAE is local or
that tiled output is automatically pixel-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class TiledDecodeError(ValueError):
    """Raised when a tiled decode contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class TileWindow:
    """One core output tile and its expanded decoder input window."""

    index: int
    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int
    expanded_y0: int
    expanded_y1: int
    expanded_x0: int
    expanded_x1: int

    @property
    def core_height(self) -> int:
        return self.core_y1 - self.core_y0

    @property
    def core_width(self) -> int:
        return self.core_x1 - self.core_x0

    @property
    def expanded_height(self) -> int:
        return self.expanded_y1 - self.expanded_y0

    @property
    def expanded_width(self) -> int:
        return self.expanded_x1 - self.expanded_x0

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "core_y0": self.core_y0,
            "core_y1": self.core_y1,
            "core_x0": self.core_x0,
            "core_x1": self.core_x1,
            "expanded_y0": self.expanded_y0,
            "expanded_y1": self.expanded_y1,
            "expanded_x0": self.expanded_x0,
            "expanded_x1": self.expanded_x1,
        }


@dataclass(frozen=True, slots=True)
class TiledDecodeResult:
    """Decoded output plus the coverage/halo evidence for one invocation."""

    output: Any
    windows: tuple[TileWindow, ...]
    output_scale: int
    telemetry: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "telemetry", dict(self.telemetry))


def plan_tiles(
    height: int,
    width: int,
    *,
    tile_height: int,
    tile_width: int,
    halo: int = 0,
) -> tuple[TileWindow, ...]:
    """Plan non-overlapping core tiles with bounded expanded input windows."""

    values = {
        "height": height,
        "width": width,
        "tile_height": tile_height,
        "tile_width": tile_width,
    }
    if any(isinstance(value, bool) or int(value) <= 0 for value in values.values()):
        raise TiledDecodeError("geometry and tile sizes must be positive integers")
    if isinstance(halo, bool) or int(halo) < 0:
        raise TiledDecodeError("halo must be a non-negative integer")
    height, width, tile_height, tile_width, halo = map(
        int, (height, width, tile_height, tile_width, halo)
    )
    windows: list[TileWindow] = []
    index = 0
    for y0 in range(0, height, tile_height):
        for x0 in range(0, width, tile_width):
            y1 = min(height, y0 + tile_height)
            x1 = min(width, x0 + tile_width)
            windows.append(
                TileWindow(
                    index=index,
                    core_y0=y0,
                    core_y1=y1,
                    core_x0=x0,
                    core_x1=x1,
                    expanded_y0=max(0, y0 - halo),
                    expanded_y1=min(height, y1 + halo),
                    expanded_x0=max(0, x0 - halo),
                    expanded_x1=min(width, x1 + halo),
                )
            )
            index += 1
    return tuple(windows)


def _decoded_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if not value:
            raise TiledDecodeError("decoder returned an empty sequence")
        return value[0]
    tensor = getattr(value, "sample", None)
    if tensor is not None:
        return tensor
    tensor = getattr(value, "images", None)
    if tensor is not None:
        return tensor
    return value


def _call_decoder(decoder: Callable[..., Any], tile: Any, kwargs: Mapping[str, Any]) -> Any:
    value = _decoded_tensor(decoder(tile, **dict(kwargs)))
    if getattr(value, "ndim", None) != 4:
        raise TiledDecodeError("decoder must return a rank-4 tensor-like value")
    return value


def _output_shape(
    tile_output: Any, *, batch: int, scale: int | None = None
) -> tuple[int, int, int, int, int]:
    shape = getattr(tile_output, "shape", None)
    if shape is None or len(shape) != 4 or int(shape[0]) != batch:
        raise TiledDecodeError("decoder output must have shape (batch, channels, height, width)")
    tile_height = int(shape[-2])
    tile_width = int(shape[-1])
    if scale is None:
        raise TiledDecodeError("decoder scale cannot be inferred before a tile window is available")
    if tile_height % scale or tile_width % scale:
        raise TiledDecodeError("decoder output is not an integer scale of the input tile")
    return int(shape[0]), int(shape[1]), tile_height, tile_width, scale


def _overlap(
    first_y0: int,
    first_y1: int,
    first_x0: int,
    first_x1: int,
    second_y0: int,
    second_y1: int,
    second_x0: int,
    second_x1: int,
) -> bool:
    return max(first_y0, second_y0) < min(first_y1, second_y1) and max(first_x0, second_x0) < min(
        first_x1, second_x1
    )


def dirty_window_indices(
    changed_mask: Any,
    windows: Iterable[TileWindow],
    *,
    halo: int,
) -> tuple[int, ...]:
    """Select core tiles whose output can be affected by a changed latent region."""

    if getattr(changed_mask, "ndim", None) != 2:
        raise TiledDecodeError("changed_mask must have shape (height, width)")
    changed = changed_mask.bool()
    coordinates = changed.nonzero(as_tuple=False)
    if int(coordinates.shape[0]) == 0:
        return ()
    y0 = max(0, int(coordinates[:, 0].min().item()) - int(halo))
    y1 = min(int(changed.shape[0]), int(coordinates[:, 0].max().item()) + 1 + int(halo))
    x0 = max(0, int(coordinates[:, 1].min().item()) - int(halo))
    x1 = min(int(changed.shape[1]), int(coordinates[:, 1].max().item()) + 1 + int(halo))
    return tuple(
        window.index
        for window in windows
        if _overlap(
            window.core_y0,
            window.core_y1,
            window.core_x0,
            window.core_x1,
            y0,
            y1,
            x0,
            x1,
        )
    )


def decode_tiled(
    decoder: Callable[..., Any],
    latents: Any,
    *,
    tile_height: int,
    tile_width: int,
    halo: int = 0,
    output_scale: int | None = None,
    decoder_kwargs: Mapping[str, Any] | None = None,
    windows: Iterable[TileWindow] | None = None,
) -> TiledDecodeResult:
    """Decode all tiles and stitch only their non-overlapping cores."""

    if getattr(latents, "ndim", None) != 4:
        raise TiledDecodeError("latents must have shape (batch, channels, height, width)")
    batch, _, height, width = (int(value) for value in latents.shape)
    planned = tuple(
        windows
        or plan_tiles(height, width, tile_height=tile_height, tile_width=tile_width, halo=halo)
    )
    if not planned:
        raise TiledDecodeError("tile plan cannot be empty")
    kwargs = dict(decoder_kwargs or {})
    first_window = planned[0]
    first_output = _call_decoder(
        decoder,
        latents[
            :,
            :,
            first_window.expanded_y0 : first_window.expanded_y1,
            first_window.expanded_x0 : first_window.expanded_x1,
        ],
        kwargs,
    )
    inferred_scale = int(
        output_scale or (int(first_output.shape[-1]) // first_window.expanded_width)
    )
    if inferred_scale <= 0:
        raise TiledDecodeError("output_scale must be positive")
    _, channels, _, _, inferred_scale = _output_shape(
        first_output,
        batch=batch,
        scale=inferred_scale,
    )
    output = latents.new_zeros((batch, channels, height * inferred_scale, width * inferred_scale))
    decoded_area = 0
    for window, tile_output in zip(
        planned,
        (first_output,)
        + tuple(
            _call_decoder(
                decoder,
                latents[
                    :, :, item.expanded_y0 : item.expanded_y1, item.expanded_x0 : item.expanded_x1
                ],
                kwargs,
            )
            for item in planned[1:]
        ),
        strict=True,
    ):
        _output_shape(tile_output, batch=batch, scale=inferred_scale)
        crop_y0 = (window.core_y0 - window.expanded_y0) * inferred_scale
        crop_y1 = crop_y0 + window.core_height * inferred_scale
        crop_x0 = (window.core_x0 - window.expanded_x0) * inferred_scale
        crop_x1 = crop_x0 + window.core_width * inferred_scale
        output[
            :,
            :,
            window.core_y0 * inferred_scale : window.core_y1 * inferred_scale,
            window.core_x0 * inferred_scale : window.core_x1 * inferred_scale,
        ] = tile_output[:, :, crop_y0:crop_y1, crop_x0:crop_x1]
        decoded_area += window.expanded_height * window.expanded_width
    core_area = height * width
    return TiledDecodeResult(
        output=output,
        windows=planned,
        output_scale=inferred_scale,
        telemetry={
            "schema": "mrun-tiled-decode-telemetry-v1",
            "tile_count": len(planned),
            "core_pixels": core_area,
            "expanded_pixels": decoded_area,
            "halo_overhead": decoded_area / core_area,
            "output_scale": inferred_scale,
            "output_shape": [int(value) for value in output.shape],
        },
    )


def decode_dirty_tiled(
    decoder: Callable[..., Any],
    latents: Any,
    previous_output: Any,
    changed_mask: Any,
    *,
    tile_height: int,
    tile_width: int,
    halo: int = 0,
    output_scale: int | None = None,
    decoder_kwargs: Mapping[str, Any] | None = None,
) -> TiledDecodeResult:
    """Decode only affected core tiles and preserve the prior output elsewhere."""

    if getattr(previous_output, "ndim", None) != 4:
        raise TiledDecodeError("previous_output must have shape (batch, channels, height, width)")
    height, width = int(latents.shape[-2]), int(latents.shape[-1])
    planned = plan_tiles(height, width, tile_height=tile_height, tile_width=tile_width, halo=halo)
    selected = dirty_window_indices(changed_mask, planned, halo=halo)
    if not selected:
        return TiledDecodeResult(
            output=previous_output.clone(),
            windows=(),
            output_scale=int(output_scale or (previous_output.shape[-1] // width)),
            telemetry={
                "schema": "mrun-tiled-decode-telemetry-v1",
                "tile_count": 0,
                "dirty_tile_count": 0,
                "preserved": True,
            },
        )
    selected_windows = tuple(planned[index] for index in selected)
    decoded = decode_tiled(
        decoder,
        latents,
        tile_height=tile_height,
        tile_width=tile_width,
        halo=halo,
        output_scale=output_scale,
        decoder_kwargs=decoder_kwargs,
        windows=selected_windows,
    )
    output = previous_output.clone()
    for window in selected_windows:
        output[
            :,
            :,
            window.core_y0 * decoded.output_scale : window.core_y1 * decoded.output_scale,
            window.core_x0 * decoded.output_scale : window.core_x1 * decoded.output_scale,
        ] = decoded.output[
            :,
            :,
            window.core_y0 * decoded.output_scale : window.core_y1 * decoded.output_scale,
            window.core_x0 * decoded.output_scale : window.core_x1 * decoded.output_scale,
        ]
    telemetry = dict(decoded.telemetry)
    telemetry.update(
        {"tile_count": len(planned), "dirty_tile_count": len(selected_windows), "preserved": False}
    )
    return TiledDecodeResult(
        output=output,
        windows=selected_windows,
        output_scale=decoded.output_scale,
        telemetry=telemetry,
    )


__all__ = [
    "TileWindow",
    "TiledDecodeError",
    "TiledDecodeResult",
    "decode_dirty_tiled",
    "decode_tiled",
    "dirty_window_indices",
    "plan_tiles",
]
