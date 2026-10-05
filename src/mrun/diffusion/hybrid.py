"""Explicit adapters for mixing FLUX-side components with the native engine."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .phase import PromptEmbeds


@dataclass(frozen=True, slots=True)
class LatentPageLayout:
    """Exact NCHW latent geometry for a page-token bridge."""

    page_rows: int
    page_cols: int
    channels: int
    tile_height: int
    tile_width: int

    def __post_init__(self) -> None:
        for name in (
            "page_rows",
            "page_cols",
            "channels",
            "tile_height",
            "tile_width",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def height(self) -> int:
        return self.page_rows * self.tile_height

    @property
    def width(self) -> int:
        return self.page_cols * self.tile_width

    @property
    def page_count(self) -> int:
        return self.page_rows * self.page_cols

    @property
    def page_dim(self) -> int:
        return self.channels * self.tile_height * self.tile_width


class FluxLatentPageAdapter:
    """Convert explicit FLUX-style NCHW latents to row-major page tokens."""

    def __init__(self, layout: LatentPageLayout) -> None:
        self.layout = layout

    def to_pages(self, latents: Tensor) -> Tensor:
        if latents.ndim != 4:
            raise ValueError("FLUX latent input must have shape (batch, channels, height, width)")
        batch, channels, height, width = latents.shape
        expected = (
            self.layout.channels,
            self.layout.height,
            self.layout.width,
        )
        if (channels, height, width) != expected:
            raise ValueError(
                f"latent geometry {(channels, height, width)} does not match {expected}"
            )
        return latents.reshape(
            batch,
            channels,
            self.layout.page_rows,
            self.layout.tile_height,
            self.layout.page_cols,
            self.layout.tile_width,
        ).permute(0, 2, 4, 1, 3, 5).reshape(
            batch,
            self.layout.page_count,
            self.layout.page_dim,
        )

    def from_pages(self, pages: Tensor) -> Tensor:
        if pages.ndim != 3:
            raise ValueError("page tokens must have shape (batch, page_count, page_dim)")
        batch, page_count, page_dim = pages.shape
        expected = (self.layout.page_count, self.layout.page_dim)
        if (page_count, page_dim) != expected:
            raise ValueError(
                f"page geometry {(page_count, page_dim)} does not match {expected}"
            )
        return pages.reshape(
            batch,
            self.layout.page_rows,
            self.layout.page_cols,
            self.layout.channels,
            self.layout.tile_height,
            self.layout.tile_width,
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            batch,
            self.layout.channels,
            self.layout.height,
            self.layout.width,
        )


class TrainableLatentPageBridge(nn.Module):
    """Trainable page-space bridge around an exact FLUX latent layout.

    ``FluxLatentPageAdapter`` only changes layout.  This module adds the
    learned channel/width transform needed before a native page denoiser can
    consume the representation, plus the inverse transform needed to return
    to the original FLUX VAE.  It is an adapter with a trainable contract, not
    evidence that either model's existing weights are interchangeable.
    """

    def __init__(
        self,
        layout: LatentPageLayout,
        native_page_dim: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if native_page_dim <= 0:
            raise ValueError("native_page_dim must be positive")
        hidden = int(hidden_dim or max(layout.page_dim, native_page_dim))
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.layout = layout
        self.native_page_dim = int(native_page_dim)
        self.adapter = FluxLatentPageAdapter(layout)
        self.flux_to_native = nn.Sequential(
            nn.Linear(layout.page_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.native_page_dim),
        )
        self.native_to_flux = nn.Sequential(
            nn.Linear(self.native_page_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, layout.page_dim),
        )

    def to_native_pages(self, latents: Tensor) -> Tensor:
        return self.flux_to_native(self.adapter.to_pages(latents).float())

    def to_flux_latents(self, pages: Tensor) -> Tensor:
        if pages.ndim != 3 or tuple(pages.shape[1:]) != (
            self.layout.page_count,
            self.native_page_dim,
        ):
            raise ValueError("native pages do not match the bridge geometry")
        return self.adapter.from_pages(self.native_to_flux(pages.float()))

    def forward(self, latents: Tensor) -> Tensor:
        return self.to_native_pages(latents)

    def roundtrip_loss(self, latents: Tensor) -> Tensor:
        reconstructed = self.to_flux_latents(self.to_native_pages(latents))
        return (reconstructed - latents.float()).square().mean()

    def contract(self) -> dict[str, Any]:
        return {
            "schema": "mrun-trainable-flux-latent-page-bridge-v1",
            "flux_page_dim": self.layout.page_dim,
            "native_page_dim": self.native_page_dim,
            "page_count": self.layout.page_count,
            "geometry": [self.layout.channels, self.layout.height, self.layout.width],
            "trainable": True,
            "trained": False,
            "vae_compatibility": "requires reconstruction/distillation training",
        }


class FluxConditioningBridge(nn.Module):
    """Trainable bridge from FLUX sequence embeddings to native conditions."""

    def __init__(self, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0 or condition_dim <= 0:
            raise ValueError("embedding_dim and condition_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.condition_dim = int(condition_dim)
        self.projection = nn.Linear(self.embedding_dim, self.condition_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(dim=1)
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"conditioning tensor must end in {self.embedding_dim} and be rank 2/3"
            )
        return self.projection(embeddings.float())

    def adapt(self, embeds: PromptEmbeds) -> PromptEmbeds:
        source = embeds.tensors.get("prompt_embeds")
        if source is None:
            raise ValueError("FLUX embeddings must expose prompt_embeds")
        condition = self(source)
        bridge_fingerprint = hashlib.sha256(
            f"{self.embedding_dim}:{self.condition_dim}".encode("ascii")
        ).hexdigest()[:16]
        return PromptEmbeds(
            key=f"{embeds.key}:native-{bridge_fingerprint}",
            tensors={"prompt_embeds": condition},
            meta={
                "source_key": embeds.key,
                "bridge": "flux-conditioning-to-native-v1",
                "embedding_dim": self.embedding_dim,
                "condition_dim": self.condition_dim,
            },
        )


class HybridNativeBackend:
    """Native denoiser using a FLUX backend for conditioning only.

    The output remains native until a separately trained latent/VAE decoder is
    supplied.  This explicit boundary prevents an untrained native tensor
    field from being mislabeled as a valid FLUX image latent.
    """

    def __init__(
        self,
        native_backend: Any,
        flux_backend: Any,
        conditioning_bridge: FluxConditioningBridge,
        *,
        output_adapter: Callable[[Any], Any] | None = None,
    ) -> None:
        for name in ("encode",):
            if not callable(getattr(flux_backend, name, None)):
                raise TypeError(f"FLUX backend must expose {name}()")
        for name in ("generate", "generate_batch"):
            if not callable(getattr(native_backend, name, None)):
                raise TypeError(f"native backend must expose {name}()")
        self.native_backend = native_backend
        self.flux_backend = flux_backend
        self.conditioning_bridge = conditioning_bridge
        self.output_adapter = output_adapter

    def encode(self, prompt: str, **params: Any) -> PromptEmbeds:
        source = self.flux_backend.encode(prompt, **params)
        return self.conditioning_bridge.adapt(source)

    def _adapt_output(self, output: Any) -> Any:
        return self.output_adapter(output) if self.output_adapter is not None else output

    def generate(self, embeds: PromptEmbeds, **kwargs: Any) -> Any:
        return self._adapt_output(self.native_backend.generate(embeds, **kwargs))

    def generate_batch(self, embeds: Any, **kwargs: Any) -> Any:
        result = self.native_backend.generate_batch(embeds, **kwargs)
        if self.output_adapter is None:
            return result
        # PhaseBatchResult is frozen; preserve its logical-row metadata while
        # replacing only the adapted payload.
        from dataclasses import replace

        return replace(result, output=self.output_adapter(result.output))

    def component_contract(self) -> dict[str, Any]:
        return {
            "conditioning": "FLUX encoder through trained bridge",
            "denoiser": "native hierarchical/routed backend",
            "latent_output": "native until an explicit latent adapter is supplied",
            "output_adapter": self.output_adapter is not None,
        }


__all__ = [
    "FluxConditioningBridge",
    "FluxLatentPageAdapter",
    "HybridNativeBackend",
    "LatentPageLayout",
    "TrainableLatentPageBridge",
]
