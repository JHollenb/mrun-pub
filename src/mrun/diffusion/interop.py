"""Explicit component interop between Diffusers/FLUX and the program VM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cache import TrajectoryCache
from .phase import PhasePipeline
from .program import DiffusionProgram


@dataclass(frozen=True, slots=True)
class FluxComponentReport:
    """Inspectable component boundary before linking a FLUX program."""

    pipeline_class: str
    encoder_components: tuple[str, ...]
    denoiser_component: str | None
    vae_component: str | None
    scheduler_component: str | None
    phase_contract_available: bool
    issues: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return self.phase_contract_available and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_class": self.pipeline_class,
            "encoder_components": list(self.encoder_components),
            "denoiser_component": self.denoiser_component,
            "vae_component": self.vae_component,
            "scheduler_component": self.scheduler_component,
            "phase_contract_available": self.phase_contract_available,
            "compatible": self.compatible,
            "issues": list(self.issues),
        }


def inspect_flux_pipeline(pipeline: Any) -> FluxComponentReport:
    """Inspect component names without moving any tensors or loading weights."""

    pipeline_class = type(pipeline).__name__
    encoder_names = tuple(
        name
        for name in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2")
        if getattr(pipeline, name, None) is not None
    )
    denoiser = next(
        (name for name in ("transformer", "unet") if getattr(pipeline, name, None) is not None),
        None,
    )
    vae = "vae" if getattr(pipeline, "vae", None) is not None else None
    scheduler = "scheduler" if getattr(pipeline, "scheduler", None) is not None else None
    issues: list[str] = []
    try:
        from .phase import EMBED_CONTRACTS

        phase_contract_available = pipeline_class in EMBED_CONTRACTS
    except Exception:  # pragma: no cover - defensive import boundary
        phase_contract_available = False
    if not phase_contract_available:
        issues.append("no registered embedding feed contract for this pipeline class")
    if denoiser is None:
        issues.append("no transformer/unet denoiser component")
    if vae is None:
        issues.append("no VAE component")
    if scheduler is None:
        issues.append("no scheduler component")
    return FluxComponentReport(
        pipeline_class=pipeline_class,
        encoder_components=encoder_names,
        denoiser_component=denoiser,
        vae_component=vae,
        scheduler_component=scheduler,
        phase_contract_available=phase_contract_available,
        issues=tuple(issues),
    )


def wrap_flux_pipeline(
    pipeline: Any,
    *,
    base_fingerprint: str,
    device: str = "cuda",
    cache_dir: str | None = None,
    device_embed_cache: bool = False,
    trajectory_cache: TrajectoryCache | None = None,
) -> tuple[DiffusionProgram, FluxComponentReport]:
    """Use a real FLUX/Diffusers pipeline as a program backend.

    This is intentionally a strict adapter: ``PhasePipeline.wrap`` validates
    the embed ABI and refuses an unknown pipeline rather than guessing how to
    feed cached embeddings back into ``__call__``.
    """

    already_wrapped = isinstance(pipeline, PhasePipeline)
    source_pipeline = pipeline._pipeline if already_wrapped else pipeline
    report = inspect_flux_pipeline(source_pipeline)
    if not report.compatible:
        raise ValueError("FLUX pipeline is not compatible: " + "; ".join(report.issues))
    phase = (
        pipeline
        if already_wrapped
        else PhasePipeline.wrap(
            pipeline,
            device=device,
            cache_dir=cache_dir,
            device_embed_cache=device_embed_cache,
            trajectory_cache=trajectory_cache,
            model_identity=base_fingerprint,
        )
    )
    program = DiffusionProgram.from_backend(
        phase,
        base_fingerprint=base_fingerprint,
        component_graph={
            "pipeline_class": report.pipeline_class,
            "execution_wrapper": type(phase).__name__,
            "wrapped_pipeline_class": report.pipeline_class,
            "backend": "PhasePipeline",
            "components": report.to_dict(),
        },
        ports={
            "input": ["prompt", "references", "seed", "resolution"],
            "output": ["image"],
            "state": ["conditioning", "latent", "scheduler_cursor"],
        },
        numerical_contract={
            "mode": "backend-authority",
            "step_granularity": "atomic_backend_schedule",
            "component_interchange": "adapter-validated",
        },
    )
    return program, report


__all__ = ["FluxComponentReport", "inspect_flux_pipeline", "wrap_flux_pipeline"]
