"""mrun.diffusion — two-phase CUDA execution for Diffusers pipelines.

Motivation (measured 2026-07-31, beast RTX 4080 16GB): FLUX.2 Klein-4B under
``enable_model_cpu_offload`` spends its wall clock shuttling the Qwen3-4B text
encoder (~8 GB bf16) and the 3.876B-param denoiser (~7.8 GB) across PCIe on
every call — 56 s/image-pair at 12% GPU utilization. The full bf16 pipeline
(~15.9 GB) does not fit the card resident, but no single *phase* needs both
halves at once. The split is temporal, not spatial:

  phase A (encode):  text encoder resident on CUDA, denoiser+VAE on CPU —
                     encode every prompt once, cache the embeddings.
  phase B (denoise): text encoder off the card, denoiser+VAE resident —
                     run every generation from the cached embeddings.

``PhasePipeline`` wraps an already-loaded (CPU-resident) Diffusers pipeline
and manages those phases, with ``mrun.guard`` RSS checks plus a VRAM ceiling
check at every phase boundary. Encoding is strictly serial per prompt (batch
padding changes the embeddings numerically); generation calls the pipeline's
own ``__call__`` through its ``prompt_embeds`` path, so a phase-cuda image is
expected to be *bitwise identical* to the offload path on the same device.

Importing this package needs neither torch nor diffusers — all heavy imports
are function-local (the scheduler/agent venvs carry neither).
"""

from .cache import EmbedCache, TrajectoryCache, TrajectoryCacheEntry
from .interop import FluxComponentReport, inspect_flux_pipeline, wrap_flux_pipeline
from .io import (
    COMPONENT_FRAME_SCHEMA,
    COMPONENT_IO_SCHEMA,
    IO_DIRECTIONS,
    IO_MUTABILITY,
    IO_STREAMS,
    ComponentFrame,
    ComponentIOError,
    ComponentIOSpec,
    PortBinding,
    PortContract,
    component_io_manifest,
    component_io_specs,
    make_component_frame,
    payload_abi,
    payload_fingerprint,
    validate_component_io_manifest,
)
from .nonflux import (
    NON_FLUX_CHECKPOINT_SCHEMA,
    NON_FLUX_PIPELINES,
    NON_FLUX_STATE_SCHEMA,
    DiffusionTrajectoryCheckpoint,
    DiffusionTrajectoryState,
    NativeDiffusionTrajectoryCheckpoint,
    NonFluxTrajectoryCheckpoint,
)
from .phase import (
    PhaseBatchResult,
    PhaseError,
    PhasePipeline,
    PhaseTelemetry,
    PromptEmbeds,
    TrajectoryCheckpoint,
)
from .program import (
    PROGRAM_CHECKPOINT_SCHEMA,
    PROGRAM_MANIFEST_SCHEMA,
    PROGRAM_STATE_SCHEMA,
    DenoiseBinding,
    DiffusionProgram,
    ProgramABIError,
    ProgramCapabilityError,
    ProgramCheckpoint,
    ProgramError,
    ProgramExtension,
    ProgramLink,
    ProgramManifest,
    ProgramRuntime,
    ProgramSession,
    ProgramState,
    ProgramStateError,
    ProgramStepResult,
)
from .resources import (
    DiffusionResourceEstimate,
    FluxResourceEstimate,
    estimate_diffusion_resources,
    estimate_flux_resources,
)
from .scene import (
    FIELD_KINDS,
    SCENE_INTERVENTION_SCHEMA,
    SCENE_STATE_SCHEMA,
    SceneField,
    SceneIntervention,
    SceneInterventionRecord,
    SceneSlot,
    SceneState,
    SceneStateDebugger,
    SceneStateError,
)
from .scheduler import BatchDispatch, ContinuousBatchScheduler
from .service import ProgramService, ProgramServiceError, create_fastapi_app
from .tiled import (
    TiledDecodeError,
    TiledDecodeResult,
    TileWindow,
    decode_dirty_tiled,
    decode_tiled,
    dirty_window_indices,
    plan_tiles,
)
from .train import PrecomputedCache, compile_dit

_NATIVE_EXPORTS = {
    "CompiledRoutedExecutor",
    "DensePageFlow",
    "NativeFlowBackend",
    "NativeFlowConfig",
    "NativeSceneBatch",
    "NativeTensorCache",
    "PageDispatchPlan",
    "PageStateCache",
    "RoutedPageFlow",
    "learned_route_mask",
    "route_supervision_loss",
    "sample_adaptive_flow",
    "sample_cached_edit_flow",
}
_DISTILL_EXPORTS = {
    "DistillationConfig",
    "RealImageDistillationBatch",
    "RealImageDistillationConfig",
    "train_distilled_student",
    "train_real_image_student",
}
_HYBRID_EXPORTS = {
    "FluxConditioningBridge",
    "FluxLatentPageAdapter",
    "HybridNativeBackend",
    "LatentPageLayout",
    "TrainableLatentPageBridge",
}


def __getattr__(name: str):
    """Load the torch-backed research organism only when one of its names is used."""

    if name in _NATIVE_EXPORTS:
        from . import native

        return getattr(native, name)
    if name in _DISTILL_EXPORTS:
        from . import distill

        return getattr(distill, name)
    if name in _HYBRID_EXPORTS:
        from . import hybrid

        return getattr(hybrid, name)
    raise AttributeError(f"module 'mrun.diffusion' has no attribute {name!r}")


__all__ = [
    "EmbedCache",
    "TrajectoryCache",
    "TrajectoryCacheEntry",
    "CompiledRoutedExecutor",
    "DensePageFlow",
    "BatchDispatch",
    "ContinuousBatchScheduler",
    "FIELD_KINDS",
    "COMPONENT_FRAME_SCHEMA",
    "COMPONENT_IO_SCHEMA",
    "ComponentFrame",
    "ComponentIOError",
    "ComponentIOSpec",
    "DenoiseBinding",
    "DistillationConfig",
    "RealImageDistillationBatch",
    "RealImageDistillationConfig",
    "DiffusionProgram",
    "DiffusionResourceEstimate",
    "FluxResourceEstimate",
    "FluxComponentReport",
    "FluxConditioningBridge",
    "FluxLatentPageAdapter",
    "HybridNativeBackend",
    "IO_DIRECTIONS",
    "IO_MUTABILITY",
    "IO_STREAMS",
    "LatentPageLayout",
    "TrainableLatentPageBridge",
    "NativeFlowBackend",
    "NativeFlowConfig",
    "NativeSceneBatch",
    "NativeTensorCache",
    "PageDispatchPlan",
    "PageStateCache",
    "PhaseError",
    "PhaseBatchResult",
    "PhasePipeline",
    "PhaseTelemetry",
    "PortBinding",
    "PortContract",
    "TrajectoryCheckpoint",
    "DiffusionTrajectoryCheckpoint",
    "DiffusionTrajectoryState",
    "NON_FLUX_CHECKPOINT_SCHEMA",
    "NON_FLUX_PIPELINES",
    "NON_FLUX_STATE_SCHEMA",
    "NativeDiffusionTrajectoryCheckpoint",
    "NonFluxTrajectoryCheckpoint",
    "PROGRAM_CHECKPOINT_SCHEMA",
    "PROGRAM_MANIFEST_SCHEMA",
    "PROGRAM_STATE_SCHEMA",
    "PrecomputedCache",
    "ProgramABIError",
    "ProgramCapabilityError",
    "ProgramCheckpoint",
    "ProgramError",
    "ProgramExtension",
    "ProgramLink",
    "ProgramManifest",
    "ProgramRuntime",
    "ProgramService",
    "ProgramServiceError",
    "ProgramSession",
    "ProgramState",
    "ProgramStateError",
    "ProgramStepResult",
    "PromptEmbeds",
    "RoutedPageFlow",
    "SCENE_INTERVENTION_SCHEMA",
    "SCENE_STATE_SCHEMA",
    "SceneField",
    "SceneIntervention",
    "SceneInterventionRecord",
    "SceneSlot",
    "SceneState",
    "SceneStateDebugger",
    "SceneStateError",
    "TileWindow",
    "TiledDecodeError",
    "TiledDecodeResult",
    "learned_route_mask",
    "route_supervision_loss",
    "sample_adaptive_flow",
    "sample_cached_edit_flow",
    "train_distilled_student",
    "train_real_image_student",
    "compile_dit",
    "create_fastapi_app",
    "decode_dirty_tiled",
    "decode_tiled",
    "dirty_window_indices",
    "component_io_manifest",
    "component_io_specs",
    "estimate_flux_resources",
    "estimate_diffusion_resources",
    "inspect_flux_pipeline",
    "make_component_frame",
    "payload_abi",
    "payload_fingerprint",
    "plan_tiles",
    "validate_component_io_manifest",
    "wrap_flux_pipeline",
]
