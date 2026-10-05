"""Two-phase (encode / denoise) execution of a Diffusers pipeline on one CUDA card.

See the package docstring for the why. Contract summary:

- ``PhasePipeline.wrap(pipeline, device="cuda")`` wraps an already-loaded,
  CPU-resident pipeline. Wrapping moves nothing; the first encode or generate
  call performs the first phase swap.
- ``encode(prompt)`` is SERIAL per prompt by design. Batching pads the token
  sequence and changes the embeddings numerically. Most pipeline classes use
  their native ``encode_prompt`` implementation; FLUX.1 uses the explicit
  masked dual-encoder contract shared with Saturn-main so its T5 padding mask
  cannot drift from the established reference path.
- ``generate(...)`` routes through the wrapped pipeline's ``__call__`` via its
  ``prompt_embeds`` parameter. Pipelines without an embeds path are rejected at
  wrap time (fail closed), never silently re-encoded per call.
- With ``device_embed_cache=True``, ``generate(...)`` keeps one validated
  device-side entry for repeated seeds/arms. The entry is invalidated before
  returning to encode, on source object/tensor changes, and on close, so its
  device footprint is bounded.
- Phase boundaries run ``mrun.guard.check_rss`` and, on CUDA, a VRAM ceiling
  check against ``VRAM_LIMIT_MB`` (the mrun agent exports the job's declared
  reservation under that name).
- Calling the wrapper like a pipeline (``wrapper(prompt=..., generator=...)``)
  is duck-compatible with ``image_atlas.runtime.generate_one``: on a cache miss
  it swaps to the encode phase, encodes, swaps back, and generates. That is
  correct but pays two extra transfers per new prompt — callers with a panel of
  prompts should call ``precompute(prompts)`` first so the swap happens exactly
  twice per job.

Lazy imports: this module must import cleanly with neither torch nor diffusers
installed (scheduler/agent venvs). Only stdlib + ``mrun.guard`` at module scope.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from .. import guard
from .cache import TrajectoryCache
from .nonflux import (
    NON_FLUX_PIPELINES,
    DiffusionTrajectoryCheckpoint,
    advance_nonflux_checkpoint,
    capture_nonflux_checkpoint,
    resume_nonflux_checkpoint,
    resume_nonflux_checkpoint_batch,
)


class PhaseError(RuntimeError):
    """A phase-cuda contract violation. Always fail closed, never degrade silently."""


@dataclass(frozen=True)
class EmbedContract:
    """How one pipeline class exposes its prompt-embedding path.

    ``feed_keys`` name the leading positions of the ``encode_prompt(...)`` return
    tuple that must be fed back into ``__call__``; trailing positions (for the
    Flux2 family, ``text_ids``) are derived deterministically from the embeds by
    the pipeline itself and are deliberately not cached.
    """

    encode_method: str = "encode_prompt"
    feed_keys: tuple[str, ...] = ("prompt_embeds",)
    # Optional positions in encode_prompt's return tuple. SDXL returns
    # (prompt, negative, pooled, negative_pooled), so its positive conditioner
    # is non-contiguous.
    feed_indices: tuple[int, ...] | None = None
    # __call__ kwargs that change the embeddings and therefore key the cache.
    encode_param_names: tuple[str, ...] = (
        "max_sequence_length",
        "text_encoder_out_layers",
    )


# Verified against the installed diffusers on the CUDA host (0.39.0):
# Flux2KleinPipeline.encode_prompt(prompt, device, num_images_per_prompt,
# prompt_embeds, max_sequence_length, text_encoder_out_layers) returns
# (prompt_embeds, text_ids); __call__ accepts prompt_embeds and recomputes
# text_ids from its shape. Unknown classes are rejected at wrap time.
EMBED_CONTRACTS: dict[str, EmbedContract] = {
    # FLUX.1's dual text stack has one sequence embedding and one pooled CLIP
    # projection.  Both are semantic conditioner outputs; dropping the pooled
    # projection would silently change the denoiser call when cached embeds are
    # fed back in.
    "FluxPipeline": EmbedContract(
        feed_keys=("prompt_embeds", "pooled_prompt_embeds"),
        encode_param_names=("max_sequence_length", "lora_scale"),
    ),
    "Flux2KleinPipeline": EmbedContract(),
    "Flux2KleinKVPipeline": EmbedContract(
        encode_param_names=("max_sequence_length", "text_encoder_out_layers")
    ),
    "Flux2Pipeline": EmbedContract(),
    # SDXL returns positive/negative sequence and pooled conditioners. Keep
    # all four so normal classifier-free guidance remains intact when the
    # phase wrapper feeds cached embeddings back into __call__.
    "StableDiffusionXLPipeline": EmbedContract(
        feed_keys=(
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ),
        feed_indices=(0, 1, 2, 3),
        encode_param_names=("lora_scale", "negative_prompt"),
    ),
    # Krea 2 returns the positive sequence embedding and its attention mask.
    # Turbo runs with guidance_scale=0, so there is no negative conditioner to
    # cache or feed back into the pipeline.
    "Krea2Pipeline": EmbedContract(
        feed_keys=("prompt_embeds", "prompt_embeds_mask"),
        encode_param_names=("max_sequence_length",),
    ),
    # Chroma returns prompt/text-id/mask tuples for both positive and negative
    # conditioning.  text_ids are derived from the embedding shapes by the
    # pipeline, so keep only the four tensors consumed by __call__.
    "ChromaPipeline": EmbedContract(
        feed_keys=(
            "prompt_embeds",
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_prompt_attention_mask",
        ),
        feed_indices=(0, 2, 3, 5),
        encode_param_names=("max_sequence_length", "lora_scale", "negative_prompt"),
    ),
}

# Diffusers constructs these image-to-image variants with ``from_pipe``. They
# share the text encoder and conditioner return layout of their text-to-image
# parents, so reuse the same fail-closed contracts instead of re-inferring the
# tuple shape in the worker.
EMBED_CONTRACTS.update(
    {
        "FluxImg2ImgPipeline": EMBED_CONTRACTS["FluxPipeline"],
        "StableDiffusionXLImg2ImgPipeline": EMBED_CONTRACTS["StableDiffusionXLPipeline"],
        "ChromaImg2ImgPipeline": EMBED_CONTRACTS["ChromaPipeline"],
    }
)

_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")
_DENOISER_ATTRS = ("transformer", "unet")
_VAE_ATTR = "vae"


@dataclass
class PromptEmbeds:
    """CPU-resident prompt embeddings with exact dtypes preserved."""

    key: str
    tensors: dict[str, Any]  # feed_key -> torch.Tensor (CPU)
    meta: dict[str, Any] = field(default_factory=dict)


def _clone_tensor_payload(value: Any) -> Any:
    """Clone a tensor payload without importing torch at module scope."""

    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        try:
            return value.detach().clone()
        except (RuntimeError, TypeError):
            pass
    return value


def _flux2_kv_cache_kind(value: Any) -> str | None:
    name = type(value).__name__
    if name in {"Flux2KVCache", "Flux2KVLayerCache"}:
        return name
    if name.startswith("Flux2KV"):
        raise PhaseError(f"unsupported native Flux2 KV cache class {name!r}")
    return None


def _clone_flux2_kv_layer_cache(value: Any, *, device: Any | None = None) -> Any:
    if not hasattr(value, "k_ref") or not hasattr(value, "v_ref"):
        raise PhaseError("Flux2KVLayerCache is missing k_ref/v_ref fields")
    clone = type(value)()
    for name in ("k_ref", "v_ref"):
        payload = getattr(value, name)
        if payload is None:
            setattr(clone, name, None)
        elif device is None:
            setattr(clone, name, _clone_tensor_payload(payload))
        else:
            moved = getattr(payload, "to", None)
            if not callable(moved):
                raise PhaseError(f"Flux2KVLayerCache.{name} is not a tensor")
            setattr(clone, name, moved(device).detach().clone())
    return clone


def _clone_flux2_kv_cache(value: Any, *, device: Any | None = None) -> Any:
    if not hasattr(value, "double_block_caches") or not hasattr(value, "single_block_caches"):
        raise PhaseError("Flux2KVCache is missing double/single block caches")
    double = list(value.double_block_caches)
    single = list(value.single_block_caches)
    clone = type(value)(len(double), len(single))
    clone.double_block_caches = [
        _clone_flux2_kv_layer_cache(item, device=device) for item in double
    ]
    clone.single_block_caches = [
        _clone_flux2_kv_layer_cache(item, device=device) for item in single
    ]
    clone.num_ref_tokens = int(getattr(value, "num_ref_tokens", 0))
    return clone


def _clone_checkpoint_payload(value: Any) -> Any:
    """Clone tensor trees used by native reference-KV trajectories.

    The ordinary checkpoint fields are tensors, but the native Klein-KV
    transformer returns a nested tuple/list of K/V tensors.  Treating that
    cache as an opaque Python object would let one replay mutate another
    branch's prefix.  Keep the helper local to the trajectory ABI so the
    historical scalar tensor fields retain their original behavior.
    """

    kind = _flux2_kv_cache_kind(value)
    if kind == "Flux2KVCache":
        return _clone_flux2_kv_cache(value)
    if kind == "Flux2KVLayerCache":
        return _clone_flux2_kv_layer_cache(value)
    if isinstance(value, Mapping):
        return type(value)((key, _clone_checkpoint_payload(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_payload(item) for item in value)
    if isinstance(value, list):
        return [_clone_checkpoint_payload(item) for item in value]
    return _clone_tensor_payload(value)


def _move_checkpoint_payload(value: Any, device: Any) -> Any:
    """Move a nested native-KV cache to the active phase device."""

    kind = _flux2_kv_cache_kind(value)
    if kind == "Flux2KVCache":
        return _clone_flux2_kv_cache(value, device=device)
    if kind == "Flux2KVLayerCache":
        return _clone_flux2_kv_layer_cache(value, device=device)
    if isinstance(value, Mapping):
        return type(value)((key, _move_checkpoint_payload(item, device)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_move_checkpoint_payload(item, device) for item in value)
    if isinstance(value, list):
        return [_move_checkpoint_payload(item, device) for item in value]
    to = getattr(value, "to", None)
    return to(device) if callable(to) else value


def _tensor_payload_fingerprint(value: Any) -> str:
    """Fingerprint tensor shape, dtype, and bytes for an in-process checkpoint."""

    import torch

    if not isinstance(value, torch.Tensor):
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in tensor.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_payload_fingerprint(value: Any) -> str:
    """Fingerprint a tensor tree without relying on object repr addresses."""

    kind = _flux2_kv_cache_kind(value)
    if kind == "Flux2KVLayerCache":
        payload = {
            "type": kind,
            "k_ref": _checkpoint_payload_fingerprint(getattr(value, "k_ref", None)),
            "v_ref": _checkpoint_payload_fingerprint(getattr(value, "v_ref", None)),
        }
    elif kind == "Flux2KVCache":
        payload = {
            "type": kind,
            "num_ref_tokens": int(getattr(value, "num_ref_tokens", 0)),
            "double": [
                _checkpoint_payload_fingerprint(item)
                for item in getattr(value, "double_block_caches", ())
            ],
            "single": [
                _checkpoint_payload_fingerprint(item)
                for item in getattr(value, "single_block_caches", ())
            ],
        }
    elif isinstance(value, Mapping):
        payload = {
            "mapping": [
                [str(key), _checkpoint_payload_fingerprint(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    elif isinstance(value, tuple):
        payload = {"tuple": [_checkpoint_payload_fingerprint(item) for item in value]}
    elif isinstance(value, list):
        payload = {"list": [_checkpoint_payload_fingerprint(item) for item in value]}
    elif value is None:
        payload = None
    else:
        payload = {"value": _tensor_payload_fingerprint(value)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    """Immutable packed-latent state at a denoising boundary.

    The checkpoint is deliberately an execution object rather than an image
    result.  It contains the state needed to replay the suffix of a supported
    pipeline without re-running its prefix.  Payloads are cloned on creation,
    so a branch cannot mutate the state used by another branch.
    """

    checkpoint_id: str
    pipeline_class: str
    step_index: int
    total_steps: int
    height: int
    width: int
    latents: Any
    latent_ids: Any
    timesteps: Any
    prompt_embeds: Any
    text_ids: Any
    # Reference images are first-class conditioning state. The generated
    # latents remain separate from these tokens so replay can append the
    # reference sequence at the denoiser boundary exactly as Flux2 does.
    reference_ids: tuple[str, ...] = ()
    reference_latents: Any | None = None
    reference_latent_ids: Any | None = None
    reference_token_count: int = 0
    guidance_scale: float = 1.0
    attention_kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "mrun-diffusion-trajectory-checkpoint-v1"
    # Appended after the original v1 fields to preserve the positional
    # TrajectoryCheckpoint ABI.  FLUX.1 uses this second conditioner stream;
    # FLUX.2 leaves it unset.
    pooled_prompt_embeds: Any | None = None
    # Native Flux2KleinKVPipeline reference attention state.  This is the
    # ref-first extract result produced at denoise step 0 and reused by every
    # later step.  It is optional so old non-KV checkpoints remain reopenable.
    reference_kv_cache: Any | None = None

    def __post_init__(self) -> None:
        if self.schema != "mrun-diffusion-trajectory-checkpoint-v1":
            raise PhaseError(f"unsupported trajectory checkpoint schema {self.schema!r}")
        if not self.checkpoint_id or not self.pipeline_class:
            raise PhaseError("trajectory checkpoint requires an id and pipeline class")
        if (
            isinstance(self.step_index, bool)
            or not 0 <= int(self.step_index) <= int(self.total_steps)
        ):
            raise PhaseError("trajectory checkpoint step_index is outside the schedule")
        if isinstance(self.total_steps, bool) or int(self.total_steps) <= 0:
            raise PhaseError("trajectory checkpoint total_steps must be positive")
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise PhaseError("trajectory checkpoint resolution must be positive")
        for name in ("latents", "latent_ids", "timesteps", "prompt_embeds", "text_ids"):
            value = getattr(self, name)
            if value is None:
                raise PhaseError(f"trajectory checkpoint is missing {name}")
            object.__setattr__(self, name, _clone_tensor_payload(value))
        if self.pooled_prompt_embeds is not None:
            object.__setattr__(
                self, "pooled_prompt_embeds", _clone_tensor_payload(self.pooled_prompt_embeds)
            )
        if self.reference_kv_cache is not None:
            object.__setattr__(
                self,
                "reference_kv_cache",
                _clone_checkpoint_payload(self.reference_kv_cache),
            )
        reference_token_count = int(self.reference_token_count)
        if reference_token_count < 0:
            raise PhaseError("trajectory checkpoint reference_token_count cannot be negative")
        if reference_token_count and (
            self.reference_latents is None or self.reference_latent_ids is None
        ):
            raise PhaseError(
                "reference_token_count requires reference_latents and reference_latent_ids"
            )
        if self.reference_latents is not None:
            object.__setattr__(
                self, "reference_latents", _clone_tensor_payload(self.reference_latents)
            )
        if self.reference_latent_ids is not None:
            object.__setattr__(
                self,
                "reference_latent_ids",
                _clone_tensor_payload(self.reference_latent_ids),
            )
        object.__setattr__(self, "reference_ids", tuple(str(value) for value in self.reference_ids))
        object.__setattr__(self, "reference_token_count", reference_token_count)
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "total_steps", int(self.total_steps))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "attention_kwargs", dict(self.attention_kwargs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "pipeline_class": self.pipeline_class,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "height": self.height,
            "width": self.width,
            "latents": _tensor_payload_fingerprint(self.latents),
            "latent_ids": _tensor_payload_fingerprint(self.latent_ids),
            "timesteps": _tensor_payload_fingerprint(self.timesteps),
            "prompt_embeds": _tensor_payload_fingerprint(self.prompt_embeds),
            "text_ids": _tensor_payload_fingerprint(self.text_ids),
            "reference_ids": self.reference_ids,
            "reference_latents": _tensor_payload_fingerprint(self.reference_latents),
            "reference_latent_ids": _tensor_payload_fingerprint(self.reference_latent_ids),
            "reference_token_count": self.reference_token_count,
            "guidance_scale": float(self.guidance_scale),
            "attention_kwargs": dict(self.attention_kwargs),
            "metadata": dict(self.metadata),
        }
        # This field did not exist in the original v1 checkpoint.  Omitting it
        # when absent preserves every historical FLUX.2 fingerprint.
        if self.pooled_prompt_embeds is not None:
            payload["pooled_prompt_embeds"] = _tensor_payload_fingerprint(
                self.pooled_prompt_embeds
            )
        if self.reference_kv_cache is not None:
            payload["reference_kv_cache"] = _checkpoint_payload_fingerprint(
                self.reference_kv_cache
            )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr).encode("utf-8")
        ).hexdigest()

    def metadata_only(self) -> dict[str, Any]:
        """Return transport-safe metadata without exposing tensor payloads."""

        return {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "fingerprint": self.fingerprint,
            "pipeline_class": self.pipeline_class,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "resolution": [self.height, self.width],
            "latents": {
                "shape": list(getattr(self.latents, "shape", ())),
                "dtype": str(getattr(self.latents, "dtype", "")),
                "device": str(getattr(self.latents, "device", "")),
            },
            "reference_ids": list(self.reference_ids),
            "reference_token_count": self.reference_token_count,
            "pooled_prompt_embeds": {
                "shape": list(getattr(self.pooled_prompt_embeds, "shape", ())),
                "dtype": str(getattr(self.pooled_prompt_embeds, "dtype", "")),
                "device": str(getattr(self.pooled_prompt_embeds, "device", "")),
            }
            if self.pooled_prompt_embeds is not None
            else None,
            "reference_latents": {
                "shape": list(getattr(self.reference_latents, "shape", ())),
                "dtype": str(getattr(self.reference_latents, "dtype", "")),
                "device": str(getattr(self.reference_latents, "device", "")),
            }
            if self.reference_latents is not None
            else None,
            "reference_kv_cache": {
                "present": self.reference_kv_cache is not None,
                "fingerprint": _checkpoint_payload_fingerprint(self.reference_kv_cache),
            }
            if self.reference_kv_cache is not None
            else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PhaseBatchResult:
    """One physical denoise call plus the logical rows it produced.

    ``PhasePipeline`` deliberately keeps prompt encoding serial: padding a
    prompt panel can change the embeddings.  Once embeddings have been
    independently encoded, compatible denoise branches can share the
    denoiser/VAE call.  This result keeps the logical row order explicit so a
    caller cannot accidentally compare a permuted batch as if it were a
    scalar run.
    """

    output: Any
    branch_ids: tuple[str, ...]
    batch_size: int
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branch_ids or len(self.branch_ids) != self.batch_size:
            raise ValueError("branch_ids must be non-empty and match batch_size")
        if len(set(self.branch_ids)) != len(self.branch_ids):
            raise ValueError("branch_ids must be unique")
        object.__setattr__(self, "telemetry", dict(self.telemetry))

    def row_outputs(self) -> tuple[Any, ...]:
        """Split common Diffusers output shapes while failing closed otherwise."""

        value = getattr(self.output, "images", self.output)
        if isinstance(value, (list, tuple)) and len(value) == self.batch_size:
            return tuple(value)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == self.batch_size:
            return tuple(value[index] for index in range(self.batch_size))
        if self.batch_size == 1:
            return (value,)
        raise PhaseError(
            "batched pipeline output cannot be split into logical rows; "
            "the backend must return one image/tensor per branch"
        )

    def rows_by_branch(self) -> dict[str, Any]:
        return dict(zip(self.branch_ids, self.row_outputs(), strict=True))


@dataclass(frozen=True, slots=True)
class _DeviceEmbedCache:
    """The single validated device-side embedding entry."""

    embeds: PromptEmbeds
    source_tensors: dict[str, Any]
    source_signature: tuple[Any, ...]
    device_tensors: dict[str, Any]
    device_signature: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class PhaseTelemetry:
    """Cumulative cache counters plus the latest CUDA allocator snapshot."""

    device_cache_hits: int
    device_tensor_transfers: int
    device_transfer_bytes: int
    device_cache_entries: int = 0
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    max_memory_allocated_mb: float | None = None
    memory_event_count: int = 0

    @property
    def cache_hits(self) -> int:
        """Short alias for callers that do not need the device qualifier."""
        return self.device_cache_hits

    @property
    def transferred_bytes(self) -> int:
        """Short alias for the logical bytes supplied to the device path."""
        return self.device_transfer_bytes


def _filter_kwargs(target: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the callable accepts (mirrors image_atlas's filter)."""
    import inspect

    if not inspect.isfunction(target) and not inspect.ismethod(target) and callable(target):
        target = target.__call__
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return dict(kwargs)
    accepted = {
        p.name
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p.name != "self"
    }
    return {name: value for name, value in kwargs.items() if name in accepted}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    return repr(value)


def embed_cache_key(pipeline_class: str, prompt: str, params: Mapping[str, Any]) -> str:
    """Deterministic cache key: pipeline class + prompt + embed-relevant params."""
    payload = json.dumps(
        {
            "v": 2,
            "pipeline_class": pipeline_class,
            "prompt": prompt,
            "params": _jsonable(dict(params)),
            # Saturn's FLUX.1 helper supplies the T5 attention mask while the
            # stock Diffusers 0.39 encode_prompt path does not.  Keep caches
            # from crossing that conditioner contract if either implementation
            # is used by a long-lived worker.
            "conditioning_contract": (
                "flux1-masked-t5-v1" if pipeline_class == "FluxPipeline" else "native-v1"
            ),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PhasePipeline:
    """Temporal split of one Diffusers pipeline across a too-small CUDA card."""

    def __init__(self) -> None:  # pragma: no cover - use wrap()
        raise PhaseError("use PhasePipeline.wrap(pipeline, ...)")

    # -- construction -----------------------------------------------------

    @classmethod
    def wrap(
        cls,
        pipeline: Any,
        *,
        device: str = "cuda",
        cache_dir: str | os.PathLike[str] | None = None,
        device_embed_cache: bool = False,
        trajectory_cache: TrajectoryCache | None = None,
        model_identity: str | None = None,
    ) -> PhasePipeline:
        import torch

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise PhaseError("phase-cuda requested but CUDA is unavailable")

        class_name = type(pipeline).__name__
        contract = EMBED_CONTRACTS.get(class_name)
        if contract is None:
            raise PhaseError(
                f"no embed contract for pipeline class {class_name!r}; "
                f"supported: {sorted(EMBED_CONTRACTS)}. Refusing to guess how "
                "cached embeddings feed back into __call__."
            )
        encode = getattr(pipeline, contract.encode_method, None)
        if not callable(encode):
            raise PhaseError(f"pipeline {class_name} does not expose {contract.encode_method}()")
        import inspect

        call_params = inspect.signature(type(pipeline).__call__).parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in call_params.values())
        missing = [k for k in contract.feed_keys if k not in call_params]
        if missing and not has_var_kw:
            raise PhaseError(
                f"pipeline {class_name}.__call__ accepts no {missing} — no embed "
                "path exists; phase-cuda cannot reproduce the in-call encode."
            )

        encoders = {}
        for name in _ENCODER_ATTRS:
            module = getattr(pipeline, name, None)
            if module is not None and isinstance(module, torch.nn.Module):
                encoders[name] = module
        if not encoders:
            raise PhaseError(f"pipeline {class_name} has no text encoder module")
        denoiser_name = next(
            (
                name
                for name in _DENOISER_ATTRS
                if isinstance(getattr(pipeline, name, None), torch.nn.Module)
            ),
            None,
        )
        if denoiser_name is None:
            raise PhaseError(f"pipeline {class_name} has neither .transformer nor .unet")

        self = object.__new__(cls)
        self._pipeline = pipeline
        self._class_name = class_name
        self._model_identity = str(model_identity or class_name)
        self._contract = contract
        self._device = str(device)
        self._external_sequential_cpu_offload = bool(
            getattr(pipeline, "_saturn_sequential_cpu_offload", False)
        )
        self._encoders = encoders  # name -> module (authoritative refs)
        self._denoiser_name = denoiser_name
        self._detached: dict[str, Any] = {}
        self._resident: str = "none"  # none | encode | denoise
        self._memory_cache: dict[str, PromptEmbeds] = {}
        self._device_embed_cache_enabled = bool(device_embed_cache)
        # Exactly one device-side entry. The entry retains the source tensor
        # objects themselves so replacement cannot be confused by id reuse.
        self._device_embed_cache: _DeviceEmbedCache | None = None
        self._device_cache_hits = 0
        self._device_tensor_transfers = 0
        self._device_transfer_bytes = 0
        self._last_memory_snapshot: dict[str, Any] = {}
        self._memory_event_count = 0
        self._compiled_denoiser: Any | None = None
        self._compile_config: dict[str, Any] | None = None
        self._disk_cache = None
        self._trajectory_cache = trajectory_cache
        if cache_dir is not None:
            from .cache import EmbedCache

            self._disk_cache = EmbedCache(cache_dir)
        self.phase_events: list[dict[str, Any]] = []
        self.encode_calls = 0
        self.fresh_encode_calls = 0
        self.encode_cache_hits = 0
        self.encode_disk_cache_hits = 0
        self.last_encode_observation: dict[str, Any] = {}
        self.unplanned_swaps = 0
        return self

    # -- attribute forwarding --------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_pipeline"], name)

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    @property
    def resident_phase(self) -> str:
        return self._resident

    def telemetry(self) -> PhaseTelemetry:
        """Return cache counters and the latest allocator snapshot.

        The memory fields are deliberately the last post-operation sample,
        rather than a guessed reservation.  Callers can record this value
        after every generation and compare it with the next sample to detect
        retained allocations independently of mrun's process-level guard.
        """
        memory = self._last_memory_snapshot
        return PhaseTelemetry(
            device_cache_hits=self._device_cache_hits,
            device_tensor_transfers=self._device_tensor_transfers,
            device_transfer_bytes=self._device_transfer_bytes,
            device_cache_entries=int(self._device_embed_cache is not None),
            memory_allocated_mb=memory.get("memory_allocated_mb"),
            memory_reserved_mb=memory.get("memory_reserved_mb"),
            max_memory_allocated_mb=memory.get("max_memory_allocated_mb"),
            memory_event_count=self._memory_event_count,
        )

    def trajectory_cache_stats(self) -> dict[str, int]:
        """Return content-addressed prefix-cache counters for debugger telemetry."""

        if self._trajectory_cache is None:
            return {"entries": 0, "hits": 0, "misses": 0, "evictions": 0, "invalidations": 0}
        return self._trajectory_cache.stats()

    def invalidate_trajectory_cache(self, dependency_keys: Iterable[str]) -> tuple[str, ...]:
        """Invalidate cached prefixes that depend on a reference or edit key."""

        if self._trajectory_cache is None:
            return ()
        return self._trajectory_cache.invalidate(dependency_keys)

    # -- phase machinery --------------------------------------------------

    def _invalidate_device_embed_cache(self) -> None:
        """Drop the sole device embedding entry and release its wrapper reference."""
        self._device_embed_cache = None

    def _move_module(self, name: str, module: Any, target: str) -> Any:
        """Move one component; keep pipeline attr + local refs pointing at the result.

        ``nn.Module.to`` is in-place, but wrapped/quantized components may return
        a new object — losing it would leave a stale reference on the pipeline.
        """
        # A lazily dispatched component can legitimately still contain meta
        # tensors when the wrapper is being torn down.  Meta tensors have no
        # storage to copy, so ``Module.to("cpu")`` raises instead of releasing
        # anything.  Do not materialize an uninitialized module merely to close
        # the phase wrapper; the caller is dropping the pipeline immediately.
        if target == "cpu" and any(
            bool(getattr(tensor, "is_meta", False))
            for tensor in (*module.parameters(), *module.buffers())
        ):
            return module
        moved = module.to(target)
        if moved is None:
            moved = module
        if moved is not module:
            if getattr(self._pipeline, name, None) is module:
                setattr(self._pipeline, name, moved)
            if name in self._encoders:
                self._encoders[name] = moved
            if name in self._detached:
                self._detached[name] = moved
        return moved

    def _vram_reserved_mb(self) -> float | None:
        if not self._device.startswith("cuda"):
            return None
        import torch

        torch.cuda.synchronize()
        return float(torch.cuda.memory_reserved()) / (1024 * 1024)

    def _record_memory_snapshot(self, label: str) -> dict[str, Any]:
        """Record allocator counters after one native generation operation.

        ``nvidia-smi``/NVML reports process-level residency, while these
        counters distinguish live tensors from blocks retained by PyTorch's
        caching allocator.  A telemetry failure must not turn a successful
        model invocation into a failed invocation, so an unavailable CUDA
        counter is represented explicitly in the event.
        """

        event: dict[str, Any] = {"label": str(label)}
        if not self._device.startswith("cuda"):
            event.update(
                {
                    "memory_allocated_mb": None,
                    "memory_reserved_mb": None,
                    "max_memory_allocated_mb": None,
                }
            )
        else:
            import torch

            try:
                torch.cuda.synchronize()
                event.update(
                    {
                        "memory_allocated_mb": float(torch.cuda.memory_allocated())
                        / (1024 * 1024),
                        "memory_reserved_mb": float(torch.cuda.memory_reserved())
                        / (1024 * 1024),
                        "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated())
                        / (1024 * 1024),
                    }
                )
            except (AttributeError, RuntimeError) as exc:
                event.update(
                    {
                        "memory_allocated_mb": None,
                        "memory_reserved_mb": None,
                        "max_memory_allocated_mb": None,
                        "memory_telemetry_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        self._memory_event_count += 1
        event["memory_event_count"] = self._memory_event_count
        self._last_memory_snapshot = event
        return dict(event)

    def _boundary(self, label: str) -> None:
        rss = guard.check_rss(f"mrun.diffusion.{label}")
        vram = self._vram_reserved_mb()
        limit_raw = os.environ.get("VRAM_LIMIT_MB", "").strip()
        limit = float(limit_raw) if limit_raw else None
        event = {"label": label, "rss_mb": rss, "vram_reserved_mb": vram}
        self.phase_events.append(event)
        if limit and vram is not None and vram > limit:
            raise PhaseError(
                f"VRAM reserved {vram:.0f} MB exceeds limit {limit:.0f} MB at {label!r}"
            )

    def _empty_cuda_cache(self) -> None:
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def _move_flux1_vae(self, target: str) -> None:
        """Keep the FLUX.1 VAE off-card while its transformer is resident.

        FLUX.1's 12B transformer plus the VAE do not fit the 16 GB CUDA
        worker with the same resident policy that FLUX.2 uses.  The native
        trajectory ABI only needs the VAE at the decode boundary, so moving
        it for that short operation preserves exact denoising while keeping
        checkpoint capture/replay within the measured VRAM envelope.
        """

        if (
            self._class_name != "FluxPipeline"
            or self._external_sequential_cpu_offload
        ):
            return
        vae = getattr(self._pipeline, _VAE_ATTR, None)
        if vae is not None:
            self._move_module(_VAE_ATTR, vae, target)

    def _reattach_encoders(self) -> None:
        for name in list(self._detached):
            setattr(self._pipeline, name, self._detached.pop(name))

    def _ensure_phase(self, phase: str) -> None:
        if phase not in ("encode", "denoise"):
            raise PhaseError(f"unknown phase {phase!r}")
        if self._resident == phase:
            return
        if self._external_sequential_cpu_offload:
            # Diffusers/Accelerate owns component placement in this mode.
            # Moving the transformer here would defeat sequential offload and
            # recreate the 12B FLUX.1 residency failure at the phase boundary.
            self._invalidate_device_embed_cache()
            self._empty_cuda_cache()
            self._resident = phase
            self._boundary(phase)
            return
        if phase == "encode":
            # Device embeddings must not keep occupying the card while the
            # encoder is brought back for a new prompt.
            self._invalidate_device_embed_cache()
            # Denoiser + VAE off the card first so the encoder never has to
            # coexist with them at peak.
            self._move_module(
                self._denoiser_name, getattr(self._pipeline, self._denoiser_name), "cpu"
            )
            vae = getattr(self._pipeline, _VAE_ATTR, None)
            if vae is not None:
                self._move_module(_VAE_ATTR, vae, "cpu")
            self._empty_cuda_cache()
            self._reattach_encoders()
            for name, module in self._encoders.items():
                self._move_module(name, module, self._device)
        else:
            # Encoders off the card AND detached from the pipeline: with no
            # accelerate hooks, DiffusionPipeline._execution_device falls back
            # to the first module's device — a CPU-resident text encoder would
            # silently pull latents onto the CPU. Detaching (attr -> None) makes
            # the denoiser define the execution device and makes any accidental
            # in-call re-encode a hard error instead of a silent slow path.
            for name, module in self._encoders.items():
                self._move_module(name, module, "cpu")
                if getattr(self._pipeline, name, None) is not None:
                    self._detached[name] = module
                    setattr(self._pipeline, name, None)
            self._empty_cuda_cache()
            self._move_module(
                self._denoiser_name,
                getattr(self._pipeline, self._denoiser_name),
                self._device,
            )
            vae = getattr(self._pipeline, _VAE_ATTR, None)
            if vae is not None:
                if self._class_name == "FluxPipeline":
                    self._move_module(_VAE_ATTR, vae, "cpu")
                else:
                    self._move_module(_VAE_ATTR, vae, self._device)
            # Moving FLUX.1's VAE off-card releases its allocations, but the
            # CUDA caching allocator otherwise keeps those blocks counted as
            # reserved at the typed phase boundary.
            self._empty_cuda_cache()
        self._resident = phase
        self._boundary(phase)

    class _PhaseContext:
        def __init__(self, owner: PhasePipeline, phase: str) -> None:
            self._owner = owner
            self._phase = phase

        def __enter__(self) -> PhasePipeline:
            self._owner._ensure_phase(self._phase)
            return self._owner

        def __exit__(self, *exc_info: Any) -> None:
            # Leaving a phase moves nothing — the next phase entry pays the
            # transfer. The context exists for explicit structure, not cleanup.
            return None

    def encode_phase(self) -> PhasePipeline._PhaseContext:
        return self._PhaseContext(self, "encode")

    def denoise_phase(self) -> PhasePipeline._PhaseContext:
        return self._PhaseContext(self, "denoise")

    def compile_denoiser(
        self,
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
    ) -> Any:
        """Install a fixed-shape ``torch.compile`` denoiser wrapper.

        Compilation is deliberately explicit and scoped to the denoise phase:
        text encoding, scheduler Python control flow, and VAE decoding remain
        outside the graph.  Reusing a compiled wrapper with a different graph
        policy is rejected so a serving process cannot silently mix benchmark
        arms.  Numerical parity is not implied; callers must record the
        ``numerical_contract`` emitted by the batch result.
        """
        self._ensure_phase("denoise")
        import torch

        if not hasattr(torch, "compile"):
            raise PhaseError("torch.compile is unavailable in this torch build")
        config = {
            "mode": mode,
            "fullgraph": bool(fullgraph),
            "dynamic": bool(dynamic),
        }
        if self._compiled_denoiser is not None:
            if self._compile_config != config:
                raise PhaseError(
                    "denoiser is already compiled with a different execution policy"
                )
            return self._compiled_denoiser
        denoiser = getattr(self._pipeline, self._denoiser_name)
        compiled = torch.compile(
            denoiser,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        setattr(self._pipeline, self._denoiser_name, compiled)
        self._compiled_denoiser = compiled
        self._compile_config = config
        return compiled

    def compile_status(self) -> dict[str, Any]:
        """Return transport-safe compiler state for telemetry and admission."""
        return {
            "compiled": self._compiled_denoiser is not None,
            "config": dict(self._compile_config or {}),
            "denoiser": self._denoiser_name,
        }

    def close(self) -> None:
        """Move everything back to CPU and reattach detached components."""
        self._invalidate_device_embed_cache()
        self._reattach_encoders()
        for name, module in self._encoders.items():
            self._move_module(name, module, "cpu")
        self._move_module(self._denoiser_name, getattr(self._pipeline, self._denoiser_name), "cpu")
        vae = getattr(self._pipeline, _VAE_ATTR, None)
        if vae is not None:
            self._move_module(_VAE_ATTR, vae, "cpu")
        self._empty_cuda_cache()
        self._resident = "none"

    # -- encode -----------------------------------------------------------

    def _encode_params(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: kwargs[name]
            for name in self._contract.encode_param_names
            if name in kwargs and kwargs[name] is not None
        }

    def _encode_flux1_main_conditioner(
        self,
        prompt: str,
        params: Mapping[str, Any],
    ) -> tuple[Any, Any] | None:
        """Encode FLUX.1 with the masked dual-encoder contract used by Saturn-main.

        Diffusers 0.39's ``FluxPipeline._get_t5_prompt_embeds`` passes only
        ``input_ids`` to T5.  Saturn-main's established FLUX.1 helper passes
        the tokenizer's full ``BatchEncoding``, including ``attention_mask``.
        Those paths produce different sequence embeddings for padded 512-token
        inputs.  Keep the difference explicit and local to FLUX.1 rather than
        changing the generic pipeline contract or silently accepting a
        conditioning mismatch at the pixel consumer.

        ``None`` is returned for partial/fake pipelines and LoRA calls that
        need Diffusers' scale hooks; normal pipelines then use the existing
        native ``encode_prompt`` path.
        """

        if self._class_name != "FluxPipeline" or params.get("lora_scale") is not None:
            return None
        pipeline = self._pipeline
        tokenizer = getattr(pipeline, "tokenizer", None)
        tokenizer_2 = getattr(pipeline, "tokenizer_2", None)
        text_encoder = getattr(pipeline, "text_encoder", None)
        text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
        if any(value is None for value in (tokenizer, tokenizer_2, text_encoder, text_encoder_2)):
            return None

        import torch

        max_sequence_length = int(params.get("max_sequence_length", 512))
        prompts = [prompt]
        with torch.inference_mode():
            t5_inputs = tokenizer_2(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            ).to(self._device)
            t5_outputs = text_encoder_2(**t5_inputs)
            prompt_embeds = t5_outputs[0].to(
                device=self._device,
                dtype=torch.bfloat16,
            )

            clip_inputs = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=int(tokenizer.model_max_length),
                return_tensors="pt",
            ).to(self._device)
            clip_outputs = text_encoder(**clip_inputs)
            pooled_prompt_embeds = clip_outputs.pooler_output.to(
                device=self._device,
                dtype=torch.bfloat16,
            )
        return prompt_embeds, pooled_prompt_embeds

    def encode(self, prompt: str, **params: Any) -> PromptEmbeds:
        """Encode ONE prompt (serial by contract), cache, and return CPU embeds."""
        import torch

        if not isinstance(prompt, str):
            raise PhaseError(
                "encode() takes one prompt string; batching pads the sequence and "
                "changes the embeddings numerically. Encode serially."
            )
        params = self._encode_params(params) if params else {}
        key = embed_cache_key(self._class_name, prompt, params)
        cached = self._memory_cache.get(key)
        if cached is not None:
            self.encode_cache_hits += 1
            self.last_encode_observation = {
                "key": key,
                "cache_hit": True,
                "memory_cache_hit": True,
                "disk_cache_hit": False,
                "native_encode_invoked": False,
                "native_encode_method": self._contract.encode_method,
                "fresh": False,
            }
            return cached
        if self._disk_cache is not None:
            cached = self._disk_cache.load(key)
            if cached is not None:
                self.encode_disk_cache_hits += 1
                self.last_encode_observation = {
                    "key": key,
                    "cache_hit": True,
                    "memory_cache_hit": False,
                    "disk_cache_hit": True,
                    "native_encode_invoked": False,
                    "native_encode_method": self._contract.encode_method,
                    "fresh": False,
                }
                self._memory_cache[key] = cached
                return cached

        if self._resident != "encode":
            self.unplanned_swaps += self._resident == "denoise"
            self._ensure_phase("encode")
        encode = getattr(self._pipeline, self._contract.encode_method)
        call_kwargs = _filter_kwargs(
            encode,
            {
                "prompt": prompt,
                "device": torch.device(self._device),
                "num_images_per_prompt": 1,
                **params,
            },
        )
        special_encoded = self._encode_flux1_main_conditioner(prompt, params)
        encoded = special_encoded
        if encoded is None:
            with torch.inference_mode():
                encoded = encode(**call_kwargs)
        self.encode_calls += 1
        self.last_encode_observation = {
            "key": key,
            "cache_hit": False,
            "memory_cache_hit": False,
            "disk_cache_hit": False,
            "native_encode_invoked": True,
            "native_encode_method": (
                "_encode_flux1_main_conditioner"
                if special_encoded is not None
                else self._contract.encode_method
            ),
            "fresh": False,
        }
        if isinstance(encoded, torch.Tensor):
            encoded = (encoded,)
        if not isinstance(encoded, (tuple, list)) or len(encoded) < len(self._contract.feed_keys):
            raise PhaseError(
                f"{self._class_name}.{self._contract.encode_method} returned "
                f"{type(encoded).__name__}; expected >= {len(self._contract.feed_keys)} "
                "tensors"
            )
        tensors: dict[str, Any] = {}
        # Trailing derived tensors (text_ids) are sliced off: the pipeline
        # recomputes them from the embeds and we never cache them.
        indices = self._contract.feed_indices or tuple(range(len(self._contract.feed_keys)))
        if len(encoded) <= max(indices, default=-1):
            raise PhaseError(f"{self._class_name}.{self._contract.encode_method} returned too few values")
        feed_values = tuple(encoded[index] for index in indices)
        for feed_key, value in zip(self._contract.feed_keys, feed_values, strict=True):
            if not isinstance(value, torch.Tensor):
                raise PhaseError(
                    f"encode output for {feed_key!r} is {type(value).__name__}, not a tensor"
                )
            # Exact dtype preserved; a bf16 embed cast through fp32 would break
            # bitwise parity with the in-call encode.
            tensors[feed_key] = value.detach().to("cpu").clone()
        embeds = PromptEmbeds(
            key=key,
            tensors=tensors,
            meta={
                "pipeline_class": self._class_name,
                "prompt": prompt,
                "params": _jsonable(params),
                "dtypes": {k: str(v.dtype) for k, v in tensors.items()},
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
            },
        )
        self._memory_cache[key] = embeds
        if self._disk_cache is not None:
            self._disk_cache.save(embeds)
        return embeds

    def encode_fresh(self, prompt: str, **params: Any) -> PromptEmbeds:
        """Encode ONE prompt while bypassing memory and disk embed caches.

        This is an explicit diagnostic/control seam for callers that need to
        prove a conditioner was produced by a new native encoder execution.
        The fresh result is not promoted into either cache, and an existing
        in-memory entry is restored afterward.  PhasePipeline remains serial
        per prompt, so cache ownership does not need a concurrent mutation
        protocol here.
        """

        if not isinstance(prompt, str):
            raise PhaseError(
                "encode_fresh() takes one prompt string; batching pads the sequence and "
                "changes the embeddings numerically. Encode serially."
            )
        params = self._encode_params(params) if params else {}
        key = embed_cache_key(self._class_name, prompt, params)
        cached = self._memory_cache.pop(key, None)
        disk_cache = self._disk_cache
        self._disk_cache = None
        try:
            encode_calls_before = self.encode_calls
            result = self.encode(prompt, **params)
            if self.encode_calls != encode_calls_before + 1:
                raise PhaseError(
                    "encode_fresh() did not invoke exactly one uncached native encoder call"
                )
            self.fresh_encode_calls += 1
            self.last_encode_observation = {
                **self.last_encode_observation,
                "fresh": True,
                "fresh_encode_call": True,
                "memory_cache_bypassed": True,
                "disk_cache_bypassed": True,
                "fresh_result_promoted": False,
                "memory_cache_entry_present_before": cached is not None,
                "fresh_encode_call_index": self.fresh_encode_calls,
            }
            return result
        finally:
            self._disk_cache = disk_cache
            self._memory_cache.pop(key, None)
            if cached is not None:
                self._memory_cache[key] = cached

    def precompute(self, prompts: Iterable[str], **params: Any) -> list[PromptEmbeds]:
        """Encode a panel of prompts serially in one encode phase."""
        return [self.encode(prompt, **params) for prompt in prompts]

    # -- generate ---------------------------------------------------------

    @staticmethod
    def _tensor_version(tensor: Any) -> int | None:
        try:
            return int(tensor._version)
        except (AttributeError, TypeError, RuntimeError):
            return None

    def _tensor_signature(self, tensors: Mapping[str, Any]) -> tuple[Any, ...]:
        """Capture tensor state that can change without replacing the object."""
        return tuple(
            (
                feed_key,
                self._tensor_version(tensor),
                str(getattr(tensor, "dtype", "")),
                tuple(getattr(tensor, "shape", ())),
            )
            for feed_key in self._contract.feed_keys
            for tensor in (tensors[feed_key],)
        )

    def _embed_signature(self, embeds: PromptEmbeds) -> tuple[Any, ...]:
        """Capture the PromptEmbeds key and current source tensor state."""
        return (embeds.key, self._tensor_signature(embeds.tensors))

    def _same_source_tensors(
        self, cached: Mapping[str, Any], current: Mapping[str, Any]
    ) -> bool:
        """Compare source tensors by object identity, never by object id."""
        return all(
            feed_key in cached
            and feed_key in current
            and cached[feed_key] is current[feed_key]
            for feed_key in self._contract.feed_keys
        )

    def _device_feed_tensors(self, embeds: PromptEmbeds) -> dict[str, Any]:
        """Resolve one validated device-side conditioner entry.

        The trajectory runtime uses the same entry as ordinary ``generate``.
        Keeping this in one helper is important: a paused/replayed branch must
        not silently pay a new conditioner transfer or use a differently cast
        prompt embedding than the scalar authority path.
        """

        if not isinstance(embeds, PromptEmbeds):
            raise PhaseError("generate() requires a PromptEmbeds (use encode()/__call__)" )
        signature = self._embed_signature(embeds)
        cached = self._device_embed_cache
        if (
            self._device_embed_cache_enabled
            and cached is not None
            and cached.embeds is embeds
            and self._same_source_tensors(cached.source_tensors, embeds.tensors)
            and cached.source_signature == signature
            and cached.device_signature == self._tensor_signature(cached.device_tensors)
        ):
            self._device_cache_hits += 1
            return dict(cached.device_tensors)

        self._invalidate_device_embed_cache()
        source_tensors = {
            feed_key: embeds.tensors[feed_key] for feed_key in self._contract.feed_keys
        }
        device_tensors = {}
        for feed_key in self._contract.feed_keys:
            tensor = source_tensors[feed_key]
            device_tensors[feed_key] = tensor.to(self._device)
            self._device_tensor_transfers += 1
            self._device_transfer_bytes += self._tensor_bytes(tensor)
        if self._device_embed_cache_enabled:
            self._device_embed_cache = _DeviceEmbedCache(
                embeds=embeds,
                source_tensors=source_tensors,
                source_signature=signature,
                device_tensors=device_tensors,
                device_signature=self._tensor_signature(device_tensors),
            )
        return dict(device_tensors)

    @staticmethod
    def _tensor_bytes(tensor: Any) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def generate(self, embeds: PromptEmbeds, **kwargs: Any) -> Any:
        """Run the wrapped pipeline's __call__ from cached embeds (denoise phase)."""
        import torch

        if not isinstance(embeds, PromptEmbeds):
            raise PhaseError("generate() requires a PromptEmbeds (use encode()/__call__)")
        self._ensure_phase("denoise")
        call_kwargs = dict(kwargs)
        call_kwargs.pop("prompt", None)
        for feed_key in self._contract.feed_keys:
            if feed_key in call_kwargs:
                raise PhaseError(f"generate() manages {feed_key!r}; do not pass it explicitly")

        device_tensors = self._device_feed_tensors(embeds)

        for feed_key, tensor in device_tensors.items():
            call_kwargs[feed_key] = tensor
        call_kwargs = _filter_kwargs(self._pipeline, call_kwargs)
        self._move_flux1_vae(self._device)
        try:
            with torch.inference_mode():
                return self._pipeline(**call_kwargs)
        finally:
            try:
                self._move_flux1_vae("cpu")
                self._empty_cuda_cache()
            finally:
                self._record_memory_snapshot("generate")

    def generate_batch(
        self,
        embeds: PromptEmbeds | Sequence[PromptEmbeds],
        *,
        branch_ids: Sequence[str] | None = None,
        initial_latents: Any | None = None,
        **kwargs: Any,
    ) -> PhaseBatchResult:
        """Run compatible pre-encoded branches in one physical denoise call.

        Encoding remains serial.  The supplied embeddings are therefore the
        exact outputs of independent prompt encodes; this method only stacks
        those CPU tensors and shares the denoise/VAE execution.  A single
        ``PromptEmbeds`` is repeated for branches such as seed/dose/spatial
        intervention ladders.  A sequence permits independently encoded
        prompts with identical feed shapes and dtypes.

        Unsupported output splitting, feed-shape disagreement, or explicit
        feed overrides raises ``PhaseError``.  There is intentionally no
        scalar fallback here: a caller claiming a batched run must receive a
        real batch call or an explicit failure.
        """
        import torch

        self._ensure_phase("denoise")
        if isinstance(embeds, PromptEmbeds):
            embed_rows = [embeds]
        else:
            embed_rows = list(embeds)
            if not embed_rows or any(not isinstance(item, PromptEmbeds) for item in embed_rows):
                raise PhaseError("generate_batch requires PromptEmbeds values")
        if branch_ids is None:
            resolved_branch_ids = tuple(f"branch-{index}" for index in range(len(embed_rows)))
        else:
            if isinstance(branch_ids, (str, bytes, bytearray)):
                raise PhaseError("generate_batch branch_ids must be a sequence of IDs")
            resolved_branch_ids = tuple(branch_ids)
        if not resolved_branch_ids:
            raise PhaseError("generate_batch requires at least one branch")
        if len(set(resolved_branch_ids)) != len(resolved_branch_ids):
            raise PhaseError("generate_batch branch_ids must be unique")
        if len(embed_rows) not in (1, len(resolved_branch_ids)):
            raise PhaseError("one PromptEmbeds or one PromptEmbeds per branch is required")
        if len(embed_rows) == 1 and len(resolved_branch_ids) > 1:
            embed_rows = embed_rows * len(resolved_branch_ids)

        call_kwargs = dict(kwargs)
        call_kwargs.pop("prompt", None)
        for feed_key in self._contract.feed_keys:
            if feed_key in call_kwargs:
                raise PhaseError(
                    f"generate_batch() manages {feed_key!r}; do not pass it explicitly"
                )

        # Resolve each row through the same validated device-side cache used by
        # scalar generation before concatenating.  The previous implementation
        # stacked CPU tensors first, which made a repeated-prompt batch pay a
        # fresh transfer every time and made the cache structurally impossible
        # to hit in the batched path.
        transfer_count_before = self._device_tensor_transfers
        transfer_bytes_before = self._device_transfer_bytes
        device_rows: list[dict[str, Any]] = []
        for item in embed_rows:
            for feed_key in self._contract.feed_keys:
                tensor = item.tensors.get(feed_key)
                if tensor is None:
                    raise PhaseError(f"PromptEmbeds is missing required feed {feed_key!r}")
                if tensor.ndim == 0:
                    raise PhaseError(f"batched feed {feed_key!r} must have a leading dimension")
                if int(tensor.shape[0]) != 1:
                    raise PhaseError(
                        f"PromptEmbeds feed {feed_key!r} must have leading dimension 1"
                    )
            device_rows.append(self._device_feed_tensors(item))

        stacked: dict[str, Any] = {}
        for feed_key in self._contract.feed_keys:
            tensors = [row[feed_key] for row in device_rows]
            reference = tensors[0]
            for tensor in tensors[1:]:
                if (
                    tuple(tensor.shape) != tuple(reference.shape)
                    or tensor.dtype != reference.dtype
                ):
                    raise PhaseError(
                        f"incompatible {feed_key!r} shapes/dtypes prevent a static batch"
                    )
            stacked[feed_key] = torch.cat(tensors, dim=0)

        device_tensors = stacked
        extra_transfer_bytes = 0
        if initial_latents is not None:
            if not isinstance(initial_latents, torch.Tensor) or initial_latents.ndim == 0:
                raise PhaseError("initial_latents must be a tensor with a leading batch dimension")
            if int(initial_latents.shape[0]) == 1 and len(resolved_branch_ids) > 1:
                initial_latents = initial_latents.repeat(
                    (len(resolved_branch_ids), *([1] * (initial_latents.ndim - 1)))
                )
            elif int(initial_latents.shape[0]) != len(resolved_branch_ids):
                raise PhaseError("initial_latents leading dimension must match branch_ids")
            call_kwargs["latents"] = initial_latents.to(self._device)
            extra_transfer_bytes = self._tensor_bytes(initial_latents)

        for feed_key, tensor in device_tensors.items():
            call_kwargs[feed_key] = tensor
        call_kwargs = _filter_kwargs(self._pipeline, call_kwargs)
        self._move_flux1_vae(self._device)
        memory_snapshot: dict[str, Any] = {}
        try:
            with torch.inference_mode():
                output = self._pipeline(**call_kwargs)
        finally:
            try:
                self._move_flux1_vae("cpu")
                self._empty_cuda_cache()
            finally:
                memory_snapshot = self._record_memory_snapshot("generate_batch")
        telemetry = {
            "physical_pipeline_calls": 1,
            "batch_size": len(resolved_branch_ids),
            "branch_ids": list(resolved_branch_ids),
            "embedding_rows": len(embed_rows),
            "shared_embed_transfer": len(resolved_branch_ids) > 1,
            "device_tensor_transfer_count": self._device_tensor_transfers
            - transfer_count_before,
            "device_transfer_bytes": (
                self._device_transfer_bytes - transfer_bytes_before + extra_transfer_bytes
            ),
            **memory_snapshot,
            "numerical_contract": (
                "exact_scalar" if len(resolved_branch_ids) == 1 else "batch_approximate"
            ),
            "pipeline_class": self._class_name,
        }
        result = PhaseBatchResult(
            output=output,
            branch_ids=resolved_branch_ids,
            batch_size=len(resolved_branch_ids),
            telemetry=telemetry,
        )
        # Validate the output contract before returning a claimed batch result.
        result.row_outputs()
        return result

    # -- checkpointed trajectory execution --------------------------------

    def capture_checkpoint(
        self,
        embeds: PromptEmbeds,
        *,
        cut_step: int,
        initial_latents: Any | None = None,
        reference_image: Any | None = None,
        cache_key: str | None = None,
        schedule_fingerprint: str | None = None,
        references: Iterable[str] = (),
        dependency_keys: Iterable[str] = (),
        step_observer: Callable[[Mapping[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> TrajectoryCheckpoint | DiffusionTrajectoryCheckpoint:
        """Pause a supported diffusion trajectory after ``cut_step`` steps.

        This is the debugger/runtime seam that ordinary ``generate_batch``
        cannot provide.  The shared prefix is executed once and the packed
        latent, timestep schedule, IDs, and conditioner are frozen in an
        immutable in-process checkpoint.  Branches can then run or rewind to
        this state without replaying the prefix.
        """

        self._ensure_phase("denoise")
        if cache_key is not None and self._trajectory_cache is None:
            raise PhaseError("cache_key requires a TrajectoryCache on PhasePipeline.wrap()")
        resolved_schedule = str(schedule_fingerprint or "")
        if cache_key is not None:
            cached = self._trajectory_cache.get(
                cache_key,
                model_identity=self._model_identity,
                conditioning_key=embeds.key,
                schedule_fingerprint=resolved_schedule,
            )
            if cached is not None:
                return cached.checkpoint
        try:
            if self._class_name == "FluxPipeline":
                checkpoint = _capture_flux1_checkpoint(
                    self,
                    embeds,
                    cut_step=cut_step,
                    initial_latents=initial_latents,
                    reference_image=reference_image,
                    references=references,
                    kwargs=kwargs,
                    step_observer=step_observer,
                )
            elif self._class_name.startswith("Flux2"):
                checkpoint = _capture_flux2_checkpoint(
                    self,
                    embeds,
                    cut_step=cut_step,
                    initial_latents=initial_latents,
                    reference_image=reference_image,
                    references=references,
                    kwargs=kwargs,
                    step_observer=step_observer,
                )
            elif self._class_name in NON_FLUX_PIPELINES:
                checkpoint = capture_nonflux_checkpoint(
                    self,
                    embeds,
                    cut_step=cut_step,
                    initial_latents=initial_latents,
                    reference_image=reference_image,
                    references=tuple(references),
                    kwargs=kwargs,
                    step_observer=step_observer,
                )
            else:
                raise PhaseError(
                    f"no trajectory contract for pipeline class {self._class_name!r}; "
                    "refusing to route it through a different diffusion family"
                )
        finally:
            self._record_memory_snapshot("capture_checkpoint")
        if cache_key is not None:
            self._trajectory_cache.put(
                checkpoint,
                key=cache_key,
                model_identity=self._model_identity,
                conditioning_key=embeds.key,
                schedule_fingerprint=resolved_schedule,
                references=references,
                dependency_keys=dependency_keys,
            )
        return checkpoint

    def resume_checkpoint(
        self,
        checkpoint: TrajectoryCheckpoint | DiffusionTrajectoryCheckpoint,
        *,
        latent_override: Any | None = None,
        prompt_embeds_override: Any | None = None,
        output_type: str = "pil",
        generator: Any | None = None,
        step_observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Any:
        """Revert to a trajectory checkpoint and run one branch suffix.

        ``prompt_embeds_override`` is an explicit lexical/conditioning branch
        control. It preserves the checkpoint latent, schedule, layout, IDs,
        guidance, and references while replacing the model-facing conditioner
        for the suffix. Family-specific derived streams are rebuilt and
        validated, so a same-shape conditioner swap is a typed operation
        rather than an implicit mutation of the checkpoint.
        """

        self._ensure_phase("denoise")
        try:
            if self._class_name == "FluxPipeline":
                output = _resume_flux1_checkpoint(
                    self,
                    checkpoint,
                    latent_override=latent_override,
                    prompt_embeds_override=prompt_embeds_override,
                    output_type=output_type,
                    step_observer=step_observer,
                )
            elif self._class_name.startswith("Flux2"):
                output = _resume_flux2_checkpoint(
                    self,
                    checkpoint,
                    latent_override=latent_override,
                    prompt_embeds_override=prompt_embeds_override,
                    output_type=output_type,
                    step_observer=step_observer,
                )
            elif self._class_name in NON_FLUX_PIPELINES:
                output = resume_nonflux_checkpoint(
                    self,
                    checkpoint,
                    latent_override=latent_override,
                    prompt_embeds_override=prompt_embeds_override,
                    output_type=output_type,
                    generator=generator,
                    step_observer=step_observer,
                )
            else:
                raise PhaseError(
                    f"no trajectory contract for pipeline class {self._class_name!r}; "
                    "refusing to route it through a different diffusion family"
                )
        finally:
            self._record_memory_snapshot("resume_checkpoint")
        return output

    def advance_checkpoint(
        self,
        checkpoint: TrajectoryCheckpoint | DiffusionTrajectoryCheckpoint,
        *,
        steps: int = 1,
        latent_override: Any | None = None,
        prompt_embeds_override: Any | None = None,
        action_provider: Callable[[Mapping[str, Any]], Any] | None = None,
        step_observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> TrajectoryCheckpoint | DiffusionTrajectoryCheckpoint:
        """Advance a supported diffusion checkpoint by an exact interval.

        Unlike :meth:`resume_checkpoint`, this method does not decode an
        image or run the rest of the suffix. It returns a new immutable
        checkpoint at ``checkpoint.step_index + steps``. The operation is
        the public debugger seam for stepping a model-shaped program while
        preserving the scheduler, IDs, conditioner, and numerical loop used
        by suffix replay. ``action_provider`` is a bounded non-FLUX execution
        seam: when supplied it replaces the native denoiser action while the
        family-native scheduler remains authoritative for every state update.
        """

        self._ensure_phase("denoise")
        try:
            if self._class_name == "FluxPipeline":
                if action_provider is not None:
                    raise PhaseError("action_provider is not registered for FluxPipeline")
                checkpoint = _advance_flux1_checkpoint(
                    self,
                    checkpoint,
                    steps=steps,
                    latent_override=latent_override,
                    prompt_embeds_override=prompt_embeds_override,
                    step_observer=step_observer,
                )
            elif self._class_name.startswith("Flux2"):
                if action_provider is not None:
                    raise PhaseError("action_provider is not registered for Flux2 pipelines")
                checkpoint = _advance_flux2_checkpoint(
                    self,
                    checkpoint,
                    steps=steps,
                    latent_override=latent_override,
                    prompt_embeds_override=prompt_embeds_override,
                    step_observer=step_observer,
                )
            elif self._class_name in NON_FLUX_PIPELINES:
                checkpoint = advance_nonflux_checkpoint(
                    self,
                    checkpoint,
                    steps=steps,
                    latent_override=latent_override,
                    prompt_embeds_override=prompt_embeds_override,
                    action_provider=action_provider,
                    step_observer=step_observer,
                )
            else:
                raise PhaseError(
                    f"no trajectory contract for pipeline class {self._class_name!r}; "
                    "refusing to route it through a different diffusion family"
                )
        finally:
            self._record_memory_snapshot("advance_checkpoint")
        return checkpoint

    def resume_checkpoint_batch(
        self,
        checkpoint: TrajectoryCheckpoint | DiffusionTrajectoryCheckpoint,
        *,
        branch_ids: Sequence[str],
        latent_overrides: Any | Sequence[Any] | None = None,
        mode: str = "exact",
        output_type: str = "pil",
    ) -> PhaseBatchResult:
        """Run many branch suffixes from one checkpoint.

        ``mode="exact"`` is the default and executes branch-local scalar
        suffixes.  It shares the expensive prefix and preserves scalar
        numerical authority.  ``mode="batched"`` fuses the suffix denoiser
        rows and is intentionally reported as a separate approximate
        numerical contract; it must not be mistaken for exact replay.
        """

        self._ensure_phase("denoise")
        memory_snapshot: dict[str, Any] = {}
        try:
            if self._class_name == "FluxPipeline":
                result = _resume_flux1_checkpoint_batch(
                    self,
                    checkpoint,
                    branch_ids=branch_ids,
                    latent_overrides=latent_overrides,
                    mode=mode,
                    output_type=output_type,
                )
            elif self._class_name.startswith("Flux2"):
                result = _resume_flux2_checkpoint_batch(
                    self,
                    checkpoint,
                    branch_ids=branch_ids,
                    latent_overrides=latent_overrides,
                    mode=mode,
                    output_type=output_type,
                )
            elif self._class_name in NON_FLUX_PIPELINES:
                result = resume_nonflux_checkpoint_batch(
                    self,
                    checkpoint,
                    branch_ids=branch_ids,
                    latent_overrides=latent_overrides,
                    mode=mode,
                    output_type=output_type,
                )
            else:
                raise PhaseError(
                    f"no trajectory contract for pipeline class {self._class_name!r}; "
                    "refusing to route it through a different diffusion family"
                )
        finally:
            memory_snapshot = self._record_memory_snapshot("resume_checkpoint_batch")
        result.telemetry.update(memory_snapshot)
        return result

    # -- duck-compatible pipeline call ------------------------------------

    def __call__(self, prompt: str | None = None, **kwargs: Any) -> Any:
        if prompt is None:
            raise PhaseError(
                "PhasePipeline.__call__ needs prompt=<str>; for pre-encoded "
                "embeddings call generate(embeds, ...)"
            )
        if not isinstance(prompt, str):
            raise PhaseError("PhasePipeline is serial per prompt; pass one prompt string per call")
        embeds = self.encode(prompt, **self._encode_params(kwargs))
        return self.generate(embeds, **kwargs)


def _require_flux2_trajectory(phase: PhasePipeline) -> Any:
    """Return the wrapped pipeline when it exposes the FLUX.2 trajectory ABI."""

    pipeline = phase.pipeline
    if not phase._class_name.startswith("Flux2"):
        raise PhaseError(
            "trajectory checkpoints currently require a Flux2 pipeline; "
            f"got {phase._class_name!r}"
        )
    required = (
        "prepare_latents",
        "_prepare_text_ids",
        "_unpack_latents_with_ids",
        "_unpatchify_latents",
        "transformer",
        "scheduler",
        "vae",
    )
    missing = [name for name in required if not hasattr(pipeline, name)]
    if missing:
        raise PhaseError(
            f"{phase._class_name} does not expose the trajectory ABI: {missing}"
        )
    if not callable(getattr(pipeline, "_prepare_text_ids", None)):
        raise PhaseError("Flux2 trajectory ABI requires _prepare_text_ids()")
    return pipeline


def _require_flux1_trajectory(phase: PhasePipeline) -> Any:
    """Return the wrapped pipeline when it exposes the FLUX.1 trajectory ABI."""

    pipeline = phase.pipeline
    if phase._class_name != "FluxPipeline":
        raise PhaseError(
            "trajectory checkpoints require a FluxPipeline or Flux2 pipeline; "
            f"got {phase._class_name!r}"
        )
    required = (
        "prepare_latents",
        "_unpack_latents",
        "transformer",
        "scheduler",
        "vae",
        "image_processor",
    )
    missing = [name for name in required if not hasattr(pipeline, name)]
    if missing:
        raise PhaseError(f"FluxPipeline does not expose the trajectory ABI: {missing}")
    if not callable(getattr(pipeline, "prepare_latents", None)):
        raise PhaseError("FluxPipeline trajectory ABI requires prepare_latents()")
    if not callable(getattr(pipeline, "_unpack_latents", None)):
        raise PhaseError("FluxPipeline trajectory ABI requires _unpack_latents()")
    return pipeline


def _trajectory_helpers(pipeline: Any) -> tuple[Any, Any]:
    """Load the pinned pipeline's schedule helpers without importing Diffusers at module scope."""

    import importlib

    module = importlib.import_module(type(pipeline).__module__)
    empirical_mu = getattr(module, "compute_empirical_mu", None)
    retrieve = getattr(module, "retrieve_timesteps", None)
    if not callable(empirical_mu) or not callable(retrieve):
        raise PhaseError(
            f"{type(pipeline).__name__} module does not expose the Flux2 schedule helpers"
        )
    return empirical_mu, retrieve


def _scheduler_config_fingerprint(scheduler: Any) -> str:
    """Fingerprint the scheduler configuration that defines a trajectory."""

    config = getattr(scheduler, "config", {})
    try:
        payload = dict(config)
    except (TypeError, ValueError):
        payload = vars(config) if hasattr(config, "__dict__") else repr(config)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr).encode(
            "utf-8"
        )
    ).hexdigest()


def _numeric_schedule(values: Any, *, field: str) -> list[float] | None:
    """Return one finite JSON-safe scheduler vector."""

    if values is None:
        return None
    import math

    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        raise PhaseError(f"{field} must be a finite numeric sequence") from None
    if not result or any(not math.isfinite(value) for value in result):
        raise PhaseError(f"{field} must be a non-empty finite numeric sequence")
    return result


def _flux2_schedule(
    pipeline: Any,
    latents: Any,
    total_steps: int,
    device: str,
    *,
    sigmas: Any = None,
) -> tuple[Any, list[float] | None]:
    """Install the native FLUX.2 inference schedule and return its input sigmas."""

    import numpy as np

    if int(total_steps) <= 0:
        raise PhaseError("num_inference_steps must be positive")
    empirical_mu, retrieve = _trajectory_helpers(pipeline)
    if bool(getattr(pipeline.scheduler.config, "use_flow_sigmas", False)):
        schedule_sigmas = None
    else:
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / int(total_steps), int(total_steps))
        schedule_sigmas = _numeric_schedule(sigmas, field="Flux.2 sigmas")
        sigmas = schedule_sigmas
    mu = empirical_mu(image_seq_len=int(latents.shape[1]), num_steps=int(total_steps))
    timesteps, resolved_steps = retrieve(
        pipeline.scheduler,
        int(total_steps),
        device,
        sigmas=sigmas,
        mu=mu,
    )
    if int(resolved_steps) != int(total_steps):
        raise PhaseError("Flux.2 scheduler changed the requested trajectory length")
    return timesteps, schedule_sigmas


def _flux2_scheduler_metadata(
    pipeline: Any,
    *,
    schedule_sigmas: list[float] | None,
) -> dict[str, Any]:
    """Capture the state that FlowMatch consumes in ``scheduler.step``."""

    scheduler = pipeline.scheduler
    scheduler_sigmas = _numeric_schedule(
        getattr(scheduler, "sigmas", None), field="Flux.2 scheduler sigmas"
    )
    if scheduler_sigmas is None:
        raise PhaseError("Flux.2 scheduler did not expose its installed sigma schedule")
    return {
        "scheduler_state_schema": "mrun-flux2-scheduler-state-v1",
        "scheduler_config_fingerprint": _scheduler_config_fingerprint(scheduler),
        "scheduler_num_inference_steps": int(
            getattr(scheduler, "num_inference_steps", len(scheduler_sigmas) - 1)
        ),
        "schedule_sigmas": schedule_sigmas,
        "scheduler_sigmas": scheduler_sigmas,
    }


def _restore_flux2_scheduler(
    pipeline: Any,
    checkpoint: TrajectoryCheckpoint,
    *,
    latents: Any,
    device: str,
) -> Any:
    """Reinstall and verify the exact scheduler state before a FLUX.2 suffix.

    Historical checkpoints did not persist scheduler state.  They are accepted
    only when the canonical default schedule reconstructs their saved timestep
    tensor exactly.  This keeps existing default-schedule handles usable while
    refusing to guess at an unrecoverable custom legacy schedule.
    """

    import torch

    scheduler = pipeline.scheduler
    metadata = dict(checkpoint.metadata)
    expected_class = metadata.get("scheduler_class")
    if expected_class and str(expected_class) != type(scheduler).__name__:
        raise PhaseError(
            "Flux.2 checkpoint scheduler class does not match the live pipeline: "
            f"{expected_class!r} != {type(scheduler).__name__!r}"
        )
    state_schema = metadata.get("scheduler_state_schema")
    if state_schema not in {None, "mrun-flux2-scheduler-state-v1"}:
        raise PhaseError(f"unsupported Flux.2 scheduler state schema {state_schema!r}")
    expected_config = metadata.get("scheduler_config_fingerprint")
    if expected_config and str(expected_config) != _scheduler_config_fingerprint(scheduler):
        raise PhaseError(
            "Flux.2 scheduler configuration changed since checkpoint capture; "
            "refusing a numerically different replay"
        )

    schedule_sigmas = metadata.get("schedule_sigmas") if state_schema else None
    scheduled, _ = _flux2_schedule(
        pipeline,
        latents,
        checkpoint.total_steps,
        device,
        sigmas=schedule_sigmas,
    )
    checkpoint_timesteps = checkpoint.timesteps.to(device)
    if not torch.equal(scheduled, checkpoint_timesteps):
        provenance = "legacy canonical" if state_schema is None else "captured"
        raise PhaseError(
            f"Flux.2 {provenance} scheduler schedule does not match the checkpoint; "
            "recapture the checkpoint with durable scheduler state"
        )

    if state_schema is not None:
        expected_steps = metadata.get("scheduler_num_inference_steps")
        actual_steps = getattr(scheduler, "num_inference_steps", checkpoint.total_steps)
        if expected_steps is not None and int(actual_steps) != int(expected_steps):
            raise PhaseError("Flux.2 scheduler inference-step count changed during restore")
        expected_sigmas = _numeric_schedule(
            metadata.get("scheduler_sigmas"), field="checkpoint scheduler sigmas"
        )
        actual_sigmas = getattr(scheduler, "sigmas", None)
        if expected_sigmas is None or not isinstance(actual_sigmas, torch.Tensor):
            raise PhaseError("Flux.2 checkpoint is missing a verifiable scheduler sigma state")
        expected_tensor = torch.tensor(
            expected_sigmas,
            dtype=actual_sigmas.dtype,
            device=actual_sigmas.device,
        )
        if not torch.equal(actual_sigmas, expected_tensor):
            raise PhaseError(
                "Flux.2 scheduler sigma state changed during restore; refusing replay"
            )
    return checkpoint_timesteps


def _reset_trajectory_scheduler(pipeline: Any, *, begin_index: int = 0) -> None:
    """Reset scheduler cursors after installing the checkpoint's full schedule."""

    scheduler = pipeline.scheduler
    if callable(getattr(scheduler, "set_begin_index", None)):
        scheduler.set_begin_index(int(begin_index))
    # Diffusers schedulers use a private cursor to avoid repeatedly searching
    # for the current timestep.  Replays must start by resolving the captured
    # timestep, not by inheriting the cursor left by the prefix or another
    # branch.
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = int(begin_index)


def _flux2_denoise_steps(
    pipeline: Any,
    *,
    latents: Any,
    latent_ids: Any,
    prompt_embeds: Any,
    text_ids: Any,
    timesteps: Any,
    start_step: int,
    end_step: int,
    guidance_scale: float,
    attention_kwargs: Mapping[str, Any],
    reference_latents: Any | None = None,
    reference_latent_ids: Any | None = None,
    reference_kv_cache: Any | None = None,
    kv_cache_out: dict[str, Any] | None = None,
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    """Run only a Flux2 denoise interval using the pipeline's own operators.

    ``step_observer`` is an optional debugger/training seam.  When supplied it
    receives a mapping after each scheduler step with the exact denoiser action
    and packed register transition consumed by the native pipeline.  The
    callback is intentionally synchronous and receives live tensors; callers
    that retain evidence must detach/clone them.  The default path is
    unchanged and pays no observation cost.
    """

    import torch

    if getattr(pipeline, "do_classifier_free_guidance", False):
        raise PhaseError(
            "trajectory replay currently requires guidance_scale <= 1; "
            "capture negative conditioning explicitly before enabling CFG"
        )
    transformer = pipeline.transformer
    context_factory = getattr(transformer, "cache_context", None)
    from contextlib import nullcontext
    native_kv = type(pipeline).__name__ == "Flux2KleinKVPipeline"
    kv_cache = (
        _clone_checkpoint_payload(reference_kv_cache)
        if reference_kv_cache is not None
        else None
    )

    with torch.inference_mode():
        for index in range(int(start_step), int(end_step)):
            timestep_value = timesteps[index]
            pipeline._current_timestep = timestep_value
            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
            latents_before = latents
            latent_model_input = latents.to(transformer.dtype)
            latent_model_ids = latent_ids
            with (
                context_factory("cond")
                if callable(context_factory)
                else nullcontext()
            ):
                if native_kv:
                    if kv_cache is not None:
                        # Native Klein-KV's cached ABI omits the reference
                        # tokens entirely and consumes the exact K/V object
                        # extracted from the ref-first step-0 call.
                        result = transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_model_ids,
                            joint_attention_kwargs=attention_kwargs,
                            return_dict=False,
                            kv_cache=kv_cache,
                            kv_cache_mode="cached",
                        )
                        noise_pred = result[0]
                    elif reference_latents is not None or reference_latent_ids is not None:
                        if reference_latents is None or reference_latent_ids is None:
                            raise PhaseError(
                                "reference_latents and reference_latent_ids must be supplied together"
                            )
                        if int(index) != 0:
                            raise PhaseError(
                                "native KV replay reached a reference-conditioned step without "
                                "the step-0 extracted cache; recapture the checkpoint"
                            )
                        # The native pipeline is explicitly ref-first.  The
                        # previous generic path was target-first and therefore
                        # changed attention positions before it ever reached
                        # the cache ABI.
                        latent_model_input = torch.cat(
                            (reference_latents.to(transformer.dtype), latent_model_input), dim=1
                        )
                        latent_model_ids = torch.cat((reference_latent_ids, latent_ids), dim=1)
                        result = transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_model_ids,
                            joint_attention_kwargs=attention_kwargs,
                            return_dict=False,
                            kv_cache_mode="extract",
                            num_ref_tokens=int(reference_latents.shape[1]),
                        )
                        noise_pred, kv_cache = result[0], result[1]
                        if kv_cache_out is not None:
                            kv_cache_out["value"] = _clone_checkpoint_payload(kv_cache)
                    else:
                        result = transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_model_ids,
                            joint_attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )
                        noise_pred = result[0]
                else:
                    if reference_latents is not None or reference_latent_ids is not None:
                        if reference_latents is None or reference_latent_ids is None:
                            raise PhaseError(
                                "reference_latents and reference_latent_ids must be supplied together"
                            )
                        latent_model_input = torch.cat(
                            (latent_model_input, reference_latents.to(transformer.dtype)), dim=1
                        )
                        latent_model_ids = torch.cat((latent_ids, reference_latent_ids), dim=1)
                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_model_ids,
                        joint_attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1) :]
            latent_dtype = latents.dtype
            latents = pipeline.scheduler.step(
                noise_pred,
                timestep_value,
                latents,
                return_dict=False,
            )[0]
            if latents.dtype != latent_dtype:
                latents = latents.to(latent_dtype)
            if step_observer is not None:
                step_observer(
                    {
                        "step_index": int(index),
                        "timestep": timestep_value,
                        "latents_before": latents_before,
                        "latent_model_input": latent_model_input,
                        "noise_pred": noise_pred,
                        "latents_after": latents,
                        "latent_ids": latent_ids,
                        "reference_latents": reference_latents,
                        "reference_latent_ids": reference_latent_ids,
                        "reference_kv_cache": kv_cache,
                    }
                )
    pipeline._current_timestep = None
    return latents


def _flux2_decode(
    pipeline: Any,
    latents: Any,
    latent_ids: Any,
    *,
    height: int,
    width: int,
    output_type: str,
) -> Any:
    """Apply the native Flux2 unpack, VAE normalization, and renderer."""

    import torch

    scale = int(pipeline.vae_scale_factor)
    latent_height = 2 * (int(height) // (scale * 2))
    latent_width = 2 * (int(width) // (scale * 2))
    if type(pipeline).__name__ == "Flux2KleinKVPipeline":
        # Klein-KV owns a two-argument id-scatter unpacker.  Calling the
        # generic Klein four-argument ABI here was the last decode-only shim
        # in the old worker and obscured the earlier trajectory mismatch.
        latents = pipeline._unpack_latents_with_ids(latents, latent_ids)
    else:
        latents = pipeline._unpack_latents_with_ids(
            latents,
            latent_ids,
            latent_height // 2,
            latent_width // 2,
        )
    latents_bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents_bn_std = torch.sqrt(
        pipeline.vae.bn.running_var.view(1, -1, 1, 1)
        + pipeline.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * latents_bn_std + latents_bn_mean
    latents = pipeline._unpatchify_latents(latents)
    if output_type == "latent":
        return latents
    if output_type not in {"pil", "np"}:
        raise PhaseError(f"unsupported trajectory output_type {output_type!r}")
    with torch.inference_mode():
        image = pipeline.vae.decode(latents, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type=output_type)


def _flux1_schedule(
    pipeline: Any,
    latents: Any,
    total_steps: int,
    device: str,
    sigmas: Any = None,
) -> Any:
    """Build the native FLUX.1 schedule used by ``FluxPipeline.__call__``."""

    import importlib

    import numpy as np

    module = importlib.import_module(type(pipeline).__module__)
    calculate_shift = getattr(module, "calculate_shift", None)
    retrieve_timesteps = getattr(module, "retrieve_timesteps", None)
    if not callable(calculate_shift) or not callable(retrieve_timesteps):
        try:
            from diffusers.pipelines.flux.pipeline_flux import (
                calculate_shift,
                retrieve_timesteps,
            )
        except ImportError as exc:
            raise PhaseError(
                "Flux.1 trajectory scheduling requires Diffusers' "
                "pipeline_flux.calculate_shift/retrieve_timesteps helpers"
            ) from exc

    if int(total_steps) <= 0:
        raise PhaseError("num_inference_steps must be positive")
    if sigmas is None:
        sigmas = np.linspace(1.0, 1 / int(total_steps), int(total_steps))
    config = pipeline.scheduler.config
    if bool(getattr(config, "use_flow_sigmas", False)):
        sigmas = None
    get = getattr(config, "get", None)
    if not callable(get):

        def get(key: str, default: Any = None) -> Any:
            return getattr(config, key, default)
    mu = calculate_shift(
        int(latents.shape[1]),
        get("base_image_seq_len", 256),
        get("max_image_seq_len", 4096),
        get("base_shift", 0.5),
        get("max_shift", 1.15),
    )
    timesteps, resolved_steps = retrieve_timesteps(
        pipeline.scheduler,
        int(total_steps),
        device,
        sigmas=sigmas,
        mu=mu,
    )
    if int(resolved_steps) != int(total_steps):
        raise PhaseError("Flux.1 scheduler changed the requested trajectory length")
    return timesteps


def _flux1_denoise_steps(
    pipeline: Any,
    *,
    latents: Any,
    latent_ids: Any,
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    text_ids: Any,
    timesteps: Any,
    start_step: int,
    end_step: int,
    guidance_scale: float,
    attention_kwargs: Mapping[str, Any],
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    """Run an exact native FLUX.1 denoise interval."""

    from contextlib import nullcontext

    import torch

    transformer = pipeline.transformer
    context_factory = getattr(transformer, "cache_context", None)
    guidance = None
    if bool(getattr(transformer.config, "guidance_embeds", False)):
        guidance = torch.full(
            (int(latents.shape[0]),),
            float(guidance_scale),
            device=latents.device,
            dtype=torch.float32,
        )
    try:
        with torch.inference_mode():
            for index in range(int(start_step), int(end_step)):
                timestep_value = timesteps[index]
                pipeline._current_timestep = timestep_value
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
                latents_before = latents
                context = context_factory("cond") if callable(context_factory) else nullcontext()
                with context:
                    noise_pred = transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=dict(attention_kwargs),
                        return_dict=False,
                    )[0]
                latent_dtype = latents.dtype
                latents = pipeline.scheduler.step(
                    noise_pred,
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]
                if latents.dtype != latent_dtype:
                    latents = latents.to(latent_dtype)
                if step_observer is not None:
                    step_observer(
                        {
                            "step_index": int(index),
                            "timestep": timestep_value,
                            "latents_before": latents_before,
                            "noise_pred": noise_pred,
                            "latents_after": latents,
                            "latent_ids": latent_ids,
                            "text_ids": text_ids,
                            "prompt_embeds": prompt_embeds,
                            "pooled_prompt_embeds": pooled_prompt_embeds,
                        }
                    )
    finally:
        pipeline._current_timestep = None
    return latents


def _flux1_decode(
    pipeline: Any,
    latents: Any,
    *,
    height: int,
    width: int,
    output_type: str,
) -> Any:
    """Apply the native FLUX.1 unpack, VAE normalization, and renderer."""

    import torch

    if output_type not in {"latent", "pil", "np"}:
        raise PhaseError(f"unsupported trajectory output_type {output_type!r}")
    if output_type == "latent":
        # Match FluxPipeline.__call__: latent output is the packed denoiser
        # register, while PIL/NP output crosses the native VAE unpack bridge.
        return latents
    unpacked = pipeline._unpack_latents(
        latents,
        int(height),
        int(width),
        pipeline.vae_scale_factor,
    )
    unpacked = (unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    vae = pipeline.vae
    target_device = str(unpacked.device)
    sequential_offload = bool(
        getattr(pipeline, "_saturn_sequential_cpu_offload", False)
    )
    moved = (
        vae
        if sequential_offload or type(pipeline).__name__ != "FluxPipeline"
        else vae.to(target_device)
    )
    if moved is not None and moved is not vae:
        pipeline.vae = moved
        vae = moved
    try:
        with torch.inference_mode():
            decode_input = (
                unpacked
                if sequential_offload
                else unpacked.to(next(vae.parameters()).device)
            )
            image = vae.decode(decode_input, return_dict=False)[0]
        return pipeline.image_processor.postprocess(image, output_type=output_type)
    finally:
        if (
            type(pipeline).__name__ == "FluxPipeline"
            and not sequential_offload
            and target_device.startswith("cuda")
        ):
            moved_back = vae.to("cpu")
            if moved_back is not None and moved_back is not vae:
                pipeline.vae = moved_back
            torch.cuda.empty_cache()


def _capture_flux1_checkpoint(
    phase: PhasePipeline,
    embeds: PromptEmbeds,
    *,
    cut_step: int,
    initial_latents: Any | None,
    reference_image: Any | None,
    references: Iterable[str],
    kwargs: Mapping[str, Any],
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> TrajectoryCheckpoint:
    """Capture a FLUX.1 prefix into the shared mrun checkpoint ABI."""

    import torch

    pipeline = _require_flux1_trajectory(phase)
    if not isinstance(embeds, PromptEmbeds):
        raise PhaseError("capture_checkpoint requires PromptEmbeds")
    params = dict(kwargs)
    params.pop("prompt", None)
    output_type = params.pop("output_type", "latent")
    if output_type not in {"latent", "pil", "np"}:
        raise PhaseError(f"unsupported trajectory output_type {output_type!r}")
    if params.pop("return_dict", True) is False:
        raise PhaseError("trajectory checkpoints always use the typed result path")
    image_alias = params.pop("image", None)
    if reference_image is not None or image_alias is not None:
        raise PhaseError("FLUX.1 trajectory checkpoints do not support reference images")
    if tuple(references):
        raise PhaseError("FLUX.1 trajectory checkpoints do not support reference ids")
    for feed_key in phase._contract.feed_keys:
        if feed_key in params:
            raise PhaseError(f"capture_checkpoint manages {feed_key!r}")
    device_feeds = phase._device_feed_tensors(embeds)
    prompt_embeds = device_feeds.get("prompt_embeds")
    pooled_prompt_embeds = device_feeds.get("pooled_prompt_embeds")
    if prompt_embeds is None or pooled_prompt_embeds is None:
        raise PhaseError(
            "FluxPipeline capture requires both prompt_embeds and pooled_prompt_embeds"
        )
    if int(prompt_embeds.shape[0]) != 1 or int(pooled_prompt_embeds.shape[0]) != 1:
        raise PhaseError("capture_checkpoint requires exactly one prompt embedding row")

    height = int(params.pop("height", pipeline.default_sample_size * pipeline.vae_scale_factor))
    width = int(params.pop("width", pipeline.default_sample_size * pipeline.vae_scale_factor))
    total_steps = int(params.pop("num_inference_steps", 50))
    guidance_scale = float(params.pop("guidance_scale", 3.5))
    attention_kwargs = params.pop("attention_kwargs", None) or {}
    sigmas = params.pop("sigmas", None)
    generator = params.pop("generator", None)
    if initial_latents is None:
        initial_latents = params.pop("latents", None)
    elif "latents" in params:
        raise PhaseError("provide initial_latents or latents, not both")
    if params:
        raise PhaseError(
            "unsupported trajectory options: " + ", ".join(sorted(map(str, params)))
        )
    if not 0 <= int(cut_step) <= total_steps:
        raise PhaseError(f"cut_step must be in [0, {total_steps}], got {cut_step}")
    if height <= 0 or width <= 0:
        raise PhaseError("trajectory resolution must be positive")

    if initial_latents is not None:
        if not isinstance(initial_latents, torch.Tensor):
            raise PhaseError("initial_latents must be a torch tensor")
        initial_latents = initial_latents.to(phase._device)
    channels = int(pipeline.transformer.config.in_channels) // 4
    # Diffusers' FluxPipeline treats a supplied ``latents`` tensor as already
    # packed, while Flux2's helper accepts the raw latent image shape.  Accept
    # both at the typed API boundary and normalize FLUX.1 raw 4-D noise with
    # the pipeline's own packer before calling prepare_latents().
    if initial_latents is not None and initial_latents.ndim == 4:
        pack = getattr(pipeline, "_pack_latents", None)
        if not callable(pack):
            raise PhaseError(
                "Flux.1 4-D initial_latents require the native _pack_latents() helper"
            )
        latent_height = 2 * (height // (int(pipeline.vae_scale_factor) * 2))
        latent_width = 2 * (width // (int(pipeline.vae_scale_factor) * 2))
        if tuple(initial_latents.shape) != (1, channels, latent_height, latent_width):
            raise PhaseError(
                "Flux.1 4-D initial_latents shape does not match the requested "
                f"resolution: got {tuple(initial_latents.shape)}, expected "
                f"(1, {channels}, {latent_height}, {latent_width})"
            )
        initial_latents = pack(
            initial_latents,
            1,
            channels,
            latent_height,
            latent_width,
        )
    latents, latent_ids = pipeline.prepare_latents(
        1,
        channels,
        height,
        width,
        prompt_embeds.dtype,
        phase._device,
        generator,
        initial_latents,
    )
    text_ids = torch.zeros(
        int(prompt_embeds.shape[1]),
        3,
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    timesteps = _flux1_schedule(pipeline, latents, total_steps, phase._device, sigmas=sigmas)
    schedule_sigmas = None
    if sigmas is not None:
        try:
            schedule_sigmas = [float(value) for value in sigmas]
        except (TypeError, ValueError):
            raise PhaseError("Flux.1 sigmas must be a finite numeric sequence") from None
    pipeline._guidance_scale = guidance_scale
    pipeline._attention_kwargs = dict(attention_kwargs)
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=0)
    latents = _flux1_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=0,
        end_step=int(cut_step),
        guidance_scale=guidance_scale,
        attention_kwargs=attention_kwargs,
        step_observer=step_observer,
    )
    identity = {
        "pipeline_class": type(pipeline).__name__,
        "step_index": int(cut_step),
        "total_steps": total_steps,
        "height": height,
        "width": width,
        "latent_fingerprint": _tensor_payload_fingerprint(latents),
        "prompt_fingerprint": _tensor_payload_fingerprint(prompt_embeds),
        "pooled_fingerprint": _tensor_payload_fingerprint(pooled_prompt_embeds),
        "schedule_fingerprint": _tensor_payload_fingerprint(timesteps),
    }
    checkpoint_id = "traj-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    return TrajectoryCheckpoint(
        checkpoint_id=checkpoint_id,
        pipeline_class=type(pipeline).__name__,
        step_index=int(cut_step),
        total_steps=total_steps,
        height=height,
        width=width,
        latents=latents,
        latent_ids=latent_ids,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        guidance_scale=guidance_scale,
        attention_kwargs=attention_kwargs,
        metadata={
            "output_type_requested_at_capture": output_type,
            "device": str(phase._device),
            "prefix_steps_executed": int(cut_step),
            "scheduler_class": type(pipeline.scheduler).__name__,
            "latent_shape": list(latents.shape),
            "conditioner_streams": ["prompt_embeds", "pooled_prompt_embeds"],
            "schedule_sigmas": schedule_sigmas,
        },
    )


def _capture_flux2_checkpoint(
    phase: PhasePipeline,
    embeds: PromptEmbeds,
    *,
    cut_step: int,
    initial_latents: Any | None,
    reference_image: Any | None,
    references: Iterable[str],
    kwargs: Mapping[str, Any],
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> TrajectoryCheckpoint:
    import torch

    pipeline = _require_flux2_trajectory(phase)
    if not isinstance(embeds, PromptEmbeds):
        raise PhaseError("capture_checkpoint requires PromptEmbeds")
    params = dict(kwargs)
    params.pop("prompt", None)
    output_type = params.pop("output_type", "latent")
    if output_type not in {"latent", "pil", "np"}:
        raise PhaseError(f"unsupported trajectory output_type {output_type!r}")
    if params.pop("return_dict", True) is False:
        raise PhaseError("trajectory checkpoints always use the typed result path")
    image_alias = params.pop("image", None)
    if reference_image is not None and image_alias is not None:
        raise PhaseError("provide reference_image or image, not both")
    if reference_image is None:
        reference_image = image_alias
    for feed_key in phase._contract.feed_keys:
        if feed_key != "prompt_embeds" and feed_key in params:
            raise PhaseError(f"capture_checkpoint manages {feed_key!r}")
    device_feeds = phase._device_feed_tensors(embeds)
    prompt_embeds = device_feeds.get("prompt_embeds")
    if prompt_embeds is None or int(prompt_embeds.shape[0]) != 1:
        raise PhaseError("capture_checkpoint requires exactly one prompt embedding row")
    text_ids = pipeline._prepare_text_ids(prompt_embeds).to(phase._device)

    height = int(params.pop("height", pipeline.default_sample_size * pipeline.vae_scale_factor))
    width = int(params.pop("width", pipeline.default_sample_size * pipeline.vae_scale_factor))
    total_steps = int(params.pop("num_inference_steps", 50))
    guidance_scale = float(params.pop("guidance_scale", 4.0))
    attention_kwargs = params.pop("attention_kwargs", None) or {}
    sigmas = params.pop("sigmas", None)
    generator = params.pop("generator", None)
    if initial_latents is None:
        initial_latents = params.pop("latents", None)
    elif "latents" in params:
        raise PhaseError("provide initial_latents or latents, not both")
    if params:
        raise PhaseError(
            "unsupported trajectory options: " + ", ".join(sorted(map(str, params)))
        )
    if not 0 <= int(cut_step) <= total_steps:
        raise PhaseError(f"cut_step must be in [0, {total_steps}], got {cut_step}")
    if guidance_scale > 1.0:
        raise PhaseError("trajectory capture currently supports guidance_scale <= 1")

    num_channels_latents = int(pipeline.transformer.config.in_channels) // 4
    if initial_latents is not None:
        if not isinstance(initial_latents, torch.Tensor):
            raise PhaseError("initial_latents must be a torch tensor")
        initial_latents = initial_latents.to(phase._device)
    latents, latent_ids = pipeline.prepare_latents(
        batch_size=1,
        num_latents_channels=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=phase._device,
        generator=generator,
        latents=initial_latents,
    )
    reference_latents = None
    reference_latent_ids = None
    reference_token_count = 0
    reference_ids = tuple(str(value) for value in references)
    if reference_image is not None:
        reference_images = (
            list(reference_image)
            if isinstance(reference_image, (list, tuple))
            else [reference_image]
        )
        if not reference_images:
            raise PhaseError("reference_image cannot be an empty sequence")
        prepared_images = [
            pipeline.image_processor.preprocess(
                image,
                height=height,
                width=width,
                resize_mode="crop",
            )
            for image in reference_images
        ]
        # Flux2's VAE reference path is argmax-deterministic for the pinned
        # Klein pipeline.  A dedicated generator keeps this ABI independent
        # from whether the caller supplied or consumed the noise generator.
        reference_generator = torch.Generator(device="cpu").manual_seed(0)
        reference_latents, reference_latent_ids = pipeline.prepare_image_latents(
            images=prepared_images,
            batch_size=1,
            generator=reference_generator,
            device=phase._device,
            dtype=prompt_embeds.dtype,
        )
        reference_token_count = int(reference_latents.shape[1])
    timesteps, schedule_sigmas = _flux2_schedule(
        pipeline,
        latents,
        total_steps,
        phase._device,
        sigmas=sigmas,
    )
    scheduler_metadata = _flux2_scheduler_metadata(
        pipeline,
        schedule_sigmas=schedule_sigmas,
    )
    if callable(getattr(pipeline.scheduler, "set_begin_index", None)):
        pipeline.scheduler.set_begin_index(0)
    pipeline._guidance_scale = guidance_scale
    pipeline._attention_kwargs = attention_kwargs
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=0)
    kv_cache_state: dict[str, Any] = {}
    latents = _flux2_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=0,
        end_step=int(cut_step),
        guidance_scale=guidance_scale,
        attention_kwargs=attention_kwargs,
        reference_latents=reference_latents,
        reference_latent_ids=reference_latent_ids,
        kv_cache_out=kv_cache_state,
        step_observer=step_observer,
    )
    reference_kv_cache = kv_cache_state.get("value")
    identity = {
        "pipeline_class": type(pipeline).__name__,
        "step_index": int(cut_step),
        "total_steps": total_steps,
        "height": height,
        "width": width,
        "latent_fingerprint": _tensor_payload_fingerprint(latents),
        "prompt_fingerprint": _tensor_payload_fingerprint(prompt_embeds),
        "schedule_fingerprint": _tensor_payload_fingerprint(timesteps),
        "reference_ids": reference_ids,
        "reference_latents": _tensor_payload_fingerprint(reference_latents),
        "reference_latent_ids": _tensor_payload_fingerprint(reference_latent_ids),
        "reference_token_count": reference_token_count,
        "reference_kv_cache": _checkpoint_payload_fingerprint(reference_kv_cache)
        if reference_kv_cache is not None
        else None,
    }
    checkpoint_id = "traj-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    return TrajectoryCheckpoint(
        checkpoint_id=checkpoint_id,
        pipeline_class=type(pipeline).__name__,
        step_index=int(cut_step),
        total_steps=total_steps,
        height=height,
        width=width,
        latents=latents,
        latent_ids=latent_ids,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        reference_ids=reference_ids,
        reference_latents=reference_latents,
        reference_latent_ids=reference_latent_ids,
        reference_token_count=reference_token_count,
        reference_kv_cache=reference_kv_cache,
        guidance_scale=guidance_scale,
        attention_kwargs=attention_kwargs,
        metadata={
            "output_type_requested_at_capture": output_type,
            "device": str(phase._device),
            "prefix_steps_executed": int(cut_step),
            "scheduler_class": type(pipeline.scheduler).__name__,
            "latent_shape": list(latents.shape),
            "reference_conditioned": bool(reference_token_count),
            "reference_token_count": reference_token_count,
            "reference_ids": list(reference_ids),
            "reference_kv_cache": reference_kv_cache is not None,
            **scheduler_metadata,
        },
    )


def _validate_trajectory_checkpoint(phase: PhasePipeline, checkpoint: TrajectoryCheckpoint) -> Any:
    if not isinstance(checkpoint, TrajectoryCheckpoint):
        raise PhaseError("resume requires a TrajectoryCheckpoint")
    pipeline = _require_flux2_trajectory(phase)
    if checkpoint.pipeline_class != type(pipeline).__name__:
        raise PhaseError(
            f"checkpoint pipeline {checkpoint.pipeline_class!r} does not match "
            f"{type(pipeline).__name__!r}"
        )
    return pipeline


def _prepare_flux2_checkpoint_state(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    pipeline = _validate_trajectory_checkpoint(phase, checkpoint)
    latents = checkpoint.latents.to(phase._device).clone()
    if latent_override is not None:
        if not isinstance(latent_override, torch.Tensor):
            raise PhaseError("latent_override must be a torch tensor")
        candidate = latent_override.to(phase._device)
        if tuple(candidate.shape) != tuple(latents.shape):
            raise PhaseError(
                f"latent_override shape {tuple(candidate.shape)} does not match "
                f"checkpoint shape {tuple(latents.shape)}"
            )
        latents = candidate.clone()
    latent_ids = checkpoint.latent_ids.to(phase._device)
    prompt_embeds = checkpoint.prompt_embeds.to(phase._device)
    text_ids = checkpoint.text_ids.to(phase._device)
    if prompt_embeds_override is not None:
        candidate = prompt_embeds_override
        if isinstance(candidate, PromptEmbeds):
            candidate = candidate.tensors.get("prompt_embeds")
        if not isinstance(candidate, torch.Tensor):
            raise PhaseError("prompt_embeds_override must be a torch tensor or PromptEmbeds")
        candidate = candidate.to(phase._device)
        if tuple(candidate.shape) != tuple(prompt_embeds.shape):
            raise PhaseError(
                f"prompt_embeds_override shape {tuple(candidate.shape)} does not match "
                f"checkpoint shape {tuple(prompt_embeds.shape)}"
            )
        if candidate.dtype != prompt_embeds.dtype:
            raise PhaseError(
                "prompt_embeds_override dtype "
                f"{candidate.dtype} does not match checkpoint dtype {prompt_embeds.dtype}"
            )
        candidate_text_ids = pipeline._prepare_text_ids(candidate).to(phase._device)
        if not torch.equal(candidate_text_ids, text_ids):
            raise PhaseError(
                "prompt_embeds_override changes positional text_ids; "
                "use a same-length conditioner with the checkpoint positional layout"
            )
        prompt_embeds = candidate.clone()
    timesteps = _restore_flux2_scheduler(
        pipeline,
        checkpoint,
        latents=latents,
        device=phase._device,
    )
    return pipeline, latents, latent_ids, prompt_embeds, text_ids, timesteps


def _prepare_flux1_checkpoint_state(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Resolve and validate the FLUX.1 latent plus dual conditioner state."""

    import torch

    if not isinstance(checkpoint, TrajectoryCheckpoint):
        raise PhaseError("resume requires a TrajectoryCheckpoint")
    pipeline = _require_flux1_trajectory(phase)
    if checkpoint.pipeline_class != type(pipeline).__name__:
        raise PhaseError(
            f"checkpoint pipeline {checkpoint.pipeline_class!r} does not match "
            f"{type(pipeline).__name__!r}"
        )
    if checkpoint.pooled_prompt_embeds is None:
        raise PhaseError(
            "FluxPipeline checkpoint is missing pooled_prompt_embeds; recapture it "
            "with the shared FLUX.1 trajectory ABI"
        )
    latents = checkpoint.latents.to(phase._device).clone()
    if latent_override is not None:
        if not isinstance(latent_override, torch.Tensor):
            raise PhaseError("latent_override must be a torch tensor")
        candidate = latent_override.to(phase._device)
        if tuple(candidate.shape) != tuple(latents.shape):
            raise PhaseError(
                f"latent_override shape {tuple(candidate.shape)} does not match "
                f"checkpoint shape {tuple(latents.shape)}"
            )
        latents = candidate.clone()
    latent_ids = checkpoint.latent_ids.to(phase._device)
    text_ids = checkpoint.text_ids.to(phase._device)
    prompt_embeds = checkpoint.prompt_embeds.to(phase._device)
    pooled_prompt_embeds = checkpoint.pooled_prompt_embeds.to(phase._device)
    if prompt_embeds_override is not None:
        candidate = prompt_embeds_override
        if isinstance(candidate, PromptEmbeds):
            tensors = candidate.tensors
        elif isinstance(candidate, Mapping):
            tensors = candidate
        else:
            raise PhaseError(
                "Flux.1 conditioner override must be PromptEmbeds or a mapping "
                "containing prompt_embeds and pooled_prompt_embeds"
            )
        candidate_prompt = tensors.get("prompt_embeds")
        candidate_pooled = tensors.get("pooled_prompt_embeds")
        if not isinstance(candidate_prompt, torch.Tensor) or not isinstance(
            candidate_pooled, torch.Tensor
        ):
            raise PhaseError(
                "Flux.1 conditioner override must contain both prompt_embeds "
                "and pooled_prompt_embeds tensors"
            )
        candidate_prompt = candidate_prompt.to(phase._device)
        candidate_pooled = candidate_pooled.to(phase._device)
        if tuple(candidate_prompt.shape) != tuple(prompt_embeds.shape):
            raise PhaseError(
                "Flux.1 prompt_embeds override shape does not match the checkpoint"
            )
        if tuple(candidate_pooled.shape) != tuple(pooled_prompt_embeds.shape):
            raise PhaseError(
                "Flux.1 pooled_prompt_embeds override shape does not match the checkpoint"
            )
        if (
            candidate_prompt.dtype != prompt_embeds.dtype
            or candidate_pooled.dtype != pooled_prompt_embeds.dtype
        ):
            raise PhaseError("Flux.1 conditioner override dtype does not match the checkpoint")
        prompt_embeds = candidate_prompt.clone()
        pooled_prompt_embeds = candidate_pooled.clone()
    timesteps = checkpoint.timesteps.to(phase._device)
    scheduled = _flux1_schedule(
        pipeline,
        latents,
        checkpoint.total_steps,
        phase._device,
        sigmas=checkpoint.metadata.get("schedule_sigmas"),
    )
    if not torch.equal(scheduled, timesteps):
        raise PhaseError(
            "Flux.1 scheduler schedule does not match the checkpoint; refusing "
            "to replay across a changed scheduler configuration"
        )
    return (
        pipeline,
        latents,
        latent_ids,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        timesteps,
    )


def _advance_flux1_checkpoint(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    steps: int,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> TrajectoryCheckpoint:
    if isinstance(steps, bool) or int(steps) <= 0:
        raise PhaseError("steps must be a positive integer")
    steps = int(steps)
    (
        pipeline,
        latents,
        latent_ids,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        timesteps,
    ) = _prepare_flux1_checkpoint_state(
        phase,
        checkpoint,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
    )
    start_step = int(checkpoint.step_index)
    end_step = start_step + steps
    if end_step > int(checkpoint.total_steps):
        raise PhaseError(
            f"cannot advance checkpoint from step {start_step} by {steps}; "
            f"trajectory ends at {checkpoint.total_steps}"
        )
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._attention_kwargs = dict(checkpoint.attention_kwargs)
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=start_step)
    latents = _flux1_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=start_step,
        end_step=end_step,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
        step_observer=step_observer,
    )
    metadata = dict(checkpoint.metadata)
    metadata.update(
        {
            "parent_checkpoint_id": str(checkpoint.checkpoint_id),
            "parent_checkpoint_fingerprint": str(checkpoint.fingerprint),
            "source_step_index": start_step,
            "steps_advanced": steps,
            "transition": "mrun.diffusion.advance_checkpoint:v1",
        }
    )
    identity = {
        "schema": checkpoint.schema,
        "parent_checkpoint_fingerprint": checkpoint.fingerprint,
        "step_index": end_step,
        "total_steps": int(checkpoint.total_steps),
        "latent_fingerprint": _tensor_payload_fingerprint(latents),
        "prompt_fingerprint": _tensor_payload_fingerprint(prompt_embeds),
        "pooled_fingerprint": _tensor_payload_fingerprint(pooled_prompt_embeds),
        "schedule_fingerprint": _tensor_payload_fingerprint(timesteps),
        "metadata": metadata,
    }
    checkpoint_id = "traj-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    return TrajectoryCheckpoint(
        checkpoint_id=checkpoint_id,
        pipeline_class=checkpoint.pipeline_class,
        step_index=end_step,
        total_steps=checkpoint.total_steps,
        height=checkpoint.height,
        width=checkpoint.width,
        latents=latents,
        latent_ids=latent_ids,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
        metadata=metadata,
        schema=checkpoint.schema,
    )


def _resume_flux1_checkpoint(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    output_type: str,
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    (
        pipeline,
        latents,
        latent_ids,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        timesteps,
    ) = _prepare_flux1_checkpoint_state(
        phase,
        checkpoint,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
    )
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._attention_kwargs = dict(checkpoint.attention_kwargs)
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=checkpoint.step_index)
    latents = _flux1_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=checkpoint.step_index,
        end_step=checkpoint.total_steps,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
        step_observer=step_observer,
    )
    images = _flux1_decode(
        pipeline,
        latents,
        height=checkpoint.height,
        width=checkpoint.width,
        output_type=output_type,
    )
    return SimpleNamespace(images=images)


def _resume_flux1_checkpoint_batch(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    branch_ids: Sequence[str],
    latent_overrides: Any | Sequence[Any] | None,
    mode: str,
    output_type: str,
) -> PhaseBatchResult:
    """Replay FLUX.1 branches exactly or with a fused suffix."""

    import torch

    ids = tuple(str(value) for value in branch_ids)
    if not ids or len(set(ids)) != len(ids):
        raise PhaseError("resume_checkpoint_batch branch_ids must be non-empty and unique")
    if mode not in {"exact", "batched"}:
        raise PhaseError("resume_checkpoint_batch mode must be 'exact' or 'batched'")
    if mode == "batched" and checkpoint.reference_kv_cache is not None:
        raise PhaseError(
            "native Klein-KV trajectory replay is scalar-only until the nested "
            "reference K/V batch ABI is registered"
        )
    overrides = _normalise_latent_overrides(latent_overrides, batch_size=len(ids))
    if mode == "exact":
        rows = []
        for override in overrides:
            result = _resume_flux1_checkpoint(
                phase,
                checkpoint,
                latent_override=override,
                prompt_embeds_override=None,
                output_type=output_type,
            )
            value = result.images
            if isinstance(value, (list, tuple)):
                rows.append(value[0])
            elif isinstance(value, torch.Tensor) and value.shape[0] == 1:
                rows.append(value[0])
            else:
                rows.append(value)
        result = PhaseBatchResult(
            output=SimpleNamespace(images=rows),
            branch_ids=ids,
            batch_size=len(ids),
            telemetry={
                "execution_mode": "exact_scalar_suffix",
                "numerical_contract": "scalar-authority",
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_fingerprint": checkpoint.fingerprint,
                "shared_prefix_reused": True,
                "shared_prefix_steps": checkpoint.step_index,
                "reused_prefix_denoiser_calls": checkpoint.step_index,
                "branch_suffix_steps": (checkpoint.total_steps - checkpoint.step_index) * len(ids),
                "physical_prefix_denoiser_calls": 0,
                "physical_suffix_denoiser_calls": (
                    checkpoint.total_steps - checkpoint.step_index
                )
                * len(ids),
                "total_physical_denoiser_calls": checkpoint.step_index
                + (checkpoint.total_steps - checkpoint.step_index) * len(ids),
                "logical_branches": len(ids),
                "revert_count": len(ids),
                "output_type": output_type,
            },
        )
        result.row_outputs()
        return result

    (
        pipeline,
        base,
        base_latent_ids,
        prompt_embeds,
        pooled_prompt_embeds,
        base_text_ids,
        timesteps,
    ) = _prepare_flux1_checkpoint_state(
        phase,
        checkpoint,
        latent_override=None,
        prompt_embeds_override=None,
    )
    rows = []
    for override in overrides:
        if override is None:
            rows.append(base.clone())
            continue
        if not isinstance(override, torch.Tensor):
            raise PhaseError("latent_overrides must contain torch tensors")
        candidate = override.to(phase._device)
        if tuple(candidate.shape) != tuple(base.shape):
            raise PhaseError("latent override shape does not match checkpoint latents")
        rows.append(candidate.clone())
    latents = torch.cat(rows, dim=0)
    prompt_embeds = prompt_embeds.expand(len(ids), *([-1] * (prompt_embeds.ndim - 1)))
    pooled_prompt_embeds = pooled_prompt_embeds.expand(
        len(ids), *([-1] * (pooled_prompt_embeds.ndim - 1))
    )
    # Diffusers' FLUX.1 latent/text ids are shared across the batch and are
    # therefore commonly rank-2.  Preserve that ABI; test doubles and future
    # pipeline revisions that return a batch dimension are handled as well.
    if base_latent_ids.ndim == 2:
        latent_ids = base_latent_ids
    else:
        latent_ids = base_latent_ids.expand(len(ids), *([-1] * (base_latent_ids.ndim - 1)))
    if base_text_ids.ndim == 2:
        text_ids = base_text_ids
    else:
        text_ids = base_text_ids.expand(len(ids), *([-1] * (base_text_ids.ndim - 1)))
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._attention_kwargs = dict(checkpoint.attention_kwargs)
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=checkpoint.step_index)
    latents = _flux1_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=checkpoint.step_index,
        end_step=checkpoint.total_steps,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
    )
    images = _flux1_decode(
        pipeline,
        latents,
        height=checkpoint.height,
        width=checkpoint.width,
        output_type=output_type,
    )
    suffix_steps = checkpoint.total_steps - checkpoint.step_index
    result = PhaseBatchResult(
        output=SimpleNamespace(images=images),
        branch_ids=ids,
        batch_size=len(ids),
        telemetry={
            "execution_mode": "batched_suffix",
            "numerical_contract": "batch-dependent",
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_fingerprint": checkpoint.fingerprint,
            "shared_prefix_reused": True,
            "shared_prefix_steps": checkpoint.step_index,
            "reused_prefix_denoiser_calls": checkpoint.step_index,
            "branch_suffix_steps": suffix_steps * len(ids),
            "physical_prefix_denoiser_calls": 0,
            "physical_suffix_denoiser_calls": suffix_steps,
            "total_physical_denoiser_calls": checkpoint.step_index + suffix_steps,
            "logical_branches": len(ids),
            "revert_count": len(ids),
            "output_type": output_type,
        },
    )
    result.row_outputs()
    return result


def _advance_flux2_checkpoint(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    steps: int,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> TrajectoryCheckpoint:
    if isinstance(steps, bool) or int(steps) <= 0:
        raise PhaseError("steps must be a positive integer")
    steps = int(steps)
    pipeline, latents, latent_ids, prompt_embeds, text_ids, timesteps = (
        _prepare_flux2_checkpoint_state(
            phase,
            checkpoint,
            latent_override=latent_override,
            prompt_embeds_override=prompt_embeds_override,
        )
    )
    start_step = int(checkpoint.step_index)
    end_step = start_step + steps
    if end_step > int(checkpoint.total_steps):
        raise PhaseError(
            f"cannot advance checkpoint from step {start_step} by {steps}; "
            f"trajectory ends at {checkpoint.total_steps}"
        )
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._attention_kwargs = dict(checkpoint.attention_kwargs)
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=start_step)
    kv_cache_state: dict[str, Any] = {}
    latents = _flux2_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=start_step,
        end_step=end_step,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
        reference_latents=(
            checkpoint.reference_latents.to(phase._device)
            if checkpoint.reference_latents is not None
            and checkpoint.reference_kv_cache is None
            else None
        ),
        reference_latent_ids=(
            checkpoint.reference_latent_ids.to(phase._device)
            if checkpoint.reference_latent_ids is not None
            and checkpoint.reference_kv_cache is None
            else None
        ),
        reference_kv_cache=(
            _move_checkpoint_payload(checkpoint.reference_kv_cache, phase._device)
            if checkpoint.reference_kv_cache is not None
            else None
        ),
        kv_cache_out=kv_cache_state,
        step_observer=step_observer,
    )
    next_kv_cache = kv_cache_state.get("value", checkpoint.reference_kv_cache)
    metadata = dict(checkpoint.metadata)
    metadata.update(
        {
            "parent_checkpoint_id": str(checkpoint.checkpoint_id),
            "parent_checkpoint_fingerprint": str(checkpoint.fingerprint),
            "source_step_index": start_step,
            "steps_advanced": steps,
            "transition": "mrun.diffusion.advance_checkpoint:v1",
        }
    )
    identity = {
        "schema": checkpoint.schema,
        "parent_checkpoint_fingerprint": checkpoint.fingerprint,
        "step_index": end_step,
        "total_steps": int(checkpoint.total_steps),
        "latent_fingerprint": _tensor_payload_fingerprint(latents),
        "prompt_fingerprint": _tensor_payload_fingerprint(prompt_embeds),
        "schedule_fingerprint": _tensor_payload_fingerprint(timesteps),
        "metadata": metadata,
    }
    checkpoint_id = "traj-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    return TrajectoryCheckpoint(
        checkpoint_id=checkpoint_id,
        pipeline_class=checkpoint.pipeline_class,
        step_index=end_step,
        total_steps=checkpoint.total_steps,
        height=checkpoint.height,
        width=checkpoint.width,
        latents=latents,
        latent_ids=latent_ids,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        reference_ids=checkpoint.reference_ids,
        reference_latents=checkpoint.reference_latents,
        reference_latent_ids=checkpoint.reference_latent_ids,
        reference_token_count=checkpoint.reference_token_count,
        reference_kv_cache=next_kv_cache,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
        metadata=metadata,
        schema=checkpoint.schema,
    )


def _resume_flux2_checkpoint(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    output_type: str,
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    pipeline, latents, latent_ids, prompt_embeds, text_ids, timesteps = (
        _prepare_flux2_checkpoint_state(
            phase,
            checkpoint,
            latent_override=latent_override,
            prompt_embeds_override=prompt_embeds_override,
        )
    )
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._attention_kwargs = dict(checkpoint.attention_kwargs)
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=checkpoint.step_index)
    latents = _flux2_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=checkpoint.step_index,
        end_step=checkpoint.total_steps,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
        reference_latents=(
            checkpoint.reference_latents.to(phase._device)
            if checkpoint.reference_latents is not None
            and checkpoint.reference_kv_cache is None
            else None
        ),
        reference_latent_ids=(
            checkpoint.reference_latent_ids.to(phase._device)
            if checkpoint.reference_latent_ids is not None
            and checkpoint.reference_kv_cache is None
            else None
        ),
        reference_kv_cache=(
            _move_checkpoint_payload(checkpoint.reference_kv_cache, phase._device)
            if checkpoint.reference_kv_cache is not None
            else None
        ),
        step_observer=step_observer,
    )
    images = _flux2_decode(
        pipeline,
        latents,
        latent_ids,
        height=checkpoint.height,
        width=checkpoint.width,
        output_type=output_type,
    )
    return SimpleNamespace(images=images)


def _normalise_latent_overrides(
    latent_overrides: Any | Sequence[Any] | None,
    *,
    batch_size: int,
) -> list[Any | None]:
    if latent_overrides is None:
        return [None] * batch_size
    import torch

    if isinstance(latent_overrides, torch.Tensor):
        if latent_overrides.ndim == 0 or int(latent_overrides.shape[0]) != batch_size:
            raise PhaseError("batched latent_overrides must align with branch_ids")
        return [latent_overrides[index : index + 1] for index in range(batch_size)]
    values = list(latent_overrides)
    if len(values) != batch_size:
        raise PhaseError("latent_overrides must align with branch_ids")
    return values


def _resume_flux2_checkpoint_batch(
    phase: PhasePipeline,
    checkpoint: TrajectoryCheckpoint,
    *,
    branch_ids: Sequence[str],
    latent_overrides: Any | Sequence[Any] | None,
    mode: str,
    output_type: str,
) -> PhaseBatchResult:
    import torch

    pipeline = _validate_trajectory_checkpoint(phase, checkpoint)
    ids = tuple(str(value) for value in branch_ids)
    if not ids or len(set(ids)) != len(ids):
        raise PhaseError("resume_checkpoint_batch branch_ids must be non-empty and unique")
    if mode not in {"exact", "batched"}:
        raise PhaseError("resume_checkpoint_batch mode must be 'exact' or 'batched'")
    overrides = _normalise_latent_overrides(latent_overrides, batch_size=len(ids))
    if mode == "exact":
        rows = []
        for override in overrides:
            result = _resume_flux2_checkpoint(
                phase,
                checkpoint,
                latent_override=override,
                prompt_embeds_override=None,
                output_type=output_type,
            )
            value = result.images
            if isinstance(value, (list, tuple)):
                rows.append(value[0])
            elif isinstance(value, torch.Tensor) and value.shape[0] == 1:
                rows.append(value[0])
            else:
                rows.append(value)
        output = SimpleNamespace(images=rows)
        telemetry = {
            "execution_mode": "exact_scalar_suffix",
            "numerical_contract": "scalar-authority",
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_fingerprint": checkpoint.fingerprint,
            "shared_prefix_reused": True,
            "shared_prefix_steps": checkpoint.step_index,
            "reused_prefix_denoiser_calls": checkpoint.step_index,
            "branch_suffix_steps": (checkpoint.total_steps - checkpoint.step_index) * len(ids),
            "physical_prefix_denoiser_calls": 0,
            "physical_suffix_denoiser_calls": (
                checkpoint.total_steps - checkpoint.step_index
            )
            * len(ids),
            "total_physical_denoiser_calls": checkpoint.step_index
            + (checkpoint.total_steps - checkpoint.step_index) * len(ids),
            "logical_branches": len(ids),
            "revert_count": len(ids),
            "output_type": output_type,
        }
        result = PhaseBatchResult(
            output=output,
            branch_ids=ids,
            batch_size=len(ids),
            telemetry=telemetry,
        )
        result.row_outputs()
        return result

    pipeline, base, base_latent_ids, base_prompt_embeds, base_text_ids, timesteps = (
        _prepare_flux2_checkpoint_state(
            phase,
            checkpoint,
            latent_override=None,
            prompt_embeds_override=None,
        )
    )
    rows = []
    for override in overrides:
        if override is None:
            rows.append(base.clone())
            continue
        if not isinstance(override, torch.Tensor):
            raise PhaseError("latent_overrides must contain torch tensors")
        candidate = override.to(phase._device)
        if tuple(candidate.shape) != tuple(base.shape):
            raise PhaseError("latent override shape does not match checkpoint latents")
        rows.append(candidate.clone())
    latents = torch.cat(rows, dim=0)
    latent_ids = base_latent_ids.expand(len(ids), -1, -1)
    prompt_embeds = base_prompt_embeds.expand(len(ids), -1, -1)
    text_ids = base_text_ids.expand(len(ids), -1, -1)
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._attention_kwargs = dict(checkpoint.attention_kwargs)
    pipeline._interrupt = False
    _reset_trajectory_scheduler(pipeline, begin_index=checkpoint.step_index)
    latents = _flux2_denoise_steps(
        pipeline,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        timesteps=timesteps,
        start_step=checkpoint.step_index,
        end_step=checkpoint.total_steps,
        guidance_scale=checkpoint.guidance_scale,
        attention_kwargs=checkpoint.attention_kwargs,
        reference_latents=(
            checkpoint.reference_latents.to(phase._device).expand(len(ids), -1, -1)
            if checkpoint.reference_latents is not None
            and checkpoint.reference_kv_cache is None
            else None
        ),
        reference_latent_ids=(
            checkpoint.reference_latent_ids.to(phase._device).expand(len(ids), -1, -1)
            if checkpoint.reference_latent_ids is not None
            and checkpoint.reference_kv_cache is None
            else None
        ),
    )
    images = _flux2_decode(
        pipeline,
        latents,
        latent_ids,
        height=checkpoint.height,
        width=checkpoint.width,
        output_type=output_type,
    )
    output = SimpleNamespace(images=images)
    suffix_steps = checkpoint.total_steps - checkpoint.step_index
    telemetry = {
        "execution_mode": "batched_suffix",
        "numerical_contract": "batch-dependent",
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_fingerprint": checkpoint.fingerprint,
        "shared_prefix_reused": True,
        "shared_prefix_steps": checkpoint.step_index,
        "reused_prefix_denoiser_calls": checkpoint.step_index,
        "branch_suffix_steps": suffix_steps * len(ids),
        "physical_prefix_denoiser_calls": 0,
        "physical_suffix_denoiser_calls": suffix_steps,
        "total_physical_denoiser_calls": checkpoint.step_index + suffix_steps,
        "logical_branches": len(ids),
        "revert_count": len(ids),
        "output_type": output_type,
    }
    result = PhaseBatchResult(
        output=output,
        branch_ids=ids,
        batch_size=len(ids),
        telemetry=telemetry,
    )
    result.row_outputs()
    return result
