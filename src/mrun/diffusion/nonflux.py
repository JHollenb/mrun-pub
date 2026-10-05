"""Native trajectory execution for the non-FLUX diffusion families.

The first trajectory ABI was intentionally shaped around FLUX.  That is not a
safe shape for SDXL (which has no FLUX positional ids) or for the newer packed
Krea/Chroma pipelines (which have different masks and image layouts).  This
module keeps those families behind an explicit, small dispatch table and uses
a slot map for the state at a denoising boundary.

There is deliberately no diffusers import at module scope.  Apart from making
the mrun scheduler importable without an accelerator stack, this makes the
slot-map object useful to a persistence adapter without importing a pipeline
implementation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from types import MappingProxyType, SimpleNamespace
from typing import Any

NON_FLUX_PIPELINES: Mapping[str, str] = MappingProxyType(
    {
        "StableDiffusionXLPipeline": "sdxl",
        "Krea2Pipeline": "krea2",
        "ChromaPipeline": "chroma",
    }
)

NON_FLUX_STATE_SCHEMA = "mrun-diffusion-trajectory-state-v1"
NON_FLUX_CHECKPOINT_SCHEMA = "mrun-diffusion-trajectory-checkpoint-v2"
_MISSING = object()


def _torch() -> Any:
    import torch

    return torch


def _inference_function(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run a native trajectory operation with autograd disabled end-to-end."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _torch().inference_mode():
            return function(*args, **kwargs)

    return wrapped


def _error(message: str) -> None:
    # Keep the module importable independently of phase.py.  The public error
    # type is imported only when an operation actually needs to fail.
    from .phase import PhaseError

    raise PhaseError(message)


def _is_tensor(value: Any) -> bool:
    try:
        return isinstance(value, _torch().Tensor)
    except ImportError:  # pragma: no cover - torch is a mrun dependency
        return False


def _clone(value: Any) -> Any:
    if _is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {str(key): _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _to_device(value: Any, device: Any) -> Any:
    if _is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _jsonable(value: Any, *, field_name: str = "value") -> Any:
    """Normalize the JSON half of a checkpoint without process-local reprs."""

    if _is_tensor(value):
        # Tensor values are normally slots, but scheduler metadata can contain
        # an explicitly supplied tensor schedule.  Store its JSON projection
        # in metadata while retaining the typed tensor in the slot map.
        return _jsonable(value.detach().to(device="cpu").tolist(), field_name=field_name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _error(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item, field_name=f"{field_name}[]") for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _error(f"{field_name} contains a non-string key")
            result[key] = _jsonable(item, field_name=f"{field_name}.{key}")
        return result
    # Numpy arrays/scalars are common in explicit schedules and scheduler
    # configs but are not part of the ABI.  Convert values that explicitly
    # advertise a JSON projection; arbitrary objects remain a fail-closed
    # error.
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            converted = to_list()
            if converted is not value:
                return _jsonable(converted, field_name=field_name)
        except (TypeError, ValueError, OverflowError):
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item(), field_name=field_name)
        except (TypeError, ValueError, OverflowError):
            pass
    _error(f"{field_name} contains unsupported value {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _tensor_fingerprint(value: Any) -> str:
    if not _is_tensor(value):
        return hashlib.sha256(
            json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in tensor.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _value_fingerprint(value: Any) -> Any:
    if _is_tensor(value):
        return {"tensor": _tensor_fingerprint(value)}
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [str(key), _value_fingerprint(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    if isinstance(value, tuple):
        return {"tuple": [_value_fingerprint(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_value_fingerprint(item) for item in value]}
    return {"json": _jsonable(value)}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class DiffusionTrajectoryState:
    """Family-specific state slots at one denoising boundary.

    Tensor slots are cloned on construction.  JSON slots are normalized and
    detached so advancing a branch cannot mutate its immutable parent.  Slot
    names intentionally match the state-cut identifier grammar and therefore
    can be handed to Saturn's ``StateCutCheckpoint`` through
    :meth:`to_statecut_payload`.
    """

    family: str
    slots: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = NON_FLUX_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NON_FLUX_STATE_SCHEMA:
            _error(f"unsupported diffusion trajectory state schema {self.schema!r}")
        if not isinstance(self.family, str) or not self.family:
            _error("diffusion trajectory state requires a family")
        if not isinstance(self.slots, Mapping) or not self.slots:
            _error("diffusion trajectory state requires at least one slot")
        copied: dict[str, Any] = {}
        for raw_name, value in self.slots.items():
            name = str(raw_name)
            if not name or any(ch.isspace() for ch in name):
                _error(f"invalid diffusion trajectory slot name {name!r}")
            copied[name] = (
                _clone(value)
                if _is_tensor(value)
                else _freeze_json(_jsonable(value, field_name=name))
            )
        metadata = _jsonable(dict(self.metadata), field_name="metadata")
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "slots", MappingProxyType(copied))
        object.__setattr__(self, "metadata", _freeze_json(metadata))

    def slot(self, name: str, default: Any = None) -> Any:
        return self.slots.get(name, default)

    @property
    def fingerprint(self) -> str:
        slots = {name: _value_fingerprint(self.slots[name]) for name in sorted(self.slots)}
        return _digest(
            {
                "schema": self.schema,
                "family": self.family,
                "slots": slots,
                "metadata": _jsonable(self.metadata),
            }
        )

    def metadata_only(self) -> dict[str, Any]:
        records: dict[str, Any] = {}
        for name, value in self.slots.items():
            if _is_tensor(value):
                records[name] = {
                    "kind": "tensor",
                    "shape": [int(item) for item in value.shape],
                    "dtype": str(value.dtype),
                }
            else:
                records[name] = {"kind": "json", "value": _jsonable(value)}
        return {
            "schema": self.schema,
            "family": self.family,
            "fingerprint": self.fingerprint,
            "slots": records,
            "metadata": _jsonable(self.metadata),
        }

    def to_statecut_payload(
        self,
        *,
        checkpoint_id: str,
        step_index: int,
        total_steps: int,
        model_identity: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return constructor kwargs for Saturn's family-neutral state cut.

        The converter is dependency-free: callers in Saturn can use
        ``StateCutCheckpoint(**state.to_statecut_payload(...))``.  Keeping the
        conversion here gives mrun persistence a stable transport story while
        preserving the dependency direction (mrun never imports Saturn).
        """

        if not isinstance(model_identity, Mapping) or not model_identity:
            _error("state-cut model_identity must be a non-empty mapping")

        specs: dict[str, dict[str, Any]] = {}
        for name, value in self.slots.items():
            if _is_tensor(value):
                specs[name] = {
                    "kind": "tensor",
                    "required": True,
                    "shape_class": f"rank_{len(value.shape)}",
                    "dtype": str(value.dtype),
                }
            else:
                specs[name] = {
                    "kind": "json",
                    "required": True,
                    "shape_class": "json_structure",
                    "dtype": "json",
                }
        state_metadata = _jsonable(self.metadata)
        merged_metadata = dict(state_metadata)
        if metadata:
            merged_metadata.update(_jsonable(dict(metadata)))
        # Keep an unambiguous copy of the state metadata beside the checkpoint
        # envelope.  Family state already uses names such as ``guidance_scale``
        # and ``attention_kwargs``; flattening the envelope over those names
        # would make a fresh-process fingerprint round-trip ambiguous.
        merged_metadata["state_metadata"] = state_metadata
        return {
            "checkpoint_id": str(checkpoint_id),
            "family": self.family,
            "state_schema_id": self.schema,
            "model_identity": _jsonable(dict(model_identity), field_name="model_identity"),
            "boundary": {"kind": "step_cut", "index": int(step_index), "total": int(total_steps)},
            "slots": {name: _clone(value) for name, value in self.slots.items()},
            "slot_specs": specs,
            "metadata": merged_metadata,
        }


@dataclass(frozen=True, slots=True)
class DiffusionTrajectoryCheckpoint:
    """Immutable non-FLUX trajectory checkpoint using a typed slot map."""

    checkpoint_id: str
    pipeline_class: str
    step_index: int
    total_steps: int
    height: int
    width: int
    state: DiffusionTrajectoryState
    guidance_scale: float = 1.0
    attention_kwargs: Mapping[str, Any] | None = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = NON_FLUX_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NON_FLUX_CHECKPOINT_SCHEMA:
            _error(f"unsupported diffusion trajectory checkpoint schema {self.schema!r}")
        if not self.checkpoint_id or not self.pipeline_class:
            _error("diffusion trajectory checkpoint requires an id and pipeline class")
        if isinstance(self.step_index, bool) or isinstance(self.total_steps, bool):
            _error("trajectory step indexes must be integers")
        if int(self.total_steps) <= 0 or not 0 <= int(self.step_index) <= int(self.total_steps):
            _error("trajectory checkpoint step_index is outside the schedule")
        if int(self.height) <= 0 or int(self.width) <= 0:
            _error("trajectory checkpoint resolution must be positive")
        if not isinstance(self.state, DiffusionTrajectoryState):
            _error("diffusion trajectory checkpoint requires DiffusionTrajectoryState")
        expected_family = NON_FLUX_PIPELINES.get(str(self.pipeline_class))
        if expected_family is None:
            _error(f"unsupported non-FLUX checkpoint pipeline class {self.pipeline_class!r}")
        if self.state.family != expected_family:
            _error(
                f"checkpoint pipeline class {self.pipeline_class!r} maps to "
                f"{expected_family!r}, not family {self.state.family!r}"
            )
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "total_steps", int(self.total_steps))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "guidance_scale", float(self.guidance_scale))
        if self.attention_kwargs is None:
            normalized_attention_kwargs = None
        else:
            normalized_attention_kwargs = _freeze_json(
                _jsonable(dict(self.attention_kwargs), field_name="attention_kwargs")
            )
        object.__setattr__(self, "attention_kwargs", normalized_attention_kwargs)
        object.__setattr__(self, "metadata", _freeze_json(_jsonable(dict(self.metadata))))

    @property
    def family(self) -> str:
        return self.state.family

    @property
    def slots(self) -> Mapping[str, Any]:
        return self.state.slots

    def slot(self, name: str, default: Any = None) -> Any:
        return self.state.slot(name, default)

    @property
    def scheduler_begin_index(self) -> Any:
        return self.state.metadata.get("scheduler_begin_index")

    @property
    def scheduler_step_index(self) -> Any:
        return self.state.metadata.get("scheduler_step_index")

    @property
    def generator_state(self) -> Any:
        return self.slot("rng.generator_state")

    # Compatibility conveniences for generic callers.  Unlike the v1 ABI,
    # absent FLUX positional ids remain absent rather than being fabricated.
    @property
    def latents(self) -> Any:
        return self.slot("latents")

    @property
    def timesteps(self) -> Any:
        return self.slot("schedule_timesteps")

    @property
    def prompt_embeds(self) -> Any:
        return self.slot("condition_prompt_embeds")

    @property
    def latent_ids(self) -> Any:
        return self.slot("layout_latent_image_ids")

    @property
    def text_ids(self) -> Any:
        return self.slot("condition_text_ids")

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "schema": self.schema,
                "checkpoint_id": self.checkpoint_id,
                "pipeline_class": self.pipeline_class,
                "step_index": self.step_index,
                "total_steps": self.total_steps,
                "height": self.height,
                "width": self.width,
                "state": self.state.fingerprint,
                "guidance_scale": self.guidance_scale,
                "attention_kwargs": _jsonable(self.attention_kwargs),
                "metadata": _jsonable(self.metadata),
            }
        )

    def metadata_only(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "fingerprint": self.fingerprint,
            "pipeline_class": self.pipeline_class,
            "family": self.family,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "resolution": [self.height, self.width],
            "state": self.state.metadata_only(),
            "guidance_scale": self.guidance_scale,
            "metadata": _jsonable(self.metadata),
        }

    def to_statecut_payload(
        self,
        *,
        model_identity: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # ``DiffusionTrajectoryState.to_statecut_payload`` always retains the
        # state's own metadata.  Add checkpoint envelope fields separately so
        # reconstructing the v2 object does not accidentally promote
        # checkpoint-level metadata into the state fingerprint.
        checkpoint_attention_kwargs = (
            None if self.attention_kwargs is None else dict(self.attention_kwargs)
        )
        merged = {
            "pipeline_class": self.pipeline_class,
            "height": self.height,
            "width": self.width,
            "guidance_scale": self.guidance_scale,
            "attention_kwargs": checkpoint_attention_kwargs,
            # Prefix aliases are retained for payload readers that consumed
            # the first v2 draft before the generic envelope was finalized.
            "checkpoint_guidance_scale": self.guidance_scale,
            "checkpoint_attention_kwargs": checkpoint_attention_kwargs,
            "checkpoint_fingerprint": self.fingerprint,
            "checkpoint_metadata": dict(self.metadata),
        }
        if metadata:
            merged.update(dict(metadata))
        return self.state.to_statecut_payload(
            checkpoint_id=self.checkpoint_id,
            step_index=self.step_index,
            total_steps=self.total_steps,
            model_identity=model_identity,
            metadata=merged,
        )

    @classmethod
    def from_statecut_payload(
        cls,
        payload: Any,
        *,
        pipeline_class: str | None = None,
        expected_family: str | None = None,
        expected_model_identity: Mapping[str, Any] | None = None,
    ) -> DiffusionTrajectoryCheckpoint:
        """Reconstruct a v2 checkpoint from Saturn-compatible state-cut data.

        ``payload`` may be the mapping returned by
        :meth:`to_statecut_payload` or a Saturn ``StateCutCheckpoint`` object.
        The latter is read through its public attributes only, so mrun keeps no
        import-time dependency on Saturn.  A caller may provide an expected
        model identity and family; mismatches fail closed before any replay.
        """

        def read(name: str, default: Any = None) -> Any:
            if isinstance(payload, Mapping):
                return payload.get(name, default)
            return getattr(payload, name, default)

        family = read("family")
        if not isinstance(family, str) or not family:
            _error("state-cut family is missing or invalid")
        state_schema = read("state_schema_id")
        if state_schema != NON_FLUX_STATE_SCHEMA:
            _error(f"unsupported state-cut state_schema_id {state_schema!r}")
        if expected_family is not None and family != expected_family:
            _error(f"state-cut family {family!r} does not match expected {expected_family!r}")
        identity = read("model_identity", {})
        if not isinstance(identity, Mapping):
            _error("state-cut model_identity must be an object")
        if expected_model_identity is not None and _jsonable(dict(identity)) != _jsonable(
            dict(expected_model_identity)
        ):
            _error("state-cut model_identity does not match the requested model")
        boundary = read("boundary", {})
        if not isinstance(boundary, Mapping):
            boundary = {
                "kind": getattr(boundary, "kind", None),
                "index": getattr(boundary, "index", None),
                "total": getattr(boundary, "total", None),
            }
        if boundary.get("kind") != "step_cut":
            _error("non-FLUX trajectory state-cut boundary must be step_cut")
        slots = read("slots", {})
        if not isinstance(slots, Mapping):
            _error("state-cut slots must be a mapping")
        # Saturn stores immutable JSON values in a mapping proxy.  Its public
        # ``tensor_slots``/``json_slots`` accessors are preferred because the
        # top-level ``slots`` mapping may intentionally expose frozen values.
        tensor_slots = getattr(payload, "tensor_slots", None)
        json_slots = getattr(payload, "json_slots", None)
        if isinstance(tensor_slots, Mapping):
            slots = dict(tensor_slots)
        elif callable(tensor_slots):
            slots = dict(tensor_slots())
        if isinstance(json_slots, Mapping):
            slots.update(dict(json_slots))
        elif callable(json_slots):
            slots.update(dict(json_slots()))
        metadata = read("metadata", {})
        if metadata is not None and not isinstance(metadata, Mapping):
            _error("state-cut metadata must be an object")
        metadata = _jsonable(dict(metadata or {}), field_name="state-cut metadata")
        checkpoint_metadata = metadata.get("checkpoint_metadata", {})
        if not isinstance(checkpoint_metadata, Mapping):
            _error("state-cut checkpoint_metadata must be an object")
        resolved_pipeline_class = pipeline_class or metadata.get("pipeline_class")
        if not resolved_pipeline_class:
            _error("state-cut metadata is missing pipeline_class")
        expected = NON_FLUX_PIPELINES.get(str(resolved_pipeline_class))
        if expected is None or expected != family:
            _error(
                "state-cut pipeline class "
                f"{resolved_pipeline_class!r} is not registered for family "
                f"{family!r}"
            )
        height = metadata.get("height")
        width = metadata.get("width")
        if height is None or width is None:
            _error("state-cut metadata is missing height/width")
        checkpoint_id = read("checkpoint_id")
        boundary_index = boundary.get("index")
        boundary_total = boundary.get("total")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            _error("state-cut checkpoint_id is missing or invalid")
        if (
            isinstance(boundary_index, bool)
            or not isinstance(boundary_index, int)
            or isinstance(boundary_total, bool)
            or not isinstance(boundary_total, int)
        ):
            _error("state-cut boundary index and total must be integers")
        try:
            height = int(height)
            width = int(width)
        except (TypeError, ValueError, OverflowError):
            _error("state-cut metadata height/width must be integers")
        nested_state_metadata = metadata.get("state_metadata")
        if nested_state_metadata is not None:
            if not isinstance(nested_state_metadata, Mapping):
                _error("state-cut state_metadata must be an object")
            state_metadata = dict(nested_state_metadata)
        else:
            state_metadata = {
                key: value
                for key, value in metadata.items()
                if key
                not in {
                    "pipeline_class",
                    "height",
                    "width",
                    "guidance_scale",
                    "attention_kwargs",
                    "checkpoint_guidance_scale",
                    "checkpoint_attention_kwargs",
                    "checkpoint_fingerprint",
                    "checkpoint_metadata",
                }
            }
        state = DiffusionTrajectoryState(str(family), slots, state_metadata)
        checkpoint = cls(
            checkpoint_id=checkpoint_id,
            pipeline_class=str(resolved_pipeline_class),
            step_index=boundary_index,
            total_steps=boundary_total,
            height=height,
            width=width,
            state=state,
            guidance_scale=float(
                metadata.get(
                    "guidance_scale",
                    metadata.get("checkpoint_guidance_scale", 1.0),
                )
            ),
            attention_kwargs=metadata.get(
                "attention_kwargs",
                metadata.get("checkpoint_attention_kwargs", {}),
            ),
            metadata=checkpoint_metadata,
        )
        expected_fingerprint = metadata.get("checkpoint_fingerprint")
        if expected_fingerprint is not None and checkpoint.fingerprint != expected_fingerprint:
            _error("state-cut checkpoint fingerprint does not validate after reconstruction")
        return checkpoint


# Names used by callers that want to state explicitly that this is the native
# non-FLUX ABI.  Keep one actual type so isinstance checks remain predictable.
NonFluxTrajectoryCheckpoint = DiffusionTrajectoryCheckpoint
NativeDiffusionTrajectoryCheckpoint = DiffusionTrajectoryCheckpoint


def _filter_kwargs(target: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return dict(kwargs)
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def _invoke(target: Any, kwargs: Mapping[str, Any]) -> Any:
    filtered = _filter_kwargs(target, kwargs)
    try:
        with _torch().inference_mode():
            return target(**filtered)
    except TypeError as exc:
        # A filtered call can hide a required positional-only parameter.  Do
        # not retry arbitrary TypeErrors from the function body; report the
        # native boundary clearly instead.
        try:
            signature = inspect.signature(target)
            required = [
                p.name
                for p in signature.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind is inspect.Parameter.POSITIONAL_ONLY
            ]
        except (TypeError, ValueError):
            required = []
        if required:
            _error(f"cannot call native diffusion method {target!r}: {exc}")
        raise


def _invoke_inference(target: Any, kwargs: Mapping[str, Any]) -> Any:
    """Call a model/VAE under inference mode even when the outer worker does not."""

    with _torch().inference_mode():
        return _invoke(target, kwargs)


def _call_output(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if not value:
            _error("native denoiser returned an empty result")
        return value[0]
    if hasattr(value, "sample"):
        return value.sample
    if hasattr(value, "prev_sample"):
        return value.prev_sample
    return value


def _prepare_extra_step_kwargs(scheduler: Any, *, generator: Any, eta: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(scheduler.step).parameters
    except (TypeError, ValueError):
        parameters = {}
    if eta is not None and (
        "eta" in parameters or any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    ):
        kwargs["eta"] = eta
    if generator is not None and (
        "generator" in parameters or any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    ):
        kwargs["generator"] = generator
    return kwargs


def _scheduler_config_dict(scheduler: Any) -> dict[str, Any]:
    config = getattr(scheduler, "config", None)
    if config is None:
        return {}
    if isinstance(config, Mapping):
        source = config
    elif hasattr(config, "items"):
        source = dict(config.items())
    else:
        source = {
            name: getattr(config, name)
            for name in dir(config)
            if not name.startswith("_") and not callable(getattr(config, name, None))
        }
    result: dict[str, Any] = {}
    for name, value in source.items():
        try:
            result[str(name)] = _jsonable(value, field_name=f"scheduler.config.{name}")
        except Exception as exc:  # noqa: BLE001 - normalize into a PhaseError below
            _error(f"scheduler config is not portable at {name!r}: {exc}")
    return result


_SCHEDULER_ATTRS = (
    "timesteps",
    "sigmas",
    "_step_index",
    "_begin_index",
    "step_index",
    "begin_index",
    "num_inference_steps",
    "_num_inference_steps",
    "order",
    "init_noise_sigma",
    "lower_order_nums",
    "model_outputs",
    "derivatives",
    "ets",
    "counter",
    "_counter",
    "is_scale_input_called",
    "cur_order",
)


def _flatten_state(prefix: str, value: Any, slots: dict[str, Any]) -> Any:
    """Flatten tensor leaves and retain a JSON reconstruction descriptor."""

    if _is_tensor(value):
        slots[prefix] = _clone(value)
        return {"kind": "tensor", "slot": prefix}
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": {
                str(key): _flatten_state(f"{prefix}.{key}", item, slots)
                for key, item in value.items()
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _flatten_state(f"{prefix}.{index}", item, slots) for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _flatten_state(f"{prefix}.{index}", item, slots) for index, item in enumerate(value)
            ],
        }
    return {"kind": "json", "value": _jsonable(value, field_name=prefix)}


def _restore_descriptor(descriptor: Mapping[str, Any], slots: Mapping[str, Any]) -> Any:
    kind = descriptor.get("kind")
    if kind == "tensor":
        slot = str(descriptor.get("slot"))
        if slot not in slots:
            _error(f"checkpoint is missing scheduler tensor slot {slot!r}")
        return _clone(slots[slot])
    if kind == "json":
        return _jsonable(descriptor.get("value"))
    if kind == "mapping":
        return {
            str(key): _restore_descriptor(item, slots)
            for key, item in dict(descriptor.get("items", {})).items()
        }
    if kind == "tuple":
        return tuple(_restore_descriptor(item, slots) for item in descriptor.get("items", ()))
    if kind == "list":
        return [_restore_descriptor(item, slots) for item in descriptor.get("items", ())]
    _error(f"unsupported scheduler state descriptor {kind!r}")


def _snapshot_scheduler(scheduler: Any, slots: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for attr in _SCHEDULER_ATTRS:
        if hasattr(scheduler, attr):
            try:
                paths[attr] = _flatten_state(f"scheduler.{attr}", getattr(scheduler, attr), slots)
            except Exception as exc:  # noqa: BLE001
                # An opaque implementation detail is not replayable.  Do not
                # silently drop known scheduler state; fail closed instead.
                _error(f"cannot snapshot scheduler.{attr}: {exc}")
    return paths


def _schedule_slots(scheduler: Any, slots: dict[str, Any]) -> tuple[Any, Any]:
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if timesteps is None:
        _error("native scheduler did not expose timesteps")
    slots["schedule_timesteps"] = _clone(timesteps)
    if sigmas is not None:
        slots["schedule_sigmas"] = _clone(sigmas)
    return slots["schedule_timesteps"], slots.get("schedule_sigmas")


def _schedule_metadata(
    scheduler: Any,
    *,
    requested_num_inference_steps: int,
    requested_timesteps: Any,
    requested_sigmas: Any,
    mu: Any = None,
    schedule_source: str = "default",
) -> dict[str, Any]:
    return {
        "scheduler_class": type(scheduler).__name__,
        "scheduler_module": type(scheduler).__module__,
        "scheduler_config": _scheduler_config_dict(scheduler),
        "scheduler_begin_index": getattr(
            scheduler, "_begin_index", getattr(scheduler, "begin_index", None)
        ),
        "scheduler_step_index": getattr(
            scheduler, "_step_index", getattr(scheduler, "step_index", None)
        ),
        "schedule_requested_num_inference_steps": int(requested_num_inference_steps),
        "schedule_requested_timesteps": _jsonable(requested_timesteps),
        "schedule_requested_sigmas": _jsonable(requested_sigmas),
        "schedule_source": schedule_source,
        "schedule_mu": None if mu is None else float(mu),
    }


def _validate_scheduler_identity(scheduler: Any, metadata: Mapping[str, Any]) -> None:
    """Validate the scheduler implementation before changing its state.

    A matching timestep tensor is not sufficient to establish replayability:
    schedulers with different transition equations can expose the same grid.
    Require the captured class and normalized configuration before calling
    ``set_timesteps`` or restoring mutable cursor/history fields.
    """

    expected_class = metadata.get("scheduler_class")
    expected_config = metadata.get("scheduler_config")
    if not isinstance(expected_class, str) or not expected_class:
        _error("checkpoint is missing scheduler_class identity")
    if not isinstance(expected_config, Mapping):
        _error("checkpoint is missing scheduler_config identity")
    actual_class = type(scheduler).__name__
    if actual_class != expected_class:
        _error(f"scheduler class {actual_class!r} does not match checkpoint {expected_class!r}")
    expected_module = metadata.get("scheduler_module")
    if expected_module is not None and type(scheduler).__module__ != expected_module:
        _error(
            f"scheduler module {type(scheduler).__module__!r} does not match "
            f"checkpoint {expected_module!r}"
        )
    actual_config = _scheduler_config_dict(scheduler)
    if _jsonable(dict(expected_config), field_name="scheduler_config") != actual_config:
        _error("scheduler configuration does not match the checkpoint")


def _default_sigmas(total_steps: int) -> list[float]:
    if total_steps <= 0:
        _error("num_inference_steps must be positive")
    if total_steps == 1:
        return [1.0]
    # Match np.linspace(1.0, 1 / total_steps, total_steps) without taking a
    # hard dependency on numpy in this lightweight checkpoint module.
    end = 1.0 / total_steps
    return [1.0 + (end - 1.0) * index / (total_steps - 1) for index in range(total_steps)]


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    """The tiny flow-match schedule helper shared by Krea and Chroma."""

    if max_seq_len == base_seq_len:
        return float(base_shift)
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return float(image_seq_len * slope + base_shift - slope * base_seq_len)


def _capture_generator_state(generator: Any, slots: dict[str, Any]) -> None:
    if generator is None:
        return
    if isinstance(generator, (list, tuple)):
        if len(generator) != 1:
            _error("trajectory capture requires one generator, not a generator list")
        generator = generator[0]
    get_state = getattr(generator, "get_state", None)
    if not callable(get_state):
        _error("generator does not expose get_state(); cannot make a replayable checkpoint")
    slots["rng.generator_state"] = _clone(get_state())
    generator_device = getattr(generator, "device", None)
    if generator_device is None:
        _error("generator does not expose its device; cannot make a portable checkpoint")
    slots["rng.generator_device"] = str(generator_device)


def _capture_global_rng_state(slots: dict[str, Any]) -> None:
    """Capture RNG streams used by schedulers or denoisers outside a generator."""

    torch = _torch()
    slots["rng.global_cpu_state"] = _clone(torch.get_rng_state())
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            slots[f"rng.global_cuda_state.{index}"] = _clone(state)


def _restore_generator(phase: Any, checkpoint: DiffusionTrajectoryCheckpoint, supplied: Any) -> Any:
    saved = checkpoint.slot("rng.generator_state")
    if saved is None:
        if supplied is not None:
            _error(
                "checkpoint captured no explicit generator; supplying one would "
                "change the stochastic replay contract"
            )
        return supplied
    torch = _torch()
    if isinstance(supplied, (list, tuple)):
        if len(supplied) != 1:
            _error("resume accepts one generator for a scalar trajectory")
        supplied = supplied[0]
    saved_device = checkpoint.slot("rng.generator_device")
    supplied_device = getattr(supplied, "device", None) if supplied is not None else None
    if saved_device is not None and supplied_device is not None:
        if str(supplied_device) != str(saved_device):
            _error(
                f"supplied generator device {supplied_device!s} does not match "
                f"checkpoint device {saved_device!s}"
            )
    device = saved_device or supplied_device or phase._device
    if supplied is None:
        try:
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError) as exc:
            _error(f"cannot recreate checkpoint generator on device {device!s}: {exc}")
    else:
        generator = supplied
        if not callable(getattr(generator, "set_state", None)):
            _error("supplied generator does not expose set_state()")
    try:
        generator.set_state(saved.detach().to(device="cpu"))
    except (RuntimeError, TypeError) as exc:
        _error(f"cannot restore checkpoint generator state: {exc}")
    restored_state = generator.get_state()
    if not _tensor_equal(restored_state, saved.detach().to(device="cpu")):
        _error("restored generator state does not match the checkpoint digest")
    return generator


def _restore_global_rng_state(checkpoint: DiffusionTrajectoryCheckpoint) -> None:
    torch = _torch()
    cpu_state = checkpoint.slot("rng.global_cpu_state")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state.detach().to(device="cpu"))
    cuda_states = []
    index = 0
    while True:
        state = checkpoint.slot(f"rng.global_cuda_state.{index}")
        if state is None:
            break
        cuda_states.append(state.detach().to(device="cpu"))
        index += 1
    if cuda_states:
        if not torch.cuda.is_available():
            _error("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            _error("checkpoint CUDA RNG device count does not match the live process")
        torch.cuda.set_rng_state_all(cuda_states)


def _set_begin_index(scheduler: Any, index: int) -> None:
    setter = getattr(scheduler, "set_begin_index", None)
    if callable(setter):
        setter(int(index))
    elif hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = int(index)
    elif hasattr(scheduler, "begin_index"):
        scheduler.begin_index = int(index)


def _scheduler_set_timesteps(
    scheduler: Any,
    *,
    total_steps: int,
    device: Any,
    saved_timesteps: Any,
    saved_sigmas: Any,
    metadata: Mapping[str, Any],
) -> None:
    setter = getattr(scheduler, "set_timesteps", None)
    if not callable(setter):
        _error("native scheduler does not expose set_timesteps()")
    try:
        setter_parameters = inspect.signature(setter).parameters
    except (TypeError, ValueError):
        setter_parameters = {}
    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in setter_parameters.values()
    )
    requested_timesteps = metadata.get("schedule_requested_timesteps")
    requested_sigmas = metadata.get("schedule_requested_sigmas")
    schedule_source = metadata.get("schedule_source", "default")
    # Reuse the exact explicit schedule where the pipeline accepted one.  For
    # a default schedule, saved sigmas still provide a stable cross-process
    # schedule for flow schedulers; ordinary DDIM/Euler schedulers can derive
    # the same grid from num_inference_steps.
    requested_count = metadata.get("schedule_requested_num_inference_steps", total_steps)
    try:
        requested_count = int(requested_count)
    except (TypeError, ValueError, OverflowError):
        _error("checkpoint has an invalid requested scheduler step count")
    if requested_count <= 0:
        _error("checkpoint has a non-positive requested scheduler step count")
    kwargs: dict[str, Any] = {"num_inference_steps": requested_count, "device": device}
    if schedule_source == "timesteps" and requested_timesteps is not None:
        if "timesteps" not in setter_parameters and not has_var_kwargs:
            _error("checkpoint requires a scheduler timesteps API")
        kwargs["timesteps"] = requested_timesteps
    elif schedule_source == "sigmas":
        if requested_sigmas is None:
            _error("checkpoint is missing the explicit sigma schedule input")
        if "sigmas" not in setter_parameters and not has_var_kwargs:
            _error("checkpoint requires a scheduler sigmas API")
        kwargs["sigmas"] = requested_sigmas
    if metadata.get("schedule_mu") is not None:
        kwargs["mu"] = metadata["schedule_mu"]
    explicit_schedule = schedule_source in {"timesteps", "sigmas"}
    if explicit_schedule:
        kwargs.pop("num_inference_steps", None)
    filtered = _filter_kwargs(setter, kwargs)
    # Some schedulers use a positional num_inference_steps parameter for a
    # generated schedule, but Diffusers explicitly omits it for timesteps or
    # sigmas supplied by the caller.
    if explicit_schedule:
        try:
            result = setter(**filtered)
        except TypeError as exc:
            _error(f"cannot restore explicit scheduler schedule: {exc}")
    elif "num_inference_steps" in filtered:
        result = setter(**filtered)
    else:
        positional = [requested_count]
        filtered.pop("num_inference_steps", None)
        try:
            result = setter(*positional, **filtered)
        except TypeError as exc:
            _error(f"cannot restore scheduler schedule: {exc}")
    del result
    actual = getattr(scheduler, "timesteps", None)
    if actual is None or not _tensor_equal(actual, saved_timesteps):
        _error("scheduler timesteps do not match the checkpoint schedule")
    if saved_sigmas is not None and hasattr(scheduler, "sigmas"):
        actual_sigmas = scheduler.sigmas
        if actual_sigmas is not None and not _tensor_equal(actual_sigmas, saved_sigmas):
            _error("scheduler sigmas do not match the checkpoint schedule")


def _tensor_equal(left: Any, right: Any) -> bool:
    if _is_tensor(left) and _is_tensor(right):
        torch = _torch()
        return (
            tuple(left.shape) == tuple(right.shape)
            and left.dtype == right.dtype
            and bool(
                torch.equal(
                    left.detach().to("cpu"),
                    right.detach().to("cpu"),
                )
            )
        )
    if _is_tensor(left) or _is_tensor(right):
        tensor = left if _is_tensor(left) else right
        other = right if _is_tensor(left) else left
        try:
            converted = _torch().as_tensor(
                other,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        except (TypeError, ValueError, RuntimeError):
            return False
        if _is_tensor(left):
            return _tensor_equal(left, converted)
        return _tensor_equal(converted, right)
    try:
        return _jsonable(left) == _jsonable(right)
    except Exception:  # noqa: BLE001
        return False


def _restore_scheduler(phase: Any, checkpoint: DiffusionTrajectoryCheckpoint) -> Any:
    pipeline = phase.pipeline
    scheduler = pipeline.scheduler
    saved_timesteps = checkpoint.slot("schedule_timesteps")
    if saved_timesteps is None:
        _error("checkpoint is missing schedule_timesteps")
    saved_sigmas = checkpoint.slot("schedule_sigmas")
    metadata = checkpoint.state.metadata
    # Check identity before ``set_timesteps`` mutates any scheduler state.
    _validate_scheduler_identity(scheduler, metadata)
    _scheduler_set_timesteps(
        scheduler,
        total_steps=checkpoint.total_steps,
        device=phase._device,
        saved_timesteps=saved_timesteps,
        saved_sigmas=saved_sigmas,
        metadata=metadata,
    )
    paths = metadata.get("scheduler_state_paths", {})
    for attr, descriptor in paths.items():
        # Diffusers exposes ``step_index``, ``begin_index``, ``order`` and
        # ``init_noise_sigma`` as read-only properties on several schedulers.
        # Their backing state is captured under the private names below; do
        # not attempt to setattr the public views during restore.
        if attr in {"step_index", "begin_index", "order", "init_noise_sigma"}:
            continue
        if attr in {"timesteps", "sigmas"}:
            # ``set_timesteps`` above reconstructed and byte-validated these
            # values. Keep its native placement: ancestral schedulers can
            # deliberately retain sigmas on CPU while timesteps and model
            # state live on CUDA. Moving both copies to the model device
            # changes arithmetic at the first resumed step.
            continue
        value = _restore_descriptor(descriptor, checkpoint.slots)
        setattr(scheduler, attr, _to_device(value, phase._device))
    # Preserve the scheduler's own begin cursor (normally 0 for text-to-image
    # generation) separately from the denoising cut.  The current step cursor
    # selects the suffix; replacing begin_index with ``checkpoint.step_index``
    # changes Euler-family state even when the next step happens to match.
    saved_begin_index = checkpoint.state.metadata.get("scheduler_begin_index")
    if saved_begin_index is None:
        saved_begin_index = 0
    _set_begin_index(scheduler, int(saved_begin_index))
    # A scheduler that had selected its step index before the cut must retain
    # it.  At step zero, None is meaningful: diffusers resolves it from the
    # begin index on the first step.
    saved_step_index = checkpoint.state.metadata.get("scheduler_step_index")
    if saved_step_index is not None:
        if not hasattr(scheduler, "_step_index"):
            _error("scheduler exposes no writable step cursor for checkpoint restore")
        scheduler._step_index = int(saved_step_index)
    elif hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    observed_begin = getattr(scheduler, "begin_index", None)
    if observed_begin is not None and saved_begin_index is not None:
        if int(observed_begin) != int(saved_begin_index):
            _error("scheduler begin_index does not match the checkpoint cursor")
    observed_step = getattr(scheduler, "step_index", None)
    if saved_step_index is None:
        if observed_step is not None:
            _error("scheduler step_index does not match the checkpoint cursor")
    elif observed_step is not None and int(observed_step) != int(saved_step_index):
        _error("scheduler step_index does not match the checkpoint cursor")
    return scheduler


def _validate_checkpoint(phase: Any, checkpoint: Any) -> DiffusionTrajectoryCheckpoint:
    if not isinstance(checkpoint, DiffusionTrajectoryCheckpoint):
        _error(
            "non-FLUX trajectory resume requires a DiffusionTrajectoryCheckpoint "
            "with the v2 slot-map schema"
        )
    expected = NON_FLUX_PIPELINES.get(phase._class_name)
    if expected is None:
        _error(f"no native non-FLUX trajectory contract for {phase._class_name!r}")
    if checkpoint.pipeline_class != phase._class_name:
        _error(
            f"checkpoint pipeline {checkpoint.pipeline_class!r} does not match "
            f"{phase._class_name!r}"
        )
    if checkpoint.family != expected:
        _error(
            f"checkpoint family {checkpoint.family!r} does not match "
            f"{phase._class_name!r} ({expected})"
        )
    saved_identity = checkpoint.state.metadata.get("model_identity")
    live_identity = str(getattr(phase, "_model_identity", phase._class_name))
    if saved_identity is None:
        _error("checkpoint is missing model_identity binding")
    if str(saved_identity) != live_identity:
        _error(
            f"checkpoint model_identity {saved_identity!r} does not match "
            f"live model {live_identity!r}"
        )
    return checkpoint


def _validate_cut(cut_step: Any, total_steps: int) -> int:
    if isinstance(cut_step, bool):
        _error("cut_step must be an integer")
    try:
        cut = int(cut_step)
    except (TypeError, ValueError):
        _error("cut_step must be an integer")
    if cut != cut_step or not 0 <= cut <= total_steps:
        _error(f"cut_step must be in [0, {total_steps}], got {cut_step}")
    return cut


def _require_embeds(embeds: Any) -> Any:
    from .phase import PromptEmbeds

    if not isinstance(embeds, PromptEmbeds):
        _error("capture_checkpoint requires PromptEmbeds")
    return embeds


def _one_row(value: Any, name: str) -> Any:
    if not _is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != 1:
        _error(f"capture_checkpoint requires exactly one {name} row")
    return value


def _condition_override(
    phase: Any,
    prompt_embeds_override: Any,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    if prompt_embeds_override is None:
        return dict(current)
    from .phase import PromptEmbeds

    if isinstance(prompt_embeds_override, PromptEmbeds):
        values = prompt_embeds_override.tensors
    elif isinstance(prompt_embeds_override, Mapping):
        values = prompt_embeds_override
    else:
        _error("prompt_embeds_override must be PromptEmbeds or a mapping of conditioner slots")
    aliases = {
        "prompt_embeds": (
            "condition_positive_prompt_embeds"
            if "condition_positive_prompt_embeds" in current
            else "condition_prompt_embeds"
        ),
        "negative_prompt_embeds": (
            "condition_negative_prompt_embeds"
            if "condition_negative_prompt_embeds" in current
            else "condition_negative_prompt_embeds"
        ),
        "pooled_prompt_embeds": (
            "condition_positive_pooled_prompt_embeds"
            if "condition_positive_pooled_prompt_embeds" in current
            else "condition_pooled_prompt_embeds"
        ),
        "negative_pooled_prompt_embeds": "condition_negative_pooled_prompt_embeds",
        "prompt_embeds_mask": "condition_prompt_embeds_mask",
        "negative_prompt_embeds_mask": "condition_negative_prompt_embeds_mask",
        "prompt_attention_mask": "condition_prompt_attention_mask",
        "negative_prompt_attention_mask": "condition_negative_prompt_attention_mask",
        "add_time_ids": "condition_positive_add_time_ids",
        "negative_add_time_ids": "condition_negative_add_time_ids",
    }
    # SDXL has semantic positive/negative streams as its override ABI.  The
    # effective CFG concatenation is derived after applying this map; accepting
    # a two-row ``condition_prompt_embeds`` override would make the branch
    # shape-dependent and could silently discard the supplied intervention.
    if "condition_positive_prompt_embeds" in current:
        aliases.update(
            {
                "condition_prompt_embeds": "condition_positive_prompt_embeds",
                "condition_add_text_embeds": "condition_positive_pooled_prompt_embeds",
                "condition_add_time_ids": "condition_positive_add_time_ids",
            }
        )
    result = dict(current)
    for raw_name, value in values.items():
        name = aliases.get(str(raw_name), str(raw_name))
        if name not in result:
            _error(f"conditioner override contains unknown slot {raw_name!r}")
        if not _is_tensor(value) or not _is_tensor(result[name]):
            _error(f"conditioner override {name!r} must be a tensor")
        if tuple(value.shape) != tuple(result[name].shape) or value.dtype != result[name].dtype:
            _error(f"conditioner override {name!r} shape/dtype does not match the checkpoint")
        result[name] = value.to(phase._device).detach().clone()
    return result


def _rebuild_sdxl_conditioners(
    conditioners: dict[str, Any],
    *,
    cfg: bool,
) -> dict[str, Any]:
    """Recompute denoiser-ready SDXL streams from semantic conditioner slots."""

    positive = conditioners.get("condition_positive_prompt_embeds")
    negative = conditioners.get("condition_negative_prompt_embeds")
    positive_pooled = conditioners.get("condition_positive_pooled_prompt_embeds")
    negative_pooled = conditioners.get("condition_negative_pooled_prompt_embeds")
    positive_ids = conditioners.get("condition_positive_add_time_ids")
    negative_ids = conditioners.get("condition_negative_add_time_ids")
    required_positive = {
        "condition_positive_prompt_embeds": positive,
        "condition_positive_pooled_prompt_embeds": positive_pooled,
        "condition_positive_add_time_ids": positive_ids,
    }
    if any(value is None for value in required_positive.values()):
        _error("SDXL checkpoint is missing positive conditioner slots")
    if cfg:
        required_negative = {
            "condition_negative_prompt_embeds": negative,
            "condition_negative_pooled_prompt_embeds": negative_pooled,
            "condition_negative_add_time_ids": negative_ids,
        }
        if any(value is None for value in required_negative.values()):
            _error("SDXL checkpoint is missing negative conditioner slots")
        for name, left, right in (
            ("prompt_embeds", positive, negative),
            ("pooled_prompt_embeds", positive_pooled, negative_pooled),
            ("add_time_ids", positive_ids, negative_ids),
        ):
            if not _is_tensor(left) or not _is_tensor(right):
                _error(f"SDXL {name} conditioner must be a tensor")
            if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
                _error(f"SDXL positive/negative {name} shapes or dtypes do not match")
        conditioners["condition_prompt_embeds"] = _torch().cat([negative, positive])
        conditioners["condition_add_text_embeds"] = _torch().cat([negative_pooled, positive_pooled])
        conditioners["condition_add_time_ids"] = _torch().cat([negative_ids, positive_ids])
    else:
        conditioners["condition_prompt_embeds"] = positive
        conditioners["condition_add_text_embeds"] = positive_pooled
        conditioners["condition_add_time_ids"] = positive_ids
    return conditioners


def _resolve_initial_latents(params: dict[str, Any], initial_latents: Any | None) -> Any | None:
    if initial_latents is not None and "latents" in params:
        _error("provide initial_latents or latents, not both")
    if initial_latents is None:
        initial_latents = params.pop("latents", None)
    return initial_latents


def _common_params(
    phase: Any,
    kwargs: Mapping[str, Any],
    *,
    initial_latents: Any | None,
    default_steps: int,
    default_guidance: float,
) -> dict[str, Any]:
    params = dict(kwargs)
    params.pop("prompt", None)
    output_type = params.pop("output_type", "latent")
    if output_type not in {"latent", "pil", "np"}:
        _error(f"unsupported trajectory output_type {output_type!r}")
    if params.pop("return_dict", True) is False:
        _error("trajectory checkpoints always use the typed result path")
    for name in phase._contract.feed_keys:
        if name in params:
            _error(f"capture_checkpoint manages {name!r}")
    raw_steps = params.pop("num_inference_steps", None)
    if isinstance(raw_steps, bool):
        _error("num_inference_steps must be a positive integer")
    sigmas = params.pop("sigmas", None)
    timesteps = params.pop("timesteps", None)
    # Diffusers' retrieve_timesteps() makes an explicit schedule authoritative
    # and returns its length as the effective number of denoising steps.  Keep
    # the checkpoint boundary aligned with that native contract even when the
    # caller omitted (or repeated a stale) num_inference_steps value.
    if sigmas is not None:
        try:
            schedule_steps = len(sigmas)
        except TypeError:
            _error("sigmas must be a finite sequence")
    elif timesteps is not None:
        try:
            schedule_steps = len(timesteps)
        except TypeError:
            _error("timesteps must be a finite sequence")
    else:
        schedule_steps = None
    try:
        total_steps = int(
            raw_steps
            if raw_steps is not None
            else (schedule_steps if schedule_steps is not None else default_steps)
        )
    except (TypeError, ValueError, OverflowError):
        _error("num_inference_steps must be a positive integer")
    if schedule_steps is not None:
        total_steps = int(schedule_steps)
    raw_attention_kwargs = params.pop("cross_attention_kwargs", _MISSING)
    if raw_attention_kwargs is _MISSING:
        raw_attention_kwargs = params.pop("attention_kwargs", _MISSING)
    attention_kwargs = None if raw_attention_kwargs is _MISSING else raw_attention_kwargs
    raw_joint_attention_kwargs = params.pop("joint_attention_kwargs", _MISSING)
    joint_attention_kwargs = (
        None if raw_joint_attention_kwargs is _MISSING else raw_joint_attention_kwargs
    )
    resolved = {
        "output_type": output_type,
        "height": params.pop("height", None),
        "width": params.pop("width", None),
        "total_steps": total_steps,
        "guidance_scale": float(params.pop("guidance_scale", default_guidance)),
        "generator": params.pop("generator", None),
        "initial_latents": _resolve_initial_latents(params, initial_latents),
        "sigmas": sigmas,
        "timesteps": timesteps,
        "eta": params.pop("eta", None),
        # These options belong to the encode phase.  The caller may repeat
        # them on the denoise call (as Diffusers' public API permits), but
        # they must not leak into the native denoiser kwargs or be mistaken
        # for unsupported conditioning.  Krea uses max_sequence_length
        # below when it has to encode a missing negative branch.
        "max_sequence_length": params.pop("max_sequence_length", None),
        "lora_scale": params.pop("lora_scale", None),
        "attention_kwargs": attention_kwargs,
        "joint_attention_kwargs": joint_attention_kwargs,
    }
    if resolved["total_steps"] <= 0:
        _error("num_inference_steps must be positive")
    resolved["params"] = params
    return resolved


def _check_unknown(params: Mapping[str, Any], allowed: Sequence[str] = ()) -> None:
    if params:
        _error("unsupported trajectory options: " + ", ".join(sorted(map(str, params))))


def _default_resolution(pipeline: Any, *, family: str) -> tuple[int, int]:
    if family == "sdxl":
        sample = getattr(getattr(pipeline, "unet", None), "config", SimpleNamespace(sample_size=64))
        sample_size = int(getattr(sample, "sample_size", 64))
    else:
        sample_size = int(getattr(pipeline, "default_sample_size", 64))
    scale = int(getattr(pipeline, "vae_scale_factor", 8))
    return sample_size * scale, sample_size * scale


def _round_krea_resolution(pipeline: Any, height: int, width: int) -> tuple[int, int]:
    """Apply Krea2's native patch-grid rounding before latent preparation."""

    multiple = int(getattr(pipeline, "vae_scale_factor", 8)) * int(
        getattr(pipeline, "patch_size", 2)
    )
    if multiple <= 0:
        _error("Krea2 pipeline exposes an invalid latent-grid scale")
    return (
        ((height + multiple - 1) // multiple) * multiple,
        ((width + multiple - 1) // multiple) * multiple,
    )


def _prepare_latents(
    pipeline: Any,
    *,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    dtype: Any,
    device: Any,
    generator: Any,
    latents: Any | None,
) -> Any:
    method = getattr(pipeline, "prepare_latents", None)
    if not callable(method):
        _error(f"{type(pipeline).__name__} has no native prepare_latents()")
    result = _invoke(
        method,
        {
            "batch_size": batch_size,
            "num_channels_latents": channels,
            "height": height,
            "width": width,
            "dtype": dtype,
            "device": device,
            "generator": generator,
            "latents": latents,
        },
    )
    if isinstance(result, (tuple, list)):
        if not result:
            _error("native prepare_latents() returned no latent state")
        return result
    return result


def _prepare_scheduler(
    pipeline: Any,
    *,
    total_steps: int,
    device: Any,
    timesteps: Any,
    sigmas: Any,
    mu: float | None,
) -> tuple[Any, int]:
    if timesteps is not None and sigmas is not None:
        _error("trajectory schedule accepts timesteps or sigmas, not both")
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is None:
        _error("native trajectory pipeline has no scheduler")
    setter = getattr(scheduler, "set_timesteps", None)
    if not callable(setter):
        _error("native trajectory scheduler has no set_timesteps()")
    try:
        setter_parameters = inspect.signature(setter).parameters
    except (TypeError, ValueError):
        setter_parameters = {}
    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in setter_parameters.values()
    )
    if timesteps is not None and "timesteps" not in setter_parameters and not has_var_kwargs:
        _error("native scheduler does not support an explicit timesteps schedule")
    if sigmas is not None and "sigmas" not in setter_parameters and not has_var_kwargs:
        _error("native scheduler does not support an explicit sigma schedule")
    kwargs: dict[str, Any] = {
        "num_inference_steps": total_steps,
        "device": device,
        "timesteps": timesteps,
        "sigmas": sigmas,
        "mu": mu,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    explicit_schedule = timesteps is not None or sigmas is not None
    if explicit_schedule:
        kwargs.pop("num_inference_steps", None)
    filtered = _filter_kwargs(setter, kwargs)
    if explicit_schedule:
        try:
            setter(**filtered)
        except TypeError as exc:
            # An explicit schedule is a distinct native API.  Retrying with a
            # positional count could silently select the scheduler's default
            # schedule while the checkpoint still claims the explicit one.
            _error(f"cannot configure native explicit scheduler schedule: {exc}")
    elif "num_inference_steps" in filtered:
        setter(**filtered)
    else:
        count = int(total_steps)
        filtered.pop("num_inference_steps", None)
        try:
            setter(count, **filtered)
        except TypeError as exc:
            _error(f"cannot configure native trajectory scheduler: {exc}")
    resolved = getattr(scheduler, "timesteps", None)
    if resolved is None:
        _error("native scheduler did not produce timesteps")
    if timesteps is not None and not _tensor_equal(resolved, timesteps):
        _error("native scheduler changed the requested timesteps schedule")
    if sigmas is not None:
        actual_sigmas = getattr(scheduler, "sigmas", None)
        if actual_sigmas is None:
            _error("native scheduler did not expose the requested sigma schedule")
        # FlowMatch schedulers treat caller-provided sigmas as input to a
        # family-native shift (including Krea2's dynamic ``mu`` shift), not as
        # the resolved schedule itself. Validate the resolved geometry here
        # and persist the scheduler's exact transformed tensors below. Resume
        # replays the raw request plus ``mu`` and byte-validates that result.
        try:
            actual_sigma_count = len(actual_sigmas)
        except TypeError:
            _error("native scheduler exposed a non-sequence sigma schedule")
        resolved_count = int(getattr(resolved, "shape", [len(resolved)])[0])
        if actual_sigma_count < resolved_count:
            _error("native scheduler produced an undersized sigma schedule")
        if _is_tensor(actual_sigmas) and not bool(_torch().isfinite(actual_sigmas).all()):
            _error("native scheduler produced a non-finite sigma schedule")
    return resolved, int(getattr(resolved, "shape", [len(resolved)])[0])


def _step_scheduler(
    scheduler: Any,
    *,
    noise: Any,
    timestep: Any,
    latents: Any,
    generator: Any,
    eta: Any,
    native_extra_step_kwargs: bool = True,
) -> Any:
    extra_kwargs = (
        _prepare_extra_step_kwargs(scheduler, generator=generator, eta=eta)
        if native_extra_step_kwargs
        else {}
    )
    with _torch().inference_mode():
        result = scheduler.step(
            noise,
            timestep,
            latents,
            return_dict=False,
            **extra_kwargs,
        )
    return _call_output(result)


def _observe(
    observer: Callable[[Mapping[str, Any]], None] | None,
    *,
    phase: str,
    index: int,
    timestep: Any,
    latents_before: Any,
    noise_pred: Any,
    latents_after: Any,
    **streams: Any,
) -> None:
    if observer is None:
        return
    payload = {
        "family": phase,
        "step_index": index,
        "timestep": _clone(timestep),
        "latents_before": _clone(latents_before),
        "noise_pred": _clone(noise_pred),
        "latents_after": _clone(latents_after),
    }
    payload.update({key: _clone(value) for key, value in streams.items()})
    observer(payload)


def _rescale_noise_cfg(noise_cfg: Any, noise_text: Any, guidance_rescale: float) -> Any:
    if not guidance_rescale:
        return noise_cfg
    dims = tuple(range(1, noise_cfg.ndim))
    std_text = noise_text.float().std(dim=dims, keepdim=True)
    std_cfg = noise_cfg.float().std(dim=dims, keepdim=True)
    noise_rescaled = noise_cfg * (std_text / (std_cfg + 1e-5))
    return guidance_rescale * noise_rescaled + (1.0 - guidance_rescale) * noise_cfg


def _restore_mps_latent_dtype(latents: Any, dtype: Any) -> Any:
    """Mirror Diffusers' MPS-only scheduler dtype workaround."""

    torch = _torch()
    if latents.dtype == dtype:
        return latents
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        return latents.to(dtype)
    return latents


def _chroma_timestep(timestep: Any, *, batch_size: int, dtype: Any) -> Any:
    """Match Chroma's native cast-before-normalize timestep boundary."""

    return timestep.expand(int(batch_size)).to(dtype) / 1000.0


def _sdxl_conditioning(
    phase: Any,
    embeds: Any,
    *,
    height: int,
    width: int,
    guidance_scale: float,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    feeds = phase._device_feed_tensors(embeds)
    required = (
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    )
    if any(name not in feeds for name in required):
        _error("SDXL capture requires all four prompt/pooled conditioner streams")
    for name in required:
        _one_row(feeds[name], name)
    # ``PhasePipeline`` may detach the text encoders before the denoise phase.
    # Diffusers' SDXL ``encode_prompt`` still recasts the sequence streams in
    # that state: with ``text_encoder_2`` absent it uses ``unet.dtype`` (and
    # with the encoder present it uses ``text_encoder_2.dtype``).  A native
    # trajectory must feed the denoiser the same post-encode tensors as the
    # high-level pipeline; retaining the cached encoder dtype here otherwise
    # makes direct and capture/resume numerically different.
    prompt_embeds = feeds["prompt_embeds"]
    negative_prompt_embeds = feeds["negative_prompt_embeds"]
    text_encoder_2 = getattr(phase.pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        prompt_dtype = getattr(text_encoder_2, "dtype", None)
    else:
        prompt_dtype = getattr(getattr(phase.pipeline, "unet", None), "dtype", None)
    if prompt_dtype is not None:
        prompt_embeds = prompt_embeds.to(dtype=prompt_dtype, device=phase._device)
        negative_prompt_embeds = negative_prompt_embeds.to(
            dtype=prompt_dtype, device=phase._device
        )
    original_size = params.pop("original_size", None) or (height, width)
    target_size = params.pop("target_size", None) or (height, width)
    crops = params.pop("crops_coords_top_left", (0, 0)) or (0, 0)
    negative_original_size = params.pop("negative_original_size", None)
    negative_target_size = params.pop("negative_target_size", None)
    negative_crops = params.pop("negative_crops_coords_top_left", (0, 0)) or (0, 0)
    original_size = tuple(original_size)
    target_size = tuple(target_size)
    crops = tuple(crops)
    if negative_original_size is not None:
        negative_original_size = tuple(negative_original_size)
    if negative_target_size is not None:
        negative_target_size = tuple(negative_target_size)
    negative_crops = tuple(negative_crops)
    projection_dim = getattr(
        getattr(getattr(phase.pipeline, "text_encoder_2", None), "config", None),
        "projection_dim",
        None,
    )
    if projection_dim is None:
        # PhasePipeline detaches text_encoder_2 during denoise.  Diffusers'
        # SDXL implementation uses the pooled stream's trailing dimension in
        # exactly this case.
        projection_dim = int(feeds["pooled_prompt_embeds"].shape[-1])
    get_ids = getattr(phase.pipeline, "_get_add_time_ids", None)
    if not callable(get_ids):
        _error("SDXL pipeline has no native _get_add_time_ids()")
    add_ids = _invoke(
        get_ids,
        {
            "original_size": original_size,
            "crops_coords_top_left": crops,
            "target_size": target_size,
            "dtype": prompt_embeds.dtype,
            "text_encoder_projection_dim": projection_dim,
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.5,
        },
    ).to(phase._device)
    if negative_original_size is not None and negative_target_size is not None:
        negative_ids = _invoke(
            get_ids,
            {
                "original_size": negative_original_size,
                "crops_coords_top_left": negative_crops,
                "target_size": negative_target_size,
                "dtype": prompt_embeds.dtype,
                "text_encoder_projection_dim": projection_dim,
                "aesthetic_score": 6.0,
                "negative_aesthetic_score": 2.5,
            },
        ).to(phase._device)
    else:
        # Native SDXL uses the positive time IDs whenever either negative
        # geometry argument is omitted.
        negative_ids = add_ids.clone()
    unet_config = getattr(getattr(phase.pipeline, "unet", None), "config", None)
    time_dim = getattr(unet_config, "time_cond_proj_dim", None)
    # SDXL's native property disables classifier-free doubling when the UNet
    # consumes the learned guidance-scale embedding.
    cfg = guidance_scale > 1.0 and time_dim is None
    if cfg:
        prompt = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled = torch.cat([feeds["negative_pooled_prompt_embeds"], feeds["pooled_prompt_embeds"]])
        time_ids = torch.cat([negative_ids, add_ids])
    else:
        prompt = feeds["prompt_embeds"]
        pooled = feeds["pooled_prompt_embeds"]
        time_ids = add_ids
    timestep_cond = None
    if time_dim is not None:
        helper = getattr(phase.pipeline, "get_guidance_scale_embedding", None)
        if not callable(helper):
            _error(
                "SDXL UNet requests timestep conditioning but pipeline lacks its embedding helper"
            )
        initial_latents = params.get("initial_latents")
        latent_dtype = (
            initial_latents.dtype if _is_tensor(initial_latents) else prompt_embeds.dtype
        )
        timestep_cond = _invoke(
            helper,
            {
                "w": torch.tensor([guidance_scale - 1.0], device=phase._device),
                "embedding_dim": int(time_dim),
                "dtype": latent_dtype,
            },
        ).to(device=phase._device, dtype=latent_dtype)
    # Retain the semantic positive/negative streams in addition to the
    # denoiser-ready concatenated streams.  The latter make replay cheap; the
    # former make a StateCut branch auditable and allow typed conditioner
    # overrides without pretending SDXL has FLUX text_ids.
    state = {
        "condition_positive_prompt_embeds": prompt_embeds,
        "condition_negative_prompt_embeds": negative_prompt_embeds,
        "condition_positive_pooled_prompt_embeds": feeds["pooled_prompt_embeds"],
        "condition_negative_pooled_prompt_embeds": feeds["negative_pooled_prompt_embeds"],
        "condition_positive_add_time_ids": add_ids,
        "condition_negative_add_time_ids": negative_ids,
        "condition_prompt_embeds": prompt,
        "condition_add_text_embeds": pooled,
        "condition_add_time_ids": time_ids,
    }
    if timestep_cond is not None:
        state["condition_timestep_cond"] = timestep_cond
    metadata = {
        "conditioner_mapping": {
            "prompt_embeds": "condition_positive_prompt_embeds",
            "negative_prompt_embeds": "condition_negative_prompt_embeds",
            "pooled_prompt_embeds": "condition_positive_pooled_prompt_embeds",
            "negative_pooled_prompt_embeds": "condition_negative_pooled_prompt_embeds",
            "time_ids": "condition_positive_add_time_ids",
            "negative_time_ids": "condition_negative_add_time_ids",
            "effective_prompt_embeds": "condition_prompt_embeds",
            "effective_pooled_prompt_embeds": "condition_add_text_embeds",
            "effective_time_ids": "condition_add_time_ids",
            "timestep_cond": "condition_timestep_cond",
        },
        "cfg_enabled": cfg,
        "guidance_rescale": float(params.pop("guidance_rescale", 0.0)),
        "original_size": list(original_size),
        "target_size": list(target_size),
        "crops_coords_top_left": list(crops),
        "negative_original_size": (
            None if negative_original_size is None else list(negative_original_size)
        ),
        "negative_target_size": (
            None if negative_target_size is None else list(negative_target_size)
        ),
        "negative_crops_coords_top_left": list(negative_crops),
    }
    return state, metadata


@_inference_function
def _capture_sdxl(
    phase: Any,
    embeds: Any,
    *,
    cut_step: int,
    initial_latents: Any | None,
    kwargs: Mapping[str, Any],
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> DiffusionTrajectoryCheckpoint:
    pipeline = phase.pipeline
    params = _common_params(
        phase, kwargs, initial_latents=initial_latents, default_steps=50, default_guidance=7.5
    )
    height, width = _default_resolution(pipeline, family="sdxl")
    height = int(params["height"] if params["height"] is not None else height)
    width = int(params["width"] if params["width"] is not None else width)
    requested_steps = params["total_steps"]
    if requested_steps <= 0:
        _error("num_inference_steps must be positive")
    conditioning, condition_metadata = _sdxl_conditioning(
        phase,
        _require_embeds(embeds),
        height=height,
        width=width,
        guidance_scale=params["guidance_scale"],
        params=params["params"],
    )
    _check_unknown(params["params"])
    prompt = conditioning["condition_prompt_embeds"]
    unet_config = getattr(getattr(pipeline, "unet", None), "config", None)
    channels = int(getattr(unet_config, "in_channels", 4))
    timesteps, resolved_steps = _prepare_scheduler(
        pipeline,
        total_steps=requested_steps,
        device=phase._device,
        timesteps=params["timesteps"],
        sigmas=params["sigmas"],
        mu=None,
    )
    total_steps = int(resolved_steps)
    cut = _validate_cut(cut_step, total_steps)
    scheduler = pipeline.scheduler
    if callable(getattr(scheduler, "set_begin_index", None)):
        scheduler.set_begin_index(0)
    latents = _prepare_latents(
        pipeline,
        batch_size=1,
        channels=channels,
        height=height,
        width=width,
        dtype=prompt.dtype,
        device=phase._device,
        generator=params["generator"],
        latents=params["initial_latents"],
    )
    if isinstance(latents, (tuple, list)):
        latents = latents[0]
    latents = latents.to(phase._device)
    pipeline._guidance_scale = params["guidance_scale"]
    pipeline._guidance_rescale = condition_metadata["guidance_rescale"]
    pipeline._interrupt = False
    for index, timestep in enumerate(timesteps):
        if index >= cut:
            break
        latent_model_input = torch_cat(latents, cfg=condition_metadata["cfg_enabled"])
        scale = getattr(scheduler, "scale_model_input", None)
        if callable(scale):
            latent_model_input = scale(latent_model_input, timestep)
        unet = getattr(pipeline, "unet", None)
        if not callable(unet):
            _error("SDXL pipeline has no callable unet")
        unet_out = _invoke(
            unet,
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": conditioning["condition_prompt_embeds"],
                "timestep_cond": conditioning.get("condition_timestep_cond"),
                "cross_attention_kwargs": params["attention_kwargs"],
                "added_cond_kwargs": {
                    "text_embeds": conditioning["condition_add_text_embeds"],
                    "time_ids": conditioning["condition_add_time_ids"],
                },
                "return_dict": False,
            },
        )
        noise = _call_output(unet_out)
        if condition_metadata["cfg_enabled"]:
            noise_uncond, noise_text = noise.chunk(2)
            noise = noise_uncond + params["guidance_scale"] * (noise_text - noise_uncond)
            noise = _rescale_noise_cfg(noise, noise_text, condition_metadata["guidance_rescale"])
        before = latents
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=params["generator"],
            eta=params["eta"],
        )
        _observe(
            step_observer,
            phase="sdxl",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
        )
    slots: dict[str, Any] = {"latents": latents, **conditioning}
    _schedule_slots(scheduler, slots)
    _capture_generator_state(params["generator"], slots)
    _capture_global_rng_state(slots)
    scheduler_paths = _snapshot_scheduler(scheduler, slots)
    state_metadata = {
        "family": "sdxl",
        "model_identity": str(phase._model_identity),
        "latent_layout": "nchw",
        "prefix_steps_executed": cut,
        "guidance_scale": params["guidance_scale"],
        "eta": params["eta"],
        "scheduler_state_paths": scheduler_paths,
        **condition_metadata,
        **_schedule_metadata(
            scheduler,
            requested_num_inference_steps=requested_steps,
            requested_timesteps=params["timesteps"],
            requested_sigmas=params["sigmas"],
            schedule_source=(
                "timesteps"
                if params["timesteps"] is not None
                else "sigmas"
                if params["sigmas"] is not None
                else "default"
            ),
        ),
    }
    state = DiffusionTrajectoryState("sdxl", slots, state_metadata)
    checkpoint_id = _checkpoint_id(pipeline, state, cut, total_steps, height, width)
    return DiffusionTrajectoryCheckpoint(
        checkpoint_id=checkpoint_id,
        pipeline_class=type(pipeline).__name__,
        step_index=cut,
        total_steps=total_steps,
        height=height,
        width=width,
        state=state,
        guidance_scale=params["guidance_scale"],
        attention_kwargs=params["attention_kwargs"],
        metadata={"output_type_requested_at_capture": params["output_type"]},
    )


def torch_cat(latents: Any, *, cfg: bool) -> Any:
    if not cfg:
        return latents
    return _torch().cat([latents, latents])


def _checkpoint_id(
    pipeline: Any, state: DiffusionTrajectoryState, cut: int, total: int, height: int, width: int
) -> str:
    return (
        "traj-"
        + _digest(
            {
                "pipeline_class": type(pipeline).__name__,
                "state": state.fingerprint,
                "step_index": cut,
                "total_steps": total,
                "height": height,
                "width": width,
            }
        )[:32]
    )


def _krea_negative_conditioning(
    phase: Any,
    params: dict[str, Any],
    *,
    positive_shape: Any,
    positive_mask_shape: Any,
    guidance_scale: float,
    max_sequence_length: int | None,
) -> tuple[Any | None, Any | None]:
    if guidance_scale <= 0:
        return None, None
    negative = params.pop("negative_prompt_embeds", None)
    negative_mask = params.pop("negative_prompt_embeds_mask", None)
    negative_prompt = params.pop("negative_prompt", None)
    if negative is not None:
        if negative_mask is None:
            _error("Krea2 negative_prompt_embeds requires negative_prompt_embeds_mask")
        negative = negative.to(phase._device)
        negative_mask = negative_mask.to(phase._device)
        _one_row(negative, "negative_prompt_embeds")
        _one_row(negative_mask, "negative_prompt_embeds_mask")
        if tuple(negative.shape) != tuple(positive_shape):
            _error("Krea2 negative conditioner shape does not match the positive conditioner")
        if tuple(negative_mask.shape) != tuple(positive_mask_shape):
            _error("Krea2 negative conditioner mask shape does not match the positive mask")
        return negative, negative_mask
    if negative_prompt is None:
        negative_prompt = ""
    encode = getattr(phase.pipeline, "encode_prompt", None)
    if not callable(encode):
        _error("Krea2 pipeline has no encode_prompt() for negative conditioning")
    # The positive embed was produced in encode phase.  Re-enter briefly only
    # when Krea guidance actually needs the negative branch; Turbo guidance=0
    # never pays this transfer.
    phase._ensure_phase("encode")
    encoded = _invoke(
        encode,
        {
            "prompt": negative_prompt,
            "device": _torch().device(phase._device),
            "num_images_per_prompt": 1,
            "max_sequence_length": (
                512 if max_sequence_length is None else int(max_sequence_length)
            ),
        },
    )
    phase._ensure_phase("denoise")
    if not isinstance(encoded, (tuple, list)) or len(encoded) < 2:
        _error("Krea2 negative encode_prompt() did not return embeds and mask")
    negative, negative_mask = encoded[:2]
    if tuple(negative.shape) != tuple(positive_shape):
        _error("Krea2 negative conditioner shape does not match the positive conditioner")
    negative = negative.to(phase._device)
    negative_mask = negative_mask.to(phase._device)
    _one_row(negative, "negative_prompt_embeds")
    _one_row(negative_mask, "negative_prompt_embeds_mask")
    if tuple(negative_mask.shape) != tuple(positive_mask_shape):
        _error("Krea2 negative conditioner mask shape does not match the positive mask")
    return negative, negative_mask


@_inference_function
def _capture_krea(
    phase: Any,
    embeds: Any,
    *,
    cut_step: int,
    initial_latents: Any | None,
    kwargs: Mapping[str, Any],
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> DiffusionTrajectoryCheckpoint:
    pipeline = phase.pipeline
    params = _common_params(
        phase, kwargs, initial_latents=initial_latents, default_steps=28, default_guidance=4.5
    )
    height, width = _default_resolution(pipeline, family="krea2")
    height = int(params["height"] if params["height"] is not None else height)
    width = int(params["width"] if params["width"] is not None else width)
    if height <= 0 or width <= 0:
        _error("Krea2 trajectory resolution must be positive")
    height, width = _round_krea_resolution(pipeline, height, width)
    requested_steps = params["total_steps"]
    embed_values = phase._device_feed_tensors(_require_embeds(embeds))
    positive = embed_values.get("prompt_embeds")
    positive_mask = embed_values.get("prompt_embeds_mask")
    if positive is None or positive_mask is None:
        _error("Krea2 capture requires prompt_embeds and prompt_embeds_mask")
    _one_row(positive, "prompt_embeds")
    _one_row(positive_mask, "prompt_embeds_mask")
    params_map = params["params"]
    negative, negative_mask = _krea_negative_conditioning(
        phase,
        params_map,
        positive_shape=positive.shape,
        positive_mask_shape=positive_mask.shape,
        guidance_scale=params["guidance_scale"],
        max_sequence_length=params["max_sequence_length"],
    )
    _check_unknown(params_map)
    if params["eta"] is not None:
        _error("Krea2 native scheduler does not accept eta")
    channels = int(
        getattr(getattr(pipeline.transformer, "config", None), "in_channels", 16)
    ) // int(getattr(pipeline, "patch_size", 1) ** 2)
    latents_result = _prepare_latents(
        pipeline,
        batch_size=1,
        channels=channels,
        height=height,
        width=width,
        dtype=positive.dtype,
        device=phase._device,
        generator=params["generator"],
        latents=params["initial_latents"],
    )
    latents = latents_result[0] if isinstance(latents_result, (tuple, list)) else latents_result
    latents = latents.to(phase._device)
    grid_height = height // (
        int(getattr(pipeline, "vae_scale_factor", 8)) * int(getattr(pipeline, "patch_size", 1))
    )
    grid_width = width // (
        int(getattr(pipeline, "vae_scale_factor", 8)) * int(getattr(pipeline, "patch_size", 1))
    )
    position = getattr(pipeline, "prepare_position_ids", None)
    if not callable(position):
        _error("Krea2 pipeline has no prepare_position_ids()")
    position_ids = position(int(positive.shape[1]), grid_height, grid_width, phase._device).to(
        phase._device
    )
    sigmas = params["sigmas"]
    if sigmas is None and params["timesteps"] is None:
        sigmas = _default_sigmas(requested_steps)
    image_seq_len = int(latents.shape[1])
    config = getattr(pipeline.scheduler, "config", {})
    get = (
        config.get
        if hasattr(config, "get")
        else lambda name, default=None: getattr(config, name, default)
    )
    if bool(getattr(getattr(pipeline, "config", None), "is_distilled", False)) or bool(
        get("is_distilled", False)
    ):
        mu = 1.15
    else:
        mu = _calculate_shift(
            image_seq_len,
            get("base_image_seq_len", 256),
            get("max_image_seq_len", 6400),
            get("base_shift", 0.5),
            get("max_shift", 1.15),
        )
    timesteps, resolved_steps = _prepare_scheduler(
        pipeline,
        total_steps=requested_steps,
        device=phase._device,
        timesteps=params["timesteps"],
        sigmas=sigmas,
        mu=mu,
    )
    total_steps = int(resolved_steps)
    cut = _validate_cut(cut_step, total_steps)
    scheduler = pipeline.scheduler
    _set_begin_index(scheduler, 0)
    pipeline._guidance_scale = params["guidance_scale"]
    pipeline._attention_kwargs = params["attention_kwargs"]
    slots: dict[str, Any] = {
        "latents": latents,
        "condition_prompt_embeds": positive,
        "condition_prompt_embeds_mask": positive_mask,
        "layout_position_ids": position_ids,
    }
    if negative is not None:
        slots["condition_negative_prompt_embeds"] = negative
        slots["condition_negative_prompt_embeds_mask"] = negative_mask
    cfg = params["guidance_scale"] > 0
    transformer = getattr(pipeline, "transformer", None)
    if not callable(transformer):
        _error("Krea2 pipeline has no callable transformer")
    for index, timestep in enumerate(timesteps):
        if index >= cut:
            break
        t = (
            (timestep / float(get("num_train_timesteps", 1000)))
            .expand(latents.shape[0])
            .to(latents.dtype)
        )
        before = latents
        cond = _call_output(
            _invoke(
                transformer,
                {
                    "hidden_states": latents,
                    "encoder_hidden_states": positive,
                    "timestep": t,
                    "position_ids": position_ids,
                    "encoder_attention_mask": positive_mask,
                    "attention_kwargs": params["attention_kwargs"],
                    "return_dict": False,
                },
            )
        )
        neg = None
        noise = cond
        if cfg:
            if negative is None or negative_mask is None:
                _error("Krea2 guidance requires negative conditioner slots")
            neg = _call_output(
                _invoke(
                    transformer,
                    {
                        "hidden_states": latents,
                        "encoder_hidden_states": negative,
                        "timestep": t,
                        "position_ids": position_ids,
                        "encoder_attention_mask": negative_mask,
                        "attention_kwargs": params["attention_kwargs"],
                        "return_dict": False,
                    },
                )
            )
            noise = cond + params["guidance_scale"] * (cond - neg)
        latents_dtype = latents.dtype
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=params["generator"],
            eta=params["eta"],
            native_extra_step_kwargs=False,
        )
        latents = _restore_mps_latent_dtype(latents, latents_dtype)
        _observe(
            step_observer,
            phase="krea2",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
            positive_noise=cond,
            negative_noise=neg,
        )
    # The checkpoint boundary is the post-cut latent, not the initial noise
    # tensor placed in the slot map before the prefix loop.
    slots["latents"] = latents
    _schedule_slots(scheduler, slots)
    _capture_generator_state(params["generator"], slots)
    _capture_global_rng_state(slots)
    scheduler_paths = _snapshot_scheduler(scheduler, slots)
    state_metadata = {
        "family": "krea2",
        "model_identity": str(phase._model_identity),
        "latent_layout": "packed_btd",
        "patch_size": int(getattr(pipeline, "patch_size", 1)),
        "latent_grid": [grid_height, grid_width],
        "conditioner_mapping": {
            "prompt_embeds": "condition_prompt_embeds",
            "prompt_embeds_mask": "condition_prompt_embeds_mask",
            "negative_prompt_embeds": "condition_negative_prompt_embeds",
            "negative_prompt_embeds_mask": "condition_negative_prompt_embeds_mask",
        },
        "cfg_enabled": cfg,
        "guidance_convention": "cond_plus_scale_cond_minus_negative",
        "guidance_scale": params["guidance_scale"],
        "schedule_sigmas_explicit": _jsonable(sigmas),
        "scheduler_state_paths": scheduler_paths,
        **_schedule_metadata(
            scheduler,
            requested_num_inference_steps=requested_steps,
            requested_timesteps=params["timesteps"],
            requested_sigmas=sigmas,
            mu=mu,
            schedule_source="timesteps" if params["timesteps"] is not None else "sigmas",
        ),
    }
    state = DiffusionTrajectoryState("krea2", slots, state_metadata)
    return DiffusionTrajectoryCheckpoint(
        checkpoint_id=_checkpoint_id(pipeline, state, cut, total_steps, height, width),
        pipeline_class=type(pipeline).__name__,
        step_index=cut,
        total_steps=total_steps,
        height=height,
        width=width,
        state=state,
        guidance_scale=params["guidance_scale"],
        attention_kwargs=params["attention_kwargs"],
        metadata={"output_type_requested_at_capture": params["output_type"]},
    )


def _chroma_conditioners(phase: Any, embeds: Any) -> dict[str, Any]:
    values = phase._device_feed_tensors(_require_embeds(embeds))
    required = {
        "prompt_embeds": "condition_prompt_embeds",
        "prompt_attention_mask": "condition_prompt_attention_mask",
        "negative_prompt_embeds": "condition_negative_prompt_embeds",
        "negative_prompt_attention_mask": "condition_negative_prompt_attention_mask",
    }
    result: dict[str, Any] = {}
    for source, target in required.items():
        value = values.get(source)
        if value is None:
            _error(f"Chroma capture requires {source}")
        _one_row(value, source)
        result[target] = value
    # PhasePipeline intentionally drops derived IDs from encode_prompt.  They
    # are deterministic zeros for Chroma and are nevertheless stored as a
    # first-class slot because the family contract consumes them explicitly.
    prompt_ids = _torch().zeros(
        result["condition_prompt_embeds"].shape[1],
        3,
        device=phase._device,
        dtype=result["condition_prompt_embeds"].dtype,
    )
    negative_ids = _torch().zeros(
        result["condition_negative_prompt_embeds"].shape[1],
        3,
        device=phase._device,
        dtype=result["condition_negative_prompt_embeds"].dtype,
    )
    result["condition_text_ids"] = prompt_ids
    result["condition_negative_text_ids"] = negative_ids
    return result


@_inference_function
def _capture_chroma(
    phase: Any,
    embeds: Any,
    *,
    cut_step: int,
    initial_latents: Any | None,
    reference_image: Any | None,
    kwargs: Mapping[str, Any],
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> DiffusionTrajectoryCheckpoint:
    if reference_image is not None:
        _error(
            "Chroma trajectory checkpoints do not support raw reference_image "
            "conditioning; pass encoded IP-adapter state explicitly"
        )
    pipeline = phase.pipeline
    params = _common_params(
        phase, kwargs, initial_latents=initial_latents, default_steps=35, default_guidance=5.0
    )
    # Diffusers' Chroma pipeline normalizes an omitted mapping before the
    # transformer boundary. Preserve that native call shape in every branch.
    if params["joint_attention_kwargs"] is None:
        params["joint_attention_kwargs"] = {}
    height, width = _default_resolution(pipeline, family="chroma")
    height = int(params["height"] if params["height"] is not None else height)
    width = int(params["width"] if params["width"] is not None else width)
    if height <= 0 or width <= 0:
        _error("Chroma trajectory resolution must be positive")
    requested_steps = params["total_steps"]
    if params["params"].pop("image", None) is not None:
        _error("Chroma image conditioning must be pre-encoded for trajectory capture")
    if params["params"].pop("ip_adapter_image", None) is not None:
        _error("Chroma raw IP-adapter images are not checkpoint-safe; pass image embeds")
    conditioners = _chroma_conditioners(phase, embeds)
    params_map = params["params"]
    # IP-Adapter conditioning changes the transformer attention kwargs and is
    # not part of Chroma's v1 slot schema yet.  Reject both raw and pre-encoded
    # forms rather than capturing bytes that the replay loop would ignore.
    for name in ("ip_adapter_image_embeds", "negative_ip_adapter_image_embeds"):
        value = params_map.pop(name, None)
        if value is not None:
            _error(
                "Chroma IP-Adapter conditioning is not checkpoint-safe in the "
                "v1 slot schema; omit adapter inputs"
            )
    _check_unknown(params_map)
    if params["eta"] is not None:
        _error("Chroma native scheduler does not accept eta")
    transformer_config = getattr(pipeline.transformer, "config", None)
    channels = int(getattr(transformer_config, "in_channels", 16)) // 4
    latents_result = _prepare_latents(
        pipeline,
        batch_size=1,
        channels=channels,
        height=height,
        width=width,
        dtype=conditioners["condition_prompt_embeds"].dtype,
        device=phase._device,
        generator=params["generator"],
        latents=params["initial_latents"],
    )
    if isinstance(latents_result, (tuple, list)):
        latents, latent_image_ids = (
            latents_result[0],
            latents_result[1] if len(latents_result) > 1 else None,
        )
    else:
        latents, latent_image_ids = latents_result, None
    latents = latents.to(phase._device)
    if latent_image_ids is None:
        helper = getattr(pipeline, "_prepare_latent_image_ids", None)
        if not callable(helper):
            _error("Chroma pipeline did not return or expose latent image ids")
        latent_image_ids = _invoke(
            helper,
            {
                "batch_size": 1,
                # prepare_latents packs a raw latent grid whose dimensions are
                # two times the image-id grid dimensions.
                "height": height // (int(getattr(pipeline, "vae_scale_factor", 8)) * 2),
                "width": width // (int(getattr(pipeline, "vae_scale_factor", 8)) * 2),
                "device": phase._device,
                "dtype": latents.dtype,
            },
        )
    if latent_image_ids.ndim == 3 and int(latent_image_ids.shape[0]) == 1:
        latent_image_ids = latent_image_ids[0]
    latent_image_ids = latent_image_ids.to(phase._device)
    position_ids = _torch().cat([conditioners["condition_text_ids"], latent_image_ids], dim=0)
    negative_position_ids = _torch().cat(
        [conditioners["condition_negative_text_ids"], latent_image_ids], dim=0
    )
    sequence_length = int(latents.shape[1])
    prepare_mask = getattr(pipeline, "_prepare_attention_mask", None)
    if callable(prepare_mask):
        attention_mask = _invoke(
            prepare_mask,
            {
                "batch_size": int(latents.shape[0]),
                "sequence_length": sequence_length,
                "dtype": latents.dtype,
                "attention_mask": conditioners["condition_prompt_attention_mask"],
            },
        )
        negative_attention_mask = _invoke(
            prepare_mask,
            {
                "batch_size": int(latents.shape[0]),
                "sequence_length": sequence_length,
                "dtype": latents.dtype,
                "attention_mask": conditioners["condition_negative_prompt_attention_mask"],
            },
        )
    else:
        ones = _torch().ones(
            int(latents.shape[0]), sequence_length, device=phase._device, dtype=_torch().bool
        )
        attention_mask = _torch().cat(
            [conditioners["condition_prompt_attention_mask"], ones], dim=1
        )
        negative_attention_mask = _torch().cat(
            [conditioners["condition_negative_prompt_attention_mask"], ones], dim=1
        )
    conditioners.update(
        {
            "layout_latent_image_ids": latent_image_ids,
            "layout_position_ids": position_ids,
            "layout_negative_position_ids": negative_position_ids,
            "layout_attention_mask": attention_mask,
            "layout_negative_attention_mask": negative_attention_mask,
        }
    )
    sigmas = params["sigmas"]
    if sigmas is None and params["timesteps"] is None:
        sigmas = _default_sigmas(requested_steps)
    config = getattr(pipeline.scheduler, "config", {})
    get = (
        config.get
        if hasattr(config, "get")
        else lambda name, default=None: getattr(config, name, default)
    )
    mu = _calculate_shift(
        int(latents.shape[1]),
        get("base_image_seq_len", 256),
        get("max_image_seq_len", 4096),
        get("base_shift", 0.5),
        get("max_shift", 1.15),
    )
    timesteps, resolved_steps = _prepare_scheduler(
        pipeline,
        total_steps=requested_steps,
        device=phase._device,
        timesteps=params["timesteps"],
        sigmas=sigmas,
        mu=mu,
    )
    total_steps = int(resolved_steps)
    cut = _validate_cut(cut_step, total_steps)
    scheduler = pipeline.scheduler
    _set_begin_index(scheduler, 0)
    pipeline._guidance_scale = params["guidance_scale"]
    pipeline._joint_attention_kwargs = params["joint_attention_kwargs"]
    cfg = params["guidance_scale"] > 1.0
    transformer = getattr(pipeline, "transformer", None)
    if not callable(transformer):
        _error("Chroma pipeline has no callable transformer")
    for index, timestep in enumerate(timesteps):
        if index >= cut:
            break
        before = latents
        t = _chroma_timestep(
            timestep,
            batch_size=int(latents.shape[0]),
            dtype=latents.dtype,
        )
        cond = _call_output(
            _invoke(
                transformer,
                {
                    "hidden_states": latents,
                    "timestep": t,
                    "encoder_hidden_states": conditioners["condition_prompt_embeds"],
                    "txt_ids": conditioners["condition_text_ids"],
                    "img_ids": latent_image_ids,
                    "joint_attention_kwargs": params["joint_attention_kwargs"],
                    "attention_mask": attention_mask,
                    "return_dict": False,
                },
            )
        )
        neg = None
        noise = cond
        if cfg:
            neg = _call_output(
                _invoke(
                    transformer,
                    {
                        "hidden_states": latents,
                        "timestep": t,
                        "encoder_hidden_states": conditioners["condition_negative_prompt_embeds"],
                        "txt_ids": conditioners["condition_negative_text_ids"],
                        "img_ids": latent_image_ids,
                        "joint_attention_kwargs": params["joint_attention_kwargs"],
                        "attention_mask": negative_attention_mask,
                        "return_dict": False,
                    },
                )
            )
            noise = neg + params["guidance_scale"] * (cond - neg)
        latents_dtype = latents.dtype
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=params["generator"],
            eta=params["eta"],
            native_extra_step_kwargs=False,
        )
        latents = _restore_mps_latent_dtype(latents, latents_dtype)
        _observe(
            step_observer,
            phase="chroma",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
            positive_noise=cond,
            negative_noise=neg,
        )
    slots: dict[str, Any] = {"latents": latents, **conditioners}
    _schedule_slots(scheduler, slots)
    _capture_generator_state(params["generator"], slots)
    _capture_global_rng_state(slots)
    scheduler_paths = _snapshot_scheduler(scheduler, slots)
    state_metadata = {
        "family": "chroma",
        "model_identity": str(phase._model_identity),
        "latent_layout": "packed_btd",
        "conditioner_mapping": {
            "prompt_embeds": "condition_prompt_embeds",
            "prompt_attention_mask": "condition_prompt_attention_mask",
            "negative_prompt_embeds": "condition_negative_prompt_embeds",
            "negative_prompt_attention_mask": "condition_negative_prompt_attention_mask",
            "text_ids": "condition_text_ids",
            "negative_text_ids": "condition_negative_text_ids",
        },
        "layout_mapping": {
            "latent_image_ids": "layout_latent_image_ids",
            "position_ids": "layout_position_ids",
            "negative_position_ids": "layout_negative_position_ids",
            "attention_mask": "layout_attention_mask",
            "negative_attention_mask": "layout_negative_attention_mask",
        },
        "cfg_enabled": cfg,
        "guidance_convention": "negative_plus_scale_positive_minus_negative",
        "guidance_scale": params["guidance_scale"],
        "schedule_sigmas_explicit": _jsonable(sigmas),
        "scheduler_state_paths": scheduler_paths,
        **_schedule_metadata(
            scheduler,
            requested_num_inference_steps=requested_steps,
            requested_timesteps=params["timesteps"],
            requested_sigmas=sigmas,
            mu=mu,
            schedule_source="timesteps" if params["timesteps"] is not None else "sigmas",
        ),
    }
    state = DiffusionTrajectoryState("chroma", slots, state_metadata)
    return DiffusionTrajectoryCheckpoint(
        checkpoint_id=_checkpoint_id(pipeline, state, cut, total_steps, height, width),
        pipeline_class=type(pipeline).__name__,
        step_index=cut,
        total_steps=total_steps,
        height=height,
        width=width,
        state=state,
        guidance_scale=params["guidance_scale"],
        attention_kwargs=params["joint_attention_kwargs"],
        metadata={"output_type_requested_at_capture": params["output_type"]},
    )


def capture_nonflux_checkpoint(
    phase: Any,
    embeds: Any,
    *,
    cut_step: int,
    initial_latents: Any | None,
    reference_image: Any | None,
    references: Sequence[str] = (),
    kwargs: Mapping[str, Any],
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> DiffusionTrajectoryCheckpoint:
    if tuple(references):
        _error("non-FLUX trajectory checkpoints do not support opaque reference ids")
    family = NON_FLUX_PIPELINES.get(phase._class_name)
    if family == "sdxl":
        if reference_image is not None:
            _error(
                "SDXL trajectory checkpoints do not support raw reference_image "
                "conditioning; img2img state is not registered in the v1 ABI"
            )
        return _capture_sdxl(
            phase,
            embeds,
            cut_step=cut_step,
            initial_latents=initial_latents,
            kwargs=kwargs,
            step_observer=step_observer,
        )
    if family == "krea2":
        if reference_image is not None:
            _error("Krea2 trajectory checkpoints do not support raw reference_image conditioning")
        return _capture_krea(
            phase,
            embeds,
            cut_step=cut_step,
            initial_latents=initial_latents,
            kwargs=kwargs,
            step_observer=step_observer,
        )
    if family == "chroma":
        return _capture_chroma(
            phase,
            embeds,
            cut_step=cut_step,
            initial_latents=initial_latents,
            reference_image=reference_image,
            kwargs=kwargs,
            step_observer=step_observer,
        )
    _error(
        f"no native non-FLUX trajectory contract for {phase._class_name!r}; "
        "refusing to route it through a different family"
    )


def _restore_state_inputs(
    phase: Any,
    checkpoint: DiffusionTrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    generator: Any,
) -> tuple[Any, dict[str, Any], Any]:
    _validate_checkpoint(phase, checkpoint)
    # Reject a different scheduler before restoring the process RNG or a
    # caller-supplied generator.  Scheduler identity is part of the replay
    # contract, not merely diagnostic metadata.
    _validate_scheduler_identity(phase.pipeline.scheduler, checkpoint.state.metadata)
    latents = checkpoint.slot("latents")
    if latents is None or not _is_tensor(latents):
        _error("checkpoint is missing tensor slot 'latents'")
    latents = latents.to(phase._device).detach().clone()
    if latent_override is not None:
        if not _is_tensor(latent_override):
            _error("latent_override must be a torch tensor")
        candidate = latent_override.to(phase._device)
        if tuple(candidate.shape) != tuple(latents.shape) or candidate.dtype != latents.dtype:
            _error("latent_override shape/dtype does not match checkpoint latents")
        latents = candidate.detach().clone()
    conditioners = {
        name: value.to(phase._device).detach().clone() if _is_tensor(value) else value
        for name, value in checkpoint.slots.items()
        if name.startswith("condition_") or name.startswith("layout_")
    }
    conditioners = _condition_override(phase, prompt_embeds_override, conditioners)
    if checkpoint.family == "sdxl":
        conditioners = _rebuild_sdxl_conditioners(
            conditioners,
            cfg=bool(checkpoint.state.metadata.get("cfg_enabled")),
        )
    restored_generator = _restore_generator(phase, checkpoint, generator)
    _restore_scheduler(phase, checkpoint)
    # Scheduler setup is normally deterministic, but restore process-global
    # streams last so any implementation-specific setup cannot consume the
    # captured stream before the first replay denoiser call.
    _restore_global_rng_state(checkpoint)
    return latents, conditioners, restored_generator


def _decode_sdxl(
    pipeline: Any, latents: Any, checkpoint: DiffusionTrajectoryCheckpoint, output_type: str
) -> Any:
    torch = _torch()
    if output_type == "latent":
        return latents
    vae = pipeline.vae
    decode_latents = latents
    config = getattr(vae, "config", SimpleNamespace(scaling_factor=0.18215))
    needs_upcasting = getattr(vae, "dtype", None) == torch.float16 and bool(
        getattr(config, "force_upcast", False)
    )
    if needs_upcasting:
        mover = getattr(vae, "to", None)
        if not callable(mover):
            _error("SDXL VAE requires float32 upcasting but has no to() method")
        with torch.inference_mode():
            mover(dtype=torch.float32)
        # Native SDXL decodes using the post-quantization convolution's dtype
        # after upcasting.  Most VAEs are fully float32 at this point; retain
        # the explicit lookup for mixed-precision implementations.
        post_quant = getattr(vae, "post_quant_conv", None)
        try:
            decode_dtype = next(post_quant.parameters()).dtype
        except (AttributeError, StopIteration):
            decode_dtype = torch.float32
        decode_latents = decode_latents.to(dtype=decode_dtype)
    elif getattr(vae, "dtype", None) is not None and decode_latents.dtype != vae.dtype:
        # Match the native MPS workaround without changing CUDA/CPU VAE
        # placement or dtype policy.
        if (
            getattr(getattr(torch, "backends", None), "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            mover = getattr(vae, "to", None)
            if callable(mover):
                with torch.inference_mode():
                    mover(dtype=decode_latents.dtype)
    has_latents_mean = hasattr(config, "latents_mean") and config.latents_mean is not None
    has_latents_std = hasattr(config, "latents_std") and config.latents_std is not None
    if has_latents_mean and has_latents_std:
        mean = torch.tensor(
            config.latents_mean, device=decode_latents.device, dtype=decode_latents.dtype
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            config.latents_std, device=decode_latents.device, dtype=decode_latents.dtype
        ).view(1, -1, 1, 1)
        decode_latents = decode_latents * std / float(config.scaling_factor) + mean
    else:
        decode_latents = decode_latents / float(getattr(config, "scaling_factor", 0.18215))
    image = _call_output(_decode_vae(vae, decode_latents))
    if needs_upcasting:
        mover = getattr(vae, "to", None)
        if callable(mover):
            with torch.inference_mode():
                mover(dtype=torch.float16)
    watermark = getattr(pipeline, "watermark", None)
    apply_watermark = getattr(watermark, "apply_watermark", None)
    if callable(apply_watermark):
        with torch.inference_mode():
            image = apply_watermark(image)
    processor = getattr(pipeline, "image_processor", None)
    return processor.postprocess(image, output_type=output_type) if processor is not None else image


def _decode_vae(vae: Any, latents: Any) -> Any:
    """Decode through the native positional ``z``/``latents`` VAE boundary."""

    decode = getattr(vae, "decode", None)
    if not callable(decode):
        _error("native VAE has no decode() method")
    try:
        parameters = inspect.signature(decode).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {}
    if "return_dict" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        kwargs["return_dict"] = False
    with _torch().inference_mode():
        return decode(latents, **kwargs)


def _decode_krea(
    pipeline: Any, latents: Any, checkpoint: DiffusionTrajectoryCheckpoint, output_type: str
) -> Any:
    if output_type == "latent":
        return latents
    unpack = getattr(pipeline, "_unpack_latents", None)
    if not callable(unpack):
        _error("Krea2 pipeline has no native _unpack_latents()")
    latents = _invoke(
        unpack, {"latents": latents, "height": checkpoint.height, "width": checkpoint.width}
    )
    vae = pipeline.vae
    latents = latents.to(getattr(vae, "dtype", latents.dtype))
    config = vae.config
    mean = (
        _torch()
        .tensor(config.latents_mean, device=latents.device, dtype=latents.dtype)
        .view(1, config.z_dim, 1, 1, 1)
    )
    std = 1.0 / _torch().tensor(
        config.latents_std, device=latents.device, dtype=latents.dtype
    ).view(1, config.z_dim, 1, 1, 1)
    latents = latents / std + mean
    image = _call_output(_decode_vae(vae, latents))[:, :, 0]
    processor = getattr(pipeline, "image_processor", None)
    return processor.postprocess(image, output_type=output_type) if processor is not None else image


def _decode_chroma(
    pipeline: Any, latents: Any, checkpoint: DiffusionTrajectoryCheckpoint, output_type: str
) -> Any:
    if output_type == "latent":
        return latents
    unpack = getattr(pipeline, "_unpack_latents", None)
    if not callable(unpack):
        _error("Chroma pipeline has no native _unpack_latents()")
    latents = _invoke(
        unpack,
        {
            "latents": latents,
            "height": checkpoint.height,
            "width": checkpoint.width,
            "vae_scale_factor": pipeline.vae_scale_factor,
        },
    )
    config = pipeline.vae.config
    latents = latents / float(config.scaling_factor) + float(config.shift_factor)
    image = _call_output(_decode_vae(pipeline.vae, latents))
    processor = getattr(pipeline, "image_processor", None)
    return processor.postprocess(image, output_type=output_type) if processor is not None else image


@_inference_function
def _resume_sdxl(
    phase: Any,
    checkpoint: DiffusionTrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    output_type: str,
    generator: Any,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> Any:
    pipeline = phase.pipeline
    latents, cond, generator = _restore_state_inputs(
        phase,
        checkpoint,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        generator=generator,
    )
    metadata = checkpoint.state.metadata
    scheduler = pipeline.scheduler
    cfg = bool(metadata.get("cfg_enabled"))
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._guidance_rescale = float(metadata.get("guidance_rescale", 0.0))
    for index in range(checkpoint.step_index, checkpoint.total_steps):
        timestep = checkpoint.slot("schedule_timesteps")[index].to(phase._device)
        model_input = torch_cat(latents, cfg=cfg)
        scale = getattr(scheduler, "scale_model_input", None)
        if callable(scale):
            model_input = scale(model_input, timestep)
        noise = _call_output(
            _invoke(
                pipeline.unet,
                {
                    "sample": model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": cond["condition_prompt_embeds"],
                    "timestep_cond": cond.get("condition_timestep_cond"),
                    "cross_attention_kwargs": checkpoint.attention_kwargs,
                    "added_cond_kwargs": {
                        "text_embeds": cond["condition_add_text_embeds"],
                        "time_ids": cond["condition_add_time_ids"],
                    },
                    "return_dict": False,
                },
            )
        )
        if cfg:
            uncond, text = noise.chunk(2)
            noise = uncond + checkpoint.guidance_scale * (text - uncond)
            noise = _rescale_noise_cfg(noise, text, pipeline._guidance_rescale)
        before = latents
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=generator,
            eta=metadata.get("eta"),
        )
        _observe(
            step_observer,
            phase="sdxl",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
        )
    return SimpleNamespace(images=_decode_sdxl(pipeline, latents, checkpoint, output_type))


@_inference_function
def _resume_krea(
    phase: Any,
    checkpoint: DiffusionTrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    output_type: str,
    generator: Any,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> Any:
    pipeline = phase.pipeline
    latents, cond, generator = _restore_state_inputs(
        phase,
        checkpoint,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        generator=generator,
    )
    scheduler = pipeline.scheduler
    cfg = bool(checkpoint.state.metadata.get("cfg_enabled"))
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._attention_kwargs = checkpoint.attention_kwargs
    config = getattr(scheduler, "config", {})
    get = (
        config.get
        if hasattr(config, "get")
        else lambda name, default=None: getattr(config, name, default)
    )
    for index in range(checkpoint.step_index, checkpoint.total_steps):
        timestep = checkpoint.slot("schedule_timesteps")[index].to(phase._device)
        t = (
            (timestep / float(get("num_train_timesteps", 1000)))
            .expand(latents.shape[0])
            .to(latents.dtype)
        )
        cond_noise = _call_output(
            _invoke(
                pipeline.transformer,
                {
                    "hidden_states": latents,
                    "encoder_hidden_states": cond["condition_prompt_embeds"],
                    "timestep": t,
                    "position_ids": cond["layout_position_ids"],
                    "encoder_attention_mask": cond["condition_prompt_embeds_mask"],
                    "attention_kwargs": pipeline._attention_kwargs,
                    "return_dict": False,
                },
            )
        )
        neg_noise = None
        noise = cond_noise
        if cfg:
            neg_noise = _call_output(
                _invoke(
                    pipeline.transformer,
                    {
                        "hidden_states": latents,
                        "encoder_hidden_states": cond["condition_negative_prompt_embeds"],
                        "timestep": t,
                        "position_ids": cond["layout_position_ids"],
                        "encoder_attention_mask": cond["condition_negative_prompt_embeds_mask"],
                        "attention_kwargs": pipeline._attention_kwargs,
                        "return_dict": False,
                    },
                )
            )
            noise = cond_noise + checkpoint.guidance_scale * (cond_noise - neg_noise)
        before = latents
        latents_dtype = latents.dtype
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=generator,
            eta=None,
            native_extra_step_kwargs=False,
        )
        latents = _restore_mps_latent_dtype(latents, latents_dtype)
        _observe(
            step_observer,
            phase="krea2",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
            positive_noise=cond_noise,
            negative_noise=neg_noise,
        )
    return SimpleNamespace(images=_decode_krea(pipeline, latents, checkpoint, output_type))


@_inference_function
def _resume_chroma(
    phase: Any,
    checkpoint: DiffusionTrajectoryCheckpoint,
    *,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    output_type: str,
    generator: Any,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> Any:
    pipeline = phase.pipeline
    latents, cond, generator = _restore_state_inputs(
        phase,
        checkpoint,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        generator=generator,
    )
    scheduler = pipeline.scheduler
    metadata = checkpoint.state.metadata
    cfg = bool(metadata.get("cfg_enabled"))
    pipeline._guidance_scale = checkpoint.guidance_scale
    pipeline._joint_attention_kwargs = checkpoint.attention_kwargs
    for index in range(checkpoint.step_index, checkpoint.total_steps):
        timestep = checkpoint.slot("schedule_timesteps")[index].to(phase._device)
        t = _chroma_timestep(
            timestep,
            batch_size=int(latents.shape[0]),
            dtype=latents.dtype,
        )
        cond_noise = _call_output(
            _invoke(
                pipeline.transformer,
                {
                    "hidden_states": latents,
                    "timestep": t,
                    "encoder_hidden_states": cond["condition_prompt_embeds"],
                    "txt_ids": cond["condition_text_ids"],
                    "img_ids": cond["layout_latent_image_ids"],
                    "joint_attention_kwargs": pipeline._joint_attention_kwargs,
                    "attention_mask": cond["layout_attention_mask"],
                    "return_dict": False,
                },
            )
        )
        neg_noise = None
        noise = cond_noise
        if cfg:
            neg_noise = _call_output(
                _invoke(
                    pipeline.transformer,
                    {
                        "hidden_states": latents,
                        "timestep": t,
                        "encoder_hidden_states": cond["condition_negative_prompt_embeds"],
                        "txt_ids": cond["condition_negative_text_ids"],
                        "img_ids": cond["layout_latent_image_ids"],
                        "joint_attention_kwargs": pipeline._joint_attention_kwargs,
                        "attention_mask": cond["layout_negative_attention_mask"],
                        "return_dict": False,
                    },
                )
            )
            noise = neg_noise + checkpoint.guidance_scale * (cond_noise - neg_noise)
        before = latents
        latents_dtype = latents.dtype
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=generator,
            eta=None,
            native_extra_step_kwargs=False,
        )
        latents = _restore_mps_latent_dtype(latents, latents_dtype)
        _observe(
            step_observer,
            phase="chroma",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
            positive_noise=cond_noise,
            negative_noise=neg_noise,
        )
    return SimpleNamespace(images=_decode_chroma(pipeline, latents, checkpoint, output_type))


def resume_nonflux_checkpoint(
    phase: Any,
    checkpoint: DiffusionTrajectoryCheckpoint,
    *,
    latent_override: Any | None = None,
    prompt_embeds_override: Any | None = None,
    output_type: str = "pil",
    generator: Any = None,
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    _validate_checkpoint(phase, checkpoint)
    if output_type not in {"latent", "pil", "np"}:
        _error(f"unsupported trajectory output_type {output_type!r}")
    family = checkpoint.family
    resume = {"sdxl": _resume_sdxl, "krea2": _resume_krea, "chroma": _resume_chroma}[family]
    return resume(
        phase,
        checkpoint,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        output_type=output_type,
        generator=generator,
        step_observer=step_observer,
    )


def advance_nonflux_checkpoint(
    phase: Any,
    checkpoint: DiffusionTrajectoryCheckpoint,
    *,
    steps: int,
    latent_override: Any | None = None,
    prompt_embeds_override: Any | None = None,
    action_provider: Callable[[Mapping[str, Any]], Any] | None = None,
    step_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> DiffusionTrajectoryCheckpoint:
    _validate_checkpoint(phase, checkpoint)
    if isinstance(steps, bool) or int(steps) <= 0:
        _error("steps must be a positive integer")
    steps = int(steps)
    end = checkpoint.step_index + steps
    if end > checkpoint.total_steps:
        _error(
            f"cannot advance checkpoint from step {checkpoint.step_index} by "
            f"{steps}; trajectory ends at {checkpoint.total_steps}"
        )
    # Run only the requested suffix by using the family resume loops against a
    # temporary checkpoint with a bounded total.  The loops need the original
    # schedule index, so a small shared implementation is clearer: resume to
    # the requested boundary with output_type=latent, then snapshot the live
    # scheduler and RNG from that boundary.
    _run_to = {"sdxl": _advance_sdxl, "krea2": _advance_krea, "chroma": _advance_chroma}[
        checkpoint.family
    ]
    return _run_to(
        phase,
        checkpoint,
        end=end,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        action_provider=action_provider,
        step_observer=step_observer,
    )


def _new_child_checkpoint(
    phase: Any,
    parent: DiffusionTrajectoryCheckpoint,
    *,
    end: int,
    latents: Any,
    cond: Mapping[str, Any],
    generator: Any,
) -> DiffusionTrajectoryCheckpoint:
    slots: dict[str, Any] = {name: _clone(value) for name, value in parent.slots.items()}
    slots["latents"] = _clone(latents)
    # ``cond`` is the post-intervention, device-ready conditioner mapping.  A
    # branch that advances with prompt_embeds_override must carry those updated
    # semantic and derived slots into the child; copying only the parent would
    # make the first transition look intervened while later resume steps revert
    # to the old prompt.
    for name, value in cond.items():
        if name.startswith("condition_") or name.startswith("layout_"):
            slots[name] = _clone(value)
    _capture_generator_state(generator, slots)
    _capture_global_rng_state(slots)
    scheduler = phase.pipeline.scheduler
    _schedule_slots(scheduler, slots)
    paths = _snapshot_scheduler(scheduler, slots)
    metadata = dict(parent.state.metadata)
    metadata.update(
        {
            "parent_checkpoint_id": parent.checkpoint_id,
            "parent_checkpoint_fingerprint": parent.fingerprint,
            "source_step_index": parent.step_index,
            "steps_advanced": end - parent.step_index,
            "scheduler_state_paths": paths,
            "scheduler_begin_index": getattr(scheduler, "_begin_index", None),
            "scheduler_step_index": getattr(scheduler, "_step_index", None),
            "transition": "mrun.diffusion.advance_checkpoint:v2",
        }
    )
    state = DiffusionTrajectoryState(parent.family, slots, metadata)
    return DiffusionTrajectoryCheckpoint(
        checkpoint_id=_checkpoint_id(
            phase.pipeline, state, end, parent.total_steps, parent.height, parent.width
        ),
        pipeline_class=parent.pipeline_class,
        step_index=end,
        total_steps=parent.total_steps,
        height=parent.height,
        width=parent.width,
        state=state,
        guidance_scale=parent.guidance_scale,
        attention_kwargs=parent.attention_kwargs,
        metadata=parent.metadata,
    )


@_inference_function
def _advance_sdxl(
    phase: Any,
    parent: DiffusionTrajectoryCheckpoint,
    *,
    end: int,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    action_provider: Callable[[Mapping[str, Any]], Any] | None,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> DiffusionTrajectoryCheckpoint:
    latents, cond, generator = _restore_state_inputs(
        phase,
        parent,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        generator=None,
    )
    scheduler = phase.pipeline.scheduler
    cfg = bool(parent.state.metadata.get("cfg_enabled"))
    for index in range(parent.step_index, end):
        timestep = parent.slot("schedule_timesteps")[index].to(phase._device)
        if action_provider is None:
            model_input = torch_cat(latents, cfg=cfg)
            if callable(getattr(scheduler, "scale_model_input", None)):
                model_input = scheduler.scale_model_input(model_input, timestep)
            scaled_latents = model_input.chunk(2)[0] if cfg else model_input
            noise = _call_output(
                _invoke(
                    phase.pipeline.unet,
                    {
                        "sample": model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": cond["condition_prompt_embeds"],
                        "timestep_cond": cond.get("condition_timestep_cond"),
                        "cross_attention_kwargs": parent.attention_kwargs,
                        "added_cond_kwargs": {
                            "text_embeds": cond["condition_add_text_embeds"],
                            "time_ids": cond["condition_add_time_ids"],
                        },
                        "return_dict": False,
                    },
                )
            )
            if cfg:
                uncond, text = noise.chunk(2)
                noise = uncond + parent.guidance_scale * (text - uncond)
                noise = _rescale_noise_cfg(
                    noise, text, float(parent.state.metadata.get("guidance_rescale", 0.0))
                )
            action_source = "native-denoiser"
        else:
            # Preserve the native scheduler input cursor and expose the exact
            # scaled denoiser input.  The provider still returns one final
            # post-CFG action shaped like ``latents``; no second CFG pass is
            # applied to that action.
            model_input = torch_cat(latents, cfg=cfg)
            if callable(getattr(scheduler, "scale_model_input", None)):
                model_input = scheduler.scale_model_input(model_input, timestep)
            scaled_latents = model_input.chunk(2)[0] if cfg else model_input
            noise = action_provider(
                {
                    "family": "sdxl",
                    "step_index": index,
                    "timestep": _clone(timestep),
                    "latents_before": _clone(latents),
                    "model_input": _clone(model_input),
                    "scaled_latents": _clone(scaled_latents),
                    "checkpoint": parent,
                    "checkpoint_fingerprint": parent.fingerprint,
                }
            )
            if not _is_tensor(noise):
                _error("action_provider must return a tensor")
            if tuple(noise.shape) != tuple(latents.shape):
                _error(
                    "action_provider returned shape "
                    f"{tuple(noise.shape)}, expected {tuple(latents.shape)}"
                )
            if noise.device != latents.device or noise.dtype != latents.dtype:
                _error(
                    "action_provider returned device/dtype "
                    f"{noise.device}/{noise.dtype}, expected {latents.device}/{latents.dtype}"
                )
            if not bool(_torch().isfinite(noise).all()):
                _error("action_provider returned non-finite values")
            action_source = "external-action-provider"
            action_sha256 = _tensor_fingerprint(noise)
        before = latents
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=generator,
            eta=parent.state.metadata.get("eta"),
        )
        _observe(
            step_observer,
            phase="sdxl",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
            action_source=action_source,
            action_sha256=action_sha256 if action_provider is not None else None,
            model_input=model_input,
            scaled_latents=scaled_latents,
        )
    return _new_child_checkpoint(
        phase, parent, end=end, latents=latents, cond=cond, generator=generator
    )


@_inference_function
def _advance_krea(
    phase: Any,
    parent: DiffusionTrajectoryCheckpoint,
    *,
    end: int,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    action_provider: Callable[[Mapping[str, Any]], Any] | None,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> DiffusionTrajectoryCheckpoint:
    if action_provider is not None:
        _error("action_provider is not registered for krea2")
    latents, cond, generator = _restore_state_inputs(
        phase,
        parent,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        generator=None,
    )
    scheduler = phase.pipeline.scheduler
    cfg = bool(parent.state.metadata.get("cfg_enabled"))
    config = getattr(scheduler, "config", {})
    get = (
        config.get
        if hasattr(config, "get")
        else lambda name, default=None: getattr(config, name, default)
    )
    for index in range(parent.step_index, end):
        timestep = parent.slot("schedule_timesteps")[index].to(phase._device)
        t = (
            (timestep / float(get("num_train_timesteps", 1000)))
            .expand(latents.shape[0])
            .to(latents.dtype)
        )
        cond_noise = _call_output(
            _invoke(
                phase.pipeline.transformer,
                {
                    "hidden_states": latents,
                    "encoder_hidden_states": cond["condition_prompt_embeds"],
                    "timestep": t,
                    "position_ids": cond["layout_position_ids"],
                    "encoder_attention_mask": cond["condition_prompt_embeds_mask"],
                    "attention_kwargs": parent.attention_kwargs,
                    "return_dict": False,
                },
            )
        )
        neg_noise = None
        noise = cond_noise
        if cfg:
            neg_noise = _call_output(
                _invoke(
                    phase.pipeline.transformer,
                    {
                        "hidden_states": latents,
                        "encoder_hidden_states": cond["condition_negative_prompt_embeds"],
                        "timestep": t,
                        "position_ids": cond["layout_position_ids"],
                        "encoder_attention_mask": cond["condition_negative_prompt_embeds_mask"],
                        "attention_kwargs": parent.attention_kwargs,
                        "return_dict": False,
                    },
                )
            )
            noise = cond_noise + parent.guidance_scale * (cond_noise - neg_noise)
        before = latents
        latents_dtype = latents.dtype
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=generator,
            eta=None,
            native_extra_step_kwargs=False,
        )
        latents = _restore_mps_latent_dtype(latents, latents_dtype)
        _observe(
            step_observer,
            phase="krea2",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
            positive_noise=cond_noise,
            negative_noise=neg_noise,
        )
    return _new_child_checkpoint(
        phase, parent, end=end, latents=latents, cond=cond, generator=generator
    )


@_inference_function
def _advance_chroma(
    phase: Any,
    parent: DiffusionTrajectoryCheckpoint,
    *,
    end: int,
    latent_override: Any | None,
    prompt_embeds_override: Any | None,
    action_provider: Callable[[Mapping[str, Any]], Any] | None,
    step_observer: Callable[[Mapping[str, Any]], None] | None,
) -> DiffusionTrajectoryCheckpoint:
    if action_provider is not None:
        _error("action_provider is not registered for chroma")
    latents, cond, generator = _restore_state_inputs(
        phase,
        parent,
        latent_override=latent_override,
        prompt_embeds_override=prompt_embeds_override,
        generator=None,
    )
    scheduler = phase.pipeline.scheduler
    cfg = bool(parent.state.metadata.get("cfg_enabled"))
    for index in range(parent.step_index, end):
        timestep = parent.slot("schedule_timesteps")[index].to(phase._device)
        t = _chroma_timestep(
            timestep,
            batch_size=int(latents.shape[0]),
            dtype=latents.dtype,
        )
        cond_noise = _call_output(
            _invoke(
                phase.pipeline.transformer,
                {
                    "hidden_states": latents,
                    "timestep": t,
                    "encoder_hidden_states": cond["condition_prompt_embeds"],
                    "txt_ids": cond["condition_text_ids"],
                    "img_ids": cond["layout_latent_image_ids"],
                    "joint_attention_kwargs": parent.attention_kwargs,
                    "attention_mask": cond["layout_attention_mask"],
                    "return_dict": False,
                },
            )
        )
        neg_noise = None
        noise = cond_noise
        if cfg:
            neg_noise = _call_output(
                _invoke(
                    phase.pipeline.transformer,
                    {
                        "hidden_states": latents,
                        "timestep": t,
                        "encoder_hidden_states": cond["condition_negative_prompt_embeds"],
                        "txt_ids": cond["condition_negative_text_ids"],
                        "img_ids": cond["layout_latent_image_ids"],
                        "joint_attention_kwargs": parent.attention_kwargs,
                        "attention_mask": cond["layout_negative_attention_mask"],
                        "return_dict": False,
                    },
                )
            )
            noise = neg_noise + parent.guidance_scale * (cond_noise - neg_noise)
        before = latents
        latents_dtype = latents.dtype
        latents = _step_scheduler(
            scheduler,
            noise=noise,
            timestep=timestep,
            latents=latents,
            generator=generator,
            eta=None,
            native_extra_step_kwargs=False,
        )
        latents = _restore_mps_latent_dtype(latents, latents_dtype)
        _observe(
            step_observer,
            phase="chroma",
            index=index,
            timestep=timestep,
            latents_before=before,
            noise_pred=noise,
            latents_after=latents,
            positive_noise=cond_noise,
            negative_noise=neg_noise,
        )
    return _new_child_checkpoint(
        phase, parent, end=end, latents=latents, cond=cond, generator=generator
    )


def resume_nonflux_checkpoint_batch(
    phase: Any,
    checkpoint: DiffusionTrajectoryCheckpoint,
    *,
    branch_ids: Sequence[str],
    latent_overrides: Any | Sequence[Any] | None,
    mode: str,
    output_type: str,
) -> Any:
    import torch

    _validate_checkpoint(phase, checkpoint)
    ids = tuple(str(value) for value in branch_ids)
    if not ids or len(set(ids)) != len(ids):
        _error("resume_checkpoint_batch branch_ids must be non-empty and unique")
    if mode not in {"exact", "batched"}:
        _error("resume_checkpoint_batch mode must be 'exact' or 'batched'")
    if mode == "batched":
        _error("non-FLUX trajectory batch fusion is not registered; use mode='exact'")
    if latent_overrides is None:
        overrides = [None] * len(ids)
    elif _is_tensor(latent_overrides):
        if latent_overrides.ndim == 0 or int(latent_overrides.shape[0]) != len(ids):
            _error("batched latent_overrides must align with branch_ids")
        overrides = [latent_overrides[index : index + 1] for index in range(len(ids))]
    else:
        overrides = list(latent_overrides)
        if len(overrides) != len(ids):
            _error("latent_overrides must align with branch_ids")
    rows = []
    for override in overrides:
        result = resume_nonflux_checkpoint(
            phase, checkpoint, latent_override=override, output_type=output_type
        )
        value = result.images
        if isinstance(value, (list, tuple)):
            rows.append(value[0])
        elif isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == 1:
            rows.append(value[0])
        else:
            rows.append(value)
    from .phase import PhaseBatchResult

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
            "physical_suffix_denoiser_calls": (checkpoint.total_steps - checkpoint.step_index)
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


__all__ = [
    "DiffusionTrajectoryCheckpoint",
    "DiffusionTrajectoryState",
    "NON_FLUX_CHECKPOINT_SCHEMA",
    "NON_FLUX_PIPELINES",
    "NON_FLUX_STATE_SCHEMA",
    "NativeDiffusionTrajectoryCheckpoint",
    "NonFluxTrajectoryCheckpoint",
    "advance_nonflux_checkpoint",
    "capture_nonflux_checkpoint",
    "resume_nonflux_checkpoint",
    "resume_nonflux_checkpoint_batch",
]
