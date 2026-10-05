"""Program-shaped serving for diffusion and flow pipelines.

This module is the runtime-first slice of the MARS/mstack proposal.  It does not
pretend that an existing Diffusers checkpoint exposes a learned spatial VM or
per-timestep denoiser ABI.  Instead it gives the current pipeline an explicit
program boundary:

``load -> link -> compile_context -> allocate -> execute -> commit ->
checkpoint/restore -> render``

The linked program owns immutable identity and compatibility.  A session owns
row-local mutable state.  A current ``PhasePipeline`` executes one complete
Diffusers schedule per ``step``; a future native model can replace that atomic
backend operation with a true denoise wave without changing the serving API.

Heavy model imports stay lazy.  The module is safe to import in scheduler and
agent environments that do not have Diffusers installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .io import (
    ComponentFrame,
    ComponentIOSpec,
    PortBinding,
    component_io_manifest,
    make_component_frame,
    payload_fingerprint,
    validate_component_io_manifest,
)
from .phase import PromptEmbeds, TrajectoryCheckpoint

PROGRAM_MANIFEST_SCHEMA = "mrun-diffusion-program-v1"
PROGRAM_STATE_SCHEMA = "mrun-diffusion-program-state-v1"
PROGRAM_CHECKPOINT_SCHEMA = "mrun-diffusion-program-checkpoint-v1"
WEIGHT_PAGE_LEASE_POLICY_SCHEMA = "mrun-diffusion-weight-page-lease-policy-v2"

_STATUSES = frozenset({"allocated", "context_compiled", "running", "completed", "closed"})


class ProgramError(RuntimeError):
    """Base class for program-link and program-session failures."""


class ProgramABIError(ProgramError):
    """A manifest, extension, backend, or state ABI did not match."""


class ProgramCapabilityError(ProgramError):
    """The selected backend cannot provide a requested program operation."""


class ProgramStateError(ProgramError):
    """An operation was attempted in an invalid session state."""


def _canonical(value: Any) -> Any:
    """Return a JSON-safe, deterministic representation or fail closed."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("program identity cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    raise TypeError(f"program identity contains unsupported value {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ProgramABIError("manifest fields must be mappings")
    return MappingProxyType(dict(_canonical(value)))


def _weight_page_lease_policy(
    resource_policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw = resource_policy.get("weight_page_lease")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProgramABIError("weight_page_lease resource policy must be a mapping")
    required = {
        "schema",
        "provider_fingerprint",
        "ordered_page_schedule",
        "device",
        "dtype",
        "numerical_lane",
        "max_retained_device_bytes",
    }
    if set(raw) != required:
        raise ProgramABIError(
            f"weight_page_lease resource policy must contain exactly {sorted(required)}"
        )
    if raw.get("schema") != WEIGHT_PAGE_LEASE_POLICY_SCHEMA:
        raise ProgramABIError("unsupported weight_page_lease resource policy schema")
    fingerprint = str(raw.get("provider_fingerprint", "")).casefold()
    if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
        raise ProgramABIError("weight_page_lease provider_fingerprint must be SHA-256")
    schedule = raw.get("ordered_page_schedule")
    if (
        not isinstance(schedule, (list, tuple))
        or not schedule
        or any(not isinstance(key, str) or not key for key in schedule)
    ):
        raise ProgramABIError(
            "weight_page_lease ordered_page_schedule must be a non-empty ordered key list"
        )
    for field_name in ("device", "dtype", "numerical_lane"):
        if not isinstance(raw.get(field_name), str) or not raw[field_name]:
            raise ProgramABIError(f"weight_page_lease {field_name} must be non-empty text")
    budget = raw.get("max_retained_device_bytes")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ProgramABIError(
            "weight_page_lease max_retained_device_bytes must be a positive integer"
        )
    return raw


def _backend_weight_page_provider(backend: Any) -> Any:
    pipeline = getattr(backend, "pipeline", backend)
    found: list[tuple[Any, Any]] = []
    for name in ("transformer", "unet"):
        denoiser = getattr(pipeline, name, None)
        provider = getattr(denoiser, "_saturn_diffusion_qstore", None)
        if provider is not None:
            found.append((denoiser, provider))
    if not found:
        raise ProgramABIError("weight_page_lease policy requires a linked paged denoiser provider")
    if any(provider is not found[0][1] for _, provider in found[1:]):
        raise ProgramABIError("linked denoisers expose different weight-page providers")
    denoiser, provider = found[0]
    report = getattr(denoiser, "_saturn_diffusion_qstore_report", None)
    if not isinstance(report, Mapping) or report.get("lease_required") is not True:
        raise ProgramABIError(
            "weight_page_lease policy requires denoiser lowering with lease_required=True"
        )
    if not callable(getattr(provider, "lease", None)):
        raise ProgramABIError("linked weight-page provider does not expose lease()")
    return provider


def _weight_page_provider_content_fingerprint(provider: Any) -> str:
    if getattr(provider, "integrity_capable", None) is not True:
        raise ProgramABIError("weight_page_lease requires an integrity-capable content provider")
    fingerprint = getattr(provider, "content_fingerprint", None)
    if not isinstance(fingerprint, str):
        raise ProgramABIError("weight_page_lease provider has no authoritative content fingerprint")
    fingerprint = fingerprint.casefold()
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ProgramABIError("weight_page_lease provider content fingerprint must be SHA-256")
    return fingerprint


def _torch_dtype(value: str) -> Any:
    import torch

    name = value.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ProgramABIError(f"unsupported weight_page_lease dtype {value!r}")
    return dtype


def _names(value: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ProgramABIError(f"{field_name} cannot contain empty names")
    if len(set(result)) != len(result):
        raise ProgramABIError(f"{field_name} must contain unique names")
    return result


@dataclass(frozen=True, slots=True)
class ProgramManifest:
    """Content-addressed identity and ABI declarations for one base program."""

    base_fingerprint: str
    component_graph: Mapping[str, Any] = field(default_factory=dict)
    conditioner_abi: str = "prompt-embeds-v1"
    latent_abi: str = "pipeline-latents-v1"
    scheduler_abi: str = "diffusers-scheduler-v1"
    vae_abi: str = "diffusers-vae-v1"
    state_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"name": PROGRAM_STATE_SCHEMA, "version": 1}
    )
    ports: Mapping[str, Any] = field(default_factory=dict)
    component_io: Mapping[str, Any] = field(default_factory=component_io_manifest)
    extensions: tuple[str, ...] = ()
    route_schema: Mapping[str, Any] = field(default_factory=dict)
    numerical_contract: Mapping[str, Any] = field(
        default_factory=lambda: {"mode": "backend-authority", "step_granularity": "atomic"}
    )
    output_contract: Mapping[str, Any] = field(
        default_factory=lambda: {"kind": "pipeline-output", "row_split": "backend"}
    )
    resource_plan: Mapping[str, Any] = field(default_factory=dict)
    schema: str = PROGRAM_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROGRAM_MANIFEST_SCHEMA:
            raise ProgramABIError(f"unsupported program manifest schema {self.schema!r}")
        for field_name in (
            "base_fingerprint",
            "conditioner_abi",
            "latent_abi",
            "scheduler_abi",
            "vae_abi",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProgramABIError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "component_graph", _mapping(self.component_graph))
        object.__setattr__(self, "state_schema", _mapping(self.state_schema))
        object.__setattr__(self, "ports", _mapping(self.ports))
        object.__setattr__(
            self,
            "component_io",
            _mapping(validate_component_io_manifest(self.component_io)),
        )
        object.__setattr__(self, "route_schema", _mapping(self.route_schema))
        object.__setattr__(self, "numerical_contract", _mapping(self.numerical_contract))
        object.__setattr__(self, "output_contract", _mapping(self.output_contract))
        object.__setattr__(self, "resource_plan", _mapping(self.resource_plan))
        object.__setattr__(self, "extensions", _names(self.extensions, "extensions"))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical manifest payload used for identity."""

        return {
            "schema": self.schema,
            "base_fingerprint": self.base_fingerprint,
            "component_graph": dict(self.component_graph),
            "conditioner_abi": self.conditioner_abi,
            "latent_abi": self.latent_abi,
            "scheduler_abi": self.scheduler_abi,
            "vae_abi": self.vae_abi,
            "state_schema": dict(self.state_schema),
            "ports": dict(self.ports),
            "component_io": dict(self.component_io),
            "extensions": list(self.extensions),
            "route_schema": dict(self.route_schema),
            "numerical_contract": dict(self.numerical_contract),
            "output_contract": dict(self.output_contract),
            "resource_plan": dict(self.resource_plan),
        }

    @property
    def fingerprint(self) -> str:
        """SHA-256 identity of the complete manifest, independent of key order."""

        return _digest(self.to_dict())

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProgramManifest:
        if not isinstance(payload, Mapping):
            raise ProgramABIError("program manifest must be a mapping")
        known = {
            "schema",
            "base_fingerprint",
            "component_graph",
            "conditioner_abi",
            "latent_abi",
            "scheduler_abi",
            "vae_abi",
            "state_schema",
            "ports",
            "component_io",
            "extensions",
            "route_schema",
            "numerical_contract",
            "output_contract",
            "resource_plan",
        }
        unknown = set(payload) - known
        if unknown:
            raise ProgramABIError(f"unknown manifest fields: {sorted(map(str, unknown))}")
        if "base_fingerprint" not in payload:
            raise ProgramABIError("manifest is missing base_fingerprint")
        return cls(
            schema=str(payload.get("schema", PROGRAM_MANIFEST_SCHEMA)),
            base_fingerprint=str(payload["base_fingerprint"]),
            component_graph=payload.get("component_graph", {}),
            conditioner_abi=str(payload.get("conditioner_abi", "prompt-embeds-v1")),
            latent_abi=str(payload.get("latent_abi", "pipeline-latents-v1")),
            scheduler_abi=str(payload.get("scheduler_abi", "diffusers-scheduler-v1")),
            vae_abi=str(payload.get("vae_abi", "diffusers-vae-v1")),
            state_schema=payload.get("state_schema", {"name": PROGRAM_STATE_SCHEMA, "version": 1}),
            ports=payload.get("ports", {}),
            component_io=payload.get("component_io", component_io_manifest()),
            extensions=tuple(payload.get("extensions", ())),
            route_schema=payload.get("route_schema", {}),
            numerical_contract=payload.get(
                "numerical_contract", {"mode": "backend-authority", "step_granularity": "atomic"}
            ),
            output_contract=payload.get(
                "output_contract", {"kind": "pipeline-output", "row_split": "backend"}
            ),
            resource_plan=payload.get("resource_plan", {}),
        )

    @classmethod
    def from_json(cls, payload: str) -> ProgramManifest:
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProgramABIError("program manifest JSON is invalid") from exc
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class ProgramExtension:
    """mstack-style extension declaration with explicit field ownership."""

    name: str
    fingerprint: str
    base_fingerprint: str | None = None
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    numerical_contract: str = "preserve"

    def __post_init__(self) -> None:
        for field_name in ("name", "fingerprint", "numerical_contract"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProgramABIError(f"extension {field_name} must be non-empty")
        if self.base_fingerprint is not None and not str(self.base_fingerprint).strip():
            raise ProgramABIError("extension base_fingerprint cannot be empty")
        object.__setattr__(self, "reads", _names(self.reads, "extension reads"))
        object.__setattr__(self, "writes", _names(self.writes, "extension writes"))
        object.__setattr__(self, "invalidates", _names(self.invalidates, "extension invalidates"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fingerprint": self.fingerprint,
            "base_fingerprint": self.base_fingerprint,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "invalidates": list(self.invalidates),
            "numerical_contract": self.numerical_contract,
        }

    @classmethod
    def from_value(cls, value: ProgramExtension | Mapping[str, Any] | str) -> ProgramExtension:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value, fingerprint=value)
        if not isinstance(value, Mapping):
            raise ProgramABIError(f"unsupported extension declaration {type(value).__name__}")
        return cls(
            name=str(value.get("name", "")),
            fingerprint=str(value.get("fingerprint", "")),
            base_fingerprint=(
                None if value.get("base_fingerprint") is None else str(value["base_fingerprint"])
            ),
            reads=tuple(value.get("reads", ())),
            writes=tuple(value.get("writes", ())),
            invalidates=tuple(value.get("invalidates", ())),
            numerical_contract=str(value.get("numerical_contract", "preserve")),
        )


@dataclass(frozen=True, slots=True)
class ProgramState:
    """Public row-local state; tensor payloads remain owned by the session."""

    session_id: str
    link_fingerprint: str
    generation: int = 0
    status: str = "allocated"
    seed: int | None = None
    resolution: tuple[int, int] | None = None
    prompt: str | None = None
    conditioning_key: str | None = None
    schedule_fingerprint: str = ""
    step_index: int = 0
    total_steps: int | None = None
    output_available: bool = False
    last_operation: str = "allocate"
    schema: str = PROGRAM_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROGRAM_STATE_SCHEMA:
            raise ProgramStateError(f"unsupported program state schema {self.schema!r}")
        if not self.session_id or not self.link_fingerprint:
            raise ProgramStateError("program state requires session_id and link_fingerprint")
        if self.status not in _STATUSES:
            raise ProgramStateError(f"unsupported program state status {self.status!r}")
        if self.generation < 0 or self.step_index < 0:
            raise ProgramStateError("program state counters cannot be negative")
        if self.total_steps is not None and self.total_steps <= 0:
            raise ProgramStateError("total_steps must be positive when supplied")
        if self.resolution is not None:
            if len(self.resolution) != 2 or any(int(value) <= 0 for value in self.resolution):
                raise ProgramStateError("resolution must be a positive (height, width) pair")
            object.__setattr__(
                self, "resolution", (int(self.resolution[0]), int(self.resolution[1]))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "session_id": self.session_id,
            "link_fingerprint": self.link_fingerprint,
            "generation": self.generation,
            "status": self.status,
            "seed": self.seed,
            "resolution": list(self.resolution) if self.resolution is not None else None,
            "prompt": self.prompt,
            "conditioning_key": self.conditioning_key,
            "schedule_fingerprint": self.schedule_fingerprint,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "output_available": self.output_available,
            "last_operation": self.last_operation,
        }


@dataclass(frozen=True, slots=True)
class DenoiseBinding:
    """A scheduler-facing immutable binding for compatible batch admission."""

    session_id: str
    link_fingerprint: str
    compatibility_key: tuple[Any, ...]
    generation: int
    step_index: int
    total_steps: int | None
    resolution: tuple[int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "link_fingerprint": self.link_fingerprint,
            "compatibility_key": list(self.compatibility_key),
            "generation": self.generation,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "resolution": list(self.resolution) if self.resolution is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ProgramCheckpoint:
    """An in-process checkpoint suitable for exact session continuation or fork."""

    checkpoint_id: str
    state: ProgramState
    embeds: PromptEmbeds | None = field(default=None, repr=False, compare=False)
    initial_latents: Any = field(default=None, repr=False, compare=False)
    output: Any = field(default=None, repr=False, compare=False)
    backend_checkpoint: Any = field(default=None, repr=False, compare=False)
    io_frame: ComponentFrame | None = field(default=None, repr=False, compare=False)
    schema: str = PROGRAM_CHECKPOINT_SCHEMA
    context_payload: Any = field(default=None, repr=False, compare=False)
    context_key: str | None = None
    context_checksum: str | None = None
    context_is_snapshot: bool = False

    def __post_init__(self) -> None:
        if self.schema != PROGRAM_CHECKPOINT_SCHEMA:
            raise ProgramStateError(f"unsupported checkpoint schema {self.schema!r}")
        if not self.checkpoint_id:
            raise ProgramStateError("checkpoint_id must be non-empty")
        if self.state.status == "running":
            raise ProgramStateError("running state cannot be checkpointed")
        if self.embeds is not None and self.context_payload is not None:
            raise ProgramStateError(
                "checkpoint cannot contain both legacy embeds and a program context payload"
            )
        if self.context_payload is None:
            if self.context_key is not None or self.context_checksum is not None:
                raise ProgramStateError(
                    "checkpoint context identity requires a program context payload"
                )
            if self.context_is_snapshot:
                raise ProgramStateError("empty checkpoint context cannot be a snapshot")
        else:
            if not isinstance(self.context_key, str) or not self.context_key.strip():
                raise ProgramStateError("family checkpoint requires a non-empty context_key")
            if (
                not isinstance(self.context_checksum, str)
                or len(self.context_checksum) != 64
                or any(character not in "0123456789abcdef" for character in self.context_checksum)
            ):
                raise ProgramStateError(
                    "family checkpoint requires a lowercase SHA-256 context_checksum"
                )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "state": self.state.to_dict(),
            "conditioning_key": self.state.conditioning_key,
            "has_context_payload": self.context_payload is not None,
            "context_key": self.context_key,
            "context_checksum": self.context_checksum,
            "context_is_snapshot": self.context_is_snapshot,
            "has_initial_latents": self.initial_latents is not None,
            "has_output": self.output is not None,
            "has_backend_checkpoint": self.backend_checkpoint is not None,
            "io_frame": self.io_frame.to_dict() if self.io_frame is not None else None,
            "backend_checkpoint": (
                self.backend_checkpoint.metadata_only()
                if callable(getattr(self.backend_checkpoint, "metadata_only", None))
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ProgramStepResult:
    """Result of one atomic backend step and its committed state transition."""

    output: Any = field(repr=False, compare=False)
    state_before: ProgramState
    state_after: ProgramState
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    io_frame: ComponentFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "telemetry", MappingProxyType(dict(self.telemetry)))


def _clone_payload(value: Any) -> Any:
    """Clone tensors without importing torch at module scope."""

    if value is None:
        return None
    if isinstance(value, PromptEmbeds):
        return PromptEmbeds(
            key=value.key,
            tensors={key: _clone_payload(tensor) for key, tensor in value.tensors.items()},
            meta=copy.deepcopy(value.meta),
        )
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        try:
            return value.detach().clone()
        except (RuntimeError, TypeError):
            pass
    if isinstance(value, dict):
        return {key: _clone_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_payload(item) for item in value)
    try:
        return copy.deepcopy(value)
    except (TypeError, RuntimeError):
        return value


def _freeze_program_context(value: Any) -> Any:
    """Freeze Python containers without copying tensor or cache storage.

    Family backends that opt into the immutable contract retain ownership of
    opaque values (including tensors).  The runtime freezes the surrounding
    containers so a public context handle cannot mutate checkpoint/session
    structure while preserving zero-copy KV-cache sharing.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_program_context(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_program_context(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_program_context(item) for item in value)
    return value


def _is_family_backend(backend: Any) -> bool:
    return callable(getattr(backend, "step_program_context", None))


def _family_context_contract(backend: Any) -> str | None:
    """Return the declared family context ownership contract.

    ``immutable`` is the fast path: the backend returns a fresh immutable
    payload for each logical transition and the runtime never clones its
    tensor/cache storage. Mutable backends must explicitly provide either a
    clone hook or a snapshot/restore pair so rollback never aliases provisional
    mutations.
    """

    if not _is_family_backend(backend):
        return None
    immutable = getattr(backend, "program_context_is_immutable", False)
    if not isinstance(immutable, bool):
        raise ProgramCapabilityError("program_context_is_immutable must be a bool declaration")
    clone = callable(getattr(backend, "clone_program_context", None))
    snapshot = callable(getattr(backend, "snapshot_program_context", None))
    restore = callable(getattr(backend, "restore_program_context", None))
    if snapshot != restore:
        raise ProgramCapabilityError(
            "mutable family backends must expose both snapshot_program_context() "
            "and restore_program_context()"
        )
    declared = [
        name
        for name, enabled in (
            ("immutable", immutable),
            ("clone", clone),
            ("snapshot", snapshot and restore),
        )
        if enabled
    ]
    if not declared:
        raise ProgramCapabilityError(
            "family backend must declare program_context_is_immutable=True, "
            "clone_program_context(), or snapshot/restore_program_context()"
        )
    if len(declared) != 1:
        raise ProgramCapabilityError(
            "family backend must declare exactly one program context ownership contract"
        )
    return declared[0]


def _family_checkpoint_id(
    state: ProgramState,
    *,
    context_key: str,
    context_checksum: str,
    context_is_snapshot: bool,
    has_initial_latents: bool,
) -> str:
    payload = {
        "state": state.to_dict(),
        "conditioning_key": state.conditioning_key,
        "context_key": context_key,
        "context_checksum": context_checksum,
        "context_is_snapshot": context_is_snapshot,
        "has_initial_latents": has_initial_latents,
    }
    return f"ckpt-{_digest(payload)[:32]}"


@dataclass(frozen=True, slots=True)
class _ProgramContextCapture:
    payload: Any = field(repr=False, compare=False)
    key: str
    checksum: str
    is_snapshot: bool = False


def _split_program_step_result(
    value: Any,
    current_context: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Normalize the optional family step protocol.

    A family backend may return a plain output when its context is unchanged,
    or ``(output, next_context, metadata)`` when it advances row-local state.
    The three-tuple is deliberately strict so malformed transactional results
    fail before session state is committed.
    """

    if not isinstance(value, tuple):
        return value, current_context, {}
    if len(value) != 3:
        raise ProgramABIError(
            "step_program_context() tuple results must contain exactly "
            "(output, next_context, metadata)"
        )
    output, next_context, metadata = value
    if next_context is None:
        raise ProgramABIError("step_program_context() returned an empty next context")
    if not isinstance(metadata, Mapping):
        raise ProgramABIError("step_program_context() metadata must be a mapping")
    return output, next_context, dict(metadata)


def _normalise_resolution(value: Sequence[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ProgramStateError("resolution must be a (height, width) pair")
    height, width = (int(value[0]), int(value[1]))
    if height <= 0 or width <= 0:
        raise ProgramStateError("resolution must be positive")
    return height, width


class DiffusionProgram:
    """Base program plus a backend capable of context compilation and execution.

    Existing diffusion backends use ``encode`` and ``generate``. Other model
    families may instead expose ``compile_program_context`` and
    ``step_program_context``. A family step returns either a plain output (the
    context is unchanged) or ``(output, next_context, metadata)``. Family
    backends must provide a content-addressed ``program_context_key`` and an
    explicit immutable, clone, or snapshot/restore ownership contract. Optional
    ``program_context_checksum``, ``program_compatibility_key``, and
    ``program_step_terminal`` hooks own family-specific identity, admission,
    and termination without changing the session API.
    """

    def __init__(self, backend: Any, manifest: ProgramManifest) -> None:
        self._validate_backend(backend, manifest)
        self.backend = backend
        self.manifest = manifest

    @staticmethod
    def _validate_backend(backend: Any, manifest: ProgramManifest) -> None:
        legacy_compile = callable(getattr(backend, "encode", None))
        legacy_step = callable(getattr(backend, "generate", None))
        family_compile = callable(getattr(backend, "compile_program_context", None))
        family_step = callable(getattr(backend, "step_program_context", None))
        if family_compile != family_step:
            raise ProgramCapabilityError(
                "program backend must expose both compile_program_context() and "
                "step_program_context()"
            )
        if family_compile and legacy_compile and legacy_step:
            raise ProgramCapabilityError(
                "program backend exposes both complete legacy and family protocols; "
                "wrap it with one unambiguous program protocol"
            )
        if not family_compile and not (legacy_compile and legacy_step):
            raise ProgramCapabilityError(
                "program backend must expose encode()/generate() or "
                "compile_program_context()/step_program_context()"
            )
        if family_compile:
            if not callable(getattr(backend, "program_context_key", None)):
                raise ProgramCapabilityError("family backend must expose program_context_key()")
            checksum_hook = getattr(backend, "program_context_checksum", None)
            if checksum_hook is not None and not callable(checksum_hook):
                raise ProgramCapabilityError(
                    "program_context_checksum must be callable when supplied"
                )
            _family_context_contract(backend)
        expected_class = manifest.component_graph.get("pipeline_class")
        if expected_class is not None:
            pipeline = getattr(backend, "pipeline", backend)
            actual_class = type(pipeline).__name__
            if str(expected_class) != actual_class:
                raise ProgramABIError(
                    f"manifest expects pipeline_class {expected_class!r}, got {actual_class!r}"
                )

    @classmethod
    def from_backend(
        cls,
        backend: Any,
        *,
        base_fingerprint: str,
        component_graph: Mapping[str, Any] | None = None,
        **manifest_kwargs: Any,
    ) -> DiffusionProgram:
        """Construct a manifest with safe defaults for an existing backend."""

        if component_graph is None:
            pipeline = getattr(backend, "pipeline", backend)
            component_graph = {"pipeline_class": type(pipeline).__name__}
        manifest = ProgramManifest(
            base_fingerprint=base_fingerprint,
            component_graph=component_graph,
            **manifest_kwargs,
        )
        return cls(backend, manifest)

    @property
    def program_fingerprint(self) -> str:
        return self.manifest.fingerprint

    def link(
        self,
        *,
        extensions: Iterable[ProgramExtension | Mapping[str, Any] | str] = (),
        references: Iterable[str] = (),
        schedule_fingerprint: str | None = None,
        resource_policy: Mapping[str, Any] | None = None,
    ) -> ProgramLink:
        if isinstance(extensions, (str, ProgramExtension)) or isinstance(extensions, Mapping):
            extension_values = (extensions,)
        else:
            extension_values = tuple(extensions)
        parsed_extensions = tuple(ProgramExtension.from_value(value) for value in extension_values)
        parsed_references = tuple(str(value) for value in references)
        if any(not value for value in parsed_references):
            raise ProgramABIError("reference handles cannot be empty")
        return ProgramLink(
            manifest=self.manifest,
            backend=self.backend,
            extensions=parsed_extensions,
            references=parsed_references,
            schedule_fingerprint=schedule_fingerprint or self.manifest.scheduler_abi,
            resource_policy=resource_policy or {},
        )


@dataclass(frozen=True, slots=True)
class ProgramLink:
    """A linked executable program with resolved extensions and resources."""

    manifest: ProgramManifest
    backend: Any = field(repr=False, compare=False)
    extensions: tuple[ProgramExtension, ...] = ()
    references: tuple[str, ...] = ()
    schedule_fingerprint: str = ""
    resource_policy: Mapping[str, Any] = field(default_factory=dict)
    link_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.schedule_fingerprint:
            raise ProgramABIError("linked program requires schedule_fingerprint")
        names = [extension.name for extension in self.extensions]
        if len(set(names)) != len(names):
            raise ProgramABIError("linked program extensions must have unique names")
        declared_writers: dict[str, str] = {}
        for extension in self.extensions:
            if (
                extension.base_fingerprint is not None
                and extension.base_fingerprint != self.manifest.base_fingerprint
            ):
                raise ProgramABIError(
                    f"extension {extension.name!r} targets {extension.base_fingerprint!r}, "
                    f"not base {self.manifest.base_fingerprint!r}"
                )
            for field_name in extension.writes:
                previous = declared_writers.get(field_name)
                if previous is not None:
                    raise ProgramABIError(
                        f"extensions {previous!r} and {extension.name!r} both write {field_name!r}"
                    )
                declared_writers[field_name] = extension.name
        object.__setattr__(self, "extensions", tuple(self.extensions))
        object.__setattr__(self, "references", tuple(str(value) for value in self.references))
        resource_policy = dict(_mapping(self.resource_policy))
        lease_policy = _weight_page_lease_policy(resource_policy)
        if lease_policy is not None:
            lease_policy = MappingProxyType(
                {
                    **dict(lease_policy),
                    "provider_fingerprint": str(lease_policy["provider_fingerprint"]).casefold(),
                    "ordered_page_schedule": tuple(lease_policy["ordered_page_schedule"]),
                }
            )
            resource_policy["weight_page_lease"] = lease_policy
            provider = _backend_weight_page_provider(self.backend)
            actual = _weight_page_provider_content_fingerprint(provider)
            if actual != lease_policy["provider_fingerprint"]:
                raise ProgramABIError(
                    "linked weight-page provider content fingerprint does not match resource policy"
                )
            _torch_dtype(lease_policy["dtype"])
        object.__setattr__(self, "resource_policy", MappingProxyType(resource_policy))
        payload = {
            "schema": "mrun-diffusion-linked-program-v1",
            "manifest_fingerprint": self.manifest.fingerprint,
            "extensions": [extension.to_dict() for extension in self.extensions],
            "references": list(self.references),
            "schedule_fingerprint": self.schedule_fingerprint,
            "resource_policy": _canonical(self.resource_policy),
        }
        object.__setattr__(self, "link_fingerprint", _digest(payload))

    def _new_weight_page_lease(self) -> Any | None:
        policy = _weight_page_lease_policy(self.resource_policy)
        if policy is None:
            return None
        provider = _backend_weight_page_provider(self.backend)
        actual = _weight_page_provider_content_fingerprint(provider)
        if actual != policy["provider_fingerprint"]:
            raise ProgramABIError(
                "weight-page provider content fingerprint changed after program link"
            )
        lease = provider.lease(
            tuple(policy["ordered_page_schedule"]),
            device=policy["device"],
            dtype=_torch_dtype(policy["dtype"]),
            numerical_lane=policy["numerical_lane"],
            require_integrity=True,
            max_retained_device_bytes=policy["max_retained_device_bytes"],
        )
        if (
            not callable(getattr(lease, "__enter__", None))
            or not callable(getattr(lease, "__exit__", None))
            or not callable(getattr(lease, "telemetry", None))
        ):
            raise ProgramABIError("weight-page provider lease must be a telemetry context manager")
        return lease

    def _weight_page_lease_telemetry(self, lease: Any) -> dict[str, Any]:
        policy = _weight_page_lease_policy(self.resource_policy)
        if policy is None or lease is None:
            return {}
        raw = lease.telemetry()
        if not isinstance(raw, Mapping):
            raise ProgramABIError("weight-page lease telemetry must be a mapping")
        telemetry = dict(_canonical(raw))
        if telemetry.get("content_fingerprint") != policy["provider_fingerprint"]:
            raise ProgramABIError("weight-page lease telemetry content fingerprint mismatch")
        layout_fingerprint = telemetry.get("store_fingerprint")
        if (
            not isinstance(layout_fingerprint, str)
            or len(layout_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in layout_fingerprint)
        ):
            raise ProgramABIError("weight-page lease telemetry layout identity is invalid")
        if telemetry.get("ordered_keys") != list(policy["ordered_page_schedule"]):
            raise ProgramABIError("weight-page lease telemetry schedule mismatch")
        if telemetry.get("numerical_lane") != policy["numerical_lane"]:
            raise ProgramABIError("weight-page lease telemetry numerical lane mismatch")
        expected_device = str(policy["device"])
        if telemetry.get("device") != expected_device:
            raise ProgramABIError("weight-page lease telemetry device mismatch")
        expected_dtype = str(_torch_dtype(policy["dtype"]))
        if telemetry.get("dtype") != expected_dtype:
            raise ProgramABIError("weight-page lease telemetry dtype mismatch")
        if telemetry.get("state") != "released":
            raise ProgramABIError("weight-page lease telemetry is not released")
        if telemetry.get("consumed_demands") != telemetry.get("declared_demands"):
            raise ProgramABIError("weight-page lease did not consume its exact schedule")
        budget = policy["max_retained_device_bytes"]
        if telemetry.get("max_retained_device_bytes") != budget:
            raise ProgramABIError("weight-page lease retention budget mismatch")
        if telemetry.get("retention_authority") != "bounded-belady-next-use":
            raise ProgramABIError("weight-page lease retention is non-authoritative")
        evictions = telemetry.get("retention_evictions")
        if isinstance(evictions, bool) or not isinstance(evictions, int) or evictions < 0:
            raise ProgramABIError("weight-page lease eviction telemetry is invalid")
        eviction_events = telemetry.get("retention_eviction_events")
        if not isinstance(eviction_events, list) or len(eviction_events) != evictions:
            raise ProgramABIError("weight-page lease eviction event ledger is invalid")
        for event in eviction_events:
            if not isinstance(event, Mapping):
                raise ProgramABIError("weight-page lease eviction event ledger is invalid")
            demand_index = event.get("demand_index")
            evicted_key = event.get("evicted_key")
            evicted_bytes = event.get("evicted_device_bytes")
            next_use = event.get("next_use_index")
            if (
                isinstance(demand_index, bool)
                or not isinstance(demand_index, int)
                or demand_index < 0
                or not isinstance(evicted_key, str)
                or not evicted_key
                or isinstance(evicted_bytes, bool)
                or not isinstance(evicted_bytes, int)
                or evicted_bytes <= 0
                or (
                    next_use is not None
                    and (
                        isinstance(next_use, bool)
                        or not isinstance(next_use, int)
                        or next_use <= demand_index
                    )
                )
            ):
                raise ProgramABIError("weight-page lease eviction event ledger is invalid")
        if telemetry.get("integrity_capable") is not True:
            raise ProgramABIError("weight-page lease telemetry is not integrity-capable")
        if telemetry.get("integrity_required") is not True:
            raise ProgramABIError("weight-page lease telemetry is non-authoritative")
        if telemetry.get("integrity_authority") != "sha256-page-verification-required":
            raise ProgramABIError("weight-page lease telemetry has no integrity authority")
        if telemetry.get("store_integrity_status") not in {
            "partially-verified",
            "fully-verified",
        }:
            raise ProgramABIError("weight-page lease telemetry integrity status is not verified")
        attempts = telemetry.get("integrity_verification_attempts")
        failures = telemetry.get("integrity_verification_failures")
        verified_pages = telemetry.get("integrity_verified_pages")
        verified_weight_bytes = telemetry.get("integrity_verified_weight_bytes")
        verified_scale_bytes = telemetry.get("integrity_verified_scale_bytes")
        verified_bytes = telemetry.get("integrity_verified_bytes")
        integers = (
            attempts,
            failures,
            verified_pages,
            verified_weight_bytes,
            verified_scale_bytes,
            verified_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ProgramABIError("weight-page lease integrity counters are invalid")
        if (
            failures != 0
            or verified_pages <= 0
            or attempts < verified_pages
            or verified_weight_bytes <= 0
            or verified_scale_bytes <= 0
            or verified_bytes != verified_weight_bytes + verified_scale_bytes
        ):
            raise ProgramABIError("weight-page lease integrity verification is incomplete")
        scope = telemetry.get("residency_measurement_scope")
        if not isinstance(scope, str) or not scope:
            raise ProgramABIError("weight-page lease telemetry lacks a measurement scope")
        current = telemetry.get("measured_retained_device_bytes_current")
        peak = telemetry.get("measured_retained_device_bytes_peak")
        if current != 0 or isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            raise ProgramABIError("weight-page lease residency telemetry is invalid")
        if peak > budget:
            raise ProgramABIError("weight-page lease exceeded its retention budget")
        return telemetry

    @property
    def compatibility_key(self) -> tuple[str, str, tuple[str, ...]]:
        """Stable key for work-template admission before row-local bindings."""

        return (
            self.link_fingerprint,
            self.schedule_fingerprint,
            tuple(self.references),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_fingerprint": self.manifest.fingerprint,
            "extensions": [extension.to_dict() for extension in self.extensions],
            "references": list(self.references),
            "schedule_fingerprint": self.schedule_fingerprint,
            "resource_policy": _canonical(self.resource_policy),
            "link_fingerprint": self.link_fingerprint,
        }

    def open_session(
        self,
        *,
        session_id: str | None = None,
        seed: int | None = None,
        resolution: Sequence[int] | None = None,
        total_steps: int | None = None,
        initial_latents: Any = None,
    ) -> ProgramSession:
        return ProgramSession(
            self,
            session_id=session_id,
            seed=seed,
            resolution=resolution,
            total_steps=total_steps,
            initial_latents=initial_latents,
        )

    def restore(
        self,
        checkpoint: ProgramCheckpoint,
        *,
        session_id: str | None = None,
    ) -> ProgramSession:
        if checkpoint.state.link_fingerprint != self.link_fingerprint:
            raise ProgramABIError("checkpoint belongs to a different linked program")
        session = self.open_session(
            session_id=session_id or checkpoint.state.session_id,
            seed=checkpoint.state.seed,
            resolution=checkpoint.state.resolution,
            total_steps=checkpoint.state.total_steps,
        )
        session._restore_checkpoint(checkpoint, allow_session_id=True)
        return session

    def step_batch(
        self,
        sessions: Sequence[ProgramSession],
        **kwargs: Any,
    ) -> tuple[ProgramStepResult, ...]:
        """Execute compatible sessions in one physical backend batch.

        The backend must expose ``generate_batch`` and return an object with a
        ``rows_by_branch()`` method, as ``PhaseBatchResult`` does.  There is no
        scalar fallback: a caller requesting a batch receives one physical
        batch or an explicit capability error.
        """

        if not sessions:
            raise ProgramStateError("step_batch requires at least one session")
        if _is_family_backend(self.backend):
            raise ProgramCapabilityError(
                "step_batch does not support family program contexts; "
                "use scalar step() until a family batch protocol is declared"
            )
        if not callable(getattr(self.backend, "generate_batch", None)):
            raise ProgramCapabilityError("linked backend does not expose generate_batch()")
        rows = tuple(sessions)
        if len({session.session_id for session in rows}) != len(rows):
            raise ProgramStateError("step_batch sessions must have unique session IDs")
        if any(session.link is not self for session in rows):
            raise ProgramABIError("step_batch sessions must belong to this exact linked program")

        locks = [session._lock for session in sorted(rows, key=lambda item: item.session_id)]
        for lock in locks:
            lock.acquire()
        source_states: tuple[ProgramState, ...] = ()
        source_contexts: tuple[Any, ...] = ()
        source_outputs: tuple[Any, ...] = ()
        source_backend_checkpoints: tuple[Any, ...] = ()
        source_frames: tuple[ComponentFrame | None, ...] = ()
        try:
            bindings = tuple(session.binding() for session in rows)
            first_binding = bindings[0]
            if any(
                binding.compatibility_key != first_binding.compatibility_key
                for binding in bindings[1:]
            ):
                raise ProgramABIError("sessions do not share a compatible denoise work template")
            source_contexts = tuple(_clone_payload(session._context_payload) for session in rows)
            source_outputs = tuple(session._output for session in rows)
            source_backend_checkpoints = tuple(session._backend_checkpoint for session in rows)
            source_frames = tuple(session._last_io_frame for session in rows)
            source_states = tuple(session._claim_batch_step() for session in rows)

            call_kwargs = dict(kwargs)
            if "latents" in call_kwargs:
                raise ProgramStateError("step_batch owns latents through session state")
            if "num_inference_steps" not in call_kwargs and first_binding.total_steps is not None:
                call_kwargs["num_inference_steps"] = first_binding.total_steps
            if "height" not in call_kwargs and first_binding.resolution is not None:
                call_kwargs["height"] = first_binding.resolution[0]
            if "width" not in call_kwargs and first_binding.resolution is not None:
                call_kwargs["width"] = first_binding.resolution[1]
            if ("generator" not in call_kwargs or call_kwargs["generator"] is None) and all(
                session.state.seed is not None for session in rows
            ):
                call_kwargs["generator"] = [
                    session._make_generator(session.state.seed) for session in rows
                ]

            initial_latents = _stack_latents(tuple(session._initial_latents for session in rows))
            embeds = tuple(session._embeds for session in rows)
            if any(embed is None for embed in embeds):
                raise ProgramStateError("all batch sessions must have compiled context")
            if initial_latents is not None:
                call_kwargs["initial_latents"] = initial_latents
            weight_lease = self._new_weight_page_lease()
            if weight_lease is None:
                result = self.backend.generate_batch(
                    embeds,
                    branch_ids=tuple(session.session_id for session in rows),
                    **call_kwargs,
                )
                weight_lease_telemetry = {}
            else:
                with weight_lease:
                    result = self.backend.generate_batch(
                        embeds,
                        branch_ids=tuple(session.session_id for session in rows),
                        **call_kwargs,
                    )
                weight_lease_telemetry = self._weight_page_lease_telemetry(weight_lease)
            row_outputs = _rows_by_branch(result, tuple(session.session_id for session in rows))
            telemetry = dict(getattr(result, "telemetry", {}))
            telemetry.update(
                {
                    "program_link_fingerprint": self.link_fingerprint,
                    "physical_program_calls": 1,
                    "batch_size": len(rows),
                    "step_granularity": "atomic_backend_schedule",
                }
            )
            if weight_lease_telemetry:
                telemetry["weight_page_lease"] = weight_lease_telemetry

            for session, source in zip(rows, source_states, strict=True):
                if (
                    session.state.generation != source.generation
                    or session.state.status != "running"
                ):
                    for rollback, rollback_state in zip(rows, source_states, strict=True):
                        rollback._state = rollback_state
                    raise ProgramStateError("batch state changed before transactional commit")

            results: list[ProgramStepResult] = []
            for session, source in zip(rows, source_states, strict=True):
                after = session._commit_step(source, row_outputs[session.session_id])
                output = row_outputs[session.session_id]
                frame = session._make_io_frame(
                    component_id="program",
                    operation="step_batch",
                    trajectory_position={
                        "execution": "atomic_backend_batch",
                        "batch_size": len(rows),
                    },
                    bindings=(
                        PortBinding(
                            port="context_handle",
                            stream="data",
                            direction="in",
                            handle=session._embeds.key if session._embeds is not None else None,
                            producer="conditioner",
                            consumer="program",
                            branch_id=session.session_id,
                            retention="checkpoint",
                        ),
                        PortBinding(
                            port="schedule",
                            stream="control",
                            direction="in",
                            handle=session._state.schedule_fingerprint,
                            producer="program",
                            consumer="program",
                            branch_id=session.session_id,
                        ),
                        PortBinding(
                            port="route",
                            stream="route",
                            direction="in",
                            handle=f"row:{session.session_id}",
                            producer="scheduler",
                            consumer="program",
                            branch_id=session.session_id,
                        ),
                        PortBinding(
                            port="trace",
                            stream="evidence",
                            direction="in",
                            handle=f"trace:{session.session_id}:step-batch",
                            producer="program",
                            consumer="debugger",
                            branch_id=session.session_id,
                        ),
                        PortBinding(
                            port="output",
                            stream="data",
                            direction="out",
                            handle=f"output:{session.session_id}:{after.generation}",
                            payload_fingerprint=payload_fingerprint(output),
                            producer="program",
                            consumer="renderer",
                            branch_id=session.session_id,
                        ),
                        PortBinding(
                            port="trace",
                            stream="evidence",
                            direction="out",
                            handle=f"trace:{session.session_id}:step-batch",
                            producer="program",
                            consumer="debugger",
                            branch_id=session.session_id,
                        ),
                    ),
                )
                row_telemetry = dict(telemetry)
                row_telemetry["branch_id"] = session.session_id
                results.append(
                    ProgramStepResult(
                        output=output,
                        state_before=source,
                        state_after=after,
                        telemetry=row_telemetry,
                        io_frame=frame,
                    )
                )
            return tuple(results)
        except Exception:
            for index, (session, source) in enumerate(zip(rows, source_states, strict=False)):
                session._state = source
                session._context_payload = source_contexts[index]
                session._output = source_outputs[index]
                session._backend_checkpoint = source_backend_checkpoints[index]
                session._last_io_frame = source_frames[index]
            raise
        finally:
            for lock in reversed(locks):
                lock.release()

    def replay_batch(
        self,
        checkpoint: ProgramCheckpoint,
        *,
        branch_ids: Sequence[str],
        latent_overrides: Any | Sequence[Any] | None = None,
        mode: str = "exact",
        output_type: str = "pil",
    ) -> Any:
        """Revert/fork a trajectory checkpoint and execute branch suffixes.

        This is separate from :meth:`step_batch` because it operates on one
        paused program state, not on independent complete generations.  The
        backend owns the physical suffix implementation; the program layer
        owns checkpoint identity, branch ordering, and the non-mutating fork
        contract.
        """

        if checkpoint.state.link_fingerprint != self.link_fingerprint:
            raise ProgramABIError("checkpoint belongs to a different linked program")
        backend_checkpoint = checkpoint.backend_checkpoint
        if not isinstance(backend_checkpoint, TrajectoryCheckpoint):
            raise ProgramCapabilityError(
                "replay_batch requires a trajectory checkpoint from session.pause()"
            )
        replay = getattr(self.backend, "resume_checkpoint_batch", None)
        if not callable(replay):
            raise ProgramCapabilityError("linked backend does not expose resume_checkpoint_batch()")
        result = replay(
            backend_checkpoint,
            branch_ids=tuple(str(value) for value in branch_ids),
            latent_overrides=latent_overrides,
            mode=mode,
            output_type=output_type,
        )
        telemetry = getattr(result, "telemetry", None)
        if isinstance(telemetry, Mapping):
            # PhaseBatchResult is frozen, so add a shallow metadata wrapper
            # only when the backend exposes its private result constructor.
            try:
                from .phase import PhaseBatchResult

                if isinstance(result, PhaseBatchResult):
                    result = replace(
                        result,
                        telemetry={
                            **dict(telemetry),
                            "program_link_fingerprint": self.link_fingerprint,
                            "program_replay": True,
                        },
                    )
            except ImportError:  # pragma: no cover - phase is a local module
                pass
        return result


class ProgramSession:
    """One row-local execution state bound to a :class:`ProgramLink`."""

    def __init__(
        self,
        link: ProgramLink,
        *,
        session_id: str | None,
        seed: int | None,
        resolution: Sequence[int] | None,
        total_steps: int | None,
        initial_latents: Any,
    ) -> None:
        if seed is not None and isinstance(seed, bool):
            raise ProgramStateError("seed must be an integer")
        if total_steps is not None and (isinstance(total_steps, bool) or int(total_steps) <= 0):
            raise ProgramStateError("total_steps must be a positive integer")
        self._link = link
        self._lock = RLock()
        self._context_payload: Any = None
        self._context_checksum: str | None = None
        self._initial_latents = initial_latents
        self._output: Any = None
        self._backend_checkpoint: TrajectoryCheckpoint | None = None
        self._last_io_frame: ComponentFrame | None = None
        self._state = ProgramState(
            session_id=session_id or uuid4().hex,
            link_fingerprint=link.link_fingerprint,
            seed=None if seed is None else int(seed),
            resolution=_normalise_resolution(resolution),
            total_steps=None if total_steps is None else int(total_steps),
            schedule_fingerprint=link.schedule_fingerprint,
        )

    @property
    def link(self) -> ProgramLink:
        return self._link

    @property
    def session_id(self) -> str:
        return self._state.session_id

    @property
    def state(self) -> ProgramState:
        return self._state

    @property
    def output(self) -> Any:
        if not self._state.output_available:
            raise ProgramStateError("session has no committed output")
        return self._output

    @property
    def context_payload(self) -> Any:
        """Return the family-owned context associated with this session.

        Immutable-family containers are read-only and can be shared without
        cloning tensor/cache storage. Mutable-family backends return an
        explicit backend clone or restored snapshot, never the live session
        payload. The accessor deliberately has no setter: lifecycle
        transitions own replacement of the payload.
        """

        if self._context_payload is None:
            raise ProgramStateError("session has no compiled program context")
        if not _is_family_backend(self._link.backend):
            return self._context_payload
        identity = self._validate_current_context()
        capture = self._capture_program_context(known_identity=identity)
        return self._restore_program_context_capture(capture)

    @property
    def _embeds(self) -> Any:
        """Compatibility alias for legacy diffusion internals and callers."""

        return self._context_payload

    @_embeds.setter
    def _embeds(self, value: Any) -> None:
        self._context_payload = value

    @property
    def last_io_frame(self) -> ComponentFrame | None:
        """Return the latest sideband frame retained for debugger inspection."""

        return self._last_io_frame

    def binding(self) -> DenoiseBinding:
        if self._state.status not in {"context_compiled", "running"}:
            raise ProgramStateError(
                f"session {self.session_id!r} is not ready for denoise binding "
                f"(status={self._state.status!r})"
            )
        self._validate_current_context()
        compatibility = self._program_compatibility_key()
        return DenoiseBinding(
            session_id=self.session_id,
            link_fingerprint=self._link.link_fingerprint,
            compatibility_key=(*self._link.compatibility_key, *compatibility),
            generation=self._state.generation,
            step_index=self._state.step_index,
            total_steps=self._state.total_steps,
            resolution=self._state.resolution,
        )

    def compile_context(self, prompt: str, **params: Any) -> Any:
        with self._lock:
            self._require_open()
            if (
                self._state.status == "completed"
                or self._state.generation > 0
                or self._state.step_index > 0
                or self._state.output_available
            ):
                raise ProgramStateError(
                    "sessions with a committed step must be forked before recompiling context"
                )
            if not isinstance(prompt, str):
                raise ProgramStateError("compile_context requires one prompt string")
            if "prompt" in params:
                raise ProgramStateError("prompt is a positional program input")
            compile_program_context = getattr(self._link.backend, "compile_program_context", None)
            if callable(compile_program_context):
                context_payload = compile_program_context(prompt, **params)
                if context_payload is None:
                    raise ProgramCapabilityError(
                        "compile_program_context() must return a context payload"
                    )
                if _family_context_contract(self._link.backend) == "immutable":
                    context_payload = _freeze_program_context(context_payload)
            else:
                context_payload = self._link.backend.encode(prompt, **params)
                if not isinstance(context_payload, PromptEmbeds):
                    raise ProgramCapabilityError("backend encode() must return PromptEmbeds")
            if _is_family_backend(self._link.backend):
                context_key, context_checksum = self._program_context_identity(context_payload)
            else:
                context_key = self._program_context_key(context_payload)
                context_checksum = None
            self._context_payload = context_payload
            self._context_checksum = context_checksum
            self._output = None
            self._backend_checkpoint = None
            self._state = replace(
                self._state,
                status="context_compiled",
                prompt=prompt,
                conditioning_key=context_key,
                output_available=False,
                last_operation="compile_context",
            )
            if not isinstance(context_payload, PromptEmbeds):
                return self.context_payload
            embeds = context_payload
            prompt_tensor = embeds.tensors.get("prompt_embeds")
            text_ids = embeds.tensors.get("text_ids")
            self._make_io_frame(
                component_id="conditioner",
                operation="encode",
                trajectory_position={"generation": self._state.generation},
                bindings=(
                    PortBinding(
                        port="prompt",
                        stream="data",
                        direction="in",
                        handle=f"prompt:{_digest(prompt)[:24]}",
                        payload_fingerprint=payload_fingerprint(prompt),
                        producer="client",
                        consumer="conditioner",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="tokenizer_config",
                        stream="control",
                        direction="in",
                        abi={"params": params},
                        producer="program",
                        consumer="conditioner",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="trace",
                        stream="evidence",
                        direction="in",
                        handle=f"trace:{self.session_id}:conditioner",
                        producer="program",
                        consumer="conditioner",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="prompt_embeds",
                        stream="data",
                        direction="out",
                        handle=embeds.key,
                        payload_fingerprint=payload_fingerprint(
                            prompt_tensor if prompt_tensor is not None else embeds
                        ),
                        producer="conditioner",
                        consumer="denoiser",
                        branch_id=self.session_id,
                        checkpoint_id=embeds.key,
                        retention="checkpoint",
                    ),
                    PortBinding(
                        port="text_ids",
                        stream="data",
                        direction="out",
                        handle=f"text-ids:{embeds.key}",
                        payload_fingerprint=(
                            payload_fingerprint(text_ids) if text_ids is not None else None
                        ),
                        producer="conditioner",
                        consumer="denoiser",
                        branch_id=self.session_id,
                        retention="checkpoint",
                    ),
                    PortBinding(
                        port="cache_handle",
                        stream="state",
                        direction="out",
                        handle=embeds.key,
                        producer="conditioner",
                        consumer="program",
                        branch_id=self.session_id,
                        retention="checkpoint",
                    ),
                    PortBinding(
                        port="trace",
                        stream="evidence",
                        direction="out",
                        handle=f"trace:{self.session_id}:conditioner",
                        producer="conditioner",
                        consumer="debugger",
                        branch_id=self.session_id,
                    ),
                ),
            )
            return embeds

    def checkpoint(self) -> ProgramCheckpoint:
        with self._lock:
            self._require_open()
            state = self._state
            current_identity = self._validate_current_context()
            legacy_embeds = (
                self._context_payload if isinstance(self._context_payload, PromptEmbeds) else None
            )
            family_capture = (
                self._capture_program_context(
                    verify_snapshot=True,
                    known_identity=current_identity,
                )
                if _is_family_backend(self._link.backend) and self._context_payload is not None
                else None
            )
            if family_capture is not None:
                checkpoint_id = _family_checkpoint_id(
                    state,
                    context_key=family_capture.key,
                    context_checksum=family_capture.checksum,
                    context_is_snapshot=family_capture.is_snapshot,
                    has_initial_latents=self._initial_latents is not None,
                )
            else:
                payload = {
                    "state": state.to_dict(),
                    "conditioning_key": state.conditioning_key,
                    "has_initial_latents": self._initial_latents is not None,
                }
                checkpoint_id = f"ckpt-{_digest(payload)[:32]}"
            return ProgramCheckpoint(
                checkpoint_id=checkpoint_id,
                state=state,
                embeds=_clone_payload(legacy_embeds),
                initial_latents=_clone_payload(self._initial_latents),
                output=_clone_payload(self._output),
                backend_checkpoint=self._backend_checkpoint,
                io_frame=self._last_io_frame,
                context_payload=(family_capture.payload if family_capture is not None else None),
                context_key=(family_capture.key if family_capture is not None else None),
                context_checksum=(family_capture.checksum if family_capture is not None else None),
                context_is_snapshot=(
                    family_capture.is_snapshot if family_capture is not None else False
                ),
            )

    def pause(self, *, cut_step: int, **kwargs: Any) -> ProgramCheckpoint:
        """Run the common trajectory prefix once and return a replay checkpoint.

        A normal ``checkpoint()`` records the program's current atomic state.
        ``pause()`` asks a trajectory-capable backend to materialize the
        denoising boundary as well, which is what makes later branch replay
        cheaper than independent full generations.
        """

        with self._lock:
            self._require_open()
            if self._state.status != "context_compiled" or self._embeds is None:
                raise ProgramStateError(
                    "pause requires a context-compiled session; fork a completed session first"
                )
            capture = getattr(self._link.backend, "capture_checkpoint", None)
            if not callable(capture):
                raise ProgramCapabilityError("linked backend does not expose capture_checkpoint()")
            backend_checkpoint = capture(
                self._embeds,
                cut_step=int(cut_step),
                **kwargs,
            )
            if not isinstance(backend_checkpoint, TrajectoryCheckpoint):
                raise ProgramCapabilityError(
                    "capture_checkpoint() did not return a TrajectoryCheckpoint"
                )
            state = replace(
                self._state,
                step_index=backend_checkpoint.step_index,
                total_steps=backend_checkpoint.total_steps,
                output_available=False,
                last_operation="pause:trajectory_checkpoint",
            )
            self._state = state
            self._backend_checkpoint = backend_checkpoint
            payload = {
                "state": state.to_dict(),
                "backend_checkpoint": backend_checkpoint.fingerprint,
            }
            checkpoint_id = f"program-{_digest(payload)[:32]}"
            frame = self._make_io_frame(
                component_id="program",
                operation="pause",
                checkpoint_id=checkpoint_id,
                trajectory_position={
                    "cut_step": int(cut_step),
                    "checkpoint_fingerprint": backend_checkpoint.fingerprint,
                },
                bindings=(
                    PortBinding(
                        port="context_handle",
                        stream="data",
                        direction="in",
                        handle=self._embeds.key,
                        producer="conditioner",
                        consumer="program",
                        branch_id=self.session_id,
                        retention="checkpoint",
                    ),
                    PortBinding(
                        port="schedule",
                        stream="control",
                        direction="in",
                        handle=self._state.schedule_fingerprint,
                        producer="program",
                        consumer="program",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="trace",
                        stream="evidence",
                        direction="in",
                        handle=f"trace:{self.session_id}:pause",
                        producer="program",
                        consumer="debugger",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="checkpoint",
                        stream="state",
                        direction="out",
                        handle=checkpoint_id,
                        payload_fingerprint=backend_checkpoint.fingerprint,
                        producer="program",
                        consumer="debugger",
                        branch_id=self.session_id,
                        checkpoint_id=checkpoint_id,
                        retention="raw",
                    ),
                    PortBinding(
                        port="trace",
                        stream="evidence",
                        direction="out",
                        handle=f"trace:{self.session_id}:pause",
                        producer="program",
                        consumer="debugger",
                        branch_id=self.session_id,
                    ),
                ),
            )
            return ProgramCheckpoint(
                checkpoint_id=checkpoint_id,
                state=state,
                embeds=_clone_payload(self._embeds),
                initial_latents=_clone_payload(self._initial_latents),
                backend_checkpoint=backend_checkpoint,
                io_frame=frame,
            )

    def restore(self, checkpoint: ProgramCheckpoint) -> None:
        with self._lock:
            self._restore_checkpoint(checkpoint, allow_session_id=False)

    def resume(
        self,
        checkpoint: ProgramCheckpoint | None = None,
        *,
        latent_override: Any | None = None,
        output_type: str = "pil",
    ) -> ProgramStepResult:
        """Revert to a trajectory checkpoint and commit one resumed branch."""

        with self._lock:
            self._require_open()
            checkpoint = checkpoint or self.checkpoint()
            if checkpoint.state.session_id != self.session_id:
                raise ProgramStateError("in-place resume requires the same session_id")
            if not isinstance(checkpoint.backend_checkpoint, TrajectoryCheckpoint):
                raise ProgramCapabilityError("resume requires a trajectory checkpoint from pause()")
            self._restore_checkpoint(checkpoint, allow_session_id=False)
            source = self._claim_step()
            try:
                resume = getattr(self._link.backend, "resume_checkpoint", None)
                if not callable(resume):
                    raise ProgramCapabilityError(
                        "linked backend does not expose resume_checkpoint()"
                    )
                output = resume(
                    checkpoint.backend_checkpoint,
                    latent_override=latent_override,
                    output_type=output_type,
                )
                after = self._commit_step(source, output)
                frame_bindings = [
                    PortBinding(
                        port="context_handle",
                        stream="data",
                        direction="in",
                        handle=self._embeds.key if self._embeds is not None else None,
                        producer="conditioner",
                        consumer="program",
                        branch_id=self.session_id,
                        retention="checkpoint",
                    ),
                    PortBinding(
                        port="schedule",
                        stream="control",
                        direction="in",
                        handle=self._state.schedule_fingerprint,
                        producer="program",
                        consumer="program",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="checkpoint",
                        stream="state",
                        direction="in",
                        handle=checkpoint.checkpoint_id,
                        producer="debugger",
                        consumer="program",
                        branch_id=self.session_id,
                        checkpoint_id=checkpoint.checkpoint_id,
                        retention="raw",
                    ),
                    PortBinding(
                        port="trace",
                        stream="evidence",
                        direction="in",
                        handle=f"trace:{self.session_id}:resume",
                        producer="debugger",
                        consumer="program",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="output",
                        stream="data",
                        direction="out",
                        handle=f"output:{self.session_id}:{after.generation}",
                        payload_fingerprint=payload_fingerprint(output),
                        producer="program",
                        consumer="renderer",
                        branch_id=self.session_id,
                    ),
                    PortBinding(
                        port="trace",
                        stream="evidence",
                        direction="out",
                        handle=f"trace:{self.session_id}:resume",
                        producer="program",
                        consumer="debugger",
                        branch_id=self.session_id,
                    ),
                ]
                frame = self._make_io_frame(
                    component_id="program",
                    operation="resume",
                    checkpoint_id=checkpoint.checkpoint_id,
                    trajectory_position={"resume_from_step": checkpoint.state.step_index},
                    bindings=frame_bindings,
                )
                return ProgramStepResult(
                    output=output,
                    state_before=source,
                    state_after=after,
                    telemetry={
                        "program_link_fingerprint": self._link.link_fingerprint,
                        "program_replay": True,
                        "physical_program_calls": 0,
                        "execution_mode": "exact_scalar_suffix",
                        "checkpoint_id": checkpoint.checkpoint_id,
                    },
                    io_frame=frame,
                )
            except Exception:
                if self._state.status == "running" and self._state.generation == source.generation:
                    self._state = source
                raise

    def fork(self, *, checkpoint: ProgramCheckpoint | None = None) -> ProgramSession:
        checkpoint = checkpoint or self.checkpoint()
        return self._link.restore(checkpoint, session_id=uuid4().hex)

    def step(self, **kwargs: Any) -> ProgramStepResult:
        with self._lock:
            self._require_open()
            family_step = _is_family_backend(self._link.backend)
            if family_step and _weight_page_lease_policy(self._link.resource_policy) is not None:
                raise ProgramCapabilityError(
                    "weight_page_lease policy supports atomic backend generate calls only"
                )
            if family_step:
                current_identity = self._validate_current_context()
                context_capture = self._capture_program_context(known_identity=current_identity)
                context_before = None
            else:
                context_capture = None
                context_before = _clone_payload(self._context_payload)
            output_before = self._output
            backend_checkpoint_before = self._backend_checkpoint
            frame_before = self._last_io_frame
            source = self._claim_step()
            try:
                if self._context_payload is None:
                    raise ProgramStateError("compile_context must run before step")
                step_program_context = getattr(self._link.backend, "step_program_context", None)
                if callable(step_program_context):
                    raw_result = step_program_context(self._context_payload, **kwargs)
                    output, next_context, step_metadata = _split_program_step_result(
                        raw_result,
                        self._context_payload,
                    )
                    if context_capture is None:  # pragma: no cover - guarded above
                        raise ProgramStateError("missing family context rollback capture")
                    current_key, current_checksum = self._program_context_identity(
                        self._context_payload
                    )
                    structured_result = isinstance(raw_result, tuple)
                    contract = _family_context_contract(self._link.backend)
                    if contract == "immutable" or not structured_result:
                        if (
                            current_key != context_capture.key
                            or current_checksum != context_capture.checksum
                        ):
                            raise ProgramABIError(
                                "step_program_context() mutated its source context; "
                                "return a new context or declare a mutable rollback contract"
                            )
                    if contract == "immutable" and next_context is not self._context_payload:
                        next_context = _freeze_program_context(next_context)
                    next_identity = self._program_context_identity(next_context)
                    execution = "program_context_step"
                else:
                    call_kwargs = self._prepare_call_kwargs(kwargs)
                    weight_lease = self._link._new_weight_page_lease()
                    if weight_lease is None:
                        output = self._link.backend.generate(
                            self._context_payload,
                            **call_kwargs,
                        )
                        weight_lease_telemetry = {}
                    else:
                        with weight_lease:
                            output = self._link.backend.generate(
                                self._context_payload,
                                **call_kwargs,
                            )
                        weight_lease_telemetry = self._link._weight_page_lease_telemetry(
                            weight_lease
                        )
                    next_context = self._context_payload
                    next_identity = None
                    step_metadata = {}
                    execution = "atomic_backend_schedule"
                after = self._commit_step(
                    source,
                    output,
                    next_context=next_context,
                    metadata=step_metadata,
                    next_identity=next_identity,
                )
                frame = self._make_io_frame(
                    component_id="program",
                    operation="step",
                    trajectory_position={"execution": execution},
                    bindings=(
                        PortBinding(
                            port="context_handle",
                            stream="data",
                            direction="in",
                            handle=source.conditioning_key,
                            producer="conditioner",
                            consumer="program",
                            branch_id=self.session_id,
                            retention="checkpoint",
                        ),
                        PortBinding(
                            port="schedule",
                            stream="control",
                            direction="in",
                            handle=self._state.schedule_fingerprint,
                            producer="program",
                            consumer="program",
                            branch_id=self.session_id,
                        ),
                        PortBinding(
                            port="trace",
                            stream="evidence",
                            direction="in",
                            handle=f"trace:{self.session_id}:step",
                            producer="program",
                            consumer="debugger",
                            branch_id=self.session_id,
                        ),
                        PortBinding(
                            port="output",
                            stream="data",
                            direction="out",
                            handle=f"output:{self.session_id}:{after.generation}",
                            payload_fingerprint=payload_fingerprint(output),
                            producer="program",
                            consumer="renderer",
                            branch_id=self.session_id,
                        ),
                        PortBinding(
                            port="trace",
                            stream="evidence",
                            direction="out",
                            handle=f"trace:{self.session_id}:step",
                            producer="program",
                            consumer="debugger",
                            branch_id=self.session_id,
                        ),
                    ),
                )
                telemetry = dict(step_metadata)
                telemetry.update(
                    {
                        "program_link_fingerprint": self._link.link_fingerprint,
                        "physical_program_calls": 1,
                        "step_granularity": execution,
                        "batch_size": 1,
                    }
                )
                if not family_step and weight_lease_telemetry:
                    telemetry["weight_page_lease"] = weight_lease_telemetry
                return ProgramStepResult(
                    output=output,
                    state_before=source,
                    state_after=after,
                    telemetry=telemetry,
                    io_frame=frame,
                )
            except Exception:
                if context_capture is not None:
                    try:
                        restored_context = self._restore_program_context_capture(context_capture)
                    except Exception as rollback_error:
                        self._poison_family_context(source)
                        raise ProgramABIError(
                            "family context rollback failed; session was closed"
                        ) from rollback_error
                    self._context_payload = restored_context
                    self._context_checksum = context_capture.checksum
                else:
                    self._context_payload = context_before
                    self._context_checksum = None
                self._state = source
                self._output = output_before
                self._backend_checkpoint = backend_checkpoint_before
                self._last_io_frame = frame_before
                raise

    def render(self) -> Any:
        """Return the backend output after a committed execution."""

        return self.output

    def close(self) -> None:
        with self._lock:
            if self._state.status == "closed":
                return
            if self._state.status == "running":
                raise ProgramStateError("cannot close a session during execution")
            self._context_payload = None
            self._context_checksum = None
            self._initial_latents = None
            self._output = None
            self._backend_checkpoint = None
            self._last_io_frame = None
            self._state = replace(self._state, status="closed", last_operation="close")

    def __enter__(self) -> ProgramSession:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _make_io_frame(
        self,
        *,
        component_id: str,
        operation: str,
        bindings: Sequence[PortBinding],
        trajectory_position: Mapping[str, Any] | None = None,
        checkpoint_id: str | None = None,
        parent_frame_id: str | None = None,
        placement: Mapping[str, Any] | None = None,
        frame_id: str | None = None,
    ) -> ComponentFrame:
        """Create, validate, and retain one debugger-visible component frame."""

        position = {
            "generation": self._state.generation,
            "step_index": self._state.step_index,
            "total_steps": self._state.total_steps,
        }
        if trajectory_position:
            position.update(dict(trajectory_position))
        resolved_placement = dict(placement or {})
        device = getattr(self._link.backend, "_device", None)
        if device is not None:
            resolved_placement.setdefault("device", str(device))
        resolved_placement.setdefault("backend", type(self._link.backend).__name__)
        frame = make_component_frame(
            component_id=component_id,
            operation=operation,
            model_identity=self._link.link_fingerprint,
            branch_id=self.session_id,
            trajectory_position=position,
            numerical_contract=self._link.manifest.numerical_contract,
            bindings=bindings,
            parent_frame_id=(
                parent_frame_id
                if parent_frame_id is not None
                else (self._last_io_frame.frame_id if self._last_io_frame is not None else None)
            ),
            checkpoint_id=checkpoint_id,
            placement=resolved_placement,
            frame_id=frame_id,
        )
        raw_spec = self._link.manifest.component_io.get(component_id)
        if raw_spec is not None:
            ComponentIOSpec.from_mapping(raw_spec).validate_frame(frame)
        self._last_io_frame = frame
        return frame

    def _require_open(self) -> None:
        if self._state.status == "closed":
            raise ProgramStateError("program session is closed")

    def _program_context_identity(self, payload: Any) -> tuple[str, str]:
        key = self._program_context_key(payload)
        checksum_hook = getattr(
            self._link.backend,
            "program_context_checksum",
            None,
        )
        if callable(checksum_hook):
            backend_checksum = checksum_hook(payload)
            if not isinstance(backend_checksum, str) or not backend_checksum.strip():
                raise ProgramABIError("program_context_checksum() must return a non-empty string")
            checksum_payload = {
                "context_key": key,
                "backend_checksum": backend_checksum,
            }
        else:
            # A required family context key is a content-addressed identity by
            # contract. Hashing it gives checkpoints a fixed-width checksum and
            # protects checkpoint metadata from accidental key corruption.
            checksum_payload = {"context_key": key}
        return key, _digest(checksum_payload)

    def _validate_current_context(self) -> tuple[str, str] | None:
        if not _is_family_backend(self._link.backend):
            return None
        if self._context_payload is None:
            if self._context_checksum is not None:
                raise ProgramABIError("empty family context retained a checksum")
            return None
        key, checksum = self._program_context_identity(self._context_payload)
        if self._state.conditioning_key != key:
            raise ProgramABIError(
                "live family context key does not match the committed program state"
            )
        if self._context_checksum != checksum:
            raise ProgramABIError(
                "live family context checksum does not match the committed payload"
            )
        return key, checksum

    def _clone_program_context(self, payload: Any) -> Any:
        clone = getattr(self._link.backend, "clone_program_context", None)
        if not callable(clone):  # pragma: no cover - construction validates this
            raise ProgramCapabilityError("family backend does not expose clone_program_context()")
        cloned = clone(payload)
        if cloned is None:
            raise ProgramABIError("clone_program_context() returned an empty context")
        if cloned is payload and isinstance(payload, (Mapping, list, set)):
            raise ProgramABIError("clone_program_context() returned the live mutable context")
        return cloned

    def _capture_program_context(
        self,
        *,
        verify_snapshot: bool = False,
        known_identity: tuple[str, str] | None = None,
    ) -> _ProgramContextCapture:
        if self._context_payload is None:
            raise ProgramStateError("session has no compiled program context")
        key, checksum = (
            self._program_context_identity(self._context_payload)
            if known_identity is None
            else known_identity
        )
        contract = _family_context_contract(self._link.backend)
        if contract == "immutable":
            return _ProgramContextCapture(
                payload=self._context_payload,
                key=key,
                checksum=checksum,
            )
        if contract == "clone":
            cloned = self._clone_program_context(self._context_payload)
            clone_key, clone_checksum = self._program_context_identity(cloned)
            if (clone_key, clone_checksum) != (key, checksum):
                raise ProgramABIError("clone_program_context() changed program context identity")
            return _ProgramContextCapture(
                payload=cloned,
                key=key,
                checksum=checksum,
            )
        if contract != "snapshot":  # pragma: no cover - construction validates this
            raise ProgramCapabilityError("family context contract is unavailable")
        snapshot_hook = getattr(
            self._link.backend,
            "snapshot_program_context",
            None,
        )
        if not callable(snapshot_hook):  # pragma: no cover - validated at construction
            raise ProgramCapabilityError(
                "family backend does not expose snapshot_program_context()"
            )
        snapshot = snapshot_hook(self._context_payload)
        if snapshot is None:
            raise ProgramABIError("snapshot_program_context() returned an empty snapshot")
        capture = _ProgramContextCapture(
            payload=snapshot,
            key=key,
            checksum=checksum,
            is_snapshot=True,
        )
        if verify_snapshot:
            self._restore_program_context_capture(capture)
        return capture

    def _restore_program_context_capture(
        self,
        capture: _ProgramContextCapture,
    ) -> Any:
        if capture.is_snapshot:
            restore = getattr(
                self._link.backend,
                "restore_program_context",
                None,
            )
            if not callable(restore):  # pragma: no cover - construction validates this
                raise ProgramCapabilityError(
                    "family backend does not expose restore_program_context()"
                )
            restored = restore(capture.payload)
            if restored is None:
                raise ProgramABIError("restore_program_context() returned an empty context")
        else:
            restored = capture.payload
        key, checksum = self._program_context_identity(restored)
        if (key, checksum) != (capture.key, capture.checksum):
            raise ProgramABIError("captured family context no longer matches its identity")
        return restored

    def _poison_family_context(self, source: ProgramState) -> None:
        self._context_payload = None
        self._context_checksum = None
        self._output = None
        self._backend_checkpoint = None
        self._last_io_frame = None
        self._state = replace(
            source,
            status="closed",
            output_available=False,
            last_operation="rollback:family_context_failed",
        )

    def _program_context_key(self, payload: Any) -> str:
        key_hook = getattr(self._link.backend, "program_context_key", None)
        if _is_family_backend(self._link.backend):
            if not callable(key_hook):  # pragma: no cover - construction validates this
                raise ProgramCapabilityError("family backend does not expose program_context_key()")
            key = key_hook(payload)
        else:
            key = getattr(payload, "key", None)
            if key is None:
                key = f"context:{payload_fingerprint(payload)}"
        if not isinstance(key, str) or not key.strip():
            raise ProgramABIError("program context key must be a non-empty string")
        return key

    def _program_compatibility_key(self) -> tuple[Any, ...]:
        compatibility_hook = getattr(
            self._link.backend,
            "program_compatibility_key",
            None,
        )
        if callable(compatibility_hook):
            key = compatibility_hook(self, self._state, self._context_payload)
            if not isinstance(key, tuple):
                raise ProgramABIError("program_compatibility_key() must return a tuple")
        else:
            key = (self._state.resolution, self._state.total_steps)
        try:
            hash(key)
        except TypeError as exc:
            raise ProgramABIError("program compatibility key must be hashable") from exc
        return key

    def _program_step_terminal(
        self,
        *,
        source: ProgramState,
        output: Any,
        next_context: Any,
        metadata: Mapping[str, Any],
    ) -> bool:
        terminal_hook = getattr(self._link.backend, "program_step_terminal", None)
        if not callable(terminal_hook):
            return True
        terminal = terminal_hook(
            self,
            source,
            next_context,
            output,
            metadata,
        )
        if not isinstance(terminal, bool):
            raise ProgramABIError("program_step_terminal() must return bool")
        return terminal

    def _claim_step(self) -> ProgramState:
        if self._state.status != "context_compiled" or self._context_payload is None:
            raise ProgramStateError(
                "step requires a context-compiled session; fork a completed session before rerun"
            )
        source = self._state
        operation = (
            "execute:program_step"
            if callable(getattr(self._link.backend, "step_program_context", None))
            else "execute:denoise_step"
        )
        self._state = replace(source, status="running", last_operation=operation)
        return source

    def _claim_batch_step(self) -> ProgramState:
        return self._claim_step()

    def _commit_step(
        self,
        source: ProgramState,
        output: Any,
        *,
        next_context: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        next_identity: tuple[str, str] | None = None,
    ) -> ProgramState:
        if self._state.generation != source.generation or self._state.status != "running":
            raise ProgramStateError("session state changed before transactional commit")
        family_step = callable(getattr(self._link.backend, "step_program_context", None))
        resolved_context = self._context_payload if next_context is None else next_context
        if resolved_context is None:
            raise ProgramStateError("program step cannot commit an empty context")
        step_metadata = dict(metadata or {})
        terminal = self._program_step_terminal(
            source=source,
            output=output,
            next_context=resolved_context,
            metadata=step_metadata,
        )
        if next_identity is not None:
            resolved_key, resolved_checksum = next_identity
        elif family_step:
            resolved_key, resolved_checksum = self._program_context_identity(resolved_context)
        else:
            resolved_key = self._program_context_key(resolved_context)
            resolved_checksum = None
        if family_step:
            verified_identity = self._program_context_identity(resolved_context)
            if verified_identity != (resolved_key, resolved_checksum):
                raise ProgramABIError("family terminal hook mutated the next program context")
        next_index = (
            source.step_index + 1
            if family_step
            else (self._state.total_steps or (source.step_index + 1))
        )
        after = replace(
            source,
            generation=source.generation + 1,
            status="completed" if terminal else "context_compiled",
            conditioning_key=resolved_key,
            step_index=next_index,
            output_available=True,
            last_operation=("commit:program_step" if family_step else "commit:denoise_step"),
        )
        self._context_payload = resolved_context
        self._context_checksum = resolved_checksum if family_step else None
        self._output = output
        self._backend_checkpoint = None
        self._state = after
        return after

    def _prepare_call_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(kwargs)
        if "latents" in result:
            if self._initial_latents is not None:
                raise ProgramStateError(
                    "session owns initial latents; do not pass latents explicitly"
                )
        elif self._initial_latents is not None:
            result["latents"] = self._initial_latents
        if "num_inference_steps" not in result and self._state.total_steps is not None:
            result["num_inference_steps"] = self._state.total_steps
        if self._state.resolution is not None:
            result.setdefault("height", self._state.resolution[0])
            result.setdefault("width", self._state.resolution[1])
        if (
            "generator" not in result or result["generator"] is None
        ) and self._state.seed is not None:
            result["generator"] = self._make_generator(self._state.seed)
        return result

    def _make_generator(self, seed: int) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a package dependency
            raise ProgramCapabilityError("seeded program execution requires torch") from exc
        device = getattr(self._link.backend, "_device", "cpu")
        try:
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError):
            generator = torch.Generator()
        return generator.manual_seed(int(seed))

    def _restore_checkpoint(self, checkpoint: ProgramCheckpoint, *, allow_session_id: bool) -> None:
        if checkpoint.state.link_fingerprint != self._link.link_fingerprint:
            raise ProgramABIError("checkpoint belongs to a different linked program")
        if checkpoint.state.status == "running":
            raise ProgramStateError("cannot restore a running checkpoint")
        if not allow_session_id and checkpoint.state.session_id != self.session_id:
            raise ProgramStateError("in-place restore requires the same session_id")
        state = checkpoint.state
        if allow_session_id and state.session_id != self.session_id:
            state = replace(state, session_id=self.session_id)
        context_payload = (
            checkpoint.context_payload
            if checkpoint.context_payload is not None
            else checkpoint.embeds
        )
        family_backend = _is_family_backend(self._link.backend)
        restored_checksum: str | None = None
        if family_backend and checkpoint.context_payload is not None:
            if checkpoint.context_key is None or checkpoint.context_checksum is None:
                raise ProgramABIError("family checkpoint is missing its context identity")
            expected_checkpoint_id = _family_checkpoint_id(
                checkpoint.state,
                context_key=checkpoint.context_key,
                context_checksum=checkpoint.context_checksum,
                context_is_snapshot=checkpoint.context_is_snapshot,
                has_initial_latents=checkpoint.initial_latents is not None,
            )
            if checkpoint.checkpoint_id != expected_checkpoint_id:
                raise ProgramABIError(
                    "family checkpoint ID does not match its state/context identity"
                )
            contract = _family_context_contract(self._link.backend)
            if checkpoint.context_is_snapshot != (contract == "snapshot"):
                raise ProgramABIError(
                    "family checkpoint context representation does not match backend contract"
                )
            capture = _ProgramContextCapture(
                payload=context_payload,
                key=checkpoint.context_key,
                checksum=checkpoint.context_checksum,
                is_snapshot=checkpoint.context_is_snapshot,
            )
            if contract == "clone":
                cloned = self._clone_program_context(context_payload)
                capture = replace(capture, payload=cloned)
            restored_context = self._restore_program_context_capture(capture)
            if contract == "immutable":
                restored_context = _freeze_program_context(restored_context)
            restored_key, restored_checksum = self._program_context_identity(restored_context)
            if (restored_key, restored_checksum) != (
                checkpoint.context_key,
                checkpoint.context_checksum,
            ):
                raise ProgramABIError("checkpoint context checksum does not match its payload")
            if state.conditioning_key != restored_key:
                raise ProgramABIError("checkpoint context key does not match its program state")
        else:
            if family_backend and checkpoint.embeds is not None:
                raise ProgramABIError(
                    "family backend cannot restore a legacy prompt-embeds checkpoint"
                )
            restored_context = _clone_payload(context_payload)
            if restored_context is not None:
                restored_key = self._program_context_key(restored_context)
                if state.conditioning_key not in {None, restored_key}:
                    raise ProgramABIError("checkpoint context key does not match its program state")
        self._state = state
        self._context_payload = restored_context
        self._context_checksum = restored_checksum
        self._initial_latents = _clone_payload(checkpoint.initial_latents)
        self._output = _clone_payload(checkpoint.output)
        self._backend_checkpoint = checkpoint.backend_checkpoint
        self._last_io_frame = checkpoint.io_frame


class ProgramRuntime:
    """Thread-safe handle registry for serving linked programs as sessions.

    This is intentionally transport-neutral. An HTTP, RPC, or local CLI
    adapter can map routes such as ``/programs``, ``/sessions``, ``/step``, and
    ``/checkpoint`` onto this registry without giving the transport direct
    ownership of model state.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._programs: dict[str, DiffusionProgram] = {}
        self._links: dict[str, ProgramLink] = {}
        self._sessions: dict[str, ProgramSession] = {}

    def register(self, program: DiffusionProgram, *, program_id: str | None = None) -> str:
        if not isinstance(program, DiffusionProgram):
            raise ProgramCapabilityError("runtime.register requires a DiffusionProgram")
        key = program_id or program.program_fingerprint
        if not key:
            raise ProgramABIError("program_id cannot be empty")
        with self._lock:
            existing = self._programs.get(key)
            if existing is not None and existing.program_fingerprint != program.program_fingerprint:
                raise ProgramABIError(f"program_id {key!r} is already bound to another manifest")
            self._programs[key] = program
        return key

    def link(self, program_id: str, **kwargs: Any) -> str:
        with self._lock:
            program = self._programs.get(program_id)
            if program is None:
                raise ProgramStateError(f"unknown program_id {program_id!r}")
            linked = program.link(**kwargs)
            self._links[linked.link_fingerprint] = linked
            return linked.link_fingerprint

    def open_session(self, link_id: str, **kwargs: Any) -> str:
        with self._lock:
            link = self._links.get(link_id)
            if link is None:
                raise ProgramStateError(f"unknown link_id {link_id!r}")
            session = link.open_session(**kwargs)
            if session.session_id in self._sessions:
                raise ProgramStateError(f"session_id {session.session_id!r} is already active")
            self._sessions[session.session_id] = session
            return session.session_id

    def get_session(self, session_id: str) -> ProgramSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ProgramStateError(f"unknown session_id {session_id!r}")
            return session

    def compile_context(self, session_id: str, prompt: str, **params: Any) -> Any:
        return self.get_session(session_id).compile_context(prompt, **params)

    def step(self, session_id: str, **kwargs: Any) -> ProgramStepResult:
        return self.get_session(session_id).step(**kwargs)

    def step_batch(
        self, session_ids: Sequence[str], **kwargs: Any
    ) -> tuple[ProgramStepResult, ...]:
        sessions = tuple(self.get_session(session_id) for session_id in session_ids)
        if not sessions:
            raise ProgramStateError("runtime.step_batch requires at least one session")
        return sessions[0].link.step_batch(sessions, **kwargs)

    def checkpoint(self, session_id: str) -> ProgramCheckpoint:
        return self.get_session(session_id).checkpoint()

    def pause(self, session_id: str, *, cut_step: int, **kwargs: Any) -> ProgramCheckpoint:
        return self.get_session(session_id).pause(cut_step=cut_step, **kwargs)

    def resume(
        self,
        session_id: str,
        checkpoint: ProgramCheckpoint | None = None,
        *,
        latent_override: Any | None = None,
        output_type: str = "pil",
    ) -> ProgramStepResult:
        return self.get_session(session_id).resume(
            checkpoint,
            latent_override=latent_override,
            output_type=output_type,
        )

    def replay_batch(
        self,
        link_id: str,
        checkpoint: ProgramCheckpoint,
        *,
        branch_ids: Sequence[str],
        latent_overrides: Any | Sequence[Any] | None = None,
        mode: str = "exact",
        output_type: str = "pil",
    ) -> Any:
        with self._lock:
            link = self._links.get(link_id)
            if link is None:
                raise ProgramStateError(f"unknown link_id {link_id!r}")
        return link.replay_batch(
            checkpoint,
            branch_ids=branch_ids,
            latent_overrides=latent_overrides,
            mode=mode,
            output_type=output_type,
        )

    def restore(
        self,
        link_id: str,
        checkpoint: ProgramCheckpoint,
        *,
        session_id: str | None = None,
    ) -> str:
        with self._lock:
            link = self._links.get(link_id)
            if link is None:
                raise ProgramStateError(f"unknown link_id {link_id!r}")
            session = link.restore(checkpoint, session_id=session_id)
            if session.session_id in self._sessions:
                raise ProgramStateError(f"session_id {session.session_id!r} is already active")
            self._sessions[session.session_id] = session
            return session.session_id

    def close_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ProgramStateError(f"unknown session_id {session_id!r}")
            session.close()
            del self._sessions[session_id]

    def close(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            for session in sessions:
                session.close()
            self._sessions.clear()


def _stack_latents(values: Sequence[Any]) -> Any:
    if not values or all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ProgramStateError("either all or none of the batch sessions may own initial latents")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a package dependency
        raise ProgramCapabilityError("batched latent state requires torch") from exc
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ProgramCapabilityError("batched initial_latents must be torch tensors")
    tensors = tuple(values)
    if any(tensor.ndim == 0 or int(tensor.shape[0]) != 1 for tensor in tensors):
        raise ProgramStateError("each session latent payload must have leading dimension one")
    try:
        return torch.cat(tensors, dim=0)
    except (RuntimeError, TypeError) as exc:
        raise ProgramABIError(
            "batch latent payloads have incompatible shape, dtype, or device"
        ) from exc


def _rows_by_branch(result: Any, branch_ids: tuple[str, ...]) -> dict[str, Any]:
    rows = getattr(result, "rows_by_branch", None)
    if not callable(rows):
        raise ProgramCapabilityError("batched backend result must expose rows_by_branch()")
    try:
        mapped = rows()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ProgramCapabilityError("batched backend output could not be split into rows") from exc
    if set(mapped) != set(branch_ids):
        raise ProgramCapabilityError("batched backend returned the wrong branch IDs")
    return {branch_id: mapped[branch_id] for branch_id in branch_ids}


__all__ = [
    "DenoiseBinding",
    "DiffusionProgram",
    "PROGRAM_CHECKPOINT_SCHEMA",
    "PROGRAM_MANIFEST_SCHEMA",
    "PROGRAM_STATE_SCHEMA",
    "ProgramABIError",
    "ProgramCapabilityError",
    "ProgramCheckpoint",
    "ProgramError",
    "ProgramExtension",
    "ProgramLink",
    "ProgramManifest",
    "ProgramRuntime",
    "ProgramSession",
    "ProgramState",
    "ProgramStateError",
    "ProgramStepResult",
]
