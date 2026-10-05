"""Typed multi-stream component IO for the MARS diffusion runtime.

The native tensor call remains the numerical authority.  These contracts make
the surrounding execution state explicit without putting a debugger or a
transport layer inside the model's kernels:

``data + control + state + route + resource + evidence``

Payloads themselves stay owned by the backend/session.  A ``ComponentFrame``
only carries handles, ABI metadata, provenance, and the sideband information
needed to replay and inspect a boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

COMPONENT_IO_SCHEMA = "mrun-diffusion-component-io-v1"
COMPONENT_FRAME_SCHEMA = "mrun-diffusion-component-frame-v1"
IO_STREAMS = ("data", "control", "state", "route", "resource", "evidence")
IO_DIRECTIONS = ("in", "out")
IO_MUTABILITY = ("immutable", "borrowed", "copy_on_write", "owned", "sideband")


class ComponentIOError(ValueError):
    """Raised when a component contract or execution frame is invalid."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    raise TypeError(f"component IO metadata contains unsupported {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _names(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ComponentIOError(f"{field_name} cannot contain empty names")
    if len(result) != len(set(result)):
        raise ComponentIOError(f"{field_name} must contain unique names")
    return result


@dataclass(frozen=True, slots=True)
class PortContract:
    """One named component port and the stream that owns its lifecycle."""

    name: str
    stream: str
    direction: str = "in"
    value_kind: str = "tensor"
    required: bool = True
    mutability: str = "borrowed"
    debug: bool = False
    abi: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ComponentIOError("port name must be non-empty")
        if self.stream not in IO_STREAMS:
            raise ComponentIOError(f"unknown IO stream {self.stream!r}")
        if self.direction not in IO_DIRECTIONS:
            raise ComponentIOError(f"unknown port direction {self.direction!r}")
        if self.mutability not in IO_MUTABILITY:
            raise ComponentIOError(f"unknown port mutability {self.mutability!r}")
        if not self.value_kind.strip():
            raise ComponentIOError("port value_kind must be non-empty")
        object.__setattr__(self, "abi", dict(_jsonable(self.abi)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stream": self.stream,
            "direction": self.direction,
            "value_kind": self.value_kind,
            "required": self.required,
            "mutability": self.mutability,
            "debug": self.debug,
            "abi": dict(self.abi),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PortContract:
        return cls(
            name=str(value.get("name", "")),
            stream=str(value.get("stream", "")),
            direction=str(value.get("direction", "in")),
            value_kind=str(value.get("value_kind", "tensor")),
            required=bool(value.get("required", True)),
            mutability=str(value.get("mutability", "borrowed")),
            debug=bool(value.get("debug", False)),
            abi=value.get("abi", {}),
        )


@dataclass(frozen=True, slots=True)
class ComponentIOSpec:
    """Declarative input/output contract for one model component."""

    component_id: str
    phase: str
    inputs: tuple[PortContract, ...] = ()
    outputs: tuple[PortContract, ...] = ()
    debug_ports: tuple[str, ...] = ()
    numerical_contract: str = "native-projection"
    schema: str = COMPONENT_IO_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPONENT_IO_SCHEMA:
            raise ComponentIOError(f"unsupported component IO schema {self.schema!r}")
        if not self.component_id.strip() or not self.phase.strip():
            raise ComponentIOError("component_id and phase must be non-empty")
        input_names = [port.name for port in self.inputs]
        output_names = [port.name for port in self.outputs]
        if len(input_names) != len(set(input_names)):
            raise ComponentIOError(f"duplicate input ports for {self.component_id!r}")
        if len(output_names) != len(set(output_names)):
            raise ComponentIOError(f"duplicate output ports for {self.component_id!r}")
        if any(port.direction != "in" for port in self.inputs):
            raise ComponentIOError(
                f"input declarations must have direction='in' for {self.component_id!r}"
            )
        if any(port.direction != "out" for port in self.outputs):
            raise ComponentIOError(
                f"output declarations must have direction='out' for {self.component_id!r}"
            )
        names = set(input_names) | set(output_names)
        unknown_debug = set(self.debug_ports) - names
        if unknown_debug:
            raise ComponentIOError(
                f"debug ports {sorted(unknown_debug)!r} are not declared for {self.component_id!r}"
            )
        object.__setattr__(self, "debug_ports", _names(self.debug_ports, "debug_ports"))

    @property
    def ports(self) -> tuple[PortContract, ...]:
        return self.inputs + self.outputs

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    def validate_frame(self, frame: ComponentFrame) -> None:
        if frame.component_id != self.component_id:
            raise ComponentIOError(
                f"frame targets {frame.component_id!r}, contract is {self.component_id!r}"
            )
        declared = {(port.direction, port.name): port for port in self.ports}
        seen: set[tuple[str, str]] = set()
        for binding in frame.bindings:
            key = (binding.direction, binding.port)
            contract = declared.get(key)
            if contract is None:
                raise ComponentIOError(
                    f"frame binds undeclared {binding.direction} port {binding.port!r}"
                )
            if binding.stream != contract.stream:
                raise ComponentIOError(
                    f"port {binding.port!r} uses stream {binding.stream!r}; "
                    f"contract requires {contract.stream!r}"
                )
            seen.add(key)
        missing = [
            port.name
            for port in self.inputs
            if port.required and ("in", port.name) not in seen
        ]
        if missing:
            raise ComponentIOError(
                f"frame for {self.component_id!r} is missing required inputs {missing!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "component_id": self.component_id,
            "phase": self.phase,
            "inputs": [port.to_dict() for port in self.inputs],
            "outputs": [port.to_dict() for port in self.outputs],
            "debug_ports": list(self.debug_ports),
            "numerical_contract": self.numerical_contract,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ComponentIOSpec:
        return cls(
            schema=str(value.get("schema", COMPONENT_IO_SCHEMA)),
            component_id=str(value.get("component_id", "")),
            phase=str(value.get("phase", "run")),
            inputs=tuple(PortContract.from_mapping(port) for port in value.get("inputs", ())),
            outputs=tuple(PortContract.from_mapping(port) for port in value.get("outputs", ())),
            debug_ports=tuple(str(port) for port in value.get("debug_ports", ())),
            numerical_contract=str(value.get("numerical_contract", "native-projection")),
        )


@dataclass(frozen=True, slots=True)
class PortBinding:
    """Metadata for one bound value; the payload stays outside the frame."""

    port: str
    stream: str
    direction: str
    handle: str | None = None
    payload_fingerprint: str | None = None
    abi: Mapping[str, Any] = field(default_factory=dict)
    producer: str | None = None
    consumer: str | None = None
    branch_id: str = "control"
    checkpoint_id: str | None = None
    retention: str = "summary"

    def __post_init__(self) -> None:
        if not self.port.strip():
            raise ComponentIOError("bound port must be non-empty")
        if self.stream not in IO_STREAMS or self.direction not in IO_DIRECTIONS:
            raise ComponentIOError("bound port has an invalid stream or direction")
        if not self.branch_id.strip():
            raise ComponentIOError("bound port branch_id must be non-empty")
        object.__setattr__(self, "abi", dict(_jsonable(self.abi)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "stream": self.stream,
            "direction": self.direction,
            "handle": self.handle,
            "payload_fingerprint": self.payload_fingerprint,
            "abi": dict(self.abi),
            "producer": self.producer,
            "consumer": self.consumer,
            "branch_id": self.branch_id,
            "checkpoint_id": self.checkpoint_id,
            "retention": self.retention,
        }


@dataclass(frozen=True, slots=True)
class ComponentFrame:
    """One component invocation and its debugger-visible sideband."""

    frame_id: str
    component_id: str
    operation: str
    model_identity: str
    branch_id: str = "control"
    trajectory_position: Mapping[str, Any] = field(default_factory=dict)
    numerical_contract: Mapping[str, Any] = field(default_factory=dict)
    bindings: tuple[PortBinding, ...] = ()
    trace_token: str = ""
    parent_frame_id: str | None = None
    checkpoint_id: str | None = None
    placement: Mapping[str, Any] = field(default_factory=dict)
    schema: str = COMPONENT_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPONENT_FRAME_SCHEMA:
            raise ComponentIOError(f"unsupported component frame schema {self.schema!r}")
        for name in ("frame_id", "component_id", "operation", "model_identity", "branch_id"):
            if not str(getattr(self, name)).strip():
                raise ComponentIOError(f"component frame {name} must be non-empty")
        binding_keys = [(binding.direction, binding.port) for binding in self.bindings]
        if len(binding_keys) != len(set(binding_keys)):
            raise ComponentIOError("component frame contains duplicate port bindings")
        token = self.trace_token or f"trace-{_digest(self._identity_payload())[:24]}"
        object.__setattr__(self, "trace_token", token)
        object.__setattr__(self, "trajectory_position", dict(_jsonable(self.trajectory_position)))
        object.__setattr__(self, "numerical_contract", dict(_jsonable(self.numerical_contract)))
        object.__setattr__(self, "placement", dict(_jsonable(self.placement)))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "component_id": self.component_id,
            "operation": self.operation,
            "model_identity": self.model_identity,
            "branch_id": self.branch_id,
            "trajectory_position": dict(self.trajectory_position),
            "numerical_contract": dict(self.numerical_contract),
            "bindings": [binding.to_dict() for binding in self.bindings],
            "parent_frame_id": self.parent_frame_id,
            "checkpoint_id": self.checkpoint_id,
            "placement": dict(self.placement),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                **self._identity_payload(),
                "frame_id": self.frame_id,
                "trace_token": self.trace_token,
            }
        )

    def validate(self, spec: ComponentIOSpec) -> None:
        spec.validate_frame(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "frame_id": self.frame_id,
            "trace_token": self.trace_token,
            "fingerprint": self.fingerprint,
        }


def payload_abi(value: Any) -> dict[str, Any]:
    """Return cheap shape/dtype/device metadata without owning the payload."""

    result: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        result["shape"] = [int(item) for item in shape]
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        result["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        result["device"] = str(device)
    return result


def payload_fingerprint(value: Any) -> str:
    """Fingerprint a payload descriptor; backends may replace it with a content hash."""

    return _digest(payload_abi(value))


def make_component_frame(
    *,
    component_id: str,
    operation: str,
    model_identity: str,
    branch_id: str = "control",
    trajectory_position: Mapping[str, Any] | None = None,
    numerical_contract: Mapping[str, Any] | None = None,
    bindings: Sequence[PortBinding] = (),
    parent_frame_id: str | None = None,
    checkpoint_id: str | None = None,
    placement: Mapping[str, Any] | None = None,
    frame_id: str | None = None,
) -> ComponentFrame:
    """Construct a deterministic frame while allowing an explicit invocation ID."""

    values = {
        "component_id": component_id,
        "operation": operation,
        "model_identity": model_identity,
        "branch_id": branch_id,
        "trajectory_position": dict(trajectory_position or {}),
        "numerical_contract": dict(numerical_contract or {}),
        "bindings": [binding.to_dict() for binding in bindings],
        "parent_frame_id": parent_frame_id,
        "checkpoint_id": checkpoint_id,
        "placement": dict(placement or {}),
    }
    resolved_id = frame_id or f"frame-{_digest(values)[:24]}"
    return ComponentFrame(
        frame_id=resolved_id,
        trace_token=f"trace-{_digest({**values, 'frame_id': resolved_id})[:24]}",
        component_id=component_id,
        operation=operation,
        model_identity=model_identity,
        branch_id=branch_id,
        trajectory_position=trajectory_position or {},
        numerical_contract=numerical_contract or {},
        bindings=tuple(bindings),
        parent_frame_id=parent_frame_id,
        checkpoint_id=checkpoint_id,
        placement=placement or {},
    )


def validate_component_io_manifest(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a component spec mapping and fail closed on malformed contracts."""

    if value is None:
        value = component_io_manifest()
    if not isinstance(value, Mapping):
        raise ComponentIOError("component_io must be a mapping of component ID to spec")
    normalized: dict[str, Any] = {}
    for component_id, raw in value.items():
        spec = raw if isinstance(raw, ComponentIOSpec) else ComponentIOSpec.from_mapping(raw)
        if spec.component_id != str(component_id):
            raise ComponentIOError(
                f"component IO key {component_id!r} does not match {spec.component_id!r}"
            )
        normalized[spec.component_id] = spec.to_dict()
    return normalized


def _in(name: str, stream: str, **kwargs: Any) -> PortContract:
    return PortContract(name=name, stream=stream, direction="in", **kwargs)


def _out(name: str, stream: str, **kwargs: Any) -> PortContract:
    return PortContract(name=name, stream=stream, direction="out", **kwargs)


def component_io_specs() -> tuple[ComponentIOSpec, ...]:
    """Return the default MARS IO contract for the current FLUX-shaped pipeline."""

    evidence = {"value_kind": "trace-token", "mutability": "sideband", "debug": True}
    return (
        ComponentIOSpec(
            component_id="program",
            phase="runtime",
            inputs=(
                _in("context_handle", "data", value_kind="conditioning-handle"),
                _in("schedule", "control", value_kind="schedule", mutability="immutable"),
                _in("checkpoint", "state", value_kind="checkpoint", required=False),
                _in("route", "route", value_kind="branch-route", required=False),
                _in("resource_lease", "resource", value_kind="device-lease", required=False),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("output", "data", value_kind="pipeline-output", mutability="owned"),
                _out(
                    "checkpoint",
                    "state",
                    value_kind="checkpoint",
                    required=False,
                    mutability="owned",
                ),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("context_handle", "checkpoint", "output", "trace"),
        ),
        ComponentIOSpec(
            component_id="conditioner",
            phase="encode",
            inputs=(
                _in("prompt", "data", value_kind="text", mutability="immutable"),
                _in("tokenizer_config", "control", value_kind="config", mutability="immutable"),
                _in("cache_handle", "state", value_kind="cache-handle", required=False),
                _in("resource_lease", "resource", value_kind="device-lease", required=False),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("prompt_embeds", "data", value_kind="tensor", mutability="owned"),
                _out("text_ids", "data", value_kind="tensor", mutability="owned"),
                _out("cache_handle", "state", value_kind="cache-handle", mutability="owned"),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("prompt_embeds", "cache_handle", "trace"),
        ),
        ComponentIOSpec(
            component_id="latent_initializer",
            phase="allocate",
            inputs=(
                _in("seed", "control", value_kind="seed", mutability="immutable"),
                _in("resolution", "control", value_kind="shape", mutability="immutable"),
                _in("schedule", "control", value_kind="schedule", mutability="immutable"),
                _in("rng_state", "state", value_kind="rng", required=False),
                _in("resource_lease", "resource", value_kind="device-lease", required=False),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("latent", "data", value_kind="tensor", mutability="owned"),
                _out("latent_ids", "data", value_kind="tensor", mutability="immutable"),
                _out("rng_state", "state", value_kind="rng", mutability="owned"),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("latent", "latent_ids", "trace"),
        ),
        ComponentIOSpec(
            component_id="denoiser",
            phase="denoise",
            inputs=(
                _in("latent", "data", value_kind="tensor"),
                _in("prompt_embeds", "data", value_kind="tensor"),
                _in("text_ids", "data", value_kind="tensor"),
                _in("latent_ids", "data", value_kind="tensor"),
                _in("timestep", "control", value_kind="scalar"),
                _in("trajectory_checkpoint", "state", value_kind="checkpoint", required=False),
                _in("route", "route", value_kind="branch-route", required=False),
                _in("weight_lease", "resource", value_kind="weight-lease"),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("prediction", "data", value_kind="tensor", mutability="owned"),
                _out("next_page_demand", "resource", value_kind="page-demand", required=False),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("latent", "prediction", "trajectory_checkpoint", "trace"),
        ),
        ComponentIOSpec(
            component_id="scheduler",
            phase="denoise",
            inputs=(
                _in("latent", "data", value_kind="tensor"),
                _in("prediction", "data", value_kind="tensor"),
                _in("timestep", "control", value_kind="scalar"),
                _in("schedule_cursor", "state", value_kind="cursor"),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("next_latent", "data", value_kind="tensor", mutability="owned"),
                _out("schedule_cursor", "state", value_kind="cursor", mutability="owned"),
                _out("checkpoint_token", "state", value_kind="checkpoint", mutability="owned"),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("prediction", "next_latent", "checkpoint_token", "trace"),
        ),
        ComponentIOSpec(
            component_id="latent_bridge",
            phase="bridge",
            inputs=(
                _in("final_latent", "data", value_kind="tensor"),
                _in("packing", "control", value_kind="layout", mutability="immutable"),
                _in("normalization", "state", value_kind="normalization"),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("vae_latent_view", "data", value_kind="tensor-view", mutability="borrowed"),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("final_latent", "vae_latent_view", "trace"),
        ),
        ComponentIOSpec(
            component_id="vae",
            phase="decode",
            inputs=(
                _in("vae_latent_view", "data", value_kind="tensor-view"),
                _in("output_selector", "control", value_kind="selector", required=False),
                _in("decoder_resource", "resource", value_kind="device-lease", required=False),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("pixel_tiles", "data", value_kind="tensor", mutability="owned"),
                _out(
                    "feature_stream",
                    "data",
                    value_kind="tensor",
                    required=False,
                    mutability="owned",
                ),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("vae_latent_view", "pixel_tiles", "trace"),
        ),
        ComponentIOSpec(
            component_id="renderer",
            phase="render",
            inputs=(
                _in("pixel_tiles", "data", value_kind="tensor"),
                _in("output_selector", "control", value_kind="selector"),
                _in("artifact_store", "resource", value_kind="artifact-handle", required=False),
                _in("trace", "evidence", **evidence),
            ),
            outputs=(
                _out("pixels", "data", value_kind="image", mutability="owned"),
                _out("artifact_handle", "state", value_kind="artifact-handle", mutability="owned"),
                _out("trace", "evidence", **evidence),
            ),
            debug_ports=("pixel_tiles", "pixels", "artifact_handle", "trace"),
        ),
    )


def component_io_manifest() -> dict[str, Any]:
    return {spec.component_id: spec.to_dict() for spec in component_io_specs()}


__all__ = [
    "COMPONENT_FRAME_SCHEMA",
    "COMPONENT_IO_SCHEMA",
    "ComponentFrame",
    "ComponentIOError",
    "ComponentIOSpec",
    "IO_DIRECTIONS",
    "IO_MUTABILITY",
    "IO_STREAMS",
    "PortBinding",
    "PortContract",
    "component_io_manifest",
    "component_io_specs",
    "make_component_frame",
    "payload_abi",
    "payload_fingerprint",
    "validate_component_io_manifest",
]
