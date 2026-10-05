"""Typed scene-state handles and debugger-visible interventions for MARS.

The state object is intentionally model-neutral. It describes ownership,
types, provenance, and address changes; a backend decides whether and how a
learned model consumes the resulting control payload. A clean result from the
state contract is therefore not evidence that an existing FLUX checkpoint has
learned semantic slots.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

SCENE_STATE_SCHEMA = "mrun-scene-state-v1"
SCENE_INTERVENTION_SCHEMA = "mrun-scene-intervention-v1"
FIELD_KINDS = (
    "attribute",
    "mask",
    "object",
    "reference",
    "relation",
    "scalar",
    "text",
    "vector",
)


class SceneStateError(ValueError):
    """Raised when typed scene state or an intervention is malformed."""


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    raise SceneStateError(f"scene values must be JSON-like, got {type(value).__name__}")


def _digest(value: Any) -> str:
    try:
        payload = json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SceneStateError("scene state is not canonicalizable") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneStateError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class SceneField:
    """One typed value at one address in the scene register file."""

    name: str
    kind: str
    value: Any
    provenance: str = "supplied"
    version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "field name"))
        object.__setattr__(self, "kind", _name(self.kind, "field kind"))
        if self.kind not in FIELD_KINDS:
            raise SceneStateError(f"unknown field kind {self.kind!r}")
        object.__setattr__(self, "value", _canonical(self.value))
        object.__setattr__(self, "provenance", _name(self.provenance, "field provenance"))
        if isinstance(self.version, bool) or int(self.version) < 0:
            raise SceneStateError("field version must be a non-negative integer")
        object.__setattr__(self, "version", int(self.version))

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "provenance": self.provenance,
            "version": self.version,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class SceneSlot:
    """Addressable collection of fields belonging to one scene entity."""

    slot_id: str
    fields: Mapping[str, SceneField] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _name(self.slot_id, "slot ID"))
        normalized: dict[str, SceneField] = {}
        for name, value in self.fields.items():
            if not isinstance(value, SceneField):
                raise SceneStateError("slot fields must contain SceneField values")
            if str(name) != value.name:
                raise SceneStateError(f"slot field key {name!r} does not match {value.name!r}")
            normalized[value.name] = value
        object.__setattr__(self, "fields", MappingProxyType(normalized))

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict(include_fingerprint=False))

    def with_field(self, value: SceneField) -> SceneSlot:
        if not isinstance(value, SceneField):
            raise SceneStateError("with_field requires a SceneField")
        return replace(self, fields={**self.fields, value.name: value})

    def without_field(self, field_name: str) -> SceneSlot:
        name = _name(field_name, "field name")
        if name not in self.fields:
            raise SceneStateError(f"slot {self.slot_id!r} has no field {name!r}")
        fields = dict(self.fields)
        del fields[name]
        return replace(self, fields=fields)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "slot_id": self.slot_id,
            "fields": {name: self.fields[name].to_dict() for name in sorted(self.fields)},
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class SceneState:
    """Immutable typed scene register file with stable content identity."""

    slots: Mapping[str, SceneSlot] = field(default_factory=dict)
    globals: Mapping[str, SceneField] = field(default_factory=dict)
    schema: str = SCENE_STATE_SCHEMA
    generation: int = 0

    def __post_init__(self) -> None:
        if self.schema != SCENE_STATE_SCHEMA:
            raise SceneStateError(f"unsupported scene state schema {self.schema!r}")
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise SceneStateError("scene generation must be non-negative")
        slots: dict[str, SceneSlot] = {}
        for slot_id, slot in self.slots.items():
            if not isinstance(slot, SceneSlot) or str(slot_id) != slot.slot_id:
                raise SceneStateError("scene slots must be keyed by matching SceneSlot IDs")
            slots[slot.slot_id] = slot
        globals_: dict[str, SceneField] = {}
        for name, value in self.globals.items():
            if not isinstance(value, SceneField) or str(name) != value.name:
                raise SceneStateError("scene globals must be keyed by matching SceneField names")
            globals_[value.name] = value
        object.__setattr__(self, "slots", MappingProxyType(slots))
        object.__setattr__(self, "globals", MappingProxyType(globals_))
        object.__setattr__(self, "generation", int(self.generation))

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict(include_fingerprint=False))

    def _next(
        self,
        *,
        slots: Mapping[str, SceneSlot] | None = None,
        globals_: Mapping[str, SceneField] | None = None,
    ) -> SceneState:
        return SceneState(
            slots=self.slots if slots is None else slots,
            globals=self.globals if globals_ is None else globals_,
            generation=self.generation + 1,
        )

    def set_field(
        self,
        slot_id: str,
        field_name: str,
        value: Any,
        *,
        kind: str = "attribute",
        provenance: str = "intervention",
    ) -> SceneState:
        slot = _name(slot_id, "slot ID")
        field = SceneField(
            field_name,
            kind,
            value,
            provenance,
            version=self._field_version(slot, field_name) + 1,
        )
        current = self.slots.get(slot, SceneSlot(slot))
        return self._next(slots={**self.slots, slot: current.with_field(field)})

    def set_global(
        self,
        field_name: str,
        value: Any,
        *,
        kind: str = "attribute",
        provenance: str = "intervention",
    ) -> SceneState:
        name = _name(field_name, "global field name")
        previous = self.globals.get(name)
        field = SceneField(
            name,
            kind,
            value,
            provenance,
            version=0 if previous is None else previous.version + 1,
        )
        return self._next(globals_={**self.globals, name: field})

    def clear_field(self, slot_id: str, field_name: str) -> SceneState:
        slot = _name(slot_id, "slot ID")
        current = self.slots.get(slot)
        if current is None:
            raise SceneStateError(f"unknown scene slot {slot!r}")
        return self._next(slots={**self.slots, slot: current.without_field(field_name)})

    def swap_fields(self, left_slot: str, right_slot: str, field_name: str) -> SceneState:
        left_name = _name(left_slot, "left slot ID")
        right_name = _name(right_slot, "right slot ID")
        field = _name(field_name, "field name")
        if left_name == right_name:
            raise SceneStateError("cannot swap a slot with itself")
        left = self.slots.get(left_name)
        right = self.slots.get(right_name)
        if left is None or right is None:
            raise SceneStateError("swap requires two existing slots")
        left_field = left.fields.get(field)
        right_field = right.fields.get(field)
        if left_field is None or right_field is None:
            raise SceneStateError("swap requires the field on both slots")
        if left_field.kind != right_field.kind:
            raise SceneStateError("swap requires matching field kinds")
        left_new = left.with_field(replace(right_field, name=field, version=left_field.version + 1))
        right_new = right.with_field(
            replace(left_field, name=field, version=right_field.version + 1)
        )
        return self._next(slots={**self.slots, left_name: left_new, right_name: right_new})

    def permute_slots(self, destination_to_source: Mapping[str, str]) -> SceneState:
        if not isinstance(destination_to_source, Mapping) or not destination_to_source:
            raise SceneStateError("slot permutation must be a non-empty mapping")
        destinations = {str(value) for value in destination_to_source}
        sources = {str(value) for value in destination_to_source.values()}
        if destinations != set(self.slots) or sources != set(self.slots):
            raise SceneStateError("slot permutation must cover every existing slot exactly once")
        return self._next(
            slots={
                destination: replace(self.slots[source], slot_id=destination)
                for destination, source in destination_to_source.items()
            }
        )

    def field(self, slot_id: str, field_name: str) -> SceneField:
        slot = self.slots.get(_name(slot_id, "slot ID"))
        if slot is None:
            raise SceneStateError(f"unknown scene slot {slot_id!r}")
        try:
            return slot.fields[_name(field_name, "field name")]
        except KeyError as exc:
            raise SceneStateError(f"unknown field {field_name!r} in slot {slot_id!r}") from exc

    def _field_version(self, slot_id: str, field_name: str) -> int:
        slot = self.slots.get(slot_id)
        return (
            0 if slot is None or field_name not in slot.fields else slot.fields[field_name].version
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a typed control payload suitable for an adapter boundary."""

        return self.to_dict(include_fingerprint=True)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "generation": self.generation,
            "globals": {name: value.to_dict() for name, value in sorted(self.globals.items())},
            "slots": {slot_id: slot.to_dict() for slot_id, slot in sorted(self.slots.items())},
        }
        if include_fingerprint:
            payload["fingerprint"] = _digest(payload)
        return payload


@dataclass(frozen=True, slots=True)
class SceneIntervention:
    """Replayable field/slot operation for debugger controls."""

    operation: str
    slot_id: str | None = None
    field_name: str | None = None
    value: Any = None
    kind: str = "attribute"
    source_slot_id: str | None = None
    destination_slot_id: str | None = None
    permutation: Mapping[str, str] = field(default_factory=dict)
    schema: str = SCENE_INTERVENTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCENE_INTERVENTION_SCHEMA:
            raise SceneStateError(f"unsupported intervention schema {self.schema!r}")
        if self.operation not in {"set", "clear", "swap", "permute"}:
            raise SceneStateError(f"unknown intervention operation {self.operation!r}")
        if self.operation in {"set", "clear"}:
            _name(self.slot_id or "", "slot ID")
            _name(self.field_name or "", "field name")
        if self.operation == "swap":
            _name(self.source_slot_id or "", "source slot ID")
            _name(self.destination_slot_id or "", "destination slot ID")
            _name(self.field_name or "", "field name")
        if self.operation == "permute" and not self.permutation:
            raise SceneStateError("permute intervention requires a mapping")

    @classmethod
    def set(
        cls,
        slot_id: str,
        field_name: str,
        value: Any,
        *,
        kind: str = "attribute",
    ) -> SceneIntervention:
        return cls("set", slot_id=slot_id, field_name=field_name, value=value, kind=kind)

    @classmethod
    def clear(cls, slot_id: str, field_name: str) -> SceneIntervention:
        return cls("clear", slot_id=slot_id, field_name=field_name)

    @classmethod
    def swap(cls, left_slot: str, right_slot: str, field_name: str) -> SceneIntervention:
        return cls(
            "swap",
            source_slot_id=left_slot,
            destination_slot_id=right_slot,
            field_name=field_name,
        )

    @classmethod
    def permute(cls, destination_to_source: Mapping[str, str]) -> SceneIntervention:
        return cls("permute", permutation=dict(destination_to_source))

    def apply(self, state: SceneState) -> SceneState:
        if not isinstance(state, SceneState):
            raise SceneStateError("interventions apply only to SceneState")
        if self.operation == "set":
            return state.set_field(
                self.slot_id or "",
                self.field_name or "",
                self.value,
                kind=self.kind,
            )
        if self.operation == "clear":
            return state.clear_field(self.slot_id or "", self.field_name or "")
        if self.operation == "swap":
            return state.swap_fields(
                self.source_slot_id or "",
                self.destination_slot_id or "",
                self.field_name or "",
            )
        return state.permute_slots(self.permutation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operation": self.operation,
            "slot_id": self.slot_id,
            "field_name": self.field_name,
            "value": _canonical(self.value),
            "kind": self.kind,
            "source_slot_id": self.source_slot_id,
            "destination_slot_id": self.destination_slot_id,
            "permutation": dict(self.permutation),
        }


@dataclass(frozen=True, slots=True)
class SceneInterventionRecord:
    """Evidence object tying an intervention to before/after state identity."""

    intervention: SceneIntervention
    before_fingerprint: str
    after_fingerprint: str
    changed_addresses: tuple[str, ...]
    before_generation: int
    after_generation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mrun-scene-intervention-record-v1",
            "intervention": self.intervention.to_dict(),
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "changed_addresses": list(self.changed_addresses),
            "before_generation": self.before_generation,
            "after_generation": self.after_generation,
        }


class SceneStateDebugger:
    """Apply typed interventions while retaining a minimal causal record."""

    @staticmethod
    def apply(
        state: SceneState,
        intervention: SceneIntervention,
    ) -> tuple[SceneState, SceneInterventionRecord]:
        before = state.to_payload()
        after_state = intervention.apply(state)
        after = after_state.to_payload()
        changed: list[str] = []
        for slot_id in sorted(set(before["slots"]) | set(after["slots"])):
            before_fields = before["slots"].get(slot_id, {}).get("fields", {})
            after_fields = after["slots"].get(slot_id, {}).get("fields", {})
            for field_name in sorted(set(before_fields) | set(after_fields)):
                if before_fields.get(field_name) != after_fields.get(field_name):
                    changed.append(f"slots.{slot_id}.{field_name}")
        for field_name in sorted(set(before["globals"]) | set(after["globals"])):
            if before["globals"].get(field_name) != after["globals"].get(field_name):
                changed.append(f"globals.{field_name}")
        return after_state, SceneInterventionRecord(
            intervention=intervention,
            before_fingerprint=state.fingerprint,
            after_fingerprint=after_state.fingerprint,
            changed_addresses=tuple(changed),
            before_generation=state.generation,
            after_generation=after_state.generation,
        )


__all__ = [
    "FIELD_KINDS",
    "SCENE_INTERVENTION_SCHEMA",
    "SCENE_STATE_SCHEMA",
    "SceneField",
    "SceneIntervention",
    "SceneInterventionRecord",
    "SceneSlot",
    "SceneState",
    "SceneStateDebugger",
    "SceneStateError",
]
