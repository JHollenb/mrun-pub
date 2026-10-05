"""Transactional CPU state target for the strict routed-only Mixtral reference lane.

This module is a G10 correctness target, not a production kernel.  It executes only newly supplied
tokens, owns explicit per-layer K/V state, and returns a provisional transaction whose commit is
epoch-bound.  The implementation exists to prove that the decomposed MoE graph has coherent
prefill/decode semantics before an MLX or CUDA lowering is allowed to claim them.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ._json import canonical_sha256, require_sha256
from .emitter import ComponentArtifact
from .errors import DecompilerError
from .reference import (
    ReferenceExecutable,
    _mixtral_config_semantics,
    _moe_dispatch,
    _moe_top_k,
    _moe_weighted_scatter_add,
    _rms_norm,
    _rotary_parameter,
    _tensor_fingerprint,
    lower_component_artifact_to_reference,
)

MIXTRAL_STATEFUL_CERTIFICATION_SCHEMA = "mrun-mixtral-stateful-certification-v1"


class MixtralStatefulError(DecompilerError):
    """A Mixtral reference state or transaction violated its G10 contract."""

    code = "mixtral_stateful_execution_failure"
    gate = "G10"


class MixtralStatefulParityError(DecompilerError):
    """Incremental Mixtral execution disagreed with stateless full-sequence execution."""

    code = "mixtral_stateful_parity_failure"
    gate = "G10"


@dataclass(frozen=True, slots=True)
class MixtralStateObservation:
    length: int
    epoch: int
    batch_size: int | None
    capacity: int
    closed: bool
    state_fingerprint: str

    def __post_init__(self) -> None:
        if self.length < 0 or self.epoch < 0 or self.capacity <= 0 or self.length > self.capacity:
            raise ValueError("invalid Mixtral state observation counters")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("invalid Mixtral state observation batch size")
        require_sha256(self.state_fingerprint, field="Mixtral state fingerprint")


@dataclass(slots=True)
class MixtralProvisionalStep:
    """One single-use, epoch-bound state delta."""

    _state: MixtralReferenceState = field(repr=False)
    base_epoch: int
    base_length: int
    token_count: int
    logits: torch.Tensor
    _next_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...] = field(repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def commit(self) -> MixtralStateObservation:
        return self._state._commit(self)

    def rollback(self) -> MixtralStateObservation:
        return self._state._rollback(self)


class MixtralReferenceState:
    """Bounded, transactional K/V owner for one batch shape."""

    def __init__(self, engine: MixtralStatefulReference, *, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Mixtral state capacity must be a positive integer")
        if capacity > engine.max_position_embeddings:
            raise ValueError("Mixtral state capacity exceeds the registered context")
        self._engine = engine
        self._capacity = capacity
        self._length = 0
        self._epoch = 0
        self._batch_size: int | None = None
        self._caches: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        self._closed = False
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def length(self) -> int:
        with self._lock:
            return self._length

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def _fingerprint_locked(self) -> str:
        caches = [
            {
                "layer": layer,
                "k": _tensor_fingerprint(key),
                "v": _tensor_fingerprint(value),
            }
            for layer, (key, value) in enumerate(self._caches)
        ]
        return canonical_sha256(
            {
                "length": self._length,
                "epoch": self._epoch,
                "batch_size": self._batch_size,
                "capacity": self._capacity,
                "closed": self._closed,
                "caches": caches,
            }
        )

    def observe(self) -> MixtralStateObservation:
        with self._lock:
            return MixtralStateObservation(
                length=self._length,
                epoch=self._epoch,
                batch_size=self._batch_size,
                capacity=self._capacity,
                closed=self._closed,
                state_fingerprint=self._fingerprint_locked(),
            )

    def kv_snapshot(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        with self._lock:
            if self._closed:
                raise MixtralStatefulError("cannot inspect a closed Mixtral state")
            return tuple(
                (key.detach().clone(), value.detach().clone()) for key, value in self._caches
            )

    def fork(self) -> MixtralReferenceState:
        with self._lock:
            if self._closed:
                raise MixtralStatefulError("cannot fork a closed Mixtral state")
            forked = MixtralReferenceState(self._engine, capacity=self._capacity)
            # Committed cache tensors are immutable: later commits replace, never mutate, them.
            forked._length = self._length
            forked._epoch = self._epoch
            forked._batch_size = self._batch_size
            forked._caches = self._caches
            return forked

    def prefill(self, token_ids: torch.Tensor | Sequence[Sequence[int]]) -> MixtralProvisionalStep:
        return self._engine._propose(self, token_ids, require_empty=True)

    def decode(self, token_ids: torch.Tensor | Sequence[Sequence[int]]) -> MixtralProvisionalStep:
        return self._engine._propose(self, token_ids, require_empty=False)

    def _snapshot_for_proposal(
        self,
        *,
        require_empty: bool,
    ) -> tuple[int, int, int | None, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        with self._lock:
            if self._closed:
                raise MixtralStatefulError("cannot execute against a closed Mixtral state")
            if require_empty and self._length != 0:
                raise MixtralStatefulError("prefill requires an empty Mixtral state")
            if not require_empty and self._length == 0:
                raise MixtralStatefulError("decode requires committed Mixtral prefix state")
            return self._epoch, self._length, self._batch_size, self._caches

    def _validate_step_locked(self, step: MixtralProvisionalStep) -> None:
        if type(step.base_epoch) is not int or type(step.base_length) is not int:
            raise MixtralStatefulError("Mixtral provisional step has invalid base counters")
        if type(step.token_count) is not int or step.token_count <= 0:
            raise MixtralStatefulError("Mixtral provisional step has an invalid token count")
        if self._length + step.token_count > self._capacity:
            raise MixtralStatefulError("Mixtral provisional step exceeds state capacity")
        if not isinstance(step.logits, torch.Tensor) or step.logits.ndim != 3:
            raise MixtralStatefulError("Mixtral provisional logits have the wrong shape")
        expected_batch = (
            self._batch_size if self._batch_size is not None else int(step.logits.shape[0])
        )
        if expected_batch <= 0:
            raise MixtralStatefulError("Mixtral provisional logits have an invalid batch")
        expected_length = self._length + step.token_count
        expected_logits = (expected_batch, step.token_count, self._engine.dimensions.vocab_size)
        if tuple(step.logits.shape) != expected_logits:
            raise MixtralStatefulError("Mixtral provisional logits have the wrong shape")
        if step.logits.device.type != "cpu":
            raise MixtralStatefulError("Mixtral provisional logits left the CPU reference target")
        if not step.logits.is_floating_point():
            raise MixtralStatefulError("Mixtral provisional logits are not floating point")
        if not torch.isfinite(step.logits).all():
            raise MixtralStatefulError("Mixtral provisional logits are non-finite")
        if not isinstance(step._next_caches, tuple):
            raise MixtralStatefulError("Mixtral provisional cache inventory is not immutable")
        if len(step._next_caches) != self._engine.num_hidden_layers:
            raise MixtralStatefulError("Mixtral provisional step has incomplete layer state")
        expected_cache = (
            expected_batch,
            self._engine.dimensions.num_key_value_heads,
            expected_length,
            self._engine.dimensions.head_dim,
        )
        for cache in step._next_caches:
            if not isinstance(cache, tuple) or len(cache) != 2:
                raise MixtralStatefulError("Mixtral provisional cache entry is malformed")
            key, value = cache
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise MixtralStatefulError("Mixtral provisional cache is not tensor-backed")
            if tuple(key.shape) != expected_cache or tuple(value.shape) != expected_cache:
                raise MixtralStatefulError("Mixtral provisional cache has the wrong shape")
            if key.device.type != "cpu" or value.device.type != "cpu":
                raise MixtralStatefulError(
                    "Mixtral provisional cache left the CPU reference target"
                )
            if key.dtype != value.dtype:
                raise MixtralStatefulError("Mixtral provisional K/V dtypes disagree")
            if not key.is_floating_point() or not value.is_floating_point():
                raise MixtralStatefulError("Mixtral provisional cache is not floating point")
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise MixtralStatefulError("Mixtral provisional cache is non-finite")

    def _commit(self, step: MixtralProvisionalStep) -> MixtralStateObservation:
        with self._lock:
            if step._state is not self or step._consumed:
                raise MixtralStatefulError(
                    "Mixtral provisional step is foreign or already consumed"
                )
            if self._closed:
                raise MixtralStatefulError("cannot commit into a closed Mixtral state")
            if step.base_epoch != self._epoch or step.base_length != self._length:
                raise MixtralStatefulError("Mixtral provisional step is stale")
            self._validate_step_locked(step)
            step._consumed = True
            self._caches = step._next_caches
            self._length += step.token_count
            self._batch_size = int(step.logits.shape[0])
            self._epoch += 1
            return self.observe()

    def _rollback(self, step: MixtralProvisionalStep) -> MixtralStateObservation:
        with self._lock:
            if step._state is not self or step._consumed:
                raise MixtralStatefulError(
                    "Mixtral provisional step is foreign or already consumed"
                )
            step._consumed = True
            return self.observe()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._caches = ()
            self._batch_size = None
            self._closed = True
            self._epoch += 1


def _apply_rotary_at_offset(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    offset: int,
    head_dim: int,
    rope_theta: float,
    inv_freq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, query_width = query.shape
    if query_width % head_dim or key.shape[-1] % head_dim:
        raise MixtralStatefulError("rotary input width is not divisible by head_dim")
    query_heads = query_width // head_dim
    key_heads = key.shape[-1] // head_dim
    q = query.reshape(batch, sequence, query_heads, head_dim).transpose(1, 2)
    k = key.reshape(batch, sequence, key_heads, head_dim).transpose(1, 2)
    if inv_freq is None:
        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        frequencies = 1.0 / (rope_theta ** (indices / head_dim))
    else:
        if tuple(inv_freq.shape) != (head_dim // 2,):
            raise MixtralStatefulError("serialized rotary frequency has the wrong shape")
        frequencies = inv_freq.to(torch.float32)
    positions = torch.arange(offset, offset + sequence, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    embedding = torch.cat((angles, angles), dim=-1)
    cosine = embedding.cos().to(q.dtype)[None, None, :, :]
    sine = embedding.sin().to(q.dtype)[None, None, :, :]
    q_first, q_second = q.chunk(2, dim=-1)
    k_first, k_second = k.chunk(2, dim=-1)
    q_rotated = torch.cat((-q_second, q_first), dim=-1)
    k_rotated = torch.cat((-k_second, k_first), dim=-1)
    q = q * cosine + q_rotated * sine
    k = k * cosine + k_rotated * sine
    return q, k


def _incremental_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    past_length: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    scale: float,
) -> torch.Tensor:
    repeats = num_attention_heads // num_key_value_heads
    repeated_key = key.repeat_interleave(repeats, dim=1)
    repeated_value = value.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query, repeated_key.transpose(-2, -1)) * scale
    query_positions = torch.arange(past_length, past_length + query.shape[2], dtype=torch.int64)
    key_positions = torch.arange(key.shape[2], dtype=torch.int64)
    disallowed = key_positions[None, :] > query_positions[:, None]
    scores = scores.masked_fill(disallowed[None, None, :, :], float("-inf"))
    probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(query.dtype)
    context = torch.matmul(probabilities, repeated_value)
    return context.transpose(1, 2).reshape(
        query.shape[0], query.shape[2], num_attention_heads * query.shape[-1]
    )


class MixtralStatefulReference:
    """Incremental G10 target over a verified Mixtral reference executable."""

    def __init__(self, executable: ReferenceExecutable) -> None:
        if executable.identity.architecture_id != "mixtral-sparse-moe-causal-decoder":
            raise MixtralStatefulError("stateful Mixtral target requires a Mixtral artifact")
        self.executable = executable
        self.model = executable.artifact.ir_bundle.model
        self.dimensions = self.model.dimensions
        self.config = executable.artifact.source.config
        self.num_experts, self.top_k, self.epsilon, _ = _mixtral_config_semantics(
            self.dimensions, self.config
        )
        self.num_hidden_layers = self.dimensions.num_hidden_layers
        self.max_position_embeddings = self.dimensions.max_position_embeddings
        self._operations = {item.operation_id: item for item in self.model.operations}

    @classmethod
    def lower(
        cls, artifact: ComponentArtifact | str | Path | ReferenceExecutable
    ) -> MixtralStatefulReference:
        executable = (
            artifact
            if isinstance(artifact, ReferenceExecutable)
            else lower_component_artifact_to_reference(artifact)
        )
        return cls(executable)

    def create_state(self, *, capacity: int | None = None) -> MixtralReferenceState:
        return MixtralReferenceState(
            self, capacity=capacity if capacity is not None else self.max_position_embeddings
        )

    def _logical_tensor(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = self.executable.store.tensor(name)
        return value if dtype is None else value.to(dtype)

    @torch.inference_mode()
    def _propose(
        self,
        state: MixtralReferenceState,
        token_ids: torch.Tensor | Sequence[Sequence[int]],
        *,
        require_empty: bool,
    ) -> MixtralProvisionalStep:
        if state._engine is not self:
            raise MixtralStatefulError("cannot execute a state owned by another Mixtral target")
        raw = token_ids if isinstance(token_ids, torch.Tensor) else torch.tensor(token_ids)
        tokens = self.executable._validate_tokens(raw)
        base_epoch, base_length, batch_size, past_caches = state._snapshot_for_proposal(
            require_empty=require_empty
        )
        if batch_size is not None and batch_size != tokens.shape[0]:
            raise MixtralStatefulError("Mixtral state batch size cannot change after commit")
        if base_length + tokens.shape[1] > state.capacity:
            raise MixtralStatefulError("Mixtral step exceeds the bounded state capacity")
        if base_length and len(past_caches) != self.num_hidden_layers:
            raise MixtralStatefulError("committed Mixtral state has incomplete layer caches")

        hidden = F.embedding(tokens, self._logical_tensor("token_embedding.weight"))
        next_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        d = self.dimensions
        for layer in range(self.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            residual = hidden
            normed = _rms_norm(
                hidden, self._logical_tensor(f"{prefix}.attention_norm.weight"), self.epsilon
            )
            query = F.linear(
                normed, self._logical_tensor(f"{attention}.q_proj.weight", normed.dtype)
            )
            key = F.linear(normed, self._logical_tensor(f"{attention}.k_proj.weight", normed.dtype))
            value = F.linear(
                normed, self._logical_tensor(f"{attention}.v_proj.weight", normed.dtype)
            )
            rotary = self._operations[f"{attention}.rotary"]
            rotary_parameters = {name: self._logical_tensor(name) for name in rotary.parameters}
            query_heads, key_heads = _apply_rotary_at_offset(
                query,
                key,
                offset=base_length,
                head_dim=d.head_dim,
                rope_theta=float(rotary.attributes["rope_theta"]),
                inv_freq=_rotary_parameter(rotary_parameters),
            )
            value_heads = value.reshape(
                tokens.shape[0], tokens.shape[1], d.num_key_value_heads, d.head_dim
            ).transpose(1, 2)
            if past_caches:
                past_key, past_value = past_caches[layer]
                all_key = torch.cat((past_key, key_heads), dim=2)
                all_value = torch.cat((past_value, value_heads), dim=2)
            else:
                all_key = key_heads
                all_value = value_heads
            context = _incremental_gqa(
                query_heads,
                all_key,
                all_value,
                past_length=base_length,
                num_attention_heads=d.num_attention_heads,
                num_key_value_heads=d.num_key_value_heads,
                scale=d.head_dim**-0.5,
            )
            hidden = residual + F.linear(
                context, self._logical_tensor(f"{attention}.o_proj.weight", context.dtype)
            )
            next_caches.append((all_key, all_value))

            residual = hidden
            moe = f"{prefix}.moe"
            routed_input = _rms_norm(
                hidden, self._logical_tensor(f"{prefix}.moe_norm.weight"), self.epsilon
            )
            router_logits = F.linear(
                routed_input, self._logical_tensor(f"{moe}.router.weight", routed_input.dtype)
            )
            routing_weights, selected_experts = _moe_top_k(
                router_logits,
                num_experts_per_token=self.top_k,
                num_routed_experts=self.num_experts,
            )
            dispatched = _moe_dispatch(
                routed_input,
                selected_experts,
                num_routed_experts=self.num_experts,
            )
            expert_outputs: list[torch.Tensor] = []
            for expert, expert_input in enumerate(dispatched):
                expert_prefix = f"{moe}.routed_experts.{expert}"
                gate = F.linear(
                    expert_input,
                    self._logical_tensor(f"{expert_prefix}.gate_proj.weight", expert_input.dtype),
                )
                up = F.linear(
                    expert_input,
                    self._logical_tensor(f"{expert_prefix}.up_proj.weight", expert_input.dtype),
                )
                expert_outputs.append(
                    F.linear(
                        F.silu(gate) * up,
                        self._logical_tensor(
                            f"{expert_prefix}.down_proj.weight", expert_input.dtype
                        ),
                    )
                )
            moe_output = _moe_weighted_scatter_add(
                routing_weights,
                selected_experts,
                expert_outputs,
                num_routed_experts=self.num_experts,
            )
            hidden = residual + moe_output
        hidden = _rms_norm(hidden, self._logical_tensor("final_norm.weight"), self.epsilon)
        logits = F.linear(hidden, self._logical_tensor("lm_head.weight", hidden.dtype))
        logits = logits[..., : d.vocab_size]
        return MixtralProvisionalStep(
            _state=state,
            base_epoch=base_epoch,
            base_length=base_length,
            token_count=int(tokens.shape[1]),
            logits=logits,
            _next_caches=tuple(next_caches),
        )


@dataclass(frozen=True, slots=True)
class MixtralStatefulCertification:
    artifact_id: str
    target_fingerprint: str
    case_input_fingerprints: tuple[str, ...]
    split_patterns: tuple[tuple[int, ...], ...]
    stateless_logits_fingerprints: tuple[str, ...]
    stateful_logits_fingerprints: tuple[str, ...]
    final_state_fingerprints: tuple[str, ...]
    maximum_absolute_error: float
    maximum_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    checks: tuple[str, ...]
    fingerprint: str
    schema_version: str = MIXTRAL_STATEFUL_CERTIFICATION_SCHEMA
    status: str = "passed"
    gate: str = "G10-stateful-prefill-decode-parity"
    execution_certified: bool = True
    production_runtime_eligible: bool = False

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "gate": self.gate,
            "artifact_id": self.artifact_id,
            "target_fingerprint": self.target_fingerprint,
            "case_input_fingerprints": list(self.case_input_fingerprints),
            "split_patterns": [list(value) for value in self.split_patterns],
            "stateless_logits_fingerprints": list(self.stateless_logits_fingerprints),
            "stateful_logits_fingerprints": list(self.stateful_logits_fingerprints),
            "final_state_fingerprints": list(self.final_state_fingerprints),
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "checks": list(self.checks),
            "execution_certified": self.execution_certified,
            "production_runtime_eligible": self.production_runtime_eligible,
        }

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.artifact_id, "artifact_id"),
            (self.target_fingerprint, "target_fingerprint"),
            (self.fingerprint, "Mixtral stateful certification fingerprint"),
            *((value, "case fingerprint") for value in self.case_input_fingerprints),
            *((value, "stateless fingerprint") for value in self.stateless_logits_fingerprints),
            *((value, "stateful fingerprint") for value in self.stateful_logits_fingerprints),
            *((value, "state fingerprint") for value in self.final_state_fingerprints),
        ):
            require_sha256(value, field=field_name)
        count = len(self.case_input_fingerprints)
        if not count or any(
            len(values) != count
            for values in (
                self.split_patterns,
                self.stateless_logits_fingerprints,
                self.stateful_logits_fingerprints,
                self.final_state_fingerprints,
            )
        ):
            raise ValueError("Mixtral stateful certification arrays are inconsistent")
        if (
            self.schema_version != MIXTRAL_STATEFUL_CERTIFICATION_SCHEMA
            or self.status != "passed"
            or self.gate != "G10-stateful-prefill-decode-parity"
            or not self.execution_certified
            or self.production_runtime_eligible
        ):
            raise ValueError("unsupported Mixtral stateful certification boundary")
        if self.checks != tuple(sorted(set(self.checks))):
            raise ValueError("Mixtral stateful checks must be sorted and unique")
        for value in (
            self.maximum_absolute_error,
            self.maximum_relative_error,
            self.absolute_tolerance,
            self.relative_tolerance,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Mixtral stateful numerical bounds must be finite and non-negative"
                )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("Mixtral stateful certification fingerprint does not match")

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}


def run_g10_mixtral_stateful_parity(
    artifact: ComponentArtifact | str | Path | ReferenceExecutable,
    cases: Sequence[tuple[Sequence[Sequence[int]] | torch.Tensor, Sequence[int]]],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> MixtralStatefulCertification:
    """Run segmented transactional execution against full-sequence IR and full K/V traces."""

    if (
        not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
        or not math.isfinite(relative_tolerance)
        or relative_tolerance < 0
    ):
        raise ValueError("G10 tolerances must be finite and non-negative")
    engine = MixtralStatefulReference.lower(artifact)
    executable = engine.executable
    input_fingerprints: list[str] = []
    split_patterns: list[tuple[int, ...]] = []
    stateless_fingerprints: list[str] = []
    stateful_fingerprints: list[str] = []
    state_fingerprints: list[str] = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    saw_segmented_case = False
    if not cases:
        raise MixtralStatefulParityError("G10 certification requires at least one segmented case")
    for case_index, (raw_tokens, raw_splits) in enumerate(cases):
        raw = raw_tokens if isinstance(raw_tokens, torch.Tensor) else torch.tensor(raw_tokens)
        tokens = executable._validate_tokens(raw)
        splits = tuple(raw_splits)
        if (
            not splits
            or any(type(value) is not int or value <= 0 for value in splits)
            or sum(splits) != tokens.shape[1]
        ):
            raise MixtralStatefulParityError(
                "G10 split pattern must contain positive widths summing to the sequence",
                details={"case_index": case_index},
            )
        saw_segmented_case = saw_segmented_case or len(splits) > 1
        state = engine.create_state(capacity=int(tokens.shape[1]))
        before = state.observe()
        first_probe = state.prefill(tokens[:, : splits[0]])
        probe_logits = first_probe.logits.detach().clone()
        rolled_back = first_probe.rollback()
        if rolled_back != before or state.observe() != before:
            raise MixtralStatefulParityError(
                "G10 rollback changed committed state", details={"case_index": case_index}
            )
        steps: list[torch.Tensor] = []
        offset = 0
        for split_index, width in enumerate(splits):
            chunk = tokens[:, offset : offset + width]
            proposal = state.prefill(chunk) if split_index == 0 else state.decode(chunk)
            stale_proposal = state.prefill(chunk) if split_index == 0 else None
            if split_index == 0 and not torch.equal(proposal.logits, probe_logits):
                raise MixtralStatefulParityError(
                    "G10 replay after rollback changed provisional logits",
                    details={"case_index": case_index},
                )
            if stale_proposal is not None and not torch.equal(
                stale_proposal.logits, proposal.logits
            ):
                raise MixtralStatefulParityError(
                    "G10 concurrent proposals changed provisional logits",
                    details={"case_index": case_index},
                )
            observation_before_commit = state.observe()
            steps.append(proposal.logits.detach().clone())
            committed = proposal.commit()
            if committed.length != offset + width or committed.epoch != split_index + 1:
                raise MixtralStatefulParityError(
                    "G10 commit counters are inconsistent", details={"case_index": case_index}
                )
            if observation_before_commit.length != offset:
                raise MixtralStatefulParityError(
                    "G10 provisional execution mutated state before commit",
                    details={"case_index": case_index},
                )
            if split_index == 0:
                try:
                    proposal.commit()
                except MixtralStatefulError as error:
                    if "already consumed" not in str(error):
                        raise MixtralStatefulParityError(
                            "G10 double-commit rejection had the wrong cause",
                            details={"case_index": case_index},
                        ) from error
                else:
                    raise MixtralStatefulParityError(
                        "G10 accepted a double commit",
                        details={"case_index": case_index},
                    )
                assert stale_proposal is not None
                try:
                    stale_proposal.commit()
                except MixtralStatefulError as error:
                    if "stale" not in str(error):
                        raise MixtralStatefulParityError(
                            "G10 stale-commit rejection had the wrong cause",
                            details={"case_index": case_index},
                        ) from error
                else:
                    raise MixtralStatefulParityError(
                        "G10 accepted an epoch-stale commit",
                        details={"case_index": case_index},
                    )
                stale_proposal.rollback()
                try:
                    stale_proposal.rollback()
                except MixtralStatefulError as error:
                    if "already consumed" not in str(error):
                        raise MixtralStatefulParityError(
                            "G10 double-rollback rejection had the wrong cause",
                            details={"case_index": case_index},
                        ) from error
                else:
                    raise MixtralStatefulParityError(
                        "G10 accepted a double rollback",
                        details={"case_index": case_index},
                    )
                batch_probe = torch.zeros((tokens.shape[0] + 1, 1), dtype=torch.int64)
                before_batch_probe = state.observe()
                try:
                    state.decode(batch_probe)
                except MixtralStatefulError as error:
                    if "batch size cannot change" not in str(error):
                        raise MixtralStatefulParityError(
                            "G10 batch-shape rejection had the wrong cause",
                            details={"case_index": case_index},
                        ) from error
                else:
                    raise MixtralStatefulParityError(
                        "G10 accepted a batch-shape change",
                        details={"case_index": case_index},
                    )
                if state.observe() != before_batch_probe:
                    raise MixtralStatefulParityError(
                        "G10 rejected batch probe changed committed state",
                        details={"case_index": case_index},
                    )
            offset += width
        stateful_logits = torch.cat(steps, dim=1)
        stateless = executable.forward(tokens).logits
        difference = (stateless.to(torch.float32) - stateful_logits.to(torch.float32)).abs()
        absolute = float(difference.max().item()) if difference.numel() else 0.0
        denominator = stateless.to(torch.float32).abs().clamp_min(torch.finfo(torch.float32).tiny)
        relative = float((difference / denominator).max().item()) if difference.numel() else 0.0
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_relative_error = max(maximum_relative_error, relative)
        if not torch.allclose(
            stateless.to(torch.float32),
            stateful_logits.to(torch.float32),
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
            raise MixtralStatefulParityError(
                "segmented stateful logits differ from full-sequence IR",
                details={
                    "case_index": case_index,
                    "maximum_absolute_error": absolute,
                    "maximum_relative_error": relative,
                },
            )
        cache_names = [
            name
            for layer in range(engine.num_hidden_layers)
            for name in (
                f"layers.{layer}.attention.k_rotary",
                f"layers.{layer}.attention.v",
            )
        ]
        full_trace = executable.trace(tokens, cache_names)
        snapshots = state.kv_snapshot()
        for layer, (state_key, state_value) in enumerate(snapshots):
            expected_key = (
                full_trace[f"layers.{layer}.attention.k_rotary"]
                .reshape(
                    tokens.shape[0],
                    tokens.shape[1],
                    engine.dimensions.num_key_value_heads,
                    engine.dimensions.head_dim,
                )
                .transpose(1, 2)
            )
            expected_value = (
                full_trace[f"layers.{layer}.attention.v"]
                .reshape(
                    tokens.shape[0],
                    tokens.shape[1],
                    engine.dimensions.num_key_value_heads,
                    engine.dimensions.head_dim,
                )
                .transpose(1, 2)
            )
            for kind, expected, observed in (
                ("key", expected_key, state_key),
                ("value", expected_value, state_value),
            ):
                cache_difference = (expected.to(torch.float32) - observed.to(torch.float32)).abs()
                cache_absolute = float(cache_difference.max().item())
                cache_denominator = (
                    expected.to(torch.float32).abs().clamp_min(torch.finfo(torch.float32).tiny)
                )
                cache_relative = float((cache_difference / cache_denominator).max().item())
                maximum_absolute_error = max(maximum_absolute_error, cache_absolute)
                maximum_relative_error = max(maximum_relative_error, cache_relative)
                if not torch.allclose(
                    expected.to(torch.float32),
                    observed.to(torch.float32),
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                ):
                    raise MixtralStatefulParityError(
                        "segmented committed K/V differs from full-sequence IR trace",
                        details={
                            "case_index": case_index,
                            "layer": layer,
                            "cache": kind,
                            "maximum_absolute_error": cache_absolute,
                            "maximum_relative_error": cache_relative,
                        },
                    )
        capacity_probe = torch.zeros((tokens.shape[0], 1), dtype=torch.int64)
        before_capacity_probe = state.observe()
        try:
            state.decode(capacity_probe)
        except MixtralStatefulError as error:
            if "capacity" not in str(error):
                raise MixtralStatefulParityError(
                    "G10 capacity rejection had the wrong cause",
                    details={"case_index": case_index},
                ) from error
        else:
            raise MixtralStatefulParityError(
                "G10 accepted execution beyond state capacity",
                details={"case_index": case_index},
            )
        if state.observe() != before_capacity_probe:
            raise MixtralStatefulParityError(
                "G10 rejected capacity probe changed committed state",
                details={"case_index": case_index},
            )
        forked = state.fork()
        if forked.observe().state_fingerprint != state.observe().state_fingerprint:
            raise MixtralStatefulParityError(
                "G10 state fork changed the committed snapshot",
                details={"case_index": case_index},
            )
        input_fingerprints.append(
            canonical_sha256(
                {"dtype": "int64", "shape": list(tokens.shape), "values": tokens.tolist()}
            )
        )
        split_patterns.append(splits)
        stateless_fingerprints.append(_tensor_fingerprint(stateless))
        stateful_fingerprints.append(_tensor_fingerprint(stateful_logits))
        final_observation = state.observe()
        state_fingerprints.append(final_observation.state_fingerprint)
        forked.close()
        if state.observe() != final_observation:
            raise MixtralStatefulParityError(
                "closing a fork changed its source state",
                details={"case_index": case_index},
            )
        state.close()
        closed = state.observe()
        if (
            not closed.closed
            or closed.length != final_observation.length
            or closed.epoch != final_observation.epoch + 1
        ):
            raise MixtralStatefulParityError(
                "G10 close counters are inconsistent",
                details={"case_index": case_index},
            )
        state.close()
        if state.observe() != closed:
            raise MixtralStatefulParityError(
                "G10 close is not idempotent",
                details={"case_index": case_index},
            )
        try:
            state.kv_snapshot()
        except MixtralStatefulError:
            pass
        else:
            raise MixtralStatefulParityError(
                "G10 exposed K/V after close",
                details={"case_index": case_index},
            )
    if not saw_segmented_case:
        raise MixtralStatefulParityError(
            "G10 certification requires at least one segmented decode case"
        )
    checks = tuple(
        sorted(
            {
                "bounded-context-capacity",
                "committed-kv-vs-full-ir-trace",
                "epoch-bound-single-use-commit",
                "fixed-batch-shape",
                "forked-committed-snapshot",
                "idempotent-close-and-release",
                "prefill-and-segmented-decode",
                "provisional-state-isolation",
                "rollback-and-replay",
                "stateless-vs-stateful-logits",
            }
        )
    )
    payload = {
        "schema_version": MIXTRAL_STATEFUL_CERTIFICATION_SCHEMA,
        "status": "passed",
        "gate": "G10-stateful-prefill-decode-parity",
        "artifact_id": executable.identity.artifact_id,
        "target_fingerprint": executable.identity.fingerprint,
        "case_input_fingerprints": input_fingerprints,
        "split_patterns": [list(value) for value in split_patterns],
        "stateless_logits_fingerprints": stateless_fingerprints,
        "stateful_logits_fingerprints": stateful_fingerprints,
        "final_state_fingerprints": state_fingerprints,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "checks": list(checks),
        "execution_certified": True,
        "production_runtime_eligible": False,
    }
    return MixtralStatefulCertification(
        artifact_id=executable.identity.artifact_id,
        target_fingerprint=executable.identity.fingerprint,
        case_input_fingerprints=tuple(input_fingerprints),
        split_patterns=tuple(split_patterns),
        stateless_logits_fingerprints=tuple(stateless_fingerprints),
        stateful_logits_fingerprints=tuple(stateful_fingerprints),
        final_state_fingerprints=tuple(state_fingerprints),
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
        checks=checks,
        fingerprint=canonical_sha256(payload),
    )


__all__ = [
    "MIXTRAL_STATEFUL_CERTIFICATION_SCHEMA",
    "MixtralProvisionalStep",
    "MixtralReferenceState",
    "MixtralStateObservation",
    "MixtralStatefulCertification",
    "MixtralStatefulError",
    "MixtralStatefulParityError",
    "MixtralStatefulReference",
    "run_g10_mixtral_stateful_parity",
]
