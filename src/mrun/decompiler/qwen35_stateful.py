"""Transactional stateful reference runtime for compiled Qwen3.5 text artifacts.

The runtime is intentionally small and correctness-first.  A stage operation always starts from
one immutable committed parent, records copy-on-write snapshots for every proposed suffix token,
and never mutates committed KV, convolution, or recurrent state.  Commit is epoch-bound and may
accept any prefix; rollback merely closes the proposal.  Same-parent branch proposals therefore
remain comparable, while a second commit fails stale before it can write.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ._json import canonical_sha256
from .reference import (
    ReferenceExecutable,
    ReferenceExecutionError,
    _qwen35_gated_delta_recurrence,
    _qwen35_head_rms_norm,
    _qwen35_partial_mrope,
    _qwen35_query_gate_split,
    _qwen35_rms_norm,
    _qwen35_rms_norm_gated,
    lower_component_artifact_to_reference,
)

QWEN35_STATEFUL_RUNTIME_ABI = "mrun-qwen35-transactional-hybrid-reference-v1"


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(contiguous.view(torch.uint8).numpy())).hexdigest()


@dataclass(slots=True)
class _HybridState:
    epoch: int
    length: int
    batch: int
    capacity: int
    conv: dict[int, torch.Tensor]
    recurrent: dict[int, torch.Tensor]
    key: dict[int, torch.Tensor]
    value: dict[int, torch.Tensor]

    def clone(self) -> _HybridState:
        return _HybridState(
            epoch=self.epoch,
            length=self.length,
            batch=self.batch,
            capacity=self.capacity,
            conv={layer: value.clone() for layer, value in self.conv.items()},
            recurrent={layer: value.clone() for layer, value in self.recurrent.items()},
            key={layer: value.clone() for layer, value in self.key.items()},
            value={layer: value.clone() for layer, value in self.value.items()},
        )

    def fingerprint(self) -> str:
        records: list[dict[str, Any]] = []
        for kind, values in (
            ("conv", self.conv),
            ("recurrent", self.recurrent),
            ("key", self.key),
            ("value", self.value),
        ):
            for layer, tensor in sorted(values.items()):
                records.append(
                    {
                        "kind": kind,
                        "layer": layer,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "sha256": _tensor_sha256(tensor),
                    }
                )
        return canonical_sha256(
            {
                "abi": QWEN35_STATEFUL_RUNTIME_ABI,
                "epoch": self.epoch,
                "length": self.length,
                "batch": self.batch,
                "capacity": self.capacity,
                "slots": records,
            }
        )


@dataclass(frozen=True, slots=True)
class Qwen35StateSnapshot:
    epoch: int
    length: int
    batch: int
    capacity: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Qwen35StatefulParityReceipt:
    """Bounded G10-style evidence, separate from stateless source/IR G8 self-consistency."""

    artifact_id: str
    case_fingerprint: str
    stateless_logits_sha256: str
    stateful_logits_sha256: str
    maximum_absolute_error: float
    maximum_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    bit_exact: bool
    certified: bool
    fingerprint: str
    schema_version: str = "mrun-qwen35-stateful-two-token-parity-v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_abi": QWEN35_STATEFUL_RUNTIME_ABI,
            "artifact_id": self.artifact_id,
            "case_fingerprint": self.case_fingerprint,
            "stateless_logits_sha256": self.stateless_logits_sha256,
            "stateful_logits_sha256": self.stateful_logits_sha256,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "bit_exact": self.bit_exact,
            "certified": self.certified,
            "fingerprint": self.fingerprint,
        }


@dataclass(slots=True)
class Qwen35ProvisionalSuffix:
    """One epoch-bound proposal; state snapshots remain runtime-private."""

    parent_epoch: int
    parent_fingerprint: str
    suffix_length: int
    logits: torch.Tensor
    _runtime_token: object
    _parent: _HybridState
    _snapshots: tuple[_HybridState, ...]
    _closed: bool = False


class Qwen35TransactionalReference:
    """Stateful prefill/decode target with atomic hybrid-state commit and rollback."""

    runtime_abi = QWEN35_STATEFUL_RUNTIME_ABI

    def __init__(
        self,
        artifact: Any,
        *,
        batch: int,
        capacity: int,
    ) -> None:
        executable = (
            artifact
            if isinstance(artifact, ReferenceExecutable)
            else lower_component_artifact_to_reference(artifact)
        )
        if executable.identity.architecture_id != "qwen3_5-hybrid-text-causal-decoder":
            raise ReferenceExecutionError(
                "Qwen3.5 stateful runtime requires a hybrid text artifact"
            )
        if type(batch) is not int or batch <= 0:
            raise ValueError("batch must be a positive integer")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.executable = executable
        self.store = executable.store
        self.config = executable.artifact.source.config["text_config"]
        self.dimensions = executable.artifact.ir_bundle.model.dimensions
        if capacity > self.dimensions.max_position_embeddings:
            raise ValueError("capacity exceeds the compiled Qwen3.5 position contract")
        self._runtime_token = object()
        self._lock = threading.RLock()
        self._state = self._initialize(batch=batch, capacity=capacity)

    def _tensor(self, logical_name: str) -> torch.Tensor:
        return self.store.tensor(logical_name)

    def _linear(self, value: torch.Tensor, logical_prefix: str) -> torch.Tensor:
        weight = self._tensor(f"{logical_prefix}.weight").to(value.dtype)
        return F.linear(value, weight)

    def _initialize(self, *, batch: int, capacity: int) -> _HybridState:
        dtype = self._tensor("token_embedding.weight").dtype
        conv: dict[int, torch.Tensor] = {}
        recurrent: dict[int, torch.Tensor] = {}
        key: dict[int, torch.Tensor] = {}
        value: dict[int, torch.Tensor] = {}
        key_heads = int(self.config["linear_num_key_heads"])
        value_heads = int(self.config["linear_num_value_heads"])
        key_dim = int(self.config["linear_key_head_dim"])
        value_dim = int(self.config["linear_value_head_dim"])
        conv_width = 2 * key_heads * key_dim + value_heads * value_dim
        conv_kernel = int(self.config["linear_conv_kernel_dim"])
        for layer, layer_type in enumerate(self.config["layer_types"]):
            if layer_type == "linear_attention":
                conv[layer] = torch.zeros((batch, conv_width, conv_kernel), dtype=dtype)
                recurrent[layer] = torch.zeros(
                    (batch, value_heads, key_dim, value_dim), dtype=torch.float32
                )
            elif layer_type == "full_attention":
                shape = (
                    batch,
                    0,
                    self.dimensions.num_key_value_heads,
                    self.dimensions.head_dim,
                )
                key[layer] = torch.empty(shape, dtype=dtype)
                value[layer] = torch.empty(shape, dtype=dtype)
            else:
                raise ReferenceExecutionError("Qwen3.5 compiled layer schedule is invalid")
        return _HybridState(
            epoch=0,
            length=0,
            batch=batch,
            capacity=capacity,
            conv=conv,
            recurrent=recurrent,
            key=key,
            value=value,
        )

    def snapshot(self) -> Qwen35StateSnapshot:
        with self._lock:
            state = self._state
            return Qwen35StateSnapshot(
                epoch=state.epoch,
                length=state.length,
                batch=state.batch,
                capacity=state.capacity,
                fingerprint=state.fingerprint(),
            )

    def _validate_suffix(self, token_ids: Any, state: _HybridState) -> torch.Tensor:
        value = token_ids if isinstance(token_ids, torch.Tensor) else torch.tensor(token_ids)
        value = value.detach().to(device="cpu")
        if value.ndim == 1:
            if state.batch != 1:
                raise ReferenceExecutionError("rank-1 suffix is valid only for batch one")
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[0] != state.batch or value.shape[1] <= 0:
            raise ReferenceExecutionError(
                "suffix must have shape [committed_batch, positive_tokens]"
            )
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise ReferenceExecutionError("suffix token IDs must use an integer dtype")
        value = value.to(torch.int64).contiguous()
        if int(value.min()) < 0 or int(value.max()) >= self.dimensions.vocab_size:
            raise ReferenceExecutionError("suffix token is outside the compiled token domain")
        if state.length + value.shape[1] > state.capacity:
            raise ReferenceExecutionError("suffix exceeds the bound hybrid-state capacity")
        return value

    def _attention_one(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        d = self.dimensions
        q = query.reshape(query.shape[0], d.num_attention_heads, 1, d.head_dim)
        k = key_cache.transpose(1, 2)
        v = value_cache.transpose(1, 2)
        repeats = d.num_attention_heads // d.num_key_value_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (d.head_dim**-0.5)
        probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
        context = torch.matmul(probabilities, v)
        return context.transpose(1, 2).reshape(query.shape[0], -1)

    def _forward_token(
        self, token_ids: torch.Tensor, parent: _HybridState
    ) -> tuple[torch.Tensor, _HybridState]:
        if token_ids.shape != (parent.batch,):
            raise ReferenceExecutionError("stateful Qwen3.5 token step has invalid batch shape")
        next_state = parent.clone()
        hidden = F.embedding(token_ids, self._tensor("token_embedding.weight"))
        epsilon = float(self.config["rms_norm_eps"])
        key_heads = int(self.config["linear_num_key_heads"])
        value_heads = int(self.config["linear_num_value_heads"])
        key_dim = int(self.config["linear_key_head_dim"])
        value_dim = int(self.config["linear_value_head_dim"])
        rope = self.config["rope_parameters"]
        rotary_dim = int(self.dimensions.head_dim * float(rope["partial_rotary_factor"]))
        for layer, layer_type in enumerate(self.config["layer_types"]):
            prefix = f"layers.{layer}"
            residual = hidden
            normed = _qwen35_rms_norm(hidden, self._tensor(f"{prefix}.mixer_norm.weight"), epsilon)
            if layer_type == "linear_attention":
                mixer = f"{prefix}.gated_delta"
                packed = self._linear(normed, f"{mixer}.in_proj_qkv")
                old_conv = parent.conv[layer]
                new_conv = torch.cat((old_conv[:, :, 1:], packed.unsqueeze(-1)), dim=-1)
                weight = self._tensor(f"{mixer}.conv1d.weight").to(packed.dtype)
                convolved = F.silu(F.conv1d(new_conv, weight, groups=new_conv.shape[1]).squeeze(-1))
                next_state.conv[layer] = new_conv
                key_width = key_heads * key_dim
                value_width = value_heads * value_dim
                query, key, value = convolved.split((key_width, key_width, value_width), dim=-1)
                z = self._linear(normed, f"{mixer}.in_proj_z")
                a = self._linear(normed, f"{mixer}.in_proj_a")
                b = self._linear(normed, f"{mixer}.in_proj_b")
                core, recurrent = _qwen35_gated_delta_recurrence(
                    query[:, None],
                    key[:, None],
                    value[:, None],
                    a[:, None],
                    b[:, None],
                    a_log=self._tensor(f"{mixer}.A_log"),
                    dt_bias=self._tensor(f"{mixer}.dt_bias"),
                    num_key_heads=key_heads,
                    num_value_heads=value_heads,
                    key_head_dim=key_dim,
                    value_head_dim=value_dim,
                    initial_state=parent.recurrent[layer],
                )
                next_state.recurrent[layer] = recurrent
                core = _qwen35_rms_norm_gated(
                    core.squeeze(1),
                    z,
                    self._tensor(f"{mixer}.norm.weight"),
                    epsilon=epsilon,
                    head_dim=value_dim,
                )
                mixer_output = self._linear(core, f"{mixer}.out_proj")
            else:
                attention = f"{prefix}.attention"
                packed_query = self._linear(normed, f"{attention}.q_proj")
                query, gate = _qwen35_query_gate_split(
                    packed_query,
                    num_attention_heads=self.dimensions.num_attention_heads,
                    head_dim=self.dimensions.head_dim,
                )
                key = self._linear(normed, f"{attention}.k_proj")
                value = self._linear(normed, f"{attention}.v_proj")
                query = _qwen35_head_rms_norm(
                    query,
                    self._tensor(f"{attention}.q_norm.weight"),
                    epsilon,
                    self.dimensions.head_dim,
                )
                key = _qwen35_head_rms_norm(
                    key,
                    self._tensor(f"{attention}.k_norm.weight"),
                    epsilon,
                    self.dimensions.head_dim,
                )
                query, key = _qwen35_partial_mrope(
                    query[:, None],
                    key[:, None],
                    head_dim=self.dimensions.head_dim,
                    rotary_dim=rotary_dim,
                    rope_theta=float(rope["rope_theta"]),
                    position_offset=parent.length,
                )
                query = query.squeeze(1)
                key = key.squeeze(1).reshape(
                    parent.batch,
                    self.dimensions.num_key_value_heads,
                    self.dimensions.head_dim,
                )
                value = value.reshape(
                    parent.batch,
                    self.dimensions.num_key_value_heads,
                    self.dimensions.head_dim,
                )
                key_cache = torch.cat((parent.key[layer], key[:, None]), dim=1)
                value_cache = torch.cat((parent.value[layer], value[:, None]), dim=1)
                next_state.key[layer] = key_cache
                next_state.value[layer] = value_cache
                context = self._attention_one(query, key_cache, value_cache)
                mixer_output = self._linear(context * torch.sigmoid(gate), f"{attention}.o_proj")
            hidden = residual + mixer_output
            residual = hidden
            normed = _qwen35_rms_norm(hidden, self._tensor(f"{prefix}.mlp_norm.weight"), epsilon)
            gate = self._linear(normed, f"{prefix}.mlp.gate_proj")
            up = self._linear(normed, f"{prefix}.mlp.up_proj")
            hidden = residual + self._linear(F.silu(gate) * up, f"{prefix}.mlp.down_proj")
        hidden = _qwen35_rms_norm(hidden, self._tensor("final_norm.weight"), epsilon)
        logits = F.linear(hidden, self._tensor("lm_head.weight").to(hidden.dtype))
        next_state.length = parent.length + 1
        return logits[..., : self.dimensions.vocab_size], next_state

    @torch.inference_mode()
    def stage(self, token_ids: Any) -> Qwen35ProvisionalSuffix:
        with self._lock:
            parent = self._state.clone()
        suffix = self._validate_suffix(token_ids, parent)
        snapshots: list[_HybridState] = []
        logits: list[torch.Tensor] = []
        state = parent
        for position in range(suffix.shape[1]):
            row, state = self._forward_token(suffix[:, position], state)
            snapshots.append(state.clone())
            logits.append(row)
        return Qwen35ProvisionalSuffix(
            parent_epoch=parent.epoch,
            parent_fingerprint=parent.fingerprint(),
            suffix_length=suffix.shape[1],
            logits=torch.stack(logits, dim=1),
            _runtime_token=self._runtime_token,
            _parent=parent,
            _snapshots=tuple(snapshots),
        )

    def stage_branches(self, suffixes: Sequence[Any]) -> tuple[Qwen35ProvisionalSuffix, ...]:
        if not suffixes:
            raise ValueError("same-parent branch expansion requires at least one suffix")
        proposals = tuple(self.stage(suffix) for suffix in suffixes)
        parents = {(item.parent_epoch, item.parent_fingerprint) for item in proposals}
        if len(parents) != 1:
            raise ReferenceExecutionError("branch proposals did not share one immutable parent")
        return proposals

    def commit(
        self, proposal: Qwen35ProvisionalSuffix, *, accepted_tokens: int | None = None
    ) -> Qwen35StateSnapshot:
        if not isinstance(proposal, Qwen35ProvisionalSuffix):
            raise TypeError("commit requires a Qwen3.5 provisional suffix")
        accepted = proposal.suffix_length if accepted_tokens is None else accepted_tokens
        if type(accepted) is not int or not 0 <= accepted <= proposal.suffix_length:
            raise ValueError("accepted_tokens must select a prefix of the staged suffix")
        with self._lock:
            if proposal._runtime_token is not self._runtime_token:
                raise ReferenceExecutionError("proposal belongs to another Qwen3.5 runtime")
            if proposal._closed:
                raise ReferenceExecutionError("proposal was already committed or rolled back")
            if (
                self._state.epoch != proposal.parent_epoch
                or self._state.fingerprint() != proposal.parent_fingerprint
            ):
                raise ReferenceExecutionError("stale Qwen3.5 proposal rejected before write")
            selected = (
                proposal._parent.clone()
                if accepted == 0
                else proposal._snapshots[accepted - 1].clone()
            )
            selected.epoch = self._state.epoch + 1
            self._state = selected
            proposal._closed = True
            return self.snapshot()

    def rollback(self, proposal: Qwen35ProvisionalSuffix) -> Qwen35StateSnapshot:
        if not isinstance(proposal, Qwen35ProvisionalSuffix):
            raise TypeError("rollback requires a Qwen3.5 provisional suffix")
        with self._lock:
            if proposal._runtime_token is not self._runtime_token:
                raise ReferenceExecutionError("proposal belongs to another Qwen3.5 runtime")
            if proposal._closed:
                raise ReferenceExecutionError("proposal was already committed or rolled back")
            proposal._closed = True
            return self.snapshot()

    def prefill(self, token_ids: Any) -> torch.Tensor:
        proposal = self.stage(token_ids)
        self.commit(proposal)
        return proposal.logits


@torch.inference_mode()
def certify_qwen35_stateful_two_token_parity(
    artifact: Any,
    prompt: Sequence[Sequence[int]] | torch.Tensor,
    first_token: Sequence[Sequence[int]] | torch.Tensor,
    second_token: Sequence[Sequence[int]] | torch.Tensor,
) -> Qwen35StatefulParityReceipt:
    """Compare full stateless prefill with prefill + two committed decode transitions.

    CPU GEMM/conv kernels may choose a different reduction geometry for ``[B,T,*]`` and
    ``[B,*]`` operands.  The receipt therefore records bit-exactness independently and applies a
    dtype-derived tolerance of eight machine epsilons, scaled by the observed reference magnitude.
    This is not an HF parity claim; it is the compiled artifact's G10-style state transition check.
    """

    executable = (
        artifact
        if isinstance(artifact, ReferenceExecutable)
        else lower_component_artifact_to_reference(artifact)
    )

    def normalized(value: Sequence[Sequence[int]] | torch.Tensor) -> torch.Tensor:
        result = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2:
            raise ValueError("parity token inputs must be rank two")
        return result.to(torch.int64)

    prompt_ids = normalized(prompt)
    first_ids = normalized(first_token)
    second_ids = normalized(second_token)
    if first_ids.shape != (prompt_ids.shape[0], 1) or second_ids.shape != first_ids.shape:
        raise ValueError("stateful parity requires one first and one second token per batch row")
    full_ids = torch.cat((prompt_ids, first_ids, second_ids), dim=1)
    stateless = executable.forward(full_ids).logits
    runtime = Qwen35TransactionalReference(
        executable, batch=prompt_ids.shape[0], capacity=full_ids.shape[1]
    )
    prefill_logits = runtime.prefill(prompt_ids)
    first = runtime.stage(first_ids)
    runtime.commit(first)
    second = runtime.stage(second_ids)
    runtime.commit(second)
    stateful = torch.cat((prefill_logits, first.logits, second.logits), dim=1)
    difference = (stateless.to(torch.float32) - stateful.to(torch.float32)).abs()
    maximum_absolute_error = float(difference.max().item())
    denominator = stateless.to(torch.float32).abs().clamp_min(torch.finfo(torch.float32).tiny)
    maximum_relative_error = float((difference / denominator).max().item())
    epsilon = torch.finfo(stateless.dtype).eps
    scale = max(1.0, float(stateless.to(torch.float32).abs().max().item()))
    absolute_tolerance = float(8.0 * epsilon * scale)
    relative_tolerance = float(8.0 * epsilon)
    bit_exact = bool(torch.equal(stateless, stateful))
    certified = bool(
        torch.allclose(
            stateless,
            stateful,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            equal_nan=False,
        )
    )
    case_fingerprint = canonical_sha256({"token_ids": full_ids.tolist()})
    payload = {
        "schema_version": "mrun-qwen35-stateful-two-token-parity-v1",
        "runtime_abi": QWEN35_STATEFUL_RUNTIME_ABI,
        "artifact_id": executable.artifact.artifact_id,
        "case_fingerprint": case_fingerprint,
        "stateless_logits_sha256": _tensor_sha256(stateless),
        "stateful_logits_sha256": _tensor_sha256(stateful),
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "bit_exact": bit_exact,
        "certified": certified,
    }
    return Qwen35StatefulParityReceipt(
        artifact_id=payload["artifact_id"],
        case_fingerprint=case_fingerprint,
        stateless_logits_sha256=payload["stateless_logits_sha256"],
        stateful_logits_sha256=payload["stateful_logits_sha256"],
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        bit_exact=bit_exact,
        certified=certified,
        fingerprint=canonical_sha256(payload),
    )


__all__ = [
    "QWEN35_STATEFUL_RUNTIME_ABI",
    "Qwen35ProvisionalSuffix",
    "Qwen35StateSnapshot",
    "Qwen35StatefulParityReceipt",
    "Qwen35TransactionalReference",
    "certify_qwen35_stateful_two_token_parity",
]
