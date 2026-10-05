"""Transactional copy-on-write StateCuts for the oversized Qwen3 MoE CUDA runtime.

The data plane is genuinely segmented: one immutable B1 parent K/V plus B branch-local tails.
The COW attention kernel reads both segments directly, so staging N branches copies zero parent
bytes. Committing the selected branch materializes one B1 state only after adjudication.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import torch

from .qwen3_moe_cuda import ForkedKVCache, StaticKVCache

Qwen3MoeStateCutOutput = Literal[
    "hidden_state_only",
    "full_logits",
    "generated_token_ids",
]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_signature(cache: StaticKVCache) -> tuple[Any, ...]:
    return (
        int(cache.batch_size),
        int(cache.capacity),
        int(cache.length),
        tuple(
            (
                int(layer.key.data_ptr()),
                int(layer.value.data_ptr()),
                tuple(int(value) for value in layer.key.shape),
                tuple(int(value) for value in layer.value.shape),
                str(layer.key.dtype),
                str(layer.key.device),
                int(layer.key._version),  # noqa: SLF001 - immutable-parent mutation guard
                int(layer.value._version),  # noqa: SLF001 - immutable-parent mutation guard
            )
            for layer in cache.layers
        ),
    )


def _committed_bytes(cache: StaticKVCache) -> int:
    return int(
        sum(
            (
                layer.key[:, :, : cache.length].numel()
                + layer.value[:, :, : cache.length].numel()
            )
            * layer.key.element_size()
            for layer in cache.layers
        )
    )


@dataclass(frozen=True, slots=True)
class Qwen3MoeKVStateCutDescriptor:
    schema: str
    cut_id: str
    parent_epoch: int
    parent_fingerprint: str
    parent_tokens: int
    parent_allocated_bytes: int
    parent_committed_bytes: int
    retention_budget_bytes: int
    attention_abi: str
    parent_copy_bytes_per_stage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Qwen3MoeKVStateCutBranch:
    cut_id: str
    branch_id: str
    parent_epoch: int


@dataclass(slots=True)
class Qwen3MoeKVStateCutContinuation:
    cut_id: str
    branch_ids: tuple[str, ...]
    parent_epoch: int
    output_contract: str
    outputs: torch.Tensor
    physical_forwards: int
    parent_copy_bytes: int
    retained_diff_bytes: int
    retained_total_bytes: int
    attention_abi: str

    def metadata(self) -> dict[str, Any]:
        return {
            "cut_id": self.cut_id,
            "branch_ids": self.branch_ids,
            "parent_epoch": self.parent_epoch,
            "output_contract": self.output_contract,
            "output_shape": tuple(int(value) for value in self.outputs.shape),
            "output_dtype": str(self.outputs.dtype),
            "output_device": str(self.outputs.device),
            "physical_forwards": self.physical_forwards,
            "parent_copy_bytes": self.parent_copy_bytes,
            "retained_diff_bytes": self.retained_diff_bytes,
            "retained_total_bytes": self.retained_total_bytes,
            "attention_abi": self.attention_abi,
        }


@dataclass(frozen=True, slots=True)
class Qwen3MoeKVStateCutReceipt:
    schema: str
    cut_id: str
    decision: str
    selected_branch_id: str | None
    parent_epoch_before: int
    parent_epoch_after: int
    parent_fingerprint_before: str
    parent_fingerprint_after: str
    branch_count: int
    parent_copy_bytes_during_stage: int
    materialized_commit_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Qwen3MoeKVStateCut:
    """One immutable prefix, one fused branch continuation, one terminal decision."""

    attention_abi = "qwen3-moe-segmented-gqa-cow-online-softmax-v1"

    def __init__(
        self,
        engine: Any,
        parent: StaticKVCache,
        *,
        retention_budget_bytes: int,
    ) -> None:
        if parent.batch_size != 1 or parent.length <= 0:
            raise ValueError("Qwen3 MoE StateCut parent must contain one committed request")
        if isinstance(retention_budget_bytes, bool) or not isinstance(
            retention_budget_bytes, Integral
        ):
            raise TypeError("retention_budget_bytes must be an integer")
        if _committed_bytes(parent) > int(retention_budget_bytes):
            raise MemoryError("Qwen3 MoE StateCut parent exceeds its retention budget")
        self._engine = engine
        self._runtime = engine._require_runtime()  # noqa: SLF001 - engine-owned native seam
        self._parent = parent
        self._budget = int(retention_budget_bytes)
        self._cut_id = uuid4().hex
        self._epoch = 0
        self._guard = _cache_signature(parent)
        self._branches: dict[str, Qwen3MoeKVStateCutBranch] = {}
        self._continuation: Qwen3MoeKVStateCutContinuation | None = None
        self._fork_cache: ForkedKVCache | None = None
        self._terminal = False
        self._lock = threading.RLock()

    @classmethod
    def prefill(
        cls,
        engine: Any,
        input_ids: Sequence[int] | np.ndarray,
        *,
        retention_budget_bytes: int,
    ) -> tuple[Qwen3MoeKVStateCut, torch.Tensor]:
        row = np.asarray(input_ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("Qwen3 MoE StateCut prefill requires one non-empty token row")
        runtime = engine._require_runtime()  # noqa: SLF001 - engine-owned native seam
        parent = runtime.new_cache(batch_size=1, capacity=int(row.size))
        result = runtime.forward(
            torch.as_tensor(row, device=engine.device, dtype=torch.long)[None, :],
            cache=parent,
            return_final_hidden=True,
            skip_lm_head=True,
        )
        assert result.final_hidden is not None
        return (
            cls(engine, parent, retention_budget_bytes=retention_budget_bytes),
            result.final_hidden,
        )

    @property
    def parent_fingerprint(self) -> str:
        return _canonical_sha256(self._guard)

    @property
    def descriptor(self) -> Qwen3MoeKVStateCutDescriptor:
        return Qwen3MoeKVStateCutDescriptor(
            schema="mrun-qwen3-moe-kv-statecut-v1",
            cut_id=self._cut_id,
            parent_epoch=self._epoch,
            parent_fingerprint=self.parent_fingerprint,
            parent_tokens=self._parent.length,
            parent_allocated_bytes=self._parent.device_bytes,
            parent_committed_bytes=_committed_bytes(self._parent),
            retention_budget_bytes=self._budget,
            attention_abi=self.attention_abi,
        )

    def _require_open(self) -> None:
        if self._terminal:
            raise RuntimeError("Qwen3 MoE StateCut transaction is terminal")

    def _assert_parent_unchanged(self) -> None:
        if _cache_signature(self._parent) != self._guard:
            raise RuntimeError("Qwen3 MoE StateCut immutable parent changed")

    def replay(self) -> Qwen3MoeKVStateCut:
        """Open a new transaction over the same immutable parent without copying it.

        The returned cut has independent branches, continuation buffers, and terminal state.
        This is the native hook used when Saturn reuses a compiled source cell across several
        candidate coalitions while preserving one physical prefix cache.
        """

        with self._lock:
            self._assert_parent_unchanged()
            return type(self)(
                self._engine,
                self._parent,
                retention_budget_bytes=self._budget,
            )

    def fork(self, branch_id: str) -> Qwen3MoeKVStateCutBranch:
        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            if self._continuation is not None:
                raise RuntimeError("cannot add Qwen3 MoE branches after continuation")
            normalized = str(branch_id).strip()
            if not normalized:
                raise ValueError("branch_id cannot be empty")
            if normalized in self._branches:
                raise ValueError(f"duplicate Qwen3 MoE StateCut branch {normalized!r}")
            branch = Qwen3MoeKVStateCutBranch(self._cut_id, normalized, self._epoch)
            self._branches[normalized] = branch
            return branch

    def continue_one(
        self,
        tokens_by_branch: Mapping[str, Sequence[int] | np.ndarray],
        *,
        output_contract: Qwen3MoeStateCutOutput = "hidden_state_only",
    ) -> Qwen3MoeKVStateCutContinuation:
        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            if self._continuation is not None:
                raise RuntimeError("Qwen3 MoE StateCut branches have already continued")
            branch_ids = tuple(self._branches)
            if not branch_ids:
                raise RuntimeError("Qwen3 MoE StateCut continuation requires at least one fork")
            if set(tokens_by_branch) != set(branch_ids):
                raise ValueError("continuation tokens must cover every fork exactly once")
            tokens: list[int] = []
            for branch_id in branch_ids:
                row = np.asarray(tokens_by_branch[branch_id], dtype=np.int64)
                if row.ndim != 1 or tuple(row.shape) != (1,):
                    raise ValueError("each Qwen3 MoE branch must supply exactly one token")
                tokens.append(int(row[0]))
            if output_contract not in ("hidden_state_only", "full_logits"):
                raise ValueError(f"unsupported Qwen3 MoE StateCut output {output_contract!r}")
            fork = self._runtime.new_fork_cache(
                self._parent,
                batch_size=len(branch_ids),
                capacity=self._parent.length + 1,
            )
            width = (
                self._runtime.hidden
                if output_contract == "hidden_state_only"
                else int(self._runtime.cfg["vocab_size"])
            )
            output_bytes = len(branch_ids) * width * torch.float32.itemsize
            retained_total = self._parent.device_bytes + fork.device_bytes + output_bytes
            if retained_total > self._budget:
                raise MemoryError(
                    "Qwen3 MoE parent + branch diff + output exceeds retention budget"
                )
            result = self._runtime.forward_cow_decode(
                torch.as_tensor(tokens, device=self._engine.device, dtype=torch.long),
                cache=fork,
                return_final_hidden=output_contract == "hidden_state_only",
                skip_lm_head=output_contract == "hidden_state_only",
            )
            self._assert_parent_unchanged()
            output = (
                result.final_hidden
                if output_contract == "hidden_state_only"
                else result.logits
            )
            assert output is not None
            continuation = Qwen3MoeKVStateCutContinuation(
                cut_id=self._cut_id,
                branch_ids=branch_ids,
                parent_epoch=self._epoch,
                output_contract=output_contract,
                outputs=output,
                physical_forwards=1,
                parent_copy_bytes=0,
                retained_diff_bytes=fork.committed_bytes,
                retained_total_bytes=retained_total,
                attention_abi=self.attention_abi,
            )
            self._fork_cache = fork
            self._continuation = continuation
            return continuation

    def generate(
        self,
        tokens_by_branch: Mapping[str, Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int,
    ) -> Qwen3MoeKVStateCutContinuation:
        """Greedily generate equal-horizon branches from one forced token per branch.

        This is equivalent to running ordinary generation on ``parent + forced_token`` for
        every branch, while physically executing one shared parent and one fused decode panel.
        The returned token matrix has shape ``[branches, max_new_tokens]``. As in ordinary
        autoregressive generation, the final predicted token has not yet been appended to K/V.
        """

        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            if self._continuation is not None:
                raise RuntimeError("Qwen3 MoE StateCut branches have already continued")
            if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
                raise TypeError("max_new_tokens must be an integer")
            if int(max_new_tokens) <= 0:
                raise ValueError("max_new_tokens must be positive")
            branch_ids = tuple(self._branches)
            if not branch_ids:
                raise RuntimeError("Qwen3 MoE StateCut generation requires at least one fork")
            if set(tokens_by_branch) != set(branch_ids):
                raise ValueError("generation tokens must cover every fork exactly once")
            tokens: list[int] = []
            for branch_id in branch_ids:
                row = np.asarray(tokens_by_branch[branch_id], dtype=np.int64)
                if row.ndim != 1 or tuple(row.shape) != (1,):
                    raise ValueError("each Qwen3 MoE branch must supply exactly one forced token")
                tokens.append(int(row[0]))

            steps = int(max_new_tokens)
            fork = self._runtime.new_fork_cache(
                self._parent,
                batch_size=len(branch_ids),
                capacity=self._parent.length + steps,
            )
            output_bytes = len(branch_ids) * steps * torch.long.itemsize
            retained_total = self._parent.device_bytes + fork.device_bytes + output_bytes
            if retained_total > self._budget:
                raise MemoryError(
                    "Qwen3 MoE parent + generation diff + output exceeds retention budget"
                )

            current = torch.as_tensor(tokens, device=self._engine.device, dtype=torch.long)
            generated: list[torch.Tensor] = []
            for _ in range(steps):
                result = self._runtime.forward_cow_decode(current, cache=fork)
                assert result.logits is not None
                current = result.logits.argmax(dim=-1)
                generated.append(current)
            self._assert_parent_unchanged()
            output = torch.stack(generated, dim=1)
            continuation = Qwen3MoeKVStateCutContinuation(
                cut_id=self._cut_id,
                branch_ids=branch_ids,
                parent_epoch=self._epoch,
                output_contract="generated_token_ids",
                outputs=output,
                physical_forwards=steps,
                parent_copy_bytes=0,
                retained_diff_bytes=fork.committed_bytes,
                retained_total_bytes=retained_total,
                attention_abi=self.attention_abi,
            )
            self._fork_cache = fork
            self._continuation = continuation
            return continuation

    def commit(self, branch_id: str) -> Qwen3MoeKVStateCutReceipt:
        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            if self._continuation is None or self._fork_cache is None:
                raise RuntimeError("Qwen3 MoE StateCut commit requires a continuation")
            normalized = str(branch_id).strip()
            try:
                selected = self._continuation.branch_ids.index(normalized)
            except ValueError as error:
                raise KeyError(f"unknown Qwen3 MoE StateCut branch {normalized!r}") from error
            before_epoch = self._epoch
            before_fingerprint = self.parent_fingerprint
            committed = self._runtime.materialize_fork_branch(self._fork_cache, selected)
            materialized_bytes = committed.device_bytes
            self._parent = committed
            self._epoch += 1
            self._guard = _cache_signature(committed)
            self._terminal = True
            return Qwen3MoeKVStateCutReceipt(
                schema="mrun-qwen3-moe-kv-statecut-receipt-v1",
                cut_id=self._cut_id,
                decision="commit",
                selected_branch_id=normalized,
                parent_epoch_before=before_epoch,
                parent_epoch_after=self._epoch,
                parent_fingerprint_before=before_fingerprint,
                parent_fingerprint_after=self.parent_fingerprint,
                branch_count=len(self._branches),
                parent_copy_bytes_during_stage=0,
                materialized_commit_bytes=materialized_bytes,
            )

    def abandon(self) -> Qwen3MoeKVStateCutReceipt:
        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            self._terminal = True
            return Qwen3MoeKVStateCutReceipt(
                schema="mrun-qwen3-moe-kv-statecut-receipt-v1",
                cut_id=self._cut_id,
                decision="abandon",
                selected_branch_id=None,
                parent_epoch_before=self._epoch,
                parent_epoch_after=self._epoch,
                parent_fingerprint_before=self.parent_fingerprint,
                parent_fingerprint_after=self.parent_fingerprint,
                branch_count=len(self._branches),
                parent_copy_bytes_during_stage=0,
                materialized_commit_bytes=0,
            )


__all__ = [
    "Qwen3MoeKVStateCut",
    "Qwen3MoeKVStateCutBranch",
    "Qwen3MoeKVStateCutContinuation",
    "Qwen3MoeKVStateCutDescriptor",
    "Qwen3MoeKVStateCutReceipt",
]
