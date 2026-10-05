"""Public transactional StateCut for Qwen3.5 convolution/GDN/full-KV state.

The durable object is metadata plus fingerprints.  Branch tensors exist only
while a same-parent native suffix is running; no full state tape is serialized.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .kernels.paged_forward import Qwen35PagedState, paged_logits_qwen35_kv_batch


class Qwen35StateCutError(RuntimeError):
    """Raised when a hybrid-state transaction violates its parent contract."""


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def _state_digest(state: Qwen35PagedState) -> str:
    digest = hashlib.sha256()
    digest.update(f"{state.capacity}:{state.pos}:{state.batch_size}:{state.dtype}".encode())
    for kind, slots in (
        ("conv", state.conv),
        ("recurrent", state.recurrent),
        ("key", state.key),
        ("value", state.value),
    ):
        for layer, value in sorted(slots.items()):
            digest.update(f"{kind}:{layer}:{tuple(value.shape)}:{value.dtype}".encode())
            digest.update(_tensor_digest(value).encode())
    return digest.hexdigest()


def _live_bytes(state: Qwen35PagedState) -> int:
    total = 0
    for slots in (state.conv, state.recurrent):
        total += sum(value.numel() * value.element_size() for value in slots.values())
    for slots in (state.key, state.value):
        total += sum(
            value[:, : state.pos].numel() * value.element_size() for value in slots.values()
        )
    return int(total)


def _repeat_parent(parent: Qwen35PagedState, branches: int) -> Qwen35PagedState:
    def repeated(
        slots: Mapping[int, torch.Tensor], *, live_prefix: bool
    ) -> dict[int, torch.Tensor]:
        result: dict[int, torch.Tensor] = {}
        for layer, value in slots.items():
            if live_prefix:
                target = torch.empty(
                    (branches, *value.shape[1:]), dtype=value.dtype, device=value.device
                )
                if parent.pos:
                    target[:, : parent.pos] = value[:, : parent.pos].expand(branches, -1, -1, -1)
            else:
                target = value.expand(branches, *value.shape[1:]).clone()
            result[layer] = target
        return result

    return Qwen35PagedState(
        capacity=parent.capacity,
        pos=parent.pos,
        batch_size=branches,
        dtype=parent.dtype,
        device=parent.device,
        conv=repeated(parent.conv, live_prefix=False),
        recurrent=repeated(parent.recurrent, live_prefix=False),
        key=repeated(parent.key, live_prefix=True),
        value=repeated(parent.value, live_prefix=True),
    )


def _select_branch(state: Qwen35PagedState, branch: int) -> Qwen35PagedState:
    def selected(slots: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {layer: value[branch : branch + 1].clone() for layer, value in slots.items()}

    return Qwen35PagedState(
        capacity=state.capacity,
        pos=state.pos,
        batch_size=1,
        dtype=state.dtype,
        device=state.device,
        conv=selected(state.conv),
        recurrent=selected(state.recurrent),
        key=selected(state.key),
        value=selected(state.value),
    )


@dataclass(frozen=True, slots=True)
class Qwen35HybridStateReceipt:
    schema: str
    parent_epoch: int
    parent_fingerprint: str
    parent_position: int
    branch_count: int
    suffix_tokens: int
    recurrent_patch_layers: tuple[tuple[int, ...], ...]
    parent_live_bytes: int
    transient_materialized_bytes: int
    durable_tensor_bytes: int = 0


@dataclass(slots=True)
class Qwen35HybridProposal:
    parent_epoch: int
    parent_fingerprint: str
    logits: torch.Tensor
    receipt: Qwen35HybridStateReceipt
    _state: Qwen35PagedState
    _owner: object
    _closed: bool = False


class Qwen35HybridStateCut:
    """One immutable parent with batched branch-local hybrid-state proposals."""

    def __init__(
        self,
        store: Any,
        parent: Qwen35PagedState,
        *,
        forward: Callable[
            [Any, np.ndarray, Qwen35PagedState], torch.Tensor
        ] = paged_logits_qwen35_kv_batch,
    ) -> None:
        if parent.batch_size != 1:
            raise Qwen35StateCutError("StateCut parent must contain exactly one request")
        self.store = store
        self._state = parent
        self._forward = forward
        self._epoch = 0
        self._owner = object()

    @property
    def parent_fingerprint(self) -> str:
        return _state_digest(self._state)

    @property
    def parent_position(self) -> int:
        return self._state.pos

    def recurrent_state(self, layer: int) -> torch.Tensor:
        """Return a defensive read-only copy of one recurrent state slot."""

        if layer not in self._state.recurrent:
            raise Qwen35StateCutError("requested layer is not recurrent")
        return self._state.recurrent[layer].clone()

    def stage(
        self,
        input_ids: np.ndarray,
        *,
        recurrent_patches: Mapping[int, Mapping[int, torch.Tensor]] | None = None,
    ) -> Qwen35HybridProposal:
        rows = np.asarray(input_ids, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
            raise Qwen35StateCutError("suffix rows must have shape [branches, tokens]")
        parent_fingerprint = self.parent_fingerprint
        branch_state = _repeat_parent(self._state, int(rows.shape[0]))
        patches = recurrent_patches or {}
        patch_inventory: list[tuple[int, ...]] = []
        for branch in range(rows.shape[0]):
            layers = patches.get(branch, {})
            patch_inventory.append(tuple(sorted(layers)))
            for layer, delta in layers.items():
                if layer not in branch_state.recurrent:
                    raise Qwen35StateCutError("patch targets a non-recurrent layer")
                target = branch_state.recurrent[layer][branch]
                if delta.shape not in (
                    target.shape,
                    branch_state.recurrent[layer][branch : branch + 1].shape,
                ):
                    raise Qwen35StateCutError("recurrent patch geometry changed")
                target.add_(delta.reshape_as(target).to(device=target.device, dtype=target.dtype))
        logits = self._forward(self.store, rows, branch_state)
        if self.parent_fingerprint != parent_fingerprint:
            raise Qwen35StateCutError("native suffix mutated the immutable parent")
        receipt = Qwen35HybridStateReceipt(
            schema="mrun-qwen35-hybrid-statecut-v1",
            parent_epoch=self._epoch,
            parent_fingerprint=parent_fingerprint,
            parent_position=self._state.pos,
            branch_count=int(rows.shape[0]),
            suffix_tokens=int(rows.shape[1]),
            recurrent_patch_layers=tuple(patch_inventory),
            parent_live_bytes=_live_bytes(self._state),
            transient_materialized_bytes=_live_bytes(branch_state),
        )
        return Qwen35HybridProposal(
            parent_epoch=self._epoch,
            parent_fingerprint=parent_fingerprint,
            logits=logits,
            receipt=receipt,
            _state=branch_state,
            _owner=self._owner,
        )

    def commit(self, proposal: Qwen35HybridProposal, branch: int) -> str:
        if proposal._owner is not self._owner or proposal._closed:
            raise Qwen35StateCutError("proposal is foreign or closed")
        if (
            proposal.parent_epoch != self._epoch
            or proposal.parent_fingerprint != self.parent_fingerprint
        ):
            raise Qwen35StateCutError("proposal parent is stale")
        if not 0 <= branch < proposal.receipt.branch_count:
            raise Qwen35StateCutError("selected branch is out of range")
        self._state = _select_branch(proposal._state, branch)
        self._epoch += 1
        proposal._closed = True
        return self.parent_fingerprint

    def restore(self, proposal: Qwen35HybridProposal) -> str:
        if proposal._owner is not self._owner or proposal._closed:
            raise Qwen35StateCutError("proposal is foreign or closed")
        if (
            proposal.parent_epoch != self._epoch
            or proposal.parent_fingerprint != self.parent_fingerprint
        ):
            raise Qwen35StateCutError("proposal parent is stale")
        proposal._closed = True
        return self.parent_fingerprint


__all__ = [
    "Qwen35HybridProposal",
    "Qwen35HybridStateCut",
    "Qwen35HybridStateReceipt",
    "Qwen35StateCutError",
]
