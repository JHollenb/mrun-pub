from __future__ import annotations

import numpy as np
import pytest
import torch

from mrun.engine.kernels.paged_forward import Qwen35PagedState
from mrun.engine.qwen35_statecut import Qwen35HybridStateCut, Qwen35StateCutError


def _state() -> Qwen35PagedState:
    return Qwen35PagedState(
        capacity=8,
        pos=2,
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        conv={0: torch.ones(1, 3, 2)},
        recurrent={0: torch.ones(1, 2, 2, 2)},
        key={1: torch.zeros(1, 8, 1, 2)},
        value={1: torch.zeros(1, 8, 1, 2)},
    )


def _forward(_store, rows: np.ndarray, state: Qwen35PagedState) -> torch.Tensor:
    state.conv[0].add_(torch.tensor(rows.sum(axis=1))[:, None, None])
    state.recurrent[0].add_(torch.tensor(rows[:, -1])[:, None, None, None])
    state.key[1][:, state.pos : state.pos + rows.shape[1]].fill_(2)
    state.value[1][:, state.pos : state.pos + rows.shape[1]].fill_(3)
    state.pos += rows.shape[1]
    return torch.tensor(rows, dtype=torch.float32)


def test_same_parent_branches_patch_commit_and_restore() -> None:
    cut = Qwen35HybridStateCut(None, _state(), forward=_forward)
    parent = cut.parent_fingerprint
    proposal = cut.stage(
        np.asarray([[4, 5], [7, 8]]),
        recurrent_patches={1: {0: torch.ones(2, 2, 2) * 10}},
    )
    assert cut.parent_fingerprint == parent
    assert proposal.receipt.durable_tensor_bytes == 0
    assert proposal.receipt.recurrent_patch_layers == ((), (0,))
    committed = cut.commit(proposal, 1)
    assert committed != parent
    assert cut.parent_position == 4
    assert torch.all(cut.recurrent_state(0) > 10)
    second = cut.stage(np.asarray([[1]]))
    before_restore = cut.parent_fingerprint
    assert cut.restore(second) == before_restore
    assert cut.parent_fingerprint == before_restore


def test_stale_or_closed_proposals_fail_closed() -> None:
    cut = Qwen35HybridStateCut(None, _state(), forward=_forward)
    first = cut.stage(np.asarray([[1]]))
    second = cut.stage(np.asarray([[2]]))
    cut.commit(first, 0)
    with pytest.raises(Qwen35StateCutError):
        cut.commit(second, 0)
