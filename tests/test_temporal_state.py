from __future__ import annotations

import pytest

from mrun.compiler import TemporalStateArena


def test_temporal_state_commit_rollback_fork_and_fixed_allocations() -> None:
    arena = TemporalStateArena(slots=4, capacity=32)
    arena.create("root", (1, 2, 3))
    child = arena.fork("root", "child")
    assert child.prefix_state_id == "root"
    assert arena.tokens("child") == arena.tokens("root") == (1, 2, 3)

    committed = arena.commit(arena.begin("child", (4, 5, 6)), 2)
    assert committed.epoch == child.epoch + 1
    assert arena.tokens("child") == (1, 2, 3, 4, 5)
    delta = arena.begin("root", (9, 10))
    before = arena.observe("root")
    after = arena.rollback(delta)
    assert after == before
    assert arena.evidence()["allocation_count"] == 4


def test_temporal_state_one_thousand_interleaved_operations_do_not_contaminate_slots() -> None:
    arena = TemporalStateArena(slots=8, capacity=256)
    for index in range(8):
        arena.create(f"s{index}", (index,))
    expected = {f"s{index}": [index] for index in range(8)}
    for step in range(1_000):
        state_id = f"s{step % 8}"
        token = 1000 + step
        delta = arena.begin(state_id, (token,))
        if step % 3:
            arena.commit(delta, 1)
            expected[state_id].append(token)
        else:
            arena.rollback(delta)
    for state_id, tokens in expected.items():
        assert arena.tokens(state_id) == tuple(tokens)
    assert arena.evidence()["active_slots"] == arena.evidence()["peak_active_slots"] == 8


def test_temporal_state_rejects_stale_commit_and_overflow() -> None:
    arena = TemporalStateArena(slots=1, capacity=2)
    arena.create("state", (1,))
    stale = arena.begin("state", (2,))
    arena.commit(stale, 1)
    with pytest.raises(RuntimeError, match="stale"):
        arena.commit(stale, 1)
    with pytest.raises(OverflowError, match="capacity"):
        arena.begin("state", (3,))
