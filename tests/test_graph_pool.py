from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from mrun.engine.graph_pool import (
    GraphTemplate,
    GraphTemplateKey,
    GraphTemplatePool,
    ResidentQStoreArena,
    ResidentQStoreArenaKey,
)


class _BackendArena:
    resident_bytes = 4096

    def __init__(self) -> None:
        self.closed = False
        self.verifications = 0

    def verify_stable_addresses(self) -> None:
        if self.closed:
            raise RuntimeError("closed")
        self.verifications += 1

    def close(self) -> None:
        self.closed = True


@dataclass
class _Executor:
    generation: int = 0
    closed: bool = False
    installed: tuple[tuple[tuple[int, ...], ...], tuple[int, ...], str] | None = None

    def rebind(self, rows: Any, token_ids: Any, *, request_id: str) -> int:
        if self.closed:
            raise RuntimeError("closed")
        self.generation += 1
        self.installed = (
            tuple(tuple(int(value) for value in row) for row in rows),
            tuple(int(value) for value in token_ids),
            request_id,
        )
        return self.generation

    def execute(self, *, expected_generation: int, request_id: str) -> tuple[Any, ...]:
        if expected_generation != self.generation:
            raise RuntimeError("stale")
        assert self.installed is not None
        if request_id != self.installed[2]:
            raise RuntimeError("contaminated")
        return self.installed

    def close(self) -> None:
        self.closed = True


def _arena() -> ResidentQStoreArena:
    key = ResidentQStoreArenaKey(
        model_identity="model@revision#store",
        numerical_contract="same-codec-w8a16",
        device_identity="cpu-fake",
        runtime_distribution_sha256="a" * 64,
        activation_dtype="fp32",
        weight_dtype="int8",
    )
    backend = _BackendArena()
    return ResidentQStoreArena(key, backend, resident_bytes=backend.resident_bytes)


def _key(arena: ResidentQStoreArena, *, sequence_length: int = 5) -> GraphTemplateKey:
    return GraphTemplateKey(
        arena_fingerprint=arena.key.fingerprint,
        execution_mode="selected-last-stateless-score",
        output_contract="selected-token-rows",
        batch_size=1,
        sequence_length=sequence_length,
        selected_row_count=6,
    )


def test_one_thousand_alternating_rebindings_are_fenced_and_uncontaminated() -> None:
    arena = _arena()
    key = _key(arena)
    template = GraphTemplate(key, arena, (_Executor(), _Executor()), template_bytes=512)

    seen: set[tuple[Any, ...]] = set()
    for index in range(1_000):
        rows = ((index % 17, index % 23, 3, 4, 5),)
        selected = tuple((index + offset) % 101 for offset in range(6))
        request_id = f"request-{index}"
        with template.lease() as lease:
            result = lease.execute(rows, selected, request_id=request_id)
        installed_rows, installed_selected, installed_request = result.output
        assert installed_rows == rows
        assert installed_selected == selected
        assert installed_request == result.request_id == request_id
        seen.add((result.lane_id, result.generation))

    evidence = template.evidence()
    assert evidence["replay_count"] == evidence["lease_count"] == 1_000
    assert len(seen) == 1_000
    assert arena.template_references == 1
    template.close()
    assert arena.template_references == 0
    arena.close()


def test_lane_epoch_refuses_stale_or_foreign_execution() -> None:
    executor = _Executor()
    first = executor.rebind(((1, 2),), (3, 4), request_id="first")
    second = executor.rebind(((5, 6),), (7, 8), request_id="second")
    assert second == first + 1
    with pytest.raises(RuntimeError, match="stale"):
        executor.execute(expected_generation=first, request_id="first")
    with pytest.raises(RuntimeError, match="contaminated"):
        executor.execute(expected_generation=second, request_id="first")


def test_pool_reuses_exact_keys_and_evicts_only_idle_templates() -> None:
    arena = _arena()
    pool = GraphTemplatePool(max_template_bytes=1_024, max_templates=1)
    first_key = _key(arena, sequence_length=5)
    second_key = _key(arena, sequence_length=16)

    def make(key: GraphTemplateKey) -> GraphTemplate:
        return GraphTemplate(key, arena, (_Executor(),), template_bytes=512)

    first = pool.get_or_create(
        first_key,
        lambda: make(first_key),
        estimated_template_bytes=512,
    )
    assert (
        pool.get_or_create(
            first_key,
            lambda: pytest.fail("cache hit invoked factory"),
            estimated_template_bytes=512,
        )
        is first
    )
    second = pool.get_or_create(
        second_key,
        lambda: make(second_key),
        estimated_template_bytes=512,
    )
    assert second is not first
    inventory = pool.inventory()
    assert inventory["template_count"] == 1
    assert inventory["template_hit_count"] == 1
    assert inventory["eviction_count"] == 1
    pool.close()
    arena.close()


def test_pool_refuses_eviction_of_an_inflight_lane() -> None:
    arena = _arena()
    pool = GraphTemplatePool(max_template_bytes=512, max_templates=1)
    first_key = _key(arena, sequence_length=5)
    second_key = _key(arena, sequence_length=16)
    first = pool.get_or_create(
        first_key,
        lambda: GraphTemplate(first_key, arena, (_Executor(),), template_bytes=512),
        estimated_template_bytes=512,
    )
    lease = first.lease()
    with pytest.raises(MemoryError, match="no evictable capacity"):
        pool.get_or_create(
            second_key,
            lambda: GraphTemplate(second_key, arena, (_Executor(),), template_bytes=512),
            estimated_template_bytes=512,
        )
    lease.release()
    pool.close()
    arena.close()
