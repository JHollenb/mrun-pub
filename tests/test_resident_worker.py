from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from mrun.compiler import ResidentTemplatePromotion
from mrun.engine import (
    GraphTemplate,
    GraphTemplateKey,
    GraphTemplatePool,
    ResidentModelWorker,
    ResidentQStoreArena,
    ResidentQStoreArenaKey,
    ResidentSelectedScoreRequest,
)


class _BackendArena:
    def verify_stable_addresses(self) -> None:
        return None

    def close(self) -> None:
        return None


class _Engine:
    @staticmethod
    def selected_last_logits_batch(rows: Any, selected: Any) -> torch.Tensor:
        return torch.tensor(
            [[sum(int(value) for value in row) + int(token) for token in selected] for row in rows],
            dtype=torch.float32,
        )


@dataclass
class _Executor:
    generation: int = 0
    rows: Any = None
    selected: Any = None
    request_id: str | None = None

    def rebind(self, rows: Any, selected: Any, *, request_id: str) -> int:
        self.generation += 1
        self.rows = rows
        self.selected = selected
        self.request_id = request_id
        return self.generation

    def execute(self, *, expected_generation: int, request_id: str) -> torch.Tensor:
        if expected_generation != self.generation or request_id != self.request_id:
            raise RuntimeError("binding contamination")
        return _Engine.selected_last_logits_batch(self.rows, self.selected)

    def close(self) -> None:
        return None


def _arena() -> ResidentQStoreArena:
    key = ResidentQStoreArenaKey(
        model_identity="model@revision#store",
        numerical_contract="exact",
        device_identity="cpu-fake",
        runtime_distribution_sha256="1" * 64,
        activation_dtype="fp32",
        weight_dtype="int8",
    )
    return ResidentQStoreArena(key, _BackendArena(), resident_bytes=10_000)


def _promotion(arena: ResidentQStoreArena) -> ResidentTemplatePromotion:
    return ResidentTemplatePromotion(
        promotion_id="cpu-fake-b1-s5-k6",
        arena_fingerprint=arena.key.fingerprint,
        template_implementation_sha256="2" * 64,
        mutable_binding_schema_sha256="3" * 64,
        numerical_contract="exact",
        output_contract="selected-token-rows",
        shape_bucket=(1, 5, 6),
        device_identity="cpu-fake",
        runtime_identity_sha256="4" * 64,
        arena_bytes=10_000,
        template_bytes=500,
        setup_ms=1.0,
        rebind_ms=0.01,
        replay_speedup_lower_95=1.1,
        lane_count=2,
        contamination_trials=1_000,
        cancellation_passed=True,
        exact_parity=True,
        evidence_sha256="5" * 64,
        wheel_sha256="6" * 64,
    )


def _request(index: int) -> ResidentSelectedScoreRequest:
    return ResidentSelectedScoreRequest(
        request_id=f"request-{index}",
        token_rows=((index % 13, 2, 3, 4, 5),),
        selected_token_ids=tuple((index + offset) % 101 for offset in range(6)),
        numerical_contract="exact",
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
    )


def test_resident_worker_serves_one_thousand_promoted_rebindings_with_exact_parity() -> None:
    arena = _arena()
    promotion = _promotion(arena)

    def factory(
        key: GraphTemplateKey,
        owner: ResidentQStoreArena,
        _record: ResidentTemplatePromotion,
    ) -> GraphTemplate:
        return GraphTemplate(key, owner, (_Executor(), _Executor()), template_bytes=500)

    worker = ResidentModelWorker(
        _Engine(),
        arena,
        GraphTemplatePool(max_template_bytes=1_000),
        (promotion,),
        factory,
        device_identity="cpu-fake",
        estimated_template_bytes=500,
    )
    for index in range(1_000):
        request = _request(index)
        response = worker.execute(request)
        expected = _Engine.selected_last_logits_batch(
            request.token_rows, request.selected_token_ids
        )
        torch.testing.assert_close(torch.tensor(response.scores), expected)
        assert response.request_id == request.request_id
        assert response.route == "resident-graph-template"
        assert response.generation is not None
    inventory = worker.inventory()
    assert inventory["graph_request_count"] == 1_000
    assert inventory["eager_fallback_count"] == 0
    assert inventory["capture_count"] == 1
    worker.close()


def test_resident_worker_fails_closed_to_matched_eager_without_promotion() -> None:
    arena = _arena()
    worker = ResidentModelWorker(
        _Engine(),
        arena,
        GraphTemplatePool(max_template_bytes=1_000),
        (),
        lambda *_args: None,
        device_identity="cpu-fake",
        estimated_template_bytes=500,
    )
    response = worker.execute(_request(1))
    assert response.route == "eager-fallback"
    assert response.fallback_reason == "no-exact-template-promotion"
    assert worker.inventory()["fallback_reasons"] == {"no-exact-template-promotion": 1}
    worker.close()
