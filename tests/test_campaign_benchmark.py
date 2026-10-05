from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mrun.compiler import campaign_benchmark as cb
from mrun.compiler import campaign_factorial as cf
from mrun.compiler.campaign import CandidateReadout, compile_candidate_campaign

_CAMPAIGN_FP = "0" * 64
_COMPILATION_FP = "1" * 64


def _qrow(shape: tuple[int, int], offset: int) -> dict[str, Any]:
    rows, columns = shape
    return {
        "kind": "qrow",
        "shape": [rows, columns],
        "w_off": offset,
        "w_len": rows * columns,
        "s_off": offset * 4,
        "s_len": rows * 4,
    }


def _fp32(length: int, offset: int) -> dict[str, Any]:
    return {
        "kind": "fp32",
        "shape": [length],
        "e_off": offset,
        "e_len": length * 4,
    }


def _manifest() -> dict[str, Any]:
    return {
        "model_name": "tiny-qwen",
        "arch": "qwen2",
        "dtype": "int8",
        "source": {"source_checkpoint_sha256": "a" * 64},
        "derived": {"derived_store_sha256": "b" * 64},
        "config": {
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "intermediate_size": 8,
            "vocab_size": 8,
            "rms_norm_eps": 1e-6,
        },
        "blocks": {
            "embed": _qrow((8, 4), 0),
            "L0.ln1": _fp32(4, 0),
            "L0.q": _qrow((4, 4), 32),
            "L0.q.bias": _fp32(4, 4),
            "L0.k": _qrow((2, 4), 48),
            "L0.k.bias": _fp32(2, 8),
            "L0.v": _qrow((2, 4), 56),
            "L0.v.bias": _fp32(2, 10),
            "L0.o": _qrow((4, 4), 64),
            "L0.ln2": _fp32(4, 12),
            "L0.gate": _qrow((8, 4), 80),
            "L0.up": _qrow((8, 4), 112),
            "L0.down": _qrow((4, 8), 144),
            "norm.final": _fp32(4, 16),
            "lm_head": {"alias": "embed"},
        },
    }


class _CampaignStore:
    compute_dtype = torch.float32

    def __init__(self) -> None:
        self.man = _manifest()
        self.content_identity_verified = False
        self.identity_status = "test-declared-unverified"
        self.source_checkpoint_sha256 = "a" * 64
        self.derived_store_sha256 = "b" * 64

    def has(self, name: str) -> bool:
        return name in self.man["blocks"]


class _CampaignEngine:
    backend = "paged"
    device = torch.device("cpu")
    name = "tiny-qwen"
    n_layer = 1
    hidden = 4
    inter = 8
    numerical_contract = "paged-qstore-established"
    subset_head_numerical_contract = "paged-qstore-subset-head-fp32-v1"
    supported_numerical_contracts = (
        numerical_contract,
        subset_head_numerical_contract,
    )

    def __init__(self) -> None:
        self.store = _CampaignStore()
        self.spec = SimpleNamespace(name=self.name)
        self.selected_body_calls = 0
        self.selected_rows: list[tuple[int, ...]] = []

    def build_work_plan(
        self,
        ids_list: list[np.ndarray],
        **kwargs: Any,
    ) -> Any:
        from mrun.compiler import build_paged_qstore_plan

        return build_paged_qstore_plan(self, ids_list, **kwargs)

    def selected_last_logits_batch(
        self,
        ids_list: list[np.ndarray],
        token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        self.selected_body_calls += 1
        rows = tuple(int(token) for token in token_ids)
        self.selected_rows.append(rows)
        return torch.stack(
            [
                torch.as_tensor(
                    [float(np.asarray(ids).sum() + 2 * token) for token in rows],
                    dtype=torch.float32,
                )
                for ids in ids_list
            ]
        )

    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        vocab_size = int(self.store.man["config"]["vocab_size"])
        vocabulary = 2.0 * torch.arange(vocab_size, dtype=torch.float32)
        return [
            torch.stack(
                [float(np.asarray(ids).sum()) + vocabulary for _ in range(len(np.asarray(ids)))]
            )
            for ids in ids_list
        ]


class _DenseCampaignEngine(_CampaignEngine):
    backend = "dense-qstore-cuda"
    device = torch.device("cuda")
    max_seq_len = 16
    numerical_contract = "torch-batched-established"
    subset_head_numerical_contract = "torch-batched-established+selected-head-fp32-v1"
    supported_numerical_contracts = (
        numerical_contract,
        subset_head_numerical_contract,
    )

    def __init__(self) -> None:
        super().__init__()
        self.store.compute_dtype = torch.bfloat16

    def build_work_plan(
        self,
        ids_list: list[np.ndarray],
        **kwargs: Any,
    ) -> Any:
        from mrun.compiler import build_dense_qstore_plan

        return build_dense_qstore_plan(self, ids_list, **kwargs)


def _readouts() -> tuple[CandidateReadout, ...]:
    return (
        CandidateReadout("alpha", (5, 1, 3)),
        CandidateReadout("beta", (3, 7, 1)),
    )


class _Store:
    def __init__(self) -> None:
        self.max_block_bytes = 2_000_000
        self._cache_budget = 0


class _Engine:
    device = "cpu"

    def __init__(self) -> None:
        self.store = _Store()


def _factorial_protocol_result(
    monkeypatch: pytest.MonkeyPatch,
) -> cf.CandidateCampaignFactorialResult:
    clock = [0.0]
    output = (
        {
            "winner_token_id": 7,
            "runner_up_token_id": 8,
            "winner_logit": 3.0,
            "runner_up_logit": 2.0,
            "margin": 1.0,
        },
    )

    def advance(seconds: float) -> tuple[dict[str, Any], ...]:
        clock[0] += seconds
        return output

    monkeypatch.setattr(cf.time, "perf_counter", lambda: clock[0])
    return cf._run_factorial_campaign_benchmark(
        _Engine(),
        cf._FactorialRunners(
            full_head_independent=lambda _rotation: advance(0.008),
            full_head_union=lambda: advance(0.002),
            selected_independent=lambda _rotation: advance(0.004),
            compiled_union=lambda: advance(0.001),
        ),
        campaign_fingerprint=_CAMPAIGN_FP,
        compilation_fingerprint=_COMPILATION_FP,
        query_count=4,
        reported_fabric="cpu",
        warmup=2,
        trials=8,
        rtol=0.0,
        atol=0.0,
        minimum_improvement_percent=1.0,
    )


def test_factorial_protocol_measures_both_orthogonal_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _factorial_protocol_result(monkeypatch)

    assert result.sharing_at_full_head.median_ratio == pytest.approx(4.0)
    assert result.sharing_at_selected_head.median_ratio == pytest.approx(4.0)
    assert result.pushdown_at_independent.median_ratio == pytest.approx(2.0)
    assert result.pushdown_at_union.median_ratio == pytest.approx(2.0)
    assert result.combined.median_ratio == pytest.approx(8.0)
    assert result.interaction_median == pytest.approx(1.0)
    assert result.sharing_improvement_demonstrated
    assert result.pushdown_improvement_demonstrated
    assert result.combined_improvement_demonstrated
    assert (
        cf.CandidateCampaignFactorialResult.from_json(result.to_json()).as_dict()
        == result.as_dict()
    )
    forged = result.as_dict()
    forged["compiled_union_parity"]["allclose"] = False
    forged["compiled_union_parity"]["exact"] = False
    with pytest.raises(ValueError, match="sharing_improvement_demonstrated"):
        cf.CandidateCampaignFactorialResult.from_dict(forged)


@pytest.mark.parametrize(
    "parity_field",
    (
        "full_head_union_parity",
        "selected_independent_parity",
        "compiled_union_parity",
    ),
)
def test_factorial_rejects_vacuous_serialized_parity(
    monkeypatch: pytest.MonkeyPatch,
    parity_field: str,
) -> None:
    result = _factorial_protocol_result(monkeypatch)
    forged = result.as_dict()
    forged[parity_field]["compared_values"] = 0

    with pytest.raises(ValueError, match="compared_values must be a positive integer"):
        cf.CandidateCampaignFactorialResult.from_dict(forged)


def test_factorial_rejects_truthy_string_in_serialized_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _factorial_protocol_result(monkeypatch)
    forged = result.as_dict()
    forged["compiled_union_parity"]["allclose"] = "false"

    with pytest.raises(ValueError, match="parity.allclose must be a boolean"):
        cf.CandidateCampaignFactorialResult.from_dict(forged)


def test_factorial_direct_invariant_rejects_vacuous_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _factorial_protocol_result(monkeypatch)
    vacuous = replace(result.compiled_union_parity, compared_values=0)

    with pytest.raises(ValueError, match="factorial parity evidence is inconsistent"):
        replace(result, compiled_union_parity=vacuous)


@pytest.mark.parametrize("engine_type", (_CampaignEngine, _DenseCampaignEngine))
def test_real_factorial_campaign_executes_and_round_trips(
    engine_type: type[_CampaignEngine],
) -> None:
    engine = engine_type()
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    result = cf.benchmark_candidate_campaign_factorial(
        engine,
        campaign,
        token_ids,
        warmup=0,
        trials=2,
        minimum_improvement_percent=0.0,
    )

    assert result.query_count == 2
    assert result.reported_fabric == ("cuda" if engine.backend == "dense-qstore-cuda" else "cpu")
    assert result.full_head_union_parity.exact
    assert result.selected_independent_parity.exact
    assert result.compiled_union_parity.exact
    assert len(result.full_head_independent.samples_ms) == 2
    assert (
        cf.CandidateCampaignFactorialResult.from_dict(result.as_dict()).as_dict()
        == result.as_dict()
    )


def test_three_leg_campaign_protocol_rotates_legs_and_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    events: list[tuple[str, int | None]] = []
    engine = _Engine()
    output = (
        {
            "winner_token_id": 7,
            "runner_up_token_id": 8,
            "winner_logit": 3.0,
            "runner_up_logit": 2.0,
            "margin": 1.0,
        },
    )

    def independent(rotation: int) -> Any:
        events.append(("independent", rotation))
        clock[0] += 0.003
        engine.store.max_block_bytes = 2_000_000
        return output

    def manual_union() -> Any:
        events.append(("manual_union", None))
        clock[0] += 0.002
        engine.store.max_block_bytes = 2_000_000
        return output

    def compiled() -> Any:
        events.append(("compiled", None))
        clock[0] += 0.001
        engine.store.max_block_bytes = 2_000_000
        return output

    monkeypatch.setattr(cb.time, "perf_counter", lambda: clock[0])
    result = cb._run_paired_campaign_benchmark(
        engine,
        cb._CampaignRunners(
            independent=independent,
            manual_union=manual_union,
            compiled=compiled,
        ),
        campaign_fingerprint=_CAMPAIGN_FP,
        compilation_fingerprint=_COMPILATION_FP,
        rewrite_ids=("share-body", "union-head"),
        query_count=3,
        reference_candidate_count=9,
        union_candidate_count=5,
        duplicate_candidate_reference_count=4,
        shared_candidate_token_count=3,
        max_candidate_multiplicity=3,
        overlap_edge_count=3,
        overlap_component_count=1,
        reported_fabric="cpu",
        warmup=1,
        trials=4,
        rtol=1e-5,
        atol=2e-5,
        minimum_improvement_percent=1.0,
    )

    # Correctness, one warmup rotation, then four timed triplets.
    assert events[-12:] == [
        ("independent", 0),
        ("manual_union", None),
        ("compiled", None),
        ("manual_union", None),
        ("compiled", None),
        ("independent", 1),
        ("compiled", None),
        ("independent", 2),
        ("manual_union", None),
        ("independent", 0),
        ("manual_union", None),
        ("compiled", None),
    ]
    assert result.independent.samples_ms == pytest.approx((3.0,) * 4)
    assert result.manual_union.samples_ms == pytest.approx((2.0,) * 4)
    assert result.compiled.samples_ms == pytest.approx((1.0,) * 4)
    assert result.independent_to_compiled.median_ratio == pytest.approx(3.0)
    assert result.independent_to_compiled.ci95 == pytest.approx((3.0, 3.0))
    assert result.independent_to_compiled.wins == 4
    assert result.independent_manual_parity.exact
    assert result.independent_compiled_parity.exact
    assert result.manual_compiled_parity.exact
    assert result.improvement_demonstrated
    assert result.independent.max_dequant_block_mb == pytest.approx(2.0)
    json.dumps(result.as_dict())
    assert cb.CandidateCampaignBenchmarkResult.from_dict(result.as_dict()) == result
    assert json.loads(result.to_json()) == result.as_dict()


def test_campaign_promotion_requires_exact_manual_compiled_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]

    def timed(value: float, output: tuple[float, ...]) -> Any:
        clock[0] += value
        return output

    monkeypatch.setattr(cb.time, "perf_counter", lambda: clock[0])
    result = cb._run_paired_campaign_benchmark(
        _Engine(),
        cb._CampaignRunners(
            independent=lambda _rotation: timed(0.003, (1.0,)),
            manual_union=lambda: timed(0.002, (1.0,)),
            compiled=lambda: timed(0.001, (1.0 + 1e-7,)),
        ),
        campaign_fingerprint=_CAMPAIGN_FP,
        compilation_fingerprint=_COMPILATION_FP,
        rewrite_ids=(),
        query_count=2,
        reference_candidate_count=4,
        union_candidate_count=3,
        duplicate_candidate_reference_count=1,
        shared_candidate_token_count=1,
        max_candidate_multiplicity=2,
        overlap_edge_count=1,
        overlap_component_count=1,
        reported_fabric="cpu",
        warmup=0,
        trials=2,
        rtol=1e-5,
        atol=2e-5,
        minimum_improvement_percent=1.0,
    )

    assert result.independent_compiled_parity.allclose
    assert not result.manual_compiled_parity.exact
    assert not result.improvement_demonstrated


def test_real_candidate_campaign_runs_three_graph_bound_legs() -> None:
    engine = _CampaignEngine()
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())
    engine.selected_body_calls = 0
    engine.selected_rows.clear()

    result = cb.benchmark_candidate_campaign(
        engine,
        campaign,
        token_ids,
        warmup=0,
        trials=3,
        minimum_improvement_percent=0.0,
    )

    assert result.query_count == 2
    assert result.reference_candidate_count == 6
    assert result.union_candidate_count == 4
    assert result.duplicate_candidate_reference_count == 2
    assert result.shared_candidate_token_count == 2
    assert result.max_candidate_multiplicity == 2
    assert result.overlap_edge_count == 1
    assert result.overlap_component_count == 1
    assert result.independent_manual_parity.exact
    assert result.independent_compiled_parity.exact
    assert result.manual_compiled_parity.exact
    assert len(result.independent.samples_ms) == 3
    assert len(result.manual_union.samples_ms) == 3
    assert len(result.compiled.samples_ms) == 3

    # Correctness plus three measured triplets: each independent campaign executes
    # two bodies, while manual and compiled each execute one stable union.
    assert engine.selected_body_calls == 16
    union = campaign.union_token_ids
    assert engine.selected_rows[:4] == [
        _readouts()[0].candidate_token_ids,
        _readouts()[1].candidate_token_ids,
        union,
        union,
    ]
    assert engine.selected_rows[8:12] == [
        union,
        union,
        _readouts()[1].candidate_token_ids,
        _readouts()[0].candidate_token_ids,
    ]
    restored = cb.CandidateCampaignBenchmarkResult.from_dict(result.as_dict())
    assert restored == result

    tampered = result.as_dict()
    tampered["compiled"]["median_ms"] += 1.0
    with pytest.raises(ValueError, match="does not match campaign samples"):
        cb.CandidateCampaignBenchmarkResult.from_dict(tampered)


def test_dense_cuda_candidate_campaign_runs_the_same_three_leg_protocol() -> None:
    engine = _DenseCampaignEngine()
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    result = cb.benchmark_candidate_campaign(
        engine,
        campaign,
        token_ids,
        warmup=0,
        trials=2,
        minimum_improvement_percent=0.0,
    )

    assert campaign.base_bundle.lowered.backend == "cuda-qstore"
    assert result.reported_fabric == "cuda"
    assert result.independent_manual_parity.exact
    assert result.independent_compiled_parity.exact
    assert result.manual_compiled_parity.exact


@pytest.mark.parametrize(
    ("query_count", "reference_count", "union_count", "match"),
    [
        (0, 2, 2, "at least one query"),
        (1, 1, 2, "cannot exceed"),
        (1, 1, 1, "at least two"),
    ],
)
def test_campaign_benchmark_rejects_invalid_counts(
    query_count: int,
    reference_count: int,
    union_count: int,
    match: str,
) -> None:
    runners = cb._CampaignRunners(
        independent=lambda _rotation: (),
        manual_union=lambda: (),
        compiled=lambda: (),
    )
    with pytest.raises(ValueError, match=match):
        cb._run_paired_campaign_benchmark(
            _Engine(),
            runners,
            campaign_fingerprint=_CAMPAIGN_FP,
            compilation_fingerprint=_COMPILATION_FP,
            rewrite_ids=(),
            query_count=query_count,
            reference_candidate_count=reference_count,
            union_candidate_count=union_count,
            duplicate_candidate_reference_count=max(
                0,
                reference_count - union_count,
            ),
            shared_candidate_token_count=0,
            max_candidate_multiplicity=1,
            overlap_edge_count=0,
            overlap_component_count=max(1, query_count),
            reported_fabric="cpu",
            warmup=0,
            trials=1,
            rtol=0.0,
            atol=0.0,
            minimum_improvement_percent=0.0,
        )
