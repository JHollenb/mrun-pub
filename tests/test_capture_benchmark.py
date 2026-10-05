from __future__ import annotations

import json
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mrun.compiler import capture_benchmark as cb
from mrun.compiler.campaign import CandidateReadout, compile_candidate_campaign
from mrun.testing.qstore_identity import (
    install_verified_test_identity,
    verified_test_manifest,
)


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
    return verified_test_manifest(
        {
            "model_name": "tiny-qwen",
            "arch": "qwen2",
            "dtype": "int8",
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
    )


class _Store:
    compute_dtype = torch.bfloat16

    def __init__(self) -> None:
        self.man = _manifest()
        install_verified_test_identity(self)

    def has(self, name: str) -> bool:
        return name in self.man["blocks"]


class _FakeCaptureExecutor:
    def __init__(
        self,
        engine: _Engine,
        ids_list: list[np.ndarray],
        token_ids: tuple[int, ...],
        warmup: int,
    ) -> None:
        self._engine = engine
        self._ids = tuple(np.asarray(ids).copy() for ids in ids_list)
        self._token_ids = token_ids
        self._warmup = warmup
        self._replays = 0
        self._controls = 0
        self.closed = False

    def _scores(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.as_tensor(
                    [float(ids.sum() + 2 * token) for token in self._token_ids],
                    dtype=torch.float32,
                )
                for ids in self._ids
            ]
        )

    def execute_eager_control(self) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("closed")
        self._controls += 1
        self._engine.events.append("matched_eager_control")
        self._engine.advance(0.004)
        return self._scores()

    def execute(self) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("closed")
        self._replays += 1
        self._engine.events.append("cuda_graph")
        self._engine.advance(0.001)
        return self._scores()

    @property
    def evidence(self) -> MappingProxyType:
        return MappingProxyType(
            {
                "graph_replay": True,
                "capture_ready": True,
                "capture_executed": True,
                "capture_backend": "fake-torch.cuda.CUDAGraph",
                "graph_backend": "fake-torch.cuda.CUDAGraph",
                "capture_mode": "selected-last-stateless-score",
                "capture_warmup_iterations": self._warmup,
                "capture_warmups": self._warmup,
                "capture_count": 1,
                "capture_static_shapes": True,
                "capture_stable_addresses": True,
                "stable_addresses_verified": True,
                "capture_resource_addresses_verified": True,
                "capture_graph_safe": True,
                "capture_input_shape": [len(self._ids), len(self._ids[0])],
                "capture_output_shape": [len(self._ids), len(self._token_ids)],
                "capture_output_dtype": "fp32",
                "capture_full_logits": False,
                "capture_stateful_kv": False,
                "capture_decode": False,
                "capture_matched_eager_control_available": True,
                "capture_matched_eager_control_basis": cb._MATCHED_CONTROL_BASIS,
                "capture_resident_bytes": 4096,
                "capture_estimated_resident_bytes": 4096,
                "capture_residency_budget_bytes": 8192,
                "capture_qrow_logical_count": 11,
                "capture_qrow_physical_count": 10,
                "capture_fp32_logical_count": 6,
                "capture_fp32_physical_count": 6,
                "capture_aliases_deduplicated": 1,
                "capture_selected_head_rows": len(self._token_ids),
                "capture_stable_address_count": 19,
                "capture_residency_non_evictable": True,
                "capture_setup_ms": 2.5,
                "capture_total_allocated_delta_bytes": 4096,
                "capture_total_reserved_delta_bytes": 4096,
                "capture_peak_allocated_delta_bytes": 4096,
                "capture_total_residency_delta_bytes": 4096,
                "capture_replay_count": self._replays,
                "capture_matched_eager_control_count": self._controls,
                "capture_executor_closed": self.closed,
            }
        )

    def close(self) -> None:
        self.closed = True


class _Engine:
    backend = "dense-qstore-cuda"
    device = torch.device("cuda")
    name = "tiny-qwen"
    n_layer = 1
    hidden = 4
    inter = 8
    max_seq_len = 16
    numerical_contract = "torch-batched-established"
    subset_head_numerical_contract = "torch-batched-established+selected-head-fp32-v1"
    supported_numerical_contracts = (
        numerical_contract,
        subset_head_numerical_contract,
    )
    cuda_graph_runtime_environment = {
        "runtime_source_sha256": "7" * 64,
        "runtime_distribution_sha256": "6" * 64,
        "wheel_sha256": "8" * 64,
        "torch_version": "test",
        "torch_cuda_version": "test",
        "cuda_driver_version": 1,
        "triton_version": "test",
        "device": "cuda:0",
        "gpu_name": "fake-cuda",
        "gpu_capability": [8, 9],
        "gpu_total_memory_bytes": 16_000_000_000,
        "store_reverified_at_ns": 1,
    }

    def __init__(self, advance: Any) -> None:
        self.store = _Store()
        self.spec = SimpleNamespace(name=self.name)
        self.advance = advance
        self.events: list[str] = []
        self.prepare_calls = 0
        self.ordinary_eager_calls = 0
        self.last_executor: _FakeCaptureExecutor | None = None

    def build_work_plan(
        self,
        ids_list: list[np.ndarray],
        **kwargs: Any,
    ) -> Any:
        from mrun.compiler import build_dense_qstore_plan

        return build_dense_qstore_plan(self, ids_list, **kwargs)

    def selected_last_logits_batch(
        self,
        ids_list: list[np.ndarray],
        token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        self.ordinary_eager_calls += 1
        self.advance(0.009)
        return torch.stack(
            [
                torch.as_tensor(
                    [float(np.asarray(ids).sum() + 2 * token) for token in token_ids],
                    dtype=torch.float32,
                )
                for ids in ids_list
            ]
        )

    def prepare_selected_last_cuda_graph(
        self,
        ids_list: list[np.ndarray],
        token_ids: tuple[int, ...],
        *,
        warmup: int,
    ) -> _FakeCaptureExecutor:
        self.prepare_calls += 1
        self.events.append("prepare")
        self.last_executor = _FakeCaptureExecutor(
            self,
            ids_list,
            tuple(token_ids),
            warmup,
        )
        return self.last_executor


class _MissingMethodEngine(_Engine):
    def __init__(self, missing_method: str) -> None:
        super().__init__(lambda _seconds: None)
        self.missing_method = missing_method

    def prepare_selected_last_cuda_graph(
        self,
        ids_list: list[np.ndarray],
        token_ids: tuple[int, ...],
        *,
        warmup: int,
    ) -> _FakeCaptureExecutor:
        executor = super().prepare_selected_last_cuda_graph(
            ids_list,
            token_ids,
            warmup=warmup,
        )
        setattr(executor, self.missing_method, None)
        return executor


def _readouts() -> tuple[CandidateReadout, ...]:
    return (
        CandidateReadout("alpha", (5, 1, 3)),
        CandidateReadout("beta", (3, 7, 1)),
    )


def _benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[cb.CudaGraphCampaignBenchmarkResult, _Engine]:
    clock = [0.0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    engine = _Engine(advance)
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(
        engine,
        token_ids,
        _readouts(),
        capture_requested=True,
    )
    monkeypatch.setattr(cb.time, "perf_counter", lambda: clock[0])
    result = cb.benchmark_cuda_graph_campaign(
        engine,
        campaign,
        token_ids,
        capture_warmup=2,
        warmup=1,
        trials=4,
        minimum_improvement_percent=1.0,
    )
    return result, engine


def test_matched_control_isolates_graph_replay_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, engine = _benchmark(monkeypatch)

    assert engine.prepare_calls == 1
    assert engine.ordinary_eager_calls == 0
    assert engine.last_executor is not None and engine.last_executor.closed
    assert engine.events[-8:] == [
        "matched_eager_control",
        "cuda_graph",
        "cuda_graph",
        "matched_eager_control",
        "matched_eager_control",
        "cuda_graph",
        "cuda_graph",
        "matched_eager_control",
    ]
    assert result.matched_eager_control.samples_ms == pytest.approx((4.0,) * 4)
    assert result.cuda_graph.samples_ms == pytest.approx((1.0,) * 4)
    assert result.matched_eager_to_cuda_graph.ratios == pytest.approx((4.0,) * 4)
    assert result.speedup == pytest.approx(4.0)
    assert result.matched_eager_to_cuda_graph.ci95 == pytest.approx((4.0, 4.0))
    assert result.matched_eager_to_cuda_graph.wins == 4
    assert result.output_parity.exact
    assert result.capture_improvement_demonstrated
    assert result.capture_setup_ms == pytest.approx(2.5)
    assert result.per_replay_savings_ms == pytest.approx(3.0)
    assert result.break_even_replays == 1
    assert result.runtime_environment["gpu_name"] == "fake-cuda"
    assert result.capture_executor_evidence["capture_replay_count"] == 6
    assert result.capture_executor_evidence["capture_matched_eager_control_count"] == 6
    assert result.capture_executor_evidence["capture_warmup_iterations"] == 2
    assert result.matched_eager_control_runtime_evidence["graph_replay"] is False
    assert result.cuda_graph_runtime_evidence["graph_replay"] is True
    assert (
        cb.CudaGraphCampaignBenchmarkResult.from_json(result.to_json()).as_dict()
        == result.as_dict()
    )
    json.dumps(result.as_dict())


def test_compile_and_benchmark_uses_one_capture_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    engine = _Engine(advance)
    monkeypatch.setattr(cb.time, "perf_counter", lambda: clock[0])
    result = cb.compile_and_benchmark_cuda_graph_campaign(
        engine,
        np.asarray([1, 2, 3], dtype=np.int64),
        _readouts(),
        capture_warmup=1,
        warmup=0,
        trials=2,
    )

    assert engine.prepare_calls == 1
    assert result.capture_executor_evidence["capture_replay_count"] == 3
    assert result.capture_executor_evidence["capture_matched_eager_control_count"] == 3


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        (
            "matched_eager_control_runtime_evidence",
            "graph_replay",
            True,
            "cannot claim graph_replay",
        ),
        (
            "matched_eager_control_runtime_evidence",
            "capture_executed",
            True,
            "cannot claim capture_executed",
        ),
        (
            "cuda_graph_runtime_evidence",
            "graph_replay",
            False,
            "must expose graph_replay",
        ),
        (
            "capture_executor_evidence",
            "capture_matched_eager_control_available",
            False,
            "capture_matched_eager_control_available=true",
        ),
        (
            "capture_executor_evidence",
            "capture_executor_closed",
            True,
            "closed before evidence snapshot",
        ),
        (
            "capture_executor_evidence",
            "capture_executed",
            "true",
            "capture_executed=true",
        ),
    ),
)
def test_serialized_runtime_evidence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value: Any,
    message: str,
) -> None:
    result, _ = _benchmark(monkeypatch)
    forged = result.as_dict()
    forged[section][key] = value

    with pytest.raises(ValueError, match=message):
        cb.CudaGraphCampaignBenchmarkResult.from_dict(forged)


def test_serialized_metrics_and_parity_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _benchmark(monkeypatch)

    forged = result.as_dict()
    forged["matched_eager_to_cuda_graph"]["wins"] = 0
    with pytest.raises(ValueError, match="win count"):
        cb.CudaGraphCampaignBenchmarkResult.from_dict(forged)

    forged = result.as_dict()
    forged["output_parity"]["allclose"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        cb.CudaGraphCampaignBenchmarkResult.from_dict(forged)

    forged = result.as_dict()
    forged["capture_improvement_demonstrated"] = False
    with pytest.raises(ValueError, match="improvement verdict"):
        cb.CudaGraphCampaignBenchmarkResult.from_dict(forged)

    forged = result.as_dict()
    forged["matched_eager_to_cuda_graph"]["latency_reduction_percent"] = 9_876.5
    forged["latency_reduction_percent"] = 9_876.5
    with pytest.raises(ValueError, match="latency reduction"):
        cb.CudaGraphCampaignBenchmarkResult.from_dict(forged)


def test_rejects_non_capture_campaign_before_preparation() -> None:
    engine = _Engine(lambda _seconds: None)
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(
        engine,
        token_ids,
        _readouts(),
        capture_requested=False,
    )

    with pytest.raises(ValueError, match="must request graph capture"):
        cb.benchmark_cuda_graph_campaign(
            engine,
            campaign,
            token_ids,
            warmup=0,
            trials=1,
        )
    assert engine.prepare_calls == 0


def test_rejects_runtime_engine_binding_drift_before_capture_preparation() -> None:
    source = _Engine(lambda _seconds: None)
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(
        source,
        token_ids,
        _readouts(),
        capture_requested=True,
    )
    wrong = _Engine(lambda _seconds: None)
    wrong.backend = "paged"

    with pytest.raises(RuntimeError, match="backend does not match"):
        cb.benchmark_cuda_graph_campaign(
            wrong,
            campaign,
            token_ids,
            warmup=0,
            trials=1,
        )
    assert wrong.prepare_calls == 0


@pytest.mark.parametrize(
    ("method", "message"),
    (
        ("execute_eager_control", "no matched eager-control method"),
        ("execute", "no replay method"),
    ),
)
def test_missing_executor_method_fails_closed_and_releases_resources(
    method: str,
    message: str,
) -> None:
    engine = _MissingMethodEngine(method)
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(
        engine,
        token_ids,
        _readouts(),
        capture_requested=True,
    )

    with pytest.raises(RuntimeError, match=message):
        cb.benchmark_cuda_graph_campaign(
            engine,
            campaign,
            token_ids,
            warmup=0,
            trials=1,
        )
    assert engine.last_executor is not None and engine.last_executor.closed
