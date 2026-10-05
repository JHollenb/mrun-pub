from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mrun.engine as engine_module
from mrun.engine._base_impl import BaseEngine
from mrun.engine.ane import ANEPagedEngine
from mrun.engine.paged import PagedEngine
from mrun.engine.report import EngineReport


class _FakeEngine(BaseEngine):
    backend = "fake"
    name = "toy"
    arch = "toy"
    device = "cpu"
    supports_batch = False
    numerical_contract = "exact"

    def __init__(self) -> None:
        self.close_calls = 0
        self._evidence: dict[str, Any] = {}

    def close(self) -> None:
        self.close_calls += 1

    def logits(self, ids: np.ndarray) -> torch.Tensor:
        return torch.zeros((len(ids), 3))

    def execution_evidence(self) -> dict[str, Any]:
        return dict(self._evidence)


@pytest.fixture(autouse=True)
def _clean_engine_pool(monkeypatch: pytest.MonkeyPatch):
    engine_module.close_pooled_engines()
    engine_module._OPENED.clear()
    monkeypatch.delenv("MRUN_ENGINE_REPORT", raising=False)
    monkeypatch.delenv("MRUN_JOB_ID", raising=False)
    yield
    engine_module.close_pooled_engines()
    engine_module._OPENED.clear()


def test_engine_pool_survives_context_manager_and_closes_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[_FakeEngine] = []

    def fake_open(*_args: Any, **_kwargs: Any) -> _FakeEngine:
        engine = _FakeEngine()
        opened.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    with engine_module.open_engine("toy", backend="hf") as first:
        assert first.close_calls == 0

    second = engine_module.open_engine("toy", backend="hf")
    assert second is first
    assert first.close_calls == 0
    assert len(opened) == 1

    engine_module.close_pooled_engines()
    assert first.close_calls == 1


def test_engine_pool_disables_reuse_for_unhashable_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[_FakeEngine] = []

    def fake_open(*_args: Any, **_kwargs: Any) -> _FakeEngine:
        engine = _FakeEngine()
        opened.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    first = engine_module.open_engine("toy", backend="hf", marker=[])
    second = engine_module.open_engine("toy", backend="hf", marker=[])
    assert first is not second
    assert len(opened) == 2


def test_apple_alias_shares_the_mlx_pool_key(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[_FakeEngine] = []

    def fake_open(*_args: Any, **_kwargs: Any) -> _FakeEngine:
        engine = _FakeEngine()
        opened.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    apple = engine_module.open_engine("toy", backend="apple")
    mlx = engine_module.open_engine("toy", backend="mlx")
    assert apple is mlx
    assert len(opened) == 1


def test_report_refreshes_runtime_facts_and_records_probe_throughput(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    report = EngineReport(
        engine,
        run_id="probe-run",
        started_at=time.perf_counter() - 0.01,
    )
    engine._evidence = {
        "compiled_shapes": [[8, 9]],
        "placement_verified": False,
    }
    engine._note_scoring("subset_head", probes=8, rows=8, seconds=0.02)

    summary = report.summary()
    assert summary["engine"]["execution_evidence"]["compiled_shapes"] == [[8, 9]]
    assert summary["engine"]["scoring_paths"]["subset_head"]["probes_per_s"] == 400.0

    leaderboard = tmp_path / "leaderboard.jsonl"
    monkeypatch.setenv("MRUN_LEADERBOARD", str(leaderboard))
    report.save(tmp_path, print_graph=False)
    row = json.loads(leaderboard.read_text().strip())
    assert row["tok_per_s"] is None
    assert row["probes_per_s"] == 400.0
    assert row["scoring_path"] == "subset_head"


def test_margin_floor_flags_rows_without_changing_winners() -> None:
    engine = _FakeEngine()
    engine.margin_floor = 0.5
    result = engine._flag_low_margin_rows(
        {
            "summary": {"winner_count": 2},
            "rows": [
                {"winner": "a", "margin": 0.2},
                {"winner": "b", "margin": 0.8},
            ],
        }
    )
    assert result["rows"][0]["margin_below_backend_error"] is True
    assert "margin_below_backend_error" not in result["rows"][1]
    assert result["summary"]["rows_below_backend_margin_error"] == 1
    assert [row["winner"] for row in result["rows"]] == ["a", "b"]


def test_cached_coreml_package_reuses_requested_compute_units(tmp_path: Path) -> None:
    package = tmp_path / "cached.mlpackage"
    package.mkdir()
    calls: list[tuple[str, Any]] = []
    loaded_model = object()

    class Models:
        @staticmethod
        def MLModel(path: str, *, compute_units: Any):
            calls.append((path, compute_units))
            return loaded_model

    engine = object.__new__(ANEPagedEngine)
    engine._ct = type("FakeCoreML", (), {"models": Models})()
    engine._cache = OrderedDict()
    engine._cache_size = 2
    engine._compiled_shapes = set()
    engine._disk_cache_hits = 0
    engine._disk_cache_misses = 0
    engine._disk_cache_dir = lambda _batch, _tokens: package
    engine._compute_units = lambda: "CPU_AND_NE"

    assert engine._get_model(8, 9) is loaded_model
    assert calls == [(str(package), "CPU_AND_NE")]
    assert engine._disk_cache_hits == 1


def test_coreml_single_logits_uses_accelerated_batch_surface() -> None:
    engine = object.__new__(ANEPagedEngine)
    expected = torch.ones((3, 5))
    calls: list[list[np.ndarray]] = []
    engine.logits_batch = lambda rows: calls.append(rows) or [expected]

    result = engine.logits(np.asarray([1, 2, 3]))
    assert result is expected
    assert len(calls) == 1
    assert ANEPagedEngine.score_forced_choice_argmax_subset is None


def test_coreml_generation_records_paged_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = object.__new__(ANEPagedEngine)
    engine._last_fallback = False
    monkeypatch.setattr(PagedEngine, "generate", lambda *_args, **_kwargs: [7, 8])

    assert engine.generate([1, 2], max_new_tokens=2) == [7, 8]
    assert engine._last_fallback is True
    assert engine._last_execution_path == "paged-generation"


def test_coreml_tap_surface_records_paged_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = object.__new__(ANEPagedEngine)
    engine._last_fallback = False
    expected = (torch.zeros((1, 1)), [])
    monkeypatch.setattr(PagedEngine, "forward_acts", lambda *_args, **_kwargs: expected)

    assert engine.forward_acts(np.asarray([1])) is expected
    assert engine._last_fallback is True
    assert engine._last_execution_path == "paged-forward-acts"
