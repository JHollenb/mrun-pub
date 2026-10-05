from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import mrun.engine as engine_module
from mrun.engine import paged as paged_module
from mrun.engine.kernels import composite_qstore


class _GuardedEngine:
    def __init__(
        self,
        *,
        fail_guard: bool = False,
        component_fingerprint: str | None = None,
    ) -> None:
        self.fail_guard = fail_guard
        self.guard_calls = 0
        self.close_calls = 0
        self.composite_store = (
            SimpleNamespace(composite_fingerprint_sha256=component_fingerprint)
            if component_fingerprint is not None
            else None
        )

    def assert_content_identity_unchanged(self) -> None:
        self.guard_calls += 1
        if self.fail_guard:
            raise RuntimeError("component artifact changed")

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _clean_pool() -> None:
    engine_module.close_pooled_engines()
    engine_module._OPENED.clear()
    yield
    engine_module.close_pooled_engines()
    engine_module._OPENED.clear()


def test_composite_pool_uses_graph_identity_and_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[_GuardedEngine] = []
    monkeypatch.setattr(
        composite_qstore,
        "inspect_component_graph_fingerprint",
        lambda _path: "a" * 64,
    )

    def fake_open(*_args: Any, **_kwargs: Any) -> _GuardedEngine:
        engine = _GuardedEngine(component_fingerprint="a" * 64)
        opened.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    first = engine_module.open_engine(
        "toy",
        backend="paged",
        component_graph="first/model-graph.json",
        component_cache_mb={"body": 2.0, "norm": 1.0},
    )
    same = engine_module.open_engine(
        "toy",
        backend="paged",
        component_graph="relocated/model-graph.json",
        component_cache_mb={"norm": 1.0, "body": 2.0},
        output_contract="full_logits",
    )
    selected = engine_module.open_engine(
        "toy",
        backend="paged",
        component_graph="first/model-graph.json",
        component_cache_mb={"body": 2.0, "norm": 1.0},
        output_contract="selected_token_rows",
    )

    assert same is first
    assert selected is not first
    assert first.guard_calls == 1
    assert len(opened) == 2


@pytest.mark.parametrize(
    "backend",
    [
        "mlx-component",
        "mlx-component-q4",
        "metal-component",
        "metal-component-q4",
    ],
)
def test_native_mlx_component_pool_uses_verified_graph_identity(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    opened: list[_GuardedEngine] = []
    monkeypatch.setattr(
        composite_qstore,
        "inspect_component_graph_fingerprint",
        lambda _path: "a" * 64,
    )

    def fake_open(*_args: Any, **_kwargs: Any) -> _GuardedEngine:
        engine = _GuardedEngine()
        engine.graph = SimpleNamespace(custody_fingerprint_sha256="a" * 64)
        opened.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    first = engine_module.open_engine(
        "toy",
        backend=backend,
        component_graph="first/model-graph.json",
        native_artifact="native/artifact",
    )
    relocated = engine_module.open_engine(
        "toy",
        backend=backend,
        component_graph="relocated/model-graph.json",
        native_artifact="native/artifact",
    )

    assert relocated is first
    assert len(opened) == 1


@pytest.mark.parametrize("backend", ["mlx-component", "metal-component"])
def test_native_mlx_component_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    import mrun.engine.mlx_component as component_module

    calls: list[tuple[str, dict[str, Any]]] = []
    expected = object()

    def fake_engine(model_name: str, **kwargs: Any) -> object:
        calls.append((model_name, kwargs))
        return expected

    monkeypatch.setattr(component_module, "MLXComponentEngine", fake_engine)
    result = engine_module._open_engine_impl(
        "toy",
        backend=backend,
        component_graph="model-graph.json",
    )

    assert result is expected
    assert calls == [("toy", {"component_graph": "model-graph.json"})]


@pytest.mark.parametrize("backend", ["mlx-component-q4", "metal-component-q4"])
def test_native_mlx_component_q4_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    import mrun.engine.mlx_component as component_module

    calls: list[tuple[str, dict[str, Any]]] = []
    expected = object()

    def fake_engine(model_name: str, **kwargs: Any) -> object:
        calls.append((model_name, kwargs))
        return expected

    monkeypatch.setattr(component_module, "MLXComponentQ4Engine", fake_engine)
    result = engine_module._open_engine_impl(
        "toy",
        backend=backend,
        component_graph="model-graph.json",
    )

    assert result is expected
    assert calls == [("toy", {"component_graph": "model-graph.json"})]


def test_dense_component_pool_uses_graph_identity_and_component_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[_GuardedEngine] = []
    monkeypatch.setattr(
        composite_qstore,
        "inspect_component_graph_fingerprint",
        lambda _path: "e" * 64,
    )

    def fake_open(*_args: Any, **_kwargs: Any) -> _GuardedEngine:
        engine = _GuardedEngine(component_fingerprint="e" * 64)
        opened.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    first = engine_module.open_engine(
        "toy",
        backend="dense-cuda",
        component_graph="first/model-graph.json",
        component_cache_mb={"body": 8.0, "egress": 1.0},
    )
    relocated = engine_module.open_engine(
        "toy",
        backend="dense-cuda",
        component_graph="relocated/model-graph.json",
        component_cache_mb={"egress": 1.0, "body": 8.0},
        output_contract="full_logits",
    )

    assert relocated is first
    assert first.guard_calls == 1
    assert len(opened) == 1


def test_composite_pool_rejects_changed_open_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite_qstore,
        "inspect_component_graph_fingerprint",
        lambda _path: "b" * 64,
    )
    engine = _GuardedEngine(fail_guard=True, component_fingerprint="b" * 64)
    monkeypatch.setattr(engine_module, "_open_engine_impl", lambda *_args, **_kwargs: engine)

    engine_module.open_engine(
        "toy",
        backend="paged",
        component_graph="model-graph.json",
    )
    with pytest.raises(RuntimeError, match="artifact changed"):
        engine_module.open_engine(
            "toy",
            backend="paged",
            component_graph="model-graph.json",
        )


def test_component_pool_rejects_mutation_between_inspection_and_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspected = "c" * 64
    mutated = "d" * 64
    graph_state = {"fingerprint": inspected}
    opened: list[_GuardedEngine] = []

    monkeypatch.setattr(
        composite_qstore,
        "inspect_component_graph_fingerprint",
        lambda _path: graph_state["fingerprint"],
    )

    def mutate_then_open(*_args: Any, **_kwargs: Any) -> _GuardedEngine:
        # Deterministically model a valid graph rewrite after pool-key inspection but before
        # PagedEngine opens it. The new graph has its own valid production fingerprint.
        graph_state["fingerprint"] = mutated
        engine = _GuardedEngine(component_fingerprint=mutated)
        opened.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "_open_engine_impl", mutate_then_open)

    with pytest.raises(RuntimeError, match="changed between pool-key inspection and engine open"):
        engine_module.open_engine(
            "toy",
            backend="paged",
            component_graph="model-graph.json",
        )

    assert opened[0].close_calls == 1
    assert engine_module._POOL == {}
    assert engine_module._OPENED == []


def test_concurrent_pool_open_constructs_exactly_one_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_count = 8
    start = threading.Barrier(worker_count)
    counter_lock = threading.Lock()
    open_count = 0

    def fake_open(*_args: Any, **_kwargs: Any) -> _GuardedEngine:
        nonlocal open_count
        with counter_lock:
            open_count += 1
        time.sleep(0.02)
        return _GuardedEngine()

    def open_one() -> _GuardedEngine:
        start.wait()
        return engine_module.open_engine(
            "toy",
            backend="paged",
            stores_dir="stores",
        )

    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        engines = list(executor.map(lambda _index: open_one(), range(worker_count)))

    assert open_count == 1
    assert all(engine is engines[0] for engine in engines)


def test_pool_close_excludes_cold_open_until_real_close_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement engine cannot overlap the old engine's resource teardown."""

    class ObservedRLock:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._state_lock = threading.Lock()
            self._owner: int | None = None
            self._depth = 0
            self.opener_attempted = threading.Event()
            self.opener_was_contended: list[bool] = []

        def __enter__(self) -> ObservedRLock:
            ident = threading.get_ident()
            with self._state_lock:
                contended = self._owner is not None and self._owner != ident
                if threading.current_thread().name == "pool-opener":
                    self.opener_was_contended.append(contended)
                    self.opener_attempted.set()
            self._lock.acquire()
            with self._state_lock:
                if self._owner == ident:
                    self._depth += 1
                else:
                    self._owner = ident
                    self._depth = 1
            return self

        def __exit__(self, *_exc: object) -> None:
            ident = threading.get_ident()
            with self._state_lock:
                assert self._owner == ident
                self._depth -= 1
                if self._depth == 0:
                    self._owner = None
            self._lock.release()

    close_started = threading.Event()
    allow_close = threading.Event()
    close_finished = threading.Event()
    lifecycle: list[str] = []

    class BlockingCloseEngine(_GuardedEngine):
        def close(self) -> None:
            lifecycle.append("close-start")
            close_started.set()
            if not allow_close.wait(timeout=5.0):
                raise TimeoutError("test did not release blocking engine close")
            lifecycle.append("close-finish")
            close_finished.set()
            super().close()

    opened: list[_GuardedEngine] = []

    def fake_open(*_args: Any, **_kwargs: Any) -> _GuardedEngine:
        if not opened:
            engine: _GuardedEngine = BlockingCloseEngine()
        else:
            lifecycle.append("replacement-construct")
            engine = _GuardedEngine()
        opened.append(engine)
        return engine

    observed_lock = ObservedRLock()
    monkeypatch.setattr(engine_module, "_POOL_LOCK", observed_lock)
    monkeypatch.setattr(engine_module, "_open_engine_impl", fake_open)
    original = engine_module.open_engine("toy", backend="paged", stores_dir="stores")

    close_thread = threading.Thread(
        target=engine_module.close_pooled_engines,
        name="pool-closer",
    )
    opened_by_thread: list[_GuardedEngine] = []
    open_errors: list[BaseException] = []

    def open_replacement() -> None:
        try:
            opened_by_thread.append(
                engine_module.open_engine("toy", backend="paged", stores_dir="stores")
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            open_errors.append(exc)

    open_thread = threading.Thread(target=open_replacement, name="pool-opener")
    close_thread.start()
    assert close_started.wait(timeout=2.0)
    open_thread.start()
    try:
        assert observed_lock.opener_attempted.wait(timeout=2.0)
        # This is sampled while real_close is deliberately blocked. It makes the regression
        # deterministic without using a scheduler-dependent sleep to infer contention.
        assert observed_lock.opener_was_contended == [True]
    finally:
        allow_close.set()

    close_thread.join(timeout=2.0)
    open_thread.join(timeout=2.0)
    assert not close_thread.is_alive()
    assert not open_thread.is_alive()
    assert open_errors == []
    assert opened_by_thread and opened_by_thread[0] is not original
    assert close_finished.is_set()
    assert lifecycle == ["close-start", "close-finish", "replacement-construct"]


class _FakeComposite:
    def __init__(self, failure_stage: str) -> None:
        self.close_calls = 0
        self.graph = SimpleNamespace(
            model_name="other" if failure_stage == "model" else "toy",
            architecture="llama" if failure_stage == "architecture" else "qwen2",
        )
        self.store = SimpleNamespace(
            device="cpu",
            cfg={
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "hidden_size": 8,
                "vocab_size": 8,
            },
            man={"arch": "unsupported" if failure_stage == "forward" else "qwen2"},
            output_contract="full_logits",
        )
        self.vocab = SimpleNamespace(token_count=9 if failure_stage == "vocab" else 7)
        self.failure_stage = failure_stage

    def for_contract(self, _contract: Any) -> Any:
        return self.store

    def validate_tokenizer(self, _tokenizer: Any) -> None:
        if self.failure_stage == "tokenizer":
            raise RuntimeError("tokenizer rejected")

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    ("failure_stage", "error", "match"),
    [
        ("model", composite_qstore.ComponentGraphError, "model does not match"),
        ("architecture", composite_qstore.ComponentGraphError, "architecture does not match"),
        ("tokenizer-load", RuntimeError, "tokenizer load failed"),
        ("tokenizer", RuntimeError, "tokenizer rejected"),
        ("vocab", ValueError, "semantic token count"),
        ("forward", NotImplementedError, "no forward"),
    ],
)
def test_composite_engine_construction_closes_on_later_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    error: type[BaseException],
    match: str,
) -> None:
    composite = _FakeComposite(failure_stage)
    spec = SimpleNamespace(name="toy", family="qwen2", hf_id="org/toy")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __len__(self) -> int:
            return 7

    tokenizer = Tokenizer()
    monkeypatch.setattr(paged_module, "resolve_model", lambda _name: spec)
    if failure_stage == "tokenizer-load":

        def fail_tokenizer_load(_spec: Any) -> Any:
            raise RuntimeError("tokenizer load failed")

        monkeypatch.setattr(paged_module, "load_tokenizer", fail_tokenizer_load)
    else:
        monkeypatch.setattr(paged_module, "load_tokenizer", lambda _spec: tokenizer)
    monkeypatch.setattr(paged_module, "CompositeQStore", lambda *_args, **_kwargs: composite)

    with pytest.raises(error, match=match):
        paged_module.PagedEngine(
            "toy",
            component_graph="model-graph.json",
            cache_mb=0,
        )

    assert composite.close_calls == 1


def test_ordinary_qstore_construction_closes_on_later_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    store_dir = tmp_path / "toy"
    store_dir.mkdir()
    (store_dir / "manifest.json").write_text("{}", encoding="utf-8")
    spec = SimpleNamespace(name="toy", family="qwen2", hf_id="org/toy")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __len__(self) -> int:
            return 7

    class Store:
        device = "cpu"
        cfg = {
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "hidden_size": 8,
            "vocab_size": 8,
        }
        man = {"arch": "unsupported"}

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    store = Store()
    monkeypatch.setattr(paged_module, "resolve_model", lambda _name: spec)
    monkeypatch.setattr(paged_module, "load_tokenizer", lambda _spec: Tokenizer())
    monkeypatch.setattr(paged_module, "QStore", lambda *_args, **_kwargs: store)

    with pytest.raises(NotImplementedError, match="no forward"):
        paged_module.PagedEngine(
            "toy",
            stores_dir=tmp_path,
            cache_mb=0,
        )

    assert store.close_calls == 1


def test_generation_logits_always_exclude_nonsemantic_padded_rows() -> None:
    engine = object.__new__(paged_module.PagedEngine)
    engine.semantic_token_count = 3
    logits = torch.tensor([0.0, 1.0, 2.0, 999.0, 998.0])

    semantic_logits = engine._generation_logits(logits)

    assert semantic_logits.tolist() == [0.0, 1.0, 2.0]
    assert int(semantic_logits.argmax().item()) == 2


def test_generate_serializes_temporary_cache_budget_mutation() -> None:
    engine = object.__new__(paged_module.PagedEngine)
    engine._execution_lock = threading.RLock()
    engine.arch = "qwen2"
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    start = threading.Barrier(2)

    class Store:
        _cache_budget = 0

        def set_cache_budget(self, cache_mb: float) -> None:
            self._cache_budget = int(cache_mb * 1e6)

    engine.store = Store()

    def fake_generate(
        _prompt: Any,
        *,
        max_new_tokens: int,
        **_kwargs: Any,
    ) -> list[int]:
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            assert engine.store._cache_budget == max_new_tokens * 1_000_000
            time.sleep(0.02)
            return [max_new_tokens]
        finally:
            with active_lock:
                active -= 1

    engine._generate = fake_generate

    def run(marker: int) -> list[int] | str:
        start.wait()
        return engine.generate(
            np.asarray([1], dtype=np.int64),
            max_new_tokens=marker,
            cache_mb=float(marker),
            kv=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, (1, 2)))

    assert sorted(results) == [[1], [2]]
    assert max_active == 1
    assert engine.store._cache_budget == 0
