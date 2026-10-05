from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest

from mrun.runtime import (
    MLX_MAMBA_CHUNKED_PREFILL_NUMERICAL_CONTRACT,
    DecodeWork,
    DeviceDescriptor,
    FallbackPolicy,
    MemoryDomain,
    MlxMambaComponentExecutionBackend,
    MlxMambaRuntime,
    MlxMambaRuntimeError,
    NativeBackendBindingError,
    OutputMode,
    OutputRequest,
    PlacementError,
    PrefillWork,
    PromotionStatus,
    WorkloadSpec,
    build_mlx_mamba_prefill_execution_shape,
    compiled_identity_from_mlx_mamba_engine,
)


def _digest(character: str) -> str:
    return character * 64


class _Graph:
    custody_fingerprint_sha256 = _digest("1")
    tokenizer_semantic_sha256 = _digest("2")
    model_name = "toy-mamba"


class _Artifact:
    artifact_sha256 = _digest("3")
    source_dtype = "F32"
    config = {
        "num_hidden_layers": 1,
        "hidden_size": 4,
        "intermediate_size": 2,
        "state_size": 2,
        "conv_kernel": 3,
        "time_step_rank": 1,
        "vocab_size": 17,
    }
    manifest = {
        "schema": "mrun-mlx-source-component-native-v1",
        "artifact_sha256": artifact_sha256,
        "recipe": {
            "codec": "raw-f32",
            "builder_abi": "mamba-builder-v1",
            "mapping_abi": "mamba-mapping-v1",
        },
        "shards": [
            {
                "role": "body",
                "filename": "body.safetensors",
                "sha256": _digest("4"),
                "bytes": 100,
            },
            {
                "role": "lexical_shared",
                "filename": "lexical.safetensors",
                "sha256": _digest("5"),
                "bytes": 40,
            },
        ],
    }


class _Engine:
    graph = _Graph()
    artifact = _Artifact()
    name = "toy-mamba"
    arch = "mamba"
    backend = "mlx-source-component"
    context_size = 1_000_000
    semantic_token_count = 17
    numerical_contract = "mamba-test-f32-v1"
    mlx_engine_active_memory_bytes_at_load = 0

    def assert_content_identity_unchanged(self) -> None:
        return None


@dataclass
class _ScalarCache:
    value: int = 0
    released: bool = False


class _ParameterLeaf:
    shape = (1,)
    dtype = "test.float32"
    nbytes = 4


class _ParameterModel:
    def __init__(self) -> None:
        self.weight = _ParameterLeaf()

    def parameters(self) -> dict[str, _ParameterLeaf]:
        return {"weight": self.weight}


def _cache_factory() -> tuple[_ScalarCache, ...]:
    return (_ScalarCache(),)


def _cache_clone(caches: Any) -> tuple[_ScalarCache, ...]:
    return tuple(_ScalarCache(cache.value) for cache in caches)


def _cache_install(targets: Any, sources: Any) -> None:
    for target, source in zip(targets, sources, strict=True):
        target.value = source.value


def _cache_bytes(caches: Any) -> int:
    if len(tuple(caches)) != 1 or any(cache.released for cache in caches):
        raise MlxMambaRuntimeError("invalid scalar Mamba state")
    return 32


def _cache_release(caches: Any) -> None:
    for cache in caches:
        cache.released = True


def _advance_value(value: int, ids: tuple[int, ...]) -> int:
    for token in ids:
        value = (value * 31 + token + 1) % 1_000_003
    return value


def _executor(ids: tuple[int, ...], caches: Any, _output: OutputRequest) -> int:
    caches[0].value = _advance_value(caches[0].value, ids)
    return caches[0].value % _Engine.semantic_token_count


def _advance(ids: tuple[int, ...], caches: Any) -> None:
    caches[0].value = _advance_value(caches[0].value, ids)


def _chunk_executor(
    ids: tuple[int, ...],
    caches: Any,
    output: OutputRequest | None,
) -> int | None:
    caches[0].value = _advance_value(caches[0].value, ids)
    return None if output is None else caches[0].value % _Engine.semantic_token_count


def _device() -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id="metal:0",
        fabric="apple-gpu",
        memory_domain=MemoryDomain.UNIFIED,
        total_bytes=10_000_000,
        available_bytes=10_000_000,
        machine_fingerprint=_digest("9"),
    )


def _workload(model: Any, *, context: int = 100, verify_tokens: int = 4) -> WorkloadSpec:
    return WorkloadSpec(
        max_batch_size=1,
        max_context_tokens=context,
        verify_tokens=verify_tokens,
        output_mode=OutputMode.NEXT_TOKEN_ARGMAX,
        numerical_contract=_Engine.numerical_contract,
        state_abi=model.state_abi,
        required_component_roles=tuple(component.role for component in model.components),
        workspace_bytes=7,
        headroom_bytes=11,
        fallback_policy=FallbackPolicy.DENY,
    )


def _runtime_factory(engine: Any, **kwargs: Any) -> MlxMambaRuntime:
    shape = kwargs.get("prefill_execution_shape")
    chunk_options = (
        {"prefill_chunk_executor": _chunk_executor}
        if getattr(shape, "chunk_size", None) is not None
        else {}
    )
    return MlxMambaRuntime.bind(
        engine,
        **kwargs,
        cache_factory=_cache_factory,
        cache_clone=_cache_clone,
        cache_install=_cache_install,
        cache_bytes=_cache_bytes,
        cache_release=_cache_release,
        executor=_executor,
        advance_executor=_advance,
        **chunk_options,
    )


def _runtime(
    *,
    context: int = 100,
    chunk_size: int | None = None,
    verify_tokens: int = 4,
) -> tuple[MlxMambaRuntime, Any]:
    engine = _Engine()
    backend = MlxMambaComponentExecutionBackend(
        engine,
        _device(),
        prefill_chunk_size=chunk_size,
        runtime_factory=_runtime_factory,
    )
    placement = backend.plan(
        backend.model,
        _workload(backend.model, context=context, verify_tokens=verify_tokens),
        _device(),
    )
    return backend.open(backend.model, placement), backend


def _prefill(runtime: MlxMambaRuntime, state: Any, ids: tuple[int, ...]) -> Any:
    return runtime.prefill(
        PrefillWork(
            request_ids=("request",),
            token_rows=(ids,),
            state=state,
            parent=state.observe(),
            output=OutputRequest(mode=OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )


def _decode(runtime: MlxMambaRuntime, state: Any, ids: tuple[int, ...]) -> Any:
    return runtime.decode(
        DecodeWork(
            request_ids=("request",),
            token_rows=(ids,),
            state=state,
            parent=state.observe(),
            output=OutputRequest(mode=OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )


def test_mamba_identity_and_placement_charge_fixed_state_not_context() -> None:
    engine = _Engine()
    model = compiled_identity_from_mlx_mamba_engine(engine)

    assert model.state_bytes_per_token == 0
    assert model.state_fixed_bytes_per_row == 32
    assert model.max_context_tokens == 1_000_000
    assert "causal-grouped-query-attention" not in model.operator_ids
    assert "mamba-selective-scan" in model.operator_ids

    backend = MlxMambaComponentExecutionBackend(
        engine, _device(), runtime_factory=lambda *_a, **_k: None
    )
    short = backend.plan(backend.model, _workload(backend.model, context=10), _device())
    assert short.state.reserved_bytes == 32
    assert short.workspace_bytes == 71

    second = MlxMambaComponentExecutionBackend(
        _Engine(), _device(), runtime_factory=lambda *_a, **_k: None
    )
    long = second.plan(second.model, _workload(second.model, context=100_000), _device())
    assert long.state.reserved_bytes == short.state.reserved_bytes
    assert long.workspace_bytes == short.workspace_bytes


def test_chunk_shape_is_fingerprinted_and_charges_bounded_tensor_workspace() -> None:
    shape = build_mlx_mamba_prefill_execution_shape(
        _Artifact.config,
        chunk_size=2,
        base_numerical_contract=_Engine.numerical_contract,
        unchunked_promotion_status=PromotionStatus.EXPERIMENTAL,
    )
    assert shape.activation_workspace_bytes == 720
    assert shape.logits_workspace_bytes == 2_312
    assert shape.workspace_bytes == 3_032
    assert shape.numerical_contract == MLX_MAMBA_CHUNKED_PREFILL_NUMERICAL_CONTRACT
    assert shape.promotion_status is PromotionStatus.EXPERIMENTAL
    assert replace(shape, chunk_size=3).fingerprint != shape.fingerprint

    engine = _Engine()
    backend = MlxMambaComponentExecutionBackend(
        engine,
        _device(),
        prefill_chunk_size=2,
        runtime_factory=_runtime_factory,
    )
    with pytest.raises(AttributeError):
        backend.prefill_execution_shape = replace(shape, chunk_size=1)  # type: ignore[misc]
    workload = _workload(backend.model, verify_tokens=2)
    placement = backend.plan(backend.model, workload, _device())
    assert placement.state.reserved_bytes == 32
    assert placement.workspace_bytes == 7 + 2 * 32 + shape.workspace_bytes

    runtime = backend.open(backend.model, placement)
    assert runtime.prefill_execution_shape == shape
    assert runtime.route.promotion_status is PromotionStatus.EXPERIMENTAL
    assert runtime.route.effective_numerical_contract == (
        MLX_MAMBA_CHUNKED_PREFILL_NUMERICAL_CONTRACT
    )
    assert runtime.route.execution_shape_fingerprint == shape.fingerprint
    runtime.close()

    rejected = MlxMambaComponentExecutionBackend(
        _Engine(),
        _device(),
        prefill_chunk_size=2,
        runtime_factory=_runtime_factory,
    )
    admitted = rejected.plan(rejected.model, _workload(rejected.model, verify_tokens=2), _device())
    with pytest.raises(PlacementError, match="exceeds budget"):
        rejected.plan(
            rejected.model,
            _workload(rejected.model, verify_tokens=2),
            _device(),
            memory_budget_bytes=admitted.total_reserved_bytes - 1,
        )


def test_chunked_prefill_and_partial_replay_are_transactional_and_bounded() -> None:
    runtime, backend = _runtime(chunk_size=2, verify_tokens=2)
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)

    step = _prefill(runtime, state, (1, 2, 3, 4, 5))
    expected_full = _advance_value(0, (1, 2, 3, 4, 5))
    assert state._caches[0].value == 0
    assert step.authority._scratch[0].value == expected_full
    assert step.output.token_ids == (expected_full % _Engine.semantic_token_count,)

    receipt = runtime.commit(step, (3,))
    assert receipt.after.lengths == (3,)
    assert state._caches[0].value == _advance_value(0, (1, 2, 3))
    telemetry = runtime.telemetry()
    counters = dict(telemetry.extra_counters)
    assert counters["mamba_chunk_size"] == 2
    assert counters["mamba_chunk_tensor_workspace_bytes"] == (
        backend.prefill_execution_shape.workspace_bytes
    )
    assert counters["mamba_chunked_prefill_calls"] == 1
    assert counters["mamba_prefill_chunks"] == 3
    assert counters["mamba_prefix_replay_chunks"] == 2
    assert telemetry.workspace_peak_bytes == (
        2 * backend.model.state_fixed_bytes_per_row
        + backend.prefill_execution_shape.workspace_bytes
    )

    with pytest.raises(MlxMambaRuntimeError, match="verification width"):
        _decode(runtime, state, (6, 7, 8))
    runtime.abandon(_decode(runtime, state, (6, 7)))


def test_chunk_failure_discards_only_scratch_and_releases_workspace_authority() -> None:
    runtime, _backend = _runtime(chunk_size=2, verify_tokens=2)
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)
    calls = 0

    def fail_second_chunk(
        ids: tuple[int, ...],
        caches: Any,
        output: OutputRequest | None,
    ) -> int | None:
        nonlocal calls
        calls += 1
        value = _chunk_executor(ids, caches, output)
        if calls == 2:
            raise RuntimeError("injected segment failure")
        return value

    runtime._prefill_chunk_executor = fail_second_chunk
    with pytest.raises(RuntimeError, match="injected segment failure"):
        _prefill(runtime, state, (1, 2, 3, 4, 5))
    assert state.observe().lengths == (0,)
    assert state._caches[0].value == 0
    assert state._pending_step_id is None
    assert dict(runtime.telemetry().extra_counters)["mamba_prefill_chunk_failures"] == 1

    runtime._prefill_chunk_executor = _chunk_executor
    runtime.abandon(_prefill(runtime, state, (1, 2, 3)))


def test_chunk_shape_tamper_between_plan_and_open_is_rejected() -> None:
    engine = _Engine()
    engine.artifact = SimpleNamespace(
        artifact_sha256=_Artifact.artifact_sha256,
        source_dtype=_Artifact.source_dtype,
        config=dict(_Artifact.config),
        manifest={
            **_Artifact.manifest,
            "recipe": dict(_Artifact.manifest["recipe"]),
            "shards": [dict(value) for value in _Artifact.manifest["shards"]],
        },
    )
    backend = MlxMambaComponentExecutionBackend(
        engine,
        _device(),
        prefill_chunk_size=2,
        runtime_factory=_runtime_factory,
    )
    placement = backend.plan(
        backend.model,
        _workload(backend.model, verify_tokens=2),
        _device(),
    )
    engine.artifact.config["hidden_size"] = 5
    with pytest.raises(NativeBackendBindingError, match="prefill execution shape changed"):
        backend.open(backend.model, placement)

    engine.artifact.config["hidden_size"] = 4
    runtime = backend.open(backend.model, placement)
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)
    engine.artifact.config["vocab_size"] = 19
    with pytest.raises(MlxMambaRuntimeError, match="execution identity changed"):
        _prefill(runtime, state, (1, 2))
    engine.artifact.config["vocab_size"] = 17
    runtime.release_state(state)
    runtime.close()


def test_full_commit_and_abandon_never_mutate_committed_state_early() -> None:
    runtime, _backend = _runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)

    step = _prefill(runtime, state, (1, 2, 3))
    assert state._caches[0].value == 0
    assert step.output.token_ids == (_advance_value(0, (1, 2, 3)) % 17,)

    receipt = runtime.commit(step, (3,))
    committed = _advance_value(0, (1, 2, 3))
    assert state._caches[0].value == committed
    assert receipt.after.lengths == (3,)
    assert receipt.state_bytes_written == 32

    decode = _decode(runtime, state, (4,))
    assert state._caches[0].value == committed
    runtime.abandon(decode)
    assert state._caches[0].value == committed
    assert state.observe().lengths == (3,)


def test_partial_commit_replays_only_accepted_prefix_and_consumes_authority() -> None:
    runtime, _backend = _runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)
    step = _prefill(runtime, state, (2, 4, 6, 8))

    receipt = runtime.commit(step, (2,))

    assert state._caches[0].value == _advance_value(0, (2, 4))
    assert receipt.after.lengths == (2,)
    telemetry = runtime.telemetry()
    assert dict(telemetry.extra_counters)["mamba_prefix_replay_forwards"] == 1
    assert dict(telemetry.extra_counters)["mamba_prefix_replay_tokens"] == 2
    assert telemetry.workspace_peak_bytes == 64
    with pytest.raises(MlxMambaRuntimeError, match="foreign or consumed"):
        runtime.commit(step, (2,))


def test_zero_commit_advances_epoch_without_writing_recurrence() -> None:
    runtime, _backend = _runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)
    step = _prefill(runtime, state, (1, 2))

    receipt = runtime.commit(step, (0,))

    assert receipt.before.epoch == 0
    assert receipt.after.epoch == 1
    assert receipt.after.lengths == (0,)
    assert receipt.state_bytes_written == 0
    assert state._caches[0].value == 0


def test_recurrent_fork_is_independent_and_charges_opaque_clone_conservatively() -> None:
    runtime, _backend = _runtime()
    source = runtime.allocate_state(owner_id="source", batch_size=1, capacity=100)
    runtime.commit(_prefill(runtime, source, (1, 3, 5)), (3,))

    result = runtime.fork_state(
        source,
        parent=source.observe(),
        owner_id="fork",
        capacity=80,
    )
    forked = result.state
    assert result.state_bytes_copied == 32
    assert forked.observe().lengths == (3,)
    assert forked._caches[0].value == source._caches[0].value

    runtime.commit(_decode(runtime, forked, (7,)), (1,))
    assert forked._caches[0].value != source._caches[0].value
    assert source.observe().lengths == (3,)


def test_failed_forward_and_pending_close_leave_committed_state_safe() -> None:
    runtime, _backend = _runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)

    def failing(ids: tuple[int, ...], caches: Any, _output: OutputRequest) -> int:
        caches[0].value = _advance_value(caches[0].value, ids)
        raise RuntimeError("injected failure")

    runtime._executor = failing
    with pytest.raises(RuntimeError, match="injected failure"):
        _prefill(runtime, state, (1, 2, 3))
    assert state.observe().lengths == (0,)
    assert state._caches[0].value == 0

    runtime._executor = _executor
    step = _prefill(runtime, state, (1,))
    with pytest.raises(MlxMambaRuntimeError, match="pending states"):
        runtime.close()
    runtime.abandon(step)
    runtime.close()


def test_reconstructed_step_and_aliased_clone_cannot_gain_commit_authority() -> None:
    runtime, _backend = _runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)
    step = _prefill(runtime, state, (1, 2))

    reconstructed = replace(step, token_counts=(3,))
    with pytest.raises(MlxMambaRuntimeError, match="foreign or consumed"):
        runtime.commit(reconstructed, (3,))
    runtime.abandon(step)

    runtime._cache_clone = lambda caches: caches
    with pytest.raises(MlxMambaRuntimeError, match="aliases a committed cache container"):
        _prefill(runtime, state, (3,))
    assert state.observe().lengths == (0,)
    runtime._cache_clone = _cache_clone
    runtime.abandon(_prefill(runtime, state, (3,)))

    duplicate = _ScalarCache()
    runtime._cache_clone = lambda _caches: (duplicate, duplicate)
    with pytest.raises(MlxMambaRuntimeError, match="repeats a cache container"):
        _prefill(runtime, state, (4,))
    assert duplicate.released
    runtime._cache_clone = _cache_clone

    created: list[_ScalarCache] = []

    def malformed_clone(_caches: Any) -> tuple[_ScalarCache, ...]:
        created.extend((_ScalarCache(), _ScalarCache()))
        return tuple(created)

    runtime._cache_clone = malformed_clone
    with pytest.raises(MlxMambaRuntimeError, match="invalid scalar Mamba state"):
        _prefill(runtime, state, (4,))
    assert created and all(cache.released for cache in created)
    runtime._cache_clone = _cache_clone
    runtime.abandon(_prefill(runtime, state, (4,)))


def test_partial_replay_rechecks_the_exact_parameter_tree() -> None:
    engine = _Engine()
    engine.model = _ParameterModel()
    backend = MlxMambaComponentExecutionBackend(
        engine,
        _device(),
        runtime_factory=_runtime_factory,
    )
    placement = backend.plan(backend.model, _workload(backend.model), _device())
    runtime = backend.open(backend.model, placement)
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)
    original = engine.model.weight

    def swap_parameter_during_forward(
        ids: tuple[int, ...], caches: Any, output: OutputRequest
    ) -> int:
        value = _executor(ids, caches, output)
        engine.model.weight = _ParameterLeaf()
        return value

    runtime._executor = swap_parameter_during_forward
    with pytest.raises(MlxMambaRuntimeError, match="parameter identity changed"):
        _prefill(runtime, state, (1, 2, 3, 4))
    assert state.observe().lengths == (0,)
    assert state._pending_step_id is None

    engine.model.weight = original
    runtime._executor = _executor
    step = _prefill(runtime, state, (1, 2, 3, 4))

    engine.model.weight = _ParameterLeaf()
    with pytest.raises(MlxMambaRuntimeError, match="parameter identity changed"):
        runtime.commit(step, (2,))
    assert state.observe().lengths == (0,)
    assert state._pending_step_id == step.step_id

    engine.model.weight = original
    receipt = runtime.commit(step, (2,))
    assert receipt.after.lengths == (2,)


def test_workspace_reservation_allows_only_one_outstanding_transaction() -> None:
    runtime, _backend = _runtime()
    first = runtime.allocate_state(owner_id="first", batch_size=1, capacity=100)
    second = runtime.allocate_state(owner_id="second", batch_size=1, capacity=100)

    step = _prefill(runtime, first, (1,))
    with pytest.raises(MlxMambaRuntimeError, match="scratch reservation"):
        _prefill(runtime, second, (2,))
    runtime.abandon(step)

    second_step = _prefill(runtime, second, (2,))
    runtime.abandon(second_step)


def test_cleanup_failure_never_makes_commit_or_abandon_outcome_ambiguous() -> None:
    runtime, _backend = _runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)

    def fail_release(_caches: Any) -> None:
        raise RuntimeError("injected cleanup failure")

    runtime._cache_release = fail_release
    receipt = runtime.commit(_prefill(runtime, state, (1, 2)), (2,))
    assert receipt.after.lengths == (2,)
    runtime.abandon(_decode(runtime, state, (3,)))
    assert state.observe().lengths == (2,)
    assert dict(runtime.telemetry().extra_counters)["mamba_cleanup_failures"] == 2

    runtime.release_state(state)
    with pytest.raises(MlxMambaRuntimeError, match="stale or released"):
        runtime._state(state)
    assert dict(runtime.telemetry().extra_counters)["mamba_cleanup_failures"] == 3


def test_forward_and_cleanup_failure_clear_pending_authority_and_workspace() -> None:
    runtime, _backend = _runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)

    def fail_forward(ids: tuple[int, ...], caches: Any, _output: OutputRequest) -> int:
        caches[0].value = _advance_value(caches[0].value, ids)
        raise RuntimeError("injected forward failure")

    def fail_release(_caches: Any) -> None:
        raise RuntimeError("injected cleanup failure")

    runtime._executor = fail_forward
    runtime._cache_release = fail_release
    with pytest.raises(RuntimeError, match="injected forward failure"):
        _prefill(runtime, state, (1,))
    assert state._pending_step_id is None
    assert state._pending_authority is None

    runtime._executor = _executor
    runtime._cache_release = _cache_release
    runtime.abandon(_prefill(runtime, state, (2,)))
    assert dict(runtime.telemetry().extra_counters)["mamba_cleanup_failures"] == 1


def test_backend_rechecks_engine_identity_between_plan_and_open() -> None:
    engine = _Engine()
    engine.artifact = SimpleNamespace(
        artifact_sha256=_Artifact.artifact_sha256,
        source_dtype=_Artifact.source_dtype,
        config=dict(_Artifact.config),
        manifest={
            **_Artifact.manifest,
            "recipe": dict(_Artifact.manifest["recipe"]),
            "shards": [dict(value) for value in _Artifact.manifest["shards"]],
        },
    )
    backend = MlxMambaComponentExecutionBackend(
        engine,
        _device(),
        runtime_factory=_runtime_factory,
    )
    placement = backend.plan(backend.model, _workload(backend.model), _device())

    engine.artifact.config["state_size"] = 3
    with pytest.raises(NativeBackendBindingError, match="identity changed"):
        backend.open(backend.model, placement)

    engine.artifact.config["state_size"] = 2
    engine.model = object()
    with pytest.raises(NativeBackendBindingError, match="loaded model changed"):
        backend.open(backend.model, placement)
    del engine.model
    runtime = backend.open(backend.model, placement)
    runtime.close()


def test_default_mlx_cache_and_executor_match_direct_mamba_stateful_decode() -> None:
    mx = pytest.importorskip("mlx.core")
    mamba = pytest.importorskip("mlx_lm.models.mamba")

    args = mamba.ModelArgs(
        model_type="mamba",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        state_size=4,
        num_hidden_layers=2,
        conv_kernel=3,
        use_bias=False,
        use_conv_bias=True,
        time_step_rank=1,
        tie_word_embeddings=True,
    )
    model = mamba.Model(args)
    model.eval()
    config = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "state_size": 4,
        "conv_kernel": 3,
        "time_step_rank": 1,
        "vocab_size": 16,
    }
    artifact = SimpleNamespace(
        artifact_sha256=_digest("3"),
        source_dtype="F32",
        config=config,
        manifest={
            "schema": "mrun-mlx-source-component-native-v1",
            "artifact_sha256": _digest("3"),
            "recipe": {
                "codec": "raw-f32",
                "builder_abi": "mamba-builder-v1",
                "mapping_abi": "mamba-mapping-v1",
            },
            "shards": [
                {
                    "role": "body",
                    "filename": "body.safetensors",
                    "sha256": _digest("4"),
                    "bytes": 100,
                }
            ],
        },
    )
    engine = SimpleNamespace(
        graph=_Graph(),
        artifact=artifact,
        name="toy-mamba",
        arch="mamba",
        backend="mlx-source-component",
        context_size=1_000,
        semantic_token_count=16,
        numerical_contract=_Engine.numerical_contract,
        mlx_engine_active_memory_bytes_at_load=0,
        model=model,
        _mx=mx,
        assert_content_identity_unchanged=lambda: None,
    )
    backend = MlxMambaComponentExecutionBackend(engine, _device())
    placement = backend.plan(backend.model, _workload(backend.model, context=100), _device())
    runtime = backend.open(backend.model, placement)
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=100)
    container_ids = tuple(id(cache) for cache in state._caches)
    committed_arrays = tuple(tuple(cache.state) for cache in state._caches)

    engine.model = object()
    with pytest.raises(MlxMambaRuntimeError, match="execution identity changed"):
        _prefill(runtime, state, (1,))
    engine.model = model

    direct_cache = model.make_cache()
    prompt = mx.array([[1, 2, 3]])
    direct_logits = model(prompt, cache=direct_cache)
    expected_prefill = mx.argmax(direct_logits[0, -1, :16]).item()
    step = _prefill(runtime, state, (1, 2, 3))
    assert step.output.token_ids == (expected_prefill,)
    # The production ArraysCache clone initially shares immutable input arrays, but the model
    # replaces only the scratch-container entries.  Committed containers and arrays remain exact.
    assert tuple(id(cache) for cache in state._caches) == container_ids
    assert all(
        actual is expected
        for cache, expected_pair in zip(state._caches, committed_arrays, strict=True)
        for actual, expected in zip(cache.state, expected_pair, strict=True)
    )
    scratch_arrays = tuple(tuple(cache.state) for cache in step.authority._scratch)
    assert all(
        scratch is not committed
        for scratch_pair, committed_pair in zip(scratch_arrays, committed_arrays, strict=True)
        for scratch, committed in zip(scratch_pair, committed_pair, strict=True)
    )
    runtime.commit(step, (3,))
    assert tuple(id(cache) for cache in state._caches) == container_ids
    assert all(
        committed is scratch
        for cache, scratch_pair in zip(state._caches, scratch_arrays, strict=True)
        for committed, scratch in zip(cache.state, scratch_pair, strict=True)
    )

    direct_logits = model(mx.array([[4]]), cache=direct_cache)
    expected_decode = mx.argmax(direct_logits[0, -1, :16]).item()
    before_decode_arrays = tuple(tuple(cache.state) for cache in state._caches)
    decode = _decode(runtime, state, (4,))
    assert decode.output.token_ids == (expected_decode,)
    assert all(
        actual is expected
        for cache, expected_pair in zip(state._caches, before_decode_arrays, strict=True)
        for actual, expected in zip(cache.state, expected_pair, strict=True)
    )
    runtime.commit(decode, (1,))
    assert state.observe().lengths == (4,)

    original = state._caches[0][0]
    state._caches[0][0] = mx.zeros(original.shape, dtype=original.dtype)
    with pytest.raises(MlxMambaRuntimeError, match="array identity changed"):
        state.observe()
    state._caches[0][0] = original
    assert state.observe().lengths == (4,)

    tampered = _decode(runtime, state, (5,))
    scratch_cache = tampered.authority._scratch[0]
    scratch_original = scratch_cache[0]
    scratch_cache[0] = mx.zeros(scratch_original.shape, dtype=scratch_original.dtype)
    with pytest.raises(MlxMambaRuntimeError, match="no longer owns"):
        runtime.commit(tampered, (1,))
    scratch_cache[0] = scratch_original
    runtime.abandon(tampered)

    source_arrays = tuple(tuple(cache.state) for cache in state._caches)
    fork_result = runtime.fork_state(
        state,
        parent=state.observe(),
        owner_id="fork",
        capacity=100,
    )
    assert fork_result.state_bytes_copied == 0
    shared_telemetry = runtime.telemetry()
    shared_counters = dict(shared_telemetry.extra_counters)
    assert shared_counters["mamba_recurrent_state_resident_bytes"] == (
        backend.model.state_fixed_bytes_per_row
    )
    assert shared_counters["mamba_recurrent_state_logical_bytes"] == (
        2 * backend.model.state_fixed_bytes_per_row
    )
    assert all(
        forked is source
        for fork_cache, source_pair in zip(
            fork_result.state._caches,
            source_arrays,
            strict=True,
        )
        for forked, source in zip(fork_cache.state, source_pair, strict=True)
    )
    fork_step = _decode(runtime, fork_result.state, (6,))
    assert all(
        actual is expected
        for cache, expected_pair in zip(state._caches, source_arrays, strict=True)
        for actual, expected in zip(cache.state, expected_pair, strict=True)
    )
    runtime.commit(fork_step, (1,))
    assert (
        dict(runtime.telemetry().extra_counters)["mamba_recurrent_state_resident_bytes"]
        == 2 * backend.model.state_fixed_bytes_per_row
    )
    assert all(
        actual is expected
        for cache, expected_pair in zip(state._caches, source_arrays, strict=True)
        for actual, expected in zip(cache.state, expected_pair, strict=True)
    )

    partial_state = runtime.allocate_state(owner_id="partial", batch_size=1, capacity=100)
    partial_step = _prefill(runtime, partial_state, (7, 8, 9))
    runtime.commit(partial_step, (2,))
    direct_partial_cache = model.make_cache()
    model(mx.array([[7, 8]]), cache=direct_partial_cache)
    mx.eval(*(value for cache in direct_partial_cache for value in cache.state))
    assert all(
        bool(mx.array_equal(actual, expected).item())
        for runtime_cache, direct_cache_layer in zip(
            partial_state._caches,
            direct_partial_cache,
            strict=True,
        )
        for actual, expected in zip(runtime_cache.state, direct_cache_layer.state, strict=True)
    )
    direct_partial_logits = model(mx.array([[10]]), cache=direct_partial_cache)
    expected_partial_decode = mx.argmax(direct_partial_logits[0, -1, :16]).item()
    partial_decode = _decode(runtime, partial_state, (10,))
    assert partial_decode.output.token_ids == (expected_partial_decode,)
    runtime.abandon(partial_decode)

    partial_original = partial_state._caches[0][1]
    partial_state._caches[0][1] = mx.zeros(
        partial_original.shape,
        dtype=partial_original.dtype,
    )
    with pytest.raises(MlxMambaRuntimeError, match="array identity changed"):
        runtime.release_state(partial_state)
    partial_state._caches[0][1] = partial_original
    runtime.release_state(partial_state)

    # Measure the actual mlx-lm segmented trajectory against one full-sequence call.  The current
    # implementation has a small nonzero F32 delta; the route is therefore separately named and
    # remains experimental rather than masquerading as the unchunked numerical contract.
    segment_ids = (1, 2, 3, 4, 5, 6, 7)
    full_cache = model.make_cache()
    full_segment_logits = model(mx.array([segment_ids]), cache=full_cache)
    segmented_cache = model.make_cache()
    segmented_logits = None
    for start in range(0, len(segment_ids), 2):
        segmented_logits = model(
            mx.array([segment_ids[start : start + 2]]),
            cache=segmented_cache,
        )
    assert segmented_logits is not None
    full_last = full_segment_logits[0, -1, :16]
    segmented_last = segmented_logits[0, -1, :16]
    mx.eval(
        full_last,
        segmented_last,
        *(value for cache in full_cache for value in cache.state),
        *(value for cache in segmented_cache for value in cache.state),
    )
    max_segment_abs = float(mx.max(mx.abs(full_last - segmented_last)).item())
    assert 0.0 < max_segment_abs <= 1e-6

    chunk_backend = MlxMambaComponentExecutionBackend(
        engine,
        _device(),
        prefill_chunk_size=2,
    )
    chunk_placement = chunk_backend.plan(
        chunk_backend.model,
        _workload(chunk_backend.model, context=100, verify_tokens=2),
        _device(),
    )
    chunk_runtime = chunk_backend.open(chunk_backend.model, chunk_placement)
    assert chunk_runtime.route.promotion_status is PromotionStatus.EXPERIMENTAL
    assert chunk_runtime.route.effective_numerical_contract == (
        MLX_MAMBA_CHUNKED_PREFILL_NUMERICAL_CONTRACT
    )
    assert chunk_runtime.route.effective_numerical_contract != engine.numerical_contract
    chunk_state = chunk_runtime.allocate_state(owner_id="chunk", batch_size=1, capacity=100)
    chunk_step = _prefill(chunk_runtime, chunk_state, segment_ids)
    assert chunk_step.output.token_ids == (int(mx.argmax(segmented_last).item()),)
    assert chunk_state.observe().lengths == (0,)
    chunk_runtime.commit(chunk_step, (len(segment_ids),))
    assert all(
        bool(mx.array_equal(actual, expected).item())
        for runtime_cache, expected_cache in zip(
            chunk_state._caches,
            segmented_cache,
            strict=True,
        )
        for actual, expected in zip(runtime_cache.state, expected_cache.state, strict=True)
    )
    chunk_counters = dict(chunk_runtime.telemetry().extra_counters)
    assert chunk_counters["mamba_chunked_prefill_calls"] == 1
    assert chunk_counters["mamba_prefill_chunks"] == 4

    chunk_partial = chunk_runtime.allocate_state(
        owner_id="chunk-partial",
        batch_size=1,
        capacity=100,
    )
    chunk_runtime.commit(_prefill(chunk_runtime, chunk_partial, (8, 9, 10, 11, 12)), (3,))
    direct_chunk_partial = model.make_cache()
    model(mx.array([[8, 9]]), cache=direct_chunk_partial)
    model(mx.array([[10]]), cache=direct_chunk_partial)
    mx.eval(*(value for cache in direct_chunk_partial for value in cache.state))
    assert all(
        bool(mx.array_equal(actual, expected).item())
        for runtime_cache, expected_cache in zip(
            chunk_partial._caches,
            direct_chunk_partial,
            strict=True,
        )
        for actual, expected in zip(runtime_cache.state, expected_cache.state, strict=True)
    )
    chunk_runtime.close()

    fork_original = fork_result.state._caches[0][0]
    fork_result.state._caches[0][0] = mx.zeros(
        fork_original.shape,
        dtype=fork_original.dtype,
    )
    with pytest.raises(MlxMambaRuntimeError, match="array identity changed"):
        runtime.close()
    assert not state._released
    assert not fork_result.state._released
    fork_result.state._caches[0][0] = fork_original
    runtime.close()
