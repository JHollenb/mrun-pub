from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from mrun.runtime import (
    MLX_PAGED_KV_CACHE_ABI,
    DenseCudaComponentExecutionBackend,
    DeviceDescriptor,
    FallbackPolicy,
    MemoryDomain,
    MlxComponentExecutionBackend,
    MlxLayerCacheSpec,
    MlxStateLayout,
    NativeBackendBindingError,
    OutputMode,
    PromotionStatus,
    WorkloadSpec,
    compiled_identity_from_dense_cuda_engine,
    compiled_identity_from_mlx_engine,
    dense_cuda_body_workspace_bytes,
)
from mrun.runtime.mlx_quantized_kv import MlxAffineKVCodec


def _digest(character: str) -> str:
    return character * 64


def _device(*, cuda: bool, available: int = 10_000_000) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id="cuda:0" if cuda else "metal:0",
        fabric="nvidia-cuda" if cuda else "apple-gpu",
        memory_domain=MemoryDomain.CUDA if cuda else MemoryDomain.UNIFIED,
        total_bytes=available,
        available_bytes=available,
        machine_fingerprint=_digest("9"),
    )


def _workload(model: Any, *, batch: int = 1) -> WorkloadSpec:
    return WorkloadSpec(
        max_batch_size=batch,
        max_context_tokens=8,
        verify_tokens=1,
        output_mode=OutputMode.NEXT_TOKEN_ARGMAX,
        numerical_contract="native-test-v1",
        state_abi=model.state_abi,
        required_component_roles=tuple(component.role for component in model.components),
        workspace_bytes=7,
        headroom_bytes=11,
        fallback_policy=FallbackPolicy.DENY,
    )


class _FakeMlxGraph:
    custody_fingerprint_sha256 = _digest("1")
    tokenizer_semantic_sha256 = _digest("2")
    model_name = "toy"
    architecture = "qwen2"


class _FakeMlxArtifact:
    artifact_sha256 = _digest("3")
    manifest = {
        "schema": "mrun-mlx-component-native-v1",
        "artifact_sha256": artifact_sha256,
        "recipe": {
            "codec": "mlx-affine-int8-g64-exact-qstore-v1",
            "builder_abi": "builder-v1",
            "mapping_abi": "mapping-v1",
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
            {
                "role": "norm",
                "filename": "norm.safetensors",
                "sha256": _digest("6"),
                "bytes": 8,
            },
        ],
    }


class _FakeMlxEngine:
    graph = _FakeMlxGraph()
    artifact = _FakeMlxArtifact()
    name = "toy"
    arch = "qwen2"
    backend = "mlx-component"
    context_size = 32
    semantic_token_count = 10
    numerical_contract = "native-test-v1"
    mlx_active_memory_mb_at_load = 0.0

    def assert_content_identity_unchanged(self) -> None:
        return None


def _mlx_layout() -> MlxStateLayout:
    layer = MlxLayerCacheSpec(
        kv_heads=2,
        key_head_dim=3,
        value_head_dim=3,
        key_dtype="float32",
        value_dtype="float32",
        key_element_bytes=4,
        value_element_bytes=4,
    )
    return MlxStateLayout(layers=(layer,), dtype_name="float32", bytes_per_token=48)


def _bf16_mlx_layout(*, head_dim: int = 32) -> MlxStateLayout:
    layer = MlxLayerCacheSpec(
        kv_heads=1,
        key_head_dim=head_dim,
        value_head_dim=head_dim,
        key_dtype="bfloat16",
        value_dtype="bfloat16",
        key_element_bytes=2,
        value_element_bytes=2,
    )
    return MlxStateLayout(
        layers=(layer,),
        dtype_name="bfloat16",
        bytes_per_token=head_dim * 4,
    )


def test_mlx_binding_uses_real_shards_and_measured_state_and_opens_only_issued_plan() -> None:
    engine = _FakeMlxEngine()
    model, layout = compiled_identity_from_mlx_engine(engine, state_layout=_mlx_layout())
    assert model.physical_allocation_bytes == 148
    assert model.state_bytes_per_token == 48
    assert model.vocab_manifest_sha256 == _digest("2")
    assert {component.role for component in model.components} == {
        "body",
        "lexical_shared",
        "norm",
    }
    assert layout == _mlx_layout()

    calls: list[dict[str, Any]] = []

    def runtime_factory(_engine: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    device = _device(cuda=False)
    backend = MlxComponentExecutionBackend(
        engine,
        device,
        state_layout=layout,
        runtime_factory=runtime_factory,
    )
    assert set(backend.capabilities(device).output_modes) == {
        OutputMode.NEXT_TOKEN_ARGMAX,
        OutputMode.NEXT_TOKEN_SAMPLE,
    }
    workload = _workload(backend.model)
    placement = backend.plan(backend.model, workload, device)
    assert placement.state.bytes_per_token == 48
    assert placement.workspace_bytes == 7
    assert placement.performance_claim_valid
    runtime = backend.open(backend.model, placement)
    assert runtime is not None
    assert calls[0]["workload"] == backend.admitted_workload(placement)
    assert calls[0]["prefill_chunk_size"] is None
    with pytest.raises(NativeBackendBindingError, match="already has a runtime"):
        backend.open(backend.model, placement)


def test_mlx_binding_threads_explicit_chunked_prefill_shape_to_runtime() -> None:
    captured: list[dict[str, Any]] = []

    def runtime_factory(_engine: Any, **kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    engine = _FakeMlxEngine()
    device = _device(cuda=False)
    backend = MlxComponentExecutionBackend(
        engine,
        device,
        state_layout=_mlx_layout(),
        prefill_chunk_size=8,
        runtime_factory=runtime_factory,
    )
    placement = backend.plan(backend.model, _workload(backend.model), device)
    backend.open(backend.model, placement)

    assert captured[0]["prefill_chunk_size"] == 8
    with pytest.raises(NativeBackendBindingError, match="context limit"):
        MlxComponentExecutionBackend(
            _FakeMlxEngine(),
            device,
            state_layout=_mlx_layout(),
            prefill_chunk_size=33,
        )


def test_mlx_binding_quantized_kv_is_identity_bound_exactly_charged_and_experimental() -> None:
    captured: list[dict[str, Any]] = []

    def runtime_factory(_engine: Any, **kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    layer = MlxLayerCacheSpec(
        kv_heads=2,
        key_head_dim=64,
        value_head_dim=64,
        key_dtype="bfloat16",
        value_dtype="bfloat16",
        key_element_bytes=2,
        value_element_bytes=2,
    )
    source_layout = MlxStateLayout(
        layers=(layer,),
        dtype_name="bfloat16",
        bytes_per_token=512,
    )
    engine = _FakeMlxEngine()
    device = _device(cuda=False)
    backend = MlxComponentExecutionBackend(
        engine,
        device,
        state_layout=source_layout,
        kv_codec=MlxAffineKVCodec(bits=4, group_size=64),
        runtime_factory=runtime_factory,
    )

    assert backend.model.state_bytes_per_token == 144
    assert backend.model.state_dtype == "mlx-affine-kv4-g64-bfloat16"
    assert backend.model.state_abi == (
        "mrun-transactional-gqa-mlx-affine-kv4-g64-bfloat16-global-attention-v1"
    )
    assert backend.capabilities(device).promotion_status.value == "experimental"
    placement = backend.plan(backend.model, _workload(backend.model), device)
    assert placement.state.bytes_per_token == 144
    backend.open(backend.model, placement)
    assert captured[0]["state_layout"].bytes_per_token == 144
    assert callable(captured[0]["cache_factory"])


def test_mlx_binding_paged_kv_is_identity_bound_once_and_runtime_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def runtime_factory(_engine: Any, **kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    class FakePagedFactory:
        cache_abi = MLX_PAGED_KV_CACHE_ABI

        def __call__(self, _capacity: int) -> tuple[()]:
            return ()

        def close(self) -> None:
            return None

    engine = _FakeMlxEngine()
    device = _device(cuda=False)
    backend = MlxComponentExecutionBackend(
        engine,
        device,
        state_layout=_mlx_layout(),
        paged_kv_page_size=4,
        paged_kv_page_count=4,
        runtime_factory=runtime_factory,
    )
    factory = FakePagedFactory()
    monkeypatch.setattr(backend, "_paged_cache_factory", lambda: factory)

    assert backend.model.state_abi == MLX_PAGED_KV_CACHE_ABI
    assert backend.model.state_bytes_per_token == 48
    assert backend.paged_kv_physical_bytes == 4 * 4 * 48
    assert backend.capabilities(device).promotion_status.value == "experimental"
    placement = backend.plan(backend.model, _workload(backend.model), device)
    # The placement's ordinary eight-token B1 arena is replaced by one exact 16-token pool.
    assert placement.state.reserved_bytes == 8 * 48
    assert placement.workspace_bytes == 7 + (16 - 8) * 48
    backend.open(backend.model, placement)
    assert captured[0]["cache_factory"] is factory
    assert captured[0]["owns_cache_factory"] is True


def test_mlx_binding_paged_kv_rejects_ambiguous_or_underprovisioned_contracts() -> None:
    engine = _FakeMlxEngine()
    device = _device(cuda=False)
    with pytest.raises(NativeBackendBindingError, match="both page size and page count"):
        MlxComponentExecutionBackend(
            engine,
            device,
            state_layout=_mlx_layout(),
            paged_kv_page_size=4,
        )
    with pytest.raises(NativeBackendBindingError, match="mutually exclusive"):
        MlxComponentExecutionBackend(
            engine,
            device,
            state_layout=_mlx_layout(),
            kv_codec=MlxAffineKVCodec(bits=4, group_size=64),
            paged_kv_page_size=4,
            paged_kv_page_count=4,
        )
    backend = MlxComponentExecutionBackend(
        engine,
        device,
        state_layout=_mlx_layout(),
        paged_kv_page_size=2,
        paged_kv_page_count=2,
    )
    with pytest.raises(NativeBackendBindingError, match="one maximum-context state"):
        backend.plan(backend.model, _workload(backend.model), device)


def test_mlx_binding_threads_explicit_paged_decode_lane_to_the_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def runtime_factory(_engine: Any, **kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    class FakePagedFactory:
        cache_abi = MLX_PAGED_KV_CACHE_ABI
        paged_decode_attention = True

        def close(self) -> None:
            return None

    lane = SimpleNamespace(close=lambda: None)
    backend = MlxComponentExecutionBackend(
        _FakeMlxEngine(),
        _device(cuda=False),
        state_layout=_bf16_mlx_layout(),
        paged_kv_page_size=4,
        paged_kv_page_count=4,
        paged_decode_attention=True,
        runtime_factory=runtime_factory,
    )
    factory = FakePagedFactory()
    monkeypatch.setattr(backend, "_paged_cache_factory", lambda: factory)
    monkeypatch.setattr(backend, "_paged_decode_attention_lane", lambda: lane)
    placement = backend.plan(backend.model, _workload(backend.model), _device(cuda=False))

    backend.open(backend.model, placement)

    assert captured[0]["cache_factory"] is factory
    assert captured[0]["owns_cache_factory"] is True
    assert captured[0]["paged_decode_attention_lane"] is lane
    assert backend.capabilities(_device(cuda=False)).promotion_status.value == "experimental"


def test_mlx_binding_paged_decode_fails_closed_before_open_on_unsupported_state() -> None:
    common = {
        "device": _device(cuda=False),
        "paged_kv_page_size": 4,
        "paged_kv_page_count": 4,
        "paged_decode_attention": True,
    }
    with pytest.raises(NativeBackendBindingError, match="requires BF16"):
        MlxComponentExecutionBackend(
            _FakeMlxEngine(),
            state_layout=_mlx_layout(),
            **common,
        )
    with pytest.raises(NativeBackendBindingError, match="unsupported Metal"):
        MlxComponentExecutionBackend(
            _FakeMlxEngine(),
            state_layout=_bf16_mlx_layout(head_dim=16),
            **common,
        )
    with pytest.raises(NativeBackendBindingError, match="power-of-two"):
        MlxComponentExecutionBackend(
            _FakeMlxEngine(),
            state_layout=_bf16_mlx_layout(),
            device=_device(cuda=False),
            paged_kv_page_size=3,
            paged_kv_page_count=6,
            paged_decode_attention=True,
        )


def test_mlx_binding_honors_only_a_boolean_engine_experimental_demotion() -> None:
    engine = _FakeMlxEngine()
    engine.experimental_runtime = True
    backend = MlxComponentExecutionBackend(
        engine,
        _device(cuda=False),
        state_layout=_mlx_layout(),
        runtime_factory=lambda *_args, **_kwargs: object(),
    )
    assert backend.capabilities(_device(cuda=False)).promotion_status.value == "experimental"

    malformed = _FakeMlxEngine()
    malformed.experimental_runtime = "experimental"
    with pytest.raises(NativeBackendBindingError, match="must be boolean"):
        MlxComponentExecutionBackend(
            malformed,
            _device(cuda=False),
            state_layout=_mlx_layout(),
        )


def test_mlx_binding_charges_engine_local_delta_not_process_wide_reference_models() -> None:
    engine = _FakeMlxEngine()
    engine.mlx_active_memory_mb_at_load = 4096.0
    engine.mlx_engine_active_memory_bytes_at_load = 200
    device = _device(cuda=False)
    backend = MlxComponentExecutionBackend(
        engine,
        device,
        state_layout=_mlx_layout(),
        runtime_factory=lambda *_args, **_kwargs: object(),
    )

    placement = backend.plan(backend.model, _workload(backend.model), device)

    # The artifact already accounts for 148 bytes of the measured 200-byte engine delta.
    assert placement.workspace_bytes == 7 + (200 - 148)


class _FakeCudaGraph:
    fingerprint = _digest("a")
    schema = "mrun-component-graph-v1"
    model_name = "toy"
    architecture = "qwen2"
    body_abi = {"semantic_sha256": _digest("b")}
    components = {
        "body": {
            "blobs": {
                "weights.i8": {"sha256": _digest("c"), "bytes": 100},
                "scales.f32": {"sha256": _digest("d"), "bytes": 20},
            }
        },
        "lexical_shared": {
            "blobs": {
                "weights.i8": {"sha256": _digest("e"), "bytes": 40},
                "scales.f32": {"sha256": _digest("f"), "bytes": 8},
            }
        },
    }

    def __init__(self) -> None:
        self.verified: list[str] = []

    def verify_component_blobs(self, role: str) -> None:
        self.verified.append(role)


class _FakeComposite:
    vocab_manifest_sha256 = _digest("7")

    def __init__(self, *, fully_resident: bool = True, resident_head: bool = True) -> None:
        self.graph = _FakeCudaGraph()
        self._fully_resident = fully_resident
        self._resident_head = resident_head

    def assert_content_identity_unchanged(self) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "resident_exact_head": self._resident_head,
            "resident_exact_head_bytes": 400,
            "providers": {
                role: {"fully_resident": self._fully_resident} for role in self.graph.components
            },
        }


class _FakeCudaEngine:
    backend = "dense-qstore-cuda"
    name = "toy"
    arch = "qwen2"
    max_seq_len = 32
    semantic_token_count = 10
    numerical_contract = "native-test-v1"
    cfg = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 16,
        "head_dim": 4,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
    }

    def __init__(self, **composite_kwargs: Any) -> None:
        self.composite_store = _FakeComposite(**composite_kwargs)
        self.store = SimpleNamespace(compute_dtype="bfloat16")


def test_cuda_binding_verifies_all_blobs_charges_exact_head_and_rejects_foreign_model() -> None:
    engine = _FakeCudaEngine()
    model = compiled_identity_from_dense_cuda_engine(engine)
    assert engine.composite_store.graph.verified == ["body", "lexical_shared"]
    assert model.physical_allocation_bytes == 168
    assert model.state_dtype == "bfloat16"
    assert model.state_bytes_per_token == 64
    assert model.max_context_tokens == 32

    captured: list[dict[str, Any]] = []

    def runtime_factory(_engine: Any, **kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    device = _device(cuda=True)
    backend = DenseCudaComponentExecutionBackend(
        engine,
        device,
        max_batch_size=8,
        runtime_factory=runtime_factory,
    )
    assert set(backend.capabilities(device).output_modes) == {
        OutputMode.NEXT_TOKEN_ARGMAX,
        OutputMode.NEXT_TOKEN_SAMPLE,
    }
    workload = _workload(backend.model, batch=8)
    placement = backend.plan(backend.model, workload, device)
    assert placement.workspace_bytes == 407 + backend.body_workspace_bytes(workload)
    assert placement.state.reserved_bytes == 64 * 8 * 8
    backend.open(backend.model, placement)
    assert captured[0]["placement"] == placement
    assert captured[0]["admitted_body_workspace_bytes"] == backend.body_workspace_bytes(workload)

    foreign = replace(backend.model, model_name="foreign")
    with pytest.raises(NativeBackendBindingError, match="foreign compiled model"):
        backend.plan(foreign, workload, device)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fully_resident": False}, "nonresident component roles"),
        ({"resident_head": False}, "exact FP32 head"),
    ],
)
def test_cuda_resident_binding_fails_closed(kwargs: dict[str, bool], message: str) -> None:
    with pytest.raises(NativeBackendBindingError, match=message):
        DenseCudaComponentExecutionBackend(_FakeCudaEngine(**kwargs), _device(cuda=True))


def test_compact_cuda_head_is_argmax_only_and_charges_no_expanded_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mrun.runtime.native_backends as native_backends

    model = compiled_identity_from_dense_cuda_engine(_FakeCudaEngine())

    class _CompactStore:
        compute_dtype = "bfloat16"
        require_triton = True

        def __init__(self, *, resident_head: bool = False) -> None:
            self.resident_head = resident_head

        def snapshot(self) -> dict[str, Any]:
            return {
                "fully_resident": True,
                "resident_exact_head": self.resident_head,
                "resident_exact_head_bytes": 400 if self.resident_head else 0,
            }

    engine = SimpleNamespace(
        backend="cuda-source-int8-compact-head",
        head_execution_mode="semantic-prefix-w8a16-top2-fp32-rerank",
        head_execution_abi="semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1",
        numerical_contract=(
            "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1+"
            "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
        ),
        direct_artifact=object(),
        composite_store=None,
        store=_CompactStore(),
        cfg=_FakeCudaEngine.cfg,
        target=SimpleNamespace(experimental_reranked_head=True),
        hidden=16,
        forward_last_top1=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        native_backends,
        "compiled_identity_from_dense_cuda_engine",
        lambda _engine: model,
    )

    device = _device(cuda=True)
    backend = DenseCudaComponentExecutionBackend(engine, device)
    capabilities = backend.capabilities(device)
    assert capabilities.output_modes == (OutputMode.NEXT_TOKEN_ARGMAX,)
    assert capabilities.promotion_status is PromotionStatus.EXPERIMENTAL
    workload = replace(
        _workload(backend.model),
        numerical_contract=engine.numerical_contract,
    )
    placement = backend.plan(backend.model, workload, device)
    assert placement.workspace_bytes == (
        7 + backend.body_workspace_bytes(workload) + backend.compact_head_workspace_bytes
    )
    assert backend.compact_head_workspace_bytes > 0

    engine.store = _CompactStore(resident_head=True)
    with pytest.raises(NativeBackendBindingError, match="changed after binding"):
        backend.capabilities(device)
    with pytest.raises(NativeBackendBindingError, match="forbids an expanded resident FP32 head"):
        DenseCudaComponentExecutionBackend(engine, device)


def test_compact_cuda_segmented_decode_is_identity_sealed_and_phase_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mrun.runtime.native_backends as native_backends

    model = compiled_identity_from_dense_cuda_engine(_FakeCudaEngine())

    class _CompactStore:
        compute_dtype = "bfloat16"
        require_triton = True

        @staticmethod
        def snapshot() -> dict[str, Any]:
            return {
                "fully_resident": True,
                "resident_exact_head": False,
                "resident_exact_head_bytes": 0,
            }

    base_contract = (
        "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1+"
        "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
    )
    target = SimpleNamespace(
        experimental_reranked_head=True,
        decode_attention_mode="segmented-flash-gqa-decode-v1",
        decode_attention_tile=64,
    )
    engine = SimpleNamespace(
        backend="cuda-source-int8-compact-head",
        head_execution_mode="semantic-prefix-w8a16-top2-fp32-rerank",
        head_execution_abi="semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1",
        numerical_contract=f"{base_contract}+segmented-flash-gqa-decode-v1",
        direct_artifact=object(),
        composite_store=None,
        store=_CompactStore(),
        cfg=_FakeCudaEngine.cfg,
        target=target,
        hidden=16,
        forward_last_top1=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        native_backends,
        "compiled_identity_from_dense_cuda_engine",
        lambda _engine: model,
    )

    device = _device(cuda=True)
    backend = DenseCudaComponentExecutionBackend(engine, device, max_batch_size=8)
    assert backend.capabilities(device).implementation_version == "5"
    workload = replace(
        _workload(backend.model, batch=8),
        numerical_contract=engine.numerical_contract,
    )
    flash_workspace = backend.body_workspace_bytes(workload)
    established_b1 = dense_cuda_body_workspace_bytes(
        SimpleNamespace(
            cfg=engine.cfg,
            store=engine.store,
            target=SimpleNamespace(decode_attention_mode="established"),
        ),
        max_batch_size=1,
        max_context_tokens=workload.max_context_tokens,
    )
    assert flash_workspace >= established_b1
    eager_b8 = dense_cuda_body_workspace_bytes(
        SimpleNamespace(
            cfg=engine.cfg,
            store=engine.store,
            target=SimpleNamespace(decode_attention_mode="established"),
        ),
        max_batch_size=8,
        max_context_tokens=workload.max_context_tokens,
    )
    assert flash_workspace < eager_b8

    fused_target = SimpleNamespace(
        experimental_reranked_head=True,
        decode_attention_mode="segmented-flash-gqa-decode-v1",
        decode_attention_tile=64,
        body_fusion_mode="residual-rms-swiglu-v1",
    )
    fused_engine = SimpleNamespace(
        **{
            **engine.__dict__,
            "target": fused_target,
            "numerical_contract": (
                f"{base_contract}+segmented-flash-gqa-decode-v1+"
                "residual-rms-swiglu-v1"
            ),
        }
    )
    fused_backend = DenseCudaComponentExecutionBackend(
        fused_engine,
        device,
        max_batch_size=8,
    )
    assert fused_backend.capabilities(device).implementation_version == "6"

    target.decode_attention_tile = 128
    with pytest.raises(NativeBackendBindingError, match="changed after binding"):
        backend.capabilities(device)


def test_qwen25_7b_4096_cuda_body_workspace_cannot_fit_in_old_unaccounted_slack() -> None:
    engine = SimpleNamespace(
        cfg={
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "head_dim": 128,
            "intermediate_size": 18_944,
        },
        store=SimpleNamespace(compute_dtype="bfloat16"),
    )

    workspace = dense_cuda_body_workspace_bytes(
        engine,
        max_batch_size=1,
        max_context_tokens=4096,
    )
    assert workspace == 4_429_283_328
    score_and_probability_lower_bound = 2 * 28 * 4096 * 4096 * 4
    assert workspace >= score_and_probability_lower_bound

    rtx_4080_bytes = 16_718_168_064
    compact_model_bytes = 7_623_395_328
    state_bytes_per_token = 57_344
    old_charge = compact_model_bytes + 32 * 4096 * state_bytes_per_token + 116_860 + 256 * 1024**2
    old_unaccounted_slack = rtx_4080_bytes - old_charge
    assert old_unaccounted_slack > 0
    assert workspace > old_unaccounted_slack
    assert old_charge + workspace > rtx_4080_bytes


def test_cuda_body_workspace_geometry_drift_fails_closed_after_binding() -> None:
    engine = _FakeCudaEngine()
    engine.cfg = dict(engine.cfg)
    device = _device(cuda=True)
    backend = DenseCudaComponentExecutionBackend(engine, device)
    engine.cfg["intermediate_size"] += 1

    with pytest.raises(NativeBackendBindingError, match="changed after binding"):
        backend.plan(backend.model, _workload(backend.model), device)


def test_direct_cuda_compiled_identity_separates_compact_head_abi() -> None:
    artifact = SimpleNamespace(
        source={
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
            "architecture": "qwen2",
            "manifest_sha256": _digest("1"),
            "model_fingerprint": _digest("2"),
            "tokenizer_custody_sha256": _digest("3"),
        },
        components={
            "body": {
                "codec": "qrow-int8",
                "layout": "row-major",
                "blobs": {
                    "weights.i8": {
                        "path": "body/weights.i8",
                        "sha256": _digest("4"),
                        "bytes": 100,
                    }
                },
            }
        },
        artifact_sha256=_digest("5"),
        recipe={"builder_abi": "builder-v1", "mapping_abi": "mapping-v1"},
    )
    common = {
        "direct_artifact": artifact,
        "store": SimpleNamespace(compute_dtype="bfloat16"),
        "assert_content_identity_unchanged": lambda: None,
        "cfg": {
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 16,
            "head_dim": 4,
            "max_position_embeddings": 64,
        },
        "name": "toy",
        "arch": "qwen2",
        "max_seq_len": 32,
        "semantic_token_count": 10,
    }
    exact = compiled_identity_from_dense_cuda_engine(
        SimpleNamespace(**common, head_execution_abi="exact-fp32-row-blocks-v1")
    )
    compact = compiled_identity_from_dense_cuda_engine(
        SimpleNamespace(
            **common,
            head_execution_abi="semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1",
        )
    )

    assert exact.compiler_abi == "builder-v1+mapping-v1+direct-no-qstore"
    assert compact.compiler_abi.endswith("+head-semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1")
    assert compact.fingerprint != exact.fingerprint


def test_bound_backends_reject_the_wrong_memory_fabric() -> None:
    with pytest.raises(NativeBackendBindingError, match="Apple unified-memory"):
        MlxComponentExecutionBackend(
            _FakeMlxEngine(),
            _device(cuda=True),
            state_layout=_mlx_layout(),
        )
    with pytest.raises(NativeBackendBindingError, match="CUDA device"):
        DenseCudaComponentExecutionBackend(_FakeCudaEngine(), _device(cuda=False))
