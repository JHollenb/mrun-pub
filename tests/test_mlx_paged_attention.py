from __future__ import annotations

from typing import Any

import pytest

from mrun.runtime import (
    MLX_PAGED_DECODE_ATTENTION_ABI,
    MLX_PAGED_DECODE_NUMERICAL_CONTRACT,
    MLX_PAGED_KV_CACHE_ABI,
    ComponentPlacement,
    DecodeWork,
    FallbackPolicy,
    MemoryDomain,
    MlxKVPagePool,
    MlxLayerCacheSpec,
    MlxMetalPagedDecodeAttention,
    MlxNativeRuntime,
    MlxPagedAttentionInput,
    MlxPagedDecodeAttentionError,
    MlxPagedDecodeAttentionLane,
    MlxPagedKVCacheFactory,
    MlxPagedKVError,
    MlxStateLayout,
    OutputMode,
    OutputRequest,
    PlacementPlan,
    PrefillWork,
    PromotionStatus,
    Residency,
    RuntimeRoute,
    StatePlacement,
    benchmark_mlx_paged_decode_attention,
)


def _mlx() -> Any:
    return pytest.importorskip("mlx.core")


def _layout(mx: Any, *, layers: int = 1, head_dim: int = 32) -> MlxStateLayout:
    spec = MlxLayerCacheSpec(
        kv_heads=1,
        key_head_dim=head_dim,
        value_head_dim=head_dim,
        key_dtype=mx.bfloat16,
        value_dtype=mx.bfloat16,
        key_element_bytes=2,
        value_element_bytes=2,
    )
    return MlxStateLayout(
        layers=(spec,) * layers,
        dtype_name="bfloat16",
        bytes_per_token=layers * 2 * head_dim * 2,
    )


def _decoder_attention(model: Any) -> Any:
    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        layers = tuple(model.model.layers)
    return layers[0].self_attn


def _pool(mx: Any, *, page_size: int = 2, page_count: int = 8) -> MlxKVPagePool:
    from mlx_lm.models.cache import create_attention_mask

    return MlxKVPagePool(
        mx=mx,
        create_attention_mask=create_attention_mask,
        page_size=page_size,
        page_count=page_count,
        kv_heads=1,
        key_head_dim=32,
        value_head_dim=32,
        key_dtype=mx.bfloat16,
        value_dtype=mx.bfloat16,
        paged_decode_attention=True,
    )


def test_metal_kernel_matches_dense_sdpa_for_nonmonotonic_partial_page_table() -> None:
    mx = _mlx()
    mx.random.seed(29)
    page_size = 4
    physical_pages = 4
    kv_heads = 2
    query_heads = 4
    head_dim = 32
    length = 10
    keys = (mx.random.normal((physical_pages, kv_heads, page_size, head_dim)) * 1.5).astype(
        mx.bfloat16
    )
    values = mx.random.normal((physical_pages, kv_heads, page_size, head_dim)).astype(mx.bfloat16)
    queries = mx.random.normal((1, query_heads, 1, head_dim)).astype(mx.bfloat16)
    logical_slots = (3, 0, 2)
    slots = mx.array(logical_slots, dtype=mx.int32)
    view = MlxPagedAttentionInput(
        keys=keys,
        values=values,
        page_slots=slots,
        length=length,
        logical_page_count=len(logical_slots),
        page_size=page_size,
        kv_heads=kv_heads,
        key_head_dim=head_dim,
        value_head_dim=head_dim,
        dtype=mx.bfloat16,
    )
    dense_keys = mx.concatenate(
        tuple(keys[slot : slot + 1] for slot in logical_slots),
        axis=2,
    )[:, :, :length, :]
    dense_values = mx.concatenate(
        tuple(values[slot : slot + 1] for slot in logical_slots),
        axis=2,
    )[:, :, :length, :]
    scale = head_dim**-0.5

    actual = MlxMetalPagedDecodeAttention(mx)(
        queries,
        view,
        scale=scale,
        query_heads=query_heads,
    )
    expected = mx.fast.scaled_dot_product_attention(
        queries,
        dense_keys,
        dense_values,
        scale=scale,
    )
    mx.eval(actual, expected)
    difference = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    mx.eval(difference)

    assert actual.shape == queries.shape
    assert bool(mx.all(mx.isfinite(actual)).item())
    # The lane intentionally has a different numerical identity: BF16 output rounding plus
    # Metal fast-exp online softmax is bounded here, but is not bitwise mlx-lm SDPA.
    assert float(mx.max(difference).item()) <= 0.01


def test_metal_kernel_and_page_view_fail_closed_on_unsupported_geometry() -> None:
    mx = _mlx()
    keys = mx.zeros((2, 2, 2, 32), dtype=mx.bfloat16)
    values = mx.zeros((2, 2, 2, 32), dtype=mx.bfloat16)
    slots = mx.array([0], dtype=mx.int32)
    view = MlxPagedAttentionInput(
        keys=keys,
        values=values,
        page_slots=slots,
        length=1,
        logical_page_count=1,
        page_size=2,
        kv_heads=2,
        key_head_dim=32,
        value_head_dim=32,
        dtype=mx.bfloat16,
    )
    kernel = MlxMetalPagedDecodeAttention(mx)

    with pytest.raises(MlxPagedDecodeAttentionError, match="shape"):
        kernel(mx.zeros((2, 2, 1, 32), dtype=mx.bfloat16), view, scale=1.0, query_heads=2)
    with pytest.raises(MlxPagedDecodeAttentionError, match="BF16"):
        kernel(mx.zeros((1, 2, 1, 32), dtype=mx.float16), view, scale=1.0, query_heads=2)
    with pytest.raises(MlxPagedDecodeAttentionError, match="divisible"):
        kernel(mx.zeros((1, 3, 1, 32), dtype=mx.bfloat16), view, scale=1.0, query_heads=3)
    with pytest.raises(ValueError, match="physical slabs"):
        MlxPagedAttentionInput(
            keys=mx.zeros((2, 2, 4, 32), dtype=mx.bfloat16),
            values=values,
            page_slots=slots,
            length=1,
            logical_page_count=1,
            page_size=2,
            kv_heads=2,
            key_head_dim=32,
            value_head_dim=32,
            dtype=mx.bfloat16,
        )
    with pytest.raises(ValueError, match="int32"):
        MlxPagedAttentionInput(
            keys=keys,
            values=values,
            page_slots=mx.array([0], dtype=mx.int64),
            length=1,
            logical_page_count=1,
            page_size=2,
            kv_heads=2,
            key_head_dim=32,
            value_head_dim=32,
            dtype=mx.bfloat16,
        )


def test_explicit_paged_k1_append_avoids_concat_and_preserves_cow_ledgers() -> None:
    mx = _mlx()
    pool = _pool(mx, page_size=2, page_count=8)
    source = pool.new_cache(capacity=8)
    branch = pool.new_cache(capacity=8)
    source_keys = mx.arange(4 * 32).reshape(1, 1, 4, 32).astype(mx.bfloat16)
    source_values = (source_keys + 200).astype(mx.bfloat16)
    dense_source_keys, dense_source_values = source.update_and_fetch(source_keys, source_values)
    mx.eval(dense_source_keys, dense_source_values)
    branch.copy_committed_prefix_from(source, 4)
    shared_tail = source.page_authorities[-1]
    assert branch.page_authorities[-1] == shared_tail
    assert branch.trim(1) == 1
    append_keys = mx.full((1, 1, 1, 32), 77, dtype=mx.bfloat16)
    append_values = mx.full((1, 1, 1, 32), 88, dtype=mx.bfloat16)
    before = pool.telemetry()

    view = branch.update_for_paged_decode(append_keys, append_values)
    mx.eval(view.keys, view.values, view.page_slots)
    after = pool.telemetry()

    assert branch.offset == source.offset == 4
    assert branch.page_authorities[-1] != shared_tail
    assert source.page_authorities[-1] == shared_tail
    assert after.concatenate_materializations == before.concatenate_materializations
    assert after.materialized_bytes == before.materialized_bytes
    assert after.paged_decode_appends == before.paged_decode_appends + 1
    assert after.page_table_materializations == before.page_table_materializations + 1
    assert after.dense_materializations_avoided == before.dense_materializations_avoided + 1
    assert after.cow_pages == before.cow_pages + 1
    assert after.cow_copy_tokens == before.cow_copy_tokens + 1
    assert after.reconciled
    no_op_before = pool.telemetry()
    assert branch.trim(0) == 0
    no_op_after = pool.telemetry()
    assert no_op_after.concatenate_materializations == no_op_before.concatenate_materializations
    assert no_op_after.materialized_bytes == no_op_before.materialized_bytes

    branch_keys = branch.keys
    branch_values = branch.values
    mx.eval(branch_keys, branch_values, dense_source_keys, dense_source_values)
    assert bool(mx.array_equal(dense_source_keys, source_keys).item())
    assert bool(mx.array_equal(dense_source_values, source_values).item())
    assert bool(mx.array_equal(branch_keys[..., :3, :], source_keys[..., :3, :]).item())
    assert bool(mx.array_equal(branch_values[..., :3, :], source_values[..., :3, :]).item())
    assert bool(mx.array_equal(branch_keys[..., 3:, :], append_keys).item())
    assert bool(mx.array_equal(branch_values[..., 3:, :], append_values).item())

    branch.release()
    source.release()
    assert pool.telemetry().reconciled
    pool.close()


def test_paged_decode_is_explicit_bf16_opt_in_and_flag_off_path_is_unchanged() -> None:
    mx = _mlx()
    from mlx_lm.models.cache import create_attention_mask

    ordinary = MlxKVPagePool(
        mx=mx,
        create_attention_mask=create_attention_mask,
        page_size=2,
        page_count=2,
        kv_heads=1,
        key_head_dim=32,
        value_head_dim=32,
        key_dtype=mx.bfloat16,
        value_dtype=mx.bfloat16,
    )
    cache = ordinary.new_cache(capacity=4)
    keys = mx.zeros((1, 1, 1, 32), dtype=mx.bfloat16)
    with pytest.raises(MlxPagedKVError, match="admitted paged-decode"):
        cache.update_for_paged_decode(keys, keys)
    assert cache.offset == 0
    returned_keys, returned_values = cache.update_and_fetch(keys, keys)
    mx.eval(returned_keys, returned_values)
    assert cache.offset == 1
    assert ordinary.telemetry().paged_decode_appends == 0
    cache.release()
    ordinary.close()

    with pytest.raises(MlxPagedKVError, match="BF16"):
        MlxKVPagePool(
            mx=mx,
            create_attention_mask=create_attention_mask,
            page_size=2,
            page_count=2,
            kv_heads=1,
            key_head_dim=32,
            value_head_dim=32,
            key_dtype=mx.float16,
            value_dtype=mx.float16,
            paged_decode_attention=True,
        )


def _exercise_wrapped_model(*, architecture: str, quantized: bool = False) -> None:
    mx = _mlx()
    from mlx_lm.models.cache import KVCache

    mx.random.seed(41)
    if architecture == "qwen2":
        from mlx_lm.models.qwen2 import Model, ModelArgs

        args = ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            num_key_value_heads=1,
        )
    else:
        from mlx_lm.models.mixtral import Model, ModelArgs

        args = ModelArgs(
            model_type="mixtral",
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_experts_per_tok=1,
            num_key_value_heads=1,
            num_local_experts=2,
        )
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    if quantized:
        import mlx.nn as nn

        nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    attention = _decoder_attention(model)
    if quantized:
        assert "Quantized" in type(attention.q_proj).__name__
    original_class = type(attention)
    weight_ids = tuple(
        id(getattr(attention, name).weight) for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    prompt = mx.array([[1, 2, 3]])
    token = mx.array([[4]])
    reference_cache = (KVCache(),)
    expected_prefill = model(prompt, cache=reference_cache)
    mx.eval(expected_prefill)
    expected_decode = model(token, cache=reference_cache)
    mx.eval(expected_decode)
    pool = _pool(mx)
    paged = pool.new_cache(capacity=8)
    lane = MlxPagedDecodeAttentionLane(
        model,
        mx=mx,
        architecture=architecture,
        state_layout=_layout(mx),
        page_size=2,
        page_count=8,
        base_numerical_contract="mlx-lm-dense-sdpa-test-v1",
    )
    try:
        assert attention is _decoder_attention(model)
        assert type(attention) is not original_class
        assert (
            tuple(
                id(getattr(attention, name).weight)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            == weight_ids
        )
        assert lane.identity.execution_abi == MLX_PAGED_DECODE_ATTENTION_ABI
        assert lane.identity.numerical_contract == MLX_PAGED_DECODE_NUMERICAL_CONTRACT
        assert lane.identity.layer_geometries == ((2, 1, 32),)
        assert lane.identity.promotion_status == "experimental"

        actual_prefill = model(prompt, cache=(paged,))
        mx.eval(actual_prefill)
        before_decode = pool.telemetry()
        actual_decode = model(token, cache=(paged,))
        mx.eval(actual_decode, *paged.evaluation_arrays())
        after_decode = pool.telemetry()

        assert bool(mx.allclose(actual_prefill, expected_prefill, rtol=0, atol=0).item())
        decode_tolerance = 0.05 if quantized else 0.01
        assert bool(
            mx.allclose(
                actual_decode,
                expected_decode,
                rtol=decode_tolerance,
                atol=decode_tolerance,
            ).item()
        )
        assert paged.offset == reference_cache[0].offset == 4
        assert (
            after_decode.concatenate_materializations == before_decode.concatenate_materializations
        )
        assert after_decode.materialized_bytes == before_decode.materialized_bytes
        assert after_decode.paged_decode_appends == before_decode.paged_decode_appends + 1
        telemetry = lane.telemetry()
        assert telemetry.installed_layers == 1
        assert telemetry.prefill_dense_calls == 1
        assert telemetry.paged_decode_calls == 1
        assert telemetry.attended_tokens == 4
        assert telemetry.logical_pages_read == 2
        assert telemetry.dense_materializations_avoided == 1
        assert telemetry.qwen2_calls == int(architecture == "qwen2")
        assert telemetry.mixtral_calls == int(architecture == "mixtral")
        assert telemetry.failures == 0
    finally:
        lane.close()
        paged.release()
        pool.close()

    assert type(attention) is original_class
    assert (
        tuple(
            id(getattr(attention, name).weight) for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        == weight_ids
    )


@pytest.mark.parametrize("architecture", ("qwen2", "mixtral"))
def test_real_tiny_model_wrapper_preserves_weights_and_routes_only_k1(architecture: str) -> None:
    _exercise_wrapped_model(architecture=architecture)


@pytest.mark.parametrize("architecture", ("qwen2", "mixtral"))
def test_real_tiny_q4_wrapper_preserves_packed_projection_weights(architecture: str) -> None:
    _exercise_wrapped_model(architecture=architecture, quantized=True)


def test_native_runtime_transaction_uses_physical_slabs_and_binds_execution_identity() -> None:
    mx = _mlx()
    from mlx_lm.models.qwen2 import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            num_key_value_heads=1,
        )
    )
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    engine = type(
        "TinyQwenEngine",
        (),
        {
            "_mx": mx,
            "model": model,
            "backend": "mlx-component",
            "context_size": 8,
            "semantic_token_count": 32,
            "numerical_contract": "mlx-lm-dense-sdpa-test-v1",
        },
    )()
    layout = _layout(mx)
    pool = _pool(mx)
    factory = MlxPagedKVCacheFactory((pool,))
    lane = MlxPagedDecodeAttentionLane(
        model,
        mx=mx,
        architecture="qwen2",
        state_layout=layout,
        page_size=2,
        page_count=8,
        base_numerical_contract=engine.numerical_contract,
    )
    state = StatePlacement(
        state_abi=MLX_PAGED_KV_CACHE_ABI,
        dtype="bfloat16",
        memory_domain=MemoryDomain.UNIFIED,
        bytes_per_token=layout.bytes_per_token,
        reserved_bytes=8 * layout.bytes_per_token,
        max_batch_size=1,
        max_context_tokens=8,
    )
    component = ComponentPlacement(
        allocation_id="allocation.body",
        component_ids=("body",),
        roles=("body",),
        codec_id="mlx-bf16",
        layout_id="mlx-linear",
        memory_domain=MemoryDomain.UNIFIED,
        residency=Residency.RESIDENT,
        physical_bytes=100,
    )
    workspace_bytes = pool.telemetry().physical_bytes - state.reserved_bytes
    placement = PlacementPlan(
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        device_fingerprint="c" * 64,
        workload_fingerprint="d" * 64,
        backend_id="mlx-component",
        device_id="metal:0",
        components=(component,),
        state=state,
        workspace_bytes=workspace_bytes,
        headroom_bytes=0,
        model_resident_bytes=100,
        total_reserved_bytes=100 + state.reserved_bytes + workspace_bytes,
        memory_budget_bytes=10_000,
        fully_resident=True,
        fallback_policy=FallbackPolicy.DENY,
        performance_claim_valid=True,
    )
    route = RuntimeRoute(
        runtime_id="runtime.paged-metal-test",
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        placement_fingerprint=placement.fingerprint,
        backend_id="mlx-component",
        device_id="metal:0",
        promotion_status=PromotionStatus.EXPERIMENTAL,
    )
    runtime = MlxNativeRuntime(
        engine,
        route=route,
        placement=placement,
        semantic_token_count=32,
        state_abi=MLX_PAGED_KV_CACHE_ABI,
        cache_factory=factory,
        state_layout=layout,
        paged_decode_attention_lane=lane,
        owns_cache_factory=True,
    )
    native_state = runtime.allocate_state(owner_id="chat", batch_size=1, capacity=8)
    prefill = runtime.prefill(
        PrefillWork(
            request_ids=("request",),
            token_rows=((1, 2, 3),),
            state=native_state,
            parent=native_state.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(prefill, (3,))
    before_decode = pool.telemetry()
    decode_parent = native_state.observe()

    decode = runtime.decode(
        DecodeWork(
            request_ids=("request",),
            token_rows=((4,),),
            state=native_state,
            parent=decode_parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    after_decode = pool.telemetry()

    assert runtime.route.promotion_status is PromotionStatus.EXPERIMENTAL
    assert runtime.route.effective_numerical_contract == MLX_PAGED_DECODE_NUMERICAL_CONTRACT
    assert runtime.route.execution_shape_fingerprint == lane.identity.fingerprint
    assert after_decode.concatenate_materializations == before_decode.concatenate_materializations
    assert after_decode.materialized_bytes == before_decode.materialized_bytes
    assert after_decode.paged_decode_appends == before_decode.paged_decode_appends + 1
    assert lane.telemetry().paged_decode_calls == 1
    counters = dict(runtime.telemetry().extra_counters)
    assert counters["paged_decode_attention_enabled"] == 1
    assert counters["paged_decode_attention_calls"] == 1
    assert counters["paged_kv_dense_materializations_avoided"] == 1

    runtime.abandon(decode)
    assert native_state.observe() == decode_parent
    assert (
        pool.telemetry().concatenate_materializations == after_decode.concatenate_materializations
    )
    assert pool.telemetry().reconciled

    original_kernel = lane._kernel  # noqa: SLF001 - injected post-append failure gate

    def fail_after_append(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected Metal dispatch failure")

    lane._kernel = fail_after_append  # type: ignore[assignment]  # noqa: SLF001
    with pytest.raises(RuntimeError, match="injected Metal dispatch failure"):
        runtime.decode(
            DecodeWork(
                request_ids=("request",),
                token_rows=((5,),),
                state=native_state,
                parent=decode_parent,
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            )
        )
    lane._kernel = original_kernel  # noqa: SLF001
    assert native_state.observe() == decode_parent
    assert lane.telemetry().failures == 1
    assert pool.telemetry().reconciled
    runtime.release_state(native_state)
    runtime.close()


def test_real_tiny_qwen_long_context_decode_matches_across_many_physical_pages() -> None:
    mx = _mlx()
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen2 import Model, ModelArgs

    mx.random.seed(53)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            num_key_value_heads=1,
        )
    )
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    prompt = (mx.arange(513, dtype=mx.int32) % 32)[None, :]
    token = mx.array([[17]])
    reference = (KVCache(),)
    reference_prefill = model(prompt, cache=reference)
    mx.eval(reference_prefill)
    expected = model(token, cache=reference)
    mx.eval(expected)
    pool = _pool(mx, page_size=64, page_count=16)
    paged = pool.new_cache(capacity=1024)
    lane = MlxPagedDecodeAttentionLane(
        model,
        mx=mx,
        architecture="qwen2",
        state_layout=_layout(mx),
        page_size=64,
        page_count=16,
        base_numerical_contract="mlx-lm-dense-sdpa-test-v1",
    )
    try:
        actual_prefill = model(prompt, cache=(paged,))
        mx.eval(actual_prefill)
        before = pool.telemetry()
        actual = model(token, cache=(paged,))
        mx.eval(actual, *paged.evaluation_arrays())
        after = pool.telemetry()

        assert bool(mx.allclose(actual_prefill, reference_prefill, rtol=0, atol=0).item())
        assert bool(mx.allclose(actual, expected, rtol=0.02, atol=0.02).item())
        assert lane.telemetry().attended_tokens == 514
        assert lane.telemetry().logical_pages_read == 9
        assert after.concatenate_materializations == before.concatenate_materializations
        assert after.materialized_bytes == before.materialized_bytes
        assert after.paged_decode_appends == before.paged_decode_appends + 1
    finally:
        lane.close()
        paged.release()
        pool.close()


def test_paged_attention_benchmark_records_long_context_crossover() -> None:
    mx = _mlx()
    points = benchmark_mlx_paged_decode_attention(
        mx,
        token_counts=(128, 32768),
        page_size=64,
        query_heads=14,
        kv_heads=2,
        head_dim=64,
        trials=7,
    )

    assert tuple(point.tokens for point in points) == (128, 32768)
    assert all(point.max_abs_error <= 0.002 for point in points)
    assert all(point.paged_median_ms > 0 for point in points)
    assert all(point.dense_materialize_median_ms > 0 for point in points)
    assert points[-1].paged_over_dense < 1.0
