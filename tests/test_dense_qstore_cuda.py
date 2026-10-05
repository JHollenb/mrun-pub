from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mrun.engine import dense_qstore_cuda as dense_module
from mrun.engine import open_engine
from mrun.engine.dense_qstore_cuda import (
    DenseQStoreCudaEngine,
    DenseQStoreKVCache,
    DenseQStoreTarget,
    DenseSelectedLastCUDAGraphExecutor,
    DenseSourceCudaInt8CompactHeadEngine,
    KVDelta,
)
from mrun.engine.kernels import dense_qstore_cuda as dense_kernels
from mrun.engine.kernels.composite_qstore import ComponentGraphError
from mrun.engine.kernels.dense_qstore_cuda import (
    CompactQRowPage,
    DenseQStore,
    DenseQStoreStats,
    fused_qrow_matmul,
    fused_qrow_reranked_argmax,
    fused_qrow_swiglu,
    fused_qrow_swiglu_reference,
    fused_qrow_top2,
    fused_residual_rms_norm,
    fused_residual_rms_norm_reference,
    rerank_qrow_candidates_fp32,
    segmented_decode_attention,
    segmented_decode_attention_reference,
    stable_attention,
    stable_attention_reference,
    stable_rms_norm,
    stable_rms_norm_reference,
)
from mrun.testing.dense_cuda_selected import _comparison, _project_full


def _delta(cache: DenseQStoreKVCache, token_count: int = 3) -> KVDelta:
    shape = (cache.batch_size, token_count, cache.num_kv_heads, cache.head_dim)
    keys = tuple(
        torch.arange(np.prod(shape), dtype=cache.dtype).view(shape) + 100 * layer
        for layer in range(cache.num_layers)
    )
    values = tuple(key + 1000 for key in keys)
    return KVDelta(
        parent_epoch=cache.epoch,
        parent_lengths=tuple(int(value) for value in cache.lengths),
        cache_id=cache.cache_id,
        keys=keys,
        values=values,
        token_count=token_count,
    )


def test_device_binding_requires_one_exact_cuda_ordinal() -> None:
    assert not dense_module._same_torch_device(torch.device("cuda:0"), torch.device("cuda"))
    assert dense_module._same_torch_device(torch.device("cuda:0"), torch.device("cuda:0"))
    assert dense_module._same_torch_device(torch.device("cpu"), torch.device("cpu"))
    assert not dense_module._same_torch_device(
        torch.device("cuda:1"),
        torch.device("cuda:0"),
    )
    assert not dense_module._same_torch_device(torch.device("cpu"), torch.device("cuda"))


def test_compact_qrow_cpu_fallback_matches_materialized_reference() -> None:
    page = CompactQRowPage(
        name="toy",
        start_row=0,
        end_row=3,
        in_features=4,
        codes=torch.tensor(
            [[1, -2, 3, 4], [-1, 2, 0, 3], [4, 1, -3, 2]],
            dtype=torch.int8,
        ),
        scales=torch.tensor([0.5, 0.25, 0.125]),
    )
    activations = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 2.0, 3.0]])

    actual, implementation = fused_qrow_matmul(page, activations)
    expected = activations @ (page.codes.float() * page.scales[:, None]).T

    assert implementation == "reference-materialized"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert page.compact_bytes < page.expanded_fp32_bytes


def test_paired_w8a16_swiglu_cpu_reference_preserves_bf16_boundaries() -> None:
    gate = CompactQRowPage(
        name="gate",
        start_row=0,
        end_row=3,
        in_features=4,
        codes=torch.tensor([[1, -2, 3, 4], [-1, 2, 0, 3], [4, 1, -3, 2]], dtype=torch.int8),
        scales=torch.tensor([0.5, 0.25, 0.125]),
    )
    up = CompactQRowPage(
        name="up",
        start_row=0,
        end_row=3,
        in_features=4,
        codes=torch.tensor([[2, 1, -1, 3], [4, -2, 1, 0], [-3, 2, 2, 1]], dtype=torch.int8),
        scales=torch.tensor([0.25, 0.125, 0.5]),
    )
    activations = torch.tensor(
        [[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 2.0, 3.0]],
        dtype=torch.bfloat16,
    )

    expected = fused_qrow_swiglu_reference(gate, up, activations)
    actual, implementation = fused_qrow_swiglu(gate, up, activations)

    assert implementation == "reference-materialized-paired-w8a16-swiglu-bf16-v1"
    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="requires CUDA BF16 with Triton"):
        fused_qrow_swiglu(gate, up, activations, require_triton=True)


def test_fused_residual_rms_reference_normalizes_the_rounded_bf16_residual() -> None:
    residual = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [0.125, 4.0, -2.0, 0.75]],
        dtype=torch.bfloat16,
    )
    update = torch.tensor(
        [[0.00390625, 0.01171875, -0.0078125, 0.015625], [0.02, -0.03, 0.04, 0.05]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([0.5, 1.5, -0.75, 2.0], dtype=torch.float32)
    expected_residual = (residual.float() + update.float()).to(torch.bfloat16)
    expected_normalized = stable_rms_norm_reference(expected_residual, weight, 1e-6)

    reference_residual, reference_normalized = fused_residual_rms_norm_reference(
        residual,
        update,
        weight,
        1e-6,
    )
    actual_residual, actual_normalized, implementation = fused_residual_rms_norm(
        residual,
        update,
        weight,
        1e-6,
    )

    assert implementation == "torch-residual-bf16-rms-reference-v1"
    assert torch.equal(reference_residual, expected_residual)
    assert torch.equal(actual_residual, expected_residual)
    torch.testing.assert_close(reference_normalized, expected_normalized, rtol=0, atol=0)
    torch.testing.assert_close(actual_normalized, expected_normalized, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="requires CUDA BF16 with Triton"):
        fused_residual_rms_norm(
            residual,
            update,
            weight,
            1e-6,
            require_triton=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_body_fusions_match_their_bf16_references() -> None:
    generator = torch.Generator().manual_seed(53)
    rows, hidden, intermediate = 5, 64, 96
    activations = torch.randn((rows, hidden), generator=generator).to(
        device="cuda",
        dtype=torch.bfloat16,
    )

    def page(name: str) -> CompactQRowPage:
        codes = torch.randint(
            -16,
            17,
            (intermediate, hidden),
            generator=generator,
            dtype=torch.int8,
        ).cuda()
        scales = (torch.rand(intermediate, generator=generator) / 16).cuda()
        return CompactQRowPage(
            name=name,
            start_row=0,
            end_row=intermediate,
            in_features=hidden,
            codes=codes,
            scales=scales,
        )

    gate = page("gate")
    up = page("up")
    expected_swiglu = fused_qrow_swiglu_reference(gate, up, activations)
    actual_swiglu, swiglu_impl = fused_qrow_swiglu(
        gate,
        up,
        activations,
        require_triton=True,
    )
    assert swiglu_impl == "triton-paired-w8a16-swiglu-bf16-v1"
    torch.testing.assert_close(actual_swiglu, expected_swiglu, rtol=2e-2, atol=2e-2)

    residual = torch.randn((rows, hidden), generator=generator).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    update = torch.randn((rows, hidden), generator=generator).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(hidden, generator=generator).cuda()
    expected_residual, expected_norm = fused_residual_rms_norm_reference(
        residual,
        update,
        weight,
        1e-6,
    )
    actual_residual, actual_norm, norm_impl = fused_residual_rms_norm(
        residual,
        update,
        weight,
        1e-6,
        require_triton=True,
    )
    assert norm_impl == "triton-residual-bf16-rms-v1"
    assert torch.equal(actual_residual, expected_residual)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-2, atol=2e-2)


def test_dense_selected_gate_projection_preserves_requested_row_order() -> None:
    full = [
        torch.arange(24, dtype=torch.float32).reshape(3, 8),
        torch.arange(24, 48, dtype=torch.float32).reshape(3, 8),
    ]
    projected = _project_full(full, (6, 1, 4))

    assert projected.tolist() == [[22.0, 17.0, 20.0], [46.0, 41.0, 44.0]]
    parity = _comparison(projected, projected.clone())
    assert parity["allclose"] and parity["exact"]
    assert parity["winner_offsets_exact"]


def test_fused_top2_and_fp32_rerank_cpu_reference() -> None:
    page = CompactQRowPage(
        name="lm_head",
        start_row=11,
        end_row=18,
        in_features=4,
        codes=torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [-1, 0, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.int8,
        ),
        scales=torch.ones(7),
    )
    activations = torch.tensor([[2.0, 3.0, -1.0, 0.5], [-2.0, 0.0, 4.0, 1.0]])

    candidates, values, implementation, working_bytes = fused_qrow_top2(page, activations)
    reranked, reranked_values = rerank_qrow_candidates_fp32(
        page,
        activations,
        candidates,
    )
    fused_reranked, _, fused_implementation, _ = fused_qrow_reranked_argmax(
        page,
        activations,
    )

    assert candidates.tolist() == [[15, 17], [16, 13]]
    assert values.tolist() == [[5.0, 5.0], [6.0, 4.0]]
    assert reranked.tolist() == [15, 16]
    assert reranked_values.tolist() == [5.0, 6.0]
    assert torch.equal(fused_reranked, reranked)
    assert implementation == "reference-materialized-top2"
    assert fused_implementation.endswith("+fp32-candidate-rerank-v1")
    assert working_bytes == 0


def test_fused_reranked_head_masks_physical_padding_before_shortlist() -> None:
    page = CompactQRowPage(
        name="lm_head",
        start_row=0,
        end_row=4,
        in_features=2,
        codes=torch.tensor([[2, 0], [0, 2], [1, 1], [100, 100]], dtype=torch.int8),
        scales=torch.ones(4),
    )
    activations = torch.ones((1, 2))

    unmasked, _, _, _ = fused_qrow_reranked_argmax(page, activations)
    masked, _, _, _ = fused_qrow_reranked_argmax(
        page,
        activations,
        semantic_row_count=3,
    )

    assert int(unmasked.item()) == 3
    assert int(masked.item()) == 0
    for invalid, error in (
        (True, TypeError),
        (1, ValueError),
        (5, ValueError),
        (2.5, TypeError),
    ):
        with pytest.raises(error, match="semantic_row_count"):
            fused_qrow_top2(page, activations, semantic_row_count=invalid)  # type: ignore[arg-type]


def test_fused_reranked_head_requires_cuda_when_requested() -> None:
    page = CompactQRowPage(
        name="lm_head",
        start_row=0,
        end_row=2,
        in_features=2,
        codes=torch.eye(2, dtype=torch.int8),
        scales=torch.ones(2),
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fused_qrow_reranked_argmax(
            page,
            torch.ones((1, 2), dtype=torch.bfloat16),
            require_triton=True,
        )


def test_compact_page_cache_deduplicates_tied_weight_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mrun.engine.kernels.dense_qstore_cuda as kernels

    store = object.__new__(DenseQStore)
    store.blocks = {
        "embed": {"kind": "qrow"},
        "lm_head": {"alias": "embed"},
    }
    store.compact_cache_budget = 1024
    store.compact_cache = kernels.OrderedDict()
    store.compact_cache_bytes = 0
    store.stats = DenseQStoreStats()
    page = CompactQRowPage(
        name="embed",
        start_row=0,
        end_row=2,
        in_features=2,
        codes=torch.eye(2, dtype=torch.int8),
        scales=torch.ones(2),
    )
    loaded: list[str] = []

    def fake_load(_store: object, name: str) -> CompactQRowPage:
        loaded.append(name)
        return page

    monkeypatch.setattr(kernels, "load_compact_qrow_page", fake_load)

    embedding = store.compact_page("embed")
    head = store.compact_page("lm_head")

    assert embedding is head
    assert loaded == ["embed"]
    assert list(store.compact_cache) == ["embed"]
    assert store.stats.page_loads == 1
    assert store.stats.cache_hits == 1


def test_prebind_keeps_extension_pages_non_evictable_and_alias_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mrun.engine.kernels.dense_qstore_cuda as kernels

    store = object.__new__(DenseQStore)
    store.blocks = {
        "embed": {
            "kind": "qrow",
            "shape": [2, 2],
            "w_off": 0,
            "w_len": 4,
            "s_off": 0,
            "s_len": 8,
        },
        "lm_head": {"alias": "embed"},
        "norm": {"kind": "fp32", "shape": [1], "e_off": 0, "e_len": 4},
    }
    store.w = np.asarray([1, 2, 3, 4], dtype=np.int8)
    store.s = np.ones(2, dtype=np.float32)
    store.e = np.asarray([2.0], dtype=np.float32)
    store.device = "cpu"
    store.compute_dtype = torch.float32
    store.stats = DenseQStoreStats()
    store._prebound_pages = {}
    store._prebound_compact_bytes = 0
    store._prebound_fp32 = {}
    store._prebound_fp32_bytes = 0
    store.compact_cache = kernels.OrderedDict()
    store.compact_cache_budget = 0
    store.compact_cache_bytes = 0
    store.fp32_aux_cache = {}
    store.fp32_aux_cache_bytes = 0
    store.pin_fp32_aux = False
    store.resident_exact_heads = {}
    store.resident_exact_head_budget_bytes = 0
    store.resident_exact_head_bytes = 0
    store.stable_block_m = 16
    page = CompactQRowPage(
        name="embed",
        start_row=0,
        end_row=2,
        in_features=2,
        codes=torch.ones((2, 2), dtype=torch.int8),
        scales=torch.ones(2),
    )
    loaded: list[str] = []

    def fake_load(_store: object, name: str) -> CompactQRowPage:
        loaded.append(name)
        return page

    monkeypatch.setattr(kernels, "load_compact_qrow_page", fake_load)
    result = store.prebind(("embed", "lm_head", "norm"), max_resident_bytes=16)

    assert result == {"qrow_pages": 1, "fp32_tensors": 1, "resident_bytes": 16}
    assert loaded == ["embed"]
    assert store.compact_page("embed") is store.compact_page("lm_head")
    assert store.fp32("norm").data_ptr() == store._prebound_fp32["norm"].data_ptr()
    assert store.stats_snapshot()["prebound_page_entries"] == 1
    assert store.stats_snapshot()["prebound_fp32_entries"] == 1
    assert store.compact_cache == {}


def test_prebind_checks_budget_before_loading_any_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = object.__new__(DenseQStore)
    store.blocks = {
        "projection": {
            "kind": "qrow",
            "shape": [2, 2],
            "w_off": 0,
            "w_len": 4,
            "s_off": 0,
            "s_len": 8,
        }
    }
    store.device = "cpu"
    store.stats = DenseQStoreStats()
    store._prebound_pages = {}
    store._prebound_compact_bytes = 0
    store._prebound_fp32 = {}
    store._prebound_fp32_bytes = 0
    loaded = []
    monkeypatch.setattr(
        dense_kernels,
        "load_compact_qrow_page",
        lambda *_args: loaded.append(True),
    )

    with pytest.raises(MemoryError, match="explicit residency budget"):
        store.prebind(("projection",), max_resident_bytes=11)
    assert loaded == []


def test_selected_rows_fp32_loads_only_named_rows_without_narrowing() -> None:
    store = object.__new__(DenseQStore)
    store.blocks = {
        "embed": {
            "kind": "qrow",
            "shape": [4, 3],
            "w_off": 0,
            "s_off": 0,
        },
        "lm_head": {"alias": "embed"},
    }
    store.w = np.asarray(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            -1,
            -2,
            -3,
            7,
            8,
            9,
        ],
        dtype=np.int8,
    )
    store.s = np.asarray([0.5, 0.25, 2.0, 0.125], dtype=np.float32)
    store.device = "cpu"
    store.compute_dtype = torch.bfloat16
    store.compact_cache_bytes = 0
    store.stats = DenseQStoreStats()

    rows = store.selected_rows_fp32("lm_head", [2, 0])

    assert rows.dtype is torch.float32
    torch.testing.assert_close(
        rows,
        torch.tensor([[-2.0, -4.0, -6.0], [0.5, 1.0, 1.5]]),
        rtol=0,
        atol=0,
    )
    assert store.stats.selected_head_calls == 1
    assert store.stats.selected_head_rows == 2
    assert store.stats.selected_head_compact_bytes == 2 * (3 + 4)


def _capture_binding_store() -> DenseQStore:
    store = object.__new__(DenseQStore)
    store.blocks = {
        "embed": {
            "kind": "qrow",
            "shape": [4, 3],
            "w_off": 0,
            "s_off": 0,
        },
        "lm_head": {"alias": "embed"},
        "norm.final": {
            "kind": "fp32",
            "shape": [3],
            "e_off": 0,
        },
        "norm.alias": {"alias": "norm.final"},
    }
    store.w = np.arange(12, dtype=np.int8)
    store.s = np.asarray([0.5, 0.25, 2.0, 0.125], dtype=np.float32)
    store.e = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    store.device = "cpu"
    store.compute_dtype = torch.float32
    store.require_triton = False
    store.stable_block_m = 16
    store.compact_cache_budget = 0
    store.compact_cache = dense_kernels.OrderedDict()
    store.compact_cache_bytes = 0
    store.stats = DenseQStoreStats()
    return store


def test_capture_bindings_pin_alias_deduplicated_resources_and_close() -> None:
    store = _capture_binding_store()
    bindings = store.prepare_capture_bindings(
        qrow_names=("embed", "lm_head"),
        fp32_names=("norm.final", "norm.alias"),
        selected_head_rows=(2, 0),
        max_resident_bytes=64,
    )

    assert bindings.resident_bytes == 64
    assert bindings.compact_page("embed") is bindings.compact_page("lm_head")
    assert bindings.fp32("norm.final") is bindings.fp32("norm.alias")
    first = bindings.selected_rows_fp32("lm_head", (2, 0))
    second = bindings.selected_rows_fp32("lm_head", (2, 0))
    assert first.data_ptr() == second.data_ptr()
    assert bindings.evidence()["capture_aliases_deduplicated"] == 2
    assert bindings.evidence()["capture_residency_non_evictable"] is True

    bindings.close()
    with pytest.raises(RuntimeError, match="closed"):
        bindings.compact_page("embed")


def test_capture_bindings_refuse_insufficient_residency_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _capture_binding_store()
    loaded: list[str] = []
    monkeypatch.setattr(
        dense_kernels,
        "load_compact_qrow_page",
        lambda _store, name: loaded.append(name),
    )

    with pytest.raises(MemoryError, match="exceeds its budget"):
        store.prepare_capture_bindings(
            qrow_names=("embed", "lm_head"),
            fp32_names=("norm.final", "norm.alias"),
            selected_head_rows=(2, 0),
            max_resident_bytes=63,
        )
    assert loaded == []


def test_resident_arena_shares_pages_and_rebinds_selected_rows() -> None:
    store = _capture_binding_store()
    arena = store.prepare_resident_arena(
        qrow_names=("embed",),
        fp32_names=("norm.final",),
        max_resident_bytes=64,
    )
    first = arena.bind_selected_rows((2, 0))
    second = arena.bind_selected_rows((1, 3))

    assert first.compact_page("embed") is second.compact_page("embed")
    assert first.compact_page("lm_head") is second.compact_page("lm_head")
    torch.testing.assert_close(
        first.selected_rows_fp32("lm_head", torch.tensor([2, 0])),
        store.selected_rows_fp32("lm_head", (2, 0)),
    )
    torch.testing.assert_close(
        second.selected_rows_fp32("lm_head", torch.tensor([1, 3])),
        store.selected_rows_fp32("lm_head", (1, 3)),
    )
    assert arena.active_bindings == 2
    with pytest.raises(RuntimeError, match="live graph bindings"):
        arena.close()
    first.close()
    second.close()
    assert arena.active_bindings == 0
    arena.close()


def test_row_stable_reductions_match_scalar_references() -> None:
    generator = torch.Generator().manual_seed(17)
    activations = torch.randn((3, 4, 8), generator=generator)
    weight = torch.randn((8,), generator=generator)

    norm, implementation = stable_rms_norm(activations, weight, 1e-6)
    expected_norm = stable_rms_norm_reference(activations, weight, 1e-6)
    assert implementation == "torch-row-reference"
    torch.testing.assert_close(norm, expected_norm, rtol=0, atol=0)

    query = torch.randn((2, 2, 4, 4), generator=generator)
    key = torch.randn((2, 5, 2, 4), generator=generator)
    value = torch.randn((2, 5, 2, 4), generator=generator)
    lengths = torch.tensor([2, 3])
    context, attention_impl = stable_attention(query, key, value, lengths)
    expected_context = stable_attention_reference(query, key, value, lengths)
    assert attention_impl == "torch-row-reference"
    torch.testing.assert_close(context, expected_context, rtol=0, atol=0)


def test_segmented_decode_reference_matches_joined_gqa_with_fixed_slot_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(29)
    query = torch.randn((2, 1, 4, 4), generator=generator)
    cache_key = torch.randn((4, 6, 2, 4), generator=generator)
    cache_value = torch.randn((4, 6, 2, 4), generator=generator)
    key_new = torch.randn((2, 1, 2, 4), generator=generator)
    value_new = torch.randn((2, 1, 2, 4), generator=generator)
    lengths = torch.tensor([3, 1])
    cache_rows = torch.tensor([3, 1], dtype=torch.int32)
    cache_key[3, 3:] = 10_000
    cache_value[3, 3:] = -10_000
    cache_key[1, 1:] = -20_000
    cache_value[1, 1:] = 20_000

    joined_key = torch.zeros((2, 4, 2, 4))
    joined_value = torch.zeros_like(joined_key)
    for request, (slot, committed) in enumerate(zip(cache_rows, lengths, strict=True)):
        slot_index = int(slot)
        count = int(committed)
        joined_key[request, :count] = cache_key[slot_index, :count]
        joined_value[request, :count] = cache_value[slot_index, :count]
        joined_key[request, count] = key_new[request, 0]
        joined_value[request, count] = value_new[request, 0]
    expected = stable_attention_reference(query, joined_key, joined_value, lengths)

    def reject_join(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("segmented decode attempted to concatenate its K/V segments")

    monkeypatch.setattr(torch, "cat", reject_join)
    reference = segmented_decode_attention_reference(
        query,
        cache_key,
        cache_value,
        key_new,
        value_new,
        lengths,
        cache_row_indices=cache_rows,
    )
    actual, implementation = segmented_decode_attention(
        query,
        cache_key,
        cache_value,
        key_new,
        value_new,
        lengths,
        cache_row_indices=cache_rows,
        sequence_tile=16,
    )

    assert implementation == "torch-segmented-online-softmax-gqa-decode-v1"
    torch.testing.assert_close(reference, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_segmented_decode_matches_ragged_gqa_reference() -> None:
    torch.manual_seed(71)
    query = torch.randn((3, 1, 28, 128), device="cuda", dtype=torch.bfloat16)
    cache_key = torch.randn((4, 257, 4, 128), device="cuda", dtype=torch.bfloat16)
    cache_value = torch.randn_like(cache_key)
    key_new = torch.randn((3, 1, 4, 128), device="cuda", dtype=torch.bfloat16)
    value_new = torch.randn_like(key_new)
    lengths = torch.tensor([0, 63, 257], device="cuda", dtype=torch.long)
    cache_rows = torch.tensor([3, 1, 2], device="cuda", dtype=torch.long)
    expected = segmented_decode_attention_reference(
        query,
        cache_key,
        cache_value,
        key_new,
        value_new,
        lengths,
        cache_row_indices=cache_rows,
    )

    actual, implementation = segmented_decode_attention(
        query,
        cache_key,
        cache_value,
        key_new,
        value_new,
        lengths,
        cache_row_indices=cache_rows,
        require_triton=True,
        sequence_tile=64,
        max_committed=257,
    )

    assert implementation == "triton-segmented-flash-gqa-decode-v1"
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_segmented_decode_attention_fails_closed_outside_bounded_abi() -> None:
    query = torch.ones((2, 1, 4, 4))
    cache_key = torch.ones((2, 4, 2, 4))
    cache_value = torch.ones_like(cache_key)
    key_new = torch.ones((2, 1, 2, 4))
    value_new = torch.ones_like(key_new)

    with pytest.raises(ValueError, match="exactly one new token"):
        segmented_decode_attention_reference(
            query.repeat(1, 2, 1, 1),
            cache_key,
            cache_value,
            key_new.repeat(1, 2, 1, 1),
            value_new.repeat(1, 2, 1, 1),
            [1, 1],
        )
    with pytest.raises(ValueError, match="power of two"):
        segmented_decode_attention(
            query,
            cache_key,
            cache_value,
            key_new,
            value_new,
            [1, 1],
            sequence_tile=24,
        )
    with pytest.raises(ValueError, match="does not cover"):
        segmented_decode_attention(
            query,
            cache_key,
            cache_value,
            key_new,
            value_new,
            [3, 1],
            max_committed=2,
        )
    with pytest.raises(ValueError, match="outside committed KV storage"):
        segmented_decode_attention_reference(
            query,
            cache_key,
            cache_value,
            key_new,
            value_new,
            [1, 1],
            cache_row_indices=torch.tensor([0, 2]),
        )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        segmented_decode_attention(
            query,
            cache_key,
            cache_value,
            key_new,
            value_new,
            [1, 1],
            require_triton=True,
        )


def test_transactional_cache_commits_only_accepted_prefixes() -> None:
    cache = DenseQStoreKVCache(
        num_layers=2,
        batch_size=2,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    delta = _delta(cache)
    stats = cache.commit(delta, [3, 1])

    assert tuple(cache.lengths) == (3, 1)
    assert stats.accepted_counts == (3, 1)
    assert stats.kv_write_bytes == (3 + 1) * 2 * 1 * 2 * 2 * 4
    torch.testing.assert_close(cache.keys[0][0, :3], delta.keys[0][0])
    torch.testing.assert_close(cache.keys[0][1, :1], delta.keys[0][1, :1])
    assert torch.count_nonzero(cache.keys[0][1, 1:]) == 0

    with pytest.raises(RuntimeError, match="stale KV delta"):
        cache.commit(delta, [0, 0])


def test_cache_prefix_templates_can_populate_reusable_slots() -> None:
    source = DenseQStoreKVCache(
        num_layers=1,
        batch_size=1,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    source.commit(_delta(source, token_count=2), [2])
    target = DenseQStoreKVCache(
        num_layers=1,
        batch_size=3,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )

    stats = target.install_requests(
        source,
        source_indices=[0, 0, 0],
        target_indices=[0, 1, 2],
    )

    assert tuple(target.lengths) == (2, 2, 2)
    assert stats.installed_lengths == (2, 2, 2)
    for request in range(3):
        torch.testing.assert_close(target.keys[0][request, :2], source.keys[0][0, :2])


class _ToyStore:
    def __init__(self) -> None:
        self.man = {"arch": "qwen2"}
        self.cfg = {
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "vocab_size": 8,
        }
        self.device = "cpu"
        self.compute_dtype = torch.float32
        self.require_triton = False
        generator = torch.Generator().manual_seed(23)
        self.weights = {
            "embed": torch.randn((8, 4), generator=generator),
            "L0.q": torch.randn((4, 4), generator=generator) / 4,
            "L0.k": torch.randn((2, 4), generator=generator) / 4,
            "L0.v": torch.randn((2, 4), generator=generator) / 4,
            "L0.o": torch.randn((4, 4), generator=generator) / 4,
            "L0.gate": torch.randn((6, 4), generator=generator) / 4,
            "L0.up": torch.randn((6, 4), generator=generator) / 4,
            "L0.down": torch.randn((4, 6), generator=generator) / 4,
            "lm_head": torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 1, 0, 0],
                    [-1, 0, 1, 0],
                    [0, -1, 0, 1],
                    [1, 0, 1, 0],
                ],
                dtype=torch.float32,
            ),
        }
        self.extras = {
            "L0.ln1": torch.ones(4),
            "L0.ln2": torch.ones(4),
            "norm.final": torch.ones(4),
        }

    def has(self, name: str) -> bool:
        return name in self.weights or name in self.extras

    def fp32(self, name: str) -> torch.Tensor:
        return self.extras[name]

    def matmul(self, name: str, activations: torch.Tensor) -> torch.Tensor:
        return activations @ self.weights[name].T

    def embed_rows(self, _name: str, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self.weights["embed"].index_select(0, torch.as_tensor(ids).reshape(-1))

    def row_blocks(self, _name: str, bs: int = 8192) -> Any:
        del bs
        yield 0, 8, self.weights["lm_head"]

    def selected_rows_fp32(
        self,
        _name: str,
        ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        return self.weights["lm_head"].index_select(
            0,
            torch.as_tensor(ids, dtype=torch.long),
        )

    def compact_page(self, _name: str) -> CompactQRowPage:
        return CompactQRowPage(
            name="lm_head",
            start_row=0,
            end_row=8,
            in_features=4,
            codes=self.weights["lm_head"].to(torch.int8),
            scales=torch.ones(8),
        )

    stable_block_m = 16


def test_dense_engine_selects_explicit_linked_image_and_prebinds_overlay(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "toy-linked"
    image.mkdir()
    (image / "manifest.json").write_text(
        '{"model_name":"tiny-qwen","arch":"qwen2",'
        '"linked_image":{"extension_id":"toy-linked-v1",'
        '"overlay_blocks":["L0.q"]}}'
    )
    captured: dict[str, Any] = {}

    class _DenseToyStore(_ToyStore):
        def __init__(self, key: str, *, root, **_kwargs: Any) -> None:
            super().__init__()
            self.directory = (root / key).resolve()
            self.man.update(
                {
                    "model_name": "tiny-qwen",
                    "linked_image": {
                        "extension_id": "toy-linked-v1",
                        "overlay_blocks": ["L0.q"],
                    },
                }
            )
            self.prebound: tuple[str, ...] | None = None

        def prebind(self, names) -> dict[str, int]:
            self.prebound = tuple(names)
            return {"qrow_pages": 1, "fp32_tensors": 0, "resident_bytes": 1}

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(
        dense_module,
        "resolve_model",
        lambda _name: SimpleNamespace(
            name="tiny-qwen",
            family="qwen2",
            hf_id="test/tiny-qwen",
        ),
    )

    class _Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __len__(self) -> int:
            return 8

    monkeypatch.setattr(dense_module, "load_tokenizer", lambda _spec: _Tokenizer())
    monkeypatch.setattr(
        dense_module,
        "_concrete_cuda_device",
        lambda _device: torch.device("cpu"),
    )
    monkeypatch.setattr(dense_module, "DenseQStore", _DenseToyStore)

    engine = DenseQStoreCudaEngine(
        "tiny-qwen",
        store_path=image,
        linked_extension_id="toy-linked-v1",
        prebind_linked_extension=True,
        device="cuda",
        require_triton=False,
    )
    assert engine.store_path == image.resolve()
    assert engine.linked_extension_id == "toy-linked-v1"
    assert engine.store.prebound == ("L0.q",)
    engine.close()
    assert captured["closed"] is True


class _ToyCaptureBindings:
    def __init__(self, store: _ToyStore, selected: tuple[int, ...]) -> None:
        self.store = store
        self.selected = selected
        self.weights = store.selected_rows_fp32("lm_head", selected)
        self.closed = False
        self.stable_addresses_verified = False
        self.resident_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*store.weights.values(), *store.extras.values(), self.weights)
        )

    def has(self, name: str) -> bool:
        return self.store.has(name)

    def verify_stable_addresses(self) -> None:
        if self.closed:
            raise RuntimeError("closed")
        self.stable_addresses_verified = True

    def fp32(self, name: str) -> torch.Tensor:
        return self.store.fp32(name)

    def matmul(self, name: str, activations: torch.Tensor) -> torch.Tensor:
        return self.store.matmul(name, activations)

    def embed_rows(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        return self.store.embed_rows(name, ids)

    def selected_rows_fp32(
        self,
        name: str,
        ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        assert name == "lm_head"
        assert tuple(ids) == self.selected
        return self.weights

    def evidence(self) -> dict[str, Any]:
        return {
            "capture_resident_bytes": self.resident_bytes,
            "capture_estimated_resident_bytes": self.resident_bytes,
            "capture_residency_budget_bytes": None,
            "capture_qrow_logical_count": 8,
            "capture_qrow_physical_count": 8,
            "capture_fp32_logical_count": 3,
            "capture_fp32_physical_count": 3,
            "capture_aliases_deduplicated": 0,
            "capture_selected_head_rows": len(self.selected),
            "capture_stable_address_count": 12,
            "capture_resource_addresses_verified": self.stable_addresses_verified,
            "capture_residency_non_evictable": True,
        }

    def close(self) -> None:
        self.closed = True


class _FakeCapturedGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class _FakeGraphDriver:
    requires_cuda = False
    name = "fake-cuda-graph"

    def __init__(self) -> None:
        self.operation_calls = 0
        self.graph = _FakeCapturedGraph()

    def capture(
        self,
        operation: Any,
        *,
        device: torch.device,
        warmup: int,
    ) -> Any:
        assert device.type == "cpu"
        output = None
        for _ in range(warmup + 1):
            self.operation_calls += 1
            output = operation()
        assert output is not None
        return dense_module._CapturedSelectedCall(
            graph=self.graph,
            output=output,
            capture_stream=object(),
        )


class _ReplayingFakeGraphDriver:
    requires_cuda = False
    name = "replaying-fake-cuda-graph"

    def capture(
        self,
        operation: Any,
        *,
        device: torch.device,
        warmup: int,
    ) -> Any:
        assert device.type == "cpu"
        for _ in range(warmup):
            operation()
        output = operation().detach().clone()

        class _Graph:
            def replay(self) -> None:
                output.copy_(operation())

        return dense_module._CapturedSelectedCall(
            graph=_Graph(),
            output=output,
            capture_stream=object(),
        )


def _toy_capture_engine() -> tuple[Any, _ToyStore]:
    store = _ToyStore()

    def prepare_capture_bindings(
        *,
        qrow_names: tuple[str, ...],
        fp32_names: tuple[str, ...],
        selected_head_rows: tuple[int, ...],
        max_resident_bytes: int | None,
    ) -> _ToyCaptureBindings:
        assert qrow_names == (
            "embed",
            "L0.q",
            "L0.k",
            "L0.v",
            "L0.o",
            "L0.gate",
            "L0.up",
            "L0.down",
        )
        assert fp32_names == ("norm.final", "L0.ln1", "L0.ln2")
        assert max_resident_bytes is None
        return _ToyCaptureBindings(store, selected_head_rows)

    store.prepare_capture_bindings = prepare_capture_bindings  # type: ignore[attr-defined]
    target = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    engine = SimpleNamespace(
        target=target,
        store=store,
        max_seq_len=8,
        cfg=store.cfg,
    )
    return engine, store


def test_mocked_cuda_graph_executor_warms_captures_replays_and_closes() -> None:
    engine, _store = _toy_capture_engine()
    driver = _FakeGraphDriver()
    rows = (np.asarray([1, 2, 3]), np.asarray([3, 2, 1]))
    selected = (6, 1, 4)
    expected = engine.target.forward_selected_last(
        torch.as_tensor(np.stack(rows)),
        engine.target.empty_cache(2),
        selected,
    ).selected_logits

    executor = DenseSelectedLastCUDAGraphExecutor(
        engine,
        rows,
        selected,
        warmup=2,
        _driver=driver,
    )
    first = executor.execute()
    second = executor.execute()
    eager_control = executor.execute_eager_control()

    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    torch.testing.assert_close(eager_control, expected, rtol=0, atol=0)
    assert driver.operation_calls == 3
    assert driver.graph.replays == 2
    assert executor.evidence["graph_replay"] is True
    assert executor.evidence["capture_executed"] is True
    assert executor.evidence["capture_replay_count"] == 2
    assert executor.evidence["capture_matched_eager_control_count"] == 1
    assert executor.evidence["capture_output_shape"] == [2, 3]
    with pytest.raises(TypeError):
        executor.evidence["graph_replay"] = False  # type: ignore[index]

    executor.close()
    assert executor.evidence["capture_executor_closed"] is True
    with pytest.raises(RuntimeError, match="closed"):
        executor.execute()
    with pytest.raises(RuntimeError, match="closed"):
        executor.execute_eager_control()


def test_rebindable_graph_executor_mutates_payloads_without_changing_addresses() -> None:
    engine, store = _toy_capture_engine()

    class _DynamicBindings(_ToyCaptureBindings):
        rebindable_head = True

        def selected_rows_fp32(self, name: str, ids: Any) -> torch.Tensor:
            return self.store.selected_rows_fp32(name, ids)

    def prepare_dynamic(
        *,
        qrow_names: tuple[str, ...],
        fp32_names: tuple[str, ...],
        selected_head_rows: tuple[int, ...],
        max_resident_bytes: int | None,
        rebindable_head: bool,
    ) -> _DynamicBindings:
        del qrow_names, fp32_names, max_resident_bytes
        assert rebindable_head is True
        return _DynamicBindings(store, selected_head_rows)

    store.prepare_capture_bindings = prepare_dynamic  # type: ignore[attr-defined]
    executor = DenseSelectedLastCUDAGraphExecutor(
        engine,
        (np.asarray([1, 2, 3]),),
        (6, 1, 4),
        warmup=1,
        rebindable=True,
        _driver=_ReplayingFakeGraphDriver(),
    )
    addresses = executor._static_addresses  # noqa: SLF001 - address-stability gate
    generation = executor.rebind(
        (np.asarray([3, 1, 2]),),
        (5, 2, 0),
        request_id="request-b",
    )
    expected = engine.target.forward_selected_last(
        torch.tensor([[3, 1, 2]]),
        engine.target.empty_cache(1),
        (5, 2, 0),
    ).selected_logits
    actual = executor.execute(
        expected_generation=generation,
        request_id="request-b",
    )
    torch.testing.assert_close(actual, expected)
    assert executor._static_addresses == addresses  # noqa: SLF001
    with pytest.raises(RuntimeError, match="generation is stale"):
        executor.execute(expected_generation=generation - 1, request_id="request-b")
    with pytest.raises(RuntimeError, match="does not own"):
        executor.execute(expected_generation=generation, request_id="request-a")
    executor.close()


def test_capture_target_fails_closed_for_full_logits_and_stateful_decode() -> None:
    engine, store = _toy_capture_engine()
    bindings = _ToyCaptureBindings(store, (6, 1, 4))
    rows = torch.tensor([[1, 2, 3]])
    cache = engine.target.empty_cache(1)
    positions = torch.zeros(1, dtype=torch.long)

    with pytest.raises(RuntimeError, match="only selected-row"):
        engine.target._forward_impl(
            rows,
            cache,
            capture_bindings=bindings,  # type: ignore[arg-type]
            capture_position_base=positions,
            synchronize=False,
        )

    cache.lengths[:] = 1
    with pytest.raises(RuntimeError, match="stateful/decode"):
        engine.target.forward_selected_last(
            rows,
            cache,
            (6, 1, 4),
            capture_bindings=bindings,  # type: ignore[arg-type]
            capture_position_base=positions,
            synchronize=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_torch_cuda_graph_captures_and_replays_toy_selected_score() -> None:
    class _CudaToyStore(_ToyStore):
        def __init__(self) -> None:
            super().__init__()
            self.device = "cuda"
            self.weights = {name: tensor.cuda() for name, tensor in self.weights.items()}
            self.extras = {name: tensor.cuda() for name, tensor in self.extras.items()}

        def selected_rows_fp32(
            self,
            _name: str,
            ids: list[int] | tuple[int, ...],
        ) -> torch.Tensor:
            return self.weights["lm_head"].index_select(
                0,
                torch.as_tensor(ids, dtype=torch.long, device="cuda"),
            )

    store = _CudaToyStore()

    def prepare_capture_bindings(
        *,
        qrow_names: tuple[str, ...],
        fp32_names: tuple[str, ...],
        selected_head_rows: tuple[int, ...],
        max_resident_bytes: int | None,
    ) -> _ToyCaptureBindings:
        del qrow_names, fp32_names, max_resident_bytes
        return _ToyCaptureBindings(store, selected_head_rows)

    store.prepare_capture_bindings = prepare_capture_bindings  # type: ignore[attr-defined]
    target = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    engine = SimpleNamespace(
        target=target,
        store=store,
        max_seq_len=8,
        cfg=store.cfg,
    )
    rows = (np.asarray([1, 2, 3]), np.asarray([3, 2, 1]))
    selected = (6, 1, 4)
    eager = target.forward_selected_last(
        torch.as_tensor(np.stack(rows), device="cuda"),
        target.empty_cache(2),
        selected,
    ).selected_logits.cpu()

    executor = DenseSelectedLastCUDAGraphExecutor(
        engine,
        rows,
        selected,
        warmup=2,
    )
    first = executor.execute()
    second = executor.execute()
    eager_control = executor.execute_eager_control()

    torch.testing.assert_close(first, eager, rtol=0, atol=0)
    torch.testing.assert_close(second, eager, rtol=0, atol=0)
    torch.testing.assert_close(eager_control, eager, rtol=0, atol=0)
    assert executor.evidence["capture_backend"] == "torch.cuda.CUDAGraph"
    assert executor.evidence["capture_replay_count"] == 2
    executor.close()


def test_toy_dense_target_runs_prefill_commit_and_decode() -> None:
    target = DenseQStoreTarget(_ToyStore(), max_seq_len=8)  # type: ignore[arg-type]
    cache = target.empty_cache(batch_size=2)
    prefill = target.forward(torch.tensor([[1, 2, 3], [3, 2, 1]]), cache)

    assert prefill.top1.shape == (2, 3)
    assert prefill.hidden.shape == (2, 3, 4)
    assert prefill.delta.token_count == 3
    cache.commit(prefill.delta, [3, 3])

    decode = target.forward(prefill.top1[:, -1:], cache)
    cache.commit(decode.delta, [1, 1])
    assert tuple(cache.lengths) == (4, 4)
    assert decode.top1.shape == (2, 1)
    assert target.reduction_calls["torch-batched-attention"] == 2
    assert target.reduction_calls["torch-batched-rms"] == 6


def test_opt_in_body_fusion_matches_bf16_reference_route_and_counts_calls() -> None:
    class _BodyFusionToyStore(_ToyStore):
        def __init__(self) -> None:
            super().__init__()
            self.compute_dtype = torch.bfloat16
            self.weights = {
                name: weight.to(torch.bfloat16) for name, weight in self.weights.items()
            }
            generator = torch.Generator().manual_seed(41)
            for name in ("L0.gate", "L0.up"):
                self.weights[name] = torch.randint(
                    -4,
                    5,
                    self.weights[name].shape,
                    generator=generator,
                    dtype=torch.int8,
                ).to(torch.bfloat16)

        def compact_page(self, name: str) -> CompactQRowPage:
            weight = self.weights[name]
            return CompactQRowPage(
                name=name,
                start_row=0,
                end_row=int(weight.shape[0]),
                in_features=int(weight.shape[1]),
                codes=weight.to(torch.int8),
                scales=torch.ones(int(weight.shape[0])),
            )

    store = _BodyFusionToyStore()
    rows = torch.tensor([[1, 2, 3], [3, 2, 1]])
    established = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    candidate = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        body_fusion_mode="residual-rms-swiglu-v1",
    )

    expected = established.forward(rows, established.empty_cache(2))
    actual = candidate.forward(rows, candidate.empty_cache(2))

    torch.testing.assert_close(actual.hidden, expected.hidden, rtol=2e-2, atol=2e-2)
    assert candidate.body_fusion_mode == "residual-rms-swiglu-v1"
    assert candidate.fused_residual_rms_calls == 2
    assert candidate.fused_gate_up_swiglu_calls == 1
    assert candidate.reduction_calls["torch-residual-bf16-rms-reference-v1"] == 2
    assert candidate.reduction_calls["reference-materialized-paired-w8a16-swiglu-bf16-v1"] == 1
    assert candidate.reduction_calls["torch-batched-rms"] == 1


def test_body_fusion_mode_and_numerical_contract_are_explicit_and_fail_closed() -> None:
    base = "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1"
    assert (
        dense_module._execution_numerical_contract(  # noqa: SLF001
            base,
            "segmented-flash-gqa-decode-v1",
            "residual-rms-swiglu-v1",
        )
        == f"{base}+segmented-flash-gqa-decode-v1+residual-rms-swiglu-v1"
    )
    assert (
        dense_module._execution_numerical_contract(  # noqa: SLF001
            base,
            "established",
            "established",
        )
        == base
    )
    with pytest.raises(ValueError, match="body_fusion_mode"):
        DenseQStoreCudaEngine("toy", body_fusion_mode="implicit-fusion")
    with pytest.raises(RuntimeError, match="requires Triton execution"):
        DenseQStoreCudaEngine(
            "toy",
            body_fusion_mode="residual-rms-swiglu-v1",
            require_triton=False,
        )
    with pytest.raises(RuntimeError, match="requires Triton execution"):
        dense_module.DenseSourceCudaInt8Engine(
            "toy",
            source_artifact="not-opened-because-admission-fails",
            body_fusion_mode="residual-rms-swiglu-v1",
            require_triton=False,
        )


def test_segmented_decode_opt_in_preserves_prefill_and_transactional_commit() -> None:
    store = _ToyStore()
    target = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        decode_attention_mode="segmented-flash-gqa-decode-v1",
        decode_attention_tile=16,
    )
    cache = target.empty_cache(batch_size=2)
    prefill = target.forward(torch.tensor([[1, 2, 3], [3, 2, 1]]), cache)
    cache.commit(prefill.delta, [3, 3])
    reference_cache = cache.clone()
    reference = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    decode_ids = prefill.top1[:, -1:]
    expected = reference.forward(decode_ids, reference_cache)
    epoch_before = cache.epoch
    lengths_before = cache.lengths.copy()
    keys_before = tuple(tensor.clone() for tensor in cache.keys)
    values_before = tuple(tensor.clone() for tensor in cache.values)

    provisional = target.forward(decode_ids, cache)

    assert cache.epoch == epoch_before
    assert np.array_equal(cache.lengths, lengths_before)
    for actual, expected_key in zip(cache.keys, keys_before, strict=True):
        torch.testing.assert_close(actual, expected_key, rtol=0, atol=0)
    for actual, expected_value in zip(cache.values, values_before, strict=True):
        torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(provisional.hidden, expected.hidden, rtol=1e-5, atol=1e-5)
    assert target.segmented_decode_prefill_fallback_calls == 1
    assert target.segmented_decode_attention_calls == 1
    assert target.reduction_calls["torch-batched-attention"] == 1
    assert target.reduction_calls["torch-segmented-online-softmax-gqa-decode-v1"] == 1

    cache.commit(provisional.delta, [1, 1])
    assert tuple(cache.lengths) == (4, 4)
    for layer in range(cache.num_layers):
        torch.testing.assert_close(
            cache.keys[layer][:, 3:4],
            provisional.delta.keys[layer],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache.values[layer][:, 3:4],
            provisional.delta.values[layer],
            rtol=0,
            atol=0,
        )


def test_forward_decode_slots_reads_shared_pool_rows_without_committing() -> None:
    store = _ToyStore()
    target = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        decode_attention_mode="segmented-flash-gqa-decode-v1",
        decode_attention_tile=16,
    )
    pool = target.empty_cache(batch_size=4)
    pool.commit(_delta(pool, token_count=3), [2, 1, 3, 0])
    pool.batch_size = 2
    pool.lengths = np.asarray([3, 2], dtype=np.int64)
    pool.row_indices = torch.tensor([2, 0], dtype=torch.int64)  # type: ignore[attr-defined]
    engine = object.__new__(DenseQStoreCudaEngine)
    engine.target = target
    engine._last_cache = None
    before_epoch = pool.epoch
    before_lengths = pool.lengths.copy()
    before_keys = tuple(tensor.clone() for tensor in pool.keys)
    before_values = tuple(tensor.clone() for tensor in pool.values)

    result = engine.forward_decode_slots(torch.tensor([[1], [2]]), pool)

    assert result.top1.shape == (2, 1)
    assert result.delta.token_count == 1
    assert engine._last_cache is pool
    assert pool.epoch == before_epoch
    assert np.array_equal(pool.lengths, before_lengths)
    for actual, expected in zip(pool.keys, before_keys, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(pool.values, before_values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match=r"shape \[B,1\]"):
        engine.forward_decode_slots(torch.tensor([[1, 2], [2, 1]]), pool)

    established_engine = object.__new__(DenseQStoreCudaEngine)
    established_engine.target = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    established_engine._last_cache = None
    with pytest.raises(RuntimeError, match="segmented-flash-gqa-decode-v1"):
        established_engine.forward_decode_slots(torch.tensor([[1], [2]]), pool)


def test_segmented_decode_mode_is_explicit_and_validated_before_model_loading() -> None:
    target = DenseQStoreTarget(_ToyStore(), max_seq_len=8)  # type: ignore[arg-type]
    assert target.decode_attention_mode == "established"
    with pytest.raises(ValueError, match="decode_attention_mode"):
        DenseQStoreTarget(  # type: ignore[arg-type]
            _ToyStore(),
            decode_attention_mode="implicit-fast-path",
        )
    with pytest.raises(ValueError, match="decode_attention_tile"):
        DenseQStoreTarget(  # type: ignore[arg-type]
            _ToyStore(),
            decode_attention_tile=24,
        )
    with pytest.raises(ValueError, match="decode_attention_mode"):
        DenseQStoreCudaEngine("toy", decode_attention_mode="implicit-fast-path")


def test_toy_dense_target_selected_head_skips_full_vocabulary_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ToyStore()
    reference_target = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    rows = torch.tensor([[1, 2, 3], [3, 2, 1]])
    reference = reference_target.forward(
        rows,
        reference_target.empty_cache(2),
        return_logits=True,
    )
    assert reference.logits is not None

    selected_target = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]

    def fail_full_head(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("selected-row execution touched the full vocabulary head")

    monkeypatch.setattr(selected_target, "_head", fail_full_head)
    selected = selected_target.forward_selected_last(
        rows,
        selected_target.empty_cache(2),
        (6, 1, 4),
    )

    torch.testing.assert_close(
        selected.selected_logits,
        reference.logits[:, -1, [6, 1, 4]],
        rtol=0,
        atol=0,
    )


def test_experimental_reranked_head_matches_toy_streamed_head() -> None:
    store = _ToyStore()
    established = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    candidate = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        experimental_reranked_head=True,
    )
    hidden = torch.tensor([[[2.0, 3.0, -1.0, 0.5], [-2.0, 0.0, 4.0, 1.0]]])

    expected, _ = established._head(hidden, return_logits=False)
    actual, logits = candidate._head(hidden, return_logits=False)

    assert torch.equal(actual, expected)
    assert logits is None
    assert candidate.reduction_calls["reference-materialized-top2+fp32-candidate-rerank-v1"] == 1


def test_reranked_head_masks_padded_rows_and_keeps_exact_logit_diagnostic() -> None:
    store = _ToyStore()
    store.weights["lm_head"][7] = torch.tensor([100.0, 100.0, 100.0, 100.0])
    exact = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        semantic_token_count=7,
    )
    candidate = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        experimental_reranked_head=True,
        semantic_token_count=7,
    )
    hidden = torch.ones((1, 2, 4))

    expected, expected_logits = exact._head(hidden, return_logits=True)
    actual, candidate_logits = candidate._head(hidden, return_logits=False)
    diagnostic_top1, diagnostic_logits = candidate._head(hidden, return_logits=True)

    assert torch.equal(actual, expected)
    assert not bool(torch.any(actual == 7))
    assert candidate_logits is None
    assert torch.equal(diagnostic_top1, expected)
    torch.testing.assert_close(diagnostic_logits, expected_logits, rtol=0, atol=0)
    assert candidate.reranked_head_calls == 1


def test_last_top1_forward_preserves_full_body_and_kv_delta() -> None:
    store = _ToyStore()
    target = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        experimental_reranked_head=True,
    )
    rows = torch.tensor([[1, 2, 3], [4, 5, 6]])

    full = target.forward(rows, target.empty_cache(2))
    bounded = target.forward_last_top1(rows, target.empty_cache(2))

    assert bounded.top1.shape == (2, 1)
    assert torch.equal(bounded.top1, full.top1[:, -1:])
    torch.testing.assert_close(bounded.hidden, full.hidden, rtol=0, atol=0)
    assert bounded.delta.token_count == full.delta.token_count == 3
    for actual, expected in zip(bounded.delta.keys, full.delta.keys, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(bounded.delta.values, full.delta.values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dense_backend_alias_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    import mrun.engine.dense_qstore_cuda as dense_module

    class _FakeEngine:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

    monkeypatch.setattr(dense_module, "DenseQStoreCudaEngine", _FakeEngine)
    engine = open_engine("toy", backend="dense-cuda", marker=17)
    assert isinstance(engine, _FakeEngine)
    assert engine.model_name == "toy"
    assert engine.kwargs == {"marker": 17}


def test_dense_capabilities_expose_stateful_contract() -> None:
    engine = object.__new__(DenseQStoreCudaEngine)
    capabilities = engine.capabilities()

    assert capabilities.logits and capabilities.logits_batch
    assert capabilities.generation and capabilities.generation_batch
    assert capabilities.persistent_kv and capabilities.transactional_kv
    assert capabilities.speculative_blocks and capabilities.compact_fused_weights
    assert capabilities.compiled_workplan and capabilities.graph_replay
    assert capabilities.approximate_quantized
    assert not capabilities.autograd
    assert not capabilities.mlp_acts
    assert not capabilities.grouped_moe


def test_dense_component_engine_wires_closed_provider_and_semantic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __len__(self) -> int:
            return 7

    class _View(_ToyStore):
        output_contract = "full_logits"

        def __init__(self) -> None:
            super().__init__()
            self.identity_checks = 0

        def assert_content_identity_unchanged(self) -> None:
            self.identity_checks += 1

    class _Composite:
        def __init__(self, graph_path: str, **kwargs: Any) -> None:
            captured["graph_path"] = graph_path
            captured["kwargs"] = kwargs
            self.graph = SimpleNamespace(model_name="tiny-qwen", architecture="qwen2")
            self.vocab = SimpleNamespace(token_count=7)
            self.composite_fingerprint_sha256 = "a" * 64
            self.view = _View()
            self.tokenizer_checks = 0
            self.close_calls = 0

        def for_contract(self, output_contract: Any) -> _View:
            captured["output_contract"] = output_contract
            return self.view

        def validate_tokenizer(self, _tokenizer: Any) -> None:
            self.tokenizer_checks += 1

        def snapshot(self) -> dict[str, Any]:
            return {
                "provider_backend": "dense-qstore-cuda",
                "residency_contract": "component-compact-lru+streamed-exact-head",
                "body_fully_resident": False,
                "providers": {
                    "body": {
                        "store_stats": {"peak_compact_resident_bytes": 0},
                    }
                },
            }

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(
        dense_module,
        "resolve_model",
        lambda _name: SimpleNamespace(
            name="tiny-qwen",
            family="qwen2",
            hf_id="test/tiny-qwen",
        ),
    )
    monkeypatch.setattr(dense_module, "load_tokenizer", lambda _spec: _Tokenizer())
    monkeypatch.setattr(dense_module, "_concrete_cuda_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(dense_module, "CompositeQStore", _Composite)

    engine = DenseQStoreCudaEngine(
        "tiny-qwen",
        component_graph="model-graph.json",
        output_contract="full_logits",
        compact_cache_mb=9.0,
        component_cache_mb={"body": 8.0, "egress": 1.0},
        require_triton=False,
    )

    assert captured["graph_path"] == "model-graph.json"
    assert captured["kwargs"] == {
        "cache_mb": 9.0,
        "component_cache_mb": {"body": 8.0, "egress": 1.0},
        "compute_dtype": "bf16",
        "provider_backend": "dense-qstore-cuda",
        "provider_device": "cpu",
        "provider_require_triton": False,
        "provider_stable_block_m": 16,
        "provider_pin_fp32_aux": True,
    }
    assert engine.semantic_token_count == 7
    assert engine.component_output_contract == "full_logits"
    assert not engine.capabilities().graph_replay
    stats = engine.runtime_stats()
    assert stats["residency_contract"] == "component-compact-lru+streamed-exact-head"
    assert stats["semantic_token_count"] == 7
    assert stats["configured_vocab_rows"] == 8

    composite = engine.composite_store
    assert composite is not None
    engine.assert_content_identity_unchanged()
    assert composite.view.identity_checks == 1
    assert composite.tokenizer_checks == 2
    with pytest.raises(NotImplementedError, match="selected-row component"):
        engine.selected_last_logits_batch([np.asarray([0])], [0])
    with pytest.raises(NotImplementedError, match="CUDA Graph selected-row"):
        engine.prepare_selected_last_cuda_graph([np.asarray([0])], [0])
    engine.close()
    engine.close()
    assert composite.close_calls == 1


@pytest.mark.parametrize(
    ("output_contract", "reranked", "message"),
    [
        ("selected_token_rows", False, "full-logit contract"),
        ("full_logits", True, "reranked head"),
    ],
)
def test_dense_component_engine_rejects_unsupported_head_routes_before_open(
    output_contract: str,
    reranked: bool,
    message: str,
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        DenseQStoreCudaEngine(
            "tiny-qwen",
            component_graph="model-graph.json",
            output_contract=output_contract,
            experimental_reranked_head=reranked,
        )


@pytest.mark.parametrize("failure", ["model", "architecture", "tokenizer"])
def test_dense_component_engine_fails_closed_on_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    instances: list[Any] = []

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __len__(self) -> int:
            return 8

    class _Composite:
        def __init__(self, _graph_path: str, **_kwargs: Any) -> None:
            self.graph = SimpleNamespace(
                model_name="wrong" if failure == "model" else "tiny-qwen",
                architecture="llama" if failure == "architecture" else "qwen2",
            )
            self.vocab = SimpleNamespace(token_count=8)
            self.view = _ToyStore()
            self.view.output_contract = "full_logits"
            self.closed = False
            instances.append(self)

        def for_contract(self, _output_contract: Any) -> _ToyStore:
            return self.view

        def validate_tokenizer(self, _tokenizer: Any) -> None:
            if failure == "tokenizer":
                raise ComponentGraphError("runtime tokenizer mismatch")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        dense_module,
        "resolve_model",
        lambda _name: SimpleNamespace(
            name="tiny-qwen",
            family="qwen2",
            hf_id="test/tiny-qwen",
        ),
    )
    monkeypatch.setattr(dense_module, "load_tokenizer", lambda _spec: _Tokenizer())
    monkeypatch.setattr(dense_module, "_concrete_cuda_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(dense_module, "CompositeQStore", _Composite)

    with pytest.raises(ComponentGraphError, match="does not match|tokenizer mismatch"):
        DenseQStoreCudaEngine(
            "tiny-qwen",
            component_graph="model-graph.json",
            require_triton=False,
        )
    assert instances[0].closed


def test_dense_target_masks_padded_rows_from_tokens_and_greedy_selection() -> None:
    store = _ToyStore()
    store.weights["lm_head"][7] = torch.tensor([100.0, 100.0, 100.0, 100.0])
    target = DenseQStoreTarget(
        store,  # type: ignore[arg-type]
        max_seq_len=8,
        semantic_token_count=7,
    )
    hidden = torch.ones((1, 1, 4))

    top1, logits = target._head(hidden, return_logits=True)
    sampled_top1, sampled_logits = target._head(
        hidden,
        return_logits=True,
        last_logits_only=True,
    )

    assert logits is not None and logits.shape[-1] == 8
    assert int(logits[..., 7].item()) == 400
    assert int(top1.item()) != 7
    assert sampled_logits is not None and sampled_logits.shape == (1, 1, 7)
    assert torch.equal(sampled_logits, logits[..., :7])
    assert torch.equal(sampled_top1, top1)
    with pytest.raises(ValueError, match="padded model rows are not tokens"):
        target.forward(torch.tensor([[7]]), target.empty_cache(1))


def test_dense_generation_bypasses_domain_sync_only_for_internal_tokens() -> None:
    store = _ToyStore()
    target = DenseQStoreTarget(store, max_seq_len=8)  # type: ignore[arg-type]
    engine = object.__new__(DenseQStoreCudaEngine)
    engine.target = target
    engine.max_seq_len = 8
    engine._last_cache = None

    generated = engine.generate_ids(torch.tensor([[0]], dtype=torch.long), max_new_tokens=4)

    assert generated.shape == (1, 4)
    assert target.token_domain_checks == 1
    assert target.trusted_generated_token_bypasses == 3
    with pytest.raises(RuntimeError, match="device-local int64"):
        target._forward_trusted_generated(
            torch.tensor([[0]], dtype=torch.int32),
            target.empty_cache(1),
        )


def test_dense_resident_exact_head_preserves_established_row_block_contract() -> None:
    class _ResidentToyStore(_ToyStore):
        def resident_exact_head_fp32(self, _name: str) -> torch.Tensor:
            return self.weights["lm_head"]

    streamed_store = _ToyStore()
    resident_store = _ResidentToyStore()
    resident_store.weights = streamed_store.weights
    resident_store.extras = streamed_store.extras
    streamed = DenseQStoreTarget(streamed_store, max_seq_len=8)  # type: ignore[arg-type]
    resident = DenseQStoreTarget(resident_store, max_seq_len=8)  # type: ignore[arg-type]
    hidden = torch.tensor([[[2.0, 3.0, -1.0, 0.5], [-2.0, 0.0, 4.0, 1.0]]])

    expected_top1, expected_logits = streamed._head(hidden, return_logits=True)
    actual_top1, actual_logits = resident._head(hidden, return_logits=True)

    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_top1, expected_top1)
    assert resident.resident_exact_head_calls == 1
    assert resident.streamed_exact_head_calls == 0


def test_failed_row_stable_model_gate_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="22/34 established B1 trace tokens"):
        DenseQStoreCudaEngine("toy", stable_reductions=True)


def test_named_row_stable_contract_passes_the_legacy_guard() -> None:
    with pytest.raises(FileNotFoundError, match="no dense int8 QStore"):
        DenseQStoreCudaEngine(
            "qwen2.5-0.5b",
            stores_dir="/definitely/missing",
            numerical_contract="row-stable-triton-v1",
        )


@pytest.mark.parametrize("contract", ["row-stable-triton-v2", "row-stable-v2", "batch-invariant"])
def test_row_stable_v2_contract_is_accepted_before_model_loading(contract: str) -> None:
    with pytest.raises(FileNotFoundError, match="no dense int8 QStore"):
        DenseQStoreCudaEngine(
            "qwen2.5-0.5b",
            stores_dir="/definitely/missing",
            numerical_contract=contract,
        )


def test_unknown_dense_numerical_contract_fails_before_model_loading() -> None:
    with pytest.raises(ValueError, match="numerical_contract must be one of"):
        DenseQStoreCudaEngine("toy", numerical_contract="not-a-contract")


def test_compact_direct_cuda_head_refuses_non_triton_execution() -> None:
    with pytest.raises(ValueError, match="requires Triton execution"):
        DenseSourceCudaInt8CompactHeadEngine("toy", require_triton=False)
