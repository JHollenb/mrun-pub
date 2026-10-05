from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mrun.engine.kernels.segmented_gqa_decode import (
    scatter_segmented_decode_kv,
    segmented_gqa_decode,
    segmented_gqa_decode_cow,
    segmented_gqa_decode_cow_reference,
    segmented_gqa_decode_reference,
)
from mrun.engine.qwen3_moe_cuda import (
    LayerWeights,
    Qwen3MoeDecodeRuntime,
    RaggedDecodeEvidence,
    _device_matches_runtime,
)
from mrun.runtime.qwen3_moe_continuous import (
    Qwen3MoeDecodeBatch,
    Qwen3MoeNativeRaggedDecodeExecutor,
)


def test_runtime_device_alias_accepts_indexed_cuda_but_not_another_index() -> None:
    assert _device_matches_runtime(torch.device("cuda:0"), "cuda")
    assert _device_matches_runtime(torch.device("cuda:1"), "cuda")
    assert _device_matches_runtime(torch.device("cuda:0"), "cuda:0")
    assert not _device_matches_runtime(torch.device("cuda:1"), "cuda:0")
    assert not _device_matches_runtime(torch.device("cpu"), "cuda")


def test_segmented_gqa_cpu_reference_matches_expanded_sdpa_only_as_oracle() -> None:
    torch.manual_seed(41)
    query = torch.randn(2, 4, 3)
    key = torch.randn(3, 2, 6, 3)
    value = torch.randn(3, 2, 6, 3)
    slots = torch.tensor([2, 0], dtype=torch.long)
    lengths = torch.tensor([3, 5], dtype=torch.long)

    actual = segmented_gqa_decode_reference(query, key, value, slots, lengths)
    expected_rows: list[torch.Tensor] = []
    for row, (slot, length) in enumerate(((2, 3), (0, 5))):
        expected_rows.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[row][None, :, None],
                key[slot, :, :length].repeat_interleave(2, dim=0)[None],
                value[slot, :, :length].repeat_interleave(2, dim=0)[None],
                dropout_p=0.0,
                is_causal=False,
            )[0, :, 0]
        )
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        segmented_gqa_decode(query, key, value, slots, lengths),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_cow_segmented_gqa_reads_shared_parent_and_row_local_deltas() -> None:
    torch.manual_seed(42)
    rows, query_heads, kv_heads, head_dim = 3, 4, 2, 3
    parent_length = 5
    query = torch.randn(rows, query_heads, head_dim)
    parent_key = torch.randn(1, kv_heads, 7, head_dim)
    parent_value = torch.randn_like(parent_key)
    branch_key = torch.randn(rows, kv_heads, 4, head_dim)
    branch_value = torch.randn_like(branch_key)
    branch_lengths = torch.tensor([1, 3, 2], dtype=torch.long)

    actual = segmented_gqa_decode_cow_reference(
        query,
        parent_key,
        parent_value,
        branch_key,
        branch_value,
        branch_lengths,
        parent_length=parent_length,
    )
    expected_rows: list[torch.Tensor] = []
    for row, branch_length in enumerate(branch_lengths.tolist()):
        key = torch.cat(
            (
                parent_key[0, :, :parent_length],
                branch_key[row, :, :branch_length],
            ),
            dim=1,
        )
        value = torch.cat(
            (
                parent_value[0, :, :parent_length],
                branch_value[row, :, :branch_length],
            ),
            dim=1,
        )
        expected_rows.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[row][None, :, None],
                key.repeat_interleave(query_heads // kv_heads, dim=0)[None],
                value.repeat_interleave(query_heads // kv_heads, dim=0)[None],
                dropout_p=0.0,
                is_causal=False,
            )[0, :, 0]
        )
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        segmented_gqa_decode_cow(
            query,
            parent_key,
            parent_value,
            branch_key,
            branch_value,
            branch_lengths,
            parent_length=parent_length,
        ),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_segmented_kv_scatter_writes_only_named_row_tails() -> None:
    key_cache = torch.full((4, 2, 7, 3), -1.0)
    value_cache = torch.full_like(key_cache, -2.0)
    key = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    value = key + 100
    slots = torch.tensor([3, 1], dtype=torch.long)
    parents = torch.tensor([2, 5], dtype=torch.long)

    scatter_segmented_decode_kv(key, value, key_cache, value_cache, slots, parents)

    torch.testing.assert_close(key_cache[3, :, 2], key[0])
    torch.testing.assert_close(key_cache[1, :, 5], key[1])
    torch.testing.assert_close(value_cache[3, :, 2], value[0])
    torch.testing.assert_close(value_cache[1, :, 5], value[1])
    assert bool((key_cache[0] == -1).all())
    assert bool((value_cache[2] == -2).all())


def test_segmented_gqa_validation_rejects_aliasing_and_invalid_lengths() -> None:
    query = torch.randn(2, 4, 3)
    key = torch.randn(2, 2, 4, 3)
    value = torch.randn_like(key)
    with pytest.raises(ValueError, match="unique"):
        segmented_gqa_decode_reference(
            query,
            key,
            value,
            torch.tensor([0, 0]),
            torch.tensor([2, 3]),
        )
    with pytest.raises(ValueError, match="capacity"):
        segmented_gqa_decode_reference(
            query,
            key,
            value,
            torch.tensor([0, 1]),
            torch.tensor([2, 5]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_segmented_gqa_cuda_kernel_matches_compact_cpu_reference() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(47)
    key_cache = torch.randn(4, 2, 65, 64, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    key = torch.randn(3, 2, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(3, 8, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([3, 0, 2], device="cuda", dtype=torch.long)
    parents = torch.tensor([7, 32, 63], device="cuda", dtype=torch.long)
    scatter_segmented_decode_kv(key, value, key_cache, value_cache, slots, parents)
    lengths = parents + 1

    actual = segmented_gqa_decode(
        query,
        key_cache,
        value_cache,
        slots,
        lengths,
        max_sequence_length=64,
    )
    expected = segmented_gqa_decode_reference(
        query.cpu(),
        key_cache.cpu(),
        value_cache.cpu(),
        slots.cpu(),
        lengths.cpu(),
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_cow_segmented_gqa_cuda_kernel_matches_compact_cpu_reference() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(48)
    rows, query_heads, kv_heads, head_dim = 3, 8, 2, 64
    parent_length = 63
    query = torch.randn(rows, query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    parent_key = torch.randn(1, kv_heads, 65, head_dim, device="cuda", dtype=torch.bfloat16)
    parent_value = torch.randn_like(parent_key)
    branch_key = torch.randn(rows, kv_heads, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    branch_value = torch.randn_like(branch_key)
    branch_lengths = torch.tensor([1, 3, 2], device="cuda", dtype=torch.long)

    actual = segmented_gqa_decode_cow(
        query,
        parent_key,
        parent_value,
        branch_key,
        branch_value,
        branch_lengths,
        parent_length=parent_length,
        max_branch_length=3,
    )
    expected = segmented_gqa_decode_cow_reference(
        query.cpu(),
        parent_key.cpu(),
        parent_value.cpu(),
        branch_key.cpu(),
        branch_value.cpu(),
        branch_lengths.cpu(),
        parent_length=parent_length,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_segmented_kv_scatter_accepts_independent_fused_qkv_view_strides() -> None:
    pytest.importorskip("triton")
    rows, kv_heads, capacity, head_dim = 3, 2, 16, 8
    key = torch.randn(rows, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_storage = torch.randn(
        rows,
        kv_heads * head_dim + 13,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = value_storage[:, : kv_heads * head_dim].view(rows, kv_heads, head_dim)
    assert key.stride() != value.stride()
    key_cache = torch.zeros(
        5,
        kv_heads,
        capacity,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_cache = torch.zeros_like(key_cache)
    slots = torch.tensor([4, 1, 3], device="cuda", dtype=torch.long)
    parents = torch.tensor([2, 7, 11], device="cuda", dtype=torch.long)

    scatter_segmented_decode_kv(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        parents,
    )
    torch.cuda.synchronize()

    for row, (slot, position) in enumerate(
        zip(slots.cpu().tolist(), parents.cpu().tolist(), strict=True)
    ):
        torch.testing.assert_close(key_cache[slot, :, position], key[row])
        torch.testing.assert_close(value_cache[slot, :, position], value[row])


class _ZeroExperts:
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        del top_indices, top_weights
        self.calls.append((layer, tuple(source.shape)))
        return torch.zeros_like(source)


def _tiny_skeleton() -> Any:
    torch.manual_seed(43)
    hidden = 8
    query_heads = 4
    kv_heads = 2
    head_dim = 2
    q = torch.randn(query_heads * head_dim, hidden) / 4
    k = torch.randn(kv_heads * head_dim, hidden) / 4
    v = torch.randn(kv_heads * head_dim, hidden) / 4
    qkv = torch.cat((q, k, v), dim=0).contiguous()
    layer = LayerWeights(
        input_norm=torch.ones(hidden),
        post_norm=torch.ones(hidden),
        q_norm=torch.ones(head_dim),
        k_norm=torch.ones(head_dim),
        q_proj=qkv[: query_heads * head_dim],
        k_proj=qkv[query_heads * head_dim : (query_heads + kv_heads) * head_dim],
        v_proj=qkv[(query_heads + kv_heads) * head_dim :],
        o_proj=torch.randn(hidden, query_heads * head_dim) / 4,
        router=torch.randn(3, hidden) / 4,
        qkv_proj=qkv,
    )
    return SimpleNamespace(
        cfg={
            "num_hidden_layers": 1,
            "num_attention_heads": query_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "norm_topk_prob": True,
        },
        device="cpu",
        dtype=torch.float32,
        embedding=torch.randn(17, hidden) / 4,
        layers=[layer],
        final_norm=torch.ones(hidden),
        lm_head=torch.randn(17, hidden) / 4,
    )


def test_fused_qkv_views_partition_one_storage_without_resident_duplication() -> None:
    skeleton = _tiny_skeleton()
    layer = skeleton.layers[0]
    assert layer.qkv_proj is not None
    qkv = layer.qkv_proj
    views = (layer.q_proj, layer.k_proj, layer.v_proj)
    assert all(
        view.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr() for view in views
    )
    assert tuple(view.storage_offset() for view in views) == (
        0,
        layer.q_proj.numel(),
        layer.q_proj.numel() + layer.k_proj.numel(),
    )
    assert sum(view.numel() for view in views) == qkv.numel()


def test_qwen_ragged_decode_matches_independent_scalar_rows_and_keeps_length_provisional() -> None:
    backend = _ZeroExperts()
    runtime = Qwen3MoeDecodeRuntime(_tiny_skeleton(), backend)
    arena = runtime.new_cache(batch_size=4, capacity=8)
    # Unused torch.empty storage may contain NaNs. Give the untouched row a
    # finite sentinel so this test verifies preservation of declared contents.
    arena.layers[0].key[0].fill_(123.25)
    arena.layers[0].value[0].fill_(-456.5)
    prompts = (torch.tensor([[1, 4]]), torch.tensor([[2, 5, 7, 3]]))
    slots = (3, 1)
    scalar_caches = []
    for prompt, slot in zip(prompts, slots, strict=True):
        scalar = runtime.new_cache(batch_size=1, capacity=8)
        runtime.forward(prompt, cache=scalar)
        scalar_caches.append(scalar)
        for target, source in zip(arena.layers, scalar.layers, strict=True):
            target.key[slot, :, : scalar.length].copy_(source.key[0, :, : scalar.length])
            target.value[slot, :, : scalar.length].copy_(source.value[0, :, : scalar.length])

    input_tokens = torch.tensor([6, 8], dtype=torch.long)
    expected_tokens: list[int] = []
    expected_logits: list[torch.Tensor] = []
    expected_routes: list[list[torch.Tensor]] = []
    for token, cache in zip(input_tokens, scalar_caches, strict=True):
        result = runtime.forward(token.reshape(1, 1), cache=cache, capture_routes=True)
        assert result.logits is not None
        expected_tokens.append(int(torch.argmax(result.logits[0]).item()))
        expected_logits.append(result.logits[0])
        expected_routes.append(result.routes)

    untouched = arena.layers[0].key[0].clone()
    untouched_value = arena.layers[0].value[0].clone()
    backend.calls.clear()
    original_scalar_length = arena.length
    evidence = runtime.forward_decode_rows(
        input_tokens,
        cache=arena,
        physical_slots=slots,
        parent_lengths=tuple(int(prompt.shape[1]) for prompt in prompts),
        capture_evidence=True,
    )

    assert isinstance(evidence, RaggedDecodeEvidence)
    assert evidence.token_ids.tolist() == expected_tokens
    assert evidence.token_ids.device.type == "cpu" and evidence.token_ids.dtype == torch.long
    torch.testing.assert_close(
        evidence.logits,
        torch.stack(expected_logits),
        rtol=1e-5,
        atol=1e-6,
    )
    assert len(evidence.routes) == len(expected_routes[0])
    for layer_index, routes in enumerate(evidence.routes):
        torch.testing.assert_close(
            routes,
            torch.cat([row[layer_index] for row in expected_routes]),
            rtol=0,
            atol=0,
        )
    assert arena.length == original_scalar_length == 0
    assert backend.calls == [(0, (2, 8))], "all logical rows must share one MoE traversal"
    assert torch.equal(arena.layers[0].key[0], untouched)
    assert torch.equal(arena.layers[0].value[0], untouched_value)
    for slot, prompt, scalar in zip(slots, prompts, scalar_caches, strict=True):
        position = int(prompt.shape[1])
        torch.testing.assert_close(
            arena.layers[0].key[slot, :, position],
            scalar.layers[0].key[0, :, position],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            arena.layers[0].value[slot, :, position],
            scalar.layers[0].value[0, :, position],
            rtol=1e-5,
            atol=1e-6,
        )


def test_qwen_ragged_decode_rejects_prefill_and_duplicate_physical_rows() -> None:
    runtime = Qwen3MoeDecodeRuntime(_tiny_skeleton(), _ZeroExperts())
    arena = runtime.new_cache(batch_size=2, capacity=4)
    with pytest.raises(ValueError, match="positive"):
        runtime.forward_decode_rows(
            torch.tensor([1]),
            cache=arena,
            physical_slots=(0,),
            parent_lengths=(0,),
        )
    with pytest.raises(ValueError, match="unique"):
        runtime.forward_decode_rows(
            torch.tensor([1, 2]),
            cache=arena,
            physical_slots=(0, 0),
            parent_lengths=(1, 1),
        )


class _NativeSeam:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def forward_decode_rows(self, input_token_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append((input_token_ids.clone(), kwargs))
        return input_token_ids + 10


def test_native_continuous_adapter_forwards_exact_slot_and_length_contract() -> None:
    seam = _NativeSeam()
    adapter = Qwen3MoeNativeRaggedDecodeExecutor(SimpleNamespace(runtime=seam))
    batch = Qwen3MoeDecodeBatch(
        dispatch_id="dispatch-test",
        request_ids=("a", "b"),
        physical_slots=(3, 1),
        storage_generations=(2, 4),
        parent_lengths=(17, 29),
        input_token_ids=(5, 6),
    )
    cache = object()
    result = adapter.execute(batch, cache=cache)

    assert result.dispatch_id == batch.dispatch_id
    assert result.row_bindings == batch.row_bindings
    assert result.next_token_ids == (15, 16)
    inputs, kwargs = seam.calls[0]
    assert inputs.tolist() == [5, 6]
    assert kwargs == {
        "cache": cache,
        "physical_slots": (3, 1),
        "parent_lengths": (17, 29),
    }
