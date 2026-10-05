from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.engine.qwen3_moe_cuda import LayerWeights, Qwen3MoeDecodeRuntime, greedy_tokens
from mrun.engine.qwen3_moe_statecut import Qwen3MoeKVStateCut


class _ZeroExperts:
    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        del layer, top_indices, top_weights
        return torch.zeros_like(source)


def _runtime() -> Qwen3MoeDecodeRuntime:
    torch.manual_seed(101)
    hidden, query_heads, kv_heads, head_dim = 8, 4, 2, 2
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
    skeleton = SimpleNamespace(
        cfg={
            "num_hidden_layers": 1,
            "num_attention_heads": query_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden,
            "vocab_size": 19,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "norm_topk_prob": True,
        },
        device="cpu",
        dtype=torch.float32,
        embedding=torch.randn(19, hidden) / 4,
        layers=[layer],
        final_norm=torch.ones(hidden),
        lm_head=torch.randn(19, hidden) / 4,
    )
    return Qwen3MoeDecodeRuntime(skeleton, _ZeroExperts())


class _Engine:
    device = "cpu"

    def __init__(self, runtime: Qwen3MoeDecodeRuntime) -> None:
        self.runtime = runtime

    def _require_runtime(self) -> Qwen3MoeDecodeRuntime:
        return self.runtime


def _scalar_hidden(
    runtime: Qwen3MoeDecodeRuntime,
    prefix: list[int],
    suffix: int,
) -> tuple[torch.Tensor, object]:
    cache = runtime.new_cache(batch_size=1, capacity=len(prefix) + 1)
    runtime.forward(torch.tensor([prefix]), cache=cache, skip_lm_head=True)
    result = runtime.forward(
        torch.tensor([[suffix]]),
        cache=cache,
        return_final_hidden=True,
        skip_lm_head=True,
    )
    assert result.final_hidden is not None
    return result.final_hidden[0], cache


def test_qwen3_moe_statecut_stages_zero_copy_parent_and_matches_scalar_rows() -> None:
    runtime = _runtime()
    engine = _Engine(runtime)
    prefix = [1, 4, 7]
    cut, _prefill = Qwen3MoeKVStateCut.prefill(
        engine,
        prefix,
        retention_budget_bytes=1_000_000,
    )
    descriptor = cut.descriptor
    assert descriptor.parent_tokens == len(prefix)
    assert descriptor.parent_copy_bytes_per_stage == 0
    parent_signature = cut.parent_fingerprint
    for branch in ("a", "b", "c"):
        cut.fork(branch)

    continuation = cut.continue_one(
        {"a": np.asarray([2]), "b": np.asarray([3]), "c": np.asarray([5])}
    )
    expected = torch.stack(
        [_scalar_hidden(runtime, prefix, suffix)[0] for suffix in (2, 3, 5)]
    )
    torch.testing.assert_close(continuation.outputs, expected, rtol=1e-5, atol=1e-6)
    assert continuation.parent_copy_bytes == 0
    assert continuation.physical_forwards == 1
    assert continuation.retained_diff_bytes == 3 * 2 * 2 * 2 * 4
    assert cut.parent_fingerprint == parent_signature

    receipt = cut.commit("b")
    assert receipt.selected_branch_id == "b"
    assert receipt.parent_copy_bytes_during_stage == 0
    assert receipt.materialized_commit_bytes == 2 * 1 * 4 * 2 * 2 * 4
    assert receipt.parent_epoch_after == receipt.parent_epoch_before + 1
    with pytest.raises(RuntimeError, match="terminal"):
        cut.fork("late")


def test_qwen3_moe_statecut_abandon_and_parent_mutation_guard() -> None:
    runtime = _runtime()
    engine = _Engine(runtime)
    cut, _ = Qwen3MoeKVStateCut.prefill(
        engine,
        [1, 2],
        retention_budget_bytes=1_000_000,
    )
    cut.fork("branch")
    cut._parent.layers[0].key.add_(1)  # noqa: SLF001 - intentional hostile mutation
    with pytest.raises(RuntimeError, match="immutable parent changed"):
        cut.continue_one({"branch": [3]})

    clean, _ = Qwen3MoeKVStateCut.prefill(
        engine,
        [1, 2],
        retention_budget_bytes=1_000_000,
    )
    clean.fork("branch")
    receipt = clean.abandon()
    assert receipt.decision == "abandon"
    assert receipt.materialized_commit_bytes == 0


def test_qwen3_moe_statecut_replay_shares_parent_but_isolates_transactions() -> None:
    runtime = _runtime()
    prototype, _ = Qwen3MoeKVStateCut.prefill(
        _Engine(runtime),
        [1, 4, 7],
        retention_budget_bytes=1_000_000,
    )
    left = prototype.replay()
    right = prototype.replay()
    assert left._parent is prototype._parent  # noqa: SLF001 - zero-copy contract
    assert right._parent is prototype._parent  # noqa: SLF001 - zero-copy contract
    assert left.parent_fingerprint == right.parent_fingerprint == prototype.parent_fingerprint

    left.fork("candidate")
    right.fork("candidate")
    left_output = left.continue_one({"candidate": [2]}).outputs
    right_output = right.continue_one({"candidate": [5]}).outputs
    assert not torch.equal(left_output, right_output)
    assert left.abandon().parent_copy_bytes_during_stage == 0
    assert right.abandon().parent_copy_bytes_during_stage == 0

    # The source prototype remains reusable after child transactions terminate.
    third = prototype.replay()
    third.fork("candidate")
    torch.testing.assert_close(third.continue_one({"candidate": [2]}).outputs, left_output)


def test_qwen3_moe_statecut_multitoken_generation_matches_native_rows() -> None:
    runtime = _runtime()
    engine = _Engine(runtime)
    prefix = [1, 4, 7]
    forced = (2, 3, 5)
    prototype, _ = Qwen3MoeKVStateCut.prefill(
        engine,
        prefix,
        retention_budget_bytes=1_000_000,
    )
    transaction = prototype.replay()
    branch_ids = tuple(f"branch-{index}" for index in range(len(forced)))
    for branch_id in branch_ids:
        transaction.fork(branch_id)
    continuation = transaction.generate(
        {
            branch_id: [token]
            for branch_id, token in zip(branch_ids, forced, strict=True)
        },
        max_new_tokens=4,
    )

    expected = torch.cat(
        [
            greedy_tokens(
                runtime,
                torch.tensor([prefix + [token]], dtype=torch.long),
                steps=4,
            )
            for token in forced
        ],
        dim=0,
    )
    assert torch.equal(continuation.outputs, expected)
    assert continuation.output_contract == "generated_token_ids"
    assert continuation.physical_forwards == 4
    assert continuation.parent_copy_bytes == 0
    assert continuation.retained_diff_bytes == len(forced) * 2 * 2 * 2 * 4 * 4
