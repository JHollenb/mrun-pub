from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.engine import open_engine
from mrun.engine.olmoe_cuda import (
    FP8_MAX,
    OLMoECudaEngine,
    align_up,
    apply_residual_patch_ops,
    grouped_fp8_tile_config,
    grouped_route_plan,
    grouped_route_pruned_queue,
    grouped_route_queue,
    grouped_route_tensors,
    grouped_route_tensors_with_inverse,
    parse_batch_sizes,
    prune_router_topk,
    quantize_fp8_weight,
)


def test_fp8_weight_quantization_is_finite_and_bounded() -> None:
    torch.manual_seed(17)
    source = torch.randn((32, 64), dtype=torch.float32) * 0.1
    quantized, scales = quantize_fp8_weight(source)
    restored = quantized.float() * scales[:, None]

    assert quantized.dtype == torch.float8_e4m3fn
    assert bool((scales > 0).all())
    assert float(quantized.float().abs().max()) <= FP8_MAX
    assert torch.isfinite(restored).all()
    relative_l2 = (restored - source).norm() / source.norm()
    assert float(relative_l2) < 0.04


def test_zero_weight_uses_nonzero_scale() -> None:
    source = torch.zeros((4, 8), dtype=torch.float32)
    quantized, scales = quantize_fp8_weight(source)

    assert bool((scales > 0).all())
    torch.testing.assert_close(quantized.float() * scales[:, None], source)


def test_grouped_route_plan_preserves_assignments() -> None:
    indices = torch.tensor([[3, 1, 4], [1, 4, 2], [3, 2, 1]], dtype=torch.long)
    weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.6, 0.3, 0.1], [0.7, 0.2, 0.1]],
        dtype=torch.float32,
    )
    plan, stats = grouped_route_plan(indices, weights)
    actual = sorted(
        (expert, int(token), float(weight))
        for expert, tokens, coefficients in plan
        for token, weight in zip(tokens.tolist(), coefficients.tolist(), strict=True)
    )
    expected = sorted(
        (int(expert), token, float(weights[token, slot]))
        for token in range(indices.shape[0])
        for slot, expert in enumerate(indices[token])
    )

    assert actual == expected
    assert stats["assignments"] == 9
    assert stats["active_experts"] == 4
    assert stats["route_reuse_x"] == 2.25


def test_grouped_route_tensors_keep_fixed_expert_width() -> None:
    indices = torch.tensor([[3, 1], [1, 2], [3, 2]], dtype=torch.long)
    weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],
        dtype=torch.float32,
    )
    token_ids, coefficients, starts, counts, _ = grouped_route_tensors(
        indices,
        weights,
        num_experts=5,
    )

    assert counts.tolist() == [0, 2, 2, 2, 0]
    assert starts.tolist() == [0, 0, 2, 4, 6]
    assert token_ids.tolist() == [0, 1, 1, 2, 0, 2]
    torch.testing.assert_close(
        coefficients,
        torch.tensor([0.3, 0.6, 0.4, 0.2, 0.7, 0.8]),
    )


def test_grouped_route_inverse_maps_original_rank_to_expert_major_row() -> None:
    indices = torch.tensor([[3, 1], [1, 2], [3, 2]], dtype=torch.long)
    weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],
        dtype=torch.float32,
    )
    token_ids, coefficients, starts, counts, inverse, _ = grouped_route_tensors_with_inverse(
        indices,
        weights,
        num_experts=5,
    )

    assert sorted(inverse.tolist()) == list(range(indices.numel()))
    for original, destination in enumerate(inverse.tolist()):
        token, rank = divmod(original, indices.shape[1])
        expert = int(indices[token, rank])
        assert int(token_ids[destination]) == token
        assert starts[expert] <= destination < starts[expert] + counts[expert]
        assert float(coefficients[destination]) == pytest.approx(float(weights[token, rank]))


def test_grouped_route_queue_matches_grouped_route_tensors_on_cpu() -> None:
    indices = torch.tensor([[3, 1], [1, 2], [3, 2]], dtype=torch.long)
    weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],
        dtype=torch.float32,
    )
    token_ids, coefficients, starts, counts, _ = grouped_route_tensors(
        indices,
        weights,
        num_experts=5,
    )
    plan, _ = grouped_route_queue(indices, weights, num_experts=5)

    plan.assert_valid()
    torch.testing.assert_close(plan.token_ids, token_ids)
    torch.testing.assert_close(plan.coefficients, coefficients)
    torch.testing.assert_close(plan.starts.to(torch.int32), starts)
    torch.testing.assert_close(plan.counts.to(torch.int32), counts)


def test_prune_router_topk_keeps_primary_route_and_drops_weak_tail() -> None:
    indices = torch.tensor(
        [[5, 4, 3, 2], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [[0.90, 0.06, 0.03, 0.01], [0.04, 0.03, 0.02, 0.01]],
        dtype=torch.float32,
    )

    token_ids, expert_ids, route_slots, coefficients, stats = prune_router_topk(
        indices,
        weights,
        min_weight=0.05,
        max_top_k=3,
    )

    assert token_ids.tolist() == [0, 0, 1]
    assert expert_ids.tolist() == [5, 4, 1]
    assert route_slots.tolist() == [0, 1, 0]
    torch.testing.assert_close(coefficients, torch.tensor([0.90, 0.06, 0.04]))
    assert stats["original_assignments"] == 8
    assert stats["kept_assignments"] == 3
    assert stats["dropped_assignments"] == 5
    assert stats["kept_fraction"] == 0.375


def test_prune_router_topk_can_renormalize_surviving_mass() -> None:
    indices = torch.tensor([[5, 4, 3, 2], [1, 2, 3, 4]], dtype=torch.long)
    weights = torch.tensor(
        [[0.90, 0.06, 0.03, 0.01], [0.04, 0.03, 0.02, 0.01]],
        dtype=torch.float32,
    )

    _tokens, _experts, _slots, coefficients, _stats = prune_router_topk(
        indices,
        weights,
        min_weight=0.05,
        max_top_k=3,
        renormalize=True,
    )

    torch.testing.assert_close(
        coefficients,
        torch.tensor([0.90 / 0.96, 0.06 / 0.96, 1.0], dtype=torch.float32),
    )


def test_grouped_route_pruned_queue_reports_actual_assignments() -> None:
    indices = torch.tensor(
        [[5, 4, 3, 2], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [[0.90, 0.06, 0.03, 0.01], [0.04, 0.03, 0.02, 0.01]],
        dtype=torch.float32,
    )

    plan, stats = grouped_route_pruned_queue(
        indices,
        weights,
        num_experts=6,
        min_weight=0.05,
        max_top_k=3,
    )

    plan.assert_valid()
    assert plan.assignments == 3
    assert plan.top_k == 4
    assert plan.counts.tolist() == [0, 1, 0, 0, 1, 1]
    assert stats["original_assignments"] == 8
    assert stats["kept_assignments"] == 3
    assert stats["metadata_s"] >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_grouped_route_queue_matches_grouped_route_tensors_on_cuda() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260719)
    rows = 256
    num_experts = 16
    top_k = 4
    scores = torch.rand((rows, num_experts), device=device, generator=generator)
    weights, indices = torch.topk(scores, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    token_ids, coefficients, starts, counts, _ = grouped_route_tensors(
        indices,
        weights,
        num_experts=num_experts,
    )
    plan, _ = grouped_route_queue(indices, weights, num_experts=num_experts)

    torch.testing.assert_close(plan.starts, starts)
    torch.testing.assert_close(plan.counts, counts)
    reference_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.long),
        counts.to(torch.long),
    )
    reference = sorted(
        zip(
            token_ids.cpu().tolist(),
            reference_expert_ids.cpu().tolist(),
            coefficients.cpu().tolist(),
            strict=True,
        )
    )
    fused = sorted(
        zip(
            plan.token_ids.cpu().tolist(),
            plan.expert_ids.cpu().tolist(),
            plan.coefficients.cpu().tolist(),
            strict=True,
        )
    )
    assert [packet[:2] for packet in fused] == [packet[:2] for packet in reference]
    assert [packet[2] for packet in fused] == pytest.approx([packet[2] for packet in reference])


def test_page_alignment_and_batch_parser() -> None:
    assert align_up(0) == 0
    assert align_up(1) == 4096
    assert align_up(4096) == 4096
    assert align_up(4097) == 8192
    assert parse_batch_sizes("512,1,128,128") == [1, 128, 512]


def test_grouped_fp8_tile_config_reads_validated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert grouped_fp8_tile_config() == (32, 64, 32, 4, 3)

    monkeypatch.setenv("MRUN_OLMOE_GROUPED_BLOCK_M", "64")
    monkeypatch.setenv("MRUN_OLMOE_GROUPED_BLOCK_N", "128")
    monkeypatch.setenv("MRUN_OLMOE_GROUPED_BLOCK_K", "64")
    monkeypatch.setenv("MRUN_OLMOE_GROUPED_WARPS", "8")
    monkeypatch.setenv("MRUN_OLMOE_GROUPED_STAGES", "4")
    assert grouped_fp8_tile_config() == (64, 128, 64, 8, 4)

    monkeypatch.setenv("MRUN_OLMOE_GROUPED_BLOCK_M", "0")
    with pytest.raises(ValueError, match="MRUN_OLMOE_GROUPED_BLOCK_M"):
        grouped_fp8_tile_config()


def test_residual_projection_removal_is_rank_one() -> None:
    source = torch.randn(3, 5)
    direction = torch.randn(5)
    patched = apply_residual_patch_ops(source, [("proj_remove", direction, None)])
    unit = direction / direction.norm()
    assert float((patched @ unit).abs().max()) < 1e-5


def test_residual_projection_removal_supports_runtime_batch_shape() -> None:
    source = torch.randn(2, 3, 5)
    direction = torch.randn(5)
    patched = apply_residual_patch_ops(source, [("proj_remove", direction, None)])
    unit = direction / direction.norm()
    assert float((patched @ unit).abs().max()) < 1e-5


def test_public_capabilities_fail_closed_for_unimplemented_taps() -> None:
    engine = object.__new__(OLMoECudaEngine)
    capabilities = engine.capabilities()

    assert capabilities.logits
    assert capabilities.logits_batch
    assert capabilities.approximate_quantized
    assert capabilities.residual_tap
    assert capabilities.generation
    assert capabilities.generation_batch
    assert capabilities.persistent_kv
    assert capabilities.compact_fused_weights
    assert capabilities.grouped_moe
    assert not capabilities.transactional_kv
    assert not capabilities.mlp_acts
    assert not capabilities.raw_model


def test_logits_batch_fuses_only_equal_length_rows() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def forward(self, input_ids: torch.Tensor, *, all_logits: bool) -> SimpleNamespace:
            assert all_logits
            self.shapes.append(tuple(input_ids.shape))
            logits = input_ids[:, :, None].expand(-1, -1, 3).float()
            return SimpleNamespace(logits=logits)

    engine = object.__new__(OLMoECudaEngine)
    engine.device = "cpu"
    engine.runtime = FakeRuntime()
    outputs = engine.logits_batch(
        [
            np.asarray([1, 2]),
            np.asarray([7]),
            np.asarray([3, 4]),
        ]
    )

    assert engine.runtime.shapes == [(2, 2), (1, 1)]
    assert [tuple(output.shape) for output in outputs] == [(2, 3), (1, 3), (2, 3)]
    assert outputs[0][0, 0].item() == 1
    assert outputs[1][0, 0].item() == 7
    assert outputs[2][0, 0].item() == 3


def test_native_residual_batches_bucket_lengths_and_forward_shared_patch() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            all_logits: bool,
            capture_hidden_states: bool = False,
            resid_patch_ops_by_layer=None,
        ) -> SimpleNamespace:
            self.calls.append(
                {
                    "shape": tuple(input_ids.shape),
                    "all_logits": all_logits,
                    "capture": capture_hidden_states,
                    "resid": resid_patch_ops_by_layer,
                }
            )
            logits = input_ids[:, :, None].expand(-1, -1, 3).float()
            if resid_patch_ops_by_layer:
                logits = logits + 100.0
            hidden_states = []
            if capture_hidden_states:
                hidden = input_ids[:, :, None].expand(-1, -1, 2).float()
                hidden_states = [hidden, hidden + 10.0, hidden + 20.0]
            return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    engine = object.__new__(OLMoECudaEngine)
    engine.device = "cpu"
    engine.runtime = FakeRuntime()
    rows = [np.asarray([1, 2]), np.asarray([7]), np.asarray([3, 4])]

    captured = engine.hidden_states_batch(rows)
    assert [tuple(state[0].shape) for state in captured] == [(2, 2), (1, 2), (2, 2)]
    assert [call["shape"] for call in engine.runtime.calls] == [(2, 2), (1, 1)]
    assert all(call["all_logits"] is False for call in engine.runtime.calls)

    engine.runtime.calls.clear()
    residual_ops = {0: [("proj_remove", np.asarray([1.0, 0.0]), None)]}
    patched = engine.forward_patched_batch(
        rows,
        resid_patch_ops_by_layer=residual_ops,
    )
    assert [tuple(output[0].shape) for output in patched] == [(2, 3), (1, 3), (2, 3)]
    assert [call["shape"] for call in engine.runtime.calls] == [(2, 2), (1, 1)]
    assert all(call["resid"] is residual_ops for call in engine.runtime.calls)
    assert patched[1][0][0, 0].item() == 107.0


def test_generate_batch_buckets_lengths_and_restores_order() -> None:
    class FakeTokenizer:
        eos_token_id = None

        def __call__(self, prompt: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            assert not add_special_tokens
            return {"input_ids": {"two": [2, 20], "one": [1]}[prompt]}

        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return ",".join(str(token) for token in tokens)

    calls: list[tuple[int, ...]] = []
    engine = object.__new__(OLMoECudaEngine)
    engine.tokenizer = FakeTokenizer()

    def fake_generate_ids(input_ids, *, max_new_tokens: int):
        rows = torch.as_tensor(input_ids)
        calls.append(tuple(rows.shape))
        return rows[:, :1].expand(-1, max_new_tokens)

    engine.generate_ids = fake_generate_ids
    outputs = engine.generate_batch(["two", "one", "two"], max_new_tokens=3)

    assert calls == [(2, 2), (1, 1)]
    assert outputs == [[2, 2, 2], [1, 1, 1], [2, 2, 2]]


def test_open_engine_dispatches_olmoe_backend(monkeypatch) -> None:
    import mrun.engine.olmoe_cuda as module

    sentinel = object()

    def fake_engine(model_name: str, **kwargs):
        assert model_name == "olmoe-1b-7b"
        assert kwargs == {"store_dir": "/tmp/store"}
        return sentinel

    monkeypatch.setattr(module, "OLMoECudaEngine", fake_engine)
    assert (
        open_engine(
            "olmoe-1b-7b",
            backend="olmoe-cuda",
            store_dir="/tmp/store",
        )
        is sentinel
    )


def test_open_engine_dispatches_moe_stream_backend(monkeypatch) -> None:
    import mrun.engine.moe_stream as module

    sentinel = object()

    def fake_engine(model_name: str, **kwargs):
        assert model_name == "qwen3-30b-a3b"
        assert kwargs == {"abort_rss_gb": 40.0}
        return sentinel

    monkeypatch.setattr(module, "MoEStreamEngine", fake_engine)
    assert (
        open_engine(
            "qwen3-30b-a3b",
            backend="moe-stream",
            abort_rss_gb=40.0,
        )
        is sentinel
    )
