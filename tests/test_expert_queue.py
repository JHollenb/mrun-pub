from __future__ import annotations

import pytest
import torch

from mrun.engine.kernels.expert_queue import (
    build_expert_queue,
    build_expert_queue_device,
    build_expert_queue_flat,
    scatter_committed,
    scatter_committed_device,
    triton,
)


def test_queue_ranges_are_stable_and_conserve_assignments() -> None:
    plan = build_expert_queue(
        torch.tensor([[2, 0], [2, 1], [3, 2]], dtype=torch.int64),
        torch.tensor([[0.6, 0.3], [0.5, 0.4], [0.2, 0.5]], dtype=torch.float32),
        num_experts=4,
        epochs=torch.tensor([7, 8, 9]),
        deadlines=torch.tensor([30, 10, 20]),
    )

    assert plan.assignments == 6
    assert plan.counts.tolist() == [1, 1, 3, 1]
    assert plan.starts.tolist() == [0, 1, 2, 5]
    assert plan.token_ids.tolist() == [0, 1, 0, 1, 2, 2]
    assert plan.route_slots.tolist() == [1, 1, 0, 0, 1, 0]
    assert plan.epochs.tolist() == [7, 8, 7, 8, 9, 9]


def test_scatter_rejects_packets_from_a_reused_slot_epoch() -> None:
    plan = build_expert_queue(
        torch.tensor([[2, 0], [2, 1], [3, 2]], dtype=torch.int64),
        torch.tensor([[0.6, 0.3], [0.5, 0.4], [0.2, 0.5]], dtype=torch.float32),
        num_experts=4,
        epochs=torch.tensor([7, 8, 9]),
        deadlines=torch.tensor([30, 10, 20]),
    )
    outputs = torch.ones((plan.assignments, 2), dtype=torch.float32)

    reduced, valid = scatter_committed(
        plan,
        outputs,
        active_epochs=torch.tensor([7, 99, 9]),
    )

    assert int(valid.sum().item()) == 4
    torch.testing.assert_close(reduced[:, 0], torch.tensor([0.9, 0.0, 0.7]))
    torch.testing.assert_close(reduced[:, 1], torch.tensor([0.9, 0.0, 0.7]))


def test_sparse_queue_supports_variable_routes_per_row() -> None:
    plan = build_expert_queue_flat(
        torch.tensor([0, 1, 2, 2], dtype=torch.int64),
        torch.tensor([2, 1, 3, 0], dtype=torch.int64),
        torch.tensor([0, 0, 0, 1], dtype=torch.int64),
        torch.tensor([0.7, 0.9, 0.8, 0.2], dtype=torch.float32),
        num_rows=3,
        num_experts=4,
        top_k=2,
        epochs=torch.tensor([5, 6, 7], dtype=torch.int64),
        deadlines=torch.tensor([50, 60, 70], dtype=torch.int64),
    )

    assert plan.assignments == 4
    assert plan.counts.tolist() == [1, 1, 1, 1]
    assert plan.starts.tolist() == [0, 1, 2, 3]
    assert plan.token_ids.tolist() == [2, 1, 0, 2]
    assert plan.route_slots.tolist() == [1, 0, 0, 0]
    plan.assert_valid()

    reduced, valid = scatter_committed(
        plan,
        torch.ones((plan.assignments, 2), dtype=torch.float32),
        active_epochs=torch.tensor([5, 6, 7], dtype=torch.int64),
    )

    assert bool(valid.all().item())
    torch.testing.assert_close(reduced[:, 0], torch.tensor([0.7, 0.9, 1.0]))
    torch.testing.assert_close(reduced[:, 1], torch.tensor([0.7, 0.9, 1.0]))


def test_sparse_queue_rejects_duplicate_slots_and_experts() -> None:
    common = {
        "num_rows": 1,
        "num_experts": 4,
        "top_k": 2,
        "epochs": torch.tensor([1], dtype=torch.int64),
        "deadlines": torch.tensor([2], dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="same route slot twice"):
        build_expert_queue_flat(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([0.6, 0.4], dtype=torch.float32),
            **common,
        )
    with pytest.raises(ValueError, match="same expert twice"):
        build_expert_queue_flat(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0.6, 0.4], dtype=torch.float32),
            **common,
        )


def test_queue_rejects_duplicate_and_out_of_range_routes() -> None:
    common = {
        "top_weights": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        "num_experts": 4,
        "epochs": torch.tensor([1]),
        "deadlines": torch.tensor([10]),
    }
    with pytest.raises(ValueError, match="same expert twice"):
        build_expert_queue(torch.tensor([[2, 2]]), **common)
    with pytest.raises(ValueError, match="out-of-range"):
        build_expert_queue(torch.tensor([[2, 4]]), **common)


def test_random_uniform_routes_keep_empty_expert_ranges_contiguous() -> None:
    generator = torch.Generator().manual_seed(20260718)
    indices = torch.stack([torch.randperm(16, generator=generator)[:4] for _ in range(64)])
    weights = torch.rand((64, 4), generator=generator)
    plan = build_expert_queue(
        indices,
        weights,
        num_experts=20,
        epochs=torch.arange(64),
        deadlines=torch.arange(64).flip(0),
    )

    plan.assert_valid()
    assert plan.assignments == 256
    assert plan.counts[16:].tolist() == [0, 0, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_queue_contract_runs_on_cuda_without_changing_layout() -> None:
    device = torch.device("cuda")
    plan = build_expert_queue(
        torch.tensor([[2, 0], [2, 1], [3, 2]], device=device, dtype=torch.int64),
        torch.tensor([[0.6, 0.3], [0.5, 0.4], [0.2, 0.5]], device=device),
        num_experts=4,
        epochs=torch.tensor([7, 8, 9], device=device),
        deadlines=torch.tensor([30, 10, 20], device=device),
    )

    outputs = torch.ones((plan.assignments, 2), device=device)
    reduced, valid = scatter_committed(
        plan,
        outputs,
        active_epochs=torch.tensor([7, 99, 9], device=device),
    )

    assert plan.token_ids.is_cuda
    assert valid.is_cuda
    torch.testing.assert_close(reduced.cpu()[:, 0], torch.tensor([0.9, 0.0, 0.7]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA/Triton not available",
)
def test_fused_device_queue_matches_torch_contract_on_cuda() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260719)
    rows = 128
    num_experts = 16
    top_k = 4
    scores = torch.rand((rows, num_experts), device=device, generator=generator)
    top_weights, top_indices = torch.topk(scores, top_k, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    epochs = torch.arange(rows, device=device, dtype=torch.int64) + 10
    deadlines = torch.arange(rows, device=device, dtype=torch.int64).flip(0)

    reference = build_expert_queue(
        top_indices,
        top_weights,
        num_experts=num_experts,
        epochs=epochs,
        deadlines=deadlines,
    )
    fused = build_expert_queue_device(
        top_indices,
        top_weights,
        num_experts=num_experts,
        epochs=epochs,
        deadlines=deadlines,
    )

    fused.assert_valid()
    torch.testing.assert_close(fused.counts.cpu().to(torch.int64), reference.counts.cpu())
    torch.testing.assert_close(fused.starts.cpu().to(torch.int64), reference.starts.cpu())
    reference_packets = sorted(
        zip(
            reference.token_ids.cpu().tolist(),
            reference.expert_ids.cpu().tolist(),
            reference.route_slots.cpu().tolist(),
            reference.epochs.cpu().tolist(),
            reference.deadlines.cpu().tolist(),
            reference.coefficients.cpu().tolist(),
            strict=True,
        )
    )
    fused_packets = sorted(
        zip(
            fused.token_ids.cpu().tolist(),
            fused.expert_ids.cpu().tolist(),
            fused.route_slots.cpu().tolist(),
            fused.epochs.cpu().tolist(),
            fused.deadlines.cpu().tolist(),
            fused.coefficients.cpu().tolist(),
            strict=True,
        )
    )
    assert [packet[:5] for packet in fused_packets] == [
        packet[:5] for packet in reference_packets
    ]
    assert [packet[5] for packet in fused_packets] == pytest.approx(
        [packet[5] for packet in reference_packets]
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA/Triton not available",
)
def test_fused_device_scatter_matches_torch_contract_on_cuda() -> None:
    device = torch.device("cuda")
    plan = build_expert_queue_device(
        torch.tensor([[2, 0], [2, 1], [3, 2]], device=device, dtype=torch.int64),
        torch.tensor([[0.6, 0.3], [0.5, 0.4], [0.2, 0.5]], device=device),
        num_experts=4,
        epochs=torch.tensor([7, 8, 9], device=device),
        deadlines=torch.tensor([30, 10, 20], device=device),
    )
    outputs = torch.ones((plan.assignments, 5), device=device)

    reduced, valid = scatter_committed_device(
        plan,
        outputs,
        active_epochs=torch.tensor([7, 99, 9], device=device),
    )

    assert valid.is_cuda
    assert int(valid.sum().item()) == 4
    torch.testing.assert_close(
        reduced.cpu(),
        torch.tensor(
            [
                [0.9, 0.9, 0.9, 0.9, 0.9],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.7, 0.7, 0.7, 0.7],
            ]
        ),
    )
