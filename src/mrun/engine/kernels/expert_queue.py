"""Expert-major packet queues for MoE runtimes.

The functions here define the reusable Expert Exchange contract: router top-k output is compiled
into contiguous per-expert packet ranges, and completed packet outputs are reduced back into token
rows only when their slot epochs still match. The implementation is intentionally pure Torch so it
can run on CPU in unit tests and on CUDA without host route lists; custom kernels can replace these
internals while preserving the public packet layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs still import the module.
    triton = None
    tl = None


@dataclass(frozen=True)
class ExpertQueuePlan:
    num_rows: int
    num_experts: int
    top_k: int
    token_ids: torch.Tensor
    expert_ids: torch.Tensor
    route_slots: torch.Tensor
    coefficients: torch.Tensor
    epochs: torch.Tensor
    deadlines: torch.Tensor
    starts: torch.Tensor
    counts: torch.Tensor

    @property
    def assignments(self) -> int:
        return int(self.token_ids.numel())

    def assert_valid(self) -> None:
        if self.num_rows <= 0:
            raise AssertionError("num_rows must be positive")
        if self.num_experts <= 0:
            raise AssertionError("num_experts must be positive")
        if self.top_k <= 0:
            raise AssertionError("top_k must be positive")
        max_assignments = self.num_rows * self.top_k
        if self.assignments > max_assignments:
            raise AssertionError(
                f"assignment conservation failed: {self.assignments} > {max_assignments}"
            )
        packet_shape = (self.assignments,)
        if tuple(self.expert_ids.shape) != packet_shape:
            raise AssertionError("expert_ids have the wrong packet shape")
        if tuple(self.route_slots.shape) != packet_shape:
            raise AssertionError("route_slots have the wrong packet shape")
        if tuple(self.coefficients.shape) != packet_shape:
            raise AssertionError("coefficients have the wrong packet shape")
        if tuple(self.epochs.shape) != packet_shape:
            raise AssertionError("epochs have the wrong packet shape")
        if tuple(self.deadlines.shape) != packet_shape:
            raise AssertionError("deadlines have the wrong packet shape")
        if int(self.counts.sum().item()) != self.assignments:
            raise AssertionError("expert counts do not conserve assignments")
        if tuple(self.starts.shape) != (self.num_experts,):
            raise AssertionError("starts have the wrong expert shape")
        if tuple(self.counts.shape) != (self.num_experts,):
            raise AssertionError("counts have the wrong expert shape")
        if self.num_experts and int(self.starts[0].item()) != 0:
            raise AssertionError("the first expert queue must begin at zero")
        if self.num_experts > 1:
            expected_starts = self.starts[:-1] + self.counts[:-1]
            if not torch.equal(self.starts[1:], expected_starts):
                raise AssertionError("expert queue ranges are not contiguous")
        if self.assignments:
            if not bool(((self.token_ids >= 0) & (self.token_ids < self.num_rows)).all().item()):
                raise AssertionError("token_ids contain an out-of-range row")
            if not bool(((self.expert_ids >= 0) & (self.expert_ids < self.num_experts)).all().item()):
                raise AssertionError("expert_ids contain an out-of-range expert")
            if not bool(((self.route_slots >= 0) & (self.route_slots < self.top_k)).all().item()):
                raise AssertionError("route_slots contain an out-of-range slot")
            if not bool(torch.isfinite(self.coefficients).all().item()):
                raise AssertionError("coefficients must be finite")
            if not bool(torch.all(self.expert_ids[:-1] <= self.expert_ids[1:]).item()):
                raise AssertionError("packets are not grouped by expert")


def _require_integral(name: str, value: torch.Tensor) -> None:
    if value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(f"{name} must contain integer values")


if triton is not None:

    @triton.jit
    def _count_kernel(
        experts_ptr,
        counts_ptr,
        assignments: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < assignments
        experts = tl.load(experts_ptr + offsets, mask=mask, other=0)
        tl.atomic_add(counts_ptr + experts, 1, mask=mask, sem="relaxed")

    @triton.jit
    def _prefix_kernel(
        counts_ptr,
        starts_ptr,
        num_experts: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.arange(0, block)
        mask = offsets < num_experts
        counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
        inclusive = tl.cumsum(counts, axis=0)
        tl.store(starts_ptr + offsets, inclusive - counts, mask=mask)

    @triton.jit
    def _queue_write_kernel(
        experts_ptr,
        weights_ptr,
        epochs_ptr,
        deadlines_ptr,
        starts_ptr,
        cursors_ptr,
        token_ids_ptr,
        expert_ids_ptr,
        route_slots_ptr,
        coefficients_ptr,
        packet_epochs_ptr,
        packet_deadlines_ptr,
        assignments: tl.constexpr,
        top_k: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < assignments
        experts = tl.load(experts_ptr + offsets, mask=mask, other=0)
        local_offsets = tl.atomic_add(cursors_ptr + experts, 1, mask=mask, sem="relaxed")
        destinations = tl.load(starts_ptr + experts, mask=mask, other=0) + local_offsets
        tokens = offsets // top_k
        slots = offsets - tokens * top_k
        coefficients = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
        epochs = tl.load(epochs_ptr + tokens, mask=mask, other=0)
        deadlines = tl.load(deadlines_ptr + tokens, mask=mask, other=0)
        tl.store(token_ids_ptr + destinations, tokens, mask=mask)
        tl.store(expert_ids_ptr + destinations, experts, mask=mask)
        tl.store(route_slots_ptr + destinations, slots, mask=mask)
        tl.store(coefficients_ptr + destinations, coefficients, mask=mask)
        tl.store(packet_epochs_ptr + destinations, epochs, mask=mask)
        tl.store(packet_deadlines_ptr + destinations, deadlines, mask=mask)

    @triton.jit
    def _scatter_reduce_kernel(
        token_ids_ptr,
        coefficients_ptr,
        packet_epochs_ptr,
        active_epochs_ptr,
        expert_outputs_ptr,
        reduced_ptr,
        valid_ptr,
        assignments: tl.constexpr,
        hidden: tl.constexpr,
        block_assignments: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        assignment_offsets = (
            tl.program_id(0) * block_assignments + tl.arange(0, block_assignments)
        )
        hidden_offsets = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
        assignment_mask = assignment_offsets < assignments
        hidden_mask = hidden_offsets < hidden
        tokens = tl.load(token_ids_ptr + assignment_offsets, mask=assignment_mask, other=0)
        packet_epochs = tl.load(
            packet_epochs_ptr + assignment_offsets,
            mask=assignment_mask,
            other=-1,
        )
        active_epochs = tl.load(active_epochs_ptr + tokens, mask=assignment_mask, other=-2)
        valid = assignment_mask & (packet_epochs == active_epochs)
        coefficients = tl.load(
            coefficients_ptr + assignment_offsets,
            mask=assignment_mask,
            other=0.0,
        )
        values = tl.load(
            expert_outputs_ptr + assignment_offsets[:, None] * hidden + hidden_offsets[None, :],
            mask=assignment_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weighted = values * coefficients[:, None]
        tl.atomic_add(
            reduced_ptr + tokens[:, None] * hidden + hidden_offsets[None, :],
            weighted,
            mask=valid[:, None] & hidden_mask[None, :],
            sem="relaxed",
        )
        tl.store(valid_ptr + assignment_offsets, valid, mask=assignment_mask)


def _validate_queue_inputs(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
    epochs: torch.Tensor,
    deadlines: torch.Tensor,
) -> tuple[int, int]:
    if top_indices.ndim != 2 or top_indices.shape[0] == 0 or top_indices.shape[1] == 0:
        raise ValueError("top_indices must have non-empty shape [rows, top_k]")
    if top_weights.shape != top_indices.shape:
        raise ValueError("top_weights must match top_indices")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    _require_integral("top_indices", top_indices)
    if not torch.is_floating_point(top_weights):
        raise ValueError("top_weights must be floating point")
    rows, top_k = top_indices.shape
    if tuple(epochs.shape) != (rows,):
        raise ValueError("epochs must have shape [rows]")
    if tuple(deadlines.shape) != (rows,):
        raise ValueError("deadlines must have shape [rows]")
    _require_integral("epochs", epochs)
    _require_integral("deadlines", deadlines)
    if top_indices.device != top_weights.device:
        raise ValueError("top_indices and top_weights must be on the same device")
    if epochs.device != top_indices.device or deadlines.device != top_indices.device:
        raise ValueError("epochs and deadlines must be on the route tensor device")
    return rows, top_k


def build_expert_queue(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
    epochs: torch.Tensor,
    deadlines: torch.Tensor,
) -> ExpertQueuePlan:
    """Compile `[rows, top_k]` router output into stable expert-major packet ranges."""
    rows, top_k = _validate_queue_inputs(
        top_indices,
        top_weights,
        num_experts=num_experts,
        epochs=epochs,
        deadlines=deadlines,
    )
    if not bool(torch.isfinite(top_weights).all().item()):
        raise ValueError("top_weights must be finite")

    indices = top_indices.to(dtype=torch.int64)
    if not bool(((indices >= 0) & (indices < num_experts)).all().item()):
        raise ValueError("top_indices contains an out-of-range expert")
    sorted_per_row = torch.sort(indices, dim=1).values
    if bool((sorted_per_row[:, 1:] == sorted_per_row[:, :-1]).any().item()):
        raise ValueError("a routed row cannot select the same expert twice")

    device = top_indices.device
    token_ids = torch.arange(rows, device=device, dtype=torch.int64).repeat_interleave(top_k)
    route_slots = torch.arange(top_k, device=device, dtype=torch.int64).repeat(rows)
    expert_ids = indices.reshape(-1)
    coefficients = top_weights.reshape(-1).to(dtype=torch.float32)
    packet_epochs = epochs.to(dtype=torch.int64).repeat_interleave(top_k)
    packet_deadlines = deadlines.to(dtype=torch.int64).repeat_interleave(top_k)

    order = torch.argsort(expert_ids, stable=True)
    expert_ids = expert_ids.index_select(0, order)
    counts = torch.bincount(expert_ids, minlength=num_experts).to(dtype=torch.int64)
    starts = torch.cumsum(counts, dim=0) - counts
    plan = ExpertQueuePlan(
        num_rows=rows,
        num_experts=num_experts,
        top_k=top_k,
        token_ids=token_ids.index_select(0, order),
        expert_ids=expert_ids,
        route_slots=route_slots.index_select(0, order),
        coefficients=coefficients.index_select(0, order),
        epochs=packet_epochs.index_select(0, order),
        deadlines=packet_deadlines.index_select(0, order),
        starts=starts,
        counts=counts,
    )
    plan.assert_valid()
    return plan


def build_expert_queue_flat(
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    route_slots: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    num_rows: int,
    num_experts: int,
    top_k: int,
    epochs: torch.Tensor,
    deadlines: torch.Tensor,
    validate: bool = True,
) -> ExpertQueuePlan:
    """Compile sparse router packets into stable expert-major packet ranges.

    `top_k` is the original maximum router width, not a promise that every row has that many
    packets. This is the queue form used by opt-in route pruning: row/slot assignments that survive
    the pruning policy are packed and sorted by expert while preserving the same scatter contract as
    dense top-k queues.
    """
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    packet_shape = tuple(token_ids.shape)
    if len(packet_shape) != 1:
        raise ValueError("token_ids must have shape [assignments]")
    if tuple(expert_ids.shape) != packet_shape:
        raise ValueError("expert_ids must match token_ids")
    if tuple(route_slots.shape) != packet_shape:
        raise ValueError("route_slots must match token_ids")
    if tuple(coefficients.shape) != packet_shape:
        raise ValueError("coefficients must match token_ids")
    if tuple(epochs.shape) != (num_rows,):
        raise ValueError("epochs must have shape [rows]")
    if tuple(deadlines.shape) != (num_rows,):
        raise ValueError("deadlines must have shape [rows]")
    _require_integral("token_ids", token_ids)
    _require_integral("expert_ids", expert_ids)
    _require_integral("route_slots", route_slots)
    _require_integral("epochs", epochs)
    _require_integral("deadlines", deadlines)
    if not torch.is_floating_point(coefficients):
        raise ValueError("coefficients must be floating point")
    device = token_ids.device
    if (
        expert_ids.device != device
        or route_slots.device != device
        or coefficients.device != device
        or epochs.device != device
        or deadlines.device != device
    ):
        raise ValueError("all sparse queue tensors must be on the same device")

    assignments = int(token_ids.numel())
    if assignments > num_rows * top_k:
        raise ValueError("sparse queue contains more assignments than dense top-k")

    token_ids_i64 = token_ids.to(dtype=torch.int64).contiguous()
    expert_ids_i64 = expert_ids.to(dtype=torch.int64).contiguous()
    route_slots_i64 = route_slots.to(dtype=torch.int64).contiguous()
    coefficients_f32 = coefficients.to(dtype=torch.float32).contiguous()
    if validate and assignments:
        if not bool(((token_ids_i64 >= 0) & (token_ids_i64 < num_rows)).all().item()):
            raise ValueError("token_ids contains an out-of-range row")
        if not bool(((expert_ids_i64 >= 0) & (expert_ids_i64 < num_experts)).all().item()):
            raise ValueError("expert_ids contains an out-of-range expert")
        if not bool(((route_slots_i64 >= 0) & (route_slots_i64 < top_k)).all().item()):
            raise ValueError("route_slots contains an out-of-range slot")
        if not bool(torch.isfinite(coefficients_f32).all().item()):
            raise ValueError("coefficients must be finite")
        route_keys = token_ids_i64 * top_k + route_slots_i64
        if int(torch.unique(route_keys).numel()) != assignments:
            raise ValueError("a routed row cannot use the same route slot twice")
        expert_keys = token_ids_i64 * num_experts + expert_ids_i64
        if int(torch.unique(expert_keys).numel()) != assignments:
            raise ValueError("a routed row cannot select the same expert twice")

    order = torch.argsort(expert_ids_i64, stable=True)
    expert_ids_sorted = expert_ids_i64.index_select(0, order)
    count_dtype = torch.int32 if token_ids.is_cuda else torch.int64
    counts = torch.bincount(expert_ids_sorted, minlength=num_experts).to(dtype=count_dtype)
    starts = torch.cumsum(counts, dim=0) - counts
    sorted_tokens = token_ids_i64.index_select(0, order)
    plan = ExpertQueuePlan(
        num_rows=num_rows,
        num_experts=num_experts,
        top_k=top_k,
        token_ids=sorted_tokens,
        expert_ids=expert_ids_sorted,
        route_slots=route_slots_i64.index_select(0, order),
        coefficients=coefficients_f32.index_select(0, order),
        epochs=epochs.to(dtype=torch.int64).index_select(0, sorted_tokens),
        deadlines=deadlines.to(dtype=torch.int64).index_select(0, sorted_tokens),
        starts=starts,
        counts=counts,
    )
    if validate:
        plan.assert_valid()
    return plan


def build_expert_queue_device(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
    epochs: torch.Tensor,
    deadlines: torch.Tensor,
    block: int = 256,
) -> ExpertQueuePlan:
    """Compile router output into expert-major queues using CUDA kernels.

    This fast path assumes upstream router code already produced valid top-k rows. Use the
    Torch/CPU `build_expert_queue` path for defensive duplicate and out-of-range validation.
    """
    if triton is None:
        raise RuntimeError("build_expert_queue_device requires Triton")
    rows, top_k = _validate_queue_inputs(
        top_indices,
        top_weights,
        num_experts=num_experts,
        epochs=epochs,
        deadlines=deadlines,
    )
    if not top_indices.is_cuda:
        raise ValueError("build_expert_queue_device requires CUDA tensors")
    assignments = rows * top_k
    experts = top_indices.to(dtype=torch.int64).contiguous().reshape(-1)
    weights = top_weights.to(dtype=torch.float32).contiguous().reshape(-1)
    packet_epochs_source = epochs.to(dtype=torch.int64).contiguous()
    packet_deadlines_source = deadlines.to(dtype=torch.int64).contiguous()
    counts = torch.zeros(num_experts, device=top_indices.device, dtype=torch.int32)
    starts = torch.empty_like(counts)
    count_grid = (triton.cdiv(assignments, block),)
    _count_kernel[count_grid](
        experts,
        counts,
        assignments=assignments,
        block=block,
    )
    prefix_block = triton.next_power_of_2(num_experts)
    _prefix_kernel[(1,)](
        counts,
        starts,
        num_experts=num_experts,
        block=prefix_block,
        num_warps=1,
    )
    cursors = torch.zeros_like(counts)
    token_ids = torch.empty(assignments, device=top_indices.device, dtype=torch.int64)
    expert_ids = torch.empty_like(token_ids)
    route_slots = torch.empty_like(token_ids)
    coefficients = torch.empty(assignments, device=top_indices.device, dtype=torch.float32)
    packet_epochs = torch.empty_like(token_ids)
    packet_deadlines = torch.empty_like(token_ids)
    _queue_write_kernel[count_grid](
        experts,
        weights,
        packet_epochs_source,
        packet_deadlines_source,
        starts,
        cursors,
        token_ids,
        expert_ids,
        route_slots,
        coefficients,
        packet_epochs,
        packet_deadlines,
        assignments=assignments,
        top_k=top_k,
        block=block,
    )
    return ExpertQueuePlan(
        num_rows=rows,
        num_experts=num_experts,
        top_k=top_k,
        token_ids=token_ids,
        expert_ids=expert_ids,
        route_slots=route_slots,
        coefficients=coefficients,
        epochs=packet_epochs,
        deadlines=packet_deadlines,
        starts=starts,
        counts=counts,
    )


def scatter_committed(
    plan: ExpertQueuePlan,
    expert_outputs: torch.Tensor,
    *,
    active_epochs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight and reduce packet outputs whose request slot epoch is still active."""
    if expert_outputs.ndim != 2 or expert_outputs.shape[0] != plan.assignments:
        raise ValueError("expert_outputs must have shape [assignments, hidden]")
    if tuple(active_epochs.shape) != (plan.num_rows,):
        raise ValueError("active_epochs must have shape [rows]")
    if expert_outputs.device != plan.token_ids.device:
        raise ValueError("expert_outputs must be on the queue tensor device")
    if active_epochs.device != plan.token_ids.device:
        raise ValueError("active_epochs must be on the queue tensor device")
    _require_integral("active_epochs", active_epochs)

    valid = plan.epochs == active_epochs.to(dtype=torch.int64).index_select(0, plan.token_ids)
    reduced = torch.zeros(
        (plan.num_rows, expert_outputs.shape[1]),
        device=expert_outputs.device,
        dtype=torch.float32,
    )
    if bool(valid.any().item()):
        rows = plan.token_ids.index_select(0, torch.nonzero(valid, as_tuple=False).flatten())
        values = expert_outputs.float()[valid] * plan.coefficients[valid, None]
        reduced.index_add_(0, rows, values)
    return reduced, valid


def scatter_committed_device(
    plan: ExpertQueuePlan,
    expert_outputs: torch.Tensor,
    *,
    active_epochs: torch.Tensor,
    block_assignments: int = 128,
    block_hidden: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA epoch-gated packet reduction using atomic add over packet rows."""
    if triton is None:
        raise RuntimeError("scatter_committed_device requires Triton")
    if expert_outputs.ndim != 2 or expert_outputs.shape[0] != plan.assignments:
        raise ValueError("expert_outputs must have shape [assignments, hidden]")
    if tuple(active_epochs.shape) != (plan.num_rows,):
        raise ValueError("active_epochs must have shape [rows]")
    if not expert_outputs.is_cuda:
        raise ValueError("scatter_committed_device requires CUDA tensors")
    if expert_outputs.device != plan.token_ids.device:
        raise ValueError("expert_outputs must be on the queue tensor device")
    if active_epochs.device != plan.token_ids.device:
        raise ValueError("active_epochs must be on the queue tensor device")
    _require_integral("active_epochs", active_epochs)
    hidden = expert_outputs.shape[1]
    reduced = torch.zeros(
        (plan.num_rows, hidden),
        device=expert_outputs.device,
        dtype=torch.float32,
    )
    valid = torch.empty(plan.assignments, device=expert_outputs.device, dtype=torch.bool)
    grid = (
        triton.cdiv(plan.assignments, block_assignments),
        triton.cdiv(hidden, block_hidden),
    )
    _scatter_reduce_kernel[grid](
        plan.token_ids,
        plan.coefficients,
        plan.epochs,
        active_epochs.to(dtype=torch.int64).contiguous(),
        expert_outputs.contiguous(),
        reduced,
        valid,
        assignments=plan.assignments,
        hidden=hidden,
        block_assignments=block_assignments,
        block_hidden=triton.next_power_of_2(block_hidden),
        num_warps=4,
    )
    return reduced, valid
