"""Small native hierarchical/routed flow organism.

This is a research backend for the program ABI, not a drop-in FLUX weight
rewrite. It makes the proposed execution contract explicit and measurable:

* each image is a fixed set of spatial pages;
* a cheap global/coarse path runs for every page;
* an expensive local path runs only for active pages;
* generation carries a global register through denoise steps;
* batches may contain independently conditioned rows and route masks;
* learned route logits, reusable dispatch plans, adaptive local steps, and
  compiled execution are available for the 10x research path.

The benchmark for this module trains dense and routed models on the same
synthetic scene family, then compares quality, active local work, latency, and
memory. The organism supports both supplied typed masks and a learned route
head, so oracle routing and learned/adaptive execution can be measured
separately.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from threading import RLock
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from .phase import PhaseBatchResult, PromptEmbeds

NATIVE_FLOW_BACKEND_ABI = "mrun-native-hierarchical-routed-flow-v1"


@dataclass(frozen=True, slots=True)
class NativeFlowConfig:
    """Static page geometry and model width for one native organism."""

    page_rows: int = 8
    page_cols: int = 8
    page_dim: int = 16
    condition_dim: int = 16
    width: int = 128
    route_width: int = 32
    local_depth: int = 4
    active_fraction: float = 0.25
    denoise_steps: int = 4

    def __post_init__(self) -> None:
        integer_fields = (
            "page_rows",
            "page_cols",
            "page_dim",
            "condition_dim",
            "width",
            "route_width",
            "local_depth",
            "denoise_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.page_dim % 4:
            raise ValueError("page_dim must be divisible by four for image assembly")
        if not 0.0 < float(self.active_fraction) <= 1.0:
            raise ValueError("active_fraction must be in (0, 1]")

    @property
    def page_count(self) -> int:
        return self.page_rows * self.page_cols

    @property
    def image_height(self) -> int:
        return self.page_rows * 2

    @property
    def image_width(self) -> int:
        return self.page_cols * 2

    @property
    def active_pages(self) -> int:
        return max(1, round(self.page_count * self.active_fraction))


@dataclass(frozen=True, slots=True)
class NativeSceneBatch:
    """Synthetic scene batch used for training and matched evaluation."""

    target_pages: Tensor
    condition: Tensor
    active_mask: Tensor
    noisy_pages: Tensor
    timestep: Tensor


@dataclass(frozen=True, slots=True)
class PageDispatchPlan:
    """Device-resident packed page indices reused across denoise layers/steps.

    The first organism rebuilt the flattened active index set inside every
    routed call.  That is correct but expensive for small workloads.  A plan
    is built once per compatible route mask and reused by the executor, which
    is the Python-level contract a fused CUDA/Triton kernel can consume.
    """

    batch_size: int
    page_count: int
    active_indices: Tensor
    active_count: int

    @classmethod
    def from_mask(cls, mask: Tensor) -> PageDispatchPlan:
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("dispatch masks must be a rank-2 bool tensor")
        indices = mask.reshape(-1).nonzero(as_tuple=False).flatten().contiguous()
        return cls(
            batch_size=int(mask.shape[0]),
            page_count=int(mask.shape[1]),
            active_indices=indices,
            active_count=int(indices.numel()),
        )

    def validate(self, *, batch_size: int, page_count: int, device: torch.device) -> None:
        if (self.batch_size, self.page_count) != (batch_size, page_count):
            raise ValueError("dispatch plan geometry does not match the model input")
        if self.active_indices.device != device:
            raise ValueError("dispatch plan must live on the model device")
        if self.active_indices.dtype != torch.int64:
            raise ValueError("dispatch plan indices must be int64")


@dataclass(frozen=True, slots=True)
class PageStateCache:
    """Immutable global/coarse state reusable by edit and branch programs."""

    shared: Tensor
    coarse_hidden: Tensor
    coarse_output: Tensor
    register: Tensor
    timestep: Tensor


class NativeTensorCache:
    """Small thread-safe LRU for immutable conditioning tensors.

    Cache hits return clones so a session can mutate its row-local tensor
    without corrupting another request.  The cache is deliberately bounded;
    model state remains owned by the runtime rather than the HTTP layer.
    """

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._items: OrderedDict[str, PromptEmbeds] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _clone(value: PromptEmbeds) -> PromptEmbeds:
        return PromptEmbeds(
            key=value.key,
            tensors={key: tensor.clone() for key, tensor in value.tensors.items()},
            meta=dict(value.meta),
        )

    def get(self, key: str) -> PromptEmbeds | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return self._clone(value)

    def put(self, key: str, value: PromptEmbeds) -> PromptEmbeds:
        with self._lock:
            self._items[key] = self._clone(value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
            return self._clone(value)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._items), "hits": self.hits, "misses": self.misses}


def _mlp(input_dim: int, width: int, depth: int) -> nn.Sequential:
    if depth <= 0:
        raise ValueError("MLP depth must be positive")
    layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.GELU()]
    for _ in range(depth - 1):
        layers.extend((nn.Linear(width, width), nn.GELU()))
    return nn.Sequential(*layers)


def _page_coordinates(config: NativeFlowConfig, *, device: torch.device) -> Tensor:
    row = torch.linspace(-1.0, 1.0, config.page_rows, device=device)
    col = torch.linspace(-1.0, 1.0, config.page_cols, device=device)
    yy, xx = torch.meshgrid(row, col, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(config.page_count, 2)


def _active_mask(
    batch_size: int,
    config: NativeFlowConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
    condition: Tensor | None = None,
    mode: str = "random",
) -> Tensor:
    if mode == "condition":
        if condition is None:
            raise ValueError("condition mask mode requires conditioning vectors")
        coordinates = _page_coordinates(config, device=device)
        x = coordinates[:, 0][None, :]
        y = coordinates[:, 1][None, :]
        scores = torch.sin(
            condition[:, 0:1] * (x + 1.3)
            + condition[:, 1:2] * (y - 0.7)
            + condition[:, 2:3]
        )
        scores = scores + 0.5 * torch.cos(
            condition[:, 3:4] * (x - y) + condition[:, 0:1]
        )
    elif mode == "random":
        scores = torch.rand(
            (batch_size, config.page_count), device=device, generator=generator
        )
    else:
        raise ValueError("mask mode must be 'random' or 'condition'")
    indices = scores.topk(config.active_pages, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def _scene_target(condition: Tensor, active_mask: Tensor, config: NativeFlowConfig) -> Tensor:
    """Render a deterministic coarse-plus-sparse-detail synthetic scene."""

    coordinates = _page_coordinates(config, device=condition.device)
    x = coordinates[:, 0].view(1, -1, 1)
    y = coordinates[:, 1].view(1, -1, 1)
    channels = []
    for index in range(config.page_dim):
        source = condition[:, None, index % config.condition_dim, None]
        base = torch.sin(
            source * (0.7 + 0.07 * index)
            + x * (index + 1.0)
            + y * (index + 2.0)
        )
        base = base + 0.25 * torch.cos(
            condition[:, None, (index + 3) % config.condition_dim, None]
            + x * (index + 2.0)
            - y * (index + 1.0)
        )
        detail = 0.35 * torch.sin(
            condition[:, None, (index + 7) % config.condition_dim, None] * 1.7
            + x * (index + 5.0)
            + y * (index + 3.0)
        )
        channels.append(base + active_mask[:, :, None].float() * detail)
    return torch.cat(channels, dim=-1).tanh()


def make_scene_batch(
    batch_size: int,
    config: NativeFlowConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
    mask_mode: str = "random",
) -> NativeSceneBatch:
    condition = torch.randn(
        (batch_size, config.condition_dim), device=device, generator=generator
    )
    active_mask = _active_mask(
        batch_size,
        config,
        device=device,
        generator=generator,
        condition=condition,
        mode=mask_mode,
    )
    target_pages = _scene_target(condition, active_mask, config)
    timestep = torch.rand((batch_size,), device=device, generator=generator).clamp_min(0.05)
    noise = torch.randn(target_pages.shape, device=device, generator=generator)
    noisy_pages = target_pages + timestep[:, None, None] * noise
    return NativeSceneBatch(
        target_pages=target_pages,
        condition=condition,
        active_mask=active_mask,
        noisy_pages=noisy_pages,
        timestep=timestep,
    )


class _PageFlowCore(nn.Module):
    """Shared coarse/local body used by dense and routed execution variants."""

    def __init__(self, config: NativeFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.position = nn.Parameter(torch.randn(config.page_count, config.width) * 0.02)
        self.condition = _mlp(config.condition_dim, config.width, 2)
        self.time = _mlp(1, config.width, 2)
        self.register_update = _mlp(config.width + config.page_dim + 1, config.width, 2)
        coarse_input = config.page_dim + config.width * 3 + 1
        local_input = config.page_dim + config.width * 4 + 1
        self.coarse = _mlp(coarse_input, config.width, 2)
        # Cheap coarse fallback for inactive pages.  The dense/reference path
        # never uses it; sparse program waves use it instead of paying the
        # full coarse MLP on every page.
        self.cheap_coarse = nn.Linear(coarse_input, config.width)
        self.local = _mlp(local_input, config.width, config.local_depth)
        self.coarse_head = nn.Linear(config.width, config.page_dim)
        self.cheap_coarse_head = nn.Linear(config.width, config.page_dim)
        self.local_head = nn.Linear(config.width, config.page_dim)
        # The route head is trained against dense-teacher supervision.  It is
        # not consulted by the dense reference path, so adding it preserves a
        # matched parameter surface while making learned routing explicit.
        self.route = _mlp(config.page_dim + config.width * 3 + 1, config.route_width, 2)
        self.route_head = nn.Linear(config.route_width, 1)

    def route_logits(
        self,
        pages: Tensor,
        condition: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Predict per-page local-work logits without mutating model state."""

        batch_size, page_count, _ = pages.shape
        if page_count != self.config.page_count:
            raise ValueError(
                f"expected {self.config.page_count} pages, received {page_count}"
            )
        condition_state = self.condition(condition)
        time_state = self.time(timestep[:, None])
        shared = torch.cat(
            (
                condition_state[:, None, :].expand(-1, page_count, -1),
                time_state[:, None, :].expand(-1, page_count, -1),
                self.position[None, :, :].expand(batch_size, -1, -1),
                timestep[:, None, None].expand(-1, page_count, -1),
            ),
            dim=-1,
        )
        return self.route_head(self.route(torch.cat((pages, shared), dim=-1))).squeeze(-1)

    def _features(
        self,
        pages: Tensor,
        condition: Tensor,
        timestep: Tensor,
        active_mask: Tensor,
        register: Tensor | None,
        dispatch_indices: Tensor | None = None,
        sparse_coarse: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, page_count, _ = pages.shape
        if page_count != self.config.page_count:
            raise ValueError(
                f"expected {self.config.page_count} pages, received {page_count}"
            )
        condition_state = self.condition(condition)
        time_state = self.time(timestep[:, None])
        if register is None:
            register = condition_state
        pooled = pages.mean(dim=1)
        register_delta = self.register_update(
            torch.cat((register, pooled, timestep[:, None]), dim=-1)
        )
        next_register = register + 0.10 * register_delta
        shared = torch.cat(
            (
                condition_state[:, None, :].expand(-1, page_count, -1),
                time_state[:, None, :].expand(-1, page_count, -1),
                self.position[None, :, :].expand(batch_size, -1, -1),
            ),
            dim=-1,
        )
        route = active_mask[:, :, None].to(pages.dtype)
        coarse_input = torch.cat((pages, shared, route), dim=-1)
        if sparse_coarse:
            if dispatch_indices is None:
                dispatch_indices = active_mask.reshape(-1).nonzero(as_tuple=False).flatten()
            flat_coarse_input = coarse_input.reshape(-1, coarse_input.shape[-1])
            flat_cheap_hidden = self.cheap_coarse(flat_coarse_input)
            coarse_hidden = flat_cheap_hidden.clone()
            if dispatch_indices.numel():
                active_hidden = self.coarse(
                    flat_coarse_input.index_select(0, dispatch_indices)
                )
                coarse_hidden = coarse_hidden.index_copy(0, dispatch_indices, active_hidden)
            coarse_hidden = coarse_hidden.reshape_as(coarse_input[..., : self.config.width])
        else:
            coarse_hidden = self.coarse(coarse_input)
        local_input = torch.cat((pages, coarse_hidden, shared, route), dim=-1)
        coarse_output = self.coarse_head(coarse_hidden)
        if sparse_coarse:
            cheap_output = self.cheap_coarse_head(
                flat_cheap_hidden.reshape_as(coarse_hidden)
            )
            inactive = ~active_mask[:, :, None]
            coarse_output = torch.where(inactive, cheap_output, coarse_output)
        return local_input, coarse_output, next_register, condition_state, time_state

    def _local_delta(self, local_input: Tensor) -> Tensor:
        return self.local_head(self.local(local_input))

    def prepare_state(
        self,
        pages: Tensor,
        condition: Tensor,
        timestep: Tensor,
    ) -> PageStateCache:
        """Compile global/coarse state once for branches or local edits."""

        batch_size, page_count, _ = pages.shape
        condition_state = self.condition(condition)
        time_state = self.time(timestep[:, None])
        pooled = pages.mean(dim=1)
        register = condition_state + 0.10 * self.register_update(
            torch.cat((condition_state, pooled, timestep[:, None]), dim=-1)
        )
        shared = torch.cat(
            (
                condition_state[:, None, :].expand(-1, page_count, -1),
                time_state[:, None, :].expand(-1, page_count, -1),
                self.position[None, :, :].expand(batch_size, -1, -1),
            ),
            dim=-1,
        )
        route = torch.ones(
            (batch_size, page_count, 1), device=pages.device, dtype=pages.dtype
        )
        coarse_input = torch.cat((pages, shared, route), dim=-1)
        coarse_hidden = self.coarse(coarse_input)
        coarse_output = self.coarse_head(coarse_hidden)
        return PageStateCache(shared, coarse_hidden, coarse_output, register, timestep)

    def forward_cached(
        self,
        pages: Tensor,
        active_mask: Tensor,
        state: PageStateCache,
        *,
        dispatch_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run only the local path against a previously compiled state."""

        if state.shared.shape[:2] != pages.shape[:2]:
            raise ValueError("cached state geometry does not match page input")
        route = active_mask[:, :, None].to(pages.dtype)
        local_input = torch.cat((pages, state.coarse_hidden, state.shared, route), dim=-1)
        flat_input = local_input.reshape(-1, local_input.shape[-1])
        if dispatch_indices is None:
            dispatch_indices = active_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if dispatch_indices.numel() == 0:
            return state.coarse_output, state.register
        active_delta = self._local_delta(flat_input.index_select(0, dispatch_indices))
        flat_delta = torch.zeros(
            (flat_input.shape[0], self.config.page_dim),
            device=pages.device,
            dtype=pages.dtype,
        )
        flat_delta = flat_delta.index_copy(0, dispatch_indices, active_delta)
        return state.coarse_output + flat_delta.reshape_as(state.coarse_output), state.register


class DensePageFlow(_PageFlowCore):
    """Dense reference: expensive local path runs for every spatial page."""

    def forward(
        self,
        pages: Tensor,
        condition: Tensor,
        timestep: Tensor,
        active_mask: Tensor,
        register: Tensor | None = None,
        dispatch_indices: Tensor | None = None,
        sparse_coarse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del dispatch_indices
        local_input, coarse_output, next_register, _, _ = self._features(
            pages,
            condition,
            timestep,
            active_mask,
            register,
            sparse_coarse=sparse_coarse,
        )
        local_delta = self._local_delta(local_input.reshape(-1, local_input.shape[-1]))
        return coarse_output + local_delta.reshape_as(coarse_output), next_register


class RoutedPageFlow(_PageFlowCore):
    """Hierarchical/routed reference: local path runs only for active pages."""

    def forward(
        self,
        pages: Tensor,
        condition: Tensor,
        timestep: Tensor,
        active_mask: Tensor,
        register: Tensor | None = None,
        dispatch_indices: Tensor | None = None,
        sparse_coarse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        local_input, coarse_output, next_register, _, _ = self._features(
            pages,
            condition,
            timestep,
            active_mask,
            register,
            dispatch_indices=dispatch_indices,
            sparse_coarse=sparse_coarse,
        )
        flat_input = local_input.reshape(-1, local_input.shape[-1])
        flat_mask = active_mask.reshape(-1)
        active_indices = (
            flat_mask.nonzero(as_tuple=False).flatten()
            if dispatch_indices is None
            else dispatch_indices
        )
        if active_indices.device != pages.device or active_indices.dtype != torch.int64:
            raise ValueError("dispatch indices must be int64 tensors on the model device")
        if active_indices.numel() == 0:
            return coarse_output, next_register
        active_delta = self._local_delta(flat_input.index_select(0, active_indices))
        flat_delta = torch.zeros(
            (flat_input.shape[0], self.config.page_dim),
            device=pages.device,
            dtype=pages.dtype,
        )
        flat_delta = flat_delta.index_copy(0, active_indices, active_delta)
        return coarse_output + flat_delta.reshape_as(coarse_output), next_register


def learned_route_mask(
    model: _PageFlowCore,
    pages: Tensor,
    condition: Tensor,
    timestep: Tensor,
    *,
    active_pages: int,
) -> Tensor:
    """Select a fixed active-page budget from learned route logits."""

    if active_pages <= 0 or active_pages > model.config.page_count:
        raise ValueError("active_pages must be within the page geometry")
    logits = model.route_logits(pages, condition, timestep)
    indices = logits.topk(active_pages, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def route_supervision_loss(
    model: _PageFlowCore,
    pages: Tensor,
    condition: Tensor,
    timestep: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Dense-teacher route loss used by the learned-router trainer."""

    logits = model.route_logits(pages, condition, timestep)
    if logits.shape != target_mask.shape:
        raise ValueError("route target geometry does not match route logits")
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        target_mask.to(dtype=logits.dtype),
    )


def sample_flow(
    model: DensePageFlow | RoutedPageFlow,
    noise: Tensor,
    condition: Tensor,
    active_mask: Tensor,
    *,
    steps: int,
    dispatch_plan: PageDispatchPlan | None = None,
    sparse_coarse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run the model as a short stateful flow program."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    current = noise
    register: Tensor | None = None
    for index in range(steps):
        timestep = current.new_full((current.shape[0],), 1.0 - index / steps)
        prediction, register = model(
            current,
            condition,
            timestep,
            active_mask,
            register,
            dispatch_indices=(dispatch_plan.active_indices if dispatch_plan else None),
            sparse_coarse=sparse_coarse,
        )
        current = prediction
    return current, register


def sample_adaptive_flow(
    model: _PageFlowCore,
    noise: Tensor,
    condition: Tensor,
    *,
    steps: int,
    active_pages: int,
    base_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, list[dict[str, int]]]:
    """Run a learned, budgeted route that can vary on every denoise wave.

    Every page still receives the cheap global/coarse update.  Only pages
    selected by the learned route receive the expensive local path.  The
    route is recomputed from the current state, allowing uncertain pages to
    receive work later while stable pages drop out early.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    current = noise
    register: Tensor | None = None
    trace: list[dict[str, int]] = []
    for index in range(steps):
        timestep = current.new_full((current.shape[0],), 1.0 - index / steps)
        learned = learned_route_mask(
            model,
            current,
            condition,
            timestep,
            active_pages=active_pages,
        )
        if base_mask is not None:
            learned = learned & base_mask
        plan = PageDispatchPlan.from_mask(learned)
        prediction, register = model(
            current,
            condition,
            timestep,
            learned,
            register,
            dispatch_indices=plan.active_indices,
            sparse_coarse=True,
        )
        current = prediction
        trace.append(
            {
                "step": index,
                "active_pages": plan.active_count,
                "total_pages": int(learned.numel()),
            }
        )
    if register is None:  # pragma: no cover - steps is validated above
        raise RuntimeError("adaptive flow did not produce a register")
    return current, register, trace


def sample_cached_edit_flow(
    model: _PageFlowCore,
    state: PageStateCache,
    pages: Tensor,
    active_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Render one local edit/branch wave using cached global state."""

    plan = PageDispatchPlan.from_mask(active_mask)
    return model.forward_cached(
        pages,
        active_mask,
        state,
        dispatch_indices=plan.active_indices,
    )


class CompiledRoutedExecutor:
    """Optional compiled executor for a fixed-shape routed wave.

    ``torch.compile`` is intentionally an opt-in accelerator.  The runtime
    API remains usable on CPU and on torch builds without a compiler.  A
    production CUDA backend can replace this object with a Triton/CUDA kernel
    without changing the program or session contracts.
    """

    def __init__(
        self,
        model: RoutedPageFlow,
        *,
        mode: str = "reduce-overhead",
    ) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("this torch build does not expose torch.compile")
        self.model = model
        self._compiled = torch.compile(model, mode=mode, dynamic=False)

    @property
    def config(self) -> NativeFlowConfig:
        return self.model.config

    def eval(self) -> CompiledRoutedExecutor:
        self.model.eval()
        return self

    def route_logits(self, pages: Tensor, condition: Tensor, timestep: Tensor) -> Tensor:
        return self.model.route_logits(pages, condition, timestep)

    def __call__(
        self,
        pages: Tensor,
        condition: Tensor,
        timestep: Tensor,
        active_mask: Tensor,
        register: Tensor | None = None,
        dispatch_indices: Tensor | None = None,
        sparse_coarse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if dispatch_indices is None:
            dispatch_indices = PageDispatchPlan.from_mask(active_mask).active_indices
        return self._compiled(
            pages,
            condition,
            timestep,
            active_mask,
            register,
            dispatch_indices,
            sparse_coarse,
        )


def pages_to_image(pages: Tensor, config: NativeFlowConfig) -> Tensor:
    """Reassemble page vectors as a small channel-last image tensor."""

    if pages.ndim != 3 or pages.shape[1:] != (config.page_count, config.page_dim):
        raise ValueError("pages have incompatible native image geometry")
    return pages.reshape(
        pages.shape[0],
        config.page_rows,
        config.page_cols,
        2,
        2,
        config.page_dim // 4,
    ).permute(0, 1, 3, 2, 4, 5).reshape(
        pages.shape[0], config.image_height, config.image_width, config.page_dim // 4
    )


class NativeFlowBackend:
    """Program backend for a trained dense or routed native flow model."""

    def __init__(
        self,
        model: DensePageFlow | RoutedPageFlow,
        config: NativeFlowConfig,
        *,
        device: torch.device | str,
        cache_size: int = 256,
        autocast_dtype: torch.dtype | None = None,
        compile_model: bool = False,
    ) -> None:
        self.model = model.to(device).eval()
        self.config = config
        self._device = torch.device(device)
        self._autocast_dtype = autocast_dtype
        self._embed_cache = NativeTensorCache(cache_size)
        self._plan_cache: OrderedDict[tuple[bool, ...], PageDispatchPlan] = OrderedDict()
        self._plan_cache_size = int(cache_size)
        self._executor: Any = self.model
        if compile_model:
            if not isinstance(self.model, RoutedPageFlow):
                raise ValueError("compiled native execution requires a routed model")
            self._executor = CompiledRoutedExecutor(self.model).eval()

    def encode(self, prompt: str, **params: Any) -> PromptEmbeds:
        del params  # native conditioning intentionally has one stable ABI
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._embed_cache.get(cache_key)
        if cached is not None:
            return cached
        raw = hashlib.sha256(prompt.encode("utf-8")).digest()
        values = [((byte / 255.0) * 2.0 - 1.0) for byte in raw]
        repeats = (self.config.condition_dim + len(values) - 1) // len(values)
        vector = (values * repeats)[: self.config.condition_dim]
        return self._embed_cache.put(cache_key, PromptEmbeds(
            key=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            tensors={"prompt_embeds": torch.tensor([vector], dtype=torch.float32)},
            meta={"prompt": prompt, "backend_abi": NATIVE_FLOW_BACKEND_ABI},
        ))

    def _dispatch_plan(self, mask: Tensor) -> PageDispatchPlan:
        key = tuple(bool(value) for value in mask.detach().cpu().reshape(-1).tolist())
        plan = self._plan_cache.get(key)
        if plan is not None:
            self._plan_cache.move_to_end(key)
            return plan
        plan = PageDispatchPlan.from_mask(mask)
        self._plan_cache[key] = plan
        self._plan_cache.move_to_end(key)
        while len(self._plan_cache) > self._plan_cache_size:
            self._plan_cache.popitem(last=False)
        return plan

    def cache_stats(self) -> dict[str, Any]:
        return {
            "conditioning": self._embed_cache.stats(),
            "dispatch_plan_entries": len(self._plan_cache),
            "compiled": isinstance(self._executor, CompiledRoutedExecutor),
        }

    def _route_mask(
        self,
        batch_size: int,
        route_mask: Tensor | Sequence[Sequence[bool]] | None,
    ) -> Tensor:
        if route_mask is None:
            mask = torch.zeros(
                (batch_size, self.config.page_count), device=self._device, dtype=torch.bool
            )
            mask[:, : self.config.active_pages] = True
            return mask
        mask = torch.as_tensor(route_mask, device=self._device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, self.config.page_count):
            raise ValueError("route_mask must have shape (batch, page_count)")
        return mask

    def _noise(self, batch_size: int, generator: Any) -> Tensor:
        if isinstance(generator, (tuple, list)):
            if len(generator) != batch_size:
                raise ValueError("one generator is required per native batch row")
            return torch.cat(
                [
                    torch.randn(
                        (1, self.config.page_count, self.config.page_dim),
                        device=self._device,
                        generator=item,
                    )
                    for item in generator
                ],
                dim=0,
            )
        return torch.randn(
            (batch_size, self.config.page_count, self.config.page_dim),
            device=self._device,
            generator=generator,
        )

    def generate(
        self,
        embeds: PromptEmbeds,
        *,
        generator: Any = None,
        num_inference_steps: int | None = None,
        route_mask: Tensor | Sequence[Sequence[bool]] | None = None,
        adaptive_routes: bool = False,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        result = self.generate_batch(
            [embeds],
            branch_ids=("row-0",),
            generator=generator,
            num_inference_steps=num_inference_steps,
            route_mask=route_mask,
            adaptive_routes=adaptive_routes,
        )
        return result.output

    def generate_batch(
        self,
        embeds: Sequence[PromptEmbeds],
        *,
        branch_ids: Sequence[str],
        generator: Any = None,
        num_inference_steps: int | None = None,
        route_mask: Tensor | Sequence[Sequence[bool]] | None = None,
        adaptive_routes: bool = False,
        **kwargs: Any,
    ) -> PhaseBatchResult:
        del kwargs
        rows = tuple(embeds)
        ids = tuple(branch_ids)
        if not rows or len(rows) != len(ids):
            raise ValueError("native batch embeds and branch_ids must have equal nonzero length")
        condition = torch.cat(
            [row.tensors["prompt_embeds"] for row in rows], dim=0
        ).to(self._device)
        mask = self._route_mask(len(rows), route_mask)
        steps = int(num_inference_steps or self.config.denoise_steps)
        noise = self._noise(len(rows), generator)
        plan = self._dispatch_plan(mask)
        context = (
            torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
            if self._autocast_dtype is not None and self._device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode():
            with context:
                if adaptive_routes:
                    final, _, route_trace = sample_adaptive_flow(
                        self._executor,
                        noise,
                        condition,
                        steps=steps,
                        active_pages=self.config.active_pages,
                        base_mask=mask,
                    )
                else:
                    final, _ = sample_flow(
                        self._executor,
                        noise,
                        condition,
                        mask,
                        steps=steps,
                        dispatch_plan=plan,
                    )
                    route_trace = []
        images = [image.detach().cpu() for image in pages_to_image(final, self.config)]
        return PhaseBatchResult(
            output=SimpleNamespace(images=images),
            branch_ids=ids,
            batch_size=len(ids),
            telemetry={
                "backend_abi": NATIVE_FLOW_BACKEND_ABI,
                "active_pages": int(mask.sum().item()),
                "total_pages": int(mask.numel()),
                "active_fraction": float(mask.float().mean().item()),
                "denoise_steps": steps,
                "conditioning_cache": self._embed_cache.stats(),
                "dispatch_plan_entries": len(self._plan_cache),
                "compiled_execution": isinstance(self._executor, CompiledRoutedExecutor),
                "adaptive_routes": adaptive_routes,
                "route_trace": route_trace,
            },
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "DensePageFlow",
    "CompiledRoutedExecutor",
    "NATIVE_FLOW_BACKEND_ABI",
    "NativeFlowBackend",
    "NativeFlowConfig",
    "NativeSceneBatch",
    "NativeTensorCache",
    "PageDispatchPlan",
    "PageStateCache",
    "RoutedPageFlow",
    "learned_route_mask",
    "make_scene_batch",
    "parameter_count",
    "pages_to_image",
    "route_supervision_loss",
    "sample_adaptive_flow",
    "sample_cached_edit_flow",
    "sample_flow",
]
