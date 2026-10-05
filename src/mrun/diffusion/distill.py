"""Teacher/student training utilities for the native program organism.

The utilities deliberately operate on the small native flow model.  They are
not a claim that an existing FLUX checkpoint can be made sparse by inference
wrapping; they provide the controlled training contract needed before scaling
the architecture.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .native import (
    DensePageFlow,
    NativeFlowConfig,
    RoutedPageFlow,
    make_scene_batch,
    route_supervision_loss,
    sample_flow,
)


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Controls a reproducible dense-teacher to few-step student run."""

    teacher_steps: int = 4
    student_steps: int = 1
    train_steps: int = 160
    batch_size: int = 16
    learning_rate: float = 2e-3
    route_loss_weight: float = 0.25
    teacher_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("teacher_steps", "student_steps", "train_steps", "batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.route_loss_weight < 0 or self.teacher_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True, slots=True)
class RealImageDistillationBatch:
    """One VAE-latent batch supplied by a real-image training pipeline.

    The runtime does not own image decoding or dataset policy.  A caller
    materializes pixels through its chosen VAE/conditioner and supplies the
    resulting noisy latents plus optional teacher route labels.  Keeping this
    record explicit prevents synthetic scene generation from being confused
    with an image-trained result.
    """

    noisy_latents: Any
    condition: Any
    timestep: Any
    target_latents: Any | None = None
    route_target: Any | None = None
    sample_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RealImageDistillationConfig:
    """Loss policy for a real-image teacher/student latent run."""

    train_steps: int = 1
    teacher_loss_weight: float = 1.0
    route_loss_weight: float = 0.0
    learning_rate: float = 2e-4
    grad_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if self.train_steps <= 0 or self.learning_rate <= 0:
            raise ValueError("train_steps and learning_rate must be positive")
        if self.teacher_loss_weight < 0 or self.route_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when supplied")


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def train_distilled_student(
    teacher: DensePageFlow,
    student: RoutedPageFlow,
    config: NativeFlowConfig,
    *,
    distillation: DistillationConfig,
    device: torch.device | str,
    seed: int,
    sparse_coarse: bool = False,
) -> dict[str, Any]:
    """Train a routed student against a dense teacher trajectory.

    The route head receives explicit active-page supervision from the synthetic
    scene generator.  The student output is simultaneously matched to the
    teacher's final state, so later benchmarks can run the learned route with
    no oracle mask.
    """

    device = torch.device(device)
    teacher = teacher.to(device).eval()
    student = student.to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=distillation.learning_rate)
    losses: list[float] = []
    route_losses: list[float] = []
    trajectory_losses: list[float] = []
    start = time.perf_counter()
    for index in range(distillation.train_steps):
        batch = make_scene_batch(
            distillation.batch_size,
            config,
            device=device,
            generator=_generator(device, seed + index),
            mask_mode="condition",
        )
        noise = torch.randn(
            batch.target_pages.shape,
            device=device,
            generator=_generator(device, seed + 10_000 + index),
        )
        with torch.inference_mode():
            teacher_target, _ = sample_flow(
                teacher,
                noise,
                batch.condition,
                batch.active_mask,
                steps=distillation.teacher_steps,
            )
        optimizer.zero_grad(set_to_none=True)
        student_prediction, _ = sample_flow(
            student,
            noise,
            batch.condition,
            batch.active_mask,
            steps=distillation.student_steps,
            sparse_coarse=sparse_coarse,
        )
        trajectory_loss = (student_prediction - teacher_target.detach()).square().mean()
        route_loss = route_supervision_loss(
            student,
            batch.noisy_pages,
            batch.condition,
            batch.timestep,
            batch.active_mask,
        )
        loss = (
            distillation.teacher_loss_weight * trajectory_loss
            + distillation.route_loss_weight * route_loss
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        route_losses.append(float(route_loss.detach().item()))
        trajectory_losses.append(float(trajectory_loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    student.eval()
    return {
        "seconds": elapsed,
        "steps": distillation.train_steps,
        "images": distillation.train_steps * distillation.batch_size,
        "images_per_second": (distillation.train_steps * distillation.batch_size) / elapsed,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "first_route_loss": route_losses[0],
        "last_route_loss": route_losses[-1],
        "first_trajectory_loss": trajectory_losses[0],
        "last_trajectory_loss": trajectory_losses[-1],
    }


def train_real_image_student(
    teacher: Any,
    student: Any,
    batches: Iterable[RealImageDistillationBatch | Mapping[str, Any]],
    *,
    student_step: Callable[[Any, RealImageDistillationBatch], Any],
    teacher_step: Callable[[Any, RealImageDistillationBatch], Any] | None = None,
    route_step: Callable[[Any, RealImageDistillationBatch], Any] | None = None,
    config: RealImageDistillationConfig | None = None,
    device: torch.device | str,
) -> dict[str, Any]:
    """Train a student on real-image latent batches through explicit adapters.

    ``student_step`` and ``teacher_step`` own model-specific scheduler and VAE
    semantics.  They must return a tensor-shaped prediction.  ``route_step``
    may return per-page logits when ``route_target`` is present.  This keeps
    the mrun training contract usable for FLUX, a native page model, or a
    future fused denoiser without pretending their forward signatures match.
    """
    config = config or RealImageDistillationConfig()
    device = torch.device(device)
    student = student.to(device).train()
    teacher = teacher.to(device).eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    teacher_losses: list[float] = []
    route_losses: list[float] = []
    started = time.perf_counter()
    iterator = iter(batches)
    for _ in range(config.train_steps):
        raw = next(iterator)
        if isinstance(raw, RealImageDistillationBatch):
            batch = raw
        elif isinstance(raw, Mapping):
            batch = RealImageDistillationBatch(
                noisy_latents=raw["noisy_latents"],
                condition=raw["condition"],
                timestep=raw["timestep"],
                target_latents=raw.get("target_latents"),
                route_target=raw.get("route_target"),
                sample_ids=tuple(str(value) for value in raw.get("sample_ids", ())),
            )
        else:
            raise TypeError("real-image batches must be mappings or RealImageDistillationBatch")
        batch = RealImageDistillationBatch(
            noisy_latents=batch.noisy_latents.to(device),
            condition=batch.condition.to(device),
            timestep=batch.timestep.to(device),
            target_latents=(
                batch.target_latents.to(device)
                if batch.target_latents is not None
                else None
            ),
            route_target=(
                batch.route_target.to(device) if batch.route_target is not None else None
            ),
            sample_ids=batch.sample_ids,
        )
        with torch.inference_mode():
            target = (
                batch.target_latents
                if batch.target_latents is not None
                else teacher_step(teacher, batch) if teacher_step is not None else None
            )
        if target is None:
            raise ValueError("a target_latents field or teacher_step is required")
        prediction = student_step(student, batch)
        teacher_loss = (prediction - target.detach()).square().mean()
        route_loss = prediction.new_zeros(())
        if route_step is not None and batch.route_target is not None:
            logits = route_step(student, batch)
            route_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.route_target.to(dtype=logits.dtype)
            )
        loss = (
            config.teacher_loss_weight * teacher_loss
            + config.route_loss_weight * route_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        teacher_losses.append(float(teacher_loss.detach().item()))
        route_losses.append(float(route_loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    student.eval()
    return {
        "schema": "mrun-real-image-distillation-v1",
        "steps": config.train_steps,
        "seconds": elapsed,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "first_teacher_loss": teacher_losses[0],
        "last_teacher_loss": teacher_losses[-1],
        "first_route_loss": route_losses[0],
        "last_route_loss": route_losses[-1],
        "image_training_contract": "caller-supplied-real-image-latents",
    }


__all__ = [
    "DistillationConfig",
    "RealImageDistillationBatch",
    "RealImageDistillationConfig",
    "train_distilled_student",
    "train_real_image_student",
]
