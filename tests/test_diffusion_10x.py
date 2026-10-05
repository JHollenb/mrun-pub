"""Focused tests for the 10x native/runtime path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from mrun.diffusion import (
    ContinuousBatchScheduler,
    DiffusionProgram,
    PhaseBatchResult,
    ProgramRuntime,
    ProgramStepResult,
    PromptEmbeds,
    RealImageDistillationBatch,
    RealImageDistillationConfig,
    inspect_flux_pipeline,
    train_real_image_student,
)
from mrun.diffusion.hybrid import (
    FluxConditioningBridge,
    FluxLatentPageAdapter,
    HybridNativeBackend,
    LatentPageLayout,
    TrainableLatentPageBridge,
)
from mrun.diffusion.native import (
    NativeFlowBackend,
    NativeFlowConfig,
    PageDispatchPlan,
    RoutedPageFlow,
    make_scene_batch,
    sample_adaptive_flow,
    sample_cached_edit_flow,
    sample_flow,
)


def _config() -> NativeFlowConfig:
    return NativeFlowConfig(
        page_rows=2,
        page_cols=3,
        page_dim=4,
        condition_dim=4,
        width=8,
        local_depth=2,
        active_fraction=0.5,
        denoise_steps=2,
    )


def test_dispatch_plan_reuse_preserves_routed_output() -> None:
    config = _config()
    torch.manual_seed(31)
    model = RoutedPageFlow(config).eval()
    generator = torch.Generator().manual_seed(32)
    batch = make_scene_batch(2, config, device=torch.device("cpu"), generator=generator)
    noise = torch.randn(batch.target_pages.shape, generator=torch.Generator().manual_seed(33))
    plan = PageDispatchPlan.from_mask(batch.active_mask)
    first, _ = sample_flow(
        model, noise, batch.condition, batch.active_mask, steps=2, dispatch_plan=plan
    )
    second, _ = sample_flow(
        model, noise, batch.condition, batch.active_mask, steps=2, dispatch_plan=plan
    )
    assert torch.equal(first, second)
    assert plan.active_count == int(batch.active_mask.sum().item())


def test_condition_route_and_adaptive_flow_produce_device_trace() -> None:
    config = _config()
    torch.manual_seed(34)
    model = RoutedPageFlow(config).eval()
    generator = torch.Generator().manual_seed(35)
    batch = make_scene_batch(
        2,
        config,
        device=torch.device("cpu"),
        generator=generator,
        mask_mode="condition",
    )
    noise = torch.randn(batch.target_pages.shape, generator=torch.Generator().manual_seed(36))
    output, register, trace = sample_adaptive_flow(
        model,
        noise,
        batch.condition,
        steps=2,
        active_pages=config.active_pages,
    )
    assert output.shape == batch.target_pages.shape
    assert register.shape == (2, config.width)
    assert len(trace) == 2
    assert all(row["active_pages"] <= 2 * config.active_pages for row in trace)


def test_cached_global_state_supports_local_edit_wave() -> None:
    config = _config()
    torch.manual_seed(361)
    model = RoutedPageFlow(config).eval()
    pages = torch.randn(1, config.page_count, config.page_dim)
    condition = torch.randn(1, config.condition_dim)
    timestep = torch.ones(1)
    state = model.prepare_state(pages, condition, timestep)
    edit_mask = torch.zeros(1, config.page_count, dtype=torch.bool)
    edit_mask[:, :1] = True
    edited = pages.clone()
    edited[:, :1] += 0.25
    output, register = sample_cached_edit_flow(model, state, edited, edit_mask)
    assert output.shape == pages.shape
    assert register.shape == (1, config.width)


def test_native_conditioning_cache_records_hits() -> None:
    config = _config()
    torch.manual_seed(37)
    backend = NativeFlowBackend(RoutedPageFlow(config), config, device="cpu")
    first = backend.encode("same prompt")
    second = backend.encode("same prompt")
    assert torch.equal(first.tensors["prompt_embeds"], second.tensors["prompt_embeds"])
    assert backend.cache_stats()["conditioning"]["hits"] == 1


def test_flux_component_inspection_is_strict_and_fail_closed() -> None:
    flux = type("Flux2KleinPipeline", (), {})()
    flux.text_encoder = object()
    flux.text_encoder_2 = object()
    flux.tokenizer = object()
    flux.tokenizer_2 = object()
    flux.transformer = object()
    flux.vae = object()
    flux.scheduler = object()
    report = inspect_flux_pipeline(flux)
    assert report.compatible is True
    assert report.denoiser_component == "transformer"

    unknown = type("UnknownPipeline", (), {})()
    unknown.transformer = object()
    assert inspect_flux_pipeline(unknown).compatible is False


def test_flux_latent_page_adapter_is_exact_roundtrip() -> None:
    layout = LatentPageLayout(
        page_rows=2,
        page_cols=3,
        channels=2,
        tile_height=2,
        tile_width=2,
    )
    latents = torch.arange(
        layout.channels * layout.height * layout.width,
        dtype=torch.float32,
    ).reshape(1, layout.channels, layout.height, layout.width)
    adapter = FluxLatentPageAdapter(layout)
    pages = adapter.to_pages(latents)
    assert pages.shape == (1, layout.page_count, layout.page_dim)
    assert torch.equal(adapter.from_pages(pages), latents)


def test_trainable_latent_page_bridge_preserves_contract_and_gradients() -> None:
    layout = LatentPageLayout(
        page_rows=2,
        page_cols=2,
        channels=2,
        tile_height=2,
        tile_width=2,
    )
    bridge = TrainableLatentPageBridge(layout, native_page_dim=5, hidden_dim=7)
    latents = torch.randn(2, layout.channels, layout.height, layout.width)
    pages = bridge.to_native_pages(latents)
    assert pages.shape == (2, layout.page_count, 5)
    reconstructed = bridge.to_flux_latents(pages)
    assert reconstructed.shape == latents.shape
    loss = bridge.roundtrip_loss(latents)
    loss.backward()
    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert bridge.contract()["vae_compatibility"] == "requires reconstruction/distillation training"


def test_real_image_distillation_adapter_trains_supplied_latent_batches() -> None:
    torch.manual_seed(371)
    teacher = torch.nn.Linear(4, 4, bias=False)
    student = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        teacher.weight.copy_(torch.eye(4))
        student.weight.zero_()
    batch = RealImageDistillationBatch(
        noisy_latents=torch.randn(3, 4),
        condition=torch.zeros(3, 1),
        timestep=torch.ones(3),
    )
    batches = [batch] * 3
    report = train_real_image_student(
        teacher,
        student,
        batches,
        student_step=lambda model, batch: model(batch.noisy_latents),
        teacher_step=lambda model, batch: model(batch.noisy_latents),
        config=RealImageDistillationConfig(train_steps=3, learning_rate=1e-2),
        device="cpu",
    )
    assert report["schema"] == "mrun-real-image-distillation-v1"
    assert report["last_loss"] < report["first_loss"]


def test_hybrid_backend_reuses_flux_conditioning_with_native_denoiser() -> None:
    config = _config()
    native = NativeFlowBackend(
        RoutedPageFlow(config),
        config,
        device="cpu",
    )

    class FakeFluxBackend:
        def encode(self, prompt: str, **kwargs: object) -> PromptEmbeds:
            del kwargs
            return PromptEmbeds(
                key=f"flux:{prompt}",
                tensors={"prompt_embeds": torch.ones(1, 3, 5)},
                meta={"source": "fake-flux"},
            )

    hybrid = HybridNativeBackend(
        native,
        FakeFluxBackend(),
        FluxConditioningBridge(embedding_dim=5, condition_dim=config.condition_dim),
    )
    embeds = hybrid.encode("a hybrid prompt")
    assert embeds.tensors["prompt_embeds"].shape == (1, config.condition_dim)
    assert hybrid.component_contract()["denoiser"] == "native hierarchical/routed backend"
    output = hybrid.generate(
        embeds,
        generator=torch.Generator().manual_seed(41),
        num_inference_steps=1,
    )
    assert len(output.images) == 1


class _Backend:
    def __init__(self) -> None:
        self.last_batch_kwargs: dict[str, object] = {}

    def encode(self, prompt: str, **kwargs: object) -> PromptEmbeds:
        del kwargs
        return PromptEmbeds(
            key=f"key:{prompt}",
            tensors={"prompt_embeds": torch.tensor([[float(len(prompt))]])},
            meta={},
        )

    def generate(self, embeds: PromptEmbeds, **kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(images=[float(embeds.tensors["prompt_embeds"].sum())])

    def generate_batch(self, embeds, *, branch_ids, **kwargs):
        self.last_batch_kwargs = dict(kwargs)
        return PhaseBatchResult(
            output=SimpleNamespace(
                images=[float(row.tensors["prompt_embeds"].sum()) for row in embeds]
            ),
            branch_ids=tuple(branch_ids),
            batch_size=len(branch_ids),
            telemetry={"physical_pipeline_calls": 1},
        )


def test_continuous_scheduler_groups_compatible_rows_and_caps_batch() -> None:
    runtime = ProgramRuntime()
    program = DiffusionProgram.from_backend(_Backend(), base_fingerprint="scheduler-test")
    program_id = runtime.register(program, program_id="scheduler-program")
    link_id = runtime.link(program_id, schedule_fingerprint="same")
    for index in range(3):
        session_id = runtime.open_session(
            link_id,
            session_id=f"row-{index}",
            resolution=(32, 32),
            total_steps=2,
        )
        runtime.compile_context(session_id, f"prompt-{index}")
    scheduler = ContinuousBatchScheduler(runtime, max_batch_size=2)
    for index in range(3):
        scheduler.enqueue(f"row-{index}")
    results = scheduler.flush(num_inference_steps=7, guidance_scale=1.0)
    assert len(results) == 3
    assert scheduler.stats()["physical_waves"] == 2
    assert scheduler.stats()["physical_rows"] == 3
    assert scheduler.stats()["execution_samples"] == 2
    assert scheduler.stats()["execution_p95_s"] >= 0.0
    assert scheduler.pending() == ()
    assert program.backend.last_batch_kwargs == {
        "height": 32,
        "width": 32,
        "num_inference_steps": 7,
        "guidance_scale": 1.0,
    }


def test_continuous_scheduler_emits_queue_telemetry_and_supports_refill() -> None:
    runtime = ProgramRuntime()
    program = DiffusionProgram.from_backend(_Backend(), base_fingerprint="queue-test")
    program_id = runtime.register(program, program_id="queue-program")
    link_id = runtime.link(program_id, schedule_fingerprint="same")
    for index in range(3):
        session_id = runtime.open_session(
            link_id,
            session_id=f"queue-row-{index}",
            resolution=(32, 32),
            total_steps=2,
        )
        runtime.compile_context(session_id, f"prompt-{index}")

    clock = [100.0]
    scheduler = ContinuousBatchScheduler(runtime, max_batch_size=2, clock=lambda: clock[0])
    scheduler.enqueue("queue-row-0", priority=1)
    scheduler.enqueue("queue-row-1")
    clock[0] = 102.5
    results = scheduler.flush(num_inference_steps=2, worker_id="worker-a")
    assert len(results) == 2
    assert all(result.telemetry["scheduler_queue_delay_s"] == 2.5 for result in results)
    assert results[0].telemetry["scheduler_batch_size"] == 2
    assert all(result.telemetry["scheduler_worker_id"] == "worker-a" for result in results)
    assert all(result.telemetry["scheduler_execution_wall_s"] >= 0.0 for result in results)
    assert scheduler.stats()["queue_delay_p95_s"] == 2.5
    assert scheduler.stats()["execution_samples"] == 1

    scheduler.refill(["queue-row-2"])
    assert scheduler.pending() == ("queue-row-2",)
    scheduler.flush(num_inference_steps=2)
    assert scheduler.stats()["submitted"] == 3
    assert scheduler.stats()["physical_waves"] == 2


def test_continuous_scheduler_cancellation_and_deadline_are_fail_closed() -> None:
    runtime = ProgramRuntime()
    program = DiffusionProgram.from_backend(_Backend(), base_fingerprint="deadline-test")
    program_id = runtime.register(program, program_id="deadline-program")
    link_id = runtime.link(program_id, schedule_fingerprint="same")
    for index in range(2):
        session_id = runtime.open_session(
            link_id,
            session_id=f"deadline-row-{index}",
            resolution=(32, 32),
            total_steps=2,
        )
        runtime.compile_context(session_id, f"prompt-{index}")

    clock = [10.0]
    scheduler = ContinuousBatchScheduler(runtime, clock=lambda: clock[0])
    scheduler.enqueue("deadline-row-0", deadline_s=1.0)
    scheduler.enqueue("deadline-row-1")
    assert scheduler.cancel("deadline-row-1", reason="test") is True
    clock[0] = 11.0
    assert scheduler.flush(num_inference_steps=2) == ()
    stats = scheduler.stats()
    assert stats["cancelled"] == 1
    assert stats["expired"] == 1
    assert stats["pending"] == 0
    events = scheduler.drain_events()
    assert any(event["event"] == "cancelled" for event in events)
    assert any(event["event"] == "expired" for event in events)


def test_continuous_scheduler_multi_worker_flush_claims_each_row_once() -> None:
    runtime = ProgramRuntime()
    program = DiffusionProgram.from_backend(_Backend(), base_fingerprint="worker-test")
    program_id = runtime.register(program, program_id="worker-program")
    link_id = runtime.link(program_id, schedule_fingerprint="same")
    for index in range(4):
        session_id = runtime.open_session(
            link_id,
            session_id=f"worker-row-{index}",
            resolution=(32, 32),
            total_steps=2,
        )
        runtime.compile_context(session_id, f"prompt-{index}")
    scheduler = ContinuousBatchScheduler(runtime, max_batch_size=2)
    for index in range(4):
        scheduler.enqueue(f"worker-row-{index}")

    def flush(worker_id: str) -> tuple[ProgramStepResult, ...]:
        return scheduler.flush(num_inference_steps=2, worker_id=worker_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(flush, ("worker-a", "worker-b"))
    combined = (*first, *second)
    assert len(combined) == 4
    assert len({result.telemetry["branch_id"] for result in combined}) == 4
    assert scheduler.pending() == ()
    assert scheduler.stats()["physical_rows"] == 4
    events = scheduler.drain_events()
    dispatches = [event for event in events if event["event"] == "dispatched"]
    assert len(dispatches) == 2
    assert {event["worker_id"] for event in dispatches} <= {"worker-a", "worker-b"}
