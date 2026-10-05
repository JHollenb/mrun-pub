"""Tests for the native hierarchical/routed flow research organism."""

from __future__ import annotations

import pytest
import torch

from mrun.diffusion import (
    DiffusionProgram,
    ProgramRuntime,
    ProgramService,
    create_fastapi_app,
)
from mrun.diffusion.native import (
    DensePageFlow,
    NativeFlowBackend,
    NativeFlowConfig,
    RoutedPageFlow,
    make_scene_batch,
    pages_to_image,
    parameter_count,
    sample_flow,
)


def _config() -> NativeFlowConfig:
    return NativeFlowConfig(
        page_rows=2,
        page_cols=2,
        page_dim=4,
        condition_dim=4,
        width=8,
        local_depth=2,
        active_fraction=0.5,
        denoise_steps=2,
    )


def test_native_scene_geometry_and_config_validation() -> None:
    config = _config()
    generator = torch.Generator(device="cpu").manual_seed(7)
    batch = make_scene_batch(3, config, device=torch.device("cpu"), generator=generator)

    assert batch.target_pages.shape == (3, config.page_count, config.page_dim)
    assert batch.condition.shape == (3, config.condition_dim)
    assert batch.active_mask.sum(dim=1).tolist() == [config.active_pages] * 3
    assert pages_to_image(batch.target_pages, config).shape == (3, 4, 4, 1)

    with pytest.raises(ValueError, match="divisible by four"):
        NativeFlowConfig(page_dim=6)


def test_dense_and_routed_models_have_matched_weights_and_outputs() -> None:
    config = _config()
    torch.manual_seed(11)
    dense = DensePageFlow(config)
    torch.manual_seed(11)
    routed = RoutedPageFlow(config)
    assert parameter_count(dense) == parameter_count(routed)
    assert all(
        torch.equal(first, second)
        for first, second in zip(dense.parameters(), routed.parameters(), strict=True)
    )

    generator = torch.Generator(device="cpu").manual_seed(13)
    batch = make_scene_batch(2, config, device=torch.device("cpu"), generator=generator)
    dense_prediction, dense_register = dense(
        batch.noisy_pages,
        batch.condition,
        batch.timestep,
        batch.active_mask,
    )
    routed_prediction, routed_register = routed(
        batch.noisy_pages,
        batch.condition,
        batch.timestep,
        batch.active_mask,
    )
    assert dense_prediction.shape == routed_prediction.shape == batch.target_pages.shape
    assert dense_register.shape == routed_register.shape == (2, config.width)
    assert not torch.equal(dense_prediction, routed_prediction)

    all_inactive = torch.zeros_like(batch.active_mask)
    routed_coarse, _ = routed(
        batch.noisy_pages,
        batch.condition,
        batch.timestep,
        all_inactive,
    )
    dense_with_all_inactive, _ = dense(
        batch.noisy_pages,
        batch.condition,
        batch.timestep,
        all_inactive,
    )
    _, routed_coarse_only, _, _, _ = routed._features(
        batch.noisy_pages,
        batch.condition,
        batch.timestep,
        all_inactive,
        None,
    )
    assert torch.equal(routed_coarse, routed_coarse_only)
    assert not torch.equal(dense_with_all_inactive, routed_coarse)


def test_native_backend_runs_through_program_runtime_and_batches_rows() -> None:
    config = _config()
    torch.manual_seed(17)
    backend = NativeFlowBackend(
        RoutedPageFlow(config),
        config,
        device=torch.device("cpu"),
    )
    program = DiffusionProgram.from_backend(
        backend,
        base_fingerprint="native-flow-test-v1",
    )
    runtime = ProgramRuntime()
    program_id = runtime.register(program, program_id="native-flow")
    link_id = runtime.link(program_id, schedule_fingerprint="native-schedule-v1")
    first_id = runtime.open_session(
        link_id,
        session_id="row-a",
        seed=101,
        resolution=(config.image_height, config.image_width),
        total_steps=config.denoise_steps,
    )
    second_id = runtime.open_session(
        link_id,
        session_id="row-b",
        seed=202,
        resolution=(config.image_height, config.image_width),
        total_steps=config.denoise_steps,
    )
    runtime.compile_context(first_id, "a red square")
    runtime.compile_context(second_id, "a blue circle")

    results = runtime.step_batch((first_id, second_id))
    assert [result.state_after.status for result in results] == ["completed", "completed"]
    assert results[0].telemetry["batch_size"] == 2
    assert results[0].telemetry["physical_program_calls"] == 1
    assert results[0].output.shape == (
        config.image_height,
        config.image_width,
        1,
    )
    assert results[0].output.dtype == torch.float32

    noise = torch.zeros(
        (1, config.page_count, config.page_dim), dtype=torch.float32
    )
    condition = torch.zeros((1, config.condition_dim), dtype=torch.float32)
    active = torch.ones((1, config.page_count), dtype=torch.bool)
    sampled, register = sample_flow(
        backend.model,
        noise,
        condition,
        active,
        steps=config.denoise_steps,
    )
    assert sampled.shape == noise.shape
    assert register.shape == (1, config.width)


def test_program_service_exposes_state_and_typed_render_metadata() -> None:
    config = _config()
    torch.manual_seed(23)
    backend = NativeFlowBackend(RoutedPageFlow(config), config, device="cpu")
    program = DiffusionProgram.from_backend(backend, base_fingerprint="service-test-v1")
    runtime = ProgramRuntime()
    program_id = runtime.register(program, program_id="service-program")
    link_id = runtime.link(program_id, schedule_fingerprint="service-schedule-v1")
    service = ProgramService(
        runtime,
        output_encoder=lambda output: {"shape": list(output.images[0].shape)},
    )

    assert service.health()["status"] == "ok"
    opened = service.open_session(
        {
            "link_id": link_id,
            "session_id": "service-row",
            "seed": 5,
            "resolution": [config.image_height, config.image_width],
            "total_steps": config.denoise_steps,
        }
    )
    assert opened["state"]["status"] == "allocated"
    context = service.compile_context("service-row", {"prompt": "a green triangle"})
    assert context["conditioning_key"]
    result = service.step("service-row")
    assert result["state_after"]["status"] == "completed"
    assert result["output"]["rows"] == 1
    assert result["render"]["shape"] == [config.image_height, config.image_width, 1]
    assert service.checkpoint("service-row")["has_output"] is True


def test_fastapi_adapter_exposes_program_session_routes() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    config = _config()
    torch.manual_seed(29)
    backend = NativeFlowBackend(RoutedPageFlow(config), config, device="cpu")
    program = DiffusionProgram.from_backend(backend, base_fingerprint="http-test-v1")
    runtime = ProgramRuntime()
    program_id = runtime.register(program, program_id="http-program")
    link_id = runtime.link(program_id, schedule_fingerprint="http-schedule-v1")
    client = TestClient(create_fastapi_app(ProgramService(runtime)))

    assert client.get("/healthz").json()["status"] == "ok"
    opened = client.post(
        "/v1/sessions",
        json={
            "link_id": link_id,
            "session_id": "http-row",
            "seed": 9,
            "resolution": [config.image_height, config.image_width],
            "total_steps": config.denoise_steps,
        },
    )
    assert opened.status_code == 200
    assert opened.json()["state"]["status"] == "allocated"
    context = client.post(
        "/v1/sessions/http-row/context",
        json={"prompt": "a yellow star"},
    )
    assert context.status_code == 200
    stepped = client.post("/v1/sessions/http-row/step", json={})
    assert stepped.status_code == 200
    assert stepped.json()["state_after"]["status"] == "completed"
