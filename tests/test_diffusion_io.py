from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mrun.diffusion import (
    ComponentIOError,
    ComponentIOSpec,
    DiffusionProgram,
    PortBinding,
    PromptEmbeds,
    component_io_manifest,
    make_component_frame,
)


class _Backend:
    def __init__(self) -> None:
        self.pipeline = SimpleNamespace()
        self._device = "cpu"

    def encode(self, prompt: str, **params: object) -> PromptEmbeds:
        value = torch.tensor([[float(len(prompt))]], dtype=torch.float32)
        return PromptEmbeds(
            key=f"embed:{prompt}",
            tensors={"prompt_embeds": value},
            meta={"params": dict(params)},
        )

    def generate(self, embeds: PromptEmbeds, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(images=[float(embeds.tensors["prompt_embeds"].sum())])


def test_default_component_contracts_cover_the_six_mars_streams() -> None:
    manifest = component_io_manifest()
    assert {port["stream"] for spec in manifest.values() for port in spec["inputs"]} >= {
        "data",
        "control",
        "state",
        "route",
        "resource",
        "evidence",
    }
    denoiser = ComponentIOSpec.from_mapping(manifest["denoiser"])
    frame = make_component_frame(
        component_id="denoiser",
        operation="forward",
        model_identity="link:test",
        bindings=(
            PortBinding("latent", "data", "in"),
            PortBinding("prompt_embeds", "data", "in"),
            PortBinding("text_ids", "data", "in"),
            PortBinding("latent_ids", "data", "in"),
            PortBinding("timestep", "control", "in"),
            PortBinding("weight_lease", "resource", "in"),
            PortBinding("trace", "evidence", "in"),
            PortBinding("prediction", "data", "out"),
            PortBinding("trace", "evidence", "out"),
        ),
    )
    denoiser.validate_frame(frame)
    assert frame.to_dict()["trace_token"].startswith("trace-")
    assert frame.fingerprint


def test_component_contract_rejects_missing_debug_input() -> None:
    program = ComponentIOSpec.from_mapping(component_io_manifest()["program"])
    frame = make_component_frame(
        component_id="program",
        operation="step",
        model_identity="link:test",
        bindings=(
            PortBinding("context_handle", "data", "in"),
            PortBinding("schedule", "control", "in"),
        ),
    )
    with pytest.raises(ComponentIOError, match="missing required inputs"):
        program.validate_frame(frame)


def test_program_emits_io_frames_for_conditioning_and_execution() -> None:
    program = DiffusionProgram.from_backend(
        _Backend(),
        base_fingerprint="base:test",
        numerical_contract={"mode": "authority", "step_granularity": "atomic"},
    )
    session = program.link(schedule_fingerprint="schedule:test").open_session(
        session_id="mars-session",
        seed=7,
        resolution=(64, 64),
        total_steps=4,
    )
    session.compile_context("a red square")
    assert session.last_io_frame is not None
    assert session.last_io_frame.component_id == "conditioner"
    assert session.last_io_frame.branch_id == "mars-session"

    result = session.step()
    assert result.io_frame is not None
    assert result.io_frame.component_id == "program"
    assert result.io_frame.operation == "step"
    assert result.io_frame.parent_frame_id is not None
    assert any(binding.port == "output" for binding in result.io_frame.bindings)

    checkpoint = session.checkpoint()
    assert checkpoint.metadata()["io_frame"] is not None
