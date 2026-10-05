"""Tests for the runtime-first, program-shaped diffusion serving layer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mrun.diffusion import (
    DiffusionProgram,
    PhaseBatchResult,
    ProgramABIError,
    ProgramCapabilityError,
    ProgramExtension,
    ProgramManifest,
    ProgramRuntime,
    PromptEmbeds,
)


class _Pipeline:
    pass


class _Backend:
    def __init__(self) -> None:
        self.pipeline = _Pipeline()
        self._device = "cpu"
        self.encode_calls: list[str] = []
        self.generate_calls: list[dict] = []
        self.batch_calls: list[dict] = []

    def encode(self, prompt: str, **params) -> PromptEmbeds:
        self.encode_calls.append(prompt)
        value = torch.tensor(
            [[float(len(prompt)), float(len(self.encode_calls))]], dtype=torch.float32
        )
        return PromptEmbeds(
            key=f"embed:{prompt}:{params.get('max_sequence_length', 512)}",
            tensors={"prompt_embeds": value},
            meta={"prompt": prompt, "params": dict(params)},
        )

    def generate(self, embeds: PromptEmbeds, **kwargs):
        self.generate_calls.append(dict(kwargs))
        value = float(embeds.tensors["prompt_embeds"].sum().item())
        return SimpleNamespace(images=[value])

    def generate_batch(self, embeds, *, branch_ids, **kwargs):
        self.batch_calls.append({"branch_ids": tuple(branch_ids), **kwargs})
        values = [float(item.tensors["prompt_embeds"].sum().item()) for item in embeds]
        return PhaseBatchResult(
            output=SimpleNamespace(images=values),
            branch_ids=tuple(branch_ids),
            batch_size=len(branch_ids),
            telemetry={"backend": "fake", "physical_pipeline_calls": 1},
        )


class _WeightLease:
    def __init__(self, provider, ordered_keys, *, complete: bool = True):
        self.provider = provider
        self.ordered_keys = tuple(ordered_keys)
        self.complete = complete
        self.state = "new"
        self.consumed = 0

    def __enter__(self):
        assert self.provider.active is None
        self.provider.active = self
        self.state = "active"
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None and self.complete:
            self.consumed = len(self.ordered_keys)
        self.provider.active = None
        self.state = "released"
        return False

    def telemetry(self):
        telemetry = {
            "store_fingerprint": self.provider.identity_fingerprint,
            "content_fingerprint": self.provider.content_fingerprint,
            "ordered_keys": list(self.ordered_keys),
            "declared_demands": len(self.ordered_keys),
            "consumed_demands": self.consumed,
            "state": self.state,
            "numerical_lane": "int8_weight_only",
            "device": "cpu",
            "dtype": "torch.float32",
            "residency_measurement_scope": "test-lease-owned-pages",
            "measured_retained_device_bytes_current": 0,
            "measured_retained_device_bytes_peak": 8,
            "integrity_capable": self.provider.integrity_capable,
            "integrity_required": True,
            "integrity_authority": "sha256-page-verification-required",
            "store_integrity_status": "partially-verified",
            "integrity_verification_attempts": 2,
            "integrity_verification_failures": 0,
            "integrity_verified_pages": 2,
            "integrity_verified_weight_bytes": 8,
            "integrity_verified_scale_bytes": 8,
            "integrity_verified_bytes": 16,
            "max_retained_device_bytes": 48,
            "retention_authority": "bounded-belady-next-use",
            "retention_evictions": 1,
            "retention_eviction_events": [
                {
                    "demand_index": 1,
                    "evicted_key": "block.proj.weight",
                    "evicted_device_bytes": 48,
                    "next_use_index": None,
                }
            ],
        }
        telemetry.update(self.provider.telemetry_overrides)
        return telemetry


class _WeightProvider:
    def __init__(
        self,
        fingerprint: str,
        *,
        complete: bool = True,
        integrity_capable: bool = True,
        telemetry_overrides=None,
    ):
        self.identity_fingerprint = "e" * 64
        self.content_fingerprint = fingerprint
        self.integrity_capable = integrity_capable
        self.complete = complete
        self.telemetry_overrides = dict(telemetry_overrides or {})
        self.active = None
        self.calls = []
        self.last_lease = None

    def lease(self, ordered_keys, **kwargs):
        self.calls.append((tuple(ordered_keys), dict(kwargs)))
        self.last_lease = _WeightLease(self, ordered_keys, complete=self.complete)
        return self.last_lease


class _PagedBackend(_Backend):
    def __init__(self, provider: _WeightProvider, *, fail: bool = False):
        super().__init__()
        self.provider = provider
        self.fail = fail
        self.pipeline.transformer = SimpleNamespace(
            _saturn_diffusion_qstore=provider,
            _saturn_diffusion_qstore_report={"lease_required": True},
        )

    def generate(self, embeds: PromptEmbeds, **kwargs):
        assert self.provider.active is not None
        if self.fail:
            raise RuntimeError("backend failed")
        return super().generate(embeds, **kwargs)


def _program(backend: _Backend | None = None):
    return DiffusionProgram.from_backend(
        backend or _Backend(),
        base_fingerprint="base:flux-klein-test",
        numerical_contract={"mode": "authority", "step_granularity": "atomic"},
    )


def _weight_policy(fingerprint: str):
    return {
        "weight_page_lease": {
            "schema": "mrun-diffusion-weight-page-lease-policy-v2",
            "provider_fingerprint": fingerprint,
            "ordered_page_schedule": ["block.0.weight", "block.1.weight"],
            "device": "cpu",
            "dtype": "torch.float32",
            "numerical_lane": "int8_weight_only",
            "max_retained_device_bytes": 48,
        }
    }


def test_manifest_is_order_independent_and_round_trips() -> None:
    first = ProgramManifest(
        base_fingerprint="base-a",
        component_graph={"pipeline_class": "_Pipeline", "b": 2, "a": 1},
        ports={"output": ["image"], "input": ["prompt"]},
    )
    second = ProgramManifest(
        base_fingerprint="base-a",
        component_graph={"a": 1, "b": 2, "pipeline_class": "_Pipeline"},
        ports={"input": ["prompt"], "output": ["image"]},
    )
    assert first.fingerprint == second.fingerprint
    assert ProgramManifest.from_json(first.to_json()).fingerprint == first.fingerprint

    with pytest.raises(ProgramABIError, match="unknown manifest fields"):
        ProgramManifest.from_dict({**first.to_dict(), "unexpected": True})


def test_link_checks_extension_base_and_field_ownership() -> None:
    program = _program()
    linked = program.link(
        extensions=(
            ProgramExtension(
                name="scene-state",
                fingerprint="ext:scene-state-v1",
                base_fingerprint="base:flux-klein-test",
                reads=("prompt",),
                writes=("scene",),
            ),
        ),
        references=("ref:character-1",),
        schedule_fingerprint="schedule:klein-v1",
    )
    assert linked.link_fingerprint
    assert linked.compatibility_key[1] == "schedule:klein-v1"
    assert linked.to_dict()["references"] == ["ref:character-1"]

    with pytest.raises(ProgramABIError, match="targets"):
        program.link(
            extensions=(
                ProgramExtension(
                    name="wrong-base",
                    fingerprint="ext:wrong-base",
                    base_fingerprint="base:other",
                ),
            )
        )

    with pytest.raises(ProgramABIError, match="both write"):
        program.link(
            extensions=(
                ProgramExtension("one", "ext:one", writes=("scene",)),
                ProgramExtension("two", "ext:two", writes=("scene",)),
            )
        )


def test_weight_page_policy_binds_provider_schedule_and_step_lease() -> None:
    fingerprint = "a" * 64
    provider = _WeightProvider(fingerprint)
    backend = _PagedBackend(provider)
    linked = _program(backend).link(resource_policy=_weight_policy(fingerprint))
    assert linked.to_dict()["resource_policy"]["weight_page_lease"]["ordered_page_schedule"] == [
        "block.0.weight",
        "block.1.weight",
    ]

    session = linked.open_session(session_id="paged", seed=4, total_steps=2)
    session.compile_context("paged prompt")
    result = session.step()

    assert provider.active is None
    assert provider.calls == [
        (
            ("block.0.weight", "block.1.weight"),
            {
                "device": "cpu",
                "dtype": torch.float32,
                "numerical_lane": "int8_weight_only",
                "require_integrity": True,
                "max_retained_device_bytes": 48,
            },
        )
    ]
    telemetry = result.telemetry["weight_page_lease"]
    assert telemetry["content_fingerprint"] == fingerprint
    assert telemetry["store_fingerprint"] == provider.identity_fingerprint
    assert telemetry["ordered_keys"] == ["block.0.weight", "block.1.weight"]
    assert telemetry["state"] == "released"
    assert telemetry["measured_retained_device_bytes_current"] == 0


def test_weight_page_policy_refuses_content_mismatch_and_incomplete_telemetry() -> None:
    with pytest.raises(ProgramABIError, match="content fingerprint"):
        _program(_PagedBackend(_WeightProvider("b" * 64))).link(
            resource_policy=_weight_policy("a" * 64)
        )

    provider = _WeightProvider("c" * 64, complete=False)
    linked = _program(_PagedBackend(provider)).link(
        resource_policy=_weight_policy(provider.content_fingerprint)
    )
    session = linked.open_session(session_id="incomplete")
    session.compile_context("prompt")
    with pytest.raises(ProgramABIError, match="exact schedule"):
        session.step()
    assert provider.active is None
    assert session.state.status == "context_compiled"


def test_weight_page_policy_refuses_legacy_and_changed_content_provider() -> None:
    legacy = _WeightProvider("a" * 64, integrity_capable=False)
    with pytest.raises(ProgramABIError, match="integrity-capable"):
        _program(_PagedBackend(legacy)).link(
            resource_policy=_weight_policy(legacy.content_fingerprint)
        )

    missing_content = _WeightProvider("a" * 64)
    missing_content.content_fingerprint = None
    with pytest.raises(ProgramABIError, match="no authoritative content fingerprint"):
        _program(_PagedBackend(missing_content)).link(resource_policy=_weight_policy("a" * 64))

    provider = _WeightProvider("b" * 64)
    linked = _program(_PagedBackend(provider)).link(
        resource_policy=_weight_policy(provider.content_fingerprint)
    )
    provider.content_fingerprint = "c" * 64
    session = linked.open_session(session_id="changed-content")
    session.compile_context("prompt")
    with pytest.raises(ProgramABIError, match="content fingerprint changed"):
        session.step()
    assert provider.active is None
    assert session.state.status == "context_compiled"


@pytest.mark.parametrize(
    ("telemetry_overrides", "message"),
    [
        ({"content_fingerprint": "f" * 64}, "content fingerprint mismatch"),
        ({"integrity_capable": False}, "not integrity-capable"),
        ({"integrity_required": False}, "non-authoritative"),
        (
            {"integrity_authority": "non-authoritative-explicit-opt-out"},
            "no integrity authority",
        ),
        ({"store_integrity_status": "verification-failed"}, "status is not verified"),
        ({"integrity_verification_failures": 1}, "verification is incomplete"),
        ({"max_retained_device_bytes": None}, "retention budget mismatch"),
        ({"retention_authority": "unbounded-discovery-only"}, "non-authoritative"),
        ({"retention_eviction_events": []}, "eviction event ledger"),
        ({"measured_retained_device_bytes_peak": 49}, "exceeded its retention budget"),
    ],
)
def test_weight_page_policy_refuses_non_authoritative_integrity_telemetry(
    telemetry_overrides, message
) -> None:
    provider = _WeightProvider("d" * 64, telemetry_overrides=telemetry_overrides)
    linked = _program(_PagedBackend(provider)).link(
        resource_policy=_weight_policy(provider.content_fingerprint)
    )
    session = linked.open_session(session_id="hostile-integrity")
    session.compile_context("prompt")
    with pytest.raises(ProgramABIError, match=message):
        session.step()
    assert provider.active is None
    assert session.state.status == "context_compiled"


def test_weight_page_lease_releases_and_rolls_back_on_backend_failure() -> None:
    provider = _WeightProvider("d" * 64)
    linked = _program(_PagedBackend(provider, fail=True)).link(
        resource_policy=_weight_policy(provider.content_fingerprint)
    )
    session = linked.open_session(session_id="failed")
    session.compile_context("prompt")
    with pytest.raises(RuntimeError, match="backend failed"):
        session.step()
    assert provider.active is None
    assert provider.last_lease.state == "released"
    assert session.state.status == "context_compiled"


def test_session_is_program_shaped_and_checkpointable() -> None:
    backend = _Backend()
    linked = _program(backend).link(schedule_fingerprint="schedule:test")
    session = linked.open_session(
        session_id="session-a",
        seed=7,
        resolution=(64, 96),
        total_steps=4,
    )
    assert session.state.status == "allocated"

    embeds = session.compile_context("a red square", max_sequence_length=512)
    assert session.state.status == "context_compiled"
    assert session.state.conditioning_key == embeds.key
    assert session.binding().compatibility_key[-2:] == ((64, 96), 4)

    checkpoint = session.checkpoint()
    embeds.tensors["prompt_embeds"][0, 0] = 999.0
    result = session.step()
    assert result.state_before.status == "context_compiled"
    assert result.state_after.status == "completed"
    assert result.state_after.generation == 1
    assert session.render() is result.output
    assert backend.generate_calls[0]["num_inference_steps"] == 4
    assert backend.generate_calls[0]["height"] == 64
    assert backend.generate_calls[0]["width"] == 96
    assert backend.generate_calls[0]["generator"].initial_seed() == 7

    restored = linked.restore(checkpoint, session_id="session-restored")
    assert restored.session_id == "session-restored"
    assert restored.state.status == "context_compiled"
    assert restored._embeds is not embeds  # checkpoint cloned the row-local payload
    assert restored._embeds is not None
    assert restored._embeds.tensors["prompt_embeds"][0, 0].item() != 999.0
    restored_result = restored.step()
    assert restored_result.state_after.status == "completed"

    forked = restored.fork(checkpoint=checkpoint)
    assert forked.session_id != checkpoint.state.session_id
    assert forked.state.status == "context_compiled"


def test_batch_step_commits_rows_transactionally() -> None:
    backend = _Backend()
    linked = _program(backend).link(schedule_fingerprint="schedule:test")
    first = linked.open_session(session_id="a", seed=1, resolution=(64, 64), total_steps=4)
    second = linked.open_session(session_id="b", seed=2, resolution=(64, 64), total_steps=4)
    first.compile_context("first")
    second.compile_context("second")

    results = linked.step_batch((first, second))
    assert [item.state_after.status for item in results] == ["completed", "completed"]
    assert [item.state_after.generation for item in results] == [1, 1]
    assert backend.batch_calls[0]["branch_ids"] == ("a", "b")
    assert len(backend.batch_calls[0]["generator"]) == 2
    assert results[0].telemetry["batch_size"] == 2
    assert first.render() != second.render()


def test_batch_rejects_incompatible_rows_without_consuming_state() -> None:
    linked = _program().link(schedule_fingerprint="schedule:test")
    first = linked.open_session(session_id="a", resolution=(64, 64))
    second = linked.open_session(session_id="b", resolution=(128, 64))
    first.compile_context("first")
    second.compile_context("second")

    with pytest.raises(ProgramABIError, match="compatible"):
        linked.step_batch((first, second))
    assert first.state.status == "context_compiled"
    assert second.state.status == "context_compiled"


def test_batch_requires_a_real_backend_batch_operation() -> None:
    class NoBatchBackend(_Backend):
        generate_batch = None

    linked = _program(NoBatchBackend()).link()
    session = linked.open_session(session_id="only")
    session.compile_context("prompt")
    with pytest.raises(ProgramCapabilityError, match="generate_batch"):
        linked.step_batch((session,))


def test_backend_pipeline_class_is_an_abi_boundary() -> None:
    backend = _Backend()
    with pytest.raises(ProgramABIError, match="pipeline_class"):
        DiffusionProgram.from_backend(
            backend,
            base_fingerprint="base",
            component_graph={"pipeline_class": "DifferentPipeline"},
        )


def test_runtime_registry_exposes_transport_neutral_program_handles() -> None:
    backend = _Backend()
    runtime = ProgramRuntime()
    program_id = runtime.register(_program(backend), program_id="flux-test")
    link_id = runtime.link(program_id, schedule_fingerprint="schedule:runtime")
    session_id = runtime.open_session(
        link_id,
        session_id="runtime-session",
        seed=3,
        resolution=(64, 64),
        total_steps=4,
    )
    runtime.compile_context(session_id, "runtime prompt")
    checkpoint = runtime.checkpoint(session_id)
    result = runtime.step(session_id)
    assert result.state_after.status == "completed"
    assert runtime.get_session(session_id).render() is result.output

    restored_id = runtime.restore(link_id, checkpoint, session_id="runtime-restored")
    assert runtime.get_session(restored_id).state.status == "context_compiled"
    runtime.close_session(session_id)
    runtime.close_session(restored_id)
