from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from mrun.runtime import (
    BackendCapabilities,
    BlobIdentity,
    CapabilityMismatch,
    CodecCapability,
    CommitResult,
    CompiledComponent,
    CompiledModelIdentity,
    DecodeWork,
    DeviceDescriptor,
    FallbackPolicy,
    HostBuffer,
    MemoryDomain,
    NativeOutput,
    OutputMode,
    OutputRequest,
    PlacementError,
    PrefillWork,
    PromotionStatus,
    ProvisionalStep,
    RuntimeTelemetry,
    SamplingPolicy,
    SamplingRequest,
    StateForkResult,
    StateObservation,
    WorkloadSpec,
    capability_gaps,
    plan_resident_placement,
    validate_placement_plan,
)


def _digest(character: str) -> str:
    return character * 64


def _blob(name: str, character: str, byte_count: int) -> BlobIdentity:
    return BlobIdentity(name, _digest(character), byte_count)


def _component(
    component_id: str,
    role: str,
    allocation_id: str,
    *,
    byte_count: int,
    character: str,
) -> CompiledComponent:
    return CompiledComponent(
        component_id=component_id,
        role=role,
        allocation_id=allocation_id,
        codec_id="mlx-affine-q4",
        layout_id="group-64-v1",
        physical_bytes=byte_count,
        blobs=(_blob(f"{allocation_id}.packed", character, byte_count),),
    )


def _model() -> CompiledModelIdentity:
    lexical_blob = _blob("lexical.packed", "d", 40)
    ingress = CompiledComponent(
        component_id="ingress",
        role="ingress",
        allocation_id="lexical-shared",
        codec_id="mlx-affine-q4",
        layout_id="group-64-v1",
        physical_bytes=40,
        blobs=(lexical_blob,),
    )
    egress = CompiledComponent(
        component_id="egress",
        role="egress",
        allocation_id="lexical-shared",
        codec_id="mlx-affine-q4",
        layout_id="group-64-v1",
        physical_bytes=40,
        blobs=(lexical_blob,),
    )
    return CompiledModelIdentity(
        model_name="tiny-qwen",
        architecture="qwen2",
        source_revision_sha256=_digest("a"),
        semantic_model_sha256=_digest("b"),
        component_graph_sha256=_digest("c"),
        vocab_manifest_sha256=_digest("e"),
        compiler_abi="test-compiler-v1",
        components=(
            _component("body", "body", "body", byte_count=100, character="f"),
            ingress,
            egress,
            _component("norm", "norm", "norm", byte_count=8, character="1"),
        ),
        operator_ids=("causal-gqa", "rmsnorm", "rope", "swiglu"),
        state_abi="transformer-kv-v1",
        state_dtype="fp16",
        state_bytes_per_token=16,
        max_context_tokens=128,
        semantic_token_count=32,
    )


def _capabilities(*, native_direct: bool = True) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="mlx-component",
        backend_abi="mrun-native-executor-v1",
        implementation_version="test-1",
        fabric="metal",
        memory_domain=MemoryDomain.UNIFIED,
        architectures=("qwen2",),
        operator_ids=("causal-gqa", "rmsnorm", "rope", "swiglu"),
        codecs=(
            CodecCapability(
                codec_id="mlx-affine-q4",
                layout_id="group-64-v1",
                component_roles=("body", "egress", "ingress", "norm"),
                native_direct=native_direct,
            ),
        ),
        state_abis=("transformer-kv-v1",),
        output_modes=(OutputMode.NEXT_TOKEN_ARGMAX, OutputMode.LAST_LOGITS),
        numerical_contracts=("mlx-q4-quality-v1",),
        max_context_tokens=128,
        max_batch_size=8,
        max_verify_tokens=4,
        transactional_state=True,
        scratch_only_steps=True,
        independently_committable_rows=True,
        supports_ragged_batches=True,
        telemetry_counters=("physical-weight-bytes", "unexpected-page-loads"),
        promotion_status=PromotionStatus.CANDIDATE,
    )


def _device(*, available_bytes: int = 10_000) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id="metal:0",
        fabric="metal",
        memory_domain=MemoryDomain.UNIFIED,
        total_bytes=20_000,
        available_bytes=available_bytes,
        machine_fingerprint=_digest("9"),
        attributes=(("gpu-cores", "14"), ("soc", "m3-pro")),
    )


def _workload(**changes) -> WorkloadSpec:
    values = {
        "max_batch_size": 2,
        "max_context_tokens": 10,
        "verify_tokens": 1,
        "output_mode": OutputMode.NEXT_TOKEN_ARGMAX,
        "numerical_contract": "mlx-q4-quality-v1",
        "state_abi": "transformer-kv-v1",
        "required_component_roles": ("body", "egress", "ingress", "norm"),
        "require_native_codecs": True,
        "workspace_bytes": 50,
        "headroom_bytes": 100,
        "fallback_policy": FallbackPolicy.DENY,
    }
    values.update(changes)
    return WorkloadSpec(**values)


def test_compiled_model_identity_is_canonical_frozen_and_alias_aware() -> None:
    first = _model()
    second = replace(first, components=tuple(reversed(first.components)))

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.physical_allocation_bytes == 148
    assert first.as_dict()["components"][0]["component_id"] == "body"
    with pytest.raises(FrozenInstanceError):
        first.model_name = "changed"  # type: ignore[misc]


def test_shared_allocation_requires_exactly_one_physical_identity() -> None:
    model = _model()
    egress = next(component for component in model.components if component.role == "egress")
    forged = replace(egress, layout_id="different-layout")

    with pytest.raises(ValueError, match="sharing an allocation_id"):
        replace(
            model,
            components=tuple(
                forged if component.component_id == "egress" else component
                for component in model.components
            ),
        )


def test_capability_matching_reports_all_semantic_gaps_before_allocation() -> None:
    model = _model()
    capabilities = replace(
        _capabilities(native_direct=False),
        architectures=("llama",),
        output_modes=(OutputMode.LAST_LOGITS,),
        transactional_state=False,
    )
    workload = _workload(max_context_tokens=129)

    gaps = capability_gaps(model, workload, capabilities, _device())

    assert any("architecture" in gap for gap in gaps)
    assert any("output mode" in gap for gap in gaps)
    assert any("context" in gap for gap in gaps)
    assert "backend lacks transactional state" in gaps
    assert sum("no native codec/layout" in gap for gap in gaps) == 4
    with pytest.raises(CapabilityMismatch) as error:
        plan_resident_placement(model, workload, capabilities, _device())
    assert error.value.gaps == gaps


def test_resident_plan_counts_tied_weights_once_and_binds_every_identity() -> None:
    model = _model()
    capabilities = _capabilities()
    device = _device()
    workload = _workload()

    plan = plan_resident_placement(model, workload, capabilities, device)

    assert plan.model_resident_bytes == 148
    assert plan.state.reserved_bytes == 320
    assert plan.total_reserved_bytes == 618
    assert plan.memory_budget_bytes == device.available_bytes
    assert len(plan.components) == 3
    lexical = next(item for item in plan.components if item.allocation_id == "lexical-shared")
    assert lexical.component_ids == ("egress", "ingress")
    assert lexical.roles == ("egress", "ingress")
    assert plan.performance_claim_valid
    assert plan.model_fingerprint == model.fingerprint
    assert plan.capability_fingerprint == capabilities.fingerprint
    assert plan.device_fingerprint == device.fingerprint
    assert plan.workload_fingerprint == workload.fingerprint
    validate_placement_plan(plan, model, workload, capabilities, device)


def test_recurrent_state_is_charged_per_row_not_per_context_position() -> None:
    model = replace(
        _model(),
        state_abi="mamba-recurrent-state-v1",
        state_bytes_per_token=0,
        state_fixed_bytes_per_row=1_024,
        max_context_tokens=1_000_000,
    )
    capabilities = replace(
        _capabilities(),
        state_abis=("mamba-recurrent-state-v1",),
        max_context_tokens=1_000_000,
    )
    workload = _workload(
        state_abi="mamba-recurrent-state-v1",
        max_batch_size=2,
        max_context_tokens=250_000,
    )

    plan = plan_resident_placement(model, workload, capabilities, _device())

    assert plan.state.bytes_per_token == 0
    assert plan.state.fixed_bytes_per_row == 1_024
    assert plan.state.reserved_bytes == 2_048
    assert plan.state.as_dict()["fixed_bytes_per_row"] == 1_024


def test_compiled_model_rejects_an_absent_mutable_state_charge() -> None:
    with pytest.raises(ValueError, match="token-scaled or fixed"):
        replace(_model(), state_bytes_per_token=0, state_fixed_bytes_per_row=0)


def test_resident_plan_rejects_insufficient_budget_and_stale_binding() -> None:
    model = _model()
    capabilities = _capabilities()
    workload = _workload()
    device = _device()

    with pytest.raises(PlacementError, match="model=148, state=320"):
        plan_resident_placement(
            model,
            workload,
            capabilities,
            device,
            memory_budget_bytes=617,
        )

    plan = plan_resident_placement(model, workload, capabilities, device)
    changed_device = replace(device, available_bytes=device.available_bytes - 1)
    with pytest.raises(PlacementError, match="stale, forged"):
        validate_placement_plan(plan, model, workload, capabilities, changed_device)


def test_correctness_fallback_and_compatibility_codec_never_claim_native_performance() -> None:
    plan = plan_resident_placement(
        _model(),
        _workload(
            require_native_codecs=False,
            fallback_policy=FallbackPolicy.CORRECTNESS_ONLY,
        ),
        _capabilities(native_direct=False),
        _device(),
    )

    assert plan.fully_resident
    assert not plan.performance_claim_valid
    assert plan.fallback_policy is FallbackPolicy.CORRECTNESS_ONLY


class _State:
    def __init__(self, observation: StateObservation, *, owner_id: str = "owner") -> None:
        self.runtime_id = observation.runtime_id
        self.state_id = observation.state_id
        self.owner_id = owner_id
        self._observation = observation

    def observe(self) -> StateObservation:
        return self._observation


class _Authority:
    def __init__(self, runtime_id: str, step_id: str) -> None:
        self.runtime_id = runtime_id
        self.step_id = step_id


def _observation(*, lengths: tuple[int, ...], epoch: int = 0) -> StateObservation:
    return StateObservation(
        runtime_id="runtime-1",
        state_id="state-1",
        generation=3,
        epoch=epoch,
        lengths=lengths,
        capacity=16,
        state_abi="transformer-kv-v1",
        storage_generation=7,
    )


def test_prefill_and_decode_work_encode_state_mode_and_capacity_semantics() -> None:
    empty = _observation(lengths=(0, 0))
    empty_state = _State(empty)
    output = OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX)

    prefill = PrefillWork(
        request_ids=("a", "b"),
        token_rows=((1, 2), (3, 4)),
        state=empty_state,
        parent=empty,
        output=output,
    )
    assert prefill.token_rows == ((1, 2), (3, 4))
    with pytest.raises(ValueError, match="committed prefix"):
        DecodeWork(
            request_ids=("a", "b"),
            token_rows=((1,), (2,)),
            state=empty_state,
            parent=empty,
            output=output,
        )

    committed = _observation(lengths=(15, 15), epoch=1)
    committed_state = _State(committed)
    with pytest.raises(OverflowError, match="capacity"):
        DecodeWork(
            request_ids=("a", "b"),
            token_rows=((1, 2), (2, 3)),
            state=committed_state,
            parent=committed,
            output=output,
        )
    with pytest.raises(ValueError, match="empty committed"):
        PrefillWork(
            request_ids=("a", "b"),
            token_rows=((1,), (2,)),
            state=committed_state,
            parent=committed,
            output=output,
        )


def test_provisional_step_and_commit_receipt_preserve_explicit_transaction_boundary() -> None:
    before = _observation(lengths=(2, 4), epoch=5)
    state = _State(before)
    output = NativeOutput(OutputMode.NEXT_TOKEN_ARGMAX, token_ids=(9, 10))
    authority = _Authority("runtime-1", "step-1")
    step = ProvisionalStep(
        runtime_id="runtime-1",
        step_id="step-1",
        request_ids=("a", "b"),
        state=state,
        parent=before,
        token_counts=(2, 2),
        output=output,
        authority=authority,
    )

    assert step.parent == before
    assert state.observe() == before
    after = replace(before, epoch=6, lengths=(4, 5))
    receipt = CommitResult(
        runtime_id="runtime-1",
        step_id="step-1",
        state_id="state-1",
        accepted_counts=(2, 1),
        before=before,
        after=after,
        state_bytes_written=48,
    )
    assert receipt.after.lengths == (4, 5)

    with pytest.raises(ValueError, match="accepted provisional prefix"):
        replace(receipt, accepted_counts=(1, 1))
    with pytest.raises(ValueError, match="authority identity"):
        replace(step, authority=_Authority("runtime-1", "different"))


def test_state_fork_receipt_is_tensor_opaque_exact_prefix_and_fresh_authority() -> None:
    source = _observation(lengths=(2, 4), epoch=5)
    forked = replace(
        source,
        state_id="state-2",
        generation=4,
        epoch=1,
        capacity=8,
        storage_generation=0,
    )
    state = _State(forked, owner_id="fork-owner")
    receipt = StateForkResult(
        runtime_id="runtime-1",
        source=source,
        forked=forked,
        state=state,
        state_bytes_copied=96,
    )

    assert receipt.state is state
    assert receipt.forked.lengths == source.lengths
    assert receipt.state_bytes_copied == 96
    with pytest.raises(ValueError, match="distinct state authority"):
        StateForkResult(
            runtime_id="runtime-1",
            source=source,
            forked=replace(source, generation=4),
            state=_State(replace(source, generation=4)),
        )
    stale_generation = replace(forked, generation=source.generation)
    with pytest.raises(ValueError, match="fresh authority generation"):
        replace(receipt, forked=stale_generation, state=_State(stale_generation))
    changed_prefix = replace(forked, lengths=(2, 3))
    with pytest.raises(ValueError, match="exact committed prefix"):
        replace(receipt, forked=changed_prefix, state=_State(changed_prefix))


def test_output_contract_makes_host_materialization_explicit() -> None:
    with pytest.raises(ValueError, match="requires selected_token_ids"):
        OutputRequest(OutputMode.SELECTED_LOGITS)
    with pytest.raises(TypeError, match="HostBuffer"):
        NativeOutput(OutputMode.LAST_LOGITS)
    with pytest.raises(ValueError, match="cannot hide"):
        NativeOutput(
            OutputMode.NEXT_TOKEN_ARGMAX,
            token_ids=(1,),
            values=HostBuffer("fp32", (1,), b"\x00\x00\x00\x00"),
        )

    values = HostBuffer("fp32", (1, 2), b"\x00" * 8)
    output = NativeOutput(OutputMode.LAST_LOGITS, values=values)
    assert output.values is values


def test_sample_output_contract_binds_one_policy_state_per_runtime_row() -> None:
    row = SamplingRequest(
        policy=SamplingPolicy(
            seed=17,
            temperature=0.8,
            top_p=0.9,
            top_k=8,
            frequency_penalty=0.25,
            presence_penalty=-0.5,
            logit_bias=((4, 1.0),),
        ),
        token_counts=((1, 2), (4, 1)),
        rng_counter=3,
    )
    request = OutputRequest(OutputMode.NEXT_TOKEN_SAMPLE, sampling=(row, row))
    parent = _observation(lengths=(0, 0))
    state = _State(parent)
    work = PrefillWork(
        request_ids=("a", "b"),
        token_rows=((1,), (2,)),
        state=state,
        parent=parent,
        output=request,
    )
    assert work.output.sampling == (row, row)
    assert NativeOutput(OutputMode.NEXT_TOKEN_SAMPLE, token_ids=(3, 4)).token_ids == (3, 4)
    with pytest.raises(ValueError, match="align"):
        PrefillWork(
            request_ids=("a", "b"),
            token_rows=((1,), (2,)),
            state=state,
            parent=parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_SAMPLE, sampling=(row,)),
        )


def test_runtime_telemetry_is_nonnegative_and_canonicalizes_backend_counters() -> None:
    telemetry = RuntimeTelemetry(
        runtime_id="runtime-1",
        route_backend_id="mlx-component",
        model_fingerprint=_digest("a"),
        placement_fingerprint=_digest("b"),
        prefill_calls=1,
        prefill_tokens=8,
        prefill_seconds=0.5,
        extra_counters=(("z-counter", 2), ("a-counter", 1)),
    )
    assert telemetry.extra_counters == (("a-counter", 1), ("z-counter", 2))

    with pytest.raises(ValueError, match="non-negative"):
        replace(telemetry, decode_tokens=-1)
