"""Family-owned lifecycle contracts for the program runtime."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from mrun.diffusion import (
    DiffusionProgram,
    ProgramABIError,
    ProgramCapabilityError,
    ProgramStateError,
)


class _AutoregressiveBackend:
    model_revision = "model:qwen-test"
    tokenizer_revision = "tokenizer:qwen-test"
    program_context_is_immutable = True

    def compile_program_context(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 2,
        sampler: str = "greedy",
    ) -> dict:
        return {
            "prompt": prompt,
            "tokens": [len(prompt)],
            "emitted": 0,
            "max_new_tokens": max_new_tokens,
            "sampler": sampler,
        }

    def step_program_context(
        self,
        context,
        *,
        fail: bool = False,
        eos: bool = False,
    ):
        if fail:
            raise RuntimeError("provisional decode failed")
        token = 100 + int(context["emitted"])
        next_context = {
            **context,
            "tokens": (*context["tokens"], token),
            "emitted": int(context["emitted"]) + 1,
        }
        return token, next_context, {"token_id": token, "eos": eos}

    def program_context_key(self, context) -> str:
        tokens = ",".join(str(token) for token in context["tokens"])
        return f"prefix:{tokens}"

    def program_context_checksum(self, context) -> str:
        return "|".join(
            (
                context["prompt"],
                repr(tuple(context["tokens"])),
                str(context["emitted"]),
                str(context["max_new_tokens"]),
                context["sampler"],
            )
        )

    def program_compatibility_key(self, session, state, context) -> tuple:
        assert session.state is state
        return (
            "dense-llm",
            self.model_revision,
            self.tokenizer_revision,
            context["sampler"],
        )

    def program_step_terminal(
        self,
        session,
        source_state,
        next_context,
        output,
        metadata,
    ) -> bool:
        assert session.session_id == source_state.session_id
        assert output == metadata["token_id"]
        return bool(metadata["eos"]) or (
            next_context["emitted"] >= next_context["max_new_tokens"]
        )


class _MutableCloneBackend:
    def compile_program_context(self, prompt: str, *, max_new_tokens: int = 2):
        return {
            "prompt": prompt,
            "tokens": [len(prompt)],
            "emitted": 0,
            "max_new_tokens": max_new_tokens,
        }

    def clone_program_context(self, context):
        return copy.deepcopy(context)

    def program_context_key(self, context) -> str:
        return f"mutable:{context['prompt']}:{context['tokens']}:{context['emitted']}"

    def step_program_context(self, context, *, fail: bool = False):
        token = 200 + context["emitted"]
        context["tokens"].append(token)
        context["emitted"] += 1
        if fail:
            context["tokens"].append(-1)
            raise RuntimeError("mutable provisional decode failed")
        return token, context, {"token_id": token}

    def program_step_terminal(self, _session, _source, next_context, _output, _metadata):
        return next_context["emitted"] >= next_context["max_new_tokens"]


def _link(backend=None):
    return DiffusionProgram.from_backend(
        backend or _AutoregressiveBackend(),
        base_fingerprint="base:qwen-test",
        component_graph={"family": "dense-llm"},
    ).link(schedule_fingerprint="decode:greedy-v1")


def test_family_context_can_commit_multiple_claimable_steps() -> None:
    session = _link().open_session(session_id="decode-row", total_steps=128)
    context = session.compile_context("hello", max_new_tokens=2)

    assert session.context_payload is context
    assert tuple(context["tokens"]) == (5,)
    assert session.state.conditioning_key == "prefix:5"
    assert session.binding().compatibility_key[-4:] == (
        "dense-llm",
        "model:qwen-test",
        "tokenizer:qwen-test",
        "greedy",
    )

    first = session.step()
    assert first.output == 100
    assert first.state_before.step_index == 0
    assert first.state_after.step_index == 1
    assert first.state_after.generation == 1
    assert first.state_after.status == "context_compiled"
    assert first.state_after.output_available is True
    assert first.telemetry["token_id"] == 100
    assert first.telemetry["step_granularity"] == "program_context_step"
    assert session.render() == 100
    assert session.binding().step_index == 1

    second = session.step()
    assert second.output == 101
    assert second.state_after.step_index == 2
    assert second.state_after.generation == 2
    assert second.state_after.status == "completed"
    assert session.state.conditioning_key == "prefix:5,100,101"


def test_immutable_family_checkpoint_is_bound_and_zero_copy_shareable() -> None:
    link = _link()
    session = link.open_session(session_id="source")
    session.compile_context("checkpoint", max_new_tokens=3)
    session.step()
    checkpoint = session.checkpoint()

    assert checkpoint.embeds is None
    assert checkpoint.context_payload is session.context_payload
    assert checkpoint.context_key == "prefix:10,100"
    assert checkpoint.context_checksum is not None
    assert checkpoint.metadata()["context_checksum"] == checkpoint.context_checksum
    with pytest.raises(TypeError):
        session.context_payload["emitted"] = 999
    with pytest.raises(AttributeError):
        session.context_payload["tokens"].append(999)

    restored = link.restore(checkpoint, session_id="restored")
    assert tuple(restored.context_payload["tokens"]) == (10, 100)
    assert restored.state.status == "context_compiled"
    restored.step()

    forked = restored.fork(checkpoint=checkpoint)
    assert forked.session_id not in {"source", "restored"}
    assert forked.context_payload == checkpoint.context_payload

    with pytest.raises(AttributeError):
        restored.context_payload = {"tokens": []}


def test_immutable_contract_never_deepcopies_opaque_cache_storage() -> None:
    class OpaqueCache:
        def __deepcopy__(self, _memo):
            raise AssertionError("immutable cache storage must not be deep-copied")

    cache = OpaqueCache()

    class CacheBackend(_AutoregressiveBackend):
        def compile_program_context(self, prompt: str, **kwargs):
            return {
                **super().compile_program_context(prompt, **kwargs),
                "cache": cache,
            }

    link = _link(CacheBackend())
    session = link.open_session(session_id="zero-copy")
    session.compile_context("cache", max_new_tokens=3)
    session.step()
    checkpoint = session.checkpoint()
    restored = link.restore(checkpoint, session_id="zero-copy-restored")

    assert checkpoint.context_payload["cache"] is cache
    assert restored.context_payload["cache"] is cache


def test_mutable_clone_contract_hides_live_payload_and_rolls_back() -> None:
    session = _link(_MutableCloneBackend()).open_session(session_id="rollback")
    external = session.compile_context("rollback", max_new_tokens=4)
    external["tokens"].append(999)
    assert session.state.conditioning_key == "mutable:rollback:[8]:0"

    session.step()
    state_before = session.state
    context_before = session.context_payload
    output_before = session.output

    with pytest.raises(RuntimeError, match="mutable provisional decode failed"):
        session.step(fail=True)

    assert session.state == state_before
    assert session.context_payload == context_before
    assert session.context_payload is not context_before
    assert session.output == output_before
    assert session.binding().generation == state_before.generation


def test_snapshot_restore_contract_round_trips_checkpoint() -> None:
    class SnapshotBackend(_MutableCloneBackend):
        clone_program_context = None

        def snapshot_program_context(self, context):
            return (
                context["prompt"],
                tuple(context["tokens"]),
                context["emitted"],
                context["max_new_tokens"],
            )

        def restore_program_context(self, snapshot):
            prompt, tokens, emitted, max_new_tokens = snapshot
            return {
                "prompt": prompt,
                "tokens": list(tokens),
                "emitted": emitted,
                "max_new_tokens": max_new_tokens,
            }

    link = _link(SnapshotBackend())
    session = link.open_session(session_id="snapshot")
    session.compile_context("hello", max_new_tokens=3)
    session.step()
    checkpoint = session.checkpoint()

    assert checkpoint.context_is_snapshot is True
    assert isinstance(checkpoint.context_payload, tuple)
    restored = link.restore(checkpoint, session_id="snapshot-restored")
    assert restored.context_payload == session.context_payload
    assert restored.context_payload is not session.context_payload


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (
            type(
                "MissingStep",
                (),
                {"compile_program_context": lambda self, prompt: {"prompt": prompt}},
            )(),
            "both compile_program_context",
        ),
        (
            type(
                "MissingKey",
                (),
                {
                    "program_context_is_immutable": True,
                    "compile_program_context": lambda self, prompt: {"prompt": prompt},
                    "step_program_context": lambda self, context: "output",
                },
            )(),
            "program_context_key",
        ),
        (
            type(
                "MissingOwnership",
                (),
                {
                    "compile_program_context": lambda self, prompt: {"prompt": prompt},
                    "step_program_context": lambda self, context: "output",
                    "program_context_key": lambda self, context: context["prompt"],
                },
            )(),
            "program_context_is_immutable",
        ),
    ],
)
def test_family_contracts_fail_before_execution(backend, message: str) -> None:
    with pytest.raises(ProgramCapabilityError, match=message):
        DiffusionProgram.from_backend(backend, base_fingerprint="broken")


def test_ambiguous_dual_protocol_is_rejected() -> None:
    class DualProtocol(_AutoregressiveBackend):
        def encode(self, prompt):
            return prompt

        def generate(self, context):
            return context

    with pytest.raises(ProgramCapabilityError, match="both complete legacy and family"):
        DiffusionProgram.from_backend(DualProtocol(), base_fingerprint="ambiguous")


@pytest.mark.parametrize(
    ("raw_result", "message"),
    [
        ((1,), "exactly"),
        ((1, None, {}), "empty next context"),
        ((1, {"ignored": True}, None), "metadata must be a mapping"),
    ],
)
def test_malformed_family_step_results_fail_closed(raw_result, message: str) -> None:
    class Malformed(_AutoregressiveBackend):
        def step_program_context(self, _context):
            return raw_result

    session = _link(Malformed()).open_session(session_id="malformed")
    context = session.compile_context("bad")
    state_before = session.state

    with pytest.raises(ProgramABIError, match=message):
        session.step()

    assert session.state == state_before
    assert session.context_payload is context


def test_bad_terminal_hook_rolls_back() -> None:
    class BadTerminal(_AutoregressiveBackend):
        def program_step_terminal(self, *args):
            return "not-a-bool"

    session = _link(BadTerminal()).open_session(session_id="bad-terminal")
    context = session.compile_context("bad")
    state_before = session.state
    with pytest.raises(ProgramABIError, match="must return bool"):
        session.step()
    assert session.state == state_before
    assert session.context_payload is context


def test_context_checksum_disambiguates_key_collision_and_detects_tamper() -> None:
    class CollidingKey(_AutoregressiveBackend):
        def program_context_key(self, _context) -> str:
            return "constant-key"

    link = _link(CollidingKey())
    first = link.open_session(session_id="first")
    second = link.open_session(session_id="second")
    first.compile_context("first")
    second.compile_context("second")
    first_checkpoint = first.checkpoint()
    second_checkpoint = second.checkpoint()

    assert first_checkpoint.context_key == second_checkpoint.context_key
    assert first_checkpoint.context_checksum != second_checkpoint.context_checksum
    assert first_checkpoint.checkpoint_id != second_checkpoint.checkpoint_id

    tampered = replace(
        first_checkpoint,
        context_payload=second_checkpoint.context_payload,
    )
    with pytest.raises(ProgramABIError, match="no longer matches its identity"):
        link.restore(tampered, session_id="tampered")


def test_middecode_recompile_is_rejected_without_state_change() -> None:
    session = _link().open_session(session_id="middecode")
    session.compile_context("first", max_new_tokens=3)
    session.step()
    state_before = session.state
    context_before = session.context_payload

    with pytest.raises(ProgramStateError, match="committed step"):
        session.compile_context("replacement")

    assert session.state == state_before
    assert session.context_payload is context_before


def test_family_step_batch_is_explicitly_rejected() -> None:
    class IncidentalBatch(_AutoregressiveBackend):
        def generate_batch(self, *_args, **_kwargs):
            raise AssertionError("family batch backend must not be invoked")

    link = _link(IncidentalBatch())
    first = link.open_session(session_id="batch-one")
    second = link.open_session(session_id="batch-two")
    first.compile_context("same")
    second.compile_context("same")

    with pytest.raises(ProgramCapabilityError, match="does not support family"):
        link.step_batch((first, second))


def test_cheating_immutable_backend_is_detected_and_session_poisoned() -> None:
    class Box:
        def __init__(self) -> None:
            self.value = 0

    class MutatingImmutable(_AutoregressiveBackend):
        def compile_program_context(self, prompt: str, **kwargs):
            return {**super().compile_program_context(prompt, **kwargs), "box": Box()}

        def program_context_checksum(self, context) -> str:
            return f"{super().program_context_checksum(context)}|{context['box'].value}"

        def step_program_context(self, context, **_kwargs):
            context["box"].value += 1
            raise RuntimeError("mutated opaque state")

    session = _link(MutatingImmutable()).open_session(session_id="poison")
    session.compile_context("unsafe")

    with pytest.raises(ProgramABIError, match="rollback failed"):
        session.step()

    assert session.state.status == "closed"
    with pytest.raises(ProgramStateError, match="no compiled"):
        _ = session.context_payload


def test_family_compatibility_is_backend_defined() -> None:
    link = _link()
    greedy = link.open_session(session_id="greedy")
    sampled = link.open_session(session_id="sampled")
    greedy.compile_context("same", sampler="greedy")
    sampled.compile_context("same", sampler="top-p")
    greedy_state = greedy.state
    sampled_state = sampled.state

    assert greedy.binding().compatibility_key != sampled.binding().compatibility_key
    assert greedy.state == greedy_state
    assert sampled.state == sampled_state


def test_plain_family_output_uses_unchanged_context_and_terminal_default() -> None:
    class StatelessFamily:
        program_context_is_immutable = True

        def compile_program_context(self, prompt: str):
            return {"prompt": prompt}

        def step_program_context(self, context):
            return context["prompt"].upper()

        def program_context_key(self, context):
            return f"prompt:{context['prompt']}"

    link = DiffusionProgram.from_backend(
        StatelessFamily(),
        base_fingerprint="base:stateless-family",
    ).link()
    session = link.open_session(session_id="plain-output")
    context = session.compile_context("hello")

    result = session.step()

    assert result.output == "HELLO"
    assert result.state_after.status == "completed"
    assert session.context_payload is context
