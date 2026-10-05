from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_decompiler_mixtral import LAYERS, _weights, _write_mixtral

from mrun.decompiler import (
    MIXTRAL_STATEFUL_CERTIFICATION_SCHEMA,
    MixtralStatefulError,
    MixtralStatefulParityError,
    MixtralStatefulReference,
    build_component_artifact,
    run_g10_mixtral_stateful_parity,
)


def _build_engine(tmp_path: Path) -> MixtralStatefulReference:
    source = tmp_path / "mixtral"
    _write_mixtral(source)
    record = build_component_artifact(source, tmp_path / "canonical")
    return MixtralStatefulReference.lower(record.path)


def test_mixtral_g10_certifies_segmented_logits_kv_and_batch(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)
    cases = [
        (torch.tensor([[1, 4, 7]], dtype=torch.int64), (1, 1, 1)),
        (torch.tensor([[2, 3, 5], [6, 7, 8]], dtype=torch.int64), (2, 1)),
    ]

    certification = run_g10_mixtral_stateful_parity(engine.executable, cases)

    assert certification.schema_version == MIXTRAL_STATEFUL_CERTIFICATION_SCHEMA
    assert certification.execution_certified
    assert not certification.production_runtime_eligible
    assert certification.split_patterns == ((1, 1, 1), (2, 1))
    assert certification.maximum_absolute_error <= certification.absolute_tolerance
    assert certification.maximum_absolute_error >= 0.0
    assert certification.maximum_relative_error >= 0.0
    assert certification.checks == tuple(sorted(certification.checks))
    assert {
        "bounded-context-capacity",
        "committed-kv-vs-full-ir-trace",
        "epoch-bound-single-use-commit",
        "fixed-batch-shape",
        "forked-committed-snapshot",
        "idempotent-close-and-release",
        "prefill-and-segmented-decode",
        "provisional-state-isolation",
        "rollback-and-replay",
        "stateless-vs-stateful-logits",
    } == set(certification.checks)
    assert certification.as_dict()["fingerprint"] == certification.fingerprint
    assert len(certification.final_state_fingerprints) == len(cases)


@pytest.mark.parametrize("tied", [False, True])
def test_mixtral_g10_handles_lexical_alias_and_empty_expert_dispatch(
    tmp_path: Path, tied: bool
) -> None:
    weights = _weights(tied=tied)
    for layer in range(LAYERS):
        router = f"model.layers.{layer}.block_sparse_moe.gate.weight"
        weights[router] = torch.zeros_like(weights[router])
    source = tmp_path / f"mixtral-tied-{tied}"
    _write_mixtral(source, tied=tied, weights=weights)
    record = build_component_artifact(source, tmp_path / "canonical")

    certification = run_g10_mixtral_stateful_parity(
        record.path,
        [([[1, 4, 7], [2, 3, 5]], (1, 2))],
    )

    assert certification.execution_certified
    assert certification.maximum_absolute_error <= certification.absolute_tolerance


def test_mixtral_state_transactions_reject_stale_and_double_use(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)
    state = engine.create_state(capacity=4)
    empty = state.observe()

    with pytest.raises(MixtralStatefulError, match="decode requires committed"):
        state.decode([[1]])

    prefill = state.prefill([[1, 4]])
    assert state.observe() == empty
    committed = prefill.commit()
    assert committed.length == 2
    assert committed.epoch == 1
    with pytest.raises(MixtralStatefulError, match="already consumed"):
        prefill.commit()
    with pytest.raises(MixtralStatefulError, match="already consumed"):
        prefill.rollback()

    stale = state.decode([[7]])
    winner = state.decode([[8]])
    winner_logits = winner.logits.detach().clone()
    winner.commit()
    after_winner = state.observe()
    assert after_winner.length == 3
    assert torch.equal(winner_logits, winner.logits)
    with pytest.raises(MixtralStatefulError, match="stale"):
        stale.commit()
    assert stale.rollback() == after_winner
    with pytest.raises(MixtralStatefulError, match="already consumed"):
        stale.rollback()
    assert state.observe() == after_winner


def test_mixtral_state_enforces_capacity_batch_and_transaction_shape(
    tmp_path: Path,
) -> None:
    engine = _build_engine(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        engine.create_state(capacity=0)
    with pytest.raises(ValueError, match="positive integer"):
        engine.create_state(capacity=True)
    with pytest.raises(ValueError, match="registered context"):
        engine.create_state(capacity=engine.max_position_embeddings + 1)

    state = engine.create_state(capacity=2)
    oversized = state.observe()
    with pytest.raises(MixtralStatefulError, match="capacity"):
        state.prefill([[1, 4, 7]])
    assert state.observe() == oversized

    proposal = state.prefill([[1]])
    proposal.token_count = 2
    with pytest.raises(MixtralStatefulError, match="logits have the wrong shape"):
        proposal.commit()
    assert proposal.rollback() == oversized
    assert state.observe() == oversized

    full = state.prefill([[1, 4]])
    full.commit()
    at_capacity = state.observe()
    with pytest.raises(MixtralStatefulError, match="capacity"):
        state.decode([[7]])
    assert state.observe() == at_capacity

    batched = engine.create_state(capacity=3)
    batched.prefill([[1], [2]]).commit()
    before_bad_batch = batched.observe()
    with pytest.raises(MixtralStatefulError, match="batch size cannot change"):
        batched.decode([[3]])
    assert batched.observe() == before_bad_batch


def test_mixtral_state_fork_diverges_and_close_isolated(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)
    state = engine.create_state(capacity=4)
    state.prefill([[1, 4]]).commit()
    forked = state.fork()
    assert forked.observe() == state.observe()
    for source_cache, fork_cache in zip(state.kv_snapshot(), forked.kv_snapshot(), strict=True):
        assert torch.equal(source_cache[0], fork_cache[0])
        assert torch.equal(source_cache[1], fork_cache[1])

    state.decode([[7]]).commit()
    forked.decode([[8]]).commit()
    assert state.length == forked.length == 3
    assert state.observe().state_fingerprint != forked.observe().state_fingerprint

    pending = state.decode([[9]])
    before_close = state.observe()
    state.close()
    closed = state.observe()
    assert closed.closed
    assert closed.length == before_close.length
    assert closed.epoch == before_close.epoch + 1
    with pytest.raises(MixtralStatefulError, match="closed"):
        pending.commit()
    assert pending.rollback() == closed
    with pytest.raises(MixtralStatefulError, match="already consumed"):
        pending.rollback()
    with pytest.raises(MixtralStatefulError, match="closed"):
        state.kv_snapshot()
    with pytest.raises(MixtralStatefulError, match="closed"):
        state.fork()
    with pytest.raises(MixtralStatefulError, match="closed"):
        state.decode([[9]])
    state.close()
    assert state.observe() == closed

    forked.decode([[10]]).commit()
    assert forked.length == 4
    assert len(forked.kv_snapshot()) == LAYERS
    forked.close()


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        ([], "at least one segmented case"),
        ([([[1, 4, 7]], (3,))], "segmented decode case"),
        ([([[1, 4, 7]], (1, 1))], "summing to the sequence"),
        ([([[1, 4, 7]], (1, 0, 2))], "positive widths"),
    ],
)
def test_mixtral_g10_rejects_uncertifiable_split_contracts(
    tmp_path: Path,
    cases: list[tuple[list[list[int]], tuple[int, ...]]],
    message: str,
) -> None:
    engine = _build_engine(tmp_path)

    with pytest.raises(MixtralStatefulParityError, match=message):
        run_g10_mixtral_stateful_parity(engine.executable, cases)


@pytest.mark.parametrize("tolerance", [-1.0, float("inf"), float("nan")])
def test_mixtral_g10_rejects_invalid_numerical_bounds(tmp_path: Path, tolerance: float) -> None:
    engine = _build_engine(tmp_path)

    with pytest.raises(ValueError, match="finite and non-negative"):
        run_g10_mixtral_stateful_parity(
            engine.executable,
            [([[1, 4]], (1, 1))],
            absolute_tolerance=tolerance,
        )


def test_mixtral_g10_does_not_claim_bitwise_segment_equivalence(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)

    with pytest.raises(
        MixtralStatefulParityError,
        match="segmented stateful logits differ|segmented committed K/V differs",
    ) as captured:
        run_g10_mixtral_stateful_parity(
            engine.executable,
            [([[1, 4, 7]], (1, 1, 1))],
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )

    assert captured.value.details["maximum_absolute_error"] > 0.0
