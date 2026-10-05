"""Engine gap-fill unit tests — head patch ops, capabilities, batched-patch fallback.

Pure-tensor + dummy-engine tests run always; real-model checks live in the model-gated
suites (test_paged_parity / test_accel_backends) and the recorder acceptance smoke.
"""

from __future__ import annotations

import numpy as np
import torch

from mrun.engine._base_impl import BaseEngine
from mrun.engine.base import EngineCapabilities
from mrun.engine.kernels.paged_forward import _apply_head_patch_ops, _apply_resid_patch_ops

# --------------------------------------------------------------------------- head patch ops


def test_head_patch_zero_single_sequence():
    x = torch.ones(5, 4, 8)  # [T, nH, hd]
    out = _apply_head_patch_ops(x, [("zero", [1, 3], None)])
    assert torch.all(out[:, 1] == 0) and torch.all(out[:, 3] == 0)
    assert torch.all(out[:, 0] == 1) and torch.all(out[:, 2] == 1)


def test_head_patch_zero_batched():
    x = torch.ones(2, 5, 4, 8)  # [B, T, nH, hd]
    out = _apply_head_patch_ops(x, [("zero", [0], None)])
    assert torch.all(out[:, :, 0] == 0)
    assert torch.all(out[:, :, 1:] == 1)


def test_head_patch_scale_and_add():
    x = torch.ones(3, 2, 4)
    out = _apply_head_patch_ops(x, [("scale", [0], 0.5)])
    assert torch.allclose(out[:, 0], torch.full((3, 4), 0.5))
    out = _apply_head_patch_ops(x, [("add_amp", [1], torch.full((4,), 2.0))])
    assert torch.allclose(out[:, 1], torch.full((3, 4), 3.0))


def test_head_patch_unknown_op_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown head patch op"):
        _apply_head_patch_ops(torch.ones(2, 2, 2), [("nope", [0], None)])


# --------------------------------------------------------------------------- resid patch ops


def test_resid_proj_remove_zeros_component_along_u():
    # h -> h - (h·u)u leaves the residual orthogonal to u (zero projection onto u).
    torch.manual_seed(0)
    h = torch.randn(5, 16)  # [T, d]
    u = torch.randn(16)
    out = _apply_resid_patch_ops(h.clone(), [("proj_remove", u, None)])
    un = u / u.norm()
    assert torch.allclose(out.float() @ un, torch.zeros(5), atol=1e-5)


def test_resid_proj_remove_renormalizes_u():
    # An unnormalized u must give the same result as its normalized form.
    torch.manual_seed(1)
    h = torch.randn(3, 8)
    u = torch.randn(8) * 7.3
    a = _apply_resid_patch_ops(h.clone(), [("proj_remove", u, None)])
    b = _apply_resid_patch_ops(h.clone(), [("proj_remove", u / u.norm(), None)])
    assert torch.allclose(a, b, atol=1e-6)


def test_resid_proj_remove_supports_true_batch_tensor():
    torch.manual_seed(2)
    hidden = torch.randn(3, 5, 8)  # [batch, tokens, hidden]
    direction = torch.randn(8)
    out = _apply_resid_patch_ops(hidden, [("proj_remove", direction, None)])
    unit = direction / direction.norm()
    assert torch.allclose(out.float() @ unit, torch.zeros(3, 5), atol=1e-5)


def test_resid_position_replace_is_token_local():
    hidden = torch.zeros(4, 3)
    donor = torch.tensor([[2.0, 3.0, 4.0]])
    out = _apply_resid_patch_ops(hidden.clone(), [("position_replace", [2], donor)])
    assert torch.equal(out[2], donor[0])
    assert torch.equal(out[[0, 1, 3]], hidden[[0, 1, 3]])


def test_resid_position_replace_supports_batched_rows():
    hidden = torch.zeros(2, 3, 4)
    donor = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    out = _apply_resid_patch_ops(hidden.clone(), [("position_replace", [1], donor)])
    assert torch.equal(out[:, 1], donor[:, 0])
    assert torch.equal(out[:, [0, 2]], hidden[:, [0, 2]])


def test_resid_position_lerp_applies_typed_dose():
    hidden = torch.zeros(3, 2)
    donor = torch.tensor([[4.0, 8.0]])
    out = _apply_resid_patch_ops(
        hidden.clone(), [("position_lerp", [1], {"value": donor, "dose": 0.25})]
    )

    assert torch.equal(out[0], hidden[0])
    assert torch.allclose(out[1], torch.tensor([1.0, 2.0]))
    assert torch.equal(out[2], hidden[2])


def test_resid_position_add_preserves_live_state_and_is_token_local():
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    delta = torch.tensor([[0.5, -1.0, 2.0, 4.0]])
    out = _apply_resid_patch_ops(hidden.clone(), [("position_add", [1], delta)])
    assert torch.equal(out[0], hidden[0])
    assert torch.equal(out[2], hidden[2])
    assert torch.equal(out[1], hidden[1] + delta[0])


def test_resid_position_add_supports_batched_rows():
    hidden = torch.zeros(2, 3, 4)
    delta = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])
    out = _apply_resid_patch_ops(hidden, [("position_add", [2], delta)])
    assert torch.equal(out[:, 2], delta[:, 0])
    assert torch.equal(out[:, :2], torch.zeros(2, 2, 4))


def test_resid_patch_unknown_op_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown resid patch op"):
        _apply_resid_patch_ops(torch.ones(2, 4), [("nope", torch.ones(4), None)])


# --------------------------------------------------------------------------- capabilities


def test_capabilities_gaps_lists_missing():
    caps = EngineCapabilities(head_patch=False, raw_model=True, exact_reference=True)
    gaps = caps.gaps()
    assert "head_patch" in gaps
    assert "raw_model" not in gaps and "logits" not in gaps


def test_base_engine_default_capabilities_follow_supports_batch():
    eng = BaseEngine()
    caps = eng.capabilities()
    assert caps.logits and caps.mlp_acts
    assert not caps.logits_batch
    eng.supports_batch = True
    assert eng.capabilities().logits_batch


# --------------------------------------------------------------------------- batch fallback


class DummyPatchedEngine(BaseEngine):
    """Minimal engine: forward_patched echoes which kwargs it saw, per row."""

    backend = "dummy"
    n_layer = 2

    def forward_patched(
        self,
        ids,
        *,
        patch_ops_by_layer=None,
        selected_maps=None,
        collect_acts=False,
        head_patch_ops_by_layer=None,
        resid_patch_ops_by_layer=None,
    ):
        logits = torch.zeros(len(ids), 3)
        acts = [torch.ones(len(ids), 4)] if collect_acts else []
        captured = {0: torch.tensor([float(len(ids))])} if selected_maps else {}
        self.last_head_ops = head_patch_ops_by_layer
        self.last_resid_ops = resid_patch_ops_by_layer
        return logits, acts, captured


def test_forward_patched_batch_fallback_loops_scalar():
    eng = DummyPatchedEngine()
    rows = eng.forward_patched_batch(
        [np.arange(3), np.arange(5)],
        patch_ops_by_layer={0: [("zero", [1], None)]},
        collect_acts=True,
    )
    assert len(rows) == 2
    assert rows[0][0].shape == (3, 3) and rows[1][0].shape == (5, 3)
    assert rows[0][1] and rows[0][1][0].shape == (3, 4)
    assert eng.last_head_ops is None  # not passed through when absent


def test_forward_patched_batch_fallback_passes_head_ops():
    eng = DummyPatchedEngine()
    head_ops = {1: [("zero", [0], None)]}
    eng.forward_patched_batch([np.arange(2)], head_patch_ops_by_layer=head_ops)
    assert eng.last_head_ops == head_ops


def test_forward_patched_batch_fallback_passes_resid_ops():
    eng = DummyPatchedEngine()
    resid_ops = {1: [("proj_remove", np.asarray([1.0, 0.0]), None)]}
    eng.forward_patched_batch([np.arange(2)], resid_patch_ops_by_layer=resid_ops)
    assert eng.last_resid_ops == resid_ops


def test_paged_residual_batch_surfaces_dispatch_and_slice(monkeypatch):
    from mrun.engine import paged as paged_module
    from mrun.engine.paged import PagedEngine

    calls = []

    def fake_batched(_store, ids_list, **kwargs):
        calls.append(kwargs)
        lengths = np.asarray([len(row) for row in ids_list], dtype=np.int64)
        batch, maximum = len(ids_list), int(lengths.max())
        if kwargs.get("collect_hidden_states"):
            states = [torch.full((batch, maximum, 2), float(index)) for index in range(4)]
            return (
                torch.zeros(batch, 2),
                lengths,
                {
                    "acts": [],
                    "captured_selected": {},
                    "head_out": [],
                    "hidden_states": states,
                },
            )
        logits = torch.arange(batch * maximum * 3, dtype=torch.float32).view(batch, maximum, 3)
        return (
            logits,
            lengths,
            {
                "acts": [],
                "captured_selected": {},
                "head_out": [],
                "hidden_states": [],
            },
        )

    monkeypatch.setattr(paged_module.pf, "batched_paged_logits", fake_batched)
    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = object()
    engine.n_layer = 3
    engine.hidden = 2
    rows = [np.arange(2), np.arange(4)]

    states = engine.hidden_states_batch(rows)
    assert [tuple(row[0].shape) for row in states] == [(2, 2), (4, 2)]
    assert len(states[0]) == engine.n_layer + 1
    assert calls[0]["return_hidden"] and calls[0]["collect_hidden_states"]

    resid_ops = {1: [("proj_remove", np.asarray([1.0, 0.0]), None)]}
    patched = engine.forward_patched_batch(rows, resid_patch_ops_by_layer=resid_ops)
    assert [tuple(output[0].shape) for output in patched] == [(2, 3), (4, 3)]
    assert calls[1]["resid_patch_ops_by_layer"] is resid_ops


# --------------------------------------------------------------------------- model-gated
import os  # noqa: E402

import pytest  # noqa: E402

requires_model = pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS"))
    != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)


@requires_model
def test_hf_gapfill_batched_matches_scalar_and_taps_work():
    """qwen2.5-0.5b: forward_patched_batch == scalar per row; attns + resid taps have the
    documented shapes; capabilities report the HF contract."""
    from mrun.engine import open_engine

    eng = open_engine("qwen2.5-0.5b", backend="hf")
    try:
        caps = eng.capabilities()
        assert caps.raw_model and caps.exact_reference and not caps.head_patch
        assert eng.arch == "qwen2"

        ids_list = eng.encode(["The capital of France is", "A reliable experiment should"])
        ops = {0: [("zero", [1, 2, 3], None)]}

        batched = eng.forward_patched_batch(ids_list, patch_ops_by_layer=ops, collect_acts=True)
        for ids, (blogits, bacts, _) in zip(ids_list, batched, strict=True):
            slogits, sacts, _ = eng.forward_patched(ids, patch_ops_by_layer=ops, collect_acts=True)
            assert torch.allclose(blogits, slogits, atol=1e-4), "batched != scalar logits"
            assert len(bacts) == eng.n_layer
            assert torch.allclose(bacts[0].float(), sacts[0].float(), atol=1e-4)

        logits, attns = eng.forward_attns(ids_list[0])
        T = len(ids_list[0])
        assert len(attns) == eng.n_layer and attns[0].shape[-2:] == (T, T)

        logits2, acts, resid = eng.forward_acts_resid(ids_list[0])
        assert resid.shape == (T, eng.hidden)
        assert len(acts) == eng.n_layer
        # pre-final-norm: resid must differ from the post-norm hidden the logits come from
        assert torch.allclose(logits2, eng.logits(ids_list[0]), atol=1e-4)
    finally:
        eng.close()


@requires_model
def test_paged_gapfill_resid_and_batched_patch_parity():
    """Qwen2.5-0.5B paged: forward_acts_resid returns the pre-final-norm residual;
    fused forward_patched_batch == scalar forward_patched per row (argmax + tolerance)."""
    from mrun.models import store_name
    from mrun.paths import stores_root

    model = "qwen2.5-0.5b"
    if not (stores_root() / store_name(model) / "manifest.json").exists():
        pytest.skip(f"no paged store for {model}")
    from mrun.engine import open_engine

    eng = open_engine(model, backend="paged")
    try:
        caps = eng.capabilities()
        assert caps.head_patch and caps.residual_tap and caps.residual_tap_batch
        assert caps.approximate_quantized

        ids_list = eng.encode(["The capital of France is", "In a short proof, the key idea is"])
        logits, acts, resid = eng.forward_acts_resid(ids_list[0])
        assert resid.shape == (len(ids_list[0]), eng.hidden)

        hidden_batch = eng.hidden_states_batch(ids_list)
        for ids, batch_states in zip(ids_list, hidden_batch, strict=True):
            scalar_states = eng.hidden_states(ids)
            assert len(batch_states) == len(scalar_states) == eng.n_layer + 1
            for batch_state, scalar_state in zip(batch_states, scalar_states, strict=True):
                assert batch_state.shape == scalar_state.shape
                assert torch.allclose(batch_state, scalar_state, atol=5e-3, rtol=1e-3)

        ops = {2: [("zero", [5, 6], None)]}
        head_ops = {1: [("zero", [0], None)]}
        batched = eng.forward_patched_batch(
            ids_list, patch_ops_by_layer=ops, head_patch_ops_by_layer=head_ops, collect_acts=True
        )
        for ids, (blogits, bacts, _) in zip(ids_list, batched, strict=True):
            slogits, sacts, _ = eng.forward_patched(
                ids, patch_ops_by_layer=ops, head_patch_ops_by_layer=head_ops, collect_acts=True
            )
            assert torch.equal(blogits.argmax(-1), slogits.argmax(-1)), "argmax drift"
            assert float((blogits - slogits).abs().max()) < 5e-3
            assert torch.allclose(bacts[2].float(), sacts[2].float(), atol=5e-3)

        direction = hidden_batch[0][3][-1].float().numpy()
        resid_ops = {2: [("proj_remove", direction, None)]}
        clean_rows = eng.logits_batch(ids_list)
        batched_resid = eng.forward_patched_batch(
            ids_list,
            resid_patch_ops_by_layer=resid_ops,
        )
        for ids, clean, (batch_logits, _, _) in zip(
            ids_list, clean_rows, batched_resid, strict=True
        ):
            scalar_logits, _, _ = eng.forward_patched(
                ids,
                resid_patch_ops_by_layer=resid_ops,
            )
            assert torch.equal(batch_logits.argmax(-1), scalar_logits.argmax(-1))
            assert float((batch_logits - scalar_logits).abs().max()) < 5e-3
            assert float((batch_logits - clean).abs().max()) > 1e-3
    finally:
        eng.close()


class DummyNoHeadEngine(BaseEngine):
    """forward_patched WITHOUT the head kwarg — like the MLX engines."""

    backend = "nohead"

    def forward_patched(
        self, ids, *, patch_ops_by_layer=None, selected_maps=None, collect_acts=False
    ):
        return torch.zeros(len(ids), 2), [], {}


def test_forward_patched_batch_fallback_raises_typed_on_missing_head_support():
    """F7: a head-patch request against an engine with no head tap must raise
    NotImplementedError (typed), never TypeError from an unexpected kwarg."""
    import pytest

    eng = DummyNoHeadEngine()
    with pytest.raises(NotImplementedError, match="head-output patches"):
        eng.forward_patched_batch(
            [np.arange(2)], head_patch_ops_by_layer={0: [("zero", [0], None)]}
        )


def test_mlx_capabilities_report_patch_support():
    """F3: MLX engines implement MLP patch + selected capture — capabilities must say so."""
    from mrun.engine.mlx import _MLXInterventionBase

    eng = _MLXInterventionBase.__new__(_MLXInterventionBase)
    caps = eng.capabilities()
    assert caps.mlp_patch and caps.selected_capture
    assert not caps.head_patch and not caps.raw_model and caps.approximate_quantized


def test_resolver_moe_and_falcon_families():
    """Fidelity audit #1: olmoe/qwen3_moe -> moe (with per-expert weights accessor),
    falcon_mamba -> mamba."""
    from mrun import arch

    def fake(mt):
        return type("M", (), {"config": type("C", (), {"model_type": mt})()})()

    assert arch.Resolver(fake("olmoe")).fam == "moe"
    assert arch.Resolver(fake("qwen3_moe")).fam == "moe"
    assert arch.Resolver(fake("falcon_mamba")).fam == "mamba"
    import pytest

    with pytest.raises(RuntimeError, match="moe_expert_down_weights"):
        arch.Resolver(fake("olmoe")).down_weight(object())


def test_write_norm_survives_bf16():
    """Fidelity audit #2: the .float()-before-.numpy() bf16 guard (numpy chokes on bf16)."""
    import torch.nn as nn

    from mrun import arch

    class Blk:
        def __init__(self):
            self.mlp = type("m", (), {"down_proj": nn.Linear(4, 8).to(torch.bfloat16)})()

    class Cfg:
        model_type = "qwen2"
        intermediate_size = 4
        hidden_size = 8

    class Mdl:
        config = Cfg()
        model = type("mm", (), {"layers": [Blk()]})()

    out = arch.Resolver(Mdl()).write_norm(Mdl(), {"intermediate": 4})
    assert out.shape == (4,) and np.isfinite(out).all()


def test_resolve_model_ambiguous_label_raises():
    """Fidelity audit #4: colliding labels must not silently pick a wrong model."""
    import pytest

    from mrun.models import resolve_model

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model("8b")
    assert resolve_model("qwen2.5-0.5b").name == "qwen2.5-0.5b"
