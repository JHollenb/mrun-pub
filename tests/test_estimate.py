"""Tests for the resource estimator (mrun.estimate)."""

import pytest

from mrun import estimate_memory, estimate_resources, load_run_history
from mrun.estimate import (
    _find_config_json,
    _params_from_dims,
    _params_from_label,
    calibrate_memory,
    dtype_bytes,
)
from mrun.io import write_json

# Real config dims; hand-verified param counts (distilgpt2 ~82M, qwen2.5-0.5b ~494M).
DISTILGPT2_DIMS = {
    "n_embd": 768,
    "n_layer": 6,
    "vocab_size": 50257,
    "n_head": 12,
    "model_type": "gpt2",
}
QWEN_05B_DIMS = {
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "vocab_size": 151936,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "tie_word_embeddings": True,
}


def test_params_from_dims_dense_and_gated():
    assert _params_from_dims(DISTILGPT2_DIMS) == pytest.approx(82e6, rel=0.02)
    assert _params_from_dims(QWEN_05B_DIMS) == pytest.approx(494e6, rel=0.02)


def test_params_from_dims_untied_adds_head():
    tied = _params_from_dims(QWEN_05B_DIMS)
    untied = _params_from_dims({**QWEN_05B_DIMS, "tie_word_embeddings": False})
    assert untied - tied == 151936 * 896  # a second vocab x hidden matrix


def test_params_from_dims_counts_all_moe_experts_and_shared_expert():
    dims = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 4,
        "moe_intermediate_size": 3,
        "shared_expert_intermediate_size": 5,
        "tie_word_embeddings": False,
    }
    assert _params_from_dims(dims) == 1520


def test_qwen3_moe_registry_label_uses_total_parameter_count():
    assert _params_from_label("qwen3-30b-a3b") == 30_000_000_000


def test_dtype_bytes():
    assert dtype_bytes("float32") == 4
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("torch.float16") == 2
    assert dtype_bytes(1) == 1
    with pytest.raises(ValueError, match="unknown dtype"):
        dtype_bytes("ternary")


def test_estimate_memory_law_is_weights_plus_overhead():
    est = estimate_memory("distilgpt2", dtype="float32", overhead_mb=400.0)
    # weights_mb is display-rounded to 0.1 MB
    assert est.weights_mb == pytest.approx(est.params * 4 / 1e6, abs=0.1)
    assert est.est_rss_mb == pytest.approx(est.weights_mb + 400.0, abs=0.2)
    # halving the dtype width halves the weight contribution
    half = estimate_memory("distilgpt2", dtype="bf16", overhead_mb=400.0)
    assert half.weights_mb == pytest.approx(est.weights_mb / 2, abs=0.1)


def _cache_config_root(tmp_path):
    hub_root = tmp_path / "hub" / "models--owner--model"
    return hub_root, hub_root / ".no_exist" / "revision" / "config.json"


@pytest.mark.parametrize("content", ["", "{not-json\n"])
def test_find_config_skips_empty_or_malformed_hf_cache_marker(monkeypatch, tmp_path, content):
    hub_root, marker = _cache_config_root(tmp_path)
    marker.parent.mkdir(parents=True)
    marker.write_text(content, encoding="utf-8")
    valid = hub_root / "snapshots" / "revision" / "config.json"
    valid.parent.mkdir(parents=True)
    valid.write_text(
        '{"hidden_size": 2, "num_hidden_layers": 1, "vocab_size": 3, '
        '"num_attention_heads": 1}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("mrun.estimate.default_hub_root", lambda: tmp_path / "hub")
    monkeypatch.setattr("mrun.estimate.models_root", lambda: tmp_path / "models")

    assert _find_config_json("owner/model") == valid
    assert estimate_memory("owner/model").param_source == "config"


def test_find_config_accepts_valid_direct_config(monkeypatch, tmp_path):
    hub_root, _marker = _cache_config_root(tmp_path)
    hub_root.mkdir(parents=True)
    direct = hub_root / "config.json"
    direct.write_text('{"model_type": "toy"}\n', encoding="utf-8")
    monkeypatch.setattr("mrun.estimate.default_hub_root", lambda: tmp_path / "hub")
    monkeypatch.setattr("mrun.estimate.models_root", lambda: tmp_path / "models")

    assert _find_config_json("owner/model") == direct


def _write_manifest(root, experiment, run_id, config, resources):
    write_json(
        root / experiment / run_id / "manifest.json",
        {"experiment": experiment, "run_id": run_id, "config": config, "resources": resources},
    )


def test_load_history_and_calibrate(tmp_path):
    _write_manifest(
        tmp_path, "forward", "aaa",
        {"model": "distilgpt2", "dtype": "float32"},
        {"wall_s": 2.5, "cpu_s": 1.6, "rss_peak_mb": 753.0},
    )
    _write_manifest(
        tmp_path, "dream_retrieval_heads", "bbb",
        {"model": "qwen2.5-0.5b", "dtype": "float32", "seed": 0},
        {"wall_s": 60.6, "cpu_s": 68.4, "rss_peak_mb": 2369.0},
    )
    history = load_run_history(tmp_path)
    assert {r.experiment for r in history} == {"forward", "dream_retrieval_heads"}

    fit = calibrate_memory(history)
    assert fit is not None
    overhead, slope = fit
    assert 0.5 < slope < 1.5  # weights should map ~1:1 to RSS
    assert 200 < overhead < 800  # framework floor


def test_calibrate_needs_two_distinct_sizes(tmp_path):
    _write_manifest(
        tmp_path, "forward", "aaa",
        {"model": "distilgpt2"}, {"rss_peak_mb": 753.0},
    )
    assert calibrate_memory(load_run_history(tmp_path)) is None


def test_estimate_resources_exact_match_uses_observed(tmp_path):
    config = {"model": "distilgpt2", "dtype": "float32", "seed": 0}
    _write_manifest(
        tmp_path, "forward", "aaa", config,
        {"wall_s": 2.5, "cpu_s": 1.6, "rss_peak_mb": 753.0},
    )
    est = estimate_resources(config, name="forward", outputs_root=tmp_path)
    assert est["wall_s_estimate"] == pytest.approx(2.5)
    assert est["cpu_s_estimate"] == pytest.approx(1.6)
    assert est["history"]["nearest"]["differing_keys"] == []


def test_estimate_resources_different_model_does_not_extrapolate_wall(tmp_path):
    _write_manifest(
        tmp_path, "dream_retrieval_heads", "bbb",
        {"model": "qwen2.5-0.5b", "dtype": "float32"},
        {"wall_s": 60.6, "cpu_s": 68.4, "rss_peak_mb": 2369.0},
    )
    est = estimate_resources(
        {"model": "qwen2.5-1.5b", "dtype": "float32"},
        name="dream_retrieval_heads",
        outputs_root=tmp_path,
    )
    # RAM still scales with params; wall/cpu must not be borrowed across models.
    assert est["wall_s_estimate"] is None
    assert est["est_rss_mb"] > 0


# --------------------------------------------------------------- activations (P0)


def test_activation_term_positive_and_grows_with_seq():
    from mrun.estimate import estimate_activation_mb

    small = estimate_activation_mb("qwen2.5-0.5b", seq_lens=[128], batch=16)
    big = estimate_activation_mb("qwen2.5-0.5b", seq_lens=[1024], batch=16)
    assert small > 0
    # attention is T^2: 8x the seq must be much more than 8x the memory
    assert big > small * 8


@pytest.fixture
def cached_qwen_dims(monkeypatch, tmp_path):
    # These tests exercise real dimensions, independently of a developer's model cache.
    config = tmp_path / "config.json"
    write_json(config, QWEN_05B_DIMS)
    monkeypatch.setattr("mrun.estimate._find_config_json", lambda model: config)


def test_recorder_task_holds_all_layer_attention(cached_qwen_dims):
    from mrun.estimate import estimate_activation_mb

    fwd = estimate_activation_mb("qwen2.5-0.5b", seq_lens=[512], batch=16, task="forward")
    rec = estimate_activation_mb("qwen2.5-0.5b", seq_lens=[512], batch=16, task="recorder")
    assert rec > fwd * 5  # 24 layers of attention maps vs 1


def test_count_logits_adds_a_vocab_scaled_term(cached_qwen_dims):
    from mrun.estimate import estimate_activation_mb

    # opt-in logits term: [B,T,vocab] fp32 on a 152k-vocab model is a multi-GB addition, and it is
    # OFF by default (so the calibrated fleet planner is unchanged).
    without = estimate_activation_mb("qwen2.5-0.5b", seq_lens=[64], batch=64, task="forward")
    with_logits = estimate_activation_mb(
        "qwen2.5-0.5b", seq_lens=[64], batch=64, task="forward", count_logits=True
    )
    added = with_logits - without
    expected = 64 * 64 * 151936 * 4 / 1e6  # B*T*vocab*fp32
    assert added == pytest.approx(expected, rel=0.02)
    assert added > 2000  # the logits alone exceed the 0.5B weights (~1GB) — the melt the fix caught


def test_recorder_estimate_covers_measured_peak():
    # Measured 2026-07-15: recorder physiology on qwen2.5-0.5b peaked 6370MB while the
    # old weights+overhead law said 2470MB (the 2x undershoot behind the MBP crash).
    from mrun.estimate import estimate_activation_mb

    weights_overhead = estimate_memory("qwen2.5-0.5b", dtype="float32").est_rss_mb
    act = estimate_activation_mb(
        "qwen2.5-0.5b", seq_lens=[512], batch=16, dtype="float32", task="recorder"
    )
    assert weights_overhead + act >= 6370.0


def test_unknown_model_gets_nontrivial_activation():
    from mrun.estimate import estimate_activation_mb

    assert estimate_activation_mb("no-such-model-xyz", seq_lens=[512], batch=16) > 100.0


def test_task_rss_factor_known_families():
    from mrun.estimate import task_rss_factor

    assert task_rss_factor("forward") == 1.0
    assert task_rss_factor("train") > 1.0
    assert task_rss_factor("nonsense-task") == 1.0
