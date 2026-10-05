"""Pure policy tests — no model loads, canned HostCaps."""

from __future__ import annotations

import os

import pytest

from mrun.policy import HostCaps, plan_run, resolve_max_batch

MAC = HostCaps(name="mbp1", ram_mb=36_000, has_mps=True, has_ane=True, cpus=12)
BEAST = HostCaps(name="beast", ram_mb=61_000, vram_mb=16_376, has_cuda=True, cpus=32)
TINY = HostCaps(name="tiny", ram_mb=8_000, cpus=4)


def test_mac_dtype_is_task_and_ram_aware():
    # P0 (2026-07-15) made macs default bf16 to stay out of swap. AMENDED 2026-07-24
    # (G5 arm E, measured): torch-cpu-bf16 scoring is 4.7x slower than fp32/Accelerate
    # (99.3s vs 21.0s on qwen2.5-0.5b) — so non-train tasks pick fp32 when the fp32
    # footprint fits comfortably; bf16 stays for train and RAM-tight models.
    plan = plan_run("qwen2.5-0.5b", host=MAC)
    assert plan.dtype == "float32"
    assert plan.backend == "hf"
    assert plan.device == "cpu"
    assert plan.max_batch == 16
    assert plan.ram_limit_mb > plan.est_ram_mb
    assert any("4.7x regression" in r for r in plan.reasons)
    assert plan_run("qwen2.5-0.5b", host=MAC, task="train").dtype == "bfloat16"
    big = plan_run("qwen2.5-7b", host=MAC)  # fp32 ~28GB > 0.35 * 18GB -> stays bf16
    assert big.dtype == "bfloat16"
    assert any("unified-memory mac default" in r for r in big.reasons)


def test_cuda_host_defaults_bf16_tensor_core():
    # Phase 1 (2026-07-15, user-approved): cuda hosts default bf16 — fp32-noTF32 left the
    # 4080's tensor cores idle. fp32 stays the oracle via env/dtype opt-out.
    plan = plan_run("qwen2.5-0.5b", host=BEAST)
    assert plan.dtype == "bfloat16"
    assert any("tensor-core default" in r for r in plan.reasons)


def test_cuda_fp32_env_opt_out(monkeypatch):
    monkeypatch.setenv("MRUN_TORCH_DTYPE", "fp32")
    assert plan_run("qwen2.5-0.5b", host=BEAST).dtype == "float32"


def test_cuda_train_stays_fp32_until_gauntlet_gated():
    plan = plan_run("qwen2.5-0.5b", "train", host=BEAST)
    assert plan.dtype == "float32"
    assert any("acquisition-gauntlet" in r for r in plan.reasons)


def test_vram_oom_guard_routes_to_cuda_paged():
    # 7B bf16 ≈ 15.4GB weights: fits beast's 61GB RAM budget (so resident HF is otherwise
    # selected) but NOT the 16GB card with activations. Route to bounded CUDA paging instead
    # of loading the full model on the card or silently moving the HF model to CPU.
    plan = plan_run("qwen2.5-7b", host=BEAST)
    assert plan.backend == "paged"
    assert plan.device == "cuda"
    assert 0 < plan.est_vram_mb < BEAST.vram_mb
    assert plan.engine_kwargs() == {"device": "cuda", "compute_dtype": "bf16"}
    assert any("overflow routes to paged execution" in r for r in plan.reasons)


def test_qwen3_moe_vram_overflow_routes_to_paged_expert_profile():
    plan = plan_run("qwen3-30b-a3b", host=BEAST)
    assert plan.backend == "qwen3-moe-cuda"
    assert plan.device == "cuda"
    assert plan.est_vram_mb < BEAST.vram_mb * 0.85
    assert any("paged artifact/profile selected" in r for r in plan.reasons)


def test_small_model_fits_vram_stays_cuda():
    plan = plan_run("qwen2.5-0.5b", host=BEAST)
    assert plan.device == "cuda"
    assert 0 < plan.est_vram_mb < BEAST.vram_mb


def test_mac_fp32_env_opt_out(monkeypatch):
    monkeypatch.setenv("MRUN_TORCH_DTYPE", "fp32")
    assert plan_run("qwen2.5-0.5b", host=MAC).dtype == "float32"


def test_explicit_dtype_wins():
    assert plan_run("qwen2.5-0.5b", host=MAC, dtype="bf16").dtype == "bfloat16"
    assert plan_run("qwen2.5-0.5b", host=MAC, dtype="fp32").dtype == "float32"
    assert plan_run("qwen2.5-0.5b", host=BEAST, dtype="bf16").dtype == "bfloat16"


def test_explicit_mps_big_model_downgrades_to_cpu():
    # Metal memory is invisible to the RSS guard — a big explicit-mps run is downgraded.
    plan = plan_run(
        "qwen2.5-1.5b",
        host=HostCaps(name="mbp1", ram_mb=12_000, has_mps=True, has_ane=True, cpus=12),
        device="mps",
        dtype="fp32",
    )
    assert plan.device == "cpu"
    assert any("unified-memory guard" in r for r in plan.reasons)


def test_bf16_env_opt_in(monkeypatch):
    monkeypatch.setenv("MRUN_TORCH_DTYPE", "bf16")
    plan = plan_run("qwen2.5-0.5b", host=MAC)
    assert plan.dtype == "bfloat16"
    assert any("env opt-in" in r for r in plan.reasons)


def test_big_model_small_host_goes_paged():
    plan = plan_run("qwen3-14b", host=TINY)
    assert plan.backend == "paged"
    assert plan.device == "cpu"
    assert plan.ram_limit_mb < TINY.ram_mb  # streamed working set, not 14B fp32 weights
    assert any("paged" in r for r in plan.reasons)


def test_big_model_big_host_stays_hf():
    plan = plan_run("qwen2.5-1.5b", host=BEAST)
    assert plan.backend == "hf"
    assert plan.device == "cuda"
    assert any("TF32" in r for r in plan.reasons)


def test_explicit_backend_recorded():
    plan = plan_run("qwen2.5-0.5b", host=MAC, backend="mlx")
    assert plan.backend == "mlx"
    assert plan.device == "mps"
    assert any("parity-gate" in r for r in plan.reasons)


def test_apple_alias_is_canonical_mlx_policy():
    plan = plan_run("qwen2.5-0.5b", host=MAC, backend="apple")
    assert plan.backend == "mlx"
    assert plan.device == "mps"
    assert any("canonicalized" in reason for reason in plan.reasons)


def test_coreml_policy_is_fp16_demoted_and_apple_only():
    plan = plan_run("qwen2.5-0.5b", host=MAC, backend="coreml")
    assert (plan.backend, plan.device, plan.dtype) == ("ane", "mps", "float16")
    assert any("DEMOTED" in reason for reason in plan.reasons)
    assert any("margins below 0.5" in reason for reason in plan.reasons)
    with pytest.raises(ValueError, match="Apple Silicon"):
        plan_run("qwen2.5-0.5b", host=BEAST, backend="ane")


def test_apple_inference_backends_reject_training():
    with pytest.raises(ValueError, match="training are not implemented"):
        plan_run("qwen2.5-0.5b", task="train", host=MAC, backend="mlx")


def test_explicit_dense_cuda_plan_binds_engine_device_and_bf16():
    plan = plan_run(
        "qwen2.5-0.5b",
        host=BEAST,
        backend="dense-qstore-cuda",
    )
    assert plan.device == "cuda"
    assert plan.dtype == "bfloat16"
    assert plan.engine_kwargs() == {
        "device": "cuda",
        "compute_dtype": "bf16",
    }


def test_explicit_dense_cuda_plan_maps_float16_to_fp16():
    plan = plan_run(
        "qwen2.5-0.5b",
        host=BEAST,
        backend="dense-qstore-cuda",
        dtype="fp16",
    )
    assert plan.dtype == "float16"
    assert plan.engine_kwargs()["compute_dtype"] == "fp16"


def test_explicit_dense_cuda_plan_requires_cuda_host_and_device():
    with pytest.raises(ValueError, match="requires a CUDA host"):
        plan_run(
            "qwen2.5-0.5b",
            host=MAC,
            backend="dense-qstore-cuda",
        )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        plan_run(
            "qwen2.5-0.5b",
            host=BEAST,
            backend="dense-qstore-cuda",
            device="cpu",
        )


def test_explicit_dense_cuda_plan_rejects_fp32():
    with pytest.raises(ValueError, match="requires bfloat16 or float16"):
        plan_run(
            "qwen2.5-0.5b",
            host=BEAST,
            backend="dense-qstore-cuda",
            dtype="fp32",
        )


def test_explicit_qwen3_moe_cuda_plan_uses_measured_envelope():
    plan = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        backend="qwen3-moe-cuda",
    )
    assert plan.backend == "qwen3-moe-cuda"
    assert plan.device == "cuda"
    assert plan.dtype == "bfloat16"
    assert plan.max_batch == 8
    assert plan.est_ram_mb == 6144.0
    assert plan.est_vram_mb == 11_296.8
    assert plan.ram_limit_mb == 7987.2
    assert plan.engine_kwargs() == {
        "device": "cuda",
        "compute_dtype": "bf16",
    }
    assert any("profile-gated" in reason for reason in plan.reasons)


def test_qwen3_moe_cache_option_changes_plan_and_engine_kwargs():
    plan = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        backend="qwen3-moe-cuda",
        backend_options={"cache_mb": 7000, "max_active_pages": 128},
    )
    assert plan.est_vram_mb == 11_196.8
    assert plan.engine_options == {
        "cache_mb": 7000,
        "max_active_pages": 128,
    }
    assert plan.engine_kwargs() == {
        "cache_mb": 7000,
        "max_active_pages": 128,
        "device": "cuda",
        "compute_dtype": "bf16",
    }


def test_qwen3_moe_w4_plan_uses_codec_geometry_and_host_tier_cap():
    plan = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        backend="qwen3-moe-cuda",
        backend_options={
            "expert_codec": "w4",
            "w4_arithmetic_policy": "w4-g128-postscale-bf16-v1",
            "host_cache_mb": 30_000,
        },
    )
    # W4 pages are 2.506752 MB: one 128-page miss slab, not the removed compact slab.
    assert plan.est_vram_mb == 11_011.6
    # The requested host tier is capped to the 6,144-page W4 store (15,401.5 MB),
    # plus the measured 6,144MB routed-runtime host baseline.
    assert plan.est_ram_mb == 21_545.5
    assert plan.ram_limit_mb == 28_009.2
    assert plan.engine_options == {
        "expert_codec": "w4",
        "w4_arithmetic_policy": "w4-g128-postscale-bf16-v1",
        "host_cache_mb": 30_000,
    }
    assert any("codec=w4" in reason for reason in plan.reasons)
    assert any("effective expert tier=15401.5MB" in reason for reason in plan.reasons)


def test_qwen3_moe_compact_and_prefetch_slabs_are_charged():
    plan = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        backend="qwen3-moe-cuda",
        backend_options={
            "expert_codec": "w4",
            "page_binding_policy": "compact-copy-v1",
            "route_prefetch": True,
        },
    )
    assert plan.est_vram_mb == 11_653.3
    assert any("device_staging_slabs=3" in reason for reason in plan.reasons)


def test_qwen3_moe_policy_rejects_unknown_codec_and_binding():
    with pytest.raises(ValueError, match="expert_codec"):
        plan_run(
            "qwen3-30b-a3b",
            host=BEAST,
            backend="qwen3-moe-cuda",
            backend_options={"expert_codec": "nf4"},
        )
    with pytest.raises(ValueError, match="page_binding_policy"):
        plan_run(
            "qwen3-30b-a3b",
            host=BEAST,
            backend="qwen3-moe-cuda",
            backend_options={"page_binding_policy": "mystery"},
        )
    with pytest.raises(ValueError, match="w4_arithmetic_policy must be one of"):
        plan_run(
            "qwen3-30b-a3b",
            host=BEAST,
            backend="qwen3-moe-cuda",
            backend_options={
                "expert_codec": "w4",
                "w4_arithmetic_policy": "postscale-unversioned",
            },
        )
    with pytest.raises(ValueError, match="requires expert_codec='w4'"):
        plan_run(
            "qwen3-30b-a3b",
            host=BEAST,
            backend="qwen3-moe-cuda",
            backend_options={"w4_arithmetic_policy": "w4-g128-postscale-bf16-v1"},
        )


def test_qwen3_moe_policy_alias_canonicalizes_backend():
    plan = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        backend="moe_qstore_cuda",
    )
    assert plan.backend == "qwen3-moe-cuda"


def test_explicit_qwen3_moe_cuda_plan_rejects_unsupported_modes():
    with pytest.raises(ValueError, match="requires a CUDA host"):
        plan_run(
            "qwen3-30b-a3b",
            host=MAC,
            backend="qwen3-moe-cuda",
        )
    with pytest.raises(ValueError, match="requires bfloat16"):
        plan_run(
            "qwen3-30b-a3b",
            host=BEAST,
            backend="qwen3-moe-cuda",
            dtype="fp32",
        )
    with pytest.raises(ValueError, match="training are not implemented"):
        plan_run(
            "qwen3-30b-a3b",
            "train",
            host=BEAST,
            backend="qwen3-moe-cuda",
            dtype="bf16",
        )


def test_qwen3_moe_default_cache_shrinks_to_fit_smaller_gpu():
    small_gpu = HostCaps(
        name="small-gpu",
        ram_mb=32_000,
        vram_mb=12_000,
        has_cuda=True,
        cpus=16,
    )
    plan = plan_run(
        "qwen3-30b-a3b",
        host=small_gpu,
        backend="qwen3-moe-cuda",
    )
    assert plan.est_vram_mb == 10_200.0
    assert plan.engine_options == {"cache_mb": 6003.2}


def test_qwen3_moe_context_capacity_is_charged_and_can_fail_before_cuda_oom():
    short = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        seq_lens=[128],
        backend="qwen3-moe-cuda",
    )
    long = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        seq_lens=[8192],
        backend="qwen3-moe-cuda",
    )
    assert short.est_vram_mb == 10_994.8
    assert short.engine_options == {}
    assert long.est_vram_mb == pytest.approx(BEAST.vram_mb * 0.85, abs=0.1)
    assert long.engine_options == {"cache_mb": 3683.0}
    assert any("capacity=8192" in reason for reason in long.reasons)

    with pytest.raises(ValueError, match="batch/concurrency, or context"):
        plan_run(
            "qwen3-30b-a3b",
            host=BEAST,
            seq_lens=[16_384],
            backend="qwen3-moe-cuda",
            backend_options={"cache_mb": 7100},
        )


def test_qwen3_moe_w4_keeps_more_expert_cache_at_b8_c4096():
    fp8 = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        seq_lens=[4096],
        backend="qwen3-moe-cuda",
    )
    w4 = plan_run(
        "qwen3-30b-a3b",
        host=BEAST,
        seq_lens=[4096],
        backend="qwen3-moe-cuda",
        backend_options={"expert_codec": "w4"},
    )

    assert fp8.engine_options == {"cache_mb": 6904.2}
    assert w4.engine_options == {"expert_codec": "w4"}
    assert fp8.est_vram_mb == 13_919.5
    assert w4.est_vram_mb == 13_830.1


def test_cuda_never_exceeds_activation_cap():
    # long sequences at hidden=2048: 64*Tmax*hidden*4 > 4GiB -> cap stays 16
    long = [40_000]
    assert resolve_max_batch("cuda", long, 4096) == 16
    assert resolve_max_batch("cuda", [512], 2048) == 64
    assert resolve_max_batch("cpu", [512], 2048) == 16


def test_gather_max_batch_env_override(monkeypatch):
    monkeypatch.setenv("GATHER_MAX_BATCH", "8")
    assert resolve_max_batch("cuda", [512], 2048) == 8


def test_threads_capped_by_host():
    plan = plan_run("distilgpt2", host=HostCaps(name="mini", ram_mb=16_000, cpus=2))
    assert plan.threads == 2


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("MRUN_TORCH_DTYPE", "PANR_TORCH_DTYPE", "GATHER_MAX_BATCH"):
        if var in os.environ:
            monkeypatch.delenv(var)


def test_mac_score_fast_path_fp16_mps():
    # 2026-07-24 G5: task="score" on a mac defaults to fp16-mps-b64 (content-lift gated,
    # delta +0.0003+/-0.0002 vs fp32; 6x scoring with prefix-KV). Scope is score-only:
    # forward/canonical stays fp32-cpu, train stays bf16, RAM-tight models fall through.
    plan = plan_run("qwen2.5-0.5b", task="score", host=MAC)
    assert (plan.dtype, plan.device, plan.max_batch) == ("float16", "mps", 64)
    assert any("mac score fast path" in r for r in plan.reasons)
    assert plan_run("qwen2.5-0.5b", task="forward", host=MAC).device == "cpu"
    assert plan_run("qwen2.5-0.5b", task="score", host=MAC, dtype="fp32").dtype == "float32"
    assert plan_run("qwen2.5-7b", task="score", host=MAC).device == "cpu"


def test_workload_decode_selects_mlx_q4_on_metal_host():
    # Regime-gated (bandwidth-roofline PoC 2026-07-23/24, measured): int4 wins B=1 decode
    # x2.34 but INVERTS on batched/prefill — so the branch keys on workload, never host.
    plan = plan_run("qwen2.5-0.5b", host=MAC, workload="decode")
    assert plan.backend == "mlx-q4"
    assert plan.device == "mps"
    assert any("workload=decode" in r for r in plan.reasons)
    # audit trail must NOT claim the backend was an explicit caller override
    assert not any("(explicit)" in r for r in plan.reasons)


def test_workload_none_changes_nothing():
    base = plan_run("qwen2.5-0.5b", host=MAC)
    assert base.backend == "hf"  # unchanged default path


def test_workload_decode_ignored_off_metal_and_on_explicit_oracle():
    # cuda host: decode workload must not hijack the canonical cuda path
    beast_plan = plan_run("qwen2.5-0.5b", host=BEAST, workload="decode")
    assert beast_plan.backend != "mlx-q4"
    # explicit fp32 = oracle intent; the speed path must stand down
    oracle = plan_run("qwen2.5-0.5b", host=MAC, workload="decode", dtype="fp32")
    assert oracle.backend != "mlx-q4"
    # train never rides an inference-only Metal backend
    train = plan_run("qwen2.5-0.5b", host=MAC, workload="decode", task="train")
    assert train.backend != "mlx-q4"


def test_workload_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown workload"):
        plan_run("qwen2.5-0.5b", host=MAC, workload="serving")


def test_workload_decode_7b_fits_metal_via_q4_sizing():
    # The headline case: 7B decode at the bandwidth roofline. A fp32/bf16-based
    # unified-memory estimate refuses it; the int4-g64 sizing (4.5/16 of fp16) admits it.
    plan = plan_run("qwen2.5-7b", host=MAC, workload="decode")
    assert plan.backend == "mlx-q4"
    assert any("int4-g64 weights" in r for r in plan.reasons)


def test_workload_validation_independent_of_mrun_plan(monkeypatch):
    # Validation must not depend on ambient env state; an ignored workload must be audible.
    base = plan_run("qwen2.5-0.5b", host=MAC)
    monkeypatch.setenv("MRUN_PLAN", __import__("json").dumps(base.as_dict()))
    with pytest.raises(ValueError, match="unknown workload"):
        plan_run("qwen2.5-0.5b", host=MAC, workload="serving")
    plan = plan_run("qwen2.5-0.5b", host=MAC, workload="decode")
    assert plan.backend == base.backend  # scheduler plan wins
    assert any("workload=decode ignored" in r for r in plan.reasons)


def test_workload_decode_score_task_keeps_score_fast_path():
    # decode workload + score task is contradictory; score fast path must win un-conflicted
    plan = plan_run("qwen2.5-0.5b", host=MAC, workload="decode", task="score")
    assert plan.backend != "mlx-q4"
