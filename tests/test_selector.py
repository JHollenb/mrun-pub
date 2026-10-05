from __future__ import annotations

from mrun.policy import HostCaps, plan_run
from mrun.selector import select_run_plan

BEAST = HostCaps(
    name="beast",
    ram_mb=61_000,
    vram_mb=16_376,
    has_cuda=True,
    cpus=32,
)


def _artifact(
    model: str,
    artifact_id: str,
    artifact_kind: str,
    *,
    bytes: int = 1_000_000_000,
    path: str = "/mnt/big/models/artifact",
    **extra,
) -> dict:
    return {
        "model": model,
        "kind": "weights" if artifact_kind == "hf-weights" else "qstore",
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "bytes": bytes,
        "path": path,
        **extra,
    }


def test_small_model_prefers_reference_hf_artifact() -> None:
    result = select_run_plan(
        "qwen2.5-0.5b",
        host=BEAST,
        artifacts=[
            _artifact("qwen2.5-0.5b", "hf:small", "hf-weights"),
            _artifact("qwen2.5-0.5b", "qstore:small", "qstore"),
        ],
    )

    assert result.used_fallback is False
    assert result.plan.backend == "hf"
    assert result.plan.device == "cuda"
    assert result.plan.artifact_id == "hf:small"
    assert result.plan.engine_profile == "hf-reference"
    assert result.candidates[0].profile_id == "hf-reference"


def test_base_model_does_not_bind_instruct_sibling_artifact() -> None:
    result = select_run_plan(
        "qwen2.5-0.5b",
        host=BEAST,
        artifacts=[
            _artifact(
                "qwen2.5-0.5b-instruct",
                "hf:instruct",
                "hf-weights",
                path="/mnt/big/llm-models/Qwen2.5-0.5B-Instruct",
            )
        ],
    )

    assert result.used_fallback is True
    assert result.plan.artifact_id is None
    assert "no matching local artifact" in result.reason


def test_cuda_vram_overflow_selects_a_paged_artifact() -> None:
    result = select_run_plan(
        "qwen2.5-7b",
        host=BEAST,
        artifacts=[
            _artifact("qwen2.5-7b", "hf:7b", "hf-weights", bytes=14_000_000_000),
            _artifact("qwen2.5-7b", "qstore:7b", "qstore", bytes=4_000_000_000),
        ],
    )

    assert result.plan.backend == "paged"
    assert result.plan.device == "cuda"
    assert result.plan.artifact_id == "qstore:7b"
    assert result.plan.engine_kwargs()["store_path"] == "/mnt/big/models/artifact"
    assert any("selector selected" in reason for reason in result.plan.reasons)


def test_large_qwen35_paged_artifact_uses_bounded_auto_policy_estimate() -> None:
    result = select_run_plan(
        "qwen3.8-27b",
        host=BEAST,
        artifacts=[
            _artifact(
                "qwen3.8-27b",
                "qstore:qwen38",
                "qstore",
                bytes=27_000_000_000,
                path="/mnt/big/qstores/Qwen3.8-27B",
            )
        ],
    )

    assert result.used_fallback is False
    assert result.plan.backend == "paged"
    assert result.plan.artifact_id == "qstore:qwen38"
    assert result.plan.engine_kwargs()["store_path"] == "/mnt/big/qstores/Qwen3.8-27B"


def test_qwen3_moe_overflow_selects_the_routed_expert_profile() -> None:
    result = select_run_plan(
        "qwen3-30b-a3b",
        host=BEAST,
        artifacts=[
            _artifact(
                "Qwen3-30B-A3B-fp8-paged-v1",
                "expert:fp8",
                "expert-store",
                bytes=8_000_000_000,
                codec="rowwise-e4m3fn-paged-v1",
            )
        ],
    )

    assert result.plan.backend == "qwen3-moe-cuda"
    assert result.plan.device == "cuda"
    assert result.plan.dtype == "bfloat16"
    assert result.plan.artifact_id == "expert:fp8"
    assert result.plan.engine_profile == "qwen3-moe-cuda"
    assert result.plan.engine_options["expert_codec"] == "fp8"


def test_qwen3_moe_selector_binds_w4_codec_to_the_selected_store() -> None:
    result = select_run_plan(
        "qwen3-30b-a3b",
        host=BEAST,
        artifacts=[
            _artifact(
                "qwen3-30b-a3b",
                "expert:w4",
                "expert-store",
                bytes=8_000_000_000,
                codec="symmetric-int4-offset8-g128-f32-paged-v1",
            )
        ],
    )

    assert result.plan.engine_options["expert_codec"] == "w4"
    assert result.plan.engine_kwargs()["expert_codec"] == "w4"


def test_explicit_qwen3_moe_backend_binds_exact_requested_codec_and_store() -> None:
    fp8_path = "/mnt/big/qstores/Qwen3-30B-A3B-fp8-paged-v1"
    result = select_run_plan(
        "qwen3-30b-a3b",
        host=BEAST,
        backend="qwen3-moe-cuda",
        dtype="bf16",
        device="cuda",
        seq_lens=[284],
        backend_options={
            "expert_codec": "fp8",
            "store_dir": fp8_path,
            "cache_mb": 9_500,
        },
        artifacts=[
            _artifact(
                "Qwen3-30B-A3B-w4-paged-v1",
                "expert:w4",
                "expert-store",
                bytes=8_000_000_000,
                codec="symmetric-int4-offset8-g128-f32-paged-v1",
                path="/mnt/big/qstores/Qwen3-30B-A3B-w4-paged-v1",
            ),
            _artifact(
                "Qwen3-30B-A3B-fp8-paged-v1",
                "expert:fp8",
                "expert-store",
                bytes=16_000_000_000,
                codec="rowwise-e4m3fn-paged-v1",
                path=fp8_path,
            ),
        ],
    )

    assert result.used_fallback is False
    assert result.plan.artifact_id == "expert:fp8"
    assert result.plan.artifact_locator["path"] == fp8_path
    assert result.plan.engine_options["expert_codec"] == "fp8"
    assert result.plan.engine_kwargs()["store_dir"] == fp8_path


def test_measured_evidence_can_override_profile_prior() -> None:
    result = select_run_plan(
        "qwen2.5-0.5b",
        host=BEAST,
        artifacts=[
            _artifact("qwen2.5-0.5b", "hf:small", "hf-weights"),
            _artifact("qwen2.5-0.5b", "qstore:small", "qstore"),
        ],
        evidence={
            "paged-int8": {"throughput_tok_s": 1_000.0, "parity_valid": True},
            "hf-reference": {"throughput_tok_s": 100.0, "parity_valid": True},
        },
    )

    assert result.plan.backend == "paged"
    assert result.plan.artifact_id == "qstore:small"
    assert result.candidates[0].reasons[0] == "measured throughput=1000 tok/s"


def test_missing_inventory_uses_legacy_policy_without_binding_an_artifact() -> None:
    expected = plan_run("qwen2.5-0.5b", host=BEAST)
    result = select_run_plan("qwen2.5-0.5b", host=BEAST, artifacts=None)

    assert result.used_fallback is True
    assert result.plan.backend == expected.backend
    assert result.plan.device == expected.device
    assert result.plan.artifact_id is None
    assert "artifact inventory unavailable" in result.reason


def test_legacy_warm_kind_rows_are_not_treated_as_exact_artifacts() -> None:
    result = select_run_plan(
        "qwen2.5-0.5b",
        host=BEAST,
        artifacts=[
            {
                "model": "qwen2.5-0.5b",
                "kind": "weights",
                "bytes": 1_000_000_000,
            }
        ],
    )

    assert result.used_fallback is True
    assert result.plan.artifact_id is None
    assert "no matching local artifact" in result.reason


def test_known_inventory_without_a_usable_store_uses_safe_cpu_hf_fallback() -> None:
    result = select_run_plan(
        "qwen2.5-7b",
        host=BEAST,
        artifacts=[_artifact("qwen2.5-7b", "hf:7b", "hf-weights")],
    )

    assert result.used_fallback is False
    assert result.plan.backend == "hf"
    assert result.plan.device == "cpu"
    assert result.plan.artifact_id == "hf:7b"
    assert result.plan.engine_kwargs()["device"] == "cpu"


def test_oversized_hf_artifact_does_not_block_paged_legacy_fallback() -> None:
    result = select_run_plan(
        "qwen3-30b-a3b",
        host=BEAST,
        artifacts=[
            _artifact(
                "qwen3-30b-a3b",
                "hf:30b",
                "hf-weights",
                bytes=61_000_000_000,
            )
        ],
    )

    assert result.used_fallback is True
    assert result.plan.backend == "qwen3-moe-cuda"
    assert result.plan.device == "cuda"
    assert result.plan.artifact_id is None
    assert "no eligible artifact" in result.reason or "legacy policy" in result.reason


def test_explicit_paged_bf16_binds_lossless_fp32_store_not_smaller_int8() -> None:
    result = select_run_plan(
        "qwen2.5-0.5b",
        host=BEAST,
        backend="paged-bf16",
        dtype="bf16",
        device="cuda",
        artifacts=[
            _artifact(
                "qwen2.5-0.5b",
                "qstore:int8",
                "qstore",
                bytes=500_000_000,
                path="/mnt/big/qstores/Qwen2.5-0.5B",
                variant="Qwen2.5-0.5B",
            ),
            _artifact(
                "qwen2.5-0.5b",
                "qstore:fp32",
                "qstore",
                bytes=2_000_000_000,
                path="/mnt/big/qstores/Qwen2.5-0.5B-fp32",
                variant="Qwen2.5-0.5B-fp32",
            ),
        ],
    )

    assert result.used_fallback is False
    assert result.plan.backend == "paged-bf16"
    assert result.plan.artifact_id == "qstore:fp32"
    assert result.plan.engine_kwargs()["store_path"].endswith("-fp32")


def test_explicit_lossless_pager_fails_closed_without_fp32_store() -> None:
    result = select_run_plan(
        "qwen2.5-0.5b",
        host=BEAST,
        backend="paged-bf16",
        dtype="bf16",
        device="cuda",
        artifacts=[
            _artifact(
                "qwen2.5-0.5b",
                "qstore:int8",
                "qstore",
                path="/mnt/big/qstores/Qwen2.5-0.5B",
            )
        ],
    )

    assert result.used_fallback is True
    assert result.plan.artifact_id is None
    assert "no matching local artifact" in result.reason


def test_explicit_int8_pager_does_not_bind_lossless_store() -> None:
    result = select_run_plan(
        "qwen2.5-0.5b",
        host=BEAST,
        backend="paged",
        dtype="bf16",
        device="cuda",
        artifacts=[
            _artifact(
                "qwen2.5-0.5b",
                "qstore:fp32",
                "qstore",
                bytes=100_000_000,
                path="/mnt/big/qstores/Qwen2.5-0.5B-fp32",
            ),
            _artifact(
                "qwen2.5-0.5b",
                "qstore:int8",
                "qstore",
                bytes=500_000_000,
                path="/mnt/big/qstores/Qwen2.5-0.5B",
            ),
        ],
    )

    assert result.used_fallback is False
    assert result.plan.backend == "paged"
    assert result.plan.artifact_id == "qstore:int8"
