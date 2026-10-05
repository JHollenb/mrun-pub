from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from mrun.engine.mlx_component import (
    MLX_COMPONENT_BITS,
    MLX_COMPONENT_CODEC,
    MLX_COMPONENT_GROUP_SIZE,
    MLX_COMPONENT_Q4_CODEC,
    MLX_COMPONENT_Q4_NATIVE_SCHEMA,
    MLXComponentArtifactError,
    MLXComponentEngine,
    MLXComponentGraphError,
    MLXComponentMappingError,
    MLXComponentQ4Engine,
    VerifiedComponentGraphReader,
    VerifiedMLXComponentArtifact,
    build_mlx_component_artifact,
    build_mlx_component_q4_artifact,
    canonical_mlx_target,
    dequantize_affine_q8_numpy,
    pack_symmetric_qrow_int8,
    requantize_qrow_affine_q4_mlx,
)

_COMPOSITE_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_mrun_test_composite_qstore_helpers",
    Path(__file__).with_name("test_composite_qstore.py"),
)
assert _COMPOSITE_HELPER_SPEC is not None and _COMPOSITE_HELPER_SPEC.loader is not None
_COMPOSITE_HELPERS = importlib.util.module_from_spec(_COMPOSITE_HELPER_SPEC)
sys.modules[_COMPOSITE_HELPER_SPEC.name] = _COMPOSITE_HELPERS
_COMPOSITE_HELPER_SPEC.loader.exec_module(_COMPOSITE_HELPERS)
_build_component_graph = _COMPOSITE_HELPERS._build_component_graph


DOMAINS_ROOT = Path(__file__).resolve().parents[2]
QWEN_05_GRAPH = (
    DOMAINS_ROOT
    / "experiments"
    / "2026-07-27-142537-disassembled-lexical-runtime-poc"
    / "artifacts"
    / "stage4-model-panel"
    / "qwen2.5-0.5b"
    / "model-graph.json"
)
RUN_MODEL_TESTS = (
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS"))
    == "1"
)


def test_qstore_signed_int8_packs_losslessly_into_mlx_affine_q8() -> None:
    rng = np.random.default_rng(91)
    codes = rng.integers(-128, 128, size=(7, 128), dtype=np.int8)
    codes[0, :4] = np.asarray([-128, -1, 0, 127], dtype=np.int8)
    scales = np.asarray([0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 3.25], dtype=np.float32)

    packed = pack_symmetric_qrow_int8(codes, scales)

    assert packed.weight.dtype == np.uint32
    assert packed.weight.shape == (7, 32)
    assert packed.scales.shape == (7, 2)
    assert packed.biases.shape == (7, 2)
    unpacked_unsigned = np.ascontiguousarray(packed.weight).view(np.uint8).reshape(codes.shape)
    unpacked_codes = np.bitwise_xor(unpacked_unsigned, np.uint8(0x80)).view(np.int8)
    np.testing.assert_array_equal(unpacked_codes, codes)
    np.testing.assert_array_equal(packed.scales, np.repeat(scales[:, None], 2, axis=1))
    np.testing.assert_array_equal(packed.biases, -128.0 * packed.scales)
    np.testing.assert_allclose(
        dequantize_affine_q8_numpy(packed),
        codes.astype(np.float32) * scales[:, None],
        rtol=2e-7,
        atol=2e-6,
    )


def test_mlx_dequantizes_lossless_pack_to_exact_qstore_weights() -> None:
    mx = pytest.importorskip("mlx.core")
    codes = np.arange(-128, 128, dtype=np.int8).reshape(4, 64)
    scales = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    packed = pack_symmetric_qrow_int8(codes, scales)

    restored = mx.dequantize(
        mx.array(packed.weight),
        mx.array(packed.scales),
        mx.array(packed.biases),
        group_size=MLX_COMPONENT_GROUP_SIZE,
        bits=MLX_COMPONENT_BITS,
        mode="affine",
    )
    mx.eval(restored)
    np.testing.assert_array_equal(np.asarray(restored), codes.astype(np.float32) * scales[:, None])


def test_q4_requantizer_reports_its_loss_and_emits_native_mlx_shapes() -> None:
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(13)
    codes = rng.integers(-128, 128, size=(5, 128), dtype=np.int8)
    row_scales = np.asarray([0.01, 0.1, 0.25, 0.5, 1.0], dtype=np.float32)
    packed = requantize_qrow_affine_q4_mlx(codes, row_scales, mx=mx)

    assert packed.weight.shape == (5, 16)
    assert packed.scales.shape == (5, 2)
    assert packed.biases.shape == (5, 2)
    assert packed.elements == codes.size
    assert packed.max_abs_error > 0.0
    assert packed.sum_squared_error > 0.0
    restored = mx.dequantize(
        packed.weight,
        packed.scales,
        packed.biases,
        group_size=MLX_COMPONENT_GROUP_SIZE,
        bits=4,
        mode="affine",
    ).astype(mx.float32)
    mx.eval(restored)
    error = np.asarray(restored) - codes.astype(np.float32) * row_scales[:, None]
    assert float(np.max(np.abs(error))) == pytest.approx(packed.max_abs_error)
    assert float(np.square(error).sum()) == pytest.approx(packed.sum_squared_error, rel=1e-6)


@pytest.mark.parametrize(
    ("architecture", "logical_name", "kind", "path"),
    [
        ("qwen2", "embed", "qrow", "model.embed_tokens"),
        ("qwen2", "L23.q", "qrow", "model.layers.23.self_attn.q_proj"),
        ("qwen2", "L7.k.bias", "fp32", "model.layers.7.self_attn.k_proj.bias"),
        ("qwen3", "L2.q_norm", "fp32", "model.layers.2.self_attn.q_norm.weight"),
        ("qwen3", "L2.k_norm", "fp32", "model.layers.2.self_attn.k_norm.weight"),
        ("llama", "L4.down", "qrow", "model.layers.4.mlp.down_proj"),
        ("llama", "norm.final", "fp32", "model.norm.weight"),
        ("llama", "lm_head", "qrow", "lm_head"),
    ],
)
def test_canonical_qwen_llama_parameter_mapping(
    architecture: str,
    logical_name: str,
    kind: str,
    path: str,
) -> None:
    target = canonical_mlx_target(logical_name, architecture)
    assert (target.kind, target.path) == (kind, path)


def test_canonical_mapping_rejects_cross_architecture_and_unknown_names() -> None:
    with pytest.raises(MLXComponentMappingError, match="only valid for qwen3"):
        canonical_mlx_target("L0.q_norm", "qwen2")
    with pytest.raises(MLXComponentMappingError, match="unsupported canonical block"):
        canonical_mlx_target("L0.router", "qwen3")
    with pytest.raises(MLXComponentMappingError, match="support qwen2/qwen3/llama"):
        canonical_mlx_target("embed", "mixtral")


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_verified_reader_matches_production_custody_and_reads_exact_spans(
    tmp_path: Path,
    tied: bool,
) -> None:
    from mrun.engine.kernels.composite_qstore import ComponentGraph

    graph_path, _graph = _build_component_graph(tmp_path, tied=tied)
    production = ComponentGraph(graph_path)
    with VerifiedComponentGraphReader(graph_path) as reader:
        assert reader.custody_fingerprint_sha256 == production.fingerprint
        assert reader.roles == (
            ("body", "lexical_shared", "norm") if tied else ("body", "egress", "ingress", "norm")
        )
        codes, scales = reader.qrow("embed")
        np.testing.assert_array_equal(
            codes,
            np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int8),
        )
        np.testing.assert_array_equal(scales, np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        np.testing.assert_array_equal(
            reader.fp32("norm.final"), np.asarray([0.75, 1.25], dtype=np.float32)
        )
        reader.assert_unchanged()


def test_verified_reader_rejects_blob_tamper_before_exposing_weights(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=True)
    record = graph["components"]["body"]
    payload_path = graph_path.parent / record["relative_path"] / "weights.i8"
    payload = bytearray(payload_path.read_bytes())
    payload[0] ^= 0xFF
    payload_path.write_bytes(payload)

    with pytest.raises(MLXComponentGraphError, match="hash mismatch"):
        VerifiedComponentGraphReader(graph_path)


def test_qrow_packer_rejects_non_native_width_and_bad_scales() -> None:
    with pytest.raises(ValueError, match="divisible"):
        pack_symmetric_qrow_int8(np.ones((2, 63), dtype=np.int8), np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="finite and positive"):
        pack_symmetric_qrow_int8(
            np.ones((2, 64), dtype=np.int8), np.asarray([1.0, 0.0], dtype=np.float32)
        )


@pytest.mark.skipif(
    not RUN_MODEL_TESTS or platform.system() != "Darwin" or not QWEN_05_GRAPH.exists(),
    reason="needs MRUN_RUN_MODEL_TESTS=1, Apple MLX, and the local Qwen2.5-0.5B graph",
)
def test_real_qwen_05_build_load_argmax_and_native_generation(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    artifact_path = build_mlx_component_artifact(QWEN_05_GRAPH, tmp_path / "native")
    assert build_mlx_component_artifact(QWEN_05_GRAPH, tmp_path / "native") == artifact_path
    artifact = VerifiedMLXComponentArtifact(artifact_path)
    assert artifact.manifest["recipe"]["codec"] == MLX_COMPONENT_CODEC
    assert artifact.manifest["source"]["model"] == "qwen2.5-0.5b"
    assert artifact.config["max_position_embeddings"] == 32768

    engine = MLXComponentEngine(
        "qwen2.5-0.5b",
        component_graph=QWEN_05_GRAPH,
        native_artifact=artifact_path,
    )
    try:
        prompts = ["The capital of France is", "2 + 2 ="]
        ids_list = engine.encode(prompts)
        native_logits = engine.logits_batch_native(ids_list)
        native_winners = [
            int(np.asarray(logits[-1, : engine.semantic_token_count]).argmax())
            for logits in native_logits
        ]

        from mrun.engine.paged import PagedEngine

        paged = PagedEngine("qwen2.5-0.5b", component_graph=QWEN_05_GRAPH, cache_mb=0.0)
        try:
            paged_winners = [
                int(paged.logits(ids)[-1, : engine.semantic_token_count].argmax())
                for ids in ids_list
            ]
        finally:
            paged.close()
        assert native_winners == paged_winners

        scalar = engine.generate(ids_list[0], max_new_tokens=4)
        batched = engine.generate_batch([ids_list[0], ids_list[0]], max_new_tokens=4)
        assert batched == [scalar, scalar]
        assert all(0 <= token < engine.semantic_token_count for token in scalar)
        report = engine.runtime_report()
        assert report["artifact_bytes"] == artifact.shard_bytes
        assert "mlx_active_memory_mb_at_load_process_wide" in report
        with pytest.raises(ValueError, match="evict globally attended context"):
            engine.generate(ids_list[0], max_new_tokens=4, max_kv_size=2)
        engine.assert_content_identity_unchanged()
    finally:
        engine.close()


@pytest.mark.skipif(
    not RUN_MODEL_TESTS or platform.system() != "Darwin" or not QWEN_05_GRAPH.exists(),
    reason="needs MRUN_RUN_MODEL_TESTS=1, Apple MLX, and the local Qwen2.5-0.5B graph",
)
def test_real_qwen_05_q4_lane_is_distinct_custodied_and_runnable(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    artifact_path = build_mlx_component_q4_artifact(QWEN_05_GRAPH, tmp_path / "native-q4")
    artifact = VerifiedMLXComponentArtifact(artifact_path)
    assert artifact.schema == MLX_COMPONENT_Q4_NATIVE_SCHEMA
    assert artifact.codec == MLX_COMPONENT_Q4_CODEC
    assert artifact.bits == 4
    assert artifact.manifest["requantization"]["qrow_blocks"] > 0
    assert artifact.manifest["requantization"]["rmse"] > 0.0

    q4 = MLXComponentQ4Engine(
        "qwen2.5-0.5b",
        component_graph=QWEN_05_GRAPH,
        native_artifact=artifact_path,
    )
    try:
        generated = q4.generate("The capital of France is", max_new_tokens=8)
        assert len(generated) == 8
        assert all(0 <= token < q4.semantic_token_count for token in generated)
        assert q4.runtime_report()["weight_bits"] == 4
        q4.assert_content_identity_unchanged()
    finally:
        q4.close()


def test_native_artifact_verifier_rejects_non_artifact_directory(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "wrong"}))
    with pytest.raises(MLXComponentArtifactError):
        VerifiedMLXComponentArtifact(tmp_path)
