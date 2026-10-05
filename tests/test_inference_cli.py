from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mrun.inference import NativeInferenceConfig
from mrun.inference import cli as inference_cli
from mrun.inference.loader import (
    LoadedNativeInference,
    _read_external_chat_template,
    _serving_workspace_bytes,
    _validate_mlx_architecture_options,
)
from mrun.runtime import OutputMode
from mrun.runtime.inference import SessionStoreTelemetry


def _argv(command: str = "describe") -> list[str]:
    return [
        command,
        "--backend",
        "mlx-q4",
        "--model",
        "qwen2.5-0.5b-instruct",
        "--component-graph",
        "graph.json",
    ]


def test_native_inference_config_is_strict_and_secret_free() -> None:
    config = NativeInferenceConfig(
        backend="dense-cuda",
        model_id="qwen2.5-0.5b-instruct",
        component_graph=Path("graph.json"),
        context_tokens=512,
        cuda_component_cache_mb=(("body", 42.5), ("norm", 1.0)),
        session_cache_bytes=1024,
    )
    payload = config.as_dict()
    assert payload["backend"] == "dense-cuda"
    assert payload["component_graph"] == "graph.json"
    assert payload["cuda_component_cache_mb"] == [["body", 42.5], ["norm", 1.0]]
    assert "api_key" not in json.dumps(payload)

    with pytest.raises(ValueError, match="unique"):
        NativeInferenceConfig(
            backend="dense-cuda",
            model_id="model",
            component_graph=Path("graph.json"),
            cuda_component_cache_mb=(("body", 1.0), ("body", 2.0)),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        NativeInferenceConfig(
            backend="mlx-q4",
            model_id="model",
            component_graph=Path("graph.json"),
            native_artifact=Path("artifact"),
            native_artifact_root=Path("root"),
        )
    with pytest.raises(ValueError, match="CUDA residency"):
        NativeInferenceConfig(
            backend="mlx-q8",
            model_id="model",
            component_graph=Path("graph.json"),
            cuda_resident_head_mb=1.0,
        )


def test_cli_configuration_and_duplicate_cache_roles() -> None:
    parser = inference_cli.build_parser()
    args = parser.parse_args(
        _argv()
        + [
            "--context",
            "768",
            "--headroom-mib",
            "12.5",
            "--session-cache-mib",
            "64",
        ]
    )
    config = inference_cli._config(args)
    assert config.context_tokens == 768
    assert config.headroom_bytes == int(12.5 * 1024**2)
    assert config.session_cache_bytes == 64 * 1024**2

    cuda_args = parser.parse_args(
        [
            "describe",
            "--backend",
            "cuda-source-int8-compact-head",
            "--model",
            "qwen2.5-7b-instruct",
            "--source-artifact",
            "canonical-source",
            "--cuda-decode-attention",
            "segmented-flash-gqa-decode-v1",
            "--cuda-decode-attention-tile",
            "128",
            "--cuda-body-fusion",
            "residual-rms-swiglu-v1",
            "--cuda-compatible-batch-size",
            "4",
            "--cuda-batch-queue-delay-ms",
            "1.5",
        ]
    )
    cuda_config = inference_cli._config(cuda_args)
    assert cuda_config.cuda_decode_attention_mode == "segmented-flash-gqa-decode-v1"
    assert cuda_config.cuda_decode_attention_tile == 128
    assert cuda_config.cuda_body_fusion_mode == "residual-rms-swiglu-v1"
    assert cuda_config.cuda_compatible_batch_size == 4
    assert cuda_config.cuda_batch_queue_delay_seconds == pytest.approx(0.0015)

    template_args = parser.parse_args(
        _argv()
        + [
            "--chat-template-file",
            "base-chat.jinja",
            "--chat-template-sha256",
            "a" * 64,
        ]
    )
    template_config = inference_cli._config(template_args)
    assert template_config.chat_template_file == Path("base-chat.jinja")
    assert template_config.chat_template_sha256 == "a" * 64

    cuda = parser.parse_args(
        [value if value != "mlx-q4" else "dense-cuda" for value in _argv()]
        + ["--cuda-component-cache-mb", "body=1", "--cuda-component-cache-mb", "body=2"]
    )
    with pytest.raises(ValueError, match="cannot repeat"):
        inference_cli._config(cuda)


def test_cli_wires_explicit_experimental_mlx_execution_controls() -> None:
    parser = inference_cli.build_parser()
    args = parser.parse_args(
        _argv()
        + [
            "--context",
            "1024",
            "--max-active-requests",
            "8",
            "--mlx-prefill-chunk-size",
            "256",
            "--mlx-compatible-batch-size",
            "4",
            "--mlx-batch-queue-delay-ms",
            "1.5",
            "--mlx-batch-scratch-mib",
            "32",
            "--mlx-kv-bits",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="transactional B1"):
        inference_cli._config(args)

    args = parser.parse_args(
        _argv()
        + [
            "--context",
            "1024",
            "--max-active-requests",
            "8",
            "--mlx-prefill-chunk-size",
            "256",
            "--mlx-compatible-batch-size",
            "4",
            "--mlx-batch-queue-delay-ms",
            "1.5",
            "--mlx-batch-scratch-mib",
            "32",
        ]
    )
    config = inference_cli._config(args)
    assert config.mlx_prefill_chunk_size == 256
    assert config.mlx_compatible_batch_size == 4
    assert config.mlx_batch_queue_delay_seconds == pytest.approx(0.0015)
    assert config.mlx_batch_scratch_bytes == 32 * 1024**2

    kv4 = inference_cli._config(parser.parse_args(_argv() + ["--mlx-kv-bits", "4"]))
    assert kv4.mlx_kv_bits == 4
    assert kv4.mlx_kv_group_size == 64


def test_mlx_execution_controls_fail_closed_outside_their_contract() -> None:
    common = {
        "backend": "mlx-q4",
        "model_id": "model",
        "component_graph": Path("graph.json"),
    }
    with pytest.raises(ValueError, match="cannot exceed context_tokens"):
        NativeInferenceConfig(
            **common,
            context_tokens=512,
            mlx_prefill_chunk_size=513,
        )
    with pytest.raises(ValueError, match="cannot exceed max_active_requests"):
        NativeInferenceConfig(
            **common,
            max_active_requests=4,
            mlx_compatible_batch_size=5,
        )
    with pytest.raises(ValueError, match="requires mlx_compatible_batch_size"):
        NativeInferenceConfig(**common, mlx_batch_scratch_bytes=1024)
    with pytest.raises(ValueError, match="finite non-negative"):
        NativeInferenceConfig(**common, mlx_batch_queue_delay_seconds=float("nan"))
    for unsupported_bits in (2, 3, 8):
        with pytest.raises(ValueError, match="only MLX KV4 group-64"):
            NativeInferenceConfig(**common, mlx_kv_bits=unsupported_bits)
    with pytest.raises(ValueError, match="requires mlx_kv_bits"):
        NativeInferenceConfig(**common, mlx_kv_group_size=32)
    with pytest.raises(ValueError, match="transactional B1"):
        NativeInferenceConfig(
            **common,
            mlx_kv_bits=4,
            mlx_compatible_batch_size=2,
        )
    with pytest.raises(ValueError, match="retained sessions are disabled"):
        NativeInferenceConfig(
            **common,
            mlx_kv_bits=4,
            session_cache_bytes=1024,
        )
    with pytest.raises(ValueError, match="require an MLX backend"):
        NativeInferenceConfig(
            backend="dense-cuda",
            model_id="model",
            component_graph=Path("graph.json"),
            mlx_compatible_batch_size=2,
        )
    for backend in ("mlx-source-q3", "mlx-source-q2"):
        weights_lowbit_kv4 = NativeInferenceConfig(
            backend=backend,
            model_id="model",
            source_artifact=Path("canonical-source"),
            mlx_kv_bits=4,
        )
        assert weights_lowbit_kv4.backend == backend
        assert weights_lowbit_kv4.mlx_kv_bits == 4
        assert weights_lowbit_kv4.mlx_kv_group_size == 64


def test_direct_source_route_and_aggregate_kv_admission_are_explicit() -> None:
    direct = NativeInferenceConfig(
        backend="mlx-source",
        model_id="qwen2.5-0.5b-instruct",
        source_artifact=Path("canonical-source"),
        context_tokens=100,
        max_active_requests=4,
        session_cache_bytes=700,
    )
    assert direct.component_graph is None
    assert (
        _serving_workspace_bytes(direct, SimpleNamespace(state_bytes_per_token=12))
        == 3 * 100 * 12 + 700
    )
    batched = NativeInferenceConfig(
        backend="mlx-source",
        model_id="qwen2.5-0.5b-instruct",
        source_artifact=Path("canonical-source"),
        context_tokens=100,
        max_active_requests=4,
        session_cache_bytes=700,
        mlx_compatible_batch_size=4,
    )
    assert (
        _serving_workspace_bytes(batched, SimpleNamespace(state_bytes_per_token=12))
        == 3 * 100 * 12 + 4 * 100 * 12 + 700
    )
    explicitly_bounded = NativeInferenceConfig(
        backend="mlx-source",
        model_id="qwen2.5-0.5b-instruct",
        source_artifact=Path("canonical-source"),
        context_tokens=100,
        max_active_requests=4,
        mlx_compatible_batch_size=4,
        mlx_batch_scratch_bytes=999,
    )
    assert (
        _serving_workspace_bytes(
            explicitly_bounded,
            SimpleNamespace(state_bytes_per_token=12),
        )
        == 3 * 100 * 12 + 999
    )
    cuda_slots = NativeInferenceConfig(
        backend="cuda-source-int8-compact-head",
        model_id="qwen2.5-7b-instruct",
        source_artifact=Path("canonical-source"),
        context_tokens=100,
        max_active_requests=8,
        cuda_decode_attention_mode="segmented-flash-gqa-decode-v1",
        cuda_compatible_batch_size=4,
    )
    assert (
        _serving_workspace_bytes(cuda_slots, SimpleNamespace(state_bytes_per_token=12))
        == 4 * 100 * 12
    )
    fixed_state = NativeInferenceConfig(
        backend="mlx-source",
        model_id="state-spaces/mamba-130m-hf",
        source_artifact=Path("canonical-source"),
        context_tokens=100,
        max_active_requests=4,
        session_cache_bytes=700,
    )
    assert (
        _serving_workspace_bytes(
            fixed_state,
            SimpleNamespace(state_bytes_per_token=0, state_fixed_bytes_per_row=64),
        )
        == 3 * 64 + 700
    )
    for backend in ("mlx-source-q4", "mlx-source-q3", "mlx-source-q2"):
        direct_quantized = NativeInferenceConfig(
            backend=backend,
            model_id="qwen2.5-0.5b-instruct",
            source_artifact=Path("canonical-source"),
        )
        assert direct_quantized.source_artifact == Path("canonical-source")
        assert direct_quantized.component_graph is None
    for backend in ("mlx-source-hybrid-q8", "mlx-source-hybrid-bf16"):
        hybrid = NativeInferenceConfig(
            backend=backend,
            model_id="qwen2.5-0.5b-instruct",
            source_artifact=Path("canonical-source"),
            native_artifact=Path(f"native-{backend}"),
            mlx_prefill_chunk_size=256,
            mlx_compatible_batch_size=2,
        )
        assert hybrid.source_artifact == Path("canonical-source")
        assert hybrid.native_artifact == Path(f"native-{backend}")
        assert hybrid.component_graph is None
    direct_cuda = NativeInferenceConfig(
        backend="cuda-source-int8",
        model_id="qwen2.5-0.5b-instruct",
        source_artifact=Path("canonical-source"),
        native_artifact=Path("native-cuda-int8"),
        cuda_component_cache_mb=(("body", 1.0),),
        cuda_resident_head_mb=2.0,
        cuda_decode_attention_mode="segmented-flash-gqa-decode-v1",
        cuda_decode_attention_tile=128,
        cuda_body_fusion_mode="residual-rms-swiglu-v1",
    )
    assert direct_cuda.source_artifact == Path("canonical-source")
    assert direct_cuda.native_artifact == Path("native-cuda-int8")
    assert direct_cuda.component_graph is None
    assert direct_cuda.cuda_decode_attention_mode == "segmented-flash-gqa-decode-v1"
    assert direct_cuda.cuda_decode_attention_tile == 128
    assert direct_cuda.cuda_body_fusion_mode == "residual-rms-swiglu-v1"
    compact_cuda = NativeInferenceConfig(
        backend="cuda-source-int8-compact-head",
        model_id="qwen2.5-0.5b-instruct",
        source_artifact=Path("canonical-source"),
        native_artifact=Path("native-cuda-int8"),
        cuda_component_cache_mb=(("body", 1.0),),
    )
    assert compact_cuda.source_artifact == Path("canonical-source")
    assert compact_cuda.cuda_resident_head_mb is None
    with pytest.raises(ValueError, match="forbids an expanded resident FP32 head"):
        NativeInferenceConfig(
            backend="cuda-source-int8-compact-head",
            model_id="qwen2.5-0.5b-instruct",
            source_artifact=Path("canonical-source"),
            cuda_resident_head_mb=2.0,
        )
    with pytest.raises(ValueError, match="power of two"):
        NativeInferenceConfig(
            backend="cuda-source-int8",
            model_id="model",
            source_artifact=Path("canonical-source"),
            cuda_decode_attention_tile=96,
        )
    with pytest.raises(ValueError, match="requires cuda_require_triton"):
        NativeInferenceConfig(
            backend="cuda-source-int8",
            model_id="model",
            source_artifact=Path("canonical-source"),
            cuda_decode_attention_mode="segmented-flash-gqa-decode-v1",
            cuda_require_triton=False,
        )
    with pytest.raises(ValueError, match="fused CUDA transformer body"):
        NativeInferenceConfig(
            backend="cuda-source-int8",
            model_id="model",
            source_artifact=Path("canonical-source"),
            cuda_body_fusion_mode="residual-rms-swiglu-v1",
            cuda_require_triton=False,
        )
    with pytest.raises(ValueError, match="requires segmented-flash"):
        NativeInferenceConfig(
            backend="cuda-source-int8",
            model_id="model",
            source_artifact=Path("canonical-source"),
            cuda_compatible_batch_size=2,
        )

    with pytest.raises(ValueError, match="requires source_artifact"):
        NativeInferenceConfig(
            backend="mlx-source",
            model_id="model",
            component_graph=Path("legacy-graph.json"),
        )
    for backend in ("mlx-source-q4", "mlx-source-q3", "mlx-source-q2"):
        with pytest.raises(ValueError, match="requires source_artifact"):
            NativeInferenceConfig(
                backend=backend,
                model_id="model",
                component_graph=Path("legacy-graph.json"),
            )
    with pytest.raises(ValueError, match="requires source_artifact"):
        NativeInferenceConfig(
            backend="mlx-source-hybrid-q8",
            model_id="model",
            component_graph=Path("legacy-graph.json"),
        )
    with pytest.raises(ValueError, match="requires source_artifact"):
        NativeInferenceConfig(
            backend="cuda-source-int8",
            model_id="model",
            component_graph=Path("legacy-graph.json"),
        )
    with pytest.raises(ValueError, match="require component_graph"):
        NativeInferenceConfig(
            backend="mlx-q8",
            model_id="model",
            source_artifact=Path("canonical-source"),
        )


def test_external_chat_template_file_is_read_once_from_stable_custody(tmp_path: Path) -> None:
    template = tmp_path / "base-chat.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    assert _read_external_chat_template(template) == "{{ messages }}"

    empty = tmp_path / "empty.jinja"
    empty.touch()
    with pytest.raises(ValueError, match="between 1 and 1048576"):
        _read_external_chat_template(empty)

    symlink = tmp_path / "redirect.jinja"
    symlink.symlink_to(template)
    with pytest.raises(ValueError, match="non-symlink"):
        _read_external_chat_template(symlink)

    with pytest.raises(ValueError, match="requires chat_template_file"):
        NativeInferenceConfig(
            backend="mlx-source",
            model_id="mamba-130m",
            source_artifact=Path("canonical-source"),
            chat_template_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        NativeInferenceConfig(
            backend="mlx-source",
            model_id="mamba-130m",
            source_artifact=Path("canonical-source"),
            chat_template_file=template,
            chat_template_sha256="not-a-digest",
        )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("mlx_compatible_batch_size", 2),
        ("mlx_batch_scratch_bytes", 1024),
        ("mlx_kv_bits", 4),
        ("mlx_paged_kv_page_size", 16),
    ],
)
def test_mamba_loader_rejects_transformer_state_controls(option: str, value: object) -> None:
    values: dict[str, object] = {
        "backend": "mlx-source",
        "model_id": "mamba-130m",
        "source_artifact": Path("canonical-source"),
    }
    if option == "mlx_paged_kv_page_size":
        values.update(
            mlx_paged_kv_page_size=value,
            mlx_paged_kv_page_count=256,
            context_tokens=4096,
        )
    else:
        values[option] = value
    if option == "mlx_batch_scratch_bytes":
        values["mlx_compatible_batch_size"] = 2
    config = NativeInferenceConfig(**values)
    with pytest.raises(ValueError, match="fixed-state Mamba forbids"):
        _validate_mlx_architecture_options(config, "mamba")

    _validate_mlx_architecture_options(config, "qwen2")


def test_mamba_loader_accepts_explicit_recurrent_prefill_chunking() -> None:
    config = NativeInferenceConfig(
        backend="mlx-source",
        model_id="mamba-130m",
        source_artifact=Path("canonical-source"),
        context_tokens=4096,
        mlx_prefill_chunk_size=128,
    )

    _validate_mlx_architecture_options(config, "mamba")


@pytest.mark.parametrize("backend", ["mlx-source-q3", "mlx-source-q2"])
def test_lowbit_loader_dispatch_is_exact_and_cleans_up_before_device_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    import mrun.decompiler.mlx_lowbit as mlx_lowbit
    import mrun.inference.loader as inference_loader

    events: list[tuple[str, object]] = []

    def engine_type(label: str):
        class FakeEngine:
            def __init__(self, model_id: str, **kwargs: object) -> None:
                events.append(("construct", (label, model_id, kwargs)))

            def close(self) -> None:
                events.append(("close", label))

        return FakeEngine

    monkeypatch.setattr(mlx_lowbit, "MLXSourceQ3Engine", engine_type("q3"))
    monkeypatch.setattr(mlx_lowbit, "MLXSourceQ2Engine", engine_type("q2"))

    class StopAfterEngine(RuntimeError):
        pass

    def stop_after_engine():
        raise StopAfterEngine("device boundary")

    monkeypatch.setattr(inference_loader, "describe_mlx_device", stop_after_engine)
    source = tmp_path / "canonical-source"
    source.mkdir()
    native = tmp_path / "native"
    with pytest.raises(StopAfterEngine, match="device boundary"):
        inference_loader.load_native_inference(
            NativeInferenceConfig(
                backend=backend,
                model_id="qwen2.5-0.5b-instruct",
                source_artifact=source,
                native_artifact=native,
            ),
            bearer_token=None,
        )

    expected = backend.removeprefix("mlx-source-")
    assert events[0][0] == "construct"
    label, model_id, kwargs = events[0][1]
    assert label == expected
    assert model_id == "qwen2.5-0.5b-instruct"
    assert kwargs == {
        "source_artifact": source,
        "native_artifact": native,
        "native_root": None,
    }
    assert events[1:] == [("close", expected)]


def test_compact_cuda_loader_binds_argmax_only_runtime_without_resident_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mrun.decompiler.cuda_native as cuda_native
    import mrun.engine.dense_qstore_cuda as dense_cuda
    import mrun.inference.loader as inference_loader

    events: list[tuple[str, object]] = []
    compact_contract = (
        "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1+"
        "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
    )

    class FakeCompactEngine:
        backend = "cuda-source-int8-compact-head"
        numerical_contract = compact_contract
        name = "qwen2.5-7b"
        semantic_token_count = 151_665
        graph = SimpleNamespace(raw={"tokenizer": {"chat_template_sha256": "a" * 64}})
        tokenizer = SimpleNamespace(chat_template="{{ messages }}")
        target = SimpleNamespace(experimental_reranked_head=True)
        store = SimpleNamespace(
            snapshot=lambda: {
                "fully_resident": True,
                "resident_exact_head": False,
                "resident_exact_head_bytes": 0,
            }
        )

        def __init__(self, model_id: str, **kwargs: object) -> None:
            events.append(("compact-engine", (model_id, kwargs)))

        def close(self) -> None:
            events.append(("engine-close", None))

    class FakeRuntime:
        def close(self) -> None:
            events.append(("runtime-close", None))

    runtime = FakeRuntime()
    model = SimpleNamespace(
        state_abi="gqa-kv-v1",
        state_bytes_per_token=64,
        state_fixed_bytes_per_row=0,
        components=(SimpleNamespace(role="body"),),
    )
    capabilities = SimpleNamespace(output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,))

    class FakeBackend:
        def __init__(
            self,
            engine: object,
            device: object,
            **kwargs: object,
        ) -> None:
            assert isinstance(engine, FakeCompactEngine)
            assert engine.store.snapshot()["resident_exact_head"] is False
            events.append(("backend", (device, kwargs)))
            self.model = model

        def capabilities(self, _device: object) -> object:
            events.append(("capabilities", capabilities.output_modes))
            return capabilities

        def plan(self, bound_model: object, workload: object, _device: object, **_kwargs: object):
            assert bound_model is model
            events.append(("plan", workload))
            return SimpleNamespace()

        def open(self, bound_model: object, _placement: object) -> FakeRuntime:
            assert bound_model is model
            events.append(("open", None))
            return runtime

    class StopAtService(RuntimeError):
        pass

    def fake_service(bound_runtime: object, **kwargs: object) -> object:
        assert bound_runtime is runtime
        events.append(("service", kwargs))
        raise StopAtService("service boundary")

    monkeypatch.setattr(
        cuda_native,
        "VerifiedSourceCudaInt8Artifact",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        dense_cuda,
        "DenseSourceCudaInt8CompactHeadEngine",
        FakeCompactEngine,
    )
    monkeypatch.setattr(
        dense_cuda,
        "DenseSourceCudaInt8Engine",
        lambda *_args, **_kwargs: pytest.fail("exact CUDA engine was selected"),
    )
    monkeypatch.setattr(
        inference_loader,
        "_auto_direct_cuda_residency",
        lambda _artifact: ({"body": 8.0}, 2048.0),
    )
    monkeypatch.setattr(inference_loader, "describe_cuda_device", lambda _index: "cuda-device")
    monkeypatch.setattr(inference_loader, "DenseCudaComponentExecutionBackend", FakeBackend)
    monkeypatch.setattr(
        inference_loader,
        "BoundChatTokenizer",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(inference_loader, "NativeGenerationService", fake_service)

    source = tmp_path / "source"
    native = tmp_path / "native"
    source.mkdir()
    native.mkdir()
    with pytest.raises(StopAtService, match="service boundary"):
        inference_loader.load_native_inference(
            NativeInferenceConfig(
                backend="cuda-source-int8-compact-head",
                model_id="qwen2.5-7b",
                source_artifact=source,
                native_artifact=native,
                context_tokens=128,
            ),
            bearer_token=None,
        )

    engine_kwargs = next(value for name, value in events if name == "compact-engine")[1]
    assert engine_kwargs["resident_exact_head_mb"] is None
    assert engine_kwargs["require_triton"] is True
    service_kwargs = next(value for name, value in events if name == "service")
    assert service_kwargs["supported_output_modes"] == (OutputMode.NEXT_TOKEN_ARGMAX,)
    assert [name for name, _value in events][-2:] == ["service", "runtime-close"]


def test_serve_security_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parser = inference_cli.build_parser()
    monkeypatch.delenv("MRUN_INFERENCE_API_KEY", raising=False)
    loopback = parser.parse_args(_argv("serve"))
    with pytest.raises(ValueError, match="set MRUN_INFERENCE_API_KEY"):
        inference_cli._serve_security(loopback)

    unauthenticated = parser.parse_args(_argv("serve") + ["--allow-unauthenticated"])
    assert inference_cli._serve_security(unauthenticated) == (None, {})

    external = parser.parse_args(
        _argv("serve")
        + [
            "--host",
            "0.0.0.0",
            "--allow-unauthenticated",
        ]
    )
    with pytest.raises(ValueError, match="unauthenticated non-loopback"):
        inference_cli._serve_security(external)

    monkeypatch.setenv("MRUN_INFERENCE_API_KEY", "not-printed")
    authenticated_external = parser.parse_args(_argv("serve") + ["--host", "0.0.0.0"])
    with pytest.raises(ValueError, match="plaintext non-loopback"):
        inference_cli._serve_security(authenticated_external)

    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("certificate")
    key.write_text("key")
    tls = parser.parse_args(
        _argv("serve")
        + ["--host", "0.0.0.0", "--ssl-certfile", str(cert), "--ssl-keyfile", str(key)]
    )
    bearer, ssl = inference_cli._serve_security(tls)
    assert bearer == "not-printed"
    assert ssl == {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}

    key_file = tmp_path / "inference.key"
    key_file.write_text("file-secret\n")
    key_file.chmod(0o600)
    from_file = parser.parse_args(_argv("serve") + ["--api-key-file", str(key_file)])
    assert inference_cli._serve_security(from_file) == ("file-secret", {})

    key_file.chmod(0o640)
    with pytest.raises(ValueError, match="group or other"):
        inference_cli._serve_security(from_file)


def test_trusted_proxy_allowlist_preserves_canonical_explicit_entries() -> None:
    parser = inference_cli.build_parser()
    args = parser.parse_args(
        _argv("serve")
        + [
            "--trusted-proxy",
            "127.0.0.1",
            "--trusted-proxy",
            "10.42.0.0/16",
            "--trusted-proxy",
            "2001:db8::1",
            "--trusted-proxy",
            "2001:db8:abcd::/48",
        ]
    )

    assert inference_cli._trusted_proxy_allowlist(args) == (
        "127.0.0.1",
        "10.42.0.0/16",
        "2001:db8::1",
        "2001:db8:abcd::/48",
    )


@pytest.mark.parametrize(
    "untrusted_value",
    (
        "*",
        "0.0.0.0/0",
        "::/0",
        "localhost",
        "proxy.internal.example",
        "127.0.0.1,10.0.0.1",
        " 127.0.0.1",
        "127.0.0.1 ",
        "192.0.2.1/24",
        "192.0.2.0/255.255.255.0",
        "2001:0db8::1",
        "fe80::1%lo0",
    ),
)
def test_trusted_proxy_rejects_wildcard_hostname_list_or_noncanonical_value(
    untrusted_value: str,
) -> None:
    parser = inference_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_argv("serve") + ["--trusted-proxy", untrusted_value])


@pytest.mark.parametrize(
    "duplicate_values",
    (
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1/32"),
        ("2001:db8::1", "2001:db8::1/128"),
    ),
)
def test_trusted_proxy_rejects_exact_or_semantic_duplicates(
    duplicate_values: tuple[str, str],
) -> None:
    parser = inference_cli.build_parser()
    args = parser.parse_args(
        _argv("serve")
        + [
            "--trusted-proxy",
            duplicate_values[0],
            "--trusted-proxy",
            duplicate_values[1],
        ]
    )

    with pytest.raises(ValueError, match="cannot repeat"):
        inference_cli._trusted_proxy_allowlist(args)


def test_serve_wires_only_the_exact_trusted_proxy_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = inference_cli.build_parser()
    config_calls: list[dict[str, object]] = []
    server_runs: list[object] = []
    closed: list[float] = []

    class FakeConfig:
        def __init__(self, app: object, **kwargs: object) -> None:
            self.app = app
            self.kwargs = kwargs
            config_calls.append(kwargs)

    class FakeServer:
        def __init__(self, server_config: FakeConfig) -> None:
            self.server_config = server_config

        def run(self) -> None:
            server_runs.append(self.server_config)

    class FakeStack:
        app = object()

        def close(self, *, drain_timeout_seconds: float) -> None:
            closed.append(drain_timeout_seconds)

    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=FakeConfig, Server=FakeServer),
    )
    monkeypatch.setattr(
        inference_cli,
        "load_native_inference",
        lambda config, *, bearer_token: FakeStack(),
    )

    default_args = parser.parse_args(_argv("serve") + ["--allow-unauthenticated"])
    assert inference_cli._serve(default_args, inference_cli._config(default_args)) == 0
    explicit_args = parser.parse_args(
        _argv("serve")
        + [
            "--allow-unauthenticated",
            "--trusted-proxy",
            "127.0.0.1",
            "--trusted-proxy",
            "10.42.0.0/16",
        ]
    )
    assert inference_cli._serve(explicit_args, inference_cli._config(explicit_args)) == 0

    assert len(server_runs) == 2
    assert config_calls[0]["proxy_headers"] is False
    assert config_calls[0]["forwarded_allow_ips"] == []
    assert config_calls[1]["proxy_headers"] is True
    assert config_calls[1]["forwarded_allow_ips"] == ["127.0.0.1", "10.42.0.0/16"]
    assert all(call["forwarded_allow_ips"] != "*" for call in config_calls)
    assert closed == [30.0, 30.0]


def test_api_key_file_rejects_noncanonical_or_ambiguous_inputs(tmp_path: Path) -> None:
    parser = inference_cli.build_parser()

    multiline = tmp_path / "multiline.key"
    multiline.write_text("one\ntwo\n")
    multiline.chmod(0o600)
    args = parser.parse_args(_argv("serve") + ["--api-key-file", str(multiline)])
    with pytest.raises(ValueError, match="exactly one canonical"):
        inference_cli._serve_security(args)

    target = tmp_path / "target.key"
    target.write_text("secret")
    target.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(target)
    args = parser.parse_args(_argv("serve") + ["--api-key-file", str(link)])
    with pytest.raises(ValueError, match="owner-only regular file"):
        inference_cli._serve_security(args)

    with pytest.raises(SystemExit):
        parser.parse_args(
            _argv("serve") + ["--api-key-file", str(target), "--api-key-env", "SOME_KEY"]
        )


def test_supervisor_runner_passes_argument_file_as_literal_argv(tmp_path: Path) -> None:
    capture = tmp_path / "argv.json"
    fake_mrun = tmp_path / "fake-mrun"
    fake_mrun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['MRUN_TEST_CAPTURE'], 'w', encoding='utf-8') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n"
    )
    fake_mrun.chmod(0o700)
    argument_file = tmp_path / "inference.args"
    literal_shell_text = "$(touch should-never-exist)"
    argument_file.write_text(
        f"# comment\n--model\nmodel with spaces\n--literal\n{literal_shell_text}\n"
    )
    key_file = tmp_path / "inference.key"
    key_file.write_text("secret\n")
    key_file.chmod(0o600)
    runner = Path(__file__).parents[1] / "scripts" / "run-inference-service.sh"
    environment = {
        **os.environ,
        "MRUN_INFERENCE_BIN": str(fake_mrun),
        "MRUN_INFERENCE_ARGS_FILE": str(argument_file),
        "MRUN_INFERENCE_KEY_FILE": str(key_file),
        "MRUN_TEST_CAPTURE": str(capture),
    }
    subprocess.run([runner], cwd=tmp_path, env=environment, check=True)
    assert json.loads(capture.read_text()) == [
        "inference",
        "serve",
        "--model",
        "model with spaces",
        "--literal",
        literal_shell_text,
        "--api-key-file",
        str(key_file),
    ]
    assert not (tmp_path / "should-never-exist").exists()


@pytest.mark.parametrize(
    "forbidden",
    (
        "--api-key-file",
        "--api-key-file=/tmp/other-key",
        "--api-key-env",
        "--api-key-env=OTHER_KEY",
        "--allow-unauthenticated",
        "--allow-unauthenticated-nonloopback",
    ),
)
def test_supervisor_runner_rejects_authentication_overrides(
    tmp_path: Path,
    forbidden: str,
) -> None:
    fake_mrun = tmp_path / "fake-mrun"
    fake_mrun.write_text(f"#!{sys.executable}\nraise SystemExit('must not execute')\n")
    fake_mrun.chmod(0o700)
    argument_file = tmp_path / "inference.args"
    argument_file.write_text(f"--model\nmodel\n{forbidden}\n")
    key_file = tmp_path / "inference.key"
    key_file.write_text("secret\n")
    key_file.chmod(0o600)
    runner = Path(__file__).parents[1] / "scripts" / "run-inference-service.sh"
    completed = subprocess.run(
        [runner],
        cwd=tmp_path,
        env={
            **os.environ,
            "MRUN_INFERENCE_BIN": str(fake_mrun),
            "MRUN_INFERENCE_ARGS_FILE": str(argument_file),
            "MRUN_INFERENCE_KEY_FILE": str(key_file),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "cannot override supervisor authentication" in completed.stderr


def test_describe_command_owns_and_closes_loaded_stack(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class FakeStack:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("close")

        def describe(self) -> dict[str, str]:
            return {"schema": "fake"}

    def fake_load(config: NativeInferenceConfig, *, bearer_token: str | None):
        assert config.backend == "mlx-q4"
        assert bearer_token is None
        return FakeStack()

    monkeypatch.setattr(inference_cli, "load_native_inference", fake_load)
    assert inference_cli.main(_argv()) == 0
    assert json.loads(capsys.readouterr().out) == {"schema": "fake"}
    assert events == ["enter", "close"]


@pytest.mark.parametrize("backend", ["mlx-source-q3", "mlx-source-q2"])
def test_lowbit_describe_cli_preserves_explicit_backend_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    backend: str,
) -> None:
    events: list[str] = []

    class FakeStack:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("close")

        def describe(self) -> dict[str, object]:
            return {
                "config": {"backend": backend},
                "route": {"promotion_status": "experimental"},
            }

    def fake_load(config: NativeInferenceConfig, *, bearer_token: str | None):
        assert config.backend == backend
        assert config.source_artifact == Path("canonical-source")
        assert config.native_artifact == Path("native-artifact")
        assert bearer_token is None
        return FakeStack()

    monkeypatch.setattr(inference_cli, "load_native_inference", fake_load)
    assert (
        inference_cli.main(
            [
                "describe",
                "--backend",
                backend,
                "--model",
                "qwen2.5-0.5b-instruct",
                "--source-artifact",
                "canonical-source",
                "--native-artifact",
                "native-artifact",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "config": {"backend": backend},
        "route": {"promotion_status": "experimental"},
    }
    assert events == ["enter", "close"]


def test_loaded_stack_drains_before_releasing_session_and_runtime() -> None:
    events: list[str] = []

    class Host:
        def begin_drain(self) -> None:
            events.append("host-drain")

        def mark_unready(self) -> None:
            events.append("host-unready")

    class Service:
        def shutdown(self, mode, *, wait: bool, timeout: float) -> bool:
            events.append(f"service-{mode.value}")
            assert wait is True
            assert timeout == 2.0
            return True

    class SessionStore:
        def close(self) -> None:
            events.append("session-close")

    class Runtime:
        def close(self) -> None:
            events.append("runtime-close")

    stack = LoadedNativeInference(
        config=NativeInferenceConfig(
            backend="mlx-q4",
            model_id="model",
            component_graph=Path("graph.json"),
        ),
        engine=object(),
        device=object(),
        backend=object(),
        model=object(),
        workload=SimpleNamespace(),
        placement=object(),
        runtime=Runtime(),
        tokenizer=SimpleNamespace(),
        session_store=SessionStore(),  # type: ignore[arg-type]
        compatible_batch_lane=None,
        service=Service(),  # type: ignore[arg-type]
        host=Host(),  # type: ignore[arg-type]
        app=object(),
    )
    stack.close(drain_timeout_seconds=2.0)
    stack.close(drain_timeout_seconds=2.0)
    assert events == [
        "host-drain",
        "service-drain",
        "host-unready",
        "session-close",
        "runtime-close",
    ]


def test_loaded_stack_describe_reconciles_session_authority() -> None:
    telemetry = SessionStoreTelemetry(
        identity_fingerprint="a" * 64,
        entries=2,
        retired_entries=1,
        active_leases=1,
        pinned_sources=2,
        stored_bytes=320,
        reserved_bytes=64,
        reserved_slots=1,
        max_entries=8,
        max_bytes=1024,
        hits=3,
        misses=4,
        installs=2,
        aborts=1,
        busy_rejections=0,
        identity_rejections=0,
        prefix_rejections=0,
        capacity_rejections=0,
        integrity_rejections=0,
        forks=2,
        fork_tokens=6,
        fork_bytes=48,
        cross_session_prefix_hits=2,
        cross_session_prefix_tokens=6,
        cross_session_prefix_bytes=48,
        ttl_evictions=0,
        lru_evictions=0,
        replacements=0,
        manual_evictions=0,
        cleanup_failures=0,
        accepting=True,
        poisoned=False,
    )

    class SessionStore:
        identity = SimpleNamespace(state_abi="fake-kv-v1")

        def telemetry(self) -> SessionStoreTelemetry:
            return telemetry

    stack = LoadedNativeInference(
        config=NativeInferenceConfig(
            backend="mlx-q4",
            model_id="model",
            component_graph=Path("graph.json"),
        ),
        engine=SimpleNamespace(runtime_report=lambda: {}),
        device=SimpleNamespace(as_dict=lambda: {}, fingerprint="device-fingerprint"),
        backend=SimpleNamespace(
            capabilities=lambda _device: SimpleNamespace(
                as_dict=lambda: {"promotion_status": "candidate"}
            )
        ),
        model=SimpleNamespace(as_dict=lambda: {}, fingerprint="model-fingerprint"),
        workload=SimpleNamespace(as_dict=lambda: {}),
        placement=SimpleNamespace(as_dict=lambda: {}, fingerprint="placement-fingerprint"),
        runtime=SimpleNamespace(
            route=SimpleNamespace(runtime_id="runtime-id"),
        ),
        tokenizer=SimpleNamespace(
            model_id="model",
            semantic_token_count=16,
            context_size=32,
            chat_template_sha256="b" * 64,
            legacy_raw_chat_template_sha256="c" * 64,
        ),
        session_store=SessionStore(),  # type: ignore[arg-type]
        compatible_batch_lane=None,
        service=object(),  # type: ignore[arg-type]
        host=object(),  # type: ignore[arg-type]
        app=object(),
    )
    assert stack.describe()["sessions"] == {
        "enabled": True,
        "store": {
            "identity_fingerprint": "a" * 64,
            "state_abi": "fake-kv-v1",
            "entries": 2,
            "retired_entries": 1,
            "active_leases": 1,
            "pinned_sources": 2,
            "stored_bytes": 320,
            "reserved_bytes": 64,
            "reserved_slots": 1,
            "max_entries": 8,
            "max_bytes": 1024,
            "budget_reconciled": True,
            "accepting": True,
            "poisoned": False,
            "cross_session_prefix": {"hits": 2, "tokens": 6, "bytes": 48},
        },
    }


def test_invalid_environment_name_is_rejected() -> None:
    args = argparse.Namespace(
        api_key_env="BAD-NAME",
        allow_unauthenticated=False,
        allow_unauthenticated_nonloopback=False,
        allow_plaintext_nonloopback=False,
        host="127.0.0.1",
        ssl_certfile=None,
        ssl_keyfile=None,
    )
    with pytest.raises(ValueError, match="valid environment"):
        inference_cli._serve_security(args)
