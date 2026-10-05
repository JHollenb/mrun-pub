from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mrun.inference import NativeInferenceConfig
from mrun.inference import cli as inference_cli
from mrun.inference.loader import _serving_workspace_bytes


def _config(**overrides: object) -> NativeInferenceConfig:
    values: dict[str, object] = {
        "backend": "mlx-q4",
        "model_id": "model",
        "component_graph": Path("graph.json"),
        "context_tokens": 64,
        "mlx_paged_kv_page_size": 16,
        "mlx_paged_kv_page_count": 8,
    }
    values.update(overrides)
    return NativeInferenceConfig(**values)  # type: ignore[arg-type]


def test_paged_mlx_service_config_charges_one_fixed_pool_not_logical_aliases() -> None:
    config = _config(
        max_active_requests=32,
        session_cache_bytes=4096,
        session_max_entries=64,
    )

    assert config.mlx_paged_kv_page_size == 16
    assert config.mlx_paged_kv_page_count == 8
    assert (
        _serving_workspace_bytes(
            config,
            SimpleNamespace(state_bytes_per_token=48),
        )
        == 0
    )


def test_paged_mlx_cli_wires_explicit_pool_geometry() -> None:
    parser = inference_cli.build_parser()
    args = parser.parse_args(
        [
            "describe",
            "--backend",
            "mlx-q4",
            "--model",
            "model",
            "--component-graph",
            "graph.json",
            "--context",
            "64",
            "--mlx-paged-kv-page-size",
            "16",
            "--mlx-paged-kv-page-count",
            "8",
        ]
    )

    config = inference_cli._config(args)
    assert config.mlx_paged_kv_page_size == 16
    assert config.mlx_paged_kv_page_count == 8


def test_paged_metal_decode_cli_is_a_separate_explicit_execution_identity() -> None:
    parser = inference_cli.build_parser()
    args = parser.parse_args(
        [
            "describe",
            "--backend",
            "mlx-q4",
            "--model",
            "model",
            "--component-graph",
            "graph.json",
            "--context",
            "64",
            "--mlx-paged-kv-page-size",
            "16",
            "--mlx-paged-kv-page-count",
            "8",
            "--mlx-paged-decode-attention",
        ]
    )

    config = inference_cli._config(args)
    assert config.mlx_paged_decode_attention is True


def test_paged_metal_decode_rejects_implicit_pages_and_chunked_prefill() -> None:
    with pytest.raises(ValueError, match="requires explicit BF16 page size"):
        NativeInferenceConfig(
            backend="mlx-q4",
            model_id="model",
            component_graph=Path("graph.json"),
            mlx_paged_decode_attention=True,
        )
    with pytest.raises(ValueError, match="separate routes"):
        _config(
            mlx_paged_decode_attention=True,
            mlx_prefill_chunk_size=16,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"mlx_paged_kv_page_count": None}, "requires both"),
        ({"mlx_paged_kv_page_size": 12}, "power of two"),
        ({"mlx_paged_kv_page_count": 2}, "at least one maximum-context"),
        ({"mlx_kv_bits": 4}, "cannot be combined"),
        ({"mlx_compatible_batch_size": 2}, "transactional B1"),
    ),
)
def test_paged_mlx_service_config_rejects_ambiguous_storage_contracts(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_paged_mlx_service_config_is_rejected_by_cuda_routes() -> None:
    with pytest.raises(ValueError, match="MLX execution options require an MLX backend"):
        NativeInferenceConfig(
            backend="dense-cuda",
            model_id="model",
            component_graph=Path("graph.json"),
            context_tokens=64,
            mlx_paged_kv_page_size=16,
            mlx_paged_kv_page_count=8,
        )
