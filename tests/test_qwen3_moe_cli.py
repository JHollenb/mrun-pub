from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mrun import cli as cli_module
from mrun.engine import qwen3_moe_cli
from mrun.engine.base import EngineCapabilities


def test_qwen_runtime_cli_preserves_fp8_and_environment_driven_defaults(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_generate(args: argparse.Namespace) -> dict[str, Any]:
        captured.update(vars(args))
        return {"status": "ok"}

    monkeypatch.setattr(qwen3_moe_cli, "_generate", fake_generate)
    monkeypatch.setattr(qwen3_moe_cli, "_emit", lambda *_args: None)

    assert qwen3_moe_cli.main(["generate", "--prompt", "Keep defaults."]) == 0
    assert captured["expert_codec"] == "fp8"
    assert captured["cache_mb"] == 7100.0
    assert captured["max_active_pages"] == 128
    for name in (
        "host_cache_mb",
        "warm_host",
        "route_prefetch",
        "cache_policy",
        "page_binding_policy",
        "prefill_page_policy",
        "route_reduction_policy",
        "w4_arithmetic_policy",
    ):
        assert captured[name] is None


def test_qwen_runtime_cli_forwards_explicit_w4_tiers_and_policies(monkeypatch) -> None:
    sentinel = object()
    opened: list[tuple[str, dict[str, Any]]] = []

    def fake_engine(model: str, **kwargs: Any) -> object:
        opened.append((model, kwargs))
        return sentinel

    def fake_generate(args: argparse.Namespace) -> dict[str, Any]:
        assert qwen3_moe_cli._engine(args) is sentinel
        return {"status": "ok"}

    monkeypatch.setattr(qwen3_moe_cli, "Qwen3MoeCudaEngine", fake_engine)
    monkeypatch.setattr(qwen3_moe_cli, "_generate", fake_generate)
    monkeypatch.setattr(qwen3_moe_cli, "_emit", lambda *_args: None)

    assert (
        qwen3_moe_cli.main(
            [
                "generate",
                "--model",
                "qwen3-30b-a3b",
                "--store-dir",
                "/stores/qwen-w4",
                "--expert-codec",
                "w4",
                "--cache-mb",
                "6400",
                "--max-active-pages",
                "64",
                "--host-cache-mb",
                "12000",
                "--no-warm-host",
                "--no-route-prefetch",
                "--cache-policy",
                "layer-frequency-lru-v1",
                "--page-binding-policy",
                "slot-indirect-v1",
                "--prefill-page-policy",
                "transient-frequency-v1",
                "--route-reduction-policy",
                "stable-route-rank-v1",
                "--w4-arithmetic-policy",
                "w4-g128-postscale-bf16-v1",
                "--verify-content",
                "--prompt",
                "Route the compact pages.",
            ]
        )
        == 0
    )
    assert opened == [
        (
            "qwen3-30b-a3b",
            {
                "store_dir": Path("/stores/qwen-w4"),
                "expert_codec": "w4",
                "cache_mb": 6400.0,
                "max_active_pages": 64,
                "host_cache_mb": 12000.0,
                "warm_host": False,
                "route_prefetch": False,
                "cache_policy": "layer-frequency-lru-v1",
                "page_binding_policy": "slot-indirect-v1",
                "prefill_page_policy": "transient-frequency-v1",
                "route_reduction_policy": "stable-route-rank-v1",
                "w4_arithmetic_policy": "w4-g128-postscale-bf16-v1",
                "verify_store_content": True,
            },
        )
    ]


class _FakeTokenizer:
    def decode(self, _tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "decoded"


class _FakeGenerationEngine:
    tokenizer = _FakeTokenizer()

    def __enter__(self) -> _FakeGenerationEngine:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(generation=True, generation_batch=True)

    def generate_batch(self, prompts: list[str], *, max_new_tokens: int) -> list[list[int]]:
        assert max_new_tokens == 1
        return [[index + 1] for index, _prompt in enumerate(prompts)]

    def runtime_report(self) -> dict[str, str]:
        return {"status": "fake"}


def test_generic_run_forwards_qwen_runtime_options_and_preserves_omission(
    tmp_path,
    monkeypatch,
) -> None:
    from mrun import engine as engine_module

    opened: list[tuple[str, str, dict[str, Any]]] = []

    def fake_open_engine(model: str, *, backend: str, **kwargs: Any) -> _FakeGenerationEngine:
        opened.append((model, backend, kwargs))
        return _FakeGenerationEngine()

    monkeypatch.setattr(engine_module, "open_engine", fake_open_engine)

    assert (
        cli_module.main(
            [
                "run",
                "qwen3-30b-a3b",
                "--backend",
                "qwen3-moe-cuda",
                "--prompt",
                "Use W4 backing.",
                "--max-new-tokens",
                "1",
                "--expert-codec",
                "w4",
                "--host-cache-mb",
                "8000",
                "--warm-host",
                "--no-route-prefetch",
                "--cache-policy",
                "layer-frequency-lru-v1",
                "--page-binding-policy",
                "slot-indirect-v1",
                "--prefill-page-policy",
                "cache-fill-v1",
                "--route-reduction-policy",
                "stable-route-rank-v1",
                "--w4-arithmetic-policy",
                "w4-g128-postscale-bf16-v1",
                "--out",
                str(tmp_path / "explicit.json"),
            ]
        )
        == 0
    )
    assert opened[-1] == (
        "qwen3-30b-a3b",
        "qwen3-moe-cuda",
        {
            "expert_codec": "w4",
            "host_cache_mb": 8000.0,
            "warm_host": True,
            "route_prefetch": False,
            "cache_policy": "layer-frequency-lru-v1",
            "page_binding_policy": "slot-indirect-v1",
            "prefill_page_policy": "cache-fill-v1",
            "route_reduction_policy": "stable-route-rank-v1",
            "w4_arithmetic_policy": "w4-g128-postscale-bf16-v1",
        },
    )

    assert (
        cli_module.main(
            [
                "run",
                "qwen3-30b-a3b",
                "--backend",
                "qwen3-moe-cuda",
                "--prompt",
                "Preserve defaults.",
                "--max-new-tokens",
                "1",
                "--out",
                str(tmp_path / "defaults.json"),
            ]
        )
        == 0
    )
    assert opened[-1] == ("qwen3-30b-a3b", "qwen3-moe-cuda", {})
