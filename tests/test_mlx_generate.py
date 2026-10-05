from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from mrun.engine.mlx import _MLXInterventionBase


class _Tokenizer:
    eos_token_id = 9

    def __call__(self, prompts, *, add_special_tokens=False):
        assert prompts == ["prompt"]
        assert add_special_tokens is True
        return {"input_ids": [[1, 2]]}

    def decode(self, ids, *, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ":".join(str(token) for token in ids)


def _engine() -> _MLXInterventionBase:
    engine = _MLXInterventionBase.__new__(_MLXInterventionBase)
    engine.model = object()
    engine._mlx_tokenizer = object()
    engine.tokenizer = _Tokenizer()
    engine._mx = SimpleNamespace(array=lambda value: ("mx", value))
    return engine


def _install_fake_mlx_lm(monkeypatch, tokens: list[int], calls: list[dict]) -> None:
    mlx_lm = ModuleType("mlx_lm")
    generate = ModuleType("mlx_lm.generate")
    sample_utils = ModuleType("mlx_lm.sample_utils")

    def make_sampler(*, temp):
        assert temp == 0.0
        return "greedy"

    def generate_step(prompt, model, **kwargs):
        calls.append(
            {
                "model": model,
                "prompt": prompt,
                **kwargs,
            }
        )
        yield from ((SimpleNamespace(item=lambda token=token: token), None) for token in tokens)

    generate.generate_step = generate_step
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)


def test_generate_prefills_once_and_stops_before_stop_token(monkeypatch):
    calls: list[dict] = []
    _install_fake_mlx_lm(monkeypatch, [3, 7, 8], calls)
    engine = _engine()

    result = engine.generate(
        "prompt",
        max_new_tokens=5,
        add_special_tokens=True,
        stop_ids=(7,),
        cache_mb=512,
    )

    assert result == [3]
    assert len(calls) == 1
    assert calls[0]["prompt"] == ("mx", [1, 2])
    assert calls[0]["max_tokens"] == 5
    assert calls[0]["sampler"] == "greedy"


def test_generate_accepts_token_ids_and_decodes_text(monkeypatch):
    calls: list[dict] = []
    _install_fake_mlx_lm(monkeypatch, [4, 5], calls)
    engine = _engine()

    result = engine.generate(np.asarray([10, 11]), max_new_tokens=2, return_text=True)

    assert result == "4:5"
    assert calls[0]["prompt"] == ("mx", [10, 11])


def test_generate_zero_tokens_does_not_load_optional_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
    engine = _engine()

    assert engine.generate([1], max_new_tokens=0) == []
    assert engine.generate([1], max_new_tokens=0, return_text=True) == ""
