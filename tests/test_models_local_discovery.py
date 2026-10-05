from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from mrun.models import ModelSpec, find_safetensors, load_tokenizer, resolve_model


def _spec() -> ModelSpec:
    return ModelSpec("race-model", "Owner/RaceModel", "auto", "test")


def test_runtime_extra_root_rejects_partial_then_discovers_complete_set(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "RaceModel"
    model_dir.mkdir()
    monkeypatch.setenv("LLM_MODELS_EXTRA_ROOT", str(tmp_path))
    first = model_dir / "model-00001-of-00002.safetensors"
    second = model_dir / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first")

    assert find_safetensors(_spec()) == []

    second.write_bytes(b"second")
    assert find_safetensors(_spec()) == [first, second]


def test_index_declared_checkpoint_requires_every_shard(monkeypatch, tmp_path):
    model_dir = tmp_path / "RaceModel"
    model_dir.mkdir()
    monkeypatch.setenv("LLM_MODELS_EXTRA_ROOT", str(tmp_path))
    first = model_dir / "model-00001-of-00002.safetensors"
    second = model_dir / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": first.name,
                    "layer.1": second.name,
                }
            }
        ),
        encoding="utf-8",
    )

    assert find_safetensors(_spec()) == []

    second.write_bytes(b"second")
    assert find_safetensors(_spec()) == [first, second]


def test_tokenizer_resolves_extra_root_added_after_module_import(monkeypatch, tmp_path):
    model_dir = tmp_path / "RaceModel"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("LLM_MODELS_EXTRA_ROOT", str(tmp_path))

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return source, kwargs

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    source, kwargs = load_tokenizer(_spec(), local_files_only=True)

    assert source == str(model_dir)
    assert kwargs == {"local_files_only": True}


def test_qwen_base_generational_endpoints_have_explicit_registry_identity():
    expected = {
        "qwen3-0.6b-base": ("Qwen/Qwen3-0.6B-Base", "qwen3"),
        "qwen3-1.7b-base": ("Qwen/Qwen3-1.7B-Base", "qwen3"),
        "qwen3-4b-base": ("Qwen/Qwen3-4B-Base", "qwen3"),
        "qwen3-8b-base": ("Qwen/Qwen3-8B-Base", "qwen3"),
        "qwen3.5-0.8b-base": ("Qwen/Qwen3.5-0.8B-Base", "qwen3_5"),
        "qwen3.5-2b-base": ("Qwen/Qwen3.5-2B-Base", "qwen3_5"),
    }
    for alias, (hf_id, family) in expected.items():
        spec = resolve_model(alias)
        assert spec.hf_id == hf_id
        assert spec.family == family


def test_qwen38_registry_includes_the_large_sparse_checkpoint():
    spec = resolve_model("qwen3.8-2.4t-a95b-fp8")

    assert spec.hf_id == "Qwen/Qwen3.8-2.4T-A95B-FP8"
    assert spec.family == "qwen3_5"
    assert spec.label == "2400b/a95b"


def test_qwen25_dense_32b_endpoints_are_available_to_the_qwen2_decompiler():
    expected = {
        "qwen2.5-32b": "Qwen/Qwen2.5-32B",
        "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    }
    for alias, hf_id in expected.items():
        spec = resolve_model(alias)
        assert spec.hf_id == hf_id
        assert spec.family == "qwen2"
        assert spec.label == "32b"
