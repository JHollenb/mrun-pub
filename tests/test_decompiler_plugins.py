from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from mrun.decompiler import (
    AdapterPluginError,
    AdapterPluginRequest,
    AdapterRegistry,
    Decompiler,
    NativeSourceComponentEmitter,
    Qwen2Adapter,
    inspect_adapter_plugin,
    load_adapter_plugin,
    registry_with_adapter_plugins,
)
from mrun.decompiler import plugins as plugin_module
from mrun.decompiler.cli import main as decompiler_main


class _Adapter:
    adapter_id = "test.plugin.adapter"
    adapter_version = "1"
    adapter_fingerprint = "a" * 64

    def match(self, source: Any, index: Any) -> Any:  # pragma: no cover - structural fixture
        raise AssertionError("not used")

    def compile_ir(self, source: Any, index: Any) -> Any:  # pragma: no cover - structural fixture
        raise AssertionError("not used")


class _CompilingAdapter(Qwen2Adapter):
    adapter_id = "test.plugin.qwen2"
    adapter_version = "1.0.0"


class _EntryPoint:
    group = "mrun.decompiler.adapters"
    name = "demo"
    value = "demo_plugin:get_adapters"
    extras: tuple[str, ...] = ()

    def __init__(self, factory: Any) -> None:
        self.factory = factory
        self.loads = 0

    def load(self) -> Any:
        self.loads += 1
        return self.factory


class _Distribution:
    version = "1.2.3"

    def __init__(self, root: Path, entry: _EntryPoint, files: tuple[str, ...]) -> None:
        self.root = root
        self.entry_points = (entry,)
        self.files = files
        self.metadata = {"Name": "demo-plugin"}

    def locate_file(self, value: str) -> Path:
        return self.root / value


def _installed_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    factory: Any | None = None,
    files: tuple[str, ...] | None = None,
) -> tuple[_Distribution, _EntryPoint]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    package = tmp_path / "demo_plugin.py"
    package.write_text("def get_adapters():\n    return ()\n", encoding="utf-8")
    metadata = tmp_path / "demo_plugin-1.2.3.dist-info" / "METADATA"
    metadata.parent.mkdir(exist_ok=True)
    metadata.write_text("Name: demo-plugin\nVersion: 1.2.3\n", encoding="utf-8")
    entry = _EntryPoint(factory or (lambda: (_Adapter(),)))
    distribution = _Distribution(
        tmp_path,
        entry,
        files or ("demo_plugin.py", "demo_plugin-1.2.3.dist-info/METADATA"),
    )
    monkeypatch.setattr(
        plugin_module.importlib.metadata,
        "distribution",
        lambda name: distribution,
    )
    return distribution, entry


def _write_tiny_qwen2(root: Path) -> None:
    root.mkdir()
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "attention_dropout": 0.0,
        "use_cache": True,
    }
    tensors = {
        "model.embed_tokens.weight": (8, 4),
        "model.norm.weight": (4,),
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.post_attention_layernorm.weight": (4,),
        "model.layers.0.self_attn.q_proj.weight": (4, 4),
        "model.layers.0.self_attn.q_proj.bias": (4,),
        "model.layers.0.self_attn.k_proj.weight": (2, 4),
        "model.layers.0.self_attn.k_proj.bias": (2,),
        "model.layers.0.self_attn.v_proj.weight": (2, 4),
        "model.layers.0.self_attn.v_proj.bias": (2,),
        "model.layers.0.self_attn.o_proj.weight": (4, 4),
        "model.layers.0.mlp.gate_proj.weight": (8, 4),
        "model.layers.0.mlp.up_proj.weight": (8, 4),
        "model.layers.0.mlp.down_proj.weight": (4, 8),
    }
    cursor = 0
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    for name, shape in sorted(tensors.items()):
        elements = 1
        for dimension in shape:
            elements *= dimension
        byte_count = elements * 2
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [cursor, cursor + byte_count],
        }
        cursor += byte_count
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(cursor)
    )


@pytest.mark.parametrize(
    "descriptor",
    (
        " demo-plugin==1.2.3:demo",
        "demo-plugin:demo",
        "demo-plugin==1.2.3",
        "demo-plugin==1.2.3:demo@ABC",
        "demo-plugin==1.2.3:demo@" + "g" * 64,
    ),
)
def test_plugin_descriptor_parser_rejects_unpinned_or_noncanonical_load_inputs(
    descriptor: str,
) -> None:
    with pytest.raises(AdapterPluginError):
        AdapterPluginRequest.parse(descriptor, require_digest=True)


def test_plugin_inspection_hashes_distribution_without_importing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _distribution, entry = _installed_plugin(tmp_path, monkeypatch)
    inspection = inspect_adapter_plugin("demo-plugin==1.2.3:demo")
    assert entry.loads == 0
    assert inspection.distribution == "demo-plugin"
    assert inspection.version == "1.2.3"
    assert inspection.entrypoint_value == "demo_plugin:get_adapters"
    assert len(inspection.files) == 2
    assert inspection.total_bytes == sum(path.byte_count for path in inspection.files)
    assert inspection.pinned_descriptor.endswith("@" + inspection.provenance_sha256)


def test_plugin_inspection_cli_prints_pinned_descriptor_without_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    _distribution, entry = _installed_plugin(tmp_path, monkeypatch)
    assert decompiler_main(["inspect-adapter-plugin", "demo-plugin==1.2.3:demo"]) == 0
    payload = json.loads(capfd.readouterr().out)
    assert payload["status"] == "installed-unloaded"
    assert payload["inspection"]["pinned_descriptor"].startswith("demo-plugin==1.2.3:demo@")
    assert entry.loads == 0


def test_plugin_load_requires_exact_digest_and_binds_it_into_adapter_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _distribution, entry = _installed_plugin(tmp_path, monkeypatch)
    inspection = inspect_adapter_plugin("demo-plugin==1.2.3:demo")
    loaded = load_adapter_plugin(inspection.pinned_descriptor)
    assert entry.loads == 1
    assert loaded.inspection == inspection
    assert len(loaded.adapters) == 1
    assert loaded.adapters[0].adapter_id == _Adapter.adapter_id
    assert loaded.adapters[0].adapter_fingerprint != _Adapter.adapter_fingerprint

    registry, plugins = registry_with_adapter_plugins(
        AdapterRegistry(),
        [inspection.pinned_descriptor],
    )
    assert len(registry.adapters) == 1
    assert len(plugins) == 1


def test_plugin_digest_mismatch_rejects_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _distribution, entry = _installed_plugin(tmp_path, monkeypatch)
    with pytest.raises(AdapterPluginError, match="authorized SHA-256"):
        load_adapter_plugin("demo-plugin==1.2.3:demo@" + "0" * 64)
    assert entry.loads == 0


def test_plugin_mutation_during_factory_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "demo_plugin.py"

    def mutating_factory() -> tuple[_Adapter, ...]:
        package.write_text("# changed while loading\n", encoding="utf-8")
        return (_Adapter(),)

    _distribution, entry = _installed_plugin(
        tmp_path,
        monkeypatch,
        factory=mutating_factory,
    )
    inspection = inspect_adapter_plugin("demo-plugin==1.2.3:demo")
    with pytest.raises(AdapterPluginError, match="changed while"):
        load_adapter_plugin(inspection.pinned_descriptor)
    assert entry.loads == 1


def test_plugin_inventory_rejects_escape_and_duplicate_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_plugin(tmp_path, monkeypatch, files=("../outside.py",))
    with pytest.raises(AdapterPluginError, match="escapes"):
        inspect_adapter_plugin("demo-plugin==1.2.3:demo")

    _installed_plugin(tmp_path, monkeypatch, files=("demo_plugin.py", "demo_plugin.py"))
    with pytest.raises(AdapterPluginError, match="duplicate"):
        inspect_adapter_plugin("demo-plugin==1.2.3:demo")


def test_plugin_registry_rejects_collision_with_builtin_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_plugin(tmp_path, monkeypatch)
    inspection = inspect_adapter_plugin("demo-plugin==1.2.3:demo")
    with pytest.raises(ValueError, match="duplicate adapter identity"):
        registry_with_adapter_plugins(
            AdapterRegistry((_Adapter(),)),
            [inspection.pinned_descriptor],
        )


def test_plugin_digest_rebinds_complete_ir_chain_and_component_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_tiny_qwen2(source)
    _installed_plugin(
        tmp_path / "installed",
        monkeypatch,
        factory=lambda: (_CompilingAdapter(),),
    )
    inspection = inspect_adapter_plugin("demo-plugin==1.2.3:demo")
    registry, loaded = registry_with_adapter_plugins(
        AdapterRegistry(), [inspection.pinned_descriptor]
    )
    result = Decompiler(registry=registry).decompile(
        source,
        source_id="Test/PluginQwen2",
        resolved_revision="a" * 40,
    )

    assert result.succeeded
    assert result.ir_bundle is not None
    bound = loaded[0].adapters[0].adapter_fingerprint
    assert bound != _CompilingAdapter().adapter_fingerprint
    assert result.report.selected_adapter_fingerprint == bound
    assert {
        result.ir_bundle.physical_weights.adapter_fingerprint,
        result.ir_bundle.model.adapter_fingerprint,
        result.ir_bundle.state.adapter_fingerprint,
        result.ir_bundle.io.adapter_fingerprint,
    } == {bound}

    built = NativeSourceComponentEmitter().build(result, tmp_path / "artifacts")
    assert built.verified_reopen
