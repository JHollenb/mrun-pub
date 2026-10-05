from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mrun.engine import component_builder
from mrun.engine.component_builder import (
    QSTORE_COMPONENT_LOWERING_SCHEMA,
    TRANSITIONAL_SCOPE,
    QStoreComponentBuildError,
    inspect_qstore_v3_component_artifact,
    lower_qstore_v3_to_component_graph,
)
from mrun.engine.kernels.composite_qstore import ComponentGraph, ComponentGraphError
from mrun.engine.kernels.qstore_build import (
    QSTORE_FILES,
    QSTORE_QUANTIZATION,
    QSTORE_SCHEMA,
)
from mrun.store_provenance import (
    build_builder_provenance,
    build_derived_provenance,
    build_source_provenance,
)


class _TokenizerBackend:
    def to_str(self) -> str:
        return '{"model":{"type":"tiny"}}'


class _TinyTokenizer:
    backend_tokenizer = _TokenizerBackend()
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = None
    unk_token_id = 2
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None

    def __len__(self) -> int:
        return 4

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return ("<s>", "</s>", "<unk>", "word")[token_id]


def _write_verified_qstore(root: Path, *, tied: bool) -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
    shard = checkpoint / "model.safetensors"
    shard.write_bytes(b"not-an-executable-checkpoint-but-exact-test-source")

    store = root / ("tied" if tied else "untied")
    store.mkdir()
    weights = bytearray(range(1, 13))
    scales = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).tobytes()
    blocks: dict[str, dict[str, Any]] = {
        "embed": {
            "kind": "qrow",
            "shape": [4, 2],
            "w_off": 0,
            "w_len": 8,
            "s_off": 0,
            "s_len": 16,
        },
        "L0.q": {
            "kind": "qrow",
            "shape": [2, 2],
            "w_off": 8,
            "w_len": 4,
            "s_off": 16,
            "s_len": 8,
        },
        "norm.final": {
            "kind": "fp32",
            "shape": [2],
            "e_off": 0,
            "e_len": 8,
        },
    }
    if tied:
        blocks["lm_head"] = {"alias": "embed"}
    else:
        blocks["lm_head"] = {
            "kind": "qrow",
            "shape": [4, 2],
            "w_off": 12,
            "w_len": 8,
            "s_off": 24,
            "s_len": 16,
        }
        weights.extend(range(13, 21))
        scales += np.asarray([7.0, 8.0, 9.0, 10.0], dtype=np.float32).tobytes()
    (store / "weights.i8").write_bytes(weights)
    (store / "scales.f32").write_bytes(scales)
    (store / "extras.f32").write_bytes(np.asarray([0.5, 1.5], dtype=np.float32).tobytes())

    source = build_source_provenance(
        checkpoint,
        [shard],
        model_name="tiny-qwen",
        hf_id="Test/TinyQwen",
        revision="0" * 40,
    )
    builder = build_builder_provenance(
        [Path(component_builder.__file__)],
        name="tests.synthetic-qstore",
        schema_version=QSTORE_SCHEMA,
        quantization=QSTORE_QUANTIZATION,
    )
    manifest = {
        "schema_version": QSTORE_SCHEMA,
        "model_name": "tiny-qwen",
        "arch": "qwen2",
        "dtype": "int8",
        "tie_word_embeddings": tied,
        "config": {
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "intermediate_size": 4,
            "vocab_size": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
        },
        "source": source,
        "builder": builder,
        "blocks": blocks,
    }
    manifest["derived"] = build_derived_provenance(
        store,
        QSTORE_FILES,
        semantic_manifest=manifest,
    )
    (store / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return store


def _flip_first_byte(path: Path) -> None:
    with path.open("r+b") as handle:
        value = handle.read(1)
        handle.seek(0)
        handle.write(bytes((value[0] ^ 1,)))


@pytest.mark.parametrize("tied", [False, True], ids=("untied", "tied"))
def test_verified_qstore_lowering_is_content_addressed_exact_and_reopenable(
    tmp_path: Path,
    tied: bool,
) -> None:
    source = _write_verified_qstore(tmp_path, tied=tied)
    output = tmp_path / "artifacts"

    built = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=_TinyTokenizer(),
        output_root=output,
    )
    reopened = inspect_qstore_v3_component_artifact(built.graph_path)
    reused = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=_TinyTokenizer(),
        output_root=output,
    )

    assert built.created is True
    assert reopened.created is False
    assert reused.created is False
    assert built.artifact_id == reopened.artifact_id == reused.artifact_id
    assert built.root.name == built.artifact_id
    assert built.graph_fingerprint_sha256 == reopened.graph_fingerprint_sha256
    assert built.source_payload_bytes == sum(
        (source / name).stat().st_size for name in QSTORE_FILES
    )
    graph = ComponentGraph(built.graph_path)
    assert graph.schema == "mrun-component-graph-v1"
    assert graph.raw["source_lineage"]["lowering_schema"] == QSTORE_COMPONENT_LOWERING_SCHEMA
    assert graph.raw["source_lineage"]["lowering_scope"] == TRANSITIONAL_SCOPE
    assert graph.raw["source_lineage"]["direct_source_artifact_lowering"] is False
    assert graph.raw["coverage"]["complete"] is True
    assert graph.raw["coverage"]["assigned_payload_bytes"] == built.source_payload_bytes
    assert graph.routes["embed"] == ("lexical_shared" if tied else "ingress")
    assert graph.routes["lm_head"] == ("lexical_shared" if tied else "egress")
    if tied:
        assert graph.logical_blocks["lm_head"] == {"alias": "embed"}
        assert set(graph.components["lexical_shared"]["allowed_names"]) == {
            "embed",
            "lm_head",
        }


def test_artifact_id_changes_with_exact_tokenizer_identity(tmp_path: Path) -> None:
    source = _write_verified_qstore(tmp_path, tied=True)
    output = tmp_path / "artifacts"
    first = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=_TinyTokenizer(),
        output_root=output,
    )
    descriptor = json.loads(json.dumps(ComponentGraph(first.graph_path).raw["tokenizer"]))
    descriptor["chat_template_sha256"] = hashlib.sha256(b"different").hexdigest()
    semantic_payload = {key: value for key, value in descriptor.items() if key != "semantic_sha256"}
    descriptor["semantic_sha256"] = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    second = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=descriptor,
        output_root=output,
    )

    assert first.artifact_id != second.artifact_id
    assert len(tuple(output.iterdir())) == 2


def test_existing_tampered_artifact_is_rejected_and_never_overwritten(tmp_path: Path) -> None:
    source = _write_verified_qstore(tmp_path, tied=False)
    output = tmp_path / "artifacts"
    built = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=_TinyTokenizer(),
        output_root=output,
    )
    payload = built.root / "components" / "egress" / "weights.i8"
    _flip_first_byte(payload)
    tampered = payload.read_bytes()

    with pytest.raises(ComponentGraphError, match="content hash mismatch"):
        inspect_qstore_v3_component_artifact(built.graph_path)
    with pytest.raises(ComponentGraphError, match="content hash mismatch"):
        lower_qstore_v3_to_component_graph(
            source_dir=source,
            tokenizer=_TinyTokenizer(),
            output_root=output,
        )
    assert payload.read_bytes() == tampered


def test_strict_reopen_recomputes_unfingerprinted_coverage(tmp_path: Path) -> None:
    source = _write_verified_qstore(tmp_path, tied=True)
    built = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=_TinyTokenizer(),
        output_root=tmp_path / "artifacts",
    )
    graph = json.loads(built.graph_path.read_text(encoding="utf-8"))
    # The base schema retained compatibility with the Stage-4 fingerprint, which omits
    # diagnostic coverage. The lowering-specific strict reopen must not trust it.
    graph["coverage"]["assigned_payload_bytes"] += 1
    built.graph_path.write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ComponentGraph(built.graph_path)  # base reader still accepts the legacy digest contract
    with pytest.raises(QStoreComponentBuildError, match="allocation coverage"):
        inspect_qstore_v3_component_artifact(built.graph_path)


def test_strict_reopen_detects_equal_shape_allocation_swap_after_full_resign(
    tmp_path: Path,
) -> None:
    source = _write_verified_qstore(tmp_path, tied=False)
    built = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=_TinyTokenizer(),
        output_root=tmp_path / "artifacts",
    )
    graph = json.loads(built.graph_path.read_text(encoding="utf-8"))
    ingress = built.root / "components" / "ingress"
    egress = built.root / "components" / "egress"
    for filename in QSTORE_FILES:
        ingress_bytes = (ingress / filename).read_bytes()
        egress_bytes = (egress / filename).read_bytes()
        assert len(ingress_bytes) == len(egress_bytes)
        (ingress / filename).write_bytes(egress_bytes)
        (egress / filename).write_bytes(ingress_bytes)
        for role, payload in (("ingress", egress_bytes), ("egress", ingress_bytes)):
            graph["components"][role]["blobs"][filename] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    for role, directory in (("ingress", ingress), ("egress", egress)):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        graph["components"][role]["semantic_content_sha256"] = (
            component_builder._component_semantic_digest(
                directory,
                manifest,
                graph["components"][role]["allowed_names"],
            )
        )
    graph["composite_fingerprint_sha256"] = hashlib.sha256(
        component_builder.canonical_json_bytes(component_builder._graph_fingerprint_payload(graph))
    ).hexdigest()
    built.graph_path.write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Both allocations have the same descriptor. A graph-only attacker can re-sign all
    # component hashes, but cannot make the reconstructed source payload match its v3 digest.
    ComponentGraph(built.graph_path)
    with pytest.raises(QStoreComponentBuildError, match="reconstruct source payload"):
        inspect_qstore_v3_component_artifact(built.graph_path)


def test_source_tamper_is_rejected_before_artifact_publication(tmp_path: Path) -> None:
    source = _write_verified_qstore(tmp_path, tied=True)
    _flip_first_byte(source / "weights.i8")
    output = tmp_path / "artifacts"

    with pytest.raises(QStoreComponentBuildError, match="fully verified"):
        lower_qstore_v3_to_component_graph(
            source_dir=source,
            tokenizer=_TinyTokenizer(),
            output_root=output,
        )
    assert not output.exists()


def test_duplicate_source_manifest_keys_are_rejected_even_when_qstore_semantics_match(
    tmp_path: Path,
) -> None:
    source = _write_verified_qstore(tmp_path, tied=True)
    manifest_path = source / "manifest.json"
    payload = manifest_path.read_text(encoding="utf-8")
    # The QStore JSON loader uses last-key-wins semantics, so this otherwise decodes to the
    # exact signed manifest. The lowering boundary independently rejects ambiguous JSON.
    manifest_path.write_text(
        payload.replace("{", '{\n  "dtype": "float16",', 1),
        encoding="utf-8",
    )

    with pytest.raises(QStoreComponentBuildError, match="duplicate JSON key 'dtype'"):
        lower_qstore_v3_to_component_graph(
            source_dir=source,
            tokenizer=_TinyTokenizer(),
            output_root=tmp_path / "artifacts",
        )


def test_failed_build_removes_staging_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_verified_qstore(tmp_path, tied=True)
    output = tmp_path / "artifacts"

    def fail_component(**_kwargs: Any) -> dict[str, Any]:
        raise QStoreComponentBuildError("injected component failure")

    monkeypatch.setattr(component_builder, "_write_component", fail_component)
    with pytest.raises(QStoreComponentBuildError, match="injected"):
        lower_qstore_v3_to_component_graph(
            source_dir=source,
            tokenizer=_TinyTokenizer(),
            output_root=output,
        )
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_partial_addressed_destination_fails_closed(tmp_path: Path) -> None:
    source = _write_verified_qstore(tmp_path, tied=True)
    descriptor = component_builder._coerce_tokenizer_descriptor(
        _TinyTokenizer(),
        configured_row_count=4,
    )
    verified = component_builder._open_verified_source(source)
    artifact_id = component_builder._artifact_id(
        source_identity=verified.identity,
        tokenizer_semantic_sha256=descriptor["semantic_sha256"],
    )
    destination = tmp_path / "artifacts" / artifact_id
    destination.mkdir(parents=True)
    marker = destination / "partial"
    marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(ComponentGraphError):
        lower_qstore_v3_to_component_graph(
            source_dir=source,
            tokenizer=descriptor,
            output_root=tmp_path / "artifacts",
        )
    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_module_cli_strict_inspect_emits_machine_readable_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_verified_qstore(tmp_path, tied=True)
    built = lower_qstore_v3_to_component_graph(
        source_dir=source,
        tokenizer=_TinyTokenizer(),
        output_root=tmp_path / "artifacts",
    )

    assert component_builder.main(["inspect", str(built.graph_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifact_id"] == built.artifact_id
    assert output["created"] is False
    assert output["scope"] == TRANSITIONAL_SCOPE
