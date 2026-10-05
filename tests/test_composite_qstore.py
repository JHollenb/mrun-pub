from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mrun.engine.kernels.composite_qstore as composite_module
from mrun.engine.kernels.composite_qstore import (
    POC_COMPONENT_GRAPH_SCHEMA,
    ComponentGraph,
    ComponentGraphError,
    ComponentOutputContractError,
    CompositeQStore,
    canonical_json_bytes,
    tokenizer_descriptor,
)


class _TinyBackendTokenizer:
    def __init__(self, payload: str = '{"model":"tiny"}') -> None:
        self.payload = payload

    def to_str(self) -> str:
        return self.payload


class _TinyTokenizer:
    def __init__(self, tokens: tuple[str, ...] = ("a", "b", "c")) -> None:
        self.tokens = tokens
        self.backend_tokenizer = _TinyBackendTokenizer()
        self.chat_template = "tiny: {{ messages }}"
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 2
        self.unk_token_id = 1
        self.sep_token_id = None
        self.cls_token_id = None
        self.mask_token_id = None

    def __len__(self) -> int:
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _qrow(
    shape: tuple[int, int],
    *,
    w_off: int,
    s_off: int,
) -> dict[str, Any]:
    rows, columns = shape
    return {
        "kind": "qrow",
        "shape": [rows, columns],
        "w_off": w_off,
        "w_len": rows * columns,
        "s_off": s_off,
        "s_len": rows * 4,
    }


def _fp32(size: int, *, e_off: int) -> dict[str, Any]:
    return {
        "kind": "fp32",
        "shape": [size],
        "e_off": e_off,
        "e_len": size * 4,
    }


def _semantic_component_digest(
    component_dir: Path,
    blocks: dict[str, dict[str, Any]],
    names: list[str],
) -> str:
    entries: list[dict[str, str]] = []
    layout = {
        "qrow": (
            ("weights.i8", "w_off", "w_len"),
            ("scales.f32", "s_off", "s_len"),
        ),
        "fp32": (("extras.f32", "e_off", "e_len"),),
    }
    for name in sorted(names):
        block = blocks[name]
        digest = hashlib.sha256()
        if "alias" in block:
            digest.update(canonical_json_bytes({"name": name, "alias": block["alias"]}))
        else:
            digest.update(
                canonical_json_bytes({"name": name, "kind": block["kind"], "shape": block["shape"]})
            )
            for filename, offset_key, length_key in layout[block["kind"]]:
                payload = (component_dir / filename).read_bytes()
                offset = int(block[offset_key])
                length = int(block[length_key])
                value = payload[offset : offset + length]
                digest.update(canonical_json_bytes({"file_kind": filename, "bytes": length}))
                digest.update(value)
        entries.append({"name": name, "semantic_block_sha256": digest.hexdigest()})
    return _sha256(canonical_json_bytes(entries))


def _logical_blocks(*, tied: bool) -> dict[str, dict[str, Any]]:
    blocks = {
        "embed": _qrow((4, 2), w_off=0, s_off=0),
        "L0.ln1": _fp32(2, e_off=0),
        "L0.q": _qrow((2, 2), w_off=8, s_off=16),
        "norm.final": _fp32(2, e_off=8),
    }
    blocks["lm_head"] = {"alias": "embed"} if tied else _qrow((4, 2), w_off=12, s_off=24)
    return blocks


def _physical_layouts(
    *,
    tied: bool,
) -> dict[str, tuple[dict[str, dict[str, Any]], bytes, bytes, bytes]]:
    body = (
        {
            "L0.ln1": _fp32(2, e_off=0),
            "L0.q": _qrow((2, 2), w_off=0, s_off=0),
        },
        np.asarray([1, -2, 3, -4], dtype=np.int8).tobytes(),
        np.asarray([0.25, 0.5], dtype=np.float32).tobytes(),
        np.asarray([1.0, 1.5], dtype=np.float32).tobytes(),
    )
    norm = (
        {"norm.final": _fp32(2, e_off=0)},
        b"\x00",
        np.asarray([0.0], dtype=np.float32).tobytes(),
        np.asarray([0.75, 1.25], dtype=np.float32).tobytes(),
    )
    embed_weights = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8).tobytes()
    embed_scales = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
    lexical = (
        {
            "embed": _qrow((4, 2), w_off=0, s_off=0),
            "lm_head": {"alias": "embed"},
        },
        embed_weights,
        embed_scales,
        np.asarray([0.0], dtype=np.float32).tobytes(),
    )
    if tied:
        return {"body": body, "norm": norm, "lexical_shared": lexical}

    ingress = (
        {"embed": _qrow((4, 2), w_off=0, s_off=0)},
        embed_weights,
        embed_scales,
        np.asarray([0.0], dtype=np.float32).tobytes(),
    )
    egress = (
        {"lm_head": _qrow((4, 2), w_off=0, s_off=0)},
        np.asarray([-1, -2, -3, -4, -5, -6, -7, -8], dtype=np.int8).tobytes(),
        np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32).tobytes(),
        np.asarray([0.0], dtype=np.float32).tobytes(),
    )
    return {"body": body, "norm": norm, "ingress": ingress, "egress": egress}


def _role_names(*, tied: bool) -> dict[str, list[str]]:
    if tied:
        return {
            "body": ["L0.ln1", "L0.q"],
            "lexical_shared": ["embed", "lm_head"],
            "norm": ["norm.final"],
        }
    return {
        "body": ["L0.ln1", "L0.q"],
        "egress": ["lm_head"],
        "ingress": ["embed"],
        "norm": ["norm.final"],
    }


def _topology(*, tied: bool) -> dict[str, Any]:
    return {
        "declared_tied": tied,
        "observed_tied": tied,
        "embed_physical_root": "embed",
        "lm_head_physical_root": "embed" if tied else "lm_head",
        "aliases": {"lm_head": "embed"} if tied else {},
        "alias_classes": {"embed": ["embed", "lm_head"]} if tied else {},
        "physical_roles": (
            ["body", "lexical_shared", "norm"] if tied else ["body", "egress", "ingress", "norm"]
        ),
    }


def _operation_contracts(*, tied: bool) -> dict[str, Any]:
    input_role = "lexical_shared" if tied else "ingress"
    base = ["body", input_role, "norm"]
    full = base if tied else [*base, "egress"]
    return {
        "full_logits": {
            "required_components": sorted(full),
            "lm_head_methods": ["row_blocks"],
        },
        "lexical_hidden": {
            "required_components": sorted(base),
            "lm_head_methods": [],
        },
        "selected_rows": {
            "required_components": sorted(full),
            "lm_head_methods": ["embed_rows"],
        },
    }


def _graph_fingerprint_payload(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": graph["schema"],
        "model": graph["model"],
        "architecture": graph["architecture"],
        "body_abi_sha256": graph["body_abi"]["semantic_sha256"],
        "tokenizer_sha256": graph["tokenizer"]["semantic_sha256"],
        "topology": graph["topology"],
        "routes": graph["routes"],
        "components": {
            role: {
                "semantic_content_sha256": record["semantic_content_sha256"],
                "manifest_sha256": record["manifest_sha256"],
                "allowed_names": record["allowed_names"],
            }
            for role, record in sorted(graph["components"].items())
        },
        "operation_contracts": graph["operation_contracts"],
    }


def _resign_graph(graph: dict[str, Any]) -> None:
    graph["composite_fingerprint_sha256"] = _sha256(
        canonical_json_bytes(_graph_fingerprint_payload(graph))
    )


def _body_abi_digest(graph: dict[str, Any]) -> str:
    payload = {
        "architecture": graph["architecture"],
        "config": graph["body_abi"]["config"],
        "tie_word_embeddings": graph["topology"]["observed_tied"],
        "logical_blocks": {
            name: (
                {"alias": block["alias"]}
                if "alias" in block
                else {"kind": block["kind"], "shape": block["shape"]}
            )
            for name, block in sorted(graph["logical_blocks"].items())
        },
    }
    return _sha256(canonical_json_bytes(payload))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_component_graph(tmp_path: Path, *, tied: bool) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / ("tied" if tied else "untied")
    components_root = root / "components"
    components_root.mkdir(parents=True)
    config = {
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 4,
        "vocab_size": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
    }
    role_names = _role_names(tied=tied)
    component_records: dict[str, dict[str, Any]] = {}
    for role, (blocks, weights, scales, extras) in _physical_layouts(tied=tied).items():
        component_dir = components_root / role
        component_dir.mkdir()
        payloads = {
            "weights.i8": weights,
            "scales.f32": scales,
            "extras.f32": extras,
        }
        for filename, payload in payloads.items():
            (component_dir / filename).write_bytes(payload)
        manifest = {
            "model_name": f"tiny-{role}",
            "arch": "qwen2",
            "dtype": "int8",
            "tie_word_embeddings": tied,
            "config": config,
            "blocks": blocks,
        }
        manifest_path = component_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        component_records[role] = {
            "role": role,
            "relative_path": f"components/{role}",
            "allowed_names": role_names[role],
            "manifest_sha256": _sha256(manifest_path.read_bytes()),
            "semantic_content_sha256": _semantic_component_digest(
                component_dir,
                blocks,
                role_names[role],
            ),
            "blobs": {
                filename: {"bytes": len(payload), "sha256": _sha256(payload)}
                for filename, payload in payloads.items()
            },
        }

    routes = {name: role for role, names in role_names.items() for name in names}
    body_abi = {
        "config": config,
        "semantic_sha256": "0" * 64,
    }
    graph = {
        "schema": POC_COMPONENT_GRAPH_SCHEMA,
        "model": "tiny-qwen",
        "architecture": "qwen2",
        "body_abi": body_abi,
        "tokenizer": tokenizer_descriptor(_TinyTokenizer()),
        "topology": _topology(tied=tied),
        "logical_blocks": _logical_blocks(tied=tied),
        "source_lineage": {"identity_status": "legacy-unverified"},
        "components": component_records,
        "routes": dict(sorted(routes.items())),
        "operation_contracts": _operation_contracts(tied=tied),
    }
    graph["body_abi"]["semantic_sha256"] = _body_abi_digest(graph)
    _resign_graph(graph)
    graph_path = root / "model-graph.json"
    _write_json(graph_path, graph)
    return graph_path, graph


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_contract_views_route_to_exact_physical_owners_and_open_lazily(
    tmp_path: Path,
    tied: bool,
) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=tied)
    store = CompositeQStore(graph_path)

    assert store.opened_roles == {"body"}
    hidden = store.for_contract("hidden_state_only")
    embedded = hidden.embed_rows("embed", np.asarray([0, 1], dtype=np.int64))
    final_norm = hidden.fp32("norm.final")
    assert embedded.shape == (2, 2)
    assert final_norm.shape == (2,)
    expected_hidden_roles = (
        {"body", "lexical_shared", "norm"} if tied else {"body", "ingress", "norm"}
    )
    assert store.opened_roles == expected_hidden_roles
    assert hidden.has("lm_head") is False

    selected = store.for_contract("selected_token_rows")
    selected_head = selected.embed_rows("lm_head", np.asarray([0, 1], dtype=np.int64))
    assert selected_head.shape == (2, 2)
    assert store.opened_roles == set(graph["components"])
    assert selected.snapshot()["route_counts"] == {
        ("lexical_shared" if tied else "egress") + ":embed_rows:lm_head": 1
    }
    if tied:
        assert selected_head.equal(embedded)
        assert store.snapshot()["providers"]["lexical_shared"]["calls"] == {
            "embed_rows:embed": 1,
            "embed_rows:lm_head": 1,
        }
    else:
        assert not selected_head.equal(embedded)
        assert "egress" not in expected_hidden_roles


def test_fp32_extras_are_owned_and_writable_without_mutating_the_mmap(tmp_path: Path) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    store = CompositeQStore(graph_path)
    try:
        tensor = store.for_contract("hidden_state_only").fp32("norm.final")
        extras = store._providers["norm"].store.e
        mapped_before = np.array(extras, copy=True)

        assert not np.shares_memory(tensor.numpy(), extras)
        tensor.add_(1.0)
        np.testing.assert_array_equal(extras, mapped_before)
    finally:
        store.close()


def test_hidden_head_denial_precedes_provider_open_and_all_telemetry(
    tmp_path: Path,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    store = CompositeQStore(graph_path)
    hidden = store.for_contract("lexical_hidden")

    with pytest.raises(ComponentOutputContractError, match="denies embed_rows"):
        hidden.embed_rows("lm_head", np.asarray([0], dtype=np.int64))
    with pytest.raises(ComponentOutputContractError, match="denies row_blocks"):
        next(hidden.row_blocks("lm_head"))

    assert store.opened_roles == {"body"}
    assert hidden.snapshot()["route_counts"] == {}
    assert store.snapshot()["providers"]["body"]["calls"] == {}


def test_composite_exposes_full_logical_manifest_with_source_offsets(
    tmp_path: Path,
) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    store = CompositeQStore(graph_path)
    body_manifest = json.loads(
        (graph_path.parent / "components/body/manifest.json").read_text(encoding="utf-8")
    )

    assert store.man["blocks"] == graph["logical_blocks"]
    assert store.blocks == graph["logical_blocks"]
    assert store.man["blocks"]["L0.q"]["w_off"] == 8
    assert body_manifest["blocks"]["L0.q"]["w_off"] == 0
    assert store.man["blocks"]["lm_head"]["w_off"] == 12
    assert (
        store.man["component_graph"]["composite_fingerprint_sha256"]
        == store.composite_fingerprint_sha256
    )
    assert (
        store.man["component_graph"]["declared_legacy_fingerprint_sha256"]
        == graph["composite_fingerprint_sha256"]
    )
    assert store.man["derived"]["derived_store_sha256"] == store.composite_fingerprint_sha256
    assert store.composite_fingerprint_sha256 != graph["composite_fingerprint_sha256"]
    assert store.content_identity_verified is False


def test_component_cache_budgets_are_one_aggregate_cap(
    tmp_path: Path,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    initial = {"body": 0.4, "ingress": 0.2, "egress": 0.3, "norm": 0.1}
    store = CompositeQStore(
        graph_path,
        cache_mb=1.0,
        component_cache_mb=initial,
    )
    full = store.for_contract("full_logits")
    full.embed_rows("embed", np.asarray([0], dtype=np.int64))
    full.fp32("norm.final")
    list(full.row_blocks("lm_head", bs=2))

    snapshot = store.snapshot()
    budgets = snapshot["component_cache_budget_bytes"]
    assert budgets == {
        "body": 400_000,
        "egress": 300_000,
        "ingress": 200_000,
        "norm": 100_000,
    }
    assert sum(budgets.values()) == snapshot["cache_budget_bytes"] == 1_000_000
    assert {
        role: provider["cache_budget_bytes"] for role, provider in snapshot["providers"].items()
    } == budgets

    store.set_component_cache_budgets({"body": 0.1, "ingress": 0.1, "egress": 0.2, "norm": 0.1})
    updated = store.snapshot()
    assert sum(updated["component_cache_budget_bytes"].values()) == 500_000
    assert updated["cache_budget_bytes"] == 500_000

    with pytest.raises(ComponentGraphError, match="exceed the aggregate"):
        CompositeQStore(
            graph_path,
            cache_mb=0.25,
            component_cache_mb={"body": 0.2, "egress": 0.2},
        )


def test_dense_component_provider_uses_compact_lru_and_preserves_torch_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    store = CompositeQStore(
        graph_path,
        component_cache_mb={"body": 0.001, "ingress": 0.001},
        compute_dtype="fp32",
        provider_backend="dense-qstore-cuda",
        provider_device="cpu",
        provider_require_triton=False,
    )
    hidden = store.for_contract("hidden_state_only")

    original_asarray = np.asarray

    def reject_tensor_asarray(value: Any, *args: Any, **kwargs: Any):
        if isinstance(value, torch.Tensor):
            raise AssertionError("component routing converted a torch tensor through NumPy")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(composite_module.np, "asarray", reject_tensor_asarray)
    ids = torch.tensor([0, 2], dtype=torch.long)
    embedded = hidden.embed_rows("embed", ids)
    projected = hidden.matmul("L0.q", torch.tensor([[2.0, 1.0]]))

    torch.testing.assert_close(
        embedded,
        torch.tensor([[0.1, 0.2], [1.5, 1.8]]),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(projected, torch.tensor([[0.0, 1.0]]), rtol=0, atol=0)
    snapshot = store.snapshot()
    assert snapshot["provider_backend"] == "dense-qstore-cuda"
    assert snapshot["residency_contract"] == "component-compact-lru+streamed-exact-head"
    assert snapshot["body_fully_resident"] is False
    assert snapshot["body_required_physical_bytes"] == 20
    assert snapshot["body_resident_physical_bytes"] == 12
    assert snapshot["providers"]["body"]["provider"] == "dense-qstore-cuda"
    assert snapshot["providers"]["body"]["cache_kind"] == "compact-device-lru"
    assert snapshot["providers"]["body"]["cache_bytes"] == 12
    assert snapshot["providers"]["body"]["cache_budget_bytes"] == 1_000

    store.set_component_cache_budgets({"body": 0.0, "ingress": 0.0})
    assert store.snapshot()["providers"]["body"]["cache_bytes"] == 0
    with pytest.raises(ComponentOutputContractError, match="padded model rows are not tokens"):
        hidden.embed_rows("embed", torch.tensor([3], dtype=torch.long))
    store.close()


def test_dense_component_pins_aux_and_admits_resident_exact_head(
    tmp_path: Path,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    budgets = {role: 0.001 for role in ("body", "ingress", "egress", "norm")}
    resident_store = CompositeQStore(
        graph_path,
        component_cache_mb=budgets,
        compute_dtype="fp32",
        provider_backend="dense-qstore-cuda",
        provider_device="cpu",
        provider_require_triton=False,
        provider_pin_fp32_aux=True,
    )
    streamed_store = CompositeQStore(
        graph_path,
        component_cache_mb=budgets,
        compute_dtype="fp32",
        provider_backend="dense-qstore-cuda",
        provider_device="cpu",
        provider_require_triton=False,
        provider_pin_fp32_aux=True,
    )
    try:
        resident = resident_store.for_contract("full_logits")
        streamed = streamed_store.for_contract("full_logits")
        value = torch.tensor([[2.0, 1.0]])
        resident.matmul("L0.q", value)
        streamed.matmul("L0.q", value)

        body = resident_store.snapshot()["providers"]["body"]
        assert body["required_physical_bytes"] == 20
        assert body["resident_physical_bytes"] == 20
        assert body["auxiliary_fp32_fully_resident"] is True
        assert body["fully_resident"] is True
        with pytest.raises(MemoryError, match="exceeds its admission budget"):
            resident_store.prepare_resident_exact_head(0.00001)

        head = resident_store.prepare_resident_exact_head(0.001)
        expected_head = torch.tensor([[-0.4, -0.8], [-0.9, -1.2], [-1.0, -1.2], [-0.7, -0.8]])
        torch.testing.assert_close(head, expected_head, rtol=0, atol=1e-6)
        assert resident.resident_exact_head_fp32() is head

        from mrun.engine.dense_qstore_cuda import DenseQStoreTarget

        hidden = torch.tensor([[[0.25, -0.5], [1.0, 0.75]]])
        resident_target = DenseQStoreTarget(
            resident,
            semantic_token_count=3,
        )
        streamed_target = DenseQStoreTarget(
            streamed,
            semantic_token_count=3,
        )
        resident_top1, resident_logits = resident_target._head(hidden, return_logits=True)
        streamed_top1, streamed_logits = streamed_target._head(hidden, return_logits=True)
        assert torch.equal(resident_logits, streamed_logits)
        assert torch.equal(resident_top1, streamed_top1)
        assert resident_target.resident_exact_head_calls == 1
        assert resident_target.streamed_exact_head_calls == 0

        snapshot = resident_store.snapshot()
        assert snapshot["resident_exact_head"] is True
        assert snapshot["resident_exact_head_bytes"] == 32
        admission = snapshot["resident_exact_head_admission"]
        assert admission["accepted"] is True
        assert admission["head_required_bytes"] == 32
        assert admission["head_budget_bytes"] == 1_000
        assert admission["all_component_roles_fully_admitted"] is True
        residency_admission = snapshot["fully_resident_admission"]
        assert residency_admission["accepted"] is True
        assert set(snapshot["providers"]) == {"body", "ingress", "egress", "norm"}
        assert all(provider["fully_resident"] for provider in snapshot["providers"].values())
        assert snapshot["body_fully_resident"] is True
        assert snapshot["residency_contract"] == ("resident-body+resident-exact-fp32-head")
        egress = snapshot["providers"]["egress"]
        assert egress["resident_exact_head"] is True
        assert egress["resident_exact_head_bytes"] == 32

        before = resident_store.snapshot()["component_cache_budget_bytes"]
        with pytest.raises(MemoryError, match="cannot retain pinned FP32 auxiliaries"):
            resident_store.set_component_cache_budgets({"body": 0.000001})
        assert resident_store.snapshot()["component_cache_budget_bytes"] == before
    finally:
        resident_store.close()
        streamed_store.close()


def test_dense_component_aux_pinning_fails_before_over_budget_execution(
    tmp_path: Path,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    with pytest.raises(MemoryError, match="pinned FP32 auxiliary residency"):
        CompositeQStore(
            graph_path,
            component_cache_mb={"body": 0.000001},
            compute_dtype="fp32",
            provider_backend="dense-qstore-cuda",
            provider_device="cpu",
            provider_require_triton=False,
            provider_pin_fp32_aux=True,
        )


def test_composite_rejects_an_open_ended_provider_backend(tmp_path: Path) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    with pytest.raises(ValueError, match="provider_backend"):
        CompositeQStore(graph_path, provider_backend="arbitrary-python-factory")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_dense_component_provider_cuda_ids_and_projection(tmp_path: Path) -> None:
    pytest.importorskip("triton")
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    store = CompositeQStore(
        graph_path,
        component_cache_mb={"body": 0.001, "ingress": 0.001},
        compute_dtype="bf16",
        provider_backend="dense-qstore-cuda",
        provider_device="cuda:0",
        provider_require_triton=True,
    )
    try:
        hidden = store.for_contract("hidden_state_only")
        ids = torch.tensor([0, 2], device="cuda:0", dtype=torch.long)
        embedded = hidden.embed_rows("embed", ids)
        projected = hidden.matmul(
            "L0.q",
            torch.tensor([[2.0, 1.0]], device="cuda:0", dtype=torch.bfloat16),
        )

        assert embedded.device == ids.device
        torch.testing.assert_close(
            embedded.float().cpu(),
            torch.tensor([[0.1, 0.2], [1.5, 1.8]]),
            rtol=0,
            atol=0.01,
        )
        torch.testing.assert_close(
            projected.float().cpu(),
            torch.tensor([[0.0, 1.0]]),
            rtol=0,
            atol=0.01,
        )
    finally:
        store.close()


def test_lazy_component_provider_opens_exactly_once_under_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    original_qstore = composite_module.QStore
    opened: list[str] = []
    opened_lock = threading.Lock()

    def counted_qstore(model_name: str, **kwargs: Any):
        with opened_lock:
            opened.append(model_name)
        time.sleep(0.005)
        return original_qstore(model_name, **kwargs)

    monkeypatch.setattr(composite_module, "QStore", counted_qstore)
    store = CompositeQStore(graph_path)
    selected = store.for_contract("selected_rows")
    barrier = threading.Barrier(8)

    def read_head():
        barrier.wait()
        return selected.embed_rows("lm_head", np.asarray([0], dtype=np.int64))

    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(lambda _index: read_head(), range(8)))

    assert all(output.shape == (1, 2) for output in outputs)
    assert opened.count("egress") == 1
    assert store.opened_roles == {"body", "egress"}


def test_vocab_guards_reject_padded_rows_before_component_access(
    tmp_path: Path,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    store = CompositeQStore(graph_path)
    full = store.for_contract("full_logits")
    selected = store.for_contract("selected_rows")

    assert store.vocab.token_count == 3
    assert store.vocab.configured_row_count == 4
    with pytest.raises(ComponentOutputContractError, match="padded model rows are not tokens"):
        full.embed_rows("embed", np.asarray([3], dtype=np.int64))
    with pytest.raises(ComponentOutputContractError, match="padded model rows are not tokens"):
        selected.embed_rows("lm_head", np.asarray([3], dtype=np.int64))
    with pytest.raises(ComponentOutputContractError, match=r"must be in \[0, 3\)"):
        full.embed_rows("embed", np.asarray([-1], dtype=np.int64))

    assert store.opened_roles == {"body"}
    assert full.snapshot()["route_counts"] == {}
    assert selected.snapshot()["route_counts"] == {}


def test_runtime_tokenizer_must_match_ordered_vocabulary(tmp_path: Path) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=True)
    store = CompositeQStore(graph_path)

    store.validate_tokenizer(_TinyTokenizer())
    with pytest.raises(ComponentGraphError, match="runtime tokenizer does not match"):
        store.validate_tokenizer(_TinyTokenizer(("a", "c", "b")))


def test_graph_and_lazily_opened_blob_tampering_fail_closed(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path / "graph", tied=False)
    graph["model"] = "tampered-model"
    _write_json(graph_path, graph)
    with pytest.raises(ComponentGraphError, match="fingerprint mismatch"):
        CompositeQStore(graph_path)

    graph_path, _ = _build_component_graph(tmp_path / "blob", tied=False)
    store = CompositeQStore(graph_path)
    egress = graph_path.parent / "components/egress/weights.i8"
    payload = bytearray(egress.read_bytes())
    payload[0] ^= 1
    egress.write_bytes(payload)
    selected = store.for_contract("selected_rows")
    with pytest.raises(ComponentGraphError, match="egress/weights.i8 content hash mismatch"):
        selected.embed_rows("lm_head", np.asarray([0], dtype=np.int64))
    assert "egress" not in store.opened_roles


def test_updated_unbound_legacy_blob_record_changes_production_identity_and_fails_semantics(
    tmp_path: Path,
) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    original = CompositeQStore(graph_path).composite_fingerprint_sha256
    egress = graph_path.parent / "components/egress/weights.i8"
    payload = bytearray(egress.read_bytes())
    payload[0] ^= 1
    egress.write_bytes(payload)
    graph["components"]["egress"]["blobs"]["weights.i8"] = {
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }
    # Stage-4's declared digest does not cover the blobs map, so it is intentionally
    # left unchanged. Production must neither reuse its identity nor trust its semantic
    # claim after the payload record was rewritten.
    _write_json(graph_path, graph)

    rewritten = CompositeQStore(graph_path)
    assert rewritten.graph.declared_fingerprint == graph["composite_fingerprint_sha256"]
    assert rewritten.composite_fingerprint_sha256 != original
    with pytest.raises(ComponentGraphError, match="semantic content hash mismatch"):
        rewritten.for_contract("selected_rows").embed_rows(
            "lm_head", np.asarray([0], dtype=np.int64)
        )


def test_component_blob_table_requires_exact_fixed_payload_set(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    graph["components"]["egress"]["blobs"] = {
        "/outside/weights.i8": {
            "bytes": 1,
            "sha256": "0" * 64,
        }
    }
    _write_json(graph_path, graph)

    with pytest.raises(ComponentGraphError, match="must declare exactly"):
        CompositeQStore(graph_path)


def test_component_locator_cannot_escape_graph_root(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    graph["components"]["egress"]["relative_path"] = "../outside"
    # Locators are deliberately excluded from the relocatable semantic fingerprint.
    _write_json(graph_path, graph)

    with pytest.raises(ComponentGraphError, match="component locator"):
        CompositeQStore(graph_path)


def test_logical_descriptor_must_match_owned_physical_manifest(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    graph["logical_blocks"]["L0.q"]["shape"] = [4, 1]
    graph["body_abi"]["semantic_sha256"] = _body_abi_digest(graph)
    _resign_graph(graph)
    _write_json(graph_path, graph)

    with pytest.raises(ComponentGraphError, match="differs from its logical descriptor"):
        CompositeQStore(graph_path)


@pytest.mark.parametrize("attack", ["zero-length", "overlapping-allocation"])
def test_graph_rejects_logical_spans_that_undercount_memory(
    tmp_path: Path,
    attack: str,
) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    if attack == "zero-length":
        graph["logical_blocks"]["L0.q"]["w_len"] = 0
        graph["logical_blocks"]["L0.q"]["s_len"] = 0
        expected = "invalid weights.i8 span"
    else:
        # Untied embed and lm_head have identical extents. Reusing the ingress offsets
        # would make the memory planner's region set count both allocations only once.
        graph["logical_blocks"]["lm_head"]["w_off"] = 0
        graph["logical_blocks"]["lm_head"]["s_off"] = 0
        expected = "overlaps"
    # The legacy graph/ABI digests deliberately omit byte spans, which is the attack:
    # graph custody must validate the spans rather than trusting those older digests.
    _write_json(graph_path, graph)

    with pytest.raises(ComponentGraphError, match=expected):
        CompositeQStore(graph_path)


def test_component_layout_gap_is_rejected_even_when_manifest_is_resigned(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    manifest_path = graph_path.parent / "components/body/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["blocks"]["L0.q"]["w_off"] = 1
    _write_json(manifest_path, manifest)
    graph["components"]["body"]["manifest_sha256"] = _sha256(manifest_path.read_bytes())
    _resign_graph(graph)
    _write_json(graph_path, graph)

    with pytest.raises(ComponentGraphError, match="gap|exceeds file bounds"):
        CompositeQStore(graph_path)


def test_body_abi_digest_is_recomputed_from_authoritative_runtime_contract(
    tmp_path: Path,
) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    graph["body_abi"]["semantic_sha256"] = "f" * 64
    _resign_graph(graph)
    _write_json(graph_path, graph)

    with pytest.raises(ComponentGraphError, match="body ABI semantic hash"):
        CompositeQStore(graph_path)


def test_route_table_cannot_reassign_a_logical_block(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path, tied=False)
    graph["routes"]["lm_head"] = "ingress"
    _resign_graph(graph)
    _write_json(graph_path, graph)

    with pytest.raises(ComponentGraphError, match="routes and component ownership differ"):
        CompositeQStore(graph_path)


def test_relocation_and_json_key_order_do_not_change_graph_identity(tmp_path: Path) -> None:
    graph_path, graph = _build_component_graph(tmp_path / "source", tied=False)
    original = ComponentGraph(graph_path).fingerprint
    relocated_root = tmp_path / "relocated"
    shutil.copytree(graph_path.parent, relocated_root)
    relocated_graph = relocated_root / "model-graph.json"
    reordered = dict(reversed(tuple(graph.items())))
    relocated_graph.write_text(json.dumps(reordered), encoding="utf-8")

    assert ComponentGraph(relocated_graph).fingerprint == original


def test_composite_row_stable_matmul_routes_and_dequantizes_component_once(
    tmp_path: Path,
) -> None:
    graph_path, _ = _build_component_graph(tmp_path, tied=False)
    stable_store = CompositeQStore(graph_path)
    reference_store = CompositeQStore(graph_path)
    stable = stable_store.for_contract("hidden_state_only")
    reference = reference_store.for_contract("hidden_state_only")
    value = torch.tensor(
        [
            [[0.25, -0.5], [0.75, 1.0]],
            [[-1.5, 0.125], [0.5, -0.25]],
            [[2.0, -1.0], [-0.75, 0.625]],
        ],
        dtype=torch.float32,
    )

    observed = stable.matmul_row_stable("L0.q", value)
    expected = torch.cat(
        tuple(reference.matmul("L0.q", value[row : row + 1]) for row in range(int(value.shape[0]))),
        dim=0,
    )

    assert torch.equal(observed, expected)
    snapshot = stable_store.snapshot()
    assert snapshot["providers"]["body"]["calls"] == {"matmul_row_stable:L0.q": 1}
    assert snapshot["providers"]["body"]["rows_read"] == {"L0.q": 2}
    assert stable.snapshot()["route_counts"] == {"body:matmul_row_stable:L0.q": 1}
