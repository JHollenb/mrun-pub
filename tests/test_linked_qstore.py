from __future__ import annotations

import numpy as np
import torch

import mrun.engine.paged as paged_module
from mrun.engine.kernels.linked_qstore import (
    LinkedQStoreError,
    OverlayQStore,
    ResolvedLinkedQStore,
)


class _FakeStore:
    def __init__(self, label: str, values: dict[str, float]) -> None:
        self.label = label
        self.values = dict(values)
        self.calls: list[tuple[str, str]] = []
        self.closed = False
        self.device = "cpu"
        self.cfg = {"label": label}
        self.man = {"dtype": "float32", "arch": "qwen2"}
        self.storage_dtype = "fp32"
        self.compute_dtype = torch.float32
        self.max_block_bytes = 64
        self.directory = "."
        self.store_identity = {"content_identity_verified": True, "label": label}
        self.source_checkpoint_sha256 = f"source-{label}"

    def has(self, name: str) -> bool:
        return name in self.values

    def _value(self, method: str, name: str) -> torch.Tensor:
        self.calls.append((method, name))
        return torch.tensor([[self.values[name]]], dtype=torch.float32)

    def fp32(self, name: str) -> torch.Tensor:
        return self._value("fp32", name)

    def weight(self, name: str) -> torch.Tensor:
        return self._value("weight", name)

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return value + self._value("matmul", name)

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return value + self._value("matmul_row_stable", name)

    def embed_rows(self, name: str, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        del ids
        return self._value("embed_rows", name)

    def row_blocks(self, name: str, bs: int = 8192):
        del bs
        yield 0, 1, self._value("row_blocks", name)

    def set_cache_budget(self, cache_mb: float) -> None:
        self.calls.append(("set_cache_budget", str(cache_mb)))

    def snapshot(self):
        return {"label": self.label}

    def close(self) -> None:
        self.closed = True


class _ResolvedFakeStore(_FakeStore):
    def __init__(self, label: str, values: dict[str, float], blocks=None) -> None:
        super().__init__(label, values)
        self.cfg = {"model": "toy", "hidden_size": 1}
        self.blocks = {
            name: {"kind": "qrow", "shape": [1, 1]}
            for name in values
        }
        if blocks:
            self.blocks.update(blocks)
        self.cache_budgets: list[float] = []

    def has(self, name: str) -> bool:
        return name in self.blocks

    def set_cache_budget(self, cache_mb: float) -> None:
        self.cache_budgets.append(float(cache_mb))


def test_overlay_dispatches_extension_and_base_without_extra_math():
    base = _FakeStore("base", {"shared": 1.0, "base_only": 2.0})
    extension = _FakeStore("extension", {"shared": 9.0, "extension_only": 3.0})
    overlay = OverlayQStore(
        base,
        extension,
        {"shared"},
        image_id="base-hash:extension-id:int8",
    )

    assert overlay.weight("shared").item() == 9.0
    assert overlay.weight("base_only").item() == 2.0
    assert overlay.matmul("shared", torch.zeros(1, 1)).item() == 9.0
    assert overlay.snapshot()["image_id"] == "base-hash:extension-id:int8"
    overlay.close()
    assert base.closed is True
    assert extension.closed is True


def test_overlay_rejects_missing_extension_block():
    base = _FakeStore("base", {"shared": 1.0})
    extension = _FakeStore("extension", {})
    try:
        OverlayQStore(base, extension, {"shared"}, image_id="bad")
    except LinkedQStoreError as exc:
        assert "shared" in str(exc)
    else:  # pragma: no cover - assertion is the test
        raise AssertionError("missing extension block was accepted")


def test_resolved_view_prebinds_routes_and_closes_aliases_to_extension():
    base = _ResolvedFakeStore(
        "base",
        {"embed": 1.0, "shared": 2.0, "base_only": 3.0},
        {"lm_head": {"alias": "embed"}},
    )
    extension = _ResolvedFakeStore("extension", {"embed": 9.0, "shared": 8.0})
    view = ResolvedLinkedQStore(
        base,
        extension,
        {"embed", "shared"},
        image_id="toy-v1",
    )

    assert view.weight("shared").item() == 8.0
    assert view.weight("base_only").item() == 3.0
    assert view.weight("lm_head").item() == 9.0
    assert view._routes["lm_head"].provider is extension
    assert view._routes["lm_head"].physical == "embed"
    assert view.store_identity["extension_route_count"] == 3
    assert view.man["linked_image"]["runtime"]["extra_forward_ops"] == 0
    assert [call for call in extension.calls if call[0] == "weight"] == [
        ("weight", "shared"),
        ("weight", "embed"),
    ]
    view.close()
    assert base.closed is True
    assert extension.closed is True


def test_resolved_view_partitions_one_cache_budget_by_routed_physical_pages():
    base = _ResolvedFakeStore(
        "base",
        {"embed": 1.0, "base_only": 3.0, "base_other": 4.0},
        {"lm_head": {"alias": "embed"}},
    )
    extension = _ResolvedFakeStore("extension", {"embed": 9.0})
    view = ResolvedLinkedQStore(
        base,
        extension,
        {"embed"},
        image_id="toy-v1",
        cache_mb=12.0,
    )

    assert base.cache_budgets == [8.0]
    assert extension.cache_budgets == [4.0]
    assert sum(base.cache_budgets + extension.cache_budgets) == 12.0
    assert view.snapshot()["provider_cache_budgets_mb"] == {"base": 8.0, "extension": 4.0}
    view.close()


def test_resolved_view_rejects_replacement_shape_mismatch():
    base = _ResolvedFakeStore("base", {"shared": 1.0})
    extension = _ResolvedFakeStore(
        "extension",
        {"shared": 9.0},
        {"shared": {"kind": "qrow", "shape": [2, 1]}},
    )
    try:
        ResolvedLinkedQStore(base, extension, {"shared"}, image_id="bad")
    except LinkedQStoreError as exc:
        assert "shape" in str(exc)
    else:  # pragma: no cover - assertion is the test
        raise AssertionError("incompatible overlay shape was accepted")


def test_paged_engine_selects_and_validates_explicit_linked_image(tmp_path, monkeypatch):
    image = tmp_path / "linked-image"
    image.mkdir()
    (image / "manifest.json").write_text(
        '{"model_name":"qwen2.5-0.5b",'
        '"linked_image":{"extension_id":"toy-linked-v1"}}'
    )

    class _Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __len__(self):
            return 4

    class _Store:
        directory = image
        device = "cpu"
        man = {
            "model_name": "qwen2.5-0.5b",
            "arch": "qwen2",
            "linked_image": {"extension_id": "toy-linked-v1"},
        }
        cfg = {
            "vocab_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 2,
            "hidden_size": 2,
        }

        def close(self):
            self.closed = True

    opened = []

    def _open_store(key, *, root, cache_mb):
        assert key == image.name
        assert root == image.parent
        assert cache_mb == 0.0
        store = _Store()
        store.closed = False
        opened.append(store)
        return store

    monkeypatch.setattr(paged_module, "QStore", _open_store)
    monkeypatch.setattr(paged_module, "load_tokenizer", lambda _spec: _Tokenizer())

    discovered = paged_module.PagedEngine(
        "qwen2.5-0.5b",
        stores_dir=tmp_path,
        linked_extension_id="toy-linked-v1",
        cache_mb=0.0,
    )
    assert discovered.store_path == image.resolve()
    assert discovered.linked_extension_id == "toy-linked-v1"
    discovered.close()
    assert opened[0].closed is True

    engine = paged_module.PagedEngine(
        "qwen2.5-0.5b",
        store_path=image,
        linked_extension_id="toy-linked-v1",
        cache_mb=0.0,
    )
    assert engine.store_path == image.resolve()
    assert engine.linked_extension_id == "toy-linked-v1"
    engine.close()
    assert opened[1].closed is True

    try:
        paged_module.PagedEngine(
            "qwen2.5-0.5b",
            store_path=image,
            linked_extension_id="wrong-id",
            cache_mb=0.0,
        )
    except ValueError as exc:
        assert "does not match requested" in str(exc)
    else:  # pragma: no cover - assertion is the test
        raise AssertionError("linked extension identity mismatch was accepted")
    assert opened[2].closed is True


def test_paged_engine_binds_sparse_extension_to_one_base_runtime(tmp_path, monkeypatch):
    base_dir = tmp_path / "qwen2.5-0.5b"
    extension_dir = tmp_path / "toy-extension"
    base_dir.mkdir()
    extension_dir.mkdir()
    (base_dir / "manifest.json").write_text(
        '{"model_name":"qwen2.5-0.5b","arch":"qwen2"}'
    )
    (extension_dir / "manifest.json").write_text(
        '{"model_name":"qwen2.5-0.5b","arch":"qwen2",'
        '"linked_image":{"extension_id":"toy-linked-v2",'
        '"overlay_blocks":["shared"]}}'
    )

    class _Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __len__(self):
            return 4

    base = _ResolvedFakeStore("base", {"shared": 1.0, "base_only": 2.0})
    base.directory = base_dir
    base.cfg.update(
        {
            "vocab_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 2,
            "hidden_size": 2,
        }
    )
    base.man.update({"model_name": "qwen2.5-0.5b", "arch": "qwen2"})
    extension = _ResolvedFakeStore("extension", {"shared": 9.0})
    extension.directory = extension_dir
    extension.cfg = dict(base.cfg)
    extension.man.update(
        {
            "model_name": "qwen2.5-0.5b",
            "arch": "qwen2",
            "linked_image": {
                "extension_id": "toy-linked-v2",
                "overlay_blocks": ["shared"],
            },
        }
    )
    opened = []

    def _open_store(key, *, root, cache_mb):
        assert root == tmp_path
        assert cache_mb == 0.0
        del key
        store = base if not opened else extension
        opened.append(store)
        return store

    monkeypatch.setattr(paged_module, "QStore", _open_store)
    monkeypatch.setattr(paged_module, "load_tokenizer", lambda _spec: _Tokenizer())

    engine = paged_module.PagedEngine(
        "qwen2.5-0.5b",
        stores_dir=tmp_path,
        linked_extension_store_path=extension_dir,
        linked_extension_id="toy-linked-v2",
        cache_mb=12.0,
    )
    assert isinstance(engine.store, ResolvedLinkedQStore)
    assert engine.store_path == base_dir.resolve()
    assert engine.linked_extension_store_path == extension_dir.resolve()
    assert engine.linked_extension_id == "toy-linked-v2"
    assert engine.store.weight("shared").item() == 9.0
    assert engine.store.weight("base_only").item() == 2.0
    engine.close()
    assert opened == [base, extension]
    assert base.closed is True
    assert extension.closed is True
