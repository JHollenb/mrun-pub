from __future__ import annotations

from types import SimpleNamespace

from mrun.catalog import artifact_from_weights, scan_store_artifacts
from mrun.server.db import DB


def test_scan_store_artifacts_discovers_suffixed_variant_and_manifest_identity(tmp_path):
    root = tmp_path / "qstores"
    store = root / "Qwen3-30B-A3B-fp8-paged-v1"
    store.mkdir(parents=True)
    (store / "weights.i8").write_bytes(b"weights")
    (store / "manifest.json").write_text(
        '{"schema_version":"mrun-qstore-v1",'
        '"model_name":"qwen3-30b-a3b", "dtype":"int8",'
        '"derived":{"derived_store_sha256":"%s"}}\n' % ("a" * 64)
    )

    registry = {
        "qwen3-30b-a3b": SimpleNamespace(
            name="qwen3-30b-a3b", hf_id="Qwen/Qwen3-30B-A3B"
        )
    }
    rows = scan_store_artifacts(
        root,
        registry=registry,
        mounts=["/", str(tmp_path)],
        host="beast",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.model == "qwen3-30b-a3b"
    assert row.kind == "qstore"
    assert row.artifact_kind == "qstore"
    assert row.variant == "Qwen3-30B-A3B-fp8-paged-v1"
    assert row.mount == str(tmp_path)
    assert row.content_hash == "a" * 64
    assert row.artifact_id == "artifact:sha256:" + "a" * 64
    assert row.as_dict()["locator"] == {
        "host": "beast",
        "mount": str(tmp_path),
        "path": str(store),
    }


def test_weight_inventory_identity_does_not_hash_weight_contents(tmp_path):
    first = tmp_path / "model-00001-of-00001.safetensors"
    first.write_bytes(b"one")
    row = artifact_from_weights([first], model="toy", mounts=[str(tmp_path)])

    assert row is not None
    assert row.kind == "weights"
    assert row.artifact_kind == "hf-weights"
    assert row.bytes == 3
    assert row.content_hash is None
    assert row.artifact_id.startswith("artifact:inventory-sha256:")


def test_inventory_persists_multiple_qstore_variants_and_locator(tmp_path):
    db = DB(tmp_path / "mrun.db")
    db.replace_inventory(
        "beast",
        [
            {
                "model": "qwen3-30b-a3b",
                "kind": "qstore",
                "variant": "fp8-paged-v1",
                "artifact_id": "artifact:fp8",
                "artifact_kind": "qstore",
                "bytes": 10,
                "path": "/mnt/big/qstores/fp8-paged-v1",
                "mount": "/mnt/big",
                "locator": {
                    "host": "beast",
                    "mount": "/mnt/big",
                    "path": "/mnt/big/qstores/fp8-paged-v1",
                },
            },
            {
                "model": "qwen3-30b-a3b",
                "kind": "qstore",
                "variant": "w4-paged-v1",
                "artifact_id": "artifact:w4",
                "artifact_kind": "qstore",
                "bytes": 8,
                "path": "/mnt/big/qstores/w4-paged-v1",
                "mount": "/mnt/big",
            },
        ],
    )

    rows = db.inventory("beast")
    assert {row["variant"] for row in rows} == {"fp8-paged-v1", "w4-paged-v1"}
    assert {row["kind"] for row in rows} == {"qstore:fp8-paged-v1", "qstore:w4-paged-v1"}
    assert rows[0]["locator"]["mount"] == "/mnt/big"
    assert set(db.hosts_with_model("qwen3-30b-a3b")["beast"]) == {
        "qstore:fp8-paged-v1",
        "qstore:w4-paged-v1",
    }
    assert set(db.hosts_with_artifacts("qwen3-30b-a3b")["beast"]) == {
        "qstore:fp8-paged-v1",
        "qstore:w4-paged-v1",
        "artifact:fp8",
        "artifact:w4",
    }
