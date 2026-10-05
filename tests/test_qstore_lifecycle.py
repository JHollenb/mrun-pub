from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mrun.engine.kernels.qstore import QStore, QStoreInt2, QStoreInt3, QStoreInt4

_LOW_BIT_STORES = (
    (QStoreInt2, "int2"),
    (QStoreInt3, "int3"),
    (QStoreInt4, "int4"),
)


def _write_low_bit_store(root: Path, dtype: str, *, rows: int = 1) -> Path:
    store_dir = root / f"toy-{dtype}"
    store_dir.mkdir()
    bits = int(dtype.removeprefix("int"))
    row_bytes = (4 * bits + 7) // 8
    (store_dir / f"weights.i{bits}").write_bytes(
        bytes((index % 251) + 1 for index in range(rows * row_bytes))
    )
    (store_dir / "scales.f32").write_bytes(np.ones(rows, dtype=np.float32).tobytes())
    (store_dir / "extras.f32").write_bytes(np.zeros(1, dtype=np.float32).tobytes())
    (store_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "toy",
                "arch": "qwen2",
                "dtype": dtype,
                "group_size": 4,
                "config": {},
                "blocks": {
                    "W": {
                        "kind": f"qrow{bits}",
                        "shape": [rows, 4],
                        "w_off": 0,
                        "row_bytes": row_bytes,
                        "s_off": 0,
                        "n_groups": 1,
                        "group_size": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return store_dir


@pytest.mark.parametrize(("store_type", "dtype"), _LOW_BIT_STORES)
def test_low_bit_store_guard_tracks_its_format_and_same_size_mutation(
    tmp_path: Path,
    store_type: type[QStore],
    dtype: str,
) -> None:
    store_dir = _write_low_bit_store(tmp_path, dtype)
    store = store_type("toy", root=tmp_path)
    bits = int(dtype.removeprefix("int"))
    weight_path = store_dir / f"weights.i{bits}"

    assert [record[0] for record in store._verified_file_stats] == [
        "manifest.json",
        f"weights.i{bits}",
        "scales.f32",
        "extras.f32",
    ]
    assert store.ring_stats() is None
    assert store.ring_allocated_bytes() == 0
    store.assert_content_identity_unchanged()

    before = weight_path.stat()
    payload = bytearray(weight_path.read_bytes())
    payload[0] ^= 1
    weight_path.write_bytes(payload)
    # Preserve size, inode, and mtime: the runtime guard must still observe the ctime
    # change rather than treating same-length replacement bytes as unchanged.
    os.utime(weight_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = weight_path.stat()
    assert after.st_size == before.st_size
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns

    with pytest.raises(RuntimeError, match="files changed after content verification"):
        store.assert_content_identity_unchanged()
    assert store.identity_status == "verified-store-files-changed"
    assert not store.content_identity_verified
    assert not store.store_identity["blob_identity_verified"]

    store.close()
    store.close()
    assert store.w is None and store.s is None and store.e is None


@pytest.mark.parametrize(("store_type", "dtype"), _LOW_BIT_STORES)
def test_low_bit_store_rejects_symlinked_payload_before_mapping(
    tmp_path: Path,
    store_type: type[QStore],
    dtype: str,
) -> None:
    store_dir = _write_low_bit_store(tmp_path, dtype)
    bits = int(dtype.removeprefix("int"))
    weight_path = store_dir / f"weights.i{bits}"
    external = tmp_path / f"external-weights.i{bits}"
    weight_path.replace(external)
    weight_path.symlink_to(external)

    with pytest.raises(RuntimeError, match=rf"must be regular, not a symlink: weights\.i{bits}"):
        store_type("toy", root=tmp_path)


@pytest.mark.parametrize(("store_type", "dtype"), _LOW_BIT_STORES)
def test_low_bit_store_invalidates_identity_after_payload_becomes_symlink(
    tmp_path: Path,
    store_type: type[QStore],
    dtype: str,
) -> None:
    store_dir = _write_low_bit_store(tmp_path, dtype)
    store = store_type("toy", root=tmp_path)
    bits = int(dtype.removeprefix("int"))
    weight_path = store_dir / f"weights.i{bits}"
    external = tmp_path / f"opened-external-weights.i{bits}"
    weight_path.replace(external)
    weight_path.symlink_to(external)

    with pytest.raises(RuntimeError, match="files changed after content verification"):
        store.assert_content_identity_unchanged()
    assert store.identity_status == "verified-store-files-changed"
    assert not store.content_identity_verified
    store.close()


def test_partial_qstore_lifecycle_without_ring_state_is_safe() -> None:
    store = QStoreInt2.__new__(QStoreInt2)

    assert store.ring_stats() is None
    assert store.ring_allocated_bytes() == 0
    store.disable_ring()
    store.close()
    store.close()
    assert store.w is None and store.s is None and store.e is None


@pytest.mark.parametrize(("store_type", "dtype"), _LOW_BIT_STORES)
def test_low_bit_row_stable_matmul_matches_independent_b1_and_format_memory_contract(
    tmp_path: Path,
    store_type: type[QStore],
    dtype: str,
) -> None:
    rows = 5
    _write_low_bit_store(tmp_path, dtype, rows=rows)
    store = store_type("toy", root=tmp_path)
    if dtype in {"int2", "int3"}:
        store.matmul_chunk_rows = 2
    value = torch.tensor(
        [
            [
                [0.25, -0.5, 0.75, 1.0],
                [-1.5, 0.125, 0.5, -0.25],
                [0.75, -0.875, 1.125, 0.375],
            ],
            [
                [2.0, -1.0, -0.75, 0.625],
                [0.5, 0.75, -1.25, 1.5],
                [-0.25, 1.75, 0.375, -0.625],
            ],
            [
                [-0.125, 0.25, 0.5, -0.75],
                [1.0, -1.0, 1.0, -1.0],
                [1.375, 0.625, -1.125, -0.375],
            ],
        ],
        dtype=torch.float32,
    )

    expected = torch.cat(
        tuple(store.matmul("W", value[row : row + 1]) for row in range(value.shape[0])),
        dim=0,
    )
    observed = store.matmul_row_stable("W", value)

    assert torch.equal(observed, expected)
    assert tuple(observed.shape) == (3, 3, rows)
    row_bytes = 4 * torch.empty((), dtype=torch.float32).element_size()
    if dtype == "int4":
        # Int4 serial/base materializes the complete matrix; row-stable preserves that contract.
        assert store.max_block_bytes == rows * row_bytes
    else:
        # Int3/Int2 remain bounded by the configured two-output-row dequant chunk.
        assert store.max_block_bytes <= 2 * row_bytes
    store.close()
