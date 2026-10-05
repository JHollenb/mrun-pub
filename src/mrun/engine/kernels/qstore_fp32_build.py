"""Build a lossless, full-precision paged QStore.

This is an explicit scientific-reference lane.  Unlike the default row-int8 QStore, every
matrix is stored as float32.  Float16, bfloat16, and float32 checkpoint values all have an
exact representation in float32, so the store changes neither the represented source values
nor their canonical block mapping.  The builder still streams one safetensors tensor at a
time and the reader materializes one matrix (or one output-head row chunk) at a time.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ...models import find_safetensors, resolve_model, store_name
from ...paths import stores_root
from ...store_provenance import (
    build_builder_provenance,
    build_derived_provenance,
    build_source_provenance,
    verify_builder_provenance,
    verify_derived_provenance,
    verify_source_provenance,
)
from .qstore_build import LexicalWeightBinding, _arch_config, _canon, _is_fp32_block, rss_mb

QSTORE_FP32_SCHEMA = "mrun-qstore-fp32-v1"
QSTORE_FP32_FILES = ("weights.f32", "extras.f32")
QSTORE_FP32_STORAGE = {
    "codec": "lossless-float32",
    "stored_dtype": "float32",
    "byte_order": "little-endian",
    "accepted_source_dtypes": ["bfloat16", "float16", "float32"],
    "conversion": "exact-widen-or-identity",
    "full_precision_blocks": "float32",
}
_SUPPORTED_ARCHITECTURES = (
    "qwen2",
    "llama",
    "qwen3",
    "qwen3_5",
    "qwen3_5_text",
    "gpt_neox",
    "mamba",
)
_EXACT_SOURCE_DTYPES = frozenset(QSTORE_FP32_STORAGE["accepted_source_dtypes"])


def _semantic_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "derived"}


def _verify_existing_store(
    out: Path,
    *,
    model_name: str,
    arch: str,
    config: dict[str, Any],
    source: dict[str, Any],
    builder: dict[str, Any],
) -> Path:
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"refusing to overwrite incomplete FP32 QStore directory: {out}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": QSTORE_FP32_SCHEMA,
        "model_name": model_name,
        "arch": arch,
        "dtype": "float32",
        "storage_contract": QSTORE_FP32_STORAGE,
        "config": config,
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"existing FP32 QStore identity/config mismatch: {actual} != {expected}")
    verify_source_provenance(manifest.get("source"), source)
    verify_builder_provenance(manifest.get("builder"), builder)
    verify_derived_provenance(
        out,
        manifest.get("derived"),
        expected_filenames=QSTORE_FP32_FILES,
        semantic_manifest=_semantic_manifest(manifest),
    )

    blocks = manifest.get("blocks")
    if not isinstance(blocks, dict):
        raise RuntimeError("existing FP32 QStore has no block table")
    rows = [block for block in blocks.values() if block.get("kind") == "f32row"]
    extras = [block for block in blocks.values() if block.get("kind") == "fp32"]
    if not rows or not extras:
        raise RuntimeError("existing FP32 QStore block table is incomplete")
    boundaries = {
        "weights.f32": max(block["w_off"] + block["w_len"] for block in rows),
        "extras.f32": max(block["e_off"] + block["e_len"] for block in extras),
    }
    actual_sizes = {name: (out / name).stat().st_size for name in QSTORE_FP32_FILES}
    if boundaries != actual_sizes:
        raise RuntimeError(f"existing FP32 QStore block/file boundary mismatch: {boundaries}")
    return out


def build(
    model_name: str,
    *,
    out_root: Path | None = None,
    store_dir_name: str | None = None,
) -> Path:
    """Build ``<store>-fp32`` without changing the default row-int8 store."""

    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    files = find_safetensors(model_name)
    if not files:
        raise FileNotFoundError(f"no safetensors for {model_name!r}")
    spec = resolve_model(model_name)
    snapshot = files[0].parent
    auto_config = AutoConfig.from_pretrained(str(snapshot))
    raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    arch = raw.get("model_type", getattr(auto_config, "model_type", "?"))
    if arch not in _SUPPORTED_ARCHITECTURES:
        supported = "/".join(_SUPPORTED_ARCHITECTURES)
        raise NotImplementedError(f"arch {arch!r} not supported ({supported})")

    source = build_source_provenance(
        snapshot,
        files,
        model_name=spec.name,
        hf_id=spec.hf_id,
    )
    builder = build_builder_provenance(
        [Path(__file__), Path(__file__).with_name("qstore_build.py")],
        name="mrun.engine.kernels.qstore_fp32_build",
        schema_version=QSTORE_FP32_SCHEMA,
        quantization=QSTORE_FP32_STORAGE,
    )
    config = _arch_config(arch, raw)
    root = Path(out_root) if out_root is not None else stores_root()
    root.mkdir(parents=True, exist_ok=True)
    directory_name = store_dir_name or store_name(model_name)
    out = root / f"{directory_name}-fp32"
    if out.exists():
        return _verify_existing_store(
            out,
            model_name=spec.name,
            arch=arch,
            config=config,
            source=source,
            builder=builder,
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{directory_name}-fp32.building-", dir=root))
    weight_path = temporary / "weights.f32"
    extras_path = temporary / "extras.f32"
    weight_offset = extras_offset = 0
    blocks: dict[str, dict[str, Any]] = {}
    seen_blocks: set[str] = set()
    lexical_binding = LexicalWeightBinding(raw)
    source_dtypes: set[str] = set()
    matrix_count = extras_count = 0

    print(f">>> fp32 qstore build — {model_name}  arch={arch}  -> {out}")
    try:
        with weight_path.open("wb") as weight_file, extras_path.open("wb") as extras_file:
            for source_file in sorted(files):
                with safe_open(str(source_file), framework="pt") as tensors:
                    for key in tensors.keys():
                        name = _canon(key, arch)
                        if name is None:
                            continue
                        if name in seen_blocks:
                            raise RuntimeError(f"duplicate canonical FP32 QStore block {name!r}")
                        seen_blocks.add(name)
                        tensor = tensors.get_tensor(key)
                        lexical_binding.observe_source(name, tensor)
                        source_dtype = str(tensor.dtype).removeprefix("torch.")
                        if source_dtype not in _EXACT_SOURCE_DTYPES:
                            raise ValueError(
                                f"{key}: source dtype {source_dtype!r} cannot be proven "
                                "lossless under the FP32 storage contract"
                            )
                        source_dtypes.add(source_dtype)
                        values = np.ascontiguousarray(
                            tensor.to(dtype=torch.float32).numpy(),
                            dtype=np.dtype("<f4"),
                        )
                        del tensor
                        if _is_fp32_block(name):
                            extras_file.write(values.tobytes())
                            blocks[name] = {
                                "kind": "fp32",
                                "shape": list(values.shape),
                                "e_off": extras_offset,
                                "e_len": values.nbytes,
                            }
                            extras_offset += values.nbytes
                            extras_count += 1
                        else:
                            if values.ndim != 2:
                                raise ValueError(f"{name}: expected 2D, got {values.shape}")
                            if lexical_binding.observe_encoded(name, weights=values):
                                weight_file.write(values.tobytes())
                                blocks[name] = {
                                    "kind": "f32row",
                                    "shape": list(values.shape),
                                    "w_off": weight_offset,
                                    "w_len": values.nbytes,
                                }
                                weight_offset += values.nbytes
                                matrix_count += 1
                        del values
                print(f"    {source_file.name}: blocks={len(blocks)}  RSS={rss_mb():.0f}MB")

        lexical_manifest = lexical_binding.finalize(blocks)
        tied = lexical_binding.declared_tied
        manifest: dict[str, Any] = {
            "schema_version": QSTORE_FP32_SCHEMA,
            "model_name": spec.name,
            "arch": arch,
            "dtype": "float32",
            "storage_contract": QSTORE_FP32_STORAGE,
            "source_dtypes": sorted(source_dtypes),
            "tie_word_embeddings": tied,
            "lexical_weight_binding": lexical_manifest,
            "config": config,
            "source": source,
            "builder": builder,
            "blocks": blocks,
        }
        manifest["derived"] = build_derived_provenance(
            temporary,
            QSTORE_FP32_FILES,
            semantic_manifest=manifest,
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(out)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    disk_mb = sum((out / name).stat().st_size for name in QSTORE_FP32_FILES) / 1e6
    print(
        f"  fp32_matrices={matrix_count} fp32_extras={extras_count} "
        f"store={disk_mb:.1f}MB peakRSS={rss_mb():.0f}MB"
    )
    print(f"  -> {out}")
    return out
