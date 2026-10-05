"""Local model manifests without tracking services or remote storage."""
from __future__ import annotations
from pathlib import Path
from typing import Any

_MODEL_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_base_model_manifest(
    checkpoint_dir: str | Path,
    *,
    name: str,
    hf_id: str,
    family: str,
    revision: str = "main",
) -> dict[str, Any]:
    """Build a portable, deterministic identity manifest for an HF base checkpoint.

    Weight shards are deliberately not re-hashed: for frontier checkpoints that would add
    hundreds of GB of I/O. The manifest hashes the authoritative safetensors index and model
    metadata, records every expected shard's exact byte size, and validates each safetensors
    header against the physical file length. Missing, empty, or truncated shards fail loudly.
    ``content_hash`` excludes the host-local path so mirrors have the same portable identity.
    """
    import hashlib
    import json

    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"base model directory does not exist: {root}")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"base model has no config.json: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    index_path = root / "model.safetensors.index.json"
    index: dict[str, Any] | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid or empty weight_map in {index_path}")
        shard_names = sorted({str(value) for value in weight_map.values()})
    else:
        shard_names = sorted(path.name for path in root.glob("*.safetensors") if path.is_file())
    if not shard_names:
        raise FileNotFoundError(f"base model has no safetensors shards: {root}")

    missing = [filename for filename in shard_names if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"base model is missing {len(missing)} expected shard(s): {', '.join(missing[:8])}"
        )
    empty = [filename for filename in shard_names if (root / filename).stat().st_size <= 0]
    if empty:
        raise ValueError(f"base model has zero-byte shard(s): {', '.join(empty[:8])}")
    from safetensors import safe_open

    invalid = []
    for filename in shard_names:
        try:
            with safe_open(str(root / filename), framework="pt") as handle:
                list(handle.keys())
        except Exception as error:  # noqa: BLE001
            invalid.append(f"{filename} ({error})")
    if invalid:
        raise ValueError(f"base model has invalid safetensors shard(s): {', '.join(invalid[:4])}")

    shards = [
        {"name": filename, "bytes": (root / filename).stat().st_size} for filename in shard_names
    ]
    metadata = []
    for filename in _MODEL_IDENTITY_FILES:
        path = root / filename
        if path.is_file():
            metadata.append(
                {"name": filename, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            )

    expert_keys = (
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "num_selected_experts",
        "n_routed_experts",
        "n_shared_experts",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "routed_scaling_factor",
        "norm_topk_prob",
    )
    experts = {key: config[key] for key in expert_keys if key in config}
    model_type = str(config.get("model_type", ""))
    is_moe = bool(experts) or "moe" in model_type or model_type.startswith("deepseek")
    architecture = {
        key: config[key]
        for key in (
            "architectures",
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "torch_dtype",
            "quantization_config",
        )
        if key in config
    }
    transient_files = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and (".incomplete" in path.name or path.name.startswith("._"))
    )
    identity = {
        "schema_version": 1,
        "kind": "hf-base-model-reference",
        "model": {
            "name": name,
            "hf_id": hf_id,
            "family": family,
            "revision": revision,
            "architecture_kind": "moe" if is_moe else "dense",
        },
        "architecture": architecture,
        "experts": experts,
        "weights": {
            "format": "safetensors",
            "sharded": index is not None,
            "shard_count": len(shards),
            "total_bytes": sum(item["bytes"] for item in shards),
            "index_total_size": (index or {}).get("metadata", {}).get("total_size"),
            "shards": shards,
        },
        "metadata_files": metadata,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = dict(identity)
    manifest["content_hash"] = hashlib.sha256(canonical).hexdigest()
    manifest["validation"] = {
        "weights_complete": True,
        "safetensors_headers_valid": True,
        "transient_files_ignored": transient_files,
    }
    return manifest
