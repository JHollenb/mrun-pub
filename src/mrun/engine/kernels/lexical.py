"""Portable lexical artifacts for composable QStore runtimes.

The transformer body never needs to know how token IDs were produced.  This module supports both
the backward-compatible combined component and the separated ``values + weights + binding``
artifacts.  Each artifact is an ordinary directory containing JSON metadata, NumPy matrices, and
when applicable a Hugging Face tokenizer directory, making the pieces easy to inspect, hash, copy,
and share between several body stores.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .composite_qstore import VocabManifest, tokenizer_descriptor

LEXICAL_SCHEMA = "mrun-lexical-component-v1"
LEXICAL_VALUES_SCHEMA = "mrun-lexical-values-v1"
LEXICAL_WEIGHTS_SCHEMA = "mrun-lexical-weights-v1"
LEXICAL_BINDING_SCHEMA = "mrun-lexical-binding-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def body_abi(config: Mapping[str, Any], architecture: str) -> dict[str, Any]:
    """Return the body contract, intentionally excluding vocabulary and tokenizer fields."""

    structural = {
        str(key): config[key]
        for key in (
            "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta", "max_position_embeddings",
        )
        if key in config
    }
    payload = {"architecture": str(architecture), "config": structural, "runtime_abi": "mrun-lexical-body-v1"}
    return {**payload, "semantic_sha256": _sha256_bytes(_canonical(payload))}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        files[str(path.relative_to(root))] = _file_sha256(path)
    return files


def _read_manifest(root: Path, schema: str, label: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{label} has no manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != schema:
        raise ValueError(f"unsupported {label} schema")
    claimed = manifest.get("semantic_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    if claimed != _sha256_bytes(_canonical(unsigned)):
        raise ValueError(f"{label} semantic fingerprint mismatch")
    return manifest


def _write_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = {**manifest, "semantic_sha256": _sha256_bytes(_canonical(manifest))}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_tokenizer(root: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(root, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot load lexical tokenizer from {root}") from exc


class LexicalValues:
    """Tokenizer and vocabulary identity, with no model parameters."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], tokenizer: Any) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.tokenizer = tokenizer
        self.token_count = int(self.manifest["token_count"])

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["semantic_sha256"])

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        tokenizer: Any,
        provenance: Mapping[str, Any] | None = None,
    ) -> LexicalValues:
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        tokenizer_root = root / "tokenizer"
        if tokenizer_root.exists():
            shutil.rmtree(tokenizer_root)
        tokenizer.save_pretrained(tokenizer_root)
        descriptor = tokenizer_descriptor(tokenizer)
        vocab = VocabManifest.from_descriptor(
            descriptor, configured_row_count=len(tokenizer)
        )
        manifest = _write_manifest(
            root,
            {
                "schema_version": LEXICAL_VALUES_SCHEMA,
                "tokenizer": descriptor,
                "vocab_manifest": vocab.as_dict(),
                "token_count": len(tokenizer),
                "files": {"tokenizer": "tokenizer"},
                "file_sha256": _directory_hashes(tokenizer_root),
                "provenance": dict(provenance or {}),
            },
        )
        return cls(root, manifest, tokenizer)

    @classmethod
    def load(cls, root: str | Path) -> LexicalValues:
        root = Path(root).expanduser().resolve()
        manifest = _read_manifest(root, LEXICAL_VALUES_SCHEMA, "lexical values")
        tokenizer_root = root / str(manifest["files"]["tokenizer"])
        if _directory_hashes(tokenizer_root) != manifest["file_sha256"]:
            raise ValueError("lexical tokenizer files are missing or changed")
        tokenizer = _load_tokenizer(tokenizer_root)
        descriptor = tokenizer_descriptor(tokenizer)
        if descriptor != manifest["tokenizer"]:
            raise ValueError("lexical values tokenizer semantic fingerprint mismatch")
        if len(tokenizer) != int(manifest["token_count"]):
            raise ValueError("lexical values token count mismatch")
        vocab = VocabManifest.from_descriptor(
            descriptor, configured_row_count=int(manifest["token_count"])
        )
        if vocab.semantic_sha256 != manifest["vocab_manifest"]["semantic_sha256"]:
            raise ValueError("lexical values vocabulary manifest mismatch")
        return cls(root, manifest, tokenizer)

    def assert_content_identity_unchanged(self) -> None:
        current = _read_manifest(self.root, LEXICAL_VALUES_SCHEMA, "lexical values")
        if current["semantic_sha256"] != self.semantic_sha256:
            raise RuntimeError("lexical values manifest changed after validation")
        if _directory_hashes(self.root / str(current["files"]["tokenizer"])) != current[
            "file_sha256"
        ]:
            raise RuntimeError("lexical tokenizer files changed after validation")


class LexicalWeights:
    """Input/output lexical matrices, independent of tokenizer files."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        embed: np.ndarray,
        lm_head: np.ndarray,
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.embed = embed
        self.lm_head = lm_head
        self.tied = bool(self.manifest["tied"])
        self.input_row_count = int(self.manifest["input_row_count"])
        self.output_row_count = int(self.manifest["output_row_count"])
        if self.embed.ndim != 2 or self.lm_head.ndim != 2:
            raise ValueError("lexical weight matrices must be rank-2")
        if self.embed.shape != (self.input_row_count, int(self.manifest["hidden_size"])):
            raise ValueError("lexical input weight shape disagrees with manifest")
        if self.lm_head.shape != (self.output_row_count, int(self.manifest["hidden_size"])):
            raise ValueError("lexical output weight shape disagrees with manifest")
        if self.tied and not np.array_equal(self.embed, self.lm_head):
            raise ValueError("tied lexical weights must use one identical matrix")

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["semantic_sha256"])

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        embed: torch.Tensor | np.ndarray,
        lm_head: torch.Tensor | np.ndarray | None,
        body_config: Mapping[str, Any],
        architecture: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> LexicalWeights:
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        embed_np = np.asarray(
            embed.detach().cpu() if isinstance(embed, torch.Tensor) else embed,
            dtype=np.float32,
        )
        head_np = embed_np if lm_head is None else np.asarray(
            lm_head.detach().cpu() if isinstance(lm_head, torch.Tensor) else lm_head,
            dtype=np.float32,
        )
        if embed_np.ndim != 2 or head_np.ndim != 2:
            raise ValueError("lexical weight matrices must be rank-2")
        if embed_np.shape[1] != head_np.shape[1]:
            raise ValueError("lexical input/output hidden sizes differ")
        tied = lm_head is None
        np.save(root / "embed.npy", embed_np)
        if not tied:
            np.save(root / "lm_head.npy", head_np)
        head_filename = "embed.npy" if tied else "lm_head.npy"
        abi = body_abi(body_config, architecture)
        manifest = _write_manifest(
            root,
            {
                "schema_version": LEXICAL_WEIGHTS_SCHEMA,
                "body_abi": abi,
                "body_abi_sha256": abi["semantic_sha256"],
                "input_row_count": int(embed_np.shape[0]),
                "output_row_count": int(head_np.shape[0]),
                "hidden_size": int(embed_np.shape[1]),
                "row_key_schema": "dense-index-v1",
                "tied": tied,
                "files": {"embed": "embed.npy", "lm_head": head_filename},
                "sha256": {
                    "embed": _file_sha256(root / "embed.npy"),
                    "lm_head": _file_sha256(root / head_filename),
                },
                "provenance": dict(provenance or {}),
            },
        )
        return cls(
            root,
            manifest,
            np.load(root / "embed.npy", mmap_mode="r", allow_pickle=False),
            np.load(root / head_filename, mmap_mode="r", allow_pickle=False),
        )

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        body_config: Mapping[str, Any],
        architecture: str,
    ) -> LexicalWeights:
        root = Path(root).expanduser().resolve()
        manifest = _read_manifest(root, LEXICAL_WEIGHTS_SCHEMA, "lexical weights")
        expected_abi = body_abi(body_config, architecture)
        if manifest["body_abi_sha256"] != expected_abi["semantic_sha256"]:
            raise ValueError("lexical weights are incompatible with the transformer body ABI")
        matrices: dict[str, np.ndarray] = {}
        for key in ("embed", "lm_head"):
            filename = root / str(manifest["files"][key])
            if not filename.is_file() or _file_sha256(filename) != manifest["sha256"][key]:
                raise ValueError(f"lexical weight matrix is missing or changed: {key}")
            matrices[key] = np.load(filename, mmap_mode="r", allow_pickle=False)
        return cls(root, manifest, matrices["embed"], matrices["lm_head"])

    def assert_content_identity_unchanged(self) -> None:
        current = _read_manifest(self.root, LEXICAL_WEIGHTS_SCHEMA, "lexical weights")
        if current["semantic_sha256"] != self.semantic_sha256:
            raise RuntimeError("lexical weights manifest changed after validation")
        for key in ("embed", "lm_head"):
            filename = self.root / str(current["files"][key])
            if _file_sha256(filename) != current["sha256"][key]:
                raise RuntimeError(f"lexical weight matrix changed after validation: {key}")


class LexicalBinding:
    """Explicit tokenizer-ID to input/output lexical-row mapping."""

    def __init__(
        self,
        root: Path | None,
        manifest: Mapping[str, Any],
        values: LexicalValues,
        weights: LexicalWeights,
        input_rows: np.ndarray,
        output_rows: np.ndarray,
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.values = values
        self.weights = weights
        self.tokenizer = values.tokenizer
        self.input_rows = np.asarray(input_rows, dtype=np.int64)
        self.output_rows = np.asarray(output_rows, dtype=np.int64)
        self.token_count = values.token_count
        # ``token_count`` is the semantic tokenizer vocabulary.  The output projection may
        # intentionally expose a larger padded row space (Qwen's native stores do this), so
        # keep that count separate from the tokenizer identity.
        self.output_token_count = int(
            self.manifest.get("output_token_count", values.token_count)
        )
        self.tied = weights.tied
        if self.input_rows.shape != (self.token_count,):
            raise ValueError("lexical binding maps must cover every tokenizer ID")
        if self.output_rows.shape != (self.output_token_count,):
            raise ValueError("lexical output binding must cover every output row")
        if self.input_rows.size and (
            self.input_rows.min() < 0
            or self.input_rows.max() >= weights.input_row_count
            or self.output_rows.min() < 0
            or self.output_rows.max() >= weights.output_row_count
        ):
            raise ValueError("lexical binding row is outside the lexical weight row space")
    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["semantic_sha256"])

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        values: LexicalValues,
        weights: LexicalWeights,
        input_rows: Sequence[int] | np.ndarray | None = None,
        output_rows: Sequence[int] | np.ndarray | None = None,
        output_token_count: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> LexicalBinding:
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        identity_input = np.arange(values.token_count, dtype=np.int64)
        requested_output_count = (
            values.token_count if output_token_count is None else int(output_token_count)
        )
        if requested_output_count <= 0 or requested_output_count > weights.output_row_count:
            raise ValueError("output_token_count must be inside the lexical output row space")
        identity_output = np.arange(requested_output_count, dtype=np.int64)
        input_np = identity_input if input_rows is None else np.asarray(input_rows, dtype=np.int64)
        output_np = identity_output if output_rows is None else np.asarray(output_rows, dtype=np.int64)
        if input_rows is None and values.token_count != weights.input_row_count:
            raise ValueError("input row mapping is required when vocabulary sizes differ")
        if output_rows is None and requested_output_count != weights.output_row_count:
            raise ValueError("output row mapping is required when vocabulary sizes differ")
        if input_np.shape != (values.token_count,):
            raise ValueError("input row mapping must cover every tokenizer ID")
        if output_np.shape != (requested_output_count,):
            raise ValueError("output row mapping must cover every output row")
        np.save(root / "input_rows.npy", input_np)
        np.save(root / "output_rows.npy", output_np)
        manifest = _write_manifest(
            root,
            {
                "schema_version": LEXICAL_BINDING_SCHEMA,
                "values_semantic_sha256": values.semantic_sha256,
                "weights_semantic_sha256": weights.semantic_sha256,
                "token_count": values.token_count,
                "output_token_count": requested_output_count,
                "files": {"input_rows": "input_rows.npy", "output_rows": "output_rows.npy"},
                "sha256": {
                    "input_rows": _file_sha256(root / "input_rows.npy"),
                    "output_rows": _file_sha256(root / "output_rows.npy"),
                },
                "provenance": dict(provenance or {}),
            },
        )
        return cls(root, manifest, values, weights, input_np, output_np)

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        values: LexicalValues,
        weights: LexicalWeights,
    ) -> LexicalBinding:
        root = Path(root).expanduser().resolve()
        manifest = _read_manifest(root, LEXICAL_BINDING_SCHEMA, "lexical binding")
        if manifest["values_semantic_sha256"] != values.semantic_sha256:
            raise ValueError("lexical binding references different lexical values")
        if manifest["weights_semantic_sha256"] != weights.semantic_sha256:
            raise ValueError("lexical binding references different lexical weights")
        if int(manifest["token_count"]) != values.token_count:
            raise ValueError("lexical binding token count mismatch")
        output_token_count = int(manifest.get("output_token_count", values.token_count))
        if output_token_count <= 0 or output_token_count > weights.output_row_count:
            raise ValueError("lexical binding output row count is invalid")
        maps: dict[str, np.ndarray] = {}
        for key in ("input_rows", "output_rows"):
            filename = root / str(manifest["files"][key])
            if not filename.is_file() or _file_sha256(filename) != manifest["sha256"][key]:
                raise ValueError(f"lexical binding map is missing or changed: {key}")
            maps[key] = np.load(filename, allow_pickle=False)
        return cls(root, manifest, values, weights, maps["input_rows"], maps["output_rows"])

    def embed_rows(
        self, ids: np.ndarray | torch.Tensor, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        index = torch.as_tensor(ids, dtype=torch.long, device=device)
        if index.numel() and (index.min() < 0 or index.max() >= self.token_count):
            raise IndexError(f"token IDs must be inside [0, {self.token_count})")
        mapped = self.input_rows[index.reshape(-1).cpu().numpy()]
        rows = np.asarray(self.weights.embed[mapped], dtype=np.float32)
        result = torch.from_numpy(np.array(rows, copy=True)).reshape(*index.shape, -1)
        return result.to(device)

    def selected_rows(
        self, ids: Sequence[int] | np.ndarray | torch.Tensor, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        index = torch.as_tensor(ids, dtype=torch.long, device=device)
        if index.numel() and (index.min() < 0 or index.max() >= self.output_token_count):
            raise IndexError(f"output token IDs must be inside [0, {self.output_token_count})")
        mapped = self.output_rows[index.reshape(-1).cpu().numpy()]
        rows = np.asarray(self.weights.lm_head[mapped], dtype=np.float32)
        return torch.from_numpy(np.array(rows, copy=True)).to(device)

    def row_blocks(self, *, bs: int = 8192, device: torch.device | str = "cpu"):
        for start in range(0, self.output_token_count, bs):
            end = min(start + bs, self.output_token_count)
            ids = torch.arange(start, end, dtype=torch.long, device="cpu")
            yield start, end, self.selected_rows(ids, device=device)

    def assert_content_identity_unchanged(self) -> None:
        if self.root is not None:
            current = _read_manifest(self.root, LEXICAL_BINDING_SCHEMA, "lexical binding")
            if current["semantic_sha256"] != self.semantic_sha256:
                raise RuntimeError("lexical binding manifest changed after validation")
            for key in ("input_rows", "output_rows"):
                filename = self.root / str(current["files"][key])
                if _file_sha256(filename) != current["sha256"][key]:
                    raise RuntimeError(f"lexical binding map changed after validation: {key}")
        self.values.assert_content_identity_unchanged()
        self.weights.assert_content_identity_unchanged()


def load_separated_lexical(
    *,
    values_path: str | Path,
    weights_path: str | Path,
    binding_path: str | Path,
    body_config: Mapping[str, Any],
    architecture: str,
) -> LexicalBinding:
    values = LexicalValues.load(values_path)
    weights = LexicalWeights.load(
        weights_path, body_config=body_config, architecture=architecture
    )
    return LexicalBinding.load(binding_path, values=values, weights=weights)


class LexicalComponent:
    """Immutable lexical artifact plus its runtime tokenizer and matrices."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], tokenizer: Any,
                 embed: np.ndarray, lm_head: np.ndarray) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.tokenizer = tokenizer
        self.embed = embed
        self.lm_head = lm_head
        self.tied = bool(self.manifest["tied"])
        if self.embed.ndim != 2 or self.lm_head.ndim != 2:
            raise ValueError("lexical matrices must be rank-2")
        if self.embed.shape[1] != self.lm_head.shape[1]:
            raise ValueError("lexical input/output hidden sizes differ")
        if self.embed.shape[0] != int(self.manifest["token_count"]):
            raise ValueError("lexical embedding row count disagrees with manifest")
        if self.lm_head.shape[0] != int(self.manifest["output_token_count"]):
            raise ValueError("lexical output row count disagrees with manifest")
        if self.tied and not np.array_equal(self.embed, self.lm_head):
            raise ValueError("tied lexical component must use one identical matrix")

    @classmethod
    def load(cls, root: str | Path, *, body_config: Mapping[str, Any], architecture: str) -> LexicalComponent:
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"lexical component has no manifest.json: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != LEXICAL_SCHEMA:
            raise ValueError("unsupported lexical component schema")
        claimed_component_sha = manifest.get("semantic_sha256")
        unsigned_manifest = {key: value for key, value in manifest.items() if key != "semantic_sha256"}
        if claimed_component_sha != _sha256_bytes(_canonical(unsigned_manifest)):
            raise ValueError("lexical component semantic fingerprint mismatch")
        expected_abi = body_abi(body_config, architecture)
        if manifest.get("body_abi_sha256") != expected_abi["semantic_sha256"]:
            raise ValueError("lexical component is incompatible with the transformer body ABI")
        tok_dir = root / "tokenizer"
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot load lexical tokenizer from {tok_dir}") from exc
        descriptor = tokenizer_descriptor(tokenizer)
        if descriptor["semantic_sha256"] != manifest["tokenizer"]["semantic_sha256"]:
            raise ValueError("lexical tokenizer semantic fingerprint mismatch")
        if int(manifest["token_count"]) != len(tokenizer) or int(manifest["output_token_count"]) != len(tokenizer):
            raise ValueError("lexical matrix vocabulary does not match tokenizer length")
        vocab = VocabManifest.from_descriptor(
            descriptor, configured_row_count=int(manifest["token_count"])
        )
        if vocab.semantic_sha256 != manifest["vocab_manifest"]["semantic_sha256"]:
            raise ValueError("lexical vocabulary manifest mismatch")
        matrices = {}
        for key in ("embed", "lm_head"):
            filename = root / str(manifest["files"][key])
            if not filename.is_file() or _file_sha256(filename) != manifest["sha256"][key]:
                raise ValueError(f"lexical matrix is missing or changed: {key}")
            matrices[key] = np.load(filename, mmap_mode="r", allow_pickle=False)
        return cls(root, manifest, tokenizer, matrices["embed"], matrices["lm_head"])

    @classmethod
    def create(cls, root: str | Path, *, tokenizer: Any, embed: torch.Tensor | np.ndarray,
               lm_head: torch.Tensor | np.ndarray | None, body_config: Mapping[str, Any],
               architecture: str, provenance: Mapping[str, Any] | None = None) -> LexicalComponent:
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        tok_dir = root / "tokenizer"
        if tok_dir.exists():
            shutil.rmtree(tok_dir)
        tokenizer.save_pretrained(tok_dir)
        embed_np = np.asarray(embed.detach().cpu() if isinstance(embed, torch.Tensor) else embed, dtype=np.float32)
        head_np = embed_np if lm_head is None else np.asarray(lm_head.detach().cpu() if isinstance(lm_head, torch.Tensor) else lm_head, dtype=np.float32)
        if embed_np.ndim != 2 or head_np.ndim != 2 or embed_np.shape[1] != head_np.shape[1]:
            raise ValueError("lexical matrices must be rank-2 with equal hidden size")
        if embed_np.shape[0] != len(tokenizer):
            raise ValueError("input embedding rows must equal tokenizer length")
        np.save(root / "embed.npy", embed_np)
        np.save(root / "lm_head.npy", head_np)
        descriptor = tokenizer_descriptor(tokenizer)
        vocab = VocabManifest.from_descriptor(descriptor, configured_row_count=len(tokenizer))
        abi = body_abi(body_config, architecture)
        manifest = {
            "schema_version": LEXICAL_SCHEMA,
            "body_abi": abi,
            "body_abi_sha256": abi["semantic_sha256"],
            "tokenizer": descriptor,
            "vocab_manifest": vocab.as_dict(),
            "token_count": int(embed_np.shape[0]),
            "output_token_count": int(head_np.shape[0]),
            "hidden_size": int(embed_np.shape[1]),
            "tied": bool(lm_head is None),
            "files": {"embed": "embed.npy", "lm_head": "lm_head.npy"},
            "sha256": {"embed": _file_sha256(root / "embed.npy"), "lm_head": _file_sha256(root / "lm_head.npy")},
            "provenance": dict(provenance or {}),
        }
        manifest["semantic_sha256"] = _sha256_bytes(_canonical(manifest))
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls(root, manifest, tokenizer, np.load(root / "embed.npy", mmap_mode="r"), np.load(root / "lm_head.npy", mmap_mode="r"))

    @property
    def token_count(self) -> int:
        return int(self.manifest["token_count"])

    @property
    def output_token_count(self) -> int:
        return int(self.manifest["output_token_count"])

    def embed_rows(self, ids: np.ndarray | torch.Tensor, *, device: torch.device | str = "cpu") -> torch.Tensor:
        index = torch.as_tensor(ids, dtype=torch.long)
        if index.numel() and (index.min() < 0 or index.max() >= self.token_count):
            raise IndexError(f"token IDs must be inside [0, {self.token_count})")
        selected = np.asarray(self.embed)[index.reshape(-1).cpu().numpy()]
        rows = torch.from_numpy(np.array(selected, dtype=np.float32, copy=True))
        return rows.reshape(*index.shape, -1).to(device)

    def row_blocks(self, *, bs: int = 8192, device: torch.device | str = "cpu"):
        for start in range(0, self.output_token_count, bs):
            end = min(start + bs, self.output_token_count)
            rows = np.array(self.lm_head[start:end], dtype=np.float32, copy=True)
            yield start, end, torch.from_numpy(rows).to(device)

    def selected_rows(self, ids: Sequence[int] | np.ndarray | torch.Tensor, *, device: torch.device | str = "cpu") -> torch.Tensor:
        index = torch.as_tensor(ids, dtype=torch.long)
        if index.numel() and (index.min() < 0 or index.max() >= self.output_token_count):
            raise IndexError(f"output token IDs must be inside [0, {self.output_token_count})")
        selected = np.asarray(self.lm_head)[index.reshape(-1).cpu().numpy()]
        return torch.from_numpy(np.array(selected, dtype=np.float32, copy=True)).to(device)

    def assert_content_identity_unchanged(self) -> None:
        manifest_path = self.root / "manifest.json"
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("semantic_sha256") != self.manifest.get("semantic_sha256"):
            raise RuntimeError("lexical component manifest changed after validation")
        for key in ("embed", "lm_head"):
            filename = self.root / str(self.manifest["files"][key])
            if _file_sha256(filename) != self.manifest["sha256"][key]:
                raise RuntimeError(f"lexical component matrix changed after validation: {key}")


def train_lexical_bridge(
    body_forward: Any,
    embedding: torch.nn.Module,
    output_projection: torch.nn.Module,
    batches: Sequence[torch.Tensor],
    *,
    steps: int,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Train only the lexical interface against a frozen body.

    ``body_forward`` receives ``embedding(input_ids)`` and must return final hidden states with
    shape ``[batch, tokens, hidden]``.  Batches are token IDs and use the ordinary shifted
    next-token objective.  This deliberately accepts modules instead of a model-specific trainer
    so local corpus loaders and HF/paged body adapters can share the same lexical-only loop.
    """

    if steps <= 0:
        raise ValueError("lexical bridge steps must be positive")
    for parameter in body_forward.parameters() if hasattr(body_forward, "parameters") else ():
        parameter.requires_grad_(False)
    parameters = [*embedding.parameters(), *output_projection.parameters()]
    if not parameters:
        raise ValueError("lexical bridge has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate))
    losses: list[float] = []
    embedding.train()
    output_projection.train()
    for step, ids in enumerate(batches):
        if step >= steps:
            break
        if ids.ndim != 2 or ids.shape[1] < 2:
            raise ValueError("lexical bridge batches must have shape [batch, tokens>=2]")
        hidden = body_forward(embedding(ids[:, :-1]))
        logits = output_projection(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if len(losses) != min(steps, len(batches)):
        raise ValueError("lexical bridge received fewer batches than requested steps")
    return losses


class LexicalQStoreView:
    """Store adapter: body methods stay on the original store; lexical names are replaced."""

    def __init__(self, body: Any, lexical: LexicalComponent) -> None:
        self._body = body
        self.lexical = lexical
        self.cfg = dict(body.cfg)
        self.cfg["vocab_size"] = lexical.output_token_count
        self.device = body.device
        self.man = body.man

    def embed_rows(self, name: str, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        if name == "embed":
            return self.lexical.embed_rows(ids, device=self.device)
        if name == "lm_head":
            return self.lexical.selected_rows(ids, device=self.device)
        return self._body.embed_rows(name, ids)

    def row_blocks(self, name: str, bs: int = 8192):
        if name == "lm_head":
            yield from self.lexical.row_blocks(bs=bs, device=self.device)
            return
        yield from self._body.row_blocks(name, bs=bs)

    def selected_rows_fp32(self, name: str, ids: Any) -> torch.Tensor:
        if name == "lm_head":
            return self.lexical.selected_rows(ids, device=self.device).float()
        return self._body.selected_rows_fp32(name, ids)

    def resident_exact_head_fp32(self, name: str = "lm_head") -> torch.Tensor | None:
        # A resident body head would silently bypass the replacement vocabulary.
        if name == "lm_head":
            return None
        getter = getattr(self._body, "resident_exact_head_fp32", None)
        return getter(name) if callable(getter) else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._body, name)

    def assert_content_identity_unchanged(self) -> None:
        self._body.assert_content_identity_unchanged()
        self.lexical.assert_content_identity_unchanged()

    def close(self) -> None:
        self._body.close()
