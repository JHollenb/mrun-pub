"""Single-tensor checkpoint readers — pull ONE weight, never the model.

A safetensors file is a JSON header followed by a flat byte blob, and every tensor's bytes are
contiguous, so a *row range* of a 2-D tensor is a contiguous byte range. That makes both of the
things this instrument needs cheap:

  * remote — one HTTP ``Range`` request per row block against
    ``huggingface.co/<repo>/resolve/<rev>/<shard>``: pennies per model instead of a full
    checkpoint download (ported from ``experiments/2026-07-25-embedding-concentration-regime/
    remote_embed_n90.py`` and ``2026-07-26-distillation-fingerprints/fp_run.py``);
  * local — an mmap'd ``get_slice``, so nothing model-sized is ever resident.

Both are exposed as one :class:`TensorSource` interface whose only required capability is
"give me rows ``[start, stop)`` of key K". Everything above it (spectrum, row-delta) streams in
row blocks and therefore has a working set of O(chunk x d), NOT O(V x d) — this is load-bearing,
not tidiness: the MacBook that produced the measured results was RAM-critical and could not hold
a 32B embedding matrix in fp32.

    src = resolve_source("Qwen/Qwen2.5-7B")      # http-range (or hub-cache if already pulled)
    key = find_embedding_key(src)                 # 'model.embed_tokens.weight'
    for start, block in iter_rows(src, key, stop=1024, chunk=256):
        ...                                       # 256 x d at a time, native dtype, cpu
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

# safetensors dtype tag -> (torch dtype, itemsize). Only the dtypes real checkpoints use for
# embeddings; anything else raises loudly rather than silently mis-striding the byte range.
DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "F64": (torch.float64, 8),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
}

# Input-embedding key by family, most common first. Order matters: a tied model exposes only
# the embedding, an untied one also has ``lm_head.weight`` — which is a DIFFERENT object and is
# never auto-selected (pass ``key="lm_head.weight"`` explicitly to fingerprint the head).
EMBED_KEYS: tuple[str, ...] = (
    "model.embed_tokens.weight",        # llama / qwen2 / qwen3 / mistral / deepseek
    "gpt_neox.embed_in.weight",         # pythia / gpt_neox
    "transformer.wte.weight",           # gpt2
    "backbone.embedding.weight",        # mamba
    "model.embed_in.weight",
    "tok_embeddings.weight",            # llama original layout
    "embed_tokens.weight",
    "transformer.word_embeddings.weight",
    "word_embeddings.weight",
    "embedding.weight",
)

_SINGLE = "model.safetensors"
_INDEX = "model.safetensors.index.json"


@dataclass(frozen=True)
class TensorMeta:
    """Where a tensor's bytes live, and how to stride them."""
    key: str
    dtype: str
    shape: tuple[int, ...]
    itemsize: int

    @property
    def rows(self) -> int:
        return int(self.shape[0])

    @property
    def row_bytes(self) -> int:
        n = self.itemsize
        for dim in self.shape[1:]:
            n *= int(dim)
        return n


class TensorSource:
    """Read individual tensors (and row ranges of them) out of a checkpoint.

    Subclasses implement :meth:`keys`, :meth:`meta`, :meth:`read_rows` and :meth:`aux`.
    ``kind`` is recorded in every result row so a comparison can be audited for the
    local-vs-remote path it came through.
    """

    kind = "abstract"
    label = "?"

    def keys(self) -> tuple[str, ...]:
        raise NotImplementedError

    def meta(self, key: str) -> TensorMeta:
        raise NotImplementedError

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        """Rows ``[start, stop)`` of ``key`` in the checkpoint's NATIVE dtype, on cpu."""
        raise NotImplementedError

    def aux(self, name: str) -> bytes | None:
        """A small side file (``tokenizer.json``, ``config.json``); None when absent."""
        raise NotImplementedError

    def read(self, key: str) -> torch.Tensor:
        """The whole tensor. Prefer :func:`iter_rows` — this materializes V x d."""
        return self.read_rows(key, 0, self.meta(key).rows)


# ------------------------------------------------------------------ local (mmap)
class LocalSource(TensorSource):
    """mmap-backed reader over a checkpoint directory (or a single ``.safetensors`` file)."""

    def __init__(self, root: str | Path, *, kind: str = "local", label: str | None = None):
        self.root = Path(root)
        self.kind = kind
        self.label = label or str(self.root)
        if self.root.is_file():
            self._shards = {None: self.root}
            self._map: dict[str, Path] | None = None
        else:
            idx = self.root / _INDEX
            if idx.exists():
                weight_map = json.loads(idx.read_text())["weight_map"]
                self._map = {k: self.root / v for k, v in weight_map.items()}
                self._shards = {}
            else:
                single = self.root / _SINGLE
                files = [single] if single.exists() else sorted(self.root.glob("*.safetensors"))
                if not files:
                    raise FileNotFoundError(f"no safetensors under {self.root}")
                self._shards = {None: files[0]} if len(files) == 1 else {}
                self._map = None
                if len(files) > 1:  # sharded without an index: map keys by opening each header
                    from safetensors import safe_open
                    self._map = {}
                    for path in files:
                        with safe_open(str(path), framework="pt") as f:
                            for k in f.keys():
                                self._map[k] = path

    def _shard(self, key: str) -> Path:
        if self._map is not None:
            if key not in self._map:
                raise KeyError(key)
            return self._map[key]
        return self._shards[None]

    def keys(self) -> tuple[str, ...]:
        if self._map is not None:
            return tuple(self._map)
        from safetensors import safe_open
        with safe_open(str(self._shards[None]), framework="pt") as f:
            return tuple(f.keys())

    def meta(self, key: str) -> TensorMeta:
        from safetensors import safe_open
        with safe_open(str(self._shard(key)), framework="pt") as f:
            sl = f.get_slice(key)
            shape = tuple(int(x) for x in sl.get_shape())
            tag = _torch_dtype_tag(sl.get_dtype())
        return TensorMeta(key, tag, shape, DTYPES[tag][1])

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        from safetensors import safe_open
        with safe_open(str(self._shard(key)), framework="pt") as f:
            return f.get_slice(key)[start:stop]

    def aux(self, name: str) -> bytes | None:
        path = (self.root.parent if self.root.is_file() else self.root) / name
        return path.read_bytes() if path.exists() else None


def _torch_dtype_tag(raw: str) -> str:
    """safetensors ``get_dtype`` returns its own tag ('BF16') or a torch repr; normalize."""
    tag = str(raw).replace("torch.", "").upper()
    tag = {"BFLOAT16": "BF16", "FLOAT16": "F16", "FLOAT32": "F32", "FLOAT64": "F64",
           "FLOAT": "F32", "DOUBLE": "F64", "HALF": "F16"}.get(tag, tag)
    if tag not in DTYPES:
        raise TypeError(f"unsupported embedding dtype {raw!r} (quantized checkpoint? "
                        "fingerprints need a float embedding)")
    return tag


# ------------------------------------------------------------------ in-memory
class ArraySource(TensorSource):
    """A :class:`TensorSource` over tensors you already hold.

    Two uses: fingerprinting a matrix produced in-process, and TESTS — the whole instrument can
    be exercised on 200x16 synthetic tensors with no checkpoint and no network, which is the
    only way it may be tested on a RAM-critical host.
    """

    kind = "arrays"

    def __init__(self, tensors: dict[str, torch.Tensor], *, label: str = "arrays",
                 aux_files: dict[str, bytes] | None = None):
        self._t = {k: v for k, v in tensors.items()}
        self.label = label
        self._aux = dict(aux_files or {})

    def keys(self) -> tuple[str, ...]:
        return tuple(self._t)

    def meta(self, key: str) -> TensorMeta:
        t = self._t[key]
        tag = _torch_dtype_tag(str(t.dtype))
        return TensorMeta(key, tag, tuple(int(x) for x in t.shape), DTYPES[tag][1])

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        return self._t[key][int(start):int(stop)]

    def aux(self, name: str) -> bytes | None:
        return self._aux.get(name)


def synthetic_tokenizer_json(v_tok: int, *, salt: str = "") -> bytes:
    """A minimal ``tokenizer.json`` declaring ``v_tok`` tokens — for :class:`ArraySource` tests
    that need the ``V_tok`` / ``vocab_sig`` path to be live."""
    vocab = {f"{salt}t{i}": i for i in range(int(v_tok))}
    return json.dumps({"model": {"vocab": vocab}, "added_tokens": []}).encode()


# ------------------------------------------------------------------ remote (HTTP Range)
class HubSource(TensorSource):
    """HTTP-Range reader against the HF hub. Downloads only the byte ranges asked for.

    Shard resolution mirrors the measured scripts: ``model.safetensors.index.json`` when present,
    otherwise a single ``model.safetensors``. A repo that is unsharded but uses a non-standard
    filename (``model-00001-of-00001.safetensors`` with no index) will 404 loudly rather than be
    guessed at — clone it, or point at the local snapshot.
    """

    kind = "http-range"

    def __init__(self, repo: str, *, revision: str = "main", timeout: int = 300,
                 token: str | None = None):
        self.repo = repo
        self.revision = revision
        self.timeout = timeout
        self.label = f"{repo}@{revision}"
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN")
        self._base = f"https://huggingface.co/{repo}/resolve/{revision}"
        self._headers: dict[str, dict] = {}
        self._weight_map: dict[str, str] | None = None
        self._resolved = False

    # -- http
    def _fetch(self, url: str, start: int | None = None, end: int | None = None,
               retries: int = 3) -> bytes:
        last: Exception | None = None
        for _attempt in range(retries):
            req = urllib.request.Request(url, headers={"User-Agent": "mrun-fingerprint/1"})
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            if start is not None:
                req.add_header("Range", f"bytes={start}-{end}")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 404):
                    raise
                last = exc
            except OSError as exc:
                last = exc
        raise RuntimeError(f"GET {url} failed after {retries} tries: {last}") from last

    def _resolve_map(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        try:
            idx = json.loads(self._fetch(f"{self._base}/{_INDEX}"))
            self._weight_map = dict(idx["weight_map"])
        except (urllib.error.HTTPError, ValueError, KeyError):
            self._weight_map = None  # unsharded: everything lives in model.safetensors

    def _shard(self, key: str) -> str:
        self._resolve_map()
        if self._weight_map is None:
            return _SINGLE
        if key not in self._weight_map:
            raise KeyError(key)
        return self._weight_map[key]

    def _header(self, shard: str) -> dict:
        if shard not in self._headers:
            url = f"{self._base}/{shard}"
            n = struct.unpack("<Q", self._fetch(url, 0, 7))[0]
            self._headers[shard] = json.loads(self._fetch(url, 8, 8 + n - 1))
            self._headers[shard]["__data_start__"] = 8 + n
        return self._headers[shard]

    # -- TensorSource
    def keys(self) -> tuple[str, ...]:
        self._resolve_map()
        if self._weight_map is not None:
            return tuple(self._weight_map)
        return tuple(k for k in self._header(_SINGLE) if not k.startswith("__"))

    def meta(self, key: str) -> TensorMeta:
        entry = self._header(self._shard(key))[key]
        tag = entry["dtype"]
        if tag not in DTYPES:
            raise TypeError(f"unsupported embedding dtype {tag!r} for {key!r}")
        return TensorMeta(key, tag, tuple(int(x) for x in entry["shape"]), DTYPES[tag][1])

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        shard = self._shard(key)
        header = self._header(shard)
        entry = header[key]
        meta = self.meta(key)
        base = header["__data_start__"] + entry["data_offsets"][0]
        start = max(0, int(start))
        stop = min(meta.rows, int(stop))
        if stop <= start:
            return torch.empty((0, *meta.shape[1:]), dtype=DTYPES[meta.dtype][0])
        lo = base + start * meta.row_bytes
        hi = base + stop * meta.row_bytes - 1
        raw = self._fetch(f"{self._base}/{shard}", lo, hi)
        want = (stop - start) * meta.row_bytes
        if len(raw) != want:
            raise RuntimeError(f"range read of {key}[{start}:{stop}] returned {len(raw)} bytes, "
                               f"expected {want} (server ignored Range?)")
        flat = torch.frombuffer(bytearray(raw), dtype=DTYPES[meta.dtype][0])
        return flat.reshape((stop - start, *meta.shape[1:]))

    def aux(self, name: str) -> bytes | None:
        try:
            return self._fetch(f"{self._base}/{name}")
        except (urllib.error.HTTPError, RuntimeError):
            return None


# ------------------------------------------------------------------ resolution
def _hub_cache_snapshot(repo: str) -> Path | None:
    """Newest snapshot dir for ``repo`` in the local HF hub cache, if it holds safetensors."""
    from ..models import default_hub_root
    snaps = default_hub_root() / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if snap.is_dir() and any(snap.glob("*.safetensors")):
            return snap
    return None


def resolve_source(model: str | Path | TensorSource, *, prefer_local: bool = True,
                   revision: str = "main") -> TensorSource:
    """Turn a model reference into a :class:`TensorSource`.

    Accepts, in order: an already-built source (passthrough), an existing local path (dir or
    ``.safetensors`` file), an ``mrun.models`` registry name, an HF repo id. When
    ``prefer_local`` (default) an HF repo id already present in the local hub cache is read
    through mmap (``kind='hub-cache'``) instead of over the network — same bytes, no transfer.
    """
    if isinstance(model, TensorSource):
        return model
    ref = str(model)
    path = Path(ref).expanduser()
    if path.exists():
        return LocalSource(path)
    repo = ref
    try:  # registry name -> hf_id / materialized local dir
        from ..models import resolve_model
        spec = resolve_model(ref)
        repo = spec.hf_id
        if spec.local_path and Path(spec.local_path).exists():
            return LocalSource(spec.local_path, kind="local", label=spec.name)
    except Exception:  # noqa: BLE001 -- unknown name is fine, treat ref as a repo id
        pass
    if prefer_local:
        snap = _hub_cache_snapshot(repo)
        if snap is not None:
            return LocalSource(snap, kind="hub-cache", label=repo)
    if "/" not in repo:
        raise ValueError(f"{ref!r} is not a local path, a known registry name, or an HF repo id")
    return HubSource(repo, revision=revision)


def find_embedding_key(source: TensorSource, key: str | None = None) -> str:
    """The input-embedding key. Explicit ``key`` is validated, never guessed around."""
    available = set(source.keys())
    if key is not None:
        if key not in available:
            raise KeyError(f"{key!r} not in {source.label}; e.g. {sorted(available)[:6]}")
        return key
    for cand in EMBED_KEYS:
        if cand in available:
            return cand
    hits = sorted(k for k in available if "embed" in k.lower() or "wte" in k.lower())
    raise KeyError(f"no known embedding key in {source.label}; candidates seen: {hits[:8]}")


# ------------------------------------------------------------------ vocab / tokenizer
_VOCAB_SIG_SAMPLE = 2048


def vocab_info(source: TensorSource) -> tuple[int | None, str | None]:
    """``(V_tok, vocab_sig)`` from the checkpoint's ``tokenizer.json``.

    ``V_tok = max token id + 1`` (base vocab plus added/special tokens) — the *real* row count;
    ``V_cfg`` (the tensor's row count) is padded for kernel alignment and differs by hundreds of
    rows, which is exactly the kind of unmatched-object slop that invalidates a comparison.

    ``vocab_sig`` is a sha256 over a deterministic 2048-id sample of ``(id, token string)``
    pairs: equal ``V_tok`` does NOT prove equal tokenizers, and the signature catches the
    remainder. Returns ``(None, None)`` when the repo ships no ``tokenizer.json``.
    """
    raw = source.aux("tokenizer.json")
    if raw is None:
        return None, None
    try:
        tk = json.loads(raw.decode("utf-8"))
        vocab = dict(tk["model"]["vocab"])
    except (ValueError, KeyError, UnicodeDecodeError):
        return None, None
    for added in tk.get("added_tokens", []) or []:
        vocab[added["content"]] = added["id"]
    inv = {int(v): k for k, v in vocab.items() if isinstance(v, int)}
    if not inv:
        return None, None
    v_tok = max(inv) + 1
    step = max(1, v_tok // _VOCAB_SIG_SAMPLE)
    h = hashlib.sha256()
    for tid in range(0, v_tok, step):
        h.update(f"{tid}\x00{inv.get(tid, '')}\x1f".encode())
    return v_tok, h.hexdigest()[:16]


def token_strings(source: TensorSource) -> dict[int, str] | None:
    """Full ``id -> token`` map (for the row-delta leg's vocab-identity gate), or None."""
    raw = source.aux("tokenizer.json")
    if raw is None:
        return None
    try:
        tk = json.loads(raw.decode("utf-8"))
        vocab = dict(tk["model"]["vocab"])
    except (ValueError, KeyError, UnicodeDecodeError):
        return None
    for added in tk.get("added_tokens", []) or []:
        vocab[added["content"]] = added["id"]
    return {int(v): k for k, v in vocab.items() if isinstance(v, int)}


def iter_rows(source: TensorSource, key: str, *, stop: int | None = None,
              start: int = 0, chunk: int = 16384) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(row_offset, block)`` over ``[start, stop)`` in ``chunk``-row blocks.

    The only way the rest of the instrument touches weights, so the resident working set is
    O(chunk x d) regardless of vocabulary size.
    """
    meta = source.meta(key)
    stop = meta.rows if stop is None else min(int(stop), meta.rows)
    chunk = max(1, int(chunk))
    for lo in range(int(start), stop, chunk):
        yield lo, source.read_rows(key, lo, min(lo + chunk, stop))
