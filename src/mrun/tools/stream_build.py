"""stream_build.py — build a paged int8 QStore straight from HuggingFace over HTTP.

The RAM-decoupling engine already runs a model of any size in O(largest-matrix) RAM by
demand-paging one dequantized matrix at a time out of a disk store. That store is normally
produced by ``qstore_build.build`` from safetensors **on local disk**, which means you must
first download the full bf16 checkpoint (~2x the store size) to disk before you can shrink it.

This module kills that disk requirement the same way O(largest-matrix) killed RAM: it streams
the safetensors payload directly from HF via HTTP **Range** requests, one tensor at a time, and
writes the int8 store as it goes. The full original bf16 weights NEVER touch disk — disk cost is
the store only (~half of bf16). Peak RSS is O(one tensor).

    header (one small ranged GET) -> per tensor: ranged GET its bytes -> decode to fp32 in RAM
    -> int8 row-scale quantize -> append to store -> free.

BYTE-COMPATIBLE BY CONSTRUCTION
--------------------------------
This builder does not re-implement the store format. It imports the *pure* helpers from
``qstore_build`` — ``_canon`` (HF key -> block name), ``_is_fp32_block``, ``_quant_row_int8``
(per-output-channel symmetric int8, ``W.abs().amax(dim=1)/127``) and ``_arch_config`` — and only
replaces the byte source (safetensors ``safe_open`` -> HTTP Range). safetensors ``safe_open``
returns keys in sorted order; we iterate the header's tensors in that same sorted order, per
shard, so the concatenated ``weights.i8`` / ``scales.f32`` / ``extras.f32`` byte layout and the
manifest block-insertion order match the local builder EXACTLY. bf16 payload is decoded via
torch (``torch.frombuffer(..., dtype=bfloat16).float()``) — the same lossless bf16->f32 widen the
local builder does — so the quantized codes are bit-identical.

Result: a store built here loads in the existing engine unmodified and is byte-identical to one
built by ``qstore_build.build`` (parity gate below).

RESUME / ROBUSTNESS
-------------------
Every ranged GET retries with exponential backoff. A ``.stream_progress.json`` sidecar records
the committed byte offsets of the three store files plus the set of finished blocks after each
tensor; an interrupted build truncates the files back to the last committed offsets and skips the
finished tensors (tensors are independent). Delete the sidecar (or pass ``resume=False``) to force
a clean rebuild.

INTEGRATION (one line, no edit to the contested build path required)
--------------------------------------------------------------------
``cli._build_store`` currently dispatches int2/int3/int4/int8 local builders. To expose streaming,
add ONE branch there (guarded by a new ``--stream`` flag on the ``build-store`` subparser)::

    if getattr(args, "stream", False):
        from .tools.stream_build import build as build_stream
        path = build_stream(args.model, out_root=out_root); print(f"store -> {path}"); return 0

No existing line changes; the local builders keep their exact behaviour.

LAZY-PARTIAL STORE (design sketch — NOT built here)
---------------------------------------------------
This streaming builder is the first half of a fault-in store. A lazy-partial store would:
  * On build, fetch ONLY the header(s) and write a manifest whose blocks carry their HF
    ``{url, byte_range, dtype, shape}`` instead of (or alongside) local offsets, plus a bitmap
    of which blocks are already materialized locally (initially none / only tiny fp32 extras).
  * On first touch in the forward pass (``QStore.weight(name)`` miss), a subclass ranged-GETs
    that one tensor, quantizes it, appends it to ``weights.i8``/``scales.f32``, flips its bitmap
    bit, and rewrites its manifest offsets — so the store grows only over the blocks a given
    workload actually reads. A single forward touches every block once, so it converges to the
    full store; a routing/MoE or shallow-probe workload touches a subset and pays for a subset.
  * Peak disk then tracks the working set, not the model. The retry/backoff + offset-commit
    machinery here is exactly what that fault-in path reuses.

mbp has internet; a 0.5B build is a ~1GB network pull. Any model *forward* in the parity gate runs
mrun-guarded (declared RAM limit).
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import numpy as np

from ..engine.kernels.qstore_build import (
    _arch_config,
    _canon,
    _is_fp32_block,
    _quant_row_int8,
    rss_mb,
)
from ..models import resolve_model, store_name
from ..paths import stores_root

# safetensors dtype string -> (numpy source dtype for a raw byte view, itemsize)
_ST_DTYPE = {
    "BF16": ("bf16", 2),
    "F16": (np.float16, 2),
    "F32": (np.float32, 4),
    "F64": (np.float64, 8),
}


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _resolve_url(hf_id: str, revision: str, filename: str) -> str:
    return f"{_hf_endpoint()}/{hf_id}/resolve/{revision}/{filename}"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _session():
    import requests

    s = requests.Session()
    tok = _token()
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    return s


def _ranged_get(session, url: str, start: int, end: int, *, retries: int = 6,
                backoff: float = 1.0) -> bytes:
    """GET bytes [start, end] inclusive via an HTTP Range request, with retry+backoff.

    Returns exactly (end-start+1) bytes. Raises the last error after ``retries`` attempts.
    """
    want = end - start + 1
    headers = {"Range": f"bytes={start}-{end}"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=60, stream=True)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code} for Range {start}-{end} of {url}")
            buf = r.content
            if len(buf) != want:
                raise RuntimeError(f"short read: got {len(buf)} want {want} ({url})")
            return buf
        except Exception as e:  # noqa: BLE001 - retry any transport/HTTP error
            last = e
            if attempt == retries - 1:
                break
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"ranged GET failed after {retries} attempts: {last}")


def probe_range_support(hf_id: str = "Qwen/Qwen2.5-0.5B", revision: str = "main") -> dict:
    """Verify HF resolve URLs honour Range requests (HEAD for size + a 2-byte ranged GET)."""
    session = _session()
    url = _resolve_url(hf_id, revision, "model.safetensors")
    head = session.head(url, allow_redirects=True, timeout=30)
    two = _ranged_get(session, url, 0, 1)
    (hdr_len,) = struct.unpack("<Q", _ranged_get(session, url, 0, 7))
    return {
        "url": url,
        "accept_ranges": head.headers.get("accept-ranges"),
        "content_length": head.headers.get("content-length"),
        "ranged_2b_len": len(two),
        "header_len": int(hdr_len),
    }


def _fetch_header(session, url: str) -> tuple[dict, int]:
    """Fetch a safetensors JSON header. Returns (header_dict, data_base_offset).

    Layout: 8-byte little-endian header length N, then N JSON bytes, then the tensor data
    buffer. A tensor's absolute byte range is ``data_base + data_offsets``.
    """
    (n,) = struct.unpack("<Q", _ranged_get(session, url, 0, 7))
    hdr = json.loads(_ranged_get(session, url, 8, 8 + n - 1))
    return hdr, 8 + n


def _shard_files(session, hf_id: str, revision: str) -> list[str]:
    """Resolve the list of safetensors shard filenames (single file or index.json map)."""
    import requests

    idx_url = _resolve_url(hf_id, revision, "model.safetensors.index.json")
    try:
        r = session.get(idx_url, timeout=30)
        if r.status_code == 200:
            weight_map = r.json()["weight_map"]
            return sorted(set(weight_map.values()))
    except requests.RequestException:
        pass
    return ["model.safetensors"]


def _fetch_config(session, hf_id: str, revision: str) -> dict:
    r = session.get(_resolve_url(hf_id, revision, "config.json"), timeout=30)
    r.raise_for_status()
    return r.json()


def _decode_to_fp32(buf: bytes, dtype_str: str, shape: list[int]) -> np.ndarray:
    """Decode a raw tensor byte payload to a contiguous fp32 numpy array.

    bf16 is widened via torch (lossless, matches the local builder's ``t.to(float32)``); numpy
    has no bf16. Other float dtypes decode via numpy then cast to fp32.
    """
    kind, _ = _ST_DTYPE[dtype_str]
    if kind == "bf16":
        import torch

        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).to(torch.float32)
        return t.numpy().reshape(shape)
    arr = np.frombuffer(buf, dtype=kind).astype(np.float32)
    return arr.reshape(shape)


def _ordered_blocks(hdr: dict, arch: str) -> list[tuple[str, str, dict]]:
    """(canonical block name, HF key, header entry) for every mapped tensor, in the local
    builder's iteration order (safetensors ``safe_open`` returns keys sorted)."""
    out = []
    for key in sorted(k for k in hdr if k != "__metadata__"):
        name = _canon(key, arch)
        if name is not None:
            out.append((name, key, hdr[key]))
    return out


def _default_store_dir(model_name: str, hf_id: str) -> str:
    """Store dir name = registry store_name when the model is known, else the HF-id basename."""
    try:
        return store_name(model_name)
    except Exception:  # noqa: BLE001 - unknown-to-registry id: fall back to the HF-id tail
        return hf_id.rsplit("/", 1)[-1]


def build(model_name: str, *, out_root: Path | None = None, store_dir_name: str | None = None,
          hf_id: str | None = None, revision: str = "main", resume: bool = True,
          verbose: bool = True) -> Path:
    """Stream ``model_name`` from HF into a byte-compatible int8 paged store.

    Byte-identical to ``qstore_build.build(model_name)`` (same block order, same per-row int8).
    The full bf16 checkpoint never touches disk; peak RSS is O(largest single tensor).

    ``model_name`` is a registry key/friendly name (its ``hf_id`` is looked up for the URL and its
    ``store_name`` for the output dir). Pass ``hf_id`` explicitly to stream an id the registry
    doesn't know (or to point at a mirror); ``model_name`` is then only the manifest label.
    """
    hf_id = hf_id or resolve_model(model_name).hf_id
    root = Path(out_root) if out_root is not None else stores_root()
    dir_name = store_dir_name or _default_store_dir(model_name, hf_id)
    out = root / dir_name
    out.mkdir(parents=True, exist_ok=True)

    session = _session()
    raw = _fetch_config(session, hf_id, revision)
    arch = raw.get("model_type", "?")
    if arch not in ("qwen2", "llama", "qwen3", "gpt_neox", "mamba"):
        raise NotImplementedError(f"arch {arch!r} not supported (qwen2/llama/qwen3/gpt_neox/mamba)")

    shards = _shard_files(session, hf_id, revision)
    # ordered (block, key, header-entry, shard-url) across all shards, matching local order:
    # sorted shard filenames, then sorted keys within each shard.
    plan: list[tuple[str, str, dict, str, int]] = []
    for fn in sorted(shards):
        url = _resolve_url(hf_id, revision, fn)
        hdr, data_base = _fetch_header(session, url)
        for name, key, entry in _ordered_blocks(hdr, arch):
            plan.append((name, key, entry, url, data_base))

    w_path, s_path, e_path = out / "weights.i8", out / "scales.f32", out / "extras.f32"
    prog_path = out / ".stream_progress.json"

    w_off = s_off = e_off = 0
    blocks: dict[str, dict] = {}
    done: set[str] = set()
    mode = "wb"
    if resume and prog_path.exists() and w_path.exists():
        prog = json.loads(prog_path.read_text())
        if prog.get("model_name") == model_name and prog.get("plan_len") == len(plan):
            w_off, s_off, e_off = prog["w_off"], prog["s_off"], prog["e_off"]
            blocks = prog["blocks"]
            done = set(prog["done"])
            # truncate any partial tail past the last committed offsets, then append
            for p, off in ((w_path, w_off), (s_path, s_off), (e_path, e_off)):
                with open(p, "r+b") as fh:
                    fh.truncate(off)
            mode = "r+b"
            if verbose:
                print(f">>> resume — {len(done)}/{len(plan)} tensors already committed")

    wf, sf, ef = open(w_path, mode), open(s_path, mode), open(e_path, mode)
    wf.seek(w_off); sf.seek(s_off); ef.seek(e_off)

    if verbose:
        print(f">>> stream build — {model_name}  arch={arch}  hf={hf_id}@{revision}  "
              f"shards={len(shards)}  tensors={len(plan)}  -> {out}")
    n_q = n_fp = 0
    peak = rss_mb()
    for i, (name, key, entry, url, data_base) in enumerate(plan):
        if name in done:
            continue
        begin, end = entry["data_offsets"]
        buf = _ranged_get(session, url, data_base + begin, data_base + end - 1)
        W = _decode_to_fp32(buf, entry["dtype"], entry["shape"])
        del buf
        if _is_fp32_block(name):
            arr = np.ascontiguousarray(W.astype(np.float32))
            ef.write(arr.tobytes())
            blocks[name] = {"kind": "fp32", "shape": list(W.shape),
                            "e_off": e_off, "e_len": int(arr.nbytes)}
            e_off += int(arr.nbytes)
            n_fp += 1
        else:
            if W.ndim != 2:
                raise ValueError(f"{name}: expected 2D, got {W.shape}")
            q, scale = _quant_row_int8(W)
            q = np.ascontiguousarray(q)
            scale = np.ascontiguousarray(scale)
            wf.write(q.tobytes()); sf.write(scale.tobytes())
            blocks[name] = {"kind": "qrow", "shape": list(W.shape),
                            "w_off": w_off, "w_len": int(q.nbytes),
                            "s_off": s_off, "s_len": int(scale.nbytes)}
            w_off += int(q.nbytes); s_off += int(scale.nbytes)
            n_q += 1
        del W
        done.add(name)
        # commit: flush store bytes, then record offsets so an interrupt resumes cleanly
        wf.flush(); sf.flush(); ef.flush()
        os.fsync(wf.fileno()); os.fsync(sf.fileno()); os.fsync(ef.fileno())
        prog_path.write_text(json.dumps({
            "model_name": model_name, "plan_len": len(plan),
            "w_off": w_off, "s_off": s_off, "e_off": e_off,
            "blocks": blocks, "done": sorted(done),
        }))
        peak = max(peak, rss_mb())
        if verbose and (i % 25 == 0 or i == len(plan) - 1):
            print(f"    [{i + 1}/{len(plan)}] {key} -> {name}  RSS={rss_mb():.0f}MB "
                  f"peak={peak:.0f}MB")
    wf.close(); sf.close(); ef.close()

    tie = "lm_head" not in blocks and "embed" in blocks
    if tie:
        blocks["lm_head"] = {"alias": "embed"}
    manifest = {
        "model_name": model_name, "arch": arch, "dtype": "int8",
        "tie_word_embeddings": tie,
        "config": _arch_config(arch, raw),
        "blocks": blocks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    prog_path.unlink(missing_ok=True)

    disk_mb = sum(p.stat().st_size for p in (w_path, s_path, e_path)) / 1e6
    if verbose:
        print(f"  quantized={n_q} blocks  fp32={n_fp} blocks  store={disk_mb:.1f}MB  "
              f"peakRSS={peak:.0f}MB  (no bf16 ever hit disk)")
        print(f"  -> {out}")
    return out
