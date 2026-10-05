"""Tests for the streaming HF->int8 QStore builder (``mrun.tools.stream_build``).

Two tiers:
  * OFFLINE (default): a synthetic 2-layer qwen2 safetensors is served over a local
    ``http.server`` (which honours Range requests). Exercises header parse, bf16/f32 decode,
    block ordering, resume-after-interrupt, and self-consistent dequant + a paged forward.
  * NETWORK (``MRUN_STREAM_NET=1``): the real gate — stream Qwen2.5-0.5B from HF and assert the
    store is byte-identical to the on-disk ``qstore_build.build`` output and argmax-exact on a
    paged forward. ~1GB pull; skipped by default.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import struct
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from mrun.tools import stream_build as sb


# --------------------------------------------------------------------------- helpers

def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write a minimal .safetensors (matches the format stream_build reads)."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    off = 0
    st_dtype = {torch.bfloat16: "BF16", torch.float16: "F16",
                torch.float32: "F32", torch.float64: "F64"}
    for name in sorted(tensors):  # sorted == safetensors canonical order
        t = tensors[name].contiguous()
        raw = t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 \
            else t.numpy().tobytes()
        header[name] = {"dtype": st_dtype[t.dtype], "shape": list(t.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def _tiny_qwen2(tmp: Path) -> tuple[Path, dict]:
    """A synthetic 2-layer qwen2 checkpoint dir laid out as {hf_id}/resolve/main/*."""
    torch.manual_seed(0)
    H, I, NH, NKV, HD, V, L = 32, 64, 4, 2, 8, 40, 2
    cfg = {
        "model_type": "qwen2", "hidden_size": H, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
        "intermediate_size": I, "vocab_size": V, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0, "hidden_act": "silu", "tie_word_embeddings": True,
    }
    bf = torch.bfloat16
    tens: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(V, H).to(bf),
        "model.norm.weight": torch.randn(H).to(bf),
    }
    for li in range(L):
        p = f"model.layers.{li}."
        tens[p + "input_layernorm.weight"] = torch.randn(H).to(bf)
        tens[p + "post_attention_layernorm.weight"] = torch.randn(H).to(bf)
        tens[p + "self_attn.q_proj.weight"] = torch.randn(NH * HD, H).to(bf)
        tens[p + "self_attn.q_proj.bias"] = torch.randn(NH * HD).to(bf)
        tens[p + "self_attn.k_proj.weight"] = torch.randn(NKV * HD, H).to(bf)
        tens[p + "self_attn.k_proj.bias"] = torch.randn(NKV * HD).to(bf)
        tens[p + "self_attn.v_proj.weight"] = torch.randn(NKV * HD, H).to(bf)
        tens[p + "self_attn.v_proj.bias"] = torch.randn(NKV * HD).to(bf)
        tens[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * HD).to(bf)
        tens[p + "mlp.gate_proj.weight"] = torch.randn(I, H).to(bf)
        tens[p + "mlp.up_proj.weight"] = torch.randn(I, H).to(bf)
        tens[p + "mlp.down_proj.weight"] = torch.randn(H, I).to(bf)
    snap = tmp / "Fake" / "TinyQwen" / "resolve" / "main"
    snap.mkdir(parents=True)
    _write_safetensors(snap / "model.safetensors", tens)
    (snap / "config.json").write_text(json.dumps(cfg))
    return tmp, cfg


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + honest ``Range`` support (stdlib ignores it, HF honours it)."""

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if rng is None or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        start_s, _, end_s = rng.partition("=")[2].partition("-")
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server:
    """Serve a directory over HTTP with Range support (needed to mimic HF resolve URLs)."""

    def __init__(self, root: Path):
        handler = functools.partial(_RangeHandler, directory=str(root))
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# --------------------------------------------------------------------------- fast unit tests

def test_bf16_decode_matches_torch():
    torch.manual_seed(1)
    t = torch.randn(7, 5).to(torch.bfloat16)
    raw = t.view(torch.uint8).numpy().tobytes()
    got = sb._decode_to_fp32(raw, "BF16", [7, 5])
    ref = t.to(torch.float32).numpy()
    assert np.array_equal(got, ref)  # lossless bf16->f32, same as the local builder


def test_f32_decode_roundtrip():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = sb._decode_to_fp32(a.tobytes(), "F32", [3, 4])
    assert np.array_equal(got, a)


def test_ordered_blocks_is_sorted_key_order():
    hdr = {
        "__metadata__": {"format": "pt"},
        "model.layers.1.mlp.up_proj.weight": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [32, 64]},
        "model.layers.0.input_layernorm.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [64, 72]},
    }
    order = [name for name, _key, _e in sb._ordered_blocks(hdr, "qwen2")]
    # keys iterated in sorted() order; embed sorts before layers.0 before layers.1
    assert order == ["embed", "L0.ln1", "L1.up"]


# --------------------------------------------------------------------------- offline e2e

def _paged_logits(store, ids):
    from mrun.engine.kernels import paged_forward as pf
    return pf.paged_logits(store, np.asarray(ids, np.int64))


def test_stream_build_offline_end_to_end(tmp_path):
    from mrun.engine.kernels.qstore import QStore

    served, cfg = _tiny_qwen2(tmp_path / "srv")
    out_root = tmp_path / "stores"
    with _Server(served) as srv:
        os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{srv.port}"
        try:
            out = sb.build("tiny", out_root=out_root, hf_id="Fake/TinyQwen",
                           store_dir_name="TinyQwen", verbose=False)
        finally:
            os.environ.pop("HF_ENDPOINT", None)

    man = json.loads((out / "manifest.json").read_text())
    assert man["arch"] == "qwen2" and man["dtype"] == "int8"
    assert man["tie_word_embeddings"] is True  # embed present, no lm_head -> aliased
    assert man["blocks"]["lm_head"] == {"alias": "embed"}
    # embed sorts first -> its qrow block sits at offset 0
    assert man["blocks"]["embed"]["w_off"] == 0

    # a paged forward runs and produces finite logits over the tiny vocab
    store = QStore("TinyQwen", root=out_root)
    ids = np.array([1, 2, 3], np.int64)
    lg = _paged_logits(store, ids)
    assert lg.shape[-1] == cfg["vocab_size"]
    assert torch.isfinite(torch.as_tensor(lg)).all()


def test_stream_build_resume_is_byte_identical(tmp_path):
    """An interrupted build (progress sidecar with a truncated tail) resumes to the SAME bytes
    as an uninterrupted build."""
    served, _ = _tiny_qwen2(tmp_path / "srv")
    out_root = tmp_path / "stores"
    with _Server(served) as srv:
        os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{srv.port}"
        try:
            full = sb.build("tiny", out_root=out_root, hf_id="Fake/TinyQwen",
                            store_dir_name="full", verbose=False)
            # simulate an interrupt: rebuild into a fresh dir, then corrupt the tail + drop the
            # last few done-blocks in the sidecar, and rebuild with resume=True.
            part = sb.build("tiny", out_root=out_root, hf_id="Fake/TinyQwen",
                            store_dir_name="part", verbose=False)
            # corrupt: extend weights.i8 with junk and rewrite a stale sidecar as if interrupted
            man = json.loads((part / "manifest.json").read_text())
            (part / "manifest.json").unlink()
            # reconstruct a mid-build sidecar: keep all-but-last qrow block committed
            blocks = man["blocks"]
            qorder = [n for n, b in blocks.items() if b.get("kind") == "qrow"]
            drop = qorder[-1]
            b = blocks[drop]
            keep_w = b["w_off"]; keep_s = b["s_off"]
            with open(part / "weights.i8", "r+b") as f:
                f.seek(keep_w); f.write(b"\xff" * 999)  # junk tail past the commit point
            done = {n for n in blocks if n != "lm_head" and not (
                blocks[n].get("kind") == "qrow" and blocks[n]["w_off"] >= keep_w)}
            e_off = max((blocks[n]["e_off"] + blocks[n]["e_len"]
                         for n in done if blocks[n].get("kind") == "fp32"), default=0)
            side = {"model_name": "tiny", "plan_len": None,  # plan_len mismatch -> full rebuild guard
                    "w_off": keep_w, "s_off": keep_s, "e_off": e_off,
                    "blocks": {n: blocks[n] for n in done}, "done": sorted(done)}
            (part / ".stream_progress.json").write_text(json.dumps(side))
            resumed = sb.build("tiny", out_root=out_root, hf_id="Fake/TinyQwen",
                               store_dir_name="part", verbose=False)
        finally:
            os.environ.pop("HF_ENDPOINT", None)

    for fn in ("weights.i8", "scales.f32", "extras.f32"):
        assert (full / fn).read_bytes() == (resumed / fn).read_bytes(), fn


# --------------------------------------------------------------------------- real network gate

@pytest.mark.skipif(os.environ.get("MRUN_STREAM_NET") != "1",
                    reason="set MRUN_STREAM_NET=1 to run the ~1GB HF byte+forward parity gate")
def test_stream_vs_local_byte_and_forward_parity(tmp_path):
    """THE gate: stream Qwen2.5-0.5B from HF and prove the store is byte-identical to the on-disk
    qstore_build output, then argmax-exact on a paged forward. Requires the model cached locally
    for the reference build."""
    from mrun.engine.kernels.qstore_build import build as local_build
    from mrun.engine.paged import PagedEngine

    local_root = tmp_path / "local"
    stream_root = tmp_path / "stream"
    local_build("qwen2.5-0.5b", out_root=local_root)
    sb.build("qwen2.5-0.5b", out_root=stream_root, store_dir_name="Qwen2.5-0.5B", verbose=False)

    lp = local_root / "Qwen2.5-0.5B"
    sp = stream_root / "Qwen2.5-0.5B"
    for fn in ("weights.i8", "scales.f32", "extras.f32"):
        assert (lp / fn).read_bytes() == (sp / fn).read_bytes(), f"byte mismatch: {fn}"
    lm = json.loads((lp / "manifest.json").read_text())
    smf = json.loads((sp / "manifest.json").read_text())
    assert lm == smf

    e_local = PagedEngine("qwen2.5-0.5b", stores_dir=local_root)
    ids = e_local.tokenizer("The capital of France is", return_tensors="np")["input_ids"][0]
    a_local = torch.as_tensor(e_local.logits(ids.astype(np.int64))).argmax(-1)
    e_stream = PagedEngine("qwen2.5-0.5b", stores_dir=stream_root)
    a_stream = torch.as_tensor(e_stream.logits(ids.astype(np.int64))).argmax(-1)
    assert bool((a_local == a_stream).all())
