"""Standalone bounded-memory LoRA execution, extracted from the current runtime."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from ..engine.kernels.paged_forward import (
    _apply_rope_b,
    _rms_norm,
    _rope_tables,
    _rotate_half,
)
from ..engine.kernels.qstore import QStore
from ..models import resolve_model, store_name
from ..paths import stores_root as default_stores_root
from torch.utils.checkpoint import checkpoint

@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ("q", "v")  # classic LoRA default
    layers: tuple[int, ...] | None = None  # None = all layers
    seed: int = 0


class PagedLoRATrainer:
    """Frozen int8 base (streamed) + trainable LoRA (resident)."""

    def __init__(
        self,
        model_name: str,
        cfg: LoRAConfig | None = None,
        *,
        stores_dir: str | Path | None = None,
        compute_dtype: str | None = "bf16",
        cache_mb: float | None = None,
    ):
        spec = resolve_model(model_name)
        self.model_name = spec.name
        key = store_name(spec)
        root = Path(stores_dir) if stores_dir is not None else default_stores_root()
        if not (root / key / "manifest.json").exists():
            raise FileNotFoundError(
                f"no paged store at {root / key}. Build it with "
                f"`mrun build-store {spec.name}`."
            )
        # Fast paths (both default-ON, both no-op on cpu):
        #   compute_dtype="bf16" -> base projection matmuls run on tensor cores + the resident
        #     cache stores bf16 (half the bytes of fp32, so a 7B/14B actually fits the card).
        #   cache_mb=None -> auto-size the resident dequant cache to the free VRAM (minus a
        #     headroom for LoRA/optimizer/activations) so the base is paid ONCE, not re-dequanted
        #     every step; LRU-evicts the coldest weights if the model overflows the budget.
        self.store = QStore(key, root=root, compute_dtype=compute_dtype)
        if cache_mb is None:
            cache_mb = self._auto_cache_mb()
        self.store.set_cache_budget(cache_mb)
        self.arch = self.store.man.get("arch", "qwen2")
        if self.arch not in ("qwen2", "qwen3", "llama"):
            raise NotImplementedError(f"paged_lora supports qwen2/qwen3/llama, not {self.arch!r}")
        self.cfg = cfg or LoRAConfig()
        c = self.store.cfg
        self.d = c["hidden_size"]
        self.nL = c["num_hidden_layers"]
        self.nH, self.nKV, self.hd = (
            c["num_attention_heads"],
            c["num_key_value_heads"],
            c["head_dim"],
        )
        self.eps, self.theta = c["rms_norm_eps"], c["rope_theta"]
        self.V = c["vocab_size"]
        self.rep = self.nH // self.nKV
        self.scale = self.hd**-0.5
        self.layers = tuple(range(self.nL)) if self.cfg.layers is None else self.cfg.layers
        # per-(layer,target) output/input dims, read from the base block shapes
        self._dims = {
            t: tuple(self.store._resolve(f"L{self.layers[0]}.{t}")["shape"])
            for t in self.cfg.targets
        }
        self.lora = self._init_lora()

    # ---- LoRA params: A[r,in] (small init), B[out,r] (zero ⇒ ΔW=0 at init) ----------------
    def _init_lora(self) -> dict:
        g = torch.Generator(device=self.store.device).manual_seed(self.cfg.seed)
        lora: dict = {}
        for L in self.layers:
            for t in self.cfg.targets:
                out, inn = self._dims[t]
                A = torch.empty(
                    self.cfg.rank,
                    inn,
                    dtype=torch.float32,
                    device=self.store.device,
                )
                A.normal_(0.0, 1.0 / self.cfg.rank, generator=g)
                A.requires_grad_(True)
                B = torch.zeros(
                    out,
                    self.cfg.rank,
                    dtype=torch.float32,
                    device=self.store.device,
                    requires_grad=True,
                )
                lora[(L, t)] = {"A": A, "B": B}
        return lora

    def parameters(self) -> list[torch.Tensor]:
        ps = []
        for v in self.lora.values():
            ps += [v["A"], v["B"]]
        return ps

    def n_trainable(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    # ---- adapter persistence: the trainable delta is the ONLY thing worth saving — the frozen
    # int8 base stays in the QStore and is never touched, so a saved adapter is tiny (r*|targets|
    # *|layers| params) and portable across any run that resolves the SAME base store. ----------
    def save_adapter(self, path: str | Path) -> None:
        """Persist the trained LoRA deltas + config to `path` (a directory): `adapter.pt` (a flat
        state dict of every A/B tensor, CPU fp32, keyed ``"{layer}.{target}.{A|B}"``) and
        `adapter_config.json` (LoRAConfig + the base model/arch it was trained against, so a
        mismatched reload fails loudly instead of silently wiring the wrong shapes)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        state: dict[str, torch.Tensor] = {}
        for (L, t), ab in self.lora.items():
            state[f"{L}.{t}.A"] = ab["A"].detach().to("cpu", torch.float32).clone()
            state[f"{L}.{t}.B"] = ab["B"].detach().to("cpu", torch.float32).clone()
        torch.save(state, out / "adapter.pt")
        meta = {
            **asdict(self.cfg),
            "layers": list(self.layers),  # the RESOLVED layer set (cfg.layers may be None)
            "model_name": self.model_name,
            "arch": self.arch,
            "n_trainable": self.n_trainable(),
        }
        (out / "adapter_config.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    def apply_adapter(self, path: str | Path, *, strict: bool = True) -> None:
        """Load a saved adapter's A/B tensors into this trainer's LoRA state, in place (weights are
        `.copy_`'d so identity/requires_grad of the live parameters is preserved — safe to call
        mid-training or at inference). Raises loudly (not silently) on a base/config mismatch:
        different arch/model, or a (layer,target) present in the checkpoint but not in this
        trainer's ``targets``/``layers`` — a shape-mismatched copy is exactly the kind of silent
        corruption Rule #2 exists to catch. Set ``strict=False`` to tolerate a target/layer subset
        mismatch (extra keys in the checkpoint are ignored; missing ones raise)."""
        src = Path(path)
        meta = json.loads((src / "adapter_config.json").read_text())
        if strict and meta.get("arch") and meta["arch"] != self.arch:
            raise ValueError(
                f"adapter at {src} was trained on arch={meta['arch']!r}, this trainer is "
                f"arch={self.arch!r} — refusing a mismatched apply"
            )
        if strict and meta.get("model_name") and meta["model_name"] != self.model_name:
            raise ValueError(
                f"adapter at {src} was trained on model={meta['model_name']!r}, this trainer "
                f"resolved model={self.model_name!r} — refusing a mismatched apply"
            )
        state = torch.load(src / "adapter.pt", map_location="cpu")
        missing = []
        for (L, t), ab in self.lora.items():
            keyA, keyB = f"{L}.{t}.A", f"{L}.{t}.B"
            if keyA not in state or keyB not in state:
                missing.append((L, t))
                continue
            with torch.no_grad():
                ab["A"].copy_(state[keyA].to(ab["A"].device, ab["A"].dtype))
                ab["B"].copy_(state[keyB].to(ab["B"].device, ab["B"].dtype))
        if missing:
            raise KeyError(
                f"adapter at {src} is missing (layer,target) entries required by this trainer's "
                f"config: {missing[:8]}{'...' if len(missing) > 8 else ''}"
            )

    def _delta(self, x: torch.Tensor, L: int, t: str) -> torch.Tensor | None:
        """LoRA contribution to the (L,t) linear's output: (x @ Aᵀ) @ Bᵀ * (alpha/rank)."""
        ab = self.lora.get((L, t))
        if ab is None:
            return None
        return (x @ ab["A"].T) @ ab["B"].T * (self.cfg.alpha / self.cfg.rank)

    def _auto_cache_mb(self) -> float:
        """Resident-cache budget from FREE VRAM minus headroom for LoRA/AdamW/activations/
        logits (the base is frozen, so the rest of the card is fair game to hold it resident).
        Under the fleet, cap to the job's declared VRAM reservation (VRAM_LIMIT_MB, set by the
        agent executor) so the cache never overfills a shared card past what admission granted
        and trips the kill line (I11/I2). cpu -> 0 (stream; host-RAM cache is opt-in)."""
        dev = str(self.store.device)
        if dev.startswith("cuda"):
            free, _total = torch.cuda.mem_get_info()
            budget = float(free)
            limit_mb = os.environ.get("VRAM_LIMIT_MB", "").strip()
            if limit_mb:
                # reservation is the hard ceiling; don't plan to hold more resident than it,
                # even if the card physically has more free right now.
                budget = min(budget, float(limit_mb) * 1e6)
            headroom = 3.0e9
            return max(0.0, budget - headroom) / 1e6
        return 0.0

    def _lin(self, x: torch.Tensor, W: torch.Tensor, L: int, t: str) -> torch.Tensor:
        # base projection in W's dtype (bf16 -> tensor cores), result back to x's dtype (fp32)
        # for the residual add + the fp32 LoRA delta. No-op cast when dtypes already match.
        y = (x.to(W.dtype) @ W.T).to(x.dtype) if W.dtype != x.dtype else x @ W.T
        dz = self._delta(x, L, t)
        return y if dz is None else y + dz

    # ---- one transformer layer (streams base, applies LoRA); checkpointed -----------------
    def _layer(self, h, L, cos, sin, causal, key_pad):
        s = self.store
        B, Tmax, _ = h.shape
        x = _rms_norm(h, s.fp32(f"L{L}.ln1"), self.eps)
        Wq, Wk, Wv = s.weight(f"L{L}.q"), s.weight(f"L{L}.k"), s.weight(f"L{L}.v")
        q = self._lin(x, Wq, L, "q")
        k = self._lin(x, Wk, L, "k")
        v = self._lin(x, Wv, L, "v")
        if s.has(f"L{L}.q.bias"):
            q = q + s.fp32(f"L{L}.q.bias")
            k = k + s.fp32(f"L{L}.k.bias")
            v = v + s.fp32(f"L{L}.v.bias")
        q = q.view(B, Tmax, self.nH, self.hd)
        k = k.view(B, Tmax, self.nKV, self.hd)
        if s.has(f"L{L}.q_norm"):
            q = _rms_norm(q, s.fp32(f"L{L}.q_norm"), self.eps)
            k = _rms_norm(k, s.fp32(f"L{L}.k_norm"), self.eps)
        q = _apply_rope_b(q, cos, sin)
        k = _apply_rope_b(k, cos, sin)
        v = v.view(B, Tmax, self.nKV, self.hd)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=2)
            v = v.repeat_interleave(self.rep, dim=2)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale + causal + key_pad
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vh).transpose(1, 2).reshape(B, Tmax, self.nH * self.hd)
        Wo = s.weight(f"L{L}.o")
        attn_out = self._lin(ctx, Wo, L, "o")
        if s.has(f"L{L}.o.bias"):
            attn_out = attn_out + s.fp32(f"L{L}.o.bias")
        h = h + attn_out

        x2 = _rms_norm(h, s.fp32(f"L{L}.ln2"), self.eps)
        Wg, Wu = s.weight(f"L{L}.gate"), s.weight(f"L{L}.up")
        g = self._lin(x2, Wg, L, "gate")
        u = self._lin(x2, Wu, L, "up")
        hid = torch.nn.functional.silu(g) * u
        Wd = s.weight(f"L{L}.down")
        h = h + self._lin(hid, Wd, L, "down")
        return h

    def _embed_and_mask(self, ids_list):
        B = len(ids_list)
        lengths = np.array([len(x) for x in ids_list], dtype=np.int64)
        Tmax = int(lengths.max())
        ids_pad = np.zeros((B, Tmax), dtype=np.int64)
        real = torch.zeros((B, Tmax), dtype=torch.bool, device=self.store.device)
        for b, x in enumerate(ids_list):
            ids_pad[b, : len(x)] = np.asarray(x, np.int64)
            real[b, : len(x)] = True
        h = self.store.embed_rows("embed", ids_pad.reshape(-1)).clone().view(B, Tmax, self.d)
        cos, sin = _rope_tables(Tmax, self.hd, self.theta)
        cos, sin = cos.to(h.device), sin.to(h.device)
        causal = torch.triu(torch.full((Tmax, Tmax), float("-inf"), device=h.device), diagonal=1)
        key_pad = torch.where(real, 0.0, float("-inf"))[:, None, None, :]
        return h, cos, sin, causal, key_pad, lengths, real

    def forward_logits(self, ids_list, *, grad: bool = True, embed=None) -> torch.Tensor:
        """Full logits [B, Tmax, V] with grad through LoRA. Each layer is checkpointed so base
        weights are re-streamed (not pinned) in backward — keeps resident O(largest matrix).
        Pass ``embed`` (the ``_embed_and_mask`` tuple) to reuse a precomputed embed/mask; ``None``
        recomputes it (preserves behaviour for standalone callers). The embed tensors carry no
        grad, so computing them outside this grad context is numerically identical."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if embed is None:
                embed = self._embed_and_mask(ids_list)
            h, cos, sin, causal, key_pad, _, _ = embed
            for L in range(self.nL):
                fn = lambda hh, LL=L: self._layer(hh, LL, cos, sin, causal, key_pad)
                if grad:
                    h = checkpoint(fn, h, use_reentrant=False)
                else:
                    h = fn(h)
            h = _rms_norm(h, self.store.fp32("norm.final"), self.eps)
            B, Tmax, _ = h.shape
            logits = torch.empty((B, Tmax, self.V), dtype=torch.float32, device=h.device)
            for start, end, Wblk in self.store.row_blocks("lm_head"):
                logits[:, :, start:end] = h @ Wblk.T
            return logits

    def _lm_logits(self, h_flat: torch.Tensor) -> torch.Tensor:
        """Logits for a flat [N, d] hidden batch: [N, V]. Streams lm_head in row blocks."""
        N = h_flat.shape[0]
        out = torch.empty((N, self.V), dtype=torch.float32, device=h_flat.device)
        for start, end, Wblk in self.store.row_blocks("lm_head"):
            out[:, start:end] = h_flat @ Wblk.T
        return out

    @torch.no_grad()
    def generate(self, prompt_ids_list, max_new_tokens: int, eos_id=None) -> list[list[int]]:
        """KV-cached greedy batched generation. Returns generated-token lists (prompt excluded).

        Decode is O(T) per step (attend the new query against cached K/V) instead of the O(T^2)
        full re-forward. Right-pads prompts and tracks per-row absolute RoPE positions, so a
        mixed-length batch decodes correctly; a growing key-validity mask hides prompt padding.
        Matches the full-re-forward greedy path token-for-token (tests/test_paged_lora KV parity)."""
        s = self.store
        dev = s.device
        cdt = s.compute_dtype
        B = len(prompt_ids_list)
        lengths = [len(p) for p in prompt_ids_list]
        Tp = max(lengths)
        Ttot = Tp + max_new_tokens
        cos_t, sin_t = _rope_tables(Ttot, self.hd, self.theta)
        cos_t, sin_t = cos_t.to(dev), sin_t.to(dev)

        def rope_bt(x, pos_ids):  # x [B,T,n,hd]; pos_ids [B,T] -> per-row/per-pos rotation
            c = cos_t[pos_ids][:, :, None, :]
            si = sin_t[pos_ids][:, :, None, :]
            return x * c + _rotate_half(x) * si

        def qkv(x, L):
            q = self._lin(x, s.weight(f"L{L}.q"), L, "q")
            k = self._lin(x, s.weight(f"L{L}.k"), L, "k")
            v = self._lin(x, s.weight(f"L{L}.v"), L, "v")
            if s.has(f"L{L}.q.bias"):
                q = q + s.fp32(f"L{L}.q.bias")
                k = k + s.fp32(f"L{L}.k.bias")
                v = v + s.fp32(f"L{L}.v.bias")
            T = x.shape[1]
            q = q.view(B, T, self.nH, self.hd)
            k = k.view(B, T, self.nKV, self.hd)
            v = v.view(B, T, self.nKV, self.hd)
            if s.has(f"L{L}.q_norm"):
                q = _rms_norm(q, s.fp32(f"L{L}.q_norm"), self.eps)
                k = _rms_norm(k, s.fp32(f"L{L}.k_norm"), self.eps)
            return q, k, v

        def mlp(h, L):
            x2 = _rms_norm(h, s.fp32(f"L{L}.ln2"), self.eps)
            g = self._lin(x2, s.weight(f"L{L}.gate"), L, "gate")
            u = self._lin(x2, s.weight(f"L{L}.up"), L, "up")
            hid = torch.nn.functional.silu(g) * u
            return h + self._lin(hid, s.weight(f"L{L}.down"), L, "down")

        def attn(qh, kh, vh, bias):  # qh [B,nH,Tq,hd]; kh/vh [B,nKV,Tk,hd]; bias [B,1,Tq,Tk]
            if self.rep > 1:
                kh = kh.repeat_interleave(self.rep, dim=1)
                vh = vh.repeat_interleave(self.rep, dim=1)
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale + bias
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, vh)  # [B,nH,Tq,hd]
            return ctx.transpose(1, 2).reshape(B, qh.shape[2], self.nH * self.hd)

        # ---- prefill: full prompt, cache per-layer K/V (pre-GQA-repeat) ----
        ids_pad = np.zeros((B, Tp), dtype=np.int64)
        real = torch.zeros((B, Tp), dtype=torch.bool, device=dev)
        for b, p in enumerate(prompt_ids_list):
            ids_pad[b, : lengths[b]] = np.asarray(p, np.int64)
            real[b, : lengths[b]] = True
        h = s.embed_rows("embed", ids_pad.reshape(-1)).clone().view(B, Tp, self.d).to(cdt)
        pos_p = torch.arange(Tp, device=dev)[None, :].expand(B, Tp)
        causal = torch.triu(torch.full((Tp, Tp), float("-inf"), device=dev), diagonal=1)
        key_pad = torch.where(real, 0.0, float("-inf"))[:, None, None, :]  # [B,1,1,Tp]
        past_k, past_v = [None] * self.nL, [None] * self.nL
        for L in range(self.nL):
            x = _rms_norm(h, s.fp32(f"L{L}.ln1"), self.eps)
            q, k, v = qkv(x, L)
            q = rope_bt(q, pos_p)
            k = rope_bt(k, pos_p)
            past_k[L], past_v[L] = k, v
            ctx = attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal + key_pad)
            ao = self._lin(ctx, s.weight(f"L{L}.o"), L, "o")
            if s.has(f"L{L}.o.bias"):
                ao = ao + s.fp32(f"L{L}.o.bias")
            h = mlp(h + ao, L)
        hN = _rms_norm(h, s.fp32("norm.final"), self.eps)
        lengths_t = torch.as_tensor(lengths, device=dev)
        h_last = hN[torch.arange(B, device=dev), lengths_t - 1]  # [B,d]
        nxt = self._lm_logits(h_last.float()).argmax(-1)  # [B]

        # ---- decode: one new token per row per step, attend vs cached K/V ----
        gen = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        kmask = real.clone()  # [B, T_cur], grows one col/step (new slot always valid)
        for step in range(max_new_tokens):
            hit_eos = (
                (nxt == eos_id)
                if eos_id is not None
                else torch.zeros(B, dtype=torch.bool, device=dev)
            )
            for b in range(B):
                if not done[b] and not bool(hit_eos[b]):
                    gen[b].append(
                        int(nxt[b])
                    )  # EOS is neither emitted nor continued (matches reference)
            done = done | hit_eos
            if bool(done.all()):
                break
            h1 = (
                s.embed_rows("embed", nxt.detach().cpu().numpy().reshape(-1))
                .clone()
                .view(B, 1, self.d)
                .to(cdt)
            )
            pos1 = (lengths_t + step)[:, None]  # [B,1] absolute position of the new token
            for L in range(self.nL):
                x = _rms_norm(h1, s.fp32(f"L{L}.ln1"), self.eps)
                q, k, v = qkv(x, L)
                q = rope_bt(q, pos1)
                k = rope_bt(k, pos1)
                past_k[L] = torch.cat([past_k[L], k], dim=1)
                past_v[L] = torch.cat([past_v[L], v], dim=1)
                bias = torch.where(
                    torch.cat([kmask, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=1),
                    0.0,
                    float("-inf"),
                )[:, None, None, :]  # [B,1,1,T_cur+1]
                ctx = attn(
                    q.transpose(1, 2), past_k[L].transpose(1, 2), past_v[L].transpose(1, 2), bias
                )
                ao = self._lin(ctx, s.weight(f"L{L}.o"), L, "o")
                if s.has(f"L{L}.o.bias"):
                    ao = ao + s.fp32(f"L{L}.o.bias")
                h1 = mlp(h1 + ao, L)
            kmask = torch.cat([kmask, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=1)
            hN = _rms_norm(h1[:, 0], s.fp32("norm.final"), self.eps)
            nxt = self._lm_logits(hN.float()).argmax(-1)
        return gen

    def loss(self, ids_list) -> torch.Tensor:
        """Next-token CE over real (non-pad) positions, averaged."""
        embed = self._embed_and_mask(ids_list)  # compute embed/mask ONCE
        real = embed[-1]  # (h,cos,sin,causal,key_pad,lengths,real)
        logits = self.forward_logits(ids_list, grad=True, embed=embed)  # [B,Tmax,V]
        B, Tmax, _ = logits.shape
        ids_pad = torch.zeros((B, Tmax), dtype=torch.long, device=logits.device)
        for b, x in enumerate(ids_list):
            ids_pad[b, : len(x)] = torch.as_tensor(np.asarray(x, np.int64), device=logits.device)
        # predict token t+1 from position t; valid where both t and t+1 are real
        pred = logits[:, :-1, :].reshape(-1, self.V)
        tgt = ids_pad[:, 1:].reshape(-1)
        valid = real[:, 1:].reshape(-1)
        return torch.nn.functional.cross_entropy(pred[valid], tgt[valid])

    def step(self, ids_list, opt: torch.optim.Optimizer) -> float:
        opt.zero_grad(set_to_none=True)
        L = self.loss(ids_list)
        L.backward()
        opt.step()
        return float(L.item())

    @property
    def working_set_mb(self) -> float:
        """O(largest single dequantized block) — populated after the first forward."""
        return self.store.max_block_bytes / 1e6
