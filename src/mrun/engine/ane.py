"""Core ML accelerator-eligible backend — DEMOTED 2026-07-24, use `mlx` instead.

Core ML's reason to exist here is that it is the only public route to the Neural Engine.
MLComputePlan measured this path placing **98% of its operations on the GPU** (NE fraction
0.011), and on identical work `mlx` is faster on every axis: **2196.8 vs 1694.5 tok/s warm,
0.62 s vs 66.93 s cold**. Everything below is therefore a GPU path carrying a Core ML tax —
19.3 s compile per (B,T), 943 MB per compiled package, an fp16->fp32 upcast inside the native
`predict()` binding, output specs silently dropped on name mismatch, placement inspectable only
via an API that aborts the process, and `CPU_AND_NE` (the one setting that reaches the NE)
returning 0/8 correct answers.

Keep it for Core ML/ANE investigation. For Apple speed use `backend="mlx"` (or the alias
`backend="apple"`).

Original notes follow.

Core ML accelerator-eligible backend (opt-in speed, qwen2/llama, fp16).

Earlier measurements called this the ANE path, but the implementation requests
``ct.ComputeUnit.ALL``. That permits Core ML to place operations on CPU, GPU, Neural Engine,
or a mixture; it does not prove ANE placement. ``ANEPagedEngine(PagedEngine)`` is retained as
a compatibility name. It overrides only ``logits_batch`` to use a compiled, cached Core ML
model; everything else stays on the proven paged path.

The Core ML path is **fp16 -> an OPT-IN SPEED backend, not the canonical-measurement path**:
argmax-exact on real text, but never gate the [M] content / CE leg on random-token parity
(fp16 error compounds ~100x over 24 layers off-distribution).

MEASURED error vs paged (Qwen2.5-0.5B, 8 real-text prompts, B=8/T=9, 2026-07-24):
last-token max|dlogit| **5.01**, last-token max|dmargin| **0.462**, argmax 8/8. The deltas are
uniform across positions (~1.7-3.7), not confined to interior tokens. So "argmax-exact on real
text" holds only where the true margin clears the error: **a probe whose margin is below ~0.5
can flip on this path regardless of how natural its text is.** Treat 0.5 as a margin floor,
not "real text" as a safety criterion. Identical numbers from a fresh compile and from a
disk-cached package, so serialization is numerically neutral. Correctness never depends on the
accelerator path: any miss (non-qwen2/llama, coremltools absent, compile/predict error, B or T past the
grid) falls back to ``PagedEngine.logits_batch``.

DESIGN (correct + bounded):
  - the causal mask is BAKED (CoreML mis-converts a runtime additive mask in this attention
    pattern); ``logits_batch`` sub-groups by EXACT length so a per-T causal model is correct;
  - weights bake ~1 GB/shape into each compiled model -> an LRU cache (default 2) bounds RAM;
  - the batch dim B snaps to a grid (B in {1,2,4,8,16,32}) so a probe suite hits a handful of
    compiles; the sequence dim does NOT snap — one compiled CoreML model is built per EXACT
    sequence length T (the baked causal mask is per-T). Pad rows (batch dim) are zeros and
    dropped on output.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paged import PagedEngine

B_GRID = (1, 2, 4, 8, 16, 32)


def _fp16_io() -> bool:
    """fp16 model I/O boundary. Default ON, MEASURED 2026-07-24.

    With no dtype the boundary defaults to fp32 both ways, so a B=8/T=9 call materialises a
    43.8 MB logit tensor where fp16 is 21.9 MB — and output bytes scale with T. Parity is
    EXACT, not merely close: argmax 8/8 and max|dmargin| 0.46188, bit-identical to the fp32
    boundary, because compute_precision is already FLOAT16 internally and the fp32 output was
    a lossless widening of fp16 values.

    Read through ONE function because the convert side and the predict side must agree; two
    `os.environ.get` calls with different defaults would feed fp32 arrays to an fp16 input.
    """
    return os.environ.get("MRUN_ANE_FP16_IO", "1") == "1"


def _cache_max_gb() -> float:
    """Disk budget for compiled packages. One (B,T) package bakes the fp16 weights —
    MEASURED 943 MB for Qwen2.5-0.5B at (8,9) — and T does not snap to a grid, so a
    mixed-length suite would otherwise grow this without bound inside the stores tree."""
    try:
        return max(0.0, float(os.environ.get("MRUN_ANE_CACHE_MAX_GB", "20")))
    except ValueError:
        return 20.0


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _evict_disk_cache(base: Path, *, keep_gb: float) -> None:
    """Evict least-recently-used packages until the tree fits ``keep_gb``.

    Never raises: a cache that cannot be pruned must not fail the run that filled it.
    """
    try:
        pkgs = sorted(
            (p for p in base.glob("*.mlpackage") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        budget = keep_gb * 1e9
        total = sum(_dir_size(p) for p in pkgs)
        while pkgs and total > budget:
            victim = pkgs.pop(0)                       # oldest mtime first
            size = _dir_size(victim)
            shutil.rmtree(victim, ignore_errors=True)
            total -= size
            if os.environ.get("MODEL_EXPERIMENTS_ANE_DEBUG"):
                print(f"[CoreML] disk-cache evicted {victim.name} ({size/1e9:.2f} GB)",
                      file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("MODEL_EXPERIMENTS_ANE_DEBUG"):
            print(f"[CoreML] disk-cache eviction skipped ({type(e).__name__}: {e})",
                  file=sys.stderr)


def _snap(n: int, grid: tuple[int, ...]) -> int | None:
    for g in grid:
        if n <= g:
            return g
    return None                                   # past the grid -> caller falls back


def _rope_tables(T: int, hd: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos(), emb.sin()


class _QwenANE(torch.nn.Module):
    """Exact mirror of paged_forward.paged_logits (qwen2/llama) with weights baked fp16, a baked
    causal mask, and the FULL lm_head. Input: h [B,T,d] (token embeddings)."""

    def __init__(self, store, B: int, T: int):
        super().__init__()
        c = store.cfg
        self.B, self.T = B, T
        self.d = int(c["hidden_size"]); self.nL = int(c["num_hidden_layers"])
        self.nH = int(c["num_attention_heads"]); self.nKV = int(c["num_key_value_heads"])
        self.hd = int(c["head_dim"]); self.eps = float(c["rms_norm_eps"])
        self.rep = self.nH // self.nKV; self.scale = self.hd ** -0.5
        cos, sin = _rope_tables(T, self.hd, float(c["rope_theta"]))
        self.register_buffer("cos", cos); self.register_buffer("sin", sin)
        # Causal mask BAKED. CoreML converts a baked mask correctly (corr 1.0) but mis-converts a
        # RUNTIME additive-mask input in this attention pattern (corr ~0.5 garbage). So each
        # compiled model assumes all rows are the same length T; logits_batch sub-groups by exact
        # length so that holds. -30000 is finite-safe in fp16 (max 65504); exp(-30000+s)≈0.
        self.register_buffer("causal", torch.triu(torch.full((T, T), -30000.0), 1))
        W = lambda n: store.weight(n).float()
        F = lambda n: store.fp32(n)
        self._has_qkv_bias = store.has("L0.q.bias")
        for L in range(self.nL):
            for nm in ("q", "k", "v", "o", "gate", "up", "down"):
                self.register_buffer(f"{nm}{L}", W(f"L{L}.{nm}"))
            if self._has_qkv_bias:
                for nm in ("q", "k", "v"):
                    self.register_buffer(f"{nm}b{L}", F(f"L{L}.{nm}.bias"))
            self.register_buffer(f"ln1_{L}", F(f"L{L}.ln1"))
            self.register_buffer(f"ln2_{L}", F(f"L{L}.ln2"))
        self.register_buffer("nf", F("norm.final"))
        self.register_buffer("lm", W("lm_head"))

    def _rms(self, x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

    def _rope(self, x):
        H = self.hd // 2
        return (x * self.cos[None, :, None, :]
                + torch.cat([-x[..., H:], x[..., :H]], -1) * self.sin[None, :, None, :])

    def forward(self, h):
        B, T, nH, nKV, hd, rep = self.B, self.T, self.nH, self.nKV, self.hd, self.rep
        mask = self.causal[None, None]                 # baked causal; all rows length T (no key-pad)
        for L in range(self.nL):
            x = self._rms(h, getattr(self, f"ln1_{L}"))
            q = x @ getattr(self, f"q{L}").T; k = x @ getattr(self, f"k{L}").T; v = x @ getattr(self, f"v{L}").T
            if self._has_qkv_bias:
                q = q + getattr(self, f"qb{L}"); k = k + getattr(self, f"kb{L}"); v = v + getattr(self, f"vb{L}")
            q = self._rope(q.view(B, T, nH, hd)); k = self._rope(k.view(B, T, nKV, hd)); v = v.view(B, T, nKV, hd)
            k = k.reshape(B, T, nKV, 1, hd).expand(B, T, nKV, rep, hd).reshape(B, T, nH, hd)
            v = v.reshape(B, T, nKV, 1, hd).expand(B, T, nKV, rep, hd).reshape(B, T, nH, hd)
            qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            p = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * self.scale + mask, -1)
            ctx = torch.matmul(p, vh).transpose(1, 2).reshape(B, T, nH * hd)
            h = h + ctx @ getattr(self, f"o{L}").T
            x2 = self._rms(h, getattr(self, f"ln2_{L}"))
            g = x2 @ getattr(self, f"gate{L}").T; u = x2 @ getattr(self, f"up{L}").T
            h = h + (torch.nn.functional.silu(g) * u) @ getattr(self, f"down{L}").T
        return self._rms(h, self.nf) @ self.lm.T


class ANEPagedEngine(PagedEngine):
    """PagedEngine whose batched logits are Core ML accelerator-eligible.

    ``backend="ane"`` remains a compatibility alias. Report the execution fabric as Core ML
    unless model-specific MLComputePlan evidence verifies preferred devices.
    """

    backend = "ane"
    reported_fabric = "coreml-unverified"
    # Winners are only trustworthy above this backend's own numerical error. MEASURED
    # max|dmargin| 0.462 vs paged (Qwen2.5-0.5B, real text, B=8/T=9, 2026-07-24); rounded up.
    # BaseEngine.score_forced_choice_* flags rows under it rather than dropping or "fixing"
    # them — the caller decides whether to re-score those probes on an exact backend.
    margin_floor = 0.5

    def __init__(self, model_name: str, *, cache_size: int = 2, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self._ane_ok = self.arch in ("qwen2", "llama")
        self._last_fallback = False                   # True iff last logits_batch fell back to paged
        self._cache: OrderedDict[tuple[int, int], object] = OrderedDict()
        self._cache_size = int(cache_size)
        self._compiled_shapes: set[tuple[int, int]] = set()
        self._disk_cache_hits = 0
        self._disk_cache_misses = 0
        self._units_checked = False        # non-default compute units are parity-gated once
        self._last_execution_path = "not-run"
        self._requested_backend_alias = "ane"
        # Why the accelerator path is unavailable, if it is. Evidence already reports THAT a
        # call fell back; without this it cannot say whether the cause was the arch, a missing
        # toolchain, or a compile error — and "backend=ane running 100% paged because
        # coremltools is not installed in this venv" is the easiest of the three to miss.
        self._unavailable_reason: str | None = (
            None if self._ane_ok else f"arch {self.arch!r} not in (qwen2, llama)"
        )
        try:
            import coremltools as ct
            self._ct = ct
        except Exception as e:  # noqa: BLE001
            self._ct = None
            self._ane_ok = False
            self._unavailable_reason = f"coremltools unavailable ({type(e).__name__})"

    def _compute_units(self):
        """Which fabrics Core ML may place on. Default **ALL** — see the warning below.

        MEASURED 2026-07-24. Under `ALL`, MLComputePlan showed Core ML putting 1296 of 1327
        real ops on the GPU and only 30 on the Neural Engine (NE fraction 0.011): every op
        reports all three devices supported, so the planner chose Metal freely. Denying it the
        GPU with `CPU_AND_NE` measured **1.415x faster** (2451.4 vs 1729.7 tok/s, Qwen2.5-0.5B
        B=8/T=9, median of 5, both pre-warmed).

        **That speedup was briefly made the default and then REVERTED: CPU_AND_NE IS FAST AND
        WRONG.** It was measured on throughput alone. Adding a parity gate gave **argmax 0/8 vs
        paged, max|dmargin| 3.997** — against 8/8 and 0.462 for `ALL`. A path that returns
        different winners is not a faster path. Root cause not established; candidates are an
        NE-specific numerical path, or the baked causal mask lowering differently once the GPU
        is denied — the same conversion surface as the historical runtime-mask bug.

        `MRUN_ANE_COMPUTE_UNITS=ALL|CPU_AND_NE|CPU_ONLY` overrides; CPU_AND_NE is for
        investigation only until it passes parity.
        """
        name = os.environ.get("MRUN_ANE_COMPUTE_UNITS", "ALL").strip().upper()
        return getattr(self._ct.ComputeUnit, name, self._ct.ComputeUnit.ALL)

    def _disk_cache_dir(self, B: int, T: int):
        """Where a compiled (B,T) package for THIS store lives, or None if disabled.

        Compiling one shape costs ~22-30 s and the in-process LRU dies with the process,
        so every run re-paid every compile. The key must pin everything that changes the
        compiled artifact: the store identity (weights are baked in), the shape, and the
        toolchain that lowered it — a coremltools upgrade must miss, not silently reuse.
        """
        if os.environ.get("MRUN_ANE_DISK_CACHE") == "0":
            return None
        try:
            store_dir = Path(self.store.directory)
            root = os.environ.get("MRUN_ANE_CACHE_DIR")
            # sibling of the store, never inside it: the store directory is subject to
            # manifest/blob integrity checks and must not gain unexpected files.
            base = Path(root) if root else store_dir.parent / ".mlcache"
            # Prefer a real content hash. This store family is often `legacy-unverified`
            # with every identity field None, so degrade to hashing the manifest bytes —
            # which still changes when the store is rebuilt. identity_status is part of the
            # key so a legacy-keyed package can never be served to a verified store.
            stamp = (
                getattr(self.store, "derived_store_sha256", None)
                or getattr(self.store, "manifest_semantic_sha256", None)
                or hashlib.sha256((store_dir / "manifest.json").read_bytes()).hexdigest()
            )
            status = getattr(self.store, "identity_status", "unknown")
            # compute_units is part of the key: ALL and CPU_AND_NE are different placement
            # contracts. Reusing one for the other could bypass the parity gate or misreport
            # which fabrics were eligible.
            sig = hashlib.sha256(
                f"{store_dir.name}|{status}|{stamp}|{self.arch}|{self.hidden}|"
                # effective value, not the raw env: unset and "=1" build the SAME artifact and
                # must share one key, or the cache fragments at ~943 MB per package.
                f"{B}x{T}|{self._compute_units()}|fp16io={_fp16_io()}|"
                f"ct{getattr(self._ct, '__version__', '?')}".encode()
            ).hexdigest()[:16]
            return base / f"{store_dir.name}-B{B}-T{T}-{sig}.mlpackage"
        except Exception as e:  # noqa: BLE001 — cache addressing must never break a run
            if os.environ.get("MODEL_EXPERIMENTS_ANE_DEBUG"):
                print(f"[CoreML] disk-cache disabled ({type(e).__name__}: {e})", file=sys.stderr)
            return None

    def _get_model(self, B: int, T: int):
        key = (B, T)
        m = self._cache.get(key)
        if m is not None:
            self._cache.move_to_end(key)
            return m
        ct = self._ct
        path = self._disk_cache_dir(B, T)
        ml = None
        if path is not None and path.exists():
            try:
                # Loading a package has its own compute-unit argument. Omitting it silently
                # reverts to Core ML's default and can make a CPU_AND_NE cache execute under
                # ALL while evidence still reports the requested policy.
                ml = ct.models.MLModel(str(path), compute_units=self._compute_units())
                self._disk_cache_hits += 1
            except Exception as e:  # noqa: BLE001 — corrupt/stale package: recompile, never fail
                if os.environ.get("MODEL_EXPERIMENTS_ANE_DEBUG"):
                    print(f"[CoreML] disk-cache miss ({type(e).__name__}: {e})", file=sys.stderr)
                ml = None
        if ml is None:
            mod = _QwenANE(self.store, B, T).eval()
            h0 = torch.zeros(B, T, self.hidden)
            traced = torch.jit.trace(mod, (h0,))
            # fp16 I/O (opt-in, MRUN_ANE_FP16_IO=1). MEASURED 2026-07-24: with no dtype the
            # boundary defaults to fp32 BOTH ways, so a B=8/T=9 call ships a 43.8 MB logit
            # tensor that is 21.9 MB in fp16 — and output bytes scale with T, so the cost
            # grows linearly with sequence length. Two cautions found the hard way:
            #   * an output spec whose NAME does not match a real output is silently dropped,
            #     so this asserts the spec afterwards instead of trusting the request;
            #   * even with the spec at FLOAT16, coremltools' Python predict() upcasts the
            #     result to float32 on return — the model emits fp16, the binding hands back
            #     fp32. Declaring fp16 still halves the buffer the binding reads from, but the
            #     copy is not avoidable through convert().
            fp16_io = _fp16_io()
            in_kw = {"dtype": np.float16} if fp16_io else {}
            ml = ct.convert(
                traced,
                inputs=[ct.TensorType(name="h", shape=(B, T, self.hidden), **in_kw)],
                compute_units=self._compute_units(),
                compute_precision=ct.precision.FLOAT16,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS14,
            )
            if fp16_io:
                oname = ml._spec.description.output[0].name
                ml = ct.convert(
                    traced,
                    inputs=[ct.TensorType(name="h", shape=(B, T, self.hidden), **in_kw)],
                    outputs=[ct.TensorType(name=oname, dtype=np.float16)],
                    compute_units=self._compute_units(),
                    compute_precision=ct.precision.FLOAT16,
                    convert_to="mlprogram",
                    minimum_deployment_target=ct.target.macOS14,
                )
                got = ml._spec.description.output[0].type.multiArrayType.dataType
                if got != 65552:      # FLOAT16 in the Core ML proto enum
                    raise RuntimeError(
                        f"MRUN_ANE_FP16_IO requested but output dtype is {got}, not FLOAT16 — "
                        "coremltools dropped the spec; refusing to claim an fp16 boundary"
                    )
            self._disk_cache_misses += 1
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ml.save(str(path))
                    _evict_disk_cache(path.parent, keep_gb=_cache_max_gb())
                except Exception as e:  # noqa: BLE001 — read-only/full disk must not fail a run
                    if os.environ.get("MODEL_EXPERIMENTS_ANE_DEBUG"):
                        print(f"[CoreML] disk-cache save failed ({type(e).__name__}: {e})",
                              file=sys.stderr)
        self._cache[key] = ml
        self._compiled_shapes.add(key)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)        # LRU evict (frees ~1 GB of baked weights)
        return ml

    def _self_check_nondefault_units(self, ids_list: list[np.ndarray], out: list) -> bool:
        """Argmax-check a NON-DEFAULT compute unit against paged, once, on first use.

        `CPU_AND_NE` — the only setting that actually reaches the Neural Engine — measured
        **0/8 argmax agreement** with paged (max|dmargin| 3.997) on 2026-07-24 while running
        1.415x faster. A setting that is fast and wrong is worse than one that is merely slow,
        and an env var is not a safe place to keep such a path armed: anyone flipping
        MRUN_ANE_COMPUTE_UNITS for a speed experiment would silently corrupt every winner.

        So a non-default unit must EARN its first call. `ALL` is exempt (it is the measured-good
        default and paying a paged forward per engine would defeat the point). Failure disables
        the accelerator for this engine's lifetime rather than warning into a log nobody reads.
        """
        try:
            ref = super().logits_batch(ids_list)
            agree = sum(
                int(a[-1].argmax() == b[-1].argmax())
                for a, b in zip(out, ref, strict=True)
            )
            if agree == len(ref):
                return True
            self._ane_ok = False
            self._unavailable_reason = (
                f"compute_units={self._compute_units()} FAILED parity self-check "
                f"({agree}/{len(ref)} argmax vs paged) — accelerator disabled for this engine"
            )
            print(f"[CoreML] {self._unavailable_reason}", file=sys.stderr)
            return False
        except Exception:  # noqa: BLE001 — cannot verify => do not trust
            self._ane_ok = False
            self._unavailable_reason = "parity self-check could not run; accelerator disabled"
            return False

    # PagedEngine's subset scorer performs a paged body forward. Exposing it here would make
    # argmax_only=True silently leave Core ML while the engine still reports backend="ane".
    # The generic batched scorer instead uses this class's Core ML logits_batch path.
    score_forced_choice_argmax_subset = None

    def logits(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
    ) -> torch.Tensor:
        """Route an ordinary single-row forward through the Core ML batch surface."""
        if patch_ops_by_layer:
            self._last_fallback = True
            self._last_execution_path = "paged-patched-logits"
            return super().logits(ids, patch_ops_by_layer=patch_ops_by_layer)
        return self.logits_batch([ids])[0]

    def last_logits_batch(self, ids_list: list[np.ndarray]) -> torch.Tensor:
        """Core ML has no last-token-only graph yet; select from its full batched output."""
        return torch.stack([row[-1] for row in self.logits_batch(ids_list)])

    def selected_last_logits_batch(
        self,
        ids_list: list[np.ndarray],
        token_ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        """Keep selected scoring on Core ML, then gather from the full-vocabulary result."""
        indices = torch.as_tensor(token_ids, dtype=torch.long)
        return self.last_logits_batch(ids_list).index_select(1, indices)

    def generate(self, prompt: str | list[int] | np.ndarray, **kwargs: Any) -> list[int] | str:
        """Generation is an explicit paged fallback until a stateful Core ML graph exists."""
        self._last_fallback = True
        self._last_execution_path = "paged-generation"
        return super().generate(prompt, **kwargs)

    def _mark_paged_surface(self, name: str) -> None:
        self._last_fallback = True
        self._last_execution_path = f"paged-{name}"

    def forward_acts(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("forward-acts")
        return super().forward_acts(*args, **kwargs)

    def forward_acts_batch(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("forward-acts-batch")
        return super().forward_acts_batch(*args, **kwargs)

    def forward_acts_resid(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("forward-acts-resid")
        return super().forward_acts_resid(*args, **kwargs)

    def forward_patched(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("forward-patched")
        return super().forward_patched(*args, **kwargs)

    def forward_patched_batch(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("forward-patched-batch")
        return super().forward_patched_batch(*args, **kwargs)

    def hidden_states(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("hidden-states")
        return super().hidden_states(*args, **kwargs)

    def hidden_states_batch(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("hidden-states-batch")
        return super().hidden_states_batch(*args, **kwargs)

    def forward_attns(self, *args: Any, **kwargs: Any):
        self._mark_paged_surface("forward-attns")
        return super().forward_attns(*args, **kwargs)

    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        if not self._ane_ok:
            self._last_fallback = True
            self._last_execution_path = "paged-unavailable"
            return super().logits_batch(ids_list)
        self._last_fallback = False
        ids = [np.asarray(x, np.int64) for x in ids_list]
        out: list[torch.Tensor | None] = [None] * len(ids)
        try:
            # sub-group by EXACT length: a compiled model bakes causal for one T (no key-padding),
            # then chunk each length-group into <=max(B_GRID) rows. Pad ROWS (batch dim) are zeros
            # and unread — attention is per-row, so they never touch real rows.
            by_len: dict[int, list[int]] = defaultdict(list)
            for i, x in enumerate(ids):
                by_len[int(len(x))].append(i)
            d = self.hidden
            for T, idxs in by_len.items():
                # compile for the EXACT length T (the baked causal mask is per-T); only the batch
                # dim B is snapped to the grid (pad rows are zeros and unread).
                for s in range(0, len(idxs), B_GRID[-1]):
                    chunk = idxs[s:s + B_GRID[-1]]
                    Bg = _snap(len(chunk), B_GRID)
                    if Bg is None:
                        raise RuntimeError("chunk past B grid")
                    ml = self._get_model(Bg, T)
                    h0 = torch.zeros(Bg, T, d)
                    for r, i in enumerate(chunk):
                        h0[r] = self.store.embed_rows("embed", ids[i]).float()
                    pred = ml.predict(
                        {"h": h0.numpy().astype(np.float16 if _fp16_io() else np.float32)}
                    )
                    arr = pred[list(pred)[0]]                       # [Bg, T, V]
                    for r, i in enumerate(chunk):
                        out[i] = torch.from_numpy(np.ascontiguousarray(arr[r]))  # [T, V]
            # A non-default compute unit must pass an argmax check against paged before its
            # results are handed back even once (see _self_check_nondefault_units).
            if not self._units_checked:
                self._units_checked = True
                if str(self._compute_units()).rsplit(".", 1)[-1] != "ALL":
                    if not self._self_check_nondefault_units(ids_list, out):
                        self._last_fallback = True
                        self._last_execution_path = "paged-parity-fallback"
                        return super().logits_batch(ids_list)
            self._last_execution_path = "coreml-logits-batch"
            return out  # type: ignore[return-value]
        except Exception as e:                       # any Core ML failure -> correctness via paged
            if os.environ.get("MODEL_EXPERIMENTS_ANE_DEBUG"):
                print(f"[CoreML] fallback ({type(e).__name__}: {e})", file=sys.stderr)
            self._last_fallback = True
            self._last_execution_path = f"paged-error-fallback:{type(e).__name__}"
            return super().logits_batch(ids_list)

    def execution_evidence(self) -> dict[str, Any]:
        """Return claim-safe Core ML execution metadata.

        The current Python path has no MLComputePlan inspection, so placement remains
        unverified even after successful prediction.
        """

        return {
            "requested_backend_alias": self._requested_backend_alias,
            "backend": self.backend,
            "reported_fabric": self.reported_fabric,
            "compute_units_request": str(
                self._compute_units() if getattr(self, "_ct", None) else "unavailable"
            ),
            "placement_verified": False,   # per-run MLComputePlan check is not wired in yet;
            #                                measured offline: see bench_ane_placement.py
            "preferred_devices": None,
            "compiled_shapes": [list(shape) for shape in sorted(self._compiled_shapes)],
            # getattr-with-default: evidence is a reporting surface and must never raise on a
            # partially-constructed engine (callers build bare instances to inspect claims).
            "disk_cache_hits": getattr(self, "_disk_cache_hits", 0),
            "disk_cache_misses": getattr(self, "_disk_cache_misses", 0),
            "last_fallback_to_paged": self._last_fallback,
            "last_execution_path": getattr(self, "_last_execution_path", "not-run"),
            "coreml_accelerated_methods": [
                "logits",
                "logits_batch",
                "last_logits_batch",
                "selected_last_logits_batch",
                "forced_choice_batched",
            ],
            "paged_fallback_methods": [
                "generate",
                "forward_acts",
                "forward_patched",
                "hidden_states",
                "attention_and_residual_taps",
            ],
            "accelerator_unavailable_reason": getattr(self, "_unavailable_reason", None),
        }

    def runtime_stats(self) -> dict[str, Any]:
        return self.execution_evidence()
