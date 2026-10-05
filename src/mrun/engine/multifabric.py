"""Multi-fabric split-batch engine: overlap CPU paged + MLX GPU on one batch.

The default backends live on distinct execution resources (CPU and Metal GPU), and each hot
loop releases the Python GIL (numpy BLAS and ``mx.eval``). A batch of forward rows can therefore
be split across them and run concurrently in threads: wall time approaches ``max`` over shards
instead of the single fastest fabric, and aggregate prefill throughput can approach the sum of
their standalone throughput.

Core ML/``ane`` is deliberately excluded from the default. MLComputePlan measured the current
model placing about 98% of operations on the GPU under ``ComputeUnit.ALL``, so running it beside
MLX double-books the same Metal fabric rather than adding Neural Engine capacity. Callers may
still include ``ane`` explicitly for placement research, but it must not be counted as a
physically independent shard until per-run placement evidence proves that claim.

This is an **opt-in speed path**, not the canonical-measurement path: accelerator shards are
fp16, so gate any content/CE leg on real-text parity vs ``paged``. ``paged`` is always the
reference; rows it scores are exact int8-stream fp32.

Rows are assigned to fabrics by a throughput weight (so the fast fabric gets the big shard and
all finish together), then reassembled in the original order. ``MultiFabricEngine`` subclasses
``BaseEngine``, so ``score_forced_choice_many`` (and its batched fast path) work unchanged — they
call ``logits_batch``, which is the split here.
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import Any

import numpy as np
import torch

from ._base_impl import BaseEngine

# Standalone batched-knee tok/s measured on Qwen2.5-0.5B (T=32), used as default split weights so
# each fabric gets a shard sized to finish at ~the same wall time. Override via `weights=`.
_DEFAULT_TPUT: dict[str, float] = {"mlx": 4208.0, "ane": 3412.0, "paged": 826.0, "hf": 200.0}


def shard_sizes(n: int, weights: list[float]) -> list[int]:
    """Split ``n`` rows into per-fabric shard sizes proportional to ``weights`` (largest-remainder
    rounding, so the sizes sum to exactly ``n``)."""
    total = float(sum(weights)) or 1.0
    raw = [n * (w / total) for w in weights]
    sizes = [int(x) for x in raw]
    rem = n - sum(sizes)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - sizes[i], reverse=True)
    for i in range(rem):
        sizes[order[i % len(order)]] += 1
    return sizes


class MultiFabricEngine(BaseEngine):
    backend = "multifabric"
    supports_batch = True

    def __init__(
        self,
        model_name: str,
        *,
        backends: tuple[str, ...] = ("paged", "mlx"),
        weights: dict[str, float] | None = None,
        device: str = "cpu",
        **open_kwargs: Any,
    ) -> None:
        from . import open_engine  # lazy: open_engine imports backend submodules

        self.model = model_name
        self.engines: list[BaseEngine] = []
        self.backend_names: list[str] = []
        self.open_errors: dict[str, str] = {}
        for b in backends:
            try:
                # The composite owns its children. A fresh child avoids sharing a pooled engine
                # whose lifetime could outlive (or be closed before) this composite.
                eng = open_engine(
                    model_name,
                    backend=b,
                    fresh=True,
                    device=device,
                    **open_kwargs,
                )
            except Exception as exc:  # a fabric that can't open is dropped, not fatal
                self.open_errors[b] = f"{type(exc).__name__}: {exc}"
                continue
            # `ane` can degrade to paged when Core ML is unavailable. Dedupe the resolved
            # backend so the same execution path is never counted as two fabrics.
            actual = getattr(eng, "backend", b)
            if actual in self.backend_names:
                eng.close()
                self.open_errors[b] = f"resolved to already-open backend {actual!r}"
                continue
            self.engines.append(eng)
            self.backend_names.append(actual)
        if not self.engines:
            raise RuntimeError(f"no fabric opened for {model_name}: {self.open_errors}")

        weight_map = {**_DEFAULT_TPUT, **(weights or {})}
        self._weights = [float(weight_map.get(n, 1.0)) for n in self.backend_names]
        # paged is first by default and remains the exact reference
        primary = self.engines[0]
        self.tokenizer = primary.tokenizer
        self.name = primary.name
        self.n_layer = int(primary.n_layer)
        self.inter = int(primary.inter)
        self.hidden = int(primary.hidden)
        self.cfg = getattr(primary, "cfg", None)
        self._primary = primary

    # -- single-row forward delegates to the reference (paged) fabric -------------
    def logits(self, ids: np.ndarray) -> torch.Tensor:
        return self._primary.logits(ids)

    def forward_acts(self, ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self._primary.forward_acts(ids)

    def down_weight(self, layer: int) -> torch.Tensor:
        return self._primary.down_weight(layer)

    # -- the split: assign shards by throughput weight, run fabrics concurrently --
    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        n = len(ids_list)
        if n == 0:
            return []
        if len(self.engines) == 1:
            return self.engines[0].logits_batch(ids_list)
        sizes = shard_sizes(n, self._weights)
        out: list[torch.Tensor | None] = [None] * n
        shards: list[tuple[BaseEngine, int, list[np.ndarray]]] = []
        pos = 0
        for eng, sz in zip(self.engines, sizes, strict=True):
            if sz <= 0:
                continue
            shards.append((eng, pos, ids_list[pos : pos + sz]))
            pos += sz

        def _work(engine: BaseEngine, start: int, sub: list[np.ndarray]) -> None:
            rows = engine.logits_batch(sub)
            for j, row in enumerate(rows):
                out[start + j] = row

        # separate engine instances + GIL-releasing inner loops ⇒ true fabric overlap
        with cf.ThreadPoolExecutor(max_workers=len(shards)) as ex:
            for fut in [ex.submit(_work, e, s, sub) for e, s, sub in shards]:
                fut.result()
        return out  # type: ignore[return-value]

    @property
    def working_set_mb(self) -> float | None:
        vals = [e.working_set_mb for e in self.engines if e.working_set_mb is not None]
        return float(sum(vals)) if vals else None

    @working_set_mb.setter
    def working_set_mb(self, _value: Any) -> None:
        return None

    def close(self) -> None:
        for eng in self.engines:
            try:
                eng.close()
            except Exception:
                pass


def open_multifabric_engine(model_name: str, **kwargs: Any) -> MultiFabricEngine:
    return MultiFabricEngine(model_name, **kwargs)
