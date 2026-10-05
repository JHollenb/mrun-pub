"""The fingerprint itself — centered-embedding Gram spectrum + cached top-k eigenbasis.

One object per checkpoint, computed from ONE tensor:

  * spectrum      n50 / n90 / n95 / participation-ratio of the centered-row Gram, i.e. how many
                  directions the input code spends its variance on (the
                  embedding-concentration-regime measure, same code path as those scripts);
  * eigenbasis    the top-k eigenvectors, ~4 MB per model. This is the artifact that makes
                  re-analysis nearly free: every pairwise question (spectral delta, principal
                  angles, a whole lineup) is answered from cached ``[d, k]`` matrices in
                  milliseconds, with no checkpoint and no network.

Matched-object discipline is baked into the row, not the prose: ``V_cfg`` (tensor rows, padded)
and ``V_tok`` (``len(tokenizer)``, the real vocabulary) are both recorded, along with which one
the spectrum was computed over (``trim``), the checkpoint ``dtype``, and a ``vocab_sig`` — so a
comparison can be refused instead of silently reporting the difference between two different
objects (see :func:`mrun.fingerprint.lineup.assert_matched`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from . import reader as _reader

SCOPE = (
    "SUBSTRATE lineage only, weight-space only. Measured 2026-07-26 "
    "(experiments/2026-07-26-distillation-fingerprints): distillation moves the input "
    "embedding by 0-5 spectral dims and leaves n90 unchanged in 3 of 5 cases, never drifting "
    "toward the teacher's signature -- so TEACHER ATTRIBUTION IS OUT OF SCOPE. Instruct-tuning "
    "is likewise a ~0-dim event, so 'distilled from X' and 'instruction-tuned from X' are NOT "
    "separable, and neither is base-vs-instruct as the substrate. Says where the weights came "
    "from; says nothing about behaviour."
)


# ------------------------------------------------------------------ core math
def centered_gram(source: _reader.TensorSource, key: str, *, rows: int, chunk: int = 16384,
                  center: str = "auto") -> tuple[np.ndarray, int]:
    """Centered-row Gram matrix ``G = sum_i (x_i - mean)(x_i - mean)^T`` over the first ``rows``
    rows of ``key``, accumulated in float64 from streamed row blocks (working set O(chunk x d)).

    ``center='two-pass'`` computes the mean first and subtracts it per block — the arrangement
    the measured scripts used. ``center='one-pass'`` streams the rows once, halving the bytes
    moved (it matters only for the HTTP path, which ``center='auto'`` therefore selects it for).
    The one-pass arm shifts by the FIRST BLOCK's mean before accumulating and applies the exact
    algebraic correction afterwards: a naive ``sum x x^T - outer(sum x, sum x)/N`` would subtract
    two large nearly-equal matrices, and embedding rows have a mean comparable to their spread,
    so the cancellation is real. Equivalence of the two arms is a unit test, not a claim.
    """
    if center == "auto":
        center = "one-pass" if source.kind == "http-range" else "two-pass"
    if center not in ("one-pass", "two-pass"):
        raise ValueError(f"center must be one-pass|two-pass|auto, got {center!r}")
    meta = source.meta(key)
    rows = min(int(rows), meta.rows)
    if rows < 2:
        raise ValueError(f"need >=2 rows to center a Gram, got {rows}")
    d = int(meta.shape[1])

    if center == "one-pass":
        gram = np.zeros((d, d), dtype=np.float64)
        acc = np.zeros(d, dtype=np.float64)
        shift: torch.Tensor | None = None
        for _, block in _reader.iter_rows(source, key, stop=rows, chunk=chunk):
            x = block.to(torch.float32)
            if shift is None:
                shift = x.mean(0)
            y = x - shift
            gram += (y.T @ y).to(torch.float64).numpy()
            acc += y.sum(0).to(torch.float64).numpy()
        gram -= np.outer(acc, acc) / rows
        return gram, rows

    acc = np.zeros(d, dtype=np.float64)
    for _, block in _reader.iter_rows(source, key, stop=rows, chunk=chunk):
        acc += block.to(torch.float32).sum(0).to(torch.float64).numpy()
    mean = torch.from_numpy(acc / rows).to(torch.float32)
    gram = np.zeros((d, d), dtype=np.float64)
    for _, block in _reader.iter_rows(source, key, stop=rows, chunk=chunk):
        x = block.to(torch.float32) - mean
        gram += (x.T @ x).to(torch.float64).numpy()
    return gram, rows


def eig_descending(gram: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(evals[d] descending, basis[d, k])`` of a symmetric Gram. Negative eigenvalues (float
    rounding on a PSD matrix / one-pass cancellation) are clamped to 0, as in the source scripts.
    """
    evals, evecs = np.linalg.eigh(gram)
    order = evals.argsort()[::-1]
    evals = np.clip(evals[order], 0.0, None)
    basis = np.ascontiguousarray(evecs[:, order][:, :max(1, int(k))], dtype=np.float32)
    return evals, basis


def spectral_counts(evals: np.ndarray) -> dict:
    """n50 / n90 / n95 / participation ratio from a descending eigenvalue spectrum."""
    total = float(evals.sum())
    if total <= 0:
        raise ValueError("degenerate spectrum (zero total variance)")
    cum = np.cumsum(evals) / total
    return {
        "n50": int((cum < 0.5).sum() + 1),
        "n90": int((cum < 0.9).sum() + 1),
        "n95": int((cum < 0.95).sum() + 1),
        "pr": float(total ** 2 / float((evals.astype(np.float64) ** 2).sum())),
        "total_var": total,
    }


# ------------------------------------------------------------------ the object
@dataclass(frozen=True, eq=False)
class EmbeddingFingerprint:
    """One checkpoint's embedding fingerprint. Immutable; JSON-able via :meth:`row`.

    ``basis`` is ``[d, k]`` float32 (top-k eigenvectors of the centered Gram, columns ordered by
    eigenvalue). ``evals`` is the descending spectrum — full length ``d`` when computed here,
    possibly truncated to ``k`` when adopted from a legacy cache, in which case ``n50``/``n90``/
    ``n95``/``pr`` are whatever was recorded alongside it (``None`` if nothing was).

    SCOPE: see :data:`SCOPE` — substrate lineage, not teacher lineage.
    """
    model: str
    key: str
    dtype: str
    source: str
    d: int
    V_cfg: int
    V_tok: int
    trim: str                      # "tok" (trimmed to len(tokenizer)) | "cfg" (untrimmed)
    basis: np.ndarray              # [d, k] float32
    evals: np.ndarray              # descending
    n50: int | None = None
    n90: int | None = None
    n95: int | None = None
    pr: float | None = None
    total_var: float | None = None
    vocab_sig: str | None = None
    rows_used: int | None = None
    center: str | None = None
    scope: str = SCOPE

    # -- derived
    @property
    def k(self) -> int:
        return int(self.basis.shape[1])

    @property
    def trimmed(self) -> bool:
        """True when the spectrum was computed over ``len(tokenizer)`` rows, not the padded
        ``vocab_size``. Comparing a trimmed to an untrimmed fingerprint is refused."""
        return self.trim == "tok"

    @property
    def object_signature(self) -> tuple[int, int, str]:
        """The tuple two fingerprints must agree on to be the same measured object."""
        return (int(self.d), int(self.V_tok), str(self.trim))

    def _ratio(self, value: int | float | None) -> float | None:
        return None if value is None else round(float(value) / self.d, 4)

    @property
    def n50_d(self) -> float | None:
        return self._ratio(self.n50)

    @property
    def n90_d(self) -> float | None:
        return self._ratio(self.n90)

    @property
    def n95_d(self) -> float | None:
        return self._ratio(self.n95)

    @property
    def pr_d(self) -> float | None:
        return self._ratio(self.pr)

    def subspace(self, k: int) -> np.ndarray:
        """Top-``k`` eigenvectors. Raises when the cache holds fewer than ``k`` — an alignment
        computed against a silently-shortened basis is not the metric it claims to be."""
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > self.k:
            raise ValueError(f"{self.model}: cached basis has k={self.k}, {k} requested "
                             "(recompute the fingerprint with a larger k)")
        if k > self.d:
            raise ValueError(f"k={k} exceeds d={self.d}")
        return self.basis[:, :k]

    def row(self) -> dict:
        """Flat JSON-able record. Always carries dtype + V_tok/V_cfg + trim (matched-object
        audit fields) and the out-of-scope note."""
        return {
            "type": "fingerprint", "model": self.model, "key": self.key, "dtype": self.dtype,
            "source": self.source, "d": self.d, "V_cfg": self.V_cfg, "V_tok": self.V_tok,
            "trim": self.trim, "trimmed": self.trimmed, "rows_used": self.rows_used,
            "vocab_sig": self.vocab_sig, "k_cached": self.k, "center": self.center,
            "n50": self.n50, "n90": self.n90, "n95": self.n95, "pr": self.pr,
            "n50_d": self.n50_d, "n90_d": self.n90_d, "n95_d": self.n95_d, "pr_d": self.pr_d,
            "ratio_n50_n90": (None if not (self.n50 and self.n90)
                              else round(self.n50 / self.n90, 4)),
            "teacher_attribution": "OUT-OF-SCOPE",
        }

    # -- cache
    def save(self, path: str | Path) -> Path:
        """Write ``basis`` + ``evals`` + the metadata row to an ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in self.row().items() if k != "type"}
        meta["total_var"] = self.total_var      # not in row() (not a reported metric), but the
                                                # spectrum is not reconstructable without it
        np.savez(path, basis=self.basis, evals=np.asarray(self.evals, dtype=np.float64),
                 meta=np.array(json.dumps(meta)))
        return path

    @classmethod
    def load(cls, path: str | Path, *, meta: dict | None = None) -> EmbeddingFingerprint:
        """Read a cached fingerprint.

        Caches written by this module carry their own metadata. A LEGACY cache (e.g. the
        ``experiments/2026-07-26-distillation-fingerprints/cache/*.npz`` files, which hold only
        ``evecs``/``evals``) has no ``d``/``V_tok``/``trim`` inside it, so the matched-object
        guard would have nothing to check — this refuses to invent them and requires an explicit
        ``meta=`` (the corresponding ``fingerprints.jsonl`` row).
        """
        with np.load(Path(path), allow_pickle=False) as z:
            keys = set(z.files)
            basis = z["basis"] if "basis" in keys else z["evecs"]   # 'evecs' = legacy name
            evals = z["evals"] if "evals" in keys else np.zeros(basis.shape[1])
            stored = json.loads(str(z["meta"].item())) if "meta" in keys else None
        merged = dict(stored or {})
        merged.update(meta or {})
        missing = [f for f in ("model", "d", "V_cfg", "V_tok", "trim", "dtype") if f not in merged]
        if missing:
            raise ValueError(
                f"{path}: cache carries no {missing} — matched-object fields cannot be inferred "
                "from arrays. Pass meta={...} (e.g. the fingerprints.jsonl row for this model).")
        return from_arrays(basis, evals=evals, **{
            f: merged.get(f) for f in
            ("model", "key", "dtype", "source", "d", "V_cfg", "V_tok", "trim", "n50", "n90",
             "n95", "pr", "total_var", "vocab_sig", "rows_used", "center")})


def from_arrays(basis: np.ndarray, *, model: str, d: int, V_cfg: int, V_tok: int,
                trim: str = "tok", dtype: str = "?", source: str = "arrays",
                key: str = "?", evals: np.ndarray | None = None, n50: int | None = None,
                n90: int | None = None, n95: int | None = None, pr: float | None = None,
                total_var: float | None = None, vocab_sig: str | None = None,
                rows_used: int | None = None, center: str | None = None) -> EmbeddingFingerprint:
    """Build a fingerprint from an eigenbasis you already have.

    The adoption path for legacy caches and for cross-repo reuse: the lineup metric needs only
    ``basis`` plus the matched-object fields (``d``, ``V_tok``, ``trim``), all of which are
    REQUIRED here precisely so the guard has something to enforce.
    """
    basis = np.ascontiguousarray(np.asarray(basis, dtype=np.float32))
    if basis.ndim != 2:
        raise ValueError(f"basis must be [d, k], got shape {basis.shape}")
    if int(basis.shape[0]) != int(d):
        raise ValueError(f"basis has {basis.shape[0]} rows but d={d}")
    if trim not in ("tok", "cfg"):
        raise ValueError(f"trim must be 'tok' or 'cfg', got {trim!r}")
    ev = np.zeros(basis.shape[1], dtype=np.float64) if evals is None else np.asarray(
        evals, dtype=np.float64)
    return EmbeddingFingerprint(
        model=str(model), key=str(key or "?"), dtype=str(dtype or "?"),
        source=str(source or "arrays"), d=int(d),
        V_cfg=int(V_cfg), V_tok=int(V_tok), trim=str(trim), basis=basis, evals=ev,
        n50=None if n50 is None else int(n50), n90=None if n90 is None else int(n90),
        n95=None if n95 is None else int(n95), pr=None if pr is None else float(pr),
        total_var=None if total_var is None else float(total_var), vocab_sig=vocab_sig,
        rows_used=None if rows_used is None else int(rows_used), center=center)


def truncate(fp: EmbeddingFingerprint, k: int) -> EmbeddingFingerprint:
    """A copy holding only the top-``k`` eigenvectors (for cheap serialization)."""
    return replace(fp, basis=np.ascontiguousarray(fp.subspace(k)))
