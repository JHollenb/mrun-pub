"""Row-delta / byte-identity leg — the companion metric to the eigenbasis lineup.

Specified by ``experiments/2026-07-26-distillation-fingerprints/fp_pairs.py`` and NEVER RUN
there (it needed both full matrices resident and that MacBook was RAM-critical), so it is
implemented here streamed: contiguous row blocks pulled through
:class:`mrun.fingerprint.reader.TensorSource`, working set O(chunk x d) per arm, with an
optional deterministic row SAMPLE that also cuts the bytes moved on the HTTP path.

What it adds over the spectrum legs: the spectrum and the eigenbasis are invariant to things a
provenance question cares about (a permutation of vocabulary rows leaves the row-space spectrum
alone), and neither can say "these two checkpoints are the SAME BYTES". Row deltas answer
per-token "which rows were touched, and by how much", and the exact-identity check answers
"was the embedding copied verbatim" — the measured students are NOT verbatim copies (n50 shifts
a few dims, cos^2 is 0.96-0.9998, not 1.0), so a genuine 1.0 here means something different
from a high alignment.

HONEST STANDING: the eigenbasis leg (:func:`mrun.fingerprint.identify_parent`) is the one with
PROVENANCE validation behind it — 4 of 5 known-ground-truth substrate parents recovered out of
near-identical sibling lineups. This leg has no such panel: it has never been asked to rank
candidate parents, and unlike alignment it has no random-subspace null, so its numbers are
DESCRIPTIVE. Treat "these two differ by relF 0.03" as a measurement, not as an identification.

First real run, MEASURED 2026-07-26 (mmap'd local checkpoints, peak RSS 0.51 GB for a
128256 x 4096 pair — this is the leg fp_pairs.py could not run at all):

| pair | relF | median row | p90 | exact rows | identical |
|---|---|---|---|---|---|
| Qwen2.5-1.5B vs itself (control) | 0 | 0 | 0 | 1.0000 | True |
| Qwen2.5-0.5B-Instruct vs 0.5B | 0.1114 | 0.1065 | 0.1395 | 0 | False |
| Qwen2.5-1.5B-Instruct vs 1.5B | 0.0273 | 0.0169 | 0.0341 | 0 | False |
| Llama-3.1-8B-Instruct vs 8B | 0.0400 | 0.0400 | 0.0524 | 0 | False |

Worth recording because it is exactly the complementarity claim, MEASURED rather than argued:
instruct-tuning is a ~0-dim event in the SPECTRUM (dn50 0/+1/-1, dn90 0 in the published ladder)
yet it moves EVERY row by 1.7-11% and leaves not one row byte-identical. The perturbation is
approximately subspace-preserving, which the spectral legs are blind to by construction. n=3
instruct pairs plus one identity control; no distillation pair has been run through this leg yet.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from . import reader as _reader
from .lineup import MatchedObjectError

# A right-arm row whose norm is below this fraction of the median row norm carries no magnitude
# to be relative to. Measured separation is ~20 orders of magnitude (Llama-3.1-8B: 2e-21 vs
# median 0.688), so the cut is nowhere near any real row.
DEGENERATE_NORM_FRAC = 1e-6


@dataclass(frozen=True)
class RowDelta:
    """Per-row difference summary between two checkpoints' embeddings. JSON-able via
    :meth:`as_dict`. ``byte_identical`` is ``None`` (not ``False``) under sampling, because a
    sample cannot prove identity."""
    left: str
    right: str
    key_left: str
    key_right: str
    d: int
    V_common: int
    n_rows: int
    sampled: bool
    dtype_left: str
    dtype_right: str
    relF: float
    row_reld_median: float
    row_reld_p90: float
    row_reld_max: float
    frac_rows_lt_1e_3: float
    frac_rows_lt_1e_2: float
    exact_row_frac: float
    byte_identical: bool | None
    vocab_mismatch_frac: float | None
    vocab_gate_n: int
    degenerate_norm_rows: int = 0
    median_row_norm: float = 0.0
    n_rows_in_stats: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "type": "row_delta", "left": self.left, "right": self.right,
            "key_left": self.key_left, "key_right": self.key_right, "d": self.d,
            "V_common": self.V_common, "n_rows": self.n_rows, "sampled": self.sampled,
            "dtype_left": self.dtype_left, "dtype_right": self.dtype_right,
            "relF": round(self.relF, 8), "row_reld_median": round(self.row_reld_median, 8),
            "row_reld_p90": round(self.row_reld_p90, 8),
            "row_reld_max": round(self.row_reld_max, 8),
            "frac_rows_lt_1e-3": round(self.frac_rows_lt_1e_3, 6),
            "frac_rows_lt_1e-2": round(self.frac_rows_lt_1e_2, 6),
            "exact_row_frac": round(self.exact_row_frac, 6),
            "degenerate_norm_rows": self.degenerate_norm_rows,
            "median_row_norm": round(self.median_row_norm, 8),
            "n_rows_in_stats": self.n_rows_in_stats,
            "byte_identical": self.byte_identical,
            "vocab_mismatch_frac": self.vocab_mismatch_frac,
            "vocab_gate_n": self.vocab_gate_n, "note": self.note,
            "validated": False,
            "teacher_attribution": "OUT-OF-SCOPE",
        }


def _blocks(indices: np.ndarray, chunk: int):
    """Group sorted row indices into chunk-aligned contiguous read blocks."""
    for blk in np.unique(indices // chunk):
        lo = int(blk) * chunk
        sel = indices[(indices >= lo) & (indices < lo + chunk)]
        yield lo, int(sel.max()) + 1, sel - lo


def row_delta(left, right, *, key: str | None = None, key_right: str | None = None,
              sample: int | None = None, chunk: int = 8192, seed: int = 0, trim: bool = True,
              vocab_gate: int = 2000, prefer_local: bool = True,
              revision: str = "main") -> RowDelta:
    """Streamed per-row deltas + exact-identity check between two checkpoints' embeddings.

    ``left``/``right`` are anything :func:`mrun.fingerprint.reader.resolve_source` accepts
    (local path, registry name, HF repo id, a source, or an in-memory tensor via
    :class:`~mrun.fingerprint.reader.ArraySource`). Rows are trimmed to
    ``min(V_tok_left, V_tok_right)`` by default and the same matched-object rule applies as for
    the spectral legs: differing ``d`` or differing ``V_tok`` RAISES
    :class:`~mrun.fingerprint.lineup.MatchedObjectError`.

    ``sample=N`` compares a deterministic N-row sample (seeded), reading only the blocks those
    rows fall in — cheap enough to run against a remote checkpoint. Sampling forfeits the
    identity claim: ``byte_identical`` becomes ``None``.

    Metrics (all per-row relative to the RIGHT arm's row norm, as ``fp_pairs.py`` specified):
    ``relF`` (Frobenius relative delta), median / p90 / max of the per-row relative delta,
    fraction of rows below 1e-3 and 1e-2, the exactly-equal-row fraction, and a sampled
    vocab-identity gate (equal ``V_tok`` does not prove equal token strings).
    """
    src_a = _reader.resolve_source(left, prefer_local=prefer_local, revision=revision)
    src_b = _reader.resolve_source(right, prefer_local=prefer_local, revision=revision)
    ka = _reader.find_embedding_key(src_a, key)
    kb = _reader.find_embedding_key(src_b, key_right if key_right is not None else key)
    ma, mb = src_a.meta(ka), src_b.meta(kb)
    if int(ma.shape[1]) != int(mb.shape[1]):
        raise MatchedObjectError(
            f"{src_a.label} vs {src_b.label}: d {ma.shape[1]} vs {mb.shape[1]} — different "
            "objects, no row comparison is defined")
    d = int(ma.shape[1])

    v_a, _ = _reader.vocab_info(src_a)
    v_b, _ = _reader.vocab_info(src_b)
    notes: list[str] = []
    if trim and v_a is not None and v_b is not None:
        if int(v_a) != int(v_b):
            raise MatchedObjectError(
                f"{src_a.label} vs {src_b.label}: len(tokenizer) {v_a} vs {v_b} — different "
                "vocabularies, no row-wise comparison is defined")
        v_common = min(int(v_a), ma.rows, mb.rows)
        trim_note = "tok"
    else:
        v_common = min(ma.rows, mb.rows)
        trim_note = "cfg"
        if trim:
            notes.append("no tokenizer.json on at least one arm: rows trimmed to "
                         "min(V_cfg) (padded rows included)")
    if v_common < 1:
        raise ValueError("no rows in common")

    if sample is None:
        indices = np.arange(v_common, dtype=np.int64)
        sampled = False
    else:
        n = min(int(sample), v_common)
        rng = random.Random(seed)
        indices = np.array(sorted(rng.sample(range(v_common), n)), dtype=np.int64)
        sampled = True

    delta = np.empty(len(indices), dtype=np.float64)     # per-row ||xa - xb||
    norm = np.empty(len(indices), dtype=np.float64)       # per-row ||xb||
    exact = 0
    compared = 0
    for lo, hi, offsets in _blocks(indices, max(1, int(chunk))):
        xa_raw = src_a.read_rows(ka, lo, hi)[offsets]
        xb_raw = src_b.read_rows(kb, lo, hi)[offsets]
        if xa_raw.dtype == xb_raw.dtype:
            exact += int((xa_raw == xb_raw).all(dim=1).sum())
        xa = xa_raw.to(torch.float32)
        xb = xb_raw.to(torch.float32)
        n_here = len(offsets)
        delta[compared:compared + n_here] = (xa - xb).norm(dim=1).to(torch.float64).numpy()
        norm[compared:compared + n_here] = xb.norm(dim=1).to(torch.float64).numpy()
        compared += n_here
    delta, norm = delta[:compared], norm[:compared]

    # DEGENERATE-NORM ROWS. Reserved/never-trained vocabulary rows are not zero but are
    # astronomically small: MEASURED 2026-07-26 on Meta-Llama-3.1-8B, 289 of 128256 rows have
    # norm ~2e-21 against a median of 0.688, and one of them moved by 0.005 in the Instruct
    # checkpoint -- a relative delta of 2.5e18 that owned row_reld_max outright (fp_pairs.py's
    # clamp(1e-30) would have reported exactly that). "Relative to a row that carries no
    # magnitude" is undefined, not enormous, so those rows are counted, excluded from the
    # relative-delta statistics, and kept in relF where the ratio-of-sums stays well defined.
    # The cut is DATA-DRIVEN (a fraction of the median row norm), not an absolute constant.
    median_norm = float(np.median(norm)) if compared else 0.0
    floor = DEGENERATE_NORM_FRAC * median_norm
    good = norm > floor
    degenerate = int((~good).sum())
    reld = delta[good] / norm[good]
    written = int(good.sum())
    num = float((delta ** 2).sum())
    den = float((norm ** 2).sum())

    same_dtype = ma.dtype == mb.dtype
    same_shape = tuple(ma.shape) == tuple(mb.shape)
    if degenerate:
        notes.append(f"{degenerate} right-arm rows have degenerate norm (< {floor:.3g}, i.e. "
                     f"{DEGENERATE_NORM_FRAC:g} x the median {median_norm:.4g} — reserved/"
                     "never-trained vocabulary): excluded from the relative-delta statistics, "
                     "kept in relF")
    if sampled:
        byte_identical: bool | None = None
        notes.append(f"sampled {compared}/{v_common} rows — identity not provable from a sample")
    else:
        byte_identical = bool(same_dtype and same_shape and exact == compared)
        if not same_dtype:
            notes.append(f"dtype differs ({ma.dtype} vs {mb.dtype}): exact-row count skipped, "
                         "identity impossible by construction")
        elif not same_shape and exact == compared:
            notes.append("all COMMON rows identical but V_cfg differs (padding-only difference)")

    mismatch: float | None = None
    gate_n = 0
    if vocab_gate > 0:
        toks_a = _reader.token_strings(src_a)
        toks_b = _reader.token_strings(src_b)
        if toks_a is not None and toks_b is not None:
            rng = random.Random(seed + 1)
            gate_ids = [rng.randrange(v_common) for _ in range(int(vocab_gate))]
            gate_n = len(gate_ids)
            mismatch = sum(1 for i in gate_ids if toks_a.get(i) != toks_b.get(i)) / gate_n
            if mismatch:
                notes.append(f"vocab identity gate: {mismatch:.1%} of {gate_n} sampled ids map "
                             "to different token strings")

    return RowDelta(
        left=src_a.label, right=src_b.label, key_left=ka, key_right=kb, d=d, V_common=v_common,
        n_rows=compared, sampled=sampled, dtype_left=ma.dtype, dtype_right=mb.dtype,
        relF=float((num / den) ** 0.5) if den > 0 else float("nan"),
        row_reld_median=float(np.median(reld)) if written else float("nan"),
        row_reld_p90=float(np.quantile(reld, 0.9)) if written else float("nan"),
        row_reld_max=float(reld.max()) if written else float("nan"),
        frac_rows_lt_1e_3=float((reld < 1e-3).mean()) if written else float("nan"),
        frac_rows_lt_1e_2=float((reld < 1e-2).mean()) if written else float("nan"),
        exact_row_frac=(exact / compared) if compared and same_dtype else 0.0,
        byte_identical=byte_identical, vocab_mismatch_frac=mismatch, vocab_gate_n=gate_n,
        degenerate_norm_rows=degenerate, median_row_norm=median_norm, n_rows_in_stats=written,
        note="; ".join([f"trim={trim_note}", *notes]))
