"""Pairwise comparison + the LINEUP — "which of these candidates is the substrate parent?"

Two measured facts fix the shape of this API and are worth stating before the code:

1. **Alignment is not a lineage tree, so there is no threshold.** Mean cos^2 of principal angles
   between top-k centered-embedding eigenbases separates a true substrate parent (0.96-0.9998)
   from wrong same-family siblings (0.19-0.67) *within one lineup*. But the numbers are not
   comparable across lineups: Qwen2.5-14B and 32B, neither derived from the other, sit at 0.64,
   while Qwen2.5-7B and its own Coder-7B descendant sit at 0.19. Heavy continued pretraining
   moves a model further from its parent than two independently trained siblings are from each
   other. Any absolute cutoff would therefore be wrong in both directions, so the entry point is
   :func:`identify_parent` -- a RANKING WITH MARGINS against candidates you supply -- and there
   is deliberately no ``is_derived_from``, no score threshold, and no way to coerce a
   :class:`ParentLineup` to a bool.

2. **Same object or no comparison.** Two embeddings are comparable only at equal ``d``, equal
   ``V_tok`` and equal trimming. :func:`assert_matched` RAISES (never warns) on a mismatch: an
   unmatched pair reliably produces a plausible-looking number, which is how this class of
   result gets retracted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spectrum import SCOPE, EmbeddingFingerprint

# The measured Llama-8B base-vs-instruct tie is a 0.0004 gap (0.9942 vs 0.9946, inverted at
# k=256); every measured true-parent win is a >=0.35 gap. 0.02 sits between them by two orders
# of magnitude in both directions. NOTE what this is: a floor on the GAP BETWEEN THE TOP TWO
# candidates, i.e. "is this lineup resolved?" It is never a cutoff on a candidate's score.
DEFAULT_AMBIGUITY_MARGIN = 0.02


class MatchedObjectError(ValueError):
    """Two fingerprints do not measure the same object, so no comparison is defined."""


def assert_matched(a: EmbeddingFingerprint, b: EmbeddingFingerprint, *,
                   strict_dtype: bool = False, strict_vocab_sig: bool = False) -> dict:
    """Enforce the lab's matched-object rule. Raises :class:`MatchedObjectError` on differing
    ``d``, differing ``V_tok``, or differing trimming; returns the audit notes otherwise.

    dtype and ``vocab_sig`` differences are RECORDED rather than fatal by default (the measured
    panel is uniformly bf16, and equal-``V_tok`` tokenizers can still differ in added-token
    strings) -- pass ``strict_dtype`` / ``strict_vocab_sig`` to promote them to errors.
    """
    problems = []
    if int(a.d) != int(b.d):
        problems.append(f"d {a.d} vs {b.d}")
    if int(a.V_tok) != int(b.V_tok):
        problems.append(f"V_tok {a.V_tok} vs {b.V_tok}")
    if a.trim != b.trim:
        problems.append(f"trim {a.trim!r} vs {b.trim!r}")
    if problems:
        raise MatchedObjectError(
            f"{a.model} vs {b.model}: not the same measured object ({'; '.join(problems)}). "
            "Embedding-fingerprint comparison requires equal d, equal len(tokenizer) and "
            "identical trimming; cross-tokenizer lineage is out of reach by construction.")
    notes: dict = {"dtype_left": a.dtype, "dtype_right": b.dtype,
                   "dtype_match": a.dtype == b.dtype,
                   "vocab_sig_left": a.vocab_sig, "vocab_sig_right": b.vocab_sig}
    if a.vocab_sig and b.vocab_sig:
        notes["vocab_sig_match"] = a.vocab_sig == b.vocab_sig
    else:
        notes["vocab_sig_match"] = None
    if strict_dtype and not notes["dtype_match"]:
        raise MatchedObjectError(
            f"{a.model} vs {b.model}: dtype {a.dtype} vs {b.dtype} (strict_dtype)")
    if strict_vocab_sig and notes["vocab_sig_match"] is False:
        raise MatchedObjectError(
            f"{a.model} vs {b.model}: tokenizer sample signatures differ despite equal V_tok "
            f"({a.vocab_sig} vs {b.vocab_sig}) (strict_vocab_sig)")
    if not notes["dtype_match"]:
        notes["warning"] = ("dtype differs between arms; a dtype-driven difference has "
                            "masqueraded as a model difference before (see the "
                            "cross-model-comparison-artifacts rule)")
    return notes


# ------------------------------------------------------------------ metrics
def subspace_alignment(a: EmbeddingFingerprint, b: EmbeddingFingerprint, *, k: int = 64,
                       **guard_kw) -> dict:
    """Mean cos^2 of the principal angles between the two top-``k`` eigenbases.

    ``1.0`` = identical subspace. The null for two independent random k-subspaces of R^d is
    exactly ``k/d``, reported as ``null_mean_cos2`` with ``ratio_to_null`` beside it, because
    ``0.19`` means very different things at ``d=1536`` and ``d=5120``.

    This is the leg with measured validation behind it (4 of 5 known-ground-truth substrate
    parents recovered out of near-identical same-family lineups; the fifth is an inherent
    base-vs-instruct ambiguity, not noise).
    """
    notes = assert_matched(a, b, **guard_kw)
    ka, kb = a.subspace(k), b.subspace(k)
    overlap = ka.T.astype(np.float64) @ kb.astype(np.float64)
    mean_cos2 = float((overlap ** 2).sum() / k)
    null = k / float(a.d)
    return {"k": int(k), "mean_cos2": mean_cos2, "null_mean_cos2": null,
            "ratio_to_null": (mean_cos2 / null) if null > 0 else float("nan"),
            "d": int(a.d), "V_tok": int(a.V_tok), "trim": a.trim, **notes}


def spectral_delta(a: EmbeddingFingerprint, b: EmbeddingFingerprint, **guard_kw) -> dict:
    """Signed spectral deltas ``a - b`` (n50/n90/n95/PR), the calibration-ladder measure.

    Measured grain: instruct-tuning and distillation are 0-5 dim events at the input side while
    continued pretraining is a 40-84 dim event -- an order of magnitude apart, which is what
    makes the ladder readable. It does NOT separate distillation from instruct-tuning.
    """
    notes = assert_matched(a, b, **guard_kw)
    out: dict = {"left": a.model, "right": b.model, **notes}
    for name in ("n50", "n90", "n95"):
        av, bv = getattr(a, name), getattr(b, name)
        out[f"d{name}"] = None if (av is None or bv is None) else int(av) - int(bv)
        out[f"{name}_left"], out[f"{name}_right"] = av, bv
    if a.pr is not None and b.pr is not None:
        out["dPR"] = round(float(a.pr) - float(b.pr), 4)
    out["d"] = int(a.d)
    return out


def compare(a: EmbeddingFingerprint, b: EmbeddingFingerprint, *, k: int = 64,
            **guard_kw) -> dict:
    """Both pairwise legs at once: spectral deltas plus top-``k`` subspace alignment.

    A pair is a pair -- read the alignment number only against other candidates for the same
    student (:func:`identify_parent`), never against a remembered cutoff.
    """
    align = subspace_alignment(a, b, k=k, **guard_kw)
    out = {"type": "pair", "left": a.model, "right": b.model,
           "spectral": spectral_delta(a, b, **guard_kw), "alignment": align,
           "scope": SCOPE}
    return out


# ------------------------------------------------------------------ the lineup
@dataclass(frozen=True)
class LineupEntry:
    """One candidate's standing in a lineup. ``rank`` 1 = best-matching in THIS lineup."""
    rank: int
    candidate: str
    mean_cos2: float
    ratio_to_null: float
    dn50: int | None = None
    dn90: int | None = None
    dn95: int | None = None
    dtype_match: bool = True
    vocab_sig_match: bool | None = None

    def as_dict(self) -> dict:
        return {"rank": self.rank, "candidate": self.candidate,
                "mean_cos2": round(self.mean_cos2, 6),
                "ratio_to_null": round(self.ratio_to_null, 2),
                "dn50": self.dn50, "dn90": self.dn90, "dn95": self.dn95,
                "dtype_match": self.dtype_match, "vocab_sig_match": self.vocab_sig_match}


@dataclass(frozen=True)
class ParentLineup:
    """A ranked lineup of candidate substrate parents for one student. **Relative by design.**

    There is no boolean here and no threshold: ``bool(lineup)`` raises, because "is this model
    derived from that one" is not a question a single alignment number can answer (measured:
    an unrelated same-family pair scored 0.64 while a true parent->descendant pair scored 0.19).
    What the object gives you is an ordering, the gap to the runner-up, and the random-subspace
    null -- read those.

    SCOPE, in the object because it belongs in the object: ``scope`` says the reading is
    SUBSTRATE lineage; ``teacher_attribution`` is always ``"OUT-OF-SCOPE"``.
    """
    student: str
    k: int
    entries: tuple[LineupEntry, ...]
    null_mean_cos2: float
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN
    d: int = 0
    V_tok: int = 0
    trim: str = "tok"
    scope: str = SCOPE
    teacher_attribution: str = "OUT-OF-SCOPE"
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- ordering
    @property
    def best_match(self) -> LineupEntry:
        """Highest-ranked candidate IN THIS LINEUP. Not evidence of derivation on its own --
        it only ever means "closer than the other candidates offered"."""
        return self.entries[0]

    @property
    def runner_up(self) -> LineupEntry:
        return self.entries[1]

    @property
    def margin(self) -> float:
        """Gap in mean cos^2 between rank 1 and rank 2 -- the number that says whether the
        lineup is resolved."""
        return float(self.entries[0].mean_cos2 - self.entries[1].mean_cos2)

    @property
    def spread(self) -> float:
        return float(self.entries[0].mean_cos2 - self.entries[-1].mean_cos2)

    @property
    def resolved(self) -> bool:
        """True when rank 1 leads rank 2 by more than ``ambiguity_margin``. This is a statement
        about the SEPARATION of two candidates, not about either candidate's score."""
        return self.margin > self.ambiguity_margin

    @property
    def verdict(self) -> str:
        if self.spread <= self.ambiguity_margin:
            return (f"NO-SEPARATION: all {len(self.entries)} candidates within "
                    f"{self.ambiguity_margin} — the lineup cannot distinguish them")
        if not self.resolved:
            return (f"AMBIGUOUS: {self.entries[0].candidate} vs {self.entries[1].candidate} "
                    f"within {self.ambiguity_margin} (margin {self.margin:.4f}) — "
                    "identifies the family/scale, not which of these two")
        return (f"RANKED: {self.best_match.candidate} leads {self.runner_up.candidate} by "
                f"{self.margin:.4f} (best {self.best_match.mean_cos2:.4f} vs null "
                f"{self.null_mean_cos2:.4f})")

    # -- misuse guards
    def __bool__(self) -> bool:
        raise TypeError(
            "ParentLineup has no truth value: a lineup ranks candidates, it does not decide "
            "'derived / not derived'. Use .best_match, .margin, .resolved and .verdict — and "
            "remember the ranking is only meaningful relative to the candidates you supplied.")

    def __float__(self) -> float:
        raise TypeError(
            "ParentLineup is not a score. Alignment has no absolute scale across lineups "
            "(measured: unrelated 14B/32B = 0.64, true parent->Coder descendant = 0.19).")

    # -- output
    def as_dict(self) -> dict:
        return {"type": "lineup", "student": self.student, "k": self.k, "d": self.d,
                "V_tok": self.V_tok, "trim": self.trim,
                "null_mean_cos2": round(self.null_mean_cos2, 6),
                "ambiguity_margin": self.ambiguity_margin,
                "ranking": [e.as_dict() for e in self.entries],
                "best_match": self.best_match.candidate,
                "margin": round(self.margin, 6), "spread": round(self.spread, 6),
                "resolved": self.resolved, "verdict": self.verdict,
                "teacher_attribution": self.teacher_attribution,
                "notes": list(self.notes), "scope": self.scope}

    def format(self) -> str:
        head = (f"lineup: {self.student}  (k={self.k}, d={self.d}, V_tok={self.V_tok}, "
                f"trim={self.trim}, null={self.null_mean_cos2:.4f})")
        rows = [f"  {e.rank}. {e.candidate:<34} cos2={e.mean_cos2:.4f}  "
                f"x_null={e.ratio_to_null:7.1f}  dn50={e.dn50!s:>5} dn90={e.dn90!s:>5}"
                for e in self.entries]
        tail = [f"  => {self.verdict}", f"  scope: teacher_attribution={self.teacher_attribution}"]
        return "\n".join([head, *rows, *[f"  ! {n}" for n in self.notes], *tail])


def rank_candidates(student: EmbeddingFingerprint,
                    candidates: dict[str, EmbeddingFingerprint] | list[EmbeddingFingerprint],
                    *, k: int = 64, ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
                    **guard_kw) -> ParentLineup:
    """Rank already-computed fingerprints. See :func:`mrun.fingerprint.identify_parent` for the
    model-reference-taking entry point (this is the same logic without the I/O)."""
    if isinstance(candidates, dict):
        items = list(candidates.items())
    else:
        items = [(fp.model, fp) for fp in candidates]
    if len(items) < 2:
        raise ValueError(
            "a lineup needs at least 2 candidates. Alignment has no absolute scale — "
            "0.64 was measured between two UNRELATED same-family models and 0.19 between a "
            "model and its own descendant — so a single-candidate 'score' is uninterpretable. "
            "Supply the plausible siblings (base / instruct / math / coder / neighbouring "
            "scale) and read the ranking.")
    labels = [lab for lab, _ in items]
    if len(set(labels)) != len(labels):
        dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
        raise ValueError(f"duplicate candidate labels in lineup: {dupes}")
    if student.model in labels:
        raise ValueError(
            f"student {student.model!r} is also one of its own candidates; self-alignment is "
            "1.0 by construction and would top every lineup")

    notes: list[str] = []
    scored: list[tuple[float, float, str, dict, dict]] = []
    for label, fp in items:
        align = subspace_alignment(student, fp, k=k, **guard_kw)
        delta = spectral_delta(student, fp, **guard_kw)
        if align.get("dtype_match") is False:
            notes.append(f"{label}: dtype {align['dtype_right']} vs student "
                         f"{align['dtype_left']} — dtype-driven difference not excluded")
        if align.get("vocab_sig_match") is False:
            notes.append(f"{label}: tokenizer sample signature differs despite equal V_tok")
        scored.append((align["mean_cos2"], align["ratio_to_null"], label, align, delta))
    scored.sort(key=lambda t: -t[0])
    entries = tuple(
        LineupEntry(rank=i + 1, candidate=label, mean_cos2=cos2, ratio_to_null=ratio,
                    dn50=delta.get("dn50"), dn90=delta.get("dn90"), dn95=delta.get("dn95"),
                    dtype_match=bool(align.get("dtype_match", True)),
                    vocab_sig_match=align.get("vocab_sig_match"))
        for i, (cos2, ratio, label, align, delta) in enumerate(scored))
    return ParentLineup(student=student.model, k=int(k), entries=entries,
                        null_mean_cos2=k / float(student.d),
                        ambiguity_margin=float(ambiguity_margin), d=int(student.d),
                        V_tok=int(student.V_tok), trim=student.trim, notes=tuple(notes))
