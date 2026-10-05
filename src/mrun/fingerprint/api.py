"""Public Python API for the embedding-provenance fingerprint.

One import surface for the whole instrument. It answers ONE question well:

    from mrun.fingerprint import identify_parent

    lineup = identify_parent("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", candidates=[
        "Qwen/Qwen2.5-Math-7B", "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-7B-Instruct"])
    print(lineup.format())     # ranked, with margins and the random-subspace null

and it answers it from ONE tensor per model: the input embedding, pulled by safetensors-header
plus HTTP Range (pennies per model), streamed in row blocks so the working set is O(chunk x d),
reduced to a centered-Gram spectrum plus a cached top-k eigenbasis (~4 MB, which makes every
subsequent question milliseconds).

MEASURED STANDING (experiments/2026-07-26-distillation-fingerprints, 18 checkpoints):
mean cos^2 of principal angles between top-k eigenbases named the true substrate parent at
0.96-0.9998 against wrong same-family siblings at 0.19-0.67, on 4 of 5 known-ground-truth
distillation cases; the fifth (R1-Distill-Llama-8B) is an inherent base-vs-instruct ambiguity
rather than noise.

SCOPE — read this before quoting a result (also carried on every returned object):
  * SUBSTRATE lineage, NOT teacher lineage. Distillation moves the embedding spectrum by 0-5
    dims and never toward the teacher's signature, so "which model taught this one" is out of
    scope and always reported as ``teacher_attribution="OUT-OF-SCOPE"``.
  * Cannot separate distilled-from-X from instruction-tuned-from-X: both are ~0-dim events at
    the input side. It reads base family and scale, not tuning stage.
  * Weight-space only, same-tokenizer only (equal ``d``, equal ``len(tokenizer)``, identical
    trimming — enforced by raising, not warning).
  * NO THRESHOLD EXISTS. Alignment is not a lineage tree: unrelated Qwen2.5-14B/32B scored 0.64
    while Qwen2.5-7B and its own Coder-7B descendant scored 0.19. Only lineups are readable,
    which is why :func:`identify_parent` takes candidates and returns a ranking.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from . import reader as _reader
from .lineup import (
    DEFAULT_AMBIGUITY_MARGIN,
    MatchedObjectError,
    ParentLineup,
    compare,
    rank_candidates,
    spectral_delta,
    subspace_alignment,
)
from .rows import RowDelta, row_delta
from .spectrum import SCOPE, EmbeddingFingerprint, centered_gram, eig_descending, from_arrays
from .spectrum import spectral_counts as _counts

DEFAULT_K = 256           # cached eigenvectors per model (~4 MB at d=4096); k=64 is the
                          # measured lineup default and is a slice of this.
DEFAULT_LINEUP_K = 64


# ------------------------------------------------------------------ cache plumbing
def cache_dir(explicit: str | Path | None = None) -> Path:
    """Where eigenbasis caches live (``MRUN_FINGERPRINT_CACHE`` → ``<artifact_root>/
    fingerprints``). The cache is a first-class feature: it is what turns a re-analysis of an
    18-model panel from a download into a few milliseconds of numpy."""
    return Path(explicit) if explicit is not None else paths.fingerprint_cache_root()


def cache_path(model: str, *, key: str | None = None, trim: bool = True, revision: str = "main",
               explicit: str | Path | None = None) -> Path:
    """Cache file for a (model, key, trimming, revision) tuple. Trimming and key are part of the
    identity because a fingerprint of the untrimmed matrix, or of ``lm_head.weight``, is a
    DIFFERENT object that must never collide with the embedding's entry."""
    stem = paths.safe_stem(f"{model}__{key or 'auto'}__{'tok' if trim else 'cfg'}"
                           + ("" if revision == "main" else f"__{revision}"))
    return cache_dir(explicit) / f"{stem}.npz"


# ------------------------------------------------------------------ per-model leg
def fingerprint(model, *, key: str | None = None, k: int = DEFAULT_K, trim: bool = True,
                chunk: int = 16384, center: str = "auto", cache: bool = True,
                cache_dir: str | Path | None = None, refresh: bool = False,
                prefer_local: bool = True, revision: str = "main",
                label: str | None = None) -> EmbeddingFingerprint:
    """Fingerprint ONE checkpoint's input embedding: centered-Gram spectrum + top-``k`` eigenbasis.

    ``model`` may be a local checkpoint dir or ``.safetensors`` file, an ``mrun.models`` registry
    name, an HF repo id (read by HTTP Range unless it is already in the local hub cache), or a
    :class:`~mrun.fingerprint.reader.TensorSource`. ``key`` defaults to the family's input
    embedding (``lm_head.weight`` is never auto-selected — pass it explicitly to fingerprint the
    output head as a separate object).

    ``trim=True`` (default, and the discipline the measured results used) computes over
    ``len(tokenizer)`` rows rather than the tensor's padded ``vocab_size``; both counts land in
    the result as ``V_tok`` / ``V_cfg`` along with ``trim`` and ``dtype``, so any later
    comparison can be audited or refused.

    Memory: rows stream in ``chunk``-row blocks, so peak is ~``chunk x d`` fp32 plus the ``d x d``
    float64 Gram — a 32B embedding fingerprints in a few hundred MB, not the 3 GB a resident fp32
    copy costs.

    Cached to ``<cache_dir>/<stem>.npz`` and re-served when the cache holds at least ``k``
    eigenvectors (``refresh=True`` forces recompute).
    """
    src = _reader.resolve_source(model, prefer_local=prefer_local, revision=revision)
    name = label or (str(model) if not isinstance(model, _reader.TensorSource) else src.label)
    path = cache_path(name, key=key, trim=trim, revision=revision, explicit=cache_dir)
    if cache and not refresh and path.exists():
        try:
            cached = EmbeddingFingerprint.load(path)
            if cached.k >= k:
                return cached
        except (ValueError, OSError, KeyError):
            pass  # unreadable/legacy cache: recompute rather than guess its provenance

    resolved_key = _reader.find_embedding_key(src, key)
    meta = src.meta(resolved_key)
    if len(meta.shape) != 2:
        raise ValueError(f"{resolved_key} has shape {meta.shape}; expected a 2-D [V, d] matrix")
    v_cfg, d = int(meta.shape[0]), int(meta.shape[1])
    v_tok, vocab_sig = _reader.vocab_info(src)
    if trim and v_tok is not None:
        rows, trim_tag, v_tok_eff = min(int(v_tok), v_cfg), "tok", min(int(v_tok), v_cfg)
    else:
        rows, trim_tag, v_tok_eff = v_cfg, "cfg", int(v_tok) if v_tok is not None else v_cfg

    gram, rows_used = centered_gram(src, resolved_key, rows=rows, chunk=chunk, center=center)
    evals, basis = eig_descending(gram, min(int(k), d))
    counts = _counts(evals)
    fp = EmbeddingFingerprint(
        model=name, key=resolved_key, dtype=meta.dtype, source=src.kind, d=d, V_cfg=v_cfg,
        V_tok=v_tok_eff, trim=trim_tag, basis=basis, evals=evals, vocab_sig=vocab_sig,
        rows_used=rows_used, center=("one-pass" if src.kind == "http-range" else "two-pass")
        if center == "auto" else center, **counts)
    if cache:
        fp.save(path)
    return fp


def _as_fingerprint(ref, *, k: int, label: str | None = None, **fp_kw) -> EmbeddingFingerprint:
    if isinstance(ref, EmbeddingFingerprint):
        return ref
    return fingerprint(ref, k=max(int(k), DEFAULT_K), label=label, **fp_kw)


# ------------------------------------------------------------------ the primary entry point
def identify_parent(student, candidates, *, k: int = DEFAULT_LINEUP_K,
                    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
                    strict_dtype: bool = False, strict_vocab_sig: bool = False,
                    **fp_kw) -> ParentLineup:
    """Rank candidate SUBSTRATE PARENTS for ``student`` by top-``k`` eigenbasis alignment.

    This is the instrument's primary entry point, and it is a LINEUP on purpose. There is no
    ``is_derived_from(student, parent)`` and no cutoff anywhere in this module, because the
    measured alignment numbers are not comparable across lineups (unrelated Qwen2.5-14B/32B =
    0.64; Qwen2.5-7B vs its own Coder-7B descendant = 0.19). At least 2 candidates are required,
    the returned :class:`~mrun.fingerprint.lineup.ParentLineup` refuses ``bool()`` and
    ``float()``, and the only interpretive quantity is the MARGIN between rank 1 and rank 2.

    ``student`` and each candidate may be a model reference (path / registry name / HF repo id)
    or an already-computed :class:`~mrun.fingerprint.spectrum.EmbeddingFingerprint` — pass
    fingerprints (or rely on the eigenbasis cache) to make a whole panel re-analysis
    near-instant. ``candidates`` may be a list, or a dict of ``{label: ref}`` when you want the
    ranking labelled with something other than the reference string.

    Raises :class:`~mrun.fingerprint.lineup.MatchedObjectError` if any candidate differs from the
    student in ``d``, ``len(tokenizer)`` or trimming — pick candidates that share the student's
    tokenizer, which also means cross-tokenizer (i.e. teacher) attribution is unreachable here
    by construction, as it should be.

    Returns a ranking; ``lineup.verdict`` is one of RANKED / AMBIGUOUS / NO-SEPARATION and
    ``lineup.teacher_attribution`` is always ``"OUT-OF-SCOPE"``.
    """
    items = candidates.items() if isinstance(candidates, dict) else [(None, c) for c in candidates]
    pairs = [(lab if lab is not None else _default_label(c), c) for lab, c in items]
    if len(pairs) < 2:
        raise ValueError(
            "identify_parent needs at least 2 candidates: alignment has no absolute scale, so a "
            "single candidate's number is uninterpretable (0.64 was measured between two "
            "UNRELATED models, 0.19 between a model and its own descendant). Offer the "
            "plausible siblings — base / instruct / math / coder / neighbouring scale.")
    student_fp = _as_fingerprint(student, k=k, **fp_kw)
    cand_fps = {lab: _as_fingerprint(ref, k=k, label=lab, **fp_kw) for lab, ref in pairs}
    return rank_candidates(student_fp, cand_fps, k=k, ambiguity_margin=ambiguity_margin,
                           strict_dtype=strict_dtype, strict_vocab_sig=strict_vocab_sig)


def _default_label(ref) -> str:
    return ref.model if isinstance(ref, EmbeddingFingerprint) else str(ref)


def compare_models(left, right, *, k: int = DEFAULT_LINEUP_K, **fp_kw) -> dict:
    """Pairwise convenience: fingerprint both, then spectral deltas + top-``k`` alignment.

    Deliberately NOT a provenance verdict — a pair has no lineup to be read against. Use it for
    the calibration ladder (how big an event was this finetune at the input side?) and use
    :func:`identify_parent` for "who is the parent".
    """
    a = _as_fingerprint(left, k=k, **fp_kw)
    b = _as_fingerprint(right, k=k, **fp_kw)
    return compare(a, b, k=k)


__all__ = [
    "identify_parent", "fingerprint", "compare_models", "row_delta",
    "EmbeddingFingerprint", "ParentLineup", "RowDelta", "MatchedObjectError",
    "from_arrays", "subspace_alignment", "spectral_delta", "compare", "rank_candidates",
    "cache_dir", "cache_path", "SCOPE", "DEFAULT_K", "DEFAULT_LINEUP_K",
    "DEFAULT_AMBIGUITY_MARGIN",
]
