"""mrun.fingerprint — embedding-provenance instrument tests.

ALL SYNTHETIC, by constraint as much as by taste: the host this instrument was built on runs
RAM-critical, so nothing here touches a real checkpoint. Every fixture is a planted-spectrum
matrix of at most 400 x 64 float32, fed through the same ``TensorSource`` interface the HTTP-range
path uses, so the guards, the streaming Gram, the lineup ranking, the row-delta leg and the CLI
are all exercised end to end with no network and no model.

The lineup fixture is a *planted* spectrum (well-separated top eigenvalues in a known rotation)
rather than iid noise, because the top-k eigenbasis of a near-degenerate spectrum is not
identifiable and the ranking assertion has to be deterministic.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from mrun.fingerprint import api, reader, spectrum
from mrun.fingerprint import lineup as _lineup
from mrun.fingerprint import rows as _rows

EMB = "model.embed_tokens.weight"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Never write eigenbasis caches into the real artifact root during tests."""
    monkeypatch.setenv("MRUN_FINGERPRINT_CACHE", str(tmp_path / "fpcache"))


# ------------------------------------------------------------------ fixtures
def _planted(seed: int, *, d: int = 64, v: int = 400, top: int = 4, gain: float = 12.0,
             floor: float = 0.4) -> np.ndarray:
    """``[v, d]`` rows drawn in a random orthonormal basis with a decaying, well-separated
    top-``top`` spectrum. The top-``top`` eigenbasis of the centered Gram is then stable to
    small perturbations, which is what makes the ranking assertions deterministic."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    scale = np.concatenate([gain / (1.0 + np.arange(top)), np.full(d - top, floor)])
    return ((rng.standard_normal((v, d)) * scale) @ q.T).astype(np.float32)


def _source(w: np.ndarray, *, label: str, v_tok: int | None = None, extra: dict | None = None,
            dtype: torch.dtype = torch.float32) -> reader.ArraySource:
    tensors = {EMB: torch.tensor(np.asarray(w), dtype=dtype)}
    tensors.update(extra or {})
    aux = {}
    if v_tok is not None:
        aux["tokenizer.json"] = reader.synthetic_tokenizer_json(v_tok)
    return reader.ArraySource(tensors, label=label, aux_files=aux)


def _fp(w: np.ndarray, *, label: str, v_tok: int | None = None, k: int = 16, **kw):
    return api.fingerprint(_source(w, label=label, v_tok=v_tok), label=label, k=k, cache=False,
                           **kw)


@pytest.fixture(scope="module")
def trio():
    """parent / child (parent + small perturbation) / decoy (independent rotation)."""
    parent = _planted(0)
    rng = np.random.default_rng(99)
    # perturbation big enough that the child is NOT a numerical copy (measured students sit at
    # cos2 0.96-0.9998, never exactly 1.0) but small next to the planted gain
    child = parent + rng.standard_normal(parent.shape).astype(np.float32) * 0.5
    decoy = _planted(7)
    return parent, child, decoy


# ------------------------------------------------------------------ spectrum leg
def test_spectrum_recovers_planted_concentration():
    fp = _fp(_planted(0), label="parent", v_tok=400)
    assert fp.d == 64 and fp.V_cfg == 400 and fp.V_tok == 400
    # 4 planted strong directions carry the variance; the isotropic arm is the anti-vacuity
    # control (an instrument that reported "concentrated" for everything would pass the first
    # half of this test alone)
    assert 0 < fp.n50 <= fp.n90 <= fp.n95 <= fp.d
    assert fp.n50 <= 2 and fp.n90 <= 5 and fp.n95 <= 8
    assert 1.0 <= fp.pr <= fp.d
    iso = _fp(np.random.default_rng(4).standard_normal((400, 64)).astype(np.float32),
              label="isotropic", v_tok=400)
    assert iso.n90 > 0.7 * iso.d > 4 * fp.n90
    assert iso.pr > 5 * fp.pr
    row = fp.row()
    # matched-object audit fields present in EVERY row (the lab's hard rule)
    for field in ("dtype", "V_tok", "V_cfg", "trim", "trimmed", "d", "k_cached"):
        assert field in row
    assert row["dtype"] == "F32" and row["trim"] == "tok" and row["trimmed"] is True
    assert row["teacher_attribution"] == "OUT-OF-SCOPE"
    assert "SUBSTRATE" in fp.scope and "TEACHER ATTRIBUTION IS OUT OF SCOPE" in fp.scope


def test_trim_to_tokenizer_length_is_recorded_and_effective():
    w = _planted(0, v=400)
    trimmed = _fp(w, label="m", v_tok=300)
    untrimmed = api.fingerprint(_source(w, label="m", v_tok=300), label="m", k=16, cache=False,
                                trim=False)
    assert (trimmed.trim, trimmed.V_tok, trimmed.rows_used) == ("tok", 300, 300)
    assert (untrimmed.trim, untrimmed.rows_used) == ("cfg", 400)
    assert trimmed.object_signature != untrimmed.object_signature


def test_no_tokenizer_falls_back_to_cfg_rows():
    fp = _fp(_planted(0), label="notok")           # no tokenizer.json in the source
    assert fp.trim == "cfg" and fp.V_tok == fp.V_cfg and fp.vocab_sig is None


def test_one_pass_center_matches_two_pass():
    """The HTTP path halves the bytes moved by centering algebraically; that must not change
    the measurement. Stress case: a large mean offset relative to the spread."""
    src = _source(_planted(0) + 3.0, label="offset")
    g2, _ = spectrum.centered_gram(src, EMB, rows=400, center="two-pass", chunk=64)
    g1, _ = spectrum.centered_gram(src, EMB, rows=400, center="one-pass", chunk=64)
    assert np.abs(g1 - g2).max() / np.abs(g2).max() < 1e-5
    e1, b1 = spectrum.eig_descending(g1, 8)
    e2, b2 = spectrum.eig_descending(g2, 8)
    assert spectrum.spectral_counts(e1)["n50"] == spectrum.spectral_counts(e2)["n50"]
    assert spectrum.spectral_counts(e1)["n90"] == spectrum.spectral_counts(e2)["n90"]
    overlap = b1.T.astype(np.float64) @ b2.astype(np.float64)
    assert (overlap ** 2).sum() / 8 > 1 - 1e-6


def test_row_block_streaming_is_chunk_invariant():
    src = _source(_planted(0), label="m")
    ref, _ = spectrum.centered_gram(src, EMB, rows=400, center="two-pass", chunk=400)
    for chunk in (7, 64, 4096):
        got, used = spectrum.centered_gram(src, EMB, rows=400, center="two-pass", chunk=chunk)
        assert used == 400
        assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-6


def test_lm_head_is_never_auto_selected():
    src = _source(_planted(0), label="untied", v_tok=400,
                  extra={"lm_head.weight": torch.tensor(_planted(3))})
    assert reader.find_embedding_key(src) == EMB
    assert reader.find_embedding_key(src, "lm_head.weight") == "lm_head.weight"
    with pytest.raises(KeyError):
        reader.find_embedding_key(src, "model.nonexistent.weight")


# ------------------------------------------------------------------ same-object guards
def test_matched_object_guard_raises_on_d_mismatch(trio):
    a = _fp(trio[0], label="a", v_tok=400)
    b = _fp(_planted(1, d=32), label="b", v_tok=400)
    with pytest.raises(_lineup.MatchedObjectError, match=r"d 64 vs 32"):
        _lineup.assert_matched(a, b)
    with pytest.raises(_lineup.MatchedObjectError):
        _lineup.subspace_alignment(a, b, k=8)


def test_matched_object_guard_raises_on_vocab_length_mismatch(trio):
    a = _fp(trio[0], label="a", v_tok=400)
    b = _fp(trio[2], label="b", v_tok=380)
    with pytest.raises(_lineup.MatchedObjectError, match=r"V_tok 400 vs 380"):
        _lineup.assert_matched(a, b)


def test_matched_object_guard_raises_on_trim_mismatch(trio):
    a = _fp(trio[0], label="a", v_tok=400)
    b = api.fingerprint(_source(trio[2], label="b", v_tok=400), label="b", k=16, cache=False,
                        trim=False)
    with pytest.raises(_lineup.MatchedObjectError, match=r"trim"):
        _lineup.assert_matched(a, b)


def test_dtype_and_vocab_sig_differences_are_recorded_then_optionally_fatal(trio):
    a = _fp(trio[0], label="a", v_tok=400)
    b_src = _source(trio[2], label="b", dtype=torch.float64)
    b_src._aux["tokenizer.json"] = reader.synthetic_tokenizer_json(400, salt="x")
    b = api.fingerprint(b_src, label="b", k=16, cache=False)
    notes = _lineup.assert_matched(a, b)                       # recorded, not fatal
    assert notes["dtype_match"] is False and notes["vocab_sig_match"] is False
    assert "dtype" in notes["warning"]
    with pytest.raises(_lineup.MatchedObjectError, match="strict_dtype"):
        _lineup.assert_matched(a, b, strict_dtype=True)
    with pytest.raises(_lineup.MatchedObjectError, match="strict_vocab_sig"):
        _lineup.assert_matched(a, b, strict_vocab_sig=True)


def test_guard_fires_through_identify_parent(trio):
    """The guard must be unavoidable from the primary entry point, not just the low-level call."""
    student = _fp(trio[1], label="student", v_tok=400)
    ok = _fp(trio[0], label="parent", v_tok=400)
    bad = _fp(_planted(5, d=32), label="other-d", v_tok=400)
    with pytest.raises(_lineup.MatchedObjectError):
        api.identify_parent(student, [ok, bad], k=8)


# ------------------------------------------------------------------ the lineup
def test_lineup_ranks_planted_parent_first(trio):
    parent, child, decoy = trio
    student = _fp(child, label="child", v_tok=400)
    result = api.identify_parent(student, {
        "parent": _fp(parent, label="parent", v_tok=400),
        "decoy": _fp(decoy, label="decoy", v_tok=400),
        "decoy2": _fp(_planted(11), label="decoy2", v_tok=400),
    }, k=4)
    assert [e.candidate for e in result.entries][0] == "parent"
    assert result.best_match.mean_cos2 > 0.95
    assert all(e.mean_cos2 < 0.5 for e in result.entries[1:])
    assert result.margin > 0.45 and result.resolved is True
    assert result.verdict.startswith("RANKED")
    assert result.null_mean_cos2 == pytest.approx(4 / 64)
    assert result.best_match.ratio_to_null > 10
    assert [e.rank for e in result.entries] == [1, 2, 3]
    d = result.as_dict()
    assert d["best_match"] == "parent" and d["teacher_attribution"] == "OUT-OF-SCOPE"
    assert "parent" in result.format() and "OUT-OF-SCOPE" in result.format()


def test_lineup_reports_ambiguity_instead_of_a_winner(trio):
    """Two candidates that are near-identical to each other must read AMBIGUOUS, the way the
    measured R1-Distill-Llama-8B base-vs-instruct case does (0.9942 vs 0.9946)."""
    parent, child, _ = trio
    rng = np.random.default_rng(5)
    twin = parent + rng.standard_normal(parent.shape).astype(np.float32) * 1e-4
    result = api.identify_parent(_fp(child, label="child", v_tok=400), {
        "parent": _fp(parent, label="parent", v_tok=400),
        "parent-twin": _fp(twin, label="twin", v_tok=400),
        "decoy": _fp(_planted(7), label="decoy", v_tok=400),
    }, k=4)
    assert result.margin < result.ambiguity_margin
    assert result.resolved is False
    assert result.verdict.startswith("AMBIGUOUS")
    assert result.spread > result.ambiguity_margin       # the decoy is still separated


def test_lineup_reports_no_separation_when_nothing_distinguishes(trio):
    parent, _, _ = trio
    rng = np.random.default_rng(6)
    a = parent + rng.standard_normal(parent.shape).astype(np.float32) * 1e-4
    b = parent + rng.standard_normal(parent.shape).astype(np.float32) * 1e-4
    result = api.identify_parent(_fp(parent + 0.0, label="student", v_tok=400), {
        "a": _fp(a, label="a", v_tok=400), "b": _fp(b, label="b", v_tok=400)}, k=4)
    assert result.verdict.startswith("NO-SEPARATION") and result.resolved is False


def test_lineup_cannot_be_used_as_a_threshold(trio):
    """Structural, not documentary: the object refuses scalar/boolean coercion and the entry
    point refuses a single candidate — because alignment has no absolute scale (measured: 0.64
    between unrelated models, 0.19 between a model and its own descendant)."""
    parent, child, decoy = trio
    student = _fp(child, label="child", v_tok=400)
    result = api.identify_parent(student, [_fp(parent, label="parent", v_tok=400),
                                           _fp(decoy, label="decoy", v_tok=400)], k=4)
    with pytest.raises(TypeError, match="no truth value"):
        bool(result)
    with pytest.raises(TypeError, match="not a score"):
        float(result)
    with pytest.raises(ValueError, match="at least 2 candidates"):
        api.identify_parent(student, [_fp(parent, label="parent", v_tok=400)], k=4)
    assert not hasattr(result, "is_derived_from")
    assert not any("threshold" in name for name in dir(result))


def test_lineup_rejects_self_and_duplicate_candidates(trio):
    parent, child, decoy = trio
    student = _fp(child, label="child", v_tok=400)
    with pytest.raises(ValueError, match="own candidates"):
        _lineup.rank_candidates(student, {"child": _fp(child, label="child", v_tok=400),
                                          "parent": _fp(parent, label="parent", v_tok=400)}, k=4)
    with pytest.raises(ValueError, match="duplicate candidate"):
        _lineup.rank_candidates(student, [_fp(parent, label="dup", v_tok=400),
                                          _fp(decoy, label="dup", v_tok=400)], k=4)


def test_alignment_is_symmetric_and_null_scaled(trio):
    parent, _, decoy = trio
    a = _fp(parent, label="a", v_tok=400)
    b = _fp(decoy, label="b", v_tok=400)
    ab = _lineup.subspace_alignment(a, b, k=8)
    ba = _lineup.subspace_alignment(b, a, k=8)
    assert ab["mean_cos2"] == pytest.approx(ba["mean_cos2"], rel=1e-9)
    assert _lineup.subspace_alignment(a, a, k=8)["mean_cos2"] == pytest.approx(1.0, abs=1e-6)
    assert ab["null_mean_cos2"] == pytest.approx(8 / 64)
    assert ab["ratio_to_null"] == pytest.approx(ab["mean_cos2"] / (8 / 64))


def test_compare_reports_spectral_delta_and_alignment(trio):
    parent, child, _ = trio
    out = _lineup.compare(_fp(child, label="child", v_tok=400),
                          _fp(parent, label="parent", v_tok=400), k=4)
    assert out["spectral"]["dn50"] is not None and out["spectral"]["dn90"] is not None
    assert out["alignment"]["mean_cos2"] > 0.9
    assert "SUBSTRATE" in out["scope"]


def test_alignment_degrades_past_the_identifiable_rank(trio):
    """k is not free: beyond the number of well-separated directions the eigenbasis is not
    identifiable and alignment falls, which is the same k-sensitivity the measured panel shows
    (wrong candidates move 0.19 -> 0.33 between k=64 and k=256). A caller comparing numbers
    across k values is comparing different objects."""
    parent, child, _ = trio
    a, b = _fp(child, label="child", v_tok=400), _fp(parent, label="parent", v_tok=400)
    at_rank = _lineup.subspace_alignment(a, b, k=4)["mean_cos2"]        # planted rank = 4
    past_rank = _lineup.subspace_alignment(a, b, k=16)["mean_cos2"]
    assert at_rank > 0.9 > past_rank


def test_subspace_larger_than_cache_raises(trio):
    fp = _fp(trio[0], label="a", v_tok=400, k=8)
    assert fp.subspace(8).shape == (64, 8)
    with pytest.raises(ValueError, match="cached basis has k=8"):
        fp.subspace(16)


# ------------------------------------------------------------------ eigenbasis cache
def test_cache_roundtrip_and_reuse(tmp_path, trio):
    src = _source(trio[0], label="cached", v_tok=400)
    first = api.fingerprint(src, label="cached", k=16, cache_dir=tmp_path)
    path = api.cache_path("cached", trim=True, explicit=tmp_path)
    assert path.exists()
    again = api.fingerprint(src, label="cached", k=16, cache_dir=tmp_path)
    assert again.row() == first.row()
    np.testing.assert_allclose(again.basis, first.basis)
    # a request for MORE eigenvectors than the cache holds must recompute, not silently truncate
    bigger = api.fingerprint(src, label="cached", k=32, cache_dir=tmp_path)
    assert bigger.k == 32
    # trimming and key are part of the cache identity
    assert api.cache_path("cached", trim=False, explicit=tmp_path) != path
    assert api.cache_path("cached", key="lm_head.weight", explicit=tmp_path) != path


def test_legacy_cache_without_metadata_is_refused(tmp_path, trio):
    """The experiment's cache/*.npz hold only evecs/evals. Adopting them is supported, but the
    matched-object fields must be supplied — never invented from array shapes."""
    fp = _fp(trio[0], label="legacy", v_tok=400, k=16)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, evecs=fp.basis, evals=fp.evals[:16])
    with pytest.raises(ValueError, match="matched-object fields cannot be inferred"):
        spectrum.EmbeddingFingerprint.load(legacy)
    adopted = spectrum.EmbeddingFingerprint.load(legacy, meta={
        "model": "legacy", "d": 64, "V_cfg": 400, "V_tok": 400, "trim": "tok", "dtype": "BF16",
        "n50": fp.n50, "n90": fp.n90})
    assert adopted.object_signature == fp.object_signature
    assert _lineup.subspace_alignment(adopted, fp, k=8)["mean_cos2"] == pytest.approx(1.0,
                                                                                     abs=1e-6)


def test_from_arrays_requires_matched_object_fields(trio):
    fp = _fp(trio[0], label="a", v_tok=400, k=8)
    with pytest.raises(TypeError):
        spectrum.from_arrays(fp.basis, model="x")            # d/V_cfg/V_tok are required
    with pytest.raises(ValueError, match="trim must be"):
        spectrum.from_arrays(fp.basis, model="x", d=64, V_cfg=400, V_tok=400, trim="whatever")
    with pytest.raises(ValueError, match="rows but d="):
        spectrum.from_arrays(fp.basis, model="x", d=32, V_cfg=400, V_tok=400)


# ------------------------------------------------------------------ row-delta leg
def test_row_delta_detects_exact_identity(trio):
    src = _source(trio[0], label="same", v_tok=400)
    rd = _rows.row_delta(src, _source(trio[0], label="same-copy", v_tok=400))
    assert rd.byte_identical is True
    assert rd.exact_row_frac == 1.0
    assert rd.relF == pytest.approx(0.0)
    assert rd.row_reld_median == pytest.approx(0.0)
    assert rd.V_common == 400 and rd.n_rows == 400 and rd.sampled is False
    assert rd.vocab_mismatch_frac == 0.0 and rd.vocab_gate_n == 2000
    assert rd.as_dict()["validated"] is False        # honest standing of this leg


def test_row_delta_measures_a_perturbation(trio):
    parent, child, decoy = trio
    near = _rows.row_delta(_source(child, label="child", v_tok=400),
                           _source(parent, label="parent", v_tok=400))
    far = _rows.row_delta(_source(decoy, label="decoy", v_tok=400),
                          _source(parent, label="parent", v_tok=400))
    assert near.byte_identical is False and near.exact_row_frac == 0.0
    assert 0 < near.relF < far.relF
    assert near.row_reld_median < far.row_reld_median
    assert near.row_reld_median <= near.row_reld_p90 <= near.row_reld_max


def test_row_delta_sampling_is_deterministic_and_forfeits_identity(trio):
    src_a = _source(trio[0], label="a", v_tok=400)
    src_b = _source(trio[0], label="b", v_tok=400)
    one = _rows.row_delta(src_a, src_b, sample=64, chunk=17)
    two = _rows.row_delta(src_a, src_b, sample=64, chunk=97)
    assert one.n_rows == two.n_rows == 64 and one.sampled is True
    assert one.relF == pytest.approx(two.relF)
    assert one.byte_identical is None            # a sample cannot prove identity
    assert "identity not provable" in one.note


def test_row_delta_guard_raises_on_unmatched_objects(trio):
    with pytest.raises(_lineup.MatchedObjectError, match="d 64 vs 32"):
        _rows.row_delta(_source(trio[0], label="a", v_tok=400),
                        _source(_planted(2, d=32), label="b", v_tok=400))
    with pytest.raises(_lineup.MatchedObjectError, match=r"len\(tokenizer\)"):
        _rows.row_delta(_source(trio[0], label="a", v_tok=400),
                        _source(trio[2], label="b", v_tok=380))


def test_row_delta_excludes_degenerate_norm_rows_from_relative_stats(trio):
    """Reserved / never-trained vocabulary rows are not zero, they are astronomically small.
    MEASURED 2026-07-26: 289 of Meta-Llama-3.1-8B's 128256 rows have norm ~2e-21 against a
    median of 0.688, and one moved by 0.005 in the Instruct checkpoint — a relative delta of
    2.5e18 that owned row_reld_max outright under fp_pairs.py's clamp(1e-30). "Relative to a row
    carrying no magnitude" is undefined, not enormous, and the cut is a fraction of the median
    row norm rather than an absolute constant."""
    left = trio[0].copy()
    right = trio[0].copy()
    right[7:11] = 2e-21                    # reserved rows on the right (denominator) arm
    left[7:11] = 0.3                       # ...which the left arm did write to
    rd = _rows.row_delta(_source(left, label="a", v_tok=400),
                         _source(right, label="b", v_tok=400))
    assert rd.degenerate_norm_rows == 4
    assert rd.n_rows == 400 and rd.n_rows_in_stats == 396      # all compared, 396 in the stats
    assert rd.median_row_norm > 1.0
    assert np.isfinite(rd.row_reld_max) and rd.row_reld_max < 1.0
    assert np.isfinite(rd.relF) and rd.relF > 0                # relF keeps them, well defined
    assert "degenerate norm" in rd.note
    assert rd.byte_identical is False and rd.exact_row_frac == pytest.approx(396 / 400)
    # the naive clamp would have produced ~1e20 here; assert the defect cannot come back
    assert rd.row_reld_max < 1e3


def test_row_delta_flags_padding_only_difference(trio):
    """Common rows identical but V_cfg differs — a real checkpoint pattern (padded vocab)."""
    padded = np.vstack([trio[0], np.zeros((8, 64), dtype=np.float32)])
    rd = _rows.row_delta(_source(trio[0], label="a", v_tok=400),
                         _source(padded, label="b", v_tok=400))
    assert rd.V_common == 400 and rd.exact_row_frac == 1.0
    assert rd.byte_identical is False and "padding-only" in rd.note


# ------------------------------------------------------------------ source-kind dispatch
def test_remote_kind_selects_the_one_pass_centering(trio):
    """``center='auto'`` must pick the bytes-halving arm for the HTTP path and the
    measured-script arm otherwise; both are recorded on the row so a result can be audited."""
    class _Remote(reader.ArraySource):
        kind = "http-range"

    remote = _Remote({EMB: torch.tensor(trio[0])}, label="remote",
                     aux_files={"tokenizer.json": reader.synthetic_tokenizer_json(400)})
    fp = api.fingerprint(remote, label="remote", k=8, cache=False)
    assert (fp.source, fp.center) == ("http-range", "one-pass")
    local = api.fingerprint(_source(trio[0], label="local", v_tok=400), label="local", k=8,
                            cache=False)
    assert (local.source, local.center) == ("arrays", "two-pass")
    # same measurement either way
    assert _lineup.subspace_alignment(fp, local, k=4)["mean_cos2"] == pytest.approx(1.0, abs=1e-6)


requires_network = pytest.mark.skipif(
    os.environ.get("MRUN_RUN_NETWORK_TESTS") != "1",
    reason="set MRUN_RUN_NETWORK_TESTS=1 to hit the HF hub")


@requires_network
def test_http_range_reads_a_row_block_without_the_checkpoint():
    """The economics claim, live: 64 rows out of a ~1 GB checkpoint. Ran green 2026-07-26 —
    Qwen2.5-0.5B embed_tokens BF16 [151936, 896], 114 KB moved, V_tok 151665 (the value recorded
    in the measured panel)."""
    src = reader.HubSource("Qwen/Qwen2.5-0.5B")
    key = reader.find_embedding_key(src)
    meta = src.meta(key)
    assert key == EMB and meta.dtype == "BF16" and meta.shape == (151936, 896)
    block = src.read_rows(key, 100, 164)
    assert tuple(block.shape) == (64, 896) and block.dtype is torch.bfloat16
    v_tok, sig = reader.vocab_info(src)
    assert v_tok == 151665 and sig


# ------------------------------------------------------------------ local checkpoint + CLI
def _write_checkpoint(root, w: np.ndarray, v_tok: int) -> None:
    from safetensors.torch import save_file
    root.mkdir(parents=True, exist_ok=True)
    save_file({EMB: torch.tensor(w)}, str(root / "model.safetensors"))
    (root / "tokenizer.json").write_bytes(reader.synthetic_tokenizer_json(v_tok))


def test_local_safetensors_checkpoint_path(tmp_path, trio):
    ckpt = tmp_path / "ckpt"
    _write_checkpoint(ckpt, trio[0], 380)
    src = reader.resolve_source(str(ckpt))
    assert isinstance(src, reader.LocalSource) and src.kind == "local"
    assert src.meta(EMB).shape == (400, 64) and src.meta(EMB).dtype == "F32"
    fp = api.fingerprint(str(ckpt), k=8)
    assert (fp.V_cfg, fp.V_tok, fp.trim, fp.rows_used) == (400, 380, "tok", 380)
    assert fp.source == "local" and fp.center == "two-pass"


def test_resolve_source_rejects_nonsense():
    with pytest.raises(ValueError, match="not a local path"):
        reader.resolve_source("definitely-not-a-model")


def test_cli_spectrum_and_lineup(tmp_path, capsys, trio):
    from mrun.cli import main

    parent, child, decoy = trio
    for name, w in (("parent", parent), ("child", child), ("decoy", decoy)):
        _write_checkpoint(tmp_path / name, w, 400)
    paths = {n: str(tmp_path / n) for n in ("parent", "child", "decoy")}

    assert main(["fingerprint", "spectrum", paths["parent"], "--k", "8", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["d"] == 64 and rows[0]["trim"] == "tok"
    assert rows[0]["teacher_attribution"] == "OUT-OF-SCOPE"

    out = tmp_path / "lineup.jsonl"
    assert main(["fingerprint", "lineup", paths["child"], "--candidate", paths["parent"],
                 "--candidate", paths["decoy"], "--k", "4", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "RANKED" in text and "parent" in text
    row = json.loads(out.read_text().splitlines()[0])
    assert row["ranking"][0]["candidate"] == paths["parent"]
    assert row["resolved"] is True

    assert main(["fingerprint", "rows", paths["parent"], paths["parent"], "--sample", "32"]) == 0
    assert "byte_identical=None" in capsys.readouterr().out


def test_cli_lineup_refuses_single_candidate(tmp_path, trio):
    from mrun.cli import main

    for name, w in (("parent", trio[0]), ("child", trio[1])):
        _write_checkpoint(tmp_path / name, w, 400)
    with pytest.raises(ValueError, match="at least 2 candidates"):
        main(["fingerprint", "lineup", str(tmp_path / "child"),
              "--candidate", str(tmp_path / "parent")])
