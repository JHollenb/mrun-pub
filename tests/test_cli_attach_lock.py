from __future__ import annotations

from mrun.cli import _exclusive_attach_lock


def test_exclusive_attach_lock_rejects_duplicate_for_same_job(tmp_path):
    with _exclusive_attach_lock("job-one", lock_root=tmp_path) as first:
        assert first is True
        with _exclusive_attach_lock("job-one", lock_root=tmp_path) as duplicate:
            assert duplicate is False


def test_exclusive_attach_lock_allows_different_jobs(tmp_path):
    with _exclusive_attach_lock("job-one", lock_root=tmp_path) as first:
        assert first is True
        with _exclusive_attach_lock("job-two", lock_root=tmp_path) as second:
            assert second is True
