import pytest

from mrun._compat import add_exception_note


@pytest.mark.parametrize("legacy", [False, True])
def test_cleanup_notes_preserve_original_error(legacy):
    error = RuntimeError("primary device failure")
    if legacy:
        error.add_note = None
    add_exception_note(error, "quarantined pages")
    add_exception_note(error, "cleanup failed")
    assert str(error) == "primary device failure"
    assert error.__notes__ == ["quarantined pages", "cleanup failed"]
