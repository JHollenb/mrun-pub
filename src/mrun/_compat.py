"""Small compatibility helpers for the supported Python versions."""


def add_exception_note(error: BaseException, note: str) -> None:
    """Retain cleanup diagnostics without replacing the original exception."""
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
    else:  # Python 3.10: preserve the same inspectable notes as Python 3.11+.
        notes = list(getattr(error, "__notes__", ()))
        notes.append(note)
        error.__notes__ = notes
