"""Privacy-conscious structured diagnostics for local and fleet runs.

The raw child log remains the complete execution record.  These helpers add the small,
stable causal envelope that is safe to put in a scheduler result/event: what phase failed,
which command was attempted, what boundary fired, and the tail that usually contains the
real exception.  Secrets are redacted before command or traceback text leaves the runner.
"""

from __future__ import annotations

import os
import re
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MAX_FAILURE_TAIL_CHARS = 12_000
MAX_FAILURE_TRACEBACK_CHARS = 12_000

_SECRET_FLAGS = {
    "--api-key",
    "--apikey",
    "--authorization",
    "--password",
    "--secret",
    "--token",
    "-p",
}
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|authorization|password|secret|token)\s*[:=]\s*)\S+"
)
_BEARER = re.compile(r"(?i)(\bbearer\s+)\S+")


def redact_text(value: object) -> str:
    """Redact common credential-shaped values from diagnostic text."""

    text = str(value)
    text = _BEARER.sub(r"\1<redacted>", text)
    return _SECRET_ASSIGNMENT.sub(r"\1<redacted>", text)


def redact_command(command: Iterable[object]) -> list[str]:
    """Return a displayable command with values after secret flags removed."""

    redacted: list[str] = []
    redact_next = False
    for raw in command:
        token = str(raw)
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        flag, separator, value = token.partition("=")
        normalized = flag.lower()
        if separator and normalized in _SECRET_FLAGS:
            redacted.append(f"{flag}=<redacted>")
        else:
            redacted.append(redact_text(token))
            if normalized in _SECRET_FLAGS:
                redact_next = True
    return redacted


def log_tail(path: str | Path | None, *, max_chars: int = MAX_FAILURE_TAIL_CHARS) -> str | None:
    """Read only the tail of a run log; missing logs are represented as ``None``."""

    if path is None:
        return None
    try:
        target = Path(path)
        if not target.is_file():
            return None
        # UTF-8 is the normal child-log encoding.  Reading a few extra bytes keeps the
        # character tail close to the requested size without loading a multi-GB log.
        with target.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_chars * 4), os.SEEK_SET)
            text = handle.read().decode("utf-8", errors="replace")
        return redact_text(text[-max_chars:]) or None
    except OSError:
        return None


def exception_details(exc: BaseException) -> dict[str, str]:
    """Serialize an exception without retaining the exception object itself."""

    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "type": type(exc).__name__,
        "message": redact_text(exc),
        "traceback": redact_text(trace[-MAX_FAILURE_TRACEBACK_CHARS:]),
    }


def failure_record(
    *,
    kind: str,
    phase: str,
    message: str,
    command: Iterable[object] | None = None,
    cwd: str | Path | None = None,
    returncode: int | None = None,
    exception: BaseException | None = None,
    log_path: str | Path | None = None,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured causal envelope attached to a failed run."""

    record: dict[str, Any] = {
        "kind": kind,
        "phase": phase,
        "message": redact_text(message),
        "observed_ts": time.time(),
    }
    if command is not None:
        record["command"] = redact_command(command)
    if cwd is not None:
        record["cwd"] = str(cwd)
    if returncode is not None:
        record["returncode"] = int(returncode)
        if int(returncode) < 0:
            record["signal"] = -int(returncode)
    if exception is not None:
        record["exception"] = exception_details(exception)
    tail = log_tail(log_path)
    if tail is not None:
        record["log_tail"] = tail
    if resources:
        record["resources"] = resources
    return record


def execution_diagnostics(
    *,
    command: Iterable[object] | None,
    cwd: str | Path | None,
    phase: str,
    phases: Iterable[dict[str, Any]] = (),
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build non-failure execution context useful when a run succeeded or was lost."""

    context: dict[str, Any] = {
        "phase": phase,
        "phases": [dict(item) for item in phases],
    }
    if command is not None:
        context["command"] = redact_command(command)
    if cwd is not None:
        context["cwd"] = str(cwd)
    if log_path is not None:
        context["log_path"] = str(log_path)
    return context


def normalize_external_result(
    *,
    state: str,
    result: object | None = None,
    detail: object | None = None,
    command: Iterable[object] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Add a compact failure envelope to an external/local terminal result.

    External callers historically sent only ``status`` and ``returncode`` while the
    scheduler stored ``cmd=["<external>"]``.  Keep that wire shape accepted, but make
    the resulting job useful to ``mrun why`` without requiring every caller to upgrade
    at once.  Existing result fields and an explicitly supplied failure record are
    preserved verbatim.  Optional ``stderr_tail``/``stdout_tail`` values are redacted
    and copied into the bounded failure tail.
    """

    normalized = dict(result) if isinstance(result, dict) else {}
    if state == "succeeded" or isinstance(normalized.get("failure"), dict):
        return normalized

    raw_returncode = normalized.get("returncode")
    returncode: int | None = None
    if raw_returncode is not None and not isinstance(raw_returncode, bool):
        try:
            returncode = int(raw_returncode)
        except (TypeError, ValueError):
            pass

    command_values: list[str] | None = None
    raw_command = command
    if raw_command is None:
        candidate = normalized.get("command")
        if isinstance(candidate, (list, tuple)):
            raw_command = candidate
    if raw_command is not None:
        if isinstance(raw_command, (str, bytes)):
            command_values = [str(raw_command)]
        else:
            command_values = [str(value) for value in raw_command]
        if command_values == ["<external>"]:
            command_values = None

    cwd_value: str | Path | None = cwd
    if cwd_value is None and isinstance(normalized.get("cwd"), str):
        cwd_value = normalized["cwd"]

    message = str(detail).strip() if detail is not None else ""
    if not message:
        message = (
            f"external process exited with return code {returncode}"
            if returncode is not None
            else f"external run finished with state {state!r} without diagnostics"
        )
    kind = "external_process_exit" if returncode is not None else f"external_{state}"
    failure = failure_record(
        kind=kind,
        phase="external.finish",
        message=message,
        command=command_values,
        cwd=cwd_value,
        returncode=returncode,
    )
    for key in ("log_tail", "stderr_tail", "stdout_tail"):
        tail = normalized.get(key)
        if tail:
            failure["log_tail"] = redact_text(str(tail))[-MAX_FAILURE_TAIL_CHARS:]
            break
    normalized["failure"] = failure
    return normalized
