from __future__ import annotations

from pathlib import Path

import pytest

from mrun.payload_sandbox import (
    PAYLOAD_SANDBOX_BACKEND,
    PayloadSandboxError,
    prepare_payload_sandbox,
)


def _job(read_path: Path) -> dict:
    return {
        "payload_kind": "shipped",
        "needs": {"payload_sandbox_v1": True},
        "config": {
            "payload_sandbox": {
                "schema": "mrun-payload-sandbox-request-v1",
                "required": True,
                "read_only_paths": [str(read_path)],
            }
        },
    }


def test_required_sandbox_builds_private_profile_without_secret_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    environment = tmp_path / "env"
    models = tmp_path / "models"
    cache = models / ".uv-cache"
    uv = tmp_path / "uv"
    for directory in (work, environment, models, cache):
        directory.mkdir(parents=True, exist_ok=True)
    uv.write_text("binary")
    monkeypatch.setattr("mrun.payload_sandbox.payload_sandbox_supported", lambda **_kwargs: True)

    launch = prepare_payload_sandbox(
        ["python", "worker.py"],
        job=_job(models),
        work_dir=work,
        env_path=environment,
        allowed_read_roots=[str(models)],
        shared_cache_paths=[str(cache)],
        bwrap="/usr/bin/bwrap",
        uv_executable=str(uv),
        system="Linux",
    )

    assert launch.strict is True
    assert launch.backend == PAYLOAD_SANDBOX_BACKEND
    command = launch.command
    assert ["--tmpfs", "/home"] == command[command.index("/home") - 1 : command.index("/home") + 1]
    assert ["--bind", str(work), str(work)] == command[
        command.index(str(work), command.index("--bind")) - 1 : command.index(
            str(work), command.index("--bind")
        )
        + 2
    ]
    assert "--unshare-pid" in command
    assert "--disable-userns" in command
    assert "--tmp-overlay" in command
    assert "MRUN_AGENT_TOKEN" not in "\0".join(command)
    assert launch.profile["host_home_visible"] is False


def test_sandbox_rejects_read_path_outside_configured_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    models = tmp_path / "models"
    secret = tmp_path / "secret"
    for directory in (work, models, secret):
        directory.mkdir()
    monkeypatch.setattr("mrun.payload_sandbox.payload_sandbox_supported", lambda **_kwargs: True)

    with pytest.raises(PayloadSandboxError, match="escapes configured roots"):
        prepare_payload_sandbox(
            ["true"],
            job=_job(secret),
            work_dir=work,
            env_path=None,
            allowed_read_roots=[str(models)],
            bwrap="/usr/bin/bwrap",
            system="Linux",
        )


def test_sandbox_fails_closed_when_bubblewrap_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    models = tmp_path / "models"
    work.mkdir()
    models.mkdir()
    monkeypatch.setattr("mrun.payload_sandbox.payload_sandbox_supported", lambda **_kwargs: False)

    with pytest.raises(PayloadSandboxError, match="was required"):
        prepare_payload_sandbox(
            ["true"],
            job=_job(models),
            work_dir=work,
            env_path=None,
            allowed_read_roots=[str(models)],
            bwrap="/usr/bin/bwrap",
            system="Linux",
        )
