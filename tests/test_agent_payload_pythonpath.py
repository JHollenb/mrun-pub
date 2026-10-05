from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mrun.agent.config import AgentConfig
from mrun.agent.executor import JobRun, _execution_receipt_requested


def test_shipped_payload_root_is_prepended_to_pythonpath(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv("MRUN_AGENT_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("MRUN_LEASE_CAPABILITY", "must-not-reach-child")
    monkeypatch.setenv("MRUN_LEASE_ID", "must-not-reach-child")
    run = object.__new__(JobRun)
    run.job_id = "job-test"
    run.job = {}
    run.res = SimpleNamespace(cpu_threads=1, ram_mb=512, vram_mb=0)
    run.cfg = SimpleNamespace(server_url=None, token=None)

    env = run._child_env(Path("/tmp/shipped-payload"))

    assert env["PYTHONPATH"] == f"/tmp/shipped-payload{os.pathsep}/existing"
    assert env["MRUN_JOB_ID"] == "job-test"
    assert "MRUN_AGENT_TOKEN" not in env
    assert "MRUN_LEASE_CAPABILITY" not in env
    assert "MRUN_LEASE_ID" not in env


def test_env_alias_virtualenv_precedes_host_command_shims(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/snap/bin:/usr/bin")
    run = object.__new__(JobRun)
    run.job_id = "job-env"
    run.job = {"env_alias": "mrun"}
    run.res = SimpleNamespace(cpu_threads=1, ram_mb=512, vram_mb=0)
    run.cfg = AgentConfig(envs={"mrun": "/home/beast/domains/mrun"})

    env = run._child_env()

    assert env["PATH"].split(os.pathsep)[:2] == [
        "/home/beast/domains/mrun/.venv/bin",
        "/home/beast/domains/mrun/bin",
    ]
    assert env["PATH"].endswith("/snap/bin:/usr/bin")


def test_guarded_debug_child_gets_only_its_narrow_credential(monkeypatch) -> None:
    monkeypatch.setenv("MRUN_TOKEN", "inherited-client")
    monkeypatch.setenv("MRUN_SERVER_TOKEN", "inherited-server")
    monkeypatch.setenv("MRUN_AGENT_TOKEN", "inherited-agent")
    monkeypatch.setenv("MRUN_LEASE_ID", "inherited-lease")
    monkeypatch.setenv("MRUN_LEASE_CAPABILITY", "inherited-capability")
    run = object.__new__(JobRun)
    run.job_id = "job-debug"
    run.job = {
        "needs": {"saturn_debug_credential_v1": True},
        "payload_custody": {"required": True},
    }
    run._debug_authorization = {
        "schema": "mrun.job-debug-credential-v1",
        "credential_id": "dbg-job-debug",
        "credential": "job-debug-only-secret",
    }
    run.res = SimpleNamespace(cpu_threads=1, ram_mb=512, vram_mb=0)
    run.cfg = SimpleNamespace(server_url="http://scheduler", token="broad-client-secret")

    env = run._child_env()

    assert env["MRUN_SERVER_URL"] == "http://scheduler"
    assert env["MRUN_DEBUG_CREDENTIAL_ID"] == "dbg-job-debug"
    assert env["MRUN_DEBUG_CREDENTIAL"] == "job-debug-only-secret"
    for name in (
        "MRUN_TOKEN",
        "MRUN_SERVER_TOKEN",
        "MRUN_AGENT_TOKEN",
        "MRUN_LEASE_ID",
        "MRUN_LEASE_CAPABILITY",
    ):
        assert name not in env


def test_guarded_debug_child_fails_closed_when_narrow_authority_is_missing() -> None:
    run = object.__new__(JobRun)
    run.job_id = "job-debug-missing-authority"
    run.job = {
        "needs": {"saturn_debug_credential_v1": True},
        "payload_custody": {"required": True},
    }
    run._debug_authorization = None
    run.res = SimpleNamespace(cpu_threads=1, ram_mb=512, vram_mb=0)
    run.cfg = SimpleNamespace(
        server_url="http://scheduler",
        token="broad-client-secret-must-not-be-used",
    )

    with pytest.raises(RuntimeError, match="omitted its narrow debugger credential"):
        run._child_env()


def test_sandboxed_debug_child_uses_positive_environment_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cloud-secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/user/1000/ssh-agent")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    monkeypatch.setenv("HF_TOKEN", "model-registry-secret")
    monkeypatch.setenv("MRUN_AGENT_TOKEN", "agent-secret")
    monkeypatch.setenv("LANG", "C.UTF-8")
    run = object.__new__(JobRun)
    run.job_id = "job-sandboxed-debug"
    run.job = {
        "env_alias": "mrun",
        "needs": {
            "saturn_debug_credential_v1": True,
            "payload_sandbox_v1": True,
        },
        "payload_custody": {"required": True},
    }
    run._debug_authorization = {
        "schema": "mrun.job-debug-credential-v1",
        "credential_id": "dbg-sandboxed",
        "credential": "narrow-only",
    }
    run.res = SimpleNamespace(cpu_threads=1, ram_mb=512, vram_mb=0)
    run.cfg = AgentConfig(
        server_url="http://scheduler",
        token="broad-client-secret",
        envs={"mrun": "/home/beast/domains/mrun"},
    )

    env = run._child_env(tmp_path)

    assert env["LANG"] == "C.UTF-8"
    assert env["HOME"] == str(tmp_path / ".sandbox-home")
    assert env["TMPDIR"] == str(tmp_path / ".sandbox-tmp")
    assert env["MRUN_DEBUG_CREDENTIAL"] == "narrow-only"
    assert env["MRUN_SERVER_URL"] == "http://scheduler"
    for name in (
        "AWS_SECRET_ACCESS_KEY",
        "SSH_AUTH_SOCK",
        "DBUS_SESSION_BUS_ADDRESS",
        "HF_TOKEN",
        "MRUN_AGENT_TOKEN",
        "MRUN_SERVER_TOKEN",
        "MRUN_TOKEN",
    ):
        assert name not in env


def test_executed_payload_receipt_is_persisted_for_explicit_v2_need() -> None:
    assert _execution_receipt_requested(
        {
            "needs": {"payload_custody_v2": True},
            "payload_custody": {"required": False},
        }
    )
    assert _execution_receipt_requested({"needs": {}, "payload_custody": {"required": True}})
    assert not _execution_receipt_requested({"needs": {}, "payload_custody": {"required": False}})
