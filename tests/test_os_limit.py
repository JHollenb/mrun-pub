from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mrun.os_limit import (
    CGROUP_V2_BACKEND,
    prepare_os_memory_limit,
    strict_scope_oom_killed,
    verify_current_cgroup_limit,
)


def _controller_root(tmp_path: Path) -> Path:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu io memory pids\n")
    return root


def test_off_leaves_command_unwrapped() -> None:
    launch = prepare_os_memory_limit(["python", "job.py"], limit_mb=512, mode="off")
    assert launch.command == ["python", "job.py"]
    assert not launch.strict and launch.backend == "off"


def test_auto_is_sampled_only_when_strict_backend_is_unsupported(tmp_path: Path) -> None:
    launch = prepare_os_memory_limit(
        ["python", "job.py"],
        limit_mb=512,
        mode="auto",
        system="Darwin",
        cgroup_root=tmp_path,
        systemd_run="/usr/bin/systemd-run",
    )
    assert launch.command == ["python", "job.py"]
    assert not launch.strict and launch.backend == "sampled-only"


def test_required_refuses_unsupported_platform(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="RLIMIT_RSS is advisory"):
        prepare_os_memory_limit(
            ["python", "job.py"],
            limit_mb=512,
            mode="required",
            system="Darwin",
            cgroup_root=tmp_path,
            systemd_run="/usr/bin/systemd-run",
        )


def test_linux_wraps_command_in_verified_systemd_scope(tmp_path: Path) -> None:
    root = _controller_root(tmp_path)
    launch = prepare_os_memory_limit(
        ["python", "job.py"],
        limit_mb=512,
        mode="required",
        identity="job/unsafe id",
        system="Linux",
        cgroup_root=root,
        systemd_run="/usr/bin/systemd-run",
    )
    assert launch.strict and launch.backend == CGROUP_V2_BACKEND
    assert launch.unit_name == "mrun-job-unsafe-id.scope"
    assert "--property=MemoryMax=536870912" in launch.command
    assert "--property=MemorySwapMax=0" in launch.command
    assert "--property=OOMPolicy=kill" in launch.command
    separator = launch.command.index("--")
    assert launch.command[separator + 2 : separator + 5] == [
        "-m",
        "mrun.os_limit",
        "exec",
    ]
    assert launch.command[-2:] == ["python", "job.py"]


def test_in_scope_verifier_accepts_exact_memory_and_zero_swap(tmp_path: Path) -> None:
    root = _controller_root(tmp_path)
    scope = root / "user.slice" / "mrun-job.scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text("536870912\n")
    (scope / "memory.swap.max").write_text("0\n")
    (scope / "memory.oom.group").write_text("1\n")
    proc = tmp_path / "proc-cgroup"
    proc.write_text("0::/user.slice/mrun-job.scope\n")

    assert verify_current_cgroup_limit(
        536870912, cgroup_root=root, proc_cgroup=proc
    ) == scope.resolve()


@pytest.mark.parametrize(
    ("memory_max", "swap_max", "oom_group", "message"),
    [
        ("max", "0", "1", "memory.max"),
        ("536875008", "0", "1", "memory.max"),
        ("536870912", "max", "1", "memory.swap.max"),
        ("536870912", "0", "0", "memory.oom.group"),
    ],
)
def test_in_scope_verifier_fails_closed(
    tmp_path: Path, memory_max: str, swap_max: str, oom_group: str, message: str
) -> None:
    root = _controller_root(tmp_path)
    scope = root / "job.scope"
    scope.mkdir()
    (scope / "memory.max").write_text(memory_max)
    (scope / "memory.swap.max").write_text(swap_max)
    (scope / "memory.oom.group").write_text(oom_group)
    proc = tmp_path / "proc-cgroup"
    proc.write_text("0::/job.scope\n")

    with pytest.raises(RuntimeError, match=message):
        verify_current_cgroup_limit(536870912, cgroup_root=root, proc_cgroup=proc)


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="os memory limit mode"):
        prepare_os_memory_limit(["true"], limit_mb=1, mode="maybe")


def test_systemd_oom_result_is_authoritative_and_cleaned(tmp_path: Path) -> None:
    root = _controller_root(tmp_path)
    launch = prepare_os_memory_limit(
        ["true"],
        limit_mb=128,
        mode="required",
        identity="oom",
        system="Linux",
        cgroup_root=root,
        systemd_run="/usr/bin/systemd-run",
    )
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "show" in command:
            return subprocess.CompletedProcess(command, 0, stdout="oom-kill\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    assert strict_scope_oom_killed(launch, 255, run=fake_run)
    assert any("show" in command for command in calls)
    assert any("reset-failed" in command for command in calls)


def test_non_oom_systemd_failure_is_not_misclassified(tmp_path: Path) -> None:
    root = _controller_root(tmp_path)
    launch = prepare_os_memory_limit(
        ["true"],
        limit_mb=128,
        mode="required",
        system="Linux",
        cgroup_root=root,
        systemd_run="/usr/bin/systemd-run",
    )

    def fake_run(command, **_kwargs):
        result = "exit-code\n" if "show" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=result, stderr="")

    assert not strict_scope_oom_killed(launch, 255, run=fake_run)
