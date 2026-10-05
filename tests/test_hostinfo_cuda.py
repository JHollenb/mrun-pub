from __future__ import annotations

from subprocess import CompletedProcess

from mrun.agent import hostinfo


def test_smi_memory_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        hostinfo.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "16376, 15943\n", ""),
    )
    assert hostinfo._smi_memory_mb() == (16376.0, 15943.0)


def test_smi_process_vram_filters_process_tree(monkeypatch) -> None:
    monkeypatch.setattr(
        hostinfo.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args[0], 0, "101, 512\n202, 1024\nmalformed\n", ""
        ),
    )
    assert hostinfo._smi_process_vram_mb({202, 303}) == 1024.0
