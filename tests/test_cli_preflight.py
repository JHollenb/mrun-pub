from __future__ import annotations

import json
from pathlib import Path

from mrun.cli import main


def _payload(tmp_path: Path) -> Path:
    root = tmp_path / "payload"
    (root / "src").mkdir(parents=True)
    (root / "src" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    return root


def test_cli_preflight_pass_exit_zero(tmp_path, capsys):
    root = _payload(tmp_path)
    rc = main(
        [
            "preflight",
            str(root),
            "--experiment",
            "cli-pf",
            "--offline",
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--cmd",
            "python",
            "src/run.py",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "preflight: passed" in out
    assert list((tmp_path / "receipts").glob("preflight-*.json"))


def test_cli_preflight_reject_exit_one_names_token(tmp_path, capsys):
    root = _payload(tmp_path)
    rc = main(
        [
            "preflight",
            str(root),
            "--experiment",
            "cli-pf",
            "--offline",
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--json",
            "--cmd",
            "python",
            "src/run.py",
            "--parent-verification",
            "custody/missing.json",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "custody/missing.json" in out
    receipt_path = next((tmp_path / "receipts").glob("preflight-*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "rejected"
