"""Create a bounded source snapshot; never copy Git history, secrets or model bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

EXCLUDED_PACKAGES = {"decode", "recorder", "mri", "tapes", "render", "mlops"}
EXCLUDED_MODULES = {"dx.py", "dx_cas.py", "dx_custody.py", "analysis_campaign.py"}
EXCLUDED_TESTS = {
    "test_manalysis_delegation.py", "test_recorder_acceptance.py", "test_recorder_device.py",
    "test_prompt_conditioned.py", "test_decode.py", "test_mri.py", "test_render.py",
    "test_mlops.py", "test_postprocess.py", "test_tapes_sink.py", "test_dx.py",
    "test_moe_mixtral_adapter.py", "test_causal_family_batch.py", "test_analysis_campaign.py",
}
RELOCATIONS = {
    "src/mrun/mri/moe_stream.py": "src/mrun/engine/moe_safetensors.py",
    "src/mrun/mri/deepseek_v4_stream.py": "src/mrun/engine/deepseek_v4_stream.py",
    "src/mrun/mri/deepseek_v4_parity.py": "src/mrun/testing/deepseek_v4_parity.py",
}


def export(upstream: Path, destination: Path) -> None:
    source_files = subprocess.check_output(
        ["git", "-C", str(upstream), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    rows = []
    excluded = []
    for relative in sorted(set(source_files)):
        parts = Path(relative).parts
        path = upstream / relative
        if not path.is_file() or path.is_symlink():
            continue
        target = RELOCATIONS.get(relative, relative)
        keep = False
        if relative in RELOCATIONS:
            keep = True
        elif parts[:2] == ("src", "mrun"):
            keep = (len(parts) < 3 or parts[2] not in EXCLUDED_PACKAGES) and path.name not in EXCLUDED_MODULES
        elif parts[0] == "tests":
            keep = path.name not in EXCLUDED_TESTS
        elif relative == "scripts/run-inference-service.sh":
            keep = True
        if path.suffix == ".bak" or path.suffix in {".pyc", ".pyo"}:
            keep = False
        if not keep:
            excluded.append(relative)
            continue
        output = destination / target
        if output.exists():
            raise FileExistsError(f"refusing to replace an existing export: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        rows.append({"source": relative, "target": target,
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    receipt = {"schema": "mrun-public-source-snapshot.v1", "source_commit": commit,
               "includes_working_tree": True, "copied": rows, "excluded": excluded}
    (destination / "SOURCE-SNAPSHOT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Copied {len(rows)} source/test files; excluded {len(excluded)} other files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    arguments = parser.parse_args()
    export(arguments.upstream.resolve(), arguments.destination.resolve())
