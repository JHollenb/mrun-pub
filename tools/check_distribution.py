"""Audit release archives and exercise a wheel in a clean, external environment."""
from __future__ import annotations

import ast
import email
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile

FORBIDDEN_IMPORTS = {
    "manalysis", "common_harness", "model_experiments", "saturn", "saturn_pub",
    "mlflow", "boto3", "minio", "trackio",
}
FORBIDDEN_DEPENDENCIES = FORBIDDEN_IMPORTS | {"mrun"}
EXCLUDED_PACKAGES = {"decode", "recorder", "mri", "tapes", "render", "mlops"}


def check_member(name: str, raw: bytes) -> None:
    path = Path(name)
    assert not set(path.parts) & {".git", ".venv", "__pycache__"}, name
    assert path.suffix not in {".bak", ".pyc", ".pt", ".safetensors", ".sqlite"}, name
    assert not path.name.startswith(".env"), name
    if "mrun" in path.parts:
        offset = path.parts.index("mrun") + 1
        assert not set(path.parts[offset:offset + 1]) & EXCLUDED_PACKAGES, name
    if path.suffix == ".py":
        for node in ast.walk(ast.parse(raw, filename=name)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            # The standalone benchmark is a consumer of public Saturn/MDB.
            # This never permits a runtime dependency from the mrun package.
            consumer_benchmark = "benchmarks" in path.parts and "mrun" not in path.parts
            forbidden = FORBIDDEN_IMPORTS - ({"saturn_pub"} if consumer_benchmark else set())
            assert not any(m.split(".")[0] in forbidden for m in modules), name


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("mrun_pub-*.whl"))
    assert len(wheels) == 1, "build exactly one release wheel into dist/"
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "mrun/server/static/index.html" in names
        assert "mrun/diffusion/checkpoint_ops.py" in names
        assert "mrun/payload_sandbox.py" in names
        assert any(name.endswith("/licenses/LICENSE") for name in names)
        assert any(name.endswith("/licenses/NOTICE") for name in names)
        metadata = email.message_from_bytes(archive.read(next(n for n in names if n.endswith("/METADATA"))))
        assert metadata["Name"] == "mrun-pub"
        for requirement in metadata.get_all("Requires-Dist", []):
            assert "git+ssh" not in requirement and "file:" not in requirement, requirement
            name = requirement.split(";")[0].split("[")[0].split(" ")[0]
            name = name.split("=")[0].split("<")[0].split(">")[0]
            assert name.replace("-", "_").lower() not in FORBIDDEN_DEPENDENCIES, requirement
        for name in names:
            check_member(name, archive.read(name))
    archives = sorted((root / "dist").glob("*.tar.gz"))
    assert len(archives) == 1, "build exactly one source archive"
    with tarfile.open(archives[0]) as archive:
        for member in archive:
            if member.isfile():
                check_member(member.name, archive.extractfile(member).read())

    with tempfile.TemporaryDirectory(prefix="mrun-public-wheel-") as directory:
        target = Path(directory)
        environment = target / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                        f"{wheel}[server]"], check=True, stdout=subprocess.DEVNULL)
        code = '''
import importlib.abc, importlib.metadata, pathlib, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','diffusers','manalysis','common_harness','model_experiments','saturn','mlflow','boto3','trackio','minio'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
import mrun
from mrun.server.app import app
from mrun.cli import main
from mrun.client.api import DEFAULT_URLS
assert DEFAULT_URLS == ('http://127.0.0.1:9025',)
assert 'site-packages' in str(pathlib.Path(mrun.__file__))
assert importlib.metadata.metadata('mrun-pub')['Name'] == 'mrun-pub'
assert main(['smoke', '--no-model']) == 0
result = mrun.run([sys.executable, '-c', 'print("public wheel")'], ram_limit_mb=128, echo=False)
assert result.ok, result
'''
        subprocess.run([str(python), "-I", "-c", code], cwd=target, check=True)
    print("Release archives and isolated installed-wheel execution passed.")


if __name__ == "__main__":
    main()
