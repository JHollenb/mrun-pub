import ast
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys


def test_source_has_no_private_or_service_imports():
    import mrun
    from conftest import FORBIDDEN

    for path in Path(mrun.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(name.split(".")[0] in FORBIDDEN for name in names), path


def test_base_and_server_import_without_torch_or_workspace(tmp_path):
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','diffusers','manalysis','common_harness','mlflow','boto3'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
import mrun
from mrun.client.api import Api
from mrun.server.app import app
from mrun.models import model_search_roots
from mrun.cli import main
assert Api()._candidates == ['http://127.0.0.1:9025']
assert main(['smoke', '--no-model']) == 0
'''
    environment = {key: value for key, value in os.environ.items() if key != "MRUN_URL"}
    environment["MRUN_SERVER_DATA"] = str(tmp_path / "server")
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, check=True)
    assert importlib.metadata.metadata("mrun-pub")["Name"] == "mrun-pub"


def test_scheduler_requires_token_by_default(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mrun.server import app as server
    from mrun.server.db import DB

    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    monkeypatch.delenv("MRUN_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setattr(server, "db", DB(tmp_path / "db.sqlite"))
    with TestClient(server.app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/hosts").status_code == 503
        monkeypatch.setenv("MRUN_TOKEN", "test-public-client")
        assert client.get("/api/hosts").status_code == 401
        assert client.get("/api/hosts", headers={"x-mrun-token":"test-public-client"}).status_code == 200


def test_unauthenticated_opt_in_is_local_only(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mrun.server import app as server
    from mrun.server.db import DB

    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    monkeypatch.setattr(server, "db", DB(tmp_path / "db.sqlite"))
    with TestClient(server.app, client=("203.0.113.1", 1234)) as client:
        assert client.get("/api/hosts").status_code == 503
