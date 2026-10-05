from __future__ import annotations

from mrun.agent import hostinfo


def test_inventory_store_roots_discovers_beast_sibling_qstores(tmp_path, monkeypatch):
    models = tmp_path / "llm-models"
    configured = models / "qstores"
    sibling = tmp_path / "qstores"
    configured.mkdir(parents=True)
    sibling.mkdir()
    monkeypatch.setattr(hostinfo, "models_root", lambda: models)

    roots = hostinfo._inventory_store_roots(configured)

    assert roots == [configured.resolve(), sibling.resolve()]
