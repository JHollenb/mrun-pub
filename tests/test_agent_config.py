from __future__ import annotations

import json
import os

from mrun.agent import main as agent_main
from mrun.agent.config import AgentConfig, load_config
from mrun.agent.main import Agent


def test_agent_model_roots_merge_with_service_environment(monkeypatch, tmp_path):
    service_root = tmp_path / "service"
    config_root = tmp_path / "config"
    monkeypatch.setenv("LLM_MODELS_EXTRA_ROOT", str(service_root))

    Agent(AgentConfig(host="test", model_roots=[str(config_root), str(service_root)]))

    assert os.environ["LLM_MODELS_EXTRA_ROOT"].split(os.pathsep) == [
        str(service_root),
        str(config_root),
    ]


def test_linux_agent_adds_mounted_model_roots(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_MODELS_EXTRA_ROOT", raising=False)
    monkeypatch.setattr(agent_main.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        agent_main,
        "disks",
        lambda: [{"mount": "/mnt/big"}, {"mount": "/mnt/hdd8tb"}],
    )

    Agent(AgentConfig(host="test", model_roots=[str(tmp_path)]))

    assert os.environ["LLM_MODELS_EXTRA_ROOT"].split(os.pathsep) == [
        str(tmp_path),
        "/mnt/big",
        "/mnt/hdd8tb",
    ]


def test_service_agent_token_overrides_json_without_reaching_child_env(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "agent.json"
    config_path.write_text(json.dumps({"agent_token": "stale-json-token"}))
    monkeypatch.setenv("MRUN_AGENT_CONFIG", str(config_path))
    monkeypatch.setenv("MRUN_AGENT_TOKEN", "independent-service-token")

    assert load_config().agent_token == "independent-service-token"
