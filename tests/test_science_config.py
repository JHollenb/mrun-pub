import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mrun.engine.base import EngineCapabilities
from mrun.science import benchmark
from mrun.science.config import ScienceConfigError, config_sha256, load_config


def _write_config(path: Path) -> None:
    path.write_text(
        """
schema: mrun-scientific-benchmark-v1
experiment:
  name: toy
  type: benchmark
model:
  name: distilgpt2
runtimes:
  - name: hf
    kind: engine
    backend: hf
test:
  batch_sizes: [1]
  context_tokens: [8]
  decode_tokens: 2
  warmup: 0
  repeats: 1
  prompts: [hello]
serve:
  type: forward
  device: cpu
  dtype: float32
""",
        encoding="utf-8",
    )


def test_loads_defaults_and_dotlist_overrides(tmp_path: Path) -> None:
    path = tmp_path / "science.yaml"
    _write_config(path)
    config = load_config(
        path,
        [
            "test.batch_sizes=[2,4]",
            "tracking.local.enabled=false",
            "runtimes[0].options.device=cuda:0",
        ],
    )
    assert config["test"]["batch_sizes"] == [2, 4]
    assert config["tracking"]["local"]["enabled"] is False
    assert config["runtimes"][0]["options"]["device"] == "cuda:0"
    assert config["model"]["path"] is None
    assert len(config_sha256(config)) == 64


def test_context_size_is_scalar_friendly_and_overrides_matrix(tmp_path: Path) -> None:
    path = tmp_path / "science.yaml"
    _write_config(path)
    config = load_config(path, ["test.context_size=64"])
    assert config["test"]["context_size"] == [64]
    assert config["test"]["context_tokens"] == [64]


def test_rejects_duplicate_runtime_names(tmp_path: Path) -> None:
    path = tmp_path / "science.yaml"
    _write_config(path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    config = load_config(path)
    config["runtimes"].append(dict(config["runtimes"][0]))
    with pytest.raises(ScienceConfigError, match="duplicate runtime"):
        from mrun.science.config import validate_config

        validate_config(config)


def test_fake_benchmark_emits_throughput_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "science.yaml"
    _write_config(path)
    config = load_config(path, [f"execution.output_root={tmp_path / 'results'}"])
    for sink in config["tracking"].values():
        sink["enabled"] = False

    class FakeEngine:
        backend = "fake"
        name = "toy"
        device = "cpu"
        supports_batch = True
        working_set_mb = 1.0
        config = {"text_config": {"max_position_embeddings": 32}}

        def encode(self, prompts: list[str], *, add_special_tokens: bool = False):
            del add_special_tokens
            return [np.arange(len(prompt) + 1, dtype=np.int64) for prompt in prompts]

        def logits_batch(self, rows):
            return [np.zeros((len(row), 4), dtype=np.float32) for row in rows]

        def generate_batch(self, rows, *, max_new_tokens: int):
            return [[1] * max_new_tokens for _ in rows]

        def capabilities(self):
            return EngineCapabilities(logits=True, logits_batch=True, generation_batch=True)

        def close(self):
            return None

    monkeypatch.setattr(benchmark, "open_engine", lambda *args, **kwargs: FakeEngine())
    result = benchmark.run_benchmark(config)
    row = result["manifest"]["results"][0]
    assert row["metrics"]["prefill_tok_s"]["median"] is not None
    assert row["metrics"]["generation_tok_s"]["median"] is not None
    assert row["metrics"]["resource"]["rss_mb_max"] is not None
    assert row["case"]["model_context_limit_tokens"] == 32
    assert row["execution"]["fabric"] == "cpu"
    assert row["execution"]["dtype"] == "fp32"
    assert result["manifest"]["model_context_by_runtime"]["hf"]["max_context_tokens"] == 32
    assert result["manifest"]["model_artifact"]["requested"] == "distilgpt2"

    too_large = load_config(
        path,
        [f"execution.output_root={tmp_path / 'too-large'}", "test.context_size=64"],
    )
    for sink in too_large["tracking"].values():
        sink["enabled"] = False
    with pytest.raises(benchmark.ScienceBenchmarkError, match="max_context_tokens=32"):
        benchmark.run_benchmark(too_large)


def test_source_snapshot_marks_untracked_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        del kwargs
        if command[1:3] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc\n", stderr="")
        if command[1:3] == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(command, 0, stdout="test\n", stderr="")
        if command[1:3] == ["status", "--porcelain=v1"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="?? src/mrun/science/runtime.py\n", stderr=""
            )
        raise AssertionError(command)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    snapshot = benchmark._source_snapshot()
    assert snapshot["dirty"] is True
    assert snapshot["working_tree_status_sha256"] == hashlib.sha256(
        b"?? src/mrun/science/runtime.py\n"
    ).hexdigest()
