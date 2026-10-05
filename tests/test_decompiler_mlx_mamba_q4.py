from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mrun.decompiler import build_component_artifact
from mrun.decompiler.mlx_native import (
    SOURCE_MLX_Q4_GROUP_SIZE,
    MLXSourceQ4Engine,
    SourceMlxArtifactError,
    VerifiedSourceMlxQ4Artifact,
    build_source_mlx_q4_artifact,
)

_CACHED_MAMBA_130M = Path(
    "/Users/jakeholl/.cache/huggingface/hub/"
    "models--state-spaces--mamba-130m-hf/snapshots/"
    "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
)


def _config() -> dict[str, object]:
    return {
        "architectures": ["MambaForCausalLM"],
        "bos_token_id": 0,
        "conv_kernel": 3,
        "expand": 2,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "layer_norm_epsilon": 1e-5,
        "model_type": "mamba",
        "num_hidden_layers": 1,
        "pad_token_id": 0,
        "residual_in_fp32": True,
        "state_size": 4,
        "time_step_rank": 4,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "vocab_size": 128,
    }


def _weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(90210)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.03

    mixer = "backbone.layers.0.mixer"
    return {
        "backbone.embeddings.weight": random(128, 64),
        "backbone.layers.0.norm.weight": torch.ones(64, dtype=torch.float32),
        f"{mixer}.A_log": random(128, 4),
        f"{mixer}.D": random(128),
        f"{mixer}.conv1d.bias": random(128),
        f"{mixer}.conv1d.weight": random(128, 1, 3),
        f"{mixer}.dt_proj.bias": random(128),
        f"{mixer}.dt_proj.weight": random(128, 4),
        f"{mixer}.in_proj.weight": random(256, 64),
        f"{mixer}.out_proj.weight": random(64, 128),
        f"{mixer}.x_proj.weight": random(12, 128),
        "backbone.norm_f.weight": torch.ones(64, dtype=torch.float32),
    }


def _write_model(path: Path) -> dict[str, torch.Tensor]:
    path.mkdir()
    weights = _weights()
    (path / "config.json").write_text(json.dumps(_config(), sort_keys=True), encoding="utf-8")
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    return weights


def _build(tmp_path: Path) -> tuple[dict[str, torch.Tensor], VerifiedSourceMlxQ4Artifact]:
    pytest.importorskip("mlx.core")
    weights = _write_model(tmp_path / "source")
    canonical = build_component_artifact(
        tmp_path / "source",
        tmp_path / "canonical",
        source_id="Test/mamba-q4",
        resolved_revision="d" * 40,
    )
    record = build_source_mlx_q4_artifact(canonical.path, tmp_path / "q4")
    return weights, VerifiedSourceMlxQ4Artifact(record.path)


def _parameters(artifact: VerifiedSourceMlxQ4Artifact) -> dict[str, dict[str, Any]]:
    return {
        str(parameter["name"]): parameter
        for shard in artifact.manifest["shards"]
        for parameter in shard["parameters"]
    }


def _emitted(artifact: VerifiedSourceMlxQ4Artifact) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for shard in artifact.manifest["shards"]:
        with safe_open(artifact.path / shard["filename"], framework="pt", device="cpu") as handle:
            output.update({name: handle.get_tensor(name) for name in handle.keys()})
    return output


def _resign_manifest(path: Path, mutate: Any) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest.pop("artifact_sha256", None)
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    manifest["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def test_mamba_q4_quantizes_only_eligible_modules_and_loads(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.utils import load_model

    source, artifact = _build(tmp_path)
    parameters = _parameters(artifact)
    emitted = _emitted(artifact)
    q4_sources = {
        str(parameter["source_tensor"])
        for parameter in parameters.values()
        if parameter["encoding"] == "mlx-affine-q4-g64"
    }
    expected_q4 = {
        "backbone.embeddings.weight",
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.0.mixer.out_proj.weight",
        "backbone.layers.0.mixer.x_proj.weight",
    }
    assert q4_sources == expected_q4
    assert artifact.manifest["coverage"]["q4_allocation_count"] == len(expected_q4)
    assert artifact.manifest["coverage"]["auxiliary_allocation_count"] == 8
    assert artifact.manifest["source"]["architecture"] == "mamba"
    assert artifact.manifest["native_runtime_candidate"] is True
    assert artifact.manifest["production_runtime_eligible"] is False

    source_exact = set(source) - expected_q4
    for name in source_exact:
        descriptor = parameters[name]
        assert descriptor["encoding"] == "source-exact"
        assert descriptor["shape"] == list(source[name].shape)
        assert torch.equal(emitted[name], source[name])
    assert parameters["backbone.layers.0.mixer.A_log"]["encoding"] == "source-exact"
    assert parameters["backbone.layers.0.mixer.conv1d.weight"]["encoding"] == "source-exact"
    assert parameters["backbone.layers.0.mixer.dt_proj.weight"]["encoding"] == "source-exact"

    reopened = VerifiedSourceMlxQ4Artifact(artifact.path)
    assert reopened.artifact_sha256 == artifact.artifact_sha256
    model, config = load_model(reopened.path, lazy=False, strict=True)
    model.eval()
    block = model.backbone.layers[0].mixer
    assert type(model.backbone.embeddings).__name__ == "QuantizedEmbedding"
    assert type(block.in_proj).__name__ == "QuantizedLinear"
    assert type(block.x_proj).__name__ == "QuantizedLinear"
    assert type(block.out_proj).__name__ == "QuantizedLinear"
    assert type(block.dt_proj).__name__ == "Linear"
    assert type(block.conv1d).__name__ == "Conv1d"
    logits = model(mx.array([[0, 1, 2]]))
    mx.eval(logits)
    assert tuple(logits.shape) == (1, 3, 128)
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert config["quantization"] == {
        "bits": 4,
        "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
        "mode": "affine",
    }


@pytest.mark.parametrize(
    ("source_tensor", "forged_encoding"),
    [
        ("backbone.layers.0.mixer.in_proj.weight", "source-exact"),
        ("backbone.layers.0.mixer.A_log", "mlx-affine-q4-g64"),
    ],
)
def test_mamba_q4_verifier_rejects_resigned_policy_evasion(
    tmp_path: Path,
    source_tensor: str,
    forged_encoding: str,
) -> None:
    _source, artifact = _build(tmp_path)
    forged = tmp_path / "forged"
    shutil.copytree(artifact.path, forged)

    def mutate(manifest: dict[str, Any]) -> None:
        matches = 0
        for shard in manifest["shards"]:
            for parameter in shard["parameters"]:
                if parameter["source_tensor"] == source_tensor:
                    parameter["encoding"] = forged_encoding
                    matches += 1
        assert matches

    _resign_manifest(forged, mutate)
    with pytest.raises(SourceMlxArtifactError, match="architecture policy"):
        VerifiedSourceMlxQ4Artifact(forged)


@pytest.mark.skipif(
    __import__("os").environ.get("MRUN_RUN_MODEL_TESTS") != "1" or not _CACHED_MAMBA_130M.exists(), reason="cached state-spaces/mamba-130m-hf is absent"
)
def test_cached_mamba_130m_q4_build_reopen_and_load(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.utils import load_model

    canonical = build_component_artifact(
        _CACHED_MAMBA_130M,
        tmp_path / "canonical",
        source_id="state-spaces/mamba-130m-hf",
        resolved_revision="1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
    )
    built = build_source_mlx_q4_artifact(canonical.path, tmp_path / "q4")
    artifact = VerifiedSourceMlxQ4Artifact(built.path)

    expected_encoding: dict[str, str] = {}
    for shard in artifact.manifest["shards"]:
        for parameter in shard["parameters"]:
            source_name = str(parameter["source_tensor"])
            source_shape = list(parameter["source_shape"])
            eligible = (
                len(source_shape) == 2
                and source_name.endswith(".weight")
                and source_shape[1] % SOURCE_MLX_Q4_GROUP_SIZE == 0
            )
            encoding = "mlx-affine-q4-g64" if eligible else "source-exact"
            assert parameter["encoding"] == encoding
            previous = expected_encoding.setdefault(source_name, encoding)
            assert previous == encoding

    assert expected_encoding["backbone.layers.0.mixer.A_log"] == "source-exact"
    assert expected_encoding["backbone.layers.0.mixer.conv1d.weight"] == "source-exact"
    assert expected_encoding["backbone.layers.0.mixer.dt_proj.weight"] == "source-exact"
    assert built.shard_bytes < artifact.manifest["coverage"]["source_allocation_bytes"]
    assert built.production_runtime_eligible is False

    model, _config_value = load_model(artifact.path, lazy=False, strict=True)
    model.eval()
    logits = model(mx.array([[0]]))
    mx.eval(logits)
    assert tuple(logits.shape) == (1, 1, 50280)
    assert bool(mx.all(mx.isfinite(logits)).item())

    shell = object.__new__(MLXSourceQ4Engine)
    shell.approximate_quantized = True
    shell.compact_fused_weights = True
    assert shell.capabilities().approximate_quantized is True
