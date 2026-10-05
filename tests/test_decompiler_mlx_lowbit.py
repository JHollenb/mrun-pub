from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import mrun.decompiler.mlx_native as mlx_native_module
from mrun.decompiler import (
    SOURCE_MLX_Q2_BITS,
    SOURCE_MLX_Q2_BUILDER_ABI,
    SOURCE_MLX_Q2_CODEC,
    SOURCE_MLX_Q2_GROUP_SIZE,
    SOURCE_MLX_Q2_NATIVE_SCHEMA,
    SOURCE_MLX_Q2_NUMERICAL_CONTRACT,
    SOURCE_MLX_Q3_BITS,
    SOURCE_MLX_Q3_BUILDER_ABI,
    SOURCE_MLX_Q3_CODEC,
    SOURCE_MLX_Q3_GROUP_SIZE,
    SOURCE_MLX_Q3_NATIVE_SCHEMA,
    SOURCE_MLX_Q3_NUMERICAL_CONTRACT,
    MLXSourceQ2Engine,
    MLXSourceQ3Engine,
    SourceMlxArtifactError,
    SourceMlxLoweringError,
    SourceMlxQ2BuildRecord,
    SourceMlxQ3BuildRecord,
    VerifiedSourceMlxQ2Artifact,
    VerifiedSourceMlxQ3Artifact,
    VerifiedSourceMlxQ4Artifact,
    build_component_artifact,
    build_source_mlx_q2_artifact,
    build_source_mlx_q3_artifact,
    build_source_mlx_q4_artifact,
)
from mrun.decompiler.cli import main as decompiler_main


@dataclass(frozen=True, slots=True)
class _Lane:
    bits: int
    schema: str
    builder_abi: str
    codec: str
    group_size: int
    numerical_contract: str
    builder: Callable[..., Any]
    verifier: type[Any]
    record_type: type[Any]

    @property
    def label(self) -> str:
        return f"q{self.bits}"

    @property
    def count_field(self) -> str:
        return f"{self.label}_allocation_count"


LANES = (
    _Lane(
        bits=SOURCE_MLX_Q2_BITS,
        schema=SOURCE_MLX_Q2_NATIVE_SCHEMA,
        builder_abi=SOURCE_MLX_Q2_BUILDER_ABI,
        codec=SOURCE_MLX_Q2_CODEC,
        group_size=SOURCE_MLX_Q2_GROUP_SIZE,
        numerical_contract=SOURCE_MLX_Q2_NUMERICAL_CONTRACT,
        builder=build_source_mlx_q2_artifact,
        verifier=VerifiedSourceMlxQ2Artifact,
        record_type=SourceMlxQ2BuildRecord,
    ),
    _Lane(
        bits=SOURCE_MLX_Q3_BITS,
        schema=SOURCE_MLX_Q3_NATIVE_SCHEMA,
        builder_abi=SOURCE_MLX_Q3_BUILDER_ABI,
        codec=SOURCE_MLX_Q3_CODEC,
        group_size=SOURCE_MLX_Q3_GROUP_SIZE,
        numerical_contract=SOURCE_MLX_Q3_NUMERICAL_CONTRACT,
        builder=build_source_mlx_q3_artifact,
        verifier=VerifiedSourceMlxQ3Artifact,
        record_type=SourceMlxQ3BuildRecord,
    ),
)


def _config(
    family: str,
    *,
    tied: bool,
    hidden: int = 64,
    intermediate: int = 128,
    vocab: int = 11,
) -> dict[str, object]:
    architecture = {
        "llama": "LlamaForCausalLM",
        "mistral": "MistralForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
    }[family]
    config: dict[str, object] = {
        "architectures": [architecture],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": hidden // 4,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "max_position_embeddings": 64,
        "model_type": family,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        "tie_word_embeddings": tied,
        "use_cache": True,
        "vocab_size": vocab,
    }
    if family == "qwen3":
        config["attention_bias"] = False
    if family == "llama":
        config.update({"attention_bias": False, "mlp_bias": False, "pretraining_tp": 1})
    if family == "mistral":
        config["sliding_window"] = None
    return config


def _weights(
    family: str,
    *,
    tied: bool,
    hidden: int = 64,
    intermediate: int = 128,
    vocab: int = 11,
) -> dict[str, torch.Tensor]:
    seed = 700 + {"qwen2": 2, "qwen3": 3, "llama": 4, "mistral": 5}[family]
    generator = torch.Generator().manual_seed(seed)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.05

    prefix = "model.layers.0"
    head_dim = hidden // 4
    kv_width = 2 * head_dim
    weights = {
        "model.embed_tokens.weight": random(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        f"{prefix}.input_layernorm.weight": torch.ones(hidden),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        f"{prefix}.self_attn.q_proj.weight": random(hidden, hidden),
        f"{prefix}.self_attn.k_proj.weight": random(kv_width, hidden),
        f"{prefix}.self_attn.v_proj.weight": random(kv_width, hidden),
        f"{prefix}.self_attn.o_proj.weight": random(hidden, hidden),
        f"{prefix}.mlp.gate_proj.weight": random(intermediate, hidden),
        f"{prefix}.mlp.up_proj.weight": random(intermediate, hidden),
        f"{prefix}.mlp.down_proj.weight": random(hidden, intermediate),
    }
    if family == "qwen2":
        weights.update(
            {
                f"{prefix}.self_attn.q_proj.bias": random(hidden),
                f"{prefix}.self_attn.k_proj.bias": random(kv_width),
                f"{prefix}.self_attn.v_proj.bias": random(kv_width),
            }
        )
    if family == "qwen3":
        weights.update(
            {
                f"{prefix}.self_attn.q_norm.weight": torch.ones(head_dim),
                f"{prefix}.self_attn.k_norm.weight": torch.ones(head_dim),
            }
        )
    if not tied:
        weights["lm_head.weight"] = random(vocab, hidden)
    return weights


def _write_model(
    root: Path,
    family: str,
    *,
    tied: bool,
    hidden: int = 64,
    intermediate: int = 128,
    vocab: int = 11,
    dtype: torch.dtype = torch.float32,
    mutate: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> dict[str, torch.Tensor]:
    root.mkdir(parents=True)
    weights = {
        name: value.to(dtype)
        for name, value in _weights(
            family,
            tied=tied,
            hidden=hidden,
            intermediate=intermediate,
            vocab=vocab,
        ).items()
    }
    if mutate is not None:
        mutate(weights)
    (root / "config.json").write_text(
        json.dumps(
            _config(
                family,
                tied=tied,
                hidden=hidden,
                intermediate=intermediate,
                vocab=vocab,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def _write_runtime_tokenizer(root: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "user": 4,
                "assistant": 5,
                ":": 6,
                "hello": 7,
                "world": 8,
                "yes": 9,
                "no": 10,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "chat_template": (
                    "{% for message in messages %}{{ message['role'] }}: "
                    "{{ message['content'] }}\n{% endfor %}assistant:"
                ),
                "eos_token": "<eos>",
                "model_max_length": 64,
                "pad_token": "<pad>",
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _build_canonical(
    tmp_path: Path,
    family: str,
    *,
    tied: bool,
    hidden: int = 64,
    intermediate: int = 128,
    dtype: torch.dtype = torch.float32,
    mutate: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> tuple[dict[str, torch.Tensor], Any]:
    source = tmp_path / f"source-{family}"
    weights = _write_model(
        source,
        family,
        tied=tied,
        hidden=hidden,
        intermediate=intermediate,
        dtype=dtype,
        mutate=mutate,
    )
    canonical = build_component_artifact(
        source,
        tmp_path / "canonical",
        source_id=f"Test/{family}",
        resolved_revision="c" * 40,
    )
    return weights, canonical


def _parameter_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for shard in manifest["shards"] for item in shard["parameters"]}


def _emitted_tensors(artifact: Any) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for shard in artifact.manifest["shards"]:
        with safe_open(artifact.path / shard["filename"], framework="pt", device="cpu") as handle:
            output.update({name: handle.get_tensor(name) for name in handle.keys()})
    return output


def _resign_manifest(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest.pop("artifact_sha256", None)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _copy_artifact(artifact: Any, parent: Path, name: str) -> Path:
    target = parent / name
    shutil.copytree(artifact.path, target)
    return target


def test_q4_post_v1_recipe_preserves_q4_payload_and_reopen_semantics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlx.core")
    assert importlib.metadata.version("mlx") == "0.32.0"
    implementation_sha = hashlib.sha256(Path(mlx_native_module.__file__).read_bytes()).hexdigest()
    historical_q4_v1_source_sha = "80dee8269bfe16f4e008a3352d67c467ee947e91e812fa94711e92a1785299c1"
    assert implementation_sha != historical_q4_v1_source_sha
    _weights_value, canonical = _build_canonical(tmp_path, "qwen2", tied=True)
    assert canonical.artifact_id == (
        "7c9df5853116d99f13485e1c1951845cb9aa7988ad615376a9327c31f8f51e40"
    )
    assert canonical.manifest_sha256 == (
        "a382b6dbf244d43f67ad72cb347fe35312a98135f59aef48bee4bd24a20a57b1"
    )

    first = build_source_mlx_q4_artifact(canonical.path, tmp_path / "q4-first")
    second = build_source_mlx_q4_artifact(canonical.path, tmp_path / "q4-second")
    first_artifact = VerifiedSourceMlxQ4Artifact(first.path)
    second_artifact = VerifiedSourceMlxQ4Artifact(second.path)

    historical_build_key = "3dcb30d0529cc33ae4301f9c65088c8c01f1c745c794e5f315b6e4abbdf2e8f4"
    historical_artifact_sha = "dcd4d6588ec08b29b3034e38e3c93bd2be971355d0d3d5870517c7b3ebf87d1a"
    assert first.build_key_sha256 == second.build_key_sha256
    assert first.build_key_sha256 != historical_build_key
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.artifact_sha256 != historical_artifact_sha
    # Registering a new built-in adapter changes the compiler implementation fingerprint and the
    # canonical source artifact ID embedded in safetensors metadata.  These file hashes therefore
    # freeze the current post-Mamba lineage; the tensor-level q4/source-exact invariants below and
    # the two independent builds remain the payload-semantics authority.
    expected_shards = {
        "model-body-00001.safetensors": (
            "0b63f3772587a2cc1b933fd31d60232be7f3e3c31064f49f82f9f4efe3fe10a8"
        ),
        "model-lexical_shared-00001.safetensors": (
            "7208532fd96bb45acecbc09694cdf54a116b39dd8917f6374e3ee567f95cdd53"
        ),
        "model-norm-00001.safetensors": (
            "4ac6e0c5d96bd1c5b76fc2ad8113d3074cf365361014500bed6d6bb319257416"
        ),
    }
    manifest_hashes: set[str] = set()
    for artifact in (first_artifact, second_artifact):
        assert {item["filename"]: item["sha256"] for item in artifact.manifest["shards"]} == (
            expected_shards
        )
        manifest_hashes.add(
            hashlib.sha256((artifact.path / "manifest.json").read_bytes()).hexdigest()
        )
        assert hashlib.sha256((artifact.path / "config.json").read_bytes()).hexdigest() == (
            "65ff3e8ed17438160aa3f6ed01b9e387e999bddaed321302e92f383cead4fb91"
        )
    assert len(manifest_hashes) == 1
    assert manifest_hashes != {"bbed1aa77f39e4d7fcfea988d0537f7a0ab6a64f441a21064dbd0830576314c9"}


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.label)
@pytest.mark.parametrize(
    ("family", "tied", "roles"),
    [
        ("qwen2", True, {"body", "norm", "lexical_shared"}),
        ("qwen3", False, {"body", "norm", "ingress", "egress"}),
        ("llama", False, {"body", "norm", "ingress", "egress"}),
    ],
)
def test_q2_q3_role_alias_inventory_packing_and_source_exact_vectors(
    tmp_path: Path,
    lane: _Lane,
    family: str,
    tied: bool,
    roles: set[str],
) -> None:
    pytest.importorskip("mlx.core")
    source_weights, canonical = _build_canonical(tmp_path, family, tied=tied)
    record = lane.builder(canonical.path, tmp_path / lane.label)
    artifact = lane.verifier(record.path)
    manifest = artifact.manifest

    assert isinstance(record, lane.record_type)
    assert record.schema_version == lane.schema
    assert manifest["schema"] == lane.schema
    assert manifest["recipe"]["builder_abi"] == lane.builder_abi
    assert manifest["recipe"]["codec"] == lane.codec
    assert manifest["recipe"]["bits"] == lane.bits
    assert manifest["recipe"]["group_size"] == lane.group_size == 64
    assert manifest["recipe"]["packed_columns_expression"] == "source_columns*bits/32"
    assert manifest["numerical_contract"] == lane.numerical_contract
    assert manifest["status"] == "native-lowered-approximate-unexecuted"
    assert manifest["approximate_quantized"]
    assert manifest["native_runtime_candidate"]
    assert not manifest["execution_certified"]
    assert not manifest["production_runtime_eligible"]
    assert {item["role"] for item in manifest["shards"]} == roles
    assert manifest["coverage"]["auxiliary_source_exact"] is True
    assert manifest["quantization"]["bits"] == lane.bits
    assert manifest["quantization"]["group_size"] == 64
    assert manifest["quantization"]["packed_storage_dtype"] == "U32"
    assert manifest["quantization"][lane.count_field] > 0
    assert math.isfinite(record.max_abs_error) and record.max_abs_error > 0
    assert math.isfinite(record.rmse) and record.rmse > 0

    emitted = _emitted_tensors(artifact)
    descriptors = _parameter_records(manifest)
    matrix_names = {name for name, value in source_weights.items() if value.ndim == 2}
    vector_names = set(source_weights) - matrix_names
    expected_names = set(vector_names)
    for name in matrix_names:
        base = name.removesuffix(".weight")
        expected_names.update({f"{base}.weight", f"{base}.scales", f"{base}.biases"})
    assert set(emitted) == expected_names
    for name in vector_names:
        assert torch.equal(emitted[name], source_weights[name])
        assert descriptors[name]["encoding"] == "source-exact"
    for name in matrix_names:
        base = name.removesuffix(".weight")
        rows, columns = source_weights[name].shape
        assert emitted[f"{base}.weight"].dtype == torch.uint32
        assert list(emitted[f"{base}.weight"].shape) == [
            rows,
            columns * lane.bits // 32,
        ]
        assert list(emitted[f"{base}.scales"].shape) == [rows, columns // 64]
        assert list(emitted[f"{base}.biases"].shape) == [rows, columns // 64]
    if tied:
        assert manifest["source"]["tied_lexical_allocation"]
        assert "lm_head.weight" not in emitted
        assert descriptors["model.embed_tokens.weight"]["logical_names"] == [
            "lm_head.weight",
            "token_embedding.weight",
        ]


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.label)
@pytest.mark.parametrize(
    ("dtype", "source_dtype"),
    [(torch.bfloat16, "BF16"), (torch.float16, "F16")],
    ids=["bf16", "f16"],
)
def test_q2_q3_preserve_allowed_source_dtype_for_vector_auxiliaries(
    tmp_path: Path,
    lane: _Lane,
    dtype: torch.dtype,
    source_dtype: str,
) -> None:
    pytest.importorskip("mlx.core")
    source_weights, canonical = _build_canonical(tmp_path, "qwen2", tied=True, dtype=dtype)
    record = lane.builder(canonical.path, tmp_path / lane.label)
    artifact = lane.verifier(record.path)
    emitted = _emitted_tensors(artifact)
    descriptors = _parameter_records(artifact.manifest)

    assert artifact.source_dtype == source_dtype
    assert artifact.manifest["recipe"]["source_dtype"] == source_dtype
    assert artifact.manifest["recipe"]["quantizer_input_dtype"] == "bfloat16"
    for name, source_value in source_weights.items():
        if source_value.ndim == 1:
            assert torch.equal(emitted[name], source_value)
            assert descriptors[name]["dtype"] == source_dtype
            assert descriptors[name]["encoding"] == "source-exact"
        else:
            base = name.removesuffix(".weight")
            assert emitted[f"{base}.scales"].dtype == torch.bfloat16
            assert emitted[f"{base}.biases"].dtype == torch.bfloat16


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.label)
def test_q2_q3_independent_output_roots_are_deterministic(tmp_path: Path, lane: _Lane) -> None:
    pytest.importorskip("mlx.core")
    _weights_value, canonical = _build_canonical(tmp_path, "qwen2", tied=True)
    first = lane.builder(canonical.path, tmp_path / "first")
    second = lane.builder(canonical.path, tmp_path / "second")
    assert first.path != second.path
    assert first.path.name == second.path.name
    assert first.build_key_sha256 == second.build_key_sha256
    assert first.artifact_sha256 == second.artifact_sha256
    first_manifest = lane.verifier(first.path).manifest
    second_manifest = lane.verifier(second.path).manifest
    assert first_manifest == second_manifest


def test_q2_q3_have_separate_content_identities_and_cross_verifiers_fail(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlx.core")
    _weights_value, canonical = _build_canonical(tmp_path, "qwen2", tied=True)
    q2 = build_source_mlx_q2_artifact(canonical.path, tmp_path / "q2")
    q3 = build_source_mlx_q3_artifact(canonical.path, tmp_path / "q3")
    assert q2.path.name != q3.path.name
    assert q2.build_key_sha256 != q3.build_key_sha256
    assert q2.artifact_sha256 != q3.artifact_sha256
    with pytest.raises(SourceMlxArtifactError, match="q2 artifact schema"):
        VerifiedSourceMlxQ2Artifact(q3.path)
    with pytest.raises(SourceMlxArtifactError, match="q3 artifact schema"):
        VerifiedSourceMlxQ3Artifact(q2.path)
    with pytest.raises(SourceMlxArtifactError, match="q4 artifact schema"):
        VerifiedSourceMlxQ4Artifact(q2.path)

    for engine_type, own, foreign, contract, backend in (
        (
            MLXSourceQ2Engine,
            q2,
            q3,
            SOURCE_MLX_Q2_NUMERICAL_CONTRACT,
            "mlx-source-q2",
        ),
        (
            MLXSourceQ3Engine,
            q3,
            q2,
            SOURCE_MLX_Q3_NUMERICAL_CONTRACT,
            "mlx-source-q3",
        ),
    ):
        assert engine_type.backend == backend
        assert engine_type.artifact_verifier(own.path).artifact_sha256 == own.artifact_sha256
        assert engine_type.experimental_runtime is True
        shell = object.__new__(engine_type)
        assert shell._artifact_numerical_contract() == contract
        with pytest.raises(SourceMlxArtifactError, match="artifact schema"):
            engine_type(
                "qwen2.5-0.5b-instruct",
                source_artifact=canonical.path,
                native_artifact=foreign.path,
            )


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.label)
def test_q2_q3_cli_lower_and_verify_stay_explicitly_unexecuted(
    tmp_path: Path, lane: _Lane, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("mlx.core")
    _weights_value, canonical = _build_canonical(tmp_path, "qwen2", tied=True)
    assert (
        decompiler_main(
            [
                f"lower-mlx-{lane.label}",
                str(canonical.path),
                "--output-root",
                str(tmp_path / lane.label),
            ]
        )
        == 0
    )
    lowered = json.loads(capsys.readouterr().out)
    assert lowered["operation"] == f"lower-mlx-{lane.label}"
    assert lowered["bits"] == lane.bits
    assert lowered["status"] == "native-lowered-approximate-unexecuted"
    assert lowered["build"]["schema_version"] == lane.schema
    assert not lowered["build"]["production_runtime_eligible"]
    assert "experimental runtime route" in lowered["certification_boundary"]

    assert decompiler_main([f"verify-mlx-{lane.label}", lowered["build"]["path"]]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["bits"] == lane.bits
    assert verified["artifact_sha256"] == lowered["build"]["artifact_sha256"]
    assert verified["approximate_quantized"]
    assert not verified["execution_certified"]
    assert not verified["production_runtime_eligible"]


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.label)
def test_q2_q3_verifiers_reject_tamper_resigned_lineage_and_internal_drift(
    tmp_path: Path, lane: _Lane
) -> None:
    pytest.importorskip("mlx.core")
    _weights_value, canonical = _build_canonical(tmp_path, "qwen2", tied=True)
    record = lane.builder(canonical.path, tmp_path / lane.label)
    artifact = lane.verifier(record.path)

    blob_tamper = _copy_artifact(artifact, tmp_path, "blob-tamper")
    shard = blob_tamper / artifact.manifest["shards"][0]["filename"]
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(SourceMlxArtifactError, match="hash"):
        lane.verifier(blob_tamper)

    lineage = _copy_artifact(artifact, tmp_path, "lineage")
    _resign_manifest(
        lineage,
        lambda manifest: manifest["source"].__setitem__("tokenizer_custody_sha256", "0" * 64),
    )
    with pytest.raises(SourceMlxArtifactError, match="lineage"):
        lane.verifier(lineage)

    recipe = _copy_artifact(artifact, tmp_path, "recipe")
    _resign_manifest(
        recipe,
        lambda manifest: manifest["recipe"].__setitem__("bits", 4),
    )
    with pytest.raises(SourceMlxArtifactError, match="recipe ABI"):
        lane.verifier(recipe)

    packed_shape = _copy_artifact(artifact, tmp_path, "packed-shape")

    def mutate_shape(manifest: dict[str, Any]) -> None:
        descriptor = next(
            item
            for shard_record in manifest["shards"]
            for item in shard_record["parameters"]
            if item["part"] == "weight"
        )
        descriptor["shape"][-1] += 1

    _resign_manifest(packed_shape, mutate_shape)
    with pytest.raises(SourceMlxArtifactError, match="shape/dtype"):
        lane.verifier(packed_shape)

    missing_part = _copy_artifact(artifact, tmp_path, "missing-part")

    def remove_part(manifest: dict[str, Any]) -> None:
        for shard_record in manifest["shards"]:
            for index, item in enumerate(shard_record["parameters"]):
                if item["part"] == "biases":
                    del shard_record["parameters"][index]
                    return
        raise AssertionError("fixture has no affine biases")

    _resign_manifest(missing_part, remove_part)
    with pytest.raises(SourceMlxArtifactError, match="header"):
        lane.verifier(missing_part)

    error_evidence = _copy_artifact(artifact, tmp_path, "error-evidence")

    def mutate_error(manifest: dict[str, Any]) -> None:
        manifest["quantization"]["blocks"][0]["rmse"] += 0.25

    _resign_manifest(error_evidence, mutate_error)
    with pytest.raises(SourceMlxArtifactError, match="block evidence"):
        lane.verifier(error_evidence)

    extra = _copy_artifact(artifact, tmp_path, "extra")
    (extra / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(SourceMlxArtifactError, match="undeclared files"):
        lane.verifier(extra)


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.label)
def test_q2_q3_fail_closed_on_width_dtype_finiteness_and_architecture(
    tmp_path: Path, lane: _Lane
) -> None:
    pytest.importorskip("mlx.core")
    narrow_root = tmp_path / "narrow"
    _weights_value, narrow = _build_canonical(
        narrow_root, "qwen2", tied=True, hidden=32, intermediate=64
    )
    with pytest.raises(SourceMlxLoweringError, match="divisible by 64"):
        lane.builder(narrow.path, narrow_root / lane.label)

    double_root = tmp_path / "double"
    _weights_value, double = _build_canonical(double_root, "qwen2", tied=True, dtype=torch.float64)
    with pytest.raises(SourceMlxLoweringError, match="BF16, F16, or F32"):
        lane.builder(double.path, double_root / lane.label)

    mixed_root = tmp_path / "mixed"

    def mixed_dtype(weights: dict[str, torch.Tensor]) -> None:
        weights["model.norm.weight"] = weights["model.norm.weight"].to(torch.float16)

    _weights_value, mixed = _build_canonical(mixed_root, "qwen2", tied=True, mutate=mixed_dtype)
    with pytest.raises(SourceMlxLoweringError, match="one stored dtype"):
        lane.builder(mixed.path, mixed_root / lane.label)

    nonfinite_root = tmp_path / "nonfinite"

    def nonfinite(weights: dict[str, torch.Tensor]) -> None:
        weights["model.layers.0.self_attn.q_proj.weight"][0, 0] = float("nan")

    _weights_value, nonfinite_artifact = _build_canonical(
        nonfinite_root, "qwen2", tied=True, mutate=nonfinite
    )
    with pytest.raises(SourceMlxLoweringError, match="non-finite"):
        lane.builder(nonfinite_artifact.path, nonfinite_root / lane.label)

    unsupported_root = tmp_path / "unsupported"
    _weights_value, unsupported = _build_canonical(unsupported_root, "mistral", tied=False)
    with pytest.raises(SourceMlxLoweringError, match="no certified mapping"):
        lane.builder(unsupported.path, unsupported_root / lane.label)


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.label)
def test_q2_q3_experimental_loader_state_session_service_and_lifecycle_smoke(
    tmp_path: Path, lane: _Lane
) -> None:
    if platform.system() != "Darwin":
        pytest.skip("the MLX runtime route requires Apple Metal")
    pytest.importorskip("mlx.core")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mrun.inference import NativeInferenceConfig, load_native_inference

    source = tmp_path / "runtime-source"
    _write_model(source, "qwen2", tied=True)
    _write_runtime_tokenizer(source)
    canonical = build_component_artifact(
        source,
        tmp_path / "canonical-runtime",
        source_id="Qwen/Qwen2.5-0.5B-Instruct",
        resolved_revision="d" * 40,
    )
    native = lane.builder(canonical.path, tmp_path / lane.label)
    config = NativeInferenceConfig(
        backend=f"mlx-source-{lane.label}",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        source_artifact=canonical.path,
        native_artifact=native.path,
        context_tokens=16,
        headroom_bytes=0,
        max_active_requests=1,
        max_new_tokens=4,
        default_max_tokens=2,
        session_cache_bytes=1024**2,
        session_max_entries=2,
    )
    with load_native_inference(config, bearer_token=None) as stack:
        description = stack.describe()
        assert description["config"]["backend"] == f"mlx-source-{lane.label}"
        assert description["route"]["promotion_status"] == "experimental"
        assert description["engine"]["production_runtime_eligible"] is False
        assert description["engine"]["approximate_quantized"] is True
        assert description["engine"]["weight_bits"] == lane.bits
        assert description["engine"]["numerical_contract"] == lane.numerical_contract
        assert description["execution"]["kv_state"] is None
        assert description["sessions"]["enabled"] is True

        with TestClient(stack.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": stack.tokenizer.model_id,
                    "messages": [{"role": "user", "content": "hello world"}],
                    "max_tokens": 2,
                    "session_id": f"runtime-{lane.label}",
                },
            )
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["completion_tokens"] >= 1
        telemetry = stack.runtime.telemetry()
        assert telemetry.prefill_calls >= 1
        assert telemetry.commits >= 1
        assert stack.session_store is not None
        assert stack.session_store.telemetry().entries == 1
        engine = stack.engine
    assert engine._closed is True
