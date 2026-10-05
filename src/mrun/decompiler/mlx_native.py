"""Direct canonical source-component lowering to native MLX safetensors.

This G9 target consumes ``mrun-native-source-component-v1`` allocation blobs directly.  It does
not reconstruct a Transformers model and does not pass through QStore.  The closed support set
mirrors the executable G8 reference target: registered dense decoders, classic routed-only
Mixtral, and Mamba1 selective-state-space causal LMs with raw, uniform floating-point identity
views.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from mrun.engine.mlx_component import (
    MLXComponentEngine,
    _canonical_json_bytes,
    _fsync_directory,
    _hash_regular_file,
    _identity,
    _read_regular_json,
    _runtime_tokenizer_descriptor,
    _safe_model_slug,
    _sha256_bytes,
    _write_durable,
)

from .emitter import ComponentArtifact, open_component_artifact
from .errors import DecompilerError
from .reference import ReferenceLoweringError, lower_component_artifact_to_reference

SOURCE_MLX_NATIVE_SCHEMA = "mrun-mlx-source-component-native-v1"
SOURCE_MLX_BUILDER_ABI = "mrun-canonical-source-to-mlx-native-v1"
SOURCE_MLX_MAPPING_ABI = "mrun-dense-hf-source-name-to-mlx-lm-v2"
_SOURCE_MLX_RAW_MAPPING_ABIS = frozenset(
    {
        "mrun-dense-hf-source-name-to-mlx-lm-v1",
        SOURCE_MLX_MAPPING_ABI,
        "mrun-gpt2-hf-namespace-to-mlx-lm-v1",
    }
)
SOURCE_MLX_SHARD_BYTES = 256 * 1024**2
SOURCE_MLX_Q4_NATIVE_SCHEMA = "mrun-mlx-source-component-q4-native-v1"
SOURCE_MLX_Q4_BUILDER_ABI = "mrun-canonical-source-to-mlx-affine-q4-native-v1"
SOURCE_MLX_Q4_CODEC = "mlx-affine-int4-g64-bf16-direct-canonical-source-v1"
SOURCE_MLX_Q4_GROUP_SIZE = 64
SOURCE_MLX_Q4_BITS = 4
SOURCE_MLX_Q4_MODE = "affine"
SOURCE_MLX_Q4_NUMERICAL_CONTRACT = (
    "mlx-source-component-q4g64-bf16-weight-approximate-source-aux-exact-v1"
)

_ARCHITECTURES = {
    "gpt2-causal-decoder": "gpt2",
    "gpt-neox-pythia-causal-decoder": "gpt_neox",
    "qwen2-dense-causal-decoder": "qwen2",
    "qwen3-dense-causal-decoder": "qwen3",
    "llama-dense-causal-decoder": "llama",
    "mistral-dense-causal-decoder": "mistral",
    "mixtral-sparse-moe-causal-decoder": "mixtral",
    "mamba1-selective-state-space-causal-decoder": "mamba",
    "phi-causal-decoder": "phi",
}
_SOURCE_MLX_Q4_ARCHITECTURES = frozenset({"llama", "mamba", "qwen2", "qwen3"})
_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "F64": 64}
_SAFE_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


class SourceMlxLoweringError(DecompilerError):
    """The canonical source artifact is outside the direct MLX G9 support set."""

    code = "source_mlx_lowering_rejection"
    gate = "G9"


class SourceMlxArtifactError(DecompilerError):
    """A derived direct-source MLX artifact failed integrity verification."""

    code = "source_mlx_artifact_failure"
    gate = "G9"


def _is_lower_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _strict_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceMlxArtifactError(f"{field} must be a positive integer")
    return int(value)


def _build_lock(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(key, threading.Lock())


def _source_dtype_codec(dtype: str) -> str:
    if dtype not in _TORCH_DTYPES:
        raise SourceMlxLoweringError(f"direct MLX lowering does not support source dtype {dtype!r}")
    return f"safetensors-{dtype.lower()}-native-v1"


def _source_q4_dtype(dtype: str) -> str:
    if dtype not in {"BF16", "F16", "F32"}:
        raise SourceMlxLoweringError(
            "direct MLX affine q4 lowering requires BF16, F16, or F32 canonical weights"
        )
    return dtype


def _tokenizer_custody_sha256(artifact: ComponentArtifact) -> str:
    """Bind the native artifact to every semantic and physical tokenizer decision."""

    io = artifact.ir_bundle.io
    payload = {
        "io_schema": io.schema_version,
        "io_fingerprint": io.fingerprint,
        "text_spaces": [item.as_dict() for item in io.text_spaces],
        "row_mappers": [item.as_dict() for item in io.row_mappers],
        "special_tokens": [item.as_dict() for item in io.special_tokens],
        "tokenizer_assets": [item.as_dict() for item in io.tokenizer_assets],
        "chat_templates": [item.as_dict() for item in io.chat_templates],
        "output_spaces": [item.as_dict() for item in io.output_spaces],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _parameter_role(logical_names: Sequence[str]) -> str:
    names = set(logical_names)
    ingress = "token_embedding.weight" in names
    egress = "lm_head.weight" in names
    if ingress and egress:
        return "lexical_shared"
    if ingress:
        return "ingress"
    if egress:
        return "egress"
    if any(name.startswith("final_norm.") for name in names):
        return "norm"
    return "body"


def _native_semantic_compatibility(architecture: str) -> str:
    if architecture == "gpt2":
        return "mlx-lm-gpt2-gelu-approx-native-parity-required"
    if architecture == "gpt_neox":
        return "mlx-lm-gpt-neox-approximate-gelu-bounded-parity-required"
    if architecture == "phi":
        return "mlx-lm-phi-approximate-gelu-bounded-parity-required"
    if architecture == "mixtral":
        return "mlx-lm-mixtral-topk-switchglu-bounded-parity-required"
    if architecture == "mamba":
        return "mlx-lm-mamba1-sequential-selective-scan-f32-bounded-parity-required"
    return "mlx-lm-source-architecture-native-parity-required"


def _native_mapping_abi(architecture: str) -> str:
    if architecture == "gpt2":
        return "mrun-gpt2-hf-namespace-to-mlx-lm-v1"
    return SOURCE_MLX_MAPPING_ABI


def _native_parameter_name(architecture: str, source_name: str) -> str:
    if architecture == "gpt2":
        prefix = "transformer."
        if not source_name.startswith(prefix):
            raise SourceMlxLoweringError(
                "tied GPT-2 direct lowering requires every allocation in transformer.*"
            )
        return source_name.removeprefix(prefix)
    return source_name


def _validated_config(artifact: ComponentArtifact) -> tuple[dict[str, Any], str, bool]:
    model = artifact.ir_bundle.model
    architecture = _ARCHITECTURES.get(model.architecture_id)
    if architecture is None:
        raise SourceMlxLoweringError("canonical artifact architecture has no direct MLX target")
    config = json.loads(_canonical_json_bytes(artifact.source.config))
    dimensions = model.dimensions
    if str(config.get("model_type")) != architecture:
        raise SourceMlxLoweringError("source config model_type differs from canonical architecture")
    if architecture == "mamba":
        expected = {
            "hidden_size": dimensions.hidden_size,
            "intermediate_size": dimensions.intermediate_size,
            "num_hidden_layers": dimensions.num_hidden_layers,
            "vocab_size": dimensions.vocab_size,
        }
        for field, value in expected.items():
            if isinstance(config.get(field), bool) or int(config.get(field, -1)) != value:
                raise SourceMlxLoweringError(
                    f"source Mamba config field {field!r} differs from ModelIR"
                )
        if (
            dimensions.num_attention_heads
            or dimensions.num_key_value_heads
            or dimensions.head_dim
            or dimensions.max_position_embeddings
        ):
            raise SourceMlxLoweringError(
                "Mamba ModelIR must declare zero attention dimensions and no positional ceiling"
            )
        for field in ("state_size", "conv_kernel", "time_step_rank"):
            value = config.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SourceMlxLoweringError(
                    f"source Mamba config field {field!r} must be a positive integer"
                )
        if config.get("hidden_act", "silu") != "silu":
            raise SourceMlxLoweringError("mlx-lm Mamba target requires SiLU")
        if config.get("use_mambapy", False) is not False:
            raise SourceMlxLoweringError("mlx-lm target does not implement MambaPy arithmetic")
        if config.get("ssm_cfg", {}) not in ({}, None):
            raise SourceMlxLoweringError("mlx-lm Mamba1 target requires an empty ssm_cfg")
        epsilon = config.get("layer_norm_epsilon", 1e-5)
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or float(epsilon) != 1e-5
        ):
            raise SourceMlxLoweringError(
                "mlx-lm Mamba1 hard-codes the registered 1e-5 RMSNorm epsilon"
            )
        stored_dtypes = {
            allocation.stored_dtype
            for allocation in artifact.ir_bundle.physical_weights.allocations
        }
        if stored_dtypes != {"F32"}:
            raise SourceMlxLoweringError(
                "direct Mamba1 MLX lowering currently requires uniform F32 source weights"
            )
        alias_sets = [
            set(item.logical_names) for item in artifact.ir_bundle.physical_weights.alias_classes
        ]
        tied = {"token_embedding.weight", "lm_head.weight"} in alias_sets
        declared_tied = config.get("tie_word_embeddings", True)
        if type(declared_tied) is not bool or declared_tied != tied:
            raise SourceMlxLoweringError(
                "source Mamba tie_word_embeddings differs from physical alias IR"
            )
        config["vocab_size"] = dimensions.physical_vocab_rows
        config["tie_word_embeddings"] = tied
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        return config, architecture, tied
    if architecture == "gpt2":
        expected = {
            "n_embd": dimensions.hidden_size,
            "n_layer": dimensions.num_hidden_layers,
            "n_head": dimensions.num_attention_heads,
            "n_positions": dimensions.max_position_embeddings,
            "vocab_size": dimensions.vocab_size,
        }
        for field, value in expected.items():
            if isinstance(config.get(field), bool) or int(config.get(field, -1)) != value:
                raise SourceMlxLoweringError(f"source config field {field!r} differs from ModelIR")
        if dimensions.num_key_value_heads != dimensions.num_attention_heads:
            raise SourceMlxLoweringError("mlx-lm GPT-2 requires symmetric attention heads")
        if dimensions.intermediate_size != 4 * dimensions.hidden_size:
            raise SourceMlxLoweringError("mlx-lm GPT-2 hard-codes MLP width to four times n_embd")
        if config.get("activation_function", "gelu_new") != "gelu_new":
            raise SourceMlxLoweringError("mlx-lm GPT-2 supports only gelu_new semantics")
        if config.get("tie_word_embeddings", True) is not True:
            raise SourceMlxLoweringError("mlx-lm GPT-2 requires tied lexical matrices")
        config["n_ctx"] = dimensions.max_position_embeddings
        config["n_inner"] = 4 * dimensions.hidden_size
        config["num_key_value_heads"] = dimensions.num_key_value_heads
        observed_head_dim = dimensions.hidden_size // dimensions.num_attention_heads
    else:
        expected = {
            "hidden_size": dimensions.hidden_size,
            "intermediate_size": dimensions.intermediate_size,
            "num_hidden_layers": dimensions.num_hidden_layers,
            "num_attention_heads": dimensions.num_attention_heads,
            "vocab_size": dimensions.vocab_size,
            "max_position_embeddings": dimensions.max_position_embeddings,
        }
        if architecture != "gpt_neox":
            expected["num_key_value_heads"] = dimensions.num_key_value_heads
        for field, value in expected.items():
            if isinstance(config.get(field), bool) or int(config.get(field, -1)) != value:
                raise SourceMlxLoweringError(f"source config field {field!r} differs from ModelIR")
        head_dim = config.get("head_dim")
        observed_head_dim = (
            dimensions.hidden_size // dimensions.num_attention_heads
            if head_dim is None
            else int(head_dim)
        )
    # mlx-lm constructs the physical embedding/head allocations from config.vocab_size. Preserve
    # the semantic token domain separately in the canonical IO contract while exposing any
    # authenticated padded rows required by the source tensors to the native model constructor.
    config["vocab_size"] = dimensions.physical_vocab_rows
    if observed_head_dim != dimensions.head_dim:
        raise SourceMlxLoweringError("source config head_dim differs from ModelIR")
    if architecture == "gpt_neox":
        raw_kv_heads = config.get("num_key_value_heads", dimensions.num_attention_heads)
        if raw_kv_heads is None:
            raw_kv_heads = dimensions.num_attention_heads
        if (
            raw_kv_heads != dimensions.num_attention_heads
            or dimensions.num_key_value_heads != dimensions.num_attention_heads
        ):
            raise SourceMlxLoweringError(
                "mlx-lm GPT-NeoX requires symmetric query/key/value head counts"
            )
        if dimensions.intermediate_size != 4 * dimensions.hidden_size:
            raise SourceMlxLoweringError(
                "mlx-lm GPT-NeoX hard-codes intermediate_size to four times hidden_size"
            )
        if config.get("hidden_act", "gelu") != "gelu":
            raise SourceMlxLoweringError("mlx-lm GPT-NeoX supports only its GELU path")
        if config.get("attention_bias", True) is not True:
            raise SourceMlxLoweringError("mlx-lm GPT-NeoX requires attention projection biases")
        if config.get("tie_word_embeddings", False) is not False:
            raise SourceMlxLoweringError("mlx-lm GPT-NeoX requires untied lexical matrices")
    if architecture == "phi":
        if config.get("hidden_act", "gelu_new") != "gelu_new":
            raise SourceMlxLoweringError("mlx-lm Phi supports only gelu_new semantics")
        if config.get("qk_layernorm", False) is not False:
            raise SourceMlxLoweringError("mlx-lm Phi target does not implement qk_layernorm")
        if config.get("tie_word_embeddings", False) is not False:
            raise SourceMlxLoweringError("mlx-lm Phi requires untied lexical matrices")
    if architecture == "mixtral":
        experts = config.get("num_local_experts")
        top_k = config.get("num_experts_per_tok")
        if (
            isinstance(experts, bool)
            or not isinstance(experts, int)
            or experts <= 0
            or isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
            or top_k > experts
        ):
            raise SourceMlxLoweringError("mlx-lm Mixtral requires a valid routed expert/top-k pair")
        if config.get("hidden_act", "silu") != "silu":
            raise SourceMlxLoweringError("mlx-lm Mixtral requires SiLU SwitchGLU experts")
        if config.get("attention_bias", False) is not False:
            raise SourceMlxLoweringError("mlx-lm Mixtral requires bias-free attention projections")
        if config.get("router_jitter_noise", 0.0) not in (0, 0.0, None):
            raise SourceMlxLoweringError("mlx-lm Mixtral target does not implement router jitter")
        if config.get("sliding_window") is not None:
            raise SourceMlxLoweringError("mlx-lm Mixtral target requires global attention")
        if config.get("rope_traditional", False) is not False:
            raise SourceMlxLoweringError("mlx-lm Mixtral target requires non-traditional RoPE")
    rope_theta: float | None = None
    if architecture != "gpt2":
        rope_scaling = config.get("rope_scaling")
        if rope_scaling is not None:
            raise SourceMlxLoweringError(
                "direct MLX target currently requires unscaled default RoPE"
            )
        raw_rope_theta = config.get("rope_theta")
        if architecture == "gpt_neox" and raw_rope_theta is None:
            raw_rope_theta = config.get("rotary_emb_base")
        if raw_rope_theta is None:
            rope_parameters = config.get("rope_parameters")
            if (
                isinstance(rope_parameters, Mapping)
                and rope_parameters.get("rope_type", "default") == "default"
            ):
                raw_rope_theta = rope_parameters.get("rope_theta")
        if (
            isinstance(raw_rope_theta, bool)
            or not isinstance(raw_rope_theta, (int, float))
            or not math.isfinite(float(raw_rope_theta))
            or float(raw_rope_theta) <= 0
        ):
            raise SourceMlxLoweringError("source config has no finite positive default rope_theta")
        rope_theta = float(raw_rope_theta)
        config["rope_theta"] = rope_theta
        config["head_dim"] = dimensions.head_dim
        config["num_key_value_heads"] = dimensions.num_key_value_heads
    if architecture == "gpt_neox":
        rope_parameters = config.get("rope_parameters")
        rotary_pct = config.get("rotary_pct", 0.25)
        if isinstance(rope_parameters, Mapping):
            rotary_pct = rope_parameters.get("partial_rotary_factor", rotary_pct)
        if (
            isinstance(rotary_pct, bool)
            or not isinstance(rotary_pct, (int, float))
            or not math.isfinite(float(rotary_pct))
            or not 0 < float(rotary_pct) <= 1
            or int(dimensions.head_dim * float(rotary_pct)) <= 0
            or int(dimensions.head_dim * float(rotary_pct)) % 2
        ):
            raise SourceMlxLoweringError("source config has no valid GPT-NeoX rotary fraction")
        if rope_theta is None:
            raise AssertionError("GPT-NeoX RoPE validation did not produce a base")
        config["rotary_emb_base"] = rope_theta
        config["rotary_pct"] = float(rotary_pct)
    if architecture == "phi":
        partial = config.get("partial_rotary_factor", 0.5)
        if (
            isinstance(partial, bool)
            or not isinstance(partial, (int, float))
            or not math.isfinite(float(partial))
            or not 0 < float(partial) <= 1
            or int(dimensions.head_dim * float(partial)) <= 0
            or int(dimensions.head_dim * float(partial)) % 2
        ):
            raise SourceMlxLoweringError("source config has no valid Phi rotary fraction")
        config["partial_rotary_factor"] = float(partial)

    alias_sets = [
        set(item.logical_names) for item in artifact.ir_bundle.physical_weights.alias_classes
    ]
    tied = {"token_embedding.weight", "lm_head.weight"} in alias_sets
    declared_tied = config.get("tie_word_embeddings", architecture == "gpt2")
    if type(declared_tied) is not bool or declared_tied != tied:
        raise SourceMlxLoweringError("source tie_word_embeddings differs from physical alias IR")
    config["tie_word_embeddings"] = tied
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    return config, architecture, tied


def _read_allocation(
    artifact: ComponentArtifact,
    allocation: Any,
    record: Mapping[str, Any],
) -> torch.Tensor:
    blob = record.get("blob")
    if not isinstance(blob, Mapping):
        raise SourceMlxArtifactError("canonical allocation record has no blob")
    expected_path = f"blobs/{allocation.allocation_id}.bin"
    if blob.get("path") != expected_path:
        raise SourceMlxArtifactError("canonical allocation blob locator is not exact")
    path = artifact.directory / expected_path
    if path.is_symlink():
        raise SourceMlxArtifactError("canonical allocation blob cannot be a symlink")
    expected_bytes = int(blob.get("byte_count", -1))
    expected_hash = str(blob.get("sha256", ""))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceMlxArtifactError("cannot open canonical allocation blob") from exc
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise SourceMlxArtifactError("canonical allocation blob must be a regular file")
        if int(initial.st_size) != expected_bytes or expected_bytes != allocation.byte_length:
            raise SourceMlxArtifactError("canonical allocation blob size mismatch")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            payload.extend(block)
            digest.update(block)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(initial) != _identity(final):
        raise SourceMlxArtifactError("canonical allocation changed while being read")
    try:
        current = path.lstat()
    except OSError as exc:
        raise SourceMlxArtifactError("canonical allocation disappeared after reading") from exc
    if _identity(current) != _identity(final) or not stat.S_ISREG(current.st_mode):
        raise SourceMlxArtifactError("canonical allocation changed while being read")
    if digest.hexdigest() != expected_hash:
        raise SourceMlxArtifactError("canonical allocation hash mismatch")
    dtype = _TORCH_DTYPES[allocation.stored_dtype]
    count = math.prod(allocation.stored_shape)
    value = torch.frombuffer(payload, dtype=dtype, count=count).reshape(allocation.stored_shape)
    return value.clone().contiguous()


def _native_recipe(
    artifact: ComponentArtifact,
    *,
    architecture: str,
    source_dtype: str,
    config_sha256: str,
) -> dict[str, Any]:
    builder_sha256, _size, _identity_value = _hash_regular_file(Path(__file__).resolve())
    return {
        "schema": SOURCE_MLX_NATIVE_SCHEMA,
        "builder_abi": SOURCE_MLX_BUILDER_ABI,
        "builder_sha256": builder_sha256,
        "mapping_abi": _native_mapping_abi(architecture),
        "codec": _source_dtype_codec(source_dtype),
        "source_dtype": source_dtype,
        "shard_bytes": SOURCE_MLX_SHARD_BYTES,
        "source_artifact_id": artifact.artifact_id,
        "source_manifest_sha256": artifact.manifest_sha256,
        "source_ir_fingerprint": artifact.ir_bundle.fingerprint,
        "effective_config_sha256": config_sha256,
    }


def _q4_config(config: Mapping[str, Any]) -> dict[str, Any]:
    effective = json.loads(_canonical_json_bytes(dict(config)))
    quantization = {
        "bits": SOURCE_MLX_Q4_BITS,
        "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
        "mode": SOURCE_MLX_Q4_MODE,
    }
    effective["quantization"] = dict(quantization)
    effective["quantization_config"] = dict(quantization)
    effective.pop("quantize_activations", None)
    return effective


def _q4_expected_allocation_encoding(
    architecture: str,
    source_tensor: str,
    source_shape: Sequence[int],
) -> str:
    """Return the one legal encoding for a source allocation.

    Dense decoder support predates partial module quantization and intentionally keeps its
    original closed contract: every matrix must be a group-aligned ``.weight`` and every
    non-matrix must be a vector.  Mamba1 has authenticated non-module matrices (``A_log``), a
    depthwise convolution kernel, and a usually non-group-aligned ``dt_proj``.  MLX-LM's model
    quantizer replaces only modules with ``to_quantized`` whose input width is group aligned.
    In the canonical Mamba1 namespace that is exactly the rank-two, group-aligned ``.weight``
    subset.  All other owned allocations must remain source-exact; none may be silently dropped
    or packed as if it were a quantized module.
    """

    shape = tuple(int(value) for value in source_shape)
    if architecture == "mamba":
        eligible = (
            len(shape) == 2
            and source_tensor.endswith(".weight")
            and shape[1] % SOURCE_MLX_Q4_GROUP_SIZE == 0
        )
        return "mlx-affine-q4-g64" if eligible else "source-exact"

    if len(shape) == 2:
        if not source_tensor.endswith(".weight"):
            raise ValueError("two-dimensional q4 allocation is not a weight")
        if shape[1] % SOURCE_MLX_Q4_GROUP_SIZE:
            raise ValueError(
                f"q4 source width for {source_tensor!r} is not divisible by "
                f"{SOURCE_MLX_Q4_GROUP_SIZE}"
            )
        return "mlx-affine-q4-g64"
    if len(shape) != 1:
        raise ValueError("direct q4 target supports only matrix weights and vector auxiliaries")
    return "source-exact"


def _q4_native_recipe(
    artifact: ComponentArtifact,
    *,
    source_dtype: str,
    config_sha256: str,
) -> dict[str, Any]:
    import importlib.metadata

    builder_sha256, _size, _identity_value = _hash_regular_file(Path(__file__).resolve())
    return {
        "schema": SOURCE_MLX_Q4_NATIVE_SCHEMA,
        "builder_abi": SOURCE_MLX_Q4_BUILDER_ABI,
        "builder_sha256": builder_sha256,
        "mapping_abi": SOURCE_MLX_MAPPING_ABI,
        "codec": SOURCE_MLX_Q4_CODEC,
        "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
        "bits": SOURCE_MLX_Q4_BITS,
        "mode": SOURCE_MLX_Q4_MODE,
        "shard_bytes": SOURCE_MLX_SHARD_BYTES,
        "quantizer": "mlx.core.quantize",
        "quantizer_input_dtype": "bfloat16",
        "auxiliary_codec": f"safetensors-{source_dtype.lower()}-source-exact-v1",
        "error_reference": f"canonical-source-{source_dtype.lower()}-float32",
        "mlx_version": importlib.metadata.version("mlx"),
        "numerical_contract": SOURCE_MLX_Q4_NUMERICAL_CONTRACT,
        "source_dtype": source_dtype,
        "source_artifact_id": artifact.artifact_id,
        "source_manifest_sha256": artifact.manifest_sha256,
        "source_fingerprint": artifact.source.fingerprint,
        "source_ir_fingerprint": artifact.ir_bundle.fingerprint,
        "source_io_fingerprint": artifact.ir_bundle.io.fingerprint,
        "source_model_fingerprint": artifact.ir_bundle.model.fingerprint,
        "source_tokenizer_custody_sha256": _tokenizer_custody_sha256(artifact),
        "effective_config_sha256": config_sha256,
        "direct_from_canonical_source": True,
        "intermediate_qstore": False,
    }


@dataclass(frozen=True, slots=True)
class SourceMlxBuildRecord:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    shard_count: int
    shard_bytes: int
    verified_reopen: bool
    direct_from_canonical_source: bool = True
    native_runtime_candidate: bool = True
    production_runtime_eligible: bool = False
    schema_version: str = SOURCE_MLX_NATIVE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "build_key_sha256": self.build_key_sha256,
            "source_artifact_id": self.source_artifact_id,
            "shard_count": self.shard_count,
            "shard_bytes": self.shard_bytes,
            "verified_reopen": self.verified_reopen,
            "direct_from_canonical_source": self.direct_from_canonical_source,
            "native_runtime_candidate": self.native_runtime_candidate,
            "production_runtime_eligible": self.production_runtime_eligible,
        }


@dataclass(frozen=True, slots=True)
class SourceMlxQ4BuildRecord:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    shard_count: int
    shard_bytes: int
    verified_reopen: bool
    max_abs_error: float
    rmse: float
    direct_from_canonical_source: bool = True
    intermediate_qstore: bool = False
    approximate_quantized: bool = True
    native_runtime_candidate: bool = True
    production_runtime_eligible: bool = False
    schema_version: str = SOURCE_MLX_Q4_NATIVE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "build_key_sha256": self.build_key_sha256,
            "source_artifact_id": self.source_artifact_id,
            "shard_count": self.shard_count,
            "shard_bytes": self.shard_bytes,
            "verified_reopen": self.verified_reopen,
            "max_abs_error": self.max_abs_error,
            "rmse": self.rmse,
            "direct_from_canonical_source": self.direct_from_canonical_source,
            "intermediate_qstore": self.intermediate_qstore,
            "approximate_quantized": self.approximate_quantized,
            "native_runtime_candidate": self.native_runtime_candidate,
            "production_runtime_eligible": self.production_runtime_eligible,
        }


@dataclass(frozen=True, slots=True)
class _DirectAffineQ4:
    weight: Any
    scales: Any
    biases: Any
    max_abs_error: float
    sum_squared_error: float
    elements: int


def _quantize_source_affine_q4(tensor: torch.Tensor, *, mx: Any) -> _DirectAffineQ4:
    if tensor.ndim != 2:
        raise TypeError("direct affine q4 source tensor must be two-dimensional")
    rows, columns = (int(value) for value in tensor.shape)
    if columns % SOURCE_MLX_Q4_GROUP_SIZE:
        raise SourceMlxLoweringError(
            f"q4 source width {columns} is not divisible by {SOURCE_MLX_Q4_GROUP_SIZE}"
        )
    reference = mx.array(tensor.float().numpy()).astype(mx.float32)
    if not bool(torch.isfinite(tensor.float()).all().item()):
        raise SourceMlxLoweringError("canonical source tensor contains non-finite values")
    quantizer_input = reference.astype(mx.bfloat16)
    quantized = mx.quantize(
        quantizer_input,
        group_size=SOURCE_MLX_Q4_GROUP_SIZE,
        bits=SOURCE_MLX_Q4_BITS,
        mode=SOURCE_MLX_Q4_MODE,
    )
    if len(quantized) != 3:
        raise SourceMlxArtifactError("MLX affine q4 quantizer returned an invalid tuple")
    weight, scales, biases = quantized
    restored = mx.dequantize(
        weight,
        scales,
        biases,
        group_size=SOURCE_MLX_Q4_GROUP_SIZE,
        bits=SOURCE_MLX_Q4_BITS,
        mode=SOURCE_MLX_Q4_MODE,
    )
    error = restored.astype(mx.float32) - reference
    maximum = mx.max(mx.abs(error))
    squared = mx.sum(mx.square(error))
    mx.eval(weight, scales, biases, maximum, squared)
    max_abs_error = float(maximum.item())
    sum_squared_error = float(squared.item())
    if not math.isfinite(max_abs_error) or not math.isfinite(sum_squared_error):
        raise SourceMlxArtifactError("MLX affine q4 error evidence is non-finite")
    return _DirectAffineQ4(
        weight=weight,
        scales=scales,
        biases=biases,
        max_abs_error=max_abs_error,
        sum_squared_error=sum_squared_error,
        elements=rows * columns,
    )


def _source_auxiliary_array(tensor: torch.Tensor, source_dtype: str, *, mx: Any) -> Any:
    if not bool(torch.isfinite(tensor.float()).all().item()):
        raise SourceMlxLoweringError("canonical source tensor contains non-finite values")
    dtype = {
        "BF16": mx.bfloat16,
        "F16": mx.float16,
        "F32": mx.float32,
    }[source_dtype]
    value = mx.array(tensor.float().numpy()).astype(dtype)
    mx.eval(value)
    return value


class VerifiedSourceMlxArtifact:
    """Strict, immutable view of a direct-source native MLX artifact."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.is_dir():
            raise SourceMlxArtifactError("direct-source MLX artifact must be a real directory")
        self.path = self.path.resolve()
        self._directory_identity = _identity(self.path.lstat())
        try:
            manifest, manifest_identity, manifest_file_sha = _read_regular_json(
                self.path / "manifest.json"
            )
        except Exception as exc:
            raise SourceMlxArtifactError("cannot verify direct-source MLX manifest") from exc
        if manifest.get("schema") != SOURCE_MLX_NATIVE_SCHEMA:
            raise SourceMlxArtifactError("unsupported direct-source MLX artifact schema")
        if (
            manifest.get("status") != "native-lowered-unexecuted"
            or manifest.get("execution_certified") is not False
            or manifest.get("native_runtime_candidate") is not True
            or manifest.get("production_runtime_eligible") is not False
        ):
            raise SourceMlxArtifactError("direct-source MLX promotion boundary is malformed")
        declared = manifest.get("artifact_sha256")
        if not _is_lower_sha256(declared):
            raise SourceMlxArtifactError("direct-source MLX artifact identity is malformed")
        unhashed = dict(manifest)
        unhashed.pop("artifact_sha256", None)
        if declared != _sha256_bytes(_canonical_json_bytes(unhashed)):
            raise SourceMlxArtifactError("direct-source MLX artifact identity mismatch")
        recipe = manifest.get("recipe")
        if not isinstance(recipe, Mapping):
            raise SourceMlxArtifactError("direct-source MLX artifact has no recipe")
        if (
            recipe.get("schema") != SOURCE_MLX_NATIVE_SCHEMA
            or recipe.get("builder_abi") != SOURCE_MLX_BUILDER_ABI
            or recipe.get("mapping_abi") not in _SOURCE_MLX_RAW_MAPPING_ABIS
            or recipe.get("shard_bytes") != SOURCE_MLX_SHARD_BYTES
        ):
            raise SourceMlxArtifactError("direct-source MLX recipe ABI is unsupported")
        builder_sha256 = recipe.get("builder_sha256")
        if not _is_lower_sha256(builder_sha256):
            raise SourceMlxArtifactError("direct-source MLX builder identity is malformed")
        for field in (
            "source_artifact_id",
            "source_manifest_sha256",
            "source_ir_fingerprint",
            "effective_config_sha256",
        ):
            if not _is_lower_sha256(recipe.get(field)):
                raise SourceMlxArtifactError(f"direct-source MLX recipe {field!r} is malformed")
        build_key = _sha256_bytes(_canonical_json_bytes(recipe))
        if manifest.get("build_key_sha256") != build_key:
            raise SourceMlxArtifactError("direct-source MLX build key mismatch")
        source_dtype = str(recipe.get("source_dtype", ""))
        if recipe.get("codec") != _source_dtype_codec(source_dtype):
            raise SourceMlxArtifactError("direct-source MLX codec differs from source dtype")

        config_record = manifest.get("config")
        if not isinstance(config_record, Mapping) or config_record.get("filename") != "config.json":
            raise SourceMlxArtifactError("direct-source MLX config record is malformed")
        try:
            config, config_identity, config_file_sha = _read_regular_json(self.path / "config.json")
        except Exception as exc:
            raise SourceMlxArtifactError("cannot verify direct-source MLX config") from exc
        config_sha = _sha256_bytes(_canonical_json_bytes(config))
        if (
            config_file_sha != config_record.get("file_sha256")
            or _strict_positive_int(config_record.get("bytes"), "config bytes")
            != int(config_identity[3])
            or config_sha != config_record.get("semantic_sha256")
            or config_sha != recipe.get("effective_config_sha256")
        ):
            raise SourceMlxArtifactError("direct-source MLX config is not recipe-bound")

        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise SourceMlxArtifactError("direct-source MLX source record is malformed")
        for field in (
            "artifact_id",
            "manifest_sha256",
            "source_fingerprint",
            "ir_bundle_fingerprint",
            "io_fingerprint",
            "model_fingerprint",
        ):
            if not _is_lower_sha256(source.get(field)):
                raise SourceMlxArtifactError(f"direct-source MLX source {field!r} is malformed")
        architecture = str(source.get("architecture"))
        legacy_compatibility_omission = (
            recipe.get("builder_abi") == SOURCE_MLX_BUILDER_ABI
            and recipe.get("mapping_abi") == "mrun-dense-hf-source-name-to-mlx-lm-v1"
            and architecture in {"llama", "qwen2", "qwen3"}
            and "runtime_numerical_compatibility" not in source
        )
        if (
            source.get("artifact_id") != recipe.get("source_artifact_id")
            or source.get("manifest_sha256") != recipe.get("source_manifest_sha256")
            or source.get("ir_bundle_fingerprint") != recipe.get("source_ir_fingerprint")
            or source.get("direct_from_canonical_source") is not True
            or source.get("intermediate_qstore") is not False
            or type(source.get("tied_lexical_allocation")) is not bool
            or source.get("architecture") not in _ARCHITECTURES.values()
            or source.get("architecture_id") not in _ARCHITECTURES
            or _ARCHITECTURES[source["architecture_id"]] != source.get("architecture")
            or (
                not legacy_compatibility_omission
                and source.get("runtime_numerical_compatibility")
                != _native_semantic_compatibility(architecture)
            )
        ):
            raise SourceMlxArtifactError("direct-source MLX source lineage is inconsistent")
        if (
            type(config.get("tie_word_embeddings")) is not bool
            or config["tie_word_embeddings"] != source["tied_lexical_allocation"]
            or config.get("model_type") != source["architecture"]
        ):
            raise SourceMlxArtifactError("direct-source MLX config/source topology differs")

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise SourceMlxArtifactError("direct-source MLX artifact has no shards")
        identities = {
            "manifest.json": manifest_identity,
            "config.json": config_identity,
        }
        observed_names: set[str] = set()
        parameter_names: set[str] = set()
        allocation_ids: set[str] = set()
        observed_roles: set[str] = set()
        shard_bytes = 0
        parameter_bytes = 0
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise SourceMlxArtifactError("direct-source MLX shard record is malformed")
            filename = shard.get("filename")
            role = shard.get("role")
            if (
                type(filename) is not str
                or not re.fullmatch(r"model-[a-z_]+-[0-9]{5}\.safetensors", filename)
                or role not in {"body", "norm", "ingress", "egress", "lexical_shared"}
                or filename in observed_names
                or not _is_lower_sha256(shard.get("sha256"))
            ):
                raise SourceMlxArtifactError("direct-source MLX shard name/role is invalid")
            expected_shard_bytes = _strict_positive_int(
                shard.get("bytes"), f"direct-source MLX shard {filename!r} bytes"
            )
            try:
                digest, size, identity = _hash_regular_file(
                    self.path / filename,
                    expected_size=expected_shard_bytes,
                    expected_sha256=shard["sha256"],
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise SourceMlxArtifactError(
                    f"direct-source MLX shard hash/identity verification failed: {filename!r}"
                ) from exc
            if digest != shard.get("sha256"):
                raise SourceMlxArtifactError("direct-source MLX shard hash drifted")
            parameters = shard.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise SourceMlxArtifactError("direct-source MLX shard has no parameter inventory")
            if any(not isinstance(item, Mapping) for item in parameters):
                raise SourceMlxArtifactError("direct-source parameter record is malformed")
            from safetensors import safe_open

            with safe_open(self.path / filename, framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
                if set(keys) != {str(item.get("name")) for item in parameters}:
                    raise SourceMlxArtifactError("direct-source shard header differs from manifest")
                metadata = handle.metadata() or {}
                if (
                    metadata.get("format") != "mlx"
                    or metadata.get("mrun-codec") != recipe["codec"]
                    or metadata.get("mrun-role") != role
                    or metadata.get("mrun-source-artifact") != source["artifact_id"]
                ):
                    raise SourceMlxArtifactError(
                        "direct-source shard metadata is not lineage-bound"
                    )
                for item in parameters:
                    name = item.get("name")
                    allocation_id = item.get("source_allocation_id")
                    shape = item.get("shape")
                    if (
                        type(name) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(name) is None
                        or type(allocation_id) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(allocation_id) is None
                        or not _is_lower_sha256(item.get("source_blob_sha256"))
                        or not isinstance(shape, list)
                        or not shape
                        or any(
                            isinstance(dimension, bool)
                            or not isinstance(dimension, int)
                            or dimension <= 0
                            for dimension in shape
                        )
                    ):
                        raise SourceMlxArtifactError(
                            "direct-source parameter descriptor is malformed"
                        )
                    if name in parameter_names or allocation_id in allocation_ids:
                        raise SourceMlxArtifactError("direct-source MLX parameter is duplicated")
                    value = handle.get_tensor(name)
                    dtype = item.get("dtype")
                    if (
                        dtype != source_dtype
                        or value.dtype != _TORCH_DTYPES[source_dtype]
                        or list(value.shape) != shape
                    ):
                        raise SourceMlxArtifactError("direct-source parameter descriptor drifted")
                    parameter_names.add(name)
                    allocation_ids.add(allocation_id)
                    parameter_bytes += int(value.numel()) * int(value.element_size())
            observed_names.add(filename)
            observed_roles.add(str(role))
            identities[filename] = identity
            shard_bytes += size
        expected_roles = (
            {"body", "norm", "lexical_shared"}
            if source["tied_lexical_allocation"]
            else {"body", "norm", "ingress", "egress"}
        )
        if observed_roles != expected_roles:
            raise SourceMlxArtifactError("direct-source MLX physical role topology is incomplete")
        coverage = manifest.get("coverage")
        allocation_count = len(allocation_ids)
        if (
            not isinstance(coverage, Mapping)
            or coverage.get("source_dtype") != source_dtype
            or coverage.get("all_source_allocations_emitted_once") is not True
            or coverage.get("physical_aliases_not_duplicated") is not True
            or coverage.get("source_allocation_count") != allocation_count
            or coverage.get("emitted_parameter_count") != allocation_count
            or coverage.get("source_allocation_bytes") != parameter_bytes
        ):
            raise SourceMlxArtifactError("direct-source MLX coverage claim is inconsistent")
        if {entry.name for entry in self.path.iterdir()} != set(identities):
            raise SourceMlxArtifactError("direct-source MLX artifact has undeclared files")
        if manifest.get("manifest_file_sha256") not in {None, manifest_file_sha}:
            raise SourceMlxArtifactError("direct-source manifest self-file hash is invalid")
        self.manifest = dict(manifest)
        self.config = config
        self.source = dict(source)
        self.artifact_sha256 = str(declared)
        self.build_key_sha256 = build_key
        self.codec = str(recipe["codec"])
        self.bits = _DTYPE_BITS[source_dtype]
        self.source_dtype = source_dtype
        self.shard_bytes = shard_bytes
        self._identities = identities

    def assert_unchanged(self) -> None:
        if _identity(self.path.lstat()) != self._directory_identity:
            raise SourceMlxArtifactError("direct-source MLX directory identity changed")
        if {entry.name for entry in self.path.iterdir()} != set(self._identities):
            raise SourceMlxArtifactError("direct-source MLX file inventory changed")
        for filename, identity in self._identities.items():
            if _identity((self.path / filename).lstat()) != identity:
                raise SourceMlxArtifactError(f"direct-source MLX file changed: {filename}")


def build_source_mlx_artifact(
    source_artifact: ComponentArtifact | str | Path,
    output_root: str | Path,
) -> SourceMlxBuildRecord:
    """Lower a verified canonical artifact directly into role-separated MLX safetensors."""

    try:
        reference = lower_component_artifact_to_reference(source_artifact)
    except ReferenceLoweringError as exc:
        raise SourceMlxLoweringError(str(exc), details=exc.details) from exc
    artifact = reference.artifact
    config, architecture, tied = _validated_config(artifact)
    allocations = artifact.ir_bundle.physical_weights.allocations
    dtypes = {item.stored_dtype for item in allocations}
    if len(dtypes) != 1:
        raise SourceMlxLoweringError("direct MLX target requires one executable source dtype")
    source_dtype = next(iter(dtypes))
    _source_dtype_codec(source_dtype)
    if any(".rotary_emb." in item.source_tensor for item in allocations):
        raise SourceMlxLoweringError(
            "serialized RoPE tensors are not yet registered in the direct MLX target"
        )

    views_by_allocation: dict[str, list[str]] = defaultdict(list)
    for view in artifact.ir_bundle.physical_weights.views:
        views_by_allocation[view.allocation_id].append(view.logical_name)
    allocation_roles = {
        allocation.allocation_id: _parameter_role(views_by_allocation[allocation.allocation_id])
        for allocation in allocations
    }
    if tied and not any(role == "lexical_shared" for role in allocation_roles.values()):
        raise SourceMlxLoweringError("tied lexical allocation did not lower as shared")

    config_sha = _sha256_bytes(_canonical_json_bytes(config))
    recipe = _native_recipe(
        artifact,
        architecture=architecture,
        source_dtype=source_dtype,
        config_sha256=config_sha,
    )
    build_key = _sha256_bytes(_canonical_json_bytes(recipe))
    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise SourceMlxArtifactError("direct MLX output root must be a real directory")
    output_root = output_root.resolve()
    target = output_root / f"{_safe_model_slug(artifact.source.source_id)}-{build_key[:16]}"

    with _build_lock(build_key):
        if target.exists():
            verified = VerifiedSourceMlxArtifact(target)
            if verified.build_key_sha256 != build_key:
                raise SourceMlxArtifactError("existing direct MLX target has a foreign build key")
            return SourceMlxBuildRecord(
                path=target,
                artifact_sha256=verified.artifact_sha256,
                build_key_sha256=build_key,
                source_artifact_id=artifact.artifact_id,
                shard_count=len(verified.manifest["shards"]),
                shard_bytes=verified.shard_bytes,
                verified_reopen=True,
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=output_root))
        try:
            config_raw = _canonical_json_bytes(config)
            config_file_sha, config_bytes = _write_durable(staging / "config.json", config_raw)
            allocation_records = {
                str(item["allocation_id"]): item for item in artifact.manifest["allocations"]
            }
            shards: list[dict[str, Any]] = []
            by_role: dict[str, list[Any]] = defaultdict(list)
            for allocation in allocations:
                if not _SAFE_TENSOR_NAME.fullmatch(allocation.source_tensor):
                    raise SourceMlxLoweringError("source tensor name is unsafe for MLX safetensors")
                by_role[allocation_roles[allocation.allocation_id]].append(allocation)

            for role in sorted(by_role):
                shard_index = 0
                tensors: dict[str, torch.Tensor] = {}
                parameters: list[dict[str, Any]] = []
                pending_bytes = 0

                def flush(role_value: str) -> None:
                    nonlocal shard_index, tensors, parameters, pending_bytes
                    if not tensors:
                        return
                    shard_index += 1
                    filename = f"model-{role_value}-{shard_index:05d}.safetensors"
                    path = staging / filename
                    save_file(
                        dict(sorted(tensors.items())),
                        path,
                        metadata={
                            "format": "mlx",
                            "mrun-codec": _source_dtype_codec(source_dtype),
                            "mrun-role": role_value,
                            "mrun-source-artifact": artifact.artifact_id,
                        },
                    )
                    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    digest, size, _identity_value = _hash_regular_file(path)
                    shards.append(
                        {
                            "filename": filename,
                            "role": role_value,
                            "bytes": size,
                            "sha256": digest,
                            "parameters": sorted(parameters, key=lambda item: item["name"]),
                        }
                    )
                    tensors = {}
                    parameters = []
                    pending_bytes = 0

                for allocation in sorted(by_role[role], key=lambda item: item.source_tensor):
                    if tensors and pending_bytes + allocation.byte_length > SOURCE_MLX_SHARD_BYTES:
                        flush(role)
                    record = allocation_records.get(allocation.allocation_id)
                    if not isinstance(record, Mapping):
                        raise SourceMlxArtifactError("canonical allocation manifest is incomplete")
                    tensor = _read_allocation(artifact, allocation, record)
                    if architecture == "gpt2" and allocation.source_tensor.endswith(".attn.bias"):
                        positions = artifact.ir_bundle.model.dimensions.max_position_embeddings
                        expected_mask = torch.tril(
                            torch.ones((positions, positions), dtype=tensor.dtype)
                        )[None, None]
                        if not torch.equal(tensor, expected_mask):
                            raise SourceMlxLoweringError(
                                "mlx-lm reconstructs GPT-2 causal masks; source mask is not "
                                "canonical"
                            )
                    native_name = _native_parameter_name(architecture, allocation.source_tensor)
                    if native_name in tensors:
                        raise SourceMlxLoweringError("native parameter mapping is not one-to-one")
                    tensors[native_name] = tensor
                    parameters.append(
                        {
                            "name": native_name,
                            "source_tensor": allocation.source_tensor,
                            "dtype": allocation.stored_dtype,
                            "shape": list(allocation.stored_shape),
                            "source_allocation_id": allocation.allocation_id,
                            "source_blob_sha256": record["blob"]["sha256"],
                        }
                    )
                    pending_bytes += allocation.byte_length
                flush(role)

            source = {
                "artifact_schema": artifact.manifest["schema_version"],
                "artifact_id": artifact.artifact_id,
                "manifest_sha256": artifact.manifest_sha256,
                "source_fingerprint": artifact.source.fingerprint,
                "ir_bundle_fingerprint": artifact.ir_bundle.fingerprint,
                "io_fingerprint": artifact.ir_bundle.io.fingerprint,
                "model_fingerprint": artifact.ir_bundle.model.fingerprint,
                "architecture_id": artifact.ir_bundle.model.architecture_id,
                "architecture": architecture,
                "runtime_numerical_compatibility": _native_semantic_compatibility(architecture),
                "tied_lexical_allocation": tied,
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
            }
            manifest: dict[str, Any] = {
                "schema": SOURCE_MLX_NATIVE_SCHEMA,
                "status": "native-lowered-unexecuted",
                "build_key_sha256": build_key,
                "recipe": recipe,
                "source": source,
                "config": {
                    "filename": "config.json",
                    "bytes": config_bytes,
                    "file_sha256": config_file_sha,
                    "semantic_sha256": config_sha,
                },
                "shards": sorted(shards, key=lambda item: item["filename"]),
                "coverage": {
                    "source_allocation_count": len(allocations),
                    "emitted_parameter_count": sum(len(item["parameters"]) for item in shards),
                    "source_allocation_bytes": sum(item.byte_length for item in allocations),
                    "source_dtype": source_dtype,
                    "all_source_allocations_emitted_once": True,
                    "physical_aliases_not_duplicated": True,
                },
                "execution_certified": False,
                "native_runtime_candidate": True,
                "production_runtime_eligible": False,
            }
            manifest["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
            _write_durable(staging / "manifest.json", _canonical_json_bytes(manifest))
            _fsync_directory(staging)
            os.replace(staging, target)
            _fsync_directory(output_root)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    verified = VerifiedSourceMlxArtifact(target)
    return SourceMlxBuildRecord(
        path=target,
        artifact_sha256=verified.artifact_sha256,
        build_key_sha256=build_key,
        source_artifact_id=artifact.artifact_id,
        shard_count=len(verified.manifest["shards"]),
        shard_bytes=verified.shard_bytes,
        verified_reopen=True,
    )


class VerifiedSourceMlxQ4Artifact:
    """Strict immutable view of a direct canonical-source affine-q4 MLX artifact."""

    _PARAMETER_FIELDS = {
        "name",
        "dtype",
        "shape",
        "part",
        "encoding",
        "source_allocation_id",
        "source_blob_sha256",
        "source_tensor",
        "source_dtype",
        "source_shape",
        "source_byte_count",
        "logical_names",
    }

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.is_dir():
            raise SourceMlxArtifactError("direct-source q4 artifact must be a real directory")
        self.path = self.path.resolve()
        self._directory_identity = _identity(self.path.lstat())
        try:
            manifest, manifest_identity, _manifest_file_sha = _read_regular_json(
                self.path / "manifest.json"
            )
        except Exception as exc:
            raise SourceMlxArtifactError("cannot verify direct-source q4 manifest") from exc
        if manifest.get("schema") != SOURCE_MLX_Q4_NATIVE_SCHEMA:
            raise SourceMlxArtifactError("unsupported direct-source q4 artifact schema")
        if (
            manifest.get("status") != "native-lowered-approximate-unexecuted"
            or manifest.get("execution_certified") is not False
            or manifest.get("native_runtime_candidate") is not True
            or manifest.get("production_runtime_eligible") is not False
            or manifest.get("approximate_quantized") is not True
            or manifest.get("numerical_contract") != SOURCE_MLX_Q4_NUMERICAL_CONTRACT
        ):
            raise SourceMlxArtifactError("direct-source q4 promotion boundary is malformed")
        declared = manifest.get("artifact_sha256")
        if not _is_lower_sha256(declared):
            raise SourceMlxArtifactError("direct-source q4 artifact identity is malformed")
        unhashed = dict(manifest)
        unhashed.pop("artifact_sha256", None)
        if declared != _sha256_bytes(_canonical_json_bytes(unhashed)):
            raise SourceMlxArtifactError("direct-source q4 artifact identity mismatch")

        recipe = manifest.get("recipe")
        if not isinstance(recipe, Mapping):
            raise SourceMlxArtifactError("direct-source q4 artifact has no recipe")
        expected_recipe = {
            "schema": SOURCE_MLX_Q4_NATIVE_SCHEMA,
            "builder_abi": SOURCE_MLX_Q4_BUILDER_ABI,
            "mapping_abi": SOURCE_MLX_MAPPING_ABI,
            "codec": SOURCE_MLX_Q4_CODEC,
            "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
            "bits": SOURCE_MLX_Q4_BITS,
            "mode": SOURCE_MLX_Q4_MODE,
            "shard_bytes": SOURCE_MLX_SHARD_BYTES,
            "quantizer": "mlx.core.quantize",
            "quantizer_input_dtype": "bfloat16",
            "numerical_contract": SOURCE_MLX_Q4_NUMERICAL_CONTRACT,
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
        }
        if any(recipe.get(key) != value for key, value in expected_recipe.items()):
            raise SourceMlxArtifactError("direct-source q4 recipe ABI is unsupported")
        source_dtype = str(recipe.get("source_dtype", ""))
        if source_dtype not in {"BF16", "F16", "F32"}:
            raise SourceMlxArtifactError("direct-source q4 source dtype is unsupported")
        if (
            recipe.get("auxiliary_codec") != f"safetensors-{source_dtype.lower()}-source-exact-v1"
            or recipe.get("error_reference") != f"canonical-source-{source_dtype.lower()}-float32"
            or type(recipe.get("mlx_version")) is not str
            or not recipe.get("mlx_version")
            or not _is_lower_sha256(recipe.get("builder_sha256"))
        ):
            raise SourceMlxArtifactError("direct-source q4 quantizer contract is invalid")
        lineage_recipe_fields = (
            "source_artifact_id",
            "source_manifest_sha256",
            "source_fingerprint",
            "source_ir_fingerprint",
            "source_io_fingerprint",
            "source_model_fingerprint",
            "source_tokenizer_custody_sha256",
            "effective_config_sha256",
        )
        for field in lineage_recipe_fields:
            if not _is_lower_sha256(recipe.get(field)):
                raise SourceMlxArtifactError(f"direct-source q4 recipe {field!r} is malformed")
        build_key = _sha256_bytes(_canonical_json_bytes(recipe))
        if manifest.get("build_key_sha256") != build_key:
            raise SourceMlxArtifactError("direct-source q4 build key mismatch")

        config_record = manifest.get("config")
        if not isinstance(config_record, Mapping) or config_record.get("filename") != "config.json":
            raise SourceMlxArtifactError("direct-source q4 config record is malformed")
        try:
            config, config_identity, config_file_sha = _read_regular_json(self.path / "config.json")
        except Exception as exc:
            raise SourceMlxArtifactError("cannot verify direct-source q4 config") from exc
        config_sha = _sha256_bytes(_canonical_json_bytes(config))
        if (
            config_file_sha != config_record.get("file_sha256")
            or _strict_positive_int(config_record.get("bytes"), "q4 config bytes")
            != int(config_identity[3])
            or config_sha != config_record.get("semantic_sha256")
            or config_sha != recipe.get("effective_config_sha256")
        ):
            raise SourceMlxArtifactError("direct-source q4 config is not recipe-bound")
        quantization_config = {
            "bits": SOURCE_MLX_Q4_BITS,
            "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
            "mode": SOURCE_MLX_Q4_MODE,
        }
        if (
            config.get("quantization") != quantization_config
            or config.get("quantization_config") != quantization_config
        ):
            raise SourceMlxArtifactError("direct-source q4 config quantization is invalid")

        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise SourceMlxArtifactError("direct-source q4 source record is malformed")
        source_hash_fields = (
            "artifact_id",
            "manifest_sha256",
            "source_fingerprint",
            "ir_bundle_fingerprint",
            "io_fingerprint",
            "model_fingerprint",
            "tokenizer_custody_sha256",
        )
        for field in source_hash_fields:
            if not _is_lower_sha256(source.get(field)):
                raise SourceMlxArtifactError(f"direct-source q4 source {field!r} is malformed")
        source_recipe_pairs = {
            "artifact_id": "source_artifact_id",
            "manifest_sha256": "source_manifest_sha256",
            "source_fingerprint": "source_fingerprint",
            "ir_bundle_fingerprint": "source_ir_fingerprint",
            "io_fingerprint": "source_io_fingerprint",
            "model_fingerprint": "source_model_fingerprint",
            "tokenizer_custody_sha256": "source_tokenizer_custody_sha256",
        }
        if any(
            source.get(left) != recipe.get(right) for left, right in source_recipe_pairs.items()
        ):
            raise SourceMlxArtifactError("direct-source q4 source lineage is not recipe-bound")
        if (
            source.get("direct_from_canonical_source") is not True
            or source.get("intermediate_qstore") is not False
            or source.get("source_dtype") != source_dtype
            or type(source.get("tied_lexical_allocation")) is not bool
            or source.get("architecture_id") not in _ARCHITECTURES
            or source.get("architecture") != _ARCHITECTURES[source["architecture_id"]]
            or config.get("model_type") != source.get("architecture")
            or config.get("tie_word_embeddings") != source.get("tied_lexical_allocation")
        ):
            raise SourceMlxArtifactError("direct-source q4 topology lineage is inconsistent")

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise SourceMlxArtifactError("direct-source q4 artifact has no shards")
        identities = {"manifest.json": manifest_identity, "config.json": config_identity}
        observed_filenames: set[str] = set()
        observed_parameter_names: set[str] = set()
        observed_roles: set[str] = set()
        allocation_parameters: dict[str, list[tuple[Mapping[str, Any], torch.Tensor, str]]] = (
            defaultdict(list)
        )
        shard_bytes = 0
        emitted_parameter_bytes = 0
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise SourceMlxArtifactError("direct-source q4 shard record is malformed")
            filename = shard.get("filename")
            role = shard.get("role")
            if (
                type(filename) is not str
                or re.fullmatch(r"model-[a-z_]+-[0-9]{5}\.safetensors", filename) is None
                or filename in observed_filenames
                or role not in {"body", "norm", "ingress", "egress", "lexical_shared"}
                or not _is_lower_sha256(shard.get("sha256"))
            ):
                raise SourceMlxArtifactError("direct-source q4 shard name/role is invalid")
            expected_size = _strict_positive_int(shard.get("bytes"), f"q4 shard {filename} bytes")
            try:
                digest, size, identity = _hash_regular_file(
                    self.path / filename,
                    expected_size=expected_size,
                    expected_sha256=str(shard["sha256"]),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise SourceMlxArtifactError(
                    f"direct-source q4 shard hash verification failed: {filename!r}"
                ) from exc
            if digest != shard.get("sha256"):
                raise SourceMlxArtifactError("direct-source q4 shard hash drifted")
            parameters = shard.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise SourceMlxArtifactError("direct-source q4 shard has no parameters")
            if any(not isinstance(item, Mapping) for item in parameters):
                raise SourceMlxArtifactError("direct-source q4 parameter record is malformed")
            from safetensors import safe_open

            with safe_open(self.path / filename, framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
                declared_names = [str(item.get("name")) for item in parameters]
                if len(set(declared_names)) != len(declared_names) or set(keys) != set(
                    declared_names
                ):
                    raise SourceMlxArtifactError("direct-source q4 shard header differs")
                metadata = handle.metadata() or {}
                if (
                    metadata.get("format") != "mlx"
                    or metadata.get("mrun-codec") != SOURCE_MLX_Q4_CODEC
                    or metadata.get("mrun-role") != role
                    or metadata.get("mrun-source-artifact") != source["artifact_id"]
                ):
                    raise SourceMlxArtifactError("direct-source q4 shard metadata drifted")
                for item in parameters:
                    if set(item) != self._PARAMETER_FIELDS:
                        raise SourceMlxArtifactError(
                            "direct-source q4 parameter inventory is not exact"
                        )
                    name = item["name"]
                    allocation_id = item["source_allocation_id"]
                    logical_names = item["logical_names"]
                    source_shape = item["source_shape"]
                    shape = item["shape"]
                    if (
                        type(name) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(name) is None
                        or name in observed_parameter_names
                        or type(allocation_id) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(allocation_id) is None
                        or not _is_lower_sha256(item["source_blob_sha256"])
                        or type(item["source_tensor"]) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(item["source_tensor"]) is None
                        or item["source_dtype"] != source_dtype
                        or not isinstance(logical_names, list)
                        or logical_names != sorted(set(logical_names))
                        or not logical_names
                        or any(type(value) is not str or not value for value in logical_names)
                        or not isinstance(source_shape, list)
                        or not source_shape
                        or any(type(value) is not int or value <= 0 for value in source_shape)
                        or not isinstance(shape, list)
                        or not shape
                        or any(type(value) is not int or value <= 0 for value in shape)
                        or _strict_positive_int(
                            item["source_byte_count"], "q4 source parameter bytes"
                        )
                        != math.prod(source_shape) * (_DTYPE_BITS[source_dtype] // 8)
                    ):
                        raise SourceMlxArtifactError(
                            "direct-source q4 parameter descriptor is malformed"
                        )
                    if _parameter_role(logical_names) != role:
                        raise SourceMlxArtifactError(
                            "direct-source q4 parameter crossed its physical role"
                        )
                    value = handle.get_tensor(name)
                    dtype_map = {
                        "U32": torch.uint32,
                        "BF16": torch.bfloat16,
                        "F16": torch.float16,
                        "F32": torch.float32,
                    }
                    dtype = item["dtype"]
                    if (
                        dtype not in dtype_map
                        or value.dtype != dtype_map[dtype]
                        or list(value.shape) != shape
                    ):
                        raise SourceMlxArtifactError(
                            "direct-source q4 parameter shape/dtype drifted"
                        )
                    observed_parameter_names.add(name)
                    emitted_parameter_bytes += int(value.numel()) * int(value.element_size())
                    allocation_parameters[allocation_id].append((item, value, str(role)))
            observed_filenames.add(filename)
            observed_roles.add(str(role))
            identities[filename] = identity
            shard_bytes += size

        q4_allocations: set[str] = set()
        auxiliary_allocations: set[str] = set()
        source_allocation_bytes = 0
        source_weight_elements = 0
        allocation_facts: dict[str, dict[str, Any]] = {}
        for allocation_id, entries in allocation_parameters.items():
            first = entries[0][0]
            lineage_fields = (
                "source_blob_sha256",
                "source_tensor",
                "source_dtype",
                "source_shape",
                "source_byte_count",
                "logical_names",
                "encoding",
            )
            if any(
                _canonical_json_bytes(entry[0].get(field))
                != _canonical_json_bytes(first.get(field))
                for entry in entries[1:]
                for field in lineage_fields
            ):
                raise SourceMlxArtifactError("direct-source q4 allocation descriptors disagree")
            source_shape = [int(value) for value in first["source_shape"]]
            source_tensor = str(first["source_tensor"])
            logical_names = list(first["logical_names"])
            source_allocation_bytes += int(first["source_byte_count"])
            allocation_facts[allocation_id] = {
                "source_tensor": source_tensor,
                "logical_names": logical_names,
                "source_shape": source_shape,
            }
            try:
                expected_encoding = _q4_expected_allocation_encoding(
                    str(source["architecture"]),
                    source_tensor,
                    source_shape,
                )
            except ValueError as exc:
                raise SourceMlxArtifactError(
                    "direct-source q4 allocation is outside its architecture policy"
                ) from exc
            if first["encoding"] != expected_encoding:
                raise SourceMlxArtifactError(
                    "direct-source q4 allocation encoding violates its architecture policy"
                )
            if first["encoding"] == "mlx-affine-q4-g64":
                rows, columns = source_shape
                base = source_tensor[: -len(".weight")]
                expected = {
                    f"{base}.weight": ("weight", "U32", [rows, columns // 8]),
                    f"{base}.scales": (
                        "scales",
                        "BF16",
                        [rows, columns // SOURCE_MLX_Q4_GROUP_SIZE],
                    ),
                    f"{base}.biases": (
                        "biases",
                        "BF16",
                        [rows, columns // SOURCE_MLX_Q4_GROUP_SIZE],
                    ),
                }
                observed = {
                    str(item["name"]): (item["part"], item["dtype"], item["shape"])
                    for item, _value, _role in entries
                }
                if observed != expected or len(entries) != 3:
                    raise SourceMlxArtifactError("direct-source q4 packed triple is incomplete")
                q4_allocations.add(allocation_id)
                source_weight_elements += rows * columns
            elif first["encoding"] == "source-exact":
                if len(entries) != 1:
                    raise SourceMlxArtifactError("source-exact auxiliary allocation is malformed")
                item = entries[0][0]
                if (
                    item["name"] != source_tensor
                    or item["part"] != "source_exact"
                    or item["dtype"] != source_dtype
                    or item["shape"] != source_shape
                ):
                    raise SourceMlxArtifactError("source-exact auxiliary tensor drifted")
                auxiliary_allocations.add(allocation_id)
            else:
                raise SourceMlxArtifactError("direct-source q4 allocation encoding is unknown")

        expected_roles = (
            {"body", "norm", "lexical_shared"}
            if source["tied_lexical_allocation"]
            else {"body", "norm", "ingress", "egress"}
        )
        if observed_roles != expected_roles:
            raise SourceMlxArtifactError("direct-source q4 physical roles are incomplete")
        coverage = manifest.get("coverage")
        if (
            not isinstance(coverage, Mapping)
            or coverage.get("source_dtype") != source_dtype
            or coverage.get("all_source_allocations_emitted_once") is not True
            or coverage.get("physical_aliases_not_duplicated") is not True
            or coverage.get("auxiliary_source_exact") is not True
            or coverage.get("source_allocation_count") != len(allocation_parameters)
            or coverage.get("q4_allocation_count") != len(q4_allocations)
            or coverage.get("auxiliary_allocation_count") != len(auxiliary_allocations)
            or coverage.get("emitted_parameter_count") != len(observed_parameter_names)
            or coverage.get("source_allocation_bytes") != source_allocation_bytes
            or coverage.get("emitted_parameter_bytes") != emitted_parameter_bytes
        ):
            raise SourceMlxArtifactError("direct-source q4 coverage claim is inconsistent")

        quantization = manifest.get("quantization")
        if not isinstance(quantization, Mapping):
            raise SourceMlxArtifactError("direct-source q4 has no error evidence")
        blocks = quantization.get("blocks")
        if not isinstance(blocks, list) or any(not isinstance(item, Mapping) for item in blocks):
            raise SourceMlxArtifactError("direct-source q4 block evidence is malformed")
        block_ids: set[str] = set()
        total_squared_error = 0.0
        total_elements = 0
        maximum_error = 0.0
        for block in blocks:
            allocation_id = block.get("source_allocation_id")
            facts = allocation_facts.get(str(allocation_id))
            elements = block.get("elements")
            max_abs = block.get("max_abs_error")
            squared = block.get("sum_squared_error")
            rmse = block.get("rmse")
            if (
                allocation_id not in q4_allocations
                or allocation_id in block_ids
                or facts is None
                or block.get("source_tensor") != facts["source_tensor"]
                or block.get("logical_names") != facts["logical_names"]
                or type(elements) is not int
                or elements <= 0
                or type(max_abs) not in {int, float}
                or type(squared) not in {int, float}
                or type(rmse) not in {int, float}
                or not all(math.isfinite(float(value)) for value in (max_abs, squared, rmse))
                or float(max_abs) < 0
                or float(squared) < 0
                or float(rmse) < 0
                or not math.isclose(
                    float(rmse), math.sqrt(float(squared) / elements), rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise SourceMlxArtifactError("direct-source q4 block evidence is inconsistent")
            block_ids.add(str(allocation_id))
            total_elements += elements
            total_squared_error += float(squared)
            maximum_error = max(maximum_error, float(max_abs))
        aggregate_rmse = math.sqrt(total_squared_error / total_elements) if total_elements else -1.0
        if (
            block_ids != q4_allocations
            or quantization.get("source_codec")
            != f"safetensors-{source_dtype.lower()}-canonical-source-v1"
            or quantization.get("quantizer") != "mlx.core.quantize"
            or quantization.get("q4_allocation_count") != len(q4_allocations)
            or quantization.get("elements") != source_weight_elements
            or total_elements != source_weight_elements
            or quantization.get("auxiliary_source_exact") is not True
            or quantization.get("auxiliary_allocation_count") != len(auxiliary_allocations)
            or not math.isclose(
                float(quantization.get("max_abs_error", -1.0)),
                maximum_error,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(quantization.get("rmse", -1.0)),
                aggregate_rmse,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise SourceMlxArtifactError("direct-source q4 aggregate evidence is inconsistent")
        if {entry.name for entry in self.path.iterdir()} != set(identities):
            raise SourceMlxArtifactError("direct-source q4 artifact has undeclared files")

        self.manifest = dict(manifest)
        self.config = config
        self.source = dict(source)
        self.artifact_sha256 = str(declared)
        self.build_key_sha256 = build_key
        self.codec = SOURCE_MLX_Q4_CODEC
        self.bits = SOURCE_MLX_Q4_BITS
        self.source_dtype = source_dtype
        self.shard_bytes = shard_bytes
        self.max_abs_error = maximum_error
        self.rmse = aggregate_rmse
        self._identities = identities

    def assert_unchanged(self) -> None:
        if _identity(self.path.lstat()) != self._directory_identity:
            raise SourceMlxArtifactError("direct-source q4 directory identity changed")
        if {entry.name for entry in self.path.iterdir()} != set(self._identities):
            raise SourceMlxArtifactError("direct-source q4 file inventory changed")
        for filename, identity in self._identities.items():
            if _identity((self.path / filename).lstat()) != identity:
                raise SourceMlxArtifactError(f"direct-source q4 file changed: {filename}")


def build_source_mlx_q4_artifact(
    source_artifact: ComponentArtifact | str | Path,
    output_root: str | Path,
) -> SourceMlxQ4BuildRecord:
    """Lower canonical float allocations directly to an MLX affine-q4 artifact."""

    try:
        reference = lower_component_artifact_to_reference(source_artifact)
    except ReferenceLoweringError as exc:
        raise SourceMlxLoweringError(str(exc), details=exc.details) from exc
    artifact = reference.artifact
    source_config, architecture, tied = _validated_config(artifact)
    if architecture not in _SOURCE_MLX_Q4_ARCHITECTURES:
        raise SourceMlxLoweringError(
            "direct q4 lowering has no certified mapping for this architecture"
        )
    allocations = artifact.ir_bundle.physical_weights.allocations
    dtypes = {item.stored_dtype for item in allocations}
    if len(dtypes) != 1:
        raise SourceMlxLoweringError("direct q4 target requires one canonical source dtype")
    source_dtype = _source_q4_dtype(next(iter(dtypes)))
    if any(".rotary_emb." in item.source_tensor for item in allocations):
        raise SourceMlxLoweringError(
            "serialized RoPE tensors are not registered in the direct q4 target"
        )

    views_by_allocation: dict[str, list[str]] = defaultdict(list)
    for view in artifact.ir_bundle.physical_weights.views:
        views_by_allocation[view.allocation_id].append(view.logical_name)
    for allocation in allocations:
        logical_names = views_by_allocation.get(allocation.allocation_id, [])
        if not logical_names:
            raise SourceMlxLoweringError("canonical allocation has no logical views")
        try:
            _q4_expected_allocation_encoding(
                architecture,
                allocation.source_tensor,
                allocation.stored_shape,
            )
        except ValueError as exc:
            raise SourceMlxLoweringError(str(exc)) from exc
    allocation_roles = {
        allocation.allocation_id: _parameter_role(views_by_allocation[allocation.allocation_id])
        for allocation in allocations
    }
    if tied and not any(role == "lexical_shared" for role in allocation_roles.values()):
        raise SourceMlxLoweringError("tied lexical allocation did not remain physically shared")

    config = _q4_config(source_config)
    config_sha = _sha256_bytes(_canonical_json_bytes(config))
    try:
        recipe = _q4_native_recipe(
            artifact,
            source_dtype=source_dtype,
            config_sha256=config_sha,
        )
        import mlx.core as mx
    except ImportError as exc:
        raise SourceMlxArtifactError("mlx is required for direct affine q4 lowering") from exc
    build_key = _sha256_bytes(_canonical_json_bytes(recipe))
    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise SourceMlxArtifactError("direct q4 output root must be a real directory")
    output_root = output_root.resolve()
    target = output_root / (f"{_safe_model_slug(artifact.source.source_id)}-q4-{build_key[:16]}")

    with _build_lock(build_key):
        if target.exists() or target.is_symlink():
            verified = VerifiedSourceMlxQ4Artifact(target)
            if verified.build_key_sha256 != build_key:
                raise SourceMlxArtifactError("existing direct q4 target has a foreign build key")
            return SourceMlxQ4BuildRecord(
                path=target,
                artifact_sha256=verified.artifact_sha256,
                build_key_sha256=build_key,
                source_artifact_id=artifact.artifact_id,
                shard_count=len(verified.manifest["shards"]),
                shard_bytes=verified.shard_bytes,
                verified_reopen=True,
                max_abs_error=verified.max_abs_error,
                rmse=verified.rmse,
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=output_root))
        try:
            config_file_sha, config_bytes = _write_durable(
                staging / "config.json", _canonical_json_bytes(config)
            )
            allocation_records = {
                str(item["allocation_id"]): item for item in artifact.manifest["allocations"]
            }
            by_role: dict[str, list[Any]] = defaultdict(list)
            for allocation in allocations:
                if _SAFE_TENSOR_NAME.fullmatch(allocation.source_tensor) is None:
                    raise SourceMlxLoweringError("source tensor name is unsafe for safetensors")
                by_role[allocation_roles[allocation.allocation_id]].append(allocation)

            shards: list[dict[str, Any]] = []
            error_records: list[dict[str, Any]] = []
            total_squared_error = 0.0
            total_elements = 0
            maximum_error = 0.0
            q4_allocation_count = 0
            auxiliary_allocation_count = 0
            emitted_parameter_bytes = 0
            for role in sorted(by_role):
                shard_index = 0
                arrays: dict[str, Any] = {}
                parameters: list[dict[str, Any]] = []
                pending_bytes = 0

                def flush(role_value: str = role) -> None:
                    nonlocal shard_index, arrays, parameters, pending_bytes
                    if not arrays:
                        return
                    shard_index += 1
                    filename = f"model-{role_value}-{shard_index:05d}.safetensors"
                    path = staging / filename
                    mx.eval(*arrays.values())
                    mx.save_safetensors(
                        str(path),
                        dict(sorted(arrays.items())),
                        metadata={
                            "format": "mlx",
                            "mrun-codec": SOURCE_MLX_Q4_CODEC,
                            "mrun-role": role_value,
                            "mrun-source-artifact": artifact.artifact_id,
                        },
                    )
                    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    digest, size, _identity_value = _hash_regular_file(path)
                    shards.append(
                        {
                            "filename": filename,
                            "role": role_value,
                            "bytes": size,
                            "sha256": digest,
                            "parameters": sorted(parameters, key=lambda item: item["name"]),
                        }
                    )
                    arrays = {}
                    parameters = []
                    pending_bytes = 0
                    mx.clear_cache()

                for allocation in sorted(by_role[role], key=lambda item: item.source_tensor):
                    record = allocation_records.get(allocation.allocation_id)
                    if not isinstance(record, Mapping) or not isinstance(
                        record.get("blob"), Mapping
                    ):
                        raise SourceMlxArtifactError("canonical allocation manifest is incomplete")
                    tensor = _read_allocation(artifact, allocation, record)
                    logical_names = sorted(views_by_allocation[allocation.allocation_id])
                    common = {
                        "source_allocation_id": allocation.allocation_id,
                        "source_blob_sha256": record["blob"]["sha256"],
                        "source_tensor": allocation.source_tensor,
                        "source_dtype": source_dtype,
                        "source_shape": list(allocation.stored_shape),
                        "source_byte_count": allocation.byte_length,
                        "logical_names": logical_names,
                    }
                    new_arrays: list[tuple[str, Any, str, str, str]] = []
                    expected_encoding = _q4_expected_allocation_encoding(
                        architecture,
                        allocation.source_tensor,
                        allocation.stored_shape,
                    )
                    if expected_encoding == "mlx-affine-q4-g64":
                        packed = _quantize_source_affine_q4(tensor, mx=mx)
                        base = allocation.source_tensor[: -len(".weight")]
                        new_arrays.extend(
                            [
                                (
                                    f"{base}.weight",
                                    packed.weight,
                                    "U32",
                                    "weight",
                                    "mlx-affine-q4-g64",
                                ),
                                (
                                    f"{base}.scales",
                                    packed.scales,
                                    "BF16",
                                    "scales",
                                    "mlx-affine-q4-g64",
                                ),
                                (
                                    f"{base}.biases",
                                    packed.biases,
                                    "BF16",
                                    "biases",
                                    "mlx-affine-q4-g64",
                                ),
                            ]
                        )
                        q4_allocation_count += 1
                        maximum_error = max(maximum_error, packed.max_abs_error)
                        total_squared_error += packed.sum_squared_error
                        total_elements += packed.elements
                        error_records.append(
                            {
                                "source_allocation_id": allocation.allocation_id,
                                "source_tensor": allocation.source_tensor,
                                "logical_names": logical_names,
                                "elements": packed.elements,
                                "max_abs_error": packed.max_abs_error,
                                "sum_squared_error": packed.sum_squared_error,
                                "rmse": math.sqrt(packed.sum_squared_error / packed.elements),
                            }
                        )
                    elif expected_encoding == "source-exact":
                        auxiliary = _source_auxiliary_array(tensor, source_dtype, mx=mx)
                        new_arrays.append(
                            (
                                allocation.source_tensor,
                                auxiliary,
                                source_dtype,
                                "source_exact",
                                "source-exact",
                            )
                        )
                        auxiliary_allocation_count += 1
                    else:  # pragma: no cover - closed helper return set
                        raise AssertionError("unknown q4 allocation encoding policy")
                    added_bytes = sum(
                        int(array.nbytes) for _name, array, _dtype, _part, _encoding in new_arrays
                    )
                    if arrays and pending_bytes + added_bytes > SOURCE_MLX_SHARD_BYTES:
                        flush()
                    for name, value, dtype, part, encoding in new_arrays:
                        if name in arrays:
                            raise SourceMlxArtifactError(
                                f"multiple source allocations map to {name!r}"
                            )
                        arrays[name] = value
                        parameters.append(
                            {
                                **common,
                                "name": name,
                                "dtype": dtype,
                                "shape": [int(dimension) for dimension in value.shape],
                                "part": part,
                                "encoding": encoding,
                            }
                        )
                    pending_bytes += added_bytes
                    emitted_parameter_bytes += added_bytes
                flush()

            if not q4_allocation_count or not auxiliary_allocation_count or total_elements <= 0:
                raise SourceMlxArtifactError(
                    "direct q4 lowering did not observe both matrix and auxiliary allocations"
                )
            reopened = open_component_artifact(artifact.directory)
            if (
                reopened.artifact_id != artifact.artifact_id
                or reopened.manifest_sha256 != artifact.manifest_sha256
            ):
                raise SourceMlxArtifactError("canonical source changed during direct q4 lowering")
            tokenizer_custody = _tokenizer_custody_sha256(artifact)
            source = {
                "artifact_schema": artifact.manifest["schema_version"],
                "artifact_id": artifact.artifact_id,
                "manifest_sha256": artifact.manifest_sha256,
                "source_fingerprint": artifact.source.fingerprint,
                "ir_bundle_fingerprint": artifact.ir_bundle.fingerprint,
                "io_fingerprint": artifact.ir_bundle.io.fingerprint,
                "model_fingerprint": artifact.ir_bundle.model.fingerprint,
                "tokenizer_custody_sha256": tokenizer_custody,
                "architecture_id": artifact.ir_bundle.model.architecture_id,
                "architecture": architecture,
                "source_dtype": source_dtype,
                "tied_lexical_allocation": tied,
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
            }
            rmse = math.sqrt(total_squared_error / total_elements)
            manifest: dict[str, Any] = {
                "schema": SOURCE_MLX_Q4_NATIVE_SCHEMA,
                "status": "native-lowered-approximate-unexecuted",
                "build_key_sha256": build_key,
                "recipe": recipe,
                "source": source,
                "config": {
                    "filename": "config.json",
                    "bytes": config_bytes,
                    "file_sha256": config_file_sha,
                    "semantic_sha256": config_sha,
                },
                "quantization": {
                    "source_codec": (f"safetensors-{source_dtype.lower()}-canonical-source-v1"),
                    "quantizer": "mlx.core.quantize",
                    "q4_allocation_count": q4_allocation_count,
                    "elements": total_elements,
                    "max_abs_error": maximum_error,
                    "rmse": rmse,
                    "auxiliary_source_exact": True,
                    "auxiliary_allocation_count": auxiliary_allocation_count,
                    "blocks": sorted(error_records, key=lambda item: item["source_allocation_id"]),
                },
                "shards": sorted(shards, key=lambda item: item["filename"]),
                "coverage": {
                    "source_allocation_count": len(allocations),
                    "q4_allocation_count": q4_allocation_count,
                    "auxiliary_allocation_count": auxiliary_allocation_count,
                    "emitted_parameter_count": sum(len(item["parameters"]) for item in shards),
                    "source_allocation_bytes": sum(item.byte_length for item in allocations),
                    "emitted_parameter_bytes": emitted_parameter_bytes,
                    "source_dtype": source_dtype,
                    "all_source_allocations_emitted_once": True,
                    "physical_aliases_not_duplicated": True,
                    "auxiliary_source_exact": True,
                },
                "numerical_contract": SOURCE_MLX_Q4_NUMERICAL_CONTRACT,
                "approximate_quantized": True,
                "execution_certified": False,
                "native_runtime_candidate": True,
                "production_runtime_eligible": False,
            }
            manifest["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
            _write_durable(staging / "manifest.json", _canonical_json_bytes(manifest))
            _fsync_directory(staging)
            try:
                staging.rename(target)
            except FileExistsError:
                verified = VerifiedSourceMlxQ4Artifact(target)
                if verified.build_key_sha256 != build_key:
                    raise SourceMlxArtifactError(
                        "concurrent direct q4 build published a different recipe"
                    ) from None
                return SourceMlxQ4BuildRecord(
                    path=target,
                    artifact_sha256=verified.artifact_sha256,
                    build_key_sha256=build_key,
                    source_artifact_id=artifact.artifact_id,
                    shard_count=len(verified.manifest["shards"]),
                    shard_bytes=verified.shard_bytes,
                    verified_reopen=True,
                    max_abs_error=verified.max_abs_error,
                    rmse=verified.rmse,
                )
            _fsync_directory(output_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    verified = VerifiedSourceMlxQ4Artifact(target)
    if verified.build_key_sha256 != build_key:
        raise SourceMlxArtifactError("published direct q4 artifact lost recipe identity")
    return SourceMlxQ4BuildRecord(
        path=target,
        artifact_sha256=verified.artifact_sha256,
        build_key_sha256=build_key,
        source_artifact_id=artifact.artifact_id,
        shard_count=len(verified.manifest["shards"]),
        shard_bytes=verified.shard_bytes,
        verified_reopen=True,
        max_abs_error=verified.max_abs_error,
        rmse=verified.rmse,
    )


class _CanonicalSourceGraphFacade:
    def __init__(
        self,
        source: ComponentArtifact,
        *,
        model_name: str,
        architecture: str,
        tokenizer_descriptor: Mapping[str, Any],
        canonical_template_sha256: str,
    ) -> None:
        self.source = source
        self.model_name = model_name
        self.architecture = architecture
        self.custody_fingerprint_sha256 = source.artifact_id
        self.declared_fingerprint_sha256 = source.artifact_id
        self.tokenizer_semantic_sha256 = str(tokenizer_descriptor["semantic_sha256"])
        self.semantic_token_count = int(tokenizer_descriptor["length"])
        self.tokenizer_canonical_sha256 = canonical_template_sha256
        self.raw = {
            "tokenizer": {
                "chat_template_sha256": tokenizer_descriptor["chat_template_sha256"],
                "canonical_chat_template_sha256": canonical_template_sha256,
                "hash_kind": "canonical-source-plus-runtime-assets",
            }
        }

    def assert_unchanged(self) -> None:
        reopened = open_component_artifact(self.source.directory)
        if (
            reopened.artifact_id != self.source.artifact_id
            or reopened.manifest_sha256 != self.source.manifest_sha256
        ):
            raise SourceMlxArtifactError("canonical source artifact changed after native open")

    def close(self) -> None:
        return None


class MLXSourceComponentEngine(MLXComponentEngine):
    """Unquantized native Metal executor directly lowered from a canonical source artifact."""

    backend = "mlx-source-component"
    artifact_schema = SOURCE_MLX_NATIVE_SCHEMA
    artifact_builder = staticmethod(build_source_mlx_artifact)
    artifact_verifier = VerifiedSourceMlxArtifact
    artifact_root_default = "~/.cache/mrun/mlx-source-component"
    approximate_quantized_default = False
    compact_fused_weights_default = False

    def __init__(
        self,
        model_name: str,
        *,
        source_artifact: str | Path,
        native_root: str | Path | None = None,
        native_artifact: str | Path | None = None,
        lazy: bool = False,
        recurrent_context_limit: int | None = None,
        **_ignored: Any,
    ) -> None:
        from transformers import AutoTokenizer

        from mrun.models import resolve_model

        self._closed = False
        source = open_component_artifact(source_artifact)
        try:
            if native_artifact is None:
                root = (
                    Path(native_root).expanduser()
                    if native_root is not None
                    else Path(self.artifact_root_default).expanduser()
                )
                built = self.artifact_builder(source, root)
                native_path = built.path
            else:
                native_path = Path(native_artifact)
            self.artifact = self.artifact_verifier(native_path)
            if self.artifact.manifest.get("schema") != self.artifact_schema:
                raise SourceMlxArtifactError(
                    f"{self.backend} requires artifact schema {self.artifact_schema!r}"
                )
            if self.artifact.source.get("artifact_id") != source.artifact_id:
                raise SourceMlxArtifactError("native MLX artifact belongs to another source")
            self.spec = resolve_model(model_name)
            self.name = self.spec.name
            source_model_id = str(source.source.source_id).lower()
            accepted_model_ids = {
                str(model_name).lower(),
                str(self.spec.name).lower(),
                str(self.spec.hf_id).lower(),
            }
            if source_model_id not in accepted_model_ids:
                raise SourceMlxLoweringError(
                    "requested model identity differs from the canonical source identity"
                )
            architecture = _ARCHITECTURES[source.ir_bundle.model.architecture_id]
            if (
                self.spec.family != architecture
                or self.artifact.source.get("architecture") != architecture
            ):
                raise SourceMlxLoweringError(
                    "requested model differs from canonical source architecture"
                )
            self.arch = architecture
            # Load only from the source-custodied assets.  The runtime therefore cannot silently
            # pair source weights with a same-named tokenizer from a mutable global cache.
            self.tokenizer = AutoTokenizer.from_pretrained(
                source.directory / "assets",
                local_files_only=True,
                trust_remote_code=False,
            )
            descriptor = _runtime_tokenizer_descriptor(self.tokenizer)
            io = source.ir_bundle.io
            source_assets = {item.path: item for item in io.tokenizer_assets}
            for path, item in source_assets.items():
                source_file = source.directory / "assets" / path
                digest, size, _identity_value = _hash_regular_file(
                    source_file,
                    expected_size=item.byte_count,
                    expected_sha256=item.sha256,
                )
                if digest != item.sha256 or size != item.byte_count:
                    raise SourceMlxArtifactError("runtime tokenizer asset custody changed")
            templates = tuple(io.chat_templates)
            selected = next((item for item in templates if item.template_id == "default"), None)
            chat_template = str(getattr(self.tokenizer, "chat_template", "") or "")
            canonical_template = _sha256_bytes(_canonical_json_bytes({"content": chat_template}))
            if selected is None and (architecture not in {"gpt_neox", "mamba"} or chat_template):
                raise SourceMlxLoweringError("canonical source has no default chat template")
            if selected is not None and canonical_template != selected.sha256:
                raise SourceMlxArtifactError(
                    "runtime tokenizer chat template differs from source IO"
                )
            if int(descriptor["length"]) > int(self.artifact.config["vocab_size"]):
                raise SourceMlxArtifactError("runtime tokenizer exceeds physical vocabulary rows")
            self.graph = _CanonicalSourceGraphFacade(
                source,
                model_name=self.name,
                architecture=architecture,
                tokenizer_descriptor=descriptor,
                canonical_template_sha256=canonical_template,
            )

            try:
                import mlx.core as mx
                from mlx_lm.utils import load_model
            except ImportError as exc:
                raise RuntimeError("MLXSourceComponentEngine requires mlx and mlx-lm") from exc
            self._mx = mx
            self.path = self.artifact.path
            active_memory_before_load = int(mx.get_active_memory())
            self.model, loaded_config = load_model(self.path, lazy=lazy, strict=True)
            self.model.eval()
            self.cfg = self.model.args
            if _sha256_bytes(_canonical_json_bytes(loaded_config)) != _sha256_bytes(
                _canonical_json_bytes(self.artifact.config)
            ):
                raise SourceMlxArtifactError("mlx-lm changed the direct-source native config")
            self.n_layer = int(self.cfg.num_hidden_layers)
            intermediate = getattr(self.cfg, "intermediate_size", None)
            if intermediate is None:
                intermediate = self.artifact.config.get("intermediate_size")
            self.inter = int(intermediate)
            self.hidden = int(self.cfg.hidden_size)
            if architecture == "mamba":
                if recurrent_context_limit is None:
                    recurrent_context_limit = 1_048_576
                if (
                    isinstance(recurrent_context_limit, bool)
                    or not isinstance(recurrent_context_limit, int)
                    or recurrent_context_limit <= 0
                ):
                    raise SourceMlxLoweringError(
                        "Mamba recurrent_context_limit must be a positive integer"
                    )
                self.context_size = recurrent_context_limit
            else:
                if recurrent_context_limit is not None:
                    raise SourceMlxLoweringError("recurrent_context_limit is valid only for Mamba")
                self.context_size = int(self.artifact.config["max_position_embeddings"])
            self.semantic_token_count = int(descriptor["length"])
            self.artifact_bytes = int(self.artifact.shard_bytes)
            active_memory_after_load = int(mx.get_active_memory())
            self.mlx_engine_active_memory_bytes_at_load = max(
                0, active_memory_after_load - active_memory_before_load
            )
            self.mlx_active_memory_mb_at_load = active_memory_after_load / 1024**2
            self.mlx_peak_memory_mb_at_load = float(mx.get_peak_memory()) / 1024**2
            self.working_set_mb = self.mlx_engine_active_memory_bytes_at_load / 1024**2
            self.approximate_quantized = self.approximate_quantized_default
            self.compact_fused_weights = self.compact_fused_weights_default
            self.numerical_contract = self._artifact_numerical_contract()
        except BaseException:
            self.close()
            raise

    def _artifact_numerical_contract(self) -> str:
        if self.arch == "gpt_neox":
            return (
                f"mlx-source-component-{self.artifact.source_dtype.lower()}-"
                "gpt-neox-approximate-gelu-native-v1"
            )
        if self.arch == "mamba":
            return "mlx-source-component-f32-mamba1-sequential-selective-scan-v1"
        return f"mlx-source-component-{self.artifact.source_dtype.lower()}-native-v1"

    def assert_content_identity_unchanged(self) -> None:
        if self._closed:
            raise SourceMlxArtifactError("direct-source native engine is closed")
        self.graph.assert_unchanged()
        self.artifact.assert_unchanged()

    def capabilities(self):
        from mrun.engine.base import EngineCapabilities

        return EngineCapabilities(
            logits=True,
            logits_batch=True,
            mlp_acts=False,
            mlp_acts_batch=False,
            approximate_quantized=self.approximate_quantized,
            generation=True,
            generation_batch=True,
            persistent_kv=True,
            compact_fused_weights=self.compact_fused_weights,
        )

    def runtime_report(self) -> dict[str, Any]:
        report = super().runtime_report()
        report.update(
            {
                "source_artifact_schema": self.artifact.source["artifact_schema"],
                "source_artifact_id": self.artifact.source["artifact_id"],
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
                "approximate_quantized": self.approximate_quantized,
                "production_runtime_eligible": self.artifact.manifest[
                    "production_runtime_eligible"
                ],
            }
        )
        return report


class MLXSourceQ4Engine(MLXSourceComponentEngine):
    """Approximate affine-q4 Metal executor directly lowered from canonical source bytes."""

    backend = "mlx-source-q4"
    artifact_schema = SOURCE_MLX_Q4_NATIVE_SCHEMA
    artifact_builder = staticmethod(build_source_mlx_q4_artifact)
    artifact_verifier = VerifiedSourceMlxQ4Artifact
    artifact_root_default = "~/.cache/mrun/mlx-source-q4"
    approximate_quantized_default = True
    compact_fused_weights_default = True

    def _artifact_numerical_contract(self) -> str:
        return SOURCE_MLX_Q4_NUMERICAL_CONTRACT


__all__ = [
    "SOURCE_MLX_BUILDER_ABI",
    "SOURCE_MLX_MAPPING_ABI",
    "SOURCE_MLX_NATIVE_SCHEMA",
    "SOURCE_MLX_Q4_BITS",
    "SOURCE_MLX_Q4_BUILDER_ABI",
    "SOURCE_MLX_Q4_CODEC",
    "SOURCE_MLX_Q4_GROUP_SIZE",
    "SOURCE_MLX_Q4_MODE",
    "SOURCE_MLX_Q4_NATIVE_SCHEMA",
    "SOURCE_MLX_Q4_NUMERICAL_CONTRACT",
    "MLXSourceComponentEngine",
    "MLXSourceQ4Engine",
    "SourceMlxArtifactError",
    "SourceMlxBuildRecord",
    "SourceMlxQ4BuildRecord",
    "SourceMlxLoweringError",
    "VerifiedSourceMlxArtifact",
    "VerifiedSourceMlxQ4Artifact",
    "build_source_mlx_artifact",
    "build_source_mlx_q4_artifact",
]
