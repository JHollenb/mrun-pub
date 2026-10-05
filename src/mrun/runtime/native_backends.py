"""Concrete, fail-closed bindings for the native component executors.

The engine implementations predate :mod:`mrun.runtime` and are deliberately useful on their
own.  This module is the custody bridge that turns an *already verified and opened* native
component engine into the backend-neutral execution protocol:

``verified blobs -> CompiledModelIdentity -> capabilities -> placement -> ModelRuntime``.

Keeping the binding over an opened engine is intentional.  MLX state geometry is observable only
after the architecture has produced K/V once, while a CUDA resident-head route has derived device
memory that is not a checkpoint blob.  Both facts must be measured and charged before admission;
guessing them from ``config.json`` is not safe.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from .contracts import (
    BackendCapabilities,
    BlobIdentity,
    CodecCapability,
    CompiledComponent,
    CompiledModelIdentity,
    DeviceDescriptor,
    MemoryDomain,
    OutputMode,
    PlacementPlan,
    PromotionStatus,
    WorkloadSpec,
)
from .dense_cuda import DenseCudaNativeRuntime
from .mlx_mamba import (
    MlxMambaPrefillExecutionShape,
    MlxMambaRuntime,
    build_mlx_mamba_prefill_execution_shape,
)
from .mlx_native import MlxNativeRuntime, MlxStateLayout, inspect_mlx_state_layout
from .mlx_paged_attention import MlxPagedDecodeAttentionLane
from .mlx_paged_kv import (
    MLX_PAGED_KV_CACHE_ABI,
    MlxKVPagePool,
    MlxPagedKVCacheFactory,
)
from .mlx_quantized_kv import FixedMlxQuantizedKVCache, MlxAffineKVCodec
from .placement import plan_resident_placement

NATIVE_STATE_ABI = "mrun-transactional-gqa-kv-v1"
MAMBA_RECURRENT_STATE_ABI = "mrun-mlx-mamba1-recurrent-state-v1"
MLX_BACKEND_ABI = "mrun-mlx-component-runtime-v1"
MLX_MAMBA_BACKEND_ABI = "mrun-mlx-mamba-component-runtime-v1"
DENSE_CUDA_BACKEND_ABI = "mrun-dense-cuda-component-runtime-v1"

_DENSE_DECODER_OPERATORS = (
    "causal-grouped-query-attention",
    "embedding-row-lookup",
    "elementwise-multiply",
    "linear",
    "linear-readout",
    "residual-add",
    "rms-norm",
    "rotary-default",
    "silu",
)
_MIXTRAL_DECODER_OPERATORS = tuple(
    sorted(
        {
            *_DENSE_DECODER_OPERATORS,
            "moe-router-linear",
            "moe-top-k-softmax",
            "moe-token-dispatch",
            "moe-routed-expert-linear",
            "moe-weighted-scatter-add",
        }
    )
)
_MAMBA_DECODER_OPERATORS = (
    "elementwise-multiply",
    "linear",
    "linear-readout",
    "mamba-causal-depthwise-convolution",
    "mamba-input-gate-split",
    "mamba-rms-norm",
    "mamba-selection-split",
    "mamba-selective-scan",
    "residual-add",
    "silu",
    "token-embedding",
)


def _mlx_operator_ids(architecture: str) -> tuple[str, ...]:
    if architecture == "mixtral":
        return _MIXTRAL_DECODER_OPERATORS
    if architecture == "mamba":
        return _MAMBA_DECODER_OPERATORS
    return _DENSE_DECODER_OPERATORS


class NativeBackendBindingError(RuntimeError):
    """An executable engine cannot prove the identity or residency it advertises."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeBackendBindingError(f"{field} must be a positive integer")
    return int(value)


def _sha256(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeBackendBindingError(f"{field} is not a lowercase SHA-256 digest")
    return value


def _dtype_name_and_bytes(value: Any) -> tuple[str, int]:
    name = str(value).lower().rsplit(".", maxsplit=1)[-1]
    aliases = {
        "bf16": ("bfloat16", 2),
        "bfloat16": ("bfloat16", 2),
        "fp16": ("float16", 2),
        "float16": ("float16", 2),
        "half": ("float16", 2),
        "fp32": ("float32", 4),
        "float32": ("float32", 4),
        "float": ("float32", 4),
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise NativeBackendBindingError(f"unsupported native state dtype {value!r}") from exc


@dataclass(frozen=True, slots=True)
class _DenseCudaBodyWorkspaceGeometry:
    layers: int
    attention_heads: int
    kv_heads: int
    hidden_size: int
    head_dim: int
    intermediate_size: int
    compute_element_bytes: int


def _dense_cuda_body_workspace_geometry(engine: Any) -> _DenseCudaBodyWorkspaceGeometry:
    cfg = getattr(engine, "cfg", None)
    store = getattr(engine, "store", None)
    if not isinstance(cfg, Mapping) or store is None:
        raise NativeBackendBindingError("CUDA body workspace requires runtime config and store")
    _dtype_name, element_bytes = _dtype_name_and_bytes(getattr(store, "compute_dtype", None))
    geometry = _DenseCudaBodyWorkspaceGeometry(
        layers=_strict_positive_int(cfg.get("num_hidden_layers"), "CUDA layer count"),
        attention_heads=_strict_positive_int(
            cfg.get("num_attention_heads"), "CUDA attention head count"
        ),
        kv_heads=_strict_positive_int(
            cfg.get("num_key_value_heads", cfg.get("num_attention_heads")),
            "CUDA KV head count",
        ),
        hidden_size=_strict_positive_int(cfg.get("hidden_size"), "CUDA hidden size"),
        head_dim=_strict_positive_int(
            cfg.get(
                "head_dim",
                _strict_positive_int(cfg.get("hidden_size"), "CUDA hidden size")
                // _strict_positive_int(
                    cfg.get("num_attention_heads"), "CUDA attention head count"
                ),
            ),
            "CUDA head dimension",
        ),
        intermediate_size=_strict_positive_int(
            cfg.get("intermediate_size"), "CUDA intermediate size"
        ),
        compute_element_bytes=element_bytes,
    )
    if geometry.hidden_size != geometry.attention_heads * geometry.head_dim:
        raise NativeBackendBindingError(
            "CUDA hidden size must equal attention heads times head dimension"
        )
    if geometry.attention_heads % geometry.kv_heads:
        raise NativeBackendBindingError("CUDA attention heads must be divisible by KV heads")
    return geometry


def _dense_cuda_body_workspace_bytes(
    geometry: _DenseCudaBodyWorkspaceGeometry,
    *,
    max_batch_size: int,
    max_context_tokens: int,
) -> int:
    """Conservatively charge the eager dense-CUDA body at the admitted dispatch ceiling.

    The established Torch attention trajectory materializes FP32 scores and probabilities with
    shape ``[B, H, T, T]``.  The executor also retains a full provisional K/V delta until commit.
    Neither allocation is part of the fixed state arena.  Linear terms below cover the tensors
    simultaneously live around attention, RMSNorm, and the gated MLP; taking their maximum avoids
    pretending mutually exclusive phases overlap while remaining conservative within each phase.
    """

    batch = _strict_positive_int(max_batch_size, "CUDA workspace batch size")
    context = _strict_positive_int(max_context_tokens, "CUDA workspace context")
    fp32_bytes = 4
    int64_bytes = 8
    bool_bytes = 1
    rows = batch * context
    square_rows = batch * context * context
    compute_bytes = geometry.compute_element_bytes

    # Result-owned allocations survive the body and overlap the selected language head.
    kv_delta_bytes = (
        rows * 2 * geometry.layers * geometry.kv_heads * geometry.head_dim * compute_bytes
    )
    hidden_result_bytes = rows * geometry.hidden_size * compute_bytes

    # At softmax/context time the established path holds scores, probabilities, the causal mask,
    # FP32 Q/K/V/context views, expanded GQA K/V, and the body tensors feeding attention.
    attention_quadratic_bytes = square_rows * (
        2 * geometry.attention_heads * fp32_bytes + bool_bytes
    )
    attention_linear_elements_compute = (
        geometry.hidden_size  # normalized hidden
        + geometry.attention_heads * geometry.head_dim  # query
        + 2 * geometry.kv_heads * geometry.head_dim  # new K/V
        + 2 * geometry.attention_heads * geometry.head_dim  # repeated GQA K/V
        + geometry.attention_heads * geometry.head_dim  # returned compute context
    )
    attention_linear_bytes = rows * (
        attention_linear_elements_compute * compute_bytes
        + 4 * geometry.attention_heads * geometry.head_dim * fp32_bytes
        + 3 * int64_bytes
    )
    attention_peak_bytes = attention_quadratic_bytes + attention_linear_bytes

    # The eager reference RMSNorm holds multiple FP32 hidden-width temporaries before narrowing.
    norm_peak_bytes = rows * (
        4 * geometry.hidden_size * fp32_bytes + geometry.hidden_size * compute_bytes + int64_bytes
    )

    # SiLU and multiply can overlap gate, up, the nonlinear temporary, and the resulting MLP row.
    # Residual/projection tensors and still-live Q/K/V are charged alongside that four-way peak.
    mlp_peak_bytes = rows * (
        (
            4 * geometry.hidden_size
            + geometry.attention_heads * geometry.head_dim
            + 2 * geometry.kv_heads * geometry.head_dim
            + 4 * geometry.intermediate_size
        )
        * compute_bytes
        + 3 * int64_bytes
    )

    return int(
        kv_delta_bytes
        + hidden_result_bytes
        + max(attention_peak_bytes, norm_peak_bytes, mlp_peak_bytes)
    )


def _dense_cuda_segmented_decode_workspace_bytes(
    geometry: _DenseCudaBodyWorkspaceGeometry,
    *,
    max_batch_size: int,
    max_context_tokens: int,
) -> int:
    """Charge established B1 prefill plus segmented one-token decode waves.

    ``segmented-flash-gqa-decode-v1`` is deliberately a phase-split ABI: prefill still follows
    the established eager B1 trajectory, while a physical decode wave contains exactly one new
    token per row and reads committed K/V in place.  The latter retains a one-token provisional
    K/V delta and linear body temporaries, but it owns neither a joined K/V prefix nor a score or
    probability matrix.  Taking the larger phase peak admits both without pretending that a
    B-wide full-context prefill is part of this execution contract.
    """

    batch = _strict_positive_int(max_batch_size, "CUDA workspace batch size")
    context = _strict_positive_int(max_context_tokens, "CUDA workspace context")
    prefill_b1 = _dense_cuda_body_workspace_bytes(
        geometry,
        max_batch_size=1,
        max_context_tokens=context,
    )
    compute_bytes = geometry.compute_element_bytes
    fp32_bytes = 4
    int64_bytes = 8

    kv_delta_bytes = (
        batch
        * 2
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * compute_bytes
    )
    hidden_result_bytes = batch * geometry.hidden_size * compute_bytes
    attention_peak_bytes = batch * (
        (
            3 * geometry.hidden_size
            + 2 * geometry.kv_heads * geometry.head_dim
            + geometry.attention_heads * geometry.head_dim
        )
        * compute_bytes
        + 2 * geometry.attention_heads * geometry.head_dim * fp32_bytes
        + 3 * int64_bytes
    )
    norm_peak_bytes = batch * (
        4 * geometry.hidden_size * fp32_bytes
        + geometry.hidden_size * compute_bytes
        + int64_bytes
    )
    mlp_peak_bytes = batch * (
        (4 * geometry.hidden_size + 4 * geometry.intermediate_size) * compute_bytes
        + 3 * int64_bytes
    )
    decode_wave = int(
        kv_delta_bytes
        + hidden_result_bytes
        + max(attention_peak_bytes, norm_peak_bytes, mlp_peak_bytes)
    )
    return max(prefill_b1, decode_wave)


def dense_cuda_body_workspace_bytes(
    engine: Any,
    *,
    max_batch_size: int,
    max_context_tokens: int,
) -> int:
    """Return the fail-closed body workspace charge for one opened dense CUDA engine."""

    geometry = _dense_cuda_body_workspace_geometry(engine)
    mode = str(
        getattr(getattr(engine, "target", None), "decode_attention_mode", "established")
    )
    if mode == "segmented-flash-gqa-decode-v1":
        return _dense_cuda_segmented_decode_workspace_bytes(
            geometry,
            max_batch_size=max_batch_size,
            max_context_tokens=max_context_tokens,
        )
    if mode != "established":
        raise NativeBackendBindingError(f"unknown CUDA decode attention mode {mode!r}")
    return _dense_cuda_body_workspace_bytes(
        geometry,
        max_batch_size=max_batch_size,
        max_context_tokens=max_context_tokens,
    )


def _codec_capabilities(
    components: tuple[CompiledComponent, ...],
) -> tuple[CodecCapability, ...]:
    roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for component in components:
        roles[(component.codec_id, component.layout_id)].add(component.role)
    return tuple(
        CodecCapability(
            codec_id=codec,
            layout_id=layout,
            component_roles=tuple(sorted(component_roles)),
            native_direct=True,
        )
        for (codec, layout), component_roles in sorted(roles.items())
    )


def compiled_identity_from_mlx_engine(
    engine: Any,
    *,
    state_layout: MlxStateLayout | None = None,
    state_abi: str = NATIVE_STATE_ABI,
) -> tuple[CompiledModelIdentity, MlxStateLayout]:
    """Build the exact runtime identity of a verified MLX component artifact.

    The native safetensor shards—not a source checkpoint estimate—are the physical component
    blobs.  The returned state charge is based on a live K/V probe unless a previously measured
    layout is supplied.
    """

    graph = getattr(engine, "graph", None)
    artifact = getattr(engine, "artifact", None)
    manifest = getattr(artifact, "manifest", None)
    if graph is None or artifact is None or not isinstance(manifest, Mapping):
        raise NativeBackendBindingError("MLX engine exposes no verified graph/artifact custody")
    assert_unchanged = getattr(engine, "assert_content_identity_unchanged", None)
    if not callable(assert_unchanged):
        raise NativeBackendBindingError("MLX engine exposes no post-open custody guard")
    assert_unchanged()

    recipe = manifest.get("recipe")
    shards = manifest.get("shards")
    if not isinstance(recipe, Mapping) or not isinstance(shards, list) or not shards:
        raise NativeBackendBindingError("MLX native manifest has no recipe or shards")
    codec = str(recipe.get("codec", ""))
    schema = str(manifest.get("schema", ""))
    if not codec or not schema:
        raise NativeBackendBindingError("MLX native manifest omits codec/schema identity")

    grouped: dict[str, list[BlobIdentity]] = defaultdict(list)
    seen_filenames: set[str] = set()
    for record in shards:
        if not isinstance(record, Mapping):
            raise NativeBackendBindingError("MLX shard record must be an object")
        role = str(record.get("role", ""))
        filename = str(record.get("filename", ""))
        if not role or not filename or filename in seen_filenames:
            raise NativeBackendBindingError("MLX shard role/filename is absent or duplicated")
        seen_filenames.add(filename)
        grouped[role].append(
            BlobIdentity(
                blob_id=filename,
                sha256=_sha256(record.get("sha256"), f"MLX shard {filename} hash"),
                byte_count=_strict_positive_int(record.get("bytes"), f"MLX shard {filename} bytes"),
            )
        )
    components = tuple(
        CompiledComponent(
            component_id=role,
            role=role,
            allocation_id=f"mlx-native.{role}",
            codec_id=codec,
            layout_id=schema,
            physical_bytes=sum(blob.byte_count for blob in blobs),
            blobs=tuple(blobs),
        )
        for role, blobs in sorted(grouped.items())
    )

    measured = state_layout or inspect_mlx_state_layout(engine)
    if measured.bytes_per_token <= 0 or not measured.dtype_name:
        raise NativeBackendBindingError("MLX state probe returned an invalid K/V layout")
    source_custody = _sha256(
        getattr(graph, "custody_fingerprint_sha256", None),
        "MLX source graph custody",
    )
    artifact_sha = _sha256(
        getattr(artifact, "artifact_sha256", manifest.get("artifact_sha256")),
        "MLX artifact identity",
    )
    vocab_sha = _sha256(
        getattr(graph, "tokenizer_semantic_sha256", None),
        "MLX tokenizer identity",
    )
    compiler_abi = "+".join(
        str(value) for value in (recipe.get("builder_abi"), recipe.get("mapping_abi")) if value
    )
    if not compiler_abi:
        raise NativeBackendBindingError("MLX artifact has no builder/mapping ABI")
    return (
        CompiledModelIdentity(
            model_name=str(getattr(engine, "name", getattr(graph, "model_name", ""))),
            architecture=str(getattr(engine, "arch", getattr(graph, "architecture", ""))),
            source_revision_sha256=source_custody,
            semantic_model_sha256=artifact_sha,
            component_graph_sha256=source_custody,
            vocab_manifest_sha256=vocab_sha,
            compiler_abi=compiler_abi,
            components=components,
            operator_ids=_mlx_operator_ids(
                str(getattr(engine, "arch", getattr(graph, "architecture", "")))
            ),
            state_abi=state_abi,
            state_dtype=measured.dtype_name,
            state_bytes_per_token=measured.bytes_per_token,
            max_context_tokens=_strict_positive_int(
                int(getattr(engine, "context_size", 0)), "MLX context size"
            ),
            semantic_token_count=_strict_positive_int(
                int(getattr(engine, "semantic_token_count", 0)),
                "MLX semantic token count",
            ),
        ),
        measured,
    )


def compiled_identity_from_mlx_mamba_engine(engine: Any) -> CompiledModelIdentity:
    """Build an honest fixed-state identity for a verified direct-source Mamba1 engine.

    A Mamba recurrence is constant-size per live sequence.  Representing it as bytes/token would
    make context admission grow linearly like transformer KV and erase the architecture's central
    property, so this path binds the v2 fixed-per-row state term explicitly.
    """

    graph = getattr(engine, "graph", None)
    artifact = getattr(engine, "artifact", None)
    manifest = getattr(artifact, "manifest", None)
    if graph is None or artifact is None or not isinstance(manifest, Mapping):
        raise NativeBackendBindingError("Mamba MLX engine exposes no verified graph/artifact")
    if str(getattr(engine, "arch", "")) != "mamba":
        raise NativeBackendBindingError("fixed-state MLX identity requires Mamba")
    assert_unchanged = getattr(engine, "assert_content_identity_unchanged", None)
    if not callable(assert_unchanged):
        raise NativeBackendBindingError("Mamba MLX engine exposes no custody guard")
    assert_unchanged()
    if str(getattr(artifact, "source_dtype", "")) != "F32":
        raise NativeBackendBindingError("registered Mamba MLX state arithmetic requires F32")

    recipe = manifest.get("recipe")
    shards = manifest.get("shards")
    config = getattr(artifact, "config", None)
    if (
        not isinstance(recipe, Mapping)
        or not isinstance(shards, list)
        or not shards
        or not isinstance(config, Mapping)
    ):
        raise NativeBackendBindingError("Mamba MLX artifact has no recipe, shards, or config")
    codec = str(recipe.get("codec", ""))
    schema = str(manifest.get("schema", ""))
    if not codec or not schema:
        raise NativeBackendBindingError("Mamba MLX artifact omits codec/schema identity")

    grouped: dict[str, list[BlobIdentity]] = defaultdict(list)
    seen_filenames: set[str] = set()
    for record in shards:
        if not isinstance(record, Mapping):
            raise NativeBackendBindingError("Mamba MLX shard record must be an object")
        role = str(record.get("role", ""))
        filename = str(record.get("filename", ""))
        if not role or not filename or filename in seen_filenames:
            raise NativeBackendBindingError("Mamba MLX shard role/filename is absent or duplicated")
        seen_filenames.add(filename)
        grouped[role].append(
            BlobIdentity(
                blob_id=filename,
                sha256=_sha256(record.get("sha256"), f"Mamba MLX shard {filename} hash"),
                byte_count=_strict_positive_int(
                    record.get("bytes"), f"Mamba MLX shard {filename} bytes"
                ),
            )
        )
    components = tuple(
        CompiledComponent(
            component_id=role,
            role=role,
            allocation_id=f"mlx-native.{role}",
            codec_id=codec,
            layout_id=schema,
            physical_bytes=sum(blob.byte_count for blob in blobs),
            blobs=tuple(blobs),
        )
        for role, blobs in sorted(grouped.items())
    )

    layers = _strict_positive_int(config.get("num_hidden_layers"), "Mamba layer count")
    intermediate = _strict_positive_int(config.get("intermediate_size"), "Mamba intermediate size")
    state_size = _strict_positive_int(config.get("state_size"), "Mamba state size")
    conv_kernel = _strict_positive_int(config.get("conv_kernel"), "Mamba convolution kernel")
    # mlx-lm retains K-1 pre-convolution inputs plus the selective-scan recurrence for each layer.
    fixed_state_bytes = layers * intermediate * (conv_kernel - 1 + state_size) * 4
    if fixed_state_bytes <= 0:
        raise NativeBackendBindingError("Mamba fixed recurrent state byte charge is invalid")
    compiler_abi = "+".join(
        str(value) for value in (recipe.get("builder_abi"), recipe.get("mapping_abi")) if value
    )
    if not compiler_abi:
        raise NativeBackendBindingError("Mamba MLX artifact has no compiler ABI")

    return CompiledModelIdentity(
        model_name=str(getattr(engine, "name", getattr(graph, "model_name", ""))),
        architecture="mamba",
        source_revision_sha256=_sha256(
            getattr(graph, "custody_fingerprint_sha256", None),
            "Mamba source graph custody",
        ),
        semantic_model_sha256=_sha256(
            getattr(artifact, "artifact_sha256", manifest.get("artifact_sha256")),
            "Mamba MLX artifact identity",
        ),
        component_graph_sha256=_sha256(
            getattr(graph, "custody_fingerprint_sha256", None),
            "Mamba component graph custody",
        ),
        vocab_manifest_sha256=_sha256(
            getattr(graph, "tokenizer_semantic_sha256", None),
            "Mamba tokenizer identity",
        ),
        compiler_abi=compiler_abi,
        components=components,
        operator_ids=_MAMBA_DECODER_OPERATORS,
        state_abi=MAMBA_RECURRENT_STATE_ABI,
        state_dtype="float32",
        state_bytes_per_token=0,
        state_fixed_bytes_per_row=fixed_state_bytes,
        max_context_tokens=_strict_positive_int(
            getattr(engine, "context_size", 0), "Mamba service context limit"
        ),
        semantic_token_count=_strict_positive_int(
            getattr(engine, "semantic_token_count", 0),
            "Mamba semantic token count",
        ),
    )


def _compiled_identity_from_direct_source_cuda_engine(
    engine: Any,
) -> CompiledModelIdentity:
    """Build identity from a verified direct-source CUDA artifact without QStore lineage."""

    artifact = getattr(engine, "direct_artifact", None)
    store = getattr(engine, "store", None)
    if artifact is None or store is None:
        raise NativeBackendBindingError("direct CUDA binding requires a native source artifact")
    guard = getattr(engine, "assert_content_identity_unchanged", None)
    if not callable(guard):
        raise NativeBackendBindingError("direct CUDA engine exposes no custody guard")
    guard()
    source = getattr(artifact, "source", None)
    components_record = getattr(artifact, "components", None)
    if not isinstance(source, Mapping) or not isinstance(components_record, Mapping):
        raise NativeBackendBindingError("direct CUDA artifact identity is incomplete")
    if (
        source.get("direct_from_canonical_source") is not True
        or source.get("intermediate_qstore") is not False
    ):
        raise NativeBackendBindingError("direct CUDA artifact crossed its source-only boundary")

    components: list[CompiledComponent] = []
    for role, record in sorted(components_record.items()):
        raw_blobs = record.get("blobs") if isinstance(record, Mapping) else None
        if not isinstance(raw_blobs, Mapping) or not raw_blobs:
            raise NativeBackendBindingError(f"direct CUDA component {role!r} has no blobs")
        blobs = tuple(
            BlobIdentity(
                blob_id=str(blob.get("path")),
                sha256=_sha256(blob.get("sha256"), f"direct CUDA {role}/{filename} hash"),
                byte_count=_strict_positive_int(
                    blob.get("bytes"), f"direct CUDA {role}/{filename} bytes"
                ),
            )
            for filename, blob in sorted(raw_blobs.items())
            if isinstance(blob, Mapping)
        )
        if len(blobs) != len(raw_blobs):
            raise NativeBackendBindingError(f"direct CUDA component {role!r} is malformed")
        components.append(
            CompiledComponent(
                component_id=str(role),
                role=str(role),
                allocation_id=f"canonical-source-cuda-int8.{role}",
                codec_id=str(record.get("codec")),
                layout_id=str(record.get("layout")),
                physical_bytes=sum(blob.byte_count for blob in blobs),
                blobs=blobs,
            )
        )

    cfg = getattr(engine, "cfg", None)
    if not isinstance(cfg, Mapping):
        raise NativeBackendBindingError("direct CUDA engine exposes no runtime config")
    dtype_name, dtype_bytes = _dtype_name_and_bytes(getattr(store, "compute_dtype", None))
    layers = _strict_positive_int(cfg.get("num_hidden_layers"), "CUDA layer count")
    attention_heads = _strict_positive_int(
        cfg.get("num_attention_heads"), "CUDA attention head count"
    )
    kv_heads = _strict_positive_int(
        cfg.get("num_key_value_heads", attention_heads), "CUDA KV head count"
    )
    hidden = _strict_positive_int(cfg.get("hidden_size"), "CUDA hidden size")
    head_dim = _strict_positive_int(
        cfg.get("head_dim", hidden // attention_heads), "CUDA head dimension"
    )
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    max_context = min(
        _strict_positive_int(cfg.get("max_position_embeddings"), "CUDA model context"),
        _strict_positive_int(getattr(engine, "max_seq_len", 0), "CUDA runtime context"),
    )
    recipe = getattr(artifact, "recipe", None)
    if not isinstance(recipe, Mapping):
        raise NativeBackendBindingError("direct CUDA artifact has no compiler recipe")
    compiler_abi = f"{recipe.get('builder_abi')}+{recipe.get('mapping_abi')}+direct-no-qstore"
    head_execution_abi = str(getattr(engine, "head_execution_abi", ""))
    if head_execution_abi and head_execution_abi != "exact-fp32-row-blocks-v1":
        compiler_abi = f"{compiler_abi}+head-{head_execution_abi}"
    return CompiledModelIdentity(
        model_name=str(getattr(engine, "name", "")),
        architecture=str(getattr(engine, "arch", source.get("architecture", ""))),
        source_revision_sha256=_sha256(
            source.get("manifest_sha256"), "direct CUDA source manifest identity"
        ),
        semantic_model_sha256=_sha256(
            source.get("model_fingerprint"), "direct CUDA semantic model identity"
        ),
        component_graph_sha256=_sha256(
            getattr(artifact, "artifact_sha256", None), "direct CUDA artifact identity"
        ),
        vocab_manifest_sha256=_sha256(
            source.get("tokenizer_custody_sha256"), "direct CUDA tokenizer identity"
        ),
        compiler_abi=compiler_abi,
        components=tuple(components),
        operator_ids=_DENSE_DECODER_OPERATORS,
        state_abi=NATIVE_STATE_ABI,
        state_dtype=dtype_name,
        state_bytes_per_token=bytes_per_token,
        max_context_tokens=max_context,
        semantic_token_count=_strict_positive_int(
            int(getattr(engine, "semantic_token_count", 0)),
            "direct CUDA semantic token count",
        ),
    )


def compiled_identity_from_dense_cuda_engine(engine: Any) -> CompiledModelIdentity:
    """Build identity from either direct-source or legacy component-QStore CUDA."""

    if getattr(engine, "direct_artifact", None) is not None:
        return _compiled_identity_from_direct_source_cuda_engine(engine)

    composite = getattr(engine, "composite_store", None)
    graph = getattr(composite, "graph", None)
    if composite is None or graph is None:
        raise NativeBackendBindingError("CUDA native binding requires a component graph engine")
    guard = getattr(composite, "assert_content_identity_unchanged", None)
    if not callable(guard):
        raise NativeBackendBindingError("CUDA component store exposes no custody guard")
    # Establish payload custody for every role, including a tied lexical role that may not have
    # been touched by the most recent output contract.
    for role in sorted(graph.components):
        graph.verify_component_blobs(role)
    guard()

    components: list[CompiledComponent] = []
    for role, record in sorted(graph.components.items()):
        raw_blobs = record.get("blobs")
        if not isinstance(raw_blobs, Mapping) or not raw_blobs:
            raise NativeBackendBindingError(f"CUDA component {role!r} has no blob custody")
        blobs = tuple(
            BlobIdentity(
                blob_id=f"{role}/{filename}",
                sha256=_sha256(blob.get("sha256"), f"CUDA {role}/{filename} hash"),
                byte_count=_strict_positive_int(blob.get("bytes"), f"CUDA {role}/{filename} bytes"),
            )
            for filename, blob in sorted(raw_blobs.items())
            if isinstance(blob, Mapping)
        )
        if len(blobs) != len(raw_blobs):
            raise NativeBackendBindingError(f"CUDA component {role!r} has a malformed blob")
        components.append(
            CompiledComponent(
                component_id=role,
                role=role,
                allocation_id=f"qstore-component.{role}",
                codec_id="qstore-int8-rowwise-fp32-scales-v1",
                layout_id="mrun-qstore-component-v1",
                physical_bytes=sum(blob.byte_count for blob in blobs),
                blobs=blobs,
            )
        )

    cfg = getattr(engine, "cfg", None)
    store = getattr(engine, "store", None)
    if not isinstance(cfg, Mapping) or store is None:
        raise NativeBackendBindingError("CUDA engine exposes no runtime config/store")
    dtype_name, dtype_bytes = _dtype_name_and_bytes(getattr(store, "compute_dtype", None))
    layers = _strict_positive_int(cfg.get("num_hidden_layers"), "CUDA layer count")
    attention_heads = _strict_positive_int(
        cfg.get("num_attention_heads"), "CUDA attention head count"
    )
    kv_heads = _strict_positive_int(
        cfg.get("num_key_value_heads", attention_heads), "CUDA KV head count"
    )
    hidden = _strict_positive_int(cfg.get("hidden_size"), "CUDA hidden size")
    head_dim = _strict_positive_int(
        cfg.get("head_dim", hidden // attention_heads), "CUDA head dimension"
    )
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    graph_fingerprint = _sha256(graph.fingerprint, "CUDA component graph custody")
    body_abi = getattr(graph, "body_abi", {})
    semantic_model = body_abi.get("semantic_sha256") if isinstance(body_abi, Mapping) else None
    max_context = min(
        _strict_positive_int(
            cfg.get("max_position_embeddings", getattr(engine, "max_seq_len", 0)),
            "CUDA model context",
        ),
        _strict_positive_int(getattr(engine, "max_seq_len", 0), "CUDA runtime context"),
    )
    return CompiledModelIdentity(
        model_name=str(getattr(engine, "name", graph.model_name)),
        architecture=str(getattr(engine, "arch", graph.architecture)),
        source_revision_sha256=graph_fingerprint,
        semantic_model_sha256=_sha256(semantic_model, "CUDA body ABI identity"),
        component_graph_sha256=graph_fingerprint,
        vocab_manifest_sha256=_sha256(
            getattr(composite, "vocab_manifest_sha256", None),
            "CUDA vocabulary identity",
        ),
        compiler_abi=f"{graph.schema}+dense-qstore-cuda-component-lowering-v1",
        components=tuple(components),
        operator_ids=_DENSE_DECODER_OPERATORS,
        state_abi=NATIVE_STATE_ABI,
        state_dtype=dtype_name,
        state_bytes_per_token=bytes_per_token,
        max_context_tokens=max_context,
        semantic_token_count=_strict_positive_int(
            int(getattr(engine, "semantic_token_count", 0)),
            "CUDA semantic token count",
        ),
    )


def describe_mlx_device(*, device_id: str = "metal:0") -> DeviceDescriptor:
    """Describe the current Apple unified-memory host for placement admission."""

    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - core dependency in production environments
        raise RuntimeError("MLX device discovery requires psutil") from exc
    memory = psutil.virtual_memory()
    attributes = (
        ("machine", platform.machine() or "unknown"),
        ("node", platform.node() or "unknown"),
        ("platform", platform.platform()),
        ("processor", platform.processor() or "unknown"),
    )
    return DeviceDescriptor(
        device_id=device_id,
        fabric="apple-gpu",
        memory_domain=MemoryDomain.UNIFIED,
        total_bytes=int(memory.total),
        available_bytes=int(memory.available),
        machine_fingerprint=_canonical_sha256(
            {"fabric": "apple-gpu", "attributes": dict(attributes)}
        ),
        attributes=attributes,
    )


def describe_cuda_device(index: int = 0) -> DeviceDescriptor:
    """Describe one live CUDA device and its currently available placement budget."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("CUDA index must be a non-negative integer")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - package requires torch
        raise RuntimeError("CUDA device discovery requires torch") from exc
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} is unavailable")
    props = torch.cuda.get_device_properties(index)
    with torch.cuda.device(index):
        available, total = torch.cuda.mem_get_info()
    attributes = (
        ("capability", f"{props.major}.{props.minor}"),
        ("name", str(props.name)),
        ("pci", str(getattr(props, "pci_bus_id", "unknown"))),
        ("total_memory", str(int(props.total_memory))),
    )
    return DeviceDescriptor(
        device_id=f"cuda:{index}",
        fabric="nvidia-cuda",
        memory_domain=MemoryDomain.CUDA,
        total_bytes=int(total),
        available_bytes=int(available),
        machine_fingerprint=_canonical_sha256(
            {"fabric": "nvidia-cuda", "attributes": dict(attributes)}
        ),
        attributes=attributes,
    )


class _BoundBackendBase:
    """One opened engine, one exact device, and auditable placement-to-open custody."""

    def __init__(
        self,
        *,
        engine: Any,
        model: CompiledModelIdentity,
        capabilities: BackendCapabilities,
        device: DeviceDescriptor,
        extra_resident_bytes: int,
        owns_engine: bool,
    ) -> None:
        self.engine = engine
        self.model = model
        self._capabilities = capabilities
        self.device = device
        self.extra_resident_bytes = max(0, int(extra_resident_bytes))
        self.owns_engine = bool(owns_engine)
        self._admitted: dict[str, WorkloadSpec] = {}
        self._opened = False

    def capabilities(self, device: DeviceDescriptor) -> BackendCapabilities:
        if device.fingerprint != self.device.fingerprint:
            raise NativeBackendBindingError("backend is bound to a different device snapshot")
        return self._capabilities

    def plan(
        self,
        model: CompiledModelIdentity,
        workload: WorkloadSpec,
        device: DeviceDescriptor,
        *,
        memory_budget_bytes: int | None = None,
    ) -> PlacementPlan:
        if model != self.model or model.fingerprint != self.model.fingerprint:
            raise NativeBackendBindingError("backend cannot plan a foreign compiled model")
        capabilities = self.capabilities(device)
        effective = replace(
            workload,
            workspace_bytes=workload.workspace_bytes + self.extra_resident_bytes,
        )
        plan = plan_resident_placement(
            model,
            effective,
            capabilities,
            device,
            memory_budget_bytes=memory_budget_bytes,
        )
        self._admitted[plan.fingerprint] = effective
        return plan

    def admitted_workload(self, placement: PlacementPlan) -> WorkloadSpec:
        try:
            return self._admitted[placement.fingerprint]
        except KeyError as exc:
            raise NativeBackendBindingError(
                "placement was not issued by this live backend binding"
            ) from exc

    def _claim_open(self, model: CompiledModelIdentity, placement: PlacementPlan) -> WorkloadSpec:
        if self._opened:
            raise NativeBackendBindingError("this bound native engine already has a runtime")
        if model != self.model or placement.model_fingerprint != model.fingerprint:
            raise NativeBackendBindingError("runtime open binds a foreign model/placement")
        workload = self.admitted_workload(placement)
        self._opened = True
        return workload


class MlxComponentExecutionBackend(_BoundBackendBase):
    """ExecutionBackend binding for a verified resident MLX/Metal component engine."""

    def __init__(
        self,
        engine: Any,
        device: DeviceDescriptor,
        *,
        state_layout: MlxStateLayout | None = None,
        kv_codec: MlxAffineKVCodec | None = None,
        paged_kv_page_size: int | None = None,
        paged_kv_page_count: int | None = None,
        paged_decode_attention: bool = False,
        prefill_chunk_size: int | None = None,
        owns_engine: bool = False,
        runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        if device.fabric != "apple-gpu" or device.memory_domain is not MemoryDomain.UNIFIED:
            raise NativeBackendBindingError("MLX backend requires an Apple unified-memory device")
        measured_source = state_layout or inspect_mlx_state_layout(engine)
        if kv_codec is not None and not isinstance(kv_codec, MlxAffineKVCodec):
            raise TypeError("kv_codec must be MlxAffineKVCodec or None")
        if (paged_kv_page_size is None) != (paged_kv_page_count is None):
            raise NativeBackendBindingError("paged MLX K/V requires both page size and page count")
        if type(paged_decode_attention) is not bool:
            raise TypeError("paged_decode_attention must be boolean")
        if paged_kv_page_size is not None:
            paged_kv_page_size = _strict_positive_int(
                paged_kv_page_size,
                "paged MLX K/V page size",
            )
            paged_kv_page_count = _strict_positive_int(
                paged_kv_page_count,
                "paged MLX K/V page count",
            )
        if kv_codec is not None and paged_kv_page_size is not None:
            raise NativeBackendBindingError(
                "paged BF16 K/V and quantized K/V are mutually exclusive storage contracts"
            )
        if paged_decode_attention:
            if paged_kv_page_size is None:
                raise NativeBackendBindingError(
                    "Metal paged decode requires explicit BF16 page size and page count"
                )
            if paged_kv_page_size & (paged_kv_page_size - 1):
                raise NativeBackendBindingError(
                    "Metal paged decode requires a power-of-two page size"
                )
            if prefill_chunk_size is not None:
                raise NativeBackendBindingError(
                    "Metal paged decode and chunked prefill require separate execution routes"
                )
            if str(getattr(engine, "arch", "")) not in {"qwen2", "mixtral"}:
                raise NativeBackendBindingError(
                    "Metal paged decode supports only Qwen2 and classic Mixtral"
                )
            if measured_source.dtype_name.lower() not in {"bf16", "bfloat16"}:
                raise NativeBackendBindingError("Metal paged decode requires BF16 K/V state")
            for layer_index, layer in enumerate(measured_source.layers):
                if (
                    layer.key_head_dim != layer.value_head_dim
                    or layer.key_head_dim % 32
                    or not 32 <= layer.key_head_dim <= 256
                    or str(layer.key_dtype).lower().rsplit(".", maxsplit=1)[-1]
                    not in {"bf16", "bfloat16"}
                    or str(layer.value_dtype).lower().rsplit(".", maxsplit=1)[-1]
                    not in {"bf16", "bfloat16"}
                ):
                    raise NativeBackendBindingError(
                        f"layer {layer_index} has unsupported Metal paged-decode geometry"
                    )
        if kv_codec is None:
            measured = measured_source
            state_abi = (
                MLX_PAGED_KV_CACHE_ABI if paged_kv_page_size is not None else NATIVE_STATE_ABI
            )
        else:
            bytes_per_token = 0
            for layer in measured_source.layers:
                bytes_per_token += layer.kv_heads * (
                    kv_codec.vector_bytes(
                        layer.key_head_dim,
                        source_element_bytes=layer.key_element_bytes,
                    )
                    + kv_codec.vector_bytes(
                        layer.value_head_dim,
                        source_element_bytes=layer.value_element_bytes,
                    )
                )
            measured = MlxStateLayout(
                layers=measured_source.layers,
                dtype_name=f"{kv_codec.codec_id}-{measured_source.dtype_name}",
                bytes_per_token=bytes_per_token,
            )
            state_abi = kv_codec.state_abi(measured_source.dtype_name)
        model, measured = compiled_identity_from_mlx_engine(
            engine,
            state_layout=measured,
            state_abi=state_abi,
        )
        if prefill_chunk_size is not None:
            prefill_chunk_size = _strict_positive_int(
                prefill_chunk_size,
                "MLX prefill chunk size",
            )
            if prefill_chunk_size > model.max_context_tokens:
                raise NativeBackendBindingError(
                    "MLX prefill chunk size exceeds the compiled context limit"
                )
        backend_id = str(getattr(engine, "backend", ""))
        experimental_runtime = getattr(engine, "experimental_runtime", False)
        if type(experimental_runtime) is not bool:
            raise NativeBackendBindingError(
                "MLX engine experimental_runtime declaration must be boolean"
            )
        capabilities = BackendCapabilities(
            backend_id=backend_id,
            backend_abi=MLX_BACKEND_ABI,
            implementation_version="3",
            fabric=device.fabric,
            memory_domain=device.memory_domain,
            architectures=(model.architecture,),
            operator_ids=model.operator_ids,
            codecs=_codec_capabilities(model.components),
            state_abis=(model.state_abi,),
            output_modes=(
                OutputMode.NEXT_TOKEN_ARGMAX,
                OutputMode.NEXT_TOKEN_SAMPLE,
            ),
            numerical_contracts=(str(getattr(engine, "numerical_contract", "")),),
            max_context_tokens=model.max_context_tokens,
            max_batch_size=1,
            max_verify_tokens=max(1, model.max_context_tokens - 1),
            transactional_state=True,
            scratch_only_steps=True,
            independently_committable_rows=True,
            supports_ragged_batches=False,
            telemetry_counters=(
                "committed_tokens",
                "device_to_host_bytes",
                "greedy_block_verifications",
                "greedy_block_tokens",
                "greedy_block_commits",
                "greedy_block_abandons",
                "greedy_block_accepted_tokens",
                "greedy_block_rejected_tokens",
                "greedy_block_failures",
                "kv_resident_bytes",
                "workspace_peak_bytes",
            ),
            promotion_status=(
                PromotionStatus.EXPERIMENTAL
                if experimental_runtime or kv_codec is not None or paged_kv_page_size is not None
                else PromotionStatus.CANDIDATE
            ),
        )
        artifact_bytes = model.physical_allocation_bytes
        measured_engine_bytes = getattr(engine, "mlx_engine_active_memory_bytes_at_load", None)
        active_bytes = max(
            0,
            int(measured_engine_bytes)
            if measured_engine_bytes is not None
            else int(float(getattr(engine, "mlx_active_memory_mb_at_load", 0.0)) * 1024 * 1024),
        )
        super().__init__(
            engine=engine,
            model=model,
            capabilities=capabilities,
            device=device,
            extra_resident_bytes=max(0, active_bytes - artifact_bytes),
            owns_engine=owns_engine,
        )
        self.state_layout = measured
        self.kv_codec = kv_codec
        self.paged_kv_page_size = paged_kv_page_size
        self.paged_kv_page_count = paged_kv_page_count
        self.paged_decode_attention = paged_decode_attention
        self.prefill_chunk_size = prefill_chunk_size
        self._runtime_factory = runtime_factory or MlxNativeRuntime.bind

    @property
    def paged_kv_physical_bytes(self) -> int:
        if self.paged_kv_page_size is None or self.paged_kv_page_count is None:
            return 0
        return (
            self.paged_kv_page_size * self.paged_kv_page_count * self.state_layout.bytes_per_token
        )

    def plan(
        self,
        model: CompiledModelIdentity,
        workload: WorkloadSpec,
        device: DeviceDescriptor,
        *,
        memory_budget_bytes: int | None = None,
    ) -> PlacementPlan:
        if self.paged_kv_page_size is not None:
            assert self.paged_kv_page_count is not None
            pool_tokens = self.paged_kv_page_size * self.paged_kv_page_count
            if pool_tokens < workload.max_context_tokens:
                raise NativeBackendBindingError(
                    "paged MLX K/V pool cannot hold one maximum-context state"
                )
            ordinary_state_bytes = (
                workload.max_batch_size
                * workload.max_context_tokens
                * self.state_layout.bytes_per_token
            )
            pool_delta = self.paged_kv_physical_bytes - ordinary_state_bytes
            if pool_delta < 0:
                raise NativeBackendBindingError(
                    "paged MLX K/V accounting is smaller than the admitted state"
                )
            workload = replace(
                workload,
                workspace_bytes=workload.workspace_bytes + pool_delta,
            )
        return super().plan(
            model,
            workload,
            device,
            memory_budget_bytes=memory_budget_bytes,
        )

    def _quantized_cache_factory(self, capacity: int) -> tuple[Any, ...]:
        codec = self.kv_codec
        if codec is None:
            raise NativeBackendBindingError("quantized cache factory has no bound codec")
        from mlx_lm.models.cache import create_attention_mask

        mx = self.engine._mx
        return tuple(
            FixedMlxQuantizedKVCache(
                mx=mx,
                create_attention_mask=create_attention_mask,
                capacity=capacity,
                kv_heads=layer.kv_heads,
                key_head_dim=layer.key_head_dim,
                value_head_dim=layer.value_head_dim,
                key_dtype=layer.key_dtype,
                value_dtype=layer.value_dtype,
                codec=codec,
            )
            for layer in self.state_layout.layers
        )

    def _paged_cache_factory(self) -> MlxPagedKVCacheFactory:
        if self.paged_kv_page_size is None or self.paged_kv_page_count is None:
            raise NativeBackendBindingError("paged cache factory has no bound pool geometry")
        from mlx_lm.models.cache import create_attention_mask

        mx = self.engine._mx
        pools: list[MlxKVPagePool] = []
        try:
            for layer_index, layer in enumerate(self.state_layout.layers):
                pools.append(
                    MlxKVPagePool(
                        mx=mx,
                        create_attention_mask=create_attention_mask,
                        page_size=self.paged_kv_page_size,
                        page_count=self.paged_kv_page_count,
                        kv_heads=layer.kv_heads,
                        key_head_dim=layer.key_head_dim,
                        value_head_dim=layer.value_head_dim,
                        key_dtype=layer.key_dtype,
                        value_dtype=layer.value_dtype,
                        paged_decode_attention=self.paged_decode_attention,
                        pool_id=f"mlx-kv-layer-{layer_index}-{uuid4().hex}",
                    )
                )
        except BaseException as operation_error:
            for pool in reversed(pools):
                try:
                    pool.close()
                except BaseException as cleanup_error:
                    operation_error.add_note(
                        f"paged MLX K/V pool cleanup also failed: {cleanup_error}"
                    )
            raise
        return MlxPagedKVCacheFactory(tuple(pools))

    def _paged_decode_attention_lane(self) -> MlxPagedDecodeAttentionLane:
        if not self.paged_decode_attention:
            raise NativeBackendBindingError("Metal paged-decode lane was not requested")
        assert self.paged_kv_page_size is not None
        assert self.paged_kv_page_count is not None
        return MlxPagedDecodeAttentionLane(
            self.engine.model,
            mx=self.engine._mx,
            architecture=self.model.architecture,
            state_layout=self.state_layout,
            page_size=self.paged_kv_page_size,
            page_count=self.paged_kv_page_count,
            base_numerical_contract=str(getattr(self.engine, "numerical_contract", "")),
        )

    def open(
        self,
        model: CompiledModelIdentity,
        placement: PlacementPlan,
    ) -> MlxNativeRuntime:
        workload = self._claim_open(model, placement)
        paged_factory: MlxPagedKVCacheFactory | None = None
        paged_decode_lane: MlxPagedDecodeAttentionLane | None = None
        try:
            arguments: dict[str, Any] = {
                "model": model,
                "workload": workload,
                "capabilities": self._capabilities,
                "device": self.device,
                "placement": placement,
                "state_layout": self.state_layout,
                "prefill_chunk_size": self.prefill_chunk_size,
                "owns_engine": self.owns_engine,
            }
            if self.kv_codec is not None:
                arguments["cache_factory"] = self._quantized_cache_factory
            elif self.paged_kv_page_size is not None:
                paged_factory = self._paged_cache_factory()
                arguments["cache_factory"] = paged_factory
                arguments["owns_cache_factory"] = True
                if self.paged_decode_attention:
                    paged_decode_lane = self._paged_decode_attention_lane()
                    arguments["paged_decode_attention_lane"] = paged_decode_lane
            return self._runtime_factory(
                self.engine,
                **arguments,
            )
        except BaseException:
            if paged_decode_lane is not None:
                try:
                    paged_decode_lane.close()
                except BaseException:
                    pass
            if paged_factory is not None:
                try:
                    paged_factory.close()
                except BaseException:
                    pass
            self._opened = False
            raise


class MlxMambaComponentExecutionBackend(_BoundBackendBase):
    """ExecutionBackend for Mamba's fixed-size MLX recurrence.

    This binding is intentionally separate from :class:`MlxComponentExecutionBackend`: it has no
    attention operators, no K/V layout probe, and context admission does not multiply mutable
    state by sequence length.  Two fixed-state scratch copies are reserved so an arbitrary
    accepted prefix can be committed exactly without ever advancing committed state in-place.
    """

    def __init__(
        self,
        engine: Any,
        device: DeviceDescriptor,
        *,
        prefill_chunk_size: int | None = None,
        owns_engine: bool = False,
        runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        if device.fabric != "apple-gpu" or device.memory_domain is not MemoryDomain.UNIFIED:
            raise NativeBackendBindingError(
                "MLX Mamba backend requires an Apple unified-memory device"
            )
        if str(getattr(engine, "arch", "")) != "mamba":
            raise NativeBackendBindingError("fixed-state MLX binding requires Mamba")
        model = compiled_identity_from_mlx_mamba_engine(engine)
        if prefill_chunk_size is not None:
            prefill_chunk_size = _strict_positive_int(
                prefill_chunk_size,
                "Mamba prefill chunk size",
            )
            if prefill_chunk_size > model.max_context_tokens:
                raise NativeBackendBindingError(
                    "Mamba prefill chunk size exceeds the compiled context limit"
                )
        try:
            prefill_execution_shape = build_mlx_mamba_prefill_execution_shape(
                getattr(engine.artifact, "config", {}),
                chunk_size=prefill_chunk_size,
                base_numerical_contract=str(getattr(engine, "numerical_contract", "")),
                unchunked_promotion_status=PromotionStatus.EXPERIMENTAL,
            )
        except (TypeError, ValueError) as exc:
            raise NativeBackendBindingError(
                "Mamba prefill execution shape is not canonical"
            ) from exc
        backend_id = str(getattr(engine, "backend", ""))
        capabilities = BackendCapabilities(
            backend_id=backend_id,
            backend_abi=MLX_MAMBA_BACKEND_ABI,
            implementation_version="1",
            fabric=device.fabric,
            memory_domain=device.memory_domain,
            architectures=(model.architecture,),
            operator_ids=model.operator_ids,
            codecs=_codec_capabilities(model.components),
            state_abis=(model.state_abi,),
            output_modes=(
                OutputMode.NEXT_TOKEN_ARGMAX,
                OutputMode.NEXT_TOKEN_SAMPLE,
            ),
            numerical_contracts=(str(getattr(engine, "numerical_contract", "")),),
            max_context_tokens=model.max_context_tokens,
            max_batch_size=1,
            max_verify_tokens=(
                prefill_chunk_size
                if prefill_chunk_size is not None
                else max(1, model.max_context_tokens - 1)
            ),
            transactional_state=True,
            scratch_only_steps=True,
            independently_committable_rows=True,
            supports_ragged_batches=False,
            telemetry_counters=(
                "committed_tokens",
                "device_to_host_bytes",
                "mamba_prefix_replay_forwards",
                "mamba_prefix_replay_tokens",
                "mamba_recurrent_state_logical_bytes",
                "mamba_recurrent_state_resident_bytes",
                "mamba_state_fork_bytes_copied",
                "mamba_state_fork_tokens",
                "mamba_state_forks",
                "mamba_cleanup_failures",
                "mamba_chunk_size",
                "mamba_chunk_tensor_workspace_bytes",
                "mamba_chunked_prefill_calls",
                "mamba_prefill_chunk_failures",
                "mamba_prefill_chunks",
                "mamba_prefix_replay_chunks",
                "workspace_peak_bytes",
            ),
            # Functional correctness is necessary but not sufficient for promotion.  This lane
            # remains explicit until real-model trajectory and throughput evidence is attached.
            promotion_status=PromotionStatus.EXPERIMENTAL,
        )
        artifact_bytes = model.physical_allocation_bytes
        measured_engine_bytes = getattr(engine, "mlx_engine_active_memory_bytes_at_load", None)
        active_bytes = max(
            0,
            int(measured_engine_bytes)
            if measured_engine_bytes is not None
            else int(float(getattr(engine, "mlx_active_memory_mb_at_load", 0.0)) * 1024 * 1024),
        )
        super().__init__(
            engine=engine,
            model=model,
            capabilities=capabilities,
            device=device,
            extra_resident_bytes=max(0, active_bytes - artifact_bytes),
            owns_engine=owns_engine,
        )
        self._runtime_factory = runtime_factory or MlxMambaRuntime.bind
        self._engine_model_identity = id(getattr(engine, "model", None))
        self._prefill_execution_shape = prefill_execution_shape

    @property
    def prefill_execution_shape(self) -> MlxMambaPrefillExecutionShape:
        """The immutable chunk/workspace identity sealed by this backend binding."""

        return self._prefill_execution_shape

    def _assert_binding_unchanged(self) -> None:
        current = compiled_identity_from_mlx_mamba_engine(self.engine)
        if current != self.model or current.fingerprint != self.model.fingerprint:
            raise NativeBackendBindingError("Mamba engine identity changed after backend binding")
        if id(getattr(self.engine, "model", None)) != self._engine_model_identity:
            raise NativeBackendBindingError("Mamba loaded model changed after backend binding")
        if str(getattr(self.engine, "backend", "")) != self._capabilities.backend_id:
            raise NativeBackendBindingError("Mamba engine backend changed after binding")
        if (
            str(getattr(self.engine, "numerical_contract", "")),
        ) != self._capabilities.numerical_contracts:
            raise NativeBackendBindingError("Mamba engine numerical contract changed after binding")
        try:
            current_shape = build_mlx_mamba_prefill_execution_shape(
                getattr(self.engine.artifact, "config", {}),
                chunk_size=self.prefill_execution_shape.chunk_size,
                base_numerical_contract=str(getattr(self.engine, "numerical_contract", "")),
                unchunked_promotion_status=PromotionStatus.EXPERIMENTAL,
            )
        except (TypeError, ValueError) as exc:
            raise NativeBackendBindingError(
                "Mamba prefill execution shape changed after backend binding"
            ) from exc
        if (
            current_shape != self.prefill_execution_shape
            or current_shape.fingerprint != self.prefill_execution_shape.fingerprint
        ):
            raise NativeBackendBindingError(
                "Mamba prefill execution shape changed after backend binding"
            )

    def plan(
        self,
        model: CompiledModelIdentity,
        workload: WorkloadSpec,
        device: DeviceDescriptor,
        *,
        memory_budget_bytes: int | None = None,
    ) -> PlacementPlan:
        self._assert_binding_unchanged()
        scratch_bytes = (
            2 * self.model.state_fixed_bytes_per_row + self.prefill_execution_shape.workspace_bytes
        )
        effective = replace(
            workload,
            workspace_bytes=workload.workspace_bytes + scratch_bytes,
        )
        return super().plan(
            model,
            effective,
            device,
            memory_budget_bytes=memory_budget_bytes,
        )

    def open(
        self,
        model: CompiledModelIdentity,
        placement: PlacementPlan,
    ) -> MlxMambaRuntime:
        self._assert_binding_unchanged()
        workload = self._claim_open(model, placement)
        try:
            return self._runtime_factory(
                self.engine,
                model=model,
                workload=workload,
                capabilities=self._capabilities,
                device=self.device,
                placement=placement,
                prefill_execution_shape=self.prefill_execution_shape,
                owns_engine=self.owns_engine,
            )
        except BaseException:
            self._opened = False
            raise


class DenseCudaComponentExecutionBackend(_BoundBackendBase):
    """ExecutionBackend binding for a verified fully resident CUDA component engine."""

    def __init__(
        self,
        engine: Any,
        device: DeviceDescriptor,
        *,
        max_batch_size: int = 1,
        owns_engine: bool = False,
        runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        if device.fabric != "nvidia-cuda" or device.memory_domain is not MemoryDomain.CUDA:
            raise NativeBackendBindingError("dense CUDA backend requires a CUDA device")
        max_batch_size = _strict_positive_int(max_batch_size, "CUDA maximum batch size")
        model = compiled_identity_from_dense_cuda_engine(engine)
        self._body_workspace_geometry = _dense_cuda_body_workspace_geometry(engine)
        composite = getattr(engine, "composite_store", None)
        direct_artifact = getattr(engine, "direct_artifact", None)
        if direct_artifact is not None:
            snapshot = engine.store.snapshot()
            if not bool(snapshot.get("fully_resident")):
                raise NativeBackendBindingError(
                    "direct CUDA resident binding requires every native component allocation"
                )
        else:
            if composite is None:
                raise NativeBackendBindingError(
                    "legacy CUDA native binding requires a component-QStore graph"
                )
            snapshot = composite.snapshot()
            providers = snapshot.get("providers")
            expected_roles = set(composite.graph.components)
            if not isinstance(providers, Mapping) or set(providers) != expected_roles:
                raise NativeBackendBindingError(
                    "CUDA resident binding requires every component provider to be opened"
                )
            nonresident = sorted(
                role
                for role, provider in providers.items()
                if not isinstance(provider, Mapping) or not bool(provider.get("fully_resident"))
            )
            if nonresident:
                raise NativeBackendBindingError(
                    f"CUDA resident binding has nonresident component roles: {nonresident!r}"
                )
        backend_id = str(getattr(engine, "backend", ""))
        target = getattr(engine, "target", None)
        decode_attention_mode = str(
            getattr(target, "decode_attention_mode", "established")
        )
        decode_attention_tile = getattr(target, "decode_attention_tile", 64)
        if decode_attention_mode not in {
            "established",
            "segmented-flash-gqa-decode-v1",
        }:
            raise NativeBackendBindingError(
                f"CUDA binding received unknown decode attention mode {decode_attention_mode!r}"
            )
        if (
            isinstance(decode_attention_tile, bool)
            or not isinstance(decode_attention_tile, int)
            or decode_attention_tile < 16
            or decode_attention_tile > 256
            or decode_attention_tile & (decode_attention_tile - 1)
        ):
            raise NativeBackendBindingError(
                "CUDA decode attention tile is not a sealed power of two"
            )
        decode_contract_suffix = (
            ""
            if decode_attention_mode == "established"
            else "+segmented-flash-gqa-decode-v1"
        )
        if decode_attention_mode != "established" and getattr(
            engine.store, "require_triton", None
        ) is not True:
            raise NativeBackendBindingError("segmented CUDA decode attention requires Triton")
        body_fusion_mode = str(getattr(target, "body_fusion_mode", "established"))
        if body_fusion_mode not in {"established", "residual-rms-swiglu-v1"}:
            raise NativeBackendBindingError(
                f"CUDA binding received unknown body fusion mode {body_fusion_mode!r}"
            )
        body_contract_suffix = (
            "" if body_fusion_mode == "established" else "+residual-rms-swiglu-v1"
        )
        if body_fusion_mode != "established" and getattr(
            engine.store, "require_triton", None
        ) is not True:
            raise NativeBackendBindingError("fused CUDA transformer body requires Triton")
        head_execution_mode = str(getattr(engine, "head_execution_mode", "exact-fp32-row-blocks"))
        compact_head = head_execution_mode == ("semantic-prefix-w8a16-top2-fp32-rerank")
        if head_execution_mode not in {
            "exact-fp32-row-blocks",
            "semantic-prefix-w8a16-top2-fp32-rerank",
        }:
            raise NativeBackendBindingError(
                f"CUDA binding received unknown head execution mode {head_execution_mode!r}"
            )
        compact_backend = backend_id == "cuda-source-int8-compact-head"
        if compact_backend != compact_head:
            raise NativeBackendBindingError(
                "CUDA compact backend and head execution mode must be selected together"
            )
        head_execution_abi = str(getattr(engine, "head_execution_abi", "exact-fp32-row-blocks-v1"))
        numerical_contract = str(getattr(engine, "numerical_contract", ""))
        if compact_head:
            expected_head_abi = "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
            expected_contract = (
                "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1+"
                "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
                f"{decode_contract_suffix}"
                f"{body_contract_suffix}"
            )
            if direct_artifact is None:
                raise NativeBackendBindingError(
                    "compact CUDA head is authorized only for the direct-source compact backend"
                )
            if head_execution_abi != expected_head_abi or numerical_contract != expected_contract:
                raise NativeBackendBindingError(
                    "compact CUDA backend mode, ABI, and numerical contract are inconsistent"
                )
            if not bool(
                getattr(getattr(engine, "target", None), "experimental_reranked_head", False)
            ):
                raise NativeBackendBindingError(
                    "compact CUDA head engine did not bind reranked target execution"
                )
            if bool(snapshot.get("resident_exact_head")) or snapshot.get(
                "resident_exact_head_bytes"
            ) not in {0, None}:
                raise NativeBackendBindingError(
                    "compact CUDA head binding forbids an expanded resident FP32 head"
                )
            if getattr(engine.store, "require_triton", None) is not True:
                raise NativeBackendBindingError("compact CUDA head binding requires Triton")
            if not callable(getattr(engine, "forward_last_top1", None)):
                raise NativeBackendBindingError(
                    "compact CUDA head binding requires bounded final-row execution"
                )
            output_modes = (OutputMode.NEXT_TOKEN_ARGMAX,)
            resident_head_bytes = 0
            from mrun.engine.kernels.dense_qstore_cuda import (
                reranked_argmax_workspace_bytes,
            )

            compact_head_workspace_bytes = reranked_argmax_workspace_bytes(
                rows=max_batch_size,
                in_features=_strict_positive_int(
                    getattr(engine, "hidden", None), "compact CUDA hidden width"
                ),
                semantic_row_count=model.semantic_token_count,
            )
            promotion_status = PromotionStatus.EXPERIMENTAL
            implementation_version = (
                "6" if body_contract_suffix else "5" if decode_contract_suffix else "4"
            )
        else:
            if bool(getattr(getattr(engine, "target", None), "experimental_reranked_head", False)):
                raise NativeBackendBindingError(
                    "exact CUDA binding cannot execute the reranked compact head"
                )
            if direct_artifact is not None and (
                backend_id != "cuda-source-int8"
                or head_execution_abi != "exact-fp32-row-blocks-v1"
                or numerical_contract
                != (
                    "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1"
                    f"{decode_contract_suffix}"
                    f"{body_contract_suffix}"
                )
            ):
                raise NativeBackendBindingError(
                    "exact direct CUDA backend mode, ABI, and numerical contract are inconsistent"
                )
            if not bool(snapshot.get("resident_exact_head")):
                raise NativeBackendBindingError(
                    "CUDA production argmax binding requires the exact FP32 head to be resident"
                )
            resident_head_bytes = _strict_positive_int(
                snapshot.get("resident_exact_head_bytes"), "resident exact head bytes"
            )
            output_modes = (
                OutputMode.NEXT_TOKEN_ARGMAX,
                OutputMode.NEXT_TOKEN_SAMPLE,
            )
            compact_head_workspace_bytes = 0
            promotion_status = PromotionStatus.CANDIDATE
            implementation_version = (
                "6" if body_contract_suffix else "5" if decode_contract_suffix else "3"
            )
        capabilities = BackendCapabilities(
            backend_id=backend_id,
            backend_abi=DENSE_CUDA_BACKEND_ABI,
            implementation_version=implementation_version,
            fabric=device.fabric,
            memory_domain=device.memory_domain,
            architectures=(model.architecture,),
            operator_ids=model.operator_ids,
            codecs=_codec_capabilities(model.components),
            state_abis=(model.state_abi,),
            output_modes=output_modes,
            numerical_contracts=(numerical_contract,),
            max_context_tokens=model.max_context_tokens,
            max_batch_size=max_batch_size,
            max_verify_tokens=1,
            transactional_state=True,
            scratch_only_steps=True,
            independently_committable_rows=True,
            supports_ragged_batches=False,
            telemetry_counters=(
                "admitted_body_workspace_bytes",
                "committed_tokens",
                "device_to_host_bytes",
                "kv_resident_bytes",
                "workspace_peak_bytes",
            ),
            promotion_status=promotion_status,
        )
        super().__init__(
            engine=engine,
            model=model,
            capabilities=capabilities,
            device=device,
            extra_resident_bytes=resident_head_bytes + compact_head_workspace_bytes,
            owns_engine=owns_engine,
        )
        self.compact_head_workspace_bytes = compact_head_workspace_bytes
        self._binding_seal = self._current_binding_seal()
        self._runtime_factory = runtime_factory or DenseCudaNativeRuntime.bind

    def _current_binding_seal(self) -> tuple[Any, ...]:
        direct_artifact = getattr(self.engine, "direct_artifact", None)
        composite = getattr(self.engine, "composite_store", None)
        snapshot = (
            self.engine.store.snapshot() if direct_artifact is not None else composite.snapshot()
        )
        return (
            id(self.engine),
            id(getattr(self.engine, "store", None)),
            str(getattr(self.engine, "backend", "")),
            str(getattr(self.engine, "head_execution_mode", "exact-fp32-row-blocks")),
            str(getattr(self.engine, "head_execution_abi", "exact-fp32-row-blocks-v1")),
            str(getattr(self.engine, "numerical_contract", "")),
            bool(
                getattr(
                    getattr(self.engine, "target", None),
                    "experimental_reranked_head",
                    False,
                )
            ),
            getattr(getattr(self.engine, "store", None), "require_triton", None),
            str(
                getattr(
                    getattr(self.engine, "target", None),
                    "decode_attention_mode",
                    "established",
                )
            ),
            int(
                getattr(
                    getattr(self.engine, "target", None),
                    "decode_attention_tile",
                    64,
                )
            ),
            str(
                getattr(
                    getattr(self.engine, "target", None),
                    "body_fusion_mode",
                    "established",
                )
            ),
            bool(snapshot.get("fully_resident", True)),
            bool(snapshot.get("resident_exact_head")),
            int(snapshot.get("resident_exact_head_bytes", 0)),
            getattr(direct_artifact, "artifact_sha256", None),
            _dense_cuda_body_workspace_geometry(self.engine),
        )

    def _assert_binding_unchanged(self) -> None:
        guard = getattr(self.engine, "assert_content_identity_unchanged", None)
        if callable(guard):
            guard()
        if self._current_binding_seal() != self._binding_seal:
            raise NativeBackendBindingError(
                "CUDA engine residency or head execution identity changed after binding"
            )

    def capabilities(self, device: DeviceDescriptor) -> BackendCapabilities:
        self._assert_binding_unchanged()
        return super().capabilities(device)

    def body_workspace_bytes(self, workload: WorkloadSpec) -> int:
        """Return the conservative body scratch charge for this workload shape."""

        if not isinstance(workload, WorkloadSpec):
            raise TypeError("CUDA body workspace requires WorkloadSpec")
        mode = str(
            getattr(getattr(self.engine, "target", None), "decode_attention_mode", "established")
        )
        if mode == "segmented-flash-gqa-decode-v1":
            return _dense_cuda_segmented_decode_workspace_bytes(
                self._body_workspace_geometry,
                max_batch_size=workload.max_batch_size,
                max_context_tokens=workload.max_context_tokens,
            )
        if mode != "established":
            raise NativeBackendBindingError(f"unknown CUDA decode attention mode {mode!r}")
        return _dense_cuda_body_workspace_bytes(
            self._body_workspace_geometry,
            max_batch_size=workload.max_batch_size,
            max_context_tokens=workload.max_context_tokens,
        )

    def plan(
        self,
        model: CompiledModelIdentity,
        workload: WorkloadSpec,
        device: DeviceDescriptor,
        *,
        memory_budget_bytes: int | None = None,
    ) -> PlacementPlan:
        self._assert_binding_unchanged()
        body_workspace_bytes = self.body_workspace_bytes(workload)
        effective = replace(
            workload,
            workspace_bytes=workload.workspace_bytes + body_workspace_bytes,
        )
        return super().plan(
            model,
            effective,
            device,
            memory_budget_bytes=memory_budget_bytes,
        )

    def open(
        self,
        model: CompiledModelIdentity,
        placement: PlacementPlan,
    ) -> DenseCudaNativeRuntime:
        self._assert_binding_unchanged()
        workload = self._claim_open(model, placement)
        admitted_body_workspace_bytes = self.body_workspace_bytes(workload)
        try:
            return self._runtime_factory(
                self.engine,
                model=model,
                workload=workload,
                capabilities=self._capabilities,
                device=self.device,
                placement=placement,
                admitted_body_workspace_bytes=admitted_body_workspace_bytes,
                owns_engine=self.owns_engine,
            )
        except BaseException:
            self._opened = False
            raise


__all__ = [
    "DENSE_CUDA_BACKEND_ABI",
    "MLX_BACKEND_ABI",
    "MLX_MAMBA_BACKEND_ABI",
    "MAMBA_RECURRENT_STATE_ABI",
    "NATIVE_STATE_ABI",
    "DenseCudaComponentExecutionBackend",
    "MlxComponentExecutionBackend",
    "MlxMambaComponentExecutionBackend",
    "NativeBackendBindingError",
    "compiled_identity_from_dense_cuda_engine",
    "compiled_identity_from_mlx_engine",
    "compiled_identity_from_mlx_mamba_engine",
    "describe_cuda_device",
    "describe_mlx_device",
    "dense_cuda_body_workspace_bytes",
]
