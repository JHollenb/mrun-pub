"""Fail-closed dense HF adapters for registered decoder-only model families.

Each installed adapter owns an exact tensor schema.  Regexes are used only for layer-indexed names
whose complete layer domain is known from config; anything else is an unexplained source tensor.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import canonical_json, canonical_sha256
from .errors import (
    AliasEvidenceError,
    CodecError,
    ConfigurationError,
    CoverageError,
    IOContractError,
    UnsupportedVariantError,
)
from .ir import (
    IOIR,
    AliasClassIR,
    AliasEvidenceIR,
    BoundAssetIR,
    CapacityEquationIR,
    ChatTemplateIR,
    CodecBindingIR,
    CommitProtocolIR,
    IRBundle,
    LogicalParameterRefIR,
    ModelDimensionsIR,
    ModelIR,
    NumericalSemanticsIR,
    OperationIR,
    OutputSpaceIR,
    PhysicalAllocationIR,
    PhysicalWeightIR,
    PortIR,
    RowMapperIR,
    SpecialTokenIR,
    StateIR,
    StateOperationIR,
    StateSlotIR,
    TensorClassificationIR,
    TensorViewIR,
    TokenSpaceIR,
    ViewTransformIR,
)
from .matching import (
    AdapterRegistry,
    MatchEvidence,
    MatchResult,
    UnsupportedFeature,
)
from .source import FrozenSourceBundle
from .tensor_index import TensorIndex, TensorRecord

_RAW_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ROTARY_BUFFER_RE = re.compile(
    r"^model(?:\.layers\.(\d+)\.self_attn)?\.rotary_emb\."
    r"(inv_freq|original_inv_freq|cos_cached|sin_cached)$"
)
_COMMON_CONFIG_KEYS = frozenset(
    {
        "_name_or_path",
        "add_cross_attention",
        "architectures",
        "auto_map",
        "bad_words_ids",
        "begin_suppress_tokens",
        "bos_token_id",
        "chunk_size_feed_forward",
        "cross_attention_hidden_size",
        "decoder_start_token_id",
        "dtype",
        "early_stopping",
        "encoder_no_repeat_ngram_size",
        "eos_token_id",
        "exponential_decay_length_penalty",
        "finetuning_task",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "id2label",
        "initializer_range",
        "is_decoder",
        "is_encoder_decoder",
        "label2id",
        "length_penalty",
        "max_length",
        "min_length",
        "model_type",
        "no_repeat_ngram_size",
        "num_beam_groups",
        "num_beams",
        "num_labels",
        "num_return_sequences",
        "output_attentions",
        "output_hidden_states",
        "output_scores",
        "pad_token_id",
        "prefix",
        "problem_type",
        "pruned_heads",
        "quantization_config",
        "remove_invalid_values",
        "repetition_penalty",
        "return_dict",
        "return_dict_in_generate",
        "sep_token_id",
        "suppress_tokens",
        "task_specific_params",
        "temperature",
        "tie_encoder_decoder",
        "tie_word_embeddings",
        "tokenizer_class",
        "top_k",
        "top_p",
        "torch_dtype",
        "torchscript",
        "transformers_version",
        "typical_p",
        "use_bfloat16",
        "use_cache",
    }
)
_DENSE_CONFIG_KEYS = frozenset(
    {
        "attention_bias",
        "attention_dropout",
        "head_dim",
        "hidden_act",
        "hidden_dropout",
        "hidden_size",
        "intermediate_size",
        "layer_types",
        "max_position_embeddings",
        "max_window_layers",
        "mlp_bias",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "partial_rotary_factor",
        "pretraining_tp",
        "rms_norm_eps",
        "rope_parameters",
        "rope_scaling",
        "rope_theta",
        "sliding_window",
        "use_sliding_window",
        "use_mrope",
        "vocab_size",
    }
)
_GPT_NEOX_CONFIG_KEYS = frozenset(
    {
        "attention_bias",
        "attention_dropout",
        "classifier_dropout",
        "head_dim",
        "hidden_act",
        "hidden_dropout",
        "hidden_size",
        "intermediate_size",
        "is_decoder",
        "layer_norm_eps",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rope_parameters",
        "rope_scaling",
        "rotary_emb_base",
        "rotary_pct",
        "use_parallel_residual",
        "vocab_size",
    }
)
_GPT2_CONFIG_KEYS = frozenset(
    {
        "_num_labels",
        "activation_function",
        "attn_pdrop",
        "embd_pdrop",
        "layer_norm_epsilon",
        "n_ctx",
        "n_embd",
        "n_head",
        "n_inner",
        "n_layer",
        "n_positions",
        "reorder_and_upcast_attn",
        "resid_pdrop",
        "scale_attn_by_inverse_layer_idx",
        "scale_attn_weights",
        "summary_activation",
        "summary_first_dropout",
        "summary_proj_to_labels",
        "summary_type",
        "summary_use_proj",
        "vocab_size",
    }
)
_PHI_CONFIG_KEYS = frozenset(
    {
        "attention_dropout",
        "embd_pdrop",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "intermediate_size",
        "layer_norm_eps",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "partial_rotary_factor",
        "qk_layernorm",
        "resid_pdrop",
        "rope_parameters",
        "rope_scaling",
        "rope_theta",
        "vocab_size",
    }
)
_MIXTRAL_CONFIG_KEYS = frozenset(
    {
        "num_experts_per_tok",
        "num_local_experts",
        "output_router_logits",
        "router_aux_loss_coef",
        "router_jitter_noise",
    }
)
_MAMBA_CONFIG_KEYS = frozenset(
    {
        "conv_kernel",
        "d_inner",
        "d_model",
        "expand",
        "fused_add_norm",
        "hidden_act",
        "hidden_size",
        "intermediate_size",
        "layer_norm_epsilon",
        "mixer_rms_eps",
        "n_layer",
        "num_hidden_layers",
        "pad_vocab_size_multiple",
        "rescale_prenorm_residual",
        "residual_in_fp32",
        "rms_norm",
        "ssm_cfg",
        "state_size",
        "time_step_floor",
        "time_step_init_scheme",
        "time_step_max",
        "time_step_min",
        "time_step_rank",
        "time_step_scale",
        "use_associative_scan",
        "use_bias",
        "use_cache",
        "use_conv_bias",
        "use_mambapy",
        "vocab_size",
    }
)


@dataclass(frozen=True, slots=True)
class _DenseConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class _MixtralConfig(_DenseConfig):
    num_local_experts: int
    num_experts_per_tok: int
    router_jitter_noise: float
    output_router_logits: bool


@dataclass(frozen=True, slots=True)
class _GPTNeoXConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    rope_theta: float
    rotary_pct: float
    tie_word_embeddings: bool
    attention_bias: bool
    use_parallel_residual: bool
    hidden_act: str


@dataclass(frozen=True, slots=True)
class _GPT2Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    tie_word_embeddings: bool
    activation_function: str

    @property
    def query_width(self) -> int:
        return self.hidden_size

    @property
    def key_value_width(self) -> int:
        return self.hidden_size


@dataclass(frozen=True, slots=True)
class _GemmaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    hidden_activation: str

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class _PhiConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    tie_word_embeddings: bool
    hidden_act: str

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class _MambaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    state_size: int
    conv_kernel: int
    time_step_rank: int
    vocab_size: int
    layer_norm_epsilon: float
    tie_word_embeddings: bool
    use_bias: bool
    use_conv_bias: bool
    residual_in_fp32: bool


@dataclass(frozen=True, slots=True)
class _TensorSpec:
    source_name: str
    logical_name: str
    shape: tuple[int, ...]
    semantic_role: str
    parameter_kind: str = "parameter"
    rule_id: str = "dense-decoder-parameter-v1"


@dataclass(frozen=True, slots=True)
class _MappedContext:
    config: _DenseConfig | _GPTNeoXConfig | _GPT2Config | _GemmaConfig | _PhiConfig | _MambaConfig
    input_rows: int
    output_rows: int
    rotary_parameters_by_layer: tuple[tuple[str, ...], ...]


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ConfigurationError(
            f"config field {key!r} must be a positive integer",
            details={"field": key, "observed": value},
        )
    return value


def _optional_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ConfigurationError(
            f"config field {key!r} must be a boolean",
            details={"field": key, "observed": value},
        )
    return value


def _positive_float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0:
        raise ConfigurationError(
            f"config field {key!r} must be a positive finite number",
            details={"field": key, "observed": value},
        )
    return float(value)


def _rope_theta(config: dict[str, Any]) -> float:
    top_level = _positive_float(config, "rope_theta", 10_000.0)
    parameters = config.get("rope_parameters")
    if parameters is None:
        return top_level
    if not isinstance(parameters, dict):
        raise ConfigurationError("rope_parameters must be an object or null")
    unknown = set(parameters) - {"rope_theta", "rope_type"}
    if unknown:
        raise ConfigurationError(
            "default-RoPE adapter does not understand rope_parameters fields",
            details={"fields": sorted(unknown)},
        )
    rope_type = parameters.get("rope_type", "default")
    if rope_type != "default":
        raise ConfigurationError(
            "only default RoPE is supported by this adapter",
            details={"rope_type": rope_type},
        )
    nested = parameters.get("rope_theta", top_level)
    if type(nested) not in {int, float} or not math.isfinite(float(nested)) or nested <= 0:
        raise ConfigurationError("rope_parameters.rope_theta must be positive and finite")
    if "rope_theta" in config and "rope_theta" in parameters and float(nested) != top_level:
        raise ConfigurationError("top-level and nested rope_theta values disagree")
    return float(nested)


def _canonical_observed(value: Any) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError):
        return repr(value)


def _view_id(logical_name: str) -> str:
    return f"view.{canonical_sha256({'logical_name': logical_name})[:24]}"


def _allocation_id(source_name: str, range_identity: str) -> str:
    return f"alloc.{canonical_sha256({'name': source_name, 'range': range_identity})[:24]}"


def _operation(
    operation_id: str,
    kind: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    parameters: tuple[str, ...] = (),
    attributes: dict[str, Any] | None = None,
) -> OperationIR:
    return OperationIR(
        operation_id=operation_id,
        kind=kind,
        inputs=inputs,
        outputs=outputs,
        parameters=tuple(sorted(parameters)),
        _attributes_json=canonical_json(attributes or {}),
    )


class DenseRotaryAdapter:
    """Shared exact schema builder; concrete subclasses select family-specific variants."""

    adapter_id = ""
    adapter_version = "1.0.0"
    model_type = ""
    architecture_id = ""
    architecture_names: frozenset[str] = frozenset()
    requires_qk_norm = False
    qwen2_fixed_qkv_bias = False
    supports_mlp_bias = False
    extra_config_keys: frozenset[str] = frozenset()
    rule_set = "dense-rotary-separated-qkv-v1"

    def __init__(self) -> None:
        self.adapter_fingerprint = canonical_sha256(
            {
                "adapter_id": self.adapter_id,
                "adapter_version": self.adapter_version,
                "model_type": self.model_type,
                "architecture_id": self.architecture_id,
                "architecture_names": sorted(self.architecture_names),
                "implementation_sha256": _IMPLEMENTATION_SHA256,
                "requires_qk_norm": self.requires_qk_norm,
                "qwen2_fixed_qkv_bias": self.qwen2_fixed_qkv_bias,
                "supports_mlp_bias": self.supports_mlp_bias,
                "extra_config_keys": sorted(self.extra_config_keys),
                "rule_set": self.rule_set,
            }
        )

    def _parse_config(self, config: dict[str, Any]) -> _DenseConfig:
        hidden_size = _positive_int(config, "hidden_size")
        intermediate_size = _positive_int(config, "intermediate_size")
        num_hidden_layers = _positive_int(config, "num_hidden_layers")
        num_attention_heads = _positive_int(config, "num_attention_heads")
        raw_kv_heads = config.get("num_key_value_heads", num_attention_heads)
        if raw_kv_heads is None:
            raw_kv_heads = num_attention_heads
        if type(raw_kv_heads) is not int or raw_kv_heads <= 0:
            raise ConfigurationError("num_key_value_heads must be a positive integer")
        num_key_value_heads = raw_kv_heads
        raw_head_dim = config.get("head_dim")
        if raw_head_dim is None:
            if hidden_size % num_attention_heads:
                raise ConfigurationError(
                    "hidden_size must be divisible by num_attention_heads when head_dim is absent"
                )
            head_dim = hidden_size // num_attention_heads
        elif type(raw_head_dim) is int and raw_head_dim > 0:
            head_dim = raw_head_dim
        else:
            raise ConfigurationError("head_dim must be a positive integer or null")
        if num_attention_heads % num_key_value_heads:
            raise ConfigurationError(
                "num_attention_heads must be divisible by num_key_value_heads",
                details={
                    "num_attention_heads": num_attention_heads,
                    "num_key_value_heads": num_key_value_heads,
                },
            )
        if head_dim % 2:
            raise ConfigurationError("head_dim must be even for default rotary embeddings")
        hidden_act = config.get("hidden_act", "silu")
        if hidden_act != "silu":
            raise ConfigurationError(
                "dense adapter currently supports only SiLU/SwiGLU",
                details={"hidden_act": hidden_act},
            )
        if self.qwen2_fixed_qkv_bias:
            attention_bias = True
        else:
            attention_bias = _optional_bool(config, "attention_bias", False)
        mlp_bias = _optional_bool(config, "mlp_bias", False)
        if mlp_bias and not self.supports_mlp_bias:
            raise ConfigurationError(f"{self.model_type} adapter does not support MLP bias")
        return _DenseConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=_positive_int(config, "vocab_size"),
            max_position_embeddings=_positive_int(config, "max_position_embeddings"),
            rms_norm_eps=_positive_float(config, "rms_norm_eps", 1e-6),
            rope_theta=_rope_theta(config),
            tie_word_embeddings=_optional_bool(config, "tie_word_embeddings", False),
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
        )

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        config = source.config
        if config.get("model_type") != self.model_type:
            return ()
        unsupported: list[UnsupportedFeature] = []
        architectures = config.get("architectures")
        if architectures is not None:
            valid = (
                isinstance(architectures, list)
                and bool(architectures)
                and all(
                    type(item) is str and item in self.architecture_names for item in architectures
                )
            )
            if not valid:
                unsupported.append(
                    UnsupportedFeature(
                        code="architecture_declaration",
                        field="architectures",
                        observed=_canonical_observed(architectures),
                        reason="architecture declaration is not owned by this adapter",
                    )
                )
        unknown_keys = sorted(
            set(config) - _COMMON_CONFIG_KEYS - _DENSE_CONFIG_KEYS - self.extra_config_keys
        )
        if unknown_keys:
            unsupported.append(
                UnsupportedFeature(
                    code="unknown_config_keys",
                    field="config",
                    observed=",".join(unknown_keys),
                    reason="unknown config keys may affect execution and cannot be ignored",
                )
            )
        if config.get("quantization_config") not in (None, {}):
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="quantization_config",
                    observed=_canonical_observed(config.get("quantization_config")),
                    reason="quantized source metadata requires a dedicated codec adapter",
                )
            )
        if config.get("auto_map") not in (None, {}, []):
            unsupported.append(
                UnsupportedFeature(
                    code="remote_code_required",
                    field="auto_map",
                    observed=_canonical_observed(config.get("auto_map")),
                    reason="repository code is never imported by this adapter",
                )
            )
        rope_scaling = config.get("rope_scaling")
        if rope_scaling is not None:
            unsupported.append(
                UnsupportedFeature(
                    code="rope_scaling",
                    field="rope_scaling",
                    observed=_canonical_observed(rope_scaling),
                    reason="this adapter currently certifies default RoPE only",
                )
            )
        partial = config.get("partial_rotary_factor", 1.0)
        if type(partial) not in {int, float} or float(partial) != 1.0:
            unsupported.append(
                UnsupportedFeature(
                    code="partial_rotary",
                    field="partial_rotary_factor",
                    observed=_canonical_observed(partial),
                    reason="partial rotary dimensions require a distinct semantic variant",
                )
            )
        if config.get("use_sliding_window") is True:
            unsupported.append(
                UnsupportedFeature(
                    code="sliding_attention",
                    field="use_sliding_window",
                    observed="true",
                    reason="sliding attention requires a windowed StateIR/lowering",
                )
            )
        if config.get("use_mrope") is True:
            unsupported.append(
                UnsupportedFeature(
                    code="multisection_rotary",
                    field="use_mrope",
                    observed="true",
                    reason="multisection rotary requires a distinct semantic adapter",
                )
            )
        layer_types = config.get("layer_types")
        if layer_types is not None:
            valid_layers = config.get("num_hidden_layers")
            if (
                not isinstance(layer_types, list)
                or type(valid_layers) is not int
                or len(layer_types) != valid_layers
                or any(item != "full_attention" for item in layer_types)
            ):
                unsupported.append(
                    UnsupportedFeature(
                        code="layer_schedule",
                        field="layer_types",
                        observed=_canonical_observed(layer_types),
                        reason="only an all-full-attention layer schedule is supported",
                    )
                )
        if self.model_type == "llama" and config.get("pretraining_tp", 1) not in (None, 1):
            unsupported.append(
                UnsupportedFeature(
                    code="pretraining_tensor_parallel_arithmetic",
                    field="pretraining_tp",
                    observed=_canonical_observed(config.get("pretraining_tp")),
                    reason="source arithmetic partitioning needs a separate numerical contract",
                )
            )
        non_float = sorted({record.storage_dtype for record in index.tensors} - _RAW_FLOAT_DTYPES)
        if non_float:
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="safetensors.dtype",
                    observed=",".join(non_float),
                    reason="raw-float adapter cannot reinterpret packed, integer, or FP8 codes",
                )
            )
        try:
            self._parse_config(config)
        except ConfigurationError as exc:
            unsupported.append(
                UnsupportedFeature(
                    code="invalid_semantic_config",
                    field="config",
                    observed=exc.message,
                    reason="semantic configuration is invalid for this adapter",
                )
            )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))

    def match(self, source: FrozenSourceBundle, index: TensorIndex) -> MatchResult:
        if index.source_fingerprint != source.fingerprint:
            raise ValueError("tensor index is not bound to the supplied frozen source")
        config = source.config
        observed_type = config.get("model_type")
        matched = observed_type == self.model_type
        names = index.by_name()
        architecture_observed = config.get("architectures", "absent")
        evidence = tuple(
            sorted(
                (
                    MatchEvidence(
                        predicate="config.architectures",
                        expected="|".join(sorted(self.architecture_names)),
                        observed=_canonical_observed(architecture_observed),
                        matched=(
                            architecture_observed == "absent"
                            or (
                                isinstance(architecture_observed, list)
                                and bool(architecture_observed)
                                and all(
                                    type(item) is str and item in self.architecture_names
                                    for item in architecture_observed
                                )
                            )
                        ),
                    ),
                    MatchEvidence(
                        predicate="config.model_type",
                        expected=self.model_type,
                        observed=_canonical_observed(observed_type),
                        matched=matched,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.embedding",
                        expected="model.embed_tokens.weight",
                        observed=("present" if "model.embed_tokens.weight" in names else "absent"),
                        matched="model.embed_tokens.weight" in names,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.layer0_q",
                        expected="model.layers.0.self_attn.q_proj.weight",
                        observed=(
                            "present"
                            if "model.layers.0.self_attn.q_proj.weight" in names
                            else "absent"
                        ),
                        matched="model.layers.0.self_attn.q_proj.weight" in names,
                    ),
                ),
                key=lambda item: item.predicate,
            )
        )
        unsupported = self._unsupported_features(source, index) if matched else ()
        codecs = (
            ("raw-float",)
            if all(record.storage_dtype in _RAW_FLOAT_DTYPES for record in index.tensors)
            else ()
        )
        return MatchResult.build(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            matched=matched,
            supported=matched and not unsupported,
            strength=100 if matched else 0,
            evidence=evidence,
            rejected_reasons=(() if matched else (f"model_type is not {self.model_type!r}",)),
            required_tensor_patterns=(
                "model.embed_tokens.weight",
                "model.layers.<0..L-1>.<exact dense block schema>",
                "model.norm.weight",
            ),
            forbidden_tensor_patterns=(
                "*.g_idx",
                "*.qweight",
                "*.qzeros",
                "unowned tensor names",
            ),
            source_codec_candidates=codecs,
            unsupported_features=unsupported,
        )

    def _expected_specs(self, config: _DenseConfig) -> dict[str, _TensorSpec]:
        specs: dict[str, _TensorSpec] = {}

        def add(
            source_name: str,
            logical_name: str,
            shape: tuple[int, ...],
            semantic_role: str,
            *,
            parameter_kind: str = "parameter",
        ) -> None:
            if source_name in specs:
                raise AssertionError(f"duplicate adapter source rule: {source_name}")
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=shape,
                semantic_role=semantic_role,
                parameter_kind=parameter_kind,
                rule_id=f"{self.adapter_id}:separate-dense-v1",
            )

        h = config.hidden_size
        q = config.query_width
        kv = config.key_value_width
        m = config.intermediate_size
        add(
            "model.embed_tokens.weight",
            "token_embedding.weight",
            (config.vocab_size, h),
            "token-embedding",
        )
        add("model.norm.weight", "final_norm.weight", (h,), "final-rms-norm")
        if not config.tie_word_embeddings:
            add(
                "lm_head.weight",
                "lm_head.weight",
                (config.vocab_size, h),
                "output-readout",
            )
        for layer in range(config.num_hidden_layers):
            source = f"model.layers.{layer}"
            logical = f"layers.{layer}"
            add(
                f"{source}.input_layernorm.weight",
                f"{logical}.attention_norm.weight",
                (h,),
                "pre-attention-rms-norm",
            )
            add(
                f"{source}.post_attention_layernorm.weight",
                f"{logical}.mlp_norm.weight",
                (h,),
                "pre-mlp-rms-norm",
            )
            for projection, shape in (
                ("q_proj", (q, h)),
                ("k_proj", (kv, h)),
                ("v_proj", (kv, h)),
                ("o_proj", (h, q)),
            ):
                add(
                    f"{source}.self_attn.{projection}.weight",
                    f"{logical}.attention.{projection}.weight",
                    shape,
                    f"attention-{projection}",
                )
            if self.requires_qk_norm:
                for name in ("q_norm", "k_norm"):
                    add(
                        f"{source}.self_attn.{name}.weight",
                        f"{logical}.attention.{name}.weight",
                        (config.head_dim,),
                        f"attention-{name}",
                    )
            bias_names = ("q_proj", "k_proj", "v_proj") if self.qwen2_fixed_qkv_bias else ()
            if config.attention_bias and not self.qwen2_fixed_qkv_bias:
                bias_names = ("q_proj", "k_proj", "v_proj", "o_proj")
            for projection in bias_names:
                if projection == "q_proj":
                    width = q
                elif projection in {"k_proj", "v_proj"}:
                    width = kv
                else:
                    width = h
                add(
                    f"{source}.self_attn.{projection}.bias",
                    f"{logical}.attention.{projection}.bias",
                    (width,),
                    f"attention-{projection}-bias",
                )
            for projection, shape in (
                ("gate_proj", (m, h)),
                ("up_proj", (m, h)),
                ("down_proj", (h, m)),
            ):
                add(
                    f"{source}.mlp.{projection}.weight",
                    f"{logical}.mlp.{projection}.weight",
                    shape,
                    f"mlp-{projection}",
                )
                if config.mlp_bias:
                    width = h if projection == "down_proj" else m
                    add(
                        f"{source}.mlp.{projection}.bias",
                        f"{logical}.mlp.{projection}.bias",
                        (width,),
                        f"mlp-{projection}-bias",
                    )
        return specs

    def _classify_rotary_buffers(
        self,
        records: dict[str, TensorRecord],
        config: _DenseConfig,
    ) -> tuple[dict[str, _TensorSpec], dict[str, tuple[str, str]]]:
        semantic: dict[str, _TensorSpec] = {}
        ignored: dict[str, tuple[str, str]] = {}
        global_names: dict[str, str] = {}
        per_layer: dict[str, set[int]] = {"inv_freq": set(), "original_inv_freq": set()}
        for name, record in records.items():
            match = _ROTARY_BUFFER_RE.fullmatch(name)
            if match is None:
                continue
            raw_layer, buffer_name = match.groups()
            if buffer_name in {"cos_cached", "sin_cached"}:
                ignored[name] = (
                    f"{self.adapter_id}:derived-rope-cache-v1",
                    "serialized cosine/sine cache is derived from the authoritative RoPE input",
                )
                continue
            expected_shape = (config.head_dim // 2,)
            if record.shape != expected_shape:
                raise ConfigurationError(
                    f"rotary buffer {name!r} has an invalid shape",
                    details={"expected": list(expected_shape), "actual": list(record.shape)},
                )
            if raw_layer is None:
                global_names[buffer_name] = name
                logical_name = f"rotary.{buffer_name}"
            else:
                layer = int(raw_layer)
                if layer >= config.num_hidden_layers:
                    raise CoverageError(
                        f"rotary buffer references out-of-range layer {layer}",
                        details={"source_name": name},
                    )
                per_layer[buffer_name].add(layer)
                logical_name = f"layers.{layer}.attention.rotary.{buffer_name}"
            semantic[name] = _TensorSpec(
                source_name=name,
                logical_name=logical_name,
                shape=expected_shape,
                semantic_role=f"rotary-{buffer_name}",
                parameter_kind="buffer",
                rule_id=f"{self.adapter_id}:serialized-rope-input-v1",
            )
        for buffer_name in ("inv_freq", "original_inv_freq"):
            if buffer_name in global_names and per_layer[buffer_name]:
                raise CoverageError(
                    "checkpoint mixes global and per-layer rotary storage",
                    details={"buffer": buffer_name},
                )
            layers = per_layer[buffer_name]
            if layers and layers != set(range(config.num_hidden_layers)):
                raise CoverageError(
                    "per-layer rotary storage does not cover every layer",
                    details={"buffer": buffer_name, "layers": sorted(layers)},
                )
        return semantic, ignored

    def _map_weights(
        self, source: FrozenSourceBundle, index: TensorIndex, config: _DenseConfig
    ) -> tuple[PhysicalWeightIR, _MappedContext]:
        records = index.by_name()
        specs = self._expected_specs(config)
        if config.tie_word_embeddings and "lm_head.weight" in records:
            raise AliasEvidenceError(
                "tied config serializes a separate lm_head allocation",
                details={"source_name": "lm_head.weight"},
            )
        if not config.tie_word_embeddings and "lm_head.weight" not in records:
            raise AliasEvidenceError(
                "untied config is missing lm_head.weight; a tie cannot be inferred",
                details={"tie_word_embeddings": False},
            )
        missing = sorted(set(specs) - set(records))
        if missing:
            raise CoverageError(
                "required source tensors are absent",
                details={"missing_tensors": missing},
            )
        rotary_specs, ignored = self._classify_rotary_buffers(records, config)
        specs.update(rotary_specs)
        unexplained = sorted(set(records) - set(specs) - set(ignored))
        if unexplained:
            raise CoverageError(
                "adapter cannot classify every source tensor",
                details={"unexplained_tensors": unexplained},
            )
        non_float = sorted(
            record.source_name
            for record in records.values()
            if record.storage_dtype not in _RAW_FLOAT_DTYPES
        )
        if non_float:
            raise CodecError(
                "raw-float adapter encountered non-float storage",
                details={"tensors": non_float},
            )

        allocations: list[PhysicalAllocationIR] = []
        views: list[TensorViewIR] = []
        classifications: list[TensorClassificationIR] = []
        allocation_by_source: dict[str, str] = {}
        for source_name in sorted(records):
            record = records[source_name]
            allocation_id = _allocation_id(source_name, record.range_identity)
            allocation_by_source[source_name] = allocation_id
            allocations.append(
                PhysicalAllocationIR(
                    allocation_id=allocation_id,
                    source_tensor=source_name,
                    source_file=record.source_file,
                    byte_offset=record.byte_offset,
                    byte_length=record.byte_length,
                    stored_shape=record.shape,
                    stored_dtype=record.storage_dtype,
                    codec=CodecBindingIR(
                        codec_id="raw-float",
                        codec_version="1.0.0",
                        stored_dtype=record.storage_dtype,
                        _parameters_json=canonical_json(
                            {
                                "byte_order": "little",
                                "packing": "none",
                                "value_semantics": "safetensors-native-float",
                            }
                        ),
                    ),
                    content_fingerprint=record.range_identity,
                )
            )
            if source_name in ignored:
                rule_id, reason = ignored[source_name]
                classifications.append(
                    TensorClassificationIR(
                        source_name=source_name,
                        allocation_id=allocation_id,
                        disposition="ignored",
                        logical_view_ids=(),
                        rule_id=rule_id,
                        reason=reason,
                    )
                )
                continue
            spec = specs[source_name]
            lexical_shape_valid = (
                source_name in {"model.embed_tokens.weight", "lm_head.weight"}
                and len(record.shape) == 2
                and record.shape[0] >= config.vocab_size
                and record.shape[1] == config.hidden_size
            )
            if not lexical_shape_valid and record.shape != spec.shape:
                raise ConfigurationError(
                    f"tensor shape mismatch for {source_name}",
                    details={"expected": list(spec.shape), "actual": list(record.shape)},
                )
            view_names = [spec.logical_name]
            if source_name == "model.embed_tokens.weight" and config.tie_word_embeddings:
                view_names.append("lm_head.weight")
            view_ids: list[str] = []
            for logical_name in sorted(view_names):
                view_id = _view_id(logical_name)
                view_ids.append(view_id)
                role = "output-readout" if logical_name == "lm_head.weight" else spec.semantic_role
                views.append(
                    TensorViewIR(
                        view_id=view_id,
                        logical_name=logical_name,
                        allocation_id=allocation_id,
                        logical_shape=record.shape,
                        transforms=(
                            ViewTransformIR(kind="identity", _parameters_json=canonical_json({})),
                        ),
                        semantic_role=role,
                        parameter_kind=spec.parameter_kind,
                    )
                )
            classifications.append(
                TensorClassificationIR(
                    source_name=source_name,
                    allocation_id=allocation_id,
                    disposition=spec.parameter_kind,
                    logical_view_ids=tuple(sorted(view_ids)),
                    rule_id=spec.rule_id,
                    reason="exact adapter-owned source tensor schema",
                )
            )

        aliases: tuple[AliasClassIR, ...] = ()
        if config.tie_word_embeddings:
            allocation_id = allocation_by_source["model.embed_tokens.weight"]
            aliases = (
                AliasClassIR(
                    class_id=f"alias.{canonical_sha256({'allocation': allocation_id})[:24]}",
                    allocation_id=allocation_id,
                    logical_names=("lm_head.weight", "token_embedding.weight"),
                    evidence=AliasEvidenceIR(
                        kind="missing-serialized-tied-readout",
                        config_fields=("tie_word_embeddings",),
                        adapter_rule=f"{self.adapter_id}:declared-lexical-tie-v1",
                        certification_status="provisional-u2",
                        required_followup=(
                            "reference-forward-parity",
                            "reference-parameter-identity",
                        ),
                    ),
                ),
            )
        physical = PhysicalWeightIR.build(
            source_fingerprint=source.fingerprint,
            tensor_index_fingerprint=index.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            allocations=tuple(sorted(allocations, key=lambda item: item.allocation_id)),
            views=tuple(sorted(views, key=lambda item: item.logical_name)),
            alias_classes=aliases,
            classifications=tuple(sorted(classifications, key=lambda item: item.source_name)),
        )
        input_rows = records["model.embed_tokens.weight"].shape[0]
        output_source = (
            records["model.embed_tokens.weight"]
            if config.tie_word_embeddings
            else records["lm_head.weight"]
        )
        if input_rows < config.vocab_size or output_source.shape[0] < config.vocab_size:
            raise IOContractError(
                "configured token space exceeds input or output row capacity",
                details={
                    "vocab_size": config.vocab_size,
                    "input_rows": input_rows,
                    "output_rows": output_source.shape[0],
                },
            )
        rotary_by_layer: list[tuple[str, ...]] = []
        for layer in range(config.num_hidden_layers):
            names: list[str] = []
            for buffer_name in ("inv_freq", "original_inv_freq"):
                global_name = f"rotary.{buffer_name}"
                local_name = f"layers.{layer}.attention.rotary.{buffer_name}"
                logical_names = {view.logical_name for view in physical.views}
                if global_name in logical_names:
                    names.append(global_name)
                elif local_name in logical_names:
                    names.append(local_name)
            rotary_by_layer.append(tuple(sorted(names)))
        return physical, _MappedContext(
            config=config,
            input_rows=input_rows,
            output_rows=output_source.shape[0],
            rotary_parameters_by_layer=tuple(rotary_by_layer),
        )

    def _linear_parameters(
        self, logical_prefix: str, projection: str, *, bias: bool
    ) -> tuple[str, ...]:
        parameters = [f"{logical_prefix}.{projection}.weight"]
        if bias:
            parameters.append(f"{logical_prefix}.{projection}.bias")
        return tuple(sorted(parameters))

    def _state_slot_ids(self, config: _DenseConfig) -> tuple[str, ...]:
        values = ["position"]
        for layer in range(config.num_hidden_layers):
            values.extend((f"layers.{layer}.k_cache", f"layers.{layer}.v_cache"))
        return tuple(sorted(values))

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        parameters = tuple(
            sorted(
                (
                    LogicalParameterRefIR(
                        logical_name=view.logical_name,
                        view_id=view.view_id,
                        semantic_role=view.semantic_role,
                        parameter_kind=view.parameter_kind,
                    )
                    for view in physical.views
                ),
                key=lambda item: item.logical_name,
            )
        )
        operations: list[OperationIR] = [
            _operation(
                "embedding",
                "token-embedding",
                ("token_ids",),
                ("embedding.hidden",),
                ("token_embedding.weight",),
                {"padding_policy": "IOIR-row-mapper"},
            )
        ]
        current = "embedding.hidden"
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            operations.append(
                _operation(
                    f"{prefix}.attention_norm",
                    "rms-norm",
                    (current,),
                    (f"{prefix}.attention_norm.hidden",),
                    (f"{prefix}.attention_norm.weight",),
                    {"epsilon": config.rms_norm_eps},
                )
            )
            normed = f"{prefix}.attention_norm.hidden"
            q_bias = config.attention_bias
            kv_bias = config.attention_bias
            o_bias = config.attention_bias and not self.qwen2_fixed_qkv_bias
            for projection, output, has_bias in (
                ("q_proj", f"{attention}.q", q_bias),
                ("k_proj", f"{attention}.k", kv_bias),
                ("v_proj", f"{attention}.v", kv_bias),
            ):
                operations.append(
                    _operation(
                        f"{attention}.{projection}",
                        "linear",
                        (normed,),
                        (output,),
                        self._linear_parameters(attention, projection, bias=has_bias),
                        {"weight_orientation": "out-in"},
                    )
                )
            q_value = f"{attention}.q"
            k_value = f"{attention}.k"
            if self.requires_qk_norm:
                operations.extend(
                    (
                        _operation(
                            f"{attention}.q_norm",
                            "head-rms-norm",
                            (q_value,),
                            (f"{attention}.q_normed",),
                            (f"{attention}.q_norm.weight",),
                            {"epsilon": config.rms_norm_eps, "head_dim": config.head_dim},
                        ),
                        _operation(
                            f"{attention}.k_norm",
                            "head-rms-norm",
                            (k_value,),
                            (f"{attention}.k_normed",),
                            (f"{attention}.k_norm.weight",),
                            {"epsilon": config.rms_norm_eps, "head_dim": config.head_dim},
                        ),
                    )
                )
                q_value = f"{attention}.q_normed"
                k_value = f"{attention}.k_normed"
            operations.append(
                _operation(
                    f"{attention}.rotary",
                    "rotary-default",
                    (q_value, k_value),
                    (f"{attention}.q_rotary", f"{attention}.k_rotary"),
                    context.rotary_parameters_by_layer[layer],
                    {
                        "head_dim": config.head_dim,
                        "max_position_embeddings": config.max_position_embeddings,
                        "rope_theta": config.rope_theta,
                    },
                )
            )
            operations.append(
                _operation(
                    f"{attention}.gqa",
                    "causal-grouped-query-attention",
                    (
                        f"{attention}.q_rotary",
                        f"{attention}.k_rotary",
                        f"{attention}.v",
                    ),
                    (f"{attention}.context",),
                    attributes={
                        "head_dim": config.head_dim,
                        "num_attention_heads": config.num_attention_heads,
                        "num_key_value_heads": config.num_key_value_heads,
                        "scale": config.head_dim**-0.5,
                        "state_slots": [
                            f"layers.{layer}.k_cache",
                            f"layers.{layer}.v_cache",
                            "position",
                        ],
                    },
                )
            )
            operations.append(
                _operation(
                    f"{attention}.o_proj",
                    "linear",
                    (f"{attention}.context",),
                    (f"{attention}.output",),
                    self._linear_parameters(attention, "o_proj", bias=o_bias),
                    {"weight_orientation": "out-in"},
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.attention_residual",
                    "residual-add",
                    (current, f"{attention}.output"),
                    (f"{prefix}.attention_residual.hidden",),
                )
            )
            residual = f"{prefix}.attention_residual.hidden"
            operations.append(
                _operation(
                    f"{prefix}.mlp_norm",
                    "rms-norm",
                    (residual,),
                    (f"{prefix}.mlp_norm.hidden",),
                    (f"{prefix}.mlp_norm.weight",),
                    {"epsilon": config.rms_norm_eps},
                )
            )
            mlp_input = f"{prefix}.mlp_norm.hidden"
            for projection in ("gate_proj", "up_proj"):
                operations.append(
                    _operation(
                        f"{prefix}.mlp.{projection}",
                        "linear",
                        (mlp_input,),
                        (f"{prefix}.mlp.{projection}.hidden",),
                        self._linear_parameters(f"{prefix}.mlp", projection, bias=config.mlp_bias),
                        {"weight_orientation": "out-in"},
                    )
                )
            operations.append(
                _operation(
                    f"{prefix}.mlp.silu",
                    "silu",
                    (f"{prefix}.mlp.gate_proj.hidden",),
                    (f"{prefix}.mlp.gate_activated",),
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.mlp.multiply",
                    "elementwise-multiply",
                    (
                        f"{prefix}.mlp.gate_activated",
                        f"{prefix}.mlp.up_proj.hidden",
                    ),
                    (f"{prefix}.mlp.intermediate",),
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.mlp.down_proj",
                    "linear",
                    (f"{prefix}.mlp.intermediate",),
                    (f"{prefix}.mlp.output",),
                    self._linear_parameters(f"{prefix}.mlp", "down_proj", bias=config.mlp_bias),
                    {"weight_orientation": "out-in"},
                )
            )
            current = f"{prefix}.output"
            operations.append(
                _operation(
                    f"{prefix}.mlp_residual",
                    "residual-add",
                    (residual, f"{prefix}.mlp.output"),
                    (current,),
                )
            )
        operations.extend(
            (
                _operation(
                    "final_norm",
                    "rms-norm",
                    (current,),
                    ("final.hidden",),
                    ("final_norm.weight",),
                    {"epsilon": config.rms_norm_eps},
                ),
                _operation(
                    "lm_head",
                    "linear-readout",
                    ("final.hidden",),
                    ("logits",),
                    ("lm_head.weight",),
                    {"weight_orientation": "rows-hidden"},
                ),
            )
        )
        return ModelIR.build(
            source_fingerprint=source.fingerprint,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            architecture_id=self.architecture_id,
            physical_weights_fingerprint=physical.fingerprint,
            dimensions=ModelDimensionsIR(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                vocab_size=config.vocab_size,
                physical_vocab_rows=context.input_rows,
                max_position_embeddings=config.max_position_embeddings,
            ),
            parameters=parameters,
            operations=tuple(operations),
            state_refs=self._state_slot_ids(config),
            input_ports=(
                PortIR(
                    name="token_ids",
                    semantic="model-token-ids",
                    dtype="int64",
                    shape=("batch", "sequence"),
                    value="token_ids",
                ),
            ),
            output_ports=(
                PortIR(
                    name="hidden_states",
                    semantic="final-normalized-hidden",
                    dtype="activation",
                    shape=("batch", "sequence", "hidden_size"),
                    value="final.hidden",
                ),
                PortIR(
                    name="logits",
                    semantic="token-row-logits",
                    dtype="activation",
                    shape=("batch", "sequence", "output_rows"),
                    value="logits",
                ),
            ),
            numerical_semantics=NumericalSemanticsIR(
                reference_contract="source-reference-required-before-u3",
                accumulation="lowering-declared",
                softmax="stable-causal-softmax",
                positional_arithmetic="default-rope-source-dtype",
                optimization_contract="preregister-before-target-execution",
            ),
        )

    def _build_state(self, source: FrozenSourceBundle, model: ModelIR) -> StateIR:
        config = model.dimensions
        slots: list[StateSlotIR] = [
            StateSlotIR(
                slot_id="position",
                kind="position-counter",
                dtype="int64",
                shape_expression=("batch",),
                ownership="request",
                lease_behavior="exclusive-epoch-bound",
                provisional_representation="next-position-delta",
                commit_rule="atomic-accepted-prefix",
                rollback_rule="discard-provisional",
                memory_charge_expression="batch * sizeof(int64)",
            )
        ]
        for layer in range(config.num_hidden_layers):
            for kind in ("k", "v"):
                slots.append(
                    StateSlotIR(
                        slot_id=f"layers.{layer}.{kind}_cache",
                        kind=f"paged-{kind}-cache",
                        dtype="activation",
                        shape_expression=(
                            "batch",
                            "capacity",
                            "num_key_value_heads",
                            "head_dim",
                        ),
                        ownership="request",
                        lease_behavior="exclusive-epoch-bound",
                        provisional_representation="append-only-page-delta",
                        commit_rule="atomic-accepted-prefix",
                        rollback_rule="discard-provisional",
                        memory_charge_expression=(
                            "batch * capacity * num_key_value_heads * head_dim * dtype_bytes"
                        ),
                    )
                )
        slots_tuple = tuple(sorted(slots, key=lambda item: item.slot_id))
        slot_ids = tuple(item.slot_id for item in slots_tuple)
        return StateIR.build(
            source_fingerprint=source.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            model_fingerprint=model.fingerprint,
            slots=slots_tuple,
            initialization=(
                StateOperationIR(
                    operation_id="initialize-request-state",
                    kind="zero-and-bind-capacity",
                    slots=slot_ids,
                    _attributes_json=canonical_json({"epoch": 0, "length": 0}),
                ),
            ),
            prefill_updates=(
                StateOperationIR(
                    operation_id="prefill-provisional-append",
                    kind="append-kv-and-position-provisionally",
                    slots=slot_ids,
                    _attributes_json=canonical_json({"mutation": "prohibited-before-commit"}),
                ),
            ),
            decode_updates=(
                StateOperationIR(
                    operation_id="decode-provisional-append",
                    kind="append-one-or-more-kv-positions-provisionally",
                    slots=slot_ids,
                    _attributes_json=canonical_json({"mutation": "prohibited-before-commit"}),
                ),
            ),
            commit_protocol=CommitProtocolIR(
                protocol_id="mrun-transactional-state-v1",
                authority="cache-issued-exclusive-lease",
                atomicity="all-slots-and-position",
                accepted_prefix_rule="zero-through-provisional-length",
                rollback="discard-delta-without-committed-mutation",
                stale_epoch_rule="reject-before-write",
            ),
            capacity_equations=(
                CapacityEquationIR(
                    quantity="kv_bytes",
                    expression=(
                        "batch * capacity * 2 * num_hidden_layers * "
                        "num_key_value_heads * head_dim * dtype_bytes"
                    ),
                    units="bytes",
                ),
                CapacityEquationIR(
                    quantity="position_bytes",
                    expression="batch * sizeof(int64)",
                    units="bytes",
                ),
            ),
        )

    def _special_tokens(
        self, config: dict[str, Any], vocab_size: int
    ) -> tuple[SpecialTokenIR, ...]:
        output: list[SpecialTokenIR] = []
        for key in (
            "bos_token_id",
            "decoder_start_token_id",
            "eos_token_id",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "pad_token_id",
            "sep_token_id",
        ):
            if key not in config or config[key] is None:
                continue
            raw = config[key]
            if type(raw) is int:
                values = (raw,)
            elif isinstance(raw, list) and all(type(item) is int for item in raw):
                values = tuple(sorted(set(raw)))
            else:
                raise IOContractError(
                    f"special token field {key!r} must be an integer, integer array, or null"
                )
            if any(value < 0 or value >= vocab_size for value in values):
                raise IOContractError(
                    f"special token field {key!r} is outside the configured token space",
                    details={"token_ids": list(values), "vocab_size": vocab_size},
                )
            output.append(SpecialTokenIR(name=key.removesuffix("_token_id"), token_ids=values))
        return tuple(sorted(output, key=lambda item: item.name))

    def _chat_templates(self, source: FrozenSourceBundle) -> tuple[ChatTemplateIR, ...]:
        tokenizer_config = source.document("tokenizer_config.json")
        if tokenizer_config is None:
            return ()
        if not isinstance(tokenizer_config, dict):
            raise IOContractError("tokenizer_config.json must contain an object")
        raw = tokenizer_config.get("chat_template")
        if raw is None:
            return ()
        templates: dict[str, str] = {}
        if type(raw) is str:
            templates["default"] = raw
        elif isinstance(raw, dict) and all(
            type(name) is str and type(template) is str for name, template in raw.items()
        ):
            templates.update(raw)
        elif isinstance(raw, list):
            for item in raw:
                if (
                    not isinstance(item, dict)
                    or set(item) != {"name", "template"}
                    or type(item["name"]) is not str
                    or type(item["template"]) is not str
                ):
                    raise IOContractError("chat_template list has an unsupported record")
                if item["name"] in templates:
                    raise IOContractError("chat_template list contains duplicate names")
                templates[item["name"]] = item["template"]
        else:
            raise IOContractError("chat_template has an unsupported representation")
        if any(not name or not template for name, template in templates.items()):
            raise IOContractError("chat templates must have non-empty names and content")
        return tuple(
            ChatTemplateIR(
                template_id=name,
                source_asset="tokenizer_config.json",
                content=template,
                sha256=canonical_sha256({"content": template}),
            )
            for name, template in sorted(templates.items())
        )

    def _build_io(
        self, source: FrozenSourceBundle, model: ModelIR, context: _MappedContext
    ) -> IOIR:
        tokenizer_roles = {
            "added-tokens",
            "chat-template",
            "special-tokens",
            "tokenizer",
            "tokenizer-config",
        }
        processor_roles = {"processor-config"}
        tokenizer_assets = tuple(
            sorted(
                (
                    BoundAssetIR(
                        path=record.path,
                        sha256=record.sha256,
                        byte_count=record.byte_count,
                        role=record.role,
                    )
                    for record in source.files
                    if record.role in tokenizer_roles
                ),
                key=lambda item: item.path,
            )
        )
        processor_assets = tuple(
            sorted(
                (
                    BoundAssetIR(
                        path=record.path,
                        sha256=record.sha256,
                        byte_count=record.byte_count,
                        role=record.role,
                    )
                    for record in source.files
                    if record.role in processor_roles
                ),
                key=lambda item: item.path,
            )
        )
        has_ordered_tokenizer = any(asset.role == "tokenizer" for asset in tokenizer_assets)
        status = "content-bound-unvalidated" if has_ordered_tokenizer else "token-id-only"
        missing = {"tokenizer-golden-roundtrip"}
        if not has_ordered_tokenizer:
            missing.add("ordered-tokenizer-assets")
        templates = self._chat_templates(source)
        if templates:
            missing.add("chat-template-golden-token-sequence")
        if any(asset.role == "chat-template" for asset in tokenizer_assets):
            missing.add("external-chat-template-content-binding")
        generation = source.document("generation_config.json")
        if generation is None:
            generation = {}
        if not isinstance(generation, dict):
            raise IOContractError("generation_config.json must contain an object")

        input_kind = (
            "identity" if context.input_rows == context.config.vocab_size else "padded-identity"
        )
        output_kind = (
            "identity" if context.output_rows == context.config.vocab_size else "padded-identity"
        )
        input_mapper = RowMapperIR(
            mapper_id="tokens-to-input-rows",
            source_space="model-token-ids",
            output_rows="token_embedding.rows",
            kind=input_kind,
            token_count=context.config.vocab_size,
            row_count=context.input_rows,
            unreachable_rows=tuple(range(context.config.vocab_size, context.input_rows)),
        )
        output_mapper = RowMapperIR(
            mapper_id="tokens-to-output-rows",
            source_space="model-token-ids",
            output_rows="lm_head.rows",
            kind=output_kind,
            token_count=context.config.vocab_size,
            row_count=context.output_rows,
            unreachable_rows=tuple(range(context.config.vocab_size, context.output_rows)),
        )
        return IOIR.build(
            source_fingerprint=source.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            model_fingerprint=model.fingerprint,
            text_spaces=(
                TokenSpaceIR(
                    space_id="model-token-ids",
                    token_count=context.config.vocab_size,
                    physical_row_count=context.input_rows,
                    ordering_contract="ordered-tokenizer-assets-sha256",
                    tokenizer_status=status,
                ),
            ),
            row_mappers=(input_mapper, output_mapper),
            special_tokens=self._special_tokens(source.config, context.config.vocab_size),
            tokenizer_assets=tokenizer_assets,
            processor_assets=processor_assets,
            chat_templates=templates,
            _generation_defaults_json=canonical_json(generation),
            output_spaces=(
                OutputSpaceIR(
                    space_id="token-logits",
                    semantic="next-token-logits",
                    row_mapper_id="tokens-to-output-rows",
                    row_count=context.output_rows,
                ),
            ),
            missing_requirements=tuple(sorted(missing)),
            portable=False,
        )

    def compile_ir(self, source: FrozenSourceBundle, index: TensorIndex) -> IRBundle:
        result = self.match(source, index)
        if not result.matched or not result.supported:
            raise UnsupportedVariantError(
                f"adapter {self.adapter_id!r} cannot compile this source",
                details={"match": result.as_dict()},
            )
        config = self._parse_config(source.config)
        physical, context = self._map_weights(source, index, config)
        model = self._build_model(source, physical, context)
        state = self._build_state(source, model)
        io = self._build_io(source, model, context)
        return IRBundle.build(physical_weights=physical, model=model, state=state, io=io)


class GPT2Adapter(DenseRotaryAdapter):
    """Exact adapter for the original HF GPT-2 Conv1D/absolute-position schema."""

    adapter_id = "mrun.hf.gpt2"
    model_type = "gpt2"
    architecture_id = "gpt2-causal-decoder"
    architecture_names = frozenset({"GPT2LMHeadModel", "GPT2Model"})
    rule_set = "gpt2-conv1d-absolute-position-v1"

    def _parse_config(self, config: dict[str, Any]) -> _GPT2Config:
        hidden_size = _positive_int(config, "n_embd")
        num_hidden_layers = _positive_int(config, "n_layer")
        num_attention_heads = _positive_int(config, "n_head")
        vocab_size = _positive_int(config, "vocab_size")
        max_positions = _positive_int(config, "n_positions")
        n_ctx = config.get("n_ctx", max_positions)
        if type(n_ctx) is not int or n_ctx != max_positions:
            raise ConfigurationError("GPT-2 n_ctx must equal n_positions")
        if hidden_size % num_attention_heads:
            raise ConfigurationError("GPT-2 n_embd must be divisible by n_head")
        raw_inner = config.get("n_inner")
        if raw_inner is None:
            intermediate_size = 4 * hidden_size
        elif type(raw_inner) is int and raw_inner > 0:
            intermediate_size = raw_inner
        else:
            raise ConfigurationError("GPT-2 n_inner must be a positive integer or null")
        activation = config.get("activation_function", "gelu_new")
        if activation != "gelu_new":
            raise ConfigurationError(
                "registered GPT-2 semantics require tanh-approximated gelu_new"
            )
        if _optional_bool(config, "scale_attn_weights", True) is not True:
            raise ConfigurationError("registered GPT-2 attention requires head-dimension scaling")
        if _optional_bool(config, "scale_attn_by_inverse_layer_idx", False):
            raise ConfigurationError("inverse-layer attention scaling is not registered")
        if _optional_bool(config, "reorder_and_upcast_attn", False):
            raise ConfigurationError("reordered/upcast GPT-2 attention is not registered")
        for field in ("attn_pdrop", "embd_pdrop", "resid_pdrop"):
            value = config.get(field, 0.0)
            if (
                type(value) not in {int, float}
                or not math.isfinite(float(value))
                or not 0 <= float(value) < 1
            ):
                raise ConfigurationError(f"GPT-2 {field} must be finite and in [0, 1)")
        return _GPT2Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=hidden_size // num_attention_heads,
            vocab_size=vocab_size,
            max_position_embeddings=max_positions,
            layer_norm_eps=_positive_float(config, "layer_norm_epsilon", 1e-5),
            tie_word_embeddings=_optional_bool(config, "tie_word_embeddings", True),
            activation_function="gelu_new",
        )

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        config = source.config
        if config.get("model_type") != self.model_type:
            return ()
        unsupported: list[UnsupportedFeature] = []
        architectures = config.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or not architectures
            or any(
                type(item) is not str or item not in self.architecture_names
                for item in architectures
            )
        ):
            unsupported.append(
                UnsupportedFeature(
                    code="architecture_declaration",
                    field="architectures",
                    observed=_canonical_observed(architectures),
                    reason="architecture declaration is not owned by the GPT-2 adapter",
                )
            )
        unknown = sorted(set(config) - _COMMON_CONFIG_KEYS - _GPT2_CONFIG_KEYS)
        if unknown:
            unsupported.append(
                UnsupportedFeature(
                    code="unknown_config_keys",
                    field="config",
                    observed=",".join(unknown),
                    reason="unknown GPT-2 config keys may affect execution",
                )
            )
        if config.get("add_cross_attention") is True:
            unsupported.append(
                UnsupportedFeature(
                    code="cross_attention",
                    field="add_cross_attention",
                    observed="true",
                    reason="cross-attention requires a distinct topology",
                )
            )
        if config.get("quantization_config") not in (None, {}):
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="quantization_config",
                    observed=_canonical_observed(config.get("quantization_config")),
                    reason="quantized GPT-2 storage requires a dedicated codec",
                )
            )
        if config.get("auto_map") not in (None, {}, []):
            unsupported.append(
                UnsupportedFeature(
                    code="remote_code_required",
                    field="auto_map",
                    observed=_canonical_observed(config.get("auto_map")),
                    reason="repository code is never imported by this adapter",
                )
            )
        non_float = sorted({record.storage_dtype for record in index.tensors} - _RAW_FLOAT_DTYPES)
        if non_float:
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="safetensors.dtype",
                    observed=",".join(non_float),
                    reason="raw-float adapter cannot reinterpret packed or integer codes",
                )
            )
        try:
            self._parse_config(config)
        except ConfigurationError as exc:
            unsupported.append(
                UnsupportedFeature(
                    code="invalid_semantic_config",
                    field="config",
                    observed=exc.message,
                    reason="semantic configuration is invalid for the GPT-2 adapter",
                )
            )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))

    def match(self, source: FrozenSourceBundle, index: TensorIndex) -> MatchResult:
        if index.source_fingerprint != source.fingerprint:
            raise ValueError("tensor index is not bound to the supplied frozen source")
        matched = source.config.get("model_type") == self.model_type
        names = index.by_name()
        architectures = source.config.get("architectures", "absent")
        evidence = tuple(
            sorted(
                (
                    MatchEvidence(
                        predicate="config.architectures",
                        expected="|".join(sorted(self.architecture_names)),
                        observed=_canonical_observed(architectures),
                        matched=(
                            architectures == "absent"
                            or (
                                isinstance(architectures, list)
                                and bool(architectures)
                                and all(
                                    type(item) is str and item in self.architecture_names
                                    for item in architectures
                                )
                            )
                        ),
                    ),
                    MatchEvidence(
                        predicate="config.model_type",
                        expected=self.model_type,
                        observed=_canonical_observed(source.config.get("model_type")),
                        matched=matched,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.embedding",
                        expected="transformer.wte.weight",
                        observed=("present" if "transformer.wte.weight" in names else "absent"),
                        matched="transformer.wte.weight" in names,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.layer0_qkv",
                        expected="transformer.h.0.attn.c_attn.weight",
                        observed=(
                            "present" if "transformer.h.0.attn.c_attn.weight" in names else "absent"
                        ),
                        matched="transformer.h.0.attn.c_attn.weight" in names,
                    ),
                ),
                key=lambda item: item.predicate,
            )
        )
        unsupported = self._unsupported_features(source, index) if matched else ()
        codecs = (
            ("raw-float",)
            if all(record.storage_dtype in _RAW_FLOAT_DTYPES for record in index.tensors)
            else ()
        )
        return MatchResult.build(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            matched=matched,
            supported=matched and not unsupported,
            strength=100 if matched else 0,
            evidence=evidence,
            rejected_reasons=(() if matched else ("model_type is not 'gpt2'",)),
            required_tensor_patterns=(
                "transformer.h.<0..L-1>.<exact GPT-2 block schema>",
                "transformer.ln_f.{weight,bias}",
                "transformer.wpe.weight",
                "transformer.wte.weight",
            ),
            forbidden_tensor_patterns=("quantized tensor codecs", "unowned tensor names"),
            source_codec_candidates=codecs,
            unsupported_features=unsupported,
        )

    def _expected_specs(self, config: _GPT2Config) -> dict[str, _TensorSpec]:
        specs: dict[str, _TensorSpec] = {}

        def add(
            source_name: str,
            logical_name: str,
            shape: tuple[int, ...],
            role: str,
            *,
            parameter_kind: str = "parameter",
        ) -> None:
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=shape,
                semantic_role=role,
                parameter_kind=parameter_kind,
                rule_id=f"{self.adapter_id}:gpt2-exact-v1",
            )

        h = config.hidden_size
        m = config.intermediate_size
        add(
            "transformer.wte.weight",
            "token_embedding.weight",
            (config.vocab_size, h),
            "token-embedding",
        )
        add(
            "transformer.wpe.weight",
            "position_embedding.weight",
            (config.max_position_embeddings, h),
            "absolute-position-embedding",
        )
        for suffix in ("weight", "bias"):
            add(
                f"transformer.ln_f.{suffix}",
                f"final_norm.{suffix}",
                (h,),
                f"final-layer-norm-{suffix}",
            )
        if not config.tie_word_embeddings:
            add("lm_head.weight", "lm_head.weight", (config.vocab_size, h), "output-readout")
        for layer in range(config.num_hidden_layers):
            source = f"transformer.h.{layer}"
            logical = f"layers.{layer}"
            for norm_source, norm_logical in (("ln_1", "attention_norm"), ("ln_2", "mlp_norm")):
                for suffix in ("weight", "bias"):
                    add(
                        f"{source}.{norm_source}.{suffix}",
                        f"{logical}.{norm_logical}.{suffix}",
                        (h,),
                        f"{norm_logical}-{suffix}",
                    )
            add(
                f"{source}.attn.bias",
                f"{logical}.attention.causal_mask",
                (1, 1, config.max_position_embeddings, config.max_position_embeddings),
                "serialized-causal-mask",
                parameter_kind="buffer",
            )
            for name, shape, role in (
                ("attn.c_attn.weight", (h, 3 * h), "attention-fused-qkv-in-out"),
                ("attn.c_attn.bias", (3 * h,), "attention-fused-qkv-bias"),
                ("attn.c_proj.weight", (h, h), "attention-output-in-out"),
                ("attn.c_proj.bias", (h,), "attention-output-bias"),
                ("mlp.c_fc.weight", (h, m), "mlp-expansion-in-out"),
                ("mlp.c_fc.bias", (m,), "mlp-expansion-bias"),
                ("mlp.c_proj.weight", (m, h), "mlp-contraction-in-out"),
                ("mlp.c_proj.bias", (h,), "mlp-contraction-bias"),
            ):
                add(f"{source}.{name}", f"{logical}.{name}", shape, role)
        return specs

    def _map_weights(
        self, source: FrozenSourceBundle, index: TensorIndex, config: _GPT2Config
    ) -> tuple[PhysicalWeightIR, _MappedContext]:
        records = index.by_name()
        specs = self._expected_specs(config)
        if config.tie_word_embeddings and "lm_head.weight" in records:
            raise AliasEvidenceError("tied GPT-2 source serializes a separate lm_head allocation")
        if not config.tie_word_embeddings and "lm_head.weight" not in records:
            raise AliasEvidenceError("untied GPT-2 source is missing lm_head.weight")
        missing = sorted(set(specs) - set(records))
        unexplained = sorted(set(records) - set(specs))
        if missing:
            raise CoverageError("required GPT-2 tensors are absent", details={"missing": missing})
        if unexplained:
            raise CoverageError(
                "GPT-2 adapter cannot classify every source tensor",
                details={"unexplained_tensors": unexplained},
            )

        allocations: list[PhysicalAllocationIR] = []
        views: list[TensorViewIR] = []
        classifications: list[TensorClassificationIR] = []
        allocation_by_source: dict[str, str] = {}
        for source_name in sorted(records):
            record = records[source_name]
            spec = specs[source_name]
            lexical = source_name in {"transformer.wte.weight", "lm_head.weight"}
            if (
                not (
                    lexical
                    and len(record.shape) == 2
                    and record.shape[0] >= config.vocab_size
                    and record.shape[1] == config.hidden_size
                )
                and record.shape != spec.shape
            ):
                raise ConfigurationError(
                    f"tensor shape mismatch for {source_name}",
                    details={"expected": list(spec.shape), "actual": list(record.shape)},
                )
            allocation_id = _allocation_id(source_name, record.range_identity)
            allocation_by_source[source_name] = allocation_id
            allocations.append(
                PhysicalAllocationIR(
                    allocation_id=allocation_id,
                    source_tensor=source_name,
                    source_file=record.source_file,
                    byte_offset=record.byte_offset,
                    byte_length=record.byte_length,
                    stored_shape=record.shape,
                    stored_dtype=record.storage_dtype,
                    codec=CodecBindingIR(
                        codec_id="raw-float",
                        codec_version="1.0.0",
                        stored_dtype=record.storage_dtype,
                        _parameters_json=canonical_json(
                            {
                                "byte_order": "little",
                                "packing": "none",
                                "value_semantics": "safetensors-native-float",
                            }
                        ),
                    ),
                    content_fingerprint=record.range_identity,
                )
            )
            logical_names = [spec.logical_name]
            if source_name == "transformer.wte.weight" and config.tie_word_embeddings:
                logical_names.append("lm_head.weight")
            view_ids: list[str] = []
            for logical_name in sorted(logical_names):
                view_id = _view_id(logical_name)
                view_ids.append(view_id)
                views.append(
                    TensorViewIR(
                        view_id=view_id,
                        logical_name=logical_name,
                        allocation_id=allocation_id,
                        logical_shape=record.shape,
                        transforms=(
                            ViewTransformIR(kind="identity", _parameters_json=canonical_json({})),
                        ),
                        semantic_role=(
                            "output-readout"
                            if logical_name == "lm_head.weight"
                            else spec.semantic_role
                        ),
                        parameter_kind=spec.parameter_kind,
                    )
                )
            classifications.append(
                TensorClassificationIR(
                    source_name=source_name,
                    allocation_id=allocation_id,
                    disposition=spec.parameter_kind,
                    logical_view_ids=tuple(sorted(view_ids)),
                    rule_id=spec.rule_id,
                    reason="exact adapter-owned GPT-2 source tensor schema",
                )
            )
        aliases: tuple[AliasClassIR, ...] = ()
        if config.tie_word_embeddings:
            allocation_id = allocation_by_source["transformer.wte.weight"]
            aliases = (
                AliasClassIR(
                    class_id=f"alias.{canonical_sha256({'allocation': allocation_id})[:24]}",
                    allocation_id=allocation_id,
                    logical_names=("lm_head.weight", "token_embedding.weight"),
                    evidence=AliasEvidenceIR(
                        kind="missing-serialized-tied-readout",
                        config_fields=("tie_word_embeddings",),
                        adapter_rule=f"{self.adapter_id}:declared-lexical-tie-v1",
                        certification_status="provisional-u2",
                        required_followup=(
                            "reference-forward-parity",
                            "reference-parameter-identity",
                        ),
                    ),
                ),
            )
        physical = PhysicalWeightIR.build(
            source_fingerprint=source.fingerprint,
            tensor_index_fingerprint=index.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            allocations=tuple(sorted(allocations, key=lambda item: item.allocation_id)),
            views=tuple(sorted(views, key=lambda item: item.logical_name)),
            alias_classes=aliases,
            classifications=tuple(sorted(classifications, key=lambda item: item.source_name)),
        )
        input_rows = records["transformer.wte.weight"].shape[0]
        output_rows = (
            input_rows if config.tie_word_embeddings else records["lm_head.weight"].shape[0]
        )
        if input_rows < config.vocab_size or output_rows < config.vocab_size:
            raise IOContractError("GPT-2 configured token space exceeds lexical row capacity")
        return physical, _MappedContext(
            config=config,
            input_rows=input_rows,
            output_rows=output_rows,
            rotary_parameters_by_layer=tuple(() for _ in range(config.num_hidden_layers)),
        )

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        if not isinstance(config, _GPT2Config):
            raise TypeError("GPT-2 model builder received a foreign config")
        parameters = tuple(
            sorted(
                (
                    LogicalParameterRefIR(
                        logical_name=view.logical_name,
                        view_id=view.view_id,
                        semantic_role=view.semantic_role,
                        parameter_kind=view.parameter_kind,
                    )
                    for view in physical.views
                ),
                key=lambda item: item.logical_name,
            )
        )
        operations: list[OperationIR] = [
            _operation(
                "embedding",
                "token-embedding",
                ("token_ids",),
                ("embedding.token",),
                ("token_embedding.weight",),
                {"padding_policy": "IOIR-row-mapper"},
            ),
            _operation(
                "position_embedding",
                "absolute-position-embedding-add",
                ("embedding.token",),
                ("embedding.hidden",),
                ("position_embedding.weight",),
                {"max_position_embeddings": config.max_position_embeddings},
            ),
        ]
        current = "embedding.hidden"
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            operations.extend(
                (
                    _operation(
                        f"{prefix}.attention_norm",
                        "layer-norm",
                        (current,),
                        (f"{prefix}.attention_norm.hidden",),
                        (f"{prefix}.attention_norm.weight", f"{prefix}.attention_norm.bias"),
                        {"epsilon": config.layer_norm_eps},
                    ),
                    _operation(
                        f"{attention}.c_attn",
                        "conv1d-linear",
                        (f"{prefix}.attention_norm.hidden",),
                        (f"{attention}.packed_qkv",),
                        (f"{prefix}.attn.c_attn.weight", f"{prefix}.attn.c_attn.bias"),
                        {"weight_orientation": "in-out"},
                    ),
                    _operation(
                        f"{attention}.split_qkv",
                        "fused-qkv-split",
                        (f"{attention}.packed_qkv",),
                        (f"{attention}.q", f"{attention}.k", f"{attention}.v"),
                        attributes={"split_width": config.hidden_size},
                    ),
                    _operation(
                        f"{attention}.mha",
                        "causal-grouped-query-attention",
                        (f"{attention}.q", f"{attention}.k", f"{attention}.v"),
                        (f"{attention}.context",),
                        (f"{attention}.causal_mask",),
                        {
                            "head_dim": config.head_dim,
                            "num_attention_heads": config.num_attention_heads,
                            "num_key_value_heads": config.num_key_value_heads,
                            "scale": config.head_dim**-0.5,
                            "state_slots": [
                                f"layers.{layer}.k_cache",
                                f"layers.{layer}.v_cache",
                                "position",
                            ],
                        },
                    ),
                    _operation(
                        f"{attention}.c_proj",
                        "conv1d-linear",
                        (f"{attention}.context",),
                        (f"{attention}.output",),
                        (f"{prefix}.attn.c_proj.weight", f"{prefix}.attn.c_proj.bias"),
                        {"weight_orientation": "in-out"},
                    ),
                    _operation(
                        f"{prefix}.attention_residual",
                        "residual-add",
                        (current, f"{attention}.output"),
                        (f"{prefix}.attention_residual.hidden",),
                    ),
                    _operation(
                        f"{prefix}.mlp_norm",
                        "layer-norm",
                        (f"{prefix}.attention_residual.hidden",),
                        (f"{prefix}.mlp_norm.hidden",),
                        (f"{prefix}.mlp_norm.weight", f"{prefix}.mlp_norm.bias"),
                        {"epsilon": config.layer_norm_eps},
                    ),
                    _operation(
                        f"{prefix}.mlp.c_fc",
                        "conv1d-linear",
                        (f"{prefix}.mlp_norm.hidden",),
                        (f"{prefix}.mlp.expanded",),
                        (f"{prefix}.mlp.c_fc.weight", f"{prefix}.mlp.c_fc.bias"),
                        {"weight_orientation": "in-out"},
                    ),
                    _operation(
                        f"{prefix}.mlp.gelu",
                        "gelu-tanh",
                        (f"{prefix}.mlp.expanded",),
                        (f"{prefix}.mlp.activated",),
                    ),
                    _operation(
                        f"{prefix}.mlp.c_proj",
                        "conv1d-linear",
                        (f"{prefix}.mlp.activated",),
                        (f"{prefix}.mlp.output",),
                        (f"{prefix}.mlp.c_proj.weight", f"{prefix}.mlp.c_proj.bias"),
                        {"weight_orientation": "in-out"},
                    ),
                )
            )
            current = f"{prefix}.output"
            operations.append(
                _operation(
                    f"{prefix}.mlp_residual",
                    "residual-add",
                    (f"{prefix}.attention_residual.hidden", f"{prefix}.mlp.output"),
                    (current,),
                )
            )
        operations.extend(
            (
                _operation(
                    "final_norm",
                    "layer-norm",
                    (current,),
                    ("final.hidden",),
                    ("final_norm.weight", "final_norm.bias"),
                    {"epsilon": config.layer_norm_eps},
                ),
                _operation(
                    "lm_head",
                    "linear-readout",
                    ("final.hidden",),
                    ("logits",),
                    ("lm_head.weight",),
                    {"weight_orientation": "rows-hidden"},
                ),
            )
        )
        return ModelIR.build(
            source_fingerprint=source.fingerprint,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            architecture_id=self.architecture_id,
            physical_weights_fingerprint=physical.fingerprint,
            dimensions=ModelDimensionsIR(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                vocab_size=config.vocab_size,
                physical_vocab_rows=context.input_rows,
                max_position_embeddings=config.max_position_embeddings,
            ),
            parameters=parameters,
            operations=tuple(operations),
            state_refs=self._state_slot_ids(config),
            input_ports=(
                PortIR(
                    name="token_ids",
                    semantic="model-token-ids",
                    dtype="int64",
                    shape=("batch", "sequence"),
                    value="token_ids",
                ),
            ),
            output_ports=(
                PortIR(
                    name="hidden_states",
                    semantic="final-normalized-hidden",
                    dtype="activation",
                    shape=("batch", "sequence", "hidden_size"),
                    value="final.hidden",
                ),
                PortIR(
                    name="logits",
                    semantic="token-row-logits",
                    dtype="activation",
                    shape=("batch", "sequence", "output_rows"),
                    value="logits",
                ),
            ),
            numerical_semantics=NumericalSemanticsIR(
                reference_contract="source-reference-required-before-u3",
                accumulation="lowering-declared",
                softmax="stable-causal-softmax",
                positional_arithmetic="learned-absolute-position-source-dtype",
                optimization_contract="preregister-before-target-execution",
            ),
        )


class PhiAdapter(DenseRotaryAdapter):
    """Exact adapter for the original Microsoft Phi-1/Phi-2 decoder schema."""

    adapter_id = "mrun.hf.phi"
    model_type = "phi"
    architecture_id = "phi-causal-decoder"
    architecture_names = frozenset({"PhiForCausalLM", "PhiModel"})
    rule_set = "phi-parallel-mlp-partial-rope-v1"

    def _parse_config(self, config: dict[str, Any]) -> _PhiConfig:
        hidden_size = _positive_int(config, "hidden_size")
        intermediate_size = _positive_int(config, "intermediate_size")
        num_hidden_layers = _positive_int(config, "num_hidden_layers")
        num_attention_heads = _positive_int(config, "num_attention_heads")
        if hidden_size % num_attention_heads:
            raise ConfigurationError("Phi hidden_size must be divisible by attention heads")
        head_dim = hidden_size // num_attention_heads
        raw_head_dim = config.get("head_dim")
        if raw_head_dim is not None and raw_head_dim != head_dim:
            raise ConfigurationError("Phi head_dim differs from hidden_size / attention heads")
        raw_kv = config.get("num_key_value_heads", num_attention_heads)
        if raw_kv is None:
            raw_kv = num_attention_heads
        if type(raw_kv) is not int or raw_kv <= 0 or num_attention_heads % raw_kv:
            raise ConfigurationError("Phi key/value head count is invalid")
        partial = config.get("partial_rotary_factor", 0.5)
        if (
            type(partial) not in {int, float}
            or not math.isfinite(float(partial))
            or not 0 < float(partial) <= 1
        ):
            raise ConfigurationError("Phi partial_rotary_factor must be finite and in (0, 1]")
        rotary_dim = int(head_dim * float(partial))
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ConfigurationError("Phi partial rotary dimension must be positive and even")
        if config.get("hidden_act", "gelu_new") != "gelu_new":
            raise ConfigurationError("registered Phi semantics require gelu_new")
        if _optional_bool(config, "qk_layernorm", False):
            raise ConfigurationError("Phi qk_layernorm requires extra parameters and operations")
        if _optional_bool(config, "tie_word_embeddings", False):
            raise ConfigurationError("registered Phi topology requires untied lexical matrices")
        if config.get("rope_scaling") is not None:
            raise ConfigurationError("registered Phi semantics require unscaled default RoPE")
        return _PhiConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=raw_kv,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            vocab_size=_positive_int(config, "vocab_size"),
            max_position_embeddings=_positive_int(config, "max_position_embeddings"),
            layer_norm_eps=_positive_float(config, "layer_norm_eps", 1e-5),
            rope_theta=_rope_theta(config),
            partial_rotary_factor=float(partial),
            tie_word_embeddings=False,
            hidden_act="gelu_new",
        )

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        config = source.config
        if config.get("model_type") != self.model_type:
            return ()
        unsupported: list[UnsupportedFeature] = []
        architectures = config.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or not architectures
            or any(
                type(item) is not str or item not in self.architecture_names
                for item in architectures
            )
        ):
            unsupported.append(
                UnsupportedFeature(
                    code="architecture_declaration",
                    field="architectures",
                    observed=_canonical_observed(architectures),
                    reason="architecture declaration is not owned by the Phi adapter",
                )
            )
        unknown = sorted(set(config) - _COMMON_CONFIG_KEYS - _PHI_CONFIG_KEYS)
        if unknown:
            unsupported.append(
                UnsupportedFeature(
                    code="unknown_config_keys",
                    field="config",
                    observed=",".join(unknown),
                    reason="unknown Phi config keys may affect execution",
                )
            )
        if config.get("quantization_config") not in (None, {}):
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="quantization_config",
                    observed=_canonical_observed(config.get("quantization_config")),
                    reason="quantized Phi storage requires a dedicated codec",
                )
            )
        if config.get("auto_map") not in (None, {}, []):
            unsupported.append(
                UnsupportedFeature(
                    code="remote_code_required",
                    field="auto_map",
                    observed=_canonical_observed(config.get("auto_map")),
                    reason="repository code is never imported by this adapter",
                )
            )
        non_float = sorted({record.storage_dtype for record in index.tensors} - _RAW_FLOAT_DTYPES)
        if non_float:
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="safetensors.dtype",
                    observed=",".join(non_float),
                    reason="raw-float adapter cannot reinterpret packed or integer codes",
                )
            )
        try:
            self._parse_config(config)
        except ConfigurationError as exc:
            unsupported.append(
                UnsupportedFeature(
                    code="invalid_semantic_config",
                    field="config",
                    observed=exc.message,
                    reason="semantic configuration is invalid for the Phi adapter",
                )
            )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))

    def _expected_specs(self, config: _PhiConfig) -> dict[str, _TensorSpec]:
        specs: dict[str, _TensorSpec] = {}

        def add(source_name: str, logical_name: str, shape: tuple[int, ...], role: str) -> None:
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=shape,
                semantic_role=role,
                rule_id=f"{self.adapter_id}:phi-exact-v1",
            )

        h, m = config.hidden_size, config.intermediate_size
        q, kv = config.query_width, config.key_value_width
        add(
            "model.embed_tokens.weight",
            "token_embedding.weight",
            (config.vocab_size, h),
            "token-embedding",
        )
        add("lm_head.weight", "lm_head.weight", (config.vocab_size, h), "output-readout")
        add("lm_head.bias", "lm_head.bias", (config.vocab_size,), "output-readout-bias")
        for suffix in ("weight", "bias"):
            add(
                f"model.final_layernorm.{suffix}",
                f"final_norm.{suffix}",
                (h,),
                f"final-layer-norm-{suffix}",
            )
        for layer in range(config.num_hidden_layers):
            source = f"model.layers.{layer}"
            logical = f"layers.{layer}"
            for suffix in ("weight", "bias"):
                add(
                    f"{source}.input_layernorm.{suffix}",
                    f"{logical}.input_norm.{suffix}",
                    (h,),
                    f"parallel-input-layer-norm-{suffix}",
                )
            for projection, width in (("q_proj", q), ("k_proj", kv), ("v_proj", kv)):
                add(
                    f"{source}.self_attn.{projection}.weight",
                    f"{logical}.attention.{projection}.weight",
                    (width, h),
                    f"attention-{projection}",
                )
                add(
                    f"{source}.self_attn.{projection}.bias",
                    f"{logical}.attention.{projection}.bias",
                    (width,),
                    f"attention-{projection}-bias",
                )
            add(
                f"{source}.self_attn.dense.weight",
                f"{logical}.attention.dense.weight",
                (h, q),
                "attention-output-projection",
            )
            add(
                f"{source}.self_attn.dense.bias",
                f"{logical}.attention.dense.bias",
                (h,),
                "attention-output-projection-bias",
            )
            for projection, shape, width in (
                ("fc1", (m, h), m),
                ("fc2", (h, m), h),
            ):
                add(
                    f"{source}.mlp.{projection}.weight",
                    f"{logical}.mlp.{projection}.weight",
                    shape,
                    f"mlp-{projection}",
                )
                add(
                    f"{source}.mlp.{projection}.bias",
                    f"{logical}.mlp.{projection}.bias",
                    (width,),
                    f"mlp-{projection}-bias",
                )
        return specs

    def _map_weights(
        self, source: FrozenSourceBundle, index: TensorIndex, config: _PhiConfig
    ) -> tuple[PhysicalWeightIR, _MappedContext]:
        records = index.by_name()
        specs = self._expected_specs(config)
        missing = sorted(set(specs) - set(records))
        unexplained = sorted(set(records) - set(specs))
        if missing:
            raise CoverageError("required Phi tensors are absent", details={"missing": missing})
        if unexplained:
            raise CoverageError(
                "Phi adapter cannot classify every source tensor",
                details={"unexplained_tensors": unexplained},
            )
        input_rows = records["model.embed_tokens.weight"].shape[0]
        output_rows = records["lm_head.weight"].shape[0]
        output_bias_rows = records["lm_head.bias"].shape[0]
        if input_rows < config.vocab_size or output_rows < config.vocab_size:
            raise IOContractError("Phi configured token space exceeds lexical row capacity")
        if output_bias_rows != output_rows:
            raise IOContractError("Phi output readout bias must cover every physical output row")
        allocations: list[PhysicalAllocationIR] = []
        views: list[TensorViewIR] = []
        classifications: list[TensorClassificationIR] = []
        for source_name in sorted(records):
            record = records[source_name]
            spec = specs[source_name]
            lexical = source_name in {"model.embed_tokens.weight", "lm_head.weight"}
            flexible_lexical_matrix = (
                lexical
                and len(record.shape) == 2
                and record.shape[0] >= config.vocab_size
                and record.shape[1] == config.hidden_size
            )
            flexible_output_bias = source_name == "lm_head.bias" and record.shape == (output_rows,)
            if not (flexible_lexical_matrix or flexible_output_bias) and record.shape != spec.shape:
                raise ConfigurationError(
                    f"tensor shape mismatch for {source_name}",
                    details={"expected": list(spec.shape), "actual": list(record.shape)},
                )
            allocation_id = _allocation_id(source_name, record.range_identity)
            view_id = _view_id(spec.logical_name)
            allocations.append(
                PhysicalAllocationIR(
                    allocation_id=allocation_id,
                    source_tensor=source_name,
                    source_file=record.source_file,
                    byte_offset=record.byte_offset,
                    byte_length=record.byte_length,
                    stored_shape=record.shape,
                    stored_dtype=record.storage_dtype,
                    codec=CodecBindingIR(
                        codec_id="raw-float",
                        codec_version="1.0.0",
                        stored_dtype=record.storage_dtype,
                        _parameters_json=canonical_json(
                            {
                                "byte_order": "little",
                                "packing": "none",
                                "value_semantics": "safetensors-native-float",
                            }
                        ),
                    ),
                    content_fingerprint=record.range_identity,
                )
            )
            views.append(
                TensorViewIR(
                    view_id=view_id,
                    logical_name=spec.logical_name,
                    allocation_id=allocation_id,
                    logical_shape=record.shape,
                    transforms=(
                        ViewTransformIR(kind="identity", _parameters_json=canonical_json({})),
                    ),
                    semantic_role=spec.semantic_role,
                    parameter_kind="parameter",
                )
            )
            classifications.append(
                TensorClassificationIR(
                    source_name=source_name,
                    allocation_id=allocation_id,
                    disposition="parameter",
                    logical_view_ids=(view_id,),
                    rule_id=spec.rule_id,
                    reason="exact adapter-owned Phi source tensor schema",
                )
            )
        physical = PhysicalWeightIR.build(
            source_fingerprint=source.fingerprint,
            tensor_index_fingerprint=index.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            allocations=tuple(sorted(allocations, key=lambda item: item.allocation_id)),
            views=tuple(sorted(views, key=lambda item: item.logical_name)),
            alias_classes=(),
            classifications=tuple(sorted(classifications, key=lambda item: item.source_name)),
        )
        return physical, _MappedContext(
            config=config,
            input_rows=input_rows,
            output_rows=output_rows,
            rotary_parameters_by_layer=tuple(() for _ in range(config.num_hidden_layers)),
        )

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        if not isinstance(config, _PhiConfig):
            raise TypeError("Phi model builder received a foreign config")
        parameters = tuple(
            sorted(
                (
                    LogicalParameterRefIR(
                        logical_name=view.logical_name,
                        view_id=view.view_id,
                        semantic_role=view.semantic_role,
                        parameter_kind=view.parameter_kind,
                    )
                    for view in physical.views
                ),
                key=lambda item: item.logical_name,
            )
        )
        operations: list[OperationIR] = [
            _operation(
                "embedding",
                "token-embedding",
                ("token_ids",),
                ("embedding.hidden",),
                ("token_embedding.weight",),
                {"padding_policy": "IOIR-row-mapper"},
            )
        ]
        current = "embedding.hidden"
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            operations.append(
                _operation(
                    f"{prefix}.input_norm",
                    "layer-norm",
                    (current,),
                    (f"{prefix}.input_norm.hidden",),
                    (f"{prefix}.input_norm.weight", f"{prefix}.input_norm.bias"),
                    {"epsilon": config.layer_norm_eps},
                )
            )
            for projection, output in (
                ("q_proj", "q"),
                ("k_proj", "k"),
                ("v_proj", "v"),
            ):
                operations.append(
                    _operation(
                        f"{attention}.{projection}",
                        "linear",
                        (f"{prefix}.input_norm.hidden",),
                        (f"{attention}.{output}",),
                        self._linear_parameters(attention, projection, bias=True),
                        {"weight_orientation": "out-in"},
                    )
                )
            operations.extend(
                (
                    _operation(
                        f"{attention}.rotary",
                        "rotary-partial-default",
                        (f"{attention}.q", f"{attention}.k"),
                        (f"{attention}.q_rotary", f"{attention}.k_rotary"),
                        attributes={
                            "head_dim": config.head_dim,
                            "max_position_embeddings": config.max_position_embeddings,
                            "rope_theta": config.rope_theta,
                            "rotary_dim": config.rotary_dim,
                        },
                    ),
                    _operation(
                        f"{attention}.gqa",
                        "causal-grouped-query-attention",
                        (
                            f"{attention}.q_rotary",
                            f"{attention}.k_rotary",
                            f"{attention}.v",
                        ),
                        (f"{attention}.context",),
                        attributes={
                            "head_dim": config.head_dim,
                            "num_attention_heads": config.num_attention_heads,
                            "num_key_value_heads": config.num_key_value_heads,
                            "scale": config.head_dim**-0.5,
                            "state_slots": [
                                f"layers.{layer}.k_cache",
                                f"layers.{layer}.v_cache",
                                "position",
                            ],
                        },
                    ),
                    _operation(
                        f"{attention}.dense",
                        "linear",
                        (f"{attention}.context",),
                        (f"{attention}.output",),
                        self._linear_parameters(attention, "dense", bias=True),
                        {"weight_orientation": "out-in"},
                    ),
                    _operation(
                        f"{prefix}.mlp.fc1",
                        "linear",
                        (f"{prefix}.input_norm.hidden",),
                        (f"{prefix}.mlp.expanded",),
                        self._linear_parameters(f"{prefix}.mlp", "fc1", bias=True),
                        {"weight_orientation": "out-in"},
                    ),
                    _operation(
                        f"{prefix}.mlp.gelu",
                        "gelu-tanh",
                        (f"{prefix}.mlp.expanded",),
                        (f"{prefix}.mlp.activated",),
                    ),
                    _operation(
                        f"{prefix}.mlp.fc2",
                        "linear",
                        (f"{prefix}.mlp.activated",),
                        (f"{prefix}.mlp.output",),
                        self._linear_parameters(f"{prefix}.mlp", "fc2", bias=True),
                        {"weight_orientation": "out-in"},
                    ),
                    _operation(
                        f"{prefix}.parallel_sum",
                        "residual-add",
                        (f"{attention}.output", f"{prefix}.mlp.output"),
                        (f"{prefix}.parallel.output",),
                    ),
                )
            )
            next_hidden = f"{prefix}.output"
            operations.append(
                _operation(
                    f"{prefix}.residual",
                    "residual-add",
                    (f"{prefix}.parallel.output", current),
                    (next_hidden,),
                )
            )
            current = next_hidden
        operations.extend(
            (
                _operation(
                    "final_norm",
                    "layer-norm",
                    (current,),
                    ("final.hidden",),
                    ("final_norm.weight", "final_norm.bias"),
                    {"epsilon": config.layer_norm_eps},
                ),
                _operation(
                    "lm_head",
                    "linear-readout",
                    ("final.hidden",),
                    ("logits",),
                    ("lm_head.weight", "lm_head.bias"),
                    {"weight_orientation": "rows-hidden"},
                ),
            )
        )
        return ModelIR.build(
            source_fingerprint=source.fingerprint,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            architecture_id=self.architecture_id,
            physical_weights_fingerprint=physical.fingerprint,
            dimensions=ModelDimensionsIR(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                vocab_size=config.vocab_size,
                physical_vocab_rows=context.input_rows,
                max_position_embeddings=config.max_position_embeddings,
            ),
            parameters=parameters,
            operations=tuple(operations),
            state_refs=self._state_slot_ids(config),
            input_ports=(
                PortIR(
                    name="token_ids",
                    semantic="model-token-ids",
                    dtype="int64",
                    shape=("batch", "sequence"),
                    value="token_ids",
                ),
            ),
            output_ports=(
                PortIR(
                    name="hidden_states",
                    semantic="final-normalized-hidden",
                    dtype="activation",
                    shape=("batch", "sequence", "hidden_size"),
                    value="final.hidden",
                ),
                PortIR(
                    name="logits",
                    semantic="token-row-logits",
                    dtype="activation",
                    shape=("batch", "sequence", "output_rows"),
                    value="logits",
                ),
            ),
            numerical_semantics=NumericalSemanticsIR(
                reference_contract="source-reference-required-before-u3",
                accumulation="lowering-declared",
                softmax="stable-causal-softmax",
                positional_arithmetic="phi-partial-default-rope-source-dtype",
                optimization_contract="preregister-before-target-execution",
            ),
        )


class GPTNeoXAdapter(DenseRotaryAdapter):
    """Exact raw-float adapter for the HF GPT-NeoX/Pythia checkpoint schema."""

    adapter_id = "mrun.hf.gpt-neox-pythia"
    model_type = "gpt_neox"
    architecture_id = "gpt-neox-pythia-causal-decoder"
    architecture_names = frozenset({"GPTNeoXForCausalLM", "GPTNeoXModel"})
    rule_set = "gpt-neox-fused-qkv-partial-rope-v1"

    def _parse_config(self, config: dict[str, Any]) -> _GPTNeoXConfig:
        hidden_size = _positive_int(config, "hidden_size")
        intermediate_size = _positive_int(config, "intermediate_size")
        num_hidden_layers = _positive_int(config, "num_hidden_layers")
        num_attention_heads = _positive_int(config, "num_attention_heads")
        if hidden_size % num_attention_heads:
            raise ConfigurationError(
                "GPT-NeoX hidden_size must be divisible by num_attention_heads"
            )
        derived_head_dim = hidden_size // num_attention_heads
        raw_head_dim = config.get("head_dim")
        if raw_head_dim is not None and raw_head_dim != derived_head_dim:
            raise ConfigurationError(
                "GPT-NeoX head_dim must equal hidden_size / num_attention_heads",
                details={"expected": derived_head_dim, "observed": raw_head_dim},
            )
        raw_kv_heads = config.get("num_key_value_heads", num_attention_heads)
        if raw_kv_heads is None:
            raw_kv_heads = num_attention_heads
        if raw_kv_heads != num_attention_heads:
            raise ConfigurationError(
                "registered GPT-NeoX fused QKV packing requires one K/V head per query head",
                details={
                    "num_attention_heads": num_attention_heads,
                    "num_key_value_heads": raw_kv_heads,
                },
            )

        hidden_act = config.get("hidden_act", "gelu")
        if hidden_act != "gelu":
            raise ConfigurationError(
                "registered GPT-NeoX numerical semantics require exact erf GELU",
                details={"hidden_act": hidden_act},
            )
        tied = _optional_bool(config, "tie_word_embeddings", False)
        if tied:
            raise ConfigurationError(
                "registered GPT-NeoX/Pythia topology requires distinct embed_in and "
                "embed_out allocations"
            )
        attention_bias = _optional_bool(config, "attention_bias", True)
        use_parallel_residual = _optional_bool(config, "use_parallel_residual", True)

        legacy_theta = _positive_float(config, "rotary_emb_base", 10_000.0)
        raw_pct = config.get("rotary_pct", 0.25)
        if (
            type(raw_pct) not in {int, float}
            or not math.isfinite(float(raw_pct))
            or not 0 < float(raw_pct) <= 1
        ):
            raise ConfigurationError("rotary_pct must be finite and in (0, 1]")
        rotary_pct = float(raw_pct)
        rope_theta = legacy_theta
        parameters = config.get("rope_parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ConfigurationError("rope_parameters must be an object or null")
            unknown = set(parameters) - {"partial_rotary_factor", "rope_theta", "rope_type"}
            if unknown:
                raise ConfigurationError(
                    "GPT-NeoX default RoPE parameters contain unknown fields",
                    details={"fields": sorted(unknown)},
                )
            if parameters.get("rope_type", "default") != "default":
                raise ConfigurationError("registered GPT-NeoX adapter supports default RoPE only")
            nested_theta = parameters.get("rope_theta", legacy_theta)
            nested_pct = parameters.get("partial_rotary_factor", rotary_pct)
            if (
                type(nested_theta) not in {int, float}
                or not math.isfinite(float(nested_theta))
                or float(nested_theta) <= 0
            ):
                raise ConfigurationError("rope_parameters.rope_theta must be positive and finite")
            if (
                type(nested_pct) not in {int, float}
                or not math.isfinite(float(nested_pct))
                or not 0 < float(nested_pct) <= 1
            ):
                raise ConfigurationError(
                    "rope_parameters.partial_rotary_factor must be finite and in (0, 1]"
                )
            if "rotary_emb_base" in config and float(nested_theta) != legacy_theta:
                raise ConfigurationError("legacy and canonical GPT-NeoX RoPE bases disagree")
            if "rotary_pct" in config and float(nested_pct) != rotary_pct:
                raise ConfigurationError("legacy and canonical GPT-NeoX rotary fractions disagree")
            rope_theta = float(nested_theta)
            rotary_pct = float(nested_pct)

        rotary_dim = int(derived_head_dim * rotary_pct)
        if rotary_dim <= 0 or rotary_dim > derived_head_dim or rotary_dim % 2:
            raise ConfigurationError(
                "GPT-NeoX rotary dimension must be positive, even, and no larger than head_dim",
                details={
                    "head_dim": derived_head_dim,
                    "rotary_pct": rotary_pct,
                    "rotary_dim": rotary_dim,
                },
            )
        return _GPTNeoXConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=derived_head_dim,
            rotary_dim=rotary_dim,
            vocab_size=_positive_int(config, "vocab_size"),
            max_position_embeddings=_positive_int(config, "max_position_embeddings"),
            layer_norm_eps=_positive_float(config, "layer_norm_eps", 1e-5),
            rope_theta=rope_theta,
            rotary_pct=rotary_pct,
            tie_word_embeddings=False,
            attention_bias=attention_bias,
            use_parallel_residual=use_parallel_residual,
            hidden_act="gelu",
        )

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        config = source.config
        if config.get("model_type") != self.model_type:
            return ()
        unsupported: list[UnsupportedFeature] = []
        architectures = config.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or not architectures
            or any(
                type(item) is not str or item not in self.architecture_names
                for item in architectures
            )
        ):
            unsupported.append(
                UnsupportedFeature(
                    code="architecture_declaration",
                    field="architectures",
                    observed=_canonical_observed(architectures),
                    reason="architecture declaration is not owned by the GPT-NeoX adapter",
                )
            )
        unknown_keys = sorted(set(config) - _COMMON_CONFIG_KEYS - _GPT_NEOX_CONFIG_KEYS)
        if unknown_keys:
            unsupported.append(
                UnsupportedFeature(
                    code="unknown_config_keys",
                    field="config",
                    observed=",".join(unknown_keys),
                    reason="unknown GPT-NeoX config keys may affect execution",
                )
            )
        if config.get("quantization_config") not in (None, {}):
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="quantization_config",
                    observed=_canonical_observed(config.get("quantization_config")),
                    reason="quantized source metadata requires a dedicated codec adapter",
                )
            )
        if config.get("auto_map") not in (None, {}, []):
            unsupported.append(
                UnsupportedFeature(
                    code="remote_code_required",
                    field="auto_map",
                    observed=_canonical_observed(config.get("auto_map")),
                    reason="repository code is never imported by this adapter",
                )
            )
        if config.get("rope_scaling") is not None:
            unsupported.append(
                UnsupportedFeature(
                    code="rope_scaling",
                    field="rope_scaling",
                    observed=_canonical_observed(config.get("rope_scaling")),
                    reason="registered GPT-NeoX semantics cover default partial RoPE only",
                )
            )
        non_float = sorted({record.storage_dtype for record in index.tensors} - _RAW_FLOAT_DTYPES)
        if non_float:
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="safetensors.dtype",
                    observed=",".join(non_float),
                    reason="raw-float adapter cannot reinterpret packed or integer codes",
                )
            )
        try:
            self._parse_config(config)
        except ConfigurationError as exc:
            unsupported.append(
                UnsupportedFeature(
                    code="invalid_semantic_config",
                    field="config",
                    observed=exc.message,
                    reason="semantic configuration is invalid for the GPT-NeoX adapter",
                )
            )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))

    def match(self, source: FrozenSourceBundle, index: TensorIndex) -> MatchResult:
        if index.source_fingerprint != source.fingerprint:
            raise ValueError("tensor index is not bound to the supplied frozen source")
        matched = source.config.get("model_type") == self.model_type
        names = index.by_name()
        architectures = source.config.get("architectures", "absent")
        evidence = tuple(
            sorted(
                (
                    MatchEvidence(
                        predicate="config.architectures",
                        expected="|".join(sorted(self.architecture_names)),
                        observed=_canonical_observed(architectures),
                        matched=(
                            architectures == "absent"
                            or (
                                isinstance(architectures, list)
                                and bool(architectures)
                                and all(
                                    type(item) is str and item in self.architecture_names
                                    for item in architectures
                                )
                            )
                        ),
                    ),
                    MatchEvidence(
                        predicate="config.model_type",
                        expected=self.model_type,
                        observed=_canonical_observed(source.config.get("model_type")),
                        matched=matched,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.embedding",
                        expected="gpt_neox.embed_in.weight",
                        observed=("present" if "gpt_neox.embed_in.weight" in names else "absent"),
                        matched="gpt_neox.embed_in.weight" in names,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.layer0_fused_qkv",
                        expected="gpt_neox.layers.0.attention.query_key_value.weight",
                        observed=(
                            "present"
                            if "gpt_neox.layers.0.attention.query_key_value.weight" in names
                            else "absent"
                        ),
                        matched="gpt_neox.layers.0.attention.query_key_value.weight" in names,
                    ),
                ),
                key=lambda item: item.predicate,
            )
        )
        unsupported = self._unsupported_features(source, index) if matched else ()
        codecs = (
            ("raw-float",)
            if all(record.storage_dtype in _RAW_FLOAT_DTYPES for record in index.tensors)
            else ()
        )
        return MatchResult.build(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            matched=matched,
            supported=matched and not unsupported,
            strength=100 if matched else 0,
            evidence=evidence,
            rejected_reasons=(() if matched else ("model_type is not 'gpt_neox'",)),
            required_tensor_patterns=(
                "embed_out.weight",
                "gpt_neox.embed_in.weight",
                "gpt_neox.final_layer_norm.{weight,bias}",
                "gpt_neox.layers.<0..L-1>.<exact fused-QKV block schema>",
            ),
            forbidden_tensor_patterns=(
                "*.g_idx",
                "*.qweight",
                "*.qzeros",
                "unowned tensor names",
            ),
            source_codec_candidates=codecs,
            unsupported_features=unsupported,
        )

    def _expected_specs(self, config: _GPTNeoXConfig) -> dict[str, _TensorSpec]:
        specs: dict[str, _TensorSpec] = {}

        def add(
            source_name: str,
            logical_name: str,
            shape: tuple[int, ...],
            semantic_role: str,
        ) -> None:
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=shape,
                semantic_role=semantic_role,
                rule_id=f"{self.adapter_id}:fused-neox-v1",
            )

        h = config.hidden_size
        m = config.intermediate_size
        add(
            "gpt_neox.embed_in.weight",
            "token_embedding.weight",
            (config.vocab_size, h),
            "token-embedding",
        )
        add("embed_out.weight", "lm_head.weight", (config.vocab_size, h), "output-readout")
        for suffix in ("weight", "bias"):
            add(
                f"gpt_neox.final_layer_norm.{suffix}",
                f"final_norm.{suffix}",
                (h,),
                f"final-layer-norm-{suffix}",
            )
        for layer in range(config.num_hidden_layers):
            source = f"gpt_neox.layers.{layer}"
            logical = f"layers.{layer}"
            for source_norm, logical_norm, role in (
                ("input_layernorm", "attention_norm", "pre-attention-layer-norm"),
                ("post_attention_layernorm", "mlp_norm", "pre-mlp-layer-norm"),
            ):
                for suffix in ("weight", "bias"):
                    add(
                        f"{source}.{source_norm}.{suffix}",
                        f"{logical}.{logical_norm}.{suffix}",
                        (h,),
                        f"{role}-{suffix}",
                    )
            add(
                f"{source}.attention.query_key_value.weight",
                f"{logical}.attention.query_key_value.weight",
                (3 * h, h),
                "attention-fused-per-head-qkv",
            )
            if config.attention_bias:
                add(
                    f"{source}.attention.query_key_value.bias",
                    f"{logical}.attention.query_key_value.bias",
                    (3 * h,),
                    "attention-fused-per-head-qkv-bias",
                )
            add(
                f"{source}.attention.dense.weight",
                f"{logical}.attention.o_proj.weight",
                (h, h),
                "attention-output-projection",
            )
            if config.attention_bias:
                add(
                    f"{source}.attention.dense.bias",
                    f"{logical}.attention.o_proj.bias",
                    (h,),
                    "attention-output-projection-bias",
                )
            for source_projection, logical_projection, shape, role in (
                ("dense_h_to_4h", "dense_h_to_4h", (m, h), "mlp-expansion"),
                ("dense_4h_to_h", "dense_4h_to_h", (h, m), "mlp-contraction"),
            ):
                add(
                    f"{source}.mlp.{source_projection}.weight",
                    f"{logical}.mlp.{logical_projection}.weight",
                    shape,
                    role,
                )
                add(
                    f"{source}.mlp.{source_projection}.bias",
                    f"{logical}.mlp.{logical_projection}.bias",
                    (shape[0],),
                    f"{role}-bias",
                )
        return specs

    def _map_weights(
        self, source: FrozenSourceBundle, index: TensorIndex, config: _GPTNeoXConfig
    ) -> tuple[PhysicalWeightIR, _MappedContext]:
        records = index.by_name()
        specs = self._expected_specs(config)
        missing = sorted(set(specs) - set(records))
        if missing:
            raise CoverageError(
                "required GPT-NeoX source tensors are absent",
                details={"missing_tensors": missing},
            )
        unexplained = sorted(set(records) - set(specs))
        if unexplained:
            raise CoverageError(
                "GPT-NeoX adapter cannot classify every source tensor",
                details={"unexplained_tensors": unexplained},
            )
        non_float = sorted(
            record.source_name
            for record in records.values()
            if record.storage_dtype not in _RAW_FLOAT_DTYPES
        )
        if non_float:
            raise CodecError(
                "raw-float GPT-NeoX adapter encountered non-float storage",
                details={"tensors": non_float},
            )

        allocations: list[PhysicalAllocationIR] = []
        views: list[TensorViewIR] = []
        classifications: list[TensorClassificationIR] = []
        for source_name in sorted(records):
            record = records[source_name]
            spec = specs[source_name]
            lexical = source_name in {"gpt_neox.embed_in.weight", "embed_out.weight"}
            lexical_shape_valid = (
                lexical
                and len(record.shape) == 2
                and record.shape[0] >= config.vocab_size
                and record.shape[1] == config.hidden_size
            )
            if not lexical_shape_valid and record.shape != spec.shape:
                raise ConfigurationError(
                    f"tensor shape mismatch for {source_name}",
                    details={"expected": list(spec.shape), "actual": list(record.shape)},
                )
            allocation_id = _allocation_id(source_name, record.range_identity)
            view_id = _view_id(spec.logical_name)
            allocations.append(
                PhysicalAllocationIR(
                    allocation_id=allocation_id,
                    source_tensor=source_name,
                    source_file=record.source_file,
                    byte_offset=record.byte_offset,
                    byte_length=record.byte_length,
                    stored_shape=record.shape,
                    stored_dtype=record.storage_dtype,
                    codec=CodecBindingIR(
                        codec_id="raw-float",
                        codec_version="1.0.0",
                        stored_dtype=record.storage_dtype,
                        _parameters_json=canonical_json(
                            {
                                "byte_order": "little",
                                "packing": "none",
                                "value_semantics": "safetensors-native-float",
                            }
                        ),
                    ),
                    content_fingerprint=record.range_identity,
                )
            )
            views.append(
                TensorViewIR(
                    view_id=view_id,
                    logical_name=spec.logical_name,
                    allocation_id=allocation_id,
                    logical_shape=record.shape,
                    transforms=(
                        ViewTransformIR(kind="identity", _parameters_json=canonical_json({})),
                    ),
                    semantic_role=spec.semantic_role,
                    parameter_kind="parameter",
                )
            )
            classifications.append(
                TensorClassificationIR(
                    source_name=source_name,
                    allocation_id=allocation_id,
                    disposition="parameter",
                    logical_view_ids=(view_id,),
                    rule_id=spec.rule_id,
                    reason="exact adapter-owned GPT-NeoX source tensor schema",
                )
            )
        physical = PhysicalWeightIR.build(
            source_fingerprint=source.fingerprint,
            tensor_index_fingerprint=index.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            allocations=tuple(sorted(allocations, key=lambda item: item.allocation_id)),
            views=tuple(sorted(views, key=lambda item: item.logical_name)),
            alias_classes=(),
            classifications=tuple(sorted(classifications, key=lambda item: item.source_name)),
        )
        input_rows = records["gpt_neox.embed_in.weight"].shape[0]
        output_rows = records["embed_out.weight"].shape[0]
        if input_rows < config.vocab_size or output_rows < config.vocab_size:
            raise IOContractError(
                "configured GPT-NeoX token space exceeds lexical row capacity",
                details={
                    "vocab_size": config.vocab_size,
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                },
            )
        return physical, _MappedContext(
            config=config,
            input_rows=input_rows,
            output_rows=output_rows,
            rotary_parameters_by_layer=tuple(() for _ in range(config.num_hidden_layers)),
        )

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        if not isinstance(config, _GPTNeoXConfig):
            raise TypeError("GPT-NeoX model builder received a foreign config")
        parameters = tuple(
            sorted(
                (
                    LogicalParameterRefIR(
                        logical_name=view.logical_name,
                        view_id=view.view_id,
                        semantic_role=view.semantic_role,
                        parameter_kind=view.parameter_kind,
                    )
                    for view in physical.views
                ),
                key=lambda item: item.logical_name,
            )
        )
        operations: list[OperationIR] = [
            _operation(
                "embedding",
                "token-embedding",
                ("token_ids",),
                ("embedding.hidden",),
                ("token_embedding.weight",),
                {"padding_policy": "IOIR-row-mapper"},
            )
        ]
        current = "embedding.hidden"
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            operations.append(
                _operation(
                    f"{prefix}.attention_norm",
                    "layer-norm",
                    (current,),
                    (f"{prefix}.attention_norm.hidden",),
                    tuple(
                        sorted(
                            (
                                f"{prefix}.attention_norm.weight",
                                f"{prefix}.attention_norm.bias",
                            )
                        )
                    ),
                    {"epsilon": config.layer_norm_eps},
                )
            )
            normed = f"{prefix}.attention_norm.hidden"
            operations.append(
                _operation(
                    f"{attention}.query_key_value",
                    "linear",
                    (normed,),
                    (f"{attention}.packed_qkv",),
                    self._linear_parameters(
                        attention, "query_key_value", bias=config.attention_bias
                    ),
                    {"weight_orientation": "out-in"},
                )
            )
            operations.append(
                _operation(
                    f"{attention}.unpack_qkv",
                    "gpt-neox-qkv-unpack",
                    (f"{attention}.packed_qkv",),
                    (f"{attention}.q", f"{attention}.k", f"{attention}.v"),
                    attributes={
                        "head_dim": config.head_dim,
                        "num_attention_heads": config.num_attention_heads,
                        "packing": "per-head-qkv",
                    },
                )
            )
            operations.append(
                _operation(
                    f"{attention}.rotary",
                    "rotary-partial-default",
                    (f"{attention}.q", f"{attention}.k"),
                    (f"{attention}.q_rotary", f"{attention}.k_rotary"),
                    attributes={
                        "head_dim": config.head_dim,
                        "max_position_embeddings": config.max_position_embeddings,
                        "rope_theta": config.rope_theta,
                        "rotary_dim": config.rotary_dim,
                    },
                )
            )
            operations.append(
                _operation(
                    f"{attention}.mha",
                    "causal-grouped-query-attention",
                    (
                        f"{attention}.q_rotary",
                        f"{attention}.k_rotary",
                        f"{attention}.v",
                    ),
                    (f"{attention}.context",),
                    attributes={
                        "head_dim": config.head_dim,
                        "num_attention_heads": config.num_attention_heads,
                        "num_key_value_heads": config.num_key_value_heads,
                        "scale": config.head_dim**-0.5,
                        "state_slots": [
                            f"layers.{layer}.k_cache",
                            f"layers.{layer}.v_cache",
                            "position",
                        ],
                    },
                )
            )
            operations.append(
                _operation(
                    f"{attention}.o_proj",
                    "linear",
                    (f"{attention}.context",),
                    (f"{attention}.output",),
                    self._linear_parameters(attention, "o_proj", bias=config.attention_bias),
                    {"weight_orientation": "out-in"},
                )
            )
            attention_residual = f"{prefix}.attention_residual.hidden"
            operations.append(
                _operation(
                    f"{prefix}.attention_residual",
                    "residual-add",
                    (current, f"{attention}.output"),
                    (attention_residual,),
                )
            )
            mlp_base = current if config.use_parallel_residual else attention_residual
            operations.append(
                _operation(
                    f"{prefix}.mlp_norm",
                    "layer-norm",
                    (mlp_base,),
                    (f"{prefix}.mlp_norm.hidden",),
                    tuple(sorted((f"{prefix}.mlp_norm.weight", f"{prefix}.mlp_norm.bias"))),
                    {"epsilon": config.layer_norm_eps},
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.mlp.dense_h_to_4h",
                    "linear",
                    (f"{prefix}.mlp_norm.hidden",),
                    (f"{prefix}.mlp.expanded",),
                    self._linear_parameters(f"{prefix}.mlp", "dense_h_to_4h", bias=True),
                    {"weight_orientation": "out-in"},
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.mlp.gelu",
                    "gelu-erf",
                    (f"{prefix}.mlp.expanded",),
                    (f"{prefix}.mlp.activated",),
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.mlp.dense_4h_to_h",
                    "linear",
                    (f"{prefix}.mlp.activated",),
                    (f"{prefix}.mlp.output",),
                    self._linear_parameters(f"{prefix}.mlp", "dense_4h_to_h", bias=True),
                    {"weight_orientation": "out-in"},
                )
            )
            current = f"{prefix}.output"
            operations.append(
                _operation(
                    f"{prefix}.mlp_residual",
                    "residual-add",
                    (attention_residual, f"{prefix}.mlp.output"),
                    (current,),
                )
            )
        operations.extend(
            (
                _operation(
                    "final_norm",
                    "layer-norm",
                    (current,),
                    ("final.hidden",),
                    ("final_norm.bias", "final_norm.weight"),
                    {"epsilon": config.layer_norm_eps},
                ),
                _operation(
                    "lm_head",
                    "linear-readout",
                    ("final.hidden",),
                    ("logits",),
                    ("lm_head.weight",),
                    {"weight_orientation": "rows-hidden"},
                ),
            )
        )
        return ModelIR.build(
            source_fingerprint=source.fingerprint,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            architecture_id=self.architecture_id,
            physical_weights_fingerprint=physical.fingerprint,
            dimensions=ModelDimensionsIR(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                vocab_size=config.vocab_size,
                physical_vocab_rows=context.input_rows,
                max_position_embeddings=config.max_position_embeddings,
            ),
            parameters=parameters,
            operations=tuple(operations),
            state_refs=self._state_slot_ids(config),
            input_ports=(
                PortIR(
                    name="token_ids",
                    semantic="model-token-ids",
                    dtype="int64",
                    shape=("batch", "sequence"),
                    value="token_ids",
                ),
            ),
            output_ports=(
                PortIR(
                    name="hidden_states",
                    semantic="final-normalized-hidden",
                    dtype="activation",
                    shape=("batch", "sequence", "hidden_size"),
                    value="final.hidden",
                ),
                PortIR(
                    name="logits",
                    semantic="token-row-logits",
                    dtype="activation",
                    shape=("batch", "sequence", "output_rows"),
                    value="logits",
                ),
            ),
            numerical_semantics=NumericalSemanticsIR(
                reference_contract="source-reference-required-before-u3",
                accumulation="lowering-declared",
                softmax="stable-causal-softmax",
                positional_arithmetic="gpt-neox-partial-default-rope-source-dtype",
                optimization_contract="preregister-before-target-execution",
            ),
        )


class Qwen2Adapter(DenseRotaryAdapter):
    adapter_id = "mrun.hf.qwen2-dense"
    model_type = "qwen2"
    architecture_id = "qwen2-dense-causal-decoder"
    architecture_names = frozenset({"Qwen2ForCausalLM", "Qwen2Model"})
    qwen2_fixed_qkv_bias = True


class MistralAdapter(DenseRotaryAdapter):
    adapter_id = "mrun.hf.mistral-dense"
    model_type = "mistral"
    architecture_id = "mistral-dense-causal-decoder"
    architecture_names = frozenset({"MistralForCausalLM", "MistralModel"})

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        unsupported = list(super()._unsupported_features(source, index))
        if source.config.get("model_type") == self.model_type:
            sliding_window = source.config.get("sliding_window")
            if sliding_window is not None:
                unsupported.append(
                    UnsupportedFeature(
                        code="sliding_attention",
                        field="sliding_window",
                        observed=_canonical_observed(sliding_window),
                        reason=(
                            "Mistral sliding attention requires a windowed StateIR and "
                            "reference target"
                        ),
                    )
                )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))


class MixtralAdapter(DenseRotaryAdapter):
    """Exact raw-float adapter for the serialized HF Mixtral sparse-MoE schema.

    Transformers may fuse experts after loading, but its portable safetensors contract remains a
    router plus per-expert ``w1``/``w2``/``w3`` tensors.  This adapter owns only that source
    contract.  Shared experts, fused source tensors, stochastic routing, and windowed attention
    remain distinct variants and therefore fail closed.
    """

    adapter_id = "mrun.hf.mixtral-sparse-moe"
    model_type = "mixtral"
    architecture_id = "mixtral-sparse-moe-causal-decoder"
    architecture_names = frozenset({"MixtralForCausalLM"})
    extra_config_keys = _MIXTRAL_CONFIG_KEYS
    rule_set = "mixtral-classic-safetensors-topk-renormalized-v1"

    def _parse_config(self, config: dict[str, Any]) -> _MixtralConfig:
        base = super()._parse_config(config)
        if base.attention_bias:
            raise ConfigurationError("registered Mixtral attention is biasless")
        num_local_experts = _positive_int(config, "num_local_experts")
        num_experts_per_tok = _positive_int(config, "num_experts_per_tok")
        if num_local_experts < 2:
            raise ConfigurationError("Mixtral requires at least two routed experts")
        if num_experts_per_tok >= num_local_experts:
            raise ConfigurationError(
                "Mixtral top-k must be smaller than the routed expert count",
                details={
                    "num_experts_per_tok": num_experts_per_tok,
                    "num_local_experts": num_local_experts,
                },
            )
        raw_aux = config.get("router_aux_loss_coef", 0.001)
        if (
            type(raw_aux) not in {int, float}
            or not math.isfinite(float(raw_aux))
            or float(raw_aux) < 0
        ):
            raise ConfigurationError("router_aux_loss_coef must be finite and non-negative")
        # The coefficient affects an optional training loss, not this logits-only inference IR.
        raw_jitter = config.get("router_jitter_noise", 0.0)
        if (
            type(raw_jitter) not in {int, float}
            or not math.isfinite(float(raw_jitter))
            or float(raw_jitter) < 0
        ):
            raise ConfigurationError("router_jitter_noise must be finite and non-negative")
        return _MixtralConfig(
            hidden_size=base.hidden_size,
            intermediate_size=base.intermediate_size,
            num_hidden_layers=base.num_hidden_layers,
            num_attention_heads=base.num_attention_heads,
            num_key_value_heads=base.num_key_value_heads,
            head_dim=base.head_dim,
            vocab_size=base.vocab_size,
            max_position_embeddings=base.max_position_embeddings,
            rms_norm_eps=base.rms_norm_eps,
            rope_theta=base.rope_theta,
            tie_word_embeddings=base.tie_word_embeddings,
            attention_bias=False,
            mlp_bias=False,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_jitter_noise=float(raw_jitter),
            output_router_logits=_optional_bool(config, "output_router_logits", False),
        )

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        unsupported = list(super()._unsupported_features(source, index))
        if source.config.get("model_type") == self.model_type:
            sliding_window = source.config.get("sliding_window")
            if sliding_window is not None:
                unsupported.append(
                    UnsupportedFeature(
                        code="sliding_attention",
                        field="sliding_window",
                        observed=_canonical_observed(sliding_window),
                        reason=(
                            "Mixtral sliding attention requires a windowed StateIR and "
                            "registered reference target"
                        ),
                    )
                )
            if source.config.get("output_router_logits") is True:
                unsupported.append(
                    UnsupportedFeature(
                        code="router_output_contract",
                        field="output_router_logits",
                        observed="true",
                        reason="router-logit outputs require an additional declared output space",
                    )
                )
            raw_jitter = source.config.get("router_jitter_noise", 0.0)
            if type(raw_jitter) in {int, float} and float(raw_jitter) != 0.0:
                unsupported.append(
                    UnsupportedFeature(
                        code="stochastic_router",
                        field="router_jitter_noise",
                        observed=_canonical_observed(raw_jitter),
                        reason="the registered inference graph has deterministic router inputs",
                    )
                )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))

    def match(self, source: FrozenSourceBundle, index: TensorIndex) -> MatchResult:
        base = super().match(source, index)
        names = index.by_name()
        router_name = "model.layers.0.block_sparse_moe.gate.weight"
        expert_name = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        evidence = tuple(
            sorted(
                (
                    *base.evidence,
                    MatchEvidence(
                        predicate="tensor.anchor.layer0_expert0_w1",
                        expected=expert_name,
                        observed="present" if expert_name in names else "absent",
                        matched=expert_name in names,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.layer0_router",
                        expected=router_name,
                        observed="present" if router_name in names else "absent",
                        matched=router_name in names,
                    ),
                ),
                key=lambda item: item.predicate,
            )
        )
        return MatchResult.build(
            adapter_id=base.adapter_id,
            adapter_version=base.adapter_version,
            adapter_fingerprint=base.adapter_fingerprint,
            matched=base.matched,
            supported=base.supported,
            strength=base.strength,
            evidence=evidence,
            rejected_reasons=base.rejected_reasons,
            required_tensor_patterns=tuple(
                sorted(
                    {
                        "model.embed_tokens.weight",
                        "model.layers.<0..L-1>.block_sparse_moe.gate.weight",
                        "model.layers.<0..L-1>.block_sparse_moe.experts.<0..E-1>.<w1|w2|w3>.weight",
                        "model.layers.<0..L-1>.<exact biasless attention schema>",
                        "model.norm.weight",
                    }
                )
            ),
            forbidden_tensor_patterns=tuple(
                sorted(
                    {
                        *base.forbidden_tensor_patterns,
                        "*.block_sparse_moe.shared_expert.*",
                        "*.mlp.experts.down_proj",
                        "*.mlp.experts.gate_up_proj",
                    }
                )
            ),
            source_codec_candidates=base.source_codec_candidates,
            unsupported_features=base.unsupported_features,
        )

    def _expected_specs(self, config: _DenseConfig) -> dict[str, _TensorSpec]:
        if not isinstance(config, _MixtralConfig):
            raise TypeError("Mixtral tensor mapper received a foreign config")
        rule_id = f"{self.adapter_id}:classic-per-expert-safetensors-v1"
        dense_specs = super()._expected_specs(config)
        specs: dict[str, _TensorSpec] = {}
        for source_name, spec in dense_specs.items():
            if ".mlp." in source_name:
                continue
            logical_name = spec.logical_name
            semantic_role = spec.semantic_role
            if source_name.endswith(".post_attention_layernorm.weight"):
                layer = source_name.split(".")[2]
                logical_name = f"layers.{layer}.moe_norm.weight"
                semantic_role = "pre-routed-moe-rms-norm"
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=spec.shape,
                semantic_role=semantic_role,
                parameter_kind=spec.parameter_kind,
                rule_id=rule_id,
            )

        def add(
            source_name: str,
            logical_name: str,
            shape: tuple[int, ...],
            semantic_role: str,
        ) -> None:
            if source_name in specs:
                raise AssertionError(f"duplicate Mixtral source rule: {source_name}")
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=shape,
                semantic_role=semantic_role,
                rule_id=rule_id,
            )

        for layer in range(config.num_hidden_layers):
            source = f"model.layers.{layer}.block_sparse_moe"
            logical = f"layers.{layer}.moe"
            add(
                f"{source}.gate.weight",
                f"{logical}.router.weight",
                (config.num_local_experts, config.hidden_size),
                "moe-router-routed-only",
            )
            for expert in range(config.num_local_experts):
                source_expert = f"{source}.experts.{expert}"
                logical_expert = f"{logical}.routed_experts.{expert}"
                for source_projection, logical_projection, shape, role in (
                    (
                        "w1",
                        "gate_proj",
                        (config.intermediate_size, config.hidden_size),
                        "moe-routed-expert-gate",
                    ),
                    (
                        "w2",
                        "down_proj",
                        (config.hidden_size, config.intermediate_size),
                        "moe-routed-expert-down",
                    ),
                    (
                        "w3",
                        "up_proj",
                        (config.intermediate_size, config.hidden_size),
                        "moe-routed-expert-up",
                    ),
                ):
                    add(
                        f"{source_expert}.{source_projection}.weight",
                        f"{logical_expert}.{logical_projection}.weight",
                        shape,
                        role,
                    )
        return specs

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        if not isinstance(config, _MixtralConfig):
            raise TypeError("Mixtral model builder received a foreign config")
        parameters = tuple(
            sorted(
                (
                    LogicalParameterRefIR(
                        logical_name=view.logical_name,
                        view_id=view.view_id,
                        semantic_role=view.semantic_role,
                        parameter_kind=view.parameter_kind,
                    )
                    for view in physical.views
                ),
                key=lambda item: item.logical_name,
            )
        )
        operations: list[OperationIR] = [
            _operation(
                "embedding",
                "token-embedding",
                ("token_ids",),
                ("embedding.hidden",),
                ("token_embedding.weight",),
                {"padding_policy": "IOIR-row-mapper"},
            )
        ]
        current = "embedding.hidden"
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}"
            attention = f"{prefix}.attention"
            operations.append(
                _operation(
                    f"{prefix}.attention_norm",
                    "rms-norm",
                    (current,),
                    (f"{prefix}.attention_norm.hidden",),
                    (f"{prefix}.attention_norm.weight",),
                    {"epsilon": config.rms_norm_eps},
                )
            )
            normed = f"{prefix}.attention_norm.hidden"
            for projection, output in (
                ("q_proj", f"{attention}.q"),
                ("k_proj", f"{attention}.k"),
                ("v_proj", f"{attention}.v"),
            ):
                operations.append(
                    _operation(
                        f"{attention}.{projection}",
                        "linear",
                        (normed,),
                        (output,),
                        (f"{attention}.{projection}.weight",),
                        {"weight_orientation": "out-in"},
                    )
                )
            operations.append(
                _operation(
                    f"{attention}.rotary",
                    "rotary-default",
                    (f"{attention}.q", f"{attention}.k"),
                    (f"{attention}.q_rotary", f"{attention}.k_rotary"),
                    context.rotary_parameters_by_layer[layer],
                    {
                        "head_dim": config.head_dim,
                        "max_position_embeddings": config.max_position_embeddings,
                        "rope_theta": config.rope_theta,
                    },
                )
            )
            operations.append(
                _operation(
                    f"{attention}.gqa",
                    "causal-grouped-query-attention",
                    (
                        f"{attention}.q_rotary",
                        f"{attention}.k_rotary",
                        f"{attention}.v",
                    ),
                    (f"{attention}.context",),
                    attributes={
                        "head_dim": config.head_dim,
                        "num_attention_heads": config.num_attention_heads,
                        "num_key_value_heads": config.num_key_value_heads,
                        "scale": config.head_dim**-0.5,
                        "state_slots": [
                            f"layers.{layer}.k_cache",
                            f"layers.{layer}.v_cache",
                            "position",
                        ],
                    },
                )
            )
            operations.append(
                _operation(
                    f"{attention}.o_proj",
                    "linear",
                    (f"{attention}.context",),
                    (f"{attention}.output",),
                    (f"{attention}.o_proj.weight",),
                    {"weight_orientation": "out-in"},
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.attention_residual",
                    "residual-add",
                    (current, f"{attention}.output"),
                    (f"{prefix}.attention_residual.hidden",),
                )
            )
            residual = f"{prefix}.attention_residual.hidden"
            moe = f"{prefix}.moe"
            operations.append(
                _operation(
                    f"{prefix}.moe_norm",
                    "rms-norm",
                    (residual,),
                    (f"{moe}.input",),
                    (f"{prefix}.moe_norm.weight",),
                    {"epsilon": config.rms_norm_eps},
                )
            )
            operations.append(
                _operation(
                    f"{moe}.router",
                    "moe-router-linear",
                    (f"{moe}.input",),
                    (f"{moe}.router_logits",),
                    (f"{moe}.router.weight",),
                    {
                        "expert_scope": "routed-only",
                        "num_routed_experts": config.num_local_experts,
                        "num_shared_experts": 0,
                        "router_bias": False,
                        "weight_orientation": "experts-hidden",
                    },
                )
            )
            operations.append(
                _operation(
                    f"{moe}.top_k",
                    "moe-top-k-softmax",
                    (f"{moe}.router_logits",),
                    (f"{moe}.routing_weights", f"{moe}.selected_experts"),
                    attributes={
                        "jitter_noise": config.router_jitter_noise,
                        "num_experts_per_token": config.num_experts_per_tok,
                        "num_routed_experts": config.num_local_experts,
                        "renormalize_selected_probabilities": True,
                        "selection": "top-k-after-softmax",
                        "softmax_dtype": "float32",
                        "tie_breaking": "source-framework-defined",
                    },
                )
            )
            expert_inputs = tuple(
                f"{moe}.routed_experts.{expert}.input" for expert in range(config.num_local_experts)
            )
            operations.append(
                _operation(
                    f"{moe}.dispatch",
                    "moe-token-dispatch",
                    (f"{moe}.input", f"{moe}.selected_experts"),
                    expert_inputs,
                    attributes={
                        "expert_scope": "routed-only",
                        "num_routed_experts": config.num_local_experts,
                        "num_shared_experts": 0,
                    },
                )
            )
            expert_outputs: list[str] = []
            for expert in range(config.num_local_experts):
                expert_prefix = f"{moe}.routed_experts.{expert}"
                expert_input = f"{expert_prefix}.input"
                for projection in ("gate_proj", "up_proj"):
                    operations.append(
                        _operation(
                            f"{expert_prefix}.{projection}",
                            "moe-routed-expert-linear",
                            (expert_input,),
                            (f"{expert_prefix}.{projection}.hidden",),
                            (f"{expert_prefix}.{projection}.weight",),
                            {
                                "expert_id": expert,
                                "expert_scope": "routed",
                                "weight_orientation": "out-in",
                            },
                        )
                    )
                operations.append(
                    _operation(
                        f"{expert_prefix}.silu",
                        "silu",
                        (f"{expert_prefix}.gate_proj.hidden",),
                        (f"{expert_prefix}.gate_activated",),
                    )
                )
                operations.append(
                    _operation(
                        f"{expert_prefix}.multiply",
                        "elementwise-multiply",
                        (
                            f"{expert_prefix}.gate_activated",
                            f"{expert_prefix}.up_proj.hidden",
                        ),
                        (f"{expert_prefix}.intermediate",),
                    )
                )
                expert_output = f"{expert_prefix}.output"
                operations.append(
                    _operation(
                        f"{expert_prefix}.down_proj",
                        "moe-routed-expert-linear",
                        (f"{expert_prefix}.intermediate",),
                        (expert_output,),
                        (f"{expert_prefix}.down_proj.weight",),
                        {
                            "expert_id": expert,
                            "expert_scope": "routed",
                            "weight_orientation": "out-in",
                        },
                    )
                )
                expert_outputs.append(expert_output)
            operations.append(
                _operation(
                    f"{moe}.combine",
                    "moe-weighted-scatter-add",
                    (
                        f"{moe}.routing_weights",
                        f"{moe}.selected_experts",
                        *expert_outputs,
                    ),
                    (f"{moe}.output",),
                    attributes={
                        "accumulation_order": "source-expert-index-order",
                        "expert_scope": "routed-only",
                        "num_routed_experts": config.num_local_experts,
                        "num_shared_experts": 0,
                    },
                )
            )
            current = f"{prefix}.output"
            operations.append(
                _operation(
                    f"{prefix}.moe_residual",
                    "residual-add",
                    (residual, f"{moe}.output"),
                    (current,),
                )
            )
        operations.extend(
            (
                _operation(
                    "final_norm",
                    "rms-norm",
                    (current,),
                    ("final.hidden",),
                    ("final_norm.weight",),
                    {"epsilon": config.rms_norm_eps},
                ),
                _operation(
                    "lm_head",
                    "linear-readout",
                    ("final.hidden",),
                    ("logits",),
                    ("lm_head.weight",),
                    {"weight_orientation": "rows-hidden"},
                ),
            )
        )
        return ModelIR.build(
            source_fingerprint=source.fingerprint,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            architecture_id=self.architecture_id,
            physical_weights_fingerprint=physical.fingerprint,
            dimensions=ModelDimensionsIR(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                vocab_size=config.vocab_size,
                physical_vocab_rows=context.input_rows,
                max_position_embeddings=config.max_position_embeddings,
            ),
            parameters=parameters,
            operations=tuple(operations),
            state_refs=self._state_slot_ids(config),
            input_ports=(
                PortIR(
                    name="token_ids",
                    semantic="model-token-ids",
                    dtype="int64",
                    shape=("batch", "sequence"),
                    value="token_ids",
                ),
            ),
            output_ports=(
                PortIR(
                    name="hidden_states",
                    semantic="final-normalized-hidden",
                    dtype="activation",
                    shape=("batch", "sequence", "hidden_size"),
                    value="final.hidden",
                ),
                PortIR(
                    name="logits",
                    semantic="token-row-logits",
                    dtype="activation",
                    shape=("batch", "sequence", "output_rows"),
                    value="logits",
                ),
            ),
            numerical_semantics=NumericalSemanticsIR(
                reference_contract="source-reference-required-before-u3",
                accumulation="source-expert-index-scatter-add-requires-reference",
                softmax="stable-causal-and-fp32-router-topk-renormalized",
                positional_arithmetic="default-rope-source-dtype",
                optimization_contract="preregister-before-target-execution",
            ),
        )


class GemmaAdapter(DenseRotaryAdapter):
    adapter_id = "mrun.hf.gemma1-dense"
    model_type = "gemma"
    architecture_id = "gemma1-dense-causal-decoder"
    architecture_names = frozenset({"GemmaForCausalLM", "GemmaModel"})
    extra_config_keys = frozenset({"hidden_activation"})
    rule_set = "gemma1-offset-rmsnorm-geglu-v1"

    def _parse_config(self, config: dict[str, Any]) -> _GemmaConfig:
        hidden_size = _positive_int(config, "hidden_size")
        intermediate_size = _positive_int(config, "intermediate_size")
        num_hidden_layers = _positive_int(config, "num_hidden_layers")
        num_attention_heads = _positive_int(config, "num_attention_heads")
        raw_kv = config.get("num_key_value_heads", num_attention_heads)
        if type(raw_kv) is not int or raw_kv <= 0:
            raise ConfigurationError("Gemma num_key_value_heads must be positive")
        if num_attention_heads % raw_kv:
            raise ConfigurationError("Gemma query heads must be divisible by key/value heads")
        raw_head_dim = config.get("head_dim")
        if raw_head_dim is None:
            if hidden_size % num_attention_heads:
                raise ConfigurationError("Gemma hidden_size must be divisible by attention heads")
            head_dim = hidden_size // num_attention_heads
        elif type(raw_head_dim) is int and raw_head_dim > 0:
            head_dim = raw_head_dim
        else:
            raise ConfigurationError("Gemma head_dim must be a positive integer or null")
        if head_dim % 2:
            raise ConfigurationError("Gemma head_dim must be even for default RoPE")
        hidden_act = config.get("hidden_act", "gelu_pytorch_tanh")
        hidden_activation = config.get("hidden_activation", hidden_act)
        if hidden_activation is None:
            hidden_activation = hidden_act
        if hidden_act != "gelu_pytorch_tanh" or hidden_activation != hidden_act:
            raise ConfigurationError("registered Gemma-1 semantics require gelu_pytorch_tanh")
        if _optional_bool(config, "attention_bias", False):
            raise ConfigurationError("registered Gemma-1 attention is biasless")
        if _optional_bool(config, "mlp_bias", False):
            raise ConfigurationError("registered Gemma-1 MLP is biasless")
        return _GemmaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=raw_kv,
            head_dim=head_dim,
            vocab_size=_positive_int(config, "vocab_size"),
            max_position_embeddings=_positive_int(config, "max_position_embeddings"),
            rms_norm_eps=_positive_float(config, "rms_norm_eps", 1e-6),
            rope_theta=_rope_theta(config),
            tie_word_embeddings=_optional_bool(config, "tie_word_embeddings", True),
            attention_bias=False,
            mlp_bias=False,
            hidden_activation="gelu_pytorch_tanh",
        )

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        if not isinstance(config, _GemmaConfig):
            raise TypeError("Gemma model builder received a foreign config")
        base = super()._build_model(source, physical, context)
        operations: list[OperationIR] = []
        for operation in base.operations:
            if operation.operation_id == "embedding":
                operations.append(
                    _operation(
                        "embedding",
                        "token-embedding",
                        operation.inputs,
                        ("embedding.raw",),
                        operation.parameters,
                        operation.attributes,
                    )
                )
                operations.append(
                    _operation(
                        "embedding_scale",
                        "scalar-multiply",
                        ("embedding.raw",),
                        ("embedding.hidden",),
                        attributes={"scalar": math.sqrt(config.hidden_size)},
                    )
                )
                continue
            kind = operation.kind
            operation_id = operation.operation_id
            if kind == "rms-norm":
                kind = "gemma-rms-norm"
            elif kind == "silu":
                kind = "gelu-tanh"
                operation_id = operation_id.removesuffix(".silu") + ".gelu"
            operations.append(
                _operation(
                    operation_id,
                    kind,
                    operation.inputs,
                    operation.outputs,
                    operation.parameters,
                    operation.attributes,
                )
            )
        return ModelIR.build(
            source_fingerprint=base.source_fingerprint,
            adapter_id=base.adapter_id,
            adapter_version=base.adapter_version,
            adapter_fingerprint=base.adapter_fingerprint,
            architecture_id=base.architecture_id,
            physical_weights_fingerprint=base.physical_weights_fingerprint,
            dimensions=base.dimensions,
            parameters=base.parameters,
            operations=tuple(operations),
            state_refs=base.state_refs,
            input_ports=base.input_ports,
            output_ports=base.output_ports,
            numerical_semantics=NumericalSemanticsIR(
                reference_contract="source-reference-required-before-u3",
                accumulation="lowering-declared",
                softmax="stable-causal-softmax",
                positional_arithmetic="gemma1-scaled-embedding-default-rope-source-dtype",
                optimization_contract="preregister-before-target-execution",
            ),
        )


class MambaAdapter(DenseRotaryAdapter):
    """Exact adapter for the original selective-state-space Mamba causal LM.

    Mamba is deliberately represented as a recurrent architecture.  The canonical model has no
    attention heads, positional embedding, or KV cache; its mutable state is the per-layer causal
    convolution window plus the selective-scan recurrence.  This adapter therefore does not reuse
    the dense decoder's state or invent transformer dimensions merely to fit the schema.
    """

    adapter_id = "mrun.hf.mamba1"
    adapter_version = "1.0.0"
    model_type = "mamba"
    architecture_id = "mamba1-selective-state-space-causal-decoder"
    architecture_names = frozenset({"MambaForCausalLM"})
    rule_set = "mamba1-selective-scan-v1"

    def _parse_config(self, config: dict[str, Any]) -> _MambaConfig:
        hidden_size = _positive_int(config, "hidden_size")
        if "d_model" in config and _positive_int(config, "d_model") != hidden_size:
            raise ConfigurationError("Mamba d_model must equal hidden_size")

        expand = config.get("expand", 2)
        if type(expand) is not int or expand <= 0:
            raise ConfigurationError("Mamba expand must be a positive integer")
        raw_intermediate = config.get("intermediate_size", config.get("d_inner"))
        if raw_intermediate is None:
            intermediate_size = hidden_size * expand
        elif type(raw_intermediate) is int and raw_intermediate > 0:
            intermediate_size = raw_intermediate
        else:
            raise ConfigurationError("Mamba intermediate_size must be a positive integer")
        if "d_inner" in config and _positive_int(config, "d_inner") != intermediate_size:
            raise ConfigurationError("Mamba d_inner must equal intermediate_size")
        if intermediate_size != hidden_size * expand:
            raise ConfigurationError("Mamba intermediate_size must equal hidden_size * expand")

        num_hidden_layers = _positive_int(config, "num_hidden_layers")
        if "n_layer" in config and _positive_int(config, "n_layer") != num_hidden_layers:
            raise ConfigurationError("Mamba n_layer must equal num_hidden_layers")

        raw_rank = config.get("time_step_rank", "auto")
        if raw_rank == "auto":
            time_step_rank = math.ceil(hidden_size / 16)
        elif type(raw_rank) is int and raw_rank > 0:
            time_step_rank = raw_rank
        else:
            raise ConfigurationError("Mamba time_step_rank must be 'auto' or a positive integer")

        if config.get("hidden_act", "silu") != "silu":
            raise ConfigurationError("Mamba adapter supports only the SiLU activation")
        if config.get("rms_norm", True) is not True:
            raise ConfigurationError("Mamba adapter requires RMS normalization")
        if config.get("ssm_cfg", {}) not in ({}, None):
            raise ConfigurationError("non-empty Mamba ssm_cfg requires a separate adapter")
        if config.get("mixer_rms_eps") is not None:
            raise ConfigurationError("Mamba mixer_rms_eps requires a separate adapter")
        if config.get("use_mambapy", False) is not False:
            raise ConfigurationError("MambaPy scan arithmetic is not registered by this adapter")

        for key, default in (
            ("fused_add_norm", False),
            ("rescale_prenorm_residual", False),
            ("use_associative_scan", True),
            ("use_cache", True),
        ):
            _optional_bool(config, key, default)
        if config.get("time_step_init_scheme", "random") not in {"random", "constant"}:
            raise ConfigurationError("unsupported Mamba time_step_init_scheme")
        time_step_min = _positive_float(config, "time_step_min", 0.001)
        time_step_max = _positive_float(config, "time_step_max", 0.1)
        if time_step_min > time_step_max:
            raise ConfigurationError("Mamba time_step_min cannot exceed time_step_max")
        _positive_float(config, "time_step_floor", 0.0001)
        _positive_float(config, "time_step_scale", 1.0)
        pad_multiple = config.get("pad_vocab_size_multiple", 1)
        if type(pad_multiple) is not int or pad_multiple <= 0:
            raise ConfigurationError("Mamba pad_vocab_size_multiple must be a positive integer")

        return _MambaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            state_size=_positive_int(config, "state_size"),
            conv_kernel=_positive_int(config, "conv_kernel"),
            time_step_rank=time_step_rank,
            vocab_size=_positive_int(config, "vocab_size"),
            layer_norm_epsilon=_positive_float(config, "layer_norm_epsilon", 1e-5),
            tie_word_embeddings=_optional_bool(config, "tie_word_embeddings", True),
            use_bias=_optional_bool(config, "use_bias", False),
            use_conv_bias=_optional_bool(config, "use_conv_bias", True),
            residual_in_fp32=_optional_bool(config, "residual_in_fp32", True),
        )

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        config = source.config
        if config.get("model_type") != self.model_type:
            return ()
        unsupported: list[UnsupportedFeature] = []
        architectures = config.get("architectures")
        if not (isinstance(architectures, list) and architectures == ["MambaForCausalLM"]):
            unsupported.append(
                UnsupportedFeature(
                    code="architecture_declaration",
                    field="architectures",
                    observed=_canonical_observed(architectures),
                    reason="only the causal-LM Mamba topology is owned by this adapter",
                )
            )
        unknown_keys = sorted(set(config) - _COMMON_CONFIG_KEYS - _MAMBA_CONFIG_KEYS)
        if unknown_keys:
            unsupported.append(
                UnsupportedFeature(
                    code="unknown_config_keys",
                    field="config",
                    observed=",".join(unknown_keys),
                    reason="unknown Mamba fields may alter recurrence semantics",
                )
            )
        if config.get("quantization_config") not in (None, {}):
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="quantization_config",
                    observed=_canonical_observed(config.get("quantization_config")),
                    reason="packed Mamba source weights need a codec-specific adapter",
                )
            )
        if config.get("auto_map") not in (None, {}):
            unsupported.append(
                UnsupportedFeature(
                    code="repository_code_dependency",
                    field="auto_map",
                    observed=_canonical_observed(config.get("auto_map")),
                    reason="repository code is not imported by the native decompiler",
                )
            )
        non_float = sorted({record.storage_dtype for record in index.tensors} - _RAW_FLOAT_DTYPES)
        if non_float:
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="safetensors.dtype",
                    observed=",".join(non_float),
                    reason="raw-float Mamba adapter cannot reinterpret packed weights",
                )
            )
        try:
            self._parse_config(config)
        except ConfigurationError as exc:
            unsupported.append(
                UnsupportedFeature(
                    code="invalid_semantic_config",
                    field="config",
                    observed=exc.message,
                    reason="semantic configuration is invalid for this Mamba adapter",
                )
            )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))

    def match(self, source: FrozenSourceBundle, index: TensorIndex) -> MatchResult:
        if index.source_fingerprint != source.fingerprint:
            raise ValueError("tensor index is not bound to the supplied frozen source")
        config = source.config
        matched = config.get("model_type") == self.model_type
        names = index.by_name()
        architecture_observed = config.get("architectures", "absent")
        evidence = tuple(
            sorted(
                (
                    MatchEvidence(
                        predicate="config.architectures",
                        expected="MambaForCausalLM",
                        observed=_canonical_observed(architecture_observed),
                        matched=architecture_observed == ["MambaForCausalLM"],
                    ),
                    MatchEvidence(
                        predicate="config.model_type",
                        expected="mamba",
                        observed=_canonical_observed(config.get("model_type")),
                        matched=matched,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.embedding",
                        expected="backbone.embeddings.weight",
                        observed=("present" if "backbone.embeddings.weight" in names else "absent"),
                        matched="backbone.embeddings.weight" in names,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.layer0_ssm",
                        expected="backbone.layers.0.mixer.A_log",
                        observed=(
                            "present" if "backbone.layers.0.mixer.A_log" in names else "absent"
                        ),
                        matched="backbone.layers.0.mixer.A_log" in names,
                    ),
                ),
                key=lambda item: item.predicate,
            )
        )
        unsupported = self._unsupported_features(source, index) if matched else ()
        anchors_match = all(item.matched for item in evidence)
        return MatchResult.build(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            matched=matched,
            supported=matched and anchors_match and not unsupported,
            strength=100 if matched and anchors_match else (80 if matched else 0),
            evidence=evidence,
            rejected_reasons=(() if matched else ("config.model_type is not 'mamba'",)),
            required_tensor_patterns=(
                "backbone.embeddings.weight",
                "backbone.layers.<0..L-1>.<exact Mamba mixer schema>",
                "backbone.norm_f.weight",
            ),
            forbidden_tensor_patterns=(
                "*.g_idx",
                "*.qweight",
                "*.qzeros",
                "unowned tensor names",
            ),
            source_codec_candidates=("raw-float",) if not unsupported else (),
            unsupported_features=unsupported,
        )

    def _expected_specs(self, config: _MambaConfig) -> dict[str, _TensorSpec]:
        specs: dict[str, _TensorSpec] = {}

        def add(
            source_name: str,
            logical_name: str,
            shape: tuple[int, ...],
            semantic_role: str,
        ) -> None:
            if source_name in specs:
                raise AssertionError(f"duplicate Mamba source rule: {source_name}")
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=shape,
                semantic_role=semantic_role,
                rule_id=f"{self.adapter_id}:mamba1-v1",
            )

        h = config.hidden_size
        m = config.intermediate_size
        n = config.state_size
        rank = config.time_step_rank
        add(
            "backbone.embeddings.weight",
            "token_embedding.weight",
            (config.vocab_size, h),
            "token-embedding",
        )
        add("backbone.norm_f.weight", "final_norm.weight", (h,), "final-rms-norm")
        if not config.tie_word_embeddings:
            add("lm_head.weight", "lm_head.weight", (config.vocab_size, h), "output-readout")
        for layer in range(config.num_hidden_layers):
            source_prefix = f"backbone.layers.{layer}"
            logical_prefix = f"layers.{layer}"
            mixer_source = f"{source_prefix}.mixer"
            mixer_logical = f"{logical_prefix}.mixer"
            add(
                f"{source_prefix}.norm.weight",
                f"{logical_prefix}.norm.weight",
                (h,),
                "pre-mixer-rms-norm",
            )
            add(
                f"{mixer_source}.in_proj.weight",
                f"{mixer_logical}.in_proj.weight",
                (2 * m, h),
                "mamba-input-gate-projection",
            )
            if config.use_bias:
                add(
                    f"{mixer_source}.in_proj.bias",
                    f"{mixer_logical}.in_proj.bias",
                    (2 * m,),
                    "mamba-input-gate-projection-bias",
                )
            add(
                f"{mixer_source}.conv1d.weight",
                f"{mixer_logical}.conv1d.weight",
                (m, 1, config.conv_kernel),
                "mamba-depthwise-causal-convolution",
            )
            if config.use_conv_bias:
                add(
                    f"{mixer_source}.conv1d.bias",
                    f"{mixer_logical}.conv1d.bias",
                    (m,),
                    "mamba-depthwise-causal-convolution-bias",
                )
            add(
                f"{mixer_source}.x_proj.weight",
                f"{mixer_logical}.x_proj.weight",
                (rank + 2 * n, m),
                "mamba-input-dependent-ssm-projection",
            )
            add(
                f"{mixer_source}.dt_proj.weight",
                f"{mixer_logical}.dt_proj.weight",
                (m, rank),
                "mamba-time-step-projection",
            )
            add(
                f"{mixer_source}.dt_proj.bias",
                f"{mixer_logical}.dt_proj.bias",
                (m,),
                "mamba-time-step-projection-bias",
            )
            add(
                f"{mixer_source}.A_log",
                f"{mixer_logical}.A_log",
                (m, n),
                "mamba-continuous-state-log",
            )
            add(
                f"{mixer_source}.D",
                f"{mixer_logical}.D",
                (m,),
                "mamba-direct-skip",
            )
            add(
                f"{mixer_source}.out_proj.weight",
                f"{mixer_logical}.out_proj.weight",
                (h, m),
                "mamba-output-projection",
            )
            if config.use_bias:
                add(
                    f"{mixer_source}.out_proj.bias",
                    f"{mixer_logical}.out_proj.bias",
                    (h,),
                    "mamba-output-projection-bias",
                )
        return specs

    def _map_weights(
        self, source: FrozenSourceBundle, index: TensorIndex, config: _MambaConfig
    ) -> tuple[PhysicalWeightIR, _MappedContext]:
        records = index.by_name()
        specs = self._expected_specs(config)
        if config.tie_word_embeddings and "lm_head.weight" in records:
            raise AliasEvidenceError(
                "tied Mamba config serializes a separate lm_head allocation",
                details={"source_name": "lm_head.weight"},
            )
        if not config.tie_word_embeddings and "lm_head.weight" not in records:
            raise AliasEvidenceError(
                "untied Mamba config is missing lm_head.weight",
                details={"tie_word_embeddings": False},
            )
        missing = sorted(set(specs) - set(records))
        if missing:
            raise CoverageError(
                "required Mamba source tensors are absent",
                details={"missing_tensors": missing},
            )
        unexplained = sorted(set(records) - set(specs))
        if unexplained:
            raise CoverageError(
                "Mamba adapter cannot classify every source tensor",
                details={"unexplained_tensors": unexplained},
            )
        non_float = sorted(
            record.source_name
            for record in records.values()
            if record.storage_dtype not in _RAW_FLOAT_DTYPES
        )
        if non_float:
            raise CodecError(
                "raw-float Mamba adapter encountered non-float storage",
                details={"tensors": non_float},
            )

        allocations: list[PhysicalAllocationIR] = []
        views: list[TensorViewIR] = []
        classifications: list[TensorClassificationIR] = []
        allocation_by_source: dict[str, str] = {}
        for source_name in sorted(records):
            record = records[source_name]
            spec = specs[source_name]
            lexical_shape_valid = (
                source_name in {"backbone.embeddings.weight", "lm_head.weight"}
                and len(record.shape) == 2
                and record.shape[0] >= config.vocab_size
                and record.shape[1] == config.hidden_size
            )
            if not lexical_shape_valid and record.shape != spec.shape:
                raise ConfigurationError(
                    f"Mamba tensor shape mismatch for {source_name}",
                    details={"expected": list(spec.shape), "actual": list(record.shape)},
                )
            allocation_id = _allocation_id(source_name, record.range_identity)
            allocation_by_source[source_name] = allocation_id
            allocations.append(
                PhysicalAllocationIR(
                    allocation_id=allocation_id,
                    source_tensor=source_name,
                    source_file=record.source_file,
                    byte_offset=record.byte_offset,
                    byte_length=record.byte_length,
                    stored_shape=record.shape,
                    stored_dtype=record.storage_dtype,
                    codec=CodecBindingIR(
                        codec_id="raw-float",
                        codec_version="1.0.0",
                        stored_dtype=record.storage_dtype,
                        _parameters_json=canonical_json(
                            {
                                "byte_order": "little",
                                "packing": "none",
                                "value_semantics": "safetensors-native-float",
                            }
                        ),
                    ),
                    content_fingerprint=record.range_identity,
                )
            )
            logical_names = [spec.logical_name]
            if source_name == "backbone.embeddings.weight" and config.tie_word_embeddings:
                logical_names.append("lm_head.weight")
            logical_view_ids: list[str] = []
            for logical_name in sorted(logical_names):
                view_id = _view_id(logical_name)
                logical_view_ids.append(view_id)
                views.append(
                    TensorViewIR(
                        view_id=view_id,
                        logical_name=logical_name,
                        allocation_id=allocation_id,
                        logical_shape=record.shape,
                        transforms=(
                            ViewTransformIR(kind="identity", _parameters_json=canonical_json({})),
                        ),
                        semantic_role=(
                            "output-readout"
                            if logical_name == "lm_head.weight"
                            else spec.semantic_role
                        ),
                        parameter_kind=spec.parameter_kind,
                    )
                )
            classifications.append(
                TensorClassificationIR(
                    source_name=source_name,
                    allocation_id=allocation_id,
                    disposition=spec.parameter_kind,
                    logical_view_ids=tuple(sorted(logical_view_ids)),
                    rule_id=spec.rule_id,
                    reason="exact adapter-owned Mamba source tensor schema",
                )
            )

        aliases: tuple[AliasClassIR, ...] = ()
        if config.tie_word_embeddings:
            allocation_id = allocation_by_source["backbone.embeddings.weight"]
            aliases = (
                AliasClassIR(
                    class_id=f"alias.{canonical_sha256({'allocation': allocation_id})[:24]}",
                    allocation_id=allocation_id,
                    logical_names=("lm_head.weight", "token_embedding.weight"),
                    evidence=AliasEvidenceIR(
                        kind="missing-serialized-tied-readout",
                        config_fields=("tie_word_embeddings",),
                        adapter_rule=f"{self.adapter_id}:declared-lexical-tie-v1",
                        certification_status="provisional-u2",
                        required_followup=(
                            "reference-forward-parity",
                            "reference-parameter-identity",
                        ),
                    ),
                ),
            )
        physical = PhysicalWeightIR.build(
            source_fingerprint=source.fingerprint,
            tensor_index_fingerprint=index.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            allocations=tuple(sorted(allocations, key=lambda item: item.allocation_id)),
            views=tuple(sorted(views, key=lambda item: item.logical_name)),
            alias_classes=aliases,
            classifications=tuple(sorted(classifications, key=lambda item: item.source_name)),
        )
        input_rows = records["backbone.embeddings.weight"].shape[0]
        output_rows = (
            input_rows if config.tie_word_embeddings else records["lm_head.weight"].shape[0]
        )
        if input_rows < config.vocab_size or output_rows < config.vocab_size:
            raise IOContractError(
                "configured Mamba token space exceeds lexical row capacity",
                details={
                    "vocab_size": config.vocab_size,
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                },
            )
        return physical, _MappedContext(
            config=config,
            input_rows=input_rows,
            output_rows=output_rows,
            rotary_parameters_by_layer=tuple(() for _ in range(config.num_hidden_layers)),
        )

    def _state_slot_ids(self, config: _MambaConfig) -> tuple[str, ...]:
        values = ["sequence_length"]
        for layer in range(config.num_hidden_layers):
            values.extend(
                (
                    f"layers.{layer}.conv_state",
                    f"layers.{layer}.recurrent_state",
                )
            )
        return tuple(sorted(values))

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        if not isinstance(config, _MambaConfig):
            raise TypeError("Mamba model builder received a foreign config")
        parameters = tuple(
            sorted(
                (
                    LogicalParameterRefIR(
                        logical_name=view.logical_name,
                        view_id=view.view_id,
                        semantic_role=view.semantic_role,
                        parameter_kind=view.parameter_kind,
                    )
                    for view in physical.views
                ),
                key=lambda item: item.logical_name,
            )
        )
        operations: list[OperationIR] = [
            _operation(
                "embedding",
                "token-embedding",
                ("token_ids",),
                ("embedding.hidden",),
                ("token_embedding.weight",),
                {"padding_policy": "IOIR-row-mapper"},
            )
        ]
        current = "embedding.hidden"
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}"
            mixer = f"{prefix}.mixer"
            operations.append(
                _operation(
                    f"{prefix}.norm",
                    "mamba-rms-norm",
                    (current,),
                    (f"{prefix}.norm.hidden",),
                    (f"{prefix}.norm.weight",),
                    {"epsilon": config.layer_norm_epsilon, "accumulation": "float32"},
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.in_proj",
                    "linear",
                    (f"{prefix}.norm.hidden",),
                    (f"{mixer}.packed_xz",),
                    self._linear_parameters(mixer, "in_proj", bias=config.use_bias),
                    {"weight_orientation": "out-in"},
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.split_xz",
                    "mamba-input-gate-split",
                    (f"{mixer}.packed_xz",),
                    (f"{mixer}.x", f"{mixer}.gate"),
                    attributes={"split_width": config.intermediate_size},
                )
            )
            conv_parameters = [f"{mixer}.conv1d.weight"]
            if config.use_conv_bias:
                conv_parameters.append(f"{mixer}.conv1d.bias")
            operations.append(
                _operation(
                    f"{mixer}.causal_conv",
                    "mamba-causal-depthwise-convolution",
                    (f"{mixer}.x",),
                    (f"{mixer}.convolved",),
                    tuple(sorted(conv_parameters)),
                    {
                        "activation": "silu",
                        "kernel_size": config.conv_kernel,
                        "state_slot": f"{prefix}.conv_state",
                    },
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.x_proj",
                    "linear",
                    (f"{mixer}.convolved",),
                    (f"{mixer}.packed_selection",),
                    (f"{mixer}.x_proj.weight",),
                    {"weight_orientation": "out-in"},
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.split_selection",
                    "mamba-selection-split",
                    (f"{mixer}.packed_selection",),
                    (f"{mixer}.dt_input", f"{mixer}.B", f"{mixer}.C"),
                    attributes={
                        "state_size": config.state_size,
                        "time_step_rank": config.time_step_rank,
                    },
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.dt_proj",
                    "linear",
                    (f"{mixer}.dt_input",),
                    (f"{mixer}.dt_pre_softplus",),
                    self._linear_parameters(mixer, "dt_proj", bias=True),
                    {"weight_orientation": "out-in"},
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.selective_scan",
                    "mamba-selective-scan",
                    (
                        f"{mixer}.convolved",
                        f"{mixer}.dt_pre_softplus",
                        f"{mixer}.B",
                        f"{mixer}.C",
                    ),
                    (f"{mixer}.scan_output",),
                    (f"{mixer}.A_log", f"{mixer}.D"),
                    {
                        "dt_activation": "softplus",
                        "state_size": config.state_size,
                        "state_slot": f"{prefix}.recurrent_state",
                        "state_update": "prefix-indexed-provisional",
                    },
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.gate_activation",
                    "silu",
                    (f"{mixer}.gate",),
                    (f"{mixer}.activated_gate",),
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.gate_scan",
                    "elementwise-multiply",
                    (f"{mixer}.scan_output", f"{mixer}.activated_gate"),
                    (f"{mixer}.gated_scan",),
                )
            )
            operations.append(
                _operation(
                    f"{mixer}.out_proj",
                    "linear",
                    (f"{mixer}.gated_scan",),
                    (f"{mixer}.output",),
                    self._linear_parameters(mixer, "out_proj", bias=config.use_bias),
                    {"weight_orientation": "out-in"},
                )
            )
            next_hidden = f"{prefix}.output"
            operations.append(
                _operation(
                    f"{prefix}.residual",
                    "residual-add",
                    (current, f"{mixer}.output"),
                    (next_hidden,),
                    attributes={
                        "residual_accumulation": (
                            "float32" if config.residual_in_fp32 else "activation-dtype"
                        )
                    },
                )
            )
            current = next_hidden
        operations.extend(
            (
                _operation(
                    "final_norm",
                    "mamba-rms-norm",
                    (current,),
                    ("final.hidden",),
                    ("final_norm.weight",),
                    {"epsilon": config.layer_norm_epsilon, "accumulation": "float32"},
                ),
                _operation(
                    "lm_head",
                    "linear-readout",
                    ("final.hidden",),
                    ("logits",),
                    ("lm_head.weight",),
                    {"weight_orientation": "rows-hidden"},
                ),
            )
        )
        return ModelIR.build(
            source_fingerprint=source.fingerprint,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            architecture_id=self.architecture_id,
            physical_weights_fingerprint=physical.fingerprint,
            dimensions=ModelDimensionsIR(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=0,
                num_key_value_heads=0,
                head_dim=0,
                vocab_size=config.vocab_size,
                physical_vocab_rows=context.input_rows,
                max_position_embeddings=0,
            ),
            parameters=parameters,
            operations=tuple(operations),
            state_refs=self._state_slot_ids(config),
            input_ports=(
                PortIR(
                    name="token_ids",
                    semantic="model-token-ids",
                    dtype="int64",
                    shape=("batch", "sequence"),
                    value="token_ids",
                ),
            ),
            output_ports=(
                PortIR(
                    name="hidden_states",
                    semantic="final-normalized-hidden",
                    dtype="activation",
                    shape=("batch", "sequence", "hidden_size"),
                    value="final.hidden",
                ),
                PortIR(
                    name="logits",
                    semantic="token-row-logits",
                    dtype="activation",
                    shape=("batch", "sequence", "output_rows"),
                    value="logits",
                ),
            ),
            numerical_semantics=NumericalSemanticsIR(
                reference_contract="source-reference-required-before-u3",
                accumulation=(
                    "float32-residual-and-ssm"
                    if config.residual_in_fp32
                    else "ssm-float32-residual-activation-dtype"
                ),
                softmax="not-applicable-no-attention",
                positional_arithmetic="none-causal-recurrence-order",
                optimization_contract="preregister-before-target-execution",
            ),
        )

    def _build_state(self, source: FrozenSourceBundle, model: ModelIR) -> StateIR:
        config = self._parse_config(source.config)
        slots: list[StateSlotIR] = [
            StateSlotIR(
                slot_id="sequence_length",
                kind="committed-sequence-length",
                dtype="int64",
                shape_expression=("batch",),
                ownership="request",
                lease_behavior="exclusive-epoch-bound",
                provisional_representation="accepted-prefix-length-delta",
                commit_rule="atomic-select-accepted-prefix",
                rollback_rule="discard-provisional",
                memory_charge_expression="batch * sizeof(int64)",
            )
        ]
        for layer in range(config.num_hidden_layers):
            slots.extend(
                (
                    StateSlotIR(
                        slot_id=f"layers.{layer}.conv_state",
                        kind="mamba-causal-convolution-window",
                        dtype="activation",
                        shape_expression=(
                            "batch",
                            "intermediate_size",
                            "conv_kernel",
                        ),
                        ownership="request",
                        lease_behavior="exclusive-epoch-bound",
                        provisional_representation="prefix-indexed-state-snapshots",
                        commit_rule="atomic-select-accepted-prefix-snapshot",
                        rollback_rule="discard-provisional-snapshots",
                        memory_charge_expression=(
                            "batch * intermediate_size * conv_kernel * dtype_bytes"
                        ),
                    ),
                    StateSlotIR(
                        slot_id=f"layers.{layer}.recurrent_state",
                        kind="mamba-selective-scan-state",
                        dtype="activation",
                        shape_expression=(
                            "batch",
                            "intermediate_size",
                            "state_size",
                        ),
                        ownership="request",
                        lease_behavior="exclusive-epoch-bound",
                        provisional_representation="prefix-indexed-state-snapshots",
                        commit_rule="atomic-select-accepted-prefix-snapshot",
                        rollback_rule="discard-provisional-snapshots",
                        memory_charge_expression=(
                            "batch * intermediate_size * state_size * dtype_bytes"
                        ),
                    ),
                )
            )
        slots_tuple = tuple(sorted(slots, key=lambda item: item.slot_id))
        slot_ids = tuple(item.slot_id for item in slots_tuple)
        operation_attributes = canonical_json(
            {
                "accepted_prefix": "select-snapshot-at-prefix-boundary",
                "committed_mutation_before_commit": False,
                "conv_kernel": config.conv_kernel,
                "state_size": config.state_size,
            }
        )
        return StateIR.build(
            source_fingerprint=source.fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            model_fingerprint=model.fingerprint,
            slots=slots_tuple,
            initialization=(
                StateOperationIR(
                    operation_id="initialize-mamba-request-state",
                    kind="zero-recurrent-state-and-bind-epoch",
                    slots=slot_ids,
                    _attributes_json=canonical_json({"epoch": 0, "sequence_length": 0}),
                ),
            ),
            prefill_updates=(
                StateOperationIR(
                    operation_id="prefill-mamba-provisional-scan",
                    kind="scan-prefix-to-provisional-state-snapshots",
                    slots=slot_ids,
                    _attributes_json=operation_attributes,
                ),
            ),
            decode_updates=(
                StateOperationIR(
                    operation_id="decode-mamba-provisional-scan",
                    kind="scan-decode-block-to-provisional-state-snapshots",
                    slots=slot_ids,
                    _attributes_json=operation_attributes,
                ),
            ),
            commit_protocol=CommitProtocolIR(
                protocol_id="mrun-transactional-recurrent-state-v1",
                authority="cache-issued-exclusive-lease",
                atomicity="all-convolution-recurrence-and-length-slots",
                accepted_prefix_rule="select-exact-prefix-snapshot-zero-through-provisional-length",
                rollback="discard-snapshots-without-committed-mutation",
                stale_epoch_rule="reject-before-write",
            ),
            capacity_equations=(
                CapacityEquationIR(
                    quantity="committed_conv_state_bytes",
                    expression=(
                        "batch * num_hidden_layers * intermediate_size * conv_kernel * dtype_bytes"
                    ),
                    units="bytes",
                ),
                CapacityEquationIR(
                    quantity="committed_recurrent_state_bytes",
                    expression=(
                        "batch * num_hidden_layers * intermediate_size * state_size * dtype_bytes"
                    ),
                    units="bytes",
                ),
                CapacityEquationIR(
                    quantity="length_bytes",
                    expression="batch * sizeof(int64)",
                    units="bytes",
                ),
                CapacityEquationIR(
                    quantity="provisional_snapshot_bytes",
                    expression=(
                        "provisional_tokens * batch * num_hidden_layers * intermediate_size * "
                        "(conv_kernel + state_size) * dtype_bytes"
                    ),
                    units="bytes",
                ),
            ),
        )


class Qwen3Adapter(DenseRotaryAdapter):
    adapter_id = "mrun.hf.qwen3-dense"
    model_type = "qwen3"
    architecture_id = "qwen3-dense-causal-decoder"
    architecture_names = frozenset({"Qwen3ForCausalLM", "Qwen3Model"})
    requires_qk_norm = True


class LlamaAdapter(DenseRotaryAdapter):
    adapter_id = "mrun.hf.llama-dense-baseline"
    model_type = "llama"
    architecture_id = "llama-dense-causal-decoder"
    architecture_names = frozenset({"LlamaForCausalLM", "LlamaModel"})
    supports_mlp_bias = True
    extra_config_keys = frozenset({"is_llama_config", "rope_interleaved"})

    def _parse_config(self, config: dict[str, Any]) -> _DenseConfig:
        if config.get("is_llama_config", True) is not True:
            raise ConfigurationError("is_llama_config must be true when present")
        if config.get("rope_interleaved", False) is not False:
            raise ConfigurationError(
                "interleaved Llama rotary coordinates require a distinct semantic adapter"
            )
        return super()._parse_config(config)


def default_adapter_registry() -> AdapterRegistry:
    from .qwen35_adapter import Qwen35Adapter

    return AdapterRegistry(
        (
            GPT2Adapter(),
            GPTNeoXAdapter(),
            LlamaAdapter(),
            GemmaAdapter(),
            MambaAdapter(),
            MistralAdapter(),
            MixtralAdapter(),
            PhiAdapter(),
            Qwen2Adapter(),
            Qwen3Adapter(),
            Qwen35Adapter(),
        )
    )
