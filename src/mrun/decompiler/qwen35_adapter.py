"""Content-bound text-runtime adapter for hybrid Qwen3.5 checkpoints.

Qwen3.5 checkpoints are multimodal containers.  The executable causal decoder lives under
``model.language_model`` and alternates gated-delta recurrent blocks with full-attention blocks;
the vision tower and the optional MTP predictor are deliberately outside this adapter's declared
text execution scope.  Their allocations are still inventoried and explicitly classified, so the
scope boundary cannot become an accidental tensor-coverage hole.

The adapter implementation hashes this file into its identity.  ``adapter_plugin`` also exposes
the exact factory shape expected by the decompiler's content-pinned plugin boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import canonical_json, canonical_sha256
from .adapters import (
    _RAW_FLOAT_DTYPES,
    DenseRotaryAdapter,
    _allocation_id,
    _DenseConfig,
    _MappedContext,
    _operation,
    _positive_float,
    _positive_int,
    _TensorSpec,
    _view_id,
)
from .errors import (
    AliasEvidenceError,
    CodecError,
    ConfigurationError,
    CoverageError,
    IOContractError,
)
from .ir import (
    AliasClassIR,
    AliasEvidenceIR,
    CapacityEquationIR,
    CodecBindingIR,
    CommitProtocolIR,
    LogicalParameterRefIR,
    ModelDimensionsIR,
    ModelIR,
    NumericalSemanticsIR,
    PhysicalAllocationIR,
    PhysicalWeightIR,
    PortIR,
    StateIR,
    StateOperationIR,
    StateSlotIR,
    TensorClassificationIR,
    TensorViewIR,
    ViewTransformIR,
)
from .matching import MatchEvidence, MatchResult, UnsupportedFeature
from .source import FrozenSourceBundle
from .tensor_index import TensorIndex

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_OUTER_KEYS = frozenset(
    {
        "_name_or_path",
        "architectures",
        "image_token_id",
        "model_type",
        "text_config",
        "tie_word_embeddings",
        "transformers_version",
        "video_token_id",
        "vision_config",
        "vision_end_token_id",
        "vision_start_token_id",
    }
)
_TEXT_KEYS = frozenset(
    {
        "_name_or_path",
        "attention_bias",
        "attention_dropout",
        "attn_output_gate",
        "dtype",
        "eos_token_id",
        "full_attention_interval",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mamba_ssm_dtype",
        "max_position_embeddings",
        "mlp_only_layers",
        "model_type",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_parameters",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
    }
)


@dataclass(frozen=True, slots=True)
class _Qwen35Config(_DenseConfig):
    layer_types: tuple[str, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int]
    mrope_interleaved: bool

    @property
    def linear_key_width(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_width(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_width(self) -> int:
        return 2 * self.linear_key_width + self.linear_value_width

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config")
    if not isinstance(value, dict):
        raise ConfigurationError("Qwen3.5 text_config must be an object")
    return value


def _bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ConfigurationError(f"Qwen3.5 text_config field {key!r} must be boolean")
    return value


class Qwen35Adapter(DenseRotaryAdapter):
    """Exact Qwen3.5 text-scope adapter with explicit hybrid state."""

    adapter_id = "mrun.hf.qwen3_5-hybrid-text"
    adapter_version = "1.0.0"
    model_type = "qwen3_5"
    architecture_id = "qwen3_5-hybrid-text-causal-decoder"
    architecture_names = frozenset({"Qwen3_5ForConditionalGeneration"})
    rule_set = "qwen3_5-nested-hybrid-text-v1"

    def __init__(self) -> None:
        self.adapter_fingerprint = canonical_sha256(
            {
                "adapter_id": self.adapter_id,
                "adapter_version": self.adapter_version,
                "architecture_id": self.architecture_id,
                "architecture_names": sorted(self.architecture_names),
                "implementation_sha256": _IMPLEMENTATION_SHA256,
                "execution_scope": "model.language_model text-only",
                "rule_set": self.rule_set,
            }
        )

    def _parse_config(self, config: dict[str, Any]) -> _Qwen35Config:
        text = _text_config(config)
        if text.get("model_type") != "qwen3_5_text":
            raise ConfigurationError("nested text_config.model_type must be 'qwen3_5_text'")
        hidden_size = _positive_int(text, "hidden_size")
        intermediate_size = _positive_int(text, "intermediate_size")
        layers = _positive_int(text, "num_hidden_layers")
        heads = _positive_int(text, "num_attention_heads")
        kv_heads = _positive_int(text, "num_key_value_heads")
        head_dim = _positive_int(text, "head_dim")
        if heads % kv_heads:
            raise ConfigurationError("Qwen3.5 attention heads must be divisible by KV heads")
        raw_layer_types = text.get("layer_types")
        if not isinstance(raw_layer_types, list) or len(raw_layer_types) != layers:
            raise ConfigurationError("Qwen3.5 layer_types must cover every decoder layer")
        layer_types = tuple(raw_layer_types)
        if any(value not in {"linear_attention", "full_attention"} for value in layer_types):
            raise ConfigurationError("Qwen3.5 layer_types contains an unknown mixer")
        if "linear_attention" not in layer_types or "full_attention" not in layer_types:
            raise ConfigurationError(
                "hybrid Qwen3.5 requires both recurrent and full-attention layers"
            )
        interval = _positive_int(text, "full_attention_interval")
        expected_full = tuple(index for index in range(layers) if (index + 1) % interval == 0)
        observed_full = tuple(
            index for index, kind in enumerate(layer_types) if kind == "full_attention"
        )
        if observed_full != expected_full:
            raise ConfigurationError("layer_types disagrees with full_attention_interval")
        if text.get("hidden_act", "silu") != "silu":
            raise ConfigurationError("Qwen3.5 adapter supports only SiLU/SwiGLU")
        if _bool(text, "attention_bias", False):
            raise ConfigurationError("Qwen3.5 attention bias requires a distinct adapter")
        if text.get("attention_dropout", 0.0) != 0.0:
            raise ConfigurationError("Qwen3.5 reference execution requires zero attention dropout")
        if not _bool(text, "attn_output_gate", True):
            raise ConfigurationError("Qwen3.5 full-attention output gate must be enabled")
        if text.get("mamba_ssm_dtype", "float32") != "float32":
            raise ConfigurationError("Qwen3.5 recurrent state must use float32")
        if text.get("mlp_only_layers", []) not in ([], ()):  # no mixer-free variant here
            raise ConfigurationError("Qwen3.5 MLP-only layers require a distinct adapter")
        key_heads = _positive_int(text, "linear_num_key_heads")
        value_heads = _positive_int(text, "linear_num_value_heads")
        if value_heads % key_heads:
            raise ConfigurationError("linear value heads must be divisible by linear key heads")
        key_dim = _positive_int(text, "linear_key_head_dim")
        value_dim = _positive_int(text, "linear_value_head_dim")
        conv_kernel = _positive_int(text, "linear_conv_kernel_dim")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict) or rope.get("rope_type") != "default":
            raise ConfigurationError("Qwen3.5 adapter supports only default partial mRoPE")
        allowed_rope = {
            "mrope_interleaved",
            "mrope_section",
            "partial_rotary_factor",
            "rope_theta",
            "rope_type",
        }
        if set(rope) - allowed_rope:
            raise ConfigurationError("Qwen3.5 rope_parameters contains unknown semantics")
        partial = _positive_float(rope, "partial_rotary_factor", 1.0)
        if partial > 1.0:
            raise ConfigurationError("partial_rotary_factor cannot exceed one")
        rotary_dim = int(head_dim * partial)
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ConfigurationError("Qwen3.5 partial rotary dimension must be positive and even")
        raw_section = rope.get("mrope_section")
        if (
            not isinstance(raw_section, list)
            or len(raw_section) != 3
            or any(type(value) is not int or value <= 0 for value in raw_section)
        ):
            raise ConfigurationError("mrope_section must contain three positive integers")
        section = tuple(raw_section)
        if sum(section) != rotary_dim // 2:
            raise ConfigurationError("mrope_section does not cover the partial rotary frequencies")
        mrope_interleaved = _bool(rope, "mrope_interleaved", False)
        if not mrope_interleaved:
            raise ConfigurationError("Qwen3.5 requires interleaved mRoPE coordinates")
        outer_tied = config.get("tie_word_embeddings", True)
        nested_tied = text.get("tie_word_embeddings", outer_tied)
        if (
            type(outer_tied) is not bool
            or type(nested_tied) is not bool
            or outer_tied != nested_tied
        ):
            raise ConfigurationError("outer and nested lexical tie declarations must agree")
        return _Qwen35Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=_positive_int(text, "vocab_size"),
            max_position_embeddings=_positive_int(text, "max_position_embeddings"),
            rms_norm_eps=_positive_float(text, "rms_norm_eps", 1e-6),
            rope_theta=_positive_float(rope, "rope_theta", 10_000.0),
            tie_word_embeddings=outer_tied,
            attention_bias=False,
            mlp_bias=False,
            layer_types=layer_types,
            linear_num_key_heads=key_heads,
            linear_num_value_heads=value_heads,
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
            linear_conv_kernel_dim=conv_kernel,
            partial_rotary_factor=partial,
            mrope_section=section,  # type: ignore[arg-type]
            mrope_interleaved=mrope_interleaved,
        )

    def _unsupported_features(
        self, source: FrozenSourceBundle, index: TensorIndex
    ) -> tuple[UnsupportedFeature, ...]:
        if source.config.get("model_type") != self.model_type:
            return ()
        unsupported: list[UnsupportedFeature] = []
        architectures = source.config.get("architectures")
        if architectures != ["Qwen3_5ForConditionalGeneration"]:
            unsupported.append(
                UnsupportedFeature(
                    code="architecture_declaration",
                    field="architectures",
                    observed=canonical_json(architectures),
                    reason="only the exact Qwen3.5 conditional-generation container is owned",
                )
            )
        outer_unknown = sorted(set(source.config) - _OUTER_KEYS)
        try:
            text = _text_config(source.config)
            text_unknown = sorted(set(text) - _TEXT_KEYS)
        except ConfigurationError:
            text = {}
            text_unknown = []
        if outer_unknown or text_unknown:
            unsupported.append(
                UnsupportedFeature(
                    code="unknown_config_keys",
                    field="config",
                    observed=",".join(
                        [
                            *(f"outer.{key}" for key in outer_unknown),
                            *(f"text.{key}" for key in text_unknown),
                        ]
                    ),
                    reason="unknown configuration fields may change hybrid execution semantics",
                )
            )
        if not isinstance(source.config.get("vision_config"), dict):
            unsupported.append(
                UnsupportedFeature(
                    code="vision_scope_declaration",
                    field="vision_config",
                    observed=repr(source.config.get("vision_config")),
                    reason="the ignored vision scope must still have an explicit configuration",
                )
            )
        non_float = sorted(
            {
                record.storage_dtype
                for record in index.tensors
                if record.storage_dtype not in _RAW_FLOAT_DTYPES
            }
        )
        if non_float:
            unsupported.append(
                UnsupportedFeature(
                    code="unregistered_source_codec",
                    field="safetensors.dtype",
                    observed=",".join(non_float),
                    reason="Qwen3.5 text adapter currently owns raw float checkpoints only",
                )
            )
        try:
            self._parse_config(source.config)
        except ConfigurationError as exc:
            unsupported.append(
                UnsupportedFeature(
                    code="invalid_semantic_config",
                    field="config.text_config",
                    observed=exc.message,
                    reason="hybrid text configuration is outside the exact adapter contract",
                )
            )
        return tuple(sorted(unsupported, key=lambda item: (item.code, item.field)))

    def match(self, source: FrozenSourceBundle, index: TensorIndex) -> MatchResult:
        if index.source_fingerprint != source.fingerprint:
            raise ValueError("tensor index is not bound to the supplied frozen source")
        names = index.by_name()
        matched = source.config.get("model_type") == self.model_type
        evidence = tuple(
            sorted(
                (
                    MatchEvidence(
                        predicate="config.model_type",
                        expected="qwen3_5",
                        observed=repr(source.config.get("model_type")),
                        matched=matched,
                    ),
                    MatchEvidence(
                        predicate="config.text_config.model_type",
                        expected="qwen3_5_text",
                        observed=repr((source.config.get("text_config") or {}).get("model_type")),
                        matched=(source.config.get("text_config") or {}).get("model_type")
                        == "qwen3_5_text",
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.embedding",
                        expected="model.language_model.embed_tokens.weight",
                        observed="present"
                        if "model.language_model.embed_tokens.weight" in names
                        else "absent",
                        matched="model.language_model.embed_tokens.weight" in names,
                    ),
                    MatchEvidence(
                        predicate="tensor.anchor.linear_layer0",
                        expected="model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                        observed=(
                            "present"
                            if "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
                            in names
                            else "absent"
                        ),
                        matched="model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
                        in names,
                    ),
                ),
                key=lambda item: item.predicate,
            )
        )
        unsupported = self._unsupported_features(source, index) if matched else ()
        return MatchResult.build(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            matched=matched,
            supported=matched and not unsupported,
            strength=100 if matched else 0,
            evidence=evidence,
            rejected_reasons=(() if matched else ("model_type is not 'qwen3_5'",)),
            required_tensor_patterns=(
                "model.language_model.<exact hybrid text schema>",
                "model.visual.* (classified outside text scope)",
                "mtp.* (classified outside text scope)",
            ),
            forbidden_tensor_patterns=("*.qweight", "*.qzeros", "unowned tensor names"),
            source_codec_candidates=("raw-float",),
            unsupported_features=unsupported,
        )

    def _expected_specs(self, config: _Qwen35Config) -> dict[str, _TensorSpec]:
        specs: dict[str, _TensorSpec] = {}

        def add(source_name: str, logical_name: str, shape: tuple[int, ...], role: str) -> None:
            specs[source_name] = _TensorSpec(
                source_name=source_name,
                logical_name=logical_name,
                shape=shape,
                semantic_role=role,
                rule_id=f"{self.adapter_id}:hybrid-text-v1",
            )

        h = config.hidden_size
        m = config.intermediate_size
        q = config.query_width
        kv = config.key_value_width
        root = "model.language_model"
        add(
            f"{root}.embed_tokens.weight",
            "token_embedding.weight",
            (config.vocab_size, h),
            "token-embedding",
        )
        add(f"{root}.norm.weight", "final_norm.weight", (h,), "final-qwen35-rms-norm")
        if not config.tie_word_embeddings:
            add("lm_head.weight", "lm_head.weight", (config.vocab_size, h), "output-readout")
        for layer, layer_type in enumerate(config.layer_types):
            source = f"{root}.layers.{layer}"
            logical = f"layers.{layer}"
            add(
                f"{source}.input_layernorm.weight",
                f"{logical}.mixer_norm.weight",
                (h,),
                "pre-mixer-qwen35-rms-norm",
            )
            add(
                f"{source}.post_attention_layernorm.weight",
                f"{logical}.mlp_norm.weight",
                (h,),
                "pre-mlp-qwen35-rms-norm",
            )
            if layer_type == "full_attention":
                attention = f"{logical}.attention"
                for name, shape in (
                    ("q_proj", (2 * q, h)),
                    ("k_proj", (kv, h)),
                    ("v_proj", (kv, h)),
                    ("o_proj", (h, q)),
                ):
                    add(
                        f"{source}.self_attn.{name}.weight",
                        f"{attention}.{name}.weight",
                        shape,
                        f"full-attention-{name}",
                    )
                add(
                    f"{source}.self_attn.q_norm.weight",
                    f"{attention}.q_norm.weight",
                    (config.head_dim,),
                    "query-qwen35-rms-norm",
                )
                add(
                    f"{source}.self_attn.k_norm.weight",
                    f"{attention}.k_norm.weight",
                    (config.head_dim,),
                    "key-qwen35-rms-norm",
                )
            else:
                mixer = f"{logical}.gated_delta"
                linear = f"{source}.linear_attn"
                for name, shape in (
                    ("in_proj_qkv", (config.linear_conv_width, h)),
                    ("in_proj_z", (config.linear_value_width, h)),
                    ("in_proj_a", (config.linear_num_value_heads, h)),
                    ("in_proj_b", (config.linear_num_value_heads, h)),
                    ("out_proj", (h, config.linear_value_width)),
                ):
                    add(
                        f"{linear}.{name}.weight",
                        f"{mixer}.{name}.weight",
                        shape,
                        f"gated-delta-{name}",
                    )
                add(
                    f"{linear}.conv1d.weight",
                    f"{mixer}.conv1d.weight",
                    (config.linear_conv_width, 1, config.linear_conv_kernel_dim),
                    "gated-delta-depthwise-convolution",
                )
                add(
                    f"{linear}.A_log",
                    f"{mixer}.A_log",
                    (config.linear_num_value_heads,),
                    "gated-delta-log-decay",
                )
                add(
                    f"{linear}.dt_bias",
                    f"{mixer}.dt_bias",
                    (config.linear_num_value_heads,),
                    "gated-delta-time-bias",
                )
                add(
                    f"{linear}.norm.weight",
                    f"{mixer}.norm.weight",
                    (config.linear_value_head_dim,),
                    "gated-delta-output-norm",
                )
            for name, shape in (
                ("gate_proj", (m, h)),
                ("up_proj", (m, h)),
                ("down_proj", (h, m)),
            ):
                add(
                    f"{source}.mlp.{name}.weight",
                    f"{logical}.mlp.{name}.weight",
                    shape,
                    f"mlp-{name}",
                )
        return specs

    def _map_weights(
        self, source: FrozenSourceBundle, index: TensorIndex, config: _Qwen35Config
    ) -> tuple[PhysicalWeightIR, _MappedContext]:
        records = index.by_name()
        specs = self._expected_specs(config)
        embedding_source = "model.language_model.embed_tokens.weight"
        ignored = {
            name: (
                f"{self.adapter_id}:declared-non-text-scope-v1",
                "allocation belongs to the declared vision or speculative-MTP scope, "
                "not the text decoder",
            )
            for name in records
            if name.startswith("model.visual.") or name.startswith("mtp.")
        }
        if config.tie_word_embeddings and "lm_head.weight" in records:
            raise AliasEvidenceError("tied Qwen3.5 config serializes a separate lm_head allocation")
        if not config.tie_word_embeddings and "lm_head.weight" not in records:
            raise AliasEvidenceError("untied Qwen3.5 config is missing lm_head.weight")
        missing = sorted(set(specs) - set(records))
        if missing:
            raise CoverageError(
                "required Qwen3.5 text tensors are absent", details={"missing_tensors": missing}
            )
        unexplained = sorted(set(records) - set(specs) - set(ignored))
        if unexplained:
            raise CoverageError(
                "Qwen3.5 adapter cannot classify every source tensor",
                details={"unexplained_tensors": unexplained},
            )
        non_float = sorted(
            name
            for name, record in records.items()
            if record.storage_dtype not in _RAW_FLOAT_DTYPES
        )
        if non_float:
            raise CodecError(
                "Qwen3.5 raw-float adapter encountered encoded tensors",
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
            lexical = source_name in {embedding_source, "lm_head.weight"}
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
            logical_names = [spec.logical_name]
            if source_name == embedding_source and config.tie_word_embeddings:
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
                        semantic_role="output-readout"
                        if logical_name == "lm_head.weight"
                        else spec.semantic_role,
                        parameter_kind="parameter",
                    )
                )
            classifications.append(
                TensorClassificationIR(
                    source_name=source_name,
                    allocation_id=allocation_id,
                    disposition="parameter",
                    logical_view_ids=tuple(sorted(view_ids)),
                    rule_id=spec.rule_id,
                    reason="exact adapter-owned Qwen3.5 text tensor schema",
                )
            )
        aliases: tuple[AliasClassIR, ...] = ()
        if config.tie_word_embeddings:
            allocation_id = allocation_by_source[embedding_source]
            aliases = (
                AliasClassIR(
                    class_id=f"alias.{canonical_sha256({'allocation': allocation_id})[:24]}",
                    allocation_id=allocation_id,
                    logical_names=("lm_head.weight", "token_embedding.weight"),
                    evidence=AliasEvidenceIR(
                        kind="nested-and-outer-declared-tied-readout-omission",
                        config_fields=("text_config.tie_word_embeddings", "tie_word_embeddings"),
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
        input_rows = records[embedding_source].shape[0]
        output_source = (
            records[embedding_source] if config.tie_word_embeddings else records["lm_head.weight"]
        )
        if input_rows < config.vocab_size or output_source.shape[0] < config.vocab_size:
            raise IOContractError("Qwen3.5 lexical rows are smaller than its token domain")
        return physical, _MappedContext(
            config=config,
            input_rows=input_rows,
            output_rows=output_source.shape[0],
            rotary_parameters_by_layer=tuple(() for _ in config.layer_types),
        )

    @staticmethod
    def _linear(prefix: str, name: str) -> tuple[str, ...]:
        return (f"{prefix}.{name}.weight",)

    def _state_slot_ids(self, config: _Qwen35Config) -> tuple[str, ...]:
        slots = ["position"]
        for layer, kind in enumerate(config.layer_types):
            if kind == "full_attention":
                slots.extend((f"layers.{layer}.k_cache", f"layers.{layer}.v_cache"))
            else:
                slots.extend((f"layers.{layer}.conv_state", f"layers.{layer}.recurrent_state"))
        return tuple(sorted(slots))

    def _build_model(
        self, source: FrozenSourceBundle, physical: PhysicalWeightIR, context: _MappedContext
    ) -> ModelIR:
        config = context.config
        if not isinstance(config, _Qwen35Config):
            raise TypeError("Qwen3.5 mapped context lost its hybrid configuration")
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
        operations = [
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
        for layer, layer_type in enumerate(config.layer_types):
            prefix = f"layers.{layer}"
            operations.append(
                _operation(
                    f"{prefix}.mixer_norm",
                    "qwen35-rms-norm",
                    (current,),
                    (f"{prefix}.mixer_norm.hidden",),
                    (f"{prefix}.mixer_norm.weight",),
                    {"epsilon": config.rms_norm_eps, "weight_center": 1.0},
                )
            )
            normed = f"{prefix}.mixer_norm.hidden"
            if layer_type == "linear_attention":
                mixer = f"{prefix}.gated_delta"
                for projection in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                    operations.append(
                        _operation(
                            f"{mixer}.{projection}",
                            "linear",
                            (normed,),
                            (f"{mixer}.{projection}.hidden",),
                            self._linear(mixer, projection),
                            {"weight_orientation": "out-in"},
                        )
                    )
                operations.append(
                    _operation(
                        f"{mixer}.causal_conv",
                        "qwen35-causal-depthwise-convolution",
                        (f"{mixer}.in_proj_qkv.hidden",),
                        (f"{mixer}.convolved",),
                        (f"{mixer}.conv1d.weight",),
                        {
                            "activation": "silu",
                            "kernel_size": config.linear_conv_kernel_dim,
                            "state_slot": f"layers.{layer}.conv_state",
                        },
                    )
                )
                operations.append(
                    _operation(
                        f"{mixer}.split_qkv",
                        "qwen35-linear-qkv-split",
                        (f"{mixer}.convolved",),
                        (f"{mixer}.query", f"{mixer}.key", f"{mixer}.value"),
                        attributes={
                            "key_width": config.linear_key_width,
                            "value_width": config.linear_value_width,
                        },
                    )
                )
                operations.append(
                    _operation(
                        f"{mixer}.recurrence",
                        "qwen35-gated-delta-recurrence",
                        (
                            f"{mixer}.query",
                            f"{mixer}.key",
                            f"{mixer}.value",
                            f"{mixer}.in_proj_a.hidden",
                            f"{mixer}.in_proj_b.hidden",
                        ),
                        (f"{mixer}.core",),
                        (f"{mixer}.A_log", f"{mixer}.dt_bias"),
                        {
                            "key_head_dim": config.linear_key_head_dim,
                            "num_key_heads": config.linear_num_key_heads,
                            "num_value_heads": config.linear_num_value_heads,
                            "state_slot": f"layers.{layer}.recurrent_state",
                            "value_head_dim": config.linear_value_head_dim,
                        },
                    )
                )
                operations.append(
                    _operation(
                        f"{mixer}.gated_norm",
                        "qwen35-rms-norm-gated",
                        (f"{mixer}.core", f"{mixer}.in_proj_z.hidden"),
                        (f"{mixer}.normalized",),
                        (f"{mixer}.norm.weight",),
                        {"epsilon": config.rms_norm_eps, "head_dim": config.linear_value_head_dim},
                    )
                )
                operations.append(
                    _operation(
                        f"{mixer}.out_proj",
                        "linear",
                        (f"{mixer}.normalized",),
                        (f"{mixer}.output",),
                        self._linear(mixer, "out_proj"),
                        {"weight_orientation": "out-in"},
                    )
                )
                mixer_output = f"{mixer}.output"
            else:
                attention = f"{prefix}.attention"
                operations.append(
                    _operation(
                        f"{attention}.q_proj",
                        "linear",
                        (normed,),
                        (f"{attention}.q_packed",),
                        self._linear(attention, "q_proj"),
                        {"weight_orientation": "out-in"},
                    )
                )
                operations.append(
                    _operation(
                        f"{attention}.split_query_gate",
                        "qwen35-query-gate-split",
                        (f"{attention}.q_packed",),
                        (f"{attention}.query", f"{attention}.gate"),
                        attributes={
                            "head_dim": config.head_dim,
                            "num_attention_heads": config.num_attention_heads,
                        },
                    )
                )
                for projection in ("k_proj", "v_proj"):
                    operations.append(
                        _operation(
                            f"{attention}.{projection}",
                            "linear",
                            (normed,),
                            (f"{attention}.{projection[0]}",),
                            self._linear(attention, projection),
                            {"weight_orientation": "out-in"},
                        )
                    )
                operations.append(
                    _operation(
                        f"{attention}.q_norm",
                        "qwen35-head-rms-norm",
                        (f"{attention}.query",),
                        (f"{attention}.q_normed",),
                        (f"{attention}.q_norm.weight",),
                        {
                            "epsilon": config.rms_norm_eps,
                            "head_dim": config.head_dim,
                            "weight_center": 1.0,
                        },
                    )
                )
                operations.append(
                    _operation(
                        f"{attention}.k_norm",
                        "qwen35-head-rms-norm",
                        (f"{attention}.k",),
                        (f"{attention}.k_normed",),
                        (f"{attention}.k_norm.weight",),
                        {
                            "epsilon": config.rms_norm_eps,
                            "head_dim": config.head_dim,
                            "weight_center": 1.0,
                        },
                    )
                )
                operations.append(
                    _operation(
                        f"{attention}.mrope",
                        "qwen35-partial-interleaved-mrope",
                        (f"{attention}.q_normed", f"{attention}.k_normed"),
                        (f"{attention}.q_rotary", f"{attention}.k_rotary"),
                        attributes={
                            "head_dim": config.head_dim,
                            "mrope_interleaved": True,
                            "mrope_section": list(config.mrope_section),
                            "rope_theta": config.rope_theta,
                            "rotary_dim": config.rotary_dim,
                        },
                    )
                )
                operations.append(
                    _operation(
                        f"{attention}.gqa",
                        "causal-grouped-query-attention",
                        (f"{attention}.q_rotary", f"{attention}.k_rotary", f"{attention}.v"),
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
                        f"{attention}.output_gate",
                        "qwen35-attention-output-gate",
                        (f"{attention}.context", f"{attention}.gate"),
                        (f"{attention}.gated_context",),
                    )
                )
                operations.append(
                    _operation(
                        f"{attention}.o_proj",
                        "linear",
                        (f"{attention}.gated_context",),
                        (f"{attention}.output",),
                        self._linear(attention, "o_proj"),
                        {"weight_orientation": "out-in"},
                    )
                )
                mixer_output = f"{attention}.output"
            operations.append(
                _operation(
                    f"{prefix}.mixer_residual",
                    "residual-add",
                    (current, mixer_output),
                    (f"{prefix}.mixer_residual.hidden",),
                )
            )
            residual = f"{prefix}.mixer_residual.hidden"
            operations.append(
                _operation(
                    f"{prefix}.mlp_norm",
                    "qwen35-rms-norm",
                    (residual,),
                    (f"{prefix}.mlp_norm.hidden",),
                    (f"{prefix}.mlp_norm.weight",),
                    {"epsilon": config.rms_norm_eps, "weight_center": 1.0},
                )
            )
            for projection in ("gate_proj", "up_proj"):
                operations.append(
                    _operation(
                        f"{prefix}.mlp.{projection}",
                        "linear",
                        (f"{prefix}.mlp_norm.hidden",),
                        (f"{prefix}.mlp.{projection}.hidden",),
                        self._linear(f"{prefix}.mlp", projection),
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
                    (f"{prefix}.mlp.gate_activated", f"{prefix}.mlp.up_proj.hidden"),
                    (f"{prefix}.mlp.intermediate",),
                )
            )
            operations.append(
                _operation(
                    f"{prefix}.mlp.down_proj",
                    "linear",
                    (f"{prefix}.mlp.intermediate",),
                    (f"{prefix}.mlp.output",),
                    self._linear(f"{prefix}.mlp", "down_proj"),
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
                    "qwen35-rms-norm",
                    (current,),
                    ("final.hidden",),
                    ("final_norm.weight",),
                    {"epsilon": config.rms_norm_eps, "weight_center": 1.0},
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
                accumulation="qwen35-fp32-norm-and-recurrence",
                softmax="stable-causal-softmax-fp32",
                positional_arithmetic="partial-interleaved-mrope-fp32",
                optimization_contract="fused-kernels-require-capability-and-parity-proof",
            ),
        )

    def _build_state(self, source: FrozenSourceBundle, model: ModelIR) -> StateIR:
        config = self._parse_config(source.config)
        slots = [
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
        for layer, kind in enumerate(config.layer_types):
            if kind == "full_attention":
                for cache in ("k", "v"):
                    slots.append(
                        StateSlotIR(
                            slot_id=f"layers.{layer}.{cache}_cache",
                            kind=f"paged-{cache}-cache",
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
            else:
                slots.append(
                    StateSlotIR(
                        slot_id=f"layers.{layer}.conv_state",
                        kind="causal-depthwise-convolution-state",
                        dtype="activation",
                        shape_expression=(
                            "batch",
                            str(config.linear_conv_width),
                            str(config.linear_conv_kernel_dim),
                        ),
                        ownership="request",
                        lease_behavior="exclusive-epoch-bound",
                        provisional_representation="copy-on-write-ring-delta",
                        commit_rule="atomic-accepted-prefix",
                        rollback_rule="discard-provisional",
                        memory_charge_expression=(
                            f"batch * {config.linear_conv_width} * "
                            f"{config.linear_conv_kernel_dim} * dtype_bytes"
                        ),
                    )
                )
                slots.append(
                    StateSlotIR(
                        slot_id=f"layers.{layer}.recurrent_state",
                        kind="gated-delta-recurrent-matrix",
                        dtype="float32",
                        shape_expression=(
                            "batch",
                            str(config.linear_num_value_heads),
                            str(config.linear_key_head_dim),
                            str(config.linear_value_head_dim),
                        ),
                        ownership="request",
                        lease_behavior="exclusive-epoch-bound",
                        provisional_representation="copy-on-write-matrix-delta",
                        commit_rule="atomic-accepted-prefix",
                        rollback_rule="discard-provisional",
                        memory_charge_expression=(
                            f"batch * {config.linear_num_value_heads} * "
                            f"{config.linear_key_head_dim} * "
                            f"{config.linear_value_head_dim} * sizeof(float32)"
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
                    operation_id="initialize-hybrid-request-state",
                    kind="zero-and-bind-hybrid-capacity",
                    slots=slot_ids,
                    _attributes_json=canonical_json({"epoch": 0, "length": 0}),
                ),
            ),
            prefill_updates=(
                StateOperationIR(
                    operation_id="prefill-hybrid-provisional",
                    kind="append-kv-and-stage-conv-recurrence",
                    slots=slot_ids,
                    _attributes_json=canonical_json({"mutation": "prohibited-before-commit"}),
                ),
            ),
            decode_updates=(
                StateOperationIR(
                    operation_id="decode-hybrid-provisional",
                    kind="append-token-kv-and-stage-conv-recurrence",
                    slots=slot_ids,
                    _attributes_json=canonical_json({"mutation": "prohibited-before-commit"}),
                ),
            ),
            commit_protocol=CommitProtocolIR(
                protocol_id="mrun-transactional-hybrid-state-v1",
                authority="cache-issued-exclusive-lease",
                atomicity="position-kv-conv-and-recurrence",
                accepted_prefix_rule="zero-through-provisional-length",
                rollback="discard-all-hybrid-deltas-without-committed-mutation",
                stale_epoch_rule="reject-before-read-or-write",
            ),
            capacity_equations=tuple(
                sorted(
                    (
                        CapacityEquationIR(
                            quantity="conv_state_bytes",
                            expression=(
                                "batch * sum_linear_layers(conv_width * conv_kernel) * dtype_bytes"
                            ),
                            units="bytes",
                        ),
                        CapacityEquationIR(
                            quantity="kv_bytes",
                            expression=(
                                "batch * capacity * 2 * num_full_attention_layers * "
                                "num_key_value_heads * head_dim * dtype_bytes"
                            ),
                            units="bytes",
                        ),
                        CapacityEquationIR(
                            quantity="position_bytes",
                            expression="batch * sizeof(int64)",
                            units="bytes",
                        ),
                        CapacityEquationIR(
                            quantity="provisional_copy_on_write_bytes",
                            expression=(
                                "accepted_suffix * (kv_delta_bytes + conv_delta_bytes + "
                                "recurrent_delta_bytes)"
                            ),
                            units="bytes",
                        ),
                        CapacityEquationIR(
                            quantity="recurrent_state_bytes",
                            expression=(
                                "batch * sum_linear_layers(num_value_heads * key_head_dim * "
                                "value_head_dim) * sizeof(float32)"
                            ),
                            units="bytes",
                        ),
                    ),
                    key=lambda item: item.quantity,
                )
            ),
        )

    def _special_tokens(self, config: dict[str, Any], vocab_size: int):  # type: ignore[no-untyped-def]
        return super()._special_tokens(_text_config(config), vocab_size)


def adapter_plugin() -> tuple[Qwen35Adapter, ...]:
    """Entry-point factory for an optionally installed, provenance-pinned plugin distribution."""

    return (Qwen35Adapter(),)


__all__ = ["Qwen35Adapter", "adapter_plugin"]
