"""qstore_build.py — stream a HF checkpoint into a block-indexed 8-bit disk store.

The whole engine claim is RAM-decoupling: you should be able to RUN a model of any size
without ever holding all its parameters resident. Step 1 is a store that lets the forward
pass demand-page one matrix at a time. This builds that store WITHOUT loading the whole
model into RAM — it streams safetensors one tensor at a time (RSS bounded by the largest
single matrix) and writes:

  <out>/manifest.json   model config + a block table {name -> shape, file offsets, kind}
  <out>/weights.i8      concatenated per-output-channel int8 weights (the big file)
  <out>/scales.f32      concatenated fp32 per-output-channel scales (one per output row)
  <out>/extras.f32      fp32 small tensors kept full-precision (norms, biases)

Quantization is per-OUTPUT-CHANNEL symmetric int8 (``W.abs().amax(dim=1)/127``). That
equality is load-bearing: it lets the paged forward prove EXACT mechanics parity against an
HF model with the same fake-quant applied (same dequantized weights => same logits).

Supported arch families:
  qwen2 / llama  — RoPE + RMSNorm + SwiGLU + GQA, optional qkv bias.
  gpt_neox       — LayerNorm(+bias) + packed QKV + partial rotary + parallel residual + GELU.
  mamba          — RMSNorm + in_proj/conv1d/selective-scan/gate/out_proj (SSM params kept fp32).
The write/output projection of every arch is stored as ``L{L}.down`` so the engine's
down_weight()/write_norm() stay arch-agnostic. gpt2 (Conv1D) is still a later extension.

Vendored from ram-decoupling/qstore_build.py: the external ``find_safetensors`` import was
replaced with ``model_experiments.models.find_safetensors`` (exact-hub match, no substring
trap) and the module-global output root with an injected ``out_root`` (``stores_root()``).
"""

from __future__ import annotations

import hashlib
import json
import re
import resource
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from ...models import find_safetensors, resolve_model, store_name
from ...paths import stores_root
from ...store_provenance import (
    build_builder_provenance,
    build_derived_provenance,
    build_source_provenance,
    verify_builder_provenance,
    verify_derived_provenance,
    verify_source_provenance,
)

QSTORE_SCHEMA = "mrun-qstore-int8-v3"
QSTORE_FILES = ("weights.i8", "scales.f32", "extras.f32")
QSTORE_QUANTIZATION = {
    "codec": "symmetric-int8",
    "granularity": "per-output-channel",
    "scale_dtype": "float32",
    "full_precision_blocks": "float32",
    "zero_point": 0,
}
LEXICAL_BINDING_SCHEMA = "mrun-qstore-lexical-binding-v1"


def _tensor_content_evidence(tensor) -> dict:
    """Return exact source-byte evidence without interpreting tensor values.

    Viewing the contiguous tensor as bytes also works for dtypes (notably bfloat16) that
    NumPy cannot represent directly.  Shape and dtype remain part of the equality proof so
    two differently interpreted byte strings are never treated as the same source tensor.
    """

    import torch

    value = tensor.detach().cpu().contiguous()
    payload = value.view(torch.uint8).numpy()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "bytes": payload.nbytes,
        "sha256": hashlib.sha256(memoryview(payload)).hexdigest(),
    }


def _encoded_content_evidence(components: dict[str, np.ndarray]) -> dict:
    if not components:
        raise ValueError("encoded QStore content evidence requires at least one component")
    records = {}
    total_bytes = 0
    for name, value in sorted(components.items()):
        array = np.ascontiguousarray(value)
        records[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "bytes": array.nbytes,
            "sha256": hashlib.sha256(memoryview(array)).hexdigest(),
        }
        total_bytes += array.nbytes
    return {"bytes": total_bytes, "components": records}


class LexicalWeightBinding:
    """Fail-closed embed/head binding policy shared by QStore builders.

    A physical alias is legal only when ``tie_word_embeddings`` is literally true in the
    source config.  If the checkpoint also serializes ``lm_head``, both the source bytes and
    the codec output must match ``embed`` exactly before the duplicate payload is omitted.
    """

    _NAMES = frozenset(("embed", "lm_head"))

    def __init__(self, raw_config: dict) -> None:
        self.declared_tied = raw_config.get("tie_word_embeddings") is True
        self._source: dict[str, dict] = {}
        self._encoded: dict[str, dict] = {}

    def observe_source(self, name: str, tensor) -> None:
        if name not in self._NAMES:
            return
        if name in self._source:
            raise RuntimeError(f"duplicate canonical QStore block {name!r}")
        self._source[name] = _tensor_content_evidence(tensor)

    def observe_encoded(self, name: str, **components: np.ndarray) -> bool:
        """Record codec output and return whether it should be physically written."""

        if name not in self._NAMES:
            return True
        if name in self._encoded:
            raise RuntimeError(f"duplicate encoded QStore block {name!r}")
        self._encoded[name] = _encoded_content_evidence(components)
        return self.should_write(name)

    def should_write(self, name: str) -> bool:
        return not (self.declared_tied and name == "lm_head")

    def finalize(self, blocks: dict[str, dict]) -> dict:
        if "embed" not in self._source or "embed" not in self._encoded or "embed" not in blocks:
            raise RuntimeError("QStore checkpoint has no physical embedding block")

        if not self.declared_tied:
            if "lm_head" not in self._source:
                raise RuntimeError(
                    "refusing to infer tied lexical weights: config does not declare "
                    "tie_word_embeddings=true and the checkpoint has no lm_head"
                )
            if "lm_head" not in self._encoded or "lm_head" not in blocks:
                raise RuntimeError("untied QStore checkpoint has no physical lm_head block")
            return {
                "schema_version": LEXICAL_BINDING_SCHEMA,
                "config_declared_tied": False,
                "disposition": "distinct-physical-blocks",
                "source_proof": {"kind": "not-applicable"},
                "encoded_proof": {"kind": "not-applicable"},
                "physical_blocks": ["embed", "lm_head"],
                "alias_saved_bytes": 0,
            }

        if "lm_head" in blocks:
            raise RuntimeError("tied QStore lm_head was written before alias verification")
        head_source = self._source.get("lm_head")
        head_encoded = self._encoded.get("lm_head")
        if head_source is not None:
            if head_source != self._source["embed"]:
                raise RuntimeError(
                    "config declares tie_word_embeddings=true but embed and lm_head "
                    "source tensors are not byte-identical"
                )
            if head_encoded is None or head_encoded != self._encoded["embed"]:
                raise RuntimeError(
                    "config declares tie_word_embeddings=true but embed and lm_head "
                    "encoded payloads are not byte-identical"
                )
            source_proof = {
                "kind": "shape-dtype-byte-length-sha256-identity",
                "shared": self._source["embed"],
            }
            encoded_proof = {
                "kind": "shape-dtype-byte-length-sha256-identity",
                "shared": self._encoded["embed"],
            }
            disposition = "verified-duplicate-alias"
        else:
            if head_encoded is not None:
                raise RuntimeError("lm_head codec output exists without a source tensor")
            source_proof = {
                "kind": "config-declared-head-omission",
                "embed": self._source["embed"],
            }
            encoded_proof = {
                "kind": "single-physical-payload",
                "embed": self._encoded["embed"],
            }
            disposition = "declared-omission-alias"

        blocks["lm_head"] = {"alias": "embed"}
        return {
            "schema_version": LEXICAL_BINDING_SCHEMA,
            "config_declared_tied": True,
            "disposition": disposition,
            "source_proof": source_proof,
            "encoded_proof": encoded_proof,
            "physical_blocks": ["embed"],
            "logical_aliases": {"lm_head": "embed"},
            "alias_saved_bytes": self._encoded["embed"]["bytes"],
        }


def preflight_lexical_binding(
    files: list[Path],
    arch: str,
    raw_config: dict,
    encode,
) -> tuple[LexicalWeightBinding, dict]:
    """Prove a lexical binding before a legacy builder opens its output blobs.

    The low-bit experimental builders predate atomic publication.  A lexical-only preflight
    lets them share the exact alias policy without allowing a declared-tie mismatch to
    overwrite an existing artifact before it is discovered.
    """

    from safetensors import safe_open

    binding = LexicalWeightBinding(raw_config)
    for source_file in sorted(files):
        with safe_open(str(source_file), framework="pt") as tensors:
            for key in tensors.keys():
                name = _canon(key, arch)
                if name not in binding._NAMES:
                    continue
                tensor = tensors.get_tensor(key)
                binding.observe_source(name, tensor)
                binding.observe_encoded(name, **encode(tensor))
                del tensor
    proof_blocks = {"embed": {"kind": "preflight"}}
    if not binding.declared_tied:
        proof_blocks["lm_head"] = {"kind": "preflight"}
    proof = binding.finalize(proof_blocks)
    return binding, proof


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024**2) if sys.platform == "darwin" else r / 1024


# ---- canonical name mapping (HF state-dict keys -> store block names), per arch ----
# The write/output projection of every arch is named ``L{L}.down`` so the engine's
# down_weight()/write_norm() are arch-agnostic (qwen mlp.down_proj, neox dense_4h_to_h,
# mamba mixer.out_proj all map to it).
_LAYER_RE = re.compile(
    r"^(?:model\.language_model\.|model\.|gpt_neox\.|backbone\.)?layers\.(\d+)\."
)


def _canon_qwen(L: int, tail: str) -> str | None:
    # qwen2 + llama share this map. qwen3 adds per-head q/k RMSNorm (q_norm/k_norm, shape
    # [head_dim]); those two tails are qwen3-only and simply never appear in a qwen2/llama
    # checkpoint, so mapping them here is harmless for the older archs and enables qwen3.
    return {
        "self_attn.q_proj.weight": f"L{L}.q",
        "self_attn.k_proj.weight": f"L{L}.k",
        "self_attn.v_proj.weight": f"L{L}.v",
        "self_attn.o_proj.weight": f"L{L}.o",
        "self_attn.q_proj.bias": f"L{L}.q.bias",
        "self_attn.k_proj.bias": f"L{L}.k.bias",
        "self_attn.v_proj.bias": f"L{L}.v.bias",
        "self_attn.q_norm.weight": f"L{L}.q_norm",  # qwen3: per-head RMSNorm on q (over head_dim)
        "self_attn.k_norm.weight": f"L{L}.k_norm",  # qwen3: per-head RMSNorm on k (over head_dim)
        "mlp.gate_proj.weight": f"L{L}.gate",
        "mlp.up_proj.weight": f"L{L}.up",
        "mlp.down_proj.weight": f"L{L}.down",
        "input_layernorm.weight": f"L{L}.ln1",
        "post_attention_layernorm.weight": f"L{L}.ln2",
    }.get(tail)


def _canon_qwen35(L: int, tail: str) -> str | None:
    """Canonical blocks for Qwen3.5's alternating GDN/full-attention decoder.

    The names intentionally retain the producer boundary: the GDN projections and its
    convolution/recurrence parameters are not collapsed into an ordinary QKV block.  The
    paged consumer uses these names to keep the convolution state, recurrent state, and full
    attention KV state separate.
    """

    return {
        "input_layernorm.weight": f"L{L}.ln1",
        "post_attention_layernorm.weight": f"L{L}.ln2",
        "self_attn.q_proj.weight": f"L{L}.q",
        "self_attn.k_proj.weight": f"L{L}.k",
        "self_attn.v_proj.weight": f"L{L}.v",
        "self_attn.o_proj.weight": f"L{L}.o",
        "self_attn.q_norm.weight": f"L{L}.q_norm",
        "self_attn.k_norm.weight": f"L{L}.k_norm",
        "linear_attn.in_proj_qkv.weight": f"L{L}.in_proj_qkv",
        "linear_attn.in_proj_z.weight": f"L{L}.in_proj_z",
        "linear_attn.in_proj_a.weight": f"L{L}.in_proj_a",
        "linear_attn.in_proj_b.weight": f"L{L}.in_proj_b",
        "linear_attn.out_proj.weight": f"L{L}.out_proj",
        "linear_attn.conv1d.weight": f"L{L}.conv1d.weight",
        "linear_attn.A_log": f"L{L}.A_log",
        "linear_attn.dt_bias": f"L{L}.dt_bias",
        "linear_attn.norm.weight": f"L{L}.gdn_norm",
        "mlp.gate_proj.weight": f"L{L}.gate",
        "mlp.up_proj.weight": f"L{L}.up",
        "mlp.down_proj.weight": f"L{L}.down",
    }.get(tail)


def _canon_neox(L: int, tail: str) -> str | None:
    return {
        "input_layernorm.weight": f"L{L}.ln1",
        "input_layernorm.bias": f"L{L}.ln1.bias",
        "post_attention_layernorm.weight": f"L{L}.ln2",
        "post_attention_layernorm.bias": f"L{L}.ln2.bias",
        "attention.query_key_value.weight": f"L{L}.qkv",
        "attention.query_key_value.bias": f"L{L}.qkv.bias",
        "attention.dense.weight": f"L{L}.o",
        "attention.dense.bias": f"L{L}.o.bias",
        "mlp.dense_h_to_4h.weight": f"L{L}.h_to_4h",
        "mlp.dense_h_to_4h.bias": f"L{L}.h_to_4h.bias",
        "mlp.dense_4h_to_h.weight": f"L{L}.down",
        "mlp.dense_4h_to_h.bias": f"L{L}.down.bias",
    }.get(tail)  # attention.bias / attention.masked_bias (causal-mask buffers) -> None


def _canon_mamba(L: int, tail: str) -> str | None:
    return {
        "norm.weight": f"L{L}.norm",
        "mixer.in_proj.weight": f"L{L}.in_proj",
        "mixer.conv1d.weight": f"L{L}.conv1d.weight",
        "mixer.conv1d.bias": f"L{L}.conv1d.bias",
        "mixer.x_proj.weight": f"L{L}.x_proj",
        "mixer.dt_proj.weight": f"L{L}.dt_proj",
        "mixer.dt_proj.bias": f"L{L}.dt_proj.bias",
        "mixer.A_log": f"L{L}.A_log",
        "mixer.D": f"L{L}.D",
        "mixer.out_proj.weight": f"L{L}.down",
    }.get(tail)


def _canon(key: str, arch: str) -> str | None:
    """Map a HF state-dict key to a canonical store block name (or None=skip)."""
    if key in (
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "gpt_neox.embed_in.weight",
        "backbone.embeddings.weight",
    ):
        return "embed"
    if key in ("lm_head.weight", "embed_out.weight"):
        return "lm_head"
    if key in (
        "model.norm.weight",
        "norm.weight",
        "model.language_model.norm.weight",
        "gpt_neox.final_layer_norm.weight",
        "backbone.norm_f.weight",
    ):
        return "norm.final"
    if key in ("gpt_neox.final_layer_norm.bias",):
        return "norm.final.bias"
    m = _LAYER_RE.search(key)
    if not m:
        return None
    L = int(m.group(1))
    tail = key.split(f"layers.{L}.", 1)[1]
    if arch in ("qwen2", "llama", "qwen3"):
        return _canon_qwen(L, tail)
    if arch in ("qwen3_5", "qwen3_5_text"):
        return _canon_qwen35(L, tail)
    if arch == "gpt_neox":
        return _canon_neox(L, tail)
    if arch == "mamba":
        return _canon_mamba(L, tail)
    return None


# blocks kept full-precision (tiny / not a 2D matmul weight: norms, biases, SSM params, conv)
def _is_fp32_block(name: str) -> bool:
    if name.endswith(
        (
            ".bias",
            ".ln1",
            ".ln2",
            ".norm",
            ".q_norm",
            ".k_norm",
            ".A_log",
            ".D",
            ".dt_bias",
            ".conv1d.weight",
            ".gdn_norm",
        )
    ):
        return True
    return name == "norm.final"


def _arch_config(arch: str, raw: dict) -> dict:
    """The config subset each arch's paged forward reads (superset keys are harmless)."""
    if arch in ("qwen2", "llama", "qwen3"):
        # qwen3 shares this exactly: head_dim is read from config (qwen3 sets it explicitly,
        # often != hidden/heads), attention_bias defaults False (qwen3 has no qkv bias), and the
        # q_norm/k_norm eps is rms_norm_eps (same field). The q/k-norm presence is detected at
        # forward time via store.has(f"L{{L}}.q_norm").
        return {
            "hidden_size": int(raw["hidden_size"]),
            "num_hidden_layers": int(raw["num_hidden_layers"]),
            "num_attention_heads": int(raw["num_attention_heads"]),
            "num_key_value_heads": int(raw.get("num_key_value_heads", raw["num_attention_heads"])),
            "head_dim": int(raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"])),
            "intermediate_size": int(raw["intermediate_size"]),
            "vocab_size": int(raw["vocab_size"]),
            "rms_norm_eps": float(raw.get("rms_norm_eps", 1e-6)),
            "rope_theta": float(raw.get("rope_theta", 10000.0)),
            "hidden_act": raw.get("hidden_act", "silu"),
            "attention_bias": bool(raw.get("attention_bias", False)),
        }
    if arch in ("qwen3_5", "qwen3_5_text"):
        text = raw.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("Qwen3.5 checkpoint has no nested text_config")
        rope = text.get("rope_parameters") or {}
        key_heads = int(text["linear_num_key_heads"])
        value_heads = int(text["linear_num_value_heads"])
        key_dim = int(text["linear_key_head_dim"])
        value_dim = int(text["linear_value_head_dim"])
        return {
            "hidden_size": int(text["hidden_size"]),
            "num_hidden_layers": int(text["num_hidden_layers"]),
            "num_attention_heads": int(text["num_attention_heads"]),
            "num_key_value_heads": int(text["num_key_value_heads"]),
            "head_dim": int(text["head_dim"]),
            "intermediate_size": int(text["intermediate_size"]),
            "vocab_size": int(text["vocab_size"]),
            "rms_norm_eps": float(text.get("rms_norm_eps", 1e-6)),
            "rope_theta": float(rope.get("rope_theta", 10_000_000.0)),
            "partial_rotary_factor": float(
                rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
            ),
            "attention_bias": bool(text.get("attention_bias", False)),
            "layer_types": list(text["layer_types"]),
            "linear_num_key_heads": key_heads,
            "linear_num_value_heads": value_heads,
            "linear_key_head_dim": key_dim,
            "linear_value_head_dim": value_dim,
            "linear_conv_kernel_dim": int(text.get("linear_conv_kernel_dim", 4)),
            "linear_conv_width": 2 * key_heads * key_dim + value_heads * value_dim,
            "linear_value_width": value_heads * value_dim,
            "attn_output_gate": bool(text.get("attn_output_gate", True)),
            "mrope_interleaved": bool(rope.get("mrope_interleaved", True)),
            "mrope_section": list(rope.get("mrope_section", [])),
        }
    if arch == "gpt_neox":
        h = int(raw["hidden_size"])
        nh = int(raw["num_attention_heads"])
        hd = h // nh
        rpct = float(raw.get("rotary_pct", raw.get("partial_rotary_factor", 1.0)))
        return {
            "hidden_size": h,
            "num_hidden_layers": int(raw["num_hidden_layers"]),
            "num_attention_heads": nh,
            "head_dim": hd,
            "rotary_ndims": int(hd * rpct),
            "intermediate_size": int(raw["intermediate_size"]),
            "vocab_size": int(raw["vocab_size"]),
            "layer_norm_eps": float(raw.get("layer_norm_eps", 1e-5)),
            "rope_theta": float(raw.get("rotary_emb_base", raw.get("rope_theta", 10000.0))),
            "use_parallel_residual": bool(raw.get("use_parallel_residual", True)),
            "hidden_act": raw.get("hidden_act", "gelu"),
        }
    if arch == "mamba":
        import math

        h = int(raw["hidden_size"])
        d_inner = int(raw.get("intermediate_size", int(raw.get("expand", 2)) * h))
        tsr = raw.get("time_step_rank", "auto")
        dt_rank = int(math.ceil(h / 16)) if tsr == "auto" else int(tsr)
        return {
            "hidden_size": h,
            "intermediate_size": d_inner,
            "state_size": int(raw.get("state_size", 16)),
            "conv_kernel": int(raw.get("conv_kernel", 4)),
            "time_step_rank": dt_rank,
            "num_hidden_layers": int(raw["num_hidden_layers"]),
            "vocab_size": int(raw["vocab_size"]),
            "layer_norm_epsilon": float(raw.get("layer_norm_epsilon", 1e-5)),
        }
    raise NotImplementedError(arch)


def _quant_row_int8(W: np.ndarray):
    """Per-output-channel (per-row, axis=1) symmetric int8. Matches _fake_quant_int8_."""
    scale = np.abs(W).max(axis=1, keepdims=True) / 127.0  # [out,1]
    scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32)
    q = np.round(W / scale).clip(-127, 127).astype(np.int8)  # [out,in]
    return q, scale[:, 0].astype(np.float32)  # scale -> [out]


def _verify_existing_store(
    out: Path,
    *,
    model_name: str,
    arch: str,
    config: dict,
    source: dict,
    builder: dict,
) -> Path:
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"refusing to overwrite incomplete QStore directory: {out}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != QSTORE_SCHEMA:
        raise RuntimeError(f"existing QStore is legacy or schema-mismatched: {out}")
    expected = {
        "model_name": model_name,
        "arch": arch,
        "dtype": "int8",
        "config": config,
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"existing QStore identity/config mismatch: {actual} != {expected}")
    verify_source_provenance(manifest.get("source"), source)
    verify_builder_provenance(manifest.get("builder"), builder)
    semantic_manifest = {key: value for key, value in manifest.items() if key != "derived"}
    verify_derived_provenance(
        out,
        manifest.get("derived"),
        expected_filenames=QSTORE_FILES,
        semantic_manifest=semantic_manifest,
    )

    blocks = manifest.get("blocks")
    if not isinstance(blocks, dict):
        raise RuntimeError("existing QStore has no block table")
    qrows = [block for block in blocks.values() if block.get("kind") == "qrow"]
    fp32 = [block for block in blocks.values() if block.get("kind") == "fp32"]
    if not qrows or not fp32:
        raise RuntimeError("existing QStore block table is incomplete")
    boundaries = {
        "weights.i8": max(block["w_off"] + block["w_len"] for block in qrows),
        "scales.f32": max(block["s_off"] + block["s_len"] for block in qrows),
        "extras.f32": max(block["e_off"] + block["e_len"] for block in fp32),
    }
    actual_sizes = {name: (out / name).stat().st_size for name in QSTORE_FILES}
    if boundaries != actual_sizes:
        raise RuntimeError(f"existing QStore block/file boundary mismatch: {boundaries}")
    return out


def build(
    model_name: str, *, out_root: Path | None = None, store_dir_name: str | None = None
) -> Path:
    """Build the int8 paged store for ``model_name`` under ``out_root`` (defaults to
    ``stores_root()``). The store directory is named by ``store_dir_name`` (defaults to the
    HF-id basename so it matches the engine's lookup key, e.g. ``Qwen2.5-0.5B``)."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    files = find_safetensors(model_name)
    if not files:
        raise FileNotFoundError(f"no safetensors for {model_name!r}")
    spec = resolve_model(model_name)
    snap = files[0].parent
    cfg = AutoConfig.from_pretrained(str(snap))
    raw = json.loads((snap / "config.json").read_text())  # authoritative: AutoConfig objects in
    arch = raw.get(
        "model_type", getattr(cfg, "model_type", "?")
    )  # this tf version drops rope_theta etc
    if arch not in (
        "qwen2",
        "llama",
        "qwen3",
        "qwen3_5",
        "qwen3_5_text",
        "gpt_neox",
        "mamba",
    ):
        raise NotImplementedError(
            f"arch {arch!r} not supported (qwen2/llama/qwen3/qwen3_5/gpt_neox/mamba)"
        )

    source = build_source_provenance(
        snap,
        files,
        model_name=spec.name,
        hf_id=spec.hf_id,
    )
    builder = build_builder_provenance(
        [Path(__file__)],
        name="mrun.engine.kernels.qstore_build",
        schema_version=QSTORE_SCHEMA,
        quantization=QSTORE_QUANTIZATION,
    )
    config = _arch_config(arch, raw)
    root = Path(out_root) if out_root is not None else stores_root()
    root.mkdir(parents=True, exist_ok=True)
    dir_name = store_dir_name or store_name(model_name)
    out = root / dir_name
    if out.exists():
        return _verify_existing_store(
            out,
            model_name=spec.name,
            arch=arch,
            config=config,
            source=source,
            builder=builder,
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{dir_name}.building-", dir=root))
    w_path = temporary / "weights.i8"
    s_path = temporary / "scales.f32"
    e_path = temporary / "extras.f32"
    w_off = s_off = e_off = 0
    blocks: dict[str, dict] = {}
    seen_blocks: set[str] = set()
    lexical_binding = LexicalWeightBinding(raw)

    print(f">>> qstore build — {model_name}  arch={arch}  snap={snap.name}  -> {out}")
    n_q = n_fp = 0
    try:
        with w_path.open("wb") as wf, s_path.open("wb") as sf, e_path.open("wb") as ef:
            for source_file in sorted(files):
                with safe_open(str(source_file), framework="pt") as st:
                    for key in st.keys():
                        name = _canon(key, arch)
                        if name is None:
                            continue
                        if name in seen_blocks:
                            raise RuntimeError(f"duplicate canonical QStore block {name!r}")
                        seen_blocks.add(name)
                        tensor = st.get_tensor(key)
                        lexical_binding.observe_source(name, tensor)
                        weight = tensor.to(dtype=torch.float32).numpy()
                        del tensor
                        if _is_fp32_block(name):
                            array = np.ascontiguousarray(weight.astype(np.float32))
                            ef.write(array.tobytes())
                            blocks[name] = {
                                "kind": "fp32",
                                "shape": list(weight.shape),
                                "e_off": e_off,
                                "e_len": array.nbytes,
                            }
                            e_off += array.nbytes
                            n_fp += 1
                        else:
                            if weight.ndim != 2:
                                raise ValueError(f"{name}: expected 2D, got {weight.shape}")
                            quantized, scale = _quant_row_int8(weight)
                            quantized = np.ascontiguousarray(quantized)
                            scale = np.ascontiguousarray(scale)
                            if lexical_binding.observe_encoded(
                                name, weights=quantized, scales=scale
                            ):
                                wf.write(quantized.tobytes())
                                sf.write(scale.tobytes())
                                blocks[name] = {
                                    "kind": "qrow",
                                    "shape": list(weight.shape),
                                    "w_off": w_off,
                                    "w_len": quantized.nbytes,
                                    "s_off": s_off,
                                    "s_len": scale.nbytes,
                                }
                                w_off += quantized.nbytes
                                s_off += scale.nbytes
                                n_q += 1
                        del weight
                print(f"    {source_file.name}: blocks={len(blocks)}  RSS={rss_mb():.0f}MB")

        lexical_manifest = lexical_binding.finalize(blocks)
        tie = lexical_binding.declared_tied
        manifest = {
            "schema_version": QSTORE_SCHEMA,
            "model_name": spec.name,
            "arch": arch,
            "dtype": "int8",
            "tie_word_embeddings": tie,
            "lexical_weight_binding": lexical_manifest,
            "config": config,
            "source": source,
            "builder": builder,
            "blocks": blocks,
        }
        manifest["derived"] = build_derived_provenance(
            temporary,
            QSTORE_FILES,
            semantic_manifest=manifest,
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(out)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    disk_mb = sum((out / name).stat().st_size for name in QSTORE_FILES) / 1e6
    print(
        f"  quantized={n_q} blocks  fp32={n_fp} blocks  "
        f"store={disk_mb:.1f}MB  peakRSS={rss_mb():.0f}MB"
    )
    print(f"  -> {out}")
    return out
