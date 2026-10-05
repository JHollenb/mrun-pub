"""Architecture-family dispatch for activation and write-vector access."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

_DENSE_SWIGLU = {"qwen2", "qwen3", "qwen3_5", "qwen3_5_text", "llama", "mistral"}
_MOE = {"qwen3_moe", "olmoe"}  # dense attention + per-expert SwiGLU MLP + a router


class Resolver:
    """Family-specific access to layers, activations, and write projections."""

    def __init__(self, model: Any):
        model_type = str(getattr(model.config, "model_type", ""))
        if model_type == "gpt_neox":
            self.family = "gpt_neox"
        elif model_type in _MOE:
            self.family = "moe"
        elif model_type in _DENSE_SWIGLU:
            self.family = "swiglu"
        elif model_type in ("mamba", "falcon_mamba"):
            # falcon_mamba is HF MambaForCausalLM-compatible (same backbone.layers.N.mixer
            # layout); alias it once so every family=="mamba" dispatch picks it up for free.
            self.family = "mamba"
        elif model_type == "gpt2":
            self.family = "gpt2"
        else:
            raise NotImplementedError(f"unsupported model_type {model_type!r}")
        # discovery-recorder name for the same family tag (ported code branches on ar.fam)
        self.fam = self.family

    def layers(self, model: Any) -> Any:
        if self.family == "gpt_neox":
            return model.gpt_neox.layers
        if self.family == "mamba":
            return model.backbone.layers
        if self.family == "gpt2":
            return model.transformer.h
        # dense SwiGLU AND moe (qwen3_moe/olmoe: dense attention, per-expert MLP)
        return model.model.layers

    def down_weight(self, block: Any) -> torch.Tensor:
        if self.family == "gpt_neox":
            return block.mlp.dense_4h_to_h.weight
        if self.family == "mamba":
            return block.mixer.out_proj.weight
        if self.family == "gpt2":
            return block.mlp.c_proj.weight.T
        if self.family == "moe":
            raise RuntimeError("moe: no single down_weight (per-expert); use moe_expert_down_weights")
        return block.mlp.down_proj.weight

    def moe_expert_down_weights(self, block: Any) -> list[torch.Tensor]:
        """MoE: list of each expert's down_proj weight [hidden, inter]. Router MLP has E experts."""
        experts = getattr(getattr(block, "mlp", block), "experts", None)
        if experts is None:
            raise RuntimeError("moe: could not find blk.mlp.experts")
        return [e.down_proj.weight for e in experts]

    def add_act_hook(self, block: Any, fn: Callable[[torch.Tensor], None]) -> Any:
        if self.family in {"gpt_neox", "gpt2"}:
            return block.mlp.act.register_forward_hook(lambda _m, _i, out: fn(out.detach()))
        projection = block.mixer.out_proj if self.family == "mamba" else block.mlp.down_proj
        return projection.register_forward_pre_hook(lambda _m, inp: fn(inp[0].detach()))

    def add_patch_hook(
        self,
        block: Any,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Any:
        if self.family in {"gpt_neox", "gpt2"}:
            return block.mlp.act.register_forward_hook(lambda _m, _i, out: fn(out))
        projection = block.mixer.out_proj if self.family == "mamba" else block.mlp.down_proj

        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            return (fn(inputs[0]), *inputs[1:])

        return projection.register_forward_pre_hook(pre_hook)

    def intermediate(self, model: Any) -> int:
        cfg = model.config
        if self.family == "mamba":
            return int(cfg.hidden_size) * int(getattr(cfg, "expand", 2))
        if self.family == "gpt2":
            n_inner = getattr(cfg, "n_inner", None)
            return int(n_inner) if n_inner else int(cfg.n_embd) * 4
        if self.family == "moe":
            return int(getattr(cfg, "moe_intermediate_size",
                               getattr(cfg, "intermediate_size", 0)) or 0)
        return int(getattr(cfg, "intermediate_size", 0) or 0)

    def write_norm(self, model: Any, dims_: dict[str, int] | None = None) -> np.ndarray:
        info = dims_ or dims(model)
        layers = self.layers(model)
        inter = int(info["intermediate"])
        out = np.zeros(len(layers) * inter, dtype=np.float64)
        for layer_idx, block in enumerate(layers):
            # .float() BEFORE .numpy(): numpy chokes on bf16 (the repo bf16 gotcha —
            # dropped in the initial port, restored by the adversarial fidelity audit)
            weight = self.down_weight(block).detach().float().cpu().numpy()
            out[layer_idx * inter : (layer_idx + 1) * inter] = np.linalg.norm(weight, axis=0)
        return out

    # discovery-recorder name for the same per-neuron write-column norms
    write_mags = write_norm

    def unembed_weight(self, model: Any) -> torch.Tensor:
        """[vocab, hidden] output/unembedding projection (family-dispatched; may be tied to embed)."""
        if self.fam == "gpt_neox":
            return model.embed_out.weight
        if self.fam == "gpt2":
            return model.lm_head.weight            # tied to wte
        return model.lm_head.weight                # dense SwiGLU + mamba

    def final_norm_weight(self, model: Any) -> torch.Tensor | None:
        """Final pre-unembed norm GAIN [hidden], or None if the family has no final norm.
        (LayerNorm centering is dropped — the gain diagonal is what the dla direction proxy needs.)"""
        if self.fam == "gpt_neox":
            return model.gpt_neox.final_layer_norm.weight
        if self.fam == "gpt2":
            return model.transformer.ln_f.weight
        if self.fam == "mamba":
            return model.backbone.norm_f.weight
        return model.model.norm.weight             # dense SwiGLU (qwen/llama/mistral)


def resolver(model: Any) -> Resolver:
    return Resolver(model)


def dims(model: Any) -> dict[str, int | str]:
    cfg = model.config
    n_head = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0) or 0)
    hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0) or 0)
    info: dict[str, int | str] = {
        "arch": str(getattr(cfg, "model_type", "")),
        "n_layer": int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)) or 0),
        "n_head": n_head,
        "hidden": hidden,
        "d_head": int(getattr(cfg, "head_dim", hidden // n_head if n_head else 0) or 0),
        "intermediate": int(getattr(cfg, "intermediate_size", 0) or 0),
    }
    if not info["intermediate"]:
        info["intermediate"] = Resolver(model).intermediate(model)
    return info
