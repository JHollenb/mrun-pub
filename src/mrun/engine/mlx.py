"""MLX (Apple Metal GPU) backend.

Runs dense transformers (qwen2/qwen3/llama, gpt_neox) and Mamba through ``mlx_lm`` on the Metal
GPU, with the same intervention surface as the other engines: the patch / capture point is the
activation feeding the MLP down/output projection (Rule #1). MLX is an optional dependency; the
engine falls back to paged when it is absent. No batched fusion yet — ``logits_batch`` uses the
BaseEngine scalar loop.

Vendored from discovery/src/common/engine.py: the hardcoded converted-model aliases were dropped
in favour of ``snapshot_dir`` resolution; mlx imports stay lazy inside ``__init__``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from ..models import load_tokenizer, resolve_model, snapshot_dir
from ._base_impl import BaseEngine


def _resolve_mlx_model_path(model_name: str) -> Path:
    """Resolve to a dir ``mlx_lm.load`` can read (config.json + weights): a literal path with a
    config.json, else the HF snapshot dir for a registry/HF model (mlx_lm loads the safetensors
    directly, no conversion needed)."""
    raw = Path(model_name).expanduser()
    if (raw / "config.json").exists():
        return raw
    snap = snapshot_dir(model_name)
    if (snap / "config.json").exists():
        return snap
    raise FileNotFoundError(f"no local MLX-readable model dir for {model_name!r}")


MLX_Q4_SUFFIX = "q4g64"


def _mlx_q4_root() -> Path:
    import os

    return Path(os.environ.get("MRUN_MLX_Q4_ROOT", "~/.cache/mrun/mlx-q4")).expanduser()


def _resolve_mlx_q4_path(model_name: str) -> Path:
    """Resolve a pre-quantized int4 group-64 MLX export for ``model_name``.

    Order: a literal already-quantized dir, then ``<MRUN_MLX_Q4_ROOT>/<name>-q4g64``. A missing
    export is converted in-place from the local HF snapshot when ``MRUN_MLX_Q4_AUTOCONVERT=1``
    (one-time, ~bits/4 of the bf16 size on disk), otherwise the error carries the exact convert
    command so the export stays a deliberate act."""
    import os

    raw = Path(model_name).expanduser()
    if (raw / "config.json").exists():
        cfg = json.loads((raw / "config.json").read_text())
        if "quantization" in cfg:
            return raw
    spec = resolve_model(model_name)
    out = _mlx_q4_root() / f"{spec.name}-{MLX_Q4_SUFFIX}"
    if (out / "config.json").exists():
        return out
    snap = _resolve_mlx_model_path(model_name)
    if os.environ.get("MRUN_MLX_Q4_AUTOCONVERT") == "1":
        from mlx_lm import convert

        out.parent.mkdir(parents=True, exist_ok=True)
        convert(str(snap), mlx_path=str(out), quantize=True, q_bits=4, q_group_size=64)
        return out
    raise FileNotFoundError(
        f"no q4g64 MLX export for {model_name!r}; create it with\n"
        f"  python -m mlx_lm convert --hf-path {snap} --mlx-path {out} -q --q-bits 4 --q-group-size 64\n"
        f"or set MRUN_MLX_Q4_AUTOCONVERT=1"
    )


def _mx_np_f32(mx, arr) -> np.ndarray:
    """mlx array → float32 numpy. bf16 mlx arrays have no numpy buffer (PEP-3118 itemsize
    mismatch), so cast to float32 inside mlx first — covers bf16/f16/int8 models uniformly."""
    arr = arr.astype(mx.float32)
    mx.eval(arr)
    return np.asarray(arr)


def _dequant_mlx_linear(mx, proj) -> torch.Tensor:
    """[out, in] float matrix for an mlx Linear or QuantizedLinear (column n = neuron n)."""
    if hasattr(proj, "scales") and hasattr(proj, "bits"):
        W = mx.dequantize(
            proj.weight,
            scales=proj.scales,
            biases=getattr(proj, "biases", None),
            group_size=proj.group_size,
            bits=proj.bits,
            mode=getattr(proj, "mode", "affine"),
        )
    else:
        W = proj.weight
    return torch.from_numpy(_mx_np_f32(mx, W))


class _MLXInterventionBase(BaseEngine):
    """Shared surface for MLX-backed intervention engines (Metal).

    The patch *installation* mechanism is architecture-specific — Mamba patches the mixer's
    ``_process_sequence`` (a named instance method), transformers patch the MLP type's
    ``__call__`` (resolved on the type) — so subclasses implement ``_install_patchers`` /
    ``_restore_patchers`` / ``down_weight``. The patch point is always the activation feeding
    the write/output projection (Rule #1)."""

    backend = "mlx"
    supports_batch = False

    def capabilities(self):
        # MLX implements MLP patching + selected capture (forward_patched); it has no head
        # tap, no attention/residual capture, no raw nn.Module, and fp16/int8 weights are
        # approximate vs the fp32 reference. Without this override the BaseEngine default
        # mis-reported mlp_patch/selected_capture as False and recorders skipped working
        # measurements (adversarial-review F3).
        from .base import EngineCapabilities

        return EngineCapabilities(
            logits=True,
            logits_batch=bool(self.supports_batch),
            mlp_acts=True,
            mlp_acts_batch=bool(self.supports_batch),
            mlp_patch=True,
            mlp_patch_batch=False,   # forward_patched_batch is the scalar-loop fallback
            head_patch=False,
            head_patch_batch=False,
            selected_capture=True,
            attentions=False,
            residual_tap=False,
            raw_model=False,
            autograd=False,
            exact_reference=False,
            approximate_quantized=True,
            generation=True,
        )

    def generate(
        self,
        prompt: str | list[int] | np.ndarray,
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        stop_ids: tuple[int, ...] = (),
        return_text: bool = False,
        cache_mb: float | None = None,
    ) -> list[int] | str:
        """Greedy generation through mlx-lm's resident, stateful decode path.

        mlx-lm prefills once and carries the transformer KV cache or Mamba recurrent state across
        tokens. ``cache_mb`` is accepted for call compatibility with ``PagedEngine.generate`` but
        has no effect: MLX weights are resident and therefore use O(model) unified memory.
        """
        del cache_mb
        if max_new_tokens <= 0:
            return "" if return_text else []

        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        if isinstance(prompt, str):
            ids = self.encode([prompt], add_special_tokens=add_special_tokens)[0].tolist()
        else:
            ids = [int(token) for token in np.asarray(prompt, dtype=np.int64).tolist()]
        eos = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.tokenizer, "eos_token_id", None)
        )
        stop = set(stop_ids) | ({int(eos)} if eos is not None else set())
        new: list[int] = []
        for token, _logprobs in generate_step(
            self._mx.array(ids),
            self.model,
            max_tokens=int(max_new_tokens),
            sampler=make_sampler(temp=0.0),
        ):
            token_id = int(token.item() if hasattr(token, "item") else token)
            if token_id in stop:
                break
            new.append(token_id)
        if return_text:
            return self.tokenizer.decode(new, skip_special_tokens=True)
        return new

    def _mx_array(self, value: Any):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return self._mx.array(value)

    def _set_mx_columns(self, x, cols, target):
        current = x[..., cols]
        return x.at[..., cols].add(target - current)

    def _apply_mlx_patch_ops(self, x, ops: list | tuple):
        mx = self._mx
        out = x
        for op, cols_t, vals in ops:
            cols = self._mx_array(cols_t).astype(mx.int32)
            if op == "zero":
                out = out.at[..., cols].multiply(0.0)
            elif op == "scale":
                out = out.at[..., cols].multiply(float(vals))
            elif op == "global_mean":
                v = self._mx_array(vals).astype(out.dtype)
                out = self._set_mx_columns(out, cols, v)
            elif op == "position_mean":
                v = self._mx_array(vals).astype(out.dtype)
                if out.ndim == 2:
                    target = v[: out.shape[0]]
                else:
                    target = mx.expand_dims(v[: out.shape[1]], axis=0)
                out = self._set_mx_columns(out, cols, target)
            elif op == "add_amp":
                v = self._mx_array(vals).astype(out.dtype)
                out = out.at[..., cols].add(v)
            elif op == "center":
                out = out.at[..., cols].add(-mx.mean(out, axis=-1, keepdims=True))
            else:
                raise ValueError(f"unknown patch op {op}")
        return out

    def _capture_mx_columns(self, x, locals_: list[int] | tuple[int, ...]) -> torch.Tensor:
        mx = self._mx
        cols = mx.array(locals_, dtype=mx.int32)
        cap = x[..., cols]
        if cap.ndim == 3 and cap.shape[0] == 1:
            cap = cap[0]
        return torch.from_numpy(_mx_np_f32(mx, cap)).to(dtype=torch.float16)

    # subclasses implement these ---------------------------------------------
    def _install_patchers(self, *, patch_ops_by_layer, selected_maps, collect_acts, acts, captured):
        raise NotImplementedError

    def _restore_patchers(self, state) -> None:
        raise NotImplementedError

    # generic forward wrappers ------------------------------------------------
    def _mx_logits(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> tuple[Any, list, dict[int, torch.Tensor]]:
        mx = self._mx
        acts: list = [None] * self.n_layer if collect_acts else []
        captured: dict[int, torch.Tensor] = {}
        state = self._install_patchers(
            patch_ops_by_layer=patch_ops_by_layer,
            selected_maps=selected_maps,
            collect_acts=collect_acts,
            acts=acts,
            captured=captured,
        )
        try:
            arr = mx.array(np.asarray(ids, dtype=np.int64)[None, :])
            logits = self.model(arr, cache=None)[0]
            mx.eval(logits)
        finally:
            self._restore_patchers(state)
        return logits, acts, captured

    def logits(self, ids: np.ndarray) -> torch.Tensor:
        logits, _, _ = self._mx_logits(ids)
        return torch.from_numpy(_mx_np_f32(self._mx, logits))

    def forward_acts(self, ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        logits, acts, _ = self._mx_logits(ids, collect_acts=True)
        return torch.from_numpy(_mx_np_f32(self._mx, logits)), acts

    def forward_patched(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        logits, acts, captured = self._mx_logits(
            ids,
            patch_ops_by_layer=patch_ops_by_layer,
            selected_maps=selected_maps,
            collect_acts=collect_acts,
        )
        return torch.from_numpy(_mx_np_f32(self._mx, logits)), acts, captured


class MambaMLXEngine(_MLXInterventionBase):
    """MLX-backed Mamba intervention engine. Patches the tensor just before ``out_proj``."""

    def __init__(self, model_name: str, **_ignored: Any):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.models.activations import swiglu

        self._mx = mx
        self._nn = nn
        self._swiglu = swiglu
        self.spec = resolve_model(model_name)
        self.name = self.spec.name
        self.path = _resolve_mlx_model_path(model_name)
        self.model, self._mlx_tokenizer = load(str(self.path), lazy=True)
        self.tokenizer = load_tokenizer(self.spec)
        self.model.eval()
        self.cfg = self.model.args
        if getattr(self.cfg, "model_type", None) != "mamba":
            raise ValueError(f"MambaMLXEngine supports Mamba only, got {self.cfg.model_type!r}")
        self.n_layer = int(self.cfg.num_hidden_layers)
        self.inter = int(self.cfg.intermediate_size)
        self.hidden = int(self.cfg.hidden_size)

    def _patched_process_sequence(
        self, mixer, x, conv_cache, state_cache, *, layer_idx, patch_ops_by_layer,
        selected_maps, collect_acts, acts, captured,
    ):
        mx, nn = self._mx, self._nn
        B, T, _D = x.shape
        xz = mixer.in_proj(x)
        x, z = xz.split(indices_or_sections=2, axis=-1)
        K = mixer.conv_kernel_size
        if conv_cache is not None:
            x_full = mx.concatenate([conv_cache, x], axis=1)
        else:
            x_full = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        conv_out = mixer.conv1d(x_full)
        new_conv_cache = x_full[:, -(K - 1):, :]
        x = nn.silu(conv_out)
        A = -mx.exp(mixer.A_log)
        current_state = state_cache
        y = []
        for t in range(T):
            y_t, current_state = mixer.ssm_step(x[:, t], A, current_state)
            y.append(y_t)
        y = mx.stack(y, axis=1)
        hid = self._swiglu(z, y)
        ops = (patch_ops_by_layer or {}).get(layer_idx, [])
        if ops:
            hid = self._apply_mlx_patch_ops(hid, ops)
        locals_ = (selected_maps or {}).get(layer_idx, {}).get("locals", [])
        if collect_acts:
            acts[layer_idx] = torch.from_numpy(_mx_np_f32(self._mx, hid[0]))
        if locals_:
            captured[layer_idx] = self._capture_mx_columns(hid, locals_)
        out = mixer.out_proj(hid)
        return out, (new_conv_cache, current_state)

    def _install_patchers(self, *, patch_ops_by_layer, selected_maps, collect_acts, acts, captured):
        hook_layers = set(patch_ops_by_layer or {}) | set(selected_maps or {})
        if collect_acts:
            hook_layers.update(range(self.n_layer))
        originals = []
        for li, layer in enumerate(self.model.layers):
            if li not in hook_layers:
                continue
            mixer = layer.mixer
            original = mixer._process_sequence

            def make(layer_idx: int):
                def patched(mixer_self, x, conv_cache, state_cache):
                    return self._patched_process_sequence(
                        mixer_self, x, conv_cache, state_cache, layer_idx=layer_idx,
                        patch_ops_by_layer=patch_ops_by_layer, selected_maps=selected_maps,
                        collect_acts=collect_acts, acts=acts, captured=captured,
                    )

                return patched

            mixer._process_sequence = MethodType(make(li), mixer)
            originals.append((mixer, original))
        return originals

    def _restore_patchers(self, originals) -> None:
        for mixer, original in originals:
            mixer._process_sequence = original

    def down_weight(self, layer: int) -> torch.Tensor:
        return _dequant_mlx_linear(self._mx, self.model.layers[int(layer)].mixer.out_proj)


class TransformerMLXEngine(_MLXInterventionBase):
    """MLX intervention engine for dense transformers (qwen2/qwen3/llama, gpt_neox).

    Patches the activation feeding the MLP down/output projection. That call goes through the
    MLP type's ``__call__`` dunder (resolved on the type), so the patch is installed on
    ``type(mlp).__call__`` with an id-keyed layer registry, and the tap recomputes the model's
    own pre-down activation verbatim so hooked and unhooked layers stay numerically identical."""

    _SUPPORTED = {
        "qwen2": "swiglu",
        "qwen3": "swiglu",
        "llama": "swiglu",
        "gpt_neox": "neox",
    }
    supports_batch = True   # dense transformers fuse a [G,T] batch through the resident GPU model

    def __init__(self, model_name: str, mlx_path: str | Path | None = None, **_ignored: Any):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.models.activations import swiglu

        self._mx = mx
        self._nn = nn
        self._swiglu = swiglu
        self._gelu = nn.gelu_approx          # mlx_lm gpt_neox uses the fast/approx GELU
        self.spec = resolve_model(model_name)
        self.name = self.spec.name
        self.path = Path(mlx_path) if mlx_path is not None else _resolve_mlx_model_path(model_name)
        self.model, self._mlx_tokenizer = load(str(self.path), lazy=True)
        self.tokenizer = load_tokenizer(self.spec)
        self.model.eval()
        self.cfg = self.model.args
        mt = getattr(self.cfg, "model_type", None)
        if mt not in self._SUPPORTED:
            raise ValueError(f"TransformerMLXEngine supports {sorted(self._SUPPORTED)}, got {mt!r}")
        self._mlp_kind = self._SUPPORTED[mt]
        self.n_layer = int(self.cfg.num_hidden_layers)
        self.hidden = int(self.cfg.hidden_size)
        inter = getattr(self.cfg, "intermediate_size", None)
        if inter is None:  # gpt_neox ModelArgs omits it — read the down-proj input width
            inter = int(self.down_weight(0).shape[-1])
        self.inter = int(inter)

    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        """Fuse a batch through the resident Metal model in ONE forward per length-group.

        mlx_lm builds a single causal mask per call and does NOT key-pad ⇒ group rows by EXACT
        length so every row in a `[G, T]` forward is full-length (same trick as the ANE path).
        Returns per-row `[T_i, V]` fp32 (the `logits_batch` contract), so forced-choice scoring
        reads answer-position logits unchanged. Unpatched fast path (no hooks) — patched scoring
        still goes through the scalar `forward_patched`."""
        from collections import defaultdict

        mx = self._mx
        ids = [np.asarray(x, dtype=np.int64) for x in ids_list]
        out: list[torch.Tensor | None] = [None] * len(ids)
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, x in enumerate(ids):
            by_len[int(len(x))].append(i)
        for _T, idxs in by_len.items():
            arr = mx.array(np.stack([ids[i] for i in idxs]))      # [G, T]
            logits = self.model(arr, cache=None)                  # [G, T, V]
            mx.eval(logits)
            npl = _mx_np_f32(mx, logits)                          # [G, T, V] fp32
            for r, i in enumerate(idxs):
                out[i] = torch.from_numpy(np.ascontiguousarray(npl[r]))   # [T, V]
        return out  # type: ignore[return-value]

    def _mlp_preactivation(self, mlp, x):
        if self._mlp_kind == "swiglu":
            return self._swiglu(mlp.gate_proj(x), mlp.up_proj(x))
        return self._gelu(mlp.dense_h_to_4h(x))

    def _mlp_down(self, mlp, pre):
        if self._mlp_kind == "swiglu":
            return mlp.down_proj(pre)
        return mlp.dense_4h_to_h(pre)

    def _down_proj(self, mlp):
        return mlp.down_proj if self._mlp_kind == "swiglu" else mlp.dense_4h_to_h

    def _install_patchers(self, *, patch_ops_by_layer, selected_maps, collect_acts, acts, captured):
        hook_layers = set(patch_ops_by_layer or {}) | set(selected_maps or {})
        if collect_acts:
            hook_layers.update(range(self.n_layer))
        layers = self.model.layers
        id2li = {id(layers[li].mlp): li for li in range(self.n_layer)}
        mlp_type = type(layers[0].mlp)
        original_call = mlp_type.__call__
        eng = self

        def patched(mlp_self, x):
            li = id2li.get(id(mlp_self))
            if li is None or li not in hook_layers:
                return original_call(mlp_self, x)
            pre = eng._mlp_preactivation(mlp_self, x)
            ops = (patch_ops_by_layer or {}).get(li, [])
            if ops:
                pre = eng._apply_mlx_patch_ops(pre, ops)
            if collect_acts:
                acts[li] = torch.from_numpy(_mx_np_f32(eng._mx, pre[0]))
            locals_ = (selected_maps or {}).get(li, {}).get("locals", [])
            if locals_:
                captured[li] = eng._capture_mx_columns(pre, locals_)
            return eng._mlp_down(mlp_self, pre)

        mlp_type.__call__ = patched
        return (mlp_type, original_call)

    def _restore_patchers(self, state) -> None:
        mlp_type, original_call = state
        mlp_type.__call__ = original_call

    def down_weight(self, layer: int) -> torch.Tensor:
        return _dequant_mlx_linear(self._mx, self._down_proj(self.model.layers[int(layer)].mlp))


def open_mlx_engine(model_name: str, quantized: str | None = None, **kwargs: Any) -> _MLXInterventionBase:
    """Resolve the MLX model and dispatch to the Mamba or dense-transformer engine.

    ``quantized="q4g64"`` selects a pre-quantized int4 group-64 export instead of the bf16
    snapshot: weights stay quantized on the Metal GPU and every Linear runs through
    ``mx.quantized_matmul`` (fused dequant, streams ~4.5 bits/weight). Speed backend — parity
    posture is the FC-winner/content-lift gate, not per-position argmax (2026-07-23 PoC:
    0.5B 213 tok/s vs 2.3 paged, FC 23/24 vs fp32; bf16 impl-noise floor is itself ~95%
    argmax with tie-margin misses)."""
    if quantized is not None:
        if quantized != MLX_Q4_SUFFIX:
            raise ValueError(f"quantized={quantized!r} unsupported; only {MLX_Q4_SUFFIX!r}")
        path = _resolve_mlx_q4_path(model_name)
        model_type = json.loads((path / "config.json").read_text()).get("model_type")
        if model_type not in TransformerMLXEngine._SUPPORTED:
            raise ValueError(f"backend='mlx-q4' has no engine for model_type {model_type!r}")
        eng = TransformerMLXEngine(model_name, mlx_path=path, **kwargs)
        eng.backend = "mlx-q4"
        return eng
    path = _resolve_mlx_model_path(model_name)
    model_type = json.loads((path / "config.json").read_text()).get("model_type")
    if model_type == "mamba":
        return MambaMLXEngine(model_name, **kwargs)
    if model_type in TransformerMLXEngine._SUPPORTED:
        return TransformerMLXEngine(model_name, **kwargs)
    raise ValueError(f"backend='mlx' has no engine for model_type {model_type!r}")
