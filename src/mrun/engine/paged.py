"""Paged (RAM-decoupling) backend.

Runs a qwen2/llama/gpt_neox/mamba forward by demand-paging weights from a :class:`QStore`.
The default store is row-int8; an explicit ``fp32=True``/``backend="paged-fp32"`` lane preserves
source values losslessly. ``paged-fp16`` and ``paged-bf16`` are explicit aliases for that same
FP32-storage lane with narrowed FP16/BF16 arithmetic; they are not FP16/BF16 weight formats. The
resident weight heap stays O(largest single matrix), so a model of any size on disk runs in bounded
RAM. qwen2/llama fuse a batch through one weight stream
(``batched_paged_logits``, 17.5x at the knee); gpt_neox/mamba fall back to a correct scalar loop.
An offline-resolved linked image can be selected with ``store_path`` or discovered by
``linked_extension_id`` under the normal store root; it remains an ordinary QStore in the forward
loop. A sparse linked extension can instead be bound with
``linked_extension_store_path``: the runtime pre-resolves logical names to one base or extension
provider, with no residual extension math.

Build the store first: ``model-experiments build-store <model>`` (or point
``MODEL_EXPERIMENTS_STORES_ROOT`` at an existing collection).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models import load_tokenizer, resolve_model, store_name
from ..paths import stores_root as default_stores_root
from ._base_impl import BaseEngine, _probe_candidates
from .base import finalize_forced_choice_row, summarize_forced_choice_rows
from .kernels import paged_forward as pf
from .kernels.composite_qstore import ComponentGraphError, CompositeQStore
from .kernels.lexical import (
    LexicalComponent,
    LexicalQStoreView,
    load_separated_lexical,
)
from .kernels.linked_qstore import LinkedQStoreError, ResolvedLinkedQStore
from .kernels.qstore import QStore, QStoreFP32, QStoreInt2, QStoreInt3, QStoreInt4


def _auto_cache_mb() -> float:
    """Resident dequant-weight budget from headroom the guard ALREADY granted.

    The cache is a measured 1.53x on generate (dequant is 79% of a CPU-paged step at
    T=1), but it was default-off because a resident cache changes the RSS profile of a
    run whose reservation was calibrated without it — the shape of I1's job deaths. So
    auto never invents headroom on CPU: it takes half of what is still free below
    RSS_LIMIT_MB (set by the agent executor from the granted reservation), leaving the
    other half for activations and logits. On CUDA, the same cache stores dequantized
    weights on the GPU, so an RSS-derived budget is the wrong resource and can fill a
    16GB card while the model is otherwise safely pageable. CUDA stays pure-streaming
    unless a caller supplies an explicit, measured ``cache_mb``. No limit in the
    environment (unguarded local run) -> 0.
    """
    if torch.cuda.is_available():
        return 0.0
    limit_mb = os.environ.get("RSS_LIMIT_MB", "").strip()
    if not limit_mb:
        return 0.0
    try:
        import psutil

        used_mb = psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001 — psutil absent/unreadable: stay with streaming
        return 0.0
    headroom = float(limit_mb) - used_mb
    return max(0.0, headroom * 0.5)


def _close_failed_store_construction(store: Any) -> None:
    """Release an opened store without replacing the original construction error."""

    try:
        store.close()
    except Exception:  # noqa: BLE001 — construction/admission error remains authoritative
        pass


def _find_linked_store(root: Path, model_name: str, extension_id: str) -> Path:
    """Find one resolved linked image for a base model and logical extension ID."""

    matches: list[Path] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(manifest, dict):
            continue
        linked_image = manifest.get("linked_image")
        if (
            str(manifest.get("model_name", "")) == model_name
            and isinstance(linked_image, Mapping)
            and str(linked_image.get("extension_id", "")) == extension_id
        ):
            matches.append(manifest_path.parent.resolve())
    if not matches:
        raise FileNotFoundError(
            f"no linked QStore image for model {model_name!r} and extension "
            f"{extension_id!r} under {root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple linked QStore images for model {model_name!r} and extension "
            f"{extension_id!r}: {matches}"
        )
    return matches[0]


def _single_flight(method: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize access to one engine's mutable store/cache/ring execution state."""

    @wraps(method)
    def wrapped(self: PagedEngine, *args: Any, **kwargs: Any) -> Any:
        lock = getattr(self, "_execution_lock", None)
        if lock is None:
            # Some narrow kernel tests construct an engine with object.__new__. Production
            # instances always install the lock before opening any store.
            lock = self._execution_lock = threading.RLock()
        with lock:
            return method(self, *args, **kwargs)

    return wrapped


class PagedEngine(BaseEngine):
    backend = "paged"
    stateful_workplan_adapter_abi = "mrun-paged-scratch-only-v1"
    numerical_contract = "paged-qstore-established"
    fp32_numerical_contract = "paged-fp32-source-exact-storage-fp32-arithmetic-v1"
    last_head_numerical_contract = "paged-qstore-last-head-fp32-v1"
    subset_head_numerical_contract = "paged-qstore-subset-head-fp32-v1"
    supported_numerical_contracts = (
        numerical_contract,
        last_head_numerical_contract,
        subset_head_numerical_contract,
    )
    supports_batch = (
        True  # qwen2/llama: ONE weight stream serves B sequences (batched_paged_logits).
    )
    #                         neox/mamba: per-method scalar fallback (still correct, just unfused).

    def __init__(
        self,
        model_name: str,
        *,
        stores_dir: str | Path | None = None,
        store_path: str | Path | None = None,
        linked_extension_id: str | None = None,
        linked_extension_store_path: str | Path | None = None,
        int2: bool = False,
        int4: bool = False,
        int3: bool = False,
        fp32: bool = False,
        device: str | None = None,  # omitted means the legacy arch-gated paged-device policy
        # device (GATHER_DEVICE_PAGED env/flag, arch-gated) — beast-proven CUDA path 8.14x
        compute_dtype: str | None = None,
        cache_mb: float | None = None,  # resident dequant-weight LRU budget;
        #                                 None = auto (grow into granted headroom), 0 = stream
        component_graph: str | Path | None = None,
        lexical_component_path: str | Path | None = None,
        lexical_values_path: str | Path | None = None,
        lexical_weights_path: str | Path | None = None,
        lexical_binding_path: str | Path | None = None,
        output_contract: Any = "full_logits",
        component_cache_mb: Mapping[str, float] | None = None,
        **_ignored: Any,
    ) -> None:
        self._execution_lock = threading.RLock()
        if sum(bool(flag) for flag in (int2, int3, int4, fp32)) > 1:
            raise ValueError("choose at most one store format: fp32, int2, int3, or int4")
        if component_graph is not None and (
            store_path is not None
            or linked_extension_id is not None
            or linked_extension_store_path is not None
        ):
            raise ValueError("component graphs cannot be combined with an explicit store image")
        separated_lexical = (
            lexical_values_path,
            lexical_weights_path,
            lexical_binding_path,
        )
        if any(value is not None for value in separated_lexical) and not all(
            value is not None for value in separated_lexical
        ):
            raise ValueError(
                "lexical_values_path, lexical_weights_path, and lexical_binding_path "
                "must be supplied together"
            )
        if lexical_component_path is not None and any(
            value is not None for value in separated_lexical
        ):
            raise ValueError("combined and separated lexical artifacts are mutually exclusive")
        if component_graph is not None and (
            lexical_component_path is not None
            or any(value is not None for value in separated_lexical)
        ):
            raise ValueError(
                "lexical components currently compose with body stores, not component graphs"
            )
        if store_path is not None and any((int2, int3, int4)):
            raise ValueError("explicit quantized store images currently use the int8 QStore reader")
        if (linked_extension_id is not None or linked_extension_store_path is not None) and any(
            (int2, int3, int4, fp32)
        ):
            raise ValueError("linked store images currently use the int8 QStore reader")
        if linked_extension_store_path is not None and store_path is not None:
            raise ValueError("shared linked stores cannot also set store_path")
        if linked_extension_store_path is not None and linked_extension_id is None:
            raise ValueError("shared linked stores require linked_extension_id")
        if component_graph is not None and any((int2, int3, int4, fp32)):
            raise ValueError("component graphs currently support only the int8 QStore path")
        self.spec = resolve_model(model_name)
        self.name = self.spec.name
        self.store_path: Path | None = None
        self.linked_extension_id: str | None = None
        self.linked_extension_store_path: Path | None = None
        self.composite_store: CompositeQStore | None = None
        self.component_output_contract: str | None = None
        self.lexical_component: LexicalComponent | None = None
        if component_graph is not None:
            if cache_mb is None:
                cache_mb = _auto_cache_mb()
            composite = CompositeQStore(
                component_graph,
                cache_mb=cache_mb,
                component_cache_mb=component_cache_mb,
            )
            try:
                if composite.graph.model_name != self.name:
                    raise ComponentGraphError(
                        "component graph model does not match requested engine "
                        f"({composite.graph.model_name!r} != {self.name!r})"
                    )
                if composite.graph.architecture != self.spec.family:
                    raise ComponentGraphError(
                        "component graph architecture does not match the model registry "
                        f"({composite.graph.architecture!r} != {self.spec.family!r})"
                    )
                store = composite.for_contract(output_contract)
                component_output_contract = store.output_contract
            except BaseException:
                _close_failed_store_construction(composite)
                raise
            self.composite_store = composite
            self.store = store
            self.component_output_contract = component_output_contract
        else:
            explicit_store_path: Path | None = None
            linked_extension_path: Path | None = None
            if linked_extension_store_path is not None:
                linked_extension_path = Path(linked_extension_store_path).expanduser().resolve()
                if linked_extension_path.name == "manifest.json":
                    linked_extension_path = linked_extension_path.parent
                if not linked_extension_path.is_dir():
                    raise FileNotFoundError(
                        f"linked extension QStore is not a directory: {linked_extension_path}"
                    )
                if not (linked_extension_path / "manifest.json").is_file():
                    raise FileNotFoundError(
                        f"linked extension QStore has no manifest.json: {linked_extension_path}"
                    )
            root = Path(stores_dir) if stores_dir is not None else default_stores_root()
            if store_path is not None:
                explicit_store_path = Path(store_path).expanduser().resolve()
                if explicit_store_path.name == "manifest.json":
                    explicit_store_path = explicit_store_path.parent
                if not explicit_store_path.is_dir():
                    raise FileNotFoundError(
                        f"explicit QStore image is not a directory: {explicit_store_path}"
                    )
                if not (explicit_store_path / "manifest.json").is_file():
                    raise FileNotFoundError(
                        f"explicit QStore image has no manifest.json: {explicit_store_path}"
                    )
                root = explicit_store_path.parent
            elif linked_extension_path is None:
                if linked_extension_id is not None:
                    explicit_store_path = _find_linked_store(
                        root,
                        self.spec.name,
                        str(linked_extension_id),
                    )
            suffix = (
                "-fp32"
                if fp32
                else "-int2"
                if int2
                else "-int3"
                if int3
                else "-int4"
                if int4
                else ""
            )
            flag = (
                " --fp32"
                if fp32
                else " --int2"
                if int2
                else " --int3"
                if int3
                else " --int4"
                if int4
                else ""
            )
            # Store-dir candidates, most-canonical first: registry store name, registry key,
            # HF-id tail, and the literal caller-provided name for hand-built stores.
            if explicit_store_path is not None:
                key = explicit_store_path.name
            else:
                candidates = [
                    store_name(self.spec),
                    self.spec.name,
                    self.spec.hf_id.split("/")[-1],
                    str(model_name),
                ]
                key = next(
                    (
                        candidate
                        for candidate in dict.fromkeys(candidates)
                        if (root / f"{candidate}{suffix}" / "manifest.json").exists()
                    ),
                    None,
                )
                if key is None:
                    tried = ", ".join(
                        f"{candidate}{suffix}" for candidate in dict.fromkeys(candidates)
                    )
                    raise FileNotFoundError(
                        f"no paged store under {root} (tried: {tried}). Build it with "
                        f"`mrun build-store {self.spec.name}{flag}` "
                        "or set MRUN_STORES_ROOT to a directory that contains it."
                    )
            # NOTE: key stays unsuffixed; QStoreIntN readers append their own suffix.
            shared_linked = linked_extension_path is not None
            if fp32:
                if cache_mb is None:
                    # The scientific reference lane keeps its advertised O(largest-matrix)
                    # default even inside a generously reserved fleet job. Callers may opt
                    # into a larger resident cache explicitly.
                    cache_mb = 0.0
                self.store = QStoreFP32(
                    key,
                    root=root,
                    cache_mb=cache_mb,
                    explicit_device=device,
                    compute_dtype=compute_dtype or "fp32",
                    suffix="" if explicit_store_path is not None else "-fp32",
                )
            elif int2:
                self.store = QStoreInt2(key, root=root)
            elif int3:
                self.store = QStoreInt3(key, root=root)
            elif int4:
                self.store = QStoreInt4(key, root=root)
            else:
                if cache_mb is None:
                    cache_mb = _auto_cache_mb()
                store_kwargs: dict[str, Any] = {
                    "cache_mb": 0.0 if shared_linked else cache_mb,
                }
                # Keep the established constructor surface for test doubles and older QStore
                # adapters when science did not explicitly request either setting.
                if device is not None:
                    store_kwargs["explicit_device"] = device
                if compute_dtype is not None:
                    store_kwargs["compute_dtype"] = compute_dtype
                self.store = QStore(
                    key,
                    root=root,
                    **store_kwargs,
                )
            if shared_linked:
                extension_store: Any | None = None
                try:
                    extension_store = QStore(
                        linked_extension_path.name,
                        root=linked_extension_path.parent,
                        cache_mb=0.0,
                    )
                    extension_actual_path = Path(extension_store.directory).resolve()
                    if extension_actual_path != linked_extension_path:
                        raise RuntimeError(
                            "QStore opened a different linked extension image than requested: "
                            f"{extension_actual_path} != {linked_extension_path}"
                        )
                    extension_model = str(extension_store.man.get("model_name", ""))
                    if extension_model != self.spec.name:
                        raise ValueError(
                            f"linked extension model {extension_model!r} does not match "
                            f"requested model {self.spec.name!r}"
                        )
                    linked_image = extension_store.man.get("linked_image")
                    if not isinstance(linked_image, Mapping):
                        raise LinkedQStoreError(
                            "linked extension QStore has no linked_image metadata"
                        )
                    actual_extension_id = linked_image.get("extension_id")
                    if str(actual_extension_id) != str(linked_extension_id):
                        raise ValueError(
                            f"linked extension {actual_extension_id!r} does not match "
                            f"requested {linked_extension_id!r}"
                        )
                    overlay_blocks = linked_image.get("overlay_blocks")
                    if not isinstance(overlay_blocks, list) or not overlay_blocks:
                        raise LinkedQStoreError(
                            "linked extension QStore must declare non-empty overlay_blocks"
                        )
                    self.store = ResolvedLinkedQStore(
                        self.store,
                        extension_store,
                        (str(name) for name in overlay_blocks),
                        image_id=str(actual_extension_id),
                        cache_mb=cache_mb,
                    )
                    extension_store = None
                    self.linked_extension_store_path = extension_actual_path
                    self.linked_extension_id = str(actual_extension_id)
                except BaseException:
                    if extension_store is not None:
                        _close_failed_store_construction(extension_store)
                    _close_failed_store_construction(self.store)
                    raise
            if explicit_store_path is not None:
                actual_path = Path(self.store.directory).resolve()
                if actual_path != explicit_store_path:
                    _close_failed_store_construction(self.store)
                    raise RuntimeError(
                        f"QStore opened a different image than requested: "
                        f"{actual_path} != {explicit_store_path}"
                    )

                manifest_model = str(self.store.man.get("model_name", ""))
                if manifest_model != self.spec.name:
                    _close_failed_store_construction(self.store)
                    raise ValueError(
                        f"explicit QStore model {manifest_model!r} does not match "
                        f"requested model {self.spec.name!r}"
                    )
                linked_image = self.store.man.get("linked_image")
                if linked_extension_id is not None:
                    actual_extension_id = (
                        str(linked_image.get("extension_id"))
                        if isinstance(linked_image, Mapping)
                        and linked_image.get("extension_id") is not None
                        else None
                    )
                    if actual_extension_id != str(linked_extension_id):
                        _close_failed_store_construction(self.store)
                        raise ValueError(
                            f"explicit QStore extension {actual_extension_id!r} does not "
                            f"match requested {linked_extension_id!r}"
                        )
                self.store_path = actual_path
                self.linked_extension_id = (
                    str(linked_image.get("extension_id"))
                    if isinstance(linked_image, Mapping)
                    and linked_image.get("extension_id") is not None
                    else None
                )
            else:
                directory = getattr(self.store, "directory", None)
                self.store_path = Path(directory).resolve() if directory is not None else None
                if not shared_linked:
                    self.linked_extension_id = None
        if fp32:
            self.backend = "paged-fp32"
            self.numerical_contract = self.fp32_numerical_contract
            self.last_head_numerical_contract = (
                "paged-fp32-last-head-source-exact-storage-fp32-arithmetic-v1"
            )
            self.subset_head_numerical_contract = (
                "paged-fp32-subset-head-source-exact-storage-fp32-arithmetic-v1"
            )
            self.supported_numerical_contracts = (
                self.numerical_contract,
                self.last_head_numerical_contract,
                self.subset_head_numerical_contract,
            )
        try:
            self.device = self.store.device  # cpu unless the arch-gated cuda flag granted it
            if lexical_component_path is not None:
                self.lexical_component = LexicalComponent.load(
                    lexical_component_path,
                    body_config=self.store.cfg,
                    architecture=str(self.store.man.get("arch", self.spec.family)),
                )
                self.store = LexicalQStoreView(self.store, self.lexical_component)
                self.tokenizer = self.lexical_component.tokenizer
            elif all(value is not None for value in separated_lexical):
                self.lexical_component = load_separated_lexical(
                    values_path=lexical_values_path,
                    weights_path=lexical_weights_path,
                    binding_path=lexical_binding_path,
                    body_config=self.store.cfg,
                    architecture=str(self.store.man.get("arch", self.spec.family)),
                )
                self.store = LexicalQStoreView(self.store, self.lexical_component)
                self.tokenizer = self.lexical_component.tokenizer
            else:
                self.tokenizer = load_tokenizer(self.spec)
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.composite_store is not None:
                self.composite_store.validate_tokenizer(self.tokenizer)
            self.cfg = self.store.cfg
            configured_vocab_rows = int(self.cfg.get("vocab_size", 0))
            if configured_vocab_rows <= 0:
                raise ValueError("model config must declare a positive vocab_size")
            if self.composite_store is not None:
                semantic_token_count = int(self.composite_store.vocab.token_count)
            else:
                try:
                    semantic_token_count = int(len(self.tokenizer))
                except (TypeError, AttributeError) as exc:
                    raise TypeError("model tokenizer must expose its semantic token count") from exc
            if semantic_token_count <= 0 or semantic_token_count > configured_vocab_rows:
                raise ValueError(
                    "semantic token count must be positive and no larger than configured "
                    f"vocabulary rows ({semantic_token_count} > {configured_vocab_rows})"
                )
            self.semantic_token_count = semantic_token_count
            self.arch = self.store.man.get("arch", "qwen2")
            self._paged_logits = {
                "qwen2": pf.paged_logits,
                "llama": pf.paged_logits,
                "qwen3": pf.paged_logits,
                "qwen3_5": pf.paged_logits_qwen35,
                "qwen3_5_text": pf.paged_logits_qwen35,
                "gpt_neox": pf.paged_logits_neox,
                "mamba": pf.paged_logits_mamba,
            }.get(self.arch)
            # qwen3 uses the qwen2/llama forward; it applies q/k-norm when the store has it.
            if self._paged_logits is None:
                raise NotImplementedError(f"paged backend has no forward for arch {self.arch!r}")
            self.n_layer = int(self.cfg["num_hidden_layers"])
            self.inter = int(self.cfg["intermediate_size"])
            self.hidden = int(self.cfg["hidden_size"])
        except BaseException:
            opened_store = (
                self.composite_store
                if self.composite_store is not None
                else getattr(self, "store", None)
            )
            if opened_store is not None:
                _close_failed_store_construction(opened_store)
            raise

    @property
    def working_set_mb(self) -> float:
        # O(largest single dequantized block); 0 until the first forward populates it, so read
        # this AFTER running, not at construction.
        return self.store.max_block_bytes / 1e6

    @working_set_mb.setter
    def working_set_mb(self, _value: Any) -> None:  # BaseEngine sets this attr; paged computes it
        return None

    @_single_flight
    def logits(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
    ) -> torch.Tensor:
        return self._paged_logits(
            self.store,
            np.asarray(ids, np.int64),
            patch_ops_by_layer=patch_ops_by_layer,
        )

    @_single_flight
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
        kv: bool | None = None,
    ) -> list[int] | str:
        """Greedy autoregressive generation via the paged logits loop — one full paged forward per
        new token (weights re-stream each step), so cost is O(new_tokens); fine for short scratchpads
        / CoT. Stops on eos (tokenizer default unless overridden) or any id in ``stop_ids``. Returns
        the NEW token ids (prompt excluded), or the decoded text when ``return_text``. This unblocks
        generate-based legs (scratchpad reasoning, instruction-following) on the paged engine.

        ``cache_mb`` sizes the resident dequant-weight cache for the duration of this call (the
        weights are re-used every token, so a cache that holds the model turns per-token re-dequant
        — ~72-79% of the CPU weight-step — into a one-time cost; bit-exact). Measured 1.5x on
        Qwen2.5-0.5B/8 tok CPU; the remainder is the uncached lm_head stream. None keeps whatever
        budget the store already has.

        ``kv`` (default env ``MRUN_PAGED_KV``, else ON for qwen2/llama/qwen3 — promoted
        2026-07-24) switches to the persistent-KV
        decode path (qwen2/llama/qwen3): prefill once, then each token runs the weight
        stream against one row and streams the lm_head against one row instead of
        replaying the full prefix. Gate: greedy-token parity with the replay path, not
        bit-exact logits (packed-shape reduction-order lesson)."""
        prev_budget = getattr(self.store, "_cache_budget", 0)
        if cache_mb is not None:
            self.store.set_cache_budget(cache_mb)
        # PROMOTED 2026-07-24: kv defaults ON for the supported archs (token-parity gate
        # passed at 0.5B CPU and 4B CUDA; measured x8.4 CPU / x2.46 CUDA at 1,024-prompt).
        # MRUN_PAGED_KV=0 or kv=False restores full-replay. Unsupported archs fall back
        # to replay silently only when kv was not explicitly requested.
        env_kv = os.environ.get("MRUN_PAGED_KV", "").strip()
        if kv is not None:
            use_kv = kv
        elif env_kv:
            use_kv = env_kv != "0"
        else:
            use_kv = self.arch in ("qwen2", "llama", "qwen3")
        try:
            if use_kv:
                if self.arch not in ("qwen2", "llama", "qwen3"):
                    raise NotImplementedError(f"kv generate not implemented for arch {self.arch!r}")
                return self._generate_kv(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    add_special_tokens=add_special_tokens,
                    stop_ids=stop_ids,
                    return_text=return_text,
                )
            return self._generate(
                prompt,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                add_special_tokens=add_special_tokens,
                stop_ids=stop_ids,
                return_text=return_text,
            )
        finally:
            if cache_mb is not None:
                self.store.set_cache_budget(prev_budget / 1e6)

    def _generate(
        self,
        prompt: str | list[int] | np.ndarray,
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        stop_ids: tuple[int, ...] = (),
        return_text: bool = False,
    ) -> list[int] | str:
        if isinstance(prompt, str):
            ids = self.encode([prompt], add_special_tokens=add_special_tokens)[0].tolist()
        else:
            ids = [int(x) for x in np.asarray(prompt).tolist()]
        eos = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.tokenizer, "eos_token_id", None)
        )
        stop = set(stop_ids) | ({eos} if eos is not None else set())
        new: list[int] = []
        for _ in range(int(max_new_tokens)):
            row = self.logits(np.asarray(ids, np.int64))[-1]  # [V] last-position logits
            nxt = int(self._generation_logits(row).argmax().item())
            if nxt in stop:
                break
            ids.append(nxt)
            new.append(nxt)
        if return_text:
            return self.tokenizer.decode(new, skip_special_tokens=True)
        return new

    def _generate_kv(
        self,
        prompt: str | list[int] | np.ndarray,
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        stop_ids: tuple[int, ...] = (),
        return_text: bool = False,
    ) -> list[int] | str:
        if isinstance(prompt, str):
            ids = self.encode([prompt], add_special_tokens=add_special_tokens)[0].tolist()
        else:
            ids = [int(x) for x in np.asarray(prompt).tolist()]
        eos = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.tokenizer, "eos_token_id", None)
        )
        stop = set(stop_ids) | ({eos} if eos is not None else set())
        c = self.cfg
        cache = pf.PagedKVCache(
            self.n_layer,
            int(c["num_key_value_heads"]),
            int(c["head_dim"]),
            capacity=len(ids) + int(max_new_tokens),
            device=self.device,
        )
        row = pf.paged_forward_kv(self.store, np.asarray(ids, np.int64), cache)  # prefill
        new: list[int] = []
        for _ in range(int(max_new_tokens)):
            nxt = int(self._generation_logits(row).argmax().item())
            if nxt in stop:
                break
            new.append(nxt)
            row = pf.paged_forward_kv(self.store, np.asarray([nxt], np.int64), cache)
        if return_text:
            return self.tokenizer.decode(new, skip_special_tokens=True)
        return new

    def _encode_generation_batch(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        add_special_tokens: bool,
    ) -> list[np.ndarray]:
        encoded: list[np.ndarray] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                row = self.encode([prompt], add_special_tokens=add_special_tokens)[0]
            else:
                row = np.asarray(prompt, dtype=np.int64)
            row = np.asarray(row, dtype=np.int64)
            if row.ndim != 1 or not row.size:
                raise ValueError("each prompt must produce a non-empty token row")
            encoded.append(row)
        return encoded

    @_single_flight
    def generate_batch(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        return_text: bool = False,
    ) -> list[list[int]] | list[str]:
        """Greedy-generate a batch while sharing each paged weight traversal.

        Qwen2/3 rows use the existing persistent KV batch kernel. Qwen3.5 rows use their
        separate convolution, recurrent, and full-attention state tensors, with equal-length
        prompt buckets sharing one hybrid-body traversal. Ragged prompts are bucketed and
        restored to caller order; finished rows continue through the state update with EOS so
        the remaining rows can keep using the same batch.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        encoded = self._encode_generation_batch(
            prompts,
            add_special_tokens=add_special_tokens,
        )
        if not encoded:
            return []
        eos = eos_token_id
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        fill_token = int(eos) if eos is not None else 0
        outputs: list[list[int] | None] = [None] * len(encoded)

        if self._can_qwen35_batch():
            by_length: dict[int, list[int]] = {}
            for index, row in enumerate(encoded):
                by_length.setdefault(int(row.size), []).append(index)
            for indices in by_length.values():
                rows = np.stack([encoded[index] for index in indices])
                state = pf.Qwen35PagedState.create(
                    self.store,
                    int(rows.shape[1]) + int(max_new_tokens),
                    batch_size=int(rows.shape[0]),
                )
                logits = pf.paged_logits_qwen35_kv_batch(self.store, rows, state)
                finished = [False] * len(indices)
                generated = [[] for _ in indices]
                for _ in range(int(max_new_tokens)):
                    next_tokens = []
                    for batch_index in range(len(indices)):
                        token = int(self._generation_logits(logits[batch_index]).argmax().item())
                        if not finished[batch_index]:
                            if eos is not None and token == int(eos):
                                finished[batch_index] = True
                            else:
                                generated[batch_index].append(token)
                        next_tokens.append(token if not finished[batch_index] else fill_token)
                    if all(finished):
                        break
                    logits = pf.paged_logits_qwen35_kv_batch(
                        self.store,
                        np.asarray(next_tokens, dtype=np.int64)[:, None],
                        state,
                    )
                for batch_index, output_index in enumerate(indices):
                    outputs[output_index] = generated[batch_index]
        elif self.arch in ("qwen2", "llama", "qwen3"):
            rows = [np.asarray(row, dtype=np.int64) for row in encoded]
            cache = pf.BatchedPagedKVCache(
                self.n_layer,
                len(rows),
                int(self.cfg["num_key_value_heads"]),
                int(self.cfg["head_dim"]),
                capacity=max(len(row) for row in rows) + int(max_new_tokens),
                device=self.device,
            )
            logits = pf.paged_forward_kv_batch(self.store, rows, cache)
            finished = [False] * len(rows)
            generated = [[] for _ in rows]
            for _ in range(int(max_new_tokens)):
                next_tokens = []
                for batch_index in range(len(rows)):
                    token = int(self._generation_logits(logits[batch_index]).argmax().item())
                    if not finished[batch_index]:
                        if eos is not None and token == int(eos):
                            finished[batch_index] = True
                        else:
                            generated[batch_index].append(token)
                    next_tokens.append(token if not finished[batch_index] else fill_token)
                if all(finished):
                    break
                logits = pf.paged_forward_kv_batch(
                    self.store,
                    [np.asarray([token], dtype=np.int64) for token in next_tokens],
                    cache,
                )
            outputs = generated
        else:
            # Preserve a correct API for non-KV paged architectures without claiming a fused
            # batch path in capabilities(). The supported Qwen families take the branches above.
            outputs = [
                self.generate(
                    row,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos,
                    add_special_tokens=False,
                    return_text=False,
                )
                for row in encoded
            ]

        finalized: list[list[int]] = []
        for output in outputs:
            if output is None:
                raise RuntimeError("paged batch generation did not populate every row")
            finalized.append([int(token) for token in output])
        if return_text:
            return [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in finalized]
        return finalized

    def _generation_logits(self, row: Any) -> torch.Tensor:
        """Restrict greedy selection to semantic tokenizer IDs, excluding padded rows."""

        logits = torch.as_tensor(row)
        token_count = int(self.semantic_token_count)
        if logits.ndim != 1 or int(logits.shape[0]) < token_count:
            raise RuntimeError("generation logits do not cover the semantic vocabulary")
        return logits[:token_count]

    @_single_flight
    def forward_acts(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if head_patch_ops_by_layer and self.arch not in ("qwen2", "llama", "qwen3"):
            raise ValueError(
                f"head-output patches are not implemented for paged arch {self.arch!r}"
            )
        acts: list = []
        kwargs: dict[str, Any] = {
            "collect_acts": acts,
            "patch_ops_by_layer": patch_ops_by_layer,
        }
        if self.arch in ("qwen2", "llama", "qwen3"):
            kwargs["head_patch_ops_by_layer"] = head_patch_ops_by_layer
        logits = self._paged_logits(
            self.store,
            np.asarray(ids, np.int64),
            **kwargs,
        )
        return logits, acts  # acts[li]: [T, inter]

    @_single_flight
    def forward_acts_resid(
        self,
        ids: np.ndarray,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Logits + per-layer MLP acts + the residual state after the last block. NOTE: this is the
        PRE-final-norm residual (the paged forward exposes the post-block residual via ``collect_hs``,
        not the post-final-norm hidden HF returns as ``hidden_states[-1]``). Used by the paged
        content-sketch, whose target is documented as the pre-final-norm residual on this backend.
        qwen2/llama/qwen3 only (the collect_hs tap lives in ``paged_logits``)."""
        if self.arch not in ("qwen2", "llama", "qwen3"):
            raise ValueError(f"forward_acts_resid: arch {self.arch!r} has no collect_hs tap")
        acts: list = []
        hs: list = []
        logits = self._paged_logits(
            self.store,
            np.asarray(ids, np.int64),
            collect_acts=acts,
            collect_hs=hs,
        )
        return logits, acts, hs[-1]  # hs[-1]: [T, hidden] pre-final-norm

    @_single_flight
    def hidden_states(self, ids: np.ndarray) -> list[torch.Tensor]:
        """Per-layer residual-stream hidden states in HF ``output_hidden_states`` layout:
        a list of ``nL+1`` tensors ``[T, hidden]`` where ``[0]`` is the embedding output and
        ``[i]`` (i=1..nL) is the residual after decoder layer ``i-1`` — so ``hidden_states[l+1]``
        is "the output of layer ``l``", the exact convention the dense depth-probe indexes
        (capability_signature ``_fit_depth_probe``: ``out.hidden_states[l+1]``).

        INDEXING GOTCHA (load-bearing): the paged ``collect_hs`` tap records the residual
        *pre*-final-norm at every layer, but HF applies ``self.norm`` to the LAST entry only
        (``hidden_states[nL] = norm(residual_after_last_layer)``; entries ``1..nL-1`` stay
        pre-norm). We reproduce HF exactly by RMSNorm-ing the top entry and leaving the rest.
        Skip this and the top-layer probe silently reads a different tensor than the dense
        oracle, shifting probe-R². qwen2/llama/qwen3 only (the collect_hs tap is in that kernel)."""
        if self.arch not in ("qwen2", "llama", "qwen3"):
            raise ValueError(f"hidden_states: arch {self.arch!r} has no collect_hs tap")
        if self.composite_store is not None:
            # The legacy scalar tap continues through the vocabulary head.  The batched
            # hidden return has the same HF layout and exits immediately after final norm.
            return self.hidden_states_batch([np.asarray(ids, np.int64)])[0]
        hs: list = []
        self._paged_logits(self.store, np.asarray(ids, np.int64), collect_hs=hs)
        # hs = [embed, out0, out1, ..., out_{nL-1}] (nL+1 entries, all pre-final-norm).
        # HF normalizes the LAST one; match it so hidden_states[nL] == HF hidden_states[nL].
        eps = self.cfg["rms_norm_eps"]
        w = self.store.fp32("norm.final")
        hs[-1] = pf._rms_norm(hs[-1].float(), w.to(hs[-1].device), eps)  # w may be cuda, hs cpu
        hs = [h.detach().to("cpu") for h in hs]  # numpy-safe, device-uniform
        return hs  # hs[i]: [T, hidden]

    @_single_flight
    def hidden_states_batch(self, ids_list: list[np.ndarray]) -> list[list[torch.Tensor]]:
        """Fuse residual capture through one paged weight stream.

        The batched kernel right-pads with causal/key masks, then this method slices every
        state back to its real token length. Its terminal entry matches scalar
        ``hidden_states``/HF: post-final-norm, not the raw final block residual.
        """

        if not self._can_batch():
            return super().hidden_states_batch(ids_list)
        ids = [np.asarray(row, np.int64) for row in ids_list]
        if not ids:
            return []
        _hidden, lengths, aux = pf.batched_paged_logits(
            self.store,
            ids,
            last_only=True,
            return_hidden=True,
            collect_hidden_states=True,
            return_aux=True,
        )
        state_buffer = aux["hidden_states"]
        return [
            [state[index, : int(lengths[index])].detach().cpu() for state in state_buffer]
            for index in range(len(ids))
        ]

    @_single_flight
    def kv_projections(self, ids: np.ndarray) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Capture raw per-layer K/V projection outputs before reshape, norm, or RoPE."""

        if self.arch not in ("qwen2", "llama", "qwen3"):
            raise ValueError(f"K/V projection capture is unsupported for arch {self.arch!r}")
        prompt = np.asarray(ids, np.int64)
        _hidden, lengths, aux = pf.batched_paged_logits(
            self.store,
            [prompt],
            last_only=True,
            return_hidden=True,
            collect_key_out=True,
            collect_value_out=True,
            return_aux=True,
        )
        length = int(lengths[0])
        return (
            [value[0, :length].detach().cpu() for value in aux["key_out"]],
            [value[0, :length].detach().cpu() for value in aux["value_out"]],
        )

    @_single_flight
    def kv_projections_batch(
        self, ids_list: list[np.ndarray]
    ) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Fuse raw pre-norm/pre-RoPE K/V capture for unequal prompt rows."""

        if self.arch not in ("qwen2", "llama", "qwen3"):
            raise ValueError(
                f"batched K/V projection capture is unsupported for arch {self.arch!r}"
            )
        rows = [np.asarray(ids, np.int64) for ids in ids_list]
        if not rows:
            return []
        _hidden, lengths, aux = pf.batched_paged_logits(
            self.store,
            rows,
            last_only=True,
            return_hidden=True,
            collect_key_out=True,
            collect_value_out=True,
            return_aux=True,
        )
        result = []
        for row_index, length in enumerate(lengths):
            width = int(length)
            result.append(
                (
                    [value[row_index, :width].detach().cpu() for value in aux["key_out"]],
                    [value[row_index, :width].detach().cpu() for value in aux["value_out"]],
                )
            )
        return result

    @_single_flight
    def forward_attns(self, ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Logits + per-layer softmax attention ``[nH, T, T]`` (the matrix the paged forward
        already materializes). Mirrors an HF ``output_attentions`` forward minus the batch dim,
        at O(largest matrix) RAM. Mamba has no softmax attention and raises. int8 weights ⇒
        attention weights are approximate vs fp32 (top-1 preserved)."""
        if self.arch == "mamba":
            raise ValueError("forward_attns: mamba has no softmax attention")
        attn: list = []
        logits = self._paged_logits(self.store, np.asarray(ids, np.int64), collect_attn=attn)
        return logits, attn  # attn[li]: [nH, T, T]

    @_single_flight
    def forward_patched(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        key_patch_ops_by_layer: dict[int, list] | None = None,
        value_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        if head_patch_ops_by_layer and self.arch not in ("qwen2", "llama", "qwen3"):
            raise ValueError(
                f"head-output patches are not implemented for paged arch {self.arch!r}"
            )
        if resid_patch_ops_by_layer and self.arch not in ("qwen2", "llama", "qwen3"):
            raise ValueError(f"resid patches are not implemented for paged arch {self.arch!r}")
        if (key_patch_ops_by_layer or value_patch_ops_by_layer) and self.arch not in (
            "qwen2",
            "llama",
            "qwen3",
        ):
            raise ValueError(f"K/V patches are not implemented for paged arch {self.arch!r}")
        acts: list = []
        captured: dict[int, torch.Tensor] = {}
        kwargs: dict[str, Any] = {
            "collect_acts": acts if collect_acts else None,
            "patch_ops_by_layer": patch_ops_by_layer,
            "capture_selected_maps": selected_maps,
            "captured_selected": captured,
        }
        if self.arch in ("qwen2", "llama", "qwen3"):
            kwargs["head_patch_ops_by_layer"] = head_patch_ops_by_layer
            kwargs["resid_patch_ops_by_layer"] = resid_patch_ops_by_layer
            kwargs["key_patch_ops_by_layer"] = key_patch_ops_by_layer
            kwargs["value_patch_ops_by_layer"] = value_patch_ops_by_layer
        logits = self._paged_logits(
            self.store,
            np.asarray(ids, np.int64),
            **kwargs,
        )
        return logits, acts, captured

    # ---- batched: pay the weight stream ONCE, serve B sequences (qwen2/llama/qwen3) -------
    def _can_batch(self) -> bool:
        return self.arch in ("qwen2", "llama", "qwen3")

    def _can_qwen35_batch(self) -> bool:
        return self.arch in ("qwen3_5", "qwen3_5_text")

    @_single_flight
    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        if self._can_qwen35_batch():
            outputs: list[torch.Tensor | None] = [None] * len(ids_list)
            by_length: dict[int, list[int]] = {}
            rows = [np.asarray(ids, np.int64) for ids in ids_list]
            for index, row in enumerate(rows):
                if row.ndim != 1 or not row.size:
                    raise ValueError("each ids row must be non-empty and one-dimensional")
                by_length.setdefault(int(row.size), []).append(index)
            for indices in by_length.values():
                batch = np.stack([rows[index] for index in indices])
                logits = pf.paged_logits_qwen35_batch(self.store, batch)
                for batch_index, output_index in enumerate(indices):
                    outputs[output_index] = logits[batch_index]
            if any(output is None for output in outputs):
                raise RuntimeError("Qwen3.5 batched logits did not populate every row")
            return [output for output in outputs if output is not None]
        if not self._can_batch():
            return [self.logits(ids) for ids in ids_list]
        ids = [np.asarray(x, np.int64) for x in ids_list]
        out, lengths = pf.batched_paged_logits(self.store, ids, last_only=False)
        return [out[i, : int(lengths[i])] for i in range(len(ids))]  # per-row [T_i, V]

    @_single_flight
    def last_logits_batch(self, ids_list: list[np.ndarray]) -> torch.Tensor:
        """Return ``[B,V]`` without allocating or writing logits for earlier positions."""

        if self._can_qwen35_batch():
            rows = [np.asarray(ids, np.int64) for ids in ids_list]
            if not rows:
                return torch.empty((0, int(self.cfg["vocab_size"])), dtype=torch.float32)
            outputs: list[torch.Tensor | None] = [None] * len(rows)
            by_length: dict[int, list[int]] = {}
            for index, row in enumerate(rows):
                if row.ndim != 1 or not row.size:
                    raise ValueError("each ids row must be non-empty and one-dimensional")
                by_length.setdefault(int(row.size), []).append(index)
            for indices in by_length.values():
                batch = np.stack([rows[index] for index in indices])
                logits = pf.paged_logits_qwen35_batch(self.store, batch, last_only=True)
                for batch_index, output_index in enumerate(indices):
                    outputs[output_index] = logits[batch_index]
            if any(output is None for output in outputs):
                raise RuntimeError("Qwen3.5 batched last logits did not populate every row")
            return torch.stack([output for output in outputs if output is not None])
        if not self._can_batch():
            return torch.stack([self.logits(ids)[-1] for ids in ids_list])
        ids = [np.asarray(x, np.int64) for x in ids_list]
        logits, _lengths = pf.batched_paged_logits(self.store, ids, last_only=True)
        return logits

    @_single_flight
    def selected_last_logits_batch(
        self,
        ids_list: list[np.ndarray],
        token_ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        """Push selected-vocabulary output rows into the paged vocabulary head."""

        if self.composite_store is not None:
            # Reject padded model rows before paying for the body forward.
            self.composite_store.vocab.validate_token_ids(token_ids)
        if not self._can_batch():
            rows = self.logits_batch(ids_list)
            indices = torch.as_tensor(token_ids, dtype=torch.long)
            return torch.stack([row[-1].index_select(0, indices) for row in rows])
        hidden = self.hidden_last_batch(ids_list).float()
        weights = self.lm_head_rows(list(token_ids)).to(hidden.device).float()
        # Reduce every requested row over the hidden dimension independently.  A [B,D]@[D,K]
        # GEMM may select a different blocking strategy when K changes, which made exact-union
        # campaign formation perturb low bits relative to the same row scored in a smaller set.
        # This row-stable formulation keeps each logical selected score independent of union size.
        return (hidden[:, None, :] * weights[None, :, :]).sum(dim=-1).detach().cpu()

    @_single_flight
    def candidate_logits_batch(
        self,
        ids_list: list[np.ndarray],
        candidate_token_ids: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Score per-row candidate sets through one hidden pass and one union head lookup."""

        union = tuple(
            dict.fromkeys(
                token for row_candidates in candidate_token_ids for token in row_candidates
            )
        )
        union_scores = self.selected_last_logits_batch(ids_list, union)
        offsets = {token: index for index, token in enumerate(union)}
        return [
            row_scores.index_select(
                0,
                torch.as_tensor(
                    [offsets[token] for token in row_candidates],
                    dtype=torch.long,
                    device=row_scores.device,
                ),
            )
            for row_scores, row_candidates in zip(
                union_scores,
                candidate_token_ids,
                strict=True,
            )
        ]

    @_single_flight
    def forward_acts_batch(
        self, ids_list: list[np.ndarray]
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        if not self._can_batch():
            return [self.forward_acts(ids) for ids in ids_list]
        ids = [np.asarray(x, np.int64) for x in ids_list]
        out, lengths, acts = pf.batched_paged_logits(
            self.store, ids, last_only=False, collect_acts=True
        )
        res = []
        for i in range(len(ids)):
            Li = int(lengths[i])
            res.append((out[i, :Li], [a[i, :Li] for a in acts]))  # ([T_i,V], acts[li]=[T_i,inter])
        return res

    @_single_flight
    def forward_patched_batch(
        self,
        ids_list: list[np.ndarray],
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        key_patch_ops_by_layer: dict[int, list] | None = None,
        value_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> list[tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]]:
        """Fused batched patched forward (qwen2/llama/qwen3): ONE weight stream applies the shared
        patch map to every row. Non-batchable archs fall back to the scalar loop."""
        if not self._can_batch():
            if key_patch_ops_by_layer or value_patch_ops_by_layer:
                raise ValueError("K/V patches require a batchable Qwen/Llama paged engine")
            return super().forward_patched_batch(
                ids_list,
                patch_ops_by_layer=patch_ops_by_layer,
                head_patch_ops_by_layer=head_patch_ops_by_layer,
                resid_patch_ops_by_layer=resid_patch_ops_by_layer,
                selected_maps=selected_maps,
                collect_acts=collect_acts,
            )
        ids = [np.asarray(x, np.int64) for x in ids_list]
        out, lengths, aux = pf.batched_paged_logits(
            self.store,
            ids,
            last_only=False,
            collect_acts=collect_acts,
            patch_ops_by_layer=patch_ops_by_layer,
            head_patch_ops_by_layer=head_patch_ops_by_layer,
            resid_patch_ops_by_layer=resid_patch_ops_by_layer,
            key_patch_ops_by_layer=key_patch_ops_by_layer,
            value_patch_ops_by_layer=value_patch_ops_by_layer,
            capture_selected_maps=selected_maps,
            return_aux=True,
        )
        acts_buf = aux.get("acts", [])
        cap_buf = aux.get("captured_selected", {})
        res = []
        for i in range(len(ids)):
            Li = int(lengths[i])
            acts_i = [a[i, :Li] for a in acts_buf] if collect_acts else []
            cap_i = {li: cap_buf[li][i, :Li] for li in cap_buf}
            res.append((out[i, :Li], acts_i, cap_i))
        return res

    @_single_flight
    def forward_patched_rows(
        self,
        ids_list: list[np.ndarray],
        patch_ops_by_layer_rows: list[dict[int, list] | None],
    ) -> list[torch.Tensor]:
        """Run distinct MLP intervention branches through one paged weight traversal.

        This is the paged Phase-0 counterpart of :meth:`HFEngine.forward_patched_rows`.
        Every logical branch occupies one batch row and carries an independent replay map;
        an unpatched row is an exact in-batch control.  Non-batchable architectures retain
        the scalar semantic fallback without claiming weight-stream amortization.
        """

        if len(patch_ops_by_layer_rows) != len(ids_list):
            raise ValueError("patch_ops_by_layer_rows must align with ids_list")
        if not self._can_batch():
            self._count_scalar_fallback("forward_patched_rows", len(ids_list))
            return [
                self.forward_patched(ids, patch_ops_by_layer=ops)[0].detach().float().cpu()
                for ids, ops in zip(ids_list, patch_ops_by_layer_rows, strict=True)
            ]
        ids = [np.asarray(value, np.int64) for value in ids_list]
        output, lengths = pf.batched_paged_logits(
            self.store,
            ids,
            last_only=False,
            patch_ops_by_layer_rows=patch_ops_by_layer_rows,
        )
        return [
            output[index, : int(lengths[index])].detach().float().cpu() for index in range(len(ids))
        ]

    @_single_flight
    def selected_last_patched_rows(
        self,
        ids_list: list[np.ndarray],
        patch_ops_by_layer_rows: list[dict[int, list] | None],
        token_ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        """Score one stable output-row union for distinct intervention branches."""

        if len(patch_ops_by_layer_rows) != len(ids_list):
            raise ValueError("patch_ops_by_layer_rows must align with ids_list")
        if not token_ids or len(token_ids) != len(set(token_ids)):
            raise ValueError("token_ids must be a non-empty stable unique union")
        if self.composite_store is not None:
            self.composite_store.vocab.validate_token_ids(token_ids)
        if not self._can_batch():
            rows = self.forward_patched_rows(ids_list, patch_ops_by_layer_rows)
            indices = torch.as_tensor(token_ids, dtype=torch.long)
            return torch.stack([row[-1].index_select(0, indices) for row in rows])
        ids = [np.asarray(value, np.int64) for value in ids_list]
        hidden, _lengths = pf.batched_paged_logits(
            self.store,
            ids,
            last_only=True,
            return_hidden=True,
            patch_ops_by_layer_rows=patch_ops_by_layer_rows,
        )
        weights = self.lm_head_rows(list(token_ids)).to(hidden.device).float()
        return (hidden.float() @ weights.T).detach().cpu()

    @_single_flight
    def selected_last_intervention_branches(
        self,
        prompt_ids: np.ndarray | list[int] | tuple[int, ...],
        *,
        cut_layer: int,
        token_ids: list[int] | tuple[int, ...],
        patch_ops_by_layer_rows: list[dict[int, list] | None],
        head_patch_ops_by_layer_rows: list[dict[int, list] | None],
        resid_patch_ops_by_layer_rows: list[dict[int, list] | None],
        key_patch_ops_by_layer_rows: list[dict[int, list] | None] | None = None,
        value_patch_ops_by_layer_rows: list[dict[int, list] | None] | None = None,
        max_branch_batch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Materialize one exact StateCut, then execute every intervention suffix together."""

        if not self._can_batch():
            raise NotImplementedError(
                "physical Intervention ScienceGraph prefix reuse requires a batchable paged arch"
            )
        branch_count = len(patch_ops_by_layer_rows)
        key_rows = (
            [None] * branch_count
            if key_patch_ops_by_layer_rows is None
            else key_patch_ops_by_layer_rows
        )
        value_rows = (
            [None] * branch_count
            if value_patch_ops_by_layer_rows is None
            else value_patch_ops_by_layer_rows
        )
        if branch_count <= 0 or any(
            len(rows) != branch_count
            for rows in (
                head_patch_ops_by_layer_rows,
                resid_patch_ops_by_layer_rows,
                key_rows,
                value_rows,
            )
        ):
            raise ValueError("all row-local intervention maps must align with the branch pack")
        if isinstance(cut_layer, bool) or not isinstance(cut_layer, int):
            raise TypeError("cut_layer must be an integer")
        if not 0 <= cut_layer < self.n_layer:
            raise ValueError("cut_layer is outside the model")
        if not token_ids or len(token_ids) != len(set(token_ids)):
            raise ValueError("token_ids must be a non-empty stable unique union")
        branch_batch = branch_count if max_branch_batch is None else max_branch_batch
        if isinstance(branch_batch, bool) or not isinstance(branch_batch, int):
            raise TypeError("max_branch_batch must be an integer")
        if branch_batch <= 0:
            raise ValueError("max_branch_batch must be positive")
        if self.composite_store is not None:
            self.composite_store.vocab.validate_token_ids(token_ids)

        prompt = np.asarray(prompt_ids, dtype=np.int64)
        if prompt.ndim != 1 or prompt.size == 0:
            raise ValueError("prompt_ids must be one non-empty token sequence")
        prefix_hidden, _prefix_lengths = pf.batched_paged_logits(
            self.store,
            [prompt],
            last_only=False,
            return_hidden=True,
            stop_layer=cut_layer,
        )
        hidden_chunks = []
        for start in range(0, branch_count, branch_batch):
            end = min(start + branch_batch, branch_count)
            width = end - start
            initial_hidden = prefix_hidden.expand(width, -1, -1).clone()
            hidden, _lengths = pf.batched_paged_logits(
                self.store,
                [prompt] * width,
                last_only=True,
                return_hidden=True,
                initial_hidden=initial_hidden,
                start_layer=cut_layer,
                patch_ops_by_layer_rows=patch_ops_by_layer_rows[start:end],
                head_patch_ops_by_layer_rows=head_patch_ops_by_layer_rows[start:end],
                resid_patch_ops_by_layer_rows=resid_patch_ops_by_layer_rows[start:end],
                key_patch_ops_by_layer_rows=key_rows[start:end],
                value_patch_ops_by_layer_rows=value_rows[start:end],
            )
            hidden_chunks.append(hidden)
        hidden = torch.cat(hidden_chunks, dim=0)
        weights = self.lm_head_rows(list(token_ids)).to(hidden.device).float()
        scores = (hidden.float() @ weights.T).detach().cpu()
        suffix_traversals = len(hidden_chunks)
        return scores, {
            "shared_prefix_materialized": True,
            "cut_layer": cut_layer,
            "prefix_rows": 1,
            "suffix_rows": branch_count,
            "max_branch_batch": branch_batch,
            "suffix_weight_traversals": suffix_traversals,
            "prefix_layer_weight_traversals": cut_layer,
            "suffix_layer_weight_traversals": (self.n_layer - cut_layer) * suffix_traversals,
            "physical_layer_weight_traversals": cut_layer
            + (self.n_layer - cut_layer) * suffix_traversals,
            "independent_layer_weight_traversals": branch_count * self.n_layer,
        }

    def capabilities(self):
        from .base import EngineCapabilities

        qwen_family = self.arch in ("qwen2", "llama", "qwen3")
        qwen35_family = self._can_qwen35_batch()
        generation_family = qwen_family or qwen35_family
        full_output = self.component_output_contract in {None, "full_logits"}
        lossless_fp32 = str(self.store.man.get("dtype")) == "float32"
        return EngineCapabilities(
            logits=full_output,
            logits_batch=(self._can_batch() or qwen35_family) and full_output,
            mlp_acts=full_output,
            mlp_acts_batch=self._can_batch() and full_output,
            mlp_patch=full_output,
            mlp_patch_batch=self._can_batch() and full_output,
            head_patch=qwen_family and full_output,
            head_patch_batch=self._can_batch() and qwen_family and full_output,
            selected_capture=full_output,
            attentions=self.arch != "mamba" and full_output,
            residual_tap=qwen_family and full_output,
            residual_tap_batch=self._can_batch() and qwen_family and full_output,
            residual_patch=qwen_family and full_output,
            raw_model=False,  # no nn.Module — weights stream from the QStore
            autograd=False,
            exact_reference=lossless_fp32,
            approximate_quantized=not lossless_fp32,
            generation=full_output,
            generation_batch=generation_family and full_output,
            persistent_kv=generation_family,
            transactional_kv=qwen_family,
            speculative_blocks=qwen_family,
            # Static WorkPlan memory/lowering descriptors currently name quantized block
            # kinds.  The FP32 lane exposes direct execution and ScienceGraph without
            # pretending those quantized plan contracts apply.
            compiled_workplan=not lossless_fp32,
            graph_replay=False,
            continuous_batching=generation_family,
            int4_execution=str(self.store.man.get("dtype")) == "int4",
            intervention_sciencegraph=qwen_family,
        )

    def runtime_report(self) -> dict[str, Any]:
        """Report the loaded numerical and content identity, including lossless custody."""

        from ..compiler.identity import bind_loaded_qstore_identity

        bound = bind_loaded_qstore_identity(self)
        return {
            "backend": self.backend,
            "model_name": bound.model_name,
            "model_identity": (
                f"{bound.model_name}@{bound.model_revision}#{bound.store_fingerprint}"
            ),
            "numerical_contract": self.numerical_contract,
            "device": str(self.device),
            "store_schema_version": self.store.man.get("schema_version"),
            "store_dtype": self.store.man.get("dtype"),
            "storage_contract": self.store.man.get("storage_contract"),
            "source_checkpoint_sha256": bound.model_revision,
            "store_fingerprint": bound.store_fingerprint,
            "content_identity_verified": bound.content_identity_verified,
            "identity_status": bound.identity_status,
        }

    def continuous_serving_capabilities(self):
        """Return this exact engine instance's fail-closed local serving route.

        Generic :class:`EngineCapabilities` values are intentionally insufficient authority for
        multiwave serving.  The route resolver binds the concrete class, backend, architecture,
        fabric, component view, store identity, and transactional adapter ABI.
        """

        from .continuous import resolve_continuous_serving_capabilities

        return resolve_continuous_serving_capabilities(self)

    @_single_flight
    def prefill_statecut(
        self,
        ids_list: list[np.ndarray],
        *,
        retention_budget_bytes: int,
        capacity: int | None = None,
    ):
        """Prefill and seal one public copy-on-write transactional KV StateCut."""

        from .statecut import PagedKVStateCut

        return PagedKVStateCut.prefill(
            self,
            ids_list,
            retention_budget_bytes=retention_budget_bytes,
            capacity=capacity,
        )

    @_single_flight
    def build_work_plan(self, ids_list: list[np.ndarray], **kwargs: Any):
        """Build a typed static plan for this already-open QStore and request batch."""

        from ..compiler import build_paged_qstore_plan

        return build_paged_qstore_plan(self, ids_list, **kwargs)

    @_single_flight
    def assert_content_identity_unchanged(self) -> None:
        """Fail a warm-pool hit if any already-validated artifact changed in place."""

        self.store.assert_content_identity_unchanged()
        if self.composite_store is not None:
            self.composite_store.validate_tokenizer(self.tokenizer)

    @_single_flight
    def close(self) -> None:
        """Idempotently release ring workers, cached tensors, and component mmaps."""

        if getattr(self, "_closed", False):
            return
        if self.composite_store is not None:
            self.composite_store.close()
        else:
            close_store = getattr(self.store, "close", None)
            if callable(close_store):
                close_store()
        self._closed = True

    @_single_flight
    def composite_snapshot(self) -> dict[str, Any] | None:
        """Return component routing/residency telemetry when this engine is disassembled."""

        if self.composite_store is None:
            return None
        return self.composite_store.snapshot()

    @_single_flight
    def execute_workplan_stateful(
        self,
        plan: Any,
        ids_list: list[np.ndarray],
        state_binding: Any,
    ) -> tuple[Any, pf.PagedKVDelta]:
        """Run one provisional WorkPlan block against explicitly versioned batched KV.

        This method never commits.  The compiler wraps the returned delta with the plan and
        handle identities; callers must explicitly accept per-row token counts through its
        commit helper.  Exceptions therefore leave committed lengths, epoch, K, and V intact.
        """

        if not self._can_batch():
            raise NotImplementedError(
                f"stateful WorkPlan execution is unavailable for paged arch {self.arch!r}"
            )
        cache = getattr(state_binding, "state", None)
        if not isinstance(cache, pf.BatchedPagedKVCache):
            raise TypeError("state binding must hold a BatchedPagedKVCache")
        if int(getattr(state_binding, "parent_epoch", -1)) != int(cache.epoch):
            raise RuntimeError("state binding epoch is stale")
        bound_lengths = tuple(int(value) for value in getattr(state_binding, "parent_lengths", ()))
        current_lengths = tuple(int(value) for value in cache.lengths)
        if bound_lengths != current_lengths:
            raise RuntimeError("state binding lengths are stale")

        rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
        if len(rows) != cache.B or not rows:
            raise ValueError("stateful WorkPlan batch does not match its KV arena")
        token_count = int(rows[0].size)
        if token_count <= 0 or any(row.ndim != 1 or int(row.size) != token_count for row in rows):
            raise ValueError("stateful WorkPlan rows must be nonempty and equal length")
        tokens = np.stack(rows)
        contract = str(getattr(getattr(plan, "output_contract", ""), "value", ""))
        graph_contract = {
            "full_logits": "full_logits",
            "last_token_logits": "full_logits",
            "selected_token_rows": "selected_rows",
            "candidate_argmax_and_margin": "selected_rows",
            "hidden_state_only": "lexical_hidden",
        }.get(contract)
        if graph_contract is None:
            raise NotImplementedError(f"stateful output contract {contract!r} is unavailable")
        if (
            self.component_output_contract is not None
            and self.component_output_contract != graph_contract
        ):
            raise ComponentGraphError(
                "component engine is bound to a different output contract "
                f"({self.component_output_contract!r} != {graph_contract!r})"
            )

        selected_rows: tuple[int, ...] = ()
        if contract == "selected_token_rows":
            selected_rows = tuple(int(value) for value in plan.required_output_rows)
            kernel_contract = "selected_token_rows"
        elif contract == "candidate_argmax_and_margin":
            selected_rows = tuple(
                dict.fromkeys(
                    int(token)
                    for row_candidates in plan.candidate_token_ids
                    for token in row_candidates
                )
            )
            kernel_contract = "selected_token_rows"
        elif contract == "hidden_state_only":
            kernel_contract = "hidden_state_only"
        else:
            kernel_contract = "full_logits"

        if self.composite_store is not None and selected_rows:
            self.composite_store.vocab.validate_token_ids(selected_rows)

        output, delta = pf.paged_forward_block(
            self.store,
            tokens,
            cache,
            output_contract=kernel_contract,
            selected_rows=selected_rows,
            last_only=contract
            in {
                "last_token_logits",
                "selected_token_rows",
                "candidate_argmax_and_margin",
            },
        )
        if contract == "full_logits":
            contract_output: Any = output
        elif contract == "last_token_logits":
            contract_output = output
        elif contract == "selected_token_rows":
            contract_output = output
        elif contract == "hidden_state_only":
            contract_output = output
        elif contract == "candidate_argmax_and_margin":
            from ..compiler.executable import candidate_outputs_from_values

            offsets = {token: index for index, token in enumerate(selected_rows)}
            values = [
                output[row_index].index_select(
                    0,
                    torch.as_tensor(
                        [offsets[int(token)] for token in row_candidates],
                        dtype=torch.long,
                    ),
                )
                for row_index, row_candidates in enumerate(plan.candidate_token_ids)
            ]
            contract_output = candidate_outputs_from_values(
                values,
                plan.candidate_token_ids,
            )
        else:
            raise AssertionError(f"unhandled stateful output contract {contract!r}")
        return contract_output, delta

    @_single_flight
    def down_weight(self, layer: int) -> torch.Tensor:
        return self.store.weight(f"L{int(layer)}.down").float().cpu()

    # ---- subset-lm_head scoring: never stream the 544 MB unembedding -----------------------
    @_single_flight
    def hidden_last_batch(self, ids_list: list[np.ndarray]) -> torch.Tensor:
        """Final hidden state at each last real token ``[B, d]``, WITHOUT streaming lm_head.
        The working set stays at the largest MLP block (~17.4 MB on Qwen2.5-0.5B), below the
        29.4 MB lm_head-row-block floor. qwen2/llama batched body only."""
        if not self._can_batch():
            raise NotImplementedError("hidden_last_batch: batched body is qwen2/llama only")
        ids = [np.asarray(x, np.int64) for x in ids_list]
        h_last, _ = pf.batched_paged_logits(self.store, ids, last_only=True, return_hidden=True)
        return h_last  # [B, d]

    @_single_flight
    def lm_head_rows(self, token_ids: list[int] | np.ndarray) -> torch.Tensor:
        """Dequantize only the requested output rows of lm_head ``[k, d]`` (no full-matrix stream)."""
        return self.store.embed_rows("lm_head", np.asarray(token_ids, np.int64))  # [k, d]

    @_single_flight
    def score_forced_choice_argmax_subset(self, probes: list[dict[str, Any]]) -> dict[str, Any]:
        """Single-token forced choice via a candidate-only lm_head subset.

        For single-token answers the full-vocab softmax denominator ``logZ`` at the last prompt
        position is shared by all of a probe's candidates, so it CANCELS: the winner and the
        margin are EXACT from the raw candidate logits ``h_last @ lm_head[cands]ᵀ``. Only the
        absolute ``avg_logprob`` is unnormalized (the shared ``-logZ`` is dropped). Never streams
        the 544 MB lm_head ⇒ lower RAM floor. Raises if any candidate is multi-token (the caller
        should fall back to :meth:`score_forced_choice_many`, where Z does not cancel)."""
        prompts: list[np.ndarray] = []
        layout: list[tuple[list[int], list[str], str, Any, int]] = []
        prompt_offsets: dict[tuple[int, ...], int] = {}
        for idx, probe in enumerate(probes):
            prompt_ids = self.encode([str(probe["prompt"])], add_special_tokens=False)[0].tolist()
            cand_tokens, answers = [], []
            for cand in _probe_candidates(probe):
                answer = str(cand["answer"])
                aid = self.encode([answer], add_special_tokens=False)[0].tolist()
                if len(aid) != 1:
                    raise ValueError(
                        f"score_forced_choice_argmax_subset needs single-token candidates; "
                        f"{answer!r} is {len(aid)} tokens"
                    )
                cand_tokens.append(int(aid[0]))
                answers.append(answer)
            prompt_key = tuple(int(value) for value in prompt_ids)
            prompt_offset = prompt_offsets.get(prompt_key)
            if prompt_offset is None:
                prompt_offset = len(prompts)
                prompt_offsets[prompt_key] = prompt_offset
                prompts.append(np.asarray(prompt_ids, dtype=np.int64))
            layout.append(
                (
                    cand_tokens,
                    answers,
                    str(probe.get("probe_id", idx)),
                    probe.get("category"),
                    prompt_offset,
                )
            )

        union = tuple(dict.fromkeys(token for candidates, *_rest in layout for token in candidates))
        union_offsets = {token: offset for offset, token in enumerate(union)}
        union_scores = self.selected_last_logits_batch(prompts, union)
        rows = []
        for cand_tokens, answers, pid, cat, prompt_offset in layout:
            logits = union_scores[
                prompt_offset,
                [union_offsets[token] for token in cand_tokens],
            ]
            scored = [
                {"answer": answers[j], "avg_logprob": float(logits[j])} for j in range(len(answers))
            ]
            rows.append(finalize_forced_choice_row(scored, probe_id=pid, category=cat))
        return {
            "summary": summarize_forced_choice_rows(rows),
            "rows": rows,
            "note": "subset lm_head: winner+margin exact (shared -logZ cancels); avg_logprob unnormalized",
            "automatic_campaign": {
                "eligible": len(layout),
                "fallback": 0,
                "logical_prompt_evaluations": len(layout),
                "physical_prompt_evaluations": len(prompts),
                "candidate_references": sum(len(item[0]) for item in layout),
                "candidate_union_rows": len(union),
            },
        }
