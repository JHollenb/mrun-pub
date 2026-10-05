"""Adapters from established mrun engines to WorkPlan v3."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from .identity import bind_loaded_qstore_identity
from .ir import (
    CaptureCapability,
    CaptureContract,
    ExecutionMode,
    OutputContract,
    TypedCaptureSpec,
    build_dense_work_plan,
)

_DENSE_PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


def _strict_strings(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of strings")
    normalized = tuple(values)
    if any(type(value) is not str for value in normalized):
        raise TypeError(f"{field} must contain only strings")
    return normalized


def _strict_integers(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values,
        (Sequence, np.ndarray),
    ):
        raise TypeError(f"{field} must be a non-string integer sequence")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise TypeError(f"{field} must be a one-dimensional integer sequence")
    normalized = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in normalized):
        raise TypeError(f"{field} must contain only integers")
    return tuple(int(value) for value in normalized)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _page_sequence(engine: Any, output_contract: OutputContract) -> tuple[str, ...]:
    store = engine.store
    # ``embed`` is a logical ingress page even when it aliases the physical vocabulary
    # allocation.  Keeping it in the plan is what lets lowering distinguish physical
    # sharing from semantic use.
    pages: list[str] = ["embed"]
    for layer in range(int(engine.n_layer)):
        for projection in _DENSE_PROJECTIONS:
            name = f"L{layer}.{projection}"
            if store.has(name):
                pages.append(name)
    if store.has("norm.final"):
        pages.append("norm.final")
    if output_contract not in {
        OutputContract.HIDDEN_STATE_ONLY,
        OutputContract.SELECTED_CAPTURE,
    }:
        pages.append("lm_head")
    return tuple(pages)


def _head_access_metadata(
    manifest: Mapping[str, Any],
    output_contract: OutputContract,
    required_output_rows: Sequence[int],
    candidate_token_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    config = manifest.get("config", {})
    vocab_size = int(config.get("vocab_size", 0)) if isinstance(config, Mapping) else 0
    if output_contract in {
        OutputContract.HIDDEN_STATE_ONLY,
        OutputContract.SELECTED_CAPTURE,
    }:
        access = "none"
        row_count = 0
    elif output_contract in {
        OutputContract.SELECTED_TOKEN_ROWS,
        OutputContract.SELECTED_CAPTURE,
    }:
        access = "rows"
        row_count = len(required_output_rows)
    elif output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        access = "rows"
        row_count = len({int(token) for row in candidate_token_ids for token in row})
    else:
        access = "all"
        row_count = vocab_size
    return {
        "logical_head_access": access,
        "logical_head_row_count": row_count,
        "configured_output_row_count": vocab_size,
        "logical_hidden_width": (
            int(config.get("hidden_size", 0)) if isinstance(config, Mapping) else 0
        ),
    }


def _kv_layout_metadata(
    manifest: Mapping[str, Any],
    *,
    capacity: int | None,
    dtype: str,
) -> dict[str, Any]:
    if capacity is None:
        return {}
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("stateful QStore planning requires a model config")
    attention_heads = int(config.get("num_attention_heads", 0))
    hidden_size = int(config.get("hidden_size", 0))
    values = {
        "kv_capacity": int(capacity),
        "kv_dtype": str(dtype),
        "kv_num_layers": int(config.get("num_hidden_layers", 0)),
        "kv_num_heads": int(config.get("num_key_value_heads", attention_heads)),
        "kv_head_dim": int(config.get("head_dim", hidden_size // max(attention_heads, 1))),
    }
    if (
        min(
            values["kv_capacity"],
            values["kv_num_layers"],
            values["kv_num_heads"],
            values["kv_head_dim"],
        )
        <= 0
    ):
        raise ValueError("stateful QStore planning requires a positive KV layout")
    return values


def _require_stateful_execution_contract(engine: Any, mode: ExecutionMode) -> None:
    if mode not in {ExecutionMode.PREFILL, ExecutionMode.DECODE}:
        return
    execute = getattr(engine, "execute_workplan_stateful", None)
    if not callable(execute):
        raise NotImplementedError(
            f"{getattr(engine, 'backend', 'unknown')} has no stateful WorkPlan executor"
        )
    capabilities = getattr(engine, "capabilities", None)
    if not callable(capabilities):
        raise NotImplementedError(
            "stateful WorkPlan engines must explicitly advertise transactional capabilities"
        )
    advertised = capabilities()
    if (
        getattr(advertised, "transactional_kv", None) is not True
        or getattr(advertised, "speculative_blocks", None) is not True
    ):
        raise NotImplementedError(
            "engine does not advertise transactional provisional-block execution"
        )


def _paged_residency_contract(engine: Any) -> tuple[str, dict[str, Any]]:
    composite = getattr(engine, "composite_store", None)
    if composite is not None:
        snapshot_method = getattr(composite, "snapshot", None)
        snapshot = snapshot_method() if callable(snapshot_method) else {}
        if not isinstance(snapshot, Mapping):
            raise TypeError("CompositeQStore snapshot must be an object")
        providers = getattr(composite, "_providers", {})
        if not isinstance(providers, Mapping) or "body" not in providers:
            raise ValueError("CompositeQStore must expose a body provider")
        raw_budgets = snapshot.get("component_cache_budget_bytes")
        if isinstance(raw_budgets, Mapping):
            budgets = {str(role): int(value) for role, value in raw_budgets.items()}
        else:
            budgets = {
                str(role): max(0, int(getattr(provider.store, "_cache_budget", 0)))
                for role, provider in providers.items()
            }
        body = providers["body"].store
        policy = str(getattr(body, "_cache_policy", "unknown"))
        ring_allocated = getattr(composite, "ring_allocated_bytes", None)
        ring_bytes = int(ring_allocated()) if callable(ring_allocated) else 0
        total_budget = snapshot.get("cache_budget_bytes", sum(budgets.values()))
        if isinstance(total_budget, bool) or not isinstance(total_budget, Integral):
            raise TypeError("CompositeQStore cache budget must be an integer")
        return (
            f"component-aggregate-{policy}",
            {
                "weight_cache_budget_bytes": max(0, int(total_budget)),
                "weight_cache_policy": policy,
                "component_cache_budgets_json": json.dumps(
                    budgets,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "ring_staging_bytes": ring_bytes,
            },
        )
    store = engine.store
    policy = str(getattr(store, "_cache_policy", "unknown"))
    ring_allocated = getattr(store, "ring_allocated_bytes", None)
    ring_bytes = int(ring_allocated()) if callable(ring_allocated) else 0
    return (
        policy,
        {
            "weight_cache_budget_bytes": max(0, int(getattr(store, "_cache_budget", 0))),
            "weight_cache_policy": policy,
            "component_cache_budgets_json": "",
            "ring_staging_bytes": ring_bytes,
        },
    )


def _component_metadata(engine: Any, output_contract: OutputContract) -> dict[str, Any]:
    composite = getattr(engine, "composite_store", None)
    actual = getattr(engine, "component_output_contract", None)
    if composite is None and actual is None:
        return {}
    if composite is None or actual is None:
        raise ValueError("component engines must expose both store and output contract")
    expected = {
        OutputContract.FULL_LOGITS: "full_logits",
        OutputContract.LAST_TOKEN_LOGITS: "full_logits",
        OutputContract.LOSS_ONLY: "full_logits",
        OutputContract.SELECTED_TOKEN_ROWS: "selected_rows",
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN: "selected_rows",
        OutputContract.HIDDEN_STATE_ONLY: "lexical_hidden",
        OutputContract.SELECTED_CAPTURE: "selected_capture",
    }[output_contract]
    if str(actual) != expected:
        raise ValueError(
            "component engine output contract does not match the WorkPlan "
            f"({actual!r} != {expected!r})"
        )
    metadata: dict[str, Any] = {
        "component_output_contract": expected,
        "composite_qstore": True,
    }
    graph_fingerprint = getattr(
        composite,
        "composite_fingerprint_sha256",
        getattr(composite, "composite_graph_fingerprint", None),
    )
    vocab_fingerprint = getattr(composite, "vocab_manifest_sha256", None)
    if graph_fingerprint is None or vocab_fingerprint is None:
        raise ValueError("CompositeQStore identity is missing graph or vocabulary fingerprint")
    metadata["component_graph_fingerprint"] = str(graph_fingerprint)
    metadata["vocab_manifest_sha256"] = str(vocab_fingerprint)
    return metadata


def _request_state_fields(
    *,
    batch_size: int,
    sequence_length: int,
    execution_mode: ExecutionMode | str,
    request_ids: Sequence[str] | None,
    request_slots: Sequence[int] | None,
    prefix_state_ids: Sequence[str],
    kv_read_handles: Sequence[str],
    kv_write_handles: Sequence[str],
    kv_capacity: int | None,
) -> tuple[
    ExecutionMode,
    tuple[str, ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    int | None,
]:
    mode = (
        execution_mode
        if isinstance(execution_mode, ExecutionMode)
        else ExecutionMode(str(execution_mode))
    )
    normalized_ids = (
        tuple(f"request-{index}" for index in range(batch_size))
        if request_ids is None
        else _strict_strings(request_ids, field="request_ids")
    )
    normalized_slots = (
        tuple(range(batch_size))
        if request_slots is None
        else _strict_integers(request_slots, field="request_slots")
    )
    reads = _strict_strings(kv_read_handles, field="kv_read_handles")
    writes = _strict_strings(kv_write_handles, field="kv_write_handles")
    capacity = kv_capacity
    if mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}:
        if not reads and not writes:
            reads = tuple(f"kv:{request_id}" for request_id in normalized_ids)
            writes = reads
        if capacity is None:
            raise ValueError("prefill/decode WorkPlans require an explicit kv_capacity")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("kv_capacity must be an integer")
        if capacity < sequence_length:
            raise ValueError("kv_capacity cannot be smaller than the planned token block")
    elif capacity is not None:
        raise ValueError("kv_capacity is only legal for prefill/decode WorkPlans")
    return (
        mode,
        normalized_ids,
        normalized_slots,
        _strict_strings(prefix_state_ids, field="prefix_state_ids"),
        reads,
        writes,
        capacity,
    )


def _qstore_context(
    engine: Any,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    expected_backend: str,
    max_seq_len: int | None = None,
) -> tuple[list[np.ndarray], int, dict[str, Any], dict[str, str], int]:
    if getattr(engine, "backend", None) != expected_backend:
        raise ValueError(f"QStore planning requires backend={expected_backend!r}")
    raw_rows = [np.asarray(ids) for ids in ids_list]
    if any(row.dtype.kind not in {"i", "u"} for row in raw_rows):
        raise TypeError("input token IDs must be integer values, not coerced floats or booleans")
    rows = [row.astype(np.int64, copy=False) for row in raw_rows]
    if not rows:
        raise ValueError("ids_list must contain at least one row")
    if any(row.ndim != 1 or not row.size for row in rows):
        raise ValueError("every input row must be a non-empty one-dimensional token array")
    lengths = {int(row.size) for row in rows}
    if len(lengths) != 1:
        raise ValueError("one static WorkPlan requires equal input lengths")
    sequence_length = next(iter(lengths))
    if max_seq_len is not None and sequence_length > max_seq_len:
        raise ValueError("input length exceeds the engine max_seq_len")

    manifest = engine.store.man
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("QStore planning requires a model config")
    composite = getattr(engine, "composite_store", None)
    semantic_vocab = getattr(getattr(composite, "vocab", None), "token_count", None)
    engine_semantic_vocab = getattr(engine, "semantic_token_count", None)
    token_limit_value = (
        engine_semantic_vocab
        if engine_semantic_vocab is not None
        else semantic_vocab
        if semantic_vocab is not None
        else config.get("vocab_size", 0)
    )
    if isinstance(token_limit_value, bool) or not isinstance(token_limit_value, (int, np.integer)):
        raise TypeError("QStore planning requires an integer semantic token limit")
    token_limit = int(token_limit_value)
    if token_limit <= 0:
        raise ValueError("QStore planning requires a positive semantic token limit")
    if any(row.size and ((row < 0).any() or (row >= token_limit).any()) for row in rows):
        raise ValueError(f"input token IDs must be inside [0, {token_limit})")
    bound = bind_loaded_qstore_identity(engine)
    identity = {
        "model_name": bound.model_name,
        "model_revision": bound.model_revision,
        "store_fingerprint": bound.store_fingerprint,
        "source_identity_status": bound.source_identity_status,
        "store_identity_status": bound.store_identity_status,
        "identity_status": bound.identity_status,
        "identity_certificate_sha256": bound.identity_certificate_sha256,
        "manifest_semantic_sha256": bound.manifest_semantic_sha256,
        "builder_source_bundle_sha256": bound.builder_source_bundle_sha256,
        "blob_records_sha256": bound.blob_records_sha256,
        "blob_identity_verified": bound.content_identity_verified,
    }
    return rows, sequence_length, manifest, identity, token_limit


def _activation_dtype(engine: Any) -> str:
    compute_dtype = str(getattr(engine.store, "compute_dtype", "float32")).removeprefix("torch.")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
        compute_dtype,
        compute_dtype,
    )


def _validate_vocab_rows(
    manifest: dict[str, Any],
    required_output_rows: Sequence[int],
    candidate_token_ids: Sequence[Sequence[int]],
    *,
    token_limit: int,
) -> None:
    config = manifest.get("config", {})
    vocab_size = int(config.get("vocab_size", 0)) if isinstance(config, dict) else 0
    required = _strict_integers(required_output_rows, field="required_output_rows")
    if isinstance(candidate_token_ids, (str, bytes, bytearray)) or not isinstance(
        candidate_token_ids,
        (Sequence, np.ndarray),
    ):
        raise TypeError("candidate_token_ids must be a non-string sequence of integer sequences")
    if isinstance(candidate_token_ids, np.ndarray) and candidate_token_ids.ndim != 2:
        raise TypeError("candidate_token_ids must be a two-dimensional integer sequence")
    candidate_rows = tuple(
        _strict_integers(row, field=f"candidate_token_ids[{index}]")
        for index, row in enumerate(candidate_token_ids)
    )
    selected = [
        *required,
        *(token for row in candidate_rows for token in row),
    ]
    if any(token < 0 or token >= token_limit for token in selected):
        raise ValueError(f"output token IDs must be inside semantic token space [0, {token_limit})")
    if vocab_size and token_limit > vocab_size:
        raise ValueError("semantic token space cannot exceed configured vocabulary rows")


def build_dense_qstore_plan(
    engine: Any,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    execution_mode: ExecutionMode | str = ExecutionMode.SCORE,
    output_contract: OutputContract | str = OutputContract.LAST_TOKEN_LOGITS,
    request_ids: Sequence[str] | None = None,
    request_slots: Sequence[int] | None = None,
    prefix_state_ids: Sequence[str] = (),
    kv_read_handles: Sequence[str] = (),
    kv_write_handles: Sequence[str] = (),
    kv_capacity: int | None = None,
    required_output_rows: Sequence[int] = (),
    candidate_token_ids: Sequence[Sequence[int]] = (),
    batch_bucket: int | None = None,
    capture_requested: bool = False,
    stable_addresses: bool = False,
    graph_safe: bool = False,
    capture_specs: Sequence[TypedCaptureSpec] = (),
    capture_capability: CaptureCapability | None = None,
) -> Any:
    """Build a static dense-QStore plan without running or loading additional model state."""

    rows, sequence_length, manifest, identity, token_limit = _qstore_context(
        engine,
        ids_list,
        expected_backend="dense-qstore-cuda",
        max_seq_len=int(engine.max_seq_len),
    )
    _validate_vocab_rows(
        manifest,
        required_output_rows,
        candidate_token_ids,
        token_limit=token_limit,
    )
    contract = (
        output_contract
        if isinstance(output_contract, OutputContract)
        else OutputContract(str(output_contract))
    )
    (
        mode,
        normalized_request_ids,
        normalized_request_slots,
        normalized_prefix_ids,
        normalized_reads,
        normalized_writes,
        normalized_capacity,
    ) = _request_state_fields(
        batch_size=len(rows),
        sequence_length=sequence_length,
        execution_mode=execution_mode,
        request_ids=request_ids,
        request_slots=request_slots,
        prefix_state_ids=prefix_state_ids,
        kv_read_handles=kv_read_handles,
        kv_write_handles=kv_write_handles,
        kv_capacity=kv_capacity,
    )
    _require_stateful_execution_contract(engine, mode)
    pushdown_method = {
        OutputContract.SELECTED_TOKEN_ROWS: "selected_last_logits_batch",
        OutputContract.SELECTED_CAPTURE: (
            None if capture_capability is None else capture_capability.runtime_entrypoint
        ),
    }.get(contract)
    output_pushdown = pushdown_method is not None and callable(
        getattr(engine, pushdown_method, None)
    )
    numerical_contract = str(engine.numerical_contract)
    if output_pushdown and contract is OutputContract.SELECTED_TOKEN_ROWS:
        numerical_contract = str(
            getattr(engine, "subset_head_numerical_contract", numerical_contract)
        )
    return build_dense_work_plan(
        model_name=identity["model_name"],
        model_revision=identity["model_revision"],
        store_fingerprint=identity["store_fingerprint"],
        batch_size=len(rows),
        sequence_length=sequence_length,
        batch_bucket=batch_bucket or _next_power_of_two(len(rows)),
        execution_mode=mode,
        output_contract=contract,
        numerical_contract=numerical_contract,
        activation_dtype=_activation_dtype(engine),
        weight_dtype=str(manifest.get("dtype", "int8")),
        accumulator_dtype="fp32",
        request_ids=normalized_request_ids,
        request_slots=normalized_request_slots,
        prefix_state_ids=normalized_prefix_ids,
        kv_read_handles=normalized_reads,
        kv_write_handles=normalized_writes,
        required_output_rows=required_output_rows,
        candidate_token_ids=candidate_token_ids,
        page_sequence=_page_sequence(engine, contract),
        cache_admission="compact-device-lru",
        compute_layout_ids=("qrow-w8a16-fixed-tile",),
        structured_operator_ids=(
            "dense-transformer",
            "persistent-kv",
            *(("transactional-kv",) if normalized_capacity is not None else ()),
        ),
        capture=CaptureContract(
            requested=capture_requested,
            static_shapes=True,
            stable_addresses=stable_addresses,
            graph_safe=graph_safe,
        ),
        capture_specs=capture_specs,
        capture_capability=capture_capability,
        metadata={
            "engine_backend": str(engine.backend),
            "engine_device": str(getattr(engine, "device", "cuda")),
            "head_output_pushdown": (
                output_pushdown and contract is OutputContract.SELECTED_TOKEN_ROWS
            ),
            "output_pushdown": output_pushdown,
            "max_seq_len": int(engine.max_seq_len),
            **_kv_layout_metadata(
                manifest,
                capacity=normalized_capacity,
                dtype=_activation_dtype(engine),
            ),
            "input_token_limit": token_limit,
            **_head_access_metadata(
                manifest,
                contract,
                required_output_rows,
                candidate_token_ids,
            ),
            **_component_metadata(engine, contract),
            "source_identity_status": identity["source_identity_status"],
            "store_identity_status": identity["store_identity_status"],
            "identity_certificate_sha256": identity["identity_certificate_sha256"],
            "manifest_semantic_sha256": identity["manifest_semantic_sha256"],
            "builder_source_bundle_sha256": identity["builder_source_bundle_sha256"],
            "blob_records_sha256": identity["blob_records_sha256"],
            "blob_identity_verified": identity["blob_identity_verified"],
        },
    )


def build_paged_qstore_plan(
    engine: Any,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    execution_mode: ExecutionMode | str = ExecutionMode.SCORE,
    output_contract: OutputContract | str = OutputContract.LAST_TOKEN_LOGITS,
    request_ids: Sequence[str] | None = None,
    request_slots: Sequence[int] | None = None,
    prefix_state_ids: Sequence[str] = (),
    kv_read_handles: Sequence[str] = (),
    kv_write_handles: Sequence[str] = (),
    kv_capacity: int | None = None,
    required_output_rows: Sequence[int] = (),
    candidate_token_ids: Sequence[Sequence[int]] = (),
    batch_bucket: int | None = None,
    capture_requested: bool = False,
    capture_specs: Sequence[TypedCaptureSpec] = (),
    capture_capability: CaptureCapability | None = None,
) -> Any:
    """Build an executable static plan for the hardware-independent paged QStore engine."""

    rows, sequence_length, manifest, identity, token_limit = _qstore_context(
        engine,
        ids_list,
        expected_backend="paged",
    )
    _validate_vocab_rows(
        manifest,
        required_output_rows,
        candidate_token_ids,
        token_limit=token_limit,
    )
    contract = (
        output_contract
        if isinstance(output_contract, OutputContract)
        else OutputContract(str(output_contract))
    )
    (
        mode,
        normalized_request_ids,
        normalized_request_slots,
        normalized_prefix_ids,
        normalized_reads,
        normalized_writes,
        normalized_capacity,
    ) = _request_state_fields(
        batch_size=len(rows),
        sequence_length=sequence_length,
        execution_mode=execution_mode,
        request_ids=request_ids,
        request_slots=request_slots,
        prefix_state_ids=prefix_state_ids,
        kv_read_handles=kv_read_handles,
        kv_write_handles=kv_write_handles,
        kv_capacity=kv_capacity,
    )
    _require_stateful_execution_contract(engine, mode)
    pushdown_method = {
        OutputContract.LAST_TOKEN_LOGITS: "last_logits_batch",
        OutputContract.SELECTED_TOKEN_ROWS: "selected_last_logits_batch",
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN: "candidate_logits_batch",
        OutputContract.HIDDEN_STATE_ONLY: "hidden_states_batch",
        OutputContract.SELECTED_CAPTURE: (
            None if capture_capability is None else capture_capability.runtime_entrypoint
        ),
    }.get(contract)
    output_pushdown = pushdown_method is not None and callable(
        getattr(engine, pushdown_method, None)
    )
    numerical_contract = str(engine.numerical_contract)
    if output_pushdown and contract is OutputContract.LAST_TOKEN_LOGITS:
        numerical_contract = str(
            getattr(engine, "last_head_numerical_contract", numerical_contract)
        )
    elif output_pushdown and contract in {
        OutputContract.SELECTED_TOKEN_ROWS,
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
    }:
        numerical_contract = str(
            getattr(engine, "subset_head_numerical_contract", numerical_contract)
        )
    cache_admission, residency_metadata = _paged_residency_contract(engine)
    return build_dense_work_plan(
        model_name=identity["model_name"],
        model_revision=identity["model_revision"],
        store_fingerprint=identity["store_fingerprint"],
        batch_size=len(rows),
        sequence_length=sequence_length,
        batch_bucket=batch_bucket or _next_power_of_two(len(rows)),
        execution_mode=mode,
        output_contract=contract,
        numerical_contract=numerical_contract,
        activation_dtype=_activation_dtype(engine),
        weight_dtype=str(manifest.get("dtype", "int8")),
        accumulator_dtype="fp32",
        request_ids=normalized_request_ids,
        request_slots=normalized_request_slots,
        prefix_state_ids=normalized_prefix_ids,
        kv_read_handles=normalized_reads,
        kv_write_handles=normalized_writes,
        required_output_rows=required_output_rows,
        candidate_token_ids=candidate_token_ids,
        page_sequence=_page_sequence(engine, contract),
        cache_admission=cache_admission,
        compute_layout_ids=(f"qrow-{manifest.get('dtype', 'int8')}-rowwise",),
        structured_operator_ids=(
            "paged-transformer",
            "fused-batch-weight-stream",
            *(("persistent-kv", "transactional-kv") if normalized_capacity is not None else ()),
        ),
        capture=CaptureContract(
            requested=capture_requested,
            static_shapes=True,
            stable_addresses=False,
            graph_safe=False,
        ),
        capture_specs=capture_specs,
        capture_capability=capture_capability,
        metadata={
            "engine_backend": str(engine.backend),
            "engine_device": str(getattr(engine, "device", "cpu")),
            "head_output_pushdown": (
                output_pushdown
                and contract
                in {
                    OutputContract.LAST_TOKEN_LOGITS,
                    OutputContract.SELECTED_TOKEN_ROWS,
                    OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
                }
            ),
            "output_pushdown": output_pushdown,
            **residency_metadata,
            **_kv_layout_metadata(
                manifest,
                capacity=normalized_capacity,
                # BatchedPagedKVCache and its provisional delta are deliberately
                # fp32, independent of the QStore compute dtype. Admission charges
                # the physical arena rather than inferring from activations.
                dtype="fp32",
            ),
            "input_token_limit": token_limit,
            **_head_access_metadata(
                manifest,
                contract,
                required_output_rows,
                candidate_token_ids,
            ),
            **_component_metadata(engine, contract),
            "source_identity_status": identity["source_identity_status"],
            "store_identity_status": identity["store_identity_status"],
            "identity_certificate_sha256": identity["identity_certificate_sha256"],
            "manifest_semantic_sha256": identity["manifest_semantic_sha256"],
            "builder_source_bundle_sha256": identity["builder_source_bundle_sha256"],
            "blob_records_sha256": identity["blob_records_sha256"],
            "blob_identity_verified": identity["blob_identity_verified"],
        },
    )
