"""Manifest-derived resource planning for dense WorkPlans.

The estimates are deterministic planning numbers, not allocator telemetry. Runtime
measurements stay separate so an estimate can never be mistaken for observed residency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from .ir import DenseWorkPlan, ExecutionMode, OutputContract

MEMORY_PLAN_SCHEMA = "mrun-memory-plan-v2"

_DTYPE_BYTES = {
    "fp32": 4,
    "fp32-class": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
    "int4": 0.5,
    "int3": 0.375,
    "int2": 0.25,
}


def _whole_bytes(elements: int, dtype: str) -> int:
    return int(elements * _DTYPE_BYTES[dtype] + 0.999999)


def resolve_manifest_block(manifest: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    """Resolve aliases with cycle detection and return the physical block name/payload."""

    blocks = manifest.get("blocks", {})
    if not isinstance(blocks, dict):
        raise TypeError("manifest blocks must be an object")
    seen: set[str] = set()
    current = name
    while True:
        if current in seen:
            raise ValueError(f"manifest alias cycle at {current!r}")
        seen.add(current)
        raw = blocks.get(current)
        if not isinstance(raw, dict):
            raise KeyError(f"manifest has no block {current!r}")
        alias = raw.get("alias")
        if alias is None:
            return current, raw
        current = str(alias)


def _block_regions(block: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    regions: list[tuple[str, int, int]] = []
    for channel in ("w", "s", "e"):
        length = int(block.get(f"{channel}_len", 0))
        if not length and channel in {"w", "s"}:
            shape = block.get("shape")
            output_rows = (
                int(shape[0])
                if isinstance(shape, list)
                and shape
                and str(block.get("kind", "")).startswith("qrow")
                else 0
            )
            if channel == "w" and output_rows and block.get("row_bytes") is not None:
                length = output_rows * int(block["row_bytes"])
            elif channel == "s" and output_rows and block.get("n_groups") is not None:
                length = output_rows * int(block["n_groups"]) * 4
        if length:
            regions.append((channel, int(block.get(f"{channel}_off", 0)), length))
    return tuple(regions)


def _regions_bytes(
    manifest: dict[str, Any],
    names: tuple[str, ...] | None = None,
) -> tuple[int, tuple[str, ...]]:
    blocks = manifest.get("blocks", {})
    selected = tuple(blocks) if names is None else names
    regions: set[tuple[str, int, int]] = set()
    missing: list[str] = []
    for name in selected:
        try:
            _, block = resolve_manifest_block(manifest, name)
        except KeyError:
            missing.append(name)
            continue
        regions.update(_block_regions(block))
    return sum(length for _, _, length in regions), tuple(sorted(set(missing)))


@dataclass(frozen=True)
class MemoryPlan:
    durable_store_bytes: int
    scheduled_page_bytes: int
    estimated_compact_read_bytes: int
    peak_promoted_weight_bytes: int
    kv_allocated_bytes: int
    output_bytes: int
    workspace_bytes: int
    resident_weight_cache_bytes: int
    ring_staging_bytes: int
    estimated_peak_active_bytes: int
    scheduled_page_count: int
    missing_pages: tuple[str, ...]
    estimate_basis: str = "manifest-derived-not-measured"
    schema_version: str = MEMORY_PLAN_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "durable_store_bytes": self.durable_store_bytes,
            "scheduled_page_bytes": self.scheduled_page_bytes,
            "estimated_compact_read_bytes": self.estimated_compact_read_bytes,
            "peak_promoted_weight_bytes": self.peak_promoted_weight_bytes,
            "kv_allocated_bytes": self.kv_allocated_bytes,
            "output_bytes": self.output_bytes,
            "workspace_bytes": self.workspace_bytes,
            "resident_weight_cache_bytes": self.resident_weight_cache_bytes,
            "ring_staging_bytes": self.ring_staging_bytes,
            "estimated_peak_active_bytes": self.estimated_peak_active_bytes,
            "scheduled_page_count": self.scheduled_page_count,
            "missing_pages": list(self.missing_pages),
            "estimate_basis": self.estimate_basis,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryPlan:
        schema = str(payload.get("schema_version", MEMORY_PLAN_SCHEMA))
        if schema != MEMORY_PLAN_SCHEMA:
            raise ValueError(f"unsupported memory-plan schema: {schema}")
        return cls(
            durable_store_bytes=int(payload["durable_store_bytes"]),
            scheduled_page_bytes=int(payload["scheduled_page_bytes"]),
            estimated_compact_read_bytes=int(
                payload.get("estimated_compact_read_bytes", payload["scheduled_page_bytes"])
            ),
            peak_promoted_weight_bytes=int(payload["peak_promoted_weight_bytes"]),
            kv_allocated_bytes=int(payload["kv_allocated_bytes"]),
            output_bytes=int(payload["output_bytes"]),
            workspace_bytes=int(payload["workspace_bytes"]),
            resident_weight_cache_bytes=int(payload.get("resident_weight_cache_bytes", 0)),
            ring_staging_bytes=int(payload.get("ring_staging_bytes", 0)),
            estimated_peak_active_bytes=int(payload["estimated_peak_active_bytes"]),
            scheduled_page_count=int(payload["scheduled_page_count"]),
            missing_pages=tuple(str(value) for value in payload.get("missing_pages", ())),
            estimate_basis=str(payload.get("estimate_basis", "manifest-derived-not-measured")),
            schema_version=schema,
        )


def plan_qstore_memory(plan: DenseWorkPlan, manifest: dict[str, Any]) -> MemoryPlan:
    """Derive compact storage, promotion, KV, output, and workspace byte budgets."""

    durable_bytes, _ = _regions_bytes(manifest)
    scheduled_bytes, missing = _regions_bytes(manifest, plan.page_sequence)
    non_head_pages = tuple(name for name in plan.page_sequence if name != "lm_head")
    compact_read_bytes, _ = _regions_bytes(manifest, non_head_pages)
    metadata = dict(plan.metadata)
    if "embed" in non_head_pages:
        _, embed = resolve_manifest_block(manifest, "embed")
        embed_shape = embed.get("shape", ())
        embed_rows = int(embed_shape[0]) if isinstance(embed_shape, list) and embed_shape else 0
        if embed_rows > 0:
            embed_regions = sum(length for _, _, length in _block_regions(embed))
            addressed_rows = plan.shape.actual_batch * plan.shape.sequence_length
            compact_read_bytes -= embed_regions
            compact_read_bytes += (embed_regions * addressed_rows + embed_rows - 1) // embed_rows
    head_output_pushdown = bool(metadata.get("head_output_pushdown", False))
    if "lm_head" in plan.page_sequence:
        _, head = resolve_manifest_block(manifest, "lm_head")
        head_regions = sum(length for _, _, length in _block_regions(head))
        head_shape = head.get("shape", ())
        head_rows = int(head_shape[0]) if isinstance(head_shape, list) and head_shape else 0
        if plan.output_contract in {
            OutputContract.FULL_LOGITS,
            OutputContract.LAST_TOKEN_LOGITS,
            OutputContract.LOSS_ONLY,
        }:
            selected_head_rows = head_rows
        elif head_output_pushdown and plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
            selected_head_rows = len(plan.required_output_rows)
        elif (
            head_output_pushdown
            and plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
        ):
            selected_head_rows = len(
                {token for row_candidates in plan.candidate_token_ids for token in row_candidates}
            )
        else:
            selected_head_rows = 0
        if head_rows:
            compact_read_bytes += (head_regions * selected_head_rows + head_rows - 1) // head_rows
    activation_bytes = int(_DTYPE_BYTES[plan.precision.activation_dtype])
    paged_streaming = "paged-transformer" in plan.structured_operator_ids
    quantized_matmul_rows = int(manifest.get("matmul_chunk_rows", 1024))
    peak_promoted = 0
    for name in plan.page_sequence:
        try:
            _, block = resolve_manifest_block(manifest, name)
        except KeyError:
            continue
        shape = block.get("shape")
        if str(block.get("kind", "")).startswith("qrow") and isinstance(shape, list):
            promoted_shape = [int(value) for value in shape]
            if paged_streaming and name == "lm_head":
                if (
                    head_output_pushdown
                    and plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS
                ):
                    promoted_shape[0] = min(
                        promoted_shape[0],
                        len(plan.required_output_rows),
                    )
                elif (
                    head_output_pushdown
                    and plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
                ):
                    promoted_shape[0] = min(
                        promoted_shape[0],
                        len(
                            {
                                token
                                for row_candidates in plan.candidate_token_ids
                                for token in row_candidates
                            }
                        ),
                    )
                else:
                    promoted_shape[0] = min(promoted_shape[0], 8192)
            elif paged_streaming and plan.precision.weight_dtype in {"int2", "int3"}:
                promoted_shape[0] = min(promoted_shape[0], quantized_matmul_rows)
            peak_promoted = max(
                peak_promoted,
                _whole_bytes(prod(promoted_shape), plan.precision.activation_dtype),
            )

    config = manifest.get("config", {})
    if not isinstance(config, dict):
        raise TypeError("manifest config must be an object")
    layers = int(config.get("num_hidden_layers", 0))
    hidden = int(config.get("hidden_size", 0))
    heads = int(config.get("num_attention_heads", 1))
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim", hidden // max(heads, 1)))
    vocab = int(config.get("vocab_size", 0))
    inter = int(config.get("intermediate_size", hidden))

    kv_bytes = 0
    stateful = plan.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}
    if stateful or "persistent-kv" in plan.structured_operator_ids:
        kv_capacity = int(metadata["kv_capacity"]) if stateful else plan.shape.sequence_bucket
        if stateful:
            expected_layout = {
                "kv_num_layers": layers,
                "kv_num_heads": kv_heads,
                "kv_head_dim": head_dim,
            }
            for key, expected_value in expected_layout.items():
                if int(metadata.get(key, -1)) != expected_value:
                    raise ValueError(f"stateful {key} metadata disagrees with the manifest")
            kv_dtype = metadata.get("kv_dtype")
            if not isinstance(kv_dtype, str) or kv_dtype not in _DTYPE_BYTES:
                raise ValueError("stateful memory planning requires supported kv_dtype metadata")
            kv_elements = layers * plan.shape.batch_bucket * kv_capacity * kv_heads * head_dim * 2
            kv_bytes = _whole_bytes(kv_elements, kv_dtype)
        else:
            kv_elements = layers * plan.shape.batch_bucket * kv_capacity * kv_heads * head_dim * 2
            kv_bytes = kv_elements * activation_bytes

    if plan.output_contract is OutputContract.FULL_LOGITS:
        output_elements = plan.shape.actual_batch * plan.shape.sequence_length * vocab
    elif plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
        output_elements = plan.shape.actual_batch * vocab
    elif plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
        output_elements = plan.shape.actual_batch * len(plan.required_output_rows)
    elif plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        candidate_union = {
            token for row_candidates in plan.candidate_token_ids for token in row_candidates
        }
        # Current paged pushdown projects one global union for every row before producing
        # row-local winner/margin summaries. Charge that real intermediate, not merely the
        # number of requested row-local candidates.
        output_elements = plan.shape.actual_batch * len(candidate_union)
    elif plan.output_contract is OutputContract.LOSS_ONLY:
        output_elements = 1
    elif plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
        output_elements = plan.shape.actual_batch * plan.shape.sequence_length * hidden
    elif plan.output_contract is OutputContract.SELECTED_CAPTURE:
        output_elements = 0
    else:
        output_elements = 0
    output_bytes = (
        sum(spec.max_retained_bytes for spec in plan.capture_specs)
        if plan.output_contract is OutputContract.SELECTED_CAPTURE
        else output_elements * 4
    )

    live_rows = plan.shape.batch_bucket * plan.shape.sequence_bucket
    workspace_elements = live_rows * max(hidden * 3, inter * 2)
    workspace_bytes = workspace_elements * activation_bytes
    if plan.output_contract is OutputContract.SELECTED_CAPTURE:
        # Specs declare pre-load upper bounds. Charge their union conservatively rather than
        # assuming reductions serialize, so admission cannot hide a capture-tape RAM spike.
        workspace_bytes += sum(spec.resource_estimate.device_bytes for spec in plan.capture_specs)
    if stateful:
        kv_dtype = str(metadata["kv_dtype"])
        provisional_elements = (
            layers * plan.shape.batch_bucket * plan.shape.sequence_bucket * kv_heads * head_dim * 2
        )
        workspace_bytes += _whole_bytes(provisional_elements, kv_dtype)
        if paged_streaming:
            # `paged_forward_block` builds a capacity-wide committed+provisional view,
            # optionally expands GQA heads, and holds scores plus softmax probabilities.
            # Charge the bucket/capacity ceiling; this is admission, not observed peak RSS.
            base_kv_elements = plan.shape.batch_bucket * kv_capacity * kv_heads * head_dim * 2
            workspace_bytes += base_kv_elements * 4
            if heads > kv_heads:
                expanded_kv_elements = plan.shape.batch_bucket * kv_capacity * heads * head_dim * 2
                workspace_bytes += expanded_kv_elements * 4
            attention_elements = (
                plan.shape.batch_bucket * plan.shape.sequence_bucket * kv_capacity * (2 * heads + 1)
            )
            workspace_bytes += attention_elements * 4
    if not head_output_pushdown and plan.output_contract in {
        OutputContract.LAST_TOKEN_LOGITS,
        OutputContract.SELECTED_TOKEN_ROWS,
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
    }:
        workspace_bytes += live_rows * vocab * 4
    resident_weight_cache_bytes = int(metadata.get("weight_cache_budget_bytes", 0))
    ring_staging_bytes = int(metadata.get("ring_staging_bytes", 0))
    if resident_weight_cache_bytes < 0 or ring_staging_bytes < 0:
        raise ValueError("residency budgets cannot be negative")
    estimated_peak = (
        peak_promoted
        + kv_bytes
        + output_bytes
        + workspace_bytes
        + resident_weight_cache_bytes
        + ring_staging_bytes
    )
    return MemoryPlan(
        durable_store_bytes=durable_bytes,
        scheduled_page_bytes=scheduled_bytes,
        estimated_compact_read_bytes=compact_read_bytes,
        peak_promoted_weight_bytes=peak_promoted,
        kv_allocated_bytes=kv_bytes,
        output_bytes=output_bytes,
        workspace_bytes=workspace_bytes,
        resident_weight_cache_bytes=resident_weight_cache_bytes,
        ring_staging_bytes=ring_staging_bytes,
        estimated_peak_active_bytes=estimated_peak,
        scheduled_page_count=len(plan.page_sequence),
        missing_pages=missing,
    )
