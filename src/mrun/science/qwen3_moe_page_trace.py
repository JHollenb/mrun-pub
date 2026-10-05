"""Deterministic, artifact-only replay of Qwen3-MoE expert-page policies.

The live CUDA runtime records page-cache counters, but comparing a cache policy does not
require another model load once the routed page IDs and their within-acquire frequencies have
been retained.  This module replays that compact metadata under a declared byte/page budget.

It intentionally does *not* claim runtime parity.  The simulator mirrors the current
``ExpertPageCache`` ordering, quota, protection, and transient-prefill rules; a real promotion
still has to prove identical routes, logits/tokens, and resource custody in the native runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TRACE_SCHEMA = "mrun-qwen3-moe-page-trace-v1"
REPORT_SCHEMA = "mrun-qwen3-moe-page-trace-ab-v1"
REQUEST_SCHEMA = "mrun-qwen3-moe-page-policy-request-v1"
MAX_TRACE_FILE_BYTES = 32 * 1024 * 1024
MAX_TRACE_EVENTS = 250_000
MAX_TRACE_PAGE_REQUESTS = 2_000_000

GLOBAL_LRU_CACHE_POLICY = "global-lru-v1"
LAYER_FREQUENCY_CACHE_POLICY = "layer-frequency-lru-v1"
CACHE_FILL_PREFILL_POLICY = "cache-fill-v1"
TRANSIENT_FREQUENCY_PREFILL_POLICY = "transient-frequency-v1"

BASELINE_ARM = "global-lru/cache-fill"
CANDIDATE_ARM = "layer-frequency-lru/transient-frequency"

_PARITY_CAVEAT = (
    "Offline replay is a deterministic policy simulation, not an exact native-runtime parity "
    "result. Promotion requires the live Qwen3-MoE implementation to consume an identical "
    "route/page-sequence digest under both arms and to reproduce exact logits, greedy tokens, "
    "cache reset state, page-binding semantics, and numerical policy. Transfer-only savings "
    "exclude compute, launch, storage, D2D materialization, and overlap effects."
)


class PageTraceError(ValueError):
    """Raised when trace metadata cannot support an exact deterministic replay."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, field_name: str) -> int:
    if not _is_int(value) or int(value) <= 0:
        raise PageTraceError(f"{field_name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, field_name: str) -> int:
    if not _is_int(value) or int(value) < 0:
        raise PageTraceError(f"{field_name} must be a non-negative integer")
    return int(value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _bounded_metadata(value: object, field_name: str) -> dict[str, str | int | float | bool | None]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PageTraceError(f"{field_name} must be a mapping of scalar metadata")
    if len(value) > 32:
        raise PageTraceError(f"{field_name} cannot contain more than 32 keys")
    result: dict[str, str | int | float | bool | None] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if len(key) > 128:
            raise PageTraceError(f"{field_name} keys cannot exceed 128 characters")
        if raw_value is not None and not isinstance(raw_value, (str, int, float, bool)):
            raise PageTraceError(
                f"{field_name}.{key} must be scalar metadata; tensor/list payloads are refused"
            )
        if isinstance(raw_value, str) and len(raw_value) > 4096:
            raise PageTraceError(f"{field_name}.{key} exceeds the 4096-character metadata cap")
        result[key] = raw_value
    return result


@dataclass(frozen=True)
class PageTraceEvent:
    """One layer acquisition; ``pages`` contains ``(expert_id, route_count)`` pairs."""

    phase: Literal["prefill", "decode"]
    step: int
    layer: int
    pages: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise PageTraceError("event phase must be 'prefill' or 'decode'")
        _nonnegative_int(self.step, "event.step")
        _nonnegative_int(self.layer, "event.layer")
        if not self.pages:
            raise PageTraceError("event.pages cannot be empty")
        seen: set[int] = set()
        normalized: list[tuple[int, int]] = []
        for index, item in enumerate(self.pages):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise PageTraceError(f"event.pages[{index}] must be [expert_id, route_count]")
            expert = _nonnegative_int(item[0], f"event.pages[{index}].expert_id")
            count = _positive_int(item[1], f"event.pages[{index}].route_count")
            if expert in seen:
                raise PageTraceError(f"event.pages contains duplicate expert {expert}")
            seen.add(expert)
            normalized.append((expert, count))
        object.__setattr__(self, "pages", tuple(sorted(normalized)))

    @property
    def request_count(self) -> int:
        """Unique page requests, matching the live cache counter."""

        return len(self.pages)

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "step": self.step,
            "layer": self.layer,
            "pages": [[expert, count] for expert, count in self.pages],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PageTraceEvent:
        raw_pages = payload.get("pages")
        if not isinstance(raw_pages, Sequence) or isinstance(raw_pages, (str, bytes)):
            raise PageTraceError("event.pages must be an array")
        return cls(
            phase=str(payload.get("phase")),  # type: ignore[arg-type]
            step=_nonnegative_int(payload.get("step"), "event.step"),
            layer=_nonnegative_int(payload.get("layer"), "event.layer"),
            pages=tuple(raw_pages),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PageAccessTrace:
    """Compact trace custody: page IDs/frequencies and scalar provenance, never page bytes."""

    layer_count: int
    page_bytes: int
    events: tuple[PageTraceEvent, ...]
    source: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    schema: str = TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_SCHEMA:
            raise PageTraceError(f"unsupported page trace schema {self.schema!r}")
        layer_count = _positive_int(self.layer_count, "layer_count")
        _positive_int(self.page_bytes, "page_bytes")
        if not self.events:
            raise PageTraceError("trace.events cannot be empty")
        if len(self.events) > MAX_TRACE_EVENTS:
            raise PageTraceError(
                f"trace.events exceeds the bounded {MAX_TRACE_EVENTS:,}-event replay limit"
            )
        page_requests = 0
        for index, event in enumerate(self.events):
            if not isinstance(event, PageTraceEvent):
                raise PageTraceError(f"trace.events[{index}] is not a PageTraceEvent")
            if event.layer >= layer_count:
                raise PageTraceError(
                    f"trace.events[{index}].layer {event.layer} is outside {layer_count} layers"
                )
            page_requests += event.request_count
            if page_requests > MAX_TRACE_PAGE_REQUESTS:
                raise PageTraceError(
                    "trace page requests exceed the bounded "
                    f"{MAX_TRACE_PAGE_REQUESTS:,}-request replay limit"
                )
        object.__setattr__(self, "source", _bounded_metadata(self.source, "source"))

    @property
    def page_requests(self) -> int:
        return sum(event.request_count for event in self.events)

    @property
    def trace_sha256(self) -> str:
        return _sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "layer_count": self.layer_count,
            "page_bytes": self.page_bytes,
            "source": dict(sorted(self.source.items())),
            "events": [event.as_dict() for event in self.events],
        }
        if include_fingerprint:
            payload["trace_sha256"] = self.trace_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PageAccessTrace:
        schema = str(payload.get("schema"))
        if schema != TRACE_SCHEMA:
            raise PageTraceError(f"unsupported page trace schema {schema!r}")
        raw_events = payload.get("events")
        if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
            raise PageTraceError("trace.events must be an array")
        events: list[PageTraceEvent] = []
        for index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, Mapping):
                raise PageTraceError(f"trace.events[{index}] must be an object")
            events.append(PageTraceEvent.from_dict(raw_event))
        trace = cls(
            schema=schema,
            layer_count=_positive_int(payload.get("layer_count"), "layer_count"),
            page_bytes=_positive_int(payload.get("page_bytes"), "page_bytes"),
            events=tuple(events),
            source=_bounded_metadata(payload.get("source"), "source"),
        )
        claimed = payload.get("trace_sha256")
        if claimed is not None and str(claimed) != trace.trace_sha256:
            raise PageTraceError("trace_sha256 does not match canonical page trace metadata")
        return trace


def load_page_access_trace(path: str | Path) -> PageAccessTrace:
    """Load the bounded JSON trace schema without importing the model runtime."""

    source = Path(path)
    try:
        size = source.stat().st_size
        if size > MAX_TRACE_FILE_BYTES:
            raise PageTraceError(
                f"page trace is {size:,} bytes; bounded loader limit is "
                f"{MAX_TRACE_FILE_BYTES:,} bytes"
            )
        payload = json.loads(source.read_text())
    except PageTraceError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise PageTraceError(f"cannot load page trace {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise PageTraceError("page trace root must be an object")
    return PageAccessTrace.from_dict(payload)


@dataclass(frozen=True)
class PageCacheBudget:
    """Explicit replay capacity. Unallocatable byte remainder is reported, never rounded up."""

    page_bytes: int
    capacity_pages: int
    requested_bytes: int | None = None

    def __post_init__(self) -> None:
        page_bytes = _positive_int(self.page_bytes, "budget.page_bytes")
        pages = _positive_int(self.capacity_pages, "budget.capacity_pages")
        if self.requested_bytes is not None:
            requested = _positive_int(self.requested_bytes, "budget.requested_bytes")
            if requested // page_bytes != pages:
                raise PageTraceError(
                    "budget.capacity_pages must equal floor(requested_bytes / page_bytes)"
                )

    @classmethod
    def from_bytes(cls, *, page_bytes: int, capacity_bytes: int) -> PageCacheBudget:
        stride = _positive_int(page_bytes, "budget.page_bytes")
        requested = _positive_int(capacity_bytes, "budget.capacity_bytes")
        pages = requested // stride
        if pages <= 0:
            raise PageTraceError("budget.capacity_bytes cannot hold one page")
        return cls(page_bytes=stride, capacity_pages=pages, requested_bytes=requested)

    @property
    def allocated_bytes(self) -> int:
        return self.page_bytes * self.capacity_pages

    @property
    def unallocated_bytes(self) -> int:
        if self.requested_bytes is None:
            return 0
        return self.requested_bytes - self.allocated_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "page_bytes": self.page_bytes,
            "capacity_pages": self.capacity_pages,
            "allocated_bytes": self.allocated_bytes,
            "requested_bytes": self.requested_bytes or self.allocated_bytes,
            "unallocated_bytes": self.unallocated_bytes,
        }


@dataclass
class _ReplayStats:
    page_requests: int = 0
    page_hits: int = 0
    page_misses: int = 0
    evictions: int = 0
    same_layer_evictions: int = 0
    over_quota_evictions: int = 0
    fallback_evictions: int = 0
    h2d_bytes: int = 0
    acquisitions: int = 0
    peak_resident_pages: int = 0
    transient_prefill_acquisitions: int = 0
    transient_prefill_pages: int = 0
    transient_prefill_admitted_pages: int = 0
    transient_prefill_replaced_pages: int = 0
    transient_prefill_skipped_protected_pages: int = 0
    decode_protected_promotions: int = 0
    decode_protected_demotions: int = 0


PageKey = tuple[int, int]


class _PolicyReplay:
    def __init__(
        self,
        *,
        layer_count: int,
        budget: PageCacheBudget,
        cache_policy: str,
        prefill_policy: str,
    ) -> None:
        self.budget = budget
        self.cache_policy = cache_policy
        self.prefill_policy = prefill_policy
        self.entries: OrderedDict[PageKey, None] = OrderedDict()
        self.decode_protected: set[PageKey] = set()
        quota, remainder = divmod(budget.capacity_pages, layer_count)
        self.layer_quotas = tuple(
            quota + (1 if layer < remainder else 0) for layer in range(layer_count)
        )
        self.layer_entry_counts = [0] * layer_count
        self.stats = _ReplayStats()

    def _eviction_candidate(
        self, key: PageKey, protected: set[PageKey]
    ) -> tuple[PageKey | None, str]:
        if self.cache_policy == GLOBAL_LRU_CACHE_POLICY:
            return next((item for item in self.entries if item not in protected), None), "fallback"
        incoming_layer = key[0]
        same_layer = next(
            (
                item
                for item in self.entries
                if item[0] == incoming_layer and item not in protected
            ),
            None,
        )
        if (
            same_layer is not None
            and self.layer_entry_counts[incoming_layer] >= self.layer_quotas[incoming_layer]
        ):
            return same_layer, "same-layer"
        over_quota = next(
            (
                item
                for item in self.entries
                if item not in protected
                and self.layer_entry_counts[item[0]] > self.layer_quotas[item[0]]
            ),
            None,
        )
        if over_quota is not None:
            return over_quota, "over-quota"
        return next((item for item in self.entries if item not in protected), None), "fallback"

    def _reserve(self, key: PageKey, protected: set[PageKey]) -> bool:
        if len(self.entries) >= self.budget.capacity_pages:
            victim, reason = self._eviction_candidate(key, protected)
            if victim is None:
                return False
            self.entries.pop(victim)
            if victim in self.decode_protected:
                self.decode_protected.remove(victim)
                self.stats.decode_protected_demotions += 1
            self.layer_entry_counts[victim[0]] -= 1
            self.stats.evictions += 1
            if reason == "same-layer":
                self.stats.same_layer_evictions += 1
            elif reason == "over-quota":
                self.stats.over_quota_evictions += 1
            else:
                self.stats.fallback_evictions += 1
        self.entries[key] = None
        self.layer_entry_counts[key[0]] += 1
        self.stats.peak_resident_pages = max(self.stats.peak_resident_pages, len(self.entries))
        return True

    def _record_requests(self, *, requests: int, hits: int) -> None:
        misses = requests - hits
        self.stats.page_requests += requests
        self.stats.page_hits += hits
        self.stats.page_misses += misses
        self.stats.h2d_bytes += misses * self.budget.page_bytes
        self.stats.acquisitions += 1

    def _acquire_resident(self, event: PageTraceEvent) -> None:
        frequencies = dict(event.pages)
        keys = [(event.layer, expert) for expert in sorted(frequencies)]
        protected = set(keys)
        admission_keys = keys
        if self.cache_policy == LAYER_FREQUENCY_CACHE_POLICY:
            admission_keys = sorted(
                keys,
                key=lambda item: (frequencies[item[1]], item[1]),
            )
        hits = sum(key in self.entries for key in keys)
        for key in admission_keys:
            if key in self.entries:
                self.entries.move_to_end(key)
                continue
            if not self._reserve(key, protected):
                raise PageTraceError(
                    "trace active set cannot fit the declared cache budget under live semantics"
                )
        self._record_requests(requests=len(keys), hits=hits)
        if event.phase == "decode":
            new_protected = set(keys) - self.decode_protected
            self.decode_protected.update(keys)
            self.stats.decode_protected_promotions += len(new_protected)

    def _acquire_transient_prefill(self, event: PageTraceEvent) -> None:
        frequencies = dict(event.pages)
        keys = [(event.layer, expert) for expert in sorted(frequencies)]
        resident = {key for key in keys if key in self.entries}
        protected_layer = {
            key
            for key in self.decode_protected
            if key in self.entries and key[0] == event.layer
        }
        seed_capacity = max(0, self.layer_quotas[event.layer] - len(protected_layer))
        ranked_seed_keys = sorted(
            (key for key in keys if key not in self.decode_protected),
            key=lambda item: (-frequencies[item[1]], item[1]),
        )[:seed_capacity]
        desired_seed_keys = set(ranked_seed_keys)
        stale_seed_keys = [
            key
            for key in self.entries
            if key[0] == event.layer
            and key not in self.decode_protected
            and key not in desired_seed_keys
        ]
        admission_keys = [key for key in ranked_seed_keys if key not in self.entries]
        for key in stale_seed_keys:
            self.entries.pop(key)
            self.layer_entry_counts[event.layer] -= 1

        admitted = 0
        protected = self.decode_protected | set(keys)
        for key in admission_keys:
            if not self._reserve(key, protected):
                break
            admitted += 1

        hits = len(resident)
        self._record_requests(requests=len(keys), hits=hits)
        self.stats.transient_prefill_acquisitions += 1
        self.stats.transient_prefill_pages += len(keys)
        self.stats.transient_prefill_admitted_pages += admitted
        self.stats.transient_prefill_replaced_pages += len(stale_seed_keys)
        self.stats.transient_prefill_skipped_protected_pages += len(admission_keys) - admitted

    def acquire(self, event: PageTraceEvent) -> None:
        if event.request_count > self.budget.capacity_pages:
            raise PageTraceError(
                f"event requests {event.request_count} pages but budget holds only "
                f"{self.budget.capacity_pages}"
            )
        if event.phase == "prefill" and self.prefill_policy == TRANSIENT_FREQUENCY_PREFILL_POLICY:
            self._acquire_transient_prefill(event)
        else:
            self._acquire_resident(event)

    def result(self, *, name: str) -> dict[str, object]:
        stats = self.stats
        if stats.page_hits + stats.page_misses != stats.page_requests:
            raise RuntimeError("simulator counters do not reconcile")
        resident_keys = [[layer, expert] for layer, expert in self.entries]
        return {
            "name": name,
            "cache_policy": self.cache_policy,
            "prefill_page_policy": self.prefill_policy,
            "page_requests": stats.page_requests,
            "page_hits": stats.page_hits,
            "page_misses": stats.page_misses,
            "hit_rate": stats.page_hits / max(1, stats.page_requests),
            "evictions": stats.evictions,
            "same_layer_evictions": stats.same_layer_evictions,
            "over_quota_evictions": stats.over_quota_evictions,
            "fallback_evictions": stats.fallback_evictions,
            "h2d_bytes": stats.h2d_bytes,
            "h2d_gb": stats.h2d_bytes / 1e9,
            "acquisitions": stats.acquisitions,
            "peak_resident_pages": stats.peak_resident_pages,
            "final_resident_pages": len(self.entries),
            "final_resident_sha256": _sha256(resident_keys),
            "layer_quotas": list(self.layer_quotas),
            "transient_prefill_acquisitions": stats.transient_prefill_acquisitions,
            "transient_prefill_pages": stats.transient_prefill_pages,
            "transient_prefill_admitted_pages": stats.transient_prefill_admitted_pages,
            "transient_prefill_replaced_pages": stats.transient_prefill_replaced_pages,
            "transient_prefill_skipped_protected_pages": (
                stats.transient_prefill_skipped_protected_pages
            ),
            "decode_protected_promotions": stats.decode_protected_promotions,
            "decode_protected_demotions": stats.decode_protected_demotions,
        }


def _simulate_arm(
    trace: PageAccessTrace,
    budget: PageCacheBudget,
    *,
    name: str,
    cache_policy: str,
    prefill_policy: str,
) -> dict[str, object]:
    if trace.page_bytes != budget.page_bytes:
        raise PageTraceError(
            f"trace page_bytes={trace.page_bytes} differs from "
            f"budget page_bytes={budget.page_bytes}"
        )
    replay = _PolicyReplay(
        layer_count=trace.layer_count,
        budget=budget,
        cache_policy=cache_policy,
        prefill_policy=prefill_policy,
    )
    for event in trace.events:
        replay.acquire(event)
    return replay.result(name=name)


def replay_page_policy_ab(
    trace: PageAccessTrace,
    budget: PageCacheBudget,
) -> dict[str, object]:
    """Replay both policy arms over the exact same immutable trace and budget."""

    baseline = _simulate_arm(
        trace,
        budget,
        name=BASELINE_ARM,
        cache_policy=GLOBAL_LRU_CACHE_POLICY,
        prefill_policy=CACHE_FILL_PREFILL_POLICY,
    )
    candidate = _simulate_arm(
        trace,
        budget,
        name=CANDIDATE_ARM,
        cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
        prefill_policy=TRANSIENT_FREQUENCY_PREFILL_POLICY,
    )
    baseline_h2d = int(baseline["h2d_bytes"])
    candidate_h2d = int(candidate["h2d_bytes"])
    signed_saved = baseline_h2d - candidate_h2d
    upper_bound_saved = max(0, signed_saved)
    speedup: float | None = None
    if candidate_h2d > 0:
        speedup = baseline_h2d / candidate_h2d
    comparison = {
        "candidate_minus_baseline_hits": int(candidate["page_hits"])
        - int(baseline["page_hits"]),
        "candidate_minus_baseline_misses": int(candidate["page_misses"])
        - int(baseline["page_misses"]),
        "candidate_minus_baseline_evictions": int(candidate["evictions"])
        - int(baseline["evictions"]),
        "signed_h2d_bytes_saved": signed_saved,
        "upper_bound_h2d_bytes_saved": upper_bound_saved,
        "upper_bound_h2d_gb_saved": upper_bound_saved / 1e9,
        "upper_bound_transfer_reduction_fraction": (
            upper_bound_saved / baseline_h2d if baseline_h2d else 0.0
        ),
        "transfer_only_speedup_upper_bound": speedup,
        "candidate_has_lower_h2d": candidate_h2d < baseline_h2d,
    }
    payload: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "trace_sha256": trace.trace_sha256,
        "trace_summary": {
            "events": len(trace.events),
            "page_requests": trace.page_requests,
            "layer_count": trace.layer_count,
            "source": dict(sorted(trace.source.items())),
        },
        "budget": budget.as_dict(),
        "arms": {"baseline": baseline, "candidate": candidate},
        "comparison": comparison,
        "policy_parity_caveat": _PARITY_CAVEAT,
        "claim_status": "offline-working-inference-only",
        "artifact_contract": {
            "retained": "page IDs, route counts, scalar metadata, hashes, and counter diffs",
            "not_retained": "expert page bytes, weights, activations, logits, or KV tensors",
        },
    }
    payload["report_sha256"] = _sha256(payload)
    return payload


def bounded_real_run_request(
    report: Mapping[str, object],
    *,
    model: str = "qwen3-30b-a3b",
) -> dict[str, object]:
    """Emit an immutable request geometry; this function never submits work."""

    if report.get("schema") != REPORT_SCHEMA:
        raise PageTraceError("bounded request requires a page-policy A/B report")
    budget = report.get("budget")
    if not isinstance(budget, Mapping):
        raise PageTraceError("A/B report has no valid budget")
    comparison = report.get("comparison")
    if not isinstance(comparison, Mapping):
        raise PageTraceError("A/B report has no valid comparison")
    candidate_wins = bool(comparison.get("candidate_has_lower_h2d"))
    request: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "submission_authorized": False,
        "source_report_sha256": report.get("report_sha256"),
        "model": model,
        "backend": "qwen3-moe-cuda",
        "scheduler": {
            "launcher": "mrun-intent-first",
            "strict_preflight": True,
            "needs": {"cuda": True},
            "host_preference": "beast",
            "hard_reservation": None,
        },
        "policy_arms": [
            {
                "name": BASELINE_ARM,
                "cache_policy": GLOBAL_LRU_CACHE_POLICY,
                "prefill_page_policy": CACHE_FILL_PREFILL_POLICY,
            },
            {
                "name": CANDIDATE_ARM,
                "cache_policy": LAYER_FREQUENCY_CACHE_POLICY,
                "prefill_page_policy": TRANSIENT_FREQUENCY_PREFILL_POLICY,
            },
        ],
        "winner_for_live_validation": CANDIDATE_ARM if candidate_wins else BASELINE_ARM,
        "cache_budget": dict(budget),
        "physical_geometry": {
            "resident_processes": 1,
            "model_loads": 1,
            "logical_arms": 2,
            "cache_resets": 2,
            "batch_sizes": [1],
            "context_tokens": 4,
            "decode_steps": 1,
            "warmups": 0,
            "measured_repeats": 1,
            "maximum_physical_forwards": 4,
            "teacher_forced_shared_inputs": True,
            "same_process_arm_order": [BASELINE_ARM, CANDIDATE_ARM],
        },
        "stop_conditions": {
            "total_wall_seconds": 840,
            "abort_on_route_digest_mismatch": True,
            "abort_on_nonfinite_logits": True,
            "abort_on_rss_or_vram_preflight_violation": True,
            "abort_on_cache_counter_nonreconciliation": True,
        },
        "required_parity": {
            "route_page_sequence_sha256_equal": True,
            "prefill_logits_sha256_float32_equal": True,
            "decode_logits_sha256_float32_equal": True,
            "greedy_token_sha256_equal": True,
            "cold_cache_before_each_arm": True,
            "same_page_binding_and_route_reduction": True,
        },
        "retention": {
            "keep": [
                "config and code fingerprints",
                "route/page metadata digest",
                "cache counter diffs",
                "logit/token hashes",
                "wall time and RSS/VRAM peaks",
            ],
            "drop": ["weights", "expert page bytes", "activations", "full logits", "KV tensors"],
        },
        "policy_parity_caveat": _PARITY_CAVEAT,
        "queue_note": (
            "One resident Qwen3-MoE load; paired cold-cache B1/C4/D1 policy replay; "
            "metadata-and-diffs only; stop before 15 minutes or on any parity/resource mismatch."
        ),
    }
    request["request_sha256"] = _sha256(request)
    return request


def replay_and_plan(
    trace: PageAccessTrace,
    budget: PageCacheBudget,
    *,
    model: str = "qwen3-30b-a3b",
) -> dict[str, object]:
    """Convenience envelope for an offline report and a non-submitting live request."""

    report = replay_page_policy_ab(trace, budget)
    return {
        "report": report,
        "bounded_real_run_request": bounded_real_run_request(report, model=model),
    }
