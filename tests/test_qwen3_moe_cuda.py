from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mrun.engine import open_engine
from mrun.engine.kernels.qstore_int4 import _quant_row_int4
from mrun.engine.qwen3_moe_cuda import (
    DEFAULT_ROUTE_REDUCTION_POLICY,
    DEFAULT_W4_ARITHMETIC_POLICY,
    EXPERT_CODEC_W4,
    LAYER_FREQUENCY_CACHE_POLICY,
    PAGE_TRACE_HARD_MAX_EVENTS,
    PAGE_TRACE_SCHEMA,
    SLOT_INDIRECT_PAGE_BINDING_POLICY,
    STABLE_ROUTE_REDUCTION_POLICY,
    TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY,
    W4_GROUPED_LAUNCH_CONFIG,
    W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
    W4_PREDOT_BF16_ARITHMETIC_POLICY,
    W4_STORE_CODEC,
    W4_STORE_FILE,
    W4_STORE_SCHEMA,
    ExpertPageCache,
    ExpertPageLayout,
    Int4ExpertPageLayout,
    LayerWeights,
    PackedFP8ExpertStore,
    PackedInt4ExpertStore,
    PagedFP8ExpertBackend,
    PagedInt4ExpertBackend,
    Qwen3MoeCudaEngine,
    Qwen3MoeDecodeRuntime,
    Qwen3MoePageTraceLimitError,
    TensorReader,
    _compact_route,
    _compact_route_with_inverse,
    _grouped_fp8_shape_contract,
    _grouped_w4_shape_contract,
    _load_skeleton,
    _native_sdpa_attention,
    _verify_compatible_promoted_builder,
    _w4_postscale_launch_config,
    build_int4_expert_store,
    dequantize_block_fp8,
    dequantize_int4_weight,
    expert_page_views,
    grouped_fp8_mm_strided,
    grouped_w4_mm_strided,
    int4_expert_page_views,
    normalize_w4_arithmetic_policy,
    quantize_fp8_weight,
    quantize_int4_weight,
    resolve_qwen3_moe_store_dir,
    stable_route_reduce,
    validate_fp8_expert_store,
    validate_int4_expert_store,
)
from mrun.store_provenance import BUILDER_SCHEMA, canonical_sha256


def _write_tiny_store(
    root: Path,
    *,
    layers: int = 2,
    experts: int = 3,
) -> Path:
    layout = ExpertPageLayout.create(hidden_size=4, intermediate_size=2)
    pages = np.empty((layers, experts, layout.page_stride), dtype=np.uint8)
    page_pattern = np.arange(layout.page_stride, dtype=np.uint8)
    for global_page, page in enumerate(pages.reshape(-1, layout.page_stride)):
        page[:] = page_pattern + np.uint8((global_page * 17) % 256)
    store = root / "tiny-store"
    store.mkdir()
    (store / "experts.fp8").write_bytes(pages.tobytes())
    (store / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "qwen3-moe-packed-expert-store-v1",
                "codec": "rowwise-e4m3fn-paged-v1",
                "data_file": "experts.fp8",
                "layers": layers,
                "num_experts": experts,
                "layout": layout.as_dict(),
            }
        ),
        encoding="utf-8",
    )
    return store


def _write_tiny_int4_store(
    root: Path,
    *,
    layers: int = 2,
    experts: int = 3,
) -> Path:
    layout = Int4ExpertPageLayout.create(hidden_size=128, intermediate_size=128)
    pages = np.empty((layers, experts, layout.page_stride), dtype=np.uint8)
    page_pattern = np.arange(layout.page_stride, dtype=np.uint8)
    for global_page, page in enumerate(pages.reshape(-1, layout.page_stride)):
        page[:] = page_pattern + np.uint8((global_page * 17) % 256)
    store = root / "tiny-int4-store"
    store.mkdir()
    (store / W4_STORE_FILE).write_bytes(pages.tobytes())
    (store / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": W4_STORE_SCHEMA,
                "codec": W4_STORE_CODEC,
                "dtype": "int4",
                "data_file": W4_STORE_FILE,
                "layers": layers,
                "num_experts": experts,
                "layout": layout.as_dict(),
            }
        ),
        encoding="utf-8",
    )
    return store


def test_expert_page_layout_is_fixed_and_aligned() -> None:
    layout = ExpertPageLayout.create(hidden_size=2048, intermediate_size=768)
    assert layout.gate_up_codes_bytes == 1536 * 2048
    assert layout.down_codes_bytes == 2048 * 768
    assert layout.page_stride % 4096 == 0
    assert layout.page_stride == 4_734_976


def test_expert_store_can_release_file_mapping_after_each_gather(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    target = torch.empty((2, store.layout.page_stride), dtype=torch.uint8)
    store.release_after_gather = True

    store.gather(0, [0, 2], target)

    assert getattr(store, "_pages", None) is None
    assert torch.count_nonzero(target) > 0


def test_int4_expert_page_layout_matches_30b_fixed_geometry() -> None:
    layout = Int4ExpertPageLayout.create(hidden_size=2048, intermediate_size=768)
    assert layout.group_size == 128
    assert layout.gate_up_codes_offset == 0
    assert layout.gate_up_codes_bytes == 1_572_864
    assert layout.gate_up_scales_offset == 1_572_864
    assert layout.gate_up_scales_bytes == 98_304
    assert layout.down_codes_offset == 1_671_168
    assert layout.down_codes_bytes == 786_432
    assert layout.down_scales_offset == 2_457_600
    assert layout.down_scales_bytes == 49_152
    assert layout.page_stride == 2_506_752
    assert layout.page_stride % 4096 == 0


def test_w4_grouped_launch_config_is_bound_to_the_sealed_ada_tuner() -> None:
    assert W4_GROUPED_LAUNCH_CONFIG == (32, 32, 32, 4, 2)


def test_w4_arithmetic_abi_defaults_to_production_predot_and_rejects_unknown() -> None:
    assert DEFAULT_W4_ARITHMETIC_POLICY == W4_PREDOT_BF16_ARITHMETIC_POLICY
    assert normalize_w4_arithmetic_policy(None) == W4_PREDOT_BF16_ARITHMETIC_POLICY
    assert (
        normalize_w4_arithmetic_policy(W4_POSTSCALE_BF16_ARITHMETIC_POLICY.upper())
        == W4_POSTSCALE_BF16_ARITHMETIC_POLICY
    )
    with pytest.raises(ValueError, match="W4 arithmetic policy"):
        normalize_w4_arithmetic_policy("unversioned-postscale")


def test_w4_postscale_launch_dispatch_reproduces_all_six_authoritative_winners() -> None:
    assert _w4_postscale_launch_config(8, 1536, 2048) == (16, 64, 64, 4, 2)
    assert _w4_postscale_launch_config(48, 1536, 2048) == (16, 64, 64, 4, 3)
    assert _w4_postscale_launch_config(32, 1536, 2048) == (16, 32, 64, 4, 2)
    for groups in (8, 48, 32):
        assert _w4_postscale_launch_config(groups, 2048, 768) == (16, 32, 64, 4, 2)
    assert _w4_postscale_launch_config(17, 256, 128) == (16, 32, 64, 4, 2)


def test_int4_quantization_matches_offset8_groupwise_contract() -> None:
    torch.manual_seed(29)
    source = torch.randn(5, 256, dtype=torch.float32)
    packed, scales = quantize_int4_weight(source)
    actual = dequantize_int4_weight(packed, scales)
    qstore_packed, qstore_scales = _quant_row_int4(source.numpy())

    assert packed.shape == (5, 128)
    assert packed.dtype == torch.uint8
    assert scales.shape == (5, 2)
    assert scales.dtype == torch.float32
    assert np.array_equal(packed.numpy(), qstore_packed)
    assert np.array_equal(scales.numpy(), qstore_scales)
    expected_even = ((source[:, 0] / scales[:, 0]).round().clamp(-7, 7) + 8).to(torch.uint8)
    expected_odd = ((source[:, 1] / scales[:, 0]).round().clamp(-7, 7) + 8).to(torch.uint8)
    assert torch.equal(packed[:, 0] & 0x0F, expected_even)
    assert torch.equal(packed[:, 0] >> 4, expected_odd)
    error_bound = scales.repeat_interleave(128, dim=1) * 0.5001
    assert bool(((actual - source).abs() <= error_bound).all())

    zeros, zero_scales = quantize_int4_weight(torch.zeros(2, 128))
    assert torch.equal(zeros, torch.full_like(zeros, 0x88))
    assert torch.equal(zero_scales, torch.ones_like(zero_scales))


def test_int4_page_views_preserve_packed_codes_and_group_scales() -> None:
    layout = Int4ExpertPageLayout.create(hidden_size=128, intermediate_size=128)
    raw = torch.zeros((2, layout.page_stride), dtype=torch.uint8)
    gate_up_source = torch.linspace(-2, 2, 256 * 128).reshape(256, 128)
    down_source = torch.linspace(-1, 1, 128 * 128).reshape(128, 128)
    gate_up_codes, gate_up_scales = quantize_int4_weight(gate_up_source)
    down_codes, down_scales = quantize_int4_weight(down_source)
    for group in range(2):
        raw[
            group,
            layout.gate_up_codes_offset : (
                layout.gate_up_codes_offset + layout.gate_up_codes_bytes
            ),
        ].copy_(gate_up_codes.reshape(-1))
        raw[
            group,
            layout.gate_up_scales_offset : (
                layout.gate_up_scales_offset + layout.gate_up_scales_bytes
            ),
        ].copy_(gate_up_scales.view(torch.uint8).reshape(-1))
        raw[
            group,
            layout.down_codes_offset : (layout.down_codes_offset + layout.down_codes_bytes),
        ].copy_(down_codes.reshape(-1))
        raw[
            group,
            layout.down_scales_offset : (layout.down_scales_offset + layout.down_scales_bytes),
        ].copy_(down_scales.view(torch.uint8).reshape(-1))

    page = int4_expert_page_views(raw, layout)
    assert page.gate_up.shape == (2, 256, 64)
    assert page.gate_up_scales.shape == (2, 256, 1)
    assert page.down.shape == (2, 128, 64)
    assert page.down_scales.shape == (2, 128, 1)
    assert torch.equal(page.gate_up[0], gate_up_codes)
    assert torch.equal(page.gate_up_scales[1], gate_up_scales)
    assert torch.equal(page.down[1], down_codes)
    assert torch.equal(page.down_scales[0], down_scales)


def test_fp8_page_views_preserve_codes_and_scales() -> None:
    layout = ExpertPageLayout.create(hidden_size=4, intermediate_size=2)
    raw = torch.zeros((2, layout.page_stride), dtype=torch.uint8)
    gate_up_source = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
    down_source = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
    gate_up_codes, gate_up_scales = quantize_fp8_weight(gate_up_source)
    down_codes, down_scales = quantize_fp8_weight(down_source)
    for group in range(2):
        raw[
            group,
            layout.gate_up_codes_offset : (
                layout.gate_up_codes_offset + layout.gate_up_codes_bytes
            ),
        ].copy_(gate_up_codes.view(torch.uint8).reshape(-1))
        raw[
            group,
            layout.gate_up_scales_offset : (
                layout.gate_up_scales_offset + layout.gate_up_scales_bytes
            ),
        ].copy_(gate_up_scales.view(torch.uint8).reshape(-1))
        raw[
            group,
            layout.down_codes_offset : (layout.down_codes_offset + layout.down_codes_bytes),
        ].copy_(down_codes.view(torch.uint8).reshape(-1))
        raw[
            group,
            layout.down_scales_offset : (layout.down_scales_offset + layout.down_scales_bytes),
        ].copy_(down_scales.view(torch.uint8).reshape(-1))
    page = expert_page_views(raw, layout)
    assert page.gate_up.shape == (2, 4, 4)
    assert page.down.shape == (2, 4, 2)
    assert torch.equal(page.gate_up[0], gate_up_codes)
    assert torch.equal(page.down[1], down_codes)
    assert torch.equal(page.gate_up_scales[1], gate_up_scales)
    assert torch.equal(page.down_scales[0], down_scales)


def test_page_cache_coalesces_misses_and_reuses_hits(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
    )
    selected, _page = cache.acquire(0, torch.tensor([[2, 0]]))
    assert selected.tolist() == [0, 2]
    assert cache.stats.page_misses == 2
    assert cache.stats.host_to_device_bytes == 2 * store.layout.page_stride
    cache.acquire(0, torch.tensor([[0, 2]]))
    assert cache.stats.page_hits == 2
    assert cache.stats.page_misses == 2
    cache.acquire(1, torch.tensor([[1, 2]]))
    assert cache.stats.page_misses == 4
    assert len(cache.entries) == cache.capacity == 2


def test_compact_live_page_trace_matches_offline_schema_and_event_semantics(
    tmp_path: Path,
) -> None:
    from mrun.science.qwen3_moe_page_trace import PageAccessTrace

    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=3,
        page_trace=True,
    )
    cache.acquire(0, torch.tensor([[2, 2, 0]]))
    cache.acquire(1, torch.tensor([[1, 1]]))
    cache.acquire(0, torch.tensor([[0]]))
    cache.acquire(0, torch.tensor([[2]]), protect=True)
    cache.acquire(1, torch.tensor([[1]]), protect=True)
    cache.acquire(0, torch.tensor([[0]]), protect=True)

    manifest = cache.export_page_trace()
    assert manifest is not None
    assert manifest == {
        "schema": PAGE_TRACE_SCHEMA,
        "layer_count": 2,
        "page_bytes": store.layout.page_stride,
        "source": {},
        "events": [
            {"phase": "prefill", "step": 0, "layer": 0, "pages": [[0, 1], [2, 2]]},
            {"phase": "prefill", "step": 0, "layer": 1, "pages": [[1, 2]]},
            {"phase": "prefill", "step": 1, "layer": 0, "pages": [[0, 1]]},
            {"phase": "decode", "step": 0, "layer": 0, "pages": [[2, 1]]},
            {"phase": "decode", "step": 0, "layer": 1, "pages": [[1, 1]]},
            {"phase": "decode", "step": 1, "layer": 0, "pages": [[0, 1]]},
        ],
        "trace_sha256": manifest["trace_sha256"],
    }
    core = {key: value for key, value in manifest.items() if key != "trace_sha256"}
    assert manifest["trace_sha256"] == canonical_sha256(core)
    assert PageAccessTrace.from_dict(manifest).as_dict() == manifest
    assert cache.page_trace_sha256 == manifest["trace_sha256"]

    drained = cache.drain_page_trace()
    assert drained == manifest
    assert cache.export_page_trace() is None
    assert cache.page_trace_sha256 is None


def test_compact_live_page_trace_hash_is_deterministic(tmp_path: Path) -> None:
    path = _write_tiny_store(tmp_path)

    def capture() -> dict[str, Any]:
        cache = ExpertPageCache(
            PackedFP8ExpertStore(path),
            device="cpu",
            cache_mb=0,
            max_active_pages=2,
            page_trace=True,
        )
        cache.acquire(0, torch.tensor([[2, 0, 2]]))
        cache.acquire(1, torch.tensor([[1]]), protect=True)
        manifest = cache.export_page_trace()
        assert manifest is not None
        return manifest

    first = capture()
    second = capture()
    assert first == second
    assert first["trace_sha256"] == second["trace_sha256"]


def test_page_trace_is_default_off_without_capture_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MRUN_QWEN3_MOE_PAGE_TRACE", raising=False)
    cache = ExpertPageCache(
        PackedFP8ExpertStore(_write_tiny_store(tmp_path)),
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
    )
    assert cache._page_trace is None  # noqa: SLF001 - zero-allocation default contract
    assert cache.page_trace_status() == {"enabled": False}
    assert cache.page_trace_sha256 is None
    assert cache.export_page_trace() is None
    assert cache.drain_page_trace() is None


def test_page_trace_caps_fail_before_cache_mutation(tmp_path: Path) -> None:
    path = _write_tiny_store(tmp_path)
    with pytest.raises(ValueError, match="hard maximum"):
        ExpertPageCache(
            PackedFP8ExpertStore(path),
            device="cpu",
            cache_mb=0,
            max_active_pages=2,
            page_trace=True,
            page_trace_max_events=PAGE_TRACE_HARD_MAX_EVENTS + 1,
        )

    request_limited = ExpertPageCache(
        PackedFP8ExpertStore(path),
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        page_trace=True,
        page_trace_max_requests=1,
    )
    with pytest.raises(Qwen3MoePageTraceLimitError, match="request cap"):
        request_limited.acquire(0, torch.tensor([[0, 1]]))
    assert not request_limited.entries
    assert request_limited.stats.as_dict()["page_requests"] == 0
    assert request_limited.export_page_trace() is None

    event_limited = ExpertPageCache(
        PackedFP8ExpertStore(path),
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        page_trace=True,
        page_trace_max_events=1,
    )
    event_limited.acquire(0, torch.tensor([[0]]))
    entries_before = dict(event_limited.entries)
    counters_before = event_limited.stats.as_dict()
    with pytest.raises(Qwen3MoePageTraceLimitError, match="event cap"):
        event_limited.acquire(1, torch.tensor([[1]]))
    assert dict(event_limited.entries) == entries_before
    assert event_limited.stats.as_dict() == counters_before

    probe = ExpertPageCache(
        PackedFP8ExpertStore(path),
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        page_trace=True,
    )
    probe.acquire(0, torch.tensor([[0]]))
    one_event_bytes = probe.page_trace_status()["manifest_bytes"]
    byte_limited = ExpertPageCache(
        PackedFP8ExpertStore(path),
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        page_trace=True,
        page_trace_max_manifest_bytes=one_event_bytes,
    )
    byte_limited.acquire(0, torch.tensor([[0]]))
    with pytest.raises(Qwen3MoePageTraceLimitError, match="manifest-byte cap"):
        byte_limited.acquire(1, torch.tensor([[1]]))


def test_page_trace_on_off_preserves_toy_pages_and_cache_counters(tmp_path: Path) -> None:
    path = _write_tiny_store(tmp_path)

    def run(*, traced: bool) -> tuple[list[tuple[torch.Tensor, ...]], dict[str, Any], dict]:
        cache = ExpertPageCache(
            PackedFP8ExpertStore(path),
            device="cpu",
            cache_mb=0,
            max_active_pages=2,
            page_trace=traced,
        )
        outputs: list[tuple[torch.Tensor, ...]] = []
        for layer, indices, protect in (
            (0, [[2, 0]], False),
            (0, [[0, 2]], True),
            (1, [[1, 2]], True),
            (1, [[2]], True),
        ):
            selected, pages = cache.acquire(
                layer,
                torch.tensor(indices),
                protect=protect,
            )
            outputs.append(
                (
                    selected.clone(),
                    pages.gate_up.clone(),
                    pages.gate_up_scales.clone(),
                    pages.down.clone(),
                    pages.down_scales.clone(),
                )
            )
        return outputs, cache.stats.as_dict(), dict(cache.entries)

    control_outputs, control_stats, control_entries = run(traced=False)
    traced_outputs, traced_stats, traced_entries = run(traced=True)
    assert len(control_outputs) == len(traced_outputs)
    for control, traced in zip(control_outputs, traced_outputs, strict=True):
        for control_tensor, traced_tensor in zip(control, traced, strict=True):
            assert torch.equal(control_tensor, traced_tensor)
    assert traced_stats == control_stats
    assert traced_entries == control_entries


def test_page_cache_uses_store_view_seam_for_int4_pages(tmp_path: Path) -> None:
    store = PackedInt4ExpertStore(_write_tiny_int4_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
    )
    selected, page, slots = cache.acquire_slot_binding(
        0,
        torch.tensor([[2, 0]]),
    )

    assert selected.tolist() == [0, 2]
    assert slots.tolist() == [0, 1]
    assert page.gate_up.shape == (cache.capacity, 256, 64)
    assert page.gate_up_scales.shape == (cache.capacity, 256, 1)
    assert page.down.shape == (cache.capacity, 128, 64)
    assert page.down_scales.shape == (cache.capacity, 128, 1)
    assert cache.stats.page_misses == 2
    assert cache.stats.host_to_device_bytes == 2 * store.layout.page_stride
    assert cache.compact is None


def test_page_cache_slot_binding_uses_physical_indices_without_compact_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
    )
    assert cache.compact is None
    expected_bytes = cache.cache.numel() + cache.miss_device.numel()
    assert cache.device_bytes == expected_bytes * cache.cache.element_size()

    # Force a non-identity logical-to-physical mapping: sorted experts [0, 2] occupy slots [1, 0].
    cache.acquire_slot_binding(0, torch.tensor([[2]]))
    cache.acquire_slot_binding(0, torch.tensor([[0]]))

    def reject_index_select(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("slot-indirect binding materialized a compact page tensor")

    monkeypatch.setattr(torch, "index_select", reject_index_select)
    selected, pages, slots = cache.acquire_slot_binding(
        0,
        torch.tensor([[0, 2]]),
    )

    assert selected.tolist() == [0, 2]
    assert slots.tolist() == [1, 0]
    assert cache.entries[(0, 0)] == 1
    assert cache.entries[(0, 2)] == 0
    assert pages.gate_up.shape[0] == cache.capacity
    assert pages.down.shape[0] == cache.capacity
    assert cache.compact is None


def test_slot_binding_policy_retains_explicit_compact_fallback(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
    )
    selected, pages = cache.acquire(0, torch.tensor([[2, 0]]))

    assert selected.tolist() == [0, 2]
    assert pages.gate_up.shape[0] == 2
    assert cache.compact is not None
    assert (
        cache.device_bytes
        == (cache.cache.numel() + cache.miss_device.numel() + cache.compact.numel())
        * cache.cache.element_size()
    )


def test_page_cache_device_bytes_counts_prefetch_staging(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        route_prefetch=True,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
    )

    assert cache.prefetch_device is not None
    expected_elements = (
        cache.cache.numel() + cache.miss_device.numel() + cache.prefetch_device.numel()
    )
    assert cache.device_bytes == expected_elements * cache.cache.element_size()


def test_page_binding_policy_defaults_to_promoted_slot_indirection(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
    )
    assert cache.page_binding_policy == SLOT_INDIRECT_PAGE_BINDING_POLICY
    assert cache.compact is None


def test_transient_prefill_requires_layer_frequency_quotas(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    with pytest.raises(ValueError, match="requires layer-frequency-lru-v1"):
        ExpertPageCache(
            store,
            device="cpu",
            cache_mb=0,
            max_active_pages=3,
            prefill_page_policy=TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY,
        )
    with pytest.raises(ValueError, match="incompatible with route-history prefetch"):
        ExpertPageCache(
            store,
            device="cpu",
            cache_mb=0,
            max_active_pages=3,
            route_prefetch=True,
            cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
            prefill_page_policy=TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY,
        )


def test_transient_prefill_admits_hot_pages_without_scan_pollution(
    tmp_path: Path,
) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=3,
        cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
        prefill_page_policy=TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY,
    )

    selected, pages, group_slots = cache.acquire_transient_prefill(
        0,
        torch.tensor([[2, 2, 2, 0, 0, 1]]),
    )

    assert selected.tolist() == [0, 1, 2]
    assert group_slots.tolist() == [1, 2, 0]
    assert pages.gate_up.shape[0] == 3
    assert set(cache.entries) == {(0, 0), (0, 2)}
    assert cache.layer_entry_counts == [2, 0]
    assert cache.stats.page_requests == 3
    assert cache.stats.page_misses == 3
    assert cache.stats.page_hits == 0
    assert cache.stats.transient_prefill_admitted_pages == 2

    cache.acquire_transient_prefill(
        1,
        torch.tensor([[1, 1, 1, 2, 0]]),
    )
    cache.acquire_slot_binding(
        0,
        torch.tensor([[0, 2]]),
        protect=True,
    )
    sealed = dict(cache.entries)
    assert set(sealed) == {(0, 0), (0, 2), (1, 1)}
    assert cache.layer_entry_counts == [2, 1]
    assert cache.decode_protected == {(0, 0), (0, 2)}

    # A second scan transfers the requested layer but cannot evict its sealed quota.
    cache.acquire_transient_prefill(
        0,
        torch.tensor([[1, 1, 1, 2, 0]]),
    )
    assert dict(cache.entries) == sealed
    assert cache.stats.transient_prefill_acquires == 3
    assert cache.stats.transient_prefill_pages == 9
    assert cache.stats.transient_prefill_admitted_pages == 3
    assert cache.stats.transient_prefill_preserved_pages == 2

    cache.reset(clear_pages=False)
    assert cache.decode_protected == {(0, 0), (0, 2)}
    cache.reset(clear_pages=True)
    assert not cache.decode_protected


def test_transient_prefill_materializes_mixed_residency_without_host_regather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path, experts=4))
    expected_pages = np.array(store._ensure_pages(), copy=True)  # noqa: SLF001
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=4,
        cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
        prefill_page_policy=TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY,
    )
    cache.acquire_slot_binding(
        0,
        torch.tensor([[0, 2]]),
        protect=True,
    )
    sealed_entries = dict(cache.entries)
    cache.reset(clear_pages=False)

    gather_calls: list[tuple[int, list[int]]] = []
    real_gather = store.gather

    def spy_gather(
        layer: int,
        expert_ids: list[int],
        target: torch.Tensor,
    ) -> None:
        gather_calls.append((layer, list(expert_ids)))
        real_gather(layer, expert_ids, target)

    monkeypatch.setattr(store, "gather", spy_gather)
    selected, _pages, group_slots = cache.acquire_transient_prefill(
        0,
        torch.tensor([[3, 3, 1, 0, 2]]),
    )

    assert selected.tolist() == [0, 1, 2, 3]
    assert group_slots.tolist() == [2, 0, 3, 1]
    assert gather_calls == [(0, [1, 3])]
    for logical, expert in enumerate(selected.tolist()):
        physical = int(group_slots[logical])
        expected = torch.from_numpy(expected_pages[store.global_page(0, expert)])
        assert torch.equal(cache.miss_device[physical], expected)
    assert dict(cache.entries) == sealed_entries
    assert cache.decode_protected == {(0, 0), (0, 2)}

    stride = store.layout.page_stride
    assert cache.stats.page_requests == 4
    assert cache.stats.page_hits == 2
    assert cache.stats.page_misses == 2
    assert cache.stats.host_to_device_bytes == 2 * stride
    assert cache.stats.transient_prefill_h2d_bytes == 2 * stride
    assert cache.stats.transient_prefill_resident_materialization_d2d_bytes == 2 * stride
    assert cache.stats.transient_prefill_admission_d2d_bytes == 0
    assert cache.stats.transient_prefill_preserved_pages == 2


def test_transient_prefill_copies_resident_page_before_stale_slot_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path, experts=4))
    expected_pages = np.array(store._ensure_pages(), copy=True)  # noqa: SLF001
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=3,
        cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
        prefill_page_policy=TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY,
    )
    cache.acquire_slot_binding(0, torch.tensor([[0, 1]]))
    cache.reset(clear_pages=False)

    gather_calls: list[tuple[int, list[int]]] = []
    real_gather = store.gather

    def spy_gather(
        layer: int,
        expert_ids: list[int],
        target: torch.Tensor,
    ) -> None:
        gather_calls.append((layer, list(expert_ids)))
        real_gather(layer, expert_ids, target)

    monkeypatch.setattr(store, "gather", spy_gather)
    selected, _pages, group_slots = cache.acquire_transient_prefill(
        0,
        torch.tensor([[2, 2, 2, 3, 3, 0]]),
    )

    assert selected.tolist() == [0, 2, 3]
    assert group_slots.tolist() == [2, 0, 1]
    assert gather_calls == [(0, [2, 3])]
    expected_resident = torch.from_numpy(expected_pages[store.global_page(0, 0)])
    assert torch.equal(cache.miss_device[2], expected_resident)
    assert set(cache.entries) == {(0, 2), (0, 3)}
    assert cache.layer_entry_counts == [2, 0]

    stride = store.layout.page_stride
    assert cache.stats.page_requests == 3
    assert cache.stats.page_hits == 1
    assert cache.stats.page_misses == 2
    assert cache.stats.host_to_device_bytes == 2 * stride
    assert cache.stats.transient_prefill_h2d_bytes == 2 * stride
    assert cache.stats.transient_prefill_resident_materialization_d2d_bytes == stride
    assert cache.stats.transient_prefill_admission_d2d_bytes == 2 * stride
    assert cache.stats.transient_prefill_admitted_pages == 2
    assert cache.stats.transient_prefill_preserved_pages == 0
    assert cache.stats.transient_prefill_replaced_pages == 2


def test_grouped_fp8_slot_shape_contract_separates_logical_and_physical_groups() -> None:
    source = torch.empty((3, 4))
    weights = torch.empty((4, 3, 4), dtype=torch.float8_e4m3fn)
    scales = torch.empty((4, 3))
    starts = torch.tensor([0, 2], dtype=torch.int64)
    counts = torch.tensor([2, 1], dtype=torch.int64)
    slots = torch.tensor([3, 1], dtype=torch.int64)

    assert _grouped_fp8_shape_contract(
        source,
        weights,
        scales,
        starts,
        counts,
        slots,
    ) == (2, 3, 4)
    with pytest.raises(ValueError, match="one physical group per logical group"):
        _grouped_fp8_shape_contract(
            source,
            weights,
            scales,
            starts,
            counts,
            None,
        )
    with pytest.raises(ValueError, match="outside physical weight storage"):
        _grouped_fp8_shape_contract(
            source,
            weights,
            scales,
            starts,
            counts,
            torch.tensor([4, 1]),
        )


def test_grouped_w4_slot_shape_contract_separates_logical_and_physical_groups() -> None:
    source = torch.empty((3, 128), dtype=torch.bfloat16)
    packed = torch.empty((4, 256, 64), dtype=torch.uint8)
    scales = torch.empty((4, 256, 1), dtype=torch.float32)
    starts = torch.tensor([0, 2], dtype=torch.int64)
    counts = torch.tensor([2, 1], dtype=torch.int64)
    slots = torch.tensor([3, 1], dtype=torch.int64)

    assert _grouped_w4_shape_contract(
        source,
        packed,
        scales,
        starts,
        counts,
        slots,
    ) == (2, 256, 128)
    with pytest.raises(ValueError, match="one physical group per logical group"):
        _grouped_w4_shape_contract(
            source,
            packed,
            scales,
            starts,
            counts,
            None,
        )
    with pytest.raises(ValueError, match="outside physical weight storage"):
        _grouped_w4_shape_contract(
            source,
            packed,
            scales,
            starts,
            counts,
            torch.tensor([4, 1]),
        )
    with pytest.raises(ValueError, match="scales have shape"):
        _grouped_w4_shape_contract(
            source,
            packed,
            torch.empty((4, 256, 2), dtype=torch.float32),
            starts,
            counts,
            slots,
        )


def test_w4_postscale_abi_rejects_non_g128_geometry_before_cuda() -> None:
    source = torch.empty((1, 64), dtype=torch.bfloat16)
    packed = torch.empty((1, 32, 32), dtype=torch.uint8)
    scales = torch.empty((1, 32, 1), dtype=torch.float32)
    starts = torch.tensor([0], dtype=torch.int64)
    counts = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(ValueError, match="requires group_size=128"):
        grouped_w4_mm_strided(
            source,
            packed,
            scales,
            starts,
            counts,
            group_size=64,
            arithmetic_policy=W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
            max_rows=1,
            out_dtype=torch.bfloat16,
        )

    source_fp32 = torch.empty((1, 128), dtype=torch.float32)
    packed_g128 = torch.empty((1, 32, 64), dtype=torch.uint8)
    scales_g128 = torch.empty((1, 32, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="requires BF16 source and output"):
        grouped_w4_mm_strided(
            source_fp32,
            packed_g128,
            scales_g128,
            starts,
            counts,
            arithmetic_policy=W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
            max_rows=1,
            out_dtype=torch.float32,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
def test_grouped_fp8_slot_indirection_matches_compact_page_kernel() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(29)
    layout = ExpertPageLayout.create(hidden_size=64, intermediate_size=32)
    raw = torch.zeros((4, layout.page_stride), device="cuda", dtype=torch.uint8)
    pages = expert_page_views(raw, layout)
    for physical_slot in range(4):
        gate_up, gate_up_scales = quantize_fp8_weight(torch.randn(64, 64))
        down, down_scales = quantize_fp8_weight(torch.randn(64, 32))
        pages.gate_up[physical_slot].copy_(gate_up.cuda())
        pages.gate_up_scales[physical_slot].copy_(gate_up_scales.cuda())
        pages.down[physical_slot].copy_(down.cuda())
        pages.down_scales[physical_slot].copy_(down_scales.cuda())
    starts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    counts = torch.tensor([2, 1], device="cuda", dtype=torch.int64)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int64)
    compact_raw = raw.index_select(0, slots)
    compact = expert_page_views(compact_raw, layout)

    cases = (
        (
            torch.randn(3, 64, device="cuda", dtype=torch.bfloat16),
            pages.gate_up,
            pages.gate_up_scales,
            compact.gate_up,
            compact.gate_up_scales,
        ),
        (
            torch.randn(3, 32, device="cuda", dtype=torch.bfloat16),
            pages.down,
            pages.down_scales,
            compact.down,
            compact.down_scales,
        ),
    )
    for source, weights, scales, compact_weights, compact_scales in cases:
        expected = grouped_fp8_mm_strided(
            source,
            compact_weights,
            compact_scales,
            starts,
            counts,
            max_rows=2,
            out_dtype=torch.bfloat16,
        )
        actual = grouped_fp8_mm_strided(
            source,
            weights,
            scales,
            starts,
            counts,
            group_slots=slots,
            max_rows=2,
            out_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
def test_grouped_w4_postscale_matches_oracle_slots_and_default_abi() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(43)
    layout = Int4ExpertPageLayout.create(hidden_size=128, intermediate_size=128)
    raw = torch.zeros((4, layout.page_stride), device="cuda", dtype=torch.uint8)
    pages = int4_expert_page_views(raw, layout)
    for physical_slot in range(4):
        gate_up, gate_up_scales = quantize_int4_weight(torch.randn(256, 128))
        pages.gate_up[physical_slot].copy_(gate_up.cuda())
        pages.gate_up_scales[physical_slot].copy_(gate_up_scales.cuda())

    starts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    counts = torch.tensor([2, 1], device="cuda", dtype=torch.int64)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int64)
    compact_raw = raw.index_select(0, slots)
    compact = int4_expert_page_views(compact_raw, layout)
    source = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)

    default = grouped_w4_mm_strided(
        source,
        compact.gate_up,
        compact.gate_up_scales,
        starts,
        counts,
        max_rows=2,
        out_dtype=torch.bfloat16,
    )
    explicit_predot = grouped_w4_mm_strided(
        source,
        compact.gate_up,
        compact.gate_up_scales,
        starts,
        counts,
        arithmetic_policy=W4_PREDOT_BF16_ARITHMETIC_POLICY,
        max_rows=2,
        out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(default, explicit_predot, rtol=0.0, atol=0.0)

    postscale_compact = grouped_w4_mm_strided(
        source,
        compact.gate_up,
        compact.gate_up_scales,
        starts,
        counts,
        arithmetic_policy=W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
        max_rows=2,
        out_dtype=torch.bfloat16,
    )
    postscale_slots = grouped_w4_mm_strided(
        source,
        pages.gate_up,
        pages.gate_up_scales,
        starts,
        counts,
        group_slots=slots,
        arithmetic_policy=W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
        max_rows=2,
        out_dtype=torch.bfloat16,
    )
    repeated = grouped_w4_mm_strided(
        source,
        pages.gate_up,
        pages.gate_up_scales,
        starts,
        counts,
        group_slots=slots,
        arithmetic_policy=W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
        max_rows=2,
        out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(postscale_slots, postscale_compact, rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated, postscale_slots, rtol=0.0, atol=0.0)

    oracle = torch.empty((3, 256), device="cuda", dtype=torch.float32)
    for group, (start, count) in enumerate(zip(starts.tolist(), counts.tolist(), strict=True)):
        dense = dequantize_int4_weight(
            compact.gate_up[group],
            compact.gate_up_scales[group],
            dtype=torch.float32,
        )
        oracle[start : start + count] = source[start : start + count].float() @ dense.t()
    relative_l2 = (postscale_slots.float() - oracle).norm() / oracle.norm()
    assert float(relative_l2.item()) <= 0.01
    torch.testing.assert_close(postscale_slots.float(), oracle, rtol=0.03, atol=0.03)
    assert not torch.equal(postscale_slots, default)


def test_layer_frequency_cache_preserves_a_hot_route_across_layer_spill(
    tmp_path: Path,
) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path, layers=2, experts=4))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=3,
        cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
    )

    # Expert 0 is the recurrent hot route. The two singleton experts enter first,
    # making expert 0 MRU even though `selected` remains sorted for searchsorted.
    selected, _ = cache.acquire(0, torch.tensor([[0, 0, 0, 1, 2]]))
    assert selected.tolist() == [0, 1, 2]
    assert list(cache.entries) == [(0, 1), (0, 2), (0, 0)]

    # Layer 0 owns three pages against a fair quota of two. A layer-1 miss evicts
    # the oldest over-quota singleton, not the frequently routed expert 0.
    cache.acquire(1, torch.tensor([[0]]))
    assert (0, 0) in cache.entries
    assert cache.layer_entry_counts == [2, 1]
    assert cache.stats.over_quota_evictions == 1

    hits_before = cache.stats.page_hits
    cache.acquire(0, torch.tensor([[0]]))
    assert cache.stats.page_hits == hits_before + 1


def test_page_cache_reset_clears_policy_and_prefetch_history(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        route_prefetch=True,
        cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
    )
    cache.acquire(0, torch.tensor([[0, 1]]))
    assert cache.entries
    assert cache._prev_routes  # noqa: SLF001 - reset-state contract

    cache.reset(clear_pages=True)

    assert not cache.entries
    assert cache.free_slots == list(range(cache.capacity - 1, -1, -1))
    assert cache.layer_entry_counts == [0] * store.layers
    assert not cache._prev_routes  # noqa: SLF001 - reset-state contract
    assert cache.prefetch_stats == {"prefetched_pages": 0, "prefetch_rounds": 0}
    assert cache.stats.as_dict()["evictions"] == 0


def test_compact_route_maps_original_expert_ids() -> None:
    top_indices = torch.tensor([[7, 2], [9, 7]], dtype=torch.long)
    top_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    selected = torch.tensor([2, 7, 9], dtype=torch.long)
    token_ids, coefficients, starts, counts = _compact_route(
        top_indices,
        top_weights,
        selected,
    )
    assert counts.tolist() == [1, 2, 1]
    assert starts.tolist() == [0, 1, 3]
    assert sorted(token_ids.tolist()) == [0, 0, 1, 1]
    assert sorted(coefficients.tolist()) == pytest.approx([0.3, 0.4, 0.6, 0.7])


def test_stable_route_reduce_is_invariant_to_expert_group_row_order() -> None:
    top_indices = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)
    top_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    selected = torch.tensor([0, 1, 2], dtype=torch.long)
    token_ids, coefficients, _starts, counts = _compact_route(
        top_indices,
        top_weights,
        selected,
    )
    expert_output = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    expected = torch.zeros((2, 2), dtype=torch.float32)
    expected.index_add_(
        0,
        token_ids,
        expert_output * coefficients[:, None],
    )

    actual = stable_route_reduce(
        expert_output,
        coefficients,
        token_ids,
        counts,
        top_indices,
        selected,
    )
    torch.testing.assert_close(actual, expected)

    # Reverse the two rows inside expert group 2 while keeping every aligned value together.
    permutation = torch.tensor([0, 1, 3, 2])
    reordered = stable_route_reduce(
        expert_output.index_select(0, permutation),
        coefficients.index_select(0, permutation),
        token_ids.index_select(0, permutation),
        counts,
        top_indices,
        selected,
    )
    assert torch.equal(reordered, actual)


def test_compiled_route_inverse_elides_stable_reconstruction() -> None:
    top_indices = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)
    top_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    selected = torch.tensor([0, 1, 2], dtype=torch.long)
    token_ids, coefficients, _starts, counts, inverse = _compact_route_with_inverse(
        top_indices,
        top_weights,
        selected,
    )
    expert_output = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    reconstructed = stable_route_reduce(
        expert_output,
        coefficients,
        token_ids,
        counts,
        top_indices,
        selected,
    )
    compiled = stable_route_reduce(
        expert_output,
        coefficients,
        token_ids,
        counts,
        top_indices,
        selected,
        inverse_assignments=inverse,
    )

    assert torch.equal(compiled, reconstructed)

    # CUDA uses atomic cursors, so rows inside one expert group may be written
    # in either order.  The inverse is emitted by that same scatter and must
    # therefore track the queue permutation rather than assume a stable sort.
    permutation = torch.tensor([0, 1, 3, 2])
    old_to_new = torch.empty_like(permutation)
    old_to_new.index_copy_(0, permutation, torch.arange(permutation.numel()))
    reordered = stable_route_reduce(
        expert_output.index_select(0, permutation),
        coefficients.index_select(0, permutation),
        token_ids.index_select(0, permutation),
        counts,
        top_indices,
        selected,
        inverse_assignments=old_to_new.index_select(0, inverse),
    )
    assert torch.equal(reordered, reconstructed)


def test_stable_route_reduction_policy_is_explicit(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
    )
    backend = PagedFP8ExpertBackend(
        cache,
        dtype=torch.float32,
        route_reduction_policy=STABLE_ROUTE_REDUCTION_POLICY,
    )
    assert backend.report()["route_reduction_policy"] == STABLE_ROUTE_REDUCTION_POLICY


def test_stable_route_reduction_is_the_default(tmp_path: Path) -> None:
    store = PackedFP8ExpertStore(_write_tiny_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
    )
    backend = PagedFP8ExpertBackend(cache, dtype=torch.float32)
    assert DEFAULT_ROUTE_REDUCTION_POLICY == STABLE_ROUTE_REDUCTION_POLICY
    assert backend.report()["route_reduction_policy"] == STABLE_ROUTE_REDUCTION_POLICY


def test_int4_backend_reuses_stable_reduction_and_slot_cache_policies(
    tmp_path: Path,
) -> None:
    store = PackedInt4ExpertStore(_write_tiny_int4_store(tmp_path))
    cache = ExpertPageCache(
        store,
        device="cpu",
        cache_mb=0,
        max_active_pages=2,
        cache_policy=LAYER_FREQUENCY_CACHE_POLICY,
        page_binding_policy=SLOT_INDIRECT_PAGE_BINDING_POLICY,
    )
    backend = PagedInt4ExpertBackend(cache, dtype=torch.bfloat16)
    report = backend.report()

    assert report["kind"] == "paged-w4-qstore"
    assert report["codec"] == W4_STORE_CODEC
    assert report["kernel"] == "triton-slot-indirect-grouped-w4a16-g128"
    assert report["arithmetic_policy"] == W4_PREDOT_BF16_ARITHMETIC_POLICY
    assert report["route_reduction_policy"] == STABLE_ROUTE_REDUCTION_POLICY
    assert report["cache_policy"] == LAYER_FREQUENCY_CACHE_POLICY
    assert report["page_binding_policy"] == SLOT_INDIRECT_PAGE_BINDING_POLICY

    postscale = PagedInt4ExpertBackend(
        cache,
        dtype=torch.bfloat16,
        w4_arithmetic_policy=W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
    ).report()
    assert postscale["arithmetic_policy"] == W4_POSTSCALE_BF16_ARITHMETIC_POLICY
    assert postscale["kernel"] == "triton-slot-indirect-grouped-w4a16-g128-postscale-bf16"


class _ZeroExperts:
    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        del layer, top_indices, top_weights
        return torch.zeros_like(source)


class _PhaseCaptureExperts(PagedFP8ExpertBackend):
    def __init__(self) -> None:
        self.phases: list[bool] = []

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        *,
        prefill: bool = False,
    ) -> torch.Tensor:
        del layer, top_indices, top_weights
        self.phases.append(prefill)
        return torch.zeros_like(source)


def _tiny_skeleton(
    *,
    heads: int = 2,
    kv_heads: int | None = None,
    head_dim: int = 4,
) -> SimpleNamespace:
    torch.manual_seed(3)
    hidden = 8
    kv_heads = heads if kv_heads is None else kv_heads
    layer = LayerWeights(
        input_norm=torch.ones(hidden),
        post_norm=torch.ones(hidden),
        q_norm=torch.ones(head_dim),
        k_norm=torch.ones(head_dim),
        q_proj=torch.randn(heads * head_dim, hidden) / 4,
        k_proj=torch.randn(kv_heads * head_dim, hidden) / 4,
        v_proj=torch.randn(kv_heads * head_dim, hidden) / 4,
        o_proj=torch.randn(hidden, heads * head_dim) / 4,
        router=torch.randn(2, hidden) / 4,
    )
    return SimpleNamespace(
        cfg={
            "num_hidden_layers": 1,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "norm_topk_prob": True,
        },
        device="cpu",
        dtype=torch.float32,
        embedding=torch.randn(13, hidden) / 4,
        layers=[layer],
        final_norm=torch.ones(hidden),
        lm_head=torch.randn(13, hidden) / 4,
    )


def test_runtime_classifies_first_single_token_as_prefill() -> None:
    backend = _PhaseCaptureExperts()
    runtime = Qwen3MoeDecodeRuntime(_tiny_skeleton(), backend)
    cache = runtime.new_cache(batch_size=1, capacity=2)

    runtime.forward(torch.tensor([[1]]), cache=cache)
    runtime.forward(torch.tensor([[2]]), cache=cache)

    assert backend.phases == [True, False]


def test_static_kv_incremental_and_full_logits_contract() -> None:
    runtime = Qwen3MoeDecodeRuntime(_tiny_skeleton(), _ZeroExperts())
    ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
    full_cache = runtime.new_cache(batch_size=1, capacity=8)
    full = runtime.forward(ids, cache=full_cache).logits
    all_cache = runtime.new_cache(batch_size=1, capacity=8)
    all_logits = runtime.forward(
        ids,
        cache=all_cache,
        all_logits=True,
    ).logits
    assert all_logits.shape == (1, 4, 13)
    torch.testing.assert_close(all_logits[:, -1], full)

    step_cache = runtime.new_cache(batch_size=1, capacity=8)
    incremental = None
    for index in range(ids.shape[1]):
        incremental = runtime.forward(
            ids[:, index : index + 1],
            cache=step_cache,
        ).logits
    assert incremental is not None
    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-5)
    assert step_cache.length == ids.shape[1]


@pytest.mark.parametrize(
    ("query_tokens", "key_tokens", "is_causal"),
    ((1, 5, False), (4, 4, True)),
)
def test_native_gqa_sdpa_matches_explicit_repeat_reference(
    query_tokens: int,
    key_tokens: int,
    is_causal: bool,
) -> None:
    torch.manual_seed(19)
    query = torch.randn(2, 4, query_tokens, 8)
    key = torch.randn(2, 2, key_tokens, 8)
    value = torch.randn(2, 2, key_tokens, 8)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        dropout_p=0.0,
        is_causal=is_causal,
    )
    actual = _native_sdpa_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_runtime_gqa_passes_compact_kv_to_native_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls: list[tuple[int, int, int, bool]] = []

    def spy_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        calls.append(
            (
                int(query.shape[1]),
                int(key.shape[1]),
                int(value.shape[1]),
                bool(kwargs.get("enable_gqa", False)),
            )
        )
        return real_sdpa(query, key, value, **kwargs)

    def reject_physical_repeat(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("runtime physically repeated grouped K/V heads")

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        spy_sdpa,
    )
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", reject_physical_repeat)
    runtime = Qwen3MoeDecodeRuntime(
        _tiny_skeleton(heads=4, kv_heads=2, head_dim=2),
        _ZeroExperts(),
    )
    cache = runtime.new_cache(batch_size=1, capacity=4)
    result = runtime.forward(torch.tensor([[1, 4, 7]]), cache=cache)

    assert result.logits is not None
    assert calls == [(4, 2, 2, True)]


def test_native_sdpa_preserves_equal_head_call_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sdpa = torch.nn.functional.scaled_dot_product_attention
    seen_kwargs: list[dict[str, Any]] = []

    def spy_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        seen_kwargs.append(kwargs)
        return real_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        spy_sdpa,
    )
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    _native_sdpa_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )
    assert len(seen_kwargs) == 1
    assert "enable_gqa" not in seen_kwargs[0]


def test_native_gqa_sdpa_fails_closed_without_enable_gqa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def legacy_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        del query, key, value, dropout_p, is_causal
        raise AssertionError("legacy SDPA must reject enable_gqa before execution")

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        legacy_sdpa,
    )
    with pytest.raises(RuntimeError, match="native enable_gqa support"):
        _native_sdpa_attention(
            torch.randn(1, 4, 1, 8),
            torch.randn(1, 2, 1, 8),
            torch.randn(1, 2, 1, 8),
            dropout_p=0.0,
            is_causal=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
def test_native_gqa_cuda_flash_only_matches_explicit_repeat_reference() -> None:
    torch.manual_seed(23)
    query = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 257, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(2, 2, 257, 64, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(4, dim=1),
        value.repeat_interleave(4, dim=1),
        dropout_p=0.0,
        is_causal=False,
    )
    actual = _native_sdpa_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_engine_capabilities_name_paging_and_telemetry() -> None:
    engine = object.__new__(Qwen3MoeCudaEngine)
    capabilities = engine.capabilities()
    assert capabilities.logits_batch
    assert capabilities.generation_batch
    assert capabilities.persistent_kv
    assert capabilities.transactional_kv
    assert capabilities.paged_experts
    assert capabilities.cache_telemetry
    assert not capabilities.autograd
    assert not capabilities.mlp_acts


@pytest.mark.parametrize(
    "alias",
    (
        "qwen3-moe-cuda",
        "qwen3_moe_cuda",
        "moe-qstore-cuda",
        "moe_qstore_cuda",
    ),
)
def test_open_engine_dispatches_qwen3_moe_aliases(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    def fake_init(self, model_name: str, **kwargs) -> None:
        self.name = model_name
        self.kwargs = kwargs

    monkeypatch.setattr(Qwen3MoeCudaEngine, "__init__", fake_init)
    engine = open_engine(
        "qwen3-30b-a3b",
        backend=alias,
        cache_mb=1234,
        expert_codec=EXPERT_CODEC_W4,
        w4_arithmetic_policy=W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
    )
    assert isinstance(engine, Qwen3MoeCudaEngine)
    assert engine.name == "qwen3-30b-a3b"
    assert engine.kwargs["cache_mb"] == 1234
    assert engine.kwargs["expert_codec"] == EXPERT_CODEC_W4
    assert engine.kwargs["w4_arithmetic_policy"] == W4_POSTSCALE_BF16_ARITHMETIC_POLICY


def test_store_resolution_uses_model_identity_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "Qwen3-30B-A3B"
    explicit = tmp_path / "explicit"
    assert resolve_qwen3_moe_store_dir(model_dir, explicit) == explicit
    configured = tmp_path / "configured"
    monkeypatch.setenv("MRUN_QWEN3_MOE_STORE_DIR", str(configured))
    assert resolve_qwen3_moe_store_dir(model_dir) == configured
    configured_w4 = tmp_path / "configured-w4"
    monkeypatch.setenv("MRUN_QWEN3_MOE_W4_STORE_DIR", str(configured_w4))
    assert resolve_qwen3_moe_store_dir(model_dir, expert_codec=EXPERT_CODEC_W4) == configured_w4


def test_measured_experiment_builder_is_accepted_only_for_same_format() -> None:
    quantization = {
        "codec": "rowwise-e4m3fn-paged-v1",
        "weight_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "granularity": "per-output-row",
        "page_alignment": 4096,
    }
    expected = {
        "schema_version": BUILDER_SCHEMA,
        "name": "mrun.engine.qwen3_moe_cuda.build_fp8_expert_store",
        "build_schema_version": "qwen3-moe-packed-expert-store-v1",
        "quantization": quantization,
    }
    source_files = [
        {
            "name": "qwen3_moe_decode.py",
            "bytes": 123,
            "sha256": "a" * 64,
        }
    ]
    actual = {
        **expected,
        "name": "generation_atlas.qwen3_moe_decode.build_fp8_expert_store",
        "source_files": source_files,
        "source_bundle_sha256": canonical_sha256(
            {
                "schema_version": BUILDER_SCHEMA,
                "files": source_files,
            }
        ),
    }
    _verify_compatible_promoted_builder(actual, expected)
    with pytest.raises(RuntimeError, match="format contract"):
        _verify_compatible_promoted_builder(
            {
                **actual,
                "quantization": {
                    **quantization,
                    "weight_dtype": "int4",
                },
            },
            expected,
        )


def _write_int4_builder_checkpoint(
    root: Path,
) -> tuple[Path, dict[str, torch.Tensor]]:
    model_dir = root / "int4-builder-checkpoint"
    model_dir.mkdir()
    cfg = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_size": 128,
        "moe_intermediate_size": 128,
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    torch.manual_seed(37)
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(cfg["num_experts"]):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        tensors[f"{prefix}.gate_proj.weight"] = torch.randn(128, 128, dtype=torch.bfloat16)
        tensors[f"{prefix}.up_proj.weight"] = torch.randn(128, 128, dtype=torch.bfloat16)
        tensors[f"{prefix}.down_proj.weight"] = torch.randn(128, 128, dtype=torch.bfloat16)
    shard_name = "model.safetensors"
    save_file(tensors, model_dir / shard_name)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name in tensors},
            }
        ),
        encoding="utf-8",
    )
    return model_dir, tensors


def test_int4_store_builder_validator_and_reader_share_one_strict_abi(
    tmp_path: Path,
) -> None:
    model_dir, tensors = _write_int4_builder_checkpoint(tmp_path)
    store_dir = tmp_path / "built-int4-store"
    manifest = build_int4_expert_store(
        model_dir,
        store_dir,
        model_name="tiny-qwen3-moe",
        hf_id="local/tiny-qwen3-moe",
    )
    layout = Int4ExpertPageLayout.create(128, 128)

    assert manifest["schema_version"] == W4_STORE_SCHEMA
    assert manifest["codec"] == W4_STORE_CODEC
    assert manifest["dtype"] == "int4"
    assert manifest["data_file"] == W4_STORE_FILE
    assert manifest["layout"] == layout.as_dict()
    assert manifest["data_bytes"] == 2 * layout.page_stride
    assert {entry["kind"] for entry in manifest["blocks"].values()} == {"int4-page"}

    validated = validate_int4_expert_store(model_dir, store_dir)
    assert validated["validation"]["geometry_verified"]
    assert not validated["validation"]["content_hashes_verified"]
    verified = validate_int4_expert_store(
        model_dir,
        store_dir,
        verify_content=True,
    )
    assert verified["validation"]["content_hashes_verified"]
    existing = build_int4_expert_store(
        model_dir,
        store_dir,
        model_name="tiny-qwen3-moe",
        hf_id="local/tiny-qwen3-moe",
    )
    assert existing["derived"] == manifest["derived"]
    with pytest.raises(RuntimeError, match="expert-store schema"):
        validate_fp8_expert_store(model_dir, store_dir)

    store = PackedInt4ExpertStore(store_dir)
    raw = torch.empty((1, layout.page_stride), dtype=torch.uint8)
    store.gather(0, [1], raw)
    page = store.view_pages(raw)
    actual_gate_up = dequantize_int4_weight(
        page.gate_up[0],
        page.gate_up_scales[0],
    )
    expected_gate_up = torch.cat(
        (
            tensors["model.layers.0.mlp.experts.1.gate_proj.weight"],
            tensors["model.layers.0.mlp.experts.1.up_proj.weight"],
        ),
        dim=0,
    ).float()
    error_bound = page.gate_up_scales[0].repeat_interleave(128, dim=1) * 0.5001
    assert bool(((actual_gate_up - expected_gate_up).abs() <= error_bound).all())


# ======================================================== checkpoint reading (index-free + FP8)
TINY_CFG = {
    "model_type": "qwen3_moe",
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "hidden_size": 8,
    "moe_intermediate_size": 6,
    "num_experts": 3,
    "num_experts_per_tok": 1,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
    "norm_topk_prob": True,
    "vocab_size": 11,
}


def _write_tiny_checkpoint(root: Path, *, tied: bool, sharded: bool = False) -> Path:
    """A minimal qwen3_moe safetensors checkpoint the skeleton loader can actually read."""
    torch.manual_seed(11)
    cfg = dict(TINY_CFG)
    hidden, heads, head_dim = cfg["hidden_size"], cfg["num_attention_heads"], cfg["head_dim"]
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(cfg["vocab_size"], hidden) / 4,
        "model.norm.weight": torch.ones(hidden),
    }
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": torch.ones(hidden),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
                f"{prefix}.self_attn.q_norm.weight": torch.ones(head_dim),
                f"{prefix}.self_attn.k_norm.weight": torch.ones(head_dim),
                f"{prefix}.self_attn.q_proj.weight": torch.randn(heads * head_dim, hidden) / 4,
                f"{prefix}.self_attn.k_proj.weight": torch.randn(heads * head_dim, hidden) / 4,
                f"{prefix}.self_attn.v_proj.weight": torch.randn(heads * head_dim, hidden) / 4,
                f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, heads * head_dim) / 4,
                f"{prefix}.mlp.gate.weight": torch.randn(cfg["num_experts"], hidden) / 4,
            }
        )
    if not tied:
        tensors["lm_head.weight"] = torch.randn(cfg["vocab_size"], hidden) / 4
    model_dir = root / ("tied" if tied else "untied")
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if sharded:
        names = sorted(tensors)
        halves = (names[: len(names) // 2], names[len(names) // 2 :])
        weight_map: dict[str, str] = {}
        for index, chunk in enumerate(halves, start=1):
            shard = f"model-0000{index}-of-00002.safetensors"
            save_file({name: tensors[name] for name in chunk}, str(model_dir / shard))
            weight_map.update({name: shard for name in chunk})
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
    else:
        save_file(tensors, str(model_dir / "model.safetensors"))
    return model_dir


def test_tensor_reader_addresses_indexed_and_index_free_checkpoints(tmp_path: Path) -> None:
    """Small checkpoints ship ONE model.safetensors and no index. Reading a named tensor out of
    them is the same range read, so the reader must not require the index file to exist."""
    flat = _write_tiny_checkpoint(tmp_path / "flat", tied=False)
    sharded = _write_tiny_checkpoint(tmp_path / "sharded", tied=False, sharded=True)
    assert not (flat / "model.safetensors.index.json").exists()

    flat_reader, sharded_reader = TensorReader(flat), TensorReader(sharded)
    assert flat_reader.has("model.embed_tokens.weight")
    assert sharded_reader.has("model.embed_tokens.weight")
    assert not flat_reader.has("nope.weight")
    assert flat_reader.get("model.embed_tokens.weight").shape == (
        TINY_CFG["vocab_size"],
        TINY_CFG["hidden_size"],
    )
    assert set(flat_reader.weight_map) == set(sharded_reader.weight_map)
    assert len(set(sharded_reader.weight_map.values())) == 2


def test_tensor_reader_dequantizes_block_scaled_fp8_weights(tmp_path: Path) -> None:
    """An FP8 checkpoint stores e4m3 CODES plus a scale. Handing a caller the raw codes silently
    reinterprets code bytes as numbers, so ``get`` must apply the scale; a tensor with no
    companion scale must come back byte-identical."""
    torch.manual_seed(5)
    reference = torch.randn(8, 8)
    scale = float(reference.abs().amax()) / 448.0
    codes = (reference / scale).to(torch.float8_e4m3fn)
    plain = torch.randn(4, 4)
    model_dir = tmp_path / "fp8"
    model_dir.mkdir()
    save_file(
        {
            "w.weight": codes,
            "w.weight_scale_inv": torch.full((1, 1), scale),
            "plain.weight": plain,
        },
        str(model_dir / "model.safetensors"),
    )
    reader = TensorReader(model_dir, dequant_dtype=torch.float32)

    assert reader.scale_key("w.weight") == "w.weight_scale_inv"
    assert reader.scale_key("plain.weight") is None
    assert reader.raw("w.weight").dtype == torch.float8_e4m3fn
    recovered = reader.get("w.weight")
    assert recovered.dtype == torch.float32
    # e4m3 has 3 mantissa bits => at most ~6.25% relative error. The claim under test is that
    # the SCALE was applied, not that fp8 round-trips exactly.
    assert torch.allclose(recovered, reference, rtol=0.07, atol=1e-4)
    # ...and skipping the scale is not a near-miss: raw codes are ~1/scale too large.
    raw_as_numbers = reader.raw("w.weight").to(torch.float32)
    assert float((raw_as_numbers - reference).abs().max()) > 100.0
    assert torch.equal(reader.get("plain.weight"), plain)


def test_block_fp8_dequant_expands_a_two_dimensional_block_grid() -> None:
    """The DeepSeek/Qwen layout has ONE scale per [block, block] tile; block size is inferred
    from the scale-vs-weight shape ratio, so the expansion must land tile-by-tile."""
    out = dequantize_block_fp8(
        torch.ones(4, 4, dtype=torch.float8_e4m3fn),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        dtype=torch.float32,
    )
    assert torch.equal(
        out,
        torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        ),
    )
    row_scaled = dequantize_block_fp8(
        torch.ones(3, 2, dtype=torch.float8_e4m3fn),
        torch.tensor([1.0, 2.0, 3.0]),
        dtype=torch.float32,
    )
    assert torch.equal(row_scaled, torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    assert torch.equal(
        dequantize_block_fp8(torch.ones(3, 2), torch.full((1, 1), 2.0), dtype=torch.float32),
        torch.full((3, 2), 2.0),
    )


def _block_quantized(rows: int, cols: int, block: int) -> tuple[torch.Tensor, ...]:
    """``(reference, codes, scale)`` for a weight blocked by ``block`` on both axes, with the
    scale divided out exactly so any dequant error is an INDEXING error, not fp8 rounding."""
    torch.manual_seed(1)
    reference = torch.randn(rows, cols)
    scale = torch.rand(-(-rows // block), -(-cols // block)) + 0.5
    expanded = scale[torch.arange(rows) // block][:, torch.arange(cols) // block]
    return reference, reference / expanded, scale


def test_block_size_is_taken_from_the_checkpoint_not_guessed_from_shapes() -> None:
    """A weight whose rows are NOT a multiple of the block is where guessing goes wrong
    silently: 300 rows in 3 scale groups is 128-blocked (128/128/44), but the quotient 300//3
    "infers" 100 and misaligns most rows onto a neighbouring block's scale. The declared block
    must be honoured, and an unguessable geometry must RAISE rather than return plausible
    nonsense."""
    reference, codes, scale = _block_quantized(300, 384, 128)
    exact = dequantize_block_fp8(codes, scale, dtype=torch.float32, block_size=(128, 128))
    assert float((exact - reference).abs().max()) < 1e-5

    with pytest.raises(ValueError, match="cannot infer the fp8 row block size"):
        dequantize_block_fp8(codes, scale, dtype=torch.float32)
    with pytest.raises(ValueError, match="declared row block"):
        dequantize_block_fp8(codes, scale, dtype=torch.float32, block_size=(64, 128))

    # the geometry that actually matters (Qwen3-Coder-480B hidden 6144, expert inter 2560)
    # divides evenly by 128, so inference is exact there
    reference, codes, scale = _block_quantized(1024, 512, 128)
    inferred = dequantize_block_fp8(codes, scale, dtype=torch.float32)
    assert float((inferred - reference).abs().max()) < 1e-5


def test_tensor_reader_uses_the_configs_declared_weight_block_size(tmp_path: Path) -> None:
    reference, codes, scale = _block_quantized(300, 384, 128)
    model_dir = tmp_path / "blocked"
    model_dir.mkdir()
    save_file(
        {"w.weight": codes, "w.weight_scale_inv": scale},
        str(model_dir / "model.safetensors"),
    )
    (model_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"weight_block_size": [128, 128]}}), encoding="utf-8"
    )
    reader = TensorReader(model_dir, dequant_dtype=torch.float32)
    assert reader.block_size == (128, 128)
    assert float((reader.get("w.weight") - reference).abs().max()) < 1e-5

    (model_dir / "config.json").unlink()  # same bytes, no declared geometry => refuse
    with pytest.raises(ValueError, match="cannot infer the fp8 row block size"):
        TensorReader(model_dir, dequant_dtype=torch.float32).get("w.weight")


def test_skeleton_reuses_the_embedding_when_the_checkpoint_ties_its_head(tmp_path: Path) -> None:
    """A tied checkpoint ships no lm_head tensor at all; the loader used to KeyError on it."""
    tied, _ = _load_skeleton(
        _write_tiny_checkpoint(tmp_path, tied=True), device="cpu", dtype=torch.float32
    )
    assert tied.tied_lm_head
    assert tied.lm_head is tied.embedding
    untied, _ = _load_skeleton(
        _write_tiny_checkpoint(tmp_path, tied=False), device="cpu", dtype=torch.float32
    )
    assert not untied.tied_lm_head
    assert untied.lm_head is not untied.embedding
    # a tied head is the SAME storage: counting it twice would overstate the resident budget
    assert tied.device_bytes < untied.device_bytes


# ================================================================== residual taps + ablation
class _ConstExperts:
    """Every routed token gets the same nonzero MoE contribution, so removing an MLP component
    is observable (a zero-output expert backend would make ablation a no-op by construction)."""

    def __init__(self, value: float = 0.25):
        self.value = value

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        del top_indices, top_weights
        return torch.full_like(source, self.value * (layer + 1))


def _two_layer_runtime() -> Qwen3MoeDecodeRuntime:
    skeleton = _tiny_skeleton()
    skeleton.cfg = {**skeleton.cfg, "num_hidden_layers": 2}
    skeleton.layers = [skeleton.layers[0], skeleton.layers[0]]
    skeleton.tied_lm_head = False
    return Qwen3MoeDecodeRuntime(skeleton, _ConstExperts())


def test_forward_captures_the_residual_stream_without_building_logits() -> None:
    runtime = _two_layer_runtime()
    ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
    result = runtime.forward(
        ids,
        cache=runtime.new_cache(batch_size=1, capacity=8),
        capture_hidden=True,
        hidden_last_only=False,
        return_final_hidden=True,
        skip_lm_head=True,
    )
    assert result.logits is None, "skip_lm_head must not build a vocab-sized tensor"
    assert len(result.hidden_states) == 3  # [embed_out, out_L0, out_L1]
    assert result.hidden_state_layers == (-1, 0, 1)
    assert result.hidden_capture_evidence is not None
    assert not result.hidden_capture_evidence.execution_time_pushdown
    assert result.hidden_capture_evidence.avoided_retained_bytes == 0
    assert all(state.shape == (1, 4, 8) for state in result.hidden_states)
    torch.testing.assert_close(result.hidden_states[0], runtime.skeleton.embedding[ids])
    assert result.final_hidden is not None and result.final_hidden.shape == (1, 4, 8)

    # hidden_last_only governs BOTH taps: it is what keeps a 62-layer capture off the budget.
    last_only = runtime.forward(
        ids,
        cache=runtime.new_cache(batch_size=1, capacity=8),
        capture_hidden=True,
        return_final_hidden=True,
        skip_lm_head=True,
    )
    assert all(state.shape == (1, 8) for state in last_only.hidden_states)
    assert last_only.final_hidden is not None and last_only.final_hidden.shape == (1, 8)
    torch.testing.assert_close(result.final_hidden[:, -1], last_only.final_hidden)
    for full, last in zip(result.hidden_states, last_only.hidden_states, strict=True):
        torch.testing.assert_close(full[:, -1], last)


def test_forward_selected_capture_filters_before_the_clone_and_reports_retention() -> None:
    runtime = _two_layer_runtime()
    ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
    full = runtime.forward(
        ids,
        cache=runtime.new_cache(batch_size=1, capacity=8),
        capture_hidden=True,
        skip_lm_head=True,
    )
    full_by_layer = dict(zip(full.hidden_state_layers, full.hidden_states, strict=True))

    cloned_boundaries: list[int] = []
    original_clone = runtime._clone_hidden_capture

    def counted_clone(
        boundary: int,
        state: torch.Tensor,
        *,
        hidden_last_only: bool,
    ) -> torch.Tensor:
        cloned_boundaries.append(boundary)
        return original_clone(
            boundary,
            state,
            hidden_last_only=hidden_last_only,
        )

    runtime._clone_hidden_capture = counted_clone  # type: ignore[method-assign]
    selected = runtime.forward(
        ids,
        cache=runtime.new_cache(batch_size=1, capacity=8),
        hidden_capture_layers=(1, -1, 1),
        skip_lm_head=True,
    )

    assert cloned_boundaries == [-1, 1], "unrequested layer 0 must never reach clone()"
    assert selected.hidden_state_layers == (-1, 1)
    assert len(selected.hidden_states) == 2
    for layer, state in zip(
        selected.hidden_state_layers,
        selected.hidden_states,
        strict=True,
    ):
        torch.testing.assert_close(state, full_by_layer[layer])

    evidence = selected.hidden_capture_evidence
    assert evidence is not None and evidence.execution_time_pushdown
    assert evidence.requested_boundaries == (-1, 1)
    assert evidence.retained_tensors == 2
    assert evidence.retained_bytes == sum(
        state.numel() * state.element_size() for state in selected.hidden_states
    )
    assert evidence.all_boundary_tape_bytes == evidence.retained_bytes * 3 // 2
    assert evidence.avoided_retained_bytes == evidence.retained_bytes // 2

    with pytest.raises(ValueError, match="-1..1"):
        runtime.forward(
            ids,
            cache=runtime.new_cache(batch_size=1, capacity=8),
            hidden_capture_layers=(2,),
            skip_lm_head=True,
        )


def test_component_ablation_removes_exactly_that_component() -> None:
    """attn ablation must equal a run whose o_proj output is zero; mlp ablation must equal a run
    whose expert backend returns zeros for that layer only."""
    ids = torch.tensor([[3, 5, 1]], dtype=torch.long)

    def run(runtime: Qwen3MoeDecodeRuntime, **kwargs: Any) -> torch.Tensor:
        result = runtime.forward(
            ids,
            cache=runtime.new_cache(batch_size=1, capacity=4),
            skip_lm_head=True,
            return_final_hidden=True,
            **kwargs,
        )
        assert result.final_hidden is not None
        return result.final_hidden

    runtime = _two_layer_runtime()
    clean = run(runtime)
    for target in ((0, "attn"), (0, "mlp"), (1, "attn"), (1, "mlp")):
        assert not torch.allclose(run(runtime, ablate=target), clean), f"{target} did nothing"

    class _ZeroLayer0Experts(_ConstExperts):
        def moe(self, layer, source, top_indices, top_weights):  # type: ignore[override]
            if layer == 0:
                return torch.zeros_like(source)
            return super().moe(layer, source, top_indices, top_weights)

    zeroed_l0 = _two_layer_runtime()
    zeroed_l0.backend = _ZeroLayer0Experts()
    torch.testing.assert_close(run(runtime, ablate=(0, "mlp")), run(zeroed_l0))

    zero_o_proj = _two_layer_runtime()
    original = zero_o_proj.skeleton.layers[1]
    zero_o_proj.skeleton.layers = [
        zero_o_proj.skeleton.layers[0],
        LayerWeights(
            **{
                **{name: getattr(original, name) for name in LayerWeights.__dataclass_fields__},
                "o_proj": torch.zeros_like(original.o_proj),
            }
        ),
    ]
    torch.testing.assert_close(run(runtime, ablate=(1, "attn")), run(zero_o_proj))

    with pytest.raises(ValueError, match="attn"):
        run(runtime, ablate=(0, "head"))
    with pytest.raises(ValueError, match="out of range"):
        run(runtime, ablate=(9, "mlp"))


def test_embedding_direction_ablation_removes_the_projection() -> None:
    runtime = _two_layer_runtime()
    ids = torch.tensor([[2, 6, 0]], dtype=torch.long)
    direction = torch.randn(8)
    unit = direction / direction.norm()
    result = runtime.forward(
        ids,
        cache=runtime.new_cache(batch_size=1, capacity=4),
        capture_hidden=True,
        hidden_last_only=False,
        skip_lm_head=True,
        ablate_embed_direction=direction,  # deliberately un-normalized: forward must normalize
    )
    assert float((result.hidden_states[0] @ unit).abs().max()) < 1e-5
    raw = runtime.skeleton.embedding[ids]
    assert float((raw @ unit).abs().max()) > 1e-3, "fixture must have a projection to remove"


def _cpu_moe_engine() -> Qwen3MoeCudaEngine:
    """The engine's analysis taps exercised on CPU. ``Qwen3MoeCudaEngine.__init__`` requires
    CUDA+Triton for the paged expert store, but the taps are plain torch over the runtime, so
    bypassing construction is the only way to cover them off a GPU host."""
    engine = object.__new__(Qwen3MoeCudaEngine)
    engine.runtime = _two_layer_runtime()
    engine.skeleton = engine.runtime.skeleton
    engine.cfg = engine.runtime.cfg
    engine.device = "cpu"
    engine.hidden = 8
    engine.n_layer = 2
    engine.init_taps()
    return engine


def test_engine_batch_taps_bucket_by_length_and_never_build_full_vocab_logits() -> None:
    engine = _cpu_moe_engine()
    rows = [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([4, 5], dtype=np.int64),
        np.array([6, 7, 8], dtype=np.int64),
    ]
    hidden = engine.hidden_last_batch(rows)
    assert hidden.shape == (3, 8)
    # bucketing regroups rows by length; caller order must survive it
    for index, row in enumerate(rows):
        torch.testing.assert_close(hidden[index], engine.hidden_last_batch([row])[0])

    head_rows = engine.lm_head_rows([2, 5])
    assert head_rows.shape == (2, 8)
    # per-row candidate ORDER must be honoured independently of the shared union lookup
    scores = engine.candidate_logits_batch(rows, ((2, 5), (5, 2), (2, 5)))
    assert [tuple(row.shape) for row in scores] == [(2,), (2,), (2,)]
    torch.testing.assert_close(scores[0], hidden[0].float() @ head_rows.T)
    torch.testing.assert_close(scores[1], (hidden[1].float() @ head_rows.T).flip(0))
    torch.testing.assert_close(scores[2], hidden[2].float() @ head_rows.T)

    states = engine.hidden_states(np.array([1, 2, 3], dtype=np.int64))
    assert len(states) == 3 and all(state.shape == (3, 8) for state in states)


def test_selected_hidden_last_batch_fuses_length_buckets_and_matches_scalar() -> None:
    rows = [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([4, 5], dtype=np.int64),
        np.array([6, 7, 8], dtype=np.int64),
    ]
    oracle = _cpu_moe_engine()
    expected = {
        layer: torch.stack(
            [oracle.hidden_states(row)[layer + 1][-1].float() for row in rows]
        )
        for layer in (-1, 1)
    }

    engine = _cpu_moe_engine()
    calls = 0
    original_forward = engine.runtime.forward

    def counted_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    engine.runtime.forward = counted_forward  # type: ignore[method-assign]
    captured = engine.selected_hidden_last_batch(rows, (-1, 1, 1))

    assert calls == 2  # two distinct prompt lengths, not three scalar forwards
    assert set(captured) == {-1, 1}
    for layer, value in captured.items():
        assert value.shape == (3, 8)
        torch.testing.assert_close(value, expected[layer])
    negotiated = engine.selected_capture_report()
    report = negotiated["last_execution"]
    assert isinstance(report, dict)
    assert report["physical_length_buckets"] == 2
    assert report["retained_tensors"] == 4  # 2 requested boundaries * 2 length buckets
    assert report["retained_bytes"] * 3 == report["all_boundary_tape_bytes"] * 2
    assert report["returned_selected_bytes"] == 3 * 2 * 8 * 4
    assert report["all_boundary_return_bytes"] == 3 * 3 * 8 * 4
    assert report["avoided_return_bytes"] == 3 * 1 * 8 * 4
    assert report["execution_time_pushdown"] is True

    with pytest.raises(ValueError, match="at least one"):
        engine.selected_hidden_last_batch(rows, ())
    with pytest.raises(ValueError, match="-1..1"):
        engine.selected_hidden_last_batch(rows, (2,))


def test_engine_ablation_scope_is_sticky_nestable_and_restores() -> None:
    engine = _cpu_moe_engine()
    rows = [np.array([1, 2, 3], dtype=np.int64)]
    clean = engine.hidden_last_batch(rows)
    direction = torch.randn(8)

    with engine.ablation(embed_direction=direction, lm_head=True):
        ablated = engine.hidden_last_batch(rows)
        assert not torch.allclose(ablated, clean)
        with engine.ablation(component=(0, "mlp")):
            # an inner COMPONENT scope must not silently cancel the outer embedding ablation
            assert engine.ablate_embed_direction is direction
            both = engine.hidden_last_batch(rows)
        assert not torch.allclose(both, ablated)
        assert engine.ablate_component is None
        unit = direction / direction.norm()
        assert float((engine.lm_head_rows([0, 1, 2]) @ unit).abs().max()) < 1e-4
    assert engine.ablate_embed_direction is None and not engine.ablate_lm_head
    torch.testing.assert_close(engine.hidden_last_batch(rows), clean)


def test_capabilities_advertise_selected_capture_but_not_a_whole_residual_batch_tap() -> None:
    engine = _cpu_moe_engine()
    capabilities = engine.capabilities()
    assert capabilities.selected_capture
    assert capabilities.residual_tap
    assert not capabilities.residual_tap_batch
    contract = engine.selected_capture_contract()
    assert contract["execution_time_pushdown"] is True
    assert contract["unrequested_boundaries_cloned"] is False
    assert contract["layer_address_range"] == (-1, 1)
    assert contract["token_selectors"] == ("last",)
