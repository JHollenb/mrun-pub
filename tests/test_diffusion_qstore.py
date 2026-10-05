from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import save_file

from mrun.diffusion.qstore import (
    DIFFUSION_QSTORE_INTEGRITY_SCHEMA,
    DIFFUSION_QSTORE_NUMERICAL_LANE,
    DIFFUSION_QSTORE_SCHEMA,
    DiffusionQStore,
    DiffusionQStoreIntegrityError,
    build,
    install_paged_linears,
)


def _component(tmp_path):
    component = tmp_path / "transformer"
    component.mkdir()
    save_file(
        {
            "block.proj.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "block.proj.bias": torch.ones(3),
            "unused.weight": torch.eye(2),
        },
        str(component / "diffusion_pytorch_model.safetensors"),
    )
    return component


class _TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.proj = torch.nn.Linear(4, 3)

    def forward(self, value):
        return self.block.proj(value)


def test_diffusion_qstore_builds_and_fuses_a_leading_batch(tmp_path):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(
        component,
        store_dir,
        model_name="test/chroma",
        pipeline_class="ChromaPipeline",
    )

    manifest = json.loads((store_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == DIFFUSION_QSTORE_SCHEMA
    assert manifest["stats"]["source_2d_weight_tensors"] == 2
    assert manifest["integrity"]["schema_version"] == DIFFUSION_QSTORE_INTEGRITY_SCHEMA
    assert set(manifest["integrity"]["blocks"]) == set(manifest["blocks"])

    store = DiffusionQStore(store_dir)
    assert store.content_fingerprint is not None
    assert len(store.content_fingerprint) == 64
    assert store.integrity_status == "capable-unverified"
    denoiser = _TinyDenoiser()
    original_bias = denoiser.block.proj.bias.detach().clone()
    report = install_paged_linears(denoiser, store)

    assert report["paged_linear_modules"] == 1
    assert report["resident_linear_modules"] == 0
    value = torch.randn(2, 5, 4)
    output = denoiser(value)
    expected = torch.nn.functional.linear(
        value,
        store.weight("block.proj.weight", device=value.device, dtype=value.dtype),
        original_bias,
    )
    assert output.shape == (2, 5, 3)
    assert torch.equal(output, expected)
    assert store.stats()["page_loads"] == 2
    assert store.stats()["linear_calls"] == 1
    assert store.stats()["input_rows"] == 2
    assert store.weight("unused.weight", device=value.device, dtype=value.dtype).shape == (2, 2)
    assert store.stats()["page_loads"] == 3
    assert store.integrity_status == "fully-verified"
    store.close()


def test_diffusion_qstore_strict_lowering_rejects_partial_coverage(tmp_path):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(
        component,
        store_dir,
        model_name="test/chroma",
        pipeline_class="ChromaPipeline",
    )
    store = DiffusionQStore(store_dir)
    denoiser = _TinyDenoiser()
    denoiser.extra = torch.nn.Linear(3, 3)
    with pytest.raises(RuntimeError, match="did not cover"):
        install_paged_linears(denoiser, store)
    store.close()


def test_diffusion_qstore_lease_binds_order_identity_lane_and_exact_bytes(tmp_path):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    other_dir = tmp_path / "other-store"
    build(component, store_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    build(component, other_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    store = DiffusionQStore(store_dir)
    other_store = DiffusionQStore(other_dir)
    keys = ("block.proj.weight", "block.proj.weight", "unused.weight")
    lease = store.lease(keys, device="cpu", dtype=torch.float32)

    with lease:
        with pytest.raises(RuntimeError, match="store identity mismatch"):
            lease.resolve(keys[0], store=other_store, device="cpu", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="device mismatch"):
            lease.resolve(keys[0], store=store, device="meta", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype mismatch"):
            lease.resolve(keys[0], store=store, device="cpu", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="numerical lane mismatch"):
            lease.resolve(
                keys[0],
                store=store,
                device="cpu",
                dtype=torch.float32,
                numerical_lane="exact_bf16",
            )
        with pytest.raises(RuntimeError, match="demand mismatch"):
            lease.resolve("unused.weight", store=store, device="cpu", dtype=torch.float32)
        first = lease.resolve(keys[0], store=store, device="cpu", dtype=torch.float32)
        repeated = lease.resolve(keys[1], store=store, device="cpu", dtype=torch.float32)
        unused = lease.resolve(keys[2], store=store, device="cpu", dtype=torch.float32)
        assert first is repeated
        assert unused.shape == (2, 2)
        active = lease.telemetry()
        assert active["state"] == "active"
        assert active["ordered_keys"] == list(keys)
        assert active["numerical_lane"] == DIFFUSION_QSTORE_NUMERICAL_LANE
        assert active["declared_demand_bytes"] == 60
        assert active["demand_bytes"] == 60
        assert active["encoded_bytes_read"] == 36
        assert active["device_transfer_bytes"] == 0
        assert active["materializations"] == 2
        assert active["lease_cache_hits"] == 1
        assert active["integrity_capable"] is True
        assert active["integrity_required"] is True
        assert active["integrity_verified_pages"] == 2
        assert active["integrity_verified_weight_bytes"] == 16
        assert active["integrity_verified_scale_bytes"] == 20
        assert active["integrity_verified_bytes"] == 36
        assert active["content_fingerprint"] == store.content_fingerprint
        assert active["store_integrity_status"] == "fully-verified"
        assert active["measured_retained_device_bytes_current"] == 64
        assert active["measured_retained_device_bytes_peak"] == 64

    released = lease.telemetry()
    assert released["state"] == "released"
    assert released["measured_retained_device_bytes_current"] == 0
    assert released["measured_retained_device_bytes_peak"] == 64
    assert store.current_lease() is None
    with pytest.raises(RuntimeError, match="not active"):
        lease.resolve(keys[0], store=store, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="cannot release"):
        lease.release()

    undeclared = store.lease(["block.proj.weight"], device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="demand mismatch"):
        with undeclared:
            undeclared.resolve("unused.weight", store=store, device="cpu", dtype=torch.float32)
    assert undeclared.telemetry()["state"] == "released"

    changed_identity = store.lease(["block.proj.weight"], device="cpu", dtype=torch.float32)
    original_identity = store.identity_fingerprint
    store.identity_fingerprint = "f" * 64
    with pytest.raises(RuntimeError, match="store fingerprint changed"):
        changed_identity.__enter__()
    store.identity_fingerprint = original_identity

    incomplete = store.lease(keys[:2], device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="before consuming its exact demand"):
        with incomplete:
            incomplete.resolve(keys[0], store=store, device="cpu", dtype=torch.float32)
    assert incomplete.telemetry()["measured_retained_device_bytes_current"] == 0

    outer = store.lease(["block.proj.weight"], device="cpu", dtype=torch.float32)
    inner = store.lease(["unused.weight"], device="cpu", dtype=torch.float32)
    outer.__enter__()
    inner.__enter__()
    with pytest.raises(RuntimeError, match="release order is stale"):
        outer.release()
    inner.resolve("unused.weight", store=store, device="cpu", dtype=torch.float32)
    inner.release()
    outer.resolve("block.proj.weight", store=store, device="cpu", dtype=torch.float32)
    outer.release()
    store.close()
    other_store.close()


def test_diffusion_qstore_lease_required_linear_and_exception_cleanup(tmp_path):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(component, store_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    store = DiffusionQStore(store_dir)
    denoiser = _TinyDenoiser()
    report = install_paged_linears(denoiser, store, lease_required=True)
    assert report["lease_required"] is True
    assert report["numerical_lane"] == DIFFUSION_QSTORE_NUMERICAL_LANE
    value = torch.randn(2, 4)
    expected_weight = store.weight("block.proj.weight", device=value.device, dtype=value.dtype)
    expected = torch.nn.functional.linear(value, expected_weight, denoiser.block.proj.bias)

    with pytest.raises(RuntimeError, match="requires an active weight lease"):
        denoiser(value)
    lease = store.lease(["block.proj.weight"], device=value.device, dtype=value.dtype)
    with lease:
        output = denoiser(value)
    assert output.shape == (2, 3)
    assert torch.equal(output, expected)

    failed = store.lease(["block.proj.weight"], device=value.device, dtype=value.dtype)
    with pytest.raises(ValueError, match="consumer failed"):
        with failed:
            failed.resolve("block.proj.weight", store=store, device=value.device, dtype=value.dtype)
            raise ValueError("consumer failed")
    assert failed.telemetry()["state"] == "released"
    assert failed.telemetry()["measured_retained_device_bytes_current"] == 0
    assert store.current_lease() is None
    store.close()


def test_diffusion_qstore_current_lease_isolated_across_threads_and_stores(tmp_path):
    component = _component(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    build(component, first_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    build(component, second_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    stores = (DiffusionQStore(first_dir), DiffusionQStore(second_dir))

    def run(store):
        lease = store.lease(["unused.weight"], device="cpu", dtype=torch.float32)
        with lease:
            assert store.current_lease() is lease
            tensor = lease.resolve("unused.weight", store=store, device="cpu", dtype=torch.float32)
            return tensor.sum().item(), lease.store_fingerprint

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, stores))
    assert results[0][0] == results[1][0]
    assert all(store.current_lease() is None for store in stores)
    for store in stores:
        store.close()


@pytest.mark.parametrize(
    ("filename", "expected_message"),
    [
        ("weights.i8", "encoded weight integrity mismatch"),
        ("scales.f32", "encoded scale integrity mismatch"),
    ],
)
def test_diffusion_qstore_lease_refuses_same_size_page_corruption(
    tmp_path, filename, expected_message
):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(component, store_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    store = DiffusionQStore(store_dir)
    page_path = store_dir / filename
    original = page_path.read_bytes()
    corrupted = bytes([original[0] ^ 1]) + original[1:]
    assert len(corrupted) == len(original)
    page_path.write_bytes(corrupted)

    lease = store.lease(["block.proj.weight"], device="cpu", dtype=torch.float32)
    with pytest.raises(DiffusionQStoreIntegrityError, match=expected_message):
        with lease:
            lease.resolve("block.proj.weight", store=store, device="cpu", dtype=torch.float32)
    telemetry = lease.telemetry()
    assert telemetry["state"] == "released"
    assert telemetry["materializations"] == 0
    assert telemetry["consumed_demands"] == 0
    assert telemetry["demand_bytes"] == 0
    assert telemetry["integrity_verified_pages"] == 0
    assert telemetry["integrity_verification_attempts"] == 1
    assert telemetry["integrity_verification_failures"] == 1
    assert telemetry["store_integrity_status"] == "verification-failed"
    assert store.stats()["page_loads"] == 0
    store.close()


def test_diffusion_qstore_legacy_integrity_requires_explicit_unverified_opt_out(tmp_path):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(component, store_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    manifest_path = store_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["integrity"]
    manifest_path.write_text(json.dumps(manifest))
    store = DiffusionQStore(store_dir)
    assert store.integrity_capable is False

    with pytest.raises(DiffusionQStoreIntegrityError, match="legacy store"):
        store.lease(["block.proj.weight"], device="cpu", dtype=torch.float32)
    lease = store.lease(
        ["block.proj.weight"],
        device="cpu",
        dtype=torch.float32,
        require_integrity=False,
    )
    with lease:
        result = lease.resolve("block.proj.weight", store=store, device="cpu", dtype=torch.float32)
        assert result.shape == (3, 4)
    telemetry = lease.telemetry()
    assert telemetry["integrity_capable"] is False
    assert telemetry["integrity_required"] is False
    assert telemetry["integrity_authority"] == "non-authoritative-explicit-opt-out"
    assert telemetry["integrity_verified_pages"] == 0
    store.close()


def test_diffusion_qstore_manifest_rejects_partial_or_malformed_integrity(tmp_path):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(component, store_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    manifest_path = store_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["integrity"]["blocks"]["unused.weight"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="integrity coverage mismatch"):
        DiffusionQStore(store_dir)

    manifest = json.loads((store_dir / "manifest.json").read_text())
    manifest["integrity"]["blocks"]["unused.weight"] = {
        "weights_i8_sha256": "not-a-digest",
        "scales_f32_sha256": "d" * 64,
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="integrity digest"):
        DiffusionQStore(store_dir)


def test_diffusion_qstore_materializes_only_the_verified_page_snapshot(tmp_path, monkeypatch):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(component, store_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    store = DiffusionQStore(store_dir)
    baseline = store.weight("block.proj.weight", device="cpu", dtype=torch.float32).clone()
    page_path = store_dir / "weights.i8"
    original = page_path.read_bytes()
    original_materialize = store._materialize_snapshot

    def mutate_after_snapshot(snapshot, *, device, dtype):
        corrupted = bytes([original[0] ^ 1]) + original[1:]
        page_path.write_bytes(corrupted)
        return original_materialize(snapshot, device=device, dtype=dtype)

    monkeypatch.setattr(store, "_materialize_snapshot", mutate_after_snapshot)
    lease = store.lease(["block.proj.weight"], device="cpu", dtype=torch.float32)
    with lease:
        result = lease.resolve("block.proj.weight", store=store, device="cpu", dtype=torch.float32)
    assert torch.equal(result, baseline)
    assert lease.telemetry()["consumed_demands"] == 1
    assert lease.telemetry()["integrity_verified_pages"] == 1

    next_lease = store.lease(["block.proj.weight"], device="cpu", dtype=torch.float32)
    with pytest.raises(DiffusionQStoreIntegrityError, match="encoded weight integrity mismatch"):
        with next_lease:
            next_lease.resolve("block.proj.weight", store=store, device="cpu", dtype=torch.float32)
    assert next_lease.telemetry()["consumed_demands"] == 0
    store.close()


def test_diffusion_qstore_belady_budget_bounds_full_schedule_and_preserves_hits(tmp_path):
    component = _component(tmp_path)
    store_dir = tmp_path / "store"
    build(component, store_dir, model_name="test/chroma", pipeline_class="ChromaPipeline")
    store = DiffusionQStore(store_dir)
    expected_projection = store.weight("block.proj.weight", device="cpu", dtype=torch.float32)
    expected_unused = store.weight("unused.weight", device="cpu", dtype=torch.float32)
    schedule = (
        "block.proj.weight",
        "unused.weight",
        "block.proj.weight",
        "block.proj.weight",
    )
    lease = store.lease(
        schedule,
        device="cpu",
        dtype=torch.float32,
        max_retained_device_bytes=48,
    )
    with lease:
        outputs = [
            lease.resolve(key, store=store, device="cpu", dtype=torch.float32) for key in schedule
        ]
        assert torch.equal(outputs[0], expected_projection)
        assert torch.equal(outputs[1], expected_unused)
        assert torch.equal(outputs[2], expected_projection)
        assert outputs[2] is outputs[3]
        active = lease.telemetry()
        assert active["max_retained_device_bytes"] == 48
        assert active["retention_authority"] == "bounded-belady-next-use"
        assert active["measured_retained_device_bytes_current"] == 48
        assert active["measured_retained_device_bytes_peak"] == 48
        assert active["retention_evictions"] == 2
        assert active["lease_cache_hits"] == 1
        assert active["materializations"] == 3
        assert [event["evicted_key"] for event in active["retention_eviction_events"]] == [
            "block.proj.weight",
            "unused.weight",
        ]
    released = lease.telemetry()
    assert released["measured_retained_device_bytes_current"] == 0
    assert released["measured_retained_device_bytes_peak"] <= 48

    oversized = store.lease(
        ["block.proj.weight"],
        device="cpu",
        dtype=torch.float32,
        max_retained_device_bytes=47,
    )
    with pytest.raises(RuntimeError, match="exceeding lease budget"):
        with oversized:
            oversized.resolve("block.proj.weight", store=store, device="cpu", dtype=torch.float32)
    assert oversized.telemetry()["consumed_demands"] == 0
    assert oversized.telemetry()["materializations"] == 0
    assert oversized.telemetry()["integrity_verification_attempts"] == 0
    store.close()
