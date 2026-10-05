from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from mrun.decompiler import build_component_artifact
from mrun.decompiler import mlx_mixtral_expert_store as expert_store_module
from mrun.decompiler.cli import main as decompiler_main
from mrun.decompiler.mlx_mixtral_expert_store import (
    MIXTRAL_EXPERT_STORE_BITS,
    MIXTRAL_EXPERT_STORE_CODEC,
    MIXTRAL_EXPERT_STORE_GROUP_SIZE,
    MixtralExpertStoreArtifactError,
    VerifiedMixtralExpertStore,
    build_mixtral_mlx_expert_store,
)
from mrun.engine.mlx_component import _canonical_json_bytes, _sha256_bytes
from mrun.inference import cli as inference_cli
from mrun.inference.loader import (
    MixtralExpertPagedSliceConfig,
    load_mixtral_expert_paged_slice,
)
from mrun.runtime.contracts import PlacementPlan
from mrun.runtime.mlx_mixtral_expert_paged import (
    BoundedMixtralExpertCache,
    MixtralExpertCacheError,
    MixtralExpertPagedRuntimeError,
    MixtralExpertPagedRuntimeSlice,
    MixtralTieredPlacementError,
    ResidentMixtralSkeleton,
    plan_mixtral_tiered_placement,
    probe_mixtral_expert_paged_hardware,
    validate_mixtral_tiered_placement,
)

HIDDEN = 64
INTERMEDIATE = 64
LAYERS = 2
EXPERTS = 3
TOP_K = 2
VOCAB = 17
PHYSICAL_ROWS = 24


def _config() -> dict[str, object]:
    return {
        "architectures": ["MixtralForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": [2, 3],
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": 64,
        "model_type": "mixtral",
        "num_attention_heads": 4,
        "num_experts_per_tok": TOP_K,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": 2,
        "num_local_experts": EXPERTS,
        "output_router_logits": False,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": VOCAB,
    }


def _weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0x5A17)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.08

    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": random(PHYSICAL_ROWS, HIDDEN),
        "model.norm.weight": torch.ones(HIDDEN, dtype=torch.float32),
        "lm_head.weight": random(PHYSICAL_ROWS, HIDDEN),
    }
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": torch.ones(HIDDEN),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(HIDDEN),
                f"{prefix}.self_attn.q_proj.weight": random(HIDDEN, HIDDEN),
                f"{prefix}.self_attn.k_proj.weight": random(32, HIDDEN),
                f"{prefix}.self_attn.v_proj.weight": random(32, HIDDEN),
                f"{prefix}.self_attn.o_proj.weight": random(HIDDEN, HIDDEN),
                # Exact ties exercise deterministic expert-index selection and an empty expert.
                f"{prefix}.block_sparse_moe.gate.weight": torch.zeros(EXPERTS, HIDDEN),
            }
        )
        for expert in range(EXPERTS):
            expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert}"
            weights.update(
                {
                    f"{expert_prefix}.w1.weight": random(INTERMEDIATE, HIDDEN),
                    f"{expert_prefix}.w2.weight": random(HIDDEN, INTERMEDIATE),
                    f"{expert_prefix}.w3.weight": random(INTERMEDIATE, HIDDEN),
                }
            )
    return weights


def _write_source(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config(), sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False}, sort_keys=True), encoding="utf-8"
    )
    save_file(_weights(), root / "model.safetensors", metadata={"format": "pt"})


@pytest.fixture(scope="module")
def expert_artifact(tmp_path_factory: pytest.TempPathFactory) -> VerifiedMixtralExpertStore:
    pytest.importorskip("mlx.core")
    root = tmp_path_factory.mktemp("mlx-mixtral-expert-pages")
    source = root / "source"
    _write_source(source)
    canonical = build_component_artifact(source, root / "canonical")
    record = build_mixtral_mlx_expert_store(canonical.path, root / "native")
    assert record.verified_reopen
    return VerifiedMixtralExpertStore(record.path)


def _placement(
    artifact: VerifiedMixtralExpertStore,
    *,
    cache_pages: int = 1,
) -> Any:
    cache_bytes = cache_pages * artifact.expert_page_tensor_bytes
    budget = artifact.skeleton_tensor_bytes + cache_bytes + artifact.expert_page_tensor_bytes + 4096
    return plan_mixtral_tiered_placement(
        artifact,
        hardware_evidence=probe_mixtral_expert_paged_hardware(),
        expert_cache_capacity_bytes=cache_bytes,
        memory_budget_bytes=budget,
        workspace_bytes=1024,
        headroom_bytes=3072,
    )


def test_artifact_is_content_bound_and_skeleton_excludes_experts(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    artifact = expert_artifact
    assert artifact.manifest["production_runtime_eligible"] is False
    assert artifact.manifest["full_model_runtime"] is False
    assert artifact.manifest["performance_claim_valid"] is False
    assert artifact.manifest["execution_certified"] is False
    assert artifact.manifest["recipe"]["codec"] == MIXTRAL_EXPERT_STORE_CODEC
    assert artifact.manifest["recipe"]["bits"] == MIXTRAL_EXPERT_STORE_BITS
    assert artifact.manifest["recipe"]["group_size"] == MIXTRAL_EXPERT_STORE_GROUP_SIZE
    assert len(artifact.expert_pages) == LAYERS * EXPERTS
    assert all(".experts." not in name for name in artifact.skeleton_tensor_names)
    assert {
        f"model.layers.{layer}.block_sparse_moe.gate.weight" for layer in range(LAYERS)
    } <= artifact.skeleton_tensor_names
    assert artifact.skeleton_tensor_bytes == 113_408
    assert artifact.expert_page_tensor_bytes == 6_912
    assert artifact.expert_store_tensor_bytes == (
        LAYERS * EXPERTS * artifact.expert_page_tensor_bytes
    )
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            assert artifact.verify_expert_page(layer, expert).is_file()


def test_verified_store_public_state_cannot_redirect_an_expert(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    original = expert_artifact.expert_page(0, 0)
    published_pages = expert_artifact.expert_pages
    published_pages[0]["filename"] = expert_artifact.expert_page(0, 1)["filename"]
    published_topology = expert_artifact.topology
    published_topology["num_local_experts"] = 999
    published_manifest = expert_artifact.manifest
    published_manifest["expert_pages"][0]["expert"] = 2

    assert expert_artifact.expert_page(0, 0) == original
    assert expert_artifact.topology["num_local_experts"] == EXPERTS
    assert expert_artifact.expert_pages[0]["expert"] == 0
    with pytest.raises(AttributeError, match="immutable"):
        expert_artifact.expert_page_tensor_bytes = 1  # type: ignore[misc]


def test_authenticated_descriptor_detects_path_swap_during_consumption(
    expert_artifact: VerifiedMixtralExpertStore,
    tmp_path: Path,
) -> None:
    import mlx.core as mx

    copied = tmp_path / "descriptor-swap"
    shutil.copytree(expert_artifact.path, copied)
    artifact = VerifiedMixtralExpertStore(copied)
    page_zero = copied / artifact.expert_page(0, 0)["filename"]
    replacement = tmp_path / "replacement.safetensors"
    shutil.copy2(copied / artifact.expert_page(0, 1)["filename"], replacement)
    with pytest.raises(MixtralExpertStoreArtifactError, match="changed while"):
        with artifact.open_verified_member(artifact.expert_page(0, 0)["filename"]) as handle:
            arrays = mx.load(handle, format="safetensors")
            mx.eval(*arrays.values())
            os.replace(replacement, page_zero)


def test_config_double_open_race_fails_closed(
    expert_artifact: VerifiedMixtralExpertStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "config-race"
    shutil.copytree(expert_artifact.path, copied)
    original_hash = expert_store_module._hash_regular_file
    raced = False

    def racing_hash(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal raced
        result = original_hash(path, *args, **kwargs)
        if path.name == "config.json" and not raced:
            raced = True
            config = json.loads(path.read_text(encoding="utf-8"))
            config["max_position_embeddings"] = int(config["max_position_embeddings"]) + 1
            path.write_bytes(_canonical_json_bytes(config))
        return result

    monkeypatch.setattr(expert_store_module, "_hash_regular_file", racing_hash)
    with pytest.raises(MixtralExpertStoreArtifactError, match="changed between"):
        VerifiedMixtralExpertStore(copied)


def test_existing_build_reopens_only_after_full_verification(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    source_id = expert_artifact.source["artifact_id"]
    source_root = expert_artifact.path.parents[1] / "canonical"
    source_artifact = next(path for path in source_root.iterdir() if path.is_dir())
    record = build_mixtral_mlx_expert_store(source_artifact, expert_artifact.path.parent)
    assert record.path == expert_artifact.path
    assert record.artifact_sha256 == expert_artifact.artifact_sha256
    assert record.source_artifact_id == source_id


def test_existing_build_rejects_rehashed_page_cache_poisoning(
    expert_artifact: VerifiedMixtralExpertStore,
    tmp_path: Path,
) -> None:
    source_root = expert_artifact.path.parents[1] / "canonical"
    source_artifact = next(path for path in source_root.iterdir() if path.is_dir())
    record = build_mixtral_mlx_expert_store(source_artifact, tmp_path / "native")
    target = record.path
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_record = manifest["expert_pages"][0]
    page_path = target / page_record["filename"]
    tensors = load_file(page_path)
    packed = tensors["w1.weight"].clone()
    packed.reshape(-1)[0] = int(packed.reshape(-1)[0].item()) ^ 1
    tensors["w1.weight"] = packed
    save_file(tensors, page_path, metadata={"mrun": "self-consistent-forgery"})
    page_payload = page_path.read_bytes()
    page_record["bytes"] = len(page_payload)
    page_record["sha256"] = hashlib.sha256(page_payload).hexdigest()
    unhashed = dict(manifest)
    unhashed.pop("artifact_sha256")
    manifest["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(unhashed))
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    VerifiedMixtralExpertStore(target)

    with pytest.raises(MixtralExpertStoreArtifactError, match="canonical-source rebuild"):
        build_mixtral_mlx_expert_store(source_artifact, target.parent)


def test_artifact_rejects_page_tamper_and_extra_inventory(
    expert_artifact: VerifiedMixtralExpertStore,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(expert_artifact.path, tampered)
    page = tampered / expert_artifact.expert_page(0, 0)["filename"]
    payload = bytearray(page.read_bytes())
    payload[-1] ^= 0x01
    page.write_bytes(payload)
    with pytest.raises(Exception, match="hash mismatch|artifact"):
        VerifiedMixtralExpertStore(tampered)

    extra = tmp_path / "extra"
    shutil.copytree(expert_artifact.path, extra)
    (extra / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(MixtralExpertStoreArtifactError, match="inventory"):
        VerifiedMixtralExpertStore(extra)


@pytest.mark.parametrize("mutation", ["quantization", "shape"])
def test_artifact_rejects_rehashed_quantization_or_shape_forgery(
    expert_artifact: VerifiedMixtralExpertStore,
    tmp_path: Path,
    mutation: str,
) -> None:
    forged = tmp_path / mutation
    shutil.copytree(expert_artifact.path, forged)
    manifest_path = forged / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "quantization":
        manifest["recipe"]["bits"] = 3
        manifest["build_key_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest["recipe"]))
    else:
        manifest["expert_pages"][0]["tensors"][0]["shape"][0] += 1
    unhashed = dict(manifest)
    unhashed.pop("artifact_sha256")
    manifest["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(unhashed))
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    with pytest.raises(MixtralExpertStoreArtifactError, match="quantization|shape|geometry"):
        VerifiedMixtralExpertStore(forged)


def test_tiered_placement_is_separate_and_accounts_for_transactional_staging(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    placement = _placement(expert_artifact)
    assert not isinstance(placement, PlacementPlan)
    assert placement.fully_resident is False
    assert placement.includes_kv_state is False
    assert placement.performance_claim_valid is False
    assert placement.expert_load_staging_bytes == expert_artifact.expert_page_tensor_bytes
    assert placement.total_reserved_bytes == (
        placement.skeleton_resident_bytes
        + placement.expert_cache_capacity_bytes
        + placement.expert_load_staging_bytes
        + placement.workspace_bytes
        + placement.headroom_bytes
    )
    assert placement.total_reserved_bytes == 131_328
    hardware = probe_mixtral_expert_paged_hardware()
    validate_mixtral_tiered_placement(placement, expert_artifact, hardware)
    with pytest.raises(MixtralTieredPlacementError, match="exceeds budget"):
        plan_mixtral_tiered_placement(
            expert_artifact,
            hardware_evidence=hardware,
            expert_cache_capacity_bytes=expert_artifact.expert_page_tensor_bytes,
            memory_budget_bytes=placement.total_reserved_bytes - 1,
            workspace_bytes=placement.workspace_bytes,
            headroom_bytes=placement.headroom_bytes,
        )
    foreign = replace(placement, artifact_sha256="f" * 64)
    with pytest.raises(MixtralTieredPlacementError, match="stale|foreign|another"):
        validate_mixtral_tiered_placement(foreign, expert_artifact, hardware)


def test_resident_skeleton_and_lru_lease_lifecycle_are_bounded(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    skeleton = ResidentMixtralSkeleton(expert_artifact)
    assert skeleton.resident_bytes == expert_artifact.skeleton_tensor_bytes
    assert skeleton.router_shape(0) == (EXPERTS, HIDDEN)
    cache = BoundedMixtralExpertCache(
        expert_artifact, capacity_bytes=expert_artifact.expert_page_tensor_bytes
    )
    lease = cache.lease(((0, 0),))
    assert lease.page_nbytes((0, 0)) == expert_artifact.expert_page_tensor_bytes
    assert not hasattr(lease, "__getitem__")
    with pytest.raises(TypeError, match="subscriptable"):
        lease[(0, 0)]  # type: ignore[index]
    with pytest.raises(MixtralExpertCacheError, match="leased|protected"):
        cache.lease(((0, 1),))
    lease.release()
    with pytest.raises(MixtralExpertCacheError, match="more than once"):
        lease.release()
    with cache.lease(((0, 1),)):
        pass
    with cache.lease(((0, 1),)):
        pass
    stats = cache.stats()
    assert cache.cached_keys == ((0, 1),)
    assert stats["page_misses"] == 2
    assert stats["page_hits"] == 1
    assert stats["page_evictions"] == 1
    assert stats["resident_bytes"] <= cache.capacity_bytes
    assert stats["peak_resident_bytes"] <= cache.capacity_bytes
    with cache.request():
        with pytest.raises(MixtralExpertCacheError, match="active request"):
            cache.close()
    cache.close()
    with pytest.raises(MixtralExpertCacheError, match="closed"):
        cache.lease(((0, 0),))
    skeleton.close()
    with pytest.raises(MixtralExpertPagedRuntimeError, match="closed"):
        skeleton.router_shape(0)


def test_hardware_evidence_is_live_and_placement_rejects_forgery(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    hardware = probe_mixtral_expert_paged_hardware()
    assert hardware.platform_system == "Darwin"
    assert hardware.metal_available is True
    forged = replace(hardware, device_name=f"{hardware.device_name}-forged")
    with pytest.raises(MixtralExpertPagedRuntimeError, match="stale|forged|another"):
        plan_mixtral_tiered_placement(
            expert_artifact,
            hardware_evidence=forged,
            expert_cache_capacity_bytes=expert_artifact.expert_page_tensor_bytes,
            memory_budget_bytes=(
                expert_artifact.skeleton_tensor_bytes + 2 * expert_artifact.expert_page_tensor_bytes
            ),
        )


def _dequantized_projection(mx: Any, arrays: Mapping[str, Any], name: str) -> np.ndarray:
    value = mx.dequantize(
        arrays[f"{name}.weight"],
        arrays[f"{name}.scales"],
        arrays[f"{name}.biases"],
        group_size=MIXTRAL_EXPERT_STORE_GROUP_SIZE,
        bits=MIXTRAL_EXPERT_STORE_BITS,
        mode="affine",
    ).astype(mx.float32)
    mx.eval(value)
    return np.asarray(value, dtype=np.float32)


def _dense_dequantized_reference(
    artifact: VerifiedMixtralExpertStore,
    hidden: np.ndarray,
    selected: np.ndarray,
    routing_weights: np.ndarray,
) -> np.ndarray:
    import mlx.core as mx

    flattened = hidden.reshape(-1, HIDDEN).astype(np.float32)
    selected_flat = selected.reshape(-1, TOP_K)
    weights_flat = routing_weights.reshape(-1, TOP_K).astype(np.float32)
    output = np.zeros_like(flattened)
    for expert in range(EXPERTS):
        top_k_indices, token_indices = np.nonzero(selected_flat.T == expert)
        if not token_indices.size:
            continue
        arrays = mx.load(artifact.verify_expert_page(0, expert))
        w1 = _dequantized_projection(mx, arrays, "w1")
        w2 = _dequantized_projection(mx, arrays, "w2")
        w3 = _dequantized_projection(mx, arrays, "w3")
        source = flattened[token_indices]
        gate = source @ w1.T
        up = source @ w3.T
        activated = (gate / (1.0 + np.exp(-gate))) * up
        down = activated @ w2.T
        weighted = down * weights_flat[token_indices, top_k_indices, None]
        np.add.at(output, token_indices, weighted)
    return output.reshape(hidden.shape)


def test_sparse_block_route_dispatch_scatter_and_qmatmul_match_dense_q4_reference(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    import mlx.core as mx

    placement = _placement(expert_artifact)
    runtime = MixtralExpertPagedRuntimeSlice(
        expert_artifact,
        placement=placement,
        hardware_evidence=probe_mixtral_expert_paged_hardware(),
    )
    generator = np.random.default_rng(0xC0FFEE)
    hidden = generator.normal(0.0, 0.2, size=(2, 3, HIDDEN)).astype(np.float32)
    result = runtime.block(0).forward(mx.array(hidden))
    selected = np.asarray(result.selected_experts, dtype=np.int64)
    routing_weights = np.asarray(result.routing_weights, dtype=np.float32)
    actual = np.asarray(result.output, dtype=np.float32)

    # Canonical source route: FP32 softmax, top-k, selected-mass renormalization.
    torch_logits = torch.zeros((2, 3, EXPERTS), dtype=torch.float32)
    expected_weights, expected_selected = torch.topk(
        F.softmax(torch_logits, dim=-1, dtype=torch.float32), TOP_K, dim=-1
    )
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    np.testing.assert_array_equal(selected, expected_selected.numpy())
    np.testing.assert_allclose(routing_weights, expected_weights.numpy(), rtol=0, atol=0)
    assert result.dispatch_rows == (6, 6, 0)

    reference = _dense_dequantized_reference(expert_artifact, hidden, selected, routing_weights)
    np.testing.assert_allclose(actual, reference, rtol=4e-4, atol=4e-5)
    repeated = np.asarray(runtime.block(0)(mx.array(hidden)), dtype=np.float32)
    np.testing.assert_array_equal(actual, repeated)
    accounting = runtime.accounting()
    assert accounting["expert_cache_resident_bytes"] <= (accounting["expert_cache_capacity_bytes"])
    assert accounting["cache"]["peak_resident_bytes"] <= (accounting["expert_cache_capacity_bytes"])
    assert accounting["cache"]["page_evictions"] >= 1
    assert accounting["cache"]["logical_staging_bytes"] == 0
    assert accounting["logical_peak_tensor_bytes"] <= accounting["logical_reserved_tensor_bytes"]
    assert accounting["includes_kv_state"] is False
    assert accounting["full_model_runtime"] is False
    assert accounting["performance_claim_valid"] is False
    runtime.close()
    with pytest.raises(MixtralExpertPagedRuntimeError, match="closed"):
        runtime.block(0)


def test_sparse_block_rejects_nonfinite_output_and_releases_lifecycle(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    import mlx.core as mx

    runtime = MixtralExpertPagedRuntimeSlice(
        expert_artifact,
        placement=_placement(expert_artifact),
        hardware_evidence=probe_mixtral_expert_paged_hardware(),
    )
    block = runtime.block(0)
    extreme = mx.full((1, 1, HIDDEN), 1e30, dtype=mx.float32)
    with pytest.raises(MixtralExpertPagedRuntimeError, match="non-finite"):
        block.forward(extreme)
    assert runtime.cache.stats()["active_requests"] == 0
    runtime.close()
    with pytest.raises(MixtralExpertPagedRuntimeError, match="closed"):
        block.forward(mx.zeros((1, 1, HIDDEN), dtype=mx.float32))


def test_explicit_inference_slice_loader_reports_only_its_real_claims(
    expert_artifact: VerifiedMixtralExpertStore,
) -> None:
    placement = _placement(expert_artifact)
    config = MixtralExpertPagedSliceConfig(
        expert_store=expert_artifact.path,
        expert_cache_capacity_bytes=placement.expert_cache_capacity_bytes,
        memory_budget_bytes=placement.memory_budget_bytes,
        workspace_bytes=placement.workspace_bytes,
        headroom_bytes=placement.headroom_bytes,
    )
    with load_mixtral_expert_paged_slice(config) as loaded:
        report = loaded.describe()
        assert report["artifact"]["artifact_sha256"] == expert_artifact.artifact_sha256
        assert report["hardware_evidence"]["metal_available"] is True
        assert report["claim_boundary"] == {
            "explicit_opt_in": True,
            "full_model_runtime": False,
            "attention_implemented": False,
            "kv_state_implemented": False,
            "token_io_implemented": False,
            "real_checkpoint_execution_certified": False,
            "performance_claim_valid": False,
            "production_runtime_eligible": False,
        }


def test_explicit_inference_slice_cli_is_describe_only(
    expert_artifact: VerifiedMixtralExpertStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        inference_cli.main(
            [
                "describe-mixtral-expert-paged",
                str(expert_artifact.path),
                "--expert-cache-pages",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "mrun-loaded-mixtral-expert-paged-slice-v1"
    assert report["claim_boundary"]["full_model_runtime"] is False
    assert "serve" not in report


def test_mixtral_expert_store_decompiler_cli_build_and_verify_are_public(
    expert_artifact: VerifiedMixtralExpertStore,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = expert_artifact.path.parents[1] / "canonical"
    source_artifact = next(path for path in source_root.iterdir() if path.is_dir())
    assert (
        decompiler_main(
            [
                "lower-mlx-mixtral-experts",
                str(source_artifact),
                "--output-root",
                str(tmp_path / "cli-native"),
            ]
        )
        == 0
    )
    lowered = json.loads(capsys.readouterr().out)
    assert lowered["status"] == "native-expert-paged-approximate-unexecuted"
    assert decompiler_main(["verify-mlx-mixtral-experts", lowered["build"]["path"]]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["full_model_runtime"] is False
    assert verified["performance_claim_valid"] is False
    assert verified["production_runtime_eligible"] is False


def test_failed_page_reload_preserves_existing_cache_transaction(
    expert_artifact: VerifiedMixtralExpertStore,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(expert_artifact.path, copied)
    artifact = VerifiedMixtralExpertStore(copied)
    cache = BoundedMixtralExpertCache(artifact, capacity_bytes=artifact.expert_page_tensor_bytes)
    with cache.lease(((0, 0),)):
        pass
    before = (cache.cached_keys, cache.resident_bytes)
    page = copied / artifact.expert_page(0, 1)["filename"]
    payload = bytearray(page.read_bytes())
    payload[-1] ^= 0x80
    page.write_bytes(payload)
    with pytest.raises(Exception, match="hash mismatch|identity changed"):
        cache.lease(((0, 1),))
    assert (cache.cached_keys, cache.resident_bytes) == before
    assert cache.stats()["page_load_failures"] == 1
    cache.close()
