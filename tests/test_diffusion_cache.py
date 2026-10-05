"""Tests for content-addressed MARS trajectory/reference cache policy."""

from __future__ import annotations

from mrun.diffusion import TrajectoryCache


def test_trajectory_cache_keys_include_reference_and_execution_identity() -> None:
    base = dict(
        model_identity="flux2-klein-4b:e7b7",
        conditioning_key="prompt:one",
        schedule_fingerprint="schedule:a",
        cut_step=2,
        resolution=(512, 512),
    )
    first = TrajectoryCache.make_key(**base, references=("ref:a",))
    same = TrajectoryCache.make_key(**base, references=("ref:a",))
    changed = TrajectoryCache.make_key(**base, references=("ref:b",))
    assert first == same
    assert first != changed


def test_trajectory_cache_keys_include_initial_latent_and_trajectory_abi() -> None:
    base = dict(
        model_identity="flux2-klein-4b:e7b7",
        conditioning_key="prompt:one",
        schedule_fingerprint="schedule:a",
        cut_step=2,
        resolution=(512, 512),
        references=("ref:a",),
        numerical_contract="saturn-real-edit-reference-v1",
        trajectory_abi="Flux2KleinKVPipeline:native-kv-ref-first-v1",
    )
    first = TrajectoryCache.make_key(**base, initial_latent_fingerprint="latent:a")
    changed_latent = TrajectoryCache.make_key(
        **base, initial_latent_fingerprint="latent:b"
    )
    abi_base = {**base, "trajectory_abi": "Flux2KleinKVPipeline:legacy-target-first-v0"}
    changed_abi = TrajectoryCache.make_key(
        **abi_base,
        initial_latent_fingerprint="latent:a",
    )
    assert first != changed_latent
    assert first != changed_abi


def test_trajectory_cache_hits_and_invalidates_reference_dependents() -> None:
    cache = TrajectoryCache(max_entries=2)
    key = TrajectoryCache.make_key(
        model_identity="model",
        conditioning_key="prompt",
        schedule_fingerprint="schedule",
        cut_step=1,
        resolution=(64, 64),
        references=("character:one",),
    )
    cache.put(
        "checkpoint",
        key=key,
        model_identity="model",
        conditioning_key="prompt",
        schedule_fingerprint="schedule",
        references=("character:one",),
        dependency_keys=("edit:face",),
    )
    assert cache.get(key, model_identity="model") is not None
    assert cache.get(key, model_identity="other") is None
    cache.put(
        "checkpoint-again",
        key=key,
        model_identity="model",
        conditioning_key="prompt",
        schedule_fingerprint="schedule",
        references=("character:one",),
        dependency_keys=("edit:face",),
    )
    assert cache.invalidate(("character:one",)) == (key,)
    assert cache.get(key) is None
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["invalidations"] == 2


def test_trajectory_cache_evicts_oldest_entry_under_bound() -> None:
    cache = TrajectoryCache(max_entries=1)
    keys = []
    for index in range(2):
        key = TrajectoryCache.make_key(
            model_identity="model",
            conditioning_key=f"prompt:{index}",
            schedule_fingerprint="schedule",
            cut_step=1,
            resolution=(64, 64),
        )
        keys.append(key)
        cache.put(
            f"checkpoint-{index}",
            key=key,
            model_identity="model",
            conditioning_key=f"prompt:{index}",
            schedule_fingerprint="schedule",
        )
    assert cache.get(keys[0]) is None
    assert cache.get(keys[1]) is not None
    assert cache.stats()["evictions"] == 1
