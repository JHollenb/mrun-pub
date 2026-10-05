"""Stateless primitives shared by accelerator-native token samplers."""

from __future__ import annotations

from .contracts import SamplingRequest

_MASK64 = (1 << 64) - 1
_INV_53 = 1.0 / float(1 << 53)


def stateless_uniform(seed: int, counter: int) -> float:
    """Return one deterministic ``[0, 1)`` variate using SplitMix64.

    This function intentionally owns no mutable RNG state.  The caller binds ``seed`` to one
    request and advances ``counter`` only after a model token commits.  The result is therefore
    invariant to thread scheduling, batching order, cancellation of other requests, and backend
    framework RNG state.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a strict integer")
    if isinstance(counter, bool) or not isinstance(counter, int):
        raise TypeError("counter must be a strict integer")
    if counter < 0:
        raise ValueError("counter must be non-negative")
    value = ((seed & _MASK64) + 0x9E3779B97F4A7C15 * (counter + 1)) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return float((value >> 11) & ((1 << 53) - 1)) * _INV_53


def sampling_adjustments(request: SamplingRequest) -> tuple[tuple[int, float], ...]:
    """Combine bias and OpenAI-style count penalties into sparse additive updates."""

    if not isinstance(request, SamplingRequest):
        raise TypeError("request must be SamplingRequest")
    policy = request.policy
    adjustments = {token: bias for token, bias in policy.logit_bias}
    for token, count in request.token_counts:
        delta = -(policy.frequency_penalty * count + policy.presence_penalty)
        adjustments[token] = adjustments.get(token, 0.0) + delta
    return tuple((token, value) for token, value in sorted(adjustments.items()) if value != 0.0)


def validate_sampling_domain(request: SamplingRequest, semantic_token_count: int) -> None:
    """Reject policy state that escapes the compiled semantic vocabulary."""

    if not isinstance(request, SamplingRequest):
        raise TypeError("request must be SamplingRequest")
    if (
        isinstance(semantic_token_count, bool)
        or not isinstance(semantic_token_count, int)
        or semantic_token_count <= 0
    ):
        raise ValueError("semantic_token_count must be a positive integer")
    policy = request.policy
    token_ids = (
        *(token for token, _ in policy.logit_bias),
        *(token for token, _ in request.token_counts),
    )
    if any(token >= semantic_token_count for token in token_ids):
        raise ValueError("sampling token metadata escapes the semantic token domain")


__all__ = [
    "sampling_adjustments",
    "stateless_uniform",
    "validate_sampling_domain",
]
