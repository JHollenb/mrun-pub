from __future__ import annotations

import math

import pytest

from mrun.runtime import SamplingPolicy, SamplingRequest
from mrun.runtime.dense_cuda import _sample_torch_row, _sample_torch_rows
from mrun.runtime.mlx_native import _sample_mlx_row
from mrun.runtime.sampling import (
    sampling_adjustments,
    stateless_uniform,
    validate_sampling_domain,
)


def _request(**updates) -> SamplingRequest:
    values = {
        "seed": 123,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": (),
    }
    values.update(updates)
    return SamplingRequest(
        policy=SamplingPolicy(**values),
        token_counts=((1, 2), (3, 1)),
        rng_counter=0,
    )


def test_splitmix_uniform_is_stateless_bounded_and_counter_owned() -> None:
    assert stateless_uniform(123, 0) == 0.7064912217637067
    assert stateless_uniform(123, 1) == 0.976596648325027
    assert stateless_uniform(-1, 0) == 0.8939429202831845
    trace = tuple(stateless_uniform(44, counter) for counter in range(16))
    interleaved = tuple(
        value
        for counter in range(16)
        for value in (stateless_uniform(44, counter), stateless_uniform(99, counter))
    )[::2]
    assert trace == interleaved
    assert all(0.0 <= value < 1.0 for value in trace)


def test_policy_and_dynamic_sampling_metadata_fail_closed() -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingPolicy(seed=0, top_p=0.0)
    with pytest.raises(ValueError, match="temperature"):
        SamplingPolicy(seed=0, temperature=math.inf)
    with pytest.raises(ValueError, match="signed 64-bit"):
        SamplingPolicy(seed=1 << 63)
    with pytest.raises(ValueError, match="sampling ABI"):
        SamplingPolicy(seed=0, sampling_abi="unknown-sampler")
    with pytest.raises(ValueError, match="unique"):
        SamplingPolicy(seed=0, logit_bias=((1, 2.0), (1, 3.0)))
    with pytest.raises(ValueError, match="positive"):
        SamplingRequest(SamplingPolicy(seed=0), ((1, 0),), 0)
    with pytest.raises(ValueError, match="semantic"):
        validate_sampling_domain(_request(logit_bias=((8, 1.0),)), 8)


def test_penalties_and_bias_combine_once_per_token() -> None:
    request = _request(
        frequency_penalty=0.5,
        presence_penalty=-0.25,
        logit_bias=((1, 2.0), (4, -1.0)),
    )
    assert sampling_adjustments(request) == (
        (1, 1.25),  # +2 bias - (2 * .5 - .25)
        (3, -0.25),
        (4, -1.0),
    )


def test_torch_sampling_supports_adjusted_argmax_top_k_top_p_and_seed() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

    adjusted = _request(
        temperature=0.0,
        frequency_penalty=1.0,
        presence_penalty=1.0,
        logit_bias=((0, 10.0),),
    )
    assert int(_sample_torch_row(logits, adjusted, semantic_token_count=5)) == 0

    top_k = _request(seed=7, top_k=2, top_p=1.0)
    top_k_tokens = {
        int(
            _sample_torch_row(
                logits,
                SamplingRequest(top_k.policy, top_k.token_counts, counter),
                semantic_token_count=5,
            )
        )
        for counter in range(32)
    }
    assert top_k_tokens <= {3, 4}
    assert top_k_tokens == {3, 4}

    nucleus = _request(seed=17, top_p=0.6)
    nucleus_tokens = {
        int(
            _sample_torch_row(
                logits,
                SamplingRequest(nucleus.policy, nucleus.token_counts, counter),
                semantic_token_count=5,
            )
        )
        for counter in range(32)
    }
    assert nucleus_tokens == {4}

    first = int(_sample_torch_row(logits, top_k, semantic_token_count=5))
    second = int(_sample_torch_row(logits, top_k, semantic_token_count=5))
    assert first == second


def test_torch_sampling_moves_only_selected_rows_and_encodes_nonfinite_failure() -> None:
    torch = pytest.importorskip("torch")
    requests = (_request(seed=1), _request(seed=2))
    logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0, 0.0]]])
    selected = _sample_torch_rows(logits, requests, semantic_token_count=4)
    assert len(selected) == 2
    assert all(0 <= token < 4 for token in selected)

    invalid = torch.tensor([0.0, float("nan"), 1.0, 2.0])
    sentinel = _sample_torch_row(invalid, requests[0], semantic_token_count=4)
    assert int(sentinel) == 4


def test_mlx_sampling_executes_on_metal_and_matches_simple_policy() -> None:
    mx = pytest.importorskip("mlx.core")
    logits = mx.array([0.0, 1.0, 2.0, 3.0, 4.0])
    request = _request(seed=7, top_k=2, top_p=0.8)
    first = _sample_mlx_row(mx, logits, request, semantic_token_count=5)
    second = _sample_mlx_row(mx, logits, request, semantic_token_count=5)
    mx.eval(first, second)
    assert int(first.item()) == int(second.item())
    assert int(first.item()) in {3, 4}

    adjusted = _request(temperature=0.0, logit_bias=((0, 20.0),))
    selected = _sample_mlx_row(mx, logits, adjusted, semantic_token_count=5)
    mx.eval(selected)
    assert int(selected.item()) == 0

    invalid = _sample_mlx_row(
        mx,
        mx.array([0.0, float("nan"), 1.0, 2.0, 3.0]),
        request,
        semantic_token_count=5,
    )
    mx.eval(invalid)
    assert int(invalid.item()) == 5
