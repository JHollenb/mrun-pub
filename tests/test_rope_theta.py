"""rope_theta guard: raw config.json wins over AutoConfig; inv_freq buffers rebuilt.

Model-gated (needs a cached qwen snapshot). The pure-math patch test runs anywhere.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from mrun.models import patch_rope_theta


class _FakeRotary(torch.nn.Module):
    def __init__(self, head_dim=64, theta=10_000.0):
        super().__init__()
        idx = torch.arange(head_dim // 2, dtype=torch.float32)
        length = head_dim // 2
        self.register_buffer("inv_freq", theta ** (-(idx / length)))


def test_patch_rope_theta_rebuilds_buffer():
    mod = _FakeRotary(theta=10_000.0)
    before = mod.inv_freq.clone()
    n = patch_rope_theta(mod, 1_000_000.0)
    assert n == 1
    assert not torch.allclose(mod.inv_freq, before)
    length = mod.inv_freq.numel()
    idx = torch.arange(length, dtype=torch.float32)
    expect = 1_000_000.0 ** (-(idx / length))
    assert torch.allclose(mod.inv_freq, expect)


@pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS"))
    != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)
def test_qwen_rope_theta_matches_raw_config():
    from mrun.models import load_hf_model, resolve_model, snapshot_dir

    spec = resolve_model("qwen2.5-0.5b")
    try:
        snap = snapshot_dir(spec)
    except FileNotFoundError:
        pytest.skip("qwen2.5-0.5b not cached locally")
    raw_theta = json.loads((snap / "config.json").read_text())["rope_theta"]
    model = load_hf_model(spec, local_files_only=True)
    assert float(model.config.rope_theta) == float(raw_theta)
    # the actual rotary buffers must agree with the raw theta, not a 10000 default
    for name, buf in model.named_buffers():
        if name.endswith("inv_freq"):
            length = buf.numel()
            idx = torch.arange(length, dtype=torch.float32)
            expect = (float(raw_theta) ** (-(idx / length))).to(buf.dtype)
            assert torch.allclose(buf.cpu().float(), expect.float(), rtol=1e-4), name
            break
    else:
        pytest.fail("no inv_freq buffer found")
