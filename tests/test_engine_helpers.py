import numpy as np
import torch

from mrun.engine import apply_patch_ops, layer_maps_for_global_neurons, length_ops_buckets


def test_apply_patch_ops_zero_scale_center():
    x = torch.ones(2, 4)
    y = apply_patch_ops(x, [("zero", [1], None), ("scale", [2], 3.0)])
    assert torch.equal(y[:, 1], torch.zeros(2))
    assert torch.equal(y[:, 2], torch.full((2,), 3.0))
    centered = apply_patch_ops(torch.tensor([[1.0, 3.0]]), [("center", [0, 1], None)])
    assert torch.allclose(centered, torch.tensor([[-1.0, 1.0]]))


def test_buckets_and_layer_maps():
    buckets = length_ops_buckets([3, 4, 30], max_batch=8, pad_tolerance=2)
    assert buckets == [[0, 1], [2]]
    maps = layer_maps_for_global_neurons(np.array([0, 3, 4]), inter=4)
    assert maps == {0: {"locals": [0, 3], "positions": [0, 1]}, 1: {"locals": [0], "positions": [2]}}
