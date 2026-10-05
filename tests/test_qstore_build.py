from mrun.engine.kernels.qstore_build import _canon


def test_qwen35_canonicalizer_skips_qwen_mtp_layers() -> None:
    # Qwen3.8 includes an auxiliary multi-token-prediction stack under ``mtp.layers``.
    # It is not part of the paged causal decoder and must not alias decoder layer 0.
    assert _canon("mtp.layers.0.input_layernorm.weight", "qwen3_5") is None
    assert (
        _canon("model.language_model.layers.0.input_layernorm.weight", "qwen3_5")
        == "L0.ln1"
    )
