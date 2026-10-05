"""mrun.diffusion phase-cuda contract tests (CPU-only; no diffusers, no GPU).

The fakes mirror the installed Flux2KleinPipeline surface that the wrapper
relies on: ``encode_prompt(prompt, device, num_images_per_prompt,
max_sequence_length) -> (prompt_embeds, text_ids)`` and a ``__call__`` that
accepts ``prompt_embeds``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mrun.diffusion import EmbedCache, PhaseError, PhasePipeline
from mrun.diffusion.phase import embed_cache_key

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Component(nn.Module):
    """Tiny module that records every .to() target; never really moves off cpu."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.to_targets: list[str] = []

    def to(self, *args, **kwargs):  # noqa: ANN002, ANN003
        target = args[0] if args else kwargs.get("device")
        self.to_targets.append(str(target))
        if str(target) == "cpu":
            return super().to("cpu")
        return self


def _prompt_tensor(prompt: str, max_sequence_length: int) -> torch.Tensor:
    seed = int.from_bytes(
        hashlib.sha256(f"{prompt}|{max_sequence_length}".encode()).digest()[:4], "big"
    )
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, 4, generator=gen).to(torch.bfloat16)


class Flux2KleinPipeline:
    """Fake with the real pipeline's embed surface (class name keys the contract)."""

    def __init__(self) -> None:
        self.text_encoder = _Component()
        self.transformer = _Component()
        self.vae = _Component()
        self.calls: list[dict] = []
        self.encode_count = 0

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        self.encode_count += 1
        embeds = _prompt_tensor(prompt, max_sequence_length)
        text_ids = torch.zeros(1, 3, 4)
        return embeds, text_ids

    def __call__(
        self,
        prompt=None,
        prompt_embeds=None,
        generator=None,
        num_inference_steps: int = 4,
        height: int = 64,
        width: int = 64,
        guidance_scale: float = 1.0,
        max_sequence_length: int = 512,
    ):
        assert prompt is None, "embed path must not re-pass the prompt"
        assert prompt_embeds is not None
        self.calls.append(
            {
                "prompt_embeds": prompt_embeds,
                "generator": generator,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
            }
        )
        return SimpleNamespace(
            images=[row.float().sum().item() for row in prompt_embeds]
        )


class FluxPipeline(Flux2KleinPipeline):
    """Fake FLUX.1 surface with the additional pooled conditioner output."""

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        embeds, text_ids = super().encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        pooled = embeds.mean(dim=1)
        return embeds, pooled, text_ids

    def __call__(
        self,
        prompt=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        **kwargs,
    ):
        assert pooled_prompt_embeds is not None
        self.calls.append({"pooled_prompt_embeds": pooled_prompt_embeds})
        return super().__call__(prompt=prompt, prompt_embeds=prompt_embeds, **kwargs)


class Krea2Pipeline(Flux2KleinPipeline):
    """Fake Krea 2 Turbo surface: positive embeds plus an attention mask."""

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        embeds, _ = super().encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        mask = torch.ones(1, embeds.shape[1], dtype=torch.bool)
        return embeds, mask

    def __call__(
        self,
        prompt=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        **kwargs,
    ):
        assert prompt is None
        assert prompt_embeds is not None
        assert prompt_embeds_mask is not None
        self.calls.append({"prompt_embeds_mask": prompt_embeds_mask})
        return super().__call__(prompt=prompt, prompt_embeds=prompt_embeds, **kwargs)


class ChromaPipeline(Flux2KleinPipeline):
    """Fake Chroma surface: positive/negative embeds plus T5 attention masks."""

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        lora_scale=None,
    ):
        embeds, text_ids = super().encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        mask = torch.ones(1, embeds.shape[1], dtype=torch.bool)
        negative = embeds.clone()
        negative_mask = mask.clone()
        return embeds, text_ids, mask, negative, text_ids.clone(), negative_mask

    def __call__(
        self,
        prompt=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        assert prompt is None
        assert prompt_embeds is not None
        assert prompt_attention_mask is not None
        assert negative_prompt_embeds is not None
        assert negative_prompt_attention_mask is not None
        self.calls.append(
            {
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_attention_mask": negative_prompt_attention_mask,
            }
        )
        return super().__call__(prompt=prompt, prompt_embeds=prompt_embeds, **kwargs)


class NoEmbedPipeline(Flux2KleinPipeline):
    def __call__(self, prompt=None, generator=None):  # no prompt_embeds
        raise AssertionError("must never be called")


def test_import_is_lazy_no_torch_no_diffusers():
    code = (
        "import sys\n"
        "import mrun.diffusion\n"
        "assert 'torch' not in sys.modules, 'torch imported at module scope'\n"
        "assert 'diffusers' not in sys.modules, 'diffusers imported at module scope'\n"
        "print('lazy-ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "lazy-ok" in out.stdout


def test_wrap_rejects_unknown_class_and_missing_embed_path():
    class MysteryPipeline(Flux2KleinPipeline):
        pass

    with pytest.raises(PhaseError, match="no embed contract"):
        PhasePipeline.wrap(MysteryPipeline(), device="cpu")
    # Same registered name but no prompt_embeds parameter -> fail closed.
    NoEmbedPipeline.__name__ = "Flux2KleinPipeline"
    with pytest.raises(PhaseError, match="no embed path"):
        PhasePipeline.wrap(NoEmbedPipeline(), device="cpu")


def test_wrap_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PhaseError, match="CUDA is unavailable"):
        PhasePipeline.wrap(Flux2KleinPipeline(), device="cuda")


def test_duck_call_encodes_once_and_feeds_embeds():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    gen = torch.Generator().manual_seed(7)
    out = wrapper(
        prompt="a red square",
        generator=gen,
        num_inference_steps=4,
        height=64,
        width=64,
        guidance_scale=1.0,
    )
    assert out.images
    assert pipe.encode_count == 1
    call = pipe.calls[-1]
    expected = _prompt_tensor("a red square", 512)
    assert call["prompt_embeds"].dtype == torch.bfloat16
    assert torch.equal(call["prompt_embeds"].view(torch.uint8), expected.view(torch.uint8))
    assert call["generator"] is gen
    assert call["num_inference_steps"] == 4

    # During denoise the encoder is detached so the pipeline cannot silently
    # re-encode and cannot define the execution device.
    assert wrapper.resident_phase == "denoise"
    assert pipe.text_encoder is None

    # Same prompt again: cache hit, no re-encode, no phase change.
    wrapper(prompt="a red square", generator=torch.Generator().manual_seed(8))
    assert pipe.encode_count == 1
    assert wrapper.encode_calls == 1

    # New prompt mid-denoise: correct but counted as an unplanned swap.
    wrapper(prompt="a blue circle", generator=torch.Generator().manual_seed(9))
    assert pipe.encode_count == 2
    assert wrapper.unplanned_swaps == 1
    # Encoder reattached for the encode, detached again for the denoise.
    assert pipe.text_encoder is None
    wrapper.close()
    assert pipe.text_encoder is wrapper._encoders["text_encoder"]


def test_encode_fresh_bypasses_embed_cache_without_promoting_result():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")

    cached = wrapper.encode("fresh conditioner")
    fresh = wrapper.encode_fresh("fresh conditioner")

    assert pipe.encode_count == 2
    assert fresh is not cached
    assert torch.equal(
        fresh.tensors["prompt_embeds"].view(torch.uint8),
        cached.tensors["prompt_embeds"].view(torch.uint8),
    )
    assert wrapper.fresh_encode_calls == 1
    assert wrapper.last_encode_observation["fresh"] is True
    assert wrapper.last_encode_observation["native_encode_invoked"] is True
    assert wrapper.last_encode_observation["memory_cache_bypassed"] is True
    assert wrapper.last_encode_observation["disk_cache_bypassed"] is True
    assert wrapper.last_encode_observation["fresh_result_promoted"] is False
    assert wrapper.encode("fresh conditioner") is cached
    assert pipe.encode_count == 2
    assert wrapper.encode_cache_hits == 1


def test_flux1_phase_contract_preserves_pooled_conditioner():
    pipe = FluxPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    embeds = wrapper.encode("pooled conditioner")
    assert set(embeds.tensors) == {"prompt_embeds", "pooled_prompt_embeds"}
    wrapper.generate(embeds, generator=torch.Generator().manual_seed(2))
    assert pipe.calls[-2]["pooled_prompt_embeds"].shape == (1, 4)


def test_krea2_phase_contract_preserves_prompt_mask():
    pipe = Krea2Pipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    embeds = wrapper.encode("krea prompt")

    assert set(embeds.tensors) == {"prompt_embeds", "prompt_embeds_mask"}
    wrapper.generate(
        embeds,
        generator=torch.Generator().manual_seed(3),
        num_inference_steps=8,
        guidance_scale=0.0,
    )
    assert pipe.calls[-2]["prompt_embeds_mask"].dtype == torch.bool


def test_chroma_phase_contract_preserves_cfg_embeds_and_masks():
    pipe = ChromaPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    embeds = wrapper.encode("chroma prompt")

    assert set(embeds.tensors) == {
        "prompt_embeds",
        "prompt_attention_mask",
        "negative_prompt_embeds",
        "negative_prompt_attention_mask",
    }
    wrapper.generate(
        embeds,
        generator=torch.Generator().manual_seed(4),
        num_inference_steps=40,
        guidance_scale=3.0,
    )
    call = pipe.calls[-2]
    assert call["prompt_attention_mask"].dtype == torch.bool
    assert call["negative_prompt_embeds"].shape == (1, 3, 4)
    assert call["negative_prompt_attention_mask"].dtype == torch.bool


def test_device_embed_cache_reuses_repeated_prompt_and_has_one_entry():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu", device_embed_cache=True)
    with wrapper.encode_phase():
        first = wrapper.encode("repeated")
        second = wrapper.encode("distinct")

    wrapper.generate(first, generator=torch.Generator().manual_seed(0))
    first_device_tensor = pipe.calls[-1]["prompt_embeds"]
    wrapper.generate(first, generator=torch.Generator().manual_seed(1))
    assert pipe.calls[-1]["prompt_embeds"] is first_device_tensor
    wrapper.generate(second, generator=torch.Generator().manual_seed(2))

    telemetry = wrapper.telemetry()
    bytes_per_embedding = (
        first.tensors["prompt_embeds"].numel() * first.tensors["prompt_embeds"].element_size()
    )
    assert telemetry.device_cache_hits == 1
    assert telemetry.cache_hits == 1
    assert telemetry.device_tensor_transfers == 2
    assert telemetry.device_transfer_bytes == 2 * bytes_per_embedding
    assert wrapper._device_embed_cache is not None
    assert wrapper._device_embed_cache.embeds is second
    assert len(wrapper._device_embed_cache.device_tensors) == 1


def test_device_embed_cache_is_opt_in_after_non_promotion():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    embeds = wrapper.encode("default miss path")

    wrapper.generate(embeds)
    wrapper.generate(embeds)

    telemetry = wrapper.telemetry()
    assert telemetry.device_cache_hits == 0
    assert telemetry.device_tensor_transfers == 2
    assert telemetry.device_cache_entries == 0
    assert wrapper._device_embed_cache is None


def test_device_embed_cache_invalidates_on_tensor_mutation_and_replacement():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu", device_embed_cache=True)
    embeds = wrapper.encode("mutable")
    original = embeds.tensors["prompt_embeds"]

    wrapper.generate(embeds)
    assert wrapper._device_embed_cache is not None
    assert wrapper._device_embed_cache.source_tensors["prompt_embeds"] is original
    wrapper.generate(embeds)
    embeds.tensors["prompt_embeds"].add_(1)
    wrapper.generate(embeds)
    replacement = embeds.tensors["prompt_embeds"].clone()
    embeds.tensors["prompt_embeds"] = replacement
    wrapper.generate(embeds)

    telemetry = wrapper.telemetry()
    bytes_per_embedding = replacement.numel() * replacement.element_size()
    assert telemetry.device_cache_hits == 1
    assert telemetry.device_tensor_transfers == 3
    assert telemetry.device_transfer_bytes == 3 * bytes_per_embedding
    assert pipe.calls[-1]["prompt_embeds"] is replacement
    assert wrapper._device_embed_cache is not None
    assert wrapper._device_embed_cache.source_tensors["prompt_embeds"] is replacement


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_embed_cache_replacement_on_real_cuda_without_diffusers():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cuda", device_embed_cache=True)
    embeds = wrapper.encode("real cuda replacement")
    original = embeds.tensors["prompt_embeds"]

    wrapper.generate(embeds)
    first_device_tensor = pipe.calls[-1]["prompt_embeds"]
    assert first_device_tensor.device.type == "cuda"
    assert wrapper._device_embed_cache is not None
    assert wrapper._device_embed_cache.source_tensors["prompt_embeds"] is original

    replacement = original.clone()
    embeds.tensors["prompt_embeds"] = replacement
    wrapper.generate(embeds)

    assert pipe.calls[-1]["prompt_embeds"].device.type == "cuda"
    assert pipe.calls[-1]["prompt_embeds"] is not first_device_tensor
    assert wrapper.telemetry().device_cache_hits == 0
    assert wrapper.telemetry().device_tensor_transfers == 2
    wrapper.close()


def test_device_embed_cache_invalidates_before_encode_and_on_close():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu", device_embed_cache=True)
    embeds = wrapper.encode("phase boundary")
    wrapper.generate(embeds)
    assert wrapper._device_embed_cache is not None

    with wrapper.encode_phase():
        pass
    assert wrapper._device_embed_cache is None

    with wrapper.denoise_phase():
        wrapper.generate(embeds)
    assert wrapper.telemetry().device_tensor_transfers == 2
    assert wrapper.telemetry().device_cache_hits == 0

    wrapper.close()
    assert wrapper._device_embed_cache is None
    assert wrapper.resident_phase == "none"


def test_precompute_then_generate_swaps_exactly_twice():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    prompts = ["p0", "p1", "p2", "p3"]
    with wrapper.encode_phase():
        embeds = wrapper.precompute(prompts)
    assert pipe.encode_count == 4
    with wrapper.denoise_phase():
        for item in embeds:
            for seed in (0, 1):
                wrapper.generate(item, generator=torch.Generator().manual_seed(seed))
    assert len(pipe.calls) == 8
    assert wrapper.unplanned_swaps == 0
    # Denoiser saw exactly one cpu (encode phase) + one cuda-target (denoise) move.
    assert pipe.transformer.to_targets == ["cpu", "cpu"]  # device is cpu here
    phase_labels = [event["label"] for event in wrapper.phase_events]
    assert phase_labels == ["encode", "denoise"]


def test_generate_batch_uses_one_physical_call_and_preserves_row_order():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    first = wrapper.encode("batch first")
    second = wrapper.encode("batch second")

    result = wrapper.generate_batch(
        [first, second],
        branch_ids=("first", "second"),
        generator=torch.Generator().manual_seed(19),
    )

    assert result.telemetry["physical_pipeline_calls"] == 1
    assert result.branch_ids == ("first", "second")
    assert len(pipe.calls) == 1
    assert pipe.calls[0]["prompt_embeds"].shape[0] == 2
    assert list(result.rows_by_branch()) == ["first", "second"]
    assert result.rows_by_branch()["first"] != result.rows_by_branch()["second"]


def test_generate_batch_repeats_one_embedding_for_intervention_rows():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    embeds = wrapper.encode("one prompt many seeds")
    result = wrapper.generate_batch(
        embeds,
        branch_ids=("seed-0", "seed-1", "seed-2"),
    )
    assert result.batch_size == 3
    assert len(result.row_outputs()) == 3
    assert torch.equal(
        pipe.calls[0]["prompt_embeds"][0],
        pipe.calls[0]["prompt_embeds"][1],
    )


def test_generate_batch_reuses_device_cache_for_repeated_embedding_rows():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu", device_embed_cache=True)
    embeds = wrapper.encode("one cached prompt many seeds")

    result = wrapper.generate_batch(
        embeds,
        branch_ids=("seed-0", "seed-1"),
    )

    assert result.telemetry["device_tensor_transfer_count"] == 1
    assert result.telemetry["device_transfer_bytes"] == (
        embeds.tensors["prompt_embeds"].numel()
        * embeds.tensors["prompt_embeds"].element_size()
    )
    assert wrapper.telemetry().device_cache_hits == 1
    assert wrapper.telemetry().device_tensor_transfers == 1
    assert pipe.calls[0]["prompt_embeds"].shape[0] == 2
    assert result.telemetry["numerical_contract"] == "batch_approximate"


def test_generate_batch_fails_closed_on_incompatible_feed_shapes():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    first = wrapper.encode("shape one")
    second = wrapper.encode("shape two")
    second.tensors["prompt_embeds"] = torch.zeros(
        1,
        4,
        4,
        dtype=second.tensors["prompt_embeds"].dtype,
    )
    with pytest.raises(PhaseError, match="incompatible"):
        wrapper.generate_batch([first, second], branch_ids=("a", "b"))


def test_serial_contract_rejects_prompt_lists():
    wrapper = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    with pytest.raises(PhaseError, match="serial"):
        wrapper(prompt=["a", "b"])  # type: ignore[arg-type]
    with pytest.raises(PhaseError, match="one prompt string"):
        wrapper.encode(["a", "b"])  # type: ignore[arg-type]


def test_disk_cache_round_trip_is_byte_identical(tmp_path):
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu", cache_dir=tmp_path)
    embeds = wrapper.encode("byte identity")
    key = embed_cache_key("Flux2KleinPipeline", "byte identity", {})
    assert embeds.key == key
    cache = EmbedCache(tmp_path)
    loaded = cache.load(key)
    assert loaded is not None
    original = embeds.tensors["prompt_embeds"]
    restored = loaded.tensors["prompt_embeds"]
    assert restored.dtype == original.dtype == torch.bfloat16
    assert torch.equal(restored.view(torch.uint8), original.view(torch.uint8))
    assert loaded.meta["prompt"] == "byte identity"

    # A second wrapper over a fresh pipeline reuses the disk entry: zero encodes.
    pipe2 = Flux2KleinPipeline()
    wrapper2 = PhasePipeline.wrap(pipe2, device="cpu", cache_dir=tmp_path)
    again = wrapper2.encode("byte identity")
    assert pipe2.encode_count == 0
    assert torch.equal(again.tensors["prompt_embeds"].view(torch.uint8), original.view(torch.uint8))


def test_cache_key_tracks_encode_params():
    base = embed_cache_key("Flux2KleinPipeline", "p", {})
    longer = embed_cache_key("Flux2KleinPipeline", "p", {"max_sequence_length": 256})
    other_model = embed_cache_key("Flux2Pipeline", "p", {})
    assert len({base, longer, other_model}) == 3


def test_flux_cache_key_separates_masked_conditioner_contract():
    native = embed_cache_key("Flux2KleinPipeline", "p", {})
    flux = embed_cache_key("FluxPipeline", "p", {})
    assert native != flux


def test_flux1_main_conditioner_passes_attention_masks():
    class Batch(dict):
        def to(self, device):
            return Batch({key: value.to(device) for key, value in self.items()})

    class Tokenizer:
        model_max_length = 77

        def __init__(self):
            self.calls = []

        def __call__(self, prompts, **kwargs):
            self.calls.append(kwargs)
            length = int(kwargs["max_length"])
            return Batch(
                input_ids=torch.zeros((1, length), dtype=torch.long),
                attention_mask=torch.ones((1, length), dtype=torch.long),
            )

    class T5(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_attention_mask = None

        def forward(self, input_ids, attention_mask):
            self.last_attention_mask = attention_mask.detach().clone()
            return (torch.zeros((1, input_ids.shape[1], 4)),)

    class CLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_attention_mask = None

        def forward(self, input_ids, attention_mask):
            self.last_attention_mask = attention_mask.detach().clone()
            return SimpleNamespace(pooler_output=torch.zeros((1, 3)))

    tokenizer = Tokenizer()
    tokenizer_2 = Tokenizer()
    t5 = T5()
    clip = CLIP()
    wrapper = object.__new__(PhasePipeline)
    wrapper._class_name = "FluxPipeline"
    wrapper._device = "cpu"
    wrapper._pipeline = SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=clip,
        text_encoder_2=t5,
    )

    prompt_embeds, pooled = wrapper._encode_flux1_main_conditioner("fox", {})

    assert tuple(prompt_embeds.shape) == (1, 512, 4)
    assert tuple(pooled.shape) == (1, 3)
    assert t5.last_attention_mask is not None
    assert clip.last_attention_mask is not None
    assert tokenizer_2.calls[0]["max_length"] == 512
    assert tokenizer.calls[0]["max_length"] == 77


def test_rss_guard_trips_at_phase_boundary(monkeypatch):
    wrapper = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    monkeypatch.setenv("RSS_LIMIT_MB", "1")
    with pytest.raises(MemoryError):
        wrapper.encode("goes over")


def test_cuda_placement_order_and_vram_guard(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cuda")
    moves: list[tuple[str, str]] = []
    monkeypatch.setattr(
        PhasePipeline,
        "_move_module",
        lambda self, name, module, target: (moves.append((name, target)), module)[1],
    )
    monkeypatch.setattr(PhasePipeline, "_empty_cuda_cache", lambda self: None)
    monkeypatch.setattr(PhasePipeline, "_vram_reserved_mb", lambda self: 123.0)

    with wrapper.encode_phase():
        pass
    # Denoiser + VAE leave the card before the encoder lands on it.
    assert moves == [("transformer", "cpu"), ("vae", "cpu"), ("text_encoder", "cuda")]
    assert wrapper.phase_events[-1]["vram_reserved_mb"] == 123.0

    moves.clear()
    with wrapper.denoise_phase():
        pass
    assert moves[0] == ("text_encoder", "cpu")
    assert ("transformer", "cuda") in moves and ("vae", "cuda") in moves
    assert pipe.text_encoder is None  # detached during denoise

    # Reserved VRAM above the agent-declared reservation fails closed.
    monkeypatch.setenv("VRAM_LIMIT_MB", "100")
    with pytest.raises(PhaseError, match="VRAM reserved 123"):
        with wrapper.encode_phase():
            pass


def test_generate_filters_unknown_kwargs():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    embeds = wrapper.encode("filter me")
    wrapper.generate(
        embeds,
        generator=torch.Generator().manual_seed(0),
        negative_prompt="not accepted by this pipeline",
    )
    assert pipe.calls  # no TypeError; unknown kwarg dropped

    with pytest.raises(PhaseError, match="manages 'prompt_embeds'"):
        wrapper.generate(embeds, prompt_embeds=torch.zeros(1))


def test_generation_records_post_operation_memory_snapshot():
    pipe = Flux2KleinPipeline()
    wrapper = PhasePipeline.wrap(pipe, device="cpu")
    embeds = wrapper.encode("memory snapshot")

    wrapper.generate(embeds)

    telemetry = wrapper.telemetry()
    assert telemetry.memory_event_count == 1
    assert telemetry.memory_allocated_mb is None
    assert telemetry.memory_reserved_mb is None
    assert telemetry.max_memory_allocated_mb is None
    assert wrapper._last_memory_snapshot["label"] == "generate"

    batch = wrapper.generate_batch(embeds, branch_ids=("a", "b"))
    assert batch.telemetry["label"] == "generate_batch"
    assert batch.telemetry["memory_event_count"] == 2
    assert wrapper.telemetry().memory_event_count == 2


def test_close_does_not_materialize_meta_component():
    """Closing a lazily dispatched pipeline must not call Module.to on meta state."""

    class MetaComponent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, device="meta"))

    encoder = MetaComponent()
    denoiser = _Component()
    pipeline = SimpleNamespace(text_encoder=encoder, transformer=denoiser, vae=None)
    wrapper = object.__new__(PhasePipeline)
    wrapper._pipeline = pipeline
    wrapper._encoders = {"text_encoder": encoder}
    wrapper._detached = {}
    wrapper._denoiser_name = "transformer"
    wrapper._device = "cpu"
    wrapper._device_embed_cache = None
    wrapper._resident = "denoise"

    wrapper.close()

    assert encoder.weight.is_meta
    assert denoiser.to_targets == ["cpu"]
    assert wrapper.resident_phase == "none"
