"""Tests for pause/revert trajectory execution and branch suffix reuse."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mrun.diffusion import DiffusionProgram, PhasePipeline, TrajectoryCache
from mrun.diffusion.phase import (
    PhaseError,
    TrajectoryCheckpoint,
    _flux2_decode,
    _flux2_denoise_steps,
)


class Flux2KVLayerCache:
    def __init__(self) -> None:
        self.k_ref = None
        self.v_ref = None


class Flux2KVCache:
    def __init__(self, num_double_layers: int, num_single_layers: int) -> None:
        self.double_block_caches = [
            Flux2KVLayerCache() for _ in range(num_double_layers)
        ]
        self.single_block_caches = [
            Flux2KVLayerCache() for _ in range(num_single_layers)
        ]
        self.num_ref_tokens = 0


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    del image_seq_len, num_steps
    return 0.0


def retrieve_timesteps(scheduler, num_inference_steps, device=None, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, int(num_inference_steps)


def calculate_shift(
    image_seq_len: int,
    base_image_seq_len: int,
    max_image_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    del image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift
    return 0.0


def test_pre_pooled_checkpoint_fixture_preserves_positional_abi_and_fingerprint() -> None:
    """The original v1 positional payload and fingerprint remain reopenable."""

    checkpoint = TrajectoryCheckpoint(
        "pre-pooled-fixture",
        "Flux2KleinPipeline",
        1,
        4,
        8,
        8,
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3, 4], dtype=torch.int64),
        torch.tensor([5.0, 4.0], dtype=torch.float32),
        torch.tensor([[0.25, 0.5]], dtype=torch.float32),
        torch.tensor([[1, 2]], dtype=torch.int64),
        ("ref-a",),
        torch.tensor([0.75], dtype=torch.float32),
        torch.tensor([9], dtype=torch.int64),
        1,
        2.5,
        {"scale": 0.125},
        {"fixture": "pre-pooled-v1"},
        "mrun-diffusion-trajectory-checkpoint-v1",
    )

    assert checkpoint.reference_ids == ("ref-a",)
    assert checkpoint.reference_token_count == 1
    assert checkpoint.guidance_scale == 2.5
    assert checkpoint.pooled_prompt_embeds is None
    assert checkpoint.fingerprint == (
        "13c8287e6b108d7b76b4cf99990daa20b3c2d96f0036e74807b95b8c9d37c164"
    )


def test_native_kv_cache_is_deep_copied_and_byte_fingerprinted() -> None:
    cache = Flux2KVCache(1, 1)
    cache.num_ref_tokens = 3
    cache.double_block_caches[0].k_ref = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    cache.double_block_caches[0].v_ref = cache.double_block_caches[0].k_ref + 10
    checkpoint = TrajectoryCheckpoint(
        checkpoint_id="native-kv",
        pipeline_class="Flux2KleinKVPipeline",
        step_index=1,
        total_steps=2,
        height=8,
        width=8,
        latents=torch.zeros((1, 1, 1, 1)),
        latent_ids=torch.zeros((1, 1, 1)),
        timesteps=torch.ones((2,)),
        prompt_embeds=torch.zeros((1, 1, 1)),
        text_ids=torch.zeros((1, 1, 1)),
        reference_token_count=3,
        reference_latents=torch.zeros((1, 3, 1)),
        reference_latent_ids=torch.zeros((1, 3, 1)),
        reference_kv_cache=cache,
    )
    original_fingerprint = checkpoint.fingerprint
    stored = checkpoint.reference_kv_cache
    assert stored is not cache
    assert stored.double_block_caches[0] is not cache.double_block_caches[0]
    cache.double_block_caches[0].k_ref[0, 0, 0] = 999
    assert checkpoint.fingerprint == original_fingerprint


class _NativeKVTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.calls: list[dict[str, object]] = []
        self.config = SimpleNamespace(in_channels=4)

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, hidden_states, timestep, guidance, encoder_hidden_states,
                txt_ids, img_ids, joint_attention_kwargs, return_dict,
                kv_cache=None, kv_cache_mode=None, num_ref_tokens=None):
        del guidance, encoder_hidden_states, txt_ids, joint_attention_kwargs, return_dict
        self.calls.append({
            "mode": kv_cache_mode,
            "ids": img_ids.detach().clone(),
            "num_ref_tokens": num_ref_tokens,
        })
        if kv_cache_mode == "extract":
            assert int(num_ref_tokens) == 1
            cache = {"reference": hidden_states[:, : int(num_ref_tokens)].detach().clone()}
            target = hidden_states[:, int(num_ref_tokens):]
            return (target * 0.2 + timestep[:, None, None] * 0.001, cache)
        if kv_cache_mode == "cached":
            assert kv_cache is not None
            reference = kv_cache["reference"].mean(dim=1, keepdim=True)
            return (hidden_states * 0.2 + reference * 0.03 + timestep[:, None, None] * 0.001,)
        return (hidden_states * 0.2 + timestep[:, None, None] * 0.001,)


class Flux2KleinKVPipeline:
    """Small native-KV organism for the ref-first trajectory contract."""

    def __init__(self) -> None:
        self.transformer = _NativeKVTransformer()
        self.scheduler = _ToyScheduler()
        self._current_timestep = None
        self.vae = _ToyVAE()
        self.vae_scale_factor = 2
        self.image_processor = _ToyProcessor()

    @staticmethod
    def _unpack_latents_with_ids(x, x_ids):
        del x_ids
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], 1, x.shape[1])

    @staticmethod
    def _unpatchify_latents(latents):
        return latents


def test_native_kv_trajectory_is_ref_first_and_replay_uses_extracted_cache() -> None:
    pipeline = Flux2KleinKVPipeline()
    pipeline.scheduler.set_timesteps(2, device="cpu")
    latent_ids = torch.zeros((1, 2, 1), dtype=torch.float32)
    reference_ids = torch.ones((1, 1, 1), dtype=torch.float32)
    latents = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    reference = torch.tensor([[[9.0, 10.0]]])
    prompt = torch.ones((1, 2, 2), dtype=torch.float32)
    text_ids = torch.zeros((1, 2, 1), dtype=torch.float32)
    full_cache: dict[str, object] = {}
    full = _flux2_denoise_steps(
        pipeline,
        latents=latents.clone(),
        latent_ids=latent_ids,
        prompt_embeds=prompt,
        text_ids=text_ids,
        timesteps=pipeline.scheduler.timesteps,
        start_step=0,
        end_step=2,
        guidance_scale=1.0,
        attention_kwargs={},
        reference_latents=reference,
        reference_latent_ids=reference_ids,
        kv_cache_out=full_cache,
    )

    pipeline.scheduler.set_timesteps(2, device="cpu")
    prefix_cache: dict[str, object] = {}
    prefix = _flux2_denoise_steps(
        pipeline,
        latents=latents.clone(),
        latent_ids=latent_ids,
        prompt_embeds=prompt,
        text_ids=text_ids,
        timesteps=pipeline.scheduler.timesteps,
        start_step=0,
        end_step=1,
        guidance_scale=1.0,
        attention_kwargs={},
        reference_latents=reference,
        reference_latent_ids=reference_ids,
        kv_cache_out=prefix_cache,
    )
    replay = _flux2_denoise_steps(
        pipeline,
        latents=prefix,
        latent_ids=latent_ids,
        prompt_embeds=prompt,
        text_ids=text_ids,
        timesteps=pipeline.scheduler.timesteps,
        start_step=1,
        end_step=2,
        guidance_scale=1.0,
        attention_kwargs={},
        reference_kv_cache=prefix_cache["value"],
    )

    assert torch.equal(full, replay)
    assert pipeline.transformer.calls[-2]["mode"] == "extract"
    assert pipeline.transformer.calls[-2]["ids"].equal(torch.cat((reference_ids, latent_ids), dim=1))
    assert pipeline.transformer.calls[-1]["mode"] == "cached"


def test_native_kv_decode_uses_two_argument_id_scatter_abi() -> None:
    pipeline = Flux2KleinKVPipeline()
    latents = torch.zeros((1, 2, 1), dtype=torch.float32)
    ids = torch.zeros((1, 2, 1), dtype=torch.float32)
    decoded = _flux2_decode(
        pipeline,
        latents,
        ids,
        height=4,
        width=4,
        output_type="latent",
    )
    assert tuple(decoded.shape) == (1, 1, 1, 2)


class _ToyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.config = SimpleNamespace(in_channels=4)

    @property
    def dtype(self):
        return self.anchor.dtype

    def cache_context(self, name: str):
        assert name == "cond"
        return nullcontext()

    def forward(
        self,
        hidden_states,
        timestep,
        guidance,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        joint_attention_kwargs,
        return_dict,
    ):
        del guidance, txt_ids, img_ids, joint_attention_kwargs, return_dict
        condition = encoder_hidden_states.mean(dim=(-1, -2), keepdim=True)
        return (hidden_states * 0.2 + condition * 0.03 + timestep[:, None, None] * 0.001,)


class _ToyScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_flow_sigmas=False)
        self.timesteps = None
        # Constructor state deliberately differs from the inference schedule.
        # A fresh-process checkpoint replay must replace this vector.
        self.sigmas = torch.tensor([0.95, 0.80, 0.60, 0.30, 0.0])
        self.num_inference_steps = 4
        self._step_index = None
        self._begin_index = 0

    def set_timesteps(self, count: int, device=None, sigmas=None, **kwargs) -> None:
        del kwargs
        if sigmas is None:
            schedule = torch.linspace(1.0, 1.0 / count, count, device=device)
        else:
            schedule = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        if int(schedule.numel()) != int(count):
            raise ValueError("toy scheduler sigma count must equal the inference-step count")
        self.timesteps = schedule * 1000.0
        self.sigmas = torch.cat((schedule, torch.zeros(1, device=device)))
        self.num_inference_steps = int(count)
        self._step_index = None
        self._begin_index = 0

    def set_begin_index(self, index: int) -> None:
        self._begin_index = int(index)

    def step(self, noise_pred, timestep, sample, return_dict=False):
        del timestep, return_dict
        if self._step_index is None:
            self._step_index = int(self._begin_index)
        sigma = self.sigmas[self._step_index].to(sample.device, sample.dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(sample.device, sample.dtype)
        self._step_index += 1
        return (sample + (sigma_next - sigma) * noise_pred,)


class _ToyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bn = nn.BatchNorm2d(1)
        self.config = SimpleNamespace(batch_norm_eps=1e-5)

    @property
    def dtype(self):
        return self.anchor.dtype

    def decode(self, latents, return_dict=False):
        del return_dict
        return (latents * self.anchor,)


class _ToyProcessor:
    @staticmethod
    def postprocess(image, output_type="pil"):
        del output_type
        return [row.detach().clone() for row in image]


class Flux2KleinPipeline:
    """Tiny Flux2-shaped organism with the same packed-latent ABI."""

    def __init__(self) -> None:
        self.text_encoder = nn.Linear(1, 1)
        self.transformer = _ToyTransformer()
        self.vae = _ToyVAE()
        self.scheduler = _ToyScheduler()
        self.image_processor = _ToyProcessor()
        self.vae_scale_factor = 2
        self.default_sample_size = 2
        self._guidance_scale = 1.0
        self._attention_kwargs = {}
        self._execution_device = torch.device("cpu")

    def encode_prompt(self, prompt, device=None, num_images_per_prompt=1, max_sequence_length=512):
        del device, num_images_per_prompt, max_sequence_length
        return torch.full((1, 2, 3), float(len(prompt))), torch.zeros(1, 2, 4)

    @staticmethod
    def _prepare_text_ids(prompt_embeds):
        return torch.zeros(
            (prompt_embeds.shape[0], prompt_embeds.shape[1], 4),
            device=prompt_embeds.device,
        )

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator=None,
        latents=None,
    ):
        del generator
        raw_height = height // (self.vae_scale_factor * 2)
        raw_width = width // (self.vae_scale_factor * 2)
        if latents is None:
            latents = torch.zeros(
                (batch_size, num_latents_channels, raw_height, raw_width),
                dtype=dtype,
                device=device,
            )
        else:
            latents = latents.to(device=device, dtype=dtype)
        packed = latents.reshape(batch_size, num_latents_channels, -1).permute(0, 2, 1)
        ids = torch.zeros((batch_size, packed.shape[1], 4), device=device)
        return packed, ids

    @staticmethod
    def _unpack_latents_with_ids(x, x_ids, height=None, width=None):
        del x_ids
        height = int(height)
        width = int(width)
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)

    @staticmethod
    def _unpatchify_latents(latents):
        return latents

    @property
    def do_classifier_free_guidance(self):
        return False

    def __call__(
        self,
        prompt=None,
        prompt_embeds=None,
        latents=None,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
        output_type="pil",
        return_dict=True,
        **kwargs,
    ):
        del prompt, kwargs, return_dict
        prompt_embeds = prompt_embeds.to("cpu")
        text_ids = self._prepare_text_ids(prompt_embeds)
        latent_ids, prepared = None, None
        prepared, latent_ids = self.prepare_latents(
            1,
            1,
            height,
            width,
            prompt_embeds.dtype,
            "cpu",
            latents=latents,
        )
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        self._guidance_scale = guidance_scale
        current = _flux2_denoise_steps(
            self,
            latents=prepared,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            timesteps=self.scheduler.timesteps,
            start_step=0,
            end_step=num_inference_steps,
            guidance_scale=guidance_scale,
            attention_kwargs={},
        )
        return SimpleNamespace(
            images=_flux2_decode(
                self,
                current,
                latent_ids,
                height=height,
                width=width,
                output_type=output_type,
            )
        )


class _Flux1Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.config = SimpleNamespace(in_channels=4, guidance_embeds=True)

    @property
    def dtype(self):
        return self.anchor.dtype

    def cache_context(self, name: str):
        assert name == "cond"
        return nullcontext()

    def forward(
        self,
        hidden_states,
        timestep,
        guidance,
        pooled_projections,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        joint_attention_kwargs,
        return_dict,
    ):
        del txt_ids, img_ids, joint_attention_kwargs, return_dict
        condition = encoder_hidden_states.mean(dim=(-1, -2), keepdim=True)
        condition = condition + pooled_projections.mean(dim=-1, keepdim=True)[:, :, None]
        if guidance is not None:
            condition = condition + guidance[:, None, None] * 0.001
        return (hidden_states * 0.2 + condition * 0.03 + timestep[:, None, None] * 0.001,)


class FluxPipeline:
    """Tiny FLUX.1-shaped organism with dual conditioner streams."""

    def __init__(self) -> None:
        self.text_encoder = nn.Linear(1, 1)
        self.text_encoder_2 = nn.Linear(1, 1)
        self.transformer = _Flux1Transformer()
        self.vae = _ToyVAE()
        self.vae.config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.25)
        self.scheduler = _ToyScheduler()
        self.image_processor = _ToyProcessor()
        self.vae_scale_factor = 2
        self.default_sample_size = 2
        self._guidance_scale = 3.5
        self._attention_kwargs = {}
        self._execution_device = torch.device("cpu")

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt=1,
        max_sequence_length=512,
        lora_scale=None,
    ):
        del device, num_images_per_prompt, max_sequence_length, lora_scale
        value = float(len(prompt))
        return (
            torch.full((1, 2, 3), value),
            torch.full((1, 5), value * 0.5),
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator=None,
        latents=None,
    ):
        del generator
        raw_height = 2 * (height // (self.vae_scale_factor * 2))
        raw_width = 2 * (width // (self.vae_scale_factor * 2))
        if latents is None:
            latents = torch.zeros(
                (batch_size, num_channels_latents, raw_height, raw_width),
                dtype=dtype,
                device=device,
            )
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, raw_height, raw_width
            )
        else:
            latents = latents.to(device=device, dtype=dtype)
        ids = torch.zeros((raw_height // 2 * (raw_width // 2), 3), device=device, dtype=dtype)
        return latents, ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        del vae_scale_factor
        raw_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        raw_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        batch_size = latents.shape[0]
        channels = latents.shape[2] // 4
        latents = latents.view(
            batch_size, raw_height // 2, raw_width // 2, channels, 2, 2,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels, raw_height, raw_width)

    def __call__(
        self,
        prompt=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        **kwargs,
    ):
        del prompt, prompt_embeds, pooled_prompt_embeds, kwargs
        return SimpleNamespace(images=[torch.zeros((1, 1, 1, 1))])

    @property
    def do_classifier_free_guidance(self):
        return False


def _kwargs(latents):
    return {
        "latents": latents,
        "num_inference_steps": 4,
        "height": 4,
        "width": 4,
        "guidance_scale": 1.0,
        "output_type": "pil",
    }


def test_pause_revert_and_exact_suffix_replay_share_prefix() -> None:
    phase = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu", device_embed_cache=True)
    embeds = phase.encode("shared prefix")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)

    scalar = phase.generate(embeds, **_kwargs(latents))
    checkpoint = phase.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    replay = phase.resume_checkpoint_batch(
        checkpoint,
        branch_ids=("control", "reverted"),
        mode="exact",
        output_type="pil",
    )

    assert torch.equal(replay.rows_by_branch()["control"], scalar.images[0])
    assert torch.equal(replay.rows_by_branch()["reverted"], scalar.images[0])
    assert replay.telemetry["shared_prefix_reused"] is True
    assert replay.telemetry["shared_prefix_steps"] == 2
    assert replay.telemetry["physical_suffix_denoiser_calls"] == 4
    assert replay.telemetry["revert_count"] == 2


def test_flux2_checkpoint_rehydrates_scheduler_in_a_fresh_pipeline() -> None:
    source = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    embeds = source.encode("durable scheduler state")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)
    checkpoint = source.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    expected = source.resume_checkpoint(checkpoint, output_type="latent").images

    assert checkpoint.metadata["scheduler_state_schema"] == (
        "mrun-flux2-scheduler-state-v1"
    )
    assert checkpoint.metadata["schedule_sigmas"] == [1.0, 0.75, 0.5, 0.25]
    assert checkpoint.metadata["scheduler_sigmas"] == [1.0, 0.75, 0.5, 0.25, 0.0]

    fresh = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    assert fresh.pipeline.scheduler.sigmas.tolist() != checkpoint.metadata["scheduler_sigmas"]
    replay = fresh.resume_checkpoint(checkpoint, output_type="latent").images
    assert torch.equal(replay, expected)

    expected_child = source.advance_checkpoint(checkpoint, steps=1)
    fresh_child = fresh.advance_checkpoint(checkpoint, steps=1)
    assert torch.equal(fresh_child.latents, expected_child.latents)

    batched_phase = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    batched = batched_phase.resume_checkpoint_batch(
        checkpoint,
        branch_ids=("a", "b"),
        mode="batched",
        output_type="latent",
    )
    assert torch.equal(batched.rows_by_branch()["a"], expected[0])
    assert torch.equal(batched.rows_by_branch()["b"], expected[0])


def test_flux2_checkpoint_legacy_default_recovers_but_custom_schedule_fails_closed() -> None:
    source = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    embeds = source.encode("legacy scheduler state")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)
    default_checkpoint = source.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    scheduler_fields = {
        "scheduler_state_schema",
        "scheduler_config_fingerprint",
        "scheduler_num_inference_steps",
        "schedule_sigmas",
        "scheduler_sigmas",
    }
    legacy = replace(
        default_checkpoint,
        metadata={
            key: value
            for key, value in default_checkpoint.metadata.items()
            if key not in scheduler_fields
        },
    )
    expected = source.resume_checkpoint(default_checkpoint, output_type="latent").images
    fresh = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    assert torch.equal(fresh.resume_checkpoint(legacy, output_type="latent").images, expected)

    custom = source.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
        sigmas=[1.0, 0.8, 0.4, 0.1],
    )
    legacy_custom = replace(
        custom,
        metadata={
            key: value for key, value in custom.metadata.items() if key not in scheduler_fields
        },
    )
    with pytest.raises(PhaseError, match="legacy canonical scheduler schedule"):
        PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu").resume_checkpoint(
            legacy_custom,
            output_type="latent",
        )


def test_step_observer_exposes_native_action_and_register_transition() -> None:
    phase = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    embeds = phase.encode("observable trajectory")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)
    observations: list[dict[str, torch.Tensor | int]] = []

    def observe(row):
        observations.append(
            {
                "step_index": int(row["step_index"]),
                "latents_before": row["latents_before"].detach().clone(),
                "noise_pred": row["noise_pred"].detach().clone(),
                "latents_after": row["latents_after"].detach().clone(),
            }
        )

    checkpoint = phase.capture_checkpoint(
        embeds,
        cut_step=4,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
        step_observer=observe,
    )

    assert [row["step_index"] for row in observations] == [0, 1, 2, 3]
    assert observations[0]["latents_before"].shape == checkpoint.latents.shape
    assert torch.equal(observations[-1]["latents_after"], checkpoint.latents)
    for row in observations:
        assert row["noise_pred"].shape == row["latents_before"].shape
        assert not torch.equal(row["latents_before"], row["latents_after"])


def test_advance_checkpoint_is_one_step_and_matches_prefix_capture() -> None:
    phase = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    embeds = phase.encode("stepwise trajectory")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)
    checkpoint = phase.capture_checkpoint(
        embeds,
        cut_step=1,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    observations: list[int] = []

    child = phase.advance_checkpoint(
        checkpoint,
        step_observer=lambda row: observations.append(int(row["step_index"])),
    )
    expected = phase.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )

    assert child.step_index == 2
    assert child.total_steps == checkpoint.total_steps
    assert child.checkpoint_id != checkpoint.checkpoint_id
    assert observations == [1]
    assert torch.equal(child.latents, expected.latents)
    assert torch.equal(checkpoint.latents, phase.capture_checkpoint(
        embeds,
        cut_step=1,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    ).latents)

    with pytest.raises(PhaseError, match="trajectory ends"):
        phase.advance_checkpoint(checkpoint, steps=4)


def test_prompt_embeds_override_changes_only_conditioner_suffix() -> None:
    phase = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    source = phase.encode("source")
    donor = phase.encode("donor")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)
    checkpoint = phase.capture_checkpoint(
        source,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    checkpoint_latent = checkpoint.latents.clone()
    checkpoint_conditioner = checkpoint.prompt_embeds.clone()

    source_result = phase.resume_checkpoint(checkpoint, output_type="pil")
    donor_result = phase.resume_checkpoint(
        checkpoint,
        prompt_embeds_override=donor,
        output_type="pil",
    )

    assert not torch.equal(source_result.images[0], donor_result.images[0])
    assert torch.equal(checkpoint.latents, checkpoint_latent)
    assert torch.equal(checkpoint.prompt_embeds, checkpoint_conditioner)
    with pytest.raises(PhaseError, match="dtype"):
        phase.resume_checkpoint(
            checkpoint,
            prompt_embeds_override=donor.tensors["prompt_embeds"].double(),
            output_type="pil",
        )


def test_flux1_checkpoint_preserves_dual_conditioner_and_all_branch_features() -> None:
    phase = PhasePipeline.wrap(FluxPipeline(), device="cpu", device_embed_cache=True)
    source = phase.encode("source")
    donor = phase.encode("donor")
    latents = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    observations: list[int] = []

    checkpoint = phase.capture_checkpoint(
        source,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
        step_observer=lambda row: observations.append(int(row["step_index"])),
    )
    assert checkpoint.pipeline_class == "FluxPipeline"
    assert checkpoint.pooled_prompt_embeds is not None
    assert checkpoint.metadata["conditioner_streams"] == [
        "prompt_embeds",
        "pooled_prompt_embeds",
    ]
    assert observations == [0, 1]

    # A cut at zero is the scalar authority for the complete suffix. The
    # shared FLUX.1 path must produce the same result from a cut at two.
    full = phase.capture_checkpoint(
        source,
        cut_step=0,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    expected = phase.resume_checkpoint(full, output_type="latent")
    replay = phase.resume_checkpoint(checkpoint, output_type="latent")
    assert torch.equal(replay.images, expected.images)

    child = phase.advance_checkpoint(checkpoint, steps=1)
    expected_child = phase.capture_checkpoint(
        source,
        cut_step=3,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    assert child.step_index == 3
    assert torch.equal(child.latents, expected_child.latents)
    assert torch.equal(child.pooled_prompt_embeds, checkpoint.pooled_prompt_embeds)

    donor_replay = phase.resume_checkpoint(
        checkpoint,
        prompt_embeds_override=donor,
        output_type="latent",
    )
    assert not torch.equal(donor_replay.images, replay.images)

    exact = phase.resume_checkpoint_batch(
        checkpoint,
        branch_ids=("control", "reverted"),
        mode="exact",
        output_type="latent",
    )
    batched = phase.resume_checkpoint_batch(
        checkpoint,
        branch_ids=("control", "reverted"),
        mode="batched",
        output_type="latent",
    )
    assert torch.equal(exact.rows_by_branch()["control"], replay.images[0])
    assert torch.equal(batched.rows_by_branch()["control"], replay.images[0])
    assert exact.telemetry["numerical_contract"] == "scalar-authority"
    assert batched.telemetry["numerical_contract"] == "batch-dependent"

    with pytest.raises(PhaseError, match="both prompt_embeds"):
        phase.resume_checkpoint(
            checkpoint,
            prompt_embeds_override={"prompt_embeds": donor.tensors["prompt_embeds"]},
            output_type="latent",
        )


def test_trajectory_cache_reuses_declared_prefix_and_invalidates_edit_reference() -> None:
    cache = TrajectoryCache()
    phase = PhasePipeline.wrap(
        Flux2KleinPipeline(),
        device="cpu",
        trajectory_cache=cache,
        model_identity="toy-flux2-revision",
    )
    embeds = phase.encode("cached prefix")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)
    key = TrajectoryCache.make_key(
        model_identity="toy-flux2-revision",
        conditioning_key=embeds.key,
        schedule_fingerprint="toy-schedule",
        cut_step=2,
        resolution=(4, 4),
        references=("reference:one",),
        initial_latent_fingerprint="latent:zero",
        trajectory_abi="Flux2KleinPipeline:toy-v2",
    )
    first = phase.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
        cache_key=key,
        schedule_fingerprint="toy-schedule",
        references=("reference:one",),
        dependency_keys=("edit:one",),
    )
    second = phase.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
        cache_key=key,
        schedule_fingerprint="toy-schedule",
        references=("reference:one",),
        dependency_keys=("edit:one",),
    )
    assert first.fingerprint == second.fingerprint
    assert phase.trajectory_cache_stats()["hits"] == 1
    alternate_key = TrajectoryCache.make_key(
        model_identity="toy-flux2-revision",
        conditioning_key=embeds.key,
        schedule_fingerprint="toy-schedule",
        cut_step=2,
        resolution=(4, 4),
        references=("reference:one",),
        initial_latent_fingerprint="latent:one",
        trajectory_abi="Flux2KleinPipeline:toy-v2",
    )
    alternate = phase.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=latents + 1,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
        cache_key=alternate_key,
        schedule_fingerprint="toy-schedule",
        references=("reference:one",),
        dependency_keys=("edit:one",),
    )
    assert alternate.fingerprint != first.fingerprint
    assert phase.invalidate_trajectory_cache(("reference:one",)) == (key, alternate_key)
    assert phase.trajectory_cache_stats()["invalidations"] == 2


def test_program_pause_checkpoint_and_non_mutating_branch_replay() -> None:
    phase = PhasePipeline.wrap(Flux2KleinPipeline(), device="cpu")
    program = DiffusionProgram.from_backend(phase, base_fingerprint="toy-flux2")
    link = program.link(schedule_fingerprint="toy-schedule")
    session = link.open_session(session_id="control", total_steps=4, resolution=(4, 4))
    session.compile_context("a reversible branch")
    latents = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1)

    checkpoint = session.pause(
        cut_step=2,
        initial_latents=latents,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=1.0,
    )
    assert checkpoint.metadata()["has_backend_checkpoint"] is True
    before = session.state
    replay = link.replay_batch(
        checkpoint,
        branch_ids=("left", "right"),
        mode="exact",
        output_type="pil",
    )

    assert replay.branch_ids == ("left", "right")
    assert replay.telemetry["program_replay"] is True
    assert session.state == before  # branch exploration does not commit the source row

    resumed = session.resume(checkpoint, output_type="pil")
    assert resumed.state_after.status == "completed"
    assert resumed.telemetry["program_replay"] is True
