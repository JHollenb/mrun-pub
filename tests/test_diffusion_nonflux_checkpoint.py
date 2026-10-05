"""Focused mechanics checks for the native non-FLUX trajectory ABI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mrun.diffusion import (
    DiffusionTrajectoryCheckpoint,
    DiffusionTrajectoryState,
    PhaseError,
    PhasePipeline,
)
from mrun.diffusion.nonflux import (
    _chroma_timestep,
    _decode_sdxl,
    _prepare_scheduler,
    _restore_scheduler,
)


class _Component(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype


class _SDXLUNet(_Component):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__(dtype)
        self.cross_attention_history = []
        self.config = SimpleNamespace(
            in_channels=4,
            sample_size=4,
            time_cond_proj_dim=None,
        )

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        timestep_cond=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        return_dict=False,
    ):
        del timestep, timestep_cond, return_dict
        self.cross_attention_history.append(cross_attention_kwargs)
        value = encoder_hidden_states.mean() + added_cond_kwargs["text_embeds"].mean()
        return (sample * 0.05 + value * self.anchor * 0.001,)


class _SDXLVAE(_Component):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def decode(self, latents, return_dict=False):
        del return_dict
        return (latents,)


class _UpcastSDXLVAE(_SDXLVAE):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float16))
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1, bias=False).to(dtype=torch.float16)
        self.config.force_upcast = True
        self.seen_decode_dtype = None

    @property
    def dtype(self):
        return self.anchor.dtype

    def decode(self, latents, return_dict=False):
        del return_dict
        self.seen_decode_dtype = latents.dtype
        return (latents,)


class StableDiffusionXLPipeline:
    """Small SDXL-shaped organism using a real stochastic Euler scheduler."""

    def __init__(
        self,
        scheduler,
        *,
        unet_dtype: torch.dtype = torch.float32,
        high_level: bool = False,
    ) -> None:
        self.text_encoder = _Component()
        self.text_encoder_2 = _Component()
        self.unet = _SDXLUNet(unet_dtype)
        self.vae = _SDXLVAE()
        self.scheduler = scheduler
        self.vae_scale_factor = 1
        self.default_sample_size = 4
        self.image_processor = SimpleNamespace(postprocess=lambda value, output_type: value)
        self._high_level = high_level
        self._guidance_scale = 1.0
        self._guidance_rescale = 0.0
        self._interrupt = False

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=None,
    ):
        del prompt, device, lora_scale
        if prompt_embeds is None:
            prompt_embeds = torch.full((1, 2, 3), 1.0013)
            negative_prompt_embeds = torch.full_like(prompt_embeds, 0.2517)
            pooled_prompt_embeds = prompt_embeds.mean(dim=1)
            negative_pooled_prompt_embeds = negative_prompt_embeds.mean(dim=1)
        embed_dtype = (
            self.text_encoder_2.dtype if self.text_encoder_2 is not None else self.unet.dtype
        )
        prompt_embeds = prompt_embeds.to(dtype=embed_dtype)
        batch_size, sequence_length, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
            batch_size * num_images_per_prompt, sequence_length, -1
        )
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=embed_dtype)
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            ).view(batch_size * num_images_per_prompt, sequence_length, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            batch_size * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(batch_size * num_images_per_prompt, -1)
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=None,
        text_encoder_projection_dim=None,
    ):
        del original_size, crops_coords_top_left, target_size, text_encoder_projection_dim
        return torch.zeros(1, 6, dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        del generator
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return torch.randn(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        timesteps=None,
        sigmas=None,
        guidance_scale=7.5,
        guidance_rescale=0.0,
        eta=0.0,
        generator=None,
        latents=None,
        output_type="pil",
        cross_attention_kwargs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        **kwargs,
    ):
        if not self._high_level:
            raise AssertionError(f"unexpected native pipeline call: {kwargs}")
        del prompt, kwargs
        height = height or 4
        width = width or 4
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        device = torch.device("cpu")
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        try:
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
                retrieve_timesteps,
            )
        except ImportError:
            # The mechanics suite intentionally runs in mrun's lean default
            # environment, where Diffusers is optional.  Keep this fake
            # high-level path equivalent to retrieve_timesteps so the
            # high-level/native parity assertion still exercises the dtype
            # boundary without making the test optional.
            if timesteps is not None and sigmas is not None:
                raise AssertionError("toy high-level path received both schedules") from None
            schedule_kwargs = {"device": device}
            if timesteps is not None:
                schedule_kwargs["timesteps"] = timesteps
            elif sigmas is not None:
                schedule_kwargs["sigmas"] = sigmas
            else:
                schedule_kwargs["num_inference_steps"] = num_inference_steps
            self.scheduler.set_timesteps(**schedule_kwargs)
            timesteps = self.scheduler.timesteps
        else:
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                timesteps,
                sigmas,
            )
        latents = self.prepare_latents(
            1,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        add_time_ids = self._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=3,
        )
        cfg = guidance_scale > 1.0
        del eta
        if cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        else:
            add_text_embeds = pooled_prompt_embeds
        for timestep in timesteps:
            latent_model_input = torch.cat([latents, latents]) if cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            noise = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                return_dict=False,
            )[0]
            if cfg:
                noise_uncond, noise_text = noise.chunk(2)
                noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = self.scheduler.step(
                noise, timestep, latents, generator=generator, return_dict=False
            )[0]
        image = latents if output_type == "latent" else self.vae.decode(latents)[0]
        return SimpleNamespace(images=image)


def test_sdxl_euler_ancestral_cpu_generator_suffix_is_bitwise_exact() -> None:
    """A fresh scheduler and CPU generator resume the stochastic suffix exactly."""

    schedulers = pytest.importorskip("diffusers.schedulers").EulerAncestralDiscreteScheduler
    initial = torch.linspace(-1.0, 1.0, 4 * 4 * 4).reshape(1, 4, 4, 4)

    full_pipeline = StableDiffusionXLPipeline(schedulers(num_train_timesteps=100))
    full_phase = PhasePipeline.wrap(full_pipeline, device="cpu")
    full_embeds = full_phase.encode("a small test")
    full = full_phase.capture_checkpoint(
        full_embeds,
        cut_step=5,
        initial_latents=initial,
        num_inference_steps=5,
        height=4,
        width=4,
        guidance_scale=2.0,
        generator=torch.Generator(device="cpu").manual_seed(91),
        output_type="latent",
    )

    suffix_pipeline = StableDiffusionXLPipeline(schedulers(num_train_timesteps=100))
    suffix_phase = PhasePipeline.wrap(suffix_pipeline, device="cpu")
    suffix_embeds = suffix_phase.encode("a small test")
    cut = suffix_phase.capture_checkpoint(
        suffix_embeds,
        cut_step=2,
        initial_latents=initial,
        num_inference_steps=5,
        height=4,
        width=4,
        guidance_scale=2.0,
        generator=torch.Generator(device="cpu").manual_seed(91),
        output_type="latent",
    )
    assert cut.slot("rng.generator_device") == "cpu"
    assert cut.slot("rng.generator_state") is not None
    assert cut.state.metadata["scheduler_step_index"] == 2
    assert cut.state.metadata["scheduler_begin_index"] == 0

    replay = suffix_phase.resume_checkpoint(cut, output_type="latent").images
    assert torch.equal(replay, full.latents)
    assert suffix_pipeline.scheduler.begin_index == 0
    assert suffix_pipeline.scheduler.step_index == 5


def test_sdxl_manual_trajectory_matches_high_level_call_after_encoder_detach() -> None:
    """Native capture must mirror SDXL's post-encode dtype recast exactly."""

    schedulers = pytest.importorskip("diffusers.schedulers").EulerAncestralDiscreteScheduler
    pipeline = StableDiffusionXLPipeline(
        schedulers(num_train_timesteps=100),
        unet_dtype=torch.bfloat16,
        high_level=True,
    )
    phase = PhasePipeline.wrap(pipeline, device="cpu")
    embeds = phase.encode("a dtype-sensitive test")
    initial = torch.linspace(-1.0, 1.0, 4 * 4 * 4).reshape(1, 4, 4, 4)
    generator = torch.Generator(device="cpu").manual_seed(17)
    checkpoint = phase.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=initial,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=2.0,
        generator=generator,
        output_type="np",
    )
    replay = phase.resume_checkpoint(checkpoint, output_type="np").images

    direct_generator = torch.Generator(device="cpu").manual_seed(17)
    direct = phase.generate(
        embeds,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=2.0,
        generator=direct_generator,
        latents=initial,
        output_type="np",
    ).images

    assert pipeline.text_encoder_2 is None
    assert checkpoint.attention_kwargs is None
    assert all(value is None for value in pipeline.unet.cross_attention_history)
    assert torch.equal(replay, direct)


def test_sdxl_manual_trajectory_matches_toy_high_level_call_by_default() -> None:
    """Exercise the high-level/native parity contract without optional Diffusers."""

    pipeline = StableDiffusionXLPipeline(
        _ToySDXLScheduler(), unet_dtype=torch.bfloat16, high_level=True
    )
    phase = PhasePipeline.wrap(pipeline, device="cpu")
    embeds = phase.encode("a default-suite parity test")
    initial = torch.linspace(-1.0, 1.0, 4 * 4 * 4).reshape(1, 4, 4, 4)
    checkpoint = phase.capture_checkpoint(
        embeds,
        cut_step=2,
        initial_latents=initial,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=2.0,
        generator=torch.Generator(device="cpu").manual_seed(17),
        output_type="latent",
    )
    replay = phase.resume_checkpoint(checkpoint, output_type="latent").images

    direct = phase.generate(
        embeds,
        num_inference_steps=4,
        height=4,
        width=4,
        guidance_scale=2.0,
        generator=torch.Generator(device="cpu").manual_seed(17),
        latents=initial,
        output_type="latent",
    ).images

    assert pipeline.text_encoder_2 is None
    assert checkpoint.attention_kwargs is None
    assert all(value is None for value in pipeline.unet.cross_attention_history)
    assert torch.equal(replay, direct)


def test_nonflux_statecut_converter_round_trips_checkpoint_fingerprint() -> None:
    state = DiffusionTrajectoryState(
        "sdxl",
        {
            "latents": torch.ones(1, 2),
            "schedule_timesteps": torch.tensor([2.0, 1.0]),
            "condition_prompt_embeds": torch.zeros(1, 1, 2),
        },
        {"latent_layout": "nchw", "scheduler_step_index": 1},
    )
    checkpoint = DiffusionTrajectoryCheckpoint(
        checkpoint_id="traj-roundtrip",
        pipeline_class="StableDiffusionXLPipeline",
        step_index=1,
        total_steps=2,
        height=8,
        width=8,
        state=state,
        guidance_scale=2.0,
        attention_kwargs={"scale": 0.5},
        metadata={"capture": "test"},
    )

    payload = checkpoint.to_statecut_payload(model_identity={"model": "fake-sdxl"})
    restored = DiffusionTrajectoryCheckpoint.from_statecut_payload(payload)
    assert restored.fingerprint == checkpoint.fingerprint
    assert torch.equal(restored.slot("latents"), checkpoint.slot("latents"))


class _ToyFlowScheduler:
    """Flow scheduler with the same explicit-sigma/cursor contract as Diffusers."""

    def __init__(self) -> None:
        self.config = {
            "num_train_timesteps": 1000,
            "base_image_seq_len": 256,
            "max_image_seq_len": 6400,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
        self.order = 1
        self._step_index = None
        self._begin_index = 0

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        del mu
        if sigmas is None:
            count = int(num_inference_steps)
            sigmas = torch.linspace(1.0, 1.0 / count, count).tolist()
        values = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        self.sigmas = torch.cat([values, torch.zeros(1, device=device)])
        self.timesteps = values * 1000
        self._step_index = None

    def set_begin_index(self, index=0):
        self._begin_index = int(index)

    def step(self, model_output, timestep, sample, return_dict=False, generator=None):
        del timestep, return_dict, generator
        if self._step_index is None:
            self._step_index = self._begin_index
        self._step_index += 1
        return (sample - 0.01 * model_output,)


class _ToySDXLScheduler(_ToyFlowScheduler):
    """Small SDXL-shaped scheduler for the default no-Diffusers test lane."""

    def __init__(self) -> None:
        super().__init__()
        self.config.update({"_class_name": "ToySDXLScheduler"})
        self.init_noise_sigma = 1.0
        self.is_scale_input_called = False

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        del mu
        if sigmas is not None:
            values = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        else:
            values = torch.arange(
                int(num_inference_steps), 0, -1, dtype=torch.float32, device=device
            )
        self.timesteps = values
        self.sigmas = torch.ones(len(values) + 1, dtype=torch.float32, device=device)
        self.num_inference_steps = len(values)
        self._step_index = None
        self._begin_index = 0
        self.is_scale_input_called = False

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def scale_model_input(self, sample, timestep):
        del timestep
        if self._step_index is None:
            self._step_index = self._begin_index
        self.is_scale_input_called = True
        return sample


class _DynamicShiftScheduler(_ToyFlowScheduler):
    """Flow scheduler that transforms a caller-provided sigma request."""

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        if sigmas is None:
            return super().set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=None,
                mu=mu,
            )
        values = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        shift = float(mu if mu is not None else 1.0)
        values = shift * values / (1.0 + (shift - 1.0) * values)
        self.sigmas = torch.cat([values, torch.zeros(1, device=device)])
        self.timesteps = values * 1000
        self._step_index = None


class _HostScheduleSDXLScheduler(_ToySDXLScheduler):
    """Scheduler whose native contract keeps schedule tensors on CPU."""

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        del device
        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device="cpu",
            sigmas=sigmas,
            mu=mu,
        )


class _ExpandedScheduleScheduler(_ToyFlowScheduler):
    """Scheduler whose native resolved grid is longer than its request."""

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        values = self.timesteps
        extra = values[-1:] - 100.0
        self.timesteps = torch.cat([values, extra])
        self.sigmas = torch.cat(
            [self.sigmas[:-1], torch.full((1,), 0.05, device=device), self.sigmas[-1:]]
        )
        self._step_index = None


class _PositionalExpandedScheduleScheduler(_ToyFlowScheduler):
    """Expanded scheduler whose requested count is positional-only."""

    def set_timesteps(self, num_inference_steps, /, device=None):
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device)
        values = self.timesteps
        self.timesteps = torch.cat([values, values[-1:] - 100.0])
        self.sigmas = torch.cat(
            [self.sigmas[:-1], torch.full((1,), 0.05, device=device), self.sigmas[-1:]]
        )
        self._step_index = None


class _DefaultOnlyScheduler:
    def set_timesteps(self, num_inference_steps=None, device=None):
        del num_inference_steps, device


def test_explicit_scheduler_schedule_fails_closed_when_native_api_is_missing() -> None:
    pipeline = SimpleNamespace(scheduler=_DefaultOnlyScheduler())
    with pytest.raises(PhaseError, match="explicit sigma schedule"):
        _prepare_scheduler(
            pipeline,
            total_steps=2,
            device="cpu",
            timesteps=None,
            sigmas=[1.0, 0.5],
            mu=None,
        )


def test_dynamic_flow_sigma_request_persists_the_native_shifted_schedule() -> None:
    pipeline = SimpleNamespace(scheduler=_DynamicShiftScheduler())
    requested = [1.0, 0.67, 0.34, 0.0]
    timesteps, resolved_steps = _prepare_scheduler(
        pipeline,
        total_steps=4,
        device="cpu",
        timesteps=None,
        sigmas=requested,
        mu=1.15,
    )

    assert resolved_steps == 4
    assert len(pipeline.scheduler.sigmas) == 5
    assert not torch.equal(pipeline.scheduler.sigmas[:4], torch.tensor(requested))
    assert torch.equal(timesteps, pipeline.scheduler.sigmas[:4] * 1000)


def test_chroma_timestep_casts_before_normalizing_like_diffusers() -> None:
    timestep = torch.tensor(0.29411765933036804, dtype=torch.float32)
    native = timestep.expand(1).to(torch.bfloat16) / 1000.0
    divide_first = (timestep / 1000.0).expand(1).to(torch.bfloat16)

    actual = _chroma_timestep(timestep, batch_size=1, dtype=torch.bfloat16)

    assert torch.equal(actual, native)
    assert not torch.equal(actual, divide_first)


def test_scheduler_restore_preserves_native_schedule_tensor_placement() -> None:
    scheduler = _HostScheduleSDXLScheduler()
    pipeline = StableDiffusionXLPipeline(scheduler)
    phase = PhasePipeline.wrap(pipeline, device="cpu")
    checkpoint = phase.capture_checkpoint(
        phase.encode("host schedule"),
        cut_step=1,
        initial_latents=torch.ones(1, 4, 4, 4),
        num_inference_steps=3,
        height=4,
        width=4,
        guidance_scale=2.0,
        output_type="latent",
    )

    synthetic_phase = SimpleNamespace(pipeline=pipeline, _device=torch.device("meta"))
    _restore_scheduler(synthetic_phase, checkpoint)

    assert pipeline.scheduler.timesteps.device.type == "cpu"
    assert pipeline.scheduler.sigmas.device.type == "cpu"


def test_resolved_scheduler_length_defines_sdxl_checkpoint_boundary() -> None:
    initial = torch.linspace(-1.0, 1.0, 4 * 4 * 4).reshape(1, 4, 4, 4)
    full_phase = PhasePipeline.wrap(
        StableDiffusionXLPipeline(_ExpandedScheduleScheduler()), device="cpu"
    )
    full = full_phase.capture_checkpoint(
        full_phase.encode("expanded schedule"),
        cut_step=4,
        initial_latents=initial,
        num_inference_steps=3,
        height=4,
        width=4,
        guidance_scale=2.0,
        output_type="latent",
    )
    assert full.total_steps == 4

    suffix_phase = PhasePipeline.wrap(
        StableDiffusionXLPipeline(_ExpandedScheduleScheduler()), device="cpu"
    )
    cut = suffix_phase.capture_checkpoint(
        suffix_phase.encode("expanded schedule"),
        cut_step=2,
        initial_latents=initial,
        num_inference_steps=3,
        height=4,
        width=4,
        guidance_scale=2.0,
        output_type="latent",
    )
    assert cut.total_steps == 4
    replay = suffix_phase.resume_checkpoint(cut, output_type="latent").images
    assert torch.equal(replay, full.latents)


def test_positional_only_scheduler_restore_uses_requested_step_count() -> None:
    initial = torch.linspace(-1.0, 1.0, 4 * 4 * 4).reshape(1, 4, 4, 4)
    full_phase = PhasePipeline.wrap(
        StableDiffusionXLPipeline(_PositionalExpandedScheduleScheduler()), device="cpu"
    )
    full = full_phase.capture_checkpoint(
        full_phase.encode("positional expanded schedule"),
        cut_step=4,
        initial_latents=initial,
        num_inference_steps=3,
        height=4,
        width=4,
        guidance_scale=2.0,
        output_type="latent",
    )

    suffix_phase = PhasePipeline.wrap(
        StableDiffusionXLPipeline(_PositionalExpandedScheduleScheduler()), device="cpu"
    )
    cut = suffix_phase.capture_checkpoint(
        suffix_phase.encode("positional expanded schedule"),
        cut_step=2,
        initial_latents=initial,
        num_inference_steps=3,
        height=4,
        width=4,
        guidance_scale=2.0,
        output_type="latent",
    )
    assert cut.total_steps == 4
    assert cut.state.metadata["schedule_requested_num_inference_steps"] == 3
    replay = suffix_phase.resume_checkpoint(cut, output_type="latent").images
    assert torch.equal(replay, full.latents)


class _ToyTransformer(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(in_channels=in_channels)


class _ToyKreaTransformer(_ToyTransformer):
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels)
        self.encoder_history = []

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        position_ids=None,
        encoder_attention_mask=None,
        attention_kwargs=None,
        return_dict=False,
    ):
        del timestep, position_ids, attention_kwargs, return_dict
        self.encoder_history.append(encoder_hidden_states.detach().clone())
        mask_term = (
            encoder_attention_mask.float().mean() if encoder_attention_mask is not None else 0
        )
        return (hidden_states * 0.1 + encoder_hidden_states.mean() * self.anchor + mask_term,)


class _ToyChromaTransformer(_ToyTransformer):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        txt_ids=None,
        img_ids=None,
        attention_mask=None,
        joint_attention_kwargs=None,
        return_dict=False,
    ):
        del timestep, txt_ids, img_ids, joint_attention_kwargs, return_dict
        mask_term = attention_mask.float().mean() if attention_mask is not None else 0
        return (hidden_states * 0.1 + encoder_hidden_states.mean() * self.anchor + mask_term,)


class _ToyVAE(nn.Module):
    def __init__(self, *, five_dimensional: bool) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        if five_dimensional:
            self.config = SimpleNamespace(z_dim=1, latents_mean=[0.0], latents_std=[1.0])
        else:
            self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def decode(self, latents, return_dict=False):
        del return_dict
        return (latents,)


class Krea2Pipeline:
    def __init__(self) -> None:
        self.text_encoder = _Component()
        self.transformer = _ToyKreaTransformer(4)
        self.vae = _ToyVAE(five_dimensional=True)
        self.scheduler = _ToyFlowScheduler()
        self.vae_scale_factor = 1
        self.patch_size = 1
        self.config = SimpleNamespace(is_distilled=True)
        self.default_sample_size = 2
        self.image_processor = SimpleNamespace(postprocess=lambda value, output_type: value)

    def encode_prompt(self, prompt, device=None, num_images_per_prompt=1, max_sequence_length=512):
        del device, num_images_per_prompt, max_sequence_length
        value = 1.0 if prompt else 0.0
        return torch.full((1, 2, 1, 1), value), torch.ones(1, 2, dtype=torch.bool)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        del generator, height, width
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return torch.ones(batch_size, 4, 4, dtype=dtype, device=device)

    @staticmethod
    def prepare_position_ids(text_seq_len, grid_height, grid_width, device):
        return torch.zeros(text_seq_len + grid_height * grid_width, 3, device=device)

    @staticmethod
    def _unpack_latents(latents, height, width):
        del height, width
        return latents.unsqueeze(2)

    def __call__(self, **kwargs):  # pragma: no cover - explicit path is under test
        raise AssertionError(kwargs)


class ChromaPipeline:
    def __init__(self) -> None:
        self.text_encoder = _Component()
        self.transformer = _ToyChromaTransformer(16)
        self.vae = _ToyVAE(five_dimensional=False)
        self.scheduler = _ToyFlowScheduler()
        self.vae_scale_factor = 1
        self.default_sample_size = 2
        self.image_processor = SimpleNamespace(postprocess=lambda value, output_type: value)

    def encode_prompt(
        self, prompt, device=None, num_images_per_prompt=1, max_sequence_length=512, lora_scale=None
    ):
        del device, num_images_per_prompt, max_sequence_length, lora_scale
        value = 1.0 if prompt else 0.0
        positive = torch.full((1, 2, 4), value)
        negative = torch.zeros_like(positive)
        mask = torch.ones(1, 2, dtype=torch.bool)
        ids = torch.zeros(2, 3)
        return positive, ids, mask, negative, ids.clone(), mask.clone()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        del batch_size, num_channels_latents, height, width, generator
        if latents is None:
            latents = torch.ones(1, 4, 4, dtype=dtype, device=device)
        return latents.to(device=device, dtype=dtype), torch.zeros(4, 3, dtype=dtype, device=device)

    @staticmethod
    def _prepare_attention_mask(batch_size, sequence_length, dtype, attention_mask=None):
        del dtype
        return torch.cat(
            [attention_mask, torch.ones(batch_size, sequence_length, dtype=torch.bool)], dim=1
        )

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        del height, width, vae_scale_factor
        return latents.unsqueeze(2)

    def __call__(self, **kwargs):  # pragma: no cover - explicit path is under test
        raise AssertionError(kwargs)


def test_sdxl_conditioner_override_rebuilds_cfg_streams_from_one_row() -> None:
    pipeline = StableDiffusionXLPipeline(_ToyFlowScheduler())
    phase = PhasePipeline.wrap(pipeline, device="cpu")
    checkpoint = phase.capture_checkpoint(
        phase.encode("prompt"),
        cut_step=1,
        initial_latents=torch.ones(1, 4, 4, 4),
        num_inference_steps=3,
        height=4,
        width=4,
        guidance_scale=2.0,
        output_type="latent",
    )
    override = {"prompt_embeds": torch.full((1, 2, 3), 2.0)}
    result = phase.resume_checkpoint(
        checkpoint,
        prompt_embeds_override=override,
        output_type="latent",
    ).images
    assert result.shape == (1, 4, 4, 4)
    assert not torch.equal(result, checkpoint.slot("latents"))

    with pytest.raises(PhaseError, match="shape/dtype"):
        phase.resume_checkpoint(
            checkpoint,
            prompt_embeds_override={
                "condition_prompt_embeds": torch.ones(2, 2, 3),
            },
            output_type="latent",
        )


def test_sdxl_decode_mirrors_native_upcast_and_watermark() -> None:
    vae = _UpcastSDXLVAE()
    watermark = SimpleNamespace(apply_watermark=lambda image: image + 7.0)
    pipeline = SimpleNamespace(
        vae=vae,
        watermark=watermark,
        image_processor=SimpleNamespace(postprocess=lambda image, output_type: image),
    )
    checkpoint = SimpleNamespace()
    latents = torch.ones(1, 4, 2, 2, dtype=torch.float16)
    result = _decode_sdxl(pipeline, latents, checkpoint, "np")
    assert vae.seen_decode_dtype == torch.float32
    assert vae.dtype == torch.float16
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.full_like(result, 8.0))


def test_krea_and_chroma_capture_resume_clone_and_family_dispatch() -> None:
    initial = torch.ones(1, 4, 4)
    for pipeline, guidance, expected_family, expected_layout in (
        (Krea2Pipeline(), 2.0, "krea2", "packed_btd"),
        (ChromaPipeline(), 2.0, "chroma", "packed_btd"),
    ):
        phase = PhasePipeline.wrap(pipeline, device="cpu")
        embeds = phase.encode("prompt")
        checkpoint = phase.capture_checkpoint(
            embeds,
            cut_step=2,
            initial_latents=initial,
            num_inference_steps=4,
            height=2,
            width=2,
            guidance_scale=guidance,
            output_type="latent",
        )
        assert checkpoint.family == expected_family
        assert checkpoint.state.metadata["latent_layout"] == expected_layout
        assert checkpoint.slot("schedule_sigmas").shape[0] == 5
        assert checkpoint.slot("rng.global_cpu_state") is not None
        payload = checkpoint.to_statecut_payload(model_identity={"model": expected_family})
        restored = DiffusionTrajectoryCheckpoint.from_statecut_payload(
            payload,
            expected_family=expected_family,
            expected_model_identity={"model": expected_family},
        )
        assert restored.fingerprint == checkpoint.fingerprint
        if expected_family == "krea2":
            assert checkpoint.slot("condition_prompt_embeds_mask") is not None
            assert checkpoint.slot("condition_negative_prompt_embeds_mask") is not None
            # The parent is a post-prefix StateCut.  Regressions that leave
            # Krea's pre-loop initial-noise slot untouched make the first
            # resumed denoising step consume the wrong latent.
            assert not torch.equal(checkpoint.slot("latents"), initial)
        else:
            assert checkpoint.slot("layout_latent_image_ids") is not None
            assert checkpoint.slot("layout_attention_mask") is not None
            assert checkpoint.attention_kwargs == {}

        parent_fingerprint = checkpoint.fingerprint
        parent_latents = checkpoint.slot("latents").clone()
        child = phase.advance_checkpoint(checkpoint, steps=1)
        assert child.step_index == 3
        assert checkpoint.fingerprint == parent_fingerprint
        assert torch.equal(checkpoint.slot("latents"), parent_latents)
        assert phase.resume_checkpoint(child, output_type="latent").images.shape == initial.shape

        with pytest.raises(
            PhaseError, match="does not match|requires a DiffusionTrajectoryCheckpoint"
        ):
            other_type = ChromaPipeline if expected_family == "krea2" else Krea2Pipeline
            other = PhasePipeline.wrap(other_type(), device="cpu")
            other.resume_checkpoint(checkpoint, output_type="latent")


def test_advance_override_is_carried_into_child_resume() -> None:
    pipeline = Krea2Pipeline()
    phase = PhasePipeline.wrap(pipeline, device="cpu")
    checkpoint = phase.capture_checkpoint(
        phase.encode("prompt"),
        cut_step=1,
        initial_latents=torch.ones(1, 4, 4),
        num_inference_steps=3,
        height=2,
        width=2,
        guidance_scale=2.0,
        output_type="latent",
    )
    replacement = torch.full_like(checkpoint.slot("condition_prompt_embeds"), 3.0)
    child = phase.advance_checkpoint(
        checkpoint,
        steps=1,
        prompt_embeds_override={"prompt_embeds": replacement},
    )
    assert torch.equal(child.slot("condition_prompt_embeds"), replacement)
    pipeline.transformer.encoder_history.clear()
    phase.resume_checkpoint(child, output_type="latent")
    # The suffix contains both positive and negative denoiser calls.  The
    # positive call must consume the replacement carried by the child.
    assert any(torch.equal(value, replacement) for value in pipeline.transformer.encoder_history)


def test_sdxl_action_provider_skips_denoiser_but_keeps_native_scheduler() -> None:
    pipeline = StableDiffusionXLPipeline(_ToySDXLScheduler())
    phase = PhasePipeline.wrap(pipeline, device="cpu")
    checkpoint = phase.capture_checkpoint(
        phase.encode("prompt"),
        cut_step=0,
        initial_latents=torch.ones(1, 4, 4, 4),
        num_inference_steps=3,
        height=4,
        width=4,
        guidance_scale=1.0,
        output_type="latent",
    )
    native_events = []
    native_child = phase.advance_checkpoint(
        checkpoint,
        steps=1,
        step_observer=lambda event: native_events.append(dict(event)),
    )
    native_calls_before = len(pipeline.unet.cross_attention_history)
    provider_events = []
    observer_events = []

    def provide(event):
        provider_events.append(event)
        return native_events[0]["noise_pred"]

    child = phase.advance_checkpoint(
        checkpoint,
        steps=1,
        action_provider=provide,
        step_observer=lambda event: observer_events.append(dict(event)),
    )

    assert child.step_index == 1
    assert len(provider_events) == 1
    assert len(pipeline.unet.cross_attention_history) == native_calls_before
    assert provider_events[0]["model_input"].shape[0] == 1
    assert torch.equal(provider_events[0]["scaled_latents"], checkpoint.latents)
    assert observer_events[0]["action_source"] == "external-action-provider"
    assert torch.equal(observer_events[0]["noise_pred"], native_events[0]["noise_pred"])
    assert torch.equal(child.latents, native_child.latents)


def test_nonflux_scheduler_identity_is_checked_before_replay() -> None:
    first = Krea2Pipeline()
    phase = PhasePipeline.wrap(first, device="cpu", model_identity="model-a")
    checkpoint = phase.capture_checkpoint(
        phase.encode("prompt"),
        cut_step=1,
        initial_latents=torch.ones(1, 4, 4),
        num_inference_steps=3,
        height=2,
        width=2,
        guidance_scale=2.0,
        output_type="latent",
    )

    # Same scheduler class but a changed transition configuration must not be
    # accepted merely because it can produce the same number of timesteps.
    second = Krea2Pipeline()
    second.scheduler.config["max_shift"] = 0.25
    other = PhasePipeline.wrap(second, device="cpu", model_identity="model-a")
    with pytest.raises(PhaseError, match="scheduler configuration"):
        other.resume_checkpoint(checkpoint, output_type="latent")

    identity_pipeline = Krea2Pipeline()
    identity_phase = PhasePipeline.wrap(identity_pipeline, device="cpu", model_identity="model-b")
    with pytest.raises(PhaseError, match="model_identity"):
        identity_phase.resume_checkpoint(checkpoint, output_type="latent")
