import importlib
from typing import Any, Callable
from dataclasses import dataclass

import numpy as np
import PIL
import torch

from diffusers import Flux2KleinPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from ..guidance import build_guidance_method
from ..guidance import CFGContext

xm = None
if is_torch_xla_available():
    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
    except (ImportError, ModuleNotFoundError):
        xm = None

XLA_AVAILABLE = xm is not None

@dataclass
class FluxStepState:
    prompt_embeds: torch.Tensor
    latent_ids: torch.Tensor
    text_ids: torch.Tensor
    neg_prompt_embeds: torch.Tensor
    neg_text_ids: torch.Tensor
    image_latents: torch.Tensor | None
    image_latent_ids: torch.Tensor | None

class Flux2KleinPipelineCustomGuidance(Flux2KleinPipeline):
    # Important: need to have is_distilled = False,
    # which is the case for Flux2Klein base 4B in checkpoint: a3b4f4849157f664bdbc776fd7453c2783562f4d
    # (Otherwise the guidance is included directly in the transofrmer network,
    # so there is no explicit unconditionnal and conditional guidance)

    def __init__(self, scheduler, vae, text_encoder, tokenizer, transformer, is_distilled=False):
        super().__init__(scheduler=scheduler, vae=vae, 
                         text_encoder=text_encoder, 
                         tokenizer=tokenizer,
                         transformer=transformer,
                         is_distilled=is_distilled)

        # set default value
        self.guidance_type = "constant"
        self.guidance_method = None
        self.guidance_method_initialized = False

    def configure_guidance(self, guidance_type: str = "constant", guidance_params: dict | None = None):
        """
        Configure the guidance method chosen.

        Args:
            guidance_type (str): Name of the guidance method.
            guidance_params (dict, optional): Parameters specific to the chosen guidance method.
        """
        self.guidance_type = guidance_type
        self.guidance_method = build_guidance_method(guidance_type, guidance_params)
        self.guidance_method_initialized = True
        return

    def _predict_model(self, latents: torch.Tensor, t: torch.Tensor, do_cfg: bool, use_original: bool = False):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        latent_model_input = latents.to(self.transformer.dtype)
        latent_image_ids = self.flux_ctx.latent_ids

        if self.flux_ctx.image_latents is not None:
            latent_model_input = torch.cat([latents, self.flux_ctx.image_latents], dim=1).to(self.transformer.dtype)
            latent_image_ids = torch.cat([self.flux_ctx.latent_ids, self.flux_ctx.image_latent_ids], dim=1)

        # cond FIRST — mirrors diffusers order
        with self.transformer.cache_context("cond"):
            noise_pred_cond = self.transformer(hidden_states=latent_model_input, timestep=timestep / 1000,
                                               guidance=None, encoder_hidden_states=self.flux_ctx.prompt_embeds,
                                               txt_ids=self.flux_ctx.text_ids, img_ids=latent_image_ids,
                                               joint_attention_kwargs=self.attention_kwargs,return_dict=False)[0]
            
        noise_pred_cond = noise_pred_cond[:, : latents.size(1)]

        if not do_cfg:
            return noise_pred_cond

        # uncond SECOND
        with self.transformer.cache_context("uncond"):
            noise_pred_uncond = self.transformer(hidden_states=latent_model_input, timestep=timestep / 1000,
                                                 guidance=None, encoder_hidden_states=self.flux_ctx.neg_prompt_embeds,
                                                 txt_ids=self.flux_ctx.neg_text_ids, img_ids=latent_image_ids,
                                                 joint_attention_kwargs=self.attention_kwargs, return_dict=False)[0]
            
        noise_pred_uncond = noise_pred_uncond[:, : latents.size(1)]

        return noise_pred_uncond, noise_pred_cond

    @torch.no_grad()
    def __call__(
        self,
        image: list[PIL.Image.Image] | PIL.Image.Image | None = None,
        prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: str | list[str] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
    ):
        r"""
        Run Flux2Klein inference with custom guidance.

        Compared to the base Flux2KleinPipeline, the guidance step is
        modified via `guidance_method.predict_velocity_field`, enabling
        plug-and-play test-time control methods.

        Args:
            image (`PIL.Image.Image` or `list[PIL.Image.Image]`, *optional*):
                Reference image(s) for conditioning.
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*):
                Height in pixels of the generated image.
            width (`int`, *optional*):
                Width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps.
            sigmas (`list[float]`, *optional*):
                Custom sigmas for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Ignored for distilled models.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format: `"pil"` or `"np"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `Flux2PipelineOutput` or a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Extra kwargs forwarded to the `AttentionProcessor`.
            callback_on_step_end (`Callable`, *optional*):
                Called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                Tensor inputs forwarded to `callback_on_step_end`.
            max_sequence_length (`int`, defaults to 512):
                Maximum token sequence length for the prompt.
            text_encoder_out_layers (`tuple[int]`):
                Layer indices used to build the final prompt embeddings.

        Returns:
            [`~pipelines.flux2.Flux2PipelineOutput`] or `tuple`.
        """
        if not self.guidance_method_initialized:
            raise ValueError(f"Guidance method not initialized. Please call `configure_guidance` with the desired guidance method and its parameters before running the pipeline.")

        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode text
        prompt_embeds, text_ids = self.encode_prompt(prompt=prompt,prompt_embeds=prompt_embeds,
                                                    device=device, num_images_per_prompt=num_images_per_prompt,
                                                    max_sequence_length=max_sequence_length,
                                                    text_encoder_out_layers=text_encoder_out_layers)

        negative_prompt_embeds_tensor = None
        negative_text_ids = None
        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds_tensor, negative_text_ids = self.encode_prompt(
                                                                    prompt=negative_prompt,
                                                                    prompt_embeds=negative_prompt_embeds,
                                                                    device=device,
                                                                    num_images_per_prompt=num_images_per_prompt,
                                                                    max_sequence_length=max_sequence_length,
                                                                    text_encoder_out_layers=text_encoder_out_layers)

        # 4. Process reference images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(batch_size=batch_size * num_images_per_prompt, 
                                                   num_latents_channels=num_channels_latents,
                                                   height=height, width=width, dtype=prompt_embeds.dtype,
                                                   device=device, generator=generator, latents=latents)

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(images=condition_images, 
                                                                         batch_size=batch_size * num_images_per_prompt,
                                                                         generator=generator, device=device, dtype=self.vae.dtype)

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # set FluxStepState
        self.flux_ctx = FluxStepState(prompt_embeds = prompt_embeds,
                                    latent_ids =  latent_ids,
                                    text_ids =text_ids,
                                    neg_prompt_embeds=negative_prompt_embeds_tensor,
                                    neg_text_ids=negative_text_ids,
                                    image_latents=image_latents,
                                    image_latent_ids=image_latent_ids)

        # 7. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # build per-step guidance context
                guidance_ctx = CFGContext(pipeline=self, latents=latents, t=t, 
                                          timestep=timestep, step_index=i, 
                                          timesteps=timesteps, guidance_scale=self.guidance_scale)

                # if CFG option: use the method predict_velocity_field
                # of the GuidanceMethod (override for each guidance method)
                if self.do_classifier_free_guidance:
                    noise_pred = self.guidance_method.predict_velocity_field(guidance_ctx)
                else:
                    # if no CFG option, just predict the noisy velocity field
                    noise_pred = self._predict_model(latents=guidance_ctx.latents, t=guidance_ctx.t, do_cfg=False)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug:
                        # https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

            torch.cuda.empty_cache()

        self._current_timestep = None

        # Decode latents
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)