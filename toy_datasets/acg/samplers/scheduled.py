"""Scheduled Sampler.

This module provides ScheduledSampler, which uses a SamplingScheduler
to control the sampling loop, enabling advanced strategies like
RePaint-style resampling and progressive retreat.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from acg.samplers.base import Sampler, SamplerOutput
from acg.samplers.scheduler import (
    BaseSamplingScheduler,
    SamplingStep,
    StandardDiffusionScheduler,
    StepType,
    compute_heat_noise_std,
)


class ScheduledSampler:
    """Sampler that uses a SamplingScheduler to control the sampling loop.

    This class combines a base sampler (e.g., HeunSampler) with a scheduler
    (e.g., AnnealedScheduler) to enable advanced sampling strategies.

    Parameters
    ----------
    base_sampler : Sampler
        The underlying sampler to use for individual COOL steps.
    scheduler : BaseSamplingScheduler, optional
        The scheduler that controls step ordering. Defaults to StandardDiffusionScheduler.

    Examples
    --------
    >>> from acg.samplers import HeunSampler, AnnealedScheduler, ScheduledSampler
    >>> base = HeunSampler()
    >>> scheduler = AnnealedScheduler(heat_jump=2, num_iterations=3)
    >>> sampler = ScheduledSampler(base, scheduler)
    >>> output = sampler.sample(denoiser, latents, sigmas)
    """

    def __init__(
        self,
        base_sampler: Sampler,
        scheduler: Optional[BaseSamplingScheduler] = None,
    ) -> None:
        self.base_sampler = base_sampler
        self.scheduler = scheduler or StandardDiffusionScheduler()

    def sample(
        self,
        denoiser: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        gammas: Optional[torch.Tensor] = None,
        step_scales: Optional[torch.Tensor] = None,
        noise_scales: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        step_callback: Optional[Callable[[SamplingStep, torch.Tensor], None]] = None,
        **kwargs: Any,
    ) -> SamplerOutput:
        """Run scheduled sampling.

        Parameters
        ----------
        denoiser : Callable
            Denoiser function D(x, sigma, ...).
        latents : torch.Tensor
            Initial noise of shape [B, ...].
        sigmas : torch.Tensor
            Noise schedule of shape [n_steps + 1].
        gammas : torch.Tensor, optional
            Gamma values for stochastic sampling.
        step_scales : torch.Tensor, optional
            Step scale factors.
        noise_scales : torch.Tensor, optional
            Noise scale factors.
        return_trajectory : bool, optional
            If True, record all intermediate states.
        step_callback : Callable, optional
            Called after each step with (step, x).
        **kwargs
            Additional arguments passed to denoiser.

        Returns
        -------
        SamplerOutput
            Sampling result.
        """
        # Convert to numpy for scheduler
        sigmas_np = sigmas.cpu().numpy() if isinstance(sigmas, torch.Tensor) else sigmas
        gammas_np = gammas.cpu().numpy() if isinstance(gammas, torch.Tensor) else gammas
        step_scales_np = (
            step_scales.cpu().numpy() if isinstance(step_scales, torch.Tensor) else step_scales
        )
        noise_scales_np = (
            noise_scales.cpu().numpy() if isinstance(noise_scales, torch.Tensor) else noise_scales
        )

        # Get schedule
        schedule = self.scheduler.get_schedule(
            sigmas_np, gammas_np, step_scales_np, noise_scales_np
        )

        x = latents
        trajectory: List[torch.Tensor] = []
        nfe = 0

        if return_trajectory:
            trajectory.append(x.clone())

        for sampling_step in schedule:
            if sampling_step.step_type == StepType.COOL:
                # Use base sampler for denoising step
                sigma = torch.tensor(
                    sampling_step.sigma_from, device=x.device, dtype=x.dtype
                )
                sigma_next = torch.tensor(
                    sampling_step.sigma_to, device=x.device, dtype=x.dtype
                )

                x = self.base_sampler.step(
                    denoiser=denoiser,
                    x=x,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    **kwargs,
                )

                # Count function evaluations (varies by sampler)
                # Heun uses 2 evals except for last step
                if sampling_step.sigma_to > 0:
                    nfe += 2
                else:
                    nfe += 1

            else:  # StepType.HEAT
                # Forward diffusion: add noise
                noise_std = compute_heat_noise_std(
                    sampling_step.sigma_from, sampling_step.sigma_to
                )
                x = x + noise_std * torch.randn_like(x)

            if return_trajectory:
                trajectory.append(x.clone())

            if step_callback is not None:
                step_callback(sampling_step, x)

        if return_trajectory:
            return SamplerOutput(
                samples=x,
                trajectories=torch.stack(trajectory, dim=1),
                sigmas=sigmas,
                num_function_evaluations=nfe,
            )

        return SamplerOutput(
            samples=x,
            sigmas=sigmas,
            num_function_evaluations=nfe,
        )

    def __repr__(self) -> str:
        return f"ScheduledSampler(base_sampler={self.base_sampler}, scheduler={self.scheduler})"
