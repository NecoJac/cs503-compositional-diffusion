"""Heun Sampler (Second-Order).

This module provides the Heun sampler, which is the recommended default
sampler in the EDM paper. It is a second-order method that provides
better accuracy than Euler at the cost of 2x function evaluations.

The Heun method uses a predictor-corrector scheme:
1. Euler prediction: x' = x + d * dt
2. Corrector: x = x + (d + d') / 2 * dt

Reference:
    Karras et al. 2022 - "Elucidating the Design Space of Diffusion-Based
    Generative Models"
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from acg.samplers.base import CFGMixin, Sampler, SamplerOutput


def sample_heun(
    denoiser: Callable[..., torch.Tensor],
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    return_trajectory: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Heun sampling (second-order).

    Uses predictor-corrector scheme for improved accuracy.
    Recommended default sampler for EDM models.

    Parameters
    ----------
    denoiser : Callable
        Denoiser function D(x, sigma, ...).
    latents : torch.Tensor
        Initial noise of shape [B, ...].
    sigmas : torch.Tensor
        Noise schedule of shape [n_steps + 1], ending with 0.
    condition : torch.Tensor, optional
        Condition tensor for conditional generation.
    guidance_scale : float, optional
        CFG scale. 1.0 = no guidance. Default: 1.0.
    return_trajectory : bool, optional
        If True, returns all intermediate states.
    **kwargs
        Additional arguments passed to denoiser.

    Returns
    -------
    torch.Tensor
        Generated samples of shape [B, ...].
        If return_trajectory=True, returns [B, n_steps, ...].
    """
    x = latents
    trajectory: List[torch.Tensor] = []

    def get_denoised(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Get denoised prediction with optional CFG."""
        if condition is not None and guidance_scale != 1.0:
            d_cond = denoiser(x_in, sigma, condition=condition, **kwargs)
            d_uncond = denoiser(x_in, sigma, condition=None, **kwargs)
            return d_uncond + guidance_scale * (d_cond - d_uncond)
        elif condition is not None:
            return denoiser(x_in, sigma, condition=condition, **kwargs)
        else:
            return denoiser(x_in, sigma, **kwargs)

    if return_trajectory:
        trajectory.append(x.clone())

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        # Get first derivative
        denoised = get_denoised(x, sigma)
        d = (x - denoised) / sigma

        # Euler prediction
        x_pred = x + d * dt

        # Corrector step (skip for last step when sigma_next = 0)
        if sigma_next > 0:
            denoised_next = get_denoised(x_pred, sigma_next)
            d_next = (x_pred - denoised_next) / sigma_next

            # Average the derivatives
            d_avg = (d + d_next) / 2
            x = x + d_avg * dt
        else:
            x = x_pred

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=1)

    return x


class HeunSampler(Sampler, CFGMixin):
    """Heun sampler (second-order, deterministic).

    The recommended default sampler for EDM models.
    Provides better accuracy than Euler at the cost of 2x function evaluations.

    Parameters
    ----------
    guidance_scale : float, optional
        CFG scale for conditional sampling. Default: 1.0.

    Examples
    --------
    >>> sampler = HeunSampler(guidance_scale=7.5)
    >>> sigmas = get_sigmas_karras(50)
    >>> output = sampler.sample(model, latents, sigmas, condition=labels)
    >>> samples = output.samples
    """

    def __init__(self, guidance_scale: float = 1.0) -> None:
        """Initialize Heun sampler."""
        self.guidance_scale = guidance_scale

    def sample(
        self,
        denoiser: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> SamplerOutput:
        """Run Heun sampling.

        Parameters
        ----------
        denoiser : Callable
            Denoiser function.
        latents : torch.Tensor
            Initial noise.
        sigmas : torch.Tensor
            Noise schedule.
        condition : torch.Tensor, optional
            Condition tensor.
        return_trajectory : bool, optional
            Whether to return full trajectory.
        **kwargs
            Additional denoiser arguments.

        Returns
        -------
        SamplerOutput
            Sampling result.
        """
        result = sample_heun(
            denoiser=denoiser,
            latents=latents,
            sigmas=sigmas,
            condition=condition,
            guidance_scale=self.guidance_scale,
            return_trajectory=return_trajectory,
            **kwargs,
        )

        # Heun uses 2 function evaluations per step (except last)
        n_steps = len(sigmas) - 1
        n_evals = n_steps * 2 - 1  # Last step only uses 1 eval

        if return_trajectory:
            return SamplerOutput(
                samples=result[:, -1],
                trajectories=result,
                sigmas=sigmas,
                num_function_evaluations=n_evals,
            )

        return SamplerOutput(
            samples=result,
            sigmas=sigmas,
            num_function_evaluations=n_evals,
        )

    def step(
        self,
        denoiser: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform a single Heun step.

        Parameters
        ----------
        denoiser : Callable
            Denoiser function.
        x : torch.Tensor
            Current noisy sample.
        sigma : torch.Tensor
            Current noise level.
        sigma_next : torch.Tensor
            Target noise level.
        condition : torch.Tensor, optional
            Condition tensor.
        **kwargs
            Additional denoiser arguments.

        Returns
        -------
        torch.Tensor
            Denoised sample at sigma_next.
        """

        def get_denoised(x_in: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            if condition is not None and self.guidance_scale != 1.0:
                d_cond = denoiser(x_in, s, condition=condition, **kwargs)
                d_uncond = denoiser(x_in, s, condition=None, **kwargs)
                return d_uncond + self.guidance_scale * (d_cond - d_uncond)
            elif condition is not None:
                return denoiser(x_in, s, condition=condition, **kwargs)
            else:
                return denoiser(x_in, s, **kwargs)

        dt = sigma_next - sigma

        # First derivative
        denoised = get_denoised(x, sigma)
        d = (x - denoised) / sigma

        # Euler prediction
        x_pred = x + d * dt

        # Corrector step (skip for last step when sigma_next = 0)
        if sigma_next > 0:
            denoised_next = get_denoised(x_pred, sigma_next)
            d_next = (x_pred - denoised_next) / sigma_next
            d_avg = (d + d_next) / 2
            return x + d_avg * dt

        return x_pred
