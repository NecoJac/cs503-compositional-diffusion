"""Euler Samplers.

This module provides Euler-based sampling algorithms:
- Euler (first-order deterministic)
- Euler Ancestral (first-order stochastic)

The Euler method is the simplest sampler, directly discretizing the ODE:
    dx/dt = (x - D(x; σ)) / σ
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from acg.samplers.base import CFGMixin, Sampler, SamplerOutput


def sample_euler(
    denoiser: Callable[..., torch.Tensor],
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    return_trajectory: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Euler sampling (first-order).

    Implements the ODE: dx/dt = (x - D(x; σ)) / σ

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

    if return_trajectory:
        trajectory.append(x.clone())

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Get denoised prediction
        if condition is not None and guidance_scale != 1.0:
            # Apply CFG
            d_cond = denoiser(x, sigma, condition=condition, **kwargs)
            d_uncond = denoiser(x, sigma, condition=None, **kwargs)
            denoised = d_uncond + guidance_scale * (d_cond - d_uncond)
        elif condition is not None:
            denoised = denoiser(x, sigma, condition=condition, **kwargs)
        else:
            denoised = denoiser(x, sigma, **kwargs)

        # Euler step
        # d = (x - denoised) / sigma  # derivative
        # x = x + d * (sigma_next - sigma)  # step
        d = (x - denoised) / sigma
        dt = sigma_next - sigma
        x = x + d * dt

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=1)

    return x


def sample_euler_ancestral(
    denoiser: Callable[..., torch.Tensor],
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    eta: float = 1.0,
    return_trajectory: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Euler Ancestral sampling (stochastic).

    Adds noise at each step for increased diversity.

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
        CFG scale. Default: 1.0.
    eta : float, optional
        Noise strength. 0 = deterministic, 1 = full noise. Default: 1.0.
    return_trajectory : bool, optional
        If True, returns all intermediate states.
    **kwargs
        Additional arguments passed to denoiser.

    Returns
    -------
    torch.Tensor
        Generated samples of shape [B, ...].
    """
    x = latents
    trajectory: List[torch.Tensor] = []

    if return_trajectory:
        trajectory.append(x.clone())

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Get denoised prediction with optional CFG
        if condition is not None and guidance_scale != 1.0:
            d_cond = denoiser(x, sigma, condition=condition, **kwargs)
            d_uncond = denoiser(x, sigma, condition=None, **kwargs)
            denoised = d_uncond + guidance_scale * (d_cond - d_uncond)
        elif condition is not None:
            denoised = denoiser(x, sigma, condition=condition, **kwargs)
        else:
            denoised = denoiser(x, sigma, **kwargs)

        # Compute sigma_down and sigma_up
        sigma_down = sigma_next
        sigma_up = 0.0

        if eta > 0 and sigma_next > 0:
            # sigma_up^2 + sigma_down^2 = sigma_next^2
            sigma_up = min(sigma_next, eta * (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2).sqrt())
            sigma_down = (sigma_next**2 - sigma_up**2).sqrt()

        # Euler step to sigma_down
        d = (x - denoised) / sigma
        x = x + d * (sigma_down - sigma)

        # Add noise if not last step
        if sigma_up > 0:
            x = x + torch.randn_like(x) * sigma_up

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=1)

    return x


class EulerSampler(Sampler, CFGMixin):
    """Euler sampler (first-order, deterministic).

    Parameters
    ----------
    guidance_scale : float, optional
        CFG scale for conditional sampling. Default: 1.0.

    Examples
    --------
    >>> sampler = EulerSampler(guidance_scale=7.5)
    >>> output = sampler.sample(model, latents, sigmas, condition=labels)
    >>> samples = output.samples
    """

    def __init__(self, guidance_scale: float = 1.0) -> None:
        """Initialize Euler sampler."""
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
        """Run Euler sampling.

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
        result = sample_euler(
            denoiser=denoiser,
            latents=latents,
            sigmas=sigmas,
            condition=condition,
            guidance_scale=self.guidance_scale,
            return_trajectory=return_trajectory,
            **kwargs,
        )

        n_steps = len(sigmas) - 1

        if return_trajectory:
            return SamplerOutput(
                samples=result[:, -1],
                trajectories=result,
                sigmas=sigmas,
                num_function_evaluations=n_steps,
            )

        return SamplerOutput(
            samples=result,
            sigmas=sigmas,
            num_function_evaluations=n_steps,
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
        """Perform a single Euler step.

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
        # Get denoised prediction with optional CFG
        if condition is not None and self.guidance_scale != 1.0:
            d_cond = denoiser(x, sigma, condition=condition, **kwargs)
            d_uncond = denoiser(x, sigma, condition=None, **kwargs)
            denoised = d_uncond + self.guidance_scale * (d_cond - d_uncond)
        elif condition is not None:
            denoised = denoiser(x, sigma, condition=condition, **kwargs)
        else:
            denoised = denoiser(x, sigma, **kwargs)

        # Euler step
        d = (x - denoised) / sigma
        dt = sigma_next - sigma
        return x + d * dt


class EulerAncestralSampler(Sampler, CFGMixin):
    """Euler Ancestral sampler (first-order, stochastic).

    Parameters
    ----------
    guidance_scale : float, optional
        CFG scale. Default: 1.0.
    eta : float, optional
        Noise strength. 0 = deterministic, 1 = full noise. Default: 1.0.
    """

    def __init__(self, guidance_scale: float = 1.0, eta: float = 1.0) -> None:
        """Initialize Euler Ancestral sampler."""
        self.guidance_scale = guidance_scale
        self.eta = eta

    def sample(
        self,
        denoiser: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> SamplerOutput:
        """Run Euler Ancestral sampling."""
        result = sample_euler_ancestral(
            denoiser=denoiser,
            latents=latents,
            sigmas=sigmas,
            condition=condition,
            guidance_scale=self.guidance_scale,
            eta=self.eta,
            return_trajectory=return_trajectory,
            **kwargs,
        )

        n_steps = len(sigmas) - 1

        if return_trajectory:
            return SamplerOutput(
                samples=result[:, -1],
                trajectories=result,
                sigmas=sigmas,
                num_function_evaluations=n_steps,
            )

        return SamplerOutput(
            samples=result,
            sigmas=sigmas,
            num_function_evaluations=n_steps,
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
        """Perform a single Euler Ancestral step.

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
        # Get denoised prediction with optional CFG
        if condition is not None and self.guidance_scale != 1.0:
            d_cond = denoiser(x, sigma, condition=condition, **kwargs)
            d_uncond = denoiser(x, sigma, condition=None, **kwargs)
            denoised = d_uncond + self.guidance_scale * (d_cond - d_uncond)
        elif condition is not None:
            denoised = denoiser(x, sigma, condition=condition, **kwargs)
        else:
            denoised = denoiser(x, sigma, **kwargs)

        # Compute sigma_down and sigma_up
        sigma_down = sigma_next
        sigma_up = 0.0

        if self.eta > 0 and sigma_next > 0:
            sigma_up = min(
                sigma_next,
                self.eta * (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2).sqrt(),
            )
            sigma_down = (sigma_next**2 - sigma_up**2).sqrt()

        # Euler step to sigma_down
        d = (x - denoised) / sigma
        x_new = x + d * (sigma_down - sigma)

        # Add noise if not last step
        if sigma_up > 0:
            x_new = x_new + torch.randn_like(x_new) * sigma_up

        return x_new

