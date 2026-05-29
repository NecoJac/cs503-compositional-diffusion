"""Base Sampler Classes.

This module provides the base classes and interfaces for diffusion samplers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

import torch


@dataclass
class SamplerOutput:
    """Output from a sampling run.

    Attributes
    ----------
    samples : torch.Tensor
        Final generated samples of shape [B, ...].
    trajectories : torch.Tensor, optional
        Full sampling trajectory of shape [B, n_steps, ...].
        Only included if `return_trajectory=True`.
    sigmas : torch.Tensor, optional
        Noise schedule used for sampling.
    num_function_evaluations : int
        Number of denoiser function calls made.
    """

    samples: torch.Tensor
    trajectories: Optional[torch.Tensor] = None
    sigmas: Optional[torch.Tensor] = None
    num_function_evaluations: int = 0
    extra: dict = field(default_factory=dict)


class Sampler(ABC):
    """Abstract base class for diffusion samplers.

    All samplers should inherit from this class and implement the
    `sample` method.
    """

    @abstractmethod
    def sample(
        self,
        denoiser: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        **kwargs: Any,
    ) -> SamplerOutput:
        """Run sampling.

        Parameters
        ----------
        denoiser : Callable
            Denoiser function D(x, sigma, ...).
        latents : torch.Tensor
            Initial noise of shape [B, ...].
        sigmas : torch.Tensor
            Noise schedule of shape [n_steps + 1].
        **kwargs
            Additional arguments passed to denoiser.

        Returns
        -------
        SamplerOutput
            Sampling result including final samples and optional trajectory.
        """
        pass

    @abstractmethod
    def step(
        self,
        denoiser: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform a single denoising step.

        Parameters
        ----------
        denoiser : Callable
            Denoiser function D(x, sigma, ...).
        x : torch.Tensor
            Current noisy sample of shape [B, ...].
        sigma : torch.Tensor
            Current noise level.
        sigma_next : torch.Tensor
            Target noise level.
        **kwargs
            Additional arguments passed to denoiser.

        Returns
        -------
        torch.Tensor
            Denoised sample at sigma_next.
        """
        pass

    def __call__(
        self,
        denoiser: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        **kwargs: Any,
    ) -> SamplerOutput:
        """Alias for sample()."""
        return self.sample(denoiser, latents, sigmas, **kwargs)


class CFGMixin:
    """Mixin for Classifier-Free Guidance support.

    Provides helper methods for CFG sampling where we need to compute
    both conditional and unconditional predictions.
    """

    def apply_cfg(
        self,
        denoiser: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 7.5,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply Classifier-Free Guidance.

        Computes: D_cfg = D_uncond + scale * (D_cond - D_uncond)

        Parameters
        ----------
        denoiser : Callable
            Denoiser that accepts (x, sigma, condition, force_uncond).
        x : torch.Tensor
            Current noisy sample.
        sigma : torch.Tensor
            Current noise level.
        condition : torch.Tensor
            Condition tensor.
        guidance_scale : float, optional
            CFG scale. 1.0 = pure conditional. Default: 7.5.
        **kwargs
            Additional arguments.

        Returns
        -------
        torch.Tensor
            CFG-guided denoised result.
        """
        # Conditional prediction
        d_cond = denoiser(x, sigma, condition=condition, force_uncond=False, **kwargs)

        if guidance_scale == 1.0:
            return d_cond

        # Unconditional prediction
        d_uncond = denoiser(x, sigma, condition=condition, force_uncond=True, **kwargs)

        # Apply CFG
        return d_uncond + guidance_scale * (d_cond - d_uncond)

    def apply_cfg_batched(
        self,
        denoiser: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 7.5,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply CFG with batched forward pass (more efficient).

        Concatenates conditional and unconditional inputs into a single batch.

        Parameters
        ----------
        denoiser : Callable
            Denoiser function.
        x : torch.Tensor
            Current noisy sample of shape [B, ...].
        sigma : torch.Tensor
            Current noise level.
        condition : torch.Tensor
            Condition tensor.
        guidance_scale : float, optional
            CFG scale. Default: 7.5.
        **kwargs
            Additional arguments.

        Returns
        -------
        torch.Tensor
            CFG-guided denoised result.
        """
        if guidance_scale == 1.0:
            return denoiser(x, sigma, condition=condition, **kwargs)

        batch_size = x.shape[0]

        # Duplicate input
        x_double = torch.cat([x, x], dim=0)
        sigma_double = sigma.repeat(2) if sigma.numel() > 1 else sigma

        # Prepare conditions: [uncond, cond]
        null_cond = torch.zeros_like(condition)
        cond_double = torch.cat([null_cond, condition], dim=0)

        # Single forward pass
        output = denoiser(x_double, sigma_double, condition=cond_double, **kwargs)

        # Split and apply CFG
        d_uncond, d_cond = output[:batch_size], output[batch_size:]
        return d_uncond + guidance_scale * (d_cond - d_uncond)
