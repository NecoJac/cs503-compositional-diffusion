"""DPM-Solver Samplers.

This module provides DPM-Solver-based sampling algorithms.

DPM-Solver is a family of efficient samplers that use exponential
integrators for faster convergence.

Reference:
    Lu et al. 2022 - "DPM-Solver: A Fast ODE Solver for Diffusion
    Probabilistic Model Sampling in Around 10 Steps"
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from acg.samplers.base import CFGMixin, Sampler, SamplerOutput


def sample_dpm_2(
    denoiser: Callable[..., torch.Tensor],
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    return_trajectory: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """DPM-Solver-2 sampling (second-order).

    Uses midpoint method in log-sigma space for efficient sampling.

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

        if sigma_next == 0:
            # Final step: just denoise
            x = get_denoised(x, sigma)
        else:
            # Midpoint in log-sigma space
            sigma_mid = (sigma * sigma_next).sqrt()

            # First step: go to midpoint
            denoised = get_denoised(x, sigma)
            d = (x - denoised) / sigma
            x_mid = x + d * (sigma_mid - sigma)

            # Second step: evaluate at midpoint, go to next
            denoised_mid = get_denoised(x_mid, sigma_mid)
            d_mid = (x_mid - denoised_mid) / sigma_mid
            x = x + d_mid * (sigma_next - sigma)

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=1)

    return x


class DPMSolver2Sampler(Sampler, CFGMixin):
    """DPM-Solver-2 sampler (second-order).

    Parameters
    ----------
    guidance_scale : float, optional
        CFG scale for conditional sampling. Default: 1.0.

    Examples
    --------
    >>> sampler = DPMSolver2Sampler(guidance_scale=7.5)
    >>> output = sampler.sample(model, latents, sigmas, condition=labels)
    >>> samples = output.samples
    """

    def __init__(self, guidance_scale: float = 1.0) -> None:
        """Initialize DPM-Solver-2 sampler."""
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
        """Run DPM-Solver-2 sampling.

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
        result = sample_dpm_2(
            denoiser=denoiser,
            latents=latents,
            sigmas=sigmas,
            condition=condition,
            guidance_scale=self.guidance_scale,
            return_trajectory=return_trajectory,
            **kwargs,
        )

        # DPM-2 uses 2 function evaluations per step (except last)
        n_steps = len(sigmas) - 1
        n_evals = n_steps * 2 - 1

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
