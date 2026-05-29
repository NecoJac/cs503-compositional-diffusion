"""Multi-Attribute Sampler for Composable Diffusion.

This module provides samplers designed for multi-attribute conditional generation,
implementing the Composable Diffusion formula:

    D_combo = D_uncond + Σ_i w_i · (D_attr_i - D_uncond)

When all weights w_i = 1, this simplifies to:
    D_combo = D_attr1 + D_attr2 + ... - (n-1) · D_uncond

Reference:
    Liu et al. 2022 - "Compositional Visual Generation with Composable
    Diffusion Models"
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from acg.samplers.base import Sampler, SamplerOutput


def sample_heun_multi_attr(
    denoiser: Callable[..., torch.Tensor],
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    attr_conditions: Optional[Dict[Union[int, str], int]] = None,
    guidance_scales: Optional[Dict[Union[int, str], float]] = None,
    return_trajectory: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Heun sampling with multi-attribute conditions (Composable Diffusion).

    Uses the Heun (second-order) method with multi-attribute conditioning.
    The denoiser should support `attr_conditions` and `guidance_scales` kwargs.

    Parameters
    ----------
    denoiser : Callable
        Multi-attribute denoiser that accepts (x, sigma, attr_conditions, guidance_scales).
        Typically a `MultiAttributeDenoiser` instance.
    latents : torch.Tensor
        Initial noise of shape [B, ...].
    sigmas : torch.Tensor
        Noise schedule of shape [n_steps + 1], ending with 0.
    attr_conditions : dict, optional
        Attribute conditions as {attr_name: attr_value} or {attr_idx: attr_value}.
        Examples:
            {"A": 0, "B": 1} - condition on attribute A=0 AND B=1
            {0: 1} - condition on first attribute = 1
        If None, performs unconditional sampling.
    guidance_scales : dict, optional
        Per-attribute guidance scales as {attr_name: scale}.
        Default: 1.0 for all attributes.
        Examples:
            {"A": 2.0, "B": 1.0} - stronger guidance for attribute A
    return_trajectory : bool, optional
        If True, returns tensor of shape [B, n_steps+1, ...] with all intermediate states.
        If False, returns final samples of shape [B, ...].
    **kwargs
        Additional arguments passed to denoiser.

    Returns
    -------
    torch.Tensor
        If return_trajectory=False: Final samples of shape [B, ...].
        If return_trajectory=True: Trajectory of shape [B, n_steps+1, ...].

    Examples
    --------
    >>> from acg.denoisers import MultiAttributeDenoiser
    >>> from acg.samplers import sample_heun_multi_attr
    >>>
    >>> denoiser = MultiAttributeDenoiser(data, [labels_A, labels_B], ["A", "B"])
    >>> sigmas = get_sigmas_karras(50, device=device)
    >>> latents = torch.randn(64, 2, device=device) * sigmas[0]
    >>>
    >>> # Generate samples satisfying A=0 AND B=1
    >>> samples = sample_heun_multi_attr(
    ...     denoiser, latents, sigmas,
    ...     attr_conditions={"A": 0, "B": 1},
    ...     guidance_scales={"A": 1.0, "B": 1.0}
    ... )
    """
    x = latents
    trajectory: List[torch.Tensor] = []

    if return_trajectory:
        trajectory.append(x.clone())

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        # Expand sigma to batch size
        sigma_batch = sigma.expand(x.shape[0])

        # Get denoised prediction with multi-attribute conditions
        denoised = denoiser(
            x, sigma_batch,
            attr_conditions=attr_conditions,
            guidance_scales=guidance_scales,
            **kwargs
        )
        d = (x - denoised) / sigma

        # Euler prediction
        x_pred = x + d * dt

        # Heun correction (skip for last step when sigma_next = 0)
        if sigma_next > 0:
            sigma_next_batch = sigma_next.expand(x.shape[0])
            denoised_next = denoiser(
                x_pred, sigma_next_batch,
                attr_conditions=attr_conditions,
                guidance_scales=guidance_scales,
                **kwargs
            )
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


class HeunMultiAttrSampler(Sampler):
    """Heun sampler for multi-attribute conditional generation.

    Implements second-order Heun sampling with support for Composable Diffusion,
    allowing multiple attributes to be composed using:

        D_combo = D_uncond + Σ_i w_i · (D_attr_i - D_uncond)

    Parameters
    ----------
    attr_conditions : dict, optional
        Default attribute conditions.
    guidance_scales : dict, optional
        Default per-attribute guidance scales.

    Examples
    --------
    >>> sampler = HeunMultiAttrSampler(
    ...     attr_conditions={"A": 0, "B": 1},
    ...     guidance_scales={"A": 1.0, "B": 2.0}
    ... )
    >>> output = sampler.sample(denoiser, latents, sigmas)
    >>> samples = output.samples
    """

    def __init__(
        self,
        attr_conditions: Optional[Dict[Union[int, str], int]] = None,
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
    ) -> None:
        """Initialize multi-attribute Heun sampler."""
        self.attr_conditions = attr_conditions
        self.guidance_scales = guidance_scales

    def sample(
        self,
        denoiser: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        attr_conditions: Optional[Dict[Union[int, str], int]] = None,
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> SamplerOutput:
        """Run Heun sampling with multi-attribute conditions.

        Parameters
        ----------
        denoiser : Callable
            Multi-attribute denoiser.
        latents : torch.Tensor
            Initial noise.
        sigmas : torch.Tensor
            Noise schedule.
        attr_conditions : dict, optional
            Attribute conditions. Overrides instance default if provided.
        guidance_scales : dict, optional
            Per-attribute guidance scales. Overrides instance default if provided.
        return_trajectory : bool, optional
            Whether to return full trajectory.
        **kwargs
            Additional denoiser arguments.

        Returns
        -------
        SamplerOutput
            Sampling result.
        """
        # Use provided conditions or fall back to instance defaults
        conditions = attr_conditions if attr_conditions is not None else self.attr_conditions
        scales = guidance_scales if guidance_scales is not None else self.guidance_scales

        result = sample_heun_multi_attr(
            denoiser=denoiser,
            latents=latents,
            sigmas=sigmas,
            attr_conditions=conditions,
            guidance_scales=scales,
            return_trajectory=return_trajectory,
            **kwargs,
        )

        # Heun uses 2 function evaluations per step (except last)
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

    def step(
        self,
        denoiser: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        attr_conditions: Optional[Dict[Union[int, str], int]] = None,
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform a single Heun step with multi-attribute conditions.

        Parameters
        ----------
        denoiser : Callable
            Multi-attribute denoiser.
        x : torch.Tensor
            Current noisy sample.
        sigma : torch.Tensor
            Current noise level.
        sigma_next : torch.Tensor
            Target noise level.
        attr_conditions : dict, optional
            Attribute conditions.
        guidance_scales : dict, optional
            Per-attribute guidance scales.
        **kwargs
            Additional denoiser arguments.

        Returns
        -------
        torch.Tensor
            Denoised sample at sigma_next.
        """
        conditions = attr_conditions if attr_conditions is not None else self.attr_conditions
        scales = guidance_scales if guidance_scales is not None else self.guidance_scales

        dt = sigma_next - sigma
        sigma_batch = sigma.expand(x.shape[0])

        # First derivative
        denoised = denoiser(
            x, sigma_batch,
            attr_conditions=conditions,
            guidance_scales=scales,
            **kwargs
        )
        d = (x - denoised) / sigma

        # Euler prediction
        x_pred = x + d * dt

        # Heun correction
        if sigma_next > 0:
            sigma_next_batch = sigma_next.expand(x.shape[0])
            denoised_next = denoiser(
                x_pred, sigma_next_batch,
                attr_conditions=conditions,
                guidance_scales=scales,
                **kwargs
            )
            d_next = (x_pred - denoised_next) / sigma_next
            d_avg = (d + d_next) / 2
            return x + d_avg * dt

        return x_pred


def sample_euler_multi_attr(
    denoiser: Callable[..., torch.Tensor],
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    attr_conditions: Optional[Dict[Union[int, str], int]] = None,
    guidance_scales: Optional[Dict[Union[int, str], float]] = None,
    return_trajectory: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Euler sampling with multi-attribute conditions (first-order).

    Simpler and faster than Heun but with lower accuracy.

    Parameters
    ----------
    denoiser : Callable
        Multi-attribute denoiser.
    latents : torch.Tensor
        Initial noise of shape [B, ...].
    sigmas : torch.Tensor
        Noise schedule of shape [n_steps + 1].
    attr_conditions : dict, optional
        Attribute conditions.
    guidance_scales : dict, optional
        Per-attribute guidance scales.
    return_trajectory : bool, optional
        If True, returns full trajectory.
    **kwargs
        Additional arguments passed to denoiser.

    Returns
    -------
    torch.Tensor
        Final samples or trajectory.
    """
    x = latents
    trajectory: List[torch.Tensor] = []

    if return_trajectory:
        trajectory.append(x.clone())

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        sigma_batch = sigma.expand(x.shape[0])

        denoised = denoiser(
            x, sigma_batch,
            attr_conditions=attr_conditions,
            guidance_scales=guidance_scales,
            **kwargs
        )
        d = (x - denoised) / sigma

        x = x + d * dt

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=1)

    return x


class EulerMultiAttrSampler(Sampler):
    """Euler sampler for multi-attribute conditional generation.

    First-order method, simpler and faster than Heun but with lower accuracy.

    Parameters
    ----------
    attr_conditions : dict, optional
        Default attribute conditions.
    guidance_scales : dict, optional
        Default per-attribute guidance scales.
    """

    def __init__(
        self,
        attr_conditions: Optional[Dict[Union[int, str], int]] = None,
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
    ) -> None:
        """Initialize multi-attribute Euler sampler."""
        self.attr_conditions = attr_conditions
        self.guidance_scales = guidance_scales

    def sample(
        self,
        denoiser: Callable[..., torch.Tensor],
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        attr_conditions: Optional[Dict[Union[int, str], int]] = None,
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> SamplerOutput:
        """Run Euler sampling with multi-attribute conditions."""
        conditions = attr_conditions if attr_conditions is not None else self.attr_conditions
        scales = guidance_scales if guidance_scales is not None else self.guidance_scales

        result = sample_euler_multi_attr(
            denoiser=denoiser,
            latents=latents,
            sigmas=sigmas,
            attr_conditions=conditions,
            guidance_scales=scales,
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
        attr_conditions: Optional[Dict[Union[int, str], int]] = None,
        guidance_scales: Optional[Dict[Union[int, str], float]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform a single Euler step with multi-attribute conditions."""
        conditions = attr_conditions if attr_conditions is not None else self.attr_conditions
        scales = guidance_scales if guidance_scales is not None else self.guidance_scales

        dt = sigma_next - sigma
        sigma_batch = sigma.expand(x.shape[0])

        denoised = denoiser(
            x, sigma_batch,
            attr_conditions=conditions,
            guidance_scales=scales,
            **kwargs
        )
        d = (x - denoised) / sigma

        return x + d * dt
