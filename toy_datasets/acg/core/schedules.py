"""Noise Schedules for Diffusion Models.

This module provides noise schedule functions for diffusion sampling.
The Karras schedule is recommended for EDM models.

Reference:
    Karras et al. 2022 - "Elucidating the Design Space of Diffusion-Based
    Generative Models" (https://arxiv.org/abs/2206.00364)
"""

from __future__ import annotations

from typing import Optional

import torch


def get_sigmas_karras(
    n_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate Karras noise schedule.

    Creates a noise schedule from sigma_max to sigma_min with a final 0,
    using the rho parameter to control the density of steps.
    Higher rho values concentrate more steps at high noise levels.

    The schedule is defined as:
        sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    Parameters
    ----------
    n_steps : int
        Number of sampling steps (not including the final 0).
    sigma_min : float, optional
        Minimum noise level. Default: 0.002.
    sigma_max : float, optional
        Maximum noise level. Default: 80.0.
    rho : float, optional
        Schedule parameter controlling step density. Default: 7.0.
        Higher values = more steps at high noise levels.
    device : torch.device, optional
        Device to place the tensor on.

    Returns
    -------
    torch.Tensor
        Noise schedule of shape [n_steps + 1], with the last element being 0.

    Examples
    --------
    >>> sigmas = get_sigmas_karras(50)
    >>> sigmas.shape
    torch.Size([51])
    >>> sigmas[0], sigmas[-1]
    (tensor(80.), tensor(0.))
    """
    ramp = torch.linspace(0, 1, n_steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)

    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    # Append 0 at the end
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_linear(
    n_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate linear noise schedule.

    Creates a simple linear schedule from sigma_max to sigma_min with a final 0.
    Less effective than Karras schedule but simpler.

    Parameters
    ----------
    n_steps : int
        Number of sampling steps (not including the final 0).
    sigma_min : float, optional
        Minimum noise level. Default: 0.002.
    sigma_max : float, optional
        Maximum noise level. Default: 80.0.
    device : torch.device, optional
        Device to place the tensor on.

    Returns
    -------
    torch.Tensor
        Noise schedule of shape [n_steps + 1], with the last element being 0.
    """
    sigmas = torch.linspace(sigma_max, sigma_min, n_steps, device=device)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_exponential(
    n_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate exponential (log-linear) noise schedule.

    Creates an exponentially spaced schedule from sigma_max to sigma_min with a final 0.

    Parameters
    ----------
    n_steps : int
        Number of sampling steps (not including the final 0).
    sigma_min : float, optional
        Minimum noise level. Default: 0.002.
    sigma_max : float, optional
        Maximum noise level. Default: 80.0.
    device : torch.device, optional
        Device to place the tensor on.

    Returns
    -------
    torch.Tensor
        Noise schedule of shape [n_steps + 1], with the last element being 0.
    """
    sigmas = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(sigma_max)),
            torch.log(torch.tensor(sigma_min)),
            n_steps,
            device=device,
        )
    )
    return torch.cat([sigmas, sigmas.new_zeros([1])])
