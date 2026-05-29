"""EDM Training Losses.

This module provides loss functions for training EDM-based diffusion models.

The core training objective is:
    L = E[λ(σ) * ||D(x + σn; σ) - x||²]

Where σ is sampled from a log-normal distribution and λ(σ) is the loss weighting.

Reference:
    Karras et al. 2022 - "Elucidating the Design Space of Diffusion-Based
    Generative Models" (https://arxiv.org/abs/2206.00364)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from acg.utils.tensor import append_dims


class EDMLoss:
    """EDM training loss for unconditional models.

    Samples noise levels from log-normal distribution and computes the
    denoising loss with EDM's loss weighting.

    Parameters
    ----------
    P_mean : float, optional
        Mean of log(σ) distribution. Default: -1.2.
    P_std : float, optional
        Standard deviation of log(σ) distribution. Default: 1.2.
    sigma_data : float, optional
        Data standard deviation for loss weighting. Default: 0.5.

    Examples
    --------
    >>> loss_fn = EDMLoss()
    >>> model = EDMPrecond(base_model)
    >>> images = torch.randn(4, 3, 32, 32)  # Clean images
    >>> loss = loss_fn(model, images)
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ) -> None:
        """Initialize EDM loss.

        Parameters
        ----------
        P_mean : float, optional
            Mean of log(σ) distribution. Default: -1.2.
        P_std : float, optional
            Standard deviation of log(σ) distribution. Default: 1.2.
        sigma_data : float, optional
            Data standard deviation. Default: 0.5.
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute EDM training loss.

        Parameters
        ----------
        model : Callable
            Denoiser model (should be wrapped with EDMPrecond).
        images : torch.Tensor
            Clean images of shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = images.shape[0]
        device = images.device

        # Sample noise levels from log-normal distribution
        log_sigma = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        sigma = log_sigma.exp()

        # Sample noise and create noisy images
        noise = torch.randn_like(images)
        noisy_images = images + append_dims(sigma, images.ndim) * noise

        # Get model prediction
        denoised = model(noisy_images, sigma)

        # Compute loss weight: λ(σ) = (σ² + σ_data²) / (σ * σ_data)²
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Compute weighted MSE loss
        loss = weight * ((denoised - images) ** 2).flatten(1).mean(1)

        return loss.mean()


class EDMLossConditional:
    """EDM training loss for conditional models.

    Same as EDMLoss but passes condition to the model.
    The model is expected to handle condition dropout internally (for CFG).

    Parameters
    ----------
    P_mean : float, optional
        Mean of log(σ) distribution. Default: -1.2.
    P_std : float, optional
        Standard deviation of log(σ) distribution. Default: 1.2.
    sigma_data : float, optional
        Data standard deviation for loss weighting. Default: 0.5.

    Examples
    --------
    >>> loss_fn = EDMLossConditional()
    >>> model = EDMPrecondConditional(base_model)
    >>> images = torch.randn(4, 3, 32, 32)
    >>> labels = torch.randint(0, 10, (4,))
    >>> loss = loss_fn(model, images, labels)
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ) -> None:
        """Initialize conditional EDM loss.

        Parameters
        ----------
        P_mean : float, optional
            Mean of log(σ) distribution. Default: -1.2.
        P_std : float, optional
            Standard deviation of log(σ) distribution. Default: 1.2.
        sigma_data : float, optional
            Data standard deviation. Default: 0.5.
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        images: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute conditional EDM training loss.

        Parameters
        ----------
        model : Callable
            Conditional denoiser model (should be wrapped with EDMPrecondConditional).
        images : torch.Tensor
            Clean images of shape [B, C, H, W].
        condition : torch.Tensor, optional
            Condition tensor (class labels, embeddings, etc.).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = images.shape[0]
        device = images.device

        # Sample noise levels from log-normal distribution
        log_sigma = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        sigma = log_sigma.exp()

        # Sample noise and create noisy images
        noise = torch.randn_like(images)
        noisy_images = images + append_dims(sigma, images.ndim) * noise

        # Get model prediction with condition
        denoised = model(noisy_images, sigma, condition)

        # Compute loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Compute weighted MSE loss
        loss = weight * ((denoised - images) ** 2).flatten(1).mean(1)

        return loss.mean()
