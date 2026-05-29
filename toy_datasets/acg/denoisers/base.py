"""Base Denoiser Classes.

This module provides the base class for denoisers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class Denoiser(ABC):
    """Abstract base class for denoisers.

    A denoiser takes a noisy input x and noise level σ, and returns
    an estimate of the clean data:

        D(x; σ) ≈ E[x_clean | x_noisy = x, noise_level = σ]

    All denoisers should inherit from this class and implement the
    `denoise` method.
    """

    @abstractmethod
    def denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute denoised prediction.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input of shape [B, ...].
        sigma : torch.Tensor
            Noise level of shape [B].
        **kwargs
            Additional arguments (e.g., condition).

        Returns
        -------
        torch.Tensor
            Denoised prediction of shape [B, ...].
        """
        pass

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Alias for denoise()."""
        return self.denoise(x, sigma, **kwargs)


class GaussianKernelMixin:
    """Mixin providing Gaussian kernel weight computation.

    Used by exact denoisers to compute weights based on distance
    to data points.
    """

    def compute_gaussian_weights(
        self,
        x: torch.Tensor,
        data: torch.Tensor,
        sigma: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Gaussian kernel weights.

        w_i = exp(-||x - x_i||² / (2σ²))

        Parameters
        ----------
        x : torch.Tensor
            Query points of shape [B, D].
        data : torch.Tensor
            Data points of shape [N, D].
        sigma : torch.Tensor
            Noise level of shape [B].
        indices : torch.Tensor, optional
            Indices of data points to consider. If None, uses all.

        Returns
        -------
        torch.Tensor
            Weights of shape [B, N'] (normalized).
        """
        if indices is not None:
            data_subset = data[indices]
        else:
            data_subset = data

        # x: [B, D], data_subset: [N', D]
        # Compute pairwise distances
        diff = x.unsqueeze(1) - data_subset.unsqueeze(0)  # [B, N', D]
        sq_dist = (diff**2).sum(dim=-1)  # [B, N']

        # Compute weights
        sigma_sq = sigma.unsqueeze(1) ** 2  # [B, 1]
        log_weights = -sq_dist / (2 * sigma_sq)

        # Normalize with log-sum-exp for numerical stability
        log_weights_max = log_weights.max(dim=1, keepdim=True).values
        weights = torch.exp(log_weights - log_weights_max)
        weights = weights / weights.sum(dim=1, keepdim=True)

        return weights
