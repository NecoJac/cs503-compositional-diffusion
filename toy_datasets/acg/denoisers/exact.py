"""Exact Denoiser (Non-Neural Network).

Based on Score Diffusion theory, when the dataset is small,
we can compute the optimal denoiser analytically without training
a neural network.

Theory:
-------
The optimal denoiser for a dataset {x_i} is:

    D*(x; σ) = Σ_i x_i · w_i / Σ_i w_i

where:
    w_i = exp(-||x - x_i||² / (2σ²))

This is the Nadaraya-Watson kernel regression estimator with
a Gaussian kernel of bandwidth σ.

Reference:
    This is the optimal Bayes estimator under squared error loss.
"""

from __future__ import annotations

from typing import Optional

import torch

from acg.denoisers.base import Denoiser, GaussianKernelMixin


class ExactDenoiser(Denoiser, GaussianKernelMixin):
    """Exact denoiser using Gaussian kernel weights.

    Given a finite dataset, computes the optimal denoising via
    kernel-weighted averaging:

        D*(x; σ) = Σ_i x_i · w_i / Σ_i w_i
        w_i = exp(-||x - x_i||² / (2σ²))

    Parameters
    ----------
    data : torch.Tensor
        Dataset of shape [N, ...].

    Attributes
    ----------
    data : torch.Tensor
        Stored dataset.
    data_flat : torch.Tensor
        Flattened dataset of shape [N, D].
    n_points : int
        Number of data points.
    data_dim : int
        Dimension of each data point.

    Examples
    --------
    >>> data = torch.randn(100, 2)  # 100 points in 2D
    >>> denoiser = ExactDenoiser(data)
    >>> x_noisy = torch.randn(16, 2)
    >>> sigma = torch.ones(16) * 0.5
    >>> x_denoised = denoiser(x_noisy, sigma)
    """

    def __init__(self, data: torch.Tensor) -> None:
        """Initialize exact denoiser with data.

        Parameters
        ----------
        data : torch.Tensor
            Dataset of shape [N, ...].
        """
        self.data = data
        self.data_flat = data.flatten(1)  # [N, D]
        self.n_points = data.shape[0]
        self.data_dim = self.data_flat.shape[1]
        self._original_shape = data.shape[1:]

    def denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute exact denoised prediction.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input of shape [B, ...].
        sigma : torch.Tensor
            Noise level of shape [B] or scalar.

        Returns
        -------
        torch.Tensor
            Denoised prediction of shape [B, ...].
        """
        original_shape = x.shape
        x_flat = x.flatten(1)  # [B, D]

        # Ensure sigma is the right shape
        if sigma.ndim == 0:
            sigma = sigma.expand(x_flat.shape[0])
        sigma = sigma.flatten()

        # Move data to same device
        data_flat = self.data_flat.to(x.device)

        # Compute weights
        weights = self.compute_gaussian_weights(x_flat, data_flat, sigma)

        # Compute weighted average: [B, N] @ [N, D] -> [B, D]
        denoised_flat = torch.mm(weights, data_flat)

        # Reshape to original
        return denoised_flat.view(original_shape)

    def get_weights(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Get weights for each data point (for visualization).

        Parameters
        ----------
        x : torch.Tensor
            Query points of shape [B, ...].
        sigma : torch.Tensor
            Noise level.

        Returns
        -------
        torch.Tensor
            Weights of shape [B, N].
        """
        x_flat = x.flatten(1)
        if sigma.ndim == 0:
            sigma = sigma.expand(x_flat.shape[0])
        sigma = sigma.flatten()

        data_flat = self.data_flat.to(x.device)
        return self.compute_gaussian_weights(x_flat, data_flat, sigma)


class ExactDenoiserEDM(ExactDenoiser):
    """EDM-formatted exact denoiser.

    This is a convenience wrapper that can be used directly with
    EDM samplers. Since the exact denoiser already produces optimal
    results, we don't need the EDM c_skip/c_out preconditioning.

    Parameters
    ----------
    data : torch.Tensor
        Dataset of shape [N, ...].
    sigma_data : float, optional
        Not used, kept for API compatibility. Default: 0.5.
    """

    def __init__(
        self,
        data: torch.Tensor,
        sigma_data: float = 0.5,
    ) -> None:
        """Initialize EDM-formatted exact denoiser."""
        super().__init__(data)
        self.sigma_data = sigma_data

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (EDM interface).

        Note: For exact denoisers, we return the optimal result directly.
        No c_skip/c_out wrapping is needed.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.

        Returns
        -------
        torch.Tensor
            Denoised prediction.
        """
        return self.denoise(x, sigma)
