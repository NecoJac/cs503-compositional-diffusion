"""Conditional Exact Denoiser.

Extends the exact denoiser to handle conditional generation.

Theory:
-------
The conditional denoiser only considers data points matching the condition:

    D*(x; σ, c) = Σ_{i: c_i=c} x_i · w_i / Σ_{i: c_i=c} w_i

For Classifier-Free Guidance:
    D_cfg = D_uncond + w · (D_cond - D_uncond)
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from acg.denoisers.base import Denoiser, GaussianKernelMixin


class ConditionalExactDenoiser(Denoiser, GaussianKernelMixin):
    """Conditional exact denoiser.

    Given a labeled dataset {(x_i, c_i)}, computes conditional denoising:

        D*(x; σ, c) = Σ_{i: c_i=c} x_i · w_i / Σ_{i: c_i=c} w_i

    Supports:
    - Discrete conditions (class labels)
    - Continuous conditions (embeddings)
    - Classifier-Free Guidance (CFG)

    Parameters
    ----------
    data : torch.Tensor
        Dataset of shape [N, ...].
    conditions : torch.Tensor
        Condition labels of shape [N] (discrete) or [N, cond_dim] (continuous).
    condition_type : str, optional
        "discrete" or "continuous". Default: "discrete".

    Examples
    --------
    >>> data = torch.randn(100, 2)
    >>> labels = torch.randint(0, 5, (100,))
    >>> denoiser = ConditionalExactDenoiser(data, labels)
    >>> x_noisy = torch.randn(16, 2)
    >>> sigma = torch.ones(16) * 0.5
    >>> condition = torch.zeros(16, dtype=torch.long)  # Class 0
    >>> x_denoised = denoiser(x_noisy, sigma, condition=condition)
    """

    def __init__(
        self,
        data: torch.Tensor,
        conditions: torch.Tensor,
        condition_type: str = "discrete",
    ) -> None:
        """Initialize conditional exact denoiser.

        Parameters
        ----------
        data : torch.Tensor
            Dataset of shape [N, ...].
        conditions : torch.Tensor
            Condition labels.
        condition_type : str, optional
            "discrete" or "continuous".
        """
        self.data = data
        self.data_flat = data.flatten(1)  # [N, D]
        self.conditions = conditions
        self.condition_type = condition_type
        self.n_points = data.shape[0]
        self._original_shape = data.shape[1:]

        if condition_type == "discrete":
            # Pre-compute indices for each class
            self.unique_conditions = torch.unique(conditions)
            self.condition_indices = {
                int(c): (conditions == c).nonzero(as_tuple=True)[0]
                for c in self.unique_conditions
            }
            self.n_classes = len(self.unique_conditions)

    def denoise_conditional(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional denoising: only use data points matching condition.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input of shape [B, ...].
        sigma : torch.Tensor
            Noise level of shape [B].
        condition : torch.Tensor
            Condition of shape [B] (discrete) or [B, cond_dim] (continuous).

        Returns
        -------
        torch.Tensor
            Denoised result.
        """
        original_shape = x.shape
        x_flat = x.flatten(1)
        batch_size = x_flat.shape[0]

        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)
        sigma = sigma.flatten()

        data_flat = self.data_flat.to(x.device)
        conditions = self.conditions.to(x.device)

        if self.condition_type == "discrete":
            return self._denoise_discrete(x_flat, sigma, condition, original_shape)
        else:
            return self._denoise_continuous(x_flat, sigma, condition, original_shape)

    def _denoise_discrete(
        self,
        x_flat: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Discrete condition denoising."""
        batch_size = x_flat.shape[0]
        device = x_flat.device

        data_flat = self.data_flat.to(device)
        result = torch.zeros_like(x_flat)

        # Process each sample
        for b in range(batch_size):
            c = int(condition[b].item())
            if c not in self.condition_indices:
                # Unknown condition: use unconditional
                indices = None
            else:
                indices = self.condition_indices[c].to(device)

            if indices is not None:
                data_subset = data_flat[indices]
            else:
                data_subset = data_flat

            # Compute weights for this sample
            x_b = x_flat[b:b+1]  # [1, D]
            sigma_b = sigma[b:b+1]  # [1]

            weights = self.compute_gaussian_weights(x_b, data_subset, sigma_b)
            result[b] = (weights @ data_subset).squeeze(0)

        return result.view(original_shape)

    def _denoise_continuous(
        self,
        x_flat: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        original_shape: torch.Size,
        cond_kernel_sigma: float = 1.0,
    ) -> torch.Tensor:
        """Continuous condition denoising with kernel weighting."""
        device = x_flat.device
        data_flat = self.data_flat.to(device)
        conditions = self.conditions.to(device)

        # Compute distance in data space
        diff_x = x_flat.unsqueeze(1) - data_flat.unsqueeze(0)  # [B, N, D]
        sq_dist_x = (diff_x**2).sum(dim=-1)  # [B, N]

        # Compute distance in condition space
        diff_c = condition.unsqueeze(1) - conditions.unsqueeze(0)  # [B, N, C]
        sq_dist_c = (diff_c**2).sum(dim=-1)  # [B, N]

        # Combined kernel
        sigma_sq = sigma.unsqueeze(1) ** 2
        cond_sigma_sq = cond_kernel_sigma ** 2

        log_weights = -sq_dist_x / (2 * sigma_sq) - sq_dist_c / (2 * cond_sigma_sq)
        log_weights_max = log_weights.max(dim=1, keepdim=True).values
        weights = torch.exp(log_weights - log_weights_max)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Weighted average
        denoised_flat = torch.mm(weights, data_flat)
        return denoised_flat.view(original_shape)

    def denoise_unconditional(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Unconditional denoising: use all data points.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.

        Returns
        -------
        torch.Tensor
            Denoised result.
        """
        original_shape = x.shape
        x_flat = x.flatten(1)

        if sigma.ndim == 0:
            sigma = sigma.expand(x_flat.shape[0])
        sigma = sigma.flatten()

        data_flat = self.data_flat.to(x.device)
        weights = self.compute_gaussian_weights(x_flat, data_flat, sigma)
        denoised_flat = torch.mm(weights, data_flat)

        return denoised_flat.view(original_shape)

    def denoise_cfg(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Classifier-Free Guidance denoising.

        D_cfg = D_uncond + scale · (D_cond - D_uncond)

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.
        condition : torch.Tensor
            Condition.
        guidance_scale : float, optional
            CFG strength. 1.0 = pure conditional. Default: 1.0.

        Returns
        -------
        torch.Tensor
            CFG-guided denoised result.
        """
        d_cond = self.denoise_conditional(x, sigma, condition)

        if guidance_scale == 1.0:
            return d_cond

        d_uncond = self.denoise_unconditional(x, sigma)
        return d_uncond + guidance_scale * (d_cond - d_uncond)

    def denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Unified denoising interface.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input.
        sigma : torch.Tensor
            Noise level.
        condition : torch.Tensor, optional
            Condition. None for unconditional.
        guidance_scale : float, optional
            CFG scale (only used when condition is not None).

        Returns
        -------
        torch.Tensor
            Denoised result.
        """
        if condition is None:
            return self.denoise_unconditional(x, sigma)
        elif guidance_scale != 1.0:
            return self.denoise_cfg(x, sigma, condition, guidance_scale)
        else:
            return self.denoise_conditional(x, sigma, condition)
