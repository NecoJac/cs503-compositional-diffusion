"""EDM Preconditioning Wrappers.

This module provides preconditioning wrappers that transform arbitrary denoiser
networks into the EDM parameterization.

The key insight from EDM is that the denoiser output should be parameterized as:
    D(x; σ) = c_skip(σ) * x + c_out(σ) * F_θ(c_in(σ) * x; c_noise(σ))

Where the scaling functions ensure:
- D(x; σ→0) → x (identity at low noise)
- D(x; σ→∞) → mean (data mean at high noise)
- Stable training across all noise levels

Reference:
    Karras et al. 2022 - "Elucidating the Design Space of Diffusion-Based
    Generative Models" (https://arxiv.org/abs/2206.00364)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from acg.utils.tensor import append_dims


class EDMPrecond(nn.Module):
    """EDM preconditioning wrapper for unconditional denoising.

    Wraps an arbitrary denoiser network F_θ into EDM's parameterization:
        D(x; σ) = c_skip * x + c_out * F_θ(c_in * x; c_noise)

    The scaling functions are:
        c_skip = σ_data² / (σ² + σ_data²)
        c_out = σ * σ_data / √(σ² + σ_data²)
        c_in = 1 / √(σ² + σ_data²)
        c_noise = log(σ) / 4

    Parameters
    ----------
    model : nn.Module
        Base denoiser network F_θ. Should accept (x, noise_level) as input.
    sigma_data : float, optional
        Data standard deviation for normalization. Default: 0.5.

    Examples
    --------
    >>> base_model = MyDenoiser()
    >>> model = EDMPrecond(base_model, sigma_data=0.5)
    >>> x_noisy = torch.randn(4, 3, 32, 32)
    >>> sigma = torch.ones(4) * 1.0
    >>> x_denoised = model(x_noisy, sigma)
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_data: float = 0.5,
    ) -> None:
        """Initialize EDM preconditioning wrapper.

        Parameters
        ----------
        model : nn.Module
            Base denoiser network F_θ.
        sigma_data : float, optional
            Data standard deviation. Default: 0.5.
        """
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with EDM preconditioning.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input of shape [B, ...].
        sigma : torch.Tensor
            Noise level of shape [B] or [B, 1, ...].

        Returns
        -------
        torch.Tensor
            Denoised prediction of shape [B, ...].
        """
        # Ensure sigma has correct shape for broadcasting
        sigma = append_dims(sigma.flatten(), x.ndim)

        # Compute scaling factors
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Apply preconditioning
        model_output = self.model(c_in * x, c_noise)

        return c_skip * x + c_out * model_output


class EDMPrecondConditional(nn.Module):
    """EDM preconditioning wrapper for conditional denoising.

    Extends EDMPrecond with support for:
    - Conditional inputs (class labels, embeddings, etc.)
    - Classifier-Free Guidance (CFG) training via condition dropout

    Parameters
    ----------
    model : nn.Module
        Conditional denoiser network. Should accept (x, noise_level, condition).
    sigma_data : float, optional
        Data standard deviation. Default: 0.5.
    uncond_prob : float, optional
        Probability of dropping condition during training for CFG. Default: 0.1.

    Examples
    --------
    >>> base_model = MyConditionalDenoiser()
    >>> model = EDMPrecondConditional(base_model, uncond_prob=0.1)
    >>> x_noisy = torch.randn(4, 3, 32, 32)
    >>> sigma = torch.ones(4)
    >>> condition = torch.randint(0, 10, (4,))  # class labels
    >>> x_denoised = model(x_noisy, sigma, condition)
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_data: float = 0.5,
        uncond_prob: float = 0.1,
    ) -> None:
        """Initialize conditional EDM preconditioning wrapper.

        Parameters
        ----------
        model : nn.Module
            Conditional denoiser network.
        sigma_data : float, optional
            Data standard deviation. Default: 0.5.
        uncond_prob : float, optional
            Condition dropout probability for CFG training. Default: 0.1.
        """
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.uncond_prob = uncond_prob

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Forward pass with conditional EDM preconditioning.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input of shape [B, ...].
        sigma : torch.Tensor
            Noise level of shape [B] or [B, 1, ...].
        condition : torch.Tensor, optional
            Condition tensor (class labels, embeddings, etc.).
        force_uncond : bool, optional
            If True, force unconditional prediction (for CFG inference).

        Returns
        -------
        torch.Tensor
            Denoised prediction of shape [B, ...].
        """
        sigma = append_dims(sigma.flatten(), x.ndim)

        # Compute scaling factors
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Process condition (dropout during training, force uncond during inference)
        processed_cond = self._process_condition(x, condition, force_uncond)

        # Apply preconditioning
        model_output = self.model(c_in * x, c_noise, processed_cond)

        return c_skip * x + c_out * model_output

    def _process_condition(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        force_uncond: bool,
    ) -> Optional[torch.Tensor]:
        """Process condition with potential dropout.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (used for batch size and device).
        condition : torch.Tensor, optional
            Original condition.
        force_uncond : bool
            Whether to force unconditional (return None).

        Returns
        -------
        torch.Tensor or None
            Processed condition with potential dropout applied.
        """
        if force_uncond or condition is None:
            return None

        # Apply condition dropout during training
        if self.training and self.uncond_prob > 0:
            batch_size = x.shape[0]
            drop_mask = torch.rand(batch_size, device=x.device) < self.uncond_prob

            if drop_mask.any():
                # Create a copy and zero out dropped conditions
                condition = condition.clone()
                if condition.ndim == 1:
                    condition[drop_mask] = 0  # Assuming 0 is the "uncond" token
                else:
                    condition[drop_mask] = 0

        return condition
