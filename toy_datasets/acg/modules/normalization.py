"""Adaptive Normalization Modules.

This module provides normalization layers that can be modulated by
conditioning information (class embeddings, timestep, etc.).

These are essential building blocks for conditional generation:
- AdaGroupNorm: Adaptive Group Normalization
- AdaLayerNorm: Adaptive Layer Normalization
- FiLM: Feature-wise Linear Modulation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaGroupNorm(nn.Module):
    """Adaptive Group Normalization.

    Modulates GroupNorm parameters based on a conditioning vector.
    The condition predicts per-channel scale (γ) and shift (β) parameters.

    This is commonly used in class-conditional and time-conditional models.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    cond_dim : int
        Dimension of conditioning vector.
    num_groups : int, optional
        Number of groups for GroupNorm. Default: 32.

    Examples
    --------
    >>> ada_gn = AdaGroupNorm(256, 512, num_groups=32)
    >>> x = torch.randn(4, 256, 16, 16)  # [B, C, H, W]
    >>> cond = torch.randn(4, 512)  # [B, cond_dim]
    >>> out = ada_gn(x, cond)  # [B, C, H, W]
    """

    def __init__(
        self,
        num_channels: int,
        cond_dim: int,
        num_groups: int = 32,
    ) -> None:
        """Initialize Adaptive Group Normalization."""
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)  # scale and shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape [B, C, H, W].
        cond : torch.Tensor
            Conditioning vector of shape [B, cond_dim].

        Returns
        -------
        torch.Tensor
            Modulated features of shape [B, C, H, W].
        """
        # Normalize
        x = self.norm(x)

        # Get adaptive parameters
        params = self.proj(cond)  # [B, C*2]
        scale, shift = params.chunk(2, dim=1)  # [B, C] each

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        # Apply modulation: (1 + scale) * x + shift
        return x * (1 + scale) + shift


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization.

    Similar to AdaGroupNorm but uses LayerNorm instead.
    Commonly used in Transformer architectures.

    Parameters
    ----------
    num_channels : int
        Number of channels/features.
    cond_dim : int
        Dimension of conditioning vector.

    Examples
    --------
    >>> ada_ln = AdaLayerNorm(256, 512)
    >>> x = torch.randn(4, 64, 256)  # [B, N, C]
    >>> cond = torch.randn(4, 512)  # [B, cond_dim]
    >>> out = ada_ln(x, cond)  # [B, N, C]
    """

    def __init__(self, num_channels: int, cond_dim: int) -> None:
        """Initialize Adaptive Layer Normalization."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape [B, N, C] or [B, C].
        cond : torch.Tensor
            Conditioning vector of shape [B, cond_dim].

        Returns
        -------
        torch.Tensor
            Modulated features of shape [B, N, C] or [B, C].
        """
        x = self.norm(x)

        params = self.proj(cond)  # [B, C*2]
        scale, shift = params.chunk(2, dim=1)  # [B, C] each

        if x.ndim == 3:
            # [B, C] -> [B, 1, C] for broadcasting with [B, N, C]
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return x * (1 + scale) + shift


class FiLM(nn.Module):
    """Feature-wise Linear Modulation.

    A simple conditioning mechanism that applies:
        FiLM(x) = γ * x + β

    where γ and β are predicted from the conditioning vector.

    Parameters
    ----------
    feature_dim : int
        Dimension of the features to modulate.
    cond_dim : int
        Dimension of the conditioning vector.

    Examples
    --------
    >>> film = FiLM(256, 512)
    >>> x = torch.randn(4, 256)  # [B, D]
    >>> cond = torch.randn(4, 512)  # [B, cond_dim]
    >>> out = film(x, cond)  # [B, D]
    """

    def __init__(self, feature_dim: int, cond_dim: int) -> None:
        """Initialize FiLM."""
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        cond : torch.Tensor
            Conditioning vector.

        Returns
        -------
        torch.Tensor
            Modulated features.
        """
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)

        # Handle different input shapes
        if x.ndim > gamma.ndim:
            # Add dimensions for broadcasting
            for _ in range(x.ndim - gamma.ndim):
                gamma = gamma.unsqueeze(-1)
                beta = beta.unsqueeze(-1)

        return gamma * x + beta
