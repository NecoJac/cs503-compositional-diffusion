"""Embedding Modules.

This module provides embedding layers for encoding various types
of conditions (class labels, timesteps, etc.) into continuous vectors.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ClassEmbedding(nn.Module):
    """Class label embedding.

    Converts discrete class labels into continuous embedding vectors.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    embed_dim : int
        Embedding dimension.

    Examples
    --------
    >>> embed = ClassEmbedding(10, 256)
    >>> labels = torch.randint(0, 10, (4,))
    >>> embeddings = embed(labels)  # [4, 256]
    """

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        """Initialize class embedding."""
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Embed class labels.

        Parameters
        ----------
        class_labels : torch.Tensor
            Class indices of shape [B].

        Returns
        -------
        torch.Tensor
            Class embeddings of shape [B, embed_dim].
        """
        return self.embedding(class_labels)


class TimestepEmbedding(nn.Module):
    """Timestep/noise level embedding.

    Uses sinusoidal positional encoding followed by an MLP.
    Similar to the positional encoding in Transformers.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension.
    hidden_dim : int, optional
        Hidden dimension for MLP. Default: 4 * embed_dim.

    Examples
    --------
    >>> embed = TimestepEmbedding(256)
    >>> timesteps = torch.randn(4)  # log(sigma) / 4
    >>> embeddings = embed(timesteps)  # [4, 256]
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """Initialize timestep embedding."""
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Parameters
        ----------
        timesteps : torch.Tensor
            Timesteps or log(sigma)/4 values of shape [B].

        Returns
        -------
        torch.Tensor
            Timestep embeddings of shape [B, embed_dim].
        """
        # Sinusoidal encoding
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)

        # timesteps: [B], emb: [half_dim]
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, embed_dim]

        # MLP projection
        return self.mlp(emb)


class CombinedEmbedding(nn.Module):
    """Combined embedding for multiple conditions.

    Combines timestep and class embeddings (and optionally others)
    into a single conditioning vector.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension.
    num_classes : int, optional
        Number of classes. If None, no class embedding.
    use_timestep : bool, optional
        Whether to include timestep embedding. Default: True.

    Examples
    --------
    >>> embed = CombinedEmbedding(256, num_classes=10)
    >>> timesteps = torch.randn(4)
    >>> labels = torch.randint(0, 10, (4,))
    >>> combined = embed(timesteps=timesteps, class_labels=labels)  # [4, 256]
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: Optional[int] = None,
        use_timestep: bool = True,
    ) -> None:
        """Initialize combined embedding."""
        super().__init__()
        self.embed_dim = embed_dim
        self.use_timestep = use_timestep
        self.use_class = num_classes is not None

        if use_timestep:
            self.timestep_embed = TimestepEmbedding(embed_dim)

        if self.use_class:
            self.class_embed = ClassEmbedding(num_classes, embed_dim)

    def forward(
        self,
        timesteps: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined embedding.

        Parameters
        ----------
        timesteps : torch.Tensor, optional
            Timesteps of shape [B].
        class_labels : torch.Tensor, optional
            Class labels of shape [B].

        Returns
        -------
        torch.Tensor
            Combined embedding of shape [B, embed_dim].
        """
        embeddings = []

        if self.use_timestep and timesteps is not None:
            embeddings.append(self.timestep_embed(timesteps))

        if self.use_class and class_labels is not None:
            embeddings.append(self.class_embed(class_labels))

        if len(embeddings) == 0:
            raise ValueError("At least one embedding must be provided")

        if len(embeddings) == 1:
            return embeddings[0]

        # Sum embeddings (could also concatenate)
        return sum(embeddings)
