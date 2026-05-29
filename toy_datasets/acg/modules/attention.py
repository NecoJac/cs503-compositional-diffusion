"""Attention Modules.

This module provides attention mechanisms for diffusion models,
particularly for incorporating sequence conditions (text, images).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Cross Attention module.

    Used to inject sequence conditions (e.g., text embeddings) into the model.
    Query comes from image features, Key/Value come from the condition.

    Parameters
    ----------
    query_dim : int
        Dimension of query features (from image).
    context_dim : int
        Dimension of context features (from condition).
    num_heads : int, optional
        Number of attention heads. Default: 8.
    head_dim : int, optional
        Dimension per head. Default: 64.
    dropout : float, optional
        Dropout probability. Default: 0.0.

    Examples
    --------
    >>> cross_attn = CrossAttention(256, 768, num_heads=8)
    >>> x = torch.randn(4, 64, 256)  # [B, N, query_dim] image features
    >>> context = torch.randn(4, 77, 768)  # [B, M, context_dim] text embeddings
    >>> out = cross_attn(x, context)  # [B, N, query_dim]
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """Initialize Cross Attention."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        # Linear projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross attention.

        Parameters
        ----------
        x : torch.Tensor
            Query features of shape [B, N, query_dim].
        context : torch.Tensor
            Context (key/value) features of shape [B, M, context_dim].

        Returns
        -------
        torch.Tensor
            Attention output of shape [B, N, query_dim].
        """
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.to_q(x)  # [B, N, inner_dim]
        k = self.to_k(context)  # [B, M, inner_dim]
        v = self.to_v(context)  # [B, M, inner_dim]

        # Reshape for multi-head attention
        # [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, N, head_dim]

        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)

        # Output projection
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self Attention module.

    Standard self-attention where query, key, value all come from
    the same input.

    Parameters
    ----------
    dim : int
        Input/output dimension.
    num_heads : int, optional
        Number of attention heads. Default: 8.
    head_dim : int, optional
        Dimension per head. Default: 64.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """Initialize Self Attention."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self attention.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape [B, N, dim].

        Returns
        -------
        torch.Tensor
            Output features of shape [B, N, dim].
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.to_qkv(x)  # [B, N, inner_dim * 3]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, N, inner_dim]

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)
        return self.to_out(out)
