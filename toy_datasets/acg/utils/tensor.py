"""Tensor Utilities.

Common tensor manipulation functions used throughout the package.
"""

from __future__ import annotations

import torch


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append dimensions to a tensor for broadcasting.

    Adds singleton dimensions at the end of the tensor until it has
    the target number of dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    target_dims : int
        Target number of dimensions.

    Returns
    -------
    torch.Tensor
        Tensor with shape [..., 1, 1, ...] having target_dims dimensions.

    Examples
    --------
    >>> x = torch.randn(4)  # shape: [4]
    >>> append_dims(x, 4).shape
    torch.Size([4, 1, 1, 1])

    >>> sigma = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma_4d = append_dims(sigma, 4)  # For [B, C, H, W] tensors
    >>> sigma_4d.shape
    torch.Size([3, 1, 1, 1])
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"Cannot append negative dims. x.ndim={x.ndim}, target_dims={target_dims}"
        )
    return x.view(*x.shape, *([1] * dims_to_append))


def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0, 1] to [-1, 1].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values in [0, 1].

    Returns
    -------
    torch.Tensor
        Normalized tensor with values in [-1, 1].
    """
    return x * 2 - 1


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    """Unnormalize tensor from [-1, 1] to [0, 1].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values in [-1, 1].

    Returns
    -------
    torch.Tensor
        Unnormalized tensor with values in [0, 1].
    """
    return (x + 1) / 2


def flatten_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Flatten all dimensions except batch dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [B, ...].

    Returns
    -------
    torch.Tensor
        Flattened tensor of shape [B, D].
    """
    return x.flatten(start_dim=1)
