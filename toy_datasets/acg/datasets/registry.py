"""Dataset Registry.

Provides a registry for easy access to datasets by name.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch

# Dataset registry: name -> generator function
DATASET_REGISTRY: Dict[str, Callable[..., Tuple[torch.Tensor, ...]]] = {}


def register_dataset(name: str) -> Callable:
    """Decorator to register a dataset generator.

    Parameters
    ----------
    name : str
        Name to register the dataset under.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @register_dataset("my_dataset")
    ... def make_my_dataset(n_samples=100):
    ...     return torch.randn(n_samples, 2), torch.zeros(n_samples)
    """
    def decorator(func: Callable) -> Callable:
        DATASET_REGISTRY[name] = func
        return func
    return decorator


def get_dataset(name: str, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
    """Get a dataset by name.

    Parameters
    ----------
    name : str
        Dataset name.
    **kwargs
        Arguments passed to dataset generator.

    Returns
    -------
    tuple of torch.Tensor
        Dataset tensors (data, labels, ...).

    Raises
    ------
    KeyError
        If dataset name not found.

    Examples
    --------
    >>> data, labels = get_dataset("circles", n_samples_per_class=100)
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")

    return DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> list:
    """List all registered datasets.

    Returns
    -------
    list of str
        Available dataset names.
    """
    return list(DATASET_REGISTRY.keys())


# Register built-in datasets
def _register_builtin_datasets() -> None:
    """Register built-in 2D toy datasets."""
    from acg.datasets.toy_2d import (
        make_checkerboard,
        make_concentric_circles,
        make_gaussian_mixture,
        make_letter_shapes,
        make_mickey_mouse,
        make_quadrant_dataset,
        make_smiley_face,
        make_spirals,
        make_swiss_roll_slices,
        make_two_moons,
    )

    DATASET_REGISTRY["circles"] = make_concentric_circles
    DATASET_REGISTRY["spirals"] = make_spirals
    DATASET_REGISTRY["checkerboard"] = make_checkerboard
    DATASET_REGISTRY["gaussian"] = make_gaussian_mixture
    DATASET_REGISTRY["moons"] = make_two_moons
    DATASET_REGISTRY["swiss_roll"] = make_swiss_roll_slices
    DATASET_REGISTRY["mickey"] = make_mickey_mouse
    DATASET_REGISTRY["smiley"] = make_smiley_face
    DATASET_REGISTRY["letter_A"] = lambda **kw: make_letter_shapes("A", **kw)
    DATASET_REGISTRY["letter_X"] = lambda **kw: make_letter_shapes("X", **kw)
    DATASET_REGISTRY["letter_Y"] = lambda **kw: make_letter_shapes("Y", **kw)
    DATASET_REGISTRY["quadrant"] = make_quadrant_dataset


# Auto-register on import
_register_builtin_datasets()
