"""Toy Datasets for Testing.

This module provides simple 2D datasets for testing and visualizing
diffusion models and conditional generation.

Available datasets:
- Concentric circles
- Spirals
- Checkerboard
- Gaussian mixture
- Two moons
- Mickey Mouse
- Smiley face
- Letter shapes

All datasets return (data, labels) tensors suitable for
conditional generation experiments.
"""

from acg.datasets.registry import DATASET_REGISTRY, get_dataset, list_datasets, register_dataset
from acg.datasets.toy_2d import (
    make_checkerboard,
    make_concentric_circles,
    make_gaussian_mixture,
    make_half_helix,
    make_letter_shapes,
    make_mickey_mouse,
    make_quadrant_dataset,
    make_smiley_face,
    make_spirals,
    make_swiss_roll_slices,
    make_two_moons,
)

__all__ = [
    # Generators
    "make_concentric_circles",
    "make_spirals",
    "make_checkerboard",
    "make_gaussian_mixture",
    "make_two_moons",
    "make_half_helix",
    "make_swiss_roll_slices",
    "make_mickey_mouse",
    "make_smiley_face",
    "make_letter_shapes",
    "make_quadrant_dataset",
    # Registry
    "get_dataset",
    "list_datasets",
    "register_dataset",
    "DATASET_REGISTRY",
]
