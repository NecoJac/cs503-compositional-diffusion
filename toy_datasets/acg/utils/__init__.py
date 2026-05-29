"""Utility Functions.

This module provides common utility functions used across the ACG package.
"""

from acg.utils.tensor import append_dims
from acg.utils.visualization import (
    plot_2d_data,
    plot_cfg_scale_comparison,
    plot_conditional_generation_comparison,
    plot_vector_field,
    plot_factor_vector_field,
    plot_denoising_field,
    plot_sampling_trajectory,
    plot_xyc_data,
    plot_xyc_sampling_trajectories,
)

__all__ = [
    "append_dims",
    "plot_2d_data",
    "plot_xyc_data",
    "plot_xyc_sampling_trajectories",
    "plot_denoising_field",
    "plot_sampling_trajectory",
    "plot_conditional_generation_comparison",
    "plot_cfg_scale_comparison",
    "plot_vector_field",
    "plot_factor_vector_field",
]
