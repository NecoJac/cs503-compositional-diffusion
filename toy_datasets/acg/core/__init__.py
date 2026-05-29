"""ACG Core Components.

This module provides the core building blocks for EDM (Elucidating the Design Space
of Diffusion-Based Generative Models) based diffusion models.

Core components:
- Noise schedules (Karras schedule)
- EDM preconditioning wrappers
- Training loss functions
"""

from acg.core.losses import EDMLoss, EDMLossConditional
from acg.core.precondition import EDMPrecond, EDMPrecondConditional
from acg.core.schedules import get_sigmas_exponential, get_sigmas_karras, get_sigmas_linear

__all__ = [
    # Schedules
    "get_sigmas_karras",
    "get_sigmas_linear",
    "get_sigmas_exponential",
    # Preconditioning
    "EDMPrecond",
    "EDMPrecondConditional",
    # Losses
    "EDMLoss",
    "EDMLossConditional",
]
