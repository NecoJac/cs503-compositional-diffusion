"""Denoisers.

This module provides denoiser implementations for diffusion models.

The exact denoisers compute the optimal denoising function analytically,
without requiring neural network training. They are useful for:
- Understanding diffusion theory
- Small-scale experiments
- Validating sampling algorithms

Available denoisers:
- ExactDenoiser: Unconditional exact denoiser
- ConditionalExactDenoiser: Conditional exact denoiser with CFG support
- MultiConditionDenoiser: Multi-condition composable denoiser
"""

from acg.denoisers.base import Denoiser
from acg.denoisers.conditional import ConditionalExactDenoiser
from acg.denoisers.exact import ExactDenoiser
from acg.denoisers.multi_condition import MultiAttributeDenoiser, MultiConditionDenoiser
from acg.denoisers.factor import FactorGraphDenoiserCondC, FactorGraphDenoiser   

__all__ = [
    "Denoiser",
    "ExactDenoiser",
    "ConditionalExactDenoiser",
    "MultiConditionDenoiser",
    "MultiAttributeDenoiser",
    "FactorGraphDenoiser",
    "FactorGraphDenoiserCondC"
]
