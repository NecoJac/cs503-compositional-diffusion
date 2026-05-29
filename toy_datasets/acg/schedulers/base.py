"""Base types for ACG Schedulers.

This module defines the core data structures used by the scheduling system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Union

import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, Sequence[float]]


class StepType(Enum):
    """Type of sampling step."""

    COOL = "cool"  # Standard Diffusion (Cooling/Denoising)
    HEAT = "heat"  # Noise injection (Heating/Forward diffusion)


@dataclass
class SamplingStep:
    """Represents a single step in the sampling schedule.

    For COOL steps:
        - sigma_from: current noise level (higher)
        - sigma_to: target noise level (lower)

    For HEAT steps:
        - sigma_from: current noise level (lower)
        - sigma_to: target noise level (higher, going back)

    Attributes
    ----------
    sigma_from : float
        Starting sigma (noise level).
    sigma_to : float
        Target sigma.
    step_type : StepType
        COOL (denoise) or HEAT (add noise).
    step_idx : int
        Original step index in the base schedule.
    iteration : int
        Which iteration this step belongs to (0 = first pass).
    is_sync_step : bool
        Whether to apply Consensus in this step.
    gamma : float
        Gamma parameter for stochastic sampling.
    step_scale : float
        Step scale factor.
    noise_scale : float
        Noise scale factor.
    """

    sigma_from: float
    sigma_to: float
    step_type: StepType
    step_idx: int = 0
    iteration: int = 0
    is_sync_step: bool = False
    gamma: float = 0.0
    step_scale: float = 1.0
    noise_scale: float = 1.0


@dataclass
class HeatConfig:
    """Configuration for a heating operation.

    Attributes
    ----------
    jump : int
        Number of steps to jump back (j_t in the paper).
    iterations : int
        Number of resampling iterations (K_t in the paper).
    height : float
        Heat height H in (0, 1], scales the noise injection.
        H=1.0 means full jump back, H<1.0 means partial.
    """

    jump: int
    iterations: int
    height: float = 1.0

    def __post_init__(self) -> None:
        if self.jump < 1:
            raise ValueError(f"jump must be >= 1, got {self.jump}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if not (0 < self.height <= 1):
            raise ValueError(f"height must be in (0, 1], got {self.height}")
