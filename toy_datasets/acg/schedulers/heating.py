"""Heating Schedules for ACG.

This module defines when and how to inject heat (noise) during sampling.
Heating schedules are independent of synchronization schedules.

Classes
-------
BaseHeatingSchedule : Abstract base class
NoHeating : Standard diffusion without heating
UniformHeating : Uniform heating within a range
DecayingHeating : Heating with decaying intensity
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from acg.schedulers.base import HeatConfig


class BaseHeatingSchedule(ABC):
    """Abstract base class for heating schedules.

    A heating schedule determines when to inject noise (heat) and
    how much to inject at each timestep.
    """

    @abstractmethod
    def get_heat_config(self, step_idx: int, n_steps: int) -> Optional[HeatConfig]:
        """Get heating configuration for a given step.

        Parameters
        ----------
        step_idx : int
            Current step index (0 to n_steps-1).
        n_steps : int
            Total number of steps.

        Returns
        -------
        HeatConfig or None
            Heating configuration if heating should occur at this step,
            None otherwise.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NoHeating(BaseHeatingSchedule):
    """No heating - standard diffusion sampling.

    This produces standard deterministic diffusion without any
    resampling or noise injection.
    """

    def get_heat_config(self, step_idx: int, n_steps: int) -> Optional[HeatConfig]:
        """Never heat."""
        return None

    def __repr__(self) -> str:
        return "NoHeating()"


class UniformHeating(BaseHeatingSchedule):
    """Uniform heating within a specified range.

    Applies the same heating configuration at regular intervals
    within [start_step, end_step).

    Parameters
    ----------
    heat_jump : int
        Number of steps to jump back after each segment.
    num_iterations : int
        Number of times to iterate each segment.
    heat_height : float
        Heat height H in (0, 1], controls how far back to actually heat.
        H=1.0 means heat back to the full jump distance.
    start_step : int, optional
        First step to apply heating (inclusive). Default: 0.
    end_step : int, optional
        Last step to apply heating (exclusive). Default: n_steps.

    Examples
    --------
    >>> heating = UniformHeating(heat_jump=2, num_iterations=3)
    >>> heating.get_heat_config(10, 50)
    HeatConfig(jump=2, iterations=3, height=1.0)
    """

    def __init__(
        self,
        heat_jump: int,
        num_iterations: int,
        heat_height: float = 1.0,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> None:
        if heat_jump < 1:
            raise ValueError(f"heat_jump must be >= 1, got {heat_jump}")
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if not (0 < heat_height <= 1):
            raise ValueError(f"heat_height must be in (0, 1], got {heat_height}")

        self.heat_jump = heat_jump
        self.num_iterations = num_iterations
        self.heat_height = heat_height
        self.start_step = start_step
        self.end_step = end_step

    def get_heat_config(self, step_idx: int, n_steps: int) -> Optional[HeatConfig]:
        """Get uniform heating config within the active range."""
        start = self.start_step if self.start_step is not None else 0
        end = self.end_step if self.end_step is not None else n_steps

        if start <= step_idx < end:
            return HeatConfig(
                jump=self.heat_jump,
                iterations=self.num_iterations,
                height=self.heat_height,
            )
        return None

    def __repr__(self) -> str:
        return (
            f"UniformHeating(heat_jump={self.heat_jump}, "
            f"num_iterations={self.num_iterations}, "
            f"heat_height={self.heat_height}, "
            f"start_step={self.start_step}, end_step={self.end_step})"
        )


class DecayingHeating(BaseHeatingSchedule):
    """Heating with decaying intensity over time.

    The number of iterations decreases as we progress through the
    diffusion process, spending more compute on early (high-noise) steps.

    Parameters
    ----------
    max_iterations : int
        Maximum iterations at the start.
    min_iterations : int
        Minimum iterations at the end.
    heat_jump : int
        Number of steps to jump back.
    heat_height : float
        Heat height H in (0, 1].
    start_step : int, optional
        First step to apply heating.
    end_step : int, optional
        Last step to apply heating.
    decay : str
        Decay type: "linear" or "exponential".
    """

    def __init__(
        self,
        max_iterations: int,
        min_iterations: int = 1,
        heat_jump: int = 2,
        heat_height: float = 1.0,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        decay: str = "linear",
    ) -> None:
        if max_iterations < min_iterations:
            raise ValueError(
                f"max_iterations must be >= min_iterations, "
                f"got max={max_iterations}, min={min_iterations}"
            )
        if decay not in ["linear", "exponential"]:
            raise ValueError(f"decay must be 'linear' or 'exponential', got {decay}")

        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.heat_jump = heat_jump
        self.heat_height = heat_height
        self.start_step = start_step
        self.end_step = end_step
        self.decay = decay

    def get_heat_config(self, step_idx: int, n_steps: int) -> Optional[HeatConfig]:
        """Get heating config with decaying iterations."""
        start = self.start_step if self.start_step is not None else 0
        end = self.end_step if self.end_step is not None else n_steps

        if not (start <= step_idx < end):
            return None

        # Compute progress within the heating range
        range_length = end - start
        if range_length <= 1:
            progress = 0.0
        else:
            progress = (step_idx - start) / (range_length - 1)

        # Compute iterations based on decay
        if self.decay == "linear":
            iterations = int(
                self.max_iterations - progress * (self.max_iterations - self.min_iterations)
            )
        else:  # exponential
            ratio = self.min_iterations / self.max_iterations
            iterations = int(self.max_iterations * (ratio ** progress))

        iterations = max(self.min_iterations, iterations)

        return HeatConfig(
            jump=self.heat_jump,
            iterations=iterations,
            height=self.heat_height,
        )

    def __repr__(self) -> str:
        return (
            f"DecayingHeating(max_iterations={self.max_iterations}, "
            f"min_iterations={self.min_iterations}, "
            f"heat_jump={self.heat_jump}, decay='{self.decay}')"
        )
