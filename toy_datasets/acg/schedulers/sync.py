"""Synchronization Schedules for ACG.

This module defines when to apply Consensus (synchronization) during sampling.
Sync schedules are independent of heating schedules.

Classes
-------
BaseSyncSchedule : Abstract base class
NoSync : Never synchronize (Greedy baseline)
AlwaysSync : Synchronize at every step
RangeSync : Synchronize within a specified range
StrideSync : Synchronize at regular intervals
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseSyncSchedule(ABC):
    """Abstract base class for synchronization schedules.

    A sync schedule determines when to apply the Consensus operator
    to align shared variables across different contexts.
    """

    @abstractmethod
    def should_sync(self, step_idx: int, n_steps: int, iteration: int) -> bool:
        """Determine whether to synchronize at this step.

        Parameters
        ----------
        step_idx : int
            Current step index (0 to n_steps-1).
        n_steps : int
            Total number of steps.
        iteration : int
            Current iteration within a heating segment.
            0 = first visit (original pass)
            >0 = after reheating (resample pass)

        Returns
        -------
        bool
            True if Consensus should be applied at this step.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NoSync(BaseSyncSchedule):
    """Never synchronize - Greedy baseline.

    Each context evolves independently without any consensus.
    This corresponds to the "Greedy" baseline in the paper.
    """

    def should_sync(self, step_idx: int, n_steps: int, iteration: int) -> bool:
        """Never sync."""
        return False

    def __repr__(self) -> str:
        return "NoSync()"


class AlwaysSync(BaseSyncSchedule):
    """Always synchronize at every step.

    This enforces maximum consistency but may hurt sample quality
    due to accumulation of consensus artifacts.

    Parameters
    ----------
    first_visit_only : bool
        If True, only sync on first visit (iteration=0).
        If False, sync on every iteration.
    """

    def __init__(self, first_visit_only: bool = True) -> None:
        self.first_visit_only = first_visit_only

    def should_sync(self, step_idx: int, n_steps: int, iteration: int) -> bool:
        """Sync at every step, respecting first_visit_only."""
        if self.first_visit_only and iteration > 0:
            return False
        return True

    def __repr__(self) -> str:
        return f"AlwaysSync(first_visit_only={self.first_visit_only})"


class RangeSync(BaseSyncSchedule):
    """Synchronize within a specified step range.

    Parameters
    ----------
    start_step : int, optional
        First step to sync (inclusive). Default: 0.
    end_step : int, optional
        Last step to sync (exclusive). Default: n_steps.
    first_visit_only : bool
        If True, only sync on first visit to each step.
        This implements the "first-visit only" policy from the paper.

    Examples
    --------
    >>> sync = RangeSync(start_step=5, end_step=45, first_visit_only=True)
    >>> sync.should_sync(10, 50, iteration=0)  # First visit
    True
    >>> sync.should_sync(10, 50, iteration=1)  # After reheat
    False
    """

    def __init__(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        first_visit_only: bool = True,
    ) -> None:
        self.start_step = start_step
        self.end_step = end_step
        self.first_visit_only = first_visit_only

    def should_sync(self, step_idx: int, n_steps: int, iteration: int) -> bool:
        """Sync within range, respecting first_visit_only."""
        # Check first_visit_only
        if self.first_visit_only and iteration > 0:
            return False

        # Check range
        start = self.start_step if self.start_step is not None else 0
        end = self.end_step if self.end_step is not None else n_steps

        return start <= step_idx < end

    def __repr__(self) -> str:
        return (
            f"RangeSync(start_step={self.start_step}, "
            f"end_step={self.end_step}, "
            f"first_visit_only={self.first_visit_only})"
        )


class StrideSync(BaseSyncSchedule):
    """Synchronize at regular intervals.

    Parameters
    ----------
    stride : int
        Sync every `stride` steps. E.g., stride=2 syncs at steps 0, 2, 4, ...
    start_step : int, optional
        First step to consider for sync.
    end_step : int, optional
        Last step to consider for sync.
    first_visit_only : bool
        If True, only sync on first visit.

    Examples
    --------
    >>> sync = StrideSync(stride=5)
    >>> [sync.should_sync(i, 50, 0) for i in range(10)]
    [True, False, False, False, False, True, False, False, False, False]
    """

    def __init__(
        self,
        stride: int,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        first_visit_only: bool = True,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.stride = stride
        self.start_step = start_step
        self.end_step = end_step
        self.first_visit_only = first_visit_only

    def should_sync(self, step_idx: int, n_steps: int, iteration: int) -> bool:
        """Sync at stride intervals."""
        if self.first_visit_only and iteration > 0:
            return False

        start = self.start_step if self.start_step is not None else 0
        end = self.end_step if self.end_step is not None else n_steps

        if not (start <= step_idx < end):
            return False

        # Check if step_idx is at a stride boundary relative to start
        return (step_idx - start) % self.stride == 0

    def __repr__(self) -> str:
        return (
            f"StrideSync(stride={self.stride}, "
            f"start_step={self.start_step}, end_step={self.end_step}, "
            f"first_visit_only={self.first_visit_only})"
        )


class SigmaRangeSync(BaseSyncSchedule):
    """Synchronize within a specified sigma (noise level) range.

    This is useful when you want to sync based on noise level rather
    than step index, which is more robust to different schedule lengths.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum sigma to sync at (inclusive).
    sigma_max : float, optional
        Maximum sigma to sync at (inclusive).
    first_visit_only : bool
        If True, only sync on first visit.

    Note
    ----
    This schedule requires sigma values to be passed during schedule
    generation. It's currently a placeholder for future implementation.
    """

    def __init__(
        self,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        first_visit_only: bool = True,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.first_visit_only = first_visit_only
        # Store sigma values when schedule is generated
        self._sigmas: Optional[list] = None

    def set_sigmas(self, sigmas: list) -> None:
        """Set sigma values for range checking."""
        self._sigmas = list(sigmas)

    def should_sync(self, step_idx: int, n_steps: int, iteration: int) -> bool:
        """Sync within sigma range."""
        if self.first_visit_only and iteration > 0:
            return False

        if self._sigmas is None:
            # Fall back to always sync if sigmas not set
            return True

        if step_idx >= len(self._sigmas):
            return False

        sigma = self._sigmas[step_idx]

        if self.sigma_min is not None and sigma < self.sigma_min:
            return False
        if self.sigma_max is not None and sigma > self.sigma_max:
            return False

        return True

    def __repr__(self) -> str:
        return (
            f"SigmaRangeSync(sigma_min={self.sigma_min}, "
            f"sigma_max={self.sigma_max}, "
            f"first_visit_only={self.first_visit_only})"
        )
