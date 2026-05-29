"""ACG Scheduler - Combines Heating and Sync Schedules.

This module provides the ACGScheduler that combines independent
heating and synchronization schedules to generate a complete
sampling schedule with COOL/HEAT steps and sync flags.
"""

from __future__ import annotations

from math import sqrt
from typing import List, Optional, Sequence, Union

import numpy as np

from acg.schedulers.base import ArrayLike, HeatConfig, SamplingStep, StepType
from acg.schedulers.heating import BaseHeatingSchedule, NoHeating
from acg.schedulers.sync import BaseSyncSchedule, NoSync, SigmaRangeSync


class ACGScheduler:
    """Scheduler that combines heating and synchronization schedules.

    This is the main scheduler for ACG (Annealed Co-Generation).
    It takes a noise schedule (sigmas) and produces a list of
    SamplingStep objects that include both COOL and HEAT operations,
    with sync flags determined by the sync schedule.

    Parameters
    ----------
    heating_schedule : BaseHeatingSchedule, optional
        Controls when and how to inject heat. Default: NoHeating().
    sync_schedule : BaseSyncSchedule, optional
        Controls when to apply Consensus. Default: NoSync().

    Examples
    --------
    >>> from acg.schedulers import ACGScheduler, UniformHeating, RangeSync
    >>> from acg import get_sigmas_karras
    >>>
    >>> # Create scheduler with heating and sync
    >>> scheduler = ACGScheduler(
    ...     heating_schedule=UniformHeating(heat_jump=2, num_iterations=3),
    ...     sync_schedule=RangeSync(first_visit_only=True),
    ... )
    >>>
    >>> # Generate schedule
    >>> sigmas = get_sigmas_karras(50, device='cpu')
    >>> steps = scheduler.get_schedule(sigmas.numpy())
    >>> print(f"Generated {len(steps)} steps")

    Notes
    -----
    The schedule generation follows Algorithm 1 from the ACG paper:

    1. Iterate through the noise schedule
    2. At each segment, check if heating is configured
    3. If heating: perform K iterations of COOL-HEAT cycles
    4. For each COOL step, check sync schedule to set is_sync_step
    5. Sync follows "first-visit only" policy when configured
    """

    def __init__(
        self,
        heating_schedule: Optional[BaseHeatingSchedule] = None,
        sync_schedule: Optional[BaseSyncSchedule] = None,
    ) -> None:
        self.heating = heating_schedule or NoHeating()
        self.sync = sync_schedule or NoSync()

    def get_schedule(
        self,
        sigmas: ArrayLike,
        gammas: Optional[ArrayLike] = None,
        step_scales: Optional[ArrayLike] = None,
        noise_scales: Optional[ArrayLike] = None,
    ) -> List[SamplingStep]:
        """Generate the complete sampling schedule.

        Parameters
        ----------
        sigmas : ArrayLike
            Noise levels [n_steps + 1], from high (sigma_max) to low (0).
        gammas : ArrayLike, optional
            Gamma values for stochastic sampling.
        step_scales : ArrayLike, optional
            Step scale factors.
        noise_scales : ArrayLike, optional
            Noise scale factors.

        Returns
        -------
        List[SamplingStep]
            Complete schedule with COOL/HEAT steps and sync flags.
        """
        sigmas = np.asarray(sigmas)
        n_steps = len(sigmas) - 1

        # Prepare optional arrays
        if gammas is None:
            gammas = np.zeros(len(sigmas))
        else:
            gammas = np.asarray(gammas)

        if step_scales is None:
            step_scales = np.ones(n_steps)
        else:
            step_scales = np.asarray(step_scales)

        if noise_scales is None:
            noise_scales = np.ones(n_steps)
        else:
            noise_scales = np.asarray(noise_scales)

        # Set sigmas for SigmaRangeSync if applicable
        if isinstance(self.sync, SigmaRangeSync):
            self.sync.set_sigmas(list(sigmas))

        steps: List[SamplingStep] = []
        i = 0  # Current step index

        while i < n_steps:
            # Check if we have heating at this step
            heat_config = self.heating.get_heat_config(i, n_steps)

            if heat_config is None:
                # No heating: single COOL step
                should_sync = self.sync.should_sync(i, n_steps, iteration=0)
                steps.append(
                    SamplingStep(
                        sigma_from=float(sigmas[i]),
                        sigma_to=float(sigmas[i + 1]),
                        step_type=StepType.COOL,
                        step_idx=i,
                        iteration=0,
                        is_sync_step=should_sync,
                        gamma=float(gammas[i + 1]),
                        step_scale=float(step_scales[i]),
                        noise_scale=float(noise_scales[i]),
                    )
                )
                i += 1
            else:
                # Heating: COOL-HEAT cycles
                jump = heat_config.jump
                iterations = heat_config.iterations
                height = heat_config.height

                # Determine segment boundaries
                segment_start = i
                segment_end = min(i + jump, n_steps)

                # Compute heat target (where to heat back to)
                # height=1.0 means heat back to segment_start
                # height<1.0 means heat back to a point between start and end
                heat_steps_back = int(jump * height)
                heat_target_idx = max(segment_end - heat_steps_back, segment_start)

                # First iteration (iteration=0)
                for j in range(segment_start, segment_end):
                    should_sync = self.sync.should_sync(j, n_steps, iteration=0)
                    steps.append(
                        SamplingStep(
                            sigma_from=float(sigmas[j]),
                            sigma_to=float(sigmas[j + 1]),
                            step_type=StepType.COOL,
                            step_idx=j,
                            iteration=0,
                            is_sync_step=should_sync,
                            gamma=float(gammas[j + 1]),
                            step_scale=float(step_scales[j]),
                            noise_scale=float(noise_scales[j]),
                        )
                    )

                # Subsequent iterations: HEAT + COOL
                for iter_idx in range(1, iterations):
                    # HEAT step: go from segment_end back to heat_target
                    steps.append(
                        SamplingStep(
                            sigma_from=float(sigmas[segment_end]),
                            sigma_to=float(sigmas[heat_target_idx]),
                            step_type=StepType.HEAT,
                            step_idx=heat_target_idx,
                            iteration=iter_idx,
                            is_sync_step=False,  # Never sync on HEAT
                            gamma=0.0,
                            step_scale=0.0,
                            noise_scale=1.0,
                        )
                    )

                    # COOL steps from heat_target to segment_end
                    for j in range(heat_target_idx, segment_end):
                        should_sync = self.sync.should_sync(j, n_steps, iteration=iter_idx)
                        steps.append(
                            SamplingStep(
                                sigma_from=float(sigmas[j]),
                                sigma_to=float(sigmas[j + 1]),
                                step_type=StepType.COOL,
                                step_idx=j,
                                iteration=iter_idx,
                                is_sync_step=should_sync,
                                gamma=float(gammas[j + 1]),
                                step_scale=float(step_scales[j]),
                                noise_scale=float(noise_scales[j]),
                            )
                        )

                i = segment_end

        return steps

    def __repr__(self) -> str:
        return f"ACGScheduler(heating={self.heating}, sync={self.sync})"


def compute_heat_noise_std(sigma_from: float, sigma_to: float) -> float:
    """Compute the noise standard deviation for forward diffusion (HEATING).

    When going from sigma_from to sigma_to (where sigma_to > sigma_from),
    we need to add noise with std = sqrt(sigma_to^2 - sigma_from^2).

    Parameters
    ----------
    sigma_from : float
        Current noise level (lower).
    sigma_to : float
        Target noise level (higher).

    Returns
    -------
    float
        Standard deviation of noise to add.

    Raises
    ------
    ValueError
        If sigma_to <= sigma_from.
    """
    if sigma_to <= sigma_from:
        raise ValueError(
            f"HEATING requires sigma_to > sigma_from, "
            f"got sigma_to={sigma_to}, sigma_from={sigma_from}"
        )
    return sqrt(sigma_to**2 - sigma_from**2)


# Convenience factory functions for common configurations


def create_greedy_scheduler(
    sync_start: Optional[int] = None,
    sync_end: Optional[int] = None,
) -> ACGScheduler:
    """Create a Greedy scheduler (no heating, sync every step).

    This configuration enforces consensus at every step without
    heating. May accumulate artifacts due to repeated consensus.

    Parameters
    ----------
    sync_start : int, optional
        First step to sync.
    sync_end : int, optional
        Last step to sync.
    """
    from acg.schedulers.sync import RangeSync

    return ACGScheduler(
        heating_schedule=NoHeating(),
        sync_schedule=RangeSync(
            start_step=sync_start,
            end_step=sync_end,
            first_visit_only=False,  # Sync on every step
        ),
    )


def create_consistent_scheduler(
    heat_jump: int = 2,
    num_iterations: int = 3,
    heat_height: float = 1.0,
    heat_start: Optional[int] = None,
    heat_end: Optional[int] = None,
    sync_start: Optional[int] = None,
    sync_end: Optional[int] = None,
) -> ACGScheduler:
    """Create a Consistent scheduler (heating + sync every pass).

    This configuration uses heating and enforces consensus at every
    step including repeated passes after heating. May accumulate
    artifacts due to repeated consensus.

    Parameters
    ----------
    heat_jump : int
        Number of steps to jump back.
    num_iterations : int
        Number of resampling iterations.
    heat_height : float
        Heat height H in (0, 1].
    heat_start : int, optional
        First step to apply heating.
    heat_end : int, optional
        Last step to apply heating.
    sync_start : int, optional
        First step to sync.
    sync_end : int, optional
        Last step to sync.
    """
    from acg.schedulers.heating import UniformHeating
    from acg.schedulers.sync import RangeSync

    return ACGScheduler(
        heating_schedule=UniformHeating(
            heat_jump=heat_jump,
            num_iterations=num_iterations,
            heat_height=heat_height,
            start_step=heat_start,
            end_step=heat_end,
        ),
        sync_schedule=RangeSync(
            start_step=sync_start,
            end_step=sync_end,
            first_visit_only=False,  # Sync on every pass (including after heating)
        ),
    )


def create_acg_scheduler(
    heat_jump: int = 2,
    num_iterations: int = 3,
    heat_height: float = 1.0,
    heat_start: Optional[int] = None,
    heat_end: Optional[int] = None,
    sync_start: Optional[int] = None,
    sync_end: Optional[int] = None,
    first_visit_only: bool = True,
) -> ACGScheduler:
    """Create an ACG scheduler with configurable heating and sync.

    This is the full ACG configuration with both heating and
    first-visit-only synchronization.

    Parameters
    ----------
    heat_jump : int
        Number of steps to jump back.
    num_iterations : int
        Number of resampling iterations.
    heat_height : float
        Heat height H in (0, 1].
    heat_start : int, optional
        First step to apply heating.
    heat_end : int, optional
        Last step to apply heating.
    sync_start : int, optional
        First step to sync.
    sync_end : int, optional
        Last step to sync.
    first_visit_only : bool
        Whether to sync only on first visit.
    """
    from acg.schedulers.heating import UniformHeating
    from acg.schedulers.sync import RangeSync

    return ACGScheduler(
        heating_schedule=UniformHeating(
            heat_jump=heat_jump,
            num_iterations=num_iterations,
            heat_height=heat_height,
            start_step=heat_start,
            end_step=heat_end,
        ),
        sync_schedule=RangeSync(
            start_step=sync_start,
            end_step=sync_end,
            first_visit_only=first_visit_only,
        ),
    )
