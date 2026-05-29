"""Sampling Schedulers for Diffusion Models.

This module provides scheduling algorithms for diffusion sampling,
including support for RePaint-style resampling (HEAT/COOL cycles)
and progressive retreat strategies.

Classes:
    BaseSamplingScheduler: Abstract base class for schedulers
    StandardDiffusionScheduler: Standard linear cooling schedule
    AnnealedScheduler: RePaint-style heating-cooling cycles
    ProgressiveRetreatScheduler: Global progressive retreat strategy

References:
    RePaint: Inpainting using Denoising Diffusion Probabilistic Models
    https://arxiv.org/abs/2201.09865
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import sqrt
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
    gamma : float
        Gamma parameter for noise injection (EDM stochastic sampling).
    step_scale : float
        Step scale factor.
    noise_scale : float
        Noise scale factor.
    step_idx : int
        Original step index (for tracking).
    is_sync_step : bool
        Whether to use synchronized consensus in this step.
    """

    sigma_from: float
    sigma_to: float
    step_type: StepType
    gamma: float
    step_scale: float
    noise_scale: float
    step_idx: int = 0
    is_sync_step: bool = True


class BaseSamplingScheduler(ABC):
    """Abstract base class for sampling schedulers.

    A scheduler converts raw sigma sequences into a list of SamplingSteps
    that may include both COOL and HEAT operations.
    """

    @abstractmethod
    def get_schedule(
        self,
        sigmas: ArrayLike,
        gammas: Optional[ArrayLike] = None,
        step_scales: Optional[ArrayLike] = None,
        noise_scales: Optional[ArrayLike] = None,
    ) -> List[SamplingStep]:
        """Convert raw sigma sequence to a list of sampling steps.

        Parameters
        ----------
        sigmas : ArrayLike
            Noise levels [num_steps + 1], from high to low (ending at 0).
        gammas : ArrayLike, optional
            Gamma values [num_steps + 1]. Defaults to zeros.
        step_scales : ArrayLike, optional
            Step scale factors [num_steps]. Defaults to ones.
        noise_scales : ArrayLike, optional
            Noise scale factors [num_steps]. Defaults to ones.

        Returns
        -------
        List[SamplingStep]
            List of SamplingStep objects defining the complete schedule.
        """

    def _prepare_arrays(
        self,
        sigmas: ArrayLike,
        gammas: Optional[ArrayLike],
        step_scales: Optional[ArrayLike],
        noise_scales: Optional[ArrayLike],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and validate input arrays."""
        sigmas = np.asarray(sigmas)
        n = len(sigmas) - 1  # Number of steps

        if gammas is None:
            gammas = np.zeros(len(sigmas))
        else:
            gammas = np.asarray(gammas)

        if step_scales is None:
            step_scales = np.ones(n)
        else:
            step_scales = np.asarray(step_scales)

        if noise_scales is None:
            noise_scales = np.ones(n)
        else:
            noise_scales = np.asarray(noise_scales)

        return sigmas, gammas, step_scales, noise_scales


class StandardDiffusionScheduler(BaseSamplingScheduler):
    """Standard scheduler that produces a linear sequence of COOL steps.

    This scheduler maintains backward compatibility with existing behavior.
    Supports optional synchronization range control.

    Parameters
    ----------
    sync_start_step : int, optional
        Step index to start using synchronization (None = from beginning).
    sync_end_step : int, optional
        Step index to stop using synchronization (None = until end).
    """

    def __init__(
        self,
        sync_start_step: Optional[int] = None,
        sync_end_step: Optional[int] = None,
    ):
        self.sync_start_step = sync_start_step
        self.sync_end_step = sync_end_step

    def _is_in_sync_range(self, step_idx: int, n: int) -> bool:
        """Check if a step index is within the synchronization range."""
        start = self.sync_start_step if self.sync_start_step is not None else 0
        end = self.sync_end_step if self.sync_end_step is not None else n
        return start <= step_idx < end

    def get_schedule(
        self,
        sigmas: ArrayLike,
        gammas: Optional[ArrayLike] = None,
        step_scales: Optional[ArrayLike] = None,
        noise_scales: Optional[ArrayLike] = None,
    ) -> List[SamplingStep]:
        """Generate standard linear cooling schedule."""
        sigmas, gammas, step_scales, noise_scales = self._prepare_arrays(
            sigmas, gammas, step_scales, noise_scales
        )
        steps = []
        n = len(sigmas) - 1

        for i in range(n):
            steps.append(
                SamplingStep(
                    sigma_from=float(sigmas[i]),
                    sigma_to=float(sigmas[i + 1]),
                    step_type=StepType.COOL,
                    gamma=float(gammas[i + 1]),
                    step_scale=float(step_scales[i]),
                    noise_scale=float(noise_scales[i]),
                    step_idx=i,
                    is_sync_step=self._is_in_sync_range(i, n),
                )
            )

        return steps

    def __repr__(self) -> str:
        return (
            f"StandardDiffusionScheduler(sync_start_step={self.sync_start_step}, "
            f"sync_end_step={self.sync_end_step})"
        )


class AnnealedScheduler(BaseSamplingScheduler):
    """Annealed Heating-Cooling scheduler for RePaint-style resampling.

    The algorithm works by:
    1. Cooling for `heat_jump` steps
    2. Heating back to the start of the segment
    3. Repeating steps 1-2 for `num_iterations` times
    4. Moving to the next segment

    Example with heat_jump=2, num_iterations=3:
        σ₀ → σ₁ → σ₂  [denoise 2 steps]
        σ₂ ← σ₀       [renoise back]
        σ₀ → σ₁ → σ₂  [denoise again]
        σ₂ ← σ₀       [renoise back]
        σ₀ → σ₁ → σ₂  [denoise 3rd time]
        σ₂ → σ₃ → σ₄  [continue to next segment]

    Parameters
    ----------
    heat_jump : int
        Number of cooling steps before heating back.
    num_iterations : int
        Number of times to iterate each segment.
    start_step : int, optional
        Step index to start annealing (None = from beginning).
    end_step : int, optional
        Step index to end annealing (None = until end).
    sync_start_step : int, optional
        Step index to start using synchronization.
    sync_end_step : int, optional
        Step index to stop using synchronization.
    sync_resample_mode : str
        Control sync usage in iterations: "first_only", "all", or "none".
    heat_height : int, optional
        Number of steps to heat back (default: heat_jump).

    References
    ----------
    RePaint: Inpainting using Denoising Diffusion Probabilistic Models
    https://arxiv.org/abs/2201.09865
    """

    def __init__(
        self,
        heat_jump: int = 2,
        num_iterations: int = 3,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        sync_start_step: Optional[int] = None,
        sync_end_step: Optional[int] = None,
        sync_resample_mode: str = "first_only",
        heat_height: Optional[int] = None,
    ):
        if heat_jump < 1:
            raise ValueError(f"heat_jump must be >= 1, got {heat_jump}")
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if sync_resample_mode not in ["first_only", "all", "none"]:
            raise ValueError(
                f"sync_resample_mode must be 'first_only', 'all', or 'none', "
                f"got {sync_resample_mode}"
            )
        if heat_height is not None:
            if heat_height < 1:
                raise ValueError(f"heat_height must be >= 1, got {heat_height}")
            if heat_height > heat_jump:
                raise ValueError(
                    f"heat_height must be <= heat_jump, "
                    f"got heat_height={heat_height}, heat_jump={heat_jump}"
                )

        self.heat_jump = heat_jump
        self.num_iterations = num_iterations
        self.start_step = start_step
        self.end_step = end_step
        self.sync_start_step = sync_start_step
        self.sync_end_step = sync_end_step
        self.sync_resample_mode = sync_resample_mode
        self.heat_height = heat_height if heat_height is not None else heat_jump

    def _is_in_sync_range(self, step_idx: int, n: int) -> bool:
        """Check if a step index is within the synchronization range."""
        start = self.sync_start_step if self.sync_start_step is not None else 0
        end = self.sync_end_step if self.sync_end_step is not None else n
        return start <= step_idx < end

    def get_schedule(
        self,
        sigmas: ArrayLike,
        gammas: Optional[ArrayLike] = None,
        step_scales: Optional[ArrayLike] = None,
        noise_scales: Optional[ArrayLike] = None,
    ) -> List[SamplingStep]:
        """Generate annealed schedule with COOL and HEAT steps."""
        sigmas, gammas, step_scales, noise_scales = self._prepare_arrays(
            sigmas, gammas, step_scales, noise_scales
        )
        steps = []
        n = len(sigmas) - 1

        # Determine annealing range
        start = self.start_step if self.start_step is not None else 0
        end = self.end_step if self.end_step is not None else n
        start = max(0, min(start, n))
        end = max(start, min(end, n))

        # Before annealing range: standard cooling
        for i in range(start):
            steps.append(
                SamplingStep(
                    sigma_from=float(sigmas[i]),
                    sigma_to=float(sigmas[i + 1]),
                    step_type=StepType.COOL,
                    gamma=float(gammas[i + 1]),
                    step_scale=float(step_scales[i]),
                    noise_scale=float(noise_scales[i]),
                    step_idx=i,
                    is_sync_step=self._is_in_sync_range(i, n),
                )
            )

        # Annealing range: segments with iterations
        i = start
        while i < end:
            segment_end = min(i + self.heat_jump, end)
            heat_target_idx = max(segment_end - self.heat_height, i)

            # First iteration
            for j in range(i, segment_end):
                in_range = self._is_in_sync_range(j, n)
                if self.sync_resample_mode == "first_only":
                    use_sync = in_range
                elif self.sync_resample_mode == "all":
                    use_sync = in_range
                else:
                    use_sync = False

                steps.append(
                    SamplingStep(
                        sigma_from=float(sigmas[j]),
                        sigma_to=float(sigmas[j + 1]),
                        step_type=StepType.COOL,
                        gamma=float(gammas[j + 1]),
                        step_scale=float(step_scales[j]),
                        noise_scale=float(noise_scales[j]),
                        step_idx=j,
                        is_sync_step=use_sync,
                    )
                )

            # Subsequent iterations: heat back and cool
            for iter_idx in range(1, self.num_iterations):
                # HEAT step
                steps.append(
                    SamplingStep(
                        sigma_from=float(sigmas[segment_end]),
                        sigma_to=float(sigmas[heat_target_idx]),
                        step_type=StepType.HEAT,
                        gamma=0.0,
                        step_scale=0.0,
                        noise_scale=1.0,
                        step_idx=heat_target_idx,
                    )
                )

                # Cool from heat_target to segment_end
                for j in range(heat_target_idx, segment_end):
                    in_range = self._is_in_sync_range(j, n)
                    if self.sync_resample_mode == "first_only":
                        use_sync = False
                    elif self.sync_resample_mode == "all":
                        use_sync = in_range
                    else:
                        use_sync = False

                    steps.append(
                        SamplingStep(
                            sigma_from=float(sigmas[j]),
                            sigma_to=float(sigmas[j + 1]),
                            step_type=StepType.COOL,
                            gamma=float(gammas[j + 1]),
                            step_scale=float(step_scales[j]),
                            noise_scale=float(noise_scales[j]),
                            step_idx=j,
                            is_sync_step=use_sync,
                        )
                    )

            i = segment_end

        # After annealing range: standard cooling
        for i in range(end, n):
            steps.append(
                SamplingStep(
                    sigma_from=float(sigmas[i]),
                    sigma_to=float(sigmas[i + 1]),
                    step_type=StepType.COOL,
                    gamma=float(gammas[i + 1]),
                    step_scale=float(step_scales[i]),
                    noise_scale=float(noise_scales[i]),
                    step_idx=i,
                    is_sync_step=self._is_in_sync_range(i, n),
                )
            )

        return steps

    def __repr__(self) -> str:
        return (
            f"AnnealedScheduler(heat_jump={self.heat_jump}, "
            f"num_iterations={self.num_iterations}, "
            f"heat_height={self.heat_height}, "
            f"start_step={self.start_step}, end_step={self.end_step}, "
            f"sync_resample_mode='{self.sync_resample_mode}')"
        )


class ProgressiveRetreatScheduler(BaseSamplingScheduler):
    """Progressive Retreat scheduler with global decreasing retreat strategy.

    Unlike AnnealedScheduler which divides the process into fixed segments,
    this scheduler performs a full cooling pass first, then progressively
    retreats with decreasing distances.

    Example with binary mode (3 retreats):
        Pass 1: σ_0 ──────────────────────────────> σ_T  (full cooling)
        Heat 1:                    σ_{T/2} <─────── σ_T  (retreat 50%)
        Pass 2:                    σ_{T/2} ────────> σ_T  (sync enabled)
        Heat 2:                           σ_{3T/4} <── σ_T  (retreat 25%)
        Pass 3:                           σ_{3T/4} ──> σ_T  (sync enabled)
        Done

    Parameters
    ----------
    retreat_mode : str
        Strategy for computing retreat points: "binary", "linear", or "custom".
    num_retreats : int
        Number of retreat operations (for binary/linear modes).
    retreat_ratios : List[float], optional
        Custom retreat points as ratios in (0, 1), sorted ascending.
    start_step : int, optional
        Step index to start the schedule.
    end_step : int, optional
        Step index to end the schedule.
    sync_start_step : int, optional
        Step index to start using synchronization.
    sync_end_step : int, optional
        Step index to stop using synchronization.
    sync_on_first_pass : bool
        Whether to use sync on the first full cooling pass.
    sync_phase_mode : str
        Control sync usage: "default", "all", "none", "even", or "odd".
    """

    def __init__(
        self,
        retreat_mode: str = "binary",
        num_retreats: int = 3,
        retreat_ratios: Optional[List[float]] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        sync_start_step: Optional[int] = None,
        sync_end_step: Optional[int] = None,
        sync_on_first_pass: bool = False,
        sync_phase_mode: str = "default",
    ):
        if retreat_mode not in ["binary", "linear", "custom"]:
            raise ValueError(
                f"retreat_mode must be 'binary', 'linear', or 'custom', "
                f"got {retreat_mode}"
            )
        if num_retreats < 1:
            raise ValueError(f"num_retreats must be >= 1, got {num_retreats}")

        if retreat_mode == "custom":
            if retreat_ratios is None or len(retreat_ratios) == 0:
                raise ValueError("retreat_ratios must be provided for custom mode")
            for r in retreat_ratios:
                if not (0 < r < 1):
                    raise ValueError(f"retreat_ratios must be in (0, 1), got {r}")
            if retreat_ratios != sorted(retreat_ratios):
                raise ValueError("retreat_ratios must be sorted in ascending order")

        if sync_phase_mode not in ["default", "all", "none", "even", "odd"]:
            raise ValueError(
                f"sync_phase_mode must be 'default', 'all', 'none', 'even', or 'odd', "
                f"got {sync_phase_mode}"
            )

        self.retreat_mode = retreat_mode
        self.num_retreats = num_retreats
        self.retreat_ratios = retreat_ratios
        self.start_step = start_step
        self.end_step = end_step
        self.sync_start_step = sync_start_step
        self.sync_end_step = sync_end_step
        self.sync_on_first_pass = sync_on_first_pass
        self.sync_phase_mode = sync_phase_mode

    def _compute_retreat_ratios(self) -> List[float]:
        """Compute retreat ratios based on mode."""
        if self.retreat_mode == "custom":
            return list(self.retreat_ratios)  # type: ignore
        elif self.retreat_mode == "binary":
            return [1 - 1 / (2**i) for i in range(1, self.num_retreats + 1)]
        else:  # linear
            n = self.num_retreats
            return [(i + 1) / (n + 1) for i in range(n)]

    def _is_in_sync_range(self, step_idx: int, n: int) -> bool:
        """Check if a step index is within the synchronization range."""
        start = self.sync_start_step if self.sync_start_step is not None else 0
        end = self.sync_end_step if self.sync_end_step is not None else n
        return start <= step_idx < end

    def _should_sync_phase(self, phase_idx: int) -> bool:
        """Determine if a given phase should use synchronization."""
        if self.sync_phase_mode == "all":
            return True
        elif self.sync_phase_mode == "none":
            return False
        elif self.sync_phase_mode == "even":
            return phase_idx % 2 == 0
        elif self.sync_phase_mode == "odd":
            return phase_idx % 2 == 1
        else:  # "default"
            if phase_idx == 0:
                return self.sync_on_first_pass
            else:
                return True

    def get_schedule(
        self,
        sigmas: ArrayLike,
        gammas: Optional[ArrayLike] = None,
        step_scales: Optional[ArrayLike] = None,
        noise_scales: Optional[ArrayLike] = None,
    ) -> List[SamplingStep]:
        """Generate Progressive Retreat schedule."""
        sigmas, gammas, step_scales, noise_scales = self._prepare_arrays(
            sigmas, gammas, step_scales, noise_scales
        )
        steps = []
        n = len(sigmas) - 1

        # Determine range
        start = self.start_step if self.start_step is not None else 0
        end = self.end_step if self.end_step is not None else n
        start = max(0, min(start, n))
        end = max(start, min(end, n))

        # Compute retreat points
        retreat_ratios = self._compute_retreat_ratios()
        range_length = end - start
        retreat_indices = [start + int(ratio * range_length) for ratio in retreat_ratios]

        # Phase 0: First pass, full cooling from start to end
        phase_idx = 0
        phase_should_sync = self._should_sync_phase(phase_idx)
        for i in range(start, end):
            in_range = self._is_in_sync_range(i, n)
            use_sync = in_range and phase_should_sync

            steps.append(
                SamplingStep(
                    sigma_from=float(sigmas[i]),
                    sigma_to=float(sigmas[i + 1]),
                    step_type=StepType.COOL,
                    gamma=float(gammas[i + 1]),
                    step_scale=float(step_scales[i]),
                    noise_scale=float(noise_scales[i]),
                    step_idx=i,
                    is_sync_step=use_sync,
                )
            )

        # Retreat passes: heat back then cool
        for retreat_num, retreat_idx in enumerate(retreat_indices):
            phase_idx = retreat_num + 1
            phase_should_sync = self._should_sync_phase(phase_idx)

            # Heat step
            steps.append(
                SamplingStep(
                    sigma_from=float(sigmas[end]),
                    sigma_to=float(sigmas[retreat_idx]),
                    step_type=StepType.HEAT,
                    gamma=0.0,
                    step_scale=0.0,
                    noise_scale=1.0,
                    step_idx=retreat_idx,
                )
            )

            # Cool from retreat point to end
            for i in range(retreat_idx, end):
                in_range = self._is_in_sync_range(i, n)
                use_sync = in_range and phase_should_sync

                steps.append(
                    SamplingStep(
                        sigma_from=float(sigmas[i]),
                        sigma_to=float(sigmas[i + 1]),
                        step_type=StepType.COOL,
                        gamma=float(gammas[i + 1]),
                        step_scale=float(step_scales[i]),
                        noise_scale=float(noise_scales[i]),
                        step_idx=i,
                        is_sync_step=use_sync,
                    )
                )

        # After range: standard cooling
        for i in range(end, n):
            steps.append(
                SamplingStep(
                    sigma_from=float(sigmas[i]),
                    sigma_to=float(sigmas[i + 1]),
                    step_type=StepType.COOL,
                    gamma=float(gammas[i + 1]),
                    step_scale=float(step_scales[i]),
                    noise_scale=float(noise_scales[i]),
                    step_idx=i,
                    is_sync_step=self._is_in_sync_range(i, n),
                )
            )

        return steps

    def __repr__(self) -> str:
        return (
            f"ProgressiveRetreatScheduler(retreat_mode='{self.retreat_mode}', "
            f"num_retreats={self.num_retreats}, "
            f"sync_on_first_pass={self.sync_on_first_pass}, "
            f"sync_phase_mode='{self.sync_phase_mode}')"
        )


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
