"""Diffusion Samplers.

This module provides sampling algorithms for diffusion models.

Available samplers:
- Euler (first-order)
- Heun (second-order, recommended)
- DPM-Solver-2 (second-order)
- Euler Ancestral (stochastic)
- Multi-Attribute (Composable Diffusion)
- ACGSampler (Annealed Co-Generation with independent heating/sync)
- ScheduledSampler (legacy scheduler-driven sampling)

All samplers follow a unified interface and can be used with both
neural network denoisers and exact denoisers.
"""

from acg.samplers.base import Sampler, SamplerOutput
from acg.samplers.dpm import DPMSolver2Sampler, sample_dpm_2
from acg.samplers.euler import EulerAncestralSampler, EulerSampler, sample_euler, sample_euler_ancestral
from acg.samplers.heun import HeunSampler, sample_heun
from acg.samplers.multi_attr import (
    EulerMultiAttrSampler,
    HeunMultiAttrSampler,
    sample_euler_multi_attr,
    sample_heun_multi_attr,
)
from acg.samplers.acg_sampler import ACGSampler
from acg.samplers.scheduled import ScheduledSampler
from acg.samplers.scheduler import (
    AnnealedScheduler,
    BaseSamplingScheduler,
    ProgressiveRetreatScheduler,
    SamplingStep,
    StandardDiffusionScheduler,
    StepType,
    compute_heat_noise_std,
)

__all__ = [
    # Base
    "Sampler",
    "SamplerOutput",
    # Euler
    "EulerSampler",
    "EulerAncestralSampler",
    "sample_euler",
    "sample_euler_ancestral",
    # Heun
    "HeunSampler",
    "sample_heun",
    # DPM
    "DPMSolver2Sampler",
    "sample_dpm_2",
    # Multi-Attribute (Composable Diffusion)
    "HeunMultiAttrSampler",
    "EulerMultiAttrSampler",
    "sample_heun_multi_attr",
    "sample_euler_multi_attr",
    # Scheduled / ACG
    "ScheduledSampler",
    "ACGSampler",
    # Schedulers
    "BaseSamplingScheduler",
    "StandardDiffusionScheduler",
    "AnnealedScheduler",
    "ProgressiveRetreatScheduler",
    "SamplingStep",
    "StepType",
    "compute_heat_noise_std",
]

