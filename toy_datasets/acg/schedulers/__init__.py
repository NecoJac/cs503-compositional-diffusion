"""ACG Schedulers Module.

This module provides scheduling algorithms for ACG (Annealed Co-Generation),
with independent control over heating and synchronization.

Architecture
------------
The scheduling system has three independent components:

1. **Noise Schedule** (from acg.core)
   - get_sigmas_karras, get_sigmas_linear, get_sigmas_exponential
   - Defines the σ sequence from high noise to low noise

2. **Heating Schedule** (this module)
   - Controls when/how to inject noise (heat)
   - NoHeating, UniformHeating, DecayingHeating

3. **Sync Schedule** (this module)
   - Controls when to apply Consensus
   - NoSync, AlwaysSync, RangeSync, StrideSync

These components are combined by ACGScheduler to produce a complete
sampling schedule with COOL/HEAT steps and sync flags.

Quick Start
-----------
>>> from acg import get_sigmas_karras
>>> from acg.schedulers import (
...     ACGScheduler, UniformHeating, RangeSync,
...     create_greedy_scheduler, create_acg_scheduler,
... )
>>>
>>> # Option 1: Use factory function
>>> scheduler = create_acg_scheduler(heat_jump=2, num_iterations=3)
>>>
>>> # Option 2: Compose manually
>>> scheduler = ACGScheduler(
...     heating_schedule=UniformHeating(heat_jump=2, num_iterations=3),
...     sync_schedule=RangeSync(first_visit_only=True),
... )
>>>
>>> # Generate schedule from sigmas
>>> sigmas = get_sigmas_karras(50)
>>> steps = scheduler.get_schedule(sigmas.numpy())

Paper Configurations
--------------------
- Greedy: create_greedy_scheduler()
- Consistent: create_consistent_scheduler()
- ACG: create_acg_scheduler(heat_jump=J, num_iterations=K)
"""

# Base types
from acg.schedulers.base import (
    ArrayLike,
    HeatConfig,
    SamplingStep,
    StepType,
)

# Heating schedules
from acg.schedulers.heating import (
    BaseHeatingSchedule,
    DecayingHeating,
    NoHeating,
    UniformHeating,
)

# Sync schedules
from acg.schedulers.sync import (
    AlwaysSync,
    BaseSyncSchedule,
    NoSync,
    RangeSync,
    SigmaRangeSync,
    StrideSync,
)

# Main scheduler
from acg.schedulers.acg_scheduler import (
    ACGScheduler,
    compute_heat_noise_std,
    create_acg_scheduler,
    create_consistent_scheduler,
    create_greedy_scheduler,
)

__all__ = [
    # Base types
    "ArrayLike",
    "HeatConfig",
    "SamplingStep",
    "StepType",
    # Heating schedules
    "BaseHeatingSchedule",
    "NoHeating",
    "UniformHeating",
    "DecayingHeating",
    # Sync schedules
    "BaseSyncSchedule",
    "NoSync",
    "AlwaysSync",
    "RangeSync",
    "StrideSync",
    "SigmaRangeSync",
    # Main scheduler
    "ACGScheduler",
    "compute_heat_noise_std",
    # Factory functions
    "create_greedy_scheduler",
    "create_consistent_scheduler",
    "create_acg_scheduler",
]
