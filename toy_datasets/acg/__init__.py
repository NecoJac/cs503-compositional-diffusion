"""ACG: Annealed Co-Generation Framework.

A modular framework for EDM-based diffusion models, focusing on:
- Conditional generation with Classifier-Free Guidance
- Multi-condition composition (Composable Diffusion)
- Annealed Co-Generation with independent heating/sync schedules
- Exact (non-neural network) denoisers for theoretical understanding

Main Components
---------------
core : Noise schedules, preconditioning, losses
samplers : Euler, Heun, DPM-Solver, Multi-Attribute
schedulers : Heating schedules, Sync schedules, ACG scheduler
denoisers : Exact, Conditional, Multi-condition
modules : Neural network building blocks
datasets : Toy datasets for testing

Quick Start
-----------
>>> from acg import get_sigmas_karras, sample_heun, ExactDenoiser
>>> from acg.datasets import get_dataset
>>>
>>> # Get data
>>> data, labels = get_dataset("gaussian", n_samples_per_class=50)
>>>
>>> # Create denoiser
>>> denoiser = ExactDenoiser(data)
>>>
>>> # Sample
>>> sigmas = get_sigmas_karras(50, device=data.device)
>>> latents = torch.randn(16, 2) * sigmas[0]
>>> samples = sample_heun(denoiser, latents, sigmas)
"""

__version__ = "0.2.0"

# Core components
from acg.core import (
    EDMLoss,
    EDMLossConditional,
    EDMPrecond,
    EDMPrecondConditional,
    get_sigmas_exponential,
    get_sigmas_karras,
    get_sigmas_linear,
)

# Samplers
from acg.samplers import (
    DPMSolver2Sampler,
    EulerAncestralSampler,
    EulerSampler,
    HeunSampler,
    Sampler,
    SamplerOutput,
    sample_dpm_2,
    sample_euler,
    sample_euler_ancestral,
    sample_heun,
)

# Denoisers
from acg.denoisers import (
    ConditionalExactDenoiser,
    Denoiser,
    ExactDenoiser,
    MultiAttributeDenoiser,
    MultiConditionDenoiser,
)

# Utils
from acg.utils import append_dims

# Schedulers (ACG-specific)
from acg.schedulers import (
    ACGScheduler,
    create_acg_scheduler,
    create_consistent_scheduler,
    create_greedy_scheduler,
)

__all__ = [
    # Version
    "__version__",
    # Core - Schedules
    "get_sigmas_exponential",
    "get_sigmas_karras",
    "get_sigmas_linear",
    # Core - Preconditioning
    "EDMPrecond",
    "EDMPrecondConditional",
    # Core - Losses
    "EDMLoss",
    "EDMLossConditional",
    # Samplers - Base
    "Sampler",
    "SamplerOutput",
    # Samplers - Implementations
    "EulerSampler",
    "EulerAncestralSampler",
    "HeunSampler",
    "DPMSolver2Sampler",
    # Samplers - Functions
    "sample_euler",
    "sample_euler_ancestral",
    "sample_heun",
    "sample_dpm_2",
    # Denoisers
    "Denoiser",
    "ExactDenoiser",
    "ConditionalExactDenoiser",
    "MultiConditionDenoiser",
    "MultiAttributeDenoiser",
    # Utils
    "append_dims",
    # Schedulers
    "ACGScheduler",
    "create_acg_scheduler",
    "create_consistent_scheduler",
    "create_greedy_scheduler",
]
