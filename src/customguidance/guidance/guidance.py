from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass
import torch

from .utils import MomentumBuffer, GuidanceTermBuffer
from .utils import constant_guidance, linear_guidance, exponential_guidance, adaptative_projected_guidance, zero_star_guidance, sliding_mode_control_guidance
from ..error import check_existing_guidance_method, check_guidance_parameters

class GuidanceMethod(ABC):
    """
    Base class for all guidance methods.
    Note: Abstract class.
    """

    def __init__(self):
        pass
    
    def reset(self):
        pass

    @abstractmethod
    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        """
        Compute the guided model prediction for one denoising step.
        """
        raise NotImplementedError


class ConstantGuidanceMethod(GuidanceMethod):

    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, do_cfg=True)
        
        return constant_guidance(pred_uncond, pred_cond, ctx.guidance_scale)


class LinearGuidanceMethod(GuidanceMethod):

    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, do_cfg=True)
        
        return linear_guidance(pred_uncond, pred_cond, ctx.guidance_scale, ctx.normalized_time())


class ExponentialGuidanceMethod(GuidanceMethod):

    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, do_cfg=True)
        
        return exponential_guidance(pred_uncond, pred_cond, ctx.guidance_scale, ctx.normalized_time())


class APGGuidanceMethod(GuidanceMethod):

    def __init__(self, momentum_value: float = 0.9, eta: float = 1.0, norm_threshold: float = 15.0):
        super().__init__()
        self.momentum_value = momentum_value
        self.momentum_buffer = MomentumBuffer(momentum_value)
        self.eta = eta
        self.norm_threshold = norm_threshold
        
    def reset(self):
        self.momentum_buffer = MomentumBuffer(self.momentum_value)

    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, do_cfg=True)
        
        apg_state = {"momentum_buffer": self.momentum_buffer, "eta": self.eta, "norm_threshold": self.norm_threshold}

        return adaptative_projected_guidance(pred_uncond, pred_cond, ctx.guidance_scale, ctx.normalized_time(), ctx.latents, apg_state)


class RectifiedPPGuidanceMethod(GuidanceMethod):

    def __init__(self, lambda_max: float, gamma: float):
        super().__init__()
        self.lambda_max = lambda_max
        self.gamma = gamma

    def _compute_dt(self, ctx: CFGContext) -> torch.Tensor:
        if ctx.step_index < len(ctx.timesteps) - 1:
            return ctx.timesteps[ctx.step_index] - ctx.timesteps[ctx.step_index + 1]
        return ctx.timesteps[ctx.step_index]

    def _compute_alpha_t(self, ctx: CFGContext) -> torch.Tensor:
        return self.lambda_max*(1-ctx.normalized_time())**self.gamma

    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        v_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, do_cfg=False, use_original = True)
        dt = self._compute_dt(ctx)
        x_mid = ctx.latents + 0.5 * dt * v_cond
        t_mid = ctx.t - 0.5 * dt

        v_uncond_mid, v_cond_mid = ctx.pipeline._predict_model(latents=x_mid, t=t_mid, do_cfg=True)
        alpha_t = self._compute_alpha_t(ctx)
        return v_cond + alpha_t * (v_cond_mid - v_uncond_mid)


class ZeroStarGuidanceMethod(GuidanceMethod):

    def __init__(self, zero_steps: int = 0, use_zero_init: bool = False):
        super().__init__()
        self.zero_steps = zero_steps
        self.use_zero_init = use_zero_init

    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, do_cfg=True)
        
        return zero_star_guidance(pred_uncond, pred_cond, ctx.guidance_scale, 
                                  self.zero_steps, self.use_zero_init, ctx.step_index)
    
class SlidingModeControlGuidanceMethod(GuidanceMethod):
    
    def __init__(self, lambda_param, k):
        super().__init__()
        self.lambda_param = lambda_param
        self.k = k
        self.guidance_term_buffer = GuidanceTermBuffer()
    
    def reset(self):
        self.guidance_term_buffer = GuidanceTermBuffer()
    
    def predict_velocity_field(self, ctx: CFGContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, do_cfg=True)
                                                        
        return sliding_mode_control_guidance(pred_uncond, pred_cond, ctx.guidance_scale, 
                                             self.lambda_param, self.k, self.guidance_term_buffer)

def build_guidance_method(guidance_type: str, params: Optional[dict[str, Any]] = None) -> GuidanceMethod:
    
    check_existing_guidance_method(guidance_type)
    check_guidance_parameters(guidance_type, params)

    if guidance_type == "constant":
        return ConstantGuidanceMethod()

    if guidance_type == "linear":
        return LinearGuidanceMethod()

    if guidance_type == "exponential":
        return ExponentialGuidanceMethod()

    if guidance_type == "APG":
        return APGGuidanceMethod(**params)

    if guidance_type == "rectified_pp":
        return RectifiedPPGuidanceMethod(**params)

    if guidance_type == "zero_star":
        return ZeroStarGuidanceMethod(**params)
    
    if guidance_type == "SMC":
        return SlidingModeControlGuidanceMethod(**params)

@dataclass
class CFGContext:
    pipeline: Any
    latents: torch.Tensor
    t: torch.Tensor
    step_index: int
    timesteps: torch.Tensor
    guidance_scale: float
    timestep: torch.Tensor

    def normalized_time(self) -> torch.Tensor:
        """
        Return normalized time t / t0.
        """
        return self.t / self.timesteps[0]