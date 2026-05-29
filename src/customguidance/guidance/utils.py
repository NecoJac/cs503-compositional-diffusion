import numpy as np
import torch

def _compute_projection(v0, v1):
    """
    Computes the projection fraction of v0 onto v1.
    
    proj = (v0 · v1) / ||v1||²

    Args:
        v0: The vector to project, shape (B, C, H, W)
        v1: The vector to project onto, shape (B, C, H, W)

    Returns:
        proj: scalar projection coefficient, shape (B, 1, 1, 1)
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1_norm_sq = (v1 * v1).sum(dim=[-1,-2,-3], keepdim=True)
    proj = (v0 * v1).sum(dim=[-1,-2,-3], keepdim=True) / (v1_norm_sq + 1e-8)
    return proj.to(dtype)


def _decompose_parallel_ortho_component(v0, v1):
    """
    Decomposes v0 into its parallel and orthogonal components with respect to v1.

    Args:
        v0: The vector to decompose, shape (B, C, H, W)
        v1: The reference vector, shape (B, C, H, W)

    Returns:
        v0_parallel: The component of v0 parallel to v1, shape (B, C, H, W)
        v0_orthogonal: The component of v0 orthogonal to v1, shape (B, C, H, W)
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    proj = _compute_projection(v0, v1)
    v0_parallel = proj * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def _to_denoise(v_t, x_t, t):
    """
    Estimate denoised image from current noisy latents and velocity prediction.
    Flow matching : x0 = x_t - t * v_t
    
    Args:
        v_t : velocity prediction (noise_pred), shape (B, C, H, W)
        x_t : current noisy latents, shape (B, C, H, W)
        t   : current timestep scalar ∈ [0, 1]
    
    Returns:
        x0 : denoised estimate, shape (B, C, H, W)
    """
    return x_t - t * v_t


def _to_noise(x0, x_t, t):
    """
    Estimate noise from current noisy latents and denoised estimate.
    Flow matching : v_t = (x_t - x0) / t
    
    Args:
        x0  : denoised estimate, shape (B, C, H, W)
        x_t : current noisy latents, shape (B, C, H, W)
        t   : current timestep scalar ∈ [0, 1]

    Returns:
        v_t : velocity prediction (noise_pred), shape (B, C, H, W)
    """
    if t < 1e-6:
        return x0
    return (x_t - x0) / t


class MomentumBuffer:
    """
    Class used for the Adaptative Projected Guidance method. It buffers the average of previous velocities. 

    Attributes:
        momentum (float): Decay factor for the running average. Higher values
            give more weight to past values. Should be in [0, 1).
        running_avg (torch.Tensor | int): Current running average, initialized
            to 0 and updated at each denoising step
    """
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_avg = 0
    def update(self, new_value):
        new_avg = self.momentum * self.running_avg
        self.running_avg = new_value + new_avg

class GuidanceTermBuffer:
    """
    Class used for Sliding Mode Control (SMC-CFG). 
    It buffers the guidance term (v_cond - v_uncond).
    """
    def __init__(self):
        self.previous_e = None
    def update(self, current_e):
        self.previous_e = current_e.detach().clone()


def constant_guidance(noise_pred_uncond, noise_pred_text, guidance_scale):
    """
    Applies constant guidance to the noise prediction.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: set so that integral from T to 0 of omega(t) = guidance_scale * T.
        time: current time step, normalized in [0, 1]; noisy = 1, denoised = 0.
    
    Returns:
        The guided noise prediction.
    """
    omega = guidance_scale
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega

def linear_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time):
    """ 
    Applies increasing linear guidance to the noise prediction.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: set so that integral from T to 0 of omega(t) = guidance_scale * T.
        time: current time step, normalized in [0, 1]; noisy = 1, denoised = 0.
    
    Returns:
        The guided noise prediction.
    """
    # Calculate the linear scaling factor based on the current step
    omega = 2 * (1-time.item()) * guidance_scale
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega

def exponential_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time):
    """ 
    Applies increasing exponential guidance to the noise prediction.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: set so that integral from T to 0 of omega(t) = guidance_scale * T.
        time: current time step, normalized in [0, 1]; noisy = 1, denoised = 0.
    
    Returns:
        The guided noise prediction.
    """
    # Calculate the exponential scaling factor based on the current step
    alpha = (guidance_scale / (np.exp(1) - 1))
    omega = alpha * (np.exp(1-time.item()))
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega


def adaptative_projected_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time, latents, APG_parameters):
    """
    Implements the Adaptative Projected Guidance (APG) method for guiding the noise prediction in a diffusion model.

    Args:
        noise_pred_uncond: The noise prediction for the unconditional input, shape (B, C, H, W).
        noise_pred_text: The noise prediction for the text input, shape (B, C, H, W).
        guidance_scale: The scale of the guidance to apply.
        time: The current time step in the diffusion process.
        APG_parameters: A dictionary containing the parameters for the APG method, including:
            - momentum_buffer: An instance of MomentumBuffer to store the momentum of the updates.
            - eta: The scaling factor for the parallel component of the update.
            - norm_threshold: The maximum allowed norm for the update.
        latents: The current noisy latents at time t, shape (B, C, H, W).
    
    Returns:
        The guided noise prediction, shape (B, C, H, W).
    """
    x0_uncond = _to_denoise(noise_pred_uncond, latents, time)
    x0_text = _to_denoise(noise_pred_text, latents, time)

    diff = x0_text - x0_uncond

    if APG_parameters["momentum_buffer"] is not None:
        APG_parameters["momentum_buffer"].update(diff)
        diff = APG_parameters["momentum_buffer"].running_avg

    if APG_parameters["norm_threshold"] > 0.0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim = [-1,-2,-3], keepdim=True)
        scale_factor = torch.minimum(ones, APG_parameters["norm_threshold"] / (diff_norm + 1e-8))
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = _decompose_parallel_ortho_component(diff, x0_uncond)

    normalized_update = diff_orthogonal + APG_parameters["eta"]*diff_parallel
    x0_guided = x0_text + (guidance_scale - 1) * normalized_update

    pred_guided = _to_noise(x0_guided, latents, time)

    return pred_guided

def zero_star_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, zeros_steps, use_zero_init, step_index):
    """
    Implements the zero* guidance method.

    Args:
        noise_pred_uncond: The noise prediction for the unconditional input, shape (B, C, H, W).
        noise_pred_text: The noise prediction for the text input, shape (B, C, H, W).
        guidance_scale: The scale of the guidance to apply.
        use_zero_init: boolean. If true, the first steps force the velocity to be zero.
        zeros_steps: number of steps where the velocity is forced to be 0.
        step_index: current step index
    
    Returns:
        The guided noise prediction, shape (B, C, H, W).
    """

    if (use_zero_init) and (step_index < zeros_steps):
        return torch.zeros_like(noise_pred_uncond)
    
    s_star = _compute_projection(noise_pred_text, noise_pred_uncond)
    pred_guided = (1-guidance_scale)*s_star*noise_pred_uncond + guidance_scale*noise_pred_text

    return pred_guided

def sliding_mode_control_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, lambda_param, k, guidance_term_buffer):
    """
    Implements the SMC (sliding mode control CFG).

    Args:
        noise_pred_uncond: The noise prediction for the unconditional input, shape (B, C, H, W).
        noise_pred_text: The noise prediction for the text input, shape (B, C, H, W).
        guidance_scale: The scale of the guidance to apply.
        lambda_param: Shape parameter of the sliding mode surface.
        k: Gain of the switching control term.
        guidance_term_buffer: store previous semantic error signal (e(t)).

    Returns:
        The guided noise prediction, shape (B, C, H, W).
    """
    current_e = noise_pred_text - noise_pred_uncond

    if (guidance_term_buffer.previous_e is None):
        guidance_term_buffer.update(current_e)
    
    sliding = (current_e - guidance_term_buffer.previous_e) + lambda_param * guidance_term_buffer.previous_e
    
    delta_guidance = - k * _smooth_sign(sliding)

    current_e = current_e + delta_guidance

    pred_guided = noise_pred_uncond + guidance_scale * delta_guidance

    guidance_term_buffer.update(current_e)
    
    return pred_guided

def _smooth_sign(x, eps=1e-6):
    """
    Compute sign.
    """
    return x / (x.abs() + eps)