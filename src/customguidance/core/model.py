import torch
from diffusers.utils import logging as diffusers_logging
from transformers.utils.logging import disable_progress_bar

from ..pipeline import StableDiffusion3PipelineCustomGuidance, Flux2KleinPipelineCustomGuidance
from ..error import *

def load_model(model: str, model_path: str, guidance_type: str, guidance_params: dict = None):
    """
    Load a generative model with a given guidance method.

    Args:
        model (str): Generative model to use. Available: ['SD3'].
        model_path (str): Path to the downloaded model.
        guidance_type (str): Guidance method to use.

    Returns:
        Loaded model pipeline.
    """
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    check_model_downloaded_path(model_path)
    check_existing_generative_model(model)
    check_existing_guidance_method(guidance_type)

    diffusers_logging.disable_progress_bar()
    disable_progress_bar()

    if model == "SD3":
        pipeline = StableDiffusion3PipelineCustomGuidance.from_pretrained(model_path, torch_dtype=torch.float32)
    
    if model == "flux2Klein":
        pipeline = Flux2KleinPipelineCustomGuidance.from_pretrained(model_path, torch_dtype=torch.float32)
    
    pipeline.configure_guidance(guidance_type=guidance_type, guidance_params=guidance_params)
    pipeline.to(torch_device)

    pipeline.set_progress_bar_config(disable=True)

    return pipeline

def generate_image(model_pipeline, prompt: str, height: int = 512, width: int = 512,
                   num_inference_steps: int = 28, guidance_scale: float = 7):
    """
    Generate an image using a loaded model pipeline.

    Args:
        model_pipeline: Loaded generative model pipeline.
        prompt (str): Text prompt for image generation.
        height (int): Height of the generated image in pixels.
        width (int): Width of the generated image in pixels.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Adherence to the prompt vs. image quality.

    Returns:
        PIL.Image: Generated image.
    """
    result = model_pipeline(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale, generator=torch.Generator().manual_seed(13))

    return result.images[0]

def run(model: str, guidance_type: str, model_path: str, prompt: str, height: int = 512, 
        width: int = 512, num_inference_steps: int = 28, guidance_scale: float = 7, 
        guidance_params: dict = None):
    """
    Run inference on a generative model with a given guidance method.

    Args:
        model (str): Generative model to use. Available: ['SD3'].
        guidance_type (str): Guidance method to use.
        model_path (str): Path to the downloaded model.
        prompt (str): Text prompt for image generation.
        height (int): Height of the generated image in pixels.
        width (int): Width of the generated image in pixels.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Adherence to the prompt vs. image quality.

    Returns:
        PIL.Image: Generated image.
    """
    model = load_model(model, model_path, guidance_type, guidance_params=guidance_params)

    generated_image = generate_image(model, prompt, height, width, num_inference_steps, guidance_scale)

    return generated_image
