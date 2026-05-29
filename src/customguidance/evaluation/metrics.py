from cleanfid import fid
import torch_fidelity
import torch
from torchmetrics.multimodal import CLIPScore
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForImageTextRetrieval, CLIPModel, CLIPProcessor
from ..error import check_existing_data_path, check_model_downloaded_path

def compute_fid(generated_image_path: str,
                real_image_path: str):
    """
    Computes the FID score between generated and real images.

    Args:
        generated_image_path (str): Path to the directory containing generated images.
        real_image_path (str): Path to the directory containing real images.
    """
    check_existing_data_path(generated_image_path)
    check_existing_data_path(real_image_path)

    score = fid.compute_fid(generated_image_path, real_image_path)
    return score

def compute_is(generated_image_path: str, 
               seed: int = 13):
    """
    Computes the Inception Score (IS) for generated images.

    Args:
        generated_image_path (str): Path to the directory containing generated images.
        seed (int): Random seed for reproducibility.
    """
    check_existing_data_path(generated_image_path)

    metrics = torch_fidelity.calculate_metrics(input1=generated_image_path, isc=True, isc_splits=10, rng_seed=seed)
    mean = metrics["inception_score_mean"]
    std  = metrics["inception_score_std"]
    return mean, std

def compute_clip_score(generated_image_path: str, 
                       prompts: list[str], 
                       clip_model_path: str):
    """
    Computes the CLIP score between generated images and their corresponding prompts.
    model: openai/clip-vit-base-patch32.

    This function used images that have been generated and saved in a folder.
    For better performance the score can be computed on images one by one after generation, 
    a step of open can be avoided. But here we favorize the "workflow".

    Args:
        generated_image_path (str): Path to the directory containing generated images.
        prompts (list[str]): List of prompts corresponding to the generated images.
        clip_model_path (str): Path to the CLIP model to be used for scoring.
    """
    check_existing_data_path(generated_image_path)
    check_model_downloaded_path(clip_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(clip_model_path).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_path)
    model.eval()

    image_paths = sorted(Path(generated_image_path).glob("*.jpg"))
    scores = []

    for path, prompt in zip(image_paths, prompts):
        image = Image.open(path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # cosine similarity between image and text embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds  = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
            score = (image_embeds * text_embeds).sum(dim=-1).item()

        scores.append(score)

    mean_score = sum(scores) / len(scores)
    return mean_score

def compute_blip_score(generated_image_path: str, 
                       prompts: list[str], blip_model_path: str):
    """
    Computes the BLIP score between generated images and their corresponding prompts.
    model: Salesforce/blip-itm-base-coco
    This function used images that have been generated and saved in a folder.
    (comments about performance are as for CLIP score)

    Args:
        generated_image_path (str): Path to the directory containing generated images.
        prompts (list[str]): List of prompts corresponding to the generated images.
    """
    check_existing_data_path(generated_image_path)
    check_model_downloaded_path(blip_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(blip_model_path)
    model = BlipForImageTextRetrieval.from_pretrained(blip_model_path).to(device)
    model.eval()

    scores = []
    image_paths = sorted(Path(generated_image_path).glob("*.jpg"))

    for path, prompt in zip(image_paths, prompts):
        image  = Image.open(path).convert("RGB")
        inputs = processor(image, prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            score = model(**inputs, use_itm_head=True).itm_score
            score = torch.nn.functional.softmax(score, dim=1)[:, 1].item()

        scores.append(score)

    mean_score = sum(scores) / len(scores)
    return mean_score

def compute_scores(images_info, number_of_images: int, score_list: list[str],
                   seed: int, path_generated_images: str = None,
                   path_generated_fid: str = None, path_original_fid: str = None,
                   clip_model_path: str = None, blip_model_path: str = None):
    """
    Compute evaluation metrics from saved image folders.

    Args:
        images_info: Dataframe containing captions and file names.
        number_of_images (int): Number of images to consider.
        score_list (list[str]): Metrics to compute.
        seed (int): Random seed.
        path_generated_images (str): Folder containing generated images for IS/CLIP/BLIP.
        path_generated_fid (str): Folder containing generated images for FID.
        path_original_fid (str): Folder containing original images for FID.
        clip_model_path (str): Path to CLIP model.
        blip_model_path (str): Path to BLIP model.

    Returns:
        dict: Dictionary containing all computed scores.
    """
    fid_score, is_mean, is_std, clip_score, blip_score = None, None, None, None, None

    prompts = images_info['caption'].iloc[:number_of_images].tolist()
    sorted_key = images_info['file_name'].iloc[:number_of_images].tolist()
    prompts = [prompt for _, prompt in sorted(zip(sorted_key, prompts))]

    if "FID" in score_list:
        fid_score = compute_fid(path_generated_fid, path_original_fid)

    if "IS" in score_list:
        is_mean, is_std = compute_is(path_generated_images, seed=seed)

    if "CLIP" in score_list:
        clip_score = compute_clip_score(path_generated_images, prompts, clip_model_path)

    if "BLIP" in score_list:
        blip_score = compute_blip_score(path_generated_images, prompts, blip_model_path)

    return {"fid": fid_score, "is_mean": is_mean, "is_std": is_std, "clip": clip_score, "blip": blip_score}