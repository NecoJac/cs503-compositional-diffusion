import pickle
import shutil
from pathlib import Path
from tqdm import tqdm

from ..data_utils import extract_image_info
from ..evaluation import compute_scores
from ..error import *
from .model import load_model, generate_image

def benchmark(model: str, guidance_types: list[str], model_path: str, data_annotations_path: str, 
              data_images_path: str, num_inference_steps: int = 28, guidance_scale: float = 7, 
              score_list: list[str] = ["FID"], number_of_images: int = 5000, run_id: str = "test_run", 
              clip_model_path: str = None, blip_model_path: str = None, seed: int = 13, height:int=512,
              width:int=512, guidance_parameters: list[dict] = None, keep_images: bool = False, save_result_path:str=""):
    """
    Run a benchmark:
    retrieve scores for guidances_types for a given generative model and a given dataset.
    The way the prompt are retrieved work for MS-COCO dataset.
    For other dataset the function "extract_image_info" need to be changed.

    Args:
        model (str): Generative model to use. Available: ['SD3'].
        guidance_types (list[str]): List of guidance methods to use.
        model_path (str): Path to the downloaded model.
        data_annotations_path (str): Path to the dataset annotations (csv file).
        data_images_path (str): Path to the dataset images (folder).
        height (int): Height of the generated image in pixels.
        width (int): Width of the generated image in pixels.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Adherence to the prompt vs. image quality.
        score_list (list[str]): List of evaluation metrics to compute. Available: ['FID', 'IS', 'CLIP', 'BLIP'].
        number_of_images (int): Number of images to generate and evaluate. (from COCO caption, max 5000)
        run_id (str): Id of the run, used to save generated images in a specific folder.
        clip_model_path (str): Path to the CLIP model to be used for scoring. Required if "CLIP" in score_list.
        blip_model_path (str): Path to the BLIP model to be used for scoring. Required if "BLIP" in score_list.
    
    Returns:
        dict: Dictionary containing the scores for each guidance method and each evaluation metric.
    """
    # Validate inputs
    check_model_downloaded_path(model_path)
    check_existing_generative_model(model)
    for guidance_method in guidance_types: check_existing_guidance_method(guidance_method)
    check_existing_data_path(data_annotations_path)
    check_existing_data_path(data_images_path)
    for score_name in score_list: check_existing_evaluation_metric(score_name)

    # get captions, dimensions and jpeg name of the original images
    images_info = extract_image_info(data_annotations_path, seed=seed, keep_divisible_16 = True)
    
    # generate images with the given model/guidance method for every prompt and save them in a target folder 
    full_score = {}
    for i, guidance_method in tqdm(enumerate(guidance_types), total=len(guidance_types), desc="Guidance methods"):

        if any(s in score_list for s in ["IS", "CLIP", "BLIP"]):
            path_generated_images = f"outputs/{run_id}/generated_images/{guidance_method}"
            Path(path_generated_images).mkdir(parents=True, exist_ok=True)

        if "FID" in score_list:
            path_original_fid = f"outputs/{run_id}/FID/original_images/{guidance_method}"
            Path(path_original_fid).mkdir(parents=True, exist_ok=True)
            path_generated_fid = f"outputs/{run_id}/FID/generated_images/{guidance_method}"
            Path(path_generated_fid).mkdir(parents=True, exist_ok=True)

        pipeline_model = load_model(model, model_path, guidance_method, guidance_parameters[i] if guidance_parameters is not None else None)

        for _, row in images_info.iloc[:number_of_images].iterrows():

            # for FID (keep same dimension between original and generated images)
            if "FID" in score_list:
                pipeline_model.guidance_method.reset()
                generated_image_fid = generate_image(pipeline_model, row['caption'], row['height'], row['width'], num_inference_steps, guidance_scale)
                generated_image_fid.save(f"{path_generated_fid}/{row['file_name']}")
                shutil.copy(f"{data_images_path}/{row['file_name']}", f"{path_original_fid}/{row['file_name']}")

            if any(s in score_list for s in ["IS", "CLIP", "BLIP"]):
                pipeline_model.guidance_method.reset()
                generated_image = generate_image(pipeline_model, row['caption'], height, width, num_inference_steps, guidance_scale)
                generated_image.save(f"{path_generated_images}/{row['file_name']}")
        
        # evaluate the score of the generated images against the original ones with the given evaluation metric(s)
        full_score[guidance_method] = compute_scores(images_info, number_of_images, score_list, seed,
                                                     path_generated_images, path_generated_fid,
                                                     path_original_fid, clip_model_path, blip_model_path)
    if not keep_images:
        shutil.rmtree(f"outputs/{run_id}")

    output_dir = Path(save_result_path) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(full_score, f)

    return full_score