import shutil
import itertools
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from ..data_utils import extract_image_info
from ..evaluation import compute_scores
from ..error import *
from .model import load_model, generate_image
from .benchmark import compute_scores
from ..configs.hyperparameters_grid import HYPERPARAMETER_GRID

def hyperparameter_search(model: str, guidance_method: str, model_path: str,
                          data_annotations_path: str, data_images_path: str,
                          num_inference_steps: int = 28, guidance_scale: float = 7,
                          score_list: list[str] = ["FID"], number_of_images: int = 5000,
                          run_id: str = "test_run", clip_model_path: str = None,
                          blip_model_path: str = None, seed: int = 13,
                          height: int = 512, width: int = 512,
                          keep_images: bool = False, save_result_path:str=""):
    """
    Run a hyperparameter search for one guidance method.

    For each hyperparameter combination in the predefined grid, generate images,
    compute scores, and save one row in a dataframe. The final dataframe is saved
    as '{guidance_method}_hyperparameter.csv'.

    Args:
        model (str): Generative model to use.
        guidance_method (str): Guidance method to test.
        model_path (str): Path to the downloaded model.
        data_annotations_path (str): Path to dataset annotations.
        data_images_path (str): Path to dataset images.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Guidance scale.
        score_list (list[str]): Metrics to compute.
        number_of_images (int): Number of images to generate/evaluate.
        run_id (str): Id of the run.
        clip_model_path (str): Path to CLIP model.
        blip_model_path (str): Path to BLIP model.
        seed (int): Random seed.
        height (int): Generation height for non-FID metrics.
        width (int): Generation width for non-FID metrics.
        keep_images (bool): Whether to keep generated images.

    Returns:
        pd.DataFrame: Dataframe containing one row per hyperparameter combination.
    """
    # Validate inputs
    check_model_downloaded_path(model_path)
    check_existing_generative_model(model)
    check_existing_guidance_method(guidance_method)
    check_existing_data_path(data_annotations_path)
    check_existing_data_path(data_images_path)
    for score_name in score_list: check_existing_evaluation_metric(score_name)

    # Get captions, dimensions and image names
    images_info = extract_image_info(data_annotations_path, seed=seed, keep_divisible_16=True)

    # Retrieve hyperparameter grid for the given method
    if guidance_method not in HYPERPARAMETER_GRID:
        raise ValueError(f"No hyperparameter grid found for guidance method '{guidance_method}'.")

    hyperparameter_grid = HYPERPARAMETER_GRID[guidance_method]
    hyperparameter_combinations = build_hyperparameter_combinations(hyperparameter_grid)

    # Load model once
    initial_params = hyperparameter_combinations[0] if hyperparameter_combinations[0] is not None else None
    pipeline_model = load_model(model, model_path, guidance_method, initial_params)

    results = []

    for i, guidance_params in tqdm(enumerate(hyperparameter_combinations), total=len(hyperparameter_combinations), desc=f"Hyperparameter search [{guidance_method}]"):
        # Reconfigure guidance without reloading the model
        pipeline_model.configure_guidance(guidance_type=guidance_method, guidance_params=guidance_params)

        combination_name = f"combination_{i}"

        if any(s in score_list for s in ["IS", "CLIP", "BLIP"]):
            path_generated_images = f"outputs/{run_id}/{guidance_method}/{combination_name}/generated_images"
            Path(path_generated_images).mkdir(parents=True, exist_ok=True)
        else:
            path_generated_images = None

        if "FID" in score_list:
            path_original_fid = f"outputs/{run_id}/{guidance_method}/{combination_name}/FID/original_images"
            path_generated_fid = f"outputs/{run_id}/{guidance_method}/{combination_name}/FID/generated_images"
            Path(path_original_fid).mkdir(parents=True, exist_ok=True)
            Path(path_generated_fid).mkdir(parents=True, exist_ok=True)
        else:
            path_original_fid = None
            path_generated_fid = None

        for _, row in images_info.iloc[:number_of_images].iterrows():
            if "FID" in score_list:
                pipeline_model.guidance_method.reset()
                generated_image_fid = generate_image(pipeline_model, row['caption'], row['height'], row['width'], num_inference_steps, guidance_scale)
                generated_image_fid.save(f"{path_generated_fid}/{row['file_name']}")
                shutil.copy(f"{data_images_path}/{row['file_name']}", f"{path_original_fid}/{row['file_name']}")

            if any(s in score_list for s in ["IS", "CLIP", "BLIP"]):
                pipeline_model.guidance_method.reset()
                generated_image = generate_image(pipeline_model, row['caption'], height, width, num_inference_steps, guidance_scale)
                generated_image.save(f"{path_generated_images}/{row['file_name']}")

        # Compute scores
        dict_score = compute_scores(images_info=images_info, number_of_images=number_of_images,
                                    score_list=score_list, seed=seed, path_generated_images=path_generated_images,
                                    path_generated_fid=path_generated_fid, path_original_fid=path_original_fid,
                                    clip_model_path=clip_model_path, blip_model_path=blip_model_path)

        # Save one row per combination
        row_result = {}
        if guidance_params is not None:
            for key, value in guidance_params.items():
                row_result[key] = value

        row_result["fid"] = dict_score["fid"]
        row_result["is_mean"] = dict_score["is_mean"]
        row_result["is_std"] = dict_score["is_std"]
        row_result["clip"] = dict_score["clip"]
        row_result["blip"] = dict_score["blip"]

        results.append(row_result)

        if not keep_images:
            shutil.rmtree(f"outputs/{run_id}/{guidance_method}/{combination_name}", ignore_errors=True)

    df_results = pd.DataFrame(results)

    output_dir = Path(save_result_path) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_dir / f"{guidance_method}_hyperparameter.csv", index=False)

    return df_results

def build_hyperparameter_combinations(param_grid: dict) -> list[dict]:
    """
    Build all combinations of the hyperparameter grid.

    Args:
        param_grid (dict): Dictionary with one list per hyperparameter.

    Returns:
        list[dict]: List of dictionaries, one per combination.
    """
    if param_grid is None or len(param_grid) == 0:
        return [None]

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    combinations = []
    for values in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, values)))

    return combinations