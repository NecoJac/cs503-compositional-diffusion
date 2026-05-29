from pathlib import Path

# LIST OF AVAILABLE MODEL / PERF METRICS / CFG METHOD
implemented_generative_model = {"SD3", "flux2Klein"}

implemented_performance_metrics = {"FID", "CLIP",
                                    "IS", "BLIP"}

implemented_guidance_methods = {"constant", "linear",
                                "exponential", "APG",
                                "rectified_pp", "zero_star",
                                "SMC"}

# REQUIRED PARAMETER FOR EACH CFG METHOD
REQUIRED_APG_PARAMETERS = {"momentum_value": (int, float), 
                           "norm_threshold": (int, float), 
                           "eta": (int, float)}

REQUIRED_ZERO_STAR_PARAMETERS = {"zero_steps": (int,),
                                 "use_zero_init": (bool,)}

REQUIRED_RECTIFIED_PP_PARAMETERS = {"lambda_max": (float,),
                                    "gamma": (float,)}

REQUIRED_SMC_PARAMETERS = {"lambda_param": (float,),
                           "k": (float,)}

REQUIRED_PARAMETERS = {"APG": REQUIRED_APG_PARAMETERS,
                       "rectified_pp": REQUIRED_RECTIFIED_PP_PARAMETERS,
                       "zero_star": REQUIRED_ZERO_STAR_PARAMETERS,
                       "SMC": REQUIRED_SMC_PARAMETERS}


def check_existing_generative_model(model_name: str):
    """Raises ValueError if the generative model is not available."""
    if model_name not in implemented_generative_model:
        raise ValueError(f"Model '{model_name}' not implemented. Available: {implemented_generative_model}.")

def check_existing_guidance_method(guidance_method_name: str):
    """Raises ValueError if the guidance method is not available."""
    if guidance_method_name not in implemented_guidance_methods:
        raise ValueError(f"Guidance method '{guidance_method_name}' not implemented. Available: {implemented_guidance_methods}.")

def check_existing_evaluation_metric(metric_name: str):
    """Raises ValueError if the evaluation metric is not available."""
    if metric_name not in implemented_performance_metrics:
        raise ValueError(f"Metric '{metric_name}' not implemented. Available: {implemented_performance_metrics}.")

def check_model_downloaded_path(model_path: str):
    """Raises FileNotFoundError if the model path does not exist."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No model found at: '{model_path}'.")
    
def check_existing_data_path(data_folder_path: str):
    """Raises FileNotFoundError if the data folder path does not exist."""
    if not Path(data_folder_path).exists():
        raise FileNotFoundError(f"No data folder found at: '{data_folder_path}'.")
    
def _check_params(guidance_type: str, guidance_params: dict, required: dict):
    for key, expected_type in required.items():
        if key not in guidance_params:
            raise ValueError(f"Missing key '{key}' in {guidance_type} parameters")
        if not isinstance(guidance_params[key], expected_type):
            raise TypeError(f"{guidance_type} parameter '{key}' must be of type {expected_type}, got {type(guidance_params[key])}")
    
def check_guidance_parameters(guidance_type: str, guidance_params: dict | None):
    """
    Raises a error if the parameters provided for a given CFG method is not suitable:
    missing parameters or wrong type !
    """
    if guidance_type in {"constant", "linear", "exponential"}:
        return
    if guidance_params is None:
        raise ValueError(f"`guidance_params` must be provided for guidance_type='{guidance_type}'")
    if not isinstance(guidance_params, dict):
        raise TypeError(f"`guidance_params` must be a dict, got {type(guidance_params)}")
    if guidance_type in REQUIRED_PARAMETERS:
        _check_params(guidance_type, guidance_params, REQUIRED_PARAMETERS[guidance_type])

