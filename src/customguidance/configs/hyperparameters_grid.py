HYPERPARAMETER_GRID = {
    "constant": {},

    "linear": {},

    "exponential": {},

    "APG": {
        "momentum_value": [-0.75, 0.0, 0.5],
        "eta": [0.0, 0.5],
        "norm_threshold": [2.5, 5, 10],
    },

    "zero_star": {
        "zero_steps": [1, 2, 3],
        "use_zero_init": [False, True],
    },

    "rectified_pp": {
        "lambda_max": [0.5, 1.0, 2.0, 3.0],
        "gamma": [0.5, 1.0, 2.0],
    },

    "SMC": {
        "lambda_param": [2.5, 5.0, 7.5],
        "k": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75],
    },
}