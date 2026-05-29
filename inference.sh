#!/bin/bash

MODEL="SD3"
#MODEL="flux2Klein"
MODEL_PATH="/home/schiesser/models/sd35_medium"
#MODEL_PATH="/home/schiesser/models/flux2_klein_base_4B"

PROMPT="a beautiful sunset over the mountains"

HEIGHT=1024
WIDTH=1024

NUM_STEPS=40
GUIDANCE_SCALE=4.5
#GUIDANCE_SCALE=4.0
GUIDANCE_TYPE="APG"
GUIDANCE_PARAMETERS='{"momentum_value": 0.3, "eta": -0.75, "norm_threshold": 15.0}'

GENERATED_IMAGES_NAME="output.png"

python scripts/inference.py \
    --model "$MODEL" \
    --guidance_type "$GUIDANCE_TYPE" \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --guidance_parameters "$GUIDANCE_PARAMETERS" \
    --output "$GENERATED_IMAGES_NAME"