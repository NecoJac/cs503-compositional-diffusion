#!/bin/bash

MODEL="SD3"
#MODEL="flux2Klein"

MODEL_PATH="/home/schiesser/models/sd35_medium"
#MODEL_PATH="/home/schiesser/models/flux2_klein_base_4B"
DATA_IMAGES_PATH="/home/schiesser/datasets/MS_COCO/val2017"
DATA_ANNOTATIONS_PATH="/home/schiesser/datasets/MS_COCO/annotations/captions_val2017.json"

CLIP_MODEL_PATH="/home/schiesser/models/clip"
BLIP_MODEL_PATH="/home/schiesser/models/blip"

SAVE_PATH="/home/schiesser"

NUM_IMAGES=5000
HEIGHT=1024
WIDTH=1024

NUM_STEPS=40
GUIDANCE_SCALE=4.5
#GUIDANCE_SCALE=4.0

GUIDANCE_TYPES=("APG" "zero_star" "rectified_pp" "SMC")

SCORES=("FID" "CLIP" "IS" "BLIP")
KEEP_IMAGES=false

RUN_ID="hparams_sd3"
#RUN_ID="hparams_flux2"
SEED=13

for GUIDANCE_METHOD in "${GUIDANCE_TYPES[@]}"
do
    echo "========================================="
    echo "Running hyperparameter search for: $GUIDANCE_METHOD"
    echo "========================================="

    python scripts/hyperparameters.py \
        --model "$MODEL" \
        --guidance_method "$GUIDANCE_METHOD" \
        --model_path "$MODEL_PATH" \
        --data_annotations_path "$DATA_ANNOTATIONS_PATH" \
        --data_images_path "$DATA_IMAGES_PATH" \
        --height $HEIGHT \
        --width $WIDTH \
        --num_inference_steps $NUM_STEPS \
        --guidance_scale $GUIDANCE_SCALE \
        --score_list "${SCORES[@]}" \
        --number_of_images $NUM_IMAGES \
        --run_id "${RUN_ID}_${GUIDANCE_METHOD}" \
        --clip_model_path "$CLIP_MODEL_PATH" \
        --blip_model_path "$BLIP_MODEL_PATH" \
        --seed $SEED \
        --keep_images $KEEP_IMAGES \
        --save_path $SAVE_PATH

    echo "Finished: $GUIDANCE_METHOD"
    echo ""
done