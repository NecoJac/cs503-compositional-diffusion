#!/bin/bash
set -e
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=$(which python3)
cd "$REPO"
export PYTHONUNBUFFERED=1
SCRATCH=${DIFFUSION_SCRATCH_ROOT:-${TMPDIR:-/tmp}/diffusion}
export HF_HOME=$SCRATCH/hf_home HF_HUB_CACHE=$SCRATCH/hf_hub HF_DATASETS_CACHE=$SCRATCH/hf_datasets TMPDIR=$SCRATCH/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

echo "=== proposal_smc only, 22 pairs, K=4 beta=1.0 ==="
date
$PY -m proposal_methods.evaluate_sd_bridge \
  --text-pairs \
    "a calm lakeside landscape:a volcanic eruption" \
    "a lush rainforest:a barren desert" \
    "a snowy mountain peak:a sandy tropical beach" \
    "a misty autumn forest:a bright spring meadow" \
    "a frozen arctic tundra:a humid jungle" \
    "a clear blue sky at noon:a dark stormy sky with lightning" \
    "a golden sunset over the ocean:a starry night sky" \
    "a foggy morning valley:a bright sunny afternoon" \
    "a heavy blizzard:a warm summer day" \
    "a rainbow over green hills:a tornado over a flat plain" \
    "a quiet rural countryside:a busy neon city at night" \
    "a sunlit cobblestone village:an empty midnight street" \
    "a bustling daytime market:a silent moonlit plaza" \
    "a mountain dawn:an urban dusk with skyscrapers" \
    "a cherry blossom park in spring:a snow covered park in winter" \
    "a golden wheat field in summer:a bare muddy field in autumn" \
    "a colorful autumn maple forest:a green summer forest" \
    "a frozen lake in winter:a blooming flower meadow in spring" \
    "a peaceful monastery garden:a chaotic urban construction site" \
    "an underwater coral reef:an outer space nebula" \
    "a medieval castle on a hill:a futuristic glass skyscraper" \
    "a quiet library interior:a loud rock concert stage" \
  --output-root proposal_outputs/sd_smc_only \
  --num-img 3 --overlap-latents 32 --n-steps 50 --guidance-scale 3.0 \
  --smc-only --smc-K 4 --smc-beta 1.0 --smc-resample-end 0.8 \
  --device cuda
echo "=== DONE $(date) ==="
