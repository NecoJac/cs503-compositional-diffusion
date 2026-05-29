#!/bin/bash
# num_img=5: 5-window panoramic bridge (1536px wide)
set -e

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
export PYTHONUNBUFFERED=1

SCRATCH=${DIFFUSION_SCRATCH_ROOT:-${TMPDIR:-/tmp}/diffusion}
export HF_HOME=$SCRATCH/hf_home HF_HUB_CACHE=$SCRATCH/hf_hub
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets TMPDIR=$SCRATCH/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

echo "=== num_img=5, SMC K=8 β=1.0 ==="
date
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs \
    "a calm lakeside landscape:a volcanic eruption" \
    "a lush rainforest:a barren desert" \
    "a snowy mountain peak:a sandy tropical beach" \
    "a clear blue sky at noon:a dark stormy sky with lightning" \
    "a cherry blossom park in spring:a snow covered park in winter" \
  --output-root proposal_outputs/num5_smc \
  --num-img 5 --overlap-latents 32 --n-steps 50 --guidance-scale 3.0 \
  --smc-only --smc-K 8 --smc-beta 1.0 --smc-resample-end 0.8 \
  --seed 42 --device cuda

echo ""
echo "=== num_img=5, bridge_correction c=0.01 ==="
date
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs \
    "a calm lakeside landscape:a volcanic eruption" \
    "a lush rainforest:a barren desert" \
    "a snowy mountain peak:a sandy tropical beach" \
    "a clear blue sky at noon:a dark stormy sky with lightning" \
    "a cherry blossom park in spring:a snow covered park in winter" \
  --output-root proposal_outputs/num5_proposal \
  --num-img 5 --overlap-latents 32 --n-steps 50 --guidance-scale 3.0 \
  --proposal-couplings 0.01 \
  --seed 42 --device cuda

echo "=== DONE $(date) ==="
echo "输出目录:"
echo "  proposal_outputs/num5_smc/smc_K8_b1.0_c0.05/*/sample_000.png   (1536x512)"
echo "  proposal_outputs/num5_proposal/bridge_correction_c0.01/*/sample_000.png   (1536x512)"
