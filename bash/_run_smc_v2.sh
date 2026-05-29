#!/bin/bash
# SMC v2: x0 disagreement FK potential (replaces eps inner product)
# Run this on the GPU interactive node with cbg_diffusion env activated.
# Usage: conda activate cbg_diffusion && bash _run_smc_v2.sh
set -e

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
export PYTHONUNBUFFERED=1

SCRATCH=${DIFFUSION_SCRATCH_ROOT:-${TMPDIR:-/tmp}/diffusion}
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$SCRATCH/hf_hub
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets
export TMPDIR=$SCRATCH/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

TEXT_PAIRS=(
  "a calm lakeside landscape:a volcanic eruption"
  "a lush rainforest:a barren desert"
  "a snowy mountain peak:a sandy tropical beach"
  "a misty autumn forest:a bright spring meadow"
  "a frozen arctic tundra:a humid jungle"
  "a clear blue sky at noon:a dark stormy sky with lightning"
  "a golden sunset over the ocean:a starry night sky"
  "a foggy morning valley:a bright sunny afternoon"
  "a heavy blizzard:a warm summer day"
  "a rainbow over green hills:a tornado over a flat plain"
  "a quiet rural countryside:a busy neon city at night"
  "a sunlit cobblestone village:an empty midnight street"
  "a bustling daytime market:a silent moonlit plaza"
  "a mountain dawn:an urban dusk with skyscrapers"
  "a cherry blossom park in spring:a snow covered park in winter"
  "a golden wheat field in summer:a bare muddy field in autumn"
  "a colorful autumn maple forest:a green summer forest"
  "a frozen lake in winter:a blooming flower meadow in spring"
  "a peaceful monastery garden:a chaotic urban construction site"
  "an underwater coral reef:an outer space nebula"
  "a medieval castle on a hill:a futuristic glass skyscraper"
  "a quiet library interior:a loud rock concert stage"
)

# ── Run 1: new SMC, K=8 β=1.0 (same config as old best, now with x0-disagree) ──
echo ""
echo "=== SMC v2: x0-disagree FK, K=8 β=1.0 seed=42 ==="
date
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs "${TEXT_PAIRS[@]}" \
  --output-root proposal_outputs/smc_v2_K8_b1 \
  --num-img 3 --overlap-latents 32 --n-steps 50 --guidance-scale 3.0 \
  --smc-only --smc-K 8 --smc-beta 1.0 --smc-resample-end 0.8 \
  --seed 42 --device cuda
echo "--- K=8 β=1.0 done ---"

# ── Run 2: K=4 β=0.5 ──
echo ""
echo "=== SMC v2: x0-disagree FK, K=4 β=0.5 seed=42 ==="
date
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs "${TEXT_PAIRS[@]}" \
  --output-root proposal_outputs/smc_v2_K4_b0.5 \
  --num-img 3 --overlap-latents 32 --n-steps 50 --guidance-scale 3.0 \
  --smc-only --smc-K 4 --smc-beta 0.5 --smc-resample-end 0.8 \
  --seed 42 --device cuda
echo "--- K=4 β=0.5 done ---"

# ── Run 3: K=8 β=2.0 ──
echo ""
echo "=== SMC v2: x0-disagree FK, K=8 β=2.0 seed=42 ==="
date
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs "${TEXT_PAIRS[@]}" \
  --output-root proposal_outputs/smc_v2_K8_b2 \
  --num-img 3 --overlap-latents 32 --n-steps 50 --guidance-scale 3.0 \
  --smc-only --smc-K 8 --smc-beta 2.0 --smc-resample-end 0.8 \
  --seed 42 --device cuda
echo "--- K=8 β=2.0 done ---"

# ── Run 4: K=8 β=1.0, guidance=7.0（视觉质量对比）──
echo ""
echo "=== SMC v2: x0-disagree FK, K=8 β=1.0 guidance=7.0 seed=42 ==="
date
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs "${TEXT_PAIRS[@]}" \
  --output-root proposal_outputs/smc_v2_K8_b1_cfg7 \
  --num-img 3 --overlap-latents 32 --n-steps 50 --guidance-scale 7.0 \
  --smc-only --smc-K 8 --smc-beta 1.0 --smc-resample-end 0.8 \
  --seed 42 --device cuda
echo "--- K=8 β=1.0 cfg=7.0 done ---"

# ── Run 5: K=8 β=0.5, guidance=7.0 ──
echo ""
echo "=== SMC v2: x0-disagree FK, K=8 β=0.5 guidance=7.0 seed=42 ==="
date
python -m proposal_methods.evaluate_sd_bridge \
  --text-pairs "${TEXT_PAIRS[@]}" \
  --output-root proposal_outputs/smc_v2_K8_b0.5_cfg7 \
  --num-img 3 --overlap-latents 32 --n-steps 50 --guidance-scale 7.0 \
  --smc-only --smc-K 8 --smc-beta 0.5 --smc-resample-end 0.8 \
  --seed 42 --device cuda
echo "--- K=8 β=0.5 cfg=7.0 done ---"

# ── Summary ──
echo ""
echo "=== SUMMARY ==="
echo "Baseline (old SMC K=8 β=1.0 cfg=3.0):   seam_mse_mean = 0.0190"
echo "Baseline (bridge_correction c=0.01 cfg=3.0): seam_mse_mean = 0.0185"
echo ""
for dir in proposal_outputs/smc_v2_K8_b1 proposal_outputs/smc_v2_K4_b0.5 proposal_outputs/smc_v2_K8_b2 proposal_outputs/smc_v2_K8_b1_cfg7 proposal_outputs/smc_v2_K8_b0.5_cfg7; do
  if [ -f "$dir/summary_metrics.csv" ]; then
    echo "[$dir]"
    cat "$dir/summary_metrics.csv"
  fi
done
echo "=== DONE $(date) ==="
