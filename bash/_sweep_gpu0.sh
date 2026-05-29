#!/bin/bash
# GPU 0 — seed × β robustness sweep
# Focus: find configs that give <0.022 consistently across multiple seeds
# Usage: CUDA_VISIBLE_DEVICES=0 conda activate cbg_diffusion && bash _sweep_gpu0.sh
set -e

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
export PYTHONUNBUFFERED=1

SCRATCH=${DIFFUSION_SCRATCH_ROOT:-${TMPDIR:-/tmp}/diffusion}
export HF_HOME=$SCRATCH/hf_home HF_HUB_CACHE=$SCRATCH/hf_hub
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets TMPDIR=$SCRATCH/tmp
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

run() {
  local K=$1 beta=$2 seed=$3 extra_label=${4:-}
  local label="smc_K${K}_b${beta}_s${seed}${extra_label}"
  local outdir="proposal_outputs/sweep_gpu0/${label}"
  echo ""
  echo "=== [GPU0] K=${K} β=${beta} seed=${seed} → ${outdir} ==="
  date
  python -m proposal_methods.evaluate_sd_bridge \
    --text-pairs "${TEXT_PAIRS[@]}" \
    --output-root "${outdir}" \
    --num-img 3 --overlap-latents 32 --n-steps 50 --guidance-scale 3.0 \
    --smc-only --smc-K "${K}" --smc-beta "${beta}" \
    --smc-resample-start 0.0 --smc-resample-end 0.8 \
    --seed "${seed}" --device cuda
  echo "  → $(tail -1 ${outdir}/summary_metrics.csv)"
}

# β=0.01 — near-naïve PoE, weight barely changes
for seed in 42 17 99 123; do run 8 0.01 $seed; done

# β=0.1 — mild reweighting
for seed in 42 17 99 123; do run 8 0.1  $seed; done

# β=2.0 — stronger than current best
for seed in 42 17 99;     do run 8 2.0  $seed; done

# β=1.0, K=16 — more particles with current best β
for seed in 42 17 99;     do run 16 1.0  $seed; done

echo ""
echo "=== [GPU0] SUMMARY ==="
echo "Baseline (bridge_correction c=0.01): seam_mse_mean = 0.0185"
echo "Baseline (SMC K=8 β=1.0 s42):    seam_mse_mean = 0.0190 (old)"
echo ""
for dir in proposal_outputs/sweep_gpu0/smc_*; do
  [ -f "${dir}/summary_metrics.csv" ] && echo "[${dir##*/}]" && tail -1 "${dir}/summary_metrics.csv"
done
echo "=== DONE $(date) ==="
