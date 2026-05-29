#!/bin/bash
# GPU 3 — 4-method comparison: diffcollage / naive / bridge_correction / proposal_smc
# Same 22 prompts, same settings — apple-to-apple comparison table.
# Usage: CUDA_VISIBLE_DEVICES=3 conda activate cbg_diffusion && bash _sweep_gpu3.sh
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

run_baselines() {
  local guidance=$1 seed=$2
  local outdir="proposal_outputs/sweep_gpu3/all4_g${guidance}_s${seed}"
  echo ""
  echo "=== [GPU3] diffcollage+naive+bridge_correction  guidance=${guidance} seed=${seed} ==="
  date
  python -m proposal_methods.evaluate_sd_bridge \
    --text-pairs "${TEXT_PAIRS[@]}" \
    --output-root "${outdir}" \
    --num-img 3 --overlap-latents 32 --n-steps 50 \
    --guidance-scale "${guidance}" \
    --proposal-couplings "0.01,0.05" \
    --seed "${seed}" --device cuda
  echo "  → $(cat ${outdir}/summary_metrics.csv)"
}

run_smc() {
  local K=$1 beta=$2 guidance=$3 seed=$4
  local outdir="proposal_outputs/sweep_gpu3/smc_K${K}_b${beta}_g${guidance}_s${seed}"
  echo ""
  echo "=== [GPU3] proposal_smc  K=${K} β=${beta} guidance=${guidance} seed=${seed} ==="
  date
  python -m proposal_methods.evaluate_sd_bridge \
    --text-pairs "${TEXT_PAIRS[@]}" \
    --output-root "${outdir}" \
    --num-img 3 --overlap-latents 32 --n-steps 50 \
    --guidance-scale "${guidance}" \
    --smc-only --smc-K 8 --smc-beta "${beta}" \
    --smc-resample-start 0.0 --smc-resample-end 0.8 \
    --seed "${seed}" --device cuda
  echo "  → $(tail -1 ${outdir}/summary_metrics.csv)"
}

# ── Round 1: fair comparison at guidance=3.0 (same as SMC best) ──────────────
run_baselines 3.0 42
run_smc 8 1.0 3.0 42

# ── Round 2: paper-like comparison at guidance=7.0 ───────────────────────────
run_baselines 7.0 42
run_smc 8 1.0 7.0 42

# ── Round 3: guidance=3.0, seed=17 (seed robustness check) ───────────────────
run_baselines 3.0 17
run_smc 8 1.0 3.0 17

# ── Round 4: guidance=5.0 (middle ground) ────────────────────────────────────
run_baselines 5.0 42
run_smc 8 1.0 5.0 42

echo ""
echo "=== [GPU3] FINAL COMPARISON TABLE ==="
echo ""
echo "method                             | seam_mse_mean | seam_mse_max"
echo "-----------------------------------|---------------|-------------"
for dir in proposal_outputs/sweep_gpu3/all4_g3.0_s42 proposal_outputs/sweep_gpu3/smc_K8_b1.0_g3.0_s42; do
  [ -f "${dir}/summary_metrics.csv" ] && grep -v "^method" "${dir}/summary_metrics.csv" | \
    awk -F, '{printf "%-35s| %-14s| %s\n", $1, $4, $5}'
done
echo ""
echo "=== DONE $(date) ==="
