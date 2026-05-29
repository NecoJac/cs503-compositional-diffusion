#!/bin/bash
# GPU 1 — guidance × n_steps sweep
# Focus: does lower guidance compress seam MSE further? does more steps help?
# Usage: CUDA_VISIBLE_DEVICES=1 conda activate cbg_diffusion && bash _sweep_gpu1.sh
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

run_smc() {
  local K=$1 beta=$2 guidance=$3 seed=$4 nsteps=${5:-50}
  local label="smc_K${K}_b${beta}_g${guidance}_s${seed}_t${nsteps}"
  local outdir="proposal_outputs/sweep_gpu1/${label}"
  echo ""
  echo "=== [GPU1] K=${K} β=${beta} guidance=${guidance} seed=${seed} nsteps=${nsteps} ==="
  date
  python -m proposal_methods.evaluate_sd_bridge \
    --text-pairs "${TEXT_PAIRS[@]}" \
    --output-root "${outdir}" \
    --num-img 3 --overlap-latents 32 --n-steps "${nsteps}" --guidance-scale "${guidance}" \
    --smc-only --smc-K "${K}" --smc-beta "${beta}" \
    --smc-resample-start 0.0 --smc-resample-end 0.8 \
    --seed "${seed}" --device cuda
  echo "  → $(tail -1 ${outdir}/summary_metrics.csv)"
}

# ── Guidance sweep (K=8 β=1.0 seed=42, the "best" config) ──────────────────
for g in 1.5 2.0 5.0 7.0; do run_smc 8 1.0 $g 42 50; done
for g in 1.5 2.0 5.0 7.0; do run_smc 8 1.0 $g 17 50; done

# ── Guidance sweep with β=0.1 (mild reweighting, often performs well) ────────
for g in 1.5 2.0 5.0; do run_smc 8 0.1 $g 42 50; done
for g in 1.5 2.0 5.0; do run_smc 8 0.1 $g 17 50; done

# ── n_steps sweep (seed=42, best guidance=3.0, β=1.0) ────────────────────────
for t in 30 80 100; do run_smc 8 1.0 3.0 42 $t; done

echo ""
echo "=== [GPU1] SUMMARY ==="
echo "Baseline (bridge_correction c=0.01): seam_mse_mean = 0.0185"
for dir in proposal_outputs/sweep_gpu1/smc_*; do
  [ -f "${dir}/summary_metrics.csv" ] && echo "[${dir##*/}]" && tail -1 "${dir}/summary_metrics.csv"
done
echo "=== DONE $(date) ==="
