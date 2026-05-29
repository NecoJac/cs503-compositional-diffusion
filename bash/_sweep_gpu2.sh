#!/bin/bash
# GPU 2 — structural sweep: overlap size, num_img, resample schedule
# Focus: does larger overlap region or more windows reduce seam MSE?
# Usage: CUDA_VISIBLE_DEVICES=2 conda activate cbg_diffusion && bash _sweep_gpu2.sh
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
  local K=$1 beta=$2 seed=$3 overlap=$4 nimg=$5 resend=$6 extra=${7:-}
  local label="smc_K${K}_b${beta}_s${seed}_ov${overlap}_ni${nimg}_re${resend}${extra}"
  local outdir="proposal_outputs/sweep_gpu2/${label}"
  echo ""
  echo "=== [GPU2] K=${K} β=${beta} seed=${seed} overlap=${overlap} num_img=${nimg} resample_end=${resend} ==="
  date
  python -m proposal_methods.evaluate_sd_bridge \
    --text-pairs "${TEXT_PAIRS[@]}" \
    --output-root "${outdir}" \
    --num-img "${nimg}" --overlap-latents "${overlap}" \
    --n-steps 50 --guidance-scale 3.0 \
    --smc-only --smc-K "${K}" --smc-beta "${beta}" \
    --smc-resample-start 0.0 --smc-resample-end "${resend}" \
    --seed "${seed}" --device cuda
  echo "  → $(tail -1 ${outdir}/summary_metrics.csv)"
}

# ── Larger overlap: 48 latents = 384px overlap (vs default 32=256px) ─────────
# Larger overlap shrinks the per-seam MSE window and gives the model more
# shared context at boundaries. Output width: 512 + 2×(64-48)×8 = 768px
for seed in 42 17 99; do run_smc 8 1.0 $seed 48 3 0.8; done
for seed in 42 17;    do run_smc 8 0.1 $seed 48 3 0.8; done

# ── num_img=5: 5-window bridge (wider panorama) ───────────────────────────────
# Output: 512 + 4×256 = 1536px. More internal seams but also more guidance signal.
for seed in 42 17; do run_smc 8 1.0 $seed 32 5 0.8; done
for seed in 42 17; do run_smc 8 0.1 $seed 32 5 0.8; done

# ── No resample (resample_end=0): pure FK weighting without particle collapse ─
# Hypothesis: resample + log_w reset may discard good particles too aggressively
for seed in 42 17 99; do run_smc 8 1.0 $seed 32 3 0.0 "_noresample"; done
for seed in 42 17;    do run_smc 8 0.1 $seed 32 3 0.0 "_noresample"; done

# ── K=32 with β=1.0: does more particles help at seed=42? ────────────────────
for seed in 42 17; do run_smc 32 1.0 $seed 32 3 0.8; done

echo ""
echo "=== [GPU2] SUMMARY ==="
echo "Baseline (bridge_correction c=0.01): seam_mse_mean = 0.0185"
for dir in proposal_outputs/sweep_gpu2/smc_*; do
  [ -f "${dir}/summary_metrics.csv" ] && echo "[${dir##*/}]" && tail -1 "${dir}/summary_metrics.csv"
done
echo "=== DONE $(date) ==="
