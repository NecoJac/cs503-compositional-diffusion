#!/usr/bin/env bash
# Sweep over diverse text-pairs for evaluate_sd_bridge.
# Each pair gets its own output directory derived from the pair content.
# Usage: bash run_text_sweep.sh [device]

set -euo pipefail

DEVICE="${1:-cuda}"

# ── Text pairs (left:right) ────────────────────────────────────────────────────
# Format: "left prompt:right prompt"
# Organised by theme for easy browsing.

PAIRS=(
    # Nature ↔ Nature
    "a calm lakeside landscape:a volcanic eruption"
    "a lush rainforest:a barren desert"
    "a snowy mountain peak:a sandy tropical beach"
    "a misty autumn forest:a bright spring meadow"
    "a frozen arctic tundra:a humid jungle"

    # Sky / Weather
    "a clear blue sky at noon:a dark stormy sky with lightning"
    "a golden sunset over the ocean:a starry night sky"
    "a foggy morning valley:a bright sunny afternoon"
    "a heavy blizzard:a warm summer day"
    "a rainbow over green hills:a tornado over a flat plain"

    # Day ↔ Night / Urban
    "a quiet rural countryside:a busy neon city at night"
    "a sunlit cobblestone village:an empty midnight street"
    "a bustling daytime market:a silent moonlit plaza"
    "a mountain dawn:an urban dusk with skyscrapers"

    # Seasons
    "a cherry blossom park in spring:a snow covered park in winter"
    "a golden wheat field in summer:a bare muddy field in autumn"
    "a colorful autumn maple forest:a green summer forest"
    "a frozen lake in winter:a blooming flower meadow in spring"

    # Abstract contrasts
    "a peaceful monastery garden:a chaotic urban construction site"
    "an underwater coral reef:an outer space nebula"
    "a medieval castle on a hill:a futuristic glass skyscraper"
    "a quiet library interior:a loud rock concert stage"
)

# ── Shared generation settings ─────────────────────────────────────────────────
N_STEPS=50
GUIDANCE_SCALE=3.0
NUM_IMG=3
OVERLAP_LATENTS=32
PROPOSAL_COUPLINGS=0.01
CORRECTION_CLIP=0.25

OUTPUT_BASE="proposal_outputs/sweep_text"

# ── Loop ──────────────────────────────────────────────────────────────────────
TOTAL=${#PAIRS[@]}
echo "Running ${TOTAL} text-pair experiments on device=${DEVICE}"
echo "========================================================"

for i in "${!PAIRS[@]}"; do
    PAIR="${PAIRS[$i]}"
    IDX=$(( i + 1 ))

    # Derive a short directory name from the pair
    SLUG=$(echo "${PAIR}" | tr ':' '__' | tr ' ' '_' | tr -cd 'A-Za-z0-9_' | cut -c1-60)
    OUTPUT_ROOT="${OUTPUT_BASE}/${IDX:02}__${SLUG}"

    echo ""
    echo "[${IDX}/${TOTAL}] Pair: ${PAIR}"
    echo "          Output: ${OUTPUT_ROOT}"

    python -m proposal_methods.evaluate_sd_bridge \
        --text-pairs "${PAIR}" \
        --output-root "${OUTPUT_ROOT}" \
        --num-img "${NUM_IMG}" \
        --overlap-latents "${OVERLAP_LATENTS}" \
        --n-steps "${N_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --proposal-couplings "${PROPOSAL_COUPLINGS}" \
        --correction-clip "${CORRECTION_CLIP}" \
        --device "${DEVICE}"

    echo "    [done] ${IDX}/${TOTAL}"
done

echo ""
echo "========================================================"
echo "All ${TOTAL} experiments finished. Results in: ${OUTPUT_BASE}/"
