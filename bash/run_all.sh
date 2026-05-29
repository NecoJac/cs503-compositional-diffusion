#!/bin/bash
# Run all three methods (diffcollage / naive / bridge_correction)
# for both image-condition and text-condition pipelines,
# then save comparison grids and metric CSVs.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── shared config ─────────────────────────────────────────────────────────────
DEVICE="cuda"
N_STEP=40
NUM_IMG=3
OVERLAP=32
GUIDANCE=1.5
SOLVER="heun"
COUPLING=0.25
SIGMA_DATA=0.5
IMPLICIT_SCALE=0.25
CORRECTION_CLIP=4.0
VIS_SCALE=4

# image-mode
IMAGE_ROOT="data/imagenet_landscapes"
IMAGE_INDICES="0,1,2"
CLASSES="lakeside:975,volcano:980,alp:970,coral_reef:973"

# text-mode
TEXT_PAIRS="lakeside:975+volcano:980,alp:970+coral_reef:973,lakeside:975+alp:970"
NUM_TEXT_SAMPLES=3

OUTPUT_IMAGE="proposal_outputs/eval_image"
OUTPUT_TEXT="proposal_outputs/eval_text"

# ── helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { printf '%.0s─' {1..60}; echo; }

COMMON_ARGS=(
    --device        "$DEVICE"
    --n-step        $N_STEP
    --num-img       $NUM_IMG
    --overlap-size  $OVERLAP
    --guidance-scale $GUIDANCE
    --solver        $SOLVER
    --coupling-strength  $COUPLING
    --sigma-data         $SIGMA_DATA
    --implicit-scale     $IMPLICIT_SCALE
    --correction-clip    $CORRECTION_CLIP
)

# ── image condition ───────────────────────────────────────────────────────────
hr
log "IMAGE CONDITION — three methods"
hr

python -m proposal_methods.evaluate_three_methods \
    --condition-type image \
    --classes        "$CLASSES" \
    --image-indices  "$IMAGE_INDICES" \
    --image-root     "$IMAGE_ROOT" \
    --output-root    "$OUTPUT_IMAGE" \
    --vis-scale      $VIS_SCALE \
    "${COMMON_ARGS[@]}"

log "Image results → $ROOT/$OUTPUT_IMAGE"

# ── text condition ────────────────────────────────────────────────────────────
hr
log "TEXT CONDITION — three methods"
hr

python -m proposal_methods.evaluate_three_methods \
    --condition-type    text \
    --text-pairs        "$TEXT_PAIRS" \
    --num-text-samples  $NUM_TEXT_SAMPLES \
    --output-root       "$OUTPUT_TEXT" \
    --vis-scale         $VIS_SCALE \
    "${COMMON_ARGS[@]}"

log "Text results → $ROOT/$OUTPUT_TEXT"

# ── final summary ─────────────────────────────────────────────────────────────
hr
log "ALL DONE"
hr
echo ""
echo "Output directories:"
echo "  Image : $ROOT/$OUTPUT_IMAGE"
echo "  Text  : $ROOT/$OUTPUT_TEXT"
echo ""
echo "Metric summary:"
for MODE in eval_image eval_text; do
    CSV="proposal_outputs/$MODE/summary_metrics.csv"
    if [ -f "$CSV" ]; then
        echo "  [$MODE]"
        column -t -s, "$CSV" | sed 's/^/    /'
        echo ""
    fi
done
