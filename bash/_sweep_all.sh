#!/bin/bash
# Distributed sweep: launch _sweep_gpu{0,1,2,3}.sh in parallel on 4 GPUs.
# Usage: bash _sweep_all.sh
# Monitor: tail -f logs/gpu0.log
set -e

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
mkdir -p logs

CONDA_BASE=/home/yunchen/miniconda3

echo "=== Sweep started $(date) ==="
echo "Logs: ${REPO}/logs/gpu{0,1,2,3}.log"
echo ""

PIDS=()
for i in 0 1 2 3; do
  bash -c "
    source ${CONDA_BASE}/etc/profile.d/conda.sh
    conda activate cbg_diffusion
    export CUDA_VISIBLE_DEVICES=${i}
    bash ${REPO}/_sweep_gpu${i}.sh
  " >"${REPO}/logs/gpu${i}.log" 2>&1 &
  PIDS+=($!)
  echo "GPU ${i} launched (PID ${PIDS[-1]}), log: logs/gpu${i}.log"
done

# Wait for all 4 and report exit status per GPU
ALL_OK=1
for i in 0 1 2 3; do
  if wait "${PIDS[$i]}"; then
    echo "GPU ${i} DONE (exit 0)"
  else
    echo "GPU ${i} FAILED (exit $?)"
    ALL_OK=0
  fi
done

echo ""
echo "=== All GPUs finished $(date) ==="
[ "$ALL_OK" -eq 1 ] && echo "Status: all OK" || echo "Status: some GPU(s) failed — check logs/"

# ── Combined summary ──────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "COMBINED SUMMARY  (seam_mse_mean, lower is better)"
echo "  Baseline bridge_correction c=0.01 : 0.0185"
echo "  Baseline SMC K=8 β=1.0 s42    : 0.0190"
echo "========================================================================"

print_dir() {
  local dir=$1
  [ -f "${dir}/summary_metrics.csv" ] || return
  grep -v "^method" "${dir}/summary_metrics.csv" | \
    awk -F, -v name="${dir##*/}" \
      'NR==1 {printf "  %-52s seam_mean=%-10s seam_max=%s\n", name, $4, $5}'
}

echo ""
echo "--- GPU0: seed × β ---"
for dir in "${REPO}"/proposal_outputs/sweep_gpu0/smc_*; do print_dir "$dir"; done

echo ""
echo "--- GPU1: guidance × n_steps ---"
for dir in "${REPO}"/proposal_outputs/sweep_gpu1/smc_*; do print_dir "$dir"; done

echo ""
echo "--- GPU2: structural (overlap / num_img / resample / K) ---"
for dir in "${REPO}"/proposal_outputs/sweep_gpu2/smc_*; do print_dir "$dir"; done

echo ""
echo "--- GPU3: 4-method comparison ---"
for dir in "${REPO}"/proposal_outputs/sweep_gpu3/*/; do
  [ -f "${dir}/summary_metrics.csv" ] || continue
  echo "  [${dir##*/}]"
  grep -v "^method" "${dir}/summary_metrics.csv" | \
    awk -F, '{printf "    %-35s seam_mean=%-10s seam_max=%s\n", $1, $4, $5}'
done
echo "========================================================================"
