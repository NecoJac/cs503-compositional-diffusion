#!/bin/bash
set -euo pipefail

IMAGE="registry.rcp.epfl.ch/cvlab-bizquier/edm:latest"
REPO_DIR="/home/bizquier/edm"
DNNLIB_CACHE_DIR="/scratch/cvlab/home/bizquier/edm/.cache/dnnlib"

CPU=1
MEM="16G"
GPU=1
JOB_NAME="diffcollage-edm"

echo "Submitting: ${JOB_NAME}"

runai submit \
  --name "${JOB_NAME}" \
  --image "${IMAGE}" \
  --cpu "${CPU}" \
  --memory "${MEM}" \
  --gpu "${GPU}" \
  --existing-pvc claimname=cvlab-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/bizquier \
  --command \
  -- /bin/bash -lc "
    set -e

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    export DNNLIB_CACHE_DIR=\"${DNNLIB_CACHE_DIR}\"
    mkdir -p \"${DNNLIB_CACHE_DIR}\"

    cd \"${REPO_DIR}\"
    pwd

    python -m pip install --user imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 einops

    python -u diffcollage_edm_fixed_end_gs.py

    "