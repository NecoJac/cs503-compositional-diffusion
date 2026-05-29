#!/bin/bash
set -euo pipefail

IMAGE="registry.rcp.epfl.ch/cvlab-bizquier/edm:latest"
REPO_DIR="/home/bizquier/edm"

CPU=1
MEM="16G"
GPU=1
JOB_NAME="test-edm-interactive"

echo "Starting interactive job: ${JOB_NAME}"

runai submit  --interactive \
  --name "${JOB_NAME}" \
  --image "${IMAGE}" \
  --cpu "${CPU}" \
  --memory "${MEM}" \
  --gpu "${GPU}" \
  --existing-pvc claimname=cvlab-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/bizquier

  runai submit -i registry.rcp.epfl.ch/cvlab-bizquier/edm:latest -g 1 --interactive