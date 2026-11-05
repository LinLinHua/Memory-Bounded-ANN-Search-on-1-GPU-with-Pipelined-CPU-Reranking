#!/bin/bash
#SBATCH --job-name=ll_deep10m
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL, DEEP10M on one A40 (run after HS has downloaded the data):
#   1. ground truth: GPU brute force, top-100 for every query -> deep10m_groundtruth.ivecs
#   2. IVF index with nlist 4096
# The base can be a 10M-row file, or the full DEEP1B file (only the first 10M rows are read).
# Submit from the project root:
#   sbatch scripts/ll_deep10m_a40.sbatch
#   DEEP_BASE=... DEEP_QUERY=... sbatch scripts/ll_deep10m_a40.sbatch   # other file names

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

DEEP_DIR="${DEEP_DIR:-data/deep10m}"
DEEP_BASE="${DEEP_BASE:-$DEEP_DIR/base.10M.fbin}"
DEEP_QUERY="${DEEP_QUERY:-$DEEP_DIR/query.public.10K.fbin}"
GT_OUT="$DEEP_DIR/deep10m_groundtruth.ivecs"
BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

hostname
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/gpu_bruteforce" "$BUILD/ivf_build"

echo "=== DEEP10M ground truth (k=100)"
"$BUILD/gpu_bruteforce" --base "$DEEP_BASE" --max-base 10000000 --query "$DEEP_QUERY" \
    --k 100 --reps 1 --check 20 --out "$GT_OUT"

echo "=== DEEP10M IVF index (nlist 4096)"
"$BUILD/ivf_build" --base "$DEEP_BASE" --max-base 10000000 --out "$DEEP_DIR/ivf4096" --nlist 4096

rm -rf "$BUILD"
