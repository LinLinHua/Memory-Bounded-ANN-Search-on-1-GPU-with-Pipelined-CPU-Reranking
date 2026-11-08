#!/bin/bash
#SBATCH --job-name=ll_deep10m
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL, DEEP10M on one A40 (needs data/deep10m from HS's prepare_deep10m):
#   1. ground truth: GPU brute force, top-100 for every query -> deep10m_groundtruth.ivecs
#   2. IVF index with nlist 4096
#   3. end-to-end check on DEEP10M: GPU scan -> CPU rerank -> Recall@10
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
INDEX_DIR="$DEEP_DIR/ivf4096"
BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results

hostname
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/gpu_bruteforce" "$BUILD/ivf_build" "$BUILD/bench_scan"

echo "=== DEEP10M ground truth (k=100)"
"$BUILD/gpu_bruteforce" --base "$DEEP_BASE" --max-base 10000000 --query "$DEEP_QUERY" \
    --k 100 --reps 1 --check 20 --out "$GT_OUT" --csv results/gpu_bruteforce_deep10m.csv

echo "=== DEEP10M IVF index (nlist 4096)"
"$BUILD/ivf_build" --base "$DEEP_BASE" --max-base 10000000 --out "$INDEX_DIR" --nlist 4096

echo "=== DEEP10M end to end"
"$BUILD/bench_scan" --base "$DEEP_BASE" --query "$DEEP_QUERY" --gt "$GT_OUT" --index "$INDEX_DIR" \
    --bits 8,4,1 --nprobe 16,32,64 --C 100,1000 --batch 128 --check 50 --reps 3 \
    --csv results/scan_deep10m.csv

rm -rf "$BUILD"
