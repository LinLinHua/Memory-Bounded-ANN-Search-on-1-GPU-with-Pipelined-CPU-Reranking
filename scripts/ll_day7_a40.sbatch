#!/bin/bash
#SBATCH --job-name=ll_day7
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL Day 4-7 on one A40:
#   1. build the SIFT1M IVF index on the GPU (k-means, nlist 1024) with 8/4/1-bit codes
#   2. Day 7 end-to-end check: GPU scan -> candidate IDs -> CPU rerank -> Recall@10,
#      plus scan-vs-CPU and CPU-vs-GPU rerank checks
# Submit from the project root:
#   sbatch scripts/ll_day7_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/sift}"
INDEX_DIR="${INDEX_DIR:-data/sift/ivf1024}"
BUILD="build/${SLURM_JOB_ID}"
SIFT=(--base "$DATA_DIR/sift_base.fvecs" --query "$DATA_DIR/sift_query.fvecs"
      --gt "$DATA_DIR/sift_groundtruth.ivecs")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results

hostname
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/ivf_build" "$BUILD/bench_scan"

echo "=== IVF build"
"$BUILD/ivf_build" --base "$DATA_DIR/sift_base.fvecs" --out "$INDEX_DIR" --nlist 1024

echo "=== Day 7 end to end"
"$BUILD/bench_scan" "${SIFT[@]}" --index "$INDEX_DIR" --bits 8,4,1 --nprobe 16,32,64 \
    --C 100,1000 --batch 256 --check 100 --reps 5 --csv results/scan_sift.csv

rm -rf "$BUILD"
