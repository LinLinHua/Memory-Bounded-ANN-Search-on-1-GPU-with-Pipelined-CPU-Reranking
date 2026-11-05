#!/bin/bash
#SBATCH --job-name=ll_day1
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL Day 1-3 on one A40: bandwidth numbers and GPU brute force on SIFT1M.
# Submit from the project root:
#   sbatch scripts/ll_day1_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/sift}"
BUILD="build/${SLURM_JOB_ID}"
SIFT=(--base "$DATA_DIR/sift_base.fvecs" --query "$DATA_DIR/sift_query.fvecs"
      --gt "$DATA_DIR/sift_groundtruth.ivecs")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results

hostname
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/gpu_bruteforce" "$BUILD/bandwidth_bench"

echo "=== bandwidth (device memory, PCIe, zero-copy)"
"$BUILD/bandwidth_bench" --mb 1024 --reps 10 --csv results/bandwidth_a40.csv

echo "=== GPU brute force, k=10 (speedup reference vs. the serial baseline)"
"$BUILD/gpu_bruteforce" "${SIFT[@]}" --k 10 --reps 5 --check 100 \
    --csv results/gpu_bruteforce_sift.csv

echo "=== GPU brute force, k=100 (ground-truth mode: recall@100 vs. the official file)"
"$BUILD/gpu_bruteforce" "${SIFT[@]}" --k 100 --reps 1 --check 20 --out "$BUILD/sift_gt100.ivecs"

rm -rf "$BUILD"
