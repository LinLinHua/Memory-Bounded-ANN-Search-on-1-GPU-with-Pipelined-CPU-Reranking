#!/bin/bash
#SBATCH --job-name=ll_day11_topc
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL Day 11: new top-C (radix select) vs the Day 7 version (full CUB sort), same job and GPU.
# Runs the full end-to-end check for both on SIFT1M and DEEP10M; all correctness checks must pass.
# Needs data/sift/ivf1024 and data/deep10m/ivf4096 (built by the Day 7 and DEEP10M scripts).
# Submit from the project root:
#   sbatch scripts/ll_day11_topc_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results

SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --index data/sift/ivf1024 --batch 256)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --index data/deep10m/ivf4096 --batch 128)
SWEEP=(--bits 8,4,1 --nprobe 16,32,64 --C 100,1000 --check 50 --reps 3)

hostname
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/bench_scan"

for topc in select sort; do
    echo "=== SIFT1M, top-C $topc"
    "$BUILD/bench_scan" "${SIFT[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/topc_sift_$topc.csv"
    echo "=== DEEP10M, top-C $topc"
    "$BUILD/bench_scan" "${DEEP[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/topc_deep10m_$topc.csv"
done

rm -rf "$BUILD"
