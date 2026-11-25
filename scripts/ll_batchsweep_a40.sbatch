#!/bin/bash
#SBATCH --job-name=ll_batchsweep
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL: queries per batch for our scan, to compare with cuVS / Faiss, which search all 10,000 at once.
# Shipped code (ll.mk), fused top-C, 8-bit and 4-bit, nprobe 64, C 100. Submit from the project root:
#   sbatch scripts/ll_batchsweep_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail
BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results
hostname
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/bench_scan"
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --index data/sift/ivf1024)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --index data/deep10m/ivf4096)
SWEEP=(--bits 8,4 --nprobe 64 --C 100 --check 20 --reps 5 --topc fused)
for b in 64 128 256 512 1024 2048 10000; do
  echo "=== DEEP10M batch $b"
  "$BUILD/bench_scan" "${DEEP[@]}" "${SWEEP[@]}" --batch $b --csv "results/batch_deep10m_b$b.csv"
  echo "=== SIFT1M batch $b"
  "$BUILD/bench_scan" "${SIFT[@]}" "${SWEEP[@]}" --batch $b --csv "results/batch_sift_b$b.csv"
done
rm -rf "$BUILD"
