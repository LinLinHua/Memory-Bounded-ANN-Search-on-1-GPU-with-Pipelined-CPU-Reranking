#!/bin/bash
#SBATCH --job-name=ll_rotexp
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL: how much moving the 1-bit query rotation out of the scan kernel helped.
#   inscan : every (query, list) block rotates its residual itself (the pre-fix version)
#   pre    : R^T q once per query, R^T c once per list (current code)
# Same index, same GPU, both top-C methods. Submit from the project root:
#   sbatch scripts/ll_rotexp_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results "$BUILD"
hostname

NVCC_FLAGS=(-std=c++17 -O3 -arch=sm_86 -Xcompiler "-march=native -fopenmp -Wall")
SRCS=(src/bench_scan.cu src/gpu_distance.cu src/cpu_rerank.cpp src/gpu_rerank.cu)
nvcc "${NVCC_FLAGS[@]}" -Isrc "${SRCS[@]}" src/ivf_scan_rotexp.cu -lcublas -o "$BUILD/bench_pre"
nvcc "${NVCC_FLAGS[@]}" -Isrc -DROTATE_IN_SCAN "${SRCS[@]}" src/ivf_scan_rotexp.cu -lcublas -o "$BUILD/bench_inscan"

SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --index data/sift/ivf1024 --batch 256)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --index data/deep10m/ivf4096 --batch 128)
SWEEP=(--bits 1 --nprobe 64 --C 100,1000 --check 50 --reps 5)

for v in pre inscan; do
  for topc in fused select; do
    echo "=== SIFT1M 1-bit, rotation $v, top-C $topc"
    "$BUILD/bench_$v" "${SIFT[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/rotexp_sift_${v}_$topc.csv"
    echo "=== DEEP10M 1-bit, rotation $v, top-C $topc"
    "$BUILD/bench_$v" "${DEEP[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/rotexp_deep10m_${v}_$topc.csv"
  done
done
rm -rf "$BUILD"
