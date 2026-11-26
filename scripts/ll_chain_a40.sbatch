#!/bin/bash
#SBATCH --job-name=ll_chain
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL: the optimization chain in one job, for one cumulative chart:
#   naive (1-byte loads, full sort) -> + 16-byte loads -> + radix select -> + fused
# 8-bit and 4-bit, nprobe 64, C 100, the usual batches (SIFT1M 256, DEEP10M 128). Submit from the root:
#   sbatch scripts/ll_chain_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail
BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results "$BUILD"
hostname
NVCC_FLAGS=(-std=c++17 -O3 -arch=sm_86 -Xcompiler "-march=native -fopenmp -Wall")
SRCS=(src/bench_scan.cu src/gpu_distance.cu src/cpu_rerank.cpp src/gpu_rerank.cu)
for v in 1 16; do
  nvcc "${NVCC_FLAGS[@]}" -Isrc -DLOAD_BYTES=$v "${SRCS[@]}" src/ivf_scan_vecexp.cu -lcublas -o "$BUILD/bench_v$v"
done
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --index data/sift/ivf1024 --batch 256)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --index data/deep10m/ivf4096 --batch 128)
SWEEP=(--bits 8,4 --nprobe 64 --C 100 --check 20 --reps 5)
for v in 1 16; do
  for topc in sort select fused; do
    echo "=== DEEP10M load $v B, top-C $topc"
    "$BUILD/bench_v$v" "${DEEP[@]}" "${SWEEP[@]}" --topc $topc --csv "results/chain_deep10m_v${v}_$topc.csv"
    echo "=== SIFT1M load $v B, top-C $topc"
    "$BUILD/bench_v$v" "${SIFT[@]}" "${SWEEP[@]}" --topc $topc --csv "results/chain_sift_v${v}_$topc.csv"
  done
done
rm -rf "$BUILD"
