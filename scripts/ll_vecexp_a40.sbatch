#!/bin/bash
#SBATCH --job-name=ll_vecexp
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL: how much the 16-byte code loads in the scan help (vectorization).
#   v1 / v4 / v16 : each thread reads its code 1, 4 or 16 bytes per load (v16 = shipped code)
# 8-bit and 4-bit, SIFT1M and DEEP10M, both top-C methods. Submit from the project root:
#   sbatch scripts/ll_vecexp_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail
BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results "$BUILD"
hostname
NVCC_FLAGS=(-std=c++17 -O3 -arch=sm_86 -Xcompiler "-march=native -fopenmp -Wall")
SRCS=(src/bench_scan.cu src/gpu_distance.cu src/cpu_rerank.cpp src/gpu_rerank.cu)
for v in 1 4 16; do
  nvcc "${NVCC_FLAGS[@]}" -Isrc -DLOAD_BYTES=$v "${SRCS[@]}" src/ivf_scan_vecexp.cu -lcublas -o "$BUILD/bench_v$v"
  echo "=== load width check, v$v (global loads in the SASS)"
  cuobjdump -sass "$BUILD/bench_v$v" | grep -oE "LDG\.E[.A-Z0-9]*" | sort | uniq -c
done
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --index data/sift/ivf1024 --batch 256)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --index data/deep10m/ivf4096 --batch 128)
SWEEP=(--bits 8,4 --nprobe 64 --C 100 --check 50 --reps 5)
for v in 1 4 16; do
  for topc in select fused; do
    echo "=== SIFT1M, load v$v, top-C $topc"
    "$BUILD/bench_v$v" "${SIFT[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/vecexp_sift_v${v}_$topc.csv"
    echo "=== DEEP10M, load v$v, top-C $topc"
    "$BUILD/bench_v$v" "${DEEP[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/vecexp_deep10m_v${v}_$topc.csv"
  done
done
rm -rf "$BUILD"
