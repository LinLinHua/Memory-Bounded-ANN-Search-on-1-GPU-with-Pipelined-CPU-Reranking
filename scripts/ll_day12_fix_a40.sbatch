#!/bin/bash
#SBATCH --job-name=ll_day12_fix
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL Day 12: the two fixes from the Day 11 analysis, on the same GPU and index.
#   1. fused scan + per-list top-C (--topc fused) vs the Day 11 version (--topc select)
#   2. RaBitQ-style 1-bit estimate (needs factors1.f32, so both indexes are rebuilt first)
# Every run includes the full correctness checks.
# Submit from the project root:
#   sbatch scripts/ll_day12_fix_a40.sbatch

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
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/ivf_build" "$BUILD/bench_scan"

echo "=== rebuild indexes (adds factors1.f32)"
"$BUILD/ivf_build" --base data/sift/sift_base.fvecs --out data/sift/ivf1024 --nlist 1024 | tail -3
"$BUILD/ivf_build" --base data/deep10m/base.10M.fbin --max-base 10000000 --out data/deep10m/ivf4096 \
    --nlist 4096 | tail -3

for topc in fused select; do
    echo "=== SIFT1M, top-C $topc"
    "$BUILD/bench_scan" "${SIFT[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/fix_sift_$topc.csv"
    echo "=== DEEP10M, top-C $topc"
    "$BUILD/bench_scan" "${DEEP[@]}" "${SWEEP[@]}" --topc "$topc" --csv "results/fix_deep10m_$topc.csv"
done

echo "=== DEEP10M, 1-bit with more lists and candidates"
"$BUILD/bench_scan" "${DEEP[@]}" --bits 1 --nprobe 64,128 --C 100,300,1000 --check 20 --reps 3 \
    --topc fused --csv results/fix_deep10m_1bit.csv

rm -rf "$BUILD"
