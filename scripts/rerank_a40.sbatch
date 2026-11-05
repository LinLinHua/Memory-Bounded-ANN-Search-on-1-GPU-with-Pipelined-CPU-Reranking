#!/bin/bash
#SBATCH --job-name=rerank_a40
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# CPU vs GPU zero-copy rerank on synthetic candidates, plus a budget check that must fail.
# Submit from the project root:
#   sbatch scripts/rerank_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/sift}"
NQ="${NQ:-10000}"
BUILD="build/${SLURM_JOB_ID}"
DATA_ARGS=(--base "$DATA_DIR/sift_base.fvecs" --query "$DATA_DIR/sift_query.fvecs"
           --gt "$DATA_DIR/sift_groundtruth.ivecs")
mkdir -p results

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
make BUILD="$BUILD" -j8 cpu gpu

echo "=== cpu vs gpu rerank"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK "$BUILD/bench_gpu_rerank" "${DATA_ARGS[@]}" --nq "$NQ" \
    --csv results/rerank_cpu_vs_gpu_sift.csv

echo "=== cpu rerank, prefetch off vs on"
for p in 0 4 8; do
    OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK "$BUILD/bench_cpu_rerank" "${DATA_ARGS[@]}" --nq "$NQ" \
        --prefetch "$p" --C 100,1000 --csv results/rerank_cpu_prefetch_sift.csv
done

echo "=== budget check (expected to fail: 1 MiB is below the candidate buffer)"
if "$BUILD/bench_gpu_rerank" "${DATA_ARGS[@]}" --nq "$NQ" --budget-mb 1 --C 100 --reps 1; then
    echo "budget check FAILED: allocation above budget was allowed"
else
    echo "budget check passed"
fi

rm -rf "$BUILD"
