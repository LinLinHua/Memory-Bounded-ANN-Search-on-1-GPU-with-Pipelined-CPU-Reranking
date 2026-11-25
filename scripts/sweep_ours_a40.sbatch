#!/bin/bash
#SBATCH --job-name=hs_sweep_ours
#SBATCH --partition=gpu
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# HS: our scan (LL's bench_scan) over the settings the memory-budget comparison needs:
# 8/4/1-bit codes, nprobe 8-128, C 100/300/1000, and three batch sizes, because the scan's
# work memory grows with the batch (128 MB at 256 queries on SIFT1M, more than the 1-bit
# index). Both top-C versions run: fused needs less memory, select is faster for 1-bit SIFT1M.
# Uses the indexes in data/<dataset>/ivf* (scripts/ll_day12_fix_a40.sbatch builds them).
# Every run includes bench_scan's correctness checks. Submit from the project root:
#   sbatch scripts/sweep_ours_a40.sbatch
#   DATASETS=sift BATCHES="32 128" sbatch scripts/sweep_ours_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results

SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --index data/sift/ivf1024)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --index data/deep10m/ivf4096)
SWEEP=(--bits 8,4,1 --nprobe 8,16,32,64,128 --C 100,300,1000 --check 20 --reps 3)

hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/bench_scan"

status=0
for ds in ${DATASETS:-sift deep10m}; do
    case "$ds" in
        sift)    args=("${SIFT[@]}") ;;
        deep10m) args=("${DEEP[@]}") ;;
        *) echo "unknown dataset $ds" >&2; exit 1 ;;
    esac
    for batch in ${BATCHES:-32 128 512}; do
        for topc in ${TOPC:-fused select}; do
            echo "=== $ds batch $batch top-C $topc"
            "$BUILD/bench_scan" "${args[@]}" "${SWEEP[@]}" --batch "$batch" --topc "$topc" \
                --csv "results/sweep_ours_${ds}_b${batch}_${topc}.csv" "$@" \
                || { echo "FAILED: $ds batch $batch $topc"; status=1; }
        done
    done
done
rm -rf "$BUILD"
exit $status
