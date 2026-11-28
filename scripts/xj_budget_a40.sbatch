#!/bin/bash
#SBATCH --job-name=xj_budget
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# XJ: the pipeline under the GPU memory budget. For budgets of 1/16 ... 1/2 of the raw FP32 data,
# which micro-batch sizes fit (index + scan work buffers), and the sequential and pipelined QPS of
# the ones that do. Settings that do not fit are recorded as over_budget.
# Uses the indexes in data/<dataset>/ivf* (scripts/ll_day12_fix_a40.sbatch builds them).
# Submit from the project root:
#   sbatch scripts/xj_budget_a40.sbatch
#   DATASETS=sift sbatch scripts/xj_budget_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail
export CUDA_MODULE_LOADING=EAGER  # kernels in the memory baseline (docs/measurement.md)

BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p results

SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --index data/sift/ivf1024)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --index data/deep10m/ivf4096)

hostname
make BUILD="$BUILD" -j8 "$BUILD/bench_pipeline"

status=0
for ds in ${DATASETS:-sift deep10m}; do
    case "$ds" in
        sift)    data=("${SIFT[@]}"); raw_mib=488;  topc1=select ;;  # 1M x 128 x 4 bytes
        deep10m) data=("${DEEP[@]}"); raw_mib=3662; topc1=fused ;;   # 10M x 96 x 4 bytes
        *) echo "unknown dataset $ds" >&2; exit 1 ;;
    esac
    for den in ${FRACTIONS:-16 8 4 2}; do
        budget=$((raw_mib / den))
        for cfg in "1 $topc1 1000" "4 fused 100" "8 fused 100"; do
            read -r bits topc C <<< "$cfg"
            echo "=== $ds: budget 1/$den = $budget MiB, $bits-bit $topc, C $C"
            "$BUILD/bench_pipeline" "${data[@]}" --label "$ds@1/$den" --bits "$bits" --topc "$topc" \
                --nprobe 64 --C "$C" --batch 16,64,128,256,1024 --reps 3 --budget-mb "$budget" \
                --csv results/xj_pipeline_budget.csv \
                || { echo "FAILED: $ds 1/$den $cfg"; status=1; }
        done
    done
done

rm -rf "$BUILD"
python3 scripts/xj_summary.py || true
exit $status
