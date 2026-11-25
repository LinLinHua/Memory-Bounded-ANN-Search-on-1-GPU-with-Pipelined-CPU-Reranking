#!/bin/bash
#SBATCH --job-name=xj_pipeline
#SBATCH --partition=gpu
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# XJ, results 3, 4 and 5 on one A40, both datasets:
#   1. bench_rerank_breakdown : CPU vs GPU rerank on real 1-bit candidates, C 10 ... 1,000, split into
#                               ID transfer / gather or PCIe / compute / result return; OpenMP scaling
#   2. bench_pipeline         : sequential vs pipelined, batch 1 ... 1,024; P50/P99 per batch
#   3. serial_bruteforce      : serial baseline (DEEP10M on 1,000 queries), skipped if already in results/
# Uses the indexes in data/<dataset>/ivf* (scripts/ll_day12_fix_a40.sbatch builds them).
# Set --cpus-per-task to the node's physical core count from logs/a40_sysinfo-*.out; the thread
# list for the scaling run follows it. Submit from the project root:
#   sbatch scripts/xj_pipeline_a40.sbatch
#   DATASETS=sift STEPS="pipeline" sbatch scripts/xj_pipeline_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

BUILD="build/${SLURM_JOB_ID}"
CORES=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$CORES
THREADS_LIST=$(python3 -c "c=$CORES; t=[1]
while t[-1]*2 < c: t.append(t[-1]*2)
t.append(c); print(','.join(map(str, t)))")
mkdir -p results

SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs)

hostname
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|NUMA node\(s\)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "cores=$CORES threads_list=$THREADS_LIST"
make BUILD="$BUILD" -j8 "$BUILD/bench_pipeline" "$BUILD/bench_rerank_breakdown" "$BUILD/serial_bruteforce"

status=0
for ds in ${DATASETS:-sift deep10m}; do
    case "$ds" in
        # The best scan settings from docs/ll_kernels_and_analysis.md: 4-bit fused; 1-bit uses
        # select on SIFT1M and fused on DEEP10M.
        sift)    data=("${SIFT[@]}" --index data/sift/ivf1024);     topc1=select ;;
        deep10m) data=("${DEEP[@]}" --index data/deep10m/ivf4096);  topc1=fused ;;
        *) echo "unknown dataset $ds" >&2; exit 1 ;;
    esac

    for step in ${STEPS:-breakdown pipeline serial}; do
        case "$step" in
        breakdown)
            echo "=== $ds: rerank breakdown and OpenMP scaling (1-bit candidates)"
            "$BUILD/bench_rerank_breakdown" "${data[@]}" --label "$ds" --bits 1 --topc "$topc1" \
                --nprobe 64 --C 10,20,50,100,200,500,1000 --scaling-C 100,1000 \
                --threads-list "$THREADS_LIST" --prefetch-list 0,4 --reps 5 \
                --csv results/xj_rerank_breakdown.csv --scaling-csv results/xj_rerank_scaling.csv \
                || { echo "FAILED: $ds breakdown"; status=1; }
            ;;
        pipeline)
            # 4-bit: little rerank work (C 100); 1-bit: heavy rerank (C 1,000).
            for cfg in "4 fused 100" "1 $topc1 1000"; do
                read -r bits topc C <<< "$cfg"
                echo "=== $ds: pipeline, $bits-bit $topc, C $C"
                "$BUILD/bench_pipeline" "${data[@]}" --label "$ds" --bits "$bits" --topc "$topc" \
                    --nprobe 64 --C "$C" --batch 1,16,64,256,1024 --reps 5 \
                    --csv results/xj_pipeline.csv \
                    || { echo "FAILED: $ds pipeline $cfg"; status=1; }
            done
            ;;
        serial)
            csv="results/serial_${ds}_a40.csv"
            if [ -s "$csv" ]; then echo "=== $ds: serial baseline already in $csv"; continue; fi
            nq=10000; [ "$ds" = deep10m ] && nq=1000
            echo "=== $ds: serial baseline, $nq queries"
            data_only=("${data[@]:0:6}")  # --base, --query, --gt without --index
            "$BUILD/serial_bruteforce" "${data_only[@]}" --nq "$nq" --k 10 --csv "$csv" \
                || { echo "FAILED: $ds serial"; status=1; }
            ;;
        *) echo "unknown step $step" >&2; exit 1 ;;
        esac
    done
done

rm -rf "$BUILD"
python3 scripts/xj_summary.py || true
exit $status
