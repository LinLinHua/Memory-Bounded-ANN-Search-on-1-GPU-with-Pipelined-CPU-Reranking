#!/bin/bash
#SBATCH --job-name=ll_day8_ncu
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL Day 8: Nsight Compute on the SIFT1M search (nprobe 64, C 100) for 8-, 4- and 1-bit codes.
# Records time and DRAM traffic of every kernel, then prints the scan kernel's bandwidth as a
# share of the A40's 696 GB/s peak. Needs data/sift/ivf1024 (built by ll_day7_a40.sbatch).
# Submit from the project root:
#   sbatch scripts/ll_day8_ncu_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/sift}"
BUILD="build/${SLURM_JOB_ID}"
OUT="results/ncu"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$OUT"

hostname
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/bench_scan"

METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum
METRICS=$METRICS,dram__throughput.avg.pct_of_peak_sustained_elapsed

for bits in 8 4 1; do
    echo "=== $bits-bit"
    # 1,024 queries in 4 batches; --reps 1 means one warm-up search and one timed search.
    ncu --metrics "$METRICS" --csv --page raw --log-file "$OUT/sift_${bits}bit_raw.csv" \
        "$BUILD/bench_scan" --base "$DATA_DIR/sift_base.fvecs" --query "$DATA_DIR/sift_query.fvecs" \
        --gt "$DATA_DIR/sift_groundtruth.ivecs" --index "$DATA_DIR/ivf1024" \
        --bits "$bits" --nprobe 64 --C 100 --nq 1024 --batch 256 --check 0 --reps 1 > /dev/null
    python3 scripts/ncu_summary.py "$OUT/sift_${bits}bit_raw.csv" --out "$OUT/sift_${bits}bit_summary.csv"
done

echo "=== full-detail report of one 4-bit scan launch (open with ncu-ui)"
ncu --set full --kernel-name regex:scan_kernel --launch-skip 4 --launch-count 1 -f -o "$OUT/scan4_full" \
    "$BUILD/bench_scan" --base "$DATA_DIR/sift_base.fvecs" --query "$DATA_DIR/sift_query.fvecs" \
    --gt "$DATA_DIR/sift_groundtruth.ivecs" --index "$DATA_DIR/ivf1024" \
    --bits 4 --nprobe 64 --C 100 --nq 1024 --batch 256 --check 0 --reps 1 > /dev/null
ncu --import "$OUT/scan4_full.ncu-rep" --page details --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy | grep -vE "^\s*$|----" | head -60

rm -rf "$BUILD"
