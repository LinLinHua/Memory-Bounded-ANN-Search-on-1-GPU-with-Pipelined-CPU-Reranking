#!/bin/bash
#SBATCH --job-name=ll_compare
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# LL: our scan against the same methods in Faiss and cuVS (scalar quantization, RaBitQ, and FP32
# IVF-Flat as reference), all measured the same way (docs/measurement.md):
# nprobe 64, C 100 (1-bit / RaBitQ also C 1000), 10,000 queries timed from pinned host memory to
# candidates on the host, one warm-up + median of 5, at batch 128 and with all queries at once.
# One dataset per job (both fit in the 1-hour debug limit separately). Submit from the root:
#   DATASET=deep10m sbatch scripts/ll_compare_a40.sbatch
#   DATASET=sift    sbatch scripts/ll_compare_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
set -uo pipefail
DS="${DATASET:?set DATASET=sift or deep10m}"
OUT="results/compare_${DS}"
mkdir -p "$OUT" "build/${SLURM_JOB_ID}"
export CUDA_MODULE_LOADING=EAGER
case "$DS" in
  sift)    DATA=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
                 --gt data/sift/sift_groundtruth.ivecs); NLIST=1024; IDX=data/sift/ivf1024 ;;
  deep10m) DATA=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
                 --gt data/deep10m/deep10m_groundtruth.ivecs); NLIST=4096; IDX=data/deep10m/ivf4096 ;;
  *) echo "unknown DATASET $DS" >&2; exit 1 ;;
esac
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
status=0
# ours: batch 128 and all queries at once (bench_scan always times host -> host)
( source scripts/env.sh
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  make -f ll.mk BUILD="build/${SLURM_JOB_ID}" -j4 "build/${SLURM_JOB_ID}/bench_scan" >/dev/null
  for b in 128 10000; do
    echo "=== ours $DS batch $b"
    "build/${SLURM_JOB_ID}/bench_scan" "${DATA[@]}" --index "$IDX" --bits 8,4 --nprobe 64 --C 100 \
        --topc fused --batch $b --check 20 --reps 5 --csv "$OUT/ours_b$b.csv"
    # select keeps every estimate of the batch; with 10,000 queries that exceeds 2^31 entries
    topcs="fused select"; [[ $b == 10000 ]] && topcs="fused"
    for topc in $topcs; do
      "build/${SLURM_JOB_ID}/bench_scan" "${DATA[@]}" --index "$IDX" --bits 1 --nprobe 64 --C 100,1000 \
          --topc $topc --batch $b --check 20 --reps 5 --csv "$OUT/ours_1bit_${topc}_b$b.csv"
    done
  done ) || { echo "FAILED: ours"; status=1; }
# cuVS IVF-Flat (uncompressed reference); batch 0 = all queries in one call
( source scripts/baseline_env.sh cuvs
  for b in 128 0; do
    echo "=== cuVS $DS batch $b"
    python -u src/cuvs_baseline.py "${DATA[@]}" --nlist $NLIST --algo ivf_flat \
        --nprobe 64 --C 100 --reps 5 --batch $b --transfer 1 --csv "$OUT/cuvs_b$b.csv"
  done ) || { echo "FAILED: cuVS"; status=1; }
# cuVS IVF-RaBitQ at 1, 2 and 4 bits per dimension
( bash scripts/build_cuvs_rabitq.sh "build/${SLURM_JOB_ID}/cuvs_rabitq_bench" | tail -1
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  for b in 128 0; do
    echo "=== cuVS RaBitQ $DS batch $b"
    "build/${SLURM_JOB_ID}/cuvs_rabitq_bench" "${DATA[@]}" --nlist $NLIST --bits 1,2,4 --nprobe 64 \
        --C 100,1000 --reps 5 --batch $b --transfer 1 --csv "$OUT/rabitq_b$b.csv"
  done ) || { echo "FAILED: RaBitQ"; status=1; }
# Faiss GPU (its search takes host arrays, so upload and download are always timed)
( source scripts/baseline_env.sh faiss
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  for b in 128 0; do
    echo "=== Faiss $DS batch $b"
    rm -f "$OUT/faiss_b$b.csv"
    for algo in gpu_ivf_sq8 gpu_ivf_sq4 gpu_ivf_flat; do
      python -u src/faiss_baseline.py "${DATA[@]}" --nlist $NLIST --algo $algo --nprobe 64 --C 100 \
          --reps 5 --batch $b --csv "$OUT/faiss_b$b.csv" || echo "FAILED: faiss $algo"
    done
  done ) || { echo "FAILED: Faiss"; status=1; }
rm -rf "build/${SLURM_JOB_ID}"
exit $status
