#!/bin/bash
#SBATCH --job-name=ll_cuvs_matched
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# LL: cuVS IVF-PQ and IVF-Flat timed the way bench_scan times our scan (same batch sizes,
# upload and download inside the timed loop), next to the original timing (all queries, already on
# the GPU). nprobe 64, C 100. Submit from the project root:
#   sbatch scripts/ll_cuvs_matched_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
source scripts/baseline_env.sh cuvs
set -euo pipefail
mkdir -p results
hostname
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --nlist 1024)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --nlist 4096)
COMMON=(--algo ivf_pq,ivf_flat --nprobe 64 --C 100 --reps 5)
for ds in deep10m sift; do
  [[ $ds == sift ]] && args=("${SIFT[@]}") || args=("${DEEP[@]}")
  for cfg in "0 0" "0 1" "128 1" "256 1" "1024 1"; do
    set -- $cfg
    echo "=== cuVS $ds batch $1 transfer $2"
    python -u src/cuvs_baseline.py "${args[@]}" "${COMMON[@]}" --batch "$1" --transfer "$2" \
        --csv "results/cuvs_matched_${ds}_b$1_t$2.csv"
  done
done
