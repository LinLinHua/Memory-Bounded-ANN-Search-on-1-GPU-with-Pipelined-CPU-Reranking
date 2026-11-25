#!/bin/bash
#SBATCH --job-name=hs_cuvs_rabitq
#SBATCH --partition=gpu
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# HS: cuVS IVF-RaBitQ (C++, src/cuvs_rabitq_bench.cu) at 1-4 bits per dimension, on the same
# data and nlist as our scan, for all queries at once and in batches of 1024 and 128.
# Needs the cuvs environment (sbatch scripts/install_baselines.sbatch). Submit from the root:
#   sbatch scripts/cuvs_rabitq_a40.sbatch
#   DATASETS=sift BATCHES="0" sbatch scripts/cuvs_rabitq_a40.sbatch --bits 1
cd "$SLURM_SUBMIT_DIR"
set -uo pipefail
module purge
module load gcc/13.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores OMP_PROC_BIND=close
export CUDA_MODULE_LOADING=EAGER  # kernels' code in the memory baseline, as for Faiss

BIN="build/${SLURM_JOB_ID}/cuvs_rabitq_bench"
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --nlist 1024)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --nlist 4096)

hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
bash scripts/build_cuvs_rabitq.sh "$BIN" 2>&1 | grep -E "error|built" || exit 1
mkdir -p results

status=0
for ds in ${DATASETS:-sift deep10m}; do
    case "$ds" in
        sift)    args=("${SIFT[@]}") ;;
        deep10m) args=("${DEEP[@]}") ;;
        *) echo "unknown dataset $ds" >&2; exit 1 ;;
    esac
    for batch in ${BATCHES:-0 1024 128}; do
        tag=""; [[ $batch != 0 ]] && tag="_b$batch"
        csv="results/cuvs_rabitq_${ds}${tag}.csv"
        rm -f "$csv"
        echo "=== IVF-RaBitQ on $ds, batch $batch"
        "$BIN" "${args[@]}" --bits 1,2,3,4 --nprobe 8,16,32,64,128 --C 100,300,1000 --reps 3 \
            --batch "$batch" --csv "$csv" "$@" || { echo "FAILED: $ds batch $batch"; status=1; }
    done
done
rm -rf "build/${SLURM_JOB_ID}"
exit $status
