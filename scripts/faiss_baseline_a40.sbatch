#!/bin/bash
#SBATCH --job-name=hs_faiss_baseline
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# HS: Faiss baselines (CPU IVF-SQ, GPU IVF-Flat/SQ8/SQ4/PQ, IVF-PQ + IndexRefineFlat) on the
# same data, nlist and nprobe as our scan and LL's cuVS runs. Needs the faiss environment
# (sbatch scripts/install_baselines.sbatch). Submit from the project root:
#   sbatch scripts/faiss_baseline_a40.sbatch                  # SIFT1M and DEEP10M
#   DATASETS=sift sbatch scripts/faiss_baseline_a40.sbatch    # one dataset
# Other src/faiss_baseline.py options can be passed as arguments, e.g. --nprobe 8,16,32,64,128.
# ALGOS picks index types; CSV_TAG names the output results/faiss_<dataset><CSV_TAG>.csv, e.g.
#   ALGOS="gpu_ivf_pq gpu_ivf_sq8" CSV_TAG=_b128 sbatch scripts/faiss_baseline_a40.sbatch --batch 128
cd "$SLURM_SUBMIT_DIR"
source scripts/baseline_env.sh faiss
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Load every kernel when the CUDA context is created, so the memory baseline taken right after
# it already holds Faiss's and cuBLAS's code and gpu_*_mb count only allocations, as for cuVS
# (RMM statistics) and our scan (DeviceBudget). With lazy loading, the code of each kernel lands
# on the GPU at its first launch and was counted as index or search memory.
export CUDA_MODULE_LOADING=EAGER

SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --nlist 1024)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --nlist 4096)

hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
git rev-parse --short HEAD 2>/dev/null || true
mkdir -p results

for ds in ${DATASETS:-sift deep10m}; do
    case "$ds" in
        sift)    args=("${SIFT[@]}") ;;
        deep10m) args=("${DEEP[@]}") ;;
        *) echo "unknown dataset $ds" >&2; exit 1 ;;
    esac
    csv="results/faiss_${ds}${CSV_TAG:-}.csv"
    rm -f "$csv"
    # One process per index type: a Faiss assertion aborts the whole process, and this way it
    # only loses that index type. Each process appends its rows to the same CSV.
    for algo in ${ALGOS:-cpu_ivf_sq8 cpu_ivf_sq4 gpu_ivf_flat gpu_ivf_sq8 gpu_ivf_sq4 gpu_ivf_pq gpu_ivf_pq_refine gpu_ivf_pq_d4 gpu_ivf_pq_d4_refine gpu_ivf_pq_d8 gpu_ivf_pq_d8_refine}; do
        echo "=== Faiss $algo on $ds"
        python -u src/faiss_baseline.py "${args[@]}" --algo "$algo" --csv "$csv" "$@" \
            || { echo "FAILED: $algo on $ds"; status=1; }
    done
done
exit "${status:-0}"
