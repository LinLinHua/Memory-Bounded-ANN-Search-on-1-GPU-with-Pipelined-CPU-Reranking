#!/bin/bash
#SBATCH --job-name=hs_cuvs_baseline
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# HS: cuVS IVF-Flat, IVF-SQ, IVF-PQ at three code sizes (pq_dim d/2, d/4, d/8, each also with
# host refine) and CAGRA, with device memory,
# using src/cuvs_baseline.py and the shared cuvs environment (sbatch
# scripts/install_baselines.sbatch). LL's ll_day11_cuvs_a40.sbatch adds the Nsight profile.
# Submit from the project root:
#   sbatch scripts/cuvs_baseline_a40.sbatch                  # SIFT1M and DEEP10M
#   DATASETS=sift sbatch scripts/cuvs_baseline_a40.sbatch    # one dataset
# Other src/cuvs_baseline.py options can be passed as arguments, e.g. --nprobe 8,16,32,64,128.
# REFINE=0 skips the host-refine rows. ALGOS picks index types; CSV_TAG names the output results/cuvs_budget_<dataset><CSV_TAG>.csv.
# CUVS_SCRIPT runs another copy of src/cuvs_baseline.py (e.g. one not yet merged).
cd "$SLURM_SUBMIT_DIR"
source scripts/baseline_env.sh cuvs
set -euo pipefail

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
    refine=(--refine)
    [[ "${REFINE:-1}" == 0 ]] && refine=()
    echo "=== cuVS on $ds"
    python -u "${CUVS_SCRIPT:-src/cuvs_baseline.py}" "${args[@]}" "${refine[@]}" \
        --algo "${ALGOS:-ivf_flat,ivf_sq,ivf_pq,ivf_pq_d4,ivf_pq_d8,cagra}" \
        --csv "results/cuvs_budget_${ds}${CSV_TAG:-}.csv" "$@"
done
