#!/bin/bash
#SBATCH --job-name=hs_smoke_baselines
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# Check the Faiss and cuVS environments on an A40 with exact search on SIFT1M.
# Submit from the project root after scripts/install_baselines.sbatch:
#   sbatch scripts/smoke_baselines_a40.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

status=0
for lib in faiss cuvs; do
    echo "=== $lib"
    (
        source scripts/baseline_env.sh "$lib"
        OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -u scripts/smoke_baselines.py --lib "$lib"
    ) || status=1
done
exit $status
