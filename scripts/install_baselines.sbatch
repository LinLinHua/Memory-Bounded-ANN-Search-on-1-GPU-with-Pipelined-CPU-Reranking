#!/bin/bash
#SBATCH --job-name=hs_install_baselines
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Create the Faiss and cuVS conda environments under envs/.
# Submit from the project root:
#   sbatch scripts/install_baselines.sbatch
# Extra arguments are passed through, e.g. sbatch scripts/install_baselines.sbatch cuvs
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
hostname
bash scripts/install_baselines.sh "$@"
