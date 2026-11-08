#!/bin/bash
#SBATCH --job-name=hs_deep_data
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Submit from the project root, after creating logs/.
# A copy of prepare_deep10m.py can be selected with DEEP_PREPARE_SCRIPT.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
umask 0002
hostname
python3 --version
python3 -u "${DEEP_PREPARE_SCRIPT:-scripts/prepare_deep10m.py}" \
    --output-dir "${DEEP_DIR:-data/deep10m}"
