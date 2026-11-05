#!/bin/bash
#SBATCH --job-name=serial_bf
#SBATCH --partition=main
#SBATCH --account=prasanna_933
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Submit from the project root:
#   sbatch scripts/run_serial_cpu.sbatch
#   sbatch --export=ALL,NQ=10000 scripts/run_serial_cpu.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/sift}"
NQ="${NQ:-100}"
CSV="${CSV:-results/serial_sift.csv}"
BIN="build/serial_bruteforce_${SLURM_JOB_ID}"

mkdir -p build results

hostname
lscpu | grep -E "Model name|^CPU\(s\)"
g++ --version | head -n 1

# Compile on the compute node because -march=native depends on its CPU.
g++ -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra src/serial_bruteforce.cpp -o "$BIN"

"$BIN" \
    --base "$DATA_DIR/sift_base.fvecs" \
    --query "$DATA_DIR/sift_query.fvecs" \
    --gt "$DATA_DIR/sift_groundtruth.ivecs" \
    --nq "$NQ" \
    --k 10 \
    --csv "$CSV"

rm -f "$BIN"
