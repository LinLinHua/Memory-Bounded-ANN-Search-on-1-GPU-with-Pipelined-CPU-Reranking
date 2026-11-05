#!/bin/bash
#SBATCH --job-name=a40_sysinfo
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# One A40 job that records the hardware and runs the serial baseline, so the allocation is not wasted.
# Submit from the project root:
#   sbatch scripts/a40_sysinfo.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -uo pipefail

DATA_DIR="${DATA_DIR:-data/sift}"
NQ="${NQ:-10000}"
BIN_DIR="build/${SLURM_JOB_ID}"
mkdir -p "$BIN_DIR" results

echo "=== host"
hostname
echo "=== cpu"
lscpu
echo "=== allocated cores"
taskset -cp $$
echo "=== numa"
numactl -H 2>/dev/null || lscpu | grep -i numa
echo "=== gpu"
nvidia-smi
nvidia-smi topo -m
nvidia-smi -q | grep -A 12 "PCI$" | grep -E "Generation|Width|Max|Current"

echo "=== stream"
g++ -std=c++17 -O3 -march=native -fopenmp src/stream_triad.cpp -o "$BIN_DIR/stream_triad"
for t in 1 2 4 8 16; do
    OMP_NUM_THREADS=$t "$BIN_DIR/stream_triad" --mb 1024 --reps 10
done

echo "=== serial baseline"
g++ -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra src/serial_bruteforce.cpp \
    -o "$BIN_DIR/serial_bruteforce"
"$BIN_DIR/serial_bruteforce" \
    --base "$DATA_DIR/sift_base.fvecs" \
    --query "$DATA_DIR/sift_query.fvecs" \
    --gt "$DATA_DIR/sift_groundtruth.ivecs" \
    --nq "$NQ" \
    --k 10 \
    --csv results/serial_sift_a40.csv

rm -rf "$BIN_DIR"
