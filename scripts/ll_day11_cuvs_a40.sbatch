#!/bin/bash
#SBATCH --job-name=ll_day11_cuvs
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL Day 11-12: cuVS IVF-PQ and IVF-Flat on the same data, nlist and nprobe as our scan,
# plus Nsight Compute on cuVS's search kernels to compare bandwidth with ours (Day 8).
# Needs a Python environment with cuvs-cu12 and cupy-cuda12x (VENV, default ~/venvs/ll_cuvs):
#   python3 -m venv ~/venvs/ll_cuvs && ~/venvs/ll_cuvs/bin/pip install cuvs-cu12 cupy-cuda12x numpy \
#       --extra-index-url=https://pypi.nvidia.com
# Submit from the project root:
#   sbatch scripts/ll_day11_cuvs_a40.sbatch

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
source "${VENV:-$HOME/venvs/ll_cuvs}/bin/activate"
set -euo pipefail

OUT="results/ncu"
mkdir -p results "$OUT"
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --nlist 1024)
DEEP=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
      --gt data/deep10m/deep10m_groundtruth.ivecs --nlist 4096)

hostname
python -c "import cuvs; print('cuvs', cuvs.__version__)"

echo "=== cuVS on SIFT1M"
python src/cuvs_baseline.py "${SIFT[@]}" --algo ivf_pq,ivf_flat --csv results/cuvs_sift.csv
echo "=== cuVS on DEEP10M"
python src/cuvs_baseline.py "${DEEP[@]}" --algo ivf_pq,ivf_flat --csv results/cuvs_deep10m.csv

METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum
METRICS=$METRICS,dram__throughput.avg.pct_of_peak_sustained_elapsed
for algo in ivf_pq ivf_flat; do
    echo "=== Nsight: cuVS $algo search kernels, SIFT1M nprobe 64, C 100"
    ncu --nvtx --nvtx-include "search/" --metrics "$METRICS" --csv --page raw \
        --log-file "$OUT/cuvs_sift_${algo}_raw.csv" \
        python src/cuvs_baseline.py "${SIFT[@]}" --algo "$algo" --nprobe 64 --C 100 --reps 1 > /dev/null
    python3 scripts/ncu_summary.py "$OUT/cuvs_sift_${algo}_raw.csv" \
        --out "$OUT/cuvs_sift_${algo}_summary.csv"
done
