#!/bin/bash
#SBATCH --job-name=ll_rqwarm
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# LL: does the throwaway warm-up change cuVS IVF-RaBitQ's speed? DEEP10M, all queries at once and
# batch 128, warm-up on / off / on, so run-to-run spread and the warm-up effect can be told apart.
cd "$SLURM_SUBMIT_DIR"
set -uo pipefail
export CUDA_MODULE_LOADING=EAGER OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
B="build/${SLURM_JOB_ID}"; OUT=results/rabitq_warmup; mkdir -p "$B" "$OUT"
bash scripts/build_cuvs_rabitq.sh "$B/cuvs_rabitq_bench" | tail -1
DATA=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin --gt data/deep10m/deep10m_groundtruth.ivecs)
i=0
for w in 1 0 1 0; do
  for b in 0 128; do
    i=$((i+1)); echo "=== run $i warm-up $w batch $b"
    "$B/cuvs_rabitq_bench" "${DATA[@]}" --nlist 4096 --bits 1,2 --nprobe 64 --C 100,1000 --reps 5 \
        --batch $b --transfer 1 --warmup $w --csv "$OUT/run${i}_w${w}_b${b}.csv" | grep -E 'nprobe|baseline'
  done
done
rm -rf "$B"
