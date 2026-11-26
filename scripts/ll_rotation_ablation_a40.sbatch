#!/bin/bash
#SBATCH --job-name=ll_rotabl
#SBATCH --partition=debug
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# LL: does the random rotation help the 1-bit codes? Builds the IVF index twice in this job,
# --rotation random and --rotation none (identity), same seed, then measures 1-bit recall@10
# after rerank for C = 10 .. 1000 (nprobe 64, batch 128, fused top-C). One dataset per job:
#   DATASET=sift    sbatch scripts/ll_rotation_ablation_a40.sbatch
#   DATASET=deep10m sbatch scripts/ll_rotation_ablation_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
set -euo pipefail
DS="${DATASET:?set DATASET=sift or deep10m}"
source scripts/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_MODULE_LOADING=EAGER
BUILD="build/${SLURM_JOB_ID}"; OUT="results/rotation_ablation_${DS}"; SCR="/scratch1/$USER/rotabl_${DS}"
mkdir -p "$BUILD" "$OUT" "$SCR"
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/ivf_build" "$BUILD/bench_scan" >/dev/null
case "$DS" in
  sift)    DATA=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
                 --gt data/sift/sift_groundtruth.ivecs); BUILDARGS=(--base data/sift/sift_base.fvecs --nlist 1024) ;;
  deep10m) DATA=(--base data/deep10m/base.10M.fbin --query data/deep10m/query.public.10K.fbin
                 --gt data/deep10m/deep10m_groundtruth.ivecs)
           BUILDARGS=(--base data/deep10m/base.10M.fbin --max-base 10000000 --nlist 4096) ;;
  *) echo "unknown DATASET $DS" >&2; exit 1 ;;
esac
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
for rot in random none; do
  echo "=== build, rotation $rot"
  "$BUILD/ivf_build" "${BUILDARGS[@]}" --out "$SCR/$rot" --rotation $rot | tail -2
done
echo "same coarse lists in both builds: $(cmp -s "$SCR/random/centroids.f32" "$SCR/none/centroids.f32" && echo yes || echo NO)"
for rot in random none; do
  echo "=== 1-bit, rotation $rot"
  "$BUILD/bench_scan" "${DATA[@]}" --index "$SCR/$rot" --bits 1 --nprobe 64 --C 10,100,300,1000 \
      --topc fused --batch 128 --check 20 --reps 3 --csv "$OUT/rot_$rot.csv"
done
rm -rf "$BUILD"
