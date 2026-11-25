#!/bin/bash
#SBATCH --job-name=hs_correctness
#SBATCH --partition=gpu
#SBATCH --account=prasanna_933
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# HS: the proposal's correctness checks in one job.
#   1. Our GPU brute force against an exact CPU search (its --check), on SIFT1M and DEEP10M,
#      writing top-100 ground truth.
#   2. Faiss exact search against that output, the stored DEEP10M ground truth and the
#      published SIFT1M answers (scripts/check_correctness.py: ties allowed within 1e-4).
#   3. Component checks of our scan on both datasets (bench_scan: scan distances against a
#      CPU decode, top-C against a CPU sort, CPU against GPU rerank) for 8/4/1-bit codes.
# The pipelined-against-sequential check belongs to XJ's pipeline. Submit from the project root:
#   sbatch scripts/check_correctness_a40.sbatch
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
set -uo pipefail

BUILD="build/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
SIFT_B=data/sift/sift_base.fvecs;       SIFT_Q=data/sift/sift_query.fvecs
DEEP_B=data/deep10m/base.10M.fbin;      DEEP_Q=data/deep10m/query.public.10K.fbin

hostname
make -f ll.mk BUILD="$BUILD" -j4 "$BUILD/gpu_bruteforce" "$BUILD/bench_scan" || exit 1
status=0
fail() { echo "FAIL  $1"; status=1; }

echo "=== 1. GPU brute force against an exact CPU search (first 100 / 20 queries)"
"$BUILD/gpu_bruteforce" --base "$SIFT_B" --query "$SIFT_Q" --k 100 --reps 1 --check 100 \
    --out "$BUILD/sift_ours_gt100.ivecs" || fail "gpu_bruteforce SIFT1M"
"$BUILD/gpu_bruteforce" --base "$DEEP_B" --max-base 10000000 --query "$DEEP_Q" --k 100 --reps 1 \
    --check 20 --out "$BUILD/deep10m_ours_gt100.ivecs" || fail "gpu_bruteforce DEEP10M"

echo "=== 2. Faiss exact search against our brute force and the ground-truth files"
(
    source scripts/baseline_env.sh faiss
    python -u scripts/check_correctness.py --base "$SIFT_B" --query "$SIFT_Q" \
        --gt published=data/sift/sift_groundtruth.ivecs --gt ours="$BUILD/sift_ours_gt100.ivecs"
) || fail "Faiss checks SIFT1M"
(
    source scripts/baseline_env.sh faiss
    python -u scripts/check_correctness.py --base "$DEEP_B" --query "$DEEP_Q" \
        --gt stored=data/deep10m/deep10m_groundtruth.ivecs --gt ours="$BUILD/deep10m_ours_gt100.ivecs"
) || fail "Faiss checks DEEP10M"

echo "=== 3. Component checks of our scan (first 200 queries checked per setting)"
for ds in sift deep10m; do
    if [[ $ds == sift ]]; then
        args=(--base "$SIFT_B" --query "$SIFT_Q" --gt data/sift/sift_groundtruth.ivecs
              --index data/sift/ivf1024 --batch 256)
    else
        args=(--base "$DEEP_B" --query "$DEEP_Q" --gt data/deep10m/deep10m_groundtruth.ivecs
              --index data/deep10m/ivf4096 --batch 128)
    fi
    for topc in fused select; do
        out=$("$BUILD/bench_scan" "${args[@]}" --bits 8,4,1 --nprobe 16,64 --C 100,1000 \
            --check 200 --reps 1 --topc "$topc" 2>&1)
        rc=$?
        echo "$ds $topc: $(echo "$out" | grep -E "CHECKS|error" | tail -1)"
        [[ $rc -eq 0 ]] || { echo "$out" | tail -30; fail "bench_scan $ds $topc"; }
    done
done

rm -rf "$BUILD"
echo
[[ $status -eq 0 ]] && echo "CORRECTNESS: ALL PASSED" || echo "CORRECTNESS: SOME FAILED"
exit $status
