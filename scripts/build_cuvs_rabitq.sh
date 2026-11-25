#!/bin/bash
# Builds src/cuvs_rabitq_bench.cu against the cuvs conda environment (its nvcc, headers and
# libcuvs). Usage: bash scripts/build_cuvs_rabitq.sh [output path, default build/cuvs_rabitq_bench]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV_ROOT:-$PROJECT_DIR/envs}/cuvs"
OUT="${1:-$PROJECT_DIR/build/cuvs_rabitq_bench}"
mkdir -p "$(dirname "$OUT")"
"$ENV/bin/nvcc" -std=c++20 -O3 -arch=sm_86 --expt-extended-lambda --expt-relaxed-constexpr \
    -Xcompiler "-fopenmp -march=native" \
    -I"$ENV/include/rapids" -I"$ENV/include" -I"$ENV/targets/x86_64-linux/include" -I"$PROJECT_DIR/src" \
    "$PROJECT_DIR/src/cuvs_rabitq_bench.cu" -o "$OUT" \
    -L"$ENV/lib" -lcuvs -lrmm -lrapids_logger -lcudart -Xlinker -rpath="$ENV/lib"
echo "built $OUT"
