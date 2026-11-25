#!/bin/bash
# Create the two conda environments used for the Faiss and cuVS baselines.
#
#   bash scripts/install_baselines.sh            # both environments
#   bash scripts/install_baselines.sh faiss      # only one
#   FORCE=1 bash scripts/install_baselines.sh    # delete and recreate
#
# Run it through scripts/install_baselines.sbatch rather than on the login node.
#
# Two environments instead of one: Faiss 1.14.1 (the last CUDA 12 build) needs
# libcuvs 25.10, which predates the cuVS IVF-RaBitQ index we want to compare
# against, so cuVS gets its own environment at 26.10.
#
# Both use CUDA 12 builds: the A40 nodes run driver 580 (CUDA 13.0), and the
# newer Faiss packages are built for CUDA 13.2. CUDA_OVERRIDE tells the solver
# which driver the GPU nodes have, since the job may run on a CPU node.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-$PROJECT_DIR/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch1/$USER/conda-pkgs}"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}"
umask 0002  # teammates in prasanna_933 can use the environments

# Faiss 1.14.1 links libmkl_*.so.2; MKL 2026 on conda-forge ships .so.3 only.
FAISS_SPECS=(python=3.12 "faiss-gpu=1.14.1" "cuda-version=12.6" "mkl>=2024.2.2,<2026" "numpy<2.3" pandas)
CUVS_SPECS=(python=3.12 "cuvs=26.10" "cuda-version=12.9" cupy numpy pandas)

module load conda/25.11.0
mkdir -p "$ENV_ROOT" "$CONDA_PKGS_DIRS"

# --override-channels keeps Anaconda's defaults channel (and its terms of
# service) out of the solve; every package comes from these three channels.
create_env() {
    local name="$1"; shift
    local prefix="$ENV_ROOT/$name"
    if [[ -d "$prefix" ]]; then
        if [[ "${FORCE:-0}" == 1 ]]; then
            echo "=== removing $prefix"
            conda env remove -y -p "$prefix"
        else
            echo "=== $prefix exists, skipping (FORCE=1 to recreate)"
            return
        fi
    fi
    echo "=== creating $prefix: $*"
    conda create -y -q -p "$prefix" --override-channels \
        -c pytorch -c rapidsai -c conda-forge "$@"
    # Exact package list, so the environment can be rebuilt and cited in the report.
    conda list -p "$prefix" --explicit > "$ENV_ROOT/$name.lock.txt"
    echo "=== wrote $ENV_ROOT/$name.lock.txt"
}

[[ $# -eq 0 ]] && set -- faiss cuvs
for t in "$@"; do
    case "$t" in
        faiss) create_env faiss "${FAISS_SPECS[@]}" ;;
        cuvs)  create_env cuvs  "${CUVS_SPECS[@]}" ;;
        *) echo "unknown environment: $t (expected faiss or cuvs)" >&2; exit 1 ;;
    esac
done

du -sh "$ENV_ROOT"/*/ 2>/dev/null || true
echo "=== done. Activate with: source scripts/baseline_env.sh faiss|cuvs"
