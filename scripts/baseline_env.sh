# Activate a baseline environment created by scripts/install_baselines.sh:
#   source scripts/baseline_env.sh faiss
#   source scripts/baseline_env.sh cuvs
# The environments bring their own CUDA libraries, so do not also source env.sh.
_name="${1:?usage: source scripts/baseline_env.sh faiss|cuvs}"
_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load conda/25.11.0
eval "$(conda shell.bash hook)"
conda activate "${ENV_ROOT:-$_project/envs}/$_name"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
unset _name _project
