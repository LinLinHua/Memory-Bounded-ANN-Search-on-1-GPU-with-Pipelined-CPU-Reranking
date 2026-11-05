# Source this file on CARC before building or running:
#   source scripts/env.sh
module purge
module load gcc/13.3.0 cuda/12.6.3

export OMP_PLACES=cores
export OMP_PROC_BIND=close
