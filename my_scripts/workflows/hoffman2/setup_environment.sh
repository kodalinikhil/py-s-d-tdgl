#!/bin/bash
# Create the Python environment used by the Hoffman2 batch job.
# Submit with qsub; dependency compilation should not run on a login node.
#$ -cwd
#$ -j y
#$ -o ddp_setup.$JOB_ID.log
#$ -l h_rt=2:00:00,h_data=4G
#$ -pe shared 1
set -euo pipefail

. /u/local/Modules/default/init/modules.sh
module load mamba

export OMP_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"
export LOKY_MAX_CPU_COUNT="${NSLOTS:-1}"
export NUMBA_NUM_THREADS="${NSLOTS:-1}"
export VECLIB_MAXIMUM_THREADS="${NSLOTS:-1}"
export BLIS_NUM_THREADS="${NSLOTS:-1}"

REPO_DIR="${TDGL_REPO_DIR:-${SGE_O_WORKDIR:-$(pwd)}}"
ENV_PREFIX="${TDGL_ENV_PREFIX:-$HOME/.conda/envs/py-s-d-tdgl}"
REBUILD_ENV="${TDGL_REBUILD_ENV:-0}"
INSTALL_REQUIRED=0

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    mamba create -y -p "$ENV_PREFIX" python=3.12 pip
    INSTALL_REQUIRED=1
fi

if ! "$ENV_PREFIX/bin/python" -c \
    'import h5py, matplotlib, numpy, scipy, tdgl' >/dev/null 2>&1; then
    INSTALL_REQUIRED=1
fi

if [[ "$REBUILD_ENV" == "1" ]] || [[ "$INSTALL_REQUIRED" == "1" ]]; then
    "$ENV_PREFIX/bin/python" -m pip install -e "$REPO_DIR"
else
    echo "Reusing the existing environment (set TDGL_REBUILD_ENV=1 to rebuild)."
fi

echo "Environment ready: $ENV_PREFIX"
