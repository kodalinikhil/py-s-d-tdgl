#!/bin/bash
# Create or update the Python environment used by the Goncalves Hoffman2 job.
# Submit through submit_goncalves.uge.sh from the repository root.
#$ -cwd
#$ -j y
#$ -o goncalves_setup.$JOB_ID.log
#$ -l h_rt=2:00:00,h_data=4G
#$ -pe shared 1

set -euo pipefail

. /u/local/Modules/default/init/modules.sh
module load mamba

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
    echo "Reusing $ENV_PREFIX (set TDGL_REBUILD_ENV=1 to reinstall)."
fi

echo "Environment ready: $ENV_PREFIX"
