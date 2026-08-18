#!/bin/bash
# Submit after all array tasks finish. Set TDGL_RESULT_ROOT to the run directory.
#$ -cwd
#$ -j y
#$ -o ddp_plot.$JOB_ID.log
#$ -l h_rt=1:00:00,h_data=4G
#$ -pe shared 1

set -euo pipefail

. /u/local/Modules/default/init/modules.sh
module load mamba

REPO_DIR="${TDGL_REPO_DIR:-$SGE_O_WORKDIR}"
ENV_PREFIX="${TDGL_ENV_PREFIX:-$HOME/.conda/envs/py-s-d-tdgl}"
RESULT_ROOT="${TDGL_RESULT_ROOT:?Set TDGL_RESULT_ROOT to the completed run directory}"
EXPECTED_RUN_ID="${TDGL_RUN_ID:-}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:?}/matplotlib"
export OMP_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"
export LOKY_MAX_CPU_COUNT="${NSLOTS:-1}"
export NUMBA_NUM_THREADS="${NSLOTS:-1}"
export VECLIB_MAXIMUM_THREADS="${NSLOTS:-1}"
export BLIS_NUM_THREADS="${NSLOTS:-1}"
mkdir -p "$MPLCONFIGDIR"

set --
for alpha in 0.7 0.8 0.9; do
    alpha_dir="$RESULT_ROOT/alpha_$alpha"
    marker="$alpha_dir/_SUCCESS"
    if [[ ! -s "$marker" ]] ||
        ! grep -Fxq "status=complete" "$marker" ||
        ! grep -Fxq "alpha=$alpha" "$marker" ||
        [[ ! -s "$alpha_dir/measurements.csv" ]] ||
        [[ ! -s "$alpha_dir/transitions.csv" ]] ||
        [[ -z "$(find "$alpha_dir/h5" -type f -name '*.h5' -print -quit 2>/dev/null)" ]]; then
        echo "Refusing to plot: alpha=$alpha is missing a valid _SUCCESS manifest or result CSV." >&2
        exit 2
    fi
    if [[ -n "$EXPECTED_RUN_ID" ]] &&
        ! grep -Fxq "run_id=$EXPECTED_RUN_ID" "$marker"; then
        echo "Refusing to plot: alpha=$alpha belongs to a different run ID." >&2
        exit 2
    fi
    RECORDED_ROWS="$(sed -n 's/^measurement_rows=//p' "$marker")"
    RECORDED_H5="$(sed -n 's/^h5_files=//p' "$marker")"
    ACTUAL_ROWS="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$alpha_dir/measurements.csv")"
    ACTUAL_H5="$(find "$alpha_dir/h5" -type f -name '*.h5' | wc -l | tr -d ' ')"
    if [[ ! "$RECORDED_ROWS" =~ ^[1-9][0-9]*$ ]] ||
        [[ ! "$RECORDED_H5" =~ ^[1-9][0-9]*$ ]] ||
        [[ "$RECORDED_ROWS" != "$ACTUAL_ROWS" ]] ||
        [[ "$RECORDED_H5" != "$ACTUAL_H5" ]]; then
        echo "Refusing to plot: alpha=$alpha result counts do not match its manifest." >&2
        exit 2
    fi
    set -- "$@" "$alpha_dir/measurements.csv"
done

mkdir -p "$RESULT_ROOT/combined"

cd "$REPO_DIR"
"$ENV_PREFIX/bin/python" my_scripts/plots/plot_d_plus_d_prime_phase_scan.py \
    "$@" \
    --output-directory "$RESULT_ROOT/combined"
