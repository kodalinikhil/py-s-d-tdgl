#!/bin/bash
# Hoffman2 Goncalves up/down field-continuation job (Altair Grid Engine).
# Mesh spacing: h=0.1 xi. Field interval: Delta(H/Hc2)=0.1.
# Submit through submit_goncalves.uge.sh from the repository root.
#$ -cwd
#$ -j y
#$ -o goncalves.$JOB_ID.log
#$ -l h_rt=23:30:00,h_data=16G
#$ -pe shared 1
#$ -m a
#$ -notify

set -euo pipefail

. /u/local/Modules/default/init/modules.sh
module load mamba

REPO_DIR="${TDGL_REPO_DIR:-$SGE_O_WORKDIR}"
ENV_PREFIX="${TDGL_ENV_PREFIX:-$HOME/.conda/envs/py-s-d-tdgl}"
RUN_ID="${TDGL_RUN_ID:-goncalves_${JOB_ID}}"
RESULT_ROOT="${TDGL_RESULT_ROOT:-$REPO_DIR/results/hoffman2/$RUN_ID}"
RESUME="${TDGL_RESUME:-0}"
EXPECTED_FIELDS=51
RUN_SIGNATURE="v1|mesh_spacing=0.1|field_step=0.1|maximum_field=2.5|solve_time=1500|up_down=1"

if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "TDGL_RESUME must be 0 or 1, not: $RESUME" >&2
    exit 2
fi
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Missing environment: $ENV_PREFIX" >&2
    echo "Run setup_goncalves_environment.uge.sh first." >&2
    exit 2
fi
if ! "$ENV_PREFIX/bin/python" -c \
    'import h5py, matplotlib, meshpy, numba, numpy, scipy, shapely, tdgl' \
    >/dev/null 2>&1; then
    echo "The Hoffman Python environment is incomplete: $ENV_PREFIX" >&2
    echo "Resubmit with TDGL_REBUILD_ENV=1." >&2
    exit 2
fi

SCRATCH_BASE="${SCRATCH:?SCRATCH is not set}/py-s-d-tdgl/$RUN_ID"
WORK_OUTPUT="$SCRATCH_BASE/goncalves"
SUCCESS_MARKER="$RESULT_ROOT/_SUCCESS"
SUCCESS_TMP="$RESULT_ROOT/.SUCCESS.${JOB_ID}.tmp"

directory_has_files() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -print -quit)" ]]
}

if [[ "$RESUME" == "0" ]] && {
    directory_has_files "$WORK_OUTPUT" || directory_has_files "$RESULT_ROOT";
}; then
    echo "Refusing to mix a fresh run with existing output." >&2
    echo "Choose a new TDGL_RUN_ID, or set TDGL_RESUME=1." >&2
    exit 2
fi
if [[ "$RESUME" == "1" && -f "$SUCCESS_MARKER" ]]; then
    if grep -Fxq "status=complete" "$SUCCESS_MARKER" &&
        grep -Fxq "run_id=$RUN_ID" "$SUCCESS_MARKER" &&
        grep -Fxq "run_signature=$RUN_SIGNATURE" "$SUCCESS_MARKER"; then
        echo "$RUN_ID is already complete."
        exit 0
    fi
    echo "Refusing an invalid or stale success marker: $SUCCESS_MARKER" >&2
    exit 2
fi

mkdir -p "$WORK_OUTPUT" "$RESULT_ROOT"
if [[ "$RESUME" == "1" ]]; then
    rsync -a --update --exclude=_SUCCESS "$RESULT_ROOT/" "$WORK_OUTPUT/"
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="$SCRATCH_BASE/matplotlib"
export NUMBA_CACHE_DIR="$SCRATCH_BASE/numba"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

sync_results() {
    rsync -a --partial --exclude=_SUCCESS "$WORK_OUTPUT/" "$RESULT_ROOT/"
}

sync_results_with_checksums() {
    rsync -a --checksum --partial --exclude=_SUCCESS \
        "$WORK_OUTPUT/" "$RESULT_ROOT/"
}

SIMULATION_PID=""
SIGNAL_NAME=""
SUCCESS_WRITTEN=0

handle_signal() {
    SIGNAL_NAME="$1"
    echo "Received $SIGNAL_NAME; requesting a resumable solver checkpoint." >&2
    if [[ -n "$SIMULATION_PID" ]]; then
        kill -INT "$SIMULATION_PID" 2>/dev/null || true
    fi
}

cleanup() {
    local status=$?
    trap - EXIT TERM INT USR1 USR2
    if [[ -n "$SIMULATION_PID" ]]; then
        kill -INT "$SIMULATION_PID" 2>/dev/null || true
        wait "$SIMULATION_PID" 2>/dev/null || true
        SIMULATION_PID=""
    fi
    rm -f "$SUCCESS_TMP"
    if [[ "$SUCCESS_WRITTEN" != "1" ]] && [[ -d "$WORK_OUTPUT" ]]; then
        echo "Syncing resumable output to $RESULT_ROOT"
        if ! sync_results; then
            echo "Warning: partial-result sync failed." >&2
            status=1
        fi
    fi
    exit "$status"
}

trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap 'handle_signal USR1' USR1
trap 'handle_signal USR2' USR2
trap cleanup EXIT

echo "Job $JOB_ID on $(hostname -s)"
echo "run_id=$RUN_ID"
echo "mesh_spacing=0.1"
echo "field_protocol=0.0:0.1:2.5:0.1:0.0 ($EXPECTED_FIELDS points)"
echo "scratch=$WORK_OUTPUT"
echo "results=$RESULT_ROOT"
date

cd "$REPO_DIR"
"$ENV_PREFIX/bin/python" my_scripts/simulations/simulate_goncalves.py \
    --mesh-spacing 0.1 \
    --field-step 0.1 \
    --maximum-field 2.5 \
    --solve-time 1500 \
    --output-directory "$WORK_OUTPUT" &
SIMULATION_PID=$!

set +e
wait "$SIMULATION_PID"
SIMULATION_STATUS=$?
set -e

if [[ -n "$SIGNAL_NAME" ]]; then
    wait "$SIMULATION_PID" 2>/dev/null || true
    SIMULATION_PID=""
    echo "Run interrupted by $SIGNAL_NAME; resubmit with TDGL_RESUME=1." >&2
    exit 75
fi
SIMULATION_PID=""
if [[ "$SIMULATION_STATUS" -ne 0 ]]; then
    echo "Simulation stopped with status $SIMULATION_STATUS." >&2
    exit "$SIMULATION_STATUS"
fi

CHECKPOINT_COUNT=$(find "$WORK_OUTPUT" -type f -name 'goncalves_Ha_*.h5' | wc -l | tr -d ' ')
if [[ "$CHECKPOINT_COUNT" -lt "$EXPECTED_FIELDS" ]] ||
    [[ ! -s "$WORK_OUTPUT/goncalves_trapped_flux.png" ]]; then
    echo "Completion verification failed; _SUCCESS will not be written." >&2
    exit 1
fi

sync_results_with_checksums
{
    printf '%s\n' \
        "status=complete" \
        "schema=1" \
        "run_id=$RUN_ID" \
        "job_id=$JOB_ID" \
        "run_signature=$RUN_SIGNATURE" \
        "field_points=$EXPECTED_FIELDS" \
        "checkpoint_files=$CHECKPOINT_COUNT" \
        "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$SUCCESS_TMP"
mv "$SUCCESS_TMP" "$SUCCESS_MARKER"
SUCCESS_WRITTEN=1

date
echo "Complete: $RESULT_ROOT"
