#!/bin/bash
# Hoffman2 high-field demonstration (Altair Grid Engine) array job.
# The three tasks target the quantitatively reliable branch of paper Eq. (9).
# Submit from the repository root with:
#   qsub my_scripts/workflows/hoffman2/run_phase_array.uge.sh
#
# Override paths at submission time when needed:
#   qsub -v TDGL_ENV_PREFIX=/path/to/env,TDGL_RESULT_ROOT=/u/project/... \
#       my_scripts/workflows/hoffman2/run_phase_array.uge.sh
#$ -cwd
#$ -j y
#$ -o ddp_phase.$JOB_ID.$TASK_ID.log
# Standard shared queues allow at most 24 hours; leave time for shutdown sync.
#$ -l h_rt=23:30:00,h_data=8G
#$ -pe shared 1
#$ -t 1-3:1
#$ -m a
#$ -notify
#$ -notify

set -euo pipefail

. /u/local/Modules/default/init/modules.sh
module load mamba

REPO_DIR="${TDGL_REPO_DIR:-$SGE_O_WORKDIR}"
ENV_PREFIX="${TDGL_ENV_PREFIX:-$HOME/.conda/envs/py-s-d-tdgl}"
RUN_ID="${TDGL_RUN_ID:-ddp_${JOB_ID}}"
RESULT_ROOT="${TDGL_RESULT_ROOT:-$REPO_DIR/results/hoffman2/$RUN_ID}"
RESUME="${TDGL_RESUME:-0}"

ALPHAS=(0.7 0.8 0.9)
FIELD_GRIDS=(
    "0,0.1,0.2,0.3,0.4,0.45,0.5,0.525,0.55,0.575,0.6,0.65,0.7,0.75"
    "0,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.675,0.7,0.725,0.75,0.8,0.85,0.9"
    "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.825,0.85,0.875,0.9,0.95"
)
TASK_INDEX=$((SGE_TASK_ID - 1))
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#ALPHAS[@]} )); then
    echo "Unexpected SGE_TASK_ID=$SGE_TASK_ID (expected 1-${#ALPHAS[@]})." >&2
    exit 2
fi
ALPHA="${ALPHAS[$TASK_INDEX]}"
FIELDS="${FIELD_GRIDS[$TASK_INDEX]}"
DRIVER_PATH="$REPO_DIR/my_scripts/simulations/simulate_d_plus_d_prime_phase_diagram.py"
DRIVER_CKSUM="$(cksum "$DRIVER_PATH" | awk '{ print $1 ":" $2 }')"
SCAN_SIGNATURE="v1|width=10|max_edge_length=0.35|boundary_strip=2|smooth=20|solve_time=1500|dt_init=0.0001|dt_max=0.02|save_every=10000|progress_interval=5000|equilibrium_tolerance=1e-5|equilibrium_window=1000|equilibrium_min_time=20|thresholds=0.001,0.003,0.01|field_units=mT|phase_amplitude_floor=0.001|down_sweep=0|stop_after_pure_points=2"

if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "TDGL_RESUME must be 0 or 1, not: $RESUME" >&2
    exit 2
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Missing environment: $ENV_PREFIX" >&2
    echo "Submit my_scripts/workflows/hoffman2/setup_environment.sh first." >&2
    exit 2
fi

SCRATCH_BASE="${SCRATCH:?SCRATCH is not set}/py-s-d-tdgl/$RUN_ID"
WORK_OUTPUT="$SCRATCH_BASE/alpha_$ALPHA"
FINAL_OUTPUT="$RESULT_ROOT/alpha_$ALPHA"
SUCCESS_MARKER="$FINAL_OUTPUT/_SUCCESS"
SUCCESS_TMP="$FINAL_OUTPUT/.SUCCESS.${JOB_ID}.${SGE_TASK_ID}.tmp"

directory_has_files() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -print -quit)" ]]
}

if [[ "$RESUME" == "0" ]] && {
    directory_has_files "$WORK_OUTPUT" || directory_has_files "$FINAL_OUTPUT";
}; then
    echo "Refusing to mix a fresh task with existing output for alpha=$ALPHA." >&2
    echo "Set TDGL_RESUME=1 to continue this run." >&2
    exit 2
fi

if [[ "$RESUME" == "1" && -f "$SUCCESS_MARKER" ]]; then
    RECORDED_ROWS="$(sed -n 's/^measurement_rows=//p' "$SUCCESS_MARKER")"
    RECORDED_H5="$(sed -n 's/^h5_files=//p' "$SUCCESS_MARKER")"
    ACTUAL_ROWS="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$FINAL_OUTPUT/measurements.csv" 2>/dev/null || true)"
    if [[ -d "$FINAL_OUTPUT/h5" ]]; then
        ACTUAL_H5="$(find "$FINAL_OUTPUT/h5" -type f -name '*.h5' | wc -l | tr -d ' ')"
    else
        ACTUAL_H5=0
    fi
    if grep -Fxq "status=complete" "$SUCCESS_MARKER" &&
        grep -Fxq "run_id=$RUN_ID" "$SUCCESS_MARKER" &&
        grep -Fxq "alpha=$ALPHA" "$SUCCESS_MARKER" &&
        grep -Fxq "fields=$FIELDS" "$SUCCESS_MARKER" &&
        grep -Fxq "scan_signature=$SCAN_SIGNATURE" "$SUCCESS_MARKER" &&
        grep -Fxq "driver_cksum=$DRIVER_CKSUM" "$SUCCESS_MARKER" &&
        [[ -s "$FINAL_OUTPUT/measurements.csv" ]] &&
        [[ -s "$FINAL_OUTPUT/transitions.csv" ]] &&
        [[ "$RECORDED_ROWS" =~ ^[1-9][0-9]*$ ]] &&
        [[ "$RECORDED_H5" =~ ^[1-9][0-9]*$ ]] &&
        [[ "$RECORDED_ROWS" == "$ACTUAL_ROWS" ]] &&
        [[ "$RECORDED_H5" == "$ACTUAL_H5" ]]; then
        echo "alpha=$ALPHA is already complete; leaving its validated result unchanged."
        exit 0
    fi
    echo "Refusing invalid or stale success marker: $SUCCESS_MARKER" >&2
    exit 2
fi

mkdir -p "$WORK_OUTPUT" "$FINAL_OUTPUT"
if [[ "$RESUME" == "1" ]]; then
    # Preserve any newer scratch files while restoring partial results copied
    # back by a previous TERM/USR handler. Never copy a success marker.
    rsync -a --update --exclude=_SUCCESS "$FINAL_OUTPUT/" "$WORK_OUTPUT/"
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="$SCRATCH_BASE/matplotlib-$SGE_TASK_ID"
export NUMBA_CACHE_DIR="$SCRATCH_BASE/numba-$SGE_TASK_ID"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export NUMBA_NUM_THREADS="${NSLOTS:-1}"
export VECLIB_MAXIMUM_THREADS="${NSLOTS:-1}"
export BLIS_NUM_THREADS="${NSLOTS:-1}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

sync_results() {
    rsync -a --partial --exclude=_SUCCESS "$WORK_OUTPUT/" "$FINAL_OUTPUT/"
}

sync_results_with_checksums() {
    rsync -a --checksum --partial --exclude=_SUCCESS \
        "$WORK_OUTPUT/" "$FINAL_OUTPUT/"
}

verify_results() {
    local differences
    [[ -s "$WORK_OUTPUT/measurements.csv" ]] || return 1
    [[ -s "$WORK_OUTPUT/transitions.csv" ]] || return 1
    awk 'END { exit !(NR >= 2) }' "$WORK_OUTPUT/measurements.csv" || return 1
    awk 'END { exit !(NR >= 2) }' "$WORK_OUTPUT/transitions.csv" || return 1
    [[ -n "$(find "$WORK_OUTPUT/h5" -type f -name '*.h5' -print -quit 2>/dev/null)" ]] || return 1
    differences="$(
        rsync -a --checksum --dry-run --itemize-changes --exclude=_SUCCESS \
            "$WORK_OUTPUT/" "$FINAL_OUTPUT/"
    )" || return 1
    [[ -z "$differences" ]]
}

SIMULATION_PID=""
SIGNAL_NAME=""
SUCCESS_WRITTEN=0

handle_signal() {
    SIGNAL_NAME="$1"
    echo "Received $SIGNAL_NAME; stopping the solver before syncing partial output." >&2
    if [[ -n "$SIMULATION_PID" ]]; then
        kill -TERM "$SIMULATION_PID" 2>/dev/null || true
    fi
}

cleanup() {
    local status=$?
    trap - EXIT TERM USR1 USR2
    if [[ -n "$SIMULATION_PID" ]]; then
        kill -TERM "$SIMULATION_PID" 2>/dev/null || true
        wait "$SIMULATION_PID" 2>/dev/null || true
    fi
    rm -f "$SUCCESS_TMP"
    if [[ "$SUCCESS_WRITTEN" != "1" ]] && [[ -d "$WORK_OUTPUT" ]]; then
        echo "Syncing resumable partial output to $FINAL_OUTPUT"
        if ! sync_results; then
            echo "Warning: partial-result sync failed." >&2
            status=1
        fi
    fi
    exit "$status"
}

trap 'handle_signal TERM' TERM
trap 'handle_signal USR1' USR1
trap 'handle_signal USR2' USR2
trap cleanup EXIT

echo "Job $JOB_ID.$SGE_TASK_ID on $(hostname -s)"
echo "alpha=$ALPHA"
echo "fields=$FIELDS"
echo "scratch=$WORK_OUTPUT"
echo "results=$FINAL_OUTPUT"
date

cd "$REPO_DIR"
set --
if [[ "$RESUME" == "1" ]]; then
    set -- "$@" --resume
fi
"$ENV_PREFIX/bin/python" my_scripts/simulations/simulate_d_plus_d_prime_phase_diagram.py \
    --alphas "$ALPHA" \
    --fields "$FIELDS" \
    --width 10 \
    --max-edge-length 0.35 \
    --boundary-strip 2 \
    --smooth 20 \
    --solve-time 1500 \
    --dt-init 0.0001 \
    --dt-max 0.02 \
    --save-every 10000 \
    --progress-interval 5000 \
    --equilibrium-tolerance 1e-5 \
    --equilibrium-window 1000 \
    --equilibrium-min-time 20 \
    --thresholds 0.001,0.003,0.01 \
    --field-units mT \
    --phase-amplitude-floor 0.001 \
    --no-down-sweep \
    --stop-after-pure-points 2 \
    --no-plots \
    --output-directory "$WORK_OUTPUT" \
    "$@" &
SIMULATION_PID=$!

set +e
wait "$SIMULATION_PID"
SIMULATION_STATUS=$?
set -e

if [[ -n "$SIGNAL_NAME" ]]; then
    wait "$SIMULATION_PID" 2>/dev/null || true
    SIMULATION_PID=""
    echo "Simulation interrupted by $SIGNAL_NAME; rerun with TDGL_RESUME=1." >&2
    case "$SIGNAL_NAME" in
        TERM) exit 143 ;;
        USR1) exit 138 ;;
        USR2) exit 140 ;;
        *) exit 130 ;;
    esac
fi
SIMULATION_PID=""
if [[ "$SIMULATION_STATUS" -ne 0 ]]; then
    echo "Simulation failed with status $SIMULATION_STATUS; partial data will be synced." >&2
    exit "$SIMULATION_STATUS"
fi

sync_results_with_checksums
if ! verify_results; then
    echo "Result verification failed; _SUCCESS will not be written." >&2
    exit 1
fi

MEASUREMENT_ROWS=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$FINAL_OUTPUT/measurements.csv")
H5_FILES=$(find "$FINAL_OUTPUT/h5" -type f -name '*.h5' | wc -l | tr -d ' ')
{
    printf '%s\n' \
        "status=complete" \
        "schema=1" \
        "run_id=$RUN_ID" \
        "job_id=$JOB_ID" \
        "task_id=$SGE_TASK_ID" \
        "alpha=$ALPHA" \
        "fields=$FIELDS" \
        "scan_signature=$SCAN_SIGNATURE" \
        "driver_cksum=$DRIVER_CKSUM" \
        "measurement_rows=$MEASUREMENT_ROWS" \
        "h5_files=$H5_FILES" \
        "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$SUCCESS_TMP"
mv "$SUCCESS_TMP" "$SUCCESS_MARKER"
SUCCESS_WRITTEN=1

date
