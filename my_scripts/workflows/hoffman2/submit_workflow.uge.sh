#!/bin/bash
# Lightweight production-Hoffman2 submission helper; run from the repository root.
set -euo pipefail

REPO_DIR="${TDGL_REPO_DIR:-$(pwd)}"
ENV_PREFIX="${TDGL_ENV_PREFIX:-$HOME/.conda/envs/py-s-d-tdgl}"
RUN_ID="${TDGL_RUN_ID:-ddp_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${TDGL_RESULT_ROOT:-$REPO_DIR/results/hoffman2/$RUN_ID}"
RESUME="${TDGL_RESUME:-0}"
cd "$REPO_DIR"

if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "TDGL_RESUME must be 0 or 1, not: $RESUME" >&2
    exit 2
fi

if [[ "$RESUME" == "0" ]] && [[ -d "$RESULT_ROOT" ]] &&
    [[ -n "$(find "$RESULT_ROOT" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse nonempty result directory: $RESULT_ROOT" >&2
    echo "Choose a new run, or set TDGL_RESUME=1 to continue it." >&2
    exit 2
fi
if [[ "$RESUME" == "1" ]] && [[ ! -d "$RESULT_ROOT" ]]; then
    echo "Cannot resume because the result directory does not exist: $RESULT_ROOT" >&2
    exit 2
fi

REBUILD_ENV="${TDGL_REBUILD_ENV:-0}"
COMMON_VARS="TDGL_REPO_DIR=$REPO_DIR,TDGL_ENV_PREFIX=$ENV_PREFIX,TDGL_REBUILD_ENV=$REBUILD_ENV"
SETUP_JOB=$(
    qsub -terse -v "$COMMON_VARS" my_scripts/workflows/hoffman2/setup_environment.sh
)
ARRAY_VARS="$COMMON_VARS,TDGL_RUN_ID=$RUN_ID,TDGL_RESULT_ROOT=$RESULT_ROOT,TDGL_RESUME=$RESUME"
ARRAY_JOB=$(
    qsub -terse \
        -hold_jid "$SETUP_JOB" \
        -v "$ARRAY_VARS" \
        my_scripts/workflows/hoffman2/run_phase_array.uge.sh
)
ARRAY_JOB_ID="${ARRAY_JOB%%.*}"
PLOT_JOB=$(
    qsub -terse \
        -hold_jid "$ARRAY_JOB_ID" \
        -v "$ARRAY_VARS" \
        my_scripts/workflows/hoffman2/plot_results.uge.sh
)

echo "Environment job: $SETUP_JOB"
echo "Simulation array: $ARRAY_JOB"
echo "Plot job: $PLOT_JOB"
echo "Results: $RESULT_ROOT"
