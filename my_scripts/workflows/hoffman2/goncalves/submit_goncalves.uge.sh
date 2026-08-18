#!/bin/bash
# Submit the Hoffman2 Goncalves environment and simulation jobs.
# Run this helper from the repository root.

set -euo pipefail

REPO_DIR="${TDGL_REPO_DIR:-$(pwd)}"
ENV_PREFIX="${TDGL_ENV_PREFIX:-$HOME/.conda/envs/py-s-d-tdgl}"
RUN_ID="${TDGL_RUN_ID:-goncalves_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${TDGL_RESULT_ROOT:-$REPO_DIR/results/hoffman2/$RUN_ID}"
RESUME="${TDGL_RESUME:-0}"
REBUILD_ENV="${TDGL_REBUILD_ENV:-0}"

cd "$REPO_DIR"

if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "TDGL_RESUME must be 0 or 1, not: $RESUME" >&2
    exit 2
fi
if [[ "$RESUME" == "0" ]] && [[ -d "$RESULT_ROOT" ]] &&
    [[ -n "$(find "$RESULT_ROOT" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse nonempty result directory: $RESULT_ROOT" >&2
    exit 2
fi
if [[ "$RESUME" == "1" ]] && [[ ! -d "$RESULT_ROOT" ]]; then
    echo "Cannot resume missing result directory: $RESULT_ROOT" >&2
    exit 2
fi

COMMON_VARS="TDGL_REPO_DIR=$REPO_DIR,TDGL_ENV_PREFIX=$ENV_PREFIX,TDGL_REBUILD_ENV=$REBUILD_ENV"
SETUP_JOB=$(
    qsub -terse -v "$COMMON_VARS" \
        my_scripts/workflows/hoffman2/goncalves/setup_goncalves_environment.uge.sh
)
RUN_VARS="$COMMON_VARS,TDGL_RUN_ID=$RUN_ID,TDGL_RESULT_ROOT=$RESULT_ROOT,TDGL_RESUME=$RESUME"
SIMULATION_JOB=$(
    qsub -terse -hold_jid "$SETUP_JOB" -v "$RUN_VARS" \
        my_scripts/workflows/hoffman2/goncalves/run_goncalves.uge.sh
)

echo "Environment job: $SETUP_JOB"
echo "Simulation job: $SIMULATION_JOB"
echo "Run ID: $RUN_ID"
echo "Results: $RESULT_ROOT"
echo "If the queue limit interrupts it, resubmit with:"
echo "  TDGL_RUN_ID=$RUN_ID TDGL_RESUME=1 bash $0"
