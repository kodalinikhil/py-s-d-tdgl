#!/bin/bash
# One-command macOS finite open-boundary framework scan for the d+d' model.
# Run from anywhere with:
#   ./my_scripts/workflows/mac/run_d_plus_d_prime_demo_mac.sh
#
# Optional overrides:
#   TDGL_PYTHON=/path/to/python TDGL_RUN_ID=my_run ./my_scripts/workflows/mac/run_d_plus_d_prime_demo_mac.sh
#   TDGL_SMOKE_TEST=1 ./my_scripts/workflows/mac/run_d_plus_d_prime_demo_mac.sh
# Resume an interrupted run by reusing its explicit run ID or output directory:
#   TDGL_RESUME=1 TDGL_RUN_ID=my_run ./my_scripts/workflows/mac/run_d_plus_d_prime_demo_mac.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_ID="${TDGL_RUN_ID:-mac_demo_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${TDGL_OUTPUT_DIR:-$REPO_DIR/results/mac/$RUN_ID}"

cd "$REPO_DIR"

if [[ -n "${TDGL_PYTHON:-}" ]]; then
    PYTHON_BIN="$TDGL_PYTHON"
else
    PYTHON_BIN=""
    for candidate in \
        "$REPO_DIR/.venv/bin/python" \
        "$(command -v python3 || true)" \
        "$(command -v python || true)"; do
        if [[ -x "$candidate" ]] && "$candidate" -c \
            'import h5py, matplotlib, numpy, scipy, tdgl' >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c \
    'import h5py, matplotlib, numpy, scipy, tdgl' >/dev/null 2>&1; then
    echo "A Python environment with this repository and its dependencies is required." >&2
    echo "From $REPO_DIR, create/activate an environment and run:" >&2
    echo "  python -m pip install -e ." >&2
    echo "Then rerun this script, or set TDGL_PYTHON=/path/to/python." >&2
    exit 2
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/py-s-d-tdgl-mpl-$RUN_ID"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

if [[ "${TDGL_RESUME:-0}" != "0" && "${TDGL_RESUME:-0}" != "1" ]]; then
    echo "TDGL_RESUME must be 0 or 1, not: ${TDGL_RESUME:-}" >&2
    exit 2
fi

set --
if [[ "${TDGL_SMOKE_TEST:-0}" == "1" ]]; then
    set -- "$@" --smoke-test
fi
if [[ "${TDGL_RESUME:-0}" == "1" ]]; then
    set -- "$@" --resume
fi

echo "Running the macOS finite open-boundary d+d' framework scan with $PYTHON_BIN"
echo "Results: $OUTPUT_DIR"

"$PYTHON_BIN" my_scripts/simulations/simulate_d_plus_d_prime_phase_diagram.py \
    --alphas 0.8 \
    --fields 0,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.675,0.7,0.725,0.75,0.8 \
    --width 8 \
    --max-edge-length 0.4 \
    --boundary-strip 1.5 \
    --smooth 10 \
    --solve-time 1500 \
    --dt-init 0.0001 \
    --dt-max 0.02 \
    --save-every 10000 \
    --progress-interval 2500 \
    --equilibrium-tolerance 1e-4 \
    --equilibrium-window 500 \
    --equilibrium-min-time 20 \
    --no-down-sweep \
    --stop-after-pure-points 2 \
    --output-directory "$OUTPUT_DIR" \
    "$@"

echo "Finished. Open $OUTPUT_DIR/amplitude_vs_field.png"
