# Goncalves sweep on Hoffman2

This workflow runs the square Goncalves simulation on a deterministic mesh
with spacing `h=0.1 xi`. It performs a continuation sweep from
`H/Hc2 = 0.0` to `2.5` and back to `0.0` in intervals of `0.1` (51 field
points total). Each field relaxes for at most 1500 dimensionless time units.

From the repository root on Hoffman2, submit both environment setup and the
simulation with:

```bash
bash my_scripts/workflows/hoffman2/goncalves/submit_goncalves.uge.sh
```

The job runs from `$SCRATCH`, synchronizes checkpoints to
`results/hoffman2/<run-id>/`, and writes `_SUCCESS` only after all 51 fields
and the final trapped-flux plot are present. The 23.5-hour queue limit may
require resubmission. Use the run ID printed by the first submission:

```bash
TDGL_RUN_ID=goncalves_YYYYMMDD_HHMMSS TDGL_RESUME=1 \
  bash my_scripts/workflows/hoffman2/goncalves/submit_goncalves.uge.sh
```

To use a project result directory or an existing environment, set
`TDGL_RESULT_ROOT` or `TDGL_ENV_PREFIX` on both the initial submission and
resubmissions.
