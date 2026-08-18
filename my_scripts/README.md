# Scientific scripts

The scripts are grouped by purpose:

- `simulations/` contains simulation drivers.
- `plots/` contains plotters and shared plotting helpers.
- `reproductions/` contains paper-reproduction workflows.
- `workflows/` contains platform-specific launch and cluster scripts.
- `manual_checks/` contains executable diagnostics that are not pytest tests.

Run commands from the repository root. Generated checkpoints, tables, and
plots belong under `results/`, which is intentionally ignored by Git. The
automated test suite remains under `tdgl/test/`.
