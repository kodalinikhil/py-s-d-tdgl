"""Reproduce the field-driven d+d' -> pure-d transition of Lei et al.

The source calculation is arXiv:cond-mat/0004227v1. It minimizes Eq. (2) with
no orbital-Zeeman coupling in a magnetic-periodic vortex-lattice unit cell and
holds the induction uniform. This driver now uses that same geometry and
fixed-background approximation directly: every positive ``b = B / Bc2`` is a
one-flux torus of area ``2 pi / b``, while ``b = 0`` uses a separate zero-flux
torus. The complete torus is bulk; there is no boundary strip.

Each field point starts independently from a common-vortex mixed ``d+d'``
seed. In particular, fields whose physical cell sizes differ are never seeded
blindly from one another. By default an upward scan stops after the
mixed-to-pure crossing has been confirmed at two consecutive points; a return
sweep remains an opt-in diagnostic. The script writes one model-neutral
schema-v2 magnetic-periodic HDF5 solution per point, crash-resilient CSV
summaries, and three plots:

* ``phase_diagram.png`` compares numerical transition brackets with Eqs. (8)
  and (9) of the paper;
* ``amplitude_vs_field.png`` shows the collapse of the bulk d' amplitude;
* ``figure2_style.png`` transposes the same data to mimic the axes of Fig. 2.

The default scan is deliberately substantial. Use ``--smoke-test`` to check
the workflow quickly before launching a converged calculation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import tdgl  # noqa: E402

PAPER_ALPHA_STAR = 1 / 3
PAPER_HIGH_FIELD_ALPHA_MIN = 2 / 3
PAPER_BC2 = 1.0
MANIFEST_NAME = "scan_manifest.json"
MANIFEST_SCHEMA_VERSION = 2
MAX_GRID_POINTS = 1_000_000
POSITIVE_FIELD_FLUX_QUANTA = 1
ZERO_FIELD_FLUX_QUANTA = 0
# At zero field the homogeneous minimum is independent of cell size. A compact
# reference torus avoids introducing a second geometry knob into the CLI.
ZERO_FIELD_CELL_AREA = 2 * math.pi
MAGNETIC_PERIODIC_BACKEND = "magnetic_periodic"
FIXED_BACKGROUND_CONTROL = "fixed_uniform_background"
DIMENSIONLESS_FIELD_UNITS = "B_c2"
_VORTEX_MODE_CACHE: dict[tuple[int, int, float], np.ndarray] = {}

MEASUREMENT_FIELDS = [
    "alpha",
    "direction",
    "sequence_index",
    "reduced_field",
    "backend",
    "field_control",
    "mean_reduced_induction",
    "flux_quanta",
    "vortex_count",
    "bulk_max_abs_d",
    "bulk_mean_abs_d",
    "max_abs_d_prime",
    "bulk_max_abs_d_prime",
    "bulk_mean_abs_d_prime",
    "normalized_bulk_max_abs_d_prime",
    "state_classification",
    "bulk_relative_phase",
    "bulk_relative_phase_coherence",
    "free_energy",
    "free_energy_per_area",
    "condensate_free_energy_density",
    "helmholtz_free_energy_density",
    "magnetic_free_energy_density",
    "saved_frame_max_density_change",
    # Deprecated compatibility alias. This diagnostic depends on save cadence
    # and is not the solver's equilibrium criterion.
    "convergence_max_density_change",
    "actual_solve_time",
    "accepted_steps",
    "equilibrium_check_enabled",
    "equilibrium_reached",
    "equilibrium_status",
    "equilibrium_error",
    "num_saved_states",
    "num_sites",
    "cell_length_x",
    "cell_length_y",
    "cell_area",
    "cell_aspect_ratio",
    "grid_nx",
    "grid_ny",
    "hx",
    "hy",
    "solve_time",
    "wall_seconds",
    "output_file",
]

TRANSITION_FIELDS = [
    "alpha",
    "direction",
    "amplitude_threshold",
    "lower_field",
    "upper_field",
    "transition_field",
    "field_uncertainty",
    "crossing_count",
    "crossing_brackets",
    "status",
    "paper_low_field",
    "paper_high_field",
]


def parse_grid(specification: str) -> np.ndarray:
    """Parse comma-separated values or an inclusive ``start:stop:step`` grid."""
    specification = specification.strip()
    if ":" not in specification:
        values = [float(value) for value in specification.split(",") if value.strip()]
        if not values:
            raise ValueError("Grid specification is empty.")
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Grid values must be finite.")
        if len(values) > MAX_GRID_POINTS:
            raise ValueError(f"Grid contains more than {MAX_GRID_POINTS:,} points.")
        return values

    parts = specification.split(":")
    if len(parts) != 3:
        raise ValueError("Range grids must have the form start:stop:step.")
    start, stop, step = (float(value) for value in parts)
    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("Grid range values must be finite.")
    if step <= 0 or stop < start:
        raise ValueError("Range grids require stop >= start and step > 0.")
    intervals = (stop - start) / step
    if not math.isfinite(intervals) or intervals + 1 > MAX_GRID_POINTS:
        raise ValueError(f"Grid contains more than {MAX_GRID_POINTS:,} points.")
    count = int(math.floor(intervals + 1e-12)) + 1
    values = start + step * np.arange(count, dtype=float)
    if values[-1] < stop - 1e-10:
        values = np.append(values, stop)
    if not np.all(np.isfinite(values)):
        raise ValueError("Grid values must be finite.")
    return values


def _require_finite(name: str, value: Any) -> float:
    """Return ``value`` as a finite float or raise a CLI-friendly error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number.")
    return number


def validate_scan_arguments(
    args,
    alphas: np.ndarray,
    fields: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    """Validate every numeric option before creating or deleting output."""
    if not (np.all(np.isfinite(alphas)) and np.all(np.isfinite(fields))):
        raise ValueError("Alpha and field grids must contain only finite values.")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Amplitude thresholds must be finite.")
    if np.any((alphas <= PAPER_ALPHA_STAR) | (alphas >= 1)):
        raise ValueError(
            "The field-driven mixed-to-pure transition requires 1/3 < alpha < 1."
        )
    if np.any((fields < 0) | (fields > PAPER_BC2)):
        raise ValueError("Reduced fields must lie in 0 <= b <= 1.")
    if np.any(thresholds <= 0):
        raise ValueError("Amplitude thresholds must be positive.")

    aspect_ratio = _require_finite("--aspect-ratio", args.aspect_ratio)
    solve_time = _require_finite("--solve-time", args.solve_time)
    dt_init = _require_finite("--dt-init", args.dt_init)
    dt_max = _require_finite("--dt-max", args.dt_max)
    equilibrium_min_time = _require_finite(
        "--equilibrium-min-time", args.equilibrium_min_time
    )
    phase_floor = _require_finite("--phase-amplitude-floor", args.phase_amplitude_floor)
    normal_floor = _require_finite(
        "--normal-state-threshold", args.normal_state_threshold
    )
    if aspect_ratio <= 0:
        raise ValueError("--aspect-ratio must be positive.")
    grid_points = args.grid_points
    if (
        isinstance(grid_points, bool)
        or not isinstance(grid_points, (int, np.integer))
        or grid_points < 3
    ):
        raise ValueError("--grid-points must be an integer of at least 3.")
    if grid_points**2 > MAX_GRID_POINTS:
        raise ValueError(
            f"The N x N grid may contain at most {MAX_GRID_POINTS:,} points."
        )
    if solve_time <= 0 or dt_init <= 0 or dt_max <= 0 or dt_init > dt_max:
        raise ValueError(
            "--solve-time, --dt-init, and --dt-max must be positive, with "
            "dt-init <= dt-max."
        )
    if not args.adaptive and dt_init != dt_max:
        raise ValueError(
            "Fixed-step mode requires --dt-init == --dt-max; pass --adaptive "
            "to allow a distinct maximum step."
        )
    if equilibrium_min_time < 0 or equilibrium_min_time > solve_time:
        raise ValueError(
            "--equilibrium-min-time must lie between zero and --solve-time."
        )
    if args.equilibrium_tolerance is not None:
        tolerance = _require_finite(
            "--equilibrium-tolerance", args.equilibrium_tolerance
        )
        if tolerance <= 0:
            raise ValueError("--equilibrium-tolerance must be positive.")
    if phase_floor < 0 or normal_floor <= 0:
        raise ValueError(
            "--phase-amplitude-floor must be nonnegative and "
            "--normal-state-threshold must be positive."
        )
    for option in ("save_every", "progress_interval", "equilibrium_window"):
        value = getattr(args, option)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"--{option.replace('_', '-')} must be an integer.")
        if value < 1:
            raise ValueError(f"--{option.replace('_', '-')} must be positive.")
    if (
        isinstance(args.stop_after_pure_points, bool)
        or not isinstance(args.stop_after_pure_points, (int, np.integer))
        or args.stop_after_pure_points < 0
    ):
        raise ValueError("--stop-after-pure-points must be nonnegative.")
    if not str(args.output_directory).strip():
        raise ValueError("--output-directory cannot be empty.")


def paper_low_field_transition(alpha: float) -> float:
    """Return the low-field root of paper Eq. (8), or NaN if none exists."""
    rhs = (3 * alpha - 1) / 2
    if rhs < 0 or rhs > 1 / math.e:
        return math.nan
    if math.isclose(rhs, 0.0, abs_tol=1e-14):
        return 0.0
    function = lambda b: b * math.log(1 / b) - rhs
    return float(brentq(function, np.finfo(float).eps, 1 / math.e))


def paper_high_field_transition(alpha: float) -> float:
    """Return paper Eq. (9)."""
    discriminant = max(0.0, 9 * alpha**2 - 12 * alpha + 4)
    return float((1 + math.sqrt(discriminant)) / 2)


def circular_relative_phase(
    d: np.ndarray,
    d_prime: np.ndarray,
    amplitude_floor: float,
) -> tuple[float, float]:
    """Torus mean phase and coherence of ``arg(d') - arg(d)``."""
    cross = d_prime * np.conj(d)
    valid = (np.abs(d) >= amplitude_floor) & (np.abs(d_prime) >= amplitude_floor)
    if not np.any(valid):
        return math.nan, math.nan
    unit_cross = cross[valid] / np.abs(cross[valid])
    mean = np.mean(unit_cross)
    if not abs(mean):
        return math.nan, 0.0
    return float(np.angle(mean)), float(abs(mean))


def saved_frame_density_change(solution: tdgl.MagneticPeriodicSolution) -> float:
    """Amplitude-squared change between saved frames (not a convergence test)."""
    if solution.num_frames < 2:
        return math.nan
    final = solution.frame(-1)
    previous = solution.frame(-2)
    final1 = np.abs(final.psi1) ** 2
    final2 = np.abs(final.psi2) ** 2
    previous1 = np.abs(previous.psi1) ** 2
    previous2 = np.abs(previous.psi2) ** 2
    return float(
        max(
            np.max(np.abs(final1 - previous1)),
            np.max(np.abs(final2 - previous2)),
        )
    )


def final_density_change(solution: tdgl.MagneticPeriodicSolution) -> float:
    """Compatibility alias for :func:`saved_frame_density_change`."""
    return saved_frame_density_change(solution)


def classify_state(
    row: dict,
    d_prime_threshold: float,
    normal_state_threshold: float,
) -> str:
    """Classify a finite bulk state without confusing the normal state for d."""
    try:
        d_prime = float(row["bulk_max_abs_d_prime"])
        # Older CSV files predate this diagnostic. Retain their old mixed/pure
        # interpretation, but all newly measured rows must contain finite |d|.
        d = float(row.get("bulk_max_abs_d", normal_state_threshold))
    except (TypeError, ValueError, KeyError):
        return "invalid"
    if not (math.isfinite(d) and math.isfinite(d_prime)) or d < 0 or d_prime < 0:
        return "invalid"
    if d < normal_state_threshold and d_prime < normal_state_threshold:
        return "normal"
    if d >= normal_state_threshold:
        return "mixed" if d_prime >= d_prime_threshold else "pure_d"
    return "d_prime_dominant"


def run_diagnostics(solution: tdgl.MagneticPeriodicSolution) -> dict:
    """Return convergence metadata stored with the final solver state."""
    state = solution.state
    equilibrium_error = float(state.get("equilibrium_error", math.nan))
    check_enabled = solution.options.equilibrium_tolerance is not None
    reached = bool(state.get("equilibrium_reached", False)) if check_enabled else None
    return {
        "actual_solve_time": float(solution.final_time),
        "accepted_steps": int(solution.final_step),
        "equilibrium_check_enabled": check_enabled,
        "equilibrium_reached": reached,
        "equilibrium_status": (
            ("reached" if reached else "time_cap") if check_enabled else "not_requested"
        ),
        "equilibrium_error": equilibrium_error,
        "num_saved_states": int(solution.num_frames),
    }


def measure_solution(
    solution: tdgl.MagneticPeriodicSolution,
    *,
    alpha: float,
    direction: str,
    sequence_index: int,
    reduced_field: float,
    solve_time: float,
    phase_amplitude_floor: float,
    state_amplitude_threshold: float,
    normal_state_threshold: float,
) -> dict:
    cell = solution.cell
    if solution.component_names != ("d", "d_prime"):
        raise RuntimeError(
            "Expected schema-v2 d+d' components ('d', 'd_prime'), found "
            f"{solution.component_names!r}."
        )
    d = solution.psi_d
    d_prime = solution.psi_d_prime
    bulk_max_d = float(np.max(np.abs(d)))
    bulk_mean_d = float(np.mean(np.abs(d)))
    max_d_prime = float(np.max(np.abs(d_prime)))
    bulk_max_d_prime = max_d_prime
    bulk_mean_d_prime = float(np.mean(np.abs(d_prime)))
    zero_field_d_prime = math.sqrt(3 * (3 * alpha - 1) / 8)
    phase, phase_coherence = circular_relative_phase(
        d,
        d_prime,
        amplitude_floor=phase_amplitude_floor,
    )
    condensate_density = float(solution.free_energy_density(include_magnetic=False))
    helmholtz_density = float(solution.free_energy_density(include_magnetic=True))
    magnetic_density = helmholtz_density - condensate_density
    free_energy = condensate_density * cell.dimensionless_area
    diagnostics = run_diagnostics(solution)
    saved_frame_change = saved_frame_density_change(solution)
    row = {
        "alpha": alpha,
        "direction": direction,
        "sequence_index": sequence_index,
        "reduced_field": reduced_field,
        "backend": MAGNETIC_PERIODIC_BACKEND,
        "field_control": FIXED_BACKGROUND_CONTROL,
        "mean_reduced_induction": solution.mean_induction,
        "flux_quanta": cell.flux_quanta,
        "vortex_count": solution.vortex_count,
        "bulk_max_abs_d": bulk_max_d,
        "bulk_mean_abs_d": bulk_mean_d,
        "max_abs_d_prime": max_d_prime,
        "bulk_max_abs_d_prime": bulk_max_d_prime,
        "bulk_mean_abs_d_prime": bulk_mean_d_prime,
        "normalized_bulk_max_abs_d_prime": (bulk_max_d_prime / zero_field_d_prime),
        "state_classification": "",  # Filled after all amplitudes are present.
        "bulk_relative_phase": phase,
        "bulk_relative_phase_coherence": phase_coherence,
        "free_energy": free_energy,
        "free_energy_per_area": condensate_density,
        "condensate_free_energy_density": condensate_density,
        "helmholtz_free_energy_density": helmholtz_density,
        "magnetic_free_energy_density": magnetic_density,
        "saved_frame_max_density_change": saved_frame_change,
        "convergence_max_density_change": saved_frame_change,
        **diagnostics,
        "num_sites": cell.num_sites,
        "cell_length_x": cell.dimensionless_lengths[0],
        "cell_length_y": cell.dimensionless_lengths[1],
        "cell_area": cell.dimensionless_area,
        "cell_aspect_ratio": cell.dimensionless_lengths[0]
        / cell.dimensionless_lengths[1],
        "grid_nx": cell.nx,
        "grid_ny": cell.ny,
        "hx": cell.hx,
        "hy": cell.hy,
        "solve_time": solve_time,
        "wall_seconds": solution.total_seconds,
        "output_file": str(Path("h5") / Path(solution.path).name),
    }
    row["state_classification"] = classify_state(
        row, state_amplitude_threshold, normal_state_threshold
    )
    validate_measurement(row)
    return row


def validate_measurement(row: dict) -> None:
    """Reject nonfinite solver output before it can create a false crossing."""
    required_finite = (
        "alpha",
        "reduced_field",
        "mean_reduced_induction",
        "flux_quanta",
        "vortex_count",
        "bulk_max_abs_d",
        "bulk_mean_abs_d",
        "max_abs_d_prime",
        "bulk_max_abs_d_prime",
        "bulk_mean_abs_d_prime",
        "normalized_bulk_max_abs_d_prime",
        "free_energy",
        "free_energy_per_area",
        "condensate_free_energy_density",
        "helmholtz_free_energy_density",
        "magnetic_free_energy_density",
        "actual_solve_time",
        "wall_seconds",
        "num_sites",
        "cell_length_x",
        "cell_length_y",
        "cell_area",
        "cell_aspect_ratio",
        "grid_nx",
        "grid_ny",
        "hx",
        "hy",
        "solve_time",
    )
    invalid = []
    for name in required_finite:
        try:
            finite = math.isfinite(float(row[name]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            invalid.append(name)
    if invalid:
        raise RuntimeError(
            "Cannot record a nonfinite measurement: " + ", ".join(invalid)
        )
    if row.get("backend") != MAGNETIC_PERIODIC_BACKEND:
        raise RuntimeError("Measurement does not use the magnetic-periodic backend.")
    if row.get("field_control") != FIXED_BACKGROUND_CONTROL:
        raise RuntimeError("Measurement does not use the fixed uniform background.")
    classification = row.get("state_classification")
    if classification not in {"mixed", "pure_d", "normal", "d_prime_dominant"}:
        raise RuntimeError(
            f"Cannot record invalid state classification {classification!r}."
        )


def _empty_transition(status: str, crossing_count: int = 0, brackets="") -> dict:
    return {
        "lower_field": math.nan,
        "upper_field": math.nan,
        "transition_field": math.nan,
        "field_uncertainty": math.nan,
        "crossing_count": crossing_count,
        "crossing_brackets": brackets,
        "status": status,
    }


def row_completion_quality(row: dict) -> str:
    """Return ``checked``, ``unchecked``, or ``failed`` for a CSV row."""
    status = str(row.get("equilibrium_status", "")).strip().lower()
    if status == "not_requested":
        return "unchecked"
    if status == "reached":
        return "checked"
    if status in {"time_cap", "failed", "unconverged"}:
        return "failed"
    if "equilibrium_reached" not in row:
        return "checked"  # Legacy rows had no explicit equilibrium metadata.
    value = row["equilibrium_reached"]
    if value is None or (isinstance(value, str) and not value.strip()):
        return "unchecked"
    if isinstance(value, str):
        return "checked" if value.strip().lower() in {"1", "true", "yes"} else "failed"
    return "checked" if bool(value) else "failed"


def transition_bracket(
    rows: Sequence[dict],
    threshold: float,
    normal_state_threshold: float = 1e-3,
) -> dict:
    """Find a unique mixed/pure-d crossing and flag recrossings explicitly."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Transition threshold must be finite and positive.")
    if not math.isfinite(normal_state_threshold) or normal_state_threshold <= 0:
        raise ValueError("Normal-state threshold must be finite and positive.")
    if not rows:
        return _empty_transition("no_data")
    try:
        finite_fields = all(math.isfinite(float(row["reduced_field"])) for row in rows)
    except (KeyError, TypeError, ValueError):
        finite_fields = False
    if not finite_fields:
        return _empty_transition("invalid_data")
    states = [classify_state(row, threshold, normal_state_threshold) for row in rows]
    crossings = []
    crossing_rows = []
    for previous, current, previous_state, current_state in zip(
        rows[:-1], rows[1:], states[:-1], states[1:]
    ):
        if {previous_state, current_state} == {"mixed", "pure_d"}:
            lower, upper = sorted(
                [float(previous["reduced_field"]), float(current["reduced_field"])]
            )
            crossings.append((lower, upper))
            crossing_rows.extend((previous, current))
    bracket_text = ";".join(f"{lower:.12g}:{upper:.12g}" for lower, upper in crossings)
    if len(crossings) > 1:
        qualities = {row_completion_quality(row) for row in crossing_rows}
        suffix = ""
        if "failed" in qualities:
            suffix = "_contain_unconverged"
        elif "unchecked" in qualities:
            suffix = "_without_equilibrium_check"
        return _empty_transition(
            "multiple_crossings" + suffix, len(crossings), bracket_text
        )
    if len(crossings) == 1:
        lower, upper = crossings[0]
        qualities = {row_completion_quality(row) for row in crossing_rows}
        if "failed" in qualities:
            status = "bracket_contains_unconverged"
        elif "unchecked" in qualities:
            status = "bracketed_without_equilibrium_check"
        else:
            status = "bracketed"
        return {
            "lower_field": lower,
            "upper_field": upper,
            "transition_field": 0.5 * (lower + upper),
            "field_uncertainty": 0.5 * (upper - lower),
            "crossing_count": 1,
            "crossing_brackets": bracket_text,
            "status": status,
        }
    any_unconverged = any(row_completion_quality(row) == "failed" for row in rows)
    if any_unconverged:
        status = "unconverged_no_crossing"
    elif "invalid" in states:
        status = "invalid_data"
    elif all(state == "mixed" for state in states):
        status = "mixed_at_all_fields"
    elif all(state == "pure_d" for state in states):
        status = "pure_at_all_fields"
    elif "normal" in states:
        status = "normal_state_reached_without_pure_d_crossing"
    else:
        status = "no_unique_crossing"
    return _empty_transition(status)


def row_is_converged(row: dict) -> bool:
    """Compatibility predicate: unchecked capped runs are usable, not failed."""
    return row_completion_quality(row) != "failed"


def transition_confirmed(
    rows: Sequence[dict],
    threshold: float,
    required_pure_points: int,
    normal_state_threshold: float = 1e-3,
) -> bool:
    """Whether a mixed point is followed by the requested pure-field tail."""
    if required_pure_points <= 0 or len(rows) <= required_pure_points:
        return False
    states = [classify_state(row, threshold, normal_state_threshold) for row in rows]
    tail_is_pure = all(state == "pure_d" for state in states[-required_pure_points:])
    tail_is_converged = all(
        row_is_converged(row) for row in rows[-required_pure_points:]
    )
    earlier_mixed = any(
        state == "mixed" and row_is_converged(row)
        for state, row in zip(
            states[:-required_pure_points], rows[:-required_pure_points]
        )
    )
    return tail_is_pure and tail_is_converged and earlier_mixed


def build_transition_rows(
    measurements: Sequence[dict],
    thresholds: Iterable[float],
    normal_state_threshold: float = 1e-3,
):
    rows = []
    alphas = sorted({float(row["alpha"]) for row in measurements})
    directions = ("up", "down")
    for alpha in alphas:
        for direction in directions:
            sweep = sorted(
                (
                    row
                    for row in measurements
                    if float(row["alpha"]) == alpha and row["direction"] == direction
                ),
                key=lambda row: int(row["sequence_index"]),
            )
            if not sweep:
                continue
            for threshold in thresholds:
                bracket = transition_bracket(sweep, threshold, normal_state_threshold)
                rows.append(
                    {
                        "alpha": alpha,
                        "direction": direction,
                        "amplitude_threshold": threshold,
                        **bracket,
                        "paper_low_field": paper_low_field_transition(alpha),
                        "paper_high_field": (
                            paper_high_field_transition(alpha)
                            if alpha >= PAPER_HIGH_FIELD_ALPHA_MIN
                            else math.nan
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Atomically replace a CSV summary after every completed solve."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def scan_configuration(args, alphas, fields, thresholds) -> dict:
    """Return the calculation-defining configuration stored for ``--resume``."""
    return {
        "backend": {
            "name": MAGNETIC_PERIODIC_BACKEND,
            "solution_schema_version": 2,
            "field_control": FIXED_BACKGROUND_CONTROL,
            "include_screening": False,
            "positive_field_flux_quanta": POSITIVE_FIELD_FLUX_QUANTA,
            "zero_field_flux_quanta": ZERO_FIELD_FLUX_QUANTA,
            "zero_field_cell_area": ZERO_FIELD_CELL_AREA,
        },
        "model": {
            "type": "DPlusDPrimeModel",
            "zeeman_coupling": 0.0,
        },
        "seed_policy": "independent_common_vortex_mixed",
        "alphas": [float(value) for value in alphas],
        "fields": [float(value) for value in fields],
        "thresholds": [float(value) for value in thresholds],
        "grid_points": int(args.grid_points),
        "aspect_ratio": float(args.aspect_ratio),
        "solve_time": float(args.solve_time),
        "dt_init": float(args.dt_init),
        "dt_max": float(args.dt_max),
        "adaptive": bool(args.adaptive),
        "save_every": int(args.save_every),
        "progress_interval": int(args.progress_interval),
        "equilibrium_tolerance": (
            None
            if args.equilibrium_tolerance is None
            else float(args.equilibrium_tolerance)
        ),
        "equilibrium_window": int(args.equilibrium_window),
        "equilibrium_min_time": float(args.equilibrium_min_time),
        "field_units": DIMENSIONLESS_FIELD_UNITS,
        "phase_amplitude_floor": float(args.phase_amplitude_floor),
        "normal_state_threshold": float(args.normal_state_threshold),
        "down_sweep": bool(args.down_sweep),
        "stop_after_pure_points": int(args.stop_after_pure_points),
        "smoke_test": bool(args.smoke_test),
    }


def ensure_safe_overwrite_target(path: Path) -> Path:
    """Reject broad paths before recursive output deletion."""
    target = path.expanduser().resolve()
    protected = {
        Path(target.anchor),
        Path.home().resolve(),
        Path.cwd().resolve(),
        REPOSITORY_ROOT.resolve(),
    }
    if any(item == target or item.is_relative_to(target) for item in protected):
        raise ValueError(
            f"Refusing to recursively delete broad/protected path: {target}"
        )
    if len(target.parts) < 3 or (target / ".git").exists():
        raise ValueError(f"Refusing to recursively delete unsafe path: {target}")
    return target


def _write_manifest(path: Path, configuration: dict) -> None:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "configuration": configuration,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def prepare_output_directory(
    output_directory: Path,
    configuration: dict,
    *,
    overwrite: bool,
    resume: bool,
) -> bool:
    """Prepare output and return whether a validated prior run is present."""
    output_directory = output_directory.expanduser().resolve()
    nonempty = output_directory.exists() and any(output_directory.iterdir())
    if nonempty and overwrite:
        shutil.rmtree(ensure_safe_overwrite_target(output_directory))
        nonempty = False
    if nonempty and not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}. Choose a new "
            "--output-directory, pass --resume, or pass --overwrite-output."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / MANIFEST_NAME
    if not nonempty:
        _write_manifest(manifest_path, configuration)
        return False
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Cannot resume {output_directory}: {MANIFEST_NAME} is missing."
        )
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read resume manifest {manifest_path}.") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported resume manifest schema in {manifest_path}.")
    saved = manifest.get("configuration")
    if saved != configuration:
        differing = sorted(
            key
            for key in set(saved or {}) | set(configuration)
            if (saved or {}).get(key) != configuration.get(key)
        )
        raise RuntimeError(
            "Resume configuration differs from the manifest for: "
            + ", ".join(differing)
        )
    return True


def read_measurements(path: Path) -> list[dict]:
    """Read a crash-resilient measurement prefix for resumption."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(MEASUREMENT_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"Cannot resume {path}: missing columns {sorted(missing)}."
            )
        return list(reader)


def _options_payload(options: tdgl.SolverOptions) -> dict:
    """Return an exact JSON-like representation for resume comparisons."""
    payload = dataclasses.asdict(options)
    return {
        name: value.value if isinstance(value, Enum) else value
        for name, value in payload.items()
    }


def validate_checkpoint(
    solution: tdgl.MagneticPeriodicSolution,
    *,
    cell: tdgl.MagneticPeriodicCell,
    args,
    output_file: Path,
) -> None:
    """Reject stale, incompatible, or interrupted periodic checkpoints."""
    if solution.cell != cell:
        raise RuntimeError(
            f"Checkpoint {output_file} does not match the exact requested "
            "cell/model/flux sector."
        )
    expected_options = make_solver_options(args, output_file)
    saved_payload = _options_payload(solution.options)
    expected_payload = _options_payload(expected_options)
    if saved_payload != expected_payload:
        differing = sorted(
            name
            for name in set(saved_payload) | set(expected_payload)
            if saved_payload.get(name) != expected_payload.get(name)
        )
        raise RuntimeError(
            f"Checkpoint {output_file} has incompatible solver options: "
            + ", ".join(differing)
        )
    if solution.component_names != ("d", "d_prime"):
        raise RuntimeError(
            f"Checkpoint {output_file} does not use d+d' schema-v2 components."
        )
    with h5py.File(solution.path, "r") as h5file:
        if (
            int(h5file.attrs.get("schema_version", -1)) != 2
            or h5file.attrs.get("backend") != MAGNETIC_PERIODIC_BACKEND
            or h5file.attrs.get("model_type") != "DPlusDPrimeModel"
            or h5file.attrs.get("field_control") != "fixed_background"
        ):
            raise RuntimeError(
                f"Checkpoint {output_file} has incompatible backend/schema metadata."
            )
    final_time = float(solution.final_time)
    if not math.isfinite(final_time):
        raise RuntimeError(f"Checkpoint {output_file} has no finite final time.")
    reached = bool(solution.state.get("equilibrium_reached", False))
    time_tolerance = max(1e-12, 2 * float(solution.final_frame.dt or args.dt_init))
    if not reached and final_time + time_tolerance < args.solve_time:
        raise RuntimeError(
            f"Checkpoint {output_file} is incomplete at t={final_time:g}."
        )


def validate_row_cell_metadata(row: dict, cell: tdgl.MagneticPeriodicCell) -> None:
    """Require resumed CSV geometry/flux metadata to match its checkpoint cell."""
    exact_values = {
        "backend": MAGNETIC_PERIODIC_BACKEND,
        "field_control": FIXED_BACKGROUND_CONTROL,
        "flux_quanta": cell.flux_quanta,
        "vortex_count": cell.flux_quanta,
        "grid_nx": cell.nx,
        "grid_ny": cell.ny,
        "num_sites": cell.num_sites,
    }
    for name, expected in exact_values.items():
        raw = row.get(name)
        try:
            actual = int(raw) if isinstance(expected, int) else raw
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Resume CSV has invalid {name} metadata.") from exc
        if actual != expected:
            raise RuntimeError(f"Resume CSV has incompatible {name} metadata.")
    numeric_values = {
        "mean_reduced_induction": cell.mean_induction,
        "cell_length_x": cell.dimensionless_lengths[0],
        "cell_length_y": cell.dimensionless_lengths[1],
        "cell_area": cell.dimensionless_area,
        "cell_aspect_ratio": (
            cell.dimensionless_lengths[0] / cell.dimensionless_lengths[1]
        ),
        "hx": cell.hx,
        "hy": cell.hy,
    }
    for name, expected in numeric_values.items():
        try:
            actual = float(row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Resume CSV has invalid {name} metadata.") from exc
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-13):
            raise RuntimeError(f"Resume CSV has incompatible {name} metadata.")


def index_existing_measurements(
    rows: Sequence[dict],
    *,
    alphas: np.ndarray,
    fields: np.ndarray,
    down_sweep: bool,
) -> dict[tuple[float, str, int], dict]:
    """Validate CSV coordinates and require a contiguous prefix per sweep."""
    alpha_values = [float(value) for value in alphas]
    indexed = {}
    indices_by_sweep: dict[tuple[float, str], list[int]] = {}
    for row in rows:
        validate_measurement(row)
        alpha = float(row["alpha"])
        if not any(math.isclose(alpha, value, abs_tol=1e-12) for value in alpha_values):
            raise RuntimeError(f"Resume CSV contains unrequested alpha={alpha}.")
        direction = row["direction"]
        if direction not in ({"up", "down"} if down_sweep else {"up"}):
            raise RuntimeError(f"Resume CSV contains invalid direction {direction!r}.")
        index = int(row["sequence_index"])
        expected_index = index if direction == "up" else len(fields) - 1 - index
        if expected_index < 0 or expected_index >= len(fields):
            raise RuntimeError("Resume CSV contains an out-of-range sequence index.")
        if not math.isclose(
            float(row["reduced_field"]), float(fields[expected_index]), abs_tol=1e-12
        ):
            raise RuntimeError("Resume CSV field grid does not match the manifest.")
        key = (alpha, direction, index)
        if key in indexed:
            raise RuntimeError(f"Resume CSV contains duplicate point {key}.")
        indexed[key] = row
        indices_by_sweep.setdefault((alpha, direction), []).append(index)
    for sweep, indices in indices_by_sweep.items():
        if sorted(indices) != list(range(max(indices) + 1)):
            raise RuntimeError(f"Resume CSV sweep {sweep} is not a contiguous prefix.")
    return indexed


def load_resume_checkpoint(
    row: dict,
    *,
    output_directory: Path,
    expected_file: Path,
    cell: tdgl.MagneticPeriodicCell,
    args,
) -> tdgl.MagneticPeriodicSolution:
    """Load the exact HDF5 file referenced by a validated resume row."""
    relative = Path(str(row["output_file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Resume CSV contains an unsafe output_file path.")
    checkpoint = (output_directory / relative).resolve()
    if checkpoint != expected_file.resolve():
        raise RuntimeError(
            f"Resume CSV points to {checkpoint}, expected {expected_file.resolve()}."
        )
    if not checkpoint.is_file():
        raise RuntimeError(f"Resume checkpoint is missing: {checkpoint}")
    try:
        solution = tdgl.MagneticPeriodicSolution.from_hdf5(str(checkpoint))
    except Exception as exc:
        raise RuntimeError(f"Cannot load resume checkpoint {checkpoint}.") from exc
    validate_row_cell_metadata(row, cell)
    validate_checkpoint(
        solution,
        cell=cell,
        args=args,
        output_file=checkpoint,
    )
    return solution


def archive_unindexed_checkpoint(path: Path) -> Path:
    """Preserve an HDF5 file that was not committed to the atomic CSV."""
    candidate = path.with_suffix(path.suffix + ".incomplete")
    counter = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f".incomplete.{counter}")
        counter += 1
    path.replace(candidate)
    return candidate


def plot_results(
    measurements: Sequence[dict],
    transitions: Sequence[dict],
    primary_threshold: float,
    output_directory: Path,
) -> None:
    def field_coordinate(row: dict) -> float:
        if row.get("backend") == MAGNETIC_PERIODIC_BACKEND:
            try:
                value = float(row["mean_reduced_induction"])
            except (KeyError, TypeError, ValueError):
                value = math.nan
            if math.isfinite(value):
                return value
        return float(row["reduced_field"])

    backends = {str(row.get("backend") or "open") for row in measurements}
    periodic = backends == {MAGNETIC_PERIODIC_BACKEND}
    geometry_label = "Magnetic-periodic" if periodic else "Finite-domain"
    region_label = "torus" if periodic else "bulk"
    alphas = sorted({float(row["alpha"]) for row in measurements})

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    analytic_alpha = np.linspace(PAPER_ALPHA_STAR, 1.0, 500)
    low_fields = np.asarray(
        [paper_low_field_transition(alpha) for alpha in analytic_alpha]
    )
    high_fields = np.asarray(
        [paper_high_field_transition(alpha) for alpha in analytic_alpha]
    )
    valid_low = np.isfinite(low_fields)
    ax.plot(
        low_fields[valid_low],
        analytic_alpha[valid_low],
        color="black",
        label="Paper Eq. (8), low-field root",
    )
    # Eq. (9) has a V-shaped mathematical continuation.  Only the branch on
    # which the transition field grows with alpha is the physical high-field
    # limit shown in the paper; the descending branch is omitted.
    valid_high = analytic_alpha >= PAPER_HIGH_FIELD_ALPHA_MIN
    ax.plot(
        high_fields[valid_high],
        analytic_alpha[valid_high],
        color="black",
        linestyle="--",
        label="Paper Eq. (9), high-field branch",
    )
    styles = (("up", "s", "tab:blue"), ("down", "^", "tab:orange"))
    for direction, marker, color in styles:
        selected = [
            row
            for row in transitions
            if row["direction"] == direction
            and math.isclose(
                float(row["amplitude_threshold"]), primary_threshold, rel_tol=1e-12
            )
            and row["status"] == "bracketed"
        ]
        if selected:
            ax.errorbar(
                [float(row["transition_field"]) for row in selected],
                [float(row["alpha"]) for row in selected],
                xerr=[float(row["field_uncertainty"]) for row in selected],
                fmt=marker,
                color=color,
                capsize=3,
                label=f"{geometry_label} {direction} scan",
            )
    ax.axvline(PAPER_BC2, color="0.5", linestyle=":", label=r"$b_{c2}=1$")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel=r"$b=B/B_{c2}$", ylabel=r"$\alpha$")
    ax.set_title(
        "Magnetic-periodic one-flux-cell transition"
        if periodic
        else "Finite-domain transition"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_directory / "phase_diagram.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(alphas)))
    for alpha, color in zip(alphas, colors):
        for direction, linestyle in (("up", "-"), ("down", "--")):
            sweep = sorted(
                (
                    row
                    for row in measurements
                    if float(row["alpha"]) == alpha and row["direction"] == direction
                ),
                key=lambda row: float(row["reduced_field"]),
            )
            if sweep:
                fields_for_sweep = [field_coordinate(row) for row in sweep]
                axes[0].plot(
                    fields_for_sweep,
                    [float(row["bulk_max_abs_d_prime"]) for row in sweep],
                    marker="o" if direction == "up" else None,
                    markersize=3,
                    linestyle=linestyle,
                    color=color,
                    label=rf"$\alpha={alpha:g}$ {direction}",
                )
                axes[1].plot(
                    fields_for_sweep,
                    [float(row["bulk_mean_abs_d"]) for row in sweep],
                    linestyle=linestyle,
                    color=color,
                )
                phase_rows = [
                    row
                    for row in sweep
                    if math.isfinite(float(row["bulk_relative_phase"]))
                ]
                axes[2].plot(
                    [field_coordinate(row) for row in phase_rows],
                    [
                        abs(float(row["bulk_relative_phase"])) / math.pi
                        for row in phase_rows
                    ],
                    linestyle=linestyle,
                    color=color,
                )
    axes[0].axhline(
        primary_threshold,
        color="black",
        linestyle=":",
        label="transition threshold",
    )
    axes[2].axhline(0.5, color="black", linestyle=":", label=r"$\pi/2$")
    axes[0].set_ylabel(rf"{region_label} $\max |d'|$")
    axes[1].set_ylabel(rf"{region_label} mean $|d|$")
    axes[2].set(xlabel=r"$b=B/B_{c2}$", ylabel=r"$|\arg(d'/d)|/\pi$")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=7)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_directory / "amplitude_vs_field.png", dpi=250)
    plt.close(fig)

    up_rows = [row for row in measurements if row["direction"] == "up"]
    fields = sorted({field_coordinate(row) for row in up_rows})
    if len(fields) > 10:
        indices = np.unique(np.linspace(0, len(fields) - 1, 10).round().astype(int))
        fields = [fields[index] for index in indices]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(fields)))
    for field, color in zip(fields, colors):
        curve = sorted(
            (
                row
                for row in up_rows
                if math.isclose(field_coordinate(row), field, abs_tol=1e-12)
            ),
            key=lambda row: float(row["alpha"]),
        )
        ax.plot(
            [float(row["alpha"]) for row in curve],
            [float(row["bulk_max_abs_d_prime"]) for row in curve],
            "o-",
            markersize=3,
            color=color,
            label=rf"$b={field:g}$",
        )
    ax.set(xlabel=r"$\alpha$", ylabel=rf"{region_label} $\max |d'|$")
    ax.set_title(f"{geometry_label} analogue of paper Fig. 2")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_directory / "figure2_style.png", dpi=250)
    plt.close(fig)


def cell_geometry(
    reduced_field: float, aspect_ratio: float
) -> tuple[float, float, int]:
    """Return ``(Lx, Ly, n)`` for a paper-style fixed-induction torus."""
    reduced_field = _require_finite("reduced_field", reduced_field)
    aspect_ratio = _require_finite("aspect_ratio", aspect_ratio)
    if reduced_field < 0 or reduced_field > PAPER_BC2:
        raise ValueError("reduced_field must lie in 0 <= b <= 1.")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive.")
    if reduced_field == 0:
        flux_quanta = ZERO_FIELD_FLUX_QUANTA
        area = ZERO_FIELD_CELL_AREA
    else:
        flux_quanta = POSITIVE_FIELD_FLUX_QUANTA
        area = 2 * math.pi * flux_quanta / reduced_field
    length_x = math.sqrt(area * aspect_ratio)
    length_y = math.sqrt(area / aspect_ratio)
    return length_x, length_y, flux_quanta


def build_cell(
    alpha: float,
    reduced_field: float,
    grid_points: int,
    aspect_ratio: float,
) -> tdgl.MagneticPeriodicCell:
    """Build the exact N x N magnetic-periodic cell for one scan point."""
    length_x, length_y, flux_quanta = cell_geometry(reduced_field, aspect_ratio)
    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=10.0,
        thickness=0.1,
        model=tdgl.DPlusDPrimeModel(alpha=alpha, zeeman_coupling=0.0),
    )
    return tdgl.MagneticPeriodicCell(
        layer=layer,
        lengths=(length_x, length_y),
        shape=(grid_points, grid_points),
        flux_quanta=flux_quanta,
        origin=(-length_x / 2, -length_y / 2),
        length_units="um",
        name="lei-d-plus-d-prime-vortex-cell",
    )


def fixed_step_stability_limit(cell: tdgl.MagneticPeriodicCell) -> float:
    """Conservative explicit diagonal-diffusion limit for the d+d' model."""
    model = cell.layer.model
    if not isinstance(model, tdgl.DPlusDPrimeModel):
        raise TypeError("fixed_step_stability_limit requires DPlusDPrimeModel.")
    relaxation = min(model.relaxation_d, model.relaxation_d_prime)
    return float(relaxation / (2 / cell.hx**2 + 2 / cell.hy**2))


def validate_fixed_step_stability(
    cell: tdgl.MagneticPeriodicCell,
    *,
    dt: float,
    reduced_field: float,
) -> None:
    """Reject a frozen-B fixed step above the discrete diffusion bound."""
    limit = fixed_step_stability_limit(cell)
    if dt > limit * (1 + 1e-13):
        raise ValueError(
            f"--dt-init={dt:g} is unstable for b={reduced_field:g}, "
            f"hx={cell.hx:g}, hy={cell.hy:g}; fixed-step D+D' diffusion "
            f"requires dt <= {limit:.8g}. Reduce the step/grid size or pass "
            "--adaptive."
        )


def uniform_mixed_amplitudes(alpha: float) -> tuple[float, float]:
    """Return the exact homogeneous zero-field mixed-state amplitudes."""
    return (
        math.sqrt(3 * (3 - alpha) / 8),
        math.sqrt(3 * (3 * alpha - 1) / 8),
    )


def independent_common_vortex_seed(
    cell: tdgl.MagneticPeriodicCell,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fresh mixed seed with common winding in the cell's flux sector."""
    alpha = cell.layer.model.alpha
    d_amplitude, d_prime_amplitude = uniform_mixed_amplitudes(alpha)
    if cell.flux_quanta == 0:
        common_vortex = np.ones(cell.shape, dtype=complex)
    elif cell.flux_quanta == 1:
        # The lowest eigenmode of the discrete covariant kinetic operator is a
        # smooth one-vortex section of the magnetic line bundle. Unlike a
        # radial phase pasted onto a torus, it obeys the magnetic seam exactly
        # and does not inject a large, artificial first timestep.
        lx, ly = cell.dimensionless_lengths
        cache_key = (cell.ny, cell.nx, round(lx / ly, 14))
        cached = _VORTEX_MODE_CACHE.get(cache_key)
        if cached is None:
            operators = tdgl.MagneticPeriodicOperators(cell)
            gradient = operators.covariant_gradient_matrix()
            kinetic = (gradient.conj().T @ gradient).tocsc()
            indices = np.arange(cell.num_sites, dtype=float)
            initial_vector = 1 + 1e-3j * indices / max(cell.num_sites - 1, 1)
            _, eigenvectors = eigsh(
                kinetic,
                k=1,
                which="SA",
                v0=initial_vector,
                tol=1e-11,
            )
            cached = eigenvectors[:, 0].reshape(cell.shape)
            anchor = cached.ravel()[np.argmax(np.abs(cached))]
            cached = cached * np.exp(-1j * np.angle(anchor))
            cached = cached / np.max(np.abs(cached))
            if operators.vortex_count(cached) != cell.flux_quanta:
                raise RuntimeError(
                    "Lowest magnetic-periodic seed mode has the wrong winding."
                )
            cached.setflags(write=False)
            _VORTEX_MODE_CACHE[cache_key] = cached
        common_vortex = cached.copy()
    else:  # This driver deliberately uses only zero- and one-flux cells.
        raise ValueError("The phase scan supports only zero or one flux quantum.")
    return (
        d_amplitude * common_vortex,
        -1j * d_prime_amplitude * common_vortex,
    )


def make_solver_options(args, output_file: Path) -> tdgl.SolverOptions:
    """Build the complete fixed-background option set used and resumed here."""
    return tdgl.SolverOptions(
        solve_time=args.solve_time,
        dt_init=args.dt_init,
        dt_max=args.dt_max,
        adaptive=bool(args.adaptive),
        terminal_psi=None,
        output_file=str(output_file.resolve()),
        field_units=DIMENSIONLESS_FIELD_UNITS,
        include_screening=False,
        save_every=args.save_every,
        progress_interval=args.progress_interval,
        equilibrium_tolerance=args.equilibrium_tolerance,
        equilibrium_window=args.equilibrium_window,
        equilibrium_min_time=args.equilibrium_min_time,
        sparse_solver="superlu",
    )


def solve_point(
    cell: tdgl.MagneticPeriodicCell,
    *,
    output_file: Path,
    args,
) -> tdgl.MagneticPeriodicSolution:
    """Solve one field from a fresh same-cell mixed vortex seed."""
    d_seed, d_prime_seed = independent_common_vortex_seed(cell)
    solution = tdgl.solve_magnetic_periodic(
        cell,
        make_solver_options(args, output_file),
        initial_psi1=d_seed,
        initial_psi2=d_prime_seed,
    )
    if solution.component_names != ("d", "d_prime"):
        raise RuntimeError("Magnetic-periodic solver did not write d+d' schema v2.")
    return solution


def configure_smoke_test(args) -> None:
    """Apply the tiny end-to-end preset without leaving invalid timing options."""
    args.grid_points = 8
    args.aspect_ratio = 1.0
    args.solve_time = 0.002
    args.dt_init = 1e-3
    args.dt_max = 1e-3
    args.adaptive = False
    args.save_every = 1
    args.equilibrium_min_time = 0.0
    args.stop_after_pure_points = 0


def run_scan(args) -> tuple[list[dict], list[dict]]:
    alphas = parse_grid(args.alphas)
    fields = np.unique(parse_grid(args.fields))
    thresholds = parse_grid(args.thresholds)
    if args.smoke_test:
        alphas = np.asarray([0.5])
        fields = np.asarray([0.0, 0.6])
        configure_smoke_test(args)
    validate_scan_arguments(args, alphas, fields, thresholds)
    if not np.any(fields == 0):
        fields = np.insert(fields, 0, 0.0)
    fields.sort()
    strictest_threshold = float(np.min(thresholds))

    # Validate every explicit fixed step before the output manifest/directory
    # can be created or overwritten. The most restrictive cell is generally
    # at high field, where a fixed N x N grid has the smallest spacing.
    if not args.adaptive:
        for alpha in alphas:
            for reduced_field in fields:
                cell = build_cell(
                    float(alpha),
                    float(reduced_field),
                    args.grid_points,
                    args.aspect_ratio,
                )
                validate_fixed_step_stability(
                    cell,
                    dt=args.dt_init,
                    reduced_field=float(reduced_field),
                )

    output_directory = Path(args.output_directory).resolve()
    configuration = scan_configuration(args, alphas, fields, thresholds)
    resumed = prepare_output_directory(
        output_directory,
        configuration,
        overwrite=args.overwrite_output,
        resume=args.resume,
    )
    h5_directory = output_directory / "h5"
    h5_directory.mkdir(parents=True, exist_ok=True)
    measurements_path = output_directory / "measurements.csv"
    transitions_path = output_directory / "transitions.csv"

    print(
        f"Magnetic-periodic fixed-background scan: {args.grid_points} x "
        f"{args.grid_points} sites, Lx/Ly={args.aspect_ratio:g}, "
        f"{'adaptive' if args.adaptive else 'fixed'} timestep."
    )
    measurements = read_measurements(measurements_path) if resumed else []
    existing = index_existing_measurements(
        measurements,
        alphas=alphas,
        fields=fields,
        down_sweep=args.down_sweep,
    )
    consumed: set[tuple[float, str, int]] = set()
    referenced = {
        (output_directory / str(row["output_file"])).resolve() for row in measurements
    }
    if resumed:
        for checkpoint in h5_directory.glob("*.h5"):
            if checkpoint.resolve() not in referenced:
                archived = archive_unindexed_checkpoint(checkpoint)
                print(f"Preserved unindexed checkpoint as {archived.name}")

    for alpha_index, alpha in enumerate(alphas):
        alpha = float(alpha)
        alpha_up_measurements: list[dict] = []
        print(f"\nalpha={alpha:g} ({alpha_index + 1}/{len(alphas)})")
        for sequence_index, reduced_field in enumerate(fields):
            reduced_field = float(reduced_field)
            cell = build_cell(alpha, reduced_field, args.grid_points, args.aspect_ratio)
            stem = f"alpha_{alpha:.6f}_up_{sequence_index:03d}_b_{reduced_field:.6f}.h5"
            output_file = h5_directory / stem
            key = (alpha, "up", sequence_index)
            if key in existing:
                print(f"  up   b={reduced_field:.5g} (resume)")
                row = existing[key]
                solution = load_resume_checkpoint(
                    row,
                    output_directory=output_directory,
                    expected_file=output_file,
                    cell=cell,
                    args=args,
                )
                consumed.add(key)
            else:
                print(
                    f"  up   b={reduced_field:.5g}, n={cell.flux_quanta}, "
                    f"L=({cell.dimensionless_lengths[0]:.5g}, "
                    f"{cell.dimensionless_lengths[1]:.5g})"
                )
                solution = solve_point(
                    cell,
                    output_file=output_file,
                    args=args,
                )
                row = measure_solution(
                    solution,
                    alpha=alpha,
                    direction="up",
                    sequence_index=sequence_index,
                    reduced_field=reduced_field,
                    solve_time=args.solve_time,
                    phase_amplitude_floor=args.phase_amplitude_floor,
                    state_amplitude_threshold=strictest_threshold,
                    normal_state_threshold=args.normal_state_threshold,
                )
                measurements.append(row)
                write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)
            alpha_up_measurements.append(row)

            if not args.down_sweep and transition_confirmed(
                alpha_up_measurements,
                strictest_threshold,
                args.stop_after_pure_points,
                args.normal_state_threshold,
            ):
                print(
                    "  transition confirmed by "
                    f"{args.stop_after_pure_points} consecutive pure-d points; "
                    "ending this alpha scan early"
                )
                break

        if args.down_sweep:
            # Direction records scan order only. Each point still receives a
            # fresh same-cell mixed seed, so no differently sized torus is ever
            # used as an implicit continuation state.
            for sequence_index, reduced_field in enumerate(fields[::-1]):
                reduced_field = float(reduced_field)
                cell = build_cell(
                    alpha, reduced_field, args.grid_points, args.aspect_ratio
                )
                stem = (
                    f"alpha_{alpha:.6f}_down_{sequence_index:03d}_"
                    f"b_{reduced_field:.6f}.h5"
                )
                output_file = h5_directory / stem
                key = (alpha, "down", sequence_index)
                if key in existing:
                    print(f"  down b={reduced_field:.5g} (resume)")
                    row = existing[key]
                    solution = load_resume_checkpoint(
                        row,
                        output_directory=output_directory,
                        expected_file=output_file,
                        cell=cell,
                        args=args,
                    )
                    consumed.add(key)
                else:
                    print(
                        f"  down b={reduced_field:.5g}, n={cell.flux_quanta}, "
                        f"L=({cell.dimensionless_lengths[0]:.5g}, "
                        f"{cell.dimensionless_lengths[1]:.5g})"
                    )
                    solution = solve_point(
                        cell,
                        output_file=output_file,
                        args=args,
                    )
                    row = measure_solution(
                        solution,
                        alpha=alpha,
                        direction="down",
                        sequence_index=sequence_index,
                        reduced_field=reduced_field,
                        solve_time=args.solve_time,
                        phase_amplitude_floor=args.phase_amplitude_floor,
                        state_amplitude_threshold=strictest_threshold,
                        normal_state_threshold=args.normal_state_threshold,
                    )
                    measurements.append(row)
                    write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)

        transitions = build_transition_rows(
            measurements, thresholds, args.normal_state_threshold
        )
        write_csv(transitions_path, transitions, TRANSITION_FIELDS)

    unused = set(existing) - consumed
    if unused:
        raise RuntimeError(
            "Resume CSV contains points beyond the valid execution prefix: "
            f"{sorted(unused)}"
        )
    transitions = build_transition_rows(
        measurements, thresholds, args.normal_state_threshold
    )
    write_csv(transitions_path, transitions, TRANSITION_FIELDS)
    if not args.no_plots:
        plot_results(
            measurements,
            transitions,
            float(thresholds[0]),
            output_directory,
        )
    return measurements, transitions


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="0.4:0.9:0.1")
    parser.add_argument("--fields", default="0:0.95:0.05")
    parser.add_argument(
        "--thresholds",
        default="0.001,0.003,0.01",
        help="Whole-torus max-|d'| thresholds; the first is used in plots.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=24,
        metavar="N",
        help="Fixed endpoint-excluded N x N grid used at every field.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=1.0,
        metavar="LX/LY",
        help="Physical magnetic-periodic cell aspect ratio Lx/Ly.",
    )
    parser.add_argument("--solve-time", type=float, default=1500.0)
    parser.add_argument("--dt-init", type=float, default=0.002)
    parser.add_argument("--dt-max", type=float, default=0.002)
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Opt into adaptive timestepping up to --dt-max. Fixed-step mode "
            "is the default and requires dt-init == dt-max."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
        help=(
            "Store one time slice every N solver steps. The default keeps "
            "long phase-diagram runs compact while retaining convergence data."
        ),
    )
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument(
        "--equilibrium-tolerance",
        type=float,
        default=1e-5,
        help=(
            "Maximum phase-aligned order-parameter change over an equilibrium "
            "window; stops a field point early when satisfied."
        ),
    )
    parser.add_argument(
        "--no-equilibrium-stop",
        action="store_const",
        const=None,
        dest="equilibrium_tolerance",
        help="Disable convergence-based early stopping and run every point to the cap.",
    )
    parser.add_argument("--equilibrium-window", type=int, default=1000)
    parser.add_argument("--equilibrium-min-time", type=float, default=20.0)
    parser.add_argument("--phase-amplitude-floor", type=float, default=1e-3)
    parser.add_argument(
        "--normal-state-threshold",
        type=float,
        default=1e-3,
        help=(
            "Whole-torus amplitude below which the dominant d component is absent; "
            "prevents the normal state from being classified as pure d."
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=REPOSITORY_ROOT / "results/d_plus_d_prime_phase_diagram",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite-output",
        action="store_true",
        help=(
            "Delete a nonempty output directory before running. Without this "
            "flag, the script refuses to mix a new scan with existing results."
        ),
    )
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible scan using its manifest, atomic CSV, and HDF5 "
            "checkpoints. A missing or empty directory starts normally."
        ),
    )
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument(
        "--down-sweep",
        action="store_true",
        help=(
            "Also visit the field grid in descending order. Every point still "
            "uses an independent mixed seed, so this is an ordering diagnostic "
            "rather than cross-cell continuation."
        ),
    )
    sweep_group.add_argument(
        "--no-down-sweep",
        action="store_false",
        dest="down_sweep",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(down_sweep=False)
    parser.add_argument(
        "--stop-after-pure-points",
        type=int,
        default=2,
        help=(
            "For an up-only scan, stop an alpha after a mixed point is followed "
            "by this many consecutive pure-d points; use 0 for the full grid."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-process plots (useful for cluster array workers).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two tiny field points to verify the full data/plot pipeline.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    measurements, transitions = run_scan(args)
    bracketed = sum(row["status"] == "bracketed" for row in transitions)
    output_directory = Path(args.output_directory).resolve()
    print(
        f"\nCompleted {len(measurements)} measurements; "
        f"found {bracketed} converged threshold brackets. "
        f"Results: {output_directory}"
    )


if __name__ == "__main__":
    main()
