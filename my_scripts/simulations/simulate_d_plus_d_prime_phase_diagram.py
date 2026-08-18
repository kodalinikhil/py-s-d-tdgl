"""Reproduce the field-driven d+d' -> pure-d transition of Lei et al.

The source calculation is arXiv:cond-mat/0004227v1.  It minimizes Eq. (2)
with no orbital-Zeeman coupling, treats the induction as uniform, and uses a
magnetic-periodic vortex-lattice unit cell.  This script uses the same free
energy and uniform-induction approximation, but it runs on a finite square
with natural (open) boundaries because magnetic-periodic boundaries are not
yet implemented by this solver.  Comparisons with the paper are therefore
qualitative until domain- and mesh-convergence checks have been performed.

For every alpha the script starts from the mixed state at low field and sweeps
``b = B / Bc2`` upward.  By default it stops after the mixed-to-pure crossing
has been confirmed at two consecutive field points; a full return sweep is an
opt-in diagnostic rather than part of the paper's primary reproduction.  It
writes one HDF5 solution per point, crash-resilient CSV summaries, and three
plots:

* ``phase_diagram.png`` compares numerical transition brackets with Eqs. (8)
  and (9) of the paper;
* ``amplitude_vs_field.png`` shows the collapse of the bulk d' amplitude;
* ``figure2_style.png`` transposes the same data to mimic the axes of Fig. 2.

The default scan is deliberately substantial.  Use ``--smoke-test`` to check
the workflow quickly before launching a converged calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import tdgl  # noqa: E402


PAPER_ALPHA_STAR = 1 / 3
PAPER_HIGH_FIELD_ALPHA_MIN = 2 / 3
PAPER_BC2 = 1.0
MANIFEST_NAME = "scan_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
MAX_GRID_POINTS = 1_000_000

MEASUREMENT_FIELDS = [
    "alpha",
    "direction",
    "sequence_index",
    "reduced_field",
    "applied_field",
    "field_units",
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
    "num_elements",
    "width",
    "max_edge_length",
    "boundary_strip",
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

    width = _require_finite("--width", args.width)
    edge_length = _require_finite("--max-edge-length", args.max_edge_length)
    boundary_strip = _require_finite("--boundary-strip", args.boundary_strip)
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
    if width <= 0 or edge_length <= 0:
        raise ValueError("--width and --max-edge-length must be positive.")
    if not 0 <= boundary_strip < width / 2:
        raise ValueError("--boundary-strip must satisfy 0 <= strip < width / 2.")
    if solve_time <= 0 or dt_init <= 0 or dt_max <= 0 or dt_init > dt_max:
        raise ValueError(
            "--solve-time, --dt-init, and --dt-max must be positive, with "
            "dt-init <= dt-max."
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
    for option in (
        "smooth",
        "save_every",
        "progress_interval",
        "equilibrium_window",
    ):
        value = getattr(args, option)
        minimum = 0 if option == "smooth" else 1
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"--{option.replace('_', '-')} must be an integer.")
        if value < minimum:
            qualifier = "nonnegative" if minimum == 0 else "positive"
            raise ValueError(f"--{option.replace('_', '-')} must be {qualifier}.")
    if args.stop_after_pure_points < 0:
        raise ValueError("--stop-after-pure-points must be nonnegative.")
    if not str(args.field_units).strip():
        raise ValueError("--field-units cannot be empty.")
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


def interior_mask(mesh, width: float, boundary_strip: float) -> np.ndarray:
    """Select sites at least ``boundary_strip`` from the square boundary."""
    distance = width / 2 - np.max(np.abs(mesh.sites), axis=1)
    mask = distance >= boundary_strip
    if not np.any(mask):
        raise ValueError(
            "The boundary strip excludes every mesh site; reduce --boundary-strip "
            "or increase --width."
        )
    return mask


def area_average(values: np.ndarray, areas: np.ndarray, mask: np.ndarray) -> float:
    return float(np.average(np.asarray(values)[mask], weights=areas[mask]))


def circular_relative_phase(
    d: np.ndarray,
    d_prime: np.ndarray,
    areas: np.ndarray,
    mask: np.ndarray,
    amplitude_floor: float,
) -> tuple[float, float]:
    """Area-weighted mean phase and coherence of ``arg(d') - arg(d)``."""
    cross = d_prime * np.conj(d)
    valid = mask & (np.abs(d) >= amplitude_floor) & (np.abs(d_prime) >= amplitude_floor)
    if not np.any(valid):
        return math.nan, math.nan
    unit_cross = cross[valid] / np.abs(cross[valid])
    mean = np.average(unit_cross, weights=areas[valid])
    if not abs(mean):
        return math.nan, 0.0
    return float(np.angle(mean)), float(abs(mean))


def saved_frame_density_change(solution: tdgl.Solution) -> float:
    """Amplitude-squared change between saved frames (not a convergence test)."""
    with h5py.File(solution.path, "r") as h5file:
        steps = sorted(int(step) for step in h5file["data"])
    if len(steps) < 2:
        return math.nan

    final_step = steps[-1]
    previous_step = steps[-2]
    solution.solve_step = final_step
    final1 = np.abs(solution.tdgl_data.psi1) ** 2
    final2 = np.abs(solution.tdgl_data.psi2) ** 2
    solution.solve_step = previous_step
    previous1 = np.abs(solution.tdgl_data.psi1) ** 2
    previous2 = np.abs(solution.tdgl_data.psi2) ** 2
    solution.solve_step = final_step
    return float(
        max(
            np.max(np.abs(final1 - previous1)),
            np.max(np.abs(final2 - previous2)),
        )
    )


def final_density_change(solution: tdgl.Solution) -> float:
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


def run_diagnostics(solution: tdgl.Solution) -> dict:
    """Return convergence metadata stored with the final solver state."""
    solution.solve_step = -1
    state = solution.tdgl_data.state or {}
    with h5py.File(solution.path, "r") as h5file:
        num_saved_states = len(h5file["data"])
    equilibrium_error = float(state.get("equilibrium_error", math.nan))
    check_enabled = solution.options.equilibrium_tolerance is not None
    reached = bool(state.get("equilibrium_reached", False)) if check_enabled else None
    return {
        "actual_solve_time": float(state.get("time", math.nan)),
        "accepted_steps": int(state.get("step", -1)),
        "equilibrium_check_enabled": check_enabled,
        "equilibrium_reached": reached,
        "equilibrium_status": (
            ("reached" if reached else "time_cap") if check_enabled else "not_requested"
        ),
        "equilibrium_error": equilibrium_error,
        "num_saved_states": num_saved_states,
    }


def measure_solution(
    solution: tdgl.Solution,
    solver: tdgl.TDGLSolver,
    *,
    alpha: float,
    direction: str,
    sequence_index: int,
    reduced_field: float,
    applied_field: float,
    field_units: str,
    width: float,
    max_edge_length: float,
    boundary_strip: float,
    solve_time: float,
    phase_amplitude_floor: float,
    state_amplitude_threshold: float,
    normal_state_threshold: float,
) -> dict:
    solution.solve_step = -1
    mesh = solution.device.mesh
    d = solution.get_order_parameter("d")
    d_prime = solution.get_order_parameter("d_prime")
    bulk = interior_mask(mesh, width, boundary_strip)
    bulk_max_d = float(np.max(np.abs(d[bulk])))
    bulk_mean_d = area_average(np.abs(d), mesh.areas, bulk)
    max_d_prime = float(np.max(np.abs(d_prime)))
    bulk_max_d_prime = float(np.max(np.abs(d_prime[bulk])))
    bulk_mean_d_prime = area_average(np.abs(d_prime), mesh.areas, bulk)
    zero_field_d_prime = math.sqrt(3 * (3 * alpha - 1) / 8)
    phase, phase_coherence = circular_relative_phase(
        d,
        d_prime,
        mesh.areas,
        bulk,
        amplitude_floor=phase_amplitude_floor,
    )
    free_energy = solver.compute_d_plus_d_prime_free_energy(d, d_prime)
    area = float(np.sum(mesh.areas))
    diagnostics = run_diagnostics(solution)
    saved_frame_change = saved_frame_density_change(solution)
    row = {
        "alpha": alpha,
        "direction": direction,
        "sequence_index": sequence_index,
        "reduced_field": reduced_field,
        "applied_field": applied_field,
        "field_units": field_units,
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
        "free_energy_per_area": free_energy / area,
        "saved_frame_max_density_change": saved_frame_change,
        "convergence_max_density_change": saved_frame_change,
        **diagnostics,
        "num_sites": len(mesh.sites),
        "num_elements": len(mesh.elements),
        "width": width,
        "max_edge_length": max_edge_length,
        "boundary_strip": boundary_strip,
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
        "applied_field",
        "bulk_max_abs_d",
        "bulk_mean_abs_d",
        "max_abs_d_prime",
        "bulk_max_abs_d_prime",
        "bulk_mean_abs_d_prime",
        "normalized_bulk_max_abs_d_prime",
        "free_energy",
        "free_energy_per_area",
        "actual_solve_time",
        "wall_seconds",
        "width",
        "max_edge_length",
        "boundary_strip",
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
        "alphas": [float(value) for value in alphas],
        "fields": [float(value) for value in fields],
        "thresholds": [float(value) for value in thresholds],
        "width": float(args.width),
        "max_edge_length": float(args.max_edge_length),
        "boundary_strip": float(args.boundary_strip),
        "smooth": int(args.smooth),
        "solve_time": float(args.solve_time),
        "dt_init": float(args.dt_init),
        "dt_max": float(args.dt_max),
        "save_every": int(args.save_every),
        "progress_interval": int(args.progress_interval),
        "equilibrium_tolerance": (
            None
            if args.equilibrium_tolerance is None
            else float(args.equilibrium_tolerance)
        ),
        "equilibrium_window": int(args.equilibrium_window),
        "equilibrium_min_time": float(args.equilibrium_min_time),
        "field_units": str(args.field_units),
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


def validate_checkpoint(
    solution: tdgl.Solution,
    *,
    device,
    args,
    alpha: float,
    output_file: Path,
) -> None:
    """Reject stale, incompatible, or interrupted HDF5 checkpoints."""
    saved_mesh = solution.device.mesh
    expected_mesh = device.mesh
    same_mesh = (
        saved_mesh.sites.shape == expected_mesh.sites.shape
        and saved_mesh.elements.shape == expected_mesh.elements.shape
        and np.allclose(saved_mesh.sites, expected_mesh.sites, rtol=0, atol=1e-13)
        and np.array_equal(saved_mesh.elements, expected_mesh.elements)
    )
    expected_model = tdgl.DPlusDPrimeModel(alpha=alpha, zeeman_coupling=0.0)
    if not same_mesh or solution.device.layer.model != expected_model:
        raise RuntimeError(
            f"Checkpoint {output_file} does not match the requested mesh/model."
        )
    options = solution.options
    expected_options = {
        "solve_time": args.solve_time,
        "dt_init": args.dt_init,
        "dt_max": args.dt_max,
        "save_every": args.save_every,
        "progress_interval": args.progress_interval,
        "equilibrium_tolerance": args.equilibrium_tolerance,
        "equilibrium_window": args.equilibrium_window,
        "equilibrium_min_time": args.equilibrium_min_time,
    }
    for name, expected in expected_options.items():
        saved = getattr(options, name)
        if saved is None or expected is None:
            matches = saved is expected
        elif isinstance(expected, int):
            matches = int(saved) == expected
        else:
            matches = math.isclose(
                float(saved), float(expected), rel_tol=1e-12, abs_tol=1e-14
            )
        if not matches:
            raise RuntimeError(
                f"Checkpoint {output_file} has incompatible option {name}."
            )
    if options.field_units != args.field_units or bool(options.include_screening):
        raise RuntimeError(f"Checkpoint {output_file} has incompatible field settings.")
    state = solution.tdgl_data.state or {}
    final_time = float(state.get("time", math.nan))
    if not math.isfinite(final_time):
        raise RuntimeError(f"Checkpoint {output_file} has no finite final time.")
    reached = bool(state.get("equilibrium_reached", False))
    time_tolerance = max(1e-12, 2 * float(state.get("dt", args.dt_init)))
    if not reached and final_time + time_tolerance < args.solve_time:
        raise RuntimeError(
            f"Checkpoint {output_file} is incomplete at t={final_time:g}."
        )


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
    device,
    args,
    alpha: float,
) -> tdgl.Solution:
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
        solution = tdgl.Solution.from_hdf5(str(checkpoint))
    except Exception as exc:
        raise RuntimeError(f"Cannot load resume checkpoint {checkpoint}.") from exc
    validate_checkpoint(
        solution,
        device=device,
        args=args,
        alpha=alpha,
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
                label=f"Finite-square {direction} sweep",
            )
    ax.axvline(PAPER_BC2, color="0.5", linestyle=":", label=r"$b_{c2}=1$")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel=r"$b=B/B_{c2}$", ylabel=r"$\alpha$")
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
                fields_for_sweep = [float(row["reduced_field"]) for row in sweep]
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
                    [float(row["reduced_field"]) for row in phase_rows],
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
    axes[0].set_ylabel(r"bulk $\max |d'|$")
    axes[1].set_ylabel(r"bulk mean $|d|$")
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
    fields = sorted({float(row["reduced_field"]) for row in up_rows})
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
                if math.isclose(float(row["reduced_field"]), field, abs_tol=1e-12)
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
    ax.set(xlabel=r"$\alpha$", ylabel=r"bulk $\max |d'|$")
    ax.set_title("Finite-square analogue of paper Fig. 2")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_directory / "figure2_style.png", dpi=250)
    plt.close(fig)


def build_device(alpha: float, width: float, max_edge_length: float, smooth: int):
    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=10.0,
        thickness=0.1,
        model=tdgl.DPlusDPrimeModel(alpha=alpha, zeeman_coupling=0.0),
    )
    boundary_points = max(101, int(math.ceil(8 * width / max_edge_length)))
    film = tdgl.Polygon(
        "film", points=tdgl.geometry.box(width, width, points=boundary_points)
    )
    device = tdgl.Device("lei-d-plus-d-prime-square", layer=layer, film=film)
    device.make_mesh(max_edge_length=max_edge_length, smooth=smooth)
    return device


def solve_point(
    device,
    seed_solution,
    *,
    reduced_field: float,
    output_file: Path,
    field_units: str,
    solve_time: float,
    dt_init: float,
    dt_max: float,
    save_every: int,
    progress_interval: int,
    equilibrium_tolerance: Optional[float],
    equilibrium_window: int,
    equilibrium_min_time: float,
):
    hc2 = device.Bc2.to(field_units).magnitude
    applied_field = float(reduced_field * hc2)
    options = tdgl.SolverOptions(
        solve_time=solve_time,
        dt_init=dt_init,
        dt_max=dt_max,
        adaptive=True,
        terminal_psi=None,
        output_file=str(output_file),
        field_units=field_units,
        include_screening=False,
        save_every=save_every,
        progress_interval=progress_interval,
        equilibrium_tolerance=equilibrium_tolerance,
        equilibrium_window=equilibrium_window,
        equilibrium_min_time=equilibrium_min_time,
    )
    solver = tdgl.TDGLSolver(
        device=device,
        options=options,
        applied_vector_potential=applied_field,
        seed_solution=seed_solution,
    )
    if seed_solution is None and math.isclose(reduced_field, 0.0, abs_tol=1e-14):
        # The paper gives the exact uniform zero-field minimum.  Starting from
        # it avoids waiting for a 1e-4 symmetry-breaking perturbation to grow,
        # which is especially slow when alpha is just above 1/3.
        alpha = device.layer.model.alpha
        d_amplitude = math.sqrt(3 * (3 - alpha) / 8)
        d_prime_amplitude = math.sqrt(3 * (3 * alpha - 1) / 8)
        solver.psi1_init.fill(d_amplitude)
        solver.psi2_init.fill(-1j * d_prime_amplitude)
    solution = solver.solve()
    if solution is None:
        raise RuntimeError("The solve was cancelled before producing data.")
    return solution, solver, applied_field


def configure_smoke_test(args) -> None:
    """Apply the tiny end-to-end preset without leaving invalid timing options."""
    args.width = 3.0
    args.max_edge_length = 0.8
    args.boundary_strip = 0.4
    args.solve_time = 0.002
    args.dt_init = 1e-4
    args.dt_max = 1e-3
    args.save_every = 1
    args.smooth = 2
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
    if not np.any(np.isclose(fields, 0.0)):
        fields = np.insert(fields, 0, 0.0)
    fields.sort()
    strictest_threshold = float(np.min(thresholds))

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
        f"Building {args.width:g} xi square with target edge length "
        f"{args.max_edge_length:g} xi..."
    )
    device = build_device(
        float(alphas[0]), args.width, args.max_edge_length, args.smooth
    )
    print(
        f"Mesh: {len(device.mesh.sites)} sites, {len(device.mesh.elements)} triangles."
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
        device.layer.model = tdgl.DPlusDPrimeModel(
            alpha=alpha,
            zeeman_coupling=0.0,
        )
        seed_solution = None
        last_up_measurement = None
        alpha_up_measurements: list[dict] = []
        print(f"\nalpha={alpha:g} ({alpha_index + 1}/{len(alphas)})")
        for sequence_index, reduced_field in enumerate(fields):
            reduced_field = float(reduced_field)
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
                    device=device,
                    args=args,
                    alpha=alpha,
                )
                consumed.add(key)
            else:
                print(f"  up   b={reduced_field:.5g}")
                solution, solver, applied_field = solve_point(
                    device,
                    seed_solution,
                    reduced_field=reduced_field,
                    output_file=output_file,
                    field_units=args.field_units,
                    solve_time=args.solve_time,
                    dt_init=args.dt_init,
                    dt_max=args.dt_max,
                    save_every=args.save_every,
                    progress_interval=args.progress_interval,
                    equilibrium_tolerance=args.equilibrium_tolerance,
                    equilibrium_window=args.equilibrium_window,
                    equilibrium_min_time=args.equilibrium_min_time,
                )
                row = measure_solution(
                    solution,
                    solver,
                    alpha=alpha,
                    direction="up",
                    sequence_index=sequence_index,
                    reduced_field=reduced_field,
                    applied_field=applied_field,
                    field_units=args.field_units,
                    width=args.width,
                    max_edge_length=args.max_edge_length,
                    boundary_strip=args.boundary_strip,
                    solve_time=args.solve_time,
                    phase_amplitude_floor=args.phase_amplitude_floor,
                    state_amplitude_threshold=strictest_threshold,
                    normal_state_threshold=args.normal_state_threshold,
                )
                measurements.append(row)
                write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)
            alpha_up_measurements.append(row)
            last_up_measurement = row
            seed_solution = solution

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
            # The maximum-field state is already converged and is the initial
            # point of the return branch, so record it without solving twice.
            down_key = (alpha, "down", 0)
            if down_key in existing:
                down_start = existing[down_key]
                expected_file = output_directory / str(
                    last_up_measurement["output_file"]
                )
                load_resume_checkpoint(
                    down_start,
                    output_directory=output_directory,
                    expected_file=expected_file,
                    device=device,
                    args=args,
                    alpha=alpha,
                )
                consumed.add(down_key)
            else:
                down_start = dict(last_up_measurement)
                down_start["direction"] = "down"
                down_start["sequence_index"] = 0
                measurements.append(down_start)
                write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)
            for sequence_index, reduced_field in enumerate(fields[-2::-1], start=1):
                reduced_field = float(reduced_field)
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
                        device=device,
                        args=args,
                        alpha=alpha,
                    )
                    consumed.add(key)
                else:
                    print(f"  down b={reduced_field:.5g}")
                    solution, solver, applied_field = solve_point(
                        device,
                        seed_solution,
                        reduced_field=reduced_field,
                        output_file=output_file,
                        field_units=args.field_units,
                        solve_time=args.solve_time,
                        dt_init=args.dt_init,
                        dt_max=args.dt_max,
                        save_every=args.save_every,
                        progress_interval=args.progress_interval,
                        equilibrium_tolerance=args.equilibrium_tolerance,
                        equilibrium_window=args.equilibrium_window,
                        equilibrium_min_time=args.equilibrium_min_time,
                    )
                    row = measure_solution(
                        solution,
                        solver,
                        alpha=alpha,
                        direction="down",
                        sequence_index=sequence_index,
                        reduced_field=reduced_field,
                        applied_field=applied_field,
                        field_units=args.field_units,
                        width=args.width,
                        max_edge_length=args.max_edge_length,
                        boundary_strip=args.boundary_strip,
                        solve_time=args.solve_time,
                        phase_amplitude_floor=args.phase_amplitude_floor,
                        state_amplitude_threshold=strictest_threshold,
                        normal_state_threshold=args.normal_state_threshold,
                    )
                    measurements.append(row)
                    write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)
                seed_solution = solution

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
        help="Bulk max-|d'| thresholds; the first is used in plots.",
    )
    parser.add_argument("--width", type=float, default=12.0, help="Square width in xi.")
    parser.add_argument(
        "--max-edge-length", type=float, default=0.25, help="Mesh scale in xi."
    )
    parser.add_argument(
        "--boundary-strip",
        type=float,
        default=2.0,
        help="Excluded edge strip in xi.",
    )
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--solve-time", type=float, default=1500.0)
    parser.add_argument("--dt-init", type=float, default=1e-4)
    parser.add_argument("--dt-max", type=float, default=0.02)
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
    parser.add_argument("--field-units", default="mT")
    parser.add_argument("--phase-amplitude-floor", type=float, default=1e-3)
    parser.add_argument(
        "--normal-state-threshold",
        type=float,
        default=1e-3,
        help=(
            "Bulk amplitude below which the dominant d component is absent; "
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
            "Also run the return branch. This doubles the work and a pure-d "
            "seed cannot regrow d' without a perturbation, so it is not used "
            "for the primary reproduction."
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
