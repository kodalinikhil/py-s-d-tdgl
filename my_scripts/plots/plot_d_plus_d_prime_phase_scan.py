"""Plot complete or partial d+d' phase-scan CSV files.

Examples
--------
Plot a local partial run::

    python my_scripts/plots/plot_d_plus_d_prime_phase_scan.py \
        results/d_plus_d_prime_phase_diagram/measurements.csv

Combine Hoffman2 array-task outputs::

    python my_scripts/plots/plot_d_plus_d_prime_phase_scan.py \
        results/hoffman2/ddp_*/alpha_*/measurements.csv \
        --output-directory results/hoffman2/combined

The plotter never replaces the simulation driver's ``transitions.csv``.  Its
recomputed transition summary is written to ``plotted_transitions.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from my_scripts.simulations.simulate_d_plus_d_prime_phase_diagram import (  # noqa: E402
    TRANSITION_FIELDS,
    build_transition_rows,
    plot_results,
    write_csv,
)


DEFAULT_THRESHOLDS = (1e-3, 3e-3, 1e-2)
TRANSITIONS_FILENAME = "plotted_transitions.csv"

# These are the minimum columns needed to identify a point, construct a sweep,
# and draw the two order-parameter amplitudes.  Newer diagnostic columns remain
# optional so that CSVs produced by older versions of the driver still load.
REQUIRED_COLUMNS = frozenset(
    {
        "alpha",
        "direction",
        "sequence_index",
        "reduced_field",
        "bulk_max_abs_d",
        "bulk_max_abs_d_prime",
        "bulk_mean_abs_d",
    }
)
KEY_NUMERIC_FIELDS = frozenset({"alpha", "sequence_index", "reduced_field"})
NUMERIC_FIELDS = frozenset(
    {
        "alpha",
        "sequence_index",
        "reduced_field",
        "applied_field",
        "bulk_max_abs_d",
        "bulk_mean_abs_d",
        "bulk_max_abs_d_prime",
        "bulk_mean_abs_d_prime",
        "normalized_bulk_max_abs_d_prime",
        "max_abs_d_prime",
        "bulk_relative_phase",
        "bulk_relative_phase_coherence",
        "convergence_max_density_change",
        "actual_solve_time",
        "accepted_steps",
        "equilibrium_error",
        "free_energy",
        "free_energy_per_area",
        "num_saved_states",
        "num_sites",
        "num_elements",
        "width",
        "max_edge_length",
        "boundary_strip",
        "solve_time",
        "wall_seconds",
    }
)
TRUE_VALUES = frozenset({"1", "true", "yes"})
FALSE_VALUES = frozenset({"0", "false", "no"})
_SOURCE_FIELDS = frozenset(
    {"_source_path", "_source_row_number", "_source_mtime_ns", "_input_index"}
)


def parse_thresholds(specification: str) -> tuple[float, ...]:
    """Parse a comma-separated collection of finite, positive thresholds."""
    try:
        thresholds = tuple(
            float(value.strip()) for value in specification.split(",") if value.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Thresholds must be comma-separated numbers."
        ) from error
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    if any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise argparse.ArgumentTypeError("Thresholds must be finite and positive.")
    # Preserve the user's primary-threshold choice while removing repeats.
    return tuple(dict.fromkeys(thresholds))


def _parse_equilibrium_status(
    value: str | None, *, path: Path, row_number: int
) -> tuple[bool | None, str]:
    if value is None or not value.strip():
        return None, "unchecked"
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True, "converged"
    if normalized in FALSE_VALUES:
        return False, "unconverged"
    raise ValueError(
        f"{path}:{row_number}: invalid equilibrium_reached value {value!r}; "
        "expected true/false, yes/no, 1/0, or blank."
    )


def _load_path(path: Path, input_index: int) -> list[dict]:
    path = path.resolve()
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError as error:
        raise ValueError(f"Cannot read measurement CSV {path}: {error}") from error

    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"{path}: missing CSV header.")
        duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
        if duplicates:
            raise ValueError(f"{path}: duplicate CSV columns: {', '.join(duplicates)}")
        missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"{path}:{row_number}: row has more values than the CSV header."
                )
            parsed = dict(row)
            for key in NUMERIC_FIELDS.intersection(parsed):
                raw = parsed.get(key)
                if raw is None or not raw.strip():
                    parsed[key] = math.nan
                    continue
                try:
                    parsed[key] = float(raw)
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{row_number}: {key} must be numeric, got {raw!r}."
                    ) from error

            for key in KEY_NUMERIC_FIELDS:
                if not math.isfinite(parsed[key]):
                    raise ValueError(
                        f"{path}:{row_number}: required key field {key} must be "
                        "finite."
                    )
            sequence_index = parsed["sequence_index"]
            if sequence_index < 0 or not sequence_index.is_integer():
                raise ValueError(
                    f"{path}:{row_number}: sequence_index must be a nonnegative "
                    "integer."
                )
            parsed["sequence_index"] = int(sequence_index)

            direction = (parsed.get("direction") or "").strip().lower()
            if direction not in {"up", "down"}:
                raise ValueError(
                    f"{path}:{row_number}: direction must be 'up' or 'down', "
                    f"got {parsed.get('direction')!r}."
                )
            parsed["direction"] = direction
            equilibrium_reached, status = _parse_equilibrium_status(
                row.get("equilibrium_reached"), path=path, row_number=row_number
            )
            recorded_status = (row.get("equilibrium_status") or "").strip().lower()
            if recorded_status == "reached":
                status = "converged"
            elif recorded_status in {"time_cap", "failed", "unconverged"}:
                status = "unconverged"
            elif recorded_status == "not_requested":
                status = "unchecked"
            parsed["equilibrium_reached"] = equilibrium_reached
            parsed["_equilibrium_status"] = status
            # The shared plot routine directly indexes the phase, while the
            # compact partial-run CSV format is allowed to omit it.
            parsed.setdefault("bulk_relative_phase", math.nan)
            parsed.setdefault("equilibrium_error", math.nan)
            parsed.setdefault("actual_solve_time", math.nan)
            parsed["_source_path"] = str(path)
            parsed["_source_row_number"] = row_number
            parsed["_source_mtime_ns"] = modified_ns
            parsed["_input_index"] = input_index
            rows.append(parsed)
    return rows


def _finite_or_negative_infinity(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return parsed if math.isfinite(parsed) else -math.inf


def _row_preference(row: dict) -> tuple:
    """Rank duplicate points without depending on filesystem enumeration order."""
    status_rank = {"unchecked": 0, "unconverged": 1, "converged": 2}[
        row["_equilibrium_status"]
    ]
    output_file = str(row.get("output_file", "")).lower()
    is_continuation = any(
        token in output_file for token in ("continu", "restart", "resume")
    )
    return (
        status_rank,
        int(is_continuation),
        _finite_or_negative_infinity(row.get("actual_solve_time")),
        _finite_or_negative_infinity(row.get("accepted_steps")),
        _finite_or_negative_infinity(row.get("solve_time")),
        int(row["_source_mtime_ns"]),
        int(row["_source_row_number"]),
        str(row["_source_path"]),
    )


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


def _rows_equivalent(left: dict, right: dict) -> bool:
    keys = (set(left) | set(right)).difference(_SOURCE_FIELDS)
    return all(_values_equal(left.get(key), right.get(key)) for key in keys)


def deduplicate_measurements(rows: Sequence[dict]) -> list[dict]:
    """Select one row for each alpha/direction/field point.

    A converged row wins over an unconverged or unchecked row.  Within that
    class, explicit continuations, longer solves, and newer source rows win in
    that order.  Conflicting duplicates are reported rather than hidden.
    """
    grouped: dict[tuple[float, str, float], list[dict]] = {}
    for row in rows:
        key = (float(row["alpha"]), row["direction"], float(row["reduced_field"]))
        grouped.setdefault(key, []).append(row)

    selected: list[dict] = []
    for key, candidates in grouped.items():
        winner = max(candidates, key=_row_preference)
        if len(candidates) > 1 and any(
            not _rows_equivalent(winner, candidate)
            for candidate in candidates
            if candidate is not winner
        ):
            sources = ", ".join(
                f"{row['_source_path']}:{row['_source_row_number']}"
                for row in candidates
            )
            warnings.warn(
                "Conflicting duplicate measurement for "
                f"alpha={key[0]:g}, direction={key[1]}, "
                f"reduced_field={key[2]:g}; selected "
                f"{winner['_source_path']}:{winner['_source_row_number']} from "
                f"[{sources}].",
                UserWarning,
                stacklevel=2,
            )
        selected.append(winner)

    direction_order = {"up": 0, "down": 1}
    return sorted(
        selected,
        key=lambda row: (
            float(row["alpha"]),
            direction_order[row["direction"]],
            int(row["sequence_index"]),
            float(row["reduced_field"]),
        ),
    )


def load_measurements(paths: Sequence[Path]) -> list[dict]:
    rows = [
        row
        for input_index, path in enumerate(paths)
        for row in _load_path(path, input_index)
    ]
    if not rows:
        raise ValueError("No measurement rows were found.")
    return deduplicate_measurements(rows)


def default_output_directory(paths: Sequence[Path]) -> Path:
    """Keep one-file behavior, but isolate any combined multi-file output."""
    resolved_parents = [path.resolve().parent for path in paths]
    if len(resolved_parents) == 1:
        return resolved_parents[0]
    common_parent = Path(os.path.commonpath([str(path) for path in resolved_parents]))
    if common_parent == Path(common_parent.anchor):
        common_parent = Path.cwd().resolve()
    return common_parent / "combined_phase_scan"


def _equilibrium_status(row: dict) -> str:
    status = row.get("_equilibrium_status")
    if status in {"converged", "unconverged", "unchecked"}:
        return status
    value = row.get("equilibrium_reached")
    if value is None:
        return "unchecked"
    return "converged" if bool(value) else "unconverged"


def _plot_status_markers(
    ax: plt.Axes,
    fields: Sequence[float],
    values: Sequence[float],
    statuses: Sequence[str],
    *,
    color,
    direction_marker: str,
) -> None:
    for status in ("converged", "unconverged", "unchecked"):
        indices = [
            index
            for index, (value, point_status) in enumerate(zip(values, statuses))
            if point_status == status and math.isfinite(float(value))
        ]
        if not indices:
            continue
        x = [fields[index] for index in indices]
        y = [values[index] for index in indices]
        if status == "unconverged":
            ax.scatter(x, y, color=color, marker="x", s=30, linewidths=1.25, zorder=3)
        elif status == "unchecked":
            ax.scatter(
                x,
                y,
                facecolors="none",
                edgecolors=color,
                marker=direction_marker,
                s=26,
                linewidths=1.1,
                zorder=3,
            )
        else:
            ax.scatter(x, y, color=color, marker=direction_marker, s=18, zorder=3)


def plot_measurements(
    rows: Sequence[dict], output_directory: Path, thresholds: Sequence[float]
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 11.5), sharex=True)
    finite_equilibrium_errors: list[float] = []
    colors = plt.cm.viridis(
        np.linspace(0.08, 0.92, len({row["alpha"] for row in rows}))
    )
    statuses_seen: set[str] = set()

    for color, alpha in zip(colors, sorted({row["alpha"] for row in rows})):
        for direction, linestyle, marker in (
            ("up", "-", "o"),
            ("down", "--", "s"),
        ):
            curve = sorted(
                (
                    row
                    for row in rows
                    if row["alpha"] == alpha and row["direction"] == direction
                ),
                key=lambda row: row["reduced_field"],
            )
            if not curve:
                continue
            fields = [float(row["reduced_field"]) for row in curve]
            statuses = [_equilibrium_status(row) for row in curve]
            statuses_seen.update(statuses)
            label = rf"$\alpha={alpha:g}$, {direction}"
            amplitude = [
                (
                    max(float(row["bulk_max_abs_d_prime"]), 1e-35)
                    if math.isfinite(float(row["bulk_max_abs_d_prime"]))
                    else math.nan
                )
                for row in curve
            ]
            bulk_d = [float(row["bulk_mean_abs_d"]) for row in curve]
            equilibrium_errors = []
            for row in curve:
                value = float(row.get("equilibrium_error", math.nan))
                if math.isfinite(value) and value > 0:
                    value = max(value, 1e-16)
                    finite_equilibrium_errors.append(value)
                else:
                    value = math.nan
                equilibrium_errors.append(value)
            solve_times = [
                float(row.get("actual_solve_time", math.nan)) for row in curve
            ]
            panel_values = (amplitude, bulk_d, equilibrium_errors, solve_times)

            for index, (ax, values) in enumerate(zip(axes, panel_values)):
                ax.plot(
                    fields,
                    values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                    label=label if index == 0 else None,
                )
                _plot_status_markers(
                    ax,
                    fields,
                    values,
                    statuses,
                    color=color,
                    direction_marker=marker,
                )

    for threshold in thresholds:
        axes[0].axhline(threshold, color="0.55", linewidth=0.8, linestyle=":")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"bulk $\max |d'|$")
    axes[0].set_title("d+d' field scan (all loaded points)")
    handles, labels = axes[0].get_legend_handles_labels()
    status_handles = {
        "converged": Line2D(
            [], [], color="0.25", marker="o", linestyle="None", label="converged"
        ),
        "unconverged": Line2D(
            [], [], color="0.25", marker="x", linestyle="None", label="not converged"
        ),
        "unchecked": Line2D(
            [],
            [],
            color="0.25",
            marker="o",
            markerfacecolor="none",
            linestyle="None",
            label="equilibrium unchecked",
        ),
    }
    handles.extend(
        status_handles[status] for status in status_handles if status in statuses_seen
    )
    labels.extend(
        status_handles[status].get_label()
        for status in status_handles
        if status in statuses_seen
    )
    axes[0].legend(handles, labels, fontsize=8, ncol=2)

    axes[1].set_ylabel(r"bulk mean $|d|$")
    if finite_equilibrium_errors:
        axes[2].set_yscale("log")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No completed equilibrium checks",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    axes[2].set_ylabel("equilibrium error")
    axes[3].set_ylabel("actual time")
    axes[3].set_xlabel(r"$b=B/B_{c2}$")
    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    output_path = output_directory / "partial_phase_scan.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def build_safe_transition_rows(
    rows: Sequence[dict], thresholds: Iterable[float]
) -> list[dict]:
    """Avoid interpreting a blank/nonfinite amplitude as a pure-d point."""
    transition_rows = build_transition_rows(rows, thresholds)
    incomplete_sweeps = {
        (float(row["alpha"]), row["direction"])
        for row in rows
        if not math.isfinite(float(row["bulk_max_abs_d_prime"]))
    }
    for transition in transition_rows:
        key = (float(transition["alpha"]), transition["direction"])
        if key not in incomplete_sweeps:
            continue
        transition.update(
            {
                "lower_field": math.nan,
                "upper_field": math.nan,
                "transition_field": math.nan,
                "field_uncertainty": math.nan,
                "status": "incomplete_data",
            }
        )
    return transition_rows


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", nargs="+", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Plot output directory. A single input defaults to its parent; "
            "multiple inputs default to a new combined_phase_scan directory."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        metavar="T[,T...]",
        help="Comma-separated positive d' thresholds (default: 1e-3,3e-3,1e-2).",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else default_output_directory(args.measurements)
    )
    rows = load_measurements(args.measurements)
    output_path = plot_measurements(rows, output_directory, args.thresholds)
    transitions = build_safe_transition_rows(rows, args.thresholds)
    transition_path = output_directory / TRANSITIONS_FILENAME
    write_csv(transition_path, transitions, TRANSITION_FIELDS)
    plot_results(rows, transitions, args.thresholds[0], output_directory)
    finite_rows = [
        row for row in rows if math.isfinite(float(row["bulk_max_abs_d_prime"]))
    ]
    print(
        f"Plotted {len(finite_rows)} of {len(rows)} measurements to {output_path}; "
        f"recomputed transitions: {transition_path}"
    )


if __name__ == "__main__":
    main()
