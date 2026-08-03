"""Reproduce the field-driven d+d' -> pure-d transition of Lei et al.

The source calculation is arXiv:cond-mat/0004227v1.  It minimizes Eq. (2)
with no orbital-Zeeman coupling, treats the induction as uniform, and uses a
magnetic-periodic vortex-lattice unit cell.  This script uses the same free
energy and uniform-induction approximation, but it runs on a finite square
with natural (open) boundaries because magnetic-periodic boundaries are not
yet implemented by this solver.  Comparisons with the paper are therefore
qualitative until domain- and mesh-convergence checks have been performed.

For every alpha the script starts from the mixed state at low field, sweeps
``b = B / Bc2`` upward, and optionally sweeps downward from the final state.
It writes one HDF5 solution per point, crash-resilient CSV summaries, and
three plots:

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
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import tdgl  # noqa: E402


PAPER_ALPHA_STAR = 1 / 3
PAPER_HIGH_FIELD_ALPHA_MIN = 2 / 3
PAPER_BC2 = 1.0

MEASUREMENT_FIELDS = [
    "alpha",
    "direction",
    "sequence_index",
    "reduced_field",
    "applied_field",
    "field_units",
    "max_abs_d_prime",
    "bulk_max_abs_d_prime",
    "bulk_mean_abs_d_prime",
    "bulk_relative_phase",
    "free_energy",
    "free_energy_per_area",
    "convergence_max_density_change",
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
        return np.asarray(values, dtype=float)

    parts = specification.split(":")
    if len(parts) != 3:
        raise ValueError("Range grids must have the form start:stop:step.")
    start, stop, step = (float(value) for value in parts)
    if step <= 0 or stop < start:
        raise ValueError("Range grids require stop >= start and step > 0.")
    count = int(math.floor((stop - start) / step + 1e-12)) + 1
    values = start + step * np.arange(count, dtype=float)
    if values[-1] < stop - 1e-10:
        values = np.append(values, stop)
    return values


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
) -> float:
    """Area-weighted circular mean of arg(d') - arg(d) in the mixed bulk."""
    cross = d_prime * np.conj(d)
    valid = mask & (np.abs(d) >= amplitude_floor) & (
        np.abs(d_prime) >= amplitude_floor
    )
    if not np.any(valid):
        return math.nan
    unit_cross = cross[valid] / np.abs(cross[valid])
    mean = np.sum(areas[valid] * unit_cross)
    return float(np.angle(mean)) if abs(mean) else math.nan


def final_density_change(solution: tdgl.Solution) -> float:
    """Maximum amplitude-squared change between the last two saved states."""
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
) -> dict:
    solution.solve_step = -1
    mesh = solution.device.mesh
    d = solution.get_order_parameter("d")
    d_prime = solution.get_order_parameter("d_prime")
    bulk = interior_mask(mesh, width, boundary_strip)
    max_d_prime = float(np.max(np.abs(d_prime)))
    bulk_max_d_prime = float(np.max(np.abs(d_prime[bulk])))
    bulk_mean_d_prime = area_average(np.abs(d_prime), mesh.areas, bulk)
    phase = circular_relative_phase(
        d,
        d_prime,
        mesh.areas,
        bulk,
        amplitude_floor=phase_amplitude_floor,
    )
    free_energy = solver.compute_d_plus_d_prime_free_energy(d, d_prime)
    area = float(np.sum(mesh.areas))
    return {
        "alpha": alpha,
        "direction": direction,
        "sequence_index": sequence_index,
        "reduced_field": reduced_field,
        "applied_field": applied_field,
        "field_units": field_units,
        "max_abs_d_prime": max_d_prime,
        "bulk_max_abs_d_prime": bulk_max_d_prime,
        "bulk_mean_abs_d_prime": bulk_mean_d_prime,
        "bulk_relative_phase": phase,
        "free_energy": free_energy,
        "free_energy_per_area": free_energy / area,
        "convergence_max_density_change": final_density_change(solution),
        "num_sites": len(mesh.sites),
        "num_elements": len(mesh.elements),
        "width": width,
        "max_edge_length": max_edge_length,
        "boundary_strip": boundary_strip,
        "solve_time": solve_time,
        "wall_seconds": solution.total_seconds,
        "output_file": str(Path(solution.path).resolve()),
    }


def transition_bracket(rows: Sequence[dict], threshold: float) -> dict:
    """Find the first mixed/pure crossing in the supplied sweep order."""
    if not rows:
        return {
            "lower_field": math.nan,
            "upper_field": math.nan,
            "transition_field": math.nan,
            "field_uncertainty": math.nan,
            "status": "no_data",
        }
    present = [float(row["bulk_max_abs_d_prime"]) >= threshold for row in rows]
    for previous, current, previous_present, current_present in zip(
        rows[:-1], rows[1:], present[:-1], present[1:]
    ):
        if previous_present != current_present:
            lower, upper = sorted(
                [float(previous["reduced_field"]), float(current["reduced_field"])]
            )
            return {
                "lower_field": lower,
                "upper_field": upper,
                "transition_field": 0.5 * (lower + upper),
                "field_uncertainty": 0.5 * (upper - lower),
                "status": "bracketed",
            }
    if all(present):
        status = "mixed_at_all_fields"
    elif not any(present):
        status = "pure_at_all_fields"
    else:
        status = "no_unique_crossing"
    return {
        "lower_field": math.nan,
        "upper_field": math.nan,
        "transition_field": math.nan,
        "field_uncertainty": math.nan,
        "status": status,
    }


def build_transition_rows(measurements: Sequence[dict], thresholds: Iterable[float]):
    rows = []
    alphas = sorted({float(row["alpha"]) for row in measurements})
    directions = ("up", "down")
    for alpha in alphas:
        for direction in directions:
            sweep = sorted(
                (
                    row
                    for row in measurements
                    if float(row["alpha"]) == alpha
                    and row["direction"] == direction
                ),
                key=lambda row: int(row["sequence_index"]),
            )
            if not sweep:
                continue
            for threshold in thresholds:
                bracket = transition_bracket(sweep, threshold)
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

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(alphas)))
    for alpha, color in zip(alphas, colors):
        for direction, linestyle in (("up", "-"), ("down", "--")):
            sweep = sorted(
                (
                    row
                    for row in measurements
                    if float(row["alpha"]) == alpha
                    and row["direction"] == direction
                ),
                key=lambda row: float(row["reduced_field"]),
            )
            if sweep:
                ax.plot(
                    [float(row["reduced_field"]) for row in sweep],
                    [float(row["bulk_max_abs_d_prime"]) for row in sweep],
                    marker="o" if direction == "up" else None,
                    markersize=3,
                    linestyle=linestyle,
                    color=color,
                    label=fr"$\alpha={alpha:g}$ {direction}",
                )
    ax.axhline(
        primary_threshold,
        color="black",
        linestyle=":",
        label="transition threshold",
    )
    ax.set(xlabel=r"$b=B/B_{c2}$", ylabel=r"bulk $\max |d'|$")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7)
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
            [float(row["max_abs_d_prime"]) for row in curve],
            "o-",
            markersize=3,
            color=color,
            label=fr"$b={field:g}$",
        )
    ax.set(xlabel=r"$\alpha$", ylabel=r"$\max |d'|$")
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


def run_scan(args) -> tuple[list[dict], list[dict]]:
    alphas = parse_grid(args.alphas)
    fields = np.unique(parse_grid(args.fields))
    thresholds = parse_grid(args.thresholds)
    if args.smoke_test:
        alphas = np.asarray([0.5])
        fields = np.asarray([0.0, 0.6])
        args.width = 3.0
        args.max_edge_length = 0.8
        args.boundary_strip = 0.4
        args.solve_time = 0.002
        args.dt_init = 1e-4
        args.dt_max = 1e-3
        args.save_every = 1
        args.smooth = 2

    if np.any((alphas <= PAPER_ALPHA_STAR) | (alphas >= 1)):
        raise ValueError(
            "The field-driven mixed-to-pure transition requires 1/3 < alpha < 1."
        )
    if np.any((fields < 0) | (fields > PAPER_BC2)):
        raise ValueError("Reduced fields must lie in 0 <= b <= 1.")
    if np.any(thresholds <= 0):
        raise ValueError("Amplitude thresholds must be positive.")
    if not np.any(np.isclose(fields, 0.0)):
        fields = np.insert(fields, 0, 0.0)
    fields.sort()

    output_directory = Path(args.output_directory).resolve()
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
    measurements: list[dict] = []

    for alpha_index, alpha in enumerate(alphas):
        alpha = float(alpha)
        device.layer.model = tdgl.DPlusDPrimeModel(
            alpha=alpha,
            zeeman_coupling=0.0,
        )
        seed_solution = None
        last_up_measurement = None
        print(f"\nalpha={alpha:g} ({alpha_index + 1}/{len(alphas)})")
        for sequence_index, reduced_field in enumerate(fields):
            reduced_field = float(reduced_field)
            print(f"  up   b={reduced_field:.5g}")
            stem = f"alpha_{alpha:.6f}_up_{sequence_index:03d}_b_{reduced_field:.6f}.h5"
            solution, solver, applied_field = solve_point(
                device,
                seed_solution,
                reduced_field=reduced_field,
                output_file=h5_directory / stem,
                field_units=args.field_units,
                solve_time=args.solve_time,
                dt_init=args.dt_init,
                dt_max=args.dt_max,
                save_every=args.save_every,
                progress_interval=args.progress_interval,
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
            )
            measurements.append(row)
            last_up_measurement = row
            seed_solution = solution
            write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)

        if not args.no_down_sweep:
            # The maximum-field state is already converged and is the initial
            # point of the return branch, so record it without solving twice.
            down_start = dict(last_up_measurement)
            down_start["direction"] = "down"
            down_start["sequence_index"] = 0
            measurements.append(down_start)
            write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)
            for sequence_index, reduced_field in enumerate(fields[-2::-1], start=1):
                reduced_field = float(reduced_field)
                print(f"  down b={reduced_field:.5g}")
                stem = (
                    f"alpha_{alpha:.6f}_down_{sequence_index:03d}_"
                    f"b_{reduced_field:.6f}.h5"
                )
                solution, solver, applied_field = solve_point(
                    device,
                    seed_solution,
                    reduced_field=reduced_field,
                    output_file=h5_directory / stem,
                    field_units=args.field_units,
                    solve_time=args.solve_time,
                    dt_init=args.dt_init,
                    dt_max=args.dt_max,
                    save_every=args.save_every,
                    progress_interval=args.progress_interval,
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
                )
                measurements.append(row)
                seed_solution = solution
                write_csv(measurements_path, measurements, MEASUREMENT_FIELDS)

        transitions = build_transition_rows(measurements, thresholds)
        write_csv(transitions_path, transitions, TRANSITION_FIELDS)

    transitions = build_transition_rows(measurements, thresholds)
    write_csv(transitions_path, transitions, TRANSITION_FIELDS)
    plot_results(measurements, transitions, float(thresholds[0]), output_directory)
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
    parser.add_argument("--solve-time", type=float, default=200.0)
    parser.add_argument("--dt-init", type=float, default=1e-4)
    parser.add_argument("--dt-max", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--field-units", default="mT")
    parser.add_argument("--phase-amplitude-floor", type=float, default=1e-3)
    parser.add_argument(
        "--output-directory",
        default="my_scripts/d_plus_d_prime_phase_diagram",
    )
    parser.add_argument("--no-down-sweep", action="store_true")
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
        f"found {bracketed} threshold crossings. Results: {output_directory}"
    )


if __name__ == "__main__":
    main()
