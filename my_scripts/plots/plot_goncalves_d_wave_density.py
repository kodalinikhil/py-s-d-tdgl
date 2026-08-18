"""Simulate and plot ``|psi_d|`` at two fields from Goncalves et al.

Each target field is relaxed independently from the paper's initial state
``psi_d=1``, ``psi_s=0`` using the self-consistent electromagnetic equation.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tdgl

from my_scripts.plots.goncalves_plotting import make_paper_mesh


REDUCED_FIELDS = (1.040, 1.125)


def make_device():
    """Build the square and material parameters used by Goncalves et al."""
    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=2.0,
        thickness=0.1,
        model=tdgl.SPlusDModel(
            eta_s=2.0,
            eta_v=1.0,
            nu=-2.0,
            tau1=8 / 3,
            tau3=16 / 3,
            tau4=2.0,
            beta_em=1.0,
        ),
    )
    film = tdgl.Polygon("film", points=tdgl.geometry.box(8, 8)).resample(160)
    device = tdgl.Device("goncalves_square", layer=layer, film=film)
    points, triangles = make_paper_mesh()
    device._create_dimensionless_mesh(points, triangles)
    return device


def solve_at_fields(device, output_dir, solve_time):
    """Relax independent, self-consistent states at the requested fields."""
    field_units = "mT"
    hc2 = device.Bc2.to(field_units).magnitude
    solutions = []

    for reduced_field in REDUCED_FIELDS:
        applied_field = reduced_field * hc2
        output_file = output_dir / f"goncalves_d_wave_H_{reduced_field:.3f}.h5"
        print(
            f"--- Solving at H/Hc2 = {reduced_field:.3f} "
            f"({applied_field:.6g} {field_units}) ---",
            flush=True,
        )
        options = tdgl.SolverOptions(
            solve_time=solve_time,
            dt_init=2.5e-3,
            dt_max=2.5e-3,
            adaptive=True,
            include_screening=True,
            equilibrium_tolerance=1e-5,
            equilibrium_window=1000,
            save_every=10000,
            progress_interval=10000,
            pause_on_interrupt=False,
            field_units=field_units,
            output_file=str(output_file),
        )
        solution = tdgl.solve(
            device,
            options=options,
            applied_vector_potential=tdgl.sources.ConstantField(
                applied_field,
                field_units=field_units,
                length_units=device.length_units,
            ),
        )
        state = solution.tdgl_data.state
        if state.get("equilibrium_reached", False):
            print(
                f"Equilibrium reached at t={state['equilibrium_time']:.6g}; "
                f"error={state['equilibrium_error']:.3e}.",
                flush=True,
            )
        else:
            print(
                f"WARNING: H/Hc2={reduced_field:.3f} reached t={state['time']:.6g} "
                "without satisfying the stationary criterion.",
                flush=True,
            )
        solutions.append(solution)

    return solutions


def continue_to_time(output_dir, final_time):
    """Continue both existing field solutions to a cumulative final time."""
    pending = []
    for reduced_field in REDUCED_FIELDS:
        output_file = output_dir / f"goncalves_d_wave_H_{reduced_field:.3f}.h5"
        if not output_file.exists():
            raise FileNotFoundError(f"Cannot continue missing result: {output_file}")

        seed = tdgl.Solution.from_hdf5(str(output_file))
        initial_time = float(seed.tdgl_data.state["time"])
        initial_step = int(seed.tdgl_data.state["step"])
        remaining_time = final_time - initial_time
        if remaining_time <= 0:
            print(f"{output_file.name} is already at t={initial_time:g}.", flush=True)
            pending.append((output_file, None, initial_time, initial_step))
            continue

        temporary_file = output_file.with_name(f".{output_file.stem}.continuing.h5")
        if temporary_file.exists():
            temporary_file.unlink()
        print(
            f"--- Continuing H/Hc2 = {reduced_field:.3f} from "
            f"t={initial_time:g} to t={final_time:g} ---",
            flush=True,
        )
        options = tdgl.SolverOptions(
            solve_time=remaining_time,
            dt_init=2.5e-3,
            dt_max=2.5e-3,
            adaptive=True,
            include_screening=True,
            equilibrium_tolerance=1e-5,
            equilibrium_window=1000,
            save_every=1000,
            field_units="mT",
            output_file=str(temporary_file),
        )
        continued = tdgl.solve(
            seed.device,
            options=options,
            applied_vector_potential=seed.applied_vector_potential,
            seed_solution=seed,
        )
        pending.append((output_file, Path(continued.path), initial_time, initial_step))

    # Do not replace either checkpoint until both continuations have succeeded.
    solutions = []
    for output_file, temporary_file, initial_time, initial_step in pending:
        if temporary_file is not None:
            with h5py.File(temporary_file, "r+") as h5file:
                for group in h5file["data"].values():
                    group.attrs["time"] += initial_time
                    group.attrs["step"] += initial_step
                    if "equilibrium_reference_step" in group.attrs:
                        group.attrs["equilibrium_reference_step"] += initial_step
                    equilibrium_time = group.attrs.get("equilibrium_time")
                    if equilibrium_time is not None and np.isfinite(equilibrium_time):
                        group.attrs["equilibrium_time"] += initial_time
                option_attrs = h5file["solution/options"].attrs
                option_attrs["solve_time"] = final_time
                option_attrs["output_file"] = str(output_file)
            temporary_file.replace(output_file)
        solutions.append(tdgl.Solution.from_hdf5(str(output_file)))
    return solutions


def draw_amplitude(ax, solution, reduced_field, levels):
    """Draw the magnitude of the d-wave order parameter on one axis."""
    mesh = solution.device.mesh
    amplitude = np.abs(solution.tdgl_data.psi2)
    contour = ax.tricontourf(
        mesh.sites[:, 0],
        mesh.sites[:, 1],
        mesh.elements,
        amplitude,
        levels=levels,
        cmap="turbo",
        extend="max",
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    ax.set_title(rf"$H/H_{{c2}}={reduced_field:.3f}$")
    return contour


def save_amplitude_plots(solutions, output_dir):
    """Save individual amplitude plots and a side-by-side comparison."""
    maximum_amplitude = max(
        float(np.max(np.abs(solution.tdgl_data.psi2))) for solution in solutions
    )
    levels = np.linspace(0, max(1.0, maximum_amplitude), 101)

    for reduced_field, solution in zip(REDUCED_FIELDS, solutions):
        fig, ax = plt.subplots(figsize=(6, 5))
        contour = draw_amplitude(ax, solution, reduced_field, levels)
        fig.colorbar(contour, ax=ax, label=r"$|\psi_d|$")
        fig.tight_layout()
        output_file = output_dir / f"goncalves_d_wave_H_{reduced_field:.3f}.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Plot saved to '{output_file}'.", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, reduced_field, solution in zip(axes, REDUCED_FIELDS, solutions):
        contour = draw_amplitude(ax, solution, reduced_field, levels)
    fig.colorbar(contour, ax=axes, label=r"$|\psi_d|$", shrink=0.9)
    output_file = output_dir / "goncalves_d_wave_H_1.040_1.125.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Comparison plot saved to '{output_file}'.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate and plot |psi_d| at H/Hc2 = 1.040 and 1.125."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results/goncalves_density",
        help="Directory for the HDF5 results and PNG plots.",
    )
    parser.add_argument(
        "--solve-time",
        type=float,
        default=1000,
        help="Maximum dimensionless relaxation time (default: 1000).",
    )
    parser.add_argument(
        "--continue-to",
        type=float,
        help="Continue existing HDF5 states to this cumulative time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.continue_to is None:
        device = make_device()
        solutions = solve_at_fields(device, output_dir, args.solve_time)
    else:
        solutions = continue_to_time(output_dir, args.continue_to)
    save_amplitude_plots(solutions, output_dir)


if __name__ == "__main__":
    main()
