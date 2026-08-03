"""Simulate and plot ``|psi_d|`` at two fields from Goncalves et al.

The material, geometry, mesh, and solver parameters match
``simulate_goncalves.py``. Each target field is solved independently from the
solver's zero-field initial condensate state; no field sweep is performed.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import tdgl


REDUCED_FIELDS = (1.040, 1.125)


def make_device():
    """Build the device used by ``simulate_goncalves.py``."""
    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=2.0,  # kappa = 2.0
        thickness=0.1,
        model=tdgl.SPlusDModel(
            eta_s=2.0,
            eta_v=1.0,
            nu=-2.0,
            tau1=2.66,
            tau3=5.33,
            tau4=2.0,
            beta_em=1.0,
        ),
    )

    width = 8
    box = tdgl.geometry.box(width, width)
    film = tdgl.Polygon("film", points=box).resample(400)
    device = tdgl.Device("goncalves_square", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.2)
    return device


def solve_at_fields(device, output_dir):
    """Jump independently from the initial state to each target field."""
    field_units = "mT"
    hc2_in_field_units = device.Bc2.to(field_units).magnitude
    solutions = []

    for reduced_field in REDUCED_FIELDS:
        applied_field = reduced_field * hc2_in_field_units
        output_file = output_dir / f"goncalves_d_wave_H_{reduced_field:.3f}.h5"
        print(
            f"--- Solving at H/Hc2 = {reduced_field:.3f} "
            f"({applied_field:.6g} {field_units}) ---"
        )
        options = tdgl.SolverOptions(
            solve_time=200,
            dt_init=0.005,
            dt_max=0.05,
            adaptive=True,
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
            seed_solution=None,
        )
        solutions.append(solution)

    return solutions


def draw_density(ax, solution, reduced_field, levels):
    """Draw ``|psi_d|`` on one axis and return the contour set."""
    mesh = solution.device.mesh
    x, y = mesh.sites[:, 0], mesh.sites[:, 1]
    d_wave_density = np.abs(solution.tdgl_data.psi2)
    contour = ax.tricontourf(
        x,
        y,
        mesh.elements,
        d_wave_density,
        levels=levels,
        cmap="turbo",
        extend="max",
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    ax.set_title(rf"$H/H_{{c2}}={reduced_field:.3f}$")
    return contour


def save_density_plots(solutions, output_dir):
    """Save individual density plots and a side-by-side comparison."""
    maximum_density = max(
        float(np.max(np.abs(solution.tdgl_data.psi2))) for solution in solutions
    )
    levels = np.linspace(0, max(1.0, maximum_density), 101)

    for reduced_field, solution in zip(REDUCED_FIELDS, solutions):
        fig, ax = plt.subplots(figsize=(6, 5))
        contour = draw_density(ax, solution, reduced_field, levels)
        fig.colorbar(contour, ax=ax, label=r"$|\psi_d|$")
        fig.tight_layout()
        output_file = output_dir / f"goncalves_d_wave_H_{reduced_field:.3f}.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Plot saved to '{output_file}'.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, reduced_field, solution in zip(axes, REDUCED_FIELDS, solutions):
        contour = draw_density(ax, solution, reduced_field, levels)
    fig.colorbar(contour, ax=axes, label=r"$|\psi_d|$", shrink=0.9)
    output_file = output_dir / "goncalves_d_wave_H_1.040_1.125.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Comparison plot saved to '{output_file}'.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot |psi_d| at H/Hc2 = 1.040 and 1.125."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for HDF5 results and PNG plots (default: this script's directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = make_device()
    solutions = solve_at_fields(device, output_dir)
    save_density_plots(solutions, output_dir)


if __name__ == "__main__":
    main()
