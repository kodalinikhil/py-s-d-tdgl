"""Run the Goncalves s+d simulation at a selected reduced field.

The material, geometry, mesh, and time-integration parameters match
``simulate_goncalves.py``.  The resulting figure shows the magnitude and phase
of both order-parameter components.  It also shows the magnetic field produced
by the simulated sheet current, evaluated at the upper film surface, and its
sum with the prescribed applied field.

The current s+d solver uses a prescribed vector potential and does not support
self-consistent magnetic screening.  Consequently, the current-induced field
shown here is a post-processing diagnostic and is not fed back into the TDGL
simulation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import tdgl


DEFAULT_REDUCED_FIELD = 0.25
DEFAULT_DEFECT_RADIUS = 0.5
FIELD_UNITS = "mT"


def make_device(
    max_edge_length: float = 0.2,
    defect_radius: float = DEFAULT_DEFECT_RADIUS,
) -> tdgl.Device:
    """Build the Goncalves square with a circular pinning hole at its center."""
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
    central_defect = tdgl.Polygon(
        "central_defect",
        points=tdgl.geometry.circle(defect_radius, points=24),
    )
    device = tdgl.Device(
        "XuRenTing_goncalves_square_with_defect",
        layer=layer,
        film=film,
        holes=[central_defect],
    )
    device.make_mesh(max_edge_length=max_edge_length)
    return device


def run_simulation(
    device: tdgl.Device,
    output_file: Path,
    reduced_field: float,
    solve_time: float = 200,
) -> tdgl.Solution:
    """Relax the Goncalves model in the selected prescribed field."""
    hc2 = device.Bc2.to(FIELD_UNITS).magnitude
    applied_field = reduced_field * hc2
    print(
        f"Solving at H/Hc2 = {reduced_field:.3f} "
        f"({applied_field:.6g} {FIELD_UNITS})"
    )

    options = tdgl.SolverOptions(
        solve_time=solve_time,
        dt_init=0.005,
        dt_max=0.05,
        adaptive=True,
        field_units=FIELD_UNITS,
        output_file=str(output_file),
    )
    return tdgl.solve(
        device,
        options=options,
        applied_vector_potential=tdgl.sources.ConstantField(
            applied_field,
            field_units=FIELD_UNITS,
            length_units=device.length_units,
        ),
    )


def current_induced_field(solution: tdgl.Solution) -> np.ndarray:
    """Return current-induced ``Bz / Bc2`` at the upper film surface.

    The thin-film current is represented by a sheet at ``z0``.  Evaluating at
    ``z0 + d/2`` avoids the sheet singularity and corresponds to the upper
    surface of the physical film.
    """
    device = solution.device
    sites = device.mesh.sites
    surface_z = device.layer.z0 + device.layer.thickness / 2
    zs = np.full(len(sites), surface_z)
    induced_field = solution.field_at_position(
        sites,
        zs=zs,
        vector=False,
        units=FIELD_UNITS,
        with_units=False,
    )
    hc2 = device.Bc2.to(FIELD_UNITS).magnitude
    return np.asarray(induced_field) / hc2


def add_level_curve_plot(
    fig,
    ax,
    mesh,
    values,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot filled level curves and isolines for site data."""
    values = np.asarray(values)
    data_min = float(np.nanmin(values)) if vmin is None else vmin
    data_max = float(np.nanmax(values)) if vmax is None else vmax
    if np.isclose(data_min, data_max):
        padding = max(abs(data_min) * 1e-6, np.finfo(float).eps)
        data_min -= padding
        data_max += padding
    levels = np.linspace(data_min, data_max, 16)

    filled = ax.tricontourf(
        mesh.sites[:, 0],
        mesh.sites[:, 1],
        mesh.elements,
        values,
        levels=levels,
        cmap=cmap,
        extend="both",
    )
    ax.tricontour(
        mesh.sites[:, 0],
        mesh.sites[:, 1],
        mesh.elements,
        values,
        levels=levels,
        colors="black",
        linewidths=0.45,
        alpha=0.55,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    fig.colorbar(filled, ax=ax, label=colorbar_label)


def add_phase_arrows(
    fig,
    ax,
    mesh,
    psi: np.ndarray,
    *,
    title: str,
    grid_size: int = 25,
    magnitude_cutoff: float = 1e-3,
):
    """Draw arrows ``(cos(arg(psi)), sin(arg(psi)))`` on a regular grid."""
    x, y = mesh.sites[:, 0], mesh.sites[:, 1]
    triangulation = mtri.Triangulation(x, y, mesh.elements)
    grid_x = np.linspace(float(x.min()), float(x.max()), grid_size)
    grid_y = np.linspace(float(y.min()), float(y.max()), grid_size)
    x_grid, y_grid = np.meshgrid(grid_x, grid_y)

    real = mtri.LinearTriInterpolator(triangulation, np.real(psi))(x_grid, y_grid)
    imag = mtri.LinearTriInterpolator(triangulation, np.imag(psi))(x_grid, y_grid)
    interpolated = np.ma.asarray(real) + 1j * np.ma.asarray(imag)
    magnitude = np.ma.abs(interpolated)
    phase = np.ma.angle(interpolated)
    cutoff = magnitude_cutoff * max(float(np.ma.max(magnitude)), np.finfo(float).eps)
    invalid = np.ma.getmaskarray(interpolated) | (np.asarray(magnitude) < cutoff)
    valid = ~invalid

    spacing = min(grid_x[1] - grid_x[0], grid_y[1] - grid_y[0])
    arrows = ax.quiver(
        x_grid[valid],
        y_grid[valid],
        np.cos(phase[valid]),
        np.sin(phase[valid]),
        phase[valid],
        cmap="twilight",
        clim=(-np.pi, np.pi),
        angles="xy",
        scale_units="xy",
        scale=1 / (0.72 * spacing),
        width=0.005,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        pivot="mid",
    )
    ax.set_title(title + r"; arrows $=(\cos\theta,\sin\theta)$")
    ax.set_aspect("equal")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    fig.colorbar(arrows, ax=ax, label=r"$\theta=\arg(\psi)$ (rad)")


def plot_solution(
    solution: tdgl.Solution,
    output_file: Path,
    reduced_field: float,
) -> None:
    """Plot both condensates and the applied/current-induced magnetic field."""
    mesh = solution.device.mesh
    data = solution.tdgl_data
    psi_d = data.psi_d
    psi_s = data.psi_s
    induced_field = current_induced_field(solution)
    total_field = reduced_field + induced_field

    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    add_level_curve_plot(
        fig,
        axes[0, 0],
        mesh,
        np.abs(psi_d),
        title=r"Magnitude $|\psi_d|$",
        colorbar_label=r"$|\psi_d|$",
        cmap="viridis",
    )
    add_level_curve_plot(
        fig,
        axes[0, 1],
        mesh,
        np.abs(psi_s),
        title=r"Magnitude $|\psi_s|$",
        colorbar_label=r"$|\psi_s|$",
        cmap="magma",
    )
    add_phase_arrows(
        fig,
        axes[1, 0],
        mesh,
        psi_d,
        title=r"Phase $\arg(\psi_d)$",
    )
    add_phase_arrows(
        fig,
        axes[1, 1],
        mesh,
        psi_s,
        title=r"Phase $\arg(\psi_s)$",
    )

    induced_limit = max(float(np.max(np.abs(induced_field))), np.finfo(float).eps)
    add_level_curve_plot(
        fig,
        axes[2, 0],
        mesh,
        induced_field,
        title=r"Current-induced field at the upper surface",
        colorbar_label=r"$B_{z,\mathrm{ind}}/B_{c2}$",
        cmap="coolwarm",
        vmin=-induced_limit,
        vmax=induced_limit,
    )
    total_deviation = max(
        float(np.max(np.abs(total_field - reduced_field))),
        np.finfo(float).eps,
    )
    add_level_curve_plot(
        fig,
        axes[2, 1],
        mesh,
        total_field,
        title=r"Applied + current-induced field at the upper surface",
        colorbar_label=r"$B_{z,\mathrm{total}}/B_{c2}$",
        cmap="coolwarm",
        vmin=reduced_field - total_deviation,
        vmax=reduced_field + total_deviation,
    )

    fig.suptitle(
        rf"Goncalves parameters at $H/H_{{c2}}={reduced_field:g}$"
        "\nInduced field is post-processed (no self-consistent screening)",
        fontsize=15,
    )
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to '{output_file}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Goncalves s+d simulation and plot both "
            "order parameters and the magnetic field."
        )
    )
    parser.add_argument(
        "--reduced-field",
        type=float,
        default=DEFAULT_REDUCED_FIELD,
        help=f"Applied H/Hc2 (default: {DEFAULT_REDUCED_FIELD:g}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the HDF5 result and PNG figure (default: this script's directory).",
    )
    parser.add_argument(
        "--solve-time",
        type=float,
        default=200,
        help="Dimensionless simulation time (default: 200, matching simulate_goncalves.py).",
    )
    parser.add_argument(
        "--max-edge-length",
        type=float,
        default=0.2,
        help="Maximum mesh edge length in units of xi (default: 0.2).",
    )
    parser.add_argument(
        "--defect-radius",
        type=float,
        default=DEFAULT_DEFECT_RADIUS,
        help=f"Central circular-hole radius in units of xi (default: {DEFAULT_DEFECT_RADIUS:g}).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the PNG from the existing HDF5 result without simulating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.solve_time <= 0:
        raise ValueError("--solve-time must be positive.")
    if args.max_edge_length <= 0:
        raise ValueError("--max-edge-length must be positive.")
    if args.defect_radius <= 0:
        raise ValueError("--defect-radius must be positive.")
    if args.reduced_field < 0:
        raise ValueError("--reduced-field must be nonnegative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    field_tag = f"{args.reduced_field:g}"
    defect_tag = f"{args.defect_radius:g}"
    output_stem = f"XuRenTing_defect_r{defect_tag}_H_{field_tag}"
    result_file = output_dir / f"{output_stem}.h5"
    if args.plot_only:
        if not result_file.exists():
            raise FileNotFoundError(f"Simulation result not found: {result_file}")
        solution = tdgl.Solution.from_hdf5(str(result_file))
    else:
        device = make_device(
            max_edge_length=args.max_edge_length,
            defect_radius=args.defect_radius,
        )
        solution = run_simulation(
            device,
            result_file,
            reduced_field=args.reduced_field,
            solve_time=args.solve_time,
        )
    plot_solution(
        solution,
        output_dir / f"{output_stem}.png",
        reduced_field=args.reduced_field,
    )


if __name__ == "__main__":
    main()
