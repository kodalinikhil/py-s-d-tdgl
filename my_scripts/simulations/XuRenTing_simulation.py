"""Plot the Xu--Ren--Ting observables from a Goncalves field-sweep state.

The default input is the existing ``H/Hc2 = 0.5`` down-sweep checkpoint from
``simulate_goncalves.py``.  The resulting figure shows the magnitude and phase
of both order-parameter components and the self-consistent local induction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import tdgl
from tdgl.finite_volume.operators import build_magnetic_field_curl


DEFAULT_INPUT_FILE = REPOSITORY_ROOT / "results/goncalves/goncalves_Ha_0.5_24.h5"
DEFAULT_OUTPUT_FILE = (
    REPOSITORY_ROOT / "results/plots/XuRenTing_goncalves_downsweep_H_0.5.png"
)
DEFAULT_REDUCED_FIELD = 0.5


def current_induced_field(solution: tdgl.Solution) -> np.ndarray:
    """Return the local self-consistent ``(B-H) / Bc2`` on mesh sites."""
    mesh = solution.device.mesh
    curl_x, curl_y = build_magnetic_field_curl(mesh)
    data = solution.tdgl_data
    applied = np.asarray(data.applied_vector_potential)
    total = applied + np.asarray(data.induced_vector_potential)
    total_field = curl_x @ total[:, 0] + curl_y @ total[:, 1]
    applied_field = curl_x @ applied[:, 0] + curl_y @ applied[:, 1]
    return np.asarray(total_field - applied_field)


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
        title=r"Self-consistent induced field $B-H$",
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
        title=r"Self-consistent local induction",
        colorbar_label=r"$B_{z,\mathrm{total}}/B_{c2}$",
        cmap="coolwarm",
        vmin=reduced_field - total_deviation,
        vmax=reduced_field + total_deviation,
    )

    fig.suptitle(
        rf"Goncalves down-sweep state at $H/H_{{c2}}={reduced_field:g}$",
        fontsize=15,
    )
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to '{output_file}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot both order parameters and the magnetic field from an "
            "existing Goncalves field-sweep checkpoint."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input HDF5 checkpoint (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--reduced-field",
        type=float,
        default=DEFAULT_REDUCED_FIELD,
        help=f"Applied H/Hc2 used for plot labels (default: {DEFAULT_REDUCED_FIELD:g}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reduced_field < 0:
        raise ValueError("--reduced-field must be nonnegative.")
    input_file = args.input_file.resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Simulation result not found: {input_file}")
    solution = tdgl.Solution.from_hdf5(str(input_file))
    if solution.device.holes:
        raise ValueError(f"Expected a defect-free checkpoint, got holes in {input_file}.")
    if not solution.tdgl_data.state.get("equilibrium_reached", False):
        raise ValueError(f"Checkpoint has not reached equilibrium: {input_file}")

    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_solution(
        solution,
        output_file,
        reduced_field=args.reduced_field,
    )


if __name__ == "__main__":
    main()
