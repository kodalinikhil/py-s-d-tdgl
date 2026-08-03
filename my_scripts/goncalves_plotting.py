"""Shared analysis and plotting helpers for the Goncalves field sweep."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Below this spatially averaged d-wave density, phase winding is dominated by
# numerical phase noise because the condensate has effectively collapsed.
MIN_CONDENSATE_DENSITY = 1e-4


def area_average(values, mesh, mask=None):
    """Return a Voronoi-area-weighted spatial average on an unstructured mesh."""
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    return float(np.average(np.asarray(values)[mask], weights=mesh.areas[mask]))


def interior_site_mask(mesh, boundary_layers=2):
    """Mask out boundary sites and nearby graph layers for derivative statistics."""
    excluded = np.zeros(len(mesh.sites), dtype=bool)
    excluded[mesh.boundary_indices] = True
    edges = mesh.edge_mesh.edges
    for _ in range(boundary_layers):
        touches_excluded = excluded[edges[:, 0]] | excluded[edges[:, 1]]
        excluded[edges[touches_excluded].ravel()] = True
    return ~excluded


def gauge_covariant_winding(solution):
    """Return signed d-wave phase winding on counterclockwise mesh triangles."""
    mesh = solution.device.mesh
    triangles = np.asarray(mesh.elements).copy()
    points = mesh.sites[triangles]
    signed_twice_area = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    clockwise = signed_twice_area < 0
    triangles[clockwise, 1], triangles[clockwise, 2] = (
        triangles[clockwise, 2].copy(),
        triangles[clockwise, 1].copy(),
    )

    edge_mesh = mesh.edge_mesh
    edge_lookup = {}
    for edge_index, (i, j) in enumerate(edge_mesh.edges):
        edge_lookup[(int(i), int(j))] = (edge_index, 1.0)
        edge_lookup[(int(j), int(i))] = (edge_index, -1.0)

    data = solution.tdgl_data
    vector_potential = np.asarray(data.applied_vector_potential)
    if data.induced_vector_potential is not None:
        vector_potential = vector_potential + np.asarray(data.induced_vector_potential)

    psi_d = data.psi2
    winding = np.zeros(len(triangles), dtype=int)
    for triangle_index, triangle in enumerate(triangles):
        covariant_circulation = 0.0
        magnetic_flux = 0.0
        for i, j in zip(triangle, np.roll(triangle, -1)):
            edge_index, orientation = edge_lookup[(int(i), int(j))]
            line_integral = orientation * np.dot(
                vector_potential[edge_index], edge_mesh.directions[edge_index]
            )
            covariant_circulation += np.angle(
                np.conjugate(psi_d[i]) * np.exp(-1j * line_integral) * psi_d[j]
            )
            magnetic_flux += line_integral
        winding[triangle_index] = int(
            np.rint((covariant_circulation + magnetic_flux) / (2 * np.pi))
        )

    centers = np.mean(mesh.sites[triangles], axis=1)
    return winding, centers, triangles


def magnetization_and_field(solution, reduced_field):
    """Return dimensionless ``(H_a / H_c2, M_z / H_c2)``."""
    device = solution.device
    ureg = device.ureg
    moment = solution.magnetic_moment(units="A * m**2")
    film_area = (
        np.sum(device.mesh.areas) * device.coherence_length**2
    ).to("m**2")
    film_volume = (film_area * device.thickness).to("m**3")
    magnetization = (moment / film_volume).to("A / m")
    H_c2 = (device.Bc2 / ureg("mu_0")).to("A / m")
    reduced_magnetization = (magnetization / H_c2).to_base_units().magnitude
    return float(reduced_field), float(reduced_magnetization)


def plot_spatial_state(solution, reduced_field, output_file, title_prefix):
    """Plot densities, current vorticity, and supercurrent for one solution."""
    data = solution.tdgl_data
    mesh = solution.device.mesh
    x, y = mesh.sites[:, 0], mesh.sites[:, 1]
    rho_d = np.abs(data.psi2) ** 2
    rho_s = np.abs(data.psi1) ** 2
    winding, vortex_centers, _ = gauge_covariant_winding(solution)
    condensate_present = area_average(rho_d, mesh) >= MIN_CONDENSATE_DENSITY

    fig, axs = plt.subplots(2, 2, figsize=(11, 10))

    sc1 = axs[0, 0].tricontourf(
        x, y, mesh.elements, rho_d, levels=100, cmap="viridis"
    )
    if condensate_present:
        positive = winding > 0
        negative = winding < 0
        axs[0, 0].scatter(
            vortex_centers[positive, 0],
            vortex_centers[positive, 1],
            marker="o",
            s=28,
            facecolors="none",
            edgecolors="white",
            linewidths=1.0,
            label="vortex",
        )
        if np.any(negative):
            axs[0, 0].scatter(
                vortex_centers[negative, 0],
                vortex_centers[negative, 1],
                marker="x",
                s=28,
                color="red",
                linewidths=1.0,
                label="antivortex",
            )
        if np.any(winding != 0):
            axs[0, 0].legend(loc="best")
    axs[0, 0].set_title(r"$|\psi_d|^2$ with phase-winding cores")
    fig.colorbar(sc1, ax=axs[0, 0])

    sc2 = axs[0, 1].tricontourf(
        x, y, mesh.elements, rho_s, levels=100, cmap="plasma"
    )
    axs[0, 1].set_title(r"$|\psi_s|^2$")
    fig.colorbar(sc2, ax=axs[0, 1])

    vorticity_scale = solution.device.K0 / solution.device.coherence_length
    vorticity = (
        solution.vorticity / vorticity_scale
    ).to_base_units().magnitude
    interior = interior_site_mask(mesh)
    vmax = float(np.percentile(np.abs(vorticity[interior]), 99.5))
    vmax = max(vmax, np.finfo(float).eps)
    sc3 = axs[1, 0].tricontourf(
        x,
        y,
        mesh.elements,
        vorticity,
        levels=np.linspace(-vmax, vmax, 101),
        cmap="coolwarm",
        extend="both",
    )
    axs[1, 0].set_title(r"Reduced current vorticity $(\nabla\times\mathbf{k})_z$")
    fig.colorbar(sc3, ax=axs[1, 0], label=r"$(\nabla\times\mathbf{K})_z/(K_0/\xi)$")

    axs[1, 1].tricontourf(
        x, y, mesh.elements, rho_d, levels=100, cmap="viridis"
    )
    current = (
        solution.supercurrent_density / solution.device.K0
    ).to_base_units().magnitude
    current_magnitude = np.linalg.norm(current, axis=1)
    display_limit = max(
        float(np.percentile(current_magnitude[interior], 95)),
        np.finfo(float).eps,
    )
    display_current = current / np.maximum(
        1.0, current_magnitude / display_limit
    )[:, np.newaxis]
    stride = max(1, len(x) // 350)
    sample = np.arange(0, len(x), stride)
    axs[1, 1].quiver(
        x[sample],
        y[sample],
        display_current[sample, 0],
        display_current[sample, 1],
        color="white",
        alpha=0.75,
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=3.0 * display_limit,
        width=0.003,
    )
    axs[1, 1].set_title(r"Reduced sheet supercurrent $\mathbf{K}_s/K_0$")

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/\xi$")
        ax.set_ylabel(r"$y/\xi$")

    fig.suptitle(f"{title_prefix}: $H_a/H_{{c2}}={reduced_field:.3f}$")
    fig.tight_layout()
    output_file = Path(output_file)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to '{output_file}'.")
