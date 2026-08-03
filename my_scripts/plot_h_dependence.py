import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import tdgl

from goncalves_plotting import (
    MIN_CONDENSATE_DENSITY,
    area_average,
    gauge_covariant_winding,
    interior_site_mask,
    magnetization_and_field,
    plot_spatial_state,
)


def result_files():
    """Return one deterministic output file per sweep step."""
    by_step = {}
    for filename in glob.glob("goncalves_Ha_*.h5"):
        stem = os.path.basename(filename).removesuffix(".h5")
        parts = stem.split("_")
        try:
            reduced_field = float(parts[2])
            step = int(parts[3].split("-")[0])
        except (IndexError, ValueError):
            print(f"Skipping unrecognized output filename: {filename}")
            continue
        candidate = (os.path.getmtime(filename), reduced_field, filename)
        if step not in by_step or candidate[0] > by_step[step][0]:
            by_step[step] = candidate
    return [
        (step, reduced_field, filename)
        for step, (_, reduced_field, filename) in sorted(by_step.items())
    ]


def plot_branches(ax, fields, values, peak_index, **kwargs):
    """Plot the up and down branches, sharing the maximum-field point."""
    ax.plot(
        fields[: peak_index + 1],
        values[: peak_index + 1],
        "o-",
        color="blue",
        label="Sweep up",
        **kwargs,
    )
    ax.plot(
        fields[peak_index:],
        values[peak_index:],
        "s--",
        color="red",
        label="Sweep down",
        **kwargs,
    )
    ax.legend()
    ax.grid(True)


def main():
    files = result_files()
    if not files:
        print("No data files found to plot!")
        return

    reduced_fields = []
    avg_psi_d = []
    avg_psi_s = []
    vortex_core_psi_s = []
    magnetization = []
    net_vortices = []
    rms_vorticity = []
    spatial_solution = None
    spatial_field = None

    for step, field, filename in files:
        print(f"Loading {filename} (H_a/H_c2={field}, step={step})...")
        try:
            solution = tdgl.Solution.from_hdf5(filename)
        except (KeyError, OSError, RuntimeError) as exc:
            print(f"Skipping unreadable output {filename}: {exc}")
            continue

        data = solution.tdgl_data
        mesh = solution.device.mesh
        rho_d = np.abs(data.psi2) ** 2
        rho_s = np.abs(data.psi1) ** 2
        mean_rho_d = area_average(rho_d, mesh)
        winding, _, triangles = gauge_covariant_winding(solution)
        condensate_present = mean_rho_d >= MIN_CONDENSATE_DENSITY

        avg_psi_d.append(mean_rho_d)
        avg_psi_s.append(area_average(rho_s, mesh))
        reduced_field, M_value = magnetization_and_field(solution, field)
        reduced_fields.append(reduced_field)
        magnetization.append(M_value)

        if condensate_present:
            vortex_triangles = winding != 0
            net_vortices.append(float(np.sum(winding)))
            if np.any(vortex_triangles):
                core_values = np.mean(rho_s[triangles[vortex_triangles]], axis=1)
                vortex_core_psi_s.append(float(np.mean(core_values)))
            else:
                vortex_core_psi_s.append(np.nan)
        else:
            net_vortices.append(np.nan)
            vortex_core_psi_s.append(np.nan)

        vorticity_scale = solution.device.K0 / solution.device.coherence_length
        vorticity = (
            solution.vorticity / vorticity_scale
        ).to_base_units().magnitude
        interior = interior_site_mask(mesh)
        rms_vorticity.append(
            np.sqrt(area_average(vorticity**2, mesh, mask=interior))
        )
        spatial_solution = solution
        spatial_field = field

    if not reduced_fields:
        print("No readable data files found to plot!")
        return

    reduced_fields = np.asarray(reduced_fields)
    peak_index = int(np.argmax(reduced_fields))

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs = axs.ravel()

    plot_branches(axs[0], reduced_fields, avg_psi_d, peak_index)
    axs[0].set_xlabel(r"$H_a/H_{c2}$")
    axs[0].set_ylabel(r"$\langle |\psi_d|^2 \rangle_A$")
    axs[0].set_title("Area-averaged d-wave density")

    plot_branches(axs[1], reduced_fields, avg_psi_s, peak_index)
    axs[1].set_xlabel(r"$H_a/H_{c2}$")
    axs[1].set_ylabel(r"$\langle |\psi_s|^2 \rangle_A$")
    axs[1].set_title("Area-averaged s-wave density")

    plot_branches(axs[2], reduced_fields, vortex_core_psi_s, peak_index)
    axs[2].set_xlabel(r"$H_a/H_{c2}$")
    axs[2].set_ylabel(r"Mean $|\psi_s|^2$ on vortex plaquettes")
    axs[2].set_title("Vortex-associated s-wave density")

    plot_branches(axs[3], reduced_fields, magnetization, peak_index)
    axs[3].set_xlabel(r"$H_a/H_{c2}$")
    axs[3].set_ylabel(r"$M_z/H_{c2}$")
    axs[3].set_title(r"Magnetization loop $M_z(H_a)$")

    plot_branches(axs[4], reduced_fields, net_vortices, peak_index)
    axs[4].set_xlabel(r"$H_a/H_{c2}$")
    axs[4].set_ylabel(r"Net phase winding $N_+-N_-$")
    axs[4].set_title("Topological vortex number")

    plot_branches(axs[5], reduced_fields, rms_vorticity, peak_index)
    axs[5].set_xlabel(r"$H_a/H_{c2}$")
    axs[5].set_ylabel(r"RMS $(\nabla\times\mathbf{K})_z/(K_0/\xi)$")
    axs[5].set_title("Bulk current vorticity")

    fig.tight_layout()
    output_file = "my_scripts/goncalves_h_dependence.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to '{output_file}'.")

    plot_spatial_state(
        spatial_solution,
        spatial_field,
        "my_scripts/goncalves_trapped_flux.png",
        title_prefix="Last readable return-leg state",
    )


if __name__ == "__main__":
    main()
