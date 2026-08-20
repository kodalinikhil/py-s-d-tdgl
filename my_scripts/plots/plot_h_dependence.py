import argparse
import glob
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tdgl
from matplotlib.ticker import MaxNLocator

from my_scripts.plots.goncalves_plotting import (
    MIN_CONDENSATE_DENSITY,
    area_average,
    gauge_covariant_winding,
    goncalves_field_sweep,
    magnetization_and_field,
    plot_spatial_state,
    stationary_gibbs_free_energy,
)

RESULT_DIRECTORY = REPOSITORY_ROOT / "results/goncalves"


def expected_sweep():
    """Return the field sequence used by ``simulate_goncalves.py``."""
    return goncalves_field_sweep()


def result_files(result_directory=RESULT_DIRECTORY, sweep=None):
    """Return the newest deterministic output file for each sweep step."""
    by_step = {}
    for filename in glob.glob(str(result_directory / "goncalves_Ha_*.h5")):
        stem = Path(filename).stem
        parts = stem.split("_")
        try:
            reduced_field = float(parts[2])
            step = int(parts[3].split("-")[0])
        except (IndexError, ValueError):
            print(f"Skipping unrecognized output filename: {filename}")
            continue
        if sweep is not None and (
            step >= len(sweep) or not np.isclose(reduced_field, sweep[step])
        ):
            print(f"Skipping file outside the configured field sweep: {filename}")
            continue
        candidate = (Path(filename).stat().st_mtime, reduced_field, filename)
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
    if peak_index < len(fields) - 1:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot field-dependent observables from Goncalves checkpoints."
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=RESULT_DIRECTORY,
        help="Directory containing goncalves_Ha_*.h5 checkpoints.",
    )
    parser.add_argument(
        "--infer-sweep",
        action="store_true",
        help="Infer the field protocol from checkpoint filenames.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output PNG path (default: RESULT_DIRECTORY/goncalves_h_dependence.png).",
    )
    parser.add_argument(
        "--title",
        default="Gonçalves field sweep (transition-refined continuation)",
        help="Figure title.",
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Do not create a spatial plot of the last readable state.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_directory = args.result_directory.resolve()
    sweep = None if args.infer_sweep else expected_sweep()
    files = result_files(result_directory, sweep)
    if not files:
        print("No data files found to plot!")
        return

    if sweep is None:
        print(f"Plotting {len(files)} available checkpoint steps.")
    else:
        print(
            f"Plotting {len(files)}/{len(sweep)} points from the "
            "configured Hc2 sweep."
        )

    reduced_fields = []
    avg_psi_d = []
    avg_psi_s = []
    vortex_core_psi_s = []
    magnetization = []
    vortex_counts = []
    free_energy = []
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
        if not data.state.get("equilibrium_reached", False):
            print(f"Skipping unconverged output {filename}.")
            continue
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
        free_energy.append(stationary_gibbs_free_energy(solution))

        if condensate_present:
            vortex_triangles = winding != 0
            vortex_counts.append(float(np.sum(np.abs(winding))))
            if np.any(vortex_triangles):
                core_values = np.mean(rho_s[triangles[vortex_triangles]], axis=1)
                vortex_core_psi_s.append(float(np.mean(core_values)))
            else:
                vortex_core_psi_s.append(np.nan)
        else:
            vortex_counts.append(np.nan)
            vortex_core_psi_s.append(np.nan)

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

    plot_branches(axs[4], reduced_fields, vortex_counts, peak_index)
    axs[4].set_xlabel(r"$H_a/H_{c2}$")
    axs[4].set_ylabel("Number of vortices")
    axs[4].set_title("Vortex count")
    axs[4].yaxis.set_major_locator(MaxNLocator(integer=True))
    annotated_up = set()
    previous_count = None
    for field, count in zip(
        reduced_fields[: peak_index + 1], vortex_counts[: peak_index + 1]
    ):
        if np.isfinite(count):
            integer_count = int(count)
            if integer_count == previous_count:
                continue
            previous_count = integer_count
            annotated_up.add((float(field), integer_count))
            axs[4].annotate(
                f"{integer_count}",
                (field, count),
                xytext=(-5, 6),
                textcoords="offset points",
                ha="right",
                fontsize=8,
            )
    previous_count = None
    for field, count in zip(
        reduced_fields[peak_index + 1 :], vortex_counts[peak_index + 1 :]
    ):
        if np.isfinite(count):
            integer_count = int(count)
            if integer_count == previous_count:
                continue
            previous_count = integer_count
            if (float(field), integer_count) in annotated_up:
                continue
            axs[4].annotate(
                f"{integer_count}",
                (field, count),
                xytext=(5, 6),
                textcoords="offset points",
                ha="left",
                fontsize=8,
            )

    plot_branches(axs[5], reduced_fields, free_energy, peak_index)
    axs[5].set_xlabel(r"$H_a/H_{c2}$")
    axs[5].set_ylabel(r"$F$")
    axs[5].set_title("Stationary Gibbs free energy (Eq. 79)")

    fig.suptitle(args.title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    result_directory.mkdir(parents=True, exist_ok=True)
    output_file = (
        args.output_file.resolve()
        if args.output_file is not None
        else result_directory / "goncalves_h_dependence.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to '{output_file}'.")

    if not args.skip_spatial:
        plot_spatial_state(
            spatial_solution,
            spatial_field,
            result_directory / "goncalves_trapped_flux.png",
            title_prefix="Last readable state",
        )


if __name__ == "__main__":
    main()
