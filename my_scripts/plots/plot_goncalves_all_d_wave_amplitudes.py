"""Plot ``|psi_d|`` for every available Goncalves field checkpoint."""

import argparse
import math
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


CHECKPOINT_PATTERN = re.compile(
    r"^goncalves_Ha_([-+0-9.]+)_(\d+)(?:-\d+)?\.h5$"
)


def checkpoint_files(result_directory):
    """Return the newest checkpoint file for each continuation step."""
    by_step = {}
    for path in result_directory.glob("goncalves_Ha_*.h5"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            print(f"Skipping unrecognized checkpoint: {path.name}")
            continue
        field = float(match.group(1))
        step = int(match.group(2))
        candidate = (path.stat().st_mtime_ns, path.name, field, path)
        if step not in by_step or candidate[:2] > by_step[step][:2]:
            by_step[step] = candidate
    return [
        (step, field, path)
        for step, (_, _, field, path) in sorted(by_step.items())
    ]


def up_sweep_only(checkpoints):
    """Keep checkpoints through the first occurrence of the maximum field."""
    if not checkpoints:
        return checkpoints
    fields = np.asarray([field for _, field, _ in checkpoints])
    peak_index = int(np.argmax(fields))
    return checkpoints[: peak_index + 1]


def load_amplitude(path):
    """Load the latest saved ``|psi_d|`` state and its mesh directly from HDF5."""
    with h5py.File(path, "r") as h5file:
        numbered_groups = sorted(
            (
                (int(name), group)
                for name, group in h5file["data"].items()
                if name.isdigit()
            ),
            key=lambda item: item[0],
        )
        state_groups = [item for item in numbered_groups if "psi2" in item[1]]
        if not state_groups:
            raise ValueError("checkpoint does not contain a saved psi2 state")
        _, state_group = state_groups[-1]
        equilibrium_group = numbered_groups[-1][1]
        equilibrium_reached = bool(
            equilibrium_group.attrs.get("equilibrium_reached", False)
        )

        if "mesh" in h5file:
            mesh = h5file["mesh"]
        else:
            mesh = h5file["solution/device/mesh"]
        sites = np.asarray(mesh["sites"])
        elements = np.asarray(mesh["elements"])
        amplitude = np.abs(np.asarray(state_group["psi2"]))
    return sites, elements, amplitude, equilibrium_reached


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot |psi_d| for all available Goncalves checkpoints."
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        required=True,
        help="Directory containing goncalves_Ha_*.h5 checkpoints.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help=(
            "Output PNG path (default: RESULT_DIRECTORY/"
            "goncalves_psi_d_amplitude_all_available_up.png)."
        ),
    )
    parser.add_argument(
        "--all-branches",
        action="store_true",
        help="Include down-sweep checkpoints as well as the up sweep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_directory = args.result_directory.resolve()
    checkpoints = checkpoint_files(result_directory)
    if not args.all_branches:
        checkpoints = up_sweep_only(checkpoints)
    if not checkpoints:
        raise SystemExit("No Goncalves checkpoints found.")

    states = []
    for step, field, path in checkpoints:
        try:
            sites, elements, amplitude, equilibrium_reached = load_amplitude(path)
        except (KeyError, OSError, ValueError) as exc:
            print(f"Skipping unreadable checkpoint {path.name}: {exc}")
            continue
        print(
            f"Loaded {path.name}: H/Hc2={field:g}, step={step}, "
            f"equilibrium={equilibrium_reached}"
        )
        states.append(
            (field, sites, elements, amplitude, equilibrium_reached)
        )
    if not states:
        raise SystemExit("No readable Goncalves checkpoints found.")

    columns = 5
    rows = math.ceil(len(states) / columns)
    maximum_amplitude = max(1.0, max(np.max(state[3]) for state in states))
    levels = np.linspace(0.0, maximum_amplitude, 101)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 3 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    contour = None
    for index, (field, sites, elements, amplitude, equilibrium_reached) in enumerate(
        states
    ):
        row, column = divmod(index, columns)
        ax = axes[row, column]
        contour = ax.tricontourf(
            sites[:, 0],
            sites[:, 1],
            elements,
            amplitude,
            levels=levels,
            cmap="turbo",
        )
        status = "" if equilibrium_reached else " (partial)"
        ax.set_title(rf"$H/H_{{c2}}={field:.1f}${status}", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if column == 0:
            ax.set_ylabel(r"$y/\xi$", fontsize=9)
        if row == rows - 1:
            ax.set_xlabel(r"$x/\xi$", fontsize=9)

    for index in range(len(states), rows * columns):
        axes.flat[index].set_visible(False)

    sweep = "field sweep" if args.all_branches else "up sweep"
    branch = "all states" if args.all_branches else "all downloaded states"
    fig.suptitle(
        rf"Goncalves {sweep}: $|\psi_d|$ for {branch} ($h=0.1$)",
        fontsize=15,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.91,
        bottom=0.055,
        top=0.94,
        wspace=0.09,
        hspace=0.20,
    )
    colorbar_axis = fig.add_axes([0.925, 0.14, 0.018, 0.72])
    colorbar = fig.colorbar(contour, cax=colorbar_axis)
    colorbar.set_label(r"$|\psi_d|$")
    colorbar.ax.tick_params(labelsize=8)

    output_file = (
        args.output_file.resolve()
        if args.output_file is not None
        else result_directory / "goncalves_psi_d_amplitude_all_available_up.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to '{output_file}'.")


if __name__ == "__main__":
    main()
