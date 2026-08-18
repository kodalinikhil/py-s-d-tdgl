import argparse
import math
import signal
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import h5py
import tdgl

from my_scripts.plots.goncalves_plotting import (
    goncalves_field_sweep,
    make_paper_mesh,
    plot_spatial_state,
)


MAX_SOLVE_TIME = 1500.0
RESULT_DIRECTORY = REPOSITORY_ROOT / "results/goncalves"
STOP_REQUESTED = False
STOP_SIGNAL = None


def uniform_field_sweep(field_step, maximum_field=2.5):
    """Return a uniform up-and-down field sweep with exact decimal endpoints."""
    if field_step <= 0:
        raise ValueError("field_step must be positive.")
    if maximum_field <= 0:
        raise ValueError("maximum_field must be positive.")
    number_of_steps = int(round(maximum_field / field_step))
    if not np.isclose(number_of_steps * field_step, maximum_field):
        raise ValueError("maximum_field must be an integer multiple of field_step.")
    sweep_up = [round(i * field_step, 12) for i in range(number_of_steps + 1)]
    return sweep_up + sweep_up[-2::-1]


def request_stop(signum, _frame):
    """Turn a batch-scheduler signal into a solver-safe interruption."""
    global STOP_REQUESTED, STOP_SIGNAL
    STOP_REQUESTED = True
    STOP_SIGNAL = signum
    print(
        f"Received signal {signum}; saving the current field checkpoint.",
        file=sys.stderr,
        flush=True,
    )
    raise KeyboardInterrupt


def latest_checkpoint(output_file):
    """Return the newest base or versioned checkpoint for one sweep point."""
    output_file = Path(output_file)
    candidates = [output_file]
    candidates.extend(output_file.parent.glob(f"{output_file.stem}-*.h5"))
    existing = [path for path in candidates if path.exists()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def solver_options(solve_time, output_file):
    """Build the common options for one field-relaxation stage."""
    return tdgl.SolverOptions(
        solve_time=solve_time,
        # The paper's electromagnetic stability bound is
        # beta*h^2/(4*kappa^2)=0.0025 for h=0.2 and kappa=2.
        dt_init=2.5e-3,
        dt_max=2.5e-3,
        adaptive=True,
        include_screening=True,
        equilibrium_tolerance=1e-5,
        equilibrium_window=1000,
        equilibrium_min_time=0,
        save_every=10000,
        progress_interval=10000,
        pause_on_interrupt=False,
        field_units="mT",
        output_file=str(output_file),
    )


def add_checkpoint_offsets(solution, initial_time, initial_step, final_time):
    """Convert a continued checkpoint's local time and step to cumulative values."""
    path = Path(solution.path)
    with h5py.File(path, "r+") as h5file:
        for group in h5file["data"].values():
            group.attrs["time"] += initial_time
            group.attrs["step"] += initial_step
            if "equilibrium_reference_step" in group.attrs:
                group.attrs["equilibrium_reference_step"] += initial_step
            equilibrium_time = group.attrs.get("equilibrium_time")
            if equilibrium_time is not None and math.isfinite(equilibrium_time):
                group.attrs["equilibrium_time"] += initial_time
        option_attrs = h5file["solution/options"].attrs
        option_attrs["solve_time"] = final_time
        option_attrs["output_file"] = str(path)
    return tdgl.Solution.from_hdf5(str(path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Goncalves square field-continuation simulation."
    )
    parser.add_argument(
        "--mesh-spacing",
        type=float,
        default=0.2,
        help="Target mesh spacing in units of xi (default: 0.2).",
    )
    parser.add_argument(
        "--field-step",
        type=float,
        help=(
            "Use a uniform up-and-down field sweep with this spacing in H/Hc2. "
            "By default, use the paper-transition sweep."
        ),
    )
    parser.add_argument(
        "--maximum-field",
        type=float,
        default=2.5,
        help="Maximum H/Hc2 for a uniform sweep (default: 2.5).",
    )
    parser.add_argument(
        "--solve-time",
        type=float,
        default=MAX_SOLVE_TIME,
        help="Maximum relaxation time at each field (default: 1500).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=RESULT_DIRECTORY,
        help="Directory for checkpoints and the final trapped-flux plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mesh_spacing <= 0:
        raise ValueError("mesh_spacing must be positive.")
    if args.solve_time <= 0:
        raise ValueError("solve_time must be positive.")
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    for signal_name in ("SIGINT", "SIGTERM", "SIGUSR1", "SIGUSR2"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), request_stop)

    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=2.0,  # kappa = 2.0
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

    # 2. Grid & Geometry
    film = tdgl.Polygon("film", points=tdgl.geometry.box(8, 8)).resample(160)
    device = tdgl.Device("goncalves_square", layer=layer, film=film)
    points, triangles = make_paper_mesh(
        side_length=8.0, spacing=args.mesh_spacing
    )
    device._create_dimensionless_mesh(points, triangles)

    # 3. Execution Protocol: Sweep Up and Down in units of Hc2.
    # ConstantField expects a dimensional magnetic field, so convert the
    # reduced field H_a / H_c2 to the SolverOptions field units at the API edge.
    field_units = "mT"
    hc2_in_field_units = device.Bc2.to(field_units).magnitude
    sweep = (
        goncalves_field_sweep()
        if args.field_step is None
        else uniform_field_sweep(args.field_step, args.maximum_field)
    )
    print(
        f"Mesh spacing h={args.mesh_spacing:g}; {len(device.mesh.sites)} sites."
    )
    print(
        f"Field protocol: {len(sweep)} points from {sweep[0]:g} to "
        f"{max(sweep):g} and back to {sweep[-1]:g}."
    )

    seed_solution = None

    for i, reduced_Ha in enumerate(sweep):
        applied_field = reduced_Ha * hc2_in_field_units
        output_file = output_directory / f"goncalves_Ha_{reduced_Ha}_{i}.h5"
        checkpoint = latest_checkpoint(output_file)
        initial_time = 0.0
        initial_step = 0
        if checkpoint is not None:
            saved = tdgl.Solution.from_hdf5(str(checkpoint))
            state = saved.tdgl_data.state
            if state.get("equilibrium_reached", False):
                print(
                    f"--- Reusing converged Ha/Hc2 = {reduced_Ha} checkpoint "
                    f"'{checkpoint}' ({i+1}/{len(sweep)}) ---"
                )
                seed_solution = saved
                continue
            initial_time = float(state["time"])
            initial_step = int(state["step"])
            if initial_time >= args.solve_time:
                print(
                    f"WARNING: Reusing capped, unconverged Ha/Hc2 = "
                    f"{reduced_Ha} checkpoint '{checkpoint}' "
                    f"({i+1}/{len(sweep)})."
                )
                seed_solution = saved
                continue
            seed_solution = saved
            print(
                f"--- Continuing Ha/Hc2 = {reduced_Ha} from t={initial_time:g} "
                f"to t={args.solve_time:g} ({i+1}/{len(sweep)}) ---"
            )
        else:
            print(
                f"--- Sweeping to Ha/Hc2 = {reduced_Ha} "
                f"({applied_field:.6g} {field_units}, {i+1}/{len(sweep)}) ---"
            )
        options = solver_options(args.solve_time - initial_time, output_file)

        solution = tdgl.solve(
            device,
            options=options,
            applied_vector_potential=tdgl.sources.ConstantField(
                applied_field,
                field_units=field_units,
                length_units=device.length_units,
            ),
            seed_solution=seed_solution,
        )
        if initial_time:
            solution = add_checkpoint_offsets(
                solution, initial_time, initial_step, args.solve_time
            )
        if not solution.tdgl_data.state.get("equilibrium_reached", False):
            print(
                f"WARNING: Ha/Hc2={reduced_Ha:g} reached the cumulative "
                f"solve-time cap {args.solve_time:g} without converging; "
                "using the capped state to seed the next field."
            )
        seed_solution = solution
        if STOP_REQUESTED:
            print("Run interrupted after saving a resumable checkpoint.", flush=True)
            return 128 + (STOP_SIGNAL or signal.SIGINT)

    # 4. Output: Plot the final return-leg state at Ha = 0.0
    print("Simulation complete. Generating plots for trapped flux state...")
    plot_spatial_state(
        seed_solution,
        sweep[-1],
        output_directory / "goncalves_trapped_flux.png",
        title_prefix="Final return-leg state",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
