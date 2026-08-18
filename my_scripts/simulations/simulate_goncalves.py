import math
import sys
from pathlib import Path

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
        field_units="mT",
        output_file=str(output_file),
    )


def add_checkpoint_offsets(solution, initial_time, initial_step):
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
        option_attrs["solve_time"] = MAX_SOLVE_TIME
        option_attrs["output_file"] = str(path)
    return tdgl.Solution.from_hdf5(str(path))


def main():
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
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
    points, triangles = make_paper_mesh(side_length=8.0, spacing=0.2)
    device._create_dimensionless_mesh(points, triangles)

    # 3. Execution Protocol: Sweep Up and Down in units of Hc2.
    # ConstantField expects a dimensional magnetic field, so convert the
    # reduced field H_a / H_c2 to the SolverOptions field units at the API edge.
    field_units = "mT"
    hc2_in_field_units = device.Bc2.to(field_units).magnitude
    sweep = goncalves_field_sweep()

    seed_solution = None

    for i, reduced_Ha in enumerate(sweep):
        applied_field = reduced_Ha * hc2_in_field_units
        output_file = RESULT_DIRECTORY / f"goncalves_Ha_{reduced_Ha}_{i}.h5"
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
            if initial_time >= MAX_SOLVE_TIME:
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
                f"to t={MAX_SOLVE_TIME:g} ({i+1}/{len(sweep)}) ---"
            )
        else:
            print(
                f"--- Sweeping to Ha/Hc2 = {reduced_Ha} "
                f"({applied_field:.6g} {field_units}, {i+1}/{len(sweep)}) ---"
            )
        options = solver_options(MAX_SOLVE_TIME - initial_time, output_file)

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
            solution = add_checkpoint_offsets(solution, initial_time, initial_step)
        if not solution.tdgl_data.state.get("equilibrium_reached", False):
            print(
                f"WARNING: Ha/Hc2={reduced_Ha:g} reached the cumulative "
                f"solve-time cap {MAX_SOLVE_TIME:g} without converging; "
                "using the capped state to seed the next field."
            )
        seed_solution = solution

    # 4. Output: Plot the final return-leg state at Ha = 0.0
    print("Simulation complete. Generating plots for trapped flux state...")
    plot_spatial_state(
        seed_solution,
        sweep[-1],
        RESULT_DIRECTORY / "goncalves_trapped_flux.png",
        title_prefix="Final return-leg state",
    )


if __name__ == "__main__":
    main()
