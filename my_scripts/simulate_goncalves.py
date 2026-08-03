import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tdgl

from goncalves_plotting import plot_spatial_state


def main():
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

    # 2. Grid & Geometry
    width = 8
    box = tdgl.geometry.box(width, width)
    film = tdgl.Polygon("film", points=box).resample(400)
    device = tdgl.Device("goncalves_square", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.2)

    # 3. Execution Protocol: Sweep Up and Down in units of Hc2.
    # ConstantField expects a dimensional magnetic field, so convert the
    # reduced field H_a / H_c2 to the SolverOptions field units at the API edge.
    field_units = "mT"
    hc2_in_field_units = device.Bc2.to(field_units).magnitude
    ha_values_up = [step / 2 for step in range(6)]
    ha_values_down = ha_values_up[-2::-1]
    sweep = ha_values_up + ha_values_down

    seed_solution = None

    for i, reduced_Ha in enumerate(sweep):
        applied_field = reduced_Ha * hc2_in_field_units
        print(
            f"--- Sweeping to Ha/Hc2 = {reduced_Ha} "
            f"({applied_field:.6g} {field_units}, {i+1}/{len(sweep)}) ---"
        )
        options = tdgl.SolverOptions(
            solve_time=200,
            dt_init=0.005,
            dt_max=0.05,
            adaptive=True,
            field_units=field_units,
            output_file=f"goncalves_Ha_{reduced_Ha}_{i}.h5",
        )

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
        seed_solution = solution

    # 4. Output: Plot the final return-leg state at Ha = 0.0
    print("Simulation complete. Generating plots for trapped flux state...")
    plot_spatial_state(
        seed_solution,
        sweep[-1],
        "my_scripts/goncalves_trapped_flux.png",
        title_prefix="Final return-leg state",
    )


if __name__ == "__main__":
    main()
