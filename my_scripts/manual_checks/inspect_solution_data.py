import tdgl


def main():
    layer = tdgl.Layer(
        model=tdgl.SingleBandModel(),
        coherence_length=1,
        london_lambda=1,
        thickness=0.1,
    )
    film = tdgl.Polygon("film", points=tdgl.geometry.circle(1))
    device = tdgl.Device("test", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.5)

    options = tdgl.SolverOptions(
        solve_time=100, save_every=100, output_file="output.h5"
    )
    solution = tdgl.solve(device, options, applied_vector_potential=0)
    data = solution.tdgl_data
    print("psi1:", data.psi1 is not None)
    print("psi2:", data.psi2 is not None)


if __name__ == "__main__":
    main()
