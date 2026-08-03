import h5py
import numpy as np
import tdgl

layer = tdgl.Layer(model=tdgl.SingleBandModel(), coherence_length=1, london_lambda=1, thickness=0.1)
film = tdgl.Polygon("film", points=tdgl.geometry.circle(1))
device = tdgl.Device("test", layer=layer, film=film)
device.make_mesh(max_edge_length=0.5)

options = tdgl.SolverOptions(solve_time=1, save_every=1)
solution = tdgl.solve(device, options, applied_vector_potential=0)
with h5py.File(solution.path, "r") as f:
    print(list(f["data"]["0"].keys()))
