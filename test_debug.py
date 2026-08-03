import tdgl
import numpy as np
from tdgl.geometry import box, circle
import numba

def test_screening_debug():
    numba.set_num_threads(2)
    
    length_units = "um"
    xi = 0.1
    london_lambda = 0.075
    thickness = 0.05
    height = 1
    width = 2
    
    layer = tdgl.Layer(
        coherence_length=xi, london_lambda=london_lambda, thickness=thickness
    )
    film = tdgl.Polygon("film", points=tdgl.geometry.box(width, height, points=301))
    device = tdgl.Device(
        "bar",
        layer=layer,
        film=film,
        length_units=length_units,
    )
    device.make_mesh(max_edge_length=xi / 2, smooth=100)
    
    options = tdgl.SolverOptions(
        solve_time=2,
        field_units="mT",
        current_units="uA",
        include_screening=True,
        monitor=False,
    )
    options.screening_tolerance = 1e-6
    options.dt_max = 1e-3
    
    screening_solution = tdgl.solve(device, options, applied_vector_potential=0.1)
    
    fluxoid_curves = [
        circle(0.25, center=(0, 0)),
        circle(0.1, center=(0.15, 0.25)),
        circle(0.3, center=(0.6, -0.1)),
        box(0.5, center=(-0.5, 0)),
        box(0.5, center=(-0.6, -0.2)),
    ]
    for curve in fluxoid_curves:
        fluxoid = screening_solution.polygon_fluxoid(curve)
        total_fluxoid = sum(fluxoid).magnitude
        error = abs(total_fluxoid / fluxoid.flux_part.magnitude)
        print("Error screening:", error)
        
test_screening_debug()
