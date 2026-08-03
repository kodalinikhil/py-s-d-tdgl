import h5py
import numpy as np
import pytest

import tdgl
from tdgl.device.models import SPlusDModel
from tdgl.finite_volume.operators import MeshOperators
from tdgl.geometry import box
from tdgl.solver.options import SolverOptions, SparseSolver
from tdgl.solver.runner import Runner
from tdgl.solver.solver import TDGLSolver


@pytest.fixture(scope="module")
def mesh():
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
    )
    film = tdgl.Polygon("film", points=box(4)).resample(101)
    device = tdgl.Device("s-plus-d-test", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.35, smooth=20)
    return device.mesh


@pytest.fixture(scope="module")
def operators(mesh):
    operators = MeshOperators(
        mesh,
        SparseSolver.SUPERLU,
        fixed_sites=np.array([], dtype=int),
        fix_psi=False,
    )
    operators.build_operators()
    operators.set_link_exponents(np.zeros((len(mesh.edge_mesh.edges), 2)))
    return operators


def make_solver(operators, model, *, adaptive=False):
    solver = TDGLSolver.__new__(TDGLSolver)
    solver.model = model
    solver.options = SolverOptions(
        solve_time=1,
        dt_init=1e-3,
        dt_max=1e-3,
        adaptive=adaptive,
    )
    solver.operators = operators
    solver.terminal_psi = None
    solver.normal_boundary_index = np.array([], dtype=int)
    solver.use_cupy = False
    solver.mu_boundary = np.zeros(len(operators.mesh.edge_mesh.boundary_edge_indices))
    solver.dA_boundary_normal = np.zeros_like(solver.mu_boundary)
    return solver


def test_standard_mode_is_default():
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
    )
    assert isinstance(layer.model, tdgl.SingleBandModel)


def test_standard_gamma_shortcut_and_multicomponent_warning():
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        gamma=3.5,
    )
    assert layer.model == tdgl.SingleBandModel(gamma=3.5)
    with pytest.warns(UserWarning, match="KWT gamma"):
        mixed_layer = tdgl.Layer(
            coherence_length=1,
            london_lambda=2,
            thickness=0.1,
            model=SPlusDModel(),
            gamma=3.5,
        )
    assert mixed_layer.model == SPlusDModel()


def test_standard_mode_short_solve_and_storage(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
    )
    film = tdgl.Polygon("film", points=box(2)).resample(41)
    device = tdgl.Device("single-band-smoke", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.5, smooth=10)
    path = tmp_path / "single-band.h5"
    solution = tdgl.solve(
        device,
        SolverOptions(
            solve_time=2e-3,
            dt_init=1e-6,
            dt_max=1e-3,
            adaptive=True,
            terminal_psi=None,
            save_every=10,
            progress_interval=1,
            output_file=str(path),
        ),
    )
    assert isinstance(solution.device.layer.model, tdgl.SingleBandModel)
    assert np.all(np.isfinite(solution.tdgl_data.psi_s))
    assert np.all(solution.tdgl_data.psi_d == 0)
    assert solution.tdgl_data.state["step"] < 100
    with h5py.File(solution.path, "r") as h5file:
        last_step = h5file["data"][str(len(h5file["data"]) - 1)]
        assert "psi1" in last_step
        assert "psi2" in last_step


def test_s_plus_d_prescribed_time_dependent_a(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=SPlusDModel(
            eta_s=0.8,
            eta_v=0.2,
            nu=-1,
            tau1=1,
            beta_em=2,
        ),
    )
    film = tdgl.Polygon("film", points=box(2)).resample(41)
    device = tdgl.Device("mixed-prescribed-a", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.5, smooth=10)
    ramped_field = tdgl.sources.LinearRamp(tmin=0, tmax=3e-3) * (
        tdgl.sources.ConstantField(1)
    )
    solution = tdgl.solve(
        device,
        SolverOptions(
            solve_time=4e-3,
            dt_init=1e-3,
            dt_max=1e-3,
            adaptive=False,
            terminal_psi=None,
            save_every=10,
            progress_interval=1,
            output_file=str(tmp_path / "mixed-prescribed-a.h5"),
        ),
        applied_vector_potential=ramped_field,
    )
    data = solution.tdgl_data
    assert np.all(np.isfinite(data.psi_d))
    assert np.all(np.isfinite(data.psi_s))
    assert np.all(np.isfinite(data.mu))
    assert np.max(np.abs(data.mu)) > 1e-12


def test_legacy_layer_model_migration(tmp_path):
    path = tmp_path / "legacy-layer.h5"
    with h5py.File(path, "w") as h5file:
        single = h5file.create_group("single")
        single.attrs.update(
            london_lambda=2,
            coherence_length=1,
            thickness=0.1,
            u=5.79,
            gamma=3.5,
        )
        mixed = h5file.create_group("mixed")
        mixed.attrs.update(
            london_lambda=2,
            coherence_length=1,
            thickness=0.1,
            u=5.79,
            gamma_s=0.8,
            alpha_s=-0.5,
            beta_s=1.2,
            gamma_1=0.7,
            gamma_2=0.3,
            epsilon=0.2,
        )
        single_layer = tdgl.Layer.from_hdf5(single)
        mixed_layer = tdgl.Layer.from_hdf5(mixed)
    assert single_layer.model == tdgl.SingleBandModel(gamma=3.5)
    assert mixed_layer.model == SPlusDModel(
        eta_s=0.8,
        eta_v=0.2,
        nu=-0.5,
        tau1=1.2,
        tau3=1.4,
        tau4=0.3,
        beta_em=1,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta_s": 0},
        {"beta_em": 0},
        {"eta_s": 0.04, "eta_v": 0.2},
        {"tau1": 0},
    ],
)
def test_model_validation(kwargs):
    with pytest.raises(ValueError):
        SPlusDModel(**kwargs).validate()


def test_directional_operators_sum_to_covariant_laplacian(operators):
    difference = operators.laplacian_x + operators.laplacian_y
    difference = difference - operators.psi_laplacian
    assert np.max(np.abs(difference.data), initial=0) < 1e-12


def test_discrete_gauge_covariance_and_current_invariance(mesh, operators):
    rng = np.random.default_rng(7)
    psi_d = rng.normal(size=len(mesh.sites)) + 1j * rng.normal(size=len(mesh.sites))
    psi_s = rng.normal(size=len(mesh.sites)) + 1j * rng.normal(size=len(mesh.sites))
    laplacians = [
        operator @ psi_d
        for operator in (
            operators.psi_laplacian,
            operators.laplacian_x,
            operators.laplacian_y,
        )
    ]
    current = operators.get_s_plus_d_supercurrent(psi_d, psi_s, eta_s=0.8, eta_v=0.2)

    chi = 0.17 * mesh.sites[:, 0] - 0.11 * mesh.sites[:, 1]
    edges = mesh.edge_mesh.edges
    directions = mesh.edge_mesh.directions
    delta_chi = chi[edges[:, 1]] - chi[edges[:, 0]]
    transformed_a = delta_chi[:, None] * directions
    transformed_a /= np.sum(directions**2, axis=1)[:, None]
    phase = np.exp(1j * chi)
    operators.set_link_exponents(transformed_a)
    transformed_laplacians = [
        operator @ (phase * psi_d)
        for operator in (
            operators.psi_laplacian,
            operators.laplacian_x,
            operators.laplacian_y,
        )
    ]
    transformed_current = operators.get_s_plus_d_supercurrent(
        phase * psi_d, phase * psi_s, eta_s=0.8, eta_v=0.2
    )

    for before, after in zip(laplacians, transformed_laplacians):
        assert np.allclose(after, phase * before, atol=2e-11, rtol=2e-11)
    assert np.allclose(transformed_current, current, atol=2e-11, rtol=2e-11)
    operators.set_link_exponents(np.zeros_like(transformed_a))


def test_mixed_gradient_has_opposite_x_y_sign(mesh, operators):
    eta_v = 0.2
    x = mesh.sites[:, 0]
    y = mesh.sites[:, 1]
    center = np.argmin(x**2 + y**2)
    mixed_from_x = eta_v * (operators.laplacian_x - operators.laplacian_y) @ (x**2)
    mixed_from_y = eta_v * (operators.laplacian_x - operators.laplacian_y) @ (y**2)
    assert mixed_from_x[center] > 0
    assert mixed_from_y[center] < 0


def test_mixed_current_has_opposite_x_y_sign(mesh, operators):
    eta_s = 0.8
    eta_v = 0.2
    unit = mesh.edge_mesh.normalized_directions
    psi_d = np.ones(len(mesh.sites), dtype=complex)

    def mixed_delta(phase):
        psi_s = np.exp(0.1j * phase)
        coupled = operators.get_s_plus_d_supercurrent(
            psi_d, psi_s, eta_s=eta_s, eta_v=eta_v
        )
        diagonal = operators.get_s_plus_d_supercurrent(
            psi_d, psi_s, eta_s=eta_s, eta_v=0
        )
        return coupled - diagonal

    delta_x = mixed_delta(mesh.sites[:, 0])
    delta_y = mixed_delta(mesh.sites[:, 1])
    horizontal = np.abs(unit[:, 1]) < 0.05
    vertical = np.abs(unit[:, 0]) < 0.05
    assert np.mean(delta_x[horizontal] * unit[horizontal, 0]) < 0
    assert np.mean(delta_y[vertical] * unit[vertical, 1]) > 0


def test_pure_d_equilibrium(operators):
    model = SPlusDModel(eta_s=0.8, eta_v=0.2, nu=-1, tau1=1)
    solver = make_solver(operators, model)
    psi_d = np.ones(len(operators.mesh.sites), dtype=complex)
    psi_s = np.zeros_like(psi_d)
    abs_d = np.ones(len(psi_d))
    abs_s = np.zeros(len(psi_d))
    result = solver.adaptive_euler_step(
        0,
        psi_d,
        psi_s,
        abs_d,
        abs_s,
        np.zeros(len(psi_d)),
        np.ones(len(psi_d)),
        1e-3,
    )
    assert np.allclose(result[0], psi_d, atol=1e-13)
    assert np.allclose(result[1], psi_s, atol=1e-13)


def test_eta_v_zero_decouples_components(operators):
    model = SPlusDModel(
        eta_s=0.8,
        eta_v=0,
        nu=-0.4,
        tau1=1,
        tau3=0,
        tau4=0,
    )
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    psi_d = np.full(nsites, 0.7 + 0.1j)
    psi_s_a = np.full(nsites, 0.2 - 0.1j)
    psi_s_b = np.full(nsites, -0.6 + 0.3j)
    args = (np.abs(psi_d) ** 2, np.zeros(nsites), np.ones(nsites), 1e-3)
    result_a = solver.adaptive_euler_step(
        0, psi_d, psi_s_a, args[0], np.abs(psi_s_a) ** 2, *args[1:]
    )
    result_b = solver.adaptive_euler_step(
        0, psi_d, psi_s_b, args[0], np.abs(psi_s_b) ** 2, *args[1:]
    )
    assert np.allclose(result_a[0], result_b[0])


def test_scalar_potential_phase_in_s_equation(operators):
    amplitude = 0.4
    model = SPlusDModel(
        eta_s=0.7,
        eta_v=0,
        nu=amplitude**2,
        tau1=1,
        tau3=0,
        tau4=0,
    )
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    psi_d = np.zeros(nsites, dtype=complex)
    psi_s = np.full(nsites, amplitude, dtype=complex)
    phi = np.full(nsites, 0.3)
    dt = 1e-3
    result = solver.adaptive_euler_step(
        0,
        psi_d,
        psi_s,
        np.zeros(nsites),
        np.full(nsites, amplitude**2),
        phi,
        np.ones(nsites),
        dt,
    )
    assert np.allclose(result[1], np.exp(-1j * phi * dt) * psi_s, atol=1e-13)


def test_terminal_values_are_clamped_after_step(operators):
    model = SPlusDModel(eta_s=0.8, eta_v=0.2, nu=-1, tau1=1)
    solver = make_solver(operators, model)
    solver.terminal_psi = 0.35
    solver.normal_boundary_index = np.array([0, 1], dtype=int)
    nsites = len(operators.mesh.sites)
    psi_d = np.ones(nsites, dtype=complex)
    psi_s = np.full(nsites, 0.1 + 0.2j)
    result = solver.adaptive_euler_step(
        0,
        psi_d,
        psi_s,
        np.abs(psi_d) ** 2,
        np.abs(psi_s) ** 2,
        np.zeros(nsites),
        np.ones(nsites),
        1e-3,
    )
    assert np.all(result[0][solver.normal_boundary_index] == solver.terminal_psi)
    assert np.all(result[1][solver.normal_boundary_index] == 0)


def test_poisson_residual_and_current_continuity(mesh, operators):
    model = SPlusDModel(
        eta_s=0.8,
        eta_v=0.2,
        nu=-1,
        tau1=1,
        beta_em=2.5,
    )
    solver = make_solver(operators, model)
    x = mesh.sites[:, 0]
    y = mesh.sites[:, 1]
    psi_d = 0.8 * np.exp(0.2j * x)
    psi_s = 0.2 * np.exp(-0.15j * y)
    dA_dt = 0.03 * np.sin(mesh.edge_mesh.centers[:, 0])
    solver.dA_boundary_normal = 0.02 * np.cos(
        mesh.edge_mesh.centers[operators.mesh.edge_mesh.boundary_edge_indices, 1]
    )
    phi, supercurrent, normal_current = solver.solve_for_observables(
        psi_d, psi_s, dA_dt
    )
    raw_supercurrent = operators.get_s_plus_d_supercurrent(
        psi_d, psi_s, eta_s=model.eta_s, eta_v=model.eta_v
    )
    assert np.allclose(supercurrent, raw_supercurrent / model.beta_em)
    effective_boundary = solver.mu_boundary - solver.dA_boundary_normal
    rhs = operators.divergence @ (supercurrent - dA_dt) - (
        operators.mu_boundary_laplacian @ effective_boundary
    )
    rhs[operators.mu_reference_index] = 0
    residual = operators.mu_laplacian @ phi - rhs
    assert np.linalg.norm(residual, ord=np.inf) < 2e-10
    continuity = operators.divergence @ (supercurrent + normal_current)
    boundary_flux = operators.mu_boundary_laplacian @ effective_boundary
    mask = np.ones(len(mesh.sites), dtype=bool)
    mask[operators.mu_reference_index] = False
    assert np.linalg.norm((continuity - boundary_flux)[mask], ord=np.inf) < 2e-10


def test_insulating_flux_for_covariantly_constant_fields(mesh, operators):
    chi = 0.13 * mesh.sites[:, 0] + 0.07 * mesh.sites[:, 1]
    edges = mesh.edge_mesh.edges
    directions = mesh.edge_mesh.directions
    delta_chi = chi[edges[:, 1]] - chi[edges[:, 0]]
    vector_potential = delta_chi[:, None] * directions
    vector_potential /= np.sum(directions**2, axis=1)[:, None]
    operators.set_link_exponents(vector_potential)
    phase = np.exp(1j * chi)
    current = operators.get_s_plus_d_supercurrent(
        phase, 0.2 * phase, eta_s=0.8, eta_v=0.2
    )
    assert np.max(np.abs(current)) < 2e-12
    assert np.max(np.abs(operators.psi_laplacian @ phase)) < 2e-11
    operators.set_link_exponents(np.zeros_like(vector_potential))


def test_first_order_timestep_convergence(operators):
    model = SPlusDModel(eta_s=1, eta_v=0, nu=-1, tau1=1)
    nsites = len(operators.mesh.sites)
    initial = 0.5
    final_time = 0.1

    def integrate(dt):
        solver = make_solver(operators, model)
        psi_d = np.full(nsites, initial, dtype=complex)
        psi_s = np.zeros(nsites, dtype=complex)
        for step in range(round(final_time / dt)):
            abs_d = np.abs(psi_d) ** 2
            psi_d, psi_s, _, _, _ = solver.adaptive_euler_step(
                step,
                psi_d,
                psi_s,
                abs_d,
                np.zeros(nsites),
                np.zeros(nsites),
                np.ones(nsites),
                dt,
            )
        return abs(psi_d[0])

    exact_sq = 1 / (1 + (initial**-2 - 1) * np.exp(-2 * final_time))
    coarse_error = abs(integrate(0.01) - np.sqrt(exact_sq))
    fine_error = abs(integrate(0.005) - np.sqrt(exact_sq))
    assert fine_error < 0.6 * coarse_error


def test_directional_operator_mesh_convergence():
    errors = []
    for max_edge_length in (0.6, 0.4):
        layer = tdgl.Layer(
            coherence_length=1,
            london_lambda=2,
            thickness=0.1,
        )
        film = tdgl.Polygon("film", points=box(4)).resample(121)
        device = tdgl.Device("mesh-convergence", layer=layer, film=film)
        device.make_mesh(max_edge_length=max_edge_length, smooth=100)
        local_operators = MeshOperators(
            device.mesh,
            SparseSolver.SUPERLU,
            fixed_sites=np.array([], dtype=int),
            fix_psi=False,
        )
        local_operators.build_operators()
        local_operators.set_link_exponents(
            np.zeros((len(device.mesh.edge_mesh.edges), 2))
        )
        x = device.mesh.sites[:, 0]
        y = device.mesh.sites[:, 1]
        interior = (np.abs(x) < 1) & (np.abs(y) < 1)
        numerical = (local_operators.laplacian_x @ (x**2))[interior].real
        areas = device.mesh.areas[interior]
        weak_average = np.sum(areas * numerical) / np.sum(areas)
        errors.append(abs(weak_average - 2))
    assert errors[1] < 0.5 * errors[0]


def test_runner_stops_and_saves_at_requested_time():
    class DataHandlerStub:
        tmp_file = None

        def __init__(self):
            self.saved = []

        def save_fixed_values(self, values):
            pass

        def save_time_step(self, state, data, running_state):
            self.saved.append((state.copy(), data.copy()))

    def update(state, running_state, dt, *, value):
        return dt, value + 1

    options = SolverOptions(
        solve_time=0.25,
        dt_init=0.1,
        dt_max=0.1,
        adaptive=False,
        save_every=10,
    )
    handler = DataHandlerStub()
    runner = Runner(
        function=update,
        options=options,
        initial_values=[0],
        names=["value"],
        data_handler=handler,
    )
    assert runner.run()
    assert runner.time == pytest.approx(options.solve_time)
    assert handler.saved[-1][0]["time"] == pytest.approx(options.solve_time)
    assert handler.saved[-1][1]["value"] == 3
