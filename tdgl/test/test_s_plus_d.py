import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import tdgl
from tdgl.device.models import SPlusDModel
from tdgl.finite_volume.mesh import Mesh
from tdgl.finite_volume.operators import MeshOperators
from tdgl.geometry import box
from tdgl.solution.data import get_current_scale
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
        use_fem_for_psi=True,
    )
    operators.build_operators(build_magnetic_diffusion=True)
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
    solver.epsilon = np.ones(len(operators.mesh.sites))
    solver.mu_boundary = np.zeros(len(operators.mesh.edge_mesh.boundary_edge_indices))
    solver.dA_boundary_normal = np.zeros_like(solver.mu_boundary)
    solver.num_edges = len(operators.mesh.edge_mesh.edges)
    solver.boundary_edge_indices = operators.mesh.edge_mesh.boundary_edge_indices
    solver._s_plus_d_magnetic_dt = None
    solver._s_plus_d_magnetic_lu = None
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


@pytest.mark.parametrize(
    "model",
    [
        tdgl.SingleBandModel(gamma=np.nan),
        tdgl.SPlusDModel(tau3=np.inf),
        tdgl.DPlusDPrimeModel(alpha=np.nan),
        tdgl.SPlusSModel(josephson_gamma=np.inf),
    ],
)
def test_all_model_coefficients_must_be_finite(model):
    with pytest.raises(ValueError, match="finite"):
        model.validate()


def test_s_plus_d_physical_current_scale_restores_beta_em():
    device = type(
        "DeviceStub",
        (),
        {
            "K0": 2.0,
            "layer": type("LayerStub", (), {"model": SPlusDModel(beta_em=3.0)})(),
        },
    )()
    assert get_current_scale(device) == pytest.approx(6.0)


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


def test_s_plus_d_equilibrium_stop_records_field_change(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=SPlusDModel(),
    )
    film = tdgl.Polygon("film", points=box(2)).resample(41)
    device = tdgl.Device("s-plus-d-equilibrium", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.5, smooth=10)
    solution = tdgl.solve(
        device,
        SolverOptions(
            solve_time=0.1,
            dt_init=1e-3,
            dt_max=1e-3,
            adaptive=False,
            terminal_psi=None,
            save_every=10,
            equilibrium_tolerance=1e6,
            equilibrium_window=2,
            output_file=str(tmp_path / "s-plus-d-equilibrium.h5"),
        ),
    )

    state = solution.tdgl_data.state
    assert state["equilibrium_reached"]
    assert state["equilibrium_steps"] == 2
    assert state["equilibrium_time"] == pytest.approx(2e-3)
    assert state["equilibrium_checks"] == 1
    assert state["equilibrium_reference_step"] == 0
    assert state["time"] < solution.options.solve_time
    assert np.isfinite(state["equilibrium_error"])
    assert np.isfinite(state["equilibrium_order_parameter_error"])
    assert np.isfinite(state["equilibrium_electromagnetic_error"])
    assert np.isfinite(state["equilibrium_raw_error"])
    assert np.isfinite(state["equilibrium_phase_shift"])
    loaded = tdgl.Solution.from_hdf5(solution.path)
    assert loaded.options.equilibrium_window == 2
    assert loaded.tdgl_data.state["equilibrium_reached"]


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


def test_s_plus_d_goncalves_screening_short_solve(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=SPlusDModel(
            eta_s=2,
            eta_v=1,
            nu=-2,
            tau1=8 / 3,
            tau3=16 / 3,
            tau4=2,
            beta_em=1,
        ),
    )
    film = tdgl.Polygon("film", points=box(2)).resample(41)
    device = tdgl.Device("s-plus-d-screening", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.5, smooth=10)
    field_units = "mT"
    applied_field = 0.2 * device.Bc2.to(field_units).magnitude
    solution = tdgl.solve(
        device,
        SolverOptions(
            solve_time=5e-4,
            dt_init=1e-4,
            dt_max=1e-4,
            adaptive=True,
            include_screening=True,
            terminal_psi=None,
            save_every=10,
            field_units=field_units,
            output_file=str(tmp_path / "s-plus-d-screening.h5"),
        ),
        applied_vector_potential=tdgl.sources.ConstantField(
            applied_field,
            field_units=field_units,
            length_units=device.length_units,
        ),
    )
    data = solution.tdgl_data
    assert np.all(np.isfinite(data.psi_d))
    assert np.all(np.isfinite(data.psi_s))
    assert np.all(np.isfinite(data.induced_vector_potential))
    assert np.all(data.mu == 0)
    assert np.linalg.norm(data.induced_vector_potential) > 0
    assert data.state["time"] == pytest.approx(5e-4)


def test_goncalves_screening_rejects_terminal_current(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=SPlusDModel(),
    )
    film = tdgl.Polygon("film", points=box(2)).resample(41)
    source = tdgl.Polygon("source", points=box(1e-2, 1, center=(1, 0)))
    drain = tdgl.Polygon("drain", points=box(1e-2, 1, center=(-1, 0)))
    device = tdgl.Device(
        "s-plus-d-screening-terminals",
        layer=layer,
        film=film,
        terminals=[source, drain],
    )
    device.make_mesh(max_edge_length=0.5, smooth=10)
    with pytest.raises(ValueError, match="does not support terminal current"):
        TDGLSolver(
            device,
            SolverOptions(solve_time=1e-3, include_screening=True),
            terminal_currents={"source": 1, "drain": -1},
        )


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
        {"relaxation_s": 0},
    ],
)
def test_model_validation(kwargs):
    with pytest.raises(ValueError):
        SPlusDModel(**kwargs).validate()


def test_zero_retry_limit_does_not_retry_failed_single_band_step(operators):
    solver = make_solver(operators, tdgl.SingleBandModel())
    solver.options.adaptive = True
    solver.options.max_solve_retries = 0
    solver.u = 1
    calls = 0

    def fail_step(**kwargs):
        nonlocal calls
        calls += 1
        return None

    solver.solve_for_psi_squared = fail_step
    zeros = np.zeros(len(operators.mesh.sites), dtype=complex)
    ones = np.ones(len(operators.mesh.sites), dtype=complex)

    with pytest.raises(RuntimeError, match="0 retries"):
        solver.adaptive_euler_step(
            0,
            zeros,
            ones,
            np.zeros(len(zeros)),
            np.ones(len(ones)),
            np.zeros(len(zeros)),
            np.ones(len(ones)),
            1e-3,
        )

    assert calls == 1


def test_zero_retry_limit_does_not_retry_multicomponent_step(operators):
    solver = make_solver(operators, SPlusDModel(eta_s=0.8, eta_v=0.2))
    solver.options.adaptive = True
    solver.options.max_solve_retries = 0
    zeros = np.zeros(len(operators.mesh.sites), dtype=complex)
    ones = np.ones(len(operators.mesh.sites), dtype=complex)

    with pytest.raises(RuntimeError, match=r"dt = 1\.00e\+00"):
        solver.adaptive_euler_step(
            0,
            ones,
            zeros,
            np.ones(len(ones)),
            np.zeros(len(zeros)),
            np.zeros(len(zeros)),
            np.full(len(ones), 100.0),
            1.0,
        )


def test_directional_operators_sum_to_covariant_laplacian(operators):
    difference = operators.laplacian_x + operators.laplacian_y
    difference = difference - operators.psi_laplacian
    assert np.max(np.abs(difference.data), initial=0) < 1e-12


def make_obtuse_boundary_operators(*, fixed_sites=None):
    # Convex Delaunay quadrilateral with one negative boundary cotangent.
    sites = np.array(
        [
            [-0.9127301292054941, -0.1724587913869715],
            [0.2459388650401242, -0.33291992561140793],
            [-0.4708402480874938, 0.47457959202953814],
            [-0.9080780818533125, 0.8166826855707694],
        ]
    )
    mesh = Mesh.from_triangulation(sites, np.array([[0, 1, 2], [0, 2, 3]]))
    operators = MeshOperators(
        mesh,
        SparseSolver.SUPERLU,
        fixed_sites=fixed_sites,
        fix_psi=fixed_sites is not None,
        use_fem_for_psi=True,
    )
    operators.set_link_exponents(np.zeros((len(operators.edges), 2)))
    return operators


def test_mixed_gradient_energy_is_positive_on_obtuse_boundary_mesh():
    operators = make_obtuse_boundary_operators()
    area = sp.diags(operators.mesh.areas)
    stiffness = -(area @ operators.psi_laplacian).toarray()
    mixed_stiffness = -(
        area @ (operators.laplacian_y - operators.laplacian_x)
    ).toarray()
    gradient_hessian = np.block(
        [
            [stiffness, mixed_stiffness],
            [mixed_stiffness, 2 * stiffness],
        ]
    )

    assert np.min(operators.psi_laplacian_weights) < 0
    assert np.linalg.eigvalsh(gradient_hessian).min() > -2e-12


def test_directional_fixed_rows_sum_to_covariant_laplacian():
    operators = make_obtuse_boundary_operators(fixed_sites=np.array([0]))
    difference = operators.laplacian_x + operators.laplacian_y
    difference = difference - operators.psi_laplacian

    assert np.max(np.abs(difference.data), initial=0) < 1e-12


def test_mixed_current_is_free_energy_derivative_on_obtuse_mesh():
    operators = make_obtuse_boundary_operators(fixed_sites=np.array([0]))
    model = SPlusDModel(eta_s=2, eta_v=1, nu=-1, tau1=1)
    solver = make_solver(operators, model)
    rng = np.random.default_rng(21)
    d = rng.normal(size=len(operators.mesh.sites)) + 1j * rng.normal(
        size=len(operators.mesh.sites)
    )
    s = rng.normal(size=len(operators.mesh.sites)) + 1j * rng.normal(
        size=len(operators.mesh.sites)
    )
    direction = rng.normal(size=len(operators.edges))
    tangent = operators.mesh.edge_mesh.normalized_directions
    perturbation = direction[:, None] * tangent
    delta = 1e-6

    plus = solver.compute_s_plus_d_free_energy(
        d,
        s,
        delta * perturbation,
        include_magnetic=False,
        average=False,
    )
    minus = solver.compute_s_plus_d_free_energy(
        d,
        s,
        -delta * perturbation,
        include_magnetic=False,
        average=False,
    )
    finite_difference = (plus - minus) / (2 * delta)
    current = operators.get_s_plus_d_supercurrent(
        d, s, eta_s=model.eta_s, eta_v=model.eta_v
    )
    edge_metric = operators.laplacian_weights * operators.edge_lengths**2
    predicted = -2 * np.sum(edge_metric * current * direction)

    assert finite_difference == pytest.approx(predicted, rel=2e-8, abs=2e-8)


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
    mixed_from_x = eta_v * (operators.laplacian_y - operators.laplacian_x) @ (x**2)
    mixed_from_y = eta_v * (operators.laplacian_y - operators.laplacian_x) @ (y**2)
    assert mixed_from_x[center] < 0
    assert mixed_from_y[center] > 0


def test_tdgl_step_uses_paper_mixed_gradient_sign(mesh, operators):
    eta_v = 0.2
    model = SPlusDModel(eta_s=0.8, eta_v=eta_v, nu=-1, tau1=1)
    solver = make_solver(operators, model)
    x = mesh.sites[:, 0]
    center = np.argmin(np.sum(mesh.sites**2, axis=1))
    d = np.zeros(len(x), dtype=complex)
    s = x**2 + 0j
    dt = 1e-5
    new_d, *_ = solver.adaptive_euler_step(
        0,
        d,
        s,
        np.zeros(len(x)),
        np.abs(s) ** 2,
        np.zeros(len(x)),
        np.ones(len(x)),
        dt,
    )
    expected = dt * eta_v * ((operators.laplacian_y - operators.laplacian_x) @ s)
    assert expected[center].real < 0
    assert np.allclose(new_d, expected)


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


def test_uniform_field_has_exact_triangle_curl_and_zero_magnetic_curl(mesh, operators):
    field = 0.37
    centers = mesh.edge_mesh.centers
    vector_potential = np.column_stack(
        [-0.5 * field * centers[:, 1], 0.5 * field * centers[:, 0]]
    )
    triangle_field = operators.get_triangle_magnetic_field(vector_potential)
    magnetic_curl = operators.get_magnetic_curl(vector_potential, field)
    assert np.allclose(triangle_field, field, atol=2e-12, rtol=2e-12)
    assert np.allclose(magnetic_curl, 0, atol=2e-11, rtol=2e-11)


def test_magnetic_diffusion_is_gibbs_energy_hessian(operators):
    mesh = operators.mesh
    tangent = mesh.edge_mesh.normalized_directions
    curl = operators.triangle_magnetic_field_curl_x @ sp.diags(tangent[:, 0])
    curl += operators.triangle_magnetic_field_curl_y @ sp.diags(tangent[:, 1])
    triangle_sites = mesh.sites[mesh.elements]
    edge_1 = triangle_sites[:, 1] - triangle_sites[:, 0]
    edge_2 = triangle_sites[:, 2] - triangle_sites[:, 0]
    triangle_areas = 0.5 * np.abs(
        edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    )
    edge_metric = mesh.edge_mesh.dual_edge_lengths * mesh.edge_mesh.edge_lengths
    finite_volume_hessian = sp.diags(edge_metric) @ operators.magnetic_diffusion
    energy_hessian = curl.T @ sp.diags(triangle_areas) @ curl
    assert np.allclose(
        finite_volume_hessian.toarray(),
        energy_hessian.toarray(),
        atol=2e-12,
        rtol=2e-12,
    )


def test_s_plus_d_free_energy_decreases_under_dissipative_step(operators):
    model = SPlusDModel(
        eta_s=0.8,
        eta_v=0.2,
        nu=-0.4,
        tau1=1.3,
        tau3=0.7,
        tau4=0.15,
    )
    solver = make_solver(operators, model)
    solver.applied_triangle_field = np.zeros(len(operators.mesh.elements))
    solver.device = type("DeviceStub", (), {"kappa": 2.0, "mesh": operators.mesh})()
    rng = np.random.default_rng(12)
    d = 0.7 + 0.03 * (
        rng.normal(size=len(operators.mesh.sites))
        + 1j * rng.normal(size=len(operators.mesh.sites))
    )
    s = 0.1 + 0.03 * (
        rng.normal(size=len(operators.mesh.sites))
        + 1j * rng.normal(size=len(operators.mesh.sites))
    )
    before = solver.compute_s_plus_d_free_energy(d, s, include_magnetic=False)
    new_d, new_s, *_ = solver.adaptive_euler_step(
        0,
        d,
        s,
        np.abs(d) ** 2,
        np.abs(s) ** 2,
        np.zeros(len(d)),
        np.ones(len(d)),
        1e-5,
    )
    after = solver.compute_s_plus_d_free_energy(new_d, new_s, include_magnetic=False)
    assert after < before


def test_s_plus_d_relative_s_relaxation_scales_only_s_update(operators):
    fast = make_solver(
        operators,
        SPlusDModel(eta_s=2, eta_v=0, nu=0.4, tau1=1.2),
    )
    slow = make_solver(
        operators,
        SPlusDModel(
            eta_s=2,
            eta_v=0,
            nu=0.4,
            tau1=1.2,
            relaxation_s=4,
        ),
    )
    d = np.full(len(operators.mesh.sites), 0.7 + 0.1j)
    s = np.full(len(operators.mesh.sites), 0.2j)
    zeros = np.zeros(len(d))
    dt = 1e-5

    fast_d, fast_s, *_ = fast.adaptive_euler_step(
        0, d, s, np.abs(d) ** 2, np.abs(s) ** 2, zeros, np.ones(len(d)), dt
    )
    slow_d, slow_s, *_ = slow.adaptive_euler_step(
        0, d, s, np.abs(d) ** 2, np.abs(s) ** 2, zeros, np.ones(len(d)), dt
    )

    assert np.allclose(slow_d - d, fast_d - d)
    assert np.allclose(slow_s - s, (fast_s - s) / 4)


def test_s_plus_d_disorder_can_suppress_both_quadratic_coefficients(operators):
    model = SPlusDModel(
        eta_s=2,
        eta_v=0,
        nu=0.8,
        tau1=1,
        nu_disorder_coupling=3,
    )
    solver = make_solver(operators, model)
    d = np.zeros(len(operators.mesh.sites), dtype=complex)
    s = np.full(len(d), 0.1j)
    epsilon = np.full(len(d), 0.75)
    dt = 1e-5

    _, updated_s, *_ = solver.adaptive_euler_step(
        0,
        d,
        s,
        np.abs(d) ** 2,
        np.abs(s) ** 2,
        np.zeros(len(d)),
        epsilon,
        dt,
    )
    nu_effective = model.nu + model.nu_disorder_coupling * (epsilon - 1)
    expected_rhs = nu_effective * s - model.tau1 * np.abs(s) ** 2 * s
    expected = s + dt * expected_rhs / model.eta_s
    assert np.allclose(updated_s, expected)


def test_s_plus_d_free_energy_uses_supplied_links_and_restores_state(operators):
    model = SPlusDModel(eta_s=0.8, eta_v=0.2)
    solver = make_solver(operators, model)
    zeros = np.zeros((len(operators.edges), 2))
    operators.set_link_exponents(zeros)
    rng = np.random.default_rng(31)
    trial = 0.2 * rng.normal(size=zeros.shape)
    d = np.ones(len(operators.mesh.sites), dtype=complex)
    s = np.zeros_like(d)
    base = solver.compute_s_plus_d_free_energy(d, s, include_magnetic=False)
    trial_energy = solver.compute_s_plus_d_free_energy(
        d, s, vector_potential=trial, include_magnetic=False
    )
    assert trial_energy > base
    assert np.array_equal(operators.link_exponents, zeros)


def test_orbital_zeeman_bound_current_is_free_energy_derivative(operators):
    rng = np.random.default_rng(22)
    magnetization = rng.normal(size=len(operators.mesh.sites))
    direction = rng.normal(size=len(operators.edges))
    tangent = operators.mesh.edge_mesh.normalized_directions
    perturbation = direction[:, None] * tangent
    zero = np.zeros_like(perturbation)
    delta = 1e-7

    def zeeman_energy(vector_potential):
        field = operators.get_magnetic_field(vector_potential)
        return -np.sum(operators.areas * magnetization * field)

    finite_difference = (
        zeeman_energy(zero + delta * perturbation)
        - zeeman_energy(zero - delta * perturbation)
    ) / (2 * delta)
    bound_current = operators.get_magnetization_current(magnetization)
    edge_metric = (
        operators.mesh.edge_mesh.dual_edge_lengths
        * operators.mesh.edge_mesh.edge_lengths
    )
    predicted = -2 * np.sum(edge_metric * bound_current * direction)
    assert finite_difference == pytest.approx(predicted, rel=2e-9, abs=2e-9)


def test_magnetic_update_decreases_gibbs_energy(operators):
    model = SPlusDModel(eta_s=0.8, eta_v=0.2, beta_em=1)
    solver = make_solver(operators, model)
    solver.device = type("DeviceStub", (), {"kappa": 2.0, "mesh": operators.mesh})()
    solver.normalized_directions = operators.mesh.edge_mesh.normalized_directions
    solver.applied_boundary_field = np.zeros(
        len(operators.mesh.edge_mesh.boundary_edge_indices)
    )
    solver.applied_triangle_field = np.zeros(len(operators.mesh.elements))
    rng = np.random.default_rng(4)
    tangent = operators.mesh.edge_mesh.normalized_directions
    induced = 1e-2 * rng.normal(size=len(tangent))[:, None] * tangent
    applied = np.zeros_like(induced)
    zeros = np.zeros(len(operators.mesh.sites), dtype=complex)

    operators.set_link_exponents(induced)
    before = solver.compute_s_plus_d_free_energy(zeros, zeros, vector_potential=induced)
    updated, _ = solver.advance_s_plus_d_vector_potential(
        zeros, zeros, applied, induced, 1e-5
    )
    operators.set_link_exponents(updated)
    after = solver.compute_s_plus_d_free_energy(zeros, zeros, vector_potential=updated)
    assert after < before
    operators.set_link_exponents(np.zeros_like(updated))


@pytest.mark.parametrize("beta_em", [0.2, 2.0, 200.0])
def test_s_plus_d_bulk_drive_has_unit_normal_state_resistivity(operators, beta_em):
    # Cover electromagnetic relaxation clocks spanning three orders of magnitude.
    model = SPlusDModel(eta_s=2, eta_v=0, beta_em=beta_em)
    solver = make_solver(operators, model)
    solver.device = type("DeviceStub", (), {"kappa": 2.0, "mesh": operators.mesh})()
    tangent = operators.mesh.edge_mesh.normalized_directions
    solver.normalized_directions = tangent
    drive = np.array([0.08, -0.03])
    solver.s_plus_d_drive_tangent = tangent @ drive
    zeros = np.zeros(len(operators.mesh.sites), dtype=complex)
    zero_vector_potential = np.zeros((len(tangent), 2))

    _, dA_dt = solver.advance_s_plus_d_vector_potential(
        zeros,
        zeros,
        zero_vector_potential,
        zero_vector_potential,
        1e-4,
    )

    # In the normal state E = -dA/dt = J_drive / beta_em.
    assert np.allclose(-dA_dt, (tangent @ drive) / model.beta_em, atol=2e-11)


def test_thin_film_screening_iterations_do_not_advance_psi_twice(operators):
    solver = make_solver(operators, tdgl.SingleBandModel(), adaptive=False)
    solver.xp = np
    solver.options.include_screening = True
    solver.options.screening_tolerance = 0.5
    solver.options.max_iterations_per_step = 3
    solver.goncalves_screening = False
    solver.dynamic_vector_potential = False
    solver.dynamic_epsilon = False
    solver.current_A_applied = np.zeros((len(operators.edges), 2))
    solver.tentative_dt = solver.options.dt_init
    solver.dt_max = solver.options.dt_max
    solver.d_psi_sq_vals = []
    solver.probe_points = None
    solver.normalized_directions = operators.mesh.edge_mesh.normalized_directions
    solver.boundary_normals = np.zeros(
        (len(operators.mesh.edge_mesh.boundary_edge_indices), 2)
    )
    solver.update_mu_boundary = lambda time: None

    initial = np.ones(len(operators.mesh.sites), dtype=complex)
    recorded_inputs = []

    def adaptive_step(step, psi2, psi1, sq2, sq1, mu, epsilon, dt):
        recorded_inputs.append(psi1.copy())
        next_psi1 = psi1 + 1
        return psi2.copy(), next_psi1, sq2.copy(), np.abs(next_psi1) ** 2, dt

    solver.adaptive_euler_step = adaptive_step
    solver.solve_for_observables = lambda psi2, psi1, dA_dt: (
        np.zeros(len(initial)),
        np.zeros(len(operators.edges)),
        np.zeros(len(operators.edges)),
    )
    screening_errors = iter([1.0, 0.0])
    solver.get_induced_vector_potential = lambda current, values, velocity: (
        values[-1],
        next(screening_errors),
    )

    class RunningStateStub:
        def append(self, name, value):
            pass

    result = solver.update(
        {"step": 0, "time": 0.0, "_remaining_time": solver.options.dt_init},
        RunningStateStub(),
        solver.options.dt_init,
        psi2=np.zeros_like(initial),
        psi1=initial,
        mu=np.zeros(len(initial)),
        supercurrent=np.zeros(len(operators.edges)),
        normal_current=np.zeros(len(operators.edges)),
        induced_vector_potential=np.zeros((len(operators.edges), 2)),
    )
    assert len(recorded_inputs) == 2
    assert all(np.array_equal(value, initial) for value in recorded_inputs)
    assert np.array_equal(result.psi1, initial + 1)


def test_dynamic_inputs_and_derivative_are_returned_at_accepted_endpoint(operators):
    solver = make_solver(operators, SPlusDModel(), adaptive=False)
    solver.xp = np
    solver.goncalves_screening = False
    solver.dynamic_vector_potential = True
    solver.dynamic_epsilon = True
    solver.current_A_applied = np.full((len(operators.edges), 2), 99.0)
    solver.tentative_dt = 0.2
    solver.dt_max = 0.2
    solver.d_psi_sq_vals = []
    solver.probe_points = None
    solver.normalized_directions = operators.mesh.edge_mesh.normalized_directions
    solver.boundary_normals = np.zeros(
        (len(operators.mesh.edge_mesh.boundary_edge_indices), 2)
    )
    solver.update_mu_boundary = lambda time: None
    solver.update_applied_vector_potential = lambda time: np.full(
        (len(operators.edges), 2), time
    )
    solver.update_epsilon = lambda time: np.full(len(operators.mesh.sites), 1 - time)
    recorded_dA_dt = []
    solver.solve_for_observables = lambda psi2, psi1, dA_dt: (
        recorded_dA_dt.append(np.asarray(dA_dt).copy())
        or np.zeros(len(operators.mesh.sites)),
        np.zeros(len(operators.edges)),
        np.zeros(len(operators.edges)),
    )

    initial_d = np.ones(len(operators.mesh.sites), dtype=complex)
    initial_s = np.zeros_like(initial_d)

    class RunningStateStub:
        def append(self, name, value):
            pass

    result = solver.update(
        {"step": 0, "time": 0.0, "_remaining_time": 0.1},
        RunningStateStub(),
        0.2,
        psi2=initial_d,
        psi1=initial_s,
        mu=np.zeros(len(initial_d)),
        supercurrent=np.zeros(len(operators.edges)),
        normal_current=np.zeros(len(operators.edges)),
        induced_vector_potential=np.zeros((len(operators.edges), 2)),
        applied_vector_potential=np.full((len(operators.edges), 2), 17.0),
        epsilon=np.full(len(initial_d), -17.0),
    )
    assert result.dt == pytest.approx(0.1)
    assert np.allclose(result.A_applied, 0.1)
    assert np.allclose(result.epsilon, 0.9)
    expected = np.sum(solver.normalized_directions, axis=1)
    assert np.allclose(recorded_dA_dt[0], expected)
    operators.set_link_exponents(np.zeros((len(operators.edges), 2)))


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
            use_fem_for_psi=True,
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


def test_runner_stops_on_windowed_order_parameter_change():
    class DataHandlerStub:
        tmp_file = None

        def __init__(self):
            self.saved = []

        def save_fixed_values(self, values):
            pass

        def save_time_step(self, state, data, running_state):
            self.saved.append((state.copy(), data.copy()))

    increments = iter([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])

    def update(state, running_state, dt, *, psi1, psi2):
        increment = next(increments)
        return dt, psi1 + increment, psi2 - increment

    options = SolverOptions(
        solve_time=1,
        dt_init=0.1,
        dt_max=0.1,
        adaptive=False,
        save_every=10,
        equilibrium_tolerance=1e-2,
        equilibrium_window=3,
        equilibrium_min_time=0.2,
    )
    handler = DataHandlerStub()
    runner = Runner(
        function=update,
        options=options,
        initial_values=[np.array([0.0]), np.array([0.0])],
        names=["psi1", "psi2"],
        data_handler=handler,
    )

    assert runner.run()
    assert runner.time == pytest.approx(0.6)
    assert runner.state["equilibrium_reached"]
    assert runner.state["equilibrium_steps"] == 3
    assert runner.state["equilibrium_checks"] == 2
    assert runner.state["equilibrium_reference_step"] == 3
    assert runner.state["equilibrium_error"] == pytest.approx(3e-4)
    assert runner.state["equilibrium_time"] == pytest.approx(0.6)
    assert handler.saved[-1][0]["equilibrium_reached"]
    assert handler.saved[-1][1]["psi1"] == pytest.approx([0.3003])
    assert handler.saved[-1][1]["psi2"] == pytest.approx([-0.3003])


def test_runner_equilibrium_includes_induced_vector_potential():
    class DataHandlerStub:
        tmp_file = None

        def __init__(self):
            self.saved = []

        def save_fixed_values(self, values):
            pass

        def save_time_step(self, state, data, running_state):
            self.saved.append((state.copy(), data.copy()))

    increments = iter([0.1, 0.1, 0.0])

    def update(
        state,
        running_state,
        dt,
        *,
        psi1,
        psi2,
        induced_vector_potential,
    ):
        increment = next(increments)
        return dt, psi1, psi2, induced_vector_potential + increment

    options = SolverOptions(
        solve_time=1,
        dt_init=0.1,
        dt_max=0.1,
        adaptive=False,
        save_every=10,
        equilibrium_tolerance=1e-2,
        equilibrium_window=1,
    )
    handler = DataHandlerStub()
    runner = Runner(
        function=update,
        options=options,
        initial_values=[
            np.array([1.0 + 0j]),
            np.array([1.0 + 0j]),
            np.zeros((2, 2)),
        ],
        names=["psi1", "psi2", "induced_vector_potential"],
        data_handler=handler,
    )

    assert runner.run()
    assert runner.time == pytest.approx(0.3)
    assert runner.state["equilibrium_reached"]
    assert runner.state["equilibrium_order_parameter_error"] == pytest.approx(0)
    assert runner.state["equilibrium_electromagnetic_error"] == pytest.approx(0)
    assert runner.state["equilibrium_checks"] == 3


def test_runner_equilibrium_check_removes_shared_global_phase():
    class DataHandlerStub:
        tmp_file = None

        def __init__(self):
            self.saved = []

        def save_fixed_values(self, values):
            pass

        def save_time_step(self, state, data, running_state):
            self.saved.append((state.copy(), data.copy()))

    phase_per_step = 0.2

    def update(state, running_state, dt, *, psi1, psi2):
        rotation = np.exp(1j * phase_per_step)
        return dt, rotation * psi1, rotation * psi2

    options = SolverOptions(
        solve_time=1,
        dt_init=0.1,
        dt_max=0.1,
        adaptive=False,
        save_every=10,
        equilibrium_tolerance=1e-12,
        equilibrium_window=3,
    )
    handler = DataHandlerStub()
    runner = Runner(
        function=update,
        options=options,
        initial_values=[
            np.array([1.0 + 0j, 1j]),
            np.array([0.5 + 0j, -0.5j]),
        ],
        names=["psi1", "psi2"],
        data_handler=handler,
    )

    assert runner.run()
    assert runner.state["equilibrium_reached"]
    assert runner.state["equilibrium_steps"] == 3
    assert runner.state["equilibrium_phase_shift"] == pytest.approx(0.6)
    assert runner.state["equilibrium_raw_error"] > 0.5
    assert runner.state["equilibrium_error"] < 1e-12


def test_runner_resets_prescribed_dynamic_state_after_thermalization():
    class DataHandlerStub:
        tmp_file = None

        def __init__(self):
            self.saved = []

        def save_fixed_values(self, values):
            pass

        def save_time_step(self, state, data, running_state):
            self.saved.append((state.copy(), data.copy()))

    def update(state, running_state, dt, *, applied_vector_potential):
        endpoint = state["time"] + dt
        return dt, np.array([endpoint])

    options = SolverOptions(
        solve_time=0.1,
        skip_time=0.2,
        dt_init=0.1,
        dt_max=0.1,
        adaptive=False,
        save_every=10,
    )
    handler = DataHandlerStub()
    runner = Runner(
        function=update,
        options=options,
        initial_values=[np.array([0.0])],
        names=["applied_vector_potential"],
        data_handler=handler,
    )
    assert runner.run()
    assert handler.saved[0][0]["time"] == 0
    assert handler.saved[0][1]["applied_vector_potential"] == pytest.approx([0.0])
    assert handler.saved[-1][0]["time"] == pytest.approx(0.1)
    assert handler.saved[-1][1]["applied_vector_potential"] == pytest.approx([0.1])
