import h5py
import numpy as np
import pytest

import tdgl
from tdgl.device.models import DPlusDPrimeModel, SPlusDModel
from tdgl.finite_volume.operators import MeshOperators
from tdgl.geometry import box
from tdgl.solver.options import SolverOptions, SparseSolver
from tdgl.solver.solver import TDGLSolver


@pytest.fixture(scope="module")
def mesh():
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
    )
    film = tdgl.Polygon("film", points=box(3)).resample(81)
    device = tdgl.Device("d-plus-d-prime-test", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.4, smooth=15)
    return device.mesh


@pytest.fixture(scope="module")
def operators(mesh):
    result = MeshOperators(
        mesh,
        SparseSolver.SUPERLU,
        fixed_sites=np.array([], dtype=int),
        fix_psi=False,
    )
    result.build_operators()
    result.set_link_exponents(np.zeros((len(mesh.edge_mesh.edges), 2)))
    return result


def make_solver(operators, model, magnetic_field=0.0, *, adaptive=False):
    solver = TDGLSolver.__new__(TDGLSolver)
    solver.model = model
    solver.options = SolverOptions(
        solve_time=1,
        dt_init=1e-4,
        dt_max=1e-4,
        adaptive=adaptive,
        terminal_psi=None,
    )
    solver.operators = operators
    solver.terminal_psi = None
    solver.normal_boundary_index = np.array([], dtype=int)
    solver.use_cupy = False
    solver.mu_boundary = np.zeros(len(operators.mesh.edge_mesh.boundary_edge_indices))
    solver.dA_boundary_normal = np.zeros_like(solver.mu_boundary)
    solver.magnetic_field = np.full(len(operators.mesh.sites), magnetic_field)
    return solver


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": np.nan},
        {"relaxation_d": 0},
        {"relaxation_d_prime": 0},
        {"em_coupling": 0},
        {"zeeman_coupling": 1},
        {"zeeman_coupling": -1},
    ],
)
def test_model_validation(kwargs):
    with pytest.raises(ValueError):
        DPlusDPrimeModel(**kwargs).validate()


def test_alpha_above_subdominant_regime_warns():
    with pytest.warns(UserWarning, match="subdominant"):
        DPlusDPrimeModel(alpha=1.01).validate()


def test_uniform_field_curl_is_exact(mesh, operators):
    field = 0.37
    x = mesh.edge_mesh.centers[:, 0]
    y = mesh.edge_mesh.centers[:, 1]
    vector_potential = np.column_stack((-0.5 * field * y, 0.5 * field * x))
    actual = operators.get_magnetic_field(vector_potential)
    assert np.allclose(actual, field, atol=2e-13, rtol=2e-13)


def test_zero_zeeman_matches_existing_s_plus_d_equations(operators):
    alpha = 0.57
    d_plus_d_prime = make_solver(
        operators,
        DPlusDPrimeModel(alpha=alpha),
    )
    mapped_s_plus_d = make_solver(
        operators,
        SPlusDModel(
            eta_s=1,
            eta_v=0,
            nu=alpha,
            tau1=1,
            tau3=4 / 3,
            tau4=1 / 3,
            beta_em=1,
        ),
    )
    rng = np.random.default_rng(23)
    nsites = len(operators.mesh.sites)
    d = 0.7 + 0.03 * (rng.normal(size=nsites) + 1j * rng.normal(size=nsites))
    d_prime = 0.2j + 0.03 * (rng.normal(size=nsites) + 1j * rng.normal(size=nsites))
    mu = np.zeros(nsites)
    epsilon = np.ones(nsites)
    dt = 1e-4
    new_d_prime, new_d, *_ = d_plus_d_prime.adaptive_euler_step(
        0,
        d_prime,
        d,
        np.abs(d_prime) ** 2,
        np.abs(d) ** 2,
        mu,
        epsilon,
        dt,
    )
    mapped_d, mapped_d_prime, *_ = mapped_s_plus_d.adaptive_euler_step(
        0,
        d,
        d_prime,
        np.abs(d) ** 2,
        np.abs(d_prime) ** 2,
        mu,
        epsilon,
        dt,
    )
    assert np.allclose(new_d, mapped_d)
    assert np.allclose(new_d_prime, mapped_d_prime)


@pytest.mark.parametrize("alpha", [0.4, 0.6, 0.9])
def test_mixed_zero_field_equilibrium(operators, alpha):
    solver = make_solver(operators, DPlusDPrimeModel(alpha=alpha))
    nsites = len(operators.mesh.sites)
    d_amplitude = np.sqrt(3 * (3 - alpha) / 8)
    d_prime_amplitude = np.sqrt(3 * (3 * alpha - 1) / 8)
    d = np.full(nsites, d_amplitude, dtype=complex)
    d_prime = np.full(nsites, -1j * d_prime_amplitude, dtype=complex)
    new_d_prime, new_d, *_ = solver.adaptive_euler_step(
        0,
        d_prime,
        d,
        np.abs(d_prime) ** 2,
        np.abs(d) ** 2,
        np.zeros(nsites),
        np.ones(nsites),
        1e-4,
    )
    assert np.allclose(new_d, d, atol=2e-13)
    assert np.allclose(new_d_prime, d_prime, atol=2e-13)


@pytest.mark.parametrize("field, expected_sign", [(0.2, -1), (-0.2, 1)])
def test_zeeman_term_selects_field_dependent_chirality(operators, field, expected_sign):
    model = DPlusDPrimeModel(alpha=0, zeeman_coupling=0.3)
    solver = make_solver(operators, model, magnetic_field=field)
    nsites = len(operators.mesh.sites)
    d = np.ones(nsites, dtype=complex)
    d_prime = np.zeros(nsites, dtype=complex)
    new_d_prime, _, *_ = solver.adaptive_euler_step(
        0,
        d_prime,
        d,
        np.zeros(nsites),
        np.ones(nsites),
        np.zeros(nsites),
        np.ones(nsites),
        1e-4,
    )
    assert np.all(expected_sign * new_d_prime.imag > 0)


def test_zeeman_energy_favors_selected_chirality(operators):
    model = DPlusDPrimeModel(alpha=0.2, zeeman_coupling=0.3)
    solver = make_solver(operators, model, magnetic_field=0.4)
    nsites = len(operators.mesh.sites)
    d = np.ones(nsites, dtype=complex)
    favored = np.full(nsites, -0.2j)
    disfavored = -favored
    assert solver.compute_d_plus_d_prime_free_energy(
        d, favored
    ) < solver.compute_d_plus_d_prime_free_energy(d, disfavored)


def test_current_is_sum_of_diagonal_component_currents(operators):
    model = DPlusDPrimeModel(em_coupling=2.5)
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    x = operators.mesh.sites[:, 0]
    y = operators.mesh.sites[:, 1]
    d = 0.8 * np.exp(0.2j * x)
    d_prime = 0.3 * np.exp(-0.1j * y)
    _, actual, _ = solver.solve_for_observables(d_prime, d, 0.0)
    expected = (
        operators.get_supercurrent(d) + operators.get_supercurrent(d_prime)
    ) / model.em_coupling
    assert np.allclose(actual, expected)


def test_layer_round_trip(tmp_path):
    model = DPlusDPrimeModel(
        alpha=0.62,
        relaxation_d=1.2,
        relaxation_d_prime=0.8,
        em_coupling=1.4,
        zeeman_coupling=-0.25,
    )
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=model,
    )
    path = tmp_path / "layer.h5"
    with h5py.File(path, "w") as h5file:
        layer.to_hdf5(h5file.create_group("layer"))
    with h5py.File(path, "r") as h5file:
        loaded = tdgl.Layer.from_hdf5(h5file["layer"])
    assert loaded == layer


def test_short_solve_and_component_access(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=DPlusDPrimeModel(alpha=0.5, zeeman_coupling=0.2),
    )
    film = tdgl.Polygon("film", points=box(2)).resample(41)
    device = tdgl.Device(
        "d-plus-d-prime-smoke",
        layer=layer,
        film=film,
        probe_points=[(-0.1, 0), (0.1, 0)],
    )
    device.make_mesh(max_edge_length=0.5, smooth=10)
    field_units = "mT"
    reduced_field = 0.2
    applied_field = reduced_field * device.Bc2.to(field_units).magnitude
    solution = tdgl.solve(
        device,
        SolverOptions(
            solve_time=2e-4,
            dt_init=1e-4,
            dt_max=1e-4,
            adaptive=False,
            terminal_psi=None,
            field_units=field_units,
            save_every=1,
            progress_interval=1,
            output_file=str(tmp_path / "d-plus-d-prime.h5"),
        ),
        applied_vector_potential=applied_field,
    )
    assert np.all(np.isfinite(solution.get_order_parameter("d")))
    assert np.all(np.isfinite(solution.get_order_parameter("d_prime")))
    assert np.all(solution.orbital_magnetization > 0)
    with h5py.File(solution.path, "r") as h5file:
        assert h5file["solution/device/layer/model"].attrs["type"] == (
            "DPlusDPrimeModel"
        )
        last_step = h5file["data"][str(len(h5file["data"]) - 1)]
        assert "psi1" in last_step
        assert "psi2" in last_step


def test_unsupported_paths_are_rejected(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=DPlusDPrimeModel(),
    )
    film = tdgl.Polygon("film", points=box(1.5)).resample(31)
    device = tdgl.Device("d-plus-d-prime-guards", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.5, smooth=5)
    with pytest.raises(ValueError, match="prescribed vector potential"):
        tdgl.solve(
            device,
            SolverOptions(
                solve_time=1e-4,
                include_screening=True,
                terminal_psi=None,
                output_file=str(tmp_path / "screening.h5"),
            ),
        )
    with pytest.raises(ValueError, match="disorder_epsilon"):
        tdgl.solve(
            device,
            SolverOptions(
                solve_time=1e-4,
                terminal_psi=None,
                output_file=str(tmp_path / "disorder.h5"),
            ),
            disorder_epsilon=0.5,
        )
