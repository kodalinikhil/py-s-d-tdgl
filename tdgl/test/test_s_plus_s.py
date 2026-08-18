import h5py
import numpy as np
import pytest

import tdgl
from tdgl.device.models import SPlusSModel
from tdgl.finite_volume.operators import MeshOperators
from tdgl.geometry import box
from tdgl.solution.data import DynamicsData
from tdgl.solver.options import SolverOptions, SparseSolver
from tdgl.solver.solver import TDGLSolver, _s_plus_s_uniform_state


@pytest.fixture(scope="module")
def mesh():
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
    )
    film = tdgl.Polygon("film", points=box(3)).resample(81)
    device = tdgl.Device("s-plus-s-test", layer=layer, film=film)
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
    result.build_operators(build_magnetic_diffusion=True)
    result.set_link_exponents(np.zeros((len(mesh.edge_mesh.edges), 2)))
    return result


def make_solver(operators, model, *, adaptive=False):
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
    return solver


@pytest.mark.parametrize(
    "kwargs",
    [
        {"b1": 0},
        {"b2": 0},
        {"k2_over_k1": 0},
        {"relaxation1": 0},
        {"relaxation2": 0},
        {"em_coupling": 0},
        {"beta_em": 0},
        {"mixed_gradient_k12": 1},
        {"phase_gamma2": 2, "density_gamma3": 0},
    ],
)
def test_model_validation(kwargs):
    with pytest.raises(ValueError):
        SPlusSModel(**kwargs).validate()


def test_zero_weak_band_quartic_is_allowed_for_positive_quadratic_term():
    SPlusSModel(a2=0.2, b2=0).validate()


def test_paper_s_plus_is_uniform_minimum():
    model = SPlusSModel(
        a1=-1,
        a2=-1,
        b1=1,
        b2=1,
        k2_over_k1=0.5,
        phase_gamma2=0.5,
        mixed_gradient_k12=0.5,
    )
    psi1, psi2 = _s_plus_s_uniform_state(model)
    assert abs(psi1) == pytest.approx(np.sqrt(2), rel=2e-6)
    assert abs(psi2) == pytest.approx(np.sqrt(2), rel=2e-6)
    assert np.angle(psi2 * np.conj(psi1)) == pytest.approx(np.pi / 2, abs=2e-6)


def test_uncoupled_uniform_equilibrium(operators):
    model = SPlusSModel(a1=-1, a2=-1, b1=1, b2=1)
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    psi1 = np.ones(nsites, dtype=complex)
    psi2 = np.ones(nsites, dtype=complex)
    result = solver.adaptive_euler_step(
        0,
        psi2,
        psi1,
        np.ones(nsites),
        np.ones(nsites),
        np.zeros(nsites),
        np.ones(nsites),
        1e-4,
    )
    assert np.allclose(result[0], psi2, atol=1e-13)
    assert np.allclose(result[1], psi1, atol=1e-13)


@pytest.mark.parametrize("gamma, expected_sign", [(0.5, 1), (-0.5, -1)])
def test_josephson_sign_matches_paper(operators, gamma, expected_sign):
    model = SPlusSModel(
        a1=0,
        a2=0,
        b1=1,
        b2=1,
        josephson_gamma=gamma,
    )
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    psi1 = np.full(nsites, 0.1, dtype=complex)
    psi2 = np.zeros(nsites, dtype=complex)
    result = solver.adaptive_euler_step(
        0,
        psi2,
        psi1,
        np.zeros(nsites),
        np.full(nsites, 0.01),
        np.zeros(nsites),
        np.ones(nsites),
        1e-4,
    )
    assert np.all(expected_sign * result[0].real > 0)


def test_complete_s_plus_s_rhs_includes_quartic_gradient_and_disorder_terms(
    operators,
):
    model = SPlusSModel(
        a1=-0.7,
        a2=-0.4,
        b1=1.1,
        b2=0.9,
        k2_over_k1=0.8,
        josephson_gamma=0.12,
        relaxation1=1.7,
        relaxation2=0.9,
        phase_gamma2=0.2,
        density_gamma3=0.3,
        mixed_gradient_k12=0.15,
        disorder_coupling1=0.4,
        disorder_coupling2=-0.2,
    )
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    psi1 = np.full(nsites, 0.6 + 0.2j)
    psi2 = np.full(nsites, 0.3 - 0.4j)
    rho1 = np.abs(psi1) ** 2
    rho2 = np.abs(psi2) ** 2
    epsilon = np.linspace(0.55, 1.0, nsites)
    lap1 = operators.psi_laplacian @ psi1
    lap2 = operators.psi_laplacian @ psi2
    a1_effective = model.a1 + model.disorder_coupling1 * (1 - epsilon)
    a2_effective = model.a2 + model.disorder_coupling2 * (1 - epsilon)
    rhs1 = (
        lap1
        + model.mixed_gradient_k12 * lap2
        - a1_effective * psi1
        - model.b1 * rho1 * psi1
        - 0.5 * model.density_gamma3 * rho2 * psi1
        - model.phase_gamma2 * np.conj(psi1) * psi2**2
        + model.josephson_gamma * psi2
    )
    rhs2 = (
        model.k2_over_k1 * lap2
        + model.mixed_gradient_k12 * lap1
        - a2_effective * psi2
        - model.b2 * rho2 * psi2
        - 0.5 * model.density_gamma3 * rho1 * psi2
        - model.phase_gamma2 * np.conj(psi2) * psi1**2
        + model.josephson_gamma * psi1
    )
    dt = 1e-5

    new_psi2, new_psi1, _, _, accepted_dt = solver.adaptive_euler_step(
        0,
        psi2,
        psi1,
        rho2,
        rho1,
        np.zeros(nsites),
        epsilon,
        dt,
    )

    assert accepted_dt == dt
    assert np.allclose(new_psi1, psi1 + dt * rhs1 / model.relaxation1)
    assert np.allclose(new_psi2, psi2 + dt * rhs2 / model.relaxation2)


def test_s_plus_s_current_uses_stiffness_ratio(operators):
    rng = np.random.default_rng(11)
    nsites = len(operators.mesh.sites)
    psi1 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    psi2 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    ratio = 0.35
    actual = operators.get_s_plus_s_supercurrent(psi1, psi2, k2_over_k1=ratio)
    expected = operators.get_supercurrent(psi1) + ratio * operators.get_supercurrent(
        psi2
    )
    assert np.allclose(actual, expected)


def test_s_plus_s_observable_current_uses_all_normalization_factors(operators):
    model = SPlusSModel(
        k2_over_k1=0.6,
        mixed_gradient_k12=0.2,
        em_coupling=1.5,
        beta_em=2.5,
    )
    solver = make_solver(operators, model)
    rng = np.random.default_rng(13)
    nsites = len(operators.mesh.sites)
    psi1 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    psi2 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    operators.set_link_exponents(np.zeros((len(operators.edges), 2)))
    expected = operators.get_s_plus_s_supercurrent(
        psi1,
        psi2,
        k2_over_k1=model.k2_over_k1,
        mixed_gradient_k12=model.mixed_gradient_k12,
    ) / (model.em_coupling * model.beta_em)

    _, actual, _ = solver.solve_for_observables(psi2, psi1, 0.0)

    assert np.allclose(actual, expected)


def test_s_plus_s_local_vector_potential_step_uses_mixed_current(operators):
    model = SPlusSModel(
        k2_over_k1=0.7,
        mixed_gradient_k12=-0.2,
        beta_em=1.8,
    )
    solver = make_solver(operators, model)
    solver.device = type("DeviceStub", (), {"kappa": 2.0, "mesh": operators.mesh})()
    solver.normalized_directions = operators.mesh.edge_mesh.normalized_directions
    solver.num_edges = len(operators.edges)
    solver._s_plus_d_magnetic_dt = None
    solver._s_plus_d_magnetic_lu = None
    rng = np.random.default_rng(19)
    nsites = len(operators.mesh.sites)
    psi1 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    psi2 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    zero_vector_potential = np.zeros((len(operators.edges), 2))
    operators.set_link_exponents(zero_vector_potential)
    raw_current = operators.get_s_plus_s_supercurrent(
        psi1,
        psi2,
        k2_over_k1=model.k2_over_k1,
        mixed_gradient_k12=model.mixed_gradient_k12,
    )
    dt = 1e-5

    updated, dA_dt = solver.advance_s_plus_s_vector_potential(
        psi2,
        psi1,
        zero_vector_potential,
        zero_vector_potential,
        dt,
    )

    tangent = solver.normalized_directions
    updated_tangent = np.einsum("ij,ij->i", updated, tangent)
    rate = dt * solver.device.kappa**2 / model.beta_em
    residual = updated_tangent + rate * (operators.magnetic_diffusion @ updated_tangent)
    assert np.allclose(residual, dt * raw_current / model.beta_em)
    assert np.allclose(dA_dt, updated_tangent / dt)


def test_dissipative_step_decreases_free_energy(operators):
    model = SPlusSModel(
        a1=-1,
        a2=-0.5,
        b1=1,
        b2=1.2,
        k2_over_k1=0.7,
        josephson_gamma=0.2,
        phase_gamma2=0.12,
        density_gamma3=0.25,
        mixed_gradient_k12=0.15,
        disorder_coupling1=0.3,
        disorder_coupling2=-0.1,
    )
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    rng = np.random.default_rng(29)
    psi1 = np.full(nsites, 0.7 + 0.1j) + 0.01 * (
        rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    )
    psi2 = np.full(nsites, 0.2 - 0.05j) + 0.01 * (
        rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    )
    solver.epsilon = np.linspace(0.6, 1.0, nsites)
    before = solver.compute_s_plus_s_free_energy(psi1, psi2)
    new_psi2, new_psi1, _, _, _ = solver.adaptive_euler_step(
        0,
        psi2,
        psi1,
        np.abs(psi2) ** 2,
        np.abs(psi1) ** 2,
        np.zeros(nsites),
        solver.epsilon,
        1e-4,
    )
    after = solver.compute_s_plus_s_free_energy(new_psi1, new_psi2)
    assert after < before


def test_layer_round_trip(tmp_path):
    model = SPlusSModel(
        a1=-0.8,
        a2=0.2,
        b1=1.1,
        b2=0.9,
        k2_over_k1=0.4,
        josephson_gamma=0.3,
        relaxation1=2,
        relaxation2=3,
        em_coupling=1.5,
        phase_gamma2=0.2,
        density_gamma3=0.4,
        mixed_gradient_k12=-0.1,
        beta_em=2.5,
        disorder_coupling1=0.7,
        disorder_coupling2=-0.3,
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
        assert h5file["layer/model"].attrs["schema_version"] == 3
    assert loaded == layer


def test_legacy_layer_migration(tmp_path):
    path = tmp_path / "legacy-s-plus-s.h5"
    with h5py.File(path, "w") as h5file:
        layer_group = h5file.create_group("layer")
        layer_group.attrs.update(
            london_lambda=2,
            coherence_length=1,
            thickness=0.1,
            u=5.79,
        )
        model_group = layer_group.create_group("model")
        model_group.attrs.update(
            type="SPlusSModel",
            eta1=2,
            eta2=3,
            alpha1=0.8,
            alpha2=0.4,
            beta1=1.1,
            beta2=0.9,
            mass_ratio_2=2.5,
            gamma_j=0.3,
        )
    with h5py.File(path, "r") as h5file:
        with pytest.warns(UserWarning, match="Migrating legacy SPlusSModel"):
            loaded = tdgl.Layer.from_hdf5(h5file["layer"])
    assert loaded.model == SPlusSModel(
        a1=-0.8,
        a2=-0.4,
        b1=1.1,
        b2=0.9,
        k2_over_k1=0.4,
        josephson_gamma=-0.3,
        relaxation1=2,
        relaxation2=3,
        em_coupling=1,
    )


def test_short_solve_uses_canonical_component_storage(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=SPlusSModel(
            a1=-1,
            a2=-0.5,
            b1=1,
            b2=1,
            k2_over_k1=0.8,
            josephson_gamma=0.1,
        ),
    )
    film = tdgl.Polygon("film", points=box(2)).resample(41)
    device = tdgl.Device(
        "s-plus-s-smoke",
        layer=layer,
        film=film,
        probe_points=[(-0.1, 0), (0.1, 0)],
    )
    device.make_mesh(max_edge_length=0.5, smooth=10)
    path = tmp_path / "s-plus-s.h5"
    solution = tdgl.solve(
        device,
        SolverOptions(
            solve_time=2e-4,
            dt_init=1e-4,
            dt_max=1e-4,
            adaptive=False,
            terminal_psi=None,
            save_every=1,
            progress_interval=1,
            output_file=str(path),
        ),
    )
    assert np.all(np.isfinite(solution.tdgl_data.psi1))
    assert np.all(np.isfinite(solution.tdgl_data.psi2))
    dynamics = DynamicsData.from_solution(solution.path)
    probe_index = solution.device.mesh.closest_site(np.array([-0.1, 0]))
    assert np.isclose(
        dynamics.theta[0, -1], np.angle(solution.tdgl_data.psi1[probe_index])
    )
    with h5py.File(solution.path, "r") as h5file:
        last_step = h5file["data"][str(len(h5file["data"]) - 1)]
        assert "psi1" in last_step
        assert "psi2" in last_step
        assert "psi_s" not in last_step
        assert "psi_d" not in last_step


def test_s_plus_s_local_screening_and_disorder_mapping(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=SPlusSModel(),
    )
    film = tdgl.Polygon("film", points=box(1.5)).resample(31)
    device = tdgl.Device("s-plus-s-guards", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.5, smooth=5)
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

    screened = tdgl.solve(
        device,
        SolverOptions(
            solve_time=1e-4,
            dt_init=1e-4,
            dt_max=1e-4,
            adaptive=False,
            include_screening=True,
            terminal_psi=None,
            save_every=1,
            output_file=str(tmp_path / "screening.h5"),
        ),
    )
    assert np.all(np.isfinite(screened.tdgl_data.induced_vector_potential))

    seed_psi1 = screened.tdgl_data.psi1.copy()
    screened.tdgl_data.mu.fill(50)
    continued = tdgl.solve(
        device,
        SolverOptions(
            solve_time=1e-4,
            dt_init=1e-4,
            dt_max=1e-4,
            adaptive=False,
            include_screening=True,
            terminal_psi=None,
            save_every=1,
            output_file=str(tmp_path / "screening-seed.h5"),
        ),
        seed_solution=screened,
    )
    phase_change = np.angle(continued.tdgl_data.psi1 * np.conj(seed_psi1))
    assert np.allclose(phase_change, 0, atol=1e-13)

    device.layer.model = SPlusSModel(em_coupling=1.2)
    with pytest.raises(ValueError, match="requires em_coupling=1"):
        tdgl.solve(
            device,
            SolverOptions(
                solve_time=1e-4,
                include_screening=True,
                terminal_psi=None,
                output_file=str(tmp_path / "invalid-screening-normalization.h5"),
            ),
        )

    device.layer.model = SPlusSModel(
        disorder_coupling1=0.8,
        disorder_coupling2=0.2,
    )
    disordered = tdgl.solve(
        device,
        SolverOptions(
            solve_time=1e-4,
            dt_init=1e-4,
            dt_max=1e-4,
            adaptive=False,
            terminal_psi=None,
            save_every=1,
            output_file=str(tmp_path / "mapped-disorder.h5"),
        ),
        disorder_epsilon=0.5,
    )
    assert np.mean(np.abs(disordered.tdgl_data.psi1)) < 1
    assert np.mean(np.abs(disordered.tdgl_data.psi2)) < 1
