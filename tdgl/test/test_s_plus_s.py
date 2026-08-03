import h5py
import numpy as np
import pytest

import tdgl
from tdgl.device.models import SPlusSModel
from tdgl.finite_volume.operators import MeshOperators
from tdgl.geometry import box
from tdgl.solution.data import DynamicsData
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
    result.build_operators()
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
    ],
)
def test_model_validation(kwargs):
    with pytest.raises(ValueError):
        SPlusSModel(**kwargs).validate()


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


def test_dissipative_step_decreases_free_energy(operators):
    model = SPlusSModel(
        a1=-1,
        a2=-0.5,
        b1=1,
        b2=1.2,
        k2_over_k1=0.7,
        josephson_gamma=0.2,
    )
    solver = make_solver(operators, model)
    nsites = len(operators.mesh.sites)
    psi1 = np.full(nsites, 0.7 + 0.1j)
    psi2 = np.full(nsites, 0.2 - 0.05j)
    before = solver.compute_s_plus_s_free_energy(psi1, psi2)
    new_psi2, new_psi1, _, _, _ = solver.adaptive_euler_step(
        0,
        psi2,
        psi1,
        np.abs(psi2) ** 2,
        np.abs(psi1) ** 2,
        np.zeros(nsites),
        np.ones(nsites),
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
        assert h5file["layer/model"].attrs["schema_version"] == 2
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


def test_unsupported_s_plus_s_paths_are_rejected(tmp_path):
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=SPlusSModel(),
    )
    film = tdgl.Polygon("film", points=box(1.5)).resample(31)
    device = tdgl.Device("s-plus-s-guards", layer=layer, film=film)
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
