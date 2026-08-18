import h5py
import numpy as np
import pytest

from tdgl.device.layer import Layer
from tdgl.device.models import SPlusDModel
from tdgl.magnetic_periodic.cell import MagneticPeriodicCell
from tdgl.magnetic_periodic.operators import MagneticPeriodicOperators


def make_cell(*, nx=7, ny=6, flux_quanta=2):
    return MagneticPeriodicCell(
        name="test-cell",
        layer=Layer(
            coherence_length=1.4,
            london_lambda=3.2,
            thickness=0.15,
            conductivity=4.5,
            model=SPlusDModel(eta_s=2, eta_v=-1, beta_em=3),
        ),
        length_x=5.6,
        length_y=4.2,
        nx=nx,
        ny=ny,
        flux_quanta=flux_quanta,
        origin=(-2.1, 0.7),
        length_units="um",
    )


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"length_x": 0}, ValueError),
        ({"length_y": np.inf}, ValueError),
        ({"nx": 1}, ValueError),
        ({"ny": 2.5}, TypeError),
        ({"flux_quanta": 1.5}, TypeError),
        ({"flux_quanta": True}, TypeError),
        ({"origin": (0, np.nan)}, ValueError),
    ],
)
def test_cell_validation(kwargs, exception):
    arguments = dict(
        layer=Layer(coherence_length=1, london_lambda=2, thickness=0.1),
        length_x=4,
        length_y=3,
        nx=5,
        ny=4,
        flux_quanta=1,
    )
    arguments.update(kwargs)
    with pytest.raises(exception):
        MagneticPeriodicCell(**arguments)


@pytest.mark.parametrize("attribute", ["coherence_length", "london_lambda"])
def test_cell_rejects_nonpositive_layer_scales(attribute):
    layer_kwargs = dict(coherence_length=1, london_lambda=2, thickness=0.1)
    layer_kwargs[attribute] = 0
    with pytest.raises(ValueError, match=attribute):
        MagneticPeriodicCell(
            layer=Layer(**layer_kwargs),
            lengths=(4, 3),
            shape=(4, 5),
            flux_quanta=1,
        )


def test_cell_grid_and_hdf5_roundtrip(tmp_path):
    cell = make_cell()
    assert cell.shape == (6, 7)
    assert cell.points.shape == (42, 2)
    assert cell.num_edges == 2 * cell.num_sites
    assert cell.kappa == pytest.approx(
        cell.layer.london_lambda / cell.layer.coherence_length
    )
    assert np.allclose(np.diff(cell.x), cell.dx)
    assert np.allclose(np.diff(cell.y), cell.dy)
    assert cell.x[0] == pytest.approx(cell.origin[0] + cell.dx / 2)
    assert cell.y[-1] == pytest.approx(cell.origin[1] + cell.length_y - cell.dy / 2)
    assert cell.mean_induction * cell.dimensionless_area == pytest.approx(4 * np.pi)

    path = tmp_path / "cell.h5"
    with h5py.File(path, "w") as h5_file:
        cell.to_hdf5(h5_file)
    loaded = MagneticPeriodicCell.from_hdf5(path)
    assert loaded == cell
    assert loaded is not cell
    assert loaded.layer is not cell.layer

    with h5py.File(path, "r+") as h5_file:
        h5_file.attrs["schema_version"] = cell.schema_version + 1
    with pytest.raises(IOError, match="Unsupported magnetic-periodic cell schema"):
        MagneticPeriodicCell.from_hdf5(path)


def test_documented_tuple_constructor_and_site_reshaping():
    layer = Layer(coherence_length=2, london_lambda=3, thickness=0.1)
    cell = MagneticPeriodicCell(
        layer=layer,
        lengths=(8, 6),
        shape=(3, 4),
        flux_quanta=1,
    )
    assert cell.length_x == 8
    assert cell.length_y == 6
    assert cell.shape == (3, 4)
    assert cell.grid_points.shape == (3, 4, 2)
    assert cell.site_grid.shape == (3, 4, 2)
    flat = np.arange(cell.num_sites)
    assert np.array_equal(cell.flatten_site(cell.reshape_site(flat)), flat)
    with pytest.raises(ValueError, match="not both"):
        MagneticPeriodicCell(
            layer=layer,
            lengths=(8, 6),
            length_x=8,
            length_y=6,
            shape=(3, 4),
            flux_quanta=1,
        )


def test_link_shape_validation_and_pack_roundtrip():
    operators = MagneticPeriodicOperators(make_cell())
    assert operators.num_edges == operators.num_links
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, *operators.shape))
    operators.set_vector_potential(a)
    assert np.array_equal(operators.vector_potential, a)
    packed = operators.pack_links(a[0], a[1])
    ax, ay = operators.unpack_links(packed)
    assert np.array_equal(ax, a[0])
    assert np.array_equal(ay, a[1])
    with pytest.raises(ValueError, match="vector potential"):
        operators.set_vector_potential(np.zeros((3, *operators.shape)))
    with pytest.raises(ValueError, match="finite"):
        bad = a.copy()
        bad[0, 0, 0] = np.nan
        operators.set_vector_potential(bad)


def test_fixed_vector_potential_reuses_cached_link_variables(monkeypatch):
    operators = MagneticPeriodicOperators(make_cell())
    original_exp = np.exp
    calls = 0

    def counting_exp(values):
        nonlocal calls
        calls += 1
        return original_exp(values)

    monkeypatch.setattr("tdgl.magnetic_periodic.operators.np.exp", counting_exp)
    psi = np.ones(operators.shape, dtype=complex)
    operators.gradient(psi)
    first_count = calls
    assert first_count > 0
    operators.gradient(psi)
    operators.laplacian(psi)
    operators.set_vector_potential(np.zeros((2, *operators.shape)))
    assert calls == first_count

    changed = np.zeros((2, *operators.shape))
    changed[0, 0, 0] = 0.1
    operators.set_vector_potential(changed)
    assert calls > first_count


@pytest.mark.parametrize("flux_quanta", [-2, -1, 0, 1, 3])
def test_wilson_loops_include_seams_and_corner(flux_quanta):
    cell = make_cell(nx=8, ny=5, flux_quanta=flux_quanta)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(12)
    a = 0.15 * rng.normal(size=(2, *cell.shape))
    field = operators.magnetic_field(a)
    expected = np.exp(-1j * field * operators.cell_area)
    loops = operators.wilson_loops(a)

    assert np.allclose(loops, expected, atol=2e-13, rtol=2e-13)
    assert loops[-1, -1] == pytest.approx(expected[-1, -1], abs=2e-13)
    assert np.allclose(loops[:, -1], expected[:, -1], atol=2e-13, rtol=2e-13)
    assert np.allclose(loops[-1], expected[-1], atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("flux_quanta", [-3, 0, 1, 4])
def test_total_flux_is_exactly_fixed(flux_quanta):
    cell = make_cell(nx=9, ny=7, flux_quanta=flux_quanta)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(23)
    a = rng.normal(size=(2, *cell.shape))
    flux = np.sum(operators.magnetic_field(a)) * operators.cell_area
    assert flux / (2 * np.pi) == pytest.approx(flux_quanta, abs=2e-13)
    assert np.sum(operators.curl(a)) == pytest.approx(0, abs=2e-13)


def test_discrete_gauge_covariance_and_current_invariance():
    cell = make_cell(nx=9, ny=8, flux_quanta=2)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(4)
    a = 0.2 * rng.normal(size=(2, *cell.shape))
    psi_d = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    psi_s = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    j, i = np.indices(cell.shape)
    chi = 0.31 * np.sin(2 * np.pi * i / cell.nx) + 0.17 * np.cos(
        2 * np.pi * j / cell.ny
    )
    transformed_a = a + operators.scalar_gradient(chi)
    phase = np.exp(1j * chi)

    before_gradient = operators.gradient(psi_d, a)
    after_gradient = operators.gradient(phase * psi_d, transformed_a)
    assert np.allclose(after_gradient, phase[None] * before_gradient, atol=3e-12)
    assert np.allclose(
        operators.laplacian_x(phase * psi_d, transformed_a),
        phase * operators.laplacian_x(psi_d, a),
        atol=3e-12,
    )
    assert np.allclose(
        operators.laplacian_y(phase * psi_d, transformed_a),
        phase * operators.laplacian_y(psi_d, a),
        atol=3e-12,
    )
    before_current = operators.s_plus_d_supercurrent(
        psi_d, psi_s, eta_s=2, eta_v=-0.4, vector_potential=a
    )
    after_current = operators.s_plus_d_supercurrent(
        phase * psi_d,
        phase * psi_s,
        eta_s=2,
        eta_v=-0.4,
        vector_potential=transformed_a,
    )
    assert np.allclose(after_current, before_current, atol=3e-12)
    assert np.allclose(
        operators.magnetic_field(transformed_a),
        operators.magnetic_field(a),
        atol=3e-12,
    )


def test_isotropic_two_component_current_is_gauge_invariant():
    cell = make_cell(nx=9, ny=8, flux_quanta=2)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(14)
    vector_potential = 0.2 * rng.normal(size=(2, *cell.shape))
    psi1 = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    psi2 = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    j, i = np.indices(cell.shape)
    chi = 0.23 * np.sin(2 * np.pi * i / cell.nx) - 0.19 * np.cos(
        2 * np.pi * j / cell.ny
    )
    transformed_a = vector_potential + operators.scalar_gradient(chi)
    phase = np.exp(1j * chi)
    kwargs = dict(k1=0.8, k2=1.3, mixed_gradient=-0.27)

    current = operators.isotropic_two_component_supercurrent(
        psi1, psi2, vector_potential=vector_potential, **kwargs
    )
    transformed = operators.isotropic_two_component_supercurrent(
        phase * psi1,
        phase * psi2,
        vector_potential=transformed_a,
        **kwargs,
    )
    assert np.allclose(transformed, current, atol=4e-12)
    assert operators.get_isotropic_two_component_supercurrent(
        psi1, psi2, vector_potential=vector_potential, **kwargs
    ) == pytest.approx(current)

    diagonal = operators.isotropic_two_component_supercurrent(
        psi1,
        psi2,
        k1=kwargs["k1"],
        k2=kwargs["k2"],
        mixed_gradient=0,
        vector_potential=vector_potential,
    )
    expected = kwargs["k1"] * operators.supercurrent(psi1, vector_potential)
    expected += kwargs["k2"] * operators.supercurrent(psi2, vector_potential)
    assert diagonal == pytest.approx(expected)


@pytest.mark.parametrize("bond", [(0, 2, 4), (1, 6, 3)])
def test_isotropic_two_component_current_is_gradient_energy_derivative(bond):
    cell = make_cell(nx=8, ny=7, flux_quanta=1)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(33)
    psi1 = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    psi2 = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    vector_potential = 0.08 * rng.normal(size=(2, *cell.shape))
    k1, k2, mixed_gradient = 0.9, 1.4, -0.21

    def gradient_energy(links):
        grad1 = operators.gradient(psi1, links)
        grad2 = operators.gradient(psi2, links)
        density = k1 * np.sum(np.abs(grad1) ** 2, axis=0)
        density += k2 * np.sum(np.abs(grad2) ** 2, axis=0)
        density += 2 * mixed_gradient * np.sum(np.real(np.conj(grad1) * grad2), axis=0)
        return float(np.mean(density))

    delta = 1e-5
    plus = vector_potential.copy()
    minus = vector_potential.copy()
    plus[bond] += delta
    minus[bond] -= delta
    derivative = (gradient_energy(plus) - gradient_energy(minus)) / (2 * delta)
    current = operators.isotropic_two_component_supercurrent(
        psi1,
        psi2,
        k1=k1,
        k2=k2,
        mixed_gradient=mixed_gradient,
        vector_potential=vector_potential,
    )
    assert derivative == pytest.approx(
        -2 * current[bond] / cell.num_sites, rel=3e-7, abs=3e-8
    )


def test_periodic_magnetization_current_is_zeeman_energy_derivative():
    cell = make_cell(nx=8, ny=7, flux_quanta=1)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(39)
    magnetization = rng.normal(size=cell.shape)
    vector_potential = 0.1 * rng.normal(size=(2, *cell.shape))
    perturbation = rng.normal(size=(2, *cell.shape))

    def zeeman_energy(links):
        return float(-np.mean(magnetization * operators.magnetic_field(links)))

    delta = 1e-7
    derivative = (
        zeeman_energy(vector_potential + delta * perturbation)
        - zeeman_energy(vector_potential - delta * perturbation)
    ) / (2 * delta)
    current = operators.magnetization_current(magnetization)
    predicted = -2 * np.sum(current * perturbation) / cell.num_sites
    assert derivative == pytest.approx(predicted, rel=3e-9, abs=3e-9)
    assert operators.get_magnetization_current(magnetization) == pytest.approx(current)
    assert operators.magnetization_current(np.ones(cell.shape)) == pytest.approx(
        0, abs=2e-13
    )


def test_new_current_helpers_validate_coefficients_and_magnetization():
    operators = MagneticPeriodicOperators(make_cell())
    psi = np.ones(operators.shape, dtype=complex)
    with pytest.raises(ValueError, match="k2"):
        operators.isotropic_two_component_supercurrent(psi, psi, k2=np.nan)
    with pytest.raises(ValueError, match="magnetization"):
        operators.magnetization_current(np.ones((2, 2)))
    with pytest.raises(ValueError, match="real"):
        operators.magnetization_current(1j * np.ones(operators.shape))


def test_zero_flux_fourier_laplacian_eigenvalue():
    cell = make_cell(nx=12, ny=9, flux_quanta=0)
    operators = MagneticPeriodicOperators(cell)
    mode_x, mode_y = 3, -2
    j, i = np.indices(cell.shape)
    mode = np.exp(2j * np.pi * (mode_x * i / cell.nx + mode_y * j / cell.ny))
    expected_eigenvalue = -4 * np.sin(np.pi * mode_x / cell.nx) ** 2 / cell.hx**2
    expected_eigenvalue -= 4 * np.sin(np.pi * mode_y / cell.ny) ** 2 / cell.hy**2
    assert np.allclose(
        operators.laplacian(mode), expected_eigenvalue * mode, atol=3e-12
    )
    matrix_result = operators.scalar_laplacian @ mode.ravel()
    assert np.allclose(
        matrix_result.reshape(cell.shape), expected_eigenvalue * mode, atol=3e-12
    )


def test_curl_adjoint_diffusion_psd_and_gauge_nullspace():
    operators = MagneticPeriodicOperators(make_cell(nx=8, ny=7, flux_quanta=1))
    rng = np.random.default_rng(5)
    a = rng.normal(size=operators.num_links)
    field = rng.normal(size=operators.num_sites)
    left = np.vdot(operators.curl_matrix @ a, field)
    right = np.vdot(a, operators.magnetic_curl_gradient @ field)
    assert left == pytest.approx(right, abs=2e-12)
    quadratic = np.vdot(a, operators.magnetic_diffusion @ a).real
    assert quadratic >= -2e-12
    assert quadratic == pytest.approx(
        np.linalg.norm(operators.curl_matrix @ a) ** 2, abs=2e-11
    )

    chi = rng.normal(size=operators.shape)
    gradient = operators.scalar_gradient(chi)
    packed_gradient = operators.pack_links(gradient[0], gradient[1])
    assert np.max(np.abs(operators.curl_matrix @ packed_gradient)) < 2e-12
    assert np.max(np.abs(operators.magnetic_diffusion @ packed_gradient)) < 2e-11


@pytest.mark.parametrize("flux_quanta", [-2, -1, 0, 1, 3])
def test_total_gauge_invariant_vorticity_equals_flux_sector(flux_quanta):
    cell = make_cell(nx=10, ny=7, flux_quanta=flux_quanta)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(90 + flux_quanta)
    a = 0.2 * rng.normal(size=(2, *cell.shape))
    psi = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    charge = operators.vorticity(psi, a)
    assert np.issubdtype(charge.dtype, np.integer)
    assert np.sum(charge) == flux_quanta
    assert operators.vortex_count(psi, a) == flux_quanta


def test_vorticity_rejects_zero_site_value():
    operators = MagneticPeriodicOperators(make_cell(flux_quanta=1))
    psi = np.ones(operators.shape, dtype=complex)
    psi[0, 0] = 0
    with pytest.raises(ValueError, match="undefined"):
        operators.vorticity(psi)
