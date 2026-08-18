import numpy as np
import pytest

from my_scripts.reproductions.reproduce_babaev_prl_105_067003 import (
    FIGURE_CURVES,
    PaperParameters,
    PinnedVortexGrid,
    homogeneous_ground_state,
)


def test_paper_coefficients_map_to_s_plus_s_model():
    parameters = PaperParameters(1, 1, alpha=0.25, beta=0.1, eta=0.35, charge=1.41)
    model = parameters.framework_model()
    assert model.a1 == -2
    assert model.b1 == 2
    assert model.a2 == pytest.approx(0.5)
    assert model.b2 == pytest.approx(0.2)
    assert model.josephson_gamma == pytest.approx(0.35)
    assert model.k2_over_k1 == 1


def test_beta_zero_ground_state_matches_paper_equation_8():
    parameters = FIGURE_CURVES[1][1]
    model = parameters.framework_model()
    u1, u2 = homogeneous_ground_state(model)
    expected_u1 = np.sqrt(1 + parameters.eta**2 / (4 * parameters.alpha))
    expected_u2 = parameters.eta * expected_u1 / (2 * parameters.alpha)
    assert u1 == pytest.approx(expected_u1)
    assert u2 == pytest.approx(expected_u2)


def test_figure4_parameters_recover_printed_ground_state_densities():
    for parameters in FIGURE_CURVES[4]:
        u1, u2 = homogeneous_ground_state(parameters.framework_model())
        assert (u2 / u1) ** 2 == pytest.approx(parameters.density_ratio, abs=0.01)
        assert u1**2 + u2**2 == pytest.approx(parameters.total_density, abs=0.01)


def test_discrete_energy_gradient_matches_directional_difference():
    parameters = FIGURE_CURVES[4][1]
    model = parameters.framework_model()
    ground_state = homogeneous_ground_state(model)
    grid = PinnedVortexGrid(width=8, points=10)
    vortices = [(-1.0, 0.0), (1.0, 0.0)]
    _, _, winding, individual_links = grid.phase_links(vortices)
    core_mask = grid.vortex_core_mask(vortices)
    fields = grid.initial_fields(
        vortices,
        model,
        parameters.charge,
        ground_state,
        individual_links,
        core_mask,
    )
    values = grid.pack(*fields)
    energy, gradient = grid.energy_and_gradient(
        values,
        winding=winding,
        model=model,
        charge=parameters.charge,
        ground_state=ground_state,
    )
    assert np.isfinite(energy)
    rng = np.random.default_rng(13)
    direction = rng.normal(size=values.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-6
    plus = grid.energy_and_gradient(
        values + step * direction,
        winding=winding,
        model=model,
        charge=parameters.charge,
        ground_state=ground_state,
    )[0]
    minus = grid.energy_and_gradient(
        values - step * direction,
        winding=winding,
        model=model,
        charge=parameters.charge,
        ground_state=ground_state,
    )[0]
    finite_difference = (plus - minus) / (2 * step)
    assert finite_difference == pytest.approx(np.dot(gradient, direction), rel=2e-6)


def test_fixed_phase_contains_requested_flux_quanta():
    grid = PinnedVortexGrid(width=10, points=12)
    _, _, winding, _ = grid.phase_links([(-1.5, 0.0), (1.5, 0.0)])
    assert np.sum(winding) == pytest.approx(4 * np.pi)
