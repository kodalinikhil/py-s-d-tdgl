from types import SimpleNamespace

import numpy as np
import pytest

from my_scripts.reproductions.reproduce_li_wang_wang_cond_mat_9906211 import (
    PRESETS,
    b_over_b0_from_bc2_fraction,
    drive_options,
    equilibrium_options,
    fit_depinning_curve,
    global_vector_from_tangential,
    hc2_over_b0,
    li_vortex_seed,
    net_vortex_count,
    one_flux_cell_side,
    paper_model,
    parse_figures,
    uniform_li_state,
)


def test_mixed_hc2_conversion_uses_linearized_determinant_roots():
    expected = {
        1.0: 1.3911399976,
        0.97: 1.3881175177,
        0.85: 1.3763352775,
        0.67: 1.3595579180,
        -1.0: 1.2456497157,
    }
    for alpha_s, root in expected.items():
        assert hc2_over_b0(alpha_s) == pytest.approx(root)


def test_reproduction_scope_is_figures_two_through_four():
    assert parse_figures(["all"]) == ("2", "3", "4")
    for excluded in ("1", "5"):
        with pytest.raises(ValueError, match="Unknown figure"):
            parse_figures([excluded])


def test_expensive_q_point_is_omitted_from_figure_three():
    assert PRESETS["smoke"].fig3_relaxations == (1.0,)
    assert PRESETS["quick"].fig3_relaxations == (1.0, 10.0)
    assert PRESETS["paper"].fig3_relaxations == (1.0, 10.0)


def test_one_flux_cell_and_reduced_field_mappings():
    assert one_flux_cell_side(0.034) == pytest.approx(np.sqrt(2 * np.pi / 0.034))
    assert b_over_b0_from_bc2_fraction(0.85, 0.034) == pytest.approx(
        0.034 * hc2_over_b0(0.85)
    )


def test_d_stiffness_clock_and_relative_s_kinetics_map_the_paper():
    q = 0.2
    model = paper_model(0.85, q)

    assert model.relaxation_s == 2.0
    assert model.beta_em == pytest.approx(2 / q)
    assert model.eta_s == 2.0
    assert model.eta_v == -1.0

    preset = PRESETS["smoke"]
    equilibrium = equilibrium_options(preset, "equilibrium.h5", model)
    driven = drive_options(preset, "drive.h5", (0.1, 0.0), model)
    assert equilibrium.solve_time == pytest.approx(
        preset.equilibrium_time * model.beta_em
    )
    assert driven.skip_time == pytest.approx(preset.drive_skip_time * model.beta_em)
    assert driven.solve_time == pytest.approx(preset.drive_measure_time * model.beta_em)


def test_uniform_d_plus_is_seed_is_a_stationary_bulk_state():
    alpha_s = 0.85
    d_order, s_order = uniform_li_state(alpha_s)
    model = paper_model(alpha_s)
    d2 = abs(d_order) ** 2
    s2 = abs(s_order) ** 2

    rhs_d = (
        d_order
        - d2 * d_order
        - 0.5 * model.tau3 * s2 * d_order
        - model.tau4 * s_order**2 * np.conj(d_order)
    )
    rhs_s = (
        model.nu * s_order
        - model.tau1 * s2 * s_order
        - 0.5 * model.tau3 * d2 * s_order
        - model.tau4 * d_order**2 * np.conj(s_order)
    )
    assert rhs_d == pytest.approx(0)
    assert rhs_s == pytest.approx(0)

    d_at_endpoint, s_at_endpoint = uniform_li_state(1.0)
    assert d_at_endpoint == pytest.approx(0)
    assert abs(s_at_endpoint) == pytest.approx(np.sqrt(3 / 4))


def test_vortex_seed_has_one_net_d_winding():
    sites = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = SimpleNamespace(sites=sites, elements=elements)
    d_order, _ = li_vortex_seed(sites, 0.85, chirality=1, num_vortices=1)
    assert net_vortex_count(mesh, d_order) == 1


def test_global_vector_reconstruction_and_depinning_fit():
    sites = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    edges = np.array([[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]])
    directions = sites[edges[:, 1]] - sites[edges[:, 0]]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    edge_mesh = SimpleNamespace(
        normalized_directions=directions,
        edges=edges,
    )
    mesh = SimpleNamespace(edge_mesh=edge_mesh, sites=sites, elements=elements)
    vector = np.array([0.13, -0.27])
    reconstructed = global_vector_from_tangential(mesh, directions @ vector)
    assert reconstructed == pytest.approx(vector)

    currents = np.linspace(0.1, 0.3, 21)
    critical, amplitude = 0.14, 0.73
    resistivity = amplitude * np.sqrt(np.maximum(0.0, 1.0 - (critical / currents) ** 2))
    fitted_critical, fitted_amplitude = fit_depinning_curve(currents, resistivity)
    assert fitted_critical == pytest.approx(critical, abs=2e-4)
    assert fitted_amplitude == pytest.approx(amplitude, abs=2e-3)
