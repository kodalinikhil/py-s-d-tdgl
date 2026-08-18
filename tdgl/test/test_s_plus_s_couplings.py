import numpy as np
import pytest

from tdgl.finite_volume.mesh import Mesh
from tdgl.finite_volume.operators import MeshOperators
from tdgl.solver.options import SparseSolver


@pytest.fixture(scope="module")
def mesh():
    """A deterministic acute triangular mesh with nonzero dual edges."""
    angles = np.arange(6) * np.pi / 3
    sites = np.vstack(
        (
            [0.0, 0.0],
            np.column_stack((np.cos(angles), np.sin(angles))),
        )
    )
    elements = np.array([[0, 1 + i, 1 + ((i + 1) % 6)] for i in range(6)], dtype=int)
    return Mesh.from_triangulation(sites, elements)


@pytest.fixture
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


def random_fields(operators, seed=17):
    rng = np.random.default_rng(seed)
    nsites = len(operators.mesh.sites)
    psi1 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    psi2 = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    return rng, psi1, psi2


def test_zero_mixed_gradient_reproduces_diagonal_current(operators):
    _, psi1, psi2 = random_fields(operators)
    k2_over_k1 = 0.37
    expected = operators.get_supercurrent(psi1) + (
        k2_over_k1 * operators.get_supercurrent(psi2)
    )

    actual_default = operators.get_s_plus_s_supercurrent(
        psi1, psi2, k2_over_k1=k2_over_k1
    )
    actual_explicit = operators.get_s_plus_s_supercurrent(
        psi1,
        psi2,
        k2_over_k1=k2_over_k1,
        mixed_gradient_k12=0.0,
    )

    assert np.array_equal(actual_default, expected)
    assert np.array_equal(actual_explicit, expected)


def test_mixed_gradient_current_matches_edge_formula(operators):
    rng, psi1, psi2 = random_fields(operators, seed=23)
    link_exponents = rng.normal(scale=0.2, size=(len(operators.edges), 2))
    operators.set_link_exponents(link_exponents)
    k2_over_k1 = 0.61
    k12 = -0.24

    grad1 = operators.psi_gradient @ psi1
    grad2 = operators.psi_gradient @ psi2
    edges0 = operators.edges[:, 0]
    diagonal = operators.get_supercurrent(psi1) + (
        k2_over_k1 * operators.get_supercurrent(psi2)
    )
    mixed = (psi1.conjugate()[edges0] * grad2 + psi2.conjugate()[edges0] * grad1).imag
    expected = diagonal + k12 * operators.condensate_current_weights * mixed

    actual = operators.get_s_plus_s_supercurrent(
        psi1,
        psi2,
        k2_over_k1=k2_over_k1,
        mixed_gradient_k12=k12,
    )

    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_mixed_gradient_current_is_gauge_invariant(operators):
    rng, psi1, psi2 = random_fields(operators, seed=31)
    link_exponents = rng.normal(scale=0.15, size=(len(operators.edges), 2))
    operators.set_link_exponents(link_exponents)
    kwargs = dict(k2_over_k1=0.73, mixed_gradient_k12=0.28)
    expected = operators.get_s_plus_s_supercurrent(psi1, psi2, **kwargs)

    phase = rng.normal(scale=0.4, size=len(operators.mesh.sites))
    edges = operators.edges
    directions = operators.edge_directions
    phase_difference = phase[edges[:, 1]] - phase[edges[:, 0]]
    transformed_links = (
        link_exponents
        + (phase_difference / np.sum(directions**2, axis=1))[:, np.newaxis] * directions
    )
    phase_factor = np.exp(1j * phase)
    operators.set_link_exponents(transformed_links)
    actual = operators.get_s_plus_s_supercurrent(
        phase_factor * psi1,
        phase_factor * psi2,
        **kwargs,
    )

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_mixed_current_is_link_derivative_of_cross_energy(operators):
    rng, psi1, psi2 = random_fields(operators, seed=47)
    link_exponents = rng.normal(scale=0.1, size=(len(operators.edges), 2))
    operators.set_link_exponents(link_exponents)
    k2_over_k1 = 0.52
    k12 = 0.21
    edge_index = 3

    def cross_energy(exponents):
        operators.set_link_exponents(exponents)
        edges = operators.edges
        integrated_links = np.einsum("ij,ij->i", exponents, operators.edge_directions)
        links = np.exp(-1j * integrated_links)
        delta1 = links * psi1[edges[:, 1]] - psi1[edges[:, 0]]
        delta2 = links * psi2[edges[:, 1]] - psi2[edges[:, 0]]
        return (
            2
            * k12
            * np.sum(
                operators.psi_laplacian_weights * np.real(delta1.conjugate() * delta2)
            )
        )

    step = 1e-7
    direction = operators.edge_directions[edge_index]
    link_perturbation = np.zeros_like(link_exponents)
    link_perturbation[edge_index] = step * direction / np.dot(direction, direction)
    derivative = (
        cross_energy(link_exponents + link_perturbation)
        - cross_energy(link_exponents - link_perturbation)
    ) / (2 * step)

    operators.set_link_exponents(link_exponents)
    full_current = operators.get_s_plus_s_supercurrent(
        psi1,
        psi2,
        k2_over_k1=k2_over_k1,
        mixed_gradient_k12=k12,
    )
    diagonal_current = operators.get_s_plus_s_supercurrent(
        psi1, psi2, k2_over_k1=k2_over_k1
    )
    mixed_current = full_current - diagonal_current
    expected_derivative = (
        -2 * operators.dual_edge_lengths[edge_index] * mixed_current[edge_index]
    )

    assert derivative == pytest.approx(expected_derivative, rel=2e-8, abs=2e-9)
