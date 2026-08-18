from types import SimpleNamespace

import numpy as np
import pint
import pytest

import tdgl
from tdgl.device.models import SPlusSModel
from tdgl.finite_volume.mesh import Mesh
from tdgl.geometry import box
from tdgl.solution.data import get_current_scale
from tdgl.solution.solution import Solution


@pytest.fixture
def diagnostic_solution():
    model = SPlusSModel(beta_em=2.5)
    layer = tdgl.Layer(
        coherence_length=1,
        london_lambda=2,
        thickness=0.1,
        model=model,
    )
    device = tdgl.Device(
        "s-plus-s-diagnostics",
        layer=layer,
        film=tdgl.Polygon("film", points=box(1, center=(0.5, 0.5))),
    )
    sites = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    mesh = Mesh.from_triangulation(sites, [[0, 1, 2], [0, 2, 3]])
    # Exercise both element orientations without changing the edge storage.
    mesh.elements[1] = [0, 3, 2]
    device.mesh = mesh

    solution = Solution.__new__(Solution)
    solution.device = device
    solution._field_units = "mT"
    solution.tdgl_data = SimpleNamespace(
        psi1=np.ones(len(sites), dtype=complex),
        psi2=np.ones(len(sites), dtype=complex),
        applied_vector_potential=np.zeros((len(mesh.edge_mesh.edges), 2)),
        induced_vector_potential=np.zeros((len(mesh.edge_mesh.edges), 2)),
    )
    return solution


def test_relative_phase_is_gauge_invariant_and_wrapped(diagnostic_solution):
    common_phase = np.array([0.2, -1.1, 2.7, -2.4])
    relative_phase = np.array([0.4, 1.5 * np.pi, -1.25 * np.pi, 4.2 * np.pi])
    data = diagnostic_solution.tdgl_data
    data.psi1 = 1.3 * np.exp(1j * common_phase)
    data.psi2 = 0.7 * np.exp(1j * (common_phase + relative_phase))
    expected = np.angle(np.exp(1j * relative_phase))

    assert np.allclose(diagnostic_solution.relative_phase, expected)
    assert np.all(diagnostic_solution.relative_phase >= -np.pi)
    assert np.all(diagnostic_solution.relative_phase <= np.pi)

    local_gauge_phase = np.array([-2.0, 0.7, 1.8, -0.3])
    gauge_factor = np.exp(1j * local_gauge_phase)
    data.psi1 *= gauge_factor
    data.psi2 *= gauge_factor
    assert np.allclose(diagnostic_solution.relative_phase, expected)


def test_local_magnetic_induction_uses_total_stored_vector_potential(
    diagnostic_solution,
):
    edge_centers = diagnostic_solution.device.mesh.edge_mesh.centers

    def uniform_field_vector_potential(field):
        x = edge_centers[:, 0]
        y = edge_centers[:, 1]
        return np.column_stack((-0.5 * field * y, 0.5 * field * x))

    applied_field = 0.37
    induced_field = -0.12
    diagnostic_solution.tdgl_data.applied_vector_potential = (
        uniform_field_vector_potential(applied_field)
    )
    diagnostic_solution.tdgl_data.induced_vector_potential = (
        uniform_field_vector_potential(induced_field)
    )
    expected_reduced = applied_field + induced_field

    reduced = diagnostic_solution.local_magnetic_induction(
        units="Bc2", with_units=False
    )
    assert reduced.shape == (len(diagnostic_solution.device.mesh.elements),)
    assert np.allclose(reduced, expected_reduced, atol=2e-15)

    reduced_with_units = diagnostic_solution.local_magnetic_induction(units="Bc2")
    assert isinstance(reduced_with_units, pint.Quantity)
    assert reduced_with_units.dimensionless
    assert np.allclose(reduced_with_units.magnitude, expected_reduced, atol=2e-15)

    physical = diagnostic_solution.local_magnetic_induction()
    expected_physical = (expected_reduced * diagnostic_solution.device.Bc2).to("mT")
    assert isinstance(physical, pint.Quantity)
    assert np.allclose(physical.magnitude, expected_physical.magnitude, atol=2e-15)
    assert physical.units == expected_physical.units


def test_local_magnetic_induction_rejects_mismatched_stored_shapes(
    diagnostic_solution,
):
    diagnostic_solution.tdgl_data.induced_vector_potential = np.zeros((1, 2))
    with pytest.raises(ValueError, match="must have the same shape"):
        diagnostic_solution.local_magnetic_induction(units="Bc2")


def test_s_plus_s_current_scale_includes_beta_em(diagnostic_solution):
    device = diagnostic_solution.device
    ratio = (get_current_scale(device) / device.K0).to_base_units().magnitude
    assert np.isclose(ratio, device.layer.model.beta_em)
