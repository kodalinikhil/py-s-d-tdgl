import numpy as np

from tdgl.finite_volume.mesh import Mesh
from tdgl.finite_volume.operators import MeshOperators
from tdgl.solver.options import SparseSolver


def test_single_triangle_mesh_from_triangulation():
    mesh = Mesh.from_triangulation(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0, 1, 2]]),
    )

    assert mesh.elements.shape == (1, 3)
    assert len(mesh.edge_mesh.edges) == 3


def test_scalar_edge_quantity_is_averaged_without_vector_normalization():
    mesh = Mesh.from_triangulation(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0, 1, 2]]),
    )
    values = np.ones(len(mesh.edge_mesh.edges))

    assert np.allclose(mesh.get_quantity_on_site(values, vector=False), 1)


def test_cocircular_mesh_does_not_require_unused_magnetic_diffusion():
    mesh = Mesh.from_triangulation(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    operators = MeshOperators(mesh, SparseSolver.SUPERLU)

    operators.build_operators()
    operators.set_link_exponents(np.zeros((len(operators.edges), 2)))

    assert operators.magnetic_diffusion is None
    vector_potential = np.zeros((len(operators.edges), 2))
    assert np.all(np.isfinite(operators.get_magnetic_field(vector_potential)))
    assert np.all(
        np.isfinite(operators.get_magnetization_current(np.ones(len(mesh.sites))))
    )
