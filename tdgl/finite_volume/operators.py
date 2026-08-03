import warnings
from typing import Callable, Tuple, Union

import numpy as np
import scipy.sparse as sp

try:
    import cupy  # type: ignore
except ImportError:
    cupy = None
else:
    from cupyx.scipy.sparse import csc_matrix, csr_matrix  # type: ignore
    from cupyx.scipy.sparse.linalg import factorized  # type: ignore

from ..solver.options import SparseSolver
from .mesh import Mesh


def _get_spmatrix_offsets_cupy(spmatrix, i, j):
    """Calculates the sparse matrix offsets for a set of rows ``i`` and columns ``j``."""
    # See _set_many() at
    # https://github.com/cupy/cupy/blob/5c32e40af32f6f9627e09d47ecfeb7e9281ccab2/cupyx/scipy/sparse/_compressed.py#L525
    i, j, M, N = spmatrix._prepare_indices(i, j)
    new_sp = csr_matrix(
        (
            cupy.arange(spmatrix.nnz, dtype=cupy.float32),
            spmatrix.indices,
            spmatrix.indptr,
        ),
        shape=(M, N),
    )
    offsets = new_sp._get_arrayXarray(i, j, not_found_val=-1).astype(cupy.int32).ravel()
    return offsets, (i, j, M, N)


def _spmatrix_set_many(spmatrix, i, j, x):
    """spmatrix.__setitem__()"""
    if sp.issparse(spmatrix):
        spmatrix[i, j] = x
        return

    i, j = spmatrix._swap(i, j)
    offsets, (i, j, M, N) = _get_spmatrix_offsets_cupy(spmatrix, i, j)

    mask = offsets > -1
    # update where possible
    spmatrix.data[offsets[mask]] = x[mask]

    if not mask.all():
        # only insertions remain
        mask = ~mask
        i = i[mask]
        i[i < 0] += M
        j = j[mask]
        j[j < 0] += N
        spmatrix._insert_many(i, j, x[mask])


def build_divergence(mesh: Mesh) -> sp.csr_array:
    """Build the divergence matrix that takes the divergence of a function living
    on the edges onto the sites.

    Args:
        mesh: The mesh.

    Returns:
        The divergence matrix.
    """
    edge_mesh = mesh.edge_mesh
    # Indices for each edge
    edge_indices = np.arange(len(edge_mesh.edges))
    # Compute the weights for each edge
    weights = edge_mesh.dual_edge_lengths
    # Rows and cols to update
    edges0 = edge_mesh.edges[:, 0]
    edges1 = edge_mesh.edges[:, 1]
    rows = np.concatenate([edges0, edges1])
    cols = np.concatenate([edge_indices, edge_indices])
    values = np.concatenate(
        [weights / mesh.areas[edges0], -weights / mesh.areas[edges1]]
    )
    return sp.csr_array(
        (values, (rows, cols)), shape=(len(mesh.sites), len(edge_mesh.edges))
    )


def build_gradient(
    mesh: Mesh,
    link_exponents: Union[np.ndarray, None] = None,
    weights: Union[np.ndarray, None] = None,
) -> sp.csr_array:
    """Build the gradient for a function living on the sites onto the edges.

    Args:
        mesh: The mesh.
        link_exponents: The value is integrated, exponentiated and used as
            a link variable.

    Returns:
        The gradient matrix.
    """
    edge_mesh = mesh.edge_mesh
    edge_indices = np.arange(len(edge_mesh.edges))
    if weights is None:
        weights = 1 / edge_mesh.edge_lengths
    if link_exponents is None:
        link_variable_weights = np.ones(len(weights))
    else:
        link_variable_weights = np.exp(
            -1j * np.einsum("ij, ij -> i", link_exponents, edge_mesh.directions)
        )
    rows = np.concatenate([edge_indices, edge_indices])
    cols = np.concatenate([edge_mesh.edges[:, 1], edge_mesh.edges[:, 0]])
    values = np.concatenate([link_variable_weights * weights, -weights])
    return sp.csr_array(
        (values, (rows, cols)), shape=(len(edge_mesh.edges), len(mesh.sites))
    )


def build_laplacian(
    mesh: Mesh,
    link_exponents: Union[np.ndarray, None] = None,
    fixed_sites: Union[np.ndarray, None] = None,
    free_rows: Union[np.ndarray, None] = None,
    fixed_sites_eigenvalues: float = 1,
    weights: Union[np.ndarray, None] = None,
) -> Tuple[sp.csc_array, np.ndarray]:
    """Build a Laplacian matrix on a given mesh.

    The default boundary condition is homogenous Neumann conditions. To get
    Dirichlet conditions, add fixed sites. To get non-homogenous Neumann condition,
    the flux needs to be specified using a Neumann boundary Laplacian matrix.

    Args:
        mesh: The mesh.
        link_exponents: The value is integrated, exponentiated and used as a
            link variable.
        fixed_sites: The sites to hold fixed.
        fixed_sites_eigenvalues: The eigenvalues for the fixed sites.

    Returns:
        The Laplacian matrix and indices of non-fixed rows.
    """
    if fixed_sites is None:
        fixed_sites = np.array([], dtype=int)

    edge_mesh = mesh.edge_mesh
    if weights is None:
        weights = edge_mesh.dual_edge_lengths / edge_mesh.edge_lengths
    if link_exponents is None:
        link_variable_weights = np.ones(len(weights))
    else:
        link_variable_weights = np.exp(
            -1j * np.einsum("ij, ij -> i", link_exponents, edge_mesh.directions)
        )
    edges0 = edge_mesh.edges[:, 0]
    edges1 = edge_mesh.edges[:, 1]
    rows = np.concatenate([edges0, edges1, edges0, edges1])
    cols = np.concatenate([edges1, edges0, edges0, edges1])
    areas0 = mesh.areas[edges0]
    areas1 = mesh.areas[edges1]
    values = np.concatenate(
        [
            weights * link_variable_weights / areas0,
            weights * link_variable_weights.conjugate() / areas1,
            -weights / areas0,
            -weights / areas1,
        ]
    )
    # Exclude all edges connected to fixed sites and set the
    # fixed site diagonal elements separately.
    if free_rows is None:
        free_rows = np.isin(rows, fixed_sites, invert=True)
    rows = rows[free_rows]
    cols = cols[free_rows]
    values = values[free_rows]
    rows = np.concatenate([rows, fixed_sites])
    cols = np.concatenate([cols, fixed_sites])
    values = np.concatenate(
        [values, fixed_sites_eigenvalues * np.ones(len(fixed_sites))]
    )
    laplacian = sp.csc_array(
        (values, (rows, cols)), shape=(len(mesh.sites), len(mesh.sites))
    )
    return laplacian, free_rows


def build_neumann_boundary_laplacian(
    mesh: Mesh, fixed_sites: Union[np.ndarray, None] = None
) -> sp.csr_array:
    """Build extra matrix for the Laplacian to set non-homogenous Neumann
    boundary conditions.

    Args:
        mesh: The mesh.
        fixed_sites: The fixed sites.

    Returns:
        The Neumann boundary matrix.
    """

    edge_mesh = mesh.edge_mesh
    boundary_index = np.arange(len(edge_mesh.boundary_edge_indices))
    # Get the boundary edges which are stored in the beginning of
    # the edge vector
    boundary_edges = edge_mesh.edges[edge_mesh.boundary_edge_indices]
    boundary_edges_length = edge_mesh.edge_lengths[edge_mesh.boundary_edge_indices]
    # Rows and cols to update
    rows = np.concatenate([boundary_edges[:, 0], boundary_edges[:, 1]])
    cols = np.concatenate([boundary_index, boundary_index])
    # The values
    values = np.concatenate(
        [
            boundary_edges_length / (2 * mesh.areas[boundary_edges[:, 0]]),
            boundary_edges_length / (2 * mesh.areas[boundary_edges[:, 1]]),
        ]
    )
    # Build the matrix
    neumann_laplacian = sp.csr_array(
        (values, (rows, cols)), shape=(len(mesh.sites), len(boundary_index))
    )
    # Change the rows corresponding to fixed sites to identity
    if fixed_sites is not None:
        # Convert laplacian to list of lists
        # This makes it quick to do slices
        neumann_laplacian = neumann_laplacian.tolil()
        # Change the rows corresponding to the fixed sites
        neumann_laplacian[fixed_sites, :] = 0

    return neumann_laplacian.tocsr(copy=False)


def build_directional_laplacian_weights(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Build finite-element edge weights for ``partial_x**2`` and
    ``partial_y**2``.

    The two directional stiffness matrices are assembled from the gradients
    of the linear basis functions on each triangle. Their sum is the usual
    cotangent/Voronoi Laplacian weight. Using these weights with the same link
    variables as the isotropic operator preserves exact discrete gauge
    covariance and gives a variational natural boundary condition.
    """
    sites = mesh.sites
    elements = mesh.elements
    edge_lookup = {
        tuple(edge): index for index, edge in enumerate(mesh.edge_mesh.edges)
    }
    weights_x = np.zeros(len(edge_lookup), dtype=float)
    weights_y = np.zeros(len(edge_lookup), dtype=float)

    for element in elements:
        coords = sites[element]
        edge_1 = coords[1] - coords[0]
        edge_2 = coords[2] - coords[0]
        twice_area = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if np.isclose(twice_area, 0):
            raise ValueError("Cannot build operators on a degenerate triangle.")
        area = abs(twice_area) / 2
        grad_x = (
            np.array(
                [
                    coords[1, 1] - coords[2, 1],
                    coords[2, 1] - coords[0, 1],
                    coords[0, 1] - coords[1, 1],
                ]
            )
            / twice_area
        )
        grad_y = (
            np.array(
                [
                    coords[2, 0] - coords[1, 0],
                    coords[0, 0] - coords[2, 0],
                    coords[1, 0] - coords[0, 0],
                ]
            )
            / twice_area
        )
        for local_i, local_j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((element[local_i], element[local_j])))
            edge_index = edge_lookup[edge]
            # The Laplacian is minus the FEM stiffness matrix.
            weights_x[edge_index] -= area * grad_x[local_i] * grad_x[local_j]
            weights_y[edge_index] -= area * grad_y[local_i] * grad_y[local_j]

    return weights_x, weights_y


def build_magnetic_field_curl(mesh: Mesh) -> Tuple[sp.csr_array, sp.csr_array]:
    """Build operators that map an edge-centered vector potential to site fields.

    The magnetic induction on each triangle is evaluated by the discrete Stokes
    theorem using the same edge-center line integrals as the link variables.
    Triangle values are then area averaged onto their incident mesh sites. For
    a linear vector potential (including the symmetric gauge for a uniform
    field), this construction returns the exact curl.
    """
    sites = mesh.sites
    elements = mesh.elements
    edge_mesh = mesh.edge_mesh
    edge_lookup = {tuple(edge): i for i, edge in enumerate(edge_mesh.edges)}

    triangle_areas = np.empty(len(elements), dtype=float)
    oriented_elements = np.empty_like(elements)
    for index, element in enumerate(elements):
        coords = sites[element]
        edge_1 = coords[1] - coords[0]
        edge_2 = coords[2] - coords[0]
        twice_area = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if np.isclose(twice_area, 0):
            raise ValueError("Cannot build curl on a degenerate triangle.")
        triangle_areas[index] = abs(twice_area) / 2
        oriented_elements[index] = element if twice_area > 0 else element[[0, 2, 1]]

    incident_areas = np.bincount(
        elements.ravel(),
        weights=np.repeat(triangle_areas, 3),
        minlength=len(sites),
    )
    rows = []
    cols = []
    values_x = []
    values_y = []
    for element in oriented_elements:
        for start, end in (
            (element[0], element[1]),
            (element[1], element[2]),
            (element[2], element[0]),
        ):
            edge_index = edge_lookup[tuple(sorted((start, end)))]
            stored_start, stored_end = edge_mesh.edges[edge_index]
            sign = 1 if (stored_start == start and stored_end == end) else -1
            dx, dy = sign * edge_mesh.directions[edge_index]
            for site_index in element:
                rows.append(site_index)
                cols.append(edge_index)
                values_x.append(dx / incident_areas[site_index])
                values_y.append(dy / incident_areas[site_index])

    shape = (len(sites), len(edge_mesh.edges))
    curl_x = sp.csr_array((values_x, (rows, cols)), shape=shape)
    curl_y = sp.csr_array((values_y, (rows, cols)), shape=shape)
    return curl_x, curl_y


class MeshOperators:
    """A container for the finite volume operators for a given mesh.

    Args:
        mesh: The :class:`tdgl.finite_volume.Mesh` instance for which to construct
            operators.
        sparse_solver: The sparse solver for which to build mesh operators.
        use_cupy: Use CuPy for linear algebra.
        fixed_sites: The indices of any sites for which the value of :math:`\\psi`
            and :math:`\\mu` are fixed as boundary conditions.

    Attributes:
        laplacian_x: Directional Laplacian matrix for the x-direction.
        laplacian_y: Directional Laplacian matrix for the y-direction.
    """

    def __init__(
        self,
        mesh: Mesh,
        sparse_solver: SparseSolver,
        use_cupy: bool = False,
        fixed_sites: Union[np.ndarray, None] = None,
        fix_psi: bool = True,
    ):
        self.mesh = mesh
        self.areas = mesh.areas
        edge_mesh = mesh.edge_mesh
        self.edges = edge_mesh.edges
        self.edge_directions = edge_mesh.directions
        self.use_cupy = use_cupy
        self.sparse_solver = sparse_solver
        self.fixed_sites = fixed_sites
        self.fix_psi = fix_psi
        self.mu_reference_index = 0
        self.laplacian_free_rows: Union[np.ndarray, None] = None
        self.divergence: Union[sp.spmatrix, None] = None
        self.mu_laplacian: Union[sp.spmatrix, None] = None
        self.mu_boundary_laplacian: Union[sp.spmatrix, None] = None
        self.mu_laplacian_lu: Union[Callable, None] = None
        self.psi_gradient: Union[sp.spmatrix, None] = None
        self.psi_laplacian: Union[sp.spmatrix, None] = None
        self.laplacian_x: Union[sp.spmatrix, None] = None
        self.laplacian_y: Union[sp.spmatrix, None] = None
        self.magnetic_field_curl_x: Union[sp.spmatrix, None] = None
        self.magnetic_field_curl_y: Union[sp.spmatrix, None] = None
        self.link_exponents: Union[np.ndarray, None] = None
        # Compute these quantities just once, as they never change.
        self.gradient_weights = 1 / edge_mesh.edge_lengths
        self.laplacian_weights = edge_mesh.dual_edge_lengths / edge_mesh.edge_lengths

        (
            self.laplacian_weights_x,
            self.laplacian_weights_y,
        ) = build_directional_laplacian_weights(mesh)
        directional_sum = self.laplacian_weights_x + self.laplacian_weights_y
        # pyTDGL uses positive boundary dual lengths, while a boundary cotangent
        # weight can be negative for an obtuse boundary triangle. Normalize the
        # FEM x/y split edgewise so its sum remains exactly the established
        # pyTDGL isotropic operator.
        nonzero = ~np.isclose(directional_sum, 0)
        scale = np.ones_like(directional_sum)
        scale[nonzero] = self.laplacian_weights[nonzero] / directional_sum[nonzero]
        self.laplacian_weights_x *= scale
        self.laplacian_weights_y *= scale
        zero = ~nonzero & ~np.isclose(self.laplacian_weights, 0)
        if np.any(zero):
            unit_x_sq = edge_mesh.normalized_directions[zero, 0] ** 2
            self.laplacian_weights_x[zero] = self.laplacian_weights[zero] * unit_x_sq
            self.laplacian_weights_y[zero] = self.laplacian_weights[zero] * (
                1 - unit_x_sq
            )
        self.mixed_current_weights = np.divide(
            self.laplacian_weights_y - self.laplacian_weights_x,
            self.laplacian_weights,
            out=np.zeros_like(self.laplacian_weights),
            where=~np.isclose(self.laplacian_weights, 0),
        )
        self.gradient_link_rows = np.arange(len(edge_mesh.edges), dtype=int)
        self.gradient_link_cols = edge_mesh.edges[:, 1]
        self.laplacian_link_rows = np.concatenate(
            [edge_mesh.edges[:, 0], edge_mesh.edges[:, 1]]
        )
        self.laplacian_link_cols = np.concatenate(
            [edge_mesh.edges[:, 1], edge_mesh.edges[:, 0]]
        )

    def build_operators(self) -> None:
        """Construct the vector potential-independent operators."""
        mesh = self.mesh
        mu_fixed_sites = np.array([self.mu_reference_index], dtype=int)
        self.mu_laplacian, _ = build_laplacian(
            mesh,
            fixed_sites=mu_fixed_sites,
            weights=self.laplacian_weights,
        )
        self.mu_boundary_laplacian = build_neumann_boundary_laplacian(
            mesh, fixed_sites=mu_fixed_sites
        )
        self.mu_gradient = build_gradient(mesh, weights=self.gradient_weights)
        self.divergence = build_divergence(mesh)
        (
            self.magnetic_field_curl_x,
            self.magnetic_field_curl_y,
        ) = build_magnetic_field_curl(mesh)
        if self.use_cupy:
            assert cupy is not None
            self.mu_boundary_laplacian = csr_matrix(self.mu_boundary_laplacian)
            self.mu_gradient = csr_matrix(self.mu_gradient)
            self.divergence = csr_matrix(self.divergence)
            self.magnetic_field_curl_x = csr_matrix(self.magnetic_field_curl_x)
            self.magnetic_field_curl_y = csr_matrix(self.magnetic_field_curl_y)
            self.areas = cupy.array(self.areas)
            self.edge_directions = cupy.array(self.edge_directions)
        if self.sparse_solver is SparseSolver.CUPY:
            assert cupy is not None
            self.mu_laplacian = csc_matrix(self.mu_laplacian)
            self.mu_laplacian_lu = factorized(self.mu_laplacian)
        elif self.sparse_solver is SparseSolver.PARDISO:
            # https://github.com/loganbvh/py-tdgl/issues/74
            # https://github.com/haasad/PyPardiso/issues/68
            self.mu_laplacian = sp.csc_matrix(self.mu_laplacian)
            self.mu_laplacian_lu = None
        else:
            use_umfpack = self.sparse_solver is SparseSolver.UMFPACK
            sp.linalg.use_solver(useUmfpack=use_umfpack)
            self.mu_laplacian_lu = sp.linalg.factorized(self.mu_laplacian)

    def set_link_exponents(self, link_exponents: np.ndarray) -> None:
        """Set the link variables and construct the covarient gradient
        and Laplacian for psi.

        Args:
            link_exponents: The value is integrated, exponentiated and used as
                a link variable.
        """
        mesh = self.mesh
        xp = cupy if self.use_cupy else np
        self.link_exponents = xp.asarray(link_exponents)
        if self.psi_gradient is None:
            # Build the matrices from scratch
            self.psi_gradient = build_gradient(
                mesh,
                link_exponents=link_exponents,
                weights=self.gradient_weights,
            )
            if self.fix_psi:
                fixed_sites = self.fixed_sites
                free_rows = self.laplacian_free_rows
            else:
                fixed_sites = free_rows = None
            self.psi_laplacian, self.laplacian_free_rows = build_laplacian(
                mesh,
                link_exponents=link_exponents,
                fixed_sites=fixed_sites,
                free_rows=free_rows,
                weights=self.laplacian_weights,
            )
            self.laplacian_x, _ = build_laplacian(
                mesh,
                link_exponents=link_exponents,
                fixed_sites=fixed_sites,
                free_rows=free_rows,
                weights=self.laplacian_weights_x,
            )
            self.laplacian_y, _ = build_laplacian(
                mesh,
                link_exponents=link_exponents,
                fixed_sites=fixed_sites,
                free_rows=free_rows,
                weights=self.laplacian_weights_y,
            )
            if self.use_cupy:
                self.psi_gradient = csr_matrix(self.psi_gradient)
                self.psi_laplacian = csr_matrix(self.psi_laplacian)
                self.laplacian_x = csr_matrix(self.laplacian_x)
                self.laplacian_y = csr_matrix(self.laplacian_y)
                self.gradient_weights = cupy.asarray(self.gradient_weights)
                self.laplacian_weights = cupy.asarray(self.laplacian_weights)
                self.laplacian_weights_x = cupy.asarray(self.laplacian_weights_x)
                self.laplacian_weights_y = cupy.asarray(self.laplacian_weights_y)
                self.mixed_current_weights = cupy.asarray(self.mixed_current_weights)
            return
        # Just update the link variables
        edges = self.edges
        directions = self.edge_directions
        if self.link_exponents is None:
            link_variables = xp.ones(len(directions))
        else:
            link_variables = xp.exp(
                -1j * xp.einsum("ij, ij -> i", self.link_exponents, directions)
            )
        with warnings.catch_warnings():
            # This is faster than re-creating the sparse matrices from scratch.
            warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
            # Update gradient for psi
            values = self.gradient_weights * link_variables
            rows = self.gradient_link_rows
            cols = self.gradient_link_cols
            # self.psi_gradient[rows, cols] = values
            _spmatrix_set_many(self.psi_gradient, rows, cols, values)
            # Update Laplacian for psi
            areas = self.areas
            weights = self.laplacian_weights
            values = xp.concatenate(
                [
                    weights * link_variables / areas[edges[:, 0]],
                    weights * link_variables.conjugate() / areas[edges[:, 1]],
                ]
            )
            weights_x = self.laplacian_weights_x
            values_x = xp.concatenate(
                [
                    weights_x * link_variables / areas[edges[:, 0]],
                    weights_x * link_variables.conjugate() / areas[edges[:, 1]],
                ]
            )
            weights_y = self.laplacian_weights_y
            values_y = xp.concatenate(
                [
                    weights_y * link_variables / areas[edges[:, 0]],
                    weights_y * link_variables.conjugate() / areas[edges[:, 1]],
                ]
            )
            # Only update rows that are not fixed by boundary conditions
            if self.fix_psi:
                free_rows = self.laplacian_free_rows[: len(self.laplacian_link_rows)]
                rows = self.laplacian_link_rows[free_rows]
                cols = self.laplacian_link_cols[free_rows]
                values = values[free_rows]
                values_x = values_x[free_rows]
                values_y = values_y[free_rows]
            else:
                rows = self.laplacian_link_rows
                cols = self.laplacian_link_cols
            # self.psi_laplacian[rows, cols] = values
            _spmatrix_set_many(self.psi_laplacian, rows, cols, values)
            _spmatrix_set_many(self.laplacian_x, rows, cols, values_x)
            _spmatrix_set_many(self.laplacian_y, rows, cols, values_y)

    def get_supercurrent(self, psi: np.ndarray):
        """Compute the supercurrent on the edges.

        Args:
            psi: The value of the complex order parameter.

        Returns:
            The supercurrent at each edge.
        """
        return (psi.conjugate()[self.edges[:, 0]] * (self.psi_gradient @ psi)).imag

    def get_magnetic_field(self, vector_potential: np.ndarray) -> np.ndarray:
        """Return the dimensionless perpendicular induction on mesh sites.

        Args:
            vector_potential: Edge-centered vector potential with shape
                ``(num_edges, 2)`` in the solver's dimensionless units.
        """
        xp = cupy if self.use_cupy else np
        vector_potential = xp.asarray(vector_potential)
        expected_shape = (len(self.edges), 2)
        if vector_potential.shape != expected_shape:
            raise ValueError(
                f"vector_potential must have shape {expected_shape}, "
                f"got {vector_potential.shape}."
            )
        return self.magnetic_field_curl_x @ vector_potential[:, 0] + (
            self.magnetic_field_curl_y @ vector_potential[:, 1]
        )

    def get_s_plus_s_supercurrent(
        self,
        psi1: np.ndarray,
        psi2: np.ndarray,
        *,
        k2_over_k1: float,
    ) -> np.ndarray:
        """Return the unscaled current of two isotropic s-wave condensates.

        ``k2_over_k1`` is the ratio of the two gradient-energy coefficients,
        so the same ratio multiplies the second condensate's current.
        """
        return self.get_supercurrent(psi1) + k2_over_k1 * self.get_supercurrent(psi2)

    def get_s_plus_d_supercurrent(
        self,
        psi_d: np.ndarray,
        psi_s: np.ndarray,
        *,
        eta_s: float,
        eta_v: float,
    ) -> np.ndarray:
        """Return the complete, unscaled d+s condensate current on edges.

        The returned current uses the paper normalization, before division by
        ``beta_em``. The mixed-current multiplier is derived from the same
        directional edge energy as ``laplacian_x - laplacian_y``. It is
        negative on x-aligned edges and positive on y-aligned edges.
        """
        edges0 = self.edges[:, 0]
        grad_d = self.psi_gradient @ psi_d
        grad_s = self.psi_gradient @ psi_s
        diagonal = (psi_d.conjugate()[edges0] * grad_d).imag
        diagonal += eta_s * (psi_s.conjugate()[edges0] * grad_s).imag
        if eta_v == 0:
            return diagonal
        mixed = (
            psi_d.conjugate()[edges0] * grad_s + psi_s.conjugate()[edges0] * grad_d
        ).imag
        return diagonal + eta_v * self.mixed_current_weights * mixed
