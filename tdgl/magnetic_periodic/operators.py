"""Gauge-covariant finite differences on a magnetic-periodic rectangle."""

from __future__ import annotations

import numbers
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .cell import MagneticPeriodicCell


class MagneticPeriodicOperators:
    r"""Structured magnetic-periodic operators in a fixed flux sector.

    Site arrays use shape ``(ny, nx)``.  The periodic dynamic vector potential
    has shape ``(2, ny, nx)`` with x bonds first and y bonds second.  The fixed
    mean induction is represented by Landau-gauge links and the magnetic
    transition function, rather than by a globally periodic vector potential.
    """

    def __init__(self, cell: MagneticPeriodicCell):
        if not isinstance(cell, MagneticPeriodicCell):
            raise TypeError("cell must be a MagneticPeriodicCell.")
        self.cell = cell
        self.shape = cell.shape
        self.num_sites = cell.num_sites
        self.num_links = 2 * self.num_sites
        self.num_edges = self.num_links
        self.hx = cell.hx
        self.hy = cell.hy
        self.cell_area = self.hx * self.hy
        self.cell_areas = np.full(self.shape, self.cell_area)
        self._site_indices = np.arange(self.num_sites).reshape(self.shape)
        self._vector_potential = np.zeros((2, *self.shape), dtype=float)
        self._cached_link_variables: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self.scalar_gradient_matrix = self._build_scalar_gradient()
        self.divergence_matrix = self._build_divergence()
        self.scalar_laplacian = self.divergence_matrix @ self.scalar_gradient_matrix
        self.curl_matrix = self._build_curl()
        self.magnetic_curl_gradient = self.curl_matrix.T.tocsr()
        self.magnetic_diffusion = (
            self.magnetic_curl_gradient @ self.curl_matrix
        ).tocsr()

    @property
    def vector_potential(self) -> np.ndarray:
        """Current periodic dynamic tangent vector potential."""
        return self._vector_potential.copy()

    @property
    def link_exponents(self) -> np.ndarray:
        """Compatibility alias for the periodic dynamic vector potential."""
        return self.vector_potential

    def _build_scalar_gradient(self) -> sp.csr_array:
        p = self._site_indices
        px = np.roll(p, -1, axis=1)
        py = np.roll(p, -1, axis=0)
        rows = np.concatenate(
            [
                p.ravel(),
                p.ravel(),
                self.num_sites + p.ravel(),
                self.num_sites + p.ravel(),
            ]
        )
        cols = np.concatenate([px.ravel(), p.ravel(), py.ravel(), p.ravel()])
        values = np.concatenate(
            [
                np.full(self.num_sites, 1 / self.hx),
                np.full(self.num_sites, -1 / self.hx),
                np.full(self.num_sites, 1 / self.hy),
                np.full(self.num_sites, -1 / self.hy),
            ]
        )
        return sp.csr_array(
            (values, (rows, cols)), shape=(self.num_links, self.num_sites)
        )

    def _build_divergence(self) -> sp.csr_array:
        p = self._site_indices
        pmx = np.roll(p, 1, axis=1)
        pmy = np.roll(p, 1, axis=0)
        rows = np.concatenate([p.ravel()] * 4)
        cols = np.concatenate(
            [
                p.ravel(),
                pmx.ravel(),
                self.num_sites + p.ravel(),
                self.num_sites + pmy.ravel(),
            ]
        )
        values = np.concatenate(
            [
                np.full(self.num_sites, 1 / self.hx),
                np.full(self.num_sites, -1 / self.hx),
                np.full(self.num_sites, 1 / self.hy),
                np.full(self.num_sites, -1 / self.hy),
            ]
        )
        return sp.csr_array(
            (values, (rows, cols)), shape=(self.num_sites, self.num_links)
        )

    def _build_curl(self) -> sp.csr_array:
        p = self._site_indices
        px = np.roll(p, -1, axis=1)
        py = np.roll(p, -1, axis=0)
        rows = np.concatenate([p.ravel()] * 4)
        cols = np.concatenate(
            [
                p.ravel(),
                py.ravel(),
                self.num_sites + p.ravel(),
                self.num_sites + px.ravel(),
            ]
        )
        values = np.concatenate(
            [
                np.full(self.num_sites, 1 / self.hy),
                np.full(self.num_sites, -1 / self.hy),
                np.full(self.num_sites, -1 / self.hx),
                np.full(self.num_sites, 1 / self.hx),
            ]
        )
        return sp.csr_array(
            (values, (rows, cols)), shape=(self.num_sites, self.num_links)
        )

    def _coerce_site(self, values: np.ndarray, name: str) -> np.ndarray:
        return self.cell.reshape_site(values, name=name)

    def reshape_site(self, values: np.ndarray, *, name: str = "values") -> np.ndarray:
        """Return site data in canonical ``(ny, nx)`` shape."""
        return self._coerce_site(values, name)

    def flatten_site(self, values: np.ndarray, *, name: str = "values") -> np.ndarray:
        """Return site data packed in row-major ``p=j*nx+i`` ordering."""
        return self.cell.flatten_site(values, name=name)

    def _coerce_vector_potential(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape == (self.num_links,):
            array = array.reshape(2, *self.shape)
        elif array.shape == (2, self.num_sites):
            array = array.reshape(2, *self.shape)
        elif array.shape == (self.num_sites, 2):
            array = array.T.reshape(2, *self.shape)
        elif array.shape == (*self.shape, 2):
            array = np.moveaxis(array, -1, 0)
        if array.shape != (2, *self.shape):
            raise ValueError(
                "vector potential must have shape "
                f"(2, {self.cell.ny}, {self.cell.nx}), ({self.num_links},), "
                f"or ({self.num_sites}, 2); got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("vector potential must contain only finite values.")
        return array

    def _coerce_tangent(self, values: np.ndarray, name: str) -> np.ndarray:
        try:
            return self._coerce_vector_potential(values)
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {exc}") from exc

    def pack_links(self, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
        """Pack link grids as ``[ax.ravel(), ay.ravel()]``.

        The same ordering is used by :attr:`curl_matrix`,
        :attr:`magnetic_curl_gradient`, and :attr:`magnetic_diffusion`.
        """
        ax = self._coerce_site(ax, "ax")
        ay = self._coerce_site(ay, "ay")
        return np.concatenate([ax.ravel(), ay.ravel()])

    def unpack_links(self, packed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack sparse link ordering into x and y grids."""
        packed = np.asarray(packed)
        if packed.shape != (self.num_links,):
            raise ValueError(
                f"packed links must have shape ({self.num_links},), got {packed.shape}."
            )
        return (
            packed[: self.num_sites].reshape(self.shape),
            packed[self.num_sites :].reshape(self.shape),
        )

    def set_vector_potential(self, vector_potential: np.ndarray) -> None:
        """Install the periodic dynamic tangent vector potential."""
        values = self._coerce_vector_potential(vector_potential)
        if self._cached_link_variables is not None and np.array_equal(
            values, self._vector_potential
        ):
            return
        self._vector_potential = values.copy()
        self._cached_link_variables = self._calculate_link_variables(
            self._vector_potential
        )

    def set_link_exponents(self, vector_potential: np.ndarray) -> None:
        """Compatibility alias for :meth:`set_vector_potential`."""
        self.set_vector_potential(vector_potential)

    def _resolve_vector_potential(
        self, vector_potential: Optional[np.ndarray]
    ) -> np.ndarray:
        if vector_potential is None:
            return self._vector_potential
        return self._coerce_vector_potential(vector_potential)

    def link_variables(
        self, vector_potential: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return forward parallel transporters ``(Ux, Uy)``.

        The internal Landau gauge is ``A_background=(0, B_n x)``.  Across the
        x seam, ``Ux`` includes the transition
        ``exp(i B_n Lx y)``.  Flux quantization makes the corner cocycle close.
        """
        if vector_potential is None:
            if self._cached_link_variables is None:
                self._cached_link_variables = self._calculate_link_variables(
                    self._vector_potential
                )
            return self._cached_link_variables
        return self._calculate_link_variables(
            self._coerce_vector_potential(vector_potential)
        )

    def _calculate_link_variables(
        self, vector_potential: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build magnetic link variables for one validated correction field."""
        a = vector_potential
        ax, ay = a
        field = self.cell.background_field
        lx, _ = self.cell.dimensionless_lengths
        ux = np.exp(-1j * self.hx * ax)
        ux = ux.copy()
        ux[:, -1] *= np.exp(1j * field * lx * self.cell.relative_y)
        uy = np.exp(-1j * self.hy * (field * self.cell.relative_x[np.newaxis, :] + ay))
        ux.setflags(write=False)
        uy.setflags(write=False)
        return ux, uy

    def covariant_gradient_matrix(
        self, vector_potential: Optional[np.ndarray] = None
    ) -> sp.csr_array:
        """Return the sparse forward covariant-gradient matrix."""
        ux, uy = self.link_variables(vector_potential)
        p = self._site_indices
        px = np.roll(p, -1, axis=1)
        py = np.roll(p, -1, axis=0)
        rows = np.concatenate(
            [
                p.ravel(),
                p.ravel(),
                self.num_sites + p.ravel(),
                self.num_sites + p.ravel(),
            ]
        )
        cols = np.concatenate([px.ravel(), p.ravel(), py.ravel(), p.ravel()])
        values = np.concatenate(
            [
                ux.ravel() / self.hx,
                np.full(self.num_sites, -1 / self.hx),
                uy.ravel() / self.hy,
                np.full(self.num_sites, -1 / self.hy),
            ]
        )
        return sp.csr_array(
            (values, (rows, cols)), shape=(self.num_links, self.num_sites)
        )

    def scalar_gradient(self, values: np.ndarray) -> np.ndarray:
        """Return the ordinary periodic forward gradient."""
        values = self._coerce_site(values, "values")
        return np.stack(
            [
                (np.roll(values, -1, axis=1) - values) / self.hx,
                (np.roll(values, -1, axis=0) - values) / self.hy,
            ]
        )

    def divergence(self, tangent: np.ndarray) -> np.ndarray:
        """Return the ordinary periodic divergence of a tangent field."""
        tangent = self._coerce_tangent(tangent, "tangent field")
        tx, ty = tangent
        return (tx - np.roll(tx, 1, axis=1)) / self.hx + (
            ty - np.roll(ty, 1, axis=0)
        ) / self.hy

    def gradient(
        self, psi: np.ndarray, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the forward covariant gradient of a site field."""
        psi = self._coerce_site(psi, "psi")
        ux, uy = self.link_variables(vector_potential)
        return np.stack(
            [
                (ux * np.roll(psi, -1, axis=1) - psi) / self.hx,
                (uy * np.roll(psi, -1, axis=0) - psi) / self.hy,
            ]
        )

    def laplacian_x(
        self, psi: np.ndarray, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the x part of the covariant Laplacian."""
        psi = self._coerce_site(psi, "psi")
        ux, _ = self.link_variables(vector_potential)
        forward = ux * np.roll(psi, -1, axis=1)
        backward = np.conj(np.roll(ux, 1, axis=1)) * np.roll(psi, 1, axis=1)
        return (forward + backward - 2 * psi) / self.hx**2

    def laplacian_y(
        self, psi: np.ndarray, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the y part of the covariant Laplacian."""
        psi = self._coerce_site(psi, "psi")
        _, uy = self.link_variables(vector_potential)
        forward = uy * np.roll(psi, -1, axis=0)
        backward = np.conj(np.roll(uy, 1, axis=0)) * np.roll(psi, 1, axis=0)
        return (forward + backward - 2 * psi) / self.hy**2

    def laplacian(
        self, psi: np.ndarray, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the complete covariant Laplacian."""
        return self.laplacian_x(psi, vector_potential) + self.laplacian_y(
            psi, vector_potential
        )

    def supercurrent(
        self, psi: np.ndarray, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the gauge-invariant single-component current on links."""
        psi = self._coerce_site(psi, "psi")
        return np.imag(
            np.conj(psi)[np.newaxis, :, :] * self.gradient(psi, vector_potential)
        )

    def get_supercurrent(
        self, psi: np.ndarray, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compatibility alias for :meth:`supercurrent`."""
        return self.supercurrent(psi, vector_potential)

    @staticmethod
    def _finite_real_coefficient(name: str, value: float) -> float:
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real number.")
        return float(value)

    def isotropic_two_component_supercurrent(
        self,
        psi1: np.ndarray,
        psi2: np.ndarray,
        *,
        k1: float = 1.0,
        k2: float = 1.0,
        mixed_gradient: float = 0.0,
        vector_potential: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Return the current of a model-neutral isotropic two-component field.

        The corresponding link gradient-energy density is

        .. math::

            k_1|D\psi_1|^2+k_2|D\psi_2|^2
            +2k_{12}\operatorname{Re}[(D\psi_1)^*D\psi_2],

        where ``mixed_gradient`` is :math:`k_{12}`.  This form covers the
        diagonal gradient sector of ``DPlusDPrimeModel`` and the isotropic
        mixed-gradient sector of ``SPlusSModel``.  Model-specific normalization
        factors are intentionally left to the caller.
        """
        k1 = self._finite_real_coefficient("k1", k1)
        k2 = self._finite_real_coefficient("k2", k2)
        mixed_gradient = self._finite_real_coefficient("mixed_gradient", mixed_gradient)
        psi1 = self._coerce_site(psi1, "psi1")
        psi2 = self._coerce_site(psi2, "psi2")
        grad1 = self.gradient(psi1, vector_potential)
        grad2 = self.gradient(psi2, vector_potential)
        current = k1 * np.imag(np.conj(psi1)[None, :, :] * grad1)
        current += k2 * np.imag(np.conj(psi2)[None, :, :] * grad2)
        if mixed_gradient:
            mixed = np.imag(
                np.conj(psi1)[None, :, :] * grad2 + np.conj(psi2)[None, :, :] * grad1
            )
            current += mixed_gradient * mixed
        return current

    def get_isotropic_two_component_supercurrent(
        self,
        psi1: np.ndarray,
        psi2: np.ndarray,
        *,
        k1: float = 1.0,
        k2: float = 1.0,
        mixed_gradient: float = 0.0,
        vector_potential: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compatibility-style alias for the model-neutral current helper."""
        return self.isotropic_two_component_supercurrent(
            psi1,
            psi2,
            k1=k1,
            k2=k2,
            mixed_gradient=mixed_gradient,
            vector_potential=vector_potential,
        )

    def s_plus_d_supercurrent(
        self,
        psi_d: np.ndarray,
        psi_s: np.ndarray,
        *,
        eta_s: float,
        eta_v: float,
        vector_potential: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the complete unscaled d+s condensate current."""
        psi_d = self._coerce_site(psi_d, "psi_d")
        psi_s = self._coerce_site(psi_s, "psi_s")
        grad_d = self.gradient(psi_d, vector_potential)
        grad_s = self.gradient(psi_s, vector_potential)
        diagonal = np.imag(np.conj(psi_d)[None, :, :] * grad_d)
        diagonal += eta_s * np.imag(np.conj(psi_s)[None, :, :] * grad_s)
        if eta_v == 0:
            return diagonal
        mixed = np.imag(
            np.conj(psi_d)[None, :, :] * grad_s + np.conj(psi_s)[None, :, :] * grad_d
        )
        signs = np.array([-1.0, 1.0])[:, None, None]
        return diagonal + eta_v * signs * mixed

    def get_s_plus_d_supercurrent(
        self,
        psi_d: np.ndarray,
        psi_s: np.ndarray,
        *,
        eta_s: float,
        eta_v: float,
        vector_potential: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compatibility alias for :meth:`s_plus_d_supercurrent`."""
        return self.s_plus_d_supercurrent(
            psi_d,
            psi_s,
            eta_s=eta_s,
            eta_v=eta_v,
            vector_potential=vector_potential,
        )

    def magnetization_current(self, magnetization: np.ndarray) -> np.ndarray:
        r"""Return the periodic bound current generated by site magnetization.

        This is the exact current associated with the link derivative of

        .. math::

            -\langle M B\rangle,

        using the same convention as the condensate helpers: the derivative
        of the cell-averaged energy with respect to a link is
        ``-2 * current / num_sites``.  It is the contribution required by the
        orbital-Zeeman term of ``DPlusDPrimeModel`` when the periodic vector
        potential is dynamical.
        """
        raw = np.asarray(magnetization)
        if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0):
            raise ValueError("magnetization must be real.")
        magnetization = self._coerce_site(
            np.asarray(np.real(raw), dtype=float), "magnetization"
        )
        if not np.all(np.isfinite(magnetization)):
            raise ValueError("magnetization must contain only finite values.")
        packed = 0.5 * np.asarray(
            self.curl_matrix.T @ magnetization.ravel(), dtype=float
        )
        ax, ay = self.unpack_links(packed)
        return np.stack([ax, ay])

    def get_magnetization_current(self, magnetization: np.ndarray) -> np.ndarray:
        """Compatibility-style alias for :meth:`magnetization_current`."""
        return self.magnetization_current(magnetization)

    def curl(self, vector_potential: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the periodic dynamic induction ``curl(a)`` on cells."""
        a = self._resolve_vector_potential(vector_potential)
        packed = self.pack_links(a[0], a[1])
        return np.asarray(self.curl_matrix @ packed).reshape(self.shape)

    def magnetic_field(
        self, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return total induction ``B_n + curl(a)`` on cells."""
        return self.cell.background_field + self.curl(vector_potential)

    def induction(self, vector_potential: Optional[np.ndarray] = None) -> np.ndarray:
        """Alias for :meth:`magnetic_field`."""
        return self.magnetic_field(vector_potential)

    def get_magnetic_field(
        self, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compatibility alias for :meth:`magnetic_field`."""
        return self.magnetic_field(vector_potential)

    def get_triangle_magnetic_field(
        self, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compatibility alias; structured magnetic values live on cells."""
        return self.magnetic_field(vector_potential)

    def magnetic_curl(
        self, vector_potential: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""Return link components ``(partial_y B, -partial_x B)``."""
        field = self.magnetic_field(vector_potential).ravel()
        packed = np.asarray(self.magnetic_curl_gradient @ field)
        ax, ay = self.unpack_links(packed)
        return np.stack([ax, ay])

    def wilson_loops(self, vector_potential: Optional[np.ndarray] = None) -> np.ndarray:
        """Return counter-clockwise Wilson loops for all plaquettes."""
        ux, uy = self.link_variables(vector_potential)
        return (
            ux
            * np.roll(uy, -1, axis=1)
            * np.conj(np.roll(ux, -1, axis=0))
            * np.conj(uy)
        )

    def vorticity(
        self,
        psi: np.ndarray,
        vector_potential: Optional[np.ndarray] = None,
        *,
        zero_tolerance: float = 1e-14,
    ) -> np.ndarray:
        """Return integer gauge-invariant vortex charge on each plaquette."""
        psi = self._coerce_site(psi, "psi")
        if np.any(np.abs(psi) <= zero_tolerance):
            raise ValueError(
                "vorticity is undefined when the order parameter vanishes at a site."
            )
        ux, uy = self.link_variables(vector_potential)
        bond_x = np.angle(np.conj(psi) * ux * np.roll(psi, -1, axis=1))
        bond_y = np.angle(np.conj(psi) * uy * np.roll(psi, -1, axis=0))
        circulation = (
            bond_x + np.roll(bond_y, -1, axis=1) - np.roll(bond_x, -1, axis=0) - bond_y
        )
        raw_charge = (
            circulation + self.magnetic_field(vector_potential) * self.cell_area
        ) / (2 * np.pi)
        charge = np.rint(raw_charge)
        if not np.allclose(raw_charge, charge, atol=2e-10, rtol=0):
            raise RuntimeError("Gauge-invariant plaquette charge is not integral.")
        return charge.astype(int)

    def vortex_count(
        self, psi: np.ndarray, vector_potential: Optional[np.ndarray] = None
    ) -> int:
        """Return total signed vortex charge, equal to the flux sector."""
        return int(np.sum(self.vorticity(psi, vector_potential)))
