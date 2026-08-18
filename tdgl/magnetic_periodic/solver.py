"""Standalone structured-grid solver with magnetic-periodic boundary conditions."""

from __future__ import annotations

import inspect
import logging
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.sparse as sp

from ..device.models import (
    DPlusDPrimeModel,
    SingleBandModel,
    SPlusDModel,
    SPlusSModel,
)
from ..solver.options import SolverOptions, SparseSolver
from .operators import MagneticPeriodicOperators
from .solution import (
    HDF5_BACKEND,
    LATEST_HDF5_SCHEMA_VERSION,
    MagneticPeriodicSolution,
    component_names_for_model,
    options_to_json,
    write_frame_components,
)

logger = logging.getLogger("magnetic_periodic")


SUPPORTED_MODELS = (SingleBandModel, SPlusDModel, DPlusDPrimeModel, SPlusSModel)


def _internal_component_names(model) -> Tuple[str, str]:
    """Return model-specific names for the two stored order-parameter arrays."""
    if isinstance(model, SingleBandModel):
        return "psi", "unused"
    if isinstance(model, SPlusDModel):
        return "d", "s"
    if isinstance(model, DPlusDPrimeModel):
        return "d", "d_prime"
    if isinstance(model, SPlusSModel):
        return "psi1", "psi2"
    raise TypeError(f"Unsupported magnetic-periodic model: {type(model).__name__}.")


def _electromagnetic_relaxation(model) -> float:
    """Return the local-Maxwell relaxation coefficient for supported models."""
    if isinstance(model, (SPlusDModel, SPlusSModel)):
        return float(model.beta_em)
    return 1.0


def _site_grid(cell, value, *, dtype=None, name="site field") -> np.ndarray:
    """Return a copied ``(ny, nx)`` site field."""
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full(cell.shape, array, dtype=array.dtype)
    elif array.shape == (cell.num_sites,):
        array = array.reshape(cell.shape)
    elif array.shape != cell.shape:
        raise ValueError(
            f"{name} must be scalar, shape {cell.shape}, or shape "
            f"({cell.num_sites},); got {array.shape}."
        )
    return np.array(array, copy=True)


def _link_grid(cell, value, *, name="link field") -> np.ndarray:
    """Return a copied ``(2, ny, nx)`` link field."""
    array = np.asarray(value, dtype=float)
    expected = (2,) + cell.shape
    num_edges = int(getattr(cell, "num_edges", 2 * cell.num_sites))
    if array.shape == (num_edges,):
        array = array.reshape(expected)
    elif array.shape == (cell.num_sites, 2):
        array = array.T.reshape(expected)
    elif array.shape != expected:
        raise ValueError(
            f"{name} must have shape {expected}, ({num_edges},), or "
            f"({cell.num_sites}, 2); got {array.shape}."
        )
    return np.array(array, copy=True)


def _cell_mean_induction(cell) -> float:
    return float(getattr(cell, "mean_induction", cell.background_field))


def _cell_kappa(cell) -> float:
    return float(
        getattr(
            cell,
            "kappa",
            cell.layer.london_lambda / cell.layer.coherence_length,
        )
    )


def _s_plus_d_gradient_density(
    model: SPlusDModel,
    operators: MagneticPeriodicOperators,
    psi_d: np.ndarray,
    psi_s: np.ndarray,
) -> np.ndarray:
    """Return the link-discretized gradient-energy density on cells."""
    grad_d = np.asarray(operators.gradient(psi_d))
    grad_s = np.asarray(operators.gradient(psi_s))
    density = np.abs(grad_d[0]) ** 2 + np.abs(grad_d[1]) ** 2
    density += model.eta_s * (np.abs(grad_s[0]) ** 2 + np.abs(grad_s[1]) ** 2)
    density += (
        2
        * model.eta_v
        * np.real(np.conj(grad_s[1]) * grad_d[1] - np.conj(grad_s[0]) * grad_d[0])
    )
    return density


def _isotropic_two_component_gradient_density(
    operators: MagneticPeriodicOperators,
    psi1: np.ndarray,
    psi2: np.ndarray,
    *,
    stiffness2: float = 1.0,
    mixed_gradient: float = 0.0,
) -> np.ndarray:
    """Return an isotropic two-component link-gradient density on cells."""
    grad1 = np.asarray(operators.gradient(psi1))
    grad2 = np.asarray(operators.gradient(psi2))
    density = np.sum(np.abs(grad1) ** 2, axis=0)
    density += stiffness2 * np.sum(np.abs(grad2) ** 2, axis=0)
    if mixed_gradient:
        density += 2 * mixed_gradient * np.real(np.sum(np.conj(grad1) * grad2, axis=0))
    return density


def _magnetic_energy_density(
    cell,
    operators: MagneticPeriodicOperators,
    vector_potential: np.ndarray,
    *,
    applied_field: Optional[float],
) -> float:
    induction = np.asarray(operators.induction(vector_potential), dtype=float)
    reference = 0.0 if applied_field is None else float(applied_field)
    return float(_cell_kappa(cell) ** 2 * np.mean((induction - reference) ** 2))


def single_band_free_energy_density(
    cell,
    psi: np.ndarray,
    vector_potential: np.ndarray,
    epsilon: Union[float, np.ndarray] = 1.0,
    *,
    include_magnetic: bool = True,
    applied_field: Optional[float] = None,
) -> float:
    """Return the structured-grid single-band GL free-energy density."""
    if not isinstance(cell.layer.model, SingleBandModel):
        raise TypeError("single_band_free_energy_density requires SingleBandModel.")
    operators = MagneticPeriodicOperators(cell)
    order = _site_grid(cell, psi, dtype=np.complex128, name="psi")
    epsilon_grid = _site_grid(cell, epsilon, dtype=float, name="epsilon")
    links = _link_grid(cell, vector_potential, name="vector_potential")
    operators.set_vector_potential(links)
    density = float(
        np.mean(
            -epsilon_grid * np.abs(order) ** 2
            + 0.5 * np.abs(order) ** 4
            + np.sum(np.abs(operators.gradient(order)) ** 2, axis=0)
        )
    )
    if include_magnetic:
        density += _magnetic_energy_density(
            cell, operators, links, applied_field=applied_field
        )
    return density


def s_plus_d_free_energy_density(
    cell,
    psi_d: np.ndarray,
    psi_s: np.ndarray,
    vector_potential: np.ndarray,
    epsilon: Union[float, np.ndarray] = 1.0,
    *,
    include_magnetic: bool = True,
    applied_field: Optional[float] = None,
) -> float:
    r"""Return the structured-grid s+d free-energy density.

    With ``applied_field=None``, the magnetic contribution is the fixed-B
    Helmholtz term :math:`\kappa^2 B^2`.  Passing a uniform applied field H uses
    the Gibbs diagnostic :math:`\kappa^2(B-H)^2` instead.  A fixed-flux solve
    alone does not determine H; the Li workflow must obtain it from a virial or
    thermodynamic derivative when constructing a Gibbs curve.
    """
    model = cell.layer.model
    if not isinstance(model, SPlusDModel):
        raise TypeError("s_plus_d_free_energy_density requires SPlusDModel.")
    operators = MagneticPeriodicOperators(cell)
    d_order = _site_grid(cell, psi_d, dtype=np.complex128, name="psi_d")
    s_order = _site_grid(cell, psi_s, dtype=np.complex128, name="psi_s")
    epsilon_grid = _site_grid(cell, epsilon, dtype=float, name="epsilon")
    links = _link_grid(cell, vector_potential, name="vector_potential")
    operators.set_vector_potential(links)

    abs_d = np.abs(d_order) ** 2
    abs_s = np.abs(s_order) ** 2
    nu_effective = model.nu + model.nu_disorder_coupling * (epsilon_grid - 1)
    potential = (
        -epsilon_grid * abs_d
        - nu_effective * abs_s
        + 0.5 * abs_d**2
        + 0.5 * model.tau1 * abs_s**2
        + 0.5 * model.tau3 * abs_d * abs_s
        + model.tau4 * np.real(np.conj(s_order) ** 2 * d_order**2)
    )
    gradient = _s_plus_d_gradient_density(model, operators, d_order, s_order)
    density = float(np.mean(potential + gradient))
    if include_magnetic:
        density += _magnetic_energy_density(
            cell, operators, links, applied_field=applied_field
        )
    return density


def d_plus_d_prime_free_energy_density(
    cell,
    d_order: np.ndarray,
    d_prime_order: np.ndarray,
    vector_potential: np.ndarray,
    epsilon: Union[float, np.ndarray] = 1.0,
    *,
    include_magnetic: bool = True,
    applied_field: Optional[float] = None,
) -> float:
    r"""Return the structured-grid :math:`d+d'` free-energy density."""
    model = cell.layer.model
    if not isinstance(model, DPlusDPrimeModel):
        raise TypeError("d_plus_d_prime_free_energy_density requires DPlusDPrimeModel.")
    operators = MagneticPeriodicOperators(cell)
    d_order = _site_grid(cell, d_order, dtype=np.complex128, name="d")
    d_prime_order = _site_grid(cell, d_prime_order, dtype=np.complex128, name="d_prime")
    epsilon_grid = _site_grid(cell, epsilon, dtype=float, name="epsilon")
    links = _link_grid(cell, vector_potential, name="vector_potential")
    operators.set_vector_potential(links)
    induction = np.asarray(operators.induction(links), dtype=float)

    abs_d = np.abs(d_order) ** 2
    abs_d_prime = np.abs(d_prime_order) ** 2
    phase_sum = np.conj(d_order) * d_prime_order + d_order * np.conj(d_prime_order)
    chirality = np.conj(d_order) * d_prime_order - d_order * np.conj(d_prime_order)
    potential = (
        -epsilon_grid * abs_d
        - model.alpha * abs_d_prime
        + 0.5 * (abs_d**2 + abs_d_prime**2)
        + (1 / 3) * abs_d * abs_d_prime
        + (1 / 6) * np.real(phase_sum**2)
        + np.real(-1j * model.zeeman_coupling * induction * chirality)
    )
    gradient = _isotropic_two_component_gradient_density(
        operators, d_order, d_prime_order
    )
    density = float(np.mean(potential + gradient))
    if include_magnetic:
        density += _magnetic_energy_density(
            cell, operators, links, applied_field=applied_field
        )
    return density


def s_plus_s_free_energy_density(
    cell,
    psi1: np.ndarray,
    psi2: np.ndarray,
    vector_potential: np.ndarray,
    epsilon: Union[float, np.ndarray] = 1.0,
    *,
    include_magnetic: bool = True,
    applied_field: Optional[float] = None,
) -> float:
    """Return the structured-grid isotropic two-band free-energy density."""
    model = cell.layer.model
    if not isinstance(model, SPlusSModel):
        raise TypeError("s_plus_s_free_energy_density requires SPlusSModel.")
    operators = MagneticPeriodicOperators(cell)
    psi1 = _site_grid(cell, psi1, dtype=np.complex128, name="psi1")
    psi2 = _site_grid(cell, psi2, dtype=np.complex128, name="psi2")
    epsilon_grid = _site_grid(cell, epsilon, dtype=float, name="epsilon")
    links = _link_grid(cell, vector_potential, name="vector_potential")
    operators.set_vector_potential(links)

    rho1 = np.abs(psi1) ** 2
    rho2 = np.abs(psi2) ** 2
    a1_effective = model.a1 + model.disorder_coupling1 * (1 - epsilon_grid)
    a2_effective = model.a2 + model.disorder_coupling2 * (1 - epsilon_grid)
    interband = np.conj(psi1) * psi2
    potential = (
        a1_effective * rho1
        + 0.5 * model.b1 * rho1**2
        + a2_effective * rho2
        + 0.5 * model.b2 * rho2**2
        + 0.5 * model.density_gamma3 * rho1 * rho2
        + model.phase_gamma2 * np.real(interband**2)
        - 2 * model.josephson_gamma * np.real(interband)
    )
    gradient = _isotropic_two_component_gradient_density(
        operators,
        psi1,
        psi2,
        stiffness2=model.k2_over_k1,
        mixed_gradient=model.mixed_gradient_k12,
    )
    density = float(np.mean(potential + gradient))
    if include_magnetic:
        density += _magnetic_energy_density(
            cell, operators, links, applied_field=applied_field
        )
    return density


def magnetic_periodic_free_energy_density(
    cell,
    order_parameter_1: np.ndarray,
    order_parameter_2: np.ndarray,
    vector_potential: np.ndarray,
    epsilon: Union[float, np.ndarray] = 1.0,
    *,
    include_magnetic: bool = True,
    applied_field: Optional[float] = None,
) -> float:
    """Dispatch the discrete free energy for any supported periodic model."""
    common = dict(
        include_magnetic=include_magnetic,
        applied_field=applied_field,
    )
    model = cell.layer.model
    if isinstance(model, SingleBandModel):
        return single_band_free_energy_density(
            cell,
            order_parameter_1,
            vector_potential,
            epsilon,
            **common,
        )
    if isinstance(model, SPlusDModel):
        return s_plus_d_free_energy_density(
            cell,
            order_parameter_1,
            order_parameter_2,
            vector_potential,
            epsilon,
            **common,
        )
    if isinstance(model, DPlusDPrimeModel):
        return d_plus_d_prime_free_energy_density(
            cell,
            order_parameter_1,
            order_parameter_2,
            vector_potential,
            epsilon,
            **common,
        )
    if isinstance(model, SPlusSModel):
        return s_plus_s_free_energy_density(
            cell,
            order_parameter_1,
            order_parameter_2,
            vector_potential,
            epsilon,
            **common,
        )
    raise TypeError(f"Unsupported magnetic-periodic model: {type(model).__name__}.")


def s_plus_d_virial_applied_field(
    cell,
    psi_d: np.ndarray,
    psi_s: np.ndarray,
    vector_potential: np.ndarray,
    epsilon: Union[float, np.ndarray] = 1.0,
) -> float:
    r"""Infer the uniform applied field from the periodic virial identity.

    For a stationary, homogeneous-coefficient two-dimensional torus,

    .. math::

        2\kappa^2 H\bar B = \langle f_{\rm grad}\rangle
            + 2\kappa^2\langle B^2\rangle.

    The identity requires spatially homogeneous GL coefficients and is
    undefined in the zero-flux sector. This helper evaluates the discrete
    right-hand side; callers remain responsible for using a sufficiently
    converged stationary state.
    """
    model = cell.layer.model
    if not isinstance(model, SPlusDModel):
        raise TypeError("s_plus_d_virial_applied_field requires SPlusDModel.")
    operators = MagneticPeriodicOperators(cell)
    d_order = _site_grid(cell, psi_d, dtype=np.complex128, name="psi_d")
    s_order = _site_grid(cell, psi_s, dtype=np.complex128, name="psi_s")
    links = _link_grid(cell, vector_potential, name="vector_potential")
    epsilon_grid = _site_grid(cell, epsilon, dtype=float, name="epsilon")
    if not np.allclose(epsilon_grid, epsilon_grid.flat[0], rtol=0, atol=1e-13):
        raise ValueError(
            "The periodic virial applied field requires spatially "
            "homogeneous coefficients."
        )
    operators.set_vector_potential(links)
    induction = np.asarray(operators.induction(links), dtype=float)
    mean_induction = float(np.mean(induction))
    if math.isclose(mean_induction, 0.0, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("The virial applied field is undefined at zero mean flux.")
    gradient = _s_plus_d_gradient_density(model, operators, d_order, s_order)
    kappa_squared = _cell_kappa(cell) ** 2
    numerator = float(np.mean(gradient)) + 2 * kappa_squared * float(
        np.mean(induction**2)
    )
    return numerator / (2 * kappa_squared * mean_induction)


def magnetic_periodic_virial_applied_field(
    cell,
    order_parameter_1: np.ndarray,
    order_parameter_2: np.ndarray,
    vector_potential: np.ndarray,
    epsilon: Union[float, np.ndarray] = 1.0,
) -> float:
    """Infer uniform applied field for a homogeneous stationary periodic model.

    The current virial expression does not include the extra scaling term from
    a nonzero d+d' orbital-Zeeman coupling, so that combination is rejected.
    """
    model = cell.layer.model
    operators = MagneticPeriodicOperators(cell)
    first = _site_grid(
        cell, order_parameter_1, dtype=np.complex128, name="order_parameter_1"
    )
    second = _site_grid(
        cell, order_parameter_2, dtype=np.complex128, name="order_parameter_2"
    )
    links = _link_grid(cell, vector_potential, name="vector_potential")
    epsilon_grid = _site_grid(cell, epsilon, dtype=float, name="epsilon")
    if not np.allclose(epsilon_grid, epsilon_grid.flat[0], rtol=0, atol=1e-13):
        raise ValueError(
            "The periodic virial applied field requires spatially "
            "homogeneous coefficients."
        )
    operators.set_vector_potential(links)
    induction = np.asarray(operators.induction(links), dtype=float)
    mean_induction = float(np.mean(induction))
    if math.isclose(mean_induction, 0.0, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("The virial applied field is undefined at zero mean flux.")

    if isinstance(model, SingleBandModel):
        gradient = np.sum(np.abs(operators.gradient(first)) ** 2, axis=0)
    elif isinstance(model, SPlusDModel):
        gradient = _s_plus_d_gradient_density(model, operators, first, second)
    elif isinstance(model, DPlusDPrimeModel):
        if model.zeeman_coupling:
            raise ValueError(
                "The periodic virial applied field is not implemented for a "
                "nonzero d+d' orbital-Zeeman coupling."
            )
        gradient = _isotropic_two_component_gradient_density(operators, first, second)
    elif isinstance(model, SPlusSModel):
        gradient = _isotropic_two_component_gradient_density(
            operators,
            first,
            second,
            stiffness2=model.k2_over_k1,
            mixed_gradient=model.mixed_gradient_k12,
        )
    else:
        raise TypeError(f"Unsupported magnetic-periodic model: {type(model).__name__}.")
    kappa_squared = _cell_kappa(cell) ** 2
    numerator = float(np.mean(gradient)) + 2 * kappa_squared * float(
        np.mean(induction**2)
    )
    return numerator / (2 * kappa_squared * mean_induction)


class MagneticPeriodicSolver:
    """TDGL dynamics on a fixed-flux magnetic-periodic rectangular cell.

    This backend is intentionally independent of ``Device``, ``Mesh``,
    ``MeshOperators``, and the open-boundary ``TDGLSolver``. It supports every
    model accepted by :class:`tdgl.Layer`. ``SPlusDModel`` and ``SPlusSModel``
    may either evolve the periodic electromagnetic correction or hold it fixed;
    the single-band and d+d' models currently use the paper-style prescribed
    background field.

    Args:
        cell: A :class:`MagneticPeriodicCell` defining the rectangular grid and
            integer flux sector.
        options: Standard time-integration and output options.
        disorder_epsilon: A scalar or static callable evaluated on the cell's
            dimensionless sites.
        seed_solution: A magnetic-periodic solution on the identical cell.
        initial_psi_d: Backwards-compatible name for the model's first stored
            order-parameter component.
        initial_psi_s: Backwards-compatible name for the model's second stored
            order-parameter component.
        initial_psi1: First component in the model's public component order:
            ``psi``, ``s``, ``d``, or ``s1``.
        initial_psi2: Second component in the model's public component order:
            ``d``, ``d_prime``, or ``s2``.
        initial_vector_potential: Optional periodic vector-potential correction.
            The fixed-flux background is carried by the operators and is not
            included in this array.
    """

    def __init__(
        self,
        cell,
        options: SolverOptions,
        *,
        disorder_epsilon: Union[float, Callable] = 1.0,
        seed_solution: Optional[MagneticPeriodicSolution] = None,
        initial_psi_d: Optional[np.ndarray] = None,
        initial_psi_s: Optional[np.ndarray] = None,
        initial_psi1: Optional[np.ndarray] = None,
        initial_psi2: Optional[np.ndarray] = None,
        initial_vector_potential: Optional[np.ndarray] = None,
    ):
        self.cell = cell
        self.options = options
        self._validate_runtime_options(options)
        self.options.validate()
        self.model = cell.layer.model
        self._validate_backend_options()
        self.operators = MagneticPeriodicOperators(cell)
        self.component_names = component_names_for_model(self.model)
        internal_names = _internal_component_names(self.model)

        if isinstance(self.model, SingleBandModel) and initial_psi_s is not None:
            raise ValueError(
                "SingleBandModel has no initial_psi_s or second component."
            )

        if initial_psi1 is not None:
            if isinstance(self.model, SPlusDModel):
                if initial_psi_s is not None:
                    raise ValueError("Use initial_psi1 or initial_psi_s, not both.")
                initial_psi_s = initial_psi1
            else:
                if initial_psi_d is not None:
                    raise ValueError("Use initial_psi1 or initial_psi_d, not both.")
                initial_psi_d = initial_psi1
        if initial_psi2 is not None:
            if isinstance(self.model, SingleBandModel):
                raise ValueError("SingleBandModel has no initial_psi2 component.")
            if isinstance(self.model, SPlusDModel):
                if initial_psi_d is not None:
                    raise ValueError("Use initial_psi2 or initial_psi_d, not both.")
                initial_psi_d = initial_psi2
            else:
                if initial_psi_s is not None:
                    raise ValueError("Use initial_psi2 or initial_psi_s, not both.")
                initial_psi_s = initial_psi2

        self.epsilon = self._evaluate_disorder(disorder_epsilon)
        self.disorder_epsilon = disorder_epsilon
        self.seed_solution = seed_solution

        supplied_initial = any(
            value is not None
            for value in (initial_psi_d, initial_psi_s, initial_vector_potential)
        )
        if seed_solution is not None and supplied_initial:
            raise ValueError(
                "seed_solution cannot be combined with explicit initial fields."
            )

        if seed_solution is not None:
            if not isinstance(seed_solution, MagneticPeriodicSolution):
                raise TypeError(
                    "seed_solution must be a MagneticPeriodicSolution from the "
                    "magnetic-periodic backend."
                )
            if seed_solution.cell != cell:
                raise ValueError(
                    "seed_solution.cell must exactly match the magnetic-periodic "
                    "cell, including grid, flux sector, layer, and gauge."
                )
            frame = seed_solution.final_frame
            if isinstance(self.model, SingleBandModel):
                first, second = frame.psi, np.zeros(cell.shape, dtype=complex)
            elif isinstance(self.model, SPlusDModel):
                first, second = frame.psi_d, frame.psi_s
            elif isinstance(self.model, DPlusDPrimeModel):
                first, second = frame.psi_d, frame.psi_d_prime
            else:
                first, second = frame.psi_s1, frame.psi_s2
            self.psi_d = _site_grid(
                cell, first, dtype=np.complex128, name="seed order_parameter_1"
            )
            self.psi_s = _site_grid(
                cell, second, dtype=np.complex128, name="seed order_parameter_2"
            )
            self.vector_potential = _link_grid(
                cell, frame.vector_potential, name="seed vector potential"
            )
        else:
            default_first, default_second = self._default_initial_fields()
            if initial_psi_d is None:
                initial_psi_d = default_first
            if initial_psi_s is None:
                initial_psi_s = default_second
            if initial_vector_potential is None:
                initial_vector_potential = np.zeros((2,) + cell.shape, dtype=float)
            self.psi_d = _site_grid(
                cell,
                initial_psi_d,
                dtype=np.complex128,
                name=f"initial {internal_names[0]}",
            )
            self.psi_s = _site_grid(
                cell,
                initial_psi_s,
                dtype=np.complex128,
                name=f"initial {internal_names[1]}",
            )
            self.vector_potential = _link_grid(
                cell, initial_vector_potential, name="initial_vector_potential"
            )

        self.drive = np.empty((2,) + cell.shape, dtype=float)
        self.drive[0].fill(float(options.s_plus_d_drive_current_x))
        self.drive[1].fill(float(options.s_plus_d_drive_current_y))

        self._magnetic_lu_dt = None
        self._magnetic_lu = None
        self._change_history = deque(maxlen=options.adaptive_window)
        self._tentative_dt = float(options.dt_init)
        self._validate_state()

    def _default_initial_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return deterministic model-aware initial fields."""
        first = np.ones(self.cell.shape, dtype=np.complex128)
        second = np.zeros(self.cell.shape, dtype=np.complex128)
        if isinstance(self.model, DPlusDPrimeModel):
            second.fill(-1e-4j)
        elif isinstance(self.model, SPlusSModel):
            # Reuse the same complete homogeneous-potential minimizer as the
            # unstructured solver, imported lazily to avoid an import cycle.
            from ..solver.solver import _s_plus_s_uniform_state

            value1, value2 = _s_plus_s_uniform_state(self.model)
            first.fill(value1)
            second.fill(value2)
        return first, second

    @staticmethod
    def _validate_runtime_options(options: SolverOptions) -> None:
        """Reject runtime selections that this standalone backend cannot use."""
        if options.gpu:
            raise ValueError("MagneticPeriodicSolver is currently CPU-only.")
        sparse_solver = options.sparse_solver
        if isinstance(sparse_solver, str):
            sparse_solver = sparse_solver.lower()
            supported = sparse_solver == SparseSolver.SUPERLU.value
        else:
            supported = sparse_solver is SparseSolver.SUPERLU
        if not supported:
            raise ValueError(
                "MagneticPeriodicSolver currently supports sparse_solver="
                "'superlu' only."
            )

    def _validate_backend_options(self) -> None:
        options = self.options
        self._validate_runtime_options(options)
        if not isinstance(self.model, SUPPORTED_MODELS):
            raise TypeError(
                "MagneticPeriodicSolver does not support model "
                f"{type(self.model).__name__}."
            )
        self.model.validate()
        if options.include_screening and isinstance(
            self.model, (SingleBandModel, DPlusDPrimeModel)
        ):
            raise ValueError(
                f"{type(self.model).__name__} currently supports a prescribed "
                "magnetic-periodic background only; set include_screening=False."
            )
        if (
            options.include_screening
            and isinstance(self.model, SPlusSModel)
            and not math.isclose(
                self.model.em_coupling, 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "SPlusSModel magnetic-periodic screening requires em_coupling=1 "
                "so the condensate and field equations vary one free energy."
            )
        drive_requested = bool(
            options.s_plus_d_drive_current_x or options.s_plus_d_drive_current_y
        )
        if drive_requested and (
            not options.include_screening
            or not isinstance(self.model, (SPlusDModel, SPlusSModel))
        ):
            raise ValueError(
                "A homogeneous periodic drive requires include_screening=True "
                "with SPlusDModel or SPlusSModel."
            )
        if options.terminal_psi is not None:
            raise ValueError(
                "A magnetic-periodic cell has no terminals or physical boundary; "
                "set terminal_psi=None."
            )
        if options.monitor:
            raise ValueError(
                "The legacy live monitor requires an unstructured mesh and is not "
                "available for MagneticPeriodicSolver."
            )

    def _evaluate_disorder(self, disorder) -> np.ndarray:
        if not callable(disorder):
            epsilon = _site_grid(
                self.cell, disorder, dtype=float, name="disorder_epsilon"
            )
        else:
            signature = inspect.signature(disorder)
            if "t" in signature.parameters:
                raise ValueError(
                    "MagneticPeriodicSolver currently supports static "
                    "disorder_epsilon only."
                )
            sites = np.asarray(self.cell.sites)
            try:
                values = disorder(sites, vectorized=True)
            except TypeError:
                try:
                    values = disorder(sites)
                except (TypeError, ValueError):
                    values = np.asarray([disorder(site) for site in sites])
            epsilon = _site_grid(
                self.cell, values, dtype=float, name="disorder_epsilon"
            )
        if not np.all(np.isfinite(epsilon)):
            raise ValueError("disorder_epsilon must be finite.")
        if np.any(epsilon > 1):
            raise ValueError("disorder_epsilon must be <= 1.")
        return epsilon

    def _magnetic_field(self, vector_potential: np.ndarray) -> np.ndarray:
        method = getattr(self.operators, "induction", None)
        if method is None:
            method = self.operators.magnetic_field
        return np.asarray(method(vector_potential), dtype=float)

    def _validate_flux(self, vector_potential: np.ndarray) -> None:
        field = self._magnetic_field(vector_potential)
        if field.shape != self.cell.shape or not np.all(np.isfinite(field)):
            raise RuntimeError("Magnetic-periodic induction is invalid.")
        expected = _cell_mean_induction(self.cell)
        if not math.isclose(
            float(np.mean(field)), expected, rel_tol=2e-12, abs_tol=2e-12
        ):
            raise RuntimeError(
                "The periodic vector-potential update changed the fixed flux: "
                f"mean B={np.mean(field):.16g}, expected {expected:.16g}."
            )

    def _validate_state(self) -> None:
        if not (
            np.all(np.isfinite(self.psi_d))
            and np.all(np.isfinite(self.psi_s))
            and np.all(np.isfinite(self.vector_potential))
        ):
            raise ValueError("Initial magnetic-periodic fields must be finite.")
        self.operators.set_vector_potential(self.vector_potential)
        self._validate_flux(self.vector_potential)

    def compute_free_energy_density(
        self,
        psi_d: Optional[np.ndarray] = None,
        psi_s: Optional[np.ndarray] = None,
        vector_potential: Optional[np.ndarray] = None,
        *,
        include_magnetic: bool = True,
        applied_field: Optional[float] = None,
    ) -> float:
        """Evaluate the backend's discrete free-energy density."""
        return magnetic_periodic_free_energy_density(
            self.cell,
            self.psi_d if psi_d is None else psi_d,
            self.psi_s if psi_s is None else psi_s,
            self.vector_potential if vector_potential is None else vector_potential,
            self.epsilon,
            include_magnetic=include_magnetic,
            applied_field=applied_field,
        )

    def _order_parameter_trial(
        self,
        psi_d: np.ndarray,
        psi_s: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        model = self.model
        operators = self.operators
        abs_d = np.abs(psi_d) ** 2

        if isinstance(model, SingleBandModel):
            lap_d = operators.laplacian(psi_d)
            gamma = model.gamma
            z = 0.5 * gamma**2 * psi_d
            with np.errstate(all="ignore"):
                w = (
                    z * abs_d
                    + psi_d
                    + (
                        (dt / self.cell.layer.u)
                        * np.sqrt(1 + gamma**2 * abs_d)
                        * ((self.epsilon - abs_d) * psi_d + lap_d)
                    )
                )
                c = np.real(w * np.conj(z))
                two_c_1 = 2 * c + 1
                w2 = np.abs(w) ** 2
                discriminant = two_c_1**2 - 4 * np.abs(z) ** 2 * w2
                if np.any(discriminant < 0) or not np.all(np.isfinite(discriminant)):
                    new_d = np.full_like(psi_d, np.nan)
                else:
                    new_density = (2 * w2) / (two_c_1 + np.sqrt(discriminant))
                    new_d = w - z * new_density
            new_s = psi_s.copy()
        elif isinstance(model, SPlusDModel):
            lap_x_d = operators.laplacian_x(psi_d)
            lap_y_d = operators.laplacian_y(psi_d)
            lap_x_s = operators.laplacian_x(psi_s)
            lap_y_s = operators.laplacian_y(psi_s)
            abs_s = np.abs(psi_s) ** 2
            nu_effective = model.nu + model.nu_disorder_coupling * (self.epsilon - 1)
            rhs_d = (
                lap_x_d
                + lap_y_d
                + self.epsilon * psi_d
                - abs_d * psi_d
                - 0.5 * model.tau3 * abs_s * psi_d
                - model.tau4 * psi_s**2 * np.conj(psi_d)
                + model.eta_v * (lap_y_s - lap_x_s)
            )
            rhs_s = (
                model.eta_s * (lap_x_s + lap_y_s)
                + nu_effective * psi_s
                - model.tau1 * abs_s * psi_s
                - 0.5 * model.tau3 * abs_d * psi_s
                - model.tau4 * psi_d**2 * np.conj(psi_s)
                + model.eta_v * (lap_y_d - lap_x_d)
            )
            new_d = psi_d + dt * rhs_d
            new_s = psi_s + dt * rhs_s / (model.eta_s * model.relaxation_s)
        elif isinstance(model, DPlusDPrimeModel):
            lap_d = operators.laplacian(psi_d)
            lap_s = operators.laplacian(psi_s)
            abs_s = np.abs(psi_s) ** 2
            induction = self._magnetic_field(self.vector_potential)
            zeeman = model.zeeman_coupling * induction
            rhs_d = (
                lap_d
                + self.epsilon * psi_d
                - abs_d * psi_d
                - (2 / 3) * abs_s * psi_d
                - (1 / 3) * psi_s**2 * np.conj(psi_d)
                + 1j * zeeman * psi_s
            )
            rhs_s = (
                lap_s
                + model.alpha * psi_s
                - abs_s * psi_s
                - (2 / 3) * abs_d * psi_s
                - (1 / 3) * psi_d**2 * np.conj(psi_s)
                - 1j * zeeman * psi_d
            )
            new_d = psi_d + (dt / model.relaxation_d) * rhs_d
            new_s = psi_s + (dt / model.relaxation_d_prime) * rhs_s
        elif isinstance(model, SPlusSModel):
            lap_d = operators.laplacian(psi_d)
            lap_s = operators.laplacian(psi_s)
            abs_s = np.abs(psi_s) ** 2
            a1_effective = model.a1 + model.disorder_coupling1 * (1 - self.epsilon)
            a2_effective = model.a2 + model.disorder_coupling2 * (1 - self.epsilon)
            rhs_d = (
                lap_d
                + model.mixed_gradient_k12 * lap_s
                - a1_effective * psi_d
                - model.b1 * abs_d * psi_d
                - 0.5 * model.density_gamma3 * abs_s * psi_d
                - model.phase_gamma2 * np.conj(psi_d) * psi_s**2
                + model.josephson_gamma * psi_s
            )
            rhs_s = (
                model.k2_over_k1 * lap_s
                + model.mixed_gradient_k12 * lap_d
                - a2_effective * psi_s
                - model.b2 * abs_s * psi_s
                - 0.5 * model.density_gamma3 * abs_d * psi_s
                - model.phase_gamma2 * np.conj(psi_s) * psi_d**2
                + model.josephson_gamma * psi_d
            )
            new_d = psi_d + (dt / model.relaxation1) * rhs_d
            new_s = psi_s + (dt / model.relaxation2) * rhs_s
        else:  # Guarded by _validate_backend_options.
            raise TypeError(f"Unsupported model: {type(model).__name__}.")

        change = max(
            float(np.max(np.abs(new_d - psi_d))),
            float(np.max(np.abs(new_s - psi_s))),
        )
        return new_d, new_s, change

    def _raw_supercurrent(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> np.ndarray:
        """Return the variational condensate current before clock scaling."""
        model = self.model
        if isinstance(model, SingleBandModel):
            return np.asarray(self.operators.supercurrent(first), dtype=float)
        if isinstance(model, SPlusDModel):
            return np.asarray(
                self.operators.s_plus_d_supercurrent(
                    first,
                    second,
                    eta_s=model.eta_s,
                    eta_v=model.eta_v,
                ),
                dtype=float,
            )

        if isinstance(model, SPlusSModel):
            method = getattr(
                self.operators, "isotropic_two_component_supercurrent", None
            )
            if method is not None:
                return np.asarray(
                    method(
                        first,
                        second,
                        k1=1.0,
                        k2=model.k2_over_k1,
                        mixed_gradient=model.mixed_gradient_k12,
                    ),
                    dtype=float,
                )
            grad1 = self.operators.gradient(first)
            grad2 = self.operators.gradient(second)
            current = np.imag(np.conj(first)[None] * grad1)
            current += model.k2_over_k1 * np.imag(np.conj(second)[None] * grad2)
            current += model.mixed_gradient_k12 * np.imag(
                np.conj(first)[None] * grad2 + np.conj(second)[None] * grad1
            )
            return np.asarray(current, dtype=float)

        if isinstance(model, DPlusDPrimeModel):
            current = np.asarray(
                self.operators.isotropic_two_component_supercurrent(first, second),
                dtype=float,
            )
            if model.zeeman_coupling:
                chirality = np.conj(first) * second - first * np.conj(second)
                magnetization = np.real(1j * model.zeeman_coupling * chirality)
                method = getattr(self.operators, "magnetization_current", None)
                if method is not None:
                    current += np.asarray(method(magnetization), dtype=float)
                else:
                    packed = 0.5 * np.asarray(
                        self.operators.magnetic_curl_gradient
                        @ np.asarray(magnetization).ravel()
                    )
                    current += packed.reshape((2,) + self.cell.shape)
            return current
        raise TypeError(f"Unsupported model: {type(model).__name__}.")

    def _normalized_supercurrent(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> np.ndarray:
        """Return current in the model's stored transport normalization."""
        raw = self._raw_supercurrent(first, second)
        model = self.model
        if isinstance(model, SPlusDModel):
            raw = raw / model.beta_em
        elif isinstance(model, SPlusSModel):
            raw = raw / (model.em_coupling * model.beta_em)
        elif isinstance(model, DPlusDPrimeModel):
            raw = raw / model.em_coupling
        return _link_grid(self.cell, raw, name="supercurrent")

    def _factorized_magnetic_step(self, dt: float):
        dt = float(dt)
        if self._magnetic_lu is None or self._magnetic_lu_dt != dt:
            rate = (
                dt
                * _cell_kappa(self.cell) ** 2
                / _electromagnetic_relaxation(self.model)
            )
            diffusion = sp.csc_matrix(self.operators.magnetic_diffusion)
            num_edges = int(getattr(self.cell, "num_edges", 2 * self.cell.num_sites))
            implicit = sp.eye(num_edges, format="csc") + rate * diffusion
            # Use SuperLU explicitly so the configured backend is never
            # silently ignored or changed by SciPy's optional UMFPACK hook.
            self._magnetic_lu = sp.linalg.splu(implicit).solve
            self._magnetic_lu_dt = dt
        return self._magnetic_lu

    def _advance_vector_potential(
        self,
        psi_d: np.ndarray,
        psi_s: np.ndarray,
        vector_potential: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_current = self._raw_supercurrent(psi_d, psi_s)
        raw_current = _link_grid(self.cell, raw_current, name="supercurrent")
        relaxation = _electromagnetic_relaxation(self.model)
        rhs = (
            vector_potential.ravel()
            + (dt / relaxation) * (raw_current - self.drive).ravel()
        )
        new_flat = self._factorized_magnetic_step(dt)(rhs)
        new_vector_potential = new_flat.reshape((2,) + self.cell.shape)
        dA_dt = (new_vector_potential - vector_potential) / dt
        if not np.all(np.isfinite(new_vector_potential)):
            raise RuntimeError(
                "Magnetic-periodic vector-potential solve is non-finite."
            )
        self._validate_flux(new_vector_potential)
        return new_vector_potential, dA_dt

    def _advance(
        self,
        psi_d: np.ndarray,
        psi_s: np.ndarray,
        vector_potential: np.ndarray,
        dt: float,
    ):
        options = self.options
        trial_dt = float(dt)
        if options.include_screening:
            self.operators.set_vector_potential(vector_potential)
        for retry in range(options.max_solve_retries + 1):
            new_d, new_s, change = self._order_parameter_trial(psi_d, psi_s, trial_dt)
            acceptable = (
                np.all(np.isfinite(new_d))
                and np.all(np.isfinite(new_s))
                and change <= 0.1
            )
            if acceptable:
                break
            if not options.adaptive or retry == options.max_solve_retries:
                raise RuntimeError(
                    "Magnetic-periodic order-parameter step failed after "
                    f"{retry} retries at dt={trial_dt:.3g}; reduce dt_init/dt_max."
                )
            trial_dt *= options.adaptive_time_step_multiplier

        if options.include_screening:
            new_vector_potential, dA_dt = self._advance_vector_potential(
                new_d, new_s, vector_potential, trial_dt
            )
            self.operators.set_vector_potential(new_vector_potential)
            supercurrent = self._normalized_supercurrent(new_d, new_s)
            normal_current = -dA_dt
        else:
            # The prescribed correction is immutable throughout a fixed-field
            # solve, so retain the same array and cached magnetic links.
            new_vector_potential = vector_potential
            # Fixed-background simulations do not use the current to advance
            # the state.  Defer this comparatively expensive diagnostic until
            # a frame is actually written, which matters when save_every is
            # large (as it is for long phase scans).
            supercurrent = None
            normal_current = None
        if options.adaptive:
            density_change = max(
                float(np.max(np.abs(np.abs(new_d) ** 2 - np.abs(psi_d) ** 2))),
                float(np.max(np.abs(np.abs(new_s) ** 2 - np.abs(psi_s) ** 2))),
            )
        else:
            density_change = 0.0
        return (
            new_d,
            new_s,
            new_vector_potential,
            supercurrent,
            normal_current,
            trial_dt,
            density_change,
        )

    def _update_adaptive_step(self, used_dt: float, density_change: float) -> None:
        if not self.options.adaptive:
            self._tentative_dt = self.options.dt_init
            return
        self._change_history.append(density_change)
        window = self.options.adaptive_window
        if len(self._change_history) >= window:
            average = max(1e-10, float(np.mean(self._change_history)))
            proposed = self.options.dt_init / average
            self._tentative_dt = float(
                np.clip(0.5 * (proposed + used_dt), 0, self.options.dt_max)
            )
        else:
            self._tentative_dt = min(
                self.options.dt_max, max(used_dt, self._tentative_dt)
            )

    def _initial_currents(self):
        self.operators.set_vector_potential(self.vector_potential)
        supercurrent = self._normalized_supercurrent(self.psi_d, self.psi_s)
        return supercurrent, np.zeros_like(supercurrent)

    @staticmethod
    def _unique_output_path(output_file: Optional[str]) -> str:
        if output_file is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="tdgl-magnetic-periodic-", suffix=".h5", delete=False
            )
            path = handle.name
            handle.close()
            return path

        requested = Path(output_file).expanduser().resolve()
        requested.parent.mkdir(parents=True, exist_ok=True)
        if not requested.exists():
            return str(requested)
        suffix = requested.suffix or ".h5"
        stem = (
            requested.name[: -len(requested.suffix)]
            if requested.suffix
            else requested.name
        )
        for serial in range(1, 1_000_000):
            candidate = requested.with_name(f"{stem}-{serial}{suffix}")
            if not candidate.exists():
                logger.warning("Output exists; writing %s instead.", candidate)
                return str(candidate)
        raise IOError(f"Unable to choose a unique output path near {requested}.")

    def _write_header(self, h5file: h5py.File) -> None:
        h5file.attrs["backend"] = HDF5_BACKEND
        h5file.attrs["schema_version"] = LATEST_HDF5_SCHEMA_VERSION
        h5file.attrs["model_type"] = type(self.model).__name__
        h5file.attrs["field_control"] = (
            "fixed_background" if not self.options.include_screening else "fixed_flux"
        )
        # A checkpoint is loadable only after the measurement loop and final
        # metadata flush have both succeeded.  This prevents an interrupted
        # incremental write from being mistaken for a completed trajectory.
        h5file.attrs["complete"] = False
        h5file.attrs["total_seconds"] = 0.0
        self.cell.to_hdf5(h5file.create_group("cell"))
        h5file.create_group("options").attrs["json"] = options_to_json(self.options)
        h5file.create_group("data", track_order=True)
        h5file.flush()

    def _public_components(self) -> Tuple[np.ndarray, ...]:
        """Return fields in the model-neutral public/HDF5 component order."""
        if isinstance(self.model, SingleBandModel):
            return (self.psi_d,)
        if isinstance(self.model, SPlusDModel):
            return self.psi_s, self.psi_d
        return self.psi_d, self.psi_s

    def _write_frame(
        self,
        h5file: h5py.File,
        save_index: int,
        *,
        step: int,
        time_value: float,
        dt: float,
        supercurrent: np.ndarray,
        normal_current: np.ndarray,
        state: dict,
    ) -> None:
        group = h5file["data"].create_group(str(save_index))
        group.attrs["step"] = int(step)
        group.attrs["time"] = float(time_value)
        group.attrs["dt"] = float(dt)
        for key, value in state.items():
            if value is not None:
                group.attrs[key] = value
        write_frame_components(group, self._public_components(), model=self.model)
        group["vector_potential"] = self.vector_potential
        group["supercurrent"] = supercurrent
        group["normal_current"] = normal_current
        group["epsilon"] = self.epsilon
        h5file.flush()

    def _frame_diagnostics(self) -> dict:
        """Return topological diagnostics without making zeros fatal.

        Plaquette phase vorticity is undefined if an order-parameter zero lies
        exactly on a site.  The magnetic flux sector itself remains exact in
        that case, so record the sector and mark the phase diagnostic as
        undefined instead of aborting otherwise valid dynamics.
        """
        try:
            vortex_count = int(self.operators.vortex_count(self.psi_d))
            vorticity_defined = True
        except ValueError as exc:
            if "undefined" not in str(exc):
                raise
            vortex_count = int(self.cell.flux_quanta)
            vorticity_defined = False
        return {
            "vortex_count": vortex_count,
            "vorticity_defined": vorticity_defined,
            "mean_induction": float(
                np.mean(self._magnetic_field(self.vector_potential))
            ),
        }

    @staticmethod
    def _aligned_state_error(current, reference):
        current_d, current_s = current
        reference_d, reference_s = reference
        overlap = np.vdot(reference_d, current_d) + np.vdot(reference_s, current_s)
        phase_shift = float(np.angle(overlap))
        align = np.exp(-1j * phase_shift)
        error = max(
            float(np.max(np.abs(align * current_d - reference_d))),
            float(np.max(np.abs(align * current_s - reference_s))),
        )
        return error, phase_shift

    def _thermalize(self) -> None:
        target = float(self.options.skip_time)
        if target <= 0:
            return
        elapsed = 0.0
        step = 0
        while elapsed < target:
            dt = min(self._tentative_dt, target - elapsed)
            (
                self.psi_d,
                self.psi_s,
                self.vector_potential,
                _,
                _,
                used_dt,
                density_change,
            ) = self._advance(self.psi_d, self.psi_s, self.vector_potential, dt)
            elapsed += used_dt
            step += 1
            self._update_adaptive_step(used_dt, density_change)
        logger.info(
            "Thermalized magnetic-periodic state for t=%g in %d steps.", elapsed, step
        )
        self._tentative_dt = float(self.options.dt_init)
        self._change_history.clear()

    def solve(self) -> MagneticPeriodicSolution:
        """Advance the coupled state and return an incrementally saved solution."""
        started = time.perf_counter()
        self._thermalize()
        output_path = self._unique_output_path(self.options.output_file)
        elapsed = 0.0
        step = 0
        save_index = 0
        supercurrent, normal_current = self._initial_currents()
        equilibrium_state = dict(
            equilibrium_reached=False,
            equilibrium_error=np.inf,
            equilibrium_order_parameter_error=np.inf,
            equilibrium_electromagnetic_error=np.inf,
            equilibrium_phase_shift=np.nan,
            equilibrium_checks=0,
            equilibrium_reference_step=0,
            equilibrium_time=np.nan,
        )
        reference_order = (self.psi_d.copy(), self.psi_s.copy())
        reference_a = self.vector_potential.copy()
        since_check = 0
        last_saved_step = None

        mode = "w" if os.path.exists(output_path) else "x"
        with h5py.File(output_path, mode) as h5file:
            self._write_header(h5file)
            self._write_frame(
                h5file,
                save_index,
                step=step,
                time_value=elapsed,
                dt=0.0,
                supercurrent=supercurrent,
                normal_current=normal_current,
                state=equilibrium_state,
            )
            save_index += 1
            last_saved_step = step

            while elapsed < self.options.solve_time:
                dt = min(self._tentative_dt, self.options.solve_time - elapsed)
                (
                    self.psi_d,
                    self.psi_s,
                    self.vector_potential,
                    supercurrent,
                    normal_current,
                    used_dt,
                    density_change,
                ) = self._advance(self.psi_d, self.psi_s, self.vector_potential, dt)
                elapsed += used_dt
                step += 1
                since_check += 1
                self._update_adaptive_step(used_dt, density_change)

                if (
                    self.options.equilibrium_tolerance is not None
                    and since_check >= self.options.equilibrium_window
                ):
                    order_error, phase_shift = self._aligned_state_error(
                        (self.psi_d, self.psi_s), reference_order
                    )
                    electromagnetic_error = float(
                        np.max(np.abs(self.vector_potential - reference_a))
                    )
                    error = max(order_error, electromagnetic_error)
                    equilibrium_state.update(
                        equilibrium_error=error,
                        equilibrium_order_parameter_error=order_error,
                        equilibrium_electromagnetic_error=electromagnetic_error,
                        equilibrium_phase_shift=phase_shift,
                        equilibrium_checks=equilibrium_state["equilibrium_checks"] + 1,
                    )
                    if (
                        elapsed >= self.options.equilibrium_min_time
                        and error <= self.options.equilibrium_tolerance
                    ):
                        equilibrium_state["equilibrium_reached"] = True
                        equilibrium_state["equilibrium_time"] = elapsed
                    else:
                        reference_order = (self.psi_d.copy(), self.psi_s.copy())
                        reference_a = self.vector_potential.copy()
                        equilibrium_state["equilibrium_reference_step"] = step
                        since_check = 0

                should_save = step % self.options.save_every == 0
                finished = elapsed >= self.options.solve_time or bool(
                    equilibrium_state["equilibrium_reached"]
                )
                if should_save or finished:
                    if supercurrent is None:
                        self.operators.set_vector_potential(self.vector_potential)
                        supercurrent = self._normalized_supercurrent(
                            self.psi_d, self.psi_s
                        )
                        normal_current = np.zeros_like(supercurrent)
                    frame_state = dict(equilibrium_state)
                    frame_state.update(self._frame_diagnostics())
                    self._write_frame(
                        h5file,
                        save_index,
                        step=step,
                        time_value=elapsed,
                        dt=used_dt,
                        supercurrent=supercurrent,
                        normal_current=normal_current,
                        state=frame_state,
                    )
                    save_index += 1
                    last_saved_step = step
                if finished:
                    break

            if step != last_saved_step:
                if supercurrent is None:
                    self.operators.set_vector_potential(self.vector_potential)
                    supercurrent = self._normalized_supercurrent(self.psi_d, self.psi_s)
                    normal_current = np.zeros_like(supercurrent)
                frame_state = dict(equilibrium_state)
                frame_state.update(self._frame_diagnostics())
                self._write_frame(
                    h5file,
                    save_index,
                    step=step,
                    time_value=elapsed,
                    dt=used_dt,
                    supercurrent=supercurrent,
                    normal_current=normal_current,
                    state=frame_state,
                )
            h5file.attrs["total_seconds"] = time.perf_counter() - started
            h5file.attrs["final_step"] = int(step)
            h5file.attrs["final_time"] = float(elapsed)
            h5file.attrs["complete"] = True
            h5file.flush()

        return MagneticPeriodicSolution.from_hdf5(output_path)


def solve_magnetic_periodic(
    cell,
    options: SolverOptions,
    *,
    disorder_epsilon: Union[float, Callable] = 1.0,
    seed_solution: Optional[MagneticPeriodicSolution] = None,
    initial_psi_d: Optional[np.ndarray] = None,
    initial_psi_s: Optional[np.ndarray] = None,
    initial_psi1: Optional[np.ndarray] = None,
    initial_psi2: Optional[np.ndarray] = None,
    initial_vector_potential: Optional[np.ndarray] = None,
) -> MagneticPeriodicSolution:
    """Solve any supported TDGL model on a magnetic-periodic rectangle."""
    return MagneticPeriodicSolver(
        cell,
        options,
        disorder_epsilon=disorder_epsilon,
        seed_solution=seed_solution,
        initial_psi_d=initial_psi_d,
        initial_psi_s=initial_psi_s,
        initial_psi1=initial_psi1,
        initial_psi2=initial_psi2,
        initial_vector_potential=initial_vector_potential,
    ).solve()


__all__ = [
    "MagneticPeriodicSolver",
    "d_plus_d_prime_free_energy_density",
    "magnetic_periodic_free_energy_density",
    "magnetic_periodic_virial_applied_field",
    "single_band_free_energy_density",
    "s_plus_s_free_energy_density",
    "s_plus_d_free_energy_density",
    "s_plus_d_virial_applied_field",
    "solve_magnetic_periodic",
]
