import inspect
import itertools
import logging
import math
import numbers
import os
from datetime import datetime
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.optimize as spo
import scipy.sparse as sp

try:
    import cupy  # type: ignore
except ImportError:
    cupy = None

try:
    import pypardiso  # type: ignore
except ImportError:
    pypardiso = None

from ..device.device import Device, TerminalInfo
from ..device.models import DPlusDPrimeModel, SingleBandModel, SPlusDModel, SPlusSModel
from ..finite_volume.operators import MeshOperators
from ..parameter import Parameter
from ..solution.solution import Solution
from ..sources.constant import ConstantField
from .options import SolverOptions, SparseSolver
from .runner import DataHandler, Runner, RunningState
from .screening import get_A_induced_cupy, get_A_induced_numba

logger = logging.getLogger("solver")


def _check_terminal_currents(
    currents: Dict[str, float], terminal_info: Sequence[TerminalInfo]
) -> None:
    """Validate one terminal-current mapping at a specific time."""
    if not isinstance(currents, dict):
        raise TypeError("Terminal current callables must return a dict.")
    names = {terminal.name for terminal in terminal_info}
    if unknown := set(currents).difference(names):
        raise ValueError(
            f"Unknown terminal(s) in terminal currents: {sorted(unknown)}."
        )
    values = []
    for name, value in currents.items():
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"Current for terminal {name!r} must be a real number.")
        if not math.isfinite(value):
            raise ValueError(f"Current for terminal {name!r} must be finite.")
        values.append(float(value))
    total_current = math.fsum(values)
    scale = max(1.0, math.fsum(abs(value) for value in values))
    if not math.isclose(total_current, 0.0, rel_tol=0.0, abs_tol=1e-12 * scale):
        raise ValueError(
            f"The sum of all terminal currents must be 0 (got {total_current:.2e})."
        )


def get_outward_boundary_normals(mesh) -> np.ndarray:
    """Return outward unit normals in boundary-edge order."""
    boundary_indices = mesh.edge_mesh.boundary_edge_indices
    boundary_edges = mesh.edge_mesh.edges[boundary_indices]
    third_sites = {}
    for element in mesh.elements:
        for i, j, k in (
            (element[0], element[1], element[2]),
            (element[1], element[2], element[0]),
            (element[2], element[0], element[1]),
        ):
            edge = tuple(sorted((i, j)))
            if edge in third_sites:
                third_sites[edge] = None
            else:
                third_sites[edge] = k

    normals = np.empty((len(boundary_edges), 2), dtype=float)
    for index, edge in enumerate(boundary_edges):
        direction = mesh.sites[edge[1]] - mesh.sites[edge[0]]
        direction /= np.linalg.norm(direction)
        normal = np.array([direction[1], -direction[0]])
        midpoint = mesh.sites[edge].mean(axis=0)
        third_site = third_sites[tuple(edge)]
        if third_site is None:
            raise ValueError(f"Boundary edge {tuple(edge)} has two adjacent triangles.")
        if np.dot(normal, mesh.sites[third_site] - midpoint) > 0:
            normal *= -1
        normals[index] = normal
    return normals


def validate_terminal_currents(
    terminal_currents: Union[Callable, Dict[str, float]],
    terminal_info: Sequence[TerminalInfo],
    solver_options: SolverOptions,
    num_evals: int = 100,
) -> None:
    """Ensure that the terminal currents satisfy current conservation."""
    if callable(terminal_currents):
        times = np.linspace(0.0, solver_options.solve_time, max(2, num_evals))
        for t in times:
            _check_terminal_currents(terminal_currents(float(t)), terminal_info)
    else:
        _check_terminal_currents(terminal_currents, terminal_info)


def _s_plus_s_uniform_state(model: SPlusSModel) -> Tuple[complex, complex]:
    """Return a deterministic homogeneous minimum for ``SPlusSModel``.

    The legacy model initialized each band at its uncoupled amplitude and used
    only the Josephson-preferred sign.  Preserve that exact behavior when the
    new interband quartics vanish.  Once either quartic is enabled, minimize the
    complete homogeneous potential over two nonnegative amplitudes and one
    relative phase.  Restricting the phase to ``[0, pi]`` chooses one member of
    a time-reversed pair deterministically; a conjugated seed solution selects
    the opposite chirality.
    """

    def uncoupled_amplitude(a: float, b: float) -> float:
        if b > 0 and a < 0:
            return max(math.sqrt(-a / b), 1e-4)
        return 1e-4

    amp1 = uncoupled_amplitude(model.a1, model.b1)
    amp2 = uncoupled_amplitude(model.a2, model.b2)
    if model.phase_gamma2 == 0 and model.density_gamma3 == 0:
        relative_sign = -1.0 if model.josephson_gamma < 0 else 1.0
        return complex(amp1), complex(relative_sign * amp2)

    def potential(values: np.ndarray) -> float:
        u1, u2, theta = values
        rho1 = u1 * u1
        rho2 = u2 * u2
        return float(
            model.a1 * rho1
            + 0.5 * model.b1 * rho1 * rho1
            + model.a2 * rho2
            + 0.5 * model.b2 * rho2 * rho2
            + 0.5 * model.density_gamma3 * rho1 * rho2
            + model.phase_gamma2 * rho1 * rho2 * math.cos(2 * theta)
            - 2 * model.josephson_gamma * u1 * u2 * math.cos(theta)
        )

    def gradient(values: np.ndarray) -> np.ndarray:
        u1, u2, theta = values
        cos_theta = math.cos(theta)
        cos_two_theta = math.cos(2 * theta)
        common = model.density_gamma3 + 2 * model.phase_gamma2 * cos_two_theta
        return np.array(
            [
                2 * model.a1 * u1
                + 2 * model.b1 * u1**3
                + common * u1 * u2**2
                - 2 * model.josephson_gamma * u2 * cos_theta,
                2 * model.a2 * u2
                + 2 * model.b2 * u2**3
                + common * u2 * u1**2
                - 2 * model.josephson_gamma * u1 * cos_theta,
                -2 * model.phase_gamma2 * u1**2 * u2**2 * math.sin(2 * theta)
                + 2 * model.josephson_gamma * u1 * u2 * math.sin(theta),
            ],
            dtype=float,
        )

    scale1 = max(amp1, 1.0)
    scale2 = max(amp2, 1.0)
    starts = [
        np.array([amp1, amp2, theta], dtype=float)
        for theta in (0.0, math.pi / 2, math.pi)
    ]
    starts.extend(
        np.array([scale1, scale2, theta], dtype=float)
        for theta in (0.0, math.pi / 2, math.pi)
    )
    starts.extend(
        [
            np.array([amp1, 0.0, 0.0], dtype=float),
            np.array([0.0, amp2, 0.0], dtype=float),
            np.zeros(3, dtype=float),
        ]
    )

    best_values = starts[0]
    best_energy = potential(best_values)
    bounds = ((0.0, None), (0.0, None), (0.0, math.pi))
    for start in starts:
        result = spo.minimize(
            potential,
            start,
            jac=gradient,
            bounds=bounds,
            method="L-BFGS-B",
        )
        values = result.x if result.success and np.all(np.isfinite(result.x)) else start
        energy = potential(values)
        if energy < best_energy:
            best_values = values
            best_energy = energy

    u1, u2, theta = best_values
    u1 = max(float(u1), 1e-4)
    u2 = max(float(u2), 1e-4)
    return complex(u1), u2 * np.exp(1j * float(theta))


class SolverResult(NamedTuple):
    """A container for the results of a single solve step.

    dt: The time step size used for the solve step
    psi1: The first order-parameter component.
    psi2: The second order-parameter component (zero in single-band mode).
    mu: The scalar potential
    supercurrent: The supercurrent density
    normal_current: The normal current density
    A_induced: The induced vector potential
    A_applied: The applied vector potential. This will be ``None`` in the case of
        a time-independent vector potential.
    epsilon: The disorder parameter, ``epsilon``. This will be ``None`` in the case of
        a time-independent ``epsilon``.
    """

    dt: float
    psi2: np.ndarray
    psi1: np.ndarray
    mu: np.ndarray
    supercurrent: np.ndarray
    normal_current: np.ndarray
    A_induced: np.ndarray
    A_applied: Optional[np.ndarray] = None
    epsilon: Optional[np.ndarray] = None


class TDGLSolver:
    """Solver for a TDGL model.

    An instance of :class:`tdgl.TDGLSolver` is created and executed
    by calling :func:`tdgl.solve`.

    Args:
        device: The :class:`tdgl.Device` to solve.
        options: An instance :class:`tdgl.SolverOptions` specifying the solver
            parameters.
        applied_vector_potential: A function or :class:`tdgl.Parameter` that computes
            the applied vector potential as a function of position ``(x, y, z)``,
            or of position and time ``(x, y, z, *, t)``. If a float ``B`` is given,
            the applied vector potential will be that of a uniform magnetic field with
            strength ``B`` ``field_units``. If the applied vector potential is time-dependent,
            this argument must be a :class:`tdgl.Parameter`.
        terminal_currents: A dict of ``{terminal_name: current}`` or a callable with signature
            ``func(time: float) -> {terminal_name: current}``, where ``current`` is a float
            in units of ``current_units`` and ``time`` is the dimensionless time.
        disorder_epsilon: A float <= 1, or a function that returns
            :math:`\\epsilon\\leq 1` as a function of position ``r=(x, y)`` or
            position and time ``(x, y, *, t)``.
            Setting :math:`\\epsilon(\\mathbf{r}, t)=T_c/T - 1 < 1` suppresses the
            order parameter at position :math:`\\mathbf{r}=(x, y)`, which can be used
            to model inhomogeneity.
        seed_solution: A :class:`tdgl.Solution` instance to use as the initial state
            for the simulation.
    """

    def __init__(
        self,
        device: Device,
        options: SolverOptions,
        applied_vector_potential: Union[Callable, float] = 0.0,
        terminal_currents: Union[Callable, Dict[str, float], None] = None,
        disorder_epsilon: Union[Callable, float] = 1.0,
        seed_solution: Optional[Solution] = None,
    ):
        self.device = device
        self.options = options
        self.options.validate()
        self.terminal_currents = terminal_currents
        self.seed_solution = seed_solution

        if self.options.gpu:
            assert cupy is not None
            self.xp = cupy
            self.use_cupy = True
        else:
            self.xp = np
            self.use_cupy = False

        mesh = self.device.mesh
        ureg = self.device.ureg
        self.probe_points = device.probe_point_indices
        length_units = ureg(self.device.length_units)
        field_units = options.field_units
        current_units = options.current_units

        edges = mesh.edge_mesh.edges
        self.num_edges = len(edges)
        normalized_directions = mesh.edge_mesh.normalized_directions
        length_units = ureg(device.length_units)
        xi = device.coherence_length.magnitude
        self.model = device.layer.model
        validate_model = getattr(self.model, "validate", None)
        if validate_model is not None:
            validate_model()
        self.goncalves_screening = isinstance(self.model, SPlusDModel) and (
            options.include_screening
        )
        self.s_plus_s_screening = isinstance(self.model, SPlusSModel) and (
            options.include_screening
        )
        self.local_screening = self.goncalves_screening or self.s_plus_s_screening
        if self.local_screening and self.use_cupy:
            raise ValueError(
                "Local multicomponent screening currently requires the CPU sparse "
                "solver."
            )
        if self.local_screening and terminal_currents is not None:
            if callable(terminal_currents) or any(terminal_currents.values()):
                raise ValueError(
                    "Local multicomponent screening uses the phi=0 vacuum-boundary "
                    "problem and does not support terminal current injection."
                )
        if self.s_plus_s_screening and not math.isclose(
            self.model.em_coupling, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "SPlusSModel local screening requires em_coupling=1 so that the "
                "condensate current and magnetic free energy use one variational "
                "normalization."
            )
        drive_current = np.array(
            [
                options.s_plus_d_drive_current_x,
                options.s_plus_d_drive_current_y,
            ],
            dtype=float,
        )
        if np.any(drive_current) and not self.goncalves_screening:
            raise ValueError(
                "The s+d bulk drive current requires SPlusDModel with "
                "include_screening=True."
            )
        self.s_plus_d_drive_current = drive_current
        self.s_plus_d_drive_tangent = normalized_directions @ drive_current
        if isinstance(self.model, DPlusDPrimeModel) and options.include_screening:
            raise ValueError(
                f"{self.model.__class__.__name__} currently supports a prescribed "
                "vector potential only; set include_screening=False."
            )
        self.u = device.layer.u
        K0 = device.K0
        A0 = device.A0
        Bc2 = device.Bc2

        # The vector potential is evaluated on the mesh edges,
        # where the edge coordinates are in dimensionful units.
        self.sites = xi * mesh.sites
        self.edge_centers = xi * mesh.edge_mesh.centers
        self.z0 = device.layer.z0 * np.ones(len(self.edge_centers), dtype=float)

        self.dynamic_vector_potential = (
            isinstance(applied_vector_potential, Parameter)
            and applied_vector_potential.time_dependent
        )
        if self.local_screening and self.dynamic_vector_potential:
            raise ValueError(
                "Local multicomponent screening requires a time-independent "
                "applied vector potential during each solve. Use continuation "
                "between successive static applied fields."
            )
        if not callable(applied_vector_potential):
            applied_vector_potential = ConstantField(
                applied_vector_potential,
                field_units=field_units,
                length_units=device.length_units,
            )
        self.applied_vector_potential = applied_vector_potential
        # Evaluate the vector potential
        self.A_scale = (
            (ureg(field_units) * length_units / (Bc2 * xi * length_units))
            .to_base_units()
            .magnitude
        )
        A_kwargs = dict(t=0) if self.dynamic_vector_potential else dict()
        current_A_applied = self.applied_vector_potential(
            self.edge_centers[:, 0], self.edge_centers[:, 1], self.z0, **A_kwargs
        )
        current_A_applied = np.asarray(current_A_applied)
        if current_A_applied.ndim != 2 or current_A_applied.shape[1] < 2:
            raise ValueError(
                f"Unexpected shape for vector_potential: {current_A_applied.shape}."
            )
        current_A_applied = self.A_scale * current_A_applied[:, :2]
        if current_A_applied.shape != self.edge_centers.shape:
            raise ValueError(
                f"Unexpected shape for vector_potential: {current_A_applied.shape}."
            )
        if not np.all(np.isfinite(current_A_applied)):
            raise ValueError("The applied vector potential must be finite.")

        # Create the epsilon parameter, which sets the local critical temperature.
        if callable(disorder_epsilon):
            argspec = inspect.getfullargspec(disorder_epsilon)
            self.dynamic_epsilon = "t" in argspec.kwonlyargs
            self.vectorized_epsilon = (
                argspec.kwonlydefaults is not None
                and argspec.kwonlydefaults.get("vectorized", False)
            )
        else:
            # epsilon constant as a function of both position and time
            _disorder_epsilon = disorder_epsilon

            def disorder_epsilon(r):
                return _disorder_epsilon * np.ones(len(r), dtype=float)

            self.vectorized_epsilon = True
            self.dynamic_epsilon = False

        self.disorder_epsilon = disorder_epsilon
        kw = dict(t=0) if self.dynamic_epsilon else dict()
        if self.vectorized_epsilon:
            epsilon = disorder_epsilon(self.sites, **kw)
        else:
            epsilon = np.array([float(disorder_epsilon(r, **kw)) for r in self.sites])
        epsilon = self._normalize_epsilon(epsilon)
        s_plus_s_disorder_mapped = isinstance(self.model, SPlusSModel) and (
            self.model.disorder_coupling1 != 0 or self.model.disorder_coupling2 != 0
        )
        unsupported_disorder = isinstance(self.model, DPlusDPrimeModel) or (
            isinstance(self.model, SPlusSModel) and not s_plus_s_disorder_mapped
        )
        if unsupported_disorder and (
            self.dynamic_epsilon or not np.allclose(epsilon, 1)
        ):
            raise ValueError(
                f"{self.model.__class__.__name__} uses fixed-temperature quadratic "
                "coefficients without an explicit disorder mapping and does not "
                "support disorder_epsilon values other than 1."
            )

        # Clear the Parameter caches
        if isinstance(self.applied_vector_potential, Parameter):
            self.applied_vector_potential._clear_cache()
        if isinstance(self.disorder_epsilon, Parameter):
            self.disorder_epsilon._clear_cache()

        # Find the current terminal sites.
        self.terminal_info = device.terminal_info()
        self.terminal_names = [term.name for term in self.terminal_info]
        for term_info in self.terminal_info:
            if term_info.length == 0:
                raise ValueError(
                    f"Terminal {term_info.name!r} does not contain any points"
                    " on the boundary of the mesh."
                )
        # Define the source-drain current.
        if terminal_currents and device.probe_points is None:
            logger.warning(
                "The terminal currents are non-null, but the device has no probe points."
            )
        terminal_names = [term.name for term in self.terminal_info]
        if terminal_currents is None:
            terminal_currents = {name: 0 for name in terminal_names}
        if callable(terminal_currents):
            raw_current_func = terminal_currents
        else:
            _check_terminal_currents(terminal_currents, self.terminal_info)
            supplied_currents = dict(terminal_currents)

            def raw_current_func(t):
                return supplied_currents

        J_scale = 4 * ((ureg(current_units) / length_units) / K0).to_base_units()
        assert J_scale.dimensionless, str(J_scale)
        J_scale = J_scale.magnitude
        if isinstance(self.model, (SPlusDModel, SPlusSModel)):
            # Multicomponent diffusion clocks store transport currents as
            # J / beta_em. Solution reconstruction restores this factor.
            J_scale /= self.model.beta_em

        def scaled_current_func(t):
            currents = raw_current_func(t)
            _check_terminal_currents(currents, self.terminal_info)
            scaled = {key: J_scale * value for key, value in currents.items()}
            _check_terminal_currents(scaled, self.terminal_info)
            return scaled

        self.current_func = scaled_current_func
        validate_terminal_currents(self.current_func, self.terminal_info, self.options)
        terminal_indices = [t.site_indices for t in self.terminal_info]
        if terminal_indices:
            normal_boundary_index = np.concatenate(terminal_indices, dtype=np.int64)
        else:
            normal_boundary_index = np.array([], dtype=np.int64)
        # Cache the terminal current densities at each time step.
        # Only update the mu boundary conditions if the current has changed.
        self.terminal_current_densities = {name: 0 for name in self.terminal_names}

        # Construct finite-volume operators
        terminal_psi = options.terminal_psi
        if (
            isinstance(self.model, (DPlusDPrimeModel, SPlusSModel))
            and len(normal_boundary_index)
            and terminal_psi is not None
        ):
            raise ValueError(
                f"{self.model.__class__.__name__} does not support the scalar "
                "terminal_psi boundary condition. Set terminal_psi=None to use "
                "natural boundary conditions."
            )
        self.terminal_psi = terminal_psi
        self.normal_boundary_index = normal_boundary_index
        logger.info("Constructing finite volume operators.")
        operators = MeshOperators(
            mesh,
            options.sparse_solver,
            use_cupy=self.use_cupy,
            fixed_sites=normal_boundary_index,
            fix_psi=(terminal_psi is not None),
            use_fem_for_psi=isinstance(self.model, SPlusDModel),
        )
        operators.build_operators(build_magnetic_diffusion=self.local_screening)
        operators.set_link_exponents(current_A_applied)
        self.operators = operators
        self.applied_triangle_field = operators.get_triangle_magnetic_field(
            current_A_applied
        )
        self.applied_boundary_field = self.applied_triangle_field[
            operators.boundary_triangle_indices
        ]
        if isinstance(self.model, DPlusDPrimeModel):
            self.magnetic_field = operators.get_magnetic_field(current_A_applied)
        else:
            self.magnetic_field = None
        if options.sparse_solver is SparseSolver.PARDISO:
            assert self.operators.mu_laplacian_lu is None
            assert pypardiso is not None

        # Initialize the order parameter and electric potential
        if isinstance(self.model, DPlusDPrimeModel):
            # Canonical order: psi1=d_(x2-y2), psi2=d_xy. An imaginary seed
            # chooses a chirality at zero field; a nonzero Zeeman term selects
            # the thermodynamically favored chirality during relaxation.
            psi1_init = np.ones(len(mesh.sites), dtype=np.complex128)
            psi2_init = np.full(len(mesh.sites), -1e-4j, dtype=np.complex128)
        elif isinstance(self.model, SPlusDModel):
            psi2_init = np.ones(len(mesh.sites), dtype=np.complex128)
            # Goncalves et al. initialize the d component to one and the s
            # component to zero.  Field and boundary anisotropy provide a
            # deterministic source for s through the mixed-gradient term.
            psi1_init = np.zeros(len(mesh.sites), dtype=np.complex128)
            if terminal_psi is not None:
                psi2_init[normal_boundary_index] = terminal_psi
                psi1_init[normal_boundary_index] = 0.0
        elif isinstance(self.model, SPlusSModel):
            uniform_psi1, uniform_psi2 = _s_plus_s_uniform_state(self.model)
            psi1_init = np.full(len(mesh.sites), uniform_psi1, dtype=np.complex128)
            psi2_init = np.full(len(mesh.sites), uniform_psi2, dtype=np.complex128)
        else:
            psi2_init = np.zeros(len(mesh.sites), dtype=np.complex128)
            psi1_init = np.ones(len(mesh.sites), dtype=np.complex128)
            if terminal_psi is not None:
                psi2_init[normal_boundary_index] = 0.0
                psi1_init[normal_boundary_index] = terminal_psi
        mu_init = np.zeros(len(mesh.sites))
        mu_boundary = np.zeros_like(mesh.edge_mesh.boundary_edge_indices, dtype=float)
        boundary_edge_indices = mesh.edge_mesh.boundary_edge_indices
        boundary_normals = get_outward_boundary_normals(mesh)
        dA_boundary_normal = np.zeros(len(boundary_edge_indices), dtype=float)

        if self.use_cupy:
            epsilon = cupy.asarray(epsilon)
            mu_boundary = cupy.asarray(mu_boundary)
            normalized_directions = cupy.asarray(normalized_directions)
            current_A_applied = cupy.asarray(current_A_applied)
            boundary_normals = cupy.asarray(boundary_normals)
            dA_boundary_normal = cupy.asarray(dA_boundary_normal)

        self.psi2_init = psi2_init
        self.psi1_init = psi1_init
        self.mu_init = mu_init
        self.epsilon = epsilon
        self.mu_boundary = mu_boundary
        self.normalized_directions = normalized_directions
        self.boundary_edge_indices = boundary_edge_indices
        self.boundary_normals = boundary_normals
        self.dA_boundary_normal = dA_boundary_normal
        self.current_A_applied = current_A_applied

        self.new_A_induced = None
        self.areas = None
        if options.include_screening and not self.local_screening:
            A_scale = (ureg("mu_0") / (4 * np.pi) * K0 / A0).to(1 / length_units)
            self.new_A_induced = np.empty((self.num_edges, 2), dtype=float)
            self.areas = A_scale.magnitude * mesh.areas * xi**2
            if self.use_cupy:
                self.areas = cupy.asarray(self.areas)
                self.edge_centers = cupy.asarray(self.edge_centers)
                self.sites = cupy.asarray(self.sites)
                self.new_A_induced = cupy.asarray(self.new_A_induced)

        # Running list of the max abs change in |psi|^2 between subsequent solve steps.
        # This list is used to calculate the adaptive time step.
        self.d_psi_sq_vals = []
        self.tentative_dt = options.dt_init
        self.dt_max = options.dt_max if options.adaptive else options.dt_init
        if isinstance(self.model, SPlusDModel):
            h_min = float(np.min(mesh.edge_mesh.edge_lengths))
            rate_d = 1.0
            rate_s = 1 / self.model.relaxation_s
            mixed_rate_sq = self.model.eta_v**2 / (
                self.model.eta_s * self.model.relaxation_s
            )
            condensate_rate = 0.5 * (
                rate_d + rate_s + np.sqrt((rate_d - rate_s) ** 2 + 4 * mixed_rate_sq)
            )
            stable_dt = h_min**2 / (4 * condensate_rate)
            if not options.adaptive and options.dt_init > stable_dt:
                raise ValueError(
                    "The fixed s+d time step exceeds the explicit stability "
                    f"bound ({options.dt_init:.3g} > {stable_dt:.3g})."
                )
            self.dt_max = min(self.dt_max, stable_dt)
            self.tentative_dt = min(self.tentative_dt, self.dt_max)
        elif isinstance(self.model, SPlusSModel) and self.model.mixed_gradient_k12:
            h_min = float(np.min(mesh.edge_mesh.edge_lengths))
            rate1 = 1 / self.model.relaxation1
            rate2 = self.model.k2_over_k1 / self.model.relaxation2
            mixed_rate_sq = self.model.mixed_gradient_k12**2 / (
                self.model.relaxation1 * self.model.relaxation2
            )
            condensate_rate = 0.5 * (
                rate1 + rate2 + np.sqrt((rate1 - rate2) ** 2 + 4 * mixed_rate_sq)
            )
            stable_dt = h_min**2 / (4 * condensate_rate)
            if not options.adaptive and options.dt_init > stable_dt:
                raise ValueError(
                    "The fixed s+s time step exceeds the mixed-gradient explicit "
                    f"stability bound ({options.dt_init:.3g} > {stable_dt:.3g})."
                )
            self.dt_max = min(self.dt_max, stable_dt)
            self.tentative_dt = min(self.tentative_dt, self.dt_max)
        self._s_plus_d_magnetic_dt = None
        self._s_plus_d_magnetic_lu = None

        if options.monitor:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    def update_mu_boundary(self, time: float) -> None:
        """Computes the terminal current density for a given time step and
        updates the scalar potential boundary conditions accordingly.

        Args:
            time: The current value of the dimensionless time.
        """
        # Compute the current density for this step
        # and update the current boundary conditions.
        currents = self.current_func(time)
        terminal_current_densities = self.terminal_current_densities
        for terminal in self.terminal_info:
            current_density = (-1 / terminal.length) * sum(
                currents.get(name, 0)
                for name in self.terminal_names
                if name != terminal.name
            )
            # Only update mu_boundary if the terminal current has changed
            if current_density != terminal_current_densities[terminal.name]:
                terminal_current_densities[terminal.name] = current_density
                self.mu_boundary[terminal.boundary_edge_indices] = current_density

    def update_applied_vector_potential(self, time: float) -> np.ndarray:
        """Evaluates the time-dependent vector potential.

        Args:
            time: The current value of the dimensionless time.

        Returns:
            The new value of the applied vector potential.
        """
        A_applied = self.applied_vector_potential(
            self.edge_centers[:, 0], self.edge_centers[:, 1], self.z0, t=time
        )
        xp = cupy if self.use_cupy else np
        A_applied = xp.asarray(A_applied)
        if A_applied.ndim != 2 or A_applied.shape[1] < 2:
            raise ValueError(
                f"Unexpected shape for vector_potential: {A_applied.shape}."
            )
        A_applied = self.A_scale * A_applied[:, :2]
        expected_shape = (self.num_edges, 2)
        if A_applied.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for vector_potential: {A_applied.shape}; "
                f"expected {expected_shape}."
            )
        if not bool(xp.all(xp.isfinite(A_applied))):
            raise ValueError("The applied vector potential must be finite.")
        return A_applied

    def _normalize_epsilon(self, epsilon: np.ndarray) -> np.ndarray:
        """Return a finite site array satisfying the model's epsilon bound."""
        xp = cupy if self.use_cupy else np
        epsilon = xp.asarray(epsilon, dtype=float)
        if epsilon.ndim == 0:
            epsilon = xp.full(len(self.sites), epsilon, dtype=float)
        expected_shape = (len(self.sites),)
        if epsilon.shape != expected_shape:
            raise ValueError(
                f"The disorder parameter epsilon must have shape {expected_shape}, "
                f"got {epsilon.shape}."
            )
        if not bool(xp.all(xp.isfinite(epsilon))):
            raise ValueError("The disorder parameter epsilon must be finite.")
        if bool(xp.any(epsilon > 1)):
            raise ValueError("The disorder parameter epsilon must be <= 1.")
        return epsilon

    def update_epsilon(self, time: float) -> np.ndarray:
        """Evaluates the time-dependent disorder parameter :math:`\\epsilon`.

        Args:
            time: The current value of the dimensionless time.

        Returns:
            The new value of :math:`\\epsilon`
        """
        if self.vectorized_epsilon:
            epsilon = self.disorder_epsilon(self.sites, t=time)
        else:
            epsilon = np.array(
                [float(self.disorder_epsilon(r, t=time)) for r in self.sites]
            )
        return self._normalize_epsilon(epsilon)

    def solve_for_psi_squared(
        self,
        *,
        psi: np.ndarray,
        abs_sq_psi: np.ndarray,
        mu: np.ndarray,
        epsilon: np.ndarray,
        gamma: float,
        u: float,
        dt: float,
        psi_laplacian: sp.spmatrix,
    ) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """Solves for :math:`\\psi^{n+1}` and :math:`|\\psi^{n+1}|^2` given
        :math:`\\psi^n` and :math:`\\mu^n`.

        Args:
            psi: The current value of the order parameter, :math:`\\psi^n`
            abs_sq_psi: The current value of the superfluid density, :math:`|\\psi^n|^2`
            mu: The current value of the electric potential, :math:`\\mu^n`
            epsilon: The disorder parameter, :math:`\\epsilon`
            gamma: The inelastic scattering parameter, :math:`\\gamma`.
            u: The ratio of relaxation times for the order parameter, :math:`u`
            dt: The time step
            psi_laplacian: The covariant Laplacian for the order parameter

        Returns:
            ``None`` if the calculation failed to converge, otherwise the new order
            parameter :math:`\\psi^{n+1}` and superfluid density :math:`|\\psi^{n+1}|^2`.
        """
        if isinstance(psi, np.ndarray):
            xp = np
        else:
            assert cupy is not None
            assert isinstance(psi, cupy.ndarray)
            xp = cupy
        U = xp.exp(-1j * mu * dt)
        z = U * gamma**2 / 2 * psi
        with np.errstate(all="raise"):
            try:
                w = z * abs_sq_psi + U * (
                    psi
                    + (dt / u)
                    * xp.sqrt(1 + gamma**2 * abs_sq_psi)
                    * ((epsilon - abs_sq_psi) * psi + psi_laplacian @ psi)
                )
                c = w.real * z.real + w.imag * z.imag
                two_c_1 = 2 * c + 1
                w2 = xp.absolute(w) ** 2
                discriminant = two_c_1**2 - 4 * xp.absolute(z) ** 2 * w2
            except Exception:
                logger.warning("Unable to solve for |psi|^2.", exc_info=True)
                return None
        if xp.any(discriminant < 0):
            return None
        new_sq_psi = (2 * w2) / (two_c_1 + xp.sqrt(discriminant))
        psi = w - z * new_sq_psi
        return psi, new_sq_psi

    def adaptive_euler_step(
        self,
        step: int,
        psi2: np.ndarray,
        psi1: np.ndarray,
        abs_sq_psi2: np.ndarray,
        abs_sq_psi1: np.ndarray,
        mu: np.ndarray,
        epsilon_disorder: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        options = self.options
        if isinstance(psi2, np.ndarray):
            xp = np
        else:
            xp = cupy

        if isinstance(self.model, SingleBandModel):
            # Fall back to original pyTDGL solver for single band
            kwargs = dict(
                psi=psi1,
                abs_sq_psi=abs_sq_psi1,
                mu=mu,
                epsilon=epsilon_disorder,
                gamma=self.model.gamma,
                u=self.u,
                dt=dt,
                psi_laplacian=self.operators.psi_laplacian,
            )
            result = self.solve_for_psi_squared(**kwargs)
            for retries in itertools.count():
                if result is not None:
                    break  # First evaluation of |psi|^2 was successful.
                if not options.adaptive or retries >= options.max_solve_retries:
                    raise RuntimeError(
                        f"Solver failed to converge in {options.max_solve_retries}"
                        f" retries at step {step} with dt = {dt:.2e}."
                        f" Try using a smaller dt_init or dt_max."
                    )
                kwargs["dt"] = dt = dt * options.adaptive_time_step_multiplier
                result = self.solve_for_psi_squared(**kwargs)

            new_psi1, new_sq_s = result
            new_psi2 = psi2
            new_sq_d = abs_sq_psi2
            if self.terminal_psi is not None and len(self.normal_boundary_index):
                new_psi1[self.normal_boundary_index] = self.terminal_psi
                new_sq_s[self.normal_boundary_index] = abs(self.terminal_psi) ** 2
            return new_psi2, new_psi1, new_sq_d, new_sq_s, dt

        # Calculate the component Laplacians. The semantic order depends on the
        # model, so the common stepping code uses component-neutral names.
        lap2 = self.operators.psi_laplacian @ psi2
        lap1 = self.operators.psi_laplacian @ psi1

        if isinstance(self.model, DPlusDPrimeModel):
            magnetic_field = self.magnetic_field
            if magnetic_field is None:
                raise RuntimeError("DPlusDPrimeModel requires a local magnetic field.")
            zeeman = self.model.zeeman_coupling * magnetic_field
            # psi1=d_(x2-y2), psi2=d_xy. These are the negative functional
            # gradients of arXiv:cond-mat/0004227 Eq. (2), plus the orbital
            # Zeeman coupling of arXiv:cond-mat/9909399 Eq. (11).
            rhs1 = (
                lap1
                + epsilon_disorder * psi1
                - abs_sq_psi1 * psi1
                - (2 / 3) * abs_sq_psi2 * psi1
                - (1 / 3) * (psi2**2) * xp.conj(psi1)
                + 1j * zeeman * psi2
            )
            rhs2 = (
                lap2
                + self.model.alpha * psi2
                - abs_sq_psi2 * psi2
                - (2 / 3) * abs_sq_psi1 * psi2
                - (1 / 3) * (psi1**2) * xp.conj(psi2)
                - 1j * zeeman * psi1
            )
        elif isinstance(self.model, SPlusDModel):
            if self.model.eta_v != 0:
                lap_x_d = self.operators.laplacian_x @ psi2
                lap_y_d = self.operators.laplacian_y @ psi2
                lap_x_s = self.operators.laplacian_x @ psi1
                lap_y_s = self.operators.laplacian_y @ psi1

                # Goncalves et al. Eqs. (17)-(18), using
                # Pi_i = i D_i for this code's D_i = partial_i - i A_i.
                mixed_d = self.model.eta_v * (lap_y_s - lap_x_s)
                mixed_s = self.model.eta_v * (lap_y_d - lap_x_d)
            else:
                mixed_d = 0
                mixed_s = 0

            # Canonical dimensionless d+s equations. Disorder always modifies
            # the d-sector coefficient and may be coupled into nu_eff.
            rhs2 = (
                lap2
                + epsilon_disorder * psi2
                - abs_sq_psi2 * psi2
                - 0.5 * self.model.tau3 * abs_sq_psi1 * psi2
                - self.model.tau4 * (psi1**2) * xp.conj(psi2)
                + mixed_d
            )
            rhs1 = (
                self.model.eta_s * lap1
                + (
                    self.model.nu
                    + self.model.nu_disorder_coupling * (epsilon_disorder - 1)
                )
                * psi1
                - self.model.tau1 * abs_sq_psi1 * psi1
                - 0.5 * self.model.tau3 * abs_sq_psi2 * psi1
                - self.model.tau4 * (psi2**2) * xp.conj(psi1)
                + mixed_s
            )
        elif isinstance(self.model, SPlusSModel):
            # Negative free-energy gradients for Lin--Maiti--Chubukov's two
            # isotropic s-wave fields. Positive josephson_gamma favors equal
            # phases; positive phase_gamma2 at zero Josephson coupling favors
            # the time-reversed relative phases +/- pi/2.
            a1_effective = self.model.a1 + self.model.disorder_coupling1 * (
                1 - epsilon_disorder
            )
            a2_effective = self.model.a2 + self.model.disorder_coupling2 * (
                1 - epsilon_disorder
            )
            rhs2 = (
                self.model.k2_over_k1 * lap2
                + self.model.mixed_gradient_k12 * lap1
                - a2_effective * psi2
                - self.model.b2 * abs_sq_psi2 * psi2
                - 0.5 * self.model.density_gamma3 * abs_sq_psi1 * psi2
                - self.model.phase_gamma2 * xp.conj(psi2) * psi1**2
                + self.model.josephson_gamma * psi1
            )
            rhs1 = (
                lap1
                + self.model.mixed_gradient_k12 * lap2
                - a1_effective * psi1
                - self.model.b1 * abs_sq_psi1 * psi1
                - 0.5 * self.model.density_gamma3 * abs_sq_psi2 * psi1
                - self.model.phase_gamma2 * xp.conj(psi1) * psi2**2
                + self.model.josephson_gamma * psi2
            )
        else:
            raise ValueError(
                f"Unsupported model for adaptive_euler_step: {type(self.model)}"
            )

        # We need an adaptive step loop
        for retries in itertools.count():
            # Rebuild the temporal link whenever a rejected step changes dt.
            U = xp.exp(-1j * mu * dt)
            # Euler step
            if isinstance(self.model, SPlusDModel):
                new_psi2 = U * (psi2 + dt * rhs2)
                new_psi1 = U * (
                    psi1 + (dt / (self.model.eta_s * self.model.relaxation_s)) * rhs1
                )
            elif isinstance(self.model, DPlusDPrimeModel):
                new_psi2 = U * (psi2 + (dt / self.model.relaxation_d_prime) * rhs2)
                new_psi1 = U * (psi1 + (dt / self.model.relaxation_d) * rhs1)
            else:
                new_psi2 = U * (psi2 + (dt / self.model.relaxation2) * rhs2)
                new_psi1 = U * (psi1 + (dt / self.model.relaxation1) * rhs1)

            # For stability, check if order parameter blew up or changed too rapidly
            if xp.all(xp.isfinite(new_psi2)) and xp.all(xp.isfinite(new_psi1)):
                change_d_values = xp.absolute(new_psi2 - U * psi2)
                change_s_values = xp.absolute(new_psi1 - U * psi1)
                if self.terminal_psi is not None and len(self.normal_boundary_index):
                    # Dirichlet sites are clamped below and do not constrain dt.
                    change_d_values[self.normal_boundary_index] = 0
                    change_s_values[self.normal_boundary_index] = 0
                change_d = xp.max(change_d_values)
                change_s = xp.max(change_s_values)
                # Enforce a max change of 0.1 per step to maintain explicit Euler stability
                if change_d <= 0.1 and change_s <= 0.1:
                    break

            if not options.adaptive or retries >= options.max_solve_retries:
                raise RuntimeError(
                    f"Solver failed to converge in {options.max_solve_retries}"
                    f" retries at step {step} with dt = {dt:.2e}."
                    f" Try using a smaller dt_init or dt_max."
                )
            dt = dt * options.adaptive_time_step_multiplier

        if self.terminal_psi is not None and len(self.normal_boundary_index):
            if isinstance(self.model, DPlusDPrimeModel):
                new_psi1[self.normal_boundary_index] = self.terminal_psi
                new_psi2[self.normal_boundary_index] = 0.0
            else:
                new_psi2[self.normal_boundary_index] = self.terminal_psi
                new_psi1[self.normal_boundary_index] = 0.0

        new_sq_d = xp.absolute(new_psi2) ** 2
        new_sq_s = xp.absolute(new_psi1) ** 2

        return new_psi2, new_psi1, new_sq_d, new_sq_s, dt

    def compute_s_plus_s_free_energy(
        self,
        psi1: np.ndarray,
        psi2: np.ndarray,
        vector_potential: Optional[np.ndarray] = None,
        include_magnetic: bool = False,
        average: bool = False,
    ) -> float:
        """Return the discrete Gibbs free energy for ``SPlusSModel``.

        The condensate terms are the exact Hermitian edge discretization whose
        variations produce the two TDGL residuals and the mixed-gradient
        current. ``vector_potential`` defaults to the links currently installed
        in the operators. Magnetic energy remains opt-in for compatibility with
        the original condensate-only diagnostic.
        """
        if not isinstance(self.model, SPlusSModel):
            raise TypeError("compute_s_plus_s_free_energy requires SPlusSModel.")
        xp = np if isinstance(psi1, np.ndarray) else cupy
        model = self.model
        operators = self.operators
        areas = operators.areas
        old_vector_potential = operators.link_exponents
        restore_links = vector_potential is not None
        if restore_links:
            operators.set_link_exponents(vector_potential)
        try:
            abs_sq_psi1 = xp.absolute(psi1) ** 2
            abs_sq_psi2 = xp.absolute(psi2) ** 2
            epsilon = xp.asarray(getattr(self, "epsilon", xp.ones_like(abs_sq_psi1)))
            a1_effective = model.a1 + model.disorder_coupling1 * (1 - epsilon)
            a2_effective = model.a2 + model.disorder_coupling2 * (1 - epsilon)
            interband_product = xp.conj(psi1) * psi2
            potential = (
                a1_effective * abs_sq_psi1
                + 0.5 * model.b1 * abs_sq_psi1**2
                + a2_effective * abs_sq_psi2
                + 0.5 * model.b2 * abs_sq_psi2**2
                + 0.5 * model.density_gamma3 * abs_sq_psi1 * abs_sq_psi2
                + model.phase_gamma2 * xp.real(interband_product**2)
                - 2 * model.josephson_gamma * xp.real(interband_product)
            )
            grad1 = operators.psi_gradient @ psi1
            grad2 = operators.psi_gradient @ psi2
            gradient = operators.psi_laplacian_weights * (
                xp.absolute(grad1) ** 2
                + model.k2_over_k1 * xp.absolute(grad2) ** 2
                + 2 * model.mixed_gradient_k12 * xp.real(xp.conj(grad1) * grad2)
            )
            energy = xp.sum(areas * potential)
            energy += xp.sum(operators.edge_lengths**2 * gradient)
        finally:
            if restore_links and old_vector_potential is not None:
                operators.set_link_exponents(old_vector_potential)

        if include_magnetic:
            if vector_potential is None:
                vector_potential = operators.link_exponents
            if vector_potential is None:
                raise ValueError("vector_potential is required for magnetic energy.")
            induction = operators.get_triangle_magnetic_field(vector_potential)
            applied = xp.asarray(self.applied_triangle_field)
            coords = self.device.mesh.sites[self.device.mesh.elements]
            edge_1 = coords[:, 1] - coords[:, 0]
            edge_2 = coords[:, 2] - coords[:, 0]
            triangle_areas = 0.5 * xp.absolute(
                edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
            )
            energy += xp.sum(
                triangle_areas * self.device.kappa**2 * (induction - applied) ** 2
            )
        if average:
            energy /= xp.sum(areas)
        return float(energy)

    def compute_s_plus_d_free_energy(
        self,
        d: np.ndarray,
        s: np.ndarray,
        vector_potential: Optional[np.ndarray] = None,
        include_magnetic: bool = True,
        average: bool = True,
    ) -> float:
        r"""Return the discrete Gibbs free energy of Goncalves et al. Eq. (78).

        The condensate terms use the same gauge links and directional finite-
        element forms as the TDGL equations.  If ``include_magnetic`` is true,
        the last term is :math:`\kappa^2(B-H)^2`, evaluated per triangle.
        ``vector_potential`` defaults to the links currently installed in the
        solver operators.  By default the integral is divided by sample area,
        matching the paper's definition; set ``average=False`` for the total.
        """
        if not isinstance(self.model, SPlusDModel):
            raise TypeError("compute_s_plus_d_free_energy requires SPlusDModel.")
        xp = np if isinstance(d, np.ndarray) else cupy
        model = self.model
        operators = self.operators
        areas = operators.areas
        old_vector_potential = operators.link_exponents
        restore_links = vector_potential is not None
        if restore_links:
            operators.set_link_exponents(vector_potential)
        try:
            abs_sq_d = xp.absolute(d) ** 2
            abs_sq_s = xp.absolute(s) ** 2
            potential = (
                -xp.asarray(self.epsilon) * abs_sq_d
                - (
                    model.nu
                    + model.nu_disorder_coupling * (xp.asarray(self.epsilon) - 1)
                )
                * abs_sq_s
                + 0.5 * abs_sq_d**2
                + 0.5 * model.tau1 * abs_sq_s**2
                + 0.5 * model.tau3 * abs_sq_d * abs_sq_s
                + model.tau4 * xp.real((xp.conj(s) ** 2) * d**2)
            )
            # Evaluate the Hermitian edge energy directly. The evolution
            # matrices replace fixed rows for Dirichlet boundary conditions,
            # so using those matrices here would not be the variational energy.
            grad_d = operators.psi_gradient @ d
            grad_s = operators.psi_gradient @ s
            edge_length_sq = operators.edge_lengths**2
            gradient = operators.psi_laplacian_weights * (
                xp.absolute(grad_d) ** 2 + model.eta_s * xp.absolute(grad_s) ** 2
            )
            gradient += (
                2
                * model.eta_v
                * (operators.laplacian_weights_y - operators.laplacian_weights_x)
                * xp.real(xp.conj(grad_s) * grad_d)
            )
            energy = xp.sum(areas * potential)
            energy += xp.sum(edge_length_sq * gradient)
        finally:
            if restore_links and old_vector_potential is not None:
                operators.set_link_exponents(old_vector_potential)

        if include_magnetic:
            if vector_potential is None:
                vector_potential = operators.link_exponents
            if vector_potential is None:
                raise ValueError("vector_potential is required for magnetic energy.")
            induction = operators.get_triangle_magnetic_field(vector_potential)
            applied = xp.asarray(self.applied_triangle_field)
            coords = self.device.mesh.sites[self.device.mesh.elements]
            edge_1 = coords[:, 1] - coords[:, 0]
            edge_2 = coords[:, 2] - coords[:, 0]
            triangle_areas = 0.5 * xp.absolute(
                edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
            )
            energy += xp.sum(
                triangle_areas * self.device.kappa**2 * (induction - applied) ** 2
            )
        if average:
            energy /= xp.sum(areas)
        return float(energy)

    def compute_d_plus_d_prime_free_energy(
        self,
        d: np.ndarray,
        d_prime: np.ndarray,
        magnetic_field: Optional[np.ndarray] = None,
    ) -> float:
        """Return the discrete condensate energy for ``DPlusDPrimeModel``.

        ``magnetic_field`` is the local dimensionless induction ``B / Bc2``.
        It defaults to the field currently stored by the solver. The magnetic
        field energy itself is omitted because it is constant for a prescribed
        vector potential.
        """
        if not isinstance(self.model, DPlusDPrimeModel):
            raise TypeError(
                "compute_d_plus_d_prime_free_energy requires DPlusDPrimeModel."
            )
        xp = np if isinstance(d, np.ndarray) else cupy
        model = self.model
        if magnetic_field is None:
            magnetic_field = self.magnetic_field
        if magnetic_field is None:
            raise ValueError("magnetic_field is required for DPlusDPrimeModel.")
        magnetic_field = xp.asarray(magnetic_field)
        abs_sq_d = xp.absolute(d) ** 2
        abs_sq_d_prime = xp.absolute(d_prime) ** 2
        phase_sum = xp.conj(d) * d_prime + d * xp.conj(d_prime)
        chirality = xp.conj(d) * d_prime - d * xp.conj(d_prime)
        potential = (
            -abs_sq_d
            - model.alpha * abs_sq_d_prime
            + 0.5 * (abs_sq_d**2 + abs_sq_d_prime**2)
            + (1 / 3) * abs_sq_d * abs_sq_d_prime
            + (1 / 6) * xp.real(phase_sum**2)
            + xp.real(-1j * model.zeeman_coupling * magnetic_field * chirality)
        )
        lap_d = self.operators.psi_laplacian @ d
        lap_d_prime = self.operators.psi_laplacian @ d_prime
        gradient = -xp.real(xp.conj(d) * lap_d)
        gradient -= xp.real(xp.conj(d_prime) * lap_d_prime)
        return float(xp.sum(self.operators.areas * (potential + gradient)))

    def solve_for_observables(
        self, psi2: np.ndarray, psi1: np.ndarray, dA_dt: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves for the scalar potential :math:`\\mu`, the supercurrent density
        :math:`\\mathbf{J}_s`, and the normal current density :math:`\\mathbf{J}_n`.

        Args:
            psi2: Component 2 (d-wave for ``SPlusDModel``).
            psi1: Component 1 (s-wave for ``SPlusDModel``).
            dA_dt: The time-derivative of the vector potential.

        Returns:
            :math:`\\mu`, :math:`\\mathbf{J}_s`, and :math:`\\mathbf{J}_n`
        """
        use_cupy = self.use_cupy
        xp = cupy if use_cupy else np
        options = self.options
        use_cupy_solver = options.sparse_solver is SparseSolver.CUPY
        operators = self.operators
        # Compute the supercurrent, scalar potential, and normal current
        if isinstance(self.model, SingleBandModel):
            supercurrent = operators.get_supercurrent(psi1)
        elif isinstance(self.model, DPlusDPrimeModel):
            supercurrent = operators.get_s_plus_s_supercurrent(
                psi1,
                psi2,
                k2_over_k1=1.0,
            )
            if self.model.zeeman_coupling:
                chirality = xp.conj(psi1) * psi2 - psi1 * xp.conj(psi2)
                magnetization = xp.real(1j * self.model.zeeman_coupling * chirality)
                supercurrent += operators.get_magnetization_current(magnetization)
            supercurrent /= self.model.em_coupling
        elif isinstance(self.model, SPlusSModel):
            supercurrent = operators.get_s_plus_s_supercurrent(
                psi1,
                psi2,
                k2_over_k1=self.model.k2_over_k1,
                mixed_gradient_k12=self.model.mixed_gradient_k12,
            ) / (self.model.em_coupling * self.model.beta_em)
        else:
            # Store and use the Poisson-normalized transport supercurrent.
            # The unscaled condensate current is J_s in the reference equations.
            supercurrent = (
                operators.get_s_plus_d_supercurrent(
                    psi2,
                    psi1,
                    eta_s=self.model.eta_s,
                    eta_v=self.model.eta_v,
                )
                / self.model.beta_em
            )
        # ``mu_boundary`` stores the imposed normal derivative of phi.  For a
        # time-dependent prescribed A, zero normal total-current flux requires
        # d_n phi = -d_t A . n on insulating boundary edges.
        effective_mu_boundary = self.mu_boundary - self.dA_boundary_normal
        rhs = (operators.divergence @ (supercurrent - dA_dt)) - (
            operators.mu_boundary_laplacian @ effective_mu_boundary
        )
        rhs[operators.mu_reference_index] = 0.0
        if use_cupy and not use_cupy_solver:
            rhs = cupy.asnumpy(rhs)
        if self.options.sparse_solver is SparseSolver.PARDISO:
            mu = pypardiso.spsolve(operators.mu_laplacian, rhs)
        else:
            mu = operators.mu_laplacian_lu(rhs)
        if use_cupy and not use_cupy_solver:
            mu = cupy.asarray(mu)
        normal_current = -(operators.mu_gradient @ mu) - dA_dt
        return mu, supercurrent, normal_current

    def get_induced_vector_potential(
        self,
        current_density: np.ndarray,
        A_induced_vals: List[np.ndarray],
        velocity: List[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """Computes a new value of the induced vector potential based on Polyak's method.

        Args:
            current_density: The total current density :math:`\\mathbf{J}_s + \\mathbf{J}_n`
            A_induced_vals: A running list of the induced vector potential for previous
                iterations of Polyak's method.
            velocity: A running list of the "velocities" for previous iterations of
                Polyak's method.

        Returns:
            A new value for the induced vector potential, and the relative error in the
            induced vector potential between this iteration of Polyak's method and the
            previous iteration.
        """
        xp = self.xp
        use_cupy = self.use_cupy
        options = self.options
        mesh = self.device.mesh
        alpha = options.screening_step_size
        beta = options.screening_step_drag
        # Evaluate the induced vector potential.
        J_site = mesh.get_quantity_on_site(current_density, use_cupy=use_cupy)
        areas = self.areas
        sites = self.sites
        edge_centers = self.edge_centers
        if use_cupy:
            threads_per_block = 512
            num_blocks = math.ceil(self.num_edges / threads_per_block)
            get_A_induced_cupy(
                (num_blocks,),
                (threads_per_block, 2),
                (J_site, areas, sites, edge_centers, self.new_A_induced),
            )
        else:
            get_A_induced_numba(J_site, areas, sites, edge_centers, self.new_A_induced)
        new_A_induced = self.new_A_induced
        # Update induced vector potential using Polyak's method
        A_induced = A_induced_vals[-1]
        dA = new_A_induced - A_induced
        velocity.append((1 - beta) * velocity[-1] + alpha * dA)
        A_induced = A_induced + velocity[-1]
        A_induced_vals.append(A_induced)
        if len(A_induced_vals) > 1:
            numerator = xp.linalg.norm(dA, axis=1)
            denominator = xp.linalg.norm(A_induced, axis=1)
            # Avoid division by zero in the case of zero A_induced
            denominator = xp.maximum(denominator, 1e-20, out=denominator)
            screening_error = float(xp.max(numerator / denominator))
            del velocity[:-2]
            del A_induced_vals[:-2]
        return A_induced, screening_error

    def advance_s_plus_d_vector_potential(
        self,
        psi_d: np.ndarray,
        psi_s: np.ndarray,
        applied_vector_potential: np.ndarray,
        induced_vector_potential: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Advance Goncalves et al. Eq. (19) by one implicit magnetic step.

        The edge scalar is the tangential component of ``A``.  The discrete
        magnetic term is applied to the induced field
        :math:`B-H=\nabla\times A_{\rm induced}`. This formulation is valid for
        nonuniform prescribed applied fields as well as uniform fields. The
        optional homogeneous drive subtracts a transport-current source from
        the condensate current, so the normal state obeys
        :math:`-\partial_t A=J_{\rm drive}/\beta_{\rm em}`.
        The
        linear magnetic diffusion
        term is implicit because the smallest dual edges of an unstructured
        mesh make the paper's explicit Cartesian-grid update prohibitively
        stiff; the condensate current remains evaluated at the accepted TDGL
        state.
        """
        if not isinstance(self.model, SPlusDModel):
            raise TypeError("advance_s_plus_d_vector_potential requires SPlusDModel.")
        return self._advance_local_vector_potential(
            psi_d,
            psi_s,
            induced_vector_potential,
            dt,
        )

    def advance_s_plus_s_vector_potential(
        self,
        psi2: np.ndarray,
        psi1: np.ndarray,
        applied_vector_potential: np.ndarray,
        induced_vector_potential: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Advance the local electromagnetic equation for ``SPlusSModel``.

        The prescribed applied field enters through the boundary curl term of
        the magnetic-diffusion operator, so only the induced field ``B-H`` is
        advanced here. ``applied_vector_potential`` is retained in the public
        signature for symmetry with :meth:`advance_s_plus_d_vector_potential`.
        """
        if not isinstance(self.model, SPlusSModel):
            raise TypeError("advance_s_plus_s_vector_potential requires SPlusSModel.")
        return self._advance_local_vector_potential(
            psi2,
            psi1,
            induced_vector_potential,
            dt,
        )

    def _advance_local_vector_potential(
        self,
        psi2: np.ndarray,
        psi1: np.ndarray,
        induced_vector_potential: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance a multicomponent local Maxwell equation by one implicit step."""
        operators = self.operators
        if isinstance(self.model, SPlusDModel):
            raw_supercurrent = operators.get_s_plus_d_supercurrent(
                psi2,
                psi1,
                eta_s=self.model.eta_s,
                eta_v=self.model.eta_v,
            )
            drive = getattr(self, "s_plus_d_drive_tangent", 0.0)
        elif isinstance(self.model, SPlusSModel):
            raw_supercurrent = operators.get_s_plus_s_supercurrent(
                psi1,
                psi2,
                k2_over_k1=self.model.k2_over_k1,
                mixed_gradient_k12=self.model.mixed_gradient_k12,
            )
            drive = 0.0
        else:
            raise TypeError(
                "Local vector-potential evolution requires SPlusDModel or "
                "SPlusSModel."
            )
        tangent = self.normalized_directions
        induced_tangent = np.einsum("ij,ij->i", induced_vector_potential, tangent)
        rate = dt * self.device.kappa**2 / self.model.beta_em
        if self._s_plus_d_magnetic_dt != dt:
            implicit_operator = sp.eye(self.num_edges, format="csc") + rate * (
                operators.magnetic_diffusion
            )
            self._s_plus_d_magnetic_lu = sp.linalg.factorized(implicit_operator)
            self._s_plus_d_magnetic_dt = dt
        rhs = induced_tangent + (dt / self.model.beta_em) * (raw_supercurrent - drive)
        new_induced_tangent = self._s_plus_d_magnetic_lu(rhs)
        dA_dt = (new_induced_tangent - induced_tangent) / dt
        new_induced = new_induced_tangent[:, None] * tangent
        return new_induced, dA_dt

    def update(
        self,
        state: Dict[str, numbers.Real],
        running_state: RunningState,
        dt: float,
        *,
        psi2: np.ndarray,
        psi1: np.ndarray,
        mu: np.ndarray,
        supercurrent: np.ndarray,
        normal_current: np.ndarray,
        induced_vector_potential: np.ndarray,
        applied_vector_potential: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """This method is called at each time step to update the state of the system.

        Args:
            state: The solver state, i.e., the solve step, time, and time step
            running_state: A container for scalar data that is saved at each time step
            dt: The time step for the previous solve step
            psi2: The d-wave order parameter
            psi1: The s-wave order parameter
            mu: The scalar potential
            supercurrent: The supercurrent density
            normal_current: The normal current density
            induced_vector_potential: The induced vector potential
            applied_vector_potential: The applied vector potential. This will be ``None``
                in the case of a time-independent vector potential.
            epsilon: The disorder parameter ``epsilon``. This will be ``None``
                in the case of a time-independent ``epsilon``.

        Returns:
            A :class:`tdgl.SolverResult` instance for the solve step.
        """
        xp = self.xp
        options = self.options
        operators = self.operators
        local_screening = getattr(
            self, "local_screening", getattr(self, "goncalves_screening", False)
        )

        step = state["step"]
        time = state["time"]
        A_induced = induced_vector_potential
        # Dynamic inputs are evaluated at both ends of the accepted interval.
        # Re-evaluating the left endpoint also makes a post-thermalization time
        # reset deterministic instead of differentiating against stale saved data.
        current_A_applied = (
            self.update_applied_vector_potential(time)
            if self.dynamic_vector_potential
            else self.current_A_applied
        )
        current_epsilon = (
            self.update_epsilon(time) if self.dynamic_epsilon else self.epsilon
        )
        if isinstance(self.model, DPlusDPrimeModel):
            self.magnetic_field = operators.get_magnetic_field(
                current_A_applied + A_induced
            )
        psi2_n = psi2
        psi1_n = psi1
        mu_n = mu
        if local_screening:
            # Local Maxwell evolution is formulated in temporal gauge. A seed
            # produced by a Poisson/prescribed-field solve can carry a nonzero
            # scalar potential, which must not enter even the first local step.
            mu_n = xp.zeros_like(mu_n)
        old_sq_psi2 = xp.absolute(psi2_n) ** 2
        old_sq_psi1 = xp.absolute(psi1_n) ** 2
        dt = min(self.tentative_dt, state.get("_remaining_time", dt))

        def endpoint_inputs(accepted_dt):
            endpoint_time = time + accepted_dt
            next_A = (
                self.update_applied_vector_potential(endpoint_time)
                if self.dynamic_vector_potential
                else current_A_applied
            )
            if self.dynamic_vector_potential:
                dA_vector_dt = (next_A - current_A_applied) / accepted_dt
                applied_dA_dt = xp.einsum(
                    "ij,ij->i", dA_vector_dt, self.normalized_directions
                )
                self.dA_boundary_normal = xp.einsum(
                    "ij,ij->i",
                    dA_vector_dt[self.boundary_edge_indices],
                    self.boundary_normals,
                )
            else:
                applied_dA_dt = 0.0
                self.dA_boundary_normal[...] = 0.0
            next_epsilon = (
                self.update_epsilon(endpoint_time)
                if self.dynamic_epsilon
                else current_epsilon
            )
            self.update_mu_boundary(endpoint_time)
            return next_A, next_epsilon, applied_dA_dt

        if local_screening:
            operators.set_link_exponents(current_A_applied + A_induced)
            psi2, psi1, abs_sq_psi2, abs_sq_psi1, dt = self.adaptive_euler_step(
                step,
                psi2_n,
                psi1_n,
                old_sq_psi2,
                old_sq_psi1,
                mu_n,
                current_epsilon,
                dt,
            )
            next_A_applied, next_epsilon, _ = endpoint_inputs(dt)
            A_induced, dA_dt = self._advance_local_vector_potential(
                psi2, psi1, A_induced, dt
            )
            operators.set_link_exponents(next_A_applied + A_induced)
            # The local bulk algorithms fix phi=0. The Ohmic normal current is
            # therefore -dA/dt; a Poisson solve would mix gauges and double-count
            # the electromagnetic response.
            mu = xp.zeros_like(mu)
            if isinstance(self.model, SPlusDModel):
                raw_supercurrent = operators.get_s_plus_d_supercurrent(
                    psi2,
                    psi1,
                    eta_s=self.model.eta_s,
                    eta_v=self.model.eta_v,
                )
            else:
                raw_supercurrent = operators.get_s_plus_s_supercurrent(
                    psi1,
                    psi2,
                    k2_over_k1=self.model.k2_over_k1,
                    mixed_gradient_k12=self.model.mixed_gradient_k12,
                )
            supercurrent = raw_supercurrent / self.model.beta_em
            normal_current = -dA_dt
            screening_iteration = 1
        else:
            screening_error = np.inf
            A_induced_vals = [A_induced]
            velocity = [0.0]  # Velocity for Polyak's method
            # This loop runs only once if options.include_screening is False.
            for screening_iteration in itertools.count():
                if screening_error < options.screening_tolerance:
                    break
                if screening_iteration >= options.max_iterations_per_step:
                    raise RuntimeError(
                        f"Screening calculation failed to converge at step {step} after"
                        f" {options.max_iterations_per_step} iterations. Relative error in"
                        f" induced vector potential: {screening_error:.2e}"
                        f" (tolerance: {options.screening_tolerance:.2e})."
                    )

                if options.include_screening:
                    operators.set_link_exponents(current_A_applied + A_induced)
                else:
                    operators.set_link_exponents(current_A_applied)

                psi2, psi1, abs_sq_psi2, abs_sq_psi1, dt = self.adaptive_euler_step(
                    step,
                    psi2_n,
                    psi1_n,
                    old_sq_psi2,
                    old_sq_psi1,
                    mu_n,
                    current_epsilon,
                    dt,
                )
                next_A_applied, next_epsilon, dA_dt = endpoint_inputs(dt)
                operators.set_link_exponents(next_A_applied + A_induced)
                mu, supercurrent, normal_current = self.solve_for_observables(
                    psi2, psi1, dA_dt
                )

                if options.include_screening:
                    A_induced, screening_error = self.get_induced_vector_potential(
                        supercurrent + normal_current, A_induced_vals, velocity
                    )
                else:
                    break

        self.current_A_applied = next_A_applied
        self.epsilon = next_epsilon
        operators.set_link_exponents(next_A_applied + A_induced)
        self.applied_triangle_field = operators.get_triangle_magnetic_field(
            next_A_applied
        )
        self.applied_boundary_field = self.applied_triangle_field[
            operators.boundary_triangle_indices
        ]
        if isinstance(self.model, DPlusDPrimeModel):
            self.magnetic_field = operators.get_magnetic_field(
                next_A_applied + A_induced
            )

        running_state.append("dt", dt)
        if self.probe_points is not None:
            # Update the voltage and phase difference
            running_state.append("mu", mu[self.probe_points])
            if isinstance(self.model, SPlusDModel):
                running_state.append("theta", xp.angle(psi2[self.probe_points]))
            else:
                running_state.append("theta", xp.angle(psi1[self.probe_points]))
        if options.include_screening:
            running_state.append("screening_iterations", screening_iteration)

        if options.adaptive:
            if isinstance(self.model, SingleBandModel):
                # Preserve the standard pyTDGL controller: the KWT update has
                # its own rejection criterion, while this outer estimate is
                # allowed to propose dt_max on the next step.
                change = xp.absolute(abs_sq_psi1 - old_sq_psi1).max()
            else:
                # Both condensates participate in a multi-component estimate.
                change_d = xp.absolute(abs_sq_psi2 - old_sq_psi2).max()
                change_s = xp.absolute(abs_sq_psi1 - old_sq_psi1).max()
                change = xp.maximum(change_d, change_s)
            self.d_psi_sq_vals.append(float(change))
            window = options.adaptive_window
            if step > window:
                new_dt = options.dt_init / max(
                    1e-10, np.mean(self.d_psi_sq_vals[-window:])
                )
                self.tentative_dt = np.clip(0.5 * (new_dt + dt), 0, self.dt_max)

        results = [dt, psi2, psi1, mu, supercurrent, normal_current, A_induced]
        if self.dynamic_vector_potential:
            results.append(next_A_applied)
        if self.dynamic_epsilon:
            results.append(next_epsilon)
        return SolverResult(*results)

    def solve(self) -> Optional[Solution]:
        """Runs the solver.

        Returns:
            A :class:`tdgl.Solution` instance. Returns ``None`` if the simulation was
            cancelled during the thermalization stage.
        """
        start_time = datetime.now()
        options = self.options
        options.validate()
        output_file = options.output_file
        seed_solution = self.seed_solution
        num_edges = self.num_edges
        probe_points = self.probe_points

        # Set the initial conditions.
        if self.seed_solution is None:
            parameters = {
                "psi2": self.psi2_init,
                "psi1": self.psi1_init,
                "mu": self.mu_init,
                "supercurrent": np.zeros(num_edges),
                "normal_current": np.zeros(num_edges),
                "induced_vector_potential": np.zeros((num_edges, 2)),
            }
        else:
            if self.seed_solution.device != self.device:
                raise ValueError(
                    "The seed_solution.device must be equal to the device being simulated."
                )
            seed_data = seed_solution.tdgl_data
            seed_induced = seed_data.induced_vector_potential
            if self.local_screening:
                # Continue the total vector potential across a field step.  The
                # new boundary condition then relaxes it toward the new H.
                seed_induced = seed_induced + (
                    seed_data.applied_vector_potential - self.current_A_applied
                )
            parameters = {
                "psi2": getattr(seed_data, "psi2", seed_data.psi),
                "psi1": getattr(seed_data, "psi1", np.zeros_like(seed_data.psi)),
                "mu": seed_data.mu,
                "supercurrent": seed_data.supercurrent,
                "normal_current": seed_data.normal_current,
                "induced_vector_potential": seed_induced,
            }

        fixed_values = []
        fixed_names = []
        if self.dynamic_vector_potential:
            parameters["applied_vector_potential"] = self.current_A_applied
        else:
            fixed_values.append(self.current_A_applied)
            fixed_names.append("applied_vector_potential")
        if self.dynamic_epsilon:
            parameters["epsilon"] = self.epsilon
        else:
            fixed_values.append(self.epsilon)
            fixed_names.append("epsilon")

        if self.use_cupy:
            # Move arrays to the GPU
            for key, val in parameters.items():
                parameters[key] = cupy.asarray(val)
            fixed_values = tuple(cupy.asarray(val) for val in fixed_values)

        running_names_and_sizes = {"dt": 1}
        if probe_points is not None:
            running_names_and_sizes["mu"] = len(probe_points)
            running_names_and_sizes["theta"] = len(probe_points)
        if options.include_screening:
            running_names_and_sizes["screening_iterations"] = 1

        with DataHandler(output_file=output_file, logger=logger) as data_handler:
            data_handler.save_mesh(self.device.mesh)
            if data_handler.tmp_file is not None:
                self.device.to_hdf5(
                    data_handler.tmp_file.create_group("solution/device")
                )
            logger.info(
                f"Simulation started at {start_time}"
                f" using sparse solver {options.sparse_solver.value!r}"
                f" and backend {('CuPy' if self.use_cupy else 'NumPy')!r}."
            )
            runner = Runner(
                function=self.update,
                options=options,
                data_handler=data_handler,
                monitor=options.monitor,
                monitor_update_interval=options.monitor_update_interval,
                initial_values=list(parameters.values()),
                names=list(parameters),
                fixed_values=tuple(fixed_values),
                fixed_names=tuple(fixed_names),
                running_names_and_sizes=running_names_and_sizes,
                logger=logger,
            )
            data_was_generated = runner.run()
            end_time = datetime.now()
            logger.info(f"Simulation ended at {end_time}")
            logger.info(f"Simulation took {end_time - start_time}")

            # Clear the Parameter caches
            if isinstance(self.applied_vector_potential, Parameter):
                self.applied_vector_potential._clear_cache()
            if isinstance(self.disorder_epsilon, Parameter):
                self.disorder_epsilon._clear_cache()

            solution = None
            if data_was_generated:
                solution = Solution(
                    device=self.device,
                    path=data_handler.output_path,
                    options=options,
                    applied_vector_potential=self.applied_vector_potential,
                    terminal_currents=self.terminal_currents,
                    disorder_epsilon=self.disorder_epsilon,
                    total_seconds=(end_time - start_time).total_seconds(),
                )
                solution.to_hdf5()
            return solution
