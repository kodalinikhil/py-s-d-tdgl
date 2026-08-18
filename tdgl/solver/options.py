import math
import numbers
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Union


class SolverOptionsError(ValueError):
    pass


class SparseSolver(Enum):
    """Supported sparse linear solvers."""

    SUPERLU: str = "superlu"
    UMFPACK: str = "umfpack"
    PARDISO: str = "pardiso"
    CUPY: str = "cupy"


@dataclass
class SolverOptions:
    """Options for the TDGL solver.

    Args:
        solve_time: Total simulation time, after any thermalization.
        skip_time: Amount of 'thermalization' time to simulate before recording data.
        dt_init: Initial time step.
        dt_max: Maximum adaptive time step.
        adaptive: Whether to use an adpative time step. Setting ``dt_init = dt_max``
            is equivalent to setting ``adaptive = False``.
        adaptive_window: Number of most recent solve steps to consider when
            computing the time step adaptively.
        max_solve_retries: The maximum number of times to reduce the time step in a
            given solve iteration before giving up.
        adaptive_time_step_multiplier: The factor by which to multiple the time
            step ``dt`` for each adaptive solve retry.
        equilibrium_tolerance: If not ``None``, stop the simulation early when the
            maximum absolute change over ``equilibrium_window`` accepted steps is
            below this value. The order parameters are compared after removing the
            optimal shared global phase rotation. When an induced vector potential
            is part of the state, its change in the solver's fixed temporal gauge is
            included as well. This extends the stationary-state criterion used by
            Gonçalves et al. to prevent electromagnetic relaxation from being
            mistaken for equilibrium. ``solve_time`` remains a hard upper bound.
        equilibrium_window: Number of accepted steps separating the two
            order-parameter configurations used in the equilibrium comparison.
        equilibrium_min_time: Minimum simulation time before equilibrium stopping
            is considered.
        terminal_psi: Fixed value for the order parameter in current terminals.
        output_file: Path to an HDF5 file in which to save the data.
            If the file name already exists, a unique name will be generated.
            If ``output_file`` is ``None``, the solver results will not be saved
            to disk.
        gpu: Use the GPU via CuPy. This option requires a GPU and the
            CuPy Python package, which can be installed via pip.
        sparse_solver: One of ``"superlu"``, ``"umfpack"``, ``"pardiso"``, or ``"cupy"``.
            ``"umfpack"`` requires suitesparse, which can be installed via conda,
            and scikit-umfpack, which can be installed via pip. ``"pardiso"``
            requires an Intel CPU and the pypardiso package, which can be
            installed via pip or conda. ``"cupy"`` requires a GPU and the
            CuPy Python package, which can be installed via pip.
        field_units: The units for magnetic fields.
        current_units: The units for currents.
        pause_on_interrupt: Pause the simulation in the event of a ``KeyboardInterrupt``.
        save_every: Save interval in units of solve steps.
        progress_interval: Minimum number of solve steps between progress bar updates.
        monitor: Plot data in real time as the simulation is running.
        monitor_update_interval: The monitor update interval in seconds.
        include_screening: Whether to include screening in the simulation. For
            ``SPlusDModel`` and ``SPlusSModel`` this advances the model's local
            bulk electromagnetic equation for the induced field ``B-H``. For
            ``SingleBandModel`` it retains pyTDGL's thin-film Biot--Savart screening
            calculation.
        max_iterations_per_step: The maximum number of screening iterations per solve
            step.
        screening_tolerance: Relative tolerance for the induced vector potential, used
            to evaluate convergence of the screening calculation within a single time
            step.
        screening_step_size: Step size :math:`\\alpha` for Polyak's method.
        screening_step_drag: Drag parameter :math:`\\beta` for Polyak's method.
        s_plus_d_drive_current_x: Dimensionless x component of the homogeneous
            bulk transport-current source for ``SPlusDModel`` local screening.
            It is normalized so that the normal state has
            ``E_x = s_plus_d_drive_current_x / beta_em``.
        s_plus_d_drive_current_y: Dimensionless y component of the homogeneous
            bulk transport-current source for ``SPlusDModel`` local screening.
        simulate_d_wave: Deprecated and ignored. Select the equation set with
            ``Layer(model=...)``.
    """

    solve_time: float
    skip_time: float = 0.0
    dt_init: float = 1e-6
    dt_max: float = 1e-1
    adaptive: bool = True
    adaptive_window: int = 10
    max_solve_retries: int = 10
    adaptive_time_step_multiplier: float = 0.25
    equilibrium_tolerance: Union[float, None] = None
    equilibrium_window: int = 1000
    equilibrium_min_time: float = 0.0
    output_file: Union[str, None] = None
    terminal_psi: Union[float, complex, None] = 0.0
    gpu: bool = False
    sparse_solver: Union[SparseSolver, str] = SparseSolver.SUPERLU
    pause_on_interrupt: bool = True
    save_every: int = 100
    progress_interval: int = 0
    monitor: bool = False
    monitor_update_interval: float = 1.0
    field_units: str = "mT"
    current_units: str = "uA"
    include_screening: bool = False
    max_iterations_per_step: int = 1000
    screening_tolerance: float = 1e-3
    screening_step_size: float = 0.1
    screening_step_drag: float = 0.5
    s_plus_d_drive_current_x: float = 0.0
    s_plus_d_drive_current_y: float = 0.0
    simulate_d_wave: Union[bool, None] = None

    def validate(self) -> None:
        if self.simulate_d_wave is not None:
            warnings.warn(
                "SolverOptions.simulate_d_wave is ignored; select SPlusDModel "
                "or SingleBandModel on Layer(model=...).",
                DeprecationWarning,
                stacklevel=2,
            )
        finite_values = {
            "solve_time": self.solve_time,
            "skip_time": self.skip_time,
            "dt_init": self.dt_init,
            "dt_max": self.dt_max,
            "adaptive_time_step_multiplier": self.adaptive_time_step_multiplier,
            "equilibrium_min_time": self.equilibrium_min_time,
            "monitor_update_interval": self.monitor_update_interval,
            "screening_tolerance": self.screening_tolerance,
            "screening_step_size": self.screening_step_size,
            "screening_step_drag": self.screening_step_drag,
            "s_plus_d_drive_current_x": self.s_plus_d_drive_current_x,
            "s_plus_d_drive_current_y": self.s_plus_d_drive_current_y,
        }
        if self.equilibrium_tolerance is not None:
            finite_values["equilibrium_tolerance"] = self.equilibrium_tolerance
        for name, value in finite_values.items():
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise SolverOptionsError(f"{name} must be a real number.")
            if not math.isfinite(value):
                raise SolverOptionsError(f"{name} must be finite.")

        if self.solve_time <= 0:
            raise SolverOptionsError("solve_time must be positive.")
        if self.skip_time < 0:
            raise SolverOptionsError("skip_time must be nonnegative.")
        if self.dt_init <= 0 or self.dt_max <= 0:
            raise SolverOptionsError("dt_init and dt_max must be positive.")
        if self.dt_init > self.dt_max:
            raise SolverOptionsError("dt_init must be less than or equal to dt_max.")

        if self.terminal_psi is not None:
            try:
                terminal_abs = float(abs(self.terminal_psi))
            except (TypeError, ValueError) as exc:
                raise SolverOptionsError(
                    "terminal_psi must be None or a finite scalar."
                ) from exc
            if not math.isfinite(terminal_abs) or not (0 <= terminal_abs <= 1):
                raise SolverOptionsError(
                    "terminal_psi must be None or have finite absolute value in [0, 1]"
                    f" (got {self.terminal_psi})."
                )

        if not (0 < self.adaptive_time_step_multiplier < 1):
            raise SolverOptionsError(
                "adaptive_time_step_multiplier must be in (0, 1)"
                f" (got {self.adaptive_time_step_multiplier})."
            )

        if self.equilibrium_tolerance is not None and self.equilibrium_tolerance <= 0:
            raise SolverOptionsError("equilibrium_tolerance must be positive or None.")

        if (
            not isinstance(self.equilibrium_window, numbers.Integral)
            or isinstance(self.equilibrium_window, bool)
            or self.equilibrium_window <= 0
        ):
            raise SolverOptionsError("equilibrium_window must be a positive integer.")
        self.equilibrium_window = int(self.equilibrium_window)

        if self.equilibrium_min_time < 0:
            raise SolverOptionsError("equilibrium_min_time must be nonnegative.")

        positive_integer_options = {
            "adaptive_window": self.adaptive_window,
            "save_every": self.save_every,
            "max_iterations_per_step": self.max_iterations_per_step,
        }
        for name, value in positive_integer_options.items():
            if (
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise SolverOptionsError(f"{name} must be a positive integer.")
            setattr(self, name, int(value))

        nonnegative_integer_options = {
            "max_solve_retries": self.max_solve_retries,
            "progress_interval": self.progress_interval,
        }
        for name, value in nonnegative_integer_options.items():
            if (
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value < 0
            ):
                raise SolverOptionsError(f"{name} must be a nonnegative integer.")
            setattr(self, name, int(value))

        if self.monitor_update_interval <= 0:
            raise SolverOptionsError("monitor_update_interval must be positive.")

        if not (0 < self.screening_step_drag <= 1):
            raise SolverOptionsError(
                "screening_step_drag must be in (0, 1]"
                f" (got {self.screening_step_drag})."
            )

        if self.screening_step_size <= 0:
            raise SolverOptionsError(
                "screening_step_size must be in > 0"
                f" (got {self.screening_step_size})."
            )

        if self.screening_tolerance <= 0:
            raise SolverOptionsError(
                "screening_tolerance must be in > 0"
                f" (got {self.screening_tolerance})."
            )

        if self.gpu:
            try:
                import cupy  # type: ignore # noqa: F401
            except ImportError:
                raise SolverOptionsError(
                    "GPU option requires a GPU and the CuPy Python package."
                )

        solver = self.sparse_solver
        if isinstance(solver, str):
            try:
                solver = SparseSolver[solver.upper()]
            except KeyError:
                valid_solvers = list(SparseSolver.__members__.keys())
                if solver not in valid_solvers:
                    raise SolverOptionsError(
                        f"sparse solver must be one of {valid_solvers!r}, got {solver}."
                    )
            self.sparse_solver = solver

        if self.sparse_solver is SparseSolver.UMFPACK:
            try:
                from scikits import umfpack  # type: ignore # noqa: F401
            except ImportError:
                raise SolverOptionsError(
                    "SparseSolver.UMFPACK requires suitesparse and the"
                    " scikit-umfpack Python package."
                )
        if self.sparse_solver is SparseSolver.PARDISO:
            try:
                import pypardiso  # type: ignore # noqa: F401
            except ImportError:
                raise SolverOptionsError(
                    "SparseSolver.PARDISO requires an Intel CPU"
                    " and the pypardiso Python package."
                )
        if self.sparse_solver is SparseSolver.CUPY:
            if not self.gpu:
                raise SolverOptionsError(
                    "SparseSolver.CUPY requires SolverOptions.gpu = True,"
                    " and therefore requires a GPU and the CuPy Python package."
                )
