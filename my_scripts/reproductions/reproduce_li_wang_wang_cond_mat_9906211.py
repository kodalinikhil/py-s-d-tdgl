"""Reproduce Li, Wang, and Wang, cond-mat/9906211, with s+d TDGL.

The paper is "Vortex dynamics of a d+is-wave superconductor" (1999).  This
script implements the three requested dynamical protocols:

2. Gibbs free-energy branches from stationary dynamical relaxations.
3. Free-flux-flow resistivity versus field and order-parameter relaxation rate.
4. Twin-boundary depinning curves and the paper's square-root fits.

The original calculation used magnetic-periodic one/two-vortex cells.  The
current framework has open variational boundaries, so Figures 2--4 are
finite-domain analogues rather than bit-for-bit reproductions.  The script
retains the paper's one-flux cell area, coefficient normalization, dissipative
dynamics, bulk drive, and time-averaged electric-field observables.  Every run
records requested H and measured mean B separately.  Figure 3 retains the
``q=1`` and ``q=10`` curves; the prohibitively expensive ``q=0.01`` branch is
intentionally omitted.

Examples
--------
Run a small end-to-end check of Figure 3::

    python my_scripts/reproductions/reproduce_li_wang_wang_cond_mat_9906211.py \
        --figures 3 --preset smoke

Run the four-panel Figure 3 protocol using the Ref. 17 defaults::

    python my_scripts/reproductions/reproduce_li_wang_wang_cond_mat_9906211.py \
        --figures 3 --preset paper

Run all protocols (expensive)::

    python my_scripts/reproductions/reproduce_li_wang_wang_cond_mat_9906211.py \
        --figures all --preset paper

Existing checkpoints and CSV files are reused.  Use ``--plot-only`` to redraw
figures without launching simulations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

import tdgl
from tdgl.finite_volume.operators import build_triangle_magnetic_field_curl


PAPER_ID = "cond-mat/9906211"
PAPER_TITLE = "Vortex dynamics of a d+is-wave superconductor"
RUN_SCHEMA_VERSION = 5
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results/reproductions/li_wang_wang_1999"
DEFAULT_FIG3_CURRENT = 0.1  # Inferred from the numerical-method Ref. 17.
DEFAULT_TWIN_SPACING = 10.8


@dataclass(frozen=True)
class ReproductionPreset:
    """Resolution and sampling controls for a reproduction run."""

    grid_points: int
    dt: float
    equilibrium_time: float
    equilibrium_tolerance: float
    equilibrium_window: int
    drive_skip_time: float
    drive_measure_time: float
    save_every: int
    fig2_fields: tuple[float, ...]
    fig3_alphas: tuple[float, ...]
    fig3_relaxations: tuple[float, ...]
    fig3_fields: tuple[float, ...]
    fig4_currents: tuple[float, ...]


PRESETS = {
    "smoke": ReproductionPreset(
        grid_points=7,
        dt=2e-3,
        equilibrium_time=2e-2,
        equilibrium_tolerance=1e-2,
        equilibrium_window=2,
        drive_skip_time=4e-3,
        drive_measure_time=1.2e-2,
        save_every=2,
        fig2_fields=(0.0, 0.2, 1.0, 1.4),
        fig3_alphas=(0.85,),
        fig3_relaxations=(1.0,),
        fig3_fields=(0.2, 0.8, 1.2),
        fig4_currents=(0.12, 0.18, 0.24),
    ),
    "quick": ReproductionPreset(
        grid_points=13,
        dt=2e-3,
        equilibrium_time=20.0,
        equilibrium_tolerance=1e-4,
        equilibrium_window=250,
        drive_skip_time=5.0,
        drive_measure_time=10.0,
        save_every=100,
        fig2_fields=tuple(np.linspace(0.0, 1.45, 16)),
        fig3_alphas=(0.97, 0.85, 0.67, -1.0),
        fig3_relaxations=(1.0, 10.0),
        fig3_fields=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2),
        fig4_currents=tuple(np.linspace(0.10, 0.28, 10)),
    ),
    "paper": ReproductionPreset(
        # Ref. 17 used a 19-by-19 square grid.  The open-boundary analogue
        # keeps that count while jittering interior vertices for a valid dual mesh.
        grid_points=19,
        dt=1e-3,
        equilibrium_time=100.0,
        equilibrium_tolerance=1e-5,
        equilibrium_window=1000,
        drive_skip_time=50.0,
        drive_measure_time=100.0,
        save_every=500,
        fig2_fields=tuple(np.linspace(0.0, 1.45, 30)),
        fig3_alphas=(0.97, 0.85, 0.67, -1.0),
        fig3_relaxations=(1.0, 10.0),
        fig3_fields=tuple(np.linspace(0.02, 1.20, 25)),
        fig4_currents=tuple(np.linspace(0.10, 0.28, 19)),
    ),
}


def hc2_over_b0(alpha_s: float) -> float:
    """Return mixed-state Hc2/B0 needed to convert Figure 3's field axis.

    The coefficients are the determinant of the paper's linearized Eq. (5) in
    its stated ``(d_0, s_2, d_4)`` basis.
    """
    coefficients = np.array(
        [60.0, -86.0 - 18.0 * alpha_s, 10.0 + 20.0 * alpha_s, -2.0 * alpha_s]
    )
    roots = np.roots(coefficients)
    physical = [
        float(root.real) for root in roots if abs(root.imag) < 1e-9 and root.real > 0
    ]
    if not physical:
        raise RuntimeError(f"No positive real Hc2 root for alpha_s={alpha_s:g}")
    return max(physical)


def b_over_b0_from_bc2_fraction(alpha_s: float, fraction: float) -> float:
    """Convert the paper's B/Bc2 axis to the framework's B/B0 units."""
    if fraction < 0:
        raise ValueError("B/Bc2 must be nonnegative")
    return fraction * hc2_over_b0(alpha_s)


def paper_model(
    alpha_s: float,
    relaxation_q: float = 1.0,
    *,
    u_s: float = 0.0,
    u_d: float = 0.0,
) -> tdgl.SPlusDModel:
    """Map Li et al. Eqs. (1)--(3) to :class:`tdgl.SPlusDModel`.

    The paper defines ``q = eta_s = 2 eta_d``. Goncalves et al. use the
    d-band stiffness to define the implicit unit diffusion clock. Rescaling
    Li's time by ``eta_d=q/2`` therefore leaves the d-sector coefficient at
    one, requires the extra relative s-sector kinetic multiplier two, and
    gives ``beta_em=2/q``. Multiplying the solver's raw ``-dA/dt`` by
    ``beta_em`` recovers the paper-clock electric field.
    """
    if relaxation_q <= 0:
        raise ValueError("relaxation_q must be positive")
    if u_s < 0 or u_d < 0:
        raise ValueError("twin strengths must be nonnegative")
    relaxation_s = 2.0
    beta_em = 2.0 / relaxation_q

    if u_s and not u_d:
        raise ValueError("u_d must be nonzero when u_s is nonzero")
    nu_disorder_coupling = 0.0 if not u_d else 2.0 * u_s / u_d
    return tdgl.SPlusDModel(
        eta_s=2.0,
        eta_v=-1.0,
        nu=2.0 * alpha_s,
        tau1=8 / 3,
        tau3=16 / 3,
        tau4=4 / 3,
        beta_em=beta_em,
        relaxation_s=relaxation_s,
        nu_disorder_coupling=nu_disorder_coupling,
    )


def uniform_li_state(alpha_s: float, chirality: int = 1) -> tuple[complex, complex]:
    """Return the homogeneous ``(d, s)`` seed in the paper normalization."""
    if chirality not in (-1, 1):
        raise ValueError("chirality must be +1 or -1")
    if 2 / 3 < alpha_s <= 1:
        d_amplitude = math.sqrt(3 * (1 - alpha_s))
        s_amplitude = math.sqrt((9 * alpha_s - 6) / 4)
        return complex(d_amplitude), complex(0, chirality * s_amplitude)
    return 1.0 + 0j, 0.0 + 0j


def one_flux_cell_side(reduced_induction: float, num_flux: int = 1) -> float:
    """Square-cell side in xi for ``num_flux`` flux quanta at B/B0."""
    if reduced_induction <= 0:
        raise ValueError("reduced_induction must be positive")
    if num_flux <= 0:
        raise ValueError("num_flux must be positive")
    return math.sqrt(2 * math.pi * num_flux / reduced_induction)


def make_square_device(
    model: tdgl.SPlusDModel,
    *,
    side_length: float,
    grid_points: int,
    kappa: float,
    mesh_seed: int,
) -> tdgl.Device:
    """Build a deterministic, D4-symmetric, nearly Cartesian open mesh."""
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=kappa,
        thickness=0.1,
        model=model,
    )
    film = tdgl.Polygon(
        "film", points=tdgl.geometry.box(side_length, side_length)
    ).resample(max(4 * grid_points, 40))
    device = tdgl.Device("li_wang_wang_square", layer=layer, film=film)

    coordinates = np.linspace(-side_length / 2, side_length / 2, grid_points)
    x, y = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((x.ravel(), y.ravel()))
    interior = (np.abs(points[:, 0]) < side_length / 2) & (
        np.abs(points[:, 1]) < side_length / 2
    )
    spacing = side_length / (grid_points - 1)
    # A generic D4-equivariant warp removes Cartesian cocircularities without
    # introducing a randomly asymmetric mesh bias.
    rng = np.random.default_rng(mesh_seed)
    a, b, c, d = rng.uniform(-1.0, 1.0, size=4)
    x_normalized = 2 * points[:, 0] / side_length
    y_normalized = 2 * points[:, 1] / side_length
    displacement_x = np.sin(np.pi * x_normalized) * (
        a
        + b * np.cos(np.pi * y_normalized)
        + c * np.cos(2 * np.pi * x_normalized)
        + d * np.cos(2 * np.pi * y_normalized)
    )
    displacement_y = np.sin(np.pi * y_normalized) * (
        a
        + b * np.cos(np.pi * x_normalized)
        + c * np.cos(2 * np.pi * y_normalized)
        + d * np.cos(2 * np.pi * x_normalized)
    )
    displacement = np.column_stack((displacement_x, displacement_y))
    max_displacement = np.max(np.linalg.norm(displacement[interior], axis=1))
    if max_displacement:
        displacement *= 0.03 * spacing / max_displacement
    points[interior] += displacement[interior]
    triangles = Delaunay(points).simplices
    device._create_dimensionless_mesh(points, triangles)
    return device


def li_vortex_seed(
    sites: np.ndarray,
    alpha_s: float,
    *,
    chirality: int,
    num_vortices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return homogeneous Li amplitudes with common unit-vortex windings."""
    if num_vortices < 0:
        raise ValueError("num_vortices must be nonnegative")
    d_bulk, s_bulk = uniform_li_state(alpha_s, chirality=chirality)
    phase = np.ones(len(sites), dtype=complex)
    core = np.ones(len(sites), dtype=float)
    if num_vortices:
        extent = np.ptp(sites, axis=0)
        if num_vortices == 1:
            centers = np.array([[0.0, 0.0]])
        else:
            angles = np.linspace(0, 2 * np.pi, num_vortices, endpoint=False)
            radius = 0.12 * min(extent)
            centers = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        for center in centers:
            displacement = sites - center
            radius = np.linalg.norm(displacement, axis=1)
            safe_radius = np.maximum(radius, np.finfo(float).eps)
            phase *= (displacement[:, 0] + 1j * displacement[:, 1]) / safe_radius
            core *= np.tanh(radius / math.sqrt(2))
    return d_bulk * core * phase, s_bulk * core * phase


def net_vortex_count(mesh, order_parameter: np.ndarray) -> int:
    """Count signed phase singularities from wrapped triangle windings."""
    psi = np.asarray(order_parameter)
    phases = np.angle(psi[mesh.elements])
    wrapped = np.angle(np.exp(1j * (np.roll(phases, -1, axis=1) - phases)))
    winding = np.rint(np.sum(wrapped, axis=1) / (2 * np.pi)).astype(int)
    coordinates = mesh.sites[mesh.elements]
    edge_1 = coordinates[:, 1] - coordinates[:, 0]
    edge_2 = coordinates[:, 2] - coordinates[:, 0]
    orientation = np.sign(
        edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    ).astype(int)
    return int(np.sum(orientation * winding))


def solution_vortex_count(solution: tdgl.Solution) -> int:
    """Return the net d-component vortex count in the final saved state."""
    return net_vortex_count(solution.device.mesh, solution.tdgl_data.psi2)


def check_vortex_retention(
    solution: tdgl.Solution,
    *,
    expected: int,
    strict: bool,
    label: str,
) -> int:
    """Warn, or fail in strict mode, when an open cell loses its vortex."""
    count = solution_vortex_count(solution)
    if count != expected:
        message = (
            f"{label}: expected {expected} vortex/vortices but found {count}. "
            "The open-boundary cell is not a fixed-flux replacement for the "
            "paper's magnetic-periodic cell."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return count


def rebind_seed_to_device(
    solution: tdgl.Solution,
    device: tdgl.Device,
) -> tdgl.Solution:
    """Reuse identical equilibrium arrays with a dynamics-only model clock."""
    rebound = copy.copy(solution)
    rebound.device = device
    return rebound


def twin_disorder_profile(
    *,
    u_d: float,
    width: float,
    spacing: float = DEFAULT_TWIN_SPACING,
) -> Callable:
    """Return a unit-area top-hat regularization of the paper's delta twins."""
    if u_d < 0 or width <= 0 or spacing <= 0:
        raise ValueError("u_d must be nonnegative; width and spacing must be positive")

    def disorder(r, *, vectorized=True):
        del vectorized
        points = np.atleast_2d(r)
        y = points[:, 1]
        distance = np.abs((y + spacing / 2) % spacing - spacing / 2)
        delta_width = (distance <= width / 2).astype(float) / width
        return 1.0 - u_d * delta_width

    return disorder


def reduced_field_source(device: tdgl.Device, reduced_field: float):
    """Return a ConstantField for H/B0, where B0 is the pure-d Bc2 scale."""
    field_units = "mT"
    dimensional = reduced_field * device.Bc2.to(field_units).magnitude
    return tdgl.sources.ConstantField(
        dimensional,
        field_units=field_units,
        length_units=device.length_units,
    )


def _slug(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace("/", "_").replace(" ", "_")
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def checkpoint_path(root: Path, figure: str, *parts: float | str) -> Path:
    tag = "__".join(_slug(part) for part in parts)
    path = root / "checkpoints" / f"figure_{figure}" / f"{tag}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_checkpoint(
    solution: tdgl.Solution,
    device: tdgl.Device,
    options: tdgl.SolverOptions,
) -> None:
    """Reject a stale, mismatched, or incomplete checkpoint."""
    saved_mesh = solution.device.mesh
    expected_mesh = device.mesh
    same_mesh = (
        saved_mesh.sites.shape == expected_mesh.sites.shape
        and saved_mesh.elements.shape == expected_mesh.elements.shape
        and np.allclose(saved_mesh.sites, expected_mesh.sites, rtol=0, atol=1e-13)
        and np.array_equal(saved_mesh.elements, expected_mesh.elements)
    )
    if not same_mesh or solution.device.layer.model != device.layer.model:
        raise RuntimeError(
            f"Checkpoint {solution.path} does not match the requested mesh/model."
        )
    for name in ("solve_time", "skip_time", "dt_init", "dt_max"):
        if not math.isclose(
            float(getattr(solution.options, name)),
            float(getattr(options, name)),
            rel_tol=1e-12,
            abs_tol=1e-14,
        ):
            raise RuntimeError(
                f"Checkpoint {solution.path} has incompatible option {name}."
            )
    state = solution.tdgl_data.state
    final_time = float(state.get("time", 0.0))
    reached = bool(state.get("equilibrium_reached", False))
    tolerance = max(1e-12, 2 * float(state.get("dt", options.dt_init)))
    if not reached and final_time + tolerance < options.solve_time:
        raise RuntimeError(
            f"Checkpoint {solution.path} is incomplete: t={final_time:g}, "
            f"expected {options.solve_time:g}."
        )


def state_diagnostics(solution: tdgl.Solution, prefix: str) -> dict:
    """Return compact completion/equilibrium metadata for a CSV row."""
    state = solution.tdgl_data.state
    result = {
        f"{prefix}_final_time": float(state.get("time", float("nan"))),
        f"{prefix}_final_step": int(state.get("step", -1)),
    }
    if "equilibrium_reached" in state:
        result[f"{prefix}_equilibrium_reached"] = bool(state["equilibrium_reached"])
        result[f"{prefix}_equilibrium_error"] = float(
            state.get("equilibrium_error", float("nan"))
        )
        result[f"{prefix}_order_parameter_error"] = float(
            state.get("equilibrium_order_parameter_error", float("nan"))
        )
        result[f"{prefix}_electromagnetic_error"] = float(
            state.get("equilibrium_electromagnetic_error", float("nan"))
        )
    return result


def warn_if_unconverged(
    solution: tdgl.Solution,
    options: tdgl.SolverOptions,
) -> None:
    state = solution.tdgl_data.state
    if options.equilibrium_tolerance is not None and not bool(
        state.get("equilibrium_reached", False)
    ):
        warnings.warn(
            f"{solution.path} reached its time cap without satisfying the "
            "coupled order-parameter/electromagnetic equilibrium criterion.",
            RuntimeWarning,
            stacklevel=2,
        )


def equilibrium_options(
    preset: ReproductionPreset,
    output_file: Path,
    model: tdgl.SPlusDModel,
) -> tdgl.SolverOptions:
    clock_scale = model.beta_em
    solve_time = clock_scale * preset.equilibrium_time
    save_every = max(1, math.ceil(clock_scale * preset.save_every))
    return tdgl.SolverOptions(
        solve_time=solve_time,
        dt_init=preset.dt,
        dt_max=preset.dt,
        adaptive=False,
        include_screening=True,
        equilibrium_tolerance=preset.equilibrium_tolerance,
        equilibrium_window=max(1, math.ceil(clock_scale * preset.equilibrium_window)),
        equilibrium_min_time=0.5 * solve_time,
        terminal_psi=None,
        save_every=save_every,
        progress_interval=max(1, 10 * save_every),
        field_units="mT",
        output_file=str(output_file),
    )


def drive_options(
    preset: ReproductionPreset,
    output_file: Path,
    current: Sequence[float],
    model: tdgl.SPlusDModel,
) -> tdgl.SolverOptions:
    current = np.asarray(current, dtype=float)
    if current.shape != (2,):
        raise ValueError("current must have shape (2,)")
    clock_scale = model.beta_em
    save_every = max(1, math.ceil(clock_scale * preset.save_every))
    return tdgl.SolverOptions(
        solve_time=clock_scale * preset.drive_measure_time,
        skip_time=clock_scale * preset.drive_skip_time,
        dt_init=preset.dt,
        dt_max=preset.dt,
        adaptive=False,
        include_screening=True,
        equilibrium_tolerance=None,
        terminal_psi=None,
        save_every=save_every,
        progress_interval=max(1, 10 * save_every),
        field_units="mT",
        output_file=str(output_file),
        s_plus_d_drive_current_x=float(current[0]),
        s_plus_d_drive_current_y=float(current[1]),
    )


def solve_or_load(
    device: tdgl.Device,
    options: tdgl.SolverOptions,
    applied_vector_potential,
    *,
    alpha_s: float,
    chirality: int,
    disorder_epsilon=1.0,
    seed_solution: tdgl.Solution | None = None,
    num_vortices: int = 0,
    plot_only: bool = False,
) -> tdgl.Solution:
    """Load an existing checkpoint or run one TDGL stage."""
    path = Path(options.output_file)
    if path.exists():
        print(f"Reusing {path}")
        solution = tdgl.Solution.from_hdf5(str(path))
        validate_checkpoint(solution, device, options)
        warn_if_unconverged(solution, options)
        return solution
    if plot_only:
        raise FileNotFoundError(f"Missing checkpoint in --plot-only mode: {path}")

    solver = tdgl.TDGLSolver(
        device=device,
        options=options,
        applied_vector_potential=applied_vector_potential,
        disorder_epsilon=disorder_epsilon,
        seed_solution=seed_solution,
    )
    if seed_solution is None:
        d_seed, s_seed = li_vortex_seed(
            solver.device.mesh.sites,
            alpha_s,
            chirality=chirality,
            num_vortices=num_vortices,
        )
        solver.psi2_init[:] = d_seed
        solver.psi1_init[:] = s_seed
    print(f"Running {path}")
    solution = solver.solve()
    if solution is None:
        raise RuntimeError(f"Simulation was cancelled before producing {path}")
    warn_if_unconverged(solution, options)
    return solution


def triangle_areas(mesh) -> np.ndarray:
    coords = mesh.sites[mesh.elements]
    edge_1 = coords[:, 1] - coords[:, 0]
    edge_2 = coords[:, 2] - coords[:, 0]
    return 0.5 * np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0])


def mean_induction(solution: tdgl.Solution) -> float:
    """Area-average the final total B/B0 over mesh triangles."""
    mesh = solution.device.mesh
    curl_x, curl_y = build_triangle_magnetic_field_curl(mesh)
    data = solution.tdgl_data
    vector_potential = np.asarray(data.applied_vector_potential) + np.asarray(
        data.induced_vector_potential
    )
    field = curl_x @ vector_potential[:, 0] + curl_y @ vector_potential[:, 1]
    return float(np.average(np.asarray(field), weights=triangle_areas(mesh)))


def induction_statistics(solution: tdgl.Solution) -> tuple[float, float]:
    """Return the temporal mean/std of spatially averaged B/B0."""
    mesh = solution.device.mesh
    curl_x, curl_y = build_triangle_magnetic_field_curl(mesh)
    applied = np.asarray(solution.tdgl_data.applied_vector_potential)
    weights = triangle_areas(mesh)
    values = []
    with h5py.File(solution.path, "r") as h5file:
        for key in sorted((int(key) for key in h5file["data"]), key=int):
            induced = np.asarray(h5file[f"data/{key}/induced_vector_potential"])
            vector_potential = applied + induced
            field = curl_x @ vector_potential[:, 0] + curl_y @ vector_potential[:, 1]
            values.append(float(np.average(field, weights=weights)))
    return float(np.mean(values)), float(np.std(values))


def global_vector_from_tangential(mesh, edge_values: np.ndarray) -> np.ndarray:
    """Reconstruct on each triangle, then return the spatial area average.

    This deliberately avoids ``Mesh.get_quantity_on_site()``, whose vector
    path contains pyTDGL's legacy factor-of-two current normalization.
    """
    values = np.asarray(edge_values, dtype=float)
    tangents = np.asarray(mesh.edge_mesh.normalized_directions)
    if values.shape != (len(tangents),):
        raise ValueError(f"Expected edge values with shape ({len(tangents)},)")
    edge_lookup = {
        tuple(sorted((int(start), int(end)))): index
        for index, (start, end) in enumerate(mesh.edge_mesh.edges)
    }
    triangle_vectors = []
    for triangle in mesh.elements:
        pairs = (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
        indices = np.array(
            [edge_lookup[tuple(sorted((int(i), int(j))))] for i, j in pairs]
        )
        local_tangents = tangents[indices]
        vector, *_ = np.linalg.lstsq(local_tangents, values[indices], rcond=None)
        triangle_vectors.append(vector)
    return np.average(
        np.asarray(triangle_vectors), axis=0, weights=triangle_areas(mesh)
    )


def time_averaged_raw_electric_field(solution: tdgl.Solution) -> np.ndarray:
    """Return solver-clock ``<-dA/dt>`` from the measurement endpoints."""
    mesh = solution.device.mesh
    with h5py.File(solution.path, "r") as h5file:
        keys = sorted((int(key) for key in h5file["data"]), key=int)
        first = h5file[f"data/{keys[0]}"]
        last = h5file[f"data/{keys[-1]}"]
        first_time = float(first.attrs["time"])
        last_time = float(last.attrs["time"])
        if last_time > first_time:
            first_a = np.asarray(first["induced_vector_potential"])
            last_a = np.asarray(last["induced_vector_potential"])
            dA_dt = (last_a - first_a) / (last_time - first_time)
            tangent = mesh.edge_mesh.normalized_directions
            edge_electric = -np.einsum("ij,ij->i", dA_dt, tangent)
        else:
            edge_electric = np.asarray(last["normal_current"])
    return global_vector_from_tangential(mesh, edge_electric)


def paper_clock_electric_block_delta(solution: tdgl.Solution) -> np.ndarray:
    """Half the signed first/second-half E difference."""
    mesh = solution.device.mesh
    with h5py.File(solution.path, "r") as h5file:
        keys = sorted((int(key) for key in h5file["data"]), key=int)
        if len(keys) < 3:
            return np.full(2, np.nan)

        def interval_electric(first_key: int, last_key: int) -> np.ndarray:
            first = h5file[f"data/{first_key}"]
            last = h5file[f"data/{last_key}"]
            duration = float(last.attrs["time"] - first.attrs["time"])
            if duration <= 0:
                return np.full(2, np.nan)
            first_a = np.asarray(first["induced_vector_potential"])
            last_a = np.asarray(last["induced_vector_potential"])
            dA_dt = (last_a - first_a) / duration
            edge_electric = -np.einsum(
                "ij,ij->i", dA_dt, mesh.edge_mesh.normalized_directions
            )
            return global_vector_from_tangential(mesh, edge_electric)

        midpoint = len(keys) // 2
        first_half = interval_electric(keys[0], keys[midpoint])
        second_half = interval_electric(keys[midpoint], keys[-1])
    beta_em = solution.device.layer.model.beta_em
    return 0.5 * beta_em * (second_half - first_half)


def paper_clock_electric_field(solution: tdgl.Solution) -> np.ndarray:
    """Convert raw solver-clock E to the paper clock used for rho/rho_n."""
    model = solution.device.layer.model
    return model.beta_em * time_averaged_raw_electric_field(solution)


def bulk_condensate_density(solution: tdgl.Solution, fraction: float = 0.7) -> float:
    """Area-average |d|^2+|s|^2 in a central square, excluding surface modes."""
    mesh = solution.device.mesh
    half_extent = np.max(np.abs(mesh.sites), axis=0) * fraction
    mask = (np.abs(mesh.sites[:, 0]) <= half_extent[0]) & (
        np.abs(mesh.sites[:, 1]) <= half_extent[1]
    )
    data = solution.tdgl_data
    density = np.abs(data.psi2) ** 2 + np.abs(data.psi1) ** 2
    return float(np.average(density[mask], weights=mesh.areas[mask]))


def discrete_gibbs_free_energy(solution: tdgl.Solution) -> float:
    """Evaluate the framework's full s+d Gibbs diagnostic on a saved state."""
    options = tdgl.SolverOptions(
        solve_time=1e-6,
        dt_init=1e-6,
        dt_max=1e-6,
        adaptive=False,
        include_screening=True,
        terminal_psi=None,
        field_units=solution.options.field_units,
    )
    solver = tdgl.TDGLSolver(
        device=solution.device,
        options=options,
        applied_vector_potential=solution.applied_vector_potential,
        # The saved sitewise epsilon is installed immediately below. Passing
        # the deserialized callable here is unsafe for legacy checkpoints,
        # whose vectorization metadata may not survive round-tripping.
        disorder_epsilon=1.0,
    )
    data = solution.tdgl_data
    total_a = np.asarray(data.applied_vector_potential) + np.asarray(
        data.induced_vector_potential
    )
    solver.epsilon = np.asarray(data.epsilon)
    return solver.compute_s_plus_d_free_energy(
        np.asarray(data.psi2),
        np.asarray(data.psi1),
        vector_potential=total_a,
        include_magnetic=True,
        average=True,
    )


def uniform_condensation_energy(alpha_s: float) -> float:
    """Uniform zero-field energy density in the framework normalization."""
    d, s = uniform_li_state(alpha_s)
    model = paper_model(alpha_s)
    d2, s2 = abs(d) ** 2, abs(s) ** 2
    return float(
        -d2
        - model.nu * s2
        + 0.5 * d2**2
        + 0.5 * model.tau1 * s2**2
        + 0.5 * model.tau3 * d2 * s2
        + model.tau4 * np.real(np.conj(s) ** 2 * d**2)
    )


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run_figure_2(args, preset: ReproductionPreset, root: Path) -> Path:
    csv_path = root / "figure_2_gibbs.csv"
    alpha_s = args.fig2_alpha
    kappa = figure_kappa(args, "2")
    if args.plot_only:
        rows = read_csv(csv_path)
    else:
        rows = []
        for field in preset.fig2_fields:
            side = one_flux_cell_side(field) if field > 0 else DEFAULT_TWIN_SPACING
            model = paper_model(alpha_s, 1.0)
            device = make_square_device(
                model,
                side_length=side,
                grid_points=preset.grid_points,
                kappa=kappa,
                mesh_seed=args.mesh_seed,
            )
            path = checkpoint_path(root, "2", "equilibrium", alpha_s, field)
            solution = solve_or_load(
                device,
                equilibrium_options(preset, path, model),
                reduced_field_source(device, field),
                alpha_s=alpha_s,
                chirality=args.chirality,
                num_vortices=int(field > 0),
                plot_only=args.plot_only,
            )
            vortex_count = check_vortex_retention(
                solution,
                expected=int(field > 0),
                strict=args.strict_vortex,
                label=f"Figure 2 H/B0={field:g}",
            )
            rows.append(
                {
                    "alpha_s": alpha_s,
                    "requested_H_over_B0": field,
                    "measured_B_over_B0": mean_induction(solution),
                    "d_vortices": vortex_count,
                    "gibbs_over_G0": discrete_gibbs_free_energy(solution),
                    "bulk_density": bulk_condensate_density(solution),
                    "kappa": kappa,
                    "checkpoint": solution.path,
                    **state_diagnostics(solution, "equilibrium"),
                }
            )
        write_csv(csv_path, rows)

    h = np.linspace(0, max(float(row["requested_H_over_B0"]) for row in rows), 400)
    meissner = uniform_condensation_energy(alpha_s) + kappa**2 * h**2
    x = np.array([float(row["requested_H_over_B0"]) for row in rows])
    y = np.array([float(row["gibbs_over_G0"]) for row in rows])
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(h, meissner, color="black", linewidth=1.0, label="Meissner branch")
    ax.axhline(0, color="0.4", linestyle=":", label="normal branch")
    ax.plot(x[order], y[order], "o-", color="black", markersize=3, label="TDGL branch")
    ax.set(
        xlabel=r"$H/B_0$",
        ylabel=r"$G/G_0$",
        title=rf"Figure 2 open-boundary Gibbs surrogate ($\alpha_s={alpha_s:g}$)",
    )
    ax.set_ylim(bottom=min(-0.65, float(np.nanmin(y)) - 0.05), top=0.5)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output = root / "figure_2_gibbs.png"
    fig.savefig(output, dpi=250)
    plt.close(fig)
    return output


def run_driven_point(
    args,
    preset: ReproductionPreset,
    root: Path,
    *,
    figure: str,
    tag_parts: Sequence[float | str],
    device: tdgl.Device,
    alpha_s: float,
    field: float,
    current: np.ndarray,
    disorder_epsilon=1.0,
    seed_solution: tdgl.Solution | None = None,
) -> tdgl.Solution:
    path = checkpoint_path(root, figure, "drive", *tag_parts)
    return solve_or_load(
        device,
        drive_options(preset, path, current, device.layer.model),
        reduced_field_source(device, field),
        alpha_s=alpha_s,
        chirality=args.chirality,
        disorder_epsilon=disorder_epsilon,
        seed_solution=seed_solution,
        plot_only=args.plot_only,
    )


def run_figure_3(args, preset: ReproductionPreset, root: Path) -> Path:
    csv_path = root / "figure_3_flux_flow.csv"
    kappa = figure_kappa(args, "3")
    if args.plot_only:
        rows = read_csv(csv_path)
    else:
        rows = []
        for alpha_s in preset.fig3_alphas:
            hc2 = hc2_over_b0(alpha_s)
            for target_b_over_bc2 in preset.fig3_fields:
                open_field = b_over_b0_from_bc2_fraction(alpha_s, target_b_over_bc2)
                equilibrium_model = paper_model(alpha_s, 2.0)
                equilibrium_device = make_square_device(
                    equilibrium_model,
                    side_length=one_flux_cell_side(open_field),
                    grid_points=preset.grid_points,
                    kappa=kappa,
                    mesh_seed=args.mesh_seed,
                )
                eq_path = checkpoint_path(
                    root,
                    "3",
                    "equilibrium",
                    alpha_s,
                    target_b_over_bc2,
                )
                equilibrium = solve_or_load(
                    equilibrium_device,
                    equilibrium_options(preset, eq_path, equilibrium_model),
                    reduced_field_source(equilibrium_device, open_field),
                    alpha_s=alpha_s,
                    chirality=args.chirality,
                    num_vortices=1,
                    plot_only=args.plot_only,
                )
                equilibrium_vortices = check_vortex_retention(
                    equilibrium,
                    expected=1,
                    strict=args.strict_vortex,
                    label=(
                        f"Figure 3 equilibrium alpha={alpha_s:g}, "
                        f"B/Bc2={target_b_over_bc2:g}"
                    ),
                )
                for relaxation_q in preset.fig3_relaxations:
                    model = paper_model(alpha_s, relaxation_q)
                    device = make_square_device(
                        model,
                        side_length=one_flux_cell_side(open_field),
                        grid_points=preset.grid_points,
                        kappa=kappa,
                        mesh_seed=args.mesh_seed,
                    )
                    dynamics_seed = rebind_seed_to_device(equilibrium, device)
                    current = np.array([args.fig3_current, 0.0])
                    driven = run_driven_point(
                        args,
                        preset,
                        root,
                        figure="3",
                        tag_parts=(
                            alpha_s,
                            relaxation_q,
                            target_b_over_bc2,
                            args.fig3_current,
                        ),
                        device=device,
                        alpha_s=alpha_s,
                        field=open_field,
                        current=current,
                        seed_solution=dynamics_seed,
                    )
                    driven_vortices = check_vortex_retention(
                        driven,
                        expected=1,
                        strict=args.strict_vortex,
                        label=(
                            f"Figure 3 drive alpha={alpha_s:g}, q={relaxation_q:g}, "
                            f"B/Bc2={target_b_over_bc2:g}"
                        ),
                    )
                    electric = paper_clock_electric_field(driven)
                    electric_delta = paper_clock_electric_block_delta(driven)
                    measured_b_over_b0, b_std = induction_statistics(driven)
                    rows.append(
                        {
                            "alpha_s": alpha_s,
                            "relaxation_q": relaxation_q,
                            "target_B_over_Bc2": target_b_over_bc2,
                            "open_boundary_H_over_B0": open_field,
                            "measured_B_over_B0": measured_b_over_b0,
                            "measured_B_over_Bc2": measured_b_over_b0 / hc2,
                            "B_time_std_over_B0": b_std,
                            "B_time_std_over_Bc2": b_std / hc2,
                            "equilibrium_d_vortices": equilibrium_vortices,
                            "driven_d_vortices": driven_vortices,
                            "current": args.fig3_current,
                            "E_parallel": electric[0],
                            "E_perpendicular": electric[1],
                            "E_parallel_block_error": abs(electric_delta[0]),
                            "E_perpendicular_block_error": abs(electric_delta[1]),
                            "rho_over_rho_n": electric[0] / args.fig3_current,
                            "bulk_density": bulk_condensate_density(driven),
                            "kappa": kappa,
                            "checkpoint": driven.path,
                            **state_diagnostics(equilibrium, "equilibrium"),
                            **state_diagnostics(driven, "drive"),
                        }
                    )
        write_csv(csv_path, rows)

    alphas = list(preset.fig3_alphas)
    if args.plot_only:
        alphas = sorted({float(row["alpha_s"]) for row in rows}, reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()
    markers = {1.0: "o", 10.0: "^"}
    for ax, alpha_s in zip(axes, alphas):
        qs = sorted(
            {
                float(row["relaxation_q"])
                for row in rows
                if np.isclose(float(row["alpha_s"]), alpha_s)
            }
        )
        for q in qs:
            curve = [
                row
                for row in rows
                if np.isclose(float(row["alpha_s"]), alpha_s)
                and np.isclose(float(row["relaxation_q"]), q)
            ]
            curve.sort(key=lambda row: float(row["measured_B_over_Bc2"]))
            ax.plot(
                [float(row["measured_B_over_Bc2"]) for row in curve],
                [float(row["rho_over_rho_n"]) for row in curve],
                marker=markers.get(q, "o"),
                linewidth=1,
                markersize=4,
                label=rf"$q={q:g}$",
            )
        ax.set_title(rf"$\alpha_s={alpha_s:g}$")
        ax.grid(alpha=0.2)
    for ax in axes[len(alphas) :]:
        ax.set_visible(False)
    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel(r"measured $B/B_{c2}$")
            ax.set_ylabel(r"$\rho/\rho_n$")
    if alphas:
        axes[0].legend()
    fig.suptitle("Li-Wang-Wang Figure 3: open-boundary vortex-flow surrogate")
    fig.tight_layout()
    output = root / "figure_3_flux_flow.png"
    fig.savefig(output, dpi=250)
    plt.close(fig)
    return output


def fit_depinning_curve(currents: np.ndarray, resistivities: np.ndarray):
    """Fit ``a sqrt(1-(Jc/J)^2)`` by a stable one-dimensional grid search."""
    currents = np.asarray(currents, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    if currents.shape != resistivities.shape or currents.ndim != 1:
        raise ValueError("currents and resistivities must be same-length vectors")
    if len(currents) < 2 or np.any(currents <= 0):
        return float("nan"), float("nan")
    candidates = np.linspace(0.8 * currents.min(), 0.999 * currents.max(), 2000)
    best = (float("inf"), float("nan"), float("nan"))
    for critical in candidates:
        basis = np.sqrt(np.maximum(0.0, 1.0 - (critical / currents) ** 2))
        norm = float(np.dot(basis, basis))
        if norm <= 0:
            continue
        amplitude = max(0.0, float(np.dot(basis, resistivities) / norm))
        residual = resistivities - amplitude * basis
        score = float(np.dot(residual, residual))
        if score < best[0]:
            best = score, critical, amplitude
    return best[1], best[2]


def run_figure_4(args, preset: ReproductionPreset, root: Path) -> Path:
    csv_path = root / "figure_4_twin_depinning.csv"
    kappa = figure_kappa(args, "4")
    field = (
        args.fig4_field
        if args.fig4_field is not None
        else 2 * math.pi / DEFAULT_TWIN_SPACING**2
    )
    cases = [
        ("d, us=ud=0.5", -1.0, 0.5, 0.5, "v"),
        ("d+is, us=0.5", 0.85, 0.5, 0.5, "s"),
        ("d+is, us=0.85", 0.85, 0.85, 0.5, "o"),
        ("d+is, us=1.5", 0.85, 1.5, 0.5, "^"),
    ]
    if args.plot_only:
        rows = read_csv(csv_path)
    else:
        rows = []
        side = DEFAULT_TWIN_SPACING
        width = args.twin_width or side / (preset.grid_points - 1)
        for label, alpha_s, u_s, u_d, _ in cases:
            model = paper_model(alpha_s, 1.0, u_s=u_s, u_d=u_d)
            device = make_square_device(
                model,
                side_length=side,
                grid_points=preset.grid_points,
                kappa=kappa,
                mesh_seed=args.mesh_seed,
            )
            disorder = twin_disorder_profile(u_d=u_d, width=width)
            epsilon_values = disorder(device.mesh.sites, vectorized=True)
            effective_u_d = float(
                np.sum((1 - epsilon_values) * device.mesh.areas) / side
            )
            effective_u_s = effective_u_d * u_s / u_d
            eq_path = checkpoint_path(
                root, "4", "equilibrium", alpha_s, u_s, u_d, field, width
            )
            equilibrium = solve_or_load(
                device,
                equilibrium_options(preset, eq_path, model),
                reduced_field_source(device, field),
                alpha_s=alpha_s,
                chirality=args.chirality,
                disorder_epsilon=disorder,
                num_vortices=1,
                plot_only=args.plot_only,
            )
            equilibrium_vortices = check_vortex_retention(
                equilibrium,
                expected=1,
                strict=args.strict_vortex,
                label=f"Figure 4 equilibrium {label}",
            )
            previous = equilibrium
            for current_value in preset.fig4_currents:
                seed = previous if not args.independent_currents else equilibrium
                driven = run_driven_point(
                    args,
                    preset,
                    root,
                    figure="4",
                    tag_parts=(alpha_s, u_s, u_d, field, width, current_value),
                    device=device,
                    alpha_s=alpha_s,
                    field=field,
                    current=np.array([current_value, 0.0]),
                    disorder_epsilon=disorder,
                    seed_solution=seed,
                )
                driven_vortices = check_vortex_retention(
                    driven,
                    expected=1,
                    strict=args.strict_vortex,
                    label=f"Figure 4 drive {label}, J={current_value:g}",
                )
                electric = paper_clock_electric_field(driven)
                electric_delta = paper_clock_electric_block_delta(driven)
                measured_b_over_b0, b_std = induction_statistics(driven)
                rows.append(
                    {
                        "case": label,
                        "alpha_s": alpha_s,
                        "u_s": u_s,
                        "u_d": u_d,
                        "twin_width": width,
                        "effective_u_s": effective_u_s,
                        "effective_u_d": effective_u_d,
                        "inferred_open_boundary_H_over_B0": field,
                        "measured_B_over_B0": measured_b_over_b0,
                        "B_time_std_over_B0": b_std,
                        "equilibrium_d_vortices": equilibrium_vortices,
                        "driven_d_vortices": driven_vortices,
                        "current": current_value,
                        "E_parallel": electric[0],
                        "E_parallel_block_error": abs(electric_delta[0]),
                        "rho_over_rho_n": electric[0] / current_value,
                        "kappa": kappa,
                        "checkpoint": driven.path,
                        **state_diagnostics(equilibrium, "equilibrium"),
                        **state_diagnostics(driven, "drive"),
                    }
                )
                previous = driven
        write_csv(csv_path, rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for label, _, _, _, marker in cases:
        curve = [row for row in rows if row["case"] == label]
        if not curve:
            continue
        curve.sort(key=lambda row: float(row["current"]))
        current = np.array([float(row["current"]) for row in curve])
        rho = np.array([float(row["rho_over_rho_n"]) for row in curve])
        fit_mask = current <= args.fig4_fit_max
        critical, amplitude = fit_depinning_curve(current[fit_mask], rho[fit_mask])
        ax.scatter(current, rho, marker=marker, s=30, label=label)
        if math.isfinite(critical):
            dense = np.linspace(current.min(), current.max(), 400)
            fit = amplitude * np.sqrt(np.maximum(0.0, 1.0 - (critical / dense) ** 2))
            ax.plot(dense, fit, linewidth=1)
    ax.set(
        xlabel=r"applied current $J$",
        ylabel=r"$\rho/\rho_n$",
        title="Li-Wang-Wang Figure 4: open-boundary twin surrogate",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output = root / "figure_4_twin_depinning.png"
    fig.savefig(output, dpi=250)
    plt.close(fig)
    return output


def figure_kappa(args, figure: str) -> float:
    """Use explicit --kappa or documented figure-specific inferred defaults."""
    if args.kappa is not None:
        return args.kappa
    # Figure 2's Meissner parabola visibly has kappa=2.  The dynamics method
    # cited as Ref. 17 used kappa=3 for free flow and twin boundaries.
    return 2.0 if figure == "2" else 3.0


def parse_figures(values: Iterable[str]) -> tuple[str, ...]:
    requested = []
    for value in values:
        requested.extend(item.strip() for item in value.split(",") if item.strip())
    if "all" in requested:
        return ("2", "3", "4")
    invalid = sorted(set(requested) - {"2", "3", "4"})
    if invalid:
        raise ValueError(f"Unknown figure selections: {invalid}")
    return tuple(dict.fromkeys(requested))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_configuration(args, preset: ReproductionPreset) -> dict:
    """Return the simulation-defining payload used to isolate checkpoints."""
    excluded = {"figures", "output_dir", "plot_only", "strict_vortex"}
    argument_values = {
        name: value for name, value in vars(args).items() if name not in excluded
    }
    return _json_ready(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "preset": args.preset,
            "preset_values": asdict(preset),
            "arguments": argument_values,
        }
    )


def configuration_digest(configuration: dict) -> str:
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def write_manifest(
    args,
    preset: ReproductionPreset,
    root: Path,
    figures,
    configuration: dict,
    digest: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if args.plot_only:
        if not path.exists():
            raise FileNotFoundError(f"Missing run manifest in --plot-only mode: {path}")
        return path
    previous_figures = []
    if path.exists():
        with path.open() as stream:
            previous_figures = json.load(stream).get("figures", [])
    manifest = {
        "paper": {"id": PAPER_ID, "title": PAPER_TITLE},
        "configuration_digest": digest,
        "configuration": configuration,
        "figures": list(dict.fromkeys([*previous_figures, *figures])),
        "preset": args.preset,
        "preset_values": asdict(preset),
        "assumptions": {
            "boundary_conditions": "open variational surrogate for magnetic-periodic cells",
            "figure_2_alpha_s": args.fig2_alpha,
            "figure_3_current": args.fig3_current,
            "figure_4_field": (
                args.fig4_field
                if args.fig4_field is not None
                else 2 * math.pi / DEFAULT_TWIN_SPACING**2
            ),
            "figure_4_fit_max_current": args.fig4_fit_max,
            "strict_vortex": args.strict_vortex,
            "kappa_override": args.kappa,
            "effective_kappa": {
                figure: figure_kappa(args, figure) for figure in ("2", "3", "4")
            },
            "chirality": args.chirality,
            "mesh_seed": args.mesh_seed,
        },
        "paper_errata": {
            "figure_2_caption": "The PDF repeats the Figure 3 caption by mistake.",
        },
        "limitations": [
            "No magnetic/twisted periodic boundary conditions.",
            "Figure 3 omits the q=0.01 relaxation branch.",
            "The paper omits several run lengths, tolerances, and figure parameters.",
            "Twin delta functions are regularized with a finite-width unit-area top hat.",
            "The open solver fixes boundary H; requested H and measured mean B differ.",
            "Figure 3 converts the paper's B/Bc2 to open-boundary H/B0.",
            "Solver-clock durations are multiplied by beta_em=2/q.",
            "A seeded vortex may escape; every CSV records its final net winding.",
        ],
    }
    with path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Figures 2--4 of Li, Wang, and Wang, cond-mat/9906211, "
            "with the repository's s+d TDGL framework."
        )
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=("3",),
        help="Figures to run: 2 3 4, comma-separated values, or all.",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from existing CSV data without running TDGL.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="Override inferred kappa=2 (Fig. 2) or kappa=3 (dynamics).",
    )
    parser.add_argument("--fig2-alpha", type=float, default=0.85)
    parser.add_argument("--fig3-current", type=float, default=DEFAULT_FIG3_CURRENT)
    parser.add_argument(
        "--fig4-field",
        type=float,
        default=None,
        help="H/B0 for Fig. 4; default infers one flux quantum in 10.8 xi square.",
    )
    parser.add_argument("--twin-width", type=float, default=None)
    parser.add_argument(
        "--fig4-fit-max",
        type=float,
        default=0.20,
        help="Maximum current included in the paper's stated low-J depinning fit.",
    )
    parser.add_argument("--chirality", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--mesh-seed", type=int, default=1999)
    parser.add_argument("--independent-currents", action="store_true")
    parser.add_argument(
        "--strict-vortex",
        action="store_true",
        help="Fail instead of warning if the open cell loses its seeded vortex.",
    )
    args = parser.parse_args()
    if args.kappa is not None and args.kappa <= 0:
        parser.error("--kappa must be positive")
    if args.fig3_current <= 0:
        parser.error("--fig3-current must be positive")
    if args.fig4_fit_max <= 0:
        parser.error("--fig4-fit-max must be positive")
    return args


def main() -> None:
    args = parse_args()
    figures = parse_figures(args.figures)
    preset = PRESETS[args.preset]
    configuration = run_configuration(args, preset)
    digest = configuration_digest(configuration)
    root = args.output_dir.resolve() / args.preset / digest
    write_manifest(args, preset, root, figures, configuration, digest)
    runners = {
        "2": run_figure_2,
        "3": run_figure_3,
        "4": run_figure_4,
    }
    outputs = []
    for figure in figures:
        outputs.append(runners[figure](args, preset, root))
    print("Generated:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
