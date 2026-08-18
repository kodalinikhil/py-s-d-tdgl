"""Reproduce the defect signatures in Lin, Maiti, and Chubukov (2016).

This script targets the two most convenient numerical results in
Phys. Rev. B 94, 064519:

* Figure 3: for a square defect, the maximum spontaneous field scales
  quadratically with defect strength for ``s+is`` and linearly for ``s+id``;
* Figure 4: a square defect produces a fourfold corner-field pattern for
  ``s+is`` and a twofold pattern for ``s+id``.

The paper specifies the static GL coefficients and defect shapes, but not the
simulation box, grid, boundary conditions, diffusion constants, conductivity,
or penetration depth.  The results here are therefore a controlled
finite-domain reproduction of the symmetry and scaling laws, not a claim of
pixel-for-pixel agreement in absolute field amplitude.

The paper parameters are

    alpha_1 = alpha_2 = -1, beta_1 = beta_2 = 1, gamma_2 = 0.5,
    m_1 = m_c = 1, m_2 = 2, gamma_1 = gamma_3 = 0.

For ``s+is`` these map directly to ``SPlusSModel``.  ``SPlusDModel`` uses the
d-wave coherence length as its coordinate unit, so the ``s+id`` mesh is
rescaled by ``sqrt(2)`` and its reduced field is multiplied by two before it is
reported in the paper's ``Hc2 = Phi0 / (2 pi xi_1**2)`` units.

Examples
--------
Quick end-to-end workflow check::

    python my_scripts/reproductions/reproduce_lin_maiti_chubukov_prb94_064519.py \
        --preset smoke

Moderate-resolution comparison::

    python my_scripts/reproductions/reproduce_lin_maiti_chubukov_prb94_064519.py \
        --preset quick

Paper-oriented mesh and defect-strength scan::

    python my_scripts/reproductions/reproduce_lin_maiti_chubukov_prb94_064519.py \
        --preset paper

Existing HDF5 checkpoints are reused.  Add ``--plot-only`` to redraw plots or
``--force`` to recompute the selected cases.

Reference: https://doi.org/10.1103/PhysRevB.94.064519
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "py-s-d-tdgl-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import Delaunay


if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import tdgl  # noqa: E402


PAPER_DOI = "10.1103/PhysRevB.94.064519"
PAPER_TITLE = (
    "Distinguishing between s+id and s+is pairing symmetries in multiband "
    "superconductors through spontaneous magnetization pattern induced by a defect"
)
RUN_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results/reproductions/lin_maiti_chubukov_2016"
MODEL_KEYS = ("s_plus_is", "s_plus_id")
MODEL_LABELS = {"s_plus_is": r"$s+is$", "s_plus_id": r"$s+id$"}


@dataclass(frozen=True)
class ReproductionPreset:
    """Numerical controls for one reproduction resolution."""

    points: int
    side_length: float
    solve_time: float
    dt_init: float
    dt_max: float
    equilibrium_tolerance: float | None
    equilibrium_window: int
    equilibrium_min_time: float
    scaling_strengths: tuple[float, ...]
    pattern_strength: float = 0.5


PRESETS = {
    "smoke": ReproductionPreset(
        points=9,
        side_length=8.0,
        solve_time=0.04,
        dt_init=2e-3,
        dt_max=1e-2,
        equilibrium_tolerance=None,
        equilibrium_window=5,
        equilibrium_min_time=0.0,
        scaling_strengths=(0.15, 0.3, 0.5),
    ),
    "quick": ReproductionPreset(
        points=21,
        side_length=12.0,
        solve_time=8.0,
        dt_init=2e-3,
        dt_max=2e-2,
        equilibrium_tolerance=1e-7,
        equilibrium_window=80,
        equilibrium_min_time=0.5,
        scaling_strengths=tuple(np.geomspace(0.03, 0.5, 7)),
    ),
    "paper": ReproductionPreset(
        points=41,
        side_length=16.0,
        solve_time=30.0,
        dt_init=1e-3,
        dt_max=2e-2,
        equilibrium_tolerance=1e-9,
        equilibrium_window=200,
        equilibrium_min_time=2.0,
        scaling_strengths=(
            0.01,
            0.02,
            0.05,
            0.08,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ),
    ),
}


@dataclass(frozen=True)
class ModelMapping:
    """Coordinate and field conversion for one solver model."""

    key: str
    coordinate_scale: float
    field_scale: float

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.key]


MODEL_MAPPINGS = {
    "s_plus_is": ModelMapping("s_plus_is", 1.0, 1.0),
    # xi_1 / xi_d = sqrt(m_2 / m_1) = sqrt(2), hence
    # Hc2_d / Hc2_1 = (xi_1 / xi_d)**2 = 2.
    "s_plus_id": ModelMapping("s_plus_id", math.sqrt(2.0), 2.0),
}


@dataclass
class FieldResult:
    """A solved defect case and its baseline-subtracted paper-unit field."""

    model_key: str
    strength: float
    shape: str
    r0: float
    solution: tdgl.Solution
    triangle_centers: np.ndarray
    field: np.ndarray
    maximum: float


def paper_s_plus_is_model() -> tdgl.SPlusSModel:
    """Return the paper's Eq. (3) in the repository's s+is convention."""
    model = tdgl.SPlusSModel(
        a1=-1.0,
        a2=-1.0,
        b1=1.0,
        b2=1.0,
        k2_over_k1=0.5,
        josephson_gamma=0.0,
        phase_gamma2=0.5,
        density_gamma3=0.0,
        mixed_gradient_k12=0.5,
        beta_em=1.0,
        disorder_coupling1=1.0,
        disorder_coupling2=1.0,
    )
    model.validate()
    return model


def paper_s_plus_id_model() -> tdgl.SPlusDModel:
    """Return the paper's Eq. (10) in the repository's s+d convention.

    ``SPlusDModel`` stores ``psi1=s`` and ``psi2=d`` and normalizes the
    d-sector stiffness to one.  Rescaling coordinates by ``sqrt(2)`` therefore
    gives ``eta_s=2`` and ``eta_v=-1``; the minus sign converts the repository's
    ``y-minus-x`` directional convention to the paper's ``x-minus-y`` term.
    """
    model = tdgl.SPlusDModel(
        eta_s=2.0,
        eta_v=-1.0,
        nu=1.0,
        tau1=1.0,
        tau3=0.0,
        tau4=0.5,
        beta_em=1.0,
        relaxation_s=1.0,
        nu_disorder_coupling=1.0,
    )
    model.validate()
    return model


def paper_model(model_key: str):
    """Return the framework model for ``model_key``."""
    if model_key == "s_plus_is":
        return paper_s_plus_is_model()
    if model_key == "s_plus_id":
        return paper_s_plus_id_model()
    raise ValueError(f"Unknown model key {model_key!r}.")


def uniform_chiral_state(chirality: int = 1) -> tuple[complex, complex]:
    """Return the paper's homogeneous ``(psi1, psi2)`` minimum."""
    if chirality not in (-1, 1):
        raise ValueError("chirality must be +1 or -1.")
    amplitude = math.sqrt(2.0)
    return complex(amplitude), complex(0.0, chirality * amplitude)


def defect_mask(points: np.ndarray, *, shape: str, r0: float) -> np.ndarray:
    """Evaluate one of the two defect shapes used for Figures 3 and 4."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")
    if not math.isfinite(r0) or r0 <= 0:
        raise ValueError("r0 must be positive and finite.")
    x = points[:, 0]
    y = points[:, 1]
    if shape == "square":
        return (np.abs(x) <= r0) & (np.abs(y) <= r0)
    if shape == "angular_n2":
        radius = np.hypot(x, y)
        theta = np.arctan2(y, x)
        return radius < r0 * np.abs(np.cos(2 * theta))
    raise ValueError("shape must be 'square' or 'angular_n2'.")


def defect_profile(
    *,
    strength: float,
    shape: str,
    r0: float,
    coordinate_scale: float,
) -> Callable:
    """Return ``epsilon=1-strength`` inside a paper-coordinate defect."""
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and in [0, 1].")
    if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be positive and finite.")

    def profile(r, *, vectorized=True):
        del vectorized
        solver_points = np.atleast_2d(np.asarray(r, dtype=float))
        paper_points = solver_points[:, :2] / coordinate_scale
        inside = defect_mask(paper_points, shape=shape, r0=r0)
        return 1.0 - strength * inside.astype(float)

    return profile


def make_device(
    model_key: str,
    *,
    side_length: float,
    points: int,
    kappa: float,
    mesh_seed: int,
) -> tdgl.Device:
    """Build a deterministic D4-symmetric mesh in the proper model coordinates."""
    if points < 7 or points % 2 == 0:
        raise ValueError("points must be an odd integer of at least 7.")
    if side_length <= 2 or not math.isfinite(side_length):
        raise ValueError("side_length must be finite and greater than 2.")
    if kappa <= 0 or not math.isfinite(kappa):
        raise ValueError("kappa must be positive and finite.")

    mapping = MODEL_MAPPINGS[model_key]
    solver_side = side_length * mapping.coordinate_scale
    solver_kappa = kappa * mapping.coordinate_scale
    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=solver_kappa,
        thickness=0.1,
        model=paper_model(model_key),
    )
    film = tdgl.Polygon(
        "film", points=tdgl.geometry.box(solver_side, solver_side)
    ).resample(max(4 * points, 40))
    device = tdgl.Device(
        f"lin_2016_{model_key}",
        layer=layer,
        film=film,
        length_units="um",
    )

    coordinates = np.linspace(-solver_side / 2, solver_side / 2, points)
    x, y = np.meshgrid(coordinates, coordinates)
    mesh_points = np.column_stack((x.ravel(), y.ravel()))
    interior = (np.abs(mesh_points[:, 0]) < solver_side / 2) & (
        np.abs(mesh_points[:, 1]) < solver_side / 2
    )
    spacing = solver_side / (points - 1)

    # A generic D4-equivariant warp removes Cartesian cocircularities while
    # retaining the rotation symmetry needed to distinguish the two patterns.
    rng = np.random.default_rng(mesh_seed)
    a, b, c, d = rng.uniform(-1.0, 1.0, size=4)
    x_normalized = 2 * mesh_points[:, 0] / solver_side
    y_normalized = 2 * mesh_points[:, 1] / solver_side
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
    maximum_displacement = np.max(np.linalg.norm(displacement[interior], axis=1))
    if maximum_displacement:
        displacement *= 0.10 * spacing / maximum_displacement
    mesh_points[interior] += displacement[interior]
    triangles = Delaunay(mesh_points).simplices
    device._create_dimensionless_mesh(mesh_points, triangles)
    if np.any(device.mesh.edge_mesh.dual_edge_lengths <= 0):
        raise RuntimeError("The reproduction mesh contains a nonpositive dual edge.")
    return device


def solver_options(
    preset: ReproductionPreset,
    output_file: Path,
) -> tdgl.SolverOptions:
    """Create local-screening relaxation options for one checkpoint."""
    return tdgl.SolverOptions(
        solve_time=preset.solve_time,
        dt_init=preset.dt_init,
        dt_max=preset.dt_max,
        adaptive=True,
        terminal_psi=None,
        include_screening=True,
        equilibrium_tolerance=preset.equilibrium_tolerance,
        equilibrium_window=preset.equilibrium_window,
        equilibrium_min_time=preset.equilibrium_min_time,
        save_every=1_000_000,
        progress_interval=10_000,
        output_file=str(output_file),
    )


def strength_slug(strength: float) -> str:
    """Return a stable filename fragment for a defect strength."""
    return f"{strength:.8g}".replace("-", "m").replace(".", "p")


def remove_checkpoint(path: Path) -> None:
    """Remove only one explicitly selected checkpoint and its temporary file."""
    for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
        if candidate.exists():
            candidate.unlink()


def evaluate_disorder_profile(profile, sites: np.ndarray) -> np.ndarray:
    """Evaluate a scalar or callable disorder profile on mesh sites."""
    if callable(profile):
        try:
            values = profile(sites, vectorized=True)
        except TypeError:
            values = np.asarray([profile(site) for site in sites])
    else:
        values = np.full(len(sites), profile)
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(len(sites), float(values))
    return values.reshape(-1)


def validate_checkpoint(
    solution: tdgl.Solution,
    *,
    device: tdgl.Device,
    options: tdgl.SolverOptions,
    profile: Callable,
) -> None:
    """Reject stale, incompatible, or interrupted reproduction checkpoints."""
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
            f"Checkpoint {solution.path} does not match the requested mesh/model. "
            "Rerun with --force."
        )

    for name in (
        "solve_time",
        "dt_init",
        "dt_max",
        "equilibrium_tolerance",
        "equilibrium_window",
        "equilibrium_min_time",
    ):
        saved = getattr(solution.options, name)
        expected = getattr(options, name)
        if saved is None or expected is None:
            matches = saved is expected
        elif isinstance(expected, (int, np.integer)):
            matches = int(saved) == int(expected)
        else:
            matches = math.isclose(
                float(saved), float(expected), rel_tol=1e-12, abs_tol=1e-14
            )
        if not matches:
            raise RuntimeError(
                f"Checkpoint {solution.path} has incompatible option {name}. "
                "Rerun with --force."
            )
    if bool(solution.options.include_screening) != bool(options.include_screening):
        raise RuntimeError(
            f"Checkpoint {solution.path} has incompatible screening settings. "
            "Rerun with --force."
        )

    sites = expected_mesh.sites
    saved_epsilon = evaluate_disorder_profile(solution.disorder_epsilon, sites)
    expected_epsilon = evaluate_disorder_profile(profile, sites)
    if saved_epsilon.shape != expected_epsilon.shape or not np.allclose(
        saved_epsilon, expected_epsilon, rtol=0, atol=1e-13
    ):
        raise RuntimeError(
            f"Checkpoint {solution.path} has a different defect profile. "
            "Rerun with --force."
        )

    state = solution.tdgl_data.state
    final_time = float(state.get("time", 0.0))
    reached = bool(state.get("equilibrium_reached", False))
    time_tolerance = max(1e-12, 2 * float(state.get("dt", options.dt_init)))
    if not reached and final_time + time_tolerance < options.solve_time:
        raise RuntimeError(
            f"Checkpoint {solution.path} is incomplete: t={final_time:g}, "
            f"expected {options.solve_time:g}. Rerun with --force."
        )


def warn_if_unconverged(
    solution: tdgl.Solution,
    options: tdgl.SolverOptions,
) -> None:
    """Warn when a tolerance-controlled solve stops at its time cap."""
    if options.equilibrium_tolerance is not None and not bool(
        solution.tdgl_data.state.get("equilibrium_reached", False)
    ):
        warnings.warn(
            f"One or more checkpoints in {Path(solution.path).parent} reached "
            "the time cap without satisfying the coupled equilibrium criterion; "
            "see the CSV/JSON diagnostics and increase --solve-time if needed.",
            RuntimeWarning,
            stacklevel=2,
        )


def solve_case(
    *,
    model_key: str,
    device: tdgl.Device,
    preset: ReproductionPreset,
    output_file: Path,
    profile: Callable,
    seed_solution: tdgl.Solution | None,
    chirality: int,
    plot_only: bool,
    force: bool,
) -> tdgl.Solution:
    """Load or calculate one local-screening equilibrium checkpoint."""
    options = solver_options(preset, output_file)
    if output_file.exists() and not force:
        solution = tdgl.Solution.from_hdf5(str(output_file))
        validate_checkpoint(
            solution,
            device=device,
            options=options,
            profile=profile,
        )
        warn_if_unconverged(solution, options)
        return solution
    if plot_only:
        raise FileNotFoundError(
            f"Missing checkpoint {output_file}. Rerun without --plot-only."
        )
    if force:
        remove_checkpoint(output_file)

    print(f"Solving {model_key}: {output_file.stem}", flush=True)
    solver = tdgl.TDGLSolver(
        device=device,
        options=options,
        applied_vector_potential=0.0,
        terminal_currents=None,
        disorder_epsilon=profile,
        seed_solution=seed_solution,
    )
    if seed_solution is None:
        psi1, psi2 = uniform_chiral_state(chirality)
        solver.psi1_init.fill(psi1)
        solver.psi2_init.fill(psi2)
    solution = solver.solve()
    if solution is None:
        raise RuntimeError(f"Simulation was cancelled for {output_file}.")
    if Path(solution.path).resolve() != output_file.resolve():
        raise RuntimeError(
            f"Solver wrote {solution.path} instead of requested {output_file}."
        )
    warn_if_unconverged(solution, options)
    return solution


def paper_triangle_field(
    solution: tdgl.Solution,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle centers and Bz in the paper's xi_1/Hc2_1 units."""
    mapping = MODEL_MAPPINGS[model_key]
    mesh = solution.device.mesh
    centers = mesh.sites[mesh.elements].mean(axis=1) / mapping.coordinate_scale
    field = solution.local_magnetic_induction(units="Bc2", with_units=False)
    return centers, mapping.field_scale * np.asarray(field)


def localized_maximum(
    centers: np.ndarray,
    field: np.ndarray,
    *,
    r0: float,
    side_length: float,
) -> float:
    """Return the maximum defect-localized absolute field."""
    radius = np.linalg.norm(centers, axis=1)
    cutoff = min(4 * r0, 0.45 * side_length)
    local = radius <= cutoff
    if not np.any(local):
        local = np.ones(len(field), dtype=bool)
    return float(np.max(np.abs(field[local])))


def solve_field_result(
    *,
    model_key: str,
    strength: float,
    shape: str,
    r0: float,
    baseline: tdgl.Solution,
    baseline_field: np.ndarray,
    preset: ReproductionPreset,
    run_directory: Path,
    chirality: int,
    plot_only: bool,
    force: bool,
) -> FieldResult:
    """Solve one defect and subtract its zero-defect numerical baseline."""
    mapping = MODEL_MAPPINGS[model_key]
    output_file = run_directory / (
        f"{model_key}_{shape}_r0_{strength_slug(r0)}_"
        f"alpha_{strength_slug(strength)}.h5"
    )
    profile = defect_profile(
        strength=strength,
        shape=shape,
        r0=r0,
        coordinate_scale=mapping.coordinate_scale,
    )
    solution = solve_case(
        model_key=model_key,
        device=baseline.device,
        preset=preset,
        output_file=output_file,
        profile=profile,
        seed_solution=baseline,
        chirality=chirality,
        plot_only=plot_only,
        force=force,
    )
    centers, total_field = paper_triangle_field(solution, model_key)
    if total_field.shape != baseline_field.shape:
        raise RuntimeError("Defect and baseline meshes do not match.")
    field = total_field - baseline_field
    maximum = localized_maximum(
        centers,
        field,
        r0=r0,
        side_length=preset.side_length,
    )
    return FieldResult(
        model_key=model_key,
        strength=strength,
        shape=shape,
        r0=r0,
        solution=solution,
        triangle_centers=centers,
        field=field,
        maximum=maximum,
    )


def fit_scaling_exponent(results: Sequence[FieldResult]) -> float:
    """Fit the weakest usable points on a log-log scale."""
    ordered = sorted(results, key=lambda item: item.strength)
    usable = [item for item in ordered if item.strength > 0 and item.maximum > 0]
    if len(usable) < 2:
        return float("nan")
    count = min(len(usable), max(3, math.ceil(len(usable) / 2)))
    selected = usable[:count]
    return float(
        np.polyfit(
            np.log([item.strength for item in selected]),
            np.log([item.maximum for item in selected]),
            1,
        )[0]
    )


def save_scaling_csv(results: Iterable[FieldResult], path: Path) -> None:
    """Write the Figure 3 measurements and terminal solver state."""
    rows = []
    for result in results:
        state = result.solution.tdgl_data.state
        rows.append(
            {
                "model": result.model_key,
                "strength": result.strength,
                "max_abs_B_over_Hc2_1": result.maximum,
                "final_time": state.get("time", np.nan),
                "final_step": state.get("step", -1),
                "equilibrium_reached": state.get("equilibrium_reached", False),
                "equilibrium_error": state.get("equilibrium_error", np.nan),
            }
        )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solution_diagnostics(solution: tdgl.Solution) -> dict:
    """Return JSON-safe completion and convergence diagnostics."""
    state = solution.tdgl_data.state
    equilibrium_error = float(state.get("equilibrium_error", float("nan")))
    return {
        "final_time": float(state.get("time", float("nan"))),
        "final_step": int(state.get("step", -1)),
        "equilibrium_reached": bool(state.get("equilibrium_reached", False)),
        "equilibrium_error": (
            equilibrium_error if math.isfinite(equilibrium_error) else None
        ),
    }


def plot_scaling(results: Sequence[FieldResult], output_file: Path) -> dict[str, float]:
    """Create the Figure 3 analogue with fitted and expected slopes."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    colors = {"s_plus_is": "tab:red", "s_plus_id": "tab:blue"}
    markers = {"s_plus_is": "s", "s_plus_id": "o"}
    exponents = {}
    for model_key in MODEL_KEYS:
        model_results = sorted(
            (result for result in results if result.model_key == model_key),
            key=lambda item: item.strength,
        )
        strengths = np.asarray([item.strength for item in model_results])
        maxima = np.asarray([item.maximum for item in model_results])
        exponent = fit_scaling_exponent(model_results)
        exponents[model_key] = exponent
        ax.loglog(
            strengths,
            maxima,
            marker=markers[model_key],
            color=colors[model_key],
            linewidth=1.5,
            label=rf"{MODEL_LABELS[model_key]} (fit {exponent:.2f})",
        )
        positive = maxima > 0
        if np.any(positive):
            expected_exponent = 2 if model_key == "s_plus_is" else 1
            anchor = np.flatnonzero(positive)[0]
            reference = (
                maxima[anchor] * (strengths / strengths[anchor]) ** expected_exponent
            )
            ax.loglog(
                strengths,
                reference,
                linestyle="--",
                color=colors[model_key],
                alpha=0.55,
                label=rf"expected $\tilde\alpha^{expected_exponent}$",
            )
    ax.set_xlabel(r"Defect strength $\tilde\alpha$")
    ax.set_ylabel(r"$\max |B_z|/H_{c2,1}$")
    ax.set_title("Lin-Maiti-Chubukov Figure 3 analogue")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)
    return exponents


def plot_patterns(results: Sequence[FieldResult], output_file: Path) -> None:
    """Create the Figure 4 analogue for a hard square defect."""
    if {result.model_key for result in results} != set(MODEL_KEYS):
        raise ValueError("Pattern plot requires one result for each model.")
    ordered = [
        next(item for item in results if item.model_key == key) for key in MODEL_KEYS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for ax, result in zip(axes, ordered):
        mapping = MODEL_MAPPINGS[result.model_key]
        mesh = result.solution.device.mesh
        sites = mesh.sites / mapping.coordinate_scale
        triangulation = mtri.Triangulation(
            sites[:, 0], sites[:, 1], triangles=mesh.elements
        )
        limit = max(float(np.max(np.abs(result.field))), np.finfo(float).eps)
        image = ax.tripcolor(
            triangulation,
            facecolors=result.field,
            shading="flat",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        square = plt.Rectangle(
            (-result.r0, -result.r0),
            2 * result.r0,
            2 * result.r0,
            fill=False,
            color="black",
            linewidth=1.2,
        )
        ax.add_patch(square)
        view = min(4 * result.r0, 0.45 * np.ptp(sites[:, 0]))
        ax.set_xlim(-view, view)
        ax.set_ylim(-view, view)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/\xi_1$")
        ax.set_ylabel(r"$y/\xi_1$")
        ax.set_title(
            f"{mapping.label}, " + rf"$\max|B_z|={result.maximum:.2e}H_{{c2,1}}$"
        )
        colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
        colorbar.set_label(r"$(B_z-B_z^{(0)})/H_{c2,1}$")
    fig.suptitle("Lin-Maiti-Chubukov Figure 4 analogue: square defect")
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def parse_figures(values: Sequence[str]) -> tuple[str, ...]:
    """Parse ``--figures`` values into a stable Figure 3/4 selection."""
    tokens = []
    for value in values:
        tokens.extend(item.strip().lower() for item in value.split(",") if item.strip())
    if "all" in tokens:
        return ("3", "4")
    unknown = sorted(set(tokens).difference({"3", "4"}))
    if unknown:
        raise ValueError(f"Unknown figure selection(s): {unknown}.")
    selected = tuple(figure for figure in ("3", "4") if figure in tokens)
    if not selected:
        raise ValueError("Select Figure 3, Figure 4, or all.")
    return selected


def configuration_slug(
    preset: ReproductionPreset,
    *,
    kappa: float,
    mesh_seed: int,
    chirality: int,
    scaling_r0: float,
    pattern_r0: float,
) -> str:
    """Return a directory name that separates incompatible checkpoints."""
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "preset": dataclasses.asdict(preset),
        "kappa": kappa,
        "mesh_seed": mesh_seed,
        "chirality": chirality,
        "scaling_r0": scaling_r0,
        "pattern_r0": pattern_r0,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    side = strength_slug(preset.side_length)
    kappa_slug = strength_slug(kappa)
    return (
        f"p{preset.points}_L{side}_k{kappa_slug}_seed{mesh_seed}_"
        f"chi{chirality:+d}_{digest}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        help="Figures to reproduce: 3, 4, or all (default: all).",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--points", type=int, help="Override the preset grid count.")
    parser.add_argument(
        "--side", type=float, help="Override the domain side in paper xi_1 units."
    )
    parser.add_argument(
        "--solve-time", type=float, help="Override each relaxation duration."
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        help="Override Figure 3 defect strengths.",
    )
    parser.add_argument("--pattern-strength", type=float, default=None)
    parser.add_argument("--scaling-r0", type=float, default=1.5)
    parser.add_argument("--pattern-r0", type=float, default=1.0)
    parser.add_argument(
        "--kappa",
        type=float,
        default=4.0,
        help="Assumed lambda/xi_1 (not specified by the paper; default: 4).",
    )
    parser.add_argument("--mesh-seed", type=int, default=2016)
    parser.add_argument("--chirality", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def apply_overrides(
    preset: ReproductionPreset,
    args: argparse.Namespace,
) -> ReproductionPreset:
    """Apply explicit CLI resolution overrides to a frozen preset."""
    changes = {}
    if args.points is not None:
        changes["points"] = args.points
    if args.side is not None:
        changes["side_length"] = args.side
    if args.solve_time is not None:
        changes["solve_time"] = args.solve_time
    if args.strengths is not None:
        changes["scaling_strengths"] = tuple(args.strengths)
    if args.pattern_strength is not None:
        changes["pattern_strength"] = args.pattern_strength
    result = dataclasses.replace(preset, **changes)
    if result.solve_time <= 0:
        raise ValueError("solve_time must be positive.")
    if any(not 0 < value <= 1 for value in result.scaling_strengths):
        raise ValueError("All Figure 3 strengths must be in (0, 1].")
    if not 0 < result.pattern_strength <= 1:
        raise ValueError("pattern_strength must be in (0, 1].")
    return result


def run_reproduction(args: argparse.Namespace) -> dict:
    figures = parse_figures(args.figures)
    if args.force and args.plot_only:
        raise ValueError("--force and --plot-only cannot be used together.")
    preset = apply_overrides(PRESETS[args.preset], args)
    if args.scaling_r0 <= 0 or args.pattern_r0 <= 0:
        raise ValueError("Defect radii must be positive.")

    run_directory = (
        args.output_dir
        / args.preset
        / configuration_slug(
            preset,
            kappa=args.kappa,
            mesh_seed=args.mesh_seed,
            chirality=args.chirality,
            scaling_r0=args.scaling_r0,
            pattern_r0=args.pattern_r0,
        )
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    baselines = {}
    baseline_fields = {}
    for model_key in MODEL_KEYS:
        device = make_device(
            model_key,
            side_length=preset.side_length,
            points=preset.points,
            kappa=args.kappa,
            mesh_seed=args.mesh_seed,
        )
        mapping = MODEL_MAPPINGS[model_key]
        baseline_path = run_directory / f"{model_key}_baseline.h5"
        baseline_profile = defect_profile(
            strength=0.0,
            shape="square",
            r0=args.pattern_r0,
            coordinate_scale=mapping.coordinate_scale,
        )
        baseline = solve_case(
            model_key=model_key,
            device=device,
            preset=preset,
            output_file=baseline_path,
            profile=baseline_profile,
            seed_solution=None,
            chirality=args.chirality,
            plot_only=args.plot_only,
            force=args.force,
        )
        baselines[model_key] = baseline
        _, baseline_fields[model_key] = paper_triangle_field(baseline, model_key)

    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "paper": {"title": PAPER_TITLE, "doi": PAPER_DOI},
        "figures": list(figures),
        "preset": args.preset,
        "numerics": dataclasses.asdict(preset),
        "assumed_kappa_lambda_over_xi1": args.kappa,
        "mesh_seed": args.mesh_seed,
        "chirality": args.chirality,
        "baseline_solver_state": {
            model_key: solution_diagnostics(solution)
            for model_key, solution in baselines.items()
        },
        "limitations": [
            "The paper does not state its domain, grid, boundary conditions, "
            "diffusion constants, conductivity, or penetration depth.",
            "This script tests field symmetry and defect-strength exponents; "
            "absolute amplitudes are implementation-dependent.",
            "Open B=H boundaries replace the paper's undocumented numerical boundary.",
        ],
        "outputs": {},
    }

    if "3" in figures:
        scaling_results = []
        for model_key in MODEL_KEYS:
            for strength in preset.scaling_strengths:
                scaling_results.append(
                    solve_field_result(
                        model_key=model_key,
                        strength=strength,
                        shape="square",
                        r0=args.scaling_r0,
                        baseline=baselines[model_key],
                        baseline_field=baseline_fields[model_key],
                        preset=preset,
                        run_directory=run_directory,
                        chirality=args.chirality,
                        plot_only=args.plot_only,
                        force=args.force,
                    )
                )
        csv_path = run_directory / "figure3_scaling.csv"
        plot_path = run_directory / "figure3_scaling.png"
        save_scaling_csv(scaling_results, csv_path)
        exponents = plot_scaling(scaling_results, plot_path)
        summary["figure3"] = {
            "defect": "abs(x) <= r0 and abs(y) <= r0",
            "r0_over_xi1": args.scaling_r0,
            "fitted_exponents": exponents,
            "expected_exponents": {"s_plus_is": 2, "s_plus_id": 1},
        }
        summary["outputs"]["figure3_csv"] = str(csv_path)
        summary["outputs"]["figure3_plot"] = str(plot_path)

    if "4" in figures:
        pattern_results = []
        for model_key in MODEL_KEYS:
            pattern_results.append(
                solve_field_result(
                    model_key=model_key,
                    strength=preset.pattern_strength,
                    shape="square",
                    r0=args.pattern_r0,
                    baseline=baselines[model_key],
                    baseline_field=baseline_fields[model_key],
                    preset=preset,
                    run_directory=run_directory,
                    chirality=args.chirality,
                    plot_only=args.plot_only,
                    force=args.force,
                )
            )
        plot_path = run_directory / "figure4_patterns.png"
        plot_patterns(pattern_results, plot_path)
        summary["figure4"] = {
            "defect": "abs(x) <= r0 and abs(y) <= r0",
            "r0_over_xi1": args.pattern_r0,
            "strength": preset.pattern_strength,
            "max_abs_B_over_Hc2_1": {
                result.model_key: result.maximum for result in pattern_results
            },
            "solver_state": {
                result.model_key: solution_diagnostics(result.solution)
                for result in pattern_results
            },
        }
        summary["outputs"]["figure4_plot"] = str(plot_path)

    summary_path = run_directory / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    with summary_path.open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_reproduction(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
