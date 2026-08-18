"""Reproduce Babaev, Carlstrom, and Speight, PRL 105, 067003 (2010).

The paper studies the static two-band s+s free energy

    F = 1/2 sum_a |(grad + i e A) psi_a|^2 + B^2 / 2
        + (|psi1|^2 - 1)^2 / 2 + alpha |psi2|^2
        + beta |psi2|^4 / 2 - eta |psi1| |psi2| cos(theta2-theta1).

``tdgl.SPlusSModel`` represents twice this condensate energy with

    a1=-2, b1=2, a2=2*alpha, b2=2*beta, gamma12=eta, k2/k1=1.

Multiplying the full energy by two does not change its minimizers or the
normalized interaction energy.  The regular ``tdgl.solve`` path prescribes A
for s+s and does not pin vortex cores, however, while the paper minimizes A
and constrains two vortex positions.  This script therefore uses
``SPlusSModel`` for the coefficient convention and supplies a script-local
static minimizer with the two missing capabilities:

* both condensate amplitudes and the gauge field are minimized together;
* the common phase winding is fixed at requested vortex positions.

The minimizer uses gauge-invariant link phases on a square finite-difference
grid.  It fixes the outer boundary to the homogeneous ground state and zero
covariant phase gradient.  Results approach the paper's infinite-domain
curves as ``--width``, ``--points``, and ``--maxiter`` are increased.

By default the script evaluates the printed parameter tables for Figs. 1 and
4, creates their interaction-energy plots, and makes the three line cuts in
Fig. 5.  This is a substantial nonlinear calculation.  Run ``--smoke-test``
first to verify the installation and output workflow on a coarse grid.

Examples
--------
Quick workflow check::

    python my_scripts/reproductions/reproduce_babaev_prl_105_067003.py --smoke-test

Converged scan (then repeat at higher resolution for a mesh study)::

    python my_scripts/reproductions/reproduce_babaev_prl_105_067003.py \\
        --points 96 --width 28 --maxiter 600

Only the profile plot::

    python my_scripts/reproductions/reproduce_babaev_prl_105_067003.py --figures 5

Reference: https://doi.org/10.1103/PhysRevLett.105.067003
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "py-s-d-tdgl-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import tdgl  # noqa: E402


PAPER_DOI = "10.1103/PhysRevLett.105.067003"
SQRT2 = math.sqrt(2)
DEFAULT_DISTANCES = SQRT2 * np.asarray(
    [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float
)
FIGURE1_DISTANCES = SQRT2 * np.asarray([2.5, 3, 4, 6, 7, 8, 10], dtype=float)
FIGURE5_DISTANCES = SQRT2 * np.asarray([3, 2, 1.5], dtype=float)


@dataclass(frozen=True)
class PaperParameters:
    """One curve in the parameter tables printed in Figs. 1-4."""

    figure: int
    curve: int
    alpha: float
    beta: float
    eta: float
    charge: float
    density_ratio: float | None = None
    total_density: float | None = None

    def framework_model(self) -> tdgl.SPlusSModel:
        """Return the exactly equivalent ``SPlusSModel`` condensate terms."""
        model = tdgl.SPlusSModel(
            a1=-2.0,
            a2=2 * self.alpha,
            b1=2.0,
            b2=2 * self.beta,
            k2_over_k1=1.0,
            josephson_gamma=self.eta,
            relaxation1=1.0,
            relaxation2=1.0,
            em_coupling=1.0,
        )
        model.validate()
        return model

    @property
    def label(self) -> str:
        return (
            rf"{self.curve}: $\alpha={self.alpha:g}$, "
            rf"$\beta={self.beta:g}$, $\eta={self.eta:g}$, "
            rf"$e={self.charge:g}$"
        )

    @property
    def slug(self) -> str:
        values = (self.alpha, self.beta, self.eta, self.charge)
        encoded = "_".join(f"{value:.8g}".replace(".", "p") for value in values)
        return f"fig{self.figure}_curve{self.curve}_{encoded}"


FIGURE_CURVES: dict[int, tuple[PaperParameters, ...]] = {
    1: tuple(
        PaperParameters(1, curve, alpha, 0.0, eta, 1.0, density_ratio=0.1)
        for curve, alpha, eta in (
            (1, 1.0, 0.63),
            (2, 0.5, 0.32),
            (3, 0.1, 0.063),
            (4, 0.05, 0.032),
        )
    ),
    4: (
        PaperParameters(4, 1, 0.1, 1.0, 1.09, 1.41, 0.5, 2.06),
        PaperParameters(4, 2, 0.25, 0.1, 0.35, 1.41, 0.36, 1.50),
        PaperParameters(4, 3, 0.1, 0.5, 0.55, 1.41, 0.5, 1.78),
        PaperParameters(4, 4, 0.1, 0.1, 0.215, 1.41, 0.5, 1.61),
    ),
}


@dataclass
class RelaxedState:
    f1: np.ndarray
    f2: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    winding: np.ndarray
    energy: float
    converged: bool
    iterations: int
    gradient_max: float
    message: str


def homogeneous_ground_state(model: tdgl.SPlusSModel) -> tuple[float, float]:
    """Find the positive homogeneous minimum of the two-band potential."""
    model.validate()
    gamma = abs(model.josephson_gamma)
    if model.b2 == 0:
        u1_sq = (gamma**2 / model.a2 - model.a1) / model.b1
        if u1_sq <= 0:
            raise ValueError("The supplied beta=0 model has no broken-symmetry state.")
        u1 = math.sqrt(u1_sq)
        return u1, gamma * u1 / model.a2

    def potential(values: np.ndarray) -> float:
        u1, u2 = values
        return float(
            model.a1 * u1**2
            + 0.5 * model.b1 * u1**4
            + model.a2 * u2**2
            + 0.5 * model.b2 * u2**4
            - 2 * gamma * u1 * u2
        )

    def gradient(values: np.ndarray) -> np.ndarray:
        u1, u2 = values
        return 2 * np.asarray(
            [
                model.a1 * u1 + model.b1 * u1**3 - gamma * u2,
                model.a2 * u2 + model.b2 * u2**3 - gamma * u1,
            ]
        )

    approximate_u1 = math.sqrt(max(-model.a1 / model.b1, 1e-6))
    approximate_u2 = gamma * approximate_u1 / max(model.a2, gamma, 1e-3)
    starts = (
        np.asarray([approximate_u1, approximate_u2]),
        np.asarray([1.0, 0.5]),
        np.asarray([1.5, 1.5]),
    )
    candidates = [
        minimize(
            potential,
            start,
            jac=gradient,
            method="L-BFGS-B",
            bounds=((0, None), (0, None)),
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
        )
        for start in starts
    ]
    result = min(candidates, key=lambda candidate: candidate.fun)
    if not result.success or np.linalg.norm(gradient(result.x), ord=np.inf) > 1e-7:
        raise RuntimeError(
            f"Unable to find the homogeneous ground state: {result.message}"
        )
    if result.fun >= -1e-12:
        raise ValueError("The model's homogeneous minimum is not superconducting.")
    return float(result.x[0]), float(result.x[1])


def wrap_phase(value: np.ndarray) -> np.ndarray:
    """Map phase differences to [-pi, pi)."""
    return (value + np.pi) % (2 * np.pi) - np.pi


class PinnedVortexGrid:
    """Gauge-invariant square-grid discretization of the static s+s energy."""

    def __init__(self, width: float, points: int):
        if not math.isfinite(width) or width <= 0:
            raise ValueError("width must be positive and finite.")
        if points < 10 or points % 2:
            raise ValueError("points must be an even integer of at least 10.")
        self.width = float(width)
        self.points = int(points)
        self.x = np.linspace(-width / 2, width / 2, points)
        self.y = self.x.copy()
        self.h = float(self.x[1] - self.x[0])
        self.X, self.Y = np.meshgrid(self.x, self.y)

        node_weights = np.ones((points, points), dtype=float)
        node_weights[[0, -1], :] *= 0.5
        node_weights[:, [0, -1]] *= 0.5
        self.node_weights = self.h**2 * node_weights

        self.horizontal_weights = np.ones((points, points - 1), dtype=float)
        self.horizontal_weights[[0, -1], :] = 0.5
        self.vertical_weights = np.ones((points - 1, points), dtype=float)
        self.vertical_weights[:, [0, -1]] = 0.5

        self.qx_free = np.ones((points, points - 1), dtype=bool)
        self.qx_free[[0, -1], :] = False
        self.qy_free = np.ones((points - 1, points), dtype=bool)
        self.qy_free[:, [0, -1]] = False
        self.num_amplitudes = (points - 2) ** 2
        self.num_qx = int(np.count_nonzero(self.qx_free))
        self.num_qy = int(np.count_nonzero(self.qy_free))

    def phase_links(
        self, vortices: Sequence[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Return wrapped phase links and plaquette windings for fixed vortices."""
        phase = np.zeros_like(self.X)
        individual_links = []
        for x_vortex, y_vortex in vortices:
            radius = np.hypot(self.X - x_vortex, self.Y - y_vortex)
            if np.min(radius) < 1e-12:
                raise ValueError(
                    "A vortex lies on a grid node. Use an even --points value and "
                    "keep vortices on y=0."
                )
            theta = np.arctan2(self.Y - y_vortex, self.X - x_vortex)
            phase += theta
            individual_links.append(
                (
                    wrap_phase(theta[:, 1:] - theta[:, :-1]),
                    wrap_phase(theta[1:, :] - theta[:-1, :]),
                )
            )
        dtheta_x = wrap_phase(phase[:, 1:] - phase[:, :-1])
        dtheta_y = wrap_phase(phase[1:, :] - phase[:-1, :])
        winding = (
            dtheta_x[:-1, :] + dtheta_y[:, 1:] - dtheta_x[1:, :] - dtheta_y[:, :-1]
        )
        expected = 2 * np.pi * len(vortices)
        if not math.isclose(float(np.sum(winding)), expected, abs_tol=1e-8):
            raise RuntimeError(
                "The discrete phase field does not contain the requested winding."
            )
        return dtheta_x, dtheta_y, winding, individual_links

    def vortex_core_mask(self, vortices: Sequence[tuple[float, float]]) -> np.ndarray:
        """Pin the nearest grid node for each prescribed vortex core."""
        mask = np.zeros((self.points, self.points), dtype=bool)
        for x_vortex, y_vortex in vortices:
            distance_sq = (self.X - x_vortex) ** 2 + (self.Y - y_vortex) ** 2
            nearest = int(np.argmin(distance_sq))
            if mask.ravel()[nearest]:
                raise ValueError(
                    "Two prescribed vortex cores map to the same grid nodes."
                )
            mask.ravel()[nearest] = True
        if np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
            raise ValueError("A pinned vortex core overlaps the outer boundary.")
        return mask

    def initial_fields(
        self,
        vortices: Sequence[tuple[float, float]],
        model: tdgl.SPlusSModel,
        charge: float,
        ground_state: tuple[float, float],
        individual_links: Sequence[tuple[np.ndarray, np.ndarray]],
        core_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct smooth isolated-vortex products for the optimizer."""
        u1, u2 = ground_state
        f1 = np.full_like(self.X, u1)
        f2 = np.full_like(self.X, u2)
        weak_mass = math.sqrt(max(model.a2, 1e-3))
        weak_core = min(5.0, max(0.75, 1 / weak_mass))
        for x_vortex, y_vortex in vortices:
            radius = np.hypot(self.X - x_vortex, self.Y - y_vortex)
            f1 *= np.tanh(radius / 0.8)
            f2 *= np.tanh(radius / weak_core)

        # Dirichlet vacuum values at the remote boundary.
        for field, value in ((f1, u1), (f2, u2)):
            field[[0, -1], :] = value
            field[:, [0, -1]] = value
            field[core_mask] = 0.0

        qx = np.zeros((self.points, self.points - 1), dtype=float)
        qy = np.zeros((self.points - 1, self.points), dtype=float)
        penetration_length = 1 / (charge * math.hypot(u1, u2))
        x_mid = 0.5 * (self.x[:-1] + self.x[1:])
        y_mid = x_mid.copy()
        for (x_vortex, y_vortex), (link_x, link_y) in zip(vortices, individual_links):
            horizontal_radius = np.hypot(
                x_mid[None, :] - x_vortex, self.y[:, None] - y_vortex
            )
            vertical_radius = np.hypot(
                self.x[None, :] - x_vortex, y_mid[:, None] - y_vortex
            )
            qx += link_x * np.exp(-horizontal_radius / penetration_length)
            qy += link_y * np.exp(-vertical_radius / penetration_length)
        qx[~self.qx_free] = 0
        qy[~self.qy_free] = 0
        return f1, f2, np.clip(qx, -np.pi, np.pi), np.clip(qy, -np.pi, np.pi)

    def pack(
        self, f1: np.ndarray, f2: np.ndarray, qx: np.ndarray, qy: np.ndarray
    ) -> np.ndarray:
        return np.concatenate(
            (
                f1[1:-1, 1:-1].ravel(),
                f2[1:-1, 1:-1].ravel(),
                qx[self.qx_free],
                qy[self.qy_free],
            )
        )

    def unpack(
        self, values: np.ndarray, ground_state: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        expected = 2 * self.num_amplitudes + self.num_qx + self.num_qy
        if values.shape != (expected,):
            raise ValueError(f"Expected {expected} variables, got {values.shape}.")
        cursor = 0
        f1 = np.full((self.points, self.points), ground_state[0], dtype=float)
        f2 = np.full((self.points, self.points), ground_state[1], dtype=float)
        next_cursor = cursor + self.num_amplitudes
        f1[1:-1, 1:-1] = values[cursor:next_cursor].reshape(
            self.points - 2, self.points - 2
        )
        cursor = next_cursor
        next_cursor += self.num_amplitudes
        f2[1:-1, 1:-1] = values[cursor:next_cursor].reshape(
            self.points - 2, self.points - 2
        )
        cursor = next_cursor
        next_cursor += self.num_qx
        qx = np.zeros((self.points, self.points - 1), dtype=float)
        qx[self.qx_free] = values[cursor:next_cursor]
        cursor = next_cursor
        qy = np.zeros((self.points - 1, self.points), dtype=float)
        qy[self.qy_free] = values[cursor:]
        return f1, f2, qx, qy

    def energy_and_gradient(
        self,
        values: np.ndarray,
        *,
        winding: np.ndarray,
        model: tdgl.SPlusSModel,
        charge: float,
        ground_state: tuple[float, float],
        core_mask: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Evaluate twice the paper energy and its analytic gradient."""
        f1, f2, qx, qy = self.unpack(values, ground_state)
        if core_mask is not None:
            f1[core_mask] = 0.0
            f2[core_mask] = 0.0
        u1, u2 = ground_state
        gamma = abs(model.josephson_gamma)
        potential = (
            model.a1 * f1**2
            + 0.5 * model.b1 * f1**4
            + model.a2 * f2**2
            + 0.5 * model.b2 * f2**4
            - 2 * gamma * f1 * f2
        )
        vacuum_potential = (
            model.a1 * u1**2
            + 0.5 * model.b1 * u1**4
            + model.a2 * u2**2
            + 0.5 * model.b2 * u2**4
            - 2 * gamma * u1 * u2
        )
        energy = float(np.sum(self.node_weights * (potential - vacuum_potential)))
        grad1 = self.node_weights * (
            2 * model.a1 * f1 + 2 * model.b1 * f1**3 - 2 * gamma * f2
        )
        grad2 = self.node_weights * (
            2 * model.a2 * f2 + 2 * model.b2 * f2**3 - 2 * gamma * f1
        )

        cosine = np.cos(qx)
        sine = np.sin(qx)
        left1, right1 = f1[:, :-1], f1[:, 1:]
        left2, right2 = f2[:, :-1], f2[:, 1:]
        weight = self.horizontal_weights
        energy += float(
            np.sum(
                weight
                * (
                    left1**2
                    + right1**2
                    - 2 * left1 * right1 * cosine
                    + model.k2_over_k1
                    * (left2**2 + right2**2 - 2 * left2 * right2 * cosine)
                )
            )
        )
        grad1[:, :-1] += 2 * weight * (left1 - right1 * cosine)
        grad1[:, 1:] += 2 * weight * (right1 - left1 * cosine)
        grad2[:, :-1] += 2 * model.k2_over_k1 * weight * (left2 - right2 * cosine)
        grad2[:, 1:] += 2 * model.k2_over_k1 * weight * (right2 - left2 * cosine)
        grad_qx = (
            2 * weight * (left1 * right1 + model.k2_over_k1 * left2 * right2) * sine
        )

        cosine = np.cos(qy)
        sine = np.sin(qy)
        lower1, upper1 = f1[:-1, :], f1[1:, :]
        lower2, upper2 = f2[:-1, :], f2[1:, :]
        weight = self.vertical_weights
        energy += float(
            np.sum(
                weight
                * (
                    lower1**2
                    + upper1**2
                    - 2 * lower1 * upper1 * cosine
                    + model.k2_over_k1
                    * (lower2**2 + upper2**2 - 2 * lower2 * upper2 * cosine)
                )
            )
        )
        grad1[:-1, :] += 2 * weight * (lower1 - upper1 * cosine)
        grad1[1:, :] += 2 * weight * (upper1 - lower1 * cosine)
        grad2[:-1, :] += 2 * model.k2_over_k1 * weight * (lower2 - upper2 * cosine)
        grad2[1:, :] += 2 * model.k2_over_k1 * weight * (upper2 - lower2 * cosine)
        grad_qy = (
            2 * weight * (lower1 * upper1 + model.k2_over_k1 * lower2 * upper2) * sine
        )

        circulation = qx[:-1, :] + qy[:, 1:] - qx[1:, :] - qy[:, :-1] - winding
        magnetic_factor = 1 / (charge**2 * self.h**2)
        energy += float(magnetic_factor * np.sum(circulation**2))
        magnetic_gradient = 2 * magnetic_factor * circulation
        grad_qx[:-1, :] += magnetic_gradient
        grad_qx[1:, :] -= magnetic_gradient
        grad_qy[:, 1:] += magnetic_gradient
        grad_qy[:, :-1] -= magnetic_gradient

        if core_mask is not None:
            grad1[core_mask] = 0.0
            grad2[core_mask] = 0.0

        gradient = self.pack(grad1, grad2, grad_qx, grad_qy)
        return energy, gradient

    def magnetic_field(
        self, qx: np.ndarray, qy: np.ndarray, winding: np.ndarray, charge: float
    ) -> np.ndarray:
        """Return B on plaquettes; the chosen winding orientation makes B negative."""
        circulation = qx[:-1, :] + qy[:, 1:] - qx[1:, :] - qy[:, :-1] - winding
        return circulation / (charge * self.h**2)

    def relax(
        self,
        parameters: PaperParameters,
        vortices: Sequence[tuple[float, float]],
        *,
        maxiter: int,
        gtol: float,
        initial_state: RelaxedState | None = None,
    ) -> RelaxedState:
        model = parameters.framework_model()
        ground_state = homogeneous_ground_state(model)
        _, _, winding, individual_links = self.phase_links(vortices)
        core_mask = self.vortex_core_mask(vortices)
        if initial_state is None:
            initial = self.initial_fields(
                vortices,
                model,
                parameters.charge,
                ground_state,
                individual_links,
                core_mask,
            )
        else:
            initial = (
                initial_state.f1,
                initial_state.f2,
                initial_state.qx,
                initial_state.qy,
            )
        values = self.pack(*initial)
        amplitude_variables = 2 * self.num_amplitudes
        bounds = [(0.0, None)] * amplitude_variables + [(None, None)] * (
            len(values) - amplitude_variables
        )

        def objective(candidate: np.ndarray) -> tuple[float, np.ndarray]:
            return self.energy_and_gradient(
                candidate,
                winding=winding,
                model=model,
                charge=parameters.charge,
                ground_state=ground_state,
                core_mask=core_mask,
            )

        result = minimize(
            objective,
            values,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxls": 40,
                "ftol": 1e-15,
                "gtol": gtol,
                "maxcor": 20,
            },
        )
        f1, f2, qx, qy = self.unpack(result.x, ground_state)
        f1[core_mask] = 0.0
        f2[core_mask] = 0.0
        _, gradient = objective(result.x)
        gradient_max = float(np.max(np.abs(gradient)))
        converged = bool(result.success) and gradient_max <= max(10 * gtol, 1e-5)
        return RelaxedState(
            f1=f1,
            f2=f2,
            qx=qx,
            qy=qy,
            winding=winding,
            energy=float(result.fun),
            converged=converged,
            iterations=int(result.nit),
            gradient_max=gradient_max,
            message=str(result.message),
        )


def state_signature(
    grid: PinnedVortexGrid,
    parameters: PaperParameters,
    vortices: Sequence[tuple[float, float]],
    maxiter: int,
    gtol: float,
) -> str:
    payload = {
        "schema": 5,
        "width": grid.width,
        "points": grid.points,
        "alpha": parameters.alpha,
        "beta": parameters.beta,
        "eta": parameters.eta,
        "charge": parameters.charge,
        "vortices": list(vortices),
        "maxiter": maxiter,
        "gtol": gtol,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def save_state(path: Path, state: RelaxedState, signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            f1=state.f1,
            f2=state.f2,
            qx=state.qx,
            qy=state.qy,
            winding=state.winding,
            energy=state.energy,
            converged=int(state.converged),
            iterations=state.iterations,
            gradient_max=state.gradient_max,
            message=state.message,
            signature=signature,
        )
    temporary.replace(path)


def load_state(path: Path, signature: str) -> RelaxedState | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["signature"].item()) != signature:
            return None
        return RelaxedState(
            f1=data["f1"],
            f2=data["f2"],
            qx=data["qx"],
            qy=data["qy"],
            winding=data["winding"],
            energy=float(data["energy"]),
            converged=bool(data["converged"]),
            iterations=int(data["iterations"]),
            gradient_max=float(data["gradient_max"]),
            message=str(data["message"].item()),
        )


def relax_cached(
    grid: PinnedVortexGrid,
    parameters: PaperParameters,
    vortices: Sequence[tuple[float, float]],
    path: Path,
    *,
    maxiter: int,
    gtol: float,
    overwrite: bool,
    max_passes: int = 3,
) -> RelaxedState:
    signature = state_signature(grid, parameters, vortices, maxiter, gtol)
    cached = None
    if not overwrite:
        cached = load_state(path, signature)
        if cached is not None and cached.converged:
            print(f"  reuse {path.name}: E={cached.energy:.9g}")
            return cached
    state = None if overwrite else cached
    for pass_index in range(1, max_passes + 1):
        state = grid.relax(
            parameters,
            vortices,
            maxiter=maxiter,
            gtol=gtol,
            initial_state=state,
        )
        save_state(path, state, signature)
        status = "converged" if state.converged else "not converged"
        print(
            f"  {path.name} pass {pass_index}: E={state.energy:.9g}, {status}, "
            f"iterations={state.iterations}, |grad|inf={state.gradient_max:.3g}"
        )
        if state.converged:
            break
    assert state is not None
    return state


MEASUREMENT_FIELDS = (
    "figure",
    "curve",
    "alpha",
    "beta",
    "eta",
    "charge",
    "paper_density_ratio",
    "paper_total_density",
    "computed_density_ratio",
    "computed_total_density",
    "u1",
    "u2",
    "separation",
    "center_single_vortex_energy",
    "matched_single_vortex_energy",
    "two_vortex_energy",
    "interaction_energy",
    "interaction_over_2Ev",
    "single_converged",
    "matched_single_converged",
    "pair_converged",
    "single_iterations",
    "matched_single_iterations",
    "pair_iterations",
    "single_gradient_max",
    "matched_single_gradient_max",
    "pair_gradient_max",
    "pair_flux",
    "expected_pair_flux",
    "width",
    "points",
    "spacing",
)


def write_measurements(path: Path, rows: Sequence[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_interaction_curve(
    grid: PinnedVortexGrid,
    parameters: PaperParameters,
    distances: Iterable[float],
    output_directory: Path,
    *,
    maxiter: int,
    gtol: float,
    overwrite: bool,
) -> tuple[list[dict], dict[float, RelaxedState]]:
    print(f"Figure {parameters.figure}, curve {parameters.curve}: {parameters.label}")
    model = parameters.framework_model()
    u1, u2 = homogeneous_ground_state(model)
    cache_directory = output_directory / "states" / parameters.slug
    single = relax_cached(
        grid,
        parameters,
        [(0.0, 0.0)],
        cache_directory / "single.npz",
        maxiter=maxiter,
        gtol=gtol,
        overwrite=overwrite,
    )
    if single.energy <= 0:
        raise RuntimeError("The relaxed single-vortex excess energy is not positive.")

    rows = []
    pair_states = {}
    for separation in distances:
        separation = float(separation)
        vortices = [(-separation / 2, 0.0), (separation / 2, 0.0)]
        reference_vortex = [(separation / 2, 0.0)]
        reference_filename = f"single_x_{separation / 2:.6f}".replace(".", "p") + ".npz"
        matched_single = relax_cached(
            grid,
            parameters,
            reference_vortex,
            cache_directory / reference_filename,
            maxiter=maxiter,
            gtol=gtol,
            overwrite=overwrite,
        )
        filename = f"pair_d_{separation:.6f}".replace(".", "p") + ".npz"
        pair = relax_cached(
            grid,
            parameters,
            vortices,
            cache_directory / filename,
            maxiter=maxiter,
            gtol=gtol,
            overwrite=overwrite,
        )
        pair_states[separation] = pair
        # Referencing an isolated vortex at the same sub-grid position cancels
        # lattice pinning and finite-boundary offsets from the weak interaction.
        interaction = pair.energy - 2 * matched_single.energy
        magnetic_field = grid.magnetic_field(
            pair.qx, pair.qy, pair.winding, parameters.charge
        )
        rows.append(
            {
                "figure": parameters.figure,
                "curve": parameters.curve,
                "alpha": parameters.alpha,
                "beta": parameters.beta,
                "eta": parameters.eta,
                "charge": parameters.charge,
                "paper_density_ratio": parameters.density_ratio,
                "paper_total_density": parameters.total_density,
                "computed_density_ratio": (u2 / u1) ** 2,
                "computed_total_density": u1**2 + u2**2,
                "u1": u1,
                "u2": u2,
                "separation": separation,
                "center_single_vortex_energy": single.energy,
                "matched_single_vortex_energy": matched_single.energy,
                "two_vortex_energy": pair.energy,
                "interaction_energy": interaction,
                "interaction_over_2Ev": interaction / (2 * matched_single.energy),
                "single_converged": single.converged,
                "matched_single_converged": matched_single.converged,
                "pair_converged": pair.converged,
                "single_iterations": single.iterations,
                "matched_single_iterations": matched_single.iterations,
                "pair_iterations": pair.iterations,
                "single_gradient_max": single.gradient_max,
                "matched_single_gradient_max": matched_single.gradient_max,
                "pair_gradient_max": pair.gradient_max,
                "pair_flux": float(np.sum(magnetic_field) * grid.h**2),
                "expected_pair_flux": -4 * np.pi / parameters.charge,
                "width": grid.width,
                "points": grid.points,
                "spacing": grid.h,
            }
        )
    return rows, pair_states


def plot_interaction_figure(
    figure_number: int,
    rows: Sequence[dict],
    output_directory: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    curves = FIGURE_CURVES[figure_number]
    if figure_number == 1:
        markers = ("o", "*", "D", "o")
        colors = ("#2354ff", "#f04444", "#18c83d", "#ef4ae0")
        linestyles = ("-", "-", "-", "-")
    else:
        markers = (None, "^", "D", "o")
        colors = ("#2354ff", "#222222", "#18c83d", "#ef4ae0")
        linestyles = ("--", "-", "-", "-")
    scale = 1000 if figure_number in (1, 4) else 1
    for parameters, marker, color, linestyle in zip(
        curves, markers, colors, linestyles
    ):
        selected = sorted(
            (
                row
                for row in rows
                if row["figure"] == figure_number and row["curve"] == parameters.curve
            ),
            key=lambda row: row["separation"],
        )
        if not selected:
            continue
        ax.plot(
            [row["separation"] for row in selected],
            [scale * row["interaction_over_2Ev"] for row in selected],
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            markersize=5,
            label=parameters.label,
        )
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.set_xlabel(r"intervortex distance / $\sqrt{2}\xi_1$")
    if scale == 1000:
        ax.set_ylabel(r"$10^3 V_{\rm int}/(2E_v)$")
    else:
        ax.set_ylabel(r"$V_{\rm int}/(2E_v)$")
    ax.set_title(f"PRL 105, 067003 (2010), Fig. {figure_number} reproduction")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    output_path = output_directory / f"figure{figure_number}_interaction.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_figure5(
    grid: PinnedVortexGrid,
    states: dict[float, RelaxedState],
    output_directory: Path,
    parameters: PaperParameters,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    center_lower = grid.points // 2 - 1
    center_upper = grid.points // 2
    cell_x = 0.5 * (grid.x[:-1] + grid.x[1:])
    cell_center = (grid.points - 2) // 2
    panel_names = ("a", "b", "c")
    for ax, panel, separation in zip(axes, panel_names, FIGURE5_DISTANCES):
        state = states[float(separation)]
        f1_line = 0.5 * (state.f1[center_lower, :] + state.f1[center_upper, :])
        f2_line = 0.5 * (state.f2[center_lower, :] + state.f2[center_upper, :])
        magnetic = -grid.magnetic_field(
            state.qx, state.qy, state.winding, parameters.charge
        )
        ax.plot(grid.x, f1_line, color="#10c83a", label=r"1) $|\psi_1|$")
        ax.plot(
            grid.x,
            f2_line,
            color="#ff3b30",
            linestyle="--",
            label=r"2) $|\psi_2|$",
        )
        ax.plot(
            cell_x,
            magnetic[cell_center, :],
            color="#2457ff",
            linestyle=":",
            linewidth=1.8,
            label="3) B",
        )
        ax.set_xlim(-8, 8)
        ax.set_xlabel(r"$x/\sqrt{2}\xi_1$")
        ax.set_title(f"{panel}) separation {separation:.2f}")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("field amplitude")
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(
        "PRL 105, 067003 (2010), Fig. 5 reproduction\n"
        + rf"$\alpha={parameters.alpha:g}$, $\beta={parameters.beta:g}$, "
        + rf"$\eta={parameters.eta:g}$, $e={parameters.charge:g}$"
    )
    fig.tight_layout()
    output_path = output_directory / "figure5_profiles.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def parse_figures(specification: str) -> tuple[int, ...]:
    figures = tuple(sorted({int(value) for value in specification.split(",")}))
    if not figures or any(figure not in (1, 4, 5) for figure in figures):
        raise argparse.ArgumentTypeError(
            "figures must be a comma-separated subset of 1,4,5"
        )
    return figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--figures", type=parse_figures, default=(1, 4, 5))
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--width", type=float, default=28.0)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--gtol", type=float, default=2e-6)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY_ROOT / "results/reproductions/babaev_prl_105_067003",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Ignore matching cached states."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one coarse Fig. 4 curve at two separations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive.")
    if args.gtol <= 0:
        raise ValueError("--gtol must be positive.")

    figures = args.figures
    points = args.points
    width = args.width
    maxiter = args.maxiter
    selected_curves = {figure: FIGURE_CURVES.get(figure, ()) for figure in figures}
    if args.smoke_test:
        figures = (4,)
        points = 24
        width = 18.0
        maxiter = min(maxiter, 80)
        selected_curves = {4: (FIGURE_CURVES[4][1],)}

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    grid = PinnedVortexGrid(width=width, points=points)
    rows: list[dict] = []
    created_plots: list[Path] = []

    for figure in figures:
        if figure == 5:
            continue
        distances = FIGURE1_DISTANCES if figure == 1 else DEFAULT_DISTANCES
        if args.smoke_test:
            distances = SQRT2 * np.asarray([2, 6], dtype=float)
        for parameters in selected_curves[figure]:
            curve_rows, _ = run_interaction_curve(
                grid,
                parameters,
                distances,
                output_directory,
                maxiter=maxiter,
                gtol=args.gtol,
                overwrite=args.overwrite,
            )
            rows.extend(curve_rows)
            write_measurements(output_directory / "measurements.csv", rows)
        created_plots.append(plot_interaction_figure(figure, rows, output_directory))

    if 5 in figures:
        parameters = FIGURE_CURVES[4][1]
        _, states = run_interaction_curve(
            grid,
            parameters,
            FIGURE5_DISTANCES,
            output_directory,
            maxiter=maxiter,
            gtol=args.gtol,
            overwrite=args.overwrite,
        )
        created_plots.append(plot_figure5(grid, states, output_directory, parameters))

    if rows:
        write_measurements(output_directory / "measurements.csv", rows)
    metadata = {
        "paper_doi": PAPER_DOI,
        "figures": figures,
        "points": points,
        "width": width,
        "spacing": grid.h,
        "maxiter": maxiter,
        "gtol": args.gtol,
        "smoke_test": args.smoke_test,
        "normalization": "interaction_energy / (2 * single_vortex_energy)",
        "plots": [str(path) for path in created_plots],
    }
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote results to {output_directory}")
    for path in created_plots:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
