import math
import warnings
from dataclasses import dataclass


@dataclass
class SingleBandModel:
    """Standard Time-Dependent Ginzburg-Landau single-band model."""

    gamma: float = 10.0


@dataclass
class SPlusDModel:
    """Dimensionless mixed d+s TDGL model.

    The convention is ``D_i = partial_i - 1j * A_i`` and
    ``D_t = partial_t + 1j * phi``. ``beta_em`` is not absorbed into the
    condensate current returned by the discrete current operator; the solver
    divides that current by ``beta_em`` before the Poisson solve and before
    storing it as a transport current. The single-band KWT ``u`` and ``gamma``
    factors are not part of this model.

    The gradient energy is positive definite only when
    ``eta_s > eta_v ** 2``.
    """

    eta_s: float = 1.0
    eta_v: float = 0.0
    nu: float = -1.0
    tau1: float = 1.0
    tau3: float = 0.0
    tau4: float = 0.0
    beta_em: float = 1.0

    def validate(self) -> None:
        if self.eta_s <= 0:
            raise ValueError("eta_s must be positive.")
        if self.beta_em <= 0:
            raise ValueError("beta_em must be positive.")
        if self.eta_s <= self.eta_v**2:
            raise ValueError("The mixed-gradient energy requires eta_s > eta_v ** 2.")
        if self.tau1 <= 0:
            raise ValueError("tau1 must be positive.")


@dataclass
class DPlusDPrimeModel:
    r"""Dimensionless :math:`d_{x^2-y^2}+d_{xy}` GL model.

    The static free energy is the model of Lei, Aruna, and Wang,
    arXiv:cond-mat/0004227, with the optional orbital Zeeman term derived by
    Wang and Wang, arXiv:cond-mat/9909399. Component 1 is the dominant
    :math:`d_{x^2-y^2}` order parameter and component 2 is the subdominant
    :math:`d_{xy}` order parameter.

    ``relaxation_d`` and ``relaxation_d_prime`` define phenomenological
    gradient-flow rates used to find equilibrium; the source papers derive a
    static GL functional rather than physical TDGL kinetics. The signed
    ``zeeman_coupling`` is :math:`\delta_k` in arXiv:cond-mat/9909399 and
    couples to the local dimensionless induction ``b = B / Bc2``.
    """

    alpha: float = 0.5
    relaxation_d: float = 1.0
    relaxation_d_prime: float = 1.0
    em_coupling: float = 1.0
    zeeman_coupling: float = 0.0

    def validate(self) -> None:
        values = {
            "alpha": self.alpha,
            "relaxation_d": self.relaxation_d,
            "relaxation_d_prime": self.relaxation_d_prime,
            "em_coupling": self.em_coupling,
            "zeeman_coupling": self.zeeman_coupling,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.relaxation_d <= 0 or self.relaxation_d_prime <= 0:
            raise ValueError("relaxation coefficients must be positive.")
        if self.em_coupling <= 0:
            raise ValueError("em_coupling must be positive.")
        if abs(self.zeeman_coupling) >= 1:
            raise ValueError("abs(zeeman_coupling) must be less than 1.")
        if self.alpha > 1:
            warnings.warn(
                "alpha > 1 is outside the paper's subdominant d_xy regime.",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class SPlusSModel:
    r"""Dimensionless isotropic two-band ``s+s`` TDGL model.

    The static free-energy density is

    .. math::

        f = \sum_{n=1}^2\left(a_n|\psi_n|^2
            + \frac{b_n}{2}|\psi_n|^4 + k_n|D\psi_n|^2\right)
            - \gamma_{12}(\psi_1\psi_2^* + \psi_2\psi_1^*),

    with ``k1 = 1`` and ``k2 = k2_over_k1``. The dissipative dynamics
    use ``relaxation1`` and ``relaxation2`` as phenomenological kinetic
    coefficients; those coefficients are not specified by the static GL
    papers. ``em_coupling`` converts the condensate current to the transport
    normalization used by the scalar-potential solver.

    The coefficients are fixed-temperature, dimensionless values. Spatially
    varying ``disorder_epsilon`` is deliberately unsupported for this model
    until its mapping to the two independent quadratic coefficients is made
    explicit.
    """

    a1: float = -1.0
    a2: float = -1.0
    b1: float = 1.0
    b2: float = 1.0
    k2_over_k1: float = 1.0
    josephson_gamma: float = 0.0
    relaxation1: float = 1.0
    relaxation2: float = 1.0
    em_coupling: float = 1.0

    def validate(self) -> None:
        if self.b1 <= 0 or self.b2 <= 0:
            raise ValueError("b1 and b2 must be positive.")
        if self.k2_over_k1 <= 0:
            raise ValueError("k2_over_k1 must be positive.")
        if self.relaxation1 <= 0 or self.relaxation2 <= 0:
            raise ValueError("relaxation1 and relaxation2 must be positive.")
        if self.em_coupling <= 0:
            raise ValueError("em_coupling must be positive.")
