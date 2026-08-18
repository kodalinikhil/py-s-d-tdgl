import math
import warnings
from dataclasses import dataclass


def _validate_finite_fields(model, names) -> None:
    for name in names:
        value = getattr(model, name)
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a finite real number.") from exc
        if not finite:
            raise ValueError(f"{name} must be finite.")


@dataclass
class SingleBandModel:
    """Standard Time-Dependent Ginzburg-Landau single-band model."""

    gamma: float = 10.0

    def validate(self) -> None:
        _validate_finite_fields(self, ("gamma",))
        if self.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative.")


@dataclass
class SPlusDModel:
    """Dimensionless mixed d+s TDGL model.

    The convention is ``D_i = partial_i - 1j * A_i`` and
    ``D_t = partial_t + 1j * phi``. ``beta_em`` is the electromagnetic
    relaxation coefficient :math:`\beta` in Goncalves et al. Eq. (19) and is
    not absorbed into the
    condensate current returned by the discrete current operator; the solver
    divides that current by ``beta_em`` before the Poisson solve and before
    storing it as a transport current. :class:`tdgl.Solution` multiplies this
    numerical current by ``beta_em`` when converting back to physical units,
    and terminal currents are divided by the same factor on input. The model
    therefore uses the order-parameter diffusion clock of the source paper
    without changing public physical current units. The single-band KWT ``u``
    and ``gamma`` factors are not part of this model.

    Goncalves et al. use the d-band stiffness to define both the coherence
    length and diffusion clock, so the d-sector kinetic coefficient is the
    implicit unit coefficient. ``relaxation_s`` is an optional positive
    multiplier that decouples the s-sector kinetic coefficient from its
    stiffness ``eta_s``; the Goncalves equations are recovered when it is
    one. ``nu_disorder_coupling`` optionally ties
    the local s-sector quadratic coefficient to ``disorder_epsilon`` through
    ``nu_eff = nu + nu_disorder_coupling * (disorder_epsilon - 1)``. This is
    useful for co-located defects that suppress the two condensates by
    different strengths.

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
    relaxation_s: float = 1.0
    nu_disorder_coupling: float = 0.0

    def validate(self) -> None:
        _validate_finite_fields(
            self,
            (
                "eta_s",
                "eta_v",
                "nu",
                "tau1",
                "tau3",
                "tau4",
                "beta_em",
                "relaxation_s",
                "nu_disorder_coupling",
            ),
        )
        if self.eta_s <= 0:
            raise ValueError("eta_s must be positive.")
        if self.beta_em <= 0:
            raise ValueError("beta_em must be positive.")
        if self.eta_s <= self.eta_v**2:
            raise ValueError("The mixed-gradient energy requires eta_s > eta_v ** 2.")
        if self.tau1 <= 0:
            raise ValueError("tau1 must be positive.")
        if self.relaxation_s <= 0:
            raise ValueError("relaxation_s must be positive.")


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
        _validate_finite_fields(self, values)
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
    r"""Dimensionless isotropic two-component ``s+s`` / ``s+is`` TDGL model.

    The static free-energy density is

    .. math::

        f = \sum_{n=1}^2\left(a_n|\psi_n|^2
            + \frac{b_n}{2}|\psi_n|^4 + k_n|D\psi_n|^2\right)
            - \gamma_{12}(\psi_1\psi_2^* + \psi_2\psi_1^*)
            + \gamma_2\operatorname{Re}(\psi_1^{*2}\psi_2^2)
            + \frac{\gamma_3}{2}|\psi_1|^2|\psi_2|^2
            + k_{12}\left[(D\psi_1)^*\cdot D\psi_2 + \mathrm{c.c.}\right],

    with ``k1 = 1``, ``k2 = k2_over_k1``, ``gamma_2 = phase_gamma2``,
    ``gamma_3 = density_gamma3``, and ``k12 = mixed_gradient_k12``.
    ``josephson_gamma`` is the bilinear intercomponent Josephson coupling;
    ``phase_gamma2`` can instead favor a time-reversal-breaking relative phase.

    The dissipative dynamics use ``relaxation1`` and ``relaxation2`` as
    phenomenological kinetic coefficients; those coefficients are not
    specified by the static GL papers. ``em_coupling`` converts the variational
    condensate current to the transport convention. ``beta_em`` is the positive
    electromagnetic relaxation coefficient and divides the current stored by
    the solver. Local screening requires ``em_coupling = 1`` so that the field
    equation and condensate equations vary the same free energy.

    The effective local quadratic coefficients are
    ``a_i_eff = a_i + disorder_coupling_i * (1 - disorder_epsilon)``. Thus a
    positive disorder coupling suppresses the corresponding component where
    ``disorder_epsilon < 1``. ``b2`` may be zero for a passive
    proximity-induced band only when ``a2`` is positive and the mixed quartic
    sector cannot drive that band unbounded below.
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
    phase_gamma2: float = 0.0
    density_gamma3: float = 0.0
    mixed_gradient_k12: float = 0.0
    beta_em: float = 1.0
    disorder_coupling1: float = 0.0
    disorder_coupling2: float = 0.0

    def validate(self) -> None:
        _validate_finite_fields(
            self,
            (
                "a1",
                "a2",
                "b1",
                "b2",
                "k2_over_k1",
                "josephson_gamma",
                "relaxation1",
                "relaxation2",
                "em_coupling",
                "phase_gamma2",
                "density_gamma3",
                "mixed_gradient_k12",
                "beta_em",
                "disorder_coupling1",
                "disorder_coupling2",
            ),
        )
        if self.b1 <= 0:
            raise ValueError("b1 must be positive.")
        if self.b2 < 0:
            raise ValueError("b2 must be nonnegative.")
        if self.b2 == 0 and self.a2 <= 0:
            raise ValueError(
                "b2 may be zero only when a2 is positive, so that the "
                "second-band potential remains bounded below."
            )
        if self.k2_over_k1 <= 0:
            raise ValueError("k2_over_k1 must be positive.")
        if abs(self.mixed_gradient_k12) >= math.sqrt(self.k2_over_k1):
            raise ValueError(
                "The mixed-gradient energy requires "
                "mixed_gradient_k12 ** 2 < k2_over_k1."
            )
        minimum_mixed_quartic = self.density_gamma3 / 2 - abs(self.phase_gamma2)
        if self.b2 > 0:
            # Multiplying the square roots avoids overflowing when two large,
            # individually finite self-couplings are supplied.
            quartic_bound = math.sqrt(self.b1) * math.sqrt(self.b2)
            if minimum_mixed_quartic <= -quartic_bound:
                raise ValueError(
                    "The quartic energy requires density_gamma3 / 2 - "
                    "abs(phase_gamma2) > -sqrt(b1 * b2)."
                )
        elif minimum_mixed_quartic < 0:
            raise ValueError(
                "When b2 is zero, density_gamma3 / 2 - abs(phase_gamma2) "
                "must be nonnegative."
            )
        if self.relaxation1 <= 0 or self.relaxation2 <= 0:
            raise ValueError("relaxation1 and relaxation2 must be positive.")
        if self.em_coupling <= 0:
            raise ValueError("em_coupling must be positive.")
        if self.beta_em <= 0:
            raise ValueError("beta_em must be positive.")
