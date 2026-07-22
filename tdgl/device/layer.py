from typing import Union

import h5py


class Layer:
    """A superconducting thin film.

    Args:
        london_lambda: The London penetration depth of the film.
        coherence_length: The superconducting coherence length of the film.
        thickness: The thickness of the film.
        conductivity: The normal state conductivity of the superconductor in
            Siemens / length_unit.
        u: The ratio of the relaxation times for the order parameter amplitude
            and phase. This value is 5.79 for dirty superconductors.
        gamma: This parameter quantifies the effect of inelastic phonon-electron
            scattering. :math:`\\gamma` is proportional to the inelastic scattering
            time and the size of the superconducting gap.
        z0: Vertical location of the film.
        gamma_d: Inelastic scattering time scale for the d-wave component.
        gamma_s: Inelastic scattering time scale for the s-wave component.
        alpha_d: Linear free energy coefficient for the d-wave component.
        alpha_s: Linear free energy coefficient for the s-wave component.
        beta_d: Nonlinear free energy coefficient for the d-wave component.
        beta_s: Nonlinear free energy coefficient for the s-wave component.
        gamma_1: Coupling constant for the $|\\psi_d|^2 |\\psi_s|^2$ mixed term.
        gamma_2: Coupling constant for the $\\psi_d^2 \\psi_s^{*2} + c.c.$ mixed term.
        epsilon: Mixed spatial gradient coupling coefficient.
    """

    def __init__(
        self,
        *,
        london_lambda: float,
        coherence_length: float,
        thickness: float,
        conductivity: Union[float, None] = None,
        u: float = 5.79,
        gamma: float = 10.0,
        z0: float = 0,
        gamma_d: float = 1.0,
        gamma_s: float = 1.0,
        alpha_d: float = 1.0,
        alpha_s: float = 1.0,
        beta_d: float = 1.0,
        beta_s: float = 1.0,
        gamma_1: float = 0.0,
        gamma_2: float = 0.0,
        epsilon: float = 0.0,
    ):
        self.london_lambda = london_lambda
        self.coherence_length = coherence_length
        self.thickness = thickness
        self.conductivity = conductivity
        self.u = u
        self.gamma = gamma
        self.z0 = z0
        self.gamma_d = gamma_d
        self.gamma_s = gamma_s
        self.alpha_d = alpha_d
        self.alpha_s = alpha_s
        self.beta_d = beta_d
        self.beta_s = beta_s
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.epsilon = epsilon

    @property
    def Lambda(self) -> float:
        """Effective magnetic penetration depth, :math:`\\Lambda=\\lambda^2/d`."""
        return self.london_lambda**2 / self.thickness

    def copy(self) -> "Layer":
        """Create a deep copy of the :class:`tdgl.Layer`."""
        return Layer(
            london_lambda=self.london_lambda,
            coherence_length=self.coherence_length,
            thickness=self.thickness,
            conductivity=self.conductivity,
            u=self.u,
            gamma=self.gamma,
            z0=self.z0,
            gamma_d=self.gamma_d,
            gamma_s=self.gamma_s,
            alpha_d=self.alpha_d,
            alpha_s=self.alpha_s,
            beta_d=self.beta_d,
            beta_s=self.beta_s,
            gamma_1=self.gamma_1,
            gamma_2=self.gamma_2,
            epsilon=self.epsilon,
        )

    def to_hdf5(self, h5_group: h5py.Group) -> None:
        """Save the :class:`tdgl.Layer` to an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` to which to save the layer.
        """
        h5_group.attrs["london_lambda"] = self.london_lambda
        h5_group.attrs["coherence_length"] = self.coherence_length
        h5_group.attrs["thickness"] = self.thickness
        h5_group.attrs["u"] = self.u
        h5_group.attrs["gamma"] = self.gamma
        h5_group.attrs["z0"] = self.z0
        h5_group.attrs["gamma_d"] = self.gamma_d
        h5_group.attrs["gamma_s"] = self.gamma_s
        h5_group.attrs["alpha_d"] = self.alpha_d
        h5_group.attrs["alpha_s"] = self.alpha_s
        h5_group.attrs["beta_d"] = self.beta_d
        h5_group.attrs["beta_s"] = self.beta_s
        h5_group.attrs["gamma_1"] = self.gamma_1
        h5_group.attrs["gamma_2"] = self.gamma_2
        h5_group.attrs["epsilon"] = self.epsilon
        if self.conductivity is not None:
            h5_group.attrs["conductivity"] = self.conductivity

    @staticmethod
    def from_hdf5(h5_group: h5py.Group) -> "Layer":
        """Load a :class:`tdgl.Layer` from an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` from which to load the layer.

        Returns:
            A new :class:`tdgl.Layer` instance.
        """

        def get(key, default=None):
            if key in h5_group.attrs:
                return h5_group.attrs[key]
            return default

        return Layer(
            london_lambda=get("london_lambda"),
            coherence_length=get("coherence_length"),
            thickness=get("thickness"),
            conductivity=get("conductivity"),
            u=get("u", 5.79),
            gamma=get("gamma", 10.0),
            z0=get("z0", 0.0),
            gamma_d=get("gamma_d", 1.0),
            gamma_s=get("gamma_s", 1.0),
            alpha_d=get("alpha_d", 1.0),
            alpha_s=get("alpha_s", 1.0),
            beta_d=get("beta_d", 1.0),
            beta_s=get("beta_s", 1.0),
            gamma_1=get("gamma_1", 0.0),
            gamma_2=get("gamma_2", 0.0),
            epsilon=get("epsilon", 0.0),
        )

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, Layer):
            return False

        return (
            self.london_lambda == other.london_lambda
            and self.coherence_length == other.coherence_length
            and self.thickness == other.thickness
            and self.conductivity == other.conductivity
            and self.u == other.u
            and self.gamma == other.gamma
            and self.z0 == other.z0
            and self.gamma_d == other.gamma_d
            and self.gamma_s == other.gamma_s
            and self.alpha_d == other.alpha_d
            and self.alpha_s == other.alpha_s
            and self.beta_d == other.beta_d
            and self.beta_s == other.beta_s
            and self.gamma_1 == other.gamma_1
            and self.gamma_2 == other.gamma_2
            and self.epsilon == other.epsilon
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"london_lambda={self.london_lambda}, "
            f"coherence_length={self.coherence_length}, "
            f"thickness={self.thickness}, "
            f"conductivity={self.conductivity}, "
            f"u={self.u}, "
            f"gamma={self.gamma}, "
            f"z0={self.z0}, "
            f"gamma_d={self.gamma_d}, "
            f"gamma_s={self.gamma_s}, "
            f"alpha_d={self.alpha_d}, "
            f"alpha_s={self.alpha_s}, "
            f"beta_d={self.beta_d}, "
            f"beta_s={self.beta_s}, "
            f"gamma_1={self.gamma_1}, "
            f"gamma_2={self.gamma_2}, "
            f"epsilon={self.epsilon}"
            f")"
        )
