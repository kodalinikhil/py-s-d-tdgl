import copy
import dataclasses
import warnings
from typing import Optional, Union

import h5py

from .models import DPlusDPrimeModel, SingleBandModel, SPlusDModel, SPlusSModel


class Layer:
    """A superconducting thin film.

    Args:
        coherence_length: The superconducting coherence length, :math:`\\xi`.
        london_lambda: The London penetration depth, :math:`\\lambda`.
        thickness: The superconducting film thickness, :math:`d`.
        model: The time-dependent Ginzburg-Landau model to use.
        conductivity: The normal state conductivity of the layer, :math:`\\sigma`.
        u: The single-band KWT order-parameter relaxation ratio, :math:`u`.
            It is used only by ``SingleBandModel``; multicomponent models carry
            their kinetic coefficients in their model parameters.
        gamma: Backwards-compatible shortcut for ``SingleBandModel(gamma=...)``.
            It is ignored, with a warning, for multi-component models.
        z0: The vertical position of the layer.
    """

    def __init__(
        self,
        *,
        london_lambda: float,
        coherence_length: float,
        thickness: float,
        model: Optional[
            Union[SingleBandModel, SPlusDModel, DPlusDPrimeModel, SPlusSModel]
        ] = None,
        conductivity: Union[float, None] = None,
        u: float = 5.79,
        gamma: Optional[float] = None,
        z0: float = 0,
    ):
        self.london_lambda = london_lambda
        self.coherence_length = coherence_length
        self.thickness = thickness
        self.conductivity = conductivity
        self.u = u
        self.z0 = z0
        if model is None:
            self.model = SingleBandModel(
                gamma=SingleBandModel.gamma if gamma is None else gamma
            )
        else:
            self.model = model
            if gamma is not None and not isinstance(model, SingleBandModel):
                warnings.warn(
                    "Layer.gamma is ignored for multi-component models; their "
                    "equations do not use the single-band KWT gamma term.",
                    UserWarning,
                    stacklevel=2,
                )
        validate_model = getattr(self.model, "validate", None)
        if validate_model is not None:
            validate_model()

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
            model=copy.deepcopy(self.model),
            conductivity=self.conductivity,
            u=self.u,
            z0=self.z0,
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
        h5_group.attrs["z0"] = self.z0
        if self.conductivity is not None:
            h5_group.attrs["conductivity"] = self.conductivity
        model_group = h5_group.create_group("model")
        model_group.attrs["type"] = self.model.__class__.__name__
        # Schema 3 adds the s+is quartic, mixed-gradient, disorder, and
        # electromagnetic coefficients. Older schemas load missing fields from
        # their dataclass defaults below.
        model_group.attrs["schema_version"] = 3
        for field in dataclasses.fields(self.model):
            model_group.attrs[field.name] = getattr(self.model, field.name)

    @staticmethod
    def from_hdf5(h5_group: h5py.Group) -> "Layer":
        """Load a :class:`tdgl.Layer` from an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` from which to load the layer.

        Returns:
            A new :class:`tdgl.Layer` instance.
        """
        kwargs = dict(
            london_lambda=h5_group.attrs["london_lambda"],
            coherence_length=h5_group.attrs["coherence_length"],
            thickness=h5_group.attrs["thickness"],
            u=h5_group.attrs.get("u", 5.79),
        )
        if "conductivity" in h5_group.attrs:
            kwargs["conductivity"] = h5_group.attrs["conductivity"]
        if "z0" in h5_group.attrs:
            kwargs["z0"] = h5_group.attrs["z0"]

        if "model" in h5_group:
            model_group = h5_group["model"]
            model_type = model_group.attrs.get("type", "SPlusDModel")
            if isinstance(model_type, bytes):
                model_type = model_type.decode()
            model_classes = {
                "SingleBandModel": SingleBandModel,
                "SPlusDModel": SPlusDModel,
                "DPlusDPrimeModel": DPlusDPrimeModel,
                "SPlusSModel": SPlusSModel,
            }
            if model_type not in model_classes:
                raise ValueError(f"Unknown superconducting model type {model_type!r}.")
            model_cls = model_classes[model_type]

            if model_cls is SPlusDModel and "eta_s" not in model_group.attrs:
                # Migrate the first s+d API to the canonical dimensionless names.
                # The old gamma_1 was the complete cross-density coefficient,
                # whereas the canonical parameter is tau3/2.
                model_kwargs = dict(
                    eta_s=model_group.attrs.get("eta1", 1.0),
                    eta_v=model_group.attrs.get("epsilon", 0.0),
                    nu=model_group.attrs.get("alpha1", -1.0),
                    tau1=model_group.attrs.get("beta1", 1.0),
                    tau3=2 * model_group.attrs.get("gamma_1", 0.0),
                    tau4=model_group.attrs.get("gamma_2", 0.0),
                    beta_em=model_group.attrs.get("beta_em", 1.0),
                )
            elif model_cls is SPlusSModel and "a1" not in model_group.attrs:
                # Migrate the experimental pre-schema s+s representation.
                # It used +alpha_n in the TDGL RHS, 1/mass_ratio_2 as the
                # second-band stiffness, and -gamma_j for the scalar coupling.
                mass_ratio_2 = model_group.attrs.get("mass_ratio_2", 1.0)
                if mass_ratio_2 <= 0:
                    raise ValueError(
                        "Legacy SPlusSModel.mass_ratio_2 must be positive."
                    )
                warnings.warn(
                    "Migrating legacy SPlusSModel coefficients to the canonical "
                    "free-energy convention.",
                    UserWarning,
                    stacklevel=2,
                )
                model_kwargs = dict(
                    a1=-model_group.attrs.get("alpha1", 1.0),
                    a2=-model_group.attrs.get("alpha2", 1.0),
                    b1=model_group.attrs.get("beta1", 1.0),
                    b2=model_group.attrs.get("beta2", 1.0),
                    k2_over_k1=1 / mass_ratio_2,
                    josephson_gamma=-model_group.attrs.get("gamma_j", 0.0),
                    relaxation1=model_group.attrs.get("eta1", 1.0),
                    relaxation2=model_group.attrs.get("eta2", 1.0),
                    em_coupling=model_group.attrs.get("em_coupling", 1.0),
                )
            else:
                model_kwargs = {}
                for field in dataclasses.fields(model_cls):
                    if field.name in model_group.attrs:
                        model_kwargs[field.name] = model_group.attrs[field.name]
            kwargs["model"] = model_cls(**model_kwargs)
        else:
            # Backward compatibility with both upstream single-component files
            # and the first flat-attribute s+d format.
            legacy_s_plus_d_fields = {
                "gamma_d",
                "gamma_s",
                "alpha_d",
                "alpha_s",
                "beta_d",
                "beta_s",
                "gamma_1",
                "gamma_2",
                "epsilon",
            }
            if legacy_s_plus_d_fields.intersection(h5_group.attrs):
                kwargs["model"] = SPlusDModel(
                    eta_s=h5_group.attrs.get("gamma_s", 1.0),
                    eta_v=h5_group.attrs.get("epsilon", 0.0),
                    nu=h5_group.attrs.get("alpha_s", -1.0),
                    tau1=h5_group.attrs.get("beta_s", 1.0),
                    tau3=2 * h5_group.attrs.get("gamma_1", 0.0),
                    tau4=h5_group.attrs.get("gamma_2", 0.0),
                    beta_em=h5_group.attrs.get("beta_em", 1.0),
                )
            else:
                kwargs["model"] = SingleBandModel(
                    gamma=h5_group.attrs.get("gamma", 10.0)
                )

        return Layer(**kwargs)

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
            and self.model == other.model
            and self.z0 == other.z0
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"london_lambda={self.london_lambda}, "
            f"coherence_length={self.coherence_length}, "
            f"thickness={self.thickness}, "
            f"conductivity={self.conductivity}, "
            f"u={self.u}, model={self.model}, "
            f"z0={self.z0}, "
            f")"
        )
