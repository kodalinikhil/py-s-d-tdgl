"""Results and HDF5 I/O for the magnetic-periodic backend.

This module deliberately does not inherit from :class:`tdgl.Solution`.  The
legacy solution type assumes an unstructured ``Device.mesh`` at many points in
its public API, whereas a magnetic-periodic cell has neither a physical
boundary nor a triangular mesh.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from ..solver.options import SolverOptions

HDF5_BACKEND = "magnetic_periodic"
LEGACY_HDF5_SCHEMA_VERSION = 1
GENERIC_COMPONENT_HDF5_SCHEMA_VERSION = 2
LATEST_HDF5_SCHEMA_VERSION = GENERIC_COMPONENT_HDF5_SCHEMA_VERSION
# Public writer version.  Keep the explicit legacy constant above for readers
# and fixtures that intentionally exercise schema 1.
HDF5_SCHEMA_VERSION = LATEST_HDF5_SCHEMA_VERSION
SUPPORTED_HDF5_SCHEMA_VERSIONS = frozenset(
    {LEGACY_HDF5_SCHEMA_VERSION, GENERIC_COMPONENT_HDF5_SCHEMA_VERSION}
)

COMPONENT_NAMES_ATTRIBUTE = "component_names"
GENERIC_COMPONENT_DATASETS = ("psi1", "psi2")
LEGACY_COMPONENT_DATASETS = ("psi_d", "psi_s")
_COMMON_FRAME_DATASETS = (
    "vector_potential",
    "supercurrent",
    "normal_current",
    "epsilon",
)
_FRAME_METADATA_ATTRIBUTES = {"step", "time", "dt", COMPONENT_NAMES_ATTRIBUTE}


def component_names_for_model(model) -> Tuple[str, ...]:
    """Return canonical component names in the model's public component order.

    The ordering agrees with the open-backend ``psi1``/``psi2`` convention:
    ``(s, d)`` for s+d, ``(d, d_prime)`` for d+d', and ``(s1, s2)`` for s+s.
    The import is local to keep the HDF5 layer independent of model modules at
    import time.
    """
    from ..device.models import (
        DPlusDPrimeModel,
        SingleBandModel,
        SPlusDModel,
        SPlusSModel,
    )

    if isinstance(model, SingleBandModel):
        return ("psi",)
    if isinstance(model, SPlusDModel):
        return ("s", "d")
    if isinstance(model, DPlusDPrimeModel):
        return ("d", "d_prime")
    if isinstance(model, SPlusSModel):
        return ("s1", "s2")
    raise TypeError(
        "Unsupported magnetic-periodic model for component metadata: "
        f"{type(model).__name__}."
    )


def _validate_component_names(names: Sequence[str]) -> Tuple[str, ...]:
    names = tuple(str(_decode(name)).strip() for name in names)
    if not 1 <= len(names) <= len(GENERIC_COMPONENT_DATASETS):
        raise ValueError("Magnetic-periodic frames support one or two components.")
    if any(not name for name in names):
        raise ValueError("Component names must be nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Component names must be unique.")
    return names


def write_frame_components(
    group: h5py.Group,
    components: Sequence[np.ndarray],
    *,
    component_names: Optional[Sequence[str]] = None,
    model=None,
) -> Tuple[str, ...]:
    """Write the model-neutral component portion of a schema-v2 frame.

    This is the writer contract for magnetic-periodic HDF5 schema version 2.
    The file header must use ``LATEST_HDF5_SCHEMA_VERSION``.  Callers provide
    either ``model`` or explicit canonical ``component_names``; providing both
    is allowed only when they agree.  Components are written as ``psi1`` and,
    for a two-component model, ``psi2`` with names recorded on the frame.

    Schema-v1 readers continue to accept the historical ``psi_d``/``psi_s``
    layout through ``LEGACY_HDF5_SCHEMA_VERSION``.
    """
    if not isinstance(group, h5py.Group):
        raise TypeError("group must be an open h5py.Group.")
    arrays = tuple(np.asarray(component) for component in components)
    if model is None and component_names is None:
        raise ValueError("Provide model or component_names when writing components.")
    model_names = component_names_for_model(model) if model is not None else None
    names = _validate_component_names(
        model_names if component_names is None else component_names
    )
    if model_names is not None and names != model_names:
        raise ValueError(
            f"Component names {names!r} do not match {type(model).__name__} "
            f"ordering {model_names!r}."
        )
    if len(arrays) != len(names):
        raise ValueError(
            f"Received {len(arrays)} component arrays for names {names!r}."
        )
    if not arrays:
        raise ValueError("At least one component array is required.")
    expected_shape = arrays[0].shape
    if len(expected_shape) != 2 or any(
        array.shape != expected_shape for array in arrays
    ):
        raise ValueError("Component arrays must share one two-dimensional shape.")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Component arrays must contain only finite values.")
    occupied = set(GENERIC_COMPONENT_DATASETS + LEGACY_COMPONENT_DATASETS).intersection(
        group
    )
    if occupied:
        raise ValueError(
            "Frame already contains component datasets: " + ", ".join(sorted(occupied))
        )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.attrs.create(
        COMPONENT_NAMES_ATTRIBUTE,
        np.asarray(names, dtype=string_dtype),
        dtype=string_dtype,
    )
    for dataset_name, array in zip(GENERIC_COMPONENT_DATASETS, arrays):
        group[dataset_name] = array
    return names


def _decode(value):
    """Decode HDF5 byte strings without changing ordinary scalar values."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cell_from_hdf5(group: h5py.Group):
    # Imported lazily so that the grid/operator module can evolve without
    # creating an import cycle through the public package namespace.
    from .cell import MagneticPeriodicCell

    return MagneticPeriodicCell.from_hdf5(group)


def _operators_for(cell):
    from .operators import MagneticPeriodicOperators

    return MagneticPeriodicOperators(cell)


def _sorted_frame_keys(data: h5py.Group) -> Tuple[str, ...]:
    """Return numeric frame keys in increasing order."""
    try:
        numbered = sorted((int(key), key) for key in data)
    except ValueError as exc:
        raise IOError("Magnetic-periodic frame names must be integers.") from exc
    if [number for number, _ in numbered] != list(range(len(numbered))):
        raise IOError("Magnetic-periodic frame indices must be contiguous from zero.")
    return tuple(key for _, key in numbered)


def options_to_json(options: SolverOptions) -> str:
    """Serialize ``SolverOptions`` including ``None`` and enum values."""
    import dataclasses
    from enum import Enum

    values = dataclasses.asdict(options)
    for key, value in tuple(values.items()):
        if isinstance(value, Enum):
            values[key] = value.value
        elif isinstance(value, np.generic):
            values[key] = value.item()
    return json.dumps(values, sort_keys=True)


def options_from_json(payload: Union[str, bytes]) -> SolverOptions:
    """Load ``SolverOptions`` written by :func:`options_to_json`."""
    values = json.loads(_decode(payload))
    options = SolverOptions(**values)
    options.validate()
    return options


@dataclass(frozen=True)
class MagneticPeriodicFrame:
    """One saved magnetic-periodic solver state, in dimensionless units.

    ``components`` are stored in the model's public component order and named
    by ``component_names``.  Prefer :meth:`get_component` in model-neutral
    code.  The named properties retain the historical s+d API and expose
    unambiguous d+d' and s+s accessors.
    """

    step: int
    time: float
    dt: float
    components: Tuple[np.ndarray, ...]
    component_names: Tuple[str, ...]
    vector_potential: np.ndarray
    supercurrent: np.ndarray
    normal_current: np.ndarray
    epsilon: np.ndarray
    state: Dict[str, Union[bool, float, int]]

    @property
    def psi1(self) -> np.ndarray:
        """First component in the owning model's public component order."""
        return self.components[0]

    @property
    def psi2(self) -> np.ndarray:
        """Second component in the owning model's public component order."""
        if len(self.components) < 2:
            raise AttributeError("This magnetic-periodic frame has one component.")
        return self.components[1]

    @property
    def component_map(self) -> Mapping[str, np.ndarray]:
        """Read-only mapping from canonical component name to array."""
        return MappingProxyType(dict(zip(self.component_names, self.components)))

    def get_component(self, name: str) -> np.ndarray:
        """Return a component by canonical name or a documented common alias."""
        if not isinstance(name, str) or not name.strip():
            raise KeyError("Component name must be a nonempty string.")
        normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "d'": "d_prime",
            "dprime": "d_prime",
            "d_xy": "d_prime",
            "s_1": "s1",
            "s_2": "s2",
            "component1": self.component_names[0],
            "psi1": self.component_names[0],
        }
        if len(self.component_names) > 1:
            aliases.update(
                component2=self.component_names[1],
                psi2=self.component_names[1],
            )
        normalized = aliases.get(normalized, normalized)
        try:
            index = self.component_names.index(normalized)
        except ValueError as exc:
            raise KeyError(
                f"Unknown component {name!r}; available names are "
                f"{self.component_names!r}."
            ) from exc
        return self.components[index]

    @property
    def psi(self) -> np.ndarray:
        """Single-band order parameter."""
        return self.get_component("psi")

    @property
    def psi_d(self) -> np.ndarray:
        """d-wave component for s+d and d+d' models."""
        return self.get_component("d")

    @property
    def psi_s(self) -> np.ndarray:
        """s-wave component for the s+d model (legacy Li API)."""
        return self.get_component("s")

    @property
    def psi_d_prime(self) -> np.ndarray:
        """Subdominant d_xy component for the d+d' model."""
        return self.get_component("d_prime")

    @property
    def psi_s1(self) -> np.ndarray:
        """First s-wave component for the s+s model."""
        return self.get_component("s1")

    @property
    def psi_s2(self) -> np.ndarray:
        """Second s-wave component for the s+s model."""
        return self.get_component("s2")

    @property
    def induced_vector_potential(self) -> np.ndarray:
        """Alias for the periodic vector-potential correction."""
        return self.vector_potential

    @classmethod
    def from_hdf5(
        cls,
        group: h5py.Group,
        *,
        cell=None,
        schema_version: Optional[int] = None,
    ) -> "MagneticPeriodicFrame":
        required_attrs = {"step", "time", "dt"}
        missing_attrs = sorted(required_attrs.difference(group.attrs))
        missing_data = sorted(set(_COMMON_FRAME_DATASETS).difference(group))
        if missing_attrs or missing_data:
            raise IOError(
                "Magnetic-periodic frame is incomplete; missing "
                f"attrs={missing_attrs}, datasets={missing_data}."
            )
        step = int(group.attrs["step"])
        time_value = float(group.attrs["time"])
        dt = float(group.attrs["dt"])
        if step < 0 or not np.isfinite(time_value) or not np.isfinite(dt) or dt < 0:
            raise IOError("Magnetic-periodic frame has invalid step/time metadata.")

        if schema_version is None:
            schema_version = (
                GENERIC_COMPONENT_HDF5_SCHEMA_VERSION
                if GENERIC_COMPONENT_DATASETS[0] in group
                else LEGACY_HDF5_SCHEMA_VERSION
            )
        if schema_version not in SUPPORTED_HDF5_SCHEMA_VERSIONS:
            raise IOError(
                f"Unsupported magnetic-periodic frame schema {schema_version}."
            )

        if schema_version == LEGACY_HDF5_SCHEMA_VERSION:
            if cell is not None and component_names_for_model(cell.layer.model) != (
                "s",
                "d",
            ):
                raise IOError(
                    "Legacy magnetic-periodic schema 1 is valid only for "
                    "SPlusDModel files."
                )
            missing_components = sorted(
                set(LEGACY_COMPONENT_DATASETS).difference(group)
            )
            if missing_components:
                raise IOError(
                    "Legacy magnetic-periodic frame is incomplete; missing "
                    f"datasets={missing_components}."
                )
            # Schema v1 was an s+d-only backend whose public component order is
            # psi1=s, psi2=d, despite the named on-disk dataset order.
            component_names = ("s", "d")
            components = (
                np.asarray(group["psi_s"]),
                np.asarray(group["psi_d"]),
            )
            component_dataset_names = ("psi_s", "psi_d")
        else:
            raw_names = group.attrs.get(COMPONENT_NAMES_ATTRIBUTE)
            if raw_names is None:
                raise IOError(
                    "Unsupported magnetic-periodic HDF5 schema 2 frame layout: "
                    "missing component_names metadata."
                )
            try:
                component_names = _validate_component_names(np.atleast_1d(raw_names))
            except ValueError as exc:
                raise IOError(
                    "Schema-v2 magnetic-periodic frame has invalid component names."
                ) from exc
            expected_names = (
                component_names_for_model(cell.layer.model)
                if cell is not None
                else None
            )
            if expected_names is not None and component_names != expected_names:
                raise IOError(
                    f"Frame component names {component_names!r} do not match "
                    f"{type(cell.layer.model).__name__} ordering {expected_names!r}."
                )
            required_components = GENERIC_COMPONENT_DATASETS[: len(component_names)]
            missing_components = sorted(set(required_components).difference(group))
            unexpected_second = len(component_names) == 1 and "psi2" in group
            if missing_components or unexpected_second:
                detail = missing_components + (
                    ["unexpected psi2"] if unexpected_second else []
                )
                raise IOError(
                    "Schema-v2 magnetic-periodic frame component datasets do not "
                    f"match its metadata: {detail}."
                )
            components = tuple(np.asarray(group[name]) for name in required_components)
            component_dataset_names = required_components

        arrays = {name: np.asarray(group[name]) for name in _COMMON_FRAME_DATASETS}
        if cell is not None:
            expected_site = cell.shape
            expected_link = (2,) + cell.shape
            expected_shapes = {
                "epsilon": expected_site,
                "vector_potential": expected_link,
                "supercurrent": expected_link,
                "normal_current": expected_link,
            }
            for name, expected in expected_shapes.items():
                if arrays[name].shape != expected:
                    raise IOError(
                        f"Magnetic-periodic frame dataset {name!r} has shape "
                        f"{arrays[name].shape}; expected {expected}."
                    )
            for name, component in zip(component_dataset_names, components):
                if component.shape != expected_site:
                    raise IOError(
                        f"Magnetic-periodic frame dataset {name!r} has shape "
                        f"{component.shape}; expected {expected_site}."
                    )
        for name, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise IOError(
                    f"Magnetic-periodic frame dataset {name!r} is non-finite."
                )
        for name, component in zip(component_dataset_names, components):
            if not np.all(np.isfinite(component)):
                raise IOError(
                    f"Magnetic-periodic frame dataset {name!r} is non-finite."
                )
        state = {
            str(key): _decode(value)
            for key, value in group.attrs.items()
            if key not in _FRAME_METADATA_ATTRIBUTES
        }
        return cls(
            step=step,
            time=time_value,
            dt=dt,
            components=components,
            component_names=component_names,
            vector_potential=arrays["vector_potential"],
            supercurrent=arrays["supercurrent"],
            normal_current=arrays["normal_current"],
            epsilon=arrays["epsilon"],
            state=state,
        )


class MagneticPeriodicSolution:
    """A saved trajectory from :class:`MagneticPeriodicSolver`.

    Spatial fields have shape ``(ny, nx)``. Link fields have shape
    ``(2, ny, nx)``, with x-directed links followed by y-directed links along
    axis zero. Operator methods also accept the corresponding flat forms.
    """

    def __init__(
        self,
        *,
        cell,
        options: SolverOptions,
        path: str,
        total_seconds: float = 0.0,
    ):
        self.cell = cell
        self.options = options
        self.path = os.path.abspath(os.fspath(path))
        self.total_seconds = float(total_seconds)
        self._frame_keys: Optional[Tuple[str, ...]] = None
        self._final_frame: Optional[MagneticPeriodicFrame] = None

    @property
    def saved_on_disk(self) -> bool:
        return os.path.exists(self.path)

    def _keys(self) -> Tuple[str, ...]:
        if self._frame_keys is None:
            with h5py.File(self.path, "r") as h5file:
                if "data" not in h5file:
                    raise IOError("Magnetic-periodic solution has no data group.")
                self._frame_keys = _sorted_frame_keys(h5file["data"])
        return self._frame_keys

    @property
    def num_frames(self) -> int:
        return len(self._keys())

    @property
    def times(self) -> np.ndarray:
        with h5py.File(self.path, "r") as h5file:
            return np.asarray(
                [float(h5file["data"][key].attrs["time"]) for key in self._keys()]
            )

    def frame(self, index: int = -1) -> MagneticPeriodicFrame:
        keys = self._keys()
        if not keys:
            raise IOError("Magnetic-periodic solution contains no saved frames.")
        key = keys[index]
        with h5py.File(self.path, "r") as h5file:
            schema_version = int(
                h5file.attrs.get("schema_version", LEGACY_HDF5_SCHEMA_VERSION)
            )
            return MagneticPeriodicFrame.from_hdf5(
                h5file["data"][key],
                cell=self.cell,
                schema_version=schema_version,
            )

    @property
    def final_frame(self) -> MagneticPeriodicFrame:
        if self._final_frame is None:
            self._final_frame = self.frame(-1)
        return self._final_frame

    @property
    def final_time(self) -> float:
        """Final recorded measurement time."""
        return self.final_frame.time

    @property
    def final_step(self) -> int:
        """Final accepted measurement-stage step."""
        return self.final_frame.step

    @property
    def state(self) -> Dict[str, Union[bool, float, int]]:
        """Metadata stored on the final frame."""
        return dict(self.final_frame.state)

    @property
    def tdgl_data(self) -> MagneticPeriodicFrame:
        """Compatibility alias for code that consumes the final solver state."""
        return self.final_frame

    def iter_frames(self) -> Iterator[MagneticPeriodicFrame]:
        """Yield saved frames in increasing simulation-time order."""
        with h5py.File(self.path, "r") as h5file:
            schema_version = int(
                h5file.attrs.get("schema_version", LEGACY_HDF5_SCHEMA_VERSION)
            )
            for key in self._keys():
                yield MagneticPeriodicFrame.from_hdf5(
                    h5file["data"][key],
                    cell=self.cell,
                    schema_version=schema_version,
                )

    @property
    def components(self) -> Tuple[np.ndarray, ...]:
        """Final component arrays in the model's public component order."""
        return self.final_frame.components

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Canonical names corresponding to :attr:`components`."""
        return self.final_frame.component_names

    def get_component(self, name: str) -> np.ndarray:
        """Return a final order-parameter component by name."""
        return self.final_frame.get_component(name)

    @property
    def psi1(self) -> np.ndarray:
        return self.final_frame.psi1

    @property
    def psi2(self) -> np.ndarray:
        return self.final_frame.psi2

    @property
    def psi(self) -> np.ndarray:
        return self.final_frame.psi

    @property
    def psi_d(self) -> np.ndarray:
        return self.final_frame.psi_d

    @property
    def psi_s(self) -> np.ndarray:
        return self.final_frame.psi_s

    @property
    def psi_d_prime(self) -> np.ndarray:
        return self.final_frame.psi_d_prime

    @property
    def psi_s1(self) -> np.ndarray:
        return self.final_frame.psi_s1

    @property
    def psi_s2(self) -> np.ndarray:
        return self.final_frame.psi_s2

    @property
    def vector_potential(self) -> np.ndarray:
        return self.final_frame.vector_potential

    @property
    def mean_induction(self) -> float:
        """Topologically fixed mean induction, in units of ``cell.Bc2``."""
        return float(getattr(self.cell, "mean_induction", self.cell.background_field))

    @property
    def induction(self) -> np.ndarray:
        """Final plaquette induction, including the fixed-flux background."""
        operators = _operators_for(self.cell)
        method = getattr(operators, "induction", None)
        if method is None:
            method = operators.magnetic_field
        return np.asarray(method(self.final_frame.vector_potential))

    @property
    def vortex_count(self) -> int:
        """Net fixed-flux vortex sector of the primary charged field.

        The d component is preferred when present; otherwise component 1 is
        used.  If a core lies exactly on a site, the phase diagnostic is
        undefined, but the magnetic-periodic flux sector remains exact and is
        returned instead.
        """
        operators = _operators_for(self.cell)
        operators.set_vector_potential(self.final_frame.vector_potential)
        frame = self.final_frame
        order_parameter = (
            frame.get_component("d") if "d" in frame.component_names else frame.psi1
        )
        try:
            return int(operators.vortex_count(order_parameter))
        except ValueError as exc:
            if "undefined" not in str(exc):
                raise
            return int(self.cell.flux_quanta)

    def time_averaged_electric_field(
        self,
        start: int = 0,
        stop: int = -1,
    ) -> np.ndarray:
        """Return ``<-dA/dt>`` between two saved frames.

        The two components are spatially averaged over x- and y-directed links.
        The unwrapped link potential is used, which is essential when a uniform
        drive advances a harmonic (zero-curl) vector-potential mode.
        """
        first = self.frame(start)
        last = self.frame(stop)
        duration = last.time - first.time
        if duration <= 0:
            return np.full(2, np.nan)
        operators = _operators_for(self.cell)
        delta = (last.vector_potential - first.vector_potential) / duration
        delta = np.asarray(delta)
        if delta.shape == (2,) + self.cell.shape:
            ax, ay = delta
        else:
            unpack = getattr(operators, "unpack_links", None)
            if unpack is None:
                ax, ay = delta.reshape((2,) + self.cell.shape)
            else:
                ax, ay = unpack(delta.ravel())
        return -np.array([np.mean(ax), np.mean(ay)], dtype=float)

    @property
    def electric_field(self) -> np.ndarray:
        """Time-averaged dimensionless electric field over the saved trajectory."""
        return self.time_averaged_electric_field()

    def electric_block_delta(self) -> np.ndarray:
        """Half the signed first/second-half electric-field difference.

        This mirrors the convergence diagnostic used by the Li reproduction
        workflow. ``nan`` is returned when fewer than three frames were saved.
        """
        if self.num_frames < 3:
            return np.full(2, np.nan)
        midpoint = self.num_frames // 2
        first = self.time_averaged_electric_field(0, midpoint)
        second = self.time_averaged_electric_field(midpoint, -1)
        return 0.5 * (second - first)

    def induction_statistics(self) -> Tuple[float, float]:
        """Temporal mean/std of the spatially averaged induction."""
        operators = _operators_for(self.cell)
        method = getattr(operators, "induction", None)
        if method is None:
            method = operators.magnetic_field
        means = [
            float(np.mean(method(frame.vector_potential)))
            for frame in self.iter_frames()
        ]
        return float(np.mean(means)), float(np.std(means))

    def free_energy_density(
        self,
        *,
        include_magnetic: bool = True,
        applied_field: Optional[float] = None,
    ) -> float:
        r"""Return the final model-specific dimensionless free-energy density.

        With ``applied_field=None``, this is the fixed-B Helmholtz density and
        includes :math:`\kappa^2B^2`. Passing a uniform H instead evaluates the
        corresponding :math:`\kappa^2(B-H)^2` Gibbs diagnostic. A fixed-flux
        trajectory does not determine H by itself.
        """
        from ..device.models import (
            DPlusDPrimeModel,
            SingleBandModel,
            SPlusDModel,
            SPlusSModel,
        )
        from .solver import magnetic_periodic_free_energy_density

        frame = self.final_frame
        common = dict(
            include_magnetic=include_magnetic,
            applied_field=applied_field,
        )
        model = self.cell.layer.model
        if isinstance(model, SingleBandModel):
            first = frame.psi if "psi" in frame.component_names else frame.psi_d
            # The model-neutral dispatcher ignores component 2 for this model.
            second = np.zeros_like(first)
        elif isinstance(model, SPlusDModel):
            # The solver's equation/free-energy dispatch uses physical (d, s)
            # order, while the established public psi1/psi2 API is (s, d).
            first, second = frame.psi_d, frame.psi_s
        elif isinstance(model, DPlusDPrimeModel):
            first = frame.psi_d
            second = (
                frame.psi_d_prime if "d_prime" in frame.component_names else frame.psi_s
            )
        elif isinstance(model, SPlusSModel):
            if frame.component_names == ("s1", "s2"):
                first, second = frame.psi_s1, frame.psi_s2
            else:
                # Compatibility for a schema-v1 file written before the
                # model-neutral component writer was adopted.
                first, second = frame.psi_d, frame.psi_s
        else:
            raise TypeError(
                "Unsupported magnetic-periodic model: " f"{type(model).__name__}."
            )
        return magnetic_periodic_free_energy_density(
            self.cell,
            first,
            second,
            frame.vector_potential,
            frame.epsilon,
            **common,
        )

    def virial_applied_field(self) -> float:
        """Infer uniform H from the final stationary fixed-flux state.

        The two-dimensional periodic virial identity assumes homogeneous GL
        coefficients. A spatial disorder profile is therefore rejected, as
        is the zero-flux sector where the identity cannot be divided by the
        mean induction.
        """
        from ..device.models import (
            DPlusDPrimeModel,
            SingleBandModel,
            SPlusDModel,
            SPlusSModel,
        )
        from .solver import magnetic_periodic_virial_applied_field

        frame = self.final_frame
        model = self.cell.layer.model
        if isinstance(model, SingleBandModel):
            first = frame.psi
            second = np.zeros_like(first)
        elif isinstance(model, SPlusDModel):
            first, second = frame.psi_d, frame.psi_s
        elif isinstance(model, DPlusDPrimeModel):
            first, second = frame.psi_d, frame.psi_d_prime
        elif isinstance(model, SPlusSModel):
            first, second = frame.psi_s1, frame.psi_s2
        else:
            raise TypeError(
                f"Unsupported magnetic-periodic model: {type(model).__name__}."
            )
        return magnetic_periodic_virial_applied_field(
            self.cell,
            first,
            second,
            frame.vector_potential,
            frame.epsilon,
        )

    def to_hdf5(self, path: str) -> None:
        """Copy the incrementally written solution to ``path``."""
        destination = os.path.abspath(os.fspath(path))
        if destination == self.path:
            return
        if os.path.exists(destination):
            raise IOError(f"Path already exists: {destination}.")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(self.path, destination)

    @classmethod
    def from_hdf5(cls, path: str) -> "MagneticPeriodicSolution":
        path = os.path.abspath(os.fspath(path))
        with h5py.File(path, "r") as h5file:
            backend = _decode(h5file.attrs.get("backend", ""))
            if backend != HDF5_BACKEND:
                raise IOError(f"Expected backend={HDF5_BACKEND!r}, found {backend!r}.")
            schema = int(h5file.attrs.get("schema_version", 0))
            if schema not in SUPPORTED_HDF5_SCHEMA_VERSIONS:
                raise IOError(
                    f"Unsupported magnetic-periodic HDF5 schema {schema}; "
                    f"supported versions are "
                    f"{sorted(SUPPORTED_HDF5_SCHEMA_VERSIONS)}."
                )
            if not bool(h5file.attrs.get("complete", False)):
                raise IOError(
                    "Magnetic-periodic checkpoint is incomplete; the solver did "
                    "not finish its final metadata flush."
                )
            if "cell" not in h5file or "options" not in h5file or "data" not in h5file:
                raise IOError(
                    "Magnetic-periodic checkpoint is missing required groups."
                )
            if not h5file["data"]:
                raise IOError("Magnetic-periodic checkpoint contains no saved frames.")
            cell = _cell_from_hdf5(h5file["cell"])
            if "json" not in h5file["options"].attrs:
                raise IOError("Magnetic-periodic checkpoint has no solver options.")
            options = options_from_json(h5file["options"].attrs["json"])
            keys = _sorted_frame_keys(h5file["data"])
            steps = []
            times = []
            last_frame = None
            for key in keys:
                last_frame = MagneticPeriodicFrame.from_hdf5(
                    h5file["data"][key], cell=cell, schema_version=schema
                )
                steps.append(last_frame.step)
                times.append(last_frame.time)
            steps = np.asarray(steps)
            times = np.asarray(times)
            if np.any(np.diff(steps) < 0) or np.any(np.diff(times) < 0):
                raise IOError(
                    "Magnetic-periodic checkpoint frames are not time ordered."
                )
            if "final_step" not in h5file.attrs or "final_time" not in h5file.attrs:
                raise IOError(
                    "Magnetic-periodic checkpoint has no final step/time metadata."
                )
            if last_frame is None:
                raise IOError("Magnetic-periodic checkpoint contains no saved frames.")
            if int(h5file.attrs["final_step"]) != last_frame.step or not np.isclose(
                float(h5file.attrs["final_time"]),
                last_frame.time,
                rtol=0,
                atol=1e-13,
            ):
                raise IOError(
                    "Magnetic-periodic checkpoint final metadata does not match "
                    "its last frame."
                )
            total_seconds = float(h5file.attrs.get("total_seconds", 0.0))
            if not np.isfinite(total_seconds) or total_seconds < 0:
                raise IOError(
                    "Magnetic-periodic checkpoint has invalid runtime metadata."
                )
        return cls(
            cell=cell,
            options=options,
            path=path,
            total_seconds=total_seconds,
        )


__all__ = [
    "COMPONENT_NAMES_ATTRIBUTE",
    "GENERIC_COMPONENT_DATASETS",
    "GENERIC_COMPONENT_HDF5_SCHEMA_VERSION",
    "HDF5_BACKEND",
    "HDF5_SCHEMA_VERSION",
    "LATEST_HDF5_SCHEMA_VERSION",
    "LEGACY_HDF5_SCHEMA_VERSION",
    "MagneticPeriodicFrame",
    "MagneticPeriodicSolution",
    "SUPPORTED_HDF5_SCHEMA_VERSIONS",
    "component_names_for_model",
    "options_from_json",
    "options_to_json",
    "write_frame_components",
]
