"""Structured rectangular cells for magnetic-periodic calculations."""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from ..device.layer import Layer


class MagneticPeriodicCell:
    r"""A uniform rectangular magnetic-periodic cell.

    ``length_x``, ``length_y``, and ``origin`` are physical coordinates in
    ``length_units``.  The grid contains exactly ``nx * ny`` independent,
    cell-centered sites; opposite sides are not duplicated.  Finite-difference
    operators use the corresponding dimensionless lengths in units of the
    layer coherence length.

    Args:
        layer: Superconducting layer and TDGL model parameters.
        lengths: Physical cell lengths ``(Lx, Ly)``.
        shape: Number of independent sites ``(ny, nx)``.
        flux_quanta: Signed integer magnetic-flux sector.
        origin: Physical coordinate of the lower-left cell corner.
        length_units: Units shared by lengths, origin, and ``Layer`` lengths.
        name: Descriptive cell name.

    ``length_x``, ``length_y``, ``nx``, and ``ny`` are supported as explicit
    keyword alternatives to ``lengths`` and ``shape``.  Mixing the tuple and
    explicit forms is rejected so that the grid definition is unambiguous.
    """

    schema_version = 1

    def __init__(
        self,
        *,
        layer: Layer,
        lengths: Optional[Tuple[float, float]] = None,
        shape: Optional[Tuple[int, int]] = None,
        flux_quanta: int,
        origin: Tuple[float, float] = (0.0, 0.0),
        length_units: str = "um",
        name: str = "magnetic_periodic_cell",
        length_x: Optional[float] = None,
        length_y: Optional[float] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
    ):
        if not isinstance(layer, Layer):
            raise TypeError(f"layer must be a Layer, got {type(layer).__name__}.")
        length_x, length_y = self._resolve_lengths(lengths, length_x, length_y)
        ny, nx = self._resolve_shape(shape, ny, nx)
        self.layer = layer
        self.length_x = self._positive_finite("length_x", length_x)
        self.length_y = self._positive_finite("length_y", length_y)
        self.nx = self._grid_size("nx", nx)
        self.ny = self._grid_size("ny", ny)
        if not isinstance(flux_quanta, numbers.Integral) or isinstance(
            flux_quanta, (bool, np.bool_)
        ):
            raise TypeError("flux_quanta must be an integer.")
        self.flux_quanta = int(flux_quanta)

        origin_array = np.asarray(origin, dtype=float)
        if origin_array.shape != (2,) or not np.all(np.isfinite(origin_array)):
            raise ValueError("origin must contain two finite coordinates.")
        self.origin = (float(origin_array[0]), float(origin_array[1]))
        if not isinstance(length_units, str) or not length_units.strip():
            raise ValueError("length_units must be a nonempty string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty string.")
        self.length_units = length_units
        self.name = name

        self._positive_finite("layer.coherence_length", layer.coherence_length)
        self._positive_finite("layer.london_lambda", layer.london_lambda)

    @classmethod
    def _resolve_lengths(
        cls,
        lengths: Optional[Tuple[float, float]],
        length_x: Optional[float],
        length_y: Optional[float],
    ) -> Tuple[float, float]:
        if lengths is not None:
            if length_x is not None or length_y is not None:
                raise ValueError(
                    "Use either lengths=(Lx, Ly) or length_x/length_y, not both."
                )
            values = np.asarray(lengths)
            if values.shape != (2,):
                raise ValueError("lengths must have shape (2,) in (Lx, Ly) order.")
            length_x, length_y = values
        elif length_x is None or length_y is None:
            raise TypeError("Provide lengths=(Lx, Ly) or both length_x and length_y.")
        return (
            cls._positive_finite("length_x", length_x),
            cls._positive_finite("length_y", length_y),
        )

    @classmethod
    def _resolve_shape(
        cls,
        shape: Optional[Tuple[int, int]],
        ny: Optional[int],
        nx: Optional[int],
    ) -> Tuple[int, int]:
        if shape is not None:
            if nx is not None or ny is not None:
                raise ValueError("Use either shape=(ny, nx) or nx/ny, not both.")
            values = np.asarray(shape)
            if values.shape != (2,):
                raise ValueError("shape must have shape (2,) in (ny, nx) order.")
            ny, nx = values
        elif nx is None or ny is None:
            raise TypeError("Provide shape=(ny, nx) or both nx and ny.")
        return cls._grid_size("ny", ny), cls._grid_size("nx", nx)

    @staticmethod
    def _positive_finite(name: str, value: float) -> float:
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive real number.")
        return float(value)

    @staticmethod
    def _grid_size(name: str, value: int) -> int:
        if not isinstance(value, numbers.Integral) or isinstance(
            value, (bool, np.bool_)
        ):
            raise TypeError(f"{name} must be an integer.")
        value = int(value)
        if value < 2:
            raise ValueError(f"{name} must be at least 2.")
        return value

    @property
    def shape(self) -> Tuple[int, int]:
        """Canonical site-array shape ``(ny, nx)``."""
        return self.ny, self.nx

    @property
    def num_sites(self) -> int:
        return self.nx * self.ny

    @property
    def num_edges(self) -> int:
        """Number of directed nearest-neighbor links, ``2 * num_sites``."""
        return 2 * self.num_sites

    @property
    def kappa(self) -> float:
        """Ginzburg--Landau parameter ``lambda / xi``."""
        return self.layer.london_lambda / self.layer.coherence_length

    @property
    def dx(self) -> float:
        """Physical x spacing."""
        return self.length_x / self.nx

    @property
    def dy(self) -> float:
        """Physical y spacing."""
        return self.length_y / self.ny

    @property
    def dimensionless_lengths(self) -> Tuple[float, float]:
        xi = self.layer.coherence_length
        return self.length_x / xi, self.length_y / xi

    @property
    def hx(self) -> float:
        """Dimensionless x spacing in coherence-length units."""
        return self.dimensionless_lengths[0] / self.nx

    @property
    def hy(self) -> float:
        """Dimensionless y spacing in coherence-length units."""
        return self.dimensionless_lengths[1] / self.ny

    @property
    def area(self) -> float:
        """Physical cell area."""
        return self.length_x * self.length_y

    @property
    def dimensionless_area(self) -> float:
        lx, ly = self.dimensionless_lengths
        return lx * ly

    @property
    def background_field(self) -> float:
        r"""Mean dimensionless induction, ``2 pi n / (Lx Ly)``."""
        return 2 * np.pi * self.flux_quanta / self.dimensionless_area

    @property
    def mean_induction(self) -> float:
        """Alias for :attr:`background_field`."""
        return self.background_field

    @property
    def x(self) -> np.ndarray:
        """Physical cell-centered x coordinates."""
        return self.origin[0] + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        """Physical cell-centered y coordinates."""
        return self.origin[1] + (np.arange(self.ny) + 0.5) * self.dy

    @property
    def points(self) -> np.ndarray:
        """Flattened physical site coordinates with shape ``(nx * ny, 2)``."""
        x, y = np.meshgrid(self.x, self.y)
        return np.column_stack((x.ravel(), y.ravel()))

    @property
    def grid_points(self) -> np.ndarray:
        """Physical site coordinates with shape ``(ny, nx, 2)``."""
        return self.points.reshape(*self.shape, 2)

    @property
    def sites(self) -> np.ndarray:
        """Flattened dimensionless site coordinates."""
        return self.points / self.layer.coherence_length

    @property
    def site_grid(self) -> np.ndarray:
        """Dimensionless site coordinates with shape ``(ny, nx, 2)``."""
        return self.sites.reshape(*self.shape, 2)

    @property
    def relative_x(self) -> np.ndarray:
        """Dimensionless cell-centered x coordinates relative to the cell origin."""
        return (np.arange(self.nx) + 0.5) * self.hx

    @property
    def relative_y(self) -> np.ndarray:
        """Dimensionless cell-centered y coordinates relative to the cell origin."""
        return (np.arange(self.ny) + 0.5) * self.hy

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        x0, y0 = self.origin
        return x0, x0 + self.length_x, y0, y0 + self.length_y

    def reshape_site(self, values: Sequence, *, name: str = "values") -> np.ndarray:
        """Return a site array in canonical ``(ny, nx)`` shape."""
        array = np.asarray(values)
        if array.shape == self.shape:
            return array
        if array.shape == (self.num_sites,):
            return array.reshape(self.shape)
        raise ValueError(
            f"{name} must have shape {self.shape} or ({self.num_sites},), "
            f"got {array.shape}."
        )

    def flatten_site(self, values: Sequence, *, name: str = "values") -> np.ndarray:
        """Return a site array in row-major packed shape ``(nx * ny,)``."""
        return self.reshape_site(values, name=name).ravel()

    def to_hdf5(self, h5_group: h5py.Group) -> None:
        """Serialize the cell into an open HDF5 group."""
        h5_group.attrs["schema_version"] = self.schema_version
        h5_group.attrs["name"] = self.name
        h5_group.attrs["length_x"] = self.length_x
        h5_group.attrs["length_y"] = self.length_y
        h5_group.attrs["nx"] = self.nx
        h5_group.attrs["ny"] = self.ny
        h5_group.attrs["flux_quanta"] = self.flux_quanta
        h5_group.attrs["origin"] = self.origin
        h5_group.attrs["length_units"] = self.length_units
        self.layer.to_hdf5(h5_group.create_group("layer"))

    @classmethod
    def from_hdf5(
        cls, path_or_group: Union[str, Path, h5py.File, h5py.Group]
    ) -> "MagneticPeriodicCell":
        """Load a cell from an HDF5 group or file path."""
        if isinstance(path_or_group, (str, Path)):
            with h5py.File(path_or_group, "r") as h5_file:
                return cls.from_hdf5(h5_file)
        if not isinstance(path_or_group, (h5py.File, h5py.Group)):
            raise TypeError("Expected an HDF5 file/group or filesystem path.")
        group = path_or_group
        schema_version = int(group.attrs.get("schema_version", 0))
        if schema_version != cls.schema_version:
            raise IOError(
                "Unsupported magnetic-periodic cell schema "
                f"{schema_version}; expected {cls.schema_version}."
            )
        required = {
            "length_x",
            "length_y",
            "nx",
            "ny",
            "flux_quanta",
            "origin",
            "length_units",
        }
        missing = sorted(required.difference(group.attrs))
        if missing or "layer" not in group:
            detail = missing + ([] if "layer" in group else ["layer"])
            raise IOError(f"Cannot load magnetic-periodic cell; missing {detail}.")

        def decode(value):
            return value.decode() if isinstance(value, bytes) else value

        return cls(
            layer=Layer.from_hdf5(group["layer"]),
            lengths=(group.attrs["length_x"], group.attrs["length_y"]),
            shape=(group.attrs["ny"], group.attrs["nx"]),
            flux_quanta=group.attrs["flux_quanta"],
            origin=tuple(group.attrs["origin"]),
            length_units=decode(group.attrs["length_units"]),
            name=decode(group.attrs.get("name", "magnetic_periodic_cell")),
        )

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, MagneticPeriodicCell):
            return False
        return (
            self.layer == other.layer
            and self.length_x == other.length_x
            and self.length_y == other.length_y
            and self.nx == other.nx
            and self.ny == other.ny
            and self.flux_quanta == other.flux_quanta
            and np.allclose(self.origin, other.origin, rtol=0, atol=0)
            and self.length_units == other.length_units
            and self.name == other.name
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, layer={self.layer!r}, "
            f"length_x={self.length_x!r}, length_y={self.length_y!r}, "
            f"nx={self.nx}, ny={self.ny}, flux_quanta={self.flux_quanta}, "
            f"origin={self.origin!r}, length_units={self.length_units!r})"
        )
