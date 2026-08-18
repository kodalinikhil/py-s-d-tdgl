"""Standalone structured-grid magnetic-periodic TDGL backend."""

from .cell import MagneticPeriodicCell
from .operators import MagneticPeriodicOperators
from .solution import MagneticPeriodicFrame, MagneticPeriodicSolution
from .solver import (
    MagneticPeriodicSolver,
    d_plus_d_prime_free_energy_density,
    magnetic_periodic_free_energy_density,
    magnetic_periodic_virial_applied_field,
    s_plus_d_free_energy_density,
    s_plus_d_virial_applied_field,
    s_plus_s_free_energy_density,
    single_band_free_energy_density,
    solve_magnetic_periodic,
)

__all__ = [
    "MagneticPeriodicCell",
    "MagneticPeriodicFrame",
    "MagneticPeriodicOperators",
    "MagneticPeriodicSolution",
    "MagneticPeriodicSolver",
    "d_plus_d_prime_free_energy_density",
    "magnetic_periodic_free_energy_density",
    "magnetic_periodic_virial_applied_field",
    "single_band_free_energy_density",
    "s_plus_d_free_energy_density",
    "s_plus_d_virial_applied_field",
    "s_plus_s_free_energy_density",
    "solve_magnetic_periodic",
]
