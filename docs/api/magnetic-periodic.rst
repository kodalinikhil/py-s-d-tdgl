.. _api-magnetic-periodic:

*********************
Magnetic-periodic API
*********************

The magnetic-periodic backend evolves a uniform rectangular grid with magnetic
translations and a fixed integer flux sector. It is independent of the
unstructured finite-device solver. See :doc:`../magnetic_periodic` for the
physical convention, supported field modes, and complete examples.

Cell and operators
==================

.. autoclass:: tdgl.MagneticPeriodicCell
    :members:

.. autoclass:: tdgl.MagneticPeriodicOperators
    :members:

Solver
======

.. autofunction:: tdgl.solve_magnetic_periodic

.. autoclass:: tdgl.MagneticPeriodicSolver
    :members:

Solutions and frames
====================

.. autoclass:: tdgl.MagneticPeriodicSolution
    :members:

.. autoclass:: tdgl.MagneticPeriodicFrame
    :members:

Free-energy diagnostics
=======================

These helpers accept dimensionless site and link arrays. Most users can call
:meth:`tdgl.MagneticPeriodicSolution.free_energy_density` or
:meth:`tdgl.MagneticPeriodicSolution.virial_applied_field` instead.

.. autofunction:: tdgl.magnetic_periodic.magnetic_periodic_free_energy_density

.. autofunction:: tdgl.magnetic_periodic.magnetic_periodic_virial_applied_field

.. autofunction:: tdgl.magnetic_periodic.single_band_free_energy_density

.. autofunction:: tdgl.magnetic_periodic.s_plus_d_free_energy_density

.. autofunction:: tdgl.magnetic_periodic.d_plus_d_prime_free_energy_density

.. autofunction:: tdgl.magnetic_periodic.s_plus_s_free_energy_density
