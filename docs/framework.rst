.. _framework:

******************
Framework overview
******************

``py-s-d-TDGL`` separates the superconducting equation set from the numerical
domain. A :class:`tdgl.Layer` owns one model, and that layer is placed in either
a finite :class:`tdgl.Device` or a :class:`tdgl.MagneticPeriodicCell`.

Choosing a backend
==================

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Capability
     - Finite-device backend
     - Magnetic-periodic backend
   * - Entry point
     - :func:`tdgl.solve`
     - :func:`tdgl.solve_magnetic_periodic`
   * - Domain
     - Arbitrary polygon with optional holes and a physical boundary
     - Rectangular torus with magnetic-translation boundaries
   * - Discretization
     - Unstructured triangular finite-volume mesh
     - Uniform structured finite-difference grid
   * - Magnetic control
     - Applied vector potential; optional screening where supported
     - Signed integer flux sector; fixed background or local screening
   * - Transport
     - Named current terminals, voltage probes, and time-dependent drives
     - No terminals; optional homogeneous bulk drive with screened s+d or s+s
   * - Output
     - :class:`tdgl.Solution`
     - :class:`tdgl.MagneticPeriodicSolution`
   * - Acceleration
     - CPU sparse solvers and optional CuPy paths
     - CPU and SuperLU only

The backends intentionally use separate solution types. A finite-device seed
cannot initialize a periodic cell, and a periodic seed must describe the
identical cell, grid, model, and flux sector.

Finite devices
==============

A finite simulation is assembled from four layers of objects:

#. :class:`tdgl.Layer` defines physical length scales, conductivity, and the
   selected model.
#. :class:`tdgl.Polygon` instances define the film, holes, and current
   terminals. The geometry helpers in :mod:`tdgl.geometry` create common
   shapes, and polygons support constructive solid geometry.
#. :class:`tdgl.Device` combines the layer and geometry, handles physical
   units, and creates the dimensionless mesh with
   :meth:`tdgl.Device.make_mesh`.
#. :class:`tdgl.SolverOptions` controls integration, convergence, output,
   sparse solves, monitoring, and screening. :func:`tdgl.solve` supplies the
   applied field, currents, disorder, and optional seed.

``applied_vector_potential`` may be a uniform magnetic-field value, a static
callable, a time-dependent callable, or a :class:`tdgl.Parameter`.
``terminal_currents`` may be a current dictionary or a callable returning one;
the currents must sum to zero. ``disorder_epsilon`` may be a scalar or a
spatial callable. See :doc:`api/solver` for the exact signatures.

Magnetic-periodic cells
=======================

A :class:`tdgl.MagneticPeriodicCell` has physical side lengths, a grid shape,
and a signed integer ``flux_quanta``. The solver represents the vector
potential as a quantized background plus a periodic correction. This preserves
the mean flux exactly while allowing local field variations when screening is
enabled. See :doc:`magnetic_periodic` for initialization, component ordering,
free-energy diagnostics, and the magnetic-translation convention.

Fields and screening
====================

With ``include_screening=False``, every model evolves against the prescribed
vector potential. Screening depends on both backend and model:

.. list-table::
   :header-rows: 1
   :widths: 26 32 32

   * - Model
     - Finite device
     - Magnetic-periodic cell
   * - ``SingleBandModel``
     - pyTDGL thin-film Biot--Savart iteration
     - fixed background only
   * - ``SPlusDModel``
     - local bulk electromagnetic relaxation
     - local fixed-flux electromagnetic relaxation
   * - ``DPlusDPrimeModel``
     - fixed prescribed vector potential only
     - fixed background only
   * - ``SPlusSModel``
     - local bulk relaxation when ``em_coupling=1``
     - local fixed-flux relaxation when ``em_coupling=1``

Local multi-component screening is CPU-only. On a finite device it uses a
static applied vector potential during a solve and does not support terminal
current injection. The s+d bulk-drive options provide a homogeneous source
for that local problem.

Integration and persistence
===========================

The solvers support adaptive time steps and can stop before ``solve_time``
when ``equilibrium_tolerance`` is satisfied over ``equilibrium_window`` after
``equilibrium_min_time``. For multi-component states, convergence removes one
optimal shared global phase; it does not independently rotate each component.

Set ``output_file`` to persist saved frames to HDF5. ``save_every`` controls
field-frame cadence. Both solution types can be loaded from HDF5 and used as
seeds within their own backend. The loaders include compatibility handling for
older single-band and early multi-component layouts, but new workflows should
use the model-neutral ``psi1`` and ``psi2`` datasets and the accessors described
in :doc:`api/solution`.

Post-processing
===============

Finite-device :class:`tdgl.Solution` objects expose currents, voltage dynamics,
interpolated fields, fluxoids, magnetic moment, vorticity, component order
parameters, relative phase for s+s, orbital magnetization for d+d', and local
induction from the stored total vector potential. They also integrate with the
snapshot, animation, and interactive visualization tools.

Periodic :class:`tdgl.MagneticPeriodicSolution` objects expose named
components, frames, currents, induction statistics, vortex count,
time-averaged electric field, model-specific free-energy density, and a virial
estimate of the applied field where its assumptions hold.

Physical scope
==============

These are two-dimensional GL models for fields that are effectively constant
through the modeled thickness. The framework does not solve a microscopic gap
equation and does not currently include thermal noise or a heat-balance
equation. The single-band KWT model has a microscopic dissipative
interpretation near :math:`T_c`; the extended models use the static free
energies and phenomenological relaxation coefficients documented in
:doc:`models`.
