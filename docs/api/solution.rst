.. _api-solution:

***************
Post-Processing
***************

The :class:`tdgl.Solution` class contains results from the finite-device
backend, including methods for post-processing and visualization. Calls to
:func:`tdgl.solve` return a :class:`tdgl.Solution`, which can be serialized to
and deserialized from disk. The magnetic-periodic backend returns the separate
:class:`tdgl.MagneticPeriodicSolution`; see :doc:`magnetic-periodic`.

For each instance ``solution`` of :class:`tdgl.Solution`, the raw data from the TDGL simulation (in dimensionless units)
are stored in ``solution.tdgl_data``, which is an instance of :class:`tdgl.solution.data.TDGLData`. Any data
that is measured at each time step in the simulation, i.e., the measured voltage and phase difference between
the :class:`tdgl.Device`'s ``probe_points``, are stored in ``solution.dynamics``, which is an instance of
:class:`tdgl.solution.data.DynamicsData`.

``solution.tdgl_data.psi1`` and ``psi2`` are the model-neutral condensate
arrays. Prefer :meth:`tdgl.Solution.get_order_parameter` for physical component
names. In s+d output, component 1 is s and component 2 is d; in d+d' they are d
and d_prime; in s+s they are the first and second bands. Legacy ``psi``,
``psi_s``, and ``psi_d`` attributes remain compatibility aliases and are not
unambiguous for every model.

Overview
--------

Post-processing methods:

* :meth:`tdgl.Solution.magnetic_moment`
* :meth:`tdgl.Solution.get_order_parameter`
* :attr:`tdgl.Solution.relative_phase`
* :attr:`tdgl.Solution.orbital_magnetization`
* :meth:`tdgl.Solution.local_magnetic_induction`
* :meth:`tdgl.Solution.interp_current_density`
* :meth:`tdgl.Solution.grid_current_density`
* :meth:`tdgl.Solution.interp_order_parameter`
* :meth:`tdgl.Solution.polygon_fluxoid`
* :meth:`tdgl.Solution.hole_fluxoid`
* :meth:`tdgl.Solution.boundary_phases`
* :meth:`tdgl.Solution.current_through_path`
* :func:`tdgl.get_current_through_paths`
* :meth:`tdgl.Solution.field_at_position`
* :meth:`tdgl.Solution.vector_potential_at_position`
* :meth:`tdgl.solution.data.DynamicsData.mean_voltage`

Visualization methods:

* :meth:`tdgl.Solution.plot_currents`
* :meth:`tdgl.Solution.plot_order_parameter`
* :meth:`tdgl.Solution.plot_scalar_potential`
* :meth:`tdgl.Solution.plot_field_at_positions`
* :meth:`tdgl.Solution.plot_vorticity`
* :meth:`tdgl.solution.data.DynamicsData.plot`
* :meth:`tdgl.solution.data.DynamicsData.plot_dt`
* :func:`tdgl.plot_current_through_paths`

I/O methods:

* :meth:`tdgl.Solution.to_hdf5`
* :meth:`tdgl.Solution.from_hdf5`
* :meth:`tdgl.Solution.delete_hdf5`

Solution
--------

.. autoclass:: tdgl.Solution
    :members:

.. autoclass:: tdgl.solution.data.TDGLData
    :members:

.. autoclass:: tdgl.solution.data.DynamicsData
    :members:

.. autoclass:: tdgl.BiotSavartField
    :show-inheritance:

.. autofunction:: tdgl.get_current_through_paths

Fluxoid Quantization
--------------------

.. seealso::

    :meth:`tdgl.Solution.polygon_fluxoid`, :meth:`tdgl.Solution.hole_fluxoid`, :meth:`tdgl.Solution.boundary_phases`

.. autoclass:: tdgl.Fluxoid
    :show-inheritance:

.. autofunction:: tdgl.make_fluxoid_polygons
