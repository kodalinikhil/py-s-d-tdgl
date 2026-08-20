.. _api-solve:

******
Solver
******

Simulating a finite :class:`tdgl.Device` for a given applied vector potential
and set of bias currents is done with :func:`tdgl.solve`. The selected
``device.layer.model`` determines whether one or two condensates are evolved.
The solver uses the unstructured finite-volume operators described in
:doc:`../background`; model-specific equations are summarized in
:doc:`../models`. Its behavior is controlled by :class:`tdgl.SolverOptions`.

For rectangular magnetic-periodic cells, use
:func:`tdgl.solve_magnetic_periodic` instead. It shares
:class:`tdgl.SolverOptions` where meaningful but has no terminals or physical
boundary; see :doc:`magnetic-periodic`.

The applied vector potential can be specified as a scalar (indicating the vector potential associated with a uniform magnetic field),
a function with signature ``func(x, y, z) -> [Ax, Ay, Az]``, or a :class:`tdgl.Parameter`. The physical units for the
applied vector potential are ``field_units * device.length_units``.

The bias or terminal currents (if any) can be specified as a dictionary like ``terminal_currents = {terminal_name: current}``,
where ``current`` is a ``float`` in units of the specified ``current_units``. For time-dependent applied currents, one can provide
a function with signature ``terminal_currents(time: float) -> {terminal_name: current}``, where ``time`` is the dimensionless time.
In either case, the sum of all terminal currents must be zero at every time step and every terminal in the device must be included 
in the dictionary to ensure current conservation.

.. autofunction:: tdgl.solve

.. autoclass:: tdgl.SolverOptions
    :members:

.. autoclass:: tdgl.TDGLSolver
    :members:

.. autoenum:: tdgl.solver.options.SparseSolver

.. autoclass:: tdgl.Parameter
