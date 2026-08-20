.. _models:

**********************
Order-parameter models
**********************

The model stored on :class:`tdgl.Layer` selects the order-parameter components,
free energy, kinetic coefficients, current operator, initialization, and
model-specific diagnostics. Omitting ``model`` creates a
:class:`tdgl.SingleBandModel`.

.. code-block:: python

   import tdgl

   layer = tdgl.Layer(
       coherence_length=0.05,
       london_lambda=0.20,
       thickness=0.01,
       conductivity=1,
       model=tdgl.SPlusDModel(eta_s=1.2, eta_v=0.25),
   )

Component conventions
=====================

Finite-device output is stored as ``psi1`` and ``psi2``. Use
:meth:`tdgl.Solution.get_order_parameter` rather than assuming a physical name
from the array position:

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Model
     - Component 1
     - Component 2
     - Model diagnostic
   * - ``SingleBandModel``
     - ``psi``
     - zero placeholder
     - standard order-parameter tools
   * - ``SPlusDModel``
     - ``s``
     - ``d``
     - component-resolved fields
   * - ``DPlusDPrimeModel``
     - ``d``
     - ``d_prime``
     - :attr:`tdgl.Solution.orbital_magnetization`
   * - ``SPlusSModel``
     - ``psi1`` / ``s1``
     - ``psi2`` / ``s2``
     - :attr:`tdgl.Solution.relative_phase`

The magnetic-periodic API records canonical component names with each frame
and exposes :meth:`tdgl.MagneticPeriodicSolution.get_component`. Its canonical
orders are ``("psi",)``, ``("s", "d")``, ``("d", "d_prime")``, and
``("s1", "s2")`` respectively.

SingleBandModel
===============

:class:`tdgl.SingleBandModel` is the pyTDGL Kramer--Watts--Tobin equation. Its
``gamma`` parameter controls the generalized relaxation term and
:attr:`tdgl.Layer.u` sets the amplitude/phase relaxation ratio. It is the only
model that uses ``Layer.u`` and KWT ``gamma``.

On finite devices it supports terminals, time-dependent applied fields, CuPy,
and pyTDGL's nonlocal thin-film screening. The periodic backend supports the
same condensate equation against a fixed magnetic-periodic background, but not
screening.

See :doc:`background` for its nondimensionalization and numerical scheme, and
cite Bishop-Van Horn, `Computer Physics Communications 291, 108799 (2023)
<https://doi.org/10.1016/j.cpc.2023.108799>`_.

SPlusDModel
===========

:class:`tdgl.SPlusDModel` implements dimensionless dissipative s+d equations
following Gonçalves et al. The dominant d-sector sets the coherence-length and
diffusion scales. The parameters have the following roles:

* ``eta_s`` is the s-sector gradient stiffness and ``eta_v`` is the
  sign-changing x/y mixed-gradient coefficient. Positive-definite gradient
  energy requires ``eta_s > eta_v**2``.
* ``nu`` is the s-sector quadratic coefficient; ``tau1`` its self-quartic
  coefficient; and ``tau3`` and ``tau4`` are density and phase-sensitive
  intercomponent couplings.
* ``relaxation_s`` changes the s-sector phenomenological kinetic coefficient
  independently of its stiffness.
* ``nu_disorder_coupling`` makes the s-sector quadratic coefficient respond
  differently to ``disorder_epsilon``.
* ``beta_em`` is the electromagnetic relaxation/current normalization used by
  the local field equation.

Both backends support fixed prescribed fields and local electromagnetic
relaxation. The finite-device backend additionally supports an optional
homogeneous bulk-drive source through ``s_plus_d_drive_current_x`` and
``s_plus_d_drive_current_y`` when screening is enabled.

Reference: Gonçalves et al., `Journal of Mathematical Physics 55, 041501
(2014) <https://doi.org/10.1063/1.4870874>`_.

DPlusDPrimeModel
================

:class:`tdgl.DPlusDPrimeModel` implements the static
:math:`d_{x^2-y^2}+d_{xy}` GL functional of Lei, Aruna, and Wang.
``alpha`` controls the subdominant d_xy quadratic term. ``relaxation_d`` and
``relaxation_d_prime`` are phenomenological gradient-flow coefficients used to
seek equilibrium; the source model does not define physical TDGL kinetics.
``em_coupling`` converts the variational condensate current to the transport
normalization.

``zeeman_coupling`` optionally couples the condensate chirality to the local
dimensionless induction. Its sign selects the favored chirality. The
finite-device solution exposes the associated
:attr:`tdgl.Solution.orbital_magnetization`.

This model currently requires a fixed prescribed vector potential
(``include_screening=False``) on both backends. The magnetic-periodic virial
helper rejects nonzero ``zeeman_coupling`` because the implemented identity
does not contain the extra orbital term.

References: Lei, Aruna, and Wang, `Physical Review B 62, 8687 (2000)
<https://doi.org/10.1103/PhysRevB.62.8687>`_; Wang and Wang,
`arXiv:cond-mat/9909399 <https://doi.org/10.48550/arXiv.cond-mat/9909399>`_.

SPlusSModel
===========

:class:`tdgl.SPlusSModel` is an isotropic two-component s+s/s+is model. In the
implemented dimensionless free energy, ``a1`` and ``a2`` are quadratic
coefficients; ``b1`` and ``b2`` are self-quartic coefficients; and
``k2_over_k1`` is the second-component gradient stiffness relative to the
first. The intercomponent sector contains:

* ``josephson_gamma`` for bilinear phase locking;
* ``phase_gamma2`` for phase-sensitive quartic coupling;
* ``density_gamma3`` for density-density coupling; and
* ``mixed_gradient_k12`` for isotropic gradient drag.

Positive ``phase_gamma2`` can favor the time-reversal-breaking relative phases
:math:`\arg(\psi_2\psi_1^*)=\pm\pi/2` when competing lower-order couplings do
not override it. :attr:`tdgl.Solution.relative_phase` returns this
gauge-invariant phase difference.

``relaxation1`` and ``relaxation2`` are phenomenological kinetic coefficients.
``disorder_coupling1`` and ``disorder_coupling2`` allow
``disorder_epsilon`` to suppress the two components by different amounts.
``em_coupling`` controls current normalization and ``beta_em`` controls local
electromagnetic relaxation. Screening requires ``em_coupling=1``.

The constructor validates bounded quartic and positive-definite gradient
sectors. In particular,
``mixed_gradient_k12**2 < k2_over_k1``. A passive band with ``b2=0`` is
allowed only under the additional boundedness conditions reported by the
constructor.

References: Zhitomirsky and Dao, `Physical Review B 69, 054508 (2004)
<https://doi.org/10.1103/PhysRevB.69.054508>`_; Lin, Maiti, and Chubukov,
`Physical Review B 94, 064519 (2016)
<https://doi.org/10.1103/PhysRevB.94.064519>`_.

Disorder and initialization
===========================

The ``disorder_epsilon`` argument is evaluated on mesh or grid sites and must
not exceed one. It directly changes the dominant single-band, s+d, and d+d'
quadratic term. The model-specific disorder-coupling parameters described
above control the response of the other s+d or s+s component.

The finite-device solver chooses a model-aware uniform initial state unless a
compatible ``seed_solution`` is supplied. The periodic solver also accepts
``initial_psi1`` and ``initial_psi2`` arrays with shape ``cell.shape``. Prefer
those generic names in new code; the older ``initial_psi_d`` and
``initial_psi_s`` arguments remain for compatibility.

Example: inspecting two components
==================================

.. code-block:: python

   solution = tdgl.solve(device, options, disorder_epsilon=epsilon)

   first = solution.get_order_parameter("psi1")
   second = solution.get_order_parameter("psi2")

   if isinstance(device.layer.model, tdgl.SPlusDModel):
       s_order = solution.get_order_parameter("s")
       d_order = solution.get_order_parameter("d")
   elif isinstance(device.layer.model, tdgl.DPlusDPrimeModel):
       d_order = solution.get_order_parameter("d")
       d_prime_order = solution.get_order_parameter("d_prime")
   elif isinstance(device.layer.model, tdgl.SPlusSModel):
       phase_difference = solution.relative_phase
