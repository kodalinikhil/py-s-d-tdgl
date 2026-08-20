.. _magnetic-periodic:

**************************************
Magnetic-periodic rectangular cells
**************************************

The magnetic-periodic solver is a standalone structured-grid backend for a
rectangular vortex-lattice unit cell. It is intended for calculations with a
fixed integer number of flux quanta and is not an alternate boundary switch on
the finite-device arbitrary-mesh solver. The class and function reference is
in :doc:`api/magnetic-periodic`.

The public cell lengths are physical values in ``length_units``. In the
equations below, :math:`L_x`, :math:`L_y`, and the grid spacings have already
been divided by the layer coherence length, and :math:`B` is in the usual
:math:`B_{c2}` units.

The two backends represent different physical domains:

* ``MagneticPeriodicSolver`` evolves a rectangular torus. Opposite sides are
  identified by magnetic translations, and there is no physical perimeter.
* The finite-device solver evolves a finite film on an arbitrary triangular mesh. It
  retains variational order-parameter conditions on the sample boundary and,
  for local electromagnetic screening, imposes :math:`B=H` at that boundary.

The periodic backend does not add seam edges to, or otherwise modify, the
finite-device open-mesh operators.

Basic use
=========

Construct a cell from a supported model layer, its physical side lengths, grid
shape, and flux sector.  ``initial_psi1`` and ``initial_psi2`` follow the
model-neutral component order described below:

.. code-block:: python

   cell = tdgl.MagneticPeriodicCell(
       layer=layer,
       lengths=(Lx, Ly),
       shape=(ny, nx),
       flux_quanta=n,
       length_units="um",
   )

   solution = tdgl.solve_magnetic_periodic(
       cell,
       options,
       disorder_epsilon=epsilon,
       initial_psi1=psi1_0,
       initial_psi2=psi2_0,
   )

An existing ``MagneticPeriodicSolution`` from the same cell can instead be
supplied as ``seed_solution``. A periodic solution is deliberately separate
from the arbitrary-mesh ``Solution``; seeds cannot be moved between the two
backends.

Supported models and components
===============================

The CPU backend supports all four models accepted by ``Layer``.  Their public
component order and field capabilities are:

.. list-table::
   :header-rows: 1
   :widths: 24 24 26 26

   * - Model
     - Canonical components
     - Fixed background
     - Evolving screening
   * - ``SingleBandModel``
     - ``("psi",)``
     - supported
     - not currently supported
   * - ``SPlusDModel``
     - ``("s", "d")``
     - supported
     - supported
   * - ``DPlusDPrimeModel``
     - ``("d", "d_prime")``
     - supported
     - not currently supported
   * - ``SPlusSModel``
     - ``("s1", "s2")``
     - supported
     - supported when ``em_coupling=1``

Use ``initial_psi1`` and, for a two-component model, ``initial_psi2`` to avoid
encoding a particular model in setup code.  The older named arguments remain
available: ``initial_psi_d`` and ``initial_psi_s`` mean the actual d and s
fields for s+d.  ``initial_psi_d`` is the single-band compatibility argument;
for d+d' and s+s, the two older arguments are backwards-compatible names for
the first and second stored arrays.  Do not provide both a generic argument
and its named equivalent.

Frames and solutions expose ``component_names``, ``psi1``, ``psi2``, and
``get_component(name)``; frames additionally provide a read-only
``component_map``.  Model-specific accessors are also available: ``psi`` for
single band, ``psi_s`` and ``psi_d`` for s+d, ``psi_d`` and
``psi_d_prime`` for d+d', and ``psi_s1`` and ``psi_s2`` for s+s.  New HDF5
files use schema 2: each frame stores arrays as ``psi1`` and, when present,
``psi2``, together with a ``component_names`` attribute.  This keeps the file
layout model-neutral while retaining the physical names needed to interpret
it.  The reader still accepts schema-1 s+d files whose datasets are
named ``psi_s`` and ``psi_d``.

Field modes
===========

``SolverOptions(include_screening=False)`` holds the periodic vector-potential
correction :math:`\mathbf a` fixed.  With the default
``initial_vector_potential=None``, :math:`\mathbf a=0` and the induction is the
uniform, exactly quantized background :math:`\overline B`.  Supplying an
``initial_vector_potential`` instead prescribes that periodic correction and
its field profile for the entire run.  This fixed-background mode works for
all four models and is currently required by ``SingleBandModel`` and
``DPlusDPrimeModel``.  In d+d', the optional orbital-Zeeman coupling uses this
prescribed local induction.

``SolverOptions(include_screening=True)`` advances :math:`\mathbf a` with the
local :math:`\phi=0` Maxwell equation while retaining the exact mean flux.  It
is available for ``SPlusDModel`` and ``SPlusSModel``; s+s screening requires
``em_coupling=1`` so the condensate and field equations vary the same free
energy.  A homogeneous periodic drive is likewise available only for these
two models with screening enabled.  The backend is CPU-only, uses SuperLU,
supports static disorder, and has no terminals, physical sample edge, or
finite-device live monitor.

Minimal fixed-field d+d' example
================================

This example uses one flux quantum and leaves the periodic correction at zero,
so the d+d' relaxation sees a spatially uniform fixed induction:

.. code-block:: python

   import numpy as np
   import tdgl

   layer = tdgl.Layer(
       coherence_length=0.05,
       london_lambda=0.20,
       thickness=0.01,
       conductivity=1,
       model=tdgl.DPlusDPrimeModel(
           alpha=0.7,
           zeeman_coupling=0.15,
       ),
   )
   cell = tdgl.MagneticPeriodicCell(
       layer=layer,
       lengths=(0.8, 0.6),
       shape=(48, 64),
       flux_quanta=1,
       length_units="um",
   )
   options = tdgl.SolverOptions(
       solve_time=10,
       dt_init=1e-3,
       dt_max=5e-2,
       include_screening=False,
       terminal_psi=None,
       save_every=100,
       output_file="d_plus_d_prime_fixed_B.h5",
   )

   d0 = np.ones(cell.shape, dtype=complex)
   d_prime0 = np.full(cell.shape, -1e-3j, dtype=complex)
   solution = tdgl.solve_magnetic_periodic(
       cell,
       options,
       initial_psi1=d0,
       initial_psi2=d_prime0,
   )
   d_final = solution.get_component("d")
   d_prime_final = solution.get_component("d_prime")
   fixed_B = cell.mean_induction

Fixed flux and magnetic translations
====================================

Let the cell be :math:`\Omega=[0,L_x)\times[0,L_y)` and let
:math:`n\in\mathbb Z` be the number of flux quanta. The mean induction is
fixed by

.. math::
   :label: magnetic-periodic-mean

   \overline B = \frac{2\pi n}{L_xL_y}.

In the Landau gauge
:math:`\overline{\mathbf A}=(0,\overline Bx)`, each condensate obeys

.. math::
   :label: magnetic-periodic-translations

   \begin{aligned}
   \psi_\alpha(x+L_x,y)
      &=e^{i\overline B L_x y}\psi_\alpha(x,y),\\
   \psi_\alpha(x,y+L_y)
      &=\psi_\alpha(x,y),
   \qquad \alpha=1,\ldots,N_{\rm components}.
   \end{aligned}

Both condensates acquire the same gauge phase. Flux quantization makes the
translations compatible at a corner:
:math:`e^{i\overline B L_xL_y}=e^{i2\pi n}=1`.

The dynamical vector potential is split into a fixed background and a periodic
part,

.. math::
   :label: magnetic-periodic-split

   \mathbf A=\overline{\mathbf A}+\mathbf a,\qquad
   B=\overline B+(\nabla\times\mathbf a)_z,

with
:math:`\mathbf a(x+L_x,y)=\mathbf a(x,y)` and
:math:`\mathbf a(x,y+L_y)=\mathbf a(x,y)`. Therefore

.. math::
   :label: magnetic-periodic-flux

   \int_\Omega B\,d^2r
      =\overline B L_xL_y
       +\int_\Omega(\nabla\times\mathbf a)_z\,d^2r
      =2\pi n.

The periodic curl contributes zero net flux, so the evolution cannot change
the selected flux sector.

The area in equation :eq:`magnetic-periodic-mean` is the *dimensionless* area
``cell.dimensionless_area``.  If physical lengths are :math:`L_x^{\rm phys}`
and :math:`L_y^{\rm phys}` and the coherence length is :math:`\xi`, then
:math:`L_x=L_x^{\rm phys}/\xi` and :math:`L_y=L_y^{\rm phys}/\xi`.  Thus one
flux quantum means dimensionless integrated flux :math:`2\pi`, exactly, and

.. math::

   \overline B
      = \frac{2\pi n\xi^2}
              {L_x^{\rm phys}L_y^{\rm phys}}.

Equivalently, in physical units the cell contains
:math:`\overline B_{\rm phys}L_x^{\rm phys}L_y^{\rm phys}=n\Phi_0`.
For a target dimensionless induction, choose
``cell.dimensionless_area = 2 * pi * n / target_B``; changing the cell area at
fixed integer ``flux_quanta`` changes the allowed mean induction.

There is no physical perimeter and no :math:`B=H` boundary condition in this
backend. An external :math:`H` is relevant only when evaluating a Gibbs
diagnostic containing :math:`\kappa^2(B-H)^2`; it is not an evolution boundary
value and does not replace the fixed-flux constraint.

For a stationary cell with spatially uniform GL coefficients and nonzero
:math:`\overline B`, the applied field can instead be recovered after the
fixed-flux relaxation from the Doria virial relation,

.. math::
   :label: magnetic-periodic-virial

   H_{\mathrm{vir}}
      =\frac{\langle f_{\mathrm{grad}}\rangle
              +2\kappa^2\langle B^2\rangle}
             {2\kappa^2\overline B}.

Here :math:`f_{\mathrm{grad}}` contains all model-specific diagonal and mixed
condensate-gradient terms. The relation is a postprocessing diagnostic, not
an additional boundary condition. The current helper rejects zero mean flux
and spatially varying disorder, for which the displayed form is insufficient;
it also rejects d+d' with nonzero orbital-Zeeman coupling because that term
requires a modified virial identity.

Structured covariant operators
==============================

The cell contains an :math:`N_y\times N_x` Cartesian grid with spacings
:math:`h_x=L_x/N_x` and :math:`h_y=L_y/N_y`. The covariant forward
differences use directed nearest-neighbor links,

.. math::
   :label: magnetic-periodic-differences

   \begin{aligned}
   (D_x^h\psi)_{ij}
      &=\frac{U^x_{ij}\psi_{i+1\bmod N_x,j}-\psi_{ij}}{h_x},\\
   (D_y^h\psi)_{ij}
      &=\frac{U^y_{ij}\psi_{i,j+1\bmod N_y}-\psi_{ij}}{h_y}.
   \end{aligned}

Here
:math:`U^\nu=\exp[-i\int_{\mathrm{bond}}\mathbf A\cdot d\boldsymbol\ell]`.
The wrap links carry the magnetic-translation phases; the site arrays
themselves do not need duplicate boundary rows or columns. The directional
Laplacians and condensate currents are assembled from these same links,
preserving discrete gauge covariance.

Let :math:`C` be the oriented bond-to-plaquette curl and let :math:`C^T` be
its adjoint in the grid energy inner product. The discrete field and local
Maxwell step have the form

.. math::
   :label: magnetic-periodic-maxwell

   \mathbf B=\overline B\mathbf 1+C\mathbf a,\qquad
   \beta_{\rm em}\dot{\mathbf a}
      =\mathbf J_{\rm cond}-\mathbf J_{\rm drive}
       -\kappa^2C^TC\mathbf a.

Thus the magnetic diffusion operator is the periodic positive-semidefinite
curl--curl operator :math:`C^TC`. Since a periodic curl has zero mean, this
step relaxes field variations while preserving the total flux exactly.
The two uniform harmonic link modes are retained rather than projected out.
A bulk drive can therefore advance an unwrapped uniform part of
:math:`\mathbf a`, producing a nonzero cell-averaged
:math:`\mathbf E=-\partial_t\mathbf a` without changing the flux sector.
