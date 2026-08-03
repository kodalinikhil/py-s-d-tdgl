# d+d' Ginzburg-Landau model implementation plan

## Goal

Add a first-class `DPlusDPrimeModel` for the dimensionless
\(d_{x^2-y^2}+i d_{xy}\) Ginzburg-Landau free energy in Lei, Aruna, and Wang,
*Field Driven Pairing State Phase Transition in d(x2-y2) + i d(xy)-Wave
Superconductors* (arXiv:cond-mat/0004227v1). The first release should relax the
two complex fields to equilibrium in a prescribed magnetic vector potential and
should make it possible to reproduce the paper's zero-field transition and its
field-driven loss of the \(d_{xy}\) component.

This is a static GL paper. Any time evolution used by this repository is a
phenomenological gradient-flow algorithm for finding equilibrium, not a
paper-derived real-time TDGL model.

## Scope and non-goals

### First release

- Represent the paper's dimensionless free energy exactly and provide the
  orbital Zeeman coupling as an explicit, default-off model parameter. The
  Zeeman convention follows Wang and Wang, arXiv:cond-mat/9909399.
- Evolve two complex order parameters with gauge-covariant dissipative gradient
  flow.
- Use the existing unstructured finite-volume mesh, link variables, scalar
  potential solve, adaptive stepping, HDF5 storage, and plotting framework.
- Support a prescribed, uniform magnetic field. This matches the paper's
  numerical approximation of uniform induction for large \(\kappa\), subject to
  finite-domain boundary effects.
- Provide model-aware names for \(d\) and \(d'\), a discrete free-energy
  diagnostic, and a reproducible field/temperature scan.

### Deferred

- Self-consistent magnetic screening for a multicomponent model.
- Magnetic-periodic boundary conditions for an exact one-vortex Abrikosov unit
  cell.
- Claims about physical order-parameter dynamics or unequal microscopic
  relaxation times.
- Mixed-gradient coupling. The absence of such a term is an essential feature
  of this paper's \(d+d'\) model.

## Paper equations and code convention

Use the repository convention

\[
D_i=\partial_i-iA_i,\qquad D_t=\partial_t+i\mu.
\]

The paper writes \(\pi=-i\nabla-\mathbf a=-i\mathbf D\), so
\(|\pi d|^2=|\mathbf Dd|^2\). In the paper's dimensionless units, the free
energy density is

\[
\begin{aligned}
f={}&-|d|^2-\alpha|d'|^2
+\frac{|d|^4+|d'|^4}{2}
+\frac{|d|^2|d'|^2}{3}\\
&+\frac{(d^*d'+d d'^*)^2}{6}
+|\mathbf Dd|^2+|\mathbf Dd'|^2+\kappa^2b^2
-i g_Z b(d^*d'-d d'^*).
\end{aligned}
\]

Here \(d=D/D_0\), \(d'=D'/D_0\),
\(\alpha=\alpha_{D'}/\alpha_D\), length is measured in the dominant-channel
coherence length \(\xi\), and field is measured in
\(B_0=\Phi_0/(2\pi\xi^2)=B_{c2}\). Therefore an input reduced field \(b\)
maps to `b * device.Bc2` at the public API boundary.

Expanding the phase-sensitive term gives

\[
\frac{(d^*d'+d d'^*)^2}{6}
=\frac{|d|^2|d'|^2}{3}
+\frac{d^{*2}d'^2+d^2d'^{*2}}{6}.
\]

Thus the complete cross-density coefficient is \(2/3\), and the negative
functional gradients are

\[
\begin{aligned}
-\frac{\delta F}{\delta d^*}={}&D^2d+d-|d|^2d
-\frac{2}{3}|d'|^2d-\frac{1}{3}d'^2d^*+i g_Z b d',\\
-\frac{\delta F}{\delta d'^*}={}&D^2d'+\alpha d'-|d'|^2d'
-\frac{2}{3}|d|^2d'-\frac{1}{3}d^2d'^*-i g_Z b d.
\end{aligned}
\]

Implement the relaxation equations as

\[
\begin{aligned}
r_dD_td&=-\delta F/\delta d^*,\\
r_{d'}D_td'&=-\delta F/\delta d'^*,
\end{aligned}
\]

with positive, dimensionless numerical relaxation coefficients. Set both to
one by default and document that changing them changes only the route to a
static solution.

The condensate current associated with the gradient terms is, in the solver's
existing normalization,

\[
\mathbf J_{\rm cond}=\operatorname{Im}(d^*\mathbf Dd)
+\operatorname{Im}(d'^*\mathbf Dd').
\]

There is no mixed-current contribution. Initially use
\(\mathbf j_s=\mathbf J_{\rm cond}/c_{\rm em}\) with `em_coupling=1` by
default, consistent with the other multicomponent implementations.

## Relationship to the current s+d implementation

The paper's equations are algebraically the following special case of the
current `SPlusDModel` coefficient convention:

| Existing s+d coefficient | d+d' value |
| --- | ---: |
| `eta_s` | \(1\) |
| `eta_v` | \(0\) |
| `nu` | \(\alpha\) |
| `tau1` | \(1\) |
| `tau3` | \(4/3\) |
| `tau4` | \(1/3\) |
| `beta_em` | \(1\) |
| `disorder_epsilon` | \(1\) for the paper's homogeneous model |

That mapping is useful as an independent implementation check. It should not be
the public API because calling the \(d_{xy}\) field an s-wave component would
make initialization, result access, documentation, and future Zeeman support
error-prone.

## Public model and data API

Add this dataclass in `tdgl/device/models.py`:

```python
@dataclass
class DPlusDPrimeModel:
    alpha: float = 0.5
    relaxation_d: float = 1.0
    relaxation_d_prime: float = 1.0
    em_coupling: float = 1.0
    zeeman_coupling: float = 0.0
```

Validation:

- all five values must be finite;
- both relaxation coefficients and `em_coupling` must be positive;
- require `abs(zeeman_coupling) < 1`, matching the stable weak-coupling regime;
- permit negative `alpha`, since the paper's temperature mapping can produce it;
- warn, rather than fail, for `alpha > 1`, because that leaves the paper's
  assumed subdominant-channel regime but remains mathematically solvable.

Use the canonical storage convention

- `psi1` = \(d_{x^2-y^2}\), the dominant component;
- `psi2` = \(d_{xy}\), the subdominant component.

Add model-aware access through
`Solution.get_order_parameter("d")` and
`Solution.get_order_parameter("d_prime")`. Do not add another model-agnostic
alias to `TDGLData`; that object cannot determine component symmetry without
the owning `Device`. Extend plot/interpolation component literals with
`"d_prime"`.

Serialize the model through the existing `Layer` model group with a new model
type string and the current schema. Export it from `tdgl/__init__.py`.

## Implementation phases

### 1. Add the model type and persistence

Files:

- `tdgl/device/models.py`
- `tdgl/device/layer.py`
- `tdgl/__init__.py`
- `tdgl/test/test_d_plus_d_prime.py`

Tasks:

1. Add and validate `DPlusDPrimeModel`.
2. Include the type in `Layer` annotations and the HDF5 model-class registry.
3. Verify copy, equality, repr, and HDF5 round trips.
4. Add a migration test that confirms old files remain unchanged. No legacy
   format exists for this new model, so no guessed migration is needed.

### 2. Add model-specific initialization and guardrails

File: `tdgl/solver/solver.py`.

Tasks:

1. Initialize `psi1` (\(d\)) to one and `psi2` (\(d'\)) to a small imaginary
   perturbation. Make the perturbation deterministic so both tests and phase
   scans are reproducible. The imaginary seed selects one of the degenerate
   chiralities without changing the equilibrium amplitudes.
2. Continue to accept `seed_solution` for field and alpha continuation scans.
3. For the first release, reject `include_screening=True` with the same clear
   prescribed-vector-potential guard used by the other multicomponent models.
4. Use `disorder_epsilon=1` in the paper-reproduction path. If the general
   solver keeps supporting nonuniform values, document that they multiply only
   the dominant \(+d\) drive and are an extension beyond the paper.
5. Reject or explicitly define scalar `terminal_psi` behavior. Recommended
   first-release behavior is to reject it for this model because the paper does
   not specify multicomponent current-terminal boundary conditions.

### 3. Implement the gradient-flow equations

File: `tdgl/solver/solver.py`.

Tasks:

1. Reuse `operators.psi_laplacian` for both components. Do not build or apply
   `laplacian_x - laplacian_y`; the paper has no mixed-gradient term.
2. Add the two right-hand sides shown above using `psi1=d` and `psi2=d_prime`.
3. Apply the existing temporal link `exp(-1j * mu * dt)` to preserve discrete
   gauge covariance.
4. Divide each explicit step by its positive relaxation coefficient.
5. Reuse the current rejected-step/adaptive-step logic, but rename local
   `d`/`s` variables to component-neutral names before adding another branch.
   This small refactor reduces component-order mistakes.
6. Keep CPU and CuPy expressions backend-neutral through the existing `xp`
   abstraction.

### 4. Implement current and free-energy diagnostics

Files:

- `tdgl/finite_volume/operators.py`
- `tdgl/solver/solver.py`

Tasks:

1. Compute the current by reusing the two-isotropic-component current helper
   with unit stiffness ratio, or rename that helper to a symmetry-neutral
   `get_two_component_supercurrent` and retain the old wrapper.
2. Divide by `em_coupling` before the scalar-potential Poisson solve and before
   storage, matching current multicomponent conventions.
3. Build a discrete Stokes curl that maps the edge-centered vector potential
   to the local site induction `b = B / Bc2` used by the Zeeman term.
4. Add `compute_d_plus_d_prime_free_energy(d, d_prime)`. For fixed vector
   potential, evaluate the site potential above plus the discrete gradient
   energy
   `-Re(conj(psi) * psi_laplacian @ psi)`, area weighted for both components.
   Include the orbital Zeeman energy but omit the standalone `kappa**2 * b**2`
   term because it is constant when the applied vector potential is prescribed.
5. Use the diagnostic in convergence tests and optionally expose it in scan
   output.

### 5. Make outputs model-aware

Files:

- `tdgl/solution/solution.py`
- `tdgl/solution/plot_solution.py`
- `tdgl/solution/data.py` (documentation only unless a generic metadata field is
  introduced)

Tasks:

1. Route `"d"` and `"d_prime"` to the right canonical fields based on the
   owning model.
2. Update plot labels to \(d_{x^2-y^2}\) and \(d_{xy}\), including relative
   phase \(\arg(d')-\arg(d)\).
3. Preserve all existing `psi1`/`psi2`, legacy `psi_s`/`psi_d`, single-band,
   and s+s loading behavior.
4. Store model parameters through `Device`/`Layer`; do not duplicate `alpha`
   on every time-step group.

### 6. Document the model and add a reproduction script

Files:

- `README.md`
- `latex/Model Equations.tex` or a new docs source section
- `my_scripts/simulate_d_plus_d_prime_phase_diagram.py`

The script should:

1. Build one sufficiently large square or disk with no current terminals and
   mesh resolution converged relative to \(\xi\).
2. Convert every reduced field with
   `applied_field = b * device.Bc2.to(field_units).magnitude`.
3. For each `alpha`, start from a converged low-field mixed state and sweep `b`
   upward, seeding every run from the previous solution. Also sweep downward to
   detect hysteresis or the weakly first-order behavior mentioned in the paper.
4. Record `max(abs(d_prime))`, an interior/bulk value excluding a boundary strip,
   the spatial mean relative phase, free energy, convergence measure, mesh size,
   and solve duration.
5. Define the numerical transition by a documented amplitude threshold and
   demonstrate that the inferred point is stable when the threshold, mesh, and
   domain size are varied.
6. Plot the numerical transition points against the paper's analytical limits:

   - low/intermediate field:
     \(b\ln(1/b)=(3\alpha-1)/2\), taking the low-field root;
   - high field:
     \(b=[1+\sqrt{9\alpha^2-12\alpha+4}]/2\), using only its physical
     high-field branch;
   - \(b_{c2}=1\).

The finite-domain solver is not expected to match the paper's vortex-unit-cell
points exactly. Label comparisons as qualitative until magnetic-periodic
boundary conditions are implemented.

### 7. Optional exact-reproduction phase: magnetic-periodic unit cell

This is a separate feature because it changes mesh topology and boundary
operators, not just the GL model.

1. Add paired boundary-site/edge maps for a parallelogram Abrikosov cell.
2. Enforce magnetic Bloch phases whose boundary mismatch produces one flux
   quantum per cell.
3. Validate gauge-invariant continuity of both order parameters and current
   across paired boundaries.
4. Set the cell area from \(BA=\Phi_0\), scan the uniform induction directly,
   and compare `max(abs(d_prime))` with Figure 2.

Do not block the first release on this phase.

## Verification matrix

### Analytic unit tests

1. **Coefficient mapping:** for identical fields and one Euler step, the new
   model agrees with `SPlusDModel(eta_s=1, eta_v=0, nu=alpha, tau1=1,
   tau3=4/3, tau4=1/3, beta_em=em_coupling)`.
2. **Pure-d equilibrium:** at zero field and \(\alpha<1/3\), \(d=1,d'=0\)
   is stationary.
3. **Mixed equilibrium:** for \(1/3<\alpha<1\), the stationary homogeneous
   amplitudes are

   \[
   |d|^2=\frac{3(3-\alpha)}8,\qquad
   |d'|^2=\frac{3(3\alpha-1)}8,
   \]

   with relative phase \(\pm\pi/2\).
4. **Phase locking:** at fixed nonzero amplitudes, the potential energy at
   relative phase \(\pm\pi/2\) is lower than at zero or \(\pi\).
5. **Current:** the discrete current equals the sum of the two ordinary
   covariant component currents divided by `em_coupling` and contains no mixed
   term.
6. **Dissipation:** with fixed \(A\), \(\mu=0\), and sufficiently small `dt`,
   one relaxation step does not increase the discrete free energy.
7. **Gauge covariance:** a site phase transformation accompanied by the
   corresponding link transformation leaves amplitudes, free energy, and
   gauge-invariant current unchanged.
8. **Zeeman symmetry:** reversing `b` reverses the selected chirality, and the
   favored chiral state has lower free energy than its conjugate.

### Integration and regression tests

1. A short CPU solve produces finite `psi1`, `psi2`, current, and potential and
   saves canonical HDF5 component names.
2. HDF5 save/load and seed continuation preserve the model and both fields.
3. Existing single-band, s+d, and s+s tests remain unchanged and pass.
4. If CuPy is available, a small CPU/GPU parity test compares one step and
   current evaluation.
5. Unsupported screening and terminal configurations fail early with precise
   messages.

### Scientific acceptance tests

1. At \(b=0\), the numerical onset of \(|d'|\) converges to
   \(\alpha_*=1/3\).
2. In the mixed phase, the bulk relative phase is \(\pm\pi/2\) away from vortex
   cores.
3. At fixed field, `max(abs(d_prime))` vanishes below a field-dependent critical
   `alpha`, reproducing the trend in Figure 2.
4. The transition field increases with `alpha` and approaches \(b=1\) as
   \(\alpha\to1\).
5. Results used for comparison include a mesh/domain convergence table and both
   scan directions.

## Recommended delivery sequence

1. **Model MVP:** phases 1-5 plus analytic and short-solve tests.
2. **Scientific workflow:** phase 6 plus convergence and phase-diagram output.
3. **Exact paper geometry:** phase 7 only if quantitative agreement with the
   paper's vortex-lattice points is required.

The MVP should be considered complete when the exact paper functional is
represented, negative-gradient flow decreases its discrete free energy, the
zero-field analytic states are recovered, outputs round-trip correctly, and no
existing model regresses. Reproduction should be considered complete only after
the field scan, convergence study, and limitations from finite boundaries are
reported together.
