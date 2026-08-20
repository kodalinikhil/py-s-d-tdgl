# py-s-d-TDGL

`py-s-d-TDGL` is an experimental Python framework for two-dimensional
Ginzburg–Landau simulations of single- and multi-component thin-film
superconductors. It extends [pyTDGL](https://github.com/loganbvh/py-tdgl) with
mixed-symmetry, chiral, and multiband order parameters, model-aware
diagnostics, local electromagnetic relaxation, and magnetic-periodic vortex
cells.

The package provides two complementary numerical backends:

| Backend | Domain and discretization | Main use |
| --- | --- | --- |
| `tdgl.solve` | Finite devices of arbitrary geometry on an unstructured triangular finite-volume mesh | Transport, defects, holes, physical edges, terminals, and time-dependent drives |
| `tdgl.solve_magnetic_periodic` | Rectangular structured grids with magnetic-translation boundary conditions and an integer flux sector | Bulk vortex lattices, fixed-flux relaxation, free energies, and virial-field diagnostics |

## Models

Select the equation set with `tdgl.Layer(model=...)`:

| Model | Components | Dynamics and notable couplings |
| --- | --- | --- |
| `SingleBandModel` | `psi` | Standard single-band Kramer–Watts–Tobin TDGL inherited from pyTDGL |
| `SPlusDModel` | `s`, `d(x²-y²)` | Gonçalves-type dissipative s+d dynamics, anisotropic mixed gradients, density/phase coupling, component-selective disorder, and optional local screening |
| `DPlusDPrimeModel` | `d(x²-y²)`, `d(xy)` | Phenomenological relaxation of the Lei–Aruna–Wang free energy, with optional orbital-Zeeman coupling to the local induction |
| `SPlusSModel` | `s1`, `s2` | Isotropic two-component s+s/s+is dynamics with Josephson, phase-sensitive quartic, density, mixed-gradient, and component-selective disorder couplings |

All four models support a prescribed vector potential on both backends.
`SPlusDModel` and `SPlusSModel` can evolve the local bulk electromagnetic
equation with `include_screening=True`. The finite-device backend retains
pyTDGL's thin-film Biot–Savart screening for `SingleBandModel`.
`DPlusDPrimeModel` currently requires `include_screening=False`.

The multi-component kinetic coefficients are phenomenological unless the
model documentation states otherwise. In particular, the d+d′ implementation
is an equilibrium-seeking gradient flow, not a microscopic real-time theory.

## Framework capabilities

- Construct films, holes, terminals, and probes from composable polygons.
- Generate unstructured Delaunay/Voronoi finite-volume meshes for arbitrary
  two-dimensional device geometries.
- Apply uniform or spatially/time-dependent vector potentials, terminal
  currents, and static disorder profiles.
- Use adaptive stepping, equilibrium-based early stopping, HDF5 checkpoints,
  and solution seeding.
- Run CPU sparse solves with SuperLU, UMFPACK, or PARDISO; use CuPy acceleration
  where supported by the finite-device backend.
- Inspect both condensates, relative phase, orbital magnetization, local
  induction, supercurrent/normal current, voltage, fluxoid, vorticity, and
  magnetic-field reconstructions.
- Plot snapshots, build animations, and export finite-device results for
  external visualization.
- Relax magnetic-periodic cells at exactly quantized mean flux and compute
  model-specific free-energy and virial applied-field diagnostics.

## Installation

Python 3.9–3.14 is supported. This fork is installed from source:

```bash
git clone https://github.com/kodalinikhil/py-s-d-tdgl.git
cd py-s-d-tdgl
python -m pip install -e .
```

For development and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

The distribution name is `py-s-d-tdgl`, while the import name remains `tdgl`.
Installing it alongside upstream `tdgl` in the same environment is therefore
not supported.

## Quick start

This minimal finite-device example relaxes an s+is-capable two-band model in a
square film:

```python
import tdgl
from tdgl.geometry import box

layer = tdgl.Layer(
    coherence_length=0.05,
    london_lambda=0.20,
    thickness=0.01,
    conductivity=1,
    model=tdgl.SPlusSModel(
        k2_over_k1=0.5,
        phase_gamma2=0.5,
        mixed_gradient_k12=0.25,
    ),
)
device = tdgl.Device(
    "square",
    layer=layer,
    film=tdgl.Polygon("film", points=box(1.0, 1.0)).resample(101).buffer(0),
    length_units="um",
)
device.make_mesh(max_edge_length=0.05)

options = tdgl.SolverOptions(
    solve_time=10,
    dt_init=1e-3,
    dt_max=5e-2,
    equilibrium_tolerance=1e-7,
    output_file="s_plus_s.h5",
)
solution = tdgl.solve(device, options, applied_vector_potential=0)

psi1 = solution.get_order_parameter("psi1")
psi2 = solution.get_order_parameter("psi2")
relative_phase = solution.relative_phase
```

For a magnetic-periodic calculation, replace the `Device` with a
`MagneticPeriodicCell` and call `tdgl.solve_magnetic_periodic`. See the
[framework guide](docs/framework.rst), [model guide](docs/models.rst),
[magnetic-periodic guide](docs/magnetic_periodic.rst), and the
[single-band quickstart notebook](docs/notebooks/quickstart.ipynb).

## Scope and status

This is research software under active development. The public API and HDF5
schemas may change before a stable release. The solvers are two-dimensional
thin-film or bulk-cell GL models; they do not provide a microscopic pairing
calculation, thermal fluctuations, or a self-heating equation.

## References and attribution

The geometry, finite-volume, single-band solver, screening, solution, and
visualization foundations derive from Logan Bishop-Van Horn's
[pyTDGL paper](https://doi.org/10.1016/j.cpc.2023.108799) and project.
The extended models draw on:

- Gonçalves et al., [mixed d+s TDGL](https://doi.org/10.1063/1.4870874).
- Lei, Aruna, and Wang, [d+id′ GL theory](https://doi.org/10.1103/PhysRevB.62.8687), and Wang and Wang's [orbital-Zeeman extension](https://doi.org/10.48550/arXiv.cond-mat/9909399).
- Zhitomirsky and Dao, [two-gap GL theory](https://doi.org/10.1103/PhysRevB.69.054508), and Lin, Maiti, and Chubukov, [s+is defect response](https://doi.org/10.1103/PhysRevB.94.064519).

MIT licensed. Maintained by Nikhil Kodali; based on pyTDGL by Logan
Bishop-Van Horn and its contributors.
