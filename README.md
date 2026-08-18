# py-s-d-TDGL

An experimental 2D finite-volume TDGL solver for thin-film superconductors,
forked from [pyTDGL](https://github.com/loganbvh/py-tdgl).

It supports four models selected through `tdgl.Layer(model=...)`:

- `SingleBandModel` — standard single-band pyTDGL (default)
- `SPlusDModel` — mixed s+d
- `DPlusDPrimeModel` — d(x2-y2)+d(xy), with optional orbital-Zeeman coupling
- `SPlusSModel` — isotropic two-component s+s/s+is, including optional
  phase-sensitive quartic, density, and mixed-gradient couplings

`DPlusDPrimeModel` requires a prescribed vector potential. `SPlusDModel` and
`SPlusSModel` can either use a prescribed vector potential or evolve their
local electromagnetic equation with `include_screening=True`. The latter two
models use phenomenological dissipative dynamics.

## Install

```bash
git clone https://github.com/kodalinikhil/py-s-d-tdgl.git
cd py-s-d-tdgl
python -m pip install -e .
```

## Use

```python
import tdgl

layer = tdgl.Layer(
    coherence_length=1,
    london_lambda=2,
    thickness=0.1,
    model=tdgl.SPlusSModel(
        k2_over_k1=0.5,
        phase_gamma2=0.5,
        mixed_gradient_k12=0.5,
    ),
)
```

Build a `tdgl.Device` and call `tdgl.solve` as shown in the
[quickstart notebook](docs/notebooks/quickstart.ipynb).

## References

- Bishop-Van Horn, *Computer Physics Communications* **291**, 108799 (2023), [pyTDGL](https://doi.org/10.1016/j.cpc.2023.108799)
- Gonçalves et al., *Journal of Mathematical Physics* **55**, 041501 (2014), [s+d](https://doi.org/10.1063/1.4870874)
- Lei, Aruna, and Wang, [d+d'](https://arxiv.org/abs/cond-mat/0004227)
- Zhitomirsky and Dao, [two-band GL](https://doi.org/10.1103/PhysRevB.69.054508)
- Lin, Maiti, and Chubukov, [s+is defect response](https://doi.org/10.1103/PhysRevB.94.064519)

MIT licensed. Maintained by Nikhil Kodali; based on pyTDGL by Logan Bishop-Van Horn and its contributors.
