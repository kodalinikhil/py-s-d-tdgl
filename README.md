# py-s-d-TDGL

An experimental 2D finite-volume TDGL solver for thin-film superconductors,
forked from [pyTDGL](https://github.com/loganbvh/py-tdgl).

It supports four models selected through `tdgl.Layer(model=...)`:

- `SingleBandModel` — standard single-band pyTDGL (default)
- `SPlusDModel` — mixed s+d
- `DPlusDPrimeModel` — d(x2-y2)+d(xy), with optional orbital-Zeeman coupling
- `SPlusSModel` — isotropic two-band s+s

`DPlusDPrimeModel` and `SPlusSModel` use phenomenological equilibrium-finding
flows. All multi-component models currently require a prescribed vector
potential (`include_screening=False`).

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
    model=tdgl.SPlusDModel(eta_v=0.25),
)
```

Build a `tdgl.Device` and call `tdgl.solve` as shown in the
[quickstart notebook](docs/notebooks/quickstart.ipynb).

## References

- Bishop-Van Horn, *Computer Physics Communications* **291**, 108799 (2023), [pyTDGL](https://doi.org/10.1016/j.cpc.2023.108799)
- Gonçalves et al., *Journal of Mathematical Physics* **55**, 041501 (2014), [s+d](https://doi.org/10.1063/1.4870874)
- Lei, Aruna, and Wang, [d+d'](https://arxiv.org/abs/cond-mat/0004227)
- Zhitomirsky and Dao, [two-band GL](https://doi.org/10.1103/PhysRevB.69.054508)

MIT licensed. Maintained by Nikhil Kodali; based on pyTDGL by Logan
Bishop-Van Horn and its contributors.
