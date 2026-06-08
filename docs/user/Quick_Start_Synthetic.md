# Quick Start: Synthetic Spectra

This guide is for the **secondary** use case of CF-LIBS: generating a
synthetic LIBS spectrum from a known plasma state. This is useful for
experimental design, instrument response modeling, sanity-checking an
inversion result, and training/benchmarking identification algorithms.

For the **primary** use case — extracting plasma temperature and
composition from a real measured spectrum — see
[Quick Start: Real Data](Quick_Start_Real_Data.md).

---

## 1. Setup

```bash
cflibs doctor
```

Then, if `cflibs` is not installed:

```bash
uv venv --python 3.12
uv pip install -e ".[dev,jax-cpu,hdf5]"
```

The bundled atomic database at `ASD_da/libs_production.db` is sufficient
for the example below.

## 2. Generate a Spectrum

```bash
cflibs forward examples/config_example.yaml --output spectrum.csv
```

Output is a two-column CSV (`wavelength_nm`, `intensity_W_m2_nm_sr`).

## 3. Edit the Plasma State

Open `examples/config_example.yaml`:

```yaml
atomic_database: ASD_da/libs_production.db

plasma:
  model: single_zone_lte
  Te: 10000.0          # electron temperature [K]
  ne: 1.0e17           # electron density [cm^-3]
  species:
    - element: Fe
      number_density: 1.0e15
    - element: H
      number_density: 1.0e16

instrument:
  resolution_fwhm_nm: 0.05    # or use 'resolving_power: 5000'

spectrum:
  lambda_min_nm: 200.0
  lambda_max_nm: 800.0
  delta_lambda_nm: 0.01
  path_length_m: 0.01
```

Practical ranges:

- `Te`: 5000–25000 K. Picosecond/nanosecond LIBS plasmas are typically
  around 8000–15000 K. Picosecond plumes are cooler.
- `ne`: 10¹⁶–10¹⁸ cm⁻³ in atmospheric-pressure plumes.
- `species.number_density`: relative ratios are what matter for line
  intensities; the absolute scale sets the optical depth.
- `resolution_fwhm_nm`: 0.01–0.05 for echelle, 0.1–0.5 for grating
  spectrometers.

For the physics meaning of each parameter, see
[Physics: Equations](../physics/Equations.md).

## 4. From Python

```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.instrument import InstrumentModel
from cflibs.radiation import SpectrumModel

db = AtomicDatabase("ASD_da/libs_production.db")

plasma = SingleZoneLTEPlasma(
    T_e=10000.0,
    n_e=1e17,
    species={"Fe": 1e15, "H": 1e16},
)
instrument = InstrumentModel(resolution_fwhm_nm=0.05)
model = SpectrumModel(
    plasma=plasma,
    atomic_db=db,
    instrument=instrument,
    lambda_min=200.0,
    lambda_max=800.0,
    delta_lambda=0.01,
    path_length_m=0.01,
)
wavelength, intensity = model.compute_spectrum()
```

## 5. Common First-Run Issues

| Issue | Fix |
|-------|-----|
| First run is slow | Reduce `lambda_max_nm − lambda_min_nm` or raise `delta_lambda_nm` while iterating on the config; widen later. |
| Atomic database not found | Use the bundled DB (`ASD_da/libs_production.db`). Only build a new one when you need elements outside the bundled coverage. |
| No lines visible in plot | Check that the chosen wavelength range overlaps strong lines for your species (Fe I has dense coverage near 240–270 nm; alkali/alkaline earth resonance lines are easier in the visible). |

---

## See Also

- [Quick Start: Real Data](Quick_Start_Real_Data.md) — extracting `T`,
  `n_e`, and composition from measured spectra.
- [Physics: Equations](../physics/Equations.md) — what `SpectrumModel`
  actually computes and why.
- [Physics: Assumptions and Validity](../physics/Assumptions_And_Validity.md)
  — the LTE / optically thin / single-zone assumptions baked into the
  forward model.
- [User Guide](User_Guide.md) — configuration field reference for forward
  modeling and inversion.
- [Manifold Generation Guide](Manifold_Generation_Guide.md) — bulk
  forward-model evaluation on a parameter grid.
