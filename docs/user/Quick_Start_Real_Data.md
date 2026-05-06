# Quick Start: Analyzing Real LIBS Spectra

This guide is for the primary use case of CF-LIBS: **taking a measured LIBS
spectrum and extracting plasma temperature, electron density, and elemental
composition without calibration standards.**

If your goal is instead to generate a synthetic spectrum from known plasma
parameters, see [Quick Start: Synthetic Spectra](Quick_Start_Synthetic.md).
For the underlying equations and the assumptions the inversion makes about
your plasma, see the [physics reference](../physics/README.md).

## Audience

You have:

- A measured spectrum (intensity vs. wavelength) from a LIBS experiment.
- A list of elements you suspect are present in the sample.
- An idea of what spectrometer you used (resolution or resolving power).

You want:

- Plasma temperature `T` (kelvin or eV).
- Electron density `n_e` (cm⁻³).
- Mass fractions `C_s` for each element.
- Uncertainties on all of the above.

You do **not** need a calibration curve, a matrix-matched standard, or a
reference spectrum. That is the point of CF-LIBS.

---

## 1. Confirm the Setup

From the repository root:

```bash
cflibs doctor
```

This prints which Python packages, atomic database, and example files are
present. If `cflibs` is not on your `PATH`, install in editable mode:

```bash
uv venv --python 3.12
uv pip install -e ".[dev,jax-cpu,hdf5]"
```

Confirm the bundled atomic database resolves:

```bash
ls ASD_da/libs_production.db
```

If it is missing, the doctor command tells you the next step. The bundled
database covers the elements needed for the examples below; only run
`cflibs generate-db` when you need data outside the bundled coverage.

---

## 2. Prepare Your Spectrum

CF-LIBS reads CSV files with **one wavelength column and one intensity
column**. The header row controls which columns are used.

Accepted wavelength column names: `wavelength`, `wavelength_nm`, `wl`,
`lambda`, `lambda_nm`.

Accepted intensity column names: `intensity`, `intensity_W_m2_nm_sr`, `I`,
`counts`, `signal`, `spectrum`, `flux`.

Wavelengths must be in **nanometers**. Intensities can be in any consistent
unit — calibrated radiance, raw detector counts, or arbitrary units. The
CF-LIBS algorithm uses **ratios** of line intensities, so the absolute
intensity scale cancels out for temperature and composition. Self-absorption
correction does need an absolute scale; that is documented separately
(see [Physics: Self-Absorption](../physics/Equations.md#self-absorption-and-curve-of-growth)).

Minimal valid input:

```csv
wavelength_nm,counts
247.94,12.0
247.95,15.3
247.96,42.1
...
```

The repository ships with twelve real single-element LIBS spectra in
`data/aalto_libs/elements/` (Al, Co, Cr, Cu, Fe, Mg, Mn, Ni, Pb, Sn, Ti, V,
Zn) and a larger mineral library in `data/aalto_libs/minerals/`. Use these
to verify your installation before running on your own data.

---

## 3. Run the End-to-End Inversion

The simplest workflow is `cflibs analyze`. It does line detection, line
selection, the iterative CF-LIBS solve, and quality scoring in one step
with sensible defaults.

```bash
cflibs analyze data/aalto_libs/elements/Fe_spectrum.csv \
    --elements Fe \
    --db-path ASD_da/libs_production.db \
    --uncertainty analytical
```

Expected output (numbers will differ slightly with version):

```
Temperature : 9821 ± 412 K
n_e         : 1.18e+17 cm^-3
Converged   : True (7 iterations)

Concentrations:
  Fe  : 1.0000 ± 0.0000
```

What just happened:

1. **Line detection** scanned the spectrum for peaks above
   `min_peak_height` and matched them to NIST transitions for the requested
   elements within `wavelength_tolerance_nm`.
2. **Line selection** kept only lines with high SNR, sufficient energy
   spread for a stable Boltzmann plot, and good spectral isolation.
   Resonance lines (high self-absorption risk) were excluded.
3. **Iterative CF-LIBS solver** alternated between (a) fitting a
   common-slope Boltzmann plot for each element to recover `T`, (b)
   applying the Saha correction to fold ionic intercepts back to the
   neutral plane, (c) applying the closure equation to turn intercepts
   into mass fractions, and (d) updating `n_e` via Saha until both `T` and
   `n_e` stopped changing.
4. **Uncertainty propagation** took the covariance of the Boltzmann fit
   through the closure equation analytically.

If `Converged: False` appears, see [Section 7](#7-when-it-does-not-converge).

---

## 4. Multi-Element Samples

Most real samples are mixtures. The element list is your **prior knowledge**:
include every element you have reason to suspect.

```bash
cflibs analyze my_steel_spectrum.csv \
    --elements Fe,Cr,Ni,Mn,C,Si \
    --db-path ASD_da/libs_production.db \
    --uncertainty analytical \
    --format table
```

Output format options:

- `--format table` — human-readable summary with concentrations and
  McWhirter LTE warning if violated. (default)
- `--format json` — machine-readable; preserves uncertainties and quality
  metrics.
- `--format csv` — `element,concentration,uncertainty` rows for
  spreadsheet ingestion.

**Do not** dump the entire periodic table into `--elements`. The
combinatorial line-matching cost grows fast, weak/spurious matches
proliferate, and the closure equation becomes ill-conditioned. Use the
elements you have physical reason to expect.

If you genuinely do not know what is in the sample, use the identification
tools first (`cflibs invert` with hybrid identification, or
`cflibs.inversion.identify` from Python) to get a candidate list, then run
`cflibs analyze` with that list.

---

## 5. Tuning the Inversion (When the Defaults Are Wrong)

The defaults in `cflibs analyze` work for many spectra. When they don't,
switch to `cflibs invert` and use a config file. This exposes every knob
the iterative solver respects.

```bash
cflibs invert my_spectrum.csv \
    --config my_inversion.yaml \
    --output result.json
```

Minimal `my_inversion.yaml`:

```yaml
atomic_database: ASD_da/libs_production.db

analysis:
  elements: [Fe, Cr, Ni, Mn, Si]

  # Detection (peak finding + DB matching)
  min_peak_height: 0.01            # fraction of max intensity
  peak_width_nm: 0.2               # integration width per peak
  wavelength_tolerance_nm: 0.1     # peak-to-DB match window
  resolving_power: 5000.0          # spectrometer R = lambda / dlambda
  min_relative_intensity: 100.0    # filter weak DB lines

  # Selection (which detected lines feed the Boltzmann plot)
  min_snr: 10.0                    # reject lines below this SNR
  min_energy_spread_ev: 2.0        # required E_k range per element
  min_lines_per_element: 3         # below this, element is skipped
  exclude_resonance: true          # avoid self-absorbed transitions
  isolation_wavelength_nm: 0.1     # min separation from neighbors
  max_lines_per_element: 20

  # Solver
  closure_mode: standard           # standard | matrix | oxide
  max_iterations: 20
  t_tolerance_k: 100.0
  ne_tolerance_frac: 0.1
  pressure_pa: 101325.0            # buffer-gas pressure
```

### Choosing `closure_mode`

- `standard` — `Σ_s C_s = 1`. Use when you measured all major elements.
  Concentrations come out as elemental mass fractions.
- `matrix` — fix the matrix element (e.g. Fe in steel) at a known
  fraction; remaining elements scale relative to it. Add
  `closure_kwargs: { matrix_element: Fe, matrix_fraction: 0.7 }`.
- `oxide` — for geological samples where elements report as oxides
  (SiO₂, Al₂O₃, …). Add `closure_kwargs: { oxide_elements: [Si, Al, Ca, Fe, Mg] }`.

The mathematics of each mode is in
[Physics: Closure Equation](../physics/Equations.md#closure-equation).

### Choosing detection thresholds

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Very few peaks detected | `min_peak_height` too strict for noisy spectrum | Lower to `0.005` or `0.002` |
| Many spurious matches | `wavelength_tolerance_nm` too loose for your resolution | Tighten to `0.05` or `0.02` |
| Lines split across two DB matches | `wavelength_tolerance_nm` too tight | Loosen toward `0.1`–`0.2` |
| All lines from one element rejected | `min_lines_per_element` not met | Lower threshold or check element wavelength range |
| Boltzmann R² is poor | Self-absorption on strong lines | Set `exclude_resonance: true` and raise `min_energy_spread_ev` |

### Setting `resolving_power`

Either set `resolving_power` (dimensionless `λ/Δλ`) or omit it and let the
fixed-FWHM mode use a pre-set `instrument.resolution_fwhm_nm`. Echelle
spectrometers should use `resolving_power`; fixed-grating monochromators
should use `resolution_fwhm_nm`.

---

## 6. Uncertainty Quantification

Three modes are exposed:

```bash
cflibs analyze spectrum.csv --elements Fe,Cu --uncertainty analytical
cflibs analyze spectrum.csv --elements Fe,Cu --uncertainty mc
```

| Mode | Cost | What it captures | When to use |
|------|------|------------------|-------------|
| `none` | — | Nothing | Quick parameter-of-merit only |
| `analytical` | ~free | Boltzmann-fit covariance propagated through closure linearly | Default for production |
| `mc` | 200× a single solve | Full nonlinear pipeline including line-detection variance | Validation runs, anomaly checks |
| `bayesian` (separate command) | minutes | Joint posterior over `T`, `n_e`, `{C_s}` with proper credible intervals | Publication-grade results, model comparison |

Bayesian mode requires `pip install cflibs[bayesian]` (NumPyro + dynesty)
and is invoked as a separate subcommand:

```bash
cflibs bayesian spectrum.csv \
    --elements Fe,Cr,Ni \
    --db-path ASD_da/libs_production.db \
    --samples 2000 --chains 4 \
    --output posterior.nc
```

The output `posterior.nc` is an ArviZ-compatible NetCDF trace. See
[User Guide § Bayesian](User_Guide.md#bayesian-inference) for the full
posterior workflow including R-hat / ESS diagnostics, posterior-predictive
checks, and corner plots.

---

## 7. When It Does Not Converge

The most common failure modes and what they indicate:

### McWhirter criterion not satisfied

The output prints `WARNING: McWhirter criterion NOT satisfied`. This means
the inferred `n_e` is below the threshold required for LTE to hold given
the observed line set:

```
n_e ≥ 1.6 × 10¹² × √T × (ΔE)³   [cm⁻³, T in K, ΔE in eV]
```

LTE is a foundational assumption of the iterative CF-LIBS solver.
Violation means the recovered `T` and `C_s` are physically suspect even if
the math converged. See
[Physics: Assumptions and Validity](../physics/Assumptions_And_Validity.md)
for the full criterion derivation and when LIBS plasmas typically exit
LTE.

### Fewer than `min_lines_per_element` for an element

The element is silently dropped from the Boltzmann fit. Check the warnings
in `--log-level DEBUG`. Typical fixes:

- Widen the wavelength range so more transitions of that element fall in
  band.
- Lower `min_snr` if the spectrum is genuinely low signal.
- Check that you spelled the element symbol correctly.

### Solver oscillates / does not converge in `max_iterations`

Usually means the Boltzmann plot is non-linear, which in turn means LTE is
violated, self-absorption is corrupting strong lines, or the line set is
too short in energy spread. Try:

- `min_energy_spread_ev: 3.0` to enforce more lever arm on the slope.
- `exclude_resonance: true`.
- Inspect the Boltzmann plot directly (see Python example in
  [User Guide § Diagnostics](User_Guide.md#inversion-diagnostics)).

### Inferred composition does not sum to 1 in unexpected ways

In `standard` closure, the residual is the unmodeled mass — it tells you
elements are present that you did not include in `--elements`. Add them
and re-run. In `matrix` closure, scaling is anchored to the matrix element
fraction you supplied; check that fraction is correct.

---

## 8. Batch Processing

Once a single spectrum works, scale up:

```bash
cflibs batch ./spectra/ \
    --elements Fe,Cr,Ni,Mn,Si \
    --db-path ASD_da/libs_production.db \
    --output batch_results.csv
```

Every `.csv` in `./spectra/` runs through the same analysis pipeline and
results land in one summary table with one row per spectrum. Failures are
logged and skipped, not aborted.

For thousands of spectra and JAX-capable hardware, the manifold workflow
is faster — see [Manifold Generation Guide](Manifold_Generation_Guide.md).

---

## 9. Python API (for Custom Workflows)

When the CLI is not enough, the same pipeline is available from Python:

```python
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.line_detection import detect_line_observations
from cflibs.inversion.line_selection import LineSelector
from cflibs.inversion.solver import IterativeCFLIBSSolver
from cflibs.io.spectrum import load_spectrum

db = AtomicDatabase("ASD_da/libs_production.db")
wl, intensity = load_spectrum("my_spectrum.csv")

detection = detect_line_observations(
    wavelength=wl,
    intensity=intensity,
    atomic_db=db,
    elements=["Fe", "Cr", "Ni"],
    wavelength_tolerance_nm=0.1,
    resolving_power=5000.0,
)
selection = LineSelector().select(
    detection.observations,
    resonance_lines=detection.resonance_lines,
)

solver = IterativeCFLIBSSolver(atomic_db=db)
result = solver.solve_with_uncertainty(selection.selected_lines)

print(f"T  = {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K")
print(f"ne = {result.electron_density_cm3:.2e} cm^-3")
for el, c in result.concentrations.items():
    err = result.concentration_uncertainties.get(el, 0.0)
    print(f"  {el}: {c:.4f} ± {err:.4f}")
```

The same `result` object exposes per-element Boltzmann fit objects, the
selected line list, and the convergence history — useful for diagnostics.
See [API Reference](../reference/API_Reference.md#inversion-module).

---

## 10. What This Pipeline Assumes About Your Plasma

The iterative CF-LIBS solver is only as good as its physical assumptions.
Before quoting a number from a CF-LIBS analysis, you should know that:

- The plasma is in **local thermodynamic equilibrium** (LTE), so the same
  `T` describes ionization, excitation, and (if relevant) molecular
  partitioning.
- The plasma is **optically thin**, or self-absorption corrections are
  applied to the lines that are not.
- The plasma is **homogeneous in `T`, `n_e`, and composition** along the
  line-of-sight integration column. Single-zone LTE is the explicit model.
- The plasma is **spatially stationary** over the gate window (no major
  evolution between integration windows).
- All elements that contribute >1% mass fraction are in your
  `--elements` list (otherwise standard closure is biased).

These are not always true. Many published CF-LIBS results that disagree
with reference values disagree because one of these assumptions failed.

The full statement of every assumption, when each holds, when it fails,
and the diagnostics CF-LIBS exposes to detect failure, is in
[Physics: Assumptions and Validity](../physics/Assumptions_And_Validity.md).

---

## See Also

- [Quick Start: Synthetic Spectra](Quick_Start_Synthetic.md) — generating a
  spectrum from known parameters, for experimental design or model checks.
- [User Guide](User_Guide.md) — full configuration reference, Python API
  for advanced cases, troubleshooting catalogue.
- [Physics: Equations](../physics/Equations.md) — exact equations the
  forward model and inversion solve.
- [Physics: Assumptions and Validity](../physics/Assumptions_And_Validity.md)
  — every assumption baked into the algorithm and how to test it.
- [API Reference](../reference/API_Reference.md) — module-by-module API.
