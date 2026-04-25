# Quick Start for Scientists

This guide is for users who understand LIBS, spectra, elements, plasma temperature, and
electron density, but do not want to learn the full codebase before getting a result.

The technical documentation is still useful later. Start here first.

## What You Can Do First

CF-LIBS has three practical entry points:

- Generate a synthetic spectrum from a YAML file.
- Analyze a measured or simulated CSV spectrum for a list of candidate elements.
- Batch-process a folder of CSV spectra once one single-spectrum workflow works.

The lowest-friction path is the command line. You can use the Python API later when you
need custom workflows.

## 1. Check Your Setup

From the repository root, run:

```bash
cflibs doctor
```

This checks the Python environment, required packages, the bundled example database, and
prints the first commands to try.

If the `cflibs` command is not found, install the package in editable mode:

```bash
uv venv --python 3.12
uv pip install -e ".[dev,jax-cpu,hdf5]"
```

You can also use:

```bash
just setup-codex
```

## 2. Generate a First Spectrum

Run the included forward-model example:

```bash
cflibs forward examples/config_example.yaml --output spectrum.csv
```

This writes `spectrum.csv` with two columns:

- `wavelength_nm`: wavelength in nanometers.
- `intensity_W_m2_nm_sr`: modeled spectral intensity.

The example uses the bundled atomic database at `ASD_da/libs_production.db`. You do not
need to regenerate the NIST database for a first run.

## 3. Edit the Scientific Inputs

Open `examples/config_example.yaml`. The most important fields are:

```yaml
plasma:
  Te: 10000.0
  ne: 1.0e17
  species:
    - element: Fe
      number_density: 1.0e15
    - element: H
      number_density: 1.0e16

instrument:
  resolution_fwhm_nm: 0.05

spectrum:
  lambda_min_nm: 200.0
  lambda_max_nm: 800.0
  delta_lambda_nm: 0.01
```

Use these as starting points:

- `Te`: electron temperature in kelvin. Picosecond LIBS plasmas are often closer to
  0.5-1.3 eV, approximately 5800-15000 K.
- `ne`: electron density in cm^-3. A practical first range is `1.0e16` to `1.0e18`.
- `species`: elements and approximate number densities. The relative values control the
  composition more than the absolute scale for many first comparisons.
- `resolution_fwhm_nm`: spectrometer resolution. Smaller values produce sharper lines.
- `lambda_min_nm`, `lambda_max_nm`: wavelength window to simulate.
- `delta_lambda_nm`: wavelength spacing. Larger values run faster but resolve less detail.

## 4. Analyze a Spectrum

The repository includes a small real-data example. Run:

```bash
cflibs analyze data/aalto_libs/elements/Fe_spectrum.csv --elements Fe --db-path ASD_da/libs_production.db
```

For your own measured spectrum, use a CSV file with one wavelength column and one
intensity column. Accepted column names include:

- Wavelength: `wavelength`, `wavelength_nm`, `wl`, `lambda`, or `lambda_nm`.
- Intensity: `intensity`, `intensity_W_m2_nm_sr`, `I`, `counts`, `signal`,
  `spectrum`, or `flux`.

Example:

```csv
wavelength_nm,counts
248.30,1200
248.31,1238
248.32,1194
```

Then run:

```bash
cflibs analyze my_spectrum.csv --elements Fe,Cu,Al --db-path ASD_da/libs_production.db
```

The element list is a candidate list. Include elements you believe might be present; do
not include the whole periodic table for a first pass.

## 5. Batch-Process a Folder

Once one spectrum works:

```bash
cflibs batch spectra/ --elements Fe,Cu,Al --db-path ASD_da/libs_production.db --output results.csv
```

This processes every `.csv` file in `spectra/` and writes a summary table.

## Common Problems

### Atomic Database Not Found

Use the bundled database:

```bash
--db-path ASD_da/libs_production.db
```

Only generate a new database when you need a different data source or element subset:

```bash
cflibs generate-db --db-path libs_production.db --elements Fe Cu Al
```

### No Usable Spectral Lines Detected

Try these changes:

- Confirm wavelength units are nanometers.
- Include the correct candidate elements in `--elements`.
- Use a narrower wavelength range around strong known lines.
- Lower the peak threshold in `examples/inversion_config_example.yaml` if the signal is weak.

### The First Run Is Slow

Use a smaller wavelength range or larger wavelength step while learning:

```yaml
spectrum:
  lambda_min_nm: 240.0
  lambda_max_nm: 270.0
  delta_lambda_nm: 0.02
```

After the workflow is correct, increase the range or resolution.

## When to Use the Other Docs

- Use `docs/User_Guide.md` when you are ready for more configuration detail.
- Use `docs/Database_Generation.md` only when you need to rebuild atomic data.
- Use `docs/Manifold_Generation_Guide.md` for large precomputed search libraries.
- Use `docs/API_Reference.md` when writing Python code directly against CF-LIBS.
