# CF-LIBS

A physics-based Python library for **Calibration-Free Laser-Induced Breakdown Spectroscopy (CF-LIBS)**: forward modeling, inversion, and analysis of LIBS plasmas in local thermodynamic equilibrium (LTE).

CF-LIBS implements the full measurement-to-inference pipeline: synthetic spectrum generation from plasma parameters → spectral preprocessing and line identification → plasma diagnostics (temperature, electron density, elemental composition) with uncertainty quantification. The library prioritizes physics-aligned algorithms over learned models — shipped code contains no neural networks or machine-learning layers.

**Use CF-LIBS for:**
- Elemental composition analysis via calibration-free LIBS
- Plasma diagnostics (electron temperature and density)
- Synthetic spectrum generation for experimental design
- Forward modeling with Saha-Boltzmann equilibrium
- Bayesian and deterministic inversion
- Benchmarking identification algorithms (ALIAS, comb, correlation)
- Real-world LIBS data validation against NIST atomic data

**Status:** Active development. Core forward model, iterative CF-LIBS inversion pipeline, Bayesian inference, and manifold pre-computation are mature. Recent work: inversion package reorganized into 6 physics-aligned sub-packages; ruff TID251 rule enforces physics-only constraint in shipped code.

## New User Path

CF-LIBS' primary use case is **extracting plasma temperature, electron density, and
elemental composition from a measured LIBS spectrum without calibration standards.**
If that is what you want, start here:

→ **[Quick Start: Real Data](docs/user/Quick_Start_Real_Data.md)** — analyze a measured spectrum.

If instead you want to generate a synthetic spectrum from known plasma parameters
(experimental design, sanity checks, benchmarking), start here:

→ **[Quick Start: Synthetic Spectra](docs/user/Quick_Start_Synthetic.md)** — forward modeling.

Sanity check your install and run the bundled real-data example:

```bash
cflibs doctor
cflibs analyze data/aalto_libs/elements/Fe_spectrum.csv \
    --elements Fe --db-path ASD_da/libs_production.db --uncertainty analytical
```

For the equations CF-LIBS evaluates and the physical assumptions of the inversion,
see the [physics reference](docs/physics/README.md). For a one-page map of all
documentation, see [docs/README.md](docs/README.md).

---

## Quick Start

### Installation

```bash
# Core (NumPy/SciPy)
pip install -e .

# Add JAX (CPU, cross-platform)
pip install -e ".[jax-cpu]"

# Add JAX (Apple Silicon Metal)
uv pip install -e ".[jax-metal]"

# Full dev stack (CPU, local machine)
uv pip install -e ".[local]"

# Full stack for cluster (NVIDIA GPU, MPI)
uv pip install -e ".[cluster]"
```

### First Forward Model

Save as `my_config.yaml`:
```yaml
plasma:
  model: single_zone_lte
  Te: 10000    # K
  ne: 1.0e17   # cm^-3
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

```bash
cflibs forward my_config.yaml --output spectrum.csv
```

---

## Architecture

CF-LIBS is organized into 18 top-level packages reflecting the physics pipeline.

### Core Physics

| Package | Purpose |
|---------|---------|
| `cflibs/plasma/` | LTE plasma state, Saha-Boltzmann equilibrium, partition functions |
| `cflibs/radiation/` | Forward model: line emissivity, line profiles (Gaussian/Doppler/Voigt), instrument convolution |
| `cflibs/atomic/` | SQLite-backed atomic database (lines, energy levels, partition functions, Stark parameters); connection pooling and query caching |
| `cflibs/instrument/` | Instrument response: fixed FWHM or resolving-power modes, wavelength calibration |

### Inversion Pipeline

The `cflibs/inversion/` package is organized into 6 sub-packages reflecting the measurement→physics→inference workflow:

| Sub-package | Purpose |
|-------------|---------|
| `common/` | Data structures (`LineObservation`, `BoltzmannFitResult`, `ElementIdentification`), PCA utilities |
| `preprocess/` | Baseline removal, noise estimation, deconvolution, outlier detection (SAM/MAD) |
| `physics/` | Boltzmann fitting, Saha correction, closure equations (standard/matrix/oxide/ILR), CDSB self-absorption, Stark broadening, line selection metrics, uncertainty propagation |
| `identify/` | Element identification: ALIAS cross-correlation, spectral comb, correlation-based matching, hybrid NNLS+ALIAS, BIC model selection |
| `solve/` | Plasma parameter inference: iterative CF-LIBS loop, closed-form ILR solver, Bayesian (NumPyro NUTS + dynesty), manifold coarse-to-fine |
| `runtime/` | Real-time DAQ streaming, temporal gate optimization, hardware interface |

Backward-compatible shims at old flat paths (`from cflibs.inversion.solver import X`) still work. Full API: [docs/reference/API_Reference.md](docs/reference/API_Reference.md).

### Validation & Analysis

| Package | Purpose |
|---------|---------|
| `cflibs/manifold/` | JAX-accelerated spectrum pre-computation over parameter grids; HDF5/Zarr storage; FAISS indexing for fast inference |
| `cflibs/benchmark/` | Synthetic corpus generation, composition metrics (Aitchison distance, ILR), dataset adapters, unified benchmark harness |
| `cflibs/validation/` | Round-trip golden spectrum tests, NIST parity checks, real-data workflows |
| `cflibs/evolution/` | LLM-driven hierarchical evolutionary search for algorithm optimization (development tool, not shipped) |

### Utilities

| Package | Purpose |
|---------|---------|
| `cflibs/core/` | Physical constants (SI + plasma units), YAML config loading/validation, JAX backend detection |
| `cflibs/io/` | Spectrum I/O (CSV, JSON, HDF5), exporters |
| `cflibs/cli/` | CLI subcommands: forward, invert, analyze, bayesian, batch, generate-db, generate-manifold |
| `cflibs/pds/` | ChemCam/SuperCam planetary data system interface |
| `cflibs/hpc/` | Cluster utilities, SLURM integration |
| `cflibs/evolution/` | LLM-driven algorithm optimization tooling (hierarchical-ES, blocklist scanner). The only place under `cflibs/` where ML libraries are permitted. |

Rust components in `native/cflibs-core/` (comb matching, partition functions) and `native/rust-plugin/` (DAQ interface).

### Physics Constraint

**Shipped CF-LIBS code is physics-only.** Machine learning is forbidden in `cflibs/` production code (ruff TID251 rule in `pyproject.toml`); the only exception is `cflibs/evolution/`, which holds optimization-process tooling (LLM perturbation generation, council synthesis) that proposes candidate code but never ships in the final algorithm.

This constraint ensures reproducibility and interpretability of inferred plasma parameters. See [docs/development/Evolution_Framework.md](docs/development/Evolution_Framework.md) for the full constraint specification and enforcement.

---

## CLI Workflows

### Forward Modeling
```bash
cflibs forward examples/config_example.yaml --output spectrum.csv
```

### Inversion
```bash
cflibs invert spectrum.csv --elements Fe Cu --config examples/inversion_config_example.yaml
cflibs analyze spectrum.csv --elements Fe,Cu --output result.json
cflibs bayesian spectrum.csv --elements Fe,Cu --output posterior.json
```

### Batch Processing
```bash
cflibs batch ./spectra --elements Fe,Cu --output-dir output/batch_results
```

### Manifold Generation (GPU-accelerated)
```bash
cflibs generate-manifold examples/manifold_config_example.yaml --progress
cflibs generate-db  # Rebuild atomic database from NIST ASD
```

For full details on CLI options, run `cflibs --help` or see [CLI documentation](docs/reference/API_Reference.md).

---

## Data Workflows

Atomic database generation and validation:

```bash
# Generate atomic database from NIST ASD (slow, hours)
python datagen_v2.py

# Build synthetic spectrum corpus
python scripts/build_synthetic_id_corpus.py \
  --db-path ASD_da/libs_production.db \
  --output-dir output/synthetic_corpus

# Benchmark line identifiers (ALIAS, comb, etc.)
python scripts/benchmark_synthetic_identifiers.py \
  --dataset-path output/synthetic_corpus/ak3_1_3_corpus_v1/corpus.json \
  --db-path ASD_da/libs_production.db \
  --output-dir output/synthetic_benchmark/ak3_1_4_v1

# Validate against NIST
python scripts/validate_nist_parity.py \
  --element Fe --T 0.8 --ne 1e17 \
  --wl-min 220 --wl-max 265 --resolving-power 1000

# Validate real experimental data
python scripts/validate_real_data.py \
  --datasets steel_245nm FeNi_380nm --no-plots
```

---

## Physics Model

CF-LIBS implements:

- **Forward model:** Saha–Boltzmann equilibrium → level populations →
  line emissivity → Voigt line profiles (Doppler + Stark) → optically
  thin transport → instrument convolution → synthetic spectrum.
- **Iterative inversion:** Boltzmann plot (common-slope, multi-element)
  → Saha correction (ionic lines → neutral plane) → closure equation
  (`standard` / `matrix` / `oxide`) → self-consistent `n_e` update,
  iterated to convergence.
- **Uncertainties:** analytical propagation through the Boltzmann
  covariance, Monte Carlo over the full pipeline, or full Bayesian
  inference (NumPyro NUTS + dynesty nested sampling).

The complete equations, symbol definitions, and citations live in
[**docs/physics/Equations.md**](docs/physics/Equations.md). The physical
assumptions (LTE, optical thinness, single-zone uniformity, McWhirter
criterion, etc.), their regimes of validity, and the diagnostics CF-LIBS
exposes for detecting violations are in
[**docs/physics/Assumptions_And_Validity.md**](docs/physics/Assumptions_And_Validity.md).
A step-by-step walkthrough of the iterative solver is in
[**docs/physics/Inversion_Algorithm.md**](docs/physics/Inversion_Algorithm.md).

---

## Development

### Quality Gates

```bash
ruff check cflibs/ tests/
black --check cflibs/
mypy cflibs/
pytest tests/ -x -q -m "not slow and not requires_db"
```

### Run Tests

```bash
pytest tests/test_plasma.py -v                          # single file
pytest tests/test_solver.py::test_name -v               # single test
pytest -m "not requires_db"                             # skip DB-dependent tests
pytest -m "unit"                                        # unit tests only
pytest tests/ --cov=cflibs --cov-report=html            # coverage report
```

Test markers: `requires_db`, `requires_jax`, `requires_bayesian`, `requires_uncertainty`, `requires_rust`, `slow`, `unit`, `integration`, `physics`, `nist_parity`.

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding style, type hints, physics documentation standards, and contribution workflow. See [AGENTS.md](AGENTS.md) for agent (Scribe, Architect, etc.) roles and responsibilities.

---

## Key Files

User and scientific documentation:

- **[docs/README.md](docs/README.md)** — Top-level documentation map.
- **[docs/user/](docs/user/)** — Quick starts, user guide, hardware/manifold/echelle guides.
- **[docs/physics/](docs/physics/)** — Equations, assumptions and validity, inversion algorithm.
- **[docs/reference/](docs/reference/)** — API reference, codebase architecture, database generation.

Project / dev / contribution:

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Coding standards, contribution guidelines.
- **[docs/development/](docs/development/)** — Evolution framework, deployment, internal dev notes.
- **[pyproject.toml](pyproject.toml)** — Dependencies, build config, ruff TID251 physics-only rule.
- **[examples/](examples/)** — YAML config templates for forward model and inversion.
- **[tests/](tests/)** — Pytest suite with physics benchmarks and NIST parity checks.

AI tooling (not user documentation):

- **[CLAUDE.md](CLAUDE.md)** — Operating manual for Claude Code agents.
- **[AGENTS.md](AGENTS.md)** — Agent roles and responsibilities.

---

## References

### Calibration-Free LIBS Theory

- **Ciucci et al. (2009)** — "Evaluation of laser-induced breakdown spectroscopy for the quantitative analysis of marble"
- **Fortes et al. (2018)** — "Calibration-free laser-induced breakdown spectroscopy"
- **Hou et al. (2021)** — "A comprehensive review on calibration-free laser-induced breakdown spectroscopy"

### Saha-Boltzmann Equilibrium

- **Salzmann (1998)** — "Atomic physics in hot plasmas" (partition functions, ionization balance)
- **Colombant & Tonon (1973)** — "X-ray emission in high-intensity laser-produced plasmas"

### Line Broadening

- **Griem (1974)** — "Spectral Line Broadening by Plasmas" (Stark, Doppler, Lorentzian mechanics)
- **Landi & Degl'Innocenti (1999)** — "Stark broadening of hydrogen lines"

---

## License

MIT License. See [LICENSE](/LICENSE) for details.

## Contact

For questions, issues, or contributions, open an issue or contact the maintainers at squires.b@gmail.com.
