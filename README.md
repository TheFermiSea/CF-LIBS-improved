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

Backward-compatible shims at old flat paths (`from cflibs.inversion.solver import X`) still work.

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
| `cflibs/experimental/ml/` | Quarantined ML modules (PINN, PLS, transfer learning); deletion candidates per CF-LIBS-improved-3fy3 |

Rust components in `native/cflibs-core/` (comb matching, partition functions) and `native/rust-plugin/` (DAQ interface).

### Physics Constraint

**Shipped CF-LIBS code is physics-only.** Machine learning is forbidden in `cflibs/` production code (ruff TID251 rule in `pyproject.toml`). ML is allowed only in:
- `cflibs/evolution/` — algorithm optimization tooling
- `cflibs/experimental/ml/` — deletion-candidate research code

This constraint ensures reproducibility and interpretability of inferred plasma parameters. See epic CF-LIBS-improved-3fy3.

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

For full details on CLI options, run `cflibs --help` or see [CLI documentation](/docs/API_Reference.md).

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

### Forward Model

Given plasma state (T_e, n_e, composition), CF-LIBS computes synthetic spectrum via:

1. **Saha-Boltzmann equilibrium** → level populations
   - Boltzmann distribution for excitation (within ionization stage)
   - Saha equation for ionization balance
2. **Line emissivity**: ε_λ = (hc/4πλ) A_ul n_u φ_ul(λ)
   - A_ul: Einstein A coefficient
   - n_u: upper-level population
   - φ_ul(λ): normalized line profile
3. **Line broadening**: Gaussian (thermal) + Lorentzian (collisions, Stark) → Voigt
4. **Instrument convolution**: detector response + spectral resolution
5. **Optical depth**: optional self-absorption via curve-of-growth

### Inversion Model

Measured spectrum → plasma parameters (T_e, n_e, composition) via iterative CF-LIBS solver:

1. **Preprocessing**: baseline removal, noise estimation, peak detection
2. **Line identification**: ALIAS/comb/correlation matching against atomic database
3. **Boltzmann plot**: extract T_e from ln(I_λ/gA) vs. E_k
4. **Saha correction**: map ionic lines to neutral plane
5. **Multi-element fit**: common-slope constraint across elements
6. **Closure equation**: charge/pressure balance + Σ C_s = 1
7. **Iterate**: refine n_e until convergence

**Uncertainties**: analytical propagation via Boltzmann covariance matrix, or Monte Carlo re-sampling.

**Bayesian option**: NumPyro NUTS sampler + dynesty nested sampling for full posterior.

For detailed equations and physics assumptions, see [CLAUDE.md](/CLAUDE.md#physics-model) and physics module docstrings.

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

See [CONTRIBUTING.md](/CONTRIBUTING.md) for coding style, type hints, physics documentation standards, and contribution workflow. See [AGENTS.md](/AGENTS.md) for agent (Scribe, Architect, etc.) roles and responsibilities.

---

## Key Files

- **[CLAUDE.md](/CLAUDE.md)** — Detailed architecture, module map, data flow, CLI workflows (canonical reference for agents)
- **[CONTRIBUTING.md](/CONTRIBUTING.md)** — Coding standards, contribution guidelines
- **[AGENTS.md](/AGENTS.md)** — Agent roles and responsibilities
- **[pyproject.toml](/pyproject.toml)** — Dependencies, build config, ruff TID251 physics-only rule
- **[examples/](/examples)** — YAML config templates for forward model and inversion
- **[tests/](/tests)** — Pytest suite with physics benchmarks and NIST parity checks

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
