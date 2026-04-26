# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

CF-LIBS is a physics-based Calibration-Free Laser-Induced Breakdown Spectroscopy library. It implements the full CF-LIBS pipeline: forward modeling (plasma parameters → synthetic spectrum), inversion (measured spectrum → plasma composition), and GPU-accelerated manifold pre-computation for fast inference.

## Physics-Only Constraint

**HARD CONSTRAINT:** The shipped CF-LIBS algorithm must not import or use: `sklearn`, `torch`, `tensorflow`, `keras`, `flax`, `equinox`, `transformers`, `jax.nn`, or `jax.experimental.stax`. Machine learning is allowed **only** in `cflibs/evolution/` (LLM-driven algorithm optimization tooling). Enforcement happens at two levels:

1. **Ruff TID251 static rule** — `pyproject.toml` bans these APIs from the shipped codebase via `[tool.ruff.lint.flake8-tidy-imports.banned-api]`.
2. **AST blocklist scanner** — `cflibs/evolution/evaluator.py` parses evolved candidate code and rejects any that violates the ban before physics evaluation (fitness = -inf).

See bead CF-LIBS-improved-3fy3 for the full specification.

## Setup

```bash
uv venv --python 3.12
pip install -e ".[dev]"
uv pip install -e ".[local]"    # Apple Silicon: JAX Metal, h5py, zarr, dev tools
uv pip install -e ".[cluster]"  # NVIDIA GPU: JAX CUDA, h5py, mpi4py
```

## Quality Gates

```bash
ruff check cflibs/ tests/
black --check cflibs/
mypy cflibs/
pytest tests/ -v
JAX_PLATFORMS=cpu pytest tests/    # force CPU backend
```

## Swarm Quality-Gate Workflow (`.swarm/profile.toml`)

For beefcake-loop parity, run gates in this order:

```bash
ruff check cflibs/ tests/
black --check cflibs/
mypy cflibs/                                  # advisory/non-blocking in swarm profile
pytest tests/ -x -q -m "not slow and not requires_db"
```

Auto-fix sequence configured in `.swarm/profile.toml`:

```bash
black cflibs/
ruff check --fix cflibs/
```

## Running Tests

```bash
pytest tests/test_plasma.py -v                  # single file
pytest tests/test_solver.py::test_name -v       # single function
pytest -m "not requires_db"                     # skip DB-dependent tests
pytest -m "not slow"                            # skip slow tests
pytest -m "unit"                                # unit tests only
pytest -m "integration"                         # integration tests only
pytest tests/ -v --benchmark-only               # benchmarks only
pytest tests/ --cov=cflibs --cov-report=html    # coverage report
```

Test markers: `requires_db`, `requires_jax`, `requires_bayesian`, `requires_uncertainty`, `requires_rust`, `slow`, `unit`, `integration`, `physics`, `nist_parity`.

JAX is forced to CPU in `conftest.py` with `jax_enable_x64=True`.

## CLI Workflows

```bash
cflibs generate-db
cflibs forward examples/config_example.yaml --output spectrum.csv
cflibs invert spectrum.csv --elements Fe Cu --config examples/inversion_config_example.yaml
cflibs analyze spectrum.csv --elements Fe,Cu --output result.json
cflibs bayesian spectrum.csv --elements Fe,Cu --output posterior.json
cflibs batch ./spectra --elements Fe,Cu --output-dir output/batch_results
cflibs generate-manifold examples/manifold_config_example.yaml --progress
```

## Evolution Tooling

```bash
python -m cflibs.evolution candidate.py                # Scan candidate for forbidden imports
python -m cflibs.evolution file_a.py file_b.py        # Scan multiple files
cat candidate.py | python -m cflibs.evolution -        # Scan from stdin
```

The AST blocklist scanner enforces the physics-only constraint (see above) on any evolved candidate code produced by the LLM-driven optimization loop. Candidates with violations are rejected before physics evaluation.

## Data And Validation Workflows

```bash
python datagen_v2.py                              # generate atomic DB (hours-long)
python scripts/build_synthetic_id_corpus.py --db-path ASD_da/libs_production.db --output-dir output/synthetic_corpus
python scripts/benchmark_synthetic_identifiers.py --dataset-path output/synthetic_corpus/ak3_1_3_corpus_v1/corpus.json --db-path ASD_da/libs_production.db --output-dir output/synthetic_benchmark/ak3_1_4_v1
python scripts/audit_synthetic_physics.py --db-path ASD_da/libs_production.db --element Fe --output output/validation/synthetic_physics_audit.json
python scripts/validate_nist_parity.py --element Fe --T 0.8 --ne 1e17 --wl-min 220 --wl-max 265 --resolving-power 1000
python scripts/run_nist_validation.py --db ASD_da/libs_production.db --output output/validation/nist_crosscheck_report.json
python scripts/validate_real_data.py --datasets steel_245nm FeNi_380nm --no-plots
python scripts/calibrate_alias.py --db-path ASD_da/libs_production.db --data-dir data --output-dir output/calibration
python scripts/generate_model_library.py chunk --chunk-id 0 --n-chunks 8 --output-dir output/model_library
python scripts/generate_model_library.py consolidate --output-dir output/model_library
python scripts/generate_model_library.py build-index --output-dir output/model_library
python scripts/generate_model_library.py submit --n-chunks 32 --output-dir output/model_library
```

## Code Intelligence (use Serena first)

The Serena MCP server is the **default code-intelligence tool** for this project. It exposes Pyright-grade symbol resolution, blast-radius analysis, and symbolic editing — all of which beat grep/Read/Edit on a Python codebase with heavy ABCs, decorators, and shim re-exports like CF-LIBS.

**Search ladder — try in order, only fall back when the previous tier returns nothing useful:**

1. **Symbol lookup (Serena).** Default `relative_path="cflibs/"` to suppress noise from `tests/`, `scripts/`, `docs/`.
   - "find a class/method by name": `find_symbol(name_path_pattern="IterativeCFLIBSSolver", depth=1, include_body=False)`
   - "find every caller / who uses X": `find_referencing_symbols(name_path="softmax_closure", relative_path="cflibs/")` (LSP-accurate; distinguishes call sites from comments/strings)
   - "what's in this directory at a glance": `get_symbols_overview(relative_path="cflibs/inversion/physics/")`
2. **Intent search.** `colgrep "where inversion line selection happens" -k 20` when you need semantic discovery (no symbol name yet).
3. **Pattern search.** `search_for_pattern` (Serena) for regex with `relative_path` scoping — quieter than raw grep, skips `.git`/`ASD_da/` automatically.
4. **Bash grep / find.** Only for non-Python files (YAML, TOML, Markdown, Rust).

**Editing (use Serena when changing whole symbols):**

- "rewrite this method's body": `find_symbol(name_path="ClassName/method", include_body=True)` then `replace_symbol_body(name_path="ClassName/method", body="...")`. Do NOT Read+Edit a 1000-line file when only one method is changing.
- "delete this function and check nothing breaks": `safe_delete_symbol(name_path="...")` — refuses if references exist.
- "rename across all files": `rename_symbol(name_path="OldName", new_name="NewName")` — LSP-coordinated, follows shim re-exports correctly.
- For sub-symbol edits (a few lines inside a method): `replace_content` with regex; cheaper than rewriting the whole body.

**Project memory.** Persistent context lives in `.serena/memories/` and survives sessions. `list_memories` to enumerate, `read_memory(name)` to load, `write_memory(name, content)` for new facts, `edit_memory(name, …)` for incremental updates. The seeded set covers physics-only constraint, inversion sub-package map, JAX backend matrix, evolution blocklist, suggested commands, code style. New non-obvious facts learned mid-session should be persisted there.

**When to NOT reach for Serena:**
- Reading a tiny config file (just use Read).
- Running tests, formatters, git commands (Bash).
- Operations on the Rust crates in `native/` (Serena's pyright doesn't cover Rust).

## Architecture

### Data Flow

**Forward model:** `PlasmaState(T,ne,composition)` → `SahaBoltzmannSolver` (ionization/excitation balance via Saha equation for ion ratios + Boltzmann distribution for level populations) → `calculate_line_emissivity()` (ε = hc/4πλ · A_ki · n_k) → Gaussian broadening via `profiles.py` → instrument convolution → synthetic spectrum.

**Inversion:** Measured spectrum → preprocessing (baseline removal, noise estimation) → peak detection → line identification (ALIAS/correlation/comb matching) → `IterativeCFLIBSSolver` { Boltzmann plot: ln(Iλ/gA) vs E_k → temperature from slope → Saha correction (map ionic lines to neutral plane) → multi-element common-slope fit → closure equation (Σ C_s = 1) → charge/pressure balance → update n_e → iterate until convergence } → `CFLIBSResult(T, ne, concentrations, uncertainties)`.

**Manifold:** JAX `jit`/`vmap` batch-generates spectra over a parameter grid (T, ne, compositions) → stored as HDF5 or Zarr → enables fast nearest-neighbor inference via FAISS index.

### Module Map

| Package | Role |
|---------|------|
| `cflibs/plasma/` | Plasma state (`SingleZoneLTEPlasma`), `SahaBoltzmannSolver`, partition functions (polynomial: log U = Σ aₙ(log T)ⁿ) |
| `cflibs/radiation/` | `SpectrumModel` (forward model orchestrator), line emissivity, profile broadening (Gaussian, Doppler, resolving-power modes) |
| `cflibs/atomic/` | `AtomicDatabase` (SQLite: lines, energy_levels, species_physics, partition_functions), connection pooling, query caching |
| `cflibs/inversion/` | Inversion pipeline — organized into 6 physics-aligned sub-packages (see below) |
| `cflibs/manifold/` | `ManifoldGenerator` (JAX-accelerated), HDF5/Zarr storage backends, loader |
| `cflibs/instrument/` | Instrument models (fixed FWHM or resolving-power mode), response curves |
| `cflibs/core/` | Constants (SI + plasma units), config loading/validation (YAML), JAX runtime detection, logging |
| `cflibs/io/` | Spectrum I/O, config loading, exporters (CSV, JSON, HDF5) |
| `cflibs/cli/` | `main.py` — argparse CLI with subcommands: forward, invert, analyze, bayesian, batch, generate-db, generate-manifold |
| `cflibs/benchmark/` | Unified benchmark harness, synthetic corpus, composition metrics (Aitchison/ILR), dataset adapters |
| `cflibs/validation/` | Round-trip validation (`GoldenSpectrum` with known ground truth), NIST parity checks |
| `cflibs/pds/` | ChemCam/SuperCam planetary data system interface |
| `cflibs/hpc/` | Cluster utilities, SLURM integration |
| `cflibs/evolution/` | Hierarchical-ES driver + physics-only blocklist scanner for LLM-driven algorithm evolution. Optimization-process tooling only (see docs/Evolution_Framework.md). |
| `native/cflibs-core/` | Rust computational core: comb matching, partition functions |
| `native/rust-plugin/` | Rust plugin interface for DAQ |

### Inversion Sub-Packages

The `cflibs/inversion/` package is organized into sub-packages reflecting the CF-LIBS measurement→physics→inference pipeline:

| Sub-package | Role |
|-------------|------|
| `common/` | Pure data structures (`LineObservation`, `BoltzmannFitResult`, `FitMethod`, `ElementIdentification`), PCA pipeline |
| `preprocess/` | Signal processing: baseline removal, noise estimation, deconvolution, wavelength calibration, outlier detection (SAM/MAD) |
| `physics/` | Saha-Boltzmann plasma physics: Boltzmann fitting, closure (standard/matrix/oxide/ILR), CDSB self-absorption, Stark broadening, line selection, quality metrics, uncertainty propagation |
| `identify/` | Element identification: ALIAS, comb, correlation, spectral NNLS, hybrid (NNLS+ALIAS), BIC model selection, line detection (transition matching) |
| `solve/` | Plasma parameter inference: iterative CF-LIBS loop, closed-form ILR solver, Bayesian (NumPyro NUTS + dynesty), joint L-BFGS-B optimizer, manifold coarse-to-fine |
| `runtime/` | Real-time: DAQ streaming, temporal gate optimization, hardware interface |

Backward-compatible shims exist at all old flat paths (`from cflibs.inversion.solver import X` still works).

### Key Abstractions

- **`AtomicDataSource`** (ABC in `atomic/database.py`): pluggable backend for atomic data queries (SQLite, NIST API, HDF5)
- **`SolverStrategy`** (ABC): pluggable plasma solvers (Saha-Boltzmann, multi-zone, non-LTE)
- **`InstrumentModelProtocol`** (Protocol): structural typing for instrument models
- **`SingleZoneLTEPlasma`** (`plasma/state.py`): core plasma state with composition conversion helpers (mass fractions ↔ number fractions ↔ number densities)
- **`BayesianForwardModel`** (`inversion/solve/bayesian.py`): JAX-compiled forward model for MCMC, supports resolving-power mode and Chebyshev baseline
- **`select_candidate_elements`** (`inversion/candidate_prefilter.py`): NNLS-based top-K prefilter for Bayesian MCMC (mandatory — full-element MCMC is intractable)

### JAX Integration

JAX is optional throughout — code gracefully degrades if unavailable. Three backends: `jax-cpu`, `jax-metal` (Apple Silicon; no float64 or complex support), `jax-cuda`. Backend detection in `cflibs/core/jax_runtime.py`. Manifold generation is the primary JAX consumer.

### Uncertainty Propagation

Two approaches in `cflibs/inversion/uncertainty.py`:
1. **Analytical** (via `uncertainties` package): fast, correlation-aware via Boltzmann covariance matrix
2. **Monte Carlo** (`MonteCarloUQ`): full pipeline re-runs with perturbed inputs, captures non-linear effects

Optional Bayesian inference via NumPyro (`cflibs/inversion/bayesian.py`).

## Coding Style

- PEP 8, 100-char line length (Black/Ruff config in `pyproject.toml`)
- Type hints for public functions, NumPy-style docstrings
- Physics-aligned naming (e.g., `saha_equation_ratio`, `compute_spectrum`)

## Cluster Notes

- Use `cflibs generate-manifold` for manifold builds.
- Do not launch the legacy `manifold-generator.py` with `mpirun`/`srun`; it is not MPI-aware.

## Commit Style

Short imperative summary (<=50 chars), optional body explaining what/why. Recent history favors verbs like "Add", "Implement", "fix".

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:

   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```

5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

## Beads Workflow (Native `bd`)

- Use native `bd` commands for issue tracking and session context.
- Session start:
  - `bd prime`
  - `bd memories`
  - `bd ready`
- Task execution:
  - `bd show <BEAD_ID>`
  - `bd comments <BEAD_ID>`
  - `bd update <BEAD_ID> --status in_progress`
  - `bd comment <BEAD_ID> "Completed X, working on Y"`
  - `bd update <BEAD_ID> --status inreview`
