# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

CF-LIBS is a physics-based Calibration-Free Laser-Induced Breakdown Spectroscopy library. It implements the full CF-LIBS pipeline: forward modeling (plasma parameters → synthetic spectrum), inversion (measured spectrum → plasma composition), and GPU-accelerated manifold pre-computation for fast inference.

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
cflibs generate-manifold examples/manifold_config_example.yaml --progress
```

## Data And Validation Workflows

```bash
python datagen_v2.py                              # generate atomic DB (hours-long)
python scripts/validate_nist_parity.py --element Fe --T 0.8 --ne 1e17 --wl-min 220 --wl-max 265 --resolving-power 1000
python scripts/run_nist_validation.py --db ASD_da/libs_production.db --output output/validation/nist_crosscheck_report.json
python scripts/validate_real_data.py --datasets steel_245nm FeNi_380nm --no-plots
python scripts/calibrate_alias.py --db-path ASD_da/libs_production.db --data-dir data --output-dir output/calibration
python scripts/generate_model_library.py chunk --chunk-id 0 --n-chunks 8 --output-dir output/model_library
python scripts/generate_model_library.py consolidate --output-dir output/model_library
python scripts/generate_model_library.py build-index --output-dir output/model_library
python scripts/generate_model_library.py submit --n-chunks 32 --output-dir output/model_library
```

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
| `cflibs/inversion/` | `IterativeCFLIBSSolver`, Boltzmann fitting (with SIGMA_CLIP/RANSAC/HUBER outlier rejection), line detection/selection, closure (standard/matrix/oxide modes), self-absorption correction, deconvolution, uncertainty |
| `cflibs/manifold/` | `ManifoldGenerator` (JAX-accelerated), HDF5/Zarr storage backends, loader |
| `cflibs/instrument/` | Instrument models (fixed FWHM or resolving-power mode), response curves |
| `cflibs/core/` | Constants (SI + plasma units), config loading/validation (YAML), JAX runtime detection, logging |
| `cflibs/io/` | Spectrum I/O, config loading, exporters (CSV, JSON, HDF5) |
| `cflibs/cli/` | `main.py` — argparse CLI with subcommands: forward, invert, analyze, bayesian, batch, generate-db, generate-manifold |
| `cflibs/validation/` | Round-trip validation (`GoldenSpectrum` with known ground truth), NIST parity checks |
| `cflibs/hpc/` | Cluster utilities, SLURM integration |

### Key Abstractions

- **`AtomicDataSource`** (ABC in `atomic/database.py`): pluggable backend for atomic data queries (SQLite, NIST API, HDF5)
- **`SolverStrategy`** (ABC): pluggable plasma solvers (Saha-Boltzmann, multi-zone, non-LTE)
- **`InstrumentModelProtocol`** (Protocol): structural typing for instrument models
- **`SingleZoneLTEPlasma`** (`plasma/state.py`): core plasma state with composition conversion helpers (mass fractions ↔ number fractions ↔ number densities)

### Advanced Inversion Modules

The `cflibs/inversion/` package contains several advanced modules beyond the core solver. The table below highlights higher-level analysis and advanced features. For core utilities like line detection, preprocessing, and Boltzmann fitting, see the full module list in cflibs/inversion/.

| Module | Role |
|--------|------|
| `closure.py` | `ClosureEquation` with three modes: **standard** (ΣC_s = 1 normalization), **matrix** (inter-element correction factors before normalization), **oxide** (converts metal concentrations to oxide wt% before closure, for geological samples) |
| `joint_optimizer.py` | Replaces iterative closure with direct optimization: softmax parameterization ensures Σc_i = 1 by construction, jointly optimizes T, n_e, and concentrations via L-BFGS-B |
| `bayesian.py` | NumPyro-based Bayesian inference: MCMC (NUTS) and nested sampling (via dynesty) for posterior distributions over T, n_e, and compositions with full uncertainty quantification |
| `streaming.py` | Real-time spectral processing: ring-buffer accumulation, running-average Boltzmann fits, DAQ interface integration for live plasma monitoring |
| `temporal.py` | Time-resolved LIBS analysis: gate-delay sweep processing, temporal evolution of T and n_e, optimal integration window selection |
| `pinn.py` | Physics-Informed Neural Network: neural surrogate for the forward model trained with physics loss terms (Saha-Boltzmann consistency, closure constraint) |
| `matrix_effects.py` | Inter-element matrix effect corrections: empirical correction factors and physics-based models for non-ideal plasma interactions |
| `quality.py` | Quality metrics for inversion results: residual diagnostics, Boltzmann plot R², line-by-line fit quality |
| `pca.py` | Principal Component Analysis for spectral dimensionality reduction |
| `pls.py` | Partial Least Squares regression for rapid chemometric predictions |
| `transfer.py` | Transfer learning / calibration transfer between instruments |
| `hybrid.py` | Hybrid CF-LIBS + chemometric approaches |

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
   bdh :force-sync  # only needed when you changed beads state; bdh mutations auto-sync
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

<!-- BEADHUB:START -->
## BeadHub Coordination Rules

This project uses `bdh` for multi-agent coordination and issue tracking, `bdh` is a wrapper on top of `bd` (beads). Commands starting with : like `bdh :status` are managed by `bdh`. Other commands are sent to `bd`.

`.beadhub` is per-worktree and is intentionally gitignored. New worktrees do
not inherit the root checkout's BeadHub identity, so before using `bdh` in a
fresh worktree you must bootstrap it:

```bash
test -f .beadhub || bdh :init --project cf-libs --role developer --update
```

In this repo's local OSS setup, the BeadHub server uses `http://localhost:8000`
as the API base (with endpoints under `/v1`). Do not append `/api` unless you
are intentionally switching this workspace to BeadHub Cloud or another server
that expects that base path.

You are expected to work and coordinate with a team of agents. ALWAYS prioritize the team vs your particular task.

You will see notifications telling you that other agents have written mails or chat messages, or are waiting for you. NEVER ignore notifications. It is rude towards your fellow agents. Do not be rude.

Your goal is for the team to succeed in the shared project.

The active project policy as well as the expected behaviour associated to your role is shown via `bdh :policy`.

## Start Here (Every Session)

```bash
bdh :policy    # READ CAREFULLY and follow diligently
bdh :status    # who am I? (alias/workspace/role) + team status
bdh ready      # find unblocked work
```

Use `bdh :help` for bdh-specific help.

## Rules

- Always use `bdh` (not `bd`) so work is coordinated
- Default to mail (`bdh :aweb mail list|open|send`) for coordination; use chat (`bdh :aweb chat pending|open|send-and-wait|send-and-leave|history|extend-wait`) when you need a conversation with another agent.
- Respond immediately to WAITING notifications — someone is blocked.
- Notifications are for YOU, the agent, not for the human.
- Don't overwrite the work of other agents without coordinating first.
- ALWAYS check what other agents are working on with bdh :status which will tell you which beads they have claimed and what files they are working on (reservations).
- `bdh` derives your identity from the `.beadhub` file in the current worktree. If you run it from another directory you will be impersonating another agent, do not do that.
- Prioritize good communication — your goal is for the team to succeed

## Using mail

Mail is fire-and-forget — use it for status updates, handoffs, and non-blocking questions.

```bash
bdh :aweb mail send <alias> "message"                         # Send a message
bdh :aweb mail send <alias> "message" --subject "API design"  # With subject
bdh :aweb mail list                                           # Check your inbox
bdh :aweb mail open <alias>                                   # Read & acknowledge
```

## Using chat

Chat sessions are persistent per participant pair. Use `--start-conversation` when initiating a new exchange (longer wait timeout).

**Starting a conversation:**

```bash
bdh :aweb chat send-and-wait <alias> "question" --start-conversation
```

**Replying (when someone is waiting for you):**

```bash
bdh :aweb chat send-and-wait <alias> "response"
```

**Final reply (you don't need their answer):**

```bash
bdh :aweb chat send-and-leave <alias> "thanks, got it"
```

**Other commands:**

```bash
bdh :aweb chat pending          # List conversations with unread messages
bdh :aweb chat open <alias>     # Read unread messages
bdh :aweb chat history <alias>  # Full conversation history
bdh :aweb chat extend-wait <alias> "need more time"  # Ask for patience
```
<!-- BEADHUB:END -->