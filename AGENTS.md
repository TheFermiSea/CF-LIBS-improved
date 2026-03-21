# Repository Guidelines

## Project Structure & Module Organization
- `cflibs/` contains the library source (core physics, atomic data, plasma state, radiation, instrument models, CLI).
- `tests/` holds pytest suites and fixtures.
- `docs/` contains user-facing documentation and API references.
- `examples/` has runnable configs and sample workflows.
- Root scripts like `datagen_v2.py` and `manifold-generator.py` are utilities for database and manifold generation.

## Build, Test, and Development Commands
- `uv venv --python 3.12` creates a virtual environment with `uv`.
- `pip install -e ".[dev]"` installs the project in editable mode with dev tools.
- `uv pip install -e ".[local]"` installs local dev extras (JAX CPU, hdf5, dev tools).
- `uv pip install -e ".[cluster]"` installs cluster extras (JAX CUDA, hdf5, mpi4py).
- `uv pip install -e ".[ci]"` installs CI-equivalent optional dependencies for full local test parity.
- `pytest tests/ -v` runs the full test suite.
- `pytest tests/ -v --benchmark-only` runs benchmark-only tests (used in CI performance workflow).
- `JAX_PLATFORMS=cpu pytest tests/` forces CPU backend for tests.
- `pytest tests/test_plasma.py -v` runs a single test file.
- `pytest tests/test_solver.py::test_name -v` runs a single test function.
- `pytest -m "not requires_db"` skips database-dependent tests.
- `pytest -m "not requires_jax"` skips JAX-dependent tests.
- `pytest -m "not slow"` skips slow tests.
- `pytest -m "unit"` runs only unit tests.
- `pytest -m "integration"` runs only integration tests.
- `black cflibs/ tests/` formats code.
- `black --check cflibs/` checks formatting.
- `ruff check cflibs/ tests/` runs linting.
- `ruff check --fix cflibs/` auto-fixes lint issues where possible.
- `mypy cflibs/` runs type checks.
- `sphinx-build -b html docs docs/_build/html` builds documentation locally.

## Swarm Quality-Gate Workflow
- `.swarm/profile.toml` defines the beefcake-loop gate order used for autonomous verification.
- Gate parity command sequence:
  - `ruff check cflibs/ tests/`
  - `black --check cflibs/`
  - `mypy cflibs/` (advisory/non-blocking in swarm profile)
  - `pytest tests/ -x -q -m "not slow and not requires_db"`
- Auto-fix sequence in `.swarm/profile.toml`:
  - `black cflibs/`
  - `ruff check --fix cflibs/`

## CLI Workflows

- `cflibs generate-db` generates the atomic database via the CLI.
- `python datagen_v2.py` runs the database generator directly.
- `cflibs forward examples/config_example.yaml --output spectrum.csv` generates a synthetic spectrum.
- `cflibs invert spectrum.csv --elements Fe Cu --config examples/inversion_config_example.yaml` runs inversion.
- `cflibs analyze spectrum.csv --elements Fe,Cu --output result.json` runs end-to-end identification + inversion with sensible defaults.
- `cflibs bayesian spectrum.csv --elements Fe,Cu --output posterior.json` runs Bayesian inversion (requires optional bayesian deps).
- `cflibs batch ./spectra --elements Fe,Cu --output-dir output/batch_results` processes a directory of CSV spectra.
- `cflibs generate-manifold examples/manifold_config_example.yaml --progress` builds a spectral manifold.

## Deployment Environment

- `uv venv --python 3.12` creates a virtual environment (Deployment guide).
- `uv pip install -e ".[local]"` installs local extras in the uv-managed env.

## Coding Style & Naming Conventions
- Follow PEP 8 with 100-char line length (Black/Ruff config in `pyproject.toml`).
- Use type hints for public functions and NumPy-style docstrings for public APIs.
- Test files: `test_*.py`; test functions: `test_*`.
- Prefer descriptive, physics-aligned naming (e.g., `saha_equation_ratio`, `compute_spectrum`).

## Testing Guidelines
- Framework: pytest with optional `pytest-cov` and `pytest-benchmark`.
- Coverage target for new code: >80% (see `CONTRIBUTING.md`).
- Use markers to scope runs (e.g., `requires_db`, `requires_jax`, `requires_bayesian`, `requires_uncertainty`, `requires_rust`, `slow`, `unit`, `integration`, `physics`, `nist_parity`).
- Coverage report: `pytest tests/ --cov=cflibs --cov-report=html`.

## CLI & Data Workflows
- `cflibs generate-manifold examples/manifold_config_example.yaml --progress` builds a spectral manifold.
- `python datagen_v2.py` generates the atomic database (long-running).
- `nohup python datagen_v2.py &` runs database generation in the background.
- `python scripts/build_synthetic_id_corpus.py --db-path ASD_da/libs_production.db --output-dir output/synthetic_corpus` builds deterministic synthetic identification corpora.
- `python scripts/benchmark_synthetic_identifiers.py --dataset-path output/synthetic_corpus/ak3_1_3_corpus_v1/corpus.json --db-path ASD_da/libs_production.db --output-dir output/synthetic_benchmark/ak3_1_4_v1` benchmarks ALIAS/comb/correlation identifier paths.
- `python scripts/audit_synthetic_physics.py --db-path ASD_da/libs_production.db --element Fe --output output/validation/synthetic_physics_audit.json` runs equation-level synthetic physics sanity checks.
- `python scripts/validate_nist_parity.py --element Fe --T 0.8 --ne 1e17 --wl-min 220 --wl-max 265 --resolving-power 1000` runs NIST parity validation.
- `python scripts/run_nist_validation.py --db ASD_da/libs_production.db --output output/validation/nist_crosscheck_report.json` runs consolidated NIST cross-check reporting.
- `python scripts/validate_real_data.py --datasets steel_245nm FeNi_380nm --no-plots` validates element ID pipelines against real datasets.
- `python scripts/calibrate_alias.py --db-path ASD_da/libs_production.db --data-dir data --output-dir output/calibration` grid-searches ALIAS thresholds on labeled data.
- `python scripts/generate_model_library.py chunk --chunk-id 0 --n-chunks 8 --output-dir output/model_library` generates one model-library chunk.
- `python scripts/generate_model_library.py consolidate --output-dir output/model_library` merges chunk outputs into one library.
- `python scripts/generate_model_library.py build-index --output-dir output/model_library` builds FAISS search index for the library.
- `python scripts/generate_model_library.py submit --n-chunks 32 --output-dir output/model_library` emits/submits SLURM array jobs for cluster generation.
- Multi-node manifold generation should use `cflibs generate-manifold`; the legacy
  `manifold-generator.py` script is not MPI-aware and should not be launched via
  `mpirun` or `srun` unless explicit MPI support is added first.

## Code Search
- Prefer semantic search with `colgrep` for code discovery and refactors.
- `colgrep "where inversion line selection happens" -k 20` explores intent-level matches.
- `colgrep -e "add_parser(" -F "cli subcommands" cflibs/cli/main.py` combines literal prefilter + semantic ranking.

## Commit & Pull Request Guidelines
- Commit messages: short imperative summary (<=50 chars) with optional body explaining what/why.
- Recent history favors verbs like "Add" or "Implement".
- PRs should include: clear description, linked issues, tests passing, and docs/examples updated when user-facing.
- Run formatting, linting, and type checks before opening a PR.

## Security & Configuration Tips
- Database generation pulls NIST data and can take hours; store generated DB artifacts outside the repo or in ignored paths.
- GPU/JAX extras are optional; choose `jax-cpu`, `jax-metal`, or `jax-cuda` based on the target environment.

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

## Beads Coordination (Native `bd`)
- Use native `bd` issue tracking commands; do not rely on BeadHub/`bdh` wrappers in this repo.
- Session start workflow:
  - `bd prime`
  - `bd memories`
  - `bd ready`
- Per-task context workflow:
  - `bd show <BEAD_ID>`
  - `bd comments <BEAD_ID>`
  - `bd update <BEAD_ID> --status in_progress`
- Progress and completion updates:
  - `bd comment <BEAD_ID> "Completed X, working on Y"`
  - `bd update <BEAD_ID> --status inreview`
