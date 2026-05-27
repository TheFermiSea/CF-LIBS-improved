# Repository Guidelines

## Project Structure & Module Organization
- `cflibs/` contains the library source (core physics, atomic data, plasma state, radiation, instrument models, CLI).
  - `cflibs/evolution/` — LLM-driven algorithm optimization tooling (hierarchical-ES, blocklist scanner). The only place under `cflibs/` where ML libraries are permitted.
- `tests/` holds pytest suites and fixtures.
- `docs/` contains user-facing documentation and API references.
- `examples/` has runnable configs and sample workflows.
- Root scripts like `datagen_v2.py` and `manifold-generator.py` are utilities for database and manifold generation.

## Build, Test, and Development Commands
- `just setup` creates a Python 3.12 `uv` environment with the baseline dev toolchain.
- `just setup-codex` creates a Python 3.12 `uv` environment with Codex-friendly local extras (`dev`, `jax-cpu`, `hdf5`).
- `just check` runs the stable local quality gate (`ruff check` including TID251 physics-only enforcement, `mypy`, fast pytest).
- `just test-fast` runs a CPU-only fast pytest slice that skips DB, Bayesian, Rust, JAX, and slow tests.
- `just typecheck-ty` runs `ty` in exploratory mode; it is not yet a required gate.
- `uv venv --python 3.12` creates a virtual environment with `uv`.
- `pip install -e ".[dev]"` installs the project in editable mode with dev tools.
- `uv pip install -e ".[local]"` installs local dev extras (JAX CPU, hdf5, dev tools).
- `uv pip install -e ".[cluster]"` installs cluster extras (JAX CUDA, hdf5, mpi4py).
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
- `ruff format --check cflibs/ tests/` evaluates Ruff formatter compatibility for a future repo-wide migration.
- `ruff check cflibs/ tests/` runs linting.
- `ruff check --fix cflibs/` auto-fixes lint issues where possible.
- `mypy cflibs/` runs type checks.
- `ty check cflibs --exit-zero` runs the next-generation type checker in advisory mode.
- `sphinx-build -b html docs docs/_build/html` builds documentation locally.

## CLI Workflows

- `cflibs generate-db` generates the atomic database via the CLI.
- `python datagen_v2.py` runs the database generator directly.
- `cflibs forward examples/config_example.yaml --output spectrum.csv` generates a synthetic spectrum.
- `cflibs invert spectrum.csv --elements Fe Cu --config examples/inversion_config_example.yaml` runs inversion.
- `cflibs generate-manifold examples/manifold_config_example.yaml --progress` builds a spectral manifold.

## Deployment Environment

- `uv venv --python 3.12` creates a virtual environment (Deployment guide).
- `uv pip install -e ".[local]"` installs local extras in the uv-managed env.

## Coding Style & Naming Conventions
- Follow PEP 8 with 100-char line length (Black/Ruff config in `pyproject.toml`).
- Prefer `just` recipes for routine local workflows so Codex, local shells, and humans use the same command surface.
- The stable local gate is `just check`; formatting remains available via `just fmt` and `just fmt-check`, but is not yet folded into the default gate.
- Evaluate `ruff format` via `just fmt-ruff-check` before switching the formatter across the repo.
- Use type hints for public functions and NumPy-style docstrings for public APIs.
- Test files: `test_*.py`; test functions: `test_*`.
- Prefer descriptive, physics-aligned naming (e.g., `saha_equation_ratio`, `compute_spectrum`).

### Physics-Only Constraint

The shipped CF-LIBS algorithm is physics-only — no neural networks, no trained models. Full forbidden-library spec, enforcement mechanism (Ruff TID251 + AST scanner), and rationale live in [`docs/Evolution_Framework.md`](docs/Evolution_Framework.md). ML is allowed only in `cflibs/evolution/` (optimization-process tooling).

### Code Intelligence (Serena MCP)

Serena MCP is the default for symbol-aware navigation and editing on this Python codebase. Search ladder:

1. Symbol lookup → `find_symbol`, `find_referencing_symbols`, `get_symbols_overview`. Default `relative_path="cflibs/"`.
2. Intent search → `colgrep` for semantic discovery without a known symbol name.
3. Pattern search → `search_for_pattern` (regex with relative-path scoping).
4. Bash grep → only for non-Python files (YAML, TOML, Rust, Markdown).

For edits to a whole function/method/class body, prefer `replace_symbol_body` over Read+Edit. For renames, use `rename_symbol` (LSP-coordinated; follows the inversion shim re-exports correctly). For caller analysis before refactor, run `find_referencing_symbols` first. Persist non-obvious project facts via `write_memory`/`edit_memory` (see `.serena/memories/` for the seeded set).

Full guidance: see [`CLAUDE.md` § Code Intelligence](CLAUDE.md#code-intelligence-use-serena-first).

## Testing Guidelines
- Framework: pytest with optional `pytest-cov` and `pytest-benchmark`.
- Coverage target for new code: >80% (see `CONTRIBUTING.md`).
- Use markers to scope runs (e.g., `requires_db`, `requires_jax`, `requires_bayesian`, `requires_uncertainty`, `requires_rust`, `slow`, `unit`, `integration`, `physics`, `nist_parity`).
- Coverage report: `pytest tests/ --cov=cflibs --cov-report=html`.

## CLI & Data Workflows
- `cflibs generate-manifold examples/manifold_config_example.yaml --progress` builds a spectral manifold.
- `python datagen_v2.py` generates the atomic database (long-running).
- `nohup python datagen_v2.py &` runs database generation in the background.
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
   bd dolt push  # only if you made bead state changes (claim/complete/update)
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

## Coordination

This project uses **native `bd` (beads)** for issue tracking and persistent context. Do **not** use `bdh` (the BeadHub Go wrapper) or any `bash ./scripts/bdh` wrapper. BeadHub was removed from this project in commit `6b64e93` and there is no BeadHub server configured here. The `bdh` binary adds a server-coordination layer (chat, mail, locks, presence) that this project doesn't run.

## Start Here (Every Session)

```bash
bd prime       # session context: workflow + ready work + memories
bd memories    # list persistent notes
bd ready       # find unblocked work
```

## Rules

- Use `bd` directly. No wrappers, no `bdh`.
- Persistent context goes in `bd remember` (Dolt-backed, survives sessions).
- `bd dolt push` after bead state changes (claim/complete/update) so the federation hub stays in sync.
- Coordination across multiple human-facing sessions on the same repo is handled by branches + PRs, not by an agent-coordination server.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
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
<!-- END BEADS INTEGRATION -->
