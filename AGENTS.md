# Repository Guidelines

## Project Structure & Module Organization
- `cflibs/` contains the library source (core physics, atomic data, plasma state, radiation, instrument models, CLI).
- `tests/` holds pytest suites and fixtures.
- `docs/` contains user-facing documentation and API references.
- `examples/` has runnable configs and sample workflows.
- Root scripts like `datagen_v2.py` and `manifold-generator.py` are utilities for database and manifold generation.

## Build, Test, and Development Commands
- `pip install -e "[dev]"` installs the project in editable mode with dev tools.
- `pytest tests/ -v` runs the full test suite.
- `pytest tests/test_plasma.py -v` runs a single test file.
- `pytest -m "not requires_db"` skips database-dependent tests.
- `black cflibs/ tests/` formats code.
- `ruff check cflibs/ tests/` runs linting.
- `mypy cflibs/` runs type checks.

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
- Use type hints for public functions and NumPy-style docstrings for public APIs.
- Test files: `test_*.py`; test functions: `test_*`.
- Prefer descriptive, physics-aligned naming (e.g., `saha_equation_ratio`, `compute_spectrum`).

## Testing Guidelines
- Framework: pytest with optional `pytest-cov` and `pytest-benchmark`.
- Coverage target for new code: >80% (see `CONTRIBUTING.md`).
- Use markers to scope runs (e.g., `requires_db`, `requires_jax`, `slow`).

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
   bd sync
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
