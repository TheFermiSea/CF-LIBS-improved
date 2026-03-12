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
- Multi-node manifold generation:
  - `mpirun -np 3 --hostfile hosts.txt python manifold-generator.py`.
  - `srun -N 3 --gpus-per-node=1 python manifold-generator.py` (SLURM).

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
