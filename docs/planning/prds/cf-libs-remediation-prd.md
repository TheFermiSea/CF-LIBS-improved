# PRD: CF-LIBS Functional Recovery and Performance Hardening

## Summary

Deep review of the CF-LIBS repository found a mix of working physics modules, incomplete production claims, runtime import hazards, and broken CLI/script workflows. This program stabilizes the public entrypoints first, restores advertised capabilities that are currently non-functional, and then completes the unfinished model-library/vector-search path and performance hardening work.

## Verified Findings

- `cflibs.inversion.__init__` eagerly imports optional JAX/NumPyro modules and only catches `ImportError`, so unrelated runtime failures in optional stacks can crash imports of otherwise-usable submodules.
- `cflibs.radiation.profiles` constructs JAX arrays at import time, which crashes on Apple Metal in the current environment with `JaxRuntimeError: UNIMPLEMENTED: default_memory_space is not supported`.
- `cflibs.inversion.hybrid.HybridInverter` defaults to `L-BFGS-B`, but `jax.scipy.optimize.minimize` in the installed JAX only supports `BFGS`; the current code silently falls back to the coarse guess instead of performing fine tuning.
- `cflibs.cli.main.dbgen_cmd` and `cflibs.atomic.database_generator.generate_database` accept `db_path` and filtering arguments but do not forward them into `datagen_v2.py`.
- `scripts/generate_model_library.py` depends on the environment’s editable install instead of the current checkout, validates imports before validating CLI arguments, contains a broken import path, and still emits placeholder zero spectra in chunk generation.
- `cflibs.inversion.correlation_identifier` documents and advertises vector mode, but the public API is incomplete because it accepts only a `vector_index` and does not carry the embedding/model metadata needed to search raw spectra.
- `setup.py` and `pyproject.toml` disagree on Python support and extras, increasing installation drift and making script behavior harder to reason about.

## Goals

- Make public imports and core CLIs safe to use in a default local development environment.
- Ensure advertised database-generation and hybrid-inversion workflows are actually functional.
- Harden script entrypoints so they operate against the current checkout rather than depending on external editable-install state.
- Replace incomplete vector/model-library workflow claims with working implementations and tests.
- Add regression tests for each restored workflow.

## Non-Goals

- Full redesign of the plasma physics core in a single pass.
- Rewriting every inversion algorithm or manifold kernel before stabilizing entrypoints.
- Solving every long-tail optimization opportunity before restoring correctness.

## Quality Gates

These commands must pass for every story:

- `./.venv/bin/python -m ruff check cflibs tests scripts`
- `JAX_PLATFORMS=cpu ./.venv/bin/python -m pytest tests/test_cli.py tests/test_generate_model_library.py tests/test_profiles.py tests/test_hybrid_inversion.py tests/test_correlation_identifier.py -q`

For packaging/script stories, also include:

- `./.venv/bin/python - <<'PY'\nimport cflibs.radiation\nimport cflibs.inversion.hybrid\nprint('imports-ok')\nPY`

For model-library/vector stories, also include:

- `JAX_PLATFORMS=cpu ./.venv/bin/python -m pytest tests/test_manifold.py tests/test_vector_index.py tests/test_correlation_identifier.py -q`

## User Stories

### US-001: Stabilize import-time optional dependency handling

As a user importing CF-LIBS modules, I need optional JAX/Bayesian stacks to fail gracefully so unrelated workflows remain usable.

Acceptance Criteria:

- `cflibs.radiation` imports without triggering JAX backend initialization failures in the reviewed local environment.
- `cflibs.inversion.hybrid` can be imported without optional Bayesian/radiation failures crashing the entire `cflibs.inversion` package.
- Optional feature flags remain accurate when optional modules are unavailable or fail during initialization.
- Regression tests cover subprocess/module import behavior.

### US-002: Repair hybrid optimization so fine tuning actually runs

As a user of hybrid inversion, I need fine tuning to execute instead of silently returning the coarse manifold guess.

Acceptance Criteria:

- `HybridInverter.invert()` uses a supported optimizer path for the default method.
- Unsupported optimizer selections fail clearly or are routed to a tested fallback backend.
- `SpectralFitter.fit()` uses the same backend-selection rules.
- Regression tests verify the default optimization path performs at least one real optimization step.

### US-003: Make database generation CLI honor its own arguments

As a user generating an atomic database, I need `--db-path` and element/filter arguments to affect the generated database.

Acceptance Criteria:

- `cflibs generate-db --db-path ... --elements ...` forwards those choices into the generator.
- `cflibs.atomic.database_generator.generate_database()` respects its function arguments instead of ignoring them.
- `datagen_v2.py` exposes a callable or CLI path that accepts output/filter parameters.
- Regression tests cover argument propagation.

### US-004: Harden script entrypoints and validation order

As a user running repository scripts directly, I need them to work from the current checkout and fail on invalid arguments before optional import/setup work.

Acceptance Criteria:

- `scripts/generate_model_library.py` resolves imports from the current checkout when run as a script.
- CLI validation for negative/invalid arguments runs before optional `cflibs.hpc` imports.
- Broken import paths in the script are corrected.
- Existing script tests pass and new regression tests cover the direct-run path.

### US-005: Replace placeholder model-library chunk generation with real spectra

As a user building a model library, I need chunk generation to emit physical spectra rather than placeholder zeros.

Acceptance Criteria:

- `chunk` mode generates non-zero spectra from a real forward-model path.
- Chunk metadata is sufficient for consolidation and later vector indexing.
- Consolidated libraries carry the data needed by search/inversion workflows.
- Regression tests validate non-placeholder output and schema consistency.

### US-006: Complete vector-mode correlation identification API

As a user of vector-based element identification, I need a public API that can embed measured spectra and use vector search without hidden state.

Acceptance Criteria:

- `CorrelationIdentifier` accepts the embedding/search context required for vector mode.
- `mode="auto"` only selects vector mode when the full vector workflow is available.
- `mode="vector"` returns detections rather than raising `NotImplementedError`.
- Regression tests cover classic, auto, and vector modes.

### US-007: Reconcile packaging metadata and script/runtime assumptions

As a developer installing CF-LIBS, I need a single, coherent packaging contract so editable installs and scripts resolve consistently.

Acceptance Criteria:

- `setup.py` and `pyproject.toml` agree on supported Python versions and extras, or legacy metadata is removed.
- Script/runtime assumptions are documented and tested against the supported install path.
- Development instructions no longer depend on stale editable-install targets.

### US-008: Benchmark and optimize the most expensive audited paths

As a user running larger CF-LIBS workloads, I need the main atomic/inversion/manifold paths to avoid obvious performance cliffs.

Acceptance Criteria:

- The audited hotspots are benchmarked with before/after numbers.
- At least one concrete performance improvement lands in the main forward/inversion/database path.
- Performance-sensitive code paths have regression coverage or benchmarks.

