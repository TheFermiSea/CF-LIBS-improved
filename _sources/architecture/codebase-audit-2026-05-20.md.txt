# CF-LIBS-improved codebase audit — 2026-05-20

Companion to [`codebase-audit-2026-05-19.md`](codebase-audit-2026-05-19.md).
This pass covers the four dimensions the prior audit deliberately skipped:
test coverage, type strictness, security/dependency hygiene, and
deprecation/dead-code purge. Same four-parallel-agent format
(feature-dev:code-explorer, code-reviewer, general-purpose, Explore).

---

## Top 10 actions

| # | Action | Severity | Effort | Bead |
|---|---|---|---|---|
| 1 | Remove 7 already-passing mypy `disable_error_code` entries (`assignment`, `arg-type`, `attr-defined`, `return-value`, `dict-item`, `operator`, `call-arg`) — pure config flip, zero source changes | high | XS | `type-1` → `CF-LIBS-improved-cclj` |
| 2 | Delete `IterativeCFLIBSSolverJax` + its parity tests (deprecated per ADR-0001, ~488 LOC) | high | S | `dead-1` → `CF-LIBS-improved-bbng` |
| 3 | Add test coverage for `cflibs/hardware/` (entire package untested) | high | M | `test-1` → `CF-LIBS-improved-sx1e` |
| 4 | Add direct tests for `cflibs/benchmark/{checkpoint,composition_eval}.py` (PR #189 module split, post-rescue logic) | high | S | `test-2` → `CF-LIBS-improved-2pxb` |
| 5 | Reactivate `tests/radiation/test_forward_model_parity.py` (skipped on a now-completed migration) | high | S | `test-3` → `CF-LIBS-improved-bmgq` |
| 6 | Add `__all__` to 15 high-signal public-API modules (hardware/, runtime/, profiles, metrics, …) | medium | S | `type-2` → `CF-LIBS-improved-wkw1` |
| 7 | Add return-type hints to ~50 public functions in top-10 files (unlocks `disallow_untyped_defs=true` later) | medium | M | `type-3` → `CF-LIBS-improved-ejsb` |
| 8 | Pin third-party GitHub Actions by SHA, not tag (peaceiris, benchmark-action) | medium | XS | `sec-1` → `CF-LIBS-improved-y6b5` |
| 9 | Add `timeout=` to `scripts/beadhub-bootstrap.sh:64` `urlopen()` call (only one in the repo missing it) | low | XS | `sec-2` → `CF-LIBS-improved-m8ar` |
| 10 | Replace `time.sleep`-based race tests in `tests/test_streaming.py` with `threading.Event` / `Queue` sync | medium | M | `test-4` → `CF-LIBS-improved-b0y9` |

Plus: `test-5` → `CF-LIBS-improved-289x` (numpy-path Stark T-factor test) and
`test-X-followups` → `CF-LIBS-improved-8vav` (7 lower-priority untested
modules + stale markers).

**Combined LOC delta if items 1, 2, 6 land:** roughly **−500 source LOC** (488 from solver-jax removal, plus the 7-line config change, plus 30 LOC of `__all__` additions which technically grow LOC but solidify the public surface).

**CI risk surface if items 8 & 9 land:** the only two security-defensive items where the cost is one line each and the upside is real (supply-chain action-tag pinning, hung-bootstrap protection).

---

## 1. Test coverage (`tests/`)

### test-1 — `cflibs/hardware/` entire package untested **(HIGH / M)**
8 files: `abc.py`, `manager.py`, `factory.py`, `laser.py`, `spectrograph.py`, `stages.py`, `flow.py`. Public symbols: `HardwareManager`, `HardwareFactory`, `HardwareComponent` ABC, 4 protocol interfaces, 4 concrete `*Hardware` classes, `HardwareStatus` enum. The only `test_distributed_mcmc.py` mention is a comment about GPU hardware, not the package. Need unit tests with mock drivers.

### test-2 — `cflibs/benchmark/{checkpoint,composition_eval}.py` direct coverage missing **(HIGH / S)**
Both were extracted from `unified.py` in PR #189; both contain logic added in PR #186 (atomic write, run_id-in-filename, evaluate_composition_workflow). `tests/benchmark/test_results_parquet.py` covers them only indirectly. Need: `tests/benchmark/test_checkpoint.py` (atomic write, SIGKILL tolerance, seq increment) and `tests/benchmark/test_composition_eval.py` (Aitchison/tier/posterior-diagnostics population).

### test-3 — Stale `pytest.skip` in `tests/radiation/test_forward_model_parity.py:174` **(HIGH / S)**
`pytest.skip("Bayesian-vs-kernel parity is T1-6 scope; bridge helper landed in T1-2.")` skips permanently. T1-6 is **complete** (the migration test in `tests/inversion/test_bayesian_forward_model_kernel_migration.py` is the guard). Reactivate the body.

### test-4 — `tests/test_streaming.py` race-prone sleep patterns **(MED / M)**
13 `time.sleep` calls including a 1-second sleep at line 942 with downstream assertions like `len(results) >= 5`. Will be flaky on a loaded CI machine. Replace with `threading.Event` / `Queue` sync.

### test-5 — Stark T-factor regression guard is JAX-only **(HIGH / S)**
`tests/radiation/test_stark_t_factor_toggle.py` pins the `(T/T_ref)^(-alpha)` formula but only in the JAX path. The numpy `stark_hwhm` path used by the non-JAX solver gets no explicit T-dependence test. Add a numpy-only test in `tests/test_stark.py`.

### Additional findings (filed as one P3 follow-up bead `test-X`)

- `cflibs/inversion/solve/spectral_refiner.py` tested only via the flat shim path
- `cflibs/benchmark/bayesian_sparse_id.py` zero tests
- `cflibs/core/cache.py` LRUCache untested
- `cflibs/inversion/preprocess/wavelength_calibration.py` zero tests
- `cflibs/inversion/candidate_prefilter.py` untested despite "mandatory" CLAUDE.md note
- Stale xfail in `tests/benchmark/test_jax_workflows.py:228` (Bayesian MCMC timeout, references closed bead `359q`)
- `tests/test_pca.py:152` tolerance `rtol=0.15` too loose

---

## 2. Type strictness (`pyproject.toml [tool.mypy]`)

### type-1 — All 7 disabled error codes on `cflibs.*` already pass **(HIGH / XS)**

Current override:
```toml
[[tool.mypy.overrides]]
module = "cflibs.*"
disable_error_code = ["assignment", "arg-type", "attr-defined", "return-value", "dict-item", "operator", "call-arg"]
```

**Correction (2026-05-20 implementation pass):** the type-strictness agent
claimed this was zero-cost. Direct verification — removing the override
and running `mypy cflibs/ --no-incremental` — surfaces **187 errors
across 45 files**. Top offenders: `cflibs/benchmark/unified.py`,
`cflibs/inversion/solve/iterative.py`, `cflibs/inversion/forward_models/`.
The audit's "zero-cost flip" framing was wrong; the override stays. The
right shape of work is per-category: re-enable one code at a time, fix
the resulting errors, drop that entry. `attr-defined` / `call-arg`
likely the cheapest first; `arg-type` / `dict-item` are the bulk of the
real work. Bead `cclj` updated accordingly.

### type-2 — Add `__all__` to 15 modules with 3+ public symbols **(MED / S)**
Top offenders by public-symbol count: `cflibs/hardware/abc.py` (42), `cflibs/inversion/runtime/streaming.py` (41), `cflibs/inversion/runtime/temporal.py` (33), `cflibs/radiation/profiles.py` (29), `cflibs/inversion/preprocess/outliers.py` (26), `cflibs/benchmark/metrics.py` (25). Concrete cleanup that signals stable API.

### type-3 — Add return-type hints to ~50 public functions **(MED / M)**
Top 10 files by missing-return-annotation count: `cflibs/radiation/profiles.py` (11), `cflibs/cli/main.py` (10), `cflibs/plasma/partition.py` (5), `cflibs/manifold/{generator,loader}.py` (4 each), `cflibs/inversion/common/pca.py` (4), `cflibs/inversion/solve/bayesian/forward.py` (3). All single-line additions; unlocks `disallow_untyped_defs = true`.

### type-4 — `Dict[str, Any]` → `TypedDict` candidates **(MED / M)**
3 highest-impact files: `cflibs/core/jax_runtime.py` (27 `Any`), `cflibs/inversion/solve/bayesian/atomic.py` (21 `Any`), `cflibs/inversion/forward_models/__init__.py` (19 `Any`). Define 3–5 Protocols/TypedDicts per file; rest follows.

### type-5 — Stub packages missing from `dev` extra **(LOW / XS)**
Only `types-PyYAML>=6.0.0` declared. Direct codebase imports of `numpy`, `scipy`, `pandas`, `matplotlib` would benefit from `types-numpy`, `types-scipy`, etc. — 4 one-line additions to `[project.optional-dependencies] dev`.

### Ratchet plan (next 2 PRs)

PR A (2–3h, low risk): items 1, 5; add return types to 50 functions; add `__all__` to 15 modules.
PR B (3–5h, medium risk): replace `Any` in `forward_models/`, `jax_runtime`, `bayesian/atomic` with Protocols/TypedDicts. Enable `disallow_untyped_defs = true` at the end.

Defer: `jaxtyping` shape annotations (large surface, low immediate ROI).

---

## 3. Security & dependency hygiene

### sec-1 — Third-party GitHub Actions pinned by tag, not SHA **(MED / XS)**
- `.github/workflows/docs.yml:36` → `peaceiris/actions-gh-pages@v3`
- `.github/workflows/performance.yml:38` → `benchmark-action/github-action-benchmark@v1` (with `auto-push: true`, explicit write permission)

Both can be tag-hijacked. Pin to full commit SHAs.

### sec-2 — `urlopen()` without `timeout=` in `scripts/beadhub-bootstrap.sh:64` **(LOW / XS)**
The only `urlopen` in the entire repo lacking `timeout=`. Every other instance in `cflibs/pds/cache.py:117`, `scripts/fetch_nist_reference_spectra.py:75`, etc., is properly bounded. One-line fix.

### sec-3 — NumPyro / JAX extras have no upper bound **(LOW / S, defensive)**
`pyproject.toml` `bayesian` extra: `numpyro>=0.14.0`, `arviz>=0.17.0`; `jax-cpu` extra: `jax[cpu]>=0.4.30`. A future major JAX/NumPyro release can silently break solver determinism. Run `uv run pip-audit -r <(uv pip compile pyproject.toml --all-extras)` in CI, and pin reasonable upper bounds based on the test matrix.

### sec-4 — `CFLIBS_BENCH_CHECKPOINT_PATH` no canonicalization **(LOW / XS, document-only)**
`cflibs/benchmark/composition_eval.py:369-384` writes `<env-supplied-path>.parts/` with `mkdir(parents=True, exist_ok=True)`. Trusted-operator input — not a runtime hazard, but worth documenting that the path is **not** sanitized against `..` traversal.

### Known non-issues (verified clean)

- `subprocess.*` calls: all list-form, no `shell=True` anywhere in `cflibs/` or `scripts/`.
- `ast.parse` / `ast.literal_eval`: all legitimate (evolution AST-blocklist, observability deserialization).
- CI workflows: no `pull_request_target` anywhere; `secrets.PYPI_API_TOKEN` only in `release.yml` (gated on release).
- No hardcoded credentials (grep clean for `sk-`, `AKIA`, `ghp_`, `xoxb-`, etc.).

---

## 4. Deprecation / dead code

### dead-1 — `IterativeCFLIBSSolverJax` + parity tests **(HIGH / S, ~488 LOC)**
`cflibs/inversion/solve/iterative.py:1696-1905` (210 LOC class) + `tests/inversion/test_solver_jax_parity.py` (278 LOC). Class is `DeprecationWarning`-marked per ADR-0001; the fallback flag `HAS_JAX_ITERATIVE_SOLVER` in `cflibs/benchmark/workflows.py` already tolerates absence. Single highest-LOC, lowest-risk deletion.

### dead-2 — `cflibs/benchmarks/` (trailing-s) shim package **(MED / XS, ~45 LOC)**
Already filed in the prior audit as `arch-4` / `CF-LIBS-improved-4ig9`. Worth restating: 4 files, all pure re-exports from `cflibs/benchmark/`, deprecation note inside says "2026-04 cleanup". Zero external callers.

### dead-3 — Inversion flat-shim files **(LOW / XS, ~12 LOC)**
Beyond the previously-noted 25+ shims, four are 3-line re-export stubs that have zero non-shim callers: `cflibs/inversion/closed_form_solver.py`, `hybrid.py`, `joint_optimizer.py`, `spectral_refiner.py`. Subsumed by the `arch-1` retarget bead — once internal code stops routing through shims, these can be deleted.

### dead-4 — `_*_lazy` wrapper sprawl in `cflibs/benchmark/workflows.py` **(MED / M)**
6+ functions: `_select_aalto_cases_lazy`, `_estimate_effective_rp_lazy`, `_load_real_spectra_lazy`, `_run_boltzmann_pipeline_lazy`, `_pipeline_joint_softmax_lazy`, `_pipeline_hybrid_manifold_lazy`. Each is a 10-line crutch that reflectively loads a script from `scripts/`. This is the library-imports-scripts layering inversion flagged in the prior audit. Defer until the workflows.py split (covered by `arch-2`).

### dead-5 — Merged-branch hygiene **(housekeeping, 0 LOC)**
`git branch -r --merged origin/dev` returns 23 fully-merged feature branches on origin. Safe to prune.

### dead-6 — Stale scripts in `scripts/` **(LOW / L, ~500–1000 LOC, needs investigation)**
40 scripts with no edits in 6+ months, no references in active workflows/tests/CI. Needs per-script triage; not a single-PR move.

---

## Out-of-scope but worth flagging

- **`cflibs/evolution/`**: LLM-driven algorithm-search tooling per CLAUDE.md. `git log --since "3 months ago" -- cflibs/evolution/` returns commits, so it's not abandoned, but a focused review would clarify whether the AST blocklist still aligns with the physics-only constraint.
- **`docs/archive/legacy/`**: 15+ historical docs flagged in the prior audit (docs-8) as needing a DEPRECATED marker. Still open.

---

## Bead-tag → real-ID mapping

| Tag | Bead ID | Title |
|---|---|---|
| `type-1` | `CF-LIBS-improved-cclj` | remove 7 mypy disable_error_code entries |
| `type-2` | `CF-LIBS-improved-wkw1` | add `__all__` to 15 modules |
| `type-3` | `CF-LIBS-improved-ejsb` | add return types to ~50 public functions |
| `test-1` | `CF-LIBS-improved-sx1e` | `cflibs/hardware/` test coverage |
| `test-2` | `CF-LIBS-improved-2pxb` | `checkpoint.py` + `composition_eval.py` direct tests |
| `test-3` | `CF-LIBS-improved-bmgq` | reactivate `test_forward_model_parity.py` |
| `test-4` | `CF-LIBS-improved-b0y9` | replace `time.sleep` race patterns in `test_streaming.py` |
| `test-5` | `CF-LIBS-improved-289x` | numpy-path Stark T-factor test |
| `test-X` | `CF-LIBS-improved-8vav` | 7-module follow-up coverage backfill |
| `sec-1` | `CF-LIBS-improved-y6b5` | pin third-party Actions by SHA |
| `sec-2` | `CF-LIBS-improved-m8ar` | add `timeout=` to `beadhub-bootstrap.sh` urlopen |
| `dead-1` | `CF-LIBS-improved-bbng` | delete `IterativeCFLIBSSolverJax` + tests |
