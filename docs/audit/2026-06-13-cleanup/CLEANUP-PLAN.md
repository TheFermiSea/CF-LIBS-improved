# CF-LIBS Cleanup & Simplification Plan

**Date:** 2026-06-13
**Branch baseline:** `dev` @ `2214c20` (== `main` per branch-topology memory)
**Synthesized from:** the six 2026-06-13 read-only audit dimensions
(reference-deprecation, dead-legacy-paths, repo-scaffolding, import-shims,
test-redundancy) cross-checked against on-disk dev state.

> **This is a PLAN ONLY.** No cleanup is performed here. Each item lists its
> gate; nothing reference-touching moves until its gate is satisfied.

---

## 1. Executive Summary

### Total cleanup opportunity

| Bucket | Approx. magnitude | When |
|---|---|---|
| Branch/worktree/disk hygiene | ~150 local + ~66 remote branch refs, **~2.4 GB** worktree disk | safe now |
| Superseded build/campaign scripts (archive) | ~6.4 KLOC moved-to-archive (HPC ~3.4K + campaign1 ~3.0K) + ~4.1 KLOC orphaned root drivers | safe now |
| Dead never-run code (orchestration shells) | ~0 LOC safe now; ~1.0 KLOC gated to M3 (lax driver + Jax subclass + benchmark workflow) | mostly M3 |
| Import-shim / facade slimming | ~25–40 LOC deletable now; ~430 LOC gated on a public-API decision (not jitpipe) | now / API decision |
| **Reference-only inversion compute** (run_pipeline path) | **~14,000 LOC** removable in stages, **net ~14,080** after preserving ~268 LOC shared kernels/constants | after M3 + shadow + oracle |
| Reference-stage test retirement | ~4,500–6,000 LOC oracle-only; ~700 LOC dead JAX-flag scaffolding; ~260 LOC mock_db dedup | mixed |

**Headline number:** roughly **14 KLOC of production reference-inversion compute**
becomes removable *in staged steps after jitpipe is promoted (M3)*, on top of
**~10 KLOC of scripts/tests/scaffolding** that is removable or archivable on
lighter gates — but **zero of the 14 KLOC is removable today** because jitpipe
is mid-build (`cflibs/jitpipe/pipeline.py` `run_one`/`run_batch` are J0
skeletons; the `--pipeline {reference,jit}` flag is not wired).

### The 3–5 biggest wins

1. **Branch + worktree massacre (safe now, ~2.4 GB).** ~150 local + ~66 remote
   merged/squashed branch refs and 21+5+30 stale worktrees reclaim **~2.4 GB**
   with zero source risk. The single largest immediate, risk-free win.
2. **Archive superseded benchmark machinery (safe now, ~6.4 KLOC).**
   `scripts/hpc/` (~3.4 KLOC) and `scripts/campaign1/` (~3.0 KLOC) are
   fully superseded by `cflibs/benchmark/scoreboard.py` with zero production/CI
   references — move to `scripts/archive/`.
3. **The 14 KLOC reference-inversion deprecation (the prize, gated on M3).**
   `line_detection.py` (2475) + `wavelength_calibration.py` (1661) +
   `iterative.py` solve body (~3000) + `run_pipeline` orchestrator (~880) + the
   LOW/MED-risk per-stage kernels (~4820) are the real simplification — but only
   after promotion + ≥2-release shadow + an independent oracle.
4. **Dead lax solve path (~1.0 KLOC, gated to M3-Stage-B).** `_solve_lax` +
   `_run_lax_while_loop` + pure_callback closure trio + the deprecated
   `IterativeCFLIBSSolverJax` subclass + its `iterative_jax` benchmark workflow
   **never route in production today** and are explicitly superseded by jitpipe's
   `scan_solve` (ADR-0004 §3 C4) — the cleanest large deletion, deferred only
   because its *component kernels* are reused by jitpipe and its tests are
   jitpipe's parity anchor.
5. **Small bookkeeping sweeps (safe now).** Delete dead `physics/cdsb.py` module
   (+ its test), drop `cflibs/benchmarks/` (trailing-s) stale dir, remove
   `tests/validation/validation/` nested dup, close beads 38xr/4ig9, fix stale
   CLAUDE.md flat-path / manifold-generator claims.

### The one hard constraint — ADR-0004 D6 (the oracle gate)

> The reference pipeline (`cflibs/inversion/pipeline.py::run_pipeline` and every
> stage it orchestrates) is the **frozen parity oracle**. Pre-M3, any
> jit/reference divergence is a *presumed jit bug*. Post-M3 the reference stays
> the parity oracle (`--pipeline=reference`) for **≥2 releases** (bug-fix-only,
> each paired with a parity-test update), then demotes to a validation oracle.
> **Final removal of any reference-touching code requires an independent oracle
> (NIST parity + golden corpus) and a ROADMAP entry.**

Consequence: nothing on the `run_pipeline` path — code *or* its parity tests —
is removed until jitpipe is **PROMOTED (M3)** with an oracle preserved. The
shared JAX kernels jitpipe imports are **never deleted, only relocated**.

---

## 2. Prioritized, Dependency-Ordered Master Table

Sorted by gate: **safe now → once jitpipe default → after M3 promotion →
after 2-release shadow + independent oracle**. Within a gate, lower-risk first.
LOC are approximate. `file:line` from dev @ `2214c20`.

| # | Target | Action | LOC | GATE | Risk | Evidence (dedup'd) |
|---|---|---|---|---|---|---|
| **SAFE NOW** | | | | | | |
| 1 | ~150 local merged/squashed branches (74 `cx/*`, 21 `overhaul/*`, 5 `arch/c*`, `fix/wave*`, `fix/pre-g1..g5`) | remove | 0 (refs) | safe now | low | `git cherry dev <b>`=0 for all `cx/*`/`arch/c*`; `overhaul/*` reached dev via squash PRs (#283/#287); `fix/pre-g*` are dev ancestors. **Exception:** `cx/wave3-cdsb`/`cx/wave3-lineselect` carry 2 unmerged refactor commits — diff first |
| 2 | ~66 remote merged branch refs (`origin/cx/*`, `origin/overhaul/*`, `origin/fix/*`) | remove | 0 (refs) | safe now | low | Remote mirrors of #1. Keep `origin/{main,dev,jitpipe,gh-pages,adr/0002-jittable-pipeline}` + 3 active `fix-*` hotfix branches |
| 3 | `.worktrees/{w1-*,w2-*,w3-*,w4-*,w5-*,w6-*,w7-*}` (21 worktrees, **~1.3 GB**) | remove | 0 (~1.3 GB disk) | safe now | low | All on `overhaul/*` branches in dev. **Exception:** `.worktrees/w7-integration` has 4 tracked mods (`scripts/campaign1/{README,objective}.py`, `tests/{benchmark/test_jax_workflows,campaign1/test_fitness_v2}.py`) — salvage first |
| 4 | `/home/brian/code/wt-pf/{g1..g5}` (5 worktrees, ~171 MB) | remove | 0 (disk) | safe now (salvage g2/g4) | med | `fix/pre-g*` MERGED-anc into dev. **Salvage:** g2-cli-drift + g4-benchpause hold *uncommitted* tracked test fixes (`test_analyze_invert_no_drift.py`, `test_bench_pause.py`) — never actioned from prior audit |
| 5 | `.claude/worktrees/agent-a{0,1,4,7}*` (4 stale, ~part of 964 MB) | archive | 0 (disk) | safe now | med | Locked at `7043445` with stale "locked claude agent (pid …)" marker — verify pid dead, unlock+remove. (26 `wf_*` jitpipe worktrees → **KEEP**, see gated table) |
| 6 | `scripts/hpc/` (6 files: submit_full_campaign, run_benchmark_sweep, generate_synthetic_benchmark, generate_basis_libraries, train_ml_classifier, analyze_benchmark_results) | archive → `scripts/archive/hpc-campaign/` | ~3434 | safe now | low | Self-contained SLURM chain; `rg` external refs = 0 except `pyproject.toml:212` (sklearn TID251 exemption for train_ml_classifier — drop on move). No CI. Superseded by `scoreboard.py` |
| 7 | `scripts/campaign1/` (driver, holdout_eval, knob_space, objective, rescore, splits, README, slurm/) | archive → `scripts/archive/campaign1/` | ~2998 | safe now | low | Optuna TPE knob-search; README documents best==baseline negative result. No prod/CI refs. Preserve README as negative-result record. (Alt: keep as dormant phase-B infra) |
| 8 | Orphaned root benchmark drivers: `run_comprehensive_benchmark.py` (746), `run_experiments.py` (608), `run_experiments_advanced.py` (633), `sweep_alias_fixes.py` (899), `run_accuracy_ablation.py` (1135) | consolidate (archive after per-file diff) | ~4021 | safe now (per-file check) | med | Pre-scoreboard one-shot drivers; near-dup `run_experiments*` pair. Ext refs only to `benchmark/unified.py` + docs. **KEEP** `run_unified_benchmark.py` + `parameter_sweep.py` (still wired). **Memory:** `sweep_alias_fixes.py` is identifier-scoring → archive (benchmark-gate), never blind-delete |
| 9 | `cflibs/inversion/physics/cdsb.py` (CDSBLineObservation, create_cdsb_observation, from_transition) + `tests/test_cdsb.py` (148) | remove | 194 (+148 test) | safe now | low | CDSBPlotter algo already deleted (`cdsb.py:8`). Module imported ONLY by `inversion/__init__.py` re-export + `tests/test_cdsb.py`. **Verified:** `self_absorption.py` mentions CDSB in *docstrings only* (no import); `run_experiments.py`/`run_accuracy_ablation.py` use their own local `_apply_cdsb_pipeline`/`apply_cdsb_correction`, NOT this module. Drop file + test + 3 `__init__` export entries (:100-102, :284-286) |
| 10 | `tests/validation/validation/` (nested dup dir, contains `accuracy/`) | archive/remove | dir | safe now | low | Accidentally-committed nested dup; the pyproject ruff `validation/validation` exclude is already gone, so it's lint-active dead weight. **Verified present** on disk. Confirm not a live fixture first |
| 11 | `cflibs/benchmarks/` (trailing-s) stale dir + bead `CF-LIBS-improved-4ig9` | remove + close bead | ~0 | safe now | low | **Verified:** `git ls-files cflibs/benchmarks/` returns nothing (source shim already deleted); only untracked `__pycache__` remains. Scrub merge-comments in `cflibs/benchmark/__init__.py:97,112,114`, close bead 4ig9 |
| 12 | bead `CF-LIBS-improved-38xr` (kill shim round-trip) | archive (verify+close) | 0 | safe now | low | **Verified done:** all 30 old flat paths raise `ModuleNotFoundError` (e.g. `import cflibs.inversion.solver` fails); acceptance grep `^from cflibs.inversion.[a-z]*$` = 0 matches. Round-trip eliminated by PR #210 (`55a7e68`) + `33e24b0`. Just close it |
| 13 | `cflibs/inversion/solve/bayesian/__init__.py` — private underscore symbols in `__all__` (`_atomic_data_arrays_from_snapshot`, `_compute_instrument_sigma`, `_resolve_total_species_density_cm3`) | remove from `__all__` | ~6 | safe now (verify test deps) | low | Re-exported privates; only `test_bayesian_forward_model_kernel_migration.py` references — keep importable from `.atomic`, stop re-exporting from facade |
| 14 | `cflibs/inversion/__init__.py:220-223` — `ConvergenceStatus`/`JointConvergenceStatus` collision aliasing | inline / document canonical | ~4 | safe now | low | Name defined in BOTH `solve/joint_optimizer.py:67` and `solve/bayesian/priors.py:170`; aggregator disambiguates. Document which is canonical; bundle into #20 aggregator slim |
| 15 | CLAUDE.md stale claims: "shims exist at all old flat paths" + "Cluster Notes" `manifold-generator.py` warning | remove (doc) | 2–3 lines | safe now | low | **Verified:** flat-path import fails (#12); root `manifold-generator.py` does not exist (only canonical `cflibs/manifold/generator.py`). On-disk CLAUDE.md:198 already correct; scrub the contradicting lines |
| 16 | `scripts/archive/migrations/` (20 applied DB migrations) | leave-as-is / optional delete | ~20 files | safe now | low | Already archived; built DB committed at `ASD_da/libs_production.db`. **Caution:** `patch_partition_functions_bc2016.py`/`fit_partition_coefficients.py`/`regenerate_partition_functions.py` touched by recent commits (789f31b, dbf785d) — verify not in-flight before any delete |
| 17 | `docs/audit/2026-06-09-overhaul/*` (10) + `docs/audit/2026-06-10-goalfirst/*` (4) | archive → `docs/archive/audit/` | 14 files | safe now (caveats) | med | Completed-wave history. **Caveat:** `04-pipeline-defaults.md` is the parity ref for campaign1 knob defaults + basis of unmerged `audit/04-pipeline-defaults` branch (31 commits) — disposition that branch first. `05-repo-cleanliness.md` is *this plan's superseded predecessor* |
| **DEPENDS ON A PUBLIC-API DECISION (not jitpipe)** | | | | | | |
| 18 | `cflibs/inversion/solve/bayesian/samplers.py:443,667` — `NumPyroNUTSSampler`/`DynestyNestedSampler` inverted-canonical aliases | consolidate (pick one direction) | ~20 | API decision | med | "Canonical" names have ZERO callers; "legacy" `MCMCSampler`/`NestedSampler` are the real API (`test_bayesian.py`, `benchmark/unified.py:2502`, `bayesian_sparse_id.py:110`). Drop unused aliases OR migrate+rename — alias currently points wrong way |
| 19 | `forward_models/__init__.py` registry: `hermann_two_region` + `lte_with_self_absorption` (+ helpers) | remove | ~120 of 220 | once jitpipe default (low-value) | low | ZERO refs outside `forward_models/__init__.py` + `test_forward_models_registry.py`. Whole package imported only by its own test. Keep `single_zone_lte` + plumbing |
| 20 | `cflibs/inversion/__init__.py` — lazy aggregator (`__getattr__` + `_MODULE_EXPORTS`, ~130 symbols) | retarget-imports then deprecate | ~400 | public-API deprecation decision | med | Near-dead internally (5 `from cflibs.inversion import X` lines repo-wide; 3 are docstrings). BUT documented public API + pinned by `tests/test_import_safety.py`. Retarget 2 internal callers, deprecate on release boundary — do NOT delete outright. Reference-independent |
| 21 | `cflibs/inversion/__init__.py` legacy export entries (SelfAbsorptionCorrector, CurveOfGrowthAnalyzer, estimate_optical_depth_from_intensity_ratio, CDSB×3, IterativeCFLIBSSolverJax) | retarget-imports | ~14 | tie each to its symbol's gate | low | Each removable symbol re-exported twice (`__all__` + lazy map). CDSB entries → safe now (with #9); SelfAbsorptionCorrector → after-oracle; IterativeCFLIBSSolverJax → after M3. Vestigial public surface, not load-bearing (single-maintainer) |
| **ONCE jitpipe IS DEFAULT (benchmark adapter exists)** | | | | | | |
| 22 | `cflibs/benchmark/unified.py:2260-2390` `_iterative_jax_*` + `iterative_jax` workflow registration | remove | ~130 | once jitpipe benchmark adapter replaces it (≈M3) | low | Parallel benchmark workflow distinct from scored scoreboard. Only consumer of deprecated `IterativeCFLIBSSolverJax`. One test xfails ("Agent B solver hasn't landed"). Removal doesn't touch scoreboard scoring |
| **AFTER M3 PROMOTION (Stage-B: reference is bug-fix-only oracle)** | | | | | | |
| 23 | KEEP `jitpipe-*`/`verify-*`/`integ-*` local branches, `origin/jitpipe`, 26 `.claude/worktrees/wf_*` worktrees, `.worktrees/{integ-j8,stage}`, `/tmp/cflibs-residual` | **do-not-prune** until M3 | — | after M3 | high | In-progress jitpipe stage work, genuinely unmerged (`jitpipe-j8` unmerged=7). Pruning loses stage work. `/tmp/cflibs-residual` (13 unmerged commits) needs explicit merge-or-abandon |
| 24 | `iterative.py` lax shells: `_solve_lax` (L2227-2380) + `solve()` routing (L1603-1612) + `_LaxFallback` (L213-220) | remove | ~170 | after M3 (Stage-B; FIRST removal) | med | **Dead in production today:** routing needs `not saha_boltzmann_graph and not diags`, but all presets set `saha_boltzmann_graph=True` + wire Stark diags → always `_solve_python`. ADR §3 C4 supersedes. **KEEP component kernels** (#31) |
| 25 | `iterative.py:721-865` `_run_lax_while_loop` driver | remove | ~145 | after M3 + jitpipe internalizes parity anchor | med | Sole non-test caller is `_solve_lax`. jitpipe `solve.py:342` has its own `scan_solve`. **Blocker:** jitpipe/tests (`solve.py:84,147,246`, `test_iterative_lax.py`) currently use it as parity anchor — move/duplicate the anchor first |
| 26 | `iterative.py:543-657` `_make_closure_callback`/`_closure_via_callback`/`_host_closure` (pure_callback ILR) | remove | ~115 | after M3 + retarget parity test | med | Single non-test caller is `_solve_lax`. Serializes under vmap, non-differentiable (ADR §4 row 9/§6.1); jitpipe uses native masked closures. `test_parity_j7.py` imports `_make_closure_callback` — retarget first |
| 27 | `iterative.py:3023-3246` `IterativeCFLIBSSolverJax` subclass (+ `solve()` override, `_saha_and_fit_jax`, `jax_backend`) | remove | 224 | after M3 (benchmark removal can pair earlier with #22) | med | Self-documented `.. deprecated:: T1-3`, emits DeprecationWarning. Not used by run_pipeline; only `benchmark/unified.py:2309` + tests + `__init__`. Routes to dead `_solve_lax`. Reuses kept kernels |
| 28 | `iterative.py:1711-1739` `_pressure_balance_ne` + fallback branch in `_update_ne_python` (:1773-1779) + lax kernel (:2358) | remove (needs replacement n_e policy) | ~50 | after M3 AND a degradation policy chosen | high | NOT dead — it's the runtime FALLBACK when Stark yields no n_e (reachable in prod even under stark_ne=True). jitpipe keeps pressure balance "only as flagged fallback" (§6.1). **Do not leave a bare hole** — needs deliberate replacement, not blind delete. lax variant dies with #24 |
| 29 | Tests pinning lax path: `test_lax_quality_parity.py` (153) + lax tests in `test_iterative_lax.py` (~380) | remove | ~533 | after M3 | med | ADR §3 C4 retires the driver. `test_parity_j7.py` already supersedes parity. **KEEP** `test_lax_quality_parity` IF joint-WLS doesn't reproduce degenerate-composition keystone-gate semantics — verify vs jitpipe `solve.py` first |
| 30 | Env-flag tests in `test_iterative_lax.py` (lines ~440-540): feature_flag_default_off, *_seeds_from_env_*, *_overrides_env_* (5 tests) | remove | ~100 | after M3 (with #24/#25) | low | Pin `CFLIBS_USE_LAX_WHILE_LOOP`/`CFLIBS_USE_JAX_BOLTZMANN` env plumbing for the superseded path. Pure scaffolding, no physics coverage lost |
| 31 | **SHARED JAX kernels** — `iterative.py`: `_saha_correct_kernel`, `_common_slope_kernel`, `_eval_partition_jax`, `_saha_ratio_per_element`, `_AtomicSnapshot`, `LoopState` (~238) + `boltzmann_jax.py` (239) + `softmax_closure.py` (129) + `closure.OXIDE_OXYGEN_PER_CATION` + `pipeline.py:59-86` `ANALYSIS_PRESETS`/`DEFAULT_ANALYSIS_PRESET` (~30) + `radiation/{kernels,profiles}.py` | **DO-NOT-REMOVE** (relocate-only after M3) | ~268+ (NOT removable) | never delete (relocate-only post-M3) | high | jitpipe imports ALL of these (`fit.py:42`, `solve.py:52-58`, `snapshot.py:422`, `parity.py:51`, `host.py:311`, `forward_id.py:60`, `stark.py:61`). D3 "reuse, never copy" twins. May RELOCATE into `cflibs/jitpipe` or shared `_jax_kernels` after M3 — as a move (retarget-imports), never a delete |
| 32 | KEEP `docs/jax-port/*-consultation.md` (5 docs) | do-not-archive until M3 | — | after M3 | med | Look historical but are ACTIVE jitpipe design inputs (referenced by ADR-0004 + J1/J3 specs). Prevent mistaken deletion |
| **AFTER 2-RELEASE SHADOW + INDEPENDENT ORACLE (Stage-C/Removal)** | | | | | | |
| 33 | `physics/preprocessing.py` (748) + `physics/line_selection.py` (451) + `physics/boltzmann.py` (1127) + `physics/closure.py` (1033 minus shared constant) + `physics/stark_ne.py` (663) + `self_absorption_observable.py` (684) + `self_absorption_inputs.py` (114) | remove (lower-risk per-stage kernels FIRST) | ~4820 | Stage-C: after ≥2-release shadow | med | Reference compute for jitpipe rows 1,5,6,7,8,9; jitpipe `preprocess/detect/fit/stark/selfabs` replace them. Per-stage tolerance contracts ADR §4. closure.py matrix/oxide also feeds `matrix_effects.py` (#37) |
| 34 | `physics/self_absorption.py:657+` `SelfAbsorptionCorrector` + `CurveOfGrowthAnalyzer` | remove (surgical class-level) | ~300–550 | Stage-C: after shadow + oracle | high | Superseded by `ObservableSelfAbsorptionCorrector` (prod wires only that, `iterative.py:1035`). **KEEP live helpers** `correct_via_doublet_ratio`/`find_doublet_pairs` (used by observable path + mirrored by jitpipe `selfabs.py`). Parity-bearing (§4 row 7) |
| 35 | `identify/line_detection.py` (2475) — detect_line_observations + comb + kdet + Rust dispatch | remove | ~2475 | Removal: M3 + 2-rel shadow + independent oracle | high | Sole production identifier (`pipeline.py:550`). Highest-concentration hazard (ADR §4 row 4, R8). Rust comb/kdet (D7) die with it. Survive as oracle longest |
| 36 | `preprocess/wavelength_calibration.py` (1661) — segmented-RANSAC + seam detect | remove | ~1661 | Removal: M3 + 2-rel shadow + independent oracle | high | `pipeline.py:552-554`. MED-HIGH behavioral risk (§4 row 3); mandates ye6t coverage fixtures + reference self-variance study (random_seed=42). Keep as validation oracle longest of front-end |
| 37 | `iterative.py` production solve body: `_solve_python` (L2101-2226) + `_fit_saha_boltzmann_graph` + `_solve_sb_graph_lstsq` (L2818-2888) + `_saha_and_fit_jax` (L3086-3193) + `_build_padded_arrays_from_obs` + `IterativeCFLIBSSolver` wrapper | remove | ~3000 (file ~3247 − ~238 shared) | Removal: M3 + 2-rel shadow + oracle (LAST solve step) | high | Sole production solver (`pipeline.py:804`). GN-step-0 ≡ SB-graph parity anchor (§6.1) — SB-graph kernels must outlive the Python loop. Plus `matrix_effects.py` (1131, on prod path via closure matrix mode) removable with this stage |
| 38 | `pipeline.py::run_pipeline` + `_detect_and_select_lines` + `_run_solver` (~880 orchestration) | remove **dead last** | ~880 (keep ~30 ANALYSIS_PRESETS) | Removal: M3 + ≥2-rel shadow + independent oracle + ROADMAP entry | high | THE governing-rule oracle (ADR §2 D6). `run_pipeline` is the production path. Preserve `ANALYSIS_PRESETS` (#31). Removed after everything else |
| 39 | Reference physics-stage tests duplicated by jitpipe parity: `test_wavelength_calibration.py` (514), `test_solver_jax_parity.py` (273), `test_boltzmann_jax_composition.py` (378), self-absorption-math in `test_self_absorption.py`, SB-graph in `test_boltzmann.py` | retarget/tag oracle-only, then trim | ~1800 oracle-only (~400-600 trimmable) | Removal: after shadow + oracle | high | Each jitpipe J-parity test runs the REAL reference fn as oracle → coverage duplicated. But D6 makes these the parity contract — KEEP (tag "oracle-only") until reference removed. Only safe-to-trim now: genuine same-math-twice numeric cases |
| 40 | `identify/{alias,comb,correlation,spectral_nnls,hybrid,model_selection}.py` JAX-twin tests: `test_alias_jax_nnls.py` (427), `test_alias_jax_boltzmann.py` (241), `test_comb_jax_correlate.py` (260), `test_correlation_jax_classic.py` (224), `test_model_selection_jax.py` (200), `test_nnls_jax_kernel.py` (209) | remove | ~1560 | after M3 (jitpipe identify default) | med | Test opt-in JAX twins (use_jax_correlate/use_jax_classic, default False, never set in pipeline). ADR §1.4 "seeds, not the rewrite". **KEEP** `test_line_detection_jax.py` (431) — tests use_jax_kdet path that IS production-routed |
| 41 | `test_solver.py::test_solver_electron_density_pressure_balance` (L250-317) | remove | ~67 | after M3 | med | Pins 1-atm pressure-balance-as-primary (audit 02-F2 demoted). `test_iterative_ne_stark.py` covers the fallback against independent oracle. Keep rest of test_solver.py |
| 42 | `test_cdsb.py` (148) | remove (with #9) | 148 | after M3 OR safe-now-with-#9 | med | See #9. If cdsb.py module removed now, this test goes with it; otherwise gated to jitpipe selfabs J5 replacement |
| 43 | `cflibs/benchmark/unified.py` (4391) reference-measurement paths vs `scoreboard.py` | consolidate/retire | ~4391 (partial) | after M3 + 2-rel shadow | high | Old harness; scoreboard.py (588) + registry (221) is new entrypoint. **scoreboard IS the jitpipe parity oracle** — do NOT retire any reference-measurement path until M3 + NIST/golden oracle preserved |
| **MOCK / DEDUP (safe now, cross-cutting)** | | | | | | |
| 44 | Duplicated `mock_db` fixture (`MagicMock(spec=AtomicDatabase)`, coeffs `[3.2188,...]`, U=25) in 13 test files | consolidate to conftest | ~260 → ~20 | safe now | low | Verbatim dup (docstrings literally say "Mirror the … mock fixture"). Promote one parametrizable `mock_db` to `tests/inversion/conftest.py` + `tests/conftest.py`. Touches no reference physics |

### Out-of-scope re-classifications (do NOT treat as reference-deprecation)

These were flagged by auditors as belonging to a *different* dimension — listed
so they are not mistakenly swept into the M3 reference-removal:

- **Benchmark-harness identifiers** (NOT on run_pipeline; its identifier is
  `line_detection.py`): standalone `identify/{alias (4926), comb (1373),
  correlation (1187), spectral_nnls (725), hybrid (418), hybrid_consensus (451),
  model_selection (508)}` — separate dead-code/benchmark dimension.
- **Alternate solvers** (test + `__init__` only, not CLI/scoreboard-routed):
  `solve/{closed_form (757), coarse_to_fine (659), joint_optimizer (1040),
  spectral_refiner (398)}` — niche subsumed by jitpipe `solve.py` post-M3 but
  they are not the frozen oracle.
- **Permanent host (never jit-ported):** `solve/bayesian/*` (sampler loop, §5.1.3),
  `runtime/` (4154), parts of `preprocess/{deconvolution,outliers,response_correction}`.
- **Not inversion at all:** `common/pca.py` (949, manifold infra).
- **Two snapshot types NOT yet retireable:** both `AtomicSnapshot`
  (`core/jax_runtime.py:430`) and `_AtomicSnapshot` (`iterative.py:382`) are
  actively bridged by jitpipe `PipelineSnapshot` — last thing to go, after M3 +
  a separate forward-kernel migration. **Not actionable.**
- **Closure modes** (standard/matrix/oxide/ilr/pwlr/dirichlet_residual): all are
  intentional user-selectable surface, ALL lifted into jitpipe (§4 row 9; ilr is
  provably ≡standard, kept for the parity identity). **Not duplication.**

---

## 3a. SAFE NOW (jitpipe-independent)

Everything here can land before jitpipe exists. None touches the frozen
`run_pipeline` oracle. Group by sub-task; each is independently committable.

### A. Branch & worktree pruning (~2.4 GB, 0 source LOC) — table #1–5

1. **Salvage first** (never actioned from prior audit): diff & commit the
   uncommitted test edits in `wt-pf/g2-cli-drift`
   (`tests/cli/test_analyze_invert_no_drift.py`) and `wt-pf/g4-benchpause`
   (`tests/scripts/test_bench_pause.py`); salvage the 4 tracked mods in
   `.worktrees/w7-integration`.
2. **Delete merged local branches** (#1) — diff-check the 2 `cx/wave3-*`
   exceptions; `git branch -D` the rest.
3. **Delete remote mirrors** (#2) — `git push origin --delete … && git fetch --prune`.
   Keep `main/dev/jitpipe/gh-pages/adr/0002` + 3 active hotfix branches.
4. **Remove completed-wave worktrees** (#3, #4) — `git worktree remove`.
5. **Unlock + remove 4 stale `agent-a*` worktrees** (#5) — verify pid dead.
6. **DO NOT** prune the 26 `wf_*` jitpipe worktrees, `integ-j8`, `stage`,
   `/tmp/cflibs-residual` (#23 — gated to M3).

### B. Superseded build/campaign script archival (~6.4 KLOC) — table #6–8

- `scripts/hpc/` → `scripts/archive/hpc-campaign/`; drop the `pyproject.toml:212`
  sklearn TID251 exemption on move.
- `scripts/campaign1/` → `scripts/archive/campaign1/` (README preserved).
- Per-file diff the 5 orphaned root drivers; archive the stale ones. KEEP
  `run_unified_benchmark.py` + `parameter_sweep.py`. Treat `sweep_alias_fixes.py`
  as benchmark-gated (archive, not delete).

### C. Dead-module + filesystem sweeps — table #9–11, #16–17

- Delete `physics/cdsb.py` + `tests/test_cdsb.py` + 3 `__init__` export entries (#9).
- Remove `cflibs/benchmarks/` stale dir; scrub merge-comments; close bead 4ig9 (#11).
- Archive/remove `tests/validation/validation/` after confirming non-fixture (#10).
- Archive `docs/audit/2026-06-09-overhaul/*` + `2026-06-10-goalfirst/*` to
  `docs/archive/audit/` — *after* dispositioning the `audit/04-pipeline-defaults`
  branch (#17).

### D. Import-shim / facade slimming (reference-independent) — table #12–14, #18, #20, #21(cdsb)

- Close bead 38xr (verify-and-close; work already done) (#12).
- Drop private underscore symbols from `bayesian/__init__.py` `__all__` (#13).
- Resolve the `ConvergenceStatus`/`JointConvergenceStatus` collision doc (#14).
- Resolve the inverted Bayesian sampler aliases — pick one direction (#18).
- Aggregator slim (#20) needs a **public-API deprecation decision** (it's a
  documented public surface) — not jitpipe-gated, but not a blind delete.

### E. Test dedup — table #44

- Promote the 13-site `mock_db` fixture to conftest.

### F. Doc corrections — table #15

- Scrub the stale CLAUDE.md flat-path claim + `manifold-generator.py` warning.

---

## 3b. GATED ON jitpipe PROMOTION

Nothing here lands until jitpipe is built and promoted. The reference is the
contract until then.

### Stage A — ORACLE-PRESERVED (now → M3): **nothing removed**

The reference is the frozen contract. Every jit/reference divergence is a
presumed jit bug recorded in a divergence ledger. Items #23 (keep jitpipe
branches/worktrees) and #32 (keep jax-port consultation docs) are *do-not-touch*
flags for this window.

### Stage B — BUGFIX-ONLY (after M3 promotion, the ≥2-release shadow)

Reference kept as `--pipeline=reference`, bug-fix-only, each fix paired with a
parity-test update. Removable here (the never-ran legacy lax path goes FIRST,
once jitpipe internalizes its own parity anchor):

- #24 `_solve_lax` shell, #25 `_run_lax_while_loop`, #26 pure_callback closure
  trio, #27 `IterativeCFLIBSSolverJax`, #22 `iterative_jax` benchmark workflow.
- #28 pressure-balance fallback — **only with a replacement n_e policy**.
- Tests: #29, #30, #40, #41, #42.
- #19 forward_models extra registry entries (low-value).
- #31 shared kernels — may RELOCATE (move, never delete).

### Stage C — VALIDATION-ORACLE (after ≥2-release shadow)

Reference demoted from parity contract to validation oracle. Remove lower-risk
per-stage kernels:

- #33 (preprocessing, line_selection, boltzmann, closure-non-shared, stark,
  selfabs-observable ~4820), #34 (SelfAbsorptionCorrector classes).
- #39 retarget remaining reference-stage tests to oracle-only, trim true dups.

### Stage D — REMOVAL (needs independent oracle = NIST parity + golden corpus + ROADMAP)

High-risk front-end + production solver + orchestrator, removed last:

- #35 `line_detection.py` (+ Rust comb/kdet D7), #36 `wavelength_calibration.py`,
  #37 `iterative.py` solve body (+ `matrix_effects.py`), #43 `unified.py`
  reference-measurement paths.
- #38 `run_pipeline` orchestrator — **dead last**, preserving `ANALYSIS_PRESETS`.

---

## 4. Staged Execution Roadmap (mapped to jitpipe milestones)

ADR-0004 milestone map: **M1 = J8** (CPU end-to-end parity), **M2 = J9**
(throughput on one V100S), **M3 = J12** (full-board superiority + promotion).

| Phase | Trigger | Do | Net LOC / disk |
|---|---|---|---|
| **P0 — Now (pre-J1)** | none | §3a A–F: branch/worktree massacre, script archival, dead-module sweeps, shim slimming, mock dedup, doc fixes. Close beads 38xr/4ig9 | ~2.4 GB disk + ~6.4 KLOC archived + ~0.5 KLOC deleted |
| **P0.5 — Public-API decision** | maintainer decision | #18 sampler aliases, #20 aggregator deprecation plan, #14 collision doc | ~430 LOC reviewable, retarget-only |
| **P1 — At M2 (J9)** | jitpipe runs end-to-end + benchmark adapter exists | #22 retire `iterative_jax` benchmark workflow once the jitpipe scoreboard adapter (ADR §8.1 J9/J10) replaces it. Prep: move jitpipe's parity anchor off `_run_lax_while_loop` (unblocks #25) | ~130 LOC |
| **P2 — At/after M3 (J12), Stage-B (≥2-rel shadow)** | promotion ADR signed; `--pipeline=reference` shadow live | #24–#28 lax shells + Jax subclass + pressure-balance (with policy); #19; #29/#30/#40/#41/#42 tests; #31 relocate shared kernels | ~1.0 KLOC code + ~2.3 KLOC tests |
| **P3 — Stage-C (after ≥2-rel shadow)** | reference demoted to validation oracle | #33 per-stage kernels (~4820); #34 SA classes; #39 oracle-only test retag/trim | ~5.3 KLOC + ~0.6 KLOC tests |
| **P4 — Stage-D (Removal)** | independent oracle (NIST parity + golden corpus) + ROADMAP entry | #35 line_detection (+Rust D7); #36 wavelength_calibration; #37 solve body (+matrix_effects); #43 unified.py paths; **then #38 run_pipeline dead last** | ~9.1 KLOC |

**Pick-up rule:** P0 / P0.5 are independent and can land in any order now. P2–P4
are strictly ordered by ADR D6 stages — never skip a stage, never delete before
its gate, and preserve an oracle at every step.

---

## 5. Risks & How to Verify Safe (high-risk items)

General rule for every code deletion: **prove zero callers before removing.** Use
the Serena ladder, not bare grep:
`find_referencing_symbols(name_path="X", relative_path="cflibs/")`, then confirm
across `tests/` + `scripts/`. For non-Python (Rust/YAML/TOML) fall back to `rg`.

| Item | Specific risk | Verification gate before action |
|---|---|---|
| #31 shared kernels | Deleting any breaks jitpipe (D3 twins) | Confirm jitpipe imports (`fit.py:42`, `solve.py:52-58`, `snapshot.py:422`, `parity.py:51`, `host.py:311`). NEVER delete — only relocate-with-retarget after M3; run jitpipe parity suite after any move |
| #25 `_run_lax_while_loop` | jitpipe + `test_iterative_lax.py` use it as parity anchor | Move/duplicate the anchor into jitpipe FIRST; re-run `test_parity_j7.py` green; only then remove |
| #26 pure_callback trio | `test_parity_j7.py` imports `_make_closure_callback` | Retarget that import to jitpipe's native closure before deleting |
| #28 pressure-balance fallback | NOT dead — degraded-spectrum n_e fallback; removing changes behavior | Choose + implement a replacement n_e degradation policy first; do not leave a bare hole. Verify via `test_iterative_ne_stark.py` |
| #34 SelfAbsorptionCorrector | Shares module with LIVE `correct_via_doublet_ratio`/`find_doublet_pairs` (used by observable path + jitpipe `selfabs.py`) | Surgical class-level removal only; keep the live helpers; confirm `ObservableSelfAbsorptionCorrector` is the proven oracle (§4 row 7) |
| #35 line_detection | Sole production identifier; highest-concentration hazard (R8) | Survive as parity oracle through Stage-B/C; remove only with NIST+golden oracle preserved; Rust comb/kdet die WITH it (verify no other Rust consumer) |
| #36 wavelength_calibration | `random_seed=42` makes the reference itself seed-dependent | Run reference self-variance study + ye6t coverage fixtures before freezing/removing bands |
| #37 / #38 solve body + run_pipeline | The governing-rule oracles | Removed dead-last, after independent oracle + ROADMAP entry; preserve `ANALYSIS_PRESETS` table (#31) |
| #9 cdsb.py | Audit said "safe now" but `tests/test_cdsb.py` still imports it | Delete the test in the same change (verified: no prod/script imports of the *module* — scripts use their own local cdsb funcs) |
| #43 unified.py | scoreboard IS the jitpipe parity oracle | Do NOT retire any reference-measurement path until M3 + NIST/golden oracle preserved |
| #20 aggregator | Documented public API + pinned by `test_import_safety.py` | Public-API deprecation decision + release boundary; retarget internal callers; never delete outright |
| #1–#5 branches/worktrees | Losing genuinely-unmerged work | `git cherry`/`merge-tree` per branch; salvage uncommitted edits in g2/g4/w7-integration; KEEP all jitpipe/verify/integ refs |

### Corrections to the audit inputs (verified on dev @ 2214c20)

- **`physics/cdsb.py` is module-import-only.** Confirmed `self_absorption.py`
  references CDSB only in **docstrings** (no import); `run_experiments.py` /
  `run_accuracy_ablation.py` define their **own local** `_apply_cdsb_pipeline` /
  `apply_cdsb_correction` — they do NOT import `cflibs.inversion.physics.cdsb`.
  So #9 is genuinely safe-now *provided* `tests/test_cdsb.py` is removed with it.
- **Flat-path shims are gone** (`import cflibs.inversion.solver` → ModuleNotFoundError);
  bead 38xr's acceptance grep returns zero. The injected CLAUDE.md "shims exist"
  claim is stale (#15).
- **Root `manifold-generator.py` does not exist**; CLAUDE.md warning is stale (#15).
- **`cflibs/benchmarks/` is untracked-only** (`git ls-files` empty); a 0-LOC
  filesystem sweep (#11).
- **Milestones:** M1=J8, M2=J9, M3=J12 (ADR-0004 §8.1/§8.3) — used for §4.
