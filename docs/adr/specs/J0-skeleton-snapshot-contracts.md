# J0 Implementation Spec — `cflibs/jitpipe/` Skeleton, `PipelineSnapshot` Unification, Tolerance Contracts

**Bead:** J0 (placeholder; real ID assigned at `bd create`) · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §8.1 row J0 · **Track:** spine (gates everything) · **Depends:** — · **Estimated effort:** 3–5 pd / 2–3 agent-sessions

> J0 is deliberately cheap and zero-contention: per ADR-0004 §8.4 it may start *before* the program start criteria hold, together with the J11 `custom_root` spike, to de-risk the two biggest unknowns (snapshot unification; implicit-diff viability).

## 1. Goals

- Create `cflibs/jitpipe/` with the module-per-stage layout of ADR-0004 §5.1.1 (stage modules may be empty stubs with signatures + docstrings; `snapshot.py`, `params.py`, `host.py`, `parity.py` get real implementations).
- **Unify the two snapshot types** — `AtomicSnapshot` (`cflibs/core/jax_runtime.py:430`, pytree-registered at `:553`, serves the forward kernel) and `_AtomicSnapshot` (`cflibs/inversion/solve/iterative.py:381-540`, serves the lax solver) — into one frozen, pytree-registered `PipelineSnapshot` built host-only from a single SQLite scan.
- Define `PipelineParams` (traced pytree: ALL continuous knobs — detection thresholds, windows, tolerances, score weights) and `StaticConfig` (hashable statics: bucket id, broadening mode, padding constants, max_iters, B — the jit cache key), mirroring the static-vs-traced split documented at `cflibs/radiation/kernels.py:28-38`.
- Land the program's contract documents: shape-bucketing + padding-constants policy (ADR-0004 §5.2 tables), tiered tolerance contracts K/S/D/B (§5.4), precision policy (§5.3).
- Enforce the boundary invariants from day one: import-hygiene test; no-SQLite-in-kernels test scaffolding.
- Fix the compile-cache default trap and apply the ADR-0001 §12 addendum.

## 2. `PipelineSnapshot` contents (measured against `ASD_da/libs_production.db`, 6.1 MB)

| Block | Shape / dtype | Size | Notes |
|---|---|---|---|
| lines | 28,727 rows × ~13 f64-equiv cols: element-idx i16, sp_num i8, wavelength, aki, ei_ev, ek_ev, gi, gk, stark_w, stark_alpha, stark_shift, aki_uncertainty, is_resonance bool, stark_source_class u8, gamma_vdw_log | ≈3.0 MB | stark_source_class measured: konjevic-λ²-scaled 22,951 / interpolated 4,574 / hydrogenic 562 / `stark_b` 244 / null 396 |
| energy_levels | (144, 676) g + E f64 + mask | ≈1.65 MB | 9,448 rows, max 676 levels/species (Fe II); uniform pad, no bucketing needed |
| partition polys | 146×5 f64 | 6 KB | NaN sentinel as in `iterative.py` |
| canonical scalar fallbacks | 175×2 | 3 KB | **eager** at build — the lazy probe at `iterative.py:479-490` becomes one-time cost; restores simple semantics |
| species_physics | 175×2 (ip_ev, atomic_mass) | 3 KB | masses feed Doppler widths in J6 (`stark_ne.py:504-506`) |
| doublet pairs (for J5) | (P, 2) i32 + ρ + r_thin | <0.5 MB | atomic-data-static; pair constants per `physics/self_absorption.py:222-233` |
| oxide stoichiometry | per-candidate (E,) vectors | negl. | from `physics/closure.py:62-73` |

Total ≈5–6 MB device-resident. Builder: `host.py`/`snapshot.py`, one SQLite scan, `.npz` cache keyed by a DB content hash (invalidation = hash mismatch). Per-bucket candidate-set assembly = element masks over the superset snapshot + the `_build_padded_arrays_from_obs` (`iterative.py:2986-3020`) / `reorder` (`:519-540`) gather pattern — the ONLY per-spectrum host↔device seam.

**Bridging requirement:** `PipelineSnapshot` must be constructible *from* and convertible *to* both existing snapshot types during the transition (J7 reuses the lax solve kernels verbatim; `forward.py` feeds `kernels.forward_model` which consumes `AtomicSnapshot` fields). Do not modify either existing type in J0.

## 3. Files to create

- `cflibs/jitpipe/__init__.py` (public API: `run_batch`, `run_one`, `PipelineSnapshot`, `PipelineParams`, `StaticConfig`; clear `ImportError` with install guidance when JAX missing — jitpipe requires JAX by definition, ADR-0004 §5.5)
- `cflibs/jitpipe/snapshot.py`, `params.py`, `host.py`, `parity.py` (real), stage stubs `preprocess.py`, `detect.py`, `calibrate.py`, `identify.py`, `fit.py`, `selfabs.py`, `stark.py`, `solve.py`, `forward.py`, `pipeline.py`
- `docs/jitpipe/contracts.md` (tier table K/S/D/B + per-stage contract registry, seeded from ADR-0004 §4/§5.4) and `docs/jitpipe/shapes.md` (bucket + padding-constant tables, §5.2)
- `tests/jitpipe/test_import_hygiene.py`, `tests/jitpipe/test_snapshot.py`, `tests/jitpipe/test_params_pytree.py`
- Separate small PR items carried by this bead: flip `cflibs/core/platform_config.py:94` cache default from the uid-skewed `/cluster/shared/jax-cache` to a per-user path (`~/.cache/cflibs/jax`) + update `docs/jax-compile-cache.md` (pathology: hung jobs 1909/1914/1915, `.serena/memories/physics_invariants_and_gotchas.md:54-56`); apply the ADR-0001 §12 addendum text (ADR-0004 §9) to `ADR-0001-radis-jaxrts-pattern-survey.md`.

## 4. Acceptance criteria

1. Package imports clean with and without CUDA; under `JAX_PLATFORMS=cpu` all tests pass (conftest already forces CPU+x64, `tests/conftest.py:20,25`).
2. `PipelineSnapshot` round-trips through `jax.tree_util.tree_flatten/unflatten`; jit + vmap smoke over a function consuming it.
3. Snapshot build is **byte-stable**: two builds from the same DB produce identical `.npz` (hash equality); cache hit skips the SQLite scan; cache invalidates on DB content change.
4. Bridge tests: fields consumed by `kernels.forward_model` (`kernels.py:763`) and `_run_lax_while_loop` (`iterative.py:721`) are reachable from `PipelineSnapshot` with identical values vs the existing builders for a 15-element candidate set (measured reference: N_lines=12,008, N_species=30, level pad (30,676) via `AtomicDatabase.snapshot`, `cflibs/atomic/database.py:846`).
5. Import-hygiene test green: no module under `jitpipe/` except `host.py`/`snapshot.py`/`parity.py` imports `sqlite3`, `cflibs.atomic.database`, `cflibs.io`, or `jitpipe.host` (mirror `tests/test_jax_import_hygiene.py`; pattern documented at `kernels.py:72-78`). Nothing outside `jitpipe/` imports `jitpipe` (grep-based check).
6. `PipelineParams` contains every continuous knob of `AnalysisPipelineConfig` (`cflibs/inversion/pipeline.py:90-152`) that the stage specs name; changing any leaf does **not** retrigger compilation (assert via `jax._src` compile-counter or cache-stats probe).
7. Contracts + shapes docs reviewed and merged; eager-fallback semantics documented.
8. ruff TID251 passes on the new package (physics-only constraint inherited automatically).

## 5. Test plan

`tests/jitpipe/test_snapshot.py` (build, cache, byte-stability, bridge parity vs both existing snapshot builders, no-SQLite-after-build query-count guard in the style of `test_iterative_lax.py:415-429`); `test_params_pytree.py` (flatten/unflatten, zero-recompile on leaf change, StaticConfig hashability); `test_import_hygiene.py`. All CPU-x64, well under the 600 s watchdog.

## 6. Risks

- **Two-snapshot impedance mismatch** (different element ordering, level-padding conventions): mitigated by the bridge tests in AC-4 and by not modifying either existing type.
- **Eager canonical fallbacks change prefetch cost**: one-time at snapshot build (process-level), not per-solve; assert build time < 5 s in tests.
- **Scope creep into stage logic**: stage modules are stubs only; any stage implementation belongs to J1–J7.

## 7. Dependencies

None. Enables everything. Files touched outside `jitpipe/`: `cflibs/core/platform_config.py` (cache default), `docs/jax-compile-cache.md`, `docs/adr/ADR-0001-radis-jaxrts-pattern-survey.md` (§12 addendum) — each a separate commit.
