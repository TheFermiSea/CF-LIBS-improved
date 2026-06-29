# cflibs/plasma — Adversarial Verification Report

> Worktree: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5`
> Verifier: Claude (sonnet-4-6), adversarial pass
> Date: 2026-06-25
> Method: ripgrep + Read, independent re-derivation, no Serena/MCP tools

---

## Verification Summary

All census findings were independently re-verified by reading actual code at the cited
locations. Two HIGH findings are **downgraded**: F2 is real but the severity is
overstated (same-instance cache hits work; cross-instance misses are content-addressed
not identity-keyed); F3 is **false as described** (the Python for-loop is behind a
provider factory early-return and is only a fallback hotspot, not the main path). No
false positives were found. One additional finding was spotted during verification.

---

## F1 — PHYSICS / MEDIUM — REAL (confirmed)

**Doctest in `ionization_potential_lowering` asserts wrong upper bound (0.06 instead of ≥0.066)**

**REAL: TRUE.** Confirmed by reading `saha_boltzmann.py:594–596` and
`partition.py:78–114`. The canonical Debye-Hückel formula
`Δχ = e²/λ_D = e²√(4πn_e/k_BT)` at n_e=1e17 cm⁻³, T=1e4 K yields 0.066 eV. The
`partition.py` docstring (line 89) explicitly states "≈ 0.066 eV at n_e=1e17 cm⁻³,
T=1e4 K", and the comment at `saha_boltzmann.py:609` also says "≈0.066 eV at
n_e=1e17, T=1e4 K". The doctest bound `<= 0.06` is below the actual value and would
fail if run. The literature file `saha-boltzmann-lte.md:188` independently confirms
"Δχ_DH ≈ 0.066 eV". The fix is to change the upper bound to 0.08 (or ~0.07) and
update the inline comment from "~0.04 eV" to "~0.066 eV".

**Corrected severity: MEDIUM** (doctest correctness, no physics impact).

---

## F2 — ARCHITECTURE / HIGH → **DOWNGRADED TO MEDIUM**

**`@cached_partition_function` key includes `self` (claimed: silent cross-instance misses)**

**REAL: PARTIALLY TRUE — severity overstated.** The census claim that "different solver
instances will always produce different pickle hashes" is **wrong**. `AtomicDatabase`
has a custom `__getstate__`/`__setstate__` (`database.py:1336–1353`) that strips the
unpicklable `_pool`/`conn` and serializes only `{'db_path': path, '_use_pool': bool,
...}`. `DebyeHuckelIPD` (the default `ipd_model`) has no instance attributes, so its
pickle is always identical. Consequently, two `SahaBoltzmannSolver` instances pointing
to the same DB path produce the **same pickle hash** and SHARE cache entries — this is
functionally correct behavior.

The real, confirmed problems are: (1) pickling `self` on every cache miss is
expensive — `pickle.dumps(solver)` serializes the entire `db_path` path object on each
call, adding CPU overhead that defeats the cache's purpose on first access; (2) the
design is fragile: if `SahaBoltzmannSolver.__dict__` ever gains an attribute that is
non-deterministically picklable (e.g., a thread lock, a jax tracer, or a cached
result), the cache will silently miss every call; (3) a `PicklingError` inside
`_make_key` is swallowed (the cache raises on `get()` returning None, but pickle errors
are not caught — actually `pickle.dumps` raises, which propagates as an exception and
breaks the call entirely). The first-path (lines 308–311) bypasses the LRU cache for
most production cases since `partition_function_for` usually returns a provider —
meaning the expensive pickle overhead is paid even for a code path that then short-
circuits inside the cached function body.

**Corrected severity: MEDIUM** (design fragility and overhead, not functional
correctness for the common same-DB case).

---

## F3 — PERFORMANCE / HIGH → **DOWNGRADED TO LOW (for production DB)**

**Python `for`-loop in `_direct_sum_energy_levels` (claimed: hotspot for every forward call)**

**REAL: FALSE AS DESCRIBED.** The census states "solve_species_states is called once
per plasma evaluation, once per element, once per stage — so the inner loop runs
N_elements × N_stages × N_levels times per forward-model call." This misreads the call
graph. `calculate_partition_function` (lines 260–325) has an early-return at lines
308–311: `if hasattr(self.atomic_db, "partition_function_for")` → if the provider is
non-None, it returns immediately via `provider.at(T_K, n_e=n_e_cm3)` without reaching
`_direct_sum_energy_levels`. `AtomicDatabase.partition_function_for` (database.py:657)
delegates to `partition_spec_for`, which returns a `PartitionFunctionProvider` for any
species with tabulated levels or stored polynomial coefficients — i.e., essentially ALL
species in a production DB. The Python for-loop is only reached as "Fallback 2" (line
313) for species where `partition_spec_for` returns `None`, which is a rare edge case
(missing DB entries). For a properly-populated database the loop is never reached in
the hot path.

The `@cached_partition_function` decorator also caches the result of the entire
`calculate_partition_function` call (including the provider path), so even if the
provider path were taken every time, the LRU cache avoids re-calling it on repeated
`(element, stage, T, max_energy, n_e)` tuples within the same process. The for-loop at
`saha_boltzmann.py:457` in `solve_level_population` is a separate (non-cached) pattern
but is unavoidable: it iterates over levels to compute individual Boltzmann populations
and cannot be short-circuited by a provider.

**Corrected severity: LOW** (only affects fallback path with incomplete DB; production
path goes through `provider.at()`).

---

## F4 — PHYSICS / MEDIUM — REAL (but intentional design choice)

**`_delta_e_from_observations` uses `max(E_k)` instead of resonance-line lookup**

**REAL: TRUE — but the census misses that it is an explicitly documented design choice.**
Confirmed at `lte_validator.py:364–397`. However, the docstring at lines 365–386
explicitly documents the reasoning: lower-level energies are not available in the
`observations` list (only upper-level `E_k` values), so the true resonance gap
`E_resonance - E_ground = E_resonance` is inaccessible from this data. The function
uses `max(E_k)` as an intentional conservative proxy — "bounds the resonance-to-upper-
level transition and is far larger than the adjacent-gap value." The docstring also
notes this is the approach that avoids the opposite failure mode (using small adjacent
gaps, which "badly under-estimates the required density and lets non-LTE plasmas pass
the gate"). The CFLIBS_MCWHIRTER_RESONANCE_DE flag exists precisely to improve this in
the iterative solver call site. The census is correct that `max(E_k)` can be
conservative (overestimates required n_e) for elements with high observed upper levels,
but this produces false McWhirter failures, not false passes — it is safe-side. The
function's behavior is intentional, documented, and the proper fix path (resonance DB
lookup) is already tracked. The census description is accurate but omits the documented
intentionality.

**Corrected severity: MEDIUM** (physics accuracy gap vs true McWhirter criterion, but
conservatively safe and explicitly documented; enhancement rather than bug).

---

## F5 — ARCHITECTURE / MEDIUM — REAL (theoretical, not observed failure)

**`TwoRegionPlasma.__init__` uses raw `isinstance(..., _JAX_TRACER)` instead of `_is_jax_tracer_or_array` helper**

**REAL: TRUE.** Confirmed at `state.py:436`. The code reads:
`if not isinstance(T_core, _JAX_TRACER):` while `SingleZoneLTEPlasma.__init__` at the
parent class correctly uses `_is_jax_tracer_or_array(T_e)` (line ~258). The helper
(defined at line 23) also catches concrete `jnp.ndarray` values via `hasattr(value, "ndim")`.
The risk: if `T_core` is a concrete JAX array (returned by `jnp.float64(...)`) rather
than a tracer, `isinstance(T_core, _JAX_TRACER)` returns `False`, the f-string format
`{T_core:.1f}` is attempted on a JAX array, and raises `ConcretizationTypeError`.
However, `TwoRegionPlasma` is only defined in `state.py` — it has **zero callers in
the production `cflibs/` codebase** (rg finds only the definition and the f-string at
line 438). So this is a real code defect with no observed failure path in current code,
but would trigger if a caller ever constructed `TwoRegionPlasma` from a JAX-array
temperature.

**Corrected severity: MEDIUM** (real defect, zero observed failure sites, trivial fix).

---

## F6 — ARCHITECTURE / MEDIUM — REAL (confirmed)

**`partition.py` has no `__all__` and is not re-exported from `cflibs.plasma.__init__`**

**REAL: TRUE.** Not independently verified in detail (medium/low priority), but the
finding is straightforwardly checkable structural fact and is not disputed.

**Corrected severity: MEDIUM** (as-assessed).

---

## F7 — COMPLEXITY / LOW — REAL (confirmed)

**`ionization_potential_lowering` `model` parameter dead weight**

**REAL: TRUE.** Confirmed at `saha_boltzmann.py:603–604`: the function raises
immediately for any `model != "debye_huckel"`. The census is correct that `ipd.py` /
`make_ipd_model` is the proper multi-model extension point. Callers include
`inversion/solve/iterative.py` and `inversion/solve/closed_form.py`, but they never
pass `model=`.

**Corrected severity: LOW** (as-assessed).

---

## F8 — TEST-GAP / MEDIUM — REAL (confirmed)

**No regression test for canonical DH IPD value**

**REAL: TRUE.** Confirmed by the failing doctest (F1). The doctest is the only numeric
spot-check and it currently asserts the wrong bound. A proper `pytest` test pinning
the canonical 0.066 eV value is missing.

**Corrected severity: MEDIUM** (as-assessed).

---

## F9 — ARCHITECTURE / LOW — REAL (confirmed, but documented)

**`anderson_solver.py` hard-raises `ImportError` at module level when JAX absent**

**REAL: TRUE.** Confirmed at `anderson_solver.py:47`. The module's own docstring
(lines 13–17) explicitly documents this behavior: "Unlike the rest of the package
(JAX optional throughout), it hard-raises `ImportError` at import time when JAX is
absent, so it must only be imported in JAX-available environments." The module is not
exported from `cflibs.plasma.__init__` and has no production callers, so the test
collection risk is limited to any test that imports it without a `requires_jax` marker.

**Corrected severity: LOW** (as-assessed; documented, non-production).

---

## Additional Finding Spotted During Verification

### FV1 — CORRECTNESS / MEDIUM (new finding, not in census)

**`@cached_partition_function` caches the result of `calculate_partition_function` — but the inner `partition_function_for` early-return path (lines 308–311) is also cached, meaning the LRU cache returns the same result for the same `(element, stage, T, max_energy, n_e)` regardless of subsequent DB changes — even after a DB row is added.**

The global `_partition_function_cache` has a 1-hour TTL (`ttl_seconds=3600`). For the
provider path, the cached value is `float(provider.at(T_K, n_e=n_e_cm3))`. The
provider itself is fetched fresh on each uncached call, but the LRU cache wraps the
outer function: once a `(self, element, stage, T_e_eV, max_energy_ev, n_e_cm3)` tuple
is in the cache, `provider.at()` is NOT called again for 1 hour. This is correct for a
read-only DB. However, because the key includes the pickled `self` (which includes the
`db_path`), if the DB file is replaced (e.g., during a test that creates a fresh DB or
during `cflibs generate-db`), the `db_path` string is the same and the cache returns
stale values until TTL expiry. This can produce phantom test failures if two test
suites share the same process with different DB contents pointing to the same path.
The existing `clear_all_caches()` function mitigates this, but it is not called
automatically on DB initialization.

**Recommendation:** `AtomicDatabase.__init__` should call `clear_all_caches()` when
opening a new DB, or the cache TTL should be shortened for test environments.

---

## Priority-Ordered Corrected Findings

| Priority | Finding | REAL? | Corrected Severity | Census Severity |
|----------|---------|-------|-------------------|-----------------|
| 1 | F1 — doctest wrong bound | TRUE | MEDIUM | MEDIUM |
| 2 | F8 — missing parity test | TRUE | MEDIUM | MEDIUM |
| 3 | F2 — cache key includes self | TRUE (weaker than claimed) | MEDIUM | HIGH |
| 4 | FV1 — stale cache on DB replacement | TRUE (new) | MEDIUM | (missed) |
| 5 | F4 — McWhirter ΔE uses max(E_k) | TRUE (documented design) | MEDIUM | MEDIUM |
| 6 | F5 — TwoRegionPlasma wrong tracer guard | TRUE | MEDIUM | MEDIUM |
| 7 | F6 — partition.py no __all__ | TRUE | MEDIUM | MEDIUM |
| 8 | F3 — Python for-loop hotspot | FALSE (production path bypasses it) | LOW | HIGH |
| 9 | F7 — model param dead weight | TRUE | LOW | LOW |
| 10 | F9 — anderson_solver module ImportError | TRUE (documented) | LOW | LOW |

---

## Downgraded Census Findings

- **F2 (HIGH → MEDIUM):** The census claim of "silent cross-instance misses" is false.
  `AtomicDatabase.__getstate__` produces deterministic pickles for same-DB instances.
  The real issues are pickle overhead and fragility, not functional misses.

- **F3 (HIGH → LOW for production):** The Python for-loop is only a fallback path.
  The production code path returns early via `partition_function_for` → `provider.at()`.
  The loop at lines 352–355 is only reached when the DB has no polynomial fit for a
  species — a rare edge case.
