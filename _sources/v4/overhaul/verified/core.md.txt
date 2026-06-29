# cflibs/core — Adversarial Verification Report

Verified: 2026-06-25  
Verifier model: claude-sonnet-4-6  
Methodology: rg + Read on actual source at `.worktrees/v4-m5/cflibs/`; each finding independently re-derived from code.

---

## Verdict Summary

| ID | Title | REAL | Corrected Severity | Notes |
|----|-------|------|--------------------|-------|
| F1 | AtomicSnapshot in jax_runtime.py | TRUE | high | Confirmed domain carrier misplaced |
| F2 | abc.py imports atomic.structures | TRUE | medium (downgraded) | Real layering smell; no actual cycle today |
| F3 | Cache None-sentinel bug | TRUE | high | Confirmed; affects ionization-potential cache |
| F4 | JAX backend ordering hazard | TRUE | medium | Real risk; warning logged but not enforced |
| F5 | pool._created decrement for pool-origin connections | TRUE (partial) | low (downgraded) | Bug path exists; normal path never triggers it |
| F6 | McWhirter constant 1.6e12 duplicated | TRUE (3 sites, not 5) | low | bayesian/atomic.py correctly uses import — not a violation |
| F7 | validate_instrument_config rejects resolving_power config | TRUE | high | Confirmed; CLI hard-errors on valid config |
| F8 | test_constants.py doesn't pin SAHA_CONST_CM3 | TRUE | medium | Confirmed |
| F9 | pool.py has no dedicated tests | TRUE | low | Confirmed; no test_pool.py exists |
| F10 | Cache TTL/sizes are magic numbers | TRUE | low | Confirmed |

---

## Detailed Verdicts

### F1 — REAL / severity: high
**AtomicSnapshot domain carrier lives in jax_runtime.py**

Confirmed. `jax_runtime.py` lines 429–551 define a 20+-field frozen dataclass with fields covering atomic line arrays, partition function polynomial coefficients, ionization potentials, Stark shift arrays, Stage-III Saha extension fields, and per-species direct-sum flags. This class is consumed directly by `cflibs/atomic/database.py` (which builds it via `db.snapshot()`), `cflibs/radiation/kernels.py`, `cflibs/radiation/spectrum_model.py`, and `cflibs/inversion/solve/bayesian/atomic.py`. The pytree registration (`_snapshot_flatten`, `_snapshot_unflatten`, `jax.tree_util.register_pytree_node`) is the only JAX-runtime concern; the dataclass itself is a pure atomic domain carrier. Any atomic data model change forces editing a module named "runtime helpers for JAX backend capability checks." Severity confirmed high.

### F2 — REAL / severity: medium (downgraded from high)
**abc.py imports cflibs.atomic.structures — inverts the layer**

Confirmed at `abc.py:12`: `from cflibs.atomic.structures import Transition, EnergyLevel`. However, `cflibs/atomic/structures.py` has zero imports from `cflibs.core.*` (verified by rg), so there is no actual import cycle today — only the potential for one. The single caller `cflibs/plasma/saha_boltzmann.py` imports both `from cflibs.core.abc import SolverStrategy, AtomicDataSource` and later uses `Transition` from `cflibs.atomic.structures` directly. The layering violation is real but the risk is latent, not manifested. Downgrade to medium: architectural smell that should be fixed but poses no runtime hazard now.

### F3 — REAL / severity: high
**Cache _make_cache_decorator silently fails to cache None results**

Confirmed. `cache.py:149` uses `if cached_value is not None:` as the hit check. `LRUCache.get()` returns `None` on miss (line 64). `get_ionization_potential()` (`database.py:534–559`) returns `None` when a species is absent from `species_physics`. With `@cached_ionization` applied, every call for a missing species: (a) misses the cache, (b) hits the DB, (c) calls `cache.set(key, None)` which stores `None`, (d) on the next call `cache.get()` returns `None`, which the wrapper interprets as a miss, and the DB is hit again. The `None`-as-sentinel design fails silently on any function that can legitimately return `None`. During benchmarks sweeping over elements not in the DB this causes O(N_queries) DB hits instead of O(1). The fix (dedicated sentinel or tuple-wrapping) is straightforward. Severity confirmed high.

### F4 — REAL / severity: medium
**JAX_BACKEND resolved at import time; configure_jax() may run after**

Confirmed. `jax_runtime.py:59` executes `JAX_BACKEND = _resolve_backend()` at module import time. `configure_jax()` in `platform_config.py` sets `os.environ["JAX_PLATFORMS"] = "cpu"` on macOS (line 80), but it does NOT call `_refresh_runtime_state()` afterward. If any module imports `jax_runtime` before `configure_jax()` is called (common since `jax_runtime` is the primary JAX facade used at ~36 import sites), `JAX_BACKEND` will be stale. The warning on line 71 fires when JAX is already in `sys.modules`, which is a different (but related) check. The `_refresh_runtime_state()` function exists and is documented precisely for this purpose, but `configure_jax()` does not invoke it. Severity confirmed medium: in practice, macOS users who call `configure_jax()` get a correct JAX CPU backend but a stale `JAX_BACKEND == "metal"` module variable, which could mislead conditional logic that branches on `JAX_BACKEND`.

### F5 — REAL (partial) / severity: low (downgraded from medium)
**pool._created decremented without is_new guard**

Confirmed as coded. In `pool.py:96–101`, if `put_nowait` raises, `_created` is decremented unconditionally. The `_created` counter is only incremented when a new connection is created (line 82), not when one is taken from the pool. So for a pool-origin connection (taken from the queue at line 76), if `put_nowait` fails, `_created` is decremented below its correct value, allowing more than `max_connections` to be created subsequently.

However, the census overstates the practical risk. The queue has `maxsize=max_connections`. If we took a connection from the pool (pool path), the queue now has one fewer item, so `put_nowait` on return has space. The "Queue Full" path is only reachable in a race where another thread also created connections in between, which is possible but pathological. In the normal single-thread or low-concurrency use case, the `except Exception` branch on line 97 is essentially unreachable. Downgrade to low: the accounting logic is wrong but the trigger condition requires a race condition unlikely in practice.

### F6 — REAL (3 sites, not 5) / severity: low
**McWhirter constant 1.6e12 duplicated**

Confirmed at 3 sites, but the census claim about `bayesian/atomic.py` is inaccurate. That file correctly imports `MCWHIRTER_CONST` from `cflibs.core.constants` (line 38) and uses it in actual computation (lines 542, 550). The `1.6e12` appearance in that file (line 535) is only in the docstring. The three actual code-level re-definitions are:
- `cflibs/benchmark/physical_consistency.py:49`: `_MCWHIRTER_CONST = 1.6e12`
- `cflibs/inversion/physics/line_selection.py:114`: `MCWHIRTER_PREFACTOR_CM3_K = 1.6e12`
- `cflibs/inversion/runtime/temporal.py:455`: inline `1.6e12 * np.sqrt(T_K) * ...`

Finding is real but the count is 3, not 5. The comment in `physical_consistency.py:45` even notes it is "Mirror of MCWHIRTER_CONST in" (core), acknowledging the duplication. Severity confirmed low.

### F7 — REAL / severity: high
**validate_instrument_config rejects resolving_power-mode configs**

Confirmed. `config.py:234` hard-errors with `ValueError("Instrument config missing 'resolution_fwhm_nm'")` when that key is absent, with no fallback to `resolving_power`. Yet `VALID_ANALYSIS_KEYS` at `config.py:140` explicitly includes `"resolving_power"` as a valid analysis key, and `InstrumentModel` (`instrument/model.py:34–105`) has full support for `resolving_power` mode via `from_resolving_power()`. The CLI uses `validate_instrument_config()` at `cli/main.py:129` for the `forward` subcommand, meaning any config file using resolving-power mode will fail at startup with a misleading error, despite the instrument model accepting it perfectly. This is a real functional gap: resolving-power mode is a first-class documented path but the config validator blocks it. Severity confirmed high.

### F8 — REAL / severity: medium
**test_constants.py doesn't pin SAHA_CONST_CM3**

Confirmed. `tests/test_constants.py` (47 lines) only checks constants exist and are positive/reasonable-order. No test derives `SAHA_CONST_CM3` from CODATA values or asserts it against a computed value. The derived value from SI constants is `2 × (2π m_e k_B / h²)^{3/2} × 1e-6 ≈ 6.037e21`; the coded value is `6.042e21` (0.08% difference from literature rounding). A test would catch accidental edits that lose the factor-of-2 electron spin degeneracy (giving ~3.018e21, a 2× Saha ratio error). Also missing: `KB_EV = KB / E_CHARGE` cross-check and `CM_TO_EV` cross-check. Severity confirmed medium.

### F9 — REAL / severity: low
**pool.py has no dedicated tests**

Confirmed. No `tests/core/test_pool.py` or any `test_pool*.py` file exists in the test tree. The pool is only exercised through `requires_db` integration tests that create real DB connections. The F5 accounting bug and thread-safety logic are invisible to the fast gate (`-m "not requires_db"`). Severity confirmed low.

### F10 — REAL / severity: low
**Cache TTL/sizes are magic numbers with no configuration path**

Confirmed. `cache.py:136–138` defines three module-level `LRUCache` instances with hardcoded `max_size` and `ttl_seconds`. The hit/miss counters (`self.hits`, `self.misses`) are plain `int` attributes with no lock — concurrent benchmark workers increment them without synchronization, producing inaccurate (but not dangerous) stats. Severity confirmed low.

---

## Missed Findings

### MF1 — PERFORMANCE / low
**_make_cache_decorator pickles `self` (AtomicDatabase) on every method call**

`LRUCache._make_key(*args, **kwargs)` uses `pickle.dumps((args, sorted(kwargs.items())))`. When a method decorated with `@cached_ionization` or `@cached_transitions` is called, `args[0]` is `self` (the `AtomicDatabase` instance). `AtomicDatabase.__getstate__` strips `_pool` and `conn`, so pickling succeeds, but it serializes the full database object (including in-memory `OrderedDict` caches of prior results) on every cache lookup — before checking whether the key is already in the cache. For a 1000-element manifold sweep, each lookup serializes the database twice (once for key derivation on cache hit, once on miss). A simple fix: use `(id(self), *args[1:])` as the key tuple for bound methods, or expose a `cache_key_prefix` per database instance. This is a performance issue, not a correctness bug.

### MF2 — CORRECTNESS / medium
**_make_cache_decorator stores result unconditionally even when None, but only reads back if not None**

This is technically part of F3 but worth separating: the `cache.set(cache_key, result)` at line 153 always runs, including when `result is None`. This means the cache fills up with `None`-valued entries for missing species that will never be returned as hits. These entries count toward `max_size` and trigger LRU eviction of *valid* non-None entries. In the worst case (many missing elements), the ionization cache fills entirely with phantom `None` entries, evicting real cached values and causing valid queries to miss repeatedly. The fix for F3 (sentinel object or tuple-wrapping) resolves this as well.

### MF3 — CORRECTNESS / low
**configure_jax() does not call _refresh_runtime_state() after setting env vars**

This is implicit in F4 but the census fix description omits the specific call needed: `configure_jax()` should end with `from cflibs.core.jax_runtime import _refresh_runtime_state; _refresh_runtime_state()` (or better, accept a module-level refresh hook). Without this, callers who do `configure_jax()` then check `cflibs.core.jax_runtime.JAX_BACKEND` get a stale value. The census describes the symptom but not the concrete one-line fix that already exists (`_refresh_runtime_state()`).
