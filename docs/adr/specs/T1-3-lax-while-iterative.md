# T1-3 Implementation Spec — `lax.while_loop` in `IterativeCFLIBSSolver`

**Bead:** `CF-LIBS-improved-14p6` · **ADR:** [ADR-0001](../ADR-0001-radis-jaxrts-pattern-survey.md) §5.4, §8.1 row T1-3 · **Wave:** 1 (parallel-safe with T1-1; T1-1's `cflibs/inversion/solve/` carve-out means no overlap) · **Soft dep:** T1-1 (shared decorator) · **Files:** `cflibs/inversion/solve/iterative.py` (≈1326 LOC, target method `solve()` at L417) · **Estimated effort:** 1 day

## 1. Goals

- Replace the Python `for _ in range(max_iter)` outer loop at `iterative.py:470` with `jax.lax.while_loop` so the iterate body is jit-traceable, `vmap`-able across batches of observations, and (eventually) `grad`-able through `(T*, n_e*)`.
- Preserve `IterativeCFLIBSSolver.solve(observations, closure_mode, **closure_kwargs) -> CFLIBSResult` signature bit-for-bit (host side; numerical parity rtol=1e-5).
- Pull **every** effectful operation (SQLite, dict mutation, string dispatch on `closure_mode`) outside the loop body. Body becomes a pure function of `(T, n_e, T_prev, n_e_prev, converged, i)` plus static (mode-specialized) and dynamic (pre-fetched atomic arrays) closures.
- Single source of truth for iterate-body math shared with the NumPy fallback (no algebra duplicated). Host/kernel pattern from T1-1 applied at method granularity.
- Feature-flag via `CFLIBS_USE_LAX_WHILE_LOOP=1` until parity proven; default off.

## 2. Current loop, annotated (`iterative.py:470-612`)

| # | Step | File:line | Status |
|---|------|-----------|--------|
| 1 | Stash `T_prev`, `ne_prev` for damping/convergence | L471-472 | trace-clean (scalar copy) |
| 2 | Per-element `U_I`, `U_II` via `_evaluate_partition_function` | L482-484, L164-187 | **needs lifting** — SQLite via `get_levels_for_species` |
| 3 | `effective_ips` via `_compute_effective_ips` (optional IPD) | L486, L238-251 | **needs lifting** — dict ops + branch; IPD itself pure |
| 4 | `_apply_saha_correction(obs_by_element, T, n_e, ips)` | L489, L253-318 | **needs lifting** — already replicated as `_saha_correct_kernel` (L833) on padded `(E, N_max)` arrays |
| 5 | `_fit_common_boltzmann_plane(corrected_obs_map)` | L492, L320-415 | **needs lifting** — already replicated as `_common_slope_kernel` (L856) |
| 6 | T update `T_new = -1/(slope·KB_EV)` + 50/50 damp | L500-507 | trace-clean (`jnp.where`) |
| 7 | Optional corona `T_corona = 0.8·T` | L509-512 | trace-clean (static gate on `two_region`) |
| 8 | `_compute_abundance_multipliers` from intercepts | L517-525, L203-236 | **needs lifting** — dict→array form |
| 9 | Closure dispatch (six modes) | L527-569 | **must stay outside as static dispatch** — see §5 |
| 10 | `n_e` pressure/charge balance + 50/50 damp | L571-602 | **needs lifting** — `for el, C_s in concentrations.items()` → array reduction |
| 11 | History append + convergence check | L604-612 | trace-clean (carry `converged` bool scalar) |

Post-loop (L614-643): `LTEValidator.validate` + `CFLIBSResult` assembly stay on host.

## 3. State tuple design

```python
class LoopState(NamedTuple):
    T_K: jax.Array           # float scalar
    n_e_cm3: jax.Array       # float scalar
    T_prev: jax.Array        # float scalar (convergence check)
    n_e_prev: jax.Array      # float scalar
    converged: jax.Array     # bool scalar
    i: jax.Array             # int32 scalar (iteration counter)
    # Per-element scratch (carry for §10 + LTE-metric output):
    U_I: jax.Array           # (E,)
    U_II: jax.Array          # (E,)
    intercepts: jax.Array    # (E,)
    concentrations: jax.Array  # (E,)
```

Registered as a NamedTuple (already a JAX pytree). Static carry — `closure_mode_id: int`, `apply_ipd: bool`, `two_region: bool`, `max_iter: int`, `t_tol_k: float`, `ne_tol_frac: float` — baked into closure via `functools.partial` on the jit'd entry point. Each `(closure_mode, n_elements, apply_ipd)` triple compiles once per solver instance.

`CFLIBSResult` does NOT need to be the carry — assembled on host from final state + static `elements` list. Pytree registration of `CFLIBSResult` deferred to T4-1.

## 4. `cond_fun` and `body_fun` sketches

```python
def _make_solve_lax(
    closure_fn,          # static: pre-resolved closure (§5)
    apply_ipd: bool,     # static
    two_region: bool,    # static
    max_iter: int,
    t_tol_k: float,
    ne_tol_frac: float,
    eval_U,              # callable: (T_K,) -> (U_I, U_II) via pre-fetched data (§6)
):
    @jit
    def _cond_fun(state: LoopState) -> jax.Array:
        return jnp.logical_and(jnp.logical_not(state.converged), state.i < max_iter)

    @jit
    def _body_fun(state: LoopState) -> LoopState:
        T_prev, ne_prev = state.T_K, state.n_e_cm3
        U_I, U_II = eval_U(T_prev)                              # (E,), (E,)
        delta_chi = ipd_kernel(state.n_e_cm3, T_prev)           # 0 if apply_ipd=False
        ip_eff = jnp.maximum(ip0 - delta_chi, 0.0)              # (E,)
        x_c, y_c = _saha_correct_kernel(x_raw, y_raw, stage, ip_eff, ...)
        fit = _common_slope_kernel(x_c, y_c, w, mask)
        slope, intercepts = fit["slope"], fit["intercepts"]
        T_new = jnp.where(slope >= 0.0, 50_000.0, -1.0 / (slope * KB_EV))
        T_K = 0.5 * T_prev + 0.5 * T_new
        T_for_saha = jnp.where(two_region, 0.3 * T_K + 0.7 * 0.8 * T_K, T_K)
        S = saha_ratio_kernel(T_for_saha, state.n_e_cm3, U_I, U_II, ip_eff)
        mult = 1.0 + jnp.maximum(S, 0.0)
        concentrations = closure_fn(intercepts, U_I, mult)      # (E,)
        eps_s = S / (1.0 + S)
        avg_Z = jnp.sum(concentrations * eps_s)
        n_tot = pressure_pa / (KB * T_K * (1.0 + avg_Z))
        ne_new = avg_Z * (n_tot * 1e-6)
        n_e = 0.5 * ne_prev + 0.5 * ne_new
        converged = jnp.logical_and(
            jnp.abs(T_K - T_prev) < t_tol_k,
            jnp.abs(n_e - ne_prev) / jnp.maximum(ne_prev, 1e-30) < ne_tol_frac,
        )
        return LoopState(T_K, n_e, T_prev, ne_prev, converged, state.i + 1,
                         U_I, U_II, intercepts, concentrations)

    @jit
    def _run(init: LoopState) -> LoopState:
        return jax.lax.while_loop(_cond_fun, _body_fun, init)
    return _run
```

Notes: `lax.while_loop` returns final state only; iteration history discarded in JAX path (NumPy path keeps it). No print/log/`.item()` inside body.

## 5. Closure-mode dispatch — pulled OUT (Option A: resolve at construction)

Six modes (`STANDARD`, `MATRIX`, `OXIDE`, `ILR`, `PWLR`, `DIRICHLET_RESIDUAL`) currently dispatch via `if/elif` on a Python string at L527-569. Strings are not jit-traceable.

**Option A (recommended).** `IterativeCFLIBSSolver._resolve_closure(closure_mode, closure_kwargs, elements)` returns a closed-over jit'd `closure_fn(intercepts, U_I, mult) -> jnp.ndarray` for the chosen mode. Static `(closure_mode, n_elements, matrix_element_idx, oxide_stoichiometry, ...)` seeds one jit compilation per unique combo per solver lifetime, cached on instance.

- `STANDARD`/`MATRIX`/`OXIDE`: pure linear algebra `q_s = exp(intercept_s) · U_I_s · mult_s` → normalize. ~20 lines each.
- `ILR`/`PWLR`/`DIRICHLET_RESIDUAL`: Helmert-basis transforms (`closure.py:45-200`) already pure NumPy → straight port. Helmert matrix `(D, D-1)` baked into closure.

Pros: zero dispatch overhead in loop, no `static_argnums` on `lax.while_loop`, one compile per `(mode, n_elements)`.

Cons: re-resolve if `closure_mode` changes between `solve()` calls. Trivial fix: cache on `Dict[(mode, n), Callable]` instance field.

**Option B** (`lax.switch` inside body) compiles every branch every time. Use only if call sites change mode per iteration — none do. **Decision: Option A.**

## 6. Per-iteration atomic-DB work — pulled OUT

| Call | Site | Pre-fetch strategy |
|---|---|---|
| `_evaluate_partition_function(el, 1, T_K)` (L483) | inside loop, per element | **Pre-fetch `(g_arr, E_arr, ip_ev)` triple** via `get_levels_for_species` once per `(el, stage)` at solver construction. Inside body: closed-form `U(T) = Σ_k g_k exp(-E_k·EV_TO_K/T)` — pure `jnp.einsum`-style reduction. Polynomial fallback (`Σ a_n (log T)^n`) is closed-form too. No interpolation. |
| `_evaluate_partition_function(el, 2, T_K)` (L484) | inside loop | same |
| `get_ionization_potential(el, 1)` (L458) | already pre-loop | already in `ips` dict; shape into `ip0: jnp.ndarray (E,)` |
| `_compute_effective_ips` (L486) | inside loop | `ip0 - delta_chi` pure arithmetic; `apply_ipd` is static-baked |
| `LTEValidator.validate` (L614) | post-loop | stays on host |

Pre-fetch produces an `AtomicSnapshot`-style frozen bundle. Ragged `Nk_el` handled by padding to `Nk_max` with `g=0` (zero contribution) and mask.

This removes ~20·E SQLite-or-cache calls per `solve()`.

## 7. Public API preservation

`IterativeCFLIBSSolver.solve(observations, closure_mode="standard", **closure_kwargs) -> CFLIBSResult` signature unchanged. `CFLIBSResult` fields unchanged including `iterations` (from `final_state.i`), `converged` (`final_state.converged`), `quality_metrics["r_squared_last"]` (re-computed on host from final state).

```python
def solve(self, observations, closure_mode="standard", **closure_kwargs):
    if _lax_while_loop_enabled() and HAS_JAX:
        return self._solve_lax(observations, closure_mode, **closure_kwargs)
    return self._solve_python(observations, closure_mode, **closure_kwargs)
```

`_solve_python` is the existing body of `solve()` extracted verbatim. Both terminate by calling shared `_assemble_result(final_state, observations, closure_mode, ...) -> CFLIBSResult` host helper — single source of truth for post-loop work.

## 8. Fallback path

If `HAS_JAX is False` or `CFLIBS_USE_LAX_WHILE_LOOP=0` (default during rollout), routes to `_solve_python`. Single source of truth enforced by a `_iterate_step(...) -> StepResult` host helper both implementations call for per-iteration math; only loop driver differs.

Shared decorator from T1-1 (`cflibs.core.jax_runtime.jit_if_available`) wraps `_body_fun`/`_cond_fun` so they no-op without JAX.

## 9. Acceptance criteria

- **Parity:** rtol=1e-5 on `(temperature_K, electron_density_cm3, concentrations[el])` across golden-spectrum round-trip in `cflibs/validation/round_trip.py` for all six closure modes.
- **vmap:** `jax.vmap(solver.solve_arrays)(observations_batch)` compiles and runs over batch of 16 perturbed observation sets sharing the `AtomicDatabase`.
- **grad (smoke):** `jax.grad(lambda T0: solver._solve_lax_scalar(obs, T_init=T0).temperature_K)(jnp.asarray(10_000.0))` returns finite scalar. Smoke test only; correct gradients via implicit-diff live in T4-1.
- **No SQLite inside loop:** instrument `AtomicDatabase` with query counter; assert `query_count_after_solve == query_count_before_solve` for warmed-cache scenario.
- **All six closure modes pass:** parametrized `pytest`.
- **Existing tests:** `pytest tests/ -k iterative` + `tests/inversion/test_solver_jax_parity.py` (tightens existing rtol=1e-3 to 1e-5 because lax.while_loop eliminates host-side dict reorderings).

## 10. Test plan

New file: `tests/test_iterative_lax.py`.

- `test_lax_while_loop_parity_vs_python` — parametrize all six closure modes; rtol=1e-5.
- `test_vmap_batched_solve` — 16 perturbed observation arrays; shape `(16,)` for `T_K`/`n_e_cm3`, `(16, E)` for concentrations.
- `test_grad_smoke` — finite, non-NaN.
- `test_convergence_iteration_count` — `|iters_lax - iters_python| ≤ 1`.
- `test_no_sqlite_inside_loop` — mock `sqlite3.Cursor.execute` or use pool stats; assert zero queries during `_solve_lax`.
- `test_feature_flag_default_off` — `CFLIBS_USE_LAX_WHILE_LOOP` unset → routes to `_solve_python` (assert via mock).

Existing: enumerate via `pytest --collect-only tests/ -k iterative`:
- `tests/test_solver.py::test_*`
- `tests/inversion/test_solver_jax_parity.py::test_*` (tighten to rtol=1e-5)
- `tests/test_two_region_fit.py::*` (corona path)
- `tests/test_e2e_pipeline.py::*`

## 11. Risks & rollback

- **Partition-function direct-sum cost inside trace.** Same FLOP count as NumPy path. Fallback to polynomial form if profiling shows bottleneck.
- **50/50 damping × `lax.while_loop` semantics.** `cond_fun` checked before body — once `converged=True`, body won't re-run. No over-iteration risk.
- **`lax.while_loop` not natively `grad`-able** for arbitrary iteration count. T1-3 only requires `grad` to trace (smoke); correct gradients land in T4-1 via `lax.custom_root`.
- **ILR/PWLR host-side branching** (`closure.py:548,559` logger.warning/error). Must be lifted to validation before the loop (raise at `solve()` entry if <2 elements with non-zero abundance) or removed in jit'd port.
- **`IterativeCFLIBSSolverJax` (L976) confusing.** Currently uses Python `for` calling `_saha_and_fit_jax` per iteration. T1-3's `_solve_lax` supersedes it. Keep as deprecated alias for one release with `DeprecationWarning` pointing at env flag.
- **Rollback:** feature flag default off until 2 weeks green CI on validation harness. Flip default to on once parity confirmed across all six modes and three benchmark datasets. If regression: unset env var in CI.

## 12. Dependencies

- **Soft dep on T1-1 (`5oar`):** benefits from shared `jit_if_available`. Can run in parallel with T1-1 because `cflibs/inversion/solve/` is in T1-1's §5 carve-out. T1-3 does its own local host/kernel split of `iterative.py`.
- **NOT dependent on T1-2** — iterative solver consumes pre-extracted `LineObservation`s; never calls the forward model.
- **Enables T4-1** (`lax.custom_root` implicit-diff).
- **Compatible with T2-7** (bisection on `log n_e`) — nested while_loop layers on top.

**Files touched:** `cflibs/inversion/solve/iterative.py` (modified), `tests/test_iterative_lax.py` (new). Estimated 1 day.
