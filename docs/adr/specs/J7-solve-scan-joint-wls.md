# J7 Implementation Spec — Solve: Fixed-K Scan Initializer + Joint WLS Production Estimator

**Bead:** J7 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §4 row 10, §3 C4, §6.1 · **Track:** B · **Depends:** J0, J4, J6 · **Estimated effort:** 10–15 pd (scan extensions ≈1 wk; joint WLS ≈1–2 wk; overlaps)

## 1. Goals

Implement `jitpipe/solve.py` per ADR-0004 §3 C4: **extend the lax path's component kernels verbatim; re-derive the production estimator.** Two layers:

1. **Fixed-K scan initializer** — the existing lax loop body, completed (SB-graph, Stark n_e, native ILR, in-body keystone gate, unified IPD, corona weighting) and converted from dynamic `lax.while_loop` to a fixed-iteration `lax.scan` (max_iterations=20, `iterative.py:882`) with converged-state freezing (`where(converged, prev, new)`) — bit-compatible with the while loop at the fixed point, reverse-differentiable, vmappable.
2. **Joint WLS Gauss–Newton/LM estimator** — the production solve, whose **GN step 0 is algebraically the SB-graph closed form** (J4's Schur identity): the exact parity anchor and warm start.

## 2. The honest gap list being closed (file:line; `solve/iterative.py`)

(i) routing: SB-graph or Stark diagnostics force `_solve_python` (`:1592-1613`) — production never reaches the lax path; (ii) ILR — the constructor default (`:1046-1050`) — and pwlr/dirichlet route through `jax.pure_callback` (`:614-655`): serial under vmap, non-differentiable; (iii) lax n_e is pressure-balance only, `ne_from_stark` hardwired False (`:2358-2364`); (iv) keystone gate + quality metrics host-side post-loop (`:2344-2365`, `:1997-2070`); (v) two-region corona weighting deferred in the lax body (`:810-818`) — lax ≠ Python when `two_region=True` with corona-sensitive elements ({Si,Fe,Ca,Al,Mg} weighting at `:1121-1156`); (vi) IPD formula drift: lax inlines Debye–Hückel (`:778-781`) while Python calls `ionization_potential_lowering` (`:1224-1226`, `plasma/saha_boltzmann.py:542+`) — unify into one jittable function (matters only when `apply_ipd=True`, default False `:886`); (vii) dynamic-trip `while_loop` is not reverse-differentiable; the grad test is try/except smoke (`tests/inversion/test_iterative_lax.py:297-363`); T4-1 `custom_root` never landed (`specs/T1-3-lax-while-iterative.md:155,181,190`); (viii) `_LaxFallback` raise/catch (`:213-220,2259-2270`) → per-spectrum validity flags.

**Reuse verbatim (trivially batched):** `_AtomicSnapshot`+`reorder` (`:381-540`, via J0's `PipelineSnapshot` bridge), `_build_padded_arrays_from_obs` (`:2986-3020`), `_eval_partition_jax` (`:658-706`), `_saha_ratio_per_element` (`:709-718`), `_saha_correct_kernel` (`:2675-2696`), `_common_slope_kernel` (`:2698-2773`), `batched_boltzmann_fit` (`boltzmann_jax.py:111-200`), degeneracy/keystone gates in numeric form (`:804-806`, `closure.py:495-535`), quality-metrics assembly seam (host, `:1997-2070`). **Discard:** dynamic while_loop as production estimator; `pure_callback` closures; the `_LaxFallback` exception protocol.

## 3. The joint WLS formulation (ADR-0004 §6.1)

minimize over θ = (ln T, α ∈ R^{E−1}, β), n_e **pinned to the J6 Stark measurement** (audit F2) with a penalty from its MAD scatter when D>1 (pressure balance only as flagged fallback):

Σ_l w_l · [ y_l − ( −E_l/(k_B T) + ln C_{s(l)}(α) − ln U_{s(l)}(T) − ln M_{s(l)}(T,n_e) + β + saha_shift_l(T,n_e) ) ]²

fixed-K GN/LM on ≤(E+2) parameters (≈20×20 normal systems), vmapped over B. The numpy precursor is `ClosedFormILRSolver` (`solve/closed_form.py:1-19,220-302,305-329`) which must freeze U_s(T), M_s(T,n_e) and bolt on 1–2 Saha passes (`:71,105`); autodiff removes exactly that limitation. Closure modes enter as the map α→C (standard/ILR identical; oxide = factor-weighted normalization inside ln C; matrix = one pinned coordinate). Initialization: scan-initializer output (or SB-graph closed form directly). Optional extension behind a flag: per-line COG escape-factor term f(τ_l(θ)) from J5 — audit F10's "fold COG into the regression".

## 4. Tolerance contracts

| Layer | Contract |
|---|---|
| Scan initializer vs `_solve_python` | inherit rtol 1e-5 (T atol 1e-3 K, conc atol 1e-7) across all six closure modes (`test_iterative_lax.py:183-204`) **including SB-graph + oxide + Stark configurations the current lax tests cannot cover**; iteration count within 1; iteration-freeze idempotence test |
| GN step 0 | ≡ SB-graph closed form, rtol 1e-10 (exact algebraic anchor) |
| Joint converged (T, C) vs reference pipeline | on the 10 scoreboard datasets: T rtol 1e-3; per-element C atol 0.5 wt % on converged solves |
| Covariance | vs finite-difference Hessian rtol 1e-4 |
| Gradients | finite and FD-verified — **hard asserts**, no try/except (supersedes the smoke test) |
| Class-level floor | rtol 1e-3 / atol 1e-4 (`test_solver_jax_parity.py:5,208-256`) |

## 5. Acceptance criteria

1. New `tests/jitpipe/test_parity_solve.py` mirroring `test_iterative_lax.py` structure, plus vmap batch=16 and grad-finiteness as hard asserts; all six closure modes parametrized; SB-graph + oxide + Stark production configuration covered for the first time.
2. GN-step-0 anchor test green (rtol 1e-10).
3. Joint solve ≤ iterative round-trip error on the golden set (`cflibs/validation/round_trip.py`); converged-fraction ≥ reference on the 10 datasets; non-converged spectra carry validity flags (never exceptions).
4. Corona-weighting and IPD-unification parity fixtures (`two_region=True` with {Si,Fe,Ca,Al,Mg}; `apply_ipd=True`) — lax-vs-Python equivalence restored where it is currently documented-divergent.
5. ILR/PWLR/Dirichlet native (no `pure_callback` in `jitpipe/`); keystone gate in-body; n_e Stark-primary with flagged pressure-balance fallback.
6. Host wrapper reconstitutes `CFLIBSResult` (`iterative.py:81`) including quality metrics assembled host-side.

## 6. Test plan

`test_parity_solve.py` (scan layer), `test_joint_wls.py` (anchor, covariance-vs-FD, golden round-trip, scoreboard-input golden tests). Convergence-freeze idempotence: run scan at K and K+5, assert identical frozen output. CPU x64; narrow subsets per watchdog rules.

## 7. Risks

MED: convergence-freezing semantics (idempotence test); optimizer robustness across 10 datasets (LM damping schedule fixed-K; fall back to returning the initializer result with a flag when GN fails its own gates); corona/IPD unification changes answers in configs that currently silently diverge between paths — fixtures make this explicit; covariance validation (FD comparisons are noisy — use well-conditioned golden fixtures).

## 8. Dependencies / files

Depends J0 (snapshot), J4 (fit/closure kernels), J6 (Stark coupling). Enables J8, J11. Files: `cflibs/jitpipe/solve.py`, tests. Reference untouched; the existing `_solve_lax` is neither modified nor deleted (it remains the reference-side lax path until Stage-C deprecation).
