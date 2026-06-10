# J6 Implementation Spec — Stark n_e via Vmapped Fixed-Iteration Levenberg–Marquardt

**Bead:** J6 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §4 row 8, risk R9 · **Track:** B · **Depends:** J0 (soft J1 for peak inputs; parity adapters can feed reference peaks) · **Estimated effort:** 5–10 pd — the largest genuinely new kernel in the program · **Risk:** MED-HIGH

## 1. Goals

Port the Stark-width n_e measurement (`cflibs/inversion/physics/stark_ne.py:394-663` `measure_stark_ne`) and its per-iteration solver coupling (`solve/iterative.py:1250-1280` `_estimate_ne_from_stark_multi`) into `jitpipe/stark.py` as a vmapped fixed-iteration LM fit. This closes lax seam (iii) of ADR-0004 §4 row 10 — the lax body is currently pressure-balance-only with `ne_from_stark` hardwired False (`iterative.py:2358-2364`) — and supplies the pinned n_e for J7's joint estimator (audit F2).

## 2. Current algorithm and jit-breakers (file:line; `physics/stark_ne.py` unless noted)

Candidate gates: literature-grade `stark_w_source ∈ ('stark_b',)` (`:77,486-490`); SNR ≥ 5 (`:492-499`); instrument-width ladder explicit / resolving-power / narrowest-line-percentile floor (`:469-481`; `estimate_instrument_fwhm` `:185-234` with half-max crossings via Python loops `:139-182`); isolation ≥ 1.5× local Gaussian FWHM (`:507-509`); preference bonus for canonical diagnostics + 0.5× resonance down-rank (`:511-519`); score-sorted, capped at `max_lines=5` (`:536-541`); per-candidate same-species multiplet-blend **DB query inside the loop** (`_has_strong_multiplet_neighbour` `:332-371`, calling `atomic_db.get_transitions` at `:356`); local-peak recentre (`:237-254`); **pinned-Gaussian Voigt fit** `_fit_lorentz_fwhm` (`:257-313`): `scipy.optimize.least_squares` (trust-region, bounds, max_nfev=400) over 5 params (area, x₀, Lorentz HWHM γ, linear baseline c₀,c₁) with σ_G pinned; gates rel-RMSE ≤ 0.25 (`:559`), resolvability floor `γ_FWHM ≥ 0.05·gauss` (`:564`), sanity band 1e14–1e20 (`:575,93-94`); closed-form inversion `n_e = REF_NE·(w/w_ref)·(T/T_ref)^α` (`radiation/stark.py:215-236`); cohort log₁₀-MAD trim at 1 decade (`:374-391,611-636`); median + 1.4826·MAD combine (`:638-656`). Solver-level: closed-form per-line re-inversion at current T, median + MAD (`iterative.py:1250-1280`).

Breakers: SQLite inside the candidate loop; `scipy.least_squares` (host, adaptive trust region, data-dependent nfev); data-dependent candidate count + Python sort + break-at-max_lines; half-max crossing loops; rejection-reason string dict (`:460-461`); isolation-dependent irregular windows (`:545`); broad exception catch (`:369`).

## 3. Redesign

- **Snapshot (J0 fields):** per-line `stark_w`, `stark_alpha`, `stark_shift`, `is_resonance`, uint8 `stark_source_class`. Measured: 28,331/28,727 lines have `stark_w`; literature-grade `stark_b` = 244 rows. Atomic masses from `species_physics` feed Doppler widths (`stark_ne.py:504-506`). The multiplet-blend gate is atomic-data-only apart from a weak Boltzmann factor: precompute per line at snapshot build, or evaluate on device against the full 29k-line table at current T — a (C, 29k) masked reduction, trivial. Eliminates the in-loop DB query.
- **Windows:** fixed-size `(C, W=64)` windows **gathered as raw samples** around the recentred peak (recentre = masked argmax — jittable); irregular physical widths become per-candidate masks over the fixed W samples. **Never resample** — window-extraction parity is the main hazard (risk R9).
- **Fit:** vmapped LM, K≈20 fixed iterations, 5×5 damped normal equations (`jnp.linalg.solve`); feasibility by transform (γ = exp(θ), area = exp(θ)) instead of box bounds; Voigt from the jittable profile kernel already used by `radiation/kernels.py` (ADR-0001 forward model). Per-candidate convergence is a mask, not a break.
- **Gates → masks; ranking + max_lines=5 → `top_k`; median/MAD/cohort-trim → masked median (J4 helper) + mask-and-recompute (two fixed passes); rejection reasons → uint8 codes.** Instrument-floor percentile must implement `np.percentile` linear interpolation exactly.
- **Solver coupling:** the width-law inversion + median combine are pure algebra over a fixed `(D,)` diagnostic array — drop directly into J7's solve body for per-iteration re-inversion at current T.

## 4. Tolerance contract

- Per-line n_e rtol 1e-3 vs the scipy fit on identical fixed windows *when both converge* — different optimizers, same smooth 5-param least-squares basin: **contract on the solution, not the optimizer path**; LM-vs-trf divergence on near-degenerate profiles is handled by the same rel-RMSE/resolvability gates rather than by chasing optimizer parity.
- Median n_e rtol 1e-3; candidate-set equality under a documented tiebreak; percentile = exact `np.percentile` interpolation.
- fp64 mandatory; never run the Voigt fit fp32 (`radiation/profiles.py:425-434` is the authority).

## 5. Acceptance criteria

1. Golden tests on real-data windows extracted from the datasets behind `docs/benchmarks/stark-vjbh-empirical-impact.md`; contract green.
2. Candidate-gate decision sets equal to the reference on the golden corpus (incl. the blend gate, SNR, isolation, source-class, instrument-floor ladder).
3. jit + vmap over (B × C) candidates; padding invariance; no-SQLite-in-kernel; grad-finite hard assert (n_e enters J7's penalty term).
4. Convergence-mask behaviour: candidates where scipy converges but LM doesn't (or vice versa) are ≤2 % of the golden cohort and every case is gate-rejected, not silently divergent — each ledgered.
5. Solver-coupling kernel (closed-form re-inversion + median) parity vs `iterative.py:1250-1280` at rtol 1e-12 (pure algebra).

## 6. Test plan

`tests/jitpipe/test_parity_stark.py`: window-extraction equality test (raw-sample gather vs reference slicing — byte-identical samples); LM-vs-scipy property test over synthetic Voigt profiles spanning γ/σ_G ∈ [0.02, 5] incl. the resolvability boundary; gate-decision A/B; cohort-trim/median exactness; golden real-data cohort.

## 7. Risks

MED-HIGH: (a) window-extraction parity — mitigated by raw-sample gather + byte-identity test; (b) LM vs scipy-trf on marginal profiles — bounded by the shared gates and AC-4's ≤2 % ledgered budget; (c) exp-transform reparameterization shifts the optimizer path — accepted, contract is on the solution.

## 8. Dependencies / files

Depends J0 (snapshot Stark metadata, masses); soft J1. Enables J7 (pinned n_e + per-iteration coupling). Files: `cflibs/jitpipe/stark.py`, tests, golden `.npz` windows. Reference untouched.
