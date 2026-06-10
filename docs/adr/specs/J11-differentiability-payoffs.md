# J11 Implementation Spec — Differentiability Payoffs: Implicit-Diff Spike, Gradient Knob-Tuning, HMC

**Bead:** J11 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §6, §8.4 ("start small regardless") · **Track:** C (research) · **Depends:** J0 (spike phase); J7 (full-model phase); J9 batch harness (opportunistic) · **Estimated effort:** 7–10 pd / ~2 wk

## 1. Goals (three deliverables)

1. **`custom_root` implicit-diff spike (early; pairs with J0 as the program's cheap de-risk).** Implement implicit differentiation through the converged (T*, n_e*) fixed point of the scan initializer via `jax.lax.custom_root` or a hand-written `custom_vjp` IFT linear solve — the deferred T4-1 item (`specs/T1-3-lax-while-iterative.md:155,181,190`). Tooling constraint (ADR-0004 §9): jaxopt is unmaintained; optimistix/lineax are equinox-dependent — **banned**; core JAX only. Host scipy L-BFGS-B driving JAX gradients is acceptable for research phases (scipy is not on the banlist). Novelty note: neither jaxrts nor ExoJAX does `custom_root` through a plasma fixed point (ADR-0001 §5.4) — paper-worthy.
2. **Gradient knob-tuning harness (§6.3).** One backward pass over a batch yields d(soft-F1)/d(continuous knobs) for ≈2–3× the cost of a forward board — vs one full board per Optuna point with no gradient. Implement the **train-soft / eval-hard relaxation map** (§6.4): presence rule → explicit-`jnp` sigmoid (no `jax.nn`; or reuse `cflibs.inversion.physics.softmax_closure`); top-K/SNR/prominence gates → soft-top-K or straight-through; quality scalars → sigmoid penalties. **Hard always:** structural masks; reparameterized physical constraints (log-T, log-n_e, ILR/softmax simplex); scoreboard scoring (the board never sees a relaxed metric); convergence flags (gradients masked). Deliver the hybrid recommended in §6.3: TPE over categorical/integer knobs with a gradient inner loop on the ~20 continuous knobs of `AnalysisPipelineConfig` (`pipeline.py:90-152`) per trial — zero recompiles since knobs are traced `PipelineParams` leaves.
3. **HMC over the full model (§6.2).** NumPyro NUTS consuming the jit likelihood (the ExoJAX pattern: forward outside the model graph; sampler loop host-side per ADR-0004 §5.1.3). The J7 joint-WLS residual, reverse-differentiated, is directly the potential over (T, n_e, α). NNLS top-K candidate prefilter stays mandatory (CLAUDE.md). Chebyshev baseline as in `BayesianForwardModel`. fp64 throughout — NUTS on fp32 plasma exponentials produces divergences. Reuse the `solve/bayesian/` priors/samplers split; do not duplicate `BayesianForwardModel`.

## 2. Acceptance criteria

1. **Spike:** gradient of (T*, n_e*) w.r.t. inputs via implicit diff agrees with central finite differences at rtol ≤1e-4 on golden spectra (`cflibs/validation/round_trip.py`); backward memory independent of iteration count (vs the unrolled scan); viability verdict written up (adopt for the initializer / scan-unroll suffices) — this answer gates nothing else, which is the point of doing it early.
2. **Knob-tuning:** end-to-end backward pass through the relaxed objective fits 16 GB at batch ≥32 (remat on the broadening stage per J9); gradients FD-verified on a 3-knob slice; a demonstration run on the optimization split shows the gradient inner loop reducing the soft-F1 objective with the **hard**-scored board confirming a non-negative aggregate move (any hard-board regression is a finding, not a failure — soft-hard gap measured and reported); benchmark-gated before any campaign adoption.
3. **HMC:** vectorized chains across ≥16 spectra of one bucket on one V100S; R-hat < 1.01 on golden synthetic posteriors; posterior means consistent with J7 point estimates within posterior σ; runtime documented.
4. All work obeys physics-only TID251; no `jax.nn`, no jaxopt/optimistix/lineax imports anywhere.

## 3. Test plan

`tests/jitpipe/test_implicit_diff.py` (FD agreement, memory probe); `tests/jitpipe/test_soft_relax.py` (soft→hard limit: τ→0 recovers hard counts; straight-through forward equals hard forward); HMC golden-posterior test marked `slow`+`requires_bayesian`. GPU runs from the parent session per watchdog rules.

## 4. Risks

Implicit-diff conditioning at weakly-converged fixed points (use well-converged golden fixtures; report conditioning); soft-objective/hard-board gap may be large for gate-dominated knobs (measure first — that result itself informs Campaign-1 design); HMC multimodality on confounder-rich candidate sets (prefilter + documented divergence counts).

## 5. Dependencies / files

Spike: J0 only. Full-model: J7 (+J9 opportunistically). Optional input to J12. Files: `cflibs/jitpipe/solve.py` (custom_root wrapper), `cflibs/jitpipe/tuning.py`, HMC glue under `cflibs/jitpipe/` consuming `solve/bayesian/` host drivers, tests. Reference untouched.
