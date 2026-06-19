# Phase 3 — SOTA implementation plan (cited, benchmark-gated)

**Date:** 2026-06-19 · **Inputs:** [`2026-06-19-cross-discipline-sota.md`](2026-06-19-cross-discipline-sota.md) (123 asta-verified methods) + an 8-stage codebase gap analysis (65 non-default candidates, 18 top-tier) + direct code verification.

## Governing discipline

1. **Physics-only** — no sklearn/torch/tf/jax.nn in shipped `cflibs/` (ML only in `cflibs/evolution/`).
2. **Benchmark-gate every accuracy-changing edit** — the repo has regressed ≥3× on un-gated algorithm changes (memory `project_benchmark_gate_identifier_changes`). No default flip without a scoreboard win.
3. **Grounded in peer-reviewed literature** — every candidate cites a verified DOI from the KB.

### Verified cautionary facts (from reading the code, not agent reports)
- **Self-absorption correction is OFF by default** (`iterative.py`: `self_absorption_mode="off"`), AND a prior SA path was **deleted because it measurably worsened** intercept inflation on real ChemCam BHVO-2 (positive-feedback loop; audit `02-inversion-solver.md` F4). ⇒ Any SA-on change is **high-risk**, benchmark-gated hardest.
- Boltzmann fit is **weighted OLS** (`np.polyfit`), not errors-in-variables. ⇒ ODR is a real gap (but E_k errors are small; leverage to be measured, not assumed).
- Multiplet-aware Boltzmann aggregation **exists in `BoltzmannPlotFitter.fit`** but is not plumbed from the production solver. ⇒ wiring, not new code.

## Track A — additive UQ / robustness (cannot regress the point estimate; merge on unit tests + lint)

These add diagnostics, calibrated intervals, or model-selection *options*; they do not change the inversion's point estimate, so they need unit tests, not a benchmark gate. They are exactly what an elite physicist/mathematician reviewer looks for (calibrated, honest UQ).

| # | candidate | where | citation |
|---|---|---|---|
| A1 | **TARP** coverage test (posterior-sample coverage, no density eval) | new `cflibs/inversion/physics/coverage.py` + benchmark hook | Lemos, Coogan, Hezaveh, Perreault-Levasseur, ICML 2023, arXiv:2302.03026 |
| A2 | **Split Conformal Prediction + CQR** — distribution-free composition intervals wrapping the inversion output | new `cflibs/inversion/physics/conformal.py` | Lei et al., JASA 2018; Romano, Patterson, Candès, NeurIPS 2019 (CQR) |
| A3 | **Simulation-Based Calibration (SBC)** rank-uniformity harness over NUTS/dynesty | extend `cflibs/inversion/solve/bayesian/diagnostics.py` | Talts, Betancourt, Simpson, Vehtari, Gelman, arXiv:1804.06788 |
| A4 | **ILR/Aitchison analytical UQ** — full-rank covariance in ILR coordinates (closes the rank-deficient closed-composition covariance) | extend `cflibs/inversion/physics/closure.py` `ClosedFormILRSolver` | Egozcue et al., Math. Geol. 2003; Aitchison 1986 |
| A5 | **SpIC / AICc** model-selection options alongside the existing BIC element prune | extend `cflibs/inversion/identify/model_selection.py` | (KB topic_3 SpIC) + Hurvich & Tsai 1989 (AICc) |
| A6 | **Optimal-Estimation closure diagnostics** — posterior covariance Ŝ, averaging kernels, DOFS reported from the joint solve | extend `cflibs/inversion/solve/joint_optimizer.py` | Rodgers, *Inverse Methods for Atmospheric Sounding*, 2000 |

## Track B — accuracy-changing (opt-in flag first, benchmark-gated, default-flip only on a scoreboard win)

Ordered by (leverage / risk). Each agent adds the method behind a flag with a citation docstring and unit tests; **the default is not changed**. The parent then runs the scoreboard (reference pipeline, optimization-tier datasets) before/after and flips the default only if aggregate metrics improve and no dataset regresses past M3 tolerances.

| # | candidate | risk | citation |
|---|---|---|---|
| B1 | **Errors-in-variables / weighted TLS (ODR)** Boltzmann/Saha-Boltzmann slope | low | Aragón & Aguilera (SAB review); ODR = Boggs & Rogers 1990 |
| B2 | **Multiplet-aware Boltzmann** wired into the default solver (already supported by `fit`) | low | Völker & Gornushkin 2023; Wakil et al. 2023 |
| B3 | **Rigorous temporal/relaxation LTE criterion** beyond McWhirter, enabled in the solver | low | Cristoforetti et al., SAB 2010 |
| B4 | **Stewart-Pyatt IPD** consistent with the partition cutoff in the Saha balance | low | Stewart & Pyatt, ApJ 1966 |
| B5 | **pLTE thermalized-lower-level** line-selection cut | medium | Cristoforetti et al. (pLTE) |
| B6 | **STARK-B (Sahal-Bréchot 2015) + Djurović 2023** evaluated Stark widths into `stark_ne.py` | medium | Sahal-Bréchot et al., STARK-B/VAMDC; Djurović, Blagojević, Konjević 2023 |
| B7 | **Columnar-Density Saha-Boltzmann (CD-SB)** as an SA-tolerant quantification path | medium | Cristoforetti/Poggialini CD-SB; Aragón & Aguilera |
| B8 | **One-point calibration (OPC)** of the Saha-Boltzmann intercept | medium | Cavalcanti et al., SAB 2013 (doi:10.1016/J.SAB.2013.05.016) |
| B9 | **C-sigma generalized curve-of-growth** joint multi-element fit | medium | Aragón & Aguilera, JQSRT 2014 (doi:10.1016/J.JQSRT.2014.07.026) |
| B10 | **Enable observable-gated SA correction by default** (doublet-ratio + Planck-ceiling) | **high** (regression history) | Voelker & Gornushkin 2023; El Sherbini |

## Benchmark-gate protocol (Track B)

```
for candidate in track_B:
  implement opt-in flag + unit tests (physics-only)            # worktree-isolated agent
  run scoreboard --pipeline reference  (baseline, optimization-tier, seed 20260610)
  run scoreboard --pipeline reference  with the flag on
  adopt (flip default) iff aggregate F1/RMSE >= baseline AND no dataset regresses past M3 tol
  else keep opt-in + record the negative result (still valuable: documents what doesn't help)
```
Gate runs on SLURM (vasp nodes) after job `3256` frees, or on a fast local subset for quick triage.

## Execution
- **Track A** launches now (parallel, worktree-isolated, file-mostly-disjoint, unit-tested) → merges on green unit tests.
- **Track B** launches in localized, benchmark-gated batches (several touch `boltzmann.py`/`iterative.py`, so not all parallel-safe).
