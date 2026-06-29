# CF-LIBS solver head-to-head (classic vs ILR vs C-sigma)

Three drop-in solvers on the SAME detected lines (shared `run_pipeline` front-end + a
`LineSelector.select` spy), `supercam_labcal` n=15, scored in **both** bases the user asked for.
Driver: `scripts/benchmark_solvers.py`.

## Result — median RMSE_wt vs truth (lower = better)

| solver | PRODUCTION (mass) | ELEMENT-NUMBER | n |
|---|---|---|---|
| **iterative (classic)** | **2.27** | **4.06** | 15 |
| ILR closed-form | 11.51 | 5.96 | 15 |
| C-sigma (U-corrected) | 17.71 | 15.63 | 15 |

**The classic iterative solver wins decisively in both bases.** Neither alternative beats it on
this data.

## Honest reading + caveats
- **ILR** is much closer in the *element-number* basis (5.96 vs 4.06) than in production (11.51).
  Its production gap is partly a **fairness caveat**: I ran ILR with `closure_mode="standard"`
  while iterative uses the geological **oxide** closure (the oxide step is worth a lot on these
  samples). Element-number — which neutralizes the closure — is the fairer ILR comparison, and
  there ILR is competitive though still behind.
- **C-sigma is worst in both bases — but NOT because of the partition-function bug** (that is
  fixed and validated; see below). The Cσ-graph method is intrinsically sensitive on real LIBS:
  it needs accurate absorption cross-sections, the optically-thin/COG regime, and a
  well-conditioned common-line fit; on real spectra these degrade and the global fit drifts.
- **Scope:** n=15, one dataset (`supercam_labcal` is the only multi-element set with local data),
  single seed. A directional result, not a final verdict.

## The C-sigma partition-function bug (found + fixed)
`fit_csigma.relative_concentrations` are **`C_s/U_s`, not a composition** — the partition
function is omitted from the cross-section and the bare fit has no DB to restore it. Confirmed on
the verified oracle fixture (`U`=3.88/3.09/4.71, all different): raw fit returns
`[0.468, 0.387, 0.145]` ≈ `C/U` `[0.480, 0.361, 0.158]`, **not** true `[0.5, 0.3, 0.2]`.
Applying `×U_s(T)`: `[0.491, 0.324, 0.185]` ≈ truth. So the dominant error is the U-omission
(fixable, ~25× worst case across elements), with a small residual from the fixture's photon-rate
forward not being Cσ's COG master curve. `solve_csigma_composition(obs, db)` applies the
correction. Tests: `tests/inversion/physics/test_csigma_composition.py` (the bug + the fix +
DB-backed smoke), `tests/oracle/test_solver_validation.py` (self-consistency + ordering +
`cog_function` pure-math). **This is the payoff of "validate solvers with cflibs-formal":** the
oracle's per-element-`U_s` test is built to catch exactly this silent ~10–25× error.

## Takeaway
On this data the **classic iterative solver is the best inversion** of the three — alternatives
don't beat it. The reachable accuracy levers remain (per the lever study) **physics corrections**
(self-absorption: a measured −0.21 wt% win) and **atomic-data quality** (`composition_error_bound`
says per-line density error dominates on high-SNR data), not the solver choice or line selection.
