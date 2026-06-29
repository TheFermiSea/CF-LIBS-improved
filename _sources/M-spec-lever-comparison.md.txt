# Formalization-derived selection levers — head-to-head comparison (complete)

All formalization-optimization levers, built to completion and benchmarked on the SAME paired
same-element-set conditioned-RMSE harness (`supercam_labcal`, n=20–30, seed 7, reference pipeline).
This is the valid apples-to-apples comparison: each lever wired (gated, default-off) and measured.

## The levers and their measured effect

| Lever | Theorem basis | Default regime | When it bites |
|---|---|---|---|
| **target_sigma_t** (derived SNR/spread/lines gates) | ErrorBudget (olsSlope_stable_l2, …) | **no-op** (SNR~10⁶ ⇒ min_snr gate inert) | low-SNR regimes where the gate binds |
| **reliability_ranked_selection** (max-energy-spread subset) | twoLineBeta_stable_sharp (2/\|ΔE\|) | **no-op** at default cap (never binds; ≤16<20 lines/el) | **when the cap binds — and then it WINS** |
| **refuse_to_report** (identifiability guards) | Identifiability + selfAbsorption_breaks_identifiability | **no-op** (0/30 flagged; healthy spectra) | degenerate/line-starved inputs |

## The one positive result: reliability ranking wins when selection binds

Forcing a binding cap (`max_lines_per_element=6`) to test the selection *criterion* itself:

| comparison (conditioned RMSE, wt%) | med A | med B | Δ(B−A) | B wins |
|---|---|---|---|---|
| score_cap6 (A) vs **reliability_cap6** (B) | 4.923 | **4.401** | **−0.522** | **17/20** |
| default20 (A) vs score_cap6 (B) | 3.762 | 4.649 | +0.887 | 0/20 |
| default20 (A) vs reliability_cap6 (B) | 3.762 | 4.401 | +0.639 | 5/20 |

**When you must choose a subset, picking the widest upper-level energy spread (the proven
`2/|ΔE|` conditioning) beats picking the highest-SNR-score lines by 0.52 wt% (17/20 spectra).**
The formalization's conditioning criterion is genuinely the better selection rule — validated, not
assumed. (`reliability_cap6` closes 0.52 of the 0.89 wt% gap that capping opens.)

## The honest meta-finding

**On high-SNR, multi-line lab data, the reference pipeline is not selection-constrained**, so every
selection/refuse lever is a no-op in the *default* regime:
- SNR ~10⁶ ⇒ the min_snr gate never rejects a line (target_sigma_t no-op, measured separately).
- ≤16 lines/element < the 20 cap ⇒ the cap never binds ⇒ every valid line is used (ranking no-op).
- every spectrum has a healthy ≥2-distinct-E T anchor ⇒ 0% flagged (refuse-to-report no-op).

And **capping at all costs accuracy here** (more high-SNR lines = better statistics: 3.76 uncapped vs
4.40–4.65 at cap 6). So on this data the right policy is: **don't cap, don't gate** — use every line.

## Where these levers ARE the right tool (their real domain)

The reliability criterion provably wins *when selection is forced* — which is exactly the
line-starved / capped / weak-signal regimes the default lab data never enters:
- **portable / handheld instruments** (few resolvable lines per element → the cap or line count binds),
- **real-time / speed-limited** inference (a deliberate small `max_lines` cap for latency),
- **low-SNR spectra** (the min_snr gate binds; target_sigma_t adapts it principledly),
- **degenerate inputs** (single-line species, single thick line + unknown τ → refuse_to_report flags
  what no estimator can recover).

So the formalization levers are **conditioning-aware tools for constrained regimes**, conformance-pinned
to proven theorems and default-off, not accuracy wins on healthy lab data. The accuracy bottleneck on
healthy data remains elsewhere (atomic-data accuracy, wavelength calibration, the solver) — consistent
with every prior M5 finding.

## Disposition
- All levers stay **default-off** (no default-regime win → no flip), available as opt-ins for the
  constrained regimes above. The reliability criterion's cap-bound win is the one measured positive.
- Scripts: `scripts/benchmark_lever_comparison.py`, `scripts/analyze_refuse_to_report.py`,
  `scripts/benchmark_target_sigma.py`. Modules conformance-pinned via `tests/oracle/` + the unit tests.
