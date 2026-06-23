# target_sigma_t wiring — benchmark findings (honest negative result)

**What:** wired the verified-`ErrorBudget`-derived thresholds (`derived_thresholds.py`) into the
line selector via a gated `target_sigma_t` knob (a target relative temperature accuracy σ_T/T),
so `min_snr`/`min_energy_spread`/`min_lines` are *derived per-spectrum* instead of tuned constants.

**Gate:** paired, same-element-set **conditioned RMSE**, flag-OFF vs `target_sigma_t ∈ {0.10, 0.05,
0.03}`, on the optimization-tier datasets with local data. Driver: `scripts/benchmark_target_sigma.py`.

## Result: measured NO-OP

| dataset | target | med_off | med_on | Δmedian | n | wins |
|---|---|---|---|---|---|---|
| supercam_labcal | 0.10 / 0.05 / 0.03 | 3.762 | 3.762 | **+0.000** | 20 | 0/20 |
| aalto | (all) | — | — | — | 0 | — (single-element Al → degenerate conditioning) |

**VERDICT: 0 improved, 0 regressed, 3 flat (±0.01 wt%). median Δ = +0.000 wt%.**

## Why (root cause — not a wiring bug)

Verified the derivation *fires* correctly on a real `supercam_labcal` spectrum (`target=0.03` →
derived `min_snr=26.7`, `min_energy_spread=2e-5`, `min_lines=2`). But:

1. **`min_snr` is the only HARD gate the derivation drives, and it never binds.** The detected lines
   on this dataset have **SNR = 449,000–5,300,000** (intensity / intensity_uncertainty). The gate
   threshold (legacy 10, or derived 27) is ~5 orders of magnitude below the *minimum* line SNR, so
   **every line passes regardless of the threshold**. Raising min_snr 10→27 rejects nothing.
2. **`min_energy_spread` and `min_lines` are advisory-only** in `LineSelector` (they emit warnings;
   they do not change the selected set). So deriving them changes no selection.

This **confirms the earlier sensitivity study from first principles**: `min_snr` (and
`min_energy_spread`) were already identified as *inert* levers; `min_lines_per_element` is
load-bearing only via downstream element handling, not the selector gate.

## Honest conclusions

- **The tuned magic numbers `min_snr=10` / `min_energy_spread=2` / `min_lines=3` were never
  load-bearing in the reference selection path.** Deriving them precisely from the verified error
  budget is *correct and generalizable*, but produces **no accuracy change** here because the
  pipeline does not bottleneck on these gates. The accuracy bottleneck remains elsewhere (atomic
  data, wavelength calibration/tolerance, solver element handling) — consistent with all prior M5
  findings.
- **The SNR gate is structurally inert because the dataset noise model gives ~10⁶ SNR.** A realistic
  per-line noise estimate (the gate was designed for SNR~10) is a prerequisite for *any* SNR gate —
  tuned or derived — to matter. This is a data/preprocessing gap, not a selector gap.

## Disposition

- **Do NOT flip the default.** No measured win → `target_sigma_t` stays `None` (default-off,
  byte-identical no-op; 94 selection/pipeline/oracle tests green).
- **Keep the knob as an opt-in.** The derived thresholds are conformance-pinned to the verified
  proof (`tests/oracle/test_derived_thresholds.py`) and *generalize to low-SNR regimes* (weak-signal
  / portable-instrument LIBS) where the gate WOULD bind — a principled alternative to a fixed 10 that
  a fixed constant cannot provide. On high-SNR lab data it is correctly a no-op.
- **The durable contribution of Bridge B is the conformance guard**, not an accuracy delta: it
  certifies the shipped thresholds match the machine-verified math regardless of benchmark coverage.
