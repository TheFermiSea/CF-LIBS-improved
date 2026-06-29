# DED-PLAN Step 7 — constrained_elements no-drop flag (scoped, ready to implement)

**Status:** diagnosed in detail 2026-06-26; deferred to a focused session (ideally
with real Ti-6Al-4V spectra to validate against). Default-OFF, so it cannot
regress the current path; the work is in the new gated branch + its test.

## Goal
With a KNOWN element set, `run_pipeline` must never drop a requested element, so
real DED spectra (which go through detection, unlike the benchmark which bypasses
it) always reach the solver. Today `run_pipeline` on a synthetic Ti-6Al-4V
spectrum fails `"No usable spectral lines detected for inversion."`

## Diagnosis (clean synthetic Ti-6Al-4V, metallic_ded preset)
- `detect_line_observations` alone (default params) DOES find 23 obs across Ti/Al/V
  with zero drops -> raw detection is fine.
- Through `detect_and_select_lines` with the metallic_ded params, all elements drop:
  `dropped_elements = {Ti: selection, Al: detection, V: selection}` -> 0 lines.
- Relaxing the SELECTOR (`min_energy_spread_ev=0, min_snr=0, min_lines_per_element=1`)
  lets Ti+V through (Ti 85 / V 15) but **Al is still dropped at DETECTION** even on
  a clean, unshifted spectrum.
- V specifically fails `min_energy_spread_ev=1.5` (V's E_k spread is 1.27 eV).

## Two drop points to gate on `constrained_elements=True`
1. **SELECTOR** (`cflibs/inversion/physics/line_selection.py`, `LineSelector.select`):
   for a requested element, do NOT drop it on `min_energy_spread_ev`, `min_snr`, or
   `min_lines_per_element`; keep its best-available lines (still rank/cap them).
2. **DETECTION** (`cflibs/inversion/identify/line_detection.py`, `detect_line_observations`):
   the shift-coherence veto and/or per-line residual gate drop Al on a clean
   spectrum. When constrained, keep the wavelength CALIBRATION but do not DROP a
   requested element's matched lines at the shift-coherence / residual stage.
   (Investigate why Al drops with zero shift: likely the consensus-residual gate
   treating the Al resonance doublet as incoherent.)

## Plan
- Add `constrained_elements: bool = False` to `AnalysisPipelineConfig`; thread to
  `detect_and_select_lines` -> both the `LineSelector` and the `detect_kwargs`.
- Add the flag to the `metallic_ded` preset (the reserved slot already noted there).
- TEST: `run_pipeline` on a synthetic Ti-6Al-4V spectrum with
  `constrained_elements=True` returns a result with all of {Ti,Al,V} present and
  recovers ~the clean-floor composition; with it False, behavior is unchanged.
- Re-run `tests/benchmarks/ded_precision` + a broad not-slow regression.

## Why deferred, not done now
It spans two production-detection drop points with a subtle clean-spectrum Al
drop still to root-cause; its payoff is only realized on REAL spectra (no open
Ti-6Al-4V data exists). Better implemented carefully in a focused session than
rushed. The synthetic benchmark (which bypasses detection) already validates the
solver to ~1 wt%, so nothing is blocked in the meantime.
