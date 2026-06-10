# Harness Bridge: BHVO-2 Gate Script vs CLI (bead vj82)

**Date:** 2026-06-10 (Wave 2)
**Branch measured:** `overhaul/w2-harness` (based on `origin/dev` @ `0223d09`, all Wave-1 fixes)
**Spectrum:** `data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv` (ChemCam, ns-laser Mars-chamber)
**Reference:** USGS BHVO-2 certified composition (10 elements, cert-10 RMSE in wt%)

## Problem

`scripts/measure_bhvo2_presence.py` (the integration gate, baseline RMSE 4.029) and a
flagless `cflibs analyze` resolved to the *same knobs* but produced different numbers
(script: T=9942 K, Al 17.0 wt%, Mn 5 obs, RMSE 4.029; CLI: T=10145 K, Al 24.3 wt%,
Mn dropped, RMSE ~5.8). Two harnesses = incomparable baselines.

## Fix

1. The shared pipeline (`AnalysisPipelineConfig`, `build_pipeline_config`,
   `detect_and_select_lines`, `run_pipeline`) moved from `cflibs/cli/main.py` to
   **`cflibs/inversion/pipeline.py`**; the CLI and the gate script both import it.
   CLI behavior is pinned bit-identical by `tests/cli/test_pipeline_defaults.py` +
   `tests/cli/test_analyze_invert_no_drift.py` (28 tests, all passing).
2. The script grew explicit one-factor ablation flags, defaults = CLI defaults:
   `--[no-]wavelength-calibration`, `--[no-]shift-coherence-veto`,
   `--[no-]confounders` (whether Ag/Sn/W/Bi/Th are added to the element request).
   `shift_coherence_veto` is now a first-class `AnalysisPipelineConfig` field /
   YAML `analysis.*` key (default True, unchanged behavior).
3. **sys.path provenance guard:** `python scripts/measure_bhvo2_presence.py` puts
   `scripts/` (not the checkout root) first on `sys.path`, so `cflibs` silently
   resolved to the venv's editable install — the **main checkout**, not the worktree
   the script sits in. The script now prepends its own repo root and records
   `cflibs.__file__` in stdout and the JSON summary.

## Root cause of the 4.029-vs-5.8 gap (measured)

The gap decomposes into **two confounded factors, neither of which is the
calibration or veto knob** (both knobs were ON in both original harnesses):

1. **Code provenance (sys.path trap), ~1.0 wt% RMSE.** The legacy script measured the
   *pre-Wave-1 main checkout*, not dev/W1. Reproduced bit-exact: running the
   pre-refactor script against the main checkout gives **RMSE 4.029, T=9942, Al
   17.05, Mn 5 obs** — the exact legacy baseline. The same script, same knobs, on
   Wave-1 code gives **RMSE 4.990, T=9783, Al 21.02** (row A). So `4.029` is not
   reachable by any knob combination on Wave-1 code; it belongs to pre-Wave-1
   physics (forward-model line-population fix, canonical partition fallback).
2. **Element request (confounders), ~0.8 wt% RMSE.** The script requests the 10
   certified elements *plus* Ag/Sn/W/Bi/Th confounders; the CLI run requested only
   the 10. On identical Wave-1 code and identical knobs this alone moves
   RMSE 4.990 -> 5.814, Al 21.0 -> 24.3, T 9783 -> 10145, and **drops Mn from 5 obs
   to 0** (rows A vs B). The wider request changes wavelength-calibration anchors
   and comb/kdet competition; Mn detection and part of the Al inflation are
   downstream of that.

**Bridge verification:** flagless `cflibs analyze --elements Si,Ti,Al,Fe,Mn,Mg,Ca,Na,K,P`
and `measure_bhvo2_presence.py --no-confounders` are now **bit-identical**
(T=10144.8, ne=2.174e17, RMSE 5.814, every element equal).

## Measurement matrix

All rows: `--closure-mode oxide --saha-boltzmann-graph` (geological preset), loc1
spectrum. `wlcal` = segmented RANSAC wavelength calibration; `veto` = shift-coherence
veto; `conf` = Ag/Sn/W/Bi/Th in the element request. RMSE over the 10 certified
elements, wt%.

| row | code | wlcal | veto | conf | T (K) | n_e (cm^-3) | RMSE | Al wt% (n_obs) | Mn n_obs | FP confounders | dropped majors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy 4.029 | pre-W1 main | on | on | yes | 9942 | 2.02e17 | **4.029** | 17.05 (6) | 5 | none | Mn,Na,K,P |
| A (defaults) | W1 | on | on | yes | 9783 | 1.99e17 | 4.990 | 21.02 (6) | 5 | none | Mn,Na,K,P |
| B (= flagless CLI) | W1 | on | on | no | 10145 | 2.17e17 | 5.814 | 24.34 (5) | 0 | none | Mn,Na,K,P |
| C | W1 | **off** | on | yes | 9871 | 1.89e17 | 3.214 | 7.17 (5) | 0 | none | Mn,**Mg**,Na,K,P |
| D | W1 | on | **off** | yes | 10206 | 2.70e17 | 7.695 | 0.00 (0) | 0 | **Ag 89.3**, W 0.9 | Ti,Al,Mn,Mg,Ca,Na,K,P |
| E | W1 | **off** | **off** | yes | 10477 | 2.65e17 | 4.705 | 0.00 (0) | 6 | Ag 34.6, Bi 22.1, W 3.0, Sn 0.7 | Al,Mn,Na,K,P |
| F | W1 | **off** | on | no | 9216 | 2.34e17 | 11.024 | 29.63 (6) | 0 | n/a | **Si**,Mn,Mg,Na,K,P |
| G | W1 | on | **off** | no | 10009 | 2.00e17 | 3.740 | 0.00 (0) | 0 | n/a | Al,Mn,Na,K,P |
| H | W1 | **off** | **off** | no | 9588 | 1.80e17 | 4.726 | 0.00 (0) | 7 | n/a | Al,Na,K,P |

Per-element predictions (wt%):

| row | Si | Ti | Al | Fe | Mn | Mg | Ca | Na | K | P |
|---|---|---|---|---|---|---|---|---|---|---|
| certified | 23.33 | 1.64 | 7.14 | 8.61 | 0.13 | 4.36 | 8.15 | 1.65 | 0.43 | 0.12 |
| legacy 4.029 (pre-W1) | 26.39 | 4.56 | 17.05 | 4.33 | 0.30 | 2.00 | 3.74 | 0 | 0 | 0 |
| A (defaults) | 24.65 | 3.71 | 21.02 | 3.88 | 0.26 | 1.85 | 3.83 | 0 | 0 | 0 |
| B (= flagless CLI) | 19.81 | 3.89 | 24.34 | 5.76 | 0 | 2.59 | 4.88 | 0 | 0 | 0 |
| C (no wlcal) | 31.08 | 5.12 | 7.17 | 7.73 | 0 | 0 | 5.25 | 0 | 0 | 0 |
| D (no veto) | 3.39 | 0.47 | 0 | 0.62 | 0 | 0.34 | 0.46 | 0 | 0 | 0 |
| E (no wlcal, no veto) | 14.13 | 1.73 | 0 | 3.28 | 0.36 | 1.03 | 1.57 | 0 | 0 | 0 |
| F (no conf, no wlcal) | 0 | 10.75 | 29.63 | 16.23 | 0 | 0 | 9.72 | 0 | 0 | 0 |
| G (no conf, no veto) | 31.49 | 5.97 | 0 | 8.94 | 0 | 4.19 | 7.48 | 0 | 0 | 0 |
| H (no conf, no wlcal, no veto) | 35.09 | 5.43 | 0 | 6.25 | 1.29 | 3.01 | 5.30 | 0 | 0 | 0 |

Raw JSON for every row: `output/bhvo2_measure/<row>.json` (not committed; regenerate
with the flags above).

## What drives Al 17 -> 24.3 and the Mn drop

- **Al inflation is NOT the calibration/veto knobs flipping** between harnesses
  (both were on in both). It decomposes as: pre-W1 -> W1 code **+4.0 wt%**
  (17.05 -> 21.02, legacy vs A) and dropping the confounders from the request
  **+3.3 wt%** (21.02 -> 24.34, A vs B).
- **However, wavelength calibration is the single largest Al lever on W1 code**:
  turning it off (C) collapses Al to 7.17 wt% (certified 7.14) and gives the best
  RMSE in the matrix (3.214) — but it also deletes Mg entirely, and without the
  confounders in the request the no-calibration path is catastrophic (F: Si deleted,
  Al 29.6, RMSE 11.0). The calibration's effect is mediated by which elements anchor
  the RANSAC fit, i.e. it is request-dependent and fragile in both directions.
- **The Mn drop is an element-request effect**: Mn survives detection (5 obs) only
  when the confounders are requested (A); with the 10-element request Mn gets 0 obs
  (B), again via shifted calibration anchors / comb competition.
- **The shift-coherence veto is load-bearing and must stay on for the gate's FP
  criterion**: with the confounders requested and the veto off (D), Ag absorbs
  **89.3 wt%** and every real major but Si/Fe is displaced; E shows the same FP
  blowout (Ag 34.6 + Bi 22.1) without calibration. Even with a clean 10-element
  request, veto-off deletes Al entirely (G, H: 0 obs).

## Status / decision left to the maintainer

No defaults were changed (CLI behavior is bit-identical; pinned by tests). The data
above is the input to that decision:

- The gate baseline on Wave-1 code with the maintainer's exact invocation is
  **4.990** (with confounders, row A) / **5.814** (CLI-parity request, row B) — the
  historical 4.029 is a pre-Wave-1 number and should be re-baselined.
- Row C (calibration off) beats all other W1 rows on RMSE but trades Mg for Al and
  is fragile (cf. F). If calibration is to be revisited, the candidate fix is in the
  calibration's anchor selection, not a blanket default flip.
- Na/K/P are never detected in any configuration (loc1, 240-906 nm) — pre-existing,
  out of scope here.
