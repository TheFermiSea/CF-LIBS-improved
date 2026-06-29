# Real-Data Composition-Accuracy Program — Summary (2026-06-27)

The "explore all levers, ground in truth" program that took CF-LIBS from a real-data failure to
a shipped, honestly-validated 4× accuracy gain. Companion: `real-steel-accuracy-levers.md`,
`real-steel-opc-promotion.md`.

## Result

**Real steel (PhdYoda/steel_266_LIBS, 36 NIST-class samples, honest held-out RMSEP, wt%):**

| stage | RMSEP | lever |
|---|---|---|
| baseline | 39.04 | naive constrained solve |
| neutral-anchor line selection | 30.43 | fix Cu ion-only Saha trap |
| + fixed optimal-T | 29.98 | (gain mostly from lines) |
| + robust OPC | 10.12 | conditioning-gated multi-standard F |
| + thin-line Fe filter | 9.56 | drop self-absorbed Fe lines |
| + CD-SB Fe matrix | **8.38** | columnar-density ordinate for self-absorbed Fe |

**4.7× reduction.** Per-element @8.38: Cu 2.5, Ni 2.2, Cr 2.8, Mn 2.7, Si 7.7, Mo 11.6, Fe 16.5.
(CD-SB keeps Fe lines via a width-derived columnar-density ordinate — composes with OPC, unlike
an intensity scale which double-corrects; keeping Fe right stops closure mass-bleed → Cu/Ni fall.)

**DED real goal (Ti-6Al-4V synthetic):** OPC cut the **V/Ti limiter 15.2% → 3.6%** (4.3×, R²
intact). Safe on the DED path (Al/Ti slightly over-corrected on clean synthetic — a real-data
bias OPC targets is absent there; on real DED data it would help, as on steel).

**DED real goal (Ti-6Al-4V REAL Mars data):** real SuperCam SCCT_TITANIUM observations (32
distinct-sol shots of the rover-deck Ti6Al4V wavelength-calibration plate; truth = nominal CCCT9
{Ti 89.5, Al 6.1, V 4.0} wt%). Constrained known-feedstock force-extraction (force-extract at the
KNOWN {Ti,Al,V} catalog line positions, peak-locked for the per-spectrometer wavelength offset,
detector-gap lines dropped) + matrix-matched OPC, **through the shipped pipeline**, takes the
honest leave-one-out held-out RMSEP from **24.1 wt% (early generic detection) → 4.88 (generic
detection, tuned) → 0.65 per-shot / 0.35 averaged**. The generic detector drops Al in ~44% of
these dense-Ti-forest spectra (Al silently vanishes from the closure); the constrained extraction
measures every known line on **100%** of spectra. Per-element @0.65: Ti 0.81, Al 0.60, V 0.49.
This is the first end-to-end real-Ti-6Al-4V DED-matrix validation (partially closes Open item 1).

**Real-data validation summary (honest held-out RMSEP, wt%):** real steel 8.38 · BHVO-2 (USGS
basalt) 8.11 · 66-standard ChemCam preflight 10.23 · real Mars Ti-6Al-4V 24.1 → 0.65/0.35
(constrained known-feedstock).

## What shipped (production cflibs, opt-in, default path byte-identical)

- `cflibs/inversion/physics/opc.py` — `OPCCalibration`, `calibrate_opc`, `apply_opc`,
  `choose_optimal_temperature`, `select_optically_thin_lines` (physics-only NumPy; structurally
  non-peeking — `calibrate_opc` sees only standards, `apply_opc` never reads recovered comp).
- `cflibs/inversion/physics/line_selection.py` — `select_lines_by_policy(policy="neutral_anchor")`.
- `cflibs/inversion/solve/iterative.py` — `fixed_temperature_K` (byte-identical when None).
- `cflibs/inversion/physics/opc.py` — also `cdsb_*` columnar-density primitives + `OPCCalibration.cdsb_scale`.
- `cflibs/inversion/pipeline.py` — opt-in `opc` + `opc_thin_filter` + `opc_cdsb_matrix` config (all
  default off → byte-identical); `cflibs/io/opc.py` JSON persistence; CLI `calibrate-opc` + `invert --opc`.
- `cflibs/inversion/physics/constrained_extraction.py` — `build_constrained_line_list`,
  `extract_peak_locked`, `constrained_extract` (peak-lock at KNOWN feedstock line positions,
  detector-gap drop; physics-only NumPy). `line_selection.select_lines_by_policy(prefer_spread=False)`
  for the strongest-cleanest joint-Saha-Boltzmann regime.
- `cflibs/inversion/pipeline.py` — opt-in `constrained_extraction` mode (`AnalysisPipelineConfig
  .constrained_extraction` / `known_elements` / `constrained_line_budget` / `…_window_nm` /
  `…_detector_gaps` / `…_instrument_fwhm_nm` / `…_search_tol_nm`): `run_pipeline` bypasses generic
  detection and composes with the shipped matrix-isolation + OPC. Default off → byte-identical.
- Shipped-API reproduction tests: held-out **8.38 wt%** (CD-SB, ≤8.5 guard) and 9.56 (thin-filter,
  ≤9.7) through the production path; real-Mars Ti-6Al-4V constrained+OPC **0.65 wt%** (≤0.8 guard,
  `tests/benchmarks/real_steel/test_opc_shipped_titanium.py`); OPC/pipeline unit tests + DED
  no-regression green.

## Diagnosis chain (each step diagnosed + NotebookLM-confirmed before fixing)

Cu (0.19 wt%) recovered as ~93% → root cause: Cu observed only via **ionized** lines at a low
fitted T → Saha `N_I ∝ N_II·exp(E_ion/kT)` explodes a phantom neutral population
(Zhao 2018; CF-LIBS review). Fix family: neutral-anchor lines, optimal-T, and OPC relative-
sensitivity calibration (Cavalcanti 2013). Fe matrix collapse → self-absorbed major lines;
fixed by *dropping* self-absorbed lines (a selection change composes with OPC; an intensity
*correction* double-counts OPC's F — empirically confirmed).

## Regime of validity (mapped)

- **Works:** dominant-matrix alloys (one major + minors) — steel (Fe 85-95%), **Ti-6Al-4V
  (Ti ~90% + Al/V)** = the DED deployment regime.
- **Does NOT generalize:** full-range binaries (Fe-Co ladder, every rung a different matrix) —
  OPC F averages to ≈1, no consistent bias to correct. Regime limit, *not* a data limit (Co has
  1022 usable DB lines). Benchmark: `tests/benchmarks/real_feco/`.

## Honesty discipline (no overfitting)

Every calibration uses standards' own data + certified truth only (in-sample conditioning gate),
never held-out samples; OPC F is a geometric mean over well-conditioned standards (clamp-saturated
degenerates filtered); leave-one-out held-out scoring throughout. Lower-but-fragile knob settings
were explicitly rejected. (Contrast: the earlier synthetic-clean Optuna "win" was an overfit,
caught + rejected by a robustness gate.)

## Open items (ranked)

1. **Ti-6Al-4V / Al-alloy real LIBS data with certified composition does not exist openly** —
   _partially closed:_ the real Mars SuperCam SCCT_TITANIUM Ti6Al4V plate (nominal-panel truth)
   now gives an end-to-end real-Ti-6Al-4V DED-matrix validation (0.65 wt% held-out via shipped
   constrained extraction + OPC). Still missing: an Al-alloy and a *certified* (not nominal-panel)
   Ti-6Al-4V CRM for absolute-accuracy confirmation. Needs in-house CRM acquisition, a data
   partnership, or paper-supplement extraction.
2. More steel (PhdYoda/266_steel_LIBS, 2160 spectra, license TBD) → deeper same-matrix stats.
3. Marginal/regime-limited levers: Stark-n_e (neutral-anchored solve is less n_e-sensitive),
   3D-CF-LIBS (needs time-resolved multi-delay data; steel set is single-delay), cluster knob
   optimization (36-sample overfit risk; results already shown stable → low headroom).
