# BHVO-2 composition-accuracy root-cause diagnosis (measured, falsifiable)

**Date:** 2026-06-04   **Branch:** feat/element-presence-redesign (off dev @ e4a9544, post #218/#219)
**Tooling:** `scripts/measure_bhvo2_presence.py`, `scripts/probe_al_detection.py`
**Data:** `data/bhvo2_usgs/{chemcam_bhvo2_loc1,csa_bhvo2_1000pulse}_spectrum.csv`
**Cert:** `cflibs/benchmark/reference_compositions.py::BHVO2_BASALT_USGS` (10 majors, mass frac)

## TL;DR

The CF-LIBS pipeline produces **garbage composition on real BHVO-2**, and the
binding constraints are **upstream of and different from** the 29p1
element-presence/comb hypothesis. Measured, in order of how-binding:

1. **Solver closure degeneracy (deepest).** Given a clean multi-element line
   set the inversion collapses to one or two elements. Fed **9 Mg + 9 Fe**
   observation lines (correct wavelengths), it returns **Mg = 0.0 %, Fe = 0.4 %**
   and dumps all mass into Na (68 %) + Ca (30 %). Production (cert-10) output is
   **Mn = 40.8 %** (cert 0.13!) on ChemCam, **Fe = 61.5 %** on CSA.
2. **Wavelength calibration.** ChemCam BHVO-2 has a constant **+0.10 nm** offset
   on every line; the global shift-scan applies **−0.15 nm** (wrong sign), so
   real lines never align. CSA has a **non-constant** dispersion error
   (Mg −0.10, Ca +0.01, Na −0.29, Si +0.40) that a single global shift cannot fix.
3. **Detection-gate cascade.** Even with correct alignment, `kdet` prefilter +
   `comb_min_matches=3` + `LineSelector(min_lines_per_element, min_energy_spread_ev=2.0)`
   drop the real majors (Al's 4 resonance lines span only 0.88 eV < 2.0).
4. (downstream, rarely reached) `rel_int>=100` SQL floor deletes Al's real lines
   (rel_int 24-26) **and every NULL-rel_int line**; resonance exclusion; the
   `_transition_strength` unit-mixing bug (rel_int ~tens vs A_ki ~1e8).

## Falsifiable measurements

| scenario (ChemCam loc1) | result |
|---|---|
| Production, cert-10 only | RMSE 14.18; Mn 40.8 / Fe 21.9 / Si 18.7; Al=Mg=Na=K=P=0 |
| Production, cert-10 + {Ag,Sn,W,Bi} | **Bi 70 %, Ag 21 %**, all real majors ~0 |
| Floor OFF | Na blows up to 96 % (confirms floor's purpose; naive removal is wrong) |
| Aligned (−0.10) + kdet off + comb relaxed, detect-only | all 10 elements get ≥1 line; Al's 4 resonance lines match within Δ≤0.016 nm |
| Aligned + sane gates + cert-10, full solve | **Na 68 / Ca 30, Mg 0, Fe 0.4, Si 0** (solver degeneracy) |

## Conclusion for 29p1

The element-presence comb redesign (`_transition_strength` + coherence veto)
is **necessary-but-far-from-sufficient**. It cannot move the north-star metric
while the solver collapses on clean input and the wavelength calibration is
broken. Re-scope: fix solver degeneracy + wavelength calibration + detection
cascade FIRST; element-presence/FP logic last.

## The Al-recovery cascade (why Al = 0)

Al peaks are real and strong (396.15 h=0.25, 394.40 h=0.16, in DB with
A_ki ~1e8). Al is dropped by, in order: wavelength offset > tolerance →
kdet prefilter → comb_min_matches=3 → rel_int floor → resonance exclusion.
Every gate must pass; the first four already fail.
