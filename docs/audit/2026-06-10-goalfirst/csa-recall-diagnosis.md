# CSA planetary recall diagnosis (bead CF-LIBS-improved-0fuh)

**Baseline (seeded board, 12 spectra, seed 20260610):** P=1.000 R=0.267 F1=0.422,
RMSE 15.82 wt% (median), 1 failure — the hardest real benchmark on the
2026-06-10 scoreboard.

**After the two calibration fixes in this bead:** P=1.000 R=0.319 F1=0.484,
0 failures. RMSE median 29.18 wt% (regression explained in §4 — a pre-existing
solve-stage mechanism that the recall win now exposes on more spectra).

Reproduce:

```bash
JAX_PLATFORMS=cpu cflibs scoreboard --datasets csa_planetary --max-spectra 12 --seed 20260610 \
    --output-dir output/scoreboard_csa
```

## 1. The instrument, first

The CSA spectra (`data/csa_planetary_libs/`, adapter
`cflibs/benchmark/datasets/csa_planetary.py`) are **stitched 11-segment
broadband axes** (198.07–970.14 nm, 13 490 samples, step 0.034–0.053 nm with
0.31–1.25 nm inter-channel gaps; `detect_ccd_seams` finds 10 seams on every
spectrum — consistent with a multi-channel Ocean-Optics-class spectrometer).
Raw pulse-averaged counts, no certified resolving power (`resolving_power=None`).

Each channel carries its **own wavelength registration error**. Measured
per-segment shift fits (trusted, RMSE ≤ 0.05 nm) on the sampled spectra:
−0.03 nm (198–266), −0.38 (266–294), −0.11 (294–471), ~0.00 (505–586),
−0.31 nm (798–911). A single global shift model cannot represent this, and the
line-matching tolerance is 0.1 nm — so whichever channels the global
compromise leaves mis-registered lose their elements.

## 2. Miss matrix (seeded 12-spectrum board, before fixes)

TP=31 FP=0 FN=85 over 116 truth-element instances. Per-element FN/truth:

| El | FN | El | FN | El | FN |
|----|------|----|------|----|------|
| Si | 12/12 | P  | 10/10 | C  | 6/7 |
| Al | 11/11 | Ca | 10/11 | Mg | 4/11 |
| Mn | 11/11 | K  | 9/10  | Fe | 3/12 |
| Na | 8/10  | Ti | 1/11  |    |     |

The misses are **systematic, not random**: the silicate majors Si and Al are
missed on every spectrum; Ca on 10/11. Fe/Ti/Mg (dense line forests in the
well-anchored 294–471 nm channel) almost always survive.

**Stage attribution:** every FN element has `predicted_wt == 0.0` exactly and
`dropped_elements[el] == "detection"` — **zero FNs at the selection gate, zero
at solve-below-epsilon**. Nothing is being solved-then-dropped; the elements
never produce observations. (`detect_and_select_lines` diagnostics, all 12
spectra.)

### Sub-stage attribution inside detection

Replaying the detection internals (kdet → comb shift-scan → shift-coherence
veto → per-line residual gate) per truth element across the 12 spectra:

| Bucket | Count | Share of FN |
|---|---|---|
| comb gate failure (dominated by `comb_min_precision`) | 69 | ~80 % |
| shift-coherence veto (passed comb, then killed) | 17 | ~20 % |
| kdet, no-peaks, selection, solve | 0 | — |

Example (Andesite73302, 308 detected peaks; comb pass needs matched ≥ 3,
precision = matched/total_peaks ≥ 0.02 → **≥ 7 matched lines**, recall ≥ 0.1,
missing_fraction ≤ 0.85):

| El | matched/30 comb lines | gates failed |
|---|---|---|
| Ca | 14 | passes comb → **vetoed** (coherence) |
| Mg | 8 | passes comb → **vetoed** (coherence) |
| K  | 6 | precision (needs 7) |
| Si | 4 | precision + missing_fraction |
| Al | 4 | precision + missing_fraction |
| Na | 4 | precision + missing_fraction |
| Mn | 3 | precision + missing_fraction |
| P  | 1 | matches + recall + … |
| C  | 1 | matches + recall + … |

## 3. Root causes, ranked

### RC1 — catalog-aliased segment fits wreck the per-channel calibration (fixed)

`calibrate_wavelength_axis_segmented` correctly fits most channels (offsets
−0.03…−0.38 nm, RMSE ≤ 0.05). But on 5+ of the 12 spectra **one short channel's
RANSAC fit locks onto a self-coherent, wrong registration 1–1.8 nm away**
(e.g. Andesite73302 seg 266–294 nm: 13 inliers, RMSE 0.025 nm, shift −1.58 nm;
RedClay seg 471–505: +1.70; Ilmenite seg 198–266: −1.06). Dense catalogs make
a wrong global minimum easy on a 20–30 nm window with 10–20 peaks.

Two downstream behaviours, both bad:

- The aliased correction tears the stitched axis apart at a seam; the
  monotonicity restore needs > 0.5 nm cumulative shift and the code **reverted
  ALL segment fits to the global single-shift model** ("large cumulative seam
  shift" warning). The global shift (≈ −0.09 nm) leaves per-channel residuals
  of −0.10…+0.30 nm against a 0.1 nm matching tolerance: Si 251.61/288.16,
  Mg 279.6/280.3/285.2, Na 818.3/819.5 all land outside or at the edge of
  tolerance → matched counts collapse → `comb_min_precision` fails (bucket 1).
- When the seams happen to swallow the discontinuity (Ilmenite), the aliased
  segment **silently corrupts its channel by > 1 nm** — Ilmenite's entire UV
  (Fe/Mn-rich) channel was displaced, leaving only Ti.

The residual wavelength-dependent error also explains bucket 2: the
shift-coherence veto pools ONE consensus residual across the whole axis. The
consensus lands on the Fe/Ti-anchored 294–471 nm channel (+0.05 nm); Ca and Mg,
whose lines sit in channels with residuals of opposite sign (−0.03…−0.19),
pass the comb and are then **vetoed as incoherent** — the single-residual
assumption is false on a stitched instrument when per-channel correction has
failed.

**Fix applied (cflibs/inversion/preprocess/wavelength_calibration.py):**

1. `segment_max_global_disagreement_nm` (default 0.5): a trusted segment fit
   whose median correction departs from the global fit's correction over the
   same samples by more than the bound is demoted to the global fallback.
   Real channel offsets measured here are ≤ 0.4 nm; aliases are ≥ 0.9 nm —
   clean separation. Worst case equals the previous post-revert behaviour
   ("no segment worse than the global model").
2. All-fallback quality inheritance: when every segment falls back to the
   global correction the stitched result IS the global fit, so it now inherits
   the global fit's quality verdict instead of being declared
   `quality_passed=False` (which made the pipeline discard a quality-passed
   axis and re-detect on the raw axis — the Graphite failure mode).

Regression tests: `tests/inversion/test_wavelength_calibration.py::
TestSegmentGlobalDisagreementGate` (4 tests covering alias demotion, plausible
offsets kept, quality inheritance pass/fail).

### RC2 — scoring design: 29/85 FNs are structurally unwinnable

Truth presence uses cutoff 0.01 wt% (`PRESENCE_CUTOFF_WT`); a predicted element
is "called present" only at ≥ 0.5 wt% (`PRESENCE_EPS_MASSFRAC`). Every truth
element certified between 0.01 and 0.5 wt% is therefore a guaranteed FN even
for a *perfect* pipeline (predicting its true concentration still falls below
the presence epsilon). On this board: **29 of 85 FNs** (C 5, Mn 8, P 9, plus
trace-level K/Al/Ca/Mg/Si/Fe rows) → **recall ceiling 0.75**. This affects
every element-wt dataset, not just CSA. Scoreboard policy — reported, not
changed here (the campaign objective shares the presence rule).

### RC3 — `comb_min_precision` scales with spectrum richness (largest residual bucket)

`precision = matched_lines / total_peaks ≥ 0.02` demands 5 matches at 250
peaks and **13 at 650 peaks** (Ilmenite: Ca matched 12 real lines and failed).
Sparse-line emitters (K: 2 strong lines; Na: 4; Al: 4; Si: ~5 on this
instrument) cannot scale their physical line count with the sample's peak
richness, so the gate structurally deletes them on busy broadband spectra.
After the calibration fixes this remains the dominant residual bucket
(64 of 79 residual FNs). Changing it is identifier-scoring tuning with known
regression risk (paper-faithful-ALIAS precedent, PR #229) — recommended for a
benchmark-gated follow-up, not changed here.

### RC4 — global-max-relative peak floor starves weak channels

`min_peak_height = 0.01 × max(intensity)`: the saturated Na D blend
(the global max on most rocks) sets the floor ~50× above the C I 247.86 nm
peak (0.17 % of max) and ~2–5× above the P I 253.6/255.3 peaks. On a stitched
instrument whose channels differ by orders of magnitude in response, a
global-max-relative floor erases entire weak channels. C and P are mostly
< 0.5 wt% here (RC2) so the recall impact on this board is small, but the
mechanism is real for any weak-channel major. Reported, not changed.

## 4. Composition: RMSE 15.82 → why, and why it moves to 29.18 after the fix

Per-element signed errors (mean over scored spectra, before): Si −21.5,
Al −6.8 (the never-found majors count as 0 predicted), and Fe +12.7, Mg +16.7,
Na +12.8, K +8.5, Ti +10.2 — **the closure renormalizes the missing majors'
mass onto whatever survived detection**. RMSE 15.82 is the same detection
failure seen through the closure equation, not an independent solve problem.

After the fix, recall improves (Na found on 6 spectra vs 2, Mn/Mg gained,
RockFer2 and Graphite both solve) but RMSE median worsens 15.82 → 29.18.
Driver: the newly admitted **alkali resonance lines are strongly self-absorbed**
(the Na D blend is the brightest feature of the spectrum) and the geological
preset runs with `apply_self_absorption='off'`, so Na solves to 84–128 "wt%"
(K to 90) — Na signed error +57. This mechanism **pre-exists this change**
(before-board AndesiteAGV2: K 89.7, Na 51.7; BasaltBHVO2: Na 83.8) — finding
more real elements simply exposes it on more spectra. Note also that predicted
element mass fractions can sum to > 1 (Andesite73302 after: Σ ≈ 1.50) — the
oxide-closure output is not a normalized composition in this regime; that is a
solve-stage bug lead in its own right.

## 5. Gate results

CSA seeded board (12 spectra, seed 20260610):

| | P | R | F1 | RMSE med | failures |
|---|---|---|---|---|---|
| before | 1.000 | 0.267 | 0.422 | 15.82 | 1 (RockFer2) |
| after  | **1.000** | **0.319** | **0.484** | 29.18 | **0** |

No-regression boards (6 spectra, seed 20260610):

| dataset | before P/R/F1 | after P/R/F1 | new FPs |
|---|---|---|---|
| aalto | 1.000/0.471/0.640 | identical | 0 |
| chemcam_calib | 0.935/0.547/0.690 | **0.968/0.566/0.714** | 0 (one FP *removed*: SRM97A Ag) |
| silva2022 | 0.000/0.000/0.000 (all 6 fail) | identical | 0 |

Targeted tests: `tests/inversion/test_wavelength_calibration.py` 44/44,
`tests/cli/test_pipeline_defaults.py` 34/34.

## 6. Recommended follow-ups (not applied here)

1. **Segment-aware residual consensus** for the shift-coherence veto and
   per-line residual gate: on seam-detected axes, pool residuals per channel
   instead of globally. Removes the remaining veto bucket (15 instances after
   fix) without weakening the FP guard on single-channel instruments.
2. **Bound `comb_min_precision`'s implied match count** (e.g. cap the
   effective requirement at `min(ceil(0.02 * total_peaks), N)` with N ~ 6–8)
   or normalize per element by its detectable line count. Largest residual
   recall bucket (64 instances); MUST be benchmark-gated across all boards
   (identifier-scoring precedent: audit Family 5 regressed F1 −0.041).
3. **Self-absorption for alkali resonance lines at solve**: evaluate
   `apply_self_absorption='observable'` (CDSB) for the geological preset, or
   down-weight resonance lines in the Boltzmann fit when an element has ≥ 3
   optically-thin lines. This is what RMSE on CSA (and the Na/K signed errors
   on chemcam_calib) is actually measuring now.
4. **Closure normalization bug lead**: predicted mass fractions summing to
   1.5 (§4) — the oxide closure's output should never exceed unity.
5. **Presence-rule mismatch** (RC2): align `PRESENCE_CUTOFF_WT` (0.01 wt%)
   with `PRESENCE_EPS_MASSFRAC` (0.5 wt%) or report "detectable-truth recall"
   alongside raw recall; 34 % of CSA FNs are unwinnable by construction.
6. **Per-channel peak floor** (RC4) for stitched instruments.

## Artifacts

- Before/after seeded boards and per-spectrum JSON: `output/w8/csa_before/`,
  `output/w8/csa_after2/`, regression boards `output/w8/reg_before/`,
  `output/w8/reg_after/` (local, not committed).
- Diagnosis scripts (replayable): `output/w8/diag_stage2.py` (stage
  attribution), `output/w8/diag_buckets.py` (sub-stage buckets),
  `output/w8/diag_forensic.py` (per-element comb/veto forensics),
  `output/w8/diag_cal_segments.py` (per-segment calibration dump).
