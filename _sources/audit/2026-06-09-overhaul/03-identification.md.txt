# Audit 03 — Element Identification + Preprocessing Stack

**Date:** 2026-06-09 · **Scope:** `cflibs/inversion/identify/`, `cflibs/inversion/preprocess/`,
`cflibs/inversion/candidate_prefilter.py`, `native/cflibs-core` comb, identification→solver hand-off.
**Method:** read-only code audit + measured benchmark artifacts (`output/post-alias-fix-d553-phase7/`,
`output/wave2_alias_bench/`) + literature grounding. No code modified.

---

## 0. Executive verdict

The repo contains **seven identifier algorithms** and **two parallel element-decision stacks** (the
benchmarked identifiers in `identify/`, and a *separate* comb-gate cascade inside
`line_detection.py` that the production `cflibs analyze`/`invert` CLI actually uses). The
benchmark-best path (**hybrid_union**, micro-F1 = 0.673, 0.46 s/spectrum) is *not* the path that
feeds the CF-LIBS solver. The known failures (comb F1≈0.01–0.03 FN-bound; BHVO-2 major-element
deletion; Ag/Sn/W/Bi single-line FPs) all trace to a small set of structural causes: serial
independently-tuned gates, non-physical or mutually inconsistent line-strength metrics, and the
absence of any multi-line intensity-coherence test on the path that reaches the Boltzmann fit.

---

## 1. Findings, ranked by expected end-to-end accuracy impact

### F1 (highest impact) — The solver consumes the *worst-scoring* identification path; the best-scoring identifier is unused in production

**(a) Code behavior.** `cflibs analyze`/`invert` route through
`_detect_and_select_lines` (`cflibs/cli/main.py:161-343`) → `detect_line_observations`
(`cflibs/inversion/identify/line_detection.py:875`) → `LineSelector`
(`cflibs/inversion/physics/line_selection.py:136`). Element presence is decided by the
comb-gate cascade inside `line_detection.py` (precision/recall/missing-fraction gates,
§3 table), **not** by ALIAS, NNLS, hybrid, or BIC model selection. None of the benchmarked
identifiers (`alias.py`, `comb.py`, `correlation.py`, `spectral_nnls.py`, `hybrid.py`,
`model_selection.py`) sit on this path; `candidate_prefilter.py` (NNLS top-K) is used only by the
Bayesian workflow.

**(b) Grounding (measured, `output/post-alias-fix-d553-phase7/id_summary.json`, n=24 spectra,
real+synthetic mixed corpus):**

| workflow | micro-F1 | micro-P | micro-R | FP/spectrum | latency (s) |
|---|---|---|---|---|---|
| **hybrid_union** (NNLS ∪ ALIAS) | **0.673** | 0.703 | 0.646 | 1.83 | **0.46** |
| alias_high_recall | 0.584 | 0.736 | 0.484 | 1.17 | 3.97 |
| hybrid_consensus_weighted | 0.528 | 0.720 | 0.416 | 1.08 | 3.27 |
| alias_v2 | 0.429 | 0.694 | 0.311 | 0.92 | 1.98 |
| spectral_nnls (standalone) | 0.417 | 0.362 | 0.491 | 5.79 | 0.27 |
| alias (strict) | 0.201 | 0.679 | 0.118 | 0.38 | 2.88 |
| comb | **0.012** | 0.200 | 0.006 | 0.17 | 0.93 |

**(c) Fix.** Wire the element-presence decision in `_detect_and_select_lines` to `hybrid_union`
(NNLS screening + ALIAS confirmation, union semantics; `hybrid.py:180-183`), keeping
`detect_line_observations` only as the *line-extraction* stage for the elements hybrid_union
accepts. This is also the literature-shaped architecture: full-spectrum decomposition to narrow
candidates + line-level confirmation mirrors the ChemCam two-stage pipeline (Wiens et al. 2013,
Spectrochim. Acta B 82, pre-flight calibration & data processing,
sciencedirect.com/science/article/pii/S0584854713000505). Benchmark-gate the change (memory:
paper-faithful ALIAS regressed F1 −0.041 — every identifier change must be benchmark-gated).

---

### F2 — Comb identifier is FN-bound by a stack of four serially-tuned gates; two are not in the published method

**(a) Code behavior** (`cflibs/inversion/identify/comb.py`):

1. **Per-tooth activation** (`_correlate_tooth`, comb.py:890-973): tooth active iff Pearson r ≥
   `tooth_activation_threshold=0.5` **and** peak amplitude > threshold, where threshold = **85th
   percentile of positive baseline residuals** (`_estimate_baseline_threshold`, comb.py:851-856).
   The percentile gate scales with *spectral richness, not noise*: in a line-rich Fe/Ti matrix most
   positive residuals ARE lines, so the 85th percentile sits at real-line amplitude and only the
   top ~15 % of an element's lines can ever activate. Gajarska et al. gate on an SNR threshold;
   the percentile amplitude gate is a local substitution.
2. **Fingerprint floor** (`_compute_fingerprint`, comb.py:1341-1373 + `is_element_detected`,
   `common/element_id.py:253-293`): fingerprint = Σ(top-10 active correlations)/min(n_teeth,10)
   ≥ `min_correlation=0.12`, plus ≥ `min_active_teeth=2`. With a 50-tooth catalog and the fixed
   denominator 10, ≥2–3 *high*-correlation active teeth are needed just to reach 0.12.
3. **Tier-2 widen-only floors** (comb.py:40, 342-357): Mn/Na/K need r ≥ 0.7 and score ≥ 0.15
   — PR #166 tightened the FP-prone alkalis in the *opposite* direction of what the measured
   failure mode (FN-bound) required.
4. **Relative threshold** (`_apply_relative_threshold`, comb.py:725-754): detected elements with
   score < min(1, `2.0×median(non-zero scores)`) are downgraded, **with no exemption for the
   maximum**. On a balanced multi-element sample (scores e.g. 0.5/0.4/0.3/0.2/0.1 → threshold
   0.6) *every element including the top scorer is rejected*. This gate is not in Gajarska et al.

**(b) Grounding.** Autodiscovery experiment #24 (confirmed, surprise +0.32,
`docs/research/findings/2026-05-14-asta-24-…md`): comb mean FN = 7.06 vs FP = 0.36 per spectrum,
Wilcoxon p = 1.45e-06 — FN-bound with the threshold stack as the structural cause. Measured F1:
0.012 (phase7 unified corpus, table above) / 0.443 on the easy 11-candidate synthetic wave2 corpus
(`output/wave2_alias_bench/bench_baseline/summary.json`) — the collapse from 0.44→0.01 tracks
candidate-set size and matrix richness, exactly what gates (1) and (4) predict. Published method:
Gajarska, Brunnbauer, Lohninger, Limbeck, *J. Anal. At. Spectrom.* 39 (2024), "Automated detection
of element-specific features in LIBS spectra", DOI 10.1039/D4JA00247D (triangular teeth, per-tooth
r ≥ 0.5 + SNR gate).

**(c) Fix.** (i) Replace the 85th-percentile amplitude gate with the noise-calibrated gate already
in `preprocess/preprocessing.py` (`estimate_noise` σ × factor); (ii) delete or max-exempt the
2×median relative threshold and let consensus/hybrid filtering absorb the FP increase (the
autodiscovery recommendation); (iii) revert Tier-2 widen-only floors pending re-measurement.
Benchmark-gate each step independently.

---

### F3 — No multi-line Boltzmann-coherence test on the solver path: single-line Ag/Sn/W/Bi coincidences reach the Boltzmann fit

**(a) Code behavior.** Three doors let ≤2-line elements through to the solver:

- **Comb fallback** (`_select_accepted_elements`, line_detection.py:550-572): when no element
  passes the comb gates, elements with matched ≥ max(1,3) are accepted; if *none*, the **top-5
  elements with ≥1 matched line** are accepted (`comb_fallback_max_elements=5`).
- **Shift-coherence veto** (line_detection.py:1946-2026): an element with n=1 matched line has
  `enough_evidence=False` → `passes_count=True`; it survives iff its single residual lies within
  tol/3 of the consensus median — a ~33 % pass probability for a *random* coincidence inside the
  tolerance window. The veto is catalog-density-invariant (good) but cannot, by construction,
  separate a single coincident resonance line of Ag/Sn/W/Bi from a real line: same E_k band, same
  shift.
- **`LineSelector`** (`physics/line_selection.py:302-313`): `min_lines_per_element=3` only
  **warns** (`_warn_insufficient_lines`); single-line elements proceed into the Boltzmann plot.

Meanwhile the coherence machinery *already exists elsewhere and is not wired in*: ALIAS has a
per-ionization-stage Boltzmann consistency check with slope-sign, physical-T, and R² gates
(`alias.py:3263-3417`, `_r2_gate_rejects` alias.py:3070-3120) plus a hard `N_matched ≥ 3`
(alias.py:2010-2012); `model_selection.py:401` has `boltzmann_consistency_filter` (BIC + Boltzmann
linearity).

**(b) Grounding.** Multi-line intensity-consistency is the standard physics discriminator: the
model-spectrum correlation method encodes relative line intensities at trial (T, n_e) so that one
coincident line cannot score (Zaytsev, Popov, Labutin et al., *Anal. Chem.* 85 (2013),
"Automatic Identification of Emission Lines in Laser-Induced Plasma by Correlation of Model and
Experimental Spectra", DOI 10.1021/ac303270q); vector-of-weighted-peaks ranking similarly relies on
multi-peak evidence (Amato et al., *Spectrochim. Acta B* 65 (2010) 664-670); the published ALIAS
includes a probabilistic multi-coefficient assessment (Noel et al., *Spectrochim. Acta B* 2025,
"Automated line identification for atomic spectroscopy (ALIAS)",
sciencedirect.com/science/article/abs/pii/S0584854725001405). CF-LIBS practice treats agreement of
per-species Boltzmann temperatures as the internal consistency check (Tognoni et al., CF-LIBS
state-of-the-art review, *Spectrochim. Acta B* 65 (2010) 1-14).

**(c) Design assessment + fix (the proposed test is feasible and cheap).** For each accepted
element with N ≥ 3 matched lines in one ionization stage: fit ln(Iλ/gA) vs E_k; require slope < 0,
T within [3 000, 50 000] K, and R² ≥ floor → else reject. For N = 2: require the two-line ratio
temperature to agree with the global T estimate within a factor ~2. For N = 1: reject unless the
line is in a user-declared single-line whitelist (Hα, Li 670.8). **ps-LIBS caveat (T = 0.5–1.3 eV):**
detectable lines cluster in a narrow E_k span, so a naïve R² gate misfires — reuse the
`low_leverage` seam already engineered in `_fit_single_stage_boltzmann` (alias.py:3387-3408:
ptp(E_k) < 0.5 eV ⇒ trust R², distrust T) and the per-stage grouping (Saha offsets stage
intercepts; pooling collapses R² to ~0.01 — alias.py:3313-3331). Implementation point: between
`_select_accepted_elements` and `_collect_observations` in `detect_line_observations`, reusing
`BoltzmannPlotFitter`. This single gate addresses the Ag/Sn/W/Bi FP-poison class wholesale instead
of per-element tuning.

---

### F4 — The production comb gates are internally contradictory and use the wrong denominators

**(a) Code behavior** (`_score_comb_for_element`, line_detection.py:1335-1389; identical in Rust,
`comb_matching.rs:81-92`):

- `precision = matched/total_peaks` — this is not element precision; it divides by **all detected
  peaks of the whole spectrum**. On a rich spectrum (200+ peaks) an element with 3 true lines
  scores 0.015 < `comb_min_precision=0.02` → fails *because other elements emit*.
- `recall = matched/expected` where expected = **all top-30 catalog comb lines**, with no
  detectability weighting at the plasma T or the spectrum's noise floor. `comb_max_missing_fraction
  = 0.85` ⇒ recall ≥ 0.15 ⇒ ≥5 of the top-30 must match. Elements with 2–4 detectable lines
  (alkalis at ps-LIBS temperatures) are structural FNs.
- `comb_min_recall = 0.1` is **dead code**: missing_fraction ≤ 0.85 already implies recall ≥ 0.15.

**(b) Grounding.** Same FN signature as F2 measured by autodiscovery #24; the BHVO-2 dropped-major
incident (Al/K/P never reached the solver) flowed through these gates after the rel_int floor.

**(c) Fix.** Compute precision against *the element's own claimed peaks*; compute expected-line
count from gA-Boltzmann-detectable lines above the measured noise floor at the estimated T (the
emissivity-threshold idea ALIAS already implements at alias.py:3872-3945); delete the dead
`comb_min_recall`. Then re-derive the pass thresholds on the benchmark.

---

### F5 — One peak-detection knob on the production path is a dynamic-range gate, not a noise gate

**(a) Code behavior.** `detect_line_observations` finds peaks with `min_peak_height=0.01`
**relative to the spectrum maximum** (`_find_peaks`, line_detection.py:2068-2105: height ≥
0.01×max, prominence = 0.005×max). A spectrum whose brightest line is 10⁴× the noise hides every
real line below 1 % of that maximum regardless of SNR; conversely a flat noisy spectrum admits
noise. The canonical noise-calibrated detector exists (`preprocess/preprocessing.py:524-616`:
height ≥ 4.0σ, prominence ≥ 1.5σ, σ from 3-pass sigma-clipped MAD) and is used by comb/correlation
(`detect_peaks_auto`) — but *not* by the production line-detection path. Additionally
`_detect_and_select_lines` (cli/main.py:170-172) passes `wavelength_tolerance_nm=0.1` and
`peak_width_nm=0.2` **explicitly**, so the R-adaptive defaults engineered in
`_resolve_adaptive_tolerances` (line_detection.py:503-527, bead s1qr.2) can never engage from the
CLI.

**(b) Grounding.** For SNR 3–7 lines (the user's stated regime): a 4σ height + 1.5σ prominence
detector retains SNR ≥ ~4 lines; the 1 %-of-max gate retains them only if the spectrum's dynamic
range is < 100:1 — false negatives correlated with matrix brightness, the worst failure mode for
trace detection. Downstream, `LineSelector(min_snr=10.0)` (line_selection.py:138,246-247) then
**hard-rejects every detected line with SNR < 10 from the Boltzmann fit** — so weak-but-real lines
at SNR 3–7 currently cannot reach the solver at all, even when detected.

**(c) Fix.** Route `detect_line_observations` peak finding through `detect_peaks_auto`
(noise-calibrated); drop the explicit 0.1/0.2 nm constants in `_detect_and_select_lines` so
R-adaptive tolerances activate; lower `min_snr` to ~3 with inverse-variance weighting in the
Boltzmann fit carrying the de-weighting (the fit already consumes `intensity_uncertainty`).

---

### F6 — Five inconsistent line-strength metrics; NIST rel_int still load-bearing in three places

**(a) Inventory (Q3).** There is no single strength function; the metrics in use:

| # | Metric | Where | Notes |
|---|---|---|---|
| 1 | g_k·A_ki·exp(−E_k/kT), kT=1.0 eV, no U(T) | `line_detection.py:1244-1302` (`_ga_boltzmann_weight`, comb ranking/top-K) | the post-BHVO-2 fix; docstring of `_transition_strength` (line_detection.py:1263-1270) still claims it blends rel_int — stale |
| 2 | floor A_ki·g_k ≥ 5e3 (no Boltzmann), rank A·g·exp(−E_k/kT) at T=10 000 K, no U(T) | `comb.py:806-814` | floor and rank use different formulas |
| 3 | A·g·exp(−E_k/kT_ref), T_ref=10 000 K (screening); full Saha-Boltzmann ε with U(T) (scoring) | `alias.py:3050-3066` vs `_compute_element_emissivities` alias.py:3419 | only ALIAS scoring is fully physical |
| 4 | floor A·g ≥ 1e4 (`min_line_strength`) + full ε in model spectrum | `correlation.py:253` | |
| 5 | floor A·g ≥ 3e3 + gA-Boltzmann at 10 000 K | `wavelength_calibration.py:403-433` | |

Residual **NIST rel_int** dependence (non-quantitative metric): the legacy forward model filters
lines at `min_ri = 10.0` in non-NIST_PARITY modes (`radiation/spectrum_model.py:378-382`) — this
shapes *synthetic benchmark spectra*, silently deleting NULL/low-rel_int lines from truth;
`validation/round_trip.py:337` (`min_relative_intensity=10.0`); DB-build default floor
`rel_int ≥ 50` (`datagen_v2.py:20,365`; `atomic/database_generator.py:21` — the shipped
`libs_production.db` was evidently built with a lower floor: 9 758 of 28 727 lines have
rel_int < 50, and Al I 394.40/396.15 (rel_int 24/26) and K I 766.49/769.90 (rel_int 25/24) are
present). The DB's `rel_int` spans **0 → 6.4e8** (mean 1.3e5) — across-species scale chaos in one
column; `datagen_v2.py:354` maps NULL→0, so *any* floor ≥ 1 silently deletes unmeasured lines.

**(b) Grounding.** NIST itself: relative intensities "should be considered as qualitative", are
source/excitation-dependent, "are not basic data and must be used with caution" (NIST ASD Lines
Help, physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html). The BHVO-2 Al deletion was the
measured consequence.

**(c) Fix.** One module-level function `line_strength(transition, T, ne) =
g_k·A_ki·exp(−E_k/kT)/U_species(T) × saha_weight(T, ne)` (all inputs already in the DB/solver),
parameterized at the instrument's T window (0.5–1.3 eV); use it for every floor, top-K, ranking,
and screening. Remove rel_int from the forward model (replace `min_ri` with a strength floor) and
from `round_trip.py`; keep rel_int only as the aki_uncertainty heuristic (`database.py:316-332`),
which is a defensible qualitative use.

---

### F7 — ALIAS is a heavily drifted fork; the drift is now load-bearing but fragile (Q6)

**(a) Code behavior.** `alias.py` (4 926 lines) implements the published skeleton — peak detection,
synthetic line list over a (T, n_e) grid, detection-rate emissivity threshold
(alias.py:3872-3945), k_sim/k_rate/k_shift, CL = k_det×P_SNR×P_maj (alias.py:4723-4819) — per Noel
et al. 2025, then adds ≥12 non-paper stages: crustal-abundance prior **P_ab**
(alias.py:796-834, 4167-4193 — multiplies CL by 0.5–0.75 for Ag/Sn/W/Mo/Au), NNLS attribution +
sparse-NNLS gate + iron-group pre-subtraction (alias.py:1489-1493), peak competition, per-stage
Boltzmann consistency + adaptive-T R² gate, relative-CL gate, SA damping (×0.3 for E_i < 0.1 eV,
alias.py:4093-4109), adaptive detection threshold, and a hard `N_matched ≥ 3`
(alias.py:2010-2012). **Internal contradiction:** `_decide` deliberately allows single-line
(N_expected ≤ 1) and all-matched doublet elements (alias.py:4806-4813), but `_build_element_id`
then unconditionally kills anything with N_matched < 3 — the single-line H/Li and doublet Na/K
allowances are dead code, a structural alkali FN source (matches asta-12/asta-30 findings).
Biggest fragilities: (i) 30+ constructor kwargs and 4 presets whose interactions are only
constrained by the benchmark grid; (ii) **P_ab is a geological prior** — for the user's ps-LIBS on
non-geological samples (alloys, solders, coatings) it systematically suppresses Ag/Sn/W — true
positives, not just the FP confounders; (iii) the legacy T estimator reads ~2.3× cold on pure Fe
(alias.py:982-987 docstring), which the adaptive R² gate now compensates by *relying on* the cold
bias (the 15 000 K threshold) — a coupled pair of errors.

**(b) Grounding.** Wave2 A/B (`output/wave2_alias_bench/`): baseline (drifted) ALIAS F1 = 0.257 vs
paper-faithful "fix" 0.216 (−0.041, the memorialized regression) — the drift is empirically better
than paper-faithful *on this corpus*, so it cannot simply be reverted; but strict-preset ALIAS
still scores only 0.201 on phase7 vs 0.584 for its own high-recall preset. Published reference:
Noel et al. 2025, *Spectrochim. Acta B* (ALIAS), URL above.

**(c) Fix.** Treat ALIAS as the *confirmation* stage inside hybrid_union only (F1), not as a
standalone product. Specific surgical fixes, each benchmark-gated: remove P_ab (or gate it behind
an explicit `sample_matrix="geological"` flag); reconcile the N_matched contradiction by honoring
the `_decide` doublet path with the F3 two-line ratio test; adopt `temperature_estimator_mode=
"robust"` after measuring it.

---

### F8 — Rust comb is *almost* consistent with Python; greedy direction differs, and Rust kdet is dead code in the default config (Q2)

**(a) Code behavior.** Gate formulas are identical (verified: `comb_matching.rs:81-92` ≡
`line_detection.py:1366-1379`). But the one-to-one matching is greedy in **opposite directions**:
Rust `count_matches` iterates *peaks*, consuming transitions (comb_matching.rs:14-52); Python
`_match_transitions_to_peaks` iterates *transitions*, consuming peaks
(line_detection.py:1642-1678). On dense overlaps the two greedy bipartite matchings can yield
different match counts → pass/fail flips at the `matched ≥ 3` boundary depending on whether the
Rust extension is importable. Separately, the Rust kdet filter computes only the legacy density
score; because `shift_coherence_veto=True` is the default, `_kdet_filter_elements` always routes to
the pure-Python path (line_detection.py:1876: `if HAS_RUST_CORE and not shift_coherence`) — the
Rust kdet is unreachable in default configs.

**(b) Grounding.** HEAD commit `6dd4045 "Test comb line ranking"` indicates active work here; no
cross-backend equivalence test covers the dense-overlap case.

**(c) Fix.** Make Rust iterate transitions-first (matching Python), add a property test comparing
backends on synthetic dense-overlap fixtures, and either port coherence to Rust or delete the
unreachable Rust kdet entry point.

---

### F9 — Wavelength calibration: the right machinery, two residual gaps (Q5/K-offset)

**(a) Code behavior.** `calibrate_wavelength_axis_segmented` (RANSAC shift/affine/quadratic with
BIC model selection, seam detection; `wavelength_calibration.py:590-680, 817+`) now runs by default
before detection (cli/main.py:271-300). Reference pool is ranked by gA-Boltzmann (good;
wavelength_calibration.py:403-438). Quality gate: ≥12 inliers, RMSE ≤ 0.10 nm, inlier span ≥ 25 %,
|correction| ≤ 2.5 nm, match-fraction check disabled by default. Gaps: (i) on quality-gate failure
it falls back silently to the ±0.5 nm *constant* shift scan — exactly the model that cannot fix a
dispersion (slope) error like K's +0.17–0.27 nm red-end offset; the failure is logged at INFO only;
(ii) `inlier_tolerance_nm=0.08` is fixed, not λ/R-aware, so at the red end (766–770 nm, where K
sits) the inlier window is ~R=9600-equivalent — tight enough to reject true pairs that carry the
very dispersion error being fitted (pre-fit pairing uses a 2.0 nm window, so candidate pairs exist;
the risk is inlier classification during RANSAC refinement).

**(b) Grounding.** Field practice keeps per-channel wavelength recalibration as a first-class
pipeline stage with dedicated reference lines and temperature-dependent pixel shifts (ChemCam Ti
calibration target; up to 1 px shift per 10 °C — Wiens et al. 2013, *Spectrochim. Acta B* 82).

**(c) Fix.** Scale `inlier_tolerance_nm` with λ/R; surface quality-gate failure as a result
warning consumed by `analyze` (it changes element decisions); add a K/Na doublet-anchored
diagnostic to `validate_real_data.py`.

---

### F10 — Preprocessing is mostly literature-sound; defaults are the issue (Q5)

Baseline: median filter (default 10 nm) + SNIP (Ryan et al. 1988, *NIM B* 34:396, correctly
implemented with LLS transform, `preprocessing.py:95-187`) + ALS (Eilers & Boelens 2005,
`preprocessing.py:190-278`) + rolling percentile + opt-in AUTO (SNR ≤ 8 → ALS). This menu matches
LIBS practice; airPLS (Zhang, Chen, Liang 2010, *Analyst* 135:1138, DOI 10.1039/b922045c) is absent
but ALS is its direct ancestor and SNIP covers the sharp-peaks-on-slow-continuum case — no gap worth
new code. Noise: 3-iteration sigma-clipped MAD (`estimate_noise`, preprocessing.py:320-360) —
correct for peak-rich spectra. The problems are *wiring*, already covered in F5 (identifiers default
to MEDIAN, never AUTO; production line detection bypasses the noise model entirely). Air/vacuum:
the DB ingests NIST `obs_wl_air(nm)` (datagen_v2.py:327,348) and every consumer (forward model,
all identifiers) reads the same single `wavelength_nm` column — internally consistent; note NIST
returns vacuum wavelengths below 200 nm in that column (convention handled by NIST, fine for a
≥200 nm instrument), and Ritz-only lines (no observed λ) are dropped at ingest (`notna()` filter)
— a catalog completeness limitation, not an inconsistency.

---

## 2. Q4 verdict — multi-line Boltzmann-coherence test

**Feasible, recommended, and 70 % already written.** See F3(c) for the design. Reuse
`BoltzmannPlotFitter` + the per-stage grouping and low-leverage seam from
`alias.py:3313-3417`, inserted after comb element acceptance in `detect_line_observations`.
Two-line elements use ratio-T agreement; one-line elements rejected unless whitelisted. Literature
precedent: model-spectrum correlation (Labutin/Zaytsev 2013, DOI 10.1021/ac303270q) and the ALIAS
probabilistic stage (Noel 2025) both encode multi-line intensity consistency; CF-LIBS reviews treat
per-species Boltzmann-T agreement as the standard internal consistency check (Tognoni et al. 2010).
Expected effect: eliminates the Ag/Sn/W/Bi class FP (single resonance-line coincidence ⇒ no
N≥2 intensity consistency at any single (T, C)), at near-zero cost to real majors that pass with
3+ lines.

## 3. Q1 — Full gate-cascade tables

### 3a. Production solver path (`cflibs analyze`/`invert`)

| # | Stage | Gate / decision | Value | Location |
|---|---|---|---|---|
| 1 | Wavelength calibration | quality gate: inliers / RMSE / span / max-correction | ≥12 / ≤0.10 nm / ≥0.25 / ≤2.5 nm | `wavelength_calibration.py:605-610` |
| 2 | — fallback on failure | constant global shift scan | ±0.5 nm | `cli/main.py:270` |
| 3 | DB line load | rel_int floor (legacy, **non-physical**) | default **None** (was 100; round_trip still 10) | `line_detection.py:1199-1236`; `round_trip.py:337` |
| 4 | DB line load | top-K per element by gA-Boltzmann | K = 60 | `cli/main.py:168` |
| 5 | Peak detection | height ≥ frac of **max intensity** (not noise) | 0.01 | `line_detection.py:2089-2105` |
| 6 | Peak detection | prominence | 0.005×max | `line_detection.py:2103` |
| 7 | Peak detection | min distance | peak_width 0.2 nm / wl_step px | `line_detection.py:2098` |
| 8 | kdet pre-filter (coherence mode, default) | candidate peaks across ±0.5 nm scan | ≥ max(2, 2) | `line_detection.py:1753-1775` |
| 9 | kdet (legacy density mode, off by default) | density-weighted score | ≥ 0.05 (drops sparse majors, documented) | `line_detection.py:1766-1775,1866-1875` |
| 10 | Comb element gate | matched lines | ≥ 3 | `line_detection.py:889` |
| 11 | Comb element gate | "precision" = matched/**total spectrum peaks** | ≥ 0.02 (wrong denominator) | `line_detection.py:1367,890` |
| 12 | Comb element gate | recall = matched/top-30 catalog lines | ≥ 0.1 (**dead** — see 13) | `line_detection.py:1368,891` |
| 13 | Comb element gate | missing fraction | ≤ 0.85 (⇒ recall ≥ 0.15, supersedes 12) | `line_detection.py:1373,892` |
| 14 | Comb fallback (no passers) | matched ≥ 3, else top-5 with ≥1 match | 5 elements (**single-line FP door**) | `line_detection.py:553-572` |
| 15 | Shift-coherence veto | ≥50 % of matches within tol/3 of consensus; ≥2 coherent if n ≥ 2; n=1 passes if coherent | 0.5 / 2 / tol·⅓ | `line_detection.py:1994-2026` |
| 16 | Match tolerance | fixed (CLI overrides adaptive) | 0.1 nm | `cli/main.py:170` |
| 17 | LineSelector | SNR | ≥ 10 (reject) | `line_selection.py:138,246` |
| 18 | LineSelector | isolation factor | ≥ 0.5 (reject) | `line_selection.py:252` |
| 19 | LineSelector | resonance exclusion | False (CLI default since fix) | `cli/main.py:327-328` |
| 20 | LineSelector | min lines/element, energy spread | 3 / 2 eV (**warn only** — no element gate) | `line_selection.py:297-313` |
| 21 | LineSelector | max lines/element | 20 | `cli/main.py:179` |

Redundant/contradictory: 12 vs 13 (dead gate); 10 vs 14 (hard floor then bypassed by its own
fallback); 5 vs the unused noise model (two peak-detection philosophies); 16 vs the adaptive
tolerance code it permanently overrides. Non-physical metrics: 3 (rel_int; now defused on this path
but alive in forward model/round-trip — F6).

### 3b. ALIAS (when used: hybrid stage-2, benchmark, consensus)

| # | Gate | Value | Location |
|---|---|---|---|
| 1 | Peak threshold = noise × factor | 3.0 strict / 2.0 high_recall | `alias.py:1164-1173` |
| 2 | Fast screening (skipped if ≤10 elements given) | ≥2 of top-10 lines matched within 2λ/R; score ≥ 0.3; top-12 candidates | `alias.py:3003-3048,1601` |
| 3 | Emissivity threshold | lowest log-decade with >50 % detection; capped at top-20 lines | `alias.py:3872-3945` |
| 4 | Match tolerance | λ/eff_R per line (pass 1); ×2 for strong lines (pass 2); 1:1 by emissivity | `alias.py:3694-3753` |
| 5 | k_det blend + N_penalty | √(N_expected/5) cap 1 | `alias.py:4779-4791` |
| 6 | CL = k_det·P_SNR·P_maj·**P_ab(crustal prior)** | P_ab ∈ {0.5, 0.75, 1.0} | `alias.py:4796-4800,4167-4193` |
| 7 | Hard gate in `_decide` | N_exp ≤ 4 ⇒ all matched; > 4 ⇒ ≥3 matched; ≤1 ⇒ 1 OK (**dead**, see 12) | `alias.py:4806-4817` |
| 8 | P_local NNLS ownership | < 0.01/0.05 ⇒ CL=0 (unless ≥70 % matched & k_sim > 0.3) | `alias.py:2119-2122` |
| 9 | P_mix, R_rat soft gates | ×[0.2,1], ×[0.5,1] | `alias.py:2126-2129` |
| 10 | Boltzmann factor (per-stage) | ×[0.5,1.0] | `alias.py:2143-2152,3263-3417` |
| 11 | Sparse-NNLS zero ⇒ CL=0 at RP < 2000 | — | `alias.py:2158-2160` |
| 12 | Hard N_matched ≥ 3 | unconditional (kills H, Na/K doublets) | `alias.py:2010-2012` |
| 13 | R² gate | adaptive_t: est_T < 15 000 K ⇒ 0.3 else 0.85 | `alias.py:3070-3120` |
| 14 | Detection threshold | CL ≥ 0.02×min(3, √(10/N_exp)) (0.01 high_recall) | `alias.py:1997-2003` |
| 15 | Relative-CL gate | CL ≥ 0.1×max_CL (global or per-ion-stage) | `alias.py:1321-1409` |

### 3c. Other identifiers (decision gates only)

| Identifier | Detection rule | Location |
|---|---|---|
| comb | tooth: r ≥ 0.5 (Mn/Na/K 0.7) ∧ amp > P85(residual⁺); element: fingerprint ≥ 0.12 (0.15) ∧ active ≥ 2; then ≥ 2×median | `comb.py:287,965,1341-1373,738` |
| correlation (Labutin-style) | confidence ≥ 0.03 ∧ ≥ 1.5×median(non-zero) | `correlation.py:246,255,437` |
| spectral_nnls | coeff > 1e-10 ∧ SNR ≥ detection_snr ∧ coeff ≥ 0.05×max (0.0 in hybrid) | `spectral_nnls.py:70-82`; `hybrid.py:70-82` |
| hybrid | NNLS(SNR ≥ 1.5) ∧/∨ ALIAS | `hybrid.py:180-183` |
| BIC model selection | remove element if BIC drops; Boltzmann linearity filter | `model_selection.py:206,401` |
| candidate_prefilter (Bayesian) | NNLS SNR ≥ 3, coeff ≥ 1e-4×max, K ∈ [3,15], multi-T offsets ±1500 K | `candidate_prefilter.py:191-200` |

## 4. Q7 — Consolidation recommendation

Keep **two** paths, retire the rest from production (keep as benchmark baselines):

1. **hybrid_union** (NNLS screen → ALIAS confirm, union): best measured F1 (0.673), fastest
   (0.46 s), and the literature-shaped architecture. Wire it into `_detect_and_select_lines` as the
   element decision (F1), with the new Boltzmann-coherence gate (F3) replacing the comb fallback
   doors.
2. **spectral_nnls + candidate_prefilter** for the Bayesian workflow (already mandatory there).

Retire from production: standalone comb (F1 0.012 — fix per F2 only if it earns a consensus-voter
seat back), standalone correlation (F1 0.271 wave2, high FPR 0.24), the 8-cell ALIAS preset grid
(keep `high_recall_v2` + `strict` only), and `hybrid_consensus_weighted` (more cost, less F1 than
hybrid_union). Every change benchmark-gated against
`scripts/benchmark_synthetic_identifiers.py` + the unified harness, per the standing rule.

## 5. References

- Noel, C. et al. (2025). *Automated line identification for atomic spectroscopy (ALIAS):
  Application to LIBS imaging data processing.* Spectrochim. Acta B.
  https://www.sciencedirect.com/science/article/abs/pii/S0584854725001405
- Gajarska, Z., Brunnbauer, L., Lohninger, H., Limbeck, A. (2024). *Automated detection of
  element-specific features in LIBS spectra.* J. Anal. At. Spectrom. DOI 10.1039/D4JA00247D.
- Zaytsev, S.M., Popov, A.M., Labutin, T.A. et al. (2013). *Automatic Identification of Emission
  Lines in Laser-Induced Plasma by Correlation of Model and Experimental Spectra.* Anal. Chem.
  DOI 10.1021/ac303270q.
- Amato, G., Cristoforetti, G., Legnaioli, S., Lorenzetti, G., Palleschi, V., Sorrentino, F.,
  Tognoni, E. (2010). *Progress towards an unassisted element identification from LIBS spectra with
  automatic ranking techniques inspired by text retrieval.* Spectrochim. Acta B 65, 664-670.
- NIST ASD Lines Help — relative intensities are qualitative, source-dependent, "not basic data".
  https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html
- Wiens, R.C. et al. (2013). *Pre-flight calibration and initial data processing for the ChemCam
  LIBS instrument on MSL.* Spectrochim. Acta B 82.
  https://www.sciencedirect.com/science/article/pii/S0584854713000505
- Ryan, C.G. et al. (1988). *SNIP, a statistics-sensitive background treatment…* NIM B 34, 396-402.
- Eilers, P.H.C., Boelens, H.F.M. (2005). *Baseline Correction with Asymmetric Least Squares
  Smoothing.* Leiden University Medical Centre report.
- Zhang, Z.-M., Chen, S., Liang, Y.-Z. (2010). *Baseline correction using adaptive iteratively
  reweighted penalized least squares.* Analyst 135, 1138. DOI 10.1039/b922045c.
- Tognoni, E. et al. (2010). *Calibration-Free LIBS: state of the art.* Spectrochim. Acta B 65, 1-14.

**In-repo evidence:** `output/post-alias-fix-d553-phase7/id_summary.json`,
`output/wave2_alias_bench/bench_{baseline,fix}/summary.json`,
`docs/research/findings/2026-05-14-asta-24…md` (comb FN-bound, Wilcoxon p=1.45e-06),
`docs/research/findings/2026-05-14-asta-{12,30}…md` (alkali misses).
