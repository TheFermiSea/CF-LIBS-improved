This is a synthesis-only task — all data is in the prompt. No tool calls needed.

# CF-LIBS Pipeline Threshold Sensitivity Study

One-at-a-time (OAT) sweeps of 11 fixed thresholds in the reference inversion pipeline, scored on two real datasets (`bhvo2_chemcam`, `supercam_labcal`) by median weighted composition RMSE (`rmse_wt`). All sweeps held the element basis constant unless noted: `median_elems` = 10.0 (bhvo2) / 15.5 (supercam), `n_ok` = 4/4 throughout — so RMSE comparisons are like-for-like (the "element-set-change trap" is checked per parameter and flagged where it bites).

---

## 1. Sensitivity ranking (most → least load-bearing)

Ranked by total RMSE range across each sweep (max − min), reported per dataset as `bhvo2 | supercam`. Boolean gates are ranked by their worst single-flip cost.

| Rank | Parameter | bhvo2 range | supercam range | Verdict |
|------|-----------|------------:|---------------:|---------|
| 1 | **min_lines_per_element** | 8.92 (2.43→11.35) | 10.89 (1.84→12.73) | **Critical.** Tightening past 5 is catastrophic — starves Boltzmann fits. |
| 2 | **wavelength_calibration** (gate) | 7.14 (off: 9.63) | 1.05 (off: 3.17) | **Critical input-quality lever.** Disabling ~4× the bhvo2 error. |
| 3 | **wavelength_tolerance_nm** | 28.96* / 3.91 clean | 2.95 (1.86→4.81) | **High** (one-sided). *0.03 collapse is a confound — see below. |
| 4 | **min_lines (max_lines_per_element)** R4 | 5.34 (3.30→8.64) | 0.37 | **High on line-rich R4 DB only**, via truncation below ~12. |
| 5 | **shift_coherence_veto** (gate) | 1.16 (off: 3.64) | 0.39 (off: 2.52) | **Load-bearing** quality gate. |
| 6 | **residual_shift_scan_nm** | 1.20 (2.49→3.69) | 0.31 (1.81→2.12) | **Sensitive, two-sided** (helps supercam, hurts bhvo2). |
| 7 | **min_peak_height** | 1.45 (2.49→3.93) | 1.78 (2.01→3.80) | **Sensitive, non-monotonic** with optimum at the default. |
| 8 | **isolation_wavelength_nm** | 1.37 (2.49→3.85) | 0.22 (2.12→2.34) | **Sensitive only at loose end** (cliff at 0.5). |
| 9 | **line_residual_gate** (gate) | 0.59 (off: 3.08) | 0.40 (off: 2.53) | Load-bearing quality gate. |
| 10 | **min_relative_intensity** | 0.62 (step at any >0) | 0.04 | Sensitive step-function; default (off) is best. |
| 11 | **stark_ne** (gate) | 0.53 (off: 3.02) | 0.13 (off: 2.25) | Load-bearing n_e constraint. |
| 12 | **max_lines_per_element** NIST | 1.48 (5→3.70) | 0.28 | Sensitive only below ~12; saturated 20–100. |
| 13 | **exclude_resonance** (gate) | 0.45 (on: 2.93) | 0.04 (on: 2.08, better) | Two-sided dataset-dependent tradeoff. |
| **— Inert —** | | | | |
| — | **min_snr** | 0.000 | 0.000 | **Arbitrary/inert** over 2–40 on these spectra. |
| — | **min_energy_spread_ev** | 0.000 | 0.000 | **Inert** — warns but does not reject lines. |
| — | **peak_width_nm** | 0.003 | 0.005 | **Inert** — detection-width absorbed before quantification. |
| — | **affine_coverage_gate** (gate) | 0.000 | 0.000 | **Inert** — already satisfied on this corpus. |
| — | **grade_aware_selection** (gate) | 0.000 | 0.000 | **Inert** — picks same lines as default selector. |

**Materially drive accuracy:** `min_lines_per_element`, `wavelength_calibration`, `wavelength_tolerance_nm`, the `min_lines` cap on the line-rich R4 DB, and the quality gates `shift_coherence_veto` / `line_residual_gate` / `stark_ne`.

**Arbitrary or inert on these benchmarks:** `min_snr`, `min_energy_spread_ev`, `peak_width_nm`, `affine_coverage_gate`, `grade_aware_selection`. These produce bit-identical RMSE across their full sweep — their defaults are *safe* but unconstrained by this corpus, so the values are not "validated," merely "not falsified." `min_snr` and `min_energy_spread_ev` are fully wired (override-verified) yet never bind, because other gates dominate line acceptance first.

---

## 2. The `max_lines_per_element` question

The cap retains the top-K strongest lines per element for selection and Boltzmann fitting.

**Does relaxing it help?** No. On both DBs the cap is **one-sided and saturated**: 20 → 30 → 50 → 100 give **bit-for-bit identical** RMSE on both datasets and both DBs. After the upstream min-line-strength gate, fewer than ~20 candidate lines per element survive, so the cap **never binds above the default** — it is inert in the 20–100 range. Raising it buys nothing.

**Does tightening it help?** Only narrowly and non-transferably. On the **NIST DB**, K=8 is the best on *both* NIST datasets (bhvo2 2.218, supercam 2.057), ~0.27/0.07 below the default. But the same K=8 **regresses the line-richer R4 DB** sharply (bhvo2 5.049 vs 3.304 at default). Below ~12 the cap truncates real signal; on R4, K=5 explodes bhvo2 to 8.644.

**Is 20 a good value?** **Yes — it is the robust cross-DB choice.** It sits on the saturated plateau (so it can't over-cap) and is the plateau top for R4 (any lower value strictly worsens R4). The NIST K=8 dip is a NIST-data-quality artifact (a few weak/noisy NIST lines beyond the 8th slightly bias the Boltzmann slope) — a NIST-only micro-gain that does not transfer.

**Does a different cap unlock R4's completeness?** **No.** R4's extra line coverage is *already captured* at K=20 (it's on the plateau) and is *actively hurt* by a lower cap. The cap is not the lever that converts R4's richer line list into better accuracy — that richness flows through regardless. R4 gives no reason to move off 20.

**Confound check:** clean. `median_elems` is constant across every K on both DBs; no scored elements dropped, no spectra failed. The cap only changes *which lines feed the fit*, not *which elements get scored*.

---

## 3. Best retuned config

Per-parameter best values, with confound flags. The honest read: **most of the apparent gains are dataset-specific and pull against each other**, so a single global retune yields little.

| Parameter | Default | Best (joint) | Δ RMSE bhvo2 | Δ RMSE supercam | Real win? |
|-----------|---------|-------------|-------------:|----------------:|-----------|
| min_lines_per_element | 3 | **2** | −0.056 | 0.000 | ✅ tiny, low-risk |
| wavelength_tolerance_nm | 0.1 | **0.2** | 0.000 | **−0.263** | ✅ clean (saturates 0.2=0.3) |
| min_peak_height | 0.01 | 0.01 (keep) | 0 | 0 | default is optimum |
| isolation_wavelength_nm | 0.1 | 0.1 (keep) | 0 | −0.007 (at 0.02) | negligible |
| residual_shift_scan_nm | 0.0 | **0.0** (keep) | 0 | (0.1 gives −0.31 but +1.20 bhvo2) | ⚠️ dataset-split |
| min_relative_intensity | None | None (keep) | 0 | 0 | default is optimum |
| max_lines_per_element | 20 | 20 (keep) | 0 | 0 | plateau; NIST-8 not transferable |
| min_snr / energy_spread / peak_width | — | keep | 0 | 0 | inert |
| boolean gates | all default | all default | 0 | 0 | default ≈ Pareto-optimal |

**Recommended retune (conservative, both-dataset-safe):**
- `min_lines_per_element`: 3 → **2**
- `wavelength_tolerance_nm`: 0.1 → **0.2**

**Expected combined effect** (assuming approximate additivity — *unverified*, needs a joint run):
- bhvo2: 2.486 → **~2.430** (−0.056, ~2%)
- supercam: 2.122 → **~1.859** (−0.263, ~12%)

**Confounds flagged (NOT real wins):**
- ⚠️ **wavelength_tolerance_nm = 0.03** "wins" nothing — it *collapses* bhvo2 to n_ok=1/4 (31.45 RMSE on a single surviving spectrum). This is a failure-mode datapoint, not an RMSE. The clean direction is *looser* (0.2), not tighter.
- ⚠️ **residual_shift_scan_nm = 0.1** improves supercam 15% but regresses bhvo2 48% — a calibration-residual tradeoff, not a global win. Keep 0.0 globally; enable only behind a calibration-offset detector.
- ⚠️ **min_lines_per_element = 5** gives supercam's best (1.841) but costs bhvo2 +1.2 — rejected; v=2 is the safe joint pick.
- ⚠️ **NIST K=8** for max_lines is a single-DB micro-gain that regresses R4 — not adopted.

**Caveat on all of the above:** these are 4-spectra/dataset medians (small-N). The −0.056 bhvo2 gain in particular is within plausible small-N noise; the −0.263 supercam tolerance gain is the only change with margin worth a confirming run.

---

## 4. Algorithm insights

**The Boltzmann fit is the fragile core; the line-count floor is its single point of failure.** `min_lines_per_element` dominates the entire sensitivity ranking (range up to 12.7 RMSE units). Demanding ≥8–12 lines per element doesn't drop *elements* (median_elems stays constant) — it starves the *kept* elements' Boltzmann-fittable line sets, collapsing the multi-element common-slope fit on under-constrained slopes. The pipeline needs **just enough** lines (2–5) to define a slope; more is wasted, fewer is catastrophic. This is the most important fragility in the system.

**Input quality, not the solver, gates real-data accuracy.** `wavelength_calibration` disabling nearly 4× the bhvo2 error (2.49→9.63), and `wavelength_tolerance_nm` too-tight starves the peak-to-line matcher into spectrum failure. Both are *matching/calibration* levers, not solver levers — consistent with the project's standing view that atomic-data and calibration input quality, not the inversion math, is the binding accuracy constraint on real spectra.

**Line-acceptance is governed by a gate hierarchy, and SNR is not at the top.** `min_snr` and `min_energy_spread_ev` are fully wired but completely inert (0.000 range over 2–40 / 0.25–3.0). The surviving lines sit far above any tested SNR floor, and energy-spread only emits a soft warning without rejecting lines. Acceptance is actually decided upstream by peak height, isolation, tolerance, and the min/max line-count caps. **Several "thresholds" are diagnostics, not gates.**

**Most knobs are saturated plateaus with a one-sided cliff, not smooth optima.** The recurring shape is: a flat region where the threshold doesn't bind, then a cliff when it starts truncating real signal (`max_lines` below 12; `isolation` at 0.5; `min_relative_intensity` at any nonzero value; `wavelength_tolerance` below 0.05). This means **the defaults mostly sit on safe plateaus with margin to the cliff** — well-chosen, not arbitrary — but it also means OAT sweeps under-explore the interaction regime where two knobs jointly approach their cliffs.

**`min_peak_height` is the rare genuine optimum at the default.** Non-monotonic with a clear minimum at 0.01: too low (0.002) admits noise that corrupts the fit, too high (0.05) discards real weak lines. The two datasets prefer slightly different optima (supercam tolerates a higher floor), and 0.01 is the robust compromise — a tuned value, not a coincidence.

**Quality gates earn their place.** `shift_coherence_veto`, `line_residual_gate`, and `stark_ne` each cost +0.1 to +1.2 RMSE when disabled — they remove genuinely-bad lines or supply a real n_e constraint. The all-default gate configuration sits at or near the Pareto optimum; **no single gate flip improves both datasets.** `exclude_resonance` is the only genuinely two-sided gate (helps supercam, hurts bhvo2).

**The two datasets have systematically different fragility.** bhvo2_chemcam (geostandard, denser Fe-group emission) is consistently more sensitive to *line culling and over-loosening* (isolation +55% vs +10%; min_relative_intensity ~14× more sensitive) — it has fewer redundant lines per element, so every line matters more. supercam_labcal benefits more from *looser matching* (tolerance, residual-shift). A single global config is therefore a compromise, and **per-dataset (or per-calibration-quality) tuning is where the remaining gains live.**

---

## 5. Next experiments

**2D interaction sweeps (OAT misses these by construction):**

1. **`min_lines_per_element` × `max_lines_per_element`** — the two line-count caps jointly define the Boltzmann line budget. The OAT plateaus may hide a coupled optimum: a slightly higher max + lower min could rebalance which lines anchor the slope. Highest-priority 2D given min_lines is the #1 lever.
2. **`wavelength_tolerance_nm` × `residual_shift_scan_nm`** — both are line-matching/calibration levers and clearly interact: a looser tolerance may make the residual-shift scan safe on bhvo2 (or redundant). Test whether tolerance=0.2 removes the bhvo2 regression from the shift scan, potentially unlocking the supercam 15% gain globally.
3. **`min_lines_per_element` × DB choice (NIST vs R4)** — R4's richer line list should change where the min-line floor bites; confirm v=2 stays safe on R4 (R4 already punishes tightening elsewhere).
4. **`wavelength_calibration` × `wavelength_tolerance_nm`** — when calibration is on, does tolerance still matter, or does calibration absorb the offset that tolerance otherwise compensates for?

**Hardcoded magic numbers needing code changes to sweep (not currently config-exposed):**

1. **The 0.10 default atomic-data uncertainty** — flows into Boltzmann weighting and the final uncertainty budget. Given the project's "data, not solver, limits accuracy" finding, this likely has real leverage on `rmse_wt` weighting; expose and sweep.
2. **Gate internal thresholds** — `shift_coherence_veto`, `line_residual_gate`, and `stark_ne` are on/off in this study, but each has an internal cutoff (coherence tolerance, residual cutoff, n_e bounds). The binary sweep shows they're load-bearing; the *continuous* threshold inside each is unexplored and probably tunable.
3. **`min_energy_spread_ev` enforcement** — currently a soft warning (`line_selection.py:420-422`). Wiring it to actually *reject* under-spread elements (not just warn) would make it a real gate; worth testing whether enforcement helps the Boltzmann slope conditioning that `min_lines` already governs.
4. **The min-line-strength prefilter upstream of `max_lines_per_element`** — this is what makes the cap saturate at 20. Its threshold is the *actual* binding constraint on line count; sweeping it would reveal whether the saturated max-cap plateau is real headroom or a hidden ceiling.

**Robustness experiments (to make the inert knobs meaningful):**

5. **Noise-injection sweep for `min_snr`** — it is inert here only because benchmark spectra are clean and well-resolved. Re-run on spectra with detected-line SNRs straddling 2–40 to characterize its true (currently untested) sensitivity before trusting the default=10 on noisy field data.
6. **Larger corpus** — every result is a 4-spectra median. The −0.056 bhvo2 gains are within plausible small-N noise; confirm the two recommended retunes (`min_lines`=2, `tolerance`=0.2) on a larger spectrum set before landing them.