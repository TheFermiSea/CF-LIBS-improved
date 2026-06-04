# Candidate-Count Fragility Audit — Identifier Pipeline (post-#216)

**Date:** 2026-06-03
**Worktree / HEAD:** `/tmp/cflibs-audit` @ `6b8fbf3` (`fix(hybrid): recover hybrid_union recall — disable NNLS mass floor in union arm (#216)`)
**Scope:** The CF-LIBS element-identifier pipeline (`cflibs/inversion/identify/`), the `hybrid_union` production-best arm, and the standalone/benchmark identifiers.
**Question under audit:** Was the #216 bug a near-isolated case, or is *"detection thresholds calibrated on small synthetic candidate sets that silently break at real 22–38-element candidate counts"* a **systemic** pattern in the identifier pipeline?

---

## 1. Headline Verdict

**NOT systemic. #216 was a near-isolated case.** The exact #216 failure shape — *a per-element score gated against a quantity that is **summed over all candidate elements**, so the effective bar tightens monotonically as the candidate count grows* — was found in **exactly one live place and one latent place**, and both are the **same mechanism** (NNLS `coeff / Σ(all_candidate_coeffs)`):

- The **production-best `hybrid_union` arm is already fixed and empirically count-flat** (recall 0.800 at n=10/20/38; §2). #216's `nnls_min_relative_coeff=0.0` default in `HybridIdentifier` is the correct and sufficient fix for the shipped union path.
- The **only other instance of the true #216 monotonic-tightening shape** is the *standalone* `SpectralNNLSIdentifier` default (`min_relative_coeff=0.05`), which #216 deliberately left in place — and which **no default CLI inversion path consumes** (`invert`/`analyze`/`bayesian` take user-supplied `--elements` and never instantiate an identifier). Its blast radius is the standalone/benchmark NNLS arm only.

Every *other* candidate that the hunt flagged as "#216-shaped" was **adversarially refuted** by direct reproduction (§4):

- The ALIAS `_apply_relative_cl_gate` and the comb/correlation median gates use **`max()` or `median()` over candidates, not `sum()`**. Their effective bar does **not** rise with raw count. Reproductions show ALIAS pinned at `max*0.1` independent of n, and the median gates moving the *opposite* direction (loosening as junk candidates accumulate) — the inverse of #216's "silent-on-synthetic, fails-on-real" property.
- The ALIAS gate is realism-sensitive to *dominance* (one strong element suppressing a true minor), which is a real but **different** fragility class, and on real ChemCam data it is **masked upstream** by the R²≥0.85 and ion-stage-pooling gates the 2026-06-03 composition-pipeline diagnosis already names as the binding constraints — so it has **no measurable effect on production output** at real candidate breadth.

The defensible conclusion: the count-scaling defect is **localized to the NNLS sum-normalized gate**, which exists in exactly two forms (union arm — fixed; standalone — left intentionally). It is **not** a pervasive design idiom across the identifier family. The meta-risk that *is* real and broad is **synthetic-only calibration provenance** (§5) — many thresholds have only ever been validated on the 11-candidate synthetic corpus — but that is a validation-coverage gap, not the count-scaling bug class.

---

## 2. The Empirical Count-Scaling Curve (the load-bearing evidence)

Controlled experiment on **one fixed real BHVO-2 spectrum** (`/tmp/cflibs-audit/data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv`, 6144 px, 240.8–905.6 nm). True 10 elements (BHVO2 basalt): Si Ti Al Fe Mn Mg Ca Na K P, all present in the 83-element basis (`/tmp/real_basis/basis_fwhm_0.1nm.h5`). Candidate count varied by subsetting basis rows (NNLS denominator) and the ALIAS element list together; +10/+28 distractor sets are monotone-nested. Full run reproducible in ~7.6 s.

| Identifier (config) | n=10 | n=20 | n=38 | Verdict |
|---|---|---|---|---|
| **`hybrid_union`** (production; `require_both=False`, `nnls_min_relative_coeff=0.0`) | **0.800** | **0.800** | **0.800** | **FLAT — #216 fixed on the production arm.** Si/Al/Na retained at all counts; only K,P missed at every count (a constant, count-independent miss). |
| `hybrid_union` *counterfactual* `nnls_min_relative_coeff=0.05` (pre-#216) | 0.600 | 0.500 | 0.500 | Reproduces the exact #216 regression (Si dropped 10→20). Proves the rig is faithful and that the **0.0 default is the only thing standing between production and the regression.** |
| `SpectralNNLSIdentifier` standalone **DEFAULT** (`min_relative_coeff=0.05`) | 0.600 | 0.500 | 0.500 | **COUNT-DEPENDENT DROP.** Si (most abundant, 23.3 wt%) silently cut 10→20. |
| `SpectralNNLSIdentifier` `min_relative_coeff=0.0` | 0.600 | 0.600 | 0.600 | **FLAT** — confirms the 0.05 floor is the *sole* cause. |
| ALIAS strict defaults | 0.000 | 0.000 | 0.000 | Flat, but pathologically zero recall on this real spectrum (a *separate* R²/ion-stage issue per the diagnosis, **not** count-dependent). |

**Mechanism, quantified (Si, `min_rel=0.0` so nothing gated):** Si `concentration_estimate = coeff / Σ(candidate coeffs)` = **0.0523** at n=10 (>0.05, detected) → **0.0474** at n=20 (<0.05, cut) → **0.0447** at n=38 (cut). Si SNR is essentially **constant** across counts (6.77 / 6.75 / 6.74). Only the *relative fraction* shrinks because the denominator (total coefficient mass) grows with added distractors. This is the #216 mechanism reproduced exactly on an independent real spectrum: a major element falls below a rising relative bar purely because more candidates were offered.

**Analytic confirmation (uniform candidate set, share = 1/n vs floor 0.05):** n=11→0.0909 PASS, n=20→0.0500 PASS, **n=21→0.0476 DROP**, n=38→0.0263 DROP, n=83→0.0120 DROP. Crosses exactly at n≈20.

**Precision-side count dependence (complementary, not a recall regression):** `hybrid_union` false positives grow with count — `+FP=[]` at n=10, `+FP=[Pb,V]` at n=20 and n=38. Precision falls 1.00→0.80 while recall stays 0.800. This is expected behaviour for a recall-favouring union arm (precision is recovered downstream) and originates in the ALIAS arm scoring more offered candidates, *not* in any count-sensitive relative floor.

**Curve summary line:** *Post-#216, the production `hybrid_union` recall-vs-candidate-count curve is FLAT (0.800 at n=10/20/38). The count-dependent collapse (0.60→0.50) survives ONLY in the standalone `SpectralNNLSIdentifier` default (`min_relative_coeff=0.05`), which is off every shipped CLI inversion path. ALIAS recall is count-flat (max-normalized gate); the comb/correlation median gates move the opposite direction from #216 (loosen, not tighten, as candidates grow).*

---

## 3. Ranked Confirmed Fragilities

Severity reflects mechanism reality AND production reach. "Production" = touches a shipped user-facing inversion or the production-best `hybrid_union` identifier's *output*.

### Rank 1 — F1: Standalone NNLS sum-normalized mass floor (the surviving #216 instance)
- **ID:** `F1-nnls-standalone-mass-floor`
- **Location:** `cflibs/inversion/identify/spectral_nnls.py:517` (`total_element_signal = np.sum(element_coeffs)` over ALL candidates), `:524` (`concentration = coeff/total_element_signal`), `:535` (`detected … and concentration >= self.min_relative_coeff`), default `DEFAULT_MIN_RELATIVE_COEFF=0.05` at `:44`.
- **Mechanism:** Byte-for-byte the #216 bug. The denominator is the sum over all candidate coefficients (83 for the real basis), so the per-element bar is structurally ~1/n-weighted and tightens monotonically with candidate count. #216 (6b8fbf3) did **not** remove it; it only set `HybridIdentifier.nnls_min_relative_coeff=0.0` (`hybrid.py:109`, forwarded `:173`) to bypass it in the union arm.
- **Severity:** medium. **Production:** false. Consumed only by the `nnls`/`spectral_nnls` benchmark workflows (`unified.py:1798` builds `SpectralNNLSIdentifier` without `min_relative_coeff` → inherits 0.05) and benchmark scripts. **Immune:** `hybrid_union` (overrides to 0.0); CLI `invert`/`analyze`/`bayesian` (no identifier-selection flag exists; they run `detect_line_observations` + solver on user-supplied elements); the Bayesian prefilter (`select_candidate_elements` reads `metadata['nnls_coefficient']/['nnls_snr']`, never `detected` — the floor is inert there).
- **Real-data evidence:** standalone NNLS on real BHVO-2, 83-element basis: recall 0.40 @0.05 vs 0.80 @0.00; majors cut despite strong SNR — Al(0.033/8.2), Mg(0.030/11.0), Na(0.013/5.7), Si(0.036/9.3). Corpus-wide (12 BHVO-2 spectra): mean true-major recall 0.475 @0.05 vs 0.650 @0.00.
- **Fix:** Replace the sum-normalized gate with a count-invariant criterion (absolute coefficient SNR is already computed; or normalize by top-K / max coefficient with a re-calibrated floor — note #216 found 5%-of-max re-admits the synthetic FP, so max-norm needs its own real-data calibration). Minimum: default the standalone identifier to recall-favouring (`0.0`) as the union arm does, or document standalone NNLS as synthetic-precision-only and unsafe above ~15 real candidates. The docstring at `:333-341` already acknowledges this scaling defect.

### Rank 2 — F2: ALIAS global relative-CL gate (dominance-sensitive, NOT count-dependent)
- **ID:** `F2-alias-relative-cl-gate`
- **Location:** `cflibs/inversion/identify/alias.py:1266-1268` (global branch: `relative_threshold = max_CL * relative_cl_threshold`; `e.confidence < relative_threshold → detected=False`), default `relative_cl_threshold=0.1` at `:835`, `relative_cl_per_ion_stage=False` at `:836`, invoked unconditionally at `:1874`; NOT overridden by `HybridIdentifier` (`hybrid.py:196-204` builds a bare `ALIASIdentifier`).
- **Mechanism:** Production-wired (the gate *does* run in the `hybrid_union` ALIAS arm). But the aggregate is **`max()`, not `sum()`** — the bar is pinned at `max_CL * 0.1` and does **not** rise with raw count. It rises only with *dominance*: a single high-CL major element drags the floor up under weaker true minors (the documented Vrabel s019 / PR #172 case where Al I at CL=0.95 force-cuts Mg II at CL=0.026). This is a real fragility but a **different class** from #216.
- **Severity:** low. **Production:** false (no measurable effect on shipped output at real breadth). On real 38-candidate ChemCam BHVO-2, toggling the gate on/off (`relative_cl_threshold` 0.1 vs 0.0) leaves the final detected set unchanged (`['Ca','Mg']` either way, recall 2/10, 0 FP). The recall collapse is driven entirely by the upstream R²≥0.85 and ion-stage-pooling gates (composition-pipeline diagnosis §3); at n=38 the elements F2 would cut are already gone before F2 runs. F2 only becomes load-bearing on the *smaller* n=11 set (where it cuts Fe at CL=0.042 because Ca dominates).
- **Real-data evidence (refutes count-dependence):** floor = `max_CL*0.1` is bit-for-bit identical at n=11 vs n=38 across 3 locations × 2 resolving powers (e.g. RP5000 loc1: 0.081592 at n11 = 0.081592 at n38, Δ=0). Adding 27 candidates raised the bar by exactly 0.
- **Fix:** Default `relative_cl_per_ion_stage=True` so a dominant neutral cannot suppress a true ion (the documented Al I → Mg II case; shipped test `tests/test_alias_unit.py:403-442` already asserts the per-ion-stage recovery), **or** thread `relative_cl_threshold`/`relative_cl_per_ion_stage` through `HybridIdentifier` so the union arm can opt to recall-favouring. Add a real-data assertion that a known minor true element survives. Note: this fix is independent of the count-scaling question — it is dominance hardening.

### Rank 3 — F4: ALIAS fast-screening top-K cap (count-dependent, latent only)
- **ID:** `F4-alias-fast-screening-topK`
- **Location:** `cflibs/inversion/identify/alias.py:2569` (keep top `max_screening_candidates`), default `12` at `:834`, gated by the `len(self.elements) <= 10` skip at `:1406`; `_always_test={'H'}` at `:1233`.
- **Mechanism:** A hard rank cutoff over the full candidate list (genuinely count-dependent: as n grows, a true minor is more likely displaced below rank 12 before scoring).
- **Severity:** low. **Production:** false. The cap is **gated off** for ≤10-element lists (`:1406`), and every shipped real CRM feeds ≤10 candidates (`bhvo2_usgs`:10, `bir1_usgs`:10, `nist_srm_612`:4, `vrabel2020`:10; `dataset.elements = sorted(true_composition.keys())`, `loaders.py:690`). The `hybrid_union` ALIAS arm sets `alias_elements = self.elements` (≤10), so screening is **skipped**. The "22–38 candidates" in the audit framing is the NNLS basis count (83), which is the F1 surface, not the ALIAS screening surface.
- **Evidence:** Verified count-dependent in isolation (`repro_screening.py`: at n=22 drops true Fe+Na, at n=83 drops Fe+Na+P). End-to-end verified inert: monkeypatch spy shows the `hybrid_union` ALIAS arm calls `_fast_screening` with an empty list (screening skipped); standalone ALIAS recall is 2/10 at n=10/22/38 identically (displaced elements already killed upstream by ion-stage/R² gates).
- **Fix:** Scale `max_screening_candidates` with candidate count, or gate purely on the per-element `score>=0.3` floor without a hard rank cap. Latent landmine if any future caller passes >13 candidates to ALIAS (e.g. `nist_steel`'s 13–31-element compositions, currently un-auto-registered). Validate all real true elements survive screening at n=38 before any such path ships.

---

## 4. What Was Checked and Found SOUND (count-invariant)

These were flagged as #216-shaped by the hunt and **adversarially refuted** by direct reproduction. They are NOT in the count-dependent monotonic-tightening class.

| ID | Location | Why sound (count-invariant) |
|---|---|---|
| **ALIAS relative-CL gate** (`F2`/`ALIAS-RELCL-GATE`) | `alias.py:1267` | Aggregate is `max()`, not `sum()`. Bar = `max_CL*0.1` is identical at n=11 and n=38 (Δ=0, verified on real data across 3 locations × 2 RPs). Tightens with *dominance*, not count. (Realism-sensitive — see Rank 2 — but not the count-scaling bug.) |
| **Correlation median gate** (`CORR-MEDIAN-GATE`) | `correlation.py:374-384` | Aggregate is `median()`. The hunt's own cited medians (0.304→0.262→0.228 as n=11→22→38) and bars (0.456→0.393→0.342) **fall** as n grows — a *loosening* gate, the opposite of #216's monotonic tightening. No distribution reproduces "passes synthetic / fails only at real 22–38." Benchmark-only (`hybrid_consensus_*`); never in `hybrid_union` or any CLI path. |
| **Comb median gate** (`COMB-MEDIAN-GATE`) | `comb.py:650-658` | Identical to above with scale 2.0. Direct mechanism test: holding a fixed true set and adding junk, bar **falls** 0.800 (n=5) → 0.200 (n=15/30); fraction of true elements cut **drops** 5/5 → 0/5 as junk grows. Harshest in the *small-n synthetic* regime, loosens on real data — inverse of #216. Benchmark-only voter; comb is NOT in `hybrid_union`. |
| **kdet rarity prefilter** (`F3-kdet-rarity-prefilter`) | `line_detection.py:1287,1316,1319` | `kdet_fraction = best_candidates/total_peaks` — `total_peaks` is fixed by the spectrum, zero `n_candidates` dependence. The median-density pivot moves the bar the **backwards** direction: on a dense real set the median rises and the effective bar *loosens* (5.27%→3.68% for a moderate element). Flip scan: every flip is "passes real, fails synthetic." True-element-protective in the realistic regime. CLI/comb arm only, not in `hybrid_union`. |
| **Correlation vector sum-norm** (`CORR-VECTOR-SUMNORM`) | `correlation.py:638,643` | The *exact* #216 sum-denominator shape (`score = weight[e]/Σ(all weights)`), and genuinely count-dependent — but **gated off in production**: the benchmark forces `mode='classic'` (`unified.py:1430`), and vector mode requires a configured FAISS index + embedder that no default path sets. Latent sibling, not an active regression. |
| **ALIAS synthetic-calibrated scalars / R² gate** (`F9`) | `alias.py:830-851`, `unified.py:985-998` | CL = `k_det·P_SNR·P_maj·P_ab` is purely per-element; `detection_threshold` and `adaptive_dt` scale by the element's OWN expected line count, never by `n_candidates`. The strict `r2_gate_mode='fixed'`/0.85 path is benchmark/preset-only; production `HybridIdentifier` inherits `adaptive_t` + `r2_gate_t_quality_threshold=15000.0` (post-#215), which spans the full ps-LIBS regime (verified: strict 0.85 only rejects at est_T≥15000 K, never across 5800–13774 K). Provenance concern survives (never re-swept on real n=38) but not count-dependent. |
| **F7: NNLS concentration-threshold sweep** | `unified.py:1984-1987` | A *second* instance of the #216 sum-denominator class, but it is a benchmark *experiment* workflow, not a shipping path. Its sweep optima are candidate-count-specific (documented). Low severity, off production. |
| **F8: `is_element_detected` Mn/Na/K floors** | `common/element_id.py:279-283` | Hardcoded per-element 0.15 score / 2-line floors. Synthetic-calibrated and a real recall risk for the exact majors #216 hit (Na/K), but **not count-dependent** (absolute per-element floors). A provenance concern, not a count-scaling bug. |

---

## 5. Recommended Actions

### Targeted fixes (in priority order)
1. **F1 (Rank 1) — close the surviving #216 instance.** Make the standalone `SpectralNNLSIdentifier` gate count-invariant: gate on absolute coefficient SNR (already computed) or top-K-normalized coefficient with a re-calibrated floor, OR default standalone `min_relative_coeff=0.0` to match the union arm, OR document standalone NNLS as synthetic-precision-only and unsafe above ~15 real candidates. (`spectral_nnls.py:44,517,524,535`.)
2. **F2 (Rank 2) — dominance-harden ALIAS** (independent of count-scaling): default `relative_cl_per_ion_stage=True` (`alias.py:836`) or thread the override through `HybridIdentifier` (`hybrid.py:196-204`) so a dominant element cannot silently suppress a true minor. The shipped test `tests/test_alias_unit.py:403-442` already pins the per-ion-stage recovery.
3. **F4 (Rank 3) — defuse the latent screening cap** before any >13-candidate ALIAS caller ships: scale `max_screening_candidates` with n, or rely on the per-element `score>=0.3` floor (`alias.py:834,2569`).
4. **Latent siblings (`CORR-VECTOR-SUMNORM`, `F7`)** — if correlation vector mode or the NNLS concentration sweep is ever promoted to a real-data path, replace the sum-normalized share with an absolute / fixed-reference score first; carry a per-path floor like #216's. Until then, document as unvalidated at 22–38 candidates.

### The meta-fix (the real systemic gap)
The count-scaling *bug* is localized, but the audit confirms a broad **validation-coverage** gap: nearly every identifier threshold's only provenance is the **11-candidate synthetic corpus** (or pre-reorg flat code), never the real 22–38-candidate BHVO-2/ChemCam distribution. The composition-pipeline diagnosis §7 Q1 already concedes the Wave-1/2 lift estimates are "bounded by reproductions on narrow synthetic spectra, not the real n=33 baseline."

**Institute a count-varying / real-data threshold-validation gate.** Concretely:
- Add a CI regression that runs each identifier (NNLS standalone, ALIAS, `hybrid_union`) on the fixed real BHVO-2 spectrum at **n = {true-count, +10, +28}** and asserts **recall does not degrade with candidate count** (the §2 rig — ~7.6 s, fully reproducible — is the template). This would have caught #216 and would catch any future sum-normalized gate.
- For every threshold default (`min_relative_coeff`, `relative_cl_threshold`, `detection_threshold`, `boltzmann_r2_min`, comb/correlation scales, the Mn/Na/K floors, `min_peak_height`, the comb acceptance floors), require a documented real-data provenance comment or a real-data sweep result. Replace `test_strict_alias_still_registered_unchanged`-style "no-change" pins (which are *not* calibration) with assertions tied to real-data recall/precision.
- Prefer count-invariant or noise-model-anchored floors over candidate-aggregate-relative floors wherever a gate must ship on a real-data path (absolute SNR, MAD-based n-sigma, fraction-of-explained-energy), so no future gate silently re-introduces the #216 shape.

---

## 6. Bottom Line

#216 was a **near-isolated case**, not the tip of a systemic count-scaling iceberg. The production-best `hybrid_union` arm is empirically count-flat (recall 0.800 at n=10/20/38). The only surviving true #216-shaped gate is the standalone NNLS floor, which no shipped CLI inversion consumes. The other "siblings" the hunt surfaced are either `max()`/`median()`-based (not count-tightening — several actually *loosen* on real data), gated off in production, or masked upstream. The genuinely broad risk is **synthetic-only threshold provenance**, addressed by the meta-fix: a real-data, count-varying validation gate for every identifier threshold.
