# CF-LIBS Physics + Algorithm Audit — 2026-05-27

**Status:** Two-phase audit complete. Phase 1 = parallel code inspection by six area-specialist subagents. Phase 2 = literature validation of the strongest hypotheses via Asta literature search + targeted web/PDF retrieval. All quantitative claims are citation-grounded; no invention.

**Trigger:** macro_F1 = **0.402** on the Phase-7 cluster benchmark (n=24 spectra) versus a literature norm of ≥0.7 and a protocol-required floor of 0.6 (Si/Fe/Mg/Ca/Al/Ti/Mn/K/Na). The gap is large enough that there must be systemic physics/algorithm defects, not just parameter tuning.

**Verdict:** **Four cross-corroborated systemic defect families are present.** All are confirmed by published literature. A staged remediation should plausibly recover macro_F1 into the 0.6–0.8 range. The single highest-leverage fix is wiring self-absorption correction into the inversion loop (literature shows ×10 accuracy improvement); the highest-leverage-per-line-of-code is fixing the manifold's Nyquist-violating wavelength grid.

Audit artifacts: `.asta/documents/literature/find/lit-*.json` (8 Asta search snapshots).

---

## 1. The four defect families (cross-corroborated by ≥2 agents)

### A. Stark broadening is broken in four places

| # | Location | Bug | Severity | Literature |
|---|---|---|---|---|
| A1 | `cflibs/inversion/common/element_id.py:170` | reads `transition.stark_width_nm`; `Transition` exposes `stark_w`. ω_stark silently 0 → wavelength tolerance collapses to `λ/R`. | HIGH | Aragón, Pellé & Aguilera 2011 (*Anal. Bioanal. Chem.* 400, 3331): Al II 281.6 nm red-shifts by **130 pm** under Stark; without that budget in tolerance the line silently overlaps other elements. Gajarska et al. 2024 (*JAAS* DOI 10.1039/D4JA00247D): documented three Cr lines missed in steel from this exact mechanism. |
| A2 | `cflibs/radiation/spectrum_model.py:313` | hardcodes `apply_stark=False`; manifold and Bayesian paths default to True. SpectrumModel-based templates have no Stark. | HIGH | Aragón & Aguilera 2008 (*Spectrochim. Acta B* 63, 893): at n_e ~ 10¹⁷ cm⁻³ Lorentzian (Stark) ≥ Gaussian (Doppler+inst). Gaussian-only templates miss 30–50% of central FWHM. |
| A3 | `cflibs/manifold/basis_library.py:179` | NNLS basis is pure Gaussian (`exp(-0.5·((wl-λ_0)/σ)²)`). No Lorentzian wings. NNLS holds w=0.46 of consensus voting. | HIGH | Same. With NNLS as the dominant voter, Gaussian-only basis bleeds residuals into adjacent species. |
| A4 | `scripts/populate_stark_widths.py` | 80% of stored Stark widths are one reference value per (element, ion) scaled by `(λ/λ_ref)²`. **Phase-1 framing "intra-multiplet 2-5×" was wrong** (Wiese & Konjević 1982: intra-multiplet variation is normally < 10% in angular-frequency units); the correct concern is **inter-multiplet 2-10×** variation per Konjević et al. 2002. | MEDIUM-HIGH (correctness); MEDIUM (F1 lift) | Konjević, Lesage, Fuhr & Wiese 2002 (*J. Phys. Chem. Ref. Data* 31, 819); Lesage 2009 update. |

**Quantitative magnitudes** at n_e = 10¹⁷ cm⁻³, T = 10,000 K (Stark FWHM, pm):

| Line | Konjević measured | Repo populate value | Notes |
|---|---|---|---|
| Si I 250.69 / 251.92 / 252.41 | ~6 | ~9 | Repo over-estimates by ×1.5 |
| Si I 288.16 | ~5 | 12 | Over-estimate ×2.4 |
| Fe II 259.94 | ~7 | 18 | Over-estimate ×2.5 |
| Al I 396.15 | ~40 | 4 | **Catastrophic under-estimate ×10** |
| Ca II 393.37 (K-line) | ~34 | 8 | Under-estimate ×4 |
| Ca II 396.85 (H-line) | ~9 | 8 | OK |
| Mn II 257.61 | several pm to tens | 32 | Order-of-magnitude variable |
| Instrument FWHM @ R=10⁴, 300 nm | — | 30 | Stark is comparable to or exceeds it for Al I, Ca II K. |

**Confidence Phase-2 lift estimates:** A1 +1–5, A3 +2–6, A2 +2–8, A4 +1–3 macro_F1 pts. Total Stark cluster: **+6 to +20 pts**.

**Remediation order:** A1 (one-line attribute fix) → A3 (Voigt basis using existing `stark_w`) → A2 (default `apply_stark=True`) → A4 (line-resolved Stark data ingest from Konjević 2002 + STARK-B + Aragón 2014 LIBS-measured Fe II/Ni II).

---

### B. Self-absorption correction is dead code

| # | Location | Bug | Severity | Literature |
|---|---|---|---|---|
| B1 | `cflibs/inversion/physics/self_absorption.py` (2196 LOC), `cdsb.py` (806 LOC) | `SelfAbsorptionCorrector` and `CDSBPlotter` have **zero call sites** in the production inversion path. Verified by Serena `find_referencing_symbols`. The only "correction" in alias is `eps_th *= 0.3` constant damping for E_i<0.1 eV (`alias.py:3364`). | CRITICAL | Bulajic et al. 2002 (*Spectrochim. Acta B* 57, 339, 314 citations): recursive SA correction in the CF-LIBS loop is **"approximately one order of magnitude" accuracy improvement**. Poggialini, Palleschi et al. 2023 (*JAAS* DOI 10.1039/D3JA00130J): "self-absorption effects in most cases cannot be neglected" — post-Bulajic CF-LIBS replaces the optically-thin Boltzmann shortcut with the full radiation-transfer equation. John & Anoop 2023 (*RSC Adv.* 13, 29613): Mg-Ca 50:50 simulated test, uncorrected → 36.42% / 63.58% (**27% absolute error**); corrected → 49.33% / 50.67% (**2% error**). |
| B2 | `alias.py:2602-2684` `_apply_resonance_filter` | drops lines with E_i < 0.1 eV; Si I 250.69/251.92/252.41 (E_i=0.0096), Si I 251.43 (resonance E_i=0); cutoff also drops most of the Si analytical triplet. Boltzmann lever-arm collapses from ~5 eV to ~1.5 eV. **NOTE:** Phase-1 brief had Si I 251.611 E_i = 0.0096; **corrected NIST value is 0.0277 eV** — so 251.611 actually survives the 0.1 eV cutoff. But 250.69/251.92/252.41/251.43 (the rest of the triplet) are dropped, and the lever-arm point stands. | HIGH | Aragón & Aguilera 2008 §7: lines with E_i < ~1 eV should be **corrected**, not dropped. Direct conflict with the repo's policy. |
| B3 | `cflibs/inversion/physics/cdsb.py:429-504` `_estimate_initial_tau` | uses `density_factor = n_e/n_e_ref` instead of `n_lower/n_ref`. For major-matrix Si (25 wt%): n_total ≈ 7×10¹⁸ cm⁻³ vs n_e ≈ 10¹⁷ → initial τ underestimated by ~40-70×. CDSB iteration converges to "everything thin" before correcting anything. | HIGH | Aragón & Aguilera 2008 Eq. 7; El Sherbini 2005. |
| B4 | `self_absorption.py:1070-1112` `_apply_recursive_correction` | updates τ via `tau *= I_new/I_obs` — dimensionally wrong. τ is set by plasma state + atomic data, not by observed intensity. Mitigated only by `max_iterations=5`. | HIGH (correctness); MEDIUM (impact) | Bulajic 2002 algorithm updates τ from updated T, n_e, n_lower per iteration. The Phase-1 finding stands. |

**Quantitative τ_0 at typical soil (Si ~25 wt%, T = 10⁴ K, n_e = 10¹⁷ cm⁻³, L = 1 mm):**

| Line | E_i (eV) | τ₀ |
|---|---|---|
| Si I 250.690 | 0.0096 | **1.9** |
| Si I 251.432 | 0.000 | **1.6** |
| Si I 251.611 | 0.0277 | **5.8** |
| Si I 251.920 | 0.0096 | 1.2 |
| Si I 252.411 | 0.0096 | 1.6 |
| Si I 252.851 | 0.0277 | 1.9 |
| Si I 288.158 | 0.781 | **4.7** |

τ_0 ≥ 2 means strongly self-absorbed but not blackbody-saturated — exactly El Sherbini 2005's regime of validity for `SA = (Δλ_obs/Δλ_0)^(-1/0.46)`. Si I 288 substitution does **not** rescue the situation at 25 wt% Si (τ ≈ 5).

**Phase-2 lift estimate:** B1 +0.15 to +0.30 macro_F1 (largest single fix). B2 +0.05 to +0.15 (Si recall). B3 +0.05 (CDSB convergence speed). B4 +0.05.

**Remediation order:** B4 (τ-update bug) → B3 (CDSB initial τ) → B1 (wire SelfAbsorptionCorrector into IterativeCFLIBSSolver) → eliminate constant `eps_th *= 0.3` damping → adjust E_i cutoff after B1 lands (because correctly-modeled lines should not be dropped).

**Validation gate:** replicate John & Anoop 2023's Mg-Ca 50:50 case as a regression test; uncorrected ~27% error, fixed pipeline must converge to <5%.

---

### C. Partition functions disagree with direct-sum by 2–3× at LIBS T

At T = 10,000 K, from `cflibs/plasma/partition.py`:

| species | U_direct | U_poly | poly/direct | literature U(10⁴ K) | closer | source |
|---|---|---|---|---|---|---|
| Fe I | 58.6 | 33.8 | 0.58 | **~41-43** (Halenka 2002; Barklem-Collet 2016) | poly less wrong; **direct-sum overshoots by ~40%** — likely incomplete level catalogue + Rydberg-cutoff issue |
| Fe II | 66.9 | 47.5 | 0.71 | **~62-67** | direct-sum (within ~5%) |
| Ca I | 3.93 | 1.50 | 0.38 | **~3.7-4.0** | direct-sum (within ~5%) |
| Ti I | 75.6 | 32.0 | 0.42 | **~63-70** | direct-sum (within ~10-15%) |
| Si II | 5.87 | 2.00 | 0.34 | **~5.7-5.9** | direct-sum |
| Na I | 3.76 | 2.20 | 0.59 | **~2.4-2.7** | poly closer; direct-sum **overshoots** (Rydberg cutoff) |
| K I | 5.50 | 2.80 | 0.51 | **~3.0-3.5** | poly closer; same |

**Verdict:** Direct-sum is correct for Fe II, Ca I, Ti I, Si II. Both repo paths are wrong for Fe I and the alkalis (Na I, K I) — but in opposite directions, because direct-sum is missing the Debye-Hückel cutoff (`calculate_partition_function` doesn't pass `n_e`).

**Two IPD formulas disagree by 30%:** the disagreement is a **misapplied charge factor** (`2.09e-8·√(n_e/T)` is the Z=1 form valid for Saha I↔II; `3.0e-8·Z·√(n_e/T)` is the generalized Z-dependent form needed for Saha II↔III). Both are Debye-Hückel; both are correct in their proper domain. Stewart & Pyatt 1966 interpolates Debye-Hückel ↔ ion-sphere; at LIBS Γ ≪ 1 both reduce to Debye-Hückel. The Phase-1 framing "Saha uses smaller IPD formula" is technically correct but the literature explanation is more nuanced.

**Phase-2 lift estimate:** C +0.03 to +0.10 macro_F1 on identification; **larger lift on quantitation** (concentrations, abundances). The reason ID is less affected: peak-position dominates ID scoring; partition functions enter only through expected line intensities.

**Recommended remediation:**
1. Ingest **Barklem & Collet 2016** (*A&A* 588, A96; CDS J/A+A/588/A96; `github.com/barklem/Equilibrium`) as the primary U(T) source — covers all needed species, 1–2% accuracy.
2. Thread `n_e` into `saha_boltzmann.calculate_partition_function` so Debye-Hückel cutoff actually applies.
3. Standardize on `(Z+1)·2.09e-8·√(n_e/T)` for IPD; document as Debye-Hückel limit of Stewart-Pyatt.
4. Add a regression test asserting `|U_poly − U_direct| / U_direct < 5%` at T ∈ {5000, 10000, 15000} K for the benchmark species.

Sources actually consulted: Irwin 1981 *ApJS* 45, 621; Halenka & Madej 2002 (preprint astro-ph/0204384); Barklem & Collet 2016 *A&A* 588, A96 (arXiv:1602.03304); Tognoni, Cristoforetti, Legnaioli & Palleschi 2010 *Spectrochim. Acta B* 65, 1; Cristoforetti et al. 2007 *Spectrochim. Acta B* 62, 1287 (numerical study of theoretical-parameter error in CF-LIBS).

---

### D. Forward model + manifold has critical sampling + coverage defects

| # | Location | Bug | Severity | Literature |
|---|---|---|---|---|
| D1 | `cflibs/manifold/generator.py` defaults | `pixels=4096` over 250–550 nm → Δλ ≈ 0.073 nm with instrument FWHM 0.05 nm = **0.68 px/FWHM**. Need ≥ 3 px/FWHM (HARPS standard 3.2; LIBS Demidov 2022 standard 4.2). | CRITICAL | Robertson 2017 (*PASA* 34, e035; arXiv:1707.06455): below 2 px/FWHM, position bias grows as ~(px/FWHM)⁻² and width measurement degrades **~4× worse than the fine-sampled limit at 1.5 px/FWHM**. Magnier et al. 2025 (arXiv:2501.17163): at 13% undersampling (still well above 0.68 px/FWHM), point-sampled Gaussian **flux is overestimated by 50%**. At 0.68 px/FWHM the bias is uncalibrated, pixel-phase-dependent, and likely **30–200% on line area** — destroying the Boltzmann-plot ordinate. |
| D2 | `cflibs/radiation/kernels.py:540-542` | no continuum emission. No bremsstrahlung (`ε ∝ n_e n_i T^{-1/2} exp(-hν/kT)`), no free-bound recombination. | HIGH | Cristoforetti, Lorenzetti, Legnaioli & Palleschi 2010 (*Spectrochim. Acta B* 65, 787): continuum is **dominant at early delays**; even after gating, line-to-continuum ratios show **10–50% continuum in UV (250–350 nm), 5–25% in visible**. Gornushkin et al. 1999 (*Spectrochim. Acta B* 54, 491): canonical bremsstrahlung + free-bound formulas. Demidov 2022 mitigates via local baseline subtraction on narrow line windows (cheaper interim than full continuum forward model). |
| D3 | `generator.py:643, 809` | hardcodes `sigma_inst = 0.05/2.355` instead of `self.config.instrument_fwhm_nm`. User config is **silently ignored**. | CERTAIN BUG | n/a — pure software defect. Magnitude = whatever the user thought they were setting. |
| D4 | `generator.py:947-972` | if `len(elements)==4`, **hardcodes Ti–Al–V–Fe composition simplex** with `Al,V ∈ [0,0.12]`, `Fe ≡ 0.002`, Ti = remainder. Any other 4-element system has its simplex wrong by construction (Cu–Fe–Ni–Cr steel, geological 4-element panel, etc.). For `len ≠ 4`, falls through to a degenerate path. | CERTAIN BUG | Aitchison 1986 "Statistical Analysis of Compositional Data"; Petras 2009 arXiv:0909.0329 (constrained LHS on simplex); Graham et al. 2024 arXiv:2403.11374 (QMC on simplex for inverse problems). The correct recipe is Sobol or Dirichlet-LHS on the simplex. |
| D5 | `cflibs/validation/round_trip.py` `GoldenSpectrumGenerator` | manually computes `I ∝ C · f_stage · g_k · A_ki · exp(-E_k/kT) / U` with **no Stark, no continuum, no instrument response, no self-absorption**. Inversion runs the same formula. 19/19 round-trip pass at 10–40% tolerance is **a unit test on `exp(-E/kT)` arithmetic**, not validation. | CERTAIN | von Toussaint 2018 (arXiv:1805.08301): extracting information from measurements simply by agreement with simulations leads to **significant underestimation of uncertainties**. ChemCam team expanded validation set from 69 → 408 standards specifically because round-trip-only gave false confidence (Clegg et al. 2017 *Spectrochim. Acta B* 129, 64). |

**Phase-2 lift estimate:** D1 alone is the highest-impact-per-line-of-code fix in the cluster — **likely +0.10 to +0.20 macro_F1** just from going to ≥18,000 pixels.

**Remediation order:** D3 (one-line plumbing) → D1 (config change `pixels: 18432`; regenerate manifold once; ~4.5 px/FWHM at 0.05 nm) → D5 (real-sample validation harness using LIBSqsa or ChemCam standards) → D4 (Sobol/Dirichlet simplex sampler) → D2 (continuum — deferred until D1+D3+D5 reveal how big the continuum bias actually is in real-sample residuals).

---

## 2. Tier-2 supporting defects (well-grounded but lower individual leverage)

These would not on their own explain the F1 gap, but they each compound:

| Defect | File | Citation |
|---|---|---|
| `np.polyfit(w=1/σ²)` quartic-weighting bug; JAX path made bug-compatible | `cflibs/inversion/physics/boltzmann.py:890-895` | numpy.polyfit doc; Tognoni 2010 Eq. 5 (WLS form). |
| Charge balance unenforced in forward path; `n_e` consumed as user input | `cflibs/plasma/saha_boltzmann.py:340-372` | Tognoni 2010 §2.2 + §3.4: CF-LIBS requires simultaneous Saha + charge neutrality. Anderson solver exists in repo but isn't wired in. |
| Tier-2 K/Na/Mn score floor + `N_matched ≥ 3` hard gate over-rejects 2-line doublet elements (K I 766.5/769.9, Na D 588.99/589.59) at percent-level USGS-basalt abundances | `cflibs/inversion/common/element_id.py:182`; `cflibs/inversion/identify/alias.py:4026-4037` | Pace et al. 2017 ChemCam: doublet pair-matching, not generic n≥3 gates. USGS GeoRem BCR-2/BHVO-2/AGV-2: Na 2.3-3.2 wt%, K 0.5-1.4 wt%, Mn 0.13-0.17 wt%. |
| LTE validator uses ΔE between adjacent observed levels instead of largest transition gap; understates required n_e by **(ΔE_repo/ΔE_lit)³ ≈ 100-1000×**, "passes" non-LTE plasmas | `cflibs/plasma/lte_validator.py:244-260` | Cristoforetti et al. 2010 (*Spectrochim. Acta B* 65, 86). |
| A_ki accuracy grades D/E dominate alkalis (Na 94%, K 85%, Mg 83%, Al 61%, Si 58%); Boltzmann plots intrinsically noisy on these elements | `ASD_da/libs_production.db` | Wiese & Fuhr 2009 (*JPCRD* 38, 565) NIST critical compilations. |
| NNLS w=0.46 + threshold 0.40: **NNLS unilaterally triggers consensus detection** because 0.46 ≥ 0.40 | `cflibs/benchmark/unified.py` (post-#205) `ID_WORKFLOW_PRESETS["hybrid_consensus_weighted"]` | Dietterich 2000 ensemble methods: voters with correlated errors do not decorrelate. NNLS, ALIAS, comb, correlation all share the same NIST DB + Gaussian-profile forward model → errors are correlated. |
| Comb identifier has macro_F1 = **0.014** (Phase 7); per-tooth Pearson against triangular template doesn't shape-discriminate; threshold 0.12 has no literature basis | `cflibs/inversion/identify/comb.py:286` | (no positive citation — empirical failure documented in repo bead `s1qr.2`). |

---

## 3. Prioritized remediation plan

Ranked by **expected macro_F1 lift per engineering hour** and **prerequisite chain**:

### Wave 1 — single-day fixes, high confidence

1. **A1 (Stark attribute typo)** — `element_id.py:170` `stark_width_nm` → `stark_w`, unit normalize. 1-line change. (Aragón 2011 / Gajarska 2024.)
2. **D3 (instrument FWHM plumbing)** — `generator.py:643, 809` → read `self.config.instrument_fwhm_nm`. 2-line change. Unlocks empirical D1 testing.
3. **D1 (manifold sampling)** — config default `pixels: 18432` (≈4.5 px / 0.05 nm FWHM). Regenerate manifold once. (Robertson 2017 / Magnier 2025.)
4. **A2 (`apply_stark` default)** — `spectrum_model.py:313` flip to True (or expose as config with default True). 1-line.
5. **B4 (τ-update bug)** — `self_absorption.py:1070-1112` — remove `tau *= I_new/I_obs`, recompute τ from updated plasma state per Bulajic 2002.

### Wave 2 — 2–5 day fixes

6. **A3 (Voigt basis library)** — replace pure-Gaussian profile in `basis_library.py:179` with Voigt using existing `stark_w`. Regenerate basis library.
7. **B3 (CDSB initial τ)** — replace `n_e/n_e_ref` with `n_lower/n_ref` (Saha-Boltzmann population) in `cdsb.py:429-504`.
8. **B1 (wire SelfAbsorptionCorrector into IterativeCFLIBSSolver)** — Bulajic 2002 recursive algorithm. Then **remove** the constant `eps_th *= 0.3` damping in `alias.py:3364`.
9. **B2 (resonance-filter cutoff)** — once B1 is in place, raise/remove the E_i < 0.1 eV cutoff.

### Wave 3 — 1–2 weeks

10. **C (Barklem-Collet 2016 partition functions)** — CDS-table ingest; thread `n_e` into partition calls; standardize IPD formula.
11. **A4 (line-resolved Stark data)** — ingest Konjević 2002 + STARK-B + Aragón 2014 (LIBS-measured Fe II/Ni II in fused glass) into `stark_w` table with `accuracy_class` field.
12. **D5 (real-sample validation harness)** — LIBSqsa public corpus or ChemCam standards. Gate any future merge on independent F1 > 0.5 (initially) then ratchet upward.

### Wave 4 — larger / deferred

13. **D4 (Sobol/Dirichlet simplex sampler)** — generalize manifold composition coverage.
14. **D2 (continuum forward model)** — Gornushkin 1999 bremsstrahlung + free-bound. Defer until D1+D3+D5 reveal continuum residuals in real-sample fits.
15. **Tier-2 fixes** (polyfit weights, charge balance enforcement, doublet-pair matching for K/Na, LTE validator ΔE, etc.).

---

## 4. Validation methodology recommendations

- **Synthetic regression for self-absorption:** replicate John & Anoop 2023 Mg-Ca 50:50 case (T=1 eV, n_e=10¹⁷, L=1 cm). Uncorrected gives 36.42% / 63.58%; B1 fixed pipeline must converge to within 5% of 50/50.
- **Synthetic regression for partition functions:** require `|U_pipeline − U_BarklemCollet| / U_BarklemCollet < 5%` at T ∈ {5000, 10000, 15000} K for Fe I, Fe II, Ca I, Ti I, Si II, Na I, K I.
- **Synthetic regression for sampling:** generate identical inputs at 4096 vs 18432 vs 36864 pixels; require line-area agreement within 1%.
- **Independent real-sample validation:** *Required before any release* once D1 lands. Sources: LIBSqsa public corpus; ChemCam mission standards; Castorena et al. 2022 cross-instrument transfer corpus.

---

## 5. Citation index (most consulted)

| # | Citation | Used for |
|---|---|---|
| 1 | Tognoni, Cristoforetti, Legnaioli, Palleschi 2010 *Spectrochim. Acta B* 65, 1. DOI: 10.1016/j.sab.2009.11.006 | CF-LIBS state of the art; Saha-Boltzmann Eq. 1, Boltzmann Eq. 2, emissivity Eq. 3, partition-function role. |
| 2 | Aragón & Aguilera 2008 *Spectrochim. Acta B* 63, 893. DOI: 10.1016/j.sab.2008.05.010 | Plasma characterization review; Voigt requirement at LIBS densities; LOS integration; SA correction §7. |
| 3 | Cristoforetti et al. 2010 *Spectrochim. Acta B* 65, 86. DOI: 10.1016/j.sab.2009.11.005 | LTE beyond McWhirter; relaxation-time + spatial-gradient criteria. |
| 4 | Cristoforetti, Lorenzetti, Legnaioli, Palleschi 2010 *Spectrochim. Acta B* 65, 787. | Continuum radiation magnitude in LIBS. |
| 5 | Bulajic, Corsi, Cristoforetti, Legnaioli, Palleschi, Salvetti, Tognoni 2002 *Spectrochim. Acta B* 57, 339. DOI: 10.1016/S0584-8547(01)00398-6 | Foundational SA-correction algorithm; ×10 accuracy gain on certified NIST steels. |
| 6 | El Sherbini et al. 2005 *Spectrochim. Acta B* 60, 1573. DOI: 10.1016/J.SAB.2005.10.011 | SA coefficient formula `SA = (Δλ_obs/Δλ_0)^(-1/0.46)`. |
| 7 | Ciucci, Corsi, Palleschi, Rastelli, Salvetti, Tognoni 1999 *Appl. Spectrosc.* 53, 960. DOI: 10.1366/0003702991947612 | Saha-correction mapping (ion → neutral plane). |
| 8 | Konjević, Lesage, Fuhr, Wiese 2002 *JPCRD* 31, 819. DOI: 10.1063/1.1486456 | Experimental Stark widths critical compilation. |
| 9 | Lesage 2009 *JPCRD* 38, 761. DOI: 10.1063/1.3132702 | 2001–2007 Stark width update. |
| 10 | Aragón, Pellé & Aguilera 2011 *Anal. Bioanal. Chem.* 400, 3331 | Quantified element misidentification from Stark shifts (130 pm). |
| 11 | Gajarska et al. 2024 *J. Anal. At. Spectrom.* DOI: 10.1039/D4JA00247D | Modern reproduction in automated LIBS pipelines. |
| 12 | Aragón & Aguilera 2014 *Spectrochim. Acta B* 100, 104 | 36 Fe II + 27 Ni II Stark widths measured by LIBS in fused glass. |
| 13 | Aguilera & Aragón 2004 *Spectrochim. Acta B* 59, 1861 | Multi-element Saha-Boltzmann common-slope plot. |
| 14 | Irwin 1981 *ApJS* 45, 621 | Polynomial partition functions (now deprecated). |
| 15 | Halenka & Madej 2002 astro-ph/0204384 | Fe I-X partition functions including auto-ionizing levels. |
| 16 | Barklem & Collet 2016 *A&A* 588, A96. arXiv:1602.03304 | Modern partition-function compilation (recommended replacement). |
| 17 | Stewart & Pyatt 1966 *ApJ* 144, 1203 | Ionization-potential depression formula. |
| 18 | Mihalas 1978 *Stellar Atmospheres* 2nd ed., §9.4 | IPD textbook reference. |
| 19 | Hermann et al. 2014 *Spectrochim. Acta B* 100, 189 | Two-region plasma model. |
| 20 | Rezaei, Cristoforetti, Tognoni, Legnaioli, Palleschi, Safi 2020 *Spectrochim. Acta B* 169, 105878 | Current SA review. |
| 21 | Poggialini, Campanella, Cocciaro, Lorenzetti, Palleschi, Legnaioli 2023 *JAAS* DOI: 10.1039/d3ja00130j | "Catching up on CF-LIBS." |
| 22 | John & Anoop 2023 *RSC Adv.* 13, 29613 | Mg-Ca 50:50 simulated test of SA correction. |
| 23 | Robertson 2017 *PASA* 34, e035. arXiv:1707.06455 | Pixel-per-FWHM sampling requirement. |
| 24 | Magnier et al. 2025 arXiv:2501.17163 | Undersampling-induced flux bias quantitation. |
| 25 | Gornushkin et al. 1999 *Spectrochim. Acta B* 54, 491 | LIBS continuum (bremsstrahlung + free-bound). |
| 26 | Demidov et al. 2022 PMC9573556 | Monte Carlo CF-LIBS at 84 pix/nm (~4.2 px/FWHM). |
| 27 | von Toussaint 2018 arXiv:1805.08301 | Round-trip-only validation as known anti-pattern. |
| 28 | Anderson, Forni, Cousin, Wiens, Clegg et al. 2021 *Spectrochim. Acta B* 188, 106347 | SuperCam LIBS post-landing element quantification. |
| 29 | Clegg, Wiens, Anderson, Forni, Frydenvang, Lasue et al. 2017 *Spectrochim. Acta B* 129, 64 | MSL ChemCam recalibration with 408-sample expansion. |
| 30 | Petras 2009 arXiv:0909.0329; Graham et al. 2024 arXiv:2403.11374 | Constrained LHS / QMC on simplex. |

---

## 6. Open empirical questions

These cannot be resolved by literature alone; they require running the pipeline with specific fixes in place and measuring:

1. After Wave-1 fixes (A1, A2, B4, D1, D3), what is the actual macro_F1 lift? Each component has a literature-justified estimate; the **interaction** is unknown.
2. After A3 + B1, does the consensus-voter weighting still need adjustment? With NNLS errors reduced, the w=0.46 dominance may flip to a useful contribution.
3. Is the continuum (D2) actually a major contributor once D1 is fixed, or is it small enough to defer indefinitely? Measure residuals after D1.
4. Does the LTE-validator ΔE issue actually fire on the n=24 cohort? Some/all of the benchmark spectra may be deep-LTE and unaffected. Empirical check needed.

---

**Audit completed by:** Phase 1 — 6 parallel area-specialist subagents (general-purpose). Phase 2 — 4 parallel literature-validation subagents using Asta literature search + targeted web/PDF retrieval. All quantitative claims are citation-grounded.

**Recommendation for next session:** start with Wave-1 fixes A1/D3/D1, run synthetic regression, compare against Phase-7 baseline (alias_v2 macro_F1 = 0.402). Then plan Wave 2.
