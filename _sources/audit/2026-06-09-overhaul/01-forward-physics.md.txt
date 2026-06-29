# Forward-Model Physics Audit — CF-LIBS

**Date:** 2026-06-09 · **Scope:** `cflibs/plasma/`, `cflibs/radiation/`, `cflibs/instrument/`, `cflibs/core/constants.py`, `ASD_da/libs_production.db`
**Context:** ps-LIBS instrument (1 ps, 1040 nm Yb:fiber); T ≈ 0.5–1.3 eV, n_e ≈ 1e16–1e18 cm⁻³. Read-only audit; all numeric spot-checks reproduced below were run against the production DB with the installed package (JAX_PLATFORMS=cpu).

Findings are **ranked by expected impact on composition accuracy** (BHVO-2 RMSE 4.03 wt%, Fe over-attributed 3–4×, Al 19 vs 7.1, Ti over).

---

## F1 — CRITICAL: Legacy `SpectrumModel` silently drops ~98 % of spectral lines (float-equality population keys)

**What the code does.** The legacy forward path matches each transition to its upper-level population via a dict key containing a *floating-point energy*:

- `cflibs/radiation/emissivity.py:74` — `key = (trans.element, trans.ionization_stage, round(trans.E_k_ev, 8))`
- `cflibs/radiation/spectrum_model.py:248` — `_build_n_upper_per_line` uses the same `round(E_k, 8)` key
- `cflibs/plasma/saha_boltzmann.py:364,788` — populations dict is keyed by the **raw** `level.energy_ev` from the `energy_levels` table.

The `lines.ek_ev` column is derived from NIST `Ek(cm-1) × CM_TO_EV` (`datagen_v2.py:351`), while `energy_levels.energy_ev` was scraped in eV units directly. The two encodings of the *same physical level* differ by ~1e-7 eV (median 1.39e-7, max 9.3e-7 eV measured), but the key requires equality at 1e-8 eV. **The lookup therefore fails for almost every line, and the line is silently dropped (`n_upper = 0`).**

**Measured impact** (production DB, T = 0.8 eV, n_e = 1e17):

- Fe/Ca/Al/Ti snapshot, 240–850 nm: **110 of 6127 lines (1.8 %) matched**; all 6017 unmatched lines sit within 1e-3 eV of an existing level (i.e., they are the same level, encoded differently).
- End-to-end `SpectrumModel.compute_spectrum()`, Fe only, 370–410 nm: **2 of 412 lines populated**; the emitted spectrum contains 84/4001 non-zero grid points.

**Blast radius.** `SpectrumModel` is the forward model behind:
- the CLI `cflibs forward` (`cflibs/cli/main.py:125`),
- NIST-parity validation (`scripts/validate_nist_parity.py:91`),
- synthetic benchmark corpora (`cflibs/benchmark/corpus.py:341`, `cflibs/benchmark/synthetic.py:493`).

The manifold/Bayesian snapshot path (`cflibs/radiation/kernels.py:283` `_saha_two_stage_populations`) is **unaffected** — it computes Boltzmann factors directly from `line_E_k_ev` without key matching. Note the kernel-vs-legacy "parity tests (rtol 1e-5)" cited in `spectrum_model.py:358-368` pass *because both paths share the injected `n_upper` array* — parity holds while both are wrong.

**Why this matters for composition.** Every synthetic-corpus benchmark, every element-ID F1 number, and every NIST-parity claim derived from `SpectrumModel` spectra was computed on spectra missing ~98 % of lines, with survivors selected essentially at random (whichever float encodings collide). Inversion components tuned against those corpora inherit the bias.

**Literature anchor.** Not a literature question — this is a software defect. The LTE line emissivity itself (ε = hc/4πλ · A_ki · n_k, `emissivity.py:53`) is the standard expression (e.g., De Giacomo & Hermann 2017, *J. Phys. D* 50, 183002, DOI 10.1088/1361-6463/aa6585, §2).

**Recommended fix.** Stop keying on floats. Either (a) join lines→levels by index at snapshot build time (tolerance ≤ 1e-4 eV) and key populations by level index, or (b) retire the legacy population-dict path entirely and route `SpectrumModel` through the snapshot Saha path with the CPU direct-sum U providers. Add a regression test asserting ≥ 99 % of snapshot lines receive non-zero `n_upper` for an Fe plasma.

---

## F2 — HIGH: Self-absorption is OFF in every inversion-facing forward model; the only path that has it computes optical depth with the wrong line width

**What the code does.**
- The unified kernel supports slab radiative transfer `I = B_λ(T)·(1 − e^{−κλ·L})` with `κλ = ελ/Bλ` (Kirchhoff, correct under LTE): `cflibs/radiation/kernels.py:515-528`.
- But `apply_self_absorption` defaults to **False** (`kernels.py:607`), and the inversion-facing callers keep it off: manifold (`cflibs/manifold/batch_forward.py:566`), Bayesian (`cflibs/inversion/solve/bayesian/forward.py:145`, `path_length_m = 0.0`). The kernel docstring itself says "manifold and Bayesian wrappers leave it False (thin plasma)" (`kernels.py:639-641`).
- The one caller that turns it on — `SpectrumModel` (`spectrum_model.py:298`) — computes the optical depth from a *mis-broadened* emissivity:
  - **LEGACY mode** (the constructor default, `spectrum_model.py:87`, and what the CLI gets): all lines share the ad-hoc scalar σ = 0.01·√(T/0.86) nm (`kernels.py:553`) ≈ 9.6 pm at 0.8 eV — ~6× wider than the Fe I Doppler σ (≈1.6 pm at 400 nm). Peak κ, hence self-absorption, is underestimated by roughly that factor.
  - **NIST_PARITY mode**: instrument σ is folded into the per-line width *before* the RT step (`spectrum_model.py:299` sets `fold_instrument_sigma=(mode==NIST_PARITY)`), so optical depth is computed on the instrument-broadened profile. With R ~ 3000 (FWHM 0.13 nm at 400 nm vs physical ~0.01 nm) peak κ is ~10× low. Physically, absorption happens at the *physical* profile; instrument convolution must come after RT. Only PHYSICAL_DOPPLER(+Stark) mode does the ordering correctly.

**What the literature says.** Self-absorption of strong/resonance lines is the dominant systematic in CF-LIBS closure and must be modeled or corrected: Bulajic et al. 2002, *Spectrochim. Acta B* 57, 339 (recursive curve-of-growth correction inside the CF-LIBS loop); El Sherbini et al. 2005, *Spectrochim. Acta B* 60 (self-absorption coefficients of Al I 394.4/396.15 — directly relevant to the Al over-estimate); Aragón & Aguilera, "CSigma graphs", *J. Quant. Spectrosc. Radiat. Transf.* 149 (2014) 90 (curve-of-growth generalization for optically thick LIP). Self-absorption is *not* negligible in ps plasmas: "Self-absorption of emission lines in picosecond-laser-produced gold plasmas", *Phys. Plasmas* 31, 043302 (2024).

**Impact estimate.** In BHVO-2 basalt at 0.8 eV/1e17, the strong Ca II 393.4/396.8, Mg II 279.5/280.3, Al I 394.4/396.15, Na I 589.0/589.6 lines are the canonical saturated lines (SA factors 1.5–5 in the LIBS literature above). Fitting optically-thin templates (manifold/Bayesian) to saturated lines forces the fit to *under-credit* the elements carrying those lines (Ca, Mg, Na, Al majors) and, through Σ C = 1 closure, redistribute weight onto elements whose intensity is spread over many weak unsaturated lines — **Fe and Ti**, exactly the observed over-attribution pattern.

**Recommended fix.** (1) Turn on the already-implemented slab RT in the manifold/Bayesian forward models (flip the flag, supply `path_length_m` ~ 0.5–2 mm, optionally as a fit parameter); (2) in `SpectrumModel`, force RT to run on the physical (Doppler+Stark Voigt) profile and apply instrument convolution strictly afterwards (the PHYSICAL_DOPPLER ordering), for all modes; (3) alternatively/additionally, weight down or exclude known resonance lines in the likelihood (the `is_resonance` flag already exists in the DB).

---

## F3 — HIGH: Neutral-species partition functions are 5–40 % low for Rydberg-rich neutrals (incomplete level lists in the DB)

**What the code does.** U(T) is computed by direct summation over `energy_levels` with an IP cutoff (`cflibs/plasma/partition.py:115-164`), exactly the method recommended by Alimohamadi & Ferland 2022 (*PASP* 134, DOI 10.1088/1538-3873/ac7664; arXiv:2203.02188). The JAX paths consume a guarded 4th-order ln-poly fit of the same sum (`partition.py:360-447`); measured fit-vs-sum bias is ≤ 2 % in the LIBS band (Fe I +0.8 %, Al I +2.1 % — the deliberate "lift" never undershoots). The polynomial machinery itself is sound.

**The problem is the level data**, scraped by `datagen_v2.py:167` (`fetch_energy_levels`) with a fragile text parser. Spot-check of the direct sums against Barklem & Collet 2016 (*A&A* 588, A96, DOI 10.1051/0004-6361/201526961; `table8_vNov2022.dat` downloaded from the authors' repository and interpolated):

| Species | U_BC(9284 K) | U_DB(9284 K) | DB/BC | U_BC(1e4 K) | U_DB(1e4 K) | DB/BC |
|---|---|---|---|---|---|---|
| Fe I | 53.03 | 52.29 | **0.986** | 59.66 | 58.63 | 0.983 |
| Fe II | 63.02 | 62.96 | **0.999** | 66.90 | 66.88 | 1.000 |
| Ti I | 72.39 | 66.89 | 0.924 | 83.20 | 75.60 | 0.909 |
| Ti II | 79.65 | 79.53 | 0.999 | 83.72 | 83.55 | 0.998 |
| Ca I | 4.35 | 3.25 | **0.747** | 5.70 | 3.93 | 0.690 |
| Ca II | 3.32 | 3.32 | 0.999 | 3.56 | 3.56 | 0.999 |
| Al I | 6.67 | 6.24 | 0.936 | 7.05 | 6.41 | 0.909 |
| Mg I | 1.45 | 1.37 | 0.945 | 1.64 | 1.51 | 0.919 |
| Na I | 4.66 | 3.26 | **0.698** | 5.85 | 3.76 | 0.643 |
| K I | 7.72 | 4.66 | **0.604** | 9.82 | 5.50 | 0.560 |
| Si I | 11.01 | 10.89 | 0.989 | 11.36 | 11.14 | 0.981 |
| Cr I | 27.93 | 26.53 | 0.950 | 33.18 | 31.23 | 0.941 |
| Mn I | 13.08 | 12.66 | 0.968 | 15.35 | 14.75 | 0.961 |
| O I | 9.33 | 9.33 | 1.000 | 9.42 | 9.42 | 1.000 |

Low-lying levels are scraped correctly (Fe I a⁵D ladder, Ca II 3d ²D, Al I ²P° verified against NIST values exactly), so the deficits come from **missing high-lying/Rydberg levels**: Ca I has 76 levels in the DB vs ~200 in NIST; Ti II stops at 8.57 eV though IP = 13.57 eV (`SELECT COUNT(*) ... energy_ev>=9` → 0 rows).

**Caveat (works in the code's favor):** in a real n_e=1e17 plasma the IPD-truncated sum is the physical one (Alimohamadi & Ferland 2022), so the "true" U lies *below* the full B&C value — but Δχ = 0.066 eV removes only states within 0.066 eV of the IP, which is far less than the 25–40 % deficits measured for Ca I/Na I/K I.

**Why this matters for composition.** CF-LIBS closure scales each element by U_s(T) (C_s ∝ U_s·e^{q_s}); U(Ca I) low by 25 % ⇒ Ca concentration low by 25 % before closure; after Σ C = 1 normalization the deficit redistributes onto Fe/Ti/Si. The Saha ratio also inherits U_II/U_I (+34 % for Ca), over-ionizing the prediction. Fe I/Fe II themselves are accurate to ≤ 1.7 % — **the Fe over-attribution is not a U(Fe) error; it is partly everyone else's U being too small.** Direct effect on basalt majors: Ca (−25 %), Na (−30 %), K (−40 %), Mg (−5 %), Al (−6 %), Ti (−8 %) — several wt% of redistribution, a material slice of the 4.03 wt% RMSE.

**Recommended fix.** Re-ingest complete level lists (NIST ASD levels CSV download rather than the regex scrape in `datagen_v2.py:167-220`), or ingest Barklem & Collet table 8 / Irwin 1981 (*ApJS* 45, 621) polynomials as the stored fallback, keeping the run-time IPD truncation. Re-run the F3 table afterwards as the acceptance test.

---

## F4 — MEDIUM-HIGH: Snapshot Saha path is two-stage only, with no IPD — inconsistent with the CPU solver and the inversion

**What the code does.** `_saha_two_stage_populations` (`cflibs/radiation/kernels.py:283-377`) computes `ratio = S·(U_II/U_I)·exp(−ip_I/kT)` with the **raw** ionization potential (`kernels.py:347-356`) and normalizes `frac_I + frac_II = 1` (`kernels.py:357-358`). The CPU solver (`cflibs/plasma/saha_boltzmann.py:94-158`) applies Debye–Hückel IPD (`eff_ip = ip − Δχ`) and includes stage III. The inversion also applies IPD (`cflibs/inversion/solve/iterative.py:1226-1229`, `closed_form.py:459-463`).

**Quantified impact (measured with the CPU solver on the production DB):**
- *IPD inconsistency:* Δχ(1e17 cm⁻³, 10⁴ K) = 0.066 eV (canonical Gaussian-CGS Debye–Hückel, `partition.py:76-112`; magnitude consistent with the Stewart & Pyatt 1966, *ApJ* 144, 1203 scale at these densities, and with the "few × 0.01 eV" expectation). exp(Δχ/kT) = **1.089** at 0.8 eV ⇒ the JAX forward's ion/neutral ratio is ~9 % lower than the CPU/inversion convention; neutral-line intensities differ ~6–7 % between paths (f_I = 0.256 for Fe at 0.8 eV/1e17).
- *Missing stage III:* fractions at T = 1.3 eV, n_e = 1e17 (CPU 3-stage solver): Ca f₃ = 0.65, Mg f₃ = 0.28, Ti f₃ = 0.11, Al f₃ = 0.07, Fe f₃ = 0.03. The two-stage kernel reassigns that population to stage II — Ca II line intensities inflated ×2.9 at the hot edge of the manifold grid (at 0.8 eV the error is ≤ 5 %, mostly Ca/Ba). Manifold grids and Bayesian priors that extend to 1.3 eV (the stated ps-LIBS band) are biased at the hot end.
- *`N_total = n_e` proxy* (`kernels.py:299-302,375`): cancels in relative composition (common scale factor) **only while the model is optically thin**; if F2's fix turns on slab RT, the absolute emissivity scale matters and this proxy must be replaced by the closure/pressure-consistent total density (the hook `total_species_density_cm3` already exists, `kernels.py:610`).

**Recommended fix.** Add stage-III terms and the Δχ-lowered exponent to `_saha_two_stage_populations` (the snapshot already carries `ionization_potential_ev`; Δχ(n_e,T) is one extra traced scalar). Acceptance: kernel vs `SahaBoltzmannSolver` ionization fractions agree to <1 % over T ∈ [0.5, 1.3] eV, n_e ∈ [1e16, 1e18].

---

## F5 — MEDIUM: Stark widths are λ²-scaled heuristics for 80 % of lines; forward Voigt and n_e diagnostic are at least mutually consistent

**What the code does.**
- Profiles: full Voigt via `scipy.special.wofz` / Weideman-1994 Faddeeva on the JAX path (`cflibs/radiation/profiles.py:261-328, 553-626`) — not Gaussian-only. Stark enters as a per-line Lorentzian HWHM γ = ½·w_ref·(n_e/1e17)·(T/T_ref)^(−α) (`kernels.py:410-449`), with the stored `stark_w` defined as electron-impact **FWHM at n_e = 1e17 cm⁻³, T = 10⁴ K** (`cflibs/radiation/stark.py:16-40`). Stark shift is applied linearly in n_e to profile centers (`kernels.py:452-479`). Ion broadening (Griem A-term) is deliberately dropped as <2–5 % at ps-LIBS densities (`kernels.py:463-466`).
- The inversion's n_e diagnostic inverts the *same* law after Voigt deconvolution of instrument+Doppler (Olivero & Longbothum 1977 inversion, `stark.py:143-273`) — forward and inverse are consistent by construction. Good.
- **Provenance, measured from the DB:** of 28 727 lines, `stark_w_source` = `konjevic_lambda_sq_scaled` for **22 951 (80 %)**, `interpolated` 4 574, `hydrogenic` 562, actual `stark_b` literature values 244 (`scripts/ingest_stark_b.py` header documents the scheme and cites Konjević, Lesage, Fuhr & Wiese 2002, *J. Phys. Chem. Ref. Data* 31, 819; Sahal-Bréchot, Dimitrijević & Ben Nessib 2014, *Atoms* 2, 225 / STARK-B).

**What the literature says.** Gigosos 2014 (*J. Phys. D* 47, 343001, DOI 10.1088/0022-3727/47/34/343001) reviews Stark models: even *measured* widths carry 15–50 % uncertainties (Konjević et al. 2002 grades), and λ²-scaling between transitions is an order-of-magnitude heuristic, not a model. Lines used as n_e diagnostics should come from the 244-line literature set, never from the scaled bulk.

**Impact.** Line *wings* and blends in the forward fit are wrong wherever γ is heuristic (most lines); the Stark-based n_e from a heuristic-width line is essentially unconstrained (error ∝ 1/w_ref). Composition impact is second-order (via blends and via n_e → Saha), but n_e error of ×2 shifts ion/neutral ratios of all elements coherently.

**Minor inconsistency:** missing-α default is 0.5 on the scalar path (`stark.py:103`) but 0.0 in the snapshot kernel (`kernels.py:421-424` — "snapshot default for missing DB entries").

**Recommended fix.** Restrict the n_e diagnostic to `stark_w_source IN ('stark_b','interpolated')` lines (provenance column already exists); harmonize the missing-α default; long-term, extend the STARK-B ingestion coverage for the basalt-relevant species (Fe I/II, Ca I/II, Mg I/II, Si I, Ti II).

---

## F6 — MEDIUM: No continuum emission model (free–free / free–bound)

**What the code does.** There is no Bremsstrahlung or recombination continuum anywhere in `cflibs/radiation/` (grep over the package: the only "continuum" hits are baseline-*removal* docstrings in `cflibs/inversion/preprocess/preprocessing.py`). `planck_radiance` (`spectrum_model.py:50-63`) is used only as the RT source function. Forward spectra are pure line spectra; measured spectra are baseline-subtracted before inversion; the Bayesian model absorbs residual continuum with a Chebyshev baseline.

**What the literature says.** LIP continuum = free–free + free–bound, scaling ∝ n_e²/√T with exponential cutoffs (De Giacomo & Hermann 2017, *J. Phys. D* 50, 183002, §2). For **ps-LIBS this omission is largely defensible**: shorter pulses substantially reduce continuum (avalanche-ionization limited), and gated ps-LIBS spectra are line-dominated after the first few ns (Gragston et al. 2020, *Appl. Spectrosc.* 74(3), "Time-Gated Single-Shot Picosecond LIBS", DOI 10.1177/0003702819885647).

**Impact.** Low for gated ps data. Residual risk: baseline subtraction under dense Fe line forests (UV) eats real line wings, biasing Fe line intensities downward and the recovered baseline upward — an inversion-side issue, not forward. If early-delay (<10 ns) ungated spectra are ever fit, add an n_e²-scaled ff+fb term.

**Recommended fix.** Document the assumption; optionally add the standard ff+fb continuum (two analytic terms, jit-friendly) behind a flag for early-delay work.

---

## F7 — LOW-MEDIUM: McWhirter implemented correctly but is the only *enforced* criterion; temporal check exists, off by default

**What the code does.** `n_e ≥ 1.6×10¹² √T (ΔE)³` (`cflibs/core/constants.py:80`, `cflibs/plasma/lte_validator.py:126`) — correct constant and form. The module explicitly cites Cristoforetti et al. 2010 (*Spectrochim. Acta B* 65, 86–95, "LTE in LIBS: Beyond the McWhirter criterion") and labels McWhirter "necessary, not sufficient" (`lte_validator.py:83-97`). A Spitzer-style temporal relaxation check exists but `check_temporal=False` by default (`lte_validator.py:289`), and no spatial-gradient (diffusion-length) check exists at all — Cristoforetti 2010 requires both for transient, inhomogeneous LIP.
ΔE selection (`lte_validator.py:215-248`) uses max(E_k) of observed lines as the gap bound — conservative (over-strict), acceptable.

**Impact.** Indirect: results carry an LTE quality flag that is too permissive for fast-evolving ps plasmas (n_e decays orders of magnitude in tens of ns). At 0.8 eV, ΔE = 3 eV: n_e,req = 4.2e15 cm⁻³ — most of the stated band passes McWhirter, so the *binding* criteria are precisely the temporal/spatial ones that are off/missing.

**Recommended fix.** Enable the temporal check by default with the gate delay/width as `plasma_lifetime_ns`; add the Cristoforetti spatial criterion (variation length vs diffusion length) as a third check.

---

## F8 — LOW: Line-list completeness filters on the forward path

- `SpectrumModel.compute_spectrum` filters snapshot lines by NIST `rel_int >= 10.0` (LEGACY/PHYSICAL/LDM) or `>= 0.01` (NIST_PARITY) (`spectrum_model.py:378`; SQL floor at `cflibs/atomic/database.py:443-445`). The DB has **4 501 lines with `rel_int = 0`** (blank NIST intensity, `fillna(0)` at `datagen_v2.py:354`) and 9 758 with `rel_int < 50`; any threshold > 0 silently removes lines that have perfectly valid A_ki but no NIST intensity entry. The database docstring itself warns `rel_int` is not cross-spectrum comparable (`database.py:375-379`). **Fix:** filter on *predicted* emissivity at nominal plasma conditions, not on NIST `rel_int`.
- Harvest keeps only lines with `obs_wl_air` present (`datagen_v2.py:327`), i.e., the NIST **air-wavelength** convention (air for 200–2000 nm; vacuum outside — NIST ASD Lines help). The 190–200 nm window requested at `datagen_v2.py:321` is therefore empty (vacuum-λ lines dropped). Irrelevant above 200 nm; relevant if a VUV-capable spectrometer is ever used.

---

## F9 — LOW: Constants, units, and conventions (audited, mostly clean)

| Item | Code | Verdict |
|---|---|---|
| Saha prefactor | `SAHA_CONST_CM3 = 6.042e21` (`constants.py:76`) | Exact value 2·(2π m_e e/h²)^{3/2}·1e-6 = **6.0371e21** → code is +0.08 %. Negligible; correct the literal anyway. Electron-spin factor 2 correctly included once (documented `constants.py:72-75`); kernel and CPU solver both use U_II/U_I without an extra 2. |
| Emissivity prefactor | ε = hc/4πλ·A_ki·n_k (`emissivity.py:53`, `kernels.py:696-702`) | Correct (no missing 4π/λ factors); cm⁻³→m⁻³ conversion explicit; output W m⁻³ nm⁻¹ sr⁻¹ then W m⁻² nm⁻¹ sr⁻¹ after slab RT. |
| Doppler width | σ = λ√(kT/mc²) (`profiles.py:231`, `kernels.py:385-389`) | Correct Gaussian σ (the historical √2 error is documented as removed). Uses T_e for heavy particles — fine under LTE, document. |
| Voigt | wofz/Weideman, no wing clipping (`profiles.py:271-310`) | Correct; FWHM via Olivero & Longbothum 1977 (`profiles.py:237-258`). |
| IPD | Gaussian-CGS Debye–Hückel, single canonical implementation (`partition.py:76-112`) | 0.021/0.066/0.209 eV at n_e = 1e16/1e17/1e18, T = 1e4 K (measured) — the expected "few × 0.01 eV" scale; Saha exponent and partition cutoff use the same Δχ on the CPU path (good); JAX kernel omits it (see F4). |
| Air vs vacuum λ | DB stores NIST `obs_wl_air` exclusively; all model λ are air | Internally consistent. Photon energy uses hc/λ_air → +0.028 % energy error (n_air ≈ 1.00028); negligible. |
| Planck/RT | `B(1−e^{−κL})`, κ = ε/B (`kernels.py:515-528`) | Correct LTE slab with stimulated emission implicit in Kirchhoff's law. |
| LEGACY broadening σ = 0.01·√(T/0.86) nm | `kernels.py:553` | Unphysical (mass/λ-independent); LEGACY is the `SpectrumModel` **default** (`spectrum_model.py:87`) and the CLI does not expose the mode. Default should be PHYSICAL_DOPPLER + Stark (manifold already defaults to it, `cflibs/manifold/config.py:131-132`). |

---

## Priority summary (expected composition-accuracy leverage)

1. **F1** — fix the float-key population matching (legacy path emits ~2 % of its lines). Invalidates SpectrumModel-derived corpora/validation until fixed.
2. **F2** — model self-absorption in the inversion-facing forward models (flags already exist); fix RT-vs-instrument ordering in NIST_PARITY/LEGACY. Directly addresses the Fe/Ti over- vs Ca/Mg/Na/Al under-attribution mechanism.
3. **F3** — re-ingest complete energy levels (Ca I −25 %, Na I −30 %, K I −40 %, Ti I −8 %, Al I −6 % partition-function deficits vs Barklem & Collet 2016 redistribute several wt% onto Fe via closure).
4. **F4** — add stage III + IPD to the snapshot Saha kernel (×2.9 Ca II error at 1.3 eV; 9 % forward/inversion ion-ratio inconsistency).
5. **F5–F9** — Stark provenance gating for n_e, continuum documentation, LTE temporal/spatial checks, rel_int filter, constants polish.

## References (located/verified during this audit)

- Barklem, P. S. & Collet, R. (2016), *A&A* 588, A96, DOI 10.1051/0004-6361/201526961 — atomic partition functions; data file `table8_vNov2022.dat` from the authors' public repository (github.com/barklem/public-data) used for the F3 table.
- Irwin, A. W. (1981), *ApJS* 45, 621 — polynomial partition-function form (basis convention handled correctly in `partition.py:851-879`).
- Alimohamadi, P. & Ferland, G. J. (2022), *PASP* 134, DOI 10.1088/1538-3873/ac7664 (arXiv:2203.02188) — direct summation + continuum-lowering truncation.
- Cristoforetti, G. et al. (2010), *Spectrochim. Acta B* 65, 86–95 — LTE beyond McWhirter (temporal/spatial criteria).
- McWhirter, R. W. P. (1965), in *Plasma Diagnostic Techniques* (Huddlestone & Leonard, eds.).
- Gigosos, M. A. (2014), *J. Phys. D: Appl. Phys.* 47, 343001, DOI 10.1088/0022-3727/47/34/343001 — Stark broadening models for plasma diagnostics.
- Konjević, N., Lesage, A., Fuhr, J. R. & Wiese, W. L. (2002), *J. Phys. Chem. Ref. Data* 31, 819–927 — critical Stark width compilation.
- Sahal-Bréchot, S., Dimitrijević, M. S. & Ben Nessib, N. (2014), *Atoms* 2, 225 — STARK-B database/SCP method.
- Bulajic, D. et al. (2002), *Spectrochim. Acta B* 57, 339 (ADS 2002AcSpB..57..339B) — self-absorption correction procedure for CF-LIBS.
- El Sherbini, A. M. et al. (2005), *Spectrochim. Acta B* 60 — self-absorption coefficients of Al I lines in LIBS.
- Aragón, C. & Aguilera, J. A. (2014), *J. Quant. Spectrosc. Radiat. Transf.* 149, 90 — CSigma graphs (curve-of-growth for optically thick LIP); also Aragón & Aguilera (2008), *Spectrochim. Acta B* 63, 893 (cited in code at `spectrum_model.py:125-128`).
- De Giacomo, A. & Hermann, J. (2017), *J. Phys. D: Appl. Phys.* 50, 183002, DOI 10.1088/1361-6463/aa6585 — LIP emission review (line + continuum mechanisms).
- Stewart, J. C. & Pyatt, K. D. (1966), *ApJ* 144, 1203 — ionization potential depression.
- *Phys. Plasmas* 31, 043302 (2024) — self-absorption of emission lines in picosecond-laser-produced gold plasmas.
- Gragston, M. et al. (2020), *Appl. Spectrosc.* 74, DOI 10.1177/0003702819885647 — time-gated single-shot ps-LIBS (reduced continuum).
- Olivero, J. J. & Longbothum, R. L. (1977), *JQSRT* 17, 233 — Voigt FWHM approximation (used in `profiles.py` and `stark.py`).
- NIST ASD Lines help (physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html) — air wavelengths for 5 000 < σ < 50 000 cm⁻¹ (≈200–2000 nm), vacuum outside.
