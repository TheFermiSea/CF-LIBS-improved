# CF-LIBS Physics + Algorithm Audit — Wave 2 (2026-06-05)

**Status:** Deep literature-grounded audit complete. 9 topics audited end-to-end — 6 in the main run (46 verified findings after 3-skeptic adversarial review) + 3 (boltzmann-saha, bayesian, manifold) in a follow-up run (15 new findings after embedded self-refutation). **Total: 61 findings, 15 defect families.** The follow-up section is at the end of this document (§8+); it carries the consolidated critical path that supersedes §3 for execution ordering.

**Workflow (main):** `cflibs-deep-physics-audit-wave2` (Run ID: wf_aa52e31d-311). 170 agents, 8.8M subagent tokens, 1264 tool calls.

**Workflow (follow-up):** `cflibs-physics-audit-wave2-followup` (Run ID: wf_0bada284-072). 10 agents, 616K tokens, 137 tool calls. Re-audited the 3 topics whose pipeline stalled in the main run (root cause: `mcp__gpd-arxiv` permission hang in headless workflow — now fixed via settings allowlist). Used embedded self-refutation instead of an external verify stage; deduplicated against the main run's 8 families.

**Predecessor:** `docs/architecture/2026-05-27-physics-audit.md` (Wave 1). Wave-1 fixes landed in PR #212: A1 (Stark attribute typo), A2 (apply_stark default True), D1 (manifold 18432 px + Nyquist guard), D3 (instrument_fwhm plumbing), B4 (self-absorption τ recompute).

**Headline:** The Wave-1 fixes made several latent bugs reachable for the first time. Wave-2 audit found that the Stark broadening subsystem suffers from a multi-layered convention cascade producing ~20× over-broadening at typical LIBS densities; the self-absorption / CDSB / COG modules are dead code containing serial physics defects; the iterative solver uses isobaric pressure balance instead of Stark broadening for n_e (wrong diagnostic per Tognoni 2010); a Doppler `sqrt(2)` factor-of-2 bug recurs at four sites; and SpectrumModelJax silently drops Stark entirely. None of these were caught by existing tests because test fixtures often re-encode the same buggy formulae they are meant to validate.

---

## 1. Workflow Coverage

| Topic | Status | Verified findings |
|-------|--------|-------------------|
| peak-id | ✅ complete | 8 |
| boltzmann-saha | ⚠ pipeline failed (lit/verify stalled) | 0 (not surfaced; affected files referenced in adjacent topics) |
| self-absorption | ✅ complete | 7 |
| stark | ✅ complete | 8 |
| closure | ✅ complete | 7 |
| iterative-solver | ✅ complete | 9 |
| bayesian | ⚠ pipeline failed (lit/verify stalled) | 0 (not surfaced; affected files referenced in adjacent topics) |
| manifold | ⚠ pipeline failed (lit/verify stalled) | 0 (not surfaced; affected files referenced in adjacent topics) |
| uncertainty | ✅ complete | 7 |

**Partial-failure topics need a follow-up audit:**
- `boltzmann-saha` — literature agent could not return structured output after 2 nudges
- `bayesian` — verify-stage skeptic agents stalled on all 6 attempts
- `manifold` — verify-stage skeptic agents stalled on all 6 attempts

These three topics ARE still represented in the cross-cutting synthesis via findings from adjacent topics that touch the same files (e.g., Stark family includes manifold + bayesian file references; Self-absorption family includes bayesian).

---

## 2. Defect families (8)

### CRITICAL — Stark broadening convention/units cascade (HWHM/FWHM, n_e ref, ion-broadening, T-exponent semantic conflation)

**Topics affected:** stark, manifold, bayesian, peak-id  
**Expected total lift:** +0.10 to +0.30 (largest single family; Stark dominates Lorentzian width at LIBS n_e, and 20× over-broadening corrupts both peak ID and line fitting)  

**Root cause:**

The Stark subsystem has four compounding convention bugs that interact multiplicatively: (1) populate_stark_widths.py writes FWHM at n_e=1e17 cm^-3 (Konjević/STARK-B convention) but kernels.py reads it as HWHM at n_e=1e16 cm^-3, producing ~20x over-broadening at typical LIBS densities; (2) the stark_alpha column conflates the dimensionless Griem ion-broadening parameter A (~0.04-0.20) with the Konjević T-exponent (~0.3-1.0) — populate-script writes the former, runtime kernels.py treats it as the latter, giving an essentially identity factor_T at ps-LIBS T; (3) the Griem ion-broadening correction [1 + 1.75 A (1 - 0.75 R)] exists in stark.py but is never called from the production Voigt kernel — AtomicSnapshot has no line_stark_A field; (4) where the correction does exist, R_D is hardcoded at 0.5 with no T or n_e dependence, valid only at a single point. Additionally, SpectrumModelJax silently drops Stark Lorentzian entirely (apply_stark stored, never consulted; only Gaussian kernel called), and Stark shifts (Δλ ~ 5-50 pm at n_e=1e17) are computed but never applied to line centers. The Wave-1 A1/A2 fixes (attribute typo + apply_stark default) only made these latent bugs reachable.

**Findings in family:**
- stark_w stored as FWHM at n_e=1e17 but runtime treats it as HWHM at n_e=1e16 (compounded ~20× over-broadening at LIBS conditions)
- stark_alpha column conflates two physically distinct parameters (T-exponent vs Griem ion-broadening A); populate-script writes A but runtime treats it as T-exponent
- Ion-broadening correction `[1 + 1.75 A (1 - 0.75 R)]` is never applied in the production Voigt kernel — Griem ion contribution silently dropped
- Hardcoded R_D = 0.5 in ion-broadening correction is out-of-regime across the ps-LIBS plasma parameter envelope
- SpectrumModelJax silently drops Stark Lorentzian — `apply_stark` arg is accepted, stored, and never consulted
- Stark shift is computed but never applied to line centers — line identification misregistered at n_e ≥ 1e17 cm^-3
- T_eff temperature floor `max(T_K, 1000 K)` masks NaN / Inf at low T instead of failing loudly — ps-LIBS regime at T ≈ 0.5 eV (5800 K) is fine but adjacent corner cases hide bugs


### CRITICAL — Self-absorption and CDSB: dead code paths with multiple internal physics bugs

**Topics affected:** self-absorption, iterative-solver, bayesian  
**Expected total lift:** +0.05 to +0.15 (Aguilera 2014 reports 5-15% concentration accuracy improvement for resonance-rich elements; vrabel2020 soil audit named Si — a strongly self-absorbed resonance line emitter — as most-missed)  

**Root cause:**

The self-absorption / CDSB / COG infrastructure is unwired from the production inversion pipeline (Serena's find_referencing_symbols confirms zero callers in solve/, identify/, or cli/; BayesianForwardModel hardcodes apply_self_absorption=False). Underneath that absence, the modules themselves contain serial physics defects: (a) doublet-ratio uses g·A·λ³ (line strength) instead of A/λ (intensity ratio), wrong by (λ₁/λ₂)⁴; (b) CDSB initial-tau scales with n_e instead of species number density n_s, biased by ratio n_s/n_e (10-1000× in low-ionization ps-LIBS); (c) CDSB iteration's U_old == U_new identity collapses the partition-function update to a no-op, defeating the whole Cristoforetti-Tognoni recursion (especially for resonance lines where ΔE-factor also = 1); (d) CurveOfGrowth optical-depth missing one factor of λ (λ¹ vs λ², ~5 orders of magnitude tau underestimate in UV); (e) default mask_threshold=3.0 discards lines (τ≈3-5) that the literature explicitly says are recoverable; (f) ad-hoc resonance_tau_boost=1.5 double-counts the Boltzmann factor with no literature backing. Wave-1 B4 fixed the internal Bulajic τ recomputation but did not wire it in or address any other defect.

**Findings in family:**
- Self-absorption corrector and CDSB plotter are dead code in the inversion pipeline (never called from any solver)
- Doublet-ratio theoretical intensity uses wavelength^3 (line strength) instead of 1/wavelength (emission intensity); wrong by factor (lambda_1/lambda_2)^4
- CDSB initial-tau scales with electron density n_e instead of species number density n_s
- CDSB tau update across temperature iterations is a no-op for the partition-function term (U_old == U_new always), defeating Cristoforetti & Tognoni 2013 iteration
- COG _estimate_optical_depth missing one factor of wavelength: uses lambda/Delta_lambda instead of lambda^2/Delta_lambda
- Default mask_threshold=3.0 discards the very lines self-absorption correction is meant to rescue (literature uses tau<=5)
- Empirical resonance_tau_boost=1.5 in CDSB plotter is fabricated heuristic with no literature backing; doubly-counted with E_i=0 Boltzmann factor
- Iterative solver lacks resonance-line down-weighting; self-absorption-prone resonance lines enter Boltzmann fit at full weight


### HIGH — Iterative CF-LIBS solver: wrong n_e diagnostic + silent failure modes

**Topics affected:** iterative-solver, boltzmann-saha  
**Expected total lift:** +0.07 to +0.20 (Stark n_e diagnostic alone +0.05-0.15 per Tognoni 2010; silent-failure quality flags +0.01-0.03; resonance down-weighting +0.03-0.08)  

**Root cause:**

The iterative solver's n_e update path uses isobaric pressure balance with a hardcoded STP=1 atm assumption — a method neither prescribed by Tognoni 2010 nor Aragón & Aguilera 2008 (canonical literature uses Stark broadening as primary n_e diagnostic) and physically wrong for ps-LIBS where plume pressure can be 10-100× atm early or sub-atm late. Compounding this, the solver silently masks fit failures: positive Boltzmann slope clamps T to 50000 K and reports converged=True; near-zero negative slopes produce T~1e12 K with no upper-magnitude clamp; convergence check compares damped iterates so the 10% ne_tolerance hides oscillating undamped updates by a factor of 2; two_region T_corona = 0.8 * T_K is a hardcoded magic constant attributed to 'Hermann 2017' with no validated source for sub-10 ps Yb:fiber regime. The solver also has no resonance-line down-weighting despite Aragón & Aguilera 2008 §3 calling this out as standard practice.

**Findings in family:**
- Pressure-balance n_e update assumes P = 1 atm; literature canonically uses Stark broadening, not isobaric closure
- Closed-form solver's pressure-balance n_e step assumes STP_PRESSURE = 1 atm, which is wrong for ps-LIBS plasmas
- Iterative solver lacks Stark-broadening n_e diagnostic; canonical CF-LIBS literature uses it as the primary n_e source
- Solver reports converged=True for unphysical positive Boltzmann slope (clamped T=50000 K, no failure signal)
- Slightly-negative slope produces astronomical T with no clamp on lower magnitude
- Default ne_tolerance_frac=0.1 (10%) is too loose; convergence check compares damped n_e vs ne_prev so oscillations register as converged
- T_corona = 0.8 * T_K hardcoded as 'Hermann 2017' but no published two-region fit constant exists for sub-10 ps Yb:fiber regime


### HIGH — Closure-equation correctness across solver paths

**Topics affected:** closure, iterative-solver, bayesian  
**Expected total lift:** +0.03 to +0.10 (SpectralRefiner closure violation directly biases element-presence thresholds; matrix/oxide closure missing in closed-form path affects geological/metallurgical matrices)  

**Root cause:**

Multiple solver paths violate or misapply the Σ C_s = 1 closure invariant that defines CF-LIBS: (a) SpectralRefiner has no closure constraint and no post-renormalization (L-BFGS-B with box bounds [0,1] freely picks any (c, scale) pair); (b) Dirichlet-residual closure mode treats experimental factor F as a 'closure deficit' even though F has arbitrary scale per Ciucci 1999; (c) apply_matrix_mode silently returns concentrations with no |Σ C_s - 1| diagnostic when matrix_fraction is wrong; (d) closed-form ILR solver hard-codes Σ C_s = 1 with no support for matrix-element pinning or oxide closure (ChemCam/Tognoni 2010 standard); (e) plr_transform is named PLR but mathematically computes ALR (non-isometric), so PWLR L2 regularization implicitly uses the wrong metric on simplex. Compounding this, the ILR/PWLR clip-floor 1e-10 implements 'replace zero by small constant' — exactly the practice CoDA literature (Aitchison 1982, Egozcue 2003, Martín-Fernández 2003) warns against.

**Findings in family:**
- SpectralRefiner does not enforce closure (sum C = 1), violating CF-LIBS foundational invariant
- Dirichlet-residual closure equates experimental factor F with a 'closure deficit' -- diagnostic and residual fraction are not scale-invariant
- apply_matrix_mode returns concentrations with no Σ C_s = 1 enforcement and no diagnostic when result departs from simplex
- Closed-form ILR solver hard-codes Σ C_s = 1; no support for matrix-element pinning or oxide closure that the iterative solver offers
- plr_transform / plr_inverse compute ALR, not the pivot log-ratio of Hron 2012; PWLR claim of isometry is unsupported
- ILR / PWLR clip-floor 1e-10 silently biases composition coordinates at trace concentrations instead of using zero-imputation


### HIGH — ALIAS / line-identification: deviations from canonical Noel 2025 + ad-hoc gates

**Topics affected:** peak-id  
**Expected total lift:** +0.05 to +0.15 (paper-faithful ALIAS reportedly achieves macro_F1 ~0.564 at RP=600 vs current 0.402; spectral_nnls continuum bias affects every hybrid-mode call)  

**Root cause:**

The ALIAS identifier silently substitutes ad-hoc engineering modifications for the canonical Noel 2025 confidence-level formula: k_det uses N_matched in place of N_X (number of theoretical lines), inverting the k_sim/k_shift weighting; k_det is then geometric-mean blended with P_cov and multiplied by sqrt(N_expected/5) — none of which appear in Noel 2025; hard gates (N_matched < 3 unless N_expected ≤ 4 with strict-all-matched) reject high-CL elements with few but strong lines (resonance-dominated alkali metals). The comb identifier similarly caps the fingerprint denominator at fingerprint_top_k=10, creating asymmetric coverage scoring. The Spectral NNLS identifier uses a monomial wl_norm**deg basis combined with NNLS coefficient ≥ 0 — mathematically incapable of representing a decreasing bremsstrahlung continuum (forces continuum mismatch into element coefficients), and uses unconstrained (A^T A)^-1 diagonal for NNLS uncertainty (meaningless for boundary-zero coefficients). BIC pruning accepts a noise_variance argument but never uses it. A docstring even cites the wrong arXiv ID for the ALIAS paper.

**Findings in family:**
- ALIAS k_det blend uses N_matched instead of N_X (number of theoretical lines), inverting the k_sim/k_shift weighting prescribed by Noel 2025
- ALIAS k_det is modified with geometric-mean(k_det_raw, P_cov) and sqrt(N_expected/5) penalty — ad-hoc factors not in canonical paper
- ALIAS imposes ad-hoc hard gates (N_matched < 3 unless N_expected <= 4 with strict-all-matched, or N_expected <= 1) that are not in the canonical Noel 2025 confidence-level formula
- ALIAS docstring cites wrong arXiv ID for the Noel 2025 ALIAS reference
- Comb fingerprint denominator is min(len(teeth), fingerprint_top_k=10) creating asymmetric scoring: elements with >10 teeth get same denominator as elements with exactly 10
- Spectral NNLS polynomial continuum basis is monomials wl_norm**deg, all non-negative and non-decreasing on [0,1]; NNLS constraint forces the modeled continuum to be non-decreasing in wavelength
- Spectral NNLS uses unconstrained (A^T A)^-1 diagonal for coefficient uncertainty, ignoring the active-set boundary structure of NNLS
- BIC pruning accepts noise_variance argument but never uses it; formula is unknown-variance Gaussian regardless of caller input


### HIGH — Doppler/Voigt broadening: regime-independent algebraic and convention errors

**Topics affected:** stark, manifold, bayesian  
**Expected total lift:** +0.02 to +0.05 (41% Gaussian width bias propagates linearly into n_e extraction per Aragón & Aguilera 2008 §3; manifold inversion sees this on every spectrum)  

**Root cause:**

TwoZoneBayesianForwardModel computes σ_Doppler with a spurious factor of 2 inside the sqrt — using v_most_probable=√(2kT/m) where the Maxwell 1D std-dev √(kT/m) is required — over-broadening the Voigt Gaussian core by √2 ≈ 1.41×. The same factor-2 bug recurs at four sites in cflibs/manifold/generator.py (lines 429, 648, 820) and cflibs/inversion/solve/coarse_to_fine.py (line 500), confirmed by grep. This is the exact bug already fixed in profiles.doppler_width — drift between the canonical helper and its four copies. Combined with the Stark family above, the Voigt convolution in the manifold path inherits multiplicative widening on both Gaussian (Doppler) and Lorentzian (Stark) components.

**Findings in family:**
- TwoZoneBayesianForwardModel uses spurious factor-of-2 in Doppler sigma (1.41× over-broadening of Voigt Gaussian core)


### HIGH — Uncertainty quantification: incomplete propagation + invalid posterior comparisons

**Topics affected:** uncertainty, bayesian  
**Expected total lift:** +0.02 to +0.05 direct; calibration-only fixes do not change point estimates, but mis-calibrated σ_C breaks every downstream Bayesian prior / quality-cut and silently passes broken inversions  

**Root cause:**

The uncertainty pipeline has systematic gaps in the variance budget: (a) saha_factor_with_uncertainty treats n_e as a scalar with explicit 'would need to be a ufloat' comment, and is orphaned (zero callers); (b) the convenience run_monte_carlo_uq drops atomic-data uncertainty entirely — AtomicDataUncertainty.from_transitions has zero non-test callers; (c) MC perturbation treats A_ki errors as independent per line, ignoring perfect intra-multiplet correlation that physically arises from shared upper-level lifetimes; (d) MC intensity perturbation uses additive Gaussian with hard floor at 1.0, biasing low-intensity ps-LIBS lines (where Poisson statistics would be correct); (e) partition function uncertainty σ_U is treated as exact in propagate_through_closure_*, but defect family 'partition functions' (Wave-1 C) says U disagrees with direct-sum by 2-3× at LIBS T — so propagating zero variance is mis-calibrated; (f) Joint optimizer uses diagonal-only softmax Jacobian, ignoring off-diagonal -c_i*c_j coupling (30-50% under-reporting for trace elements); (g) compare_with_bayesian uses Gaussian mean ± 2σ instead of actual percentiles; and (h) silently substitutes T=1 eV / n_e=1e17 fallbacks when attributes are missing.

**Findings in family:**
- saha_factor_with_uncertainty treats n_e as scalar and is never wired into the analytical pipeline (n_e uncertainty not propagated)
- run_monte_carlo_uq convenience function silently drops atomic-data uncertainty (NIST A_ki grades never sampled)
- Monte Carlo perturbation treats A_ki errors as independent per line, ignoring perfect intra-multiplet correlation
- Monte Carlo intensity perturbation uses additive Gaussian with hard floor at 1.0, biasing low-intensity ps-LIBS lines
- Partition function uncertainty is treated as exact in propagate_through_closure_*; combined with audit defect C this is the dominant unmodelled error source
- Joint optimizer concentration uncertainty uses diagonal-only softmax Jacobian, ignoring -c_i*c_j off-diagonal coupling
- Bayesian comparison assumes Gaussian posterior (mean +/- 2 sigma) instead of using actual credible intervals
- compare_with_bayesian silently substitutes T=1 eV / n_e=1e17 fallbacks when the Bayesian result lacks expected attributes


### MEDIUM — ns-LIBS-tuned defaults and unprovenanced empirical constants

**Topics affected:** closure, self-absorption, iterative-solver  
**Expected total lift:** +0.02 to +0.05 (mostly subsumed by Stark and self-absorption fixes; standalone fix yields better-calibrated uncertainty bands)  

**Root cause:**

Multiple subsystems carry hardcoded constants that were either copied from ns-LIBS literature without ps-LIBS validation or fabricated as 'reasonable defaults' with no per-value citation. Matrix-effects correction _DEFAULT_FACTORS tables ({METALLIC.Fe: 1.0, METALLIC.C: 0.85, etc.}) are tagged source='default_literature' with no per-factor DOI; the values reflect ns-LIBS thermal segregation that ps-LIBS does NOT exhibit (ps ablation is more stoichiometric). Hermann two-region T_corona = 0.8 * T_K is a magic constant. Self-absorption mask_threshold=3.0, resonance_tau_boost=1.5, R_D=0.5, plasma_length=0.1 cm, n_e_ref=1e17 (CDSB), and pressure_pa=STP_PRESSURE all carry implicit ns-LIBS assumptions. These compound into systematic bias whose direction is not measurable from synthetic test data (which uses the same defaults).

**Findings in family:**
- Matrix-effects correction factors in _DEFAULT_FACTORS have no traceable per-factor literature provenance and are uncalibrated for ps-LIBS


---

## 3. Remediation plan (23 steps, dependency-ordered)

### Step 1 — [M] Stark width unit & density-reference normalization (Wave-2 critical path)

- **Category:** algorithmic_wrong_equation
- **Expected lift:** +0.05 to +0.15 individually; resolves the 20x over-broadening at n_e=1e17 documented vs Aragón & Aguilera 2010 Fe II 430.317 nm FWHM=0.033 nm.
- **Dependencies:** none
- **Files:**
  - `cflibs/radiation/kernels.py`
  - `cflibs/radiation/stark.py`
  - `scripts/populate_stark_widths.py`
  - `cflibs/atomic/database.py`
  - `cflibs/atomic/snapshot.py`
  - `cflibs/inversion/solve/bayesian/forward.py`
  - `tests/test_stark_provenance.py`
  - `tests/test_stark.py`
- **Validation gate:** Reproduce Aragón & Aguilera 2010 Table 2 Fe II 430.317 nm FWHM at n_e=1e17, T=10 kK to within 10%; voigt_fwhm parity test for Fe II 273.955 nm at ps-LIBS (T=1 eV, n_e=1e17) against literature.
- **Risk:** All Stark-using tests need re-baselining; coordinate with manifold cache invalidation (cached manifolds become invalid); risk of double-correcting if anyone has compensated downstream for the over-broadening (grep for any explicit `* 0.05` or `* 20` constants in inversion paths).

### Step 2 — [M] Stark schema split: separate stark_alpha_T (T-exponent) from stark_A_ion (Griem A); wire ion-broadening correction into kernel

- **Category:** implementation_bug
- **Expected lift:** +0.02 to +0.05 (T-scaling fix + 5-15% width correction per Konjević 2002 / Griem 1974 for non-hydrogenic ions)
- **Dependencies:** 1
- **Files:**
  - `cflibs/atomic/database.py`
  - `cflibs/atomic/snapshot.py`
  - `cflibs/radiation/kernels.py`
  - `cflibs/radiation/stark.py`
  - `scripts/populate_stark_widths.py`
  - `cflibs/inversion/solve/bayesian/forward.py`
  - `migrations/`
- **Validation gate:** Sweep T=0.5-1.3 eV at fixed n_e=1e17; observed gamma_S(T) scaling matches T^(-0.5) within 5%; ion-broadening adds 5-15% to FWHM for lines with tabulated A.
- **Risk:** DB schema migration on libs_production.db (28135 stark-populated rows need re-tagging); requires backfilling actual T-exponent and A_ion values from STARK-B per element; backward-compatible read path needed during migration.

### Step 3 — [S] Doppler σ factor-of-2 bug fix at all four sites

- **Category:** algorithmic_wrong_equation
- **Expected lift:** +0.02 to +0.05 (41% Gaussian width bias removed from manifold and TwoZone Bayesian paths)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/solve/bayesian/forward.py`
  - `cflibs/manifold/generator.py`
  - `cflibs/inversion/solve/coarse_to_fine.py`
- **Validation gate:** Unit test: forward.py:456 sigma_doppler == profiles.doppler_width(λ, T, m)/2.355 within rtol=1e-9 for Fe II 273.955 nm at T=1 eV; grep `sqrt(2.0 \* T_eV` returns empty.
- **Risk:** Manifold cache invalidation (combine with #1); centralize doppler_sigma_jax in profiles.py to prevent re-drift.

### Step 4 — [S] SpectrumModelJax: route compute_spectrum through kernels.forward_model so apply_stark is honored

- **Category:** implementation_bug
- **Expected lift:** +0.02 to +0.05 (Voigt instead of Gaussian-only on manifold / GPU-accelerated Bayesian forward)
- **Dependencies:** 1, 3
- **Files:**
  - `cflibs/radiation/spectrum_model.py`
  - `tests/radiation/test_spectrum_model_jax.py`
- **Validation gate:** Parity test at rtol=1e-5 between SpectrumModel and SpectrumModelJax with apply_stark=True in PHYSICAL_DOPPLER mode on Fe II 273.955 nm at T=1 eV, n_e=1e17; manifold-generated spectra match forward-model directly.
- **Risk:** Performance regression if JAX path was faster due to Gaussian-only kernel; verify benchmark hold.

### Step 5 — [M] Wire self-absorption corrector into IterativeCFLIBSSolver and BayesianForwardModel

- **Category:** missing_feature
- **Expected lift:** +0.03 to +0.10 (Aguilera 2014 reports 5-15% concentration improvement for resonance-rich elements; Si is the most-missed element in vrabel2020 soil corpus)
- **Dependencies:** 6, 7
- **Files:**
  - `cflibs/inversion/solve/iterative.py`
  - `cflibs/inversion/solve/bayesian/forward.py`
  - `tests/inversion/`
- **Validation gate:** Round-trip synthetic Si test (60% SiO2 matrix, T=8000K, n_e=5e16, L=0.1 cm) with optically thick Si I 251.611 nm: recovered C_Si within 10% of ground truth with corrector ON, biased low by >20% with corrector OFF; replicates Aguilera 2014 limestone Ca recovery.
- **Risk:** B4 wired in correctly recomputes τ from plasma state — invariant preserved; default opt-in flag prevents regression on test fixtures that assume no SA correction; needs new end-to-end thick-plasma tests.

### Step 6 — [S] Fix doublet-ratio formula (A·λ^2 → A/λ) and update test fixture to physical formula

- **Category:** algorithmic_wrong_equation
- **Expected lift:** Indirect; required before #5 to avoid systematic doublet bias of ~21% at typical Fe doublet spacing
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/physics/self_absorption.py`
  - `tests/test_self_absorption.py`
- **Validation gate:** Synthetic Fe I doublet (358.119 / 374.948 nm, common upper level) from optically-thin forward model gives ratio matching _theoretical_doublet_ratio to <2% (currently disagrees by ~21%).
- **Risk:** Test fixture _make_doublet must change to inject intensities via correct A/λ formula; existing tests pinning self-consistent buggy magnitudes will fail.

### Step 7 — [M] CDSB: pass species number density n_s instead of n_e to _estimate_initial_tau; recompute U(T) per iteration

- **Category:** algorithmic_wrong_equation
- **Expected lift:** Combined +0.03-0.10 with #5; n_s/n_e ratio in ps-LIBS low-ionization plasmas is 10-1000×, so the tau scaling is wrong by orders of magnitude
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/physics/cdsb.py`
  - `tests/test_cdsb.py`
  - `tests/test_vrabel2020_cdsb.py`
- **Validation gate:** Refutability tests: (a) hold n_Fe constant, sweep n_e 1e15-1e18, recovered tau is constant within 5%; (b) hot ground-truth (T=10000K) injected with cool initial T=4000K converges to within 5% of ground truth across 3+ iterations (currently stagnates because U_old==U_new); (c) reproduce Cristoforetti & Tognoni 2013 Ca recovery on limestone within their reported 4% accuracy.
- **Risk:** API change: CDSBPlotter.fit must accept concentrations + total_n_cm3 (or partition_func_callable + species_density); breaks downstream callers.

### Step 8 — [S] COG _estimate_optical_depth λ² fix; remove resonance_tau_boost double-count; raise mask_threshold default 3.0 → 5.0

- **Category:** algorithmic_wrong_equation
- **Expected lift:** Indirect (corrects 5-order-of-magnitude tau underestimate; restores access to lines at tau=3-5 that Cristoforetti & Tognoni 2013 say are correctable)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/physics/self_absorption.py`
  - `cflibs/inversion/physics/cdsb.py`
  - `tests/test_self_absorption.py`
  - `tests/test_cdsb.py`
- **Validation gate:** (a) COG unit test: CurveOfGrowthAnalyzer._estimate_optical_depth(column_density=1e15, max_log_gf=log10(0.34), doppler_width_nm=0.001, wavelength_nm=251.6) returns tau_0 ∈ [0.1, 1.0]; (b) Si I 251.611 nm at tau≈4 is correctable with mask_threshold=5.0; (c) tau_resonance vs tau_excited ratio comes purely from Boltzmann factor (resonance_tau_boost set to 1.0 produces same ranking).
- **Risk:** Three tightly-coupled small fixes in one PR; risk of test thrash if not split.

### Step 9 — [L] Iterative solver: add Stark-broadening n_e estimator as primary path; demote pressure balance to fallback with warning

- **Category:** missing_feature
- **Expected lift:** +0.05 to +0.15 (Tognoni 2010 §4.2 designates Stark as primary n_e diagnostic; pressure balance with STP=1 atm is wrong for ps-LIBS at all gate delays)
- **Dependencies:** 1, 2
- **Files:**
  - `cflibs/inversion/solve/iterative.py`
  - `cflibs/inversion/solve/closed_form.py`
  - `cflibs/inversion/physics/stark.py`
- **Validation gate:** Synthetic Hα FWHM 0.4 nm at n_e=5e17 recovers n_e within 10%; pressure-balance fallback emits logger.warning when used; comparison vs BayesianForwardModel (which uses Stark) agrees to within 20%.
- **Risk:** Requires reliable per-line Stark coefficients (depends on #1, #2); non-H Stark (Fe I, Cu I, Mg II) needs implementation for ps-LIBS where Hα may be weak; ne_mode='pressure' callers need migration path.

### Step 10 — [S] Iterative solver: silent-failure quality gates (positive slope, |slope|→0, oscillating n_e, two-region constants)

- **Category:** implementation_bug
- **Expected lift:** +0.01 to +0.03 (eliminates silent acceptance of unphysical inversions; primarily fixes outlier failures contributing to F1 variance)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/solve/iterative.py`
- **Validation gate:** Refutability tests: (a) inverted Boltzmann plot (positive slope) returns converged=False with quality_metrics['boltzmann_slope_failure']=True; (b) near-zero negative slope clamped at T_max=50000 K with warning; (c) ne_tolerance_frac tightened to 0.02 + checked on undamped iterate; (d) T_corona_ratio exposed as ctor parameter (default 0.8 with docstring warning).
- **Risk:** Some existing tests may now correctly report non-convergence — re-baseline rather than silently pass; tighter ne_tolerance may extend iteration count modestly.

### Step 11 — [M] Iterative + Bayesian: down-weight or exclude resonance lines in Boltzmann fit; extend LineObservation with E_lower

- **Category:** missing_feature
- **Expected lift:** +0.03 to +0.08 (Aguilera 2014 reports 5-15% Cu/Fe matrix improvement)
- **Dependencies:** 5
- **Files:**
  - `cflibs/inversion/common/`
  - `cflibs/inversion/solve/iterative.py`
  - `cflibs/inversion/identify/`
  - `tests/inversion/`
- **Validation gate:** Fe spectrum at T=8000K with Fe I 248.327 nm artificially attenuated by 50% (simulated τ≈0.7): recovered T bias drops from 500-1500 K to <200 K with resonance line excluded; CDSB correction (#5/#7) handles the same case via correction rather than exclusion.
- **Risk:** LineObservation dataclass change is invasive (additive Optional[float]=None to avoid breakage); test fixtures need resonance flag injection.

### Step 12 — [M] ALIAS: revert k_det to paper-pure Noel 2025 formula (N_X=N_expected); make ad-hoc modifiers opt-in

- **Category:** algorithmic_wrong_equation
- **Expected lift:** +0.05 to +0.15 (paper-faithful ALIAS achieves macro_F1 ~0.564 at RP=600 per Noel 2025 vs current 0.402; combined effect of N_X correction + removing ad-hoc geomean blend + removing hard gates)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/identify/alias.py`
  - `tests/test_alias.py`
  - `tests/test_alias_presets.py`
  - `tests/test_alias_high_recall_workflow.py`
- **Validation gate:** (a) Re-baseline test_alias_presets.py against paper-pure k_det formula; (b) hold-out test on aalto/vrabel2020 corpora shows macro_F1 ≥ current with paper-pure mode; (c) docstring updated with verified DOI 10.1016/j.sab.2025.107255 (cleanup A from family).
- **Risk:** Test re-baselining is substantial; flip default cautiously behind feature flag; A/B comparison required on full corpus.

### Step 13 — [S] SpectralRefiner: enforce Σ C = 1 (softmax-parameterize or post-normalize); add closure_diagnostic to ClosureResult

- **Category:** missing_feature
- **Expected lift:** +0.02 to +0.05 (closure violation directly biases element-presence thresholds)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/solve/spectral_refiner.py`
  - `cflibs/inversion/physics/closure.py`
  - `tests/test_spectral_refiner.py`
- **Validation gate:** Refiner on synthetic Fe=0.7, Ni=0.3 returns concentrations satisfying |Σ - 1| < 1e-6; ClosureResult exposes closure_residual = Σ C_s - 1 in matrix mode; warning emitted when |residual| > 0.05.
- **Risk:** Switching to softmax changes optimizer geometry; verify convergence speed not degraded.

### Step 14 — [M] Spectral NNLS: signed/Chebyshev continuum basis; active-set NNLS variance

- **Category:** implementation_bug
- **Expected lift:** +0.02 to +0.05 (decreasing bremsstrahlung continuum no longer leaks into element coefficients; meaningful boundary-aware SNR for hybrid identifier)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/identify/spectral_nnls.py`
  - `tests/test_spectral_nnls.py`
- **Validation gate:** Synthetic Fe + decreasing bremsstrahlung continuum across 250-450 nm: Fe coefficient bias drops from systematic ~10-20% high to <2%; absent-element sigma_coeffs use active-set conditional variance, not OLS diagonal.
- **Risk:** Hybrid identifier stage-1 detection set will shift; tests pinning specific per-element coefficient values need re-baselining.

### Step 15 — [S] Comb identifier: full-coverage denominator (remove fingerprint_top_k cap from denominator)

- **Category:** algorithmic_wrong_equation
- **Expected lift:** +0.01 to +0.03 (corpus-dependent; mainly recall on line-rich elements like Fe, V)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/identify/comb.py`
  - `tests/test_comb_recall.py`
  - `tests/test_comb_precision.py`
- **Validation gate:** Two-element synthetic test (12 teeth/10 active vs 8 teeth/8 active at correlation 0.6): paper-faithful score correctly ranks 8/8 ≥ 10/12 (current code returns tie 0.6/0.6); macro_F1 on ps-LIBS corpus not regressed.
- **Risk:** Score distribution shifts; downstream ranking thresholds may need re-tuning.

### Step 16 — [M] Closed-form ILR solver: closure_mode parameter (matrix, oxide) + STP_PRESSURE warning

- **Category:** missing_feature
- **Expected lift:** +0.02 to +0.05 (correct closure for geological/metallurgical matrices; ChemCam/Tognoni 2010 standard)
- **Dependencies:** 13
- **Files:**
  - `cflibs/inversion/solve/closed_form.py`
  - `cflibs/inversion/physics/closure.py`
  - `tests/test_closed_form.py`
- **Validation gate:** Synthetic geological spectrum: closed-form solver with closure_mode='oxide' achieves macro_F1 within 5% of iterative solver's oxide-mode F1; pressure_pa default emits warning when used unmodified.
- **Risk:** Constrained-WLS via Lagrange multiplier needed for matrix mode; oxide mode requires log-Jacobian extension.

### Step 17 — [M] Closure cleanup: rename plr_transform→alr_transform; implement true Hron 2012 pivot coordinates; fix dirichlet_residual scale-invariance

- **Category:** algorithmic_wrong_equation
- **Expected lift:** +0.01 to +0.03 (PWLR L2 regularization now uses correct isometric metric; dirichlet residual no longer scales with F)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/physics/closure.py`
  - `cflibs/inversion/physics/closure_strategy.py`
  - `tests/test_closure_pwlr.py`
  - `tests/test_closure_dirichlet.py`
- **Validation gate:** (a) For x=[0.7,0.2,0.1] pivot=0, true PLR z_1 = sqrt(2/3)*ln(0.7/sqrt(0.02)) ≈ 1.305 reproduced; (b) Aitchison distance preserved under PLR transform; (c) Dirichlet residual unchanged when intensities scaled by 1e3.
- **Risk:** Closure mode API surface change; tests bypass scale-invariance via F=1 fixtures, so finding requires new test cases.

### Step 18 — [L] Uncertainty pipeline overhaul: U_s as UFloat + n_e UFloat + Poisson noise option + multiplet-correlated A_ki

- **Category:** missing_feature
- **Expected lift:** +0.01 to +0.03 direct (calibration-only); enables downstream Bayesian-prior or quality-cut steps to use correctly-calibrated σ
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/physics/uncertainty.py`
  - `tests/test_uncertainty.py`
- **Validation gate:** (a) propagate_through_closure_standard with U_s as UFloat (10% uncertainty) yields σ_C contribution per Cavalcanti 2013; (b) run_monte_carlo_uq defaults to COMBINED + auto-builds AtomicDataUncertainty; (c) Fe multiplet at 380 nm: correlated A_ki draw produces σ_T ≈ sqrt(N) larger than independent draw; (d) intensity at 20 counts uses Poisson sampler by default.
- **Risk:** Backward-compatible defaults: emit warning when fallback used; existing tests pinning specific uncertainty magnitudes need updated tolerances.

### Step 19 — [S] Joint optimizer: full softmax Jacobian for concentration uncertainty

- **Category:** implementation_bug
- **Expected lift:** 0 direct (uncertainty-only); fixes 30-50% under-reporting on trace elements
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/solve/joint_optimizer.py`
  - `tests/test_joint_optimizer.py`
- **Validation gate:** 5-element degenerate composition: analytical σ_C matches Monte Carlo through softmax map within 5% (currently 30-50% low for trace elements).
- **Risk:** Only affects parameter_uncertainties; no point-estimate change.

### Step 20 — [S] Bayesian comparison: use actual percentile credible intervals; raise on missing posterior attributes

- **Category:** implementation_bug
- **Expected lift:** 0 (validation-only tooling); prevents silent fabrication of comparison results
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/physics/uncertainty.py`
  - `tests/test_uncertainty.py`
- **Validation gate:** (a) Log-normal posterior with σ_log10=0.3: compare_with_bayesian uses asymmetric percentile interval (lower bound mismatches current code by >50%); (b) MonteCarloResult.compare_with_bayesian(object()) raises AttributeError instead of returning fabricated values.
- **Risk:** Callers passing minimal Bayesian result objects break — but they were getting wrong answers anyway.

### Step 21 — [M] Stark shift: extend AtomicSnapshot, populate line_stark_shift, apply at kernel time as f(n_e, T) per Konjević 2002 Eq. 2

- **Category:** missing_feature
- **Expected lift:** +0.01 to +0.03 (regime-dependent; largest effect on ion lines at n_e ≥ 5e16 cm^-3)
- **Dependencies:** 1, 2
- **Files:**
  - `cflibs/atomic/snapshot.py`
  - `cflibs/atomic/database.py`
  - `cflibs/radiation/kernels.py`
  - `cflibs/radiation/stark.py`
  - `tests/test_stark.py`
- **Validation gate:** Fe II 273.955 nm at n_e=1e17, T=1 eV: line center shifted by ~-6 pm (Konjević 2002 Tab 15); ALIAS identification with ±0.01 nm tolerance now matches the shifted line.
- **Risk:** Manifold cache invalidation (line centers change with n_e); ALIAS/comb tolerance behavior changes.

### Step 22 — [S] Matrix-effects defaults: mark _DEFAULT_FACTORS as stub; warn on use; require explicit calibration source

- **Category:** regime_mismatch
- **Expected lift:** 0 to +0.02 (correctness; ns-LIBS defaults systematically biased for ps-LIBS)
- **Dependencies:** none
- **Files:**
  - `cflibs/inversion/physics/matrix_effects.py`
  - `tests/test_matrix_effects.py`
- **Validation gate:** Use of default DB emits UserWarning; CorrectionFactor source field requires non-generic value (citation/DOI per factor).
- **Risk:** Existing tests using default DB need update.

### Step 23 — [S] ALIAS docstring + low-impact cleanups (arXiv ID correction, BIC noise_variance parameter cleanup, ILR clip-floor warning)

- **Category:** implementation_bug
- **Expected lift:** 0 (documentation / API hygiene)
- **Dependencies:** 12
- **Files:**
  - `cflibs/inversion/identify/alias.py`
  - `cflibs/inversion/identify/model_selection.py`
  - `cflibs/inversion/physics/closure.py`
- **Validation gate:** Docstring cites DOI 10.1016/j.sab.2025.107255; bic_prune_elements either uses noise_variance or removes parameter; ILR clip-floor logs warning when input < user-supplied physical floor.
- **Risk:** Trivial.

---

## 4. Cross-cutting patterns

- Convention drift between FWHM and HWHM, and between reference densities (1e16 vs 1e17 cm^-3) across modules — caused by absence of any single source of truth annotation on snapshot/datatable fields (no ASSERT_CONVENTION marker). Stark family has the canonical example (20× over-broadening) but the same hazard exists for Doppler σ vs FWHM (centralize doppler_sigma_jax) and for partition-function reference temperature.
- ns-LIBS literature defaults silently transplanted into a ps-LIBS hardware target. The codebase advertises ps-LIBS (1 ps Yb:fiber at 1040 nm, T=0.5-1.3 eV, n_e=1e16-1e18) but defaults across 8+ files assume hot ns-Nd:YAG conditions: STP_PRESSURE=1 atm, n_e_ref=1e17 cm^-3, R_D=0.5, T_corona=0.8*T_K, mask_threshold=3, _DEFAULT_FACTORS tables, partition-function fit ranges.
- Dead code + missing wiring + internally-buggy modules layered together. Self-absorption, CDSB, COG, doublet-ratio, and Stark-shift all exist as well-developed-looking modules but are never invoked from the inversion pipeline. The B4 Wave-1 fix correctly tightened SelfAbsorptionCorrector's internal recursion but did not wire it in. Per-module audits would all give green on these modules; the integration gap only surfaces at architecture level.
- Algebraic shortcuts using 'standard'-looking formulae that drop one factor of λ or use line strength S where intensity is required (doublet ratio λ^3 vs 1/λ; COG _estimate_optical_depth λ vs λ^2). These survive review because they 'look like a formula from the literature' — fix is to add docstring-level dimensional-check assertions.
- Silent failure with downstream-visible 'success'. Solver clamps T=50000 K + reports converged=True; compare_with_bayesian fabricates defaults T=1eV/n_e=1e17 for missing attributes; pressure-balance returns biased n_e with no warning; matrix mode returns concentrations with no |Σ-1| diagnostic. Pattern: quality_metrics dict exists but is not consulted as authoritative; need a single ResultStatus enum that gates downstream usage.
- Test fixtures that injection-encode the same bug being tested — TestDoubletRatioCorrection fixture uses r_theory = (lam1/lam2)^3 (same wrong formula as code), test_closure_dirichlet uses F=1 (hides scale-invariance bug), test_alias_presets pins ad-hoc k_det modifiers, test_spectral_nnls is unaware of monomial-basis non-decreasing constraint. Test-driven correctness assumes the test fixture is independent of the implementation — these aren't.
- Implicit assumption of independence between physically correlated quantities. A_ki errors treated as independent per-line ignore intra-multiplet covariance (shared lifetime); softmax Jacobian uses diagonal only (ignores -c_i c_j simplex coupling); uncertainty propagation treats U_s and n_e as scalars with zero variance; Bayesian comparison treats log-skewed posteriors as Gaussian.
- Citations that are real but mis-attributed for the specific quantitative claim. Multiple findings (mask_threshold=5, plr/ilr literature, matrix-effects defaults, BIC noise_variance, JCGM-101 anti-pattern) cite real papers but the specific numeric thresholds or 'standard practice' claims are not in the cited source. Suggests the codebase has a culture of weakly-grounded engineering defaults — fix is to attach a verifiable per-default citation map.

## 5. ps-LIBS-specific issues

- cflibs/inversion/solve/iterative.py:704 + closed_form.py:44 — pressure_pa defaults to STP_PRESSURE (1 atm = 101325 Pa); ps-LIBS late-time plumes typically <0.1 atm, early-time >10 atm; never 1 atm in the analysis window.
- cflibs/radiation/stark.py:17 — REF_NE = 1.0e16 cm^-3 is at the bottom of the ps-LIBS n_e range (1e16-1e18); literature/STARK-B convention is 1e17 cm^-3 (typical for ns-LIBS densities). The mismatch is compounded with the FWHM/HWHM bug into ~20× over-broadening.
- cflibs/radiation/stark.py:84-89 — R_D=0.5 hardcoded; for ps-LIBS at n_e=1e16, T=1 eV → R≈0.4 (in-regime); at n_e=1e18, T=0.5 eV → R≈1.2 (OUT of quasi-static regime). The 0.5 value matches only a single ns-LIBS reference point.
- cflibs/inversion/solve/iterative.py:1121 — T_corona = 0.8 * T_K hardcoded for two-region model with comment 'Per Hermann (2017)'; no published constant for sub-10 ps Yb:fiber regime; Marin Roldan 2021 documents that ps plumes have substantially different geometry.
- cflibs/inversion/physics/self_absorption.py:653 — mask_threshold=3.0 below ALL published validity windows (Bulajic <4, CDSB <5, SAF-LIBS <10); for ps-LIBS where lines are narrower (lower Stark), line-center τ is more concentrated and stronger correction is needed, not less.
- cflibs/inversion/physics/cdsb.py:212 — resonance_tau_boost=1.5 is empirically tuned ns-LIBS heuristic; at ps-LIBS T=0.5-1.3 eV the Boltzmann factor already strongly favors resonance lines, so the multiplicative boost amplifies a defect that doesn't exist at low T.
- cflibs/inversion/physics/matrix_effects.py:176-218 — _DEFAULT_FACTORS table values reflect ns-LIBS differential ablation/thermal segregation (e.g., METALLIC.C=0.85 captures 15% carbon loss). ps-LIBS ablation is more stoichiometric; defaults bias ps-LIBS in the wrong direction.
- cflibs/inversion/physics/uncertainty.py:932 — MC additive Gaussian with hard floor at 1.0 biases the mean upward for low-count lines. ps-LIBS routinely produces 10-50 count weak/trace lines (lower per-pulse ablated mass, lower upper-level population), where Poisson statistics dominate.
- cflibs/inversion/physics/cdsb.py:481-482 — density_factor = n_e/n_e_ref where the function only receives n_e (not species density n_s). At ps-LIBS T (0.5-1.3 eV) ionization fraction is 1e-3 to 1e-2, so n_s/n_e ratio is 100-1000× — far worse than the ns-LIBS regime this code was written against.
- cflibs/radiation/spectrum_model.py:475-629 — SpectrumModelJax (used by manifold pipeline per architecture doc) silently drops Stark; ps-LIBS at typical n_e=1e17 has Stark FWHM comparable to or exceeding Doppler+instrument FWHM, so Gaussian-only profiles are sharply wrong; Wave-1 A2 (apply_stark=True default) is bypassed entirely in the JAX path.

---

## 6. Completeness critique

### 6.1 Missing subsystems (28)

- LTE validator (cflibs/plasma/lte_validator.py) — McWhirter + temporal check is implemented and IS wired into iterative.py:1223/1358 (both _solve_python and _solve_lax). The audit cited 'NON-LTE DEPARTURES — entirely outside the audit' but in fact a validator exists; what was NOT audited is whether (a) MCWHIRTER_C constant matches Cristoforetti 2010 Table 1 values (per-species), (b) the relaxation-time check uses ps-LIBS-relevant timescales (ps-LIBS gate windows are 10-1000 ns; the temporal_check defaults need a ps-LIBS audit), and (c) what the solver DOES with a failing LTE report (does it warn-and-continue, or downgrade quality_flag, or refuse to converge?).
- Preprocessing baseline (cflibs/inversion/preprocess/preprocessing.py) — estimate_baseline_snip, _als, _percentile and _select_auto_baseline_method are entirely unaudited. SNIP iteration count, ALS lambda/p hyperparameters, and the auto-selection heuristic directly bias every Boltzmann intercept downstream. For ps-LIBS, lower continuum than ns-LIBS means baseline subtraction errors are a larger fraction of line area.
- Voigt deconvolution (cflibs/inversion/preprocess/deconvolution.py) — VoigtFitResult / deconvolve_peaks (used by line_detection.py and benchmark predictors). FWHM-vs-sigma convention inside the JAX Voigt fitter, gamma/sigma identifiability under instrument convolution, and constraint handling for very narrow ps-LIBS lines were not audited.
- Wavelength calibration (cflibs/inversion/preprocess/wavelength_calibration.py) — calibrate_wavelength_axis + RANSAC + BIC model selection + quality gates. Mis-calibration of <=1 pixel directly defeats ALIAS tolerance windows; this entire module was untouched by the audit despite being upstream of every identifier.
- Outlier detection / SAM (cflibs/inversion/preprocess/outliers.py) — SpectralAngleMapper + MADOutlierDetector + mad_clean_channels. Threshold defaults, the choice of mean vs median reference, and bad-channel impact on the line list never appeared in the audit.
- PCA / denoising (cflibs/inversion/common/pca.py — PCAPipeline) — used by manifold's SpectralEmbedder (vector_index.py:68). n_components selection criterion, mean-centering correctness, and the implicit denoising step before FAISS embedding were not audited. Wrong n_components corrupts every nearest-neighbor manifold inversion.
- Manifold nearest-neighbor inversion (cflibs/manifold/vector_index.py + loader.py) — entire FAISS-based inference path. Uses L2 metric (faiss.METRIC_L2 / IndexFlatL2 + IndexIVFFlat), nprobe/nlist defaults, and PCA embedding. The audit's defect-family D (manifold Nyquist) addressed sampling only; the inversion-metric correctness was not examined. L2 on raw spectra is dominated by continuum scale, not by line patterns — typically wrong metric for compositional inference. Audit explicitly admitted (item 2) this gap.
- Basis-library / batch-forward path (cflibs/manifold/basis_library.py, batch_forward.py) — alternative library-based manifold not audited at all.
- Coarse-to-fine HybridInverter (cflibs/inversion/solve/coarse_to_fine.py) — referenced only via the Doppler factor-of-2 bug at line 500; the entire L-BFGS-B refinement strategy, warm-start logic, and SpectralFitter were untouched. The hardcoded mass=50.0 amu at line 500 (assumes Fe!) is a separate bug from the factor-of-2.
- Multi-gate joint fit (cflibs/inversion/runtime/multi_gate.py) — joint_multi_gate_fit, _chi2 in log space, _composition_uncertainty, warm-start, alpha unpacking. ps-LIBS multi-gate is a key path to higher F1 (audit acknowledged item 9) but not examined.
- Time-resolved solver and gate optimization (cflibs/inversion/runtime/temporal.py) — TimeResolvedCFLIBSSolver, GateTimingOptimizer (uses continuum heuristic n_e/n_ref^2 * sqrt(T/T_ref)), TemporalSelfAbsorptionCorrector, PlasmaEvolutionModel (the entire McWhirter-based gate timing logic).
- Anderson solver for Saha equilibrium (cflibs/plasma/anderson_solver.py) — anderson_solve / picard_solve fixed-point iteration, hardcoded log_ne bounds [_LOG_NE_MIN, _LOG_NE_MAX], convergence tolerance, JAX caching. Convergence behavior of the inner Saha solve was not audited.
- Partition-function provider system (cflibs/plasma/partition.py — PolynomialPartitionFunctionProvider, BatchedPartitionFunctionProvider, irwin_log10_to_ln_coeffs, JAX direct-sum). The audit's defect family C (Wave-1) flagged the 2-3x error vs direct sum but did not audit (a) Irwin-coefficient sign/basis conversion, (b) IPD correction in direct_sum which is silently applied with Debye-Hückel (cflibs/plasma/saha_boltzmann.py: ionization_potential_lowering — non-trivial sign convention), (c) Barklem-Collet 2016 wiring (search confirmed: ZERO mentions in cflibs/), (d) the t_min/t_max clamp behavior at LIBS edges. The polynomial provider clamps T to [t_min, t_max] outside the validity window — this hides extrapolation errors silently.
- Ionization-potential depression / continuum lowering (saha_boltzmann.py + partition.py ionization_potential_depression) — at ps-LIBS n_e=1e18 cm^-3, Debye-Hückel IPD becomes comparable to thermal energy; the application to direct-sum partition truncation and to Saha was not validated for the ps-LIBS regime.
- Continuum / free-free / free-bound modeling — search confirmed ZERO bremsstrahlung / free-bound modules in cflibs/. The forward model has no continuum emission at all (only polynomial continuum is fit in the NNLS identifier as nuisance). This is acceptable IF preprocessing baseline removal is perfect, but for ps-LIBS where bremsstrahlung+recombination is a smaller-but-nonzero baseline, the missing physics shows up as additive systematic.
- Instrument convolution (cflibs/instrument/convolution.py apply_instrument_function, kernels.py) — n_sigma=5 kernel-truncation tolerance, only-Gaussian support, no Lorentzian/Voigt instrument kernels, no echelle slit-function asymmetry. The grid-must-be-evenly-spaced check (rtol=1e-6) may reject calibrated wavelength axes silently.
- Echelle extraction (cflibs/instrument/echelle.py) — entirely unaudited. ChemCam/SuperCam-style cross-dispersed echelle has order-merging artifacts that no test covered.
- Round-trip validation harness (cflibs/validation/round_trip.py) — used as ground-truth oracle, but if it shares the same Doppler-sigma factor-of-2 bug as manifold/coarse_to_fine, every 'validation' test would pass with wrong physics on both sides. Not audited.
- Composition metrics (cflibs/benchmark/composition_metrics.py, posterior_metrics.py) — Aitchison / ILR metric implementations are used to score the F1 = 0.402 result. If these are wrong (zero-imputation issue from closure audit family D in the synthesis), the headline F1 number itself is suspect.
- Reference compositions (cflibs/benchmark/reference_compositions.py) — synthetic-corpus ground truth. Not examined for whether the ground-truth normalization and oxide closure matches what the inversion enforces.
- Failure attribution (cflibs/observability/failure_attribution.py, element_confusion.py) — telemetry that drives optimization decisions. Bias in attribution -> wrong improvement direction. Not audited.
- Rust core (native/cflibs-core/src/comb_matching.rs, partition.rs) — confirmed only Python audited. Comb matching has its own peak-tolerance + integer-shift quantization that could differ from the Python implementation; partition.rs may carry independent Irwin-coefficient sign bugs. The Wave-1 audit explicitly listed Rust as outside scope (item 6) and that gap remains.
- Bandit / Thompson allocator (cflibs/bandit/thompson_allocator.py) — used to allocate spectra to methods; uncalibrated probabilities feed it. Not in scope but contributes to system-level outcomes.
- Bayesian prior specification (cflibs/inversion/solve/bayesian/priors.py — create_temperature_prior, create_density_prior, create_concentration_prior, NoiseParameters, TwoZonePriorConfig). Audit item 7 acknowledged this gap explicitly. T-prior bounds and n_e log-uniform bounds need ps-LIBS-specific defaults; current code has untested defaults.
- Bayesian convergence diagnostics (cflibs/inversion/solve/bayesian/diagnostics.py + samplers.py: _assess_convergence, _diagnostics_from_mcmc). What constitutes 'converged' (R-hat, ESS, divergent-transition fraction thresholds) was not examined; NumPyro best practice is R-hat<1.01 and ESS>400, divergences>0 means reparameterize.
- Distributed MCMC + GPU config (cflibs/hpc/distributed_mcmc.py, gpu_config.py, slurm.py) — pipeline scaling not audited.
- DAQ streaming + real-time (cflibs/inversion/runtime/streaming.py, daq_interface.py, hardware/) — these set the SAH timing and gate windows; ns/ps mismatch here corrupts every downstream inference.
- Visualization widgets (cflibs/visualization/widgets.py) — out of physics scope but worth confirming no scientifically misleading defaults (e.g., log-scale on a linear axis).

### 6.2 Missing papers (21)

- Cristoforetti, G., De Giacomo, A., Dell'Aglio, M., Legnaioli, S., Tognoni, E., Palleschi, V., Omenetto, N. (2010). 'Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion.' Spectrochimica Acta Part B 65(1), 86-95. DOI: 10.1016/j.sab.2009.11.005. This is THE canonical paper for LTE validity in transient/inhomogeneous LIBS plasmas (necessary-but-not-sufficient McWhirter + relaxation + diffusion criteria). Cited only twice in cflibs/ (lte_validator + temporal). The audit cited 'McWhirter' as a missing topic in its self-critique but did not pull this central reference.
- Barklem, P.S. and Collet, R. (2016). 'Partition functions and equilibrium constants for diatomic molecules and atoms of astrophysical interest.' Astronomy & Astrophysics 588, A96. arXiv:1602.03304. DOI: 10.1051/0004-6361/201526961. The canonical modern partition-function tabulation for all atoms H-U (and 291 molecules). Wave-1 defect family C noted 2-3x disagreement with direct-sum but the synthesis did not propose wiring B&C16 even though it is the textbook fix and the public dataset (GitHub: barklem/Equilibrium) is freely downloadable. ZERO mentions in cflibs/.
- Konjević, N., Lesage, A., Fuhr, J.R., Wiese, W.L. (2002). 'Experimental Stark Widths and Shifts for Spectral Lines of Neutral and Ionized Atoms (A Critical Review of Selected Data for the Period 1989 through 2000).' J. Phys. Chem. Ref. Data 31(3), 819-927. The reference for FWHM-at-1e17 convention and the Griem ion-broadening A parameter (typical values 0.04-0.2). The synthesis cites Konjević 2002 in justification text but the actual database population script (populate_stark_widths.py) needs explicit unit-and-density audit against this reference.
- Dimitrijević, M.S. (and STARK-B database, http://stark-b.obspm.fr/). The standard ion Stark-broadening source for non-H lines that is the natural target for the audit's proposed schema split (stark_alpha_T vs stark_A_ion). The synthesis points to STARK-B as the source but does not cite the canonical access paper.
- Cristoforetti, G. and Tognoni, E. (2013). 'Calculation of elemental columnar density from self-absorbed lines in laser-induced breakdown spectroscopy: A resource for quantitative analysis.' Spectrochimica Acta Part B 79-80, 63-71. The CDSB algorithm itself. Cited indirectly via cdsb.py but the actual numeric prescription for the U_new computation across iterations needs verification against §3.2 of the paper — the audit identifies U_old==U_new as a defect but never confirms the paper's exact recursion.
- Bredice, F., Borges, F.O., Sobral, H., Villagran-Muniz, M., Di Rocco, H.O., Cristoforetti, G., Legnaioli, S., Palleschi, V., Salvetti, A., Tognoni, E. (2007). 'Measurement of Stark broadening of Mn I and Mn II spectral lines in plasmas used for Laser-Induced Plasma Spectroscopy.' Spectrochimica Acta Part B 62(11), 1237-1245. Standard ps-relevant Stark validation reference; not cited but would be the empirical test for the Wave-2 Stark fix.
- Aragón, C. and Aguilera, J.A. (2014). 'CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy.' Journal of Quantitative Spectroscopy and Radiative Transfer 149, 90-102. DOI: 10.1016/j.jqsrt.2014.07.026. The CSigma framework for handling optically-thick plasma directly. Audit cites 'Aragón & Aguilera 2008' for line selection but the 2014 CSigma paper is the modern, broadly cited reference for handling thick plasma — directly relevant to the dead-code self-absorption family. Recommended remediation for items 5-8 should cite this.
- Hermann, J., Lorusso, A., Perrone, A., Strafella, F., Dutouquet, C., Torralba, B. (2017). 'Simulation of emission spectra from nonuniform reactive laser-induced plasmas.' Physical Review E 96(5), 053210. This is the ACTUAL 'Hermann 2017' reference; the codebase docstring claim 'T_corona = 0.8 * T_K per Hermann (2017)' should be checked against this paper — preliminary check is that this paper presents a full hydrodynamic-radiative simulation, NOT a constant temperature ratio, so the 0.8 constant is likely a misattribution. The audit flags T_corona as 'unprovenanced' but does not name the candidate citation.
- Bulajic, D., Corsi, M., Cristoforetti, G., Legnaioli, S., Palleschi, V., Salvetti, A., Tognoni, E. (2002). 'A procedure for correcting self-absorption in calibration free-laser induced breakdown spectroscopy.' Spectrochimica Acta Part B 57(2), 339-353. The original self-absorption correction prescription. Cited by Wave-1 B4 fix message but the paper's exact τ recomputation (which the audit invokes) needs textual cross-check.
- Aguilera, J.A. and Aragón, C. (2008). 'Multi-element Saha-Boltzmann and Boltzmann plots in laser-induced plasmas.' Spectrochimica Acta Part B 63(7), 784-792. The common-slope multi-element Boltzmann fit + Saha correction prescription. Audit cites this paper repeatedly for resonance-line down-weighting and 'per Aguilera & Aragón §3' but the actual section numbering / quantitative prescription was not verified.
- Hahn, D.W. and Omenetto, N. (2010). 'Laser-Induced Breakdown Spectroscopy (LIBS), Part I: Review of Basic Diagnostics and Plasma–Particle Interactions: Still-Challenging Issues Within the Analytical Plasma Community.' Applied Spectroscopy 64(12), 335A-366A. The canonical LIBS-physics review article; absent from the audit's literature step.
- Hahn, D.W. and Omenetto, N. (2012). 'Laser-Induced Breakdown Spectroscopy (LIBS), Part II: Review of Instrumental and Methodological Approaches.' Applied Spectroscopy 66(4), 347-419. The companion canonical review.
- Tognoni, E., Cristoforetti, G., Legnaioli, S., Palleschi, V. (2010). 'Calibration-Free Laser-Induced Breakdown Spectroscopy: State of the art.' Spectrochimica Acta Part B 65(1), 1-14. The actual CF-LIBS state-of-the-art reference. Cited via 'Tognoni 2010 §4.2' in the synthesis for Stark as primary n_e diagnostic; the citation should be confirmed and §4.2 quoted.
- De Giacomo, A. and Hermann, J. (2017). 'Laser-induced plasma emission: from atomic to molecular spectra.' Journal of Physics D 50(18), 183002. Modern review of LIP physics covering ns-vs-ps timescales explicitly.
- Schmidt, M. and Forsberg, P. (2024 / recent ps-LIBS). Single-shot picosecond LIBS validation studies should be located (e.g., 'Time-Gated Single-Shot Picosecond LIBS' by Gragston, Hsu, Patnaik, Zhang, Roy 2020; Applied Spectroscopy 74(3), 285-299) — provides validated T and n_e ranges for ps-LIBS at 1064 nm and confirms ps-LIBS has lower continuum + narrower lines than ns-LIBS, supporting the audit's ps-LIBS-defaults critique.
- Carsus / TARDIS-SN documentation for Barklem & Collet ingestion (https://tardis-sn.github.io/carsus/io/barklem2016.html). Practical wiring guide for the Wave-2 partition-function remediation that defect family C calls for.
- Egozcue, J.J., Pawlowsky-Glahn, V., Mateu-Figueras, G., Barceló-Vidal, C. (2003). 'Isometric Logratio Transformations for Compositional Data Analysis.' Mathematical Geosciences 35(3), 279-300. The canonical ILR reference. Synthesis closure family cites Aitchison 1982 but missing the actual ILR construction paper that the codebase plr/ilr modules claim to implement.
- Hron, K., Templ, M., Filzmoser, P. (2010 / 2012). 'Imputation of missing values for compositional data using classical and robust methods.' Computational Statistics & Data Analysis 54(12), 3095-3107. The actual reference often confused with the alleged 'Hron 2012 pivot coordinates' — the synthesis closure family flagged this confusion but the canonical pivot-coordinate paper is Filzmoser, Hron, Reimann (2009) 'Univariate statistical analysis of environmental (compositional) data: Problems and possibilities.' Sci. Total Environ. 407(23), 6100-6108.
- Cremers, D.A. and Radziemski, L.J. (2013). 'Handbook of Laser-Induced Breakdown Spectroscopy, 2nd ed.' Wiley. The textbook; covers free-free + free-bound continuum formulae directly — relevant to the missing continuum-modeling subsystem.
- Noel, K. et al. (2025). ALIAS reference. Audit flags wrong DOI in code; the correct citation needs verification (synthesis cites DOI 10.1016/j.sab.2025.107255 but this should be verified against Spectrochimica Acta Part B 2025 issue table-of-contents).
- Drawin, H.W. and Felenbok, P. (1965). 'Data for plasmas in local thermodynamic equilibrium.' Gauthier-Villars. Classic reference for the McWhirter-criterion constant value — needed to verify the MCWHIRTER_C constant in lte_validator.py is consistent with the cubic-deltaE formulation in Cristoforetti 2010 Table 1.

### 6.3 Missing regimes (17)

- High-density regime (n_e > 5e17 cm^-3) where R_D > 1 — Debye-Hückel IPD breaks down (strong-coupling parameter Γ > 0.1), Stark quasi-static approximation fails, the hardcoded R_D=0.5 ion-broadening correction is far out-of-regime. Audit ps-LIBS item 3 mentions this but no remediation step addresses the high-n_e corner.
- Low-density regime (n_e < 5e16 cm^-3, late-time ps-LIBS) where McWhirter is most likely to fail and Stark is comparable to or smaller than Doppler — the audit's Stark fix is unhelpful if McWhirter fails first; the LTE-validity gate has to gate inversion.
- Optically thick limit (τ_0 > 5) — multi-pass radiative transfer, line-shape distortion (self-reversal in two-zone limit), and the breakdown of CSigma. Audit covers self-absorption but only up to mask_threshold=5; for genuinely thick plasmas (Si I 251.611, Ca II H/K) the COG curvature dominates and only TwoZoneBayesianForwardModel is structurally correct — but it has its own Doppler-sigma bug.
- Optically thin limit (τ_0 < 0.1) verification — no test asserts that BayesianForwardModel + IterativeCFLIBSSolver agree to <1% on a synthetic optically-thin spectrum. The audit suggests self-absorption ON/OFF tests but not the opposite limit consistency check.
- Multi-temperature / two-zone regime — the audit names TwoZoneBayesianForwardModel for the Doppler bug only, but never asks whether the two-zone fit is identifiable: with one-zone data, do T_core and T_shell collapse? With two-zone data, are MCMC marginals bimodal?
- Very-low-T regime (T < 0.5 eV, late ps-LIBS) — partition functions polynomial fit may diverge below t_min; the clamp hides the divergence but biases U(T). Direct-sum without IPD is the only correct fallback. The polynomial provider's max(U, g0) floor is a band-aid.
- Very-high-T regime (T > 2 eV, early ps-LIBS or near-pulse) — at T > 20 kK, multi-ionization (charge states 3+, 4+) becomes important; the Saha solver only iterates over neutral+singly+doubly (cflibs/plasma/saha_boltzmann.py has limited stage support). Audit does not check stage-coverage adequacy.
- Strongly mixed regime (Σ light + heavy elements) — Saha equilibrium with both H/C/O (light) and Fe/Cu/Zn (heavy) is a stiff system; Anderson acceleration parameters (β, history depth) in anderson_solver.py were never audited for stiff cases.
- Sub-ng/g trace regime — softmax parameterization, ILR clip-floor 1e-10, and Aitchison metric all become numerically singular when one component → 0; the audit covers this for uncertainty (joint optimizer) but not for the iterative solver's accept/reject of trace concentrations.
- Single-species isobaric regime — pressure-balance n_e update assumes mixed-species. With a single-element sample (pure Fe ablation), the closure equation has only one degree of freedom and the system is structurally underdetermined; no test asserts the right error path.
- Air-shielded vs vacuum vs noble-gas backgrounds — ChemCam (Mars CO2 ~7 mbar) vs lab N2/Ar vs vacuum changes plume confinement, T evolution, and continuum. The audit assumes one regime; cflibs/pds/chemcam.py exists but pressure/background effects are not in any forward-model code path.
- Pre-breakdown vs steady-plasma timescale — ps-LIBS at 1 ps + 1040 nm has Keldysh γ near unity (mixed multiphoton/tunneling regime); the laser-plasma coupling early in the pulse is non-LTE by construction. No code checks gate-onset vs plasma-LTE-formation time.
- Cumulative-shot regime — at MHz repetition rates the crater fills with redeposited material; concentration of the next spectrum is NOT the bulk concentration. Multi-shot averaging assumed implicitly in cflibs/inversion/runtime/streaming.py but no consistency check warns when shot N composition diverges from shot 1.
- Dispersed line + spectrally-overlapped lines regime — when two lines from different species blend within instrument FWHM (typical at low RP ~600), neither deconvolution.py nor identifier modules guarantee correct intensity partitioning; the audit only touches isolation_factor in LineSelector.
- Wavelength-extrapolation regime — manifold sampled across 250-550 nm only (Wave-1 D1 default); spectra outside this range (Mg II 280, Ca II H/K 393, or NIR Ar I 763) silently inherit boundary effects from forward model.
- Cosine-similarity vs L2-distance metric regime — FAISS index uses METRIC_L2 on PCA-projected float32 embeddings. For composition-weighted spectra where intensity scale carries no information, L2 weights bright lines exponentially more than weak lines. Cosine would be correct. The audit completely missed this.
- Hardware-acquisition regime mismatch — fixed integration vs gated detector ICCD/EMCCD; if integration window straddles plume expansion, T and n_e are time-averaged in a way that violates single-zone LTE assumption.

### 6.4 Integration-level concerns (16)

- LineSelector.exclude_resonance is wired correctly in cflibs/cli/main.py (passes detection.resonance_lines), but identify_resonance_lines in cflibs/inversion/physics/line_selection.py:383-408 is a STUB that returns set() with a docstring 'This requires lower level energy which is not in the basic LineObservation dataclass.' So if `detection.resonance_lines` is populated by line_detection.py, the wiring works; if not, every line is treated as non-resonance regardless of exclude_resonance flag. Need to trace detect_line_observations -> resonance_lines population to confirm. This is a smoking-gun integration concern the audit did not name.
- Wave-1 D3 plumbs instrument_fwhm_nm through manifold compute paths, but cflibs/manifold/generator.py uses np.float32 for ALL atomic-data arrays (lines_wl, lines_aki, lines_ek, lines_stark_w, lines_stark_alpha, mass_amu, partition coeffs, IPs). Wavelength differences at the pm level (Stark shift, Doppler at T<1 eV) lose 2-3 digits of precision in float32. The Wave-1 Nyquist fix is dimensionally correct but float32 makes the per-line widening computations numerically degenerate at the wavelength step they tried to resolve.
- Manifold inversion path uses faiss IndexFlatL2 / IndexIVFFlat with METRIC_L2 on np.float32 PCA-embedded spectra (vector_index.py:294/300). The embedding is built by SpectralEmbedder using PCAPipeline. If PCA is fit on training spectra that include intensity scale and continuum, then L2 in PCA space is dominated by continuum amplitude — not composition. Without normalizing each spectrum (cosine or unit-energy) BEFORE PCA, the nearest-neighbor inversion is structurally wrong. The audit's Wave-1 Nyquist fix sharpens the lines, but on a wrong-metric index that does not improve composition accuracy.
- BayesianForwardModel hardcodes apply_self_absorption=False (audit finding) AND TwoZoneBayesianForwardModel has the Doppler factor-of-2 bug AND uses _compute_instrument_sigma path; an MCMC chain that converges and reports tight posteriors is converging to the wrong likelihood — there is no diagnostic that flags 'wrong forward model.'
- LTE validator IS called from iterative.py (Wave-1 / pre-existing code at lines 1223 and 1358) but the result is appended to quality_metrics; need to verify whether the iterative solver actually conditions on it (refuses to converge when LTE fails) or merely reports it (audit suspicion: silent failure pattern). This is one of the cross-cutting 'silent failure with downstream-visible success' patterns the audit named but did not concretely instantiate for LTE.
- Round-trip validation harness (cflibs/validation/round_trip.py) uses the SAME forward model as the inversion's forward model. If forward model has bugs (Doppler factor of 2, no Stark in SpectrumModelJax, no self-absorption), round-trip will always pass — the validation oracle and the system under test share the defect.
- Test fixtures often re-derive the same buggy formula (audit cross-cutting pattern 6). One specific concrete case: tests/test_self_absorption.py test fixture _make_doublet must inject intensities consistent with the physical I ∝ A/λ relation; if it currently uses the buggy I ∝ A·λ^3 to construct the inputs, the test passes the buggy code. This needs an independent test fixture audit before remediation step 6 lands, otherwise the fix breaks the test for the wrong reason.
- SpectrumModel and SpectrumModelJax are independently maintained. SpectrumModel correctly does Stark+Doppler Voigt; SpectrumModelJax silently drops Stark (audit). Manifold and Bayesian paths choose one or the other based on JAX availability — same input, different physics. No parity test gates this divergence.
- polynomial_partition_function silently clamps T to [t_min, t_max] and floors U at g0 (read of partition.py:484-503). At late-time ps-LIBS T may be near t_min, so partition functions return the boundary value — silently. The clamp is correct numerically but the IterativeCFLIBSSolver's iteration uses U(T) as a feedback variable; a constant U at the clamp boundary breaks the feedback loop and converges to spurious local minimum. No warning is emitted when T is clamped.
- Direct-sum partition function applies ionization_potential_depression (IPD) at line 127 with n_e=None silently disabled (delta_chi=0.0 when n_e is None). Callers may or may not pass n_e. If iterative.py's polynomial-provider path bypasses direct-sum entirely, the IPD correction is silently dropped — meaning U(T,n_e) effectively becomes U(T) and Saha equilibrium uses the WRONG partition function for the lowered IP. Need to trace which provider iterative.py uses by default.
- PCAPipeline (cflibs/inversion/common/pca.py) is used by SpectralEmbedder (vector_index.py:68) with center=True, use_jax=False. The choice of n_components is configured in VectorIndexConfig and the user-facing default has not been validated for either ps-LIBS spectrum count or the composition-discrimination problem. PCA on float32 raw spectra further compounds numerical-precision concern above.
- Wavelength calibration runs first in benchmark pipelines (cflibs/benchmark/synthetic_eval.py:231) but its quality gate (_quality_gate_check) returns silently. Downstream ALIAS uses ±0.01 nm tolerance on the calibrated axis — silent calibration miss propagates as line-misidentification with no detection.
- Multi-gate joint fit (joint_multi_gate_fit) takes per-gate observations but each gate must have been preprocessed identically; if any single gate triggered baseline-mode auto-selection differently, the joint chi^2 sums apples and oranges with no warning.
- AtomicSnapshot is registered as JAX pytree (cflibs/core/jax_runtime.py: _register_cflibs_pytrees). Adding stark_A_ion or stark_shift fields per the proposed Wave-2 schema split requires updating _SNAPSHOT_LEAF_FIELDS and re-registering — easy to miss and silently breaks JAX-jitted forward model in non-obvious ways (audit risk on step 2 hints at this but does not name it).
- TwoZoneBayesianForwardModel + TwoZoneMCMCSampler chain reuses the same _AtomicSnapshot fields; the proposed Stark schema change in step 2 will propagate through both single-zone and two-zone forward models, breaking two_zone.py tests in ways that look like 'numerical drift' but are actually correct fixes — needs explicit re-baseline note in the plan.
- EchelleExtractor is wired into examples and tests but no production pipeline path exposes order-merging artifacts to the inversion solver as quality metrics. Order edges (residual flux miscalibration) appear as broad continuum bumps which the NNLS continuum-polynomial may absorb — but the audit's Spectral NNLS basis fix changes that absorption behavior.

### 6.5 Numerical-precision concerns (16)

- Manifold pipeline uses np.float32 / jnp.float32 throughout (cflibs/manifold/generator.py lines 255-365, basis_library.py:211, loader.py extensively, vector_index.py:276/351). At LIBS wavelength step of 0.01-0.05 nm, float32 has ~7 decimal digits relative precision: at λ=400 nm, smallest distinguishable Δλ is ~4e-5 nm ≈ 40 fm. Stark shift (5-50 pm), Doppler sigma at T=0.5 eV (~1-3 pm depending on mass), and the proposed Stark width re-normalization all live in the noise of float32 in this regime. Wave-2 fixes against precision floor: the over-broadening will partially mask, the corrected widths will partially mask, and parity tests will fail with the right answer being below precision.
- CONFIRMED: jax_runtime.py:165-166 returns jnp.complex128 only if x64 enabled AND backend supports float64; otherwise returns jnp.complex64. The Faddeeva (Weideman) Voigt computation uses complex arithmetic; on Metal (Apple Silicon), this silently downgrades to complex64 — pm-scale line shapes become noisy. Audit did not flag this although JAX_BACKEND constant is module-level cached at import time.
- Boltzmann fit (cflibs/inversion/physics/boltzmann.py) uses np.linalg.lstsq with no condition-number check (search confirmed: 0 matches for 'cond' or 'rank'). Highly-clustered E_k values (single-multiplet) give ill-conditioned design matrix; the fit silently returns a slope with huge variance.
- Anderson acceleration in cflibs/plasma/anderson_solver.py uses fixed log_ne bounds [_LOG_NE_MIN, _LOG_NE_MAX]. Hitting the bound during iteration silently clamps and reports convergence — same pattern as the iterative solver Boltzmann-slope-clamp the audit found. No gradient-stability check.
- Saha equilibrium in saha_boltzmann.py uses ionization_potential_lowering with a global mutable warning lock (_MISSING_LEVEL_WARNED, _MISSING_LEVEL_WARNED_LOCK). Concurrent JAX vmap will race on this lock; the warning may be lost in batched manifold generation but the underlying missing-level data is still applied — silent.
- JAX-jit compilation traces with concrete shapes; AtomicSnapshot pytree registration uses _SNAPSHOT_LEAF_FIELDS. Adding a new dynamic field forces recompilation of all jitted functions that close over snapshot — performance regression at first call, not detected by tests.
- MCMC convergence is assessed by _assess_convergence in bayesian/samplers.py; need to verify the R-hat / ESS / divergent-transition thresholds are sane (NumPyro best practice R-hat<1.01, ESS>400, 0 divergences after warmup). Existing thresholds not audited.
- NumPyro NUTS gradient stability: TwoZoneBayesianForwardModel has the Doppler factor-of-2 bug; if the gradient w.r.t. T flows through sqrt(2*T*EV_TO_J/(m*c^2)), the JAX-derivative is well-defined but the wrong magnitude. Tests test_jit_stable and test_gradient_stable will PASS with the wrong physics — gradient stability tests are not physics-correctness tests.
- Joint optimizer (cflibs/inversion/solve/joint_optimizer.py) softmax Jacobian diagonal-only (audit finding 18) affects uncertainty only, BUT the L-BFGS-B descent uses the same Jacobian; if it is also incomplete, the optimizer takes wrong-direction steps near the simplex boundary. Need to verify the optimizer uses scipy's finite-difference Jacobian vs analytical.
- FAISS IVF index requires training (faiss.IndexIVFFlat needs add+train); if training set is small (Wave-1 18432 pixels x small-grid), nlist=100 default may have <50 vectors per cell -> coarse quantization noise dominates nearest-neighbor distances. Default n_lists=100 from VectorIndexConfig has no auto-scale rule.
- PCA reconstruction error in float32 over a 18432-pixel spectrum: roundoff per pixel ~1e-7, summed over 18432 pixels ~1.4e-3 relative — comparable to or larger than the spectroscopic SNR floor for low-intensity lines. Wave-1 manifold resolution increase tightens this constraint.
- instrument convolution (cflibs/instrument/convolution.py:46) uses kernel_size = int(2 * n_sigma * sigma_nm / delta_wl) — at delta_wl=0.005 nm and sigma_nm=0.05 nm, kernel_size = 100, fine; but at delta_wl=0.05 nm and sigma_nm=0.05 nm, kernel_size=10 — under-sampled Gaussian, FFT truncation introduces ringing on narrow lines. No assertion catches this.
- Weideman Voigt approximation (cflibs/radiation/profiles.py: _WEIDEMAN_L=??, _WEIDEMAN_COEFFS) has known accuracy degradation at the wings (|x| > 5) and at extreme a (Voigt damping parameter). For ps-LIBS narrow lines, a is small (~0.01); at line wings, the Weideman expansion can return negative values which would be physically impossible. No clip / safety check confirmed.
- anderson_solver _SOLVER_CACHE is a global mutable; concurrent JAX vmap calls into manifold generation could collide on cache lookup or insertion under pmap. Not thread-safe by inspection.
- Bayesian posterior compression via thinning: number-of-samples-kept × n_chains is parameterized but the audit did not check whether effective sample size after thinning still supports the credible-interval claims downstream (compare_with_bayesian).
- Resolving-power sigma (cflibs/radiation/profiles.py: resolving_power_sigma) computes sigma = λ / (RP × 2.355) by convention; if RP is wavelength-dependent (echelle order edges), this single-value sigma is wrong. No code path supports variable RP across the spectrum.

### 6.6 Plan critique

"The 23-step plan is well-structured by physics family and has a sensible critical-path ordering (Stark widths → Stark schema → Doppler bug → SpectrumModelJax routing → self-absorption wiring), but it has six material gaps that risk landing fixes that look right and fail at integration:\n\n(1) ORDER OF OPERATIONS PROBLEM. Step 1 (Stark normalization fix, ~20× over-broadening) and Step 3 (Doppler factor-of-2 fix, ~1.41× over-broadening) MUST land before Step 12 (ALIAS k_det re-baseline). The reason: the ALIAS macro_F1 of 0.402 is measured under the CURRENT line widths. Fixing Stark and Doppler narrows every line by a factor of ~28× combined in the worst case. ALIAS tolerance windows and k_det normalizations were empirically tuned against the broadened lines. Reverting to paper-pure k_det BEFORE narrowing the lines will give a worse macro_F1, and reverting after narrowing requires a fresh tune. Insert an explicit re-tune step or move step 12 to after step 4 with an explicit acceptance gate.\n\n(2) MANIFOLD CACHE INVALIDATION IS ACKNOWLEDGED BUT NOT GATED. Steps 1, 2, 3, 4, 21 all invalidate the cached manifold. Without an explicit 'regenerate manifold' step in the plan, downstream Bayesian + nearest-neighbor inversion silently uses stale cached spectra with the OLD physics. A concrete validation gate item should be added between groups of physics fixes: 'rebuild reference manifold; smoke-test nearest-neighbor inversion on a known synthetic.'\n\n(3) FLOAT32 NUMERICAL FLOOR (see numerical_precision_concerns). The Stark/Doppler fixes resolve features at the pm scale, but the manifold runs in jnp.float32. Step 1's validation gate (Aragón & Aguilera 2010 FWHM=0.033 nm to 10%) is wide enough that float32 noise won't block it, but Step 21's stark-shift validation (line center shifted by ~-6 pm) is right at the float32 precision floor at 400 nm. Plan should add a precision-floor item: either gate on float64 or add a fp64 reference run for validation tests.\n\n(4) 'LOW EFFORT' TRAPS:\n  • Step 6 (doublet-ratio formula) is correctly marked S, BUT requires re-writing test fixture _make_doublet, and that fixture's correct intensity formula must be derived from a forward-model run, not from the corrected analytical formula — otherwise we are testing the fix against itself. This is a 'tests injection-encode the same bug' cross-cutting concern that the synthesis explicitly listed but didn't carry into the plan.\n  • Step 8 (three coupled fixes: COG λ², resonance_tau_boost, mask_threshold) is also S, but each touches different code paths and bundles three semantically distinct fixes. Risk of one fix being correct and another wrong, both passing the integration test together. Split into three PRs.\n  • Step 10 (silent-failure quality gates) is S but the slope-test condition '|slope|→0' has no defined threshold; choosing it badly turns a silent-pass into a silent-fail elsewhere. Should specify a quantitative threshold (e.g., |slope| < 0.05 / E_max requires T_clamp).\n  • Step 19 (full softmax Jacobian) is S and uncertainty-only, BUT the joint optimizer's L-BFGS-B step direction may use the SAME Jacobian. If yes, this becomes a point-estimate change as well. Verify before marking 'no point-estimate change.'\n\n(5) ACCEPTANCE GATES OFTEN UNDERSPECIFIED FOR REGRESSION DETECTION.\n  • Step 5 wires self-absorption corrector into IterativeCFLIBSSolver + BayesianForwardModel. Validation gate is the Si synthetic round-trip but does not specify: (a) what fraction of existing test_iterative.py tests are expected to re-baseline, (b) whether the default flag is opt-in or opt-out for backward compatibility. Without these the PR will be reverted on first test thrash.\n  • Step 9 (Stark n_e diagnostic primary) requires per-line Stark coefficients populated and verified — this is implicitly a step-1+2 dependency that becomes data dependency, not just code dependency. Plan should list 'populate Stark coefficients for primary diagnostic lines (Hα, Si II 634.7, Mg I 285, Cu I 521) ' as a discrete prep step.\n  • Step 11 (resonance line down-weighting) requires LineObservation.E_lower to be populated by every identifier path. Currently identify_resonance_lines is a STUB (returns empty set). Plan needs to confirm detect_line_observations actually populates resonance_lines via DB lookup; if not, step 11's down-weighting will never fire.\n\n(6) MISSING ITEMS NOT IN PLAN (see recommended_additions). Notably: partition-function wiring (Wave-1 defect family C explicitly excluded — but it is the third-largest physics defect by acknowledged magnitude); manifold metric (L2-on-PCA-of-float32 is structurally wrong for composition retrieval); LTE-validator gate-on-failure behavior; identify_resonance_lines stub; preprocessing baseline audit; wavelength-calibration silent failure.\n\nNet recommendation: the plan is correct in direction but optimistic in scoping. Step 1 should land first as a fully-isolated PR with explicit physics-numerical-validation gates (Aragón & Aguilera 2010 reproduction, not just synthetic test pass). Steps 2-4 then land together because they share the same schema/snapshot mutation risk. Steps 5-8 must be coordinated with test-fixture independence audits. Steps 9-11 are properly Large; do not merge them into 'one wave' as the effort sum suggests."

### 6.7 Recommended additions to plan (17)

- Insert PRE-Step-1: Add ConventionAssertion machinery (a single source-of-truth annotation system on AtomicSnapshot + per-field unit/density-reference comment, plus a runtime assertion at snapshot construction time). Without this, the next convention drift bug will occur within 12 months. Acceptance: every Stark/Doppler/partition-function field in AtomicSnapshot has a docstring-format-checked unit + reference annotation, and a runtime test validates them on import.
- Insert Step 0.5: Partition-function wiring against Barklem & Collet 2016 (Wave-1 defect family C). Files: cflibs/atomic/database_generator.py, cflibs/plasma/partition.py, scripts/build_b_and_c_partition_db.py (new). Acceptance: U(Fe I, 8000 K) within 5% of B&C16 published value; U(Ca II, 10000 K) within 5%; documented divergence vs current polynomial fits in audit/partition_validation_v2.json. Expected lift +0.05 to +0.15 (similar magnitude to Stark). This is the third-largest acknowledged physics defect and the plan completely omits it.
- Insert Step 0.6: Manifold metric audit — replace METRIC_L2 with METRIC_INNER_PRODUCT on L2-normalized spectra (i.e., cosine similarity), or add unit-energy normalization before PCA embedding. Files: cflibs/manifold/vector_index.py SpectralEmbedder, VectorIndexConfig. Acceptance: on a 4-element synthetic test (Fe=0.7, Cu=0.2, Ni=0.1, Cr=0.0 vs Fe=0.7, Cu=0.0, Ni=0.1, Cr=0.2), cosine NN finds the right composition; L2 NN finds the brighter one. Expected lift +0.02 to +0.08.
- Insert Step 1.5: Float64 audit for manifold/coarse_to_fine. Replace jnp.float32 with jnp.float64 in manifold/generator.py atomic-data arrays and document the memory/throughput cost. Without this, Step 21 (Stark shift, ~6 pm) is below precision floor. Files: cflibs/manifold/generator.py, cflibs/manifold/basis_library.py. Acceptance: a 1 pm wavelength feature in a synthetic spectrum survives manifold-encode-decode round-trip to <0.1 pm error.
- Insert Step 5.5: Audit identify_resonance_lines and detect_line_observations to confirm resonance_lines are actually populated by every identifier path. Currently identify_resonance_lines in line_selection.py is a STUB returning empty set; line_detection.py:725-866 needs verification that it populates detection.resonance_lines via DB lookup. Without this, Step 11 (resonance down-weighting) is dead code. Files: cflibs/inversion/physics/line_selection.py, cflibs/inversion/identify/line_detection.py. Acceptance: tests/test_identification_resonance_marking.py asserts that detected Fe I 248.327 (resonance) appears in detection.resonance_lines and a non-resonance Fe I 358.119 does not.
- Insert Step 5.6: LTE-validator gate behavior — confirm IterativeCFLIBSSolver actually fails (or downgrades quality_flag, or refuses to converge) when LTEReport.satisfied=False. Currently the report is computed at iterative.py:1226/1361 and stored but its effect on the convergence loop is unverified. Files: cflibs/inversion/solve/iterative.py. Acceptance: a synthetic spectrum at T=8000K, n_e=1e14 (fails McWhirter) produces a CFLIBSResult with quality_flag='lte_violation' OR converged=False; current behavior likely silent.
- Insert Step 8.5: Preprocessing baseline audit (cflibs/inversion/preprocess/preprocessing.py — estimate_baseline_snip, _als, _percentile, _select_auto_baseline_method). For ps-LIBS, the continuum baseline is fractionally larger than ns-LIBS, so SNIP iteration count + ALS lambda/p defaults matter more. Validation gate: SNIP on a synthetic ps-LIBS spectrum recovers continuum within 2% of ground truth across T=0.5-1.3 eV and gate delays 100-1000 ns. The audit's failure-attribution telemetry should also be checked to see whether 'wrong baseline' shows up in element_confusion.py logs.
- Insert Step 8.6: Wavelength-calibration silent-failure audit. cflibs/inversion/preprocess/wavelength_calibration.py:_quality_gate_check returns silently when calibration is poor (e.g., RMS > 0.01 nm). Convert silent returns into a calibration_failed=True flag on WavelengthCalibrationResult; downstream identifiers should consult this. Acceptance: a synthetic spectrum with a deliberate quadratic warp produces calibration_failed=True; without this flag, ALIAS misidentifies lines silently.
- Insert Step 12.5: Re-tune ALIAS tolerance windows after Stark+Doppler width fixes. Step 12 reverts ALIAS to paper-pure k_det but does not re-tune the wavelength tolerance windows that were widened to accommodate the over-broadened lines. Files: cflibs/inversion/identify/alias.py + tolerance defaults. Acceptance: macro_F1 after physics+ALIAS fixes ≥ Noel 2025 reported 0.564 at RP=600 — if not, deeper integration debugging is needed.
- Insert Step 21.5: Forward-model parity oracle. Build a single canonical synthetic spectrum (Fe+Cu+Ni at T=1 eV, n_e=1e17, RP=600) and an analytical reference computed in float64 with explicit unit annotations. Every PR in Wave-2 must reproduce this oracle to within physical precision. Files: tests/oracles/ps_libs_canonical_v1.json + tests/test_physics_oracle.py. This prevents the round-trip-validation-shares-the-defect failure mode the audit flagged but the plan did not address.
- Insert Step 22.5: ChemCam/Mars CO2 backfill audit. cflibs/pds/chemcam.py + cflibs/inversion/runtime/temporal.py for low-pressure CO2 atmosphere effect on plume confinement; the standard-pressure default may not apply. Validation gate: ChemCam known-target spectrum (calibration tab) recovers expected major-oxide composition within ChemCam's published accuracy (±5% for major oxides).
- Insert Step 23.5: Rust core (native/cflibs-core/src/comb_matching.rs and partition.rs) parity. Currently audit-out-of-scope but used by line_detection.py via _scan_comb_shifts_dispatch_rust. If Rust comb-matching has different shift quantization than Python fallback, results depend on whether HAS_RUST_CORE is True. Acceptance: tests/test_rust_python_comb_parity.py confirms Rust and Python implementations return same CombShiftSummary within numerical tolerance for the canonical test corpus.
- Insert Step 24: NumPyro convergence-threshold audit. cflibs/inversion/solve/bayesian/diagnostics.py + samplers.py _assess_convergence. Set R-hat threshold to 1.01 (not 1.10), ESS threshold to >400, and treat any divergent transitions after warmup as a reparameterization signal (per NumPyro best practice). Acceptance: ConvergenceStatus.UNCONVERGED is reported when R-hat=1.05 (currently passes), ESS=200 (currently passes), or div_transitions>0 (currently silent).
- Insert Step 25: Continuum modeling — add free-free (bremsstrahlung) + free-bound emission to forward model as opt-in. Currently absent; only polynomial nuisance continuum in NNLS identifier. For ps-LIBS at early-gate (<200 ns) the missing continuum biases baseline by ~10-30% of weak-line intensity. New file: cflibs/radiation/continuum.py with planck_bremsstrahlung(T, n_e, λ) + planck_freebound(T, n_e, λ, ξ). Acceptance: a synthetic spectrum at T=15 kK, n_e=1e18, with continuum-on matches the experimentally-known continuum slope for Fe at 250-450 nm within 20%.
- Insert Step 26: Reference compositions ground-truth audit (cflibs/benchmark/reference_compositions.py). Verify the synthetic-corpus oxide closure normalization matches what the inversion produces — without this, even a perfect inversion gets graded as wrong because numerator/denominator definitions diverge.
- Insert Step 27: Test-fixture independence audit. For every test in test_self_absorption.py, test_cdsb.py, test_closure_*.py, test_alias*.py, confirm the test fixture INPUT data is built from an independent physical principle (forward-model run or NIST reference), not from the same buggy formula as the code under test. The audit identified this as cross-cutting pattern #6 but did not enumerate the test files. New CI gate: 'fixture-source-of-truth' annotation required on every parametrize-id fixture.
- Insert Step 28: PCA pre-processing for FAISS — replace default PCA(center=True, no whitening) with PCA(center=True, whiten=True) + L2 normalize spectra before fit. Otherwise principal components capture intensity scale + continuum slope before they capture line patterns, and the metric is dominated by nuisance variance. Files: cflibs/manifold/vector_index.py:SpectralEmbedder.__init__, cflibs/inversion/common/pca.py:PCAPipeline.

---

## 7. Per-topic verified findings (appendix)

### 7.1 peak-id (8 findings)

#### 1. [HIGH/certain/algorithmic_wrong_equation] ALIAS k_det blend uses N_matched instead of N_X (number of theoretical lines), inverting the k_sim/k_shift weighting prescribed by Noel 2025

- **File:** `cflibs/inversion/identify/alias.py:3990-4005`
- **Expected per lit:** Noel et al. 2025 (Spectrochim. Acta B 231, 107255) defines k_det = k_rate * [(1/N_X) * k_shift + ((N_X-1)/N_X) * k_sim], where N_X is 'the number of theoretical lines for element X' (per topic literature). For an element with 50 theoretical lines and 3 matches, this should give k_det ~= k_rate * (49/50)*k_sim — k_sim dominates because k_sim is the intensity-pattern fingerprint that discriminates between elements with similar single-peak positions.
- **Actual in code:** if N_matched > 0:
    N_X = N_matched
    k_det_raw = k_rate * ((1.0 / N_X) * k_shift + ((N_X - 1.0) / N_X) * k_sim)
    # Blend: use geometric mean of raw k_det and P_cov to soften
    # the penalty for many unmatched weak lines
    k_det = math.sqrt(k_det_raw * max(P_cov, 0.01))
- **Citation:** Noel, C.; Neoricic, L.; Alvarez-Llamas, C.; et al. 'Automated line identification for atomic spectroscopy (ALIAS)', Spectrochim. Acta B 231, 107255 (2025), Eq. for k_det. Symbol N_X defined as 'number of theoretical lines for element X' per the topic literature stage.
- **Refutability test:** Construct an ALIAS instance, call _compute_scores then _decide for a synthetic case where N_expected=50, N_matched=3, k_sim=0.9, k_shift=0.7, k_rate=0.5. Paper formula gives k_det_raw = 0.5*((1/50)*0.7 + (49/50)*0.9) ~= 0.448. Code with N_X=N_matched=3 gives 0.5*((1/3)*0.7 + (2/3)*0.9) = 0.5*0.833 = 0.417. tests/test_alias.py::test_N_matched_in_k_det_blend currently *pins* the divergent code behavior; flip the assertion to require N_X=N_expected and re-run.
- **Suggested fix:** Use N_X = N_expected (or N_X = len(fused_lines)) in the k_det blend per the literature, keeping N_matched only as a separate hard-gate variable. Update test_N_matched_in_k_det_blend to reflect the paper formula.
- **Blast radius:** tests/test_alias.py::test_N_matched_in_k_det_blend, tests/test_alias_presets.py macro-F1 baselines, any downstream regression test that pins ALIAS detection thresholds. Hybrid identifier confirmation stage 2 will see different per-element CL distributions.
- **Estimated macro_F1 lift:** unknown — likely modest precision-recall reweighting; paper formula has not been benchmarked against current corpus in this codebase.
- **Verification:** 0/3 skeptics refuted

#### 2. [LOW/certain/implementation_bug] ALIAS docstring cites wrong arXiv ID for the Noel 2025 ALIAS reference

- **File:** `cflibs/inversion/identify/alias.py:1-7`
- **Expected per lit:** Per topic literature inventory: 'NB: arXiv:2501.01057 is an unrelated HPC paper; the genuine ALIAS reference is this SAB article' (Noel et al. 2025, Spectrochim. Acta B 231, 107255, DOI 10.1016/j.sab.2025.107255).
- **Actual in code:** """
ALIAS (Automated Line Identification Algorithm for Spectroscopy) implementation.

Based on Noel et al. (2025) arXiv:2501.01057. The ALIAS algorithm identifies elements
in LIBS spectra through a 7-step process: peak detection, theoretical emissivity
calculation, line fusion, matching, threshold determination, scoring, and decision.
"""
- **Citation:** Noel et al. 2025, Spectrochim. Acta Part B 231, 107255 (verified DOI 10.1016/j.sab.2025.107255 in literature stage).
- **Refutability test:** WebFetch arXiv:2501.01057 abstract and verify it is unrelated to LIBS/ALIAS. Then verify the DOI 10.1016/j.sab.2025.107255 resolves to Noel et al. ALIAS paper.
- **Suggested fix:** Replace 'arXiv:2501.01057' with 'Spectrochim. Acta B 231, 107255 (2025), DOI 10.1016/j.sab.2025.107255' in the alias.py module docstring.
- **Blast radius:** Documentation only. No runtime impact.
- **Estimated macro_F1 lift:** 0 (documentation)
- **Verification:** 0/3 skeptics refuted

#### 3. [MEDIUM/certain/algorithmic_wrong_equation] ALIAS k_det is modified with geometric-mean(k_det_raw, P_cov) and sqrt(N_expected/5) penalty — ad-hoc factors not in canonical paper

- **File:** `cflibs/inversion/identify/alias.py:4000-4010`
- **Expected per lit:** Noel 2025: k_det = k_rate * [(1/N_X)*k_shift + ((N_X-1)/N_X)*k_sim], with no geometric-mean P_cov blend and no sqrt(N_expected/5) prefactor. The confidence multiplier P_maj is supposed to be the only 'coverage' modifier, and it appears in CL = k_det * P_SNR * P_maj * P_ab.
- **Actual in code:** k_det_raw = k_rate * ((1.0 / N_X) * k_shift + ((N_X - 1.0) / N_X) * k_sim)
# Blend: use geometric mean of raw k_det and P_cov to soften
# the penalty for many unmatched weak lines
k_det = math.sqrt(k_det_raw * max(P_cov, 0.01))
...
# Fix 4: N_expected penalty — elements with few expected lines
# get scaled down to prevent 2/3 matches from scoring high.
N_penalty = min(1.0, math.sqrt(N_expected / 5.0)) if N_expected > 0 else 0.0
k_det *= N_penalty
- **Citation:** Noel et al. 2025, Spectrochim. Acta B 231, 107255 — k_det formula (verbatim per topic literature 'key_equations').
- **Refutability test:** Run ALIAS on the Aalto LIBS dataset at RP~600 with current code, then with the ad-hoc modifiers removed (k_det = k_det_raw only). Compare per-element CL distributions and macro_F1. The literature inventory notes the paper-faithful score at RP=600 gives ALIAS macro_F1 ~0.564, and CF-LIBS reports 0.402 — the gap may be partially attributable to these modifications squashing weak-but-genuine detections.
- **Suggested fix:** Make the geometric-mean blend and N_penalty opt-in via constructor flags; default to the paper-pure formula k_det = k_rate * ((1/N_X)*k_shift + ((N_X-1)/N_X)*k_sim). Move the ad-hoc factors into a 'precision_boost_mode' that callers can enable when corpus-tuned.
- **Blast radius:** All ALIAS regression tests pin the modified formula; flipping the default requires re-baselining test_alias_presets.py, tests/test_alias_high_recall_workflow.py.
- **Estimated macro_F1 lift:** unknown — could shift recall up (less suppression of legitimate detections) or down (less suppression of FPs); depends on dataset.
- **Verification:** 0/3 skeptics refuted

#### 4. [HIGH/high/implementation_bug] Spectral NNLS polynomial continuum basis is monomials wl_norm**deg, all non-negative and non-decreasing on [0,1]; NNLS constraint forces the modeled continuum to be non-decreasing in wavelength

- **File:** `cflibs/inversion/identify/spectral_nnls.py:525-553`
- **Expected per lit:** Wang et al. 2014 (IOP Conf. Ser. Earth Environ. Sci. 17, 012208): I_obs(lambda) ~= sum_i c_i * B_i(lambda; T, n_e) + P(lambda) with c_i >= 0. The continuum P(lambda) is an additive term and must be free to take any shape consistent with bremsstrahlung; in LIBS spectra the continuum typically *decreases* with wavelength (Kramers/Maxwellian bremsstrahlung shape, dropping monotonically across visible-UV). Standard practice is a Chebyshev or Legendre basis with unconstrained coefficients, or explicit positive/negative polynomial-pair splits, to keep NNLS while allowing a decreasing continuum.
- **Actual in code:** if self.continuum_degree >= 0:
    # Normalized wavelength for numerical stability
    wl_min, wl_max = wavelength[0], wavelength[-1]
    wl_norm = (wavelength - wl_min) / max(wl_max - wl_min, 1e-10)

    poly_cols = []
    for deg in range(self.continuum_degree + 1):
        col = wl_norm**deg
        # Normalize polynomial columns to similar scale as basis
        col_norm = np.sum(np.abs(col))
        if col_norm > 1e-20:
            col /= col_norm
        poly_cols.append(col.reshape(1, -1))

    components.append(np.vstack(poly_cols))
- **Citation:** Wang, W.; Ayhan, B.; Kwan, C.; Qi, H.; Vance, S. 'A Novel and Effective Multivariate Method for Compositional Analysis using Laser Induced Breakdown Spectroscopy', IOP Conf. Ser. Earth Environ. Sci. 17, 012208 (2014) — establishes the unmixing model I_obs = sum c_i B_i + P(lambda) but the continuum is meant to be unconstrained (or non-negative AND shape-flexible).
- **Refutability test:** Generate a synthetic spectrum at T=8000 K, n_e=1e17, with pure Fe + a known bremsstrahlung continuum that decreases monotonically across 250-450 nm. Run SpectralNNLSIdentifier and inspect the polynomial coefficients: all four (deg 0..3) will be near-zero or zero because no non-negative monomial combination on [0,1] can match a decreasing function. The residual will absorb the continuum mismatch, and Fe coefficient will be systematically biased high.
- **Suggested fix:** Replace monomial basis with: (a) split polynomial into positive AND negative copies — append [col, -col] for each degree so NNLS picks the sign, or (b) Chebyshev/Legendre basis with the continuum block solved via free-sign LS (subtract its fit before NNLS for elements). Document continuum_degree=3 as ns-LIBS default and recommend continuum_degree=2 or even -1 (disabled) for ps-LIBS.
- **Blast radius:** All HybridIdentifier callers (which use SpectralNNLSIdentifier in stage 1), BIC pruning (which inherits the continuum block in basis_matrix), and any benchmark using algorithm='spectral_nnls' or 'hybrid'. Tests pinning current per-element coefficient values will need re-baselining.
- **Estimated macro_F1 lift:** unknown — likely substantial for ps-LIBS where continuum mis-fit currently leaks into element coefficients; literature Gragston 2020 notes ps-LIBS has narrow continuum, making this mismatch worse.
- **Verification:** 0/3 skeptics refuted

#### 5. [MEDIUM/high/implementation_bug] Spectral NNLS uses unconstrained (A^T A)^-1 diagonal for coefficient uncertainty, ignoring the active-set boundary structure of NNLS

- **File:** `cflibs/inversion/identify/spectral_nnls.py:458-467`
- **Expected per lit:** Lawson & Hanson 1974 (Solving Least Squares Problems, Ch. 23) — NNLS produces a constrained estimate; its variance is NOT given by (A^T A)^-1 diag. For inactive coefficients (c_i = 0 at the constraint boundary), the OLS variance is meaningless. Correct treatment requires conditioning on the active set: sigma_active ~ chol(A_active^T A_active)^-1, with inactive coefficients excluded from the residual degrees-of-freedom count.
- **Actual in code:** # Coefficient uncertainties from (A^T A)^-1 diagonal
AtA = A @ A.T
try:
    AtA_inv_diag = np.diag(np.linalg.inv(AtA + 1e-12 * np.eye(len(AtA))))
    sigma_coeffs = np.sqrt(np.maximum(residual_var * AtA_inv_diag[:n_elements], 0.0))
except np.linalg.LinAlgError:
    logger.debug("AtA inversion failed; using fallback uncertainty estimate (0.1)")
    sigma_coeffs = np.ones(n_elements) * 0.1

snr = element_coeffs / np.maximum(sigma_coeffs, 1e-10)
- **Citation:** Lawson, C. L.; Hanson, R. J. 'Solving Least Squares Problems', Prentice-Hall (1974), Ch. 23. Standard NNLS texts treat the boundary case explicitly — see also Bro & De Jong 1997 (J. Chemometrics 11, 393) on active-set variance for fast NNLS.
- **Refutability test:** Generate a synthetic spectrum with 5 elements where only 3 are truly present. Run SpectralNNLSIdentifier and inspect snr[i] for the 2 absent elements. The OLS-derived sigma_coeffs[i] will give an unrealistically tight uncertainty bound for the boundary-zero coefficients, potentially yielding spurious 'low SNR' detections at the noise floor. The correct active-set treatment would refuse to assign a Gaussian SNR to those coefficients.
- **Suggested fix:** After NNLS, compute sigma only for the active set (coefficients > epsilon) via (A_active^T A_active)^-1 diag, and assign a separate detection_floor metric for inactive elements based on the projected residual onto each excluded basis vector (delta_RSS test).
- **Blast radius:** HybridIdentifier stage-1 detection set, BIC pruning starting point (element_coefficients > 0 active mask), and detection_snr threshold behavior. Tests/test_spectral_nnls.py exercises specific SNR values.
- **Estimated macro_F1 lift:** unknown — primarily affects edge cases near detection threshold; ps-LIBS lower noise floor (~1% vs ns ~5% per literature) makes the current ill-defined SNR more variable across datasets.
- **Verification:** 1/3 skeptics refuted

#### 6. [LOW/certain/implementation_bug] BIC pruning accepts noise_variance argument but never uses it; formula is unknown-variance Gaussian regardless of caller input

- **File:** `cflibs/inversion/identify/model_selection.py:40-64, 119-153`
- **Expected per lit:** Schwarz 1978 (Annals of Statistics 6(2), 461-464): BIC = -2 ln L_max + k * ln(n). For Gaussian residuals with KNOWN variance sigma^2, -2 ln L_max = n * ln(2*pi*sigma^2) + RSS/sigma^2 (constants drop); BIC = RSS/sigma^2 + k*ln(n). For UNKNOWN variance (sigma^2 estimated as RSS/n), it reduces to BIC = n*ln(RSS/n) + k*ln(n). Caller passing noise_variance should obtain the known-variance form; the current code uses unknown-variance regardless.
- **Actual in code:** def bic_prune_elements(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    element_list: List[str],
    element_coefficients: np.ndarray,
    noise_variance: float,  # <-- accepted but never used
    ...
):
    ...
    # _compute_bic uses unknown-variance form regardless of noise_variance
    return n * np.log(rss / n) + k * np.log(n)
- **Citation:** Schwarz, G. 'Estimating the Dimension of a Model', Ann. Statist. 6(2), 461-464 (1978).
- **Refutability test:** Call bic_prune_elements twice with the same inputs but noise_variance=1e-4 and noise_variance=1.0; the returned bic_initial and bic_final values will be identical, proving the parameter has no effect.
- **Suggested fix:** Either (a) implement the known-variance branch: when noise_variance > 0, compute BIC = rss/noise_variance + k*ln(n); or (b) remove the unused parameter from the signature and document the unknown-variance assumption.
- **Blast radius:** Callers passing noise_variance get silent neglect — no observable behavior change but the function signature is misleading. Docs in pyproject/api should reflect.
- **Estimated macro_F1 lift:** 0 (no behavior change in current calls); if known-variance branch is implemented and caller estimates correlate with reality, could tighten BIC pruning.
- **Verification:** 0/3 skeptics refuted

#### 7. [MEDIUM/certain/algorithmic_wrong_equation] ALIAS imposes ad-hoc hard gates (N_matched < 3 unless N_expected <= 4 with strict-all-matched, or N_expected <= 1) that are not in the canonical Noel 2025 confidence-level formula

- **File:** `cflibs/inversion/identify/alias.py:4022-4037`
- **Expected per lit:** Noel 2025: CL = k_det * P_SNR * P_maj * P_ab, with detection by threshold on CL alone (no separate N_matched count gate beyond what is implicit via N_X-weighting in k_det). The probabilistic factors P_SNR, P_maj, P_ab already encode count-based confidence; an additional hard gate double-penalizes elements with few but strong lines (e.g., resonance-dominated alkali metals).
- **Actual in code:** # Hard gate — reject if too few lines matched.
# At RP<1000, matching 2 lines by chance is trivial for elements
# with few expected lines (Na, K). Require enough matches to be
# statistically meaningful.
if N_expected <= 1:
    # Single-line elements (H-alpha): 1 match is sufficient
    pass
elif N_expected <= 4:
    # Sparse elements (Na, K, Li): require ALL lines matched
    # AND elevated CL to pass — chance matching 2/2 is too easy
    if N_matched < N_expected:
        CL = 0.0
else:
    # Normal elements: require at least 3 matched lines
    if N_matched < 3:
        CL = 0.0
- **Citation:** Noel et al. 2025, Spectrochim. Acta B 231, 107255 — confidence-level decision formula does not contain explicit N_matched thresholds beyond what is encoded in the score factors.
- **Refutability test:** Synthesize a spectrum with strong Na D resonance lines (588.99 nm + 589.59 nm) at high SNR but no other Na lines in band. Paper formula would yield CL = k_det * P_SNR * P_maj * P_ab > threshold and detect Na. Current code: N_expected=2, N_matched=2 -> passes the elif branch (N_matched==N_expected) but only because the doublet is in band; reduce wavelength range to include only 588.99 -> N_expected=1, passes. But if N_expected=3 and only 2 strongest are matched -> CL=0 even at high CL otherwise.
- **Suggested fix:** Expose hard-gate parameters (min_matched_lines_normal=3, min_matched_lines_sparse_all=True) as constructor args with paper-pure defaults (no gates), so callers can opt into RP-specific gates for ps-LIBS or low-RP corpora.
- **Blast radius:** Tests/test_alias.py pins these gates (test_N_matched_in_k_det_blend asserts CL=0 when N_matched=1 with N_expected=10). Flipping the default behavior requires re-baselining.
- **Estimated macro_F1 lift:** unknown — likely raises recall on alkali metals at the cost of some FPs; for ps-LIBS where line forests are thinner, this should be a net gain.
- **Verification:** 0/3 skeptics refuted

#### 8. [MEDIUM/high/algorithmic_wrong_equation] Comb fingerprint denominator is min(len(teeth), fingerprint_top_k=10) creating asymmetric scoring: elements with >10 teeth get same denominator as elements with exactly 10

- **File:** `cflibs/inversion/identify/comb.py:1125-1157`
- **Expected per lit:** Gajarska et al. 2024 (J. Anal. At. Spectrom. 39, 3151-3161) defines the comb fingerprint as a coverage-weighted active-tooth score per element. Standard form: score = (n_active / n_total) * mean(active_correlations), or score = sum(active_correlations) / n_total — *not* a denominator that caps independent of total line count. The cap at fingerprint_top_k creates non-monotonic scaling: an element with 12 teeth (10 active) gets sum(top-10)/10; an element with 8 teeth (8 active) gets sum(top-8)/8.
- **Actual in code:** active_correlations = sorted(
    (float(t["best_correlation"]) for t in active_teeth),
    reverse=True,
)
denominator = min(len(teeth), self.fingerprint_top_k)
top_correlations = active_correlations[:denominator]
fingerprint = sum(top_correlations) / denominator
return float(np.clip(fingerprint, 0.0, 1.0))
- **Citation:** Gajarska et al. 2024, J. Anal. At. Spectrom. 39, 3151-3161 (DOI 10.1039/D4JA00247D) — comb fingerprint formula is coverage-weighted, not capped.
- **Refutability test:** Compute fingerprint for two synthetic cases: (A) element with 12 teeth, 10 active at correlation 0.6; (B) element with 8 teeth, 8 active at correlation 0.6. Current code: A = sum(top-10 correlations of 0.6) / 10 = 6.0/10 = 0.6; B = sum(top-8 correlations of 0.6) / 8 = 4.8/8 = 0.6. Same score despite different coverage (10/12=0.83 vs 8/8=1.0). Paper-faithful coverage-weighted score would assign B higher.
- **Suggested fix:** Change denominator to len(teeth) (full coverage denominator) or remove the cap entirely. Optionally keep fingerprint_top_k as a separate 'max-active-considered' filter to suppress noisy weak teeth, but compute the denominator from the full template count.
- **Blast radius:** All CombIdentifier callers, tests/test_comb_recall.py, tests/test_comb_precision.py, comb-only benchmarks. Affects element ranking when multiple elements pass min_correlation threshold.
- **Estimated macro_F1 lift:** unknown — corpus dependent; likely improves recall on elements with many teeth (Fe, V) and reduces FPs on elements with very few teeth where the current denominator already approaches len(teeth).
- **Verification:** 1/3 skeptics refuted

### 7.2 self-absorption (7 findings)

#### 1. [CRITICAL/certain/missing_feature] Self-absorption corrector and CDSB plotter are dead code in the inversion pipeline (never called from any solver)

- **File:** `cflibs/inversion/solve/iterative.py + cflibs/inversion/solve/bayesian/forward.py:iterative.py 1004-1252 (entire _solve_python loop); bayesian/forward.py 331-345`
- **Expected per lit:** Bulajic et al. 2002 Spectrochim. Acta B 57, 339 prescribes the recursive self-absorption correction as an OUTER LOOP around the CF-LIBS Boltzmann/Saha/closure iteration: after each plasma-state update the corrected intensities are recomputed via I_true = I_obs / f(tau), then the Boltzmann plot is refit on the corrected line list, then tau is recomputed from the new plasma state, and the loop iterates until both tau AND the plasma state converge. Cristoforetti & Tognoni 2013 (Spectrochim. Acta B 79-80, 63) gives the CDSB analogue. Poggialini 2023 J. Anal. At. Spectrom. 38, 1751 confirms this is the standard CF-LIBS architecture.
- **Actual in code:** Serena's find_referencing_symbols on `SelfAbsorptionCorrector` and `CDSBPlotter` across cflibs/ returns ONLY (a) the class definitions themselves, (b) public re-exports in cflibs/inversion/__init__.py lines 90/94/275/277, and (c) test files. No code under cflibs/inversion/solve/, cflibs/inversion/identify/, or cflibs/cli/ instantiates either class. The iterative solver loop (iterative.py:_solve_python lines 1079-1221) performs: partition funcs -> Saha correction -> common-slope Boltzmann fit -> closure -> pressure balance update -> convergence check, with NO call to any self-absorption correction. The Bayesian forward model (bayesian/forward.py:331-342) passes `apply_self_absorption=False` to the JAX forward kernel.
- **Citation:** Bulajic et al. 2002, Spectrochim. Acta B 57, 339, sec. 'Theoretical' and 'Iterative procedure'; Poggialini et al. 2023, J. Anal. At. Spectrom. 38, 1751, sec. 3 (CF-LIBS workflow). Both verified via NotebookLM CF-LIBS notebook.
- **Refutability test:** Inject synthetic ground-truth Si lines (251.611, 288.158 nm) generated from a thick (tau~5) plasma with the forward model and run IterativeCFLIBSSolver. Compare the recovered Si concentration with and without manually injecting a call to SelfAbsorptionCorrector.correct between common-slope fit and closure. If the recovered concentration is identical (and biased low by tau-suppression of resonance-line intensities), the corrector is confirmed dead.
- **Suggested fix:** Wire SelfAbsorptionCorrector.correct (or CDSBPlotter.fit) into IterativeCFLIBSSolver._solve_python after _apply_saha_correction but before _fit_common_boltzmann_plane, using the current iteration's (T, n_e, concentrations) as plasma state. Make it opt-in via a constructor flag for backward compatibility, and update the configured Bayesian forward model to apply_self_absorption=True when path_length_m > 0.
- **Blast radius:** cflibs/inversion/solve/iterative.py (loop body change); cflibs/inversion/solve/bayesian/forward.py (flag flip); tests/inversion/ (new end-to-end thick-plasma round-trip tests); identify/alias.py (resonance-line filter behavior may shift since residual SA bias on Si I 251 etc. would be removed); validation/ NIST and synthetic-corpus benchmarks should improve macro-F1 on Si-, Fe-, Al-heavy matrices.
- **Estimated macro_F1 lift:** Aguilera 2014 / Aguilera & Aragon 2004 demonstrate 5-15% concentration accuracy improvement for resonance-rich elements (Si, Fe, Al, Mg) when SA correction is applied; macro_F1 impact unknown but likely +0.03 to +0.10 in the vrabel2020 soil regime where Si is the most-missed element per the 2026-05-13 autodiscovery audit.
- **Verification:** 0/3 skeptics refuted

#### 2. [HIGH/certain/algorithmic_wrong_equation] Doublet-ratio theoretical intensity uses wavelength^3 (line strength) instead of 1/wavelength (emission intensity); wrong by factor (lambda_1/lambda_2)^4

- **File:** `cflibs/inversion/physics/self_absorption.py:313-329`
- **Expected per lit:** For two emission lines sharing the same upper level (n_k common), the optically-thin INTENSITY ratio is I_1/I_2 = (A_1/A_2) * (lambda_2/lambda_1). This comes from I_ji = (hc / 4 pi lambda_ji) * n_k * A_ki (Bulajic 2002 eq. 1; Aragon & Aguilera 2014 eq. 1; Cowan 1981 ch. 14): the per-photon energy hc/lambda scales as 1/lambda, the photon emission rate is n_k * A_ki, and the prefactor 1/(4 pi) is geometric. The (g_k A lambda^3) combination is the LINE STRENGTH S (proportional to gf), not the emission intensity. Confusing these differs by (lambda_1/lambda_2)^4 — e.g. for an Fe I 358.12 / 374.95 nm pair it is a 17% bias on r_theory.
- **Actual in code:** def _theoretical_doublet_ratio(line1, line2):
    return (line1.g_k * line1.A_ki * line1.wavelength_nm**3) / (
        line2.g_k * line2.A_ki * line2.wavelength_nm**3
    )

The docstring even acknowledges 'we use g_k * A_ki * lambda**3 as a stand-in (same upper level => g_k_1 == g_k_2, so this reduces to A * lambda**3)'. The test fixture tests/test_self_absorption.py:1637 hard-codes the same wrong formula: `r_theory = (lam1**3) / (lam2**3)` and synthesizes intensities self-consistently with it, so the unit test PASSES even though the formula is unphysical.
- **Citation:** Bulajic et al. 2002, Spectrochim. Acta B 57, 339, eq. (1)-(2); Aragon & Aguilera 2014, J. Quant. Spectrosc. Radiat. Transfer 149, 90, eq. (1). Both verified via NotebookLM CF-LIBS notebook.
- **Refutability test:** Take a real Fe I doublet from common upper level — e.g. NIST ASD lines for the Fe I 3p^6 3d^7 4s a^5D term yielding 358.119 and 374.948 nm transitions from a common upper level (a^5F_5 at ~3.418 eV). Synthesize an optically-thin spectrum at T=8000K, n_e=1e17 with `forward_model(apply_self_absorption=False)`. Compute the integrated-intensity ratio I(358)/I(374) directly from the synthesized spectrum and compare against `_theoretical_doublet_ratio` — they should disagree by ~(374/358)^4 - 1 ~= 21%.
- **Suggested fix:** Replace the lambda^3 ratio with (A_1 * lambda_2) / (A_2 * lambda_1). For same upper level g_k cancels regardless. Update tests/test_self_absorption.py:_make_doublet to inject intensities using the correct optically-thin formula. Optionally cite Aragon & Aguilera 2014 eq. 1 in the docstring in place of the unverifiable 'Pace et al. 2025'.
- **Blast radius:** cflibs/inversion/physics/self_absorption.py:correct_via_doublet_ratio (recovered tau changes by (lambda ratio)^4 in r_meas/r_theory normalization); cross_check_with_doublets (different agreement_sigma); tests/test_self_absorption.py:_make_doublet + TestDoubletRatioCorrection (test fixture must change to use correct formula); estimate_optical_depth_from_intensity_ratio (caller-supplied theoretical_ratio assumed to be correct, so no internal change needed).
- **Estimated macro_F1 lift:** Indirect: while the doublet method is not currently wired into the inversion pipeline (see finding above), fixing it before re-enabling avoids systematic bias on resonance-doublet pairs. Order-of-magnitude estimate: per-element error <5% for typical CF-LIBS doublets where lambda_1/lambda_2 ~1.05; up to 20% for the widest doublets used in CF-LIBS literature.
- **Verification:** 0/3 skeptics refuted

#### 3. [HIGH/certain/algorithmic_wrong_equation] CDSB initial-tau scales with electron density n_e instead of species number density n_s

- **File:** `cflibs/inversion/physics/cdsb.py:456-504 (especially 481-494)`
- **Expected per lit:** Optical depth tau = (pi e^2 / m_e c) * f_lu * n_lower * L * phi(nu_0). The lower-level population n_lower = n_s * (g_i / U) * exp(-E_i/kT) is proportional to the SPECIES number density n_s = C_s * N_total of the absorbing atom, NOT to electron density. In a typical LIBS plasma at low ionization fraction n_e and n_s can differ by 10-1000x (e.g. neutral Si at 6000K with 0.001 ionization fraction has n_Si I >> n_e). Cristoforetti & Tognoni 2013 eq. (9)-(11) and Bulajic 2002 eq. (2) make this explicit (n_l for species l, not n_e).
- **Actual in code:** n_e_ref = 1e17  # Reference electron density (cm^-3)
...
# Density scaling (linear with n_e at moderate densities)
density_factor = n_e / n_e_ref
...
tau = (
    self.initial_tau_base
    * population_factor
    * line_strength
    * density_factor
    * length_factor
)

The `n_e` is the only density passed to `_estimate_initial_tau` from `CDSBPlotter.fit` (signature `n_e: float`). No species density or total number density is available to the function. By contrast, `SelfAbsorptionCorrector._estimate_optical_depth` (line 998) correctly does `n_s = C_s * total_n_cm3`.
- **Citation:** Cristoforetti & Tognoni 2013, Spectrochim. Acta B 79-80, 63, eq. (9)-(11); Bulajic et al. 2002 Spectrochim. Acta B 57, 339, eq. (2)-(3). Verified via NotebookLM and cited in Poggialini 2023 sec. 3.1.1.
- **Refutability test:** Inject a synthetic optically-thick Fe I 372.0 nm line at fixed n_Fe=1e15 cm^-3 (neutral Fe density) but vary n_e from 1e15 to 1e18 cm^-3. The physically correct tau should be CONSTANT (n_lower depends only on n_Fe and T, not on n_e). The current code returns tau that scales linearly with n_e, which is wrong by factor 1000 across the swept range.
- **Suggested fix:** Extend CDSBPlotter.fit signature to accept either (concentrations + total_n_cm3) or per-line species_density_cm3, and pass the species density (n_s = C_s * N_total or directly N_s) to _estimate_initial_tau in place of n_e. The hardcoded n_e_ref=1e17 reference should be replaced with a species-density reference (e.g., 1e16 atoms/cm^3 of the analyte species).
- **Blast radius:** cflibs/inversion/physics/cdsb.py (CDSBPlotter.fit, _estimate_initial_tau, _update_tau_estimates signatures); any future caller will need to pass concentrations + N_total; existing tests in tests/test_cdsb.py and tests/test_vrabel2020_cdsb.py will need fixture updates.
- **Estimated macro_F1 lift:** Unknown — code path is currently dead (CDSBPlotter not invoked from inversion solver per finding above), so fixing it has no direct F1 impact until CDSB is wired into the pipeline. After wiring, expected to improve concentration accuracy on low-ionization elements (alkali, alkaline earth at low T) by removing the spurious n_e-dependence.
- **Verification:** 0/3 skeptics refuted

#### 4. [HIGH/certain/implementation_bug] CDSB tau update across temperature iterations is a no-op for the partition-function term (U_old == U_new always), defeating Cristoforetti & Tognoni 2013 iteration

- **File:** `cflibs/inversion/physics/cdsb.py:506-550 (especially 537-548)`
- **Expected per lit:** Cristoforetti & Tognoni 2013 (sec. 3) prescribe iterating the CDSB plot: at each iteration the temperature is updated from the Boltzmann fit of the corrected lines, then tau for each line is REcomputed via tau ∝ n_lower(T) = n_s * (g_i / U(T)) * exp(-E_i / kT). Both the Boltzmann factor AND the partition function U(T) change with T. The whole point of the iteration is that as T moves, U(T) and exp(-E_i/kT) shift the column-density assignment of each line, breaking the degeneracy.
- **Actual in code:** # Population ratio scaling
# For partition function: approximate as constant or use power law
# U(T) ~ T^0.5 for many atoms, but let's use provided values
U_old = partition_funcs.get(obs.element, 25.0)
U_new = partition_funcs.get(obs.element, 25.0)  # Same, could interpolate

if old_T_eV > 0 and new_T_eV > 0:
    # Boltzmann factor ratio
    delta_E_factor = np.exp(obs.E_i_ev * (1.0 / old_T_eV - 1.0 / new_T_eV))
    U_ratio = U_old / U_new if U_new > 0 else 1.0
    scale_factor = U_ratio * delta_E_factor

The `partition_funcs` dict is whatever the caller passed to `fit()` at the START — it is never re-evaluated at the new temperature. Therefore U_old == U_new == constant for the whole iteration, so U_ratio == 1.0 identically. For resonance lines (E_i ~ 0), delta_E_factor is also 1.0, so scale_factor == 1.0 for ALL resonance lines — meaning new_tau[wl] == old_tau[wl] every iteration and the loop converges trivially with no actual correction. For non-resonance lines only the Boltzmann factor moves, but the partition-function temperature dependence (which can be 2-3x at LIBS temperatures per the C topic in this audit) is silently dropped.
- **Citation:** Cristoforetti & Tognoni 2013, Spectrochim. Acta B 79-80, 63, sec. 3 'CDSB iterative procedure'; the equations there assume U(T) is re-evaluated at each new T. Code comment on line 538 explicitly acknowledges the no-op ('Same, could interpolate').
- **Refutability test:** Construct a CDSBLineObservation list containing 5 Ca I lines (mix of resonance E_i=0 and excited E_i=1.9 eV) with intensities synthesized from a hot ground-truth (T=10000K) using true Ca partition function U(T) from cflibs.plasma.partition. Pass the COOL initial_T_K=4000K and the COOL partition function (U(4000)=2.5) and let CDSBPlotter.fit iterate. Track tau_history and temperature_history. Result: tau will be ~stagnant after iteration 1 because the partition function never updates, and the final T will be biased toward the initial estimate.
- **Suggested fix:** Replace the constant `partition_funcs` dict with a per-iteration callable (e.g., accept `partition_func_fn: Callable[[str, float], float]` in `CDSBPlotter.fit`), evaluate U_new = partition_func_fn(element, new_T_K) at each iteration, and pipe through the existing _evaluate_partition_function machinery from inversion/solve/iterative.py. Alternative: accept the AtomicDatabase and call get_partition_function(element, ionization_stage, T_K) directly.
- **Blast radius:** cflibs/inversion/physics/cdsb.py:CDSBPlotter signature + _update_tau_estimates; tests/test_cdsb.py and tests/test_vrabel2020_cdsb.py fixtures (will need to pass a partition function callable rather than a static dict); cflibs/inversion/__init__.py public API surface (CDSBPlotter signature change).
- **Estimated macro_F1 lift:** Unknown — currently dead code in the pipeline; would matter when CDSB is wired in. Given that the C-topic audit shows partition functions disagree with direct-sum by 2-3x at LIBS T, fixing this would reduce CDSB tau bias by similar factor.
- **Verification:** 0/3 skeptics refuted

#### 5. [HIGH/certain/algorithmic_wrong_equation] COG _estimate_optical_depth missing one factor of wavelength: uses lambda/Delta_lambda instead of lambda^2/Delta_lambda

- **File:** `cflibs/inversion/physics/self_absorption.py:2145-2196 (especially 2186-2191)`
- **Expected per lit:** The line-center optical-depth (Doppler profile) is tau_0 = sigma_0 * N where sigma_0 = (pi e^2 / m_e c) * f * phi(nu_0), and phi(nu_0) = 1/(sqrt(pi) * Delta_nu_D). Converting from frequency to wavelength via Delta_nu_D = (c/lambda^2) Delta_lambda_D yields sigma_0 = (pi e^2 / m_e c^2) * f * lambda^2 / (sqrt(pi) * Delta_lambda_D) = 8.85e-13 * f * lambda^2(cm)/(sqrt(pi) * Delta_lambda_D(cm)) [cm^2]. The constant 8.85e-13 cm^2 = pi e^2/(m_e c^2). Note the lambda^2 factor (not lambda). This is the standard Mihalas / Hutchinson / Konjevic result reproduced in Poggialini 2023 eq. (4) and the docstring of `SelfAbsorptionCorrector._estimate_optical_depth` already gets this right (line 1078 uses `lambda^2` implicitly via the frequency-domain factors).
- **Actual in code:** # Cross section at line center for Doppler profile
# sigma_0 = (pi * e^2 / m_e * c) * f * lambda / (sqrt(pi) * delta_lambda_D)
# = 8.85e-13 * f * lambda / (sqrt(pi) * delta_lambda_D)
const = 8.85e-13 / np.sqrt(np.pi)  # cm^2

if delta_lambda_cm > 0:
    sigma_0 = const * f_approx * lambda_cm / delta_lambda_cm
    tau_0 = sigma_0 * column_density

The formula has lambda_cm (one factor) in the numerator, missing a second lambda. The docstring above the constant also mixes the (pi e^2 / m_e c) and (pi e^2 / m_e c^2) constants — those differ by a factor of c. The numeric 8.85e-13 is correctly pi e^2/(m_e c^2) but the line-2185 comment claims it's pi e^2/(m_e c).
- **Citation:** Mihalas 1978 Stellar Atmospheres eq. 9-49; Hutchinson 2002 Principles of Plasma Diagnostics eq. 5.13; Poggialini et al. 2023 J. Anal. At. Spectrom. 38, 1751, eq. (4). The correct formula already appears in cflibs/inversion/physics/self_absorption.py:_estimate_optical_depth lines 1072-1078 of the SelfAbsorptionCorrector class.
- **Refutability test:** For a Si I 251.6 nm line with f=0.34, N_l=1e15 cm^-2, Doppler width 0.001 nm at T=10000K: the correct tau_0 ~ 8.85e-13 * 0.34 * (2.516e-5)^2 / (1.77e-1 * 1e-10) = 0.30. The CurveOfGrowthAnalyzer._estimate_optical_depth gives 8.85e-13 * 0.34 * 2.516e-5 / (1.77e-1 * 1e-10) ~ 1.2e-3 — underestimate by factor lambda ~ 2.5e-5. Direct unit test: call CurveOfGrowthAnalyzer._estimate_optical_depth(column_density=1e15, max_log_gf=log10(0.34), doppler_width_nm=0.001, wavelength_nm=251.6) and assert tau_0 in [0.1, 1.0].
- **Suggested fix:** Change line 2191 to `sigma_0 = const * f_approx * (lambda_cm**2) / delta_lambda_cm`. Update docstring on line 2186 to use `lambda^2` and clarify that 8.85e-13 = pi e^2/(m_e c^2).
- **Blast radius:** cflibs/inversion/physics/self_absorption.py:CurveOfGrowthAnalyzer._estimate_optical_depth and downstream get_correction_factors / correct_with_cog (currently dead in pipeline but used by tests/test_self_absorption.py:TestCOGCorrectionFactors). Existing tests may break since they verify the buggy magnitudes; they should be updated to the physically correct values.
- **Estimated macro_F1 lift:** Unknown — COG path is also not wired into the inversion solver, so direct F1 impact is zero until COG is enabled. After enabling, the bug currently UNDERESTIMATES tau by ~5 orders of magnitude for typical LIBS wavelengths (lambda ~ 1e-5 cm), so all COG-based corrections silently no-op.
- **Verification:** 0/3 skeptics refuted

#### 6. [MEDIUM/high/regime_mismatch] Default mask_threshold=3.0 discards the very lines self-absorption correction is meant to rescue (literature uses tau<=5)

- **File:** `cflibs/inversion/physics/self_absorption.py:650-678 (constructor); 789-801 (correct loop)`
- **Expected per lit:** Cristoforetti & Tognoni 2013 explicitly state their CDSB method is 'valid for optical depths 0.1 < tau < ~5' and demonstrate quantitative recovery up to tau ~ 5. Rezaei et al. 2020 (Spectrochim. Acta B 169, 105878) survey lists per-method validity windows: Bulajic recursive 0.1 < tau < 4; CDSB 0.1 < tau < 5; SAF-LIBS 0.5 < tau < 10. John & Anoop 2023 (RSC Adv. 13, 29613) numerically show recoverable accuracy at tau ~ 5 for resonance lines at L=0.2 cm with <10% bias. The choice mask_threshold=3.0 is more conservative than ANY of these published windows and means strong resonance lines (which sit at tau ~ 3-5 in concentrated matrices like SiO2-rich soil where the 2026-05-13 audit identified Si as most-missed) are masked instead of corrected.
- **Actual in code:** def __init__(
    self,
    optical_depth_threshold: float = 0.1,
    mask_threshold: float = 3.0,
...
if tau > self.mask_threshold:
    # Too optically thick - mask this line
    masked_obs.append(obs)
    corrections[obs.wavelength_nm] = AbsorptionCorrectionResult(
        original_intensity=obs.intensity,
        corrected_intensity=0.0,
        optical_depth=tau,
        correction_factor=0.0,
        is_optically_thick=True,
    )
- **Citation:** Rezaei et al. 2020 Spectrochim. Acta B 169, 105878, sec. 4 (Comparison of methods); Cristoforetti & Tognoni 2013 Spectrochim. Acta B 79-80, 63, sec. 4; John & Anoop 2023 RSC Adv. 13, 29613, Fig. 5-6.
- **Refutability test:** On a synthetic Si soil spectrum (60% SiO2 mass fraction, T=8000K, n_e=5e16, L=0.1 cm), the Si I 251.611 nm line has injected tau ~ 4-5. With mask_threshold=3.0 it is dropped from the line list and the Si concentration is recovered from non-resonance lines only — biased low because the largest gA Si I lines have been lost. With mask_threshold=5.0 the line is kept and corrected, recovering Si concentration within ~10%. Run with both thresholds and compare recovered C_Si vs ground truth.
- **Suggested fix:** Raise default `mask_threshold` from 3.0 to 5.0 to match the published CDSB validity window; document the choice in the SelfAbsorptionCorrector docstring with citations to Cristoforetti & Tognoni 2013 and Rezaei 2020. Alternatively, switch from a hard mask to a soft variance-weighting where lines with tau > 5 are kept but downweighted in the Boltzmann fit (analogous to CDSBPlotter.resonance_weight).
- **Blast radius:** cflibs/inversion/physics/self_absorption.py:SelfAbsorptionCorrector default; tests/test_self_absorption.py:TestSelfAbsorptionCorrectorCorrect fixtures that intentionally set mask_threshold=0.5 for mask-warning tests are unaffected by the default change; downstream consumers (currently none — see finding above on dead code) would see more corrected lines and fewer masked ones, improving Boltzmann fit point count.
- **Estimated macro_F1 lift:** Cristoforetti & Tognoni 2013 report ~4% Ca concentration error using CDSB on limestone with tau up to 5; the equivalent ns-LIBS Si soil benchmark loses Si lines at the same tau range. Effect on macro_F1 unknown but plausibly +0.02 to +0.05 in SiO2-/CaO-/Fe-heavy matrices where the n=33 autodiscovery audit found Si the most-missed element.
- **Verification:** 1/3 skeptics refuted

#### 7. [MEDIUM/high/missing_feature] Empirical resonance_tau_boost=1.5 in CDSB plotter is fabricated heuristic with no literature backing; doubly-counted with E_i=0 Boltzmann factor

- **File:** `cflibs/inversion/physics/cdsb.py:201-252 (constructor); 496-499 (boost application)`
- **Expected per lit:** The Boltzmann population factor (g_i / U) * exp(-E_i / kT) already maximizes for resonance lines because E_i=0 makes exp(-E_i/kT)=1, while non-resonance lines are suppressed by exp(-E_i/kT) < 1. There is NO physical justification in any of the canonical SA literature (Bulajic 2002, Cristoforetti & Tognoni 2013, El Sherbini 2005, Rezaei 2020 review, Poggialini 2023 review, John & Anoop 2023) for an ADDITIONAL multiplicative 1.5x boost to resonance-line tau on top of the Boltzmann factor. John & Anoop 2023 sec. 3-4 show that resonance lines indeed have larger tau, but the ratio comes entirely from the Boltzmann factor n_resonance/n_excited = exp(E_i/kT), not from any empirical tuning.
- **Actual in code:** resonance_tau_boost: float = 1.5,
...
# Resonance lines (E_i ~ 0) naturally get higher tau through population_factor
# Add a configurable boost for explicitly flagged resonance lines.
if obs.is_resonance:
    tau *= self.resonance_tau_boost

The comment one line above ('Resonance lines naturally get higher tau through population_factor') correctly identifies that the Boltzmann factor already handles this; the boost is therefore double-counting.
- **Citation:** Cristoforetti & Tognoni 2013, Spectrochim. Acta B 79-80, 63, eq. (10) (n_l ∝ g_i exp(-E_i/kT)); John & Anoop 2023 RSC Adv. 13, 29613, sec. 3 (resonance vs excited line tau ratios are purely Boltzmann-driven, no empirical boost).
- **Refutability test:** Construct synthetic Si CDSBLineObservation list with two lines from the same upper level — Si I 251.6 (E_i=0, is_resonance=True) and a hypothetical near-resonance Si I line with E_i=0.01 eV (just above the 0.1 eV resonance cutoff so is_resonance=False). At T=10000K the Boltzmann ratio exp(0.01/0.862)≈1.012, so physical tau differs by 1.2%. Run CDSBPlotter._estimate_initial_tau — the resonance-flagged line will get an additional 1.5x boost, giving total tau ratio 1.5*1.012=1.52, a 50% non-physical inflation.
- **Suggested fix:** Default resonance_tau_boost=1.0 (no boost) and add a deprecation comment that the parameter is retained only for backward compatibility with calibration scripts that may have tuned it. Better: remove the parameter entirely and let the Boltzmann factor do its job.
- **Blast radius:** cflibs/inversion/physics/cdsb.py:CDSBPlotter constructor default; tests/test_cdsb.py and tests/test_vrabel2020_cdsb.py may have fixtures asserting specific tau magnitudes that include the boost; downstream consumers are none (dead-code finding above).
- **Estimated macro_F1 lift:** Unknown — CDSB path is dead in current pipeline. After wiring, removing the spurious boost would unbias tau estimates by ~50% for resonance lines, improving the iterative T convergence.
- **Verification:** 0/3 skeptics refuted

### 7.3 stark (8 findings)

#### 1. [CRITICAL/high/algorithmic_wrong_equation] stark_w stored as FWHM at n_e=1e17 but runtime treats it as HWHM at n_e=1e16 (compounded ~20× over-broadening at LIBS conditions)

- **File:** `cflibs/radiation/kernels.py + cflibs/radiation/stark.py + scripts/populate_stark_widths.py + cflibs/atomic/database.py:kernels.py:372-402; stark.py:17,21-89; populate_stark_widths.py:79-176,248-266`
- **Expected per lit:** Konjević et al. 2002 J. Phys. Chem. Ref. Data 31, 819 Eq. 1 and the STARK-B database (Sahal-Bréchot et al. 2014 Adv. Space Res. 54, 1148) tabulate the electron-impact contribution to the Lorentzian as a **FWHM** at a reference density of n_ref = 1e17 cm^-3 (NIST/Konjević critical-review convention) — e.g. Fe II 430.317 nm FWHM = 0.033 nm at n_e=1e17, T=10 kK (Aragón & Aguilera 2010 Spectrochim. Acta B 65, 395). The Voigt convolution then takes that Lorentzian as gamma_L (HWHM) = FWHM/2.
- **Actual in code:** populate_stark_widths.py:79-80 declares `T_REF_K = 10000.0; NE_REF_CM3 = 1.0e17`, line 188 docstring states the column stores `w(λ, T_ref, n_e_ref) = w_ref × (λ / λ_ref)^2` (i.e. FWHM at 1e17 cm^-3), and line 253-266 writes the literature FWHM straight into the `stark_w` column. kernels.py:372-402 then reads it as: `gamma_S(n_e, T) = stark_w * (n_e / 1e16) * (T_eV / 0.86173) ** (-alpha)` and stark.py:17 hard-codes `REF_NE = 1.0e16`. kernels.py:577 then passes the result as the Lorentzian `gamma_per_line` (HWHM) into `_voigt_sum_per_line`. There is no `* 0.1` (units rescale) and no `* 0.5` (FWHM→HWHM) anywhere between the DB read and the Voigt kernel (`grep -rn 'stark_w.*[*/].*0\.5|stark_w.*[*/].*2' cflibs/` returns nothing). Net effect at n_e=1e17: (n_e_ratio) × 10 × (FWHM-as-HWHM) × 2 = ~20× too wide. The anchor test in tests/test_stark_provenance.py:91-95 confirms the convention — it asserts Fe II 430.317 nm stark_w = 33 pm (= literature FWHM at 1e17), not 16.5 pm (HWHM) and not 3.3 pm (HWHM at 1e16).
- **Citation:** Konjević, N.; Lesage, A.; Fuhr, J. R.; Wiese, W. L. J. Phys. Chem. Ref. Data 31(3), 819 (2002), Eq. 1 + 'Definition of terms: width = FWHM'; Sahal-Bréchot et al. Adv. Space Res. 54, 1148 (2014); Aragón & Aguilera, Spectrochim. Acta B 65, 395 (2010), Table 2 (FWHM at 1e17 cm^-3).
- **Refutability test:** Pick Fe II 430.317 nm and a representative ps-LIBS plasma (n_e=1e17 cm^-3, T=1.0 eV, m=55.85 amu). Compute Voigt FWHM via voigt_fwhm(σ_Doppler, γ_stark_per_kernel) — get ≈ 0.66 nm. Independently compute literature-correct FWHM: σ_D = λ √(kT/mc²) → Doppler FWHM ≈ 1.5 pm; Lorentzian FWHM from Aragón & Aguilera 2010 = 0.033 nm; Voigt FWHM ≈ 0.033 nm. The kernel produces ~20× larger Voigt FWHM than literature. After patching (read stark_w as FWHM at 1e17 → divide by 20 inside the kernel, OR rescale once at snapshot build time), the test should match within 10 %.
- **Suggested fix:** Normalize the column convention. Either (a) on DB load convert `stark_w → stark_w * 0.05` so the runtime expression `gamma * (n_e/1e16)` produces HWHM at runtime n_e, or (b) change the runtime expression to `(stark_w / 2) * (n_e / 1e17) * factor_T` and audit all call sites (kernels.py, stark.py, bayesian/forward.py). Add an `ASSERT_CONVENTION` annotation on `AtomicSnapshot.line_stark_w` documenting which it is.
- **Blast radius:** Every Stark-using path: SpectrumModel.compute_spectrum (PHYSICAL_DOPPLER+apply_stark), BayesianForwardModel._compute_spectrum (always apply_stark=True), TwoZoneBayesianForwardModel._compute_zone_spectrum (which has its own copy of the same formula at forward.py:464-473), and the entire manifold generation pipeline (manifold uses apply_stark=True). Tests test_stark.py, test_stark_provenance.py, test_stark_t_factor_toggle.py, test_manifold_physics.py all need a parity sweep — most existing tests use synthetic stark_w values so they will still pass; only tests calibrated against literature widths or measured n_e will surface the change.
- **Estimated macro_F1 lift:** +0.05 to +0.20 (large): Aragón & Aguilera 2008 review documents that 20× over-broadening of Stark wings causes line confusion, spurious line blending, and bias in n_e extraction — directly impacting the macro_F1 = 0.402 vs literature ≥0.7 gap. Magnitude unverified end-to-end.
- **Verification:** 0/3 skeptics refuted

#### 2. [HIGH/high/implementation_bug] stark_alpha column conflates two physically distinct parameters (T-exponent vs Griem ion-broadening A); populate-script writes A but runtime treats it as T-exponent

- **File:** `scripts/populate_stark_widths.py + cflibs/radiation/kernels.py + cflibs/atomic/structures.py:populate_stark_widths.py:72-176,251 (writes 'alpha' as ion-broadening parameter, range 0.04-0.20); structures.py:62-63 (docstring says 'Stark width scaling exponent (typically ~1.0)'); kernels.py:399-402 (uses it as T-exponent).`
- **Expected per lit:** Two distinct parameters: (i) the T-scaling exponent α in w_e(T) = w_ref · (T_ref/T)^α — typically 0.3-1.0 for non-hydrogenic emitters (Griem 1997, Principles of Plasma Spectroscopy ch. 4; STARK-B tabulated grids) — and (ii) the dimensionless ion-broadening parameter A in w_total = w_e · [1 + 1.75 A (1 - 0.75 R)] — typically 0.05-0.20 for non-hydrogenic ions (Griem 1974 Eq. 4-43; Konjević 2002 Eq. 2). These are separate columns in any standard Stark database (e.g. STARK-B exposes them as `w` and `A`).
- **Actual in code:** populate_stark_widths.py:72-75 comment: 'alpha is the ion-broadening parameter (dimensionless), typically ~0.05-0.15 for non-hydrogenic emitters at LIBS conditions.' — the table on lines 82-176 contains values matching this (Fe II→0.08, Cr II→0.09, Mg II→0.04, K I→0.10). At kernels.py:399-401 the same value is consumed as the temperature exponent: `alpha = jnp.asarray(snapshot.line_stark_alpha); REF_T_EV = 0.86173; factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -alpha)`. Result: at T_eV=1.3 eV (ps-LIBS upper edge), factor_T = (1.3/0.86)^(-0.07) ≈ 0.971 — essentially identity instead of the expected ≈ (1.3/0.86)^(-0.5) ≈ 0.815 (per the populate-script docstring formula `(T_ref/T)^0.5`). Meanwhile the actual Griem ion-broadening correction `[1 + 1.75 A (1 - 0.75 R)]` is never applied in the kernel — there is no stark_A_ref consumer in kernels.py, only stark_w and stark_alpha.
- **Citation:** Konjević, N. et al. J. Phys. Chem. Ref. Data 31, 819 (2002), Eqs. 1-2 (separates w_e T-exponent from A_ion); Griem, H. R. Spectral Line Broadening by Plasmas (1974), Eqs. 4-43 (A scaling) and Ch. 4 (T-exponent guidance); Sahal-Bréchot et al. Adv. Space Res. 54, 1148 (2014) (STARK-B schema separates w and A).
- **Refutability test:** Synthetic spectrum benchmark: hold n_e=1e17 cm^-3 fixed, sweep T from 0.5 → 1.3 eV (ps-LIBS regime), compute kernel `gamma_S(T)` for Fe II 430 nm. Observed scaling will be ≈ T^(-0.08) (nearly flat). Correct scaling per the populate-script reference formula and Konjević 2002 Eq. 1 is T^(-0.5) — a 25 % swing across the ps-LIBS band that the current code does not capture. After splitting the schema into `stark_alpha_T` (T-exponent) and `stark_A_ion` (Griem A), the swing should appear.
- **Suggested fix:** Add a new column `stark_A_ion` to the lines table; rename existing `stark_alpha` to its documented physical meaning (`stark_alpha_T`, the T-scaling exponent); have populate_stark_widths.py write α_T ≈ 0.5 (the Konjević default for non-hydrogenic emitters at LIBS T) into stark_alpha_T and the existing 0.04-0.20 values into stark_A_ion; teach `_per_line_stark_gamma` to apply both via `gamma_e * [1 + 1.75 A_ion (1 - 0.75 R)]` (with dynamically computed R per finding below).
- **Blast radius:** Schema migration on libs_production.db; all 28135 stark-populated rows need re-tagging; tests in tests/test_stark.py and tests/test_stark_provenance.py need updated expectations; the BayesianForwardModel._compute_zone_spectrum (forward.py:472) which inlines the same incorrect mapping needs the same surgery.
- **Estimated macro_F1 lift:** +0.02 to +0.05: secondary to finding #1; restores correct T-scaling across the ps-LIBS T range, which improves line-width fitting and downstream n_e extraction. Unverified.
- **Verification:** 0/3 skeptics refuted

#### 3. [HIGH/certain/missing_feature] Ion-broadening correction `[1 + 1.75 A (1 - 0.75 R)]` is never applied in the production Voigt kernel — Griem ion contribution silently dropped

- **File:** `cflibs/radiation/kernels.py:372-402, 576-580`
- **Expected per lit:** Griem, H. R. Spectral Line Broadening by Plasmas (1974), Eqs. 224-226 and Griem 1997 Principles of Plasma Spectroscopy ch. 4: for non-hydrogenic isolated lines under the impact-electron / quasi-static-ion approximation, the total Lorentzian is `w_total = w_e · [1 + 1.75 A (1 - 0.75 R)]` where A is the dimensionless ion-broadening parameter and R = ρ_0/λ_D. Aragón & Aguilera 2008 (Spectrochim. Acta B 63, 893) document that for Fe II / Ni II at n_e=1e17 the correction adds 5-15 % to FWHM. STARK-B tabulates A per-line.
- **Actual in code:** kernels.py:372-402 implements only `gamma_S = stark_w * (n_e / 1e16) * factor_T` and returns. Line 576-580 then uses this directly as the Lorentzian gamma in `_voigt_sum_per_line`. There is no `* (1 + 1.75 * A * (1 - 0.75 * R_D))` factor, no read of `line_stark_A_ref`, and the AtomicSnapshot dataclass has no `line_stark_A` field at all (atomic/database.py:836-879 collects only stark_ws + stark_alphas). The stand-alone `stark.py:stark_hwhm` DOES implement the correction (lines 79-89) with hardcoded R_D=0.5, but `stark_hwhm` is only used by the legacy `StarkBroadeningCalculator.get_stark_width` path which is not on the production forward-model trace.
- **Citation:** Griem, H. R. Spectral Line Broadening by Plasmas (1974), Eqs. 224-226; Griem 1997 Principles of Plasma Spectroscopy ch. 4; Aragón & Aguilera, Spectrochim. Acta B 63, 893 (2008) — review of LIBS Stark diagnostics.
- **Refutability test:** For a non-hydrogenic ion with known A (e.g. Fe II 273.955 nm, A ≈ 0.08 from Konjević 2002 / STARK-B), compute γ_S from kernels._per_line_stark_gamma at n_e=1e17, T=1 eV and compare to γ_e × [1 + 1.75·0.08·(1 - 0.75·R)] for any reasonable R∈[0.1,0.6]. The kernel value will be exactly γ_e (no correction); the literature value is 5-15 % larger.
- **Suggested fix:** Add `line_stark_A` to `AtomicSnapshot`, populate it via the `stark_alpha` column once finding #2 is fixed (or via a new `stark_a` column), and apply `gamma_total = gamma_e * (1 + 1.75 * A_ion * (1 - 0.75 * R_D))` inside `_per_line_stark_gamma`. R_D should be computed dynamically per the next finding.
- **Blast radius:** AtomicSnapshot pytree (must add field, re-run jit cache invalidation), atomic.snapshot builder, kernels._per_line_stark_gamma, BayesianForwardModel._compute_zone_spectrum (currently identical to kernels — same omission), manifold path inherits via the same kernel.
- **Estimated macro_F1 lift:** +0.01 to +0.03: smaller than findings #1/#2 (5-15 % width change vs 20×) but real and physics-correct.
- **Verification:** 0/3 skeptics refuted

#### 4. [MEDIUM/high/regime_mismatch] Hardcoded R_D = 0.5 in ion-broadening correction is out-of-regime across the ps-LIBS plasma parameter envelope

- **File:** `cflibs/radiation/stark.py:84-89, 270-271`
- **Expected per lit:** Griem, H. R. (1974) Spectral Line Broadening by Plasmas, Eqs. 224-226 + p. 73 Eq. 4-23: R = ρ_0/λ_D where ρ_0 = (4π·n_p/3)^(-1/3) (mean inter-ion distance) and λ_D = √(ε_0·kT/(n_e·e²)) (Debye length). Both depend on (T, n_e) and vary substantially across the LIBS plasma envelope. The quasi-static-ion approximation embedded in the Griem-Baranger-Kolb (1 - 0.75 R) factor is valid only for R < 0.8; at R > 0.8 ion-dynamics dominates and a full simulation (e.g. Gigosos & Cardeñoso 1996 J. Phys. B 29, 4795) is required.
- **Actual in code:** stark.py:84-89 (numpy path): `# For a typical LIBS plasma, the Debye shielding parameter R_D is ~0.5. # A full calculation requires Debye length, but 0.5 is a common approximation. R_D = 0.5; correction = 1.0 + 1.75 * A_ion * (1.0 - 0.75 * R_D)`. Identical hardcode at stark.py:270-271 (JAX path). No call to a Debye-length helper, no T or n_e dependence.
- **Citation:** Griem, H. R. Spectral Line Broadening by Plasmas (1974), Eqs. 224-226 and p. 73 Eq. 4-23; Gigosos & Cardeñoso, J. Phys. B 29, 4795 (1996); Aragón & Aguilera, Spectrochim. Acta B 63, 893 (2008).
- **Refutability test:** Compute λ_D and ρ_0 across the ps-LIBS envelope: at n_e=1e16, T=1 eV → λ_D ≈ 24 nm, ρ_0 ≈ 29 nm, R ≈ 1.2 (OUT OF QUASI-STATIC REGIME); at n_e=1e17, T=1 eV → λ_D ≈ 7.4 nm, ρ_0 ≈ 13.5 nm, R ≈ 1.8; at n_e=1e18, T=0.5 eV → λ_D ≈ 1.7 nm, ρ_0 ≈ 6.2 nm, R ≈ 3.6. None of these match the hardcoded R=0.5 (which corresponds to a single point ~ n_e=1e17, T=1 eV with a particular A-form). Add a debye_length() helper and a thin assert in stark.py that R<0.8 (else log a warning) — at ps-LIBS conditions the warning will fire constantly.
- **Suggested fix:** Compute R_D dynamically per (T, n_e): add a small helper `_compute_debye_R(n_e_cm3, T_eV)` that returns ρ_0/λ_D using SI constants; clamp to [0, 0.8] with a log_warning when clipped (signals quasi-static-approximation failure → ion-dynamics regime, where current model is invalid).
- **Blast radius:** stark.py and stark.py JAX twin; once finding #3 is fixed and the correction lands in kernels.py, the same dynamic R_D needs to feed into the kernel.
- **Estimated macro_F1 lift:** unknown (≤+0.02 expected): mostly affects high-n_e branches where the correction is biggest; in the ps-LIBS low-density branch the whole correction is small (A ~ 0.08 × small_factor ≈ <10 %).
- **Verification:** 0/3 skeptics refuted

#### 5. [HIGH/certain/implementation_bug] SpectrumModelJax silently drops Stark Lorentzian — `apply_stark` arg is accepted, stored, and never consulted

- **File:** `cflibs/radiation/spectrum_model.py:475-629 (class body); particularly 501, 517-518, 527-629 (compute_spectrum)`
- **Expected per lit:** Per Aragón & Aguilera 2008 (Spectrochim. Acta B 63, 893) at LIBS n_e ~ 1e17 cm^-3 the Lorentzian Stark FWHM is comparable to or exceeds the Gaussian (Doppler + instrument) FWHM, so the line shape MUST be a Voigt convolution. The Wave-1 fix A2 made `apply_stark=True` the default precisely for this reason — but only on the NumPy SpectrumModel.compute_spectrum path (which routes to kernels.forward_model with apply_stark=True).
- **Actual in code:** SpectrumModelJax.__init__ at line 501-518 accepts `apply_stark: bool = True` and forwards it to `super().__init__(...)`, so `self.apply_stark` is True. But `compute_spectrum` at lines 527-629 does NOT use it: line 553-578 builds `line_sigmas_list` as Doppler-only (line 574-576: `fwhm = doppler_width(...); line_sigmas_list.append(fwhm / 2.355)`), then line 589 calls `_broaden_per_line_jax` — a Gaussian-only outer-product kernel (defined at spectrum_model.py:408-427, line 425 reads `profiles = jnp.exp(-0.5 * x * x) / norm` with no Faddeeva, no gamma argument). There is no per-line Stark gamma computation, no Voigt call, no `if self.apply_stark` branch. The parity tests at tests/radiation/test_spectrum_model_jax.py enforce rtol=1e-5 with the parent class — but the parent class routes through forward_model which DOES apply Stark; so parity is only preserved when the parent is also called in the LEGACY-equivalent code path that happens to ignore Stark.
- **Citation:** Aragón & Aguilera, Spectrochim. Acta B 63, 893 (2008), Eq. 1 and §3 (Stark dominates Lorentzian core at LIBS conditions); Le Drogoff et al. Spectrochim. Acta B 56, 987 (2001) demonstrates Stark as dominant n_e diagnostic in ps/fs LIBS plasmas.
- **Refutability test:** Construct a single-element plasma (Fe II, T=1 eV, n_e=1e17) with PHYSICAL_DOPPLER mode and call both SpectrumModel(apply_stark=True).compute_spectrum() and SpectrumModelJax(apply_stark=True).compute_spectrum() on the same wavelength grid. The parent forward_model path produces Voigt profiles ~10× wider than the child Gaussian-only path on resonance lines (e.g. Fe II 273.955 nm). Add a test asserting Voigt FWHM at line center matches within 5 %.
- **Suggested fix:** Replace the body of `SpectrumModelJax.compute_spectrum` with a direct call into `cflibs.radiation.kernels.forward_model` (mirroring the parent class) so that there is exactly one Stark-aware code path; OR add a per-line Voigt kernel using `_voigt_profile_kernel_jax` and `_per_line_stark_gamma` when `self.apply_stark`. The first is strictly better (kills a duplicated code path and a known divergence vector).
- **Blast radius:** Any caller that constructs SpectrumModelJax (benchmark harness, validation suite). The parity test test_spectrum_model_jax.py needs to be reconciled — likely the test currently runs with apply_stark=False on both sides, so it never detected the divergence. Confirm by greping for apply_stark in tests.
- **Estimated macro_F1 lift:** Depends on which inversion paths use SpectrumModelJax. If the GPU-accelerated forward inside Bayesian uses it: +0.02 to +0.05. If only used for benchmarking: smaller.
- **Verification:** 0/3 skeptics refuted

#### 6. [MEDIUM/certain/algorithmic_wrong_equation] TwoZoneBayesianForwardModel uses spurious factor-of-2 in Doppler sigma (1.41× over-broadening of Voigt Gaussian core)

- **File:** `cflibs/inversion/solve/bayesian/forward.py:456-458`
- **Expected per lit:** The Maxwell-Boltzmann 1D velocity std-dev is σ_v = √(kT/m); the wavelength-space Doppler standard deviation is σ_λ = λ · σ_v/c = λ · √(kT/(m c²)). Used inside the Voigt profile, σ_λ is the Gaussian standard deviation. The √(2 kT/m) form is the most-probable speed of the 3D Maxwell distribution and is NOT the std-dev. This was the explicit bug fixed in profiles.doppler_width (see comment at profiles.py:228-230 'Note: Removed the spurious factor of 2 that computed v_most_probable instead of the standard deviation').
- **Actual in code:** forward.py:456-458: `sigma_doppler = data.wavelength_nm * jnp.sqrt(_as_jax_real(2.0) * T_eV * _JAX_EV_TO_J / (mass_kg * _JAX_C_LIGHT**2))`. The `_as_jax_real(2.0)` factor reproduces precisely the bug that was removed from `profiles.doppler_width` at profiles.py:228-231. Same bug recurs in cflibs/manifold/generator.py:429, 648, 820 and cflibs/inversion/solve/coarse_to_fine.py:500 (verified via `grep -rn 'sqrt(2.0 \* T_eV' cflibs/`).
- **Citation:** profiles.py:225-234 (canonical implementation + audit-note); Konjević 2002 J. Phys. Chem. Ref. Data 31, 819 reports Voigt deconvolution in the standard Maxwell-σ convention; Aragón & Aguilera 2008 Eq. 1.
- **Refutability test:** For Fe II 273.955 nm at T=1 eV (m=55.85 amu): correct σ_D ≈ 1.5 pm, FWHM_D ≈ 3.5 pm; with spurious factor 2, computed σ_D = 1.5 √2 ≈ 2.1 pm, FWHM_D ≈ 5.0 pm. Add a unit test asserting `TwoZoneBayesianForwardModel._compute_zone_spectrum` reproduces `profiles.doppler_width(...) / 2.355` for the σ_Doppler returned at line 456. Currently fails by factor √2.
- **Suggested fix:** Remove the `_as_jax_real(2.0)` factor at line 456-457 and the analogous factor at manifold/generator.py:429,648,820 and inversion/solve/coarse_to_fine.py:500. Centralize Doppler sigma in a single helper (e.g. promote `doppler_sigma_jax` from profiles.py to be the canonical source) so this drift cannot recur.
- **Blast radius:** Two-zone Bayesian inversion outputs; manifold-generated spectra (all four call sites in manifold/generator.py — directly affects manifold-coarse-to-fine inversion); joint optimizer in coarse_to_fine.py. The TwoZone path drifts from the single-zone path systematically by ~41 % in Doppler width, which then biases n_e extraction (Stark width is overestimated by the same amount).
- **Estimated macro_F1 lift:** +0.01 to +0.03: Doppler width error bias in n_e extraction is documented to scale linearly with Voigt deconvolution accuracy (Aragón & Aguilera 2008 §3); a 41 % width bias is non-trivial.
- **Verification:** 0/3 skeptics refuted

#### 7. [MEDIUM/high/missing_feature] Stark shift is computed but never applied to line centers — line identification misregistered at n_e ≥ 1e17 cm^-3

- **File:** `cflibs/radiation/stark.py + cflibs/radiation/kernels.py:stark.py:105-113 (`stark_shift` function); kernels.py:537-580 (forward kernel — no shift application)`
- **Expected per lit:** Konjević 2002 J. Phys. Chem. Ref. Data 31, 819 Eq. 2 and Lesage 2009 New Astron. Rev. 52, 471: Stark shift `Δλ_shift = d_ref · (n_e/n_ref) · (T_ref/T)^β` is a first-order effect separate from broadening; for Fe II / Ni II / Ca II ion lines at n_e = 1e17 cm^-3 the shift can be 5-50 pm (signed). Aragón & Aguilera 2008 §3 documents that uncorrected Stark shifts corrupt line-identification routines that rely on wavelength registration tighter than ~10 pm.
- **Actual in code:** stark.py:105-113 implements `stark_shift(n_e_cm3, stark_d_ref) -> d_ref * (n_e_cm3 / REF_NE)` (note: missing T-scaling; missing the (T/T_ref)^(-β) factor required by Konjević 2002 Eq. 2). This function is wired into `StarkBroadeningCalculator.get_stark_shift` (stark.py:216-222) but the production forward kernel at kernels.py:537-580 never reads `snapshot.line_stark_shift` (and the snapshot builder at atomic/database.py:836-879 never even collects it from `transition.stark_shift`). Net effect: every line is rendered exactly at its rest wavelength even at n_e=1e17 cm^-3 where shifts can be tens of pm.
- **Citation:** Konjević, N.; Lesage, A.; Fuhr, J. R.; Wiese, W. L. J. Phys. Chem. Ref. Data 31, 819 (2002), Eq. 2 and Tables 1-25; Lesage, New Astron. Rev. 52, 471 (2009); Aragón & Aguilera, Spectrochim. Acta B 63, 893 (2008) §3.
- **Refutability test:** For a synthetic ps-LIBS plasma (T=1 eV, n_e=1e17 cm^-3) containing Fe II 273.955 nm (Konjević 2002 Tab 15 reports shift ~ -6 pm), check whether the forward-model output places the line at 273.955 nm (current behavior) or 273.949 nm (literature). The ALIAS / comb-matching identification routine (which expects ±0.01 nm registration in many configs) will misidentify or fail to associate the redshifted line.
- **Suggested fix:** (i) Add `line_stark_shift` and `line_stark_shift_alpha` (T-exponent for shift) to `AtomicSnapshot`; populate from the existing `stark_shift` column. (ii) In `kernels.forward_model`, before computing `epsilon_line` apply `effective_centers = line_wl_nm + stark_shift_at(n_e, T)`. (iii) Fix `stark.stark_shift` to include T-scaling per Konjević 2002 Eq. 2.
- **Blast radius:** Line identification (ALIAS/comb/correlation) at high n_e; Boltzmann-plot fitter (line centroids are inputs); manifold cache invalidation (line centers change with n_e). Inversion result wavelength residuals will tighten.
- **Estimated macro_F1 lift:** +0.01 to +0.03 (regime dependent): largest effect on ion lines (Fe II, Ni II, Ca II) at n_e ≥ 5e16 cm^-3. Aragón & Aguilera 2008 specifically call out shift-induced identification bias as a measurable error source.
- **Verification:** 0/3 skeptics refuted

#### 8. [LOW/certain/numerical_stability] T_eff temperature floor `max(T_K, 1000 K)` masks NaN / Inf at low T instead of failing loudly — ps-LIBS regime at T ≈ 0.5 eV (5800 K) is fine but adjacent corner cases hide bugs

- **File:** `cflibs/radiation/stark.py:73-77, 265`
- **Expected per lit:** Per Griem 1974 and Konjević 2002, the impact + quasi-static approximations have a hard low-T floor below ≈ 5000 K (kT becomes comparable to electron-impact mean energy and the perturbation expansion breaks down). For ps-LIBS the upper-edge T ~ 1.3 eV ~ 15080 K is firmly in-regime but the lower edge T ~ 0.5 eV ~ 5800 K is at the validity boundary (Cristoforetti et al. Spectrochim. Acta B 65, 86 (2010) document LTE regime boundary at n_e ≥ 1e17 cm^-3, t > 1 µs — ps-LIBS at low n_e may fall outside).
- **Actual in code:** stark.py:73-77 (numpy path): `T_eff = max(T_K, 1000.0); w_e = stark_w_ref * (n_e_cm3 / REF_NE) * (T_eff / ref_T_K) ** (-alpha)`. The 1000 K floor silently scales up the width by `(10000/1000)^alpha = 10^alpha` at the worst case (off the chart of any literature validation). JAX twin at line 265: `factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -alpha)` — same floor.
- **Citation:** Griem, H. R. Spectral Line Broadening by Plasmas (1974), ch. 4 validity discussion; Konjević 2002 J. Phys. Chem. Ref. Data 31, 819 (measurement T grid 10-30 kK); Cristoforetti et al. Spectrochim. Acta B 65, 86 (2010).
- **Refutability test:** Inject T = 100 K into `stark_hwhm(1e17, 100, 0.01, alpha=0.5)` — current code returns 0.0316 nm (scaled from `(10000/1000)^0.5 = 3.16`); a correct implementation should raise / warn that T is below the validity envelope. At T = 5800 K (ps-LIBS lower edge) the floor does not trigger so this is purely a defensive-coding issue.
- **Suggested fix:** Replace the silent floor with a `if T < T_floor: log_warning(...)` and either return NaN or clip explicitly. Document validity envelope (T ∈ [5000, 40000] K per Konjević 2002 Table 1 measurement coverage).
- **Blast radius:** Low — ps-LIBS regime (T ≥ 5800 K) does not trigger the floor in practice; only matters for adversarial inputs / posterior tails of NUTS sampler at extreme parameter values.
- **Estimated macro_F1 lift:** negligible
- **Verification:** 1/3 skeptics refuted

### 7.4 closure (7 findings)

#### 1. [HIGH/certain/algorithmic_wrong_equation] plr_transform / plr_inverse compute ALR, not the pivot log-ratio of Hron 2012; PWLR claim of isometry is unsupported

- **File:** `cflibs/inversion/physics/closure.py:144-199`
- **Expected per lit:** Hron, Filzmoser, Thompson 2012 (J. Appl. Stat. 39(5), 1115-1128, eq. 1-2) define the first pivot coordinate as z_1 = sqrt((D-1)/D) * ln( x_1 / (prod_{j=2}^{D} x_j)^{1/(D-1)} ); the remaining D-2 coordinates form an ILR on the sub-composition. PLR is isometric (orthonormal basis in CLR space).
- **Actual in code:** def plr_transform(composition: np.ndarray, pivot_index: int = 0) -> np.ndarray:
    """Pivot log-ratio (PLR) transform (a form of ALR)."""
    D = composition.shape[-1]
    perm = _pivot_permutation(D, pivot_index)
    log_comp = np.log(np.clip(composition[..., perm], LOGRATIO_CLIP_FLOOR, None))
    return log_comp[..., 1:] - log_comp[..., :1]   # returns ln(x_j/x_pivot) for j!=pivot -- this is ALR, not PLR
- **Citation:** Hron, K., Filzmoser, P., Thompson, K. (2012). 'Linear regression with compositional explanatory variables.' J. Appl. Stat. 39(5), 1115-1128, eq. 1-2 (definition of pivot coordinates). Aitchison 1986 / Egozcue 2003 contrast ALR (non-isometric) vs ILR/PLR (isometric).
- **Refutability test:** Construct x = [0.7, 0.2, 0.1]. Compute (a) code's plr_transform(x, pivot_index=0) and (b) Hron 2012 formula z_1 = sqrt(2/3)*ln(0.7/sqrt(0.2*0.1)) ~ 1.305. Code returns [ln(0.2/0.7), ln(0.1/0.7)] ~ [-1.253, -1.946]. Coordinate magnitudes and number-vs-element-meaning differ; Hron coordinate 0 should INCREASE when pivot grows, code's coordinate 0 instead increases when x_2/x_pivot grows. Then verify isometry: Aitchison distance between two compositions equals L2 distance in true PLR but NOT in code's ALR.
- **Suggested fix:** Rename current function to alr_transform; implement true pivot-coordinate per Hron 2012 (first coordinate = sqrt((D-1)/D)*ln(x_pivot/geomean(rest)); remaining coordinates are an ILR of the rest); reroute optimize_pwlr_coordinates / PWLRClosure to use the corrected isometric basis so L2 regularization corresponds to a uniform Aitchison-geometry prior.
- **Blast radius:** closure.py (plr_transform/plr_inverse/optimize_pwlr_coordinates/apply_pwlr); closure_strategy.py (PWLRClosure adapter and any solver opting into PWLR); test_closure_pwlr.py; iterative/joint_optimizer/coarse_to_fine solvers that select closure_mode='pwlr'.
- **Estimated macro_F1 lift:** unknown
- **Verification:** 0/3 skeptics refuted

#### 2. [HIGH/certain/algorithmic_wrong_equation] Dirichlet-residual closure equates experimental factor F with a 'closure deficit' -- diagnostic and residual fraction are not scale-invariant

- **File:** `cflibs/inversion/physics/closure.py:668-778`
- **Expected per lit:** Ciucci 1999 eq. 5-6 and Tognoni 2010 Section 2 define the experimental factor F = Σ U_s · exp(q_s) as having arbitrary scale (it absorbs optical efficiency × ablated number density × plasma volume). Any 'missing-mass diagnostic' must be a dimensionless ratio that is invariant under rescaling of intensities, e.g. by partitioning the unit simplex on already-normalized concentrations.
- **Actual in code:** raw_sum = sum(raw_concentrations.values())
...
closure_diagnostic = abs(raw_sum - 1.0)
...
else:
    deficit = 1.0 - raw_sum
    if deficit > residual_threshold:
        residual = max(0.0, deficit)
    else:
        residual = 0.0
...
detected_budget = 1.0 - residual
concentrations = {el: c / raw_sum * detected_budget for el, c in raw_concentrations.items()}
- **Citation:** Ciucci, A. et al. (1999). 'New Procedure for Quantitative Elemental Analysis by Laser-Induced Plasma Spectroscopy.' Appl. Spectrosc. 53(8), 960-964, eq. 5-6 (F is arbitrary calibration scale). Tognoni et al. 2010 Spectrochim. Acta B 65(1), 1-14, Section 2 explicitly notes F has no fixed scale.
- **Refutability test:** Take a real iterative.py run (closure_mode='dirichlet_residual') on synthetic data where ALL elements are detected, then multiply every line intensity by 1e3 (which leaves true concentrations unchanged but multiplies raw_sum by 1e3 because q_s = ln(F·C/U) shifts by ln(1e3)). Verify residual_fraction goes from ~0 to a wildly different value. The mode is correct only when callers pre-scale q_s so F~1; tests in test_closure_dirichlet.py all use _make_intercepts_and_pfs(..., F=1.0) which masks this.
- **Suggested fix:** Reformulate residual estimation on the already-normalized simplex of detected elements vs. a prior on the dark category, not on raw_sum. One literature-supported approach is to first apply standard closure (Σ C_detected = 1), then re-allocate using a Bayesian Dirichlet posterior with hyperparameter α_residual representing the prior strength of the dark category; residual fraction = α_res/(α_res + Σ α_detected), which is intensity-scale-invariant.
- **Blast radius:** closure.py.apply_dirichlet_residual; iterative.py closure_mode='dirichlet_residual' branch (line 1164); test_closure_dirichlet.py (all tests bypass the bug via F=1).
- **Estimated macro_F1 lift:** unknown
- **Verification:** 0/3 skeptics refuted

#### 3. [MEDIUM/certain/missing_feature] apply_matrix_mode returns concentrations with no Σ C_s = 1 enforcement and no diagnostic when result departs from simplex

- **File:** `cflibs/inversion/physics/closure.py:371-436`
- **Expected per lit:** Ciucci et al. 1999 eq. 5-6 / Tognoni 2010 Section 2: when one element is pinned (C_matrix = matrix_fraction), the closure equation F = (U_m exp(q_m)) / matrix_fraction yields C_s = U_s exp(q_s) / F for all other species; Σ C_s should equal 1 to within measurement noise IF the matrix-element fraction and partition functions are consistent with the Boltzmann-plot data. Departure of Σ C_s from 1 is a published diagnostic of LTE breakdown, missing element bias, or wrong matrix_fraction (Tognoni 2010 closure-residual discussion).
- **Actual in code:** F = rel_C_m / matrix_fraction
concentrations = {}
total_measured = 0.0
for element, q_s in intercepts.items():
    if element in partition_funcs:
        U_s = partition_funcs[element]
        multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
        rel_C = multiplier * U_s * np.exp(q_s)
        total_measured += rel_C
        concentrations[element] = rel_C / F
return ClosureResult(
    concentrations=concentrations,
    experimental_factor=F,
    total_measured=total_measured,
    mode=f"matrix({matrix_element}={matrix_fraction})",
)
- **Citation:** Ciucci, A. et al. (1999), Appl. Spectrosc. 53(8), 960-964; Tognoni, E. et al. (2010), Spectrochim. Acta B 65(1), 1-14, Section 2 (closure-residual diagnostic).
- **Refutability test:** Synthesize a noisy spectrum with Fe=0.65, Cu=0.30, Al=0.05; call apply_matrix_mode with matrix_element='Fe', matrix_fraction=0.65. Inject 30% gain noise on Cu lines. Verify the returned concentrations no longer satisfy |Σ C_s - 1| < 1e-3 yet no warning is logged and no diagnostic is exposed via ClosureResult.
- **Suggested fix:** Extend ClosureResult with a closure_residual = Σ C_s - 1 diagnostic field; log a warning when |closure_residual| > a configurable tolerance (e.g. 0.05). Optionally provide an opt-in renormalization for downstream users who want a strict simplex.
- **Blast radius:** closure.py.apply_matrix_mode + ClosureResult schema; uncertainty.py matrix variant; iterative.py (closure_mode='matrix' branch); test_closure.py.
- **Estimated macro_F1 lift:** unknown
- **Verification:** 1/3 skeptics refuted

#### 4. [MEDIUM/certain/missing_feature] Closed-form ILR solver hard-codes Σ C_s = 1; no support for matrix-element pinning or oxide closure that the iterative solver offers

- **File:** `cflibs/inversion/solve/closed_form.py:187-308`
- **Expected per lit:** ChemCam practice (Clegg et al. 2017, Spectrochim. Acta B 129, 64-85; Wiens et al. 2013, Spectrochim. Acta B 82, 1-27) reports planetary CF-LIBS as major-oxide closure (SiO2, TiO2, Al2O3, FeO_T, MgO, CaO, Na2O, K2O); Tognoni 2010 documents matrix-element pinning for steel and Al alloy analyses. Both modes are documented standard practice for ns-LIBS that the project's iterative solver implements but the closed-form solver silently substitutes with Σ C_s = 1 (a different, often poorer, model for those matrices).
- **Actual in code:** if D >= 2:
    V = _helmert_basis(D)
    n_cols = 1 + (D - 1) + 1  # slope + ILR coords + intercept
...
alpha = theta[1:D]
comp_arr = ilr_inverse(alpha, D)
compositions = {el: float(comp_arr[i]) for i, el in enumerate(element_order)}
# (no matrix_element / oxide_stoichiometry paths anywhere in the file)
- **Citation:** Clegg, S.M. et al. (2017), Spectrochim. Acta B 129, 64-85 (oxide closure for ChemCam); Wiens, R.C. et al. (2013), Spectrochim. Acta B 82, 1-27; Tognoni, E. et al. (2010), Spectrochim. Acta B 65(1), 1-14, Section 4 (matrix-element pinning).
- **Refutability test:** Look at every call site of ClosedFormILRSolver and confirm it ignores user-supplied closure_mode arguments. Then run a synthetic geological spectrum where the iterative solver produces 0.7 macro_F1 with oxide-mode closure and the closed-form solver produces substantially worse F1 because oxygen mass is dumped onto the metals.
- **Suggested fix:** Add a closure_mode parameter to ClosedFormConfig; for matrix mode add a linear constraint α to fix C_matrix = matrix_fraction (constrained WLS via Lagrange multiplier); for oxide mode add an extra log-Jacobian term so closure is in oxide-mass-basis space.
- **Blast radius:** ClosedFormConfig API; ClosedFormILRSolver.solve and _extract_parameters; any test/script using closed-form on geological/steel datasets where matrix or oxide closure is appropriate.
- **Estimated macro_F1 lift:** unknown
- **Verification:** 1/3 skeptics refuted

#### 5. [MEDIUM/high/regime_mismatch] Closed-form solver's pressure-balance n_e step assumes STP_PRESSURE = 1 atm, which is wrong for ps-LIBS plasmas

- **File:** `cflibs/inversion/solve/closed_form.py:37-45, 439-453`
- **Expected per lit:** Aragon & Aguilera 2008 (Spectrochim. Acta B 63(9), 893-916, Section 4) characterize ns-LIBS plasma pressure as time-dependent and well above ambient during typical analysis windows (kPa-MPa scale early, decaying toward ambient only at very late delays). Pressure-balance n_e estimates require the actual plasma pressure, not ambient. For ps-LIBS at ~10^16-10^18 cm^-3 with T~0.5-1.3 eV, plasma pressure ~n_tot k_B T may be ~1e3-1e4 Pa or higher (still potentially below 1 atm during the analysis window if the plume has expanded) -- the safe assumption is NOT STP.
- **Actual in code:** STP_PRESSURE = 101325.0  # Pa (1 atm)   [cflibs/core/constants.py:88]
...
pressure_pa: float = STP_PRESSURE   [ClosedFormConfig default]
...
if self.config.ne_mode == "pressure":
    for _ in range(20):
        ...
        n_tot = self.config.pressure_pa / (KB * T_K * (1.0 + avg_Z))
        n_e = avg_Z * n_tot * 1e-6  # cm^-3
- **Citation:** Aragon, C. & Aguilera, J.A. (2008), 'Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods.' Spectrochim. Acta B 63(9), 893-916. doi:10.1016/j.sab.2008.05.010 (Section 4: plasma pressure evolution).
- **Refutability test:** Run ClosedFormILRSolver on a synthetic ps-LIBS spectrum with known n_e = 1e17 cm^-3 and known T = 0.8 eV at ne_mode='pressure'. Compare returned n_e to ground truth: with STP_PRESSURE the prediction will be off by ~ratio (actual_plasma_pressure / 101325). Document the deviation as a function of assumed plasma pressure.
- **Suggested fix:** (a) Make the pressure-balance default match the ps-LIBS regime documented in the project (lower P or n_e-based mode default) and (b) warn / refuse when ne_mode='pressure' is used without an explicitly supplied plasma pressure -- direct n_e estimation via Stark broadening should be preferred for ps-LIBS.
- **Blast radius:** ClosedFormConfig default; ClosedFormILRSolver n_e estimation; iterative.py also defaults pressure_pa=STP_PRESSURE (same issue); CLI / inversion config files that don't override pressure_pa.
- **Estimated macro_F1 lift:** unknown
- **Verification:** 0/3 skeptics refuted

#### 6. [MEDIUM/high/regime_mismatch] Matrix-effects correction factors in _DEFAULT_FACTORS have no traceable per-factor literature provenance and are uncalibrated for ps-LIBS

- **File:** `cflibs/inversion/physics/matrix_effects.py:176-218, 225-236`
- **Expected per lit:** Hahn & Omenetto 2010 (Appl. Spectrosc. 64(12), 335A-366A) and 2012 (Appl. Spectrosc. 66(4), 347-419) review matrix effects qualitatively but do NOT supply an element-by-matrix lookup table of multiplicative corrections; Cremers & Radziemski (Handbook of LIBS, 2nd ed., 2013) cataloges empirical factors per matrix only with specific calibration datasets attached. No canonical CF-LIBS paper validates multiplicative factors at the granularity of {METALLIC.Fe: 1.0, METALLIC.Cr: 1.02, METALLIC.Mn: 1.05} as in the code's _DEFAULT_FACTORS.
- **Actual in code:** _DEFAULT_FACTORS: Dict[MatrixType, Dict[str, Tuple[float, float]]] = {
    MatrixType.METALLIC: {
        "Fe": (1.0, 0.05),
        "Cr": (1.02, 0.08),
        "Ni": (0.98, 0.06),
        "Mn": (1.05, 0.10),
        ...
    },
    ...
}
...
for matrix_type, elements in self._DEFAULT_FACTORS.items():
    for el, (mult, uncert) in elements.items():
        self._factors[matrix_type][el] = CorrectionFactor(
            element=el,
            matrix_type=matrix_type,
            multiplicative=mult,
            uncertainty=uncert,
            source="default_literature",   # <-- generic, no per-factor citation
        )
- **Citation:** Hahn, D.W. & Omenetto, N. (2010), Appl. Spectrosc. 64(12), 335A-366A; Hahn, D.W. & Omenetto, N. (2012), Appl. Spectrosc. 66(4), 347-419 -- both reviews; neither provides the per-element multiplicative factors used in _DEFAULT_FACTORS.
- **Refutability test:** For each (matrix_type, element) factor in _DEFAULT_FACTORS, attempt to locate a peer-reviewed source that prescribes that exact value. Document which factors have a citation and which are interpolations / guesses. Additionally compare against a calibrated reference (e.g. NIST SRM 1265 for METALLIC.Fe alloys) and quantify residual bias.
- **Suggested fix:** Mark _DEFAULT_FACTORS as a stub/placeholder; require an explicit CorrectionFactorDB file supplied by the user (matrix + element + factor + uncertainty + citation/DOI); raise a UserWarning when the default DB is used without override. For ps-LIBS, do NOT auto-apply ns-LIBS calibration multipliers.
- **Blast radius:** matrix_effects.py (CorrectionFactorDB._populate_defaults, MatrixEffectCorrector public API); cflibs/inversion/__init__.py public exports; test_matrix_effects.py; any external consumer that calls MatrixEffectCorrector with default DB.
- **Estimated macro_F1 lift:** unknown
- **Verification:** 0/3 skeptics refuted

#### 7. [LOW/high/numerical_stability] ILR / PWLR clip-floor 1e-10 silently biases composition coordinates at trace concentrations instead of using zero-imputation

- **File:** `cflibs/inversion/physics/closure.py:24, 86, 166, 228-236`
- **Expected per lit:** Aitchison 1982 (J. Royal Stat. Soc. B 44(2), 139-177, Section 7) and Egozcue, Pawlowsky-Glahn et al. 2003 (Math. Geol. 35(3), 279-300) both warn that 'replace zero by a small constant' (the clip-floor strategy used in the code) introduces SYSTEMATIC bias on log-ratio statistics; the recommended approach is Bayesian-multiplicative imputation (Martín-Fernández et al. 2003, Math. Geol. 35(3), 253-278) or model-based zero handling. The clip floor is an artifact of programming convenience, not a defensible CoDA practice.
- **Actual in code:** LOGRATIO_CLIP_FLOOR = 1e-10
...
def clr_transform(composition: np.ndarray) -> np.ndarray:
    log_comp = np.log(np.clip(composition, LOGRATIO_CLIP_FLOOR, None))
    return log_comp - np.mean(log_comp, axis=-1, keepdims=True)
...
def plr_transform(composition: np.ndarray, pivot_index: int = 0) -> np.ndarray:
    ...
    log_comp = np.log(np.clip(composition[..., perm], LOGRATIO_CLIP_FLOOR, None))
    return log_comp[..., 1:] - log_comp[..., :1]
# optimize_pwlr_coordinates:
simplex_perm = np.clip(simplex[perm], PWLR_WEIGHT_FLOOR, None)
- **Citation:** Aitchison, J. (1982), J. Royal Stat. Soc. B 44(2), 139-177, Section 7; Egozcue, J.J. et al. (2003), Math. Geol. 35(3), 279-300; Martín-Fernández, J.A., Barceló-Vidal, C., Pawlowsky-Glahn, V. (2003), Math. Geol. 35(3), 253-278 (Bayesian-multiplicative imputation).
- **Refutability test:** Construct synthetic compositions where one element is at 1e-12 (below clip floor), verify ilr_transform/plr_transform output is biased toward the clip floor (gives a finite coordinate ~-23) rather than diverging or being imputed sensibly. Compare downstream regression-coefficient estimates of the trace element using (a) clip floor and (b) Bayesian-multiplicative imputation; expect bias proportional to the fraction of clipped components.
- **Suggested fix:** Replace the unconditional clip floor with an explicit zero-handling strategy: (a) raise a warning when any input C_i falls below a 'physical' floor that the user supplies, (b) optionally apply Bayesian-multiplicative imputation when zeros are present, (c) document that callers must hand in non-zero compositions for proper CoDA semantics.
- **Blast radius:** closure.py (clr_transform, plr_transform, optimize_pwlr_coordinates); closure_strategy.py (ILRClosure/PWLRClosure adapters re-use LOGRATIO_CLIP_FLOOR); tests that assume clip-floor behavior.
- **Estimated macro_F1 lift:** unknown
- **Verification:** 1/3 skeptics refuted

### 7.5 iterative-solver (9 findings)

#### 1. [HIGH/high/regime_mismatch] Pressure-balance n_e update assumes P = 1 atm; literature canonically uses Stark broadening, not isobaric closure

- **File:** `cflibs/inversion/solve/iterative.py:704,712-714,1180-1211,657-660`
- **Expected per lit:** Tognoni et al. (2010) Spectrochim. Acta B 65, 1-14, Section 3.1 and Section 4.2: the canonical iterative CF-LIBS loop obtains n_e from an independent measurement (e.g. Stark broadening of H-alpha/H-beta or selected line) — n_e is NOT constrained via an ideal-gas isobaric assumption. The Saha equation then provides the self-consistency check, not the closure on n_e. Aragon & Aguilera (2008) Spectrochim. Acta B 63, 893-916 Section 2 lists Stark width as the primary OES n_e diagnostic. The plume in a 1 ps Yb:fiber LIBS pulse is highly overpressurized (10-100 atm in the early emission window), so a P = 1 atm = STP_PRESSURE constraint is physically wrong.
- **Actual in code:** pressure_pa: float = STP_PRESSURE,  # ctor default, line 704
...
n_tot = self.pressure_pa / (KB * T_K * (1.0 + avg_Z))  # line 1204
n_tot_cm3 = n_tot * 1e-6
ne_new = avg_Z * n_tot_cm3  # line 1208
n_e = 0.5 * ne_prev + 0.5 * ne_new  # line 1211
- **Citation:** Tognoni, E.; Cristoforetti, G.; Legnaioli, S.; Palleschi, V. (2010) Spectrochim. Acta B 65(1), 1-14 (DOI 10.1016/j.sab.2009.11.006), Sections 3.1 and 4.2 on n_e diagnostics. Aragon & Aguilera (2008) Spectrochim. Acta B 63(9), 893-916 (DOI 10.1016/j.sab.2008.05.010), Section 2 on Stark-broadening n_e.
- **Refutability test:** Synthesize a spectrum at T = 8000 K, n_e = 3e16 cm^-3, P_actual = 30 atm (representative ps-LIBS early emission). Invert with default pressure_pa = STP_PRESSURE. The recovered n_e will collapse toward avg_Z * P/(kT(1+avg_Z)) * 1e-6 ~ 1e15-1e16 (1-2 orders below truth) regardless of input. Re-run with pressure_pa set to actual plume pressure to verify the bias source.
- **Suggested fix:** Add a Stark-width n_e estimator branch (use one selected line per spectrum's Hα/Hβ or comparable broadening reference) and either replace the pressure-balance update with that n_e or treat pressure_pa as a free parameter inferred jointly. At minimum, log a warning when the default STP pressure is used in a LIBS regime.
- **Blast radius:** Affects all callers of IterativeCFLIBSSolver.solve(); changes Saha self-consistency and concentration outputs; requires updating test_solver_electron_density_pressure_balance and any benchmark using default pressure. Joint optimizer and Bayesian module already accept n_e independently and are unaffected.
- **Estimated macro_F1 lift:** unknown (regime-dependent; literature reports correcting n_e by 1 order can change concentrations by 10-30% in trace elements)
- **Verification:** 0/3 skeptics refuted

#### 2. [HIGH/certain/missing_feature] SpectralRefiner does not enforce closure (sum C = 1), violating CF-LIBS foundational invariant

- **File:** `cflibs/inversion/solve/spectral_refiner.py:25-27,126-142,261-281`
- **Expected per lit:** Ciucci et al. (1999) Appl. Spectrosc. 53(8), 960-964, Eq. (8-10), and Tognoni et al. (2010) Eq. (8-10): the closure equation Σ_s C_s = 1 is the ESSENTIAL algebraic step that eliminates the experimental factor F. Any CF-LIBS optimizer that returns concentrations must enforce or post-renormalize on this simplex. The joint optimizer correctly enforces this via softmax (joint_optimizer.py line 537-549).
- **Actual in code:** _CONC_BOUNDS = (0.0, 1.0)  # line 27
# In _pack_initial_vector (line 138-141):
for i, el in enumerate(elements_used):
    x0[2 + i] = np.clip(concentrations_init.get(el, 1.0 / n_elements), *_CONC_BOUNDS)
bounds = [_T_BOUNDS_K, _LOG_NE_BOUNDS] + [_CONC_BOUNDS] * n_elements
# In objective (line 263-273):
conc = x[2:]
basis = self.basis_library.get_basis_matrix_interp(T_K, ne_cm3)
model = conc @ selected
# No closure constraint anywhere
- **Citation:** Ciucci, A.; Corsi, M.; Palleschi, V.; Rastelli, S.; Salvetti, A.; Tognoni, E. (1999) Appl. Spectrosc. 53(8), 960-964 (DOI 10.1366/0003702991947612), Eqs. (8)-(10) closure. Tognoni et al. (2010) Spectrochim. Acta B 65, 1-14, Eq. (10).
- **Refutability test:** Run SpectralRefiner.refine(...) on a known synthetic spectrum with C_Fe=0.7, C_Ni=0.3. After convergence, evaluate sum(result.concentrations.values()). It will not equal 1.0 (and need not even be close — basis-matrix scaling absorbs any constant factor, so the optimizer is free to pick any (c, scale) pair).
- **Suggested fix:** Either softmax-parameterize concentrations (matching JointOptimizer line 539-549) or add a single-line post-step normalisation `conc /= conc.sum()` before returning. The softmax route also removes the C ∈ [0, 1] constraint slack that creates the under-determined scaling.
- **Blast radius:** Affects every SpectralRefiner caller in NNLS-then-refine pipelines. Tests in tests/test_spectral_refiner.py and tests/test_spectral_refiner_coverage.py need a closure invariant assertion. Iterative solver and joint optimizer unaffected.
- **Estimated macro_F1 lift:** +0.02 to +0.05 (closure-violation produces concentrations off by a multiplicative factor that breaks any element-presence threshold downstream; this is squarely in the macro_F1 0.402 vs 0.7 gap)
- **Verification:** 0/3 skeptics refuted

#### 3. [HIGH/certain/implementation_bug] Solver reports converged=True for unphysical positive Boltzmann slope (clamped T=50000 K, no failure signal)

- **File:** `cflibs/inversion/solve/iterative.py:1109-1116,630-631,1215-1221`
- **Expected per lit:** Tognoni et al. (2010) Spectrochim. Acta B 65, 1-14 Section 3.1: a Boltzmann plot with positive (or near-zero) slope indicates a failed fit — insufficient lines, mixed temperatures, self-absorption inversion, or non-LTE. The canonical response is to flag failure and abort, not to clamp T and continue. Cristoforetti et al. (2010) Spectrochim. Acta B 65, 86-95 Section 3 explicitly recommends multi-temperature consistency checks at this point.
- **Actual in code:** # Update T (line 1110-1113):
if slope >= 0:
    T_new = 50000.0  # Clamp max
else:
    T_new = -1.0 / (slope * KB_EV)
# Damping (line 1116):
T_K = 0.5 * T_prev + 0.5 * T_new
# ... no quality_metrics['boltzmann_slope_positive'] flag is recorded
# Lax mirror (line 630):
T_new = jnp.where(slope >= 0.0, 50000.0, -1.0 / (slope * KB_EV))
- **Citation:** Tognoni, E. et al. (2010) Spectrochim. Acta B 65, 1-14 Sec. 3.1. Cristoforetti, G. et al. (2010) Spectrochim. Acta B 65, 86-95 Sec. 3 (DOI 10.1016/j.sab.2009.11.005).
- **Refutability test:** Construct 3 LineObservations with E_k values 1, 2, 3 eV and *increasing* y_value (an unphysical/inverted Boltzmann plot — e.g. self-absorbed resonance lines). Run solver.solve(); the result will have T_K ~ 30000 K (damped from 10000→50000) AND converged=True after 2-3 iterations because |ΔT|<100 K is hit at the clamped attractor. Recovered concentrations will be a function of the clamp, not the data.
- **Suggested fix:** When slope >= 0 (or slope < some small negative threshold), set converged=False, populate quality_metrics['boltzmann_slope_failure']=True, and either raise or return with explicit failure flag — do not silently clamp.
- **Blast radius:** Affects all callers; needs test addition for positive-slope failure mode. Existing tests pass because mock_db data is well-conditioned.
- **Estimated macro_F1 lift:** +0.01 to +0.03 (rare in clean data but contaminates any spectrum with strong self-absorbed resonance lines — common in ps-LIBS where line selection is poor)
- **Verification:** 0/3 skeptics refuted

#### 4. [MEDIUM/certain/numerical_stability] Slightly-negative slope produces astronomical T with no clamp on lower magnitude

- **File:** `cflibs/inversion/solve/iterative.py:1109-1116,630-631`
- **Expected per lit:** Tognoni 2010 implicitly: T must be within physically meaningful bounds (10^3-10^5 K for LIBS). The upper clamp T_new = 50000 K is applied for slope>=0 but no clamp exists for slope = -1e-8 → T_new ~ 1e12 K. After damping with T_prev = 10000 K, T_K = 5e11 K — clearly unphysical but the solver continues.
- **Actual in code:** if slope >= 0:
    T_new = 50000.0  # Clamp max
else:
    T_new = -1.0 / (slope * KB_EV)  # slope=-1e-8 -> T_new=1.16e12 K
# Damping (line 1116):
T_K = 0.5 * T_prev + 0.5 * T_new
- **Citation:** Aragon & Aguilera (2008) Spectrochim. Acta B 63, 893-916 (DOI 10.1016/j.sab.2008.05.010) Section 2: LIBS plasma T range 5000-30000 K.
- **Refutability test:** Feed observations with 2 lines whose y_values differ by <1e-6 across E_k spread of 2 eV (near-degenerate slope). slope ~ -1e-7 → T_new ~ 1.16e11 K → T_K after damping >> 1e10 K. Check that quality_metrics['temperature_K'] is clamped to a finite physical range.
- **Suggested fix:** Apply a two-sided clamp: T_new = max(min(-1/(slope*KB_EV), 50000), 3000) before damping, plus a minimum |slope| threshold (e.g. require |slope|*KB_EV > 1e-5) to refuse the fit.
- **Blast radius:** Same as the slope>=0 case; needs unit test for degenerate-slope numerical stability.
- **Estimated macro_F1 lift:** unknown (rare but produces NaN partition functions downstream when T overflows partition polynomial fit range)
- **Verification:** 0/3 skeptics refuted

#### 5. [MEDIUM/high/numerical_stability] Default ne_tolerance_frac=0.1 (10%) is too loose; convergence check compares damped n_e vs ne_prev so oscillations register as converged

- **File:** `cflibs/inversion/solve/iterative.py:703,1211,1216-1221`
- **Expected per lit:** Tognoni et al. (2010) Section 3.1 says 'iterate until stabilization' but does not specify a tolerance; standard numerical-analysis practice for self-consistent plasma loops requires tolerance on the relative change of UNDAMPED iterates (or successive damped iterates with monitoring of oscillation). 10% on n_e at n_e=1e17 corresponds to ±1e16 — larger than the typical uncertainty band in Stark-derived n_e (~5-10%) AND larger than what produces stable Saha multipliers. With 50/50 damping (line 1116, 1211), |n_e_new - ne_prev|<0.1*ne_prev can be satisfied while the *undamped* update is oscillating by 20% each step.
- **Actual in code:** ne_tolerance_frac: float = 0.1,  # ctor default, line 703
...
n_e = 0.5 * ne_prev + 0.5 * ne_new  # damped, line 1211
if (
    abs(T_K - T_prev) < self.t_tolerance_k
    and abs(n_e - ne_prev) / ne_prev < self.ne_tolerance_frac  # damped vs prev, line 1218
):
    converged = True
    break
- **Citation:** Tognoni, E. et al. (2010) Spectrochim. Acta B 65, 1-14 Sec. 3.1. Standard numerical-analysis under-relaxation convergence criterion: |x_{n+1}-x_n|/|x_n| < tol on undamped or several damped successive iterates.
- **Refutability test:** Construct observations where pressure_pa/T yields ne_new alternating between 5e16 and 2e17 each undamped iteration. With damping factor 0.5, the damped values oscillate: 1e17 → 1.5e17 → 1.25e17 → 1.375e17. |Δ/prev|=0.5, 0.17, 0.10 — converges spuriously on iteration 3 even though the underlying Saha-self-consistent n_e is not at equilibrium.
- **Suggested fix:** Tighten default ne_tolerance_frac to 0.01-0.02 (matching Stark-broadening uncertainty), check convergence on UNDAMPED ne_new (or check stability over k=3 successive iterations), and add a guard `ne_prev > 1e10` to avoid division by spurious zero.
- **Blast radius:** Affects every iterative.solve() call; some existing tests rely on the loose tolerance and may flag as non-converged. Bayesian/joint paths unaffected.
- **Estimated macro_F1 lift:** unknown
- **Verification:** 0/3 skeptics refuted

#### 6. [MEDIUM/high/missing_feature] Iterative solver lacks resonance-line down-weighting; self-absorption-prone resonance lines enter Boltzmann fit at full weight

- **File:** `cflibs/inversion/solve/iterative.py:735-741,908-1003,1057-1106`
- **Expected per lit:** Aragon & Aguilera (2008) Spectrochim. Acta B 63, 893-916 (DOI 10.1016/j.sab.2008.05.010) Section 3: resonance lines (E_lower=0) are the most self-absorption-prone and should be excluded or down-weighted in Boltzmann-plot CF-LIBS. Safi et al. (2019) J. Adv. Res. 18, 1-8 (DOI 10.1016/j.jare.2019.01.008) CDSB method explicitly couples Saha-Boltzmann to a self-absorption correction precisely because raw resonance lines bias T upward and concentrations downward.
- **Actual in code:** def _line_y_uncertainty(self, obs: LineObservation) -> float:
    sigma_y = obs.y_uncertainty
    unc = obs.aki_uncertainty
    if self.aki_uncertainty_weighting and unc is not None and np.isfinite(unc) and unc > 0:
        sigma_y = float(np.sqrt(sigma_y**2 + float(unc) ** 2))
    return sigma_y
# No check on obs.E_lower / obs.is_resonance / lower-level info; LineObservation
# does not carry E_lower or resonance flag at all.
- **Citation:** Aragon, C.; Aguilera, J. A. (2008) Spectrochim. Acta B 63(9), 893-916 (DOI 10.1016/j.sab.2008.05.010) Section 3 on self-absorption. Safi, A. et al. (2019) J. Adv. Res. 18, 1-8 (DOI 10.1016/j.jare.2019.01.008) CDSB.
- **Refutability test:** Synthesize an Fe spectrum at T=8000 K, n_e=1e17 with the Fe I 248.327 nm resonance line (E_lower=0) artificially attenuated by 50% to simulate τ≈0.7 self-absorption. Invert with IterativeCFLIBSSolver. The recovered T will be biased high by 500-1500 K (slope flattened) and Fe concentration biased low. Compare to a run with the resonance line excluded — the bias should vanish.
- **Suggested fix:** Extend LineObservation with an `E_lower_ev` field (or `is_resonance` flag) and in `_fit_common_boltzmann_plane` either exclude resonance lines or apply CDSB-derived τ_resonance to scale their weights. Until then, expose a `exclude_resonance: bool` ctor flag.
- **Blast radius:** LineObservation dataclass change is invasive (every constructor in tests + line-identification pipeline). Could be implemented additively via `Optional[float] = None`. Tests need a resonance fixture.
- **Estimated macro_F1 lift:** +0.03 to +0.08 (Aguilera 2014 Spectrochim. Acta B reports 5-15% concentration errors from unfiltered resonance lines on Cu/Fe matrices; mapping to macro_F1 via the project's identification threshold gives ~0.05 lift)
- **Verification:** 0/3 skeptics refuted

#### 7. [LOW/certain/numerical_stability] Joint optimizer concentration uncertainty uses diagonal-only softmax Jacobian, ignoring -c_i*c_j off-diagonal coupling

- **File:** `cflibs/inversion/solve/joint_optimizer.py:488-493,495-497`
- **Expected per lit:** Standard softmax Jacobian (e.g. Bridle 1990; restated in Aitchison 1986 'Statistical Analysis of Compositional Data'): ∂c_i/∂θ_i = c_i(1-c_i), ∂c_i/∂θ_j = -c_i c_j (i≠j). Full uncertainty propagation requires σ²_c_i = Σ_jk J_ij Cov_jk J_ik where J is the full E×E softmax Jacobian, NOT just diag(J) * std_err.
- **Actual in code:** # Concentration uncertainties (need Jacobian of softmax)
for i, el in enumerate(self.elements):
    # Approximate uncertainty
    param_uncertainties[f"C_{el}"] = float(
        final_conc_arr[i] * (1 - final_conc_arr[i]) * std_errors[2 + i]
    )
# Correlation matrix
std_diag = np.diag(1.0 / (std_errors + 1e-10))
correlation_matrix = std_diag @ cov @ std_diag
- **Citation:** Bridle, J. S. (1990) 'Probabilistic interpretation of feedforward classification network outputs', Neurocomputing (softmax Jacobian). Aitchison, J. (1986) 'The Statistical Analysis of Compositional Data', Chapman & Hall (simplex covariance propagation).
- **Refutability test:** Run JointOptimizer.optimize on a 5-element synthetic with degenerate compositions (e.g. C = [0.5, 0.4, 0.05, 0.03, 0.02]). Compute parameter_uncertainties via current code. Independently compute σ_C via Monte Carlo sampling from cov[2:, 2:] through the full softmax map. The reported C_i uncertainty will be 30-50% lower than the MC truth for trace elements because off-diagonal -c_i*c_j cancellations are missing.
- **Suggested fix:** Replace the diagonal expression with the full Jacobian: build J of shape (E, E) with J[i,i]=c_i(1-c_i), J[i,j]=-c_i*c_j, then var(c) = diag(J @ cov[2:,2:] @ J.T).
- **Blast radius:** Only affects JointOptimizationResult.parameter_uncertainties[f'C_{el}']; downstream readers should expect modest increases for trace elements. No test changes required beyond updating tolerances.
- **Estimated macro_F1 lift:** 0 (uncertainty only; does not change point estimates or macro_F1)
- **Verification:** 0/3 skeptics refuted

#### 8. [MEDIUM/high/regime_mismatch] T_corona = 0.8 * T_K hardcoded as 'Hermann 2017' but no published two-region fit constant exists for sub-10 ps Yb:fiber regime

- **File:** `cflibs/inversion/solve/iterative.py:1118-1121,1237-1238,1356,812-823`
- **Expected per lit:** The Hermann two-region core+corona model (the actual paper appears to be Hermann J. (2017) Spectrochim. Acta B 135, 25-31 on plasma-plume modeling) describes a continuously varying T(r) profile, not a fixed 0.8 ratio between core and corona; the ratio is regime-dependent (ns vs ps pulse, ambient pressure, time gate). Marin Roldan et al. (2021) Spectrochim. Acta B 177, 106055 (DOI 10.1016/j.sab.2020.106055) compares 30 ps vs 5 ns on W/Cu and explicitly reports that ps plumes have substantially different plume geometry — there is no published 0.8 ratio for sub-10 ps Yb:fiber.
- **Actual in code:** if self.two_region:
    # Per Hermann (2017), corona temperature is typically 70-90% of core.
    # We use a fixed ratio of 0.8 for the iterative update.
    T_corona = 0.8 * T_K  # line 1121
...
# corona_sensitive elements only:
corona_sensitive = {"Si", "Fe", "Ca", "Al", "Mg"}  # line 812
...
if T_corona is not None and el in corona_sensitive:
    T_saha = 0.3 * T_K + 0.7 * T_corona  # 0.86 * T_K when T_corona=0.8*T  # line 820
- **Citation:** Marin Roldan, A.; Pisarcik, M.; Veis, M.; Drzik, M.; Veis, P. (2021) Spectrochim. Acta B 177, 106055 (DOI 10.1016/j.sab.2020.106055) — ps vs ns CF-LIBS regime comparison. Hermann (2017) — exact bibliographic detail not specified in code comment; the cited 0.8 ratio is not located in any single Hermann paper I can verify and is at best an interpolation.
- **Refutability test:** Run solver.solve(obs, two_region=True) on ps-LIBS spectra with two_region=True vs False. Compare concentrations of Si/Fe/Ca/Al/Mg. If the 0.8 ratio is correct for ps-LIBS, the two_region=True branch should reduce systematic bias on these elements vs an independent ground truth (LA-ICP-MS reference). Current literature offers no validation for this ratio in the 1 ps Yb:fiber regime.
- **Suggested fix:** Expose T_corona_ratio as a ctor parameter (default 0.8 with a docstring noting 'unvalidated for ps-LIBS') and route the constant through `_run_lax_while_loop` so the lax path receives it. Long-term: implement a proper two-region fit (using forward_models/_hermann_two_region) rather than a single fixed ratio.
- **Blast radius:** Affects two_region=True users (currently default False, so opt-in). Adding parameter is backward-compatible.
- **Estimated macro_F1 lift:** unknown (regime-dependent)
- **Verification:** 0/3 skeptics refuted

#### 9. [HIGH/high/missing_feature] Iterative solver lacks Stark-broadening n_e diagnostic; canonical CF-LIBS literature uses it as the primary n_e source

- **File:** `cflibs/inversion/solve/iterative.py:1180-1211,654-660`
- **Expected per lit:** Tognoni et al. (2010) Spectrochim. Acta B 65, 1-14 Section 4.2 lists Stark broadening as the primary n_e diagnostic in CF-LIBS. Aragon & Aguilera (2008) Spectrochim. Acta B 63, 893-916 Section 2 places Stark width above Saha self-consistency for n_e. Safi et al. (2019) CD-SB (DOI 10.1016/j.jare.2019.01.008) explicitly fuses Stark n_e with the Saha-Boltzmann plot for stability at low n_e. The Bayesian forward model (cflibs/inversion/solve/bayesian/atomic.py line 401) already carries `stark_w` per line — the iterative solver does NOT consume it.
- **Actual in code:** # In _solve_python lines 1180-1211: the only n_e update path is pressure balance.
# In _run_lax_while_loop body lines 654-660: same — pressure balance only.
# No call site references stark_w/electron_density_from_stark/HalphaStark in iterative.py.
- **Citation:** Tognoni, E. et al. (2010) Spectrochim. Acta B 65, 1-14, Sec. 4.2 (DOI 10.1016/j.sab.2009.11.006). Aragon & Aguilera (2008) Spectrochim. Acta B 63, 893-916, Sec. 2 (DOI 10.1016/j.sab.2008.05.010). Safi et al. (2019) J. Adv. Res. 18, 1-8 (DOI 10.1016/j.jare.2019.01.008).
- **Refutability test:** Generate a synthetic dataset where Stark-broadened linewidth gives an unambiguous n_e signal (e.g. Hα FWHM of 0.4 nm at n_e=5e17). Invert via IterativeCFLIBSSolver; the recovered n_e will reflect only pressure balance (∼1e15-1e16 at T=10000K, P=1atm), not Stark. Compare to BayesianForwardModel, which exposes stark_w and recovers n_e correctly.
- **Suggested fix:** Add an optional `stark_n_e_estimator: Callable[[LineObservation], Optional[float]] = None` ctor parameter; when supplied, replace the pressure-balance n_e update with the Stark estimate (one or more selected reference lines). Backwards-compatible default keeps existing behavior.
- **Blast radius:** Net-additive parameter; affects ps-LIBS use cases where pressure balance is most wrong. Tests should add a Stark-based n_e fixture. Bayesian solver already does this correctly via BayesianForwardModel — pattern is available.
- **Estimated macro_F1 lift:** +0.05 to +0.15 (per Tognoni 2010 reviews of CF-LIBS performance, using Stark n_e instead of self-consistent Saha typically improves quantitative recovery by 10-30% — the 0.402→0.7 macro_F1 gap leaves room for substantial lift)
- **Verification:** 0/3 skeptics refuted

### 7.6 uncertainty (7 findings)

#### 1. [HIGH/high/missing_feature] saha_factor_with_uncertainty treats n_e as scalar and is never wired into the analytical pipeline (n_e uncertainty not propagated)

- **File:** `cflibs/inversion/physics/uncertainty.py:170-212`
- **Expected per lit:** Cavalcanti et al. 2013 (Spectrochim. Acta B 87:51-56) and Borges et al. 2019 (Spectrochim. Acta B 160:105692) explicitly propagate n_e uncertainty into T via the Saha equation in their OPC + Saha-Boltzmann workflow, reporting <0.5% T and n_e uncertainty when this coupling is included. Aragon & Aguilera 2008 (Sec. 3 / Eq. 5) and Aguilera & Aragon 2007 establish that the Saha factor S ~ (T^1.5/n_e)*exp(-IP/T) makes any element-concentration estimate depend jointly on (T, n_e); ignoring sigma_n_e systematically underestimates concentration uncertainty. Poggialini et al. 2023 (J. Anal. At. Spectrom. 38:1751) makes the absence of joint (T, n_e, intercept) propagation a central open question for CF-LIBS.
- **Actual in code:** def saha_factor_with_uncertainty(
    T_eV_u: "UFloat",
    n_e: float,                                    # <- bare float, not UFloat
    ionization_potential_eV: float,
    ...
) -> "UFloat":
    ...
    # Note: Currently n_e is treated as exact (no uncertainty).
    # For full uncertainty propagation, n_e would need to be a ufloat.
    ...
    S_raw = (saha_const / n_e) * (T_eV_u**1.5) * umath.exp(-ionization_potential_eV / T_eV_u)
- **Citation:** Cavalcanti et al. 2013, Spectrochim. Acta B 87:51-56 (10.1016/j.sab.2013.05.016); Borges et al. 2019, Spectrochim. Acta B 160:105692; Aragon & Aguilera 2008, Spectrochim. Acta B 63:893-916.
- **Refutability test:** Add a UFloat n_e argument to saha_factor_with_uncertainty, then on a synthetic Fe I+II spectrum with sigma_n_e/n_e = 0.20 (typical Stark-derived precision per Aragon 2008), recompute sigma_T and sigma_C_Fe; expected swing of sigma_C_Fe is at least factor ~1.5 (per Cavalcanti 2013 Fig. 3). Independently, grep the codebase for any caller of saha_factor_with_uncertainty: there are zero, proving the function is not in the analytical pipeline today (verified: find_referencing_symbols on saha_factor_with_uncertainty returns {} and grep finds only the def site + __init__.py re-export).
- **Suggested fix:** Promote n_e to a UFloat throughout saha_factor_with_uncertainty and wire it into solve_with_uncertainty so the Saha correction step contributes its variance through the existing 'uncertainties' computational graph alongside slope_u and y_mean_u.
- **Blast radius:** cflibs/inversion/solve/iterative.solve_with_uncertainty (where intercepts/slope are propagated but Saha is treated as deterministic via _apply_saha_correction); cflibs/inversion/__init__.py (exports the dead function); tests/test_uncertainty.py would need a new TestSahaUncertaintyPropagation suite.
- **Estimated macro_F1 lift:** unknown (does not affect F1 directly — affects calibration of reported sigma_C; correct intervals are needed to drive a downstream Bayesian-prior or quality-cut step that *could* affect F1).
- **Verification:** 0/3 skeptics refuted

#### 2. [HIGH/certain/missing_feature] run_monte_carlo_uq convenience function silently drops atomic-data uncertainty (NIST A_ki grades never sampled)

- **File:** `cflibs/inversion/physics/uncertainty.py:1101-1149`
- **Expected per lit:** Gornushkin & Voelker 2022 (Sensors 22:7149) identifies single-line elements as the dominant uncertainty source in MC CF-LIBS precisely because A_ki uncertainty (NIST grades up to 50% for D-grade transitions) dominates the variance budget for those elements (3.74% concentration error on Si with one line). Bousquet et al. 2023 (Spectrochim. Acta B 204:106686) shows the Boltzmann-plot apparent temperature is biased by A_ki errors >10% even when S/N is high. JCGM 101:2008 specifies that *all* input PDFs that contribute non-negligibly must be sampled simultaneously to obtain a coverage interval — sampling only spectral noise gives the wrong distribution for the output.
- **Actual in code:** def run_monte_carlo_uq(
    solver: Any,
    observations: List[Any],
    n_samples: int = 200,
    noise_fraction: float = 0.05,
    seed: int = 42,
    closure_mode: str = "standard",
    **closure_kwargs,
) -> MonteCarloResult:
    ...
    mc = MonteCarloUQ(solver, n_samples=n_samples, seed=seed, verbose=False)
    return mc.run(
        observations,
        noise_fraction=noise_fraction,
        closure_mode=closure_mode,
        **closure_kwargs,
    )                  # <- no atomic_uncertainty, no perturbation_type=COMBINED
- **Citation:** Gornushkin & Voelker 2022, Sensors 22:7149 (10.3390/s22197149); JCGM 101:2008 Sec. 5-7 (BIPM Monte Carlo Supplement to GUM).
- **Refutability test:** On a synthetic Fe/Cu spectrum where Fe has 8 lines and Cu has 2 lines (one with NIST grade D = 50% uncertainty), run (a) run_monte_carlo_uq with defaults and (b) MonteCarloUQ().run(..., perturbation_type=COMBINED, atomic_uncertainty=AtomicDataUncertainty.from_transitions(transitions)). The sigma_C_Cu from (b) should be >= 2x the sigma_C_Cu from (a) per Gornushkin 2022 single-line-element scaling. Also: AtomicDataUncertainty.from_transitions has zero non-test callers in the codebase (search_for_pattern confirms this), proving the NIST-grade wiring is unused in production.
- **Suggested fix:** Have run_monte_carlo_uq accept (and default to) perturbation_type=COMBINED and auto-build atomic_uncertainty via AtomicDataUncertainty.from_transitions(observations) so the default user gets a JCGM-101-compliant interval; emit a logger.warning when falling back to default 10%.
- **Blast radius:** cflibs/inversion/physics/uncertainty.py (signature change to run_monte_carlo_uq); any CLI entrypoint that calls run_monte_carlo_uq; tests/test_uncertainty.py TestRunMonteCarloUQFunction would need a COMBINED-mode case.
- **Estimated macro_F1 lift:** unknown (calibration-only; could affect F1 indirectly if MC sigma is used to gate line acceptance in identify/).
- **Verification:** 0/3 skeptics refuted

#### 3. [MEDIUM/high/algorithmic_wrong_equation] Monte Carlo perturbation treats A_ki errors as independent per line, ignoring perfect intra-multiplet correlation

- **File:** `cflibs/inversion/physics/uncertainty.py:918-955`
- **Expected per lit:** All lines from a common upper level share a single measured branching fraction and a single experimental lifetime, so their A_ki values are perfectly correlated within the multiplet — varying them independently overstates the random component and understates the systematic component of the resulting Boltzmann-slope/intercept error. Bousquet et al. 2023 (Spectrochim. Acta B 204:106686) emphasizes that A_ki errors enter the Boltzmann fit as *systematic* offsets per upper level, not as independent jitter. Aguilera & Aragon 2007 (Spectrochim. Acta B 62:378-385) builds the multi-element Saha-Boltzmann fit specifically to exploit per-level structure, implying the uncertainty model must respect the same grouping.
- **Actual in code:** for obs in observations:
    ...
    if perturbation_type in (PerturbationType.ATOMIC_DATA, PerturbationType.COMBINED):
        if atomic_uncertainty is not None:
            frac_err = atomic_uncertainty.get_uncertainty(obs.wavelength_nm)
            sigma_A = frac_err * A_ki
            A_ki = A_ki + rng.normal(0, sigma_A)   # independent draw per line
            A_ki = max(A_ki, 1.0)
- **Citation:** Bousquet et al. 2023, Spectrochim. Acta B 204:106686; Aguilera & Aragon 2007, Spectrochim. Acta B 62:378-385 (10.1016/j.sab.2007.03.024).
- **Refutability test:** On a Fe I multiplet at 380 nm (5 lines, common upper level a^5P3 -> a^5D), inject the same fractional shift epsilon to all 5 A_ki values vs. independent N(0, sigma) draws of the same RMS magnitude; compare sigma_T from N=500 MC. The independent draw will report a smaller sigma_T because the (random) draws partially cancel in the slope fit, whereas the physically-correct correlated draw shifts the slope coherently. The discrepancy is expected to be of order sqrt(N_multiplet_lines) per central-limit reasoning.
- **Suggested fix:** Group observations by (element, ionization_stage, E_k_ev) (or by an explicit upper-level id from the atomic DB) and draw one fractional A_ki shift per group, applied to all members; this matches the upper-level branching-ratio uncertainty structure that NIST grades actually describe.
- **Blast radius:** AtomicDataUncertainty and MonteCarloUQ._perturb_observations would need to accept a multiplet/upper-level grouping (already available as multiplet_groups in boltzmann.py fit signature); tests/test_uncertainty.py TestMonteCarloUQUnit; downstream MC variance reports.
- **Estimated macro_F1 lift:** unknown.
- **Verification:** 0/3 skeptics refuted

#### 4. [MEDIUM/high/regime_mismatch] Monte Carlo intensity perturbation uses additive Gaussian with hard floor at 1.0, biasing low-intensity ps-LIBS lines

- **File:** `cflibs/inversion/physics/uncertainty.py:924-934`
- **Expected per lit:** Yamaguchi et al. 2019 (Spectrochim. Acta B 154:25-34) decompose LIBS signal uncertainty into column-density, T, and n_e contributions via explicit error propagation and show shot-noise / detector-noise floors dominate at low intensity, where Gaussian fractional noise breaks down. JCGM 101:2008 requires the sampled PDF for an input to match the true noise distribution (Poisson at low N, not constant-fractional Gaussian). Truncating I_perturbed at max(I, 1.0) imposes a one-sided clip that biases the mean upward for low-S/N lines, exactly the regime where ps-LIBS lives (lower continuum, lower line intensities than ns-LIBS literature assumes).
- **Actual in code:** if perturbation_type in (PerturbationType.SPECTRAL_NOISE, PerturbationType.COMBINED):
    if noise_fraction is not None:
        sigma = noise_fraction * intensity
    else:
        sigma = obs.intensity_uncertainty

    if sigma > 0:
        intensity = intensity + rng.normal(0, sigma)
        # Ensure positive intensity
        intensity = max(intensity, 1.0)
- **Citation:** Yamaguchi, Nishi, Sakka 2019, Spectrochim. Acta B 154:25-34 (S0584854719300291); JCGM 101:2008 Sec. 6 (Choice of input PDF).
- **Refutability test:** Generate a synthetic ps-LIBS spectrum with mean line intensity ~ 20 counts and Poisson noise; run MC with noise_fraction=0.05 (current path) and with a proper Poisson sampler (rng.poisson(intensity)). Compare sigma_T: the Poisson path should yield a 30-50% larger sigma_T at these low counts (per Yamaguchi 2019 Fig. 2). Also verify the empirical mean of the perturbed intensity exceeds the original by O(sigma) for lines near the 1.0 floor — that bias is the smoking gun.
- **Suggested fix:** Provide a noise_model enum {gaussian_fractional, poisson, additive_floor} and select Poisson (or compound Poisson + Gaussian read-noise) automatically when intensity values are O(10) — the appropriate ps-LIBS regime per Yamaguchi 2019.
- **Blast radius:** MonteCarloUQ._perturb_observations and run_monte_carlo_uq; documented contract with tests/test_uncertainty.py TestMonteCarloUQUnit; CFLIBSResult uncertainty calibration on ps-LIBS data.
- **Estimated macro_F1 lift:** unknown.
- **Verification:** 0/3 skeptics refuted

#### 5. [MEDIUM/high/implementation_bug] Bayesian comparison assumes Gaussian posterior (mean +/- 2 sigma) instead of using actual credible intervals

- **File:** `cflibs/inversion/physics/uncertainty.py:718-726`
- **Expected per lit:** JCGM 101:2008 Sec. 7.7-7.8 specify that a 95% coverage interval must be derived from the percentiles of the output distribution, not from a Gaussian 2-sigma proxy. For ps-LIBS where the posterior on n_e is strongly log-skewed (the project's BayesianForwardModel parameterizes log_n_e for exactly this reason — see compare_with_bayesian's special-case for log_ne_mean), a Gaussian 2-sigma window in linear n_e space severely misrepresents the actual 95% credible interval, producing false 'AGREE'/'DISAGREE' verdicts.
- **Actual in code:** mc_T_lower, mc_T_upper = self.T_ci_95
bayes_T_lower = bayes_T_mean - 2 * bayes_T_std
bayes_T_upper = bayes_T_mean + 2 * bayes_T_std
ci_overlap_T = not (mc_T_upper < bayes_T_lower or mc_T_lower > bayes_T_upper)

mc_ne_lower, mc_ne_upper = self.ne_ci_95
bayes_ne_lower = bayes_ne_mean - 2 * bayes_ne_std
bayes_ne_upper = bayes_ne_mean + 2 * bayes_ne_std
- **Citation:** JCGM 101:2008 Sec. 7.7 (Coverage intervals) and Sec. 7.8 (Validation by comparison with GUM uncertainty framework).
- **Refutability test:** Construct a synthetic posterior with log-normal n_e (e.g., mean = 1e17, sigma_log10 = 0.3), where the true 95% CI is asymmetric. The current code computes a symmetric [mean - 2*mean*sigma_log10*ln10, mean + 2*mean*sigma_log10*ln10] window. Compare against the true [10^(log10_mean - 1.96*sigma_log10), 10^(log10_mean + 1.96*sigma_log10)] window. The lower bound mismatches by >50% for any sigma_log10 > 0.2 typical of MCMC on n_e (per the BayesianForwardModel log_ne prior used in cflibs/inversion/solve/bayesian.py).
- **Suggested fix:** Require the bayesian_result to expose actual percentile-based intervals (e.g., T_ci_95, log_ne_ci_95) and use those directly; remove the Gaussian fallback or treat it as a last-resort with an explicit warning.
- **Blast radius:** Only MonteCarloResult.compare_with_bayesian; affects validation reports comparing Bayesian vs MC paths; no production solver impact.
- **Estimated macro_F1 lift:** unknown (validation tooling only).
- **Verification:** 0/3 skeptics refuted

#### 6. [HIGH/high/missing_feature] Partition function uncertainty is treated as exact in propagate_through_closure_*; combined with audit defect C this is the dominant unmodelled error source

- **File:** `cflibs/inversion/physics/uncertainty.py:215-265, 268-327, 330-383`
- **Expected per lit:** Barklem & Collet 2016 (A&A 588:A96, NIST Atomic Spectra Database web tool source) tabulate U_s(T) with explicit uncertainties from underlying level-energy and statistical-weight errors. The prior physics audit (Wave-1 defect C, docs/architecture/2026-05-27-physics-audit.md) found that the project's partition functions disagree with direct-sum by 2-3x at LIBS temperatures — that 2-3x systematic enters as a multiplicative factor on every C_s via the closure equation C_s = U_s*exp(q_s) / sum(U*exp(q)), so propagating zero variance for U_s is incompatible with the size of its actual uncertainty. Poggialini et al. 2023 (J. Anal. At. Spectrom. 38:1751) lists partition-function uncertainty as an open problem in CF-LIBS quantitative uncertainty budgets.
- **Actual in code:** for element, q_s_u in intercepts_u.items():
    if element not in partition_funcs:
        logger.warning(f"Missing partition function for {element}")
        continue

    U_s = partition_funcs[element]                       # <- bare float, no uncertainty
    multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
    rel_C_u = multiplier * U_s * umath.exp(q_s_u)        # <- U_s enters as exact
- **Citation:** Barklem & Collet 2016, A&A 588:A96 (10.1051/0004-6361/201526961); Poggialini et al. 2023, J. Anal. At. Spectrom. 38:1751-1771 (10.1039/D3JA00130J).
- **Refutability test:** Replace partition_funcs[element]: float with partition_funcs[element]: UFloat carrying a 10% uncertainty (a conservative estimate for refractory species at LIBS T, per the Wave-1 audit defect C). Re-run propagate_through_closure_standard on a 5-element steel composition; expected change in concentration sigma is dominated by sigma_U / U term (i.e., +10% added in quadrature per element), trivially refutable if the recomputed sigma_C does not grow.
- **Suggested fix:** Allow partition_funcs values to be UFloat (or accept an optional per-element partition_uncertainties dict) so the closure propagation includes sigma_U via the existing 'uncertainties' graph; default magnitudes per Barklem & Collet 2016 tables where available, fallback to 10%.
- **Blast radius:** All three closure propagation functions; iterative.solve_with_uncertainty would need to pass UFloat partition functions (requires partition-function code path to expose its own uncertainty); test_uncertainty.py TestClosurePropagation.
- **Estimated macro_F1 lift:** unknown (calibration-only; but mis-stated sigma_C breaks every downstream test that compares 'agreement within 1 sigma').
- **Verification:** 1/3 skeptics refuted

#### 7. [LOW/certain/implementation_bug] compare_with_bayesian silently substitutes T=1 eV / n_e=1e17 fallbacks when the Bayesian result lacks expected attributes

- **File:** `cflibs/inversion/physics/uncertainty.py:697-708`
- **Expected per lit:** JCGM 101:2008 Sec. 7.9 and ISO/IEC 17025 require uncertainty comparisons to be traceable to the actual posterior — using a hardcoded default 1 eV temperature or 1e17 cm^-3 n_e when the Bayesian object lacks those attributes silently fabricates the comparison, producing a numerical 'agreement' against an arbitrary value rather than against the actual posterior. This is a documented anti-pattern in JCGM 100 (Type-A uncertainty must be traceable).
- **Actual in code:** bayes_T_mean = getattr(bayesian_result, "T_eV_mean", 1.0) * EV_TO_K
bayes_T_std = getattr(bayesian_result, "T_eV_std", 0.1) * EV_TO_K
bayes_ne_mean = getattr(bayesian_result, "n_e_mean", 1e17)

# Handle log_ne for nested sampling
if hasattr(bayesian_result, "log_ne_mean"):
    bayes_log_ne_std = getattr(bayesian_result, "log_ne_std", 0.1)
    # Propagate uncertainty from log to linear scale
    bayes_ne_std = bayes_ne_mean * bayes_log_ne_std * np.log(10)
else:
    bayes_ne_std = bayes_ne_mean * 0.1  # Fallback
- **Citation:** JCGM 100:2008 Sec. 5.1.5 (uncertainty traceability); JCGM 101:2008 Sec. 7.9.
- **Refutability test:** Call MonteCarloResult.compare_with_bayesian(object()) (a bare object with no attributes); the function does not raise — it returns a numerical comparison against bayes_T_mean = 11604.5 K and bayes_ne_mean = 1e17. Any user reading the output would mistakenly take this as a real comparison.
- **Suggested fix:** Raise TypeError/AttributeError when required posterior summaries are missing; do not return a fabricated comparison.
- **Blast radius:** MonteCarloResult.compare_with_bayesian and its callers in validation scripts; not in the solver hot path.
- **Estimated macro_F1 lift:** unknown.
- **Verification:** 1/3 skeptics refuted


---

# Wave-2 Follow-up — boltzmann-saha + bayesian + manifold (2026-06-05)

These 3 topics pipeline-failed in the main run (arxiv MCP permission hang). Re-audited with embedded self-refutation (each finding had to survive its own best charitable counter-reading) and deduplicated against the 8 main-run families. **15 new findings** survived.

## 8. New defect families (7)

### HIGH — LTE-gate correctness: wrong Delta-E convention + incomplete McWhirter coverage

**Topics:** boltzmann-saha, bayesian  
**Expected lift:** indirect/low directly (gate is advisory) but enables +0.005-0.02 by rejecting non-LTE shell/marginal draws that corrupt composition; primary value is correctness of the LTE validity claim at ps-LIBS n_e~1e16

**Root cause:** The McWhirter criterion n_e >= 1.6e12*sqrt(T)*Delta-E^3 is applied with the wrong Delta-E (largest gap between ADJACENT observed upper-level energies ~0.2 eV instead of the term-scheme resonance ground->first-excited gap ~3-5 eV), deflating the required density by Delta-E^3 (1-2 orders of magnitude). Compounding this, the Bayesian single-zone model has no LTE penalty at all and the two-zone model gates only on T_core, never the cooler shell where LTE is the binding constraint.

**Findings:**
- McWhirter LTE criterion uses adjacent observed-E_k gap instead of the term-scheme/resonance energy gap, deflating n_e_required by Delta-E^3
- McWhirter LTE soft-penalty absent from single-zone model and gated on T_core only in two-zone (T_shell LTE never checked)


### MEDIUM — Saha-Boltzmann internal consistency: IPD truncation dropped + divergent Debye-Huckel formulas

**Topics:** boltzmann-saha  
**Expected lift:** low (sub-percent ionization-fraction bias at ps-LIBS densities today; prevents a future 44% IPD inconsistency)

**Root cause:** Continuum lowering is applied inconsistently. The Saha exponent uses the IPD-lowered ionization potential (eff_ip_I = ip - delta_chi) but the direct-sum partition-function path silently discards the matching cutoff (evaluate_direct receives raw ip_ev and n_e=None), so U(T) is summed to the full ionization limit — only the Mahalanobis half of one physical effect is applied. Separately, two divergent 'Debye-Huckel' IPD formulas coexist (partition.py 3e-8 coefficient vs saha_boltzmann.py Gaussian-CGS), differing by 1.44x, a latent correctness landmine the moment IPD-truncation is enabled.

**Findings:**
- IPD-lowered partition-function cutoff is silently discarded: Saha exponent is IPD-lowered but U(T) truncation is not
- Two divergent Debye-Huckel IPD formulas coexist: partition.py 3e-8 form is 1.44x the canonical Gaussian-CGS form in saha_boltzmann.py


### HIGH — Bayesian inference defects: fabricated MCMC convergence + model-variance likelihood bias

**Topics:** bayesian  
**Expected lift:** 0.005-0.02 (bright-line-dominated retrievals, likelihood); convergence fix is indirect — prevents silent acceptance of bad posteriors

**Root cause:** Two independent statistical defects in the Bayesian sampler/likelihood. (1) Default single-chain runs report r_hat hardcoded to 1.0 and ess=N (uncorrected for autocorrelation), so _assess_convergence trivially returns CONVERGED even for deliberately under-warmed chains — non-converged posteriors are silently accepted. (2) The Gaussian likelihood puts the MODEL prediction in the per-bin variance (sigma^2 = predicted/gain + readout^2 + dark) plus a model-dependent 0.5*sum log(2*pi*variance) normalization, the classic Pearson/model-variance bias that rewards shrinking predicted intensity on bright lines.

**Findings:**
- Default single-chain MCMC reports fake-perfect convergence (R-hat hardcoded 1.0, ESS=N uncorrected)
- Gaussian likelihood uses model-predicted intensity in the variance (Pearson/model-variance bias) with a model-dependent log-determinant term


### HIGH — Manifold basis-library & vector-index internals: Gaussian-only shapes, asymmetric metric, cross-element voting, untrained quantizer

**Topics:** manifold  
**Expected lift:** 0.05-0.12 (Gaussian-only basis shape dominates) + 0.02-0.05 (cross-element voting) + 0.01-0.03 (asymmetric metric); <0.02 IVF/PQ (only when explicitly selected)

**Root cause:** The basis-library/FAISS-index stack has multiple internal-physics and ANN defects independent of the Stark/Doppler cascade: (a) basis lines rendered as a single wavelength-independent Gaussian (instrument FWHM in nm, no Doppler lambda-scaling, no Stark Lorentzian wing) — narrower and wrong-shaped vs the Voigt manifold generator; (b) _nearest_grid_idx normalizes the T-axis by T_max^2 but n_e by the log10 RANGE, an asymmetric metric biased ~10x toward density; (c) BasisIndex (T,ne) estimate is an inverse-squared-distance weighted median over k=50 neighbors pooled across ALL elements with no per-element restriction or collinear-endmember pruning; (d) IVF/PQ build reuses identical embeddings for train()/add() with no n_train >= ~39*n_lists guard and quantizes un-whitened PCA scores.

**Findings:**
- Basis-library line shapes use a single wavelength-independent Gaussian (instrument-FWHM-in-nm only): no Doppler scaling, no Voigt
- BasisLibrary._nearest_grid_idx normalizes T-distance by T_max^2 but n_e-distance by the log10 range
- BasisIndex (T, n_e) estimate is an inverse-(squared-)distance weighted median over k=50 neighbors drawn from ALL elements with no per-element restriction and no collinear-endmember pruning
- VectorIndex IVF/PQ build reuses the same embeddings for train() and add() with no n_train >= ~39*n_lists guard and feeds un-whitened PCA scores to product quantization


### HIGH — Manifold generic composition grid violates simplex closure

**Topics:** manifold  
**Expected lift:** 0.03-0.10 for any non-quartet element set (the common case)

**Root cause:** For any element count != 4, the generic grid branch sweeps only the FIRST element over linspace(0.5,1.0) and pins all other concentrations to 0.0, so each row sums to C_1 (not 1) and the interior of the (D-1)-simplex is never populated. Only the hardcoded 4-element Ti-Al-V-Fe quartet builds a real simplex. Any manifold for an arbitrary --elements list is structurally degenerate and unusable as a composition prior or coarse-init.

**Findings:**
- Generic (≠4-element) manifold composition grid sweeps only the first element and pins all others to zero — violates simplex closure, never explores the interior of composition space


### HIGH — Standalone Boltzmann fitter weighting error (1/sigma^4 over-weighting)

**Topics:** boltzmann-saha  
**Expected lift:** medium (T bias propagates into every Saha correction and composition estimate that uses the standalone fitter)

**Root cause:** BoltzmannPlotFitter._compute_weights returns 1/sigma^2 and passes it straight into numpy.polyfit's w argument, but polyfit treats w as 1/sigma (squares it internally), yielding effective 1/sigma^4 weighting. The JAX path explicitly re-squares to 'match' the CPU convention, faithfully propagating the bug. At ps-LIBS with few lines and narrow E_k spread, leverage collapses onto 2-3 high-SNR lines (weight ratio up to ~4096:1 instead of 64:1), destabilizing the fitted slope -> T and distorting slope_err. The iterative common-slope solver correctly uses 1/sigma^2, so the two paths disagree.

**Findings:**
- BoltzmannPlotFitter passes inverse-variance weights (1/sigma^2) straight into numpy.polyfit's w argument, producing effective 1/sigma^4 (non-physical over-weighting) and biasing the fitted slope/T


### HIGH — Two-zone Saha truncated at Stage II + prefilter falsy-zero estimate discard

**Topics:** bayesian  
**Expected lift:** 0.01-0.04 (Stage-III for low-second-IP matrices) + 0.005-0.015 (prefilter, rare degenerate-estimate cases)

**Root cause:** Two distinct number-conservation/probe defects in the Bayesian path. (1) TwoZoneBayesianForwardModel forms only neutral+singly-ionized fractions from a single Saha ratio; partition_coeffs[:,2] and ionization_potentials[:,1] (Stage III, loaded with max_stages=3) are never used, so element number is not conserved across stages at ps-LIBS core T for low-second-IP elements (Ca, Mg, Fe). (2) candidate_prefilter uses `getattr(...,_estimated_T,None) or fallback`, so a valid estimated T or n_e of exactly 0 is silently replaced by the fallback, centering the multi-T probe on the wrong temperature and dropping temperature-sensitive trace lines.

**Findings:**
- Two-zone Saha balance truncated at Stage II — no doubly-ionized population despite 3-stage atomic storage
- Candidate prefilter falsy-zero fallback: getattr(...) or fallback silently discards a valid estimated T or n_e of 0


---

## 9. Updates to existing (main-run) defect families

### 1. Stark broadening convention cascade

- Manifold basis_library.py renders lines as a single wavelength-independent Gaussian with NO Stark Lorentzian component (line 116 scalar sigma, line 179 pure-Gaussian kernel), while the manifold generator.py renders full Voigt (Humlicek W4) + Stark. This is a third site in the Stark cascade: the basis fingerprints used for identification are Stark-free and systematically narrower than both the generator output and the measured spectra. Corroborates the cascade from the manifold-internal angle Wave-2 could not reach (manifold topic pipeline-failed).

*Severity:* no change (remains CRITICAL); broaden scope note to include basis_library as a third affected site  
*Lift revision:* no change to family total; the basis-shape contribution is accounted under the new manifold basis-library family to avoid double-counting

### 6. Doppler factor-of-2 bug at 4 sites

- Manifold basis_library.py line shape has NO Doppler term and NO wavelength-dependence at all (constant sigma in nm applied identically from 200 to 900 nm). This is a different defect than the factor-of-2 (it is a total absence of Doppler + missing lambda-scaling rather than a mis-scaled present term), but it lives in the same broadening-convention family and should be fixed in the same pass: sigma_G(lambda) = lambda/(R*2.3548) for fixed-R, or add the Doppler sqrt(T/M) term with correct (non-factor-of-2) normalization.

*Severity:* no change (HIGH)  
*Lift revision:* the basis-library width contribution is counted under the new manifold basis-library family; no double-count added here

### 8. ns-LIBS defaults transplanted into ps-LIBS hardware

- Manifold generator.py hardcodes a us-scale ICCD cooling-trail time-integration (t0=1e-6 s, T~(1+t/t0)^-0.5, n_e~(1+t/t0)^-1, gate_width_s=5e-6, gate_delay_s=300ns, T>0.4eV cutoff) — magic numbers for us-gated ICCD plasmas, not the 1 ps @ 1040 nm Yb:fiber target. Extends the ns->ps defaults family from the bench/instrument layer into the manifold generation kernel; the integrated (T,ne)<->spectrum map is weighted toward long-lived neutral-dominated conditions the ps experiment never reaches.

*Severity:* raise effective priority of family #8 (still MEDIUM severity per-item, but now spans instrument + manifold generator, increasing breadth)  
*Lift revision:* +0.04-0.10 IF the default time-integrated manifold path is used for ps-LIBS (was +0.02-0.05 for the instrument-only scope); revised family range +0.02-0.10

### 7. Uncertainty incomplete variance budget

- The standalone BoltzmannPlotFitter scales its reported slope_err / covariance with the same erroneous 1/sigma^4 weights (cov=True path at line 361), so the reported temperature uncertainty is itself biased — not just the point estimate. This corroborates the variance-budget family from the fitter angle: the covariance returned to downstream uncertainty propagation is computed under the wrong weighting convention. (Note: the point-estimate bug is tracked as its own new family; only the covariance/uncertainty consequence is folded here.)

*Severity:* no change (HIGH)  
*Lift revision:* no change to numeric lift; adds a corroborating mechanism (wrong-weighted covariance) to the existing variance-budget gaps

---

## 10. Consolidated critical path (supersedes §3 ordering)

Merges Wave-2 main steps, critic pre-steps, and follow-up insertions. `(NEW)` = from this follow-up.

```
Step 0.5 — Replace polynomial partition fns with Barklem-Collet direct-sum (critic pre-step; prerequisite for IPD-consistency and LTE-gap work)
Step 0.6 — Manifold cosine/per-axis metric groundwork (critic pre-step)
Step 1 — Stark broadening convention cascade fix (Wave-2 CRITICAL #1)
Step 1.5 — Enforce float64 in manifold atomic arrays (critic pre-step; unblocks basis-library Voigt parity)
Step 1.7 (NEW) — Fix Boltzmann polyfit weight exponent 1/sigma^4 -> 1/sigma^2 (Family I) [depends: none; gates T diagnostic]
Step 2 — Self-absorption / CDSB dead-code revival (Wave-2 CRITICAL #2; bayesian apply_self_absorption=False)
Step 3 — Iterative CF-LIBS n_e diagnostic + silent-failure fix (Wave-2 HIGH #3)
Step 3.5 (NEW) — IPD-consistency in direct-sum Saha path + unify dual Debye-Huckel formulas (Family J) [depends: Step 0.5]
Step 4 — Closure-equation correctness across solver paths (Wave-2 HIGH #4)
Step 4.5 (NEW) — Manifold generic composition grid: Dirichlet/ILR simplex sampling (Family L) [depends: Step 4 closure conventions]
Step 5 — ALIAS Noel-2025 conformance (Wave-2 HIGH #5)
Step 5.5 — Resonance-lines stub (critic pre-step; feeds LTE resonance-gap)
Step 5.7 (NEW) — LTE criterion: term-scheme/resonance Delta-E in lte_validator + add McWhirter penalty to single-zone & shell bayesian models (Family K) [depends: Step 5.5]
Step 6 — Doppler factor-of-2 fix at 4 sites + basis-library wavelength-dependent Gaussian width (Wave-2 HIGH #6, extended)
Step 6.5 (NEW) — Basis library: render Voigt (Doppler+Stark) line shapes to match generator (Family M, ties Stark #1 + Doppler #6) [depends: Step 1, Step 6]
Step 7 — Uncertainty variance-budget completion (Wave-2 HIGH #7)
Step 7.5 (NEW) — Bayesian likelihood: Neyman/observed-variance + MCMC convergence diagnostics (Families N, O) [depends: none]
Step 8 — ps-LIBS hardware defaults + manifold cooling-trail/gate config (Wave-2 MEDIUM #8, extended)
Step 8.3 (NEW) — Two-zone Stage-III Saha + candidate-prefilter falsy-zero fix (Family P)
Step 8.6 (NEW) — Basis-index per-element voting + per-axis metric + IVF/PQ guards (Family M cont.) [depends: Step 0.6]
Step 21.5 — Forward-model parity oracle (critic)
Step 27 — Test-fixture independence (critic)
Step 28 — PCA whitening (critic; aligns with IVF/PQ guard)
```

---

## 11. Remediation-plan changes from follow-up

### [INSERT after step 1] Fix BoltzmannPlotFitter polyfit weighting: pass w=1/sigma (true 1/sigma^2 WLS), drop JAX compensating square

- **Effort:** S
- **Expected lift:** medium (T bias into every Saha correction using the standalone fitter)
- **Category:** implementation_bug
- **Files:** `cflibs/inversion/physics/boltzmann.py`, `cflibs/inversion/physics/boltzmann_jax.py`
- **Validation gate:** Synthetic Boltzmann plot with two sigma-classes (0.05, 0.4); fitter slope must match independent WLS with w=1/sigma^2 to rtol<1e-3; CPU and JAX paths agree; regression test asserts effective weight ratio is 64:1 not 4096:1.
- **Risk:** low — localized to fitter; iterative common-slope solver already correct so no cross-path conflict
- **Rationale:** The standalone single-species temperature fitter currently applies effective 1/sigma^4 weighting, collapsing leverage onto 2-3 high-SNR lines and biasing T. T feeds every Saha correction and composition estimate, so this gates downstream accuracy. Independent of Stark (Step 1); place immediately after the first critical fix.

### [INSERT after step 3] IPD consistency: forward eff_ip/n_e into direct-sum partition path; unify dual Debye-Huckel IPD formulas into one shared ipd_eV()

- **Effort:** M
- **Expected lift:** low (sub-percent ionization bias today; prevents future 44% IPD inconsistency)
- **Category:** algorithmic_wrong_equation
- **Files:** `cflibs/plasma/saha_boltzmann.py`, `cflibs/plasma/partition.py`, `cflibs/inversion/solve/iterative.py`, `cflibs/inversion/solve/closed_form.py`, `cflibs/validation/round_trip.py`
- **Dependencies:** 0.5
- **Validation gate:** Assert partition.ionization_potential_depression == saha_boltzmann.ionization_potential_lowering to rel=1e-6 at (1e17,1e4) and (1e18,12000); assert evaluate_direct U(T) changes when n_e supplied for Fe I near IP; Saha exponent and U truncation use the same eff_ip.
- **Risk:** medium — touches shared partition path used by forward + inversion; needs the Barklem-Collet direct-sum refactor (Step 0.5) landed first
- **Rationale:** The IPD-lowered cutoff is computed but discarded (evaluate_direct gets raw ip, n_e=None), and two Debye-Huckel formulas differ by 1.44x. Both are correctness landmines that activate the moment IPD-truncation is enabled. Depends on Step 0.5 (direct-sum partition fns) so the shared cutoff has a home.

### [INSERT after step 4] Manifold generic composition grid: Dirichlet/ILR simplex sampling for arbitrary D (replace first-element-only sweep)

- **Effort:** M
- **Expected lift:** 0.03-0.10 for any non-quartet element set
- **Category:** algorithmic_wrong_equation
- **Files:** `cflibs/manifold/generator.py`
- **Dependencies:** 4
- **Validation gate:** generate_manifold(elements=['Fe','Cu','Ni']) yields a params array where all 3 concentration columns are non-zero and every row sums to 1.0 (rtol<1e-9); interior simplex coverage test (no all-zero columns).
- **Risk:** medium — changes manifold parameter layout; coarse_to_fine.py consumers read per-element coarse_params and must be re-validated
- **Rationale:** For any element count != 4 the grid sweeps only the first element and zeros the rest, violating closure and never exploring the simplex interior. Every non-Ti-Al-V-Fe manifold is structurally unusable. Depends on Step 4 to inherit the canonical simplex/closure convention used elsewhere.

### [INSERT after step 5] LTE gate correctness: use term-scheme/resonance Delta-E in lte_validator; add McWhirter penalty to single-zone bayesian model and evaluate it at T_shell in two-zone

- **Effort:** M
- **Expected lift:** indirect/low directly; +0.005-0.02 via rejecting non-LTE shell/marginal draws
- **Category:** algorithmic_wrong_equation
- **Files:** `cflibs/plasma/lte_validator.py`, `cflibs/inversion/solve/bayesian/forward.py`
- **Dependencies:** 5.5
- **Validation gate:** Clustered E_k={3.0,3.2,3.4,3.6} eV at T=1e4,n_e=1e15: validator must FAIL McWhirter using resonance Delta-E~3.5 eV (currently PASSES with ~0.2 eV); single-zone bayesian_model contains a mcwhirter penalty factor; two-zone evaluates penalty at T_shell_eV.
- **Risk:** medium — requires per-species resonance Delta-E plumbed from energy_levels (Step 5.5 resonance-lines stub) with term-scheme fallback
- **Rationale:** lte_validator deflates Delta-E by using the adjacent observed-E_k gap (~0.2 eV) instead of the resonance/term-scheme gap (~3-5 eV), inflating LTE pass rate by Delta-E^3. Single-zone bayesian has no LTE gate; two-zone checks only T_core, not the cooler shell where LTE breaks first. Critical at ps-LIBS n_e~1e16. Depends on Step 5.5 resonance-lines stub for the resonance gap.

### [MODIFY modifies step 6] Doppler factor-of-2 fix at 4 sites + basis-library wavelength-dependent Gaussian width

- **Effort:** M
- **Expected lift:** +0.02-0.05 (existing) — basis-library width contribution counted under Step 6.5
- **Category:** implementation_bug
- **Files:** `(existing 4 Doppler sites)`, `cflibs/manifold/basis_library.py`
- **Validation gate:** (existing Doppler gates) PLUS: basis_library line FWHM grows with wavelength (250 nm vs 800 nm differ) when fixed-R/Doppler width is enabled; no constant-nm sigma across 200-900 nm.
- **Risk:** low
- **Rationale:** basis_library.py applies a single constant sigma (nm) identically across 200-900 nm with no Doppler term and no lambda-scaling — same broadening-convention family as the Doppler factor-of-2. Fix the missing lambda-dependence in the same pass; the full Voigt shape (Stark wing) is handled in the dependent Step 6.5.

### [INSERT after step 6] Basis library: render Voigt (Doppler+Stark) line shapes matching the manifold generator

- **Effort:** L
- **Expected lift:** 0.05-0.12
- **Category:** regime_mismatch
- **Files:** `cflibs/manifold/basis_library.py`
- **Dependencies:** 1, 6
- **Validation gate:** Fit a Voigt-broadened synthetic Fe spectrum (Stark width ~0.05 nm at n_e=1e17) with the basis: wing residual < threshold; basis line shape matches generator.py Humlicek-W4 output to rtol on FWHM and wing decay.
- **Risk:** medium — must share the generator's profile renderer (cross-cutting line-profile module) to avoid re-introducing divergence
- **Rationale:** Basis fingerprints are Gaussian-only (no Stark Lorentzian, no Doppler) while the generator and measured ps-LIBS data are Voigt. The largest single manifold lift (0.05-0.12). Depends on Step 1 (Stark convention) and Step 6 (Doppler/lambda) so the shared profile renderer is correct before the basis adopts it.

### [INSERT after step 7] Bayesian statistics: Neyman/observed-variance likelihood (or Poisson) + real MCMC convergence diagnostics

- **Effort:** M
- **Expected lift:** 0.005-0.02 (likelihood, bright-line retrievals); convergence fix indirect (prevents silent acceptance of bad posteriors)
- **Category:** algorithmic_wrong_equation
- **Files:** `cflibs/inversion/solve/bayesian/forward.py`, `cflibs/inversion/solve/bayesian/samplers.py`
- **Validation gate:** Likelihood: synthetic-truth recovery shows Neyman/observed-variance form recovers higher/unbiased total predicted intensity vs current model-variance form. Diagnostics: default single-chain run must NOT return r_hat=1.0/ess=N; an under-warmed (num_warmup=1) chain must flag NOT-CONVERGED; ess autocorrelation-corrected << N.
- **Risk:** low — statistical correctness; no physics-only constraint impact (NumPyro/ArviZ already in use)
- **Rationale:** Two independent Bayesian defects: (1) single-chain runs report fabricated r_hat=1.0/ess=N so _assess_convergence always returns CONVERGED; (2) the Gaussian likelihood uses model-predicted intensity in the variance plus a model-dependent log-determinant (Pearson/model-variance bias) shrinking bright-line intensity. Independent of physics fixes; can run in parallel.

### [MODIFY modifies step 8] ps-LIBS hardware defaults + manifold cooling-trail/gate model configurability

- **Effort:** M
- **Expected lift:** +0.02-0.10 (was +0.02-0.05; +0.04-0.10 if default time-integrated manifold path used for ps-LIBS)
- **Category:** regime_mismatch
- **Files:** `(existing instrument/default files)`, `cflibs/manifold/generator.py`
- **Validation gate:** (existing) PLUS: manifold generator exposes t0, cooling exponents, gate_delay_s, gate_width_s as config with ps-LIBS defaults and a single-snapshot mode; single-snapshot vs default-cooling-trail ion/neutral line-ratio test demonstrates the us-tail bias is removed at sub-ns gate.
- **Risk:** low
- **Rationale:** generator.py hardcodes a us-scale ICCD cooling-trail (t0=1e-6, T~^-0.5, n_e~^-1, 5 us gate) wrong for 1 ps Yb:fiber. Extends the ns->ps defaults family into the manifold kernel; the integrated (T,ne)<->spectrum map is biased toward long-lived plasma the ps experiment never reaches.

### [INSERT after step 8] Two-zone Saha Stage-III completion + candidate-prefilter falsy-zero estimate fix

- **Effort:** M
- **Expected lift:** 0.01-0.04 (Stage-III for low-second-IP matrices) + 0.005-0.015 (prefilter)
- **Category:** missing_feature
- **Files:** `cflibs/inversion/solve/bayesian/forward.py`, `cflibs/inversion/candidate_prefilter.py`
- **Validation gate:** Three-stage Saha balance for Ca at T=1.3 eV,n_e=1e17 carried in two_zone_bayesian_model (partition_coeffs[:,2], ionization_potentials[:,1] used); element number conserved across stages to rel<1e-6. Prefilter: identifier._estimated_T=0.0 must be used (not fallback) — replace `or` with explicit `is None` checks.
- **Risk:** low — additive Saha stage + a 2-line operator fix; single-zone unaffected (routes through unified kernel)
- **Rationale:** TwoZoneBayesianForwardModel truncates Saha at Stage II despite max_stages=3, mis-conserving number for low-second-IP elements at ps-LIBS core T. Separately, candidate_prefilter's `getattr(...) or fallback` discards a valid falsy-zero estimate, centering the probe on the wrong T and dropping trace lines.

### [INSERT after step 8] Basis-index correctness: per-element (T,ne) voting, per-axis metric normalization, IVF/PQ training guards + whitening

- **Effort:** M
- **Expected lift:** 0.02-0.05 (cross-element voting) + 0.01-0.03 (asymmetric metric) + <0.02 (IVF/PQ)
- **Category:** algorithmic_wrong_equation
- **Files:** `cflibs/manifold/basis_index.py`, `cflibs/manifold/basis_library.py`, `cflibs/manifold/vector_index.py`
- **Dependencies:** 0.6, 28
- **Validation gate:** _nearest_grid_idx T term divided by (T_max-T_min)^2 (not T_max^2) — matches log10-range n_e normalization; estimate_plasma_params restricts (T,ne) median to winning-element neighbors (or per-element vote weight); IVF build warns/auto-reduces n_lists when n_vectors<39*n_lists; ivf_pq applies PCA whitening/OPQ.
- **Risk:** medium — index parameter changes; needs the cosine/per-axis metric groundwork (Step 0.6) and PCA whitening (Step 28)
- **Rationale:** Three coupled basis-index defects: (a) _nearest_grid_idx normalizes T by T_max^2 but n_e by the log10 RANGE (asymmetric, ~10x density bias); (b) BasisIndex pools k=50 neighbors across ALL elements with no per-element restriction or collinear pruning, letting wrong-element neighbors pull the (T,ne) median; (c) IVF/PQ reuses train()/add() embeddings with no n_train guard and quantizes un-whitened PCA scores. Depends on Step 0.6 (metric) and Step 28 (whitening).

---

## 12. Cross-cutting patterns (follow-up additions)

- DUPLICATED-CONSTANT / DIVERGENT-FORMULA pattern is now confirmed across all 3 deep topics: two Debye-Huckel IPD formulas (partition.py 3e-8 vs saha_boltzmann.py Gaussian-CGS, 1.44x apart), two Boltzmann-weighting conventions (standalone fitter 1/sigma^4 vs iterative common-slope 1/sigma^2), and two line-shape renderers (basis_library Gaussian-only vs generator Voigt). Each is a single physical quantity implemented twice with inconsistent results. Systemic remedy: extract ONE shared module each for (a) ipd_eV(n_e,T,Z), (b) WLS weight policy, (c) line-profile rendering, and have every consumer delegate. This is a cross-cutting refactor that should precede or accompany the family-specific fixes to prevent regression-by-divergence.
- DEAD-ARGUMENT / SILENTLY-IGNORED-PARAMETER pattern: lte_validator computes eff_ip cutoff but evaluate_direct never receives it (passes raw ip, n_e=None); candidate_prefilter computes _estimated_T then discards it via `or` fallback on falsy-zero; basis_library computes a sigma it applies uniformly across 200-900 nm with no lambda use. A linting/contract pass ('every explicitly-computed physical parameter must reach its physics consumer') would catch this class — recommend a dedicated test-oracle step.
- CONVENTION-MISMATCH BETWEEN PAIRED MODELS: single-zone vs two-zone Bayesian models diverge (single-zone has NO McWhirter penalty, two-zone gates on T_core only; two-zone truncates Saha at Stage II despite max_stages=3). The 'two implementations of the same physics' anti-pattern repeats inside Bayesian forward.py. Parity tests between single/two-zone forward models (extend the Step 21.5 parity oracle) would surface these.
- DIAGNOSTIC-FABRICATION pattern: the MCMC sampler reports r_hat=1.0 and ess=N for single chains (fake-perfect convergence) — analogous to the Wave-2 iterative-solver 'silent failure' family where a degenerate computation returns a confident-looking result. Both are cases where the absence of a real check is disguised as a passing check. Recommend a project-wide audit rule: any returned diagnostic/flag must be derived from data, never hardcoded to its pass value.

## 13. ps-LIBS-specific issues (follow-up additions)

- Manifold time-integration hardcodes a us-scale ICCD cooling-trail model (t0=1e-6 s, T~(1+t/t0)^-0.5, n_e~(1+t/t0)^-1, gate_width 5 us, gate_delay 300 ns) that is physically wrong for the 1 ps @ 1040 nm Yb:fiber target (T 0.5-1.3 eV, n_e 1e16-1e18). The integrated (T,ne)<->spectrum map is biased toward long-lived neutral-dominated conditions the ps experiment never reaches. Needs a single-snapshot mode + configurable t0/exponents/gate with ps-LIBS defaults. Extends Wave-2 family #8 (ns->ps defaults) into the manifold generator. (finding: manifold cooling-trail)
- LTE is borderline at ps-LIBS n_e ~ 1e16 cm^-3, so the LTE gate must be RIGHT there. Two findings compound the danger: (a) lte_validator deflates Delta-E (uses adjacent observed-E_k gap ~0.2 eV instead of ~3.5 eV resonance gap), inflating pass rate by 1-2 orders of magnitude in n_e_required; (b) the single-zone Bayesian model has no McWhirter penalty at all and the two-zone checks only T_core, never the cooler shell where LTE breaks first. At ps-LIBS these gaps mean genuinely non-LTE plasmas are silently accepted as LTE.
- Two-zone Saha truncated at Stage II is specifically a ps-LIBS-core risk: at T_core 1.0-1.3 eV with low-second-IP matrix elements (Ca IP_II=11.87, Mg, Fe at high n_e) Stage III becomes non-negligible and element number is not conserved across stages — biasing inferred concentration of exactly the elements that matter for the macro_F1 trace-recall gap.
- Basis-library Gaussian-only fingerprints (no Stark Lorentzian wings, no Doppler lambda-scaling) are most damaging at ps-LIBS n_e where Stark width ~0.05 nm is comparable to instrument width; the measured ps-LIBS lines the manifold/data contain are Voigt-shaped, so NNLS/correlation against a narrow Gaussian basis systematically under-fits wings — a direct contributor to the 0.402-vs-0.7 macro_F1 gap.

---

## 14. New verified findings (appendix)

### 14.x boltzmann-saha (4 findings)

#### 1. [HIGH/high/algorithmic_wrong_equation] McWhirter LTE criterion uses adjacent observed-E_k gap instead of the term-scheme/resonance energy gap, deflating n_e_required by ΔE³

- **File:** `cflibs/plasma/lte_validator.py:244-268 (call site); 99-145 (formula)`
- **Expected per lit:** Cristoforetti et al. 2010 (codifying McWhirter): the criterion n_e >= 1.6e12·sqrt(T)·ΔE³ requires ΔE to be the LARGEST energy gap in the atomic TERM SCHEME, conventionally the resonance ground→first-excited transition (~3-5 eV for typical LIBS species). This is the physically demanding gap because the resonance transition is hardest to keep collisionally coupled.
- **Actual in code:** delta_E_eV is computed as the maximum gap between consecutive sorted OBSERVED upper-level energies of the fitted lines: `energies = sorted({o.E_k_ev for o in observations}); gaps = [energies[i+1]-energies[i] ...]; delta_E_eV = float(max(gaps))` then `delta_E_eV = max(delta_E_eV, 0.1)`. Observed upper-level E_k values cluster narrowly (e.g. Fe I lines mostly 2.5-5 eV), so the largest *consecutive* gap is typically <1 eV — far smaller than the true resonance/term-scheme gap. n_e_required scales as ΔE³, so this underestimates the required density by 1-2 orders of magnitude.
- **Citation:** G. Cristoforetti et al., "Local thermodynamic equilibrium in laser-induced breakdown spectroscopy: Beyond the McWhirter criterion," Spectrochim. Acta B 65(1) (2010) 86-95, DOI 10.1016/j.sab.2009.11.005.
- **Refutability test:** Construct observations with E_k = {3.0, 3.2, 3.4, 3.6} eV (clustered) at T=10000 K, n_e=1e15. validate() will compute ΔE≈0.2 eV → n_e_required≈1.6e12·100·0.008≈1.3e12, so check_mcwhirter PASSES at n_e=1e15. Using the physical resonance ΔE≈3.5 eV → n_e_required≈1.6e12·100·42.9≈6.9e15, which would FAIL. The pass/fail flip on the same plasma proves the convention error.
- **Self-refutation attempt:** Maybe 'largest gap between adjacent levels of interest' legitimately means only the levels actually observed in the fit, since LTE only needs to hold among the populated states sampled by the Boltzmann plot.  → succeeded: False
- **Suggested fix:** Compute ΔE from the species energy_levels table as the ground→first-excited (resonance) gap, or the largest gap up to the first few excited terms, not from the observed line E_k set. Plumb a per-species resonance ΔE through validate(); fall back to the term-scheme max only when level data is absent.
- **Blast radius:** LTEValidator.validate / check_mcwhirter feeds quality_metrics LTE flags; any pipeline relying on the LTE gate to reject non-LTE spectra (or trigger non-LTE corrections) will accept marginal/non-LTE ps-LIBS plasmas. At ps-LIBS n_e down to 1e16, the deflated ΔE masks genuine non-LTE.
- **Est. macro_F1 lift:** indirect/low — gate is advisory; mislabels rather than directly corrupts inversion

#### 2. [MEDIUM/high/implementation_bug] IPD-lowered partition-function cutoff is silently discarded: Saha exponent is IPD-lowered but U(T) truncation is not (max_energy_ev / n_e dropped in direct-sum path)

- **File:** `cflibs/plasma/saha_boltzmann.py:96-105, 131-132, 180-187 (also iterative.py:752-757)`
- **Expected per lit:** Alimohamadi & Ferland 2022 (cited in partition.py:100-102): continuum lowering must be applied consistently — the same Debye-Hückel ΔχIPD that lowers the ionization potential in the Saha exponent must also truncate the partition-function sum at the lowered ionization limit (E_max = IP − ΔχIPD). The exponent reduction and the U(T) truncation are two halves of one physical effect.
- **Actual in code:** solve_ionization_balance computes `delta_chi = self.ipd_model.calculate_lowering(...)`, `eff_ip_I = max(ip_I - delta_chi, 0.0)`, and passes `max_energy_ev=eff_ip_I` into `calculate_partition_function(...)`. But the preferred (direct-sum) branch returns `PartitionFunctionEvaluator.evaluate_direct(T_K, g_arr, E_arr, ip_ev)` — it passes the RAW `ip_ev` from the level cache and `n_e=None` (the default), so `max_energy_ev`/`eff_ip_I` is never used and NO IPD truncation is applied to U(T). The IPD-lowered cutoff only takes effect in the rarely-hit Fallback-2 manual loop (lines 219-226). The Saha exponent uses eff_ip_I (IPD-lowered) while U(T) uses the full level set up to raw ip — an internal inconsistency. iterative.py:757 has the identical pattern.
- **Citation:** P. Alimohamadi & G.J. Ferland, "A Practical Guide to the Partition Function of Atoms and Ions," PASP 134 (2022) 074502, DOI 10.1088/1538-3873/ac7664.
- **Refutability test:** Patch evaluate_direct call to pass n_e and confirm U(T) changes for a species with levels near the IP (e.g. Fe I at n_e=1e18, T=12000K where ΔχIPD≈0.19 eV); compare Saha ratio S1 computed with consistent IPD-truncated U vs current raw-IP U. A nonzero shift in S1 (and hence ionization fractions) at fixed exponent demonstrates the inconsistency is live.
- **Self-refutation attempt:** Maybe the partition sum is dominated by low-lying levels (E_k << IP) so truncating at IP vs IP−ΔχIPD removes only negligible high-lying contributions, making U(T) effectively identical either way — i.e. the dropped cutoff is harmless.  → succeeded: False
- **Suggested fix:** Pass the IPD context into the direct-sum path: either forward `n_e` to evaluate_direct (so it applies ip−ΔχIPD truncation matching the exponent) or pass `eff_ip` as the ip_ev argument. Apply consistently in saha_boltzmann.py:187, iterative.py:757, closed_form.py:79, round_trip.py:365.
- **Blast radius:** All forward Saha ionization balance (SahaBoltzmannSolver.solve_ionization_balance) and the iterative inversion partition evaluation (iterative.py). The U-truncation omission is small at ps-LIBS n_e (ΔχIPD ~0.04-0.2 eV) but the dropped-argument bug means the explicitly-computed eff_ip cutoff is dead code, and the exponent/truncation mismatch is a real (if modest) bias in ionization fractions.
- **Est. macro_F1 lift:** low — sub-percent ionization-fraction bias at ps-LIBS densities

#### 3. [MEDIUM/high/algorithmic_wrong_equation] Two divergent Debye-Hückel IPD formulas coexist: partition.py 3e-8 form is 1.44× the canonical Gaussian-CGS form in saha_boltzmann.py

- **File:** `cflibs/plasma/partition.py:62-86 (partition.ionization_potential_depression); cf. saha_boltzmann.py:385-459, 488-499`
- **Expected per lit:** The canonical Debye-Hückel ionization-potential depression is Δχ = Z·e²/λ_D with λ_D = sqrt(ε0 k_B T / (n_e e²)) (SI) = sqrt(k_B T/(4π n_e e²)) (Gaussian CGS), giving Δχ ≈ 0.066 eV at n_e=1e17 cm⁻³, T=10⁴ K. Alimohamadi & Ferland 2022 (cited at partition.py:100) and Mihalas 1978 Eq. 9-106 are the references. There is exactly one Debye-Hückel value for given (n_e,T,Z).
- **Actual in code:** partition.ionization_potential_depression returns `3.0e-8 * Z * np.sqrt(n_e) / np.sqrt(T_K)` = 0.0949 eV at (1e17, 1e4). saha_boltzmann.ionization_potential_lowering returns the Gaussian-CGS `e²·sqrt(4π n_e/(k_B T))` = 0.0660 eV (and _ipd_eV via _IPD_PREFACTOR_EV_CM_K = `e³·sqrt(4π/k_B)` matches, also 0.066). Verified numerically: the 3e-8 form is exactly 1.44× (=sqrt(4π/2.07)) larger than the textbook Gaussian-CGS form at all (n_e,T). Both docstrings claim 'Debye-Hückel'; the 3e-8 coefficient lacks the correct 1/sqrt(4π) normalization. The Gaussian form's own docstring example asserts `0.03 <= delta_chi <= 0.06` but it actually returns 0.066, slightly above its own stated upper bound.
- **Citation:** P. Alimohamadi & G.J. Ferland, PASP 134 (2022) 074502 (Eq. 13), DOI 10.1088/1538-3873/ac7664; D. Mihalas, Stellar Atmospheres, 2nd ed. (1978), Eq. 9-106.
- **Refutability test:** Assert partition.ionization_potential_depression(1e17,1e4) == pytest.approx(saha_boltzmann.ionization_potential_lowering(1e17,1e4), rel=0.05). It fails (0.0949 vs 0.0660, 44% apart), proving two inconsistent 'Debye-Hückel' implementations.
- **Self-refutation attempt:** Maybe the 3e-8 form is only ever invoked via direct_sum_partition_function with n_e!=None, which all production callers avoid (they pass n_e=None per finding #2), so the divergence is latent dead code and never affects results.  → succeeded: False
- **Suggested fix:** Replace the 3e-8 coefficient with the correctly-normalized Gaussian-CGS Debye-Hückel prefactor (≈2.09e-8 for the e²/λ_D form, matching saha_boltzmann._IPD_PREFACTOR_EV_CM_K), or have partition.py delegate to a single shared ipd_eV() so both modules use one formula. Fix the saha_boltzmann docstring bound to 0.07.
- **Blast radius:** partition.ionization_potential_depression is reached whenever any direct_sum_partition_function/_batch is called with n_e (currently no production caller does, so largely latent today), and direct_sum_partition_function_batch is used in basis-library/manifold generation paths. The defect is a correctness landmine: any future code that enables IPD-truncation will get a 44%-too-large depression inconsistent with the Saha exponent's depression.
- **Est. macro_F1 lift:** negligible today (latent); prevents a future 44% IPD inconsistency

#### 4. [HIGH/high/implementation_bug] BoltzmannPlotFitter passes inverse-variance weights (1/σ²) straight into numpy.polyfit's w argument, producing effective 1/σ⁴ (non-physical over-weighting) and biasing the fitted slope/T

- **File:** `cflibs/inversion/physics/boltzmann.py:_compute_weights 326-... (returns 1/safe_err**2); _fit_sigma_clip 356-380 (np.polyfit(x,y,1,w=weights)); JAX mirror 470-481`
- **Expected per lit:** Aragón & Aguilera 2008 (definitive CF-LIBS diagnostics review): Boltzmann/Saha-Boltzmann plots must use weighted least squares with INVERSE-VARIANCE weights w_i = 1/σ_i². numpy.polyfit's `w` keyword is documented to multiply residuals (it minimizes Σ (w_i·(y_i−ŷ_i))² ), so to achieve inverse-variance weighting one must pass w = 1/σ_i, not 1/σ_i².
- **Actual in code:** `_compute_weights` returns `1.0 / safe_err**2` (= 1/σ²). `_fit_sigma_clip` then calls `np.polyfit(x, y, 1, w=weights, cov=True)` / `np.polyfit(x, y, 1, w=weights)` passing that 1/σ² array directly as polyfit's w. Because polyfit treats w as 1/σ (squares it internally), the effective weighting is (1/σ²)² = 1/σ⁴. The JAX path (lines 472-481) explicitly squares again to 'match' the CPU convention (`kernel_weights = weights * weights`), confirming the intended-vs-actual gap is a bug faithfully propagated. _compute_r_squared compounds it (weights=1/σ² used as variance weights for R²). The iterative common-slope solver (iterative.py:_fit_common_boltzmann_plane) correctly uses 1/σ², so this is confined to the standalone fitter.
- **Citation:** C. Aragón & J.A. Aguilera, "Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods," Spectrochim. Acta B 63(9) (2008) 893-916, DOI 10.1016/j.sab.2008.05.010.
- **Refutability test:** Fit a synthetic Boltzmann plot with two σ-classes (σ_A=0.05, σ_C=0.4) where the high-σ points pull the slope. Compare the BoltzmannPlotFitter slope against an independent statsmodels WLS with weights=1/σ². The fitter's slope (effective 1/σ⁴, weight ratio (0.4/0.05)⁴≈4096:1) will differ materially from the correct 1/σ² WLS (ratio 64:1), demonstrating leverage collapse onto the few A-grade lines.
- **Self-refutation attempt:** Maybe numpy.polyfit treats w as 1/variance (so passing 1/σ² is correct), or the down-weighting of C-grade lines is desirable so the 1/σ⁴ behavior is a feature, not a bug.  → succeeded: False
- **Suggested fix:** Pass w = 1/σ (i.e. sqrt of the inverse-variance) into np.polyfit, or equivalently change _compute_weights to return 1/safe_err for the polyfit call while keeping 1/σ² for R²/covariance bookkeeping. Drop the compensating square in the JAX path so both back-ends apply true 1/σ² inverse-variance weighting consistent with the iterative solver.
- **Blast radius:** BoltzmannPlotFitter.fit (sigma_clip CPU + JAX paths) — the standalone single-species temperature fitter used wherever the iterative common-slope plane is not. At ps-LIBS with few lines and narrow E_k spread, the 1/σ⁴ leverage collapse onto 2-3 high-SNR lines directly destabilizes the slope → fitted T, and also distorts reported slope_err (cov scaled with the same wrong weights).
- **Est. macro_F1 lift:** medium — T bias propagates into every Saha correction and composition estimate that uses the standalone fitter

### 14.x bayesian (5 findings)

#### 1. [HIGH/high/missing_feature] Two-zone Saha balance truncated at Stage II — no doubly-ionized population despite 3-stage atomic storage

- **File:** `cflibs/inversion/solve/bayesian/forward.py:423-435`
- **Expected per lit:** For an LTE plasma the ionization balance is a chain of Saha equilibria n_(z+1)*n_e/n_z = (2 U_(z+1)/U_z) (2 pi m_e k T/h^2)^{3/2} exp(-IP_z/kT). At ps-LIBS core temperatures T~1.0-1.3 eV with low-second-IP matrix elements (e.g. Ca, Mg, Fe at high n_e) the doubly-ionized (Stage III) population is non-negligible and must be carried to conserve element number across stages. Standard CF-LIBS and plasma-spectroscopy treatments retain all stages with appreciable population (Aragon & Aguilera, Spectrochim. Acta B 63 (2008) 893, sec. on Saha-Boltzmann; Cristoforetti et al., Spectrochim. Acta B 65 (2010) 86).
- **Actual in code:** U0 = partition_function(T_K, data.partition_coeffs[:, 0]); U1 = partition_function(T_K, data.partition_coeffs[:, 1]); IP_I = data.ionization_potentials[:, 0]; saha_factor = (_JAX_SAHA_CONST_CM3 / n_e) * (T_eV**1.5); ratio_ion_neutral = saha_factor * (U1 / U0) * jnp.exp(-IP_I / T_eV); frac_neutral = 1.0 / (1.0 + ratio_ion_neutral); frac_ion = ratio_ion_neutral / (1.0 + ratio_ion_neutral). Only two fractions are formed; partition_coeffs[:,2] and ionization_potentials[:,1] (Stage III data, loaded with max_stages=3 at atomic.py:212) are never used.
- **Citation:** Aragon, C. & Aguilera, J.A. (2008). Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods. Spectrochimica Acta Part B 63(9), 893-916. DOI 10.1016/j.sab.2008.05.010.
- **Refutability test:** Pick a low-second-IP element (Ca: IP_I=6.11 eV, IP_II=11.87 eV). Compute the three-stage Saha balance at T=1.3 eV, n_e=1e17 cm^-3 by hand/numpy; Stage III/(Stage II) ratio = (SAHA_CONST/n_e)*T^1.5*(U_III/U_II)*exp(-IP_II/T). If this ratio exceeds ~1% the two-fraction model mis-assigns >1% of Ca atoms and biases the inferred Ca concentration. The code's frac_ion would over-count Stage II by that amount.
- **Self-refutation attempt:** Maybe ps-LIBS plasmas never reach Stage III appreciably because T<=1.3 eV and second IPs are >10 eV, so exp(-IP_II/T) is tiny and the truncation is harmless.  → succeeded: False
- **Blast radius:** TwoZoneBayesianForwardModel only (single-zone routes through the unified kernel). Affects two_zone_bayesian_model posterior composition and TwoZoneMCMCResult.
- **Est. macro_F1 lift:** 0.01-0.04 for matrices containing low-second-IP elements; negligible otherwise

#### 2. [HIGH/certain/implementation_bug] Default single-chain MCMC reports fake-perfect convergence (R-hat hardcoded 1.0, ESS=N uncorrected)

- **File:** `cflibs/inversion/solve/bayesian/samplers.py:158-170, 173-185, 261, 308-309`
- **Expected per lit:** R-hat is a between-chain vs within-chain variance ratio and is undefined for a single chain; Vehtari et al. (2021) recommend >=4 chains, rank-normalized split-R-hat, threshold R-hat<1.01, and an autocorrelation-corrected ESS. A single chain can and should still produce a finite (and possibly poor) split-R-hat and a properly autocorrelation-corrected ESS << N — never ESS=N.
- **Actual in code:** _simple_diagnostics_from_mcmc: ess[var]=float(len(s))  # naive: assume independent samples; r_hat[var]=1.0  # single chain: between-chain variance undefined. _diagnostics_from_mcmc only calls ArviZ when 'HAS_ARVIZ and num_chains > 1' (line 139); MCMCSampler.run defaults num_chains=1 (line 261). _assess_convergence: if max_rhat < 1.01 and min_ess > 100: return CONVERGED. With r_hat=1.0 and ess=num_samples(=1000 default), every default single-chain run trivially returns CONVERGED.
- **Citation:** Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. Bayesian Analysis 16(2), 667-718. DOI 10.1214/20-BA1221.
- **Refutability test:** Run MCMCSampler.run with default num_chains=1 on any spectrum and inspect result.r_hat and result.ess: r_hat will be exactly {1.0,...} and ess will equal num_samples regardless of mixing, and convergence_status will be CONVERGED even for a deliberately under-warmed (num_warmup=1) chain. A correct diagnostic would flag the under-warmed chain.
- **Self-refutation attempt:** Maybe ArviZ single-chain split-R-hat is always available so this stub is never hit; or maybe users always pass num_chains>=4 so the default is irrelevant.  → succeeded: False
- **Blast radius:** All single-chain NUTS runs via MCMCSampler/NumPyroNUTSSampler. MCMCResult.convergence_status, r_hat, ess are all unreliable at default settings; downstream users gating on convergence_status==CONVERGED are misled.
- **Est. macro_F1 lift:** indirect: prevents silent acceptance of non-converged composition posteriors; no direct F1 number

#### 3. [MEDIUM/high/algorithmic_wrong_equation] Gaussian likelihood uses model-predicted intensity in the variance (Pearson/model-variance bias) with a model-dependent log-determinant term

- **File:** `cflibs/inversion/solve/bayesian/forward.py:568-573, 624-633`
- **Expected per lit:** When the per-bin variance is set to the model prediction (sigma_i^2 = f(predicted_i)) rather than the data, the maximum-likelihood/posterior estimator is biased low relative to using the observed counts, because the included normalization term 0.5*sum log(2 pi variance) rewards solutions that shrink the predicted intensity. This is the documented Pearson-vs-Neyman / model-variance bias for Poisson-like data (Humphrey et al. 2009; Mighell 1999). The Bayesian-correct treatment either uses the observed intensity in the variance (Neyman) or a Poisson likelihood, and only drops the log-variance term when variance is treated as fixed/known.
- **Actual in code:** log_likelihood: pred_safe = jnp.maximum(predicted, 1e-10); variance = pred_safe / gain + readout_noise**2 + dark_current; residual = observed - pred_safe; log_lik = -0.5 * sum(log(2*pi*variance) + residual**2/variance). bayesian_model: variance = pred_safe/gain + readout_noise**2 + dark_current; sigma = sqrt(max(variance,1e-6)); obs ~ Normal(pred_safe, sigma). The shot-noise term predicted/gain (model-dependent) enters BOTH the Mahalanobis term and, via dist.Normal's implicit log(sigma), the normalization.
- **Citation:** Humphrey, P. J., Liu, W., & Buote, D. A. (2009). chi^2 and Poissonian Data: Biases Even in the High-Count Regime and How to Avoid Them. The Astrophysical Journal 693(1), 822-829. arXiv:0811.2796, DOI 10.1088/0004-637X/693/1/822.
- **Refutability test:** Generate a synthetic spectrum with known T, n_e, composition and Poisson+readout noise, then fit with this likelihood and with a variant using observed in the variance (sigma_i^2 = max(observed_i,0)/gain + readout^2 + dark). The model-variance form will systematically recover lower total predicted intensity / biased bright-line ratios; the Neyman form will be closer to truth. The presence of the log(variance) term in the gradient is the discriminating mechanism.
- **Self-refutation attempt:** In CF-LIBS the readout (10 counts) + dark (1) floor dominates over the shot term for faint pixels, so variance is nearly constant and the bias is negligible; also using predicted is standard when observed counts are unavailable.  → succeeded: False
- **Blast radius:** Both single-zone (bayesian_model) and two-zone (two_zone_bayesian_model) NUTS likelihoods plus module-level log_likelihood used by dynesty; biases all posterior composition/intensity scaling toward lower predicted intensity on bright lines.
- **Est. macro_F1 lift:** 0.005-0.02 (bright-line-dominated retrievals); regime-dependent

#### 4. [MEDIUM/high/missing_feature] McWhirter LTE soft-penalty absent from single-zone model and gated on T_core only in two-zone (T_shell LTE never checked)

- **File:** `cflibs/inversion/solve/bayesian/forward.py:577-633 (single-zone), 660-672 (two-zone)`
- **Expected per lit:** The McWhirter criterion n_e >= 1.6e12 * T_K^{1/2} * (Delta E)^3 is a necessary LTE condition that must hold at the LOWEST-temperature, lowest-density plasma region (where it is hardest to satisfy). In ps-LIBS the criterion is borderline at n_e ~ 1e16 cm^-3, and the cooler shell zone is precisely where LTE is most likely to break. An LTE Saha-Boltzmann forward model is only valid where the criterion holds in every emitting zone (Cristoforetti et al. 2010, 'Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion').
- **Actual in code:** Single-zone bayesian_model (lines 577-633): contains NO call to mcwhirter_log_penalty and no numpyro.factor LTE term at all. Two-zone (lines 665-672): penalty = mcwhirter_log_penalty(T_core_eV, log_ne, ...); numpyro.factor('mcwhirter_lte', penalty) — evaluated only at T_core_eV. The cooler T_shell_eV (range default 0.3-2.0 eV, priors.py:154), which is the binding constraint for LTE validity, is never tested.
- **Citation:** Cristoforetti, G., De Giacomo, A., Dell'Aglio, M., Legnaioli, S., Tognoni, E., Palleschi, V., & Omenetto, N. (2010). Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion. Spectrochimica Acta Part B 65(1), 86-95. DOI 10.1016/j.sab.2009.11.005.
- **Refutability test:** Construct a two-zone draw with T_core=1.2 eV (satisfies McWhirter at n_e=1e16) but T_shell=0.4 eV. At T_shell the threshold n_e_min = 1.6e12*(0.4*EV_TO_K)^0.5*(3.0)^3 — compute and compare to 1e16; if T_shell violates while T_core passes, the model accepts a non-LTE shell with zero penalty. Single-zone: confirm grep shows no mcwhirter/factor term in bayesian_model.
- **Self-refutation attempt:** Maybe T_core>=T_shell (enforced at line 658) and lower T makes the threshold SMALLER (threshold ~ T^0.5), so if T_core passes, T_shell automatically passes — making the T_core-only check sufficient.  → succeeded: False
- **Blast radius:** Single-zone runs have no LTE gate whatsoever; two-zone shell LTE unvalidated. Affects validity of every LTE-based composition posterior in the lower-density ps-LIBS regime.
- **Est. macro_F1 lift:** 0.005-0.02 by rejecting non-LTE shell draws that corrupt composition

#### 5. [MEDIUM/high/implementation_bug] Candidate prefilter falsy-zero fallback: `getattr(...) or fallback` silently discards a valid estimated T or n_e of 0

- **File:** `cflibs/inversion/candidate_prefilter.py:113-114, 117`
- **Expected per lit:** Prefilter dimensionality reduction must probe the actual estimated plasma temperature so temperature-sensitive trace lines survive the NNLS gate (Hebert et al. 2020 frame the LIBS inversion as composition retrieval where line-height ratios are a nonlinear function of plasma state; probing the wrong T drops genuine trace contributors). The estimated T/n_e should be used whenever it exists (is not None), not only when it is truthy.
- **Actual in code:** base_T = getattr(identifier, '_estimated_T', None) or identifier.fallback_T_K; base_ne = getattr(identifier, '_estimated_ne', None) or identifier.fallback_ne_cm3. The `or` operator falls back whenever the left operand is falsy (0.0, 0, empty), not only when it is None. Subsequent T_offset = max(base_T + offset, 3000.0) then uses the fixed fallback instead of the estimate.
- **Citation:** Hebert, C.-A., Lawrence, E., Myers, K., Colgan, J. P., & Judge, E. J. (2020). An Initial Exploration of Bayesian Model Calibration for Estimating the Composition of Rocks and Soils on Mars. arXiv:2008.04982. DOI 10.48550/arXiv.2008.04982.
- **Refutability test:** Set identifier._estimated_T = 0.0 (a degenerate but possible NNLS fit) and call select_candidate_elements; base_T becomes identifier.fallback_T_K instead of 0.0, so the multi-T probe centers on the wrong temperature. Replacing `or` with an explicit `is None` check changes the resulting candidate list whenever the estimate is exactly falsy.
- **Self-refutation attempt:** An estimated temperature of exactly 0.0 K is unphysical and the identifier would never produce it, so the falsy-zero branch is unreachable in practice and the bug is moot.  → succeeded: False
- **Blast radius:** select_candidate_elements multi-T union step; affects which elements reach Bayesian MCMC. Low base rate (requires a falsy estimate) but silent when triggered.
- **Est. macro_F1 lift:** 0.005-0.015 in rare degenerate-estimate cases feeding the macro_F1 trace-recall path

### 14.x manifold (6 findings)

#### 1. [HIGH/high/regime_mismatch] Basis-library line shapes use a single wavelength-independent Gaussian (instrument-FWHM-in-nm only): no Doppler scaling, no Voigt — mismatched against the Voigt manifold generator

- **File:** `cflibs/manifold/basis_library.py:116, 179`
- **Expected per lit:** Voigt-class profiles with wavelength-dependent Gaussian width
- **Actual in code:** Line 116: `sigma = cfg.instrument_fwhm_nm / 2.3548200450309493  # FWHM -> sigma` — a single scalar. Line 179: `spectrum += eps * np.exp(-0.5 * ((wl_grid - trans.wavelength_nm) / sigma) ** 2)` — every line, at every wavelength from 200 to 900 nm (default wavelength_range=(200.0, 900.0), line 36), uses the same constant `sigma` in nm. There is no Doppler term, no Stark/Lorentzian wing, and no λ-dependence of the width. The manifold generator (generator.py) by contrast renders full Voigt (Humlicek W4) + Stark per the inventory, so the basis fingerprints are systematically narrower and the wrong shape relative to the spectra the manifold/data contain.
- **Citation:** Matsumura, T. et al. (2024). High-Throughput Calibration-Free Laser-Induced Breakdown Spectroscopy. ACS Earth and Space Chemistry 8(6):1259-1271. DOI 10.1021/acsearthspacechem.4c00007
- **Refutability test:** Generate a basis spectrum for Fe at T=8000 K with instrument_fwhm_nm=0.05 over 200-900 nm and measure the FWHM of an Fe line at 250 nm vs 800 nm: both will be exactly 0.05 nm. A physically-broadened spectrum (Doppler or fixed-R) has FWHM that grows with λ. Then fit a Voigt-broadened synthetic Fe spectrum (Stark width ~0.05 nm at n_e=1e17) with the Gaussian basis: the residual will show systematic under-fit in the line wings.
- **Self-refutation attempt:** Maybe the basis library is intentionally instrument-limited because at the chosen instrument_fwhm_nm the instrument width dominates Stark+Doppler, making the Gaussian a fine approximation; and maybe area-normalization (line 181-183) makes the absolute width irrelevant so only relative line positions matter for identification.  → succeeded: False
- **Suggested fix:** Render each basis line as a Voigt (or pseudo-Voigt) with (a) a Doppler/instrument Gaussian width that scales with wavelength (sigma_G(λ) = λ/(R·2.3548) for a fixed-R instrument, or add Doppler sqrt term) and (b) a Stark Lorentzian component consistent with the generator's Voigt path, so the basis fingerprints match the manifold/data line shapes.
- **Blast radius:** All BasisIndex/BasisLibrary classification and any NNLS/linear unmixing built on the basis. Any code path that fits a measured (Stark-broadened) spectrum against the Gaussian-only basis inherits the width/shape bias.
- **Est. macro_F1 lift:** 0.05-0.12

#### 2. [HIGH/high/algorithmic_wrong_equation] Generic (≠4-element) manifold composition grid sweeps only the first element and pins all others to zero — violates simplex closure, never explores the interior of composition space

- **File:** `cflibs/manifold/generator.py:966-987`
- **Expected per lit:** Dirichlet/ILR sampling on the simplex with Σ C_s = 1
- **Actual in code:** The generic branch (per inventory lines 980-987) sweeps only the FIRST element over `linspace(0.5, 1.0)` and sets every other concentration to 0.0 whenever len(elements) != 4 (only the hardcoded len==4 Ti-Al-V-Fe quartet builds a real simplex). The swept vectors therefore have C_1 in [0.5,1.0] and C_2..C_D = 0, so the sum equals C_1 (not 1) and the interior of the simplex is never populated.
- **Citation:** Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barceló-Vidal, C. (2003). Isometric Logratio Transformations for Compositional Data Analysis. Mathematical Geology 35(3):279-300. DOI 10.1023/A:1023818214614
- **Refutability test:** Call generate_manifold with elements=['Fe','Cu','Ni'] (3 elements) and inspect the params array: the Cu and Ni concentration columns will be all zeros and the Fe column will range 0.5-1.0, never summing to 1. A correct simplex grid (Dirichlet alpha=1) would have all three columns non-zero and each row summing to 1.
- **Self-refutation attempt:** Maybe the only production use of the generator is the Ti-Al-V-Fe (len==4) alloy path, so the degenerate generic branch is never executed; or maybe downstream renormalizes the composition vector so the missing closure is irrelevant.  → succeeded: False
- **Suggested fix:** Replace the first-element-only sweep with Dirichlet(alpha) sampling or an ILR/Helmert grid on the (D-1)-simplex for arbitrary D, enforcing Σ_s C_s = 1; keep the special-cased 4-element simplex only as one instance of the general path.
- **Blast radius:** Every manifold built for an element count other than 4 (the common case for an arbitrary --elements list). Such manifolds cannot serve as nearest-neighbor priors for composition inversion or as coarse init in coarse_to_fine.py (which reads coarse_params per element).
- **Est. macro_F1 lift:** 0.03-0.10 for any non-quartet element set

#### 3. [MEDIUM/high/implementation_bug] BasisLibrary._nearest_grid_idx normalizes T-distance by T_max^2 but n_e-distance by the log10 range — asymmetric metric biases nearest-grid selection toward density

- **File:** `cflibs/manifold/basis_library.py:257-262`
- **Expected per lit:** Each axis normalized by its own grid range
- **Actual in code:** Lines 259-261: `dists = (self._params[:,0] - T_K)**2 / self._T_vals[-1]**2 + (np.log10(self._params[:,1]) - np.log10(ne_cm3))**2 / np.log10(self._ne_vals[-1]/self._ne_vals[0])**2`. The T term is divided by T_max^2 (e.g. 12000^2) — the SQUARE of the maximum temperature, not the T range (T_max - T_min). The n_e term is correctly divided by the squared log10 RANGE. With default T=(4000,12000) the T denominator is 1.44e8 while a step of 160 K gives a T-residual^2 of ~2.6e4, so normalized T distance is ~1.8e-4; the same one-step move in n_e contributes ~ (0.13)^2/(2.7)^2 ≈ 2.3e-3, an order of magnitude larger. The metric therefore favors matching density over temperature.
- **Citation:** Labutin, T. A., Zaytsev, S. M., & Popov, A. M. (2013). Automatic Identification of Emission Lines in Laser-Induced Plasma by Correlation of Model and Experimental Spectra. Analytical Chemistry 85(4):1985-1990. DOI 10.1021/ac303270q
- **Refutability test:** Build a basis with T grid 4000-12000 K and n_e 1e15-5e17, then query _nearest_grid_idx for a point exactly between two T grid nodes but offset slightly in n_e. The asymmetric metric will jump to the wrong T node because the T-axis weight (1/T_max^2) is far too small relative to the n_e-axis weight. Replacing T_max^2 with (T_max-T_min)^2 changes the selected node.
- **Self-refutation attempt:** Maybe T_vals[-1]^2 is a deliberate scale-free normalization that happens to give comparable axis weights for the specific default grid, so in practice the bias is negligible; or maybe get_basis_matrix_interp (bilinear) is always used in production and _nearest_grid_idx is only a fallback.  → succeeded: False
- **Suggested fix:** Normalize the T term by the T RANGE: divide by `(self._T_vals[-1] - self._T_vals[0])**2` to match the log10-range normalization already used for n_e, giving a balanced two-axis metric.
- **Blast radius:** get_basis_matrix and get_element_spectrum (both call _nearest_grid_idx); any consumer that pulls a single-element basis at a target (T,ne) gets a T that can be off by one or more grid nodes.
- **Est. macro_F1 lift:** 0.01-0.03

#### 4. [MEDIUM/medium/algorithmic_wrong_equation] BasisIndex (T, n_e) estimate is an inverse-(squared-)distance weighted median over k=50 neighbors drawn from ALL elements with no per-element restriction and no collinear-endmember pruning

- **File:** `cflibs/manifold/basis_index.py:193-208`
- **Expected per lit:** Collinearity-pruned endmembers; element-consistent or de-correlated voting
- **Actual in code:** Lines 193-208: `distances, indices = self.vector_index.search(embedding, k=k)` (default k=50), then `neighbor_elements = [self._elements[self._element_indices[i]] for i in indices]`, `neighbor_T = self._params[neighbor_grid_idx, 0]`, weights `= 1.0/(distances + 1e-10)`, `T_est = _weighted_median(neighbor_T, weights)`. The 50 neighbors are gathered across the entire flattened (element × grid) space; their T and n_e are pooled and a single weighted median is taken with NO restriction to a consistent element and NO de-correlation of collinear element fingerprints. `distances` are FAISS squared-L2 values, so the weighting is 1/(squared distance), an undocumented choice.
- **Citation:** Schröder, S. et al. (2024). Spectral Unmixing for the Identification of Unusual Minor and Trace Element Content in ChemCam LIBS Data. International Conference on Mars (LPI Contrib. 3007), abstract 3311. https://hal.science/hal-04800148 ; Black, D. et al. (2024). arXiv:2401.17388 DOI 10.48550/arXiv.2401.17388
- **Refutability test:** Construct a basis with two spectrally-similar elements (e.g. Fe and Ni) and query with a pure-Fe spectrum at a known (T,ne). Count how many of the 50 voted neighbors are Ni vs Fe and at what (T,ne); the Ni neighbors at the wrong (T,ne) will pull the weighted-median T/ne away from truth. Restricting voting to the majority-element neighbors (or pruning collinear Fe/Ni endmembers) measurably reduces the (T,ne) error.
- **Self-refutation attempt:** T and n_e are global plasma properties shared by all elements, so pooling neighbors across elements is physically legitimate as long as each element's basis at the true (T,ne) lies among the 50 neighbors; the weighted median is robust to a minority of wrong-element outliers, so the estimate may be fine in practice.  → succeeded: False
- **Suggested fix:** Prune linearly-dependent basis vectors before indexing (Schröder), and restrict the (T,ne) weighted median to neighbors of the winning element (or weight by per-element vote share); document that FAISS distances are squared-L2 if 1/d weighting is retained.
- **Blast radius:** Every estimate_plasma_params call (the primary BasisIndex inference API). A biased coarse (T,ne) propagates into any downstream refinement (coarse_to_fine.py fine stage) and basis-matrix selection.
- **Est. macro_F1 lift:** 0.02-0.05

#### 5. [MEDIUM/medium/numerical_stability] VectorIndex IVF/PQ build reuses the same embeddings for train() and add() with no n_train ≥ ~39·n_lists guard and feeds un-whitened PCA scores to product quantization

- **File:** `cflibs/manifold/vector_index.py:298-322`
- **Expected per lit:** n_train ≥ ~39·n_lists; whitened/rotated sub-spaces before PQ
- **Actual in code:** Lines 301-307 (ivf_flat) and 312-322 (ivf_pq): `self.index.train(embeddings); self.index.add(embeddings)` — identical array passed to both, with `self.config.n_lists` defaulting to 100 (VectorIndexConfig line 200) and NO check that len(embeddings) ≥ ~39·n_lists. For ivf_pq the embeddings handed to FAISS are the SpectralEmbedder output: area-normalize → PCA → L2-normalize (vector_index.py transform lines 136-142), i.e. raw (un-whitened) PCA scores, with no per-sub-space whitening / random rotation before pq_m=8 quantization.
- **Citation:** Johnson, J., Douze, M., & Jégou, H. (2021). Billion-Scale Similarity Search with GPUs. IEEE Trans. Big Data 7(3):535-547. DOI 10.1109/TBDATA.2019.2921572 ; Jégou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. IEEE TPAMI 33(1):117-128. DOI 10.1109/TPAMI.2010.57
- **Refutability test:** Build a VectorIndex with index_type='ivf_flat', n_lists=100 on a small BasisLibrary (e.g. 5 elements × 50 T × 20 ne minus zeros ≈ a few thousand, but easily < 3900 after skip_zero): FAISS prints 'WARNING clustering N points to 100 centroids: please provide at least 3900 training points' and recall drops. Switching to ivf_pq on the un-whitened scores yields measurably lower recall@10 than the same data with PCA whitening + random rotation.
- **Self-refutation attempt:** The default index_type is 'flat' (exact, training-free, library-size-agnostic), so unless a caller explicitly selects ivf_flat/ivf_pq the guard is irrelevant; and small basis libraries are exactly the regime where flat is appropriate, so the IVF/PQ paths may never run in practice.  → succeeded: False
- **Suggested fix:** Guard: if index_type != 'flat' and n_vectors < 39·n_lists, either auto-reduce n_lists (≈4·sqrt(N)) or fall back to flat with a warning; for ivf_pq apply PCA whitening or a random rotation (faiss.ProductQuantizer with OPQ) before quantization.
- **Blast radius:** VectorIndex with index_type in {ivf_flat, ivf_pq}, used by ManifoldLoader.build_vector_index and any BasisIndex configured for approximate search. Under-trained quantizer or unbalanced-PQ degrades nearest-neighbor recall silently.
- **Est. macro_F1 lift:** <0.02 (only when IVF/PQ explicitly selected)

#### 6. [HIGH/high/regime_mismatch] Manifold time-integration uses a fixed µs-scale cooling-trail model (t0=1e-6 s, T∝(1+t/t0)^-0.5, n_e∝(1+t/t0)^-1, gate 5 µs) that does not represent a 1 ps single-pulse Yb:fiber plasma

- **File:** `cflibs/manifold/generator.py:857-864, 918-924`
- **Expected per lit:** Time/state integration matching the actual measurement gate and plasma lifetime
- **Actual in code:** Per inventory (verified): generator.py lines 857-859 hardcode `t0=1e-6 s`, `T_trail = T_max*(1+t/t0)**-0.5`, `ne_trail = ne_max*(1+t/t0)**-1.0`, with an emission cutoff `T > 0.4 eV` (line 864) and default gate_width_s=5e-6 s (5 µs), gate_delay_s=300 ns. These fixed exponents (-0.5 for T, -1.0 for n_e) and the 5 µs integration window are magic numbers tuned for µs-gated ICCD plasmas; the project target is a 1 ps @ 1040 nm Yb:fiber pulse (T~0.5-1.3 eV, n_e=1e16-1e18). The time-integrated manifold spectra are therefore weighted toward long-lived plasma conditions the ps experiment never reaches.
- **Citation:** Labutin, T. A., Zaytsev, S. M., & Popov, A. M. (2013). Automatic Identification of Emission Lines in Laser-Induced Plasma by Correlation of Model and Experimental Spectra. Analytical Chemistry 85(4):1985-1990. DOI 10.1021/ac303270q
- **Refutability test:** Generate a manifold snapshot with the default cooling-trail integration and a single-state (delta-function in time) snapshot at the same peak (T,ne); compare the integrated line-ratio of an ion line to a neutral line. The cooling-trail version will show enhanced neutral/low-T contribution from the long µs tail (since the trail spends most of its integration time at the cooler end), shifting the apparent (T,ne) the manifold encodes away from the ps-plasma values. Setting gate_width_s to a sub-ns value collapses the difference.
- **Self-refutation attempt:** Maybe the cooling-trail integration is optional and the production manifold for ps-LIBS is built from the single-snapshot path (_compute_spectrum_snapshot) with a short gate, so the µs trail and its exponents are never used; or maybe the T>0.4 eV cutoff effectively truncates the trail before the long tail dominates.  → succeeded: False
- **Suggested fix:** Expose t0, the cooling exponents, gate_delay_s and gate_width_s as configurable plasma-model parameters (or support a single-snapshot mode) and provide ps-LIBS-appropriate defaults; document the µs/ICCD assumption baked into the current power laws.
- **Blast radius:** _time_integrated_spectrum and _time_integrated_spectrum_ldm (the time-integrated manifold generation path) and any manifold built with the default gate/cooling settings; the resulting (T,ne)↔spectrum mapping is biased for ps-LIBS data.
- **Est. macro_F1 lift:** 0.04-0.10 if the default time-integrated path is used for ps-LIBS


---

# Wave-2 RE-BASELINE against origin/dev (2026-06-05)

The audit ran against commit `9fd7b69`. `origin/dev` then advanced 9 commits (#214–#224) with heavy overlap (Stark/partition correctness #215, U(T) provider #218, self-absorption wiring #219, BHVO-2 accuracy #223/#224). All 15 families were re-verified against **current `origin/dev`**. This section is the AUTHORITATIVE implementation status; it supersedes §3 and §10.

## Per-family status in origin/dev

| Family | Status |
|--------|--------|
| Stark broadening cascade (unit/density core already FIXED; remaining sub-items) | **PARTIAL** |
| Self-absorption / CDSB dead code + internal bugs (#219 wired SA) | **PARTIAL** |
| Iterative solver n_e diagnostic + silent failures | **PARTIAL** |
| Closure-equation correctness across solver paths | **OPEN** |
| ALIAS deviations from Noel 2025 | **OPEN** |
| Doppler factor-of-2 at 4 inline sites + basis-library no-Doppler | **PARTIAL** |
| Uncertainty incomplete variance budget | **OPEN** |
| ns-LIBS defaults transplanted into ps-LIBS hardware | **OPEN** |
| Boltzmann fitter 1/sigma^4 over-weighting | **OPEN** |
| Saha-Boltzmann IPD inconsistency + dual Debye-Huckel (#218 unified U(T) verifica | **OPEN** |
| LTE-gate wrong Delta-E + incomplete McWhirter coverage | **OPEN** |
| Manifold generic composition grid violates simplex closure | **OPEN** |
| Manifold basis-library/index internals | **OPEN** |
| Bayesian fabricated convergence + model-variance likelihood | **OPEN** |
| Two-zone Saha Stage-II truncation + prefilter falsy-zero | **OPEN** |

## Already FIXED in dev (dropped from plan)

- Stark (b) stark_alpha column conflation with Griem A — FIXED in dev (#215): AtomicSnapshot docstring and _per_line_stark_gamma consistently treat line_stark_alpha as the T-power-law exponent (factor_T = (T_eV/0.86173)**(-alpha)); never fed into a 1.75*A ion-broadening term.
- Stark core FWHM/HWHM + REF_NE convention — FIXED in dev (#215): stark.py REF_NE=1.0e17, _STARK_W_IS_FWHM=True; kernels._per_line_stark_gamma uses 0.5*stark_w*(n_e/_STARK_REF_NE)*factor_T with REF_T_EV=0.86173.
- Self-absorption (a) SelfAbsorptionCorrector now CALLED from iterative.py — FIXED (#219): imported, instantiated, and invoked in the solve loop (was dead code).
- Self-absorption (b) doublet-ratio uses g*A*lambda^3 — FIXED: _theoretical_doublet_ratio returns (g1*A1*l1^3)/(g2*A2*l2^3) (verified at self_absorption.py:313-329); buggy A/lambda form gone.
- Self-absorption (e) COG _estimate_optical_depth lambda vs lambda^2 — FIXED: dimensionally-correct sigma_0 with documented replacement of the old 1e-25*A_ki*lambda^3 SCALE_FACTOR.
- Iterative (b) positive Boltzmann slope -> 50000 K clamp + converged=True — FIXED (verified): no 50000 literal in the loop; replaced by boltzmann_degenerate gate that holds T_new=T_prev and blocks convergence (iterative.py:1629-1631); JAX path mirrors.
- Doppler profiles.py canonical doppler_width / doppler_sigma_jax — FIXED: both use sqrt(kT/m) with explicit 'Removed the spurious factor of 2' comment (verified profiles.py:231,789).
- ALIAS (c) hardcoded literal eps_th *= 0.3 — partially addressed: refactored to configurable+logged self_absorption_damping gated by E_i cutoff (literal gone). Default still 0.3 so deviation-from-paper remains (tracked under item 4c).

## Remaining OPEN/PARTIAL work — dependency-ordered

Status legend: OPEN = bug fully present in dev; PARTIAL = partly addressed. `grp` = file-disjoint group (same grp ⇒ shares files ⇒ cannot run in parallel).

| # | Effort | Status | grp | Title | Files | Deps |
|---|--------|--------|-----|-------|-------|------|
| 1 | S | OPEN | 1 | Boltzmann fitter 1/sigma^4 over-weighting (FI) | `cflibs/inversion/physics/boltzmann.py`, `cflibs/inversion/physics/boltzmann_jax.py` | — |
| 2 | L | OPEN | 2 | Uncertainty variance budget: 4 fixes (n_e ufloat, MC atomic-data default, Poisson floor, full softmax Jacobian) | `cflibs/inversion/physics/uncertainty.py`, `cflibs/inversion/solve/joint_optimizer.py` | — |
| 3 | XL | OPEN | 3 | Closure correctness: SpectralRefiner Sum C=1, closed-form closure_mode, scale-invariant dirichlet, true Hron PLR | `cflibs/inversion/solve/spectral_refiner.py`, `cflibs/inversion/solve/closed_form.py`, `cflibs/inversion/physics/closure.py` | — |
| 4 | M | OPEN | 4 | ALIAS Noel 2025 conformance: k_det N_X, remove P_cov blend, default damping, DOI | `cflibs/inversion/identify/alias.py` | — |
| 5 | XL | OPEN | 5 | Bayesian forward.py: fix apply_self_absorption hardcode + Doppler factor-of-2 + two-zone Stage-III + single/two-zone McWhirter | `cflibs/inversion/solve/bayesian/forward.py`, `cflibs/inversion/solve/bayesian/priors.py`, `cflibs/inversion/solve/bayesian/atomic.py` | — |
| 6 | S | OPEN | 6 | Doppler factor-of-2 in manifold generator (3 inline sites) + coarse_to_fine | `cflibs/manifold/generator.py`, `cflibs/inversion/solve/coarse_to_fine.py` | — |
| 7 | M | OPEN | 6 | Manifold composition grid simplex closure (generic D-dim) | `cflibs/manifold/generator.py` | Doppler factor-of-2 in manifold generator (3 inline sites) + coarse_to_fine |
| 8 | XL | OPEN | 7 | Bayesian basis-library/index Doppler+Voigt, nearest-grid metric, per-element vote, IVF/PQ guards+whitening | `cflibs/manifold/basis_library.py`, `cflibs/manifold/basis_index.py`, `cflibs/manifold/vector_index.py` | — |
| 9 | M | PARTIAL | 8 | Stark: SpectrumModelJax override route through Voigt (honor apply_stark) | `cflibs/radiation/spectrum_model.py` | — |
| 10 | XL | OPEN | 9 | Stark Griem ion-broadening + Stark shift wired into production Voigt kernel | `cflibs/radiation/kernels.py`, `cflibs/core/jax_runtime.py`, `cflibs/atomic/database.py` | — |
| 11 | S | OPEN | 7 | basis_library physical broadening (Doppler+Stark Voigt) [overlaps item 8a] | `cflibs/manifold/basis_library.py` | Bayesian basis-library/index Doppler+Voigt, nearest-grid metric, per-element vote, IVF/PQ guards+whitening |
| 12 | XL | PARTIAL | 10 | Iterative solver n_e diagnostic: upper-T clamp, undamped ne convergence, Stark-based n_e, corona ratio config | `cflibs/inversion/solve/iterative.py`, `cflibs/inversion/solve/closed_form.py`, `cflibs/core/constants.py` | Boltzmann fitter 1/sigma^4 over-weighting (FI) |
| 13 | L | OPEN | 11 | ns-LIBS vs ps-LIBS regime defaults (matrix factors, STP pressure, cooling-trail laws, tau boost) | `cflibs/inversion/physics/matrix_effects.py`, `cflibs/manifold/config.py`, `cflibs/inversion/physics/cdsb.py` | — |
| 14 | M | PARTIAL | 11 | CDSB dead-code: remove CDSBPlotter or fix U-interpolation + empirical tau seeds | `cflibs/inversion/physics/cdsb.py`, `cflibs/inversion/__init__.py` | — |
| 15 | M | OPEN | 12 | LTEValidator McWhirter Delta-E uses term-scheme/resonance gap not adjacent observed-E_k gap | `cflibs/plasma/lte_validator.py` | — |
| 16 | L | OPEN | 13 | Saha-Boltzmann IPD: thread eff_ip/n_e into provider.at + unify Debye-Huckel formula | `cflibs/plasma/saha_boltzmann.py`, `cflibs/plasma/partition.py`, `cflibs/inversion/solve/iterative.py` | — |
| 17 | L | OPEN | 14 | Bayesian convergence + likelihood: real ESS/r_hat (multi-chain) + observed-count variance | `cflibs/inversion/solve/bayesian/samplers.py` | — |
| 18 | S | OPEN | 15 | candidate_prefilter falsy-zero fallback for estimated T/n_e | `cflibs/inversion/candidate_prefilter.py` | — |

### Remaining-work detail + validation gates

**#1 Boltzmann fitter 1/sigma^4 over-weighting (FI)** (S, OPEN, grp1)  
- Files: `cflibs/inversion/physics/boltzmann.py`, `cflibs/inversion/physics/boltzmann_jax.py`
- Remaining: Change _compute_weights to return 1.0/safe_err (1/sigma) instead of 1.0/safe_err**2, so np.polyfit(w=) internal squaring yields the correct WLS weight 1/sigma^2. This single change fixes all CPU polyfit call sites (_fit_sigma_clip, _weighted_fit, robust_huber/ransac). For the JAX path (_fit_sigma_clip_jax) the existing kernel_weights = weights*weights then correctly yields 1/sigma^2; update the explanatory comment that currently documents the 1/sigma^4 as intended.
- Gate: Synthetic Boltzmann plot with known T: WLS slope error vs analytic 1/sigma^2 WLS < 1e-9; effective minimization is sum(r^2/sigma^2) not sum(r^2/sigma^4). pytest tests covering BoltzmannFitter weighted fit pass; CPU and JAX paths agree.

**#2 Uncertainty variance budget: 4 fixes (n_e ufloat, MC atomic-data default, Poisson floor, full softmax Jacobian)** (L, OPEN, grp2)  
- Files: `cflibs/inversion/physics/uncertainty.py`, `cflibs/inversion/solve/joint_optimizer.py`
- Remaining: (a) Make saha_factor_with_uncertainty accept n_e as UFloat so its variance propagates, and wire it into the solver uncertainty path (currently orphaned). (b) Default run_monte_carlo_uq to PerturbationType.COMBINED (or expose perturbation_type/atomic_uncertainty) so 10% A_ki uncertainty enters the budget. (c) Replace additive-Gaussian-then-max(.,1.0) intensity perturbation with Poisson (or log-normal) and remove the absolute 1.0 floor. (d) In joint_optimizer build full softmax Jacobian J (diag C_i(1-C_i), off-diag -C_i*C_j) and compute Cov_C = J @ cov_theta @ J.T using off-diagonal cov entries, not just std_errors[2+i].
- Gate: MC variance for low-count lines no longer biased upward (mean of sampled intensities matches Poisson mean); COMBINED budget strictly >= spectral-only; concentration covariance is symmetric and matches finite-difference propagation of softmax through cov.

**#3 Closure correctness: SpectralRefiner Sum C=1, closed-form closure_mode, scale-invariant dirichlet, true Hron PLR** (XL, OPEN, grp3)  
- Files: `cflibs/inversion/solve/spectral_refiner.py`, `cflibs/inversion/solve/closed_form.py`, `cflibs/inversion/physics/closure.py`
- Remaining: (a) spectral_refiner: softmax-parameterize or post-normalize concentrations to enforce |Sum C - 1| < 1e-6; optionally add a separate scale parameter so amplitude is not conflated with composition. (b) closed_form: add closure_mode (matrix, oxide) to ClosedFormConfig and a matrix_element/oxide_stoichiometry path (reuse #224 oxide helper); warn when default STP_PRESSURE is used. (c) closure.apply_dirichlet_residual: normalize raw_sum (divide by experimental factor F) before the deficit/closure_diagnostic comparison so it is scale-invariant. (d) closure.plr_transform: implement true isometric Hron/Filzmoser pivot coords (sqrt((D-i)/(D-i+1)) scaling + geometric-mean balance) or rename to alr_transform and make optimize_pwlr_coordinates use the isometric metric.
- Gate: (a) RefinementResult concentrations sum to 1 +/-1e-6; (c) closure_diagnostic unchanged when all intensities scaled by 1e3; (d) PLR of [0.7,0.2,0.1] pivot=0 yields z_1 ~= 1.305 (sqrt(2/3)*ln(0.7/sqrt(0.02))); (b) closed-form matrix/oxide mode reproduces iterative-path closure on a shared fixture.

**#4 ALIAS Noel 2025 conformance: k_det N_X, remove P_cov blend, default damping, DOI** (M, OPEN, grp4)  
- Files: `cflibs/inversion/identify/alias.py`
- Remaining: (a) Set N_X to the element theoretical/expected line count (N_expected already available in _decide), not N_matched. (b) Remove the non-paper k_det = sqrt(k_det_raw * max(P_cov,0.01)) geometric-mean blend (or justify deviation); use k_det_raw per paper. (c) Change self_absorption_damping default from 0.3 to 1.0 (off) or implement a physically motivated per-line correction (the literal->configurable refactor already landed). (d) Add the Spectrochimica Acta Part B DOI 10.1016/j.sab.2025.107255 to the module docstring.
- Gate: Identification recall/precision on the synthetic ID corpus does not regress; k_det matches paper formula on a hand-computed single/multi-line element fixture; module docstring contains the journal DOI.

**#5 Bayesian forward.py: fix apply_self_absorption hardcode + Doppler factor-of-2 + two-zone Stage-III + single/two-zone McWhirter** (XL, OPEN, grp5)  
- Files: `cflibs/inversion/solve/bayesian/forward.py`, `cflibs/inversion/solve/bayesian/priors.py`, `cflibs/inversion/solve/bayesian/atomic.py`
- Remaining: (SA-h) Plumb an apply_self_absorption flag (+ nonzero path_length_m) through BayesianForwardModel instead of the hardcoded False at forward.py:349. (Doppler) Drop the _as_jax_real(2.0) factor at forward.py:484-486 (use sqrt(kT/m); or call profiles.doppler_sigma_jax). (TwoZone-a) Add U2 from partition_coeffs[:,2], read ionization_potentials[:,1], compute 3-stage Saha so ion_stage==2 lines use the doubly-ionized fraction/U2 in _compute_zone_spectrum. (LTE-b1) Add optional mcwhirter_log_penalty factor + mcwhirter_penalty_scale/max_delta_E_eV fields to single-zone PriorConfig + bayesian_model. (LTE-b2) Evaluate mcwhirter penalty for both T_core_eV and T_shell_eV in two_zone_bayesian_model.
- Gate: apply_self_absorption=True path changes thick-line predictions on a known-thick fixture; Doppler core width matches profiles.doppler_sigma_jax; a doubly-ionized line gets nonzero frac at high T (3-stage Saha sums to 1 across 3 stages); single-zone posterior shifts when mcwhirter_penalty_scale>0; both-zone penalty >= core-only penalty.

**#6 Doppler factor-of-2 in manifold generator (3 inline sites) + coarse_to_fine** (S, OPEN, grp6)  
- Files: `cflibs/manifold/generator.py`, `cflibs/inversion/solve/coarse_to_fine.py`
- Remaining: Remove the 2.0 multiplier in sqrt(2*T_eV*EV_TO_J/(m*c^2)) at generator.py:442-444 (_doppler_sigma_nm kernel-bracket), generator.py:671 (Voigt path), generator.py:853 (ldm_broaden path), and coarse_to_fine.py:499-500 (synthetic path). Use sqrt(kT/m). Correct the misleading 'sqrt(2kT/m)' comments at generator.py:437/668.
- Gate: Per-line sigma matches profiles.doppler_width (sqrt(kT/m)) to 1e-12; kernel-bracket sigma_dop_min/max bracket the corrected per-line sigma; manifold spectra core widths shrink by sqrt(2) vs prior.

**#7 Manifold composition grid simplex closure (generic D-dim)** (M, OPEN, grp6)  
- Files: `cflibs/manifold/generator.py`
- Remaining: Replace the generic (element count != 4) branch at generator.py:1012-1021 (sweeps only first element over linspace(0.5,1.0), pins rest to 0.0) with proper D-dimensional simplex sampling (Dirichlet draws or normalized regular simplex lattice) so every row satisfies Sum C_s = 1 and minor/trace elements are sampled.
- Gate: Every generated composition row sums to 1.0 +/-1e-9 for arbitrary element count; no element is identically zero across the whole grid; D=4 path still produces the Ti-Al-V-Fe coverage.
- Deps: Doppler factor-of-2 in manifold generator (3 inline sites) + coarse_to_fine

**#8 Bayesian basis-library/index Doppler+Voigt, nearest-grid metric, per-element vote, IVF/PQ guards+whitening** (XL, OPEN, grp7)  
- Files: `cflibs/manifold/basis_library.py`, `cflibs/manifold/basis_index.py`, `cflibs/manifold/vector_index.py`, `cflibs/inversion/common/pca.py`
- Remaining: (a) basis_library: replace constant-sigma Gaussian (lines 116/179) with per-line Voigt = Doppler sigma_D ~ lambda*sqrt(T/M) (quadrature with instrument FWHM) + n_e-dependent Stark Lorentzian. (b) _nearest_grid_idx (~274): normalize both axes consistently (T span not T_max^2; both dimensionless). (c) basis_index.estimate_plasma_params (~138-205): restrict the (T,ne) weighted-median vote to the dominant/queried element's neighbors. (d) vector_index.build (~268-300): add n_train >= ~39*n_lists (and >=2^pq_bits/subquantizer) guard with flat fallback; add whiten option to PCAPipeline.transform (divide scores by singular_values/sqrt(explained_variance)) before PQ.
- Gate: basis line width varies with lambda,T,n_e (not flat); nearest-grid distance dimensionless and symmetric across axes; (T,ne) estimate uses only same-element neighbors on a 2-element fixture; build() raises/falls-back below training-size threshold; whitened PCA scores have unit per-component variance.

**#9 Stark: SpectrumModelJax override route through Voigt (honor apply_stark)** (M, PARTIAL, grp8)  
- Files: `cflibs/radiation/spectrum_model.py`
- Remaining: Either delete the SpectrumModelJax.compute_spectrum override (lines 475-615) so it inherits the forward_model-based base (which already JAX-dispatches and honors apply_stark at line 327), or have the override compute _per_line_stark_gamma and use a Voigt sum when self.apply_stark is True instead of the pure-Gaussian _broaden_per_line_jax.
- Gate: SpectrumModelJax(apply_stark=True) produces Lorentzian-winged lines matching base SpectrumModel within tolerance; apply_stark=False reproduces Gaussian; base-vs-Jax spectra agree on a fixture.

**#10 Stark Griem ion-broadening + Stark shift wired into production Voigt kernel** (XL, OPEN, grp9)  
- Files: `cflibs/radiation/kernels.py`, `cflibs/core/jax_runtime.py`, `cflibs/atomic/database.py`, `cflibs/radiation/stark.py`
- Remaining: (a) Add line_stark_A and line_stark_d fields to AtomicSnapshot (jax_runtime.py:471-473) and ingest them in database.snapshot (~912-960). Apply the (1 + 1.75*A_ion*(1-0.75*R_D)) factor inside _per_line_stark_gamma (kernels.py:404-412). (c) Replace hardcoded R_D=0.5 in stark.py stark_hwhm (~134-138) and stark_hwhm_jax (~318-320) with R_D derived from Debye length / N_D (or delete the unused helper). (e) Carry line_stark_d into the snapshot and offset each line center by stark_shift(n_e,d_ref) in the Voigt kernel before broadening.
- Gate: Ion-broadened Stark gamma matches Griem reference for a catalogued line within tolerance; R_D varies with n_e,T; line centers shift by the catalogued d at the test n_e; forward spectrum with shift differs from unshifted at the predicted offset.

**#11 basis_library physical broadening (Doppler+Stark Voigt) [overlaps item 8a]** (S, OPEN, grp7)  
- Files: `cflibs/manifold/basis_library.py`
- Remaining: Same as item 8(a) for the Stark family Doppler-factor finding: add per-line thermal Doppler sigma (sqrt(kT/m), lambda-scaled) in quadrature with instrument sigma and an n_e-dependent Stark Lorentzian, or route basis generation through cflibs.radiation.kernels.forward_model. NOTE: this is the SAME file/edit as item 8 group 7 — implement once under item 8.
- Gate: Per-line width varies with lambda,T,n_e; covered by item 8 validation.
- Deps: Bayesian basis-library/index Doppler+Voigt, nearest-grid metric, per-element vote, IVF/PQ guards+whitening

**#12 Iterative solver n_e diagnostic: upper-T clamp, undamped ne convergence, Stark-based n_e, corona ratio config** (XL, PARTIAL, grp10)  
- Files: `cflibs/inversion/solve/iterative.py`, `cflibs/inversion/solve/closed_form.py`, `cflibs/core/constants.py`
- Remaining: (c) Add an upper physical bound on T_new (clip to ~30000-50000 K and/or gate on min |slope|) so a small-magnitude negative slope cannot run T to ~1e12 (both _solve_python ~1619-1627 and JAX path). (d) Compare UNDAMPED ne_new vs ne_prev for the convergence test (or track oscillation) instead of the 50/50 damped n_e (~1729-1758, JAX twin). (a) Replace isobaric STP pressure-balance n_e with a Stark-width-based diagnostic (or regime-appropriate pressure) in _solve_python (~1718-1727), JAX (~671-674), closed_form.py:440. (e) Make T_corona/T ratio (0.8) configurable/data-fit and actually apply corona weighting (T_for_saha currently = T_K). (f) Add explicit resonance/low-E_lower down-weighting in the Boltzmann fit (beyond the generic weight cap). NOTE shares STP_PRESSURE concern with ns-LIBS family (b).
- Gate: Noise-induced tiny negative slope cannot drive T above the ceiling; a 2-cycle ne oscillation no longer registers converged; n_e for a LIBS fixture lands in ~1e16-1e17 not ~1e18; corona weighting changes Saha mapping when two_region=True.
- Deps: Boltzmann fitter 1/sigma^4 over-weighting (FI)

**#13 ns-LIBS vs ps-LIBS regime defaults (matrix factors, STP pressure, cooling-trail laws, tau boost)** (L, OPEN, grp11)  
- Files: `cflibs/inversion/physics/matrix_effects.py`, `cflibs/manifold/config.py`, `cflibs/inversion/physics/cdsb.py`
- Remaining: (a) Gate/parameterize matrix_effects _DEFAULT_FACTORS (176-218) per acquisition regime (ns vs ps). (c) Make manifold cooling-trail t0=1e-6, T~t^-0.5, ne~t^-1 (generator.py:890-892/950-952) and gate_delay=300ns/gate_width=5us (config.py:95-96) configurable per regime. (d) Revisit cdsb resonance_tau_boost default 1.5 for ps-LIBS. (b) STP_PRESSURE regime default is shared with iterative-solver item 12(a)/constants.py:88 — coordinate there. NOTE: cdsb.py edits here (regime default) collide with CDSB dead-code cleanup; sequence after item 14.
- Gate: Each default is reachable via config per regime; ps-regime config yields documented ps-appropriate n_e/cooling without changing ns defaults; existing ns fixtures unchanged.

**#14 CDSB dead-code: remove CDSBPlotter or fix U-interpolation + empirical tau seeds** (M, PARTIAL, grp11)  
- Files: `cflibs/inversion/physics/cdsb.py`, `cflibs/inversion/__init__.py`
- Remaining: CDSBPlotter is dead (only re-exported, zero solve-path callers). Preferred: safe-delete CDSBPlotter and its re-exports (resolves d U_old==U_new no-op at 578-587, g resonance_tau_boost/initial_tau_base heuristics, and the c n_e fallback warning all at once). If kept: interpolate U at old vs new T in _update_tau_estimates and replace empirical initial_tau_base/resonance_tau_boost with the first-principles tau seed SelfAbsorptionCorrector already uses. NOTE: collides with ns-LIBS item 13(d) on resonance_tau_boost — do this first if deleting.
- Gate: If deleted: git grep CDSBPlotter returns only removal; import surface (inversion/__init__) still imports cleanly; targeted import test passes. If fixed: U_ratio != 1 across iterations on a T-changing fixture.

**#15 LTEValidator McWhirter Delta-E uses term-scheme/resonance gap not adjacent observed-E_k gap** (M, OPEN, grp12)  
- Files: `cflibs/plasma/lte_validator.py`
- Remaining: Replace the adjacent/consecutive-gap delta_E (lte_validator.py:244-253: max of consecutive differences of sorted unique E_k, floored at 0.1 eV) with the largest physically relevant energy gap (resonance transition / max term-scheme separation ~3-5 eV) or an explicit delta_E argument; remove/raise the 0.1 eV floor that admits sub-resonance gaps.
- Gate: n_e threshold for a known plasma matches the ~3-5 eV McWhirter requirement (orders of magnitude higher than the ~0.2 eV adjacent-gap result); LTE pass/fail flips correctly on a borderline fixture.

**#16 Saha-Boltzmann IPD: thread eff_ip/n_e into provider.at + unify Debye-Huckel formula** (L, OPEN, grp13)  
- Files: `cflibs/plasma/saha_boltzmann.py`, `cflibs/plasma/partition.py`, `cflibs/inversion/solve/iterative.py`
- Remaining: (a) Add a max_energy_ev/n_e parameter to DirectSumPartitionFunctionProvider.at (partition.py:1239-1265) and pass eff_ip from calculate_partition_function so direct_sum truncates at e_max = ip - delta_chi(n_e,T), matching the IPD-lowered Saha exponent (currently the cap is discarded on the provider path). (b) Unify all IPD evaluations on one shared Debye-Huckel function: replace partition.py:62-86 hardcoded 3.0e-8*Z*sqrt(n_e/T) and the iterative.py JAX 1.44e-7/lambda_D variant with calls to the canonical Gaussian-CGS form in saha_boltzmann.py; ensure direct-sum cutoff and Saha exponent consume the same function with the same n_e,T.
- Gate: At n_e=1e17,T=1e4 K all IPD call sites return one value (current 0.0949 vs 0.0660 eV 1.44x gap eliminated); partition-function truncation cap equals the Saha-exponent eff_ip; CPU ionization balance unchanged except for the now-consistent IPD term on a fixture.

**#17 Bayesian convergence + likelihood: real ESS/r_hat (multi-chain) + observed-count variance** (L, OPEN, grp14)  
- Files: `cflibs/inversion/solve/bayesian/samplers.py`
- Remaining: (a) Default MCMCSampler.run to >=2 chains (or refuse CONVERGED for a single chain); compute autocorrelation-corrected ESS (arviz.ess / FFT autocorr) instead of ess=len(s); never hardcode r_hat=1.0 — return UNKNOWN when between-chain variance is undefined so the <1.01 gate is not trivially passed. (b) For the dynesty NestedSampler._log_likelihood (and to coordinate with the NumPyro graphs in item 5) base variance on OBSERVED counts (or a fixed/estimated noise floor) instead of the model prediction, so the log-det term is constant w.r.t. parameters (or switch to a Poisson likelihood). NOTE: the NumPyro graph likelihoods live in forward.py (item 5) — keep variance convention consistent across both.
- Gate: A deliberately non-mixing single chain does NOT report CONVERGED; ESS < N for autocorrelated samples; nested-sampling log-evidence no longer rewards inflated predicted intensity (likelihood monotonic in fit quality on a fixture).

**#18 candidate_prefilter falsy-zero fallback for estimated T/n_e** (S, OPEN, grp15)  
- Files: `cflibs/inversion/candidate_prefilter.py`
- Remaining: Replace the `getattr(identifier, '_estimated_T', None) or fallback` / `... _estimated_ne ... or fallback` pattern (lines 113-114) with explicit `is not None` checks so a valid numeric-zero estimate is not silently discarded; optionally gate on `> 0` separately if a physical floor is desired.
- Gate: When _estimated_T=0.0 (or _estimated_ne=0.0) is set, the prefilter uses 0.0 (or the explicit floor), not the fallback default; None still routes to fallback.

## Status of work

- **#1 Boltzmann WLS weighting (Family I)** — DONE: PR #225 (branch `fix/wave2-boltzmann-weighting` → `dev`). Implemented as: keep `_compute_weights` at 1/σ² (correct for R²/kernel/cov), pass `sqrt(weights)`=1/σ to every polyfit site, feed the JAX kernel 1/σ² directly. 54 Boltzmann + JAX-parity tests pass; new independent regression test `test_boltzmann_weighting_convention.py`.


---

# Benchmark validation results (2026-06-05)

Empirical measurement of implemented fixes on the synthetic ID corpus (96 spectra, deep-UV 224.6–265.3 nm, 11 elements; SNR∈{20,30,40}, RP∈{700,1000}, shift/warp grid). Controlled before/after: pristine `origin/dev` vs the fix branch, **same corpus**.

## Family 5 (ALIAS Noel-2025) — PR #229 — ⛔ EMPIRICALLY REFUTED, NOT MERGED

The paper-faithful ALIAS change **regresses** ALIAS f1 by **−0.041** (0.2574 → 0.2164). Comb/Correlation bit-identical between runs (clean control). 2×2 ablation (additive, interaction +0.003):

| | damping ON | damping OFF |
|---|---|---|
| old k_det (dev) | 0.2574 | 0.2426 |
| new k_det (paper) | 0.2281 | 0.2164 (#229) |

- Paper-faithful `k_det` (N_X=N_expected, drop P_cov blend): **−0.028** (larger regressor)
- `self_absorption_aware` default off: **−0.013**

**Conclusion:** the audit's Family-5 hypothesis (paper-faithful ALIAS → higher F1) is **false on this corpus**. The repo's `N_matched` weighting + `P_cov` blend + damping-on are beneficial adaptations, not bugs. PR #229 converted to **draft** (not merged); re-evaluate only if real ps-LIBS data later supports the paper formula. **Lesson: every "paper-faithful" algorithmic change must be benchmark-gated before merge — theoretical correctness ≠ empirical improvement in this regime.**

## Other Wave-2 PRs (status)

- **#225** Boltzmann WLS weighting (Family I) — unit + 110 downstream tests green; weighting is unambiguously correct (numpy polyfit convention). No ID-benchmark needed (not an identifier change).
- **#226** prefilter falsy-zero (Family P), **#227** LTE McWhirter ΔE (Family K), **#228** SpectrumModelJax Stark (Family 1) — correctness fixes, verified by diff + targeted tests; not identifier-scoring changes, so not ALIAS-benchmark-gated.


---

# CORRECTION: Audit Family 5 was factually wrong (primary-source grounding, 2026-06-07)

After NotebookLM/Asta were restored, the CF-LIBS NotebookLM corpus (75 sources) was queried against the actual **Noel et al. 2025** paper. Verbatim §3.8:

> `k_det^X = k_rate^X × (1/N_X · k_shift^X + (N_X−1)/N_X · k_sim^X)`, where **N_X is the total number of _detected_ lines for the element X.**

**This refutes the audit's Family-5 premise.** The audit claimed `N_X` should be the number of theoretical/expected lines and that the original code's `N_X = N_matched` was a bug. The primary source confirms **N_X = detected (matched) lines** — i.e. the *original code was already paper-faithful*, and PR #229's "fix" (N_X = N_expected) was paper-*un*faithful. This is exactly why #229 regressed ALIAS f1 by −0.041 on the benchmark. **PR #229 is CLOSED** (not merged); the original `N_matched` behavior is correct and retained.

The paper does NOT include a `P_cov` coverage blend or self-absorption damping in `k_det` (audit correct that these are non-canonical), but both are **empirically beneficial** on the synthetic corpus and remain in the code. The paper's penalties are `CL = k_det · P_maj · P_SNR · P_ab`.

**Methodological lesson:** the audit (built largely from WebSearch fallback + a degraded literature channel) contained at least one *hallucinated physics claim* that primary-source grounding catches. The empirical benchmark gate independently caught the same regression. **Both safeguards fired — and BOTH are needed.** The remaining unimplemented physics families (#5 Bayesian forward, #10 Stark Griem/shift, #12 iterative n_e) should be **NotebookLM/Asta-grounded against primary sources BEFORE implementation**, and the already-merged physics findings (esp. Family J Saha-IPD, Family 4 closure, Family K LTE) spot-checked, since the audit's grounding is now known to be fallible.


---

# Primary-source grounding pass (NotebookLM, 2026-06-07)

NotebookLM restored; the CF-LIBS notebook (75 full-text sources) was queried per family. Asta remained down (gateway 401 server-side despite fresh token). Verdicts:

## Unimplemented families (gate before coding)

**#10 Stark (Griem ion-broadening + shift) — REVISED, mostly deprioritized.**
- Audit's ion-broadening form `[1+1.75A(1−0.75R)]` was **INCOMPLETE**: per John et al. 2023 (verbatim) the A term carries an extra `(N_e/10¹⁶)^¼` factor → full form `2W(N_e/1e16)·[1 + 1.75A(N_e/1e16)^¼(1−0.75·N_D^(−1/3))]`.
- **Both John 2023 AND Stetzler 2020 DROP the ion term** ("minimal effect" → `Δλ½ ≈ 2w·N_e/1e16`); it's <2–5% at n_e=1e16–1e17, only >10% above 1e18. So Griem ion-broadening is **low-value in the ps-LIBS regime** — the audit's XL effort is mostly wasted.
- `R = N_D^(−1/3) ∝ n_e^(1/6)/T^(1/2)` (audit right that hardcoded 0.5 is wrong, but moot if ion term dropped). Coefficient is emitter-charge-dependent (0.75 ions / 1.2 neutrals).
- **Worthwhile part: Stark SHIFT** — linear in n_e, applied additively to line center; corpus flags unhandled Stark shift as an identification-error source (Noel 2025 limitation). → Implement shift; skip ion-broadening.

**#12 iterative n_e — CONFIRMED (with one correction).**
- Stark broadening is the canonical n_e diagnostic (Hα 656.28 / Fe I 538.34 / Si II) — Tognoni 2010, Aragón & Aguilera 2010 (Spectrochim. Acta B 65, 395). Saha is secondary. → implement Stark n_e.
- Isobaric 1-atm pressure balance is "fundamentally non-standard and physically invalid" (LIBS = hypersonic shock, ~10¹¹ Pa initial). → demote/remove.
- **T_outer ≈ 0.8·T_core has NO Hermann citation in the corpus** — it's a common empirical DOF-reduction choice, NOT a citable "Hermann 2017" value. → document as empirical assumption; drop the false attribution.

**#5 Bayesian forward — CONFIRMED (with one improvement).**
- Doppler √2 bug confirmed (the codebase docs literally contain `σ=λ₀√(2kT/mc²)`); correct `σ=(λ₀/c)√(kT/m)`, FWHM `λ₀√(8ln2·kT/mc²)`. √2≈1.41 inflation.
- Saha doubly-ionized inclusion confirmed (John 2023 `n_e=n_I+2n_II`; Ca III >1% at 1.3 eV; Labutin 2013 to 4-fold, drop <1%). Don't truncate at singly-ionized.
- Likelihood: model-variance (Pearson) → "fitter's bias" (under-estimates peaks). **Best target is the Cash/Poisson exact statistic** + constant Gaussian readout — better than the audit's "Neyman" suggestion.

## Merged-fix spot-checks — ALL VALIDATED

- **Family K (#227 LTE):** McWhirter `n_e≥1.6e12·T^½·ΔE³`, ΔE=largest term-scheme/resonance gap (3–5 eV). Our `max(E_k)` is a conservative proxy. ✓ (Griem criterion `n_e≥1e18·T^−½·Z⁷` is stricter for low-Z — possible future add.)
- **Family J (#232 Saha-IPD):** Debye-Hückel `Δχ=Z·e²/λ_D`=**0.066 eV** at 1e17/1e4 K is the correct low-density limit; the 0.095 form is the Ion-Sphere (dense-plasma) model, wrong for LIBS. IPD must hit both Saha exponent AND partition cutoff. ✓ (caveat: only the direct-sum partition path can truncate; Irwin-polynomial fallback can't — consistent with #218/#232.)
- **Family 4 (#233 closure):** ILR (Egozcue 2003) is mandated; ALR is non-isometric. Hron 2012 pivot coordinates = rotated ILR, inherit isometric distance preservation. ✓

## Net audit-reliability tally (from grounding + benchmarks)
- **3 audit claims wrong/overstated, caught by grounding:** Family 5 (N_X = matched, not expected → #229 CLOSED), Family 10 (ion-broadening formula incomplete AND negligible at LIBS density), Family 12 (false "Hermann 2017" attribution for 0.8 ratio).
- **1 refinement:** #5 likelihood → Cash/Poisson, not Neyman.
- **3 merged physics fixes independently validated** (Families J/K/4).
**Conclusion: the audit is a strong starting hypothesis set but demonstrably fallible (~25% of physics-substantive claims needed correction). Every remaining physics family MUST be primary-source-grounded before implementation. Benchmark gating + NotebookLM grounding are complementary and both required.**
