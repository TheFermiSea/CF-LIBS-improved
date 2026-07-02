# Physics-First-Principles Audit

**Scope:** A first-principles physics audit of the shipped CF-LIBS pipeline, synthesizing seven domain investigations (plasma-state / temporal-spatial validity, radiative transfer / self-absorption, ionization & electron density, atomic data, lineshape & instrument, closure & matrix effects, forward-model completeness). Each domain investigation was adversarially verified against the actual code, the LIBS literature, and the `cflibs-formal` Lean spec. This document deduplicates issues that share a root cause, ranks them by expected real-data accuracy impact for the two program goals, and gives each surviving issue a concrete, design-level physics change (not a knob), a falsifiable first experiment, and an effort/dependency ordering.

**Two goals, ranked separately:**
- **(a) DED Ti-6Al-4V drift tracking** — a *constrained, known* element set {Ti, Al, V}; precision and *ratios* matter far more than absolute wt%; nominal feedstock is a prior; oxides/geology are out of scope.
- **(b) Absolute composition generally** — steel minors, geology, unknown matrices.

---

## Methods

**Triangulation.** Every claim was required to survive three independent lenses before being treated as actionable:
1. **Code** — exact file:line cites into the shipped path (not a docstring or an unwired module). Distinguishing the *default* path from opt-in/experimental capability was decisive: several "missing physics" claims collapsed once the exported-but-off capability was found (two-zone Bayesian RT, CDSB self-absorption, C-sigma columnar solver, `SpectralResponseCorrection`).
2. **Literature** — the physics deficiency had to correspond to an established LIBS/plasma-spectroscopy result (Aguilera & Aragón, Tognoni/Cristoforetti, Aragón & Aguilera C-sigma, Gigosos Stark, Lawler-group lifetime/branching-fraction transition probabilities, Ciucci/Tognoni CF-LIBS foundations).
3. **Formal spec** — `cflibs-formal` was used as a *soundness envelope*: which theorem's hypothesis the shipped code violates on real data (e.g. `ForwardMap.lean` hardcodes a *scalar* F_cal, so a wavelength-dependent E(λ) sits outside every closure theorem; `saha_joint_identifiability` requires an observed ion stage, which SCCT lacks; `CompositionIdentifiability.lean` assumes identical/correct A_ki, vacuous under D/E-grade bias).

**Adversarial verification.** Each domain's raw findings were re-checked against the code and against the program's own *measured* real-data symptoms. Findings were graded CONFIRMED / WEAKENED / UNVERIFIED, and the magnitude claims were tested against the documented facts that (i) the real-data floor is atomic-data-limited (~0.171 loss, Kurucz beat NIST on SuperCam) and (ii) n_e imputation moves DED composition only ~1-2% per decade by CF-LIBS cancellation. This deliberately downgraded several dramatic-sounding claims to "directionally real, magnitude overstated." The measured symptoms were treated as phenomena to *explain*, never as conclusions to accept.

**Deduplication.** The 21 raw findings collapse into eight root-cause clusters. Where the same physics surfaced in multiple domains (e.g. the Saha stage-III inverse/forward asymmetry appeared in ionization, plasma-state, and closure), the strongest-verified instance anchors the cluster and the others are cross-referenced.

---

## Root-cause clusters (dedup map)

| Cluster | Root cause | Raw findings merged | Strongest verdict |
|---|---|---|---|
| **A. Atomic-data scale** | Absolute + relative g·A error is a *systematic, source-correlated bias*, not random variance; partition completeness secondary | atomic #1 (absolute, lifetime-anchor), atomic #2 (relative, common-upper-level), atomic #3 (partition completeness) | CONFIRMED (#1) |
| **B. n_e is imputed, not measured** | On real data n_e is Earth-STP pressure-balance imputed (no Stark-B, no ion lines); several real diagnostics exist unused | ionization #1 (SB inter-stage offset), ionization #3 (Balmer Stark), forward #3 (continuum), closure #4 (multiplier rides imputed n_e), lineshape #2 (per-channel LSF) | CONFIRMED (#1) |
| **C. Saha ladder forward/inverse asymmetry** | Inverse abundance multiplier & charge balance truncate at stage II; forward uses stage III | ionization #2, plasma-state #2, closure #4 (partial) | CONFIRMED (ion #2) |
| **D. Thin fit forward vs thick data** | Every *inference* forward is optically thin with no fitted optical depth; the accurate iterative path corrects observably but the full-spectrum/joint/Bayesian path does not | radiative #1-#4, forward #1, plasma-state #3 (two-zone) | WEAKENED |
| **E. Relative closure vs ratio/columnar reporting** | Default estimator normalizes Σ C=1 over detected set; ratios (the DED deliverable) already cancel the denominator but are not reported | closure #1, closure #2 (completeness), plasma-state #1 (gate integral, partly) | WEAKENED |
| **F. Forward completeness (geology)** | Molecular bands and physical continuum entirely absent from forward | forward #2 (molecular), forward #3 (continuum) | CONFIRMED (#2) |
| **G. Oxide redox / instrument response (geology & in-house)** | Fixed oxidation state; wavelength-dependent E(λ) off / unrecoverable on in-house data | closure #3 (oxide), lineshape #1 (E(λ)) | WEAKENED |
| **H. Temporal / spatial integration** | Single-snapshot single-zone fit of a gate-integrated, LOS-integrated cooling plasma | plasma-state #1, plasma-state #3 | WEAKENED |

---

## Ranking

### For goal (a) — DED Ti-6Al-4V drift tracking (ratios/precision)

1. **Report ratios / columnar densities, not closure-normalized wt%** (Cluster E, closure #1) — near-zero effort, directly serves the ratio deliverable, removes the mass-slosh that the OPC band-aid was compensating for.
2. **Atomic-data scale — relative g·A self-consistency + absolute anchoring** (Cluster A) — the dominant real-data floor; tightens T (hence Ti/Al ratio) and the whole fit.
3. **Thick-line handling on the fit path** (Cluster D) — Ti/Al majors saturate; wiring the *existing observable-anchored* corrector into the full-spectrum/joint path stops the T-box-edge runaway and Ti/Al ratio distortion.
4. **Saha stage-III inverse/forward symmetry** (Cluster C) — cheap correctness fix; matters mainly for hot-core (≥1.2 eV) spectra.
5. **Measure n_e instead of imputing it** (Cluster B) — *second-order for DED composition by cancellation*, but first-order for the trust/LTE surface and for not silently reporting on an unconstrained parameter.

### For goal (b) — absolute composition generally

1. **Atomic-data scale** (Cluster A) — dominant.
2. **Thick-line handling on the fit path** (Cluster D) — matrix elements (steel Fe, geology majors).
3. **Undetected-mass completeness** (Cluster E, closure #2) — 1/(1−m) inflation of every metal by unmodeled C/N/O/H.
4. **Measure n_e** (Cluster B) — ion-observed minors (steel Cr), LTE trust.
5. **Geology forward completeness** (Cluster F + G-oxide + H two-zone) — molecular bands, physical oxygen balance, two-zone LOS; this is the bulk of the ~8-10 wt% geology floor together with atomic data.
6. **Saha stage-III symmetry** (Cluster C).

---

## Top issues (detail)

### Issue 1 — Atomic-data g·A error is a systematic bias, not random variance (Cluster A)

**Physics deficiency (plainly).** CF-LIBS composition is `C_s ∝ U_s(T)·exp(intercept_s)`. The intercept of each species' Boltzmann plot carries the *absolute scale* of that species' g·A values. A NIST D/E-grade transition probability is a fixed 50-100% *bias* in an unknown direction, correlated within a multiplet and within a source paper — not a zero-mean independent Gaussian. The shipped code models it as independent random variance and *down-weights* poorly-graded lines (WLS inverse-variance). You cannot average or weight your way out of a scale error: down-weighting a biased line lowers its leverage but leaves the surviving (also-biased) ensemble mean shifted. Worse, where a numeric grade is missing the A_ki uncertainty is *fabricated from NIST relative intensity*, an emission/line-strength proxy uncorrelated with transition-probability accuracy.

**Evidence chain.**
- Code: `boltzmann.py:1176-1211` (`_build_sigma_y` adds σ(A_ki) in quadrature to σ_y), `iterative.py:1153-1160` (same into WLS weights), default-on via `aki_uncertainty_weighting=True` (`pipeline.py:277`); fabricated rel_int→grade at `database.py:343-372`; composition map `iterative.py:618`.
- Symptom: real-data floor is atomic-data-limited (~0.171 loss on real vs ~0 solver floor on synthetic); **Kurucz beat NIST on SuperCam** — an internally-consistent single-source g·A set has *one coherent scale error* (partly absorbable by closure) versus NIST's *many independent* per-source offsets (not absorbable). OPC "averages to ~1 on Fe-Co full-range binary" because a per-element constant cannot represent a per-species-and-per-line scale error.
- Literature: Lawler/Den Hartog Wisconsin-group practice — absolute A_ki = branching fraction / radiative lifetime, because relative branching fractions and laser-measured lifetimes are accurate to a few percent while the absolute scale is not (Sm 2010, La II 2025, Er II 2008); NIST accuracy grades (Kramida/ASD).
- Formal: `CompositionIdentifiability.lean:132` assumes `hAeq` (identical/correct A), so the estimator's soundness proof is *vacuous* under real A_ki bias; `SOLVER_FORMALIZATION_GAPS.md` gap #2 (atomic-data perturbation channel missing).

**Concrete physics change.**
- Replace the diagonal-variance A_ki *weighting* with absolute-scale *anchoring*: for each species, renormalize the adopted A_ki set so the summed decay rate from each upper level matches an independently-measured radiative lifetime (`A_ki = BR_ki/τ_k`), ingesting lifetimes/branching fractions where available (Lawler/Den Hartog Fe-group, VALD/DREAM, Kurucz). The per-species intercept then carries a lifetime-limited (~few %) scale error instead of a D/E-grade (50-100%) one, and the residual error becomes a single coherent per-species offset that closure/OPC *can* absorb.
- Add an **in-plasma relative-g·A self-calibration** (atomic #2): group observed lines by shared upper level (already in the DB `energy_levels`), use the exact T-, n_e-, population-independent identity `I_i/I_j = (g_iA_iλ_j)/(g_jA_jλ_i)` to measure the relative-A_ki residual per group, and refine each species' relative g·A to minimize cross-line (and cross-shot) residual *before* the slope fit. This physically reproduces "why Kurucz beat NIST" from your own plasma without an external standard. Replace the fabricated rel_int→grade uncertainty with this physically-derived per-line relative-A_ki error.
- Partition completeness (atomic #3) is a *minor* companion for the DED/steel transition metals (Fe I/II unobserved-level effect <~1%); the code already has the `BarklemCollet2016` authoritative-source hook (`partition.py:456,534-538`) — wire a complete-level tabulation for validation but do not expect >1 wt% from it on Ti/Cr/V/Fe (the ≥10-20% deficits are alkalis/alkaline-earths, absent from these targets).

**Falsifiable first experiment.** On steel_266 and SuperCam SCCT: (1) compute per-species Boltzmann-plot RMS residual with NIST g·A vs Kurucz g·A — predict Kurucz's within-species scatter is materially lower and tracks the 0.171 loss reduction; (2) re-solve after renormalizing each species' A_ki to lifetime sum-rules and measure held-out RMSEP vs the current weighted fit. Confirmed if lifetime-anchored g·A recovers most of the Kurucz-over-NIST gain without adopting Kurucz wholesale.

**Effort / dependency.** High effort (data ingest + a self-calibration pass), but **highest-value** for both goals and *independent* of the other clusters. Do first. The self-calibration pass (atomic #2) is a smaller sub-step that can land before the full lifetime ingest.

---

### Issue 2 — Report Aitchison ratios / columnar densities, not closure-normalized wt% (Cluster E, DED-critical)

**Physics deficiency.** The default estimator normalizes `Σ C_s = 1` over the *detected* set (`closure_mode="standard"`, `apply_standard` at `closure.py:757-813`). Every C_s shares the denominator `Σ_t rel_t`, so any per-element intensity/atomic-data/self-absorption error in one element moves *all* fractions (formally, `CompositionRobustness.lean:147` bounds the whole-vector error by `2·card·δ/Ŝ`). But the DED deliverable is a *ratio* — and the ratio `C_V/C_Ti = rel_V/rel_Ti` **already cancels the shared denominator**. The program has been paying for absolute-fraction slosh (V/Ti limiter 15.2%→3.6% via OPC band-aiding) to recover something the ratio never needed.

**Evidence chain.**
- Code: default `closure_mode="standard"` (`iterative.py:1675`); columnar-density solver `solve_csigma_composition` (`csigma.py:509`) is implemented but never imported by the default pipeline (`csigma.py:4-8`).
- Symptom: "down-weighting dense Ti/Al lines shifted mass to sparse V (closure is relative)"; the DED V/Ti limiter improved only via OPC, a per-matrix calibration band-aid.
- Literature: Aitchison (1986) subcompositional coherence — ratios are matrix- and detected-set-invariant; Aragón & Aguilera C-sigma (2014) solve columnar density N_s directly.
- Formal: `MatrixEffects.lean:110` `recoveredComposition_ratio_matrix_invariant` — closure-normalized *ratios* are already provably matrix-invariant; `SOLVER_FORMALIZATION_GAPS.md` Tier-3 #13 recommends "prefer ratios/deltas."

**Corrected scope (important).** The adversarial pass refuted the raw finding's stronger claim. Swapping in the C-sigma columnar estimator is **not** justified by the denominator argument: N_Ti is still fit from Ti's lines, so perturbing Ti lines moves the V/Ti ratio in *both* methods. C-sigma's real edge is optical-thickness handling (curve-of-growth), which belongs to Cluster D. The cheap, correct fix here is to **report ratios from the existing closure output** — they already cancel the denominator and are provably matrix-invariant.

**Concrete physics change.**
- Make the DED/tracking deliverable the Aitchison log-ratio `ln(N_V/N_Ti)` (and `ln(N_Al/N_Ti)`) computed directly from the closure relatives, *not* the closure-normalized wt%. Treat Σ C=1 as an optional final projection, never the estimator, for the tracking goal.
- Keep the C-sigma columnar solver on the roadmap but attribute its benefit to self-absorption (Cluster D), not denominator avoidance.

**Falsifiable first experiment.** On Ti-6Al-4V (real SCCT + synthetic): perturb/down-weight the Ti (or Fe) line set and measure the change in recovered V. Predict the *absolute* V fraction moves substantially (slosh) while the reported `ln(N_V/N_Ti)` is stable — and that the ratio is already as stable in closure output as in a C-sigma solve (falsifying the "need C-sigma for invariant ratios" claim). Compare V/Ti-ratio scatter across sols with wt% vs log-ratio reporting.

**Effort / dependency.** **Lowest effort, top DED impact.** Pure output/reporting change plus a small benchmark. Do immediately; no dependency on other clusters.

---

### Issue 3 — Optically-thin fit forward vs optically-thick real data (Cluster D)

**Physics deficiency.** For n_e ~ 1e17 cm⁻³ the strongest resonance lines of the matrix element are optically thick: peak intensity saturates as `B·(1−e^{−τ})` (curve-of-growth √τ branch) while a thin model predicts linear growth in n_k·L. Every *inference* forward is pure optically thin with no fitted optical-depth degree of freedom: `full_spectrum.py:432-440` (`apply_self_absorption=False`, `path_length_m=0.0`), `bayesian/forward.py:152-153`, `manifold/batch_forward.py:566`, `iterative.py` default False. Only the standalone `SpectrumModel` (`spectrum_model.py:246`) enables the RT term. With no way to suppress over-bright resonance lines, the prior-free full-spectrum fit rides T to a box edge to fake the missing saturation.

**Evidence chain.**
- Code: the four thin-forward defaults above; the RT escape-factor kernel exists (`kernels.py:671`) but is off in every fit forward.
- Symptom: "prior-free full-spectrum fit rides T to a box edge (T↔composition degeneracy)"; "Fe self-absorption collapsed steel accuracy"; "closure trade — errors slosh to sparse V" (a thin model over-reads the saturated Ti/Al majors' area deficit and closure pushes the missing mass onto sparse V).
- Literature: Gornushkin 1999 (COG), Bulajic 2002 (SA correction), El Sherbini 2005, Aragón & Aguilera 2008 (C-sigma thick branch), Hermann 2017 (full-synthetic continuum+SA).
- Formal: `SelfAbsorptionInverse.lean` — per-line density is *lost* under thick observation; `SOLVER_FORMALIZATION_GAPS.md` Tier-2 #9/#10 — only the thin/saturation-onset branch is proven, the thick √τ branch is unformalized.

**Corrected scope (critical — avoids the rejected approach).** The raw finding's prescribed fix — compute per-line τ from the *composition-derived* lower-level column density — is precisely the approach the project already audited (finding F4) as a **positive-feedback loop that made real ChemCam BHVO-2 worse**, and deliberately replaced with an observable-anchored data-side correction. A free global N·L scalar is largely degenerate with the intensity normalization already fit. So:
- The accurate **iterative** path *already* applies an observable-gated COG correction (`self_absorption_observable.py`: doublet ratio + Planck-ceiling curve-of-growth + suspect down-weighting, wired at `iterative.py:1718-1719`) — literature-grounded (Bulajic 2002, Völker & Gornushkin 2023, El Sherbini 2005) and applied to *observed* intensities, not composition-fed.
- The genuine gap is that the **full-spectrum/joint/Bayesian** raw-spectrum path (which fits the whole spectrum, not a line list) does not use this observable corrector.

**Concrete physics change.**
- Wire the *existing* observable-anchored COG/self-absorption corrector into the full-spectrum/joint/Bayesian objective (pre-correct thick observed lines to their thin-equivalent intensities, or exclude flagged-thick lines), rather than adding a new composition-derived τ DOF.
- For genuinely self-reversed resonance lines (hot core seen through a cool shell — a single isothermal slab `I=B(1−e^{−τ})` is monotonic and *cannot* dip below the wings), route through the **already-exported** two-zone Bayesian RT forward (`TwoZoneBayesianForwardModel`, `forward.py:375-390`, with a fittable `optical_depth_scale`/`shell_fraction`) instead of the single slab. This also covers plasma-state #3 (LOS core+periphery) and the self-reversal mechanism (`SelfReversal.lean`).
- Companion corrections when the corrector *is* used: (i) the absorber column must use the *heavy-particle* density, not the `N_total=n_e` proxy (radiative #2: ionization fraction ~1-10%, so N_total ≈ 10-100× n_e; compute from composition × P/kT or the Saha-closed neutral+ion sum); (ii) the escape factor must match the actual Voigt/Stark line shape (radiative #4: replace the Doppler-only `doppler_cog_escape_factor` gated at τ≤3 with a Voigt-COG escape factor covering the √τ damping asymptote). These two are only worth doing *after* the observable corrector is on the fit path.

**Falsifiable first experiment.** On steel_266 / SuperCam Ti-6Al-4V, run the joint/full-spectrum solver twice: current thin objective vs the observable-corrector-enabled objective. Predict the corrected fit reduces the V over-estimate and the Ti/Al ratio error, and stops T pinning at the box edge (inspect `fit_temperature_K` vs bounds and the T-at-bound rate on held-out). Cross-check on a real steel_245nm Fe resonance line that shows a flat/reversed core.

**Effort / dependency.** Medium. **Do the wiring of the existing corrector first** (low risk, reuses audited code); defer the heavy-particle column (radiative #2) and Voigt-COG escape (radiative #4) as follow-ups. Two-zone routing is a separate, medium-effort path already exported. Explicitly *do not* implement composition-fed per-line τ.

---

### Issue 4 — n_e is imputed from an Earth-STP equation of state, never measured (Cluster B)

**Physics deficiency.** On real data n_e is not measured. The primary path needs a DB line flagged `stark_b` (coverage ~0.85% of real lines; ~0.2-1.8% for Ti/Al/V), which yields nothing for the DED/SuperCam target; the fallback is an isobaric pressure balance at hardcoded Earth STP (P=101325 Pa, `iterative.py:1788-1816`). This imputes n_e ~1.4-2.7e17 cm⁻³ versus realistic late-gate LIBS 1-3e16 — roughly a decade high — and every ion line is then remapped to the neutral plane with that wrong n_e (`_apply_saha_correction`, `iterative.py:1391-1394`). Meanwhile the multi-element Saha-Boltzmann plot *already stacks* neutral+ion lines on one plane (`_fit_saha_boltzmann_graph`, `iterative.py:1531-1610`) but consumes n_e as an *input* and discards the inter-stage ordinate offset that measures it — the offset residual is thrown away.

**Evidence chain.**
- Code: pressure-balance fallback `iterative.py:1788-1816`; SB graph derives the shift from *input* n_e (`iterative.py:1585-1589`) and returns only slope + per-element intercepts; the only n_e diagnostics in the codebase are Stark-width based (`radiation/stark.py`, `physics/stark_ne.py`) — no SB-offset or two-line inversion anywhere.
- Symptom: SCCT "ZERO ion-stage lines → n_e never measured, silently imputed"; held-out 0.65 wt% was achieved *with* the imputed n_e. Steel Cr under-estimate where Cr is seen via Cr II lines.
- Literature: Aguilera & Aragón 2007, Yalcin/Crosley/Smith/Faris 1999 — n_e from the SB inter-stage offset once T is fixed by the shared −1/kT slope (standard, well-posed for any element seen in two adjacent stages, e.g. Fe I + Fe II in steel); Gigosos 2003/1996 — Balmer Stark width ~n_e^0.68, nearly T- and atomic-data-independent; Tognoni 2010 — Stark/Saha n_e is an input, pressure balance is *not* a LIBS n_e diagnostic.
- Formal: `SahaInverse.lean` (`sahaBoltzmann_shift_eq_log_saha`, `saha_joint_identifiability`) — n_e is recoverable from a neutral pair plus an ion line, "never taken as input"; `SOLVER_FORMALIZATION_GAPS.md` gaps #3, #25.

**Corrected scope.** For the *constrained, all-low-IP DED target*, n_e is **second-order for composition**: it enters the ion→neutral correction as ×n_e and the abundance multiplier as ×1/n_e, which cancel for strongly-ionized low-IP metals (total abundance ∝ n_e⁰). A full-decade n_e sweep moved Ti-6Al-4V by only ~1-2% absolute. So the n_e work matters for (i) the **trust/LTE surface** (McWhirter, `overall_reliable` currently evaluated on the imputed n_e — a spectrum can pass on a fabricated parameter), and (ii) **ion-observed minors** (steel Cr), *not* for the DED composition magnitude. Even for Cr, the specific ">1 wt%" prediction is the least certain part: an element seen only via ion lines and mostly ionized (Cr I IP=6.77 eV) has C_s nearly n_e-independent as S₁≫1; the stronger bias falls on multi-stage majors (Fe) whose (1+S) multipliers shift with a decade-wrong n_e.

**Concrete physics change.**
- Invert the SB inter-stage offset for n_e whenever any element has both a neutral and an ion line (the general multi-element case): choose n_e that minimizes the neutral/ion intercept split on the shared-slope plane (Aguilera-Aragón) — wire the offset already present in `_fit_saha_boltzmann_graph` into an n_e *estimate* instead of importing one.
- Add a dedicated **Balmer (Hα/Hβ) Stark n_e** path using the Gigosos-Cardeñoso n_e^0.68 width relation (a proper w(n_e), not the shared linear column; Hβ 486.1 nm preferred for weaker self-absorption); H is present from hydrated minerals, organics, and shield-gas/ambient. This is the one diagnostic needing neither ion lines nor tabulated Stark-B widths.
- Only when *no* ion line exists anywhere and no H line is present, fall to a real diagnostic (continuum level, Cluster F #3 — but that needs absolute radiometric calibration the pipeline lacks). **Retire Earth-STP pressure balance as anything but a last-resort, and never let `overall_reliable` pass on an imputed n_e.**

**Falsifiable first experiment.** On steel_245nm: run the iterative solve with (a) current pressure-balance n_e vs (b) n_e from the SB inter-stage offset; compare recovered Fe/Cr against certified values and the neutral-vs-ion intercept residual. Predict (b) reduces the intercept split and shifts multi-stage majors; on SCCT (zero ion lines) both give identical composition, confirming the offset path is inactive (not harmful) there. Separately, on any Mars/BHVO-2/DED spectrum with a 656.3/486.1 nm line, compare Gigosos-Balmer n_e to the pressure-balance value (predict ~decade disagreement) and check whether it flips the McWhirter verdict.

**Effort / dependency.** Medium. SB-offset inversion reuses existing machinery (low-medium). Balmer path is a new estimator (medium). Independent of Clusters A/D. Priority is higher for goal (b) and for the trust surface than for DED composition.

---

### Issue 5 — Saha ladder forward/inverse asymmetry (stage III truncated in the inverse) (Cluster C)

**Physics deficiency.** The forward populates three stages (`denom = 1 + S1 + S1·S2` = n_I+n_II+n_III, `saha_boltzmann.py:249-256`; `bayesian/forward.py:488-494` citing John et al. 2023 on >1% doubly-ionized fraction at core T). The inverse abundance multiplier is `1 + max(S,0)` = 1 + n_II/n_I only (`iterative.py:1232`), with charge `eps_s = S/(1+S)`, Z≤1 (`iterative.py:1809`) — stage III IP/U is never fetched. This is a genuine forward/inverse asymmetry that breaks round-trip self-consistency and *explains the perverse Cr result*: correcting Cr III partitions feeds the forward/reliability path while the inverse multiplier still ignores stage III, so improving one side of an asymmetric ladder increases the mismatch.

**Evidence chain.**
- Code: forward `saha_boltzmann.py:249-256`; inverse `iterative.py:1232, 1809`.
- Symptom: "Cr systematically under-estimated on steel; CORRECT Cr III partition functions made it WORSE."
- Literature: Griem, Plasma Spectroscopy (full Saha chain); John et al. 2023 (three-stage balance).
- Formal: `MultiSpecies` gap #7 (per-species U/ladder generalization); `TemporalEvolution.lean:73,149-154` scopes the recovery to the two-stage envelope, so the code steps outside the verified region for high-χ elements.

**Corrected scope (magnitude is strongly T-dependent).** Numerically verified: at hot-core T=1.2 eV, n_e=2e16, Cr's n_III/n_tot ≈ 0.30, so the inverse under-counts Cr by ~43% — large. But at realistic late-gate steel T (0.8-1.0 eV) the stage-III fraction for Cr is only 0-4%, so the effect is a few percent at most there. The asymmetry is real and unambiguous *regardless* of magnitude; it dominates only for hot-core / high-Z spectra. Note also: this is a *mass-in-the-sum* issue (plasma-state #2), distinct from partition values — better III partitions merely re-weight I↔II without recovering the missing III population.

**Concrete physics change.** Extend the inverse density completion from `n_I·(1+S/n_e)` to the full Saha chain `Σ_z n_z = n_I·Σ_z Π_{k≤z}(S_k/n_e)` including stage III whenever the equilibrium fraction f_III exceeds numerical negligibility at the fitted (T,n_e); extend `_compute_abundance_multipliers` to `1 + S1 + S1·S2` and the charge balance to `avgZ = (S1 + 2·S1·S2)/(1+S1+S1·S2)`, fetching IP_III/U_III exactly as the forward already does. Make the inverse ladder *identical* to the forward.

**Falsifiable first experiment.** Round-trip: synthesize a Ti-6Al-4V + Cr spectrum with the forward 3-stage model at T=1.2 eV, n_e=2e16, then invert. Predict the current 2-stage inverse under-recovers Cr/Ti by the neglected S1·S2 fraction and the 3-stage inverse closes the round-trip to <0.3 wt%; confirm low-f_III elements (Cu, Ni) are unchanged. Then re-run real steel at late-gate T and confirm the effect is small there (a few percent), consistent with the T-dependence.

**Effort / dependency.** **Low effort** (mirror existing forward code into the inverse), high correctness value. Do early. No dependency, but interacts with Cluster B (the multiplier rides n_e) — fixing n_e first makes the stage-III fix meaningful rather than riding a wrong n_e.

---

### Issue 6 — Undetected-mass completeness: 1/(1−m) inflation of every metal (Cluster E, closure #2)

**Physics deficiency.** `apply_standard` sums only over *detected* elements with a partition function. On steel the candidate set is metals, so carbon, nitrogen, oxygen, hydrogen (interstitials, surface oxide, entrained air) contribute zero to the denominator and every recovered metal fraction is scaled up by 1/(1−m), with m the undetected mass fraction (`MatrixEffects.lean:157,172,182`: `Ĉ_s = C_s/(1−m)`, unbounded). The CF-LIBS closure sum is over *all* species present (Ciucci et al. 1999), not the detected subset. The missing-mass Dirichlet mode exists (`closure.py:1110`) but is opt-in and needs an absolute F that uncalibrated real spectra lack.

**Evidence chain.** Code `closure.py:757-813` (no undetected term); symptom "un-modeled C/N/O/H inflate everything on steel," per-element Fe=16.5, Mo=11.6 RMSEP residual a relative-only closure cannot remove; literature Ciucci 1999 / Tognoni 2010; formal `MatrixEffects.lean:131-196`, gap #13.

**Concrete physics change.** Add an explicit unmeasured-light-element mass channel from a *measured* proxy: include C (193.1 nm) / O (777 nm) / H (656 nm) columnar densities in the total when in-window, or bound m from an independent stoichiometric prior. When m cannot be measured, report metal fractions as *ratios / upper bounds* (physics-honest per the inflation theorem), not absolute wt% — which dovetails with Issue 2.

**Falsifiable first experiment.** On steel_266: drop a real minor element from the candidate set and verify every remaining metal rescales by the predicted 1/(1−m) while pairwise ratios stay fixed (direct test of the inflation theorem). Then include measured C/O line densities and check held-out RMSEP for light-element-bearing samples drops toward the ratio-only floor.

**Effort / dependency.** Medium. Mainly a goal-(b) lever (steel/absolute). For DED, subsumed by "report ratios" (Issue 2). Depends on line-detection covering the light-element windows.

---

### Issue 7 — Geology forward completeness: molecular bands, oxide redox, two-zone LOS (Clusters F + G-oxide + H)

**Physics deficiency (three coupled gaps, geology-scoped).**
- **Molecular bands entirely absent** (forward #2, CONFIRMED). No molecular code anywhere; diatomic band systems (AlO B-X ~464-520 nm, CN violet ~388 nm, C2 Swan ~516 nm, TiO γ, CaO) dominate wide windows in O/N/C-bearing and oxide-plume plasmas and overlap the 380-520 nm region where major Al/Ca/Ti lines sit. SNIP/ALS baseline (`preprocessing.py`) cannot separate ro-vibrationally structured bands from atomic lines, so band flux biases every line area feeding the Boltzmann plot.
- **Oxide closure assumes a fixed oxidation state** (closure #3). `apply_oxide_mode` (`closure.py:883-949`) weights each cation by a compile-time O-per-cation constant (Fe=1.5 → Fe2O3). But Fe2O3 is a *reporting* convention (Jochum 2016); basaltic Fe is dominantly Fe2+ (FeO). The inferred oxygen mass — the *dominant* element, never measured — is wrong, biasing the whole oxide sum.
- **Homogeneous single-zone fit of a LOS-integrated core+periphery plasma** (plasma-state #3). A real laser plasma has strong radial T/n_e gradients; the cool periphery produces self-reversal of strong resonance lines. The capability *exists and is exported* (two-zone Bayesian RT, Issue 3) but the default iterative path is single-zone.

**Evidence chain.** Code: grep confirms no molecular emission (`radiation/__init__.py:7` advertises it, unimplemented); `closure.py:62-73` hard-coded oxide factors; `state.py:223` single zone. Symptom: BHVO-2 8.11 wt%, ChemCam preflight 10.23 wt% — geology remains bad. Literature: Bai & Motto-Ros 2014 (AlO T-window), Hermann 2017 (full model incl. bands), Dungan et al. 2022 (excess oxygen with ferric iron), Clegg et al. 2017 (ChemCam abandons stoichiometric closure for multivariate calibration), Aguilera & Aragón 2004 (two-region). Formal: molecular/oxide sequestration explicitly scoped OUT (`MatrixEffects.lean:48-49`); `SpatialForward.lean` (radial profile identifiable only from chord-resolved intensities — single-shot LIBS supplies no chords).

**Corrected scope.** This cluster is the **geology floor**, competing with (and coupled to) the atomic-data floor (Cluster A). It is **out of scope for the DED Ti/Al/V mission** (oxides/geology explicitly excluded). The magnitude attribution ("bulk of the 8-10 wt%") is asserted, not demonstrated, and the project's own record attributes the real floor primarily to atomic data. So: real, several-wt%-plausible on geology, but do not treat as the primary program lever.

**Concrete physics change.** Add LTE molecular-band emission terms (AlO, TiO, CN, C2, CaO) via Hönl-London × Franck-Condon band strengths and a molecular partition function, sharing plasma T but exposing T_mol as a separate parameter (formation is non-LTE in the cooling tail) — greenfield, needs an ADR. Make the oxygen balance physical: treat O/cation as a constrained free parameter for redox-variable cations (Fe, Ti, Mn, V, Cr) or derive it from a measured O 777 nm line; default basaltic Fe to FeO. Route geological/high-optical-depth matrices through the exported two-zone Bayesian forward.

**Falsifiable first experiment.** On BHVO-2 and ChemCam preflight: mask/model the 380-520 nm region and re-invert — predict wt% error drops for Al/Ca/Si; sweep the Fe O/cation factor (1.0↔1.5) and predict a minimum away from the hard-coded 1.5 for reduced standards; re-fit with the two-zone forward and predict it keeps the self-reversed lines currently discarded and lowers geology RMSEP.

**Effort / dependency.** High (molecular is greenfield + ADR). Goal-(b) geology only. Two-zone routing is medium (already exported). Depends on nothing but is lower priority than A/D/B for the stated DED mission.

---

### Issue 8 — Gate time-integration and instrument response (Clusters H, G-response) — real but bounded

**Physics deficiency.**
- **Gate integration** (plasma-state #1): a gated line intensity is `∫ ε(T(t),n_e(t)) dt` over the cooling trail the manifold generator already models (`generator.py:1067-1068`, 5 µs / t0=1 µs), but the inversion fits one static (T, n_e). Because populations are exponential in 1/T, the integral of exponentials ≠ exponential of the mean, so a time-integrated Boltzmann plot is concave and a single slope yields a biased effective T.
- **Wavelength-dependent response E(λ)** (lineshape #1): CF-LIBS cancels only the scalar F via closure; a residual E(λ) rotates the Boltzmann plot. Recoverable from the spectrum via the branching-ratio method (Li/Smith/Omenetto 2014).

**Corrected scope (both substantially mitigated / already handled).**
- Gate integration is **mitigated by standard delayed/short gating** and by the solver's existing R² Boltzmann-plot linearity gate (`iterative.py:889-891`), which *rejects* the concave multi-T signature rather than silently fitting it. The single-gate "fit a cooling quadrature" fix is **underdetermined** (a knob) — avoid it. The clean, sound lever is the **multi-gate** regime (`runtime/multi_gate.py`, shared composition + per-gate T,n_e), which is genuinely unwired; route through it *when multiple gate delays exist*, otherwise leave alone.
- E(λ) is **already implemented and applied** (`response_correction.py` wired at `pipeline.py:1649-1657`) and correctly *default-off* for ChemCam/SuperCam because CL5/CCS products are vendor-radiometrically corrected (re-applying would double-correct). The genuine gap is only for *in-house* spectra lacking a supplied curve: add the branching-ratio *auto-recovery* (jointly infer smooth E(λ) with a seam step + per-line τ from same-upper-level pairs). Per-channel LSF (lineshape #2) is a real but sub-wt% refinement (constant-FWHM-in-nm per grating segment vs one global R; non-Gaussian wings), and the flagship DED alloys are neutral-dominated where it barely bites.

**Falsifiable first experiments.** Gate: synthesize with the manifold's `_time_integrated_spectrum` at known composition, invert with the single-snapshot solver vs an instantaneous snapshot at the intensity-weighted mean (T,n_e); the RMSEP difference isolates the integration bias — predict it is small for delayed/short gates and collapses under multi-gate routing. Response: inject a known E(λ) with a channel step into a clean synthetic spectrum and confirm composition error scales with the step; then branching-ratio-recover E(λ) on an *in-house* dataset and re-benchmark.

**Effort / dependency.** Low-medium and **low priority.** Multi-gate routing is the only clean lever (medium, already implemented). Do not build the single-gate cooling quadrature. E(λ) auto-recovery only matters for in-house instruments.

---

## Reinterpreted program conclusions

1. **The OPC one-point F-factor is compensating for the atomic-data scale error (Issue 1), not fixing physics.** OPC's per-element geometric-mean F corrects a *constant per-element absolute-g·A-scale bias* on matrix-matched standards — which is exactly the coherent-scale-error signature. That is why it wins on dominant-matrix alloys (steel, Ti) but "averages to ~1 on Fe-Co full-range binary": a per-element constant cannot represent an error that is really per-species-and-per-line. **Implication:** OPC is a band-aid over Cluster A; fixing the atomic data (lifetime/branching anchoring + in-plasma relative-gf self-calibration) attacks the root cause and should reduce the program's reliance on matrix-matched OPC calibration.

2. **The DED V/Ti limiter (15.2%→3.6% via OPC) was compensating for relative-closure mass-slosh on *absolute fractions* (Issue 2), a quantity the ratio deliverable never needed.** Reporting `ln(N_V/N_Ti)` from the existing closure output sidesteps the slosh structurally, without per-matrix calibration. **Implication:** the DED goal has a near-free structural win that was previously bought with calibration.

3. **"Correct Cr III partition functions made Cr worse" is now explained** as the forward/inverse Saha-ladder asymmetry (Issue 5) plus the imputed-n_e sensitivity (Issue 4), *not* a partition-value problem. Better III partitions re-weight the forward/reliability path while the inverse multiplier still drops stage-III mass, widening the mismatch. **Implication:** stop treating the Cr regression as evidence that partitions are wrong; it is evidence the inverse ladder must match the forward.

4. **"Kurucz beat NIST on SuperCam" is the direct fingerprint of Cluster A**, not an argument to adopt Kurucz wholesale: an internally-consistent single-source g·A set carries one coherent (absorbable) scale error versus NIST's many independent per-source offsets. The in-plasma relative-gf self-calibration (atomic #2) reproduces this benefit from your own data.

5. **"Enabling forward self-absorption collapsed steel accuracy while selection composed" is not evidence against self-absorption physics** — it is evidence against the *composition-derived-τ* forward approach (the audited F4 positive-feedback loop). The correct lever (Issue 3) is wiring the *observable-anchored* corrector, which already works on the iterative path, into the full-spectrum/joint path.

6. **The held-out 0.65 wt% SCCT result rests on an unconstrained (imputed) n_e (Issue 4).** It is not invalidated (composition is n_e-insensitive for that target by cancellation), but the *reliability flag* passing on a fabricated parameter is a trust-surface defect: `overall_reliable` must not pass on an imputed n_e.

7. **The geology floor (8-10 wt%) is a coupled atomic-data + forward-completeness problem (Clusters A + F/G/H), not a single missing feature.** Two-zone RT and self-absorption capability already exist and are exported; the remaining true gaps are molecular bands and physical oxygen balance. This is out of scope for the DED mission.

---

## Do-NOT-do list (verified-sound dead ends and knob-work to stop)

**Verified physically sound — do not "fix":**
- The single-(T,n_e) Saha-Boltzmann *algebra* and the calibration-free closure (F_cal cancels; two-stage n_e cancellation) — the defect is the model applied to integrated data, not the arithmetic.
- The homogeneous isothermal LTE slab RT *formula* `I=B(1−e^{−κL})` with `κ=Σ ε_l/B` — correct within its single-zone assumption; the issue is it is off in the fit forwards and uses N_total=n_e.
- Voigt profiles (scipy `wofz` numpy path; 36-term Weideman JAX path) — no wing clipping; Doppler width `σ=λ√(kT/mc²)` — correct (spurious factor-2 already removed).
- Stark width convention (FWHM@1e17/10000K single source of truth; Olivero-Longbothum exact deconvolution) and the Stark n_e *estimator design* (pinned-Gaussian Voigt fit, resonance down-ranking, multiplet-blend veto, MAD trim) — sound; the binding constraint is data coverage, not the estimator.
- IPD / partition truncation at IP−Δχ with Debye lowering — physically correct cutoff; only level *completeness below it* is (minorly) at issue.
- `SpectralResponseCorrection` math and its default-off for vendor-corrected ChemCam/SuperCam — correct; re-applying would double-correct.
- CDSB (width-only columnar density) and `select_optically_thin_lines` — legitimate, physics-safe; they genuinely cannot double-count with an intensity correction.
- ILR/PWLR closure modes as *coordinate infrastructure* — but they are mathematically equivalent to standard closure for a single pass, so they are **not** a mass-slosh or completeness fix.
- Log-sum-exp stabilization in closure — offset cancels in normalization; not an error source.
- The McWhirter/Cristoforetti LTE validator framed as necessary-not-sufficient — sound; the issue is upstream (the fit never uses a self-consistent n_e).

**Knob-work / rejected approaches to stop:**
- **Do not** add a composition-derived per-line τ DOF to the fit forward (the audited F4 positive-feedback loop that worsened BHVO-2). Wire the observable-anchored corrector instead.
- **Do not** build the single-gate "2-point cooling quadrature" fit — underdetermined from one integrated spectrum; use multi-gate routing where multiple delays exist, otherwise leave the single-snapshot fit gated by the R² linearity check.
- **Do not** swap in the C-sigma columnar estimator on the default path *for the denominator argument* — ratios already cancel the denominator; C-sigma's real value is self-absorption (Issue 3), not slosh.
- **Do not** spend the accuracy budget chasing n_e *magnitude* for the DED all-low-IP target (composition ∝ n_e⁰ by cancellation); do the n_e work for the trust surface and ion-observed minors.
- **Do not** "add A_ki uncertainty" — it is already tracked and folded into the WLS; the down-weighting *model* is the problem, and a better weight is still a no-op against a scale bias. Anchor the scale instead.
- **Do not** expect partition-completeness corrections to move Ti/Cr/V/Fe by >~1 wt% (the ≥10-20% deficits are alkalis/alkaline-earths, absent from the DED/steel targets).
- **Do not** treat OPC F-factor tuning as accuracy progress — it band-aids Cluster A.

---

## Consolidated action ordering

| Order | Action | Cluster | Goal | Effort | Depends on |
|---|---|---|---|---|---|
| 1 | Report Aitchison log-ratios from closure output | E | (a) | Low | — |
| 2 | In-plasma relative-gf self-calibration (common-upper-level) | A | (a)(b) | Med | — |
| 3 | Wire existing observable SA corrector into full-spectrum/joint path | D | (a)(b) | Med | — |
| 4 | Make inverse Saha ladder = forward 3-stage (multiplier + charge) | C | (a)(b) | Low | (best after 5) |
| 5 | n_e from SB inter-stage offset; retire Earth-STP fallback; gate reliability | B | (b), trust | Med | — |
| 6 | Lifetime/branching-fraction absolute A_ki ingest | A | (a)(b) | High | 2 |
| 7 | Balmer (Hα/Hβ) Stark n_e path | B | (b), trust | Med | 5 |
| 8 | Undetected-mass channel (C/O/H) or ratio/bound reporting | E | (b) | Med | line detection |
| 9 | Heavy-particle column + Voigt-COG escape (after 3) | D | (b) | Med | 3 |
| 10 | Geology: molecular bands (ADR) + physical O balance + two-zone routing | F/G/H | (b) geology | High | — |
| 11 | In-house E(λ) branching-ratio auto-recovery; per-channel LSF | G | in-house | Low-Med | — |

---

## Executive summary (top 5)

1. **Atomic-data g·A is a systematic scale bias, not random variance** — this is the ~0.171 real-data floor ("Kurucz beat NIST"); fix by lifetime/branching-fraction absolute anchoring + in-plasma relative-gf self-calibration. **Impact: dominant for both goals; several wt% geology, 1-3 wt% steel minors.**
2. **Report ratios, not closure-normalized wt%, for DED** — ratios already cancel the shared denominator that causes V/Ti mass-slosh (OPC's 15.2→3.6 gain was band-aiding this). **Impact: top DED lever, near-zero effort.**
3. **Every inference forward is optically thin** — wire the *existing observable-anchored* self-absorption corrector into the full-spectrum/joint path (NOT composition-fed τ); stops T-box-edge and Ti/Al ratio distortion. **Impact: ~1-2 wt% steel Fe, DED Ti/Al ratio.**
4. **n_e is Earth-STP imputed, never measured** — invert the Saha-Boltzmann inter-stage offset (and add Balmer Stark); second-order for DED composition but first-order for the trust/LTE surface and steel Cr. **Impact: trust-surface + ion-observed minors.**
5. **Inverse Saha ladder truncates at stage II while the forward uses stage III** — a round-trip asymmetry that explains the perverse Cr-partition regression; make the inverse match the forward. **Impact: cheap; up to ~40% for hot-core/high-Z, few % at late-gate steel T.**

**Report path:** `/home/brian/code/CF-LIBS-improved/.worktrees/no-fallback/docs/research/physics-first-principles-audit.md`

---

## Addendum (2026-07-02): Issue 1 anchoring arm — FALSIFIED where testable

The Lawler/Den Hartog lifetime-anchoring experiment ran (overlay DB, 2,304 lab-anchored lines,
production DB verifiably untouched). Result: **median ln(A_anchored/A_NIST) ≈ 0 for every
covered species** — NIST ASD already sources the Wisconsin gf-values there, so no coherent
scale correction exists to apply. The σ_rms table is a grade-weighted worst-case *bound*, not
a realized bias. Coverage of the pipeline's actually-selected lines is the binding limiter
(Fe I/Fe II: 0%); where covered, micro-improvements (Cr −0.49, Ni −0.19 RMSEP) but a net
held-out regression (+0.40) via an Fe selection cascade → the overlay stays OPT-IN, off the
default path. Reinterpretation: the ~0.171 real-data atomic-data loss is NOT Lawler-vs-NIST
scale error; the remaining suspects are the specific older-source gf's of the selected
persistent lines (testable per-line vs an independent Kurucz ingest) or a non-A_ki mechanism.
The in-plasma relative-gA self-calibration (Issue 1a) is unaffected — it measures whatever
lines are actually used.
